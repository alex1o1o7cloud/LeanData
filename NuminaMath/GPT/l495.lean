import Mathlib
import Mathlib.Algebra.Field
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group
import Mathlib.Algebra.Group.Defs
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Geometry.Triangle.Basic
import Mathlib.Init.Function.Meta
import Mathlib.Logic.Basic
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.CondCount
import Mathlib.Probability.Notation
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Probability
import Mathlib.Topology.Algebra.Order
import Mathlib.Topology.Euclidean.Basic

namespace greatest_integer_radius_l495_495677

theorem greatest_integer_radius (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
begin
  have h1 : r^2 < 75, from (lt_div_iff (real.pi_pos)).mp (by rwa [mul_div_cancel_left _ (ne_of_gt real.pi_pos)]),
  -- To show the highest integer r satisfies r^2 < 75 is 8:
  have : r ≤ nat.floor (real.sqrt 75), 
  { exact nat.le_floor_of_lt (by exact_mod_cast h1) },
  -- Since nat.floor (real.sqrt 75) == 8
  exact le_trans this (by norm_num)
end

end greatest_integer_radius_l495_495677


namespace geom_series_sum_l495_495850

theorem geom_series_sum : 
  let a₀ := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 5 in
  ∑ i in Finset.range n, a₀ * r ^ i = 341 / 1024 := 
  by
    sorry

end geom_series_sum_l495_495850


namespace price_per_unit_max_profit_l495_495524

-- Part 1: Finding the Prices

theorem price_per_unit (x y : ℕ) 
  (h1 : 2 * x + 3 * y = 690) 
  (h2 : x + 4 * y = 720) : 
  x = 120 ∧ y = 150 :=
by
  sorry

-- Part 2: Maximizing Profit

theorem max_profit (m : ℕ) 
  (h1 : m ≤ 3 * (40 - m)) 
  (h2 : 120 * m + 150 * (40 - m) ≤ 5400) : 
  (m = 20) ∧ (40 - m = 20) :=
by
  sorry

end price_per_unit_max_profit_l495_495524


namespace construct_circle_exists_l495_495235

noncomputable def construct_circle (e : Line) (A : Point) (O : Point) (r : ℝ) : Prop :=
  ∃ C : Point, ∃ new_circle : Circle,
    -- Conditions
    A ∈ e ∧ 
    (new_circle.center = C ∧ 
     new_circle.radius = distance C A ∧ 
     -- Orthogonality conditions
     distance O C ^ 2 = O.radius ^ 2 + new_circle.radius ^ 2)

theorem construct_circle_exists (e : Line) (A : Point) (O : Point) (r : ℝ) [A ∈ e] : Prop :=
  construct_circle e A O r :=
by 
  sorry

end construct_circle_exists_l495_495235


namespace tamara_is_17_over_6_times_taller_than_kim_l495_495061

theorem tamara_is_17_over_6_times_taller_than_kim :
  ∀ (T K : ℕ), T = 68 → T + K = 92 → (T : ℚ) / K = 17 / 6 :=
by
  intros T K hT hSum
  -- proof steps go here, but we use sorry to skip the proof
  sorry

end tamara_is_17_over_6_times_taller_than_kim_l495_495061


namespace rectangular_field_area_l495_495195

theorem rectangular_field_area (w : ℕ) (h : ℕ) (P : ℕ) (A : ℕ) (h_length : h = 3 * w) (h_perimeter : 2 * (w + h) = 72) : A = w * h := 
by
  -- Given conditions
  have h_eq : h = 3 * w := h_length
  rewrite [h_eq] at *
  -- Given perimeter equals 8w = 72
  have w_value : w = 9 := sorry
  -- Length calculated as 3w = 27
  have h_value : h = 27 := sorry
  -- Now we calculate the area
  calc
    A = w * h : by sorry
    ... = 9 * 27 : by sorry
    ... = 243 : by sorry

end rectangular_field_area_l495_495195


namespace sphere_intersection_area_independent_of_position_l495_495292

theorem sphere_intersection_area_independent_of_position
  (O K : Point)
  (r varrho : ℝ)
  (hr_pos : 0 < r)
  (hvarrho_ge_half_r : varrho ≥ r / 2) :
  let common_area := π * r^2 in
  ∀ (K : Point), common_area = π * r^2 :=
by
  -- Proof steps would go here.
  sorry

end sphere_intersection_area_independent_of_position_l495_495292


namespace s_of_4_l495_495405

noncomputable def t (x : ℚ) : ℚ := 5 * x - 14
noncomputable def s (y : ℚ) : ℚ := 
  let x := (y + 14) / 5
  x^2 + 5 * x - 4

theorem s_of_4 : s (4) = 674 / 25 := by
  sorry

end s_of_4_l495_495405


namespace people_not_show_up_l495_495912

theorem people_not_show_up 
  (invited : ℕ) 
  (people_per_table : ℕ) 
  (tables_used : ℕ) 
  (h_invited : invited = 47) 
  (h_people_per_table : people_per_table = 5) 
  (h_tables_used : tables_used = 8) : 
  (invited - tables_used * people_per_table = 7) :=
by
  have h_total_seated : tables_used * people_per_table = 40, by
    rw [h_people_per_table, h_tables_used],
    exact nat.mul_pos (by norm_num) (by norm_num),
  have h_people_not_show_up : invited - 40 = 7, by
    rw [h_invited, h_total_seated],
    norm_num,
  exact h_people_not_show_up

end people_not_show_up_l495_495912


namespace profit_function_l495_495913

open Nat

-- Definitions based on conditions
def initial_price : ℕ := 10
def unit_cost : ℕ := 8
def initial_sales_volume : ℕ := 100

-- Function for daily profit y given x
def daily_profit (x : ℕ) : ℕ := 10 * (x + 2) * (10 - x)

-- The Lean theorem statement for the translated proof problem
theorem profit_function :
  ∀ (x : ℕ), (0 ≤ x ∧ x ≤ 10) → 
  let y := daily_profit x in
  (y = 10 * (x + 2) * (10 - x)) ∧ 
  ((x = 3) → (y = 350)) ∧ 
  ((y = 360) → (x = 4)) :=
by
  intro x
  intro hx
  let y := daily_profit x
  split
  . -- Prove y = 10 * (x + 2) * (10 - x)
    unfold daily_profit
    rfl
  . split
    . -- Prove ((x = 3) → (y = 350))
      intro hx3
      rw [hx3]
      unfold daily_profit
      rfl
    . -- Prove ((y = 360) → (x = 4))
      intro hy360
      unfold daily_profit at hy360
      sorry

end profit_function_l495_495913


namespace product_evaluation_l495_495942

theorem product_evaluation :
  ∏ (n : ℕ) in Finset.range 15, ((n + 1) * (n + 3)) / ((n + 5)^2) = (5 : ℚ) / 456 :=
by
  sorry

end product_evaluation_l495_495942


namespace rectangular_field_area_l495_495191

theorem rectangular_field_area :
  ∃ (w l : ℝ), (l = 3 * w) ∧ (2 * (l + w) = 72) ∧ (l * w = 243) :=
by {
  sorry
}

end rectangular_field_area_l495_495191


namespace train_length_l495_495926

theorem train_length (
  (time : ℝ) (man_speed : ℝ) (train_speed : ℝ) (length : ℝ)
  (time_eq : time = 26.997840172786177)
  (man_speed_eq : man_speed = 3)
  (train_speed_eq : train_speed = 63)
  (length_eq : length = 1349.89)
  (convert : ℝ → ℝ := λ speed_in_kmph, speed_in_kmph * 1000 / 3600) :
  length = convert (train_speed - man_speed) * time :=
begin
  sorry
end

end train_length_l495_495926


namespace num_paths_without_interference_l495_495979

constant num_paths : ℤ

theorem num_paths_without_interference 
    (A : (ℕ × ℕ) := (2, 2))
    (B : (ℕ × ℕ) := (3, 2))
    (C : (ℕ × ℕ) := (4, 2))
    (D : (ℕ × ℕ) := (5, 2))
    (E : (ℕ × ℕ) := (6, 2))
    (F : (ℕ × ℕ) := (6, 3))
    (G : (ℕ × ℕ) := (7, 2))
    (H : (ℕ × ℕ) := (7, 3))
    (total_paths : ℕ := 3003) :
    num_paths = 1364 :=
by
    sorry

end num_paths_without_interference_l495_495979


namespace molecular_properties_correct_l495_495842

def atomic_weight : Type := ℝ

structure Element Composition :=
  (atomic_weight_calcium : atomic_weight)
  (atomic_weight_bromine : atomic_weight)
  (atomic_weight_oxygen : atomic_weight)
  (atomic_weight_hydrogen : atomic_weight)
  (calcium_atoms : ℕ)
  (bromine_atoms : ℕ)
  (oxygen_atoms : ℕ)
  (hydrogen_atoms : ℕ)

def molecule : Composition := {
  atomic_weight_calcium := 40.08,
  atomic_weight_bromine := 79.904,
  atomic_weight_oxygen := 15.999,
  atomic_weight_hydrogen := 1.008,
  calcium_atoms := 1,
  bromine_atoms := 2,
  oxygen_atoms := 3,
  hydrogen_atoms := 2
}

def molecular_weight (composition : Composition) : atomic_weight :=
  composition.calcium_atoms * composition.atomic_weight_calcium +
  composition.bromine_atoms * composition.atomic_weight_bromine +
  composition.oxygen_atoms * composition.atomic_weight_oxygen +
  composition.hydrogen_atoms * composition.atomic_weight_hydrogen

theorem molecular_properties_correct (composition : Composition) :
  molecular_weight composition = 249.901 ∧
  (composition.calcium_atoms * composition.atomic_weight_calcium / molecular_weight composition) * 100 ≈ 16.035 ∧
  (composition.bromine_atoms * composition.atomic_weight_bromine / molecular_weight composition) * 100 ≈ 63.943 ∧
  (composition.oxygen_atoms * composition.atomic_weight_oxygen / molecular_weight composition) * 100 ≈ 19.204 ∧
  (composition.hydrogen_atoms * composition.atomic_weight_hydrogen / molecular_weight composition) * 100 ≈ 0.807 :=
by {
  simp [molecular_weight, molecule],
  sorry
}

end molecular_properties_correct_l495_495842


namespace exists_perfect_square_in_selection_l495_495040

/-- 
  Out of the natural numbers from 1 to 100, 50 different numbers were chosen.
  It turned out that the sum of no two of them equals 100.
  Prove that among the chosen numbers, there will always be a square of some integer.
-/
theorem exists_perfect_square_in_selection :
  ∀ (S : Finset ℕ), (∀ a ∈ S, a ≥ 1 ∧ a ≤ 100) → S.card = 50 →
  (∀ a b ∈ S, a + b ≠ 100) → (∃ x ∈ S, ∃ k : ℕ, x = k^2) :=
by
  sorry

end exists_perfect_square_in_selection_l495_495040


namespace father_gave_l495_495756

-- Definitions for the conditions
def mother_gave : ℕ := 80
def spent_on_candy : ℕ := 50
def uncle_gave : ℕ := 70
def final_amount : ℕ := 140

-- Proof statement
theorem father_gave : ∃ (F : ℕ), mother_gave + F - spent_on_candy + uncle_gave = final_amount :=
by
  exists 40
  sorry

end father_gave_l495_495756


namespace parabola_line_no_intersect_sum_of_roots_l495_495737

theorem parabola_line_no_intersect_sum_of_roots (n : ℝ) :
  let P := (λ x : ℝ, x^2)
  let Q := (10, 6 : ℝ)
  let discriminant := n^2 - 40*n + 24
  ∀ a b : ℝ, (a < n ∧ n < b) ∧ discriminant < 0 → a + b = 40 :=
by
  intros P Q discriminant a b h
  sorry

end parabola_line_no_intersect_sum_of_roots_l495_495737


namespace total_clips_correct_l495_495037

def clips_in_april : ℕ := 48
def clips_in_may : ℕ := clips_in_april / 2
def total_clips : ℕ := clips_in_april + clips_in_may

theorem total_clips_correct : total_clips = 72 := by
  sorry

end total_clips_correct_l495_495037


namespace geometric_sequence_divisible_l495_495407

theorem geometric_sequence_divisible (a1 a2 : ℝ) (h1 : a1 = 5 / 8) (h2 : a2 = 25) :
  ∃ n : ℕ, n = 7 ∧ (40^(n-1) * (5/8)) % 10^7 = 0 :=
by
  sorry

end geometric_sequence_divisible_l495_495407


namespace greatest_integer_radius_l495_495679

theorem greatest_integer_radius (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
begin
  have h1 : r^2 < 75, from (lt_div_iff (real.pi_pos)).mp (by rwa [mul_div_cancel_left _ (ne_of_gt real.pi_pos)]),
  -- To show the highest integer r satisfies r^2 < 75 is 8:
  have : r ≤ nat.floor (real.sqrt 75), 
  { exact nat.le_floor_of_lt (by exact_mod_cast h1) },
  -- Since nat.floor (real.sqrt 75) == 8
  exact le_trans this (by norm_num)
end

end greatest_integer_radius_l495_495679


namespace book_page_count_l495_495584

def total_pages_in_book (pages_three_nights_ago pages_two_nights_ago pages_last_night pages_tonight total_pages : ℕ) : Prop :=
  pages_three_nights_ago = 15 ∧
  pages_two_nights_ago = 2 * pages_three_nights_ago ∧
  pages_last_night = pages_two_nights_ago + 5 ∧
  pages_tonight = 20 ∧
  total_pages = pages_three_nights_ago + pages_two_nights_ago + pages_last_night + pages_tonight

theorem book_page_count : total_pages_in_book 15 30 35 20 100 :=
by {
  sorry
}

end book_page_count_l495_495584


namespace sum_of_palindromic_primes_less_than_100_l495_495779

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m > 1 ∧ m * m ≤ n → n % m ≠ 0

def is_palindromic_prime (n : ℕ) : Prop := 
  is_prime n ∧ 
  n / 10 = (n % 10) * 10 + n / 10 ∧ 
  is_prime (n % 10 * 10 + n / 10)

theorem sum_of_palindromic_primes_less_than_100 : 
  let palindromic_primes := [11, 13, 17, 31, 37, 71, 73, 79, 97] in
  palindromic_primes.sum = 429 := 
  by 
  sorry

end sum_of_palindromic_primes_less_than_100_l495_495779


namespace profit_15000_l495_495895

theorem profit_15000
  (P : ℝ)
  (invest_mary : ℝ := 550)
  (invest_mike : ℝ := 450)
  (total_invest := invest_mary + invest_mike)
  (share_ratio_mary := invest_mary / total_invest)
  (share_ratio_mike := invest_mike / total_invest)
  (effort_share := P / 6)
  (invest_share_mary := share_ratio_mary * (2 * P / 3))
  (invest_share_mike := share_ratio_mike * (2 * P / 3))
  (mary_total := effort_share + invest_share_mary)
  (mike_total := effort_share + invest_share_mike)
  (condition : mary_total - mike_total = 1000) :
  P = 15000 :=  
sorry

end profit_15000_l495_495895


namespace angle_in_fourth_quadrant_l495_495341

variable (α : ℝ)

def is_in_first_quadrant (α : ℝ) : Prop := 0 < α ∧ α < 90

def is_in_fourth_quadrant (θ : ℝ) : Prop := 270 < θ ∧ θ < 360

theorem angle_in_fourth_quadrant (h : is_in_first_quadrant α) : is_in_fourth_quadrant (360 - α) := sorry

end angle_in_fourth_quadrant_l495_495341


namespace sum_of_palindromic_primes_l495_495787

def is_palindromic_prime (n : ℕ) : Prop :=
  n < 100 ∧ 10 ≤ n ∧ n.Prime ∧ let reversed := (n % 10) * 10 + (n / 10) in 
  reversed.Prime

theorem sum_of_palindromic_primes : 
  ∑ p in List.filter is_palindromic_prime (List.range 100), p = 429 :=
by
  sorry

end sum_of_palindromic_primes_l495_495787


namespace proposition_D_true_l495_495630

-- Given definitions for the constants
variables {a b : ℝ}

-- Condition 
axiom h : a > b ∧ b > 0

-- Proof statement for proposition D being true
theorem proposition_D_true : (1 / 2) ^ a < (1 / 2) ^ b :=
by sorry

end proposition_D_true_l495_495630


namespace probability_is_22_over_117_l495_495155
open BigOperators

noncomputable def probability_two_faces_painted : ℝ :=
  let total_cubes := 27
  let painted_two_cubes := 12
  let total_ways_to_choose_two := Nat.choose total_cubes 2
  let ways_to_choose_two_with_two_faces := Nat.choose painted_two_cubes 2
  ways_to_choose_two_with_two_faces / total_ways_to_choose_two

theorem probability_is_22_over_117 : probability_two_faces_painted = 22 / 117 :=
by
  have h1 : Nat.choose 27 2 = 351 := by sorry
  have h2 : Nat.choose 12 2 = 66 := by sorry
  unfold probability_two_faces_painted
  rw [h1, h2]
  norm_num
  sorry

end probability_is_22_over_117_l495_495155


namespace incorrect_parallelism_of_planes_l495_495739

variables (α β : Plane) (m n : Line)
hypotheses (H_non_coincident_planes : α ≠ β)
  (H_non_coincident_lines : m ≠ n)
  (H_m_parallel_α : m ∥ α)
  (H_m_parallel_β : m ∥ β)
  (H_n_subset_α : n ⊆ α)
  (H_n_parallel_β : n ∥ β)
  
theorem incorrect_parallelism_of_planes :
  ¬ (α ∥ β) :=
sorry

end incorrect_parallelism_of_planes_l495_495739


namespace problem_proof_l495_495615

-- Define the sequences a_n and b_n and their properties:
def a_seq (n : ℕ) (hn : n > 0) : ℕ := 2^(n-1)
def b_seq (n : ℕ) : ℕ := n^2

-- Given T_n < nS_n - a for sequences {a_n} and {b_n}, prove a ≤ 0
def sequence_inequality (a : ℝ) : Prop :=
  ∀ (n : ℕ), n > 0 → let T_n := (∑ i in List.range n, (a_seq (i+1) sorry) * Real.sqrt (b_seq (i+1))) in
  let S_n := 2^(n-1) - 1 in
  T_n < n * S_n - a

-- No pairs of positive integers (m, n) exist such that b_1, a_m, and b_n form an arithmetic sequence with n > 1
def no_arithmetic_triplet (m n : ℕ) : Prop :=
  ∀ (m n : ℕ), m > 0 → n > 1 → ¬(1 + n^2 = 2^(m-1))

-- Main theorem proving the above three parts
theorem problem_proof :
  (∀ (n : ℕ), n > 0 → a_seq n sorry = 2^(n-1)) ∧
  (∀ (n : ℕ), n > 0 → b_seq n = n^2) ∧
  (∃ (a : ℝ), sequence_inequality a → a ≤ 0) ∧
  ∀ (m n : ℕ), m > 0 → n > 1 → no_arithmetic_triplet m n := sorry

end problem_proof_l495_495615


namespace find_divisor_l495_495508

-- Define the conditions
def dividend : ℕ := 22
def quotient : ℕ := 7
def remainder : ℕ := 1

-- The divisor is what we need to find
def divisor : ℕ := 3

-- The proof problem: proving that the given conditions imply the divisor is 3
theorem find_divisor :
  ∃ d : ℕ, dividend = d * quotient + remainder ∧ d = divisor :=
by
  use 3
  -- Replace actual proof with sorry for now
  sorry

end find_divisor_l495_495508


namespace trailing_zeros_302_factorial_l495_495817

theorem trailing_zeros_302_factorial : 
  (∑ i in {1, 2, 3, 4}.to_finset, if i = 1 then 302 / 5^i else 302 / 5 ^ (i + 1)) = 74 := 
by sorry

end trailing_zeros_302_factorial_l495_495817


namespace count_S_elements_l495_495984

noncomputable def A (n : ℕ) (h : 3 ≤ n) : Set ℕ := {a | ∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ a = a i}
def S (A : Set ℕ) (n : ℕ) : Set ℕ := {x | ∃ i j : ℕ, 1 ≤ i ∧ i ≤ j ∧ j ≤ n ∧ x = (a i) + (a j)}

theorem count_S_elements (a : ℕ → ℕ) (n : ℕ) (d : ℕ) (h₁ : 3 ≤ n) (h₂ : ∀ k, a (k+1) - a k = d) (h₃ : 0 < d) : 
 ∃ (S : Set ℕ), ∀ A, S = { x : ℕ | ∃ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ n ∧ x = (a i) + (a j) } ∧ (S.card = 2 * n - 3) :=
 sorry

end count_S_elements_l495_495984


namespace right_triangle_acute_angle_30_l495_495538

theorem right_triangle_acute_angle_30 (α β : ℝ) (h1 : α = 60) (h2 : α + β + 90 = 180) : β = 30 :=
by
  sorry

end right_triangle_acute_angle_30_l495_495538


namespace abs_pos_of_ne_zero_l495_495878

theorem abs_pos_of_ne_zero (a : ℤ) (h : a ≠ 0) : |a| > 0 := sorry

end abs_pos_of_ne_zero_l495_495878


namespace temperature_relation_before_boiling_l495_495875

variables {T t : ℝ}

-- Given conditions
def temp_increase_per_minute (t : ℝ) : Prop := ∀ Δt, T (t + Δt) = T t + 7 * Δt
def initial_temp : T 0 = 30

-- Goal: Prove the relationship
theorem temperature_relation_before_boiling :
  ∀ (t : ℝ), t < 10 → T = 7 * t + 30 :=
begin
  intros t ht,
  sorry
end

end temperature_relation_before_boiling_l495_495875


namespace smallest_n_rotation_matrix_l495_495581

theorem smallest_n_rotation_matrix :
  let M := matrix.of_fun (λ i j, 
    if i = 0 ∧ j = 0 then 1 / real.sqrt 2 
    else if i = 0 ∧ j = 1 then -1 / real.sqrt 2 
    else if i = 1 ∧ j = 0 then 1 / real.sqrt 2 
    else 1 / real.sqrt 2) in
  ∀ n: ℕ, (M ^ n = 1) ↔ n = 8 :=
by
  sorry

end smallest_n_rotation_matrix_l495_495581


namespace plane_intersects_ellipsoid_in_ellipse_l495_495436

theorem plane_intersects_ellipsoid_in_ellipse :
  (∀ x y z : ℝ, (z - 2 = 0) → (x^2 / 12 + y^2 / 4 + z^2 / 16 = 1) → 
  (∃ a b : ℝ, (a = 3) ∧ (b = √3) ∧ (x^2 / a^2 + y^2 / b^2 = 1))) :=
by
  sorry -- Proof is omitted

end plane_intersects_ellipsoid_in_ellipse_l495_495436


namespace pyramid_height_l495_495209

-- Define the conditions of the problem
def square_base_perimeter : ℝ := 40
def vertex_to_apex_distance : ℝ := 10
def is_right_pyramid : Prop := true  -- This is just a placeholder to indicate this condition

-- Set up the goal in Lean
theorem pyramid_height (square_base_perimeter = 40) (vertex_to_apex_distance = 10) (is_right_pyramid) : height = 5 * Real.sqrt 2 :=
by
  sorry

end pyramid_height_l495_495209


namespace integral_simplification_l495_495971

theorem integral_simplification (C : ℝ) :
  ∫ (x : ℝ) in (0 : ℝ)..(1 : ℝ), (sin (2 * x)) / (1 + cos (2 * x) + (sin (2 * x))^2) dx =
  (1 / 6) * log (abs ((2 - cos (2 * x)) / (1 + cos (2 * x)))) + C := by sorry

end integral_simplification_l495_495971


namespace monotonic_increasing_interval_l495_495816

theorem monotonic_increasing_interval (k : ℤ) :
  ∀ x : ℝ, (k * π + π / 6 ≤ x ∧ x ≤ k * π + 2 * π / 3) →
    ∀ y : ℝ, y = 2 * sin (-2 * x - π / 6) →
    ∀ z : ℝ, (k * π + π / 6 ≤ z ∧ z ≤ k * π + 2 * π / 3) →
    x ≤ z →
    2 * sin (-2 * x - π / 6) ≤ 2 * sin (-2 * z - π / 6) :=
by
  sorry

end monotonic_increasing_interval_l495_495816


namespace transformation_impossible_l495_495117

def P (W : List ℕ) : ℕ :=
  W.enumFrom 1 |>.foldl (λ acc (i, w) => acc + i * w) 0

theorem transformation_impossible : ¬ ∃ (f : List ℕ → List ℕ), (f = λ W, W ++ W ++ W) ∧ (P [1, 0] ≡ P [0, 1] [MOD 3]) :=
by
  sorry

end transformation_impossible_l495_495117


namespace math_problem_l495_495706

theorem math_problem (x : ℝ) (h : x = 0.18 * 4750) : 1.5 * x = 1282.5 :=
by
  sorry

end math_problem_l495_495706


namespace num_integers_between_l495_495330

-- Define the constants
def a : ℝ := 10
def b₁ : ℝ := 0.5
def b₂ : ℝ := 0.6

-- Define the cubes
def x : ℝ := (a + b₁)^3
def y : ℝ := (a + b₂)^3

-- Define the function to count the integers within the interval
def count_integers_between (x y : ℝ) : ℕ :=
  let min_int := Int.ceil x
  let max_int := Int.floor y
  (max_int - min_int + 1).toNat

-- The statement to prove
theorem num_integers_between : count_integers_between x y = 33 := by
  sorry

end num_integers_between_l495_495330


namespace area_of_rectangle_l495_495183

theorem area_of_rectangle (w l : ℝ) (h₁ : w = l / 3) (h₂ : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end area_of_rectangle_l495_495183


namespace geom_arith_sequences_l495_495309

-- Definitions
def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop := ∀ n, a (n+1) = a n * q
def arith_seq (b : ℕ → ℝ) (d : ℝ) : Prop := ∀ n, b (n+1) = b n + d

-- Conditions
variable (a : ℕ → ℝ)
variable (b : ℕ → ℝ)
variable (q : ℝ) (h_q : q > 1)
variable (a1_eq : a 0 * (1 + q + q^2) = 13)
variable (arith_cond : b 0 = a 0 ∧ b 2 = a 1 + 2 ∧ b 4 = a 2)

-- Required proofs
theorem geom_arith_sequences :
  (geom_seq a q ∧ arith_seq b 2) →
  (∀ n, a n = 3^n-1) ∧ (∀ n, b n = 2*n - 1) ∧
  ∀ n, (S : ℕ → ℝ) (S 2n+1 = (Σ i in finset.range(2n+1), if odd i then a (i/2) else b (i/2)) = ((9^(n+1) - 1)/8 + 2*n^2 + n)) ∧
  (∑ i in finset.range n, ((2 * b i - 4) * a (i + 1) - 1) / ((b (a (i + 1) + 1)) * (b (a (i + 2) + 1))) = -n / (2 * 3^(n + 1) + 1)) :=
sorry

end geom_arith_sequences_l495_495309


namespace eight_S_three_l495_495957

def custom_operation_S (a b : ℤ) : ℤ := 4 * a + 6 * b + 3

theorem eight_S_three : custom_operation_S 8 3 = 53 := by
  sorry

end eight_S_three_l495_495957


namespace tan_inequality_solution_l495_495084

theorem tan_inequality_solution (k : ℤ) :
  ∀ x, x ∈ set.Ioc (k * real.pi - real.pi / 2) (k * real.pi - real.pi / 4) → real.tan x ≤ -1 := sorry

end tan_inequality_solution_l495_495084


namespace conical_paper_hat_lateral_surface_area_l495_495883

theorem conical_paper_hat_lateral_surface_area :
  ∀ (r h : ℝ), r = 10 → h = 50 → (1/2) * (2 * real.pi * r) * h = 500 * real.pi :=
by
  intros r h r_eq h_eq
  rw [r_eq, h_eq]
  norm_num
  rw [mul_assoc, mul_assoc, mul_comm (1 / 2) _, mul_assoc, mul_comm 2 (10 : ℝ), mul_div_cancel', mul_assoc, mul_one]
  exact real.pi_ne_zero

end conical_paper_hat_lateral_surface_area_l495_495883


namespace rectangle_area_ratio_l495_495953

-- Define points in complex plane or as tuples (for 2D geometry)
structure Point where
  x : ℝ
  y : ℝ

-- Rectangle vertices
def A : Point := {x := 0, y := 0}
def B : Point := {x := 1, y := 0}
def C : Point := {x := 1, y := 2}
def D : Point := {x := 0, y := 2}

-- Centroid of triangle BCD
def E : Point := {x := 1.0, y := 1.333}

-- Point F such that DF = 1/4 * DA
def F : Point := {x := 1.5, y := 0}

-- Calculate areas of triangles and quadrilateral
noncomputable def area_triangle (P Q R : Point) : ℝ :=
  0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x))

noncomputable def area_rectangle : ℝ :=
  2.0  -- Area of rectangle ABCD (1 * 2)

noncomputable def problem_statement : Prop :=
  let area_DFE := area_triangle D F E
  let area_ABEF := area_rectangle - area_triangle A B F - area_triangle D A F
  area_DFE / area_ABEF = 1 / 10.5

theorem rectangle_area_ratio :
  problem_statement :=
by
  sorry

end rectangle_area_ratio_l495_495953


namespace lcm_smallest_value_l495_495346

/-- The smallest possible value of lcm(k, l) for positive 5-digit integers k and l such that gcd(k, l) = 5 is 20010000. -/
theorem lcm_smallest_value (k l : ℕ) (h1 : 10000 ≤ k ∧ k < 100000) (h2 : 10000 ≤ l ∧ l < 100000) (h3 : Nat.gcd k l = 5) : Nat.lcm k l = 20010000 :=
sorry

end lcm_smallest_value_l495_495346


namespace toothpicks_at_20th_stage_l495_495473

def toothpicks_in_stage (n : ℕ) : ℕ :=
  4 + 3 * (n - 1)

theorem toothpicks_at_20th_stage : toothpicks_in_stage 20 = 61 :=
by 
  sorry

end toothpicks_at_20th_stage_l495_495473


namespace geometric_series_sum_l495_495860

theorem geometric_series_sum :
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  (a + a * r + a * r^2 + a * r^3 + a * r^4) = 341 / 1024 :=
by
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  -- Proof steps go here
  sorry

end geometric_series_sum_l495_495860


namespace greatest_int_radius_lt_75pi_l495_495697

noncomputable def circle_radius_max (A : ℝ) (π : ℝ) : ℕ :=
  Nat.floor (Real.sqrt (A / π))

theorem greatest_int_radius_lt_75pi :
  circle_radius_max 75 Real.pi = 8 := by
  sorry

end greatest_int_radius_lt_75pi_l495_495697


namespace y_intercept_of_line_l495_495483

def equation (x y : ℝ) : Prop := 3 * x - 5 * y = 10

theorem y_intercept_of_line : equation 0 (-2) :=
by
  sorry

end y_intercept_of_line_l495_495483


namespace harkamal_payment_l495_495659

noncomputable def calculate_total_cost : ℝ :=
  let price_grapes := 8 * 70
  let price_mangoes := 9 * 45
  let price_apples := 5 * 30
  let price_strawberries := 3 * 100
  let price_oranges := 10 * 40
  let price_kiwis := 6 * 60
  let total_grapes_and_apples := price_grapes + price_apples
  let discount_grapes_and_apples := 0.10 * total_grapes_and_apples
  let total_oranges_and_kiwis := price_oranges + price_kiwis
  let discount_oranges_and_kiwis := 0.05 * total_oranges_and_kiwis
  let total_mangoes_and_strawberries := price_mangoes + price_strawberries
  let tax_mangoes_and_strawberries := 0.12 * total_mangoes_and_strawberries
  let total_amount := price_grapes + price_mangoes + price_apples + price_strawberries + price_oranges + price_kiwis
  total_amount - discount_grapes_and_apples - discount_oranges_and_kiwis + tax_mangoes_and_strawberries

theorem harkamal_payment : calculate_total_cost = 2150.6 :=
by
  sorry

end harkamal_payment_l495_495659


namespace log_base2_domain_l495_495813

theorem log_base2_domain :
  (∀ x : ℝ, ∃ y : ℝ, log (2 : ℝ) x = y ↔ x > 0) → (set.Ioi 0 = {x : ℝ | 0 < x}) :=
by
  intro h
  sorry

end log_base2_domain_l495_495813


namespace cost_of_plastering_per_sq_m_l495_495213

-- Define the dimensions and total cost as given conditions
def length := 25
def width := 12
def depth := 6
def total_cost := 223.2

-- Calculate the areas
def long_walls_area := 2 * (length * depth)
def wide_walls_area := 2 * (width * depth)
def bottom_area := length * width

-- Calculate the total plastered area
def total_area := long_walls_area + wide_walls_area + bottom_area

-- Calculate the cost per square meter
def cost_per_sq_m := total_cost / total_area

-- The theorem states that the cost per square meter is 0.3 paise
theorem cost_of_plastering_per_sq_m : cost_per_sq_m = 0.3 := 
  by 
    -- The proof is skipped for now, represented by sorry
    sorry

end cost_of_plastering_per_sq_m_l495_495213


namespace differentiable_limit_l495_495290

open Real

theorem differentiable_limit (f : ℝ → ℝ) (h : differentiable_at ℝ f 1) :
  (tendsto (λ Δx, (f (1 + Δx) - f 1) / (3 * Δx)) (𝓝 0) (𝓝 (1 / 3 * deriv f 1))) :=
begin
  sorry
end

end differentiable_limit_l495_495290


namespace sum_palindromic_primes_lt_100_eq_429_l495_495785

def is_palindromic_prime (n : ℕ) : Prop :=
  (n ≥ 10 ∧ n < 100) ∧
  (Prime n ∧ 
   let d1 := n / 10 in
   let d2 := n % 10 in
   Prime (d2 * 10 + d1))

theorem sum_palindromic_primes_lt_100_eq_429 : 
  (∑ n in Finset.filter is_palindromic_prime (Finset.range 100), n) = 429 :=
by sorry

end sum_palindromic_primes_lt_100_eq_429_l495_495785


namespace simson_lines_concurrent_l495_495607

-- Define the points and properties in terms of Lean expressions
variables {A B C D E F L M N : Type*}

-- Assume A, B, C are points forming triangle ABC
variable [is_triangle A B C]

-- Define points D, E, F as perpendicular foot from A, B, C to sides BC, CA, AB respectively
variables (H₁ : is_foot_perpendicular D A B C)
           (H₂ : is_foot_perpendicular E B C A)
           (H₃ : is_foot_perpendicular F C A B)

-- Define points L, M, N as midpoints of sides BC, CA, AB respectively
variables (H₄ : is_midpoint L B C)
           (H₅ : is_midpoint M C A)
           (H₆ : is_midpoint N A B)

-- Main theorem: Simson lines of points L, M, N w.r.t. triangle DEF and 
-- Simson lines of points D, E, F w.r.t. triangle LMN are concurrent
theorem simson_lines_concurrent :
  simson_line_concurrency L M N D E F :=
sorry

end simson_lines_concurrent_l495_495607


namespace juanita_sunscreen_cost_l495_495001

theorem juanita_sunscreen_cost:
  let bottles_per_month := 1
  let months_in_year := 12
  let cost_per_bottle := 30.0
  let discount_rate := 0.30
  let total_bottles := bottles_per_month * months_in_year
  let total_cost_before_discount := total_bottles * cost_per_bottle
  let discount_amount := discount_rate * total_cost_before_discount
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  total_cost_after_discount = 252.00 := 
by
  sorry

end juanita_sunscreen_cost_l495_495001


namespace sum_of_palindromic_primes_l495_495789

def is_palindromic_prime (n : ℕ) : Prop :=
  n < 100 ∧ 10 ≤ n ∧ n.Prime ∧ let reversed := (n % 10) * 10 + (n / 10) in 
  reversed.Prime

theorem sum_of_palindromic_primes : 
  ∑ p in List.filter is_palindromic_prime (List.range 100), p = 429 :=
by
  sorry

end sum_of_palindromic_primes_l495_495789


namespace must_be_composite_l495_495972

theorem must_be_composite
  (n : ℕ)
  (h_n : n > 1)
  (b : ℕ → ℕ)
  (h_distinct : ∀ i j, i ≠ j → b i ≠ b j)
  (h_pos : ∀ i, b i > 0)
  (h_condition : ∀ k : ℕ, ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (∏ i in finset.range n, (b i + k)) = a ^ b) :
  ∃ r s : ℕ, r > 1 ∧ s > 1 ∧ n = r * s :=
sorry

end must_be_composite_l495_495972


namespace roots_x_squared_minus_6x_plus_25_eq_3_pm_4i_l495_495078

noncomputable def roots_of_quad_eq : ℂ → ℂ → ℂ → set ℂ
| a b c := {x | a * x^2 + b * x + c = 0 }

theorem roots_x_squared_minus_6x_plus_25_eq_3_pm_4i :
  roots_of_quad_eq 1 (-6) 25 = {3 + 4 * complex.I, 3 - 4 * complex.I} := by
  sorry

end roots_x_squared_minus_6x_plus_25_eq_3_pm_4i_l495_495078


namespace Sum_S7_l495_495080

-- Define the arithmetic sequence and the terms and sum
def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n, a n = a1 + n * d

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a 0 + (n - 1) * (a 1 - a 0))

theorem Sum_S7 :
  ∃ (a : ℕ → ℝ) (a1 d : ℝ), 
    arithmetic_sequence a a1 d ∧ 
    a 2 + a 3 + a 4 = 9 ∧
    sum_of_first_n_terms a 7 = 21 :=
by {
  sorry
}

end Sum_S7_l495_495080


namespace find_f_of_3_l495_495573

def f (x : ℕ) : ℕ := sorry

theorem find_f_of_3 (x : ℤ) (h : (x ^ (2 ^ 2009 - 1) - 1) * f x = (x + 1) * (x ^ 2 + 1) * (x ^ 4 + 1) * ∏ i in finset.range (2 ^ 2008), (x ^ 2 ^ i + 1) - 1) :
  f 3 = 3 := 
by {
  sorry
}

end find_f_of_3_l495_495573


namespace red_balls_count_l495_495713

theorem red_balls_count (R W : ℕ) (h1 : R / W = 4 / 5) (h2 : W = 20) : R = 16 := sorry

end red_balls_count_l495_495713


namespace sum_first_20_odds_is_400_l495_495491

-- Define the n-th odd positive integer
def odd_integer (n : ℕ) : ℕ := 2 * n + 1

-- Define the sum of the first n odd positive integers as a function
def sum_first_n_odds (n : ℕ) : ℕ := (n * (2 * n + 1)) / 2

-- Theorem statement: sum of the first 20 odd positive integers is 400
theorem sum_first_20_odds_is_400 : sum_first_n_odds 20 = 400 := 
  sorry

end sum_first_20_odds_is_400_l495_495491


namespace inequality_solution_l495_495443

noncomputable def solve_inequality (x : ℝ) : Prop :=
  ((x - 3) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0

theorem inequality_solution : {x : ℝ | solve_inequality x} = 
  {x : ℝ | x < 2} ∪ {x : ℝ | 3 < x ∧ x < 4} ∪ {x : ℝ | 5 < x ∧ x < 6} ∪ {x : ℝ | x > 7} :=
by
  sorry

end inequality_solution_l495_495443


namespace angle_BPF_eq_angle_CPE_l495_495305

open Triangle

variables {A B C P G E F : Point}
variables (H_internal : InternalPoint P (Triangle.mk A B C))
variables (H_angles : ∠ B P A = ∠ C P A)
variables (G_on_AP : OnSegment G A P)
variables (E_on_AC : OnLineIntersection (LineThrough B G) (LineThrough A C) = E)
variables (F_on_AB : OnLineIntersection (LineThrough O G) (LineThrough A B) = F)

theorem angle_BPF_eq_angle_CPE : ∠ B P F = ∠ C P E :=
by
  sorry

end angle_BPF_eq_angle_CPE_l495_495305


namespace tangent_circle_radius_l495_495934

def ellipse_center : ℝ := 0
def major_axis_length : ℝ := 12
def minor_axis_length : ℝ := 10

def focal_distance (a b : ℝ) : ℝ := real.sqrt (a^2 - b^2)

def circle_focal_center : ℝ := focal_distance (major_axis_length / 2) (minor_axis_length / 2)

def circle_radius_for_tangent (r : ℝ) : Prop :=
  (∀ x y : ℝ, x^2 / (major_axis_length / 2)^2 + y^2 / (minor_axis_length / 2)^2 = 1 →
   (x - circle_focal_center)^2 + y^2 = r^2) →
  (∃ x : ℝ, x^2 / (major_axis_length / 2)^2 + ((r^2 - (x - circle_focal_center)^2) / (minor_axis_length / 2)^2) = 1)

theorem tangent_circle_radius :
  ∃ r : ℝ, circle_radius_for_tangent r ∧ r = 2 :=
sorry

end tangent_circle_radius_l495_495934


namespace sphere_surface_area_l495_495098

theorem sphere_surface_area (V : ℝ) (π : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : ∀ r, V = (4/3) * π * r^3)
  (h2 : V = 72 * π) : A = 36 * π * 2^(2/3) :=
by 
  sorry

end sphere_surface_area_l495_495098


namespace benzene_molecular_weight_l495_495487

theorem benzene_molecular_weight (w: ℝ) (h: 4 * w = 312) : w = 78 :=
by
  sorry

end benzene_molecular_weight_l495_495487


namespace min_distance_from_point_on_circle_to_line_l495_495736

theorem min_distance_from_point_on_circle_to_line :
  ∀ (P : ℝ × ℝ), (P.1^2 + P.2^2 = 1) → 
  let d := |3 * P.1 - 4 * P.2 - 10| / (real.sqrt (3^2 + (-4)^2)) in
  d - 1 = 1 :=
begin
  sorry
end

end min_distance_from_point_on_circle_to_line_l495_495736


namespace trig_order_descending_l495_495434

theorem trig_order_descending :
  let θ₁ := 2 * Real.pi / 5
  let θ₂ := 6 * Real.pi / 5
  let θ₃ := 7 * Real.pi / 5
  θ₃ = Real.pi + θ₁ →
  sin θ₁ > 0 →
  cos θ₂ < 0 →
  tan θ₃ = tan θ₁ →
  θ₁ < Real.pi / 2 → 
  tan θ₁ > sin θ₁ →
  tan θ₃ > sin θ₁ ∧ sin θ₁ > cos θ₂ := by
  intros θ₁ θ₂ θ₃ h1 h2 h3 h4 h5 h6
  sorry

end trig_order_descending_l495_495434


namespace meryll_questions_l495_495415

theorem meryll_questions :
  let total_mc := 35
  let total_ps := 15
  let written_mc := (2/5 : ℝ) * total_mc
  let written_ps := (1/3 : ℝ) * total_ps
  let remaining_mc := total_mc - written_mc
  let remaining_ps := total_ps - written_ps
  remaining_mc + remaining_ps = 31 :=
by
  let total_mc := 35
  let total_ps := 15
  let written_mc := (2/5 : ℝ) * total_mc
  let written_ps := (1/3 : ℝ) * total_ps
  let remaining_mc := total_mc - written_mc
  let remaining_ps := total_ps - written_ps
  have h1 : remaining_mc = 21 := by sorry
  have h2 : remaining_ps = 10 := by sorry
  show remaining_mc + remaining_ps = 31 from by sorry

end meryll_questions_l495_495415


namespace ratio_C_D_l495_495237

noncomputable def C : ℚ :=
  ∑' (n : ℕ) in {k | k % 2 = 0 ∧ k % 4 ≠ 0}.to_finset, 1 / (n ^ 2)

noncomputable def D : ℚ :=
  ∑' (n : ℕ) in {k | k % 4 = 0}.to_finset, ((-1) ^ (n / 4 + 1)) / (n ^ 2)

theorem ratio_C_D : C / D = 16 := by
  sorry

end ratio_C_D_l495_495237


namespace complex_equation_solution_l495_495234

open Complex

theorem complex_equation_solution (x y : ℝ) :
  ((-5 + 2 * I) * (x: ℂ) - (3 - 4 * I) * (y: ℂ) = 2 - I) ↔ 
  (x = -5 / 14 ∧ y = -1 / 14) :=
by
  sorry

end complex_equation_solution_l495_495234


namespace smallest_percentage_all_correct_l495_495915

theorem smallest_percentage_all_correct (p1 p2 p3 : ℝ) 
  (h1 : p1 = 0.9) 
  (h2 : p2 = 0.8)
  (h3 : p3 = 0.7) :
  ∃ x, x = 0.4 ∧ (x ≤ 1 - ((1 - p1) + (1 - p2) + (1 - p3))) :=
by 
  sorry

end smallest_percentage_all_correct_l495_495915


namespace inequality_part1_inequality_part2_l495_495054

theorem inequality_part1 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := 
sorry

theorem inequality_part2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c)^2 ≥ 3 * (a * b + b * c + c * a) :=
begin
  have h1 : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := inequality_part1 a b c ha hb hc,
  have h2 : (a + b + c)^2 = a^2 + b^2 + c^2 + 2 * (a * b + b * c + c * a), by ring,
  rw h2,
  linarith,
end

end inequality_part1_inequality_part2_l495_495054


namespace triangle_area_l495_495593

def point (α : Type) := prod α α

theorem triangle_area (A B C : point ℤ) (h1 : A = (2, -3)) (h2 : B = (0, 1)) (h3 : C = (4, -4)) :
  let v := (fst C - fst A, snd C - snd A),
      w := (fst C - fst B, snd C - snd B) in
  (abs (v.1 * w.2 - v.2 * w.1) / 2 = 3) :=
by {
  sorry
}

end triangle_area_l495_495593


namespace ice_cubes_count_l495_495727

theorem ice_cubes_count (num_cups : ℕ) (pieces_per_cup : ℕ) (total_ice_cubes : ℕ)
  (h1 : num_cups = 6)
  (h2 : pieces_per_cup = 5)
  (h3 : total_ice_cubes = num_cups * pieces_per_cup) :
  total_ice_cubes = 30 := 
by
  rw [h1, h2,← h3]
  rfl

end ice_cubes_count_l495_495727


namespace complex_root_circle_radius_l495_495928

theorem complex_root_circle_radius :
  (∃ (z : ℂ), (z + 2) ^ 5 = 64 * z ^ 5) →
  (∃ (r : ℝ), r = 2 / 15) :=
begin
  sorry
end

end complex_root_circle_radius_l495_495928


namespace trapezoid_EFGH_area_l495_495941

structure Point where
  x : ℝ
  y : ℝ

def trapezoid_area (E F G H : Point) : ℝ :=
  let EF := (F.y - E.y).abs
  let GH := real.sqrt ((G.x - H.x) ^ 2 + (G.y - H.y) ^ 2)
  let height := (G.x - E.x).abs
  1/2 * (EF + GH) * height

theorem trapezoid_EFGH_area : 
  trapezoid_area ⟨0, 0⟩ ⟨0, 4⟩ ⟨6, 4⟩ ⟨3, -2⟩ = 12 + 18 * real.sqrt 5 := 
by
  sorry

end trapezoid_EFGH_area_l495_495941


namespace real_numbers_inequality_l495_495621

theorem real_numbers_inequality (x1 x2 x3 : ℝ)
  (h1 : x1 + x2 > x3) 
  (h2 : x2 + x3 > x1) 
  (h3 : x3 + x1 > x2) : 
  (2 / 3) * ((∑ i in Finset.range 3, [x1, x2, x3].nth_le i (by simp; linarith)) * 
             (∑ i in Finset.range 3, ([x1, x2, x3].nth_le i (by simp; linarith))^2)) > 
  (∑ i in Finset.range 3, ([x1, x2, x3].nth_le i (by simp; linarith))^3) + x1 * x2 * x3 :=
begin
  sorry
end

end real_numbers_inequality_l495_495621


namespace digit_start_l495_495819

theorem digit_start (a n p q : ℕ) (hp : a * 10^p < 2^n) (hq : 2^n < (a + 1) * 10^p)
  (hr : a * 10^q < 5^n) (hs : 5^n < (a + 1) * 10^q) :
  a = 3 :=
by
  -- The proof goes here.
  sorry

end digit_start_l495_495819


namespace count_pairs_l495_495662

-- Define the conditions
def isValidPair (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ (m^2 + n < 22)

-- Define the function to count valid pairs
def countValidPairs : ℕ :=
  (Finset.range 5).sum (λ m, (Finset.range 22).count (λ n, isValidPair m n))

-- State the theorem
theorem count_pairs : countValidPairs = 54 :=
by sorry

end count_pairs_l495_495662


namespace projection_AK_onto_AB_l495_495064

-- Define the necessary structures and properties.

variables {A B C K F : Point} (triangle_ABC : Triangle A B C)
variables [IsCircumcircle triangle_ABC.circle K] [AngleBisector A B C K]

-- Statement of the problem in Lean
theorem projection_AK_onto_AB (hK : isAngleBisector triangle_ABC A K)
  (hProj : isProjection A K triangle_ABC A B F) :
  (segmentLength F A) = (1/2) * (segmentLength A B + segmentLength A C) :=
sorry

end projection_AK_onto_AB_l495_495064


namespace square_of_complex_l495_495673

theorem square_of_complex (z : ℂ) (h : z = complex.mk 2 5) : z^2 = complex.mk (-21) 20 :=
by
  intro h
  rw [h]
  calc
    complex.mk 2 5 * complex.mk 2 5 = complex.mk 4 20 + complex.mk 0 (-25) := by simp
    ... = complex.mk (-21) 20 := by simp

end square_of_complex_l495_495673


namespace find_integer_pairs_l495_495142

theorem find_integer_pairs
  (h s : ℕ)
  (h_nonzero : h > 0)
  (s_nonzero : s > 0)
  (lines_condition : ∀ (i j k : ℕ), 
    i ≠ j → j ≠ k → k ≠ i → 
    ∀ (hi hj hk : ℕ), hi = 1 → hj = 1 → hk = s → i + j + k ≠ h + s)
  : h ⋅ (s + 1) + 1 + (s ⋅ (s + 1)) / 2 = 1992 
    ↔ (h = 995 ∧ s = 1) ∨ (h = 176 ∧ s = 10) ∨ (h = 80 ∧ s = 21) := 
by 
  sorry

end find_integer_pairs_l495_495142


namespace valentines_left_l495_495764

theorem valentines_left (initial valentines_to_children valentines_to_neighbors valentines_to_coworkers : ℕ) (h_initial : initial = 30)
  (h1 : valentines_to_children = 8) (h2 : valentines_to_neighbors = 5) (h3 : valentines_to_coworkers = 3) : initial - (valentines_to_children + valentines_to_neighbors + valentines_to_coworkers) = 14 := by
  sorry

end valentines_left_l495_495764


namespace problem_1_problem_2_l495_495280

open Set Real

noncomputable def A : Set ℝ := {x | 2^x > 1}

noncomputable def B : Set ℝ := {x | -1 < x ∧ x < 1}

noncomputable def C (a : ℝ) : Set ℝ := {x | x < a}

theorem problem_1 :
  A ∪ B = { x : ℝ | x > -1 } ∧ 
  (compl A) ∩ B = { x : ℝ | -1 < x ∧ x ≤ 0 } :=
by
  sorry

theorem problem_2 (a : ℝ) (h : B ∪ C a = C a) : a ≥ 1 :=
by 
  sorry

end problem_1_problem_2_l495_495280


namespace price_per_unit_max_profit_l495_495523

-- Part 1: Finding the Prices

theorem price_per_unit (x y : ℕ) 
  (h1 : 2 * x + 3 * y = 690) 
  (h2 : x + 4 * y = 720) : 
  x = 120 ∧ y = 150 :=
by
  sorry

-- Part 2: Maximizing Profit

theorem max_profit (m : ℕ) 
  (h1 : m ≤ 3 * (40 - m)) 
  (h2 : 120 * m + 150 * (40 - m) ≤ 5400) : 
  (m = 20) ∧ (40 - m = 20) :=
by
  sorry

end price_per_unit_max_profit_l495_495523


namespace problem_solution_l495_495157

noncomputable def total_profit (x : ℤ) : ℤ :=
  let second_batch := 2 * x
  let first_batch_cost := 40000
  let second_batch_cost := 88000
  let total_cost := first_batch_cost + second_batch_cost
  let selling_price := 28
  let discount_price := selling_price * 8 / 10
  let total_revenue := selling_price * (x + second_batch - 100) + discount_price * 100
  total_revenue - total_cost

theorem problem_solution : (x : ℤ) (h1 : 40000 = x * u) (h2 : 88000 = 2 * x * (u + 2)) :
  x = 2000 ∧ total_profit 2000 = 39440 :=
by
  sorry

end problem_solution_l495_495157


namespace textbook_cost_l495_495277

theorem textbook_cost 
  (credits : ℕ) 
  (cost_per_credit : ℕ) 
  (facility_fee : ℕ) 
  (total_cost : ℕ) 
  (num_textbooks : ℕ) 
  (total_spent : ℕ) 
  (h1 : credits = 14) 
  (h2 : cost_per_credit = 450) 
  (h3 : facility_fee = 200) 
  (h4 : total_spent = 7100) 
  (h5 : num_textbooks = 5) :
  (total_cost - (credits * cost_per_credit + facility_fee)) / num_textbooks = 120 :=
by
  sorry

end textbook_cost_l495_495277


namespace compound_interest_time_period_l495_495484

noncomputable def calculate_time_period (P A r : ℝ) (n : ℕ) : ℝ := 
  (Real.log (A / P)) / (n * Real.log (1 + r / n))

theorem compound_interest_time_period :
  let P := 4500
  let r := 0.10
  let A := 5445
  let n := 1
  calculate_time_period P A r n = 2 := 
by 
  rw [calculate_time_period, P, r, A, n]
  -- The specific calculations and log evaluations are skipped
  sorry

end compound_interest_time_period_l495_495484


namespace fraction_inequality_l495_495426

theorem fraction_inequality (a : ℝ) (h : a ≠ 2) : (1 / (a^2 - 4 * a + 4) > 2 / (a^3 - 8)) :=
by sorry

end fraction_inequality_l495_495426


namespace Ronald_eggs_initially_l495_495048

def total_eggs_shared (friends eggs_per_friend : Nat) : Nat :=
  friends * eggs_per_friend

theorem Ronald_eggs_initially (eggs : Nat) (candies : Nat) (friends : Nat) (eggs_per_friend : Nat)
  (h1 : friends = 8) (h2 : eggs_per_friend = 2) (h_share : total_eggs_shared friends eggs_per_friend = 16) :
  eggs = 16 := by
  sorry

end Ronald_eggs_initially_l495_495048


namespace functional_equation_solution_l495_495253

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x) + f (f y) = 2 * y + f (x - y)) ↔ (∀ x : ℝ, f x = x) := by
  sorry

end functional_equation_solution_l495_495253


namespace angle_B_parallel_lines_l495_495410

/-- Given lines l and k are parallel, m∠A = 120° and m∠C = 80°, 
    prove that m∠B = 160°. -/
theorem angle_B_parallel_lines (l k : Line) (A B C : Point)
  (h_parallel : l ∥ k) (h_angle_A : m∠A = 120) (h_angle_C : m∠C = 80) : 
  m∠B = 160 :=
by
  sorry

end angle_B_parallel_lines_l495_495410


namespace prime_cube_plus_nine_prime_l495_495337

theorem prime_cube_plus_nine_prime (P : ℕ) (hP : Nat.Prime P) (h : Nat.Prime (P^3 + 9)) : P^2 - 7 = -3 := by
  sorry

end prime_cube_plus_nine_prime_l495_495337


namespace quadratic_inequality_solution_l495_495983

theorem quadratic_inequality_solution (y : ℝ) : 
  (y^2 - 9 * y + 14 ≤ 0) ↔ (2 ≤ y ∧ y ≤ 7) :=
sorry

end quadratic_inequality_solution_l495_495983


namespace probability_sum_5_of_two_dice_rolls_l495_495549

theorem probability_sum_5_of_two_dice_rolls : 
  let die_faces := ({1, 2, 3, 4, 5, 6} : set ℕ),
      trials := 2
  in ∑ x in die_faces, ∑ y in die_faces, (if (x + y = 5) then 1 else 0) / (die_faces.card ^ trials) = 1 / 9 := 
by
  -- Definitions of a die with 6 faces and rolling it twice.
  let die_faces := ({1, 2, 3, 4, 5, 6} : set ℕ)
  let trials := 2

  /- The number of possible outcomes is die_faces.card ^ trials -/
  let total_outcomes := die_faces.card ^ trials
  
  /- Sum over all outcomes; only include those where the sum is 5 -/
  let favorable_sum := ∑ x in die_faces, ∑ y in die_faces, if (x + y = 5) then 1 else 0

  /- The required probability is the ratio of favorable outcomes to total outcomes -/
  have h_favorable : favorable_sum = 4 := sorry
  have h_total : total_outcomes = 36 := sorry

  /- Now we compute the probability -/
  have h_probability : (favorable_sum : ℚ) / total_outcomes = 1 / 9 := sorry

  exact h_probability

end probability_sum_5_of_two_dice_rolls_l495_495549


namespace rectangle_area_is_243_square_meters_l495_495179

-- Given conditions
variables (w l : ℝ)
-- The width is one-third of its length
def width_one_third_length : Prop := w = l / 3

-- The perimeter of the rectangle is 72 meters
def perimeter_72 : Prop := 2 * (w + l) = 72

-- The area of the rectangle
def area : ℝ := w * l

-- Proof statement
theorem rectangle_area_is_243_square_meters 
  (h1 : width_one_third_length w l) 
  (h2 : perimeter_72 w l) : area w l = 243 := 
by 
sorry

end rectangle_area_is_243_square_meters_l495_495179


namespace negation_of_exists_l495_495466

theorem negation_of_exists (x : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - x + 1 > 0) :=
by
  sorry

end negation_of_exists_l495_495466


namespace sheila_hourly_wage_l495_495896

theorem sheila_hourly_wage :
  ∀ (h1 : 8 * 3 = 24)
    (h2 : 6 * 2 = 12)
    (h3 : 24 + 12 = 36)
    (h4 : 360 / 36 = 10),
  360 / (8 * 3 + 6 * 2) = 10 := by
  intros
  rw [← h1, ← h2, ← h3, ← h4]
  exact h4

end sheila_hourly_wage_l495_495896


namespace meryll_questions_l495_495418

/--
Meryll wants to write a total of 35 multiple-choice questions and 15 problem-solving questions. 
She has written \(\frac{2}{5}\) of the multiple-choice questions and \(\frac{1}{3}\) of the problem-solving questions.
We need to prove that she needs to write 31 more questions in total.
-/
theorem meryll_questions : (35 - (2 / 5) * 35) + (15 - (1 / 3) * 15) = 31 := by
  sorry

end meryll_questions_l495_495418


namespace product_greater_than_half_l495_495794

theorem product_greater_than_half (n : ℕ) (h : 2 ≤ n) :
  (∏ k in Finset.range (n + 1) \ Finset.range 2, (1 - (1 / (k^2 : ℝ)))) > (1 / 2) :=
sorry

end product_greater_than_half_l495_495794


namespace solve_for_y_l495_495057

theorem solve_for_y (y : ℝ) (h : 50^4 = 10^y) : y = 6.79588 :=
sorry

end solve_for_y_l495_495057


namespace ducks_cows_problem_l495_495366

theorem ducks_cows_problem (D C : ℕ) (h : 2 * D + 4 * C = 2 * (D + C) + 24) : C = 12 := 
  sorry

end ducks_cows_problem_l495_495366


namespace set_union_intersection_l495_495029

-- Definitions
def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}
def C : Set ℤ := {1, 2}

-- Theorem statement
theorem set_union_intersection : (A ∩ B ∪ C) = {0, 1, 2} :=
by
  sorry

end set_union_intersection_l495_495029


namespace rhombus_fraction_of_square_l495_495770

-- Define the vertices of the rhombus
def vertex_1 : (ℝ × ℝ) := (3, 2)
def vertex_2 : (ℝ × ℝ) := (4, 3)
def vertex_3 : (ℝ × ℝ) := (3, 4)
def vertex_4 : (ℝ × ℝ) := (2, 3)

-- Define the side length of the larger square
def side_length_larger_square : ℝ := 6

-- Areas and the fraction:
def area_rhombus : ℝ := 1
def area_larger_square : ℝ := side_length_larger_square^2  -- which is 36
def target_fraction : ℝ := area_rhombus / area_larger_square  -- which is 1/36

-- Proof goal: The fraction of the area of the larger square that is covered by the rhombus
theorem rhombus_fraction_of_square :
  target_fraction = 1 / 36 := 
by 
  -- The detailed proof isn't required, so we provide a stub.
  sorry

end rhombus_fraction_of_square_l495_495770


namespace fraction_lt_sqrt2_bound_l495_495404

theorem fraction_lt_sqrt2_bound (m n : ℕ) (h : (m : ℝ) / n < Real.sqrt 2) :
  (m : ℝ) / n < Real.sqrt 2 * (1 - 1 / (4 * (n * n))) :=
sorry

end fraction_lt_sqrt2_bound_l495_495404


namespace minimize_perimeter_of_quadrilateral_l495_495723

theorem minimize_perimeter_of_quadrilateral
  (A B : Point) (angle_acute : acute_angle A B)
  (M N : Point) (M_inside : inside_angle M angle_acute) (N_inside : inside_angle N angle_acute)
  (M1 N1 : Point) (M1_symmetric : symmetric_point M M1 (side A))
  (N1_symmetric : symmetric_point N N1 (side B))
  (K L : Point) (K_on_side : on_side K A) (L_on_side : on_side L B) :
  perimeter_quad M K L N = perimeter_path M1 K L N1 ↔ K L ∈ intersection_line_segment M1N1 (side A, side B) :=
by
  sorry

end minimize_perimeter_of_quadrilateral_l495_495723


namespace overlapping_area_fraction_l495_495170

variable (Y X : ℝ)
variable (hY : 0 < Y)
variable (hX : X = (1 / 8) * (2 * Y - X))

theorem overlapping_area_fraction : X = (2 / 9) * Y :=
by
  -- We define the conditions and relationships stated in the problem
  -- Prove the theorem accordingly
  sorry

end overlapping_area_fraction_l495_495170


namespace geometric_series_sum_l495_495848

theorem geometric_series_sum :
  let a := (1/4 : ℚ)
  ∧ let r := (1/4 : ℚ)
  ∧ let n := (5 : ℕ)
  → ∑ i in finset.range n, a * (r ^ i) = 341 / 1024 :=
by
  intro a r n
  apply sorry

end geometric_series_sum_l495_495848


namespace greatest_integer_a10_l495_495079

def sequence (a : ℕ → ℝ) : Prop := 
  a 0 = 0 ∧ ∀ n, a (n + 1) = (8 / 5) * a n + (6 / 5) * Real.sqrt (4^n - a n^2)

theorem greatest_integer_a10 
  {a : ℕ → ℝ}
  (H : sequence a) : 
  ⌊a 10⌋ = 983 := 
sorry

end greatest_integer_a10_l495_495079


namespace abs_pos_of_ne_zero_l495_495877

theorem abs_pos_of_ne_zero (a : ℤ) (h : a ≠ 0) : |a| > 0 := sorry

end abs_pos_of_ne_zero_l495_495877


namespace determine_n_l495_495032

variable (ξ : ℕ)
variable (n : ℕ)
variable (h1 : Π k, k ∈ Finset.range n → 1 ≤ n ∧ P (ξ = k) = 1 / n)
variable (h2 : P (ξ < 4) = 0.3)

theorem determine_n (h1 : Π k, k ∈ Finset.range n → 1 ≤ n ∧ P (ξ = k) = 1 / n) (h2 : P (ξ < 4) = 0.3) : n = 10 :=
sorry

end determine_n_l495_495032


namespace number_of_odd_terms_in_expansion_l495_495347

theorem number_of_odd_terms_in_expansion (p q : ℤ) (hp : Odd p) (hq : Odd q) : 
  let terms := [binom 4 0 * p^4, binom 4 1 * p^3 * q, binom 4 2 * p^2 * q^2, binom 4 3 * p * q^3, binom 4 4 * q^4] in
  (terms.filter Odd).length = 2 := by
  sorry

end number_of_odd_terms_in_expansion_l495_495347


namespace max_value_of_a_l495_495242

theorem max_value_of_a {a b c : ℝ} 
  (h1 : a^2 - b * c - 8 * a + 7 = 0) 
  (h2 : b^2 + c^2 + b * c - 6 * a + 6 = 0) :
  a ≤ 9 :=
sorry

example : ∃(a b c : ℝ), a^2 - b * c - 8 * a + 7 = 0 ∧ b^2 + c^2 + b * c - 6 * a + 6 = 0 ∧ a = 9 :=
begin
  use [9, 4, 4],
  split,
  {
    -- verify the first condition
    have h1 : 9^2 - 4*4 - 8*9 + 7 = 0,
    { norm_num },
    exact h1,
  },
  {
    -- verify the second condition
    have h2 : 4^2 + 4^2 + 4*4 - 6*9 + 6 = 0,
    { norm_num },
    exact h2,
  }
end

end max_value_of_a_l495_495242


namespace min_value_x3y3z2_is_1_over_27_l495_495019

noncomputable def min_value_x3y3z2 (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z) (h' : 1 / x + 1 / y + 1 / z = 9) : ℝ :=
  x^3 * y^3 * z^2

theorem min_value_x3y3z2_is_1_over_27 (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z)
  (h' : 1 / x + 1 / y + 1 / z = 9) : min_value_x3y3z2 x y z h h' = 1 / 27 :=
sorry

end min_value_x3y3z2_is_1_over_27_l495_495019


namespace john_total_spent_l495_495729

/-- John's expenditure calculations -/
theorem john_total_spent :
  let computer_cost := 1500
  let peripherals_cost := computer_cost / 5
  let original_video_card_cost := 300
  let upgraded_video_card_cost := original_video_card_cost * 2
  let additional_upgrade_cost := upgraded_video_card_cost - original_video_card_cost
  let total_spent := computer_cost + peripherals_cost + additional_upgrade_cost
  total_spent = 2100 :=
by
  sorry

end john_total_spent_l495_495729


namespace slope_of_line_l495_495598

theorem slope_of_line (x1 y1 x2 y2 : ℝ) (h1 : x1 = 1) (h2 : y1 = 3) (h3 : x2 = 6) (h4 : y2 = -7) : 
  (y2 - y1) / (x2 - x1) = -2 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end slope_of_line_l495_495598


namespace ratio_bc_cd_l495_495504

theorem ratio_bc_cd (a b c d e : ℝ)
  (de : ℝ) (ab : ℝ) (ac : ℝ) (ae : ℝ)
  (ha : a < b) (hb : b < c) (hc : c < d) (hd : d < e)
  (hab : ab = b - a) (hac : ac = c - a) (hae : ae = e - a) (hde : de = e - d)
  (hab_val : ab = 5) (hac_val : ac = 11) (hae_val : ae = 18) (hde_val : de = 4) :
  (c - b) / (d - c) = 2 :=
by
  have bc := c - b
  have cd := d - c
  have h_ac : ac = ab + bc := by sorry
  have h_ae : ae = ab + bc + cd + de := by sorry
  have h_bc_val : bc = ac - ab := by sorry
  have h_cd_val : cd = ae - ab - bc - de := by sorry
  rw [hab_val, hac_val, hae_val, hde_val, h_bc_val, h_cd_val]
  linarith

end ratio_bc_cd_l495_495504


namespace key_is_l495_495619

theorem key_is (p : ℕ) (hp_odd_prime : p.prime ∧ p % 2 = 1) :
  (∃ k : ℕ, k > 0 ∧ ∃ n : ℕ, n > 0 ∧ n * n = k * k - p * k) ->
  ∃ k : ℕ, k = (p + 1) * (p + 1) / 4 :=
by
  sorry

end key_is_l495_495619


namespace leading_digit_log_fraction_l495_495805

-- Definitions based on conditions
variable (M : ℝ) (a : ℝ)

-- Additional conditions as given in the problem
-- Here, we assume that the leading digit of 6 * 47 * log10 M is a given
axiom leading_digit_cond : leading_digit (6 * 47 * Real.log10 M) = a

-- Theorem to prove the question == answer given the conditions
theorem leading_digit_log_fraction :
  leading_digit (Real.log10 (1000 / M)) = 2 - a ∨ leading_digit (Real.log10 (1000 / M)) = 3 - a :=
sorry

end leading_digit_log_fraction_l495_495805


namespace smallest_positive_number_is_x2_l495_495600

noncomputable def x1 : ℝ := 14 - 4 * Real.sqrt 17
noncomputable def x2 : ℝ := 4 * Real.sqrt 17 - 14
noncomputable def x3 : ℝ := 23 - 7 * Real.sqrt 14
noncomputable def x4 : ℝ := 65 - 12 * Real.sqrt 34
noncomputable def x5 : ℝ := 12 * Real.sqrt 34 - 65

theorem smallest_positive_number_is_x2 :
  x2 = 4 * Real.sqrt 17 - 14 ∧
  (0 < x1 ∨ 0 < x2 ∨ 0 < x3 ∨ 0 < x4 ∨ 0 < x5) ∧
  (∀ x : ℝ, (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4 ∨ x = x5) → 0 < x → x2 ≤ x) := sorry

end smallest_positive_number_is_x2_l495_495600


namespace circus_dogs_l495_495939

theorem circus_dogs (x : ℕ) (h1 : 2 * x + 4 * x = 36) : 2 * x = 12 :=
by
  have h2 : 6 * x = 36 := h1
  have h3 : x = 6 := by linarith
  have h4 : 2 * x + 2 * x = 12 := by linarith
  exact h4

end circus_dogs_l495_495939


namespace intersection_A_B_l495_495320

variable (A : Set ℤ) (B : Set ℤ)

-- Define the set A and B
def set_A : Set ℤ := {0, 1, 2}
def set_B : Set ℤ := {x | 1 < x ∧ x < 4}

theorem intersection_A_B :
  set_A ∩ set_B = {2} :=
by
  sorry

end intersection_A_B_l495_495320


namespace cosine_inequality_of_tetrahedron_l495_495068

theorem cosine_inequality_of_tetrahedron (DA DB DC : ℝ) (α β γ : ℝ)
  (h_perpendicular1 : DA ≠ 0 ∧ DB ≠ 0 ∧ DC ≠ 0)
  (h_perpendicular2 : (DA * DB = 0) ∧ (DB * DC = 0) ∧ (DC * DA = 0)):
  cos α * cos β * cos γ ≤ sqrt 3 / 9 := 
by sorry

end cosine_inequality_of_tetrahedron_l495_495068


namespace total_compensation_correct_l495_495148

noncomputable def regular_rate : ℝ := 15
def regular_hours : ℝ := 40
noncomputable def overtime_rate : ℝ := regular_rate * 1.75
def total_hours_worked : ℝ := 54.32

def regular_pay : ℝ := regular_rate * regular_hours
def overtime_hours : ℝ := total_hours_worked - regular_hours
def overtime_pay : ℝ := overtime_rate * overtime_hours
def total_compensation : ℝ := regular_pay + overtime_pay

theorem total_compensation_correct :
  total_compensation = 975.90 := by
  sorry

end total_compensation_correct_l495_495148


namespace find_x_l495_495406

theorem find_x (a b x: ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : x > 0)
    (h4 : (4 * a)^(4 * b) = a^b * x^(2 * b)) : x = 16 * a^(3 / 2) := by
  sorry

end find_x_l495_495406


namespace evaluate_expression_l495_495583

def a : ℕ := 3
def b : ℕ := 2

theorem evaluate_expression : (a^2 * a^5) / (b^2 / b^3) = 4374 := by
  sorry

end evaluate_expression_l495_495583


namespace profit_per_meter_l495_495548

theorem profit_per_meter 
  (total_meters : ℕ)
  (cost_price_per_meter : ℝ)
  (total_selling_price : ℝ)
  (h1 : total_meters = 92)
  (h2 : cost_price_per_meter = 83.5)
  (h3 : total_selling_price = 9890) : 
  (total_selling_price - total_meters * cost_price_per_meter) / total_meters = 24.1 :=
by
  sorry

end profit_per_meter_l495_495548


namespace rectangular_field_area_l495_495194

theorem rectangular_field_area :
  ∃ (w l : ℝ), (l = 3 * w) ∧ (2 * (l + w) = 72) ∧ (l * w = 243) :=
by {
  sorry
}

end rectangular_field_area_l495_495194


namespace inequality_solution_l495_495445

noncomputable def solve_inequality (x : ℝ) : Prop :=
  ((x - 3) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0

theorem inequality_solution : {x : ℝ | solve_inequality x} = 
  {x : ℝ | x < 2} ∪ {x : ℝ | 3 < x ∧ x < 4} ∪ {x : ℝ | 5 < x ∧ x < 6} ∪ {x : ℝ | x > 7} :=
by
  sorry

end inequality_solution_l495_495445


namespace angle_BFP_half_B_l495_495381

variables {A B C : Type} [MetricSpace A]

structure Incenter (a b c : ℝ) (I : A) :=
(angle_A : ∠ A B C = 60)
(line_parallels_to_AC : ∃ F : A, I.line.parallel_to (AC) ∧ I ∈ F.line)

structure Point_P_on_BC (BC : ℝ) (P : A) :=
(point_chosen_on_BC : 3 * BP = BC)

theorem angle_BFP_half_B {A B C I F P : A} [MetricSpace A]
  (T : Triangle A B C)
  (I_center : Incenter A B C I)
  (P_point : Point_P_on_BC BC P)
  (BP_relation : BC / 3 = BP) :
  ∠ BFP = ∠ B / 2 :=
by sorry

end angle_BFP_half_B_l495_495381


namespace decodeSequence_correct_l495_495042

noncomputable def decodeSequence : ℕ × ℕ × ℕ × ℕ × ℕ :=
  let a := 3
  let d := 12
  (a, a + d, a + 2 * d, a + 3 * d, a + 4 * d)

theorem decodeSequence_correct :
  decodeSequence = (3, 15, 27, 39, 51) :=
by 
  unfold decodeSequence
  simp
  sorry

end decodeSequence_correct_l495_495042


namespace find_constants_for_sin_cube_l495_495829

theorem find_constants_for_sin_cube :
  ∃ c d : ℝ, (∀ θ : ℝ, sin θ ^ 3 = c * sin (3 * θ) + d * sin θ) 
  ∧ c = -1 / 4 ∧ d = 3 / 4 :=
begin
  sorry
end

end find_constants_for_sin_cube_l495_495829


namespace positive_difference_between_diagonals_l495_495231

def original_grid : Matrix (Fin 5) (Fin 5) ℕ :=
  ![[1, 2, 3, 4, 5],
    [10, 11, 12, 13, 14],
    [19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28],
    [31, 32, 33, 34, 35]]

def transformed_grid : Matrix (Fin 5) (Fin 5) ℕ :=
  ![[1, 2, 3, 4, 5],
    [14, 13, 12, 11, 10],
    [23, 22, 21, 20, 19],
    [24, 25, 26, 27, 28],
    [35, 34, 33, 32, 31]]

def main_diagonal_sum (m : Matrix (Fin 5) (Fin 5) ℕ) : ℕ :=
  m 0 0 + m 1 1 + m 2 2 + m 3 3 + m 4 4 

def secondary_diagonal_sum (m : Matrix (Fin 5) (Fin 5) ℕ) : ℕ :=
  m 0 4 + m 1 3 + m 2 2 + m 3 1 + m 4 0

theorem positive_difference_between_diagonals : 
  abs (main_diagonal_sum transformed_grid - secondary_diagonal_sum transformed_grid) = 4 := by
  sorry

end positive_difference_between_diagonals_l495_495231


namespace eden_avg_speed_20_mph_l495_495505

-- Definitions based on conditions
def speed_1 : ℝ := 15
def speed_2 : ℝ := 30
def distance_1 : ℝ := 20
def distance_2 : ℝ := 20
def total_distance : ℝ := distance_1 + distance_2

-- Statement of the problem
theorem eden_avg_speed_20_mph : 
  let time_1 := distance_1 / speed_1 in
  let time_2 := distance_2 / speed_2 in
  let total_time := time_1 + time_2 in
  (total_distance / total_time) = 20 :=
by
  sorry

end eden_avg_speed_20_mph_l495_495505


namespace circle_tangents_problem_l495_495654

open Real

-- Definition of the given conditions
variables {O O₁ O₂ : Point} {a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : Line}

-- Definitions to reflect conditions in the problem
def internal_tangent (c₁ c₂ : Circle) (l : Line) : Prop := IsTangent c₁ l ∧ IsTangent c₂ l ∧ ∃ (p ∈ c₁) (q ∈ c₂), l = LineThrough p q
def external_tangent (c₁ c₂ : Circle) (l : Line) : Prop := IsTangent c₁ l ∧ IsTangent c₂ l ∧ ∃ (p ∈ c₁) (q ∈ c₂), l = LineThrough p q

-- Center intersection definitions
def center_intersection (l₁ l₂ : Line) (O : Point) : Prop := ∃ p, p = O ∧ MeantBy l₁ p ∧ MeantBy l₂ p

-- Our mathematically equivalent proof problem
theorem circle_tangents_problem
  (h1 : ¬ (Intersect (Circle O₁) (Circle O₂))) 
  (h2 : internal_tangent (Circle O₁) (Circle O₂) a₁) 
  (h3 : internal_tangent (Circle O₁) (Circle O₂) a₂)
  (h4 : external_tangent (Circle O₁) (Circle O₂) a₃) 
  (h5 : external_tangent (Circle O₁) (Circle O₂) a₄)
  (h6 : IsTangent (Circle O₁) a₅ ∧ LineThrough a₅ O₂)
  (h7 : IsTangent (Circle O₁) a₆ ∧ LineThrough a₆ O₂)
  (h8 : IsTangent (Circle O₂) a₇ ∧ LineThrough a₇ O₁)
  (h9 : IsTangent (Circle O₂) a₈ ∧ LineThrough a₈ O₁)
  (h10 : center_intersection a₁ a₂ O) :
  ∃ r₁ r₂ : ℝ, Circle O r₁ ∧ Circle O r₂ ∧ ( ∀ (l = a₃ ∨ l = a₄), IsTangent (Circle O r₁) l ) ∧ 
  ( ∀ (l = a₅ ∨ l = a₆ ∨ l = a₇ ∨ l = a₈), IsTangent (Circle O r₂) l ) ∧ r₂ = r₁ / 2 :=
by sorry

end circle_tangents_problem_l495_495654


namespace geometric_series_sum_l495_495864

theorem geometric_series_sum :
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  (a + a * r + a * r^2 + a * r^3 + a * r^4) = 341 / 1024 :=
by
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  -- Proof steps go here
  sorry

end geometric_series_sum_l495_495864


namespace find_possible_values_l495_495363
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def satisfies_conditions (a bc de fg : ℕ) : Prop :=
  (a % 2 = 0) ∧ (is_prime bc) ∧ (de % 5 = 0) ∧ (fg % 3 = 0) ∧
  (fg - de = de - bc) ∧ (de - bc = bc - a)

theorem find_possible_values :
  ∃ (debc1 debc2 : ℕ),
    (satisfies_conditions 6 (debc1 % 100) ((debc1 / 100) % 100) ((debc1 / 10000) % 100)) ∧
    (satisfies_conditions 6 (debc2 % 100) ((debc2 / 100) % 100) ((debc2 / 10000) % 100)) ∧
    (debc1 = 2013 ∨ debc1 = 4023) ∧
    (debc2 = 2013 ∨ debc2 = 4023) :=
  sorry

end find_possible_values_l495_495363


namespace non_zero_number_is_9_l495_495509

theorem non_zero_number_is_9 (x : ℝ) (hx : x ≠ 0) (h : (x + x^2) / 2 = 5 * x) : x = 9 :=
sorry

end non_zero_number_is_9_l495_495509


namespace arithmetic_mean_of_prime_numbers_l495_495975

-- Define the list of numbers
def num_list := [14, 17, 19, 22, 26, 31]

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Filter the prime numbers from the list
def prime_numbers : list ℕ := num_list.filter is_prime

-- Define the arithmetic mean function
def arithmetic_mean (l : list ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

-- State the theorem
theorem arithmetic_mean_of_prime_numbers :
  arithmetic_mean prime_numbers = 67 / 3 :=
by
  sorry

end arithmetic_mean_of_prime_numbers_l495_495975


namespace minimum_x_value_l495_495159

theorem minimum_x_value
  (sales_jan_may june_sales x : ℝ)
  (h_sales_jan_may : sales_jan_may = 38.6)
  (h_june_sales : june_sales = 5)
  (h_total_sales_condition : sales_jan_may + june_sales + 2 * june_sales * (1 + x / 100) + 2 * june_sales * (1 + x / 100)^2 ≥ 70) :
  x = 20 := by
  sorry

end minimum_x_value_l495_495159


namespace geom_series_sum_l495_495852

theorem geom_series_sum : 
  let a₀ := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 5 in
  ∑ i in Finset.range n, a₀ * r ^ i = 341 / 1024 := 
  by
    sorry

end geom_series_sum_l495_495852


namespace max_radius_of_circle_l495_495693

theorem max_radius_of_circle (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
by
  sorry

end max_radius_of_circle_l495_495693


namespace palindromic_primes_sum_l495_495771

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.reverse.foldl (λ acc d, acc * 10 + d) 0

def is_palindromic_prime (n : ℕ) : Prop :=
  n < 100 ∧ 10 ≤ n ∧ is_prime n ∧ is_prime (reverse_digits n)

def sum_of_palindromic_primes : ℕ :=
  ((list.range 100).filter is_palindromic_prime).sum

theorem palindromic_primes_sum : sum_of_palindromic_primes = 429 := 
  by 
    sorry

end palindromic_primes_sum_l495_495771


namespace palindromic_primes_sum_l495_495774

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.reverse.foldl (λ acc d, acc * 10 + d) 0

def is_palindromic_prime (n : ℕ) : Prop :=
  n < 100 ∧ 10 ≤ n ∧ is_prime n ∧ is_prime (reverse_digits n)

def sum_of_palindromic_primes : ℕ :=
  ((list.range 100).filter is_palindromic_prime).sum

theorem palindromic_primes_sum : sum_of_palindromic_primes = 429 := 
  by 
    sorry

end palindromic_primes_sum_l495_495774


namespace total_games_basketball_tournament_l495_495765

theorem total_games_basketball_tournament :
  ∃ games : ℕ,
  let preliminary_games := 2,
      first_round_games := 8,
      second_round_games := 4,
      third_round_games := 2,
      final_game := 1
  in games = preliminary_games + first_round_games + second_round_games + third_round_games + final_game ∧ games = 17 :=
by
  sorry

end total_games_basketball_tournament_l495_495765


namespace rectangular_field_area_l495_495198

theorem rectangular_field_area (w : ℕ) (h : ℕ) (P : ℕ) (A : ℕ) (h_length : h = 3 * w) (h_perimeter : 2 * (w + h) = 72) : A = w * h := 
by
  -- Given conditions
  have h_eq : h = 3 * w := h_length
  rewrite [h_eq] at *
  -- Given perimeter equals 8w = 72
  have w_value : w = 9 := sorry
  -- Length calculated as 3w = 27
  have h_value : h = 27 := sorry
  -- Now we calculate the area
  calc
    A = w * h : by sorry
    ... = 9 * 27 : by sorry
    ... = 243 : by sorry

end rectangular_field_area_l495_495198


namespace quadrilateral_MNPQ_inscribed_in_circle_with_center_O_l495_495066

-- Definitions from the conditions
variables (A B C D O M N P Q : Point)

-- Conditions translations
variable (h1 : cyclic_quad A B C D)
variable (h2 : intersect_bisector O A C B D)  -- O is the intersection of diagonals
variable (h3 : sec_inter A O B (circumcircle A O B) BC = M) -- M is the second intersection
variable (h4 : sec_inter A O B (circumcircle A O B) AD = N) -- N is the second intersection
variable (h5 : sec_inter C O D (circumcircle C O D) BC = P) -- P is the second intersection
variable (h6 : sec_inter C O D (circumcircle C O D) AD = Q) -- Q is the second intersection

-- Proof goal
theorem quadrilateral_MNPQ_inscribed_in_circle_with_center_O :
  is_concyclic M N P Q ∧ circumcenter M N P Q = O :=
by
  sorry

end quadrilateral_MNPQ_inscribed_in_circle_with_center_O_l495_495066


namespace sum_sqrt_inequality_l495_495289

theorem sum_sqrt_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (3 / 2) * (a + b + c) ≥ (Real.sqrt (a^2 + b * c) + Real.sqrt (b^2 + c * a) + Real.sqrt (c^2 + a * b)) :=
by
  sorry

end sum_sqrt_inequality_l495_495289


namespace tetrahedron_height_l495_495104

theorem tetrahedron_height (r1 r2 r3 r4 : ℝ) (d12 d13 d23 d14 d24 d34 : ℝ)
    (h1 : r1 = 2) (h2 : r2 = 2) (h3 : r3 = 2) (h4 : r4 = 3)
    (h5 : d12 = 4) (h6 : d13 = 4) (h7 : d23 = 4)
    (h8 : d14 = 5) (h9 : d24 = 5) (h10 : d34 = 5) : 
    (h : ℝ) := 
  h = (sqrt(177) + 9 * sqrt(3)) / 3 :=
sorry

end tetrahedron_height_l495_495104


namespace peter_can_transfer_all_money_into_two_accounts_peter_cannot_always_transfer_all_money_into_one_account_l495_495041

-- Define the conditions
variable (a b c : ℕ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)

-- Part 1
theorem peter_can_transfer_all_money_into_two_accounts :
  ∃ x y, (x + y = a + b + c ∧ y = 0) ∨
          (∃ z, (a + b + c = x + y + z ∧ y = 0 ∧ z = 0)) :=
  sorry

-- Part 2
theorem peter_cannot_always_transfer_all_money_into_one_account :
  ((a + b + c) % 2 = 1 → ¬ ∃ x, x = a + b + c) :=
  sorry

end peter_can_transfer_all_money_into_two_accounts_peter_cannot_always_transfer_all_money_into_one_account_l495_495041


namespace evaluate_expression_at_three_l495_495250

-- Define the evaluation of the expression (x^x)^(x^x) at x=3
theorem evaluate_expression_at_three : (3^3)^(3^3) = 27^27 := by
  sorry

end evaluate_expression_at_three_l495_495250


namespace sum_of_first_70_terms_l495_495704

-- Variable definitions for the conditions, sum function of the arithmetic progression (AP)
variables (a d : ℝ)

-- Condition 1: Sum of the first 20 terms equals 200
def condition1 : Prop :=
  20 / 2 * (2 * a + 19 * d) = 200

-- Condition 2: Sum of the first 50 terms equals 50
def condition2 : Prop :=
  50 / 2 * (2 * a + 49 * d) = 50

-- The question to prove the sum of the first 70 terms
theorem sum_of_first_70_terms (h1 : condition1 a d) (h2 : condition2 a d) :
  35 * (2 * a + 69 * d) = -350 := 
begin
  sorry
end

end sum_of_first_70_terms_l495_495704


namespace minimal_sum_of_table_l495_495371

-- Define the conditions
variables (T : ℕ → ℕ → ℕ)
variable (sumOfSubrectangle : ℕ → ℕ → ℕ → ℕ → ℕ)

-- Define the sum of a subrectangle
def subrectangle_sum_condition (x1 y1 : ℕ) (x2 y2 : ℕ) : Prop :=
  x2 - x1 = 2 ∧ y2 - y1 = 1 ∧ sumOfSubrectangle x1 y1 x2 y2 = 10 ∨
  x2 - x1 = 1 ∧ y2 - y1 = 2 ∧ sumOfSubrectangle x1 y1 x2 y2 = 10

-- Sum the entire table
def total_sum (T : ℕ → ℕ → ℕ) : ℕ :=
  (Finset.range 8).sum (λ i, (Finset.range 8).sum (λ j, T i j))

-- Final theorem statement
theorem minimal_sum_of_table
  (h : ∀ x1 y1 x2 y2, subrectangle_sum_condition (x1) (y1) (x2) (y2)) :
  total_sum T = 105 :=
sorry

end minimal_sum_of_table_l495_495371


namespace sum_first_20_odds_is_400_l495_495490

-- Define the sequence of the first 20 positive odd integers
def sequence (n : ℕ) : ℕ := 1 + 2 * n

-- Define the sum of the first 'n' terms of an arithmetic sequence
def sum_arithmetic_sequence (a l n : ℕ) : ℕ := ((a + l) * n) / 2

-- Define the sum of the first 20 positive odd integers
def sum_first_20_odds : ℕ := sum_arithmetic_sequence 1 39 20

-- Claim that the sum of the first 20 positive odd integers is 400
theorem sum_first_20_odds_is_400 : sum_first_20_odds = 400 :=
by
  -- Proof omitted
  sorry

end sum_first_20_odds_is_400_l495_495490


namespace smallest_angle_l495_495393

variables (a b c : ℝ^3)
variables (θ : ℝ)

def norm (v : ℝ^3) : ℝ := real.sqrt (v.1^2 + v.2^2 + v.3^2)
def dot (u v : ℝ^3) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
def cross (u v : ℝ^3) : ℝ^3 := (
  u.2 * v.3 - u.3 * v.2,
  u.3 * v.1 - u.1 * v.3,
  u.1 * v.2 - u.2 * v.1
)

noncomputable def angle (u v : ℝ^3) : ℝ :=
  real.arccos ((dot u v) / ((norm u) * (norm v)))

theorem smallest_angle (h₁ : norm a = 1)
  (h₂ : norm b = 1)
  (h₃ : norm c = 3)
  (h₄ : a × (a × c) + b = 0) :
  angle a c = real.arccos (2 * real.sqrt 2 / 3) :=
sorry

end smallest_angle_l495_495393


namespace projection_of_vector_a_on_b_l495_495323

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let norm_b := Real.sqrt (b.1^2 + b.2^2)
  dot_product / norm_b

theorem projection_of_vector_a_on_b
  (a b : ℝ × ℝ) 
  (ha : Real.sqrt (a.1^2 + a.2^2) = 1)
  (hb : Real.sqrt (b.1^2 + b.2^2) = 2)
  (theta : ℝ)
  (h_theta : theta = Real.pi * (5/6)) -- 150 degrees in radians
  (h_cos_theta : Real.cos theta = -(Real.sqrt 3 / 2)) :
  vector_projection a b = -Real.sqrt 3 / 2 := 
by
  sorry

end projection_of_vector_a_on_b_l495_495323


namespace packs_of_green_balls_l495_495412

variables (packs_red packs_yellow packs_green balls_per_pack total_balls : ℕ)

def total_balls_red : ℕ := packs_red * balls_per_pack
def total_balls_yellow : ℕ := packs_yellow * balls_per_pack
def total_balls_green : ℕ := packs_green * balls_per_pack

def total_balls_calculated : ℕ := total_balls_red + total_balls_yellow + total_balls_green

theorem packs_of_green_balls
    (h1 : packs_red = 4)
    (h2 : packs_yellow = 8)
    (h3 : balls_per_pack = 10)
    (h4 : total_balls = 160)
    (h5 : total_balls_calculated = total_balls) :
    packs_green = 4 :=
by 
  sorry

end packs_of_green_balls_l495_495412


namespace area_of_triangle_l495_495510

theorem area_of_triangle (P r : ℝ) (hP : P = 40) (hr : r = 2.5) : 
  let s := P / 2 in
  let A := r * s in
  A = 50 := 
by
  sorry

end area_of_triangle_l495_495510


namespace palindromic_primes_sum_l495_495775

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.reverse.foldl (λ acc d, acc * 10 + d) 0

def is_palindromic_prime (n : ℕ) : Prop :=
  n < 100 ∧ 10 ≤ n ∧ is_prime n ∧ is_prime (reverse_digits n)

def sum_of_palindromic_primes : ℕ :=
  ((list.range 100).filter is_palindromic_prime).sum

theorem palindromic_primes_sum : sum_of_palindromic_primes = 429 := 
  by 
    sorry

end palindromic_primes_sum_l495_495775


namespace ab_le_neg_one_over_2019_l495_495023

theorem ab_le_neg_one_over_2019
  (u : Fin 2019 → ℝ)
  (h_sum : ∑ i, u i = 0)
  (h_sum_sq : ∑ i, (u i)^2 = 1)
  (a : ℝ) (ha : a = Finset.max' (Finset.univ.image u) sorry)
  (b : ℝ) (hb : b = Finset.min' (Finset.univ.image u) sorry) :
  a * b ≤ -1 / 2019 := 
  by 
  sorry

end ab_le_neg_one_over_2019_l495_495023


namespace equilateral_triangle_condition_l495_495294

noncomputable def P_seq (A : ℕ → ℂ) (P0 : ℂ) : ℕ → ℂ
| 0       := P0
| (k + 1) := (P_seq k * (complex.exp (-2 * real.pi * complex.I / 3))) + 
              (A (k + 1) * (1 - complex.exp (-2 * real.pi * complex.I / 3)))

theorem equilateral_triangle_condition (A : ℕ → ℂ) (P0 : ℂ)
  (hA_periodic : ∀ s ≥ 4, A s = A (s - 3))
  (hP1986 : P_seq A P0 1986 = P0) :
  (∃ B C : ℂ, A 1 = 0 ∧ A 2 = B ∧ A 3 = C ∧ B * (complex.exp (2 * real.pi * complex.I / 3)) = C ∧ C * (complex.exp (2 * real.pi * complex.I / 3)) = B) :=
begin
  sorry
end

end equilateral_triangle_condition_l495_495294


namespace probability_25_sixes_probability_at_least_one_one_expected_number_of_sixes_expected_sum_of_faces_l495_495606

-- Describe the conditions
def cube_formation : Prop :=
  ∃ (cubes : Fin 27 → Fin 6 → ℝ), -- each die face has an equal probability of 1/6
    (∀ i, (∑ j, cubes i j = 1) ∧ (∀ j, cubes i j = 1 / 6))  -- valid probabilities

-- Part a)
theorem probability_25_sixes (h : cube_formation) : 
  let p := (31 : ℝ) / (2^13 * 3^18) in p > 0 := sorry
  
-- Part b)
theorem probability_at_least_one_one (h : cube_formation) :
  let p := 1 - (5^6 : ℝ) / (2^2 * 3^18) in p > 0 := sorry

-- Part c)
theorem expected_number_of_sixes (h : cube_formation) :
  let e := 9 in e > 0 := sorry

-- Part d)
theorem expected_sum_of_faces (h : cube_formation) : 
  let e := 6 - (5^6 : ℝ) / (2 * 3^17) in e > 0 := sorry

end probability_25_sixes_probability_at_least_one_one_expected_number_of_sixes_expected_sum_of_faces_l495_495606


namespace inequality_proof_l495_495313

theorem inequality_proof (a b : ℝ) (h_a : a > 0) (h_b : 3 + b = a) : 
  3 / b + 1 / a >= 3 :=
sorry

end inequality_proof_l495_495313


namespace tan_beta_identity_l495_495343

theorem tan_beta_identity (α β : ℝ) (h1 : Real.tan α = 1/3) (h2 : Real.tan (α + β) = 1/2) :
  Real.tan β = 1/7 :=
sorry

end tan_beta_identity_l495_495343


namespace calculate_expression_l495_495228

noncomputable def sqrt3 := real.sqrt 3
noncomputable def cbrt27 := real.cbrt 27

theorem calculate_expression : sqrt3 * (1 - sqrt3) - |(-sqrt3)| + cbrt27 = 0 :=
by
  -- proof steps will go here.
  sorry

end calculate_expression_l495_495228


namespace ratio_is_eight_l495_495022

noncomputable def calculate_ratio (A B C : ℝ × ℝ × ℝ) (l m n : ℝ) :=
  let p : ℝ := l + 1
  let q : ℝ := m + 1
  let r : ℝ := n + 1
  let BC2 : ℝ := 4 * (q^2 + r^2) 
  let AC2 : ℝ := 4 * (p^2 + r^2)
  let AB2 : ℝ := 4 * (p^2 + q^2)
  (AB2 + AC2 + BC2) / ((l+1)^2 + (m+1)^2 + (n+1)^2)

theorem ratio_is_eight (A B C : ℝ × ℝ × ℝ) (l m n : ℝ) 
  (mid_BC : (l + 1, 0, 0)) 
  (mid_AC : (0, m + 1, 0))
  (mid_AB : (0, 0, n + 1)) :
  calculate_ratio A B C l m n = 8 :=
by
  sorry

end ratio_is_eight_l495_495022


namespace rectangular_field_area_l495_495173

theorem rectangular_field_area :
  ∃ (w l : ℝ), w = l / 3 ∧ 2 * (w + l) = 72 ∧ w * l = 243 :=
by
  sorry

end rectangular_field_area_l495_495173


namespace xiao_wang_program_output_l495_495887

theorem xiao_wang_program_output (n : ℕ) (h : n = 8) : (n : ℝ) / (n^2 + 1) = 8 / 65 := by
  sorry

end xiao_wang_program_output_l495_495887


namespace average_of_first_18_even_numbers_l495_495890

theorem average_of_first_18_even_numbers : 
  let first_term := 2
  let difference := 2
  let n := 18
  let last_term := first_term + (n - 1) * difference
  let average := (first_term + last_term) / 2
  average = 19 := 
by 
  let first_term := 2
  let difference := 2
  let n := 18
  let last_term := first_term + (n - 1) * difference
  let average := (first_term + last_term) / 2
  show average = 19 from sorry

end average_of_first_18_even_numbers_l495_495890


namespace ellipse_problem_l495_495301

-- Definitions based on conditions
def ellipse_eq (P : ℝ × ℝ) : Prop := (P.1^2 / 9) + (P.2^2 / 4) = 1
def foci (a b : ℝ) : Prop := a = 3 ∧ b = 2
def cos_angle_f (c : ℝ) : Prop := c = 1 / 3

-- Define the proof problem in Lean 4
theorem ellipse_problem
    (P : ℝ × ℝ)
    (h1 : ellipse_eq P)
    (h2 : foci 3 2)
    (h3 : cos_angle_f (1 / 3)) :
    ∃ (F1 F2 : ℝ × ℝ),
      let |PF1| := (P.1 - F1.1) ^ 2 + (P.2 - F1.2) ^ 2,
          |PF2| := (P.1 - F2.1) ^ 2 + (P.2 - F2.2) ^ 2,
          c := real.sqrt (3 ^ 2 - 2 ^ 2)
      in (PF1 + PF2 = 6 + 2 * c) ∧
         (1 / 2 * |PF1| * |PF2| * real.sqrt (1 - (1 / 3) ^ 2) = 2 * real.sqrt 2) ∧
         (1 / 2 * 2 * c * P.2 = 2 * real.sqrt 2) ∧
         (|PF1| * |PF2| * (1 / 3) = 2)  :=
by sorry

end ellipse_problem_l495_495301


namespace verify_sum_of_new_rates_proof_l495_495464

-- Given conditions and initial setup
variable (k : ℕ)
variable (h_initial : ℕ := 5 * k) -- Hanhan's initial hourly rate
variable (x_initial : ℕ := 4 * k) -- Xixi's initial hourly rate
variable (increment : ℕ := 20)    -- Increment in hourly rates

-- New rates after increment
variable (h_new : ℕ := h_initial + increment) -- Hanhan's new hourly rate
variable (x_new : ℕ := x_initial + increment) -- Xixi's new hourly rate

-- Given ratios
variable (initial_ratio : h_initial / x_initial = 5 / 4) 
variable (new_ratio : h_new / x_new = 6 / 5)

-- Target sum of the new hourly rates
def sum_of_new_rates_proof : Prop :=
  h_new + x_new = 220

theorem verify_sum_of_new_rates_proof : sum_of_new_rates_proof k :=
by
  sorry

end verify_sum_of_new_rates_proof_l495_495464


namespace probability_between_lines_l495_495705

-- Definitions for lines k and n
def line_k (x : ℝ) : ℝ := -3 * x + 9
def line_n (x : ℝ) : ℝ := -6 * x + 9

-- Theorem statement
theorem probability_between_lines : 
  let area_k := (1/2) * 3 * 9,
      area_n := (1/2) * 1.5 * 9,
      area_between := area_k - area_n
  in area_between / area_k = 0.5 :=
by sorry

end probability_between_lines_l495_495705


namespace tetrahedron_interior_point_sum_gt_540_l495_495428

noncomputable def tetrahedron_interior_point_angle_sum (A B C D P : Point) : Prop :=
  let α := angle B P C
  let β := angle C P A
  let γ := angle A P B
  ∃ (x y z : Angle), 
  (x + y + z < α + β + γ) ∧
  (sum_subtended_angles P A B C D > (3 * Real.pi))

theorem tetrahedron_interior_point_sum_gt_540 {A B C D P : Point}
  (h1 : ∃ (α β γ : Angle), α + β + γ = 3 * Real.pi)
  (h2 : ∃ (x y z : Angle), x + y + z < α + β + γ) :
  sum_subtended_angles A B C D P > (3 * Real.pi) := sorry

end tetrahedron_interior_point_sum_gt_540_l495_495428


namespace part_a_part_b_l495_495825

noncomputable def volume_of_prism (V : ℝ) : ℝ :=
  (9 / 250) * V

noncomputable def max_volume_of_prism (V : ℝ) : ℝ :=
  (1 / 12) * V

theorem part_a (V : ℝ) :
  volume_of_prism V = (9 / 250) * V :=
  by sorry

theorem part_b (V : ℝ) :
  max_volume_of_prism V = (1 / 12) * V :=
  by sorry

end part_a_part_b_l495_495825


namespace num_ordered_pairs_eq_ten_l495_495274

def harmonic_mean (a b : ℕ) : ℚ := 2 * a * b / (a + b)

theorem num_ordered_pairs_eq_ten :
  ∃ (S : Finset (ℕ × ℕ)), (∀ p ∈ S, p.1 < p.2 ∧ harmonic_mean p.1 p.2 = 5^10) ∧ S.card = 10 :=
begin
  sorry
end

end num_ordered_pairs_eq_ten_l495_495274


namespace value_of_n_l495_495395

theorem value_of_n (n : ℕ) (h1 : 0 < n) (h2 : n < Real.sqrt 65) (h3 : Real.sqrt 65 < n + 1) : n = 8 := 
sorry

end value_of_n_l495_495395


namespace find_function_simpler_find_function_l495_495168

-- Define the function f as per the problem statement
def f (x : ℕ) : ℚ := x / 60

-- Define the theorem to prove the relationship given the conditions
theorem find_function : ∀ (x : ℕ), x + 17 = 60 * f x → f 3 = 1 / 3 :=
by
  intros x hx
  have h : 3 + 17 = 60 * f 3 := by -- explicitly use the given condition x = 3 
    sorry
  have h' : f 3 = 20 / 60 := by -- the transformation based on the given main equation
    sorry
  have h'' : 20 / 60 = 1 / 3 := by -- simplify 20/60
    sorry
  exact h''.trans h'

-- Alternative simpler proof given a direct substitution x = 3 to show f(3) = 1 / 3
theorem simpler_find_function (x : ℕ) (hx : x = 3) : f x = 1 / 3 :=
by
  rw [hx]
  sorry

end find_function_simpler_find_function_l495_495168


namespace largest_value_of_h_squared_k_l495_495803

theorem largest_value_of_h_squared_k (h k : ℝ → ℝ) 
  (h_range : ∀ x, -3 ≤ h x ∧ h x ≤ 5)
  (k_range : ∀ x, 0 ≤ k x ∧ k x ≤ 4) : 
  ∃ d, d = 100 ∧ (∀ x, h(x)^2 * k(x) ≤ d) :=
by
  use 100
  split
  sorry
  intros x
  have h_bound : -3 ≤ h x ∧ h x ≤ 5 := h_range x
  have k_bound : 0 ≤ k x ∧ k x ≤ 4 := k_range x
  sorry

end largest_value_of_h_squared_k_l495_495803


namespace alcohol_percentage_after_additions_l495_495134

/-
Problem statement:
A 40-liter solution of alcohol and water is 5% alcohol. If 4.5 liters of alcohol and 5.5 liters of water are added to this solution, what percent of the solution produced is alcohol?

Conditions:
1. Initial solution volume = 40 liters
2. Initial percentage of alcohol = 5%
3. Volume of alcohol added = 4.5 liters
4. Volume of water added = 5.5 liters

Correct answer:
The percent of the solution that is alcohol after the additions is 13%.
-/

theorem alcohol_percentage_after_additions (initial_volume : ℝ) (initial_percentage : ℝ) 
  (alcohol_added : ℝ) (water_added : ℝ) :
  initial_volume = 40 ∧ initial_percentage = 5 ∧ alcohol_added = 4.5 ∧ water_added = 5.5 →
  ((initial_percentage / 100 * initial_volume + alcohol_added) / (initial_volume + alcohol_added + water_added) * 100) = 13 :=
by simp; sorry

end alcohol_percentage_after_additions_l495_495134


namespace radius_of_sphere_correct_l495_495118

noncomputable def radius_of_sphere_intersecting_face_diagonals : ℝ :=
  let r := sqrt (7 / 8 - sqrt 2 / 2) in
  r

theorem radius_of_sphere_correct (edge_length : ℝ) (h : edge_length = 1) :
  radius_of_sphere_intersecting_face_diagonals = sqrt (7 / 8 - sqrt 2 / 2) :=
sorry

end radius_of_sphere_correct_l495_495118


namespace convert_neg_300_deg_to_rad_l495_495236

theorem convert_neg_300_deg_to_rad :
  -300 * (Real.pi / 180) = - (5 / 3) * Real.pi :=
by
  sorry

end convert_neg_300_deg_to_rad_l495_495236


namespace log_m_n_condition_l495_495342

open Real

theorem log_m_n_condition {M N : ℝ} 
  (H1 : log M N = 2 * log N M)
  (H2 : M ≠ N)
  (H3 : M * N > 0)
  (H4 : M ≠ 1)
  (H5 : N ≠ 1) : 
  1 < M * N ∧ M * N < 2 := 
by 
  sorry

end log_m_n_condition_l495_495342


namespace sphere_surface_area_of_given_volume_l495_495089

-- Definition of the problem conditions
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * π * r^3

def surface_area_of_sphere (r : ℝ) : ℝ := 4 * π * r^2

-- Statement of the theorem to be proved
theorem sphere_surface_area_of_given_volume :
  (∃ (r : ℝ), volume_of_sphere r = 72 * π ∧ surface_area_of_sphere r = 36 * π * 2 ^ (2 / 3)) :=
sorry

end sphere_surface_area_of_given_volume_l495_495089


namespace min_magnitude_bound_l495_495024

open Complex

theorem min_magnitude_bound (z1 z2 z3 ω1 ω2 : ℂ) (h1 : ∥z1∥ ≤ 1)
    (h2 : ∥z2∥ ≤ 1) (h3 : ∥z3∥ ≤ 1)
    (h_eq : ∀ z : ℂ, (z - z1) * (z - z2) + (z - z2) * (z - z3) + (z - z3) * (z - z1) = 
                        (z - ω1) * (z - ω2)) :
    (∀ j, j ∈ {1, 2, 3} → min (abs (z1 - ω1)) (abs (z1 - ω2)) ≤ 1 ∧
                        min (abs (z2 - ω1)) (abs (z2 - ω2)) ≤ 1 ∧
                        min (abs (z3 - ω1)) (abs (z3 - ω2)) ≤ 1) :=
by
  sorry

end min_magnitude_bound_l495_495024


namespace percentage_increase_first_year_l495_495821

-- Define the original price P and the percentage increase x during the first year
variable (P : ℝ) (x : ℝ)

-- Define the conditions given in the problem
-- The price increased by x% in the first year
def first_year_increased_price : ℝ := P * (1 + x / 100)

-- The price decreased by 15% in the second year
def second_year_price : ℝ := (first_year_increased_price P x) * 0.85

-- The price at the end of the 2-year period was 110.5% of the original price
def final_price_after_two_years : ℝ := P * (110.5 / 100)

-- The theorem that we need to prove
theorem percentage_increase_first_year :
  (1 + x / 100) * 0.85 = 110.5 / 100 → x = 30 := 
by
  -- proof placeholder
  sorry

end percentage_increase_first_year_l495_495821


namespace meryll_questions_l495_495416

theorem meryll_questions :
  let total_mc := 35
  let total_ps := 15
  let written_mc := (2/5 : ℝ) * total_mc
  let written_ps := (1/3 : ℝ) * total_ps
  let remaining_mc := total_mc - written_mc
  let remaining_ps := total_ps - written_ps
  remaining_mc + remaining_ps = 31 :=
by
  let total_mc := 35
  let total_ps := 15
  let written_mc := (2/5 : ℝ) * total_mc
  let written_ps := (1/3 : ℝ) * total_ps
  let remaining_mc := total_mc - written_mc
  let remaining_ps := total_ps - written_ps
  have h1 : remaining_mc = 21 := by sorry
  have h2 : remaining_ps = 10 := by sorry
  show remaining_mc + remaining_ps = 31 from by sorry

end meryll_questions_l495_495416


namespace paint_needed_l495_495670

theorem paint_needed (h₁ : 1) (h₆ : 6) (pint_paint : ℝ) (similar_statues : Prop)
  (count_statues : ℕ) (height_small_statue : ℕ) : pint_paint = 120 :=
by
  -- conditions
  have condition1 : h₁ = 1 := by sorry,
  have condition2 : h₆ = 6 := by sorry,
  have condition3 : count_statues = 1080 := by sorry,
  have condition4 : height_small_statue = 2 := by sorry,
  have condition5 : similar_statues = true := by sorry,
  -- prove the required pints of paint
  sorry

end paint_needed_l495_495670


namespace sum_of_palindromic_primes_l495_495786

def is_palindromic_prime (n : ℕ) : Prop :=
  n < 100 ∧ 10 ≤ n ∧ n.Prime ∧ let reversed := (n % 10) * 10 + (n / 10) in 
  reversed.Prime

theorem sum_of_palindromic_primes : 
  ∑ p in List.filter is_palindromic_prime (List.range 100), p = 429 :=
by
  sorry

end sum_of_palindromic_primes_l495_495786


namespace length_BI_is_correct_l495_495011

noncomputable def incenter (A B C : Type*) [has_dist A] [has_dist B] [has_dist C] [inhabited A] [inhabited B] [inhabited C] : Type* :=
sorry -- definition of incenter

open_locale classical

variables {A B C I : Type*}
variables [inhabited A] [inhabited B] [inhabited C] [has_dist A] [has_dist B] [has_dist C]

-- Definitions of an isosceles triangle
def is_isosceles (a b c : ℝ) : Prop := a = b

def triangle_incenter_length (a b c : ℝ) (AB AC BC : ℝ) : ℝ :=
let s := (AB + AC + BC) / 2 in
let A := real.sqrt (s * (s - AB) * (s - AC) * (s - BC)) in
let r := A / s in
real.sqrt (r^2 + (BC / 2)^2)

-- Given condition, define problem variables and the proof statement
def triangle_ABC_length_BI : ℝ :=
triangle_incenter_length 6 6 8 6 6 8

-- The main theorem statement
theorem length_BI_is_correct : triangle_ABC_length_BI = 2 * real.sqrt 5 :=
sorry

end length_BI_is_correct_l495_495011


namespace union_of_P_and_neg_RQ_l495_495652

noncomputable def R : Set ℝ := Set.univ

noncomputable def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

noncomputable def Q : Set ℝ := {x | -2 < x ∧ x < 2}

noncomputable def neg_RQ : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}

theorem union_of_P_and_neg_RQ : 
  P ∪ neg_RQ = {x | x ≤ -2 ∨ 1 ≤ x ∧ x ≤ 3} :=
by
  sorry

end union_of_P_and_neg_RQ_l495_495652


namespace ratio_of_areas_l495_495008

theorem ratio_of_areas (s : ℝ) (ABCD : Prop) (isosceles_right_triangles : Prop)
  (E F G H : ℝ × ℝ) :
  (area (square EFGH) / area (square ABCD) = 2) :=
begin
  -- Assume ABCD is a square
  assume h1 : square ABCD,
  -- Assume isosceles right triangles are placed on each side of the square
  assume h2 : isosceles_right_triangles,
  -- Assume E, F, G, H are centers of these triangles
  assume h3 : E = (s / 2, -s / 2) ∧ F = (s + s / 2, s / 2) ∧ G = (s / 2, s + s / 2) ∧ H = (-s / 2, s / 2),
  -- Now we state the ratio of the areas
  sorry
end

end ratio_of_areas_l495_495008


namespace pedestrians_closest_time_distance_l495_495836

noncomputable def pedestrian_distance (t : ℝ) : ℝ :=
  (30 - 3 * t)^2 + (40 - 6 * t)^2

theorem pedestrians_closest_time_distance :
  let t := 22 / 3 in
  pedestrian_distance t = 80 :=
by
  sorry

end pedestrians_closest_time_distance_l495_495836


namespace find_general_term_of_sequence_l495_495650

open Nat

def a (n : ℕ) : ℝ :=
  if h : n > 0 then
    let k := if n % 2 = 0 then n / 2 else (n + 1) / 2
    if n % 2 = 0 then
      1 / 24 + 1 / (3 * (16^k))
    else
      14 / (3 * (16^k)) - 1 / 24
  else
    0

theorem find_general_term_of_sequence 
  (a1 : ℝ) (h : a1 = 1 / 4) (h_rec : ∀ n : ℕ, n > 0 → a n + a (n + 1) = 5 / (4^(n + 1))) :
  ∀ n : ℕ, n > 0 →
    a n = 
    (if n % 2 = 0 then
      let k := n / 2
      1 / 24 + 1 / (3 * 16^k)
    else
      let k := (n + 1) / 2
      14 / (3 * 16^k) - 1 / 24)
:= sorry

end find_general_term_of_sequence_l495_495650


namespace conference_duration_is_960_l495_495154

-- The problem statement definition
def conference_sessions_duration_in_minutes (day1_hours : ℕ) (day1_minutes : ℕ) (day2_hours : ℕ) (day2_minutes : ℕ) : ℕ :=
  (day1_hours * 60 + day1_minutes) + (day2_hours * 60 + day2_minutes)

-- The theorem we want to prove given the above conditions
theorem conference_duration_is_960 :
  conference_sessions_duration_in_minutes 7 15 8 45 = 960 :=
by 
  -- The proof is omitted
  sorry

end conference_duration_is_960_l495_495154


namespace mateo_orange_bottles_is_1_l495_495731

def number_of_orange_bottles_mateo_has (mateo_orange : ℕ) : Prop :=
  let julios_orange_bottles := 4
  let julios_grape_bottles := 7
  let mateos_grape_bottles := 3
  let liters_per_bottle := 2
  let julios_total_liters := (julios_orange_bottles + julios_grape_bottles) * liters_per_bottle
  let mateos_grape_liters := mateos_grape_bottles * liters_per_bottle
  let mateos_total_liters := (mateo_orange * liters_per_bottle) + mateos_grape_liters
  let additional_liters_to_julio := 14
  julios_total_liters = mateos_total_liters + additional_liters_to_julio

/-
Prove that Mateo has exactly 1 bottle of orange soda (assuming the problem above)
-/
theorem mateo_orange_bottles_is_1 : number_of_orange_bottles_mateo_has 1 :=
sorry

end mateo_orange_bottles_is_1_l495_495731


namespace total_weight_proof_l495_495531

-- Definitions of the variables and conditions given in the problem
variable (M D C : ℕ)
variable (h1 : D + C = 60)  -- Daughter and grandchild together weigh 60 kg
variable (h2 : C = 1 / 5 * M)  -- Grandchild's weight is 1/5th of grandmother's weight
variable (h3 : D = 42)  -- Daughter's weight is 42 kg

-- The goal is to prove the total weight is 150 kg
theorem total_weight_proof (M D C : ℕ) (h1 : D + C = 60) (h2 : C = 1 / 5 * M) (h3 : D = 42) :
  M + D + C = 150 :=
by
  sorry

end total_weight_proof_l495_495531


namespace sphere_surface_area_of_given_volume_l495_495087

-- Definition of the problem conditions
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * π * r^3

def surface_area_of_sphere (r : ℝ) : ℝ := 4 * π * r^2

-- Statement of the theorem to be proved
theorem sphere_surface_area_of_given_volume :
  (∃ (r : ℝ), volume_of_sphere r = 72 * π ∧ surface_area_of_sphere r = 36 * π * 2 ^ (2 / 3)) :=
sorry

end sphere_surface_area_of_given_volume_l495_495087


namespace inequality_holds_for_all_real_l495_495439

theorem inequality_holds_for_all_real (x : ℝ) : x^2 + 6 * x + 8 ≥ -(x + 4) * (x + 6) :=
  sorry

end inequality_holds_for_all_real_l495_495439


namespace min_C2_D2_at_36_l495_495398

noncomputable def min_value_C2_D2 (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 2) (hz : 0 ≤ z ∧ z ≤ 3) : ℝ :=
  let C := (Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12))
  let D := (Real.sqrt (x + 1) + Real.sqrt (y + 2) + Real.sqrt (z + 3))
  C^2 - D^2

theorem min_C2_D2_at_36 (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 2) (hz : 0 ≤ z ∧ z ≤ 3) : 
  min_value_C2_D2 x y z hx hy hz = 36 :=
sorry

end min_C2_D2_at_36_l495_495398


namespace circles_intersect_line_l495_495835

theorem circles_intersect_line (m c : ℝ)
  (hA : (1 : ℝ) - 3 + c = 0)
  (hB : 1 = -(m - 1) / (-4)) :
  m + c = -1 :=
by
  sorry

end circles_intersect_line_l495_495835


namespace ellipse_area_50_pi_l495_495710

noncomputable def semi_major_axis_length (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) / 2

theorem ellipse_area_50_pi :
  let a := semi_major_axis_length (-5) 2 15 2,
      b := 5,
      center := ((-5 + 15) / 2, (2 + 2) / 2)
  in
  (5 - center.1) ^ 2 / (a ^ 2) + (6 - center.2) ^ 2 / (b ^ 2) = 0.64 ∧
  a = 10 →
  let area := Real.pi * a * b in
  area = 50 * Real.pi :=
by {
  let a := semi_major_axis_length (-5) 2 15 2,
  let b := 5,
  let center := ((-5 + 15) / 2, (2 + 2) / 2),
  have h : (5 - center.1) ^ 2 / (a ^ 2) + (6 - center.2) ^ 2 / (b ^ 2) = 0.64, sorry,
  have ha : a = 10, sorry,
  let area := Real.pi * a * b,
  show area = 50 * Real.pi, sorry
}

end ellipse_area_50_pi_l495_495710


namespace triple_H_2_l495_495922

def H (x : ℝ) : ℝ := 2 * x - 1

theorem triple_H_2 : H(H(H(2))) = 9 :=
by
  -- Prove that H(H(H(2))) = 9 given H(x) = 2x - 1
  sorry

end triple_H_2_l495_495922


namespace students_arrangement_l495_495563

def num_students := 5
def num_females := 2
def num_males := 3
def female_A_cannot_end := true
def only_two_males_next_to_each_other := true

theorem students_arrangement (h1: num_students = 5)
                             (h2: num_females = 2)
                             (h3: num_males = 3)
                             (h4: female_A_cannot_end = true)
                             (h5: only_two_males_next_to_each_other = true) :
    ∃ n, n = 48 :=
by
  sorry

end students_arrangement_l495_495563


namespace divisible_by_2017_l495_495288

-- Define the condition
def alternating_harmonic_series (m n : ℕ) (h : Nat.gcd m n = 1) : ℚ :=
  ∑ i in Finset.range 1344 + 1, (-1) ^ (i + 1) * (1 / i : ℚ)

-- State the theorem to be proved
theorem divisible_by_2017 (m n : ℕ) (h : Nat.gcd m n = 1) :
  ∃ m n, alternating_harmonic_series m n h = (m : ℚ) / n ∧ 2017 ∣ m :=
sorry

end divisible_by_2017_l495_495288


namespace discount_percentage_is_20_l495_495530

-- Define the initial conditions
def purchase_price : ℝ := 60
def markup_percentage : ℝ := 0.25
def gross_profit : ℝ := 4

-- Define the selling price variable
def selling_price (purchase_price : ℝ) (markup_percentage : ℝ) : ℝ :=
  let S := purchase_price / (1 - markup_percentage)
  S

-- Define the discount percentage variable
def discount_percentage (selling_price gross_profit purchase_price : ℝ) : ℝ :=
  let D := (selling_price - (purchase_price + gross_profit))
  (D / selling_price) * 100

-- The proof problem statement
theorem discount_percentage_is_20 :
  discount_percentage (selling_price purchase_price markup_percentage) gross_profit purchase_price = 20 :=
by
  sorry

end discount_percentage_is_20_l495_495530


namespace shooter_variance_l495_495211

def scores : List ℝ := [9.7, 9.9, 10.1, 10.2, 10.1] -- Defining the scores

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length -- Calculating the mean

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length -- Defining the variance

theorem shooter_variance :
  variance scores = 0.032 :=
by
  sorry -- Proof to be provided later

end shooter_variance_l495_495211


namespace sum_geometric_series_is_correct_l495_495865

def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_geometric_series_is_correct
  (a r : ℚ) (n : ℕ)
  (h_a : a = 1/4)
  (h_r : r = 1/4)
  (h_n : n = 5) :
  geometric_series_sum a r n = 341 / 1024 :=
by
  rw [h_a, h_r, h_n]
  -- Now we can skip the proof.
  sorry

end sum_geometric_series_is_correct_l495_495865


namespace converse_tangency_l495_495795

-- Given conditions
variables (O : Point) (R : ℝ) (P : Plane) (M : Point)
variable [EuclideanGeometry]

-- Condition: P is a plane that intersects the sphere at exactly one point M
def plane_tangent_sphere (P : Plane) (O : Point) (R : ℝ) (M : Point) : Prop :=
  (on_plane P M) ∧ (distance O M = R) ∧
  (∀ (Q : Point), Q ∉ {M} → on_plane P Q → distance O Q > R)

-- Proof goal: The distance from O to the plane P is equal to R
theorem converse_tangency (O : Point) (R : ℝ) (P : Plane) (M : Point) 
  [EuclideanGeometry] (h : plane_tangent_sphere P O R M) :
  distance_point_plane O P = R :=
sorry

end converse_tangency_l495_495795


namespace triangle_inequalities_l495_495722

variable {a b c : ℝ} (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0)
def s := (a + b + c) / 2
def t_a := sorry -- definition involves elaborate calculation itself.
def t_b := sorry 
def t_c := sorry
def m_a := real.sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4)
def m_b := real.sqrt ((2 * a^2 + 2 * c^2 - b^2) / 4)
def m_c := real.sqrt ((2 * a^2 + 2 * b^2 - c^2) / 4)

theorem triangle_inequalities 
    (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0)
    : t_a^2 + t_b^2 + t_c^2 ≤ s^2 ∧ s^2 ≤ m_a^2 + m_b^2 + m_c^2 :=
sorry

end triangle_inequalities_l495_495722


namespace sum_of_palindromic_primes_less_than_100_l495_495780

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m > 1 ∧ m * m ≤ n → n % m ≠ 0

def is_palindromic_prime (n : ℕ) : Prop := 
  is_prime n ∧ 
  n / 10 = (n % 10) * 10 + n / 10 ∧ 
  is_prime (n % 10 * 10 + n / 10)

theorem sum_of_palindromic_primes_less_than_100 : 
  let palindromic_primes := [11, 13, 17, 31, 37, 71, 73, 79, 97] in
  palindromic_primes.sum = 429 := 
  by 
  sorry

end sum_of_palindromic_primes_less_than_100_l495_495780


namespace range_of_y_l495_495634

open Real

noncomputable def y (x : ℝ) : ℝ := (sin x)^2 - 3 * (sin x) + 2

theorem range_of_y : 
  ∃ (a b : ℝ), (∀ x ∈ Icc 0 (2 * π), a ≤ y x ∧ y x ≤ b) ∧ (a = 0) ∧ (b = 6) :=
by
  sorry

end range_of_y_l495_495634


namespace validate_true_proposition_l495_495558

theorem validate_true_proposition :
  let A := ∃ x ∈ Ioo 0 real.pi, real.sin x = real.tan x in
  let B := ¬ ∀ x : ℝ, x^2 + x + 1 > 0 → ∃ x0 : ℝ, x0^2 + x0 + 1 ≤ 0 in
  let C := ∀ θ : ℝ, ¬ (∀ x : ℝ, real.sin (2 * x + θ) = real.sin (-(2 * x + θ))) in
  let D := ∀ A B C : ℝ, A + B + C = real.pi ∧ (real.sin A + real.sin B = real.cos A + real.cos B ↔ C = real.pi/2) in
  D :=
by
  intros A B C
  sorry

end validate_true_proposition_l495_495558


namespace circle_through_point_has_same_center_l495_495263

theorem circle_through_point_has_same_center (M : ℝ × ℝ) (C : ℝ × ℝ)
  (hM : M = (-1, 1)) (hC : C = (2, -3))
  (center_condition : (x - C.1)^2 + (y - C.2)^2 = 16) :
  ∃ (r : ℝ), (x - C.1)^2 + (y - C.2)^2 = r^2 ∧ r = 5 := 
begin
  sorry
end

end circle_through_point_has_same_center_l495_495263


namespace right_angled_triangle_l495_495130

-- Define the lengths of the sides of the triangle
def a : ℕ := 3
def b : ℕ := 4
def c : ℕ := 5

-- The theorem to prove that these lengths form a right-angled triangle
theorem right_angled_triangle : a^2 + b^2 = c^2 :=
by
  sorry

end right_angled_triangle_l495_495130


namespace coprime_divisors_property_l495_495254

theorem coprime_divisors_property (n : ℕ) 
  (h : ∀ a b : ℕ, a ∣ n → b ∣ n → gcd a b = 1 → (a + b - 1) ∣ n) : 
  (∃ k : ℕ, ∃ p : ℕ, Nat.Prime p ∧ n = p ^ k) ∨ (n = 12) :=
sorry

end coprime_divisors_property_l495_495254


namespace solution_l495_495409

theorem solution (A B C : ℚ) (h1 : A + B = 10) (h2 : 2 * A = 3 * B + 5) (h3 : A * B * C = 120) :
  A = 7 ∧ B = 3 ∧ C = 40 / 7 := by
  sorry

end solution_l495_495409


namespace solution_smallest_n_eq_10_l495_495399

noncomputable def smallest_possible_value_n : ℕ :=
  Inf {n : ℕ |
    ∃ (x : Fin n → ℝ), 
      (∀ i : Fin n, 0 ≤ x i ∧ |x i| ≤ 1) ∧ 
      (∑ i, |x i| = 10 + |∑ i, x i|)
  }

theorem solution_smallest_n_eq_10 : smallest_possible_value_n = 10 :=
  sorry

end solution_smallest_n_eq_10_l495_495399


namespace companyKW_percent_of_combined_assets_l495_495947

noncomputable def ax := A_x : ℝ  -- Assets of Company x
noncomputable def ay := A_y : ℝ  -- Assets of Company y
noncomputable def az := A_z : ℝ  -- Assets of Company z

def companyKW_price (ax : ℝ) : ℝ := 1.6 * ax -- Price of Company KW in terms of x's assets
def company_x_with_debt (ax : ℝ) : ℝ := ax * 0.8 -- Company x's assets after debt
def company_y_with_stocks (ay : ℝ) : ℝ := ay * 1.3 -- Company y's assets with stocks
def company_z_grown_assets (az : ℝ) : ℝ := az * 1.1 -- Company z's assets after growth

def combined_assets (ax ay az : ℝ) : ℝ := 
  company_x_with_debt ax + company_y_with_stocks ay + company_z_grown_assets az

def kw_percent_of_combined (kw_price combined : ℝ) : ℝ :=
  (kw_price / combined) * 100

theorem companyKW_percent_of_combined_assets :
  (ax = 1.25 * ay) →
  (az = 0.5 * ay) →
  kw_percent_of_combined (companyKW_price ax) (combined_assets ax ay az) ≈ 70.18 :=
by
  intros
  sorry

end companyKW_percent_of_combined_assets_l495_495947


namespace conic_is_ellipse_l495_495499

-- Define the two foci points.
structure Point where
  x : ℝ
  y : ℝ

def focus1 : Point := { x := 0, y := 2 }
def focus2 : Point := { x := 6, y := -4 }

-- Define the given equation as a predicate on (x, y).
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y - 2)^2) + Real.sqrt ((x - 6)^2 + (y + 4)^2) = 12

-- Define the distance between the two foci.
def distance_between_foci : ℝ :=
  Real.sqrt ((focus2.x - focus1.x)^2 + (focus2.y - focus1.y)^2)

-- Define the non-degenerate condition.
def non_degenerate_ellipse_condition : Prop :=
  12 > distance_between_foci

-- Prove that the given equation represents an ellipse.
theorem conic_is_ellipse (x y : ℝ) :
  conic_equation x y → non_degenerate_ellipse_condition → True := 
by
  -- It suffices to assume the proof follows from the provided conditions.
  sorry

end conic_is_ellipse_l495_495499


namespace clock_correction_time_l495_495525

theorem clock_correction_time :
  let time_loss_per_day : ℝ := 15 / 60
  let days_elapsed : ℝ := 9 + 6 / 24
  let total_time_loss : ℝ := (15 / 1440) * (days_elapsed * 24)
  let correction : ℝ := total_time_loss * 60
  correction = 138.75 :=
by
  let time_loss_per_day : ℝ := 15 / 60
  let days_elapsed : ℝ := 9 + 6 / 24
  let total_time_loss : ℝ := (15 / 1440) * (days_elapsed * 24)
  let correction : ℝ := total_time_loss * 60
  have : correction = 138.75 := sorry
  exact this

end clock_correction_time_l495_495525


namespace points_per_win_l495_495806

theorem points_per_win (W : ℕ) (total_games wins losses draws points : ℕ) 
  (h_total_games : total_games = 20) 
  (h_wins : wins = 14) 
  (h_losses : losses = 2) 
  (h_draws : draws = total_games - (wins + losses)) 
  (h_points_for_draw : ∀ x, x = 1) 
  (h_points : points = 46) :
  points = (W * wins) + (draws * 1) → W = 3 :=
by
  intros h
  have : draws = 4, from by 
    rw [h_total_games, h_wins, h_losses]
    norm_num
  have : points = 14 * W + 4, from by 
    rw [←h_draws, h_points_for_draw, h_points]
    norm_num
  rw [this, h] at h_points
  sorry

end points_per_win_l495_495806


namespace mowing_time_approximately_correct_l495_495420

noncomputable def timeToMowLawn 
  (length width : ℝ) -- dimensions of the lawn in feet
  (swath overlap : ℝ) -- swath width and overlap in inches
  (speed : ℝ) : ℝ :=  -- walking speed in feet per hour
  (length * (width / ((swath - overlap) / 12))) / speed

theorem mowing_time_approximately_correct
  (h_length : ∀ (length : ℝ), length = 100)
  (h_width : ∀ (width : ℝ), width = 120)
  (h_swath : ∀ (swath : ℝ), swath = 30)
  (h_overlap : ∀ (overlap : ℝ), overlap = 6)
  (h_speed : ∀ (speed : ℝ), speed = 4500) :
  abs (timeToMowLawn 100 120 30 6 4500 - 1.33) < 0.01 := -- assert the answer is approximately 1.33 with a tolerance
by
  intros
  have length := h_length 100
  have width := h_width 120
  have swath := h_swath 30
  have overlap := h_overlap 6
  have speed := h_speed 4500
  rw [length, width, swath, overlap, speed]
  simp [timeToMowLawn]
  sorry

end mowing_time_approximately_correct_l495_495420


namespace determine_xyz_l495_495964

theorem determine_xyz (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x + 1/y = 5) (h5 : y + 1/z = 2) (h6 : z + 1/x = 8/3) :
  x * y * z = 8 + 3 * Real.sqrt 7 :=
by
  sorry

end determine_xyz_l495_495964


namespace find_number_l495_495139

def exceeding_condition (x : ℝ) : Prop :=
  x = 0.16 * x + 84

theorem find_number : ∃ x : ℝ, exceeding_condition x ∧ x = 100 :=
by
  -- Proof goes here, currently omitted.
  sorry

end find_number_l495_495139


namespace like_terms_m_n_sum_l495_495299

theorem like_terms_m_n_sum :
  ∃ (m n : ℕ), (2 : ℤ) * x ^ (3 * n) * y ^ (m + 4) = (-3 : ℤ) * x ^ 9 * y ^ (2 * n) ∧ m + n = 5 :=
by 
  sorry

end like_terms_m_n_sum_l495_495299


namespace magnitude_of_z_l495_495753

noncomputable def z_satisfies_eq (z : ℂ) : Prop := z + 9 = 10 * conj z + 22 * complex.I

theorem magnitude_of_z : ∃ z : ℂ, z_satisfies_eq z ∧ complex.abs z = real.sqrt 5 :=
by
  sorry

end magnitude_of_z_l495_495753


namespace madame_marten_behavior_l495_495515

theorem madame_marten_behavior :
  let P_good := 1 / 2,
      P_bad := 1 / 2,
      P_umbrella := 2 / 3,
      P_umbrella_good := 1 / 2,
      P_umbrella_bad := 5 / 6
  in
  (P_umbrella = P_umbrella_good * P_good + P_umbrella_bad * P_bad) :=
by
  let P_good := 1 / 2
  let P_bad := 1 / 2
  let P_umbrella := 2 / 3
  let P_umbrella_good := 1 / 2
  let P_umbrella_bad := 5 / 6
  -- Given equation to prove
  have h : P_umbrella = P_umbrella_good * P_good + P_umbrella_bad * P_bad
  -- Assigned variables with known values
  have h1 : 2 / 3 = 1 / 2 * 1 / 2 + P_umbrella_bad * 1 / 2 := sorry
  from sorry

end madame_marten_behavior_l495_495515


namespace classroom_needs_more_money_l495_495540

theorem classroom_needs_more_money 
    (goal : ℕ) 
    (raised_from_two_families : ℕ) 
    (raised_from_eight_families : ℕ) 
    (raised_from_ten_families : ℕ) 
    (H : goal = 200) 
    (H1 : raised_from_two_families = 2 * 20) 
    (H2 : raised_from_eight_families = 8 * 10) 
    (H3 : raised_from_ten_families = 10 * 5) 
    (total_raised : ℕ := raised_from_two_families + raised_from_eight_families + raised_from_ten_families) : 
    (goal - total_raised) = 30 := 
by 
  sorry

end classroom_needs_more_money_l495_495540


namespace curve_consists_of_two_lines_curve_is_hyperbola_with_asymptotes_l495_495641

variables {m n : ℝ} (hx : m = 0) (hy : n > 0) (hz : m * n < 0)

theorem curve_consists_of_two_lines : mx^2 + ny^2 = 1 → (x^2 + y^2 = 1 / n) :=
sorry

theorem curve_is_hyperbola_with_asymptotes : mx^2 + ny^2 = 1 → (m > 0) ∧ (n < 0) ∧ (∀ x y, y = ± sqrt (-m / n) * x) :=
sorry

end curve_consists_of_two_lines_curve_is_hyperbola_with_asymptotes_l495_495641


namespace quadratic_roots_sign_l495_495396

theorem quadratic_roots_sign (p q : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x * y = q ∧ x + y = -p) ↔ q < 0 :=
sorry

end quadratic_roots_sign_l495_495396


namespace sum_of_palindromic_primes_less_than_100_l495_495776

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m > 1 ∧ m * m ≤ n → n % m ≠ 0

def is_palindromic_prime (n : ℕ) : Prop := 
  is_prime n ∧ 
  n / 10 = (n % 10) * 10 + n / 10 ∧ 
  is_prime (n % 10 * 10 + n / 10)

theorem sum_of_palindromic_primes_less_than_100 : 
  let palindromic_primes := [11, 13, 17, 31, 37, 71, 73, 79, 97] in
  palindromic_primes.sum = 429 := 
  by 
  sorry

end sum_of_palindromic_primes_less_than_100_l495_495776


namespace greatest_radius_l495_495687

theorem greatest_radius (r : ℕ) : (π * (r : ℝ)^2 < 75 * π) ↔ r ≤ 8 := 
by
  sorry

end greatest_radius_l495_495687


namespace max_radius_of_circle_l495_495690

theorem max_radius_of_circle (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
by
  sorry

end max_radius_of_circle_l495_495690


namespace find_sum_of_squares_l495_495391

variables {Point : Type*} [inner_product_space ℝ Point]

noncomputable def centroid (P Q R : Point) : Point :=
  (P + Q + R) / 3

theorem find_sum_of_squares
  (P Q R : Point)
  (G := centroid P Q R)
  (h : dist G P ^ 2 + dist G Q ^ 2 + dist G R ^ 2 = 72) :
  dist P Q ^ 2 + dist P R ^ 2 + dist Q R ^ 2 = 216 :=
sorry

end find_sum_of_squares_l495_495391


namespace total_arrangements_l495_495151

-- Question: 
-- Given 6 teachers and 4 schools with specific constraints, 
-- prove that the number of different ways to arrange the teachers is 240.

def teachers : List Char := ['A', 'B', 'C', 'D', 'E', 'F']

def schools : List Nat := [1, 2, 3, 4]

def B_and_D_in_same_school (assignment: Char → Nat) : Prop :=
  assignment 'B' = assignment 'D'

def each_school_has_at_least_one_teacher (assignment: Char → Nat) : Prop :=
  ∀ s ∈ schools, ∃ t ∈ teachers, assignment t = s

noncomputable def num_arrangements : Nat := sorry -- This would actually involve complex combinatorial calculations

theorem total_arrangements : num_arrangements = 240 :=
  sorry

end total_arrangements_l495_495151


namespace xiao_wang_program_output_l495_495886

theorem xiao_wang_program_output (n : ℕ) (h : n = 8) : (n : ℝ) / (n^2 + 1) = 8 / 65 := by
  sorry

end xiao_wang_program_output_l495_495886


namespace custom_factorial_minus_product_l495_495015

noncomputable def custom_mul (a b : ℕ) : ℕ := a * b + a + b

noncomputable def custom_product (l : List ℕ) : ℕ :=
  l.foldl custom_mul 1

theorem custom_factorial_minus_product :
  10.factorial - custom_product [2, 3, 4, 5, 6, 7, 8, 9] = 1 :=
by
  sorry

end custom_factorial_minus_product_l495_495015


namespace distance_BC_is_7n_l495_495711

noncomputable def distance_between_islands (n : ℕ) : ℕ :=
  let AB := 8 * n
  let AC := 5 * n
  let angleBAC := 60
  let cos60 := 1 / 2
  let BC_squared := AB^2 + AC^2 - 2 * AB * AC * cos60
  let BC := Nat.sqrt BC_squared
  BC

theorem distance_BC_is_7n (n : ℕ) : distance_between_islands n = 7 * n := by
  -- Definitions
  let AB := 8 * n
  let AC := 5 * n
  let angleBAC := 60
  let cos60 := 1 / 2

  -- Applying Law of Cosines
  let BC_squared : ℕ := AB^2 + AC^2 - 2 * AB * AC * cos60
  have h1 : BC_squared = 49 * n^2 := by sorry
  have h2 : Nat.sqrt BC_squared = 7 * n := by sorry

  show distance_between_islands n = 7 * n from h2

end distance_BC_is_7n_l495_495711


namespace sum_of_basic_terms_divisible_by_4_l495_495512

theorem sum_of_basic_terms_divisible_by_4 (n : ℕ) (h : n ≤ 4) 
  (a : Fin n → Fin n → ℤ) (h_val : ∀ i j, a i j = 1 ∨ a i j = -1) :
  ∃ k : ℤ, (∑ σ : Equiv.Perm (Fin n), ∏ i, a i (σ i)) = 4 * k :=
by
  sorry

end sum_of_basic_terms_divisible_by_4_l495_495512


namespace max_parallelepipeds_in_cube_l495_495841

-- Definitions of dimensions and conditions
def cube_side_length : ℕ := 6
def parallelepiped_size : List ℕ := [1, 1, 4]

-- The total volume of the cube
def volume_cube (n : ℕ) : ℕ := n^3

-- The total volume of the parallelepiped
def volume_parallelepiped (sizes : List ℕ) : ℕ := sizes.foldl (*) 1

-- The proof statement that the maximum number of parallelepipeds fitting inside the cube is 52
theorem max_parallelepipeds_in_cube : 
  let max_parallelepipeds := 52 in
  (volume_cube cube_side_length) / (volume_parallelepiped parallelepiped_size) = max_parallelepipeds :=
by {
  sorry
}

end max_parallelepipeds_in_cube_l495_495841


namespace compute_p_q_l495_495014

theorem compute_p_q :
  ∃ p q : ℤ, 
    (∃ x : ℝ, (∃ a b : ℝ, (a * a * a = x) 
    ∧ (b * b * b = 20 - x) 
    ∧ (a + b = 2)
    ∧ (x = (p : ℝ) - real.sqrt (q : ℝ))
    ∧ (p + q = 118))) :=
sorry

end compute_p_q_l495_495014


namespace sum_of_palindromic_primes_l495_495790

def is_palindromic_prime (n : ℕ) : Prop :=
  n < 100 ∧ 10 ≤ n ∧ n.Prime ∧ let reversed := (n % 10) * 10 + (n / 10) in 
  reversed.Prime

theorem sum_of_palindromic_primes : 
  ∑ p in List.filter is_palindromic_prime (List.range 100), p = 429 :=
by
  sorry

end sum_of_palindromic_primes_l495_495790


namespace correct_statement_l495_495931

-- Definitions of the given conditions and correct answer in Lean
variables {a b : ℝ}
variables {m : ℝ}
variables {α : ℝ}
variables {f : ℝ → ℝ}
variables {a b : ℝ}

-- Definitions of the propositions
def ln_increasing (a b : ℝ) (h : 0 < b) : Prop := a > b → Real.log a > Real.log b
def vectors_perpendicular (m : ℝ) : Prop := (1 * m) + (m * (2 * m - 1)) = 0
def contrapositive_first_quadrant (α : ℝ) : Prop := ¬(0 < α ∧ α < pi / 2) → ((-315 * (pi/180) = α) ∧ (α > 0 ∧ α < 2 * pi))

-- A function continuous and product < 0
def function_continuous_product (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ContinuousOn f (Set.Icc a b) ∧ f a * f b < 0 →

-- The contrapositive we need to verify if it's false
def contrapositive_false (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  function_continuous_product  f a b → (∃ x, a < x ∧ x < b ∧ f x = 0) → f a * f b < 0

theorem correct_statement : option D →
  ¬(ln_increasing a b) ∧ ¬(vectors_perpendicular m) ∧ ¬(contrapositive_first_quadrant α) ∧ contrapositive_false f a b :=
sorry

end correct_statement_l495_495931


namespace find_k_l495_495891

theorem find_k (k : ℝ) 
  (h : (1 / 2) ^ 23 * (1 / 81) ^ k = (1 / 18) ^ 23) :
  k = 11.5 :=
sorry

end find_k_l495_495891


namespace circumcenter_locus_of_triangle_MBN_l495_495215

-- Define the problem structure
variable {A B C P M N O G: Point} -- Declare the variables as abstract points in the plane

-- Conditions of the problem
axiom hABC : EquilateralTriangle A B C
axiom hP : (InteriorPoint P A B C) ∧ (Angle A P C = 120)
axiom hM : LineIntersect A B (Ray P C) M
axiom hN : LineIntersect B C (Ray P A) N

-- Definition of the problem statement
theorem circumcenter_locus_of_triangle_MBN:
  locus_of_circumcenter_triangle M B N = perp_bisector_segment B G
  sorry

end circumcenter_locus_of_triangle_MBN_l495_495215


namespace smallest_positive_integer_congruence_l495_495120

theorem smallest_positive_integer_congruence : ∃ x : ℕ, 0 < x ∧ x < 31 ∧ (5 * x ≡ 17 [MOD 31]) ∧ (∀ y : ℕ, 0 < y ∧ y < 31 ∧ (5 * y ≡ 17 [MOD 31]) → x ≤ y) :=
by {
  let x := 25,
  use x,
  split,
  -- prove 0 < x
  exact Nat.zero_lt_succ 24,
  split,
  -- prove x < 31
  exact Nat.lt_of_sub_lt_sub_right (Nat.zero_lt_succ 5),
  split,
  -- prove 5 * x ≡ 17 [MOD 31]
  have h1 : 5 * 25 = 125, from rfl,
  have h2 : 125 % 31 = 2, from rfl,
  have h3 : 17 % 31 = 17, from rfl,
  have h4 : 17 % 31 = 6, from rfl,
  rwa [← h1, ← h3] at h2,
  intro y,
  assume h_pos : 0 < y,
  assume h_lt : y < 31,
  assume h_congr : 5 * y ≡ 17 [MOD 31],

  
  -- prove x is smallest
  sorry
}

end smallest_positive_integer_congruence_l495_495120


namespace ratio_of_segments_l495_495370

-- Let's define the problem setup and conditions
variable {A B C D M N E : Type} -- points
variable {h : ℝ} -- height of the trapezoid 

-- defining segment lengths and the diameters
def is_trapezoid (A B C D : Type) : Prop := sorry
def diameter (BD AC : Type) (M N : Type) : Prop := sorry
def height (PQ : Type) (h : ℝ) : Prop := sorry

-- defining the segments
def segment_DP (P D : Type) (h : ℝ) : Prop := sorry
def segment_AQ (A Q : Type) (h : ℝ) : Prop := sorry

-- Lean statement of the proof problem
theorem ratio_of_segments (h : ℝ) (A B C D M N E : Type)
  (H1 : is_trapezoid A B C D)
  (H2 : diameter (A D) (B C))
  (H3 : diameter (B D) (A C))
  (H4 : height (P Q) h)
  (H5 : segment_DP P D (h / real.sqrt 3))
  (H6 : segment_AQ A Q h) :
  AE / ED = real.sqrt 3 :=
sorry

end ratio_of_segments_l495_495370


namespace closest_fraction_l495_495565

def total_medals := 120
def won_medals := 25

theorem closest_fraction : abs ((won_medals / total_medals) - (1/5)) < min (abs((won_medals / total_medals) - (1/4))) 
     (min (abs((won_medals / total_medals) - (1/6))) (min (abs((won_medals / total_medals) - (1/7))) 
     (abs((won_medals / total_medals) - (1/8))))) := by
  sorry

end closest_fraction_l495_495565


namespace find_sum_l495_495135

theorem find_sum 
  (R : ℝ) -- Original interest rate
  (P : ℝ) -- Principal amount
  (h: (P * (R + 3) * 3 / 100) = ((P * R * 3 / 100) + 81)): 
  P = 900 :=
sorry

end find_sum_l495_495135


namespace square_of_complex_l495_495674

theorem square_of_complex (z : ℂ) (h : z = complex.mk 2 5) : z^2 = complex.mk (-21) 20 :=
by
  intro h
  rw [h]
  calc
    complex.mk 2 5 * complex.mk 2 5 = complex.mk 4 20 + complex.mk 0 (-25) := by simp
    ... = complex.mk (-21) 20 := by simp

end square_of_complex_l495_495674


namespace possible_numbers_in_top_cell_l495_495818

-- Defining the set of numbers
def numbers : Set ℕ := {1, 2, 4, 5, 8, 10}

-- Defining the sum of the columns
def column_sum := 10

-- Proving that given these conditions, the numbers in the top cell are one of {1, 4, 5}
theorem possible_numbers_in_top_cell : 
  ∀ (columns : List (List ℕ)),
  (∀ col ∈ columns, (∑ n in col, n) = column_sum) →
  (column (1 3)).length = 1 → -- one of the columns is a single cell
  (columns.flatten.to_set = numbers) →
  exists (x ∈ {1, 4, 5} ), (x ∈ (columns[2].head)) :=
begin
  sorry
end

end possible_numbers_in_top_cell_l495_495818


namespace coplanarity_conditions_l495_495929

variables (OM OA OB OC : Vector) (x y z : ℝ)

theorem coplanarity_conditions :
  ¬ (OM = 2 • OA - OB - OC ∧ x + y + z = 1) ∧
  ¬ (OM = (1 / 5) • OA + (1 / 3) • OB + (1 / 2) • OC ∧ x + y + z = 1) ∧
  ¬ (OM + OA + OB + OC = (0 : Vector) ∧ x + y + z = 1) :=
sorry

end coplanarity_conditions_l495_495929


namespace correct_statements_l495_495643

noncomputable def f (x : ℝ) : ℝ := real.sqrt (4 - (x - 2) ^ 2)
variables (x_1 x_2 : ℝ)
variables (h1 : 2 < x_1) (h2 : x_1 < x_2) (h3 : x_2 < 4)

theorem correct_statements :
  ((x_2 * f x_1 > x_1 * f x_2) ∧ ((x_2 - x_1) * (f x_2 - f x_1) < 0)) :=
sorry

end correct_statements_l495_495643


namespace sum_geometric_series_is_correct_l495_495869

def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_geometric_series_is_correct
  (a r : ℚ) (n : ℕ)
  (h_a : a = 1/4)
  (h_r : r = 1/4)
  (h_n : n = 5) :
  geometric_series_sum a r n = 341 / 1024 :=
by
  rw [h_a, h_r, h_n]
  -- Now we can skip the proof.
  sorry

end sum_geometric_series_is_correct_l495_495869


namespace inverse_function_correct_l495_495073

def f (x : ℝ) : ℝ := 2^x + 3

def finv (x : ℝ) : ℝ := log x / log 2

theorem inverse_function_correct (x : ℝ) (h : x > 3) : f (finv (x - 3)) = x :=
by
  sorry

end inverse_function_correct_l495_495073


namespace geom_series_sum_l495_495851

theorem geom_series_sum : 
  let a₀ := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 5 in
  ∑ i in Finset.range n, a₀ * r ^ i = 341 / 1024 := 
  by
    sorry

end geom_series_sum_l495_495851


namespace absolute_value_positive_l495_495880

theorem absolute_value_positive (a : ℝ) (h : a ≠ 0) : |a| > 0 := by
  sorry

end absolute_value_positive_l495_495880


namespace intersection_complement_of_B_l495_495359

open Set

variables {U : Type*} [TopologicalSpace U]

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

def B : Set ℝ := {x | 0 < x ∧ x < 1}

def C_U_B : Set ℝ := ∁ B

theorem intersection_complement_of_B :
  A ∩ C_U_B = {x : ℝ | (-2 ≤ x ∧ x ≤ 0) ∨ (1 ≤ x ∧ x ≤ 2)} :=
by
  sorry

end intersection_complement_of_B_l495_495359


namespace jane_books_l495_495383

theorem jane_books (b : ℕ) (h_cond : 3000 ≤ 11 * b) : ∃ k : ℕ, k = 273 :=
by {
  let b_min := (3000 / 11 : ℝ).ceil.to_nat,  -- Using ceiling function to get the smallest integer >= 3000/11
  have : b_min = 273,  -- We know from the solution that 3000 / 11 ≈ 272.727... which rounds up to 273
  sorry
}

end jane_books_l495_495383


namespace inverse_matrix_sum_zero_l495_495648

open Matrix

theorem inverse_matrix_sum_zero (a b c d : ℚ) :
  (matrix.mul (λ i j => ([5, 1], [7, 3] : (Fin 2 × Fin 2) → ℚ) i j)
              (λ i j => ([a, b], [c, d] : (Fin 2 × Fin 2) → ℚ)  i j)
              (1 : Fin 2)
              (0 : Fin 2))
              = (1 : Fin 2 → Fin 2 → ℚ)
#if  
  let inverse : 
###
∑ 

where 

decided eq valgleaorably decidable
(
text → function :
  lemma matrix_basis_eqg std_cai
)
#
noncomputable def 
inverse_matrix (a b c d : ℚ) :
  matrix.mul (λ i j => ([5, 1], [7, 3] : (Fin 2 × Fin 2) → ℚ) i j)
                       (λ i j => ([a, b], [c, d] : (Fin 2 × Fin 2) → ℚ) i j) =
              (1 : Fin 2 × Fin 2 → ℚ) := ℕ

matrix.eq_dec_arith_exists:
matrixeigen_proof_eq
in 
noncomputable
.

lemma matrix_values_sum (a, b, c, d : matrix.mul ) 
 ∈  [matrixeigen_mul_sequence, 
  
])

:=   
where 
definitions lean_types real_nums;

aux : 
(ℚ).
assert := [class EQ]
∀ axiom
lemma_upperbound
(c,d)*[term_class of 2]≡ max_triv_eq_arithmetic base → 

reach_dl val_bound lemma_eq
matrix_summable
Fin 2× Fin { Fin_le



%\[ (matrix.unit_inverses )
%(\forall ) 
matrix_type_sum :=
(%[lean_macro_unfold _ 
matrix_def,

([|5, 1|], [7, 3] matrix inverse_eq)]
 in  [ fin_descriptive_sum 

\]

basis [: base := series 


ψ∆ (i j)
tppd_eq 
mat (2, SY)= 
LEANCLASS
%\Ψ[unit_inverse_sum )

#eqproftaxa :class
  summable terms_circ matrix_val_rep :=
  (\sum (=%[(≥ all_bounds ]) \[is_dec \(Fin_square∣ℕ) f
    
lemma_eqbase_parts0
---(lean_macro sum_eq syntax ≡consr)
(\ψ)matrix_lemma 

alignment ≡partial
  
  eql_verif = [
 (reduce_all inv_mul)⊎ ℤ 
ℕ lemma_prooftype_eq 
 (%[ ∀ term_class > 2 ) :

matrix.descri_mat_lean
basis (Eq_lemma simp  ∀ lean_eigen_sum_basis := \leantheom approx..≤ )}

lemma_eq_to (ιetric ) :




lemma sum_column : 

let sum [→ {≡

fin.matrix_sum_class [ψ lean.reduc_simplex) :
final (\square )

*



   a + b + c + d = 0 :=]{lean.begin}

[lean.base_eq_calc_sum(essmt_class_fin)] :=
 [2class_bound_sum] ***ψ
%[∀ matrixotta.)([summation_eq)]) end
endsmt
new 


--{[assert opdecide_matrix )
end lemma 
lemma_solvable{:
fin.zero.sum}lemma_metaclass lean_endproblems_eq_by_END·
wellfoundations
lemma endpts lean.eq.decidebound
*@%[lemma_eq*:fin 2 sum :=
<<fin.eq_sum -- analytical_eq'=>"lean.eq_decidable == axioms]> ]}

end := \lean Fin_sum_bounds ∃ final_eq cơ sum_boundary_nin_lemma.

}%[matrix_proof_sum_eqbounded_eq_push

end inverse_matrix_sum_zero_l495_495648


namespace g_four_times_of_three_l495_495752

noncomputable def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else 4 * x - 1

theorem g_four_times_of_three :
  g (g (g (g 3))) = 3 := by
  sorry

end g_four_times_of_three_l495_495752


namespace time_to_cut_kids_hair_l495_495725

noncomputable def time_women : ℕ := 50
noncomputable def time_men : ℕ := 15
noncomputable def num_women : ℕ := 3
noncomputable def num_men : ℕ := 2
noncomputable def num_kids : ℕ := 3
noncomputable def total_time : ℕ := 255

theorem time_to_cut_kids_hair (time_kid : ℕ) :
  time_women * num_women + time_men * num_men + time_kid * num_kids = total_time →
  time_kid = 25 :=
begin
  intros h,
  sorry
end

end time_to_cut_kids_hair_l495_495725


namespace cube_root_equality_l495_495871

/-- Given condition: c is defined as the sum of four terms each being 5^7. -/
def c : ℝ := 5^7 + 5^7 + 5^7 + 5^7

/-- Main theorem: The cube root of c equals 100 times the cube root of 10. -/
theorem cube_root_equality : real.cbrt c = 100 * real.cbrt 10 := by
  sorry

end cube_root_equality_l495_495871


namespace highest_value_of_a_l495_495265

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

def highest_a : Nat :=
  7

theorem highest_value_of_a (a : Nat) 
  (last_three_digits := a * 100 + 53)
  (number := 4 * 10^8 + 3 * 10^7 + 7 * 10^6 + 5 * 10^5 + 2 * 10^4 + a * 10^3 + 5 * 10^2 + 3 * 10^1 + 9) :
  (∃ a, last_three_digits % 8 = 0 ∧ sum_of_digits number % 9 = 0 ∧ number % 12 = 0 ∧ a <= 9) → a = highest_a :=
by
  intros
  sorry

end highest_value_of_a_l495_495265


namespace greatest_common_divisor_90_m_l495_495479

theorem greatest_common_divisor_90_m {m : ℕ} (h1 : ∃ p : ℕ, Nat.Prime p ∧ 90 = 2 * 3^2 * 5) (h2 : ∀ d : ℕ, d ∣ 90 ↔ d ∣ m) :
  let gcd := Nat.gcd 90 m in
  gcd = 9 := 
by
  sorry

end greatest_common_divisor_90_m_l495_495479


namespace similar_triangles_x_value_l495_495539

theorem similar_triangles_x_value :
  ∀ (x : ℝ), right_triangle_similar_legs 12 9 x 7 → x = 9.33 := by
sorry

-- Definitions to support the theorem
def right_triangle_similar_legs (a1 a2 : ℝ) (b1 b2 : ℝ) : Prop :=
  ∀ (k : ℝ), k ≠ 0 ∧ a1 = k * b1 ∧ a2 = (9/7) * b2

end similar_triangles_x_value_l495_495539


namespace number_of_triangles_with_interior_vertices_l495_495063

theorem number_of_triangles_with_interior_vertices (h : ∀ (p : Set ℝ) (hk : p.finite ∧ p.card = 10),
  ∀ (x ∈ p) ∀ (y ∈ p), x ≠ y → (exists! z ∈ Int, 
  ∀ (a b ∈ p), a ≠ b → a ≠ x ∧ a ≠ y ∧ b ≠ x ∧ b ≠ y → ∃ (c : ℕ), p.pairwise (λ (i j : ℝ), is_chord c i j) ∧ 
  ∀ (i j k : ℝ), is_intersection_inside i j k) → 
  finset.card (set_of (λ t : set (set ℝ), finset.card t = 3)) = 4200 :=
sorry

end number_of_triangles_with_interior_vertices_l495_495063


namespace how_many_cards_per_week_l495_495384

theorem how_many_cards_per_week 
  (x : ℕ)
  (c1 : nat = 20)
  (c2 : ∀ n, n ≤ 4 → nat = n * x)
  (c3 : ∀ n, n % 2 = 0 → n = (n + 4x - 4))
  (c4 : nat = 40) 
  :
  x = 6 :=
sorry

end how_many_cards_per_week_l495_495384


namespace integer_count_between_l495_495328

theorem integer_count_between (a b : ℝ) (ha : a = (10.5)^3) (hb : b = (10.6)^3) :
  (b.floor - a.ceil + 1 = 33) :=
by
  have h1 : a = 1157.625 := by rw [ha]; norm_num
  have h2 : b = 1191.016 := by rw [hb]; norm_num
  sorry

end integer_count_between_l495_495328


namespace complex_abs_bound_l495_495625

theorem complex_abs_bound (a : ℝ) (z : ℂ) (ha : 0 < a ∧ a < 2) (hz : z = a + 1 * complex.I) :
  1 < complex.abs z ∧ complex.abs z < real.sqrt 5 :=
sorry

end complex_abs_bound_l495_495625


namespace smallest_digit_to_correct_l495_495457

def incorrect_sum : ℕ := 2104
def correct_sum : ℕ := 738 + 625 + 841
def difference : ℕ := correct_sum - incorrect_sum

theorem smallest_digit_to_correct (d : ℕ) (h : difference = 100) :
  d = 6 := 
sorry

end smallest_digit_to_correct_l495_495457


namespace proof_total_costs_l495_495828

def cost_company_a (distance: ℝ) : ℝ :=
  let first_segment_cost := 2.10
  let remaining_distance := distance - 1/5
  let remaining_segment_cost := remaining_distance * 5 * 0.40
  first_segment_cost + remaining_segment_cost

def cost_company_b (distance: ℝ) : ℝ :=
  let first_segment_cost := 3.00
  let remaining_distance := distance - 1/4
  let remaining_segment_cost := remaining_distance * 4 * 0.50
  first_segment_cost + remaining_segment_cost

def cost_company_c (distance: ℝ) : ℝ :=
  let cost_per_mile := 1.50
  let pickup_fee := 2.00
  (distance * cost_per_mile) + pickup_fee

theorem proof_total_costs (distance: ℝ) (h: distance = 8) : 
  cost_company_a distance = 17.70 ∧ 
  cost_company_b distance = 18.50 ∧ 
  cost_company_c distance = 14.00 := 
by
  sorry

end proof_total_costs_l495_495828


namespace simplest_common_denominator_l495_495823

theorem simplest_common_denominator (x y : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (d : ℤ), d = x^2 * y^2 ∧ ∀ (a b : ℤ), 
    (∃ (k : ℤ), a = k * (x^2 * y)) ∧ (∃ (m : ℤ), b = m * (x * y^2)) → d = lcm a b :=
by
  sorry

end simplest_common_denominator_l495_495823


namespace functional_equation_satisfies_conditions_l495_495959

def f : ℝ → ℝ
| x := if x ≤ 1 then 0 else 1 - 1/x

theorem functional_equation_satisfies_conditions :
  (f 1 = 0) ∧
  (∀ x, x > 1 → f x > 0) ∧
  (∀ x y, (0 ≤ x ∧ 0 ≤ y ∧ x + y > 0) → f (x * f y) * f y = f (x * y / (x + y))) :=
by
  -- Proof will be inserted here
  sorry

end functional_equation_satisfies_conditions_l495_495959


namespace radius_of_circle_l495_495210

-- Define the properties and given values
def right_angle_triangle (a b c : ℝ) : Prop := a * a + b * b = c * c
def area_of_triangle (a b : ℝ) : ℝ := (1 / 2) * a * b
def area_of_circle (r : ℝ) : ℝ := π * r * r

-- Given the sides of the triangle
def a := 6.0
def b := 8.0
def c := 10.0

-- Main theorem to prove
theorem radius_of_circle : ∃ r : ℝ, right_angle_triangle a b c 
                        ∧ area_of_triangle a b = 24
                        ∧ area_of_circle r = 24
                        ∧ abs (r - 2.8) < 0.1 :=
 by
  sorry

end radius_of_circle_l495_495210


namespace largest_sum_at_vertex_l495_495161

def cube_faces := {1, 2, 3, 4, 5, 6}

def opposite_sum := 8

-- Define opposite pairs according to the problem's conditions.
def opposite_pairs := [(1, 6), (2, 5), (3, 4)]

noncomputable def largest_sum_adjacent_faces : ℕ := 11

theorem largest_sum_at_vertex (faces : set ℕ) (opposite_faces : list (ℕ × ℕ)) (sum_opposite : ℕ) :
  faces = cube_faces →
  opposite_faces = opposite_pairs →
  sum_opposite = opposite_sum →
  ∃ v1 v2 v3 ∈ faces, 11 = v1 + v2 + v3 ∧
  (∀ (op1 op2 : ℕ), (op1, op2) ∈ opposite_faces → op1 ≠ v1 ∨ op2 ≠ v1) ∧
  (∀ (op1 op2 : ℕ), (op1, op2) ∈ opposite_faces → op1 ≠ v2 ∨ op2 ≠ v2) ∧
  (∀ (op1 op2 : ℕ), (op1, op2) ∈ opposite_faces → op1 ≠ v3 ∨ op2 ≠ v3) :=
by sorry

end largest_sum_at_vertex_l495_495161


namespace distinct_constructions_l495_495536

/-- The problem sets up conditions for constructing a rectangular prism and asks for the number of distinct constructions, modulo rotations. -/
def number_of_distinct_constructions : ℕ :=
  let total_configurations := 116 in
  total_configurations
  
theorem distinct_constructions : number_of_distinct_constructions = 116 := 
by
  sorry

end distinct_constructions_l495_495536


namespace rectangle_area_l495_495203

theorem rectangle_area (w l : ℝ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end rectangle_area_l495_495203


namespace nth_term_arithmetic_sequence_l495_495070

theorem nth_term_arithmetic_sequence (a : ℤ) (d : ℤ) (a1 a2 a3 : ℤ)
  (h1 : a1 = a - 1)
  (h2 : a2 = a + 1)
  (h3 : a3 = 2 * a + 3)
  (h_d : d = a2 - a1) :
  ∃ a_n : ℕ → ℤ, (a_n n = 2 * n - 3) :=
by 
  have : d = 2, from sorry,
  have : a = 0, from sorry,
  let a_n := λ n : ℕ, 2 * n - 3,
  use a_n,
  intro n,
  exact sorry

end nth_term_arithmetic_sequence_l495_495070


namespace sheila_earning_per_hour_l495_495897

def sheila_hours_per_day_mwf : ℕ := 8
def sheila_days_mwf : ℕ := 3
def sheila_hours_per_day_tt : ℕ := 6
def sheila_days_tt : ℕ := 2
def sheila_total_earnings : ℕ := 432

theorem sheila_earning_per_hour : (sheila_total_earnings / (sheila_hours_per_day_mwf * sheila_days_mwf + sheila_hours_per_day_tt * sheila_days_tt)) = 12 := by
  sorry

end sheila_earning_per_hour_l495_495897


namespace probability_of_code_A_in_seventh_week_l495_495099

-- Define the four distinct codes.
inductive Code
| A | B | C | D

open Code

-- Define the transition matrix.
def transition_matrix : Matrix (Fin 4) (Fin 4) ℚ :=
  ![
    [ 0, 1/3, 1/3, 1/3 ],
    [ 1/3, 0, 1/3, 1/3 ],
    [ 1/3, 1/3, 0, 1/3 ],
    [ 1/3, 1/3, 1/3, 0 ]
  ]

-- Define the initial state vector representing the first week.
def initial_state : Vector ℚ (Fin 4) :=
  ![1, 0, 0, 0]

-- Define exponentiation of the transition matrix.
def matrix_pow (M : Matrix (Fin 4) (Fin 4) ℚ) (n : Nat) : Matrix (Fin 4) (Fin 4) ℚ :=
  M ^ n

-- Define the state vector after six weeks by multiplying the initial state with the sixth power of the transition matrix.
def state_vector_after_six_weeks : Vector ℚ (Fin 4) :=
  initial_state ⬝ (matrix_pow transition_matrix 6)

-- Define the probability that code A is used in the seventh week.
def probability_code_A_in_seventh_week : ℚ :=
  state_vector_after_six_weeks 0

-- The theorem to prove.
theorem probability_of_code_A_in_seventh_week : probability_code_A_in_seventh_week = 61/243 := by
  sorry

end probability_of_code_A_in_seventh_week_l495_495099


namespace proposition_1_proposition_3_l495_495633

variables {Line Plane : Type}
variables (m n : Line) (α β γ : Plane)

-- Condition predicates
def parallel (p q : Plane) : Prop := sorry -- parallelism of p and q
def perpendicular (p q : Plane) : Prop := sorry -- perpendicularly of p and q
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry -- parallelism of line and plane
def line_perpendicular_plane (l : Line) (p : Plane) : Prop := sorry -- perpendicularity of line and plane
def line_in_plane (l : Line) (p : Plane) : Prop := sorry -- line is in the plane

-- Proposition ①
theorem proposition_1 (h1 : parallel α β) (h2 : parallel α γ) : parallel β γ := sorry

-- Proposition ③
theorem proposition_3 (h1 : line_perpendicular_plane m α) (h2 : line_parallel_plane m β) : perpendicular α β := sorry

end proposition_1_proposition_3_l495_495633


namespace ab_range_l495_495012

theorem ab_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = a + b + 8) : a * b ≥ 16 :=
sorry

end ab_range_l495_495012


namespace remainder_of_number_of_towers_l495_495911

-- Define the conditions and constants
def cubes := {k : ℕ | 1 ≤ k ∧ k ≤ 10}

def can_place_on (a b : ℕ) : Prop := a ≤ b + 3

noncomputable def count_towers : ℕ :=
  -- Recursively compute the number of towers using the provided rules.
  let T : ℕ → ℕ
    | 1 => 1
    | 2 => 2
    | 3 => T 2 * 3
    | (m + 1) => if m ≥ 3 then T m * 4 else T m  in
  T 10

theorem remainder_of_number_of_towers :
  count_towers % 1000 = 304 :=
sorry

end remainder_of_number_of_towers_l495_495911


namespace sum_palindromic_primes_lt_100_eq_429_l495_495784

def is_palindromic_prime (n : ℕ) : Prop :=
  (n ≥ 10 ∧ n < 100) ∧
  (Prime n ∧ 
   let d1 := n / 10 in
   let d2 := n % 10 in
   Prime (d2 * 10 + d1))

theorem sum_palindromic_primes_lt_100_eq_429 : 
  (∑ n in Finset.filter is_palindromic_prime (Finset.range 100), n) = 429 :=
by sorry

end sum_palindromic_primes_lt_100_eq_429_l495_495784


namespace AL_LB_ratio_l495_495463

-- Definition of the problem conditions
variables (S : Type) [metric_space S] [normed_space ℝ S] -- S is a metric space and a normed vector space over ℝ
variables (circle S : set S) (O : S) (r : ℝ)
variable [circle O r] -- circle with center O and radius r
variables (A B C D E K L : S) -- Points on the circle

-- Conditions in the problem
variables (diameters_AB_CD_perpendicular : ∠ A O B = π / 2 ∧ ∠ C O D = π / 2)
variables (chord_EA_intersects_CD_at_K : intersect (chord E A) (diameter C D K))
variables (chord_EC_intersects_AB_at_L : intersect (chord E C) (diameter A B L))
variable (CK_KD_ratio : ratio (segment C K : segment K D) = 2 / 1)

-- Question: Prove that AL / LB = 3 / 1
theorem AL_LB_ratio :
  AL LB = 3 / 1 :=
sorry

end AL_LB_ratio_l495_495463


namespace shaded_region_volume_l495_495081

theorem shaded_region_volume :
  let r1 := 4   -- radius of the first cylinder
  let h1 := 2   -- height of the first cylinder
  let r2 := 1   -- radius of the second cylinder
  let h2 := 5   -- height of the second cylinder
  let V1 := π * r1^2 * h1 -- volume of the first cylinder
  let V2 := π * r2^2 * h2 -- volume of the second cylinder
  V1 + V2 = 37 * π :=
by
  sorry

end shaded_region_volume_l495_495081


namespace tan_neg440_eq_neg_sqrt_one_minus_m_sq_div_m_l495_495665

theorem tan_neg440_eq_neg_sqrt_one_minus_m_sq_div_m (m : ℝ) (h : Real.cos (80 * Real.pi / 180) = m) :
    Real.tan (-440 * Real.pi / 180) = - (Real.sqrt (1 - m^2) / m) :=
by
  -- proof goes here
  sorry

end tan_neg440_eq_neg_sqrt_one_minus_m_sq_div_m_l495_495665


namespace combined_transformation_matrix_l495_495963

open Matrix Real

noncomputable def R : Matrix (Fin 2) (Fin 2) ℝ := 
  !![
    Real.cos (135 * Real.pi / 180), -Real.sin (135 * Real.pi / 180),
    Real.sin (135 * Real.pi / 180),  Real.cos (135 * Real.pi / 180)
  ]

def F : Matrix (Fin 2) (Fin 2) ℝ := 
  !![
    1, 0,
    0, -1
  ]

theorem combined_transformation_matrix :
  let M := F ⬝ R in
  M = !![
    -1/Real.sqrt 2, -1/Real.sqrt 2,
    -1/Real.sqrt 2,  1/Real.sqrt 2
  ] :=
by {
  let M := F ⬝ R,
  sorry
}

end combined_transformation_matrix_l495_495963


namespace find_sum_due_l495_495460

variable (BD TD FV : ℝ)

-- given conditions
def condition_1 : Prop := BD = 80
def condition_2 : Prop := TD = 70
def condition_3 : Prop := BD = TD + (TD * BD / FV)

-- goal statement
theorem find_sum_due (h1 : condition_1 BD) (h2 : condition_2 TD) (h3 : condition_3 BD TD FV) : FV = 560 :=
by
  sorry

end find_sum_due_l495_495460


namespace roots_quadratic_identity_l495_495755

theorem roots_quadratic_identity :
  ∀ (r s : ℝ), (r^2 - 5 * r + 3 = 0) ∧ (s^2 - 5 * s + 3 = 0) → r^2 + s^2 = 19 :=
by
  intros r s h
  sorry

end roots_quadratic_identity_l495_495755


namespace tan_2theta_l495_495311

theorem tan_2theta (θ : ℝ) (h1 : ∃ (p : ℝ × ℝ), p.1 ≠ 0 ∧ p.2 = 2 * p.1) :
  tan (2 * θ) = -4 / 3 :=
by
  sorry

end tan_2theta_l495_495311


namespace valid_passwords_l495_495932

theorem valid_passwords (total_passwords restricted_passwords : Nat) 
  (h_total : total_passwords = 10^4)
  (h_restricted : restricted_passwords = 8) : 
  total_passwords - restricted_passwords = 9992 := by
  sorry

end valid_passwords_l495_495932


namespace water_level_drop_l495_495360

theorem water_level_drop :
  (∀ x : ℝ, x > 0 → (x = 4) → (x > 0 → x = 4)) →
  ∃ y : ℝ, y < 0 ∧ (y = -1) :=
by
  sorry

end water_level_drop_l495_495360


namespace smallest_positive_period_of_f_l495_495083

def f (x : ℝ) : ℝ := (sin (4 * x)) / (1 + cos (4 * x))

theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧
  (∀ t, (∀ x, f (x + t) = f x) → t ≥ T) ∧
  T = π / 2 := 
sorry

end smallest_positive_period_of_f_l495_495083


namespace sum_of_digits_N_l495_495532

noncomputable def sum_of_digits (n : ℕ) : ℕ := 
  if n = 0 then 0 
  else sum_of_digits (n / 10) + n % 10

def N : ℕ := 2 + (2 * 10^1 + 2) + (2 * 10^2 + 2) + (2 * 10^3 + 2) + ... + (2 * 10^19 + 2)

theorem sum_of_digits_N : sum_of_digits N = 42 := 
  sorry

end sum_of_digits_N_l495_495532


namespace equal_selection_probability_l495_495986

theorem equal_selection_probability :
  ∀ (total_students eliminated_students remaining_students selected_students : ℕ),
  total_students = 2004 →
  eliminated_students = 4 →
  remaining_students = total_students - eliminated_students →
  selected_students = 50 →
  (∀ student ∈ (finset.range remaining_students), 
   probability_of_selection student remaining_students selected_students = 
   (1 / selected_students : ℚ)) :=
by
  intros _ _ _ _ h_total h_elim h_remain h_selected
  sorry

end equal_selection_probability_l495_495986


namespace prime_square_minus_seven_l495_495339

theorem prime_square_minus_seven (P : ℕ) (hP_prime : Prime P) (h_prime_P3_plus_9 : Prime (P^3 + 9)) : P^2 - 7 = -3 :=
sorry

end prime_square_minus_seven_l495_495339


namespace cos_value_of_transformed_angle_l495_495989

theorem cos_value_of_transformed_angle (α : ℝ)
  (h : Real.sin ((π / 6) + α) = (√3 / 3)) :
  Real.cos ((10 * π / 3) - α) = - (√3 / 3) :=
by
  sorry

end cos_value_of_transformed_angle_l495_495989


namespace angle_BED_50_degrees_l495_495707

theorem angle_BED_50_degrees
  (A B C D E : Point)
  (ABC : Triangle A B C)
  (hA : angle_in_triangle ABC A = 60)
  (hC : angle_in_triangle ABC C = 70)
  (hD_on_AB : D ∈ segment A B)
  (hE_on_BC : E ∈ segment B C)
  (hDE_eq_BE : length D E = length E B) :
  angle_in_triangle (Triangle.mk D B E) (angle BED) = 50 :=
  sorry

end angle_BED_50_degrees_l495_495707


namespace ducks_drinking_l495_495903

theorem ducks_drinking (total_d : ℕ) (drank_before : ℕ) (drank_after : ℕ) :
  total_d = 20 → drank_before = 11 → drank_after = total_d - (drank_before + 1) → drank_after = 8 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end ducks_drinking_l495_495903


namespace february_1_is_sunday_if_february_14_is_saturday_l495_495671

-- Define the days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
deriving DecidableEq, Inhabited

open Day

-- Define February 14 as a Saturday
def February_14 := Saturday

-- Define a function to calculate the day of the week after a given number of days
def day_of_week_after (start_day : Day) (days_after : ℕ) : Day :=
  let days := [Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday]
  days[(days.indexOf start_day + days_after % 7) % 7]

-- Theorem statement: Prove that February 1 is a Sunday if February 14 is a Saturday
theorem february_1_is_sunday_if_february_14_is_saturday 
  (h : February_14 = Saturday) : day_of_week_after February_14 13 = Sunday :=
  sorry

end february_1_is_sunday_if_february_14_is_saturday_l495_495671


namespace sum_of_values_such_that_f_x_eq_1_l495_495027

def f : ℝ → ℝ := λ x =>
  if x ≤ 2 then
    x^2 - 4 * x + 3
  else
    (3 * x) / 4 + 1 / 2

theorem sum_of_values_such_that_f_x_eq_1 : ∑ x in {x : ℝ | f x = 1}.to_finset, x = 4 :=
by
  sorry

end sum_of_values_such_that_f_x_eq_1_l495_495027


namespace fixed_point_of_exponential_graph_l495_495517

theorem fixed_point_of_exponential_graph (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x y : ℝ), x = 1 ∧ y = 2 ∧ y = a^(x-1) + 1 := 
by 
  use 1
  use 2
  split
  { refl }
  split
  { refl }
  
  sorry

end fixed_point_of_exponential_graph_l495_495517


namespace line_circle_intersection_l495_495273

theorem line_circle_intersection (k : ℝ) :
  (∃ (x y : ℝ), y = k * x + 1 ∧ x ^ 2 + y ^ 2 = 2) ∧ ¬(∃ x y, y = k * x + 1 ∧ x = 0 ∧ y = 0) :=
by by
  sorry

end line_circle_intersection_l495_495273


namespace propositions_truth_count_l495_495603

theorem propositions_truth_count : ∃ (a b c : ℝ), 
  ((a > b) → (ac < bc) = false) ∧
  ((ac^2 > bc^2) → (a > b) = true) ∧
  ((a < b) ∧ (b < 0) → (a^2 > ab) ∧ (ab > b^2) = true) ∧
  ((a > b) ∧ (∀a > b, (1 / a) > (1 / b)) → (a > 0) ∧ (b < 0) = true) ∧
  (true_propositions_count = 3) :=
by
  sorry

end propositions_truth_count_l495_495603


namespace smallest_positive_integer_congruence_l495_495121

theorem smallest_positive_integer_congruence : ∃ x : ℕ, 0 < x ∧ x < 31 ∧ (5 * x ≡ 17 [MOD 31]) ∧ (∀ y : ℕ, 0 < y ∧ y < 31 ∧ (5 * y ≡ 17 [MOD 31]) → x ≤ y) :=
by {
  let x := 25,
  use x,
  split,
  -- prove 0 < x
  exact Nat.zero_lt_succ 24,
  split,
  -- prove x < 31
  exact Nat.lt_of_sub_lt_sub_right (Nat.zero_lt_succ 5),
  split,
  -- prove 5 * x ≡ 17 [MOD 31]
  have h1 : 5 * 25 = 125, from rfl,
  have h2 : 125 % 31 = 2, from rfl,
  have h3 : 17 % 31 = 17, from rfl,
  have h4 : 17 % 31 = 6, from rfl,
  rwa [← h1, ← h3] at h2,
  intro y,
  assume h_pos : 0 < y,
  assume h_lt : y < 31,
  assume h_congr : 5 * y ≡ 17 [MOD 31],

  
  -- prove x is smallest
  sorry
}

end smallest_positive_integer_congruence_l495_495121


namespace exists_function_f_l495_495424

theorem exists_function_f (a b : ℕ) (h1 : a > 1) (h2 : b > 1) :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f(a * f(n)) = b * n :=
sorry

end exists_function_f_l495_495424


namespace walkway_time_stopped_l495_495166

noncomputable def effective_speed_with_walkway (v_p v_w : ℝ) : ℝ := v_p + v_w
noncomputable def effective_speed_against_walkway (v_p v_w : ℝ) : ℝ := v_p - v_w

theorem walkway_time_stopped (v_p v_w : ℝ) (h1 : effective_speed_with_walkway v_p v_w = 2)
                            (h2 : effective_speed_against_walkway v_p v_w = 2 / 3) :
    (200 / v_p) = 150 :=
by sorry

end walkway_time_stopped_l495_495166


namespace find_price_max_profit_l495_495521

/-
Part 1: Prove the price per unit of type A and B
-/

def price_per_unit (x y : ℕ) : Prop :=
  (2 * x + 3 * y = 690) ∧ (x + 4 * y = 720)

theorem find_price :
  ∃ x y : ℕ, price_per_unit x y ∧ x = 120 ∧ y = 150 :=
by
  sorry

/-
Part 2: Prove the maximum profit with constraints
-/

def conditions (m : ℕ) : Prop :=
  m ≤ 3 * (40 - m) ∧ 120 * m + 150 * (40 - m) ≤ 5400

def profit (m : ℕ) : ℕ :=
  (160 - 120) * m + (200 - 150) * (40 - m)

theorem max_profit :
  ∃ m : ℕ, 20 ≤ m ∧ m ≤ 30 ∧ conditions m ∧ profit m = profit 20 :=
by
  sorry

end find_price_max_profit_l495_495521


namespace determine_length_AY_l495_495043

noncomputable def length_of_AY 
  (A B C D Y : Point) (circle_diameter : ℝ)
  (h1 : OnCircle A circle_diameter) 
  (h2 : OnCircle B circle_diameter) 
  (h3 : OnCircle C circle_diameter) 
  (h4 : OnCircle D circle_diameter)
  (h5 : Y ∈ diameter (A, D)) 
  (h6 : distance B Y = distance C Y) 
  (h7 : angle (A, B, C) = 12 * (π / 180)) 
  (h8 : angle (B, Y, C) = 36 * (π / 180)): ℝ :=
  sin (12 * π / 180) * sin (12 * π / 180) * (csc (18 * π / 180))

theorem determine_length_AY 
  (A B C D Y : Point) (circle_diameter : ℝ)
  (h1 : OnCircle A circle_diameter) 
  (h2 : OnCircle B circle_diameter) 
  (h3 : OnCircle C circle_diameter) 
  (h4 : OnCircle D circle_diameter)
  (h5 : Y ∈ diameter (A, D)) 
  (h6 : distance B Y = distance C Y) 
  (h7 : angle (A, B, C) = 12 * (π / 180)) 
  (h8 : angle (B, Y, C) = 36 * (π / 180)) :
  AY = length_of_AY A B C D Y circle_diameter h1 h2 h3 h4 h5 h6 h7 h8 :=
sorry

end determine_length_AY_l495_495043


namespace two_digit_number_l495_495554

noncomputable def numOfFactors (n : ℕ) : ℕ :=
  (List.range n).count (fun d => n % d = 0)

def A_statement (n : ℕ) : Prop :=
  even (numOfFactors n) ↔ n > 50

def B_statement (n : ℕ) : Prop :=
  (odd n ∨ n > 60)

def C_statement (n : ℕ) : Prop :=
  (even n ∨ n > 70)

theorem two_digit_number (n : ℕ) (h : A_statement n ∧ B_statement n ∧ C_statement n) : n = 64 :=
sorry

end two_digit_number_l495_495554


namespace rectangular_field_area_l495_495190

theorem rectangular_field_area :
  ∃ (w l : ℝ), (l = 3 * w) ∧ (2 * (l + w) = 72) ∧ (l * w = 243) :=
by {
  sorry
}

end rectangular_field_area_l495_495190


namespace sum_of_elements_in_A_l495_495468

open Set

def A : Set ℤ := {x : ℤ | x^2 + x - 2 ≤ 0}

theorem sum_of_elements_in_A : (∑ x in A, x) = -2 :=
by
  sorry

end sum_of_elements_in_A_l495_495468


namespace Three_Circles_Concur_l495_495045

open EuclideanGeometry

noncomputable def problem_statement : Prop :=
∀ (A B C D E F X Y Z P : Point), 
  (D ∈ line_segment B C) ∧ (E ∈ line_segment C A) ∧ (F ∈ line_segment A B) ∧
  line BE ≠ line CF → line BE = line CF ∩ X ∧
  line CF ≠ line AD → line CF = line AD ∩ Y ∧
  line AD ≠ line BE → line AD = line BE ∩ Z ∧
  (P ∈ circumcircle X B C) ∧ (P ∈ circumcircle Y C A) ∧ (P ∈ circumcircle Z A B) ∧
  P ≠ A ∧ P ≠ B → prove_cocyclic X B C Y C A Z A B P.

theorem Three_Circles_Concur : problem_statement := sorry

end Three_Circles_Concur_l495_495045


namespace area_of_rhombus_l495_495507

-- Definition of points in 2D Cartesian Coordinate system
structure Point2D where
  x : ℝ
  y : ℝ

def vertices : List Point2D :=
  [{ x := 0, y := 3.5 }, { x := 8, y := 0 }, { x := 0, y := -3.5 }, { x := -8, y := 0 }]

-- Calculate the distance between two points
def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

-- Diagonals' lengths (d1 and d2)
def d1 : ℝ := distance { x := 8, y := 0 } { x := -8, y := 0 }
def d2 : ℝ := distance { x := 0, y := 3.5 } { x := 0, y := -3.5 }

-- Proving the area of the rhombus
theorem area_of_rhombus : (d1 * d2) / 2 = 56 := by
  sorry

end area_of_rhombus_l495_495507


namespace problem_statement_l495_495666

-- Given conditions
variables {a b c d : ℝ}
hypothesis h1 : a * b = 1
hypothesis h2 : c + d = 0

-- The statement we need to prove
theorem problem_statement :
  (a * b)^4 - 3 * (c + d)^3 = 1 :=
by
  -- We state sorry here as we are not providing the actual proof
  sorry

end problem_statement_l495_495666


namespace find_sum_c_d_l495_495668

noncomputable def g (c d x : ℝ) := c * x + d
noncomputable def g_inv (d c x : ℝ) := d * x - 2 * c

theorem find_sum_c_d (c d : ℝ) (h1 : ∀ x, g c d (g_inv d c x) = x) :
  c + d = 3 * real.cbrt 2 :=
by sorry

end find_sum_c_d_l495_495668


namespace find_k_values_l495_495259

/-- 
Prove that the values of k such that the positive difference between the 
roots of 3x^2 + 5x + k = 0 equals the sum of the squares of the roots 
are exactly (70 + 10sqrt(33))/8 and (70 - 10sqrt(33))/8.
-/
theorem find_k_values (k : ℝ) :
  (∀ (a b : ℝ), (3 * a^2 + 5 * a + k = 0 ∧ 3 * b^2 + 5 * b + k = 0 ∧ |a - b| = a^2 + b^2))
  ↔ (k = (70 + 10 * Real.sqrt 33) / 8 ∨ k = (70 - 10 * Real.sqrt 33) / 8) :=
sorry

end find_k_values_l495_495259


namespace find_C_value_find_b_value_l495_495380

noncomputable def angles_as_sides_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
  a/b = sin A / sin B ∧ b/c = sin B / sin C ∧ c/a = sin C / sin A

theorem find_C_value {A B C a b c : ℝ}
  (h1 : angles_as_sides_triangle a b c A B C)
  (h2 : (cos A / a) + (cos B / b) = (2 * c * cos C) / (a * b)) :
  C = π / 3 :=
sorry

theorem find_b_value
  (a : ℝ) (c : ℝ) (h : c = sqrt 5) 
  (C : ℝ) (hC : C = π/3) :
  b = 1 + sqrt 2 :=
sorry

end find_C_value_find_b_value_l495_495380


namespace greatest_int_radius_lt_75pi_l495_495699

noncomputable def circle_radius_max (A : ℝ) (π : ℝ) : ℕ :=
  Nat.floor (Real.sqrt (A / π))

theorem greatest_int_radius_lt_75pi :
  circle_radius_max 75 Real.pi = 8 := by
  sorry

end greatest_int_radius_lt_75pi_l495_495699


namespace cats_sold_l495_495535

theorem cats_sold (siamese : ℕ) (house : ℕ) (left : ℕ) (total_initial : siamese = 19 ∧ house = 45 ∧ left = 8) : siamese + house - left = 56 :=
by
  rcases total_initial with ⟨h₁, h₂, h₃⟩
  rw [h₁, h₂, h₃]
  simp
  sorry

end cats_sold_l495_495535


namespace find_c_l495_495300

theorem find_c (c : ℝ) (h : ∃ β : ℝ, (5 + β = -c) ∧ (5 * β = 45)) : c = -14 := 
  sorry

end find_c_l495_495300


namespace max_radius_of_circle_l495_495691

theorem max_radius_of_circle (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
by
  sorry

end max_radius_of_circle_l495_495691


namespace probability_sum_5_of_two_dice_rolls_l495_495550

theorem probability_sum_5_of_two_dice_rolls : 
  let die_faces := ({1, 2, 3, 4, 5, 6} : set ℕ),
      trials := 2
  in ∑ x in die_faces, ∑ y in die_faces, (if (x + y = 5) then 1 else 0) / (die_faces.card ^ trials) = 1 / 9 := 
by
  -- Definitions of a die with 6 faces and rolling it twice.
  let die_faces := ({1, 2, 3, 4, 5, 6} : set ℕ)
  let trials := 2

  /- The number of possible outcomes is die_faces.card ^ trials -/
  let total_outcomes := die_faces.card ^ trials
  
  /- Sum over all outcomes; only include those where the sum is 5 -/
  let favorable_sum := ∑ x in die_faces, ∑ y in die_faces, if (x + y = 5) then 1 else 0

  /- The required probability is the ratio of favorable outcomes to total outcomes -/
  have h_favorable : favorable_sum = 4 := sorry
  have h_total : total_outcomes = 36 := sorry

  /- Now we compute the probability -/
  have h_probability : (favorable_sum : ℚ) / total_outcomes = 1 / 9 := sorry

  exact h_probability

end probability_sum_5_of_two_dice_rolls_l495_495550


namespace area_inside_circle_outside_triangle_correct_l495_495935

noncomputable def area_inside_circle_outside_triangle : ℝ :=
  let radius := 1 / 2
  let side_length := 1
  let area_circle := π * radius^2
  let height_triangle := (√3 / 2) * side_length
  let area_triangle := (1 / 2) * side_length * height_triangle
  area_circle - area_triangle

theorem area_inside_circle_outside_triangle_correct :
  area_inside_circle_outside_triangle = (π / 4 - √3 / 4) :=
by
  unfold area_inside_circle_outside_triangle
  sorry

end area_inside_circle_outside_triangle_correct_l495_495935


namespace bob_winning_strategy_l495_495217

-- Define the game conditions
def board_size : ℕ := 2020
def total_cells : ℕ := board_size * board_size

-- Define the points calculation function
def points (row_colored : ℕ) (col_colored : ℕ) : ℕ :=
  row_colored + col_colored

-- Define the symmetry move
def symmetric_cell (i j : ℕ) : (ℕ × ℕ) :=
  (i, board_size - j + 1)

-- Define the main theorem
theorem bob_winning_strategy :
  ∃ diff : ℕ, diff = 2040200 ∧ (∀ moves : list (ℕ × ℕ),
    (∀ move ∈ moves, move.1 < board_size ∧ move.2 < board_size) →
    (∀ i, i < moves.length → (if i % 2 = 0 then (moves.nth i).is_some ∧ moves.nth i = some (i / board_size, i % board_size)
                             else  ∃ j, j < board_size ∧ (moves.nth (i-1)).is_some ∧ moves.nth (i-1) = some (j, i / board_size))))
→ max_difference moves = 2040200 :=
sorry

end bob_winning_strategy_l495_495217


namespace arithmetic_sequence_n_2005_l495_495302

/-- Define an arithmetic sequence with first term a₁ = 1 and common difference d = 3. -/
def arithmetic_sequence (n : ℕ) : ℤ := 1 + (n - 1) * 3

/-- Statement of the proof problem. -/
theorem arithmetic_sequence_n_2005 : 
  ∃ n : ℕ, arithmetic_sequence n = 2005 ∧ n = 669 := 
sorry

end arithmetic_sequence_n_2005_l495_495302


namespace problem_equation_ellipse_l495_495390

noncomputable def ellipse_c : Prop :=
  ∃ (a b : ℝ) (A : ℝ × ℝ), a > b ∧ b > 0 ∧ a = 2 ∧ A = (1, 3 / 2) ∧ 
  (2 * a = 4) ∧ 
  ((1 / a^2) + ((3 / 2)^2 / b^2) = 1) ∧ 
  (b^2 = 3) ∧ 
  (a = 2) ∧ c^2 = 1 ∧ 
  (∀ (F1 F2 : ℝ × ℝ), 
    F1 = (-1, 0) ∧ 
    F2 = (1, 0) → 
    (∀ (P : ℝ × ℝ), 
      P = (2 * x + 1, 2 * y) ∧ 
      ((x + 1 / 2)^2 + (4 * y^2 / 3) = 1)))

theorem problem_equation_ellipse :
  ellipse_c → 
  (∃ (x y : ℝ), (x / 4 + y / 3 = 1) ∧ (∀ (M : ℝ × ℝ), M = (x + 1 / 2, y) → ((x + 1 / 2)^2 + 4 * y^2 / 3 = 1))) :=
sorry

end problem_equation_ellipse_l495_495390


namespace proof_problem_l495_495282

noncomputable def problem_statement (a : Real) : Prop :=
  tan a = 3 →
  (sin (a - π) + cos (π - a)) / (sin (π / 2 - a) + cos (π / 2 + a)) = 2

theorem proof_problem (a : Real) : problem_statement a := sorry

end proof_problem_l495_495282


namespace class_mean_correct_l495_495709

noncomputable def new_class_mean (number_students_midterm : ℕ) (avg_score_midterm : ℚ)
                                 (number_students_next_day : ℕ) (avg_score_next_day : ℚ)
                                 (number_students_final_day : ℕ) (avg_score_final_day : ℚ)
                                 (total_students : ℕ) : ℚ :=
  let total_score_midterm := number_students_midterm * avg_score_midterm
  let total_score_next_day := number_students_next_day * avg_score_next_day
  let total_score_final_day := number_students_final_day * avg_score_final_day
  let total_score := total_score_midterm + total_score_next_day + total_score_final_day
  total_score / total_students

theorem class_mean_correct :
  new_class_mean 50 65 8 85 2 55 60 = 67 :=
by
  sorry

end class_mean_correct_l495_495709


namespace max_t_squared_value_l495_495900

noncomputable def max_t_squared (R : ℝ) : ℝ :=
  let PR_QR_sq_sum := 4 * R^2
  let max_PR_QR_prod := 2 * R^2
  PR_QR_sq_sum + 2 * max_PR_QR_prod

theorem max_t_squared_value (R : ℝ) : max_t_squared R = 8 * R^2 :=
  sorry

end max_t_squared_value_l495_495900


namespace count_magical_integers_l495_495923

noncomputable def is_magical (n : ℕ) : Prop :=
  (⌊real.sqrt (⌈real.sqrt n⌉)⌋ = ⌈real.sqrt (⌊real.sqrt n⌋)⌉)

theorem count_magical_integers :
  (set.indicator (set.Icc 1 10000) (λ n, if is_magical n then 1 else 0)).sum = 1330 :=
by
  sorry

end count_magical_integers_l495_495923


namespace solve_problem_l495_495589

def has_only_prime_divisors_2_5 (N : ℕ) : Prop := ∀ p, nat.prime p → p ∣ N → (p = 2 ∨ p = 5)

theorem solve_problem (N : ℕ) :
  has_only_prime_divisors_2_5 N →
  ∃ (M : ℕ), (N + 25 = M * M) →
  (N = 200 ∨ N = 2000) :=
by
  intros h₁ h₂
  sorry

end solve_problem_l495_495589


namespace rectangular_field_area_l495_495192

theorem rectangular_field_area :
  ∃ (w l : ℝ), (l = 3 * w) ∧ (2 * (l + w) = 72) ∧ (l * w = 243) :=
by {
  sorry
}

end rectangular_field_area_l495_495192


namespace rectangular_field_area_l495_495172

theorem rectangular_field_area :
  ∃ (w l : ℝ), w = l / 3 ∧ 2 * (w + l) = 72 ∧ w * l = 243 :=
by
  sorry

end rectangular_field_area_l495_495172


namespace geometric_series_sum_l495_495856

theorem geometric_series_sum
  (a r : ℚ) (n : ℕ)
  (ha : a = 1/4)
  (hr : r = 1/4)
  (hn : n = 5) :
  (∑ i in finset.range n, a * r^i) = 341 / 1024 := by
  sorry

end geometric_series_sum_l495_495856


namespace area_of_rectangle_l495_495186

theorem area_of_rectangle (w l : ℝ) (h₁ : w = l / 3) (h₂ : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end area_of_rectangle_l495_495186


namespace probability_neither_event_l495_495077

-- Definitions of given probabilities
def P_soccer_match : ℚ := 5 / 8
def P_science_test : ℚ := 1 / 4

-- Calculations of the complements
def P_no_soccer_match : ℚ := 1 - P_soccer_match
def P_no_science_test : ℚ := 1 - P_science_test

-- Independence of events implies the probability of neither event is the product of their complements
theorem probability_neither_event :
  (P_no_soccer_match * P_no_science_test) = 9 / 32 :=
by
  sorry

end probability_neither_event_l495_495077


namespace factorization_l495_495970

variable (b : ℕ)

theorem factorization : 145 * b^2 + 29 * b = 29 * b * (5 * b + 1) := by sorry

end factorization_l495_495970


namespace probability_of_winning_prize_l495_495245

def total_balls : ℕ := 10
def winning_balls : Finset ℕ := {6, 7, 8, 9, 10}

theorem probability_of_winning_prize : 
  ((winning_balls.card : ℚ) / (total_balls : ℚ)) = 1 / 2 := sorry

end probability_of_winning_prize_l495_495245


namespace geom_series_sum_l495_495853

theorem geom_series_sum : 
  let a₀ := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 5 in
  ∑ i in Finset.range n, a₀ * r ^ i = 341 / 1024 := 
  by
    sorry

end geom_series_sum_l495_495853


namespace tan_2A_cos_pi_over_3_minus_A_l495_495408

noncomputable def line_slope := 4 / 3
noncomputable def tan_A := 4 / 3

theorem tan_2A : 
  Real.tan (2 * (Real.atan line_slope)) = -24 / 7 := 
  by 
  sorry

theorem cos_pi_over_3_minus_A : 
  Real.cos ((Real.pi / 3) - (Real.atan tan_A)) = (3 + 4 * Real.sqrt 3) / 10 := 
  by
  sorry

end tan_2A_cos_pi_over_3_minus_A_l495_495408


namespace ice_cream_cone_cost_l495_495955

theorem ice_cream_cone_cost (total_sales : ℝ) (free_cones_given : ℕ) (cost_per_cone : ℝ) 
  (customers_per_group : ℕ) (cones_sold_per_group : ℕ) 
  (h1 : total_sales = 100)
  (h2: free_cones_given = 10)
  (h3: customers_per_group = 6)
  (h4: cones_sold_per_group = 5) :
  cost_per_cone = 2 := sorry

end ice_cream_cone_cost_l495_495955


namespace rectangle_area_is_243_square_meters_l495_495180

-- Given conditions
variables (w l : ℝ)
-- The width is one-third of its length
def width_one_third_length : Prop := w = l / 3

-- The perimeter of the rectangle is 72 meters
def perimeter_72 : Prop := 2 * (w + l) = 72

-- The area of the rectangle
def area : ℝ := w * l

-- Proof statement
theorem rectangle_area_is_243_square_meters 
  (h1 : width_one_third_length w l) 
  (h2 : perimeter_72 w l) : area w l = 243 := 
by 
sorry

end rectangle_area_is_243_square_meters_l495_495180


namespace find_AB_l495_495719

-- Definitions based on conditions
variables (AB CD : ℝ)

-- Given conditions
def area_ratio_condition : Prop :=
  AB / CD = 5 / 3

def sum_condition : Prop :=
  AB + CD = 160

-- The main statement to be proven
theorem find_AB (h_ratio : area_ratio_condition AB CD) (h_sum : sum_condition AB CD) :
  AB = 100 :=
by
  sorry

end find_AB_l495_495719


namespace base_addition_example_l495_495594

theorem base_addition_example : ∃ b : ℕ, 
  (∀ (d1 d2 d3 d4 d5 d6 : ℕ), 
    d1 = 2 ∧ d2 = 4 ∧ d3 = 5 ∧ d4 = 1 ∧ d5 = 3 ∧ d6 = 2 →
    (d1 * b^2 + d2 * b + d3) + (d4 * b^2 + d5 * b + d6) = (4 * b^2 + 0 * b + 0) ∧ 
    (d1 < b) ∧ (d2 < b) ∧ (d3 < b) ∧ (d4 < b) ∧ (d5 < b) ∧ (d6 < b)) ∧ 
  b = 8 :=
begin
  sorry
end

end base_addition_example_l495_495594


namespace part_I_part_II_l495_495285

noncomputable def z1 : ℂ := (3 - complex.i) / (1 + complex.i)

-- Part Ⅰ
theorem part_I : abs z1 = real.sqrt 5 := by sorry

-- Part Ⅱ
noncomputable def z2 : ℂ := 1 + 2 * complex.i

theorem part_II (h1 : (im : ℂ → ℝ) (z2) = 2)
                (h2 : im (z1 * z2) = 0) : z2 = 1 + 2 * complex.i := by sorry

end part_I_part_II_l495_495285


namespace max_value_of_ratio_l495_495746

noncomputable def maximum_value_ratio : ℝ :=
  let a : ℂ := real.sqrt 17
  let b : ℂ := complex.I * real.sqrt 19
  have H : ∀ z : ℂ, complex.abs z = 1 → (∃ k : ℝ, k = complex.abs (a - z) / complex.abs (b - z)) := sorry
  ∃ k : ℝ, ∀ z : ℂ, complex.abs z = 1 → (complex.abs (a - z) / complex.abs (b - z)) ≤ k ∧ k = (4 / 3)

theorem max_value_of_ratio : maximum_value_ratio = (4 / 3) :=
  sorry

end max_value_of_ratio_l495_495746


namespace geom_seq_general_term_and_sum_arith_seq_sum_20_l495_495456

-- part (I): Proof for the general term of the geometric sequence and the sum of the first n terms.
theorem geom_seq_general_term_and_sum (n : ℕ) : 
  let a₁ := 1, r := 3 in
  (∀ n, a n = r^(n-1)) ∧ (S n = (r^n - 1) / 2) :=
by
  sorry

-- part (II): Proof for the sum of the first 20 terms of the arithmetic sequence.
theorem arith_seq_sum_20 : 
  let a₁ := 1, r := 3 in
  let b₁ := 3, b₃ := 13, d := 5 in
  T 20 = 1010 :=
by
  sorry

end geom_seq_general_term_and_sum_arith_seq_sum_20_l495_495456


namespace find_second_discount_l495_495815

-- Define the problem conditions
def original_price : ℝ := 70
def customer_pays : ℝ := 59.85
def first_discount : ℝ := 10  -- 10%

-- Define the final price after the first discount
def price_after_first_discount := original_price * (1 - first_discount / 100)

-- Define the final price after both discounts
def final_price := customer_pays

-- Define the second discount calculation
def second_discount := 100 * (price_after_first_discount - final_price) / price_after_first_discount

-- State the theorem
theorem find_second_discount : second_discount = 5 := by
  sorry

end find_second_discount_l495_495815


namespace find_q_l495_495802

variable (x : ℝ)

def f (x : ℝ) := (5 * x^4 + 15 * x^3 + 30 * x^2 + 10 * x + 10)
def g (x : ℝ) := (2 * x^6 + 4 * x^4 + 10 * x^2)
def q (x : ℝ) := (-2 * x^6 + x^4 + 15 * x^3 + 20 * x^2 + 10 * x + 10)

theorem find_q :
  (∀ x, q x + g x = f x) ↔ (∀ x, q x = -2 * x^6 + x^4 + 15 * x^3 + 20 * x^2 + 10 * x + 10)
:= sorry

end find_q_l495_495802


namespace geometric_series_sum_l495_495846

theorem geometric_series_sum :
  let a := (1/4 : ℚ)
  ∧ let r := (1/4 : ℚ)
  ∧ let n := (5 : ℕ)
  → ∑ i in finset.range n, a * (r ^ i) = 341 / 1024 :=
by
  intro a r n
  apply sorry

end geometric_series_sum_l495_495846


namespace height_ratio_l495_495945

noncomputable def ratio_height (r : ℝ) (h_B h_V : ℝ) : Prop :=
  let V_B := π * r^2 * h_B in
  let V_V := 4 * π * r^2 * h_V in
  (8 / V_B) = (16 / V_V)

theorem height_ratio (r h_B h_V : ℝ) (H : ratio_height r h_B h_V) : h_V / h_B = 1 / 2 :=
by
  sorry

end height_ratio_l495_495945


namespace scheduling_arrangements_count_l495_495246

def Person := ℕ -- We represent persons A, B, and C as natural numbers 0, 1, 2 respectively.

def is_valid_arrangement (days: List Person) : Prop :=
  days.length = 5 ∧ -- Arrangement must span 5 days
  ∀ p, p ∈ [0, 1, 2] → p ∈ days ∧ -- Each person must be on duty at least one day
  (∀ i, i < 4 → days.get! i ≠ days.get! (i + 1)) ∧ -- No one can be on duty for two consecutive days
  days.get! 1 ≠ 0 -- A is not scheduled on the second day

def count_valid_arrangements : ℕ :=
  (List.permutations [0, 1, 2, 0, 1, 2].erase [0, 1, 2, 0, 1, 2]) -- Remove one arrangement because each person appears exactly twice 
  .filter is_valid_arrangement .length

theorem scheduling_arrangements_count : count_valid_arrangements = 8 :=
  by sorry

end scheduling_arrangements_count_l495_495246


namespace rona_age_l495_495047

theorem rona_age (R : ℕ) (hR1 : ∀ Rachel Collete : ℕ, Rachel = 2 * R ∧ Collete = R / 2 ∧ Rachel - Collete = 12) : R = 12 :=
sorry

end rona_age_l495_495047


namespace booth_visibility_correct_l495_495514

noncomputable def booth_visibility (L : ℝ) : ℝ × ℝ :=
  let ρ_min := L
  let ρ_max := (1 + Real.sqrt 2) / 2 * L
  (ρ_min, ρ_max)

theorem booth_visibility_correct (L : ℝ) (hL : L > 0) :
  booth_visibility L = (L, (1 + Real.sqrt 2) / 2 * L) :=
by
  sorry

end booth_visibility_correct_l495_495514


namespace num_valid_pairs_equals_four_l495_495924

theorem num_valid_pairs_equals_four 
  (a b : ℕ) (ha : a > 0) (hb : b > 0) (hba : b > a)
  (hcond : a * b = 3 * (a - 4) * (b - 4)) :
  ∃! (s : Finset (ℕ × ℕ)), s.card = 4 ∧ 
    ∀ (p : ℕ × ℕ), p ∈ s → p.1 > 0 ∧ p.2 > 0 ∧ p.2 > p.1 ∧
      p.1 * p.2 = 3 * (p.1 - 4) * (p.2 - 4) := sorry

end num_valid_pairs_equals_four_l495_495924


namespace find_k_value_l495_495322

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 5 * x^2 + 3 * x + 7
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := 3 * x^3 - k * x^2 + 4

theorem find_k_value : (f 5 - g 5 k = 45) → k = 27 / 25 :=
by
  intro h
  sorry

end find_k_value_l495_495322


namespace second_puppy_weight_l495_495113

variables (p1 p2 c1 c2 : ℝ)

-- Conditions from the problem statement
axiom h1 : p1 + p2 + c1 + c2 = 36
axiom h2 : p1 + c2 = 3 * c1
axiom h3 : p1 + c1 = c2
axiom h4 : p2 = 1.5 * p1

-- The question to prove: how much does the second puppy weigh
theorem second_puppy_weight : p2 = 108 / 11 :=
by sorry

end second_puppy_weight_l495_495113


namespace purely_periodic_denominator_l495_495660

theorem purely_periodic_denominator :
  ∀ q : ℕ, (∃ a : ℕ, (∃ b : ℕ, q = 99 ∧ (a < 10) ∧ (b < 10) ∧ (∃ f : ℝ, f = ↑a / (10 * q) ∧ ∃ g : ℝ, g = (0.01 * ↑b / (10 * (99 / q))))) → q = 11 ∨ q = 33 ∨ q = 99) :=
by sorry

end purely_periodic_denominator_l495_495660


namespace find_solutions_trig_inequality_l495_495255

theorem find_solutions_trig_inequality :
  ∀ x ∈ Icc (-π / 4) (7 * π / 4),
    (sin x) ^ 2018 + (cos x) ^ (-2019) ≥ (cos x) ^ 2018 + (sin x) ^ (-2019)
    ↔ x ∈ (Ico (-π / 4) 0 ∪ Ico (π / 4) (π / 2) ∪ Ioc (π) (5 * π / 4) ∪ Ioc (3 * π / 2) (7 * π / 4)) :=
by
  sorry

end find_solutions_trig_inequality_l495_495255


namespace output_for_input_8_is_8_over_65_l495_495884

def function_f (n : ℕ) : ℚ := n / (n^2 + 1)

theorem output_for_input_8_is_8_over_65 : function_f 8 = 8 / 65 := by
  sorry

end output_for_input_8_is_8_over_65_l495_495884


namespace surface_area_of_sphere_l495_495093

noncomputable def volume : ℝ := 72 * Real.pi

theorem surface_area_of_sphere (r : ℝ) (h : (4 / 3) * Real.pi * r^3 = volume) :
  4 * Real.pi * r^2 = 36 * Real.pi * (Real.cbrt 2)^2 :=
by
  sorry

end surface_area_of_sphere_l495_495093


namespace find_box_value_l495_495627

theorem find_box_value (r x : ℕ) 
  (h1 : x + r = 75)
  (h2 : (x + r) + 2 * r = 143) : 
  x = 41 := 
by
  sorry

end find_box_value_l495_495627


namespace rectangle_area_is_243_square_meters_l495_495177

-- Given conditions
variables (w l : ℝ)
-- The width is one-third of its length
def width_one_third_length : Prop := w = l / 3

-- The perimeter of the rectangle is 72 meters
def perimeter_72 : Prop := 2 * (w + l) = 72

-- The area of the rectangle
def area : ℝ := w * l

-- Proof statement
theorem rectangle_area_is_243_square_meters 
  (h1 : width_one_third_length w l) 
  (h2 : perimeter_72 w l) : area w l = 243 := 
by 
sorry

end rectangle_area_is_243_square_meters_l495_495177


namespace central_angle_of_cone_lateral_surface_in_degrees_l495_495636

def diameter := 4
def slant_height := 6

def central_angle (d : ℕ) (l : ℕ) : ℝ :=
  let r := d / 2
  let C := Real.pi * d
  let L := C
  let θ := L / l
  θ * 180 / Real.pi

theorem central_angle_of_cone_lateral_surface_in_degrees :
  central_angle diameter slant_height = 120 := by
  sorry

end central_angle_of_cone_lateral_surface_in_degrees_l495_495636


namespace rank_trigonometric_values_l495_495431

theorem rank_trigonometric_values :
  tan (7 * Real.pi / 5) > sin (2 * Real.pi / 5) ∧ sin (2 * Real.pi / 5) > cos (6 * Real.pi / 5) := 
by
  sorry

end rank_trigonometric_values_l495_495431


namespace f_28_eq_inverse_inverse_l495_495318

noncomputable def f1 (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

noncomputable def f : ℕ → (ℝ → ℝ) 
| 0     := id
| (n+1) := λ x, f1 (f n x)

theorem f_28_eq_inverse_inverse (x : ℝ) : 
  (f 35 x = f 5 x) → f 28 x = 1 / (1 - x) :=
sorry

end f_28_eq_inverse_inverse_l495_495318


namespace greatest_radius_l495_495686

theorem greatest_radius (r : ℕ) : (π * (r : ℝ)^2 < 75 * π) ↔ r ≤ 8 := 
by
  sorry

end greatest_radius_l495_495686


namespace cos_of_angle_B_l495_495354

theorem cos_of_angle_B (A B C : ℝ) (h1 : A + B + C = Real.pi) 
  (h2 : 6 * Real.sin A = 4 * Real.sin B) (h3 : 4 * Real.sin B = 3 * Real.sin C) : 
  Real.cos B = Real.sqrt 7 / 4 :=
by
  sorry

end cos_of_angle_B_l495_495354


namespace max_CA_CB_l495_495388

def ellipse_eq (x y : ℝ) : Prop := (x^2 / 8) + (y^2 / 4) = 1
def line_eq (k x y : ℝ) : Prop := y = k * x + 1
def left_vertex : ℝ × ℝ := (-2 * Real.sqrt 2, 0)
def intersects_ellipse (k : ℝ) (x_A y_A x_B y_B : ℝ) : Prop :=
  ellipse_eq x_A y_A ∧ line_eq k x_A y_A ∧ ellipse_eq x_B y_B ∧ line_eq k x_B y_B

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def total_distance (k : ℝ) (x_A y_A x_B y_B : ℝ) : ℝ :=
  distance (left_vertex) (x_A, y_A) + distance (left_vertex) (x_B, y_B)

noncomputable def max_total_distance : ℝ := 7.94403

theorem max_CA_CB (k : ℝ) (x_A y_A x_B y_B : ℝ) :
  intersects_ellipse k x_A y_A x_B y_B →
  total_distance k x_A y_A x_B y_B = max_total_distance :=
sorry

end max_CA_CB_l495_495388


namespace number_of_cards_above_1999_l495_495543

def numberOfCardsAbove1999 (n : ℕ) : ℕ :=
  if n < 2 then 0
  else if numberOfCardsAbove1999 (n-1) = n-2 then 1
  else numberOfCardsAbove1999 (n-1) + 2

theorem number_of_cards_above_1999 : numberOfCardsAbove1999 2000 = 927 := by
  sorry

end number_of_cards_above_1999_l495_495543


namespace cost_price_of_book_l495_495136

theorem cost_price_of_book (SP P : ℝ) (h_SP : SP = 200) (h_P : P = 0.20) : 
  let C := SP / (1 + P) in C = 200 / 1.20 := by
  sorry

end cost_price_of_book_l495_495136


namespace verna_sherry_total_weight_l495_495116

theorem verna_sherry_total_weight (haley verna sherry : ℕ)
  (h1 : verna = haley + 17)
  (h2 : verna = sherry / 2)
  (h3 : haley = 103) :
  verna + sherry = 360 :=
by
  sorry

end verna_sherry_total_weight_l495_495116


namespace find_AB_CDEF_l495_495378

-- Note the assumption that A, B, C, D, E, and F are all distinct single-digit numbers.

theorem find_AB_CDEF : ∃ (A B C D E F : ℕ), 
  (A < 10) ∧ (B < 10) ∧ (C < 10) ∧ (D < 10) ∧ (E < 10) ∧ (F < 10) ∧ (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (A ≠ F) ∧ 
  (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (B ≠ F) ∧ (C ≠ D) ∧ (C ≠ E) ∧ (C ≠ F) ∧ (D ≠ E) ∧ (D ≠ F) ∧ (E ≠ F) ∧ 
  (
    let ABCD := A * 1000 + B * 100 + C * 10 + D in
    let CD := C * 10 + D in
    let C8CEC := C * 10000 + 8 * 1000 + C * 100 + E * 10 + C in
    let F8F6C := F * 10000 + 8 * 1000 + F * 100 + 6 * 10 + C in
    ABCD + CD = C8CEC ∧
    ABCD + CD + ABCD = F8F6C
  ) ∧ ABCD = 201973 := sorry

end find_AB_CDEF_l495_495378


namespace intersection_eq_l495_495622

def setA : Set ℕ := {0, 1, 2, 3, 4, 5 }
def setB : Set ℕ := { x | |(x : ℤ) - 2| ≤ 1 }

theorem intersection_eq :
  setA ∩ setB = {1, 2, 3} := by
  sorry

end intersection_eq_l495_495622


namespace time_to_decorate_l495_495758

variable (mia_rate billy_rate total_eggs : ℕ)

theorem time_to_decorate (h_mia : mia_rate = 24) (h_billy : billy_rate = 10) (h_total : total_eggs = 170) :
  total_eggs / (mia_rate + billy_rate) = 5 :=
by
  sorry

end time_to_decorate_l495_495758


namespace horizontal_distance_travel_l495_495214

noncomputable def radius : ℝ := 2
noncomputable def angle_degrees : ℝ := 30
noncomputable def angle_radians : ℝ := angle_degrees * (Real.pi / 180)
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
noncomputable def cos_theta : ℝ := Real.cos angle_radians
noncomputable def horizontal_distance (r : ℝ) (θ : ℝ) : ℝ := (circumference r) * (Real.cos θ)

theorem horizontal_distance_travel (r : ℝ) (θ : ℝ) (h_radius : r = 2) (h_angle : θ = angle_radians) :
  horizontal_distance r θ = 2 * Real.pi * Real.sqrt 3 := 
by
  sorry

end horizontal_distance_travel_l495_495214


namespace total_weight_of_8_moles_of_BaCl2_l495_495485

-- Define atomic weights
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_Cl : ℝ := 35.45

-- Define the molecular weight of BaCl2
def molecular_weight_BaCl2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_Cl

-- Define the number of moles
def moles : ℝ := 8

-- Define the total weight calculation
def total_weight : ℝ := molecular_weight_BaCl2 * moles

-- The theorem to prove
theorem total_weight_of_8_moles_of_BaCl2 : total_weight = 1665.84 :=
by sorry

end total_weight_of_8_moles_of_BaCl2_l495_495485


namespace at_least_240_students_l495_495546

-- Define the total number of students
def total_students : ℕ := 1200

-- Define the 80th percentile score
def percentile_80_score : ℕ := 103

-- Define the number of students below the 80th percentile
def students_below_80th_percentile : ℕ := total_students * 80 / 100

-- Define the number of students with at least the 80th percentile score
def students_at_least_80th_percentile : ℕ := total_students - students_below_80th_percentile

-- The theorem to prove
theorem at_least_240_students : students_at_least_80th_percentile ≥ 240 :=
by
  -- Placeholder proof, to be filled in as the actual proof
  sorry

end at_least_240_students_l495_495546


namespace find_n_l495_495007

open Set

-- Definitions
def S : Set (ℕ × ℕ) := { p | 1 ≤ p.1 ∧ p.1 ≤ 2012 ∧ 1 ≤ p.2 ∧ p.2 ≤ 2012 }

def N (a b : ℕ) : Set (ℕ × ℕ) :=
  { (a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1) }

-- Conditions and Theorem
noncomputable def T (S : Set (ℕ × ℕ)) (n : ℕ) : Set (ℕ × ℕ) := 
  { p ∈ S | p.1 <= n ∧ p.2 <= n }

theorem find_n (n : ℕ) (T : Set (ℕ × ℕ))
  (hT : ∀ t ∈ T, t ∈ S) -- Ensuring all points in T are from S
  (hexp: (∑ (p : ℕ × ℕ) in T, (N p.1 p.2 ∩ T).card).toNat = 4) :
  n = 2013 := sorry

end find_n_l495_495007


namespace angle_BCG_eq_angle_BCF_l495_495745

open EuclideanGeometry

variables (A B C D E F G : Point)
variables (hD : lies_on D (line_segment A C))
variables (hBD_CD : distance B D = distance C D)
variables (hE : lies_on E (line_segment B C))
variables (hG : intersection_point (line_through A E) (line_through B D) = some G)
variables (hF : lies_on F (line_through A B))
variables (hEF_par_BD : parallel (line_through E F) (line_through B D))

theorem angle_BCG_eq_angle_BCF :
  angle (line_through B C) (line_through C G) = angle (line_through B C) (line_through C F) :=
by
  sorry

end angle_BCG_eq_angle_BCF_l495_495745


namespace rectangle_area_is_243_square_meters_l495_495178

-- Given conditions
variables (w l : ℝ)
-- The width is one-third of its length
def width_one_third_length : Prop := w = l / 3

-- The perimeter of the rectangle is 72 meters
def perimeter_72 : Prop := 2 * (w + l) = 72

-- The area of the rectangle
def area : ℝ := w * l

-- Proof statement
theorem rectangle_area_is_243_square_meters 
  (h1 : width_one_third_length w l) 
  (h2 : perimeter_72 w l) : area w l = 243 := 
by 
sorry

end rectangle_area_is_243_square_meters_l495_495178


namespace geometric_series_sum_l495_495861

theorem geometric_series_sum :
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  (a + a * r + a * r^2 + a * r^3 + a * r^4) = 341 / 1024 :=
by
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  -- Proof steps go here
  sorry

end geometric_series_sum_l495_495861


namespace time_to_decorate_l495_495757

variable (mia_rate billy_rate total_eggs : ℕ)

theorem time_to_decorate (h_mia : mia_rate = 24) (h_billy : billy_rate = 10) (h_total : total_eggs = 170) :
  total_eggs / (mia_rate + billy_rate) = 5 :=
by
  sorry

end time_to_decorate_l495_495757


namespace max_value_harmonic_series_l495_495664

theorem max_value_harmonic_series (k l m : ℕ) (hk : 0 < k) (hl : 0 < l) (hm : 0 < m)
  (h : 1/k + 1/l + 1/m < 1) : 
  (1/2 + 1/3 + 1/7) = 41/42 := 
sorry

end max_value_harmonic_series_l495_495664


namespace find_AB_l495_495792

noncomputable def point := ℝ × ℝ

noncomputable def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

variables (A B C D E : point)
variable (x : ℝ)

-- Conditions
axioms
(h1 : B.1 - A.1 = x)
(h2 : C.1 - B.1 = 15)
(h3 : D.1 - C.1 = x)
(h4 : distance B E = 13)
(h5 : distance C E = 13)
(h6 : 2 * (distance A E + distance D E + distance A D) = 2 * (13 + 13 + 15))

-- Theorem to prove
theorem find_AB : x = 953 / 82 := 
sorry

end find_AB_l495_495792


namespace sum_palindromic_primes_lt_100_eq_429_l495_495781

def is_palindromic_prime (n : ℕ) : Prop :=
  (n ≥ 10 ∧ n < 100) ∧
  (Prime n ∧ 
   let d1 := n / 10 in
   let d2 := n % 10 in
   Prime (d2 * 10 + d1))

theorem sum_palindromic_primes_lt_100_eq_429 : 
  (∑ n in Finset.filter is_palindromic_prime (Finset.range 100), n) = 429 :=
by sorry

end sum_palindromic_primes_lt_100_eq_429_l495_495781


namespace kenya_more_peanuts_l495_495387

-- Define the number of peanuts Jose has
def Jose_peanuts : ℕ := 85

-- Define the number of peanuts Kenya has
def Kenya_peanuts : ℕ := 133

-- The proof problem: Prove that Kenya has 48 more peanuts than Jose
theorem kenya_more_peanuts : Kenya_peanuts - Jose_peanuts = 48 :=
by
  -- The proof will go here
  sorry

end kenya_more_peanuts_l495_495387


namespace machine_value_percentage_decrease_l495_495074

theorem machine_value_percentage_decrease (initial_value : ℕ) (value_after_two_years : ℕ) (percentage_decrease : ℝ) :
  initial_value = 8000 ∧ value_after_two_years = 5600 →
  percentage_decrease ≈ 16.33 :=
by
  sorry

end machine_value_percentage_decrease_l495_495074


namespace distance_from_blast_site_l495_495916

-- Define constants
def sound_speed : ℝ := 330  -- speed of sound in m/s
def time_heard_second_blast : ℝ := (30 * 60) + 24  -- time heard second blast in seconds
def time_actual_second_blast : ℝ := 30 * 60  -- time second blast occurred in seconds
def time_difference : ℝ := time_heard_second_blast - time_actual_second_blast  -- time difference when the sound reached

-- Define the theorem
theorem distance_from_blast_site : (sound_speed * time_difference) = 7920 := by
  sorry

end distance_from_blast_site_l495_495916


namespace eval_expression_l495_495969

theorem eval_expression (x : ℝ) (hx : x ≠ 0) : 
  x^4 - x^(-4) = (x - x^(-1)) * (x + x^(-1)) * (x^2 + x^(-2)) :=
by
  sorry

end eval_expression_l495_495969


namespace puzzle_solution_exists_l495_495797

/-- Define the puzzle problem:
  2014 + YEAR = SOCHI, where different letters represent different digits. -/
theorem puzzle_solution_exists :
  ∃ (Г О Д С Ч И : ℕ),
    (Г ≠ О ∧ Г ≠ Д ∧ Г ≠ С ∧ Г ≠ Ч ∧ Г ≠ И ∧ 
    О ≠ Д ∧ О ≠ С ∧ О ≠ Ч ∧ О ≠ И ∧ 
    Д ≠ С ∧ Д ≠ Ч ∧ Д ≠ И ∧ 
    С ≠ Ч ∧ С ≠ И ∧
    Ч ≠ И) ∧
    (2014 + (100 * Г + 10 * О + Д) = 
     1000 * С + 100 * О + 10 * Ч + И) ∧
    ((Г = 8 ∧ О = 9 ∧ Д = 1 ∧ С = 2 ∧ Ч = 0 ∧ И = 5) ∨
     (Г = 8 ∧ О = 9 ∧ Д = 3 ∧ С = 2 ∧ Ч = 0 ∧ И = 7) ∨
     (Г = 8 ∧ О = 9 ∧ Д = 6 ∧ С = 2 ∧ Ч = 1 ∧ И = 0)) :=
by sorry

end puzzle_solution_exists_l495_495797


namespace coefficient_x4_in_expansion_l495_495065

theorem coefficient_x4_in_expansion : (Finset.Ico 0 5).sum (λ r, if 2 * (8 - r) = 4 then (Nat.choose 8 r) * 2^r else 0) = 1120 := by
  sorry

end coefficient_x4_in_expansion_l495_495065


namespace cube_splitting_odd_numbers_l495_495272

theorem cube_splitting_odd_numbers (m : ℕ) (h1 : m > 1) (h2 : ∃ k, 2 * k + 1 = 333) : m = 18 :=
sorry

end cube_splitting_odd_numbers_l495_495272


namespace find_x_l495_495585

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : ⌈x⌉ * x = 220) : x = 14.67 :=
sorry

end find_x_l495_495585


namespace problem_3_pow_m_n_l495_495279

-- Variables and conditions
variable (m n : ℝ)
variable (h1 : 3^m = 4)
variable (h2 : 3^n = 5)

-- The target proof
theorem problem_3_pow_m_n :
  3^(m - n) = (4 / 5) :=
sorry

end problem_3_pow_m_n_l495_495279


namespace sum_of_elements_in_A_inter_Z_l495_495651

noncomputable def A : Set ℝ := {x | abs (x - 2) < 3}
def Z : Set ℤ := Set.univ

theorem sum_of_elements_in_A_inter_Z :
  (∑ k in (A ∩ (Z.map coe)).to_finset, k) = 10 :=
by
  sorry

end sum_of_elements_in_A_inter_Z_l495_495651


namespace find_k_l495_495071

theorem find_k (k : ℝ) :
  let A := (-3,0)
  let B := (0,-3)
  let X := (0,10)
  let Y := (18,k)
  (0 - 10) / (18 - 0) = (B.2 - A.2) / (B.1 - A.1) →
  k = -8 :=
by
  -- Using the conditions
  let A := (-3,0 : ℝ × ℝ)
  let B := (0,-3 : ℝ × ℝ)
  let X := (0,10 : ℝ × ℝ)
  have slope_AB : (B.2 - A.2) / (B.1 - A.1) = -1 :=
    calc
      (B.2 - A.2) / (B.1 - A.1) = (-3 - 0) / (0 - (-3)) : by sorry
      ... = -3 / 3 : by sorry
      ... = -1 : by sorry
  let Y := (18,k : ℝ × ℝ)
  have slope_XY : (k - 10) / 18 = -1 :=
    calc
      (k - 10) / 18 = (0 - 10) / (18 - 0) : by sorry -- Using the given equality
      ... = -1 : by sorry -- Slope between points (0,10) and (18,0) computed similarly
  have k_eq : k - 10 = -18 := by sorry
  have k_val : k = -8 := by
    calc
      k = 10 - 18 :  sorry -- Solving the equation
      ... = -8 : sorry
  exact k_val

end find_k_l495_495071


namespace volume_and_height_of_tetrahedron_l495_495226

noncomputable def V := 12
noncomputable def h := 2 * Real.sqrt 3
def A₁ := (5, 2, 0)
def A₂ := (2, 5, 0)
def A₃ := (1, 2, 4)
def A₄ := (-1, 1, 1)

theorem volume_and_height_of_tetrahedron :
  let V_actual := (1 : ℝ) / 6 * Real.abs (
    (by {exact -3, sorry}) -- placeholder for determinant calculation
  )
  let S := (1 : ℝ) / 2 * Real.norm (by { exact ⟨12, 12, 12⟩, sorry }) -- placeholder for cross product calculation
  let h_actual := by { exact 3 * V_actual / S, sorry } -- placeholder for height calculation
  V_actual = V ∧ h_actual = h := sorry

end volume_and_height_of_tetrahedron_l495_495226


namespace minimum_box_cost_l495_495656

theorem minimum_box_cost {V : ℝ} (total_volume : V = 3060000)
    (small_volume_usable : ℝ := (10 * 10 * 10 * 0.8))
    (medium_volume_usable : ℝ := (20 * 20 * 15 * 0.8 / 2))
    (large_volume_usable : ℝ := (40 * 40 * 30 * 0.8 / 5))
    (small_price : ℝ := 0.60) (medium_price : ℝ := 0.90) (large_price : ℝ := 1.20)
    (num_large_boxes : ℝ := 80) : ℝ :=
  let large_cost := num_large_boxes * large_price in
  if total_volume / large_volume_usable <= num_large_boxes then large_cost else
  sorry

end minimum_box_cost_l495_495656


namespace exists_c_d_rel_prime_l495_495013

theorem exists_c_d_rel_prime (a b : ℤ) :
  ∃ c d : ℤ, ∀ n : ℤ, gcd (a * n + c) (b * n + d) = 1 :=
sorry

end exists_c_d_rel_prime_l495_495013


namespace find_number_of_observations_l495_495831

theorem find_number_of_observations 
  (n : ℕ) 
  (mean_before_correction : ℝ)
  (incorrect_observation : ℝ)
  (correct_observation : ℝ)
  (mean_after_correction : ℝ) 
  (h0 : mean_before_correction = 36)
  (h1 : incorrect_observation = 23)
  (h2 : correct_observation = 45)
  (h3 : mean_after_correction = 36.5) 
  (h4 : (n * mean_before_correction + (correct_observation - incorrect_observation)) / n = mean_after_correction) : 
  n = 44 := 
by
  sorry

end find_number_of_observations_l495_495831


namespace math_problem_proof_l495_495669

theorem math_problem_proof (a b x y : ℝ) 
  (h1: x = a) 
  (h2: y = b)
  (h3: a + a = b * a)
  (h4: y = a)
  (h5: a * a = a + a)
  (h6: b = 3) : 
  x * y = 4 := 
by 
  sorry

end math_problem_proof_l495_495669


namespace Misha_earnings_needed_l495_495419

-- Define the conditions and the goal in Lean 4
def Misha_current_dollars : ℕ := 34
def Misha_target_dollars : ℕ := 47

theorem Misha_earnings_needed : Misha_target_dollars - Misha_current_dollars = 13 := by
  sorry

end Misha_earnings_needed_l495_495419


namespace nadya_pies_proof_l495_495766

variables (x r b s : ℕ)

def condition_1 := r = x
def condition_2 := r + r = x * 2
def condition_3 := b = r - 14
def condition_4 := s = (r + b) / 2

theorem nadya_pies_proof :
  r = 21 → b = 7 → s = 14 → (condition_1 x r b s) ∧ (condition_2 x r b s) ∧ (condition_3 x r b s) ∧ (condition_4 x r b s) :=
by
  sorry

end nadya_pies_proof_l495_495766


namespace find_values_of_pqr_l495_495988

def A (p : ℝ) := {x : ℝ | x^2 + p * x - 2 = 0}
def B (q r : ℝ) := {x : ℝ | x^2 + q * x + r = 0}
def A_union_B (p q r : ℝ) := A p ∪ B q r = {-2, 1, 5}
def A_intersect_B (p q r : ℝ) := A p ∩ B q r = {-2}

theorem find_values_of_pqr (p q r : ℝ) :
  A_union_B p q r → A_intersect_B p q r → p = -1 ∧ q = -3 ∧ r = -10 :=
by
  sorry

end find_values_of_pqr_l495_495988


namespace simplify_proof_l495_495800

noncomputable def simplify_expression (x : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) (hx_1 : x ≠ -1) : ℝ :=
  (1 - 1/x) / ((1 - x^2) / x)

theorem simplify_proof (x : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) (hx_1 : x ≠ -1) : 
  simplify_expression x hx hx1 hx_1 = -1 / (1 + x) := by 
  sorry

end simplify_proof_l495_495800


namespace legs_on_ground_l495_495150

def numberOfLegsWalkingOnGround (horses men : ℕ) (half_ride_half_walk : men / 2) (total_horses : horses = 10) : ℕ :=
  let walking_men := men / 2
  2 * walking_men + 4 * horses

theorem legs_on_ground (horses men : ℕ) (half_ride_half_walk : men / 2) (total_horses : horses = 10) h : 
  horses = men → numberOfLegsWalkingOnGround horses men half_ride_half_walk total_horses = 50 :=
by
  intro h
  sorry

end legs_on_ground_l495_495150


namespace measuring_rod_with_rope_l495_495936

theorem measuring_rod_with_rope (x y : ℝ) (h1 : x = y + 5) (h2 : (1 / 2) * x = y - 5) :
  x = y + 5 ∧ (1 / 2) * x = y - 5 :=
by
  split
  · exact h1
  · exact h2

end measuring_rod_with_rope_l495_495936


namespace total_money_made_l495_495822

-- Define the conditions
def dollars_per_day : Int := 144
def number_of_days : Int := 22

-- State the proof problem
theorem total_money_made : (dollars_per_day * number_of_days = 3168) :=
by
  sorry

end total_money_made_l495_495822


namespace distance_between_anastasia_bananastasia_l495_495518

theorem distance_between_anastasia_bananastasia (sound_speed : ℝ) (time_diff : ℝ) 
(anastasia_first : bool) : ℝ :=
  if sound_speed = 343 ∧ time_diff = 5 ∧ anastasia_first = false then 1715 else sorry

end distance_between_anastasia_bananastasia_l495_495518


namespace card_pair_probability_deck_l495_495526

theorem card_pair_probability_deck (cards : Finset ℕ) (h₁ : cards.card = 48) :
  let num_pairs : ℕ := 12 * (nat.choose 4 2)
  let total_choices : ℕ := (nat.choose 48 2)
  let prob_pair : ℚ := (num_pairs : ℚ) / (total_choices : ℚ)
  num_pairs.gcd total_choices = 1 →
  num_pairs / total_choices = 3 / 47 →
  m + n = 50 :=
by
  intros num_pairs total_choices prob_pair h_gcd h_ratio
  sorry

end card_pair_probability_deck_l495_495526


namespace seeds_in_first_plot_is_300_l495_495602

-- Definitions for conditions
variable {x : ℕ} -- Number of seeds planted in the first plot
def seeds_in_second_plot := 200
def germination_rate_first_plot := 0.15
def germination_rate_second_plot := 0.35
def total_germination_rate := 0.23

-- Proof that the number of seeds in the first plot is 300
theorem seeds_in_first_plot_is_300 
  (h : 0.15 * x + 0.35 * seeds_in_second_plot = 0.23 * (x + seeds_in_second_plot)) : 
  x = 300 :=
by {
  sorry
}

end seeds_in_first_plot_is_300_l495_495602


namespace trigonometric_identity_l495_495251

theorem trigonometric_identity :
  (1 / Real.cos (80 * (Real.pi / 180)) - Real.sqrt 3 / Real.sin (80 * (Real.pi / 180)) = 4) :=
by
  sorry

end trigonometric_identity_l495_495251


namespace greatest_integer_radius_l495_495676

theorem greatest_integer_radius (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
begin
  have h1 : r^2 < 75, from (lt_div_iff (real.pi_pos)).mp (by rwa [mul_div_cancel_left _ (ne_of_gt real.pi_pos)]),
  -- To show the highest integer r satisfies r^2 < 75 is 8:
  have : r ≤ nat.floor (real.sqrt 75), 
  { exact nat.le_floor_of_lt (by exact_mod_cast h1) },
  -- Since nat.floor (real.sqrt 75) == 8
  exact le_trans this (by norm_num)
end

end greatest_integer_radius_l495_495676


namespace area_of_triangle_l495_495721

-- Definitions of the variables and the area calculation
variable (a b c : ℝ)
variable (A B C : ℝ)

-- Area of the triangle and given values
def triangle_area (a b C : ℝ) : ℝ := (1 / 2) * a * b * Real.sin C

theorem area_of_triangle :
  b = 7 → c = 5 → B = 2 * π / 3 → a = 3 → triangle_area 3 5 (2 * π / 3) = 21 * Real.sqrt 3 / 4 := by
  sorry

end area_of_triangle_l495_495721


namespace find_number_of_observations_l495_495830

theorem find_number_of_observations 
  (n : ℕ) 
  (mean_before_correction : ℝ)
  (incorrect_observation : ℝ)
  (correct_observation : ℝ)
  (mean_after_correction : ℝ) 
  (h0 : mean_before_correction = 36)
  (h1 : incorrect_observation = 23)
  (h2 : correct_observation = 45)
  (h3 : mean_after_correction = 36.5) 
  (h4 : (n * mean_before_correction + (correct_observation - incorrect_observation)) / n = mean_after_correction) : 
  n = 44 := 
by
  sorry

end find_number_of_observations_l495_495830


namespace intersection_point_of_lines_l495_495962

noncomputable def line1 (x : ℝ) : ℝ := 3 * x - 4

noncomputable def line2 (x : ℝ) : ℝ := -1 / 3 * x + 10 / 3

def point : ℝ × ℝ := (4, 2)

theorem intersection_point_of_lines :
  ∃ (x y : ℝ), line1 x = y ∧ line2 x = y ∧ (x, y) = (2.2, 2.6) :=
by
  sorry

end intersection_point_of_lines_l495_495962


namespace solve_inequality_l495_495992

noncomputable def solution_set (a b : ℝ) (x : ℝ) : Prop :=
x < -1 / b ∨ x > 1 / a

theorem solve_inequality (a b : ℝ) (x : ℝ)
  (h_a : a > 0) (h_b : b > 0) :
  (-b < 1 / x ∧ 1 / x < a) ↔ solution_set a b x :=
by
  sorry

end solve_inequality_l495_495992


namespace locus_of_projections_l495_495820

theorem locus_of_projections (Pi : Plane) (A : Point) (cone : Cone) (M : Point) 
  (h1 : Pi.contains( cone.vertex )) 
  (h2 : Pi.perpendicular_to_axis (cone.axis))
  (h3 : Pi.contains( A )) 
  (h4 : cone.surface.contains( M )) 
  (h5 : light_ray_reflection (A, M, cone, Pi)) 
  : locus_of_projections Pi M = {p | p.1^2 + p.2^2 - (2 * (A.x * (tan(cone.alpha))^2) * p.1) / (1 + (tan(cone.alpha))^2) = 0} :=
sorry

end locus_of_projections_l495_495820


namespace locus_of_M_ratio_CD_AB_l495_495028

theorem locus_of_M (P : ℝ × ℝ) (C_1 : set (ℝ × ℝ)) (hP : P ∈ C_1) 
  (hC1 : ∀ (x y : ℝ), (x, y) ∈ C_1 ↔ x^2 + y^2 = 2)
  (M Q : ℝ × ℝ) (hMQP : ∀ (x y : ℝ), (x, y) = P → sqrt 2 * dist M Q = dist (0,0) P)
  : ∃ C_2 : set (ℝ × ℝ), (∀ (x y : ℝ), (x, y) ∈ C_2 ↔ x^2 / 2 + y^2 = 1) := 
sorry

theorem ratio_CD_AB (T : ℝ × ℝ) (hT : T.1 = 2) (C D A B : ℝ × ℝ)
  (hA : A ∈ (circle_intersections_with_tangent_line T)) 
  (hB : B ∈ (circle_intersections_with_tangent_line T)) 
  (hAB : ∃ m b : ℝ, line_eq m b A B)
  (hCD : ∃ m b : ℝ, line_eq m b C D) 
  : ∃ (s : ℝ), 0 ≤ s ∧ range_of_values : set ℝ, range_of_values = 
    {x | sqrt 2 / 2 ≤ x ∧ x < 1} ∧ dist_ratio s (dist C D) (dist A B) := 
sorry

end locus_of_M_ratio_CD_AB_l495_495028


namespace find_solutions_trig_inequality_l495_495256

theorem find_solutions_trig_inequality :
  ∀ x ∈ Icc (-π / 4) (7 * π / 4),
    (sin x) ^ 2018 + (cos x) ^ (-2019) ≥ (cos x) ^ 2018 + (sin x) ^ (-2019)
    ↔ x ∈ (Ico (-π / 4) 0 ∪ Ico (π / 4) (π / 2) ∪ Ioc (π) (5 * π / 4) ∪ Ioc (3 * π / 2) (7 * π / 4)) :=
by
  sorry

end find_solutions_trig_inequality_l495_495256


namespace rectangular_coords_transformation_l495_495921

noncomputable def sphericalToRectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
(ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem rectangular_coords_transformation :
  let ρ := Real.sqrt (2 ^ 2 + (-3) ^ 2 + 6 ^ 2)
  let φ := Real.arccos (6 / ρ)
  let θ := Real.arctan (-3 / 2)
  sphericalToRectangular ρ (Real.pi + θ) φ = (-2, 3, 6) :=
by
  sorry

end rectangular_coords_transformation_l495_495921


namespace palindromic_primes_sum_l495_495773

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.reverse.foldl (λ acc d, acc * 10 + d) 0

def is_palindromic_prime (n : ℕ) : Prop :=
  n < 100 ∧ 10 ≤ n ∧ is_prime n ∧ is_prime (reverse_digits n)

def sum_of_palindromic_primes : ℕ :=
  ((list.range 100).filter is_palindromic_prime).sum

theorem palindromic_primes_sum : sum_of_palindromic_primes = 429 := 
  by 
    sorry

end palindromic_primes_sum_l495_495773


namespace transform_sin_graph_l495_495108

theorem transform_sin_graph :
  (∀ x : ℝ, 2 * sin (x) = 2 * sin (3 * (x - (-π / 6)))) →
  (∀ x : ℝ, 2 * sin (x) = 2 * sin ((1/3) * (x - π / 6)) → False) →
  (∀ x : ℝ, 2 * sin (x) = 2 * sin (3 * (x + π / 6)) → False) →
  (∀ x : ℝ, 2 * sin (x) = 2 * sin ((1/3) * (x + π / 6)) → False) →
  (∀ x : ℝ, 2 * sin (3 * (x + π / 6)) = 2 * sin ((1/3) * (x - π / 6)) → False) →
  (∀ x : ℝ, 2 * sin (3 * (x + π / 6)) = 2 * sin (3 * (x - π / 6)) → False) →
  (∀ x : ℝ, 2 * sin (3 * (x + π / 6)) = 2 * sin ((1/3) * (x + π / 6)) → False) →
  (∀ x : ℝ, 2 * sin ((1/3) * (x - π / 6)) = 2 * sin (3 * (x + π / 6)) → False) →
  (∀ x : ℝ, 2 * sin ((1/3) * (x - π / 6)) = 2 * sin ((1/3) * (x + π / 6)) → False) →
  (∀ x : ℝ, 2 * sin (3 * (x - π / 6)) = 2 * sin (3 * (x + π / 6)) → False) →
  true :=
sorry

end transform_sin_graph_l495_495108


namespace sum_first_20_odds_is_400_l495_495489

-- Define the sequence of the first 20 positive odd integers
def sequence (n : ℕ) : ℕ := 1 + 2 * n

-- Define the sum of the first 'n' terms of an arithmetic sequence
def sum_arithmetic_sequence (a l n : ℕ) : ℕ := ((a + l) * n) / 2

-- Define the sum of the first 20 positive odd integers
def sum_first_20_odds : ℕ := sum_arithmetic_sequence 1 39 20

-- Claim that the sum of the first 20 positive odd integers is 400
theorem sum_first_20_odds_is_400 : sum_first_20_odds = 400 :=
by
  -- Proof omitted
  sorry

end sum_first_20_odds_is_400_l495_495489


namespace greatest_int_radius_lt_75pi_l495_495696

noncomputable def circle_radius_max (A : ℝ) (π : ℝ) : ℕ :=
  Nat.floor (Real.sqrt (A / π))

theorem greatest_int_radius_lt_75pi :
  circle_radius_max 75 Real.pi = 8 := by
  sorry

end greatest_int_radius_lt_75pi_l495_495696


namespace number_of_N_satisfying_x_power_floor_eq_N_l495_495266

theorem number_of_N_satisfying_x_power_floor_eq_N :
  {N : ℕ | 1 ≤ N ∧ N ≤ 2018 ∧ ∃ (x : ℝ) (n : ℕ), n ≤ x ∧ x < n+1 ∧ x^n = N}.to_finset.card = 412 :=
begin
  -- Proof would go here.
  sorry
end

end number_of_N_satisfying_x_power_floor_eq_N_l495_495266


namespace parametric_polar_relationship_l495_495375

-- Definitions of given conditions
def point_P : ℝ × ℝ := (1, -5)
def inclination_angle_l : ℝ := Real.pi / 3
def polar_coordinates_C : ℝ × ℝ := (4, Real.pi / 2)
def radius_C : ℝ := 4

-- Parametric equations of the line l
def parametric_eq_l (t : ℝ) : ℝ × ℝ :=
  (1 + (1/2) * t, -5 + (Real.sqrt 3 / 2) * t)

-- Polar equation of the circle C
def polar_eq_C (θ : ℝ) : ℝ :=
  8 * Real.sin θ

-- Positional relationship between line l and circle C
def positional_relationship_l_C : String :=
  if Real.abs ((-4 - 5 - Real.sqrt 3) / 2) > 4 then "disjoint" else "not disjoint"

-- Proof problem
theorem parametric_polar_relationship : 
  parametric_eq_l t = (1 + (1/2) * t, -5 + (Real.sqrt 3 / 2) * t) 
  ∧ polar_eq_C θ = 8 * Real.sin θ 
  ∧ positional_relationship_l_C = "disjoint" := 
by 
  sorry

end parametric_polar_relationship_l495_495375


namespace wires_cannot_be_compared_l495_495115

-- Defining the lengths of the two wires
variables (L₁ L₂ : ℝ) (h_same_len : L₁ = L₂)

-- Defining the cuts on the wires
def cut_first_wire (L : ℝ) : ℝ := L * (1 - 1/4)
def cut_second_wire (L : ℝ) : ℝ := L - 1/4

-- Statement of the theorem
theorem wires_cannot_be_compared (L : ℝ) (h_positive : L > 0) :
  (cut_first_wire L = cut_second_wire L) ↔ false :=
by sorry

end wires_cannot_be_compared_l495_495115


namespace sum_of_consecutive_even_numbers_l495_495430

theorem sum_of_consecutive_even_numbers 
  (n k : ℕ) (hn : n > 2) (hk : k > 2) : 
  ∃ a, n * (n-1)^(k-1) = ∑ j in range n, (2 * a + 2 * j) := 
sorry

end sum_of_consecutive_even_numbers_l495_495430


namespace total_sum_is_2696_l495_495961

def numbers := (100, 4900)

def harmonic_mean (a b : ℕ) : ℕ :=
  2 * a * b / (a + b)

def arithmetic_mean (a b : ℕ) : ℕ :=
  (a + b) / 2

theorem total_sum_is_2696 : 
  harmonic_mean numbers.1 numbers.2 + arithmetic_mean numbers.1 numbers.2 = 2696 :=
by
  sorry

end total_sum_is_2696_l495_495961


namespace hitting_target_exactly_three_times_l495_495714

def hitting_target_probability :
  ℕ → (list ℕ) → (list ℕ) → Prop
  | shots, target_numbers, hits_numbers => 
    (shots = 4) → 
    (target_numbers = [1, 2, 3, 4, 5, 6]) → 
    (hits_numbers = [6830, 3013, 7055, 7430, 7740, 4422, 7884, 2604, 3346, 0952, 6807, 9706, 5774, 5725, 6576, 5929, 9768, 6071, 9138, 6754]) →
    (counts.each digit in target_numbers appears exactly three times in hits_numbers) →
    (probability = 25%)

theorem hitting_target_exactly_three_times :
  hitting_target_probability 4 [1, 2, 3, 4, 5, 6] [6830, 3013, 7055, 7430, 7740, 4422, 7884, 2604, 3346, 0952, 6807, 9706, 5774, 5725, 6576, 5929, 9768, 6071, 9138, 6754] 25 :=
by 
  sorry

end hitting_target_exactly_three_times_l495_495714


namespace camel_adjacent_impossible_l495_495140

def camel_move_impossible (board : Fin 10 × Fin 10) (start : Fin 10 × Fin 10) : Prop :=
  ∀ (adj : Fin 10 × Fin 10), 
    adjacent start adj →
    ¬(camel_can_move start adj)

def adjacent (s1 s2 : Fin 10 × Fin 10) : Prop :=
  (s1.fst = s2.fst ∧ (s1.snd = s2.snd + 1 ∨ s1.snd + 1 = s2.snd)) ∨
  (s1.snd = s2.snd ∧ (s1.fst = s2.fst + 1 ∨ s1.fst + 1 = s2.fst))

def camel_can_move (start dest : Fin 10 × Fin 10) : Prop :=
  let (i, j) := start;
  let (i', j') := dest;
  abs (i' - i) = 1 ∧ abs (j' - j) = 3 ∨
  abs (i' - i) = 3 ∧ abs (j' - j) = 1

theorem camel_adjacent_impossible (board : Fin 10 × Fin 10) (start : Fin 10 × Fin 10) :
  camel_move_impossible board start :=
by
  -- proof required here
  sorry

end camel_adjacent_impossible_l495_495140


namespace dot_product_example_l495_495655

-- Define the angles
def angle1 : ℝ := 23 * Real.pi / 180
def angle2 : ℝ := 67 * Real.pi / 180
def angle3 : ℝ := 53 * Real.pi / 180
def angle4 : ℝ := 37 * Real.pi / 180

-- Define the vectors
def vec_a : ℝ × ℝ := (Real.cos angle1, Real.cos angle2)
def vec_b : ℝ × ℝ := (Real.cos angle3, Real.cos angle4)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Theorem: the dot product of given vectors
theorem dot_product_example : dot_product vec_a vec_b = (Real.sqrt 3) / 2 := by
  sorry

end dot_product_example_l495_495655


namespace vasya_2000th_hit_l495_495481

theorem vasya_2000th_hit : (let sequence := [1, 2, 3, 4, 5, 6, 5, 4, 3, 2] in
  sequence[(2000 - 1) % sequence.length]) = 2 := 
by
  sorry

end vasya_2000th_hit_l495_495481


namespace inequality_holds_for_all_x_l495_495353

theorem inequality_holds_for_all_x : (∀ x ∈ set.Ioc 0 (1/2 : ℝ), x^2 + 2 * a * x + 1 ≥ 0) → a ≥ -5/4 := by
  sorry

end inequality_holds_for_all_x_l495_495353


namespace geometric_sequence_preserving_functions_l495_495956

def is_geometric_sequence {α : Type*} [LinearOrderedField α] (a : ℕ → α) : Prop :=
∀ n, a n * a (n + 2) = (a (n + 1))^2

def is_gsp_function {α : Type*} [LinearOrderedField α] (f : α → α) : Prop :=
∀ a : ℕ → α, is_geometric_sequence a → is_geometric_sequence (λ n, f (a n))

theorem geometric_sequence_preserving_functions {α : Type*} [LinearOrderedField α] :
  (is_gsp_function (λ x : α, x^2)) ∧ (is_gsp_function (λ x : α, Real.sqrt (abs x))) ∧
  ¬(is_gsp_function (λ x : α, 2^x)) ∧ ¬(is_gsp_function (λ x : α, Real.log (abs x))) :=
by
  sorry

end geometric_sequence_preserving_functions_l495_495956


namespace rearrange_conditions_implies_average_10_l495_495467

def largest_not_last_but_in_first_four (l : List ℤ) := 
  ∃ n, l.indexOf n ≤ 3 ∧ l.indexOf n ≠ l.length - 1 ∧ n = 18

def smallest_not_first_but_in_middle_three (l : List ℤ) := 
  ∃ n, 1 ≤ l.indexOf n ∧ l.indexOf n ≤ 3 ∧ n = -3

def median_not_first_or_last (l : List ℤ) := 
  List.All (fun n => l.indexOf n ≠ 0 ∧ l.indexOf n ≠ l.length - 1) [11, 15]

def average_of_first_and_last_is_10 (l : List ℤ) :=
  let first := l.head
  let last := l.getLast (by simp)
  (first + last) / 2 = 10

theorem rearrange_conditions_implies_average_10 :
  ∀ l : List ℤ, 
    largest_not_last_but_in_first_four l ∧
    smallest_not_first_but_in_middle_three l ∧
    median_not_first_or_last l →
    average_of_first_and_last_is_10 l := by
  sorry

end rearrange_conditions_implies_average_10_l495_495467


namespace area_of_rectangle_l495_495187

theorem area_of_rectangle (w l : ℝ) (h₁ : w = l / 3) (h₂ : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end area_of_rectangle_l495_495187


namespace geometric_series_sum_l495_495567

/-- 
The series is given as 1/2^2 + 1/2^3 + 1/2^4 + 1/2^5 + 1/2^6 + 1/2^7 + 1/2^8.
First term a = 1/4 and common ratio r = 1/2 and number of terms n = 7. 
The sum should be 127/256.
-/
theorem geometric_series_sum :
  let a := 1 / 4
  let r := 1 / 2
  let n := 7
  let S := (a * (1 - r^n)) / (1 - r)
  S = 127 / 256 :=
by
  sorry

end geometric_series_sum_l495_495567


namespace smallest_positive_integer_congruence_l495_495123

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ, x = 26 ∧ 5 * x % 31 = 17 :=
by
  use 26
  split
  exact rfl
  norm_num
  sorry

end smallest_positive_integer_congruence_l495_495123


namespace middle_three_cards_sum_l495_495052

def five_red_cards : List ℕ := [1, 2, 3, 4, 5]
def four_blue_cards : List ℕ := [3, 4, 5, 6]
def alternating_stack (ls : List ℕ) : Prop :=
  ∀ i, i < ls.length - 1 → (ls.nth_le i sorry ∈ five_red_cards ↔ ls.nth_le (i + 1) sorry ∈ four_blue_cards)

def divides_neighbours (ls : List ℕ) : Prop :=
  ∀ i, i < ls.length - 1 →
  ((ls.nth_le i sorry ∈ five_red_cards → ls.nth_le (i+1) sorry % ls.nth_le i sorry = 0) ∧
   (ls.nth_le (i+1) sorry ∈ five_red_cards → ls.nth_le i sorry % ls.nth_le (i+1) sorry = 0))

def valid_stack (ls : List ℕ) : Prop :=
  alternating_stack ls ∧ divides_neighbours ls

theorem middle_three_cards_sum (s : List ℕ) (h1 : valid_stack s) (h2 : s.length = 9):
    s.nth_le 3 sorry + s.nth_le 4 sorry + s.nth_le 5 sorry = 12 :=
sorry

end middle_three_cards_sum_l495_495052


namespace bromine_is_liquid_at_25C_1atm_l495_495127

-- Definitions for the melting and boiling points
def melting_point (element : String) : Float :=
  match element with
  | "Br" => -7.2
  | "Kr" => -157.4 -- Not directly used, but included for completeness
  | "P" => 44.1 -- Not directly used, but included for completeness
  | "Xe" => -111.8 -- Not directly used, but included for completeness
  | _ => 0.0 -- default case; not used

def boiling_point (element : String) : Float :=
  match element with
  | "Br" => 58.8
  | "Kr" => -153.4
  | "P" => 280.5 -- Not directly used, but included for completeness
  | "Xe" => -108.1
  | _ => 0.0 -- default case; not used

-- Define the condition of the problem
def is_liquid_at (element : String) (temperature : Float) (pressure : Float) : Bool :=
  melting_point element < temperature ∧ temperature < boiling_point element

-- Goal statement
theorem bromine_is_liquid_at_25C_1atm : is_liquid_at "Br" 25 1 = true :=
by
  sorry

end bromine_is_liquid_at_25C_1atm_l495_495127


namespace greatest_int_radius_lt_75pi_l495_495695

noncomputable def circle_radius_max (A : ℝ) (π : ℝ) : ℕ :=
  Nat.floor (Real.sqrt (A / π))

theorem greatest_int_radius_lt_75pi :
  circle_radius_max 75 Real.pi = 8 := by
  sorry

end greatest_int_radius_lt_75pi_l495_495695


namespace decorate_eggs_time_calculation_l495_495760

/-- Definition of Mia's and Billy's egg decorating rates, total number of eggs to be decorated, and the calculated time when working together --/
def MiaRate : ℕ := 24
def BillyRate : ℕ := 10
def totalEggs : ℕ := 170
def combinedRate : ℕ := MiaRate + BillyRate

theorem decorate_eggs_time_calculation :
  (totalEggs / combinedRate) = 5 := by
  sorry

end decorate_eggs_time_calculation_l495_495760


namespace final_position_west_total_fuel_consumed_l495_495528

def travelRecords : List Int := [-9, 7, -3, -6, -8, 5]

def east_is_positive_west_is_negative : Int -> Bool := fun dir => dir >= 0

def fuelConsumptionPerKm : Float := 0.1

theorem final_position_west :
  List.sum travelRecords = -14 :=
by
  unfold travelRecords
  norm_num

theorem total_fuel_consumed :
  (List.sum (List.map Int.natAbs travelRecords) : Float) * fuelConsumptionPerKm = 3.8 :=
by
  unfold travelRecords fuelConsumptionPerKm
  norm_num
  -- Additional steps to manage type conversion from Nat to Float if necessary
  sorry

end final_position_west_total_fuel_consumed_l495_495528


namespace construct_tangent_circles_l495_495614

structure Circle (α : Type*) where
  center : α
  radius : ℝ

variables {α : Type*} [EuclideanSpace ℝ α]

def point_inside_circle (A : α) (k : Circle α) : Prop :=
  (∥A - k.center∥ < k.radius)

def ray_originating_from (A : α) : set (line ℝ α) :=
  {l | ∃ θ : ℝ, l = {B | B = A + θ * v} }

theorem construct_tangent_circles
  (k : Circle α) (A : α)
  (e f : set (line ℝ α))
  (H_inside : point_inside_circle A k)
  (H_e_ray : e ∈ ray_originating_from A)
  (H_f_ray : f ∈ ray_originating_from A)
  : ∃ k1 k2 : Circle α, 
      k1.center ∈ (∂ k.center, k.radius) ∧
      k2.center ∈ (∂ k.center, k.radius) ∧
      (∀ B ∈ e ∪ f, ∥B - k1.center∥ = k1.radius) ∧
      (∀ B ∈ e ∪ f, ∥B - k2.center∥ = k2.radius) :=
sorry

end construct_tangent_circles_l495_495614


namespace inequality_solution_l495_495452

theorem inequality_solution :
  {x : ℝ | ((x > 4) ∧ (x < 5)) ∨ ((x > 6) ∧ (x < 7)) ∨ (x > 7)} =
  {x : ℝ | (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0} :=
sorry

end inequality_solution_l495_495452


namespace greatest_integer_radius_l495_495675

theorem greatest_integer_radius (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
begin
  have h1 : r^2 < 75, from (lt_div_iff (real.pi_pos)).mp (by rwa [mul_div_cancel_left _ (ne_of_gt real.pi_pos)]),
  -- To show the highest integer r satisfies r^2 < 75 is 8:
  have : r ≤ nat.floor (real.sqrt 75), 
  { exact nat.le_floor_of_lt (by exact_mod_cast h1) },
  -- Since nat.floor (real.sqrt 75) == 8
  exact le_trans this (by norm_num)
end

end greatest_integer_radius_l495_495675


namespace carrie_weekly_earning_l495_495230

-- Definitions and conditions
def iphone_cost : ℕ := 800
def trade_in_value : ℕ := 240
def weeks_needed : ℕ := 7

-- Calculate the required weekly earning
def weekly_earning : ℕ := (iphone_cost - trade_in_value) / weeks_needed

-- Problem statement: Prove that Carrie makes $80 per week babysitting
theorem carrie_weekly_earning :
  weekly_earning = 80 := by
  sorry

end carrie_weekly_earning_l495_495230


namespace rectangular_field_area_l495_495174

theorem rectangular_field_area :
  ∃ (w l : ℝ), w = l / 3 ∧ 2 * (w + l) = 72 ∧ w * l = 243 :=
by
  sorry

end rectangular_field_area_l495_495174


namespace original_expenditure_mess_l495_495472

theorem original_expenditure_mess
    (x : ℝ) -- original average expenditure per head
    (h_initial_students : 35) 
    (h_new_students : 7) 
    (h_increase_per_day : 84) 
    (h_diminished : 1)
    (h_new_expenditure : 42 * (x - 1)) -- new expenditure per day
    (h_original_expenditure : 35 * x) -- original expenditure per day
  :
    42 * (x - 1) - 35 * x = 84 → 35 * 18 = 630 := 
   -- prove that the original expenditure of the mess was Rs. 630 per day given the conditions.
by
  intros h_eq_x
  let x := 18 -- From solving the equation in steps
  have h_original_expenditure : 35 * 18 = 630 := by
    linarith
  exact h_original_expenditure

end original_expenditure_mess_l495_495472


namespace kevin_expected_away_time_l495_495732

theorem kevin_expected_away_time
  (leak_rate : ℝ)
  (bucket_capacity : ℝ)
  (bucket_factor : ℝ)
  (leak_rate_eq : leak_rate = 1.5)
  (bucket_capacity_eq : bucket_capacity = 36)
  (bucket_factor_eq : bucket_factor = 2)
  : ((bucket_capacity / bucket_factor) / leak_rate) = 12 :=
by
  rw [bucket_capacity_eq, leak_rate_eq, bucket_factor_eq]
  sorry

end kevin_expected_away_time_l495_495732


namespace collinearity_of_centers_l495_495734

noncomputable def touches (Γ : Circle) (l : Line) : Prop := sorry
noncomputable def circumcenter (Δ : Triangle) : Point := sorry
noncomputable def incenter (Δ : Triangle) : Point := sorry
noncomputable def center (Γ : Circle) : Point := sorry

theorem collinearity_of_centers
  (ABC : Triangle)
  (Γ_A Γ_B Γ_C Γ : Circle)
  (h_eq_rad: radius Γ_A = radius Γ_B ∧ radius Γ_B = radius Γ_C)
  (h_disjoint : disjoint Γ_A Γ_B ∧ disjoint Γ_B Γ_C ∧ disjoint Γ_A Γ_C)
  (h_touch_A : touches Γ_A (line_through ABC.A ABC.B) ∧ touches Γ_A (line_through ABC.A ABC.C))
  (h_touch_B : touches Γ_B (line_through ABC.A ABC.B) ∧ touches Γ_B (line_through ABC.B ABC.C))
  (h_touch_C : touches Γ_C (line_through ABC.B ABC.C) ∧ touches Γ_C (line_through ABC.C ABC.A))
  (h_touch_ext : touches Γ Γ_A ∧ touches Γ Γ_B ∧ touches Γ Γ_C):
  let O := circumcenter ABC,
      I := incenter ABC,
      O_1 := center Γ in
  collinear [O, I, O_1] :=
sorry

end collinearity_of_centers_l495_495734


namespace imaginary_part_conjugate_l495_495999

theorem imaginary_part_conjugate (z : ℂ) (h : z = (3 - complex.i) / (3 + complex.i)) :
  complex.im z.conj = 3 / 5 :=
by 
  sorry

end imaginary_part_conjugate_l495_495999


namespace parameter_solution_set_l495_495591

def solution_set (a : ℝ) : set (ℝ × ℝ) :=
  { p : ℝ × ℝ | 
    let x := p.fst, y := p.snd in
    x^2 + (y - a)^2 = 64 ∧ (|x| - 6)^2 + (|y| - 8)^2 = 100 }

def has_exactly_two_solutions (a : ℝ) : Prop :=
  ∃ s1 s2 : ℝ × ℝ, s1 ≠ s2 ∧ ∀ s : ℝ × ℝ, s ∈ solution_set a → s = s1 ∨ s = s2

theorem parameter_solution_set :
  { a : ℝ | has_exactly_two_solutions a } =
  {-8 - 12 * Real.sqrt 2} ∪ (Ioo (-24 : ℝ) (-8)) ∪ (Ico 8 24) ∪ {12 * Real.sqrt 2 + 8} :=
by
  sorry

end parameter_solution_set_l495_495591


namespace smallest_n_for_convex_100gon_l495_495599

def isConvexPolygon (P : List (Real × Real)) : Prop := sorry -- Assumption for polygon convexity
def canBeIntersectedByTriangles (P : List (Real × Real)) (n : ℕ) : Prop := sorry -- Assumption for intersection by n triangles

theorem smallest_n_for_convex_100gon :
  ∀ (P : List (Real × Real)),
  isConvexPolygon P →
  List.length P = 100 →
  (∀ n, canBeIntersectedByTriangles P n → n ≥ 50) ∧ canBeIntersectedByTriangles P 50 :=
sorry

end smallest_n_for_convex_100gon_l495_495599


namespace area_of_BEIH_l495_495502

def calculate_area_of_quadrilateral (A B C D E F I H : (ℝ × ℝ)) : ℝ := 
  sorry

theorem area_of_BEIH : 
  let A := (0, 3)
  let B := (0, 0)
  let C := (3, 0)
  let D := (3, 3)
  let E := (0, 1.5)
  let F := (1, 0)
  let I := (3 / 5, 9 / 5)
  let H := (3 / 4, 3 / 4)
  calculate_area_of_quadrilateral A B C D E F I H = 27 / 40 :=
sorry

end area_of_BEIH_l495_495502


namespace roots_sum_roots_product_algebraic_expression_l495_495304

theorem roots_sum (x1 x2 : ℝ) (h : x1^2 - x1 - 1 = 0 ∧ x2^2 - x2 - 1 = 0) :
  x1 + x2 = 1 :=
sorry

theorem roots_product (x1 x2 : ℝ) (h : x1^2 - x1 - 1 = 0 ∧ x2^2 - x2 - 1 = 0) :
  x1 * x2 = -1 :=
sorry

theorem algebraic_expression (x1 x2 : ℝ) (h : x1^2 - x1 - 1 = 0 ∧ x2^2 - x2 - 1 = 0) :
  x1^2 + x2^2 = 3 :=
sorry

end roots_sum_roots_product_algebraic_expression_l495_495304


namespace license_plates_count_l495_495158

noncomputable def num_license_plates : Nat :=
  let num_w := 26 * 26      -- number of combinations for w
  let num_w_orders := 2     -- two possible orders for w
  let num_digits := 10 ^ 5  -- number of combinations for 5 digits
  let num_positions := 6    -- number of valid positions for w
  2 * num_positions * num_digits * num_w

theorem license_plates_count : num_license_plates = 809280000 := by
  sorry

end license_plates_count_l495_495158


namespace mouse_start_farthest_point_l495_495919

open Real

-- Define conditions
def cheese_location : (ℝ × ℝ) := (9, 15)
def mouse_initial_location : (ℝ × ℝ) := (3, -3)
def mouse_line (x : ℝ) : ℝ := -4 * x + 9

-- Define the theorem
theorem mouse_start_farthest_point :
  let p := (9, 15) in
  let q := (3, -3) in
  let line := (λ x : ℝ, -4 * x + 9) in
  let perpendicular_slope := (1 : ℝ) / 4 in
  ∃ (a b : ℝ),
    let perpendicular_line := (λ x : ℝ, 15 + 1 / 4 * (x - 9)) in
    let intersection_x := (9 - 69 / 4) / (17 / 4) in
    let intersection_y := line intersection_x in
    let a_plus_b := intersection_x + intersection_y in
    a = intersection_x ∧ b = intersection_y ∧ a_plus_b = 252 / 17 :=
sorry

end mouse_start_farthest_point_l495_495919


namespace avg_of_other_40_results_l495_495458

-- Declare basic variables and conditions
variables (x : ℝ) (avg1 avg_total : ℝ)
variable h1 : avg1 = 40
variable h2 : avg_total = 48

-- Prove the average of the other 40 results
theorem avg_of_other_40_results (h1 : avg1 = 40) (h2 : avg_total = 48) : x = 60 :=
sorry

end avg_of_other_40_results_l495_495458


namespace sum_S_le_10000_l495_495389

-- Define the condition: For all positive integers x and y such that x + y ∈ S, either x ∈ S or y ∈ S (or both)
def condition (S : Set ℕ) : Prop :=
  ∀ (x y : ℕ), x > 0 → y > 0 → x + y ∈ S → x ∈ S ∨ y ∈ S

-- Define the problem statement
theorem sum_S_le_10000 (S : Set ℕ) (h_cond : condition S) (h_card : S.to_finset.card = 100) :
  S.to_finset.sum id ≤ 10000 :=
by {
  sorry
}

end sum_S_le_10000_l495_495389


namespace inequality_holds_l495_495423

noncomputable def verify_inequality (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) : Prop :=
  real.sqrt ((a + c) * (b + d)) ≥ real.sqrt (a * b) + real.sqrt (c * d)

theorem inequality_holds (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  verify_inequality a b c d ha hb hc hd :=
by
  sorry

end inequality_holds_l495_495423


namespace count_integers_between_cubes_l495_495333

theorem count_integers_between_cubes (a b : ℝ) (h1 : a = 10.5) (h2 : b = 10.6) : 
  let lower_bound := a^3
  let upper_bound := b^3
  let first_integer := Int.ceil lower_bound
  let last_integer := Int.floor upper_bound
  (last_integer - first_integer + 1) = 33 :=
by
  -- Definitions for clarity
  let lower_bound := a^3
  let upper_bound := b^3
  let first_integer := Int.ceil lower_bound
  let last_integer := Int.floor upper_bound
  
  -- Skipping the proof
  sorry

end count_integers_between_cubes_l495_495333


namespace proving_four_digit_number_l495_495062

def distinct (a b c d : Nat) : Prop :=
a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def same_parity (x y : Nat) : Prop :=
(x % 2 = 0 ∧ y % 2 = 0) ∨ (x % 2 = 1 ∧ y % 2 = 1)

def different_parity (x y : Nat) : Prop :=
¬same_parity x y

theorem proving_four_digit_number :
  ∃ (A B C D : Nat),
    distinct A B C D ∧
    (different_parity A B → B ≠ 4) ∧
    (different_parity B C → C ≠ 3) ∧
    (different_parity C D → D ≠ 2) ∧
    (different_parity D A → A ≠ 1) ∧
    A + D < B + C ∧
    1000 * A + 100 * B + 10 * C + D = 2341 :=
by
  sorry

end proving_four_digit_number_l495_495062


namespace prime_square_minus_seven_l495_495340

theorem prime_square_minus_seven (P : ℕ) (hP_prime : Prime P) (h_prime_P3_plus_9 : Prime (P^3 + 9)) : P^2 - 7 = -3 :=
sorry

end prime_square_minus_seven_l495_495340


namespace flip_and_remaining_single_head_l495_495373

theorem flip_and_remaining_single_head (i j : ℕ) :
    (∀ m n, 0 ≤ m ∧ m < 2007 ∧ 0 ≤ n ∧ n < 2007 →
        (flip_four_consecutive : ℕ → ℕ → Prop) →
        ((m % 4 = 0 ∧ n % 4 = 0) ↔ (tails_except_one : ℕ → ℕ → Prop)))
    ↔ (i % 4 = 0 ∧ j % 4 = 0)
:=
  sorry

end flip_and_remaining_single_head_l495_495373


namespace inequality_solution_l495_495442

noncomputable def solve_inequality (x : ℝ) : Prop :=
  ((x - 3) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0

theorem inequality_solution : {x : ℝ | solve_inequality x} = 
  {x : ℝ | x < 2} ∪ {x : ℝ | 3 < x ∧ x < 4} ∪ {x : ℝ | 5 < x ∧ x < 6} ∪ {x : ℝ | x > 7} :=
by
  sorry

end inequality_solution_l495_495442


namespace lana_total_pages_l495_495003

-- Given conditions
def lanas_initial_pages : Nat := 8
def duanes_total_pages : Nat := 42
def duanes_percentage : Float := 0.7
def alexas_total_pages : Nat := 48
def alexas_percentage : Float := 0.25

-- Define the proof statement
theorem lana_total_pages :
  lanas_initial_pages + (duanes_percentage * duanes_total_pages).toInt + (alexas_percentage * alexas_total_pages).toInt = 49 :=
by
  sorry

end lana_total_pages_l495_495003


namespace bug_visits_at_least_ten_vertices_l495_495146

def dodecahedron : Type := sorry
def vertices (d : dodecahedron) : Finset vertex := sorry
def edges (d : dodecahedron) : Finset (vertex × vertex) := sorry
def moves : ℕ := 12

axiom start_vertex (d : dodecahedron) : vertex
axiom connected_vertices (v : vertex) : Finset vertex

def probability_of_visiting_at_least_ten_unique_vertices_after_twelve_moves (d : dodecahedron) : ℚ :=
  let total_paths := 3 ^ 12
  let favorable_paths := sorry -- combinatorial calculation here
  favorable_paths / total_paths

theorem bug_visits_at_least_ten_vertices (d : dodecahedron) :
  ∃ (M N : ℕ), probability_of_visiting_at_least_ten_unique_vertices_after_twelve_moves(d) = (M : ℚ) / (N : ℚ) := sorry

end bug_visits_at_least_ten_vertices_l495_495146


namespace range_of_a_l495_495031

noncomputable def f (x a : ℝ) := Real.exp (-x) - 2 * x - a

def curve (x : ℝ) := x ^ 3 + x

def y_in_range (x : ℝ) := x >= -2 ∧ x <= 2

theorem range_of_a : ∀ (a : ℝ), (∃ x, y_in_range (curve x) ∧ f (curve x) a = curve x) ↔ a ∈ Set.Icc (Real.exp (-2) - 6) (Real.exp 2 + 6) := by
  sorry

end range_of_a_l495_495031


namespace correct_options_l495_495357

variable (a c : ℝ)
variable (x : ℝ)

def inequality_solution_set : Prop := 
  ∀ x : ℝ, (ax^2 + 2*x + c < 0) ↔ (x < -1 ∨ x > 2)

def option_A : Prop := a + c = 2
def option_B : Prop := ∀ x : ℝ, (ax + c ≤ 0) ↔ (x ≤ 2)
def option_C : Prop := c^(1 / a) = 1 / 2
def option_D : Prop := 
  ∃ x0 : ℝ, (x^2 - 2*a*x + c = 0) ∧ (∀ x1, x^2 - 2*a*x1 + c = 0 → x0 = x1) 

theorem correct_options
    (ineq_sol : inequality_solution_set a c)
    (opt_A : option_A a c)
    (opt_B : ¬ option_B a c)
    (opt_C : option_C a c)
    (opt_D : option_D a c) :
    True := sorry

end correct_options_l495_495357


namespace midpoints_opposite_edges_intersect_at_single_point_l495_495796

structure Tetrahedron (A B C D : Type) :=
  (M : midpoint A B)
  (N : midpoint C D)
  (K : midpoint B C)
  (L : midpoint A D)

theorem midpoints_opposite_edges_intersect_at_single_point
  (A B C D : Type)
  (tet : Tetrahedron A B C D) :
  ∃ P, ∀ M N K L, (tet.M = M) → (tet.N = N) → (tet.K = K) → (tet.L = L) → 
    (line_segment M N).intersect_at P ∧
    (line_segment K L).intersect_at P := sorry

end midpoints_opposite_edges_intersect_at_single_point_l495_495796


namespace range_of_x_l495_495308

theorem range_of_x {y : ℝ} (h : 1 ≤ 4^y - 3 * 2^y + 3 ∧ 4^y - 3 * 2^y + 3 ≤ 7) : 
  y ∈ set.Iic 0 ∨ y ∈ set.Icc 1 2 :=
by
  sorry

end range_of_x_l495_495308


namespace difference_of_fractions_l495_495125

theorem difference_of_fractions (h₁ : 1/10 * 8000 = 800) (h₂ : (1/20) / 100 * 8000 = 4) : 800 - 4 = 796 :=
by
  sorry

end difference_of_fractions_l495_495125


namespace sales_contest_max_difference_l495_495808

theorem sales_contest_max_difference :
  let editorsSales := [6, 8, 9, 5, 2, 1]
  let writersSales := [4, 2, 6, 8, 6, 5]
  let percentage_difference (e w : ℕ) : ℝ := (|e - w| / min e w) * 100
  let differences := list.map₂ (λ e w => percentage_difference e w) editorsSales writersSales
  list.maximum differences = some 400 :=
by 
  let editorsSales := [6, 8, 9, 5, 2, 1]
  let writersSales := [4, 2, 6, 8, 6, 5]
  let percentage_difference (e w : ℕ) : ℝ := (|e - w| / min e w) * 100
  let differences := list.map₂ (λ e w => percentage_difference e w) editorsSales writersSales
  have h1 : differences = [50, 300, 50, 60, 200, 400], by sorry
  have h2 : list.maximum [50, 300, 50, 60, 200, 400] = some 400, by sorry
  exact eq.trans h1 h2

end sales_contest_max_difference_l495_495808


namespace remainder_when_divide_l495_495893

theorem remainder_when_divide (r : ℕ) (hr : r = 123) : r % 15 = 3 :=
by
  have h : r % 15 = 3
  sorry

end remainder_when_divide_l495_495893


namespace correct_operation_l495_495497

theorem correct_operation : (sqrt 3 * sqrt 2 = sqrt 6) :=
by
  exact sqrt_mul sqrt_pos_of_pos sqrt_pos_of_pos sqrt_pos_of_pos

end correct_operation_l495_495497


namespace problem_translation_l495_495010

variables {a : ℕ → ℤ} (S : ℕ → ℤ)

-- Definition of the arithmetic sequence and its sum function
def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ (d : ℤ), ∀ (n m : ℕ), a (n + 1) = a n + d

-- Sum of the first n terms defined recursively
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  if n = 0 then 0 else a n + sum_first_n_terms a (n - 1)

-- Conditions
axiom h1 : is_arithmetic_sequence a
axiom h2 : S 5 > S 6

-- To be proved: Option D does not necessarily hold
theorem problem_translation : ¬(a 3 + a 6 + a 12 < 2 * a 7) := sorry

end problem_translation_l495_495010


namespace all_bulbs_on_from_any_start_l495_495899

-- Define a bulb and its state (on or off)
inductive BulbState
| on
| off

-- Represent the vertices and the center bulb
structure BulbSystem where
  A : Fin 12 → BulbState -- A type bulbs on the regular 24-gon
  B : Fin 12 → BulbState -- B type bulbs on the regular 24-gon
  C : BulbState           -- Center bulb

-- Type of moves
inductive MoveType
| twoVertices (x : Fin 24) (y : Fin 24) -- Two vertices with odd number of vertices between
| threeVertices (x : Fin 24)            -- Three vertices forming an equilateral triangle

-- Function for toggling a bulb's state
def toggle : BulbState → BulbState
| BulbState.on => BulbState.off
| BulbState.off => BulbState.on

-- Function to apply a move on the BulbSystem
def apply_move : BulbSystem → MoveType → BulbSystem
| ⟨A, B, C⟩, MoveType.twoVertices x y =>
    let A' := A.update x (toggle (A x))
    let A'' := A'.update y (toggle (A' y))
    let C' := toggle C
    {A := A'', B := B, C := C'}
| ⟨A, B, C⟩, MoveType.threeVertices x =>
    let A' := A.update x (toggle (A x))
    let A'' := A'.update ((x + 4) % 12) (toggle (A' ((x + 4) % 12)))
    let A''' := A''.update ((x + 8) % 12) (toggle (A'' ((x + 8) % 12)))
    let C' := toggle C
    {A := A''', B := B, C := C'}

-- Proposition: From any starting configuration, we can reach the state where all bulbs are on
theorem all_bulbs_on_from_any_start (initial : BulbSystem) : 
  ∃ moves : List MoveType, 
    let final_state := List.foldl apply_move initial moves
    (∀ i, final_state.A i = BulbState.on) ∧ 
    (∀ i, final_state.B i = BulbState.on) ∧
    final_state.C = BulbState.on :=
sorry

end all_bulbs_on_from_any_start_l495_495899


namespace surface_area_of_sphere_l495_495094

noncomputable def volume : ℝ := 72 * Real.pi

theorem surface_area_of_sphere (r : ℝ) (h : (4 / 3) * Real.pi * r^3 = volume) :
  4 * Real.pi * r^2 = 36 * Real.pi * (Real.cbrt 2)^2 :=
by
  sorry

end surface_area_of_sphere_l495_495094


namespace probability_square_not_touching_vertex_l495_495768

theorem probability_square_not_touching_vertex :
  let total_squares := 64
  let squares_touching_vertices := 16
  let squares_not_touching_vertices := total_squares - squares_touching_vertices
  let probability := (squares_not_touching_vertices : ℚ) / total_squares
  probability = 3 / 4 :=
by
  sorry

end probability_square_not_touching_vertex_l495_495768


namespace added_number_and_common_difference_l495_495475

theorem added_number_and_common_difference (x : ℝ) (d : ℝ) :
  (∃ x, (5 + x) = y₁ ∧ (7 + x) = y₂ ∧ (10 + x) = y₃) →
  (∃ d, ((y₁^2, y₂^2, y₃^2) forms an arithmetic sequence with common difference d)) →
  x = -31 / 8 ∧ (d = 1 / real.sqrt 2 ∨ d = -(1 / real.sqrt 2)) :=
by
  assume h₁ h₂,
  sorry

end added_number_and_common_difference_l495_495475


namespace Sally_age_in_two_years_l495_495270

theorem Sally_age_in_two_years (h : ∀ (t : ℕ), t - 5 = 7 → t = 12) : 12 + 2 = 14 :=
by 
  have age_today := h 12 (by norm_num)
  exact age_today

end Sally_age_in_two_years_l495_495270


namespace equal_distances_l495_495613

variables {A B C D P Q R M: Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace M]
variables [InnerProductGeometrySpace A] [InnerProductGeometrySpace B] [InnerProductGeometrySpace C] [InnerProductGeometrySpace D]
variables [InnerProductGeometrySpace P] [InnerProductGeometrySpace Q] [InnerProductGeometrySpace R] [InnerProductGeometrySpace M]

-- Assume \angle C is 90 degrees
axiom angle_C_is_90 : ∀ (a b c: A), angle a b c = 90

-- Assume D is inside the triangle ΔABC
axiom D_inside_triangle : ∀ (d a b c: D), inside_triangle d a b c

-- Lines AD, BD, and CD intersect BC, CA, and AB at P, Q, and R
axiom AD_intersects_BC_at_P : ∀ (a b c d: A) (p: P), intersect (line.ad a d) (line.bc b c) = p
axiom BD_intersects_CA_at_Q : ∀ (b c a d: B) (q: Q), intersect (line.bd b d) (line.ca c a) = q
axiom CD_intersects_AB_at_R : ∀ (c a b d: C) (r: R), intersect (line.cd c d) (line.ab a b) = r

-- M is the midpoint of \overline{PQ}
axiom M_midpoint_PQ : ∀ (p q m: M), midpoint p q m

-- Given ∠BRP = ∠PRC
axiom angle_BRP_eq_angle_PRC : ∀ (b r p: B) (p r c: P), angle b r p = angle p r c

-- Prove MR = MC
theorem equal_distances {m r c: inner_product_geometry_space} : dist m r = dist m c := by
  sorry

end equal_distances_l495_495613


namespace maximum_area_of_cross_section_l495_495208

noncomputable def maximum_cross_section_area : ℝ :=
  let A := (4 : ℝ, 4, 0)
  let B := (4, -4, 0)
  let C := (-4, -4, 0)
  let D := (-4, 4, 0)
  let E := (4, 4, 2)
  let F := (4, -4, 4)
  let G := (-4, -4, 5)
  let H := (-4, 4, 3)
  let plane := (3 : ℝ, -5, 2, 20)
  let area := 9 * Real.sqrt 38
  area

theorem maximum_area_of_cross_section :
  let A := (4 : ℝ, 4, 0)
  let B := (4, -4, 0)
  let C := (-4, -4, 0)
  let D := (-4, 4, 0)
  let E := (4, 4, 2)
  let F := (4, -4, 4)
  let G := (-4, -4, 5)
  let H := (-4, 4, 3)
  let plane := (3 : ℝ, -5, 2, 20)
  maximum_cross_section_area = 9 * Real.sqrt 38 := sorry

end maximum_area_of_cross_section_l495_495208


namespace athleteA_time_to_complete_race_l495_495894

theorem athleteA_time_to_complete_race
    (v : ℝ)
    (t : ℝ)
    (h1 : v = 1000 / t)
    (h2 : v = 948 / (t + 18)) :
    t = 18000 / 52 := by
  sorry

end athleteA_time_to_complete_race_l495_495894


namespace simplify_logarithms_eq_1_75_l495_495438

def simplify_logarithms : ℝ :=
  let term1 := 1 / (Real.log 3 / Real.log 12 + 1)
  let term2 := 1 / (Real.log 5 / Real.log 20 + 1)
  let term3 := 1 / (Real.log 2 / Real.log 8 + 1)
  term1 + term2 + term3

theorem simplify_logarithms_eq_1_75 : simplify_logarithms = 1.75 :=
  by
  sorry

end simplify_logarithms_eq_1_75_l495_495438


namespace probability_of_symmetry_line_l495_495952

-- Define the setup of the problem
def point := (ℕ × ℕ)

-- Define center point P
def P : point := (6, 6)

-- Define the grid
def grid : set point := { (i, j) | 1 ≤ i ∧ i ≤ 11 ∧ 1 ≤ j ∧ j ≤ 11 }

-- Define symmetric points
def symmetric_points_through_P : set point :=
  { Q | Q ∈ grid ∧
        ((Q.1 = 11 - P.1 + 1 ∧ Q.2 = P.2) ∨  -- Vertical symmetry
         (Q.1 = P.1 ∧ Q.2 = 11 - P.2 + 1) ∨  -- Horizontal symmetry
         (Q.1 - P.1 = P.2 - Q.2) ∨            -- Diagonal symmetry (\ direction)
         (Q.1 + Q.2 = P.1 + P.2)) }           -- Diagonal symmetry (/ direction)

-- Define the proof problem
theorem probability_of_symmetry_line : (symmetric_points_through_P.card : ℚ) / ((grid.erase P).card : ℚ) = 1 / 3 :=
by 
  -- Omitted the proof
  sorry

end probability_of_symmetry_line_l495_495952


namespace problem_part1_problem_part2_problem_part3_l495_495649

-- Conditions
def quadratic_function_exists (a b c : ℝ) : Prop :=
  (∀ x, a * x^2 + b * x + c = 0 → 1 ≤ x ∧ x ≤ 3) ∧ (0 * a^2 + 0 * b + c = 3)

-- Problem statements
theorem problem_part1 (a b c : ℝ) (h : quadratic_function_exists a b c) :
  ∃ (f : ℝ → ℝ), f x = x^2 - 4 * x + 3 := 
sorry

theorem problem_part2 (a b c t : ℝ) (h1 : quadratic_function_exists a b c)
  (h2 : ∀x ∈ set.Icc (-1 : ℝ) (2 : ℝ), (a * x^2 + b * x + c) - (2 * t - 4) * x = 2) :
  t = -1 ∨ t = 1 :=
sorry

theorem problem_part3 (a b c λ : ℝ) (h : quadratic_function_exists a b c)
  (h_ineq : ∀ x ∈ set.Ici (4 : ℝ), λ * (a * x^2 + b * x + c) - (x - 2) > 0) :
  λ > 2/3 :=
sorry

end problem_part1_problem_part2_problem_part3_l495_495649


namespace measure_of_angle_GSD_l495_495720

theorem measure_of_angle_GSD (DOG DGO GOD GSD : ℝ)
  (h1 : DOG = DGO)
  (h2 : GOD = 30)
  (h3 : DGO = DOG)
  (h4 : ∃ S, (GSD = DOG / 2)) :
  GSD = 67.5 :=
by
  sorry

end measure_of_angle_GSD_l495_495720


namespace integral_of_f_l495_495315

def f : ℝ → ℝ :=
  λ x, if -2 ≤ x ∧ x ≤ 0 then real.sqrt (4 - x^2) else if 0 < x ∧ x ≤ 2 then x + 2 else 0

theorem integral_of_f :
  ∫ x in -2..2, f x = real.pi + 6 :=
by
  sorry

end integral_of_f_l495_495315


namespace product_of_possible_N_l495_495225

-- Defining the variables and conditions
variables (A M N : ℕ)

-- Conditions as per the setup
def condition1 : Prop := A = M + N
def condition2 : Prop := A - 10 - (M + 4) = 4 ∨ A - 10 - (M + 4) = -4

-- Main statement to prove
theorem product_of_possible_N : (∃ N : ℕ, (A = M + N) ∧ ((A - 10 - (M + 4) = 4) ∨ (A - 10 - (M + 4) = -4))) → (∃ N1 N2 : ℕ, N1 = 10 ∧ N2 = 18 ∧ N1 * N2 = 180) :=
by
  sorry

end product_of_possible_N_l495_495225


namespace trig_identity_l495_495888

theorem trig_identity (α : ℝ) : 
  (2 * (Real.sin (4 * α))^2 - 1) / 
  (2 * (1 / Real.tan (Real.pi / 4 + 4 * α)) * (Real.cos (5 * Real.pi / 4 - 4 * α))^2) = -1 :=
by
  sorry

end trig_identity_l495_495888


namespace find_BF_length_l495_495377

-- Define the quadrilateral ABCD with the given right angles and points.
variable (A B C D E F : Point)
variable (h₁ : ∠BAC = 90°)
variable (h₂ : ∠ACD = 90°)

-- Define that E and F are on AC
variable (h₃ : E ∈ Line A C)
variable (h₄ : F ∈ Line A C)

-- Define DE and BF as perpendicular to AC
variable (h₅ : Perpendicular Line D E Line A C)
variable (h₆ : Perpendicular Line B F Line A C)

-- Define the given lengths
variable (h₇ : distance A E = 4)
variable (h₈ : distance D E = 6)
variable (h₉ : distance C E = 8)

-- State the theorem to find the length of BF
theorem find_BF_length : distance B F = 72 / 8.5 := 
by sorry

end find_BF_length_l495_495377


namespace car_mpg_in_city_l495_495147

theorem car_mpg_in_city:
  ∃ (h c T : ℝ), 
    (420 = h * T) ∧ 
    (336 = c * T) ∧ 
    (c = h - 6) ∧ 
    (c = 24) :=
by
  sorry

end car_mpg_in_city_l495_495147


namespace table_tennis_ways_l495_495374

/-
Conditions:
1. n (total players) = 6, with Wang Hao and 5 young players.
2. k (players competing) = 3.
3. Wang Hao is not the last player to compete.
-/
theorem table_tennis_ways (WangHao_not_last : True) : 
  ∑ m in {0,1}, Nat.choose 5 2 * (if m = 1 then 2 else 1) + Nat.choose 5 3 = 30 :=
by sorry

end table_tennis_ways_l495_495374


namespace math_problem_l495_495791

def Q_on_diagonal_and_circumcenters (W X Y Z Q O₃ O₄ : Point) : Prop :=
  Q ∈ line AC ∧ WQ > CQ ∧
  circumcenter WZQ O₃ ∧ circumcenter XYQ O₄

def triangle_properties (W X Y Z Q O₃ O₄ : Point) : Prop :=
  WZ = 10 ∧ ∠ O₃ Q O₄ = 150°

noncomputable def find_c_plus_d (W X Y Z Q O₃ O₄ : Point) 
  (cond1 : Q_on_diagonal_and_circumcenters W X Y Z Q O₃ O₄)
  (cond2 : triangle_properties W X Y Z Q O₃ O₄) : Nat :=
  let c := 150
  let d := 50
  c + d

theorem math_problem (W X Y Z Q O₃ O₄ : Point)
  (cond1 : Q_on_diagonal_and_circumcenters W X Y Z Q O₃ O₄)
  (cond2 : triangle_properties W X Y Z Q O₃ O₄) :
  find_c_plus_d W X Y Z Q O₃ O₄ cond1 cond2 = 200 :=
  by sorry

end math_problem_l495_495791


namespace geometric_series_sum_l495_495847

theorem geometric_series_sum :
  let a := (1/4 : ℚ)
  ∧ let r := (1/4 : ℚ)
  ∧ let n := (5 : ℕ)
  → ∑ i in finset.range n, a * (r ^ i) = 341 / 1024 :=
by
  intro a r n
  apply sorry

end geometric_series_sum_l495_495847


namespace distance_from_point_to_line_correct_l495_495262

noncomputable def distance_from_point_to_line : ℝ :=
  let a := (2:ℝ, -2:ℝ, 3:ℝ) in
  let p1 := (1:ℝ, 3:ℝ, -1:ℝ) in
  let p2 := (0:ℝ, 0:ℝ, 2:ℝ) in
  -- Direction vector d
  let d := (p1.1 - p2.1, p1.2 - p2.2, p1.3 - p2.3) in
  -- Distance computation here
  let numerator_sqrt := Real.sqrt ((-45/19)^2 + (23/19)^2 + (-14/19)^2) in
  let denominator := 19 in
  numerator_sqrt / denominator

theorem distance_from_point_to_line_correct :
  distance_from_point_to_line = (Real.sqrt 2750) / 19 :=
by
  -- Proof omitted
  sorry

end distance_from_point_to_line_correct_l495_495262


namespace gain_percentage_calculation_l495_495212

theorem gain_percentage_calculation 
  (C S : ℝ)
  (h1 : 30 * S = 40 * C) :
  (10 * S / (30 * C)) * 100 = 44.44 :=
by
  sorry

end gain_percentage_calculation_l495_495212


namespace number_of_ways_to_choose_providers_l495_495733

theorem number_of_ways_to_choose_providers :
  ∃ (n m k l : ℕ), n = 25 ∧ m = 24 ∧ k = 23 ∧ l = 22 ∧ n * m * k * l = 303600 :=
begin
  use 25,
  use 24,
  use 23,
  use 22,
  repeat { split, refl },
  norm_num,
end

end number_of_ways_to_choose_providers_l495_495733


namespace median_of_data_set_is_one_l495_495459

-- Given definitions and conditions
variables {a b : ℤ}  -- assume a and b are integers

def average_condition : Prop := 1 + 2 + a = 6
def mode_condition : Prop := ∀ x, x ∉ {{-1, a, 1, 2, b} \ {a}}
def target_median : ℤ := 1
def is_median (s : list ℤ) (m : ℤ) : Prop := list.median s = some m

-- Lean 4 statement
theorem median_of_data_set_is_one 
    (h_avg : average_condition)
    (h_mode : mode_condition) : 
    is_median [-1, a, 1, 2, b] target_median :=
sorry

end median_of_data_set_is_one_l495_495459


namespace ping_pong_balls_sold_l495_495471

theorem ping_pong_balls_sold (total_baseballs initial_baseballs initial_pingpong total_baseballs_sold total_balls_left : ℕ)
  (h1 : total_baseballs = 2754)
  (h2 : initial_pingpong = 1938)
  (h3 : total_baseballs_sold = 1095)
  (h4 : total_balls_left = 3021) :
  initial_pingpong - (total_balls_left - (total_baseballs - total_baseballs_sold)) = 576 :=
by sorry

end ping_pong_balls_sold_l495_495471


namespace sum_of_palindromic_primes_less_than_100_l495_495778

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m > 1 ∧ m * m ≤ n → n % m ≠ 0

def is_palindromic_prime (n : ℕ) : Prop := 
  is_prime n ∧ 
  n / 10 = (n % 10) * 10 + n / 10 ∧ 
  is_prime (n % 10 * 10 + n / 10)

theorem sum_of_palindromic_primes_less_than_100 : 
  let palindromic_primes := [11, 13, 17, 31, 37, 71, 73, 79, 97] in
  palindromic_primes.sum = 429 := 
  by 
  sorry

end sum_of_palindromic_primes_less_than_100_l495_495778


namespace sum_squares_sequence_l495_495379

theorem sum_squares_sequence (a : ℕ → ℕ) 
  (h : ∀ n : ℕ, 0 < n → (∑ i in finset.range n, a (i+1)) = 2^n - 1) :
  ∀ n : ℕ, 0 < n → (∑ i in finset.range n, (a (i+1))^2) = (4^n - 1) / 3 :=
by
  intros
  sorry

end sum_squares_sequence_l495_495379


namespace isosceles_obtuse_triangle_smallest_angle_degrees_l495_495561

theorem isosceles_obtuse_triangle_smallest_angle_degrees :
  ∀ (α β γ : ℝ), 
      α = 1.2 * 90 ∧ 
      α > 90 ∧ 
      α + β + γ = 180 ∧ 
      (β = γ ∧ α ≠ β) →
      β = 36.0 :=
by
  intros α β γ h,
  obtain ⟨h1, h2, h3, h4⟩ := h,
  sorry

end isosceles_obtuse_triangle_smallest_angle_degrees_l495_495561


namespace fn_fixed_points_l495_495747

def f1 (x : ℝ) : ℝ := x^2 - 2

def fn (n : ℕ) (x : ℝ) : ℝ :=
  nat.rec_on n (λ x, x) (λ n fn_minus_one x, f1 (fn_minus_one x))

theorem fn_fixed_points (n : ℕ) : 
  ∃ S : finset ℝ, S.card = 2^n ∧ ∀ x ∈ S, fn n x = x := 
sorry

end fn_fixed_points_l495_495747


namespace area_of_rectangle_l495_495188

theorem area_of_rectangle (w l : ℝ) (h₁ : w = l / 3) (h₂ : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end area_of_rectangle_l495_495188


namespace papers_delivered_to_sunday_only_houses_l495_495002

-- Define the number of houses in the route and the days
def houses_in_route : ℕ := 100
def days_monday_to_saturday : ℕ := 6

-- Define the number of customers that do not get the paper on Sunday
def non_customers_sunday : ℕ := 10
def total_papers_per_week : ℕ := 720

-- Define the required number of papers delivered on Sunday to houses that only get the paper on Sunday
def papers_only_on_sunday : ℕ :=
  total_papers_per_week - (houses_in_route * days_monday_to_saturday) - (houses_in_route - non_customers_sunday)

theorem papers_delivered_to_sunday_only_houses : papers_only_on_sunday = 30 :=
by
  sorry

end papers_delivered_to_sunday_only_houses_l495_495002


namespace count_odd_prime_f_l495_495741

def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ i, n % i = 0) (finset.range (n + 1))).sum id

def is_odd_prime (p : ℕ) : Prop :=
  prime p ∧ (p % 2 = 1)

theorem count_odd_prime_f (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 50) :
  (finset.filter (λ n, is_odd_prime (sum_of_divisors n))
    (finset.filter (λ n, 1 ≤ n ∧ n ≤ 50) (finset.range 51))).card = 5 :=
sorry

end count_odd_prime_f_l495_495741


namespace molecular_weight_constant_l495_495486

/-- 
For a compound, its molecular weight remains constant regardless of the number of moles.
The molecular weight is defined as the mass of one mole of the compound.
We can validate this by showing that the total mass of any number of moles (n) of the compound is 
n times the molecular weight.
--/

theorem molecular_weight_constant (compound : Type) 
  (molecular_weight : ℝ)
  (number_of_moles : ℝ)
  (molecular_weight_value : molecular_weight = 3264)
  (number_of_moles_value : number_of_moles = 8) : 
  molecular_weight = 3264 ∧ number_of_moles * molecular_weight = 26112 :=
begin
  sorry
end

end molecular_weight_constant_l495_495486


namespace invalid_transformation_of_equation_l495_495132

theorem invalid_transformation_of_equation (x y m : ℝ) (h : x = y) :
  (m = 0 → (x = y → x / m = y / m)) = false :=
by
  sorry

end invalid_transformation_of_equation_l495_495132


namespace sin_bound_sin_sum_bound_l495_495901

-- Statement for Problem (I)
theorem sin_bound (x : ℝ) : 
  abs (Real.sin x) > 1 / 3 ∨ abs (Real.sin (x + 1)) > 1 / 3 := 
sorry

-- Statement for Problem (II)
theorem sin_sum_bound : 
  ∑ n in Finset.range (29 - 10 + 1), (abs (Real.sin (n + 10)) / (n + 10)) > 1 / 6 := 
sorry

end sin_bound_sin_sum_bound_l495_495901


namespace ellipse_standard_equation_fixed_intersection_point_l495_495296

-- Definitions and conditions from the problem
variables {a b c : ℝ} (e : ℝ) (A B P Q : ℝ × ℝ)

-- Ellipse conditions
def ellipse (a b : ℝ) : set (ℝ × ℝ) := {p | p.1^2 / a^2 + p.2^2 / b^2 = 1}
def point_on_ellipse (a b : ℝ) (p : ℝ × ℝ) : Prop := p ∈ ellipse a b

-- Problem statement
theorem ellipse_standard_equation (a b : ℝ) (e : ℝ) (H1 : a > b) (H2 : b > 0) 
    (H3 : e = √3 / 2) (H4 : point_on_ellipse a b (2, 0)) :
  (a = 2 ∧ b = 1 ∧ ∀ x y, (x, y) ∈ ellipse a b ↔ x^2 / 4 + y^2 = 1) :=
sorry

theorem fixed_intersection_point (a b : ℝ) (P : ℝ × ℝ) (H1 : P = (1, 0))
    (H2 : point_on_ellipse a b (2, 0)) :
  ∃ Q : ℝ × ℝ, Q = (4, 0) ∧ ∀ (A B : ℝ × ℝ), reflection (B, P) (Q, (∀ x y, line_through x y A B (P)))))
    :
  (Q = (4, 0) ∧ fixed_point_line_AB' (P Q (4, 0), line_through AB' Q)) :=
sorry

end ellipse_standard_equation_fixed_intersection_point_l495_495296


namespace geometric_series_sum_l495_495849

theorem geometric_series_sum :
  let a := (1/4 : ℚ)
  ∧ let r := (1/4 : ℚ)
  ∧ let n := (5 : ℕ)
  → ∑ i in finset.range n, a * (r ^ i) = 341 / 1024 :=
by
  intro a r n
  apply sorry

end geometric_series_sum_l495_495849


namespace geometric_series_sum_l495_495845

theorem geometric_series_sum :
  let a := (1/4 : ℚ)
  ∧ let r := (1/4 : ℚ)
  ∧ let n := (5 : ℕ)
  → ∑ i in finset.range n, a * (r ^ i) = 341 / 1024 :=
by
  intro a r n
  apply sorry

end geometric_series_sum_l495_495845


namespace maximize_lambda_absolute_value_l495_495239

noncomputable def Fibonacci_polynomials (n : ℕ) : ℕ → ℤ
| 0     := 1
| 1     := n
| (k + 2) := n * Fibonacci_polynomials k + Fibonacci_polynomials (k + 1)

theorem maximize_lambda_absolute_value :
  ∃ (λ : ℕ → ℤ), (∀ x, x ^ 1000 = ∑ i in finset.range 1001, λ i * Fibonacci_polynomials i x) ∧
  (∀ k, k = 32 → (∀ j, |λ k| ≥ |λ j|)) :=
sorry

end maximize_lambda_absolute_value_l495_495239


namespace find_radius_l495_495153

theorem find_radius (A B C D : Point) (r b c : ℝ)
  (circle1 : Circle A B r) 
  (circle1_intersects_BC_at_D : intersects circle1 B C D)
  (AB_eq_c : dist A B = c) 
  (AC_eq_b : dist A C = b) :
  ∃ R, Circle A D C R ∧ R = (b * r) / c := by
  sorry

end find_radius_l495_495153


namespace andrea_average_distance_per_day_l495_495937

theorem andrea_average_distance_per_day
  (total_distance : ℕ := 168)
  (fraction_completed : ℚ := 3/7)
  (total_days : ℕ := 6)
  (days_completed : ℕ := 3) :
  (total_distance * (1 - fraction_completed) / (total_days - days_completed)) = 32 :=
by sorry

end andrea_average_distance_per_day_l495_495937


namespace log_product_eq_one_l495_495227

noncomputable def log_base (b a : ℝ) := Real.log a / Real.log b

theorem log_product_eq_one :
  log_base 2 3 * log_base 9 4 = 1 := 
by {
  sorry
}

end log_product_eq_one_l495_495227


namespace volume_of_box_ground_area_of_box_l495_495207

-- Given conditions
variable (l w h : ℕ)
variable (hl : l = 20)
variable (hw : w = 15)
variable (hh : h = 5)

-- Define volume and ground area
def volume (l w h : ℕ) : ℕ := l * w * h
def ground_area (l w : ℕ) : ℕ := l * w

-- Theorem to prove the correct volume
theorem volume_of_box : volume l w h = 1500 := by
  rw [hl, hw, hh]
  sorry

-- Theorem to prove the correct ground area
theorem ground_area_of_box : ground_area l w = 300 := by
  rw [hl, hw]
  sorry

end volume_of_box_ground_area_of_box_l495_495207


namespace count_integer_pairs_meeting_conditions_l495_495661

theorem count_integer_pairs_meeting_conditions :
  {p : ℤ × ℤ // let x := p.1 in let y := p.2 in (x^2 + y^2 < 25) ∧ (x^2 + y^2 < 10 * x) ∧ (x^2 + y^2 < 10 * y) }.to_finset.card = 8 := 
sorry

end count_integer_pairs_meeting_conditions_l495_495661


namespace max_value_is_sqrt_41_l495_495751

noncomputable def max_value (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 4) : ℝ := 
  if h : 6 * a + 3 * b + 10 * c ≤ sqrt 41 then sqrt 41 else 0

theorem max_value_is_sqrt_41 (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 4) :
  max_value a b c h = sqrt 41 :=
sorry

end max_value_is_sqrt_41_l495_495751


namespace quadratic_zero_points_probability_l495_495631

theorem quadratic_zero_points_probability :
  let a_values : Set ℤ := {-1, 0, 1, 2}
  let b_values : Set ℤ := {-1, 0, 1, 2}
  let total_combinations := a_values.card * b_values.card
  let zero_points_count := a_values.card * b_values.card - 3 -- Calculated manually as shown in the steps
  let probability := (zero_points_count : ℚ) / total_combinations
  probability = 13/16 :=
by
  sorry

end quadratic_zero_points_probability_l495_495631


namespace distance_AD_btw_41_and_42_l495_495422

noncomputable def distance_between (x y : ℝ × ℝ) : ℝ :=
  Real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)

theorem distance_AD_btw_41_and_42 :
  let A := (0, 0)
  let B := (15, 0)
  let C := (15, 5 * Real.sqrt 3)
  let D := (15, 5 * Real.sqrt 3 + 30)

  41 < distance_between A D ∧ distance_between A D < 42 :=
by
  sorry

end distance_AD_btw_41_and_42_l495_495422


namespace solve_for_y_l495_495440

theorem solve_for_y (y : ℝ) (h : 3 * y ^ (1 / 4) - 5 * (y / y ^ (3 / 4)) = 2 + y ^ (1 / 4)) : y = 16 / 81 :=
by
  sorry

end solve_for_y_l495_495440


namespace incorrect_expressions_l495_495286

noncomputable def x : ℂ := Complex.exp (2 * Real.pi * Complex.I / 4)
noncomputable def y : ℂ := Complex.exp (-2 * Real.pi * Complex.I / 4)

theorem incorrect_expressions :
  ¬(x^6 + y^6 = 2) ∧ ¬(x^{10} + y^{10} = 2) ∧ ¬(x^{14} + y^{14} = 2) :=
by
  sorry

end incorrect_expressions_l495_495286


namespace sum_geometric_series_is_correct_l495_495868

def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_geometric_series_is_correct
  (a r : ℚ) (n : ℕ)
  (h_a : a = 1/4)
  (h_r : r = 1/4)
  (h_n : n = 5) :
  geometric_series_sum a r n = 341 / 1024 :=
by
  rw [h_a, h_r, h_n]
  -- Now we can skip the proof.
  sorry

end sum_geometric_series_is_correct_l495_495868


namespace time_for_plastic_foam_drift_l495_495907

def boat_speed_in_still_water : ℝ := sorry
def speed_of_water_flow : ℝ := sorry
def distance_between_docks : ℝ := sorry

theorem time_for_plastic_foam_drift (x y s t : ℝ) 
(hx : 6 * (x + y) = s)
(hy : 8 * (x - y) = s)
(t_eq : t = s / y) : 
t = 48 := 
sorry

end time_for_plastic_foam_drift_l495_495907


namespace number_of_valid_pairs_l495_495596

theorem number_of_valid_pairs : 
  (∑ a in finset.Icc 10 99,
      (finset.Icc 10 99).filter (λ b, 10 ≤ a + b ∧ a + b ≤ 99)).card = 3240 := 
sorry

end number_of_valid_pairs_l495_495596


namespace domain_of_log_function_l495_495067

theorem domain_of_log_function :
  { x : ℝ | log 3 (x^2 - x - 2) ∈ ℝ } = { x : ℝ | x < -1 ∨ x > 2 } :=
by
  ext x
  simp
  sorry  -- Proof omitted

end domain_of_log_function_l495_495067


namespace solve_system_equations_l495_495059

noncomputable def radians_of_degrees (deg : ℝ) : ℝ :=
  (deg * Real.pi) / 180

theorem solve_system_equations (x y : ℝ) :
  Real.cot x + Real.tan y = 2 ∧ Real.sin x * Real.cos y = 1 / 4 →
  ((x = radians_of_degrees 199.2649305 ∧ y = radians_of_degrees 139.2649305) ∨
   (x = radians_of_degrees 130.7350694 ∧ y = radians_of_degrees 70.7350694)) :=
by
  sorry

end solve_system_equations_l495_495059


namespace rectangular_field_area_l495_495171

theorem rectangular_field_area :
  ∃ (w l : ℝ), w = l / 3 ∧ 2 * (w + l) = 72 ∧ w * l = 243 :=
by
  sorry

end rectangular_field_area_l495_495171


namespace range_of_function_l495_495580

noncomputable def function_range (x : ℝ) : ℝ :=
  (cos x) ^ 2 - 2 * (sin x)

theorem range_of_function :
  ∀ (x : ℝ), -2 ≤ function_range x ∧ function_range x ≤ 2 :=
by
  intros x
  -- Placeholder for factual proof
  sorry

end range_of_function_l495_495580


namespace michael_classes_selection_l495_495761

theorem michael_classes_selection :
  let total_classes := 10
  let advanced_classes := 3
  let required_advanced := 2
  let total_selection := 5
  let remaining_classes := total_classes - advanced_classes
  let remaining_selection := total_selection - required_advanced
  ∃ (ways_advanced : ℕ) (ways_remaining : ℕ),
    ways_advanced = Nat.choose advanced_classes required_advanced ∧
    ways_remaining = Nat.choose remaining_classes remaining_selection ∧
    ways_advanced * ways_remaining = 105 :=
by 
  let total_classes := 10
  let advanced_classes := 3
  let required_advanced := 2
  let total_selection := 5
  let remaining_selection := total_selection - required_advanced
  let remaining_classes := total_classes - advanced_classes
  use Nat.choose advanced_classes required_advanced
  use Nat.choose remaining_classes remaining_selection
  have h1 : Nat.choose advanced_classes required_advanced = 3 := by sorry
  have h2 : Nat.choose remaining_classes remaining_selection = 35 := by sorry
  rw [h1, h2]
  exact mul_eq 105 3 35 sorry

end michael_classes_selection_l495_495761


namespace rectangular_field_area_l495_495200

theorem rectangular_field_area (w : ℕ) (h : ℕ) (P : ℕ) (A : ℕ) (h_length : h = 3 * w) (h_perimeter : 2 * (w + h) = 72) : A = w * h := 
by
  -- Given conditions
  have h_eq : h = 3 * w := h_length
  rewrite [h_eq] at *
  -- Given perimeter equals 8w = 72
  have w_value : w = 9 := sorry
  -- Length calculated as 3w = 27
  have h_value : h = 27 := sorry
  -- Now we calculate the area
  calc
    A = w * h : by sorry
    ... = 9 * 27 : by sorry
    ... = 243 : by sorry

end rectangular_field_area_l495_495200


namespace number_of_streams_l495_495376

structure Lake :=
  (S A B C D : Type)

structure Transition :=
  (transition : Lake → Lake → Prop)
  (transitive : ∀ l1 l2, transition l1 l2 → ∃ l3, transition l1 l3 ∧ transition l3 l2)

noncomputable def fish_move (l : Lake) (t : Transition) (start : l.S) ([] : List l.S) : l.S :=
  by sorry

theorem number_of_streams (l : Lake) (t : Transition) 
  (condition : (∀ fish, fish_move l t l.S fish = l.S → 375 / 1000)
           ∧ (∀ fish, fish_move l t l.S fish = l.B → 625 / 1000))
  (S_not_reach_others : ∀ s, s ≠ l.S → s ≠ l.B)
  (B_routes : ∃ l', t.transition l.S l.A ∧ t.transition l.A l.B) : 
  ∃ n, n = 3 :=
by
  sorry

end number_of_streams_l495_495376


namespace productivity_increase_l495_495107

theorem productivity_increase (planned_productivity : ℝ) (new_productivity : ℝ) (total_area : ℝ) (initial_days : ℕ) (final_days : ℕ) :
  planned_productivity = 15 →
  new_productivity > planned_productivity →
  total_area = 20 →
  initial_days = 20 / 15 →  -- Since initial planned days would have been ceil(20/15) = ceil(1.333) = 2 days
  final_days = initial_days - 1 →
  final_days = 1 →

  let percentage_increase := ((new_productivity - planned_productivity) / planned_productivity) * 100 in
  percentage_increase = 33.33 := by sorry

end productivity_increase_l495_495107


namespace total_money_spent_on_clothing_l495_495414

theorem total_money_spent_on_clothing (cost_shirt cost_jacket : ℝ)
  (h_shirt : cost_shirt = 13.04) (h_jacket : cost_jacket = 12.27) :
  cost_shirt + cost_jacket = 25.31 :=
sorry

end total_money_spent_on_clothing_l495_495414


namespace inequality_solution_l495_495446

theorem inequality_solution
  (x : ℝ) :
  x ∉ {2, 3, 4, 5, 6, 7} →
  ((x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0 ↔ 
  (x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 6) ∨ (7 < x)) :=
by
  -- Proof goes here
  sorry

end inequality_solution_l495_495446


namespace angles_symmetric_about_y_axis_l495_495310

theorem angles_symmetric_about_y_axis (α β : ℝ) (k : ℤ) (h : β = (2 * ↑k + 1) * Real.pi - α) : 
  α + β = (2 * ↑k + 1) * Real.pi :=
sorry

end angles_symmetric_about_y_axis_l495_495310


namespace egmo_task_problem_l495_495827

-- Define the problem
def scores := ℕ × ℕ × ℕ

theorem egmo_task_problem:
  ∀ (students : Fin 49 → scores), ∃ (s1 s2 : Fin 49), s1 ≠ s2 ∧
  (students s1).1 ≤ (students s2).1 ∧
  (students s1).2 ≤ (students s2).2 ∧
  (students s1).3 ≤ (students s2).3 := 
by
  -- Proof will be provided here
  sorry

end egmo_task_problem_l495_495827


namespace sum_geometric_series_is_correct_l495_495866

def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_geometric_series_is_correct
  (a r : ℚ) (n : ℕ)
  (h_a : a = 1/4)
  (h_r : r = 1/4)
  (h_n : n = 5) :
  geometric_series_sum a r n = 341 / 1024 :=
by
  rw [h_a, h_r, h_n]
  -- Now we can skip the proof.
  sorry

end sum_geometric_series_is_correct_l495_495866


namespace system_has_one_solution_iff_a_eq_zero_l495_495960

-- Define the system of equations
def system (a x y : ℝ) : Prop :=
  x + y = 144 ∧ xy = 5184 + 0.1 * a^2

-- State the theorem
theorem system_has_one_solution_iff_a_eq_zero {a x y : ℝ} :
  (∃ x y, system a x y) ↔ a = 0 :=
sorry

end system_has_one_solution_iff_a_eq_zero_l495_495960


namespace find_complex_z_l495_495624

open Complex

theorem find_complex_z (z : ℂ) (h : (1 - 2 * I) * conj(z) = 5 + 10 * I) : z = -3 - 4 * I :=
sorry

end find_complex_z_l495_495624


namespace circle_locus_l495_495839

noncomputable
def geometric_locus (A B C : Point) (Ω : Circle) (r : ℝ) :=
  centroid A B C

theorem circle_locus
  (A B C : Point)
  (Ω : Circle)
  {M : Point}
  (hM : M ∈ Ω)
  (G : Point := centroid A B C)
  (H : Point := orthocenter A B C)
  (N : Point := λ GM, N): -- missing some definition here
  locus_of_N (A B C : Point) (Ω : Circle) :=
  equidistant_location H :: set (point_space) (radius (circumscribed_circle(A B C)) * 2) :=
sorry

end circle_locus_l495_495839


namespace count_integers_between_cubes_l495_495334

theorem count_integers_between_cubes (a b : ℝ) (h1 : a = 10.5) (h2 : b = 10.6) : 
  let lower_bound := a^3
  let upper_bound := b^3
  let first_integer := Int.ceil lower_bound
  let last_integer := Int.floor upper_bound
  (last_integer - first_integer + 1) = 33 :=
by
  -- Definitions for clarity
  let lower_bound := a^3
  let upper_bound := b^3
  let first_integer := Int.ceil lower_bound
  let last_integer := Int.floor upper_bound
  
  -- Skipping the proof
  sorry

end count_integers_between_cubes_l495_495334


namespace smaller_angle_at_7_15_l495_495224

theorem smaller_angle_at_7_15 
  (hour_hand_rate : ℕ → ℝ)
  (minute_hand_rate : ℕ → ℝ)
  (hour_time : ℕ)
  (minute_time : ℕ)
  (top_pos : ℝ)
  (smaller_angle : ℝ) 
  (h1 : hour_hand_rate hour_time + (minute_time/60) * hour_hand_rate hour_time = 217.5)
  (h2 : minute_hand_rate minute_time = 90.0)
  (h3 : |217.5 - 90.0| = smaller_angle) :
  smaller_angle = 127.5 :=
by
  sorry

end smaller_angle_at_7_15_l495_495224


namespace speech_combinations_l495_495368

def combin (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem speech_combinations:
  let companyA_reps := 2 in
  let other_companies := 4 in
  let case1 := combin companyA_reps 1 * combin other_companies 2 in
  let case2 := combin other_companies 3 in
  case1 + case2 = 16 :=
by
  unfold combin
  rw [mul_comm, mul_assoc]
  unfold Nat.factorial
  rw [div_eq_iff]
  simp
  exact Nat.factorial_ne_zero 1
  exact Nat.fac_eq 4
  sorry

end speech_combinations_l495_495368


namespace equilateral_triangle_circumcircle_l495_495735

theorem equilateral_triangle_circumcircle {A B C Q P : Point} (h_equilateral : equilateral_triangle A B C)
  (hQ_on_BC : Q ∈ segment B C) (hP_on_circumcircle : P ∈ circumcircle A B C ∧ is_intersection P (line A Q)) :
  1 / (dist P Q) = 1 / (dist P B) + 1 / (dist P C) :=
by sorry

end equilateral_triangle_circumcircle_l495_495735


namespace find_S17_l495_495744

noncomputable def S (n : ℕ) := (a₁ : ℝ) (d : ℝ) (S : ℕ → ℝ) := (n / 2) * (2 * a₁ + (n - 1) * d)

theorem find_S17 (a₁ : ℝ) (d : ℝ) (S : ℕ → ℝ) 
  (h₁ : d = 1)
  (h₂ : S 9 - S 4 = 10) 
  (h₃ : ∀ n : ℕ, S n = (n / 2) * (2 * a₁ + (n - 1) * d)) :
  S 17 = 68 :=
by
  sorry

end find_S17_l495_495744


namespace conclusion1_conclusion2_conclusion3_l495_495478

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : ⟪a, b⟫ = 0)

-- The following theorems state the conclusions that need to be proven

theorem conclusion1 : ⟪a, b⟫ = 0 :=
by sorry

theorem conclusion2 : ∥a + b∥ = ∥a - b∥ :=
by sorry

theorem conclusion3 : ∥a∥^2 + ∥b∥^2 = ∥a + b∥^2 :=
by sorry

end conclusion1_conclusion2_conclusion3_l495_495478


namespace two_pairs_more_likely_l495_495838

-- Definitions based on conditions
def total_outcomes : ℕ :=
  6 ^ 5

def three_of_a_kind : ℕ :=
  (Nat.choose 5 3) * 6 * 5 * 4

def two_pairs : ℕ :=
  5 * 6 * (Nat.choose 4 2) * 5 * 4

theorem two_pairs_more_likely :
  (two_pairs : ℚ) / total_outcomes > (three_of_a_kind : ℚ) / total_outcomes :=
  sorry

end two_pairs_more_likely_l495_495838


namespace integer_root_possibilities_l495_495232

noncomputable def possible_integer_roots (b c d e f : ℤ) : set ℕ :=
  {n | ∃ (r₁ r₂ r₃ r₄ r₅ : ℤ), 
      (x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) = (x - r₁) * (x - r₂) * (x - r₃) * (x - r₄) * (x - r₅)
      ∧ multiset.card (multiset.of_list [r₁, r₂, r₃, r₄, r₅])
      ∈ ∈ {0, 1, 2, 3, 4, 5} }

theorem integer_root_possibilities (b c d e f : ℤ) :
  possible_integer_roots b c d e f = {0, 1, 2, 3, 4, 5} :=
sorry

end integer_root_possibilities_l495_495232


namespace quadratic_solutions_l495_495060

theorem quadratic_solutions (x : ℝ) :
  (4 * x^2 - 6 * x = 0) ↔ (x = 0) ∨ (x = 3 / 2) :=
sorry

end quadratic_solutions_l495_495060


namespace restaurant_tip_difference_l495_495133

theorem restaurant_tip_difference
  (a b : ℝ)
  (h1 : 0.15 * a = 3)
  (h2 : 0.25 * b = 3)
  : a - b = 8 := 
sorry

end restaurant_tip_difference_l495_495133


namespace books_returned_percentage_l495_495163

def percentage_books_returned (initial_books final_books loaned_out : ℕ) : ℕ :=
  let books_left = initial_books - final_books
  let books_returned = loaned_out - books_left
  (books_returned * 100) / loaned_out

theorem books_returned_percentage
  (initial_books : ℕ) (final_books : ℕ) (loaned_out : ℕ)
  (h_initial : initial_books = 75)
  (h_final : final_books = 66)
  (h_loaned : loaned_out = 29) :
  percentage_books_returned initial_books final_books loaned_out = 70 := 
by
  sorry

end books_returned_percentage_l495_495163


namespace solution_set_of_inequality_l495_495086

theorem solution_set_of_inequality (x : ℝ) : 
  (x^2 - abs x - 2 < 0) ↔ (-2 < x ∧ x < 2) := 
sorry

end solution_set_of_inequality_l495_495086


namespace sum_of_palindromic_primes_less_than_100_l495_495777

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m > 1 ∧ m * m ≤ n → n % m ≠ 0

def is_palindromic_prime (n : ℕ) : Prop := 
  is_prime n ∧ 
  n / 10 = (n % 10) * 10 + n / 10 ∧ 
  is_prime (n % 10 * 10 + n / 10)

theorem sum_of_palindromic_primes_less_than_100 : 
  let palindromic_primes := [11, 13, 17, 31, 37, 71, 73, 79, 97] in
  palindromic_primes.sum = 429 := 
  by 
  sorry

end sum_of_palindromic_primes_less_than_100_l495_495777


namespace tan_double_angle_l495_495612

theorem tan_double_angle 
  (x : ℝ) 
  (h1 : x ∈ Ioo (- (π / 2)) 0) 
  (h2 : Real.sin x = - (3 / 5)) : 
  Real.tan (2 * x) = - (24 / 7) :=
by 
  sorry

end tan_double_angle_l495_495612


namespace profit_difference_l495_495169

noncomputable def ratio_X := 3 / 5
noncomputable def ratio_Y := 2 / 3
noncomputable def ratio_Z := 3 / 7

theorem profit_difference :
  let total_profit := 4000
      total_ratio := (63 + 70 + 45 : ℝ)
      part_value := total_profit / total_ratio
      share_X := (63 / total_ratio) * total_profit
      share_Y := (70 / total_ratio) * total_profit
      share_Z := (45 / total_ratio) * total_profit
  in share_Y - share_Z = 561.80 :=
by
  sorry

end profit_difference_l495_495169


namespace rectangle_area_l495_495204

theorem rectangle_area (w l : ℝ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end rectangle_area_l495_495204


namespace Wolfgang_marble_count_l495_495474

theorem Wolfgang_marble_count
  (W L M : ℝ)
  (hL : L = 5/4 * W)
  (hM : M = 2/3 * (W + L))
  (hTotal : W + L + M = 60) :
  W = 16 :=
by {
  sorry
}

end Wolfgang_marble_count_l495_495474


namespace imaginary_condition_l495_495811

theorem imaginary_condition (m : ℝ) : (3 * m - 2 = 0) → m = ⅔ := by
  intro h
  sorry

end imaginary_condition_l495_495811


namespace curve_C1_polar_equation_curve_C2_cartesian_equation_distance_AB_l495_495717

noncomputable def parametric_curve_C1 (θ : ℝ) : ℝ × ℝ :=
  (1 + Real.cos θ, Real.sin θ)

noncomputable def polar_curve_C2 (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * Real.sin θ = 3

noncomputable def θ_line (θ : ℝ) : ℝ := θ

theorem curve_C1_polar_equation :
  ∀ θ, (parametric_curve_C1 θ).1^2 + (parametric_curve_C1 θ).2^2 = 2 * (parametric_curve_C1 θ).1 :=
  sorry

theorem curve_C2_cartesian_equation :
  ∀ ρ θ, polar_curve_C2 ρ θ → 
    (ρ * Real.cos θ)^2 + (ρ * Real.sin θ - 2)^2 = 7 :=
  sorry

theorem distance_AB :
  let ρ1 := 1
  let ρ2 := Real.sqrt 3 + Real.sqrt 6
  ∀ θ, θ = π / 3 → big.OccupyΑ (θ) =
    |ρ1 - ρ2| = Real.sqrt 6 + Real.sqrt 3 - 1 :=
  sorry

end curve_C1_polar_equation_curve_C2_cartesian_equation_distance_AB_l495_495717


namespace minimum_sum_L_l495_495026

theorem minimum_sum_L (a : Fin 100 → Fin 100)
  (h_perm : ∀ i, ∃ j, a i = j + 1 ∧ ∀ k, a k = j → i = k) :
  ∑ k in (Finset.range 100), L (k, a) = 715 :=
sorry

-- Definitions of the conditions
def L (k : ℕ) (a : Fin n → Fin (n + 1)) : ℕ :=
  (max (longest_increasing_subseq (k, a)) 
       (longest_decreasing_subseq (k, a)))

def longest_increasing_subseq : ℕ → ℕ
  -- definition skipped

def longest_decreasing_subseq : ℕ → ℕ
  -- definition skipped

end minimum_sum_L_l495_495026


namespace find_values_of_p_q_r_s_l495_495750

noncomputable theory

variables (A B C : ℝ)
variables (p q r s : ℕ)
variables (cos : ℝ → ℝ)
variables (sin : ℝ → ℝ)
variables (is_right_angle : B = π / 2)

-- Assuming the required conditions
axiom cos_A_cos_B_sin_A_sin_B_cos_C :
  cos A ^ 2 + cos B ^ 2 + 2 * sin A * sin B * cos C = 16 / 9

axiom sin_B_sin_C_cos_B_cos_C_sin_A :
  sin B ^ 2 + sin C ^ 2 + 2 * cos B * cos C * sin A = 25 / 16

axiom right_angle_B : cos B = 0 ∧ sin B = 1

-- Problem statement to prove
theorem find_values_of_p_q_r_s :
  ∃ p q r s : ℕ, 
 (cos C ^ 2 + cos A ^ 2 + 2 * sin C * sin A * sin B = ((p - q * sqrt r) : ℝ) / s) ∧
 gcd (p + q) s = 1 ∧ 
 ∀ k : ℕ, r ≠ k ^ 2 ∧ 
 (1 : ℕ) + (0 : ℕ) + (1 : ℕ) + (1 : ℕ) = 3 :=
begin
  sorry
end

end find_values_of_p_q_r_s_l495_495750


namespace book_page_count_l495_495501

theorem book_page_count (pages_per_night : ℝ) (nights : ℝ) : pages_per_night = 120.0 → nights = 10.0 → pages_per_night * nights = 1200.0 :=
by
  sorry

end book_page_count_l495_495501


namespace combined_area_ratio_l495_495542

theorem combined_area_ratio (s : ℝ) (h₁ : s > 0) : 
  let r := s / 2
  let area_semicircle := (1/2) * π * r^2
  let area_quarter_circle := (1/4) * π * r^2
  let area_square := s^2
  let combined_area := area_semicircle + area_quarter_circle
  let ratio := combined_area / area_square
  ratio = 3 * π / 16 :=
by
  sorry

end combined_area_ratio_l495_495542


namespace max_a2m_value_l495_495618

noncomputable def a_seq (n : ℕ) : ℕ := sorry  -- Define the sequence a_n appropriately
noncomputable def max_value_a2m (m : ℕ) : ℕ := 2^m - 1

theorem max_a2m_value (m : ℕ) (h : 0 < m) : ∀ n k, 
    a_seq n > 0 ∧ a_seq 1 = 1 ∧ 
    a_seq n ∣ (a_seq k + a_seq (k+1) + ... + a_seq (k+n-1)) → 
    max (a_seq (2*m)) = max_value_a2m m := 
by sorry

end max_a2m_value_l495_495618


namespace geometric_series_sum_l495_495858

theorem geometric_series_sum
  (a r : ℚ) (n : ℕ)
  (ha : a = 1/4)
  (hr : r = 1/4)
  (hn : n = 5) :
  (∑ i in finset.range n, a * r^i) = 341 / 1024 := by
  sorry

end geometric_series_sum_l495_495858


namespace sum_of_middle_three_cards_l495_495050

theorem sum_of_middle_three_cards : ∃ (R B : list ℕ), 
  (R = [1, 2, 3, 4, 5] ∧ B = [3, 4, 5, 6]) →
  (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 5 → R.nth i ≠ none → 
  (∃ (j : ℕ), 1 ≤ j ∧ j ≤ 4 ∧ B.nth j ≠ none ∧ (¬ (j = 0 ∧ i = 5) ∧ ¬ (j = 4 ∧ i = 5)) ∧ R.nth i = B.nth j)) →
  R.nth 1 == some 1 →
  R.nth 4 == some 4 →
  (R.nth 3 == some 3 ∧ B.nth 2 == some 3 ∧ B.nth 2 == some 6) →
  ∑ (R, B), R = 5 ∧ B = 6.

end sum_of_middle_three_cards_l495_495050


namespace solve_for_a_l495_495629

theorem solve_for_a (a : Real) (h_pos : a > 0) (h_eq : (fun x => x^2 + 4) ((fun x => x^2 - 2) a) = 18) : 
  a = Real.sqrt (Real.sqrt 14 + 2) := by 
  sorry

end solve_for_a_l495_495629


namespace total_cages_l495_495534

theorem total_cages (initial_puppies initial_kittens initial_birds : ℕ) 
                    (sold_puppies sold_kittens sold_birds : ℕ) 
                    (puppies_per_cage kittens_per_cage birds_per_cage : ℕ) :
  let remaining_puppies := initial_puppies - sold_puppies
  let remaining_kittens := initial_kittens - sold_kittens
  let remaining_birds   := initial_birds - sold_birds
  let puppy_cages := remaining_puppies / puppies_per_cage
  let kitten_cages := remaining_kittens / kittens_per_cage
  let bird_cages := nat.ceil (remaining_birds / (birds_per_cage : ℝ)) in
  initial_puppies = 13 ∧ initial_kittens = 10 ∧ initial_birds = 15 ∧
  sold_puppies = 7 ∧ sold_kittens = 4 ∧ sold_birds = 5 ∧
  puppies_per_cage = 2 ∧ kittens_per_cage = 3 ∧ birds_per_cage = 4 →
  puppy_cages + kitten_cages + bird_cages = 8 :=
by
  sorry

end total_cages_l495_495534


namespace negation_of_exists_l495_495465

theorem negation_of_exists (x : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - x + 1 > 0) :=
by
  sorry

end negation_of_exists_l495_495465


namespace distinct_or_rational_l495_495298

theorem distinct_or_rational (S : set ℕ) (hS : ∀ (n : ℕ), n ∈ S → 0 < n):
  (∃ (F G : finset ℕ), F ≠ G ∧ ∀ x ∈ F, x ∈ S ∧ ∀ x ∈ G, x ∈ S ∧ ∑ x in F, (1 : ℚ) / x = ∑ x in G, (1 : ℚ) / x) ∨
  (∃ (r : ℚ), 0 < r ∧ r < 1 ∧ ∀ (F : finset ℕ), (∀ x ∈ F, x ∈ S) → ∑ x in F, (1 : ℚ) / x ≠ r) :=
by
  sorry

end distinct_or_rational_l495_495298


namespace prime_p_and_cube_l495_495102

noncomputable def p : ℕ := 307

theorem prime_p_and_cube (a : ℕ) (h : a^3 = 16 * p + 1) : 
  Nat.Prime p := by
  sorry

end prime_p_and_cube_l495_495102


namespace lenny_boxes_of_pens_l495_495004

namespace PensProblem

def numberOfPensLeftAfterDonations (P : ℕ) : ℕ :=
  (0.60 * P - 0.25 * (0.60 * P)).natAbs

def numberOfBoxesOfPens (P : ℕ) (pensPerBox : ℕ) : ℕ :=
  P / pensPerBox

theorem lenny_boxes_of_pens (P pens_per_box : ℕ) 
  (h1 : pens_per_box = 5)
  (h2 : (0.60 * P).natAbs - 0.25 * (0.60 * P).natAbs = 45)
  : numberOfBoxesOfPens P pens_per_box = 20 := by
  sorry

end PensProblem

end lenny_boxes_of_pens_l495_495004


namespace inequality_solution_l495_495450

theorem inequality_solution :
  {x : ℝ | ((x > 4) ∧ (x < 5)) ∨ ((x > 6) ∧ (x < 7)) ∨ (x > 7)} =
  {x : ℝ | (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0} :=
sorry

end inequality_solution_l495_495450


namespace find_f_neg_1_l495_495743

-- Define the function f and its properties
def f (x : ℝ) : ℝ := if x >= 0 then 4 * x - 1 else -(4 * (-x) - 1)

-- Prove that f(-1) = -3
theorem find_f_neg_1 : f (-1) = -3 := by
  sorry

end find_f_neg_1_l495_495743


namespace least_k_l495_495977

def f (t : ℝ) : ℝ := t^2 - t + 1

lemma least_k_property (k : ℝ) (x y z : ℝ) (h : ¬ (0 < x ∧ 0 < y ∧ 0 < z)) :
  k * f x * f y * f z ≥ f (xyz) :=
sorry

theorem least_k : ∃ k, (∀ x y z : ℝ, ¬ (0 < x ∧ 0 < y ∧ 0 < z) → k * f x * f y * f z ≥ f (xyz)) ∧ k = 16 / 9 :=
begin
  use 16 / 9,
  intros x y z h,
  exact least_k_property (16 / 9) x y z h,
end

end least_k_l495_495977


namespace difference_heaviest_lightest_box_total_excess_weight_total_selling_price_l495_495469

theorem difference_heaviest_lightest_box :
  let differences := [(-1.5, 2), (-1, 6), (-0.5, 10), (1, 8), (2, 4)] in
  let heaviest := 2 in
  let lightest := -1.5 in
  heaviest - lightest = 3.5 := by sorry

theorem total_excess_weight :
  let differences := [(-1.5, 2), (-1, 6), (-0.5, 10), (1, 8), (2, 4)] in
  let total_excess := List.foldr (λ (pair : (ℝ × ℕ)) acc => acc + pair.1 * pair.2) 0 differences in
  total_excess = 2 := by sorry

theorem total_selling_price :
  let standard_weight := 20 in
  let num_boxes := 30 in
  let differences := [(-1.5, 2), (-1, 6), (-0.5, 10), (1, 8), (2, 4)] in
  let total_excess := List.foldr (λ (pair : (ℝ × ℕ)) acc => acc + pair.1 * pair.2) 0 differences in
  let total_weight := num_boxes * standard_weight + total_excess in
  let price_per_kg := 6 in
  total_weight * price_per_kg = 3612 := by sorry

end difference_heaviest_lightest_box_total_excess_weight_total_selling_price_l495_495469


namespace geometric_series_sum_l495_495862

theorem geometric_series_sum :
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  (a + a * r + a * r^2 + a * r^3 + a * r^4) = 341 / 1024 :=
by
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  -- Proof steps go here
  sorry

end geometric_series_sum_l495_495862


namespace parity_of_expression_l495_495394

theorem parity_of_expression
  (a b c : ℕ) 
  (h_a_odd : a % 2 = 1) 
  (h_b_odd : b % 2 = 1) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0)
  (h_c_pos : c > 0) :
  ((3^a + (b + 2)^2 * c) % 2 = 1 ↔ c % 2 = 0) ∧ 
  ((3^a + (b + 2)^2 * c) % 2 = 0 ↔ c % 2 = 1) :=
by sorry

end parity_of_expression_l495_495394


namespace sum_of_x_values_sum_of_solutions_proof_l495_495980

theorem sum_of_x_values (x : ℝ) (h1 : 0 < x) (h2 : x < 90) 
  (h3 : Real.sin (3 * Real.pi * x / 180)^3 + Real.sin (5 * Real.pi * x / 180)^3 = 
            8 * Real.sin (4 * Real.pi * x / 180)^3 * Real.sin (Real.pi * x / 180)^3) :
  x = 60 ∨ x = 45 ∨ x = 36 := 
sorry

theorem sum_of_solutions : ℝ := 
begin
  have h1 : sum_of_x_values 60, sorry,
  have h2 : sum_of_x_values 45, sorry,
  have h3 : sum_of_x_values 36, sorry,
  exact h1 + h2 + h3
end

theorem proof : sum_of_solutions = 141 := sorry

end sum_of_x_values_sum_of_solutions_proof_l495_495980


namespace average_additional_minutes_per_day_l495_495712

def daily_differences : List ℤ := [20, 5, -5, 0, 15, -10, 10]

theorem average_additional_minutes_per_day :
  (List.sum daily_differences / daily_differences.length) = 5 := by
  sorry

end average_additional_minutes_per_day_l495_495712


namespace part_one_part_two_l495_495317

def f (x : ℝ) : ℝ := |x| + |x - 1|

theorem part_one (m : ℝ) (h : ∀ x, f x ≥ |m - 1|) : m ≤ 2 := by
  sorry

theorem part_two (a b : ℝ) (M : ℝ) (ha : 0 < a) (hb : 0 < b) (hM : a^2 + b^2 = M) (hM_value : M = 2) : a + b ≥ 2 * a * b := by
  sorry

end part_one_part_two_l495_495317


namespace john_total_spent_l495_495728

/-- John's expenditure calculations -/
theorem john_total_spent :
  let computer_cost := 1500
  let peripherals_cost := computer_cost / 5
  let original_video_card_cost := 300
  let upgraded_video_card_cost := original_video_card_cost * 2
  let additional_upgrade_cost := upgraded_video_card_cost - original_video_card_cost
  let total_spent := computer_cost + peripherals_cost + additional_upgrade_cost
  total_spent = 2100 :=
by
  sorry

end john_total_spent_l495_495728


namespace minimum_a_plus_2b_no_a_b_such_that_l495_495991

noncomputable def minimum_value (a b : ℝ) :=
  a + 2 * b

theorem minimum_a_plus_2b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a * b = a + 2 * b + 3) : 
  minimum_value a b ≥ 6 :=
sorry

theorem no_a_b_such_that (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a * b = a + 2 * b + 3) : 
  a^2 + 4 * b^2 ≠ 17 :=
sorry

end minimum_a_plus_2b_no_a_b_such_that_l495_495991


namespace find_y_coordinate_and_sum_l495_495400

def A : (ℝ × ℝ) := (-4, 0)
def B : (ℝ × ℝ) := (-1, 2)
def C : (ℝ × ℝ) := (1, 2)
def D : (ℝ × ℝ) := (4, 0)

def distance (P Q : ℝ × ℝ) : ℝ := 
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem find_y_coordinate_and_sum :
  ∃ y : ℝ, ∃ a b c d : ℝ, 
  (∀ P : ℝ × ℝ, distance P A + distance P D = 10 → 
    distance P B + distance P C = 10 → 
    P.2 = y ∧ y = ( -a + b * real.sqrt c ) / d ) ∧ 
  a = 12 ∧ b = 12 ∧ c = 6 ∧ d = 5 ∧ a + b + c + d = 35 :=
sorry

end find_y_coordinate_and_sum_l495_495400


namespace find_a_l495_495642

theorem find_a 
  (a : ℝ)
  (curve : ℝ → ℝ := λ x, x^4 + a * x + 1)
  (deriv_curve : ℝ → ℝ := λ x, 4 * x^3 + a)
  (slope_at_neg1 : deriv_curve (-1) = 8) : 
a = -6 := 
by
  sorry

end find_a_l495_495642


namespace fraction_is_three_eights_l495_495160

-- The given number
def number := 48

-- The fraction 'x' by which the number exceeds by 30
noncomputable def fraction (x : ℝ) : Prop :=
number = number * x + 30

-- Our goal is to prove that the fraction is 3/8
theorem fraction_is_three_eights : fraction (3 / 8) :=
by
  -- We reduced the goal proof to a simpler form for illustration, you can solve it rigorously
  sorry

end fraction_is_three_eights_l495_495160


namespace rectangle_area_l495_495206

theorem rectangle_area (w l : ℝ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end rectangle_area_l495_495206


namespace taller_cycle_height_l495_495111

theorem taller_cycle_height
    (shadow_taller : ℝ)
    (height_shorter : ℝ)
    (shadow_shorter : ℝ)
    (h : ℝ) :
    shadow_taller = 5 → height_shorter = 2 → shadow_shorter = 4 → h = (height_shorter * shadow_taller) / shadow_shorter → h = 2.5 :=
by
    intros h_shadow_taller h_height_shorter h_shadow_shorter h_proportion
    rw [h_shadow_taller, h_height_shorter, h_shadow_shorter] at h_proportion
    exact h_proportion
-- Proof is skipped with sorry
-- sorry

end taller_cycle_height_l495_495111


namespace inequality_solution_l495_495447

theorem inequality_solution
  (x : ℝ) :
  x ∉ {2, 3, 4, 5, 6, 7} →
  ((x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0 ↔ 
  (x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 6) ∨ (7 < x)) :=
by
  -- Proof goes here
  sorry

end inequality_solution_l495_495447


namespace students_not_picked_l495_495826

def total_students : ℕ := 58
def number_of_groups : ℕ := 8
def students_per_group : ℕ := 6

theorem students_not_picked :
  total_students - (number_of_groups * students_per_group) = 10 := by 
  sorry

end students_not_picked_l495_495826


namespace quadratic_root_range_l495_495701

open Real

namespace MathProof

theorem quadratic_root_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (x^2 + m * x + 1 = 0) ∧ (y^2 + m * y + 1 = 0)) → 
  m ∈ (-∞, -2) ∪ (2, ∞) :=
by
  -- We need to show that if the quadratic equation has two distinct real roots,
  -- then the parameter m lies within the specified interval.
  sorry

end MathProof

end quadratic_root_range_l495_495701


namespace three_sport_players_l495_495369

def total_members := 50
def B := 22
def T := 28
def Ba := 18
def BT := 10
def BBa := 8
def TBa := 12
def N := 4
def All := 8

theorem three_sport_players : B + T + Ba - (BT + BBa + TBa) + All = total_members - N :=
by
suffices h : 22 + 28 + 18 - (10 + 8 + 12) + 8 = 50 - 4
exact h
-- The detailed proof is left as an exercise
sorry

end three_sport_players_l495_495369


namespace jenna_remaining_money_l495_495385

theorem jenna_remaining_money (m c : ℝ) (h : (1 / 4) * m = (1 / 2) * c) : (m - c) / m = 1 / 2 :=
by
  sorry

end jenna_remaining_money_l495_495385


namespace checkerboard_probability_l495_495039

-- Define the number of squares in the checkerboard and the number on the perimeter
def total_squares : Nat := 10 * 10
def perimeter_squares : Nat := 10 + 10 + (10 - 2) + (10 - 2)

-- The number of squares not on the perimeter
def inner_squares : Nat := total_squares - perimeter_squares

-- The probability that a randomly chosen square does not touch the outer edge
def probability_not_on_perimeter : ℚ := inner_squares / total_squares

theorem checkerboard_probability :
  probability_not_on_perimeter = 16 / 25 :=
by
  -- proof goes here
  sorry

end checkerboard_probability_l495_495039


namespace numerical_form_463001_l495_495500

theorem numerical_form_463001 : 463001 = 463001 := by
  rfl

end numerical_form_463001_l495_495500


namespace num_odd_functions_l495_495101

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f1 (x : ℝ) : ℝ := 1 / x
def f2 (x : ℝ) : ℝ := 10^(Real.log x)
def f3 (x : ℝ) : ℝ := -x^3

theorem num_odd_functions : 
  (if is_odd_function f1 then 1 else 0)
  + (if is_odd_function f2 then 1 else 0)
  + (if is_odd_function f3 then 1 else 0) = 2 := 
by 
  sorry

end num_odd_functions_l495_495101


namespace korotyshkas_equal_malyshkas_l495_495708

theorem korotyshkas_equal_malyshkas (A B : Type) 
  [fintype A] [fintype B] 
  (knows_girls : A → fin 6 → B) 
  (knows_men : B → fin 6 → A) 
  : fintype.card A = fintype.card B := 
begin
  sorry
end

end korotyshkas_equal_malyshkas_l495_495708


namespace integer_count_between_l495_495329

theorem integer_count_between (a b : ℝ) (ha : a = (10.5)^3) (hb : b = (10.6)^3) :
  (b.floor - a.ceil + 1 = 33) :=
by
  have h1 : a = 1157.625 := by rw [ha]; norm_num
  have h2 : b = 1191.016 := by rw [hb]; norm_num
  sorry

end integer_count_between_l495_495329


namespace identity_function_l495_495587

theorem identity_function {f : ℕ → ℕ} (h : ∀ a b : ℕ, 0 < a → 0 < b → a - f b ∣ a * f a - b * f b) :
  ∀ a : ℕ, 0 < a → f a = a :=
by
  sorry

end identity_function_l495_495587


namespace inequality_solution_l495_495444

noncomputable def solve_inequality (x : ℝ) : Prop :=
  ((x - 3) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0

theorem inequality_solution : {x : ℝ | solve_inequality x} = 
  {x : ℝ | x < 2} ∪ {x : ℝ | 3 < x ∧ x < 4} ∪ {x : ℝ | 5 < x ∧ x < 6} ∪ {x : ℝ | x > 7} :=
by
  sorry

end inequality_solution_l495_495444


namespace acid_solution_l495_495106

theorem acid_solution (n y : ℝ) (h : n > 30) (h1 : y = 15 * n / (n - 15)) :
  (n / 100) * n = ((n - 15) / 100) * (n + y) :=
by
  sorry

end acid_solution_l495_495106


namespace inequality_proof_l495_495995

variable (a b : ℝ)

theorem inequality_proof (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) :
  (1 / a) + (1 / b) + (1 / (a * b)) ≥ 8 :=
  by
    sorry

end inequality_proof_l495_495995


namespace real_roots_iff_integer_roots_iff_l495_495590

noncomputable def discriminant (k : ℝ) : ℝ := (k + 1)^2 - 4 * k * (k - 1)

theorem real_roots_iff (k : ℝ) : 
  (discriminant k ≥ 0) ↔ (∃ (a b : ℝ), kx ^ 2 + (k + 1) * x + (k - 1) = 0) := sorry

theorem integer_roots_iff (k : ℝ) : 
  (∃ (a b : ℤ), kx ^ 2 + (k + 1) * x + (k - 1) = 0) ↔ 
  (k = 0 ∨ k = 1 ∨ k = -1/7) := sorry

-- These theorems need to be proven within Lean 4 itself

end real_roots_iff_integer_roots_iff_l495_495590


namespace rectangle_area_l495_495202

theorem rectangle_area (w l : ℝ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end rectangle_area_l495_495202


namespace prob_distinct_real_roots_prob_no_real_roots_l495_495646

-- Define function f based on the given conditions
noncomputable def f (a b x : ℝ) : ℝ := x^2 - 2 * a * x + b^2

-- Condition set for a and b in the set {0,1,2,3} and {0,1,2}
def a_set := {0.0, 1.0, 2.0, 3.0} : Set ℝ
def b_set := {0.0, 1.0, 2.0} : Set ℝ

-- Condition interval for a and b
def a_intv := Set.Icc 0 2
def b_intv := Set.Icc 0 3

-- Define the events for distinct real roots and no real roots
def event_distinct_real_roots (a b : ℝ) : Prop :=
  (a ∈ a_set) ∧ (b ∈ b_set) ∧ (a > b)

def event_no_real_roots (a b : ℝ) : Prop :=
  (a ∈ a_intv) ∧ (b ∈ b_intv) ∧ (a <= b)

-- Lean statements for the proof problems

-- Prove the probability that f(x)=0 has two distinct real roots is 1/2
theorem prob_distinct_real_roots : 
  (cardinal.mk {p : ℝ × ℝ // event_distinct_real_roots p.fst p.snd}.toList.length) /
  (cardinal.mk {p : ℝ × ℝ // (p.fst ∈ a_set ∧ p.snd ∈ b_set)}.toList.length) = 1 / 2 :=
sorry

-- Prove the probability that f(x)=0 has no real roots is 2/3
theorem prob_no_real_roots : 
  (measure_theory.volume (set_of (λ p : ℝ × ℝ, event_no_real_roots p.fst p.snd))) /
  (measure_theory.volume {p : ℝ × ℝ // (p.fst ∈ a_intv ∧ p.snd ∈ b_intv)}) = 2 / 3 :=
sorry

end prob_distinct_real_roots_prob_no_real_roots_l495_495646


namespace expedition_ratios_l495_495386

theorem expedition_ratios (F : ℕ) (S : ℕ) (L : ℕ) (R : ℕ) 
  (h1 : F = 3) 
  (h2 : S = F + 2) 
  (h3 : F + S + L = 18) 
  (h4 : L = R * S) : 
  R = 2 := 
sorry

end expedition_ratios_l495_495386


namespace sin_cos_pow_eq_l495_495608

theorem sin_cos_pow_eq (sin cos : ℝ → ℝ) (x : ℝ) (h₀ : sin x + cos x = -1) (n : ℕ) : 
  sin x ^ n + cos x ^ n = (-1) ^ n :=
by
  sorry

end sin_cos_pow_eq_l495_495608


namespace sum_odd_integers_200_400_l495_495488

theorem sum_odd_integers_200_400 : 
  (Finset.sum (Finset.filter (λ x, x % 2 = 1) (Finset.Ico 200 401))) = 30000 :=
by
  sorry

end sum_odd_integers_200_400_l495_495488


namespace find_coordinates_of_Q_l495_495716

-- Definitions of the given points
def O : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (6, 8)

-- Definition of the rotation angle
def theta : ℝ := 3 * Real.pi / 4

-- The expected result after rotation
def expected_Q : ℝ × ℝ := (-7 * Real.sqrt 2, -Real.sqrt 2)

-- Now we state the theorem that needs to be proved
theorem find_coordinates_of_Q :
  let Q := (P.1 * Real.cos(theta) - P.2 * Real.sin(theta), P.1 * Real.sin(theta) + P.2 * Real.cos(theta)) in
  Q = expected_Q :=
by
  -- The actual proof is omitted
  sorry

end find_coordinates_of_Q_l495_495716


namespace sequence_221_l495_495575

-- Definitions for the sequence terms
def t_1 : ℝ := 2

def t (n : ℕ) : ℝ :=
  if n = 1 then t_1
  else if n % 2 = 0 then (n + t (n - 1))
       else (t (n - 1) / n)

-- The theorem we wish to prove
theorem sequence_221 : t 221 = 1 / 221 :=
sorry

end sequence_221_l495_495575


namespace larger_triangle_perimeter_l495_495925

theorem larger_triangle_perimeter (a b c a' b' c' : ℝ)
    (h1 : a = 9) (h2 : b = 12) (h3 : c = real.sqrt (a^2 + b^2))
    (h4 : a' = 2 * a) (h5 : b' = 2 * b) (h6 : c' = 2 * c)
    (h7 : c' = 30) : a' + b' + c' = 72 := 
by
  -- Conditions from the problem used in defining hypotheses.
  have h8 : c = 15, from real.sqrt_eq_iff_sq_eq.mpr ⟨15, by norm_num⟩ -- Calculated hypotenuse of smaller triangle.
  simp [h8, h4, h5, h6, h1, h2]
  sorry

end larger_triangle_perimeter_l495_495925


namespace earl_ellen_time_l495_495248

def rate_earl_std := 36 -- Earl's rate for standard envelopes
def rate_earl_lrg := 24 -- Earl's rate for larger envelopes

def time_ellen_std := 1.5 -- Time Ellen takes for 36 standard envelopes
def time_ellen_lrg := 2 -- Time Ellen takes for 36 larger envelopes

def envelopes_std := 150 -- Number of standard-sized envelopes
def envelopes_lrg := 90 -- Number of larger-sized envelopes

def circulars_per_std := 2 -- Circulars per standard envelope
def circulars_per_lrg := 3 -- Circulars per larger envelope

def rate_ellen_std := 36 / 1.5 -- Ellen's rate for standard envelopes
def rate_ellen_lrg := 36 / 2 -- Ellen's rate for larger envelopes

def rate_together_std := rate_earl_std + rate_ellen_std
def rate_together_lrg := rate_earl_lrg + rate_ellen_lrg

-- Time calculation for standard and larger envelopes
def time_std := envelopes_std / rate_together_std
def time_lrg := envelopes_lrg / rate_together_lrg

-- Total time taken for both sets of envelopes
def total_time := time_std + time_lrg

-- The time should be approximately 4.64 minutes
def proof_problem (total_time : ℝ) := total_time ≈ 4.64

-- Show that the calculated total time is approximately 4.64 minutes
theorem earl_ellen_time : proof_problem (total_time) :=
by sorry

end earl_ellen_time_l495_495248


namespace sum_of_squares_inequality_l495_495297

variable {n : ℕ} (hn : n ≥ 2)
variable {x : Fin n → ℝ}

theorem sum_of_squares_inequality
  (sum_eq_zero : ∑ i, x i = 0)
  (pairs_condition : ∀ t > 0, ∃ k ≤ (1 / t : ℝ), k = (set_of (λ (ij : Fin n × Fin n), abs (x ij.1 - x ij.2) ≥ t)).card)
  : ∑ i, (x i) ^ 2 < (1 / (n : ℝ)) * (max i, x i - min i, x i)^2 :=
sorry

end sum_of_squares_inequality_l495_495297


namespace solve_f_le_2_l495_495314

-- Define the piecewise function f
noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 1 then 2^(1 - x) else 1 - Real.logb 2 x

-- Define the statement that proving f(x) ≤ 2 leads to the solution set [0, +∞)
theorem solve_f_le_2 : {x : ℝ | f x ≤ 2} = set.Ici 0 := 
by
  sorry

end solve_f_le_2_l495_495314


namespace replace_star_l495_495495

theorem replace_star (x : ℕ) : 2 * 18 * 14 = 6 * x * 7 → x = 12 :=
sorry

end replace_star_l495_495495


namespace parallel_chords_mn_l495_495044

open EuclideanGeometry

theorem parallel_chords_mn {A B C D E F M N : Point}
  (h_circle : cyclic {A, B, C, D, E, F})
  (h_intersect_M : intersect_line (line_through E C) (line_through A D) = some M)
  (h_intersect_N : intersect_line (line_through B E) (line_through D F) = some N)
  (h_parallel_ABC_CF : parallel (line_through A B) (line_through C F)) :
  parallel (line_through M N) (line_through A B) ∧ parallel (line_through M N) (line_through C F) :=
sorry

end parallel_chords_mn_l495_495044


namespace greatest_integer_radius_l495_495682

theorem greatest_integer_radius (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 := 
by {
  have h1 : r^2 < 75, from (div_lt_div_right (by linarith [pi_pos])).mp h,
  have sqrt_75_lt_9 : sqrt 75 < 9 := by norm_num,
  have sqrt_75_lt_9' : r < 9 := lt_of_pow_lt_pow_of_lt_two h1 sqrt_75_lt_9,
  exact le_of_lt (int.le_iff_coe.mp sqrt_75_lt_9')
}

end greatest_integer_radius_l495_495682


namespace perfect_square_sum_remainder_l495_495025

theorem perfect_square_sum_remainder : 
  (∑ n in { n : ℕ | (n^2 + 12 * n - 2006).is_square }, n) % 1000 = 0 :=
by
  sorry

end perfect_square_sum_remainder_l495_495025


namespace relationship_among_abc_l495_495740

noncomputable def a : ℝ := 2 ^ 0.3
def b : ℝ := 0.3 ^ 2
noncomputable def c : ℝ := Real.log 0.3 / Real.log 2

theorem relationship_among_abc : c < b ∧ b < a :=
by
  sorry

end relationship_among_abc_l495_495740


namespace right_triangle_height_l495_495537

theorem right_triangle_height
  (h : ℕ)
  (base : ℕ)
  (rectangle_area : ℕ)
  (same_area : (1 / 2 : ℚ) * base * h = rectangle_area)
  (base_eq_width : base = 5)
  (rectangle_area_eq : rectangle_area = 45) :
  h = 18 :=
by
  sorry

end right_triangle_height_l495_495537


namespace max_radius_of_circle_l495_495692

theorem max_radius_of_circle (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
by
  sorry

end max_radius_of_circle_l495_495692


namespace observations_count_correct_l495_495832

noncomputable def corrected_observations (n : ℕ) : ℕ :=
  if 36 * n + 22 = 36.5 * n then n else 0

theorem observations_count_correct :
  ∃ n : ℕ, 36 * n + 22 = 36.5 * n ∧ corrected_observations n = 44 :=
by {
  sorry
}

end observations_count_correct_l495_495832


namespace cannot_all_positive_l495_495898

/-- The initial configuration of a 4x4 grid containing +1 and -1. --/
def initial_grid : Matrix (Fin 4) (Fin 4) ℤ :=
  !![![1, 1, 1, 1], 
     ![1, 1, 1, 1], 
     ![1, 1, 1, -1], 
     ![1, 1, 1, 1]]

/-- Allowed operations: flip all signs in a row, column, or diagonal. --/
inductive Operation
  | flip_row (i : Fin 4)
  | flip_column (j : Fin 4)
  | flip_diag (k : ℤ)

open Operation

/-- Performing an operation on the grid. --/
def perform_op (op : Operation) (grid : Matrix (Fin 4) (Fin 4) ℤ) : Matrix (Fin 4) (Fin 4) ℤ :=
  match op with
  | flip_row i    => grid.update_row i (grid.row i.map (λ x => -x))
  | flip_column j => grid.update_column j (grid.column j.map (λ x => -x))
  | flip_diag k   => sorry -- define the flip on diagonals as required

/-- Prove that it is impossible to obtain a grid with all positive numbers. --/
theorem cannot_all_positive :
  ∀ ops, let final_grid := ops.foldl (λ g op => perform_op op g) initial_grid in
  ¬ ∀ (i j : Fin 4), final_grid i j > 0 :=
by
  intros
  sorry 

end cannot_all_positive_l495_495898


namespace equivalent_statements_l495_495876

variables (P Q : Prop)

theorem equivalent_statements (h : P → Q) : 
  ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) ↔ (P → Q) := by
sorry

end equivalent_statements_l495_495876


namespace sequence_sum_zero_l495_495566

theorem sequence_sum_zero :
  ∑ k in finset.range 500, (4*k + 1) - (4*k + 2) - (4*k + 3) + (4*k + 4) = 0 :=
by
  sorry

end sequence_sum_zero_l495_495566


namespace existence_of_tangent_quadrilateral_l495_495910

theorem existence_of_tangent_quadrilateral
  (F : Point)
  (circle_centered_at_F : Circle F)
  (parabola_with_focus_F : Parabola F)
  (common_points : ∃ X Y, X ≠ Y ∧ IsPointOnCircle X circle_centered_at_F ∧ IsPointOnCircle Y circle_centered_at_F ∧ IsPointOnParabola X parabola_with_focus_F ∧ IsPointOnParabola Y parabola_with_focus_F) :
  ∃ A B C D, A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ IsPointOnCircle A circle_centered_at_F ∧ IsPointOnCircle B circle_centered_at_F ∧ IsPointOnCircle C circle_centered_at_F ∧ IsPointOnCircle D circle_centered_at_F ∧ 
  IsTangent (Line_through A B) parabola_with_focus_F ∧ IsTangent (Line_through B C) parabola_with_focus_F ∧ IsTangent (Line_through C D) parabola_with_focus_F ∧ IsTangent (Line_through D A) parabola_with_focus_F := 
sorry

end existence_of_tangent_quadrilateral_l495_495910


namespace lassis_from_mangoes_l495_495229

theorem lassis_from_mangoes (lassis_per_4_mangoes: ℕ) (mangoes: ℕ) (lassis_per_1_mango: ℕ) : (lassis_per_4_mangoes = 20) → (mangoes = 16) → (lassis_per_1_mango = 5) → (lassis_per_1_mango * mangoes = 80) :=
by {
  intro h1,
  intro h2,
  intro h3,
  rw [h1, h2, h3],
  simp,
  sorry
}

end lassis_from_mangoes_l495_495229


namespace problem_l495_495352

-- Define the function f
def f (x : ℝ) (k : ℕ) := log 2 x + x - k

-- The main theorem
theorem problem (k : ℕ) (h1 : ∀ x, 2 < x → x < 3 → f x k = 0 → strict_mono_on (f (λ x, f x k)) (set.Ioo 2 3)) 
  (h2 : ∃! x, 2 < x ∧ x < 3 ∧ f x k = 0) : k = 4 :=
sorry

end problem_l495_495352


namespace y_increase_by_20_l495_495364

-- Define the conditions
def relationship (Δx Δy : ℕ) : Prop :=
  Δy = (11 * Δx) / 5

-- The proof problem statement
theorem y_increase_by_20 : relationship 5 11 → relationship 20 44 :=
by
  intros h
  sorry

end y_increase_by_20_l495_495364


namespace prime_cube_plus_nine_prime_l495_495338

theorem prime_cube_plus_nine_prime (P : ℕ) (hP : Nat.Prime P) (h : Nat.Prime (P^3 + 9)) : P^2 - 7 = -3 := by
  sorry

end prime_cube_plus_nine_prime_l495_495338


namespace differentiation_operations_correctness_l495_495128

theorem differentiation_operations_correctness :
  ((D x : ℝ) cos x = -sin x) ∧
  ((D x : ℝ) (3^x) = 3^x * log 3) ∧
  ((D x : ℝ) (log 10 x) = 1 / (x * log 10)) ∧
  ((D x : ℝ) (x^2 * cos x) = 2 * x * cos x - x^2 * sin x) :=
by
  sorry

end differentiation_operations_correctness_l495_495128


namespace eq1_solution_eq2_solution_l495_495058

theorem eq1_solution (x : ℝ) (h : 6 * x - 7 = 4 * x - 5) : x = 1 :=
by
  sorry

theorem eq2_solution (x : ℝ) (h : (1 / 2) * x - 6 = (3 / 4) * x) : x = -24 :=
by
  sorry

end eq1_solution_eq2_solution_l495_495058


namespace cookfire_logs_burned_per_hour_l495_495914

theorem cookfire_logs_burned_per_hour 
  (x : ℕ) (h_condition : (let after3hours := ((6 - x + 2) - x + 2) - x + 2 in after3hours = 3)) : x = 3 :=
by
  sorry

end cookfire_logs_burned_per_hour_l495_495914


namespace rectangle_area_is_243_square_meters_l495_495182

-- Given conditions
variables (w l : ℝ)
-- The width is one-third of its length
def width_one_third_length : Prop := w = l / 3

-- The perimeter of the rectangle is 72 meters
def perimeter_72 : Prop := 2 * (w + l) = 72

-- The area of the rectangle
def area : ℝ := w * l

-- Proof statement
theorem rectangle_area_is_243_square_meters 
  (h1 : width_one_third_length w l) 
  (h2 : perimeter_72 w l) : area w l = 243 := 
by 
sorry

end rectangle_area_is_243_square_meters_l495_495182


namespace problem_statement_l495_495144

noncomputable def f : ℝ → ℝ
| x => if x > 0 then x * (1 - x) else x * (1 + x)

theorem problem_statement (x : ℝ) (h : x < 0) :
  f(x) = x * (1 + x) :=
by
  unfold f
  split_ifs with h1 h2
  . exfalso
    linarith
  . exact rfl
  . exact rfl

end problem_statement_l495_495144


namespace length_of_BC_l495_495401

open Triangle Real

theorem length_of_BC (A B C I D E F : Point) (h_iso : is_isosceles_triangle A B C)
  (h_incenter : is_incenter I)
  (h_AI : dist A I = 3)
  (h_inradius : dist I D = 2)
  (h_touching : touches_incircle I A D B C) :
  dist B C = 4 * sqrt 5 :=
sorry

end length_of_BC_l495_495401


namespace gcd_ab_a2b2_eq_one_or_two_l495_495038

-- Definitions and conditions
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Problem statement
theorem gcd_ab_a2b2_eq_one_or_two (a b : ℕ) (h : coprime a b) : 
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
by
  sorry

end gcd_ab_a2b2_eq_one_or_two_l495_495038


namespace complex_number_solution_l495_495996

theorem complex_number_solution (z : ℂ) (h : ∥z∥ + z = 2 + 4 * complex.I) : z = -3 + 4 * complex.I :=
sorry

end complex_number_solution_l495_495996


namespace arithmetic_sequence_property_l495_495718

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_property (h1 : is_arithmetic_sequence a)
  (h2 : a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - (1/2) * a 8 = 8 :=
sorry

end arithmetic_sequence_property_l495_495718


namespace product_real_roots_l495_495268

theorem product_real_roots (a b c d e : ℝ) (h : a = 1 ∧ b = -4 ∧ c = 6 ∧ d = -2 ∧ e = -2010) :
  let f := λ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e in
  product_real_roots f = 1 - Real.sqrt 2010 :=
by sorry

end product_real_roots_l495_495268


namespace min_variance_Y_l495_495020

open ProbabilityTheory -- To use concepts and notation from probability theory.

-- Define a Bernoulli random variable X and establish the conditions.
variables {Ω : Type*} {p : ℝ}
variables [ProbabilitySpace Ω]
variable (X : Ω → ℝ) -- Bernoulli random variable Bernoulli(p)

-- Define the conditions for X.
axiom X_bernoulli : ∀ ω : Ω, X ω = 1 ∨ X ω = 0
axiom prob_X_1 : ℙ {ω | X ω = 1} = p
axiom prob_X_0 : ℙ {ω | X ω = 0} = 1 - p

-- Define Y such that X and Y are independent and their sum is symmetric.
variable (Y : Ω → ℝ)
axiom Y_independent_X : ∀ {A B : Set Ω}, IndependentSets (Set.Pairwise (λ ω, ω ∈ A ∩ {ω | X ω ≠ 0})) (λ ω, ω ∈ B ∩ {ω | Y ω ≠ 0})
axiom sum_symmetric : ∀ A, ℙ (Y ∈ A) = ℙ (Y ∈ -A)

-- Establish the minimum variance of Y.
theorem min_variance_Y : p ≠ 1 / 2 → ∃ f : ℝ → ℝ, (∀ y, (1 - p) * f y + p * f (1 + y) ≤ y ^ 2 - p) →
  (∃ g : ℝ → ℝ, g (1 + y) = - g y ∧ (1 - p) * (π * f y) + p * (π * f (1 + y)) ≤ y ^ 2 - p) →
  var Y = p * (1 - p) ∧ (p = 1 / 2 → var Y = 0) :=
sorry

end min_variance_Y_l495_495020


namespace second_horse_revolutions_l495_495477

theorem second_horse_revolutions 
  (r1 r2 : ℝ) (R1 R2 : ℕ) (h1 : r1 = 15) (h2 : r2 = 5) (h3 : R1 = 20) 
  (H : 2 * real.pi * r1 * R1 = 2 * real.pi * r2 * R2) :
  R2 = 60 :=
sorry

end second_horse_revolutions_l495_495477


namespace sum_B_l495_495030

open Set

/-- Define set A -/
def A : Set ℤ := {2, 0, 1, 3}

/-- Define set B based on conditions given in the problem -/
def B : Set ℤ :=
  { x | -x ∈ A ∧ 2 - x*x ∉ A }

/-- The goal is to prove that the sum of all elements in B is -5 -/
theorem sum_B : ∑ x in B.to_finset, x = -5 :=
by
  sorry

end sum_B_l495_495030


namespace midpoint_inequality_1_midpoint_inequality_2_l495_495046

open_locale classical

variables {A B C D M N : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]

-- Conditions: Points M and N are the midpoints of segments AB and CD respectively
def is_midpoint (M B : Type) [metric_space M] [metric_space B] := dist M B = dist B C / 2

-- Proof problem statements
theorem midpoint_inequality_1 (M N A B C D : Type) [metric_space M] [metric_space N] [metric_space A] 
  [metric_space B] [metric_space C] [metric_space D]
  (h1 : is_midpoint M A B) (h2 : is_midpoint N C D) :
  dist M N ≤ 1 / 2 * (dist A C + dist B D) :=
sorry

theorem midpoint_inequality_2 (M N A B C D : Type) [metric_space M] [metric_space N] [metric_space A] 
  [metric_space B] [metric_space C] [metric_space D]
  (h1 : is_midpoint M A B) (h2 : is_midpoint N C D) :
  dist M N ≤ 1 / 2 * (dist B C + dist A D) :=
sorry

end midpoint_inequality_1_midpoint_inequality_2_l495_495046


namespace one_positive_real_solution_l495_495240

noncomputable def f (x : ℝ) : ℝ := x^4 + 5 * x^3 + 10 * x^2 + 2023 * x - 2021

theorem one_positive_real_solution : 
  ∃! x : ℝ, 0 < x ∧ f x = 0 :=
by
  -- Proof goes here
  sorry

end one_positive_real_solution_l495_495240


namespace distance_P_to_AB_l495_495105

def point_P_condition (P : ℝ) : Prop :=
  P > 0 ∧ P < 1

def parallel_line_property (P : ℝ) (h : ℝ) : Prop :=
  h = 1 - P / 1

theorem distance_P_to_AB (P h : ℝ) (area_total : ℝ) (area_smaller : ℝ) :
  point_P_condition P →
  parallel_line_property P h →
  (area_smaller / area_total) = 1 / 3 →
  h = 2 / 3 :=
by
  intro hP hp hratio
  sorry

end distance_P_to_AB_l495_495105


namespace max_min_difference_l495_495644

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem max_min_difference :
  ∀ (f : ℝ → ℝ), (∀ x, 1 ≤ x ∧ x ≤ 2 → f x = 1 / x) →
  let A := Sup { y | ∃ x, 1 ≤ x ∧ x ≤ 2 ∧ y = f x } in
  let B := Inf { y | ∃ x, 1 ≤ x ∧ x ≤ 2 ∧ y = f x } in
  A - B = 1 / 2 :=
by
  intros f hA
  let S := { y | ∃ x, 1 ≤ x ∧ x ≤ 2 ∧ hA x y }
  let A := Sup S
  let B := Inf S
  have h1 : A = 1 := sorry
  have h2 : B = 1 / 2 := sorry
  show A - B = 1 / 2 from sorry

end max_min_difference_l495_495644


namespace smallest_integer_half_square_third_cube_l495_495844

-- Definitions for the problem
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

-- Statement of the problem
theorem smallest_integer_half_square_third_cube :
  ∃ n : ℕ, n > 0 ∧ is_perfect_square (n / 2) ∧ is_perfect_cube (n / 3) ∧
            ∀ m : ℕ, m > 0 ∧ is_perfect_square (m / 2) ∧ is_perfect_cube (m / 3) → m ≥ n :=
begin
  use 648,
  split,
  { norm_num },
  split,
  { use 18,
    norm_num },
  split,
  { use 6,
    norm_num },
  intros m hm hms hmc,
  sorry -- Proof that 648 is the smallest such number
end

end smallest_integer_half_square_third_cube_l495_495844


namespace SO_perpendicular_to_A1B1C_l495_495421

open EuclideanGeometry

variables (S A B C A1 B1 C1 O : Point)

-- Define the conditions as assumptions
axiom SA_edge : is_edge S A
axiom SB_edge : is_edge S B
axiom SC_edge : is_edge S C

axiom A1_on_SA : is_on_segment A1 S A
axiom B1_on_SB : is_on_segment B1 S B
axiom C1_on_SC : is_on_segment C1 S C

axiom planes_parallel : parallel (plane A1 B1 C1) (plane A B C)

axiom center_of_sphere : is_center_of_sphere O {S, A, B, C1}

-- State the theorem to be proved
theorem SO_perpendicular_to_A1B1C (h : center_of_sphere S A B C1 O) :
  perpendicular (line_through S O) (plane A1 B1 C) :=
sorry

end SO_perpendicular_to_A1B1C_l495_495421


namespace cost_of_sandwiches_and_sodas_l495_495053

theorem cost_of_sandwiches_and_sodas (sandwich_cost soda_cost : ℕ) (n_sandwiches n_sodas total_cost : ℕ) 
(h1 : sandwich_cost = 4) (h2 : soda_cost = 3) 
(h3 : n_sandwiches = 3) (h4 : n_sodas = 5) 
(h5 : total_cost = n_sandwiches * sandwich_cost + n_sodas * soda_cost) : 
total_cost = 27 := 
by 
  rw [h1, h2, h3, h4] 
  simp [h5] 
  exact h5

#check cost_of_sandwiches_and_sodas

end cost_of_sandwiches_and_sodas_l495_495053


namespace cards_eaten_by_hippopotamus_l495_495726

-- Defining the initial and remaining number of cards
def initial_cards : ℕ := 72
def remaining_cards : ℕ := 11

-- Statement of the proof problem
theorem cards_eaten_by_hippopotamus (initial_cards remaining_cards : ℕ) : initial_cards - remaining_cards = 61 :=
by
  sorry

end cards_eaten_by_hippopotamus_l495_495726


namespace problem_part1_problem_part2_l495_495635

variable {a b : ℤ}

theorem problem_part1 
  (h1 : |a - 2| = 0 ∨ (b + 1)^2 = 0) : b^a = 1 :=
by
-- Proof goes here
sorried_value

theorem problem_part2 
  (h2 : |a - 2| = 0 ∨ (b + 1)^2 = 0) : a^3 + b^15 = 7 :=
by
-- Proof goes here
sorried_value

-- Auxiliary Definitions
noncomputable def sorried_value : ∀ {T : Type}, T :=
-- This serves as a placeholder to ensure the statement code compiles
by sorry

end problem_part1_problem_part2_l495_495635


namespace total_meters_examined_l495_495220

theorem total_meters_examined (x : ℕ) (h1 : 0.10 * x = 20) : x = 200 :=
by {
  sorry
}

end total_meters_examined_l495_495220


namespace product_real_parts_eq_l495_495578

theorem product_real_parts_eq :
  let x := complex
  let a : x := -1 + (complex.ofReal (real.fourthRoot 2)) * complex.cos (-π / 8)
  let b : x := -1 - (complex.ofReal (real.fourthRoot 2)) * complex.cos (-π / 8)
  let re_a : real := a.re
  let re_b : real := b.re
  ((x^2 + 2*x + complex.I = 0) → (re_a * re_b = (1 - real.sqrt 2) / 2)) := 
sorry

end product_real_parts_eq_l495_495578


namespace angle_between_vectors_l495_495287

variables 
  (a b : ℝ^3) -- Define a and b as vectors in ℝ^3

-- Condition: magnitudes of vectors a and b
def mag_a : ℝ := ‖a‖
def mag_b : ℝ := ‖b‖

-- Condition: a - b is perpendicular to a
def perp_cond : ((a - b) ⬝ a) = 0

-- The angle between vectors a and b
noncomputable def angle_a_b : ℝ := real.arccos ((a ⬝ b) / (‖a‖ * ‖b‖))

-- Now we use the conditions and prove the angle
theorem angle_between_vectors 
  (h1 : mag_a = sqrt 2) 
  (h2 : mag_b = 2) 
  (h3 : perp_cond) :
  angle_a_b = π / 4 := 
sorry

end angle_between_vectors_l495_495287


namespace power_function_value_l495_495638

theorem power_function_value (f : ℝ → ℝ) (h1 : ∃ α : ℝ, ∀ x : ℝ, f(x) = x^α) 
  (h2 : f 2 = Real.sqrt 2) : f (1 / 9) = 1 / 3 :=
sorry

end power_function_value_l495_495638


namespace infinite_B_for_floor_eq_l495_495429

theorem infinite_B_for_floor_eq 
  (n: ℕ)
  (H: n ≥ 1980) : 
  ∃ (B: ℕ), ∃^∞ (x y: ℕ), (⌊x ^ (3/2): ℝ⌋ + ⌊y ^ (3/2): ℝ⌋ = B) :=
sorry

end infinite_B_for_floor_eq_l495_495429


namespace finite_non_representable_sum_of_distinct_perfect_squares_l495_495437

theorem finite_non_representable_sum_of_distinct_perfect_squares :
  {n : ℕ | ¬(∃ (s : finset ℕ), s.sum (λ x, x^2) = n)}.finite := 
sorry

end finite_non_representable_sum_of_distinct_perfect_squares_l495_495437


namespace total_bill_correct_l495_495564

-- Definitions based on conditions
def number_of_adults : Nat := 2
def number_of_children : Nat := 5
def cost_per_meal : Nat := 3

-- The main theorem to be proven
theorem total_bill_correct : number_of_adults + number_of_children = 7
  → 7 * cost_per_meal = 21 :=
by {
  intro h,
  rw h,
  calc 7 * 3 = 21 : by norm_num
}

#check total_bill_correct

end total_bill_correct_l495_495564


namespace EF_squared_correct_l495_495454

-- Define the problem setup and the proof goal.
theorem EF_squared_correct :
  ∀ (A B C D E F : Type)
  (side : ℝ)
  (h1 : side = 10)
  (BE DF AE CF : ℝ)
  (h2 : BE = 7)
  (h3 : DF = 7)
  (h4 : AE = 15)
  (h5 : CF = 15)
  (EF_squared : ℝ),
  EF_squared = 548 :=
by
  sorry

end EF_squared_correct_l495_495454


namespace intersection_complement_l495_495653

universe u

noncomputable def U := @Set.univ ℝ
noncomputable def A := {x : ℝ | x^2 < 1}
noncomputable def B := {x : ℝ | x^2 - 2 * x > 0}
noncomputable def C_R (B : Set ℝ) := {x : ℝ | ¬ (x ∈ B)}

theorem intersection_complement :
  (A ∩ C_R(B)) = {x : ℝ | 0 ≤ x ∧ x < 1} :=
by {
  sorry
}

end intersection_complement_l495_495653


namespace ab_fraction_inequality_l495_495425

theorem ab_fraction_inequality (a b : ℝ) (ha : 0 < a) (ha1 : a < 1) (hb : 0 < b) (hb1 : b < 1) :
  (a * b * (1 - a) * (1 - b)) / ((1 - a * b) ^ 2) < 1 / 4 :=
by
  sorry

end ab_fraction_inequality_l495_495425


namespace area_of_rectangle_l495_495185

theorem area_of_rectangle (w l : ℝ) (h₁ : w = l / 3) (h₂ : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end area_of_rectangle_l495_495185


namespace find_a_value_l495_495261

theorem find_a_value (a x y : ℝ) :
  (|y| + |y - x| ≤ a - |x - 1| ∧ (y - 4) * (y + 3) ≥ (4 - x) * (3 + x)) → a = 7 :=
by
  sorry

end find_a_value_l495_495261


namespace folding_line_segment_squared_l495_495533

-- Define the basic structure of the problem
structure EquilateralTriangle (A B C : Point) :=
  (side_length : ℝ)
  (eq_side_length : dist A B = side_length ∧ dist B C = side_length ∧ dist C A = side_length)

-- Define the Lean structure for the given conditions
structure FoldingCondition (A B C : Point) :=
  (triangle : EquilateralTriangle A B C)
  (dist_BX : ℝ)
  (touch_point_on_BC : ∃ X, dist B X = dist_BX ∧ collinear {B, X, C})

-- Define the problem using the given conditions
def problem (A B C : Point) : Prop :=
  ∃ P Q, FoldingCondition A B C ∧
  let PQ_squared := (∃ PA QA pq_sq : ℝ, 
    PA = 15 - dist_BX ∧
    QA = 15 - dist_BX ∧  -- Both distances need to be derived out temporarily for simplicity in Lean
    pq_sq = (PA)^2 - (PA * QA) + (QA)^2 in
  pq_sq = (32761 * 809) / 244036)

-- State the theorem
theorem folding_line_segment_squared (A B C : Point) : problem A B C :=
by sorry

end folding_line_segment_squared_l495_495533


namespace first_player_has_winning_strategy_l495_495545

-- Define the classic chessboard
def Chessboard : Type := Fin 8 × Fin 8

-- Define the rules of the game
def is_valid_move (src dest : Chessboard) : Prop :=
  let (x1, y1) := src in
  let (x2, y2) := dest in
  (x2 > x1 ∧ y2 = y1) ∨  -- Move right
  (x2 = x1 ∧ y2 > y1) ∨  -- Move upward
  (x2 > x1 ∧ y2 > y1)    -- Move diagonally right and upward

-- Define the winning position
def WinningPosition : Chessboard := (7, 7)  -- H8 (7,7) in zero-based index

-- Starting position
def StartPosition : Chessboard := (2, 0)  -- C1 (2,0) in zero-based index

-- Define the winning strategy existence for the first player
theorem first_player_has_winning_strategy : 
  ∃ strategy : (Chessboard → Chessboard), 
    ∀ current_position : Chessboard, current_position = StartPosition → strategy current_position = WinningPosition :=
  sorry

end first_player_has_winning_strategy_l495_495545


namespace third_quadrant_condition_l495_495881

-- Define the conditions for the third quadrant
def in_third_quadrant (p: ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0

-- Translate the problem statement to a Lean theorem
theorem third_quadrant_condition (a b : ℝ) (h1 : a + b < 0) (h2 : a * b > 0) : in_third_quadrant (a, b) :=
sorry

end third_quadrant_condition_l495_495881


namespace smallest_number_of_roses_to_buy_l495_495413

-- Definitions representing the conditions
def group_size1 : ℕ := 9
def group_size2 : ℕ := 19

-- Statement representing the problem and solution
theorem smallest_number_of_roses_to_buy : Nat.lcm group_size1 group_size2 = 171 := 
by 
  sorry

end smallest_number_of_roses_to_buy_l495_495413


namespace canonical_equations_of_line_intersection_l495_495882

theorem canonical_equations_of_line_intersection
  (x y z : ℝ)
  (h1 : 2 * x - 3 * y + z + 6 = 0)
  (h2 : x - 3 * y - 2 * z + 3 = 0) :
  (∃ (m n p x0 y0 z0 : ℝ), 
  m * (x + 3) = n * y ∧ n * y = p * z ∧ 
  m = 9 ∧ n = 5 ∧ p = -3 ∧ 
  x0 = -3 ∧ y0 = 0 ∧ z0 = 0) :=
sorry

end canonical_equations_of_line_intersection_l495_495882


namespace sequence_term_formula_l495_495356

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1

theorem sequence_term_formula (n : ℕ) (h : n ≥ 2) :
  ∏ i in (Finset.range n).map (Function.Embedding.coe (Nat.succ_embedding _)), a_n i = n^2 :=
by
  sorry

end sequence_term_formula_l495_495356


namespace algebraic_identity_neg_exponents_l495_495968

theorem algebraic_identity_neg_exponents (x y z : ℂ) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (y * z + x * z + x * y) * x⁻¹ * y⁻¹ * z⁻¹ * (x + y + z)⁻¹ :=
by
  sorry

end algebraic_identity_neg_exponents_l495_495968


namespace geometric_sequence_quot_l495_495392

noncomputable def geometric_sequence (a : ℕ → ℚ) := 
  ∀ n : ℕ, a (n + 1) = a n * q

def Sn (a : ℕ → ℚ) (n : ℕ) := 
  a 1 * (1 - q^(n+1)) / (1 - q)

theorem geometric_sequence_quot (a : ℕ → ℚ) [geometric_sequence a] (q : ℚ) 
  (h : a 3 - 8 * a 6 = 0) :
  (Sn a 8) / (Sn a 4) = 17 / 16 :=
by
  sorry

end geometric_sequence_quot_l495_495392


namespace cesaro_sum_100_term_sequence_l495_495982

noncomputable def cesaroSum (P : List ℝ) : ℝ :=
  (List.range P.length).sum.map (λ k => (P.take (k+1)).sum) / P.length

theorem cesaro_sum_100_term_sequence (P : List ℝ) (h : P.length = 99)
  (h₁ : cesaroSum P = 1000) :
  cesaroSum (1 :: P) = 991 :=
by
  sorry

end cesaro_sum_100_term_sequence_l495_495982


namespace greatest_integer_radius_l495_495678

theorem greatest_integer_radius (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
begin
  have h1 : r^2 < 75, from (lt_div_iff (real.pi_pos)).mp (by rwa [mul_div_cancel_left _ (ne_of_gt real.pi_pos)]),
  -- To show the highest integer r satisfies r^2 < 75 is 8:
  have : r ≤ nat.floor (real.sqrt 75), 
  { exact nat.le_floor_of_lt (by exact_mod_cast h1) },
  -- Since nat.floor (real.sqrt 75) == 8
  exact le_trans this (by norm_num)
end

end greatest_integer_radius_l495_495678


namespace f_2011_8_l495_495667

-- Define the function f
def sum_of_digits (n : ℕ) : ℕ := n.digitSum

-- Define f as described in the problem
def f (n : ℕ) : ℕ := sum_of_digits (n^2 + 1)

-- Define the sequence of functions f_k
@[simp]
def f_k : ℕ → ℕ → ℕ
| 0, n := n
| (k+1), n := f (f_k k n)

-- Theorem to prove
theorem f_2011_8 : f_k 2011 8 = 11 :=
sorry

end f_2011_8_l495_495667


namespace incenter_on_altitude_of_acute_triangle_l495_495715

theorem incenter_on_altitude_of_acute_triangle
    (A B C B1 C1 A0 P Q : Type)
    [triangle : is_acute_angled_triangle A B C]
    [altitude_B : is_altitude A B B1]
    [altitude_C : is_altitude A C C1]
    [midpoint_A0 : is_midpoint A0 B C]
    [intersection_P : intersects_at_line A0 B1 (parallel_line_through A (line B C)) P]
    [intersection_Q : intersects_at_line A0 C1 (parallel_line_through A (line B C)) Q] :
    incenter_of_triangle P A0 Q lies_on_altitude A (line B C) := 
sorry

end incenter_on_altitude_of_acute_triangle_l495_495715


namespace latest_time_temp_decreasing_l495_495767

theorem latest_time_temp_decreasing (t : ℝ) 
  (h1 : -t^2 + 12 * t + 55 = 82) 
  (h2 : ∀ t0 : ℝ, -2 * t0 + 12 < 0 → t > t0) : 
  t = 6 + (3 * Real.sqrt 28 / 2) :=
sorry

end latest_time_temp_decreasing_l495_495767


namespace part_I_part_II_l495_495993

noncomputable def f (x a b : ℝ) : ℝ := Real.exp x - a * x^2 - 2 * x + b

theorem part_I (a : ℝ) (h : a > 0) :
  ∃ x : ℝ, (Real.exp x - 2 * a * x - 2) < 0 :=
by sorry

theorem part_II (a : ℝ) (h1 : a > 0) (h2 : ∀ x : ℝ, f x a b > 0) :
  ∃ (b_min : ℤ), (b_min = 0) ∧ ∀ b' : ℤ, b' ≥ b_min → ∀ x : ℝ, f x a b' > 0 :=
by sorry

end part_I_part_II_l495_495993


namespace largest_number_is_A_l495_495129

-- Definitions of the numbers
def numA := 8.45678
def numB := 8.456777777 -- This should be represented properly with an infinite sequence in a real formal proof
def numC := 8.456767676 -- This should be represented properly with an infinite sequence in a real formal proof
def numD := 8.456756756 -- This should be represented properly with an infinite sequence in a real formal proof
def numE := 8.456745674 -- This should be represented properly with an infinite sequence in a real formal proof

-- Lean statement to prove that numA is the largest number
theorem largest_number_is_A : numA > numB ∧ numA > numC ∧ numA > numD ∧ numA > numE :=
by
  -- Proof not provided, sorry to skip
  sorry

end largest_number_is_A_l495_495129


namespace hypercube_ant_paths_l495_495966

/-
Erin the ant starts at a given corner of a hypercube (4-dimensional cube) and crawls along exactly 15 edges
in such a way that she visits every corner exactly once and then finds that she is unable to return along
an edge to her starting point. Prove that the number of paths meeting these conditions is 24.
-/
theorem hypercube_ant_paths : ∃ (paths : Nat), paths = 24 ∧ 
  -- Erin starts at a given corner of a hypercube (4-dimensional cube)
  (∃ (starting_corner : Finvec 4), 
    -- She crawls along exactly 15 edges
    (∀ (edge_count : Nat), edge_count = 15) ∧
    -- She visits every corner exactly once
    (∀ (v : Finvec 4), v.distinct) ∧ 
    -- She is unable to return along an edge to her starting point
    (¬∃ (edge : Finvec 4), edge.start = starting_corner ∧ edge.end = starting_corner)) :=
by
  sorry

end hypercube_ant_paths_l495_495966


namespace rank_trigonometric_values_l495_495432

theorem rank_trigonometric_values :
  tan (7 * Real.pi / 5) > sin (2 * Real.pi / 5) ∧ sin (2 * Real.pi / 5) > cos (6 * Real.pi / 5) := 
by
  sorry

end rank_trigonometric_values_l495_495432


namespace obtuse_triangles_from_20_points_l495_495516

theorem obtuse_triangles_from_20_points :
  ∀ (P : Fin 20 → ℝ × ℝ),
    (∀ i j k : Fin 20, i ≠ j → j ≠ k → i ≠ k → ¬ collinear_three_points (P i) (P j) (P k)) →
    (number_of_obtuse_triangles P = 1140) :=
by
  -- Placeholder proof
  sorry

end obtuse_triangles_from_20_points_l495_495516


namespace proof_of_k_bound_l495_495033

noncomputable def sets_with_nonempty_intersection_implies_k_bound (k : ℝ) : Prop :=
  let M := {x : ℝ | -1 ≤ x ∧ x < 2}
  let N := {x : ℝ | x ≤ k + 3}
  M ∩ N ≠ ∅ → k ≥ -4

theorem proof_of_k_bound (k : ℝ) : sets_with_nonempty_intersection_implies_k_bound k := by
  intro h
  have : -1 ≤ k + 3 := sorry
  linarith

end proof_of_k_bound_l495_495033


namespace solution_inequality_l495_495085

theorem solution_inequality {x : ℝ} : x - 1 > 0 ↔ x > 1 := 
by
  sorry

end solution_inequality_l495_495085


namespace original_price_of_shirt_l495_495082

theorem original_price_of_shirt (discounted_price : ℝ) (discount_percentage : ℝ) 
  (h_discounted_price : discounted_price = 780) (h_discount_percentage : discount_percentage = 0.20) 
  : (discounted_price / (1 - discount_percentage) = 975) := by
  sorry

end original_price_of_shirt_l495_495082


namespace p_is_sufficient_not_necessary_for_q_l495_495994

-- Definitions for conditions p and q
def p (x : ℝ) := x^2 - x - 20 > 0
def q (x : ℝ) := 1 - x^2 < 0

-- The main statement
theorem p_is_sufficient_not_necessary_for_q:
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) :=
by
  sorry

end p_is_sufficient_not_necessary_for_q_l495_495994


namespace greatest_integer_radius_l495_495681

theorem greatest_integer_radius (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 := 
by {
  have h1 : r^2 < 75, from (div_lt_div_right (by linarith [pi_pos])).mp h,
  have sqrt_75_lt_9 : sqrt 75 < 9 := by norm_num,
  have sqrt_75_lt_9' : r < 9 := lt_of_pow_lt_pow_of_lt_two h1 sqrt_75_lt_9,
  exact le_of_lt (int.le_iff_coe.mp sqrt_75_lt_9')
}

end greatest_integer_radius_l495_495681


namespace geom_series_sum_l495_495854

theorem geom_series_sum : 
  let a₀ := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 5 in
  ∑ i in Finset.range n, a₀ * r ^ i = 341 / 1024 := 
  by
    sorry

end geom_series_sum_l495_495854


namespace annie_laps_bonnie_first_lap_l495_495562

theorem annie_laps_bonnie_first_lap :
  ∀ (v : ℕ) (t : ℕ),
    let d_bonnie := v * t,
        d_annie := 1.5 * v * t,
        track_length := 500 in
    d_annie = d_bonnie + track_length → 
    d_annie / track_length = 3 := 
by
  intros v t d_bonnie d_annie track_length h
  rw [h, one_div_two_mul, add_div, track_length_mul]
  sorry

end annie_laps_bonnie_first_lap_l495_495562


namespace lemonade_in_pitcher_l495_495555

theorem lemonade_in_pitcher (iced_tea lemonade total_pitcher total_in_drink lemonade_ratio : ℚ)
  (h1 : iced_tea = 1/4)
  (h2 : lemonade = 5/4)
  (h3 : total_in_drink = iced_tea + lemonade)
  (h4 : lemonade_ratio = lemonade / total_in_drink)
  (h5 : total_pitcher = 18) :
  (total_pitcher * lemonade_ratio) = 15 :=
by
  sorry

end lemonade_in_pitcher_l495_495555


namespace area_of_shaded_region_l495_495799

theorem area_of_shaded_region
  (A B C D : Point)
  (h_inscribed : is_inscribed_rectangle A B C D)
  (h_center : center_of_circle D)
  (h_B_on_circle : B_on_circle B D)
  (AD : ℝ)
  (CD : ℝ)
  (h_AD : AD = 5)
  (h_CD : CD = 4) :
  let AC := (AD^2 + CD^2) ^ (1/2)
  let area_circle := π * AC^2
  let area_quarter_circle := (1/4) * area_circle
  let area_rectangle := AD * CD
  let area_shaded := area_quarter_circle - area_rectangle in
  area_shaded = (41 * π / 4) - 20 :=
by sorry

end area_of_shaded_region_l495_495799


namespace greatest_integer_satisfying_inequality_l495_495264

theorem greatest_integer_satisfying_inequality : ∃ x : ℤ, (3 * x + 2 < 7 - 2 * x) ∧ (∀ y : ℤ, (3 * y + 2 < 7 - 2 * y) → y ≤ x) := by
  let x := 0
  use x
  split
  {
    linarith
  }
  {
    intros y hy
    linarith
  }

end greatest_integer_satisfying_inequality_l495_495264


namespace river_bend_students_more_than_pets_l495_495247

theorem river_bend_students_more_than_pets 
  (students_per_classroom : ℕ)
  (rabbits_per_classroom : ℕ)
  (hamsters_per_classroom : ℕ)
  (number_of_classrooms : ℕ)
  (total_students : ℕ := students_per_classroom * number_of_classrooms)
  (total_rabbits : ℕ := rabbits_per_classroom * number_of_classrooms)
  (total_hamsters : ℕ := hamsters_per_classroom * number_of_classrooms)
  (total_pets : ℕ := total_rabbits + total_hamsters) :
  students_per_classroom = 24 ∧ rabbits_per_classroom = 2 ∧ hamsters_per_classroom = 3 ∧ number_of_classrooms = 5 →
  total_students - total_pets = 95 :=
by
  sorry

end river_bend_students_more_than_pets_l495_495247


namespace count_valid_numbers_l495_495586

def is_valid_digit (d : ℕ) : Prop := d = 5 ∨ d = 7

def is_seven_digit_number (n : ℕ) : Prop := 
  1000000 ≤ n ∧ n < 10000000

def uses_valid_digits (n : ℕ) : Prop :=
  ∀ k, 0 ≤ k < 7 → is_valid_digit ((n / 10^k) % 10)

def divisible_by_35 (n : ℕ) : Prop :=
  n % 35 = 0

def valid_seven_digit_number (n : ℕ) : Prop :=
  is_seven_digit_number n ∧ uses_valid_digits n ∧ divisible_by_35 n

theorem count_valid_numbers : 
  ∃ (count : ℕ), count = 9 ∧ ∀ n, valid_seven_digit_number n ↔ n ∈ (1:fin count.succ) :=
sorry

end count_valid_numbers_l495_495586


namespace beta_value_l495_495295

theorem beta_value (α β : ℝ) (hα_acute : 0 < α ∧ α < π / 2) (hβ_acute : 0 < β ∧ β < π / 2)
  (h1 : sin α = sqrt 5 / 5)
  (h2 : sin (α - β) = -sqrt 10 / 10) :
  β = π / 4 :=
sorry

end beta_value_l495_495295


namespace rectangle_area_is_243_square_meters_l495_495181

-- Given conditions
variables (w l : ℝ)
-- The width is one-third of its length
def width_one_third_length : Prop := w = l / 3

-- The perimeter of the rectangle is 72 meters
def perimeter_72 : Prop := 2 * (w + l) = 72

-- The area of the rectangle
def area : ℝ := w * l

-- Proof statement
theorem rectangle_area_is_243_square_meters 
  (h1 : width_one_third_length w l) 
  (h2 : perimeter_72 w l) : area w l = 243 := 
by 
sorry

end rectangle_area_is_243_square_meters_l495_495181


namespace initial_customers_l495_495553

theorem initial_customers (x : ℕ) (h : x - 3 + 39 = 50) : x = 14 :=
by
  sorry

end initial_customers_l495_495553


namespace ceil_sqrt_250_eq_16_l495_495967

theorem ceil_sqrt_250_eq_16 : ⌈Real.sqrt 250⌉ = 16 :=
by
  have h1 : (15 : ℝ) < Real.sqrt 250 := sorry
  have h2 : Real.sqrt 250 < 16 := sorry
  exact sorry

end ceil_sqrt_250_eq_16_l495_495967


namespace probability_pq_condition_l495_495663

theorem probability_pq_condition :
  (∃ p q : ℤ, 1 ≤ p ∧ p ≤ 20 ∧ pq - 6p - 3q = 3) ↔ (fraction 3 20) :=
sorry

end probability_pq_condition_l495_495663


namespace trig_order_descending_l495_495433

theorem trig_order_descending :
  let θ₁ := 2 * Real.pi / 5
  let θ₂ := 6 * Real.pi / 5
  let θ₃ := 7 * Real.pi / 5
  θ₃ = Real.pi + θ₁ →
  sin θ₁ > 0 →
  cos θ₂ < 0 →
  tan θ₃ = tan θ₁ →
  θ₁ < Real.pi / 2 → 
  tan θ₁ > sin θ₁ →
  tan θ₃ > sin θ₁ ∧ sin θ₁ > cos θ₂ := by
  intros θ₁ θ₂ θ₃ h1 h2 h3 h4 h5 h6
  sorry

end trig_order_descending_l495_495433


namespace greatest_radius_l495_495689

theorem greatest_radius (r : ℕ) : (π * (r : ℝ)^2 < 75 * π) ↔ r ≤ 8 := 
by
  sorry

end greatest_radius_l495_495689


namespace integer_count_between_l495_495327

theorem integer_count_between (a b : ℝ) (ha : a = (10.5)^3) (hb : b = (10.6)^3) :
  (b.floor - a.ceil + 1 = 33) :=
by
  have h1 : a = 1157.625 := by rw [ha]; norm_num
  have h2 : b = 1191.016 := by rw [hb]; norm_num
  sorry

end integer_count_between_l495_495327


namespace imaginary_part_of_conjugate_l495_495072

-- Define the complex number
def complex_number : Complex := (2 + Complex.i) / (1 - 2 * Complex.i)

-- Define the conjugate of the complex number
def complex_conjugate : Complex := Complex.conj complex_number

-- State that the imaginary part of the conjugate is -1
theorem imaginary_part_of_conjugate : Complex.im complex_conjugate = -1 :=
sorry

end imaginary_part_of_conjugate_l495_495072


namespace fabric_difference_fabric_total_l495_495763

noncomputable def fabric_used_coat : ℝ := 1.55
noncomputable def fabric_used_pants : ℝ := 1.05

theorem fabric_difference : fabric_used_coat - fabric_used_pants = 0.5 :=
by
  sorry

theorem fabric_total : fabric_used_coat + fabric_used_ppants = 2.6 :=
by
  sorry

end fabric_difference_fabric_total_l495_495763


namespace sphere_surface_area_l495_495097

theorem sphere_surface_area (V : ℝ) (π : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : ∀ r, V = (4/3) * π * r^3)
  (h2 : V = 72 * π) : A = 36 * π * 2^(2/3) :=
by 
  sorry

end sphere_surface_area_l495_495097


namespace two_digit_product_l495_495435

theorem two_digit_product (A B C : ℕ) (hA : A < 10) (hB : B < 10) (hAB : A * 10 + B = 12) (hC : C < 10) : 
  let AB := A * 10 + B in
  let ACC := A * 100 + C * 10 + C in
  AB * AB = ACC :=
by
  sorry

end two_digit_product_l495_495435


namespace cylinder_radius_l495_495461

/-
The following defines the conditions from the problem:
1. The points A, A1, and D.
2. The geometry of the cylinder with its axis parallel to BD1.
3. The side length of the square base and the length of the lateral edges.
-/

structure Point := (x : ℝ) (y : ℝ) (z : ℝ)

def A : Point := ⟨0, 0, 0⟩
def A1 : Point := ⟨0, 0, real.sqrt 3⟩
def D : Point := ⟨0, 1, 0⟩
def BD1 : Point := ⟨1, 1, real.sqrt 3⟩

-- Define the is_parallel relation denoting parallel vectors
def is_parallel (v₁ v₂ : Point) : Prop :=
  ∃ λ : ℝ, v₂ = ⟨λ * v₁.x, λ * v₁.y, λ * v₁.z⟩

-- Define the cylinder based on its properties
structure Cylinder :=
  (radius : ℝ)
  (axis_parallel_to : Point)
  (points_on_surface : list Point)

-- Cylinder properties per problem conditions
def cylinder : Cylinder :=
  { radius := (4 * real.sqrt 2) / 5
  , axis_parallel_to := BD1
  , points_on_surface := [A, A1, D]
  }

theorem cylinder_radius :
  (cylinder.radius = (4 * real.sqrt 2) / 5) :=
  sorry

end cylinder_radius_l495_495461


namespace area_ratio_of_squares_l495_495076

-- Definition of squares, and their perimeters' relationship
def perimeter (side_length : ℝ) := 4 * side_length

theorem area_ratio_of_squares (a b : ℝ) (h : perimeter a = 4 * perimeter b) : (a * a) = 16 * (b * b) :=
by
  -- We assume the given condition
  have ha : a = 4 * b := sorry
  -- We then prove the area ratio
  sorry

end area_ratio_of_squares_l495_495076


namespace solve_for_y_l495_495981

theorem solve_for_y (y : ℝ) : sqrt (4 - 5 * y) = 8 → y = -12 := 
by
  sorry

end solve_for_y_l495_495981


namespace sum_of_squares_of_sines_of_degrees_is_ten_l495_495948

def sum_of_squares_of_sines_of_degrees : ℝ :=
  ∑ i in finset.range 19, real.sin (i * 10 * real.pi / 180) ^ 2

theorem sum_of_squares_of_sines_of_degrees_is_ten :
  sum_of_squares_of_sines_of_degrees = 10 := 
sorry

end sum_of_squares_of_sines_of_degrees_is_ten_l495_495948


namespace most_likely_outcome_l495_495605

/-- Four children were born at City Hospital yesterday. Assume each child is equally likely 
    to be a boy or a girl. Prove that the most likely outcome among the following is that 
    3 are of one gender and 1 is of the other gender.-/
theorem most_likely_outcome :
  let prob := λ (n : ℕ) => (Nat.choose 4 n) * (1/2 : ℚ)^4 in
  max (prob 0) (max (prob 4) (max (prob 2) ((prob 1) + (prob 1)))) = (prob 1) + (prob 1) :=
by
  sorry

end most_likely_outcome_l495_495605


namespace part1_part2_l495_495754

noncomputable def f (a x : ℝ) := Real.exp (2 * x) - 4 * a * Real.exp x - 2 * a * x
noncomputable def g (a x : ℝ) := x^2 + 5 * a^2
noncomputable def F (a x : ℝ) := f a x + g a x

theorem part1 (a : ℝ) : (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ a ≤ 0 :=
by sorry

theorem part2 (a : ℝ) : ∀ x : ℝ, F a x ≥ 4 * (1 - Real.log 2)^2 / 5 :=
by sorry

end part1_part2_l495_495754


namespace no_real_ordered_triples_l495_495267

theorem no_real_ordered_triples (x y z : ℝ) (h1 : x + y = 3) (h2 : xy - z^2 = 4) : false :=
sorry

end no_real_ordered_triples_l495_495267


namespace tangent_equal_inclination_l495_495005

noncomputable def f (x : ℝ) : ℝ := sorry -- assuming there's a function f with properties as given

variables {A B P : ℝ × ℝ} -- points on the curve
variables {f' : ℝ → ℝ} -- the derivative of f

-- Conditions:
-- f is continuous and has continuous derivative
axiom continuous_f : continuous f
axiom continuous_f' : continuous f'
-- arc \widehat{AB} is concave to the chord AB
axiom concave_arcs : ∀ (a b : ℝ), a ≤ b → concave_on (Icc a b) f

-- P is the point on the arc \widehat{AB} where AP + PB is maximal
axiom maximal_sum : ∀ (p : ℝ × ℝ),
  p ∈ segment ℝ A B →
  (∀ (q : ℝ × ℝ), q ∈ segment ℝ A B → dist A P + dist P B ≥ dist A q + dist q B)

-- Prove the angles condition
theorem tangent_equal_inclination :
  ∃ (t : ℝ), (tangent_ypoint f P t := ∀ (a b : ℝ × ℝ), a ≤ b → concave_on (Icc a b) f) :=
sorry

end tangent_equal_inclination_l495_495005


namespace greatest_radius_l495_495685

theorem greatest_radius (r : ℕ) : (π * (r : ℝ)^2 < 75 * π) ↔ r ≤ 8 := 
by
  sorry

end greatest_radius_l495_495685


namespace shop_owner_percentage_profit_l495_495503

section ShopOwnerProfit

variable {buy_cheat_percent : ℝ} (sell_cheat_percent : ℝ) (initial_weight : ℝ) (initial_price : ℝ)

def percentage_profit (buy_cheat_percent : ℝ) (sell_cheat_percent : ℝ) (initial_weight : ℝ) (initial_price : ℝ) : ℝ :=
  let actual_weight_buy := initial_weight * (1 + buy_cheat_percent / 100)
  let actual_weight_sell := initial_weight * (1 - sell_cheat_percent / 100)
  let sell_amount := actual_weight_buy / actual_weight_sell * initial_price
  let profit := sell_amount - initial_price
  (profit / initial_price) * 100

theorem shop_owner_percentage_profit :
  percentage_profit 14 20 100 100 = 42.5 :=
by
  sorry

end ShopOwnerProfit

end shop_owner_percentage_profit_l495_495503


namespace imaginary_part_conjugate_l495_495997

theorem imaginary_part_conjugate (z : ℂ) (h : z = (3 - complex.i) / (3 + complex.i)) :
  complex.im (conj z) = 3 / 5 :=
sorry

end imaginary_part_conjugate_l495_495997


namespace intersection_points_count_l495_495954

theorem intersection_points_count (B : ℝ) (hB : 0 < B) :
  ∃ p : ℕ, p = 4 ∧ (∀ x y : ℝ, (y = B * x^2 ∧ y^2 + 4 * y - 2 = x^2 + 5 * y) ↔ p = 4) := by
sorry

end intersection_points_count_l495_495954


namespace necessary_but_not_sufficient_l495_495281

variables (α β : Plane) (m : Line)

-- Define what it means for planes and lines to be perpendicular
def plane_perpendicular (p1 p2 : Plane) : Prop := sorry
def line_perpendicular_plane (l : Line) (p : Plane) : Prop := sorry

-- The main theorem to be established
theorem necessary_but_not_sufficient :
  (plane_perpendicular α β) → (line_perpendicular_plane m β) ∧ ¬ ((plane_perpendicular α β) ↔ (line_perpendicular_plane m β)) :=
sorry

end necessary_but_not_sufficient_l495_495281


namespace avg_last_three_l495_495809

theorem avg_last_three (a b c d e f g : ℝ)
  (h1 : (a + b + c + d + e + f + g) / 7 = 62)
  (h2 : (a + b + c + d) / 4 = 55) :
  (e + f + g) / 3 = 71.333333 :
by
  sorry

end avg_last_three_l495_495809


namespace not_equivalent_expression_l495_495496

/--
Let A, B, C, D be expressions defined as follows:
A := 3 * (x + 2)
B := (-9 * x - 18) / -3
C := (1/3) * (3 * x) + (2/3) * 9
D := (1/3) * (9 * x + 18)

Prove that only C is not equivalent to 3 * x + 6.
-/
theorem not_equivalent_expression (x : ℝ) :
  let A := 3 * (x + 2)
  let B := (-9 * x - 18) / -3
  let C := (1/3) * (3 * x) + (2/3) * 9
  let D := (1/3) * (9 * x + 18)
  C ≠ 3 * x + 6 :=
by
  intros A B C D
  sorry

end not_equivalent_expression_l495_495496


namespace distance_AC_l495_495793

variable (A B C : Type) [geometry.line (A, B, C)]
variable (dAB dBC : ℝ)
variable (hAB : dAB = 5)
variable (hBC : dBC = 4)

theorem distance_AC (h : on_same_line A B C) : 
  ∃ d : ℝ, (d = 1 ∨ d = 9) :=
  sorry

end distance_AC_l495_495793


namespace find_quaterns_l495_495973

theorem find_quaterns {
  x y z w : ℝ
} : 
  (x + y = z^2 + w^2 + 6 * z * w) → 
  (x + z = y^2 + w^2 + 6 * y * w) → 
  (x + w = y^2 + z^2 + 6 * y * z) → 
  (y + z = x^2 + w^2 + 6 * x * w) → 
  (y + w = x^2 + z^2 + 6 * x * z) → 
  (z + w = x^2 + y^2 + 6 * x * y) → 
  ( (x, y, z, w) = (0, 0, 0, 0) 
    ∨ (x, y, z, w) = (1/4, 1/4, 1/4, 1/4) 
    ∨ (x, y, z, w) = (-1/4, -1/4, 3/4, -1/4) 
    ∨ (x, y, z, w) = (-1/2, -1/2, 5/2, -1/2)
  ) :=
  sorry

end find_quaterns_l495_495973


namespace area_of_triangle_PQR_l495_495920

theorem area_of_triangle_PQR :
  ∀ (PS PR height: ℝ), PS = 4 → PR = 2 → height = 4 →
  (1 / 2) * PR * height = 4 :=
by
  intros PS PR height hPS hPR hheight
  rw [hPS, hPR, hheight]
  sorry

end area_of_triangle_PQR_l495_495920


namespace rectangular_field_area_l495_495189

theorem rectangular_field_area :
  ∃ (w l : ℝ), (l = 3 * w) ∧ (2 * (l + w) = 72) ∧ (l * w = 243) :=
by {
  sorry
}

end rectangular_field_area_l495_495189


namespace minimum_cost_to_buy_additional_sheets_l495_495494

def total_sheets : ℕ := 98
def students : ℕ := 12
def cost_per_sheet : ℕ := 450

theorem minimum_cost_to_buy_additional_sheets : 
  (students * (1 + total_sheets / students) - total_sheets) * cost_per_sheet = 4500 :=
by {
  sorry
}

end minimum_cost_to_buy_additional_sheets_l495_495494


namespace observations_count_correct_l495_495833

noncomputable def corrected_observations (n : ℕ) : ℕ :=
  if 36 * n + 22 = 36.5 * n then n else 0

theorem observations_count_correct :
  ∃ n : ℕ, 36 * n + 22 = 36.5 * n ∧ corrected_observations n = 44 :=
by {
  sorry
}

end observations_count_correct_l495_495833


namespace find_M_range_of_a_l495_495645

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + a + 2

noncomputable def M (a : ℝ) : ℝ :=
  if a ≥ 0 then 2 * a^2 + 3 * a + 3
  else 2 * a^2 - a + 3

noncomputable def g (x a : ℝ) : ℝ := f x a + |x^2 - 1|

theorem find_M (a : ℝ) : M(a) = if a ≥ 0 then 2 * a^2 + 3 * a + 3 else 2 * a^2 - a + 3 := 
by {
  sorry
}

theorem range_of_a (a : ℝ) : (∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < 3 ∧ g x1 a = 0 ∧ g x2 a = 0) ↔ (1 + Real.sqrt 3 < a ∧ a < 19 / 5) := 
by {
  sorry
}

end find_M_range_of_a_l495_495645


namespace middle_three_cards_sum_l495_495051

def five_red_cards : List ℕ := [1, 2, 3, 4, 5]
def four_blue_cards : List ℕ := [3, 4, 5, 6]
def alternating_stack (ls : List ℕ) : Prop :=
  ∀ i, i < ls.length - 1 → (ls.nth_le i sorry ∈ five_red_cards ↔ ls.nth_le (i + 1) sorry ∈ four_blue_cards)

def divides_neighbours (ls : List ℕ) : Prop :=
  ∀ i, i < ls.length - 1 →
  ((ls.nth_le i sorry ∈ five_red_cards → ls.nth_le (i+1) sorry % ls.nth_le i sorry = 0) ∧
   (ls.nth_le (i+1) sorry ∈ five_red_cards → ls.nth_le i sorry % ls.nth_le (i+1) sorry = 0))

def valid_stack (ls : List ℕ) : Prop :=
  alternating_stack ls ∧ divides_neighbours ls

theorem middle_three_cards_sum (s : List ℕ) (h1 : valid_stack s) (h2 : s.length = 9):
    s.nth_le 3 sorry + s.nth_le 4 sorry + s.nth_le 5 sorry = 12 :=
sorry

end middle_three_cards_sum_l495_495051


namespace tangent_abscissa_l495_495544

theorem tangent_abscissa (x₀ : ℝ) :
  ∃ x₀, (y = (x₀ + 2) * exp x₀) ∧ (tangent_eq := λ(x₀ : ℝ), (λ x, (x₀ + 3) * exp x₀ * (x - x₀) + (x₀ + 2) * exp x₀))
  (tangent_eq 0 = 0) ↔ (x₀ = -1 + real.sqrt 3 ∨ x₀ = -1 - real.sqrt 3) := by
  sorry

end tangent_abscissa_l495_495544


namespace palindromic_primes_sum_l495_495772

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.reverse.foldl (λ acc d, acc * 10 + d) 0

def is_palindromic_prime (n : ℕ) : Prop :=
  n < 100 ∧ 10 ≤ n ∧ is_prime n ∧ is_prime (reverse_digits n)

def sum_of_palindromic_primes : ℕ :=
  ((list.range 100).filter is_palindromic_prime).sum

theorem palindromic_primes_sum : sum_of_palindromic_primes = 429 := 
  by 
    sorry

end palindromic_primes_sum_l495_495772


namespace greatest_integer_radius_l495_495680

theorem greatest_integer_radius (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 := 
by {
  have h1 : r^2 < 75, from (div_lt_div_right (by linarith [pi_pos])).mp h,
  have sqrt_75_lt_9 : sqrt 75 < 9 := by norm_num,
  have sqrt_75_lt_9' : r < 9 := lt_of_pow_lt_pow_of_lt_two h1 sqrt_75_lt_9,
  exact le_of_lt (int.le_iff_coe.mp sqrt_75_lt_9')
}

end greatest_integer_radius_l495_495680


namespace surface_area_of_sphere_l495_495091

noncomputable def volume : ℝ := 72 * Real.pi

theorem surface_area_of_sphere (r : ℝ) (h : (4 / 3) * Real.pi * r^3 = volume) :
  4 * Real.pi * r^2 = 36 * Real.pi * (Real.cbrt 2)^2 :=
by
  sorry

end surface_area_of_sphere_l495_495091


namespace sphere_surface_area_l495_495095

theorem sphere_surface_area (V : ℝ) (π : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : ∀ r, V = (4/3) * π * r^3)
  (h2 : V = 72 * π) : A = 36 * π * 2^(2/3) :=
by 
  sorry

end sphere_surface_area_l495_495095


namespace combined_tax_rate_l495_495889

theorem combined_tax_rate (Mork_income Mindy_income : ℝ) (h1 : Mindy_income = 3 * Mork_income)
  (tax_Mork tax_Mindy : ℝ) (h2 : tax_Mork = 0.10 * Mork_income) (h3 : tax_Mindy = 0.20 * Mindy_income)
  : (tax_Mork + tax_Mindy) / (Mork_income + Mindy_income) = 0.175 :=
by
  sorry

end combined_tax_rate_l495_495889


namespace decorate_eggs_time_calculation_l495_495759

/-- Definition of Mia's and Billy's egg decorating rates, total number of eggs to be decorated, and the calculated time when working together --/
def MiaRate : ℕ := 24
def BillyRate : ℕ := 10
def totalEggs : ℕ := 170
def combinedRate : ℕ := MiaRate + BillyRate

theorem decorate_eggs_time_calculation :
  (totalEggs / combinedRate) = 5 := by
  sorry

end decorate_eggs_time_calculation_l495_495759


namespace decreasing_power_function_l495_495582

theorem decreasing_power_function (m : ℝ) :
  (∀ x > 0, f(x) > f(y) if x < y) ∧ (m^2 - 4m + 4 > 0) ∧ (m^2 - 6m + 8 < 0) ↔ (m = 3) := 
sorry

end decreasing_power_function_l495_495582


namespace g_is_odd_l495_495724

def g (x : ℝ) : ℝ := log (x - sqrt (1 + x^2))

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  intros x
  sorry

end g_is_odd_l495_495724


namespace smallest_shift_l495_495804

-- Defines the periodic property of the function f
def periodic (f : ℝ → ℝ) (T : ℝ) :=
  ∀ x, f (x - T) = f x

-- Defines the required property for the function g, derived from f
def shifted_property (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x, f ((x - a) / 6) = f (x / 6)

theorem smallest_shift (f : ℝ → ℝ) (h : periodic f 30) : 
  ∃ a > 0, shifted_property f a ∧ ∀ b, b < a → ¬ shifted_property f b :=
by
  use 180
  split
  -- To show 180 > 0
  show 180 > 0,
  from by norm_cast; exact dec_trivial
  
  split
  -- To show the property holds for a = 180
  show shifted_property f 180,
  from sorry  -- Proof goes here
  
  -- To show no smaller positive shift works
  show ∀ b, b < 180 → ¬ shifted_property f b,
  from sorry  -- Proof goes here

end smallest_shift_l495_495804


namespace determine_x_l495_495577

theorem determine_x :
  ∃ x : ℕ, 1^(x+3) + 2^x + 3^(x+1) + 4^(x-1) = 272 ∧ x = 2 :=
begin
  use 2,
  sorry,
end

end determine_x_l495_495577


namespace union_condition_implies_l495_495623

-- Define set A as per the given condition
def setA : Set ℝ := { x | x * (x - 1) ≤ 0 }

-- Define set B as per the given condition with parameter a
def setB (a : ℝ) : Set ℝ := { x | Real.log x ≤ a }

-- Given condition A ∪ B = A, we need to prove that a ≤ 0
theorem union_condition_implies (a : ℝ) (h : setA ∪ setB a = setA) : a ≤ 0 := 
by
  sorry

end union_condition_implies_l495_495623


namespace find_f_neg2008_add_2009_l495_495637

def f (x : ℝ) : ℝ := if h : x ∈ Set.Ico 0 2 then Real.log 2 (x + 1) else sorry

theorem find_f_neg2008_add_2009 :
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, x ≥ 0 → f (x + 2) = f x) →
  (∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x = Real.log 2 (x + 1)) →
  f (-2008) + f 2009 = 1 :=
by
  intros
  sorry

end find_f_neg2008_add_2009_l495_495637


namespace mouse_lives_correct_l495_495908

def cat_lives : ℕ := 9
def dog_lives : ℕ := cat_lives - 3
def mouse_lives : ℕ := dog_lives + 7

theorem mouse_lives_correct : mouse_lives = 13 :=
by
  sorry

end mouse_lives_correct_l495_495908


namespace exists_circle_touching_given_circles_and_line_l495_495810

-- Define the given radii
def r1 := 1
def r2 := 3
def r3 := 4

-- Prove that there exists a circle with a specific radius touching the given circles and line AB
theorem exists_circle_touching_given_circles_and_line (x : ℝ) :
  ∃ (r : ℝ), r > 0 ∧ (r + r1) = x ∧ (r + r2) = x ∧ (r + r3) = x :=
sorry

end exists_circle_touching_given_circles_and_line_l495_495810


namespace magnitude_of_F1_correct_l495_495306

noncomputable def angle_between_forces (F1 F2 : ℝ) 
  (angle_F1_F2 : ℝ) (resultant_angle_F1 : ℝ) 
  (magnitude_resultant : ℝ) : ℝ :=
  if angle_F1_F2 = π / 2 ∧ resultant_angle_F1 = π / 4 ∧ magnitude_resultant = 10 then
    5 * Real.sqrt 2
  else 
    sorry

theorem magnitude_of_F1_correct (F1 F2 : ℝ) (angle_F1_F2 : ℝ) 
  (resultant_angle_F1 : ℝ) (magnitude_resultant : ℝ) :
  angle_F1_F2 = π / 2 ∧ resultant_angle_F1 = π / 4 ∧ magnitude_resultant = 10 →
  F1 = angle_between_forces F1 F2 angle_F1_F2 resultant_angle_F1 magnitude_resultant :=
begin
  intro h,
  simp [angle_between_forces, h],
end

end magnitude_of_F1_correct_l495_495306


namespace div_by_prime_l495_495018

theorem div_by_prime (p q : ℕ) (S_q : ℚ) (m n : ℤ) (hp_prime : Prime p) (hp_odd : Odd p)
  (hq_def : q = (3 * p - 5) / 2) 
  (hsq_def : S_q = ∑ k in Finset.range (q + 1), (1 : ℚ) / (k * (k + 1) * (k + 2)))
  (h_eq : (1 : ℚ) / p - 2 * S_q = (m : ℚ) / n) : 
  p ∣ (m - n) := 
sorry

end div_by_prime_l495_495018


namespace living_room_curtain_length_l495_495221

theorem living_room_curtain_length :
  let length_bolt := 16
  let width_bolt := 12
  let area_bolt := length_bolt * width_bolt
  let area_left := 160
  let area_cut := area_bolt - area_left
  let length_bedroom := 2
  let width_bedroom := 4
  let area_bedroom := length_bedroom * width_bedroom
  let area_living_room := area_cut - area_bedroom
  let width_living_room := 4
  area_living_room / width_living_room = 6 :=
by
  sorry

end living_room_curtain_length_l495_495221


namespace geometric_series_sum_l495_495863

theorem geometric_series_sum :
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  (a + a * r + a * r^2 + a * r^3 + a * r^4) = 341 / 1024 :=
by
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  -- Proof steps go here
  sorry

end geometric_series_sum_l495_495863


namespace bowling_average_l495_495657

theorem bowling_average (gretchen_score mitzi_score beth_score : ℤ) (h1 : gretchen_score = 120) (h2 : mitzi_score = 113) (h3 : beth_score = 85) :
  (gretchen_score + mitzi_score + beth_score) / 3 = 106 :=
by
  sorry

end bowling_average_l495_495657


namespace inequality_solution_l495_495257

open Real

theorem inequality_solution :
  {x | x ∈ Icc (-π/4) (7 * π / 4) ∧ (sin x)^2018 + (cos x)^(-2019) ≥ (cos x)^2018 + (sin x)^(-2019)}
  = {x | x ∈ Ico (-π/4) 0 ∪ Ico π/4 (π/2) ∪ Ioc π (5 * π / 4) ∪ Ioc (3 * π / 2) (7 * π / 4)} :=
sorry

end inequality_solution_l495_495257


namespace star_inequalities_impossible_l495_495427

theorem star_inequalities_impossible
  (A B C D E F G H I K : ℝ)
  (H₁ : A < B) (H₂ : B < C) (H₃ : C < D)
  (H₄ : D < E) (H₅ : E < F)
  (H₆ : F < G) (H₇ : G < H)
  (H₈ : H < I) (H₉ : I < K)
  (H₁₀ : K < A) :
  False :=
begin
  sorry
end

end star_inequalities_impossible_l495_495427


namespace intersection_M_N_l495_495321

open Set

def M : Set ℝ := {y | ∃ x : ℝ, y = 2^x}
def N : Set ℝ := {y | ∃ x : ℝ, y = real.sqrt (2 * x - x^2)}

theorem intersection_M_N : M ∩ N = {y | 0 < y ∧ y ≤ 2} :=
by
  sorry

end intersection_M_N_l495_495321


namespace max_value_of_quadratic_l495_495595

theorem max_value_of_quadratic (t : ℝ) (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) :
  let y := x^2 - 2 * t * x + 1 in
  (t < -1 ∨ (-1 ≤ t ∧ t < 0)) → y ≤ -2 * t + 2 ∧
  (t = 0) → y ≤ 2 ∧
  (0 < t ∧ t ≤ 1 ∨ t > 1) → y ≤ 2 * t + 2 :=
sorry

end max_value_of_quadratic_l495_495595


namespace find_OC_l495_495990

-- Define the set up and conditions
section
variables {ABC : Type} {A B C P O : ABC}
variables [field ABC] [has_measure ABC] [metric_space ABC]

-- Assume triangle ABC is a right triangle with angle ACB = 90 degrees
def right_triangle (A B C : ABC) : Prop :=
  ∃ (u v w : ℕ), ∠ACB = 90 ∧ ∥u∥ = A ∧ ∥v∥ = B ∧ ∥w∥ = C

-- Assume the lengths and perpendicular properties
def orthogonal_projection (P O : ABC) (ABC : set ABC) : Prop := ∀ Q ∈ ABC, ∠POQ = 90
def PA_eq_PB_eq_PC (P A B C : ABC) : Prop := ∥P - A∥ = ∥P - B∥ ∧ ∥P - B∥ = ∥P - C∥
def AB_length (A B : ABC) (l : ℕ) : Prop := ∥A - B∥ = l

-- Auxiliary definitions
def midpoint (A B : ABC) : ABC := A + B / 2
def is_circumcenter (O A B C : ABC) : Prop := ∥O - A∥ = ∥O - B∥ ∧ ∥O - B∥ = ∥O - C∥

-- The main mathematical statement
theorem find_OC
  (h_rt : right_triangle A B C)
  (h_pa_pb_pc : PA_eq_PB_eq_PC P A B C)
  (h_po_perp_abc : orthogonal_projection P O {A, B, C})
  (h_ab_len : AB_length A B 8)
  (h_O_midpoint_AB : O = midpoint A B) :
  ∥O - C∥ = 4 :=
begin
  sorry
end
end

end find_OC_l495_495990


namespace problem_1_problem_2_problem_3_l495_495519

section basketball_team

-- Definition of conditions
def num_games : ℕ := 6
def prob_win : ℚ := 1 / 3
def prob_loss : ℚ := 2 / 3

-- Problem 1
theorem problem_1 : 
  (prob_loss ^ 2 * prob_win) = 4 / 27 := sorry

-- Problem 2
theorem problem_2 : 
  (nat.choose num_games 3 * (prob_win ^ 3) * (prob_loss ^ 3)) = 160 / 729 := sorry

-- Problem 3
theorem problem_3 : 
  (num_games * prob_win) = 2 := sorry

end basketball_team

end problem_1_problem_2_problem_3_l495_495519


namespace matrix_multiplication_correct_l495_495951

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![
    ![2, 3, 1],
    ![7, -1, 0],
    ![0, 4, -2]
  ]

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![
    ![1, -5, 2],
    ![0, 4, 3],
    ![1, 0, -1]
  ]

def C : Matrix (Fin 3) (Fin 3) ℤ :=
  ![
    ![3, 2, 12],
    ![7, -39, 11],
    ![-2, 16, 14]
  ]

theorem matrix_multiplication_correct : A.mul B = C := by
  sorry

end matrix_multiplication_correct_l495_495951


namespace solve_for_x_l495_495801

theorem solve_for_x : ∃ x : ℝ, 3^(3*x + 2) = 1/27 ∧ x = -5/3 :=
by
  exists (-5/3)
  split
  · sorry
  · sorry

end solve_for_x_l495_495801


namespace max_noncotinaining_subsets_theorem_l495_495402

open Finset

noncomputable def max_noncotinaining_subsets (n : ℕ) : ℕ :=
  (nat.choose n (n / 2))

theorem max_noncotinaining_subsets_theorem (S : Finset ℕ) (n : ℕ) (A : Finset (Finset ℕ))
  (hS : S = (range n).map nat.succ)
  (hA : ∀ {a b}, a ∈ A → b ∈ A → (a ⊆ b ∨ b ⊆ a) → a = b) :
  A.card ≤ max_noncotinaining_subsets n :=
sorry

end max_noncotinaining_subsets_theorem_l495_495402


namespace find_a_l495_495345

-- Definition of the conditions
def condition1 (a b : ℝ) : Prop :=
  a * Real.sqrt 2 - Real.sqrt b = Real.sqrt 2

def condition2 (b : ℝ) : Prop :=
  b = Real.floor (2 * Real.pi + 2)

-- The main theorem we want to prove
theorem find_a (a b : ℝ) (h1 : condition1 a b) (h2 : condition2 b) : a = 3 :=
by
  sorry

end find_a_l495_495345


namespace prove_inequality_l495_495513

noncomputable def inequality_proof (n : ℕ) (x : Fin n → ℝ) (hx : ∀ i, 0 < x i) 
(hsum : (∑ i in Fin.range n, x i) = n) : Prop :=
  (∑ i in Fin.range n, i / (1 + ∑ j in Fin.range i, x i ^ j)) 
  ≤ 
  (∑ i in Fin.range n, (i + 1) / (1 + ∑ j in Fin.range (i + 1), x i ^ j))

theorem prove_inequality (n : ℕ) (x : Fin n → ℝ) 
  (hx : ∀ i, 0 < x i) 
  (hne : 3 ≤ n) 
  (hsum : (∑ i in Fin.range n, x i) = n) : 
  inequality_proof n x hx hsum :=
sorry

end prove_inequality_l495_495513


namespace area_of_rectangle_l495_495184

theorem area_of_rectangle (w l : ℝ) (h₁ : w = l / 3) (h₂ : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end area_of_rectangle_l495_495184


namespace quadratic_equation_with_means_l495_495700

theorem quadratic_equation_with_means (a b : ℝ) (h1 : (a + b) = 15) (h2 : a * b = 144) :
  ∃ p q, (p = a ∨ p = b) ∧ (q = a ∨ q = b) ∧ p ≠ q ∧ polynomial.Coeffs (X^2 - 15 * X + 144) == [1,-15,144] :=
by
  sorry

end quadratic_equation_with_means_l495_495700


namespace sphere_surface_area_l495_495096

theorem sphere_surface_area (V : ℝ) (π : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : ∀ r, V = (4/3) * π * r^3)
  (h2 : V = 72 * π) : A = 36 * π * 2^(2/3) :=
by 
  sorry

end sphere_surface_area_l495_495096


namespace simplest_quadratic_radical_is_D_l495_495557

theorem simplest_quadratic_radical_is_D :
  ∀ (A B C D : ℝ), A = real.sqrt 12 → B = real.sqrt (3 / 2) → C = (1 / real.sqrt 5) → D = -real.sqrt 2 →
  D = -real.sqrt 2 :=
by
  intros A B C D hA hB hC hD
  exact hD

# Note that the proof is trivial as the condition D = -real.sqrt 2 directly satisfies the conclusion. The essence of the proof is in satisfying the condition.

end simplest_quadratic_radical_is_D_l495_495557


namespace harry_worked_36_hours_l495_495137

def harry_pay (H x : ℤ) : ℤ :=
if H <= 24 then H * x else 24 * x + (H - 24) * (3 * x / 2)

def james_pay (x : ℤ) : ℤ :=
40 * x + (41 - 40) * 2 * x

theorem harry_worked_36_hours
  (x : ℤ) (H : ℤ)
  (HarryPay JamesPay SamePay : Prop) :
  HarryPay = harry_pay H x →
  JamesPay = james_pay x →
  SamePay = (harry_pay H x = james_pay x) →
  H = 36 := 
by
  intros hPay jPay samePay
  rw [hPay, jPay, samePay]
  sorry

end harry_worked_36_hours_l495_495137


namespace average_speed_is_correct_l495_495165

constant length1 : ℝ := 250
constant length2 : ℝ := 200
constant length3 : ℝ := 145
constant time1 : ℝ := 3
constant time2 : ℝ := 2
constant time3 : ℝ := 1

def total_distance : ℝ := length1 + length2 + length3
def total_time_minutes : ℝ := time1 + time2 + time3
def total_time_hours : ℝ := total_time_minutes / 60
def total_distance_kilometers : ℝ := total_distance / 1000
def average_speed : ℝ := total_distance_kilometers / total_time_hours

theorem average_speed_is_correct : average_speed = 5.95 := by
  sorry

end average_speed_is_correct_l495_495165


namespace rectangular_field_area_l495_495193

theorem rectangular_field_area :
  ∃ (w l : ℝ), (l = 3 * w) ∧ (2 * (l + w) = 72) ∧ (l * w = 243) :=
by {
  sorry
}

end rectangular_field_area_l495_495193


namespace general_term_of_A_inter_B_l495_495034

def setA : Set ℕ := { n*n + n | n : ℕ }
def setB : Set ℕ := { 3*m - 1 | m : ℕ }

theorem general_term_of_A_inter_B (k : ℕ) :
  let a_k := 9*k^2 - 9*k + 2
  a_k ∈ setA ∩ setB ∧ ∀ n ∈ setA ∩ setB, n = a_k :=
sorry

end general_term_of_A_inter_B_l495_495034


namespace infinite_points_with_sum_of_squares_condition_l495_495570

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a circle centered at origin with given radius
def isWithinCircle (P : Point2D) (r : ℝ) :=
  P.x^2 + P.y^2 ≤ r^2

-- Define the distance squared from a point to another point
def dist2 (P Q : Point2D) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the problem
theorem infinite_points_with_sum_of_squares_condition :
  ∃ P : Point2D, isWithinCircle P 1 → (dist2 P ⟨-1, 0⟩ + dist2 P ⟨1, 0⟩ = 3) :=
by  
  sorry

end infinite_points_with_sum_of_squares_condition_l495_495570


namespace sum_first_20_odds_is_400_l495_495492

-- Define the n-th odd positive integer
def odd_integer (n : ℕ) : ℕ := 2 * n + 1

-- Define the sum of the first n odd positive integers as a function
def sum_first_n_odds (n : ℕ) : ℕ := (n * (2 * n + 1)) / 2

-- Theorem statement: sum of the first 20 odd positive integers is 400
theorem sum_first_20_odds_is_400 : sum_first_n_odds 20 = 400 := 
  sorry

end sum_first_20_odds_is_400_l495_495492


namespace triangle_area_is_24_l495_495840

-- Define the vertices
def vertex1 : ℝ × ℝ := (3, 2)
def vertex2 : ℝ × ℝ := (3, -4)
def vertex3 : ℝ × ℝ := (11, -4)

-- Define a function to calculate the area of a triangle given its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1))

-- Prove the area of the triangle with the given vertices is 24 square units
theorem triangle_area_is_24 : triangle_area vertex1 vertex2 vertex3 = 24 := by
  sorry

end triangle_area_is_24_l495_495840


namespace gcd_problem_l495_495950

def a := 47^11 + 1
def b := 47^11 + 47^3 + 1

theorem gcd_problem : Nat.gcd a b = 1 := 
by
  sorry

end gcd_problem_l495_495950


namespace sum_of_middle_three_cards_l495_495049

theorem sum_of_middle_three_cards : ∃ (R B : list ℕ), 
  (R = [1, 2, 3, 4, 5] ∧ B = [3, 4, 5, 6]) →
  (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 5 → R.nth i ≠ none → 
  (∃ (j : ℕ), 1 ≤ j ∧ j ≤ 4 ∧ B.nth j ≠ none ∧ (¬ (j = 0 ∧ i = 5) ∧ ¬ (j = 4 ∧ i = 5)) ∧ R.nth i = B.nth j)) →
  R.nth 1 == some 1 →
  R.nth 4 == some 4 →
  (R.nth 3 == some 3 ∧ B.nth 2 == some 3 ∧ B.nth 2 == some 6) →
  ∑ (R, B), R = 5 ∧ B = 6.

end sum_of_middle_three_cards_l495_495049


namespace average_xyz_l495_495348

theorem average_xyz (x y z : ℝ) (h1 : x = 3) (h2 : y = 2 * x) (h3 : z = 3 * y) : 
  (x + y + z) / 3 = 9 :=
by
  sorry

end average_xyz_l495_495348


namespace find_y_l495_495126

theorem find_y (x y : ℕ) (hx_positive : 0 < x) (hy_positive : 0 < y) (hmod : x % y = 9) (hdiv : (x : ℝ) / (y : ℝ) = 96.25) : y = 36 :=
sorry

end find_y_l495_495126


namespace relationship_among_a_b_c_l495_495284

noncomputable def a : ℝ := Real.tan (Real.pi / 180 * 50)
noncomputable def b : ℝ := 1 + Real.cos (Real.pi / 180 * 20)
noncomputable def c : ℝ := 2 * Real.sin (Real.pi / 180 * 160)

theorem relationship_among_a_b_c : c < a ∧ a < b := by
  have hc : c = 2 * Real.sin (Real.pi / 180 * 20) := by
    rw [Real.sin (Real.pi - (Real.pi / 180 * 20))]
    rw [Real.sin (Real.pi / 180 * 20)]
    sorry
  sorry

end relationship_among_a_b_c_l495_495284


namespace total_marbles_l495_495556

variable (Albert Angela Allison Addison Alex : ℕ)

constants (h1 : Albert = 3 * Angela)
          (h2 : Angela = Allison + 8)
          (h3 : Allison = 28)
          (h4 : Addison = 2 * Albert)
          (h5 : Alex = Allison + 5)
          (h6 : Alex = Angela / 2)

theorem total_marbles :
  Albert + Angela + Allison + Addison + Alex = 421 := by
  sorry

end total_marbles_l495_495556


namespace inequality_solution_l495_495448

theorem inequality_solution
  (x : ℝ) :
  x ∉ {2, 3, 4, 5, 6, 7} →
  ((x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0 ↔ 
  (x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 6) ∨ (7 < x)) :=
by
  -- Proof goes here
  sorry

end inequality_solution_l495_495448


namespace absolute_value_positive_l495_495879

theorem absolute_value_positive (a : ℝ) (h : a ≠ 0) : |a| > 0 := by
  sorry

end absolute_value_positive_l495_495879


namespace ratio_c_to_d_l495_495349

theorem ratio_c_to_d (a b c d : ℚ) 
  (h1 : a / b = 3 / 4) 
  (h2 : b / c = 7 / 9) 
  (h3 : a / d = 0.4166666666666667) : 
  c / d = 5 / 7 := 
by
  -- Proof not needed
  sorry

end ratio_c_to_d_l495_495349


namespace cube_root_equality_l495_495870

/-- Given condition: c is defined as the sum of four terms each being 5^7. -/
def c : ℝ := 5^7 + 5^7 + 5^7 + 5^7

/-- Main theorem: The cube root of c equals 100 times the cube root of 10. -/
theorem cube_root_equality : real.cbrt c = 100 * real.cbrt 10 := by
  sorry

end cube_root_equality_l495_495870


namespace quadratic_interval_inequality_l495_495639

theorem quadratic_interval_inequality (a b c : ℝ) :
  (∀ x : ℝ, -1 / 2 < x ∧ x < 2 → a * x^2 + b * x + c > 0) →
  a < 0 ∧ c > 0 :=
sorry

end quadratic_interval_inequality_l495_495639


namespace sequence_definition_l495_495604

theorem sequence_definition (n : ℕ) (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = n^2 + n + 1) →
  (∀ a, a 0 = 0 ∧ (a 1 = 3 ∧ ∀ n, n ≥ 2 → a n = S n - S (n-1))) →
  (a n = if n = 1 then 3 else 2 * n) := 
by
  intros hS ha
  sorry

end sequence_definition_l495_495604


namespace find_breadth_l495_495703

theorem find_breadth (p l : ℕ) (h_p : p = 600) (h_l : l = 100) (h_perimeter : p = 2 * (l + b)) : b = 200 :=
by
  sorry

end find_breadth_l495_495703


namespace cos_C_is_neg_half_l495_495361

noncomputable def sin_cos_ratio (A B C : ℝ) (sinA sinB sinC : ℝ) :=
  ∃ k : ℝ, sinA = 3 * k ∧ sinB = 5 * k ∧ sinC = 7 * k

noncomputable def cos_of_angle_C (A B C : ℝ) (sinA sinB sinC : ℝ) :=
  \cos C

theorem cos_C_is_neg_half (A B C : ℝ) (sinA sinB sinC : ℝ) :
  sin_cos_ratio A B C sinA sinB sinC →
  cos_of_angle_C A B C sinA sinB sinC = -1 / 2 :=
begin
  intro h,
  sorry
end

end cos_C_is_neg_half_l495_495361


namespace remainder_mod_1000_l495_495749

-- Definitions
def p (x : ℕ) : ℕ := x^2020 + x^2019 + x^2018 + ⋯ + x + 1

def divisor (x : ℕ) : ℕ := x^5 + x^4 + 2x^3 + x^2 + x + 1

noncomputable def remainder_polynomial (x : ℕ) : ℕ :=
  let q := p(x) / divisor(x)
  p(x) - q * divisor(x)

-- Proof Statement
theorem remainder_mod_1000 :
  abs (remainder_polynomial 2020) % 1000 = 21 := by
    sorry

end remainder_mod_1000_l495_495749


namespace cube_root_of_sum_l495_495872

theorem cube_root_of_sum (x : ℝ) (a : ℝ) (h : x = 5^7) (k : a = 4 * x) :
  (∛a) = 2^(2/3) * 5^(7/3) :=
by
  sorry

end cube_root_of_sum_l495_495872


namespace comparison_of_a_b_c_l495_495303

theorem comparison_of_a_b_c (a b c : ℝ) (h_a : a = Real.log 2) (h_b : b = 5^(-1/2 : ℝ)) (h_c : c = Real.sin (Real.pi / 6)) : 
  b < c ∧ c < a :=
by
  sorry

end comparison_of_a_b_c_l495_495303


namespace regular_hexagon_AX_length_l495_495056

theorem regular_hexagon_AX_length {A B C D E F X : ℝ} (h : real) 
  (h_hex : is_regular_hexagon A B C D E F) 
  (h_side : dist A B = 4) 
  (h_BX : dist B X = 2 * dist A B) : 
  dist A X = 2 * real.sqrt 26 :=
by
  sorry

end regular_hexagon_AX_length_l495_495056


namespace number_of_correct_propositions_l495_495930

theorem number_of_correct_propositions :
  let P := ∀ x : ℝ, sin x ≤ 1 
  let Q := ∀ a b : ℝ, a < b → a^2 < b^2 
  let R := ∀ x : ℝ, ¬ (x^2 - x > 0)
  let S := ∀ x : ℝ, x > 2 → x^2 > 4 
  let P1 := ¬ (P ↔ Q) 
  let P2 := P → ¬ Q 
  let P3 := ¬ ∃ x : ℝ, x^2 - x > 0 ↔ ∀ x : ℝ, x^2 - x ≤ 0
  let P4 := (∀ x : ℝ, x > 2 ↔ x^2 > 4)
  (P1 ∧ ¬P2 ∧ P3 ∧ ¬P4) :=
by
  sorry

end number_of_correct_propositions_l495_495930


namespace miriam_pushups_l495_495036

theorem miriam_pushups :
  let p_M := 5
  let p_T := 7
  let p_W := 2 * p_T
  let p_Th := (p_M + p_T + p_W) / 2
  let p_F := p_M + p_T + p_W + p_Th
  p_F = 39 := by
  sorry

end miriam_pushups_l495_495036


namespace rectangular_field_area_l495_495176

theorem rectangular_field_area :
  ∃ (w l : ℝ), w = l / 3 ∧ 2 * (w + l) = 72 ∧ w * l = 243 :=
by
  sorry

end rectangular_field_area_l495_495176


namespace find_days_l495_495344

variables (a d e k m : ℕ) (y : ℕ)

-- Assumptions based on the problem
def workers_efficiency_condition : Prop := 
  (a * e * (d * k) / (a * e)) = d

-- Conclusion we aim to prove
def target_days_condition : Prop :=
  y = (a * a) / (d * k * m)

theorem find_days (h : workers_efficiency_condition a d e k) : target_days_condition a d k m y :=
  sorry

end find_days_l495_495344


namespace projection_vector_correct_l495_495837

/-- Definition of the vectors given in the conditions -/
def v1 : ℝ × ℝ := (-6, 2)
def v2 : ℝ × ℝ := (3, 4)

/-- Definition of the resulting projection vector to be proven -/
def p : ℝ × ℝ := (-12/17, 54/17)

/-- Given v1 and v2, we aim to prove that the resulting projection vector p is correct -/
theorem projection_vector_correct : 
  ∃ t : ℝ, p = (9 * t - 6, 2 * t + 2) ∧
             (9 * t - 6) * 9 + (2 * t + 2) * 2 = 0 :=
begin
  use 10 / 17,
  split,
  { simp },
  { sorry }
end

end projection_vector_correct_l495_495837


namespace trajectory_length_l495_495223

-- Definitions of the points involved
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Definition of the square ABCD
def square_ABCD : Prop :=
  let A := Point.mk 0 0 in
  let B := Point.mk 12 0 in
  let C := Point.mk 12 12 in
  let D := Point.mk 0 12 in
  true

-- Definition of point E on the extension of BC
def point_E : Prop :=
  let B := Point.mk 12 0 in
  let E := Point.mk 28 12 in
  (E.x = B.x + 16) ∧ (E.y = 12)

-- Definition of the trajectory of the center H as point P moves from C to E
def trajectory_H : Prop :=
  let P_C := Point.mk 12 12 in
  let P_E := Point.mk 28 12 in
  let H_C := Point.mk (12 / 2) 6 in
  let H_E := Point.mk (28 / 2) 6 in
  true

-- Proven length of trajectory of center H
theorem trajectory_length :
  square_ABCD → point_E → trajectory_H → (H : Point) → (H.x ∈ set.Icc (H_C.x) (H_E.x)) →
  ∀ P, (P = Point.mk 12 12 ∨ P = Point.mk 28 12) →
  dist (Point.mk 6 6) (Point.mk 14 6) = 8 :=
by
  intro h1 h2 h3 H H_in P P_cond
  -- Use the given conditions and definitions
  sorry

end trajectory_length_l495_495223


namespace meryll_questions_l495_495417

/--
Meryll wants to write a total of 35 multiple-choice questions and 15 problem-solving questions. 
She has written \(\frac{2}{5}\) of the multiple-choice questions and \(\frac{1}{3}\) of the problem-solving questions.
We need to prove that she needs to write 31 more questions in total.
-/
theorem meryll_questions : (35 - (2 / 5) * 35) + (15 - (1 / 3) * 15) = 31 := by
  sorry

end meryll_questions_l495_495417


namespace greatest_integer_radius_l495_495683

theorem greatest_integer_radius (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 := 
by {
  have h1 : r^2 < 75, from (div_lt_div_right (by linarith [pi_pos])).mp h,
  have sqrt_75_lt_9 : sqrt 75 < 9 := by norm_num,
  have sqrt_75_lt_9' : r < 9 := lt_of_pow_lt_pow_of_lt_two h1 sqrt_75_lt_9,
  exact le_of_lt (int.le_iff_coe.mp sqrt_75_lt_9')
}

end greatest_integer_radius_l495_495683


namespace find_price_max_profit_l495_495522

/-
Part 1: Prove the price per unit of type A and B
-/

def price_per_unit (x y : ℕ) : Prop :=
  (2 * x + 3 * y = 690) ∧ (x + 4 * y = 720)

theorem find_price :
  ∃ x y : ℕ, price_per_unit x y ∧ x = 120 ∧ y = 150 :=
by
  sorry

/-
Part 2: Prove the maximum profit with constraints
-/

def conditions (m : ℕ) : Prop :=
  m ≤ 3 * (40 - m) ∧ 120 * m + 150 * (40 - m) ≤ 5400

def profit (m : ℕ) : ℕ :=
  (160 - 120) * m + (200 - 150) * (40 - m)

theorem max_profit :
  ∃ m : ℕ, 20 ≤ m ∧ m ≤ 30 ∧ conditions m ∧ profit m = profit 20 :=
by
  sorry

end find_price_max_profit_l495_495522


namespace quadratic_has_only_positive_roots_l495_495351

theorem quadratic_has_only_positive_roots (m : ℝ) :
  (∀ (x : ℝ), x^2 + (m + 2) * x + (m + 5) = 0 → x > 0) →
  -5 < m ∧ m ≤ -4 :=
by 
  -- added sorry to skip the proof.
  sorry

end quadratic_has_only_positive_roots_l495_495351


namespace difference_between_max_and_min_l495_495976

noncomputable def maxThree (a b c : ℝ) : ℝ :=
  max a (max b c)

noncomputable def minThree (a b c : ℝ) : ℝ :=
  min a (min b c)

theorem difference_between_max_and_min :
  maxThree 0.12 0.23 0.22 - minThree 0.12 0.23 0.22 = 0.11 :=
by
  sorry

end difference_between_max_and_min_l495_495976


namespace angle_BMD_is_120_l495_495006

open EuclideanGeometry

-- Define the rhombus ABCD with angle A = 60 degrees
variables {A B C D E F M : Point}
variables (h_rhombus : rhombus A B C D)
variables (h_angle_A : ∡ A = 60)
variables (h_on_AD : E ≠ D ∧ collinear {A, D, E}) 
variables (h_CE_AB : intersects (line_through C E) (line_through A B) F)
variables (h_DF_BE : intersects (line_through D F) (line_through B E) M)

-- The goal is to show that the angle BMD is 120 degrees regardless of the position of E on AD
theorem angle_BMD_is_120 (h_rhombus : rhombus A B C D) (h_angle_A : ∡ A = 60)
  (h_on_AD : E ≠ D ∧ collinear {A, D, E}) 
  (h_CE_AB : intersects (line_through C E) (line_through A B) F)
  (h_DF_BE : intersects (line_through D F) (line_through B E) M) :
  ∡ (B, M, D) = 120 :=
sorry

end angle_BMD_is_120_l495_495006


namespace largest_of_numbers_l495_495559

theorem largest_of_numbers (a b c d : ℝ) 
  (ha : a = 0) (hb : b = -1) (hc : c = 3.5) (hd : d = Real.sqrt 13) : 
  ∃ x, x = Real.sqrt 13 ∧ (x > a) ∧ (x > b) ∧ (x > c) ∧ (x > d) :=
by
  sorry

end largest_of_numbers_l495_495559


namespace avg_marks_in_6_subjects_l495_495938

-- Define the given conditions
def avg_marks_5_subjects : ℝ := 74
def marks_6th_subject : ℝ := 80
def number_of_subjects : ℝ := 6

-- Define the total marks calculation
def total_marks_5_subjects : ℝ := avg_marks_5_subjects * 5
def total_marks_6_subjects : ℝ := total_marks_5_subjects + marks_6th_subject

-- Define the average calculation
def avg_marks_6_subjects : ℝ := total_marks_6_subjects / number_of_subjects

-- The theorem to prove
theorem avg_marks_in_6_subjects : avg_marks_6_subjects = 75 :=
by
  -- proof
  sorry

end avg_marks_in_6_subjects_l495_495938


namespace tower_arrangements_l495_495152

def totalCubes : ℕ := 16
def blueCubes : ℕ := 9
def redCubes : ℕ := 3
def greenCubes : ℕ := 4
def towerHeight : ℕ := 14

theorem tower_arrangements :
  (nat.find_greatest (λ n, n.factorial / ((2.factorial) * (2.factorial) * (4.factorial)) +
                            n.factorial / (3.factorial * 7.factorial * 4.factorial) +
                            n.factorial / ((2.factorial) * 9.factorial * 3.factorial)) 14) = 15093 :=
sorry

end tower_arrangements_l495_495152


namespace max_radius_of_circle_l495_495694

theorem max_radius_of_circle (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
by
  sorry

end max_radius_of_circle_l495_495694


namespace sum_geom_series_rational_l495_495271

def is_rational (x: ℕ) : Prop := ∃ a b : ℤ, x * b = a

theorem sum_geom_series_rational {a b : ℚ} {n : ℕ} (h : a^2 + b^2 = 1) (h_odd : n % 2 = 1) :
  is_rational (|∑ j in finset.range n, complex.of_real a + complex.I * complex.of_real b|) :=
sorry

end sum_geom_series_rational_l495_495271


namespace boyd_total_friends_l495_495730

-- Conditions
def julian_total_friends : ℕ := 80
def julian_boys_friends : ℕ := 48
def julian_girls_friends : ℕ := 32
def boyd_girls_friends : ℕ := 64

-- Theorem: Boyd's total friends
theorem boyd_total_friends : ∃ F : ℕ, 36/100 * F = boyd_girls_friends and (64/100 * F = boyd_girls_friends) → F = 100 :=
by
  sorry

end boyd_total_friends_l495_495730


namespace elle_june_6_water_l495_495411

-- Define the initial conditions and the term definition for the arithmetic sequence
def elle_water_consumption (d : ℝ) (n : ℕ) : ℝ := d + 8 * (n - 1)

-- Total water consumption from June 1 to June 6
def total_water : ℝ :=
  (∑ i in Finset.range 6, elle_water_consumption d (i + 1))

-- Given condition: total water consumption over 6 days is 140 liters
def water_condition : Prop := total_water = 140

-- The goal is to prove the amount of water drank on June 6
theorem elle_june_6_water (d : ℝ) (water_condition : total_water = 140) : 
    elle_water_consumption d 6 = 43.33 :=
sorry  -- Proof not provided

end elle_june_6_water_l495_495411


namespace nuts_per_student_l495_495905

theorem nuts_per_student (bags nuts_per_bag students : ℕ) (h_bags : bags = 65) (h_nuts_per_bag : nuts_per_bag = 15) (h_students : students = 13) : 
  (bags * nuts_per_bag) / students = 75 := 
by
  rw [h_bags, h_nuts_per_bag, h_students]
  norm_num
  sorry

end nuts_per_student_l495_495905


namespace number_in_max_sum_set_l495_495480

-- Define the conditions:
-- Digits from 0 to 9 are used exactly once in total.
def unique_digit_set : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Check if a number uses a digit exactly once
def uses_digits_exactly_once (n : ℕ) (used : Finset ℕ) := 
  (Finset.range 10).val = used.val ∧ (string.to_list (n.repr)).nodup

-- Form three four-digit numbers:
-- Extract digits for validation
def extract_digits (n : ℕ) : List ℕ := 
  string.to_list (n.repr).map (λ c, c.to_nat - '0'.to_nat)

-- Ensure unique digits from 0 to 9 are used among three numbers
def forms_highest_sum (n1 n2 n3 : ℕ) : Prop :=
  let digits_used := (extract_digits n1 ++ extract_digits n2 ++ extract_digits n3).to_finset in
  used_digits.val = unique_digit_set.val ∧ 
  n1 + n2 + n3 = (extract_digits n1).sum + (extract_digits n2).sum + (extract_digits n3).sum

-- Proven statement
theorem number_in_max_sum_set :
  ∃ (n1 n2 n3 : ℕ), 
    n1 ∈ unique_digit_set ∧
    n2 ∈ unique_digit_set ∧
    n3 ∈ unique_digit_set ∧
    forms_highest_sum n1 n2 n3 ∧
    (9654 = n1 ∨ 9654 = n2 ∨ 9654 = n3) :=
sorry

end number_in_max_sum_set_l495_495480


namespace symmetric_line_probability_l495_495233

theorem symmetric_line_probability :
  let grid := set.univ.to_finset.image (λ (i : ℕ × ℕ), (i.1, i.2)) in
  let P : ℕ × ℕ := (4, 4) in
  let points := grid.erase P in
  let symmetric_points := points.filter (λ Q, 
    Q.1 = P.1 ∨ Q.2 = P.2 ∨ Q.1 + Q.2 = P.1 + P.2 ∨ Q.1 - Q.2 = P.1 - P.2) in
  (symmetric_points.card / points.card : ℚ) = 1 / 2 :=
by 
  sorry

end symmetric_line_probability_l495_495233


namespace smallest_positive_integer_congruence_l495_495122

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ, x = 26 ∧ 5 * x % 31 = 17 :=
by
  use 26
  split
  exact rfl
  norm_num
  sorry

end smallest_positive_integer_congruence_l495_495122


namespace make_one_appears_l495_495748

theorem make_one_appears (m n : ℕ) (hmn_coprime : Nat.coprime m n)
  (initial_numbers : set ℚ := {⟨m, n⟩, ⟨n, m⟩}) :
  (∃ k : ℕ, m + n = 2 ^ k) ↔
  (∃ steps : list (ℚ × ℚ), ∀ step ∈ steps, 
    step.fst ∈ initial_numbers ∧ step.snd ∈ initial_numbers ∧ 
    (step.snd = (step.fst + step.snd) / 2 ∨ step.snd = 2 * step.fst * step.snd / (step.fst + step.snd)) ∧ 
    (1 ∈ initial_numbers ∨ ∃ x ∈ initial_numbers, x = 1)) :=
by
  sorry

end make_one_appears_l495_495748


namespace train_time_36kmph_200m_l495_495506

/-- How many seconds will a train 200 meters long running at the rate of 36 kmph take to pass a certain telegraph post? -/
def time_to_pass_post (length_of_train : ℕ) (speed_kmph : ℕ) : ℕ :=
  length_of_train * 3600 / (speed_kmph * 1000)

theorem train_time_36kmph_200m : time_to_pass_post 200 36 = 20 := by
  sorry

end train_time_36kmph_200m_l495_495506


namespace original_workers_l495_495156

theorem original_workers (x y : ℝ) (h : x = (65 / 100) * y) : y = (20 / 13) * x :=
by sorry

end original_workers_l495_495156


namespace symmetry_of_transformed_function_l495_495834

theorem symmetry_of_transformed_function :
  ∀ (x : ℝ), 
  let f := λ x, 3 * sin (3 * x - π / 4)
  let g := λ x, 3 * sin (3 * x + π / 2) - 4,
  center_of_symmetry g = (0, -2) := by
  sorry

end symmetry_of_transformed_function_l495_495834


namespace problem_statement_l495_495312

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else Real.log x

theorem problem_statement : f (f 10) = 2 := by
  sorry

end problem_statement_l495_495312


namespace no_solution_for_k_eq_4_l495_495493

theorem no_solution_for_k_eq_4 (x k : ℝ) (h₁ : x ≠ 4) (h₂ : x ≠ 8) : (k = 4) → ¬ ((x - 3) * (x - 8) = (x - k) * (x - 4)) :=
by
  sorry

end no_solution_for_k_eq_4_l495_495493


namespace solution_set_for_f_gt_2x_plus_4_l495_495307

noncomputable section

open Real

variables {f : ℝ → ℝ}

theorem solution_set_for_f_gt_2x_plus_4 
  (h_dom : ∀ x : ℝ, x ∈ ℝ)
  (h_f_neg1 : f (-1) = 2)
  (h_f_deriv : ∀ x : ℝ, deriv f x > 2) :
  {x : ℝ | f x > 2 * x + 4} = {x : ℝ | x > -1} :=
sorry

end solution_set_for_f_gt_2x_plus_4_l495_495307


namespace rico_more_dogs_than_justin_l495_495569

theorem rico_more_dogs_than_justin 
  (justin_dogs : ℕ := 14) 
  (camden_legs : ℕ := 72) 
  (camden_ratio : ℚ := 3/4) :
  let camden_dogs := camden_legs / 4 in
  let rico_dogs := camden_dogs * (4/3) in
  rico_dogs - justin_dogs = 10 := 
by
  sorry

end rico_more_dogs_than_justin_l495_495569


namespace pq_plus_four_mul_l495_495397

theorem pq_plus_four_mul (h : ∀ x, (x - 6) * (3 * x + 8) = x^2 - 15 * x + 54) :
  let p := 6
  let q := 9
  (p + 4) * (q + 4) = 130 :=
by
  have p := 6
  have q := 9
  rw [h, p, q]
  sorry

end pq_plus_four_mul_l495_495397


namespace max_islands_correct_l495_495560

noncomputable def max_islands (N : ℕ) : Prop :=
  ∀ N ≥ 7, (∀ i j, i ≠ j → connected(i, j) ≤ 1) ∧ 
  (∀ i, bridges(i) ≤ 5) ∧ 
  (∀ subset : ℕ → Prop, (∀ x, subset x → x < 7) → ∃ i j, subset i ∧ subset j ∧ connected(i, j)) → 
  N ≤ 36

theorem max_islands_correct : max_islands 36 := sorry

end max_islands_correct_l495_495560


namespace expression_value_l495_495874

theorem expression_value :
  ( (120^2 - 13^2) / (90^2 - 19^2) * ((90 - 19) * (90 + 19)) / ((120 - 13) * (120 + 13)) ) = 1 := by
  sorry

end expression_value_l495_495874


namespace half_tuples_xn_eq_n_l495_495403

noncomputable def count_tuples (d n : ℕ) :=
  { x : Fin n.succ → ℕ // (∀ i : Fin n.succ, x i ≤ n) ∧ Monotone (λ i => x i) ∧ d ∣ Finset.univ.sum (λ i => x i) }

theorem half_tuples_xn_eq_n (d n : ℕ) (h1 : 0 < n) (h2 : 0 < d) (h3 : d ∣ n) :
  let M := count_tuples d n;
  let N := { x ∈ M | x n = n };
  2 * N.to_finset.card = M.to_finset.card :=
by
  sorry

end half_tuples_xn_eq_n_l495_495403


namespace ted_cookies_eaten_l495_495276

def cookies_per_tray : ℕ := 12
def trays_per_day : ℕ := 2
def days_baking : ℕ := 6
def cookies_per_day : ℕ := trays_per_day * cookies_per_tray
def total_cookies_baked : ℕ := days_baking * cookies_per_day
def cookies_eaten_by_frank : ℕ := days_baking
def cookies_before_ted : ℕ := total_cookies_baked - cookies_eaten_by_frank
def cookies_left_after_ted : ℕ := 134

theorem ted_cookies_eaten : cookies_before_ted - cookies_left_after_ted = 4 := by
  sorry

end ted_cookies_eaten_l495_495276


namespace condition_sufficiency_l495_495620

theorem condition_sufficiency (x : ℝ) :
  (2 ≤ x ∧ x ≤ 3) → (x < -3 ∨ x ≥ 1) ∧ (∃ x : ℝ, (x < -3 ∨ x ≥ 1) ∧ ¬(2 ≤ x ∧ x ≤ 3)) :=
by
  sorry

end condition_sufficiency_l495_495620


namespace sin_cos_eq_sqrt2_l495_495260

theorem sin_cos_eq_sqrt2 (x : ℝ) (h0 : 0 ≤ x) (h1 : x ≤ 2 * Real.pi) (h2 : Real.sin x - Real.cos x = Real.sqrt 2) :
  x = (3 * Real.pi) / 4 :=
sorry

end sin_cos_eq_sqrt2_l495_495260


namespace four_digit_numbers_count_l495_495326

theorem four_digit_numbers_count :
  (∃ n : Fin 9999 → Nat, 
     (∀ i : Fin 9998, n i + n (i + 1) ≤ 2) ∧ 
     (∀ i : Fin 9997, n i + n (i + 1) + n (i + 2) ≥ 3)
  ) → 
  (card { n : Fin 9999 → Nat | 
     (∀ i : Fin 9998, n i + n (i + 1) ≤ 2) ∧ 
     (∀ i : Fin 9997, n i + n (i + 1) + n (i + 2) ≥ 3)} = 1) := 
by
  sorry

end four_digit_numbers_count_l495_495326


namespace fibonacci_series_sum_equals_l495_495009

def fibonacci (n : ℕ) : ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

noncomputable def fibonacci_series_sum : ℚ := ∑' n, (fibonacci n : ℚ) / (7^n)

theorem fibonacci_series_sum_equals : fibonacci_series_sum = (49 : ℚ) / 287 := by
  sorry

end fibonacci_series_sum_equals_l495_495009


namespace water_usage_difference_l495_495572

variable (a b : ℝ)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (ha_plus_4 : a + 4 ≠ 0)

theorem water_usage_difference :
  b / a - b / (a + 4) = 4 * b / (a * (a + 4)) :=
by
  sorry

end water_usage_difference_l495_495572


namespace find_angle_C_l495_495626

-- Given conditions
variable {A B C : ℝ}
variable (h_triangle : A + B + C = π)
variable (h_tanA : Real.tan A = 1/2)
variable (h_cosB : Real.cos B = 3 * Real.sqrt 10 / 10)

-- The proof statement
theorem find_angle_C :
  C = 3 * π / 4 := by
  sorry

end find_angle_C_l495_495626


namespace students_preferring_windows_is_correct_l495_495904

-- Define the total number of students surveyed
def total_students : ℕ := 210

-- Define the number of students preferring Mac
def students_preferring_mac : ℕ := 60

-- Define the number of students preferring both Mac and Windows equally
def students_preferring_both : ℕ := students_preferring_mac / 3

-- Define the number of students with no preference
def students_no_preference : ℕ := 90

-- Calculate the total number of students with a preference
def students_with_preference : ℕ := total_students - students_no_preference

-- Calculate the number of students preferring Windows
def students_preferring_windows : ℕ := students_with_preference - (students_preferring_mac + students_preferring_both)

-- State the theorem to prove that the number of students preferring Windows is 40
theorem students_preferring_windows_is_correct : students_preferring_windows = 40 :=
by
  -- calculations based on definitions
  unfold students_preferring_windows students_with_preference students_preferring_mac students_preferring_both students_no_preference total_students
  sorry

end students_preferring_windows_is_correct_l495_495904


namespace calculate_sum_theta_120_l495_495944

noncomputable def sum_of_cis_sequence_is_cis_120 : Prop :=
  ∃ (r : ℝ), r > 0 ∧ (80.to_real.rad.cos + 90.to_real.rad.cos + 100.to_real.rad.cos + 110.to_real.rad.cos + 
                     120.to_real.rad.cos + 130.to_real.rad.cos + 140.to_real.rad.cos + 
                     150.to_real.rad.cos + 160.to_real.rad.cos) = r

theorem calculate_sum_theta_120 : sum_of_cis_sequence_is_cis_120 :=
sorry

end calculate_sum_theta_120_l495_495944


namespace odd_function_negative_one_value_l495_495742

def f (x : ℝ) : ℝ := 
if x >= 0 then 2^x + 2*x + (-1) else -(2^(-x) + 2*(-x) + (-1))

theorem odd_function_negative_one_value : 
  (∀ x : ℝ, f (-x) = -f x) → (f (0) = 0) → f (-1) = -3 := 
by
  intros h1 h2
  -- We will skip the proof with sorry
  sorry

end odd_function_negative_one_value_l495_495742


namespace nine_sided_convex_polygon_diagonals_l495_495162

theorem nine_sided_convex_polygon_diagonals :
  ∃ (D : ℕ), D = 27 ∧ ∀ (n : ℕ), n = 9 → D = (n * (n - 3)) / 2 :=
by
  existsi (27 : ℕ)
  split
  -- the answer
  assume D
  exact rfl
  -- the condition n = 9
  assume (n : ℕ)
  assume h : n = 9
  rw h
  exact rfl
  sorry

end nine_sided_convex_polygon_diagonals_l495_495162


namespace forty_percent_of_N_is_360_l495_495138

-- Definitions based on the conditions
def N : ℝ := some (exists (λ N, (1/4) * (1/3) * (2/5) * N = 30))

-- Main statement to be proven
theorem forty_percent_of_N_is_360 : (0.40 * N = 360) := 
sorry

end forty_percent_of_N_is_360_l495_495138


namespace exists_zero_in_interval_l495_495814

def f (x : ℝ) : ℝ := Real.exp x - x - 2

theorem exists_zero_in_interval : ∃ x ∈ Ioo 1 2, f x = 0 := by
  sorry

end exists_zero_in_interval_l495_495814


namespace comparison_l495_495610

noncomputable def a : ℝ := 0.2 ^ 2
noncomputable def b : ℝ := 2 ^ 0.3
noncomputable def c : ℝ := Real.log 2 / Real.log 0.2

theorem comparison : b > a ∧ a > c := 
by
  sorry

end comparison_l495_495610


namespace statue_of_liberty_model_height_l495_495933

theorem statue_of_liberty_model_height :
  let scale_ratio : Int := 30
  let actual_height : Int := 305
  round (actual_height / scale_ratio) = 10 := by
  sorry

end statue_of_liberty_model_height_l495_495933


namespace range_of_a_l495_495617

theorem range_of_a {a : ℝ} : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ - 3 / 5 < a ∧ a ≤ 1 := sorry

end range_of_a_l495_495617


namespace intersecting_lines_unique_plane_l495_495143

theorem intersecting_lines_unique_plane (P : Prop) :
  (∀ (l1 l2 : Line), intersect l1 l2 → ∃! (π : Plane), in_plane l1 π ∧ in_plane l2 π) ↔ P :=
by
  sorry

end intersecting_lines_unique_plane_l495_495143


namespace problem_statement_l495_495672

theorem problem_statement
  (m : ℝ) 
  (h : m + (1/m) = 5) :
  m^2 + (1 / m^2) + 4 = 27 :=
by
  -- Parameter types are chosen based on the context and problem description.
  sorry

end problem_statement_l495_495672


namespace total_selling_price_l495_495547

theorem total_selling_price 
  (n : ℕ) (p : ℕ) (c : ℕ) 
  (h_n : n = 85) (h_p : p = 15) (h_c : c = 85) : 
  (c + p) * n = 8500 :=
by
  sorry

end total_selling_price_l495_495547


namespace geometric_series_sum_l495_495857

theorem geometric_series_sum
  (a r : ℚ) (n : ℕ)
  (ha : a = 1/4)
  (hr : r = 1/4)
  (hn : n = 5) :
  (∑ i in finset.range n, a * r^i) = 341 / 1024 := by
  sorry

end geometric_series_sum_l495_495857


namespace solution_set_of_inequality_l495_495455

theorem solution_set_of_inequality (f : ℝ → ℝ) (h_decreasing : ∀ x y, x < y → f x > f y)
  (h1 : f 0 = 3) (h2 : f 3 = -1) :
  {x : ℝ | abs (f (x+1) - 1) < 2} = set.Ioo (-1 : ℝ) 2 :=
  -- the main proof problem statement
sorry

end solution_set_of_inequality_l495_495455


namespace shoppers_share_amount_l495_495382

noncomputable def calculate_share : ℝ :=
let giselle := 120 in
let isabella := giselle + 15 in
let sam := isabella - 45 in
let valentina := 2 * sam in
let ethan := isabella - 75 in
let isabella_donation := 0.2 * isabella in
let sam_donation := 0.15 * sam in
let giselle_donation := 0.1 * giselle in
let valentina_donation := 0.25 * valentina in
let ethan_donation := 0.3 * ethan in
(let total_donated := isabella_donation + sam_donation + giselle_donation + valentina_donation + ethan_donation in
  total_donated / 4).round(2)

theorem shoppers_share_amount :
  calculate_share = 28.88 :=
by
  -- Here would normally be the detailed proof steps which is omitted.
  sorry

end shoppers_share_amount_l495_495382


namespace sum_of_roots_eq_l495_495812

theorem sum_of_roots_eq (b1 b2 b3 : ℝ) : 
  let x1₁ := (-b1 + real.sqrt 1) / 2,
      x1₂ := (-b1 - real.sqrt 1) / 2,
      x2₁ := (-b2 + real.sqrt 4) / 2,
      x2₂ := (-b2 - real.sqrt 4) / 2,
      x3₁ := (-b3 + real.sqrt 9) / 2,
      x3₂ := (-b3 - real.sqrt 9) / 2 in
  x1₁ + x2₁ + x3₂ = x1₂ + x2₂ + x3₁ :=
by {
  sorry
}

end sum_of_roots_eq_l495_495812


namespace min_cubes_needed_l495_495917

-- Definition of a cube with snaps and receptacle holes
structure Cube :=
  (has_one_snap : Bool)   -- True if the cube has one snap
  (has_two_snaps : Bool)  -- True if the cube has two snaps

-- Condition: We have cubes with 1 snap and 2 snaps 
def cubes_with_snaps : List Cube := 
  [⟨true, false⟩, ⟨false, true⟩] -- Example: list containing such cubes to represent conditions

-- The minimum number of cubes needed to satisfy the condition is 6
theorem min_cubes_needed : ∃ cubes : List Cube, 
  cubes.length = 6 ∧ 
  (∀ cube ∈ cubes, (cube.has_one_snap ∨ cube.has_two_snaps)) ∧
  (∀ cube ∈ cubes, cover_all_snaps cube) -- This function needs to be defined to show that all snaps are covered.
:= by
  sorry

end min_cubes_needed_l495_495917


namespace inequality_solution_l495_495451

theorem inequality_solution :
  {x : ℝ | ((x > 4) ∧ (x < 5)) ∨ ((x > 6) ∧ (x < 7)) ∨ (x > 7)} =
  {x : ℝ | (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0} :=
sorry

end inequality_solution_l495_495451


namespace probability_at_least_one_4_in_7_rolls_probability_exactly_one_4_in_7_rolls_probability_at_most_one_4_in_7_rolls_l495_495119

-- Define the probability of rolling a die 7 times and the probabilities related to rolling a 4.

def prob_not_4_single_roll : ℝ := 5 / 6

def prob_not_4_in_7_rolls : ℝ := prob_not_4_single_roll ^ 7

def prob_at_least_one_4_in_7_rolls : ℝ := 1 - prob_not_4_in_7_rolls

def prob_exactly_one_4_in_7_rolls : ℝ := 7 * (1 / 6) * (prob_not_4_single_roll ^ 6)

def prob_at_most_one_4_in_7_rolls : ℝ := prob_not_4_in_7_rolls + prob_exactly_one_4_in_7_rolls

theorem probability_at_least_one_4_in_7_rolls :
  prob_at_least_one_4_in_7_rolls = 1 - (5 / 6) ^ 7 :=
  by
  sorry

theorem probability_exactly_one_4_in_7_rolls :
  prob_exactly_one_4_in_7_rolls = 7 * (1 / 6) * (5 / 6) ^ 6 :=
  by
  sorry

theorem probability_at_most_one_4_in_7_rolls :
  prob_at_most_one_4_in_7_rolls = (5 / 6) ^ 7 + 7 * (1 / 6) * (5 / 6) ^ 6 :=
  by
  sorry

end probability_at_least_one_4_in_7_rolls_probability_exactly_one_4_in_7_rolls_probability_at_most_one_4_in_7_rolls_l495_495119


namespace length_of_median_l495_495293

-- Definitions of the points A, B, C, D in Cartesian coordinates
def A : (ℝ × ℝ × ℝ) := (1, 0, 0)
def B : (ℝ × ℝ × ℝ) := (0, 2, 0)
def C : (ℝ × ℝ × ℝ) := (0, 0, 3)
def D : (ℝ × ℝ × ℝ) := (0, 0, 0)

-- Proof that the length of the median vector from D to the centroid of ABC is sqrt(14) / 3
theorem length_of_median :
  let M := ((1 / 3, 2 / 3, 1 / 3) : ℝ × ℝ × ℝ) in
  ∥ (M.1, M.2, M.3) - (D.1, D.2, D.3) ∥ = real.sqrt(14) / 3 :=
by {
  -- Proof omitted
  sorry
}

end length_of_median_l495_495293


namespace probability_sum_is_five_when_die_thrown_twice_l495_495552

-- Conditions
def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}
def is_uniformly_dense_die (die : Finset ℕ) : Prop :=
  ∀ x ∈ die, 1 ≤ x ∧ x ≤ 6

-- Question rephrased into a Lean 4 theorem statement
theorem probability_sum_is_five_when_die_thrown_twice :
  let outcomes := { (x, y) | x ∈ die_faces ∧ y ∈ die_faces };
  let favorable_outcomes := { (x, y) | x ∈ die_faces ∧ y ∈ die_faces ∧ x + y = 5 } in
  ∃ (h : is_uniformly_dense_die die_faces),
    (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ) = 1 / 9 :=
by
  sorry

end probability_sum_is_five_when_die_thrown_twice_l495_495552


namespace range_of_f_l495_495579

def f (x : ℝ) : ℝ := 2 * sin x * cos x ^ 2 / (1 + sin x)

theorem range_of_f :
  ∀ y, y ∈ set.range f ↔ y ∈ set.Ioo (-4 : ℝ) (1 / 2 : ℝ) ∨ y = (1 / 2 : ℝ) :=
by {
  -- The main proof body would go here
  sorry
}

end range_of_f_l495_495579


namespace complex_fraction_product_l495_495017

theorem complex_fraction_product (a b : ℝ) (h : (1:ℂ) + 7 * complex.i / (2 - complex.i) = a + b * complex.i) : a * b = -3 := 
sorry

end complex_fraction_product_l495_495017


namespace count_integers_between_cubes_l495_495335

theorem count_integers_between_cubes (a b : ℝ) (h1 : a = 10.5) (h2 : b = 10.6) : 
  let lower_bound := a^3
  let upper_bound := b^3
  let first_integer := Int.ceil lower_bound
  let last_integer := Int.floor upper_bound
  (last_integer - first_integer + 1) = 33 :=
by
  -- Definitions for clarity
  let lower_bound := a^3
  let upper_bound := b^3
  let first_integer := Int.ceil lower_bound
  let last_integer := Int.floor upper_bound
  
  -- Skipping the proof
  sorry

end count_integers_between_cubes_l495_495335


namespace num_integers_between_l495_495332

-- Define the constants
def a : ℝ := 10
def b₁ : ℝ := 0.5
def b₂ : ℝ := 0.6

-- Define the cubes
def x : ℝ := (a + b₁)^3
def y : ℝ := (a + b₂)^3

-- Define the function to count the integers within the interval
def count_integers_between (x y : ℝ) : ℕ :=
  let min_int := Int.ceil x
  let max_int := Int.floor y
  (max_int - min_int + 1).toNat

-- The statement to prove
theorem num_integers_between : count_integers_between x y = 33 := by
  sorry

end num_integers_between_l495_495332


namespace value_of_x_minus_y_l495_495355

theorem value_of_x_minus_y (x y : ℝ) (h1 : x = -(-3)) (h2 : |y| = 5) (h3 : x * y < 0) : x - y = 8 := 
sorry

end value_of_x_minus_y_l495_495355


namespace pairs_of_polygons_with_angle_ratio_l495_495114

theorem pairs_of_polygons_with_angle_ratio :
  ∃ n, n = 2 ∧ (∀ {k r : ℕ}, (k > 2 ∧ r > 2) → 
  (4 * (180 * r - 360) = 3 * (180 * k - 360) →
  ((k = 3 ∧ r = 18) ∨ (k = 2 ∧ r = 6)))) :=
by
  -- The proof should be provided here, but we skip it
  sorry

end pairs_of_polygons_with_angle_ratio_l495_495114


namespace brad_reads_26_pages_per_day_l495_495325

-- Define conditions
def greg_daily_reading : ℕ := 18
def brad_extra_pages : ℕ := 8

-- Define Brad's daily reading
def brad_daily_reading : ℕ := greg_daily_reading + brad_extra_pages

-- The theorem to be proven
theorem brad_reads_26_pages_per_day : brad_daily_reading = 26 := by
  sorry

end brad_reads_26_pages_per_day_l495_495325


namespace QX_value_l495_495946

variables {O A P B Q X : Type}
variables (d_AB : ℝ) (PQ BQ PB QX : ℝ)
variables (l_tangent : Prop) (l_condition : Prop)

def Circle (center : Type) (radius : ℝ) : Prop := sorry
def Diameter {c : Type} (d : Type) : Prop := sorry
def Chord {c : Type} (ch : Type) : Prop := sorry
def Perpendicular {a b c : Type} : Prop := sorry
def Angle (α β γ : Type) : ℝ := sorry

noncomputable
def find_QX : ℝ :=
  if h : Circle O (d_AB / 2) ∧
          Chord (Circle O (d_AB / 2)) (\overline{CD}) ∧
          Perpendicular \overline{AB} \overline{CD} ∧
          \overline{CD} ∈ Circle A \∩ Circle O ∧
          Q ∈ minor_arc \overline{CD} (Circle A) ∧
          Angle A Q P + Angle Q P B = 60 ∧
          Line tangent (Circle A) at Q ∧
          Point X on Line l_tangent such that (dist P X = dist B X) ∧
          PQ = 13 ∧
          BQ = 35 ∧
          PB = 43 then \frac{43 \sqrt{3}}{3} else 0

theorem QX_value : find_QX d_AB PQ BQ PB QX l_tangent l_condition = \frac{43 \sqrt{3}}{3} :=
sorry

end QX_value_l495_495946


namespace probability_one_absent_one_present_l495_495362

-- Definitions based on given conditions
def prob_absent := 1 / 15
def prob_present := 14 / 15

-- The proof statement for the probability problem
theorem probability_one_absent_one_present :
  ((prob_present * prob_absent) + (prob_absent * prob_present)) * 100 = 12.4 := 
sorry

end probability_one_absent_one_present_l495_495362


namespace jo_integer_max_l495_495000
noncomputable def jo_integer : Nat :=
  let n := 166
  n

theorem jo_integer_max (n : Nat) (h1 : n < 200) (h2 : ∃ k : Nat, n + 2 = 9 * k) (h3 : ∃ l : Nat, n + 4 = 10 * l) : n ≤ jo_integer := 
by
  unfold jo_integer
  sorry

end jo_integer_max_l495_495000


namespace equilateral_triangle_area_sum_l495_495462

theorem equilateral_triangle_area_sum (r : ℝ) (a b : ℕ) (h : r = 40)
    (h1 : ∀ (O1 O2 O3 : ℝ), dist O1 O2 = 2 * r ∧ dist O2 O3 = 2 * r ∧ dist O3 O1 = 2 * r)
    (h2 : ∀ (P1 P2 P3 : ℝ), P1P2 = 160 ∧ P2P3 = 160 ∧ P3P1 = 160)
    (area : ℝ := (6400 * real.sqrt 3)) :
    area = real.sqrt a + real.sqrt b →
    a + b = 5520000 :=
by
  intros
  sorry

end equilateral_triangle_area_sum_l495_495462


namespace surface_area_of_sphere_l495_495092

noncomputable def volume : ℝ := 72 * Real.pi

theorem surface_area_of_sphere (r : ℝ) (h : (4 / 3) * Real.pi * r^3 = volume) :
  4 * Real.pi * r^2 = 36 * Real.pi * (Real.cbrt 2)^2 :=
by
  sorry

end surface_area_of_sphere_l495_495092


namespace minimum_chord_length_l495_495702

theorem minimum_chord_length (a : ℝ) : 
  let C := λ x y : ℝ, (x - real.arcsin a) * (x - real.arccos a) + (y - real.arcsin a) * (y + real.arccos a),
      line_eq := λ x : ℝ, x = real.pi / 4,
      chord_length := λ d : ℝ, ∀ a : ℝ, C(real.pi / 4, y) = 0 -> d = 2 * sqrt((real.pi / 4 - y)^2)
  in (∃ d : ℝ, chord_length d ∧ (∀ d' : ℝ, chord_length d' → d' ≥ real.pi / 2))
:= sorry

end minimum_chord_length_l495_495702


namespace probability_log10_integer_four_digit_l495_495167

noncomputable def is_four_digit (N : ℕ) : Prop :=
  1000 ≤ N ∧ N ≤ 9999

noncomputable def log10_is_integer (N : ℕ) : Prop :=
  ∃ k : ℕ, N = 10^k

theorem probability_log10_integer_four_digit :
  (∑ x in Icc 1000 9999, if log10_is_integer x then 1 else 0 : ℝ) / (∑ x in Icc 1000 9999, 1 : ℝ) = 1/9000 :=
by
  sorry

end probability_log10_integer_four_digit_l495_495167


namespace arc_length_of_regular_pentagon_l495_495365

-- Definitions based on conditions from part a)
def radius : ℝ := 5
def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
def length_of_arc (r : ℝ) : ℝ := circumference r / 5

-- Statement to prove
theorem arc_length_of_regular_pentagon : length_of_arc radius = 2 * Real.pi := by
  sorry

end arc_length_of_regular_pentagon_l495_495365


namespace output_for_input_8_is_8_over_65_l495_495885

def function_f (n : ℕ) : ℚ := n / (n^2 + 1)

theorem output_for_input_8_is_8_over_65 : function_f 8 = 8 / 65 := by
  sorry

end output_for_input_8_is_8_over_65_l495_495885


namespace unit_vector_perpendicular_l495_495278

theorem unit_vector_perpendicular (x y : ℝ)
  (h1 : 4 * x + 2 * y = 0) 
  (h2 : x^2 + y^2 = 1) :
  (x = (Real.sqrt 5) / 5 ∧ y = -(2 * (Real.sqrt 5) / 5)) ∨ 
  (x = -(Real.sqrt 5) / 5 ∧ y = 2 * (Real.sqrt 5) / 5) :=
sorry

end unit_vector_perpendicular_l495_495278


namespace find_a_b_and_monotonic_intervals_find_range_for_c_l495_495316

noncomputable def f (x a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c
noncomputable def f' (x a b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

theorem find_a_b_and_monotonic_intervals {c : ℝ} :
  (∀ x, f'(x, -3/2, -6) = 0 → (x = -1 ∨ x = 2)) ∧
  (∀ x, (f'(x, -3/2, -6) < 0 ↔ -1 < x ∧ x < 2) ∧ 
       (f'(x, -3/2, -6) > 0 ↔ (x < -1 ∨ x > 2))) :=
by { sorry }

theorem find_range_for_c : 
  ∀ c, (∀ x ∈ Set.Icc (-2 : ℝ) 3, f(x, -3/2, -6, c) + 3/2 * c < c^2) ↔ 
  (c < -1 ∨ 7/2 < c) :=
by { sorry }

end find_a_b_and_monotonic_intervals_find_range_for_c_l495_495316


namespace office_average_salary_l495_495372

theorem office_average_salary :
  let avg_salary_officers := 420 in
  let avg_salary_non_officers := 110 in
  let num_officers := 15 in
  let num_non_officers := 450 in
  let total_salary_officers := avg_salary_officers * num_officers in
  let total_salary_non_officers := avg_salary_non_officers * num_non_officers in
  let total_salary := total_salary_officers + total_salary_non_officers in
  let total_employees := num_officers + num_non_officers in
  let avg_salary_all_employees := total_salary / total_employees in
  avg_salary_all_employees = 120 :=
by sorry

end office_average_salary_l495_495372


namespace rectangle_area_l495_495201

theorem rectangle_area (w l : ℝ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end rectangle_area_l495_495201


namespace find_c_l495_495601

theorem find_c (x : ℝ) : ∃ (p q : ℝ), (p^2 = 6) ∧ (q^2 = 16) ∧ (6 * x^2 + 8 * real.sqrt 6 * x + 16 = (p * x + q)^2) :=
sorry

end find_c_l495_495601


namespace negation_is_correct_l495_495075

-- Define the original proposition as a predicate on real numbers.
def original_prop : Prop := ∀ x : ℝ, 4*x^2 - 3*x + 2 < 0

-- State the negation of the original proposition
def negation_of_original_prop : Prop := ∃ x : ℝ, 4*x^2 - 3*x + 2 ≥ 0

-- The theorem to prove the correctness of the negation of the original proposition
theorem negation_is_correct : ¬original_prop ↔ negation_of_original_prop := by
  sorry

end negation_is_correct_l495_495075


namespace tickets_per_ride_factor_l495_495109

theorem tickets_per_ride_factor (initial_tickets spent_tickets remaining_tickets : ℕ) 
  (h1 : initial_tickets = 40) 
  (h2 : spent_tickets = 28) 
  (h3 : remaining_tickets = initial_tickets - spent_tickets) : 
  ∃ k : ℕ, remaining_tickets = 12 ∧ (∀ m : ℕ, m ∣ remaining_tickets → m = k) → (k ∣ 12) :=
by
  sorry

end tickets_per_ride_factor_l495_495109


namespace peasant_woman_eggs_l495_495918

theorem peasant_woman_eggs (x : ℕ) :
  (∀ x, (let remaining := x - (x / 3 + 15)) in
   x = (x / 3 + 15) + (7 * remaining / 9 + 10) → x = 90) :=
by 
  sorry

end peasant_woman_eggs_l495_495918


namespace sum_of_reciprocals_l495_495824

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 16) (h2 : x * y = 48) : (1 / x + 1 / y) = (1 / 3) :=
by
  sorry

end sum_of_reciprocals_l495_495824


namespace recurring_decimal_of_division_l495_495252

theorem recurring_decimal_of_division (a b : ℤ) (h1 : a = 60) (h2 : b = 55) : (a : ℝ) / (b : ℝ) = 1.09090909090909090909090909090909 :=
by
  -- Import the necessary definitions and facts
  sorry

end recurring_decimal_of_division_l495_495252


namespace concentration_of_alcohol_after_mixing_l495_495511

/-- Given the proportions of water to alcohol in two solutions and 
the condition that they are mixed in equal amounts, prove the 
concentration of alcohol in the resulting solution. -/
theorem concentration_of_alcohol_after_mixing (waterA alcoholA waterB alcoholB : ℕ)
  (hA : waterA = 4) (hA' : alcoholA = 1) (hB : waterB = 2) (hB' : alcoholB = 3) :
  let totalWater := waterA + waterB,
      totalAlcohol := alcoholA + alcoholB,
      totalMixture := totalWater + totalAlcohol
  in (totalAlcohol.toRat / totalMixture.toRat) = 0.4 :=
by sorry

end concentration_of_alcohol_after_mixing_l495_495511


namespace imaginary_part_conjugate_l495_495998

theorem imaginary_part_conjugate (z : ℂ) (h : z = (3 - complex.i) / (3 + complex.i)) :
  complex.im (conj z) = 3 / 5 :=
sorry

end imaginary_part_conjugate_l495_495998


namespace find_coefficients_l495_495319

noncomputable def g (x : ℝ) : ℝ := x^2 + 74 * x + 1369

theorem find_coefficients :
  (∃ x₀ : ℝ, ∀ x : ℝ, g(x) = (x + 37)^2) ∧
  (∃ x₁ : ℝ, ∀ x : ℝ, g(x₁^5 + 2*x₁ - 1) + g(x₁^5 + 3*x₁ + 1) = 0) :=
sorry

end find_coefficients_l495_495319


namespace increasing_function_greater_at_a_squared_plus_one_l495_495016

variable (f : ℝ → ℝ) (a : ℝ)

def strictly_increasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x < f y

theorem increasing_function_greater_at_a_squared_plus_one :
  strictly_increasing f → f (a^2 + 1) > f a :=
by
  sorry

end increasing_function_greater_at_a_squared_plus_one_l495_495016


namespace households_320_degrees_or_higher_l495_495909

open ProbabilityMeas

variables (μ : ℚ) (σ : ℚ) (n : ℕ)

def normal_consumption_distribution : Prop :=
  (μ = 300) ∧ (σ = 10) ∧ (n = 1000)

def reference_data :=
  (P(μ - σ < ξ < μ + σ) = 0.6826) ∧
  (P(μ - 2 * σ < ξ < μ + 2 * σ) = 0.9544) ∧
  (P(μ - 3 * σ < ξ < μ + 3 * σ) = 0.9974)

noncomputable def higher_consumption_probability : ℚ :=
  (1 - 0.954) / 2

noncomputable def households_with_higher_consumption : ℕ :=
  n * higher_consumption_probability

theorem households_320_degrees_or_higher (h : normal_consumption_distribution μ σ n) (r : reference_data) :
  households_with_higher_consumption 23 := sorry

end households_320_degrees_or_higher_l495_495909


namespace distance_z10_from_origin_l495_495238

def z : ℕ → ℂ
| 0       := 0
| (n + 1) := z n^2 + (1 + complex.I)

theorem distance_z10_from_origin :
  (complex.abs (z 10)) = real.sqrt 45205 :=
sorry

end distance_z10_from_origin_l495_495238


namespace a_sufficient_not_necessary_l495_495283

-- Definitions
def a (a : ℝ) : Prop := a > 1
def b (a : ℝ) : Prop := a^2 > 1

-- Lean statement
theorem a_sufficient_not_necessary : ∀ (a : ℝ), a (a) → b (a) ∧ ¬(∀ (a : ℝ), b (a) → a (a)) :=
by
  intros a ha
  unfold a b at *
  constructor
  -- Prove that a > 1 implies a^2 > 1
  sorry
  -- Prove that a > 1 is not necessary for a^2 > 1
  sorry

end a_sufficient_not_necessary_l495_495283


namespace median_possible_values_l495_495738

theorem median_possible_values (S : Set ℤ) 
  (h1 : S.card = 11) 
  (h2 : {1, 5, 7, 11, 13, 18, 21} ⊆ S) : 
  ∃ medians : Set ℤ, medians.card = 5 ∧ ∀ m ∈ medians, is_median m S :=
sorry

end median_possible_values_l495_495738


namespace sphere_surface_area_of_given_volume_l495_495090

-- Definition of the problem conditions
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * π * r^3

def surface_area_of_sphere (r : ℝ) : ℝ := 4 * π * r^2

-- Statement of the theorem to be proved
theorem sphere_surface_area_of_given_volume :
  (∃ (r : ℝ), volume_of_sphere r = 72 * π ∧ surface_area_of_sphere r = 36 * π * 2 ^ (2 / 3)) :=
sorry

end sphere_surface_area_of_given_volume_l495_495090


namespace real_part_0_or_3_complex_part_not_0_or_3_purely_imaginary_at_2_no_second_quadrant_l495_495985

def z (m : ℝ) : ℂ := (m^2 - 5 * m + 6 : ℝ) + (m^2 - 3 * m : ℝ) * Complex.I

theorem real_part_0_or_3 (m : ℝ) : (m^2 - 3 * m = 0) ↔ (m = 0 ∨ m = 3) := sorry

theorem complex_part_not_0_or_3 (m : ℝ) : (m^2 - 3 * m ≠ 0) ↔ (m ≠ 0 ∧ m ≠ 3) := sorry

theorem purely_imaginary_at_2 (m : ℝ) : (m^2 - 5 * m + 6 = 0) ∧ (m^2 - 3 * m ≠ 0) ↔ (m = 2) := sorry

theorem no_second_quadrant (m : ℝ) : ¬(m^2 - 5 * m + 6 < 0 ∧ m^2 - 3 * m > 0) := sorry

end real_part_0_or_3_complex_part_not_0_or_3_purely_imaginary_at_2_no_second_quadrant_l495_495985


namespace pentagon_division_l495_495576

noncomputable def vertices := (A B C D E : Point)

noncomputable def area (p : Pentagon) : ℝ :=
  sorry

def parallel_line (p : Point) (l : Line) : Line :=
  sorry

theorem pentagon_division (A B C D E : Point) (T : ℝ) :
  ∃ l1 l2 l3 : Line, 
  parallel_line A B = l1 ∧
  parallel_line A B = l2 ∧
  parallel_line A B = l3 ∧
  (area (to_pentagon A B C D E) / 4) = (area (polygon_section A l1)) = (area (polygon_section A l2)) = (area (polygon_section A l3)) := sorry

end pentagon_division_l495_495576


namespace find_AE_l495_495141

theorem find_AE 
  (ABC : Triangle)
  (AB AC : ℝ)
  (b c : ℝ) (h_bc : b > c)
  (D : Point) (h1 : D ∈ ABC.BC)
  (E : Point) (h2 : E ∈ ABC.AC)
  (hAD : AngleBisector ABC.A ABC.D)
  (hDE_perp_AD : Perpendicular D E AD) :
  AE = (2 * b * c) / (b + c) :=
sorry

end find_AE_l495_495141


namespace eq_solution_set_l495_495588

theorem eq_solution_set (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a^b = b^(a^a)) :
  (a, b) = (1, 1) ∨ (a, b) = (2, 16) ∨ (a, b) = (3, 27) :=
by
  sorry

end eq_solution_set_l495_495588


namespace general_term_of_sequence_l495_495628

theorem general_term_of_sequence 
  (a : ℕ → ℝ)
  (log_a : ℕ → ℝ)
  (h1 : ∀ n, log_a n = Real.log (a n)) 
  (h2 : ∃ d, ∀ n, log_a (n + 1) - log_a n = d)
  (h3 : d = Real.log 3)
  (h4 : log_a 0 + log_a 1 + log_a 2 = 6 * Real.log 3) : 
  ∀ n, a n = 3 ^ n :=
by
  sorry

end general_term_of_sequence_l495_495628


namespace thor_fraction_correct_l495_495762

-- Define the initial conditions
def moes_money : ℕ := 12
def lokis_money : ℕ := 10
def nicks_money : ℕ := 8
def otts_money : ℕ := 6

def thor_received_from_each : ℕ := 2

-- Calculate total money each time
def total_initial_money : ℕ := moes_money + lokis_money + nicks_money + otts_money
def thor_total_received : ℕ := 4 * thor_received_from_each
def thor_fraction_of_total : ℚ := thor_total_received / total_initial_money

-- The theorem to prove
theorem thor_fraction_correct : thor_fraction_of_total = 2/9 :=
by
  sorry

end thor_fraction_correct_l495_495762


namespace question_inequality_l495_495358

theorem question_inequality (m : ℝ) :
  (∀ x : ℝ, ¬ (m * x ^ 2 - m * x - 1 ≥ 0)) ↔ (-4 < m ∧ m ≤ 0) :=
sorry

end question_inequality_l495_495358


namespace polar_eq_is_parabola_l495_495241

-- Definitions as directly appearing in the conditions
def polar_eq (r θ : ℝ) : Prop := r = 6 * (Real.tan θ) * (Real.sec θ)

-- Assertion incorporating the conditions and the expected answer
theorem polar_eq_is_parabola (r θ : ℝ) (h : polar_eq r θ) : ∃a b : ℝ, (a ≠ 0) ∧ (r^2 * (Real.cos θ)^2) = 6 * r * (Real.sin θ) :=
by
  sorry

end polar_eq_is_parabola_l495_495241


namespace sum_palindromic_primes_lt_100_eq_429_l495_495782

def is_palindromic_prime (n : ℕ) : Prop :=
  (n ≥ 10 ∧ n < 100) ∧
  (Prime n ∧ 
   let d1 := n / 10 in
   let d2 := n % 10 in
   Prime (d2 * 10 + d1))

theorem sum_palindromic_primes_lt_100_eq_429 : 
  (∑ n in Finset.filter is_palindromic_prime (Finset.range 100), n) = 429 :=
by sorry

end sum_palindromic_primes_lt_100_eq_429_l495_495782


namespace number_of_subsets_of_intersection_l495_495350

theorem number_of_subsets_of_intersection (A : Set ℕ) (B : Set ℕ)
  (hA : A = {0, 1, 2, 3}) (hB : B = {1, 2, 4}) : 
  (Finset.powerset (Finset.inter (Finset.of (Set.toFinset A)) (Finset.of (Set.toFinset B)))).card = 4 := 
by
  sorry

end number_of_subsets_of_intersection_l495_495350


namespace sum_mod_six_l495_495843

theorem sum_mod_six (n : ℤ) : ((10 - 2 * n) + (4 * n + 2)) % 6 = 0 :=
by {
  sorry
}

end sum_mod_six_l495_495843


namespace bert_money_left_l495_495940

theorem bert_money_left
  (initial_amount : ℝ)
  (spent_hardware_store_fraction : ℝ)
  (amount_spent_dry_cleaners : ℝ)
  (spent_grocery_store_fraction : ℝ)
  (final_amount : ℝ) :
  initial_amount = 44 →
  spent_hardware_store_fraction = 1/4 →
  amount_spent_dry_cleaners = 9 →
  spent_grocery_store_fraction = 1/2 →
  final_amount = initial_amount - (spent_hardware_store_fraction * initial_amount) - amount_spent_dry_cleaners - (spent_grocery_store_fraction * (initial_amount - (spent_hardware_store_fraction * initial_amount) - amount_spent_dry_cleaners)) →
  final_amount = 12 :=
by
  sorry

end bert_money_left_l495_495940


namespace sum_of_palindromic_primes_l495_495788

def is_palindromic_prime (n : ℕ) : Prop :=
  n < 100 ∧ 10 ≤ n ∧ n.Prime ∧ let reversed := (n % 10) * 10 + (n / 10) in 
  reversed.Prime

theorem sum_of_palindromic_primes : 
  ∑ p in List.filter is_palindromic_prime (List.range 100), p = 429 :=
by
  sorry

end sum_of_palindromic_primes_l495_495788


namespace greatest_radius_l495_495688

theorem greatest_radius (r : ℕ) : (π * (r : ℝ)^2 < 75 * π) ↔ r ≤ 8 := 
by
  sorry

end greatest_radius_l495_495688


namespace L_shaped_tile_impossible_l495_495965

-- Define the "L-shaped" tile
structure LShapedTile where
  pos : Fin 4 → (ℕ × ℕ)
  valid_shape : (pos (0)) = (0,0) ∧ (pos (1)) = (0,1) ∧ (pos (2)) = (1,0) ∧ (pos (3)) = (1,1)

def valid_tiling (m n : ℕ) (tiles : list LShapedTile) : Prop :=
  ∀ t1 t2 t3 : LShapedTile,
    t1 ∈ tiles → t2 ∈ tiles → t3 ∈ tiles →
    ¬(t1.pos (0) ∧ (t2.pos (1)) ∧ (t3.pos (2)) = (3,2)) ∧ -- Condition 1
    ∀ pt : ℕ × ℕ, (pt ∉ {t1.pos 0, t1.pos 1, t1.pos 2} ∧ ∃ t4 : LShapedTile, t4 ∈ tiles ∧ t4.pos = pt) → 
                  (pt ∉ {t2.pos 0, t2.pos 1, t2.pos 2}) → 
                  (pt ∉ {t3.pos 0, t3.pos 1, t3.pos 2})   -- Condition 2

theorem L_shaped_tile_impossible :
  ¬∃ (m n : ℕ) (tiles : list LShapedTile), valid_tiling m n tiles :=
by
  sorry

end L_shaped_tile_impossible_l495_495965


namespace ferris_wheel_seat_capacity_l495_495902

-- Define the given conditions
def people := 16
def seats := 4

-- Define the problem and the proof goal
theorem ferris_wheel_seat_capacity : people / seats = 4 := by
  sorry

end ferris_wheel_seat_capacity_l495_495902


namespace mass_of_sodium_acetate_formed_l495_495243

-- Define the reaction conditions and stoichiometry
def initial_moles_acetic_acid : ℝ := 3
def initial_moles_sodium_hydroxide : ℝ := 4
def initial_reaction_moles_acetic_acid_with_sodium_carbonate : ℝ := 2
def initial_reaction_moles_sodium_carbonate : ℝ := 1
def product_moles_sodium_acetate_from_step1 : ℝ := 2
def remaining_moles_acetic_acid : ℝ := initial_moles_acetic_acid - initial_reaction_moles_acetic_acid_with_sodium_carbonate
def product_moles_sodium_acetate_from_step2 : ℝ := remaining_moles_acetic_acid
def total_moles_sodium_acetate : ℝ := product_moles_sodium_acetate_from_step1 + product_moles_sodium_acetate_from_step2
def molar_mass_sodium_acetate : ℝ := 82.04

-- Translate to the equivalent proof problem
theorem mass_of_sodium_acetate_formed :
  total_moles_sodium_acetate * molar_mass_sodium_acetate = 246.12 :=
by
  -- The detailed proof steps would go here
  sorry

end mass_of_sodium_acetate_formed_l495_495243


namespace X2_minus_2_irreducible_l495_495958

def polynomial_irrational_irrationality (p : Polynomial ℚ) : Prop :=
  ¬ ∃ (a b : Polynomial ℝ), a * b = p ∧ a.degree + b.degree = p.degree ∧
    a ≠ 1 ∧ b ≠ 1

def polynomial_factorization_real (p : Polynomial ℝ) : Prop :=
  ∃ (a b : Polynomial ℝ), a * b = p ∧ a.degree + b.degree = p.degree ∧
    a ≠ 1 ∧ b ≠ 1

theorem X2_minus_2_irreducible 
  (f : Polynomial ℚ)
  (hf : f = Polynomial.cyclotomic 2 0 + Polynomial.C (-2)) :
  polynomial_irrational_irrationality f ∧ polynomial_factorization_real f :=
sorry

end X2_minus_2_irreducible_l495_495958


namespace hana_entire_collection_value_l495_495658

theorem hana_entire_collection_value (a b : ℝ) (h : 4/7 * a = 28) : a = 49 :=
by
  have h1 : a = 28 / (4/7),
  { calc
      a = 28 / (4 / 7) : by rw [mul_div_cancel' (4/7 * a) (4/7) (rational.ne_zero_of_ne_zero rational.zero_ne_one (rational.ne_zero_of_ne_zero (7) (ne_of_gt (by norm_num))))], 
      calc 28 / (4 / 7) = 28 * (7 / 4),
      sorry, -- This simplifies the division to multiplication by the reciprocal.
      calc 28 / (4 / 7) = (28 * 7) / 4,
      sorry, -- Simplifies further to final result
      calc (28 * 7) / 4 = 196 / 4,
      sorry, -- Simplified
      calc 196 / 4 = 49,
      sorry, -- Final simplification
  }
  exact h1

end hana_entire_collection_value_l495_495658


namespace total_food_l495_495100

def animals := 6
def food_per_animal_per_month := 4

theorem total_food (n : ℕ) (f_per_animal : ℕ) (total_f : ℕ) (h1 : n = animals) (h2 : f_per_animal = food_per_animal_per_month) : total_f = n * f_per_animal := 
  by sorry

example : total_food 6 4 24 _ _ :=
  by { rw [h1, h2], exact rfl } -- This test case shows how the theorem can be instantiated.

end total_food_l495_495100


namespace shortest_is_Bob_l495_495367

variable (Amy Bob Carla Dan Eric : ℕ)

-- Conditions translated to Lean.
def condition1 : Prop := Amy > Carla
def condition2 : Prop := Eric > Dan ∧ Dan > Bob
def condition3 : Prop := Carla > Eric

-- Proving who is the shortest.
theorem shortest_is_Bob : condition1 Amy Carla ∧ condition2 Eric Dan Bob ∧ condition3 Carla Eric → Bob < Amy ∧ Bob < Carla ∧ Bob < Dan ∧ Bob < Eric := by
  intros h
  cases h with h1 h
  cases h with h2 h3
  sorry

end shortest_is_Bob_l495_495367


namespace rectangular_field_area_l495_495196

theorem rectangular_field_area (w : ℕ) (h : ℕ) (P : ℕ) (A : ℕ) (h_length : h = 3 * w) (h_perimeter : 2 * (w + h) = 72) : A = w * h := 
by
  -- Given conditions
  have h_eq : h = 3 * w := h_length
  rewrite [h_eq] at *
  -- Given perimeter equals 8w = 72
  have w_value : w = 9 := sorry
  -- Length calculated as 3w = 27
  have h_value : h = 27 := sorry
  -- Now we calculate the area
  calc
    A = w * h : by sorry
    ... = 9 * 27 : by sorry
    ... = 243 : by sorry

end rectangular_field_area_l495_495196


namespace greatest_integer_radius_l495_495684

theorem greatest_integer_radius (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 := 
by {
  have h1 : r^2 < 75, from (div_lt_div_right (by linarith [pi_pos])).mp h,
  have sqrt_75_lt_9 : sqrt 75 < 9 := by norm_num,
  have sqrt_75_lt_9' : r < 9 := lt_of_pow_lt_pow_of_lt_two h1 sqrt_75_lt_9,
  exact le_of_lt (int.le_iff_coe.mp sqrt_75_lt_9')
}

end greatest_integer_radius_l495_495684


namespace reaction_completion_l495_495597

-- Definitions from conditions
def NaOH_moles : ℕ := 2
def H2O_moles : ℕ := 2

-- Given the balanced equation
-- 2 NaOH + H2SO4 → Na2SO4 + 2 H2O

theorem reaction_completion (H2SO4_moles : ℕ) :
  (2 * (NaOH_moles / 2)) = H2O_moles → H2SO4_moles = 1 :=
by 
  -- Skip proof
  sorry

end reaction_completion_l495_495597


namespace rectangular_field_area_l495_495175

theorem rectangular_field_area :
  ∃ (w l : ℝ), w = l / 3 ∧ 2 * (w + l) = 72 ∧ w * l = 243 :=
by
  sorry

end rectangular_field_area_l495_495175


namespace find_a_l495_495291

noncomputable def line_eq (a x y : ℝ) : Prop := a * x + y - 2 = 0
noncomputable def circle_eq (x y a : ℝ) : Prop := (x - 1)^2 + (y - a)^2 = 4
noncomputable def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop := 
  (dist A B = dist B C) ∧ (dist B C = dist C A)

theorem find_a (a : ℝ) :
  (∃ x y : ℝ, line_eq a x y ∧ circle_eq x y a) ∧ 
  ∀ A B C : ℝ × ℝ, is_equilateral_triangle A B C
  → a = 4 + real.sqrt 15 ∨ a = 4 - real.sqrt 15 :=
sorry

end find_a_l495_495291


namespace first_and_second_bags_l495_495103

def bags_apples (A B C : ℕ) : Prop :=
  (A + B + C = 24) ∧ (B + C = 18) ∧ (A + C = 19)

theorem first_and_second_bags (A B C : ℕ) (h : bags_apples A B C) :
  A + B = 11 :=
sorry

end first_and_second_bags_l495_495103


namespace abs_sum_factors_l495_495892

theorem abs_sum_factors (a b c d : ℤ) : 
  (6 * x ^ 2 + x - 12 = (a * x + b) * (c * x + d)) →
  (|a| + |b| + |c| + |d| = 12) :=
by
  intros h
  sorry

end abs_sum_factors_l495_495892


namespace B_fills_tank_alone_in_40_minutes_l495_495529

theorem B_fills_tank_alone_in_40_minutes :
  (B fills the tank in 40 minutes) :=
  let rate_A := 1 / 60
  let T_B := 40
  let rate_B := 1 / T_B
  let half_time := 15
  let full_time := 2 * half_time
  let fill_fraction_B := half_time * rate_B
  let fill_fraction_A_and_B := half_time * (rate_A + rate_B)
  2 * half_time * (rate_A + rate_B) = 1 :=
begin
  sorry
end

end B_fills_tank_alone_in_40_minutes_l495_495529


namespace rectangular_field_area_l495_495197

theorem rectangular_field_area (w : ℕ) (h : ℕ) (P : ℕ) (A : ℕ) (h_length : h = 3 * w) (h_perimeter : 2 * (w + h) = 72) : A = w * h := 
by
  -- Given conditions
  have h_eq : h = 3 * w := h_length
  rewrite [h_eq] at *
  -- Given perimeter equals 8w = 72
  have w_value : w = 9 := sorry
  -- Length calculated as 3w = 27
  have h_value : h = 27 := sorry
  -- Now we calculate the area
  calc
    A = w * h : by sorry
    ... = 9 * 27 : by sorry
    ... = 243 : by sorry

end rectangular_field_area_l495_495197


namespace sin_double_angle_neg_l495_495609

variable (α : Real)
variable (h1 : Real.tan α < 0)
variable (h2 : Real.sin α = -Real.sqrt 3 / 3)

theorem sin_double_angle_neg (h1 : Real.tan α < 0) (h2 : Real.sin α = -Real.sqrt 3 / 3) : 
  Real.sin (2 * α) = -2 * Real.sqrt 2 / 3 := 
by 
  sorry

end sin_double_angle_neg_l495_495609


namespace collatz_eighth_term_is_one_collatz_possible_initial_values_count_l495_495807

def collatz_step : ℕ → ℕ
| 0       := 0 -- usually we define collatz only for positive integers
| (n + 1) := if (n + 1) % 2 = 0 then (n + 1) / 2 else (3 * (n + 1) + 1)

def collatz_nth_term (n : ℕ) : ℕ → ℕ 
| 0       := n
| (k + 1) := collatz_step (collatz_nth_term k)

theorem collatz_eighth_term_is_one (n : ℕ) : ∃ n, collatz_nth_term n 8 = 1 :=
sorry

theorem collatz_possible_initial_values_count : (set.univ.filter (λ n, collatz_nth_term n 8 = 1)).to_finset.card = 6 :=
sorry

end collatz_eighth_term_is_one_collatz_possible_initial_values_count_l495_495807


namespace proposition_A_proposition_B_proposition_C_proposition_D_l495_495498

-- Definitions based on conditions
def is_trapezoid (ABCD : Type) [quadrilateral ABCD] : Prop := True

def reciprocal (x y : ℚ) : Prop := x = 1 / y ∧ y = 1 / x

def is_rational (a : ℚ) : Prop := True

def is_rhombus (ABCD : Type) [quadrilateral ABCD] : Prop := True

-- Theorems to prove the propositions
theorem proposition_A (ABCD : Type) [quadrilateral ABCD] (ht : is_trapezoid ABCD) :
  (∑ angles in ABCD.interiorAngles, angles = 360) :=
by sorry

theorem proposition_B (x y : ℚ) (hr : reciprocal x y) : 
  x * y = 1 :=
by sorry

theorem proposition_C (a : ℚ) (hr : is_rational a) : 
  a^2 + 1 ≥ 1 :=
by sorry

theorem proposition_D (ABCD : Type) [quadrilateral ABCD] (hr : is_rhombus ABCD) :
  ¬(centrally_symmetric ABCD ∧ ¬axisymmetric ABCD) :=
by sorry

end proposition_A_proposition_B_proposition_C_proposition_D_l495_495498


namespace calc_fraction_l495_495568
-- Import necessary libraries

-- Define the necessary fractions and the given expression
def expr := (5 / 6) * (1 / (7 / 8 - 3 / 4))

-- State the theorem
theorem calc_fraction : expr = 20 / 3 := 
by
  sorry

end calc_fraction_l495_495568


namespace max_almond_weight_l495_495149

def mix_ratio := (5, 2, 3, 4)  -- (almonds, walnuts, cashews, pistachios)
def total_weight : ℕ := 800
def budget : ℕ := 1500
def cost_per_pound := (6, 5, 8, 10)  -- (almonds, walnuts, cashews, pistachios)
def min_almond_fraction : ℚ := 0.3

theorem max_almond_weight (w_almonds : ℕ) (w_walnuts : ℕ) 
  (w_cashews : ℕ) (w_pistachios : ℕ) : 
  w_almonds.toRat / total_weight.toRat ≥ min_almond_fraction ∧ 
  w_almonds * cost_per_pound.1 + 
  w_walnuts * cost_per_pound.2 + 
  w_cashews * cost_per_pound.3 + 
  w_pistachios * cost_per_pound.4 ≤ budget ∧ 
  w_almonds + w_walnuts + w_cashews + w_pistachios = total_weight → 
  w_almonds = 240 :=
by sorry

end max_almond_weight_l495_495149


namespace pattern_perimeter_l495_495222

theorem pattern_perimeter : 
  let side_length : ℝ := 2 -- side length in cm
  let num_squares := 6
  let num_triangles := 6
  let num_hexagons := 1 in
  let perimeter_square := 4 * side_length -- per square
  let perimeter_triangle := 3 * side_length -- per triangle
  let perimeter_hexagon := 6 * side_length -- per hexagon
  let total_squares_length := (perimeter_square - 2 * side_length)^num_squares -- removing shared sides
  let total_triangles_length := (perimeter_triangle - 2 * side_length)^num_triangles -- removing shared sides
  let total_hexagons_length := perimeter_hexagon -- hexagon forms outer part completely
    
  total_squares_length + total_triangles_length + total_hexagons_length = 24 :=
sorry

end pattern_perimeter_l495_495222


namespace price_of_stock_l495_495592

def investment : ℝ := 6800
def annual_income : ℝ := 3000
def dividend_rate : ℝ := 0.6
def face_value : ℝ := 100

theorem price_of_stock :
  let dividend_yield := (annual_income / investment) * 100 in
  let dividend_per_share := face_value * dividend_rate in
  let price_of_stock := dividend_per_share / (dividend_yield / 100) in
  price_of_stock = 136 := by
  sorry

end price_of_stock_l495_495592


namespace sum_geometric_series_is_correct_l495_495867

def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_geometric_series_is_correct
  (a r : ℚ) (n : ℕ)
  (h_a : a = 1/4)
  (h_r : r = 1/4)
  (h_n : n = 5) :
  geometric_series_sum a r n = 341 / 1024 :=
by
  rw [h_a, h_r, h_n]
  -- Now we can skip the proof.
  sorry

end sum_geometric_series_is_correct_l495_495867


namespace average_change_in_elevation_per_hour_l495_495069

theorem average_change_in_elevation_per_hour :
  ∀ (elevation_ontario elevation_erie : ℝ) (travel_time : ℝ),
  elevation_ontario = 75.00 →
  elevation_erie = 174.28 →
  travel_time = 8 →
  (elevation_erie - elevation_ontario) / travel_time = 12.41 :=
by
  intros elevation_ontario elevation_erie travel_time
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end average_change_in_elevation_per_hour_l495_495069


namespace greatest_int_radius_lt_75pi_l495_495698

noncomputable def circle_radius_max (A : ℝ) (π : ℝ) : ℕ :=
  Nat.floor (Real.sqrt (A / π))

theorem greatest_int_radius_lt_75pi :
  circle_radius_max 75 Real.pi = 8 := by
  sorry

end greatest_int_radius_lt_75pi_l495_495698


namespace eval_f_f_neg2_l495_495611

open Real

def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then π
  else 0

theorem eval_f_f_neg2 : f (f (-2)) = π := by
  sorry

end eval_f_f_neg2_l495_495611


namespace parabola_tangent_angle_l495_495269

noncomputable def tangent_slope_angle : Real :=
  let x := (1 / 2 : ℝ)
  let y := x^2
  let slope := (deriv (fun x => x^2)) x
  Real.arctan slope

theorem parabola_tangent_angle :
  tangent_slope_angle = Real.pi / 4 :=
by
sorry

end parabola_tangent_angle_l495_495269


namespace rectangle_area_l495_495205

theorem rectangle_area (w l : ℝ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end rectangle_area_l495_495205


namespace solve_eq_log_base_l495_495441

theorem solve_eq_log_base (x : ℝ) : (9 : ℝ)^(x+8) = (10 : ℝ)^x → x = Real.logb (10 / 9) ((9 : ℝ)^8) := by
  intro h
  sorry

end solve_eq_log_base_l495_495441


namespace angle_between_vectors_60_degrees_l495_495324

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Conditions
def condition1 : Prop := (a + 2 • b) ⬝ (a - b) = -6
def condition2 : Prop := ∥a∥ = 1
def condition3 : Prop := ∥b∥ = 2

-- Theorem statement
theorem angle_between_vectors_60_degrees
  (h1 : condition1 a b)
  (h2 : condition2 a)
  (h3 : condition3 b) : 
  real.angle_of_vectors a b = π / 3 :=
sorry

end angle_between_vectors_60_degrees_l495_495324


namespace probability_sum_is_five_when_die_thrown_twice_l495_495551

-- Conditions
def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}
def is_uniformly_dense_die (die : Finset ℕ) : Prop :=
  ∀ x ∈ die, 1 ≤ x ∧ x ≤ 6

-- Question rephrased into a Lean 4 theorem statement
theorem probability_sum_is_five_when_die_thrown_twice :
  let outcomes := { (x, y) | x ∈ die_faces ∧ y ∈ die_faces };
  let favorable_outcomes := { (x, y) | x ∈ die_faces ∧ y ∈ die_faces ∧ x + y = 5 } in
  ∃ (h : is_uniformly_dense_die die_faces),
    (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ) = 1 / 9 :=
by
  sorry

end probability_sum_is_five_when_die_thrown_twice_l495_495551


namespace m_over_n_lt_sqrt2_l495_495021

theorem m_over_n_lt_sqrt2 (m n : ℕ) (h : m < real.sqrt 2 * n) : 
  (m:ℝ)/(n:ℝ) < real.sqrt 2 * (1 - 1 / (4 * n^2)) := 
by
  sorry

end m_over_n_lt_sqrt2_l495_495021


namespace projection_magnitude_is_eight_l495_495978

-- defining vectors a and b
def a : ℝ × ℝ := (7, -4)
def b : ℝ × ℝ := (-8, 6)

-- function to compute the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- function to compute the Euclidean norm (magnitude) of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

-- function to compute the magnitude of the projection of vector a onto vector b
def projection_magnitude (a b : ℝ × ℝ) : ℝ :=
  Real.abs (dot_product a b / magnitude b)

-- stating the theorem
theorem projection_magnitude_is_eight : projection_magnitude a b = 8 :=
by
  sorry

end projection_magnitude_is_eight_l495_495978


namespace material_needed_for_second_type_l495_495571

namespace CherylProject

def first_material := 5 / 9
def leftover_material := 1 / 3
def total_material_used := 5 / 9

theorem material_needed_for_second_type :
  0.8888888888888889 - (5 / 9 : ℝ) = 0.3333333333333333 := by
  sorry

end CherylProject

end material_needed_for_second_type_l495_495571


namespace arithmetic_sequence_general_formula_range_of_k_l495_495035

theorem arithmetic_sequence_general_formula (a_n S_n : Nat → Int) (S_5 : Int) (h1 : S_5 = 25) 
(h2 : ∀ n, S_n = n * a_n + (n * (n - 1)) / 2 * 3) :
  ∃ (a1 d : Int), a_n = 3 * n - 4 := sorry

theorem range_of_k (a_n S_n : Nat → Int) (k : Real) (h1 : ∀ n, S_n = -n + 3 * (n * (n - 1) / 2)) 
(h2 : ∀ n, 2 * S_n + 8 * n + 27 > (-1)^n * k * (a_n + 4)) : 
  -7 < k ∧ k < 29 / 4 := sorry

end arithmetic_sequence_general_formula_range_of_k_l495_495035


namespace cube_root_of_sum_l495_495873

theorem cube_root_of_sum (x : ℝ) (a : ℝ) (h : x = 5^7) (k : a = 4 * x) :
  (∛a) = 2^(2/3) * 5^(7/3) :=
by
  sorry

end cube_root_of_sum_l495_495873


namespace coprime_divides_product_l495_495055

theorem coprime_divides_product {a b n : ℕ} (h1 : Nat.gcd a b = 1) (h2 : a ∣ n) (h3 : b ∣ n) : ab ∣ n :=
by
  sorry

end coprime_divides_product_l495_495055


namespace sequence_inequality_for_k_l495_495647

theorem sequence_inequality_for_k (k : ℝ) : 
  (∀ n : ℕ, 0 < n → (n + 1)^2 + k * (n + 1) + 2 > n^2 + k * n + 2) ↔ k > -3 :=
sorry

end sequence_inequality_for_k_l495_495647


namespace find_ellipse_equation_max_area_triangle_PAB_l495_495616

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (2*a = 4) ∧ (1/a = Real.sqrt 2) ∧ (a^2 = 4 ∧ b^2 = 2)

noncomputable def is_point_on_ellipse (x y a b : ℝ) : Prop :=
  (y^2 / a^2 + x^2 / b^2 = 1)

noncomputable def intersects_ellipse (a b m : ℝ) : Prop :=
  let Δ := (2 * Real.sqrt 2 * m)^2 - 16 * (m^2 - 4)
  in Δ > 0 ∧ Real.abs m < 2 * Real.sqrt 2

noncomputable def max_triangle_area (a b m x1 x2 : ℝ) : ℝ :=
  let |AB| := Real.sqrt 3 * Real.sqrt ((x1 + x2)^2 - 4 * x1 * x2)
  let d := Real.abs m / Real.sqrt 3
  in 1/2 * |AB| * d

theorem find_ellipse_equation (a b : ℝ) :
  ellipse_equation a b → (y^2 / 4 + x^2 / 2 = 1) :=
sorry

theorem max_area_triangle_PAB (a b m : ℝ) (P : ℝ × ℝ) :
  let x1 := - Real.sqrt 2 / 2 * m
  let x2 := (m^2 - 4) / 4
  (is_point_on_ellipse 1 (Real.sqrt 2) a b) ∧ intersects_ellipse a b m →
  max_triangle_area a b m x1 x2 = Real.sqrt 2 :=
sorry

end find_ellipse_equation_max_area_triangle_PAB_l495_495616


namespace tom_average_score_increase_l495_495476

def initial_scores : List ℕ := [72, 78, 81]
def fourth_exam_score : ℕ := 90

theorem tom_average_score_increase :
  let initial_avg := (initial_scores.sum : ℚ) / (initial_scores.length : ℚ)
  let total_score_after_fourth := initial_scores.sum + fourth_exam_score
  let new_avg := (total_score_after_fourth : ℚ) / (initial_scores.length + 1 : ℚ)
  new_avg - initial_avg = 3.25 := by 
  -- Proof goes here
  sorry

end tom_average_score_increase_l495_495476


namespace sphere_surface_area_of_given_volume_l495_495088

-- Definition of the problem conditions
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * π * r^3

def surface_area_of_sphere (r : ℝ) : ℝ := 4 * π * r^2

-- Statement of the theorem to be proved
theorem sphere_surface_area_of_given_volume :
  (∃ (r : ℝ), volume_of_sphere r = 72 * π ∧ surface_area_of_sphere r = 36 * π * 2 ^ (2 / 3)) :=
sorry

end sphere_surface_area_of_given_volume_l495_495088


namespace modulus_of_Z_l495_495640

-- Define the complex number Z
def Z : ℂ := complex.mk (sqrt 3) (-1)

-- State the theorem that given Z as defined, its modulus is 2
theorem modulus_of_Z : complex.abs Z = 2 :=
by
  sorry

end modulus_of_Z_l495_495640


namespace cos_180_eq_neg_one_l495_495949

theorem cos_180_eq_neg_one :
  (∃ (rotate : ℝ × ℝ → ℝ × ℝ), (rotate (1, 0) = (-1, 0)) ∧ (∀ x y, rotate (x, y) = (-x, -y))) →
  Real.cos (Real.pi) = -1 :=
by
  intros h
  have h1 := h.1
  rw [Real.pi_eq_two_pi_div_two, Real.cos_pi_div_two_add]    -- Using the identity that pi = 2*pi*(1/2) and cos(π + x) = -cos(x)
  rw Real.cos_pi
  exact -1

end cos_180_eq_neg_one_l495_495949


namespace b_100_is_15001_5_l495_495574

def sequence_b : ℕ → ℝ
| 0       => 0  -- We will define b_1 as sequence_b 1, so b_0 is irrelevant.
| 1       => 3
| (n + 1) => sequence_b n + 3 * n

theorem b_100_is_15001_5 : sequence_b 100 = 15001.5 :=
  sorry

end b_100_is_15001_5_l495_495574


namespace ratio_ramesh_xyz_l495_495798

theorem ratio_ramesh_xyz 
(total_profit : ℝ) 
(ratio_xyz_rajeev : ℚ) 
(rajeev_share : ℝ) 
(h1 : total_profit = 36000) 
(h2 : ratio_xyz_rajeev = 8 / 9) 
(h3 : rajeev_share = 12000) 
: ∃ ratio_ramesh_xyz : ℚ, ratio_ramesh_xyz = 5 / 4 :=
by
  -- Definitions of shares based on conditions
  let X : ℝ := (8 / 9 : ℚ) * rajeev_share
  let R : ℝ := total_profit - (X + rajeev_share)

  -- Simple conditions for R and X
  have hX : X = (8 / 9 : ℚ) * rajeev_share := by sorry
  have hR : R = total_profit - (X + rajeev_share) := by sorry

  -- Ratio calculation step
  let ratio_ramesh_xyz : ℚ := (R / X : ℝ).to_rat simpl

  -- Prove the required ratio
  use ratio_ramesh_xyz
  rw [←Rat.to_rat_of_int.div_eq_div_of_int]
  sorry

end ratio_ramesh_xyz_l495_495798


namespace inequality_solution_l495_495453

theorem inequality_solution :
  {x : ℝ | ((x > 4) ∧ (x < 5)) ∨ ((x > 6) ∧ (x < 7)) ∨ (x > 7)} =
  {x : ℝ | (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0} :=
sorry

end inequality_solution_l495_495453


namespace additional_height_last_two_floors_l495_495520

-- Definitions of the problem conditions
def num_floors : ℕ := 20
def height_per_floor : ℕ := 3
def building_total_height : ℤ := 61

-- Condition on the height of first 18 floors
def height_first_18_floors : ℤ := 18 * 3

-- Height of the last two floors
def height_last_two_floors : ℤ := building_total_height - height_first_18_floors
def height_each_last_two_floor : ℤ := height_last_two_floors / 2

-- Height difference between the last two floors and the first 18 floors
def additional_height : ℤ := height_each_last_two_floor - height_per_floor

-- Theorem to prove
theorem additional_height_last_two_floors :
  additional_height = 1 / 2 := 
sorry

end additional_height_last_two_floors_l495_495520


namespace task_a_task_b_l495_495482

section Geometry

-- Define the planes and lines
variables {π σ : Type*} [plane π] [plane σ]
variables {l : Type*} [line l]

-- Task (a) conditions
variable (h₁ : l ⊥ π)
variable (h₂ : σ ⟂ π)
variable (h₃ : σ contains l)

-- Task (b) conditions
variable (h₄ : l ∥ π)
variable (h₅ : σ ∥ l)

-- Conclusion (a): Plane σ is perpendicular to plane π
theorem task_a : σ ⟂ π :=
  by sorry

-- Conclusion (b): Plane σ intersects plane π along a line parallel to l
theorem task_b : ∃ (m : Type*) [line m], m ∥ l ∧ intersect(σ, π) = m :=
  by sorry

end Geometry

end task_a_task_b_l495_495482


namespace cover_punctures_probability_equal_arcs_l495_495145

/- Part a -/
theorem cover_punctures (A B C : ℝ) (hA : 0 ≤ A) (hB : 0 ≤ B) (hC : 0 ≤ C) (hA_lt_3 : A < 3) (hB_lt_3 : B < 3) (hC_lt_3 : C < 3) :
  ∃ (D : set ℝ), (∀ x ∈ {A, B, C}, x ∈ D) ∧ (∀ x ∈ D, 0 ≤ x ∧ x < 3) ∧ measure_theory.measure.le_of_subset (set.Icc 0 3) ((D).indicator 1).measure (measure_theory.volume (set.Ioo 0 (2 : ℝ))) :=
sorry

/- Part b -/
theorem probability_equal_arcs (A B C : ℝ) (hA : 0 ≤ A) (hB : 0 ≤ B) (hC : 0 ≤ C) (hA_lt_3 : A < 3) (hB_lt_3 : B < 3) (hC_lt_3 : C < 3) :
  (∀ x y z : ℝ, (0 ≤ x ∧ x ≤ 1) ∧ (1 ≤ y ∧ y ≤ 2) ∧ (2 ≤ z ∧ z ≤ 3)) → (Α = 0 ∧ Β < Γ)  ≤ (2/3 : ℝ) :=
sorry

end cover_punctures_probability_equal_arcs_l495_495145


namespace probability_of_drawing_diamond_or_ace_l495_495927

-- Define the number of diamonds
def numDiamonds : ℕ := 13

-- Define the number of other Aces
def numOtherAces : ℕ := 3

-- Define the total number of cards in the deck
def totalCards : ℕ := 52

-- Define the number of desirable outcomes (either diamonds or Aces)
def numDesirableOutcomes : ℕ := numDiamonds + numOtherAces

-- Define the probability of drawing a diamond or an Ace
def desiredProbability : ℚ := numDesirableOutcomes / totalCards

theorem probability_of_drawing_diamond_or_ace :
  desiredProbability = 4 / 13 :=
by
  sorry

end probability_of_drawing_diamond_or_ace_l495_495927


namespace result_is_square_of_2998_l495_495943

noncomputable def calculation_result : ℝ :=
  100 * 29.98 * 2.998 * 1000

theorem result_is_square_of_2998 :
  calculation_result = (2998:ℝ) * (2998:ℝ) := by
  have h1 : (29.98 * 100 : ℝ) = 2998 := by
    norm_num
  have h2 : (2.998 * 1000 : ℝ) = 2998 := by
    norm_num
  have h3: 100 * (29.98 * 2.998 * 1000) = 100 * (2998 * 2998) := by
    rw [h1, h2]
  norm_num at h3
  rw [mul_assoc, ← mul_assoc 100 (2998 * 2998) 1, one_mul] at h3
  exact h3

end result_is_square_of_2998_l495_495943


namespace geometric_series_sum_l495_495859

theorem geometric_series_sum
  (a r : ℚ) (n : ℕ)
  (ha : a = 1/4)
  (hr : r = 1/4)
  (hn : n = 5) :
  (∑ i in finset.range n, a * r^i) = 341 / 1024 := by
  sorry

end geometric_series_sum_l495_495859


namespace sum_palindromic_primes_lt_100_eq_429_l495_495783

def is_palindromic_prime (n : ℕ) : Prop :=
  (n ≥ 10 ∧ n < 100) ∧
  (Prime n ∧ 
   let d1 := n / 10 in
   let d2 := n % 10 in
   Prime (d2 * 10 + d1))

theorem sum_palindromic_primes_lt_100_eq_429 : 
  (∑ n in Finset.filter is_palindromic_prime (Finset.range 100), n) = 429 :=
by sorry

end sum_palindromic_primes_lt_100_eq_429_l495_495783


namespace correct_word_for_blank_l495_495336

-- Define the initial condition of the sentence context implying a future day reference
def sentence_refers_to_future_day : Prop := 
  "I felt so bad all day yesterday that I decided this morning I couldn’t face _ day like that."
  noting "tomorrow" or "future day" context.

-- Define grammatical correctness function for words fitting the blank
def is_grammatically_correct_word (word : String) : Prop :=
  match word with
  | "other"   => false
  | "another" => true
  | "the other" => false
  | "others"  => false
  | _         => false

-- The proof problem statement combining the conditions and the solution
theorem correct_word_for_blank : is_grammatically_correct_word "another" :=
by
  sorry

end correct_word_for_blank_l495_495336


namespace inequality_solution_l495_495449

theorem inequality_solution
  (x : ℝ) :
  x ∉ {2, 3, 4, 5, 6, 7} →
  ((x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0 ↔ 
  (x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 6) ∨ (7 < x)) :=
by
  -- Proof goes here
  sorry

end inequality_solution_l495_495449


namespace quadratic_in_interval_l495_495974

noncomputable def quadratic_function (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 - 2 * a * x + 1

theorem quadratic_in_interval : 
  (∀ x ∈ Icc (0 : ℝ) (2 : ℝ), abs((quadratic_function a x)) ≤ 2) → 
  a ∈ Icc (-1 : ℝ) (0 : ℝ) ∨ a ∈ Ioo (0 : ℝ) (3 : ℝ) → 
  (↑(2 : ℝ) - ↑(-1 : ℝ) + (3 : ℝ) - 0 = 4) :=
begin
  sorry
end

end quadratic_in_interval_l495_495974


namespace nuts_distributive_problem_l495_495110

theorem nuts_distributive_problem (x y : ℕ) (h1 : 70 ≤ x + y) (h2 : x + y ≤ 80) (h3 : (3 / 4 : ℚ) * x + (1 / 5 : ℚ) * (y + (1 / 4 : ℚ) * x) = (x : ℚ) + 1) :
  x = 36 ∧ y = 41 :=
by
  sorry

end nuts_distributive_problem_l495_495110


namespace inequality_solution_l495_495258

open Real

theorem inequality_solution :
  {x | x ∈ Icc (-π/4) (7 * π / 4) ∧ (sin x)^2018 + (cos x)^(-2019) ≥ (cos x)^2018 + (sin x)^(-2019)}
  = {x | x ∈ Ico (-π/4) 0 ∪ Ico π/4 (π/2) ∪ Ioc π (5 * π / 4) ∪ Ioc (3 * π / 2) (7 * π / 4)} :=
sorry

end inequality_solution_l495_495258


namespace complex_number_quadrant_l495_495632

theorem complex_number_quadrant 
  (i : ℂ) (hi : i.im = 1 ∧ i.re = 0)
  (x y : ℝ) 
  (h : (x + i) * i = y - i) : 
  x < 0 ∧ y < 0 := 
sorry

end complex_number_quadrant_l495_495632


namespace cyclist_south_speed_l495_495112

theorem cyclist_south_speed :
  ∃ (v : ℝ), (let t := 1.4285714285714286 in ∀ (north_speed south_speed : ℝ),
                north_speed = 10 → south_speed = v →
                (north_speed + south_speed) * t = 50) ∧ v = 25 :=
by
  sorry

end cyclist_south_speed_l495_495112


namespace rectangular_field_area_l495_495199

theorem rectangular_field_area (w : ℕ) (h : ℕ) (P : ℕ) (A : ℕ) (h_length : h = 3 * w) (h_perimeter : 2 * (w + h) = 72) : A = w * h := 
by
  -- Given conditions
  have h_eq : h = 3 * w := h_length
  rewrite [h_eq] at *
  -- Given perimeter equals 8w = 72
  have w_value : w = 9 := sorry
  -- Length calculated as 3w = 27
  have h_value : h = 27 := sorry
  -- Now we calculate the area
  calc
    A = w * h : by sorry
    ... = 9 * 27 : by sorry
    ... = 243 : by sorry

end rectangular_field_area_l495_495199


namespace sum_of_divisors_pow_sum_of_divisors_prod_sum_of_divisors_prod_three_equal_sum_of_divisors_605_637_equal_sum_of_divisors_99_125_equal_sum_of_divisors_8214000_18396875_sum_of_divisors_72_sum_of_divisors_399_l495_495124

variables {p q r : ℕ} {α β γ : ℕ}

def is_prime (n : ℕ) : Prop := ∀ x ∈ (finset.range (n+1)).filter (λ d, d > 0), d = 1 ∨ d = n

theorem sum_of_divisors_pow (hp : is_prime p) (hα : α > 0) :
  (finset.range (α + 1)).sum (λ i, p^i) = (p^(α+1) - 1) / (p - 1) := sorry

theorem sum_of_divisors_prod (hp : is_prime p) (hq : is_prime q) (hα : α > 0) (hβ : β > 0) :
  (finset.range (α + 1)).sum (λ i, (finset.range (β + 1)).sum (λ j, (p^i) * (q^j))) =
  ((p^(α+1) - 1) / (p-1)) * ((q^(β+1) - 1) / (q-1)) := sorry

theorem sum_of_divisors_prod_three (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) (hα : α > 0) (hβ : β > 0) (hγ : γ > 0) :
  (finset.range (α + 1)).sum (λ i, (finset.range (β + 1)).sum (λ j, (finset.range (γ + 1)).sum (λ k, (p^i) * (q^j) * (r^k)))) =
  ((p^(α+1) - 1) / (p-1)) * ((q^(β+1) - 1) / (q-1)) * ((r^(γ+1) - 1) / (r-1)) := sorry

theorem equal_sum_of_divisors_605_637 :
  let n1 := 605; let n2 := 637 in 
  (∑ d in (finset.range (n1+1)).filter (λ x, n1 % x = 0), d) =
  (∑ d in (finset.range (n2+1)).filter (λ x, n2 % x = 0), d) :=
sorry

theorem equal_sum_of_divisors_99_125 :
  let n1 := 99; let n2 := 125 in 
  (∑ d in (finset.range (n1+1)).filter (λ x, n1 % x = 0), d) =
  (∑ d in (finset.range (n2+1)).filter (λ x, n2 % x = 0), d) :=
sorry

theorem equal_sum_of_divisors_8214000_18396875 :
  let n1 := 8214000; let n2 := 18396875 in 
  (∑ d in (finset.range (n1+1)).filter (λ x, n1 % x = 0), d) =
  (∑ d in (finset.range (n2+1)).filter (λ x, n2 % x = 0), d) :=
sorry

theorem sum_of_divisors_72 (n : ℕ) :
  (∑ d in (finset.range (n+1)).filter (λ x, n % x = 0), d) = 72 ↔ 
  n = 71 ∨ n = 46 ∨ n = 51 ∨ n = 55 ∨ n = 30 :=
sorry

theorem sum_of_divisors_399 (n : ℕ) :
  (∑ d in (finset.range (n+1)).filter (λ x, n % x = 0), d) = 399 ↔ 
  n = 196 ∨ n = 242 :=
sorry

end sum_of_divisors_pow_sum_of_divisors_prod_sum_of_divisors_prod_three_equal_sum_of_divisors_605_637_equal_sum_of_divisors_99_125_equal_sum_of_divisors_8214000_18396875_sum_of_divisors_72_sum_of_divisors_399_l495_495124


namespace time_for_A_to_finish_work_alone_l495_495906

theorem time_for_A_to_finish_work_alone :
  (∃ A : ℕ, (1 / A : ℚ) / ((1 / A : ℚ) + 1 / 15) = 1980 / 3300) → 
  ∃ A : ℕ, A = 10 :=
begin
  sorry
end

end time_for_A_to_finish_work_alone_l495_495906


namespace net_income_l495_495216

def tax (y : ℝ) : ℝ :=
  if y < 800 then 0 else 0.2 * (y - 800)

theorem net_income (x : ℝ) (h₁ : tax x = 130) : x - 130 = 1320 :=
by
  -- Preliminary steps
  sorry

end net_income_l495_495216


namespace geometric_series_sum_l495_495855

theorem geometric_series_sum
  (a r : ℚ) (n : ℕ)
  (ha : a = 1/4)
  (hr : r = 1/4)
  (hn : n = 5) :
  (∑ i in finset.range n, a * r^i) = 341 / 1024 := by
  sorry

end geometric_series_sum_l495_495855


namespace remainder_M_divided_by_1000_l495_495470

/-- Define flag problem parameters -/
def flagpoles: ℕ := 2
def blue_flags: ℕ := 15
def green_flags: ℕ := 10

/-- Condition: Two flagpoles, 15 blue flags and 10 green flags -/
def arrangable_flags (flagpoles blue_flags green_flags: ℕ) : Prop :=
  blue_flags + green_flags = 25 ∧ flagpoles = 2

/-- Condition: Each pole contains at least one flag -/
def each_pole_has_flag (arranged_flags: ℕ) : Prop :=
  arranged_flags > 0

/-- Condition: No two green flags are adjacent in any arrangement -/
def no_adjacent_green_flags (arranged_greens: ℕ) : Prop :=
  arranged_greens > 0

/-- Main theorem statement with correct answer -/
theorem remainder_M_divided_by_1000 (M: ℕ) : 
  arrangable_flags flagpoles blue_flags green_flags ∧ 
  each_pole_has_flag M ∧ 
  no_adjacent_green_flags green_flags ∧ 
  M % 1000 = 122
:= sorry

end remainder_M_divided_by_1000_l495_495470


namespace incorrect_statement_c_l495_495131

def supplementary_angles (α β : ℝ) : Prop :=
  α + β = 180

def incorrect_supplementary_definition : Prop :=
  ∀ α β : ℝ, (α + β = 180) → ¬(angles_are_on_same_side α β)

def angles_are_on_same_side (α β : ℝ) : Prop :=
  -- Assuming a definition here just for syntax sake; actual definition should come from geometry
  true -- placeholder for actual condition stating both angles are on the same side

theorem incorrect_statement_c : incorrect_supplementary_definition :=
by sorry

end incorrect_statement_c_l495_495131


namespace largest_difference_rounding_l495_495218

variable (A B : ℝ)
variable (estimate_A estimate_B : ℝ)
variable (within_A within_B : ℝ)
variable (diff : ℝ)

axiom est_A : estimate_A = 55000
axiom est_B : estimate_B = 58000
axiom cond_A : within_A = 0.15
axiom cond_B : within_B = 0.10

axiom bounds_A : 46750 ≤ A ∧ A ≤ 63250
axiom bounds_B : 52727 ≤ B ∧ B ≤ 64444

noncomputable def max_possible_difference : ℝ :=
  max (abs (B - A)) (abs (A - B))

theorem largest_difference_rounding :
  max_possible_difference A B = 18000 :=
by
  sorry

end largest_difference_rounding_l495_495218


namespace shopkeeper_profit_percentage_l495_495541

def cost_price (C : ℝ) : ℝ := 15 * C
def selling_price (C : ℝ) : ℝ := 20 * C
def profit (C : ℝ) : ℝ := selling_price C - cost_price C
def profit_percentage (C : ℝ) : ℝ := (profit C / cost_price C) * 100

theorem shopkeeper_profit_percentage (C : ℝ) : profit_percentage C = 33.33 := by
  sorry

end shopkeeper_profit_percentage_l495_495541


namespace identify_roles_l495_495769

-- Define the number of liars and truth-tellers
def num_liars : Nat := 1000
def num_truth_tellers : Nat := 1000

-- Define the properties of the individuals
def first_person_is_liar := true
def second_person_is_truth_teller := true

-- The main statement equivalent to the problem
theorem identify_roles : first_person_is_liar = true ∧ second_person_is_truth_teller = true := by
  sorry

end identify_roles_l495_495769


namespace ceiling_fraction_evaluation_l495_495249

theorem ceiling_fraction_evaluation :
  (Int.ceil ((19 : ℚ) / 8 - Int.ceil ((45 : ℚ) / 19)) / Int.ceil ((45 : ℚ) / 8 + Int.ceil ((8 * 19 : ℚ) / 45))) = 0 :=
by
  sorry

end ceiling_fraction_evaluation_l495_495249


namespace pentagon_diagonals_l495_495164

theorem pentagon_diagonals (pentagon : Type) [fintype pentagon] [decidable_eq pentagon] 
  (h : fintype.card pentagon = 5) :
  ∃ diagonal_count : ℕ, diagonal_count = 2 :=
by {
  use 2,
  sorry
}

end pentagon_diagonals_l495_495164


namespace find_c_l495_495275

theorem find_c (c : ℝ) (h : ∀ x : ℝ, c * x^2 ≥ real.log (1 + x^2)) :
  2 * (c / 3 - (real.log(2) - 2 + real.pi / 2)) = 4 → c = 3 * real.log(2) + 3 * real.pi / 2 :=
by {
  sorry
}

end find_c_l495_495275


namespace number_of_initial_values_l495_495987

noncomputable def f (x : ℝ) : ℝ :=
  if 2 * x < 1 then 2 * x else 2 * x - 1

def sequence (x0 : ℝ) (n : ℕ) : ℝ :=
  Nat.iterate f n x0

theorem number_of_initial_values :
  ∃! n : ℕ, n = 64 ∧ ∀ (x0 : ℝ), (0 ≤ x0 ∧ x0 < 1) → sequence x0 6 = x0 → count (λ x0, (0 ≤ x0 ∧ x0 < 1) ∧ sequence x0 6 = x0) n := sorry

end number_of_initial_values_l495_495987


namespace theoretical_yield_H2SO4_l495_495244

-- Define the theoretical yield calculation problem in terms of moles of reactions and products
theorem theoretical_yield_H2SO4 
  (moles_SO3 : ℝ) (moles_H2O : ℝ) 
  (reaction : moles_SO3 + moles_H2O = 2.0 + 1.5) 
  (limiting_reactant_H2O : moles_H2O = 1.5) : 
  1.5 = moles_H2O * 1 :=
  sorry

end theoretical_yield_H2SO4_l495_495244


namespace zoo_ticket_problem_l495_495527

theorem zoo_ticket_problem :
  ∀ (total_amount adult_ticket_cost children_ticket_cost : ℕ)
    (num_adult_tickets : ℕ),
  total_amount = 119 →
  adult_ticket_cost = 21 →
  children_ticket_cost = 14 →
  num_adult_tickets = 4 →
  6 = (num_adult_tickets + (total_amount - num_adult_tickets * adult_ticket_cost) / children_ticket_cost) :=
by 
  intros total_amount adult_ticket_cost children_ticket_cost num_adult_tickets 
         total_amt_eq adult_ticket_cost_eq children_ticket_cost_eq num_adult_tickets_eq
  sorry

end zoo_ticket_problem_l495_495527


namespace num_integers_between_l495_495331

-- Define the constants
def a : ℝ := 10
def b₁ : ℝ := 0.5
def b₂ : ℝ := 0.6

-- Define the cubes
def x : ℝ := (a + b₁)^3
def y : ℝ := (a + b₂)^3

-- Define the function to count the integers within the interval
def count_integers_between (x y : ℝ) : ℕ :=
  let min_int := Int.ceil x
  let max_int := Int.floor y
  (max_int - min_int + 1).toNat

-- The statement to prove
theorem num_integers_between : count_integers_between x y = 33 := by
  sorry

end num_integers_between_l495_495331


namespace ellipse_major_axis_length_l495_495219

noncomputable def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_tangent_to_line (f1 f2 : point) (y : ℝ) : Prop := distance (f1.1, 2 * y - f1.2) f2 = 53

theorem ellipse_major_axis_length :
  ∃ (f1 f2 : point), f1 = (3, 15) ∧ f2 = (31, 40) ∧ is_tangent_to_line f1 f2 5 :=
begin
  use (3, 15),
  use (31, 40),
  split; simp,
  split; simp,
  { dsimp [is_tangent_to_line], rw distance,
    -- Skip the proof
    sorry }
end

end ellipse_major_axis_length_l495_495219
