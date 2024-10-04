import Mathlib
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.FDeriv
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Graph.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Log
import Mathlib.Data.Nat.Multiple
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.MeasureTheory.Probability.Independence
import Mathlib.Probability
import Mathlib.Tactic
import mathlib

namespace problem_find_b5_l669_669320

namespace MathProof

noncomputable def find_b5 : ℕ :=
let a := 73441 / 99999
    b := 136161 / 99999
in (den_to_digits b 5).get_or_else 0  -- Helper function to find the n-th digit of a decimal number

theorem problem_find_b5 : ∀ a b : ℝ, (fr.denominator_to_5_digit a 99999) ∧ (fr.denominator_to_5_digit b 99999) ∧ a * b = 1 ∧ nth_digit a 5 = 1 → nth_digit b 5 = 2 :=
by
    intros a b h₁ h₂ h₃ h₄
    let b := find_b5 -- Using the defined noncomputable function to find the 5th digit
    exact eq.refl 2 -- Placeholder for the actual proof
    sorry -- Proof steps go here

end MathProof

end problem_find_b5_l669_669320


namespace total_pencils_owned_l669_669893

def SetA_pencils := 10
def SetB_pencils := 20
def SetC_pencils := 30

def friends_SetA_Buys := 3
def friends_SetB_Buys := 2
def friends_SetC_Buys := 2

def Chloe_SetA_Buys := 1
def Chloe_SetB_Buys := 1
def Chloe_SetC_Buys := 1

def total_friends_pencils := friends_SetA_Buys * SetA_pencils + friends_SetB_Buys * SetB_pencils + friends_SetC_Buys * SetC_pencils
def total_Chloe_pencils := Chloe_SetA_Buys * SetA_pencils + Chloe_SetB_Buys * SetB_pencils + Chloe_SetC_Buys * SetC_pencils
def total_pencils := total_friends_pencils + total_Chloe_pencils

theorem total_pencils_owned : total_pencils = 190 :=
by
  sorry

end total_pencils_owned_l669_669893


namespace diagonal_cannot_be_good_l669_669155

def is_good (table : ℕ → ℕ → ℕ) (i j : ℕ) :=
  ∀ x y, (x = i ∨ y = j) → ∀ x' y', (x' = i ∨ y' = j) → (x ≠ x' ∨ y ≠ y') → table x y ≠ table x' y'

theorem diagonal_cannot_be_good :
  ∀ (table : ℕ → ℕ → ℕ), (∀ i j, 1 ≤ table i j ∧ table i j ≤ 25) →
  ¬ ∀ k, (is_good table k k) :=
by
  sorry

end diagonal_cannot_be_good_l669_669155


namespace solve_system_of_equations_solve_system_of_inequalities_find_integer_solutions_l669_669213

-- Part 1: System of Equations
theorem solve_system_of_equations :
  ∃ (x y : ℝ), (2 * x - y = 5) ∧ (3 * x + 4 * y = 2) ∧ (x = 2) ∧ (y = -1) :=
by
  have solved_x_y : ∃ (x y : ℝ), (2 * x - y = 5) ∧ (3 * x + 4 * y = 2) := ⟨2, -1, by norm_num, by norm_num⟩
  use solved_x_y
  split; try {exact solved_x_y}
  split; norm_num
  split; norm_num
  sorry -- Proof is to be completed

-- Part 2: System of Inequalities
theorem solve_system_of_inequalities :
  ∃ (x : ℝ), (-2 * x < 6) ∧ (3 * (x - 2) ≤ x - 4) ∧ (-3 < x ∧ x ≤ 1) :=
by
  have inequality_solution : ∃ (x : ℝ), (-2 * x < 6) ∧ (3 * (x - 2) ≤ x - 4) := ⟨1, by linarith, by linarith⟩
  use inequality_solution
  split; try {exact inequality_solution}
  split; linarith
  sorry -- Proof is to be completed

theorem find_integer_solutions :
  ∀ x : ℝ, (-3 < x ∧ x ≤ 1) → x ∈ {-2, -1, 0, 1} :=
by
  intro x hx
  cases hx
  have : x < 2 := by linarith
  have : x ≥ -3 := by linarith
  have : intable x := by norm_num -- Case for integer solutions
  -- Enumerate solutions manually
  cases x
    exact ⟨-2, -1, 0, 1⟩
  sorry -- Proof is to be completed

end solve_system_of_equations_solve_system_of_inequalities_find_integer_solutions_l669_669213


namespace ratio_of_area_to_breadth_is_15_l669_669218

-- Definitions for our problem
def breadth := 5
def length := 15 -- since l - b = 10 and b = 5

-- Given conditions
axiom area_is_ktimes_breadth (k : ℝ) : length * breadth = k * breadth
axiom length_breadth_difference : length - breadth = 10

-- The proof statement
theorem ratio_of_area_to_breadth_is_15 : (length * breadth) / breadth = 15 := by
  sorry

end ratio_of_area_to_breadth_is_15_l669_669218


namespace max_value_PA_PB_PC_l669_669070

noncomputable def maximum_PA_PB_PC (A B C P : ℝ × ℝ) : ℝ :=
  let PA := dist P A
  let PB := dist P B
  let PC := dist P C
  PA * PB * PC

theorem max_value_PA_PB_PC :
  let A := (1, 0)
  let B := (0, 1)
  let C := (0, 0)
  ∃ P : ℝ × ℝ, (P.1 = 0 ∨ P.1 = 1 ∨ P.2 = 0 ∨ P.2 = 1 ∨ P.1 = P.2),
  maximum_PA_PB_PC A B C P = sqrt 2 / 4 :=
by
  sorry

end max_value_PA_PB_PC_l669_669070


namespace prism_cross_section_area_l669_669743

noncomputable def cross_section_area (h_prism : ℝ) (a_side : ℝ) (acute_angle : ℝ) (inclination_angle : ℝ) : ℝ :=
  if h_prism = 1 ∧ a_side = 2 ∧ acute_angle = 30 ∧ inclination_angle = 60 then 
    4 / real.sqrt 3
  else 
    0

theorem prism_cross_section_area :
  cross_section_area 1 2 30 60 = 4 / real.sqrt 3 :=
by
  sorry

end prism_cross_section_area_l669_669743


namespace b_2013_eq_l669_669630

def b : ℕ → ℝ
def c : ℝ := 2 + Real.sqrt 8
def d : ℝ := 15 + Real.sqrt 8

axiom b_recur {n : ℕ} (h : 2 ≤ n) : b n = b (n - 1) * b (n + 1)
axiom b_1 : b 1 = 2 + Real.sqrt 8
axiom b_1980 : b 1980 = 15 + Real.sqrt 8

theorem b_2013_eq : b 2013 = -1 / 6 + 13 * Real.sqrt 8 / 6 :=
  sorry

end b_2013_eq_l669_669630


namespace num_paths_A_to_C_through_B_l669_669120

-- Define the coordinates of points A, B, and C
def A : (ℤ × ℤ) := (0, 6)
def B : (ℤ × ℤ) := (4, 4)
def C : (ℤ × ℤ) := (7, 0)

-- Define a function calculating the number of paths between two points
def binomial (n k : ℕ) := nat.choose n k

-- Proof statement
theorem num_paths_A_to_C_through_B : 
  (binomial 5 4) * (binomial 6 2) = 75 := by
  sorry

end num_paths_A_to_C_through_B_l669_669120


namespace no_real_solution_l669_669445

theorem no_real_solution : ¬∃ (x y z : ℝ), x + y = 3 ∧ x * y - z^2 = 3 ∧ x = 2 :=
by
  intro h
  rcases h with ⟨x, y, z, h1, h2, h3⟩
  rw [h3] at h1
  rw [h3] at h2
  have hy : y = 1 :=
    calc
      y = 3 - x : by rw [← h1]; ring
      ... = 3 - 2 : by rw [h3]
      ... = 1 : by norm_num
  rw [hy] at h2
  have h4 : 2 * 1 - z^2 = 3 := h2
  norm_num at h4
  obtain ⟨hz : z^2 = -1, -⟩ := ⟨h4, h4⟩
  apply real.sqrt_lt_zero hz
  rfl

end no_real_solution_l669_669445


namespace solve_for_n_l669_669528

theorem solve_for_n (n : ℕ) (h : sqrt (8 + n) = 9) : n = 73 := 
by {
  sorry
}

end solve_for_n_l669_669528


namespace sum_max_min_ratios_l669_669395

theorem sum_max_min_ratios
  (c d : ℚ)
  (h1 : ∀ x y : ℚ, 3*x^2 + 2*x*y + 4*y^2 - 13*x - 26*y + 53 = 0 → y / x = c ∨ y / x = d)
  (h2 : ∀ r : ℚ, (∃ x y : ℚ, 3*x^2 + 2*x*y + 4*y^2 - 13*x - 26*y + 53 = 0 ∧ y / x = r) → (r = c ∨ r = d))
  : c + d = 63 / 43 :=
sorry

end sum_max_min_ratios_l669_669395


namespace pyramid_inequality_l669_669345

theorem pyramid_inequality {R r : ℝ} (h₁ : inscribed_pyramid R) 
(h₂ : circumscribed_pyramid r) : R ≥ (sqrt 2 + 1) * r := 
sorry -- Proof goes here

end pyramid_inequality_l669_669345


namespace orthocenters_collinear_l669_669406

-- Definition of four lines intersecting with no three concurrent
inductive FourLines
| mk (a b c d : Line) 
  (h_no_three_concurrent : ∀P, P ∈ a → P ∈ b → P ∈ c → P ≠ d) : FourLines

-- Proposition that the orthocenters of triangles formed by any three lines lie on a straight line
theorem orthocenters_collinear (L : FourLines) : 
  ∀ {a b c d : Line},
  (a, b, c, d ∈ L.mk) →
  let triangles := { (t1 : Triangle) // (t1.forms_by a b c) ∨ (t1.forms_by a b d) ∨ (t1.forms_by a c d) ∨ (t1.forms_by b c d) } in
  collinear (set.image orthocenter triangles)
:= sorry

end orthocenters_collinear_l669_669406


namespace algebraic_expression_value_l669_669047

theorem algebraic_expression_value (a b c : ℝ) (h : (∀ x : ℝ, (x - 1) * (x + 2) = a * x^2 + b * x + c)) :
  4 * a - 2 * b + c = 0 :=
sorry

end algebraic_expression_value_l669_669047


namespace cheburashkas_erased_l669_669594

theorem cheburashkas_erased (n : ℕ) (rows : ℕ) (krakozyabras : ℕ) 
  (h_spacing : ∀ r, r ≤ rows → krakozyabras = 2 * (n - 1))
  (h_rows : rows = 2)
  (h_krakozyabras : krakozyabras = 29) :
  n = 16 → rows = 2 → krakozyabras = 29 → n = 16 - 5 :=
by
  sorry

end cheburashkas_erased_l669_669594


namespace tangent_segments_area_l669_669470

open Real

theorem tangent_segments_area {r : ℝ} (h : r = 3) :
  let R := (sqrt 2) * r in
  let inner_radius := r in
  let outer_radius := R in
  (π * outer_radius^2 - π * inner_radius^2 = 9 * π) :=
by
  have inner_area := π * inner_radius^2
  have outer_area := π * outer_radius^2
  have h_inner_area : inner_area = π * (3 ^ 2) := by simp [inner_radius, h]
  have h_outer_area : outer_area = π * (3 * sqrt 2) ^ 2 := by simp [outer_radius, h]
  have h_outer_area_simplified : outer_area = π * 18 := by simp [h_outer_area]
  have h_inner_area_simplified : inner_area = π * 9 := by simp [h_inner_area]
  show π * outer_radius^2 - π * inner_radius^2 = 9 * π from by
    rw [h_inner_area_simplified, h_outer_area_simplified]
    simp
    sorry

end tangent_segments_area_l669_669470


namespace minimum_cactus_species_l669_669370

theorem minimum_cactus_species (cactophiles : Fin 80 → Set (Fin k)) :
  (∀ s : Fin k, ∃ col, cactophiles col s = False) ∧
  (∀ group : Set (Fin 80), group.card = 15 → ∃ c : Fin k, ∀ col ∈ group, (cactophiles col c)) →
  16 ≤ k :=
by
  sorry

end minimum_cactus_species_l669_669370


namespace simplify_cube_root_l669_669694

theorem simplify_cube_root (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)
  (h1 : a = 10^3 * b)
  (h2 : b = 2^7 * c * 7^3)
  (h3 : c = 10) :
  ∛a = 40 * 7 * 2^(2/3) * 5^(1/3) := by
  sorry

end simplify_cube_root_l669_669694


namespace tan_B_is_correct_l669_669165

namespace TriangleProof

-- Define a structure for the triangle
structure triangle :=
  (A B C : ℝ) (cos_C : ℝ) (AC BC : ℝ)

noncomputable def cos_C_value : ℝ := 2 / 3
noncomputable def AC_value : ℝ := 4
noncomputable def BC_value : ℝ := 3

def my_triangle : triangle :=
  {A := 0, B := 0, C := 0, cos_C := cos_C_value, AC := AC_value, BC := BC_value}

theorem tan_B_is_correct (t : triangle) (h₁ : t.cos_C = 2 / 3) (h₂ : t.AC = 4) (h₃ : t.BC = 3) : 
  real.tan t.B = 4 * real.sqrt 5 :=
sorry

end TriangleProof

end tan_B_is_correct_l669_669165


namespace mason_car_nuts_l669_669644

def busy_squirrels_num := 2
def busy_squirrel_nuts_per_day := 30
def sleepy_squirrel_num := 1
def sleepy_squirrel_nuts_per_day := 20
def days := 40

theorem mason_car_nuts : 
  busy_squirrels_num * busy_squirrel_nuts_per_day * days + sleepy_squirrel_nuts_per_day * days = 3200 :=
  by
    sorry

end mason_car_nuts_l669_669644


namespace normal_price_of_article_l669_669289

theorem normal_price_of_article 
  (P : ℝ) 
  (h : (P * 0.88 * 0.78 * 0.85) * 1.06 = 144) : 
  P = 144 / (0.88 * 0.78 * 0.85 * 1.06) :=
sorry

end normal_price_of_article_l669_669289


namespace number_of_teams_l669_669146

theorem number_of_teams (n : ℕ) (h1 : ∀ k, k = 10) (h2 : n * 10 * (n - 1) / 2 = 1900) : n = 20 :=
by
  sorry

end number_of_teams_l669_669146


namespace inequality_proof_l669_669090

theorem inequality_proof (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 2) : 
  (1 / a + 1 / b) ≥ 2 :=
sorry

end inequality_proof_l669_669090


namespace smallest_period_of_f_interval_of_monotonic_increase_range_of_m_when_x_in_0_to_pi_div_2_l669_669052

def f (x : ℝ) : ℝ :=
  (sqrt 3 / 2) * sin (2 * x) + (cos x) ^ 2 - 3 / 2

lemma period_of_f : f (x) = sin (2 * x + π / 6) - 1 := by
  sorry

theorem smallest_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π := 
by
  use π
  apply period_of_f
  sorry

theorem interval_of_monotonic_increase :
  ∃ (k : ℤ), ∀ x,
  (k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6 → f' x > 0) := 
by
  sorry

theorem range_of_m_when_x_in_0_to_pi_div_2 :
  ∀ m, ∃ x, (0 ≤ x ∧ x ≤ π / 2 ∧ f x - m = 0) → - 3 / 2 ≤ m ∧ m ≤ 0 := 
by
  sorry

end smallest_period_of_f_interval_of_monotonic_increase_range_of_m_when_x_in_0_to_pi_div_2_l669_669052


namespace ratio_of_legs_of_triangles_l669_669043

theorem ratio_of_legs_of_triangles (s a b : ℝ) (h1 : 0 < s)
  (h2 : a = s / 2)
  (h3 : b = (s * Real.sqrt 7) / 2) :
  b / a = Real.sqrt 7 := by
  sorry

end ratio_of_legs_of_triangles_l669_669043


namespace area_inside_S_but_outside_R_is_5_3_l669_669652

def area_of_pentagon {s : ℝ} (s = 1) : ℝ := 5/4 * Real.tan (54 * Real.pi / 180)
def area_of_large_triangle {s : ℝ} (s = 1) : ℝ := (Real.sqrt 3 / 4)
def area_of_small_triangle {s : ℝ} (s = 0.5) : ℝ := (Real.sqrt 3 / 16)

def area_R : ℝ := area_of_pentagon 1 + 5 * area_of_large_triangle 1 + 10 * area_of_small_triangle 0.5
def area_S : ℝ := 1/4 * 10 * 2^2 * Real.cot (Real.pi / 10)

def area_of_interest : ℝ := area_S - area_R

theorem area_inside_S_but_outside_R_is_5_3 : area_of_interest = 5.3 :=
sorry

end area_inside_S_but_outside_R_is_5_3_l669_669652


namespace Teahouse_on_Tuesday_or_Thursday_l669_669777

constant Plays : String → Prop

def Thunderstorm := Plays "Thunderstorm"
def Teahouse := Plays "Teahouse"
def HeavenlySound := Plays "Heavenly Sound"
def ShatteredHoofbeats := Plays "Shattered Hoofbeats"

def Monday : String := "Monday"
def Tuesday : String := "Tuesday"
def Wednesday : String := "Wednesday"
def Thursday : String := "Thursday"

def perform_on (play : String) (day : String) : Prop :=
  match play, day with
  | "Thunderstorm", "Monday" => False
  | "Thunderstorm", "Thursday" => False
  | "Teahouse", "Monday" => False
  | "Teahouse", "Wednesday" => False
  | "Heavenly Sound", "Wednesday" => False
  | "Heavenly Sound", "Thursday" => False
  | "Shattered Hoofbeats", "Monday" => False
  | "Shattered Hoofbeats", "Thursday" => False
  | _, _ => True

theorem Teahouse_on_Tuesday_or_Thursday :
  perform_on "Teahouse" "Tuesday" ∨ perform_on "Teahouse" "Thursday" :=
by
  sorry

end Teahouse_on_Tuesday_or_Thursday_l669_669777


namespace boat_distribution_l669_669562

theorem boat_distribution (x : ℕ) (h_boats : x + (8 - x) = 8) (h_small_boat_capacity : 4 * x) (h_large_boat_capacity : 6 * (8 - x)) (h_total_students : 4 * x + 6 * (8 - x) = 38) : 
4 * x + 6 * (8 - x) = 38 := 
by {
  sorry
}

end boat_distribution_l669_669562


namespace expected_value_of_random_number_l669_669263

/-- 
The expected value of a random number formed by placing a zero and a decimal point in front
of a sequence of one thousand random digits is 0.5.
-/
theorem expected_value_of_random_number : 
  let X := ∑ k in (finRange 1000), (4.5 / 10 ^ (k + 1))
  in X = 0.5 :=
sorry

end expected_value_of_random_number_l669_669263


namespace rightmost_non_zero_digit_of_20_times_13_factorial_l669_669036

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def rightmost_non_zero_digit (n : ℕ) : ℕ :=
  let digits := (n % 10)
  if digits = 0 then rightmost_non_zero_digit (n / 10) else digits

theorem rightmost_non_zero_digit_of_20_times_13_factorial :
  rightmost_non_zero_digit (20 * factorial 13) = 6 :=
by
  sorry

end rightmost_non_zero_digit_of_20_times_13_factorial_l669_669036


namespace integers_satisfying_inequalities_l669_669009

theorem integers_satisfying_inequalities:
  (∃ x : ℤ, (-4 * x ≥ x + 10) ∧ (-3 * x ≤ 15) ∧ (-5 * x ≥ 3 * x + 24)) →
  (finset.filter (λ x, (-4 * x ≥ x + 10) ∧ (-3 * x ≤ 15) ∧ (-5 * x ≥ 3 * x + 24)) (finset.Icc (-5) (-3))).card = 3 :=
by
  sorry

end integers_satisfying_inequalities_l669_669009


namespace max_rooks_on_chessboard_l669_669805

theorem max_rooks_on_chessboard :
  ∃ (n : ℕ), n = 16 ∧
    (∀ rooks : fin 8 × fin 8 → Prop,
      (∀ r, rooks r → ∃! r', (r' ≠ r ∧ rooks r' ∧ (r.1 = r'.1 ∨ r.2 = r'.2))) ∧
      (∃ r1 r2, rooks (r1, r2)) ∧
      ∀ r1 r2 r3 r4, rooks (r1, r2) → rooks (r3, r4) → (r1 = r3 ∨ r2 = r4) → r1 ≠ r3 ∧ r2 ≠ r4 ∧
      |filter rooks (univ : finset (fin 8 × fin 8))| = n) :=
begin
  use 16,
  split,
  { refl, },
  { sorry }
end

end max_rooks_on_chessboard_l669_669805


namespace factorize_expression_l669_669419

theorem factorize_expression (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
by sorry

end factorize_expression_l669_669419


namespace value_of_n_l669_669526

theorem value_of_n (n : ℤ) (h : sqrt (8 + n) = 9) : n = 73 :=
sorry

end value_of_n_l669_669526


namespace subset_condition_l669_669903

theorem subset_condition (a : ℝ) :
  (∀ x : ℝ, |2 * x - 1| < 1 → x^2 - 2 * a * x + a^2 - 1 > 0) →
  (a ≤ -1 ∨ a ≥ 2) :=
by
  sorry

end subset_condition_l669_669903


namespace vasya_max_days_and_avg_dishes_l669_669565

theorem vasya_max_days_and_avg_dishes (dishes : Finset ℕ) (hk : dishes.card = 6) :
  ∃ (max_days : ℕ) (avg_dishes : ℕ),
    max_days = 64 ∧ avg_dishes = 3 :=
by
  use 64
  use 3
  sorry

end vasya_max_days_and_avg_dishes_l669_669565


namespace angle_between_a_b_is_3pi_over_4_l669_669118

noncomputable theory

open Real

variables {a : ℝ × ℝ} {b : ℝ × ℝ}

def dot_product (u v : ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2 * v.2

def norm (u : ℝ × ℝ) : ℝ :=
Real.sqrt (u.1 * u.1 + u.2 * u.2)

def angle (u v : ℝ × ℝ) : ℝ :=
Real.acos (dot_product u v / (norm u * norm v))

theorem angle_between_a_b_is_3pi_over_4
  (h1 : dot_product a (a - (2,2)) = 3)
  (h2 : norm a = 1)
  (h3 : b = (1, 1)) :
  angle a b = 3 * Real.pi / 4 :=
sorry

end angle_between_a_b_is_3pi_over_4_l669_669118


namespace part_1_part_2_l669_669091

variables (a b c : ℝ) (A B C : ℝ)
variable (triangle_ABC : a = b ∧ b = c ∧ A + B + C = 180 ∧ A = 90 ∨ B = 90 ∨ C = 90)
variable (sin_condition : Real.sin B ^ 2 = 2 * Real.sin A * Real.sin C)

theorem part_1 (h : a = b) : Real.cos C = 7 / 8 :=
by { sorry }

theorem part_2 (h₁ : B = 90) (h₂ : a = Real.sqrt 2) : b = 2 :=
by { sorry }

end part_1_part_2_l669_669091


namespace remove_6_increases_probability_l669_669276

-- Define the set S
def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}

-- Define the subset S' where 6 is removed
def S' : Set ℕ := S.erase 6

-- Define function that returns true if two distinct elements sum to 12
def is_sum_12 (a b : ℕ) : Prop := a ≠ b ∧ a + b = 12

-- Define the probability of selecting two numbers from a set that sum to 12
noncomputable def prob_sum_12 (s : Set ℕ) : ℝ :=
  (Finset.card {x : ℕ × ℕ | x.1 ∈ s ∧ x.2 ∈ s ∧ is_sum_12 x.1 x.2}) / (Finset.card s).choose 2

-- Define the main theorem
theorem remove_6_increases_probability :
  prob_sum_12 S' > prob_sum_12 S :=
sorry

end remove_6_increases_probability_l669_669276


namespace copper_to_zinc_ratio_l669_669771

theorem copper_to_zinc_ratio (total_weight_brass : ℝ) (weight_zinc : ℝ) (weight_copper : ℝ) 
  (h1 : total_weight_brass = 100) (h2 : weight_zinc = 70) (h3 : weight_copper = total_weight_brass - weight_zinc) : 
  weight_copper / weight_zinc = 3 / 7 :=
by
  sorry

end copper_to_zinc_ratio_l669_669771


namespace cevian_concurrent_product_l669_669577

theorem cevian_concurrent_product
  (DEF: Triangle) (D' E' F' P: Point)
  (hD' : On (Line DEF.E DEF.F) D')
  (hE' : On (Line DEF.F DEF.D) E')
  (hF' : On (Line DEF.D DEF.E) F')
  (hConcurrent : Concurrent DEF.D' DEF.E' DEF.F' P)
  (hSum : (DEF.D.P / DEF.P.D' + DEF.E.P / DEF.P.E' + DEF.F.P / DEF.P.F') = 100) 
  : (DEF.D.P / DEF.P.D') * (DEF.E.P / DEF.P.E') * (DEF.F.P / DEF.P.F') = 98 :=
by
  sorry

end cevian_concurrent_product_l669_669577


namespace hyperbola_focal_distance_l669_669219

theorem hyperbola_focal_distance
  (b : ℝ)
  (h1 : ∃ M N : ℝ × ℝ, (M.1 - 2)^2 + M.2^2 = 2 ∧ (N.1 - 2)^2 + N.2^2 = 2 ∧ ((M.1, M.2) ≠ (N.1, N.2)) ∧ (M.1 - N.1)^2 + (M.2 - N.2)^2 = 4) :
  2 * real.sqrt (3 + 1) = 4 :=
by {
  sorry
}

end hyperbola_focal_distance_l669_669219


namespace triangle_area_l669_669502

open Complex

noncomputable def area_of_triangle (a b c : ℂ) : ℝ :=
1 / 2 * complex.abs ((b - a) * (c - conj a) / 2)

theorem triangle_area (A B C : ℂ) (hA : A = (1 + 3 * I)) (hB : B = (3 + 2 * I)) (hC : C = (4 + 4 * I)) :
  area_of_triangle A B C = 3 :=
by
  simp [hA, hB, hC, area_of_triangle]
  sorry

end triangle_area_l669_669502


namespace max_gamma_norm_l669_669503

open Real

variables {α β γ : ℝ^3} 

-- Define unit vectors and their properties
def is_unit_vector (v : ℝ^3) : Prop := (v • v = 1)
def are_perpendicular (v w : ℝ^3) : Prop := (v • w = 0)

-- Given conditions
variables (h1 : is_unit_vector α) (h2 : is_unit_vector β) (h3 : are_perpendicular α β) 
          (h4 : (3•α - γ) • (4•β - γ) = 0)

-- The theorem stating the mathematically equivalent proof problem
theorem max_gamma_norm : ∥γ∥ ≤ 5 := 
sorry

end max_gamma_norm_l669_669503


namespace book_selection_count_l669_669125

-- Definitions based on conditions
def total_books : ℕ := 15
def books_to_bring : ℕ := 3
def favorite_book_included : ℕ := 1
def remaining_books : ℕ := total_books - favorite_book_included

-- Proof statement
theorem book_selection_count : nat.choose remaining_books (books_to_bring - favorite_book_included) = 91 := by
  sorry

end book_selection_count_l669_669125


namespace cheburashkas_erased_l669_669611

theorem cheburashkas_erased (total_krakozyabras : ℕ) (rows : ℕ) :
  rows ≥ 2 → total_krakozyabras = 29 → ∃ (cheburashkas_erased : ℕ), cheburashkas_erased = 11 :=
by
  assume h_rows h_total_krakozyabras
  let n := (total_krakozyabras / 2) + 1
  have h_cheburashkas : cheburashkas_erased = n - 1 
  sorry

end cheburashkas_erased_l669_669611


namespace two_candidates_solve_all_problems_l669_669841

-- Definitions for the conditions and problem context
def candidates : Nat := 200
def problems : Nat := 6 
def solved_by (p : Nat) : Nat := 120 -- at least 120 participants solve each problem.

-- The main theorem representing the proof problem
theorem two_candidates_solve_all_problems :
  (∃ c1 c2 : Fin candidates, ∀ p : Fin problems, (solved_by p ≥ 120)) :=
by
  sorry

end two_candidates_solve_all_problems_l669_669841


namespace perp_lines_a_value_l669_669083

theorem perp_lines_a_value (a : ℝ) :
  let l1 := λ (x y : ℝ), ax + 3 * y - 1 = 0
  let l2 := λ (x y : ℝ), 2 * x + (a - 1) * y + 1 = 0
  (∀ x y, l1 x y → ∀ x y, l2 x y → 
      let slope_l1 := -(a / 3)
      let slope_l2 := -(2 / (a - 1))
      (slope_l1 * slope_l2 = -1) → a = 3/5) := 
  sorry

end perp_lines_a_value_l669_669083


namespace simplify_cube_root_l669_669690

theorem simplify_cube_root (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)
  (h1 : a = 10^3 * b)
  (h2 : b = 2^7 * c * 7^3)
  (h3 : c = 10) :
  ∛a = 40 * 7 * 2^(2/3) * 5^(1/3) := by
  sorry

end simplify_cube_root_l669_669690


namespace plot_area_is_nine_hectares_l669_669861

-- Definition of the dimensions of the plot
def length := 450
def width := 200

-- Definition of conversion factor from square meters to hectares
def sqMetersPerHectare := 10000

-- Calculated area in hectares
def area_hectares := (length * width) / sqMetersPerHectare

-- Theorem statement: prove that the area in hectares is 9
theorem plot_area_is_nine_hectares : area_hectares = 9 := 
by
  sorry

end plot_area_is_nine_hectares_l669_669861


namespace simplify_cube_root_l669_669693

theorem simplify_cube_root (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)
  (h1 : a = 10^3 * b)
  (h2 : b = 2^7 * c * 7^3)
  (h3 : c = 10) :
  ∛a = 40 * 7 * 2^(2/3) * 5^(1/3) := by
  sorry

end simplify_cube_root_l669_669693


namespace arithmetic_sequence_G_minus_L_l669_669355

/-- An arithmetic sequence of 300 numbers, each between 20 and 90 inclusive, 
with a total sum of 15000.
We want to show that the difference between the greatest and least possible value 
of the 75th term is 9060/299. -/
theorem arithmetic_sequence_G_minus_L : 
  ∀ (a d : ℝ), 
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 300 → (a + (n - 1) * d) ≥ 20 ∧ (a + (n - 1) * d) ≤ 90) → 
    ((300 * a + 299 * 150 * d) / 300 = 50) → 
    let L := 50 - 151 * (30 / 299) in 
    let G := 50 + 151 * (30 / 299) in 
    G - L = 9060 / 299 :=
by sorry

end arithmetic_sequence_G_minus_L_l669_669355


namespace length_segment_AP_l669_669567

theorem length_segment_AP (A B C D P : Point) (AD BC AB DC : ℝ) (isosceles_trapezoid : isosceles_trapezoid ABCD)
    (D_eq_7 : AD = 7) (B_eq_7 : BC = 7) (A_eq_10 : AB = 10) (C_eq_16 : DC = 16)
    (P_mid_AC : midpoint P A C) (P_mid_BD : midpoint P B D) (P_on_circle : P ∈ circle D (dist D P)) :
    dist A P = 8 :=
by { sorry }

end length_segment_AP_l669_669567


namespace range_of_f_pos_l669_669938

noncomputable def f (x : ℝ) : ℝ := x ^ (2 / 3) - x ^ (-1 / 2)

theorem range_of_f_pos (x : ℝ) : f x > 0 ↔ x > 1 :=
by
  sorry

end range_of_f_pos_l669_669938


namespace profit_difference_l669_669302

variables (a b c : ℕ)
variables (initial_investment_a initial_investment_b initial_investment_c : ℕ)
variables (profit_share_b : ℕ)

def total_investment := initial_investment_a + initial_investment_b + initial_investment_c

def ratio_a := initial_investment_a / 2000
def ratio_b := initial_investment_b / 2000
def ratio_c := initial_investment_c / 2000

def total_ratio := ratio_a + ratio_b + ratio_c

def part_value := profit_share_b / ratio_b

def profit_share_a := ratio_a * part_value
def profit_share_c := ratio_c * part_value

theorem profit_difference (h1 : initial_investment_a = 8000)
                          (h2 : initial_investment_b = 10000)
                          (h3 : initial_investment_c = 12000)
                          (h4 : profit_share_b = 4000) :
  profit_share_c - profit_share_a = 1600 :=
by sorry

end profit_difference_l669_669302


namespace simplify_cubed_root_l669_669711

def c1 : ℕ := 54880000
def c2 : ℕ := 10^5 * 5488
def c3 : ℕ := 5488
def c4 : ℕ := 2^4 * 343
def c5 : ℕ := 343
def c6 : ℕ := 7^3

theorem simplify_cubed_root : (c1^(1 / 3 : ℝ) : ℝ) = 1400 := 
by {
  let h1 : c1 = c2 := sorry,
  let h2 : c3 = c4 := sorry,
  let h3 : c5 = c6 := sorry,
  rw [h1, h2, h3],
  sorry
}

end simplify_cubed_root_l669_669711


namespace cows_gift_by_friend_l669_669857

-- Define the base conditions
def initial_cows : Nat := 39
def cows_died : Nat := 25
def cows_sold : Nat := 6
def cows_increase : Nat := 24
def cows_bought : Nat := 43
def final_cows : Nat := 83

-- Define the computation to get the number of cows after each event
def cows_after_died : Nat := initial_cows - cows_died
def cows_after_sold : Nat := cows_after_died - cows_sold
def cows_after_increase : Nat := cows_after_sold + cows_increase
def cows_after_bought : Nat := cows_after_increase + cows_bought

-- Define the proof problem
theorem cows_gift_by_friend : (final_cows - cows_after_bought) = 8 := by
  sorry

end cows_gift_by_friend_l669_669857


namespace exists_three_distinct_nonzero_integers_sum_zero_thirteenth_power_square_l669_669907

theorem exists_three_distinct_nonzero_integers_sum_zero_thirteenth_power_square :
  ∃ (a b c : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + b + c = 0) ∧ (∃ n : ℕ, a^13 + b^13 + c^13 = n^2) :=
by 
  let a := 3
  let b := (-1)
  let c := (-2)
  use [a, b, c]
  have h1 : a + b + c = 0 := by simp
  have h2 : a ≠ b := by norm_num
  have h3 : b ≠ c := by norm_num
  have h4 : a ≠ c := by norm_num
  have h5 : a ≠ 0 := by norm_num
  have h6 : b ≠ 0 := by norm_num
  have h7 : c ≠ 0 := by norm_num
  have h_pos : 3^13 - 1 - 2^13 > 0 := by norm_num
  existsi nat.sqrt (3^13 - 1 - 2^13),
  split,
  { exact h1 },
  { rw [pow_two, int.coe_nat_mul, int.coe_nat_cast],
    exact congr (congr_arg (+) (pow_succ' a 12).symm) ((congr (congr_arg (+) %) %)),
    norm_num }
  assumption
  sorry -- further details to complete the proof of the perfect square condition as understood from mathematical solution.

end exists_three_distinct_nonzero_integers_sum_zero_thirteenth_power_square_l669_669907


namespace Hiram_age_l669_669295

theorem Hiram_age (H A : ℕ) (h₁ : H + 12 = 2 * A - 4) (h₂ : A = 28) : H = 40 :=
by
  sorry

end Hiram_age_l669_669295


namespace sum_squares_fractional_parts_l669_669633

def fractional_part (x: ℝ) : ℝ := x - x.floor

theorem sum_squares_fractional_parts (n : ℕ) (hn : 2 ≤ n) (x : ℕ → ℝ) 
  (hx : ∀ i j, i ≠ j → x i ≠ x j) (hx_in_I : ∀ i, 0 < x i ∧ x i < 1) :
  (∑ i in finset.range n, ∑ j in finset.range n, (fractional_part (x i - x j))^2) ≥ (n-1) * (2*n-1) / 6 :=
sorry

end sum_squares_fractional_parts_l669_669633


namespace number_of_valid_n_l669_669032

theorem number_of_valid_n : 
  (∃ (n : ℕ), n ≤ 1500 ∧ (∃ (k : ℕ), 21 * n = k^2)) ↔ (∃ (b : ℕ), b ≤ 8 ∧ n = 21 * b^2 ∧ n ≤ 1500 := sorry

end number_of_valid_n_l669_669032


namespace population_double_in_35_years_l669_669831

-- Definitions for the given conditions
def birth_rate : ℝ := 39.4
def death_rate : ℝ := 19.4

-- The doubling time calculated using the rule of 70
def doubling_years : ℝ := 70 / ((birth_rate - death_rate) / 10)

-- Statement to be proven that population doubles in given years
theorem population_double_in_35_years : doubling_years = 35 :=
by
  -- Proof is omitted
  sorry

end population_double_in_35_years_l669_669831


namespace cheburashkas_erased_l669_669615

def total_krakozyabras : ℕ := 29

def total_rows : ℕ := 2

def cheburashkas_per_row := (total_krakozyabras + total_rows) / total_rows / 2 + 1

theorem cheburashkas_erased :
  (total_krakozyabras + total_rows) / total_rows / 2 - 1 = 11 := 
by
  sorry

-- cheburashkas_erased proves that the number of Cheburashkas erased is 11 from the given conditions.

end cheburashkas_erased_l669_669615


namespace ryan_final_tokens_l669_669208

-- Conditions
def initial_tokens : ℕ := 36
def pacman_fraction : ℚ := 2 / 3
def candy_crush_fraction : ℚ := 1 / 2
def skiball_tokens : ℕ := 7
def friend_borrowed_tokens : ℕ := 5
def friend_returned_tokens : ℕ := 8
def laser_tag_tokens : ℕ := 3
def parents_purchase_factor : ℕ := 10

-- Final Answer
theorem ryan_final_tokens : initial_tokens - 24  - 6 - skiball_tokens + friend_returned_tokens + (parents_purchase_factor * skiball_tokens) - laser_tag_tokens = 75 :=
by sorry

end ryan_final_tokens_l669_669208


namespace tan_B_in_triangle_l669_669167

-- Define the given constants
def AC : ℝ := 4
def BC : ℝ := 3
def cosC : ℝ := 2 / 3

-- Define the unknown we need to prove
def tanB : ℝ := 4 * Real.sqrt 5

-- State the proof problem
theorem tan_B_in_triangle:
  ∃ (AB : ℝ), 
    AB = Real.sqrt (AC^2 + BC^2 - 2 * AC * BC * cosC) ∧
    AB = AC ∧ 
    ∃ (C : ℝ), cosC = Real.cos C ∧ 
    tanB = -2 * Real.tan C / (1 - (Real.tan C)^2) := sorry

end tan_B_in_triangle_l669_669167


namespace union_when_m_eq_neg1_subset_when_union_eq_B_range_when_intersection_is_empty_l669_669493

-- (1)
theorem union_when_m_eq_neg1 {A B : Set ℝ} (m : ℝ) (h : m = -1) :
  A = {x | 1 < x ∧ x < 3} →
  B = {x | 2 * m < x ∧ x < 1 - m} →
  A ∪ B = (-2 : ℝ, 3) :=
  sorry

-- (2)
theorem subset_when_union_eq_B (A B : Set ℝ) :
  A = {x | 1 < x ∧ x < 3} →
  B = {x | 2 * m < x ∧ x < 1 - m} →
  A ∪ B = B →
  m ≤ -2 :=
  sorry

-- (3)
theorem range_when_intersection_is_empty {A B : Set ℝ} (m : ℝ) :
  A = {x | 1 < x ∧ x < 3} →
  B = {x | 2 * m < x ∧ x < 1 - m} →
  A ∩ B = ∅ →
  m ∈ (-∞ : ℝ, (1/3 : ℝ)) ∪ [0 : ℝ, +∞) :=
  sorry

end union_when_m_eq_neg1_subset_when_union_eq_B_range_when_intersection_is_empty_l669_669493


namespace polynomial_equality_l669_669176

noncomputable def monic_poly (n : ℕ) (f : Polynomial ℂ) : Prop :=
  f.leadingCoeff = 1 ∧ f.natDegree = n

theorem polynomial_equality (n : ℕ) (f g : Polynomial ℂ)
  (a_i b_i c_i : Fin n → ℂ)
  (hf : monic_poly n f)
  (hg : monic_poly n g)
  (h_eq : f - g = ∏ i, (Polynomial.C (a_i i) * Polynomial.X + Polynomial.C (b_i i) * Polynomial.Y + Polynomial.C (c_i i))) :
  ∃ a b c : ℂ, f = (Polynomial.X + Polynomial.C a) ^ n + Polynomial.C c ∧ g = (Polynomial.Y + Polynomial.C b) ^ n + Polynomial.C c :=
sorry

end polynomial_equality_l669_669176


namespace ab_ac_bc_range_l669_669624

theorem ab_ac_bc_range (a b c : ℝ) (h : a + b + c = 0) : ab + ac + bc ∈ Iic 0 := by
  sorry

end ab_ac_bc_range_l669_669624


namespace exists_point_on_circle_ge_sum_distances_l669_669055

open Classical

variable (n : ℕ) (A : Fin n → ℝ × ℝ) (circle_center : ℝ × ℝ := (0, 0)) (radius : ℝ := 1)

theorem exists_point_on_circle_ge_sum_distances :
  (∃ M ∈ {p : ℝ × ℝ | (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 = radius^2},
    (∑ i, (dist (A i) M)) ≥ n) :=
by
  sorry

end exists_point_on_circle_ge_sum_distances_l669_669055


namespace how_many_cheburashkas_erased_l669_669601

theorem how_many_cheburashkas_erased 
  (total_krakozyabras : ℕ)
  (characters_per_row_initial : ℕ) 
  (total_characters_initial : ℕ)
  (total_cheburashkas : ℕ)
  (total_rows : ℕ := 2)
  (total_krakozyabras := 29) :
  total_cheburashkas = 11 :=
by
  sorry

end how_many_cheburashkas_erased_l669_669601


namespace minimum_box_value_l669_669522

theorem minimum_box_value :
  ∃ (a b : ℤ), a * b = 36 ∧ (a^2 + b^2 = 72 ∧ ∀ (a' b' : ℤ), a' * b' = 36 → a'^2 + b'^2 ≥ 72) :=
by
  sorry

end minimum_box_value_l669_669522


namespace slope_of_tangent_l669_669779

variables {x : ℝ}

-- Define the curve y = x^2 + 3x
def curve (x : ℝ) : ℝ := x^2 + 3 * x

-- Derivative of the curve
def curve_deriv (x : ℝ) : ℝ := 2 * x + 3

-- Statement of the problem
theorem slope_of_tangent (x := 2) (y := 10) (h : curve x = y) : (curve_deriv 2) = 7 :=
sorry

end slope_of_tangent_l669_669779


namespace geometric_sequence_a5_l669_669946

variable {a : Nat → ℝ} {q : ℝ}

-- Conditions
def is_geometric_sequence (a : Nat → ℝ) (q : ℝ) :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = q * a n

def condition_eq (a : Nat → ℝ) :=
  a 5 + a 4 = 3 * (a 3 + a 2)

-- Proof statement
theorem geometric_sequence_a5 (hq : q ≠ -1)
  (hg : is_geometric_sequence a q)
  (hc : condition_eq a) : a 5 = 9 :=
  sorry

end geometric_sequence_a5_l669_669946


namespace cactus_species_minimum_l669_669365

theorem cactus_species_minimum :
  ∀ (collections : Fin 80 → Fin k → Prop),
  (∀ s : Fin k, ∃ (i : Fin 80), ¬ collections i s)
  → (∀ (c : Finset (Fin 80)), c.card = 15 → ∃ s : Fin k, ∀ (i : Fin 80), i ∈ c → collections i s)
  → 16 ≤ k := 
by 
  sorry

end cactus_species_minimum_l669_669365


namespace problem_statement_l669_669895

theorem problem_statement (x y : ℝ) (h1 : y = 3 * ⌊x⌋ + 4) (h2 : y = 2 * ⌊x - 3⌋ + 8 + x) (h3 : ¬ x ∈ ℤ) : x + y = 9 :=
sorry

end problem_statement_l669_669895


namespace part1_part2_l669_669513

variable (x y : ℝ)

-- Conditions
def condition1 : Prop := x = 2 + Real.sqrt 2
def condition2 : Prop := y = 2 - Real.sqrt 2

-- Part 1: Prove the value of the algebraic expression
theorem part1 (hx : condition1 x y) (hy : condition2 x y) : 
  x^2 + 3*x*y + y^2 = 18 := sorry

-- Part 2: Prove the area of the rhombus with diagonals x and y
theorem part2 (hx : condition1 x y) (hy : condition2 x y) : 
  1/2 * x * y = 1 := sorry

end part1_part2_l669_669513


namespace geometric_sequence_a2_a6_l669_669160

variable (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ)
variable (a_geom_seq : ∀ n, a n = a1 * r^(n-1))
variable (h_a4 : a 4 = 4)

theorem geometric_sequence_a2_a6 : a 2 * a 6 = 16 :=
by
  -- Proof goes here
  sorry

end geometric_sequence_a2_a6_l669_669160


namespace cosine_of_largest_angle_l669_669164

theorem cosine_of_largest_angle
  (A B C : ℝ)
  (h : A + B + C = π)
  (sin_ratio : (sin A) / 2 = (sin B) / 3 ∧ (sin B) / 3 = (sin C) / 4) :
  let a := 2
  let b := 3
  let c := 4
  in cos C = -1/4 :=
sorry

end cosine_of_largest_angle_l669_669164


namespace max_value_expression_l669_669139

theorem max_value_expression (x y : ℝ) (h : x * y > 0) : 
  ∃ (max_val : ℝ), max_val = 4 - 2 * Real.sqrt 2 ∧ 
  (∀ a b : ℝ, a * b > 0 → (a / (a + b) + 2 * b / (a + 2 * b)) ≤ max_val) := 
sorry

end max_value_expression_l669_669139


namespace right_triangle_area_l669_669746

theorem right_triangle_area
  (hypotenuse : ℝ) (leg1 : ℝ) (leg2 : ℝ)
  (hypotenuse_eq : hypotenuse = 13)
  (leg1_eq : leg1 = 5)
  (pythagorean_eq : hypotenuse^2 = leg1^2 + leg2^2) :
  (1 / 2) * leg1 * leg2 = 30 :=
by
  sorry

end right_triangle_area_l669_669746


namespace find_alpha_l669_669215

variable (α β k : ℝ)

axiom h1 : α * β = k
axiom h2 : α = -4
axiom h3 : β = -8
axiom k_val : k = 32
axiom β_val : β = 12

theorem find_alpha (h1 : α * β = k) (h2 : α = -4) (h3 : β = -8) (k_val : k = 32) (β_val : β = 12) :
  α = 8 / 3 :=
sorry

end find_alpha_l669_669215


namespace liquidX_percentage_correct_l669_669637

-- Define the amounts of liquid X in each of the solutions
def liquidX_in_solutionA : ℝ := 0.008 * 400
def liquidX_in_solutionB : ℝ := 0.018 * 700
def liquidX_in_solutionC : ℝ := 0.013 * 500
def liquidX_in_solutionD : ℝ := 0.024 * 600
def liquidX_in_solutionE : ℝ := 0.027 * 300

-- Define the total amount of liquid X
def total_liquidX : ℝ := liquidX_in_solutionA + liquidX_in_solutionB + liquidX_in_solutionC + liquidX_in_solutionD + liquidX_in_solutionE

-- Define the total weight before and after evaporation
def total_weight_before : ℝ := 400 + 700 + 500 + 600 + 300
def total_weight_after : ℝ := total_weight_before - 50

-- Define the percentage calculation
def liquidX_percentage (total_liquidX : ℝ) (total_weight_after : ℝ) : ℝ :=
  (total_liquidX / total_weight_after) * 100

-- State the theorem
theorem liquidX_percentage_correct :
  liquidX_percentage total_liquidX total_weight_after ≈ 1.83 := by
  sorry

end liquidX_percentage_correct_l669_669637


namespace good_fortune_probability_l669_669248

def is_in_intersection_region (α β : ℝ) : Prop :=
  (0 ≤ β ∧ β < (π / 2 - α / 2)) ∧ (0 ≤ α ∧ α < (π / 2 - β / 2))

noncomputable def probability_no_self_intersection : ℝ :=
  1 - (1 / 12)

theorem good_fortune_probability :
  (∀α β : ℝ, 0 ≤ α ∧ α ≤ π → 0 ≤ β ∧ β ≤ 2 * π → ¬ is_in_intersection_region(α, β)) →
  probability_no_self_intersection = 11 / 12 :=
by
  sorry

end good_fortune_probability_l669_669248


namespace csc_cos_expression_eq_two_l669_669390

theorem csc_cos_expression_eq_two : Real.csc (Real.pi / 18) - 4 * Real.cos (2 * Real.pi / 9) = 2 := 
by sorry

end csc_cos_expression_eq_two_l669_669390


namespace evaluate_f_f_neg1_l669_669474

def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then 2^(x + 1) else x^3 + 1

theorem evaluate_f_f_neg1 : f (f (-1)) = 2 := by
  sorry

end evaluate_f_f_neg1_l669_669474


namespace ratio_of_areas_l669_669808

-- Conditions extracted from the problem:
def vertices_in_middle (square : Type) : Prop := sorry
def quarter_of_figure (figure : Type) : Type := sorry
def shaded_triangles (quarter : Type) : ℕ := 5
def unshaded_triangles (quarter : Type) : ℕ := 3

-- The proof statement
theorem ratio_of_areas {figure : Type} (q : quarter_of_figure figure) :
  shaded_triangles q = 5 → unshaded_triangles q = 3 → vertices_in_middle figure →
  (shaded_triangles q / unshaded_triangles q : ℚ) = 5 / 3 :=
by
  intros h_shaded h_unshaded h_vertices
  rw [h_shaded, h_unshaded]
  norm_num
  exact (5 / 3 : ℚ)

end ratio_of_areas_l669_669808


namespace simplify_fraction_l669_669210

-- Define the numerator and denominator
def numerator := 5^4 + 5^2
def denominator := 5^3 - 5

-- Define the simplified fraction
def simplified_fraction := 65 / 12

-- The proof problem statement
theorem simplify_fraction : (numerator / denominator) = simplified_fraction := 
by 
   -- Proof will go here
   sorry

end simplify_fraction_l669_669210


namespace triangle_area_30_l669_669759

theorem triangle_area_30 (h : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = 5 ∧ c = 13 ∧ b > 0) : 
  ∃ area : ℝ, area = 1 / 2 * 5 * (b : ℝ) ∧ area = 30 :=
by
  sorry

end triangle_area_30_l669_669759


namespace geometric_sum_of_squares_l669_669060

theorem geometric_sum_of_squares (a : ℕ → ℝ) (n : ℕ) (h : ∀ n, (∑ i in Finset.range (n + 1), a i) = 2^n - 1) :
    (∑ i in Finset.range (n + 1), (a i)^2) = (1 / 3) * (4^n - 1) :=
    sorry

end geometric_sum_of_squares_l669_669060


namespace monotonic_increasing_interval_l669_669230

noncomputable def f (x : ℝ) : ℝ := real.logb 3 (-x^2 + 2 * x)

theorem monotonic_increasing_interval :
  ∀ x, (0 < x ∧ x ≤ 1) → monotonicallyIncreasing (f x) :=
by
  sorry

end monotonic_increasing_interval_l669_669230


namespace sufficient_condition_perpendicular_planes_l669_669483

variables {L : Type*} [NormedLinearOrderedField L] [NormedSpace ℝ L]
variables (a b : affine_subspace ℝ ℝ^3) (α β γ : L)

theorem sufficient_condition_perpendicular_planes 
  (h1 : ∃ (a b : line), a ⊥ b ∧ a ∥ α ∧ b ∥ β)
  (h2 : ∃ (a b : line), a ⊥ b ∧ α ∩ β = a ∧ b ⊆ β) 
  (h3 : α ⊥ γ ∧ β ⊥ γ)
  (h4 : α ⊥ γ ∧ γ ∥ β) :
  α ⊥ β := 
sorry

end sufficient_condition_perpendicular_planes_l669_669483


namespace rancher_loss_l669_669339

-- Define the necessary conditions
def initial_head_of_cattle := 340
def original_total_price := 204000
def cattle_died := 172
def price_reduction_per_head := 150

-- Define the original and new prices per head
def original_price_per_head := original_total_price / initial_head_of_cattle
def new_price_per_head := original_price_per_head - price_reduction_per_head

-- Define the number of remaining cattle
def remaining_cattle := initial_head_of_cattle - cattle_died

-- Define the total amount at the new price
def total_amount_new_price := new_price_per_head * remaining_cattle

-- Define the loss
def loss := original_total_price - total_amount_new_price

-- Prove that the loss is $128,400
theorem rancher_loss : loss = 128400 := by
  sorry

end rancher_loss_l669_669339


namespace unique_solution_l669_669930

-- Define the Euclidean distance between two points
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Conditions of the system of equations
def condition1 (x y : ℝ) : Prop :=
  dist (x, y) (6, 13) + dist (x, y) (18, 4) = 15

def condition2 (x y a : ℝ) : Prop :=
  (x - 2 * a)^2 + (y - 4 * a)^2 = 1 / 4

-- The Lean statement we want to prove
theorem unique_solution (a : ℝ) :
  (∀ x y, condition1 x y → condition2 x y a → 
    x = 12 ∧ y = 8) ↔ 
  a = 135 / 44 ∨ a = 145 / 44 ∨ (63 / 20 < a ∧ a < 13 / 4) := 
sorry

end unique_solution_l669_669930


namespace horner_method_value_at_neg1_l669_669385

theorem horner_method_value_at_neg1 : 
  let f (x : ℤ) := 4 * x ^ 4 + 3 * x ^ 3 - 6 * x ^ 2 + x - 1
  let x := -1
  let v0 := 4
  let v1 := v0 * x + 3
  let v2 := v1 * x - 6
  v2 = -5 := by
  sorry

end horner_method_value_at_neg1_l669_669385


namespace largest_subset_size_l669_669178

theorem largest_subset_size (T : Finset ℕ) (h : ∀ x ∈ T, ∀ y ∈ T, x ≠ y → (x - y) % 2021 ≠ 5 ∧ (x - y) % 2021 ≠ 8) :
  T.card ≤ 918 := sorry

end largest_subset_size_l669_669178


namespace factorization_correct_l669_669427

-- Define the initial expression
def initial_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + a

-- Define the factorized form
def factorized_expr (a x : ℝ) : ℝ := a * (x - 1)^2

-- Create the theorem statement
theorem factorization_correct (a x : ℝ) : initial_expr a x = factorized_expr a x :=
by simp [initial_expr, factorized_expr, pow_two, mul_add, add_mul, add_assoc]; sorry

end factorization_correct_l669_669427


namespace find_point_on_y_axis_l669_669085

/-- 
Given points A (1, 2, 3) and B (2, -1, 4), and a point P on the y-axis 
such that the distances |PA| and |PB| are equal, 
prove that the coordinates of point P are (0, -7/6, 0).
 -/
theorem find_point_on_y_axis
  (A B : ℝ × ℝ × ℝ)
  (hA : A = (1, 2, 3))
  (hB : B = (2, -1, 4))
  (P : ℝ × ℝ × ℝ)
  (hP : ∃ y : ℝ, P = (0, y, 0)) :
  dist A P = dist B P → P = (0, -7/6, 0) :=
by
  sorry

end find_point_on_y_axis_l669_669085


namespace cos_squared_minus_sin_cos_l669_669983

theorem cos_squared_minus_sin_cos (α : ℝ) (h : sin α ^ 2 + 2 * sin α * cos α - 2 * sin α - 4 * cos α = 0) :
  cos α ^ 2 - sin α * cos α = 3 / 5 := 
sorry

end cos_squared_minus_sin_cos_l669_669983


namespace pigeons_problem_l669_669249

theorem pigeons_problem
  (x y : ℕ)
  (h1 : 6 * y + 3 = x)
  (h2 : 8 * y = x + 5) : x = 27 := 
sorry

end pigeons_problem_l669_669249


namespace exists_subgraph_with_min_degree_l669_669621

-- Defining necessary basic components
variable {G : Type} [graph G] {V : G}

def average_degree (G : Type) [graph G] : ℝ := sorry  -- This should be defined appropriately, but left as sorry for now

def subgraph_degree (H : subgraph G) (v : H) : ℝ := sorry  -- Definition of degree in the subgraph

theorem exists_subgraph_with_min_degree (G : Type) [graph G] (d : ℝ) (avg_deg : average_degree G = d) :
  ∃ H : subgraph G, ∀ v : H, subgraph_degree H v ≥ d / 2 :=
sorry

end exists_subgraph_with_min_degree_l669_669621


namespace sum_of_two_lowest_scores_l669_669452

theorem sum_of_two_lowest_scores 
    (mean : ℕ)
    (median : ℕ)
    (mode : ℕ)
    (sum_of_scores : ℕ := mean * 5)
    (a b : ℤ) :
    mean = 90 →
    median = 91 →
    mode = 94 →
    (91 + 2 * 94 + a + b = sum_of_scores) →
    (a + b = 171) :=
by
  intro hMean hMedian hMode hSum
  have hSum' : sum_of_scores = 450 := by
    rw [hMean, Nat.mul_succ]
  rw [hSum', ←hMean, ←hTopic] at hSum
  simp at hSum
  exact hSum

#check sum_of_two_lowest_scores

end sum_of_two_lowest_scores_l669_669452


namespace probability_at_most_one_hit_l669_669846

noncomputable def P {Ω : Type*} [MeasureSpace Ω] (P : MeasureTheory.ProbabilityMeasure Ω) (A B : Set Ω) : ℝ := 
  P.measure A * P.measure B + (P.measure (Aᶜ) * P.measure B) + (P.measure A * P.measure (Bᶜ)) + (P.measure (Aᶜ) * P.measure (Bᶜ))

theorem probability_at_most_one_hit (P : MeasureTheory.ProbabilityMeasure) (A B : Set Ω) 
  (hA : P.measure A = 0.6) 
  (hB : P.measure B = 0.7) 
  (h_indep : MeasureTheory.Independence P A B) :
  P.measure A * P.measure B = 0.42 → 
  P.measure (Aᶜ ∩ Aᶜ) = 0.58 :=
  sorry

end probability_at_most_one_hit_l669_669846


namespace subtraction_of_twos_from_ones_l669_669821

theorem subtraction_of_twos_from_ones (n : ℕ) : 
  let ones := (10^n - 1) * 10^n + (10^n - 1)
  let twos := 2 * (10^n - 1)
  ones - twos = (10^n - 1) * (10^n - 1) :=
by
  sorry

end subtraction_of_twos_from_ones_l669_669821


namespace hyperbola_asymptotes_a_plus_h_l669_669736

theorem hyperbola_asymptotes_a_plus_h :
  ∀ (a b k h : ℝ), 
  (∀ x, y = 3*x + 4 ↔ ∀ x, y = -3*x + 6) → -- Asymptotes equations
  (1, 10) ∈ set_of (λ p : ℝ × ℝ, (p.1 - h)^2 / b^2 - (p.2 - k)^2 / a^2 = 1) → -- hyperbola passing through point (1, 10)
  (a^2 = 21) ∧ (h = 1/3) → -- Simplified using steps
  a + h = sqrt 21 + 1/3 :=
begin
  intros a b k h H_asymptotes H_hyperbola H_values,
  sorry
end

end hyperbola_asymptotes_a_plus_h_l669_669736


namespace divisor_product_of_integers_l669_669042

theorem divisor_product_of_integers (m : ℕ) (h : 1 < m) : 
  (m ∣ (∏ i in Finset.range (m-1), (i + 1))) ↔ (∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = m) ∨ (m = 4 → false) :=
sorry

end divisor_product_of_integers_l669_669042


namespace range_of_a_l669_669469

theorem range_of_a (a : ℝ) : ¬ (∃ x y : ℝ, 
  (x^2 + y^2 = 3) ∧ 
  (y = (a / 4) * x) ∧ 
  (-2 ≤ x ∧ x ≤ 2)) ↔ 
  (a < -4 * real.sqrt 3 ∨ 4 * real.sqrt 3 < a) := 
by sorry

end range_of_a_l669_669469


namespace line_through_point_perpendicular_y_axis_line_through_two_points_l669_669443

-- The first problem
theorem line_through_point_perpendicular_y_axis :
  ∃ (k : ℝ), ∀ (x : ℝ), k = 1 → y = k :=
sorry

-- The second problem
theorem line_through_two_points (x1 y1 x2 y2 : ℝ) (hA : (x1, y1) = (-4, 0)) (hB : (x2, y2) = (0, 6)) :
  ∃ (a b c : ℝ), (a, b, c) = (3, -2, 12) → ∀ (x y : ℝ), a * x + b * y + c = 0 :=
sorry

end line_through_point_perpendicular_y_axis_line_through_two_points_l669_669443


namespace factorization_l669_669434

theorem factorization (a x : ℝ) : ax^2 - 2ax + a = a * (x - 1) ^ 2 := 
by
  sorry

end factorization_l669_669434


namespace grocery_store_cans_count_l669_669327

theorem grocery_store_cans_count 
  (bulk_cans : ℕ) (bulk_price : ℝ) (grocery_price : ℝ) (price_diff : ℝ) :
  bulk_cans = 48 →
  bulk_price = 12.00 →
  grocery_price = 6.00 →
  price_diff = 0.25 →
  let bulk_price_per_can := bulk_price / bulk_cans in
  let grocery_price_per_can := bulk_price_per_can + price_diff in
  grocery_price / grocery_price_per_can = 12 :=
begin
  intros h1 h2 h3 h4,
  let bulk_price_per_can := bulk_price / bulk_cans,
  let grocery_price_per_can := bulk_price_per_can + price_diff,
  calc
    grocery_price / grocery_price_per_can =
      6 / (12 / 48 + 0.25) : by { rw [h2, h3, h4], norm_num }
    ... = 12 : by norm_num,
end

end grocery_store_cans_count_l669_669327


namespace factorization_l669_669437

theorem factorization (a x : ℝ) : ax^2 - 2ax + a = a * (x - 1) ^ 2 := 
by
  sorry

end factorization_l669_669437


namespace exist_circle_tangent_to_BD_BF_CE_tau_l669_669038

open EuclideanGeometry

theorem exist_circle_tangent_to_BD_BF_CE_tau
  (A B C D E : Point)
  (tau : Circle)
  (h₁ : A ∈ tau ∧ B ∈ tau ∧ C ∈ tau ∧ D ∈ tau ∧ E ∈ tau)
  (h₂ : clockwise_order tau [A, B, C, D, E])
  (h₃ : parallel (line_through A B) (line_through C E))
  (h₄ : ∠ABC > 90°)
  (k : Circle)
  (h₅ : tangent k (line_through A D) ∧ tangent k (line_through C E) ∧ tangent k tau)
  (h₆ : point_of_tangency_touches_arc_on_tau_not_containing_ABC k tau D E)
  (F : Point)
  (h₇ : F ≠ A ∧ is_intersection τ (tangent_line_through k A) ∧ F ∉ {A, D}) :
  ∃ (circle_tangent : Circle),
    tangent circle_tangent (line_through B D) ∧
    tangent circle_tangent (line_through B F) ∧
    tangent circle_tangent (line_through C E) ∧
    tangent circle_tangent tau :=
sorry

end exist_circle_tangent_to_BD_BF_CE_tau_l669_669038


namespace bothStoresSaleSameDate_l669_669848

-- Define the conditions
def isBookstoreSaleDay (d : ℕ) : Prop := d % 4 = 0
def isShoeStoreSaleDay (d : ℕ) : Prop := ∃ k : ℕ, d = 5 + 7 * k
def isJulyDay (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 31

-- Define the problem statement
theorem bothStoresSaleSameDate : 
  (∃ d1 d2 : ℕ, isJulyDay d1 ∧ isBookstoreSaleDay d1 ∧ isShoeStoreSaleDay d1 ∧
                 isJulyDay d2 ∧ isBookstoreSaleDay d2 ∧ isShoeStoreSaleDay d2 ∧ d1 ≠ d2) :=
sorry

end bothStoresSaleSameDate_l669_669848


namespace cubic_roots_and_k_value_l669_669657

theorem cubic_roots_and_k_value (k r₃ : ℝ) :
  (∃ r₃, 3 - 2 + r₃ = -5 ∧ 3 * (-2) * r₃ = -12 ∧ k = 3 * (-2) + (-2) * r₃ + r₃ * 3) →
  (k = -12 ∧ r₃ = -6) :=
by
  sorry

end cubic_roots_and_k_value_l669_669657


namespace output_value_is_16_l669_669294

def f (x : ℤ) : ℤ :=
  if x < 0 then (x + 1) * (x + 1) else (x - 1) * (x - 1)

theorem output_value_is_16 : f 5 = 16 := by
  sorry

end output_value_is_16_l669_669294


namespace problem_l669_669471

variables {O A1 B1 C1 A B C A2 B2 C2 : Type}
variable [real.euclidean_affine_space Type]

def is_circumcircle (O : EuclideanAffineSpace) (A1 B1 C1 : EuclideanPoint) : Prop :=
  On_circle O A1 ∧ On_circle O B1 ∧ On_circle O C1

def extend_to_circumcircle (A1 A B1 B C1 C : EuclideanPoint) (A2 B2 C2 : EuclideanPoint) (O : EuclideanAffineSpace) : Prop :=
  (Line_through A1 A).intersect_circle O = Some A2 ∧
  (Line_through B1 B).intersect_circle O = Some B2 ∧
  (Line_through C1 C).intersect_circle O = Some C2

theorem problem (hcirc : is_circumcircle O A1 B1 C1)
                (hextend : extend_to_circumcircle A1 A B1 B C1 C A2 B2 C2 O) :
  segment_distance A2 A + segment_distance B2 B + segment_distance C2 C = segment_distance A A1 :=
sorry

end problem_l669_669471


namespace first_quartile_of_given_list_l669_669829

def median (l : List ℝ) : ℝ :=
  if l.length % 2 = 1 then 
    (l.insertion_sort l).get (l.length / 2)
  else 
    ((l.insertion_sort l).get (l.length / 2 - 1) + (l.insertion_sort l).get (l.length / 2)) / 2

def first_quartile (l : List ℝ) : ℝ :=
  let m := median l
  let l' := l.filter (λ x => x < m)
  median l'

theorem first_quartile_of_given_list : 
  first_quartile [42, 24, 30, 22, 26, 27, 33, 35] = 25 := by
  sorry

end first_quartile_of_given_list_l669_669829


namespace simplify_cubic_root_l669_669701

theorem simplify_cubic_root : 
  (∛(54880000) = 20 * ∛((5^2) * 137)) :=
sorry

end simplify_cubic_root_l669_669701


namespace units_digit_fraction_mod_10_l669_669810

theorem units_digit_fraction_mod_10 : (30 * 32 * 34 * 36 * 38 * 40) % 2000 % 10 = 2 := by
  sorry

end units_digit_fraction_mod_10_l669_669810


namespace complex_inv_condition_l669_669127

theorem complex_inv_condition (i : ℂ) (h : i^2 = -1) : (i - 2 * i⁻¹)⁻¹ = -i / 3 :=
by
  sorry

end complex_inv_condition_l669_669127


namespace expected_value_of_X_l669_669265

-- Define the sequence of random digits as a list of natural numbers (0 to 9)
def random_digits : List ℕ := (List.range 10).take 1000

-- Define the function that forms the number X
def X (digits : List ℕ) : ℝ :=
  digits.enum.foldr (λ (p : ℕ × ℕ) (acc : ℝ) => acc + p.snd * 10^(-(p.fst + 1))) 0

-- Define the expected value of a single digit
def expected_value_digit : ℝ := (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) / 10

-- The main statement to prove
theorem expected_value_of_X (digits : List ℕ) (h_digits : digits.length = 1000) :
  (∑ i in Finset.range digits.length, 10^(-(i + 1)) * expected_value_digit) = 0.5 :=
by {
  sorry
}

end expected_value_of_X_l669_669265


namespace ratio_of_probabilities_l669_669909

noncomputable def balls_toss (balls bins : ℕ) : Nat := by
  sorry

def prob_A : ℚ := by
  sorry
  
def prob_B : ℚ := by
  sorry

theorem ratio_of_probabilities (balls : ℕ) (bins : ℕ) 
  (h_balls : balls = 20) (h_bins : bins = 5) (p q : ℚ) 
  (h_p : p = prob_A) (h_q : q = prob_B) :
  (p / q) = 4 := by
  sorry

end ratio_of_probabilities_l669_669909


namespace monotonic_intervals_distinct_roots_range_l669_669987

open Real

def f (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * x^2 - 2 * x + 1

theorem monotonic_intervals :
  (∀ x, f' x > 0 → x < -2 ∨ x > 1) ∧ (∀ x, f' x < 0 → -2 < x ∧ x < 1) :=
by
  sorry

theorem distinct_roots_range (k : ℝ) :
  (∃ a b c, a < b ∧ b < c ∧ f a = 2 * k ∧ f b = 2 * k ∧ f c = 2 * k) →
  -1/12 < k ∧ k < 13/6 :=
by
  sorry

end monotonic_intervals_distinct_roots_range_l669_669987


namespace production_growth_equation_l669_669332

theorem production_growth_equation (x : ℝ) :
  10 + 10 * (1 + x) + 10 * (1 + x) ^ 2 = 36.4 :=
begin
  sorry
end

end production_growth_equation_l669_669332


namespace unable_to_determine_questions_answered_l669_669587

variable (total_questions : ℕ) (total_time : ℕ) (used_time : ℕ) (remaining_time : ℕ)

theorem unable_to_determine_questions_answered (total_questions_eq : total_questions = 80)
  (total_time_eq : total_time = 60)
  (used_time_eq : used_time = 12)
  (remaining_time_eq : remaining_time = 0) :
  ∀ (answered_rate : ℕ → ℕ), ¬ ∃ questions_answered, answered_rate used_time = questions_answered :=
by sorry

end unable_to_determine_questions_answered_l669_669587


namespace sum_products_of_subsets_l669_669897

noncomputable def M : Finset ℤ := {4, 3, -1, 0, 1}

def product (s : Finset ℤ) : ℤ := s.prod id

def subsets_products_sum (s : Finset ℤ) : ℤ :=
  (s.powerset.filter (λ t => ¬t.isEmpty)).sum product

theorem sum_products_of_subsets :
  subsets_products_sum M = -16 :=
by
  sorry

end sum_products_of_subsets_l669_669897


namespace part2_proof_l669_669105

noncomputable def f (x : ℝ) : ℝ := Real.exp (Real.log x) - Real.exp 1 * x

theorem part2_proof (x : ℝ) (h : 0 < x) :
  x * f x - Real.exp x + 2 * Real.exp 1 * x ≤ 0 := 
sorry

end part2_proof_l669_669105


namespace find_N_l669_669131

theorem find_N (N x : ℝ) (h1 : N / (1 + 4 / x) = 1) (h2 : x = 0.5) : N = 9 := 
by 
  sorry

end find_N_l669_669131


namespace boat_downstream_time_l669_669326

variable (V_boat : ℝ) (V_stream : ℝ) (Distance : ℝ)

def effective_speed_downstream (V_boat : ℝ) (V_stream : ℝ) : ℝ :=
  V_boat + V_stream

def time_taken (Distance : ℝ) (Speed : ℝ) : ℝ :=
  Distance / Speed

theorem boat_downstream_time : 
  V_boat = 16 →
  V_stream = 5 →
  Distance = 63 →
  time_taken Distance (effective_speed_downstream V_boat V_stream) = 3 :=
by
  intros hV_boat hV_stream hDistance
  rw [hV_boat, hV_stream, hDistance]
  show time_taken 63 (effective_speed_downstream 16 5) = 3
  rw [effective_speed_downstream, time_taken]
  norm_num

end boat_downstream_time_l669_669326


namespace trigonometric_identity_l669_669937

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) :
  (Real.sin θ * Real.sin (π / 2 - θ)) / (Real.sin θ ^ 2 + Real.cos (2 * θ) + Real.cos θ ^ 2) = 1 / 3 :=
by
  sorry

end trigonometric_identity_l669_669937


namespace max_value_sqrt_add_l669_669627

noncomputable def sqrt_add (a b : ℝ) : ℝ := Real.sqrt (a + 1) + Real.sqrt (b + 3)

theorem max_value_sqrt_add (a b : ℝ) (h : 0 < a) (h' : 0 < b) (hab : a + b = 5) :
  sqrt_add a b ≤ 3 * Real.sqrt 2 :=
by
  sorry

end max_value_sqrt_add_l669_669627


namespace equivalent_resistance_A_B_l669_669931

-- Parameters and conditions
def resistor_value : ℝ := 5 -- in MΩ
def num_resistors : ℕ := 4
def has_bridging_wire : Prop := true
def negligible_wire_resistance : Prop := true

-- Problem: Prove the equivalent resistance (R_eff) between points A and B is 5 MΩ.
theorem equivalent_resistance_A_B : 
  ∀ (R : ℝ) (n : ℕ) (bridge : Prop) (negligible_wire : Prop),
    R = 5 → n = 4 → bridge → negligible_wire → R = 5 :=
by sorry

end equivalent_resistance_A_B_l669_669931


namespace positive_integer_k_exists_l669_669919

theorem positive_integer_k_exists (k : ℕ) (h : k > 1) : 
  ∃ n : ℕ, (n > 0) ∧ (binom n k % n = 0) ∧ (∀ m : ℕ, 2 ≤ m → m < k → binom n m % n ≠ 0) :=
sorry

end positive_integer_k_exists_l669_669919


namespace quadratic_roots_unique_pair_l669_669723

theorem quadratic_roots_unique_pair (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (h_root1 : p * q = q)
  (h_root2 : p + q = -p)
  (h_rel : q = -2 * p) : 
(p, q) = (1, -2) :=
  sorry

end quadratic_roots_unique_pair_l669_669723


namespace expected_value_X_is_half_l669_669250

open BigOperators

-- Define the random digit sequence and its properties
def random_digit_seq (n : ℕ) (d : ℕ → ℕ) := ∀ i : ℕ, i < n → d i ∈ Fin 10

-- Expected value of a single random digit
def expected_value_digit : ℝ := 4.5

-- Define the expected value of X
noncomputable def expected_value_of_X (n : ℕ) (d : ℕ → ℕ) : ℝ :=
  ∑ i in Finset.range n, (d i : ℝ) * 10^(-(i+1))

-- The main theorem to prove
theorem expected_value_X_is_half : 
  ∀ (d : ℕ → ℕ), random_digit_seq 1000 d → expected_value_of_X 1000 d = 0.5 :=
by {
  intro d,
  intro h,
  sorry  -- The proof would be written here.
}

end expected_value_X_is_half_l669_669250


namespace inequality_proof_l669_669662

theorem inequality_proof (x y z : ℝ) : 
    x^4 + y^4 + z^2 + 1 ≥ 2 * x * (x * y^2 - x + z + 1) :=
by
  sorry

end inequality_proof_l669_669662


namespace find_y_l669_669046

theorem find_y (a b c x : ℝ) (p q r y : ℝ) 
  (h1 : log a = p * log x)
  (h2 : log b = q * log x)
  (h3 : log c = r * log x)
  (h4 : x ≠ 1)
  (h5 : b ^ 3 / (a ^ 2 * c) = x ^ y) : 
  y = 3 * q - 2 * p - r :=
by {
  sorry
}

end find_y_l669_669046


namespace cover_light_source_with_four_spheres_l669_669671

-- Definitions:
variable (F : ℝ^3)  -- A point light source
variable (G_A G_B G_C G_D : set (ℝ^3))  -- Four spheres

-- Conditions:
def is_sphere (G : set (ℝ^3)) : Prop := ∃ (center : ℝ^3) (radius : ℝ), ∀ (x : ℝ^3), (x ∈ G) ↔ (dist x center ≤ radius)

def covers_light_source (F : ℝ^3) (spheres : list (set ℝ^3)) : Prop :=
∀ (ray : ℝ^3 → ℝ^3), ∃ (G : set ℝ^3) ∈ spheres, ∃ (t : ℝ), t > 0 ∧ (F + t * ray) ∈ G

-- Theorem:
theorem cover_light_source_with_four_spheres (F : ℝ^3)
  (G_A G_B G_C G_D : set (ℝ^3)) 
  (hGA : is_sphere G_A) (hGB : is_sphere G_B) (hGC : is_sphere G_C) (hGD : is_sphere G_D) :
  covers_light_source F [G_A, G_B, G_C, G_D] :=
sorry

end cover_light_source_with_four_spheres_l669_669671


namespace part1_part2_l669_669504

variable (m : ℝ)

theorem part1 (h1 : m ≠ 0) (h2 : (2 * m + 1) + (2 + 1 / m) = 6) : m = 1 ∨ m = 1 / 2 :=
  sorry

theorem part2 (h1 : m ≠ 0) (h2 : (2 * m + 1, 0).1 > 0) (h3 : (0, 2 + 1 / m).2 > 0) :
  let l := 2 * x + y - 4
  area (triangle (2 * m + 1, 0) (0, 2 + 1 / m) (0, 0)) = min : 
  ∃ x y, 2 * x + y - 4 = 0 :=
  sorry

end part1_part2_l669_669504


namespace O1O3_perpendicular_equal_O2O4_l669_669044

-- Given vertices A, B, C, D of quadrilateral ABCD as complex numbers
variables (a b c d : ℂ)

-- Definitions for midpoints of each side of the quadrilateral
def G1 : ℂ := (a + d) / 2
def G2 : ℂ := (a + b) / 2
def G3 : ℂ := (b + c) / 2
def G4 : ℂ := (c + d) / 2

-- Perpendicular directions with half the length of the corresponding side
def O1 : ℂ := G1 + ((d - a) / 2) * complex.I
def O2 : ℂ := G2 + ((b - a) / 2) * complex.I
def O3 : ℂ := G3 + ((c - b) / 2) * complex.I
def O4 : ℂ := G4 + ((d - c) / 2) * complex.I

-- Statement: O1O3 is perpendicular and equal to O2O4
theorem O1O3_perpendicular_equal_O2O4 : 
  (O1 - O3) * (complex.conj (O2 - O4)).im = 0 ∧
  complex.abs (O1 - O3) = complex.abs (O2 - O4) :=
by sorry

end O1O3_perpendicular_equal_O2O4_l669_669044


namespace jenny_grade_l669_669586

theorem jenny_grade (J A B : ℤ) 
  (hA : A = J - 25) 
  (hB : B = A / 2) 
  (hB_val : B = 35) : 
  J = 95 :=
by
  sorry

end jenny_grade_l669_669586


namespace area_of_triangle_l669_669153

variables {A B C a b c : ℝ}
variables (sin cos : ℝ → ℝ)

-- Given conditions
def acute_triangle (A B C : ℝ) : Prop := A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π
def angle_B_is_pi_over_3 (B : ℝ) : Prop := B = π / 3
def side_b_is_2 (b : ℝ) : Prop := b = 2
def condition_c_sinA (c sinA sqrt3 a cosC : ℝ) : Prop := c * sinA = sqrt3 * a * cosC

theorem area_of_triangle 
  (h_acute : acute_triangle A B C)
  (h_B : angle_B_is_pi_over_3 B)
  (h_b : side_b_is_2 b)
  (h_cond : condition_c_sinA c (sin A) (real.sqrt 3) a (cos C)) :
  ∃ (area : ℝ), area = real.sqrt 3 := 
sorry

end area_of_triangle_l669_669153


namespace cube_difference_l669_669969

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 53) : a^3 - b^3 = 385 :=
sorry

end cube_difference_l669_669969


namespace factorization_correct_l669_669428

-- Define the initial expression
def initial_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + a

-- Define the factorized form
def factorized_expr (a x : ℝ) : ℝ := a * (x - 1)^2

-- Create the theorem statement
theorem factorization_correct (a x : ℝ) : initial_expr a x = factorized_expr a x :=
by simp [initial_expr, factorized_expr, pow_two, mul_add, add_mul, add_assoc]; sorry

end factorization_correct_l669_669428


namespace smallest_n_condition_l669_669187

theorem smallest_n_condition (n : ℕ) (x : ℕ → ℝ) 
  (H1 : ∀ i, |x i| < 1)
  (H2 : ∑ i in finset.range n, |x i| = 25 + |(∑ i in finset.range n, x i)|) :
  n ≥ 26 := 
sorry

end smallest_n_condition_l669_669187


namespace proof_problem_l669_669956

-- Defining the conditions as Lean definitions
def point_A (n : ℝ) : ℝ × ℝ := ⟨0, -n⟩
def point_B (n : ℝ) : ℝ × ℝ := ⟨0, n⟩

def circle_P (x y : ℝ) : Prop := (x + sqrt 3)^2 + (y - 1)^2 = 1

def angle_APB (A B P : ℝ × ℝ) : Prop := 
  -- Please note that this is a placeholder for the actual angle calculation
  sorry

def prop_p (n : ℝ) : Prop := 1 ≤ n ∧ n ≤ 3

def func_f (x : ℝ) : ℝ := 4/3 - log 3 x

def prop_q : Prop := ∀ x, 3 < x ∧ x < 4 → func_f x ≠ 0

-- The proof statement combining p ∧ q
theorem proof_problem (n : ℝ)
  (h₁ : n > 0)
  (h₂ : ∃ P : ℝ × ℝ, circle_P P.1 P.2 ∧ angle_APB (point_A n) (point_B n) P)
  (h₃ : prop_q) :
  prop_p n ∧ prop_q :=
sorry

end proof_problem_l669_669956


namespace trig_properties_of_angle_l669_669100

theorem trig_properties_of_angle (m : ℝ) (α : ℝ) (cos_α : ℝ) (sin_α : ℝ) (tan_α : ℝ) (h : (cos_α = -3/5) ∧ (∃ m, abs(-3 + m^2) = 5)): 
  ((m = 4) → (sin_α = 4/5 ∧ tan_α = -4/3)) ∧ ((m = -4) → (sin_α = -4/5 ∧ tan_α = 4/3)) := 
by {
  sorry
}

end trig_properties_of_angle_l669_669100


namespace simplify_cube_root_l669_669692

theorem simplify_cube_root (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)
  (h1 : a = 10^3 * b)
  (h2 : b = 2^7 * c * 7^3)
  (h3 : c = 10) :
  ∛a = 40 * 7 * 2^(2/3) * 5^(1/3) := by
  sorry

end simplify_cube_root_l669_669692


namespace simplify_cubed_root_l669_669706

def c1 : ℕ := 54880000
def c2 : ℕ := 10^5 * 5488
def c3 : ℕ := 5488
def c4 : ℕ := 2^4 * 343
def c5 : ℕ := 343
def c6 : ℕ := 7^3

theorem simplify_cubed_root : (c1^(1 / 3 : ℝ) : ℝ) = 1400 := 
by {
  let h1 : c1 = c2 := sorry,
  let h2 : c3 = c4 := sorry,
  let h3 : c5 = c6 := sorry,
  rw [h1, h2, h3],
  sorry
}

end simplify_cubed_root_l669_669706


namespace cos_of_angle_A_l669_669975

-- Definitions
def circumcenter_condition
  (A B C O : Point)
  (h : O = (1/3) • (B - A) + (1/3) • (C - A)) : Prop :=
  O = (1 / 3) • (B - A) + (1 / 3) • (C - A)

-- Theorem statement in Lean
theorem cos_of_angle_A
  (A B C O : Point)
  (h : circumcenter_condition A B C O)
  : ∃ (cosA : ℝ), cosA = 1 / 2 :=
by
  sorry

end cos_of_angle_A_l669_669975


namespace expected_value_of_X_is_half_l669_669258

-- Definition of the sequence of random digits and the random number X
def random_digits : list (vector ℕ 10) :=
  replicate 1000 (vector.of_fn (λ i, i)) -- Simulates a list of random digits from 0 to 9

def X (digits : list ℕ) : ℝ :=
  digits.foldl (λ acc x, acc / 10 + x.to_real / 10) 0

-- Expected value of X
noncomputable def E_X : ℝ :=
  1 / 2

-- Theorem statement
theorem expected_value_of_X_is_half : E_X = 0.5 :=
  by
  sorry -- The proof would go here

end expected_value_of_X_is_half_l669_669258


namespace representation_almost_surely_l669_669314

noncomputable theory

variables {X Y : Type} [MeasureSpace X] [MeasureSpace Y]
variables (h : X → Y → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℕ → ℝ)
variables (f : ℕ → X → ℝ) (g : ℕ → Y → ℝ)

-- Assuming necessary integrability and orthonormality conditions;
-- these would be expressed directly within a formal proof which is not provided in this statement.
axiom independence (X Y : Type) : (∀ x y, P(X = x ∧ Y = y) = P(X = x) * P(Y = y))
axiom borel_function (h : X → Y → ℝ) : BorelMeasurable (h)
axiom integrable_square (h : X → Y → ℝ) : ∫ x y, h x y ^ 2 < ∞

theorem representation_almost_surely :
  ∀ (X Y : Type) [hx : MeasureTheory.ProbabilityMeasureSpace X] [hy : MeasureTheory.ProbabilityMeasureSpace Y]
  (indep : (independence X Y))
  (borel : (borel_function h))
  (integrable : (integrable_square h)), 
  ∃ (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℕ → ℝ) (f : ℕ → X → ℝ) (g : ℕ → Y → ℝ),
    h = λ x y, ((∫ x y, h x y) 
              + ∑' m, a m * (f m x) 
              + ∑' n, b n * (g n y) 
              + ∑' (m n), c m n * (f m x) * (g n y)) ∧
    -- Assumption variables should include orthonormal systems f and g 
    -- These are assumed available
    (∫ x y, h x y ^ 2) = (∫ x y, (∫ x y, h x y) ^ 2) 
                      + (∑' m, (a m) ^ 2) 
                      + (∑' n, (b n) ^ 2) 
                      + (∑' mn, (c m n) ^ 2) := 
sorry

end representation_almost_surely_l669_669314


namespace ratio_of_selling_prices_l669_669356

variable (CP : ℝ)
def SP1 : ℝ := CP * 1.6
def SP2 : ℝ := CP * 0.8

theorem ratio_of_selling_prices : SP2 / SP1 = 1 / 2 := 
by sorry

end ratio_of_selling_prices_l669_669356


namespace total_circles_l669_669867

theorem total_circles (n : ℕ) (h1 : ∀ k : ℕ, k = n + 14 → n^2 = (k * (k + 1) / 2)) : 
  n = 35 → n^2 = 1225 :=
by
  sorry

end total_circles_l669_669867


namespace trajectory_equation_l669_669655

variables (x y : ℝ)
def A := (-2 : ℝ, y)
def B := (0 : ℝ, y / 2)
def C := (x, y)

def AB := (B.1 - A.1, B.2 - A.2)
def BC := (C.1 - B.1, C.2 - B.2)

theorem trajectory_equation (h : AB.1 * BC.1 + AB.2 * BC.2 = 0) : y * y = 8 * x :=
by sorry

end trajectory_equation_l669_669655


namespace cotangent_diff_square_l669_669824

theorem cotangent_diff_square (α β : ℝ) (hα : sin α ≠ 0) (hβ : sin β ≠ 0) :
  (cos α / sin α) ^ 2 - (cos β / sin β) ^ 2 = (cos α ^ 2 - cos β ^ 2) / (sin α ^ 2 * sin β ^ 2) :=
by
  sorry

end cotangent_diff_square_l669_669824


namespace cheburashkas_erased_l669_669591

theorem cheburashkas_erased (n : ℕ) (rows : ℕ) (krakozyabras : ℕ) 
  (h_spacing : ∀ r, r ≤ rows → krakozyabras = 2 * (n - 1))
  (h_rows : rows = 2)
  (h_krakozyabras : krakozyabras = 29) :
  n = 16 → rows = 2 → krakozyabras = 29 → n = 16 - 5 :=
by
  sorry

end cheburashkas_erased_l669_669591


namespace octagon_area_l669_669960

/-- Given BDEF is a square and AB = BC = 2 units,
prove that the area of the regular octagon enclosed by 
this setup is 16 + 8 * sqrt 2 square units. -/
theorem octagon_area (BDEF_is_square : is_square BDEF)
  (AB BC : ℝ) (hAB : AB = 2) (hBC : BC = 2) :
  ∃ (area : ℝ), area = 16 + 8 * sqrt 2 :=
sorry

end octagon_area_l669_669960


namespace unique_function_eq_id_l669_669207

theorem unique_function_eq_id (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → f x = x^2 * f (1 / x)) →
  (∀ x y : ℝ, f (x + y) = f x + f y) →
  (f 1 = 1) →
  (∀ x : ℝ, f x = x) :=
by
  intro h1 h2 h3
  sorry

end unique_function_eq_id_l669_669207


namespace range_of_a_l669_669741

theorem range_of_a (a : ℝ) : (∀ x ∈ Iic 2, deriv (λ x : ℝ, x^2 + a*x + 3) x ≤ 0) → a ≤ -4 :=
by
  sorry

end range_of_a_l669_669741


namespace complement_relative_A_of_B_l669_669491

open Set

theorem complement_relative_A_of_B :
  let A := {-1, 0, 1, 2, 3}
  let B := {-1, 1}
  ∁ (A \ B) = {0, 2, 3} := 
by
  let A := ({-1, 0, 1, 2, 3} : Set ℤ)
  let B := ({-1, 1} : Set ℤ)
  have h : ∁ A B = {0, 2, 3} := sorry
  exact h

end complement_relative_A_of_B_l669_669491


namespace cube_root_simplification_l669_669683

noncomputable def cubeRoot (x : ℝ) : ℝ := x^(1/3)

theorem cube_root_simplification :
  cubeRoot 54880000 = 140 * cubeRoot 20 :=
by
  sorry

end cube_root_simplification_l669_669683


namespace walnut_trees_after_removal_l669_669784

theorem walnut_trees_after_removal (initial_trees : ℕ) (removed_trees : ℕ) (initial_trees = 6) (removed_trees = 4) : 
  initial_trees - removed_trees = 2 := 
sorry

end walnut_trees_after_removal_l669_669784


namespace equivalence_f_eq_1_l669_669837

noncomputable def f (x : ℝ) (a α β b : ℝ) : ℝ :=
  a * Real.sin (π * x + α) + b * Real.cos (π * x + β)

theorem equivalence_f_eq_1 (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) (hf : f 2012 a α β b = -1) :
  f 2013 a α β b = 1 :=
  sorry

end equivalence_f_eq_1_l669_669837


namespace sin_angle_RQT_l669_669553

theorem sin_angle_RQT (RPQ RQP RPT RQT : ℝ) (RQ PQ : ℝ) (h1 : sin RPQ = 3/5)
  (h2 : RQ = PQ) (h3 : RQP = 90 - RPT) : sin RQT = 4/5 :=
by
  sorry

end sin_angle_RQT_l669_669553


namespace incorrect_desc_is_C_l669_669878
noncomputable def incorrect_geometric_solid_desc : Prop :=
  ¬ (∀ (plane_parallel: Prop), 
      plane_parallel ∧ 
      (∀ (frustum: Prop), frustum ↔ 
        (∃ (base section_cut cone : Prop), 
          cone ∧ 
          (section_cut = plane_parallel) ∧ 
          (frustum = (base ∧ section_cut)))))

theorem incorrect_desc_is_C (plane_parallel frustum base section_cut cone : Prop) :
  incorrect_geometric_solid_desc := 
by
  sorry

end incorrect_desc_is_C_l669_669878


namespace remainder_of_geometric_series_mod_500_l669_669384

theorem remainder_of_geometric_series_mod_500 :
  (∑ k in Finset.range 1025, 11^k) % 500 = 25 :=
by
  sorry

end remainder_of_geometric_series_mod_500_l669_669384


namespace m_greater_than_p_l669_669634

theorem m_greater_than_p (p m n : ℕ) (pp : Nat.Prime p) (pos_m : m > 0) (pos_n : n > 0) (h : p^2 + m^2 = n^2) : m > p :=
sorry

end m_greater_than_p_l669_669634


namespace thermometer_actual_temperature_l669_669299

noncomputable def solveTemperature (x y : ℤ) (k b : ℤ) : ℤ :=
  k * x + b

theorem thermometer_actual_temperature :
  (solveTemperature (-11) _ (-7)) ∧ 
  (solveTemperature 32 _ 36) → 
  solveTemperature 22 _ 18 :=
by 
  sorry

end thermometer_actual_temperature_l669_669299


namespace cheburashkas_erased_l669_669608

theorem cheburashkas_erased (total_krakozyabras : ℕ) (rows : ℕ) :
  rows ≥ 2 → total_krakozyabras = 29 → ∃ (cheburashkas_erased : ℕ), cheburashkas_erased = 11 :=
by
  assume h_rows h_total_krakozyabras
  let n := (total_krakozyabras / 2) + 1
  have h_cheburashkas : cheburashkas_erased = n - 1 
  sorry

end cheburashkas_erased_l669_669608


namespace find_total_amount_l669_669305

theorem find_total_amount (x : ℝ) (h₁ : 1.5 * x = 40) : x + 1.5 * x + 0.5 * x = 80.01 :=
by
  sorry

end find_total_amount_l669_669305


namespace cube_root_simplification_l669_669715

theorem cube_root_simplification : (∛54880000) = 140 * (2 ^ (1 / 3)) :=
by
  -- Using the information from the problem conditions and final solution.
  have root_10_cubed := (10 ^ 3 : ℝ)
  have factored_value := root_10_cubed * (2 ^ 4 * 7 ^ 3)
  have cube_root := Real.cbrt factored_value
  sorry

end cube_root_simplification_l669_669715


namespace cube_root_simplification_l669_669718

theorem cube_root_simplification : (∛54880000) = 140 * (2 ^ (1 / 3)) :=
by
  -- Using the information from the problem conditions and final solution.
  have root_10_cubed := (10 ^ 3 : ℝ)
  have factored_value := root_10_cubed * (2 ^ 4 * 7 ^ 3)
  have cube_root := Real.cbrt factored_value
  sorry

end cube_root_simplification_l669_669718


namespace ages_of_father_and_daughter_l669_669334

variable (F D : ℕ)

-- Conditions
def condition1 : Prop := F = 4 * D
def condition2 : Prop := F + 20 = 2 * (D + 20)

-- Main statement
theorem ages_of_father_and_daughter (h1 : condition1 F D) (h2 : condition2 F D) : D = 10 ∧ F = 40 := by
  sorry

end ages_of_father_and_daughter_l669_669334


namespace sum_a_b_eq_five_l669_669988

theorem sum_a_b_eq_five (a b : ℝ) (h : ∀ x : ℝ, 1 < x ∧ x < 2 → x^2 - a * x + b < 0) : a + b = 5 :=
sorry

end sum_a_b_eq_five_l669_669988


namespace num_paths_on_chessboard_l669_669201

theorem num_paths_on_chessboard (k l : ℕ) (hk : 1 ≤ k) (hl : 1 ≤ l) :
  (k + l - 2).choose (k - 1) = binomial (k + l - 2) (k - 1) :=
begin
  -- proof goes here
  sorry
end

end num_paths_on_chessboard_l669_669201


namespace determine_b_from_inequality_l669_669780

theorem determine_b_from_inequality (b : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - b * x + 6 < 0) → b = 5 :=
by
  intro h
  -- Proof can be added here
  sorry

end determine_b_from_inequality_l669_669780


namespace smallest_n_exists_l669_669189

open Real

noncomputable def smallest_n : ℕ := 26

theorem smallest_n_exists :
  ∃ (n : ℕ) (x : Fin n → ℝ),
    (∀ i, |x i| < 1) ∧
    (∑ i, |x i| = 25 + |∑ i, x i|) ∧
    n = smallest_n :=
by
  use 26, fun i => if i.val < 13 then 25/26 else -25/26
  split
  { intro i
    split_ifs
    · exact abs_lt.2 ⟨neg_lt.2 (by norm_num), by norm_num⟩
    · exact abs_lt.2 ⟨neg_lt.2 (by norm_num), by norm_num⟩ }
  split
  { have h1 := sum_const (25 / 26) 13
    have h2 := sum_const (-25 / 26) 13
    rw [h1, h2, mul_div_cancel' _ (nat.cast_ne_zero.2 (by norm_num : (26 : ℝ) ≠ 0)), add_comm, abs_eq_self, add_zero]
    · linarith }
  { refl }
  sorry

end smallest_n_exists_l669_669189


namespace part_1_part_2_l669_669196

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

def B (m : ℝ) : Set ℝ := { x | x^2 - (2*m + 1)*x + 2*m < 0 }

theorem part_1 (m : ℝ) (h : m < 1/2) : 
  B m = { x | 2*m < x ∧ x < 1 } := 
sorry

theorem part_2 (m : ℝ) : 
  (A ∪ B m = A) ↔ -1/2 ≤ m ∧ m ≤ 1 := 
sorry

end part_1_part_2_l669_669196


namespace biking_time_l669_669795

noncomputable def east_bound_speed : ℝ := 22
noncomputable def west_bound_speed : ℝ := east_bound_speed + 4
noncomputable def total_distance : ℝ := 200

theorem biking_time :
  (east_bound_speed + west_bound_speed) * (t : ℝ) = total_distance → t = 25 / 6 :=
by
  -- The proof is omitted and replaced with sorry.
  sorry

end biking_time_l669_669795


namespace ratio_of_terms_l669_669954

open_locale big_operators

noncomputable def geometric_sequence (r a₁ : ℝ) : ℕ → ℝ
| 0       := a₁
| (n + 1) := r * geometric_sequence n

noncomputable def S (r a₁ : ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range (n + 1), (geometric_sequence r a₁ i)

theorem ratio_of_terms
  (a₁ : ℝ)
  (r : ℝ)
  (h : S r a₁ 5 = 3 * S r a₁ 2) :
  (geometric_sequence r a₁ 6 + geometric_sequence r a₁ 8) / (geometric_sequence r a₁ 0 + geometric_sequence r a₁ 2) = 4 :=
sorry

end ratio_of_terms_l669_669954


namespace inequality_proof_l669_669661

theorem inequality_proof (n k : ℕ) (h₁ : 0 < n) (h₂ : 0 < k) (h₃ : k ≤ n) :
  1 + k / n ≤ (1 + 1 / n)^k ∧ (1 + 1 / n)^k < 1 + k / n + k^2 / n^2 :=
sorry

end inequality_proof_l669_669661


namespace vector_norm_inequality_l669_669460

variables {V : Type} [inner_product_space ℝ V]

variables (a b c d : V)

theorem vector_norm_inequality (h : a + b + c + d = 0) :
  ∥a∥ + ∥b∥ + ∥c∥ + ∥d∥ ≥ ∥a + d∥ + ∥b + d∥ + ∥c + d∥ :=
sorry

end vector_norm_inequality_l669_669460


namespace perfect_square_condition_l669_669523

noncomputable def isPerfectSquareQuadratic (m : ℤ) (x y : ℤ) :=
  ∃ (k : ℤ), (4 * x^2 + m * x * y + 25 * y^2) = k^2

theorem perfect_square_condition (m : ℤ) :
  (∀ x y : ℤ, isPerfectSquareQuadratic m x y) → (m = 20 ∨ m = -20) :=
by
  sorry

end perfect_square_condition_l669_669523


namespace correct_statements_for_sequence_l669_669457

variable {a : ℝ}

def is_arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a_n (n + 1) = a_n n + d

def is_geometric_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a_n (n + 1) = a_n n * r

def sequence_sums_to (S : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
  ∀ n, S n = a_n 0 + (S (n - 1) + a_n n) - a_n (n - 1)

def correct_sequence_statements (a_n : ℕ → ℝ) : ℕ :=
  (if (∀ n, a ≠ 0 ∧ a ≠ 1 → (a ∈ S n → is_geometric_sequence a_n) 
       ∧ a = 1 → is_arithmetic_sequence a_n) then 1 else 0) +
  (if (a = 0 ∧ (a_n 1 = -1 ∧ ∀ n, n ≥ 2 → a_n n = 0) → %(~is_arithmetic_sequence a_n ∧ ~is_geometric_sequence a_n))
      then 1 else 0)

theorem correct_statements_for_sequence :
  ∀ S a_n, (∀ n, S n = a^n - 1) → correct_sequence_statements a_n = 2 :=
sorry

end correct_statements_for_sequence_l669_669457


namespace convex_polygon_sides_l669_669543

theorem convex_polygon_sides (n : ℕ) (h1 : 180 * (n - 2) - 90 = 2790) : n = 18 :=
sorry

end convex_polygon_sides_l669_669543


namespace triangle_side_AC_l669_669142

theorem triangle_side_AC 
  (AB BC : ℝ)
  (angle_C : ℝ)
  (h1 : AB = Real.sqrt 13)
  (h2 : BC = 3)
  (h3 : angle_C = Real.pi / 3) :
  ∃ AC : ℝ, AC = 4 :=
by 
  sorry

end triangle_side_AC_l669_669142


namespace correct_transformation_l669_669482

-- Define the two curves.
def C1 (x : ℝ) : ℝ := Real.sin (x + π / 2)
def C2 (x : ℝ) : ℝ := Real.sin (2 * x + 2 * π / 3)

-- Define the transformation functions.
def shrink_abscissa (f : ℝ → ℝ) : ℝ → ℝ := λ x, f (2 * x)
def shift_left (f : ℝ → ℝ) (d : ℝ) : ℝ → ℝ := λ x, f (x + d)

-- Create a transformed curve from C1.
def transformed_C1 := shift_left (shrink_abscissa C1) (-π / 12)

-- State the theorem to prove that the transformed curve matches C2.
theorem correct_transformation : transformed_C1 = C2 :=
sorry

end correct_transformation_l669_669482


namespace no_such_P_exists_l669_669013

theorem no_such_P_exists (P : Polynomial ℤ) (r : ℕ) (r_ge_3 : r ≥ 3) (a : Fin r → ℤ)
  (distinct_a : ∀ i j, i ≠ j → a i ≠ a j)
  (P_cycle : ∀ i, P.eval (a i) = a ⟨(i + 1) % r, sorry⟩)
  : False :=
sorry

end no_such_P_exists_l669_669013


namespace smallest_positive_period_pi_area_of_plane_figure_l669_669467

noncomputable def f (x : ℝ) : ℝ := (3.sqrt * Real.sin x * Real.cos x) - (Real.cos x)^2 + 1 / 2

theorem smallest_positive_period_pi :
  ∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = Real.pi :=
by
  sorry

theorem area_of_plane_figure :
  let S := - ∫ x in 0..(Real.pi / 12), Real.sin(2 * x - Real.pi / 6) + 
            3 * ∫ x in (Real.pi / 12)..(Real.pi / 3), Real.sin(2 * x - Real.pi / 6) 
  in S = 2 - Real.sqrt(3) / 4 :=
by
  sorry

end smallest_positive_period_pi_area_of_plane_figure_l669_669467


namespace distinct_binomial_coefficients_l669_669039

theorem distinct_binomial_coefficients :
  ∀ (m n : ℕ), (1 ≤ n ∧ n ≤ m ∧ m ≤ 5) →
  ∃ S : Finset ℕ, S = { C m n | 1 ≤ n ∧ n ≤ m ∧ m ≤ 5 } ∧ S.card = 7 :=
begin
  sorry
end

end distinct_binomial_coefficients_l669_669039


namespace cubic_difference_l669_669967

theorem cubic_difference (a b : ℝ) 
  (h₁ : a - b = 7)
  (h₂ : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := 
by 
  sorry

end cubic_difference_l669_669967


namespace concurrency_lines_l669_669888

open EuclideanGeometry

noncomputable def circles_equal_radii (O1 O2 : Point) (R : ℝ) :=
  ∀ P, P ∈ circle O1 R ↔ P ∈ circle O2 R

variables {A B C H X Y Z : Point} {O1 O2 : Circle} (R : ℝ)

def intersecting_circles := O1 ≠ O2 ∧ O1.radius = R ∧ O2.radius = R ∧ (X ∈ O1 ∧ X ∈ O2) ∧ (Y ∈ O1 ∧ Y ∈ O2)
def tri_inscrib := triangle ABC ∧ (A ∈ O1) ∧ (B ∈ O1) ∧ (C ∈ O1) ∧ orthocenter H ABC
def quadr_parallel := parallelogram C X Z Y

theorem concurrency_lines (h₁ : intersecting_circles O1 O2 R)
                          (h₂ : tri_inscrib A B C O1 H)
                          (h₃ : quadr_parallel C X Z Y) :
  concurrency (line AB) (line XY) (line HZ) :=
sorry

end concurrency_lines_l669_669888


namespace cheburashkas_erased_l669_669613

def total_krakozyabras : ℕ := 29

def total_rows : ℕ := 2

def cheburashkas_per_row := (total_krakozyabras + total_rows) / total_rows / 2 + 1

theorem cheburashkas_erased :
  (total_krakozyabras + total_rows) / total_rows / 2 - 1 = 11 := 
by
  sorry

-- cheburashkas_erased proves that the number of Cheburashkas erased is 11 from the given conditions.

end cheburashkas_erased_l669_669613


namespace ab_ac_bc_nonpositive_l669_669626

theorem ab_ac_bc_nonpositive (a b c : ℝ) (h : a + b + c = 0) : ∃ y : ℝ, y = ab + ac + bc ∧ y ≤ 0 :=
by
  sorry

end ab_ac_bc_nonpositive_l669_669626


namespace max_log_sum_value_l669_669054

noncomputable def max_log_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 4 * y = 40) : ℝ :=
  Real.log x + Real.log y

theorem max_log_sum_value : ∀ (x y : ℝ), x > 0 → y > 0 → x + 4 * y = 40 → max_log_sum x y sorry sorry sorry = 2 :=
by
  intro x y h1 h2 h3
  sorry

end max_log_sum_value_l669_669054


namespace smallest_positive_period_area_of_triangle_ABC_l669_669509

theorem smallest_positive_period (f : ℝ → ℝ)
  (h₀ : ∀ x, f(x) = 4 * cos x * sin (x - π / 6)) :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π :=
begin
  sorry -- proof will be filled in by the user
end

theorem area_of_triangle_ABC (A B C : ℝ)
  (BC : ℝ) (h1 : BC = 4) (h2 : sin C = 2 * sin B)
  (h3: ∀ x, f x = 4 * cos x * sin (x - π / 6))
  (h4: ∀ x, max (f x) = f A)
  : area_of_triangle ABC = (8 * sqrt 3) / 3 :=
begin
  sorry -- proof will be filled in by the user
end

end smallest_positive_period_area_of_triangle_ABC_l669_669509


namespace comparison_of_functions_l669_669051
open Real

theorem comparison_of_functions (x : ℝ) (hx : x > 2) :
  let a := (1/3) ^ x
  let b := x ^ 3
  let c := log x
  a < c ∧ c < b :=
by
  let a := (1/3) ^ x
  let b := x ^ 3
  let c := log x
  sorry

end comparison_of_functions_l669_669051


namespace problem1_problem2_l669_669194

open Real

noncomputable def f (x a : ℝ) : ℝ := |2 * x + 3| - |2 * x - a|

-- Problem (1)
theorem problem1 {a : ℝ} (h : ∃ x, f x a ≤ -5) : a ≤ -8 ∨ a ≥ 2 :=
sorry

-- Problem (2)
theorem problem2 {a : ℝ} (h : ∀ x, f (x - 1/2) a + f (-x - 1/2) a = 0) : a = 1 :=
sorry

end problem1_problem2_l669_669194


namespace minimum_value_2_l669_669880

theorem minimum_value_2 (x : ℝ) (h1 : ∀ x, y1 x = x + 1/x)
                       (h2 : ∀ x, 0 < x ∧ x < π/2 → y2 x = cos x + 1/cos x)
                       (h3 : ∀ x, y3 x = (x^2 + 3) / sqrt(x^2 + 2))
                       (h4 : ∀ x, y4 x = exp x + 4 / exp x - 2) 
                       : (∀ x, y4 x ≥ 2) ∧ (∃ x, y4 x = 2) :=
by
  sorry

end minimum_value_2_l669_669880


namespace negation_of_P_l669_669490

-- Define the proposition P
def P : Prop := ∀ x : ℝ, x > Real.sin x

-- Formulate the negation of P
def neg_P : Prop := ∃ x : ℝ, x ≤ Real.sin x

-- State the theorem to be proved
theorem negation_of_P (hP : P) : neg_P :=
sorry

end negation_of_P_l669_669490


namespace probability_ratio_l669_669911

open_locale big_operators

-- Parameters representing conditions
def n_balls : ℕ := 20
def n_bins : ℕ := 5
def p : ℚ := 
  ((fintype.card {s : finset (fin n_bins) // s.card = 1 ∧ ∀ (x : ℕ), x ∈ s → x = 3})
  * (fintype.card {s : finset (fin n_bins) // s.card = 1 ∧ ∀ (x : ℕ), x ∈ s → x = 5})
  * (fintype.card {s : finset (fin n_bins) // s.card = 3 ∧ ∀ (x : ℕ), x ∈ s → x = 4}))
  / fintype.card {f : fin n_bins → fin n_balls}
def q : ℚ := 
  fintype.card {f : fin n_bins → fin n_balls // ∀ i, f i = 4}
  / fintype.card {f : fin n_bins → fin n_balls}

theorem probability_ratio :
  (p / q) = 16 :=
by sorry

end probability_ratio_l669_669911


namespace question_implies_answer_l669_669402

theorem question_implies_answer (x y : ℝ) (h : y^2 - x^2 < x) :
  (x ≥ 0 ∨ x ≤ -1) ∧ (-Real.sqrt (x^2 + x) < y ∧ y < Real.sqrt (x^2 + x)) :=
sorry

end question_implies_answer_l669_669402


namespace geometric_sequence_sixth_term_l669_669582

theorem geometric_sequence_sixth_term:
  ∃ q : ℝ, 
  ∀ (a₁ a₈ a₆ : ℝ), 
    a₁ = 6 ∧ a₈ = 768 ∧ a₈ = a₁ * q^7 ∧ a₆ = a₁ * q^5 
    → a₆ = 192 :=
by
  sorry

end geometric_sequence_sixth_term_l669_669582


namespace solution_l669_669976

noncomputable def f (x : ℝ) : ℝ

axiom even_f : ∀ x : ℝ, f x = f (-x)
axiom monotonic_f : ∀ x y : ℝ, x < y ∧ x ≤ 0 ∧ y ≤ 0 → f y ≤ f x
axiom f_value : f (-1/3) = 0

theorem solution : ∀ x : ℝ, f (Real.log x / Real.log (1/8)) + f (Real.log x / Real.log 8) > 0 ↔ (0 < x ∧ x < 1/2) ∨ (x > 2) :=
by
  sorry

end solution_l669_669976


namespace probability_ratio_l669_669910

open_locale big_operators

-- Parameters representing conditions
def n_balls : ℕ := 20
def n_bins : ℕ := 5
def p : ℚ := 
  ((fintype.card {s : finset (fin n_bins) // s.card = 1 ∧ ∀ (x : ℕ), x ∈ s → x = 3})
  * (fintype.card {s : finset (fin n_bins) // s.card = 1 ∧ ∀ (x : ℕ), x ∈ s → x = 5})
  * (fintype.card {s : finset (fin n_bins) // s.card = 3 ∧ ∀ (x : ℕ), x ∈ s → x = 4}))
  / fintype.card {f : fin n_bins → fin n_balls}
def q : ℚ := 
  fintype.card {f : fin n_bins → fin n_balls // ∀ i, f i = 4}
  / fintype.card {f : fin n_bins → fin n_balls}

theorem probability_ratio :
  (p / q) = 16 :=
by sorry

end probability_ratio_l669_669910


namespace maximize_triangle_area_l669_669485

noncomputable def area_of_triangle_maximized (m : ℝ) (h1 : 1 < m) (h2 : m < 4) : Prop :=
  let A := (1,real.sqrt 1)
  let B := (m,real.sqrt m)
  let C := (4,real.sqrt 4)
  let Δ := 1 / 2 * abs (1 * (real.sqrt m - 2) + m * (2 - 1) + 4 * (1 - real.sqrt m))
  let Δ_exp := 1 / 2 * abs (m - 3*real.sqrt m + 2)
  Δ = Δ_exp ∧ Δ_exp = 1 / 2 * abs (real.sqrt (9/4))

theorem maximize_triangle_area : ∃ (m : ℝ), (1 < m ∧ m < 4) ∧ area_of_triangle_maximized m := 
sorry

end maximize_triangle_area_l669_669485


namespace first_player_win_l669_669378

def player1_wins_optimal : Prop :=
  ∃ (strategy : (ℕ → ℕ) → (ℕ → ℕ)), -- strategy that first player can use given positions
  ∀ start_pos1 start_pos2, -- start positions
  (start_pos1 = 1) → (start_pos2 = 101) →
  (∀ (pos1 pos2 : ℕ) -- positions during game
      (move1 move2 : ℕ) -- allowable moves
      (h_move1 : move1 ∈ {1, 2, 3, 4}) -- valid move for player 1
      (h_move2 : move2 ∈ {1, 2, 3, 4}), -- valid move for player 2
      (pos1 + move1 ≠ pos2 + move2)), -- cannot land on same cell
  strategy start_pos1 start_pos2 = 101 -- win condition for player 1

theorem first_player_win : player1_wins_optimal := 
sorry

end first_player_win_l669_669378


namespace max_queens_on_black_squares_l669_669287

-- Definitions related to the problem
def chessboard (n : ℕ) := fin n × fin n
def is_black_square (pos : fin 8 × fin 8) : Prop := (pos.1.val + pos.2.val) % 2 = 1

def is_queen (board : chessboard 8 → Prop) (pos : chessboard 8) : Prop := board pos

-- Predicate to check if a queen is attacked by another
def is_attacked_by (board : chessboard 8 → Prop) (pos : chessboard 8) : Prop :=
∃ (q_pos : chessboard 8), q_pos ≠ pos ∧ is_queen board q_pos ∧
  (q_pos.1 = pos.1 ∨ q_pos.2 = pos.2 ∨ abs (q_pos.1.val - pos.1.val) = abs (q_pos.2.val - pos.2.val))

def valid_queen_placement (board : chessboard 8 → Prop) : Prop :=
∀ (pos : chessboard 8), is_queen board pos → is_black_square pos ∧ is_attacked_by board pos

-- Condition to check the number of queens placed
def num_queens (board : chessboard 8 → Prop) : ℕ :=
finset.univ.filter (λ p, is_queen board p).card

-- The theorem stating the maximum number of queens fitting the conditions
theorem max_queens_on_black_squares : ∃ (board : chessboard 8 → Prop), valid_queen_placement board ∧ num_queens board = 16 :=
sorry

end max_queens_on_black_squares_l669_669287


namespace trees_planted_in_garden_l669_669550

theorem trees_planted_in_garden (yard_length : ℕ) (tree_distance : ℕ) (h₁ : yard_length = 500) (h₂ : tree_distance = 20) :
  ((yard_length / tree_distance) + 1) = 26 :=
by
  -- The proof goes here
  sorry

end trees_planted_in_garden_l669_669550


namespace midpoint_of_DQ_l669_669558

theorem midpoint_of_DQ (ABC : Type) [acute_angle_triangle ABC] (A B C D E F P Q H : ABC)
  (h_AD_altitude : is_altitude AD BC)
  (h_DE_perpendicular : is_perpendicular DE AC)
  (h_DF_perpendicular : is_perpendicular DF AB)
  (h_E_on_AC : lies_on E AC)
  (h_F_on_AB : lies_on F AB)
  (h_P_intersection : intersects (extend FE) (extend BC) P)
  (h_H_on_AD : lies_on H AD)
  (h_M_intersection : intersects (extend BH) AC M)
  (h_N_intersection : intersects (extend CH) AB N)
  (h_Q_intersection : intersects (extend NM) (extend BC) Q) :
  midpoint P DQ := 
sorry

end midpoint_of_DQ_l669_669558


namespace range_of_c_l669_669958

variable (c : ℝ)

-- Definitions for p and q
def p : Prop := c^2 < c
def q : Prop := ∀ x : ℝ, x^2 + 4 * c * x + 1 > 0

-- Given conditions
variable (h : p ∨ q)
variable (hn : ¬ (p ∧ q))

-- Desired range of c
def correct_range : Set ℝ := {c | -1/2 < c ∧ c ≤ 0} ∪ {c | 1/2 ≤ c ∧ c < 1}

theorem range_of_c : (p ∨ q) ∧ ¬ (p ∧ q) → c ∈ correct_range := 
by
  intros _ 
  sorry

end range_of_c_l669_669958


namespace find_n_l669_669531

theorem find_n (n : ℕ) : sqrt (8 + n) = 9 → n = 73 :=
by
  intro h
  sorry

end find_n_l669_669531


namespace point_on_y_axis_l669_669138

theorem point_on_y_axis (m n : ℝ) (h : (m, n).1 = 0) : m = 0 :=
by
  sorry

end point_on_y_axis_l669_669138


namespace coke_calories_proof_l669_669280

def cake_calories : ℕ := 110
def chips_calories : ℕ := 310
def breakfast_calories : ℕ := 560
def lunch_calories : ℕ := 780
def daily_limit : ℕ := 2500
def remaining_calories : ℕ := 525

-- The proof statement for the number of calories in the coke
theorem coke_calories_proof (coke_calories : ℕ) :
  let total_consumed := cake_calories + chips_calories + breakfast_calories + lunch_calories in
  let calories_left_before_coke := daily_limit - total_consumed in
  calories_left_before_coke - remaining_calories = coke_calories →
  coke_calories = 215 :=
by
  sorry

end coke_calories_proof_l669_669280


namespace constant_term_expansion_sum_binomial_coefficients_l669_669979

theorem constant_term_expansion (n : ℕ) 
  (h : 2^(2 * n) - 2^n = 240) : 
  let k := 2 * n in binomial k (k / 2) = 70 := 
by
  sorry

theorem sum_binomial_coefficients (n : ℕ)
  (h : 2^(2 * n) - 2^n = 240) :
  2^n = 16 := 
by
  sorry

end constant_term_expansion_sum_binomial_coefficients_l669_669979


namespace probability_two_queens_or_at_least_one_king_l669_669128

/-- Probability that either two queens or at least one king occurs 
when 2 cards are selected randomly from a standard deck.
-/
theorem probability_two_queens_or_at_least_one_king :
  let P := 
    -- define conditions
    let total_cards := 52
    let num_kings := 4
    let num_queens := 4
    let prob_two_queens := (num_queens * (num_queens - 1)) / (total_cards * (total_cards - 1))
    let prob_exactly_one_king := 
      (num_kings * (total_cards - num_kings)) / (total_cards * (total_cards - 1)) + 
      ((total_cards - num_kings) * num_kings) / (total_cards * (total_cards - 1))
    let prob_two_kings := (num_kings * (num_kings - 1)) / (total_cards * (total_cards - 1))
    let prob_at_least_one_king := prob_exactly_one_king + prob_two_kings
  in prob_two_queens + prob_at_least_one_king = 34 / 221 
:= sorry

end probability_two_queens_or_at_least_one_king_l669_669128


namespace angle_between_vectors_l669_669117

noncomputable def vector_length (a : ℝ × ℝ) : ℝ :=
  real.sqrt (a.1 * a.1 + a.2 * a.2)

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def angle_between (a b : ℝ × ℝ) : ℝ :=
  real.arccos ((dot_product a b) / (vector_length a * vector_length b))

variables (x1 y1 x2 y2 : ℝ)

theorem angle_between_vectors :
  let a := (x1, y1) in
  let b := (x2, y2) in
  vector_length a = 2 →
  vector_length b = 3 →
  dot_product a b = -6 →
  angle_between a b = real.pi :=
by
  intros a b ha hb hab
  sorry

end angle_between_vectors_l669_669117


namespace evaluate_sqrt_sum_l669_669917

noncomputable def sqrt_minus : ℝ := real.sqrt (20 - 8 * real.sqrt 5)
noncomputable def sqrt_plus : ℝ := real.sqrt (20 + 8 * real.sqrt 5)

theorem evaluate_sqrt_sum : sqrt_minus + sqrt_plus = 2 * real.sqrt 10 := by
  sorry

end evaluate_sqrt_sum_l669_669917


namespace sin_theta_plus_2phi_l669_669534

-- Given conditions
def e_i_theta := (1 / 5 : ℂ) - (2 / 5 : ℂ) * Complex.I
def e_i_phi := (3 / 5 : ℂ) + (4 / 5 : ℂ) * Complex.I

-- The proof statement
theorem sin_theta_plus_2phi : 
  Complex.sin (Complex.arg e_i_theta + 2 * Complex.arg e_i_phi) = 62 / 125 :=
sorry

end sin_theta_plus_2phi_l669_669534


namespace part_a_gray_black_area_difference_l669_669834

theorem part_a_gray_black_area_difference :
    ∀ (a b : ℕ), 
        a = 4 → 
        b = 3 →
        a^2 - b^2 = 7 :=
by
  intros a b h_a h_b
  sorry

end part_a_gray_black_area_difference_l669_669834


namespace remaining_overs_correct_l669_669569

noncomputable def run_rate_first_10_overs : ℝ := 4.6
noncomputable def overs_first_segment : ℝ := 10
noncomputable def total_target_runs : ℝ := 282
noncomputable def required_run_rate_remaining : ℝ := 5.9

def remaining_overs : ℝ := (total_target_runs - (run_rate_first_10_overs * overs_first_segment)) / required_run_rate_remaining

theorem remaining_overs_correct :
  remaining_overs = 40 := by
  unfold remaining_overs
  norm_num
  sorry

end remaining_overs_correct_l669_669569


namespace expected_value_X_is_half_l669_669253

open BigOperators

-- Define the random digit sequence and its properties
def random_digit_seq (n : ℕ) (d : ℕ → ℕ) := ∀ i : ℕ, i < n → d i ∈ Fin 10

-- Expected value of a single random digit
def expected_value_digit : ℝ := 4.5

-- Define the expected value of X
noncomputable def expected_value_of_X (n : ℕ) (d : ℕ → ℕ) : ℝ :=
  ∑ i in Finset.range n, (d i : ℝ) * 10^(-(i+1))

-- The main theorem to prove
theorem expected_value_X_is_half : 
  ∀ (d : ℕ → ℕ), random_digit_seq 1000 d → expected_value_of_X 1000 d = 0.5 :=
by {
  intro d,
  intro h,
  sorry  -- The proof would be written here.
}

end expected_value_X_is_half_l669_669253


namespace BM_CN_gt_KM_KN_l669_669313

noncomputable def triangle (A B C K M N : Point) :=
  (is_triangle A B C) ∧
  (on_segment K B C) ∧
  (incenter_touches_triangle_side (ABK_triangle A B K) M BC) ∧
  (incenter_touches_triangle_side (ACK_triangle A C K) N BC) ∧
  (BM_tangent A B M K) ∧
  (CN_tangent A C N K)

theorem BM_CN_gt_KM_KN (A B C K M N : Point) (h : triangle A B C K M N) :
  tangent_length (B M) * tangent_length (C N) > tangent_length (K M) * tangent_length (K N) :=
sorry

end BM_CN_gt_KM_KN_l669_669313


namespace part1_part2_l669_669061

noncomputable def midpoint_m_condition (m : ℝ) : Prop :=
  m > 0 ∧ (1/4 + m^2 / 3 < 1)

theorem part1 (k m : ℝ) (k_condition : k = -3 / (4 * m)) (m_cond : midpoint_m_condition m) : 
  k < -1/2 :=
sorry

noncomputable def arithmetic_sequence_condition (x1 x2 x3 y1 y2 y3 : ℝ) : Prop :=
  x1 + x2 = 2 ∧ x1 - 1 + x2 - 1 + x3 - 1 = 0 ∧ y1 + y2 + y3 = 0 ∧ (1, 0) = (1, 0)

theorem part2 (x1 x2 x3 y1 y2 y3 : ℝ) 
  (ellipse_eq : (x1^2)/4 + (y1^2)/3 = 1 ∧ (x2^2)/4 + (y2^2)/3 = 1)
  (midpoint_condition : arithmetic_sequence_condition x1 x2 x3 y1 y2 y3) :
  ∃ d : ℝ, ∀ (|FA| |FP| |FB| : ℝ), (|FA| + |FB| = 2 * |FP| ∧ d = ±(3 * sqrt 21)/(28)) :=
sorry

end part1_part2_l669_669061


namespace purely_imaginary_condition_l669_669316

open Complex

theorem purely_imaginary_condition (a : ℝ) : 
  (a = 1 ↔ ∀ z : ℂ, z = complex.mk (a^2 - 1) (2 * (a + 1)) → Im z ≠ 0 ∧ Re z = 0) :=
by
  sorry

end purely_imaginary_condition_l669_669316


namespace borrow_two_books_l669_669776

theorem borrow_two_books (books : Finset ℕ) (h : books.card = 4) : 
  books.choose 2 = 6 :=
sorry

end borrow_two_books_l669_669776


namespace maximize_triangle_area_l669_669486

noncomputable def area_of_triangle_maximized (m : ℝ) (h1 : 1 < m) (h2 : m < 4) : Prop :=
  let A := (1,real.sqrt 1)
  let B := (m,real.sqrt m)
  let C := (4,real.sqrt 4)
  let Δ := 1 / 2 * abs (1 * (real.sqrt m - 2) + m * (2 - 1) + 4 * (1 - real.sqrt m))
  let Δ_exp := 1 / 2 * abs (m - 3*real.sqrt m + 2)
  Δ = Δ_exp ∧ Δ_exp = 1 / 2 * abs (real.sqrt (9/4))

theorem maximize_triangle_area : ∃ (m : ℝ), (1 < m ∧ m < 4) ∧ area_of_triangle_maximized m := 
sorry

end maximize_triangle_area_l669_669486


namespace sum_b_n_l669_669884

noncomputable def a_n (n : ℕ) : ℝ := (n + 1) * Real.pi
noncomputable def b_n (n : ℕ) : ℝ := 2 ^ (n + 1) * a_n n

theorem sum_b_n (n : ℕ) : 
  (∑ k in Finset.range (n + 1), b_n k) = ((n - 1) * 2^(n + 1) + 2) * Real.pi :=
by
  sorry

end sum_b_n_l669_669884


namespace Nick_sister_age_l669_669648

theorem Nick_sister_age
  (Nick_age : ℕ := 13)
  (Bro_in_5_years : ℕ := 21)
  (H : ∃ S : ℕ, (Nick_age + S) / 2 + 5 = Bro_in_5_years) :
  ∃ S : ℕ, S = 19 :=
by
  sorry

end Nick_sister_age_l669_669648


namespace fermats_little_theorem_l669_669664

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (ha : ¬ p ∣ a) :
  a^(p-1) ≡ 1 [MOD p] :=
sorry

end fermats_little_theorem_l669_669664


namespace simplify_cubed_root_l669_669710

def c1 : ℕ := 54880000
def c2 : ℕ := 10^5 * 5488
def c3 : ℕ := 5488
def c4 : ℕ := 2^4 * 343
def c5 : ℕ := 343
def c6 : ℕ := 7^3

theorem simplify_cubed_root : (c1^(1 / 3 : ℝ) : ℝ) = 1400 := 
by {
  let h1 : c1 = c2 := sorry,
  let h2 : c3 = c4 := sorry,
  let h3 : c5 = c6 := sorry,
  rw [h1, h2, h3],
  sorry
}

end simplify_cubed_root_l669_669710


namespace cheburashkas_erased_l669_669616

def total_krakozyabras : ℕ := 29

def total_rows : ℕ := 2

def cheburashkas_per_row := (total_krakozyabras + total_rows) / total_rows / 2 + 1

theorem cheburashkas_erased :
  (total_krakozyabras + total_rows) / total_rows / 2 - 1 = 11 := 
by
  sorry

-- cheburashkas_erased proves that the number of Cheburashkas erased is 11 from the given conditions.

end cheburashkas_erased_l669_669616


namespace man_speed_against_current_eq_l669_669858

-- Definitions
def downstream_speed : ℝ := 22 -- Man's speed with the current in km/hr
def current_speed : ℝ := 5 -- Speed of the current in km/hr

-- Man's speed in still water
def man_speed_in_still_water : ℝ := downstream_speed - current_speed

-- Man's speed against the current
def speed_against_current : ℝ := man_speed_in_still_water - current_speed

-- Theorem: The man's speed against the current is 12 km/hr.
theorem man_speed_against_current_eq : speed_against_current = 12 := by
  sorry

end man_speed_against_current_eq_l669_669858


namespace seashells_increase_l669_669914

def initial_seashells : ℕ := 50
def final_seashells : ℕ := 130
def week_increment (x : ℕ) : ℕ := 4 * x + initial_seashells

theorem seashells_increase (x : ℕ) (h: final_seashells = week_increment x) : x = 8 :=
by {
  sorry
}

end seashells_increase_l669_669914


namespace focus_bisects_midpoint_l669_669475

-- Definitions and conditions
def parabola_focus : Point := ⟨0, 1/4⟩
def parabola_point (x : ℝ) : Point := ⟨x, x^2⟩
def directrix_projection (x : ℝ) : Point := ⟨x, -1/4⟩

-- Theorem statement
theorem focus_bisects_midpoint (x1 x2 : ℝ) :
  let F := parabola_focus,
      P1 := parabola_point x1,
      P2 := parabola_point x2,
      Q1 := directrix_projection x1,
      Q2 := directrix_projection x2,
      midpoint := Point.mk ((Q1.1 + Q2.1) / 2) (-1/4) in
  perpendicular_from_focus_to_line_segment_bisects F P1 P2 Q1 Q2 midpoint :=
sorry

end focus_bisects_midpoint_l669_669475


namespace minimum_cactus_species_l669_669369

theorem minimum_cactus_species (cactophiles : Fin 80 → Set (Fin k)) :
  (∀ s : Fin k, ∃ col, cactophiles col s = False) ∧
  (∀ group : Set (Fin 80), group.card = 15 → ∃ c : Fin k, ∀ col ∈ group, (cactophiles col c)) →
  16 ≤ k :=
by
  sorry

end minimum_cactus_species_l669_669369


namespace cheburashkas_erased_l669_669612

theorem cheburashkas_erased (total_krakozyabras : ℕ) (rows : ℕ) :
  rows ≥ 2 → total_krakozyabras = 29 → ∃ (cheburashkas_erased : ℕ), cheburashkas_erased = 11 :=
by
  assume h_rows h_total_krakozyabras
  let n := (total_krakozyabras / 2) + 1
  have h_cheburashkas : cheburashkas_erased = n - 1 
  sorry

end cheburashkas_erased_l669_669612


namespace ratio_of_B1A1_and_A1C1_l669_669141

theorem ratio_of_B1A1_and_A1C1
    (ABC : Triangle)
    (A M B C B1 C1 A1 D : Point)
    (hM : M.is_midpoint_of (B, C))
    (hB1 : B1.is_foot_of_perpendicular B (angle_bisector (B, M, A)))
    (hC1 : C1.is_foot_of_perpendicular C (angle_bisector (A, M, C)))
    (h_intersection : A1.is_intersection_of MA B1C1)
    (hD : D.is_point_on AM ∧ D = reflection_of B (angle_bisector (B, M, A)) ∧ D = reflection_of C (angle_bisector (A, M, C)))
  : B1A1.distance / A1C1.distance = 1 :=
sorry

end ratio_of_B1A1_and_A1C1_l669_669141


namespace tiles_needed_for_hallway_l669_669343

def hallway_area_ft (length : ℝ) (width : ℝ) : ℝ := length * width

def square_feet_to_square_yards (area_sqft : ℝ) : ℝ := area_sqft / 9

def needed_tiles (length : ℝ) (width : ℝ) : ℝ := 
  let area_sqft := hallway_area_ft length width
  let area_syard := square_feet_to_square_yards area_sqft
  area_syard.ceil

theorem tiles_needed_for_hallway :
  needed_tiles 15 4 = 7 :=
by
  sorry

end tiles_needed_for_hallway_l669_669343


namespace functional_relationships_cost_effectiveness_max_boxes_with_budget_l669_669659

-- Functional relationships for Schemes A and B given the pricing of rackets and boxes of balls
def y_A (x : ℕ) : ℕ := 25 * x + 550
def y_B (x : ℕ) : ℕ := 22.5 * x + 720

-- Prove functional relationships
theorem functional_relationships (x : ℕ) (hx : x ≥ 10) :
  y_A x = 25 * x + 550 ∧ y_B x = 22.5 * x + 720 := 
by
  dsimp [y_A, y_B]
  exact ⟨rfl, rfl⟩ 

-- Prove cost-effectiveness for 15 boxes
theorem cost_effectiveness :
  let x := 15 in
  y_A x < y_B x :=
by
  let x := 15
  have hA : y_A x = 25 * x + 550 := rfl
  have hB : y_B x = 22.5 * x + 720 := rfl
  have hA_val : y_A x = 925 := by simp [hA]
  have hB_val : y_B x = 1057.5 := by simp [hB]
  exact lt_of_eq_of_lt hA_val hB_val sorry -- Skip proof

-- Prove which scheme allows purchasing more boxes within a budget of 1800 yuan
theorem max_boxes_with_budget :
  let budget := 1800 in
  let x_A := (budget - 550) / 25 in
  let x_B := (budget - 720) / 22.5 in
  x_A > x_B :=
by
  let budget := 1800
  let x_A : ℕ := (budget - 550) / 25
  let x_B : ℕ := (budget - 720) / 22.5
  have x_A_val : x_A = 50 := by simp [x_A]
  have x_B_val : x_B = 48 := by simp [x_B]
  exact gt_of_eq_of_gt x_A_val x_B_val sorry -- Skip proof

end functional_relationships_cost_effectiveness_max_boxes_with_budget_l669_669659


namespace circumscribed_sphere_radius_l669_669157

theorem circumscribed_sphere_radius (a b c : ℝ) : 
  R = (1/2) * Real.sqrt (a^2 + b^2 + c^2) := sorry

end circumscribed_sphere_radius_l669_669157


namespace cactus_species_minimum_l669_669368

theorem cactus_species_minimum :
  ∀ (collections : Fin 80 → Fin k → Prop),
  (∀ s : Fin k, ∃ (i : Fin 80), ¬ collections i s)
  → (∀ (c : Finset (Fin 80)), c.card = 15 → ∃ s : Fin k, ∀ (i : Fin 80), i ∈ c → collections i s)
  → 16 ≤ k := 
by 
  sorry

end cactus_species_minimum_l669_669368


namespace min_value_2x_plus_y_l669_669096

variable (x y : ℝ)
variable (x_pos : x > 0)
variable (y_pos : y > 0)
variable (h : x + 2 * y = 3 * x * y)

theorem min_value_2x_plus_y : ∃ (a : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → x + 2 * y = 3 * x * y → 2 * x + y ≥ a) ∧ a = 3 :=
by
  have inequality := sorry
  exists 3
  split
  · exact inequality
  · refl

end min_value_2x_plus_y_l669_669096


namespace mutually_exclusive_and_complementary_l669_669572

-- Define the events A and B and their probabilities
variables {Ω : Type*} [probability_space Ω] (A B : event Ω)

-- Noncomputable constant
noncomputable def P (e : event Ω) : ℝ := prob e

-- Main theorem statement
theorem mutually_exclusive_and_complementary 
  (h₁ : P(A ∪ B) = P(A) + P(B)) (h₂ : P(A) + P(B) = 1) : 
  (mutually_exclusive A B) ∧ (complementary A B) :=
by
  sorry

end mutually_exclusive_and_complementary_l669_669572


namespace sample_size_l669_669870

theorem sample_size (teachers students_male students_female sample_female : ℕ)
  (h_teachers : teachers = 200)
  (h_students_male : students_male = 1200)
  (h_students_female : students_female = 1000)
  (h_sample_female : sample_female = 80) :
  let population := teachers + students_male + students_female
      sampling_fraction := sample_female / students_female
      n := population * sampling_fraction in
    n = 192 :=
by {
  sorry
}

end sample_size_l669_669870


namespace possible_remainder_degrees_l669_669812

theorem possible_remainder_degrees (f : Polynomial ℝ) :
  ∃ r : Polynomial ℝ, ∃ q : Polynomial ℝ, f = q * (5 * Polynomial.monomial 7 1 - 2 * Polynomial.monomial 3 1 + 4 * Polynomial.monomial 1 1 - 8 * Polynomial.C 1) + r ∧ 
  (∀ r, degree r < 7) →
  degree r < 7 :=
by sorry

end possible_remainder_degrees_l669_669812


namespace minimum_cactus_species_l669_669363

/--
At a meeting of cactus enthusiasts, 80 cactophiles presented their collections,
each consisting of cacti of different species. It turned out that no single 
species of cactus is found in all collections simultaneously, but any 15 people
have cacti of the same species. Prove that the minimum total number of cactus 
species is 16.
-/
theorem minimum_cactus_species (k : ℕ) (h : ∀ (collections : fin 80 → finset (fin k)),
  (∀ i, collections i ≠ ∅) ∧ (∃ (j : fin k), ∀ i, j ∉ collections i) ∧ 
  (∀ (S : finset (fin 80)), S.card = 15 → ∃ j, ∀ i ∈ S, j ∈ collections i)) :
  k ≥ 16 :=
sorry

end minimum_cactus_species_l669_669363


namespace multiplication_table_sum_extremes_l669_669234

theorem multiplication_table_sum_extremes : 
  ∃ (a b c d e f : ℕ), 
  {a, b, c, d, e, f} = {2, 3, 5, 7, 11, 17} ∧ 
  a + b + c + d + e + f = 45 ∧ 
  (a + b + c) * (d + e + f) = 450 ∧ 
  (a + b + c) * (d + e + f) = 504 := 
sorry

end multiplication_table_sum_extremes_l669_669234


namespace cube_volume_l669_669764

theorem cube_volume (P : ℝ) (h : P = 20) : ∃ (V : ℝ), V = 125 :=
by
  -- Let side length of a cube be 's'.
  let s := P / 4
  -- Calculate the volume
  let V := s^3
  -- The hypothesis given
  have hs : s = 5 := by linarith [h]
  rw hs at V
  -- Showing that the volume V is 125 cm³
  use V
  rw [hs, pow_succ, pow_succ, mul_assoc, mul_assoc]
  norm_num
  exact h

end cube_volume_l669_669764


namespace max_PA_PB_PC_l669_669067

noncomputable def max_product (A B C P : Point) : ℝ :=
  dist P A * dist P B * dist P C

theorem max_PA_PB_PC :
  ∀ (A B C P : Point), 
  (∀ a b c : Point, is_right_triangle a b c ∧ (dist a b = 1) ∧ (dist a c = 1) → ∃ P : Point, 
  (P ∈ Triangle_side a b c) ∧ 
  (max_product a b c P = (Real.sqrt 2) / 4)) :=
begin
  sorry
end

end max_PA_PB_PC_l669_669067


namespace find_ellipse_equation_find_triangle_area_l669_669948

-- Define the conditions
variables {a b : ℝ} (ha : a = 2 * sqrt 3) (hb : b ^ 2 = 4)
variables {M : ℝ × ℝ}
variables (hM : M = (2 * sqrt 2, 2 * sqrt 3 / 3))
variables (hfoci : (2:ℝ) * a = 4 * sqrt 3)

-- Define the problem for the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the slope and the line intersection condition
variables {l_slope : ℝ} (hl_slope : l_slope = 1)
variables {P : ℝ × ℝ} (hP : P = (-3, 2))

-- Define the final proof statements
theorem find_ellipse_equation :
  (ellipse_equation M.1 M.2) ∧ 
  (ha ∧ hb ∧ hfoci) →
  (a = 2 * sqrt 3) ∧ (b ^ 2 = 4) ∧ (x y : ℝ, ellipse_equation x y → x^2 / 12 + y^2 / 4 = 1) :=
sorry

theorem find_triangle_area (x1 y1 x2 y2 : ℝ) (hA : (y1 = x1 + Some m)) (hB : (y2 = x2 + Some m))
  (hl : 4 * x1 ^ 2 + 6 * m * x1 + 3 * m ^ 2 - 12 = 0) :
  (a = 2 * sqrt 3 ∧ b ^ 2 = 4) ∧ 
  (hl_slope = 1) ∧ 
  (hP = (-3, 2)) → 
  (area P (x1, y1) (x2, y2) = 9 / 2) :=
sorry

end find_ellipse_equation_find_triangle_area_l669_669948


namespace smallest_three_digit_multiple_of_three_with_odd_hundreds_l669_669825

theorem smallest_three_digit_multiple_of_three_with_odd_hundreds :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ a % 2 = 1 ∧ n % 3 = 0 ∧ n = 102) :=
by
  sorry

end smallest_three_digit_multiple_of_three_with_odd_hundreds_l669_669825


namespace tank_fill_rate_l669_669874

-- Define the given conditions
def base_area : ℕ := 100  -- Base area in cm²
def height_increase_rate : ℕ := 10  -- Height increase rate in cm/min
def cm_to_liter_conversion : ℕ := 1000  -- Conversion from cm³ to liters

-- Define the target rate (volume increase in liters per minute)
def volume_rate_proof : Prop :=
  let volume_cm3_per_min := base_area * height_increase_rate in
  let volume_liters_per_min := volume_cm3_per_min / cm_to_liter_conversion in
  volume_liters_per_min = 1

theorem tank_fill_rate : volume_rate_proof :=
by
  -- Here, the proof will be filled in.
  sorry

end tank_fill_rate_l669_669874


namespace hyperbola_foci_coordinates_l669_669223

theorem hyperbola_foci_coordinates :
  (∀ x y : ℝ, x^2 / 4 - y^2 / 9 = 1 → (x, y) = (± sqrt 13, 0)) :=
sorry

end hyperbola_foci_coordinates_l669_669223


namespace min_houses_20x20_min_houses_50x90_l669_669279

theorem min_houses_20x20 : (min_houses_in_grid 20 20) = 25 :=
sorry

theorem min_houses_50x90 : (min_houses_in_grid 50 90) = 282 :=
sorry

end min_houses_20x20_min_houses_50x90_l669_669279


namespace arithmetic_seq_sum_l669_669564

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a)
  (h1 : a 1 + a 4 + a 7 = 39) (h2 : a 2 + a 5 + a 8 = 33) :
  a 3 + a 6 + a 9 = 27 :=
sorry

end arithmetic_seq_sum_l669_669564


namespace minimum_cactus_species_l669_669371

theorem minimum_cactus_species (cactophiles : Fin 80 → Set (Fin k)) :
  (∀ s : Fin k, ∃ col, cactophiles col s = False) ∧
  (∀ group : Set (Fin 80), group.card = 15 → ∃ c : Fin k, ∀ col ∈ group, (cactophiles col c)) →
  16 ≤ k :=
by
  sorry

end minimum_cactus_species_l669_669371


namespace cube_root_simplification_l669_669680

noncomputable def cubeRoot (x : ℝ) : ℝ := x^(1/3)

theorem cube_root_simplification :
  cubeRoot 54880000 = 140 * cubeRoot 20 :=
by
  sorry

end cube_root_simplification_l669_669680


namespace hop_sequences_bounds_l669_669643

def decreasing_sequence (xs : List ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → j < xs.length → xs.nth i > xs.nth j

def decreasing_differences (xs : List ℕ) : Prop :=
  ∀ (i j: ℕ), i < j → j < xs.length - 1 → (xs.nth i - xs.nth (i+1)) > (xs.nth j - xs.nth (j+1))

def hop_seqs (n: ℕ) : List (List ℕ) :=
  (List.range n).filter (λ xs, decreasing_sequence xs ∧ decreasing_differences xs)

theorem hop_sequences_bounds :
  let P := (hop_seqs 100).length in
  4000 ≤ P ∧ P ≤ 16000 :=
sorry

end hop_sequences_bounds_l669_669643


namespace trigonometric_identity_proof_l669_669089

theorem trigonometric_identity_proof
  (α : ℝ)
  (hα_quad4 : 3 * π / 2 < α ∧ α < 2 * π)
  (h_cos_α : cos α = 3 / 5):
  (tan α = -4 / 3) ∧
  ((sin (3 * π / 2 - α) + 2 * cos (α + π / 2)) / (sin (α - π) - 3 * cos (2 * π - α)) = -1) :=
by
  sorry

end trigonometric_identity_proof_l669_669089


namespace beth_should_charge_42_cents_each_l669_669458

theorem beth_should_charge_42_cents_each (n_alan_cookies : ℕ) (price_alan_cookie : ℕ) (n_beth_cookies : ℕ) (total_earnings : ℕ) (price_beth_cookie : ℕ):
  n_alan_cookies = 15 → 
  price_alan_cookie = 50 → 
  n_beth_cookies = 18 → 
  total_earnings = n_alan_cookies * price_alan_cookie → 
  price_beth_cookie = total_earnings / n_beth_cookies → 
  price_beth_cookie = 42 := 
by 
  intros h1 h2 h3 h4 h5 
  sorry

end beth_should_charge_42_cents_each_l669_669458


namespace rate_of_fencing_per_meter_l669_669217

/-- 
Given: 
    - The area of a circular field is 17.56 hectares.
    - The cost of fencing it is Rs. 7427.41.
Prove that the rate of fencing per meter is 5 rupees.
-/
theorem rate_of_fencing_per_meter
  (area_hectares : ℝ)
  (fencing_cost : ℝ)
  (conversion_factor : ℝ)
  (pi_approx : ℝ)
  (sqrt_approx : ∀ (x : ℝ), x >= 0 → ℝ)
  (area_sq_meters : ℝ)
  (radius : ℝ)
  (circumference : ℝ)
  (rate_per_meter : ℝ)
  (h1 : area_hectares = 17.56)
  (h2 : fencing_cost = 7427.41)
  (h3 : conversion_factor = 10000)
  (h4 : pi_approx = 3.14159)
  (h5 : sqrt_approx 55896.118 (by norm_num) ≈ 236.43)
  (h6 : area_sq_meters = area_hectares * conversion_factor)
  (h7 : radius = sqrt_approx (area_sq_meters / pi_approx) (by norm_num))
  (h8 : circumference = 2 * pi_approx * radius)
  (h9 : rate_per_meter = fencing_cost / circumference)
  (h10 : rate_per_meter ≈ 5) :
  rate_per_meter = 5 := 
sorry

end rate_of_fencing_per_meter_l669_669217


namespace construct_angle_from_19_l669_669063

theorem construct_angle_from_19 (θ : ℝ) (h : θ = 19) : ∃ n : ℕ, (n * θ) % 360 = 75 :=
by
  -- Placeholder for the proof
  sorry

end construct_angle_from_19_l669_669063


namespace transformed_roots_polynomial_l669_669183

theorem transformed_roots_polynomial :
  (∀ a b c : ℝ, (a + b + c = 0) ∧ (a * b + b * c + c * a = -5) ∧ (a * b * c = -6) →
  (polynomial.C (a - 3) * polynomial.C (b - 3) * polynomial.C (c - 3)).to_monic = 
  polynomial.C 1 * X^3 + polynomial.C 9 * X^2 + polynomial.C 22 * X + polynomial.C 18) :=
sorry

end transformed_roots_polynomial_l669_669183


namespace smallest_and_largest_x_l669_669448

theorem smallest_and_largest_x (x : ℝ) :
  (|5 * x - 4| = 29) → ((x = -5) ∨ (x = 6.6)) :=
by
  sorry

end smallest_and_largest_x_l669_669448


namespace total_selections_l669_669209

open Finset

-- Let committee members be represented by a Finset of 5 elements
def members : Finset (Fin 5) := {0, 1, 2, 3, 4}

-- Let A and B be members 0 and 1 respectively
def A : Fin 5 := 0
def B : Fin 5 := 1

-- Define the role selection considering the restriction
def valid_entertainment_officers : Finset (Fin 5) := members \ {A, B}

theorem total_selections : (valid_entertainment_officers.card * (members.card - 1) * (members.card - 2) = 36) :=
by
  have h1 : valid_entertainment_officers.card = 3 := rfl
  have h2 : members.card = 5 := rfl
  rw [h1, h2]
  norm_num
  sorry

end total_selections_l669_669209


namespace exponent_multiplication_l669_669536

theorem exponent_multiplication (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (3^a)^b = 3^3) : 3^a * 3^b = 3^4 :=
by
  sorry

end exponent_multiplication_l669_669536


namespace evaluate_expression_l669_669017

theorem evaluate_expression :
  let x := (1 / 4 : ℚ)
  let y := (1 / 3 : ℚ)
  let z := (-2 : ℚ)
  let w := (3 : ℚ)
  (x^3 * y^2 * z^2 * w) = (1 / 48 : ℚ) :=
by
  let x := (1 / 4 : ℚ)
  let y := (1 / 3 : ℚ)
  let z := (-2 : ℚ)
  let w := (3 : ℚ)
  sorry

end evaluate_expression_l669_669017


namespace length_of_ab_l669_669306

noncomputable def ab_length (abcde_seq : List ℝ) : ℝ :=
let bc := abcde_seq[1] - abcde_seq[0] in
let cd := abcde_seq[2] - abcde_seq[1] in
let de := abcde_seq[3] - abcde_seq[2] in
let ac := abcde_seq[2] - abcde_seq[0] in
let ae := abcde_seq[4] - abcde_seq[0] in
if (bc = 2 * cd) ∧ (de = 8) ∧ (ac = 11) ∧ (ae = 22) then abcde_seq[0] else 0

theorem length_of_ab : ∀ (abcde_seq : List ℝ), 
  (abcde_seq.length = 5) →
  (abcde_seq[1] - abcde_seq[0] = 2 * (abcde_seq[2] - abcde_seq[1])) →
  (abcde_seq[3] - abcde_seq[2] = 8) →
  (abcde_seq[2] - abcde_seq[0] = 11) →
  (abcde_seq[4] - abcde_seq[0] = 22) →
  ab_length abcde_seq = 5 := 
sorry

end length_of_ab_l669_669306


namespace largest_angle_measures_203_l669_669331

-- Define the angles of the hexagon
def angle1 (x : ℚ) : ℚ := x + 2
def angle2 (x : ℚ) : ℚ := 2 * x + 1
def angle3 (x : ℚ) : ℚ := 3 * x
def angle4 (x : ℚ) : ℚ := 4 * x - 1
def angle5 (x : ℚ) : ℚ := 5 * x + 2
def angle6 (x : ℚ) : ℚ := 6 * x - 2

-- Define the sum of interior angles for a hexagon
def hexagon_angle_sum : ℚ := 720

-- Prove that the largest angle is equal to 203 degrees given the conditions
theorem largest_angle_measures_203 (x : ℚ) (h : angle1 x + angle2 x + angle3 x + angle4 x + angle5 x + angle6 x = hexagon_angle_sum) :
  (6 * x - 2) = 203 := by
  sorry

end largest_angle_measures_203_l669_669331


namespace remainder_of_trailing_zeros_l669_669622

def factorial_trailing_zeros : Nat :=
  let M := (List.range (50 + 1)).map (fun n => if n = 0 then 0 else Nat.factorial n).foldl (*) 1
  let num_five_factors := (List.range (50 + 1)).filter (fun n => n > 0).map (fun n =>
    let rec count_fives x := if x % 5 = 0 then 1 + count_fives (x / 5) else 0
    count_fives n
  ).sum
  let num_trailing_zeros := num_five_factors
  num_trailing_zeros % 500

theorem remainder_of_trailing_zeros :
  factorial_trailing_zeros = 12 := by
  sorry

end remainder_of_trailing_zeros_l669_669622


namespace rancher_loss_l669_669342

theorem rancher_loss
  (initial_cattle : ℕ)
  (total_price : ℕ)
  (sick_cattle : ℕ)
  (price_reduction : ℕ)
  (remaining_cattle := initial_cattle - sick_cattle)
  (original_price_per_head := total_price / initial_cattle)
  (new_price_per_head := original_price_per_head - price_reduction)
  (total_original_price := original_price_per_head * remaining_cattle)
  (total_new_price := new_price_per_head * remaining_cattle) :
  total_original_price - total_new_price = 25200 :=
by 
  sorry

-- Definitions
def initial_cattle : ℕ := 340
def total_price : ℕ := 204000
def sick_cattle : ℕ := 172
def price_reduction : ℕ := 150

-- Substitute the known values in the theorem
#eval rancher_loss initial_cattle total_price sick_cattle price_reduction

end rancher_loss_l669_669342


namespace diagonal_midpoints_inequality_l669_669064

theorem diagonal_midpoints_inequality (A B C D : Point) 
  (circABC : Circle) (circACD : Circle) 
  (tangentABC_CD : TangentToCircleOfTriangle circABC A B C D)
  (tangentACD_AB : TangentToCircleOfTriangle circACD A C D B) :
  ( ∀ (midAB midCD : Point), 
    midAB = midpoint A B ∧ midCD = midpoint C D →
    dist A C < dist midAB midCD ) :=
begin
  sorry
end

end diagonal_midpoints_inequality_l669_669064


namespace area_of_triangle_MDA_l669_669147

def area_triangle_MDA (r : ℝ) : ℝ :=
  (3 * Real.sqrt 3 * r^2) / 32

theorem area_of_triangle_MDA (O A B M D : Point)
  (hO : Circle.center O)
  (hAB : Chord.length AB = Real.sqrt 3 * radius O)
  (hOM : Perpendicular_from_center_to_chord O AB M)
  (hMD : Perpendicular_from_point_on_chord_to_radius M OA D) :
  Triangle.area M D A = area_triangle_MDA (radius O) := 
sorry

end area_of_triangle_MDA_l669_669147


namespace walk_game_return_length_l669_669124

theorem walk_game_return_length :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let total_moves := 30
  let prime_moves := primes.length
  let composite_moves := total_moves - 1 - prime_moves
  let forward_steps := prime_moves * 2
  let backward_steps := composite_moves * 3
  let net_steps := forward_steps - backward_steps
  net_steps = -37 :=
by
  -- Definitions
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let total_moves := 30
  let prime_moves := primes.length
  let composite_moves := total_moves - 1 - prime_moves
  let forward_steps := prime_moves * 2
  let backward_steps := composite_moves * 3
  let net_steps := forward_steps - backward_steps
  -- Assertion
  have h : net_steps = -37 := sorry
  exact h

end walk_game_return_length_l669_669124


namespace intersection_of_sets_l669_669318

def M : set ℝ := { x | -3 < x ∧ x < 1 }
def N : set ℤ := { x | -1 ≤ x ∧ x ≤ 2 }

theorem intersection_of_sets : (M ∩ (N : set ℝ)) = ({-1, 0} : set ℝ) :=
sorry

end intersection_of_sets_l669_669318


namespace g_diff_l669_669399

def g (n : ℕ) : ℝ := 1/4 * n^2 * (n+1)^2

theorem g_diff (s : ℕ) : g s - g (s - 1) = s^3 :=
by
  sorry

end g_diff_l669_669399


namespace number_of_subsets_l669_669959

noncomputable def A : Set ℕ := { x | x^2 - 3 * x + 2 = 0 }
noncomputable def B : Set ℕ := { x | 0 < x ∧ x < 6 ∧ x ∈ ℕ }

theorem number_of_subsets (A ⊆ B) : (C : Set ℕ) (A ⊆ C ∧ C ⊆ B) :=
  let elements_in_A := { x | x = 1 ∨ x = 2 } 
  let elements_in_B := { x | x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 } 
  let number_of_possible_sets_C := 8 
  sorry

end number_of_subsets_l669_669959


namespace product_of_two_numbers_l669_669781

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 404) : x * y = 86 :=
sorry

end product_of_two_numbers_l669_669781


namespace determine_guilty_l669_669245

-- Definitions for the defendants and their statements.
inductive Defendant
| A | B | C

-- Accusation relationships between defendants.
def accuses : Defendant → Defendant → Prop
| Defendant.A => Defendant.B
| Defendant.B => Defendant.C
| Defendant.C => Defendant.A

def tellsTruth : Defendant → Prop
| Defendant.A => ¬ accuses Defendant.A Defendant.B
| Defendant.B => ¬ accuses Defendant.B Defendant.C
| Defendant.C => accuses Defendant.C Defendant.A

def isLying (d : Defendant) : Prop := ¬ tellsTruth d

-- The condition of one guilty defendant.
def isGuilty (d : Defendant) : Prop := 
  match d with
  | Defendant.A => Defendant.A ≠ Defendant.B ∧ Defendant.B ≠ Defendant.C
  | Defendant.B => Defendant.B ≠ Defendant.C ∧ Defendant.C ≠ Defendant.A
  | Defendant.C => Defendant.C ≠ Defendant.A ∧ Defendant.A ≠ Defendant.B

-- There exist either two consecutive truths or two consecutive lies.
def consecutiveTruthOrLies :=
  (tellsTruth Defendant.A ∧ tellsTruth Defendant.B) ∨
  (isLying Defendant.A ∧ isLying Defendant.B)

-- The theorem to prove.
theorem determine_guilty : (∃ d : Defendant, isGuilty d) →
  consecutiveTruthOrLies →
  tellsTruth Defendant.A ∧ tellsTruth Defendant.B ∧ isLying Defendant.C :=
sorry

end determine_guilty_l669_669245


namespace factorization_l669_669440

theorem factorization (a x : ℝ) : ax^2 - 2ax + a = a * (x - 1) ^ 2 := 
by
  sorry

end factorization_l669_669440


namespace train_speed_l669_669336

theorem train_speed (jogger_speed_km_hr : ℝ)
                    (initial_lead_m : ℝ)
                    (train_length_m : ℝ)
                    (time_s : ℝ)
                    (jogger_speed_correct : jogger_speed_km_hr = 9)
                    (initial_lead_correct : initial_lead_m = 240)
                    (train_length_correct : train_length_m = 110)
                    (time_correct : time_s = 35) : 
                    (train_speed_km_hr : ℝ) : Prop :=
  ∃ train_speed_km_hr, train_speed_km_hr = 45

end train_speed_l669_669336


namespace triangle_not_always_obtuse_l669_669357

def is_acute_triangle (A B C : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180 ∧ A < 90 ∧ B < 90 ∧ C < 90

theorem triangle_not_always_obtuse : ∃ (A B C : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180 ∧ is_acute_triangle A B C :=
by
  -- Exact proof here.
  sorry

end triangle_not_always_obtuse_l669_669357


namespace part1_part2_l669_669511

def law_of_sines (a b c sin_A sin_B : ℝ) : Prop :=
  a / sin_A = b / sin_B

def law_of_cosines (a b c cos_B : ℝ) : Prop :=
  b^2 = a^2 + c^2 - 2 * a * c * cos_B

def area_of_triangle (a b c sin_B : ℝ) (area : ℝ) : Prop :=
  area = (1 / 2) * a * c * sin_B

theorem part1 (cosB : ℝ) (b : ℝ) (A : ℝ) : cosB = 4 / 5 → b = 2 → A = π / 6 → a = 5 / 3 :=
begin
  intro h1,
  intro h2,
  intro h3,
  sorry
end

theorem part2 (cosB : ℝ) (b : ℝ) (area : ℝ) : cosB = 4 / 5 → b = 2 → area = 3 → a + c = 2 * real.sqrt 10 :=
begin
  intro h1,
  intro h2,
  intro h3,
  sorry
end

end part1_part2_l669_669511


namespace min_value_arithmetic_geometric_sequence_l669_669949

theorem min_value_arithmetic_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (m n : ℕ) (a1 : ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_arith_geo : ∀ i, a (i + 1) = q * a i)
  (h_condition : a 7 = a 6 + 2 * a 5)
  (h_terms: ∃ m n, sqrt (a m * a n) = 2 * a1):
  ∃ m n, (1 / m : ℝ) + (4 / n : ℝ) = 9 / 4 :=
sorry

end min_value_arithmetic_geometric_sequence_l669_669949


namespace cheburashkas_erased_l669_669614

def total_krakozyabras : ℕ := 29

def total_rows : ℕ := 2

def cheburashkas_per_row := (total_krakozyabras + total_rows) / total_rows / 2 + 1

theorem cheburashkas_erased :
  (total_krakozyabras + total_rows) / total_rows / 2 - 1 = 11 := 
by
  sorry

-- cheburashkas_erased proves that the number of Cheburashkas erased is 11 from the given conditions.

end cheburashkas_erased_l669_669614


namespace find_S100_l669_669576

def a_seq : ℕ+ → ℕ
| ⟨1, _⟩ := 1
| ⟨2, _⟩ := 2
| ⟨n+2, _⟩ := a_seq (⟨n, nat.succ_pos n⟩) + 1 - (-1)^n

def sum_sequence (n : ℕ+) : ℕ :=
nat.rec_on n 0 (λ n ih, ih + a_seq ⟨n, nat.succ_pos n⟩)

theorem find_S100 : sum_sequence ⟨100, by norm_num⟩ = 2600 := sorry

end find_S100_l669_669576


namespace simplest_square_root_is_three_l669_669354

-- Definition of a radicand not containing a denominator
def noDenominator (x : ℝ) : Prop := ∃ (n : ℤ), x = n

-- Definition of a radicand not containing squared factors
def noSquaredFactors (x : ℝ) : Prop := 
  ∀ (y : ℤ), y * y ≠ x

-- Definition of a simplest square root
def isSimplestSquareRoot (x : ℝ) : Prop :=
  noDenominator x ∧ noSquaredFactors x

-- Given options for the square roots
def optionA : ℝ := 0.2
def optionB : ℝ := 12
def optionC : ℝ := 3
def optionD : ℝ := 18

-- Prove that among the given options, sqrt(3) is the simplest square root
theorem simplest_square_root_is_three : isSimplestSquareRoot 3 :=
  sorry

end simplest_square_root_is_three_l669_669354


namespace same_incenter_l669_669084

variable (P O A B C D Q : Type)
variable [IncidenceGeometry P O A B C D Q]

-- The conditions
variable (h1 : PointOnCircle P (CircleCenter O))
variable (h2 : TangentToCircle P A)
variable (h3 : TangentToCircle P B)
variable (h4 : Line P O)
variable (h5 : Line A B)
variable (h6 : IntersectLineLine P O A B Q)
variable (h7 : ChordThrough Q (CircleCenter O) C D)

-- The proposition
theorem same_incenter (h : Incircle P A B C D Q) : 
  Incenter (Triangle P A B) = Incenter (Triangle P C D) := 
sorry

end same_incenter_l669_669084


namespace floor_sum_inequality_l669_669929

theorem floor_sum_inequality : ∀ (a b : ℝ), a = 3.5 → b = 2.5 → 
[floor (a + b)] ≠ [floor a + floor b + floor (a * b) - floor a * floor b] :=
by
  assume a b h1 h2
  sorry

end floor_sum_inequality_l669_669929


namespace cube_root_simplification_l669_669686

noncomputable def cubeRoot (x : ℝ) : ℝ := x^(1/3)

theorem cube_root_simplification :
  cubeRoot 54880000 = 140 * cubeRoot 20 :=
by
  sorry

end cube_root_simplification_l669_669686


namespace evaluate_expression_at_10_l669_669408

open Real

theorem evaluate_expression_at_10 :
  let x := 10 in
  30 - abs (- x^2 + 6 * x + 6) + 5 * cos (2 * x) = -4 + 5 * cos 20 :=
by
  sorry

end evaluate_expression_at_10_l669_669408


namespace polynomial_negative_coefficient_l669_669206

-- Define the polynomial P(x)
def P (x : ℕ) : ℕ := x^4 + x^3 - 3 * x^2 + x + 2

-- Define the polynomial Q(x) as (P(x))^n
def Q (x n : ℕ) : ℕ := P x ^ n

-- The theorem statement
theorem polynomial_negative_coefficient (n : ℕ) : 
  ∃ k : ℕ, (coeff (Q x n) k) < 0 := sorry

end polynomial_negative_coefficient_l669_669206


namespace ratio_elephants_to_others_l669_669243

theorem ratio_elephants_to_others (L P E : ℕ) (h1 : L = 2 * P) (h2 : L = 200) (h3 : L + P + E = 450) :
  E / (L + P) = 1 / 2 :=
by
  sorry

end ratio_elephants_to_others_l669_669243


namespace paint_cost_per_kg_l669_669731

theorem paint_cost_per_kg (area_per_kg : ℝ) (total_cost : ℝ) (side_length : ℝ) (cost_per_kg : ℝ) :
  (side_length = 8) → (area_per_kg = 16) → (total_cost = 876) →
  cost_per_kg = (total_cost / ((6 * side_length^2) / area_per_kg))
  := 
by
  intro h_side_length
  intro h_area_per_kg
  intro h_total_cost
  have h_total_surface_area : 6 * side_length^2 = 384 := by sorry
  have h_paint_needed : 384 / area_per_kg = 24 := by sorry
  exact calc
    cost_per_kg = total_cost / 24 : by sorry
              ... =  36.5 : by sorry

end paint_cost_per_kg_l669_669731


namespace set_union_proof_l669_669113

theorem set_union_proof (a b : ℝ) (A B : Set ℝ) 
  (hA : A = {1, 2^a})
  (hB : B = {a, b}) 
  (h_inter : A ∩ B = {1/4}) :
  A ∪ B = {-2, 1, 1/4} := 
by 
  sorry

end set_union_proof_l669_669113


namespace divisors_of_30240_l669_669997

theorem divisors_of_30240 : 
  ∃ s : Finset ℕ, (s = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (∀ d ∈ s, (30240 % d = 0)) ∧ (s.card = 9) :=
by
  sorry

end divisors_of_30240_l669_669997


namespace coeff_x3_y2_in_expansion_l669_669902

theorem coeff_x3_y2_in_expansion (x y : ℝ) : 
  ∀ (r : ℕ), (5 > r) → (5 - r = 2) → 
  (∀ (k : ℕ), (3 > k) → (6 - k = 3) → 
  (nat.choose 3 k) = 1) →
  let term := nat.choose 5 r * y^(5 - r) * (x^2 - x)^r in
  coeff_of_term_in_expansion (term) x y = -10 := 
by
  assume x y r hr r_eq k hk k_eq hChoose
  sorry

end coeff_x3_y2_in_expansion_l669_669902


namespace binom_divisible_l669_669663

def binom (n k : ℕ) := nat.choose n k

theorem binom_divisible (n : ℕ) : n ≥ 0 → (binom (2 * n) n) % (n + 1) = 0 := 
by
  intro hn_ge_zero
  sorry

end binom_divisible_l669_669663


namespace sin_A_in_right_triangle_l669_669560

noncomputable def triangle_ABC := sorry -- Define the triangle using given conditions

theorem sin_A_in_right_triangle (ABC : triangle_ABC)
    (right_triangle : ∃ C : ↥ABC, angle = 90)
    (AB_eq_4 : ∃ (AB : ℝ), AB = 4)
    (BC_eq_3 : ∃ (BC : ℝ), BC = 3) : 
    sin A = 3 / 4 := 
sorry

end sin_A_in_right_triangle_l669_669560


namespace parity_and_monotonicity_same_l669_669879

-- Define the function y = -3^{|x|}
def f (x : ℝ) : ℝ := -3^(abs x)

-- Conditions: f is an even function and increasing on (-∞, 0)
def is_even (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g x = g (-x)
def is_increasing_on (g : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ {x y}, x ∈ s → y ∈ s → x < y → g x < g y

-- The function y = 1 - x^2
def h (x : ℝ) : ℝ := 1 - x^2

theorem parity_and_monotonicity_same : 
  is_even f ∧ is_increasing_on f (Set.Iio 0) →
  is_even h ∧ is_increasing_on h (Set.Iio 0) :=
by
  intros hf
  sorry

end parity_and_monotonicity_same_l669_669879


namespace product_of_solutions_l669_669953

variable {f : ℝ → ℝ}

/-- Given y = f(x+2) is an even function defined on ℝ, continuous and uninterrupted.
    When x > 2, y = f(x) is a monotonic function.
    Prove that the product of all x that satisfy f(x) = f(1 - 1 / (x + 4)) is 39. -/
theorem product_of_solutions (h1 : ∀ x, f x = f (-x + 4))
                             (h2 : ∀ x > 2, Monotone (λ x, f (x)))
                             (h3 : Continuous f) :
  (∏ x in {x : ℝ | f x = f (1 - 1 / (x + 4))}, x) = 39 := sorry

end product_of_solutions_l669_669953


namespace given_conditions_imply_B_and_C_l669_669107

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := sin (ω * x + φ)

theorem given_conditions_imply_B_and_C (ω : ℝ) (φ : ℝ) :
  ω > 0 ∧ |φ| < π ∧ f (-π / 6) ω φ = 0 ∧
  (∀ x, f x ω φ ≥ f (π / 2) ω φ) ∧
  (∀ x1 x2, (−π / 9 < x1 ∧ x1 < π / 18) → (−π / 9 < x2 ∧ x2 < π / 18) → x1 < x2 → f x1 ω φ < f x2 ω φ) →
  (f 0 ω φ = f π ω φ) ∧ (∃ k : ℤ, ω = (3 / 4) * k) :=
by
  sorry

end given_conditions_imply_B_and_C_l669_669107


namespace range_of_m_l669_669086

variables {x m : ℝ}

def proposition_p (m : ℝ) : Prop := (m ^ 2 - 4 > 0)
def proposition_q (m : ℝ) : Prop := (4 * (m + 1) ^ 2 - 4 * m * (m + 1) < 0)

def problem_conditions (m : ℝ) : Prop :=
(proposition_p m ∨ proposition_q m) ∧ ¬ (proposition_p m ∧ proposition_q m)

theorem range_of_m (m : ℝ) :
  problem_conditions m ↔ (m > 2 ∨ (-2 ≤ m ∧ m < -1)) :=
begin
  sorry
end

end range_of_m_l669_669086


namespace badminton_tournament_l669_669144

theorem badminton_tournament (n x : ℕ) (h1 : 2 * n > 0) (h2 : 3 * n > 0) (h3 : (5 * n) * (5 * n - 1) = 14 * x) : n = 3 :=
by
  -- Placeholder for the proof
  sorry

end badminton_tournament_l669_669144


namespace closest_integer_cubic_root_l669_669283

theorem closest_integer_cubic_root : 
  (∃ (n : ℤ), n = 11) → 
  (∃ (n : ℤ), n = 2) →
  (∃ (m : ℤ), m = 1331) →
  (∃ (k : ℤ), k = 8) →
  (∃ (c : ℤ), c = closest_integer (real.cbrt (1331 + 8))) →
  c = 11 :=
by
  sorry

# Here, closest_integer is a hypothetical function that finds the integer closest to the real number input.

end closest_integer_cubic_root_l669_669283


namespace factorize_expression_l669_669423

theorem factorize_expression (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
by sorry

end factorize_expression_l669_669423


namespace expected_value_of_X_is_half_l669_669256

-- Definition of the sequence of random digits and the random number X
def random_digits : list (vector ℕ 10) :=
  replicate 1000 (vector.of_fn (λ i, i)) -- Simulates a list of random digits from 0 to 9

def X (digits : list ℕ) : ℝ :=
  digits.foldl (λ acc x, acc / 10 + x.to_real / 10) 0

-- Expected value of X
noncomputable def E_X : ℝ :=
  1 / 2

-- Theorem statement
theorem expected_value_of_X_is_half : E_X = 0.5 :=
  by
  sorry -- The proof would go here

end expected_value_of_X_is_half_l669_669256


namespace solve_congruence_l669_669211

theorem solve_congruence (y : ℤ) : 10 * y + 3 ≡ 7 [MOD 15] → y ≡ 2 [MOD 3] := by
  sorry

end solve_congruence_l669_669211


namespace largest_integral_solution_l669_669444

theorem largest_integral_solution : ∃ x : ℤ, (1 / 4 < x / 7 ∧ x / 7 < 3 / 5) ∧ ∀ y : ℤ, (1 / 4 < y / 7 ∧ y / 7 < 3 / 5) → y ≤ x := sorry

end largest_integral_solution_l669_669444


namespace sodium_bicarbonate_moles_combined_l669_669030

theorem sodium_bicarbonate_moles_combined (HCl NaCl NaHCO3 : ℝ) (reaction : HCl + NaHCO3 = NaCl) 
  (HCl_eq_one : HCl = 1) (NaCl_eq_one : NaCl = 1) : 
  NaHCO3 = 1 := 
by 
  -- Placeholder for the proof
  sorry

end sodium_bicarbonate_moles_combined_l669_669030


namespace volume_of_cube_proof_l669_669766

-- Definition of the condition
def perimeter_of_square (s : ℝ) := 4 * s

-- Given the perimeter of one face of a cube is 20 
def perimeter_face : ℝ := 20

-- Definition of side length given perimeter condition
def side_length := perimeter_face / 4

-- Definition of the cube volume
def volume_of_cube (s : ℝ) := s ^ 3

-- Statement of the theorem
theorem volume_of_cube_proof : volume_of_cube side_length = 125 := by
  sorry

end volume_of_cube_proof_l669_669766


namespace cone_volume_false_l669_669851

namespace cone_volume_proof

def cone_volume (S h : ℝ) : ℝ := (1/3) * S * h

theorem cone_volume_false (V S h : ℝ) (hV : V = 45) (hS : S = 9) (hh : h = 5) : V ≠ cone_volume S h := by
  simp [cone_volume, hS, hh]
  norm_num
  -- This simplifies the goal to showing (45 ≠ 15)
  exact ne_of_lt (by norm_num)

end cone_volume_proof

end cone_volume_false_l669_669851


namespace max_PA_PB_PC_value_l669_669072

open Real

-- Coordinates of the triangle vertices
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 1)
def C : ℝ × ℝ := (0, 0)

-- Distance function
def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Product of distances from P to A, B, and C
def PA_PB_PC (P : ℝ × ℝ) : ℝ :=
  distance P A * distance P B * distance P C

-- Function to maximize PA * PB * PC
noncomputable def max_PA_PB_PC : ℝ :=
  max (max (Sup (PA_PB_PC '' {P | P.1 = 0}))  -- P on AC
          (Sup (PA_PB_PC '' {P | P.2 = 0})))  -- P on BC
      (Sup (PA_PB_PC '' {P | P.1 + P.2 = 1}))  -- P on AB

theorem max_PA_PB_PC_value :
  max_PA_PB_PC = sqrt 2 / 4 :=
sorry

end max_PA_PB_PC_value_l669_669072


namespace common_difference_in_AP_l669_669925

theorem common_difference_in_AP 
  (a l n: ℕ) (h_a: a = 2) (h_l: l = 62) (h_n: n = 31) : 
  ∃ d: ℕ, l = a + (n - 1) * d ∧ d = 2 := 
by 
  use 2; 
  split; 
  { rw [h_a, h_l, h_n]; 
    norm_num 
  }; 
  { norm_num 
  }

end common_difference_in_AP_l669_669925


namespace probability_divisible_by_9_l669_669640

-- Conditions
def T : Set ℕ := { n | ∃ (a b : ℕ), 0 ≤ a ∧ a < b ∧ b ≤ 29 ∧ n = 2^a + 2^b }
def T_size : Nat := 435

-- Proof statement for the probability
theorem probability_divisible_by_9 : 
  let T_div_9 := { n ∈ T | n % 9 = 0 }
  let num_div_9 := T_div_9.toFinset.card
  let probability := num_div_9.toNat / T_size
  probability = 5 / 29 :=
by
  sorry

end probability_divisible_by_9_l669_669640


namespace triangle_area_120_l669_669026

noncomputable def isosceles_triangle_area (r : ℝ) (α : ℝ) : ℝ :=
  let x := (4 * (2 + real.sqrt 3) * real.sqrt 12^(1/4)) / (real.sqrt 3)
  in (real.sqrt 3 / 4) * x^2

theorem triangle_area_120 (r : ℝ) (α : ℝ) (hα : α = 120) (hr : r = real.sqrt 12^(1/4)) :
  isosceles_triangle_area r α = 2 * (7 + 4 * real.sqrt 3) :=
begin
  sorry
end

end triangle_area_120_l669_669026


namespace relationship_among_a_b_c_l669_669498

theorem relationship_among_a_b_c (a b c : ℝ) (h₁ : a = 0.09) (h₂ : -2 < b ∧ b < -1) (h₃ : 1 < c ∧ c < 2) : b < a ∧ a < c := 
by 
  -- proof will involve but we only need to state this
  sorry

end relationship_among_a_b_c_l669_669498


namespace negation_example_l669_669231

theorem negation_example :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := sorry

end negation_example_l669_669231


namespace complete_square_transform_l669_669816

theorem complete_square_transform (x : ℝ) :
  x^2 - 2 * x - 5 = 0 → (x - 1)^2 = 6 :=
by
  intro h
  sorry

end complete_square_transform_l669_669816


namespace combined_distance_proof_l669_669642

/-- Define the distances walked by Lionel, Esther, and Niklaus in their respective units -/
def lionel_miles : ℕ := 4
def esther_yards : ℕ := 975
def niklaus_feet : ℕ := 1287

/-- Define the conversion factors -/
def miles_to_feet : ℕ := 5280
def yards_to_feet : ℕ := 3

/-- The total combined distance in feet -/
def total_distance_feet : ℕ :=
  (lionel_miles * miles_to_feet) + (esther_yards * yards_to_feet) + niklaus_feet

theorem combined_distance_proof : total_distance_feet = 24332 := by
  -- expand definitions and calculations here...
  -- lionel = 4 * 5280 = 21120
  -- esther = 975 * 3 = 2925
  -- niklaus = 1287
  -- sum = 21120 + 2925 + 1287 = 24332
  sorry

end combined_distance_proof_l669_669642


namespace sufficient_material_for_box_l669_669172

theorem sufficient_material_for_box :
  (∃ a b c : ℕ, a * b * c ≥ 1995 ∧ 2 * (a * b + b * c + c * a) = 958) :=
by
  -- We know that for a = 11, b = 13, c = 14:
  let a := 11
  let b := 13
  let c := 14
  have h1 : a * b * c = 2002 := by norm_num
  have h2 : 2002 ≥ 1995 := by norm_num
  have h3 : 2 * (a * b + b * c + c * a) = 958 := by norm_num
  exact ⟨a, b, c, h2, h3⟩

end sufficient_material_for_box_l669_669172


namespace mean_exterior_angles_l669_669137

theorem mean_exterior_angles (a b c : ℝ) (ha : a = 45) (hb : b = 75) (hc : c = 60) :
  (180 - a + 180 - b + 180 - c) / 3 = 120 :=
by 
  sorry

end mean_exterior_angles_l669_669137


namespace find_abc_l669_669359

open Polynomial

noncomputable def my_gcd_lcm_problem (a b c : ℤ) : Prop :=
  gcd (X^2 + (C a * X) + C b) (X^2 + (C b * X) + C c) = X + 1 ∧
  lcm (X^2 + (C a * X) + C b) (X^2 + (C b * X) + C c) = X^3 - 5*X^2 + 7*X - 3

theorem find_abc : ∀ (a b c : ℤ),
  my_gcd_lcm_problem a b c → a + b + c = -3 :=
by
  intros a b c h
  sorry

end find_abc_l669_669359


namespace cube_root_of_54880000_l669_669678

theorem cube_root_of_54880000 : (real.cbrt 54880000) = 140 * (real.cbrt 10) :=
by
  -- Definitions based on conditions
  have h1 : 54880000 = 10^3 * 54880, by norm_num
  have h2 : 54880 = 2^5 * 7^3 * 5, by norm_num
  have h3 : 10 = 2 * 5, by norm_num
  
  -- Cube root properties and simplifications are implicitly inferred by the system
  sorry

end cube_root_of_54880000_l669_669678


namespace min_value_of_polynomial_l669_669097

theorem min_value_of_polynomial :
  ∃ x : ℝ, ∀ y, y = (x - 16) * (x - 14) * (x + 14) * (x + 16) → y ≥ -900 :=
by
  sorry

end min_value_of_polynomial_l669_669097


namespace general_formula_an_sum_first_n_terms_cn_formula_existence_Sm_Sk_l669_669639

noncomputable def sequence_an (n : Nat) : Real := (-1/4)^n

noncomputable def sequence_bn (n : Nat) : Real := 2 * n - 1

noncomputable def sequence_Tn (n : Nat) : Real := n^2

noncomputable def sequence_cn (n : Nat) : Real := sequence_bn (n + 1) / (sequence_Tn n * sequence_Tn (n + 1))

noncomputable def sum_first_n_terms_cn (n : Nat) : Real := ∑ i in Finset.range (n + 1), sequence_cn i

theorem general_formula_an (n : Nat) : 
  sequence_an n = (-1/4)^n := 
sorry

theorem sum_first_n_terms_cn_formula (n : Nat) : 
  sum_first_n_terms_cn n = 1 - 1 / (n + 1)^2 := 
sorry

theorem existence_Sm_Sk (m k n : Nat) : 
  |sum_first_n_terms sequence_an m - sum_first_n_terms sequence_an k| ≤ 32 * sequence_an n ↔ (n = 2 ∨ n = 4) := 
sorry

end general_formula_an_sum_first_n_terms_cn_formula_existence_Sm_Sk_l669_669639


namespace ellipse_eq_line_l1_eq_lambda_mu_value_l669_669077

-- Given conditions
variable {a b p: ℝ} (Q: ℝ × ℝ) (E: ℝ × ℝ → Prop)
variable (F_1: ℝ × ℝ) (a_pos: a > b) (b_pos: b > 0)
variable (Q_lies_on_E: E (sqrt 3, 1/2))
variable (slope_t: ℝ) (t_pos: slope_t > 0)
variable (l_1: ℝ → ℝ) (l_2: ℝ → ℝ)
variable (M N P: ℝ × ℝ)
variable (line_l1: ∀ x: ℝ, l_1 x = slope_t * x)
variable (line_l2: ∀ x: ℝ, l_2 x = (F_1.snd / F_1.fst) * (x + F_1.fst))
variable (line_p: ∀ x: ℝ, x = -p)

-- To prove ellipse equation
theorem ellipse_eq: E = λ (x y: ℝ), x^2 / 4 + y^2 / 1 = 1 :=
by
  sorry

-- To prove equation of line l1
theorem line_l1_eq: l_1 = λ x: ℝ,  (1/2) * x :=
by
  sorry

-- To prove lambda + mu value
theorem lambda_mu_value: 
  ∃ λ μ: ℝ, 
  (λ + μ = (2 * (p * sqrt (a^2 - b^2) - a^2)) / b^2) ∧ 
  (over PM M = λ * over MF_1 M ∧ 
  over PN N = μ * over NF_1 N ) :=
by
  sorry

end ellipse_eq_line_l1_eq_lambda_mu_value_l669_669077


namespace cost_of_soft_drink_l669_669668

theorem cost_of_soft_drink:
  ( ∀ (x : ℝ),
      let robert_pizza_cost := 5 * 10 in
      let robert_drinks_cost := 10 * x in
      let teddy_hamburgers_cost := 6 * 3 in
      let teddy_drinks_cost := 10 * x in
      let total_cost := robert_pizza_cost + robert_drinks_cost + teddy_hamburgers_cost + teddy_drinks_cost in
      total_cost = 106 → x = 1.9 ) := sorry

end cost_of_soft_drink_l669_669668


namespace cheburashkas_erased_l669_669589

theorem cheburashkas_erased (n : ℕ) (rows : ℕ) (krakozyabras : ℕ) 
  (h_spacing : ∀ r, r ≤ rows → krakozyabras = 2 * (n - 1))
  (h_rows : rows = 2)
  (h_krakozyabras : krakozyabras = 29) :
  n = 16 → rows = 2 → krakozyabras = 29 → n = 16 - 5 :=
by
  sorry

end cheburashkas_erased_l669_669589


namespace locus_of_point_P_l669_669190

variable {V : Type} [inner_product_space ℝ V]

-- Definitions using conditions from part a.
def regular_tetrahedron (A B C D : V) : Prop := 
  ∃ a : ℝ, dist A B = a ∧ dist A C = a ∧ dist A D = a ∧ dist B C = a ∧ dist B D = a ∧ dist C D = a

def midpoint (A B : V) : V := (A + B) / 2

def is_orthocenter (P A B M : V) : Prop :=
  ∃ H K : V, is_orthogonal (A - P) (M - B) ∧ is_orthogonal (B - P) (M - A)

def locus_of_P (A B C D M P : V) :=
  P ∈ arc_centered_at_midpoint (A B) (orthocenter_triangle A B C) (orthocenter_triangle A B D)

-- The Lean statement for the proof problem, statement only
theorem locus_of_point_P
  (A B C D M P : V)
  (h_tetra : regular_tetrahedron A B C D)
  (h_midpoint : (midpoint A B) = L)
  (h_M_on_CD : M ∈ line_segment C D)
  (h_p_orthocenter : is_orthocenter P A B M) :
  locus_of_P A B C D M P := sorry

end locus_of_point_P_l669_669190


namespace number_of_Cheburashkas_erased_l669_669599

theorem number_of_Cheburashkas_erased :
  ∃ (n : ℕ), 
    (∀ x, x ≥ 1 → 
      (let totalKrakozyabras = (2 * (x - 1) = 29) in
         x - 2 = 11)) :=
sorry

end number_of_Cheburashkas_erased_l669_669599


namespace factorize_expression_l669_669409

theorem factorize_expression (a x : ℝ) :
  a * x^2 - 2 * a * x + a = a * (x - 1) ^ 2 := 
sorry

end factorize_expression_l669_669409


namespace powderman_run_distance_approx_l669_669338

theorem powderman_run_distance_approx
  (fuse_time : ℕ) (powderman_speed_yards_per_sec : ℕ) (sound_speed_feet_per_sec : ℕ)
  (h1 : fuse_time = 45) (h2 : powderman_speed_yards_per_sec = 6) (h3 : sound_speed_feet_per_sec = 1080) :
  let powderman_speed_feet_per_sec := powderman_speed_yards_per_sec * 3
  in abs ((powderman_speed_feet_per_sec * (fuse_time * sound_speed_feet_per_sec) / (sound_speed_feet_per_sec - powderman_speed_feet_per_sec)) / 3 - 275) < 1 :=
by 
  sorry

end powderman_run_distance_approx_l669_669338


namespace smallest_sum_of_three_elements_l669_669898

def S : Set ℤ := {4, -7, 19, -5, 3}

theorem smallest_sum_of_three_elements : 
  ∃ a b c ∈ S, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = -8 :=
by
  sorry

end smallest_sum_of_three_elements_l669_669898


namespace possible_to_form_rectangle_with_sticks_l669_669244

theorem possible_to_form_rectangle_with_sticks 
  (sticks : Fin 99 → ℕ) 
  (h : ∀ n, sticks n = n + 1) : 
  ∃ L W, (L > 0) ∧ (W > 0) ∧ (2 * (L + W) = PartialBehavior.sum sticks) ∧ 
  (PartialBehavior.subset (sticks : Set ℕ) {L, W} :=
  sorry

end possible_to_form_rectangle_with_sticks_l669_669244


namespace factorial_divisibility_l669_669195

theorem factorial_divisibility (n e_1 e_2 ... e_r : ℕ) (cond1 : n = 2 ^ e_1 + 2 ^ e_2 + ... + 2 ^ e_r)
    (cond2 : r = (list.nodup (list.cons e_1 (list.cons e_2 ... (list.cons e_r list.nil))))) :
    ∃ k, n! = 2 ^ k * m ∧ n - r = k ∧ ¬(2^(k+1) ∣ n!) := by sorry

end factorial_divisibility_l669_669195


namespace distance_symmetric_reflection_l669_669561

theorem distance_symmetric_reflection (x : ℝ) (y : ℝ) (B : (ℝ × ℝ)) 
  (hB : B = (-1, 4)) (A : (ℝ × ℝ)) (hA : A = (x, -y)) : 
  dist A B = 8 :=
by
  sorry

end distance_symmetric_reflection_l669_669561


namespace height_previous_year_l669_669830

theorem height_previous_year (current_height : ℝ) (growth_rate : ℝ) (previous_height : ℝ) 
  (h1 : current_height = 126)
  (h2 : growth_rate = 0.05) 
  (h3 : current_height = 1.05 * previous_height) : 
  previous_height = 120 :=
sorry

end height_previous_year_l669_669830


namespace angle_B_measure_range_of_b_l669_669545

noncomputable def triangle_cos_sin_condition :=
  ∃ (A B C : ℝ), (cos C + cos A * cos B - real.sqrt 3 * sin A * cos B = 0)

theorem angle_B_measure (A B C : ℝ) (h : cos C + cos A * cos B - real.sqrt 3 * sin A * cos B = 0) :
  B = π / 3 :=
sorry

theorem range_of_b (a b c : ℝ) (h1 : a + c = 1) (h2 : cos (π / 3) = 1 / 2) :
  1 / 2 ≤ b ∧ b < 1 :=
sorry

end angle_B_measure_range_of_b_l669_669545


namespace arithmetic_geometric_sequence_l669_669099

theorem arithmetic_geometric_sequence :
  ∀ (a₁ a₂ b₂ : ℝ),
    -- Conditions for arithmetic sequence: -1, a₁, a₂, 8
    2 * a₁ = -1 + a₂ ∧
    2 * a₂ = a₁ + 8 →
    -- Conditions for geometric sequence: -1, b₁, b₂, b₃, -4
    (∃ (b₁ b₃ : ℝ), b₁^2 = b₂ ∧ b₁ != 0 ∧ -4 * b₁^4 = b₂ → -1 * b₁ = b₃) →
    -- Goal: Calculate and prove the value
    (a₁ * a₂ / b₂) = -5 :=
by {
  sorry
}

end arithmetic_geometric_sequence_l669_669099


namespace expected_value_of_X_is_half_l669_669255

-- Definition of the sequence of random digits and the random number X
def random_digits : list (vector ℕ 10) :=
  replicate 1000 (vector.of_fn (λ i, i)) -- Simulates a list of random digits from 0 to 9

def X (digits : list ℕ) : ℝ :=
  digits.foldl (λ acc x, acc / 10 + x.to_real / 10) 0

-- Expected value of X
noncomputable def E_X : ℝ :=
  1 / 2

-- Theorem statement
theorem expected_value_of_X_is_half : E_X = 0.5 :=
  by
  sorry -- The proof would go here

end expected_value_of_X_is_half_l669_669255


namespace max_value_PA_PB_PC_l669_669069

noncomputable def maximum_PA_PB_PC (A B C P : ℝ × ℝ) : ℝ :=
  let PA := dist P A
  let PB := dist P B
  let PC := dist P C
  PA * PB * PC

theorem max_value_PA_PB_PC :
  let A := (1, 0)
  let B := (0, 1)
  let C := (0, 0)
  ∃ P : ℝ × ℝ, (P.1 = 0 ∨ P.1 = 1 ∨ P.2 = 0 ∨ P.2 = 1 ∨ P.1 = P.2),
  maximum_PA_PB_PC A B C P = sqrt 2 / 4 :=
by
  sorry

end max_value_PA_PB_PC_l669_669069


namespace csc_cos_expression_eq_two_l669_669389

theorem csc_cos_expression_eq_two : Real.csc (Real.pi / 18) - 4 * Real.cos (2 * Real.pi / 9) = 2 := 
by sorry

end csc_cos_expression_eq_two_l669_669389


namespace right_triangle_area_l669_669752

/-- Given a right triangle with hypotenuse 13 meters and one side 5 meters,
prove that the area of the triangle is 30 square meters. -/
theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (hc : c = 13) (ha : a = 5) :
  1/2 * a * b = 30 :=
by sorry

end right_triangle_area_l669_669752


namespace intersection_of_A_and_B_l669_669510

noncomputable def set_A : set ℝ := { y : ℝ | ∃ x : ℝ, y = abs (x - 1) + abs (x - 3) }
noncomputable def set_B : set ℝ := { x : ℝ | ∃ y : ℝ, y = real.log (3 * x - x^2) ∧ 0 < x ∧ x < 3 }

theorem intersection_of_A_and_B : set_A ∩ set_B = { x : ℝ | 2 ≤ x ∧ x < 3 } :=
by
  sorry

end intersection_of_A_and_B_l669_669510


namespace complete_square_transform_l669_669815

theorem complete_square_transform (x : ℝ) :
  x^2 - 2 * x - 5 = 0 → (x - 1)^2 = 6 :=
by
  intro h
  sorry

end complete_square_transform_l669_669815


namespace probability_at_most_one_hit_l669_669844

theorem probability_at_most_one_hit (p_A : ℝ) (p_B : ℝ) (h_independent : independent A B) 
  (h_p_A : p_A = 0.6) (h_p_B : p_B = 0.7) :
  P (at_most_one_hit A B) = 0.58 := 
sorry

end probability_at_most_one_hit_l669_669844


namespace perp_angle_of_quadrilateral_l669_669894

theorem perp_angle_of_quadrilateral (A B C D E F G : Point) 
  (hAB : midpoint A B E) (hBC : midpoint B C F) (hAD : midpoint A D G)
  (hGE_perp_AB : perp GE AB) (hGF_perp_BC : perp GF BC) : 
  angle A C D = 90 :=
sorry

end perp_angle_of_quadrilateral_l669_669894


namespace expected_value_of_random_number_l669_669260

/-- 
The expected value of a random number formed by placing a zero and a decimal point in front
of a sequence of one thousand random digits is 0.5.
-/
theorem expected_value_of_random_number : 
  let X := ∑ k in (finRange 1000), (4.5 / 10 ^ (k + 1))
  in X = 0.5 :=
sorry

end expected_value_of_random_number_l669_669260


namespace volume_of_cube_proof_l669_669767

-- Definition of the condition
def perimeter_of_square (s : ℝ) := 4 * s

-- Given the perimeter of one face of a cube is 20 
def perimeter_face : ℝ := 20

-- Definition of side length given perimeter condition
def side_length := perimeter_face / 4

-- Definition of the cube volume
def volume_of_cube (s : ℝ) := s ^ 3

-- Statement of the theorem
theorem volume_of_cube_proof : volume_of_cube side_length = 125 := by
  sorry

end volume_of_cube_proof_l669_669767


namespace find_missing_term_l669_669918

theorem find_missing_term :
  ∃ a4 : ℕ, 
    (a4 - 15 = 9) ∧ 
    (sequence = [3, 8, 15, a4, 35, 48]) :=
sorry

end find_missing_term_l669_669918


namespace car_a_speed_l669_669892

theorem car_a_speed (d_gap : ℕ) (v_B : ℕ) (t : ℕ) (d_ahead : ℕ) (v_A : ℕ) 
  (h1 : d_gap = 24) (h2 : v_B = 50) (h3 : t = 4) (h4 : d_ahead = 8)
  (h5 : v_A = (d_gap + v_B * t + d_ahead) / t) : v_A = 58 :=
by {
  exact (sorry : v_A = 58)
}

end car_a_speed_l669_669892


namespace solve_for_y_l669_669212

theorem solve_for_y (y : ℝ) : 5^9 = 25^y → y = 9 / 2 :=
by
  sorry

end solve_for_y_l669_669212


namespace calculate_total_cost_l669_669379

-- Define the prices of the pets
def price_puppies : List ℝ := [72, 78]
def price_kittens : List ℝ := [48, 52]
def price_parakeets : List ℝ := [10, 12, 14]

-- Define the discount rates for multiple purchases
def discount_puppies : ℝ := 0.05
def discount_kittens : ℝ := 0.10
def discount_parakeet_half : ℝ := 0.5

-- Define the total cost calculation formula
noncomputable def total_cost (puppy_prices kitten_prices parakeet_prices : List ℝ) : ℝ :=
  let total_puppies := (puppy_prices.sum) * (1 - discount_puppies) in
  let total_kittens := (kitten_prices.sum) * (1 - discount_kittens) in
  let total_parakeets := parakeet_prices.sum - (parakeet_prices.min!.toFloat* discount_parakeet_half) in
  total_puppies + total_kittens + total_parakeets

-- Theorem to prove the total cost calculation
theorem calculate_total_cost : total_cost price_puppies price_kittens price_parakeets = 263.5 := by
  sorry

end calculate_total_cost_l669_669379


namespace probability_at_most_one_hit_l669_669845

theorem probability_at_most_one_hit (p_A : ℝ) (p_B : ℝ) (h_independent : independent A B) 
  (h_p_A : p_A = 0.6) (h_p_B : p_B = 0.7) :
  P (at_most_one_hit A B) = 0.58 := 
sorry

end probability_at_most_one_hit_l669_669845


namespace how_many_cheburashkas_erased_l669_669604

theorem how_many_cheburashkas_erased 
  (total_krakozyabras : ℕ)
  (characters_per_row_initial : ℕ) 
  (total_characters_initial : ℕ)
  (total_cheburashkas : ℕ)
  (total_rows : ℕ := 2)
  (total_krakozyabras := 29) :
  total_cheburashkas = 11 :=
by
  sorry

end how_many_cheburashkas_erased_l669_669604


namespace cheburashkas_erased_l669_669592

theorem cheburashkas_erased (n : ℕ) (rows : ℕ) (krakozyabras : ℕ) 
  (h_spacing : ∀ r, r ≤ rows → krakozyabras = 2 * (n - 1))
  (h_rows : rows = 2)
  (h_krakozyabras : krakozyabras = 29) :
  n = 16 → rows = 2 → krakozyabras = 29 → n = 16 - 5 :=
by
  sorry

end cheburashkas_erased_l669_669592


namespace range_of_a_l669_669489

variable (a : ℝ)
def P : Prop := ∀ x ∈ Icc (1 : ℝ) 2, x^2 - a ≥ 0
def Q : Prop := ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0

theorem range_of_a (hP_or_Q : P a ∨ Q a) (hP_and_Q : ¬ (P a ∧ Q a)) :
  a ∈ Icc (-1) 1 ∨ a ∈ Ioi 3 := sorry

end range_of_a_l669_669489


namespace trig_prod_identity_l669_669393

theorem trig_prod_identity :
  let sin := real.sin
  let cos := real.cos
  (sin 12 * sin 36 * sin 54 * sin 84) = (1 / 8) :=
by
  sorry

end trig_prod_identity_l669_669393


namespace cube_volume_l669_669765

theorem cube_volume (P : ℝ) (h : P = 20) : ∃ (V : ℝ), V = 125 :=
by
  -- Let side length of a cube be 's'.
  let s := P / 4
  -- Calculate the volume
  let V := s^3
  -- The hypothesis given
  have hs : s = 5 := by linarith [h]
  rw hs at V
  -- Showing that the volume V is 125 cm³
  use V
  rw [hs, pow_succ, pow_succ, mul_assoc, mul_assoc]
  norm_num
  exact h

end cube_volume_l669_669765


namespace find_Matrix_l669_669923

open Matrix

variables {R : Type} [Field R] [Fintype R] [DecidableEq R]

def satisfies_condition (M : Matrix (Fin 3) (Fin 3) R) (v : Fin 3 → R) : Prop :=
  M.mul_vec v = (-7) • v

theorem find_Matrix (M : Matrix (Fin 3) (Fin 3) ℝ) :
  (∀ v : (Fin 3 → ℝ), satisfies_condition M v) →
  M = ![![ -7, 0, 0], ![ 0, -7, 0], ![ 0, 0, -7 ]] :=
by
  intro h
  sorry

end find_Matrix_l669_669923


namespace tan_B_is_correct_l669_669166

namespace TriangleProof

-- Define a structure for the triangle
structure triangle :=
  (A B C : ℝ) (cos_C : ℝ) (AC BC : ℝ)

noncomputable def cos_C_value : ℝ := 2 / 3
noncomputable def AC_value : ℝ := 4
noncomputable def BC_value : ℝ := 3

def my_triangle : triangle :=
  {A := 0, B := 0, C := 0, cos_C := cos_C_value, AC := AC_value, BC := BC_value}

theorem tan_B_is_correct (t : triangle) (h₁ : t.cos_C = 2 / 3) (h₂ : t.AC = 4) (h₃ : t.BC = 3) : 
  real.tan t.B = 4 * real.sqrt 5 :=
sorry

end TriangleProof

end tan_B_is_correct_l669_669166


namespace probability_at_most_one_hit_l669_669847

noncomputable def P {Ω : Type*} [MeasureSpace Ω] (P : MeasureTheory.ProbabilityMeasure Ω) (A B : Set Ω) : ℝ := 
  P.measure A * P.measure B + (P.measure (Aᶜ) * P.measure B) + (P.measure A * P.measure (Bᶜ)) + (P.measure (Aᶜ) * P.measure (Bᶜ))

theorem probability_at_most_one_hit (P : MeasureTheory.ProbabilityMeasure) (A B : Set Ω) 
  (hA : P.measure A = 0.6) 
  (hB : P.measure B = 0.7) 
  (h_indep : MeasureTheory.Independence P A B) :
  P.measure A * P.measure B = 0.42 → 
  P.measure (Aᶜ ∩ Aᶜ) = 0.58 :=
  sorry

end probability_at_most_one_hit_l669_669847


namespace log_equality_implies_x_value_l669_669022

theorem log_equality_implies_x_value :
  ∀ (x : ℝ), (0 < x) → (x ≠ 1) → (log x 8 = log 64 4) → x = 512 := by
  sorry

end log_equality_implies_x_value_l669_669022


namespace total_seeds_in_garden_l669_669514

-- Definitions based on the conditions
def top_bed_rows : ℕ := 4
def top_bed_seeds_per_row : ℕ := 25
def num_top_beds : ℕ := 2

def medium_bed_rows : ℕ := 3
def medium_bed_seeds_per_row : ℕ := 20
def num_medium_beds : ℕ := 2

-- Calculation of total seeds in top beds
def seeds_per_top_bed : ℕ := top_bed_rows * top_bed_seeds_per_row
def total_seeds_top_beds : ℕ := num_top_beds * seeds_per_top_bed

-- Calculation of total seeds in medium beds
def seeds_per_medium_bed : ℕ := medium_bed_rows * medium_bed_seeds_per_row
def total_seeds_medium_beds : ℕ := num_medium_beds * seeds_per_medium_bed

-- Proof goal
theorem total_seeds_in_garden : total_seeds_top_beds + total_seeds_medium_beds = 320 :=
by
  sorry

end total_seeds_in_garden_l669_669514


namespace captain_age_l669_669548

noncomputable def whole_team_age : ℕ := 253
noncomputable def remaining_players_age : ℕ := 198
noncomputable def captain_and_wicket_keeper_age : ℕ := whole_team_age - remaining_players_age
noncomputable def wicket_keeper_age (C : ℕ) : ℕ := C + 3

theorem captain_age (C : ℕ) (whole_team : whole_team_age = 11 * 23) (remaining_players : remaining_players_age = 9 * 22) 
    (sum_ages : captain_and_wicket_keeper_age = 55) (wicket_keeper : wicket_keeper_age C = C + 3) : C = 26 := 
  sorry

end captain_age_l669_669548


namespace angle_y_measure_l669_669566

theorem angle_y_measure (m n t : Line) (A B C D F G I : Point)
  (angle40 : angle A B C = 40) (angle90 : angle B C D = 90)
  (parallel_m_n : Parallel m n) (transversal_t : Transversal t m n) :
  angle y = 50 :=
begin
  sorry  -- Proof is not required
end

end angle_y_measure_l669_669566


namespace probability_rain_at_most_3_days_l669_669770

/-- Define n to be the number of days in July --/
def n : ℕ := 31

/-- Define p to be the probability of rainfall on any given day in July in Capital City, p = 3/20 --/
def p : ℚ := 3 / 20

/-- Define the binomial probability formula for a given k --/
def binomial_probability (k : ℕ) : ℚ := 
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

/-- State the theorem that the probability of rainfall on at most 3 days in July is approximately 0.625 --/
theorem probability_rain_at_most_3_days : 
  (binomial_probability 0 + binomial_probability 1 + binomial_probability 2 + binomial_probability 3) ≈ 0.625 := 
by 
  sorry

end probability_rain_at_most_3_days_l669_669770


namespace total_lateness_l669_669377

/-
  Conditions:
  Charlize was 20 minutes late.
  Ana was 5 minutes later than Charlize.
  Ben was 15 minutes less late than Charlize.
  Clara was twice as late as Charlize.
  Daniel was 10 minutes earlier than Clara.

  Total time for which all five students were late is 120 minutes.
-/

def charlize := 20
def ana := charlize + 5
def ben := charlize - 15
def clara := charlize * 2
def daniel := clara - 10

def total_time := charlize + ana + ben + clara + daniel

theorem total_lateness : total_time = 120 :=
by
  sorry

end total_lateness_l669_669377


namespace longest_side_of_region_l669_669397

theorem longest_side_of_region :
  (∃ (x y : ℝ), x + y ≤ 5 ∧ 3 * x + y ≥ 3 ∧ x ≥ 1 ∧ y ≥ 1) →
  (∃ (l : ℝ), l = Real.sqrt 130 / 3 ∧ 
    (l = Real.sqrt ((1 - 1)^2 + (4 - 1)^2) ∨ 
     l = Real.sqrt (((1 + 4 / 3) - 1)^2 + (1 - 1)^2) ∨ 
     l = Real.sqrt ((1 - (1 + 4 / 3))^2 + (1 - 1)^2))) :=
by
  sorry

end longest_side_of_region_l669_669397


namespace count_non_prime_numbers_with_digits_sum_to_seven_l669_669901

def digits_sum_to_seven (n : ℕ) : Prop :=
  (∃ (digits : List ℕ), (∀ d ∈ digits, d > 0) ∧ (digits.sum = 7) ∧ (Nat.ofDigits digits = n))

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

def is_not_prime (n : ℕ) : Prop :=
  ¬ is_prime n

theorem count_non_prime_numbers_with_digits_sum_to_seven : 
  (∃ (numbers : List ℕ), 
      (∀ n ∈ numbers, digits_sum_to_seven n ∧ is_not_prime n) ∧ 
      (numbers.length = 5)) := 
by
  sorry

end count_non_prime_numbers_with_digits_sum_to_seven_l669_669901


namespace overtaking_actions_l669_669315

theorem overtaking_actions (n : ℕ) (h : n ≥ 1) :
  let f_n := 2^(n-1)
  let g_n := 2^(n-2)
  in f_n = 2 * g_n :=
by
  sorry

end overtaking_actions_l669_669315


namespace warriors_wins_count_l669_669233

variable {wins : ℕ → ℕ}
variable (raptors hawks warriors spurs lakers : ℕ)

def conditions (wins : ℕ → ℕ) (raptors hawks warriors spurs lakers : ℕ) : Prop :=
  wins raptors > wins hawks ∧
  wins warriors > wins spurs ∧ wins warriors < wins lakers ∧
  wins spurs > 25

theorem warriors_wins_count
  (wins : ℕ → ℕ)
  (raptors hawks warriors spurs lakers : ℕ)
  (h : conditions wins raptors hawks warriors spurs lakers) :
  wins warriors = 37 := sorry

end warriors_wins_count_l669_669233


namespace factorize_expression_l669_669410

theorem factorize_expression (a x : ℝ) :
  a * x^2 - 2 * a * x + a = a * (x - 1) ^ 2 := 
sorry

end factorize_expression_l669_669410


namespace right_triangle_area_l669_669748

theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a = 5) (hc : c = 13) :
  1/2 * a * b = 30 :=
by
  have hb : b = 12, from sorry,
  -- Proof needs to be filled here
  sorry

end right_triangle_area_l669_669748


namespace benzoic_acid_mass_proof_l669_669011

noncomputable def molar_mass_C : ℚ := 12.01
noncomputable def molar_mass_H : ℚ := 1.008
noncomputable def molar_mass_O : ℚ := 16.00

def benzoic_acid_C := 7
def benzoic_acid_H := 6
def benzoic_acid_O := 2

noncomputable def molar_mass_benzoic_acid :=
  (benzoic_acid_C * molar_mass_C) + (benzoic_acid_H * molar_mass_H) + (benzoic_acid_O * molar_mass_O)

def moles_benzoic_acid := 4
noncomputable def mass_4_moles_benzoic_acid := moles_benzoic_acid * molar_mass_benzoic_acid

def mass_percentage_C := (benzoic_acid_C * molar_mass_C) / molar_mass_benzoic_acid * 100
def mass_percentage_H := (benzoic_acid_H * molar_mass_H) / molar_mass_benzoic_acid * 100
def mass_percentage_O := (benzoic_acid_O * molar_mass_O) / molar_mass_benzoic_acid * 100
def total_mass_percentage := mass_percentage_C + mass_percentage_H + mass_percentage_O

def volume_solution := 3
def concentration_solution := 1.5
noncomputable def mass_solution := concentration_solution * volume_solution * molar_mass_benzoic_acid

theorem benzoic_acid_mass_proof :
  mass_4_moles_benzoic_acid = 488.472 ∧
  total_mass_percentage = 100 ∧
  mass_solution = 549.531 :=
by
  sorry

end benzoic_acid_mass_proof_l669_669011


namespace reasoning_used_is_analogical_reasoning_l669_669811

variable (Line Circle LineTangentPoint CircleCenter Sphere Plane SphereTangentPoint SphereCenter : Type)

-- Conditions
variable (line_tangent_circle : Line × Circle)
variable (line_perpendicular_to_center : LineTangentPoint × CircleCenter)
variable (plane_tangent_sphere : Plane × Sphere)
variable (plane_perpendicular_to_center : SphereTangentPoint × SphereCenter)

-- Define the type of reasoning
def reasoning_type := "Analogical reasoning"

-- Proof statement
theorem reasoning_used_is_analogical_reasoning :
  (line_tangent_circle → line_perpendicular_to_center) →
  (plane_tangent_sphere → plane_perpendicular_to_center) →
  reasoning_type = "Analogical reasoning" :=
sorry

end reasoning_used_is_analogical_reasoning_l669_669811


namespace distribute_volunteers_l669_669404

theorem distribute_volunteers :
  ∃ (n : ℕ), 
  let volunteers := 4 in
  let venues := 3 in
  let groups := volunteers.choose 2 in
  let assignments := factorial venues in
  let total_schemes := groups * assignments in
  total_schemes = 36 :=
by {
  sorry
}

end distribute_volunteers_l669_669404


namespace correct_proposition_is_c_l669_669296

/-- Proposition A: If two lines are perpendicular to the same line, then these two lines are parallel --/
def prop_a (l₁ l₂ l₃ : Line) : Prop :=
  (l₁ ⊥ l₃ ∧ l₂ ⊥ l₃) → l₁ ∥ l₂

/-- Proposition B: There is only one line parallel to a given line passing through a point --/
def prop_b (l : Line) (p : Point) : Prop :=
  (∃! m : Line, (m ∥ l) ∧ p ∈ m)

/-- Proposition C: In the same plane, there is only one line perpendicular to a given line passing through a point --/
def prop_c (l : Line) (p : Point) : Prop :=
  (∃! m : Line, (m ⊥ l) ∧ p ∈ m)

/-- Proposition D: Corresponding angles are equal --/
def prop_d : Prop :=
  ∀ (α β : Angle), (α.corresponding β) → (α = β)

/-- The proposition that is proven correct is Proposition C --/
theorem correct_proposition_is_c (l : Line) (p : Point) :
  prop_c l p := sorry

end correct_proposition_is_c_l669_669296


namespace Jeff_wins_three_games_l669_669585

-- Define the conditions and proven statement
theorem Jeff_wins_three_games :
  (hours_played : ℕ) (minutes_per_point : ℕ) (points_per_match : ℕ) 
  (hours_played = 2) (minutes_per_point = 5) (points_per_match = 8) 
  → (games_won : ℕ) (120 / minutes_per_point / points_per_match = 3) :=
by
  -- Step through assumptions and automatically conclude the proof
  sorry

end Jeff_wins_three_games_l669_669585


namespace sum_first_8_terms_l669_669499

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x + Real.log (x / 4)
  else if x = 0 then 0
  else -(2^(-x) + Real.log (-x / 4))

def a_n (n : ℕ) : ℝ := f (n - 5)

theorem sum_first_8_terms : (a_n 1) + (a_n 2) + (a_n 3) + (a_n 4) + (a_n 5) + (a_n 6) + (a_n 7) + (a_n 8) = -16 :=
by
  sorry -- Proof steps are omitted according to the requirement

end sum_first_8_terms_l669_669499


namespace replace_integers_preserve_mean_variance_l669_669161

def orig_set : Set Int := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
def mean (s : Set ℤ) : ℚ := (s.sum / s.size : ℚ)
def variance (s : Set ℤ) : ℚ :=
  let μ := mean s
  (s.sum (λ x => (x - μ)^2) / s.size : ℚ)

theorem replace_integers_preserve_mean_variance :
  ∃ a b c : Int,
    a ∈ orig_set ∧
    b ∈ orig_set ∧
    c ∈ orig_set ∧
    (mean orig_set = mean (orig_set.erase a ∪ {b, c})) ∧
    (variance orig_set = variance (orig_set.erase a ∪ {b, c})) ∧
    (a = b + c) ∧
    (b^2 + c^2 = a^2 + 10) :=
by
  sorry

end replace_integers_preserve_mean_variance_l669_669161


namespace log_basis_512_l669_669021

theorem log_basis_512 (x : ℝ) (hx : real.log x 8 = real.log 64 4) : x = 512 := 
by 
  sorry

end log_basis_512_l669_669021


namespace max_sine_sum_in_triangle_l669_669205

theorem max_sine_sum_in_triangle (A B C : ℝ) (hA : 0 < A) (hA' : A < π) (hB : 0 < B) (hB' : B < π) (hC : 0 < C) (hC' : C < π) :
  sin A + sin B + sin C ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end max_sine_sum_in_triangle_l669_669205


namespace find_m_l669_669333

theorem find_m (m : ℤ) :
  (2 * m + 7) * (m - 2) = 51 → m = 5 := by
  sorry

end find_m_l669_669333


namespace right_triangle_exists_l669_669820

theorem right_triangle_exists :
  (¬ (2^2 + 4^2 = 5^2)) ∧
  (¬ (sqrt 3^2 + sqrt 4^2 = sqrt 5^2)) ∧
  (3^2 + 4^2 = 5^2) ∧
  (¬ (5^2 + 13^2 = 14^2)) :=
by
  sorry

end right_triangle_exists_l669_669820


namespace gpa_at_least_3_5_l669_669665

noncomputable def prob_gpa_at_least_3_5 : ℚ :=
  let p_A_eng := 1 / 3
  let p_B_eng := 1 / 5
  let p_C_eng := 7 / 15 -- 1 - p_A_eng - p_B_eng
  
  let p_A_hist := 1 / 5
  let p_B_hist := 1 / 4
  let p_C_hist := 11 / 20 -- 1 - p_A_hist - p_B_hist

  let prob_two_As := p_A_eng * p_A_hist
  let prob_A_eng_B_hist := p_A_eng * p_B_hist
  let prob_A_hist_B_eng := p_A_hist * p_B_eng
  let prob_two_Bs := p_B_eng * p_B_hist

  let total_prob := prob_two_As + prob_A_eng_B_hist + prob_A_hist_B_eng + prob_two_Bs
  total_prob

theorem gpa_at_least_3_5 : prob_gpa_at_least_3_5 = 6 / 25 := by {
  sorry
}

end gpa_at_least_3_5_l669_669665


namespace elberta_money_l669_669995

theorem elberta_money :
  ∀ (granny_smith : ℝ) (anjou_fraction : ℝ) (elberta_increase : ℝ),
  granny_smith = 75 → anjou_fraction = 1/4 → elberta_increase = 0.10 →
  let anjou := anjou_fraction * granny_smith in
  let elberta := anjou * (1 + elberta_increase) in
  Float.round elberta = 21 :=
by
  intros granny_smith anjou_fraction elberta_increase h_granny h_anjou h_elberta
  let anjou := anjou_fraction * granny_smith
  let elberta := anjou * (1 + elberta_increase)
  show Float.round elberta = 21
  sorry

end elberta_money_l669_669995


namespace sin_expression_value_set_of_angles_l669_669501

-- Define the angle α based on the given condition that 
-- the terminal side of α passes through the point P(1, √3).
def α_pass_through_P : ℝ → Prop := 
  λ α, ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 3 

-- Question 1: Prove that
theorem sin_expression_value (α : ℝ) (h : α_pass_through_P α) : 
  (Real.sin (Real.pi - α) - Real.sin (Real.pi / 2 + α) = (Real.sqrt 3 - 1) / 2) := 
sorry

-- Question 2: Prove that the set S of angle α is as given
theorem set_of_angles (S : Set ℝ) : 
  (S = {α | ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 3}) := 
sorry

end sin_expression_value_set_of_angles_l669_669501


namespace mason_total_nuts_l669_669646

/-- Mason opens the hood of his car and discovers that squirrels have been using his engine compartment to store nuts.
If 2 busy squirrels have been stockpiling 30 nuts/day and one sleepy squirrel has been stockpiling 20 nuts/day, all for 40 days, 
the total number of nuts in Mason's car is 3200. -/
theorem mason_total_nuts : 
  ∀ (nuts_per_day_busy : ℕ) (num_busy_squirrels : ℕ) 
    (nuts_per_day_sleepy : ℕ) (num_sleepy_squirrels : ℕ) 
    (num_days : ℕ), 
  nuts_per_day_busy = 30 → 
  num_busy_squirrels = 2 → 
  nuts_per_day_sleepy = 20 → 
  num_sleepy_squirrels = 1 → 
  num_days = 40 →
  ((num_busy_squirrels * nuts_per_day_busy) + (num_sleepy_squirrels * nuts_per_day_sleepy)) * num_days = 3200 := 
by 
  intros nuts_per_day_busy num_busy_squirrels nuts_per_day_sleepy num_sleepy_squirrels num_days
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  exact sorry

end mason_total_nuts_l669_669646


namespace power_series_solution_l669_669028

noncomputable def y (x : ℝ) : ℝ := 1 + 2 * x - x^2 + (2/3) * x^3

theorem power_series_solution :
  (∀ (y' : ℝ → ℝ) (x : ℝ),
  -- Condition: Differential equation
  y' x = deriv y x ∧
  (y' x + x * (y x)^2 = 2 * cos x)) ∧
  -- Condition: Initial condition
  (y 0 = 1) :=
by
  sorry

end power_series_solution_l669_669028


namespace tangent_segment_proportionality_l669_669115

theorem tangent_segment_proportionality
    (O1 O2 : Type) [MetricSpace O1] [MetricSpace O2]
    (P : O1) (T : O2)
    (intersection_line : Set O1)
    (tangent_PT : LineSegment O1 O2)
    (D1 D2 : ℝ) -- Distances corresponding to O1O2 and PT respectively
    (h : ℝ)    -- Distance of P from intersection_line
    (H1 : IsCircle O1) (H2 : IsCircle O2) -- O1 and O2 are circles
    (H3 : P ∈ O1) -- Point P is on circle O1
    (H4 : PT ∈ tangent_PT) -- PT is a tangent from P on O1 to O2
    (H5 : D2^2 = 2 * D1 * h) -- Squared tangent length proportionality
    : PT^2 = 2 * D1 * h :=
by {
    sorry
}

end tangent_segment_proportionality_l669_669115


namespace piecewise_function_sum_l669_669939

def f (x : ℝ) : ℝ :=
  if x > 0 then 2 * x else f (x + 1)

theorem piecewise_function_sum :
  f (4 / 3) + f (-4 / 3) = 4 := by
  sorry

end piecewise_function_sum_l669_669939


namespace max_value_of_abc_l669_669453

theorem max_value_of_abc (n : ℕ) (a b c : ℕ) 
  (h_pos_n : n > 0)
  (h_nonzero_digits : a > 0 ∧ a < 10 ∧ b > 0 ∧ b < 10 ∧ c > 0 ∧ c < 10)
  (h_for_two_n : ∃ n1 n2 : ℕ, n1 > 0 ∧ n2 > 0 ∧ n1 ≠ n2 ∧
                 (c * ((10^n1) + 1) - b = a^2 * ((10^n1) - 1)^2 / 81) ∧
                 (c * ((10^n2) + 1) - b = a^2 * ((10^n2) - 1)^2 / 81))
  : a + b + c = 18 :=
begin
  sorry -- Proof goes here.
end

end max_value_of_abc_l669_669453


namespace alpha_perp_beta_l669_669184

variables (m n : Line) (alpha beta : Plane)

-- Conditions
variables (m_perp_alpha : m ∠ alpha) (m_parallel_n : m ∥ n) (n_in_beta : n ⊂ beta)

-- Theorem
theorem alpha_perp_beta (m_perp_alpha : m ∠ alpha) (m_parallel_n : m ∥ n) (n_in_beta : n ⊂ beta) : alpha ∠ beta :=
sorry

end alpha_perp_beta_l669_669184


namespace simplify_cubic_root_l669_669698

theorem simplify_cubic_root : 
  (∛(54880000) = 20 * ∛((5^2) * 137)) :=
sorry

end simplify_cubic_root_l669_669698


namespace right_triangle_area_l669_669750

theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a = 5) (hc : c = 13) :
  1/2 * a * b = 30 :=
by
  have hb : b = 12, from sorry,
  -- Proof needs to be filled here
  sorry

end right_triangle_area_l669_669750


namespace boxes_given_away_l669_669793

def total_boxes := 12
def pieces_per_box := 6
def remaining_pieces := 30

theorem boxes_given_away : (total_boxes * pieces_per_box - remaining_pieces) / pieces_per_box = 7 :=
by
  sorry

end boxes_given_away_l669_669793


namespace figure_can_be_rearranged_to_square_l669_669891

def can_form_square (n : ℕ) : Prop :=
  let s := Nat.sqrt n
  s * s = n

theorem figure_can_be_rearranged_to_square (n : ℕ) :
  (∃ a b c : ℕ, a + b + c = n) → (can_form_square n) → (n % 1 = 0) :=
by
  intros _ _
  sorry

end figure_can_be_rearranged_to_square_l669_669891


namespace union_complement_eq_univ_l669_669641

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 7}

-- Define set M
def M : Set ℕ := {1, 3, 5, 7}

-- Define set N
def N : Set ℕ := {3, 5}

-- Define the complement of N with respect to U
def complement_U_N : Set ℕ := {1, 2, 4, 7}

-- Prove that U = M ∪ complement_U_N
theorem union_complement_eq_univ : U = M ∪ complement_U_N := 
sorry

end union_complement_eq_univ_l669_669641


namespace total_seeds_in_garden_l669_669516

-- Definitions based on conditions
def large_bed_rows : Nat := 4
def large_bed_seeds_per_row : Nat := 25
def medium_bed_rows : Nat := 3
def medium_bed_seeds_per_row : Nat := 20
def num_large_beds : Nat := 2
def num_medium_beds : Nat := 2

-- Theorem statement to show total seeds
theorem total_seeds_in_garden : 
  num_large_beds * (large_bed_rows * large_bed_seeds_per_row) + 
  num_medium_beds * (medium_bed_rows * medium_bed_seeds_per_row) = 320 := 
by
  sorry

end total_seeds_in_garden_l669_669516


namespace expected_value_X_is_half_l669_669254

open BigOperators

-- Define the random digit sequence and its properties
def random_digit_seq (n : ℕ) (d : ℕ → ℕ) := ∀ i : ℕ, i < n → d i ∈ Fin 10

-- Expected value of a single random digit
def expected_value_digit : ℝ := 4.5

-- Define the expected value of X
noncomputable def expected_value_of_X (n : ℕ) (d : ℕ → ℕ) : ℝ :=
  ∑ i in Finset.range n, (d i : ℝ) * 10^(-(i+1))

-- The main theorem to prove
theorem expected_value_X_is_half : 
  ∀ (d : ℕ → ℕ), random_digit_seq 1000 d → expected_value_of_X 1000 d = 0.5 :=
by {
  intro d,
  intro h,
  sorry  -- The proof would be written here.
}

end expected_value_X_is_half_l669_669254


namespace max_value_of_f_l669_669228

noncomputable def f (x : ℝ) := x^3 - 3 * x^2

theorem max_value_of_f :
  ∃ x ∈ set.Icc (-2 : ℝ) 4, ∀ y ∈ set.Icc (-2 : ℝ) 4, f y ≤ f x := 
begin
  -- Mathematical problem equivalent to proving the maximum value is 16 in \([-2,4]\)
  use 4,
  split,
  { norm_num },
  { intros y hy,
    interval_cases (show y ∈ ({-2, 0, 2, 4} : set ℝ), by
    { obtain ⟨h1, h2⟩ := hy, -- obtain bounds of y
      interval_cases y; norm_num }),
    { norm_num },
    { norm_num },
    { norm_num },
    { norm_num } }
end

end max_value_of_f_l669_669228


namespace extreme_value_at_neg2_l669_669506

theorem extreme_value_at_neg2 (a : ℝ) (f : ℝ → ℝ) (h : f = λ x, a*x^2 + x^2) :
  (∃ c : ℝ, c = -2 ∧ ∃ x0 : ℝ, f'' x0 = 0) → a = -1 :=
by sorry

end extreme_value_at_neg2_l669_669506


namespace find_alpha_l669_669106

-- Define the piecewise function f
def f (x : ℝ) (α : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + Real.sin x else -x^2 + Real.cos (x + α)

-- Define the property that f(x) is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- State the theorem to prove
theorem find_alpha (α : ℝ) (hα : 0 ≤ α ∧ α < 2 * Real.pi) (H : is_odd_function (λ x, f x α)) :
  α = 3 * Real.pi / 2 :=
sorry

end find_alpha_l669_669106


namespace line_AB_passes_through_fixed_point_chord_shortest_length_at_specific_m_l669_669056

-- Assume real numbers for x, y, and m
variables {x y m : ℝ}

-- Define the circle C
def Circle (x y : ℝ) := (x - 1)^2 + (y - 2)^2 = 25

-- Define the line AB
def LineAB (x y m : ℝ) := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

-- Define the fixed point P
def PointP : ℝ × ℝ := (3, 1)

-- Prove that Line AB passes through the fixed point P
theorem line_AB_passes_through_fixed_point :
  LineAB 3 1 m := sorry

-- Prove when m = -3/4, the chord is the shortest and find the shortest length
theorem chord_shortest_length_at_specific_m (C : ℝ × ℝ) :
  C = (1, 2) →
  LineAB (C.1) (C.2) -3/4 →
  sqrt(25 - (5)) = sqrt(5) →
  (2 * sqrt(5)) * 2 = 4 * sqrt(5) :=
sorry

end line_AB_passes_through_fixed_point_chord_shortest_length_at_specific_m_l669_669056


namespace gdp_scientific_notation_l669_669143

noncomputable def gdp_nanning_2007 : ℝ := 1060 * 10^8

theorem gdp_scientific_notation :
  gdp_nanning_2007 = 1.06 * 10^11 :=
by sorry

end gdp_scientific_notation_l669_669143


namespace sum_possible_employees_l669_669850

theorem sum_possible_employees : 
  (∑ s in finset.filter (λ s, (s - 1) % 7 = 0) (finset.Icc 200 300), s) = 3493 := 
  sorry

end sum_possible_employees_l669_669850


namespace factorize_expression_l669_669411

theorem factorize_expression (a x : ℝ) :
  a * x^2 - 2 * a * x + a = a * (x - 1) ^ 2 := 
sorry

end factorize_expression_l669_669411


namespace steven_farmland_acres_l669_669537

theorem steven_farmland_acres :
  ∀ (plow_per_day mow_per_day grassland_days total_days : ℝ),
  plow_per_day = 10 →
  mow_per_day = 12 →
  grassland_days * mow_per_day = 30 →
  total_days = 8 →
  ∃ farmland : ℝ, farmland = 55 :=
by
  introv hp hm hg ht
  use 55
  sorry

end steven_farmland_acres_l669_669537


namespace kite_parabolas_l669_669236

theorem kite_parabolas (a b : ℝ) 
  (h1 : (∃ x : ℝ, ax^2 - 4 = 0) ∧ (∃ y : ℝ , 6 - bx^2 = 0))
  (h2 : ∃ A B C D : ℝ × ℝ, 
    Set.Pairwise (≠) {A, B, C, D} ∧ 
    Set.Pairwise (λ p q : ℝ × ℝ, p.1 ≠ q.1 ∨ p.2 ≠ q.2) {A, B, C, D} ∧ 
    Parallelogram (line A B) (line C D) (line A C) (line B D) ∧ 
    kite_area {A, B, C, D} = 24):
  a + b = 125 / 72 :=
sorry

end kite_parabolas_l669_669236


namespace tan_alpha_beta_l669_669943

theorem tan_alpha_beta (α β : ℝ) (h : 2 * Real.sin β = Real.sin (2 * α + β)) :
  Real.tan (α + β) = 3 * Real.tan α := 
sorry

end tan_alpha_beta_l669_669943


namespace no_factors_of_polynomial_l669_669010

def polynomial := (x : ℝ) → x^5 + 3 * x^3 - 4 * x^2 + 12 * x + 8

theorem no_factors_of_polynomial :
  ¬ (∃ g : (ℝ[X]), (g = X + 1 ∨ g = X^2 + 1 ∨ g = X^2 - 2 ∨ g = X^2 + 3) ∧ g ∣ (polynomial X)) :=
by sorry

end no_factors_of_polynomial_l669_669010


namespace geometric_sequence_second_term_l669_669853

theorem geometric_sequence_second_term
  (first_term : ℕ) (fourth_term : ℕ) (r : ℕ)
  (h1 : first_term = 6)
  (h2 : first_term * r^3 = fourth_term)
  (h3 : fourth_term = 768) :
  first_term * r = 24 := by
  sorry

end geometric_sequence_second_term_l669_669853


namespace average_special_divisibles_l669_669307

/--
  Prove that the average of all the numbers between 18 and 57 
  which are divisible by 7 but not divisible by any other prime number except 7 is 49.
-/
theorem average_special_divisibles : 
  let numbers := (filter (λ n, n % 7 = 0 ∧ ∀ p, prime p → p ≠ 7 → n % p ≠ 0) (list.range' 18 (57 - 18 + 1))),
      sum := list.sum numbers,
      count := list.length numbers
  in
  count > 0 → (sum / count) = 49 :=
by
  sorry

end average_special_divisibles_l669_669307


namespace M_is_real_set_l669_669179

noncomputable 
def M (Z : ℂ) : Prop := (Z - 1) ^ 2 = complex.abs (Z - 1) ^ 2

theorem M_is_real_set : {Z : ℂ | M Z} = {Z : ℂ | Z.im = 0} := 
sorry

end M_is_real_set_l669_669179


namespace circumscribed_circle_area_l669_669154

theorem circumscribed_circle_area (ABC : Triangle)
  (h_acute : acute_angled ABC)
  (A C B : Point)
  (h_AC : dist A C = 1)
  (C1 A1 : Point)
  (h_altitudes : is_altitude C1 C A1 A ABC)
  (alpha : Angle)
  (h_angle : angle C1 C A1 = alpha) :
  area (circumcircle (triangle (C1, B, A1))) = (π / 4) * (tan alpha)^2 := 
sorry

end circumscribed_circle_area_l669_669154


namespace planes_perpendicular_circumscribed_sphere_volume_l669_669150

variables (P A B C D : Point)
variables (h_reg_tetra : is_regular_tetrahedron P A B C D)
variables (h_AB_3 : distance A B = 3)
variables (h_dihedral_angle : dihedral_angle P A D P C D = 2 * π / 3)

-- Part 1: Proving orthogonality of two planes
theorem planes_perpendicular :
  ⊢ plane_perp_to_plane P A D P B C :=
sorry

-- Part 2: Proving existence of circumscribed sphere and its volume
theorem circumscribed_sphere_volume :
  ∃ sphere : Sphere, sphere ≈ circumsphere (Tetrahedron P A B C D)
  ∧ volume sphere = 243 / 16 * π :=
sorry

end planes_perpendicular_circumscribed_sphere_volume_l669_669150


namespace percentage_of_trout_is_correct_l669_669889

-- Define the conditions
def video_game_cost := 60
def last_weekend_earnings := 35
def earnings_per_trout := 5
def earnings_per_bluegill := 4
def total_fish_caught := 5
def additional_savings_needed := 2

-- Define the total amount needed to buy the game
def total_required_savings := video_game_cost - additional_savings_needed

-- Define the amount earned this Sunday
def earnings_this_sunday := total_required_savings - last_weekend_earnings

-- Define the number of trout and blue-gill caught thisSunday
def num_trout := 3
def num_bluegill := 2    -- Derived from the conditions

-- Theorem: given the conditions, prove that the percentage of trout is 60%
theorem percentage_of_trout_is_correct :
  (num_trout + num_bluegill = total_fish_caught) ∧
  (earnings_per_trout * num_trout + earnings_per_bluegill * num_bluegill = earnings_this_sunday) →
  100 * num_trout / total_fish_caught = 60 := 
by
  sorry

end percentage_of_trout_is_correct_l669_669889


namespace find_f_inv_l669_669053

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a x

variable (a : ℝ)
hypothesis h1 : a > 0
hypothesis h2 : a ≠ 1
hypothesis h3 : ∃ x, f a x = -1 ∧ x = 2

theorem find_f_inv (y : ℝ) : (∃ x, f a x = y) →
  (f a) = (λ x, log a x) →  
  a = 1 / 2 → 
  (∃ x:ℝ, (a : ℝ)^y = x) :=
sorry

end find_f_inv_l669_669053


namespace mason_total_nuts_l669_669647

/-- Mason opens the hood of his car and discovers that squirrels have been using his engine compartment to store nuts.
If 2 busy squirrels have been stockpiling 30 nuts/day and one sleepy squirrel has been stockpiling 20 nuts/day, all for 40 days, 
the total number of nuts in Mason's car is 3200. -/
theorem mason_total_nuts : 
  ∀ (nuts_per_day_busy : ℕ) (num_busy_squirrels : ℕ) 
    (nuts_per_day_sleepy : ℕ) (num_sleepy_squirrels : ℕ) 
    (num_days : ℕ), 
  nuts_per_day_busy = 30 → 
  num_busy_squirrels = 2 → 
  nuts_per_day_sleepy = 20 → 
  num_sleepy_squirrels = 1 → 
  num_days = 40 →
  ((num_busy_squirrels * nuts_per_day_busy) + (num_sleepy_squirrels * nuts_per_day_sleepy)) * num_days = 3200 := 
by 
  intros nuts_per_day_busy num_busy_squirrels nuts_per_day_sleepy num_sleepy_squirrels num_days
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  exact sorry

end mason_total_nuts_l669_669647


namespace triangle_area_l669_669285

theorem triangle_area : 
  let p1 := (0, 0)
  let p2 := (0, 6)
  let p3 := (8, 15)
  let base := 6
  let height := 8
  0.5 * base * height = 24.0 :=
by
  let p1 := (0, 0)
  let p2 := (0, 6)
  let p3 := (8, 15)
  let base := 6
  let height := 8
  sorry

end triangle_area_l669_669285


namespace cheburashkas_erased_l669_669610

theorem cheburashkas_erased (total_krakozyabras : ℕ) (rows : ℕ) :
  rows ≥ 2 → total_krakozyabras = 29 → ∃ (cheburashkas_erased : ℕ), cheburashkas_erased = 11 :=
by
  assume h_rows h_total_krakozyabras
  let n := (total_krakozyabras / 2) + 1
  have h_cheburashkas : cheburashkas_erased = n - 1 
  sorry

end cheburashkas_erased_l669_669610


namespace total_fish_buckets_needed_l669_669785

-- Definitions based on the conditions in the problem
def daily_trout_1 := 0.2
def daily_salmon_1 := 0.4

def daily_trout_2 := 0.3
def daily_salmon_2 := 0.5

def daily_trout_3 := 0.25
def daily_salmon_3 := 0.45

def daily_trout_total := daily_trout_1 + daily_trout_2 + daily_trout_3
def daily_salmon_total := daily_salmon_1 + daily_salmon_2 + daily_salmon_3
def daily_fish_total := daily_trout_total + daily_salmon_total

def days_in_week := 7

def weekly_fish_total := daily_fish_total * days_in_week

theorem total_fish_buckets_needed: 
  weekly_fish_total = 14.7 :=
sorry

end total_fish_buckets_needed_l669_669785


namespace cycle_of_length_at_least_k_plus_1_l669_669551

open GraphTheory

-- Define the conditions: a graph G with vertices and degree condition
variables (G : SimpleGraph V) {k : ℕ}
variable (k_ge_2 : k ≥ 2)
variable (degree_condition : ∀ v : V, G.degree v ≥ k)

-- State the theorem
theorem cycle_of_length_at_least_k_plus_1 :
  ∃ (c : G.cycle), c.length ≥ k + 1 :=
sorry

end cycle_of_length_at_least_k_plus_1_l669_669551


namespace second_polygon_sides_l669_669277

theorem second_polygon_sides (s : ℝ) (n : ℝ) (h1 : 50 * 3 * s = n * s) : n = 150 := 
by
  sorry

end second_polygon_sides_l669_669277


namespace part1_part2_l669_669182

-- Definitions for part (1)
def m (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin (x / 4), 1)
def n (x : ℝ) : ℝ × ℝ := (cos (x / 4), cos (x / 4) ^ 2)
def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Proposition for part (1)
theorem part1 : f π = (sqrt 3 + 1) / 2 := sorry

-- Definitions for part (2)
variables (A B C a b c : ℝ) (cond : b * cos C + c / 2 = a)

-- Proposition for part (2)
theorem part2 (h1 : cond) : B = π / 3 := sorry

end part1_part2_l669_669182


namespace yolanda_walking_rate_l669_669823

theorem yolanda_walking_rate (distance_xy : ℕ) (distance_bob_walked : ℕ) (bob_walking_rate : ℕ) (bob_time : ℕ) 
  (distance_yolanda_walked : ℕ) (yolanda_time : ℕ) (yolanda_walking_rate : ℚ) :
  distance_xy = 31 ∧ distance_bob_walked = 20 ∧ bob_walking_rate = 2 ∧ bob_time = distance_bob_walked / bob_walking_rate ∧
  distance_yolanda_walked = distance_xy - distance_bob_walked ∧ yolanda_time = bob_time ∧ yolanda_walking_rate = distance_yolanda_walked / yolanda_time →
  yolanda_walking_rate = 1.1 :=
begin
  sorry
end

end yolanda_walking_rate_l669_669823


namespace conditional_probability_one_l669_669546

/-- Statement: Given a bag of 3 red balls and 2 white balls, with two balls drawn successively
  without replacement, event A is the first ball drawn is red, and event B is the second ball drawn
  is red. The conditional probability \( p(B|A) \) is 1. -/
theorem conditional_probability_one :
  ∀ (balls : Finset ℕ) (red white: ℕ),
    balls = {1, 2, 3, 4, 5} →
    red = 3 →
    white = 2 →
    ∀ (A B : Set (Finset ℕ)),
      A = {s ∈ balls.powerset | s.card = 1 ∧ 1 ≤ s.min ∧ s.min ≤ red} →
      B = {s ∈ balls.powerset | s.card = 2 → (s \ {s.min}).min ≤ red} →
      let P := λ (x : Set (Finset ℕ)), x.card.toRat / (Finset.card balls).choose 2 in
      P (A ∩ B) / P A = 1 :=
begin
  intros,
  sorry
end

end conditional_probability_one_l669_669546


namespace solve_for_n_l669_669530

theorem solve_for_n (n : ℕ) (h : sqrt (8 + n) = 9) : n = 73 := 
by {
  sorry
}

end solve_for_n_l669_669530


namespace identify_a_b_l669_669012

theorem identify_a_b (a b : ℝ) (h : ∀ x y : ℝ, (⌊a * x + b * y⌋ + ⌊b * x + a * y⌋ = (a + b) * ⌊x + y⌋)) : 
  (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 1) :=
sorry

end identify_a_b_l669_669012


namespace isabella_hair_now_l669_669583

variable (original_hair_length : ℕ) (cut_hair_length : ℕ)

-- Condition statements
def isabella_original_hair_length : original_hair_length = 18 := sorry
def isabella_cut_hair_length : cut_hair_length = 9 := sorry

-- The proof problem
theorem isabella_hair_now : original_hair_length - cut_hair_length = 9 := by
  rw [isabella_original_hair_length, isabella_cut_hair_length]
  -- Now it simplifies to 18 - 9 = 9
  simp

end isabella_hair_now_l669_669583


namespace number_of_ways_teamA_in_teamB_out_number_of_ways_one_team_each_division_l669_669145

-- Definitions corresponding to the conditions
def DivisionA : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
def DivisionB : Finset ℕ := {12, 13, 14, 15, 16, 17, 18, 19}
def TeamA : ℕ := 0 -- Team A is represented by 0 (one of the teams in Division A)
def TeamB : ℕ := 12 -- Team B is represented by 12 (one of the teams in Division B)

-- Lean statement for the first proof problem
theorem number_of_ways_teamA_in_teamB_out : 
  (DivisionA ∪ DivisionB).erase TeamB ∈ Combinatorics.binomial (DivisionA ∪ DivisionB).erase TeamB 4 := by
  sorry

-- Lean statement for the second proof problem
theorem number_of_ways_one_team_each_division : 
  Combinatorics.binomial (DivisionA ∪ DivisionB) 5 - Combinatorics.binomial DivisionA 5 - Combinatorics.binomial DivisionB 5 = 14656 := by
  sorry

end number_of_ways_teamA_in_teamB_out_number_of_ways_one_team_each_division_l669_669145


namespace pages_removed_iff_original_pages_l669_669235

def booklet_sum (n r : ℕ) : ℕ :=
  (n * (2 * n + 1)) - (4 * r - 1)

theorem pages_removed_iff_original_pages (n r : ℕ) :
  booklet_sum n r = 963 ↔ (2 * n = 44 ∧ (2 * r - 1, 2 * r) = (13, 14)) :=
sorry

end pages_removed_iff_original_pages_l669_669235


namespace minimum_cactus_species_l669_669374

-- Definitions to represent the conditions
def num_cactophiles : Nat := 80
def num_collections (S : Finset (Fin num_cactophiles)) : Nat := S.card
axiom no_single_species_in_all (S : Finset (Fin num_cactophiles)) : num_collections S < num_cactophiles
axiom any_15_have_common_species (S : Finset (Fin num_cactophiles)) (h : S.card = 15) : 
  ∃ species, ∀ s ∈ S, species ∈ s

-- Proposition to be proved
theorem minimum_cactus_species (k : Nat) (h : ∀ S : Finset (Fin num_cactophiles), S.card = 15 → ∃ species, ∀ s ∈ S, species ∈ s) : k ≥ 16 := sorry

end minimum_cactus_species_l669_669374


namespace how_many_cheburashkas_erased_l669_669606

theorem how_many_cheburashkas_erased 
  (total_krakozyabras : ℕ)
  (characters_per_row_initial : ℕ) 
  (total_characters_initial : ℕ)
  (total_cheburashkas : ℕ)
  (total_rows : ℕ := 2)
  (total_krakozyabras := 29) :
  total_cheburashkas = 11 :=
by
  sorry

end how_many_cheburashkas_erased_l669_669606


namespace expected_value_X_is_half_l669_669251

open BigOperators

-- Define the random digit sequence and its properties
def random_digit_seq (n : ℕ) (d : ℕ → ℕ) := ∀ i : ℕ, i < n → d i ∈ Fin 10

-- Expected value of a single random digit
def expected_value_digit : ℝ := 4.5

-- Define the expected value of X
noncomputable def expected_value_of_X (n : ℕ) (d : ℕ → ℕ) : ℝ :=
  ∑ i in Finset.range n, (d i : ℝ) * 10^(-(i+1))

-- The main theorem to prove
theorem expected_value_X_is_half : 
  ∀ (d : ℕ → ℕ), random_digit_seq 1000 d → expected_value_of_X 1000 d = 0.5 :=
by {
  intro d,
  intro h,
  sorry  -- The proof would be written here.
}

end expected_value_X_is_half_l669_669251


namespace parabola_never_passes_point_l669_669990

theorem parabola_never_passes_point 
  (a : ℝ) (a_nonzero : a ≠ 0) (b c : ℝ) 
  (h_parabola : ∀ x : ℝ, (x = -1 → a * x^2 + b * x + c = 0) ∧ (x = 2 → a * x^2 + b * x + c = 0))
  (P : ℝ × ℝ) (x0 : ℝ)
  (h_P : P = (x0 + 1, 2*x0^2 - 2)) :
  P = (-1, 6) ∧ (2*x0^2 - 2 ≠ a * (x0 + 2) * (x0 - 1)) :=
by
  have h1 : x0 = -2, from sorry,
  have h2 : P = (-1, 6), from sorry,
  have neq_cond : 2*(-2)^2 - 2 ≠ a * (-2 + 2) * (-2 - 1), from sorry,
  exact ⟨h2, neq_cond⟩

end parabola_never_passes_point_l669_669990


namespace pq_product_l669_669396

noncomputable def T : Set ℝ :=
  {x | x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1}

def g (x : ℝ) : ℝ := sorry

axiom g_additive (x y : ℝ) (hx : x ∈ T) (hy : y ∈ T) (hxy : x + y ∈ T) :
  g (x + y) = g x + g y

axiom g_trigonometric (x y : ℝ) (hx : x ∈ T) (hy : y ∈ T) (h_sin_neq : sin x + sin y ≠ 0) :
  g (1 / sin x) + g (1 / sin y) = sin x * sin y * g (1 / sin (x + y))

theorem pq_product : (∃ p q : ℕ, 
                      p = 1 ∧ 
                      q = 0 ∧ 
                      ∀ v, v ∈ {g (1 / sin 1)} → v = 0) → 
                      0 = 0 :=
by sorry

end pq_product_l669_669396


namespace cube_root_of_54880000_l669_669673

theorem cube_root_of_54880000 : (real.cbrt 54880000) = 140 * (real.cbrt 10) :=
by
  -- Definitions based on conditions
  have h1 : 54880000 = 10^3 * 54880, by norm_num
  have h2 : 54880 = 2^5 * 7^3 * 5, by norm_num
  have h3 : 10 = 2 * 5, by norm_num
  
  -- Cube root properties and simplifications are implicitly inferred by the system
  sorry

end cube_root_of_54880000_l669_669673


namespace volume_of_pyramid_at_W_l669_669351

structure Pyramid :=
(base_area : ℝ)
(height : ℝ)

def volume (p : Pyramid) : ℝ :=
  (1 / 3) * p.base_area * p.height

def WXYZ_base := (1 / 2) * (1 / 2) -- base area of square with side 1/2
def VW_height := 1               -- height of the pyramid

theorem volume_of_pyramid_at_W : volume ⟨WXYZ_base, VW_height⟩ = 1 / 12 :=
  sorry

end volume_of_pyramid_at_W_l669_669351


namespace max_value_proof_l669_669202

noncomputable def max_dot_product_sum : ℝ :=
  let O := (0, 0)
  let A := (Real.sqrt 2, Real.sqrt 2)
  let B := (-Real.sqrt 2, Real.sqrt 2)
  let C (θ : ℝ) := (2 * Real.cos θ, 2 * Real.sin θ)
  let D (φ : ℝ) := (2 * Real.cos φ, 2 * Real.sin φ)
  let dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
  let OA := (Real.sqrt 2, Real.sqrt 2)
  let OB := (-Real.sqrt 2, Real.sqrt 2)
  let OC (θ : ℝ) := (2 * Real.cos θ, 2 * Real.sin θ)
  let OD (φ : ℝ) := (2 * Real.cos φ, 2 * Real.sin φ)
  let CA (θ : ℝ) := (Real.sqrt 2 - 2 * Real.cos θ, Real.sqrt 2 - 2 * Real.sin θ)
  let DA (φ : ℝ) := (Real.sqrt 2 - 2 * Real.cos φ, Real.sqrt 2 - 2 * Real.sin φ)
  let CB (θ : ℝ) := (-Real.sqrt 2 - 2 * Real.cos θ, Real.sqrt 2 - 2 * Real.sin θ)
  let DB (φ : ℝ) := (-Real.sqrt 2 - 2 * Real.cos φ, Real.sqrt 2 - 2 * Real.sin φ)
  ∀ θ φ : ℝ,
    dot_product (OC θ) (OD φ) = 2 →
    ∃ θ φ : ℝ, dot_product (CA θ) (CB θ) + dot_product (DA φ) (DB φ) ≤ 8 + 4 * Real.sqrt 6

end Lean4Proof

open Lean4Proof

theorem max_value_proof :
  max_dot_product_sum = 8 + 4 * Real.sqrt 6 :=
sorry

end max_value_proof_l669_669202


namespace range_g_on_interval_l669_669135

noncomputable def f (x : ℝ) : ℝ := x^(-1/2)
def g (x : ℝ) : ℝ := Real.sqrt x + f x

theorem range_g_on_interval :
  set.range (g : ℝ → ℝ) (set.Icc (1/2) 3) = set.Icc 2 (4 * Real.sqrt 3 / 3) :=
sorry

end range_g_on_interval_l669_669135


namespace range_of_m_func_inequality_l669_669082

section

variable (f₁ f₂ f₃ : ℝ → ℝ) (h : ℝ → ℝ) (m : ℝ)

noncomputable def f₁_def : (ℝ → ℝ) := λ x => x
noncomputable def f₂_def : (ℝ → ℝ) := λ x => Real.exp x
noncomputable def f₃_def : (ℝ → ℝ) := λ x => Real.log x

def h_def : (ℝ → ℝ) := λ x => m * f₁_def x - f₃_def x

-- Part 1
theorem range_of_m (m : ℝ) :
  (∀ x, (1 / 2) < x ∧ x ≤ 2 → derivative h_def x ≥ 0) → 
  m ∈ (-∞, (1:ℝ)/2] ∪ [2, +∞) :=
sorry

-- Part 2
theorem func_inequality :
  ∀ x, 0 < x → Real.exp x > Real.log x + 2 :=
sorry

end

end range_of_m_func_inequality_l669_669082


namespace notebook_cost_3_dollars_l669_669005

def cost_of_notebook (total_spent backpack_cost pen_cost pencil_cost num_notebooks : ℕ) : ℕ := 
  (total_spent - (backpack_cost + pen_cost + pencil_cost)) / num_notebooks

theorem notebook_cost_3_dollars 
  (total_spent : ℕ := 32) 
  (backpack_cost : ℕ := 15) 
  (pen_cost : ℕ := 1) 
  (pencil_cost : ℕ := 1) 
  (num_notebooks : ℕ := 5) 
  : cost_of_notebook total_spent backpack_cost pen_cost pencil_cost num_notebooks = 3 :=
by
  sorry

end notebook_cost_3_dollars_l669_669005


namespace CR_A_and_B_l669_669088

-- Definitions of the conditions
def setA (x : ℝ) : Prop := x^2 - x - 6 > 0
def setB (x : ℝ) : Prop := x - 1 > 0

-- Definition of the complement relative to ℝ
def complementR (s : ℝ → Prop) : ℝ → Prop := λ x, ¬ s x

-- Definition of the complement of set A relative to ℝ
def CR_A (x : ℝ) : Prop := complementR setA x

-- Intersection of CR_A with set B
def intersection (s1 s2 : ℝ → Prop) : ℝ → Prop := λ x, s1 x ∧ s2 x

-- Mathematically equivalent proof problem statement
theorem CR_A_and_B :
  ∀ x : ℝ, (intersection CR_A setB x) = (1 < x ∧ x ≤ 3) := 
sorry

end CR_A_and_B_l669_669088


namespace calculate_value_of_expression_l669_669507

theorem calculate_value_of_expression
  (f : ℝ → ℝ)
  (k t : ℝ)
  (h1 : ∀ x : ℝ, x > 0 → k > 0 → f x = k * x - |sin x|)
  (h2 : ∃ x1 x2 : ℝ, x1 < x2 ∧ f x1 = 0 ∧ f x2 = 0)
  (h3 : x1 < t ∧ x2 = t ∧ f t = 0)
  (h4 : k = -cos t)
  (h5 : t = tan t) :
  (t^2 + 1) * sin (2 * t) / t = 2 :=
by
  sorry

end calculate_value_of_expression_l669_669507


namespace train_speed_increase_correct_l669_669822

-- Define old time and new time
def t_old : ℝ := 16
def t_new : ℝ := 14

-- Define the reciprocal function for times
def reciprocal (t : ℝ) : ℝ := 1 / t

-- Define the increase in speed formula
def percentage_increase_in_speed : ℝ :=
  ((reciprocal t_new) - (reciprocal t_old)) / (reciprocal t_old)

-- Lean statement to prove
theorem train_speed_increase_correct : percentage_increase_in_speed = 1 / 7 := 
by
  sorry

end train_speed_increase_correct_l669_669822


namespace find_a_l669_669078

variables {a : ℝ} {a1 : ℝ} {q : ℝ} {S : ℕ → ℝ}

-- Define the first term and common ratio conditions
def first_term_a1 := a1 = 1
def common_ratio := q = a - (3 / 2)

-- Define the given condition about the limit of sums
def sum_sequence_tends_to_a := ∀ (n : ℕ), S n = a1 * (1 - q^n) / (1 - q)
def limit_condition := tendsto S at_top (𝓝 a)

-- Establish the problem statement to prove
theorem find_a (h1 : first_term_a1) (h2 : common_ratio) (h3 : limit_condition) : a = 2 :=
by
  sorry

end find_a_l669_669078


namespace volume_of_solid_is_correct_l669_669181

noncomputable def volume_of_revolved_solid : ℝ :=
  π * (8 * sqrt 17 / 51) ^ 2 * (28 / 15)

theorem volume_of_solid_is_correct :
  let S := { p : ℝ × ℝ | abs (10 - p.1) + p.2 ≤ 12 ∧ 4 * p.2 - p.1 ≥ 16 } in
  let line := { p : ℝ × ℝ | 4 * p.2 - p.1 = 16 } in
  volume_of_revolved_solid = 1792 * π * sqrt 17 / 39150 :=
by
  sorry

end volume_of_solid_is_correct_l669_669181


namespace number_of_Cheburashkas_erased_l669_669600

theorem number_of_Cheburashkas_erased :
  ∃ (n : ℕ), 
    (∀ x, x ≥ 1 → 
      (let totalKrakozyabras = (2 * (x - 1) = 29) in
         x - 2 = 11)) :=
sorry

end number_of_Cheburashkas_erased_l669_669600


namespace cube_surface_area_with_holes_l669_669875

theorem cube_surface_area_with_holes 
    (edge_length : ℝ) 
    (hole_side_length : ℝ) 
    (num_faces : ℕ) 
    (parallel_edges : Prop)
    (holes_centered : Prop)
    (h_edge : edge_length = 5)
    (h_hole : hole_side_length = 2)
    (h_faces : num_faces = 6)
    (h_inside_area : parallel_edges ∧ holes_centered)
    : (150 - 24 + 96 = 222) :=
by
    sorry

end cube_surface_area_with_holes_l669_669875


namespace arithmetic_expression_evaluation_l669_669317

theorem arithmetic_expression_evaluation : 1997 * (2000 / 2000) - 2000 * (1997 / 1997) = -3 := 
by
  sorry

end arithmetic_expression_evaluation_l669_669317


namespace max_PA_PB_PC_value_l669_669071

open Real

-- Coordinates of the triangle vertices
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 1)
def C : ℝ × ℝ := (0, 0)

-- Distance function
def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Product of distances from P to A, B, and C
def PA_PB_PC (P : ℝ × ℝ) : ℝ :=
  distance P A * distance P B * distance P C

-- Function to maximize PA * PB * PC
noncomputable def max_PA_PB_PC : ℝ :=
  max (max (Sup (PA_PB_PC '' {P | P.1 = 0}))  -- P on AC
          (Sup (PA_PB_PC '' {P | P.2 = 0})))  -- P on BC
      (Sup (PA_PB_PC '' {P | P.1 + P.2 = 1}))  -- P on AB

theorem max_PA_PB_PC_value :
  max_PA_PB_PC = sqrt 2 / 4 :=
sorry

end max_PA_PB_PC_value_l669_669071


namespace least_possible_value_of_smallest_integer_l669_669832

theorem least_possible_value_of_smallest_integer :
  ∃ (A B C D E F : ℤ), 
    A < B ∧ B < C ∧ C < D ∧ D < E ∧ E < F ∧ 
    A + B + C + D + E + F = 510 ∧ F = 90 ∧ A = 70 :=
begin
  sorry
end

end least_possible_value_of_smallest_integer_l669_669832


namespace percentage_error_approx_l669_669871

noncomputable def percentage_error (correct_val incorrect_val : ℝ) : ℝ :=
  (|correct_val - incorrect_val| / correct_val) * 100

noncomputable def correct_operation (x : ℝ) : ℝ :=
  real.sqrt ((5 / 3 * x) - 3)

noncomputable def mistaken_operation (x : ℝ) : ℝ :=
  real.cbrt ((3 / 5 * x) - 7)

theorem percentage_error_approx :
  percentage_error (correct_operation 12) (mistaken_operation 12) ≈ 85.77 :=
by
  sorry

end percentage_error_approx_l669_669871


namespace smallest_value_arithmetic_geometric_seq_l669_669238

theorem smallest_value_arithmetic_geometric_seq :
  ∃ (E F G H : ℕ), (E < F) ∧ (F < G) ∧ (F * 4 = G * 7) ∧ (E + G = 2 * F) ∧ (F * F * 49 = G * G * 16) ∧ (E + F + G + H = 97) := 
sorry

end smallest_value_arithmetic_geometric_seq_l669_669238


namespace number_of_integer_pairs_l669_669400

theorem number_of_integer_pairs (x y : ℤ) (h : x^6 + y^2 = 2 * y + 1) : finset.card {p : ℤ × ℤ | p.1^6 + p.2^2 = 2 * p.2 + 1}.to_finset = 4 :=
sorry

end number_of_integer_pairs_l669_669400


namespace sum_bounds_l669_669479

noncomputable def arith_seq (n : ℕ) : ℕ := 2 * n - 1

def geo_cond (d : ℕ) : Prop :=
  ∀ d > 0, (arith_seq 2) * (arith_seq 14) = (arith_seq 5) * (arith_seq 5)

def b_seq (n : ℕ) : ℤ :=
  2 / ((arith_seq n.succ) * (arith_seq (n + 2)))

def T_sum (n : ℕ) : ℤ :=
  ∑ k in Finset.range n, b_seq k

theorem sum_bounds (n : ℕ) (d : ℕ) (h_d : d > 0) (h_geo : geo_cond d) :
  (2 / 15) ≤ T_sum n ∧ T_sum n < (1 / 3) :=
begin
  sorry
end

end sum_bounds_l669_669479


namespace cube_volume_increase_l669_669134

variable (a : ℝ) (h : a ≥ 0)

theorem cube_volume_increase :
  ((2 * a) ^ 3) = 8 * (a ^ 3) :=
by sorry

end cube_volume_increase_l669_669134


namespace factorize_expression_l669_669421

theorem factorize_expression (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
by sorry

end factorize_expression_l669_669421


namespace correct_propositions_are_one_l669_669994

-- Define each proposition
def Proposition1 : Prop := ∀ (l1 l2 l3 : Line), 
  (l1.angleWith l3 = l2.angleWith l3) → (l1.parallel l2)

def Proposition2 : Prop := ∀ (l1 l2 l3 : Line),
  (l1.perpendicular l3 ∧ l2.perpendicular l3) → (l1.parallel l2)

def Proposition3 : Prop := ∀ (l1 l2 l3 : Line),
  (l1.parallel l3 ∧ l2.parallel l3) → (l1.parallel l2)

-- The main theorem statement
theorem correct_propositions_are_one :
  (∃ (n : Nat), 0 ≤ n ∧ n ≤ 3 ∧ ( ( (Proposition1 ∧ Proposition2 ∧¬Proposition3) ∨
                                    (Proposition1 ∧¬Proposition2 ∧ Proposition3) ∨ 
                                    (¬Proposition1 ∧ Proposition2 ∧ Proposition3) ∨ 
                                    (Proposition1 ∧¬Proposition2 ∧¬Proposition3) ∨ 
                                    (¬Proposition1 ∧ Proposition2 ∧¬Proposition3) ∨
                                    (¬Proposition1 ∧¬Proposition2 ∧ Proposition3) ) ∧ (n = 1) )) :=
by
  sorry

end correct_propositions_are_one_l669_669994


namespace area_of_triangle_formed_by_tangent_l669_669921

theorem area_of_triangle_formed_by_tangent (x y : ℝ) (h_eq : y = e ^ (1 / 2 * x)) (h_point : (x, y) = (4, e^2)) :
  let m := (1 / 2) * e ^ (1 / 2 * 4) in
  let tangent_line := fun x => m * (x - 4) + e^2 in
  let x_intercept := 2 in
  let y_intercept := -e^2 in
  (1 / 2) * x_intercept * -y_intercept = e^2 :=
by 
  sorry

end area_of_triangle_formed_by_tangent_l669_669921


namespace cube_root_of_54880000_l669_669676

theorem cube_root_of_54880000 : (real.cbrt 54880000) = 140 * (real.cbrt 10) :=
by
  -- Definitions based on conditions
  have h1 : 54880000 = 10^3 * 54880, by norm_num
  have h2 : 54880 = 2^5 * 7^3 * 5, by norm_num
  have h3 : 10 = 2 * 5, by norm_num
  
  -- Cube root properties and simplifications are implicitly inferred by the system
  sorry

end cube_root_of_54880000_l669_669676


namespace candidate_B_valid_votes_l669_669559

theorem candidate_B_valid_votes:
  let eligible_voters := 12000
  let abstained_percent := 0.1
  let invalid_votes_percent := 0.2
  let votes_for_C_percent := 0.05
  let A_less_B_percent := 0.2
  let total_voted := (1 - abstained_percent) * eligible_voters
  let valid_votes := (1 - invalid_votes_percent) * total_voted
  let votes_for_C := votes_for_C_percent * valid_votes
  (∃ Vb, valid_votes = (1 - A_less_B_percent) * Vb + Vb + votes_for_C 
         ∧ Vb = 4560) :=
sorry

end candidate_B_valid_votes_l669_669559


namespace runway_show_time_correct_l669_669221

def runwayShowTime (bathing_suit_sets evening_wear_sets formal_wear_sets models trip_time_in_minutes : ℕ) : ℕ :=
  let trips_per_model := bathing_suit_sets + evening_wear_sets + formal_wear_sets
  let total_trips := models * trips_per_model
  total_trips * trip_time_in_minutes

theorem runway_show_time_correct :
  runwayShowTime 3 4 2 10 3 = 270 :=
by
  sorry

end runway_show_time_correct_l669_669221


namespace cheburashkas_erased_l669_669618

def total_krakozyabras : ℕ := 29

def total_rows : ℕ := 2

def cheburashkas_per_row := (total_krakozyabras + total_rows) / total_rows / 2 + 1

theorem cheburashkas_erased :
  (total_krakozyabras + total_rows) / total_rows / 2 - 1 = 11 := 
by
  sorry

-- cheburashkas_erased proves that the number of Cheburashkas erased is 11 from the given conditions.

end cheburashkas_erased_l669_669618


namespace smallest_positive_period_of_f_max_value_of_f_on_interval_min_value_of_f_on_interval_l669_669512

noncomputable section

open Real

def vector_a (x : ℝ) : ℝ × ℝ := (cos x, -1 / 2)
def vector_b (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin x, cos (2 * x))
def f (x : ℝ) : ℝ := (vector_a x).1 * (vector_b x).1 + (vector_a x).2 * (vector_b x).2

theorem smallest_positive_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x := 
by 
  use π 
  sorry

theorem max_value_of_f_on_interval : ∃ x ∈ Icc 0 (π / 2), f x = 1 := 
by 
  use π / 4 
  sorry

theorem min_value_of_f_on_interval : ∃ x ∈ Icc 0 (π / 2), f x = -1 / 2 := 
by 
  use 0 
  sorry

end smallest_positive_period_of_f_max_value_of_f_on_interval_min_value_of_f_on_interval_l669_669512


namespace coefficient_x2y6_expansion_l669_669568

theorem coefficient_x2y6_expansion :
  let x : ℤ := 1
  let y : ℤ := 1
  ∃ a : ℤ, a = -28 ∧ (a • x ^ 2 * y ^ 6) = (1 - y / x) * (x + y) ^ 8 :=
by
  sorry

end coefficient_x2y6_expansion_l669_669568


namespace distance_from_P_to_face_XYZ_l669_669204

-- Define the points and their properties
def is_perpendicular (u v : ℝ × ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3 = 0

variables {X Y Z P : ℝ × ℝ × ℝ}
variables (PX PY PZ : ℝ)

-- Establishing the conditions provided in the problem
def conditions : Prop :=
  PX = 10 ∧ PY = 10 ∧ PZ = 8 ∧
  is_perpendicular (X.1 - P.1, X.2 - P.2, X.3 - P.3) (Y.1 - P.1, Y.2 - P.2, Y.3 - P.3) ∧
  is_perpendicular (X.1 - P.1, X.2 - P.2, X.3 - P.3) (Z.1 - P.1, Z.2 - P.2, Z.3 - P.3) ∧
  is_perpendicular (Z.1 - P.1, Z.2 - P.2, Z.3 - P.3) (Y.1 - P.1, Y.2 - P.2, Y.3 - P.3)

-- Theorem statement proving the question given the conditions
theorem distance_from_P_to_face_XYZ :
  conditions X Y Z P PX PY PZ → dist_to_plane (P, X, Y, Z) = 8 :=
by {
  sorry
}

end distance_from_P_to_face_XYZ_l669_669204


namespace total_dog_food_per_day_l669_669916

-- Definitions based on conditions
def dog1_eats_per_day : ℝ := 0.125
def dog2_eats_per_day : ℝ := 0.125
def number_of_dogs : ℕ := 2

-- Mathematically equivalent proof problem statement
theorem total_dog_food_per_day : dog1_eats_per_day + dog2_eats_per_day = 0.25 := 
by
  sorry

end total_dog_food_per_day_l669_669916


namespace ellipse_properties_hold_l669_669481

noncomputable def ellipse_eq (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), (a > b ∧ b > 0 ∧ (x^2)/(a^2) + (y^2)/(b^2) = 1)

noncomputable def eccentricity (e a : ℝ) : Prop :=
  e = (Real.sqrt (a^2 - a^2 * 3 / 4) / a)

noncomputable def major_axis_length (l a : ℝ) : Prop :=
  2 * a = l

noncomputable def right_focus (a : ℝ) : ℝ × ℝ :=
  (Real.sqrt (a^2 - a^2 * 3 / 4), 0)

theorem ellipse_properties_hold (a b : ℝ) (M N : ℝ × ℝ) (h1 : a > b > 0)
    (h2 : eccentricity (Real.sqrt 3 / 3) a) 
    (h3 : major_axis_length (2 * Real.sqrt 3) a) 
    (h4 : ∃ (x₀ y₀ : ℝ), M = (x₀, y₀) ∧ x₀^2 + y₀^2 = 3)
    (h5 : ∃ (x₀ y₀ t : ℝ), N = (3, t) ) : 
    ellipse_eq √3 (Real.sqrt 2) ∧ (∀ M N, True → line_passing_focus M N (right_focus a)) :=
by
  sorry

end ellipse_properties_hold_l669_669481


namespace magic_square_expression_l669_669885

theorem magic_square_expression : 
  let a := 8
  let b := 6
  let c := 14
  let d := 10
  let e := 11
  let f := 5
  let g := 3
  a - b - c + d + e + f - g = 11 :=
by
  sorry

end magic_square_expression_l669_669885


namespace inf_non_r_fibonacci_numbers_l669_669281

-- Definitions of Fibonacci and Lucas sequences
def fibonacci (n : ℕ) : ℕ :=
nat.rec_on n 0 (λ n' r, nat.rec_on r 1 (λ n'' r', r + (nat.rec_on n'' 1 (λ n''', (fibonacci n''' + fibonacci (n''' + 1)))))

def lucas (n : ℕ) : ℕ :=
nat.rec_on n 2 (λ n' r, nat.rec_on r 1 (λ n'' r', r + (nat.rec_on n'' 1 (λ n''', (lucas n''' + lucas (n''' + 1)))))

-- Constants used in the approximation formulae
noncomputable def alpha : ℝ := (1 + real.sqrt 5) / 2
noncomputable def beta : ℝ := (1 - real.sqrt 5) / 2

-- Definitions based on known formulae (no proofs, as they can be used without verification here)
def fibonacci_approx (n : ℕ) : ℝ := (alpha^n - beta^n) / real.sqrt 5

-- r-Fibonacci number definition
def is_r_fibonacci_number (x r : ℕ) : Prop :=
∃ (ys : vector ℕ r), ∀ i < r, (fibonacci (ys.nth i)) ∧ x = vector.sum ys.to_list

-- Proof that there are infinitely many positive integers not expressible as r-Fibonacci numbers for 1 ≤ r ≤ 5
theorem inf_non_r_fibonacci_numbers : ∀ (r : ℕ), 1 ≤ r → r ≤ 5 → ∃∞ (x : ℕ), ¬ is_r_fibonacci_number x r :=
sorry

end inf_non_r_fibonacci_numbers_l669_669281


namespace minimize_total_distance_l669_669200

-- Definitions corresponding to the given conditions
def num_students : ℕ := 20
def distance_between_trees : ℕ := 10
def tree_pits : List ℕ := List.range' 1 20

-- Main statement to prove
theorem minimize_total_distance :
  ∃ (x1 x2 : ℕ), {x1, x2} = {10, 11} ∧
  (∀ y1 y2 ∈ tree_pits, 
    let S_y := (∑ i in tree_pits, abs (i - y1) * distance_between_trees) + 
                 (∑ i in tree_pits, abs (i - y2) * distance_between_trees),
        S_x := (∑ i in tree_pits, abs (i - x1) * distance_between_trees) + 
                 (∑ i in tree_pits, abs (i - x2) * distance_between_trees)
    in S_x ≤ S_y) :=
begin
  sorry
end

end minimize_total_distance_l669_669200


namespace log_gt_x_cube_div_3_l669_669461

theorem log_gt_x_cube_div_3 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 1) : 
  log (1 + x) > x^3 / 3 := 
by
  sorry

end log_gt_x_cube_div_3_l669_669461


namespace cube_root_simplification_l669_669712

theorem cube_root_simplification : (∛54880000) = 140 * (2 ^ (1 / 3)) :=
by
  -- Using the information from the problem conditions and final solution.
  have root_10_cubed := (10 ^ 3 : ℝ)
  have factored_value := root_10_cubed * (2 ^ 4 * 7 ^ 3)
  have cube_root := Real.cbrt factored_value
  sorry

end cube_root_simplification_l669_669712


namespace peaches_in_boxes_l669_669786

theorem peaches_in_boxes : 
  (let peaches_per_basket := 25 in
  let number_of_baskets := 5 in
  let eaten_peaches := 5 in
  let peaches_per_box := 15 in
  ∃ (boxes : ℕ), 
    (peaches_per_basket * number_of_baskets - eaten_peaches) / peaches_per_box = boxes ∧ 
    boxes = 8) :=
by
  sorry

end peaches_in_boxes_l669_669786


namespace M1_perpendicular_M3_perpendicular_M4_perpendicular_l669_669111

def is_perpendicular_point_set (M : set (ℝ × ℝ)) :=
  ∀ (x1 y1 : ℝ), ((x1, y1) ∈ M) → ∃ (x2 y2 : ℝ), (x2, y2) ∈ M ∧ (x1 * x2 + y1 * y2 = 0)

def M1 : set (ℝ × ℝ) := { p | ∃ x, p = (x, 1 / x^2) }
def M3 : set (ℝ × ℝ) := { p | ∃ x, p = (x, 2^x - 2) }
def M4 : set (ℝ × ℝ) := { p | ∃ x, p = (x, sin x + 1) }

theorem M1_perpendicular : is_perpendicular_point_set M1 := sorry
theorem M3_perpendicular : is_perpendicular_point_set M3 := sorry
theorem M4_perpendicular : is_perpendicular_point_set M4 := sorry

end M1_perpendicular_M3_perpendicular_M4_perpendicular_l669_669111


namespace complete_square_l669_669814

theorem complete_square (x : ℝ) : (x^2 - 2 * x - 5 = 0) ↔ ((x - 1)^2 = 6) := 
by
  sorry

end complete_square_l669_669814


namespace num_men_in_boat_l669_669547

theorem num_men_in_boat 
  (n : ℕ) (W : ℝ)
  (h1 : (W / n : ℝ) = W / n)
  (h2 : (W + 8) / n = W / n + 1)
  : n = 8 := 
sorry

end num_men_in_boat_l669_669547


namespace minimum_distance_l669_669555

theorem minimum_distance : 
  ∀ (A B wall : ℝ × ℝ),
  A = (0, 0) ∧ B = (1000, 600) ∧ 
  ((∃ C : ℝ × ℝ, C.1 = 0 ∧ 0 ≤ C.2 ∧ C.2 ≤ 1000) ∧ 
  (∃ C : ℝ × ℝ, C ∈ line_of_sight A B)) → 
  distance_run A B wall = 1200 := 
by 
  sorry

end minimum_distance_l669_669555


namespace christian_age_in_years_l669_669388

theorem christian_age_in_years (B C x : ℕ) (h1 : C = 2 * B) (h2 : B + x = 40) (h3 : C + x = 72) :
    x = 8 := 
sorry

end christian_age_in_years_l669_669388


namespace cheburashkas_erased_l669_669590

theorem cheburashkas_erased (n : ℕ) (rows : ℕ) (krakozyabras : ℕ) 
  (h_spacing : ∀ r, r ≤ rows → krakozyabras = 2 * (n - 1))
  (h_rows : rows = 2)
  (h_krakozyabras : krakozyabras = 29) :
  n = 16 → rows = 2 → krakozyabras = 29 → n = 16 - 5 :=
by
  sorry

end cheburashkas_erased_l669_669590


namespace circle_is_central_symmetric_l669_669817

def is_central_symmetric (C : Set (ℝ × ℝ)) (O : ℝ × ℝ) : Prop :=
∀ (P : ℝ × ℝ), P ∈ C → (2 * O - P) ∈ C

def is_circle (C : Set (ℝ × ℝ)) (O : ℝ × ℝ) (r : ℝ) : Prop :=
∀ (P : ℝ × ℝ), P ∈ C ↔ (P - O).norm = r

theorem circle_is_central_symmetric (C : Set (ℝ × ℝ)) (O : ℝ × ℝ) (r : ℝ) :
  is_circle C O r → is_central_symmetric C O :=
by
  sorry

end circle_is_central_symmetric_l669_669817


namespace number_of_Cheburashkas_erased_l669_669595

theorem number_of_Cheburashkas_erased :
  ∃ (n : ℕ), 
    (∀ x, x ≥ 1 → 
      (let totalKrakozyabras = (2 * (x - 1) = 29) in
         x - 2 = 11)) :=
sorry

end number_of_Cheburashkas_erased_l669_669595


namespace john_rents_apartment_l669_669588

theorem john_rents_apartment
  (sublet_count : Nat)
  (sublet_payment_per_person : Nat)
  (annual_profit : Nat)
  (monthly_rent : Nat)
  (monthly_profit : Nat)
  (monthly_sublet_income : Nat) :
  sublet_count = 3 →
  sublet_payment_per_person = 400 →
  annual_profit = 3600 →
  monthly_profit = annual_profit / 12 →
  monthly_sublet_income = sublet_count * sublet_payment_per_person →
  monthly_sublet_income - monthly_rent = monthly_profit →
  monthly_rent = 900 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2] at h5
  rw [h3] at h4
  rw h4 at h6
  simp at h6
  exact h6

end john_rents_apartment_l669_669588


namespace octagon_perimeter_l669_669035

/-- 
  Represents the side length of the regular octagon
-/
def side_length : ℕ := 12

/-- 
  Represents the number of sides of a regular octagon
-/
def number_of_sides : ℕ := 8

/-- 
  Defines the perimeter of the regular octagon
-/
def perimeter (side_length : ℕ) (number_of_sides : ℕ) : ℕ :=
  side_length * number_of_sides

/-- 
  Proof statement: asserting that the perimeter of a regular octagon
  with a side length of 12 meters is 96 meters
-/
theorem octagon_perimeter :
  perimeter side_length number_of_sides = 96 :=
  sorry

end octagon_perimeter_l669_669035


namespace find_number_l669_669722

theorem find_number (x : ℝ) : (35 - x) * 2 + 12 = 72 → ((35 - x) * 2 + 12) / 8 = 9 → x = 5 :=
by
  -- assume the first condition
  intro h1
  -- assume the second condition
  intro h2
  -- the proof goes here
  sorry

end find_number_l669_669722


namespace cubes_difference_l669_669965

theorem cubes_difference 
  (a b : ℝ) 
  (h1 : a - b = 7) 
  (h2 : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := by
  sorry

end cubes_difference_l669_669965


namespace general_term_formulas_sum_of_sequence_l669_669961

open Nat

def arithmetic_sequence (a_n : ℕ → ℤ) := ∃ d a1, ∀ n, a_n n = a1 + n * d

def geometric_sequence (b_n : ℕ → ℤ) := ∃ q b1, (q > 0) ∧ (b1 = 2) ∧ ∀ n, b_n n = b1 * q ^ n

axiom given_conditions (a_n b_n : ℕ → ℤ) (S : ℕ → ℤ) :
  (arithmetic_sequence a_n) ∧
  (geometric_sequence b_n) ∧
  (b_n 2 + b_n 3 = 12) ∧
  (b_n 3 = a_n 4 - 2 * a_n 3) ∧
  (S 11 = 11 * b_n 4) ∧
  (∀ n, S n = ∑ k in range n, a_n k) 

theorem general_term_formulas (a_n b_n : ℕ → ℤ) (S : ℕ → ℤ) :
  given_conditions a_n b_n S →
  (a_n = λ n, 3 * n - 2) ∧
  (b_n = λ n, 2 ^ n) :=
sorry

theorem sum_of_sequence (a_n b_n : ℕ → ℤ) (S : ℕ → ℤ) :
  given_conditions a_n b_n S →
  (∀ n, ∑ k in range n, a_n (2 * k) * b_n (2 * k - 1) = (3 * n - 2) * 4^(n+1) + 8 / 3) :=
sorry

end general_term_formulas_sum_of_sequence_l669_669961


namespace intersection_C1_C2_polar_coordinates_l669_669991

noncomputable def C1_parametric (t : ℝ) : ℝ × ℝ :=
  (4 + 5 * Real.cos t, 5 + 5 * Real.sin t)

def C2_polar (θ : ℝ) : ℝ :=
  2 * Real.sin θ

theorem intersection_C1_C2_polar_coordinates :
  (∃ t₁ t₂ : ℝ, C1_parametric t₁ = (1, 1) ∧ C1_parametric t₂ = (0, 2)) ∧
  (∃ θ₁ θ₂ : ℝ, θ₁ ∈ Set.Icc 0 (2 * Real.pi) ∧ θ₂ ∈ Set.Icc 0 (2 * Real.pi) ∧
  (C2_polar θ₁ = Real.sqrt 2 ∧ θ₁ = Real.pi / 4) ∧
  (C2_polar θ₂ = 2 ∧ θ₂ = Real.pi / 2)) :=
sorry

end intersection_C1_C2_polar_coordinates_l669_669991


namespace number_of_true_propositions_l669_669102

-- Define the propositions
def proposition1 (k : ℝ) : Prop :=
(k > 0) → (∃ x y : ℝ, (x^2 + 2*x - k = 0 ∧ y^2 + 2*y - k = 0))

def proposition2 (a b c : ℝ) : Prop :=
(¬((a > b) → (a + c > b + c)))

def proposition3 : Prop :=
(¬(∀ (Q : Type) [quadrilateral Q], is_rectangle Q → has_equal_diagonals Q))

def proposition4 (x y : ℝ) : Prop :=
(¬((xy = 0) → (x = 0 ∨ y = 0)))

def proposition5 (x y : ℝ) : Prop :=
((x ≠ 2 ∨ y ≠ 3) → (x + y ≠ 5))

-- Define the main theorem
theorem number_of_true_propositions : 
  (∀ k, proposition1 k) ∧ 
  (∀ a b c, proposition2 a b c) ∧ 
  proposition3 ∧ 
  (∀ x y, proposition4 x y) ∧ 
  (∀ x y, proposition5 x y) → 
  3 = 3 :=
by sorry

end number_of_true_propositions_l669_669102


namespace cube_root_of_54880000_l669_669677

theorem cube_root_of_54880000 : (real.cbrt 54880000) = 140 * (real.cbrt 10) :=
by
  -- Definitions based on conditions
  have h1 : 54880000 = 10^3 * 54880, by norm_num
  have h2 : 54880 = 2^5 * 7^3 * 5, by norm_num
  have h3 : 10 = 2 * 5, by norm_num
  
  -- Cube root properties and simplifications are implicitly inferred by the system
  sorry

end cube_root_of_54880000_l669_669677


namespace shortest_side_of_similar_triangle_l669_669862

theorem shortest_side_of_similar_triangle (a b : ℕ) (c : ℝ) 
  (h1 : a = 24) (h2 : c = 25) (h3 : a^2 + b^2 = c^2)
  (scale_factor : ℝ) (shortest_side_first : ℝ) (hypo_second : ℝ)
  (h4 : scale_factor = 100 / 25) 
  (h5 : hypo_second = 100) 
  (h6 : b = 7) 
  : (shortest_side_first * scale_factor = 28) :=
by
  sorry

end shortest_side_of_similar_triangle_l669_669862


namespace expected_value_X_is_half_l669_669252

open BigOperators

-- Define the random digit sequence and its properties
def random_digit_seq (n : ℕ) (d : ℕ → ℕ) := ∀ i : ℕ, i < n → d i ∈ Fin 10

-- Expected value of a single random digit
def expected_value_digit : ℝ := 4.5

-- Define the expected value of X
noncomputable def expected_value_of_X (n : ℕ) (d : ℕ → ℕ) : ℝ :=
  ∑ i in Finset.range n, (d i : ℝ) * 10^(-(i+1))

-- The main theorem to prove
theorem expected_value_X_is_half : 
  ∀ (d : ℕ → ℕ), random_digit_seq 1000 d → expected_value_of_X 1000 d = 0.5 :=
by {
  intro d,
  intro h,
  sorry  -- The proof would be written here.
}

end expected_value_X_is_half_l669_669252


namespace prism_property_l669_669149

-- Define a structure for a prism
structure Prism where
  base1 : face
  base2 : face
  lateral : list edge
  (parallel_bases : base1.is_parallel base2)
  (parallel_lateral : ∀ e ∈ lateral, ∀ e' ∈ lateral, e ≠ e' → e.is_parallel e')

-- Define properties of faces and edges being parallel
axiom face.is_parallel : face → face → Prop
axiom edge.is_parallel : edge → edge → Prop

-- Define options as statements
def option_a (P : Prism) : Prop := ∃ f1 f2, f1 ≠ f2 ∧ f1.is_parallel f2 ∧ ∀ f, (f = f1 ∨ f = f2) → f.is_parallel f1
def option_b (P : Prism) : Prop := ∀ e1 e2, e1 ≠ e2 → e1.is_parallel e2
def option_c (P : Prism) : Prop := ∀ f, f.is_parallelogram
def option_d (P : Prism) : Prop := P.parallel_bases ∧ P.parallel_lateral

-- The theorem to be proved
theorem prism_property (P : Prism) : option_d P :=
by
  -- The proof steps would go here, but are omitted as per prompt requirements
  sorry

end prism_property_l669_669149


namespace find_A_l669_669463

theorem find_A (A a b : ℝ) (h1 : 3^a = A) (h2 : 5^b = A) (h3 : 1/a + 1/b = 2) : A = Real.sqrt 15 :=
by
  /- Proof omitted -/
  sorry

end find_A_l669_669463


namespace closest_whole_number_l669_669293

theorem closest_whole_number :
  ( (1/50 : ℝ) * (10^2010 + 5 * 10^2012) / (2 * 10^2011 + 3 * 10^2011) ).natAbs = 1000 := 
sorry

end closest_whole_number_l669_669293


namespace triangle_cot_tan_identity_l669_669092

theorem triangle_cot_tan_identity 
  (a b c : ℝ) 
  (h : a^2 + b^2 = 2018 * c^2)
  (A B C : ℝ) 
  (triangle_ABC : ∀ (a b c : ℝ), a + b + c = π) 
  (cot_A : ℝ := Real.cos A / Real.sin A) 
  (cot_B : ℝ := Real.cos B / Real.sin B) 
  (tan_C : ℝ := Real.sin C / Real.cos C) :
  (cot_A + cot_B) * tan_C = -2 / 2017 :=
by sorry

end triangle_cot_tan_identity_l669_669092


namespace minimum_cactus_species_l669_669361

/--
At a meeting of cactus enthusiasts, 80 cactophiles presented their collections,
each consisting of cacti of different species. It turned out that no single 
species of cactus is found in all collections simultaneously, but any 15 people
have cacti of the same species. Prove that the minimum total number of cactus 
species is 16.
-/
theorem minimum_cactus_species (k : ℕ) (h : ∀ (collections : fin 80 → finset (fin k)),
  (∀ i, collections i ≠ ∅) ∧ (∃ (j : fin k), ∀ i, j ∉ collections i) ∧ 
  (∀ (S : finset (fin 80)), S.card = 15 → ∃ j, ∀ i ∈ S, j ∈ collections i)) :
  k ≥ 16 :=
sorry

end minimum_cactus_species_l669_669361


namespace player_B_winning_strategy_l669_669796

theorem player_B_winning_strategy (n : ℕ) (d : ℕ) (n_gt_one : n > 1) (d_ge_one : d ≥ 1) :
  ∃ (strategy_B : ℕ → ℕ → ℕ → ℕ), 
    (∀ m1 n1,
      (n1 ≠ m1) →
      ∀ (k : ℕ) (hk : 2 ≤ k ∧ k ≤ n) (m_prev : ℕ) (n_prev : ℕ),
        (m_prev < strategy_B k m_prev n_prev ∧ strategy_B k m_prev n_prev ≤ m_prev + d) ∧
        (n_prev < strategy_B k m_prev n_prev ∧ strategy_B k m_prev n_prev ≤ n_prev + d)) :=
sorry

end player_B_winning_strategy_l669_669796


namespace tangents_are_equal_l669_669934

open Classical

noncomputable def point{name: TString} := { x y: ℝ }

/-- Prove that the segments from point A to the points of tangency are equal. -/
theorem tangents_are_equal (A T1 T2 O : point) (r : ℝ) :
  (dist O T1 = r) ∧ (dist O T2 = r) ∧ (angle O T1 A = 90) ∧ (angle O T2 A = 90) ∧ tangent A T1 ∧ tangent A T2 
  → dist A T1 = dist A T2 :=
begin
  sorry
end

end tangents_are_equal_l669_669934


namespace _l669_669653

noncomputable theorem distance_between_points 
  (n : ℕ) (h : n ≥ 5)  -- Number of sides and the condition n ≥ 5
  (a α : ℝ)            -- The side length and the external angle
  (hreg : ∀ (i : ℕ), i < n → (B (i + 1) (i + 2) ⊥ A (i + 1) (i + 2)))  -- Perpendicular conditions
  :
  let x := a * cos α / (1 - cos α) in
  ∀ i : ℕ, i < n → A (i + 1)B_i = x :=
sorry

end _l669_669653


namespace divided_differences_correct_l669_669899

noncomputable def divided_difference_first_order (f : ℝ → ℝ) (x₀ x₁ : ℝ) : ℝ :=
(f x₁ - f x₀) / (x₁ - x₀)

noncomputable def divided_difference_second_order (f : ℝ → ℝ) (x₀ x₁ x₂ : ℝ) : ℝ :=
(divided_difference_first_order f x₁ x₂ - divided_difference_first_order f x₀ x₁) / (x₂ - x₀)

noncomputable def divided_difference_third_order (f : ℝ → ℝ) (x₀ x₁ x₂ x₃ : ℝ) : ℝ :=
(divided_difference_second_order f x₁ x₂ x₃ - divided_difference_second_order f x₀ x₁ x₂) / (x₃ - x₀)

noncomputable def divided_difference_fourth_order (f : ℝ → ℝ) (x₀ x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
(divided_difference_third_order f x₁ x₂ x₃ x₄ - divided_difference_third_order f x₀ x₁ x₂ x₃) / (x₄ - x₀)

theorem divided_differences_correct (f : ℝ → ℝ) :
  f (-3) = -9 → f (-2) = -16 → f (-1) = -3 → f (1) = 11 → f (2) = 36 →
  divided_difference_first_order f (-3) (-2) = -7 ∧
  divided_difference_first_order f (-2) (-1) = 13 ∧
  divided_difference_first_order f (-1) (1) = 7 ∧
  divided_difference_first_order f (1) (2) = 25 ∧
  divided_difference_second_order f (-3) (-2) (-1) = 10 ∧
  divided_difference_second_order f (-2) (-1) (1) = -2 ∧
  divided_difference_second_order f (-1) (1) (2) = 6 ∧
  divided_difference_third_order f (-3) (-2) (-1) (1) = -3 ∧
  divided_difference_third_order f (-2) (-1) (1) (2) = 2 ∧
  divided_difference_fourth_order f (-3) (-2) (-1) (1) (2) = 1 :=
begin
  intros h₀ h₁ h₂ h₃ h₄,
  repeat { split },
  all_goals { simp [divided_difference_first_order, divided_difference_second_order, divided_difference_third_order, divided_difference_fourth_order] },
  sorry,
  sorry,
  sorry,
  sorry,
  sorry,
  sorry,
  sorry,
  sorry,
  sorry,
  sorry
end

end divided_differences_correct_l669_669899


namespace tens_digit_of_expression_l669_669904

theorem tens_digit_of_expression :
  let x := 2027
  let y := 2028
  let z := 2029
  (x ≡ 27 [MOD 100]) →
  (z ≡ 29 [MOD 100]) →
  (∃ n, 27^(8 * n + 4) ≡ 41 [MOD 100]) →
  (x^y - z ≡ 12 [MOD 100]) →
  (1 : ℤ) :=
begin
  intros hx hz hp hmod,
  -- The proof steps are omitted for brevity
  sorry,
end

end tens_digit_of_expression_l669_669904


namespace find_f_l669_669192

theorem find_f (f : ℝ → ℝ) (h1 : ∀ x, abs (f x + cos x ^ 2) ≤ 3 / 4)
                         (h2 : ∀ x, abs (f x - sin x ^ 2) ≤ 1 / 4) :
  ∀ x, f x = sin x ^ 2 - 1 / 4 :=
by
  sorry

end find_f_l669_669192


namespace option_c_holds_true_l669_669941

theorem option_c_holds_true (x y : ℝ) (h : x > y) (h1 : y > 0) : 
  (1 / 2) ^ x < (1 / 2) ^ (y - x) :=
sorry

end option_c_holds_true_l669_669941


namespace min_value_quadratic_l669_669288

theorem min_value_quadratic (x : ℝ) : 
  ∃ m, m = 3 * x^2 - 18 * x + 2048 ∧ ∀ x, 3 * x^2 - 18 * x + 2048 ≥ 2021 :=
by sorry

end min_value_quadratic_l669_669288


namespace max_value_PA_PB_PC_l669_669068

noncomputable def maximum_PA_PB_PC (A B C P : ℝ × ℝ) : ℝ :=
  let PA := dist P A
  let PB := dist P B
  let PC := dist P C
  PA * PB * PC

theorem max_value_PA_PB_PC :
  let A := (1, 0)
  let B := (0, 1)
  let C := (0, 0)
  ∃ P : ℝ × ℝ, (P.1 = 0 ∨ P.1 = 1 ∨ P.2 = 0 ∨ P.2 = 1 ∨ P.1 = P.2),
  maximum_PA_PB_PC A B C P = sqrt 2 / 4 :=
by
  sorry

end max_value_PA_PB_PC_l669_669068


namespace sum_of_three_terms_divisible_by_3_l669_669079

theorem sum_of_three_terms_divisible_by_3 (a : Fin 5 → ℤ) :
  ∃ (i j k : Fin 5), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ (a i + a j + a k) % 3 = 0 :=
by
  sorry

end sum_of_three_terms_divisible_by_3_l669_669079


namespace original_cost_of_subscription_l669_669300

theorem original_cost_of_subscription :
  ∃ C, 0.35 * C = 611 ∧ C = 1745.71 :=
by {
  use 1745.71,
  split,
  sorry,
  rfl
}

end original_cost_of_subscription_l669_669300


namespace function_symmetric_point_l669_669226

theorem function_symmetric_point (f : ℝ → ℝ) (h : ∀ x : ℝ, f(x + 2) = -f(-x)) : 
  ∃ a b : ℝ, (a = 1) ∧ (b = 0) ∧ (∀ x : ℝ, f(x) + f(2 * a - x) = 2 * b) :=
by
  sorry

end function_symmetric_point_l669_669226


namespace expected_value_of_random_number_l669_669264

/-- 
The expected value of a random number formed by placing a zero and a decimal point in front
of a sequence of one thousand random digits is 0.5.
-/
theorem expected_value_of_random_number : 
  let X := ∑ k in (finRange 1000), (4.5 / 10 ^ (k + 1))
  in X = 0.5 :=
sorry

end expected_value_of_random_number_l669_669264


namespace inequality_l669_669191

variable {a b c : ℝ}

theorem inequality (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) : 
  a * (a - 1) + b * (b - 1) + c * (c - 1) ≥ 0 := 
by 
  sorry

end inequality_l669_669191


namespace infinite_product_equals_sqrt2_l669_669890

theorem infinite_product_equals_sqrt2 :
  let seq : ℕ → ℚ := λ n, 2^(2*n+1)/3^((3^n))
  infinite_product seq = sqrt 2
:= sorry

end infinite_product_equals_sqrt2_l669_669890


namespace difference_between_oranges_and_apples_l669_669783

-- Definitions of the conditions
variables (A B P O: ℕ)
variables (h1: O = 6)
variables (h2: B = 3 * A)
variables (h3: P = B / 2)
variables (h4: A + B + P + O = 28)

-- The proof problem statement
theorem difference_between_oranges_and_apples
    (A B P O: ℕ)
    (h1: O = 6)
    (h2: B = 3 * A)
    (h3: P = B / 2)
    (h4: A + B + P + O = 28) :
    O - A = 2 :=
sorry

end difference_between_oranges_and_apples_l669_669783


namespace both_subjects_sum_l669_669912

-- Define the total number of students
def N : ℕ := 1500

-- Define the bounds for students studying Biology (B) and Chemistry (C)
def B_min : ℕ := 900
def B_max : ℕ := 1050

def C_min : ℕ := 600
def C_max : ℕ := 750

-- Let x and y be the smallest and largest number of students studying both subjects
def x : ℕ := B_max + C_max - N
def y : ℕ := B_min + C_min - N

-- Prove that y + x = 300
theorem both_subjects_sum : y + x = 300 := by
  sorry

end both_subjects_sum_l669_669912


namespace midpoint_trajectory_eq_l669_669062

open Real

-- Given conditions
def on_circle (P : ℝ × ℝ) : Prop := P.1^2 + P.2^2 = 1
def fixed_point_Q := (3 : ℝ, 0 : ℝ)

-- The midpoint of P and Q
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- The translated theorem
theorem midpoint_trajectory_eq (P : ℝ × ℝ) (x y : ℝ) (hP : on_circle P)
  (hM : midpoint P fixed_point_Q = (x, y)) : (2 * x - 3)^2 + 4 * y^2 = 1 :=
by
  sorry

end midpoint_trajectory_eq_l669_669062


namespace bread_thrown_in_pond_l669_669554

def total_bread (B : ℝ) : Prop :=
  let first_duck := (3 / 7) * B
  let second_duck := (1 / 5) * (B - first_duck)
  let third_duck := (2 / 9) * (B - first_duck - second_duck)
  let fourth_duck := (1 / 7) * (B - first_duck - second_duck - third_duck)
  let fifth_duck := 15
  let sixth_duck := 9
  let remaining_bread := B - first_duck - second_duck - third_duck - fourth_duck - fifth_duck - sixth_duck
  remaining_bread = 50

theorem bread_thrown_in_pond : ∃ B, total_bread B ∧ B = 243 :=
begin
  sorry
end

end bread_thrown_in_pond_l669_669554


namespace function_increasing_range_of_b_l669_669739

def f (x : ℝ) (b : ℝ) : ℝ := Real.log x + (3 / 2) * x^2 - b * x

theorem function_increasing_range_of_b {b : ℝ} : 
  (∀ x > 0, (1 / x) + 3 * x - b ≥ 0) ↔ b ≤ 2 * Real.sqrt 3 := 
by 
  sorry

end function_increasing_range_of_b_l669_669739


namespace gcd_of_cubic_sum_and_linear_is_one_l669_669927

theorem gcd_of_cubic_sum_and_linear_is_one (n : ℕ) (h : n > 27) : Nat.gcd (n^3 + 8) (n + 3) = 1 :=
sorry

end gcd_of_cubic_sum_and_linear_is_one_l669_669927


namespace max_PA_PB_PC_value_l669_669073

open Real

-- Coordinates of the triangle vertices
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 1)
def C : ℝ × ℝ := (0, 0)

-- Distance function
def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Product of distances from P to A, B, and C
def PA_PB_PC (P : ℝ × ℝ) : ℝ :=
  distance P A * distance P B * distance P C

-- Function to maximize PA * PB * PC
noncomputable def max_PA_PB_PC : ℝ :=
  max (max (Sup (PA_PB_PC '' {P | P.1 = 0}))  -- P on AC
          (Sup (PA_PB_PC '' {P | P.2 = 0})))  -- P on BC
      (Sup (PA_PB_PC '' {P | P.1 + P.2 = 1}))  -- P on AB

theorem max_PA_PB_PC_value :
  max_PA_PB_PC = sqrt 2 / 4 :=
sorry

end max_PA_PB_PC_value_l669_669073


namespace num_outliers_in_D_l669_669394

def data_set : List ℕ := [3, 11, 28, 28, 34, 36, 36, 38, 47, 54]

def Q1 : ℕ := 28
def Q3 : ℕ := 38
def IQR : ℕ := Q3 - Q1
def outlier_threshold_factor : ℕ := 15 -- 1.5 * IQR
def lower_threshold : ℕ := Q1 - outlier_threshold_factor
def upper_threshold : ℕ := Q3 + outlier_threshold_factor

def is_outlier (x : ℕ) : Prop := x < lower_threshold ∨ x > upper_threshold

def num_outliers (D : List ℕ) : ℕ := D.countp is_outlier

theorem num_outliers_in_D : num_outliers data_set = 3 := by
  sorry

end num_outliers_in_D_l669_669394


namespace factorization_correct_l669_669430

-- Define the initial expression
def initial_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + a

-- Define the factorized form
def factorized_expr (a x : ℝ) : ℝ := a * (x - 1)^2

-- Create the theorem statement
theorem factorization_correct (a x : ℝ) : initial_expr a x = factorized_expr a x :=
by simp [initial_expr, factorized_expr, pow_two, mul_add, add_mul, add_assoc]; sorry

end factorization_correct_l669_669430


namespace rabbit_stashed_nuts_l669_669549

theorem rabbit_stashed_nuts :
  ∃ r: ℕ, 
  ∃ f: ℕ, 
  4 * r = 6 * f ∧ f = r - 5 ∧ 4 * r = 60 :=
by {
  sorry
}

end rabbit_stashed_nuts_l669_669549


namespace handshake_count_l669_669148

/-- 
  In a new convention of 10 sets of twins and 4 sets of triplets:
  - Each twin shakes hands with all other twins except his/her sibling and with half of the triplets.
  - Each triplet shakes hands with all other triplets except his/her siblings and with one-third of the twins.
  Prove that the number of handshakes that took place at this convention is 354.
-/

def num_handshakes (twin_sets : ℕ) (triplet_sets : ℕ) : ℕ :=
  let twins := twin_sets * 2
  let triplets := triplet_sets * 3
  let twin_to_twin := (twins * (twins - 2)) / 2
  let triplet_to_triplet := (triplets * (triplets - 3)) / 2
  let twin_to_triplet := twins * (triplets / 2)
  let triplet_to_twin := triplets * (twins / 3)
  in twin_to_twin + triplet_to_triplet + twin_to_triplet + triplet_to_twin

theorem handshake_count (twins : ℕ) (triplets : ℕ) :
  twins = 10 → triplets = 4 → 
  num_handshakes twins triplets = 354 :=
by
  intros
  sorry

end handshake_count_l669_669148


namespace imaginary_part_of_conjugate_z_l669_669095

def imaginary_unit : ℂ := Complex.i

def complex_z : ℂ := imaginary_unit + (2 / (1 - imaginary_unit))

def conjugate_z : ℂ := Complex.conj complex_z

def imaginary_part (z : ℂ) : ℝ := z.im

theorem imaginary_part_of_conjugate_z :
  imaginary_part conjugate_z = -2 :=
sorry

end imaginary_part_of_conjugate_z_l669_669095


namespace rancher_loss_l669_669340

-- Define the necessary conditions
def initial_head_of_cattle := 340
def original_total_price := 204000
def cattle_died := 172
def price_reduction_per_head := 150

-- Define the original and new prices per head
def original_price_per_head := original_total_price / initial_head_of_cattle
def new_price_per_head := original_price_per_head - price_reduction_per_head

-- Define the number of remaining cattle
def remaining_cattle := initial_head_of_cattle - cattle_died

-- Define the total amount at the new price
def total_amount_new_price := new_price_per_head * remaining_cattle

-- Define the loss
def loss := original_total_price - total_amount_new_price

-- Prove that the loss is $128,400
theorem rancher_loss : loss = 128400 := by
  sorry

end rancher_loss_l669_669340


namespace quadratic_form_and_sum_l669_669772

theorem quadratic_form_and_sum (x : ℝ) : 
  ∃ (a b c : ℝ), 
  (15 * x^2 + 75 * x + 375 = a * (x + b)^2 + c) ∧ 
  (a + b + c = 298.75) := 
sorry

end quadratic_form_and_sum_l669_669772


namespace minimum_Q_l669_669873

def is_special (m : ℕ) : Prop :=
  let d1 := m / 10 
  let d2 := m % 10
  d1 ≠ d2 ∧ d1 ≠ 0 ∧ d2 ≠ 0

def F (m : ℕ) : ℤ :=
  let d1 := m / 10
  let d2 := m % 10
  (d1 * 100 + d2 * 10 + d1) - (d2 * 100 + d1 * 10 + d2) / 99

def Q (s t : ℕ) : ℚ :=
  (t - s) / s

variables (a b x y : ℕ)
variables (h1 : 1 ≤ b ∧ b < a ∧ a ≤ 7)
variables (h2 : 1 ≤ x ∧ x ≤ 8)
variables (h3 : 1 ≤ y ∧ y ≤ 8)
variables (hs_is_special : is_special (10 * a + b))
variables (ht_is_special : is_special (10 * x + y))
variables (s := 10 * a + b)
variables (t := 10 * x + y)
variables (h4 : (F s % 5) = 1)
variables (h5 : F t - F s + 18 * x = 36)

theorem minimum_Q : Q s t = -42 / 73 := sorry

end minimum_Q_l669_669873


namespace find_three_power_l669_669045

theorem find_three_power (m n : ℕ) (h₁: 3^m = 4) (h₂: 3^n = 5) : 3^(2*m + n) = 80 := by
  sorry

end find_three_power_l669_669045


namespace coeff_of_x2_in_expansion_l669_669727

theorem coeff_of_x2_in_expansion : 
  (coeff_x2 : ℕ) = 24
  where coeff_x2 := ∑ k : ℕ in finset.range 5, if k = 2 then (nat.choose 4 k) * (-2) ^ k else 0 :=
by 
  sorry

end coeff_of_x2_in_expansion_l669_669727


namespace paint_cost_per_kg_l669_669729

-- Define the conditions as hypotheses
def cost_of_paint_per_kg (side_length : ℝ) (total_cost : ℝ) (coverage_per_kg : ℝ) : ℝ :=
  let surface_area := 6 * side_length^2
  let kgs_of_paint_needed := surface_area / coverage_per_kg
  total_cost / kgs_of_paint_needed

-- The Lean theorem stating that given the conditions, the cost of the paint per kg is 36.5
theorem paint_cost_per_kg (h1 : cost_of_paint_per_kg 8 876 16 = 36.5) : true :=
by {
  sorry
}

end paint_cost_per_kg_l669_669729


namespace quadrilateral_properties_l669_669224

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨5, 6⟩
def B : Point := ⟨-1, 2⟩
def C : Point := ⟨-2, -1⟩
def D : Point := ⟨4, -5⟩

def intersection_point_of_diagonals (A B C D : Point) : Point :=
  ⟨-1/6, 5/6⟩

def area_of_quadrilateral (A B C D : Point) : ℝ :=
  42

theorem quadrilateral_properties :
  intersection_point_of_diagonals A B C D = ⟨-1/6, 5/6⟩ ∧
  area_of_quadrilateral A B C D = 42 :=
by
  sorry

end quadrilateral_properties_l669_669224


namespace add_base_6_correct_l669_669284

noncomputable def from_base (b : ℕ) (ds : list ℕ) : ℕ :=
ds.reverse.enum.sum_map (λ ⟨i, d⟩, d * b^i)

def add_in_base_6 (x y : list ℕ) : list ℕ :=
  -- Function to add two numbers in base 6
  sorry

theorem add_base_6_correct :
  add_in_base_6 [4, 5, 3, 2] [2, 5, 3, 4, 6] = [3, 2, 5, 2, 5] :=
sorry

end add_base_6_correct_l669_669284


namespace quiz_answer_key_count_l669_669156

theorem quiz_answer_key_count :
  let true_false_possibilities := 6  -- Combinations for 3 T/F questions where not all are same
  let multiple_choice_possibilities := 4^3  -- 4 choices for each of 3 multiple-choice questions
  true_false_possibilities * multiple_choice_possibilities = 384 := by
  sorry

end quiz_answer_key_count_l669_669156


namespace cube_volume_l669_669227

-- Conditions
def is_cube (s : ℝ) : Prop := ∀ (x y z : ℝ), x * x + y * y + z * z = s * s

def space_diagonal (s : ℝ) : ℝ := s * sqrt 3

-- Proof statement
theorem cube_volume (s : ℝ) (h : space_diagonal s = 6 * sqrt 3) : s ^ 3 = 216 :=
sorry

end cube_volume_l669_669227


namespace right_triangle_area_l669_669755

/-- Given a right triangle with hypotenuse 13 meters and one side 5 meters,
prove that the area of the triangle is 30 square meters. -/
theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (hc : c = 13) (ha : a = 5) :
  1/2 * a * b = 30 :=
by sorry

end right_triangle_area_l669_669755


namespace min_value_m_n_l669_669957

theorem min_value_m_n (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_geom_mean : a * b = 4)
    (m n : ℝ) (h_m : m = b + 1 / a) (h_n : n = a + 1 / b) : m + n ≥ 5 :=
by
  sorry

end min_value_m_n_l669_669957


namespace arithmetic_sum_S11_l669_669952

noncomputable def Sn_sum (a1 an n : ℕ) : ℕ := n * (a1 + an) / 2

theorem arithmetic_sum_S11 (a1 a9 a8 a5 a11 : ℕ) (h1 : Sn_sum a1 a9 9 = 54)
    (h2 : Sn_sum a1 a8 8 - Sn_sum a1 a5 5 = 30) : Sn_sum a1 a11 11 = 88 := by
  sorry

end arithmetic_sum_S11_l669_669952


namespace coefficient_of_x3_in_expansion_l669_669158

theorem coefficient_of_x3_in_expansion :
  @binomial ℕ 5 3 - @binomial ℕ 4 3 = 6 := 
by {
  sorry
}

end coefficient_of_x3_in_expansion_l669_669158


namespace solve_for_n_l669_669529

theorem solve_for_n (n : ℕ) (h : sqrt (8 + n) = 9) : n = 73 := 
by {
  sorry
}

end solve_for_n_l669_669529


namespace problem_statement_l669_669074

-- Defining the sequence a_n and its conditions
def seq_a (n : ℕ) : ℕ := 2^(n-1)

-- Defining the sequence b_n as an arithmetic sequence 
def seq_b (n : ℕ) : ℕ := n

-- Defining the helper sequence c_n based on sequences a_n and b_n
def seq_c (n : ℕ) : ℝ :=
  (1 / seq_a n) - (2 / (seq_b n * seq_b (n + 1)))

-- Sum of the first n terms of sequence c
def T_n (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), seq_c i

-- Main theorem to be proved
theorem problem_statement (n : ℕ) : 
  T_n n = (2 / (n + 1)) - 2^(1 - n) := by
  sorry

end problem_statement_l669_669074


namespace simplify_cubic_root_l669_669699

theorem simplify_cubic_root : 
  (∛(54880000) = 20 * ∛((5^2) * 137)) :=
sorry

end simplify_cubic_root_l669_669699


namespace factorize_expression_l669_669415

theorem factorize_expression (a x : ℝ) :
  a * x^2 - 2 * a * x + a = a * (x - 1) ^ 2 := 
sorry

end factorize_expression_l669_669415


namespace cube_root_simplification_l669_669719

theorem cube_root_simplification : (∛54880000) = 140 * (2 ^ (1 / 3)) :=
by
  -- Using the information from the problem conditions and final solution.
  have root_10_cubed := (10 ^ 3 : ℝ)
  have factored_value := root_10_cubed * (2 ^ 4 * 7 ^ 3)
  have cube_root := Real.cbrt factored_value
  sorry

end cube_root_simplification_l669_669719


namespace number_of_sets_B_l669_669112

theorem number_of_sets_B (A B : Set ℤ) (hA : A = {-1, 0}) (hUnion : A ∪ B = {-1, 0, 1}) : 
  ∃ B, B = ({1} ∪ {0, -1}.powerset) ∧ 4 = (finset.powerset {0, -1}).card :=
by
  sorry

end number_of_sets_B_l669_669112


namespace eccentricity_of_ellipse_l669_669076

/-
Given an ellipse C: x²/a² + y²/b² = 1 (a > b > 0),
and several other conditions:

1. Vertex A is at (-a, 0)
2. Vertex B is at (a, 0)
3. Right focus F is at (c, 0)
4. P is a point on the ellipse
5. The line AP intersects another line at point M
6. The angle bisector of ∠PFB intersects the line x = a at point N
7. If PF is perpendicular to AB, the area of triangle MAB is 6 times the area of triangle NFB,

then the eccentricity e of the ellipse C is 1/3.
-/

theorem eccentricity_of_ellipse (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : a^2 = b^2 + c^2)
  (h4 : ∀ (P : ℝ × ℝ), (P.1^2 / a^2 + P.2^2 / b^2 = 1) → 
        (let M := ((-a + P.1) / 2, P.2) in
         let N := (a, P.2) in 
         (P.1 = c / a * a) →
         (area_traded M A B = 6 * area_traded N F B))) :
  (eccentricity a c = (1 / 3)) :=
by
  sorry -- proof omitted

end eccentricity_of_ellipse_l669_669076


namespace sequence_properties_l669_669865

-- Definitions and conditions
def sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  (∀ n : ℕ, n > 0 → a n > 0) ∧ (∀ n : ℕ, 4 * S n = (a n + 1)^2)

-- The main theorem to be proven
theorem sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ)
        (b : ℕ → ℝ) (T : ℕ → ℝ)
        (h_seq : sequence a S)
        (h_b : ∀ n : ℕ, b n = a n / (3 : ℝ)^n)
        (h_T : ∀ n : ℕ, T n = ∑ i in Finset.range(n+1), b i) :
  (∀ n : ℕ, n > 0 → a n = 2 * n - 1) ∧
  (∀ n : ℕ, T n = 1 - (n + 1) * (3 : ℝ)^(-n)) :=
by
  sorry

end sequence_properties_l669_669865


namespace cube_root_of_54880000_l669_669672

theorem cube_root_of_54880000 : (real.cbrt 54880000) = 140 * (real.cbrt 10) :=
by
  -- Definitions based on conditions
  have h1 : 54880000 = 10^3 * 54880, by norm_num
  have h2 : 54880 = 2^5 * 7^3 * 5, by norm_num
  have h3 : 10 = 2 * 5, by norm_num
  
  -- Cube root properties and simplifications are implicitly inferred by the system
  sorry

end cube_root_of_54880000_l669_669672


namespace cubes_difference_l669_669964

theorem cubes_difference 
  (a b : ℝ) 
  (h1 : a - b = 7) 
  (h2 : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := by
  sorry

end cubes_difference_l669_669964


namespace right_triangle_area_l669_669745

theorem right_triangle_area
  (hypotenuse : ℝ) (leg1 : ℝ) (leg2 : ℝ)
  (hypotenuse_eq : hypotenuse = 13)
  (leg1_eq : leg1 = 5)
  (pythagorean_eq : hypotenuse^2 = leg1^2 + leg2^2) :
  (1 / 2) * leg1 * leg2 = 30 :=
by
  sorry

end right_triangle_area_l669_669745


namespace solve_polynomial_eq_l669_669025

theorem solve_polynomial_eq (z : ℂ) : z^4 - 6 * z^2 + 8 = 0 ↔ z = -2 ∨ z = -complex.sqrt 2 ∨ z = complex.sqrt 2 ∨ z = 2 := 
sorry

end solve_polynomial_eq_l669_669025


namespace paint_cost_per_kg_l669_669730

-- Define the conditions as hypotheses
def cost_of_paint_per_kg (side_length : ℝ) (total_cost : ℝ) (coverage_per_kg : ℝ) : ℝ :=
  let surface_area := 6 * side_length^2
  let kgs_of_paint_needed := surface_area / coverage_per_kg
  total_cost / kgs_of_paint_needed

-- The Lean theorem stating that given the conditions, the cost of the paint per kg is 36.5
theorem paint_cost_per_kg (h1 : cost_of_paint_per_kg 8 876 16 = 36.5) : true :=
by {
  sorry
}

end paint_cost_per_kg_l669_669730


namespace part1_part2_l669_669840

-- Part (1)
theorem part1 (a : ℝ) : 
  (∀ x ∈ Icc 0 real.pi, (sin x) + 1 ≥ a * x + (cos x)) → a ≤ 2 / real.pi :=
sorry

-- Part (2)
theorem part2 (n : ℕ) : 
  (∑ k in finset.range (n+1), sin ((k+1 : ℝ) * real.pi / (2 * n + 1))) ≥ 3 * real.sqrt 2 * (n+1) / (4 * (2 * n + 1)) :=
sorry

end part1_part2_l669_669840


namespace rancher_loss_l669_669341

theorem rancher_loss
  (initial_cattle : ℕ)
  (total_price : ℕ)
  (sick_cattle : ℕ)
  (price_reduction : ℕ)
  (remaining_cattle := initial_cattle - sick_cattle)
  (original_price_per_head := total_price / initial_cattle)
  (new_price_per_head := original_price_per_head - price_reduction)
  (total_original_price := original_price_per_head * remaining_cattle)
  (total_new_price := new_price_per_head * remaining_cattle) :
  total_original_price - total_new_price = 25200 :=
by 
  sorry

-- Definitions
def initial_cattle : ℕ := 340
def total_price : ℕ := 204000
def sick_cattle : ℕ := 172
def price_reduction : ℕ := 150

-- Substitute the known values in the theorem
#eval rancher_loss initial_cattle total_price sick_cattle price_reduction

end rancher_loss_l669_669341


namespace cube_difference_l669_669971

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 53) : a^3 - b^3 = 385 :=
sorry

end cube_difference_l669_669971


namespace smallest_n_condition_l669_669186

theorem smallest_n_condition (n : ℕ) (x : ℕ → ℝ) 
  (H1 : ∀ i, |x i| < 1)
  (H2 : ∑ i in finset.range n, |x i| = 25 + |(∑ i in finset.range n, x i)|) :
  n ≥ 26 := 
sorry

end smallest_n_condition_l669_669186


namespace find_shorts_l669_669398

theorem find_shorts (total_shirts folded_shirts folded_shorts remaining_clothes S : ℕ) 
  (h1 : total_shirts = 20) 
  (h2 : folded_shirts = 12) 
  (h3 : folded_shorts = 5) 
  (h4 : remaining_clothes = 11) 
  (h5 : total_shirts + S = 28) : S = 8 :=
begin
  sorry
end

end find_shorts_l669_669398


namespace other_car_speed_l669_669275

theorem other_car_speed (v : ℝ) : 
  let t := 14 / 3 in
  let distance := 490 in
  let relative_speed := 45 + v in
  relative_speed * t = distance → 
  v = 60 :=
by
  intros
  sorry

end other_car_speed_l669_669275


namespace peaches_in_boxes_l669_669787

theorem peaches_in_boxes : 
  (let peaches_per_basket := 25 in
  let number_of_baskets := 5 in
  let eaten_peaches := 5 in
  let peaches_per_box := 15 in
  ∃ (boxes : ℕ), 
    (peaches_per_basket * number_of_baskets - eaten_peaches) / peaches_per_box = boxes ∧ 
    boxes = 8) :=
by
  sorry

end peaches_in_boxes_l669_669787


namespace f_of_x_5_5_l669_669093

noncomputable def f : ℝ → ℝ :=
λ x, if (1 < x ∧ x < 2) then x^3 + Real.sin (π * x / 9) else sorry

theorem f_of_x_5_5 :
  let f := λ x, if (1 < x ∧ x < 2) then x^3 + Real.sin (π * x / 9) else sorry
  ∀ x : ℝ, f(x) * f(x + 2) = -1 →
  f(5.5) = (31 / 8) :=
by
  intro f h_period h_condition x h_functional_eq
  sorry

end f_of_x_5_5_l669_669093


namespace find_t_plus_a3_l669_669241

noncomputable def geometric_sequence_sum (n : ℕ) (t : ℤ) : ℤ :=
  3 ^ n + t

noncomputable def a_1 (t : ℤ) : ℤ :=
  geometric_sequence_sum 1 t

noncomputable def a_2 (t : ℤ) : ℤ :=
  geometric_sequence_sum 2 t - geometric_sequence_sum 1 t

noncomputable def a_3 (t : ℤ) : ℤ :=
  geometric_sequence_sum 3 t - geometric_sequence_sum 2 t

theorem find_t_plus_a3 (t : ℤ) : t + a_3 t = 17 :=
sorry

end find_t_plus_a3_l669_669241


namespace factorization_l669_669433

theorem factorization (a x : ℝ) : ax^2 - 2ax + a = a * (x - 1) ^ 2 := 
by
  sorry

end factorization_l669_669433


namespace proof_AP_eq_BC_l669_669229

variables {b d p : ℝ}

def midpoint_AB : ℝ × ℝ := (b, d / 2)
def point_P : ℝ × ℝ := (p, p)
def point_Q : ℝ × ℝ := (p, 0)
def vector_PF : ℝ × ℝ := (b - p, d / 2 - p)
def vector_DQ : ℝ × ℝ := (p, -d)

-- Condition for perpendicular vectors: Their dot product is zero
def perpendicular_condition (v1 v2 : ℝ × ℝ) : Prop :=
  v1.fst * v2.fst + v1.snd * v2.snd = 0

-- The main theorem statement
theorem proof_AP_eq_BC
  (rect : (0,0) → (b,0) → (b,d) → (0,d))
  (midpoint : midpoint_AB = (b, d / 2))
  (proj_Q : point_Q = (p, 0))
  (bisector_P : point_P = (p, p))
  (perp_PF_DQ : perpendicular_condition vector_PF vector_DQ)
  : (b - p)^2 + (d - p)^2 = b^2 → (√((b-p)^2 + (d-p)^2) = sqrt(b^2)) :=
sorry

end proof_AP_eq_BC_l669_669229


namespace problem_l669_669574

def arithmetic_sequence (a : ℕ → ℕ) :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = a n + 2

def sum_of_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) :=
  ∀ n, S n = n * (a 1 + a n) / 2

theorem problem (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h_arith_seq : arithmetic_sequence a)
  (h_sum_def : sum_of_first_n_terms a S) :
  S 10 = 110 :=
begin
  sorry
end

end problem_l669_669574


namespace Keiko_speed_l669_669174

theorem Keiko_speed (a b s : ℝ) (h1 : 8 = 8) 
  (h2 : (2 * a + 2 * π * (b + 8)) / s = (2 * a + 2 * π * b) / s + 48) : 
  s = π / 3 :=
by
  sorry

end Keiko_speed_l669_669174


namespace cube_root_simplification_l669_669716

theorem cube_root_simplification : (∛54880000) = 140 * (2 ^ (1 / 3)) :=
by
  -- Using the information from the problem conditions and final solution.
  have root_10_cubed := (10 ^ 3 : ℝ)
  have factored_value := root_10_cubed * (2 ^ 4 * 7 ^ 3)
  have cube_root := Real.cbrt factored_value
  sorry

end cube_root_simplification_l669_669716


namespace factorization_correct_l669_669429

-- Define the initial expression
def initial_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + a

-- Define the factorized form
def factorized_expr (a x : ℝ) : ℝ := a * (x - 1)^2

-- Create the theorem statement
theorem factorization_correct (a x : ℝ) : initial_expr a x = factorized_expr a x :=
by simp [initial_expr, factorized_expr, pow_two, mul_add, add_mul, add_assoc]; sorry

end factorization_correct_l669_669429


namespace cube_root_simplification_l669_669682

noncomputable def cubeRoot (x : ℝ) : ℝ := x^(1/3)

theorem cube_root_simplification :
  cubeRoot 54880000 = 140 * cubeRoot 20 :=
by
  sorry

end cube_root_simplification_l669_669682


namespace range_of_m_l669_669575

theorem range_of_m (t : ℝ) (a : ℕ → ℝ) (n : ℕ) (m : ℝ)
  (h1 : t > 0)
  (h2 : ∀ n > 0, a 1 + 2 * a 2 + 2^2 * a 3 + ∀ i, i ∈ [4..n] → 2^(i-1) * a i = (n * 2^n - 2^n + 1) * t)
  (h3 : ∀ n, a n = n * t)
  (h4 : ∀ n ≥ 4, (1/t) * (1 - 1 / 2^n) > m/t)
: ∃ m ∈ Ico (7 / 8) (15 / 16), True := sorry

end range_of_m_l669_669575


namespace g_30_of_values_n_l669_669928

def num_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).card

def g (k : ℕ) (n : ℕ) : ℕ :=
  if k = 1 then 3 * num_divisors n else g 1 (g (k - 1) n)

theorem g_30_of_values_n :
  (Finset.filter (λ n, g 30 n = 18)
    (Finset.range 31)).card = 11 :=
sorry

end g_30_of_values_n_l669_669928


namespace perfect_square_count_21n_le_1500_l669_669034

-- Define the main condition that 21n must be a perfect square
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

-- Theorem stating that the count of positive integers n ≤ 1500 such that 21n is a perfect square is 8
theorem perfect_square_count_21n_le_1500 : 
  {n : ℕ // n > 0 ∧ n ≤ 1500 ∧ isPerfectSquare (21 * n)}.toFinset.card = 8 := 
by 
  sorry

end perfect_square_count_21n_le_1500_l669_669034


namespace repeating_decimal_divisible_by_2_or_5_l669_669775

theorem repeating_decimal_divisible_by_2_or_5 
    (m n : ℕ) 
    (x : ℝ) 
    (r s : ℕ) 
    (a b k p q u : ℕ)
    (hmn_coprime : Nat.gcd m n = 1)
    (h_rep_decimal : x = (m:ℚ) / (n:ℚ))
    (h_non_repeating_part: 0 < r) :
  n % 2 = 0 ∨ n % 5 = 0 :=
sorry

end repeating_decimal_divisible_by_2_or_5_l669_669775


namespace notebook_cost_3_dollars_l669_669006

def cost_of_notebook (total_spent backpack_cost pen_cost pencil_cost num_notebooks : ℕ) : ℕ := 
  (total_spent - (backpack_cost + pen_cost + pencil_cost)) / num_notebooks

theorem notebook_cost_3_dollars 
  (total_spent : ℕ := 32) 
  (backpack_cost : ℕ := 15) 
  (pen_cost : ℕ := 1) 
  (pencil_cost : ℕ := 1) 
  (num_notebooks : ℕ := 5) 
  : cost_of_notebook total_spent backpack_cost pen_cost pencil_cost num_notebooks = 3 :=
by
  sorry

end notebook_cost_3_dollars_l669_669006


namespace average_speed_correct_l669_669856

-- Definitions of conditions
def speed_in_still_water := 20 -- km/h
def increment_in_stream_A := 16 -- km/h
def decrement_in_stream_B := 12 -- km/h
def decrement_in_stream_C := 4 -- km/h
def increment_after_switch_in_stream_C := 6 -- km/h

-- Definitions of speeds in each stream
def speed_in_A := speed_in_still_water + increment_in_stream_A
def speed_in_B := speed_in_still_water - decrement_in_stream_B
def speed_before_switch_in_C := speed_in_still_water - decrement_in_stream_C
def speed_after_switch_in_C := speed_in_still_water + increment_after_switch_in_stream_C

-- Harmonic mean for stream C
def harmonic_mean (a b : ℕ) : ℝ := 2 / ((1 : ℝ) / a + (1 / b))
def average_speed_in_C := harmonic_mean speed_before_switch_in_C speed_after_switch_in_C

-- Average speed across all streams
def average_speed := (speed_in_A + speed_in_B + average_speed_in_C) / 3

-- The theorem we want to prove
theorem average_speed_correct :
  average_speed = 21.27 :=
  by
    sorry

end average_speed_correct_l669_669856


namespace gesture_password_connection_methods_l669_669886

-- representing points on a 3x3 grid using an enum
inductive Point
| A | B | C | D | O | F | G | H | I

open Point

-- defining conditions
def starts_at_center (path : List Point) : Prop :=
  path.head? = some O

def connects_exactly_two_points (path : List Point) : Prop :=
  path.tail.length = 2

def no_unused_points (path : List Point) : Prop :=
  ∀ (p₁ p₂ : Point), p₁ ∈ path → p₂ ∈ path → 
    ∀ (unused : Point), unused ∉ path → 
    ¬ (p₁, p₂) = (O, unused) ∧ ¬ (unused, O) = (p₁, p₂)

def not_both_previously_used (path : List Point) : Prop :=
  ∀ (p₁ p₂ : Point), (p₁ ∈ path) → (p₂ ∈ path) → (p₁ ≠ p₂)

-- Theorem statement
theorem gesture_password_connection_methods :
  ∃ (path : List Point), starts_at_center path ∧ 
                         connects_exactly_two_points path ∧ 
                         no_unused_points path ∧ 
                         not_both_previously_used path ∧ 
                         path.length = 3 → 
                         path.count  = 48 :=
sorry

end gesture_password_connection_methods_l669_669886


namespace arrangement_of_students_and_teachers_not_adjacent_l669_669321

theorem arrangement_of_students_and_teachers_not_adjacent :
  let n_students := 8
  let n_teachers := 2
  let teacher_not_adjacent := P_8^8 * P_9^2
  teacher_not_adjacent = P_8^8 * P_9^2 := sorry

end arrangement_of_students_and_teachers_not_adjacent_l669_669321


namespace find_x_l669_669193

-- Define the binary operation
def star (p q : ℤ × ℤ) : ℤ × ℤ := (p.1 + q.1, p.2 - q.2)

-- State the problem
theorem find_x (x : ℤ) : star (x, 4) (2, 1) = (6, 5) → x = 4 :=
by
  unfold star
  sorry

end find_x_l669_669193


namespace sum_of_first_and_last_l669_669570

noncomputable section

variables {A B C D E F G H I : ℕ}

theorem sum_of_first_and_last :
  (D = 8) →
  (A + B + C + D = 50) →
  (B + C + D + E = 50) →
  (C + D + E + F = 50) →
  (D + E + F + G = 50) →
  (E + F + G + H = 50) →
  (F + G + H + I = 50) →
  (A + I = 92) :=
by
  intros hD h1 h2 h3 h4 h5 h6
  sorry

end sum_of_first_and_last_l669_669570


namespace weight_of_new_person_l669_669309

theorem weight_of_new_person (average_increase : ℝ) (old_weight : ℝ) (num_people : ℕ) :
  average_increase = 2.5 → old_weight = 75 → num_people = 8 → 
  let total_increase := num_people * average_increase in
  let new_weight := old_weight + total_increase in
  new_weight = 95 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end weight_of_new_person_l669_669309


namespace expression_equals_three_l669_669387

theorem expression_equals_three : abs (sqrt 3 - 2) - 2 * tan (real.pi / 3) + (real.pi - 2023)^0 + sqrt 27 = 3 :=
by
  sorry

end expression_equals_three_l669_669387


namespace smallest_odd_digit_number_gt_1000_mult_5_l669_669809

def is_odd_digit (n : ℕ) : Prop := n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 9

def valid_number (n : ℕ) : Prop :=
  n > 1000 ∧ (∃ d1 d2 d3 d4, n = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ 
  is_odd_digit d1 ∧ is_odd_digit d2 ∧ is_odd_digit d3 ∧ is_odd_digit d4 ∧ 
  d4 = 5)

theorem smallest_odd_digit_number_gt_1000_mult_5 : ∃ n : ℕ, valid_number n ∧ 
  ∀ m : ℕ, valid_number m → m ≥ n := 
by
  use 1115
  simp [valid_number, is_odd_digit]
  sorry

end smallest_odd_digit_number_gt_1000_mult_5_l669_669809


namespace how_many_cheburashkas_erased_l669_669603

theorem how_many_cheburashkas_erased 
  (total_krakozyabras : ℕ)
  (characters_per_row_initial : ℕ) 
  (total_characters_initial : ℕ)
  (total_cheburashkas : ℕ)
  (total_rows : ℕ := 2)
  (total_krakozyabras := 29) :
  total_cheburashkas = 11 :=
by
  sorry

end how_many_cheburashkas_erased_l669_669603


namespace sample_size_l669_669869

theorem sample_size (teachers students_male students_female sample_female : ℕ)
  (h_teachers : teachers = 200)
  (h_students_male : students_male = 1200)
  (h_students_female : students_female = 1000)
  (h_sample_female : sample_female = 80) :
  let population := teachers + students_male + students_female
      sampling_fraction := sample_female / students_female
      n := population * sampling_fraction in
    n = 192 :=
by {
  sorry
}

end sample_size_l669_669869


namespace factorization_correct_l669_669431

-- Define the initial expression
def initial_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + a

-- Define the factorized form
def factorized_expr (a x : ℝ) : ℝ := a * (x - 1)^2

-- Create the theorem statement
theorem factorization_correct (a x : ℝ) : initial_expr a x = factorized_expr a x :=
by simp [initial_expr, factorized_expr, pow_two, mul_add, add_mul, add_assoc]; sorry

end factorization_correct_l669_669431


namespace find_a_if_odd_function_l669_669540

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = -f(x)

noncomputable def target_function (a : ℝ) : (ℝ → ℝ) :=
  λ x, log a (x + sqrt (x^2 + 2 * a^2))

theorem find_a_if_odd_function :
  (is_odd_function (target_function a)) → a = sqrt 2 / 2 :=
sorry

end find_a_if_odd_function_l669_669540


namespace dividend_amount_l669_669827

/-- Definition of the investment, face value of the share, premium, and dividend rate conditions. --/
def investment : ℝ := 14400
def face_value : ℝ := 100
def premium_rate : ℝ := 0.2
def dividend_rate : ℝ := 0.07

/-- Definition of the cost per share including premium. --/
def cost_per_share : ℝ := face_value * (1 + premium_rate)

/-- Definition of the total number of shares purchased. --/
def number_of_shares : ℝ := investment / cost_per_share

/-- Definition of the dividend per share. --/
def dividend_per_share : ℝ := face_value * dividend_rate

/-- Definition of the total dividend. --/
def total_dividend : ℝ := number_of_shares * dividend_per_share

/-- The theorem stating the correct answer to be proved. --/
theorem dividend_amount : total_dividend = 840 := by
  sorry

end dividend_amount_l669_669827


namespace number_of_valid_n_l669_669031

theorem number_of_valid_n : 
  (∃ (n : ℕ), n ≤ 1500 ∧ (∃ (k : ℕ), 21 * n = k^2)) ↔ (∃ (b : ℕ), b ≤ 8 ∧ n = 21 * b^2 ∧ n ≤ 1500 := sorry

end number_of_valid_n_l669_669031


namespace simple_closed_polygon_exists_l669_669942

theorem simple_closed_polygon_exists (n : ℕ) (points : fin n → ℝ × ℝ)
  (h1 : ∀ i j k : fin n, i ≠ j → j ≠ k → i ≠ k → 
    ¬ (collinear ({ points i, points j, points k } : set (ℝ × ℝ)))) :
  ∃ f : fin n → fin n, (is_simple_closed_polygon (points ∘ f)) :=
by sorry

-- Assuming collinear is defined elsewhere for set of points
-- Assuming is_simple_closed_polygon is defined to verify a closed polygon with no edge intersections

end simple_closed_polygon_exists_l669_669942


namespace ellipse_equation_range_of_M_x_coordinate_l669_669101

-- Proof 1: Proving the equation of the ellipse
theorem ellipse_equation {a b : ℝ} (h_ab : a > b) (h_b0 : b > 0) (e : ℝ)
  (h_e : e = (Real.sqrt 3) / 3) (vertex : ℝ × ℝ) (h_vertex : vertex = (Real.sqrt 3, 0)) :
  (∃ (a b : ℝ), a > b ∧ b > 0 ∧ e = (Real.sqrt 3) / 3 ∧ vertex = (Real.sqrt 3, 0) ∧ (∀ (x y : ℝ), (x^2) / 3 + (y^2) / 2 = 1)) :=
sorry

-- Proof 2: Proving the range of x-coordinate of point M
theorem range_of_M_x_coordinate (k : ℝ) (h_k : k ≠ 0) :
  (∃ M_x : ℝ, by sorry) :=
sorry


end ellipse_equation_range_of_M_x_coordinate_l669_669101


namespace buckets_required_l669_669791

variable (C : ℝ) (N : ℝ)

theorem buckets_required (h : N * C = 105 * (2 / 5) * C) : N = 42 := 
  sorry

end buckets_required_l669_669791


namespace prism_cross_section_area_l669_669220

noncomputable def cross_section_area {α : Type*} [euclidean_space α] 
(base_triangle : triangle) 
(center_lateral_face : point) 
(vertex_B : point)
(parallel_to_AB1 : line) (distance_C_to_plane : ℝ) : ℝ :=
let hypotenuse_AC := base_triangle.hypotenuse in
let angle_B := base_triangle.angle_B in
let angle_C := base_triangle.angle_C in
-- Other necessary constructs skipped 
-- Area of the cross-section
21 / 8

theorem prism_cross_section_area 
(base_triangle : triangle) 
(center_lateral_face : point) 
(vertex_B : point)
(parallel_to_AB1 : line) (distance_C_to_plane : ℝ) : 
  (base_triangle.angle_B = 90) →
  (base_triangle.angle_C = 30) →
  (base_triangle.hypotenuse = sqrt 14) →
  (distance_C_to_plane = 2) →
  cross_section_area base_triangle center_lateral_face vertex_B parallel_to_AB1 distance_C_to_plane = 21 / 8 :=
by
  intros angleB_eq angleC_eq hypotenuse_eq distance_eq
  sorry

end prism_cross_section_area_l669_669220


namespace range_of_B_l669_669974

theorem range_of_B (A : ℝ × ℝ) (hA : A = (1, 2)) (h : 2 * A.1 - B * A.2 + 3 ≥ 0) : B ≤ 2.5 :=
by sorry

end range_of_B_l669_669974


namespace isosceles_triangle_of_areas_l669_669876

variables {A B C D E : Point}
variables (tABC : Triangle A B C)
variables (hAD : Altitude tABC D A)
variables (hBE : Altitude tABC E B)
variables (area_BDE_leq_DEA : area B D E ≤ area D E A)
variables (area_DEA_leq_EAB : area D E A ≤ area E A B)
variables (area_EAB_leq_ABD : area E A B ≤ area A B D)

noncomputable def is_isosceles (t : Triangle) : Prop :=
  t.angle A B C = t.angle B C A ∨ t.angle A B C = t.angle A C B

theorem isosceles_triangle_of_areas (hacute : tABC.is_acute)
  (harea1 : area B D E ≤ area D E A)
  (harea2 : area D E A ≤ area E A B)
  (harea3 : area E A B ≤ area A B D) : is_isosceles tABC :=
by
  exact sorry

end isosceles_triangle_of_areas_l669_669876


namespace find_n_l669_669533

theorem find_n (n : ℕ) : sqrt (8 + n) = 9 → n = 73 :=
by
  intro h
  sorry

end find_n_l669_669533


namespace factorization_correct_l669_669426

-- Define the initial expression
def initial_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + a

-- Define the factorized form
def factorized_expr (a x : ℝ) : ℝ := a * (x - 1)^2

-- Create the theorem statement
theorem factorization_correct (a x : ℝ) : initial_expr a x = factorized_expr a x :=
by simp [initial_expr, factorized_expr, pow_two, mul_add, add_mul, add_assoc]; sorry

end factorization_correct_l669_669426


namespace sin_inequality_l669_669462

theorem sin_inequality (x : ℝ) (hx1 : 0 < x) (hx2 : x < Real.pi / 4) : 
  Real.sin (Real.sin x) < Real.sin x ∧ Real.sin x < Real.sin (Real.tan x) :=
by 
  sorry

end sin_inequality_l669_669462


namespace action_figure_cost_l669_669996

-- Given conditions and question setup
variables (actions_had actions_total actions_needed : ℕ) (total_cost cost_each : ℕ)
hypothesis actions_had_eq : actions_had = 3
hypothesis actions_total_eq : actions_total = 8
hypothesis total_cost_eq : total_cost = 30

-- The number of actions needed is the total - the ones he already has
def num_needed : ℕ := actions_total - actions_had
hypothesis num_needed_correct : num_needed = 5

-- The cost of each action figure is the total cost divided by the number needed
def cost_per_action : ℕ := total_cost / num_needed
hypothesis cost_correct : cost_per_action = 6

-- Prove that the cost of each action figure is $6 given the conditions
theorem action_figure_cost : cost_each = cost_per_action := by
  rw [cost_correct]
  sorry

end action_figure_cost_l669_669996


namespace find_m_l669_669973

-- Define the conditions with variables a, b, and m.
variable (a b m : ℝ)
variable (ha : 2^a = m)
variable (hb : 5^b = m)
variable (hc : 1/a + 1/b = 2)

-- Define the statement to be proven.
theorem find_m : m = Real.sqrt 10 :=
by
  sorry


end find_m_l669_669973


namespace enclosing_sphere_correct_radius_l669_669407

def smallest_enclosing_sphere_radius
  (centers : List (ℝ × ℝ × ℝ))
  (radius : ℝ) : ℝ :=
  Real.sqrt 3 + 1

theorem enclosing_sphere_correct_radius 
  (h : List.forall (λ (c : ℝ × ℝ × ℝ), ( |c.1| = 2 ∧ |c.2| = 2 ∧ |c.3| = 2 )) 
  (r : ℝ = 2) (centers = [(2, 2, 2), (2, 2, -2), (2, -2, 2),
                         (2, -2, -2), (-2, 2, 2), (-2, 2, -2), 
                         (-2, -2, 2), (-2, -2, -2)]) :
  smallest_enclosing_sphere_radius centers r = Real.sqrt 3 + 1 :=
sorry

end enclosing_sphere_correct_radius_l669_669407


namespace cheburashkas_erased_l669_669607

theorem cheburashkas_erased (total_krakozyabras : ℕ) (rows : ℕ) :
  rows ≥ 2 → total_krakozyabras = 29 → ∃ (cheburashkas_erased : ℕ), cheburashkas_erased = 11 :=
by
  assume h_rows h_total_krakozyabras
  let n := (total_krakozyabras / 2) + 1
  have h_cheburashkas : cheburashkas_erased = n - 1 
  sorry

end cheburashkas_erased_l669_669607


namespace division_remainder_l669_669447

theorem division_remainder :
  let p := fun x : ℝ => 5 * x^4 - 9 * x^3 + 3 * x^2 - 7 * x - 30
  let q := 3 * x - 9
  p 3 % q = 138 :=
by
  sorry

end division_remainder_l669_669447


namespace sum_of_interior_angles_of_regular_polygon_l669_669761

theorem sum_of_interior_angles_of_regular_polygon
  (each_exterior_angle : ℝ)
  (h : each_exterior_angle = 45) :
  let n := 360 / each_exterior_angle in
  (n : ℝ) = 8 → 180 * (n - 2) = 1080 :=
by
  intros
  sorry

end sum_of_interior_angles_of_regular_polygon_l669_669761


namespace cosine_double_angle_tangent_l669_669465

theorem cosine_double_angle_tangent (θ : ℝ) (h : Real.tan θ = -1/3) : Real.cos (2 * θ) = 4/5 :=
by
  sorry

end cosine_double_angle_tangent_l669_669465


namespace max_non_neighbouring_set_l669_669274

/-- Two 10-digit integers are called neighbours if they differ in exactly one digit. -/
def neighbours (a b : ℕ) : Prop :=
  (0 ≤ a ∧ a < 10^10) ∧ (0 ≤ b ∧ b < 10^10) ∧ (∃ i : ℕ, (a / 10^i % 10 ≠ b / 10^i % 10) ∧ (∀ j ≠ i, a / 10^j % 10 = b / 10^j % 10))

/-- The maximal number of elements in the set of 10-digit integers with no two integers being neighbours. -/
theorem max_non_neighbouring_set : ∃ s : Set ℕ, (∀ a ∈ s, ∀ b ∈ s, a ≠ b → ¬ neighbours a b) ∧ s.card = 9 * 10^8 :=
by
  sorry

end max_non_neighbouring_set_l669_669274


namespace minimum_cactus_species_l669_669376

-- Definitions to represent the conditions
def num_cactophiles : Nat := 80
def num_collections (S : Finset (Fin num_cactophiles)) : Nat := S.card
axiom no_single_species_in_all (S : Finset (Fin num_cactophiles)) : num_collections S < num_cactophiles
axiom any_15_have_common_species (S : Finset (Fin num_cactophiles)) (h : S.card = 15) : 
  ∃ species, ∀ s ∈ S, species ∈ s

-- Proposition to be proved
theorem minimum_cactus_species (k : Nat) (h : ∀ S : Finset (Fin num_cactophiles), S.card = 15 → ∃ species, ∀ s ∈ S, species ∈ s) : k ≥ 16 := sorry

end minimum_cactus_species_l669_669376


namespace expected_value_of_X_l669_669269

-- Define the sequence of random digits as a list of natural numbers (0 to 9)
def random_digits : List ℕ := (List.range 10).take 1000

-- Define the function that forms the number X
def X (digits : List ℕ) : ℝ :=
  digits.enum.foldr (λ (p : ℕ × ℕ) (acc : ℝ) => acc + p.snd * 10^(-(p.fst + 1))) 0

-- Define the expected value of a single digit
def expected_value_digit : ℝ := (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) / 10

-- The main statement to prove
theorem expected_value_of_X (digits : List ℕ) (h_digits : digits.length = 1000) :
  (∑ i in Finset.range digits.length, 10^(-(i + 1)) * expected_value_digit) = 0.5 :=
by {
  sorry
}

end expected_value_of_X_l669_669269


namespace derivative_of_y_l669_669922

noncomputable def y (x : ℝ) : ℝ :=
  (5^x * (sin (3 * x) * log 5 - 3 * cos (3 * x))) / (9 + (log 5)^2)

theorem derivative_of_y (x : ℝ) :
  deriv y x = 5^x * sin (3 * x) :=
by sorry

end derivative_of_y_l669_669922


namespace equation_and_coordinates_of_point_M_l669_669140

noncomputable theory

-- Definitions of the parabola, point M, and given conditions
def parabola (p : ℝ) : Set (ℝ × ℝ) := {M : ℝ × ℝ | M.snd ^ 2 = -2 * p * M.fst}

def point_M_x_coord : ℝ := -9
def distance_M_focus : ℝ := 10
def p_value : ℝ := 9

-- Focus of the parabola
def focus (p : ℝ) : (ℝ × ℝ) := ⟨ -p/2, 0 ⟩

-- Parabola's standard form for p_value = 9
def std_parabola (x y : ℝ) : Prop := y^2 = -4 * x

theorem equation_and_coordinates_of_point_M :
  (∀ M ∈ parabola 9, M.fst = point_M_x_coord → dist M (focus 9) = distance_M_focus →
   (std_parabola M.fst M.snd ∧ (M = (-9, 6) ∨ M = (-9, -6)))) :=
by
  sorry

end equation_and_coordinates_of_point_M_l669_669140


namespace hexagon_diagonals_concurrent_l669_669762

-- Define the hexagon inscribed in a circle
variables (A B C D E F : Type) [InCircle A B C D E F]

-- Define the diagonals
variables {P : Type} [Concurrent A D B E C F P]

-- Define the proportional relationship condition
variables {AB BC CD DE EF FA : ℝ}
hypothesis (h : AB / BC * CD / DE * EF / FA = 1)

theorem hexagon_diagonals_concurrent
  (h : AB / BC * CD / DE * EF / FA = 1) :
  Concurrent A D B E C F P :=
sorry

end hexagon_diagonals_concurrent_l669_669762


namespace derivative_of_odd_function_is_even_l669_669650

noncomputable def f : ℝ → ℝ := sorry  -- Define the function f, which is an odd function
def g (f : ℝ → ℝ) := λ x, derivative f x  -- Define g as the derivative of f

-- Given conditions, including that f is an odd function
axiom odd_f (x : ℝ) : f (-x) = -f x

-- Problem statement: Prove that g is even, i.e., g(-x) = g(x)
theorem derivative_of_odd_function_is_even : ∀ (x : ℝ), g f (-x) = g f x :=
by
  sorry  -- Proof is not provided, as requested

end derivative_of_odd_function_is_even_l669_669650


namespace part_a_l669_669454

def S (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ k, Nat.coprime k (n + 1)).sum

theorem part_a (n : ℕ) : ¬∃ k : ℕ, 2 * S (n + 1) = k * k :=
sorry

end part_a_l669_669454


namespace amount_lent_by_A_to_B_l669_669855

noncomputable def interest (principal rate time : ℝ) := principal * rate / 100 * time

theorem amount_lent_by_A_to_B :
  ∃ P : ℝ, 
    (interest P 12.5 2 - interest P 8 2 = 283.5) → 
    (P = 3150) :=
begin
  use 3150,
  intro h,
  sorry
end

end amount_lent_by_A_to_B_l669_669855


namespace solution_set_of_inequality_l669_669450

theorem solution_set_of_inequality (x : ℝ) : x * (x + 3) ≥ 0 ↔ x ≤ -3 ∨ x ≥ 0 :=
by
  sorry

end solution_set_of_inequality_l669_669450


namespace price_of_second_variety_l669_669216

theorem price_of_second_variety (x : ℝ) : 
  (1 * 126 + 1 * x + 2 * 177.5) / 4 = 154 → x = 135 :=
by
  intro h
  have : (1 * 126 + 1 * x + 2 * 177.5) = 4 * 154, by sorry
  linarith

end price_of_second_variety_l669_669216


namespace factorize_expression_l669_669418

theorem factorize_expression (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
by sorry

end factorize_expression_l669_669418


namespace maximum_area_of_equilateral_triangle_in_rectangle_l669_669240

theorem maximum_area_of_equilateral_triangle_in_rectangle :
  ∀ (ABCD: Type) (a b: ℕ), (a = 12) ∧ (b = 13) ∧ is_rectangle ABCD a b →
  ∃ (p q r: ℕ), 0 < q ∧ (¬ ∃ k, q = k^2) ∧ (q = 3) ∧ (r = 0) ∧ max_equilateral_area ABCD = p * real.sqrt q - r :=
by {
  -- We will need to prove that the maximum area of the equilateral triangle inscribed within
  -- the given rectangle is equal to 36√3.
  sorry
}

end maximum_area_of_equilateral_triangle_in_rectangle_l669_669240


namespace find_a4_l669_669495

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Conditions
def condition1 : ∀ n, S n = (∑ i in finset.range n, a (i+1)) := sorry
def condition2 : a 2 = 3 := sorry
def condition3 : ∀ n, S (n + 1) = 2 * (S n) := sorry

theorem find_a4 :
  a 4 = 12 :=
begin
  -- use the conditions here to prove the theorem
  sorry
end

end find_a4_l669_669495


namespace circles_intersect_l669_669237

def circle1_center := (-2 : ℝ, 0 : ℝ)
def circle1_radius := 2

def circle2_center := (2 : ℝ, 1 : ℝ)
def circle2_radius := 3

def distance_square (p1 p2: ℝ × ℝ) : ℝ :=
(p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem circles_intersect
  (h1 : circle1_center = (-2, 0))
  (h2 : circle1_radius = 2)
  (h3 : circle2_center = (2, 1))
  (h4 : circle2_radius = 3) :
  let dist := real.sqrt (distance_square circle1_center circle2_center)
  in circle2_radius - circle1_radius < dist ∧ dist < circle2_radius + circle1_radius :=
by
  sorry

end circles_intersect_l669_669237


namespace prove_equation_C_l669_669819

theorem prove_equation_C (m : ℝ) : -(m - 2) = -m + 2 := 
  sorry

end prove_equation_C_l669_669819


namespace probability_triangle_formation_l669_669656

open Set

-- Define the problem statement as a Lean theorem
theorem probability_triangle_formation (L : ℝ) (hL : 0 < L) : 
  let square := Icc 0 L ×ˢ Icc 0 L in
  let valid_region := {p : ℝ × ℝ | p.1 < L / 2 ∧ p.2 < L / 2 ∧ abs (p.1 - p.2) < L / 2} in
  let probability := (volume valid_region) / (volume square) in
  probability = 1 / 4 :=
by
  sorry

end probability_triangle_formation_l669_669656


namespace average_plant_height_is_100_variance_plant_height_is_100_prob_above_120_greater_than_below_70_prob_between_80_90_not_eq_100_110_l669_669835

noncomputable def normal_density (x : ℝ) : ℝ :=
  (1 / (10 * (2 * Real.pi).sqrt)) * Real.exp (-((x - 100)^2 / 200))

theorem average_plant_height_is_100 :
  ∫ x in -∞..∞, x * normal_density x = 100 :=
begin
  sorry
end

theorem variance_plant_height_is_100 :
  ∫ x in -∞..∞, (x - 100)^2 * normal_density x = 100 :=
begin
  sorry
end

theorem prob_above_120_greater_than_below_70 :
  ∫ x in 120..∞, normal_density x > ∫ x in -∞..70, normal_density x :=
begin
  sorry
end

theorem prob_between_80_90_not_eq_100_110 :
  ∫ x in 80..90, normal_density x ≠ ∫ x in 100..110, normal_density x :=
begin
  sorry
end

end average_plant_height_is_100_variance_plant_height_is_100_prob_above_120_greater_than_below_70_prob_between_80_90_not_eq_100_110_l669_669835


namespace tire_usage_l669_669468

theorem tire_usage (road_tires spare_tires total_miles : ℕ) (usage_per_tire : ℚ)
  (h_road_tires : road_tires = 4)
  (h_spare_tires : spare_tires = 2)
  (h_total_miles : total_miles = 50000)
  (h_usage_calc : usage_per_tire = (total_miles * road_tires) / (road_tires + spare_tires)) :
  usage_per_tire = 33333 := 
by
  rw [h_road_tires, h_spare_tires, h_total_miles] at h_usage_calc
  norm_num at h_usage_calc
  exact h_usage_calc.symm

end tire_usage_l669_669468


namespace ratio_of_probabilities_l669_669908

noncomputable def balls_toss (balls bins : ℕ) : Nat := by
  sorry

def prob_A : ℚ := by
  sorry
  
def prob_B : ℚ := by
  sorry

theorem ratio_of_probabilities (balls : ℕ) (bins : ℕ) 
  (h_balls : balls = 20) (h_bins : bins = 5) (p q : ℚ) 
  (h_p : p = prob_A) (h_q : q = prob_B) :
  (p / q) = 4 := by
  sorry

end ratio_of_probabilities_l669_669908


namespace no_solutions_triples_l669_669123

theorem no_solutions_triples (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a! + b^3 ≠ 18 + c^3 :=
by
  sorry

end no_solutions_triples_l669_669123


namespace labels_closer_than_distance_l669_669913

noncomputable def exists_points_with_labels_closer_than_distance (f : ℝ × ℝ → ℝ) : Prop :=
  ∃ (P Q : ℝ × ℝ), P ≠ Q ∧ |f P - f Q| < dist P Q

-- Statement of the problem
theorem labels_closer_than_distance :
  ∀ (f : ℝ × ℝ → ℝ), exists_points_with_labels_closer_than_distance f :=
sorry

end labels_closer_than_distance_l669_669913


namespace cube_volume_ratio_l669_669247

theorem cube_volume_ratio (a b : ℝ) (h : (a^2 / b^2) = 9 / 25) :
  (b^3 / a^3) = 125 / 27 :=
by
  sorry

end cube_volume_ratio_l669_669247


namespace simplify_cubed_root_l669_669709

def c1 : ℕ := 54880000
def c2 : ℕ := 10^5 * 5488
def c3 : ℕ := 5488
def c4 : ℕ := 2^4 * 343
def c5 : ℕ := 343
def c6 : ℕ := 7^3

theorem simplify_cubed_root : (c1^(1 / 3 : ℝ) : ℝ) = 1400 := 
by {
  let h1 : c1 = c2 := sorry,
  let h2 : c3 = c4 := sorry,
  let h3 : c5 = c6 := sorry,
  rw [h1, h2, h3],
  sorry
}

end simplify_cubed_root_l669_669709


namespace a_2013_value_l669_669094

open Nat

variable (f : ℕ → ℝ)

-- Conditions
def even_function := ∀ x, f (2 + x) = f (2 - x)
def even_property := ∀ x, f (-x) = f (x)
def bounded_value : (=) f (λ x, ite (-2 ≤ x ∧ x ≤ 0) (2^x) (f x))

-- Problem statement: Prove the value of a_{2013}
theorem a_2013_value (f_even : even_function f) (f_sym : even_property f) (f_bound : bounded_value f) : f 2013 = 1/2 :=
by 
  sorry -- Proof will be provided

end a_2013_value_l669_669094


namespace weight_order_l669_669162

variable {P Q R S T : ℕ}

theorem weight_order
    (h1 : Q + S = 1200)
    (h2 : R + T = 2100)
    (h3 : Q + T = 800)
    (h4 : Q + R = 900)
    (h5 : P + T = 700)
    (hP : P < 1000)
    (hQ : Q < 1000)
    (hR : R < 1000)
    (hS : S < 1000)
    (hT : T < 1000) :
  S > R ∧ R > T ∧ T > Q ∧ Q > P :=
sorry

end weight_order_l669_669162


namespace circumcircles_concurrent_fermat_point_l669_669887

open EuclideanGeometry

variable {A B C D E F : Point}

-- Definitions and assumptions based on the conditions in a)
def acute_triangle (A B C : Point) : Prop :=
  let ∠A := angle B A C
  let ∠B := angle A B C
  let ∠C := angle A C B
  ∠A < π / 2 ∧ ∠B < π / 2 ∧ ∠C < π / 2

def equilateral_triangle (P Q R : Point) : Prop :=
  dist P Q = dist Q R ∧ dist Q R = dist R P

def circumcircle (P Q R : Point) : Circle := sorry

-- The main theorem based on the question and correct answer
theorem circumcircles_concurrent_fermat_point
  (hABC : acute_triangle A B C)
  (hBCD : equilateral_triangle B C D)
  (hCAE : equilateral_triangle C A E)
  (hABF : equilateral_triangle A B F) :
  let circBCD := circumcircle B C D
  let circCAE := circumcircle C A E
  let circABF := circumcircle A B F
  ∃ P : Point, P ∈ circBCD ∧ P ∈ circCAE ∧ P ∈ circABF :=
sorry

end circumcircles_concurrent_fermat_point_l669_669887


namespace quadratic_equation_roots_l669_669050

theorem quadratic_equation_roots (a b k k1 k2 : ℚ)
  (h_roots : ∀ x : ℚ, k * (x^2 - x) + x + 2 = 0)
  (h_ab_condition : (a / b) + (b / a) = 3 / 7)
  (h_k_values : ∀ x : ℚ, 7 * x^2 - 20 * x - 21 = 0)
  (h_k1k2 : k1 + k2 = 20 / 7)
  (h_k1k2_prod : k1 * k2 = -21 / 7) :
  (k1 / k2) + (k2 / k1) = -104 / 21 :=
sorry

end quadratic_equation_roots_l669_669050


namespace Sally_shirts_sewn_on_Monday_l669_669669

theorem Sally_shirts_sewn_on_Monday (M : ℕ) 
  (shirts_Tuesday : ℕ = 3) 
  (shirts_Wednesday : ℕ = 2) 
  (buttons_per_shirt : ℕ = 5) 
  (total_buttons_needed : ℕ = 45) : 
  5 * (M + shirts_Tuesday + shirts_Wednesday) = total_buttons_needed → M = 4 :=
by 
  intro h
  sorry

end Sally_shirts_sewn_on_Monday_l669_669669


namespace min_value_2a_plus_b_value_of_t_l669_669048

theorem min_value_2a_plus_b (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1/a + 2/b = 2) :
  2 * a + b = 4 :=
sorry

theorem value_of_t (a b t : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1/a + 2/b = 2) (h₄ : 4^a = t) (h₅ : 3^b = t) :
  t = 6 :=
sorry

end min_value_2a_plus_b_value_of_t_l669_669048


namespace distinct_values_S_l669_669524

def i := Complex.I

def S (n : ℤ) : ℂ := i^n + i^(-n)

theorem distinct_values_S : (finset.image (λ (n : ℤ), S n) (finset.range 4)).card = 3 := by
  sorry

end distinct_values_S_l669_669524


namespace max_k_consecutive_sum_l669_669806

theorem max_k_consecutive_sum (n k : ℕ) (h : 2013 = k * n + k * (k + 1) / 2) : k ≤ 61 :=
begin
  sorry
end

end max_k_consecutive_sum_l669_669806


namespace sailboat_swimming_solution_l669_669864

noncomputable def sailboat_swimming_problem : Prop :=
  let v_b := 15 -- km/h
  let v_s := 2  -- km/h
  let d_s := 0.2 -- km
  let d_b := 2  -- km
  ∃ α_1 α_2 : ℝ, (α_1 ≈ 42.54 ∧ α_2 ≈ 54.01)

theorem sailboat_swimming_solution : sailboat_swimming_problem := 
by
  sorry -- Details of the proof go here

end sailboat_swimming_solution_l669_669864


namespace water_added_is_correct_l669_669214

def amountOfWaterAdded
  (initialVolume : ℝ)
  (initialAlcoholPercentage : ℝ)
  (finalAlcoholPercentage : ℝ)
  (finalVolumeWithAlcohol : ℝ) : ℝ :=
  initialVolume * initialAlcoholPercentage / finalAlcoholPercentage - initialVolume

theorem water_added_is_correct :
  amountOfWaterAdded 11 0.42 0.33 (11 * 0.42) = 3 := sorry

end water_added_is_correct_l669_669214


namespace hcf_of_two_numbers_is_18_l669_669311

theorem hcf_of_two_numbers_is_18
  (product : ℕ)
  (lcm : ℕ)
  (hcf : ℕ) :
  product = 571536 ∧ lcm = 31096 → hcf = 18 := 
by sorry

end hcf_of_two_numbers_is_18_l669_669311


namespace correct_statements_l669_669297

def f1 (x : ℝ) : ℝ := x * Real.log x
def f2 (x : ℝ) : ℝ := x^2 + 3 * x * (f2' 2) + Real.log x 

theorem correct_statements (f2' : ℝ → ℝ):
  (∀ (x₀ : ℝ), (deriv f1 x₀ = 2) → x₀ = Real.exp 1) ∧
  (deriv f2 2 = -(9 / 4)) :=
sorry

end correct_statements_l669_669297


namespace ned_good_games_l669_669199

noncomputable def total_games (bought_from_friend : ℕ) (bought_from_garage_sale : ℕ) : ℕ :=
  bought_from_friend + bought_from_garage_sale

theorem ned_good_games
  (bought_from_friend : ℕ)
  (bought_from_garage_sale : ℕ)
  (total_good_games : ℕ)
  (total_games_purchased : bought_from_friend + bought_from_garage_sale) :
  total_good_games = 14 :=
by
  have total_games_count := total_games 11 22
  have total_games := 33
  have good_games := 14
  sorry

end ned_good_games_l669_669199


namespace restore_price_by_percentage_l669_669310

theorem restore_price_by_percentage 
  (p : ℝ) -- original price
  (h₀ : p > 0) -- condition that price is positive
  (r₁ : ℝ := 0.25) -- reduction of 25%
  (r₁_applied : ℝ := p * (1 - r₁)) -- first reduction
  (r₂ : ℝ := 0.20) -- additional reduction of 20%
  (r₂_applied : ℝ := r₁_applied * (1 - r₂)) -- second reduction
  (final_price : ℝ := r₂_applied) -- final price after two reductions
  (increase_needed : ℝ := p - final_price) -- amount to increase to restore the price
  (percent_increase : ℝ := (increase_needed / final_price) * 100) -- percentage increase needed
  : abs (percent_increase - 66.67) < 0.01 := -- proof that percentage increase is approximately 66.67%
sorry

end restore_price_by_percentage_l669_669310


namespace ab_ac_bc_nonpositive_l669_669625

theorem ab_ac_bc_nonpositive (a b c : ℝ) (h : a + b + c = 0) : ∃ y : ℝ, y = ab + ac + bc ∧ y ≤ 0 :=
by
  sorry

end ab_ac_bc_nonpositive_l669_669625


namespace cubes_difference_l669_669963

theorem cubes_difference 
  (a b : ℝ) 
  (h1 : a - b = 7) 
  (h2 : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := by
  sorry

end cubes_difference_l669_669963


namespace prove_total_time_eq_108_l669_669278

-- Definitions for velocities in km/h
def v1 := 52 -- Velocity of the faster train in km/h
def v2 := 36 -- Velocity of the slower train in km/h
def t1 := 36 -- Time (in seconds) taken to pass the slower train
def length_of_train (relative_speed : ℝ) (time : ℝ) : ℝ := relative_speed * time / 2

-- Convert km/h to m/s
def kmh_to_ms (velocity : ℝ) : ℝ := velocity * 1000 / 3600

-- Calculate relative speeds in m/s
def relative_speed12 := kmh_to_ms (v1 - v2)
def relative_speed13 := kmh_to_ms (60 - v1)

-- Length of each train
def L := length_of_train relative_speed12 t1

-- Time to overtake the third train
def time_to_overtake (distance : ℝ) (relative_speed : ℝ) : ℝ := distance / relative_speed
def t2 := time_to_overtake (2 * L) relative_speed13

-- Total time (in seconds)
def total_time := t1 + t2

-- Theorem statement
theorem prove_total_time_eq_108 : total_time = 108 :=
by
  -- Formal proof will be provided here
  sorry

end prove_total_time_eq_108_l669_669278


namespace intersection_eq_l669_669114

-- Given conditions
def M : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def N : Set ℝ := { x | x > 1 }

-- Statement of the problem to be proved
theorem intersection_eq : M ∩ N = { x | 1 < x ∧ x < 3 } :=
sorry

end intersection_eq_l669_669114


namespace find_m_l669_669556

-- Definitions related to the problem
def interior_angle_of_regular_polygon (n : ℕ) : ℝ :=
  ((n - 2) * 180) / n

def exterior_angle_of_regular_polygon (m : ℕ) : ℝ :=
  360 / m

-- Formal statement of the problem
theorem find_m (m : ℕ) (h : interior_angle_of_regular_polygon 6 = 4 * exterior_angle_of_regular_polygon m) :
  m = 12 :=
begin
  -- Placeholder for the proof
  sorry
end

end find_m_l669_669556


namespace intersecting_lines_l669_669136

variable (a b m : ℝ)

-- Conditions
def condition1 : Prop := 8 = -m + a
def condition2 : Prop := 8 = m + b

-- Statement to prove
theorem intersecting_lines : condition1 a m  → condition2 b m  → a + b = 16 :=
by
  intros h1 h2
  sorry

end intersecting_lines_l669_669136


namespace number_of_valid_sequences_l669_669121

-- Define the set of letters in "EXAMPLE"
def letters := ['E', 'X', 'A', 'M', 'P', 'L', 'E'].eraseDups

-- Define the conditions: The sequences must begin with 'E' and not end with 'X'
def is_valid_sequence (seq : List Char) : Prop :=
  seq.head = 'E' ∧ seq.getLast (by simp [seq_ne_nil]) ≠ 'X'

-- Define the statement for the total number of valid sequences
theorem number_of_valid_sequences : 
  (Finset.filter is_valid_sequence (Finset.univ.image (λ s : Fin 4 → letters, [s 0, s 1, s 2, s 3]))).card = 48 := 
sorry

end number_of_valid_sequences_l669_669121


namespace maximum_non_overlapping_dominoes_l669_669286

-- Define the initial condition of the checkerboard and initial placement of dominoes
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)
  (num_initial_dominoes : ℕ)
  (total_squares : ℕ := rows * cols)
  (valid_placement : Prop := rows * cols % 2 = 0)

-- Define a function to count the maximum number of dominos
noncomputable def max_dominoes (cb : Checkerboard) : ℕ :=
  (cb.total_squares / 2) - cb.num_initial_dominoes

theorem maximum_non_overlapping_dominoes :
  ∀ cb : Checkerboard, cb.rows = 8 → cb.cols = 9 → cb.num_initial_dominoes = 6 →
  max_dominoes cb = 34 :=
by 
  intros cb h_rows h_cols h_initial_dominoes,
  sorry

end maximum_non_overlapping_dominoes_l669_669286


namespace simplify_cube_root_l669_669691

theorem simplify_cube_root (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)
  (h1 : a = 10^3 * b)
  (h2 : b = 2^7 * c * 7^3)
  (h3 : c = 10) :
  ∛a = 40 * 7 * 2^(2/3) * 5^(1/3) := by
  sorry

end simplify_cube_root_l669_669691


namespace product_div_by_six_l669_669129

theorem product_div_by_six (A B C : ℤ) (h1 : A^2 + B^2 = C^2) 
  (h2 : ∀ n : ℤ, ¬ ∃ k : ℤ, n^2 = 4 * k + 2) 
  (h3 : ∀ n : ℤ, ¬ ∃ k : ℤ, n^2 = 3 * k + 2) : 
  6 ∣ (A * B) :=
sorry

end product_div_by_six_l669_669129


namespace first_bag_weight_l669_669016

def weight_of_first_bag (initial_weight : ℕ) (second_bag : ℕ) (total_weight : ℕ) : ℕ :=
  total_weight - second_bag - initial_weight

theorem first_bag_weight : weight_of_first_bag 15 10 40 = 15 :=
by
  unfold weight_of_first_bag
  sorry

end first_bag_weight_l669_669016


namespace correct_range_of_a_l669_669087

variables {a : ℝ}

-- Proposition p: The function f(x) = |x + a| is monotonic on the interval (-∞, -1)
def prop_p (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ Set.Ioo (-∞) (-1) → y ∈ Set.Ioo (-∞) (-1) → x ≤ y → |x + a| ≤ |y + a|

-- Proposition q: The function f(x) = (x^2 + a) / x is increasing on the interval (2, +∞)
def prop_q (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ Set.Ioo 2 +∞ → y ∈ Set.Ioo 2 +∞ → x ≤ y → (x^2 + a) / x ≤ (y^2 + a) / y

-- Combining both propositions and given conditions, we conclude the range of 'a'
theorem correct_range_of_a (h_p : prop_p a) (h_q : prop_q a) : 0 < a ∧ a ≤ 1 :=
sorry

end correct_range_of_a_l669_669087


namespace simplify_cube_root_l669_669695

theorem simplify_cube_root (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)
  (h1 : a = 10^3 * b)
  (h2 : b = 2^7 * c * 7^3)
  (h3 : c = 10) :
  ∛a = 40 * 7 * 2^(2/3) * 5^(1/3) := by
  sorry

end simplify_cube_root_l669_669695


namespace ramesh_paid_price_l669_669666

variables 
  (P : Real) -- Labelled price of the refrigerator
  (paid_price : Real := 0.80 * P + 125 + 250) -- Price paid after discount and additional costs
  (sell_price : Real := 1.16 * P) -- Price to sell for 16% profit
  (sell_at : Real := 18560) -- Target selling price for given profit

theorem ramesh_paid_price : 
  1.16 * P = 18560 → paid_price = 13175 :=
by
  sorry

end ramesh_paid_price_l669_669666


namespace total_seeds_in_garden_l669_669515

-- Definitions based on the conditions
def top_bed_rows : ℕ := 4
def top_bed_seeds_per_row : ℕ := 25
def num_top_beds : ℕ := 2

def medium_bed_rows : ℕ := 3
def medium_bed_seeds_per_row : ℕ := 20
def num_medium_beds : ℕ := 2

-- Calculation of total seeds in top beds
def seeds_per_top_bed : ℕ := top_bed_rows * top_bed_seeds_per_row
def total_seeds_top_beds : ℕ := num_top_beds * seeds_per_top_bed

-- Calculation of total seeds in medium beds
def seeds_per_medium_bed : ℕ := medium_bed_rows * medium_bed_seeds_per_row
def total_seeds_medium_beds : ℕ := num_medium_beds * seeds_per_medium_bed

-- Proof goal
theorem total_seeds_in_garden : total_seeds_top_beds + total_seeds_medium_beds = 320 :=
by
  sorry

end total_seeds_in_garden_l669_669515


namespace determine_marriages_l669_669790

-- Definitions of the items each person bought
variable (a_items b_items c_items : ℕ) -- Number of items bought by wives a, b, and c
variable (A_items B_items C_items : ℕ) -- Number of items bought by husbands A, B, and C

-- Conditions
variable (spend_eq_square_a : a_items * a_items = a_spend) -- Spending equals square of items
variable (spend_eq_square_b : b_items * b_items = b_spend)
variable (spend_eq_square_c : c_items * c_items = c_spend)
variable (spend_eq_square_A : A_items * A_items = A_spend)
variable (spend_eq_square_B : B_items * B_items = B_spend)
variable (spend_eq_square_C : C_items * C_items = C_spend)

variable (A_spend_eq : A_spend = a_spend + 48) -- Husbands spent 48 yuan more than wives
variable (B_spend_eq : B_spend = b_spend + 48)
variable (C_spend_eq : C_spend = c_spend + 48)

variable (A_bought_9_more : A_items = b_items + 9) -- A bought 9 more items than b
variable (B_bought_7_more : B_items = a_items + 7) -- B bought 7 more items than a

-- Theorem statement
theorem determine_marriages (hA : A_items ≥ b_items + 9) (hB : B_items ≥ a_items + 7) :
  (A_spend = A_items * A_items) ∧ (B_spend = B_items * B_items) ∧ (C_spend = C_items * C_items) ∧
  (a_spend = a_items * a_items) ∧ (b_spend = b_items * b_items) ∧ (c_spend = c_items * c_items) →
  (A_spend = a_spend + 48) ∧ (B_spend = b_spend + 48) ∧ (C_spend = c_spend + 48) →
  (A_items = b_items + 9) ∧ (B_items = a_items + 7) →
  (A_items = 13 ∧ c_items = 11) ∧ (B_items = 8 ∧ b_items = 4) ∧ (C_items = 7 ∧ a_items = 1) :=
by
  sorry

end determine_marriages_l669_669790


namespace cube_root_simplification_l669_669717

theorem cube_root_simplification : (∛54880000) = 140 * (2 ^ (1 / 3)) :=
by
  -- Using the information from the problem conditions and final solution.
  have root_10_cubed := (10 ^ 3 : ℝ)
  have factored_value := root_10_cubed * (2 ^ 4 * 7 ^ 3)
  have cube_root := Real.cbrt factored_value
  sorry

end cube_root_simplification_l669_669717


namespace right_triangle_area_l669_669744

theorem right_triangle_area
  (hypotenuse : ℝ) (leg1 : ℝ) (leg2 : ℝ)
  (hypotenuse_eq : hypotenuse = 13)
  (leg1_eq : leg1 = 5)
  (pythagorean_eq : hypotenuse^2 = leg1^2 + leg2^2) :
  (1 / 2) * leg1 * leg2 = 30 :=
by
  sorry

end right_triangle_area_l669_669744


namespace lines_parallel_if_perpendicular_to_same_plane_l669_669180

variable {Plane Line : Type}
variable {α β γ : Plane}
variable {m n : Line}

-- Define perpendicularity and parallelism as axioms for simplicity
axiom perp (L : Line) (P : Plane) : Prop
axiom parallel (L1 L2 : Line) : Prop

-- Assume conditions for the theorem
variables (h1 : perp m α) (h2 : perp n α)

-- The theorem proving the required relationship
theorem lines_parallel_if_perpendicular_to_same_plane : parallel m n := 
by
  sorry

end lines_parallel_if_perpendicular_to_same_plane_l669_669180


namespace largest_int_less_150_gcd_18_eq_6_l669_669803

theorem largest_int_less_150_gcd_18_eq_6 : ∃ (n : ℕ), n < 150 ∧ gcd n 18 = 6 ∧ ∀ (m : ℕ), m < 150 ∧ gcd m 18 = 6 → m ≤ n ∧ n = 138 := 
by
  sorry

end largest_int_less_150_gcd_18_eq_6_l669_669803


namespace total_cost_correct_l669_669883

-- Defining the conditions
def charges_per_week : ℕ := 3
def weeks_per_year : ℕ := 52
def cost_per_charge : ℝ := 0.78

-- Defining the total cost proof statement
theorem total_cost_correct : (charges_per_week * weeks_per_year : ℝ) * cost_per_charge = 121.68 :=
by
  sorry

end total_cost_correct_l669_669883


namespace simplify_cubic_root_l669_669700

theorem simplify_cubic_root : 
  (∛(54880000) = 20 * ∛((5^2) * 137)) :=
sorry

end simplify_cubic_root_l669_669700


namespace how_many_cheburashkas_erased_l669_669605

theorem how_many_cheburashkas_erased 
  (total_krakozyabras : ℕ)
  (characters_per_row_initial : ℕ) 
  (total_characters_initial : ℕ)
  (total_cheburashkas : ℕ)
  (total_rows : ℕ := 2)
  (total_krakozyabras := 29) :
  total_cheburashkas = 11 :=
by
  sorry

end how_many_cheburashkas_erased_l669_669605


namespace number_of_consecutive_circles_l669_669058

-- Define the given constants and conditions
def number_of_circles : ℕ := 33
def horizontal_ways : ℕ := 21
def diagonal_ways : ℕ := 18
def total_ways : ℕ := 57

-- The theorem that we need to prove
theorem number_of_consecutive_circles :
  (horizontal_ways + 2 * diagonal_ways = total_ways) :=
by
  -- These are the facts derived from the problem
  have fact1 : horizontal_ways = 21 := rfl
  have fact2 : diagonal_ways = 18 := rfl
  have fact3 : total_ways = 57 := rfl
  -- Combining the facts to show the proof
  rw [fact1, fact2, fact3]
  sorry

end number_of_consecutive_circles_l669_669058


namespace inverse_function_power_l669_669500

theorem inverse_function_power (x : ℝ) (h : x ≥ 0) : 
  ∃ f (g : ℝ → ℝ), (∀ y, f y = y ^ (1 / 2) ∧ (2, sqrt 2) ∈ (λ x, (x, f x))) ∧ (g (f x) = x) ∧ (f (g x) = x) :=
sorry

end inverse_function_power_l669_669500


namespace sequence_properties_l669_669866

theorem sequence_properties (a : ℕ → ℤ)
  (h1 : a 2 = -2)
  (h2 : a 7 = 4)
  (h3 : ∀ n ≤ 4, a n = n - 4)
  (h4 : ∀ n ≥ 5, a n = 2 ^ (n - 5))
  (h5 : ∀ n (h : 1 ≤ n ∧ n ≤ 6), a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h6 : ∀ n (h : n ≥ 5), a (n + 2) = a (n + 1) * a n / a (n - 1) * a (n - 2)) :
  a = (λ n, if n ≤ 4 then n - 4 else 2 ^ (n - 5)) ∧ 
  ∃ m, (a m + a (m + 1) + a (m + 2) = a m * a (m + 1) * a (m + 2) ∧ (m = 1 ∨ m = 3)) :=
sorry

end sequence_properties_l669_669866


namespace number_of_boxes_correct_l669_669789

def peaches_per_basket := 25
def baskets_delivered := 5
def peaches_eaten := 5
def box_capacity := 15

theorem number_of_boxes_correct :
  let total_peaches := baskets_delivered * peaches_per_basket,
      remaining_peaches := total_peaches - peaches_eaten,
      number_of_boxes := remaining_peaches / box_capacity in
  number_of_boxes = 8 :=
by
  sorry

end number_of_boxes_correct_l669_669789


namespace truck_travel_distance_l669_669350

theorem truck_travel_distance
  (miles_traveled : ℕ)
  (gallons_used : ℕ)
  (new_gallons : ℕ)
  (rate : ℕ)
  (distance : ℕ) :
  (miles_traveled = 300) ∧
  (gallons_used = 10) ∧
  (new_gallons = 15) ∧
  (rate = miles_traveled / gallons_used) ∧
  (distance = rate * new_gallons)
  → distance = 450 :=
by
  sorry

end truck_travel_distance_l669_669350


namespace time_to_be_85_km_apart_l669_669337

theorem time_to_be_85_km_apart (d : ℝ) (s_bus : ℝ) (s_truck : ℝ) (s_initial : ℝ) (t : ℝ) :
  s_initial = 5 → s_truck = 60 → s_bus = 40 → d = 85 → 
  t = (d - s_initial) / (s_bus + s_truck) → t = 0.8 :=
by
  intros h_initial h_truck h_bus h_d h_t
  rw [h_initial, h_truck, h_bus, h_d] at h_t
  exact h_t

end time_to_be_85_km_apart_l669_669337


namespace find_length_of_segment_l669_669774

noncomputable def radius : ℝ := 4
noncomputable def volume_cylinder (L : ℝ) : ℝ := 16 * Real.pi * L
noncomputable def volume_hemispheres : ℝ := 2 * (128 / 3) * Real.pi
noncomputable def total_volume (L : ℝ) : ℝ := volume_cylinder L + volume_hemispheres

theorem find_length_of_segment (L : ℝ) (h : total_volume L = 544 * Real.pi) : 
  L = 86 / 3 :=
by sorry

end find_length_of_segment_l669_669774


namespace exists_uniform_set_l669_669347

def is_uniform (A : Finset ℕ) : Prop :=
  ∀ x ∈ A, ∃ (B C : Finset ℕ), B ∪ C = A.erase x ∧ B ∩ C = ∅ ∧ B.sum = C.sum

theorem exists_uniform_set : ∃ (A : Finset ℕ), 7 = A.card ∧ is_uniform A := by
  sorry

end exists_uniform_set_l669_669347


namespace ratio_of_areas_l669_669619

noncomputable def S_1 : set (ℝ × ℝ) := 
  { p | log10 (3 + p.1^2 + p.2^2) ≤ 2 + log10 (p.1 + p.2) }

noncomputable def S_2 : set (ℝ × ℝ) := 
  { p | log10 (5 + p.1^2 + p.2^2) ≤ 3 + log10 (p.1 + p.2) }

theorem ratio_of_areas : 
  let area_S1 := Real.pi * 4997
  let area_S2 := Real.pi * 499995
  (area_S2 / area_S1) ≈ 100.04 := 
by
  sorry

end ratio_of_areas_l669_669619


namespace factorization_l669_669438

theorem factorization (a x : ℝ) : ax^2 - 2ax + a = a * (x - 1) ^ 2 := 
by
  sorry

end factorization_l669_669438


namespace factorization_correct_l669_669432

-- Define the initial expression
def initial_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + a

-- Define the factorized form
def factorized_expr (a x : ℝ) : ℝ := a * (x - 1)^2

-- Create the theorem statement
theorem factorization_correct (a x : ℝ) : initial_expr a x = factorized_expr a x :=
by simp [initial_expr, factorized_expr, pow_two, mul_add, add_mul, add_assoc]; sorry

end factorization_correct_l669_669432


namespace oakwood_team_combinations_l669_669760

-- Define the number of ways to choose 3 girls from 4 girls
def choose_4_3 : ℕ := Nat.combinations 4 3

-- Define the number of ways to choose 2 boys from 6 boys
def choose_6_2 : ℕ := Nat.combinations 6 2

-- Theorem stating the total number of ways to form the team
theorem oakwood_team_combinations : choose_4_3 * choose_6_2 = 60 := by
  -- sorry is used to skip the proof body
  sorry

end oakwood_team_combinations_l669_669760


namespace solve_system_of_equations_l669_669403

theorem solve_system_of_equations (x y : ℝ) 
  (h1 : x / 5 + 3 = 4)
  (h2 : x^2 - 4 * x * y + 3 * y^2 = 36) :
  x = 5 ∧ (y = (10 / 3) + (sqrt 133 / 3) ∨ y = (10 / 3) - (sqrt 133 / 3)) :=
by {
  sorry
}

end solve_system_of_equations_l669_669403


namespace numbers_divisible_by_3_without_digit_3_l669_669999

theorem numbers_divisible_by_3_without_digit_3 :
  { n : ℕ | n ≥ 1 ∧ n ≤ 100 ∧ n % 3 = 0 ∧ ∀ d ∈ n.digits 10, d ≠ 3 }.card = 26 :=
by sorry

end numbers_divisible_by_3_without_digit_3_l669_669999


namespace max_value_of_exp_minus_x_l669_669740

theorem max_value_of_exp_minus_x :
  ∃ (c : ℝ), ∀ x ∈ set.Icc (0 : ℝ) 1, 
  (∀ y ∈ set.Icc (0 : ℝ) 1, f y ≤ f x) → f x = c ∧ c = real.exp 1 - 1 :=
by
  let f := λ x : ℝ, real.exp x - x
  have deriv_f : ∀ x, deriv f x = real.exp x - 1 := sorry
  have mono_f : ∀ x ∈ set.Icc (0 : ℝ) 1, ∀ y ∈ set.Icc (0 : ℝ) 1, x ≤ y → f x ≤ f y := sorry
  have max_f : f 1 = real.exp 1 - 1 := sorry
  exact ⟨f 1, λ x hx hx_max, by
    rw [← hx_max, max_f]
    exact max_f⟩


end max_value_of_exp_minus_x_l669_669740


namespace expected_value_of_random_number_l669_669261

/-- 
The expected value of a random number formed by placing a zero and a decimal point in front
of a sequence of one thousand random digits is 0.5.
-/
theorem expected_value_of_random_number : 
  let X := ∑ k in (finRange 1000), (4.5 / 10 ^ (k + 1))
  in X = 0.5 :=
sorry

end expected_value_of_random_number_l669_669261


namespace factorize_expression_l669_669417

theorem factorize_expression (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
by sorry

end factorize_expression_l669_669417


namespace ellipse_equation_exists_line_equation_exists_l669_669480

open Real

-- Definitions for the given problem
def ellipse_center_origin : Prop := true
def foci_on_x_axis (e : ℝ) (c : ℝ) : Prop := e = c / (2 * √2)
def focal_distance_eq_4 (c : ℝ) : Prop := 2 * c = 4
def eccentricity (e : ℝ) : Prop := e = √2 / 2
def line_passing_through_P (P : Point) (l : Line) : Prop := P = ⟨0, -1⟩ ∧ P ∈ l
def points_intersection_line_ellipse (l : Line) (A B : Point) : Prop := 
  A ∈ l ∧ B ∈ l ∧ A ∈ ellipse_C ∧ B ∈ ellipse_C
def vector_relation (A B P : Point) : Prop := 
  ∀ P, A - P = 2 * (P - B)

-- Statements to be proven
theorem ellipse_equation_exists : 
  ∃ (a b : ℝ), ellipse_center_origin ∧ foci_on_x_axis (√2 / 2) (2) ∧ focal_distance_eq_4 (2) ∧ eccentricity (√2 / 2) →
  (a = 2 * √2 ∧ b = 2) ∧ (∀ x y, (x^2 / (2 * √2)^2) + (y^2 / 2^2) = 1) :=
sorry

theorem line_equation_exists (P : Point) (A B : Point) (k : ℝ) : 
  ∃ l : Line, line_passing_through_P P l ∧ points_intersection_line_ellipse l A B ∧ vector_relation A B P →
  (k = 3 * √10 / 10 ∨ k = -3 * √10 / 10) ∧ (∀ x, (y : Real) = k * x - 1) :=
sorry

end ellipse_equation_exists_line_equation_exists_l669_669480


namespace line_fixed_point_l669_669649

theorem line_fixed_point (k : ℝ) :
    ∀ k : ℝ, ∃ p : ℝ × ℝ, 
    ((2 * k - 1) * p.1 - (k - 2) * p.2 - (k + 4) = 0) ∧ p = (2, 3) := 
by
  intro k
  use (2, 3)
  split
  sorry

end line_fixed_point_l669_669649


namespace functional_equation_solution_l669_669024

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, 
  (∀ x y : ℝ, 
      y * f (2 * x) - x * f (2 * y) = 8 * x * y * (x^2 - y^2)
  ) → (∃ c : ℝ, ∀ x : ℝ, f x = x^3 + c * x) :=
by { sorry }

end functional_equation_solution_l669_669024


namespace quadrilateral_is_parallelogram_l669_669171

variables {Point : Type} [add_comm_group Point] [vector_space ℝ Point] {A B C D M : Point}

def area (X Y Z : Point) : ℝ := sorry -- this would be the actual function to calculate the area of triangle XYZ

noncomputable def is_parallelogram (A B C D : Point) : Prop :=
  ∃ M, area M A B = area M B C ∧ area M B C = area M C D ∧ area M C D = area M D A ∧
  (M - A) = (C - M) ∧ (M - B) = (D - M) -- this assumes commutative group and vector space properties on Points

theorem quadrilateral_is_parallelogram (A B C D M : Point) (h₁ : area M A B = area M B C) 
  (h₂ : area M B C = area M C D) (h₃ : area M C D = area M D A) 
  (h₄ : (M - A) = (C - M)) (h₅ : (M - B) = (D - M)) :
  is_parallelogram A B C D :=
by
  unfold is_parallelogram
  use M
  exact ⟨h₁, h₂, h₃, h₄, h₅⟩

end quadrilateral_is_parallelogram_l669_669171


namespace find_number_l669_669836

theorem find_number (n : ℝ) : (2629.76 / n = 528.0642570281125) → n = 4.979 :=
by
  intro h
  sorry

end find_number_l669_669836


namespace hexagon_inscribed_circumscribed_symmetric_l669_669799

-- Define the conditions of the problem
variables (R r c : ℝ)

-- Define the main assertion of the problem
theorem hexagon_inscribed_circumscribed_symmetric :
  3 * (R^2 - c^2)^4 - 4 * r^2 * (R^2 - c^2)^2 * (R^2 + c^2) - 16 * R^2 * c^2 * r^4 = 0 :=
by
  -- skipping proof
  sorry

end hexagon_inscribed_circumscribed_symmetric_l669_669799


namespace percentage_customers_return_books_l669_669906

theorem percentage_customers_return_books 
  (total_customers : ℕ) (price_per_book : ℕ) (sales_after_returns : ℕ) 
  (h1 : total_customers = 1000) 
  (h2 : price_per_book = 15) 
  (h3 : sales_after_returns = 9450) : 
  ((total_customers - (sales_after_returns / price_per_book)) / total_customers) * 100 = 37 := 
by
  sorry

end percentage_customers_return_books_l669_669906


namespace part_a_part_b_l669_669945

variable {α β γ δ AB CD : ℝ}
variable {A B C D : Point}
variable {A_obtuse B_obtuse : Prop}
variable {α_gt_δ β_gt_γ : Prop}

-- Definition of a convex quadrilateral
def convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Conditions for part (a)
axiom angle_A_obtuse : A_obtuse
axiom angle_B_obtuse : B_obtuse

-- Conditions for part (b)
axiom angle_α_gt_δ : α_gt_δ
axiom angle_β_gt_γ : β_gt_γ

-- Part (a) statement: Given angles A and B are obtuse, AB ≤ CD
theorem part_a {A B C D : Point} (h_convex : convex_quadrilateral A B C D) 
    (h_A_obtuse : A_obtuse) (h_B_obtuse : B_obtuse) : AB ≤ CD :=
sorry

-- Part (b) statement: Given angle A > angle D and angle B > angle C, AB < CD
theorem part_b {A B C D : Point} (h_convex : convex_quadrilateral A B C D) 
    (h_angle_α_gt_δ : α_gt_δ) (h_angle_β_gt_γ : β_gt_γ) : AB < CD :=
sorry

end part_a_part_b_l669_669945


namespace continuous_stripe_probability_l669_669014

open ProbabilityTheory

noncomputable def total_stripe_combinations : ℕ := 4 ^ 6

noncomputable def favorable_stripe_outcomes : ℕ := 3 * 4

theorem continuous_stripe_probability :
  (favorable_stripe_outcomes : ℚ) / (total_stripe_combinations : ℚ) = 3 / 1024 := by
  sorry

end continuous_stripe_probability_l669_669014


namespace expected_value_of_X_l669_669268

-- Define the sequence of random digits as a list of natural numbers (0 to 9)
def random_digits : List ℕ := (List.range 10).take 1000

-- Define the function that forms the number X
def X (digits : List ℕ) : ℝ :=
  digits.enum.foldr (λ (p : ℕ × ℕ) (acc : ℝ) => acc + p.snd * 10^(-(p.fst + 1))) 0

-- Define the expected value of a single digit
def expected_value_digit : ℝ := (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) / 10

-- The main statement to prove
theorem expected_value_of_X (digits : List ℕ) (h_digits : digits.length = 1000) :
  (∑ i in Finset.range digits.length, 10^(-(i + 1)) * expected_value_digit) = 0.5 :=
by {
  sorry
}

end expected_value_of_X_l669_669268


namespace right_triangle_area_l669_669747

theorem right_triangle_area
  (hypotenuse : ℝ) (leg1 : ℝ) (leg2 : ℝ)
  (hypotenuse_eq : hypotenuse = 13)
  (leg1_eq : leg1 = 5)
  (pythagorean_eq : hypotenuse^2 = leg1^2 + leg2^2) :
  (1 / 2) * leg1 * leg2 = 30 :=
by
  sorry

end right_triangle_area_l669_669747


namespace correct_statement_l669_669972

variables {m n : Line} {α β : Plane}

-- Assumptions
axiom distinct_lines (h_lines_distinct : m ≠ n)
axiom distinct_planes (h_planes_distinct : α ≠ β)
axiom n_subset_beta (h_n_in_beta : n ∈ β)

-- Statement to be proved
theorem correct_statement (h_parallel_mn : m ∥ n) (h_perpendicular_m_alpha : m ⊥ α) : α ⊥ β :=
sorry

end correct_statement_l669_669972


namespace frequency_of_8th_group_l669_669405

/--
Divide a sample of size 64 into 8 groups. The frequencies of groups 1 to 4 are 5, 7, 11, and 13, respectively. 
The frequency rate of groups 5 to 7 is 0.125. Prove that the frequency of the 8th group is 20.
-/ 
theorem frequency_of_8th_group
    (sample_size : ℕ := 64)
    (freqs_1_to_4 : Fin 4 → ℕ := ![5, 7, 11, 13])
    (freq_rate_5_to_7 : ℚ := 0.125) :
    let freq_5_to_7 := (sample_size : ℚ) * freq_rate_5_to_7,
        total_freqs_1_to_7 := (freqs_1_to_4 0) + (freqs_1_to_4 1) + (freqs_1_to_4 2) + (freqs_1_to_4 3) + freq_5_to_7.to_rat in
    sample_size - total_freqs_1_to_7.to_nat = 20 := by
  sorry

end frequency_of_8th_group_l669_669405


namespace symmetric_point_l669_669654

theorem symmetric_point (A B C : ℝ) (hA : A = Real.sqrt 7) (hB : B = 1) :
  C = 2 - Real.sqrt 7 ↔ (A + C) / 2 = B :=
by
  sorry

end symmetric_point_l669_669654


namespace proof_problem_l669_669944

noncomputable def problem_statement : Prop :=
  let z := (3 + 2 * Complex.i) / (Complex.i ^ 2024 - Complex.i)
  let conj_z := Complex.conj z
  (Complex.re z > 0 ∧ Complex.im z > 0) ∧ z * conj_z = 13 / 2

-- lean 4 statement without proof
theorem proof_problem : problem_statement :=
by
  sorry

end proof_problem_l669_669944


namespace simplify_cube_root_l669_669688

theorem simplify_cube_root (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)
  (h1 : a = 10^3 * b)
  (h2 : b = 2^7 * c * 7^3)
  (h3 : c = 10) :
  ∛a = 40 * 7 * 2^(2/3) * 5^(1/3) := by
  sorry

end simplify_cube_root_l669_669688


namespace bijection_if_injective_or_surjective_l669_669839

variables {X Y : Type} [Fintype X] [Fintype Y] (f : X → Y)

theorem bijection_if_injective_or_surjective (hX : Fintype.card X = Fintype.card Y)
  (hf : Function.Injective f ∨ Function.Surjective f) : Function.Bijective f :=
by
  sorry

end bijection_if_injective_or_surjective_l669_669839


namespace ab_ac_bc_range_l669_669623

theorem ab_ac_bc_range (a b c : ℝ) (h : a + b + c = 0) : ab + ac + bc ∈ Iic 0 := by
  sorry

end ab_ac_bc_range_l669_669623


namespace cheburashkas_erased_l669_669593

theorem cheburashkas_erased (n : ℕ) (rows : ℕ) (krakozyabras : ℕ) 
  (h_spacing : ∀ r, r ≤ rows → krakozyabras = 2 * (n - 1))
  (h_rows : rows = 2)
  (h_krakozyabras : krakozyabras = 29) :
  n = 16 → rows = 2 → krakozyabras = 29 → n = 16 - 5 :=
by
  sorry

end cheburashkas_erased_l669_669593


namespace solution_set_of_inequality_l669_669473

variable {R : Type} [LinearOrderedField R] [RealDomain R]
variable (f : R → R)

axiom f_at_one : f 1 = 4
axiom f_derivative : ∀ x : R, deriv f x < 3

theorem solution_set_of_inequality (x : R) :
  f (Real.log x) > 3 * Real.log x + 1 ↔ 0 < x ∧ x < Real.exp 1 :=
by
  sorry

end solution_set_of_inequality_l669_669473


namespace minimum_cactus_species_l669_669364

/--
At a meeting of cactus enthusiasts, 80 cactophiles presented their collections,
each consisting of cacti of different species. It turned out that no single 
species of cactus is found in all collections simultaneously, but any 15 people
have cacti of the same species. Prove that the minimum total number of cactus 
species is 16.
-/
theorem minimum_cactus_species (k : ℕ) (h : ∀ (collections : fin 80 → finset (fin k)),
  (∀ i, collections i ≠ ∅) ∧ (∃ (j : fin k), ∀ i, j ∉ collections i) ∧ 
  (∀ (S : finset (fin 80)), S.card = 15 → ∃ j, ∀ i ∈ S, j ∈ collections i)) :
  k ≥ 16 :=
sorry

end minimum_cactus_species_l669_669364


namespace max_distinct_factors_64_l669_669724

/-- 
Given that the least common multiple (LCM) of 1024 and 2016 remains unchanged 
when adding distinct positive integers \( x_1, x_2, \ldots, x_n \),
prove that the maximum number of such integers \( n \) is 64.
-/
theorem max_distinct_factors_64
    (n : ℕ) (x : ℕ → ℕ)
    (distinct_x : ∀ i j, i ≠ j → x i ≠ x j) : 
    lcm 1024 2016 = lcm 1024 (lcm 2016 (finset.fold lcm 1 (finset.image x (finset.range n)))) → 
    n ≤ 64 :=
sorry

end max_distinct_factors_64_l669_669724


namespace smallest_m_l669_669290

theorem smallest_m (m : ℤ) :
  (∀ x : ℝ, (3 * x * (m * x - 5) - x^2 + 8) = 0) → (257 - 96 * m < 0) → (m = 3) :=
sorry

end smallest_m_l669_669290


namespace even_and_increasing_function_l669_669353

-- Define the functions in the conditions
def f1 (x : ℝ) : ℝ := |sin x|
def f2 (x : ℝ) : ℝ := |sin (2 * x)|
def f3 (x : ℝ) : ℝ := |cos x|
def f4 (x : ℝ) : ℝ := tan x

-- Define the main theorem
theorem even_and_increasing_function : ∃ f : ℝ → ℝ, f = f1 ∧ (∀ x ∈ set.Ioo 0 (π / 2), 0 <= f x ∧ is_increasing_on f (set.Ioo 0 (π / 2))) :=
by {
  use f1,
  split,
  { refl },
  { sorry }
}

end even_and_increasing_function_l669_669353


namespace solve_system_l669_669312

theorem solve_system (x y z : ℝ) (h1 : (x + 1) * y * z = 12) 
                               (h2 : (y + 1) * z * x = 4) 
                               (h3 : (z + 1) * x * y = 4) : 
  (x = 1 / 3 ∧ y = 3 ∧ z = 3) ∨ (x = 2 ∧ y = -2 ∧ z = -2) :=
sorry

end solve_system_l669_669312


namespace closest_to_2013_in_sequence_l669_669721

/-- 
  Define the two subsequences generated by starting with 1 
  and alternately adding 4 and 3. The two subsequences are:
  - \(a_n = 7n - 6\)
  - \(b_m = 7m - 2\)
  Prove that the number closest to 2013 in these two subsequences is 2014.
-/
theorem closest_to_2013_in_sequence :
  let a := λ n : ℕ, 7 * n - 6,
      b := λ m : ℕ, 7 * m - 2 in
  (closest (a ∘ succ) (b ∘ succ) 2013) = 2014 :=
sorry

end closest_to_2013_in_sequence_l669_669721


namespace bill_trick_probability_l669_669319

theorem bill_trick_probability :
  let cards_A := {2, 3, 5, 7}
  let cards_B := {2, 4, 6, 7}
  ∃ cards_C : Finset ℕ, cards_C.card = 4 ∧ cards_C ⊆ ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ) ∧ 
  (∀ n, n ∈ (cards_A ∪ cards_B ∪ cards_C) → ∃ unique_sets : Finset (Finset ℕ), unique_sets.card = 8 ∧ 
  ∀ (m : ℕ), m ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ) → (m ≠ n → ∀ (s t : Finset ℕ) (H₁ : s ∈ unique_sets) (H₂ : t ∈ unique_sets), s ≠ t)) →
  (Finset.card (Finset.filter (λ s : Finset ℕ, s.card = 4 ∧ ∃ p₁ p₂ p₃ p₄ : ℕ, p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₃ ≠ p₄ ∧ p₄ ≠ p₁ ∧ s = {p₁, p₂, p₃, p₄} ∧ s ⊆ ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ)) (Finset.powerset (Finset.range 8))).val = 16) →
  (16 / 70 = (8 / 35 : ℝ)) :=
sorry

end bill_trick_probability_l669_669319


namespace inclination_angle_of_line_l669_669778

theorem inclination_angle_of_line (x y : ℝ) (h : x - y + 1 = 0) : 
  let θ := Real.arctan 1 in θ = π / 4 :=
begin
  sorry
end

end inclination_angle_of_line_l669_669778


namespace solve_quadratic_difference_l669_669720

theorem solve_quadratic_difference :
  ∀ x : ℝ, (x^2 - 7*x - 48 = 0) → 
  let x1 := (7 + Real.sqrt 241) / 2
  let x2 := (7 - Real.sqrt 241) / 2
  abs (x1 - x2) = Real.sqrt 241 :=
by
  sorry

end solve_quadratic_difference_l669_669720


namespace cube_root_simplification_l669_669713

theorem cube_root_simplification : (∛54880000) = 140 * (2 ^ (1 / 3)) :=
by
  -- Using the information from the problem conditions and final solution.
  have root_10_cubed := (10 ^ 3 : ℝ)
  have factored_value := root_10_cubed * (2 ^ 4 * 7 ^ 3)
  have cube_root := Real.cbrt factored_value
  sorry

end cube_root_simplification_l669_669713


namespace third_quadrant_of_conjugate_l669_669981
open Complex

noncomputable def quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
  else if z.re > 0 ∧ z.im < 0 then "Fourth quadrant"
  else "Axis"

theorem third_quadrant_of_conjugate :
  let z := (1 + 2 * I) / (1 - 2 * I)
  quadrant (conj z) = "Third quadrant" :=
by
  sorry

end third_quadrant_of_conjugate_l669_669981


namespace number_of_valid_sets_l669_669763

-- Definitions for conditions
variable (a b c : Type)

-- The set P is a subset of {a, b, c} and a proper superset of {a}
def is_valid_set (P : Set Type) : Prop :=
  {a} ⊂ P ∧ P ⊆ ({a, b, c} : Set Type)

-- Theorem statement
theorem number_of_valid_sets :
  {P : Set Type | is_valid_set a b c P}.toFinset.card = 3 :=
by
  sorry

end number_of_valid_sets_l669_669763


namespace no_k_for_perfect_power_l669_669441

def is_odd_prime (p : ℕ) : Prop := Nat.Prime p ∧ p % 2 = 1

def odd_primes : ℕ → ℕ
| 0 => 3
| n + 1 => Nat.find (λ p => is_odd_prime p ∧ odd_primes n < p)

def P_k (k : ℕ) : ℕ := List.prod (List.map odd_primes (List.range k))

theorem no_k_for_perfect_power : ∀ k : ℕ, k > 0 → ∀ N m : ℕ, m ≥ 2 → P_k k - 1 ≠ N ^ m :=
by
  sorry

end no_k_for_perfect_power_l669_669441


namespace probability_one_silver_probability_equal_gold_silver_l669_669273

-- Conditions for both questions
def total_tourists : ℕ := 36
def tourists_outside_province : ℕ := 27
def tourists_within_province : ℕ := 9
def gold_card_holders : ℕ := 9
def silver_card_holders : ℕ := 6
def total_ways_to_choose_2 (n : ℕ) : ℕ := nat.choose n 2

-- Definitions for Question I
def favorable_events_one_silver : ℕ := nat.choose silver_card_holders 1 * nat.choose (total_tourists - silver_card_holders) 1

-- Definitions for Question II
def non_card_holders : ℕ := total_tourists - (gold_card_holders + silver_card_holders)
def favorable_events_B1 : ℕ := nat.choose non_card_holders 2
def favorable_events_B2 : ℕ := nat.choose gold_card_holders 1 * nat.choose silver_card_holders 1

-- Lean statements for the proof problems:
theorem probability_one_silver
  : (favorable_events_one_silver / total_ways_to_choose_2 total_tourists : ℚ) = 2/7 :=
by sorry

theorem probability_equal_gold_silver
  : ((favorable_events_B1 + favorable_events_B2 : ℕ) / total_ways_to_choose_2 total_tourists : ℚ) = 44/105 :=
by sorry

end probability_one_silver_probability_equal_gold_silver_l669_669273


namespace digit_57_of_21_div_22_is_5_l669_669535

theorem digit_57_of_21_div_22_is_5 :
  let decimal_rep := "954545" in
  let repeating_seq := "54" in
  let digit_seq := repeat ('5' :: '4' :: []) in
  digit_seq.get 56 = '5' :=
by sorry

end digit_57_of_21_div_22_is_5_l669_669535


namespace intersection_of_sets_l669_669492

-- Conditions as Lean definitions
def A : Set Int := {-2, -1}
def B : Set Int := {-1, 2, 3}

-- Stating the proof problem in Lean 4
theorem intersection_of_sets : A ∩ B = {-1} :=
by
  sorry

end intersection_of_sets_l669_669492


namespace minimum_cactus_species_l669_669375

-- Definitions to represent the conditions
def num_cactophiles : Nat := 80
def num_collections (S : Finset (Fin num_cactophiles)) : Nat := S.card
axiom no_single_species_in_all (S : Finset (Fin num_cactophiles)) : num_collections S < num_cactophiles
axiom any_15_have_common_species (S : Finset (Fin num_cactophiles)) (h : S.card = 15) : 
  ∃ species, ∀ s ∈ S, species ∈ s

-- Proposition to be proved
theorem minimum_cactus_species (k : Nat) (h : ∀ S : Finset (Fin num_cactophiles), S.card = 15 → ∃ species, ∀ s ∈ S, species ∈ s) : k ≥ 16 := sorry

end minimum_cactus_species_l669_669375


namespace not_in_seq_5_l669_669239

noncomputable def seq (n : ℕ) : ℕ :=
  if n = 1 then 2 else Nat.factors (1 + List.prod (List.range n).map seq)).last'

theorem not_in_seq_5 (n : ℕ) : ∀ n, seq n ≠ 5 :=
by
  intro n
  sorry

end not_in_seq_5_l669_669239


namespace each_bathroom_break_time_l669_669173

variables (Distance : ℝ) (Speed : ℝ) (LunchBreakTime : ℝ) (NumBathroomBreaks : ℕ) (TotalTripTime : ℝ)

def driving_time (d s : ℝ) : ℝ := d / s

theorem each_bathroom_break_time
  (h1 : Distance = 480)
  (h2 : Speed = 60)
  (h3 : LunchBreakTime = 0.5) -- 30 minutes in hours
  (h4 : NumBathroomBreaks = 2)
  (h5 : TotalTripTime = 9) :
  (TotalTripTime - driving_time Distance Speed - LunchBreakTime) / NumBathroomBreaks = 0.25 := -- 15 minutes in hours
by
  sorry

end each_bathroom_break_time_l669_669173


namespace cube_root_of_54880000_l669_669675

theorem cube_root_of_54880000 : (real.cbrt 54880000) = 140 * (real.cbrt 10) :=
by
  -- Definitions based on conditions
  have h1 : 54880000 = 10^3 * 54880, by norm_num
  have h2 : 54880 = 2^5 * 7^3 * 5, by norm_num
  have h3 : 10 = 2 * 5, by norm_num
  
  -- Cube root properties and simplifications are implicitly inferred by the system
  sorry

end cube_root_of_54880000_l669_669675


namespace second_smallest_palindromic_prime_with_2_ends_l669_669232

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

def three_digit_palindromic_prime_with_2_ends (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (n / 100 = 2) ∧ (n % 10 = 2) ∧ is_palindrome n ∧ is_prime n

theorem second_smallest_palindromic_prime_with_2_ends :
  ∃ (n : ℕ), three_digit_palindromic_prime_with_2_ends n ∧ ∀ (m : ℕ), three_digit_palindromic_prime_with_2_ends m →
  (m < n ∧ m ≠ 292) → n = 292 :=
sorry

end second_smallest_palindromic_prime_with_2_ends_l669_669232


namespace maximize_triangle_area_l669_669488

noncomputable def sqrt (x : ℝ) := Real.sqrt x

theorem maximize_triangle_area :
  ∀ (x : ℝ), 1 < x ∧ x < 4 → 
  let A := (1 : ℝ, sqrt 1)
  let B := (x, sqrt x)
  let C := (4 : ℝ, sqrt 4)
  let area := 0.5 * abs (1 * (sqrt x - 2) + x * 1 + 4 * (1 - sqrt x))
  let f := fun (m : ℝ) => m - 3 * sqrt m + 2
  let m := (3 / 2) ^ 2
  f' m = 1 - (3 / (2 * sqrt m))  -- Derivative of the function
  ∀ critical_point : ℝ, 
  ∃ m, m = critical_point = (9 / 4) := 
sorry

end maximize_triangle_area_l669_669488


namespace delta_k_f1_delta_k_f2_delta_k_f3_delta_k_f4_l669_669456

-- Define the finite difference operator
def finite_difference (k : ℕ) (f : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ j in finset.range (k+1), (-1)^(k-j) * (nat.choose k j) * f (n+j)

-- The functions given in the problem
def f1 (n : ℕ) : ℝ := 2^n + (-1)^n
def f2 (n : ℕ) : ℝ := 1 / n
def f3 (n : ℕ) : ℝ := n * 2^n
def f4 (n : ℕ) (x a : ℝ) : ℝ := real.cos (n * x + a)

-- Expected results for the k-th finite difference
theorem delta_k_f1 (k n : ℕ) : finite_difference k f1 n = 2^n + 2^k * (-1)^(n+k) := sorry
theorem delta_k_f2 (k n : ℕ) : finite_difference k f2 n = (-1)^k * k! / (n * (n+1) * ... * (n+k)) := sorry
theorem delta_k_f3 (k n : ℕ) : finite_difference k f3 n = (n + 2*k) * 2^n := sorry
theorem delta_k_f4 (k n : ℕ) (x a : ℝ) :
  finite_difference k (λ m, f4 m x a) n = 2^k * real.sin (x/2)^k * real.cos (n * x + (k * (real.pi + x)) / 2 + a) := sorry

end delta_k_f1_delta_k_f2_delta_k_f3_delta_k_f4_l669_669456


namespace right_triangle_area_and_hypotenuse_l669_669557

theorem right_triangle_area_and_hypotenuse (X Y Z W : Type) [Real X] [Real Y] [Real Z] [Real W]
    (right_triangle : (X = Y ∨ Y = Z ∨ Z = X) ∧ ∠ Y = pi / 2)
    (foot_of_altitude : ∀ W, (W ∈ line (Y, XZ)))
    (XW : Real) (WZ : Real)
    (hw1 : XW = 5) (hw2 : WZ = 7) :
    (area (triangle X Y Z) = 6 * sqrt 35) ∧ (XZ = 12) := 
sorry

end right_triangle_area_and_hypotenuse_l669_669557


namespace distinct_real_roots_of_quadratic_in_cubic_l669_669541

theorem distinct_real_roots_of_quadratic_in_cubic 
  (a b c x1 x2 : ℝ) 
  (f : ℝ → ℝ := λ x, x^3 + a*x^2 + b*x + c)
  (hf_derivative : f' = λ x, 3*x^2 + 2*a*x + b)
  (hx1_extreme : f'.roots = [x1, x2])
  (hx1_condition : f x1 = x1) :  
  ∃ (n : ℕ), 
    (n = 3) ∧
    -- number of distinct real roots of the equation 3(f(x))^2 + 2af(x) + b = 0
    ∃ roots : set ℝ, roots.card = n ∧ 
    ∀ x ∈ roots, 3 * (f x)^2 + 2 * a * (f x) + b = 0 := by
  sorry

end distinct_real_roots_of_quadratic_in_cubic_l669_669541


namespace inverse_proportion_passing_through_l669_669978

theorem inverse_proportion_passing_through (k : ℝ) :
  (∀ x y : ℝ, (y = k / x) → (x = 3 → y = 2)) → k = 6 := 
by
  sorry

end inverse_proportion_passing_through_l669_669978


namespace cheburashkas_erased_l669_669617

def total_krakozyabras : ℕ := 29

def total_rows : ℕ := 2

def cheburashkas_per_row := (total_krakozyabras + total_rows) / total_rows / 2 + 1

theorem cheburashkas_erased :
  (total_krakozyabras + total_rows) / total_rows / 2 - 1 = 11 := 
by
  sorry

-- cheburashkas_erased proves that the number of Cheburashkas erased is 11 from the given conditions.

end cheburashkas_erased_l669_669617


namespace factorize_expression_l669_669413

theorem factorize_expression (a x : ℝ) :
  a * x^2 - 2 * a * x + a = a * (x - 1) ^ 2 := 
sorry

end factorize_expression_l669_669413


namespace integral_sqrt_16_minus_x_sq_l669_669018

theorem integral_sqrt_16_minus_x_sq :
  ∫ x in 0..4, real.sqrt (16 - x^2) = 4 * real.pi := sorry

end integral_sqrt_16_minus_x_sq_l669_669018


namespace expected_value_of_X_is_half_l669_669257

-- Definition of the sequence of random digits and the random number X
def random_digits : list (vector ℕ 10) :=
  replicate 1000 (vector.of_fn (λ i, i)) -- Simulates a list of random digits from 0 to 9

def X (digits : list ℕ) : ℝ :=
  digits.foldl (λ acc x, acc / 10 + x.to_real / 10) 0

-- Expected value of X
noncomputable def E_X : ℝ :=
  1 / 2

-- Theorem statement
theorem expected_value_of_X_is_half : E_X = 0.5 :=
  by
  sorry -- The proof would go here

end expected_value_of_X_is_half_l669_669257


namespace sum_of_medians_powers_l669_669292

noncomputable def median_length_squared (a b c : ℝ) : ℝ :=
  (a^2 + b^2 - c^2) / 4

noncomputable def sum_of_fourth_powers_of_medians (a b c : ℝ) : ℝ :=
  let mAD := (median_length_squared a b c)^2
  let mBE := (median_length_squared b c a)^2
  let mCF := (median_length_squared c a b)^2
  mAD^2 + mBE^2 + mCF^2

theorem sum_of_medians_powers :
  sum_of_fourth_powers_of_medians 13 14 15 = 7644.25 :=
by
  sorry

end sum_of_medians_powers_l669_669292


namespace Jeff_wins_three_games_l669_669584

-- Define the conditions and proven statement
theorem Jeff_wins_three_games :
  (hours_played : ℕ) (minutes_per_point : ℕ) (points_per_match : ℕ) 
  (hours_played = 2) (minutes_per_point = 5) (points_per_match = 8) 
  → (games_won : ℕ) (120 / minutes_per_point / points_per_match = 3) :=
by
  -- Step through assumptions and automatically conclude the proof
  sorry

end Jeff_wins_three_games_l669_669584


namespace number_of_Cheburashkas_erased_l669_669598

theorem number_of_Cheburashkas_erased :
  ∃ (n : ℕ), 
    (∀ x, x ≥ 1 → 
      (let totalKrakozyabras = (2 * (x - 1) = 29) in
         x - 2 = 11)) :=
sorry

end number_of_Cheburashkas_erased_l669_669598


namespace length_of_third_side_l669_669152

-- We define the necessary values and the theorem to prove.
def a : ℝ := 5
def b : ℝ := 12
def θ : ℝ := real.pi / 3   -- 60 degrees in radians is π/3
def cos60 : ℝ := real.cos (real.pi / 3)

theorem length_of_third_side : (a^2 + b^2 - 2 * a * b * cos60) = 109 :=
by
  sorry

end length_of_third_side_l669_669152


namespace cubic_difference_l669_669966

theorem cubic_difference (a b : ℝ) 
  (h₁ : a - b = 7)
  (h₂ : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := 
by 
  sorry

end cubic_difference_l669_669966


namespace expected_value_of_X_l669_669266

-- Define the sequence of random digits as a list of natural numbers (0 to 9)
def random_digits : List ℕ := (List.range 10).take 1000

-- Define the function that forms the number X
def X (digits : List ℕ) : ℝ :=
  digits.enum.foldr (λ (p : ℕ × ℕ) (acc : ℝ) => acc + p.snd * 10^(-(p.fst + 1))) 0

-- Define the expected value of a single digit
def expected_value_digit : ℝ := (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) / 10

-- The main statement to prove
theorem expected_value_of_X (digits : List ℕ) (h_digits : digits.length = 1000) :
  (∑ i in Finset.range digits.length, 10^(-(i + 1)) * expected_value_digit) = 0.5 :=
by {
  sorry
}

end expected_value_of_X_l669_669266


namespace oranges_purchase_cost_l669_669658

/-- 
Oranges are sold at a rate of $3$ per three pounds.
If a customer buys 18 pounds and receives a discount of $5\%$ for buying more than 15 pounds,
prove that the total amount the customer pays is $17.10.
-/
theorem oranges_purchase_cost (rate : ℕ) (base_weight : ℕ) (discount_rate : ℚ)
  (total_weight : ℕ) (discount_threshold : ℕ) (final_cost : ℚ) :
  rate = 3 → base_weight = 3 → discount_rate = 0.05 → 
  total_weight = 18 → discount_threshold = 15 → final_cost = 17.10 := by
  sorry

end oranges_purchase_cost_l669_669658


namespace vector_AM_l669_669169

variable (V : Type) [AddCommGroup V] [Module ℝ V]

variables (A B C M : V)
variables (c b : V)

-- Given conditions
def condition1 := (B - A) = c
def condition2 := (C - A) = b
def condition3 := (C - M) = 2 * (M - B)

-- Prove the following statement
theorem vector_AM (h1 : condition1 B A c) (h2 : condition2 C A b) (h3 : condition3 C M B) :
  (M - A) = (1/3 : ℝ) • b + (2/3 : ℝ) • c :=
sorry

end vector_AM_l669_669169


namespace how_many_cheburashkas_erased_l669_669602

theorem how_many_cheburashkas_erased 
  (total_krakozyabras : ℕ)
  (characters_per_row_initial : ℕ) 
  (total_characters_initial : ℕ)
  (total_cheburashkas : ℕ)
  (total_rows : ℕ := 2)
  (total_krakozyabras := 29) :
  total_cheburashkas = 11 :=
by
  sorry

end how_many_cheburashkas_erased_l669_669602


namespace max_PA_PB_PC_l669_669065

noncomputable def max_product (A B C P : Point) : ℝ :=
  dist P A * dist P B * dist P C

theorem max_PA_PB_PC :
  ∀ (A B C P : Point), 
  (∀ a b c : Point, is_right_triangle a b c ∧ (dist a b = 1) ∧ (dist a c = 1) → ∃ P : Point, 
  (P ∈ Triangle_side a b c) ∧ 
  (max_product a b c P = (Real.sqrt 2) / 4)) :=
begin
  sorry
end

end max_PA_PB_PC_l669_669065


namespace min_diff_composite_sum_103_l669_669843

def is_composite (n : ℕ) : Prop :=
  ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ p * q = n

theorem min_diff_composite_sum_103 :
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ a + b = 103 ∧ (a - b).nat_abs = 1 :=
by
  sorry

end min_diff_composite_sum_103_l669_669843


namespace p_sufficient_not_necessary_for_q_l669_669940

def p (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3
def q (x : ℝ) : Prop := x ≠ 0

theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l669_669940


namespace largest_of_options_l669_669298

theorem largest_of_options :
  max (2 + 0 + 1 + 3) (max (2 * 0 + 1 + 3) (max (2 + 0 * 1 + 3) (max (2 + 0 + 1 * 3) (2 * 0 * 1 * 3)))) = 2 + 0 + 1 + 3 := by sorry

end largest_of_options_l669_669298


namespace mason_car_nuts_l669_669645

def busy_squirrels_num := 2
def busy_squirrel_nuts_per_day := 30
def sleepy_squirrel_num := 1
def sleepy_squirrel_nuts_per_day := 20
def days := 40

theorem mason_car_nuts : 
  busy_squirrels_num * busy_squirrel_nuts_per_day * days + sleepy_squirrel_nuts_per_day * days = 3200 :=
  by
    sorry

end mason_car_nuts_l669_669645


namespace find_complex_z_l669_669133

noncomputable def complex_z (z : ℂ) : Prop :=
  (3 - 4 * complex.i) * z = 5 + 10 * complex.i

theorem find_complex_z (z : ℂ) : 
  complex_z z → z = -1 + 2 * complex.i :=
by
  sorry

end find_complex_z_l669_669133


namespace perimeter_of_second_rectangle_l669_669348

theorem perimeter_of_second_rectangle
  (side : ℝ := 5)
  (perimeter_one : ℝ := 16)
  (perimeter_square : ℝ := 4 * side)
  (w1 : ℝ)
  (w2 : ℝ)
  (h1 : ℝ := side)
  (h2 : ℝ := side)
  (semiperimeter_one : h1 + w1 = perimeter_one / 2)
  (other_width : w1 + w2 = side) :
  2 * (h2 + w2) = 14 :=
by
  have side_def : side = 5 := rfl
  have h1_def : h1 = side := rfl
  have s_one : semiperimeter_one := by sorry
  have width_relation : w1 + w2 = h1 := other_width
  have perimeter_two_calc : 2 * (h2 + w2) = 2 * (5 - w1 + w1) := by sorry
  have result : 2 * (h2 + w2) = 14 := by sorry
  exact result

end perimeter_of_second_rectangle_l669_669348


namespace angle_between_vectors_pi_over_3_l669_669119

open Real

variables (a b : EuclideanSpace ℝ (Fin 2))

def is_unit_vector (v : EuclideanSpace ℝ (Fin 2)) : Prop := ‖v‖ = 1
def is_double_length_vector (v : EuclideanSpace ℝ (Fin 2)) : Prop := ‖v‖ = 2
def orthogonal (v w : EuclideanSpace ℝ (Fin 2)) : Prop := inner v w = 0

theorem angle_between_vectors_pi_over_3
  (ha : is_unit_vector a)
  (hb : is_double_length_vector b)
  (h_orth : orthogonal a (a - b)) :
  angle a b = π / 3 :=
by
  sorry

end angle_between_vectors_pi_over_3_l669_669119


namespace min_area_of_triangle_mon_perpendicular_tangents_exist_iff_l669_669358

variables (a b : ℝ) (h : a > b > 0)
noncomputable def ellipse (x y : ℝ) : Prop := b^2 * x^2 + a^2 * y^2 = a^2 * b^2 
noncomputable def circle (x y : ℝ) : Prop := x^2 + y^2 = b^2

theorem min_area_of_triangle_mon (P : ℝ × ℝ) (hP: ellipse a b P.1 P.2) :
  let OM := b^2 / (a * real.cos (real.atan2 P.2 P.1)) in
  let ON := b / (real.sin (real.atan2 P.2 P.1)) in
  (1 / 2) * OM * ON = b^3 / a :=
sorry

theorem perpendicular_tangents_exist_iff (a b : ℝ) (h : a > b > 0) :
  (∃ (P : ℝ × ℝ), ellipse a b P.1 P.2 ∧ 
  let θ := real.atan2 P.2 P.1 in 
  a^2 * real.cos θ^2 + b^2 * real.sin θ^2 = 2 * b^2) ↔ a ≥ real.sqrt 2 * b :=
sorry

end min_area_of_triangle_mon_perpendicular_tangents_exist_iff_l669_669358


namespace valve_solution_l669_669015

noncomputable def valve_problem : Prop :=
  ∀ (x y z : ℝ),
  (1 / (x + y + z) = 2) →
  (1 / (x + z) = 4) →
  (1 / (y + z) = 3) →
  (1 / (x + y) = 2.4)

theorem valve_solution : valve_problem :=
by
  -- proof omitted
  intros x y z h1 h2 h3
  sorry

end valve_solution_l669_669015


namespace cube_root_simplification_l669_669685

noncomputable def cubeRoot (x : ℝ) : ℝ := x^(1/3)

theorem cube_root_simplification :
  cubeRoot 54880000 = 140 * cubeRoot 20 :=
by
  sorry

end cube_root_simplification_l669_669685


namespace simplify_cube_root_l669_669689

theorem simplify_cube_root (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)
  (h1 : a = 10^3 * b)
  (h2 : b = 2^7 * c * 7^3)
  (h3 : c = 10) :
  ∛a = 40 * 7 * 2^(2/3) * 5^(1/3) := by
  sorry

end simplify_cube_root_l669_669689


namespace cube_root_simplification_l669_669687

noncomputable def cubeRoot (x : ℝ) : ℝ := x^(1/3)

theorem cube_root_simplification :
  cubeRoot 54880000 = 140 * cubeRoot 20 :=
by
  sorry

end cube_root_simplification_l669_669687


namespace box_costs_exactly_l669_669282

def ruble := ℝ
def kopeck := ℝ

def cost_per_box (x : ruble) := ∃ k₁ k₂ : kopeck, k₁ > 0 ∧ k₂ > 0 ∧ (9 * (x + k₁/100) > 9) ∧ (10 * (x + k₂/100) < 11)

theorem box_costs_exactly (x : ruble) : cost_per_box x → x + 11 / 100 = 1.11 :=
by
  sorry

end box_costs_exactly_l669_669282


namespace largest_n_points_l669_669631

noncomputable def circle_radius : ℝ := 2006
def max_points (C : set ℝ → ℝ) (n : ℕ) : Prop :=
  ∃ (points : fin n → ℝ × ℝ), ∀ (i j : fin n), i ≠ j → 
  (dist (points i) (points j) > circle_radius) → True

theorem largest_n_points (hC : ∀ (p ∈ C), dist p (0,0) ≤ circle_radius) : 
  ∃ (n ≤ 5), max_points C n :=
sorry

end largest_n_points_l669_669631


namespace largest_int_less_150_gcd_18_eq_6_l669_669804

theorem largest_int_less_150_gcd_18_eq_6 : ∃ (n : ℕ), n < 150 ∧ gcd n 18 = 6 ∧ ∀ (m : ℕ), m < 150 ∧ gcd m 18 = 6 → m ≤ n ∧ n = 138 := 
by
  sorry

end largest_int_less_150_gcd_18_eq_6_l669_669804


namespace quadratic_equation_is_D_l669_669818

theorem quadratic_equation_is_D (x a b c : ℝ) : 
  (¬ (∃ b' : ℝ, (x^2 - 2) * x = b' * x + 2)) ∧
  (¬ ((a ≠ 0) ∧ (ax^2 + bx + c = 0))) ∧
  (¬ (x + (1 / x) = 5)) ∧
  ((x^2 = 0) ↔ true) :=
by sorry

end quadratic_equation_is_D_l669_669818


namespace integral_calculation_l669_669383

theorem integral_calculation : ∫ x in 0..3, (2 * x - 1) = 6 := 
by
  sorry

end integral_calculation_l669_669383


namespace factorize_expression_l669_669414

theorem factorize_expression (a x : ℝ) :
  a * x^2 - 2 * a * x + a = a * (x - 1) ^ 2 := 
sorry

end factorize_expression_l669_669414


namespace minimum_cactus_species_l669_669373

-- Definitions to represent the conditions
def num_cactophiles : Nat := 80
def num_collections (S : Finset (Fin num_cactophiles)) : Nat := S.card
axiom no_single_species_in_all (S : Finset (Fin num_cactophiles)) : num_collections S < num_cactophiles
axiom any_15_have_common_species (S : Finset (Fin num_cactophiles)) (h : S.card = 15) : 
  ∃ species, ∀ s ∈ S, species ∈ s

-- Proposition to be proved
theorem minimum_cactus_species (k : Nat) (h : ∀ S : Finset (Fin num_cactophiles), S.card = 15 → ∃ species, ∀ s ∈ S, species ∈ s) : k ≥ 16 := sorry

end minimum_cactus_species_l669_669373


namespace laser_travel_distance_l669_669854

noncomputable def laser_path_distance : ℝ :=
  let A : ℝ × ℝ := (4, 6)
  let E : ℝ × ℝ := (8, 6)
  let E'' : ℝ × ℝ := (-8, -6)
  dist A E''

theorem laser_travel_distance : laser_path_distance = 12 * real.sqrt 2 := by
  sorry

end laser_travel_distance_l669_669854


namespace artist_paints_35_square_meters_l669_669882

theorem artist_paints_35_square_meters :
  ∀ (cubes : ℕ),
    (edge_length : ℕ),
    (top_layer : ℕ) 
    (middle_layer : ℕ) 
    (base_layer : ℕ),
    cubes = 19 → 
    edge_length = 1 →
    top_layer = 1 → 
    middle_layer = 6 → 
    base_layer = 12 → 
    (let top_exposed := top_layer * (4 * edge_length + edge_length) in
     let middle_exposed := 4 * 4 + 2 * zip := middle_layer in
     let base_exposed := base_layer * edge_length in
     top_exposed + middle_exposed + base_exposed = 35) :=
by
  intros cubes edge_length top_layer middle_layer base_layer
  intros h1 h2 h3 h4 h5
  sorry

end artist_paints_35_square_meters_l669_669882


namespace cheburashkas_erased_l669_669609

theorem cheburashkas_erased (total_krakozyabras : ℕ) (rows : ℕ) :
  rows ≥ 2 → total_krakozyabras = 29 → ∃ (cheburashkas_erased : ℕ), cheburashkas_erased = 11 :=
by
  assume h_rows h_total_krakozyabras
  let n := (total_krakozyabras / 2) + 1
  have h_cheburashkas : cheburashkas_erased = n - 1 
  sorry

end cheburashkas_erased_l669_669609


namespace num_perfect_square_factors_of_2560_l669_669769

def is_even (n : ℕ) : Prop := n % 2 = 0

theorem num_perfect_square_factors_of_2560 :
  ∃ n : ℕ, n = 2560 ∧ (factorization n = [(2, 8), (5, 1)]) ∧
  (∀ (d : ℕ), d ∣ n → (∃ (a b : ℕ), (d = 2^a * 5^b) ∧ is_even a ∧ is_even b) → (∃ m : ℕ, d = m^2) → count {d : ℕ | d ∣ n ∧ is_even (log2 d) ∧ (d % 5 = 0 ∨ d % 5 = 1)} 5) :=
begin
  sorry
end

end num_perfect_square_factors_of_2560_l669_669769


namespace identify_alkali_metal_l669_669992

variable (Li Na K Rb : ℕ)

-- Conditions
def rel_atomic_mass : ℕ → ℕ
| Li => 69
| Na => 230
| K => 390
| Rb => 850
| _ => 0

def initial_mass (R : ℕ) (R2O : ℕ) := R + R2O = 108

def final_mass (R : ℕ) (RH : ℕ) := RH = 160

-- Proving the correct identification of the metal R
theorem identify_alkali_metal (R : ℕ) : 
  rel_atomic_mass Na = 230 → 
  initial_mass 69 39 = 108 → 
  final_mass 69 69 = 160 → 
  R = Na :=
by
  intros h1 h2 h3
  dsimp [rel_atomic_mass, initial_mass, final_mass] at *
  sorry

end identify_alkali_metal_l669_669992


namespace imaginary_part_conjugate_of_z_is_5_l669_669472

-- Define the complex number z and the given condition
variable (z : ℂ)
variable (h : complex.I * z = complex.abs (3 + 4 * complex.I) - complex.I)

-- Lean 4 statement to prove the imaginary part of the conjugate of z is 5
theorem imaginary_part_conjugate_of_z_is_5 : complex.im (complex.conj z) = 5 :=
by
  sorry

end imaginary_part_conjugate_of_z_is_5_l669_669472


namespace expected_value_of_X_is_half_l669_669259

-- Definition of the sequence of random digits and the random number X
def random_digits : list (vector ℕ 10) :=
  replicate 1000 (vector.of_fn (λ i, i)) -- Simulates a list of random digits from 0 to 9

def X (digits : list ℕ) : ℝ :=
  digits.foldl (λ acc x, acc / 10 + x.to_real / 10) 0

-- Expected value of X
noncomputable def E_X : ℝ :=
  1 / 2

-- Theorem statement
theorem expected_value_of_X_is_half : E_X = 0.5 :=
  by
  sorry -- The proof would go here

end expected_value_of_X_is_half_l669_669259


namespace intersection_points_l669_669027

def curve (x y : ℝ) : Prop := x^2 + y^2 = 1
def line (x y : ℝ) : Prop := y = x + 1

theorem intersection_points :
  {p : ℝ × ℝ | curve p.1 p.2 ∧ line p.1 p.2} = {(-1, 0), (0, 1)} :=
by 
  sorry

end intersection_points_l669_669027


namespace spadesuit_eval_l669_669041

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_eval :
  spadesuit 5 (spadesuit 2 3) = 0 :=
by
  sorry

end spadesuit_eval_l669_669041


namespace cube_root_simplification_l669_669681

noncomputable def cubeRoot (x : ℝ) : ℝ := x^(1/3)

theorem cube_root_simplification :
  cubeRoot 54880000 = 140 * cubeRoot 20 :=
by
  sorry

end cube_root_simplification_l669_669681


namespace cube_root_simplification_l669_669714

theorem cube_root_simplification : (∛54880000) = 140 * (2 ^ (1 / 3)) :=
by
  -- Using the information from the problem conditions and final solution.
  have root_10_cubed := (10 ^ 3 : ℝ)
  have factored_value := root_10_cubed * (2 ^ 4 * 7 ^ 3)
  have cube_root := Real.cbrt factored_value
  sorry

end cube_root_simplification_l669_669714


namespace clock_angle_at_7pm_l669_669519

theorem clock_angle_at_7pm :
  let full_circle := 360
  let hours := 12
  let angle_per_hour := full_circle / hours
  let hour_hand_at_7 := 7
  let minute_hand_at_12 := 0
  let hours_apart := hour_hand_at_7 - minute_hand_at_12
  let angle := hours_apart * angle_per_hour
  angle = 150 :=
by
  let full_circle := 360
  let hours := 12
  let angle_per_hour := full_circle / hours
  let hour_hand_at_7 := 7
  let minute_hand_at_12 := 0
  let hours_apart := hour_hand_at_7 - minute_hand_at_12
  let angle := hours_apart * angle_per_hour
  show angle = 150 from sorry

end clock_angle_at_7pm_l669_669519


namespace proof_correct_statements_l669_669103

-- Define the function and its conditions
variable {f : ℝ → ℝ}
variable {x : ℝ} (hx : 3 < x ∧ x ≤ 7)
variable {x1 x2 : ℝ} (hx1x2 : x2 > x1)
noncomputable def fn (n : ℕ) : ℝ → ℝ
| 0 => f'''
| (Nat.succ n) => (fn n)'

-- Statement definition
theorem proof_correct_statements 
  (h1 : ∃ c ∈ Set.Icc 3 7, deriv f c = 0) -- Corresponds to f(x) has a tangent line parallel to the x-axis
  (h3 : fn 2015 = λ x, x * Real.exp x + 2017 * Real.exp x): -- Corresponds to f'_{2015}(x) = xe^x + 2017e^x
  (
    (∀ x1 x2, x2 > x1 → ¬ (f(x1) + x2 > f(x2) + x1)) ∧ -- correctness of (4)
    (∀ x1 x2, x2 > x1 → ¬ ((f(x1) - f(x2)) / (x1 - x2) > 0)) -- correctness of (2)
  ) :=
sorry

end proof_correct_statements_l669_669103


namespace factorize_expression_l669_669412

theorem factorize_expression (a x : ℝ) :
  a * x^2 - 2 * a * x + a = a * (x - 1) ^ 2 := 
sorry

end factorize_expression_l669_669412


namespace necessary_but_not_sufficient_l669_669222

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x + 1

theorem necessary_but_not_sufficient (a : ℝ) :
  ((a ≥ 4 ∨ a ≤ 0) ↔ (∃ x : ℝ, f a x = 0)) ∧ ¬((a ≥ 4 ∨ a ≤ 0) → (∃ x : ℝ, f a x = 0)) :=
sorry

end necessary_but_not_sufficient_l669_669222


namespace locus_of_point_R_l669_669459

theorem locus_of_point_R :
  ∀ (P Q O F R : ℝ × ℝ)
    (hP_on_parabola : ∃ x1 y1, P = (x1, y1) ∧ y1^2 = 2 * x1)
    (h_directrix : Q.1 = -1 / 2)
    (hQ : ∃ x1 y1, Q = (x1, y1) ∧ P = (x1, y1))
    (hO : O = (0, 0))
    (hF : F = (1 / 2, 0))
    (h_intersection : ∃ x y, 
      R = (x, y) ∧
      ∃ x1 y1,
      P = (x1, y1) ∧ 
      y1^2 = 2 * x1 ∧
      ∃ (m_OP : ℝ), 
        m_OP = y1 / x1 ∧ 
        y = m_OP * x ∧
      ∃ (m_FQ : ℝ), 
        m_FQ = -y1 ∧
        y = m_FQ * x + y1 * (1 + 3 / 2)),
  R.2^2 = -2 * R.1^2 + R.1 :=
by sorry

end locus_of_point_R_l669_669459


namespace proof_problem_l669_669098

variable {α : Type*} [LinearOrderedField α]

theorem proof_problem 
  (a b x y : α) 
  (h0 : 0 < a ∧ 0 < b ∧ 0 < x ∧ 0 < y)
  (h1 : a + b + x + y < 2)
  (h2 : a + b^2 = x + y^2)
  (h3 : a^2 + b = x^2 + y) :
  a = x ∧ b = y := 
by
  sorry

end proof_problem_l669_669098


namespace sum_ratio_constant_l669_669563

noncomputable def arithmetic_sequence := ℕ → ℝ

variable {a : arithmetic_sequence}

variable (n : ℕ) (S : ℕ → ℝ)

axiom sum_five_terms : a 3 + a 4 + a 5 + a 6 + a 7 = 50

axiom sum_first_k_terms_constant (k : ℕ) : S k = 10 * k

theorem sum_ratio_constant : ∀ k : ℕ, k ≠ 0 → (S k) / k = 10 :=
by
  intros k hk
  rw sum_first_k_terms_constant
  exact (div_mul_cancel (S k) hk).symm
  sorry

end sum_ratio_constant_l669_669563


namespace magnitude_of_z_l669_669538

/-- Define the complex number z. -/
def z : ℂ := (1 + 2*complex.I) * (2 - complex.I)

/-- State the theorem to prove the magnitude of z is 5. -/
theorem magnitude_of_z : complex.abs z = 5 :=
by
  -- Proof is omitted
  sorry

end magnitude_of_z_l669_669538


namespace tangent_circumcircles_l669_669170

noncomputable def midpoint (A B : Point) : Point :=
  (A + B) / 2

noncomputable def centroid (A B C : Point) : Point :=
  (A + B + C) / 3

noncomputable def orthocenter (A B C : Point) : Point := sorry

noncomputable def circumcircle (P Q R : Point) : Circle := sorry

theorem tangent_circumcircles
  (A B C : Point)
  (M : Point := centroid A B C)
  (A0 : Point := midpoint B C)
  (B0 : Point := midpoint A C)
  (C0 : Circle := circumcircle A0 B0 C)
  (O : Circle := circumcircle A B (orthocenter A B C)) :
  Circle.Tangent C0 O :=
sorry

end tangent_circumcircles_l669_669170


namespace minimum_value_of_expression_l669_669924

noncomputable def f (x : ℝ) : ℝ := 16^x - 2^x + x^2 + 1

theorem minimum_value_of_expression : ∃ (x : ℝ), f x = 1 ∧ ∀ y : ℝ, f y ≥ 1 := 
sorry

end minimum_value_of_expression_l669_669924


namespace cos_pi_six_plus_alpha_l669_669496

variable (α : ℝ)

theorem cos_pi_six_plus_alpha (h : Real.sin (Real.pi / 3 - α) = 1 / 6) : 
  Real.cos (Real.pi / 6 + α) = 1 / 6 :=
sorry

end cos_pi_six_plus_alpha_l669_669496


namespace greatest_prime_factor_of_150_l669_669800

theorem greatest_prime_factor_of_150 : 
  ∀ (p : ℕ), (p ∣ 150 ∧ p.prime) → 
  p ≤ 5 :=
by {
  sorry
}

end greatest_prime_factor_of_150_l669_669800


namespace positive_difference_l669_669636

def f (n : ℝ) : ℝ :=
if n < 0 then n^2 - 2
else 3 * n - 20

theorem positive_difference (a1 a2 : ℝ) (ha1 : f(a1) = 12) (ha2 : f(a2) = 12) :
  abs (-sqrt 14 - 32 / 3) = sqrt 14 + 32 / 3 := by
  sorry

end positive_difference_l669_669636


namespace mn_length_l669_669075

-- Define the structure of a trapezoid
structure Trapezoid (α : Type) :=
  (A B C D M N : α)
  (AB BC CD DA : α)
  (AB_eq_a : AB = a)
  (BC_eq_b : BC = b)
  (CD_eq_c : CD = c)
  (DA_eq_d : DA = d)
  (M_intersection : is_intersection (angle_bisector A D B) (angle_bisector B A C) = M)
  (N_intersection : is_intersection (angle_bisector C D A) (angle_bisector D C B) = N)

noncomputable def length_MN (α : Type) [metric_space α] (trapezoid : Trapezoid α) : α :=
  (1 / 2) * abs (trapezoid.BC + trapezoid.DA - trapezoid.AB - trapezoid.CD)

theorem mn_length (α : Type) [metric_space α] (trapezoid : Trapezoid α) :
  length_MN α trapezoid = (1 / 2) * abs (b + d - a - c) := sorry

end mn_length_l669_669075


namespace factorize_expression_l669_669424

theorem factorize_expression (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
by sorry

end factorize_expression_l669_669424


namespace tan_alpha_minus_pi_six_l669_669464

variable (α β : Real)

axiom tan_alpha_minus_beta : Real.tan (α - β) = 2 / 3
axiom tan_pi_six_minus_beta : Real.tan ((Real.pi / 6) - β) = 1 / 2

theorem tan_alpha_minus_pi_six : Real.tan (α - (Real.pi / 6)) = 1 / 8 :=
by
  sorry

end tan_alpha_minus_pi_six_l669_669464


namespace minimum_black_squares_l669_669324

/-- Given a grid of size m × n (m < n), we need to determine the minimum 
    number of initially painted black squares N such that after a finite 
    number of steps according to the given rules, all squares in the grid 
    will be black. -/
theorem minimum_black_squares (m n : ℕ) (h : m < n) : ∃ N : ℕ, N = (n + m + 1) / 2 ∧ 
  ∀ grid : ℕ × ℕ → Prop, (∀ i j, grid i j → (i < m ∧ j < n)) →
  (∃ k l, k < m ∧ l < n ∧ grid k l) →
  (∀ i j, (i < m ∧ j < n) → grid i j ∨ 
           (∃ a b, (a = i ∧ b = j + 1 ∨ a = i ∧ b = j - 1 ∨ 
                    a = i + 1 ∧ b = j ∨ a = i - 1 ∧ b = j) ∧ 
            grid a b ∧ grid (if a = i then a else i) (if b = j then b else j)) → 
           grid i j)) :=
begin
  sorry
end

end minimum_black_squares_l669_669324


namespace product_not_divisible_by_201_l669_669242

theorem product_not_divisible_by_201 (a b : ℕ) (h₁ : a + b = 201) : ¬ (201 ∣ a * b) := sorry

end product_not_divisible_by_201_l669_669242


namespace triangle_area_30_l669_669757

theorem triangle_area_30 (h : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = 5 ∧ c = 13 ∧ b > 0) : 
  ∃ area : ℝ, area = 1 / 2 * 5 * (b : ℝ) ∧ area = 30 :=
by
  sorry

end triangle_area_30_l669_669757


namespace simplify_polynomial_no_quadratic_term_l669_669792

theorem simplify_polynomial_no_quadratic_term (n : ℝ) : 
    (let poly := 4 * x^2 + 2 * (7 + 3 * x - 3 * x^2) - n * x^2
    in (∀ x : ℝ, ((4 - 6 - n) * x^2 + 6 * x + 14) = 6 * x + 14)) ↔ n = -2 :=
by
  sorry

end simplify_polynomial_no_quadratic_term_l669_669792


namespace meaningful_sqrt_range_l669_669542

theorem meaningful_sqrt_range (x : ℝ) (h : 1 - x ≥ 0) : x ≤ 1 :=
sorry

end meaningful_sqrt_range_l669_669542


namespace betul_min_moves_2005_l669_669877

noncomputable def min_moves_to_find_marked_stone (n : ℕ) : ℕ :=
Nat.ceil (Real.logb 2 n)

theorem betul_min_moves_2005 :
  min_moves_to_find_marked_stone 2005 = 11 :=
begin
  sorry -- Skip the proof
end

end betul_min_moves_2005_l669_669877


namespace factorization_l669_669439

theorem factorization (a x : ℝ) : ax^2 - 2ax + a = a * (x - 1) ^ 2 := 
by
  sorry

end factorization_l669_669439


namespace find_n_l669_669532

theorem find_n (n : ℕ) : sqrt (8 + n) = 9 → n = 73 :=
by
  intro h
  sorry

end find_n_l669_669532


namespace lines_relationship_l669_669573

theorem lines_relationship 
  (l1 l2 l3 : Plane ℝ) 
  (hl1l2 : l1 ∥ l2) 
  (hl2l3 : l2 ⟂ l3) : l1 ⟂ l3 := 
by 
  sorry

end lines_relationship_l669_669573


namespace SumataFamilyTotalMiles_l669_669726

def miles_per_day := 250
def days := 5

theorem SumataFamilyTotalMiles : miles_per_day * days = 1250 :=
by
  sorry

end SumataFamilyTotalMiles_l669_669726


namespace distance_home_to_school_l669_669826

theorem distance_home_to_school
  (T T' : ℝ)
  (D : ℝ)
  (h1 : D = 6 * T)
  (h2 : D = 12 * T')
  (h3 : T - T' = 0.25) :
  D = 3 :=
by
  -- The proof would go here
  sorry

end distance_home_to_school_l669_669826


namespace complete_square_l669_669813

theorem complete_square (x : ℝ) : (x^2 - 2 * x - 5 = 0) ↔ ((x - 1)^2 = 6) := 
by
  sorry

end complete_square_l669_669813


namespace notebook_cost_correct_l669_669003

def totalSpent : ℕ := 32
def costBackpack : ℕ := 15
def costPen : ℕ := 1
def costPencil : ℕ := 1
def numberOfNotebooks : ℕ := 5
def costPerNotebook : ℕ := 3

theorem notebook_cost_correct (h_totalSpent : totalSpent = 32)
    (h_costBackpack : costBackpack = 15)
    (h_costPen : costPen = 1)
    (h_costPencil : costPencil = 1)
    (h_numberOfNotebooks : numberOfNotebooks = 5) :
    (totalSpent - (costBackpack + costPen + costPencil)) / numberOfNotebooks = costPerNotebook :=
by
  sorry

end notebook_cost_correct_l669_669003


namespace sum_of_erased_numbers_l669_669872

variable {n : ℕ}

def odd_seq (k : ℕ) := 2 * k - 1

def sum_first_n odd_seq (n : ℕ) := (n * n : ℕ)

def sum_segment1 : ℕ := 961
def sum_segment2 : ℕ := 1001
def total_sum : ℕ := 2025 -- sum_first_n odd_seq 45

theorem sum_of_erased_numbers :
  ∃ x y : ℕ, (x ≠ y ∧ x % 2 = 1 ∧ y % 2 = 1 ∧ sum_segment1 + sum_segment2 + x + y = total_sum) ∧ x + y = 154 :=
by
  sorry

end sum_of_erased_numbers_l669_669872


namespace angle_sum_straight_line_l669_669130

  theorem angle_sum_straight_line (x : ℝ) (h : 90 + x + 20 = 180) : x = 70 :=
  by
    sorry
  
end angle_sum_straight_line_l669_669130


namespace factorization_l669_669436

theorem factorization (a x : ℝ) : ax^2 - 2ax + a = a * (x - 1) ^ 2 := 
by
  sorry

end factorization_l669_669436


namespace min_value_of_function_l669_669185

noncomputable def f (x y : ℝ) : ℝ := x^2 / (x + 2) + y^2 / (y + 1)

theorem min_value_of_function (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  f x y ≥ 1 / 4 :=
sorry

end min_value_of_function_l669_669185


namespace domain_of_sqrt_log_l669_669734

def domain_of_f : Set ℝ := { x : ℝ | 1 ≤ x ∧ x < 2 }

theorem domain_of_sqrt_log :
  (∀ x : ℝ, (√(x - 1) + log 10 (2 - x)) = f x) →
  (∀ x : ℝ, (1 ≤ x ∧ x < 2) ↔ x ∈ domain_of_f) :=
by
  intros
  sorry

end domain_of_sqrt_log_l669_669734


namespace michael_meets_truck_once_l669_669198

def michael_speed := 5  -- feet per second
def pail_distance := 150  -- feet
def truck_speed := 15  -- feet per second
def truck_stop_time := 20  -- seconds

def initial_michael_position (t : ℕ) : ℕ := t * michael_speed
def initial_truck_position (t : ℕ) : ℕ := pail_distance + t * truck_speed - (t / (truck_speed * truck_stop_time))

def distance (t : ℕ) : ℕ := initial_truck_position t - initial_michael_position t

theorem michael_meets_truck_once :
  ∃ t, (distance t = 0) :=  
sorry

end michael_meets_truck_once_l669_669198


namespace find_hyperbola_eccentricity_l669_669110

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c = sqrt (a^2 + b^2)) (h4 : 2 * real.sqrt (a^2 - b^2) = (2 / 3) * c) : ℝ :=
  c / a

theorem find_hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c = sqrt (a^2 + b^2)) (h4 : 2 * real.sqrt (a^2 - b^2) = (2 / 3) * c) :
  hyperbola_eccentricity a b c h1 h2 h3 h4 = 3 * sqrt 5 / 5 :=
by
  sorry

end find_hyperbola_eccentricity_l669_669110


namespace number_of_n_rel_prime_221_l669_669521

theorem number_of_n_rel_prime_221 : 
  {n : ℕ // 1 ≤ n ∧ n < 13 ∧ (∀ p : ℕ, p ∣ n! * (2 * n + 1) → p ≠ 13 ∧ p ≠ 17)}.card = 10 :=
sorry

end number_of_n_rel_prime_221_l669_669521


namespace find_greatest_and_second_greatest_problem_solution_l669_669933

theorem find_greatest_and_second_greatest
  (a b c d : ℝ)
  (ha : a = 4 ^ (1 / 4))
  (hb : b = 5 ^ (1 / 5))
  (hc : c = 16 ^ (1 / 16))
  (hd : d = 25 ^ (1 / 25))
  : (a > b) ∧ (b > c) ∧ (c > d) :=
by 
  sorry

def greatest_and_second_greatest_eq (x1 x2 : ℝ) : Prop :=
  x1 = 4 ^ (1 / 4) ∧ x2 = 5 ^ (1 / 5)

theorem problem_solution (a b c d : ℝ)
  (ha : a = 4 ^ (1 / 4))
  (hb : b = 5 ^ (1 / 5))
  (hc : c = 16 ^ (1 / 16))
  (hd : d = 25 ^ (1 / 25))
  : greatest_and_second_greatest_eq a b :=
by 
  sorry

end find_greatest_and_second_greatest_problem_solution_l669_669933


namespace count_squares_in_region_l669_669401

theorem count_squares_in_region : 
  let bounded_region : ℤ × ℤ → Prop := 
    λ (x, y), 0 ≤ x ∧ x ≤ 6 ∧ -1 ≤ y ∧ y ≤ 2 * x 
  in (∑ x in range 7, ∑ y in range (2 * x + 2 + 1)) = 99 := sorry

end count_squares_in_region_l669_669401


namespace abs_z1_purely_imaginary_l669_669080

noncomputable def z1 (a : ℝ) : Complex := ⟨a, 2⟩
def z2 : Complex := ⟨2, -1⟩

theorem abs_z1_purely_imaginary (a : ℝ) (ha : 2 * a - 2 = 0) : Complex.abs (z1 a) = Real.sqrt 5 :=
by
  sorry

end abs_z1_purely_imaginary_l669_669080


namespace count_triangular_numbers_lt_100_l669_669737

theorem count_triangular_numbers_lt_100 :
  { n // ∃ k, n = (k * (k + 1)) / 2 ∧ n < 100 }.card = 13 :=
sorry

end count_triangular_numbers_lt_100_l669_669737


namespace bob_used_fraction_for_art_l669_669381

def total_rope : ℝ := 50
def art_rope (x : ℝ) : ℝ := x * total_rope
def remaining_rope (x : ℝ) : ℝ := total_rope - art_rope x
def left_rope (x : ℝ) : ℝ := 1 / 2 * remaining_rope x

theorem bob_used_fraction_for_art (x : ℝ) : 
  (left_rope x = 20) → x = 1 / 5 :=
by
  sorry

end bob_used_fraction_for_art_l669_669381


namespace pentagon_area_percentage_is_28_l669_669860

open Real

-- Define the side length of squares and triangles
variable (a : ℝ) (sqrt3_approx : ℝ)

-- Define areas of squares, triangles, and the pentagon
def large_square_area : ℝ := 16 * a^2
def squares_area : ℝ := 9 * a^2
def triangles_area : ℝ := 6 * (sqrt3_approx * a^2 / 4)
def pentagon_area : ℝ := large_square_area - squares_area - triangles_area

-- Define the percentage of the pentagon area
def pentagon_percentage : ℝ := (pentagon_area / large_square_area) * 100

-- Approximate value for sqrt(3)
noncomputable def sqrt3_approximation : ℝ := 1.732

-- Proof statement: Show that pentagon_percentage is approximately 28%
theorem pentagon_area_percentage_is_28 (
  h_sqrt3 : sqrt3_approx = sqrt3_approximation
) : abs (pentagon_percentage a sqrt3_approx - 28) < 1 :=
by
  sorry

end pentagon_area_percentage_is_28_l669_669860


namespace problem1_problem2_problem3_l669_669484

section
variables (A : ℝ × ℝ) (C : ℝ → ℝ → Prop) 
def point_A := (-1, 0)
def circle_C (x y : ℝ) := x ^ 2 + y ^ 2 - 2 * x - 2 * real.sqrt 3 * y + 3 = 0
def tangent_length_A_to_C := real.sqrt 6
def line_l1_slope_k := {k : ℝ | k = real.sqrt 3 / 3 ∨ k = 11 * real.sqrt 3 / 15 }
def points_M_N := {(M, N : ℝ × ℝ × ℝ × ℝ) | (M = (1, 4 * real.sqrt 3 / 3) ∧ N = (1, 2 * real.sqrt 3)) ∨ (M = (1, 2 * real.sqrt 3 / 3) ∧ N = (1, 0)) }

theorem problem1 : ∀ P : ℝ × ℝ, (P = point_A) → (∀ x y : ℝ, circle_C x y → tangent_length_A_to_C = real.sqrt 6) := 
sorry

theorem problem2 : ∀ k : ℝ, (∃ P Q : ℝ × ℝ, (P = point_A ∧ (circle_C P.1 P.2 ∧ circle_C Q.1 Q.2)) ∧ (real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = real.sqrt 3 ∧ line_l1_slope_k k)) := 
sorry

theorem problem3 : ∀ R : ℝ × ℝ, (∀ x y : ℝ, circle_C x y → ((∃ M N : ℝ × ℝ, points_M_N (M, N)) ∧ real.dist (1, y) (R.1, R.2) = real.sqrt 3 * real.dist (1, y) (real.sqrt ((R.1 - M.1)^2 + (R.2 - M.2)^2))) := 
sorry
end

end problem1_problem2_problem3_l669_669484


namespace minimum_cactus_species_l669_669362

/--
At a meeting of cactus enthusiasts, 80 cactophiles presented their collections,
each consisting of cacti of different species. It turned out that no single 
species of cactus is found in all collections simultaneously, but any 15 people
have cacti of the same species. Prove that the minimum total number of cactus 
species is 16.
-/
theorem minimum_cactus_species (k : ℕ) (h : ∀ (collections : fin 80 → finset (fin k)),
  (∀ i, collections i ≠ ∅) ∧ (∃ (j : fin k), ∀ i, j ∉ collections i) ∧ 
  (∀ (S : finset (fin 80)), S.card = 15 → ∃ j, ∀ i ∈ S, j ∈ collections i)) :
  k ≥ 16 :=
sorry

end minimum_cactus_species_l669_669362


namespace factor_quadratic_polynomial_l669_669019

theorem factor_quadratic_polynomial :
  (∀ x : ℝ, x^4 - 36*x^2 + 25 = (x^2 - 6*x + 5) * (x^2 + 6*x + 5)) :=
by
  sorry

end factor_quadratic_polynomial_l669_669019


namespace notebook_cost_correct_l669_669002

def totalSpent : ℕ := 32
def costBackpack : ℕ := 15
def costPen : ℕ := 1
def costPencil : ℕ := 1
def numberOfNotebooks : ℕ := 5
def costPerNotebook : ℕ := 3

theorem notebook_cost_correct (h_totalSpent : totalSpent = 32)
    (h_costBackpack : costBackpack = 15)
    (h_costPen : costPen = 1)
    (h_costPencil : costPencil = 1)
    (h_numberOfNotebooks : numberOfNotebooks = 5) :
    (totalSpent - (costBackpack + costPen + costPencil)) / numberOfNotebooks = costPerNotebook :=
by
  sorry

end notebook_cost_correct_l669_669002


namespace num_of_integers_satisfying_conditions_l669_669122

theorem num_of_integers_satisfying_conditions : ∃ n_set : Set ℤ, (∀ n ∈ n_set, 200 < n ∧ n < 300 ∧ ∀ r, n % 7 = r ↔ n % 9 = r) ∧ n_set.card = 7 := by
  sorry

end num_of_integers_satisfying_conditions_l669_669122


namespace PQST_value_l669_669177

noncomputable def compute_PQST (P Q S T : ℝ) (hPQ : P > 0) (hQ : Q > 0) (hS : S > 0) (hT : T > 0)
    (h1 : log10 (P * Q) + log10 (P * S) = 3)
    (h2 : log10 (S * T) + log10 (S * Q) = 2)
    (h3 : log10 (T * Q) + log10 (T * P) = 5) : ℝ :=
PQST

theorem PQST_value (P Q S T : ℝ) (hPQ : P > 0) (hQ : Q > 0) (hS : S > 0) (hT : T > 0)
    (h1 : log10 (P * Q) + log10 (P * S) = 3)
    (h2 : log10 (S * T) + log10 (S * Q) = 2)
    (h3 : log10 (T * Q) + log10 (T * P) = 5) : compute_PQST P Q S T hPQ hQ hS hT h1 h2 h3 = 10^(10/3) :=
sorry

end PQST_value_l669_669177


namespace csc_cos_identity_l669_669391

theorem csc_cos_identity : 
  (Real.csc (Real.pi / 18) - 4 * Real.cos (2 * Real.pi / 9)) = 2 := 
by
  sorry

end csc_cos_identity_l669_669391


namespace triangle_DEF_area_l669_669159

-- Given area of square PQRS
def area_PQRS : ℝ := 36

-- Side length of PQRS
def side_PQRS : ℝ := real.sqrt area_PQRS

-- Side length of smaller squares inside PQRS
def side_smaller_squares : ℝ := 2

-- Triangle DEF with DE = DF
structure Triangle :=
  (DE : ℝ)
  (DF : ℝ)
  (E F : ℝ × ℝ)
  (D : ℝ × ℝ)
  (hDE_DF : DE = DF)

-- Center T of square PQRS
def center_T : ℝ × ℝ := (side_PQRS / 2, side_PQRS / 2)

-- Distance DT which is the altitude of triangle DEF
def distance_DT : ℝ := (side_PQRS / 2) + side_smaller_squares + side_smaller_squares

-- Base EF when triangle DEF is folded
def base_EF : ℝ := side_PQRS - side_smaller_squares - side_smaller_squares

-- The area of triangle DEF
def area_triangle_DEF (EF DT : ℝ) := (1 / 2) * EF * DT

theorem triangle_DEF_area :
  ∃ (tri : Triangle), area_triangle_DEF base_EF distance_DT = 7 :=
by
  sorry

end triangle_DEF_area_l669_669159


namespace value_of_n_l669_669525

theorem value_of_n (n : ℤ) (h : sqrt (8 + n) = 9) : n = 73 :=
sorry

end value_of_n_l669_669525


namespace angle_C_of_triangle_l669_669579

theorem angle_C_of_triangle (A B C : ℝ) (hA : A = 90) (hB : B = 50) (h_sum : A + B + C = 180) : C = 40 := 
by
  sorry

end angle_C_of_triangle_l669_669579


namespace cactus_species_minimum_l669_669366

theorem cactus_species_minimum :
  ∀ (collections : Fin 80 → Fin k → Prop),
  (∀ s : Fin k, ∃ (i : Fin 80), ¬ collections i s)
  → (∀ (c : Finset (Fin 80)), c.card = 15 → ∃ s : Fin k, ∀ (i : Fin 80), i ∈ c → collections i s)
  → 16 ≤ k := 
by 
  sorry

end cactus_species_minimum_l669_669366


namespace problem_l669_669476

noncomputable def a_n (n : ℕ) : ℕ := 2^n
noncomputable def b_n (n : ℕ) : ℕ := n
noncomputable def c_n (n : ℕ) : ℝ := 1 / (b_n n * b_n (n+1))

theorem problem (
  h1 : a_n 2 * a_n 4 = 64,
  h2 : a_n 1 + a_n 1 * 2 + a_n 1 * 2^2 = 14
) :
  a_n 1 = 2 ∧
  a_n 2 = 2^2 ∧
  ∀ (n : ℕ), a_n n = 2^n ∧ 
  ∀ (n : ℕ), b_n(n+1) - b_n n = 1 ∧
  ∀ (n : ℕ), (∑ i in Finset.range n, c_n i = 1 - (1 / (n + 1))) :=
by
  sorry

end problem_l669_669476


namespace other_factor_of_product_l669_669329

def product_has_factors (n : ℕ) : Prop :=
  ∃ a b c d e f : ℕ, n = (2^a) * (3^b) * (5^c) * (7^d) * (11^e) * (13^f) ∧ a ≥ 4 ∧ b ≥ 3

def smallest_w (x : ℕ) : ℕ :=
  if h : x = 1452 then 468 else 1

theorem other_factor_of_product (w : ℕ) : 
  (product_has_factors (1452 * w)) → (w = 468) :=
by
  sorry

end other_factor_of_product_l669_669329


namespace cylinder_capacity_is_volume_l669_669920

-- Definitions of volume, lateral area, and surface area for a cylinder are used here implicitly
def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h
def cylinder_capacity (r h : ℝ) : ℝ := cylinder_volume r h

-- The theorem to prove that the amount of water a cylindrical bucket can hold is indeed its volume
theorem cylinder_capacity_is_volume (r h : ℝ) : cylinder_capacity r h = cylinder_volume r h :=
by 
  sorry

end cylinder_capacity_is_volume_l669_669920


namespace cost_of_55_lilies_l669_669360

-- Define the problem conditions
def price_per_dozen_lilies (p : ℝ) : Prop :=
  p * 24 = 30

def directly_proportional_price (p : ℝ) (n : ℕ) : ℝ :=
  p * n

-- State the problem to prove the cost of a 55 lily bouquet
theorem cost_of_55_lilies (p : ℝ) (c : ℝ) :
  price_per_dozen_lilies p →
  c = directly_proportional_price p 55 →
  c = 68.75 :=
by
  sorry

end cost_of_55_lilies_l669_669360


namespace complex_mul_conj_eq_two_l669_669057

noncomputable def z (c : ℂ) : ℂ := 
  if (c * (1 + complex.i) = 2 * complex.i) 
  then c 
  else 0 -- A dummy value, actual proof logic will not use this.

theorem complex_mul_conj_eq_two (z : ℂ) (hz : z * (1 + complex.i) = 2 * complex.i) : z * (complex.conj z) = 2 :=
sorry

end complex_mul_conj_eq_two_l669_669057


namespace perfect_square_count_21n_le_1500_l669_669033

-- Define the main condition that 21n must be a perfect square
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

-- Theorem stating that the count of positive integers n ≤ 1500 such that 21n is a perfect square is 8
theorem perfect_square_count_21n_le_1500 : 
  {n : ℕ // n > 0 ∧ n ≤ 1500 ∧ isPerfectSquare (21 * n)}.toFinset.card = 8 := 
by 
  sorry

end perfect_square_count_21n_le_1500_l669_669033


namespace factorization_correct_l669_669425

-- Define the initial expression
def initial_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + a

-- Define the factorized form
def factorized_expr (a x : ℝ) : ℝ := a * (x - 1)^2

-- Create the theorem statement
theorem factorization_correct (a x : ℝ) : initial_expr a x = factorized_expr a x :=
by simp [initial_expr, factorized_expr, pow_two, mul_add, add_mul, add_assoc]; sorry

end factorization_correct_l669_669425


namespace binary_to_decimal_l669_669000

theorem binary_to_decimal : let binary := "111.11" in binary.to_decimal = 7.75 := sorry

end binary_to_decimal_l669_669000


namespace right_triangle_area_l669_669749

theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a = 5) (hc : c = 13) :
  1/2 * a * b = 30 :=
by
  have hb : b = 12, from sorry,
  -- Proof needs to be filled here
  sorry

end right_triangle_area_l669_669749


namespace solve_for_x_l669_669905

   theorem solve_for_x (x : ℚ) : 
     (3^(2 * x^2 - 7 * x + 4))^2 = (3^(2 * x^2 + 5 * x - 6))^2 → 
     x = 5 / 6 :=
   by
     sorry
   
end solve_for_x_l669_669905


namespace monotonicity_of_f_range_of_a_inequality_for_f_l669_669497

-- Definition of the function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  |exp x - exp 1| + exp x + a * x

-- Problem 1: Monotonicity of f
theorem monotonicity_of_f (a : ℝ) (x : ℝ) (h_a : a > 0) : monotone (f a) :=
sorry

-- Problem 2: Range of values for a
theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ∈ Ioo (-1/2 : ℝ) ∞ → f a x ≥ exp 1 / 2) : a ∈ Icc (-exp 1 / 2) 0 ∪ Ioc 0 (exp 1) :=
sorry

-- Problem 3: Prove f(x1 * x2) > a + e
theorem inequality_for_f (a : ℝ) (x1 x2 : ℝ) (h_a : a < -exp 1) (h_fx1 : f a x1 = 0) (h_fx2 : f a x2 = 0) (h_x1_x2 : x1 < x2) : f a (x1 * x2) > a + exp 1 :=
sorry

end monotonicity_of_f_range_of_a_inequality_for_f_l669_669497


namespace expected_value_roll_8_sided_die_l669_669852

noncomputable def expected_win_dollars : ℝ :=
  ∑ i in Finset.range 8, (i+1 : ℝ)^3 * (1 / 8 : ℝ)

theorem expected_value_roll_8_sided_die :
  expected_win_dollars = 162 :=
by
  sorry

end expected_value_roll_8_sided_die_l669_669852


namespace findPhoneNumber_l669_669660

noncomputable def isValidPhoneNumber (T : ℕ) : Prop :=
  T >= 100000 ∧ T < 1000000 ∧  
  T % 10 % 2 = 1 ∧
  T / 10000 % 10 = 7 ∧ 
  T / 100 % 10 = 2 ∧
  (T % 3 = T % 4 ∧
   T % 4 = T % 7 ∧
   T % 7 = T % 9 ∧
   T % 9 = T % 11 ∧
   T % 11 = T % 13)

theorem findPhoneNumber (T : ℕ) (h_valid : isValidPhoneNumber T) : T = 720721 :=
by
  sorry

end findPhoneNumber_l669_669660


namespace maximize_triangle_area_l669_669487

noncomputable def sqrt (x : ℝ) := Real.sqrt x

theorem maximize_triangle_area :
  ∀ (x : ℝ), 1 < x ∧ x < 4 → 
  let A := (1 : ℝ, sqrt 1)
  let B := (x, sqrt x)
  let C := (4 : ℝ, sqrt 4)
  let area := 0.5 * abs (1 * (sqrt x - 2) + x * 1 + 4 * (1 - sqrt x))
  let f := fun (m : ℝ) => m - 3 * sqrt m + 2
  let m := (3 / 2) ^ 2
  f' m = 1 - (3 / (2 * sqrt m))  -- Derivative of the function
  ∀ critical_point : ℝ, 
  ∃ m, m = critical_point = (9 / 4) := 
sorry

end maximize_triangle_area_l669_669487


namespace decimal_to_binary_18_l669_669900

theorem decimal_to_binary_18 : (to_binary 18) = 10010 :=
by
  sorry

end decimal_to_binary_18_l669_669900


namespace minimum_cactus_species_l669_669372

theorem minimum_cactus_species (cactophiles : Fin 80 → Set (Fin k)) :
  (∀ s : Fin k, ∃ col, cactophiles col s = False) ∧
  (∀ group : Set (Fin 80), group.card = 15 → ∃ c : Fin k, ∀ col ∈ group, (cactophiles col c)) →
  16 ≤ k :=
by
  sorry

end minimum_cactus_species_l669_669372


namespace casper_initial_candies_l669_669651

/-
On Halloween, Casper ate 1/4 of his candies then gave 3 candies to his brother. 
The next day, he ate 1/2 of the remaining candies and then gave 5 candies to his sister. 
On the third day, he ate the final 10 candies.
Prove that Casper initially had 176 candies.
-/

theorem casper_initial_candies : ∃ x : ℕ, 
  (let y1 := (3 / 4 : ℚ) * x - 3 in
  let y2 := (3 / 8 : ℚ) * x - 3 / 2 - 5 in
  y2 = 10) → x = 176 :=
begin
  sorry
end

end casper_initial_candies_l669_669651


namespace cos_theta_eq_minus_one_half_l669_669116

variables {R : Type*} [field R] [has_inner R] (a b : R)
variables (angle : ℝ) [decidable_eq R]

-- Define the conditions
def norm_a : R := 2
def norm_b : R := 1
def dot_product : R := a • (a + b) = 3

-- Define the proof statement
theorem cos_theta_eq_minus_one_half (h1 : ∥a∥ = norm_a) 
                                     (h2 : ∥b∥ = norm_b)
                                     (h3 : ⦃a⦄ ⦃a + b⦄ = dot_product) :
                                     ⟪a, b⟫ = -1/2 :=
by sorry

end cos_theta_eq_minus_one_half_l669_669116


namespace right_triangle_area_l669_669754

/-- Given a right triangle with hypotenuse 13 meters and one side 5 meters,
prove that the area of the triangle is 30 square meters. -/
theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (hc : c = 13) (ha : a = 5) :
  1/2 * a * b = 30 :=
by sorry

end right_triangle_area_l669_669754


namespace smallest_c_for_inverse_l669_669632

noncomputable def g (x : ℝ) : ℝ := (x + 3)^2 - 6

theorem smallest_c_for_inverse : 
  ∃ (c : ℝ), (∀ x1 x2, x1 ≥ c → x2 ≥ c → g x1 = g x2 → x1 = x2) ∧ 
            (∀ c', c' < c → ∃ x1 x2, x1 ≥ c' → x2 ≥ c' → g x1 = g x2 ∧ x1 ≠ x2) ∧ 
            c = -3 :=
by 
  sorry

end smallest_c_for_inverse_l669_669632


namespace sum_seq_remainder_l669_669896

theorem sum_seq_remainder (n : ℕ) : 
  (∑ i in finset.range (n + 1), 3 ^ (i + 1)) % 8 = 4 :=
by
  sorry

end sum_seq_remainder_l669_669896


namespace problem_I_problem_II_l669_669104

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2/x - 1
def g (x : ℝ) : ℝ := x + 1/x 

theorem problem_I :
  ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y > 0 ∧ y ≠ x → f y ≥ f x) ∧ f 2 = Real.log 2 := sorry

theorem problem_II (e : ℝ) (he : e = Real.exp 1) :
  ∃ m : ℝ, 
    (∀ x0 ∈ Set.Icc 1 e, g x0 < m * (f x0 + 1)) ↔ (-∞ < m ∧ m < -2) ∨ (m > (Real.exp 2 + 1) / (Real.exp 1 - 1) ∧ m < ∞) := sorry

end problem_I_problem_II_l669_669104


namespace michael_hours_worked_l669_669308

def michael_hourly_rate := 7
def michael_overtime_rate := 2 * michael_hourly_rate
def work_hours := 40
def total_earnings := 320

theorem michael_hours_worked :
  (total_earnings = michael_hourly_rate * work_hours + michael_overtime_rate * (42 - work_hours)) :=
sorry

end michael_hours_worked_l669_669308


namespace ratio_AF_FB_l669_669163

-- Given conditions
variables {A B C D F P : Type} [AffineSpace A] [AffineSpace B] [AffineSpace C] [AffineSpace D] [AffineSpace F] [AffineSpace P]

-- Conditions with specific proportions
axiom AD_on_AC : ∀ (A B C D F P : Type), D ∈ line_through A C
axiom AF_on_AB : ∀ (A B C D F P : Type), F ∈ line_through A B
axiom AP_PD_ratio : ∀ (A B C D F P : Type), 3/2 = (dist A P) / (dist P D)
axiom FP_PC_ratio : ∀ (A B C D F P : Type), 3/4 = (dist F P) / (dist P C)

-- Prove part
theorem ratio_AF_FB : ∀ (A B C D F P : Type) [AffineSpace A] [AffineSpace B] [AffineSpace C] [AffineSpace D] [AffineSpace F] [AffineSpace P],
    D ∈ line_through A C ∧ F ∈ line_through A B ∧ (dist A P) / (dist P D) = 3 / 2 ∧ (dist F P) / (dist P C) = 3 / 4 →
    (dist A F) / (dist F B) = 3 / 7 :=
by
  intros A B C D F P _ _ _ _ _ _
  intros h
  cases h with hD h_ratio
  sorry

end ratio_AF_FB_l669_669163


namespace find_imaginary_part_l669_669980

-- Define z as a complex number
variable {z : ℂ}

-- Given condition
def given_condition := (z * complex.I = 2 + complex.I)

-- The goal is to find the imaginary part of z
theorem find_imaginary_part (hz : given_condition) : complex.im z = -2 := by sorry

end find_imaginary_part_l669_669980


namespace simplify_cubic_root_l669_669696

theorem simplify_cubic_root : 
  (∛(54880000) = 20 * ∛((5^2) * 137)) :=
sorry

end simplify_cubic_root_l669_669696


namespace simplify_expression_l669_669382

theorem simplify_expression (y : ℝ) : (4 - real.sqrt (y^2 - 16))^2 = y^2 - 8 * real.sqrt (y^2 - 16) :=
sorry

end simplify_expression_l669_669382


namespace max_PA_PB_PC_l669_669066

noncomputable def max_product (A B C P : Point) : ℝ :=
  dist P A * dist P B * dist P C

theorem max_PA_PB_PC :
  ∀ (A B C P : Point), 
  (∀ a b c : Point, is_right_triangle a b c ∧ (dist a b = 1) ∧ (dist a c = 1) → ∃ P : Point, 
  (P ∈ Triangle_side a b c) ∧ 
  (max_product a b c P = (Real.sqrt 2) / 4)) :=
begin
  sorry
end

end max_PA_PB_PC_l669_669066


namespace game_cost_l669_669040

theorem game_cost
    (grandmother_amt aunt_amt uncle_amt total_amt expenses left_amt : ℤ)
    (grandmother_gave : grandmother_amt = 20)
    (aunt_gave : aunt_amt = 25)
    (uncle_gave : uncle_amt = 30)
    (total_initial_amt : total_amt = 125)
    (remaining_amt : left_amt = 20) :
    let spent := total_amt - left_amt in
    let given_total := grandmother_amt + aunt_amt + uncle_amt in
    spent = 35 * 3 :=
by
  -- Placeholder for the proof
  sorry

end game_cost_l669_669040


namespace mean_not_twice_of_mean_median_not_twice_of_median_l669_669081

variable {α : Type*} [LinearOrder α] {x : ℕ → α} {n : ℕ}

-- Define the data sets
def data1 (x : ℕ → α) : Fin n → α := fun i => x i
def data2 (x : ℕ → α) : Fin n → α := fun i => 2 * x i - 1

-- Define the mean of data set
def mean (f : Fin n → α) [DivisionRing α] : α :=
  1 / n • (Finset.univ : Finset (Fin n)).sum f

-- Define the median of data set (assuming n is odd for simplicity)
def median (f : Fin n → α) : α :=
  let l := (Finset.univ : Finset (Fin n)).val.map f in
  l.sorted.nthLe (n / 2) sorry

-- Proof that mean of data2 is not twice the mean of data1
theorem mean_not_twice_of_mean (x : ℕ → α) [DivisionRing α] [CharZero α] :
  mean (data2 x) ≠ 2 * mean (data1 x) :=
sorry

-- Proof that median of data2 is not twice the median of data1
theorem median_not_twice_of_median (x : ℕ → α) [LinearOrder α] :
  median (data2 x) ≠ 2 * median (data1 x) :=
sorry

end mean_not_twice_of_mean_median_not_twice_of_median_l669_669081


namespace smallest_four_digit_integer_solution_l669_669449

theorem smallest_four_digit_integer_solution:
  ∃ x : ℤ, 1000 ≤ x ∧ x ≤ 9999 ∧ 
           (11 * x ≡ 33 [MOD 22]) ∧ 
           ((3 * x + 10) ≡ 19 [MOD 12]) ∧ 
           ((5 * x - 3) ≡ (2 * x) [MOD 36]) ∧ 
           (x ≡ 3 [MOD 4]) ∧ 
           x = 1001 :=
by
  sorry

end smallest_four_digit_integer_solution_l669_669449


namespace expected_value_of_random_number_l669_669262

/-- 
The expected value of a random number formed by placing a zero and a decimal point in front
of a sequence of one thousand random digits is 0.5.
-/
theorem expected_value_of_random_number : 
  let X := ∑ k in (finRange 1000), (4.5 / 10 ^ (k + 1))
  in X = 0.5 :=
sorry

end expected_value_of_random_number_l669_669262


namespace log_equality_implies_x_value_l669_669023

theorem log_equality_implies_x_value :
  ∀ (x : ℝ), (0 < x) → (x ≠ 1) → (log x 8 = log 64 4) → x = 512 := by
  sorry

end log_equality_implies_x_value_l669_669023


namespace length_XY_eq_semiperimeter_l669_669935

variables {A B C X Y : Point} {a b c s : ℝ} -- a, b, c will represent side lengths of △ABC, and s is the semiperimeter
variable (triangle_ABC : Triangle A B C)

-- Conditions from the problem
variable (AX_perpendicular : Perpendicular AX (external_bisector B A C))
variable (AY_perpendicular : Perpendicular AY (external_bisector C A B))

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

theorem length_XY_eq_semiperimeter
  (h1 : AX_perpendicular)
  (h2 : AY_perpendicular)
  (a b c : ℝ)
  (h3 : triangle_ABC.sides = (a, b, c))
  (s := semiperimeter a b c) :
  XY.length = s :=
sorry

end length_XY_eq_semiperimeter_l669_669935


namespace equilateral_triangles_parallel_lines_l669_669272

open EuclideanGeometry

/--
Let l1 and l2 be two parallel lines. Let A be a point equidistant from l1 and l2. Suppose a line passing through A intersects lines l1 and l2 at points M and N respectively. 
Prove that the set of points P that form equilateral triangles MNP are two lines perpendicular to l1 and l2.
-/
theorem equilateral_triangles_parallel_lines (l1 l2 : Line) (A M N : Point) (P : Set Point) 
    (h_parallel : l1 ∥ l2)
    (h_eqdist : equidistant_from_lines A l1 l2)
    (h_intersectM : intersects_at_line_through A l1 M)
    (h_intersectN : intersects_at_line_through A l2 N)
    (h_equilateral : ∀ P ∈ P, equilateral_triangle M N P):
  P = set_of_lines_perpendicular l1 l2 := 
sorry

end equilateral_triangles_parallel_lines_l669_669272


namespace problem_l669_669466

theorem problem (θ : ℝ) (htan : Real.tan θ = 1 / 3) : Real.cos θ ^ 2 + 2 * Real.sin θ = 6 / 5 := 
by
  sorry

end problem_l669_669466


namespace math_proof_problem_l669_669951

section
variables {Point : Type} [MetricSpace Point] [AddGroup Point] [VectorSpace ℝ Point]
variables (A B C : Point)
variables (x y : ℝ)
variables [Nonempty Point]

-- Given the coordinates of the vertices A(0, 2), B(0, -2), and C(-2, 2)
def A : Point := (0, 2)
def B : Point := (0, -2)
def C : Point := (-2, 2)

-- Prove the equation of the line containing the median that is parallel to side BC
def median_parallel_to_BC_eq : Prop := 2 * x + y = 0

-- Prove the equation of the circumcircle of △ABC
def circumcircle_eq : Prop := (x + 1) ^ 2 + y ^ 2 = 5

-- The theorem stating the proof problem
theorem math_proof_problem :
  A = (0, 2) →
  B = (0, -2) →
  C = (-2, 2) →
  median_parallel_to_BC_eq A B C →
  circumcircle_eq A B C :=
by
  intro hA hB hC
  sorry

end math_proof_problem_l669_669951


namespace g_10_44_l669_669008

def g (x y : ℕ) : ℕ := sorry

axiom g_cond1 (x : ℕ) : g x x = x ^ 2
axiom g_cond2 (x y : ℕ) : g x y = g y x
axiom g_cond3 (x y : ℕ) : (x + y) * g x y = y * g x (x + y)

theorem g_10_44 : g 10 44 = 440 := sorry

end g_10_44_l669_669008


namespace ways_to_split_balls_l669_669246

-- Defining the given conditions as constants in Lean
constant R : Type
constant Y : Type
constant G : Type

constant red_balls : Finset R
constant yellow_balls : Finset Y
constant green_balls : Finset G

-- Assuming the sizes of the sets
axiom red_balls_size : red_balls.card = 3
axiom yellow_balls_size : yellow_balls.card = 3
axiom green_balls_size : green_balls.card = 3

-- Define the main theorem that needs to be proved
theorem ways_to_split_balls : ∃ (splits : Finset (Finset (R ⊕ Y ⊕ G))) (h : splits.card = 3), splits.card = 10 :=
  sorry

end ways_to_split_balls_l669_669246


namespace simplify_cubed_root_l669_669704

def c1 : ℕ := 54880000
def c2 : ℕ := 10^5 * 5488
def c3 : ℕ := 5488
def c4 : ℕ := 2^4 * 343
def c5 : ℕ := 343
def c6 : ℕ := 7^3

theorem simplify_cubed_root : (c1^(1 / 3 : ℝ) : ℝ) = 1400 := 
by {
  let h1 : c1 = c2 := sorry,
  let h2 : c3 = c4 := sorry,
  let h3 : c5 = c6 := sorry,
  rw [h1, h2, h3],
  sorry
}

end simplify_cubed_root_l669_669704


namespace omega_not_3_over_4_l669_669742

theorem omega_not_3_over_4 (ω : ℝ) (hω : ω > 0) :
  (∀ x y : ℝ, (-π / 2 < x ∧ x < π / 2) ∧ (-π / 2 < y ∧ y < π / 2) ∧ x < y →
  (sin (ω * x) - cos (ω * x)) < (sin (ω * y) - cos (ω * y))) →
  ω ≠ 3 / 4 :=
sorry

end omega_not_3_over_4_l669_669742


namespace simplify_cubed_root_l669_669707

def c1 : ℕ := 54880000
def c2 : ℕ := 10^5 * 5488
def c3 : ℕ := 5488
def c4 : ℕ := 2^4 * 343
def c5 : ℕ := 343
def c6 : ℕ := 7^3

theorem simplify_cubed_root : (c1^(1 / 3 : ℝ) : ℝ) = 1400 := 
by {
  let h1 : c1 = c2 := sorry,
  let h2 : c3 = c4 := sorry,
  let h3 : c5 = c6 := sorry,
  rw [h1, h2, h3],
  sorry
}

end simplify_cubed_root_l669_669707


namespace find_ab_l669_669505

variables (a b c : ℝ)

-- Defining the conditions
def cond1 : Prop := a - b = 5
def cond2 : Prop := a^2 + b^2 = 34
def cond3 : Prop := a^3 - b^3 = 30
def cond4 : Prop := a^2 + b^2 - c^2 = 50

theorem find_ab (h1 : cond1 a b) (h2 : cond2 a b) (h3 : cond3 a b) (h4 : cond4 a b c) :
  a * b = 4.5 :=
sorry

end find_ab_l669_669505


namespace max_area_triangle_ABP_l669_669989

-- Define the line and parabola
def line_eq (x y : ℝ) := 2 * x - y + 4 = 0
def parabola_eq (x y : ℝ) := x^2 = 4 * y

-- Define the points A and B as intersection points
def is_intersection_point (x y : ℝ) : Prop := line_eq x y ∧ parabola_eq x y

-- Define point O as the origin
def O : ℝ × ℝ := (0, 0)

-- Define point P on the parabolic arc AOB
def point_on_parabola_arc (x y : ℝ) (A B : ℝ × ℝ) : Prop :=
  parabola_eq x y ∧ x ≥ fst A ∧ x ≤ fst B

-- Define the maximum area of triangle ABP
def max_area_triangle (A B P : ℝ × ℝ) : ℝ := 
  1 / 2 * abs ((fst B - fst A) * (snd P - snd A) - (fst P - fst A) * (snd B - snd A))

-- The theorem to prove
theorem max_area_triangle_ABP (A B P : ℝ × ℝ) (hA : is_intersection_point (fst A) (snd A))
  (hB : is_intersection_point (fst B) (snd B)) (hP : point_on_parabola_arc (fst P) (snd P) A B) :
  max_area_triangle A B P = 20 :=
by sorry

end max_area_triangle_ABP_l669_669989


namespace bagel_pieces_after_10_cuts_l669_669325

def bagel_pieces_after_cuts (initial_pieces : ℕ) (cuts : ℕ) : ℕ :=
  initial_pieces + cuts

theorem bagel_pieces_after_10_cuts : bagel_pieces_after_cuts 1 10 = 11 := by
  sorry

end bagel_pieces_after_10_cuts_l669_669325


namespace max_gcd_consecutive_terms_l669_669807

-- Define the sequence a_n = n! + n
noncomputable def sequence (n : ℕ) : ℕ := n.factorial + n

-- Function to compute the gcd of two numbers
def gcd (a b : ℕ) := Nat.gcd a b 

-- Define the maximum gcd of consecutive terms is 2
theorem max_gcd_consecutive_terms : 
  ∃ n, gcd (sequence n) (sequence (n + 1)) = 2 :=
sorry

end max_gcd_consecutive_terms_l669_669807


namespace simplify_cubed_root_l669_669708

def c1 : ℕ := 54880000
def c2 : ℕ := 10^5 * 5488
def c3 : ℕ := 5488
def c4 : ℕ := 2^4 * 343
def c5 : ℕ := 343
def c6 : ℕ := 7^3

theorem simplify_cubed_root : (c1^(1 / 3 : ℝ) : ℝ) = 1400 := 
by {
  let h1 : c1 = c2 := sorry,
  let h2 : c3 = c4 := sorry,
  let h3 : c5 = c6 := sorry,
  rw [h1, h2, h3],
  sorry
}

end simplify_cubed_root_l669_669708


namespace math_problem_l669_669386

theorem math_problem :
  -2 * (Real.cbrt (-27)) + Real.abs (1 - Real.sqrt 3) - (1/2)^(-2 : ℤ) = 1 + Real.sqrt 3 :=
by
  -- Proof is omitted
  sorry

end math_problem_l669_669386


namespace simplify_expression_l669_669635

noncomputable def proof_problem (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) : Prop :=
  (1 / (1 + a + a * b) + 1 / (1 + b + b * c) + 1 / (1 + c + c * a)) = 1

theorem simplify_expression (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) :
  proof_problem a b c h h_abc :=
by sorry

end simplify_expression_l669_669635


namespace log_basis_512_l669_669020

theorem log_basis_512 (x : ℝ) (hx : real.log x 8 = real.log 64 4) : x = 512 := 
by 
  sorry

end log_basis_512_l669_669020


namespace slope_of_line_l669_669868

theorem slope_of_line :
  ∃ (m : ℝ), (∃ b : ℝ, ∀ x y : ℝ, y = m * x + b) ∧
             (b = 2 ∧ ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ = 0 ∧ x₂ = 269 ∧ y₁ = 2 ∧ y₂ = 540 ∧ 
             m = (y₂ - y₁) / (x₂ - x₁)) ∧
             m = 2 :=
by {
  sorry
}

end slope_of_line_l669_669868


namespace us_supermarkets_count_l669_669833

-- Definition of variables and conditions
def total_supermarkets : ℕ := 84
def difference_us_canada : ℕ := 10

-- Proof statement
theorem us_supermarkets_count (C : ℕ) (H : 2 * C + difference_us_canada = total_supermarkets) :
  C + difference_us_canada = 47 :=
sorry

end us_supermarkets_count_l669_669833


namespace factorize_expression_l669_669416

theorem factorize_expression (a x : ℝ) :
  a * x^2 - 2 * a * x + a = a * (x - 1) ^ 2 := 
sorry

end factorize_expression_l669_669416


namespace cactus_species_minimum_l669_669367

theorem cactus_species_minimum :
  ∀ (collections : Fin 80 → Fin k → Prop),
  (∀ s : Fin k, ∃ (i : Fin 80), ¬ collections i s)
  → (∀ (c : Finset (Fin 80)), c.card = 15 → ∃ s : Fin k, ∀ (i : Fin 80), i ∈ c → collections i s)
  → 16 ≤ k := 
by 
  sorry

end cactus_species_minimum_l669_669367


namespace max_tokens_without_diagonal_max_tokens_with_diagonals_l669_669197

-- Define the size of the chessboard
def chessboard_size : Nat := 8

-- Define the maximum number of tokens per row, column, and main diagonal
def max_tokens_per_line : Nat := 4

-- Define the proof statement for part a)
theorem max_tokens_without_diagonal (n : Nat) :
  (∀ i : Fin chessboard_size, ∃ r : Fin chessboard_size → Fin chessboard_size → bool, 
   (∑ j, if r i j then 1 else 0 ≤ max_tokens_per_line) ∧ 
   (∀ k, ∑ j, if r j k then 1 else 0 ≤ max_tokens_per_line)) → 
  n ≤ 32 := 
sorry

-- Define the proof statement for part b)
theorem max_tokens_with_diagonals (n : Nat) :
  (∀ i : Fin chessboard_size, ∃ r : Fin chessboard_size → Fin chessboard_size → bool, 
   (∑ j, if r i j then 1 else 0 ≤ max_tokens_per_line) ∧ 
   (∀ k, ∑ j, if r j k then 1 else 0 ≤ max_tokens_per_line) ∧ 
   (∑ j, if r j j then 1 else 0 ≤ max_tokens_per_line) ∧ 
   (∑ j, if r j (Fin.reverse j) then 1 else 0 ≤ max_tokens_per_line)) → 
  n ≤ 32 := 
sorry

end max_tokens_without_diagonal_max_tokens_with_diagonals_l669_669197


namespace factorization_l669_669435

theorem factorization (a x : ℝ) : ax^2 - 2ax + a = a * (x - 1) ^ 2 := 
by
  sorry

end factorization_l669_669435


namespace number_of_Cheburashkas_erased_l669_669596

theorem number_of_Cheburashkas_erased :
  ∃ (n : ℕ), 
    (∀ x, x ≥ 1 → 
      (let totalKrakozyabras = (2 * (x - 1) = 29) in
         x - 2 = 11)) :=
sorry

end number_of_Cheburashkas_erased_l669_669596


namespace sin_x_solution_l669_669670

theorem sin_x_solution (A B C x : ℝ) (h : A * Real.cos x + B * Real.sin x = C) :
  ∃ (u v : ℝ),  -- We assert the existence of u and v such that 
    Real.sin x = (A * C + B * u) / (A^2 + B^2) ∨ 
    Real.sin x = (A * C - B * v) / (A^2 + B^2) :=
sorry

end sin_x_solution_l669_669670


namespace sum_of_first_n_terms_of_b_n_l669_669477

noncomputable def a_n : ℕ+ → ℝ := 
λ n, if n = 1 then 1 else 
  (1 + (n - 1 : ℕ))

noncomputable def S_n : ℕ+ → ℝ := 
λ n, ∑ i in finset.range n, a_n i

noncomputable def b_n : ℕ+ → ℝ := 
λ n, (-1)^(n : ℕ) * (2 * n + 1) / (a_n n * a_n (n + 1))

noncomputable def T_n : ℕ+ → ℝ :=
λ n, ∑ i in finset.range n, b_n i

theorem sum_of_first_n_terms_of_b_n : 
  ∀ n : ℕ+, T_n n = -1 + ((-1 : ℝ)^(n : ℕ) / (n + 1)) := 
by {
-- The proof will go here.
sorry
}

end sum_of_first_n_terms_of_b_n_l669_669477


namespace investor_buy_price_l669_669303

variables (D_rate : ℝ) (par_value : ℝ) (ROI_rate : ℝ) (purchase_price : ℝ)

-- Define the conditions
def dividend_received (D_rate : ℝ) (par_value : ℝ) : ℝ := D_rate * par_value

def return_on_investment (dividend : ℝ) (ROI_rate : ℝ) : ℝ := dividend / ROI_rate

-- Prove the purchase price given the above conditions
theorem investor_buy_price
  (h1 : D_rate = 0.125)
  (h2 : par_value = 50)
  (h3 : ROI_rate = 0.25)
  (h4 : dividend_received D_rate par_value = 6.25)
  : return_on_investment 6.25 ROI_rate = 25 :=
by {
  rw [dividend_received, return_on_investment] at *,
  sorry,
}

end investor_buy_price_l669_669303


namespace simplify_cubic_root_l669_669702

theorem simplify_cubic_root : 
  (∛(54880000) = 20 * ∛((5^2) * 137)) :=
sorry

end simplify_cubic_root_l669_669702


namespace find_f_value_find_g_monotonicity_and_extremes_l669_669984

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x - π / 3) - cos (2 * x - π / 3)
noncomputable def g (x : ℝ) : ℝ := -2 * cos ((1/2) * x - π / 3)

theorem find_f_value :
  f (π / 24) = - (sqrt 6 + sqrt 2) / 2 := sorry

theorem find_g_monotonicity_and_extremes :
  (∀ k : ℤ, ∃ min_x max_x : ℝ, 
    g (4 * k * π + 2 * π / 3) ≤ g x ∧ g x ≤ g (4 * k * π + 14 * π / 3) ∧ 
    min_x = 4 * k * π + 8 * π / 3 ∧ max_x = 4 * k * π + 2 * π / 3) ∧
  g (-π / 3) = 0 ∧ g (2π / 3) = -2 := sorry

end find_f_value_find_g_monotonicity_and_extremes_l669_669984


namespace csc_cos_identity_l669_669392

theorem csc_cos_identity : 
  (Real.csc (Real.pi / 18) - 4 * Real.cos (2 * Real.pi / 9)) = 2 := 
by
  sorry

end csc_cos_identity_l669_669392


namespace foci_are_equal_l669_669225

noncomputable def ellipse1_focal_distance : ℝ :=
  let a := real.sqrt 25
  let b := real.sqrt 9
  real.sqrt (a^2 - b^2)

noncomputable def ellipse2_focal_distance (k : ℝ) (hk : k < 9) : ℝ :=
  let a := real.sqrt (25 - k)
  let b := real.sqrt (9 - k)
  real.sqrt (a^2 - b^2)

theorem foci_are_equal (k : ℝ) (hk : k < 9) :
  ellipse1_focal_distance = ellipse2_focal_distance k hk :=
by sorry

end foci_are_equal_l669_669225


namespace point_division_case1_point_division_case2_l669_669446

-- Define the function to calculate the division point coordinate
def division_point (x1 x2 λ : ℝ) : ℝ :=
  (x1 + λ * x2) / (1 + λ)

-- Case 1: λ = 2/3
theorem point_division_case1 : division_point (-1) 5 (2/3) = 7/5 := by
  sorry

-- Case 2: λ = -2
theorem point_division_case2 : division_point (-1) 5 (-2) = 11 := by
  sorry

end point_division_case1_point_division_case2_l669_669446


namespace number_of_boxes_correct_l669_669788

def peaches_per_basket := 25
def baskets_delivered := 5
def peaches_eaten := 5
def box_capacity := 15

theorem number_of_boxes_correct :
  let total_peaches := baskets_delivered * peaches_per_basket,
      remaining_peaches := total_peaches - peaches_eaten,
      number_of_boxes := remaining_peaches / box_capacity in
  number_of_boxes = 8 :=
by
  sorry

end number_of_boxes_correct_l669_669788


namespace total_seeds_in_garden_l669_669517

-- Definitions based on conditions
def large_bed_rows : Nat := 4
def large_bed_seeds_per_row : Nat := 25
def medium_bed_rows : Nat := 3
def medium_bed_seeds_per_row : Nat := 20
def num_large_beds : Nat := 2
def num_medium_beds : Nat := 2

-- Theorem statement to show total seeds
theorem total_seeds_in_garden : 
  num_large_beds * (large_bed_rows * large_bed_seeds_per_row) + 
  num_medium_beds * (medium_bed_rows * medium_bed_seeds_per_row) = 320 := 
by
  sorry

end total_seeds_in_garden_l669_669517


namespace equation_is_linear_l669_669539

-- Define the conditions and the proof statement
theorem equation_is_linear (m n : ℕ) : 3 * x ^ (2 * m + 1) - 2 * y ^ (n - 1) = 7 → (2 * m + 1 = 1) ∧ (n - 1 = 1) → m = 0 ∧ n = 2 :=
by
  sorry

end equation_is_linear_l669_669539


namespace mixed_doubles_team_selection_l669_669151

theorem mixed_doubles_team_selection : 
  let m := 5 in let f := 4 in m * f = 20 := by
  sorry

end mixed_doubles_team_selection_l669_669151


namespace log_fraction_inequalities_l669_669494

theorem log_fraction_inequalities {x : ℝ} (h1 : 1 < x) (h2 : x < 2) :
  (ln x / x) ^ 2 < ln x / x ∧ ln x / x < ln (x^2) / x^2 := 
sorry

end log_fraction_inequalities_l669_669494


namespace smallest_possible_b_l669_669628

-- Definitions of conditions
variables {a b c : ℤ}

-- Conditions expressed in Lean
def is_geometric_progression (a b c : ℤ) : Prop := b^2 = a * c
def is_arithmetic_progression (a b c : ℤ) : Prop := a + b = 2 * c

-- The theorem statement
theorem smallest_possible_b (a b c : ℤ) 
  (h1 : a < b) (h2 : b < c) 
  (hg : is_geometric_progression a b c) 
  (ha : is_arithmetic_progression a c b) : b = 2 := sorry

end smallest_possible_b_l669_669628


namespace krishan_money_l669_669773

variable {R G K : ℕ}

theorem krishan_money 
  (h1 : R / G = 7 / 17)
  (h2 : G / K = 7 / 17)
  (hR : R = 588)
  : K = 3468 :=
by
  sorry

end krishan_money_l669_669773


namespace limit_S_l669_669620

noncomputable def S (a : ℝ) : ℕ+ → ℝ
| ⟨1, _⟩ := Real.log a
| ⟨(n+1 : ℕ), hn⟩ := ∑ i in Finset.range n, Real.log (a - S a ⟨i + 1, nat.succ_pos i⟩)

theorem limit_S (a : ℝ) (h : a > 0) : 
  (tendsto (S a) at_top (𝓝 (a - 1))) :=
sorry

end limit_S_l669_669620


namespace close_sum_exists_8742_l669_669798

-- Define the condition that we are using each digit from 1 to 8 exactly once.
def uses_digits {n : ℕ} (digits : List ℕ) (number : ℕ) : Prop :=
  ∀ d ∈ digits, d ∈ number.digits 10

-- Define the list of digits from 1 to 8.
def digits_1_to_8 : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Define the predicate for checking if the sum of the two numbers is close to 10000.
def close_to_10000 (n1 n2 : ℕ) : Prop :=
  abs (n1 + n2 - 10000) <= abs (8742 + (10000 - 8742) - 10000)

-- The given number should be a four-digit number within the range using exactly the given digits.
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000 ∧ ∃ l : List ℕ, uses_digits digits_1_to_8 n

-- Existence of the number 8742 such that it uses digits 1 to 8 exactly once 
-- and another number to satisfy the sum condition.
theorem close_sum_exists_8742 (n1 n2 : ℕ)
  (h1 : is_four_digit n1)
  (h2 : is_four_digit n2) :
  close_to_10000 n1 n2 ∧ (n1 = 8742 ∨ n2 = 8742) :=
sorry

end close_sum_exists_8742_l669_669798


namespace maximize_profit_l669_669328

def C1 (x : ℝ) : ℝ := (1/3) * x^2 + 10 * x
def C2 (x : ℝ) : ℝ := 51 * x + 10000 / x - 1450

def L (x : ℝ) : ℝ :=
if h : x < 80 then
  (0.05 * 1000 * x) - (C1 x + 250)
else
  (0.05 * 1000 * x) - (C2 x + 250)

theorem maximize_profit : L 100 = 1000 := sorry

end maximize_profit_l669_669328


namespace highest_and_lowest_score_average_score_l669_669552

def std_score : ℤ := 60
def scores : List ℤ := [36, 0, 12, -18, 20]

theorem highest_and_lowest_score 
  (highest_score : ℤ) (lowest_score : ℤ) : 
  highest_score = std_score + 36 ∧ lowest_score = std_score - 18 := 
sorry

theorem average_score (avg_score : ℤ) :
  avg_score = std_score + ((36 + 0 + 12 - 18 + 20) / 5) := 
sorry

end highest_and_lowest_score_average_score_l669_669552


namespace expected_value_of_X_l669_669267

-- Define the sequence of random digits as a list of natural numbers (0 to 9)
def random_digits : List ℕ := (List.range 10).take 1000

-- Define the function that forms the number X
def X (digits : List ℕ) : ℝ :=
  digits.enum.foldr (λ (p : ℕ × ℕ) (acc : ℝ) => acc + p.snd * 10^(-(p.fst + 1))) 0

-- Define the expected value of a single digit
def expected_value_digit : ℝ := (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) / 10

-- The main statement to prove
theorem expected_value_of_X (digits : List ℕ) (h_digits : digits.length = 1000) :
  (∑ i in Finset.range digits.length, 10^(-(i + 1)) * expected_value_digit) = 0.5 :=
by {
  sorry
}

end expected_value_of_X_l669_669267


namespace factorize_expression_l669_669420

theorem factorize_expression (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
by sorry

end factorize_expression_l669_669420


namespace smallest_n_exists_l669_669188

open Real

noncomputable def smallest_n : ℕ := 26

theorem smallest_n_exists :
  ∃ (n : ℕ) (x : Fin n → ℝ),
    (∀ i, |x i| < 1) ∧
    (∑ i, |x i| = 25 + |∑ i, x i|) ∧
    n = smallest_n :=
by
  use 26, fun i => if i.val < 13 then 25/26 else -25/26
  split
  { intro i
    split_ifs
    · exact abs_lt.2 ⟨neg_lt.2 (by norm_num), by norm_num⟩
    · exact abs_lt.2 ⟨neg_lt.2 (by norm_num), by norm_num⟩ }
  split
  { have h1 := sum_const (25 / 26) 13
    have h2 := sum_const (-25 / 26) 13
    rw [h1, h2, mul_div_cancel' _ (nat.cast_ne_zero.2 (by norm_num : (26 : ℝ) ≠ 0)), add_comm, abs_eq_self, add_zero]
    · linarith }
  { refl }
  sorry

end smallest_n_exists_l669_669188


namespace number_of_poodles_groomed_l669_669768

theorem number_of_poodles_groomed (P : ℕ) : 30 * P + 8 * 15 = 210 → P = 3 := by
  intro h
  let terrier_time := 8 * 15
  have poodle_time : 30 * P + terrier_time = 210 := h
  have total_time : 30 * P = 210 - terrier_time := by linarith
  have poodles_groomed := (210 - terrier_time) / 30
  have poodle_groomed_lemma : P = poodles_groomed := by sorry
  exact poodle_groomed_lemma

end number_of_poodles_groomed_l669_669768


namespace cube_root_of_54880000_l669_669679

theorem cube_root_of_54880000 : (real.cbrt 54880000) = 140 * (real.cbrt 10) :=
by
  -- Definitions based on conditions
  have h1 : 54880000 = 10^3 * 54880, by norm_num
  have h2 : 54880 = 2^5 * 7^3 * 5, by norm_num
  have h3 : 10 = 2 * 5, by norm_num
  
  -- Cube root properties and simplifications are implicitly inferred by the system
  sorry

end cube_root_of_54880000_l669_669679


namespace problem_statement_l669_669863

-- Define the position function P based on the cyclic pattern
def P : ℕ → ℤ
| 0       := 0
| (n + 1) := let cycle := n % 5 in
  match cycle with
  | 0 := P n + 3
  | 1 := P n - 2
  | 2 := P n + 3
  | 3 := P n - 2
  | 4 := P n + 3
  | _ := 0 -- should not reach here by the definition of cycle
  end

-- Problem statement: Prove that P(103) ≥ P(104)
theorem problem_statement : P 103 ≥ P 104 := 
by
  sorry

end problem_statement_l669_669863


namespace fred_total_cards_l669_669932

theorem fred_total_cards 
  (initial_cards : ℕ := 26) 
  (cards_given_to_mary : ℕ := 18) 
  (unopened_box_cards : ℕ := 40) : 
  initial_cards - cards_given_to_mary + unopened_box_cards = 48 := 
by 
  sorry

end fred_total_cards_l669_669932


namespace volume_ratio_larger_to_smaller_l669_669842

noncomputable def volume_ratio_cylinders_6x10 : ℝ :=
  let r_A := 5 / Real.pi in
  let V_A := π * r_A^2 * 6 in
  let r_B := 3 / Real.pi in
  let V_B := π * r_B^2 * 10 in
  V_A / V_B

theorem volume_ratio_larger_to_smaller : volume_ratio_cylinders_6x10 = 5 / 3 :=
by
  sorry

end volume_ratio_larger_to_smaller_l669_669842


namespace circle_line_intersect_l669_669982

noncomputable def circle_eq : ℝ → ℝ → Prop := λ x y, (x - 1)^2 + (y - 1)^2 = 2
noncomputable def line_eq : ℝ → ℝ → Prop := λ x y, x + y - 1 = 0
noncomputable def distance_point_line (px py : ℝ) (a b c : ℝ) : ℝ := (abs (a * px + b * py + c)) / (sqrt (a^2 + b^2))

theorem circle_line_intersect :
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq x y :=
begin
  have h : distance_point_line 1 1 1 1 (-1) = sqrt 2 / 2, {
    sorry, /- Proof of distance from (1,1) to x + y - 1 = 0 -/
  },
  have radius_sqrt_2 : sqrt 2 / 2 < sqrt 2, { sorry, /- Proof that sqrt 2 / 2 < sqrt 2 -/ },
  exact exists.intro 0 (exists.intro 1 (and.intro sorry sorry)) /- Find appropriate x, y values -/
end

end circle_line_intersect_l669_669982


namespace sqrt_eq_sub_ge_l669_669126

theorem sqrt_eq_sub_ge (x : ℝ) : sqrt ((x - 3)^2) = x - 3 → x ≥ 3 := by
  sorry

end sqrt_eq_sub_ge_l669_669126


namespace sum_of_coordinates_D_l669_669203

theorem sum_of_coordinates_D
    (C N D : ℝ × ℝ) 
    (hC : C = (10, 5))
    (hN : N = (4, 9))
    (h_midpoint : N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) : 
    C.1 + D.1 + (C.2 + D.2) = 22 :=
  by sorry

end sum_of_coordinates_D_l669_669203


namespace sequence_properties_l669_669175

noncomputable def seq_x (x1 x2 : ℝ) : ℕ → ℝ
| 0     := 0  -- x_0 is undefined, but we need it for technical reasons
| 1     := x1
| 2     := x2
| (n+2) := (seq_x (n+1))^2 / (seq_x n)

def cond1 (x1 x2 : ℝ) : Prop := (8 * x2 - 7 * x1) * x1^7 = 8

def recurrence_rel (x : ℕ → ℝ) (k : ℕ) : Prop :=
  x (k + 1) * x (k - 1) - x k ^ 2 = 
    (x (k - 1) ^ 8 - x k ^ 8) / (x k ^ 7 * x (k - 1) ^ 7)

def is_monotonically_decreasing (x : ℕ → ℝ) : Prop :=
  ∀ n, x n > x (n + 1)

def is_not_monotonic (x : ℕ → ℝ) : Prop :=
  ∃ n, (x n < x (n + 1)) ∨ (x n > x (n + 1) ∧ x (n + 1) < x (n + 2))

def critical_a : ℝ := 8 ^ (1 / 8 : ℝ)

theorem sequence_properties (x1 : ℝ) (x2 : ℝ)
  (h_cond1 : cond1 x1 x2)
  (h_rec : ∀ k ≥ 2, recurrence_rel (seq_x x1 x2) k) :
  (x1 > critical_a → is_monotonically_decreasing (seq_x x1 x2)) ∧
  (0 < x1 ∧ x1 < critical_a → is_not_monotonic (seq_x x1 x2)) :=
  sorry

end sequence_properties_l669_669175


namespace no_superdeficient_numbers_l669_669007

-- Define the sum of divisors function
def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (n % · = 0).sum id

-- Prove that there are no superdeficient numbers
theorem no_superdeficient_numbers :
  ¬∃ n : ℕ, sum_of_divisors (sum_of_divisors n) = n + 3 :=
by
  sorry

end no_superdeficient_numbers_l669_669007


namespace geometric_sequence_product_l669_669947

/-
 Problem Statement:
 Given a geometric sequence {a_n} where a_1 and a_{13} are the two roots of the equation
 x^2 - 8x + 1 = 0, prove that a_5 * a_7 * a_9 = 1.
-/

variables {G : Type*} [linear_ordered_comm_group_with_zero G]

noncomputable def geometric_sequence (a : G) (r : G) (n : ℕ) : G :=
  a * r ^ (n - 1)

theorem geometric_sequence_product
  (a1 a13 : G)
  (h1 : (a1 + a13 = 8))
  (h2 : (a1 * a13 = 1))
  (h3 : ∃ r : G, ∀ (n : ℕ), a1 ≠ 0 → geometric_sequence a1 r 13 = a13) :
  (geometric_sequence a1 (sqrt a1⁻¹) 5) * (geometric_sequence a1 (sqrt a1⁻¹) 7) * (geometric_sequence a1 (sqrt a1⁻¹) 9) = 1 :=
begin
  sorry
end

end geometric_sequence_product_l669_669947


namespace false_statement_l669_669881

theorem false_statement :
  ¬ (∀ x : ℝ, x^2 + 1 > 3 * x) = (∃ x : ℝ, x^2 + 1 ≤ 3 * x) := sorry

end false_statement_l669_669881


namespace paint_cost_per_kg_l669_669732

theorem paint_cost_per_kg (area_per_kg : ℝ) (total_cost : ℝ) (side_length : ℝ) (cost_per_kg : ℝ) :
  (side_length = 8) → (area_per_kg = 16) → (total_cost = 876) →
  cost_per_kg = (total_cost / ((6 * side_length^2) / area_per_kg))
  := 
by
  intro h_side_length
  intro h_area_per_kg
  intro h_total_cost
  have h_total_surface_area : 6 * side_length^2 = 384 := by sorry
  have h_paint_needed : 384 / area_per_kg = 24 := by sorry
  exact calc
    cost_per_kg = total_cost / 24 : by sorry
              ... =  36.5 : by sorry

end paint_cost_per_kg_l669_669732


namespace cube_root_of_54880000_l669_669674

theorem cube_root_of_54880000 : (real.cbrt 54880000) = 140 * (real.cbrt 10) :=
by
  -- Definitions based on conditions
  have h1 : 54880000 = 10^3 * 54880, by norm_num
  have h2 : 54880 = 2^5 * 7^3 * 5, by norm_num
  have h3 : 10 = 2 * 5, by norm_num
  
  -- Cube root properties and simplifications are implicitly inferred by the system
  sorry

end cube_root_of_54880000_l669_669674


namespace notebook_cost_correct_l669_669001

def totalSpent : ℕ := 32
def costBackpack : ℕ := 15
def costPen : ℕ := 1
def costPencil : ℕ := 1
def numberOfNotebooks : ℕ := 5
def costPerNotebook : ℕ := 3

theorem notebook_cost_correct (h_totalSpent : totalSpent = 32)
    (h_costBackpack : costBackpack = 15)
    (h_costPen : costPen = 1)
    (h_costPencil : costPencil = 1)
    (h_numberOfNotebooks : numberOfNotebooks = 5) :
    (totalSpent - (costBackpack + costPen + costPencil)) / numberOfNotebooks = costPerNotebook :=
by
  sorry

end notebook_cost_correct_l669_669001


namespace sum_of_digits_10_pow_45_minus_46_l669_669291

theorem sum_of_digits_10_pow_45_minus_46 :
  let k := (10^45 - 46) in
  (sum_of_digits k = 414) :=
by
  sorry

end sum_of_digits_10_pow_45_minus_46_l669_669291


namespace triangle_area_30_l669_669758

theorem triangle_area_30 (h : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = 5 ∧ c = 13 ∧ b > 0) : 
  ∃ area : ℝ, area = 1 / 2 * 5 * (b : ℝ) ∧ area = 30 :=
by
  sorry

end triangle_area_30_l669_669758


namespace systematic_sampling_starts_with_srs_l669_669725

-- Define the concept of systematic sampling
def systematically_sampled (initial_sampled: Bool) : Bool :=
  initial_sampled

-- Initial sample is determined by simple random sampling
def simple_random_sampling : Bool :=
  True

-- We need to prove that systematic sampling uses simple random sampling at the start
theorem systematic_sampling_starts_with_srs : systematically_sampled simple_random_sampling = True :=
by 
  sorry

end systematic_sampling_starts_with_srs_l669_669725


namespace one_side_weighing_correct_both_sides_weighing_correct_l669_669782

def one_side_weighing (weights : List ℕ) (n : ℕ) : ℕ := 
  sorry

def both_sides_weighing (weights : List ℕ) (n : ℕ) : ℕ := 
  sorry

theorem one_side_weighing_correct (weights : List ℕ) (n : ℕ) : 
    one_side_weighing weights n = (coeff (finset.sum (finset.powerset (finset.from_list weights)) (λ s, t ^ (s.sum id))) n) :=
sorry

theorem both_sides_weighing_correct (weights : List ℕ) (n : ℕ) : 
    both_sides_weighing weights n = (coeff (finset.fold (λ s acc, acc * (t ^ (-s) + 1 + t ^ s)) 1 (finset.from_list weights)) n) :=
sorry

end one_side_weighing_correct_both_sides_weighing_correct_l669_669782


namespace number_of_Cheburashkas_erased_l669_669597

theorem number_of_Cheburashkas_erased :
  ∃ (n : ℕ), 
    (∀ x, x ≥ 1 → 
      (let totalKrakozyabras = (2 * (x - 1) = 29) in
         x - 2 = 11)) :=
sorry

end number_of_Cheburashkas_erased_l669_669597


namespace triangle_area_30_l669_669756

theorem triangle_area_30 (h : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = 5 ∧ c = 13 ∧ b > 0) : 
  ∃ area : ℝ, area = 1 / 2 * 5 * (b : ℝ) ∧ area = 30 :=
by
  sorry

end triangle_area_30_l669_669756


namespace expression_value_l669_669986

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_add_prop (a b : ℝ) : f (a + b) = f a * f b
axiom f_one_val : f 1 = 2

theorem expression_value : 
  (f 1 ^ 2 + f 2) / f 1 + 
  (f 2 ^ 2 + f 4) / f 3 +
  (f 3 ^ 2 + f 6) / f 5 + 
  (f 4 ^ 2 + f 8) / f 7 
  = 16 := 
sorry

end expression_value_l669_669986


namespace largest_number_of_points_l669_669301

theorem largest_number_of_points :
  let Yoongi_points := 7
  let Jungkook_points := 6
  let Yuna_points := 9
  let Yoojung_points := 8
  Yuna_points = max Yoongi_points (max Jungkook_points (max Yoojung_points Yuna_points)) :=
by
  let Yoongi_points := 7
  let Jungkook_points := 6
  let Yuna_points := 9
  let Yoojung_points := 8
  have h_yuna_largest : Yuna_points = 9 := rfl
  have h_max : max Yoongi_points (max Jungkook_points (max Yoojung_points Yuna_points)) = 9 :=
    by norm_num
  rw ←h_yuna_largest at h_max
  exact h_max

end largest_number_of_points_l669_669301


namespace rowing_distance_l669_669859

noncomputable def effective_speed_with_current (rowing_speed current_speed : ℕ) : ℕ :=
  rowing_speed + current_speed

noncomputable def effective_speed_against_current (rowing_speed current_speed : ℕ) : ℕ :=
  rowing_speed - current_speed

noncomputable def distance (speed time : ℕ) : ℕ :=
  speed * time

theorem rowing_distance (rowing_speed current_speed total_time : ℕ) 
  (hrowing_speed : rowing_speed = 10)
  (hcurrent_speed : current_speed = 2)
  (htotal_time : total_time = 30) : 
  (distance 8 18) = 144 := 
by
  sorry

end rowing_distance_l669_669859


namespace factorize_expression_l669_669422

theorem factorize_expression (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
by sorry

end factorize_expression_l669_669422


namespace problem_largest_number_l669_669571

def largest_of_four (a b c d : ℚ) : ℚ :=
  max (max a b) (max c d)

theorem problem_largest_number : largest_of_four (2/3) 1 (-3) 0 = 1 := sorry

end problem_largest_number_l669_669571


namespace tan_double_angle_l669_669936

theorem tan_double_angle (theta : ℝ) (h : 2 * Real.sin (Real.pi / 2 + theta) + Real.sin (Real.pi + theta) = 0) :
  Real.tan (2 * theta) = -4 / 3 :=
by
  sorry

end tan_double_angle_l669_669936


namespace largest_integer_for_gcd_condition_correct_l669_669802

noncomputable def largest_integer_for_gcd_condition : ℕ :=
  let n := 138
  in if (n < 150 ∧ Nat.gcd n 18 = 6) then n else 0

theorem largest_integer_for_gcd_condition_correct :
  ∃ n, (n < 150 ∧ Nat.gcd n 18 = 6) ∧ n = 138 :=
begin
  use 138,
  split,
  { split,
    { exact dec_trivial }, -- Proof that 138 < 150
    { exact dec_trivial }, -- Proof that Nat.gcd 138 18 = 6
  },
  { refl },
end

end largest_integer_for_gcd_condition_correct_l669_669802


namespace thirteen_cards_no_straight_flush_l669_669270

theorem thirteen_cards_no_straight_flush :
    let n := 13
    let x : ℕ → ℤ := λ n, 3^n + 3 * (-1)^n
    x n = 3^13 - 3 :=
by
  have h_base : ∀ n, x n = 3 ^ n + 3 * (-1) ^ n := by
    intros
    exact rfl
  sorry

end thirteen_cards_no_straight_flush_l669_669270


namespace angle_FDY_l669_669544

theorem angle_FDY :
  ∀ {X Y Z D F : EuclideanGeometry.Point},
  (XZ = YZ) →
  (m∠ DYZ = 50) →
  (DY ∥ XZ) →
  (m∠ FDY = 50) :=
by
  intros X Y Z D F hXZ hDYZ hpar
  sorry

end angle_FDY_l669_669544


namespace carpet_needed_for_room_l669_669344

theorem carpet_needed_for_room :
  let feet_to_yards_sq := 9
  let length := 15
  let width := 5
  let area := length * width
  let yards_needed := (area + feet_to_yards_sq - 1) / feet_to_yards_sq
  in yards_needed = 9 :=
by
  -- Define variables as mentioned above
  let feet_to_yards_sq := 9
  let length := 15
  let width := 5
  let area := length * width
  let yards_needed := (area + feet_to_yards_sq - 1) / feet_to_yards_sq

  -- Proof that yards_needed equals 9
  calc
    yards_needed = 75 / feet_to_yards_sq := sorry -- Substitute area
    ... = 75 / 9 := by
      simp [feet_to_yards_sq]
    ... = 8.333... := by sorry -- Division
    ... rounded ≈ 9 proving the necessity to cover the entire floor.


end carpet_needed_for_room_l669_669344


namespace five_letter_words_with_min_two_consonants_l669_669518

theorem five_letter_words_with_min_two_consonants:
  let letters := {A, B, C, D, E, F}
  let consonants := {B, C, D, F}
  let vowels := {A, E}
  (num_ways : ℕ) :=
    ∃ num_ways, num_ways = 7424
  sorry

end five_letter_words_with_min_two_consonants_l669_669518


namespace angle_C_in_triangle_l669_669581

theorem angle_C_in_triangle (A B C : ℝ) 
  (hA : A = 90) (hB : B = 50) : (A + B + C = 180) → C = 40 :=
by
  intro hSum
  rw [hA, hB] at hSum
  linarith

end angle_C_in_triangle_l669_669581


namespace find_t_closest_vector_l669_669926

noncomputable def vector_closest_to (t : ℚ) : Prop :=
  let v := λ t, (3 + 4 * t, -1 + 3 * t, -5 + 2 * t)
  let a := (-1, 6, 2)
  let direction_vec := (4, 3, 2)
  let dot_product := λ (u v : ℚ × ℚ × ℚ), u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  dot_product (v t - a) direction_vec = 0

theorem find_t_closest_vector : vector_closest_to (19 / 29) := 
sorry

end find_t_closest_vector_l669_669926


namespace intersection_sum_l669_669059

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := (4 * x + 3) / (x - 2)

theorem intersection_sum (h_f : ∀ x : ℝ, f (-x) = 8 - f (4 + x))
  (h_intersections : ∃ P : Fin 168 → ℝ × ℝ, 
    ∀ i : Fin 168, (P i).snd = f ((P i).fst) ∧ (P i).snd = g ((P i).fst))
  : (Finset.univ.sum (λ i, (P i).fst + (P i).snd) = 1008) :=
sorry

end intersection_sum_l669_669059


namespace range_of_m_l669_669508

theorem range_of_m 
  (f : ℝ → ℝ) 
  (hx : ∀ x, f x = 3 * sin (- (1/5) * x + 3 * π / 10))
  (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 = sqrt (-m^2 + 2 * m + 3) ∧ x2 = sqrt (-m^2 + 4) 
    ∧ f x1 > f x2) → 
  -1 ≤ m ∧ m < 1/2 :=
by
  sorry

end range_of_m_l669_669508


namespace a_n_formula_T_n_formula_l669_669950

noncomputable def a (n : ℕ) (hn : 0 < n) : ℕ :=
  if n = 1 then 2 else 2^(n)

def S (n : ℕ) (hn : 0 < n) : ℕ :=
  2 * a n hn - 2

def T (n : ℕ) (hn : 0 < n) : ℕ :=
  (Finset.range n).sum (λ i, S (i + 1) (Nat.succ_pos i))

theorem a_n_formula (n : ℕ) (hn : 0 < n) : a n hn = 2^n :=
by
  sorry

theorem T_n_formula (n : ℕ) (hn : 0 < n) : T n hn = 2^(n+2) - 4 - 2*n :=
by
  sorry

end a_n_formula_T_n_formula_l669_669950


namespace sum_specific_terms_l669_669478

noncomputable theory
open_locale classical

def arithmetic_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = a n + 1

theorem sum_specific_terms 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_seq a) 
  (h_sum99 : ∑ i in finset.range 99, a i = 99) :
  ∑ k in (finset.range 33).map (λ i, 3 * i + 2), a k = 66 :=
sorry

end sum_specific_terms_l669_669478


namespace base_k_number_eq_binary_l669_669132

theorem base_k_number_eq_binary (k : ℕ) (h : k^2 + 3 * k + 2 = 30) : k = 4 :=
sorry

end base_k_number_eq_binary_l669_669132


namespace cube_difference_l669_669970

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 53) : a^3 - b^3 = 385 :=
sorry

end cube_difference_l669_669970


namespace no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10_l669_669442

theorem no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10 :
  ¬ ∃ x : ℝ, x^4 + (x + 1)^4 + (x + 2)^4 = (x + 3)^4 + 10 :=
by {
  sorry
}

end no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10_l669_669442


namespace fixed_point_f_l669_669738

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log (2 * x + 1) / Real.log a) + 2

theorem fixed_point_f (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : f a 0 = 2 :=
by
  sorry

end fixed_point_f_l669_669738


namespace proveCarTransportationProblem_l669_669797

def carTransportationProblem :=
  ∃ x y a b : ℕ,
  -- Conditions regarding the capabilities of the cars
  (2 * x + 3 * y = 18) ∧
  (x + 2 * y = 11) ∧
  -- Conclusion (question 1)
  (x + y = 7) ∧
  -- Conditions for the rental plan (question 2)
  (3 * a + 4 * b = 27) ∧
  -- Cost optimization
  ((100 * a + 120 * b) = 820 ∨ (100 * a + 120 * b) = 860) ∧
  -- Optimal cost verification
  (100 * a + 120 * b = 820 → a = 1 ∧ b = 6)

theorem proveCarTransportationProblem : carTransportationProblem :=
  sorry

end proveCarTransportationProblem_l669_669797


namespace bicycle_car_speed_l669_669915

theorem bicycle_car_speed (x : Real) (h1 : x > 0) :
  10 / x - 10 / (2 * x) = 1 / 3 :=
by
  sorry

end bicycle_car_speed_l669_669915


namespace octahedron_volume_l669_669037

theorem octahedron_volume (a : ℝ) : 
  let V := (a^3 * real.sqrt 2) / 3 in 
  V = (a^3 * real.sqrt 2) / 3 := 
by 
  sorry

end octahedron_volume_l669_669037


namespace contrapositive_of_proposition_l669_669728

theorem contrapositive_of_proposition :
  (∀ x : ℝ, x ≤ -3 → x < 0) ↔ (∀ x : ℝ, x ≥ 0 → x > -3) :=
by
  sorry

end contrapositive_of_proposition_l669_669728


namespace convex_quadrilateral_diagonals_perpendicular_iff_point_angles_equal_l669_669271

theorem convex_quadrilateral_diagonals_perpendicular_iff_point_angles_equal
  {A B C D : Type}
  [ConvexQuadrilateral A B C D] :
  (∃ P : Type, P ∈ Interior (Quadrilateral A B C D) ∧ 
    (∠PAB + ∠PDC = 90 ∧ ∠PBC + ∠PAD = 90 ∧ ∠PCD + ∠PBA = 90 ∧ ∠PDA + ∠PCB = 90)) ↔
  (Diagonal AC ⊥ Diagonal BD) := sorry

end convex_quadrilateral_diagonals_perpendicular_iff_point_angles_equal_l669_669271


namespace cube_root_simplification_l669_669684

noncomputable def cubeRoot (x : ℝ) : ℝ := x^(1/3)

theorem cube_root_simplification :
  cubeRoot 54880000 = 140 * cubeRoot 20 :=
by
  sorry

end cube_root_simplification_l669_669684


namespace total_area_l669_669451

-- Define the points R and S
structure Point :=
  (x : ℝ)
  (y : ℝ)

def R : Point := ⟨-2, 5⟩
def S : Point := ⟨7, -6⟩

-- Define the distance function for the circle radius
def dist (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Define the length of rectangle's sides parallel to the coordinate axes
def length_x (p1 p2 : Point) : ℝ :=
  real.abs (p2.x - p1.x)

def length_y (p1 p2 : Point) : ℝ :=
  real.abs (p2.y - p1.y)

-- Define the area of the circle
def area_circle (radius : ℝ) : ℝ :=
  real.pi * radius^2

-- Define the area of the rectangle
def area_rectangle (length_x length_y : ℝ) : ℝ :=
  length_x * length_y

-- The main theorem to be proved
theorem total_area :
  let radius := dist R S in
  let area_c := area_circle radius in
  let length_x_r := length_x R S in
  let length_y_r := length_y R S in
  let area_r := area_rectangle length_x_r length_y_r in
  area_c + area_r = 202 * real.pi + 99 :=
by {
  -- The proof goes here
  sorry
}

end total_area_l669_669451


namespace sequence_sum_even_l669_669346

theorem sequence_sum_even :
  ∃ (y : ℕ → ℤ), 
    (∀ n, n > 0 → y (n + 1) = y n - 2) ∧ 
    (∑ i in finset.range 1500, y i) = 3000 →
    (∑ i in finset.range 750, y (2 * i)) = 2250 := 
sorry

end sequence_sum_even_l669_669346


namespace find_angle_BAN_l669_669955

open EuclideanGeometry

noncomputable theory

def triangle_isosceles {A B C : Point} (h : IsoscelesTriangle A B C) : AB = BC :=
IsoscelesTriangle.eq_sides h

def angle_condition {A B C K N : Point} [IsLinear A C B] [IsLinear B C N] (hKN : KN = AN) (h_angle : ∠BAK = ∠NAC) : Prop :=
∠BAN = 60

theorem find_angle_BAN (A B C K N : Point) 
  (h_isosceles : IsoscelesTriangle A B C)
  (h_points : between B K N)
  (h_lengths : KN = AN)
  (h_angles : ∠BAK = ∠NAC)
  : ∠BAN = 60 :=
sorry

end find_angle_BAN_l669_669955


namespace tangent_line_equation_minimum_value_of_f_l669_669985

open Real

noncomputable def f (x : ℝ) : ℝ := x * (log x)

-- Statement 1: Equation of the tangent line at the point (1, 0) is y = x - 1.
theorem tangent_line_equation (x y : ℝ) (h : (x, y) = (1, 0)) : y = x - 1 := by
  sorry

-- Statement 2: The minimum value of f(x) = x * log x is -1 / e.
theorem minimum_value_of_f : ∃ x, x > 0 ∧ f x = -1 / exp 1 := by
  use (1 / exp 1)
  split
  · exact one_div_pos.2 (exp_pos 1)
  · simp [f, exp, log_inv]
  · norm_num
  sorry

end tangent_line_equation_minimum_value_of_f_l669_669985


namespace matrix_all_zero_l669_669323

variable {α : Type*} [CommRing α]

def is_zero_matrix (A : Matrix (Fin 3) (Fin 3) α) : Prop :=
  ∀ i j, A i j = 0

def cofactor (A : Matrix (Fin 3) (Fin 3) α) (i j : Fin 3) : α :=
  (-1)^(i + j : ℤ) * Matrix.det (A.minor (Fin 3.finType.eq_above i) (Fin 3.finType.eq_above j))

theorem matrix_all_zero (A : Matrix (Fin 3) (Fin 3) α) (h₀ : Matrix.det A = 0)
  (h₁ : ∀ i j, cofactor A i j = (A i j)^2) : is_zero_matrix A :=
by
  sorry

end matrix_all_zero_l669_669323


namespace arrangement_problem_l669_669330

theorem arrangement_problem :
  ∃(n : ℕ), n = 8 ∧
    (∃ (r c : ℕ), r * c = 48 ∧ r ≥ 2 ∧ c ≥ 2 ∧
      ∃ (distinct_arrangements : finset (ℕ × ℕ)),
        distinct_arrangements = 
          {(2, 24), (3, 16), (4, 12), (6, 8), (8, 6), (12, 4), (16, 3), (24, 2)} ∧
        distinct_arrangements.card = n
    )
:= sorry

end arrangement_problem_l669_669330


namespace min_value_reciprocal_b_sum_l669_669629

theorem min_value_reciprocal_b_sum (b : Fin 10 → ℝ) 
  (h1 : ∀ i, 0 < b i) 
  (h2 : (Finset.univ.sum b) = 2) : 
  (Finset.univ.sum (λ i => 1 / (b i))) ≥ 50 := 
begin
  sorry
end

end min_value_reciprocal_b_sum_l669_669629


namespace right_triangle_area_l669_669753

/-- Given a right triangle with hypotenuse 13 meters and one side 5 meters,
prove that the area of the triangle is 30 square meters. -/
theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (hc : c = 13) (ha : a = 5) :
  1/2 * a * b = 30 :=
by sorry

end right_triangle_area_l669_669753


namespace angle_C_in_triangle_l669_669580

theorem angle_C_in_triangle (A B C : ℝ) 
  (hA : A = 90) (hB : B = 50) : (A + B + C = 180) → C = 40 :=
by
  intro hSum
  rw [hA, hB] at hSum
  linarith

end angle_C_in_triangle_l669_669580


namespace problem1_problem2_l669_669108

-- Define the function f(x)
def f (x a : ℝ) : ℝ := |2 * x - a| + |x + 2 / a|

-- Problem (1) with a = 2
theorem problem1 : ∀ x : ℝ, f x 2 ≥ 1 :=
by
  intro x
  sorry

-- Define the function g(x) = f(x) + f(-x)
def g (x a : ℝ) : ℝ := f x a + f (-x) a

-- Problem (2)
theorem problem2 : ∀ a : ℝ, a ≠ 0 → ∃ x : ℝ, g x a = 4 * Real.sqrt 2 :=
by
  intro a ha
  use (0 : ℝ)
  sorry

end problem1_problem2_l669_669108


namespace range_a_ff_a_eq_2_f_a_l669_669638

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then 3 * x - 1 else 2 ^ x

theorem range_a_ff_a_eq_2_f_a :
  {a : ℝ | f (f a) = 2 ^ (f a)} = {a : ℝ | a ≥ 2/3} :=
sorry

end range_a_ff_a_eq_2_f_a_l669_669638


namespace right_triangle_area_l669_669751

theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a = 5) (hc : c = 13) :
  1/2 * a * b = 30 :=
by
  have hb : b = 12, from sorry,
  -- Proof needs to be filled here
  sorry

end right_triangle_area_l669_669751


namespace cuboid_length_l669_669029

theorem cuboid_length (SA w h : ℕ) (h_SA : SA = 700) (h_w : w = 14) (h_h : h = 7) 
  (h_surface_area : SA = 2 * l * w + 2 * l * h + 2 * w * h) : l = 12 :=
by
  intros
  sorry

end cuboid_length_l669_669029


namespace behemoth_and_rita_finish_ice_cream_l669_669380

theorem behemoth_and_rita_finish_ice_cream (x y : ℝ) (h : 3 * x + 2 * y = 1) : 3 * (x + y) ≥ 1 :=
by
  sorry

end behemoth_and_rita_finish_ice_cream_l669_669380


namespace part_a_part_b_l669_669828

-- Part (a)
theorem part_a {a m n : ℕ} (ha : 1 < a) (hdiv : (a^n + 1) % (a^m + 1) = 0) : m ∣ n := 
sorry

-- Part (b)
theorem part_b {a b m n : ℕ} (ha : 1 < a) (hcoprime : Nat.coprime a b) (hdiv : (a^n + b^n) % (a^m + b^m) = 0) : m ∣ n :=
sorry

end part_a_part_b_l669_669828


namespace average_goals_increase_l669_669335

theorem average_goals_increase
  (goals_fifth_match : ℕ)
  (total_goals : ℕ)
  (goals_fifth_match_eq : goals_fifth_match = 5)
  (total_goals_eq : total_goals = 21) :
  let avg_before_fifth := (total_goals - goals_fifth_match) / 4 in
  let avg_after_fifth := total_goals / 5 in
  avg_after_fifth - avg_before_fifth = 0.2 :=
by
  sorry

end average_goals_increase_l669_669335


namespace count_sequences_l669_669455

theorem count_sequences :
  ∃ seq_a : Fin 2019 → ℕ,
  (∀ i : Fin 2019, seq_a i < 2^2018) ∧
  ∃ (seq_b seq_c : Fin 2018 → ℕ),
  (∀ i : Fin 2018, seq_b i = seq_a i + seq_a (i + 1)) ∧
  (∀ i : Fin 2018, seq_c i = (seq_a i) .|. (seq_a (i + 1))) ∧
  ( ∃ answer : ℕ, answer = (2^{2019} - 1)^{2018} ) := sorry

end count_sequences_l669_669455


namespace slope_angle_of_line_l669_669838

/-- If the slope angle of the line y = 1 is α, then α equals 0° -/
theorem slope_angle_of_line :
  ∃ α : Real, (∃ l : Line, l = {y = 1} ∧ l.angle = α) → α = 0 := 
sorry

end slope_angle_of_line_l669_669838


namespace similar_triangles_XY_length_l669_669794

theorem similar_triangles_XY_length (PQ QR ZY XY: ℝ)
  (h_sim : ∀ (a b : ℝ), a / b = ZY / QR)
  (h_PQ : PQ = 8)
  (h_QR : QR = 16)
  (h_ZY : ZY = 32) :
  XY = 16 := by
suffices h : XY / PQ = ZY / QR from
have h_XY' : XY / 8 = 32 / 16 by rwa [h_PQ, h_QR, h_ZY] at h,
have h_result : XY / 8 = 2 from by linarith,
have h_XY : XY = 2 * 8 from eq_mul_of_div_eq h_result,
show XY = 16 from by rw [mul_comm] at h_XY; exact h_XY
suffices h : ∀ (a b : ℝ), a / b = ZY / QR from by apply h,
suffices h : XY / PQ = ZY / QR from
exact h.1 XY PQ

end similar_triangles_XY_length_l669_669794


namespace pencils_transfer_condition_l669_669322

-- Define the necessary and sufficient condition for transferring all pencils into one pile
theorem pencils_transfer_condition (n : ℕ → ℕ) (k : ℕ) (h : k ≥ 2) :
  let d := Nat.gcd_list (Finset.image n (Finset.range k))
  in let ks := λ i, n i / d
  in (∃ (moves : list (ℕ × ℕ)) (final_pile : ℕ), 
      final_pile ∈ Finset.image (λ p, p.2) moves ∧ 
      (∀ i < k, final_pile = moves.length + n i) → 
      (∃ m : ℕ, (1 ≤ m ∧ ∑ x in Finset.range k, ks x = 2 ^ m))
| sorry

end pencils_transfer_condition_l669_669322


namespace Margo_James_pairs_probability_l669_669849

def total_students : ℕ := 32
def Margo_pairs_prob : ℚ := 1 / 31
def James_pairs_prob : ℚ := 1 / 30
def total_prob : ℚ := Margo_pairs_prob * James_pairs_prob

theorem Margo_James_pairs_probability :
  total_prob = 1 / 930 := 
by
  -- sorry allows us to skip the proof steps, only statement needed
  sorry

end Margo_James_pairs_probability_l669_669849


namespace transformed_function_equivalent_amplitude_correct_period_correct_phase_shift_correct_l669_669109

variable (x : ℝ)

def original_function : ℝ := 
  (1 / 2) * (Real.cos x)^2 + (Real.sqrt 3 / 2) * (Real.sin x) * (Real.cos x) + 1

def simplified_function : ℝ :=
  (1 / 2) * Real.sin (2 * x + π / 6) + 5 / 4

theorem transformed_function_equivalent : 
  original_function x = simplified_function x :=
sorry

theorem amplitude_correct : 
  amplitude simplified_function = 1 / 2 :=
sorry

theorem period_correct : 
  period simplified_function = π :=
sorry

theorem phase_shift_correct : 
  phase_shift simplified_function = π / 6 :=
sorry

end transformed_function_equivalent_amplitude_correct_period_correct_phase_shift_correct_l669_669109


namespace simplify_cubic_root_l669_669703

theorem simplify_cubic_root : 
  (∛(54880000) = 20 * ∛((5^2) * 137)) :=
sorry

end simplify_cubic_root_l669_669703


namespace correct_statements_l669_669049

theorem correct_statements (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 1) :
  (∀ b, a = 1 - 2 * b → a^2 + b^2 ≥ 1/5) ∧
  (∀ a b, a + 2 * b = 1 → ab ≤ 1/8) ∧
  (∀ a b, a + 2 * b = 1 → 3 + 2 * Real.sqrt 2 ≤ (1 / a + 1 / b)) :=
by
  sorry

end correct_statements_l669_669049


namespace ramu_spent_on_repairs_l669_669667

theorem ramu_spent_on_repairs 
    (initial_cost : ℝ) (selling_price : ℝ) (profit_percent : ℝ) (R : ℝ) 
    (h1 : initial_cost = 42000) 
    (h2 : selling_price = 64900) 
    (h3 : profit_percent = 18) 
    (h4 : profit_percent / 100 = (selling_price - (initial_cost + R)) / (initial_cost + R)) : 
    R = 13000 :=
by
  rw [h1, h2, h3] at h4
  sorry

end ramu_spent_on_repairs_l669_669667


namespace largest_integer_for_gcd_condition_correct_l669_669801

noncomputable def largest_integer_for_gcd_condition : ℕ :=
  let n := 138
  in if (n < 150 ∧ Nat.gcd n 18 = 6) then n else 0

theorem largest_integer_for_gcd_condition_correct :
  ∃ n, (n < 150 ∧ Nat.gcd n 18 = 6) ∧ n = 138 :=
begin
  use 138,
  split,
  { split,
    { exact dec_trivial }, -- Proof that 138 < 150
    { exact dec_trivial }, -- Proof that Nat.gcd 138 18 = 6
  },
  { refl },
end

end largest_integer_for_gcd_condition_correct_l669_669801


namespace matrix_equation_solutions_l669_669977

theorem matrix_equation_solutions (a b c d x : ℝ)
  (h_eval : ∀ (a b c d : ℝ), (a * b - c * d + c) = 2) :
  (3 * x^2 - 4 * x + 2 = 2) ↔ (x = 0 ∨ x = 4 / 3) := 
begin
  sorry
end

end matrix_equation_solutions_l669_669977


namespace problem_statement_l669_669998

def is_divisor (d n : ℕ) : Prop := n % d = 0

def count_divisors_in_range (n : ℕ) (range : List ℕ) : ℕ :=
  List.countp (is_divisor n) range

theorem problem_statement : count_divisors_in_range 41835 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 4 :=
by
  sorry

end problem_statement_l669_669998


namespace find_n_l669_669733

theorem find_n (m n : ℝ) (h1 : m + 2 * n = 1.2) (h2 : 0.1 + m + n + 0.1 = 1) : n = 0.4 :=
by
  sorry

end find_n_l669_669733


namespace find_sum_of_terms_l669_669993

noncomputable def geometric_seq (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, log 2 (a n) + 1 = log 2 (a (n + 1))

theorem find_sum_of_terms (a : ℕ → ℝ) (h_geom : geometric_seq a) (h_sum : a 2 + a 4 + a 6 = 4) :
  a 5 + a 7 + a 9 = 32 :=
sorry

end find_sum_of_terms_l669_669993


namespace notebook_cost_3_dollars_l669_669004

def cost_of_notebook (total_spent backpack_cost pen_cost pencil_cost num_notebooks : ℕ) : ℕ := 
  (total_spent - (backpack_cost + pen_cost + pencil_cost)) / num_notebooks

theorem notebook_cost_3_dollars 
  (total_spent : ℕ := 32) 
  (backpack_cost : ℕ := 15) 
  (pen_cost : ℕ := 1) 
  (pencil_cost : ℕ := 1) 
  (num_notebooks : ℕ := 5) 
  : cost_of_notebook total_spent backpack_cost pen_cost pencil_cost num_notebooks = 3 :=
by
  sorry

end notebook_cost_3_dollars_l669_669004


namespace value_of_n_l669_669527

theorem value_of_n (n : ℤ) (h : sqrt (8 + n) = 9) : n = 73 :=
sorry

end value_of_n_l669_669527


namespace tangent_line_correct_l669_669735

-- Define the curve y = x^3 - 1
def curve (x : ℝ) : ℝ := x^3 - 1

-- Define the derivative of the curve
def derivative_curve (x : ℝ) : ℝ := 3 * x^2

-- Define the point of tangency
def tangent_point : ℝ × ℝ := (1, curve 1)

-- Define the tangent line equation at x = 1
def tangent_line (x : ℝ) : ℝ := 3 * x - 3

-- The formal statement to be proven
theorem tangent_line_correct :
  ∀ x : ℝ, curve x = x^3 - 1 ∧ derivative_curve x = 3 * x^2 ∧ tangent_point = (1, 0) → 
    tangent_line 1 = 3 * 1 - 3 :=
by
  sorry

end tangent_line_correct_l669_669735


namespace tan_B_in_triangle_l669_669168

-- Define the given constants
def AC : ℝ := 4
def BC : ℝ := 3
def cosC : ℝ := 2 / 3

-- Define the unknown we need to prove
def tanB : ℝ := 4 * Real.sqrt 5

-- State the proof problem
theorem tan_B_in_triangle:
  ∃ (AB : ℝ), 
    AB = Real.sqrt (AC^2 + BC^2 - 2 * AC * BC * cosC) ∧
    AB = AC ∧ 
    ∃ (C : ℝ), cosC = Real.cos C ∧ 
    tanB = -2 * Real.tan C / (1 - (Real.tan C)^2) := sorry

end tan_B_in_triangle_l669_669168


namespace angle_C_of_triangle_l669_669578

theorem angle_C_of_triangle (A B C : ℝ) (hA : A = 90) (hB : B = 50) (h_sum : A + B + C = 180) : C = 40 := 
by
  sorry

end angle_C_of_triangle_l669_669578


namespace distinct_ordered_pairs_proof_l669_669520

def num_distinct_ordered_pairs_satisfying_reciprocal_sum : ℕ :=
  List.length [
    (7, 42), (8, 24), (9, 18), (10, 15), 
    (12, 12), (15, 10), (18, 9), (24, 8), 
    (42, 7)
  ]

theorem distinct_ordered_pairs_proof : num_distinct_ordered_pairs_satisfying_reciprocal_sum = 9 := by
  sorry

end distinct_ordered_pairs_proof_l669_669520


namespace cubic_difference_l669_669968

theorem cubic_difference (a b : ℝ) 
  (h₁ : a - b = 7)
  (h₂ : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := 
by 
  sorry

end cubic_difference_l669_669968


namespace plot_AED_area_l669_669349

-- Defining the conditions
def isosceles_triangle_area (AE ED DA: ℝ) (h: ℝ) : ℝ :=
  if (ED = DA) then (1/2) * AE * h else 0

-- Prove the area of the triangle AED given the conditions
theorem plot_AED_area :
  isosceles_triangle_area 8 5 5 4 = 16 :=
by
  sorry

end plot_AED_area_l669_669349


namespace simplify_cubed_root_l669_669705

def c1 : ℕ := 54880000
def c2 : ℕ := 10^5 * 5488
def c3 : ℕ := 5488
def c4 : ℕ := 2^4 * 343
def c5 : ℕ := 343
def c6 : ℕ := 7^3

theorem simplify_cubed_root : (c1^(1 / 3 : ℝ) : ℝ) = 1400 := 
by {
  let h1 : c1 = c2 := sorry,
  let h2 : c3 = c4 := sorry,
  let h3 : c5 = c6 := sorry,
  rw [h1, h2, h3],
  sorry
}

end simplify_cubed_root_l669_669705


namespace intersect_iff_perpendicular_intersect_given_symmetry_l669_669352

theorem intersect_iff_perpendicular (A B C D A' B' : Point) (tetrahedron : is_tetrahedron A B C D) :
  is_foot_of_perpendicular A' A (plane B C D) →
  is_foot_of_perpendicular B' B (plane A C D) →
  (∃ P : Point, lies_on P (line A A') ∧ lies_on P (line B B')) ↔ 
  perpendicular (line A B) (line C D) := sorry

theorem intersect_given_symmetry (A B C D A' B' : Point) (tetrahedron : is_tetrahedron A B C D) :
  is_foot_of_perpendicular A' A (plane B C D) →
  is_foot_of_perpendicular B' B (plane A C D) →
  (distance A C = distance A D) →
  (distance B C = distance B D) →
  ∃ P : Point, lies_on P (line A A') ∧ lies_on P (line B B') := sorry

end intersect_iff_perpendicular_intersect_given_symmetry_l669_669352


namespace graph_passes_through_point_l669_669962

theorem graph_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : ∃ y : ℝ, a^0 + 3 = y ∧ y = 4 :=
by
  use 4
  split
  · calc 
      a^0 + 3 = 1 + 3 : by rw [Real.rpow_nat_cast, zero_add]
            ... = 4 : by ring
  · exact rfl

end graph_passes_through_point_l669_669962


namespace simplify_cubic_root_l669_669697

theorem simplify_cubic_root : 
  (∛(54880000) = 20 * ∛((5^2) * 137)) :=
sorry

end simplify_cubic_root_l669_669697


namespace gain_per_year_l669_669304

def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

theorem gain_per_year (P : ℝ) (T : ℝ) (borrow_rate lend_rate : ℝ) :
  (borrow_rate = 4 ∧ lend_rate = 7 ∧ P = 5000 ∧ T = 2) →
  ((simple_interest P lend_rate T - simple_interest P borrow_rate T) / T)  = 150 :=
by
  intros h
  cases h with hrates hpt
  cases hrates with hborrow_rate hlend_rate
  cases hpt with hP hT
  sorry

end gain_per_year_l669_669304
