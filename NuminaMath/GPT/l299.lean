import Mathlib
import Mathlib.Algebra.AbsoluteValue
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Monad
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Parity
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.QuadraticEquation
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Calculus.TangentDeriv
import Mathlib.Analysis.Special.Theorems
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Fib
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Function
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.MeasureTheory.IntervalIntegral
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Real.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.LibrarySearch
import Mathlib.Topology.Basic

namespace initial_salt_quantity_l299_299621

variable (x : ℝ)
axiom h1 : 0.4 * x + 420 = 3 * (x - 0.4 * x - 420)

theorem initial_salt_quantity : x = 1200 :=
by 
  have h2 : x - 0.4 * x - 420 = 0.6 * x - 420 := by ring
  rw [h2] at h1
  linarith

end initial_salt_quantity_l299_299621


namespace solve_quadratic_l299_299418

theorem solve_quadratic (x : ℝ) : x^2 - 2*x = 0 ↔ (x = 0 ∨ x = 2) :=
by
  sorry

end solve_quadratic_l299_299418


namespace largest_A_smallest_A_l299_299843

noncomputable def is_coprime_with_12 (n : Nat) : Prop :=
  Nat.gcd n 12 = 1

noncomputable def rotated_number (n : Nat) : Option Nat :=
  if n < 10^7 then none else
  let b := n % 10
  let k := n / 10
  some (b * 10^7 + k)

noncomputable def satisfies_conditions (B : Nat) : Prop :=
  B > 44444444 ∧ is_coprime_with_12 B

theorem largest_A :
  ∃ (B : Nat), satisfies_conditions B ∧ rotated_number B = some 99999998 :=
sorry

theorem smallest_A :
  ∃ (B : Nat), satisfies_conditions B ∧ rotated_number B = some 14444446 :=
sorry

end largest_A_smallest_A_l299_299843


namespace _l299_299338

noncomputable def exists_fixed_point_in_interval (f : ℝ → ℝ) (a b : ℝ) :
  continuous_on f (set.Icc a b) → (f a < a) → (f b > b) → ∃ ξ ∈ set.Ioo a b, f ξ = ξ :=
by
  intros h_cont h_left h_right
  let F := λ x, f x - x
  have hF_cont : continuous_on F (set.Icc a b),
  from continuous_on.sub h_cont continuous_on_id
  have hFa : F a < 0,
  from by
    calc F a = f a - a : rfl
         ... < 0 : by linarith [h_left]
  have hFb : F b > 0,
  from by
    calc F b = f b - b : rfl
         ... > 0 : by linarith [h_right]
  have h_sign_change : F a * F b < 0,
  from by linarith [hFa, hFb]
  obtain ⟨ξ, hξ1, hξ2⟩ := intermediate_value_theorem F hF_cont a b ⟨hFa, hFb⟩
  use ξ
  constructor
  · exact hξ1
  · exact hξ2.symm

end _l299_299338


namespace lateral_surface_area_of_cone_l299_299215

theorem lateral_surface_area_of_cone (diameter height : ℝ) (h_d : diameter = 2) (h_h : height = 2) :
  let radius := diameter / 2
  let slant_height := Real.sqrt (radius ^ 2 + height ^ 2)
  π * radius * slant_height = Real.sqrt 5 * π := 
  by
    sorry

end lateral_surface_area_of_cone_l299_299215


namespace two_digit_numbers_l299_299971

theorem two_digit_numbers (x y : ℕ) (h1 : 1 ≤ x ∧ x ≤ 9) (h2 : 1 ≤ y ∧ y ≤ 9) :
  (x + y) % 3 = 0 →
  10 * x + y - 27 = 10 * y + x →
  {10 * x + y} = {63, 96} :=
by
  sorry

end two_digit_numbers_l299_299971


namespace time_for_B_to_complete_work_l299_299805

theorem time_for_B_to_complete_work 
  (A B C : ℝ)
  (h1 : A = 1 / 4) 
  (h2 : B + C = 1 / 3) 
  (h3 : A + C = 1 / 2) :
  1 / B = 12 :=
by
  -- Proof is omitted, as per instruction.
  sorry

end time_for_B_to_complete_work_l299_299805


namespace not_necessarily_divisor_sixty_four_l299_299733

theorem not_necessarily_divisor_sixty_four (k : ℤ) (h : (k * (k + 1) * (k + 2)) % 8 = 0) :
  ¬ ((k * (k + 1) * (k + 2)) % 64 = 0) := 
sorry

end not_necessarily_divisor_sixty_four_l299_299733


namespace percent_of_companyA_l299_299466

noncomputable theory

variables (A B : ℝ)
variable hA : 0.10 * A + 0.30 * B = 0.25 * (A + B)

theorem percent_of_companyA (h_merge: B = 3 * A) : (A / (A + B)) * 100 = 25 :=
by
  rw h_merge
  calc
    (A / (A + 3 * A)) * 100
      = (A / (4 * A)) * 100 : by rw add_mul
  ... = (1 / 4) * 100       : by rw [mul_comm, div_mul_cancel _ (by linarith)]
  ... = 25                  : by norm_num

-- Sorry to skip the proof steps as instructed

end percent_of_companyA_l299_299466


namespace compute_sum_D_neg_l299_299317

def S (N : ℕ) : ℕ := (Nat.binaryRep N).count 1

def D (N : ℕ) : ℕ := S (N + 1) - S N

theorem compute_sum_D_neg (sum_D_neg : ℕ) :
  (sum_D_neg = ∑ n in Finset.filter (λ n => 1 ≤ n ∧ n ≤ 2017 ∧ D n < 0) (Finset.range 2018), D n) → 
  sum_D_neg = -1002 := 
sorry

end compute_sum_D_neg_l299_299317


namespace mark_performance_length_l299_299343

theorem mark_performance_length :
  ∃ (x : ℕ), (x > 0) ∧ (6 * 5 * x = 90) ∧ (x = 3) :=
by
  sorry

end mark_performance_length_l299_299343


namespace residue_at_zero_l299_299174

noncomputable def f (z : ℂ) : ℂ := complex.cos z * complex.sin (1 / z)

theorem residue_at_zero :
  complex.residue f 0 = ∑ n : ℕ, 1 / ((2 * n)! * (2 * n + 1)!) :=
by
  sorry

end residue_at_zero_l299_299174


namespace function_value_sum_l299_299228

def f : ℝ → ℝ :=
  λ x, if x ≥ 0 then x * (x + 4) else x * (x - 4)

theorem function_value_sum : f 1 + f (-3) = 26 := 
by {
  -- Here you would normally write the proof
  sorry
}

end function_value_sum_l299_299228


namespace ratio_second_third_l299_299029

theorem ratio_second_third (S T : ℕ) (h_sum : 200 + S + T = 500) (h_third : T = 100) : S / T = 2 := by
  sorry

end ratio_second_third_l299_299029


namespace probability_exactly_5_heads_in_7_flips_l299_299045

theorem probability_exactly_5_heads_in_7_flips : 
  let outcome_space := (finset.range 2).product (finset.range 2).product (finset.range 2).product (finset.range 2).product (finset.range 2).product (finset.range 2).product (finset.range 2),
      heads := λ (outcome : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ), (outcome.1 + outcome.2.1 + outcome.2.2.1 + outcome.2.2.2.1 + outcome.2.2.2.2.1 + outcome.2.2.2.2.2.1 + outcome.2.2.2.2.2.2) = 5 in
  (outcome_space.filter heads).card / outcome_space.card = 21 / 128 := 
by
  sorry

end probability_exactly_5_heads_in_7_flips_l299_299045


namespace largest_and_smallest_A_l299_299842

noncomputable def is_coprime_with_12 (n : ℕ) : Prop := 
  Nat.gcd n 12 = 1

def problem_statement (A_max A_min : ℕ) : Prop :=
  ∃ B : ℕ, B > 44444444 ∧ is_coprime_with_12 B ∧
  (A_max = 9 * 10^7 + (B - 9) / 10) ∧
  (A_min = 1 * 10^7 + (B - 1) / 10)

theorem largest_and_smallest_A :
  problem_statement 99999998 14444446 := sorry

end largest_and_smallest_A_l299_299842


namespace Ceva_Ratio_Ratio_Points_l299_299852

variables
  {A B C H D E F G O P : Type} [EuclideanGeometry]  -- Assuming Euclidean geometry context
  (triangle_ABC : Triangle A B C)                   -- Triangle ABC
  (H_interior : H ∈ interior (triangle_ABC))        -- H is an interior point of triangle ABC
  (D_inter : Line (A, H) ∩ Line (B, C) = {D})       -- Intersection of AH and BC is D
  (E_inter : Line (B, H) ∩ Line (C, A) = {E})       -- Intersection of BH and CA is E
  (F_inter : Line (C, H) ∩ Line (A, B) = {F})       -- Intersection of CH and AB is F
  (G_inter : Line (F, E) ∩ Line (B, C) = {G})       -- Intersection of FE and BC is G
  (O_midpoint : midpoint O D G)                     -- O is midpoint of DG
  (circle_O : circle O (distance O D))              -- Circle with center O and radius OD
  (P_on_FE : P ∈ (circle_O ∩ Line (F, E)))          -- P is on the intersection of the circle and FE

-- Prove part 1
theorem Ceva_Ratio : (BD/DC) = (BG/GC) := sorry

-- Prove part 2
theorem Ratio_Points : (PB/PC) = (BD/DC) := sorry

end Ceva_Ratio_Ratio_Points_l299_299852


namespace part1_part2_l299_299954

set_option linter.unusedVariables false

-- Conditions: sequence definition and initial term
variable {a : ℕ → ℝ}
variable h_seq : ∀ n ≥ 2, a (n-1) - a n = a n * a (n-1)
variable h_a1 : a 1 = 1

-- First part
theorem part1 (h_seq h_a1) :
  (∀ n, 1 ≤ n → (1 / a n) = n) :=
sorry

-- Second part
def b (n : ℕ) : ℝ := (2^(n-1)) / a n

def T (n : ℕ) : ℝ := ∑ i in Finset.range n, b (i + 1)

theorem part2 (h_seq h_a1) :
  ∀ n, T n = (n - 1) * 2^n + 1 :=
sorry

end part1_part2_l299_299954


namespace trig_expression_value_l299_299906

variable (α : Real)
hypothesis tan_alpha_eq : Real.tan α = 2

theorem trig_expression_value :
  3 * Real.sin α ^ 2 - Real.cos α * Real.sin α + 1 = 3 :=
sorry

end trig_expression_value_l299_299906


namespace physics_class_size_l299_299116

theorem physics_class_size (total_students physics_only math_only both : ℕ) 
  (h1 : total_students = 100)
  (h2 : physics_only + math_only + both = total_students)
  (h3 : both = 10)
  (h4 : physics_only + both = 2 * (math_only + both)) :
  physics_only + both = 62 := 
by sorry

end physics_class_size_l299_299116


namespace train_crossing_bridge_time_l299_299968

def length_of_train : ℝ := 110
def speed_of_train_kmh : ℝ := 72
def length_of_bridge : ℝ := 132
def kmh_to_mps (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600

theorem train_crossing_bridge_time :
  let total_distance := length_of_train + length_of_bridge
  let speed_of_train := kmh_to_mps speed_of_train_kmh
  let time := total_distance / speed_of_train
  time = 12.1 :=
by
  -- Proof goes here
  sorry

end train_crossing_bridge_time_l299_299968


namespace max_unique_mature_worms_l299_299038

-- Definitions to represent the conditions
def initial_worm : list string := ["Head", "W"]

def split_segment (segment : string) : list string :=
  match segment with
  | "W" => ["W", "G"]
  | "G" => ["G", "W"]
  | _ => []

-- Function to generate the day 2 worm
def day2_worm (worm : list string) : list (list string) :=
  [worm ++ split_segment (worm.getLast (by simp))]

-- Function to generate worm configurations for day 3
def day3_worms (worm : list string) : list (list string) :=
  worm.mapWithIndex (λ idx segment =>
    let prefix := worm.take (idx + 1)
    let split := split_segment segment
    let suffix := worm.drop (idx + 1)
    prefix ++ split ++ suffix)

-- Function to generate worm configurations for day 4
def day4_worms (worms : list (list string)) : list (list string) :=
  worms.bind (λ worm => worm.mapWithIndex (λ idx segment =>
    let prefix := worm.take (idx + 1)
    let split := split_segment segment
    let suffix := worm.drop (idx + 1)
    prefix ++ split ++ suffix))

-- Function to count unique worms
def unique_worm_count (worms : list (list string)) : ℕ :=
  worms.eraseDuplicates.length

-- Statement of the theorem
theorem max_unique_mature_worms : unique_worm_count (day4_worms (day3_worms (day2_worm initial_worm))) = 4 :=
by
  sorry

end max_unique_mature_worms_l299_299038


namespace arrange_subsets_sequence_l299_299567

open Set

-- Definitions: Sets A and B, Symmetric difference, and Set S
variable {A B : Set ℕ}

def symmetric_difference (A B : Set ℕ) : Set ℕ :=
  (A \ B) ∪ (B \ A)

def S (n : ℕ) : Set ℕ := {x | x ∈ Finset.range (n + 1)}

-- The problem setup
variable (n : ℕ)
variable (hn : n > 3)

-- Main statement: proving the possibility of required arrangement
theorem arrange_subsets_sequence :
  ∃ sequence : List (Set ℕ), (∀ (i j : ℕ) (hi : i < sequence.length) (hj : j < sequence.length), |symmetric_difference (sequence.nth_le i hi) (sequence.nth_le j hj)| = 3) :=
sorry

end arrange_subsets_sequence_l299_299567


namespace problem_divisibility_l299_299320

theorem problem_divisibility (k : ℕ) (hk : k > 1) (p : ℕ) (hp : p = 6 * k + 1) (hprime : Prime p) 
  (m : ℕ) (hm : m = 2^p - 1) : 
  127 * m ∣ 2^(m - 1) - 1 := 
sorry

end problem_divisibility_l299_299320


namespace geometric_sequence_condition_l299_299284

def arithmetic_sequence (a1 d : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d

def b_n (a : ℕ → ℕ) (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

def T_n (b : ℕ → ℚ) (n : ℕ) : ℚ := (finset.range (n + 1)).sum b

theorem geometric_sequence_condition :
  let a_n := arithmetic_sequence 2 3
  let b_n := λ n, 1 / (a_n n * a_n (n + 1))
  let T_1 := T_n b_n 1
  let T_m := T_n b_n m
  let T_n := T_n b_n n
  1 < m → m < n → 
  T_1 * T_n n = T_m ^ 2 →
  m = 2 ∧ n = 10 :=
begin
  sorry -- Proof is omitted
end

end geometric_sequence_condition_l299_299284


namespace sum_cubes_mod_6_l299_299173

theorem sum_cubes_mod_6 :
  (∑ i in Finset.range 99, i^3) % 6 = 5 := by
begin
  sorry
end

end sum_cubes_mod_6_l299_299173


namespace determine_e_l299_299407

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := 3*x^3 + d*x^2 + e*x + f

theorem determine_e (d f : ℝ) (h1 : f = 18) (h2 : -f/3 = -6) (h3 : -d/3 = -6) (h4 : 3 + d + e + f = -6) : e = -45 :=
sorry

end determine_e_l299_299407


namespace pencil_distribution_l299_299765

theorem pencil_distribution (total_pens : ℕ) (total_pencils : ℕ) (max_students : ℕ) 
  (h1 : total_pens = 1001) (h2 : total_pencils = 910) (h3 : max_students = 91) : 
  total_pencils / max_students = 10 :=
by
  sorry

end pencil_distribution_l299_299765


namespace perp_circles_in_acute_triangle_l299_299283

-- Main definition of the problem
theorem perp_circles_in_acute_triangle
  {A B C A1 B1 : Type*} [triangle ABC] :
  acute_angled_triangle ABC → 
  is_altitude A A1 ABC →
  is_altitude B B1 ABC →
  ∃ k1 k2 : circle, has_diameter k2 A B ∧ circumcircle k1 A1 B1 C 
  → perpendicular_intersect k1 k2 := 
begin
  sorry
end

end perp_circles_in_acute_triangle_l299_299283


namespace evaluate_ff_neg10_l299_299950

noncomputable def f : ℝ → ℝ := 
λ x, if x ≥ 0 then 2^(x - 2) else Real.log (-x)

theorem evaluate_ff_neg10 : f (f (-10)) = 1/2 := by
  sorry

end evaluate_ff_neg10_l299_299950


namespace expected_digits_icosahedral_die_l299_299890

theorem expected_digits_icosahedral_die :
  let faces := (1 : Finset ℕ).filter (λ n, n ≤ 20) in
  let one_digit_faces := faces.filter (λ n, n < 10) in
  let two_digit_faces := faces.filter (λ n, n ≥ 10) in
  let probability_one_digit := (one_digit_faces.card : ℚ) / faces.card in
  let probability_two_digit := (two_digit_faces.card : ℚ) / faces.card in
  let expected_digits := (probability_one_digit * 1) + (probability_two_digit * 2) in
  expected_digits = 31 / 20 := sorry

end expected_digits_icosahedral_die_l299_299890


namespace least_b_value_proof_l299_299328

noncomputable def least_value_of_b (a b : ℕ) : ℕ :=
  if h1 : ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ 
                (∃ p q : ℕ, (p.prime ∧ q.prime ∧ (a = p^3 ∨ a = p * q))) ∧ 
                (∃ k : ℕ, b = k ∧ 2 * a = k) ∧
                b % a = 0
  then b
  else 0

theorem least_b_value_proof : ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ 
                                    (∃ p q : ℕ, (p.prime ∧ q.prime ∧ (a = p^3 ∨ a = p * q))) ∧ 
                                    (∃ k : ℕ, b = k ∧ 2 * a = k) ∧
                                    b % a = 0 ∧
                                    least_value_of_b a b = 60 := 
by
  sorry

end least_b_value_proof_l299_299328


namespace number_of_hard_drives_sold_l299_299492

theorem number_of_hard_drives_sold 
    (H : ℕ)
    (price_per_graphics_card : ℕ := 600)
    (price_per_hard_drive : ℕ := 80)
    (price_per_cpu : ℕ := 200)
    (price_per_ram_pair : ℕ := 60)
    (graphics_cards_sold : ℕ := 10)
    (cpus_sold : ℕ := 8)
    (ram_pairs_sold : ℕ := 4)
    (total_earnings : ℕ := 8960)
    (earnings_from_graphics_cards : graphics_cards_sold * price_per_graphics_card = 6000)
    (earnings_from_cpus : cpus_sold * price_per_cpu = 1600)
    (earnings_from_ram : ram_pairs_sold * price_per_ram_pair = 240)
    (earnings_from_hard_drives : H * price_per_hard_drive = 80 * H) :
  graphics_cards_sold * price_per_graphics_card +
  cpus_sold * price_per_cpu +
  ram_pairs_sold * price_per_ram_pair +
  H * price_per_hard_drive = total_earnings → H = 14 :=
by
  intros h
  sorry

end number_of_hard_drives_sold_l299_299492


namespace perpendicular_to_base_of_triangle_reflection_l299_299669

open EuclideanGeometry

theorem perpendicular_to_base_of_triangle_reflection
  {A B C D X Y : Point} (hABC : Triangle A B C) 
  (hX : midpoint X A B) (hY : midpoint Y A C)
  (hD : D ≠ midpoint B C) (hAngle : ∠XDY = ∠BAC) :
  perpendicular AD BC :=
by
  sorry

end perpendicular_to_base_of_triangle_reflection_l299_299669


namespace expression_equals_neg_four_l299_299126

noncomputable def calc_expression : ℝ :=
  (π - 2)^0 - 2 * real.sqrt 3 * 2⁻¹ - real.sqrt 16 + abs (1 - real.sqrt 3)

theorem expression_equals_neg_four : calc_expression = -4 :=
by
  sorry

end expression_equals_neg_four_l299_299126


namespace toll_formula_l299_299759

theorem toll_formula (x : ℕ) (t : ℝ) (e : ℕ → ℝ) : 
  (∀ (y : ℕ), 2 + 4 * y = 18 → 1 + y = x) →
  (∀ (t₀ : ℝ), t₀ = 2 → (t₀ = 0.50 + e x) → t = t₀) →
  e x = 0.30 * x :=
by 
  intro h₁ h₂
  specialize h₁ 4
  have h_axles := h₁ (by norm_num)
  rw [h_axles] at h₂
  sorry

end toll_formula_l299_299759


namespace cube_moves_l299_299090

theorem cube_moves (initial_pos : (ℝ × ℝ × ℝ)) (initial_orientation : (string)) :
  ∃ final_pos : (ℝ × ℝ × ℝ), (initial_pos = final_pos) ∧ (initial_orientation ≠ "bottom") :=
by {
  sorry
}

end cube_moves_l299_299090


namespace negation_P_l299_299204

def P : Prop := ∀ x > 0, exp x ≥ 1

theorem negation_P : ¬P ↔ ∃ x₀ > 0, exp x₀ < 1 :=
by 
  sorry

end negation_P_l299_299204


namespace acute_angle_lambda_range_l299_299207

noncomputable def is_acute_angle (a b : ℝ × ℝ) : Prop := 
  a.1 * b.1 + a.2 * b.2 > 0

noncomputable def valid_lambda_range (λ : ℝ) : Prop := 
  (λ < 1 / 2) ∧ λ ≠ -2

theorem acute_angle_lambda_range (λ : ℝ):
  let i : ℝ × ℝ := (1, 0)
  let j : ℝ × ℝ := (0, 1)
  let a := (1, -2)
  let b := (1, λ)
  (is_acute_angle a b) ↔ (valid_lambda_range λ) :=
by
  sorry

end acute_angle_lambda_range_l299_299207


namespace numberOfValuesSatisfyingProperties_l299_299813

noncomputable def numberOfSolutions (z : ℂ) (f : ℂ → ℂ) (h : |z| = 7 ∧ f(z) = z) := 2

theorem numberOfValuesSatisfyingProperties
  (f : ℂ → ℂ)
  (hf : ∀ z : ℂ, f(z) = Complex.I * Complex.conj(z))
  (h : ∀ z : ℂ, |z| = 7 → (f(z) = z ↔ (z = Complex.sqrt(Complex.I * Complex.sqrt(2) * 49)) ∨ (z = -Complex.sqrt(Complex.I * Complex.sqrt(2) * 49)))) :
  ∃ z : ℂ, h z -> numberOfSolutions z f h = 2 := by
  sorry

end numberOfValuesSatisfyingProperties_l299_299813


namespace markup_percentage_l299_299097

theorem markup_percentage {C : ℝ} (hC0: 0 < C) (h1: 0 < 1.125 * C) : 
  ∃ (x : ℝ), 0.75 * (1.20 * C * (1 + x / 100)) = 1.125 * C ∧ x = 25 := 
by
  have h2 : 1.20 = (6 / 5 : ℝ) := by norm_num
  have h3 : 0.75 = (3 / 4 : ℝ) := by norm_num
  sorry

end markup_percentage_l299_299097


namespace Z_is_real_Z_is_pure_imaginary_Z_is_in_first_quadrant_l299_299703

def Z_real (m : ℝ) : Prop :=
  m = -1 ∨ m = -2

def Z_pure_imaginary (m : ℝ) : Prop :=
  m = 3 ∨ m = -1

def Z_first_quadrant (m : ℝ) : Prop :=
  m < -2 ∨ m > 3

theorem Z_is_real (m : ℝ) : 
  (Z_real m) ↔ (m^2 + 3*m + 2 = 0 ∧ m^2 - 2*m - 2 > 0) :=
sorry 

theorem Z_is_pure_imaginary (m : ℝ) :
  (Z_pure_imaginary m) ↔ (Real.log (m^2 - 2*m - 2) = 0 ∧ m^2 + 3*m + 2 ≠ 0) :=
sorry

theorem Z_is_in_first_quadrant (m : ℝ) :
  (Z_first_quadrant m) ↔ (Real.log (m^2 - 2*m - 2) > 0 ∧ m^2 + 3*m + 2 > 0) :=
sorry

end Z_is_real_Z_is_pure_imaginary_Z_is_in_first_quadrant_l299_299703


namespace real_imaginary_product_l299_299220

def Z := (1 : ℂ) + I
def W := (2 : ℂ) - I
def Z_mul := Z * W

theorem real_imaginary_product : (Z_mul.re * Z_mul.im = 3) :=
by
  -- sorry can be used to skip the proof step
  sorry

end real_imaginary_product_l299_299220


namespace square_form_l299_299977

theorem square_form (m n : ℤ) : 
  ∃ k l : ℤ, (2 * m^2 + n^2)^2 = 2 * k^2 + l^2 :=
by
  let x := (2 * m^2 + n^2)
  let y := x^2
  let k := 2 * m * n
  let l := 2 * m^2 - n^2
  use k, l
  sorry

end square_form_l299_299977


namespace A_inter_B_eq_set_l299_299236

-- Define the sets A and B according to the given conditions
def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℚ := {x | -1 < x ∧ x < 2}

-- State the theorem to prove that A ∩ B = {0, 1}
theorem A_inter_B_eq_set : (A ∩ B) = {0, 1} :=
by
  sorry

end A_inter_B_eq_set_l299_299236


namespace prop2_prop4_l299_299592

variables {Line Plane : Type} [has_parallel Line Plane] [has_perpendicular Line Plane]

-- Given conditions
variable (m n : Line)
variable (α β : Plane)
variable (differentLines : m ≠ n)
variable (differentPlanes : α ≠ β)

-- Proposition 2: If m ⊥ α and m ∥ β, then α ⊥ β.
theorem prop2 (h1 : m ⊥ α) (h2 : m ∥ β) : α ⊥ β := sorry

-- Proposition 4: If m ⊥ α and α ∥ β, then m ⊥ β.
theorem prop4 (h1 : m ⊥ α) (h2 : α ∥ β) : m ⊥ β := sorry

end prop2_prop4_l299_299592


namespace unique_a_for_system_solution_l299_299556

-- Define the variables
variables (a b x y : ℝ)

-- Define the system of equations
def system_has_solution (a b : ℝ) : Prop :=
  ∃ x y : ℝ, 2^(b * x) + (a + 1) * b * y^2 = a^2 ∧ (a-1) * x^3 + y^3 = 1

-- Main theorem statement
theorem unique_a_for_system_solution :
  a = -1 ↔ ∀ b : ℝ, system_has_solution a b :=
sorry

end unique_a_for_system_solution_l299_299556


namespace carla_drive_distance_l299_299133

theorem carla_drive_distance
    (d1 d3 : ℕ) (gpm : ℕ) (gas_price total_cost : ℕ) 
    (x : ℕ)
    (hx : 2 * gas_price = 1)
    (gallon_cost : ℕ := total_cost / gas_price)
    (total_distance   : ℕ := gallon_cost * gpm)
    (total_errand_distance : ℕ := d1 + x + d3 + 2 * x)
    (h_distance : total_distance = total_errand_distance) :
  x = 10 :=
by
  -- begin
  -- proof construction
  sorry

end carla_drive_distance_l299_299133


namespace elena_probability_at_least_one_correct_l299_299155

-- Conditions
def total_questions := 30
def choices_per_question := 4
def guessed_questions := 6
def incorrect_probability_single := 3 / 4

-- Expression for the probability of missing all guessed questions
def probability_all_incorrect := (incorrect_probability_single) ^ guessed_questions

-- Calculation from the solution
def probability_at_least_one_correct := 1 - probability_all_incorrect

-- Problem statement to prove
theorem elena_probability_at_least_one_correct : probability_at_least_one_correct = 3367 / 4096 :=
by sorry

end elena_probability_at_least_one_correct_l299_299155


namespace shaded_area_correct_l299_299291

-- Let square_4x4 represent the 4-inch by 4-inch square
def square_4x4_area := 4 * 4

-- Let square_10x10 represent the 10-inch by 10-inch square
def HF := 14
def AH := 10
def HF_area := HF * HF

-- Compute the side ratio from the similar triangles
def side_ratio := (AH: ℝ) / (HF: ℝ) -- 10/14 = 5/7

-- Given that GF is 4 inches
def GF := 4

-- Compute DG using the ratio
def DG := (side_ratio * GF : ℝ) -- 20/7 inches

-- Compute the area of triangle DGF
def triangle_DGF_area := (1 / 2) * DG * GF

-- Compute the area of the shaded region
def shaded_area := square_4x4_area - triangle_DGF_area

-- The area of the shaded region should be 72/7 square inches
theorem shaded_area_correct : shaded_area = 72 / 7 := by
  simp [square_4x4_area, DG, triangle_DGF_area, shaded_area]
  sorry

end shaded_area_correct_l299_299291


namespace num_factors_and_sum_of_factors_of_60_l299_299626

theorem num_factors_and_sum_of_factors_of_60 :
  let n := 60
  let pf := (2^2 * 3^1 * 5^1)
  let num_factors := ∏ p in [2^0, 2^1, 2^2, 3^0, 3^1, 5^0, 5^1], count_factors pf p
  let sum_factors := ∑ p in [2^0, 2^1, 2^2, 3^0, 3^1, 5^0, 5^1], sum_factors pf p
  find_factors_and_sum n = (12, 198.375) :=
  sorry

end num_factors_and_sum_of_factors_of_60_l299_299626


namespace correct_average_marks_l299_299821

theorem correct_average_marks :
  ∀ (n : ℕ) (incorrect_avg : ℝ) (incorrect_marks : ℕ → ℝ)
    (correct_marks : ℕ → ℝ) (students : ℕ),
    n = 60 →
    incorrect_avg = 82 →
    incorrect_marks 0 = 68 →
    correct_marks 0 = 78 →
    incorrect_marks 1 = 91 →
    correct_marks 1 = 95 →
    incorrect_marks 2 = 74 →
    correct_marks 2 = 84 →
    students = 60 →
    ((incorrect_avg * n + ((correct_marks 0 - incorrect_marks 0) +
                           (correct_marks 1 - incorrect_marks 1) +
                           (correct_marks 2 - incorrect_marks 2))) / students) = 82.40 :=
begin
  intros n incorrect_avg incorrect_marks correct_marks students h1 h2 h3 h4 h5 h6 h7 h8 h9,
  rw [h1, h2, h3, h4, h5, h6, h7, h8, h9],
  sorry
end

end correct_average_marks_l299_299821


namespace area_of_circle_l299_299715

-- Define points C and D as specified in the conditions
def C : ℝ × ℝ := (-1, 4)
def D : ℝ × ℝ := (3, 11)

-- Define a function to calculate the area of the circle given these conditions
def circle_area (C D : ℝ × ℝ) : ℝ := 
  let diameter := real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) in
  let radius := diameter / 2 in
  π * radius^2

theorem area_of_circle :
  circle_area C D = 65 * π / 4 := by
  sorry

end area_of_circle_l299_299715


namespace problem_statement_l299_299569

theorem problem_statement (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) 
    (h3 : m + 5 < n) 
    (h4 : (m + 3 + m + 7 + m + 13 + n + 4 + n + 5 + 2 * n + 3) / 6 = n + 3)
    (h5 : (↑((m + 13) + (n + 4)) / 2 : ℤ) = n + 3) : 
  m + n = 37 :=
by
  sorry

end problem_statement_l299_299569


namespace angle_BOC_l299_299265

theorem angle_BOC (A B C E F O : Type) [HasAngle A B C] [HasAngle A B E] [HasAngle E F O] [HasInteriorAngle A] 
  (h_A : ∠A = 50) [HeightFrom BE B O] [HeightFrom CF C O] :
  ∠BOC = 50 ∨ ∠BOC = 130 := 
sorry

end angle_BOC_l299_299265


namespace maria_distance_after_second_stop_l299_299885

-- Define the main entities and values from the problem.
def total_distance : ℝ := 280
def first_stop_distance : ℝ := total_distance / 2
def remaining_distance_after_first_stop : ℝ := total_distance - first_stop_distance
def second_stop_distance : ℝ := remaining_distance_after_first_stop / 4
def remaining_distance_after_second_stop : ℝ := remaining_distance_after_first_stop - second_stop_distance

theorem maria_distance_after_second_stop :
  remaining_distance_after_second_stop = 105 :=
by
  -- Place your proof here, fulfilling the theorem's statement.
  sorry

end maria_distance_after_second_stop_l299_299885


namespace exp_equiv_l299_299904

/-- Given conditions: -/
variables (m n : ℝ)
variable (h1 : 3^m = 8)
variable (h2 : 3^n = 2)

/-- Theorem that needs to be proven: -/
theorem exp_equiv : 3^(2 * m - 3 * n + 1) = 24 :=
by {
  sorry
}

end exp_equiv_l299_299904


namespace edward_work_hours_edward_work_hours_overtime_l299_299064

variable (H : ℕ) -- H represents the number of hours worked
variable (O : ℕ) -- O represents the number of overtime hours

theorem edward_work_hours (H_le_40 : H ≤ 40) (earning_eq_210 : 7 * H = 210) : H = 30 :=
by
  -- Proof to be filled in here
  sorry

theorem edward_work_hours_overtime (H_gt_40 : H > 40) (earning_eq_210 : 7 * 40 + 14 * (H - 40) = 210) : False :=
by
  -- Proof to be filled in here
  sorry

end edward_work_hours_edward_work_hours_overtime_l299_299064


namespace shirts_total_cost_l299_299680

def shirt_cost_problem : Prop :=
  ∃ (first_shirt_cost second_shirt_cost total_cost : ℕ),
    first_shirt_cost = 15 ∧
    first_shirt_cost = second_shirt_cost + 6 ∧
    total_cost = first_shirt_cost + second_shirt_cost ∧
    total_cost = 24

theorem shirts_total_cost : shirt_cost_problem := by
  sorry

end shirts_total_cost_l299_299680


namespace mean_of_remaining_students_l299_299088

variable (k : ℕ) (h1 : k > 20)

def mean_of_class (mean : ℝ := 10) := mean
def mean_of_20_students (mean : ℝ := 16) := mean

theorem mean_of_remaining_students 
  (h2 : mean_of_class = 10)
  (h3 : mean_of_20_students = 16) :
  let remaining_students := (k - 20)
  let total_score_20 := 20 * mean_of_20_students
  let total_score_class := k * mean_of_class
  let total_score_remaining := total_score_class - total_score_20
  let mean_remaining := total_score_remaining / remaining_students
  mean_remaining = (10 * k - 320) / (k - 20) :=
sorry

end mean_of_remaining_students_l299_299088


namespace systematic_sampling_condition_l299_299387

theorem systematic_sampling_condition (population sample_size total_removed segments individuals_per_segment : ℕ) 
  (h_population : population = 1650)
  (h_sample_size : sample_size = 35)
  (h_total_removed : total_removed = 5)
  (h_segments : segments = sample_size)
  (h_individuals_per_segment : individuals_per_segment = (population - total_removed) / sample_size)
  (h_modulo : population % sample_size = total_removed)
  :
  total_removed = 5 ∧ segments = 35 ∧ individuals_per_segment = 47 := 
by
  sorry

end systematic_sampling_condition_l299_299387


namespace trajectory_equation_right_angled_triangle_l299_299203

-- Variables and conditions
variables {F : Point} (P Q : Point) (x y : ℝ) {m : ℝ}
-- Point definition
def F := (0, 1)
def l := ∀ x : ℝ, y = -1
-- Moving point P and Q
def P := (x, y)
def Q := (x, -1)
-- Vector operations
def vector_dot (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Conditions
def condition1 : Prop := vector_dot (Q - P) (Q - F) = vector_dot (F - P) (F - Q)
def condition2 (M A B : Point) : Prop :=
  let l1 := ∀ x : ℝ, y = -m in -- line definition
  is_perpendicular (vector_line_segment M A) (vector_line_segment A B)

-- Proof statement
theorem trajectory_equation (h : condition1): ∃ C : ℝ × ℝ, C = (P : ℝ × ℝ) ∧ x^2 = 4y :=
sorry

theorem right_angled_triangle (h : condition2 M A B): ∃ M_points : List Point, M_points.length = 2 :=
sorry

end trajectory_equation_right_angled_triangle_l299_299203


namespace line_equation_x_2y_eq_1_l299_299169

def line_eqn (a b : ℝ) : Prop :=
  ∃ (l : ℝ → ℝ), (∀ x, l x = (-1 / 2) * x + 3 / 2) ∧ l 3 = -1 ∧ a = 2 * b ∧ (a ≠ 0 → ∀ x y, y = l x ↔ x + 2 * y - 1 = 0)

theorem line_equation_x_2y_eq_1 :
  ∃ (a b : ℝ), a = 2 * b ∧ (∀ x y, y = (-1 / 2) * x + 3 / 2 → x + 2 * y - 1 = 0) :=
begin
  sorry
end

end line_equation_x_2y_eq_1_l299_299169


namespace no_term_is_5_l299_299025

def seq_a : ℕ → ℕ 
| 1 := 2
| (n + 2) := Nat.factorization.finest (seq_a 1 * ∏ i in finset.range (n + 1), seq_a (i + 2) + 1)

theorem no_term_is_5 :
  ∀ n : ℕ, seq_a (n + 1) ≠ 5 :=
by sorry

end no_term_is_5_l299_299025


namespace gerald_price_l299_299966

-- Define the conditions provided in the problem

def price_hendricks := 200
def discount_percent := 20
def discount_ratio := 0.80 -- since 20% less means Hendricks paid 80% of what Gerald paid

-- Question to be answered: Prove that the price Gerald paid equals $250
-- P is what Gerald paid

theorem gerald_price (P : ℝ) (h : price_hendricks = discount_ratio * P) : P = 250 :=
by
  sorry

end gerald_price_l299_299966


namespace angle_between_unit_vectors_l299_299242

noncomputable def angle_between_vectors (a b : ℝ) : ℝ :=
  real.arccos (a * b)

theorem angle_between_unit_vectors (a b : EuclideanSpace ℝ (Fin 2)) 
  (ha : ∥a∥ = 1) (hb : ∥b∥ = 1)
  (h : ∥a + b∥ = real.sqrt 3 * ∥b∥) :
  angle_between_vectors (euclidean_space.inner a b) = real.pi / 3 :=
by 
  sorry

end angle_between_unit_vectors_l299_299242


namespace x_needs_6_days_l299_299069

theorem x_needs_6_days (x_time y_time days_worked : ℕ) (h1 : x_time = 18) (h2 : y_time = 15) (h3 : days_worked = 10) : 
  let y_rate := 1 / y_time
      work_completed_by_y := days_worked * y_rate
      remaining_work := 1 - work_completed_by_y
      x_rate := 1 / x_time
      additional_days_for_x := remaining_work / x_rate
  in
  additional_days_for_x = 6 :=
by
  sorry

end x_needs_6_days_l299_299069


namespace probability_two_black_balls_l299_299083

theorem probability_two_black_balls (white_balls black_balls drawn_balls : ℕ) 
  (h_w : white_balls = 4) (h_b : black_balls = 7) (h_d : drawn_balls = 2) :
  let total_ways := Nat.choose (white_balls + black_balls) drawn_balls
  let black_ways := Nat.choose black_balls drawn_balls
  (black_ways / total_ways : ℚ) = 21 / 55 :=
by
  sorry

end probability_two_black_balls_l299_299083


namespace zoe_candy_bars_needed_l299_299459

def total_cost : ℝ := 485
def grandma_contribution : ℝ := 250
def per_candy_earning : ℝ := 1.25
def required_candy_bars : ℕ := 188

theorem zoe_candy_bars_needed :
  (total_cost - grandma_contribution) / per_candy_earning = required_candy_bars :=
by
  sorry

end zoe_candy_bars_needed_l299_299459


namespace vip_seat_cost_l299_299829

theorem vip_seat_cost
  (V : ℝ)
  (G V_T : ℕ)
  (h1 : 20 * G + V * V_T = 7500)
  (h2 : G + V_T = 320)
  (h3 : V_T = G - 276) :
  V = 70 := by
sorry

end vip_seat_cost_l299_299829


namespace total_dog_food_consumption_l299_299156

theorem total_dog_food_consumption ( 
  daily_A : ℝ := 0.125,
  daily_B : ℝ := 0.25,
  daily_C : ℝ := 0.375,
  daily_D : ℝ := 0.5,
  daily_E : ℝ := 0.75,
  extra_Sunday_C : ℝ := 0.1,
  less_Sunday_E : ℝ := 0.1
) : 
  let total_days := 30
  let weeks := 4
  let weekdays := 26 -- days from Mon-Sat
  let sundays := 4
  let total_weekday_consumption := (daily_A + daily_B + daily_C + daily_D + daily_E) * weekdays
  let sunday_C := daily_C + extra_Sunday_C
  let sunday_E := daily_E - less_Sunday_E
  let total_sunday_consumption := (daily_A + daily_B + sunday_C + daily_D + sunday_E) * sundays
  let total_consumption := total_weekday_consumption + total_sunday_consumption
  total_consumption = 60 
:=
by
  sorry

end total_dog_food_consumption_l299_299156


namespace find_pq_cube_l299_299233

theorem find_pq_cube (p q : ℝ) (h1 : p + q = 5) (h2 : p * q = 3) : (p + q) ^ 3 = 125 := 
by
  -- This is where the proof would go
  sorry

end find_pq_cube_l299_299233


namespace first_player_wins_optimal_play_l299_299855

open Function

noncomputable def optimal_play_winner : Prop :=
  let grid_length := 101
  ∃ (player1_pos player2_pos : ℕ) (turn : ℕ),
    player1_pos = 1 ∧ player2_pos = grid_length ∧ 
    ∀ turn >= 0, 
        (if turn % 2 = 0 then player1_pos else player2_pos)
        + (if turn % 2 = 0 then 1 else -1) * (1 ≤ 4)
      ≠ (if turn % 2 = 0 then player2_pos else player1_pos)
      main_winner turn = 1 ∧ player1_pos = 101

theorem first_player_wins_optimal_play : optimal_play_winner :=
begin
  sorry
end

end first_player_wins_optimal_play_l299_299855


namespace remainder_when_divided_by_5_l299_299822

theorem remainder_when_divided_by_5 
  (k : ℕ)
  (h1 : k % 6 = 5)
  (h2 : k < 42)
  (h3 : k % 7 = 3) : 
  k % 5 = 2 := 
by 
  sorry

end remainder_when_divided_by_5_l299_299822


namespace second_player_wins_l299_299713

theorem second_player_wins (initial_numbers : List ℕ) (h : initial_numbers = [25, 36]) :
  ∃ strategy : ℕ → ℕ, strategy 2 = 'second player wins' :=
begin
  sorry
end

end second_player_wins_l299_299713


namespace no_coprime_xy_multiple_l299_299641

theorem no_coprime_xy_multiple (n : ℕ) (hn : ∀ d : ℕ, d ∣ n → d^2 ∣ n → d = 1)
  (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) (h_coprime : Nat.gcd x y = 1) :
  ¬ ((x^n + y^n) % ((x + y)^3) = 0) :=
by
  sorry

end no_coprime_xy_multiple_l299_299641


namespace three_layer_rug_area_l299_299179

theorem three_layer_rug_area 
  (A B C D : ℕ) 
  (hA : A = 350) 
  (hB : B = 250) 
  (hC : C = 45) 
  (h_formula : A = B + C + D) : 
  D = 55 :=
by
  sorry

end three_layer_rug_area_l299_299179


namespace correct_average_calculation_l299_299388

-- Conditions as definitions
def incorrect_average := 5
def num_values := 10
def incorrect_num := 26
def correct_num := 36

-- Statement to prove
theorem correct_average_calculation : 
  (incorrect_average * num_values + (correct_num - incorrect_num)) / num_values = 6 :=
by
  -- Placeholder for the proof
  sorry

end correct_average_calculation_l299_299388


namespace expected_babies_is_1008_l299_299491

noncomputable def babies_expected_after_loss
  (num_kettles : ℕ)
  (pregnancies_per_kettle : ℕ)
  (babies_per_pregnancy : ℕ)
  (loss_percentage : ℤ) : ℤ :=
  let total_babies := num_kettles * pregnancies_per_kettle * babies_per_pregnancy
  let survival_rate := (100 - loss_percentage) / 100
  total_babies * survival_rate

theorem expected_babies_is_1008 :
  babies_expected_after_loss 12 20 6 30 = 1008 :=
by
  sorry

end expected_babies_is_1008_l299_299491


namespace smallest_value_of_3b_plus_2_l299_299631

theorem smallest_value_of_3b_plus_2 (b : ℝ) (h : 8 * b^2 + 7 * b + 6 = 5) : (∃ t : ℝ, t = 3 * b + 2 ∧ (∀ x : ℝ, 8 * x^2 + 7 * x + 6 = 5 → x = b → t ≤ 3 * x + 2)) :=
sorry

end smallest_value_of_3b_plus_2_l299_299631


namespace area_tripled_radius_increase_l299_299001

theorem area_tripled_radius_increase (m r : ℝ) (h : (r + m)^2 = 3 * r^2) :
  r = m * (1 + Real.sqrt 3) / 2 :=
sorry

end area_tripled_radius_increase_l299_299001


namespace cost_of_first_variety_of_tea_l299_299497

theorem cost_of_first_variety_of_tea 
  (x : ℝ) 
  (blend_ratio : 5 * x + 60 = c)
  (sell_price_per_kg : 21) 
  (gain_percent: 1.12 * c = 168) :
  x = 18 := 
by sorry

end cost_of_first_variety_of_tea_l299_299497


namespace fifa_world_cup_matches_l299_299281

theorem fifa_world_cup_matches (teams groups knockout_teams: ℕ) 
  (round_robin_groups teams_per_group matches_per_group: ℕ) :
  teams = 24 →
  round_robin_groups = 6 →
  groups = round_robin_groups →
  teams_per_group * groups = teams →
  matches_per_group = (teams_per_group * (teams_per_group - 1)) / 2 →
  (teams_per_group = 4) →
  (knockout_teams = 16) →
  total_matches = (round_robin_groups * matches_per_group) + (knockout_teams - 1) →
  total_matches = 51 :=
begin
  sorry
end

end fifa_world_cup_matches_l299_299281


namespace max_irrational_in_chessboard_l299_299279

open_locale classical

def is_irrational (x : ℝ) : Prop := ¬ ∃ q : ℚ, x = q

theorem max_irrational_in_chessboard : 
  ∃ (a b : fin 2019 → ℝ), 
    (∀ i j : fin 2019, i ≠ j → a i ≠ a j) ∧           -- Unique numbers in the first row
    (∀ i : fin 2019, ∃ j : fin 2019, b j = a i) ∧     -- Second row is a permutation of the first
    (∀ i : fin 2019, a i ≠ b i ∧ (a i + b i) ∈ ℚ) ∧   -- Numbers in each column add to a rational number
    (card (set_of (λ x, ∃ i, a i = x ∨ b i = x) ∧ is_irrational x) = 4032) := -- Maximum irrational count

by {
    sorry                                                -- Proof to be filled in later
}

end max_irrational_in_chessboard_l299_299279


namespace count_correct_props_l299_299947

-- Define the propositions as Booleans
def prop1 : Prop := ∀ (l1 l2 : Plane) (p : Plane), (l1 ∥ p) ∧ (l2 ∥ p) → (l1 ∥ l2)
def prop2 : Prop := ∀ (l : Line) (p1 p2 : Plane), (l ⊥ p1) → (p1 ∩ p2 = l) → (p1 ⊥ p2)
def prop3 : Prop := ∀ (l1 l2 l3 : Line), (l1 ⊥ l3) ∧ (l2 ⊥ l3) → (l1 ∥ l2)
def prop4 : Prop := ∀ (p1 p2 : Plane) (l : Line), (p1 ⊥ p2) → (¬(l ⊥ (p1 ∩ p2))) → (l ⊥ p2)

-- Define a function to count the number of true propositions
def num_correct_props (props : List Prop) : Nat :=
  props.foldl (λ acc prop, if prop then acc + 1 else acc) 0

-- State the main theorem
theorem count_correct_props :
  num_correct_props [¬prop1, prop2, ¬prop3, prop4] = 2 :=
by sorry -- Proof is skipped

end count_correct_props_l299_299947


namespace max_repeating_decimal_min_repeating_decimal_l299_299356

noncomputable def number : ℝ := 0.20120415

theorem max_repeating_decimal : ∃ d, d = 0.2012041, repeating (0.2012041, 5) → d = 0.2012041 * 10^(-7) + 5/(10^(-7) - 1) := sorry

theorem min_repeating_decimal : ∃ d, d = 0.2, repeating (0.2, 0120415) → d = 0.2 * 10^(-1) + 0120415 / (10^(-7) - 1) := sorry

end max_repeating_decimal_min_repeating_decimal_l299_299356


namespace monotonic_intervals_find_f_max_l299_299603

noncomputable def f (x : ℝ) : ℝ := Real.log x / x

theorem monotonic_intervals :
  (∀ x, 0 < x → x < Real.exp 1 → 0 < (1 - Real.log x) / x^2) ∧
  (∀ x, x > Real.exp 1 → (1 - Real.log x) / x^2 < 0) :=
sorry

theorem find_f_max (m : ℝ) (h : m > 0) :
  if 0 < 2 * m ∧ 2 * m ≤ Real.exp 1 then f (2 * m) = Real.log (2 * m) / (2 * m)
  else if m ≥ Real.exp 1 then f m = Real.log m / m
  else f (Real.exp 1) = 1 / Real.exp 1 :=
sorry

end monotonic_intervals_find_f_max_l299_299603


namespace no_constant_term_l299_299775

theorem no_constant_term : ∀ (n a b m : ℕ), (n = 8) → (a = 3) → (b = 2) → (m = 2) →
  ¬ (∃ k : ℤ, (3 * k - 16 = 0) ∧ ↑k ∈ finset.Ico 0 (n + 1)) :=
by
  intros n a b m hn ha hb hm
  rw [hn, ha, hb, hm]
  intro h
  have : ∃ k : ℤ, (3 * k - 16 = 0) ∧ ↑k ∈ finset.Ico 0 9 := h
  obtain ⟨k, hk⟩ := this
  sorry

end no_constant_term_l299_299775


namespace triangles_similar_l299_299689

-- Assume the necessary definitions for circles, intersections, and points exist in our context.
variables {Γ Γ' : Circle} {P Q A A' O O' : Point} 

-- Given conditions in Lean style
def circles_intersect (Γ Γ' : Circle) (P Q : Point) : Prop :=
  P ∈ Γ ∧ P ∈ Γ' ∧ Q ∈ Γ ∧ Q ∈ Γ'

def point_on_circle (A : Point) (Γ : Circle) : Prop :=
  A ∈ Γ

def line_intersects_circle (A P : Point) (Γ' : Circle) (A' : Point) : Prop :=
  lies_on (line A P) A' ∧ A' ∈ Γ'

-- Theorem to be proven
theorem triangles_similar (Γ Γ' : Circle) (P Q A A' O O' : Point)
  (h1 : circles_intersect Γ Γ' P Q)
  (h2 : point_on_circle A Γ)
  (h3 : line_intersects_circle A P Γ' A') :
  similar (triangle A Q A') (triangle O Q O') :=
sorry

end triangles_similar_l299_299689


namespace every_positive_integer_is_dapper_l299_299099

def begins_with_2008 (m : ℕ) : Prop :=
  ∃ k : ℕ, m / 10^k = 2008

theorem every_positive_integer_is_dapper : ∀ n : ℕ, 0 < n → ∃ m : ℕ, begins_with_2008 m ∧ m % n = 0 :=
by {
  intro n,
  intro hn,
  sorry
}

end every_positive_integer_is_dapper_l299_299099


namespace triangle_area_l299_299522

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

theorem triangle_area :
  area_of_triangle (0, 0) (0, 6) (8, 10) = 24 :=
by
  sorry

end triangle_area_l299_299522


namespace max_element_of_list_l299_299818

theorem max_element_of_list (list : List ℕ) (h_len : list.length = 6) (h_med : list.nth_le 2 (by linarith) = 4 ∧ list.nth_le 3 (by linarith) = 4) (h_mean : (list.sum / 6 : ℝ) = 15) : 
  list.maximum = 78 := 
sorry

end max_element_of_list_l299_299818


namespace count_difference_l299_299512

-- Given definitions
def count_six_digit_numbers_in_ascending_order_by_digits : ℕ := by
  -- Calculation using binomial coefficient
  exact Nat.choose 9 6

def count_six_digit_numbers_with_one : ℕ := by
  -- Calculation using binomial coefficient with fixed '1' in one position
  exact Nat.choose 8 5

def count_six_digit_numbers_without_one : ℕ := by
  -- Calculation subtracting with and without 1
  exact count_six_digit_numbers_in_ascending_order_by_digits - count_six_digit_numbers_with_one

-- Theorem to prove
theorem count_difference : 
  count_six_digit_numbers_with_one - count_six_digit_numbers_without_one = 28 :=
by
  sorry

end count_difference_l299_299512


namespace cory_chairs_l299_299146

theorem cory_chairs (total_cost table_cost chair_cost C : ℕ) (h1 : total_cost = 135) (h2 : table_cost = 55) (h3 : chair_cost = 20) (h4 : total_cost = table_cost + chair_cost * C) : C = 4 := 
by 
  sorry

end cory_chairs_l299_299146


namespace chinese_count_l299_299515

theorem chinese_count (total_people : ℕ) (americans : ℕ) (australians : ℕ) (chinese : ℕ) 
  (h1 : total_people = 49) (h2 : americans = 16) (h3 : australians = 11) : 
  chinese = total_people - (americans + australians) → chinese = 22 :=
by
  intro h
  rw [h1, h2, h3] at h
  rw [Nat.add_comm, Nat.add_sub_assoc] at h
  rw [Nat.add_comm] at h
  assumption

#eval chinese_count 49 16 11 22 rfl rfl rfl rfl -- Should evaluate to true

end chinese_count_l299_299515


namespace minimum_value_of_ratio_l299_299923

theorem minimum_value_of_ratio 
  {a b c : ℝ} (h_a : a ≠ 0) 
  (h_f'0 : 2 * a * 0 + b > 0)
  (h_f_nonneg : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) :
  (∃ x : ℝ, a * x^2 + b * x + c ≥ 0) ∧ (1 + (a + c) / b = 2) := sorry

end minimum_value_of_ratio_l299_299923


namespace f_monotonic_intervals_f_maximum_value_on_interval_l299_299605

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_monotonic_intervals :
  (∀ x : ℝ, 0 < x → x < Real.e → f x < f Real.e) ∧
  (∀ x : ℝ, x > Real.e → f x < f Real.e) :=
sorry

theorem f_maximum_value_on_interval (m : ℝ) (hm : m > 0) :
  ∃ x_max : ℝ, x_max ∈ (Set.Icc m (2 * m)) ∧
  (∀ x ∈ (Set.Icc m (2 * m)), f x ≤ f x_max) ∧
  ((m₁ : 0 < m ∧ m ≤ Real.e / 2 → x_max = 2 * m ∧ f x_max = (Real.log (2 * m)) / (2 * m)) ∨
   (m₂ : m ≥ Real.e → x_max = m ∧ f x_max = (Real.log m) / m) ∨
   (m₃ : Real.e / 2 < m ∧ m < Real.e → x_max = Real.e ∧ f x_max = 1 / Real.e)) :=
sorry

end f_monotonic_intervals_f_maximum_value_on_interval_l299_299605


namespace total_cost_of_shirts_is_24_l299_299682

-- Definitions based on conditions
def cost_first_shirt : ℕ := 15
def cost_difference : ℕ := 6
def cost_second_shirt : ℕ := cost_first_shirt - cost_difference

-- The proof problem statement: Calculate total cost given the conditions
theorem total_cost_of_shirts_is_24 : cost_first_shirt + cost_second_shshirt = 24 :=
by
  sorry

end total_cost_of_shirts_is_24_l299_299682


namespace next_calendar_year_l299_299039

theorem next_calendar_year (y : ℕ) (y = 1990) : ∃ n, y + n = 2004 :=
by {
  -- conditions (leap year and day shifts)
  let is_leap_year := λ y, (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0),
  have h1: ¬is_leap_year 1990 := by simp [is_leap_year, (by norm_num : ¬ 1900 = 0)],
  have h2: is_leap_year 1992 := by simp [is_leap_year, (by norm_num : 1992 % 4 = 0)],
  
  -- find n
  use 14,
  exact sorry
}

end next_calendar_year_l299_299039


namespace max_amount_can_receive_l299_299355

-- Define the different grade categories and their respective rewards
def rewards (B+ A A+ : Nat) : Nat :=
  5 * B+ + 10 * A + 20 * A+

-- Define the conditions for the bonuses
def bonuses (B+ A A+ : Nat) : Nat :=
  if A+ >= 3 ∧ A >= 2 then 50
  else if A >= 4 ∧ A+ >= 1 then 30
  else if A >= 5 ∧ A+ == 0 then 10 * A
  else 0

-- Define the total amount based on the grades
def total_amount (B+ A A+ : Nat) : Nat :=
  rewards B+ A A+ + bonuses B+ A A+

-- Problem statement to prove the maximum amount given 12 courses
theorem max_amount_can_receive : (∃ (B+ A A+ : Nat), B+ + A + A+ = 12 ∧ total_amount B+ A A+ = 290) :=
by
  sorry

end max_amount_can_receive_l299_299355


namespace point_in_fourth_quadrant_l299_299639

theorem point_in_fourth_quadrant (m : ℝ) : (m-1 > 0 ∧ 2-m < 0) ↔ m > 2 :=
by
  sorry

end point_in_fourth_quadrant_l299_299639


namespace range_of_f_inequality_l299_299704

noncomputable def f (x : ℝ) : ℝ := Real.exp (|x|) - 2 / (x^2 + 3)

theorem range_of_f_inequality :
  {x : ℝ | f x > f (2 * x - 1)} = set.Ioo (1 / 3) 1 := by
  sorry

end range_of_f_inequality_l299_299704


namespace value_of_x_l299_299237

theorem value_of_x (x : ℕ) (M : Set ℕ) :
  M = {0, 1, 2} →
  M ∪ {x} = {0, 1, 2, 3} →
  x = 3 :=
by
  sorry

end value_of_x_l299_299237


namespace quadrilateral_circles_cover_l299_299717

theorem quadrilateral_circles_cover (A B C D : Point) : 
  let circle1 := mk_circle_diameter A B,
      circle2 := mk_circle_diameter B C,
      circle3 := mk_circle_diameter C D,
      circle4 := mk_circle_diameter D A in
  ∀ P ∈ quadrilateral A B C D, 
    P ∈ circle1 ∨ P ∈ circle2 ∨ P ∈ circle3 ∨ P ∈ circle4 :=
by
  sorry

end quadrilateral_circles_cover_l299_299717


namespace complex_number_roots_l299_299330

theorem complex_number_roots (p q r : ℂ) (h1 : p + q + r = 2) (h2 : p * q + q * r + r * p = 3) (h3 : p * q * r = 2) :
  (p = 1 ∨ p = (1 + complex.I * real.sqrt 7) / 2 ∨ p = (1 - complex.I * real.sqrt 7) / 2) ∧
  (q = 1 ∨ q = (1 + complex.I * real.sqrt 7) / 2 ∨ q = (1 - complex.I * real.sqrt 7) / 2) ∧
  (r = 1 ∨ r = (1 + complex.I * real.sqrt 7) / 2 ∨ r = (1 - complex.I * real.sqrt 7) / 2) :=
sorry

end complex_number_roots_l299_299330


namespace angle_AFH_eq_90_l299_299903

open Real EuclideanGeometry

variables (O P A D B C E F H : Point)
variables (h_outer: P ≠ O and P not in set_of_points_in_circle O)
variables (h1 : tangent_to_circle O P A)
variables (h2 : tangent_to_circle O P D)
variables (h3 : secant_to_circle O P B C)
variables (h4 : orthogonal_to_line BC AE E)
variables (h5 : intersects_in_point_again DE F O)
variables (h6 : orthocenter_triangle H A B C)

theorem angle_AFH_eq_90:
  ∠AFH = 90° :=
  by sorry

end angle_AFH_eq_90_l299_299903


namespace term_2007_in_sequence_is_4_l299_299013

-- Definition of the function to compute the sum of the squares of the digits of a number
def sum_of_squares_of_digits (n : Nat) : Nat := 
  n.digits.sum (λ d => d * d)

-- Definition of the sequence based on the given rules
def sequence : Nat → Nat
| 0 => 2007
| (n + 1) => sum_of_squares_of_digits (sequence n)

-- Theorem stating that the 2007th term in the sequence is 4
theorem term_2007_in_sequence_is_4 : sequence 2007 = 4 :=
  sorry -- Proof skipped

end term_2007_in_sequence_is_4_l299_299013


namespace g_of_neg2_l299_299147

def g (x : ℤ) : ℤ := x^3 - x^2 + x

theorem g_of_neg2 : g (-2) = -14 := 
by
  sorry

end g_of_neg2_l299_299147


namespace smallest_prime_divides_l299_299563

theorem smallest_prime_divides (p : ℕ) (a : ℕ) 
  (h1 : Prime p) (h2 : p > 100) (h3 : a > 1) (h4 : p ∣ (a^89 - 1) / (a - 1)) :
  p = 179 := 
sorry

end smallest_prime_divides_l299_299563


namespace smallest_and_largest_square_digits_l299_299772

def all_digits_once (n : ℕ) : Prop :=
  let digits := (nat.digits 10 n).erase_dup.length == 10 ∧ ∀ d, d ∈ (nat.digits 10 n) -> d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

theorem smallest_and_largest_square_digits :
  ∃ (n₁ n₂ : ℕ), all_digits_once (n₁^2) ∧ all_digits_once (n₂^2) ∧
  (∀ m₁, all_digits_once (m₁^2) → n₁^2 ≤ m₁^2) ∧ 
  (∀ m₂, all_digits_once (m₂^2) → n₂^2 >= m₂^2) ∧ 
  n₁ = 32043 ∧ n₂ = 99066 :=
by {
  sorry
}

end smallest_and_largest_square_digits_l299_299772


namespace mean_of_remaining_four_numbers_l299_299002

theorem mean_of_remaining_four_numbers 
  (a b c d max_num : ℝ) 
  (h1 : max_num = 105) 
  (h2 : (a + b + c + d + max_num) / 5 = 92) : 
  (a + b + c + d) / 4 = 88.75 :=
by
  sorry

end mean_of_remaining_four_numbers_l299_299002


namespace problem_a_plus_i_l299_299294

noncomputable def sequence := ℕ → ℤ

constants (A B C D E F G H I : ℤ)
constants t : sequence

axioms
  (h1 : D = 7)
  (h2 : I = 10)
  (h3 : t 0 = A)
  (h4 : t 1 = B)
  (h5 : t 2 = C)
  (h6 : t 3 = D)
  (h7 : t 4 = E)
  (h8 : t 5 = F)
  (h9 : t 6 = G)
  (h10 : t 7 = H)
  (h11 : t 8 = I)
  (h12 : ∀ n : ℕ, t n + t (n + 1) + t (n + 2) = 36)

theorem problem_a_plus_i : A + I = 17 :=
sorry

end problem_a_plus_i_l299_299294


namespace collinear_points_l299_299725

theorem collinear_points (A B C D E F G H I J K L M : Type*)
  [nonagon A B C D E F G H I]
  [hexagon A J K L M I]
  (shared_vertices : A = A ∧ I = I) :
  collinear H M D :=
sorry

end collinear_points_l299_299725


namespace parabola_vertex_shift_l299_299405

def shift_parabola (x : ℝ) : ℝ := (x - 2)^2 + 3

theorem parabola_vertex_shift :
  let original_vertex := (1, 2) in
  let shifted_vertex := (3, 5) in
  (shift_parabola 1, shift_parabola 2 + 3) = shifted_vertex := 
sorry

end parabola_vertex_shift_l299_299405


namespace product_of_real_values_l299_299539

theorem product_of_real_values (r : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (1 / (3 * x)) = (r - x) / 8 → (3 * x * x - 3 * r * x + 8 = 0)) →
  r = 4 * Real.sqrt 6 / 3 ∨ r = -(4 * Real.sqrt 6 / 3) →
  r * -r = -32 / 3 :=
by
  intro h_x
  intro h_r
  sorry

end product_of_real_values_l299_299539


namespace volume_displacement_square_l299_299092

-- Define the given conditions
def radius_cylinder := 5
def height_cylinder := 12
def side_length_cube := 10

theorem volume_displacement_square :
  let r := radius_cylinder
  let h := height_cylinder
  let s := side_length_cube
  let cube_diagonal := s * Real.sqrt 3
  let w := (125 * Real.sqrt 6) / 8
  w^2 = 1464.0625 :=
by
  sorry

end volume_displacement_square_l299_299092


namespace solve_for_x_opposites_l299_299629

theorem solve_for_x_opposites (x : ℝ) (h : -2 * x = -(3 * x - 1)) : x = 1 :=
by {
  sorry
}

end solve_for_x_opposites_l299_299629


namespace dihedral_angle_eq_l299_299698

variables {P A B C M : Point}
variables (angle_ABC : ∠ A B C = 90) (midpoint_M : midpoint_of M P A)
variables (AB_len : dist A B = 1) (AC_len : dist A C = 2) (AP_len : dist A P = sqrt 2)

theorem dihedral_angle_eq :
  dihedral_angle M B C A = arctan (2 / 3) :=
sorry

end dihedral_angle_eq_l299_299698


namespace invitation_methods_l299_299024

noncomputable def combination (n k : ℕ) : ℕ := Nat.choose n k

theorem invitation_methods (A B : Type) (students : Finset Type) (h : students.card = 10) :
  (∃ s : Finset Type, s.card = 6 ∧ A ∉ s ∧ B ∉ s) ∧ 
  (∃ t : Finset Type, t.card = 6 ∧ (A ∈ t ∨ B ∉ t)) →
  (combination 10 6 - combination 8 4 = 140) :=
by
  sorry

end invitation_methods_l299_299024


namespace sin_inequality_l299_299913

theorem sin_inequality (x y : ℝ) (h1 : 0 < x) (h2 : x < y) (h3 : y < 1) : 0 < sin x ∧ sin x < sin y := by
  sorry

end sin_inequality_l299_299913


namespace AC_plus_third_BA_l299_299574

def point := (ℝ × ℝ)

def A : point := (2, 4)
def B : point := (-1, -5)
def C : point := (3, -2)

noncomputable def vec (p₁ p₂ : point) : point :=
  (p₂.1 - p₁.1, p₂.2 - p₁.2)

noncomputable def scal_mult (scalar : ℝ) (v : point) : point :=
  (scalar * v.1, scalar * v.2)

noncomputable def vec_add (v₁ v₂ : point) : point :=
  (v₁.1 + v₂.1, v₁.2 + v₂.2)

theorem AC_plus_third_BA : 
  vec_add (vec A C) (scal_mult (1 / 3) (vec B A)) = (2, -3) :=
by
  sorry

end AC_plus_third_BA_l299_299574


namespace centipede_earthworm_meeting_time_l299_299455

theorem centipede_earthworm_meeting_time :
  let speed_centipede := 5 / 3 -- in meters per minute
  let speed_earthworm := 5 / 2 -- in meters per minute
  let initial_gap    := 20    -- in meters
  initial_gap / (speed_earthworm - speed_centipede) = 24 := by
  let speed_centipede := 5 / 3
  let speed_earthworm := 5 / 2
  let initial_gap := 20
  calc
    initial_gap / (speed_earthworm - speed_centipede) =
      20 / ( (5 / 2) - (5 / 3) ) : by sorry
    ... = 20 / (15 / 6 - 10 / 6) : by sorry
    ... = 20 / ( 5 / 6) : by sorry
    ... = 24 : by sorry

end centipede_earthworm_meeting_time_l299_299455


namespace number_between_Kristina_Nadya_l299_299070

-- Define the friends
inductive Friend
| Kristina | Nadya | Marina | Liza | Galya
deriving DecidableEq

open Friend

-- Establish conditions
def is_first (f : Friend) : Prop := f = Marina
def is_last (f : Friend) : Prop := f = Kristina
def is_second (f : Friend) : Prop := f = Nadya
def is_second_to_last (f : Friend) : Prop := f = Galya
def is_fourth (f : Friend) : Prop := f = Galya
noncomputable def position (f : Friend) : Nat :=
  match f with
  | Marina => 1
  | Nadya => 2
  | Liza => 3
  | Galya => 4
  | Kristina => 5

theorem number_between_Kristina_Nadya : 
  ∀ (friends : List Friend), 
    friends = [Marina, Nadya, Liza, Galya, Kristina] -> 
    (position Kristina - position Nadya - 1) = 3 :=
by 
  intros
  simp [position, Nat.sub]
  rfl

end number_between_Kristina_Nadya_l299_299070


namespace range_of_a_exists_a_l299_299235

def setA : set ℝ := {x | abs (x - 1) < 2}
def setB (a : ℝ) : set ℝ := {x | x^2 + a * x - 6 < 0}
def setC : set ℝ := {x | x^2 - 2 * x - 15 < 0}

theorem range_of_a (a : ℝ) : (setA ∪ setB a = setB a) ↔ (a ∈ Icc (-5 : ℝ) (-1 : ℝ)) := sorry

theorem exists_a (a : ℝ) : ∃ a, (setA ∪ setB a = setB a ∩ setC) := sorry

end range_of_a_exists_a_l299_299235


namespace identically_zero_l299_299319

noncomputable def continuous_function_zero (f : ℝ × ℝ → ℝ) : Prop :=
  (∀ R : set (ℝ × ℝ), measurable_set R ∧ volume R = 1 → ∫∫ r in R, f (r.1, r.2) d(r.1, r.2) = 0)

theorem identically_zero {f : ℝ × ℝ → ℝ}
    (h1 : continuous f)
    (h2 : ∀ R : set (ℝ × ℝ), measurable_set R ∧ volume R = 1 → ∫∫ r in R, f (r.1, r.2) d(r.1, r.2) = 0) :
  ∀ (x y : ℝ), f x y = 0 :=
sorry

end identically_zero_l299_299319


namespace central_angle_of_sector_l299_299104

theorem central_angle_of_sector (r S : ℝ) (h_r : r = 2) (h_S : S = 4) : 
  ∃ α : ℝ, α = 2 ∧ S = (1/2) * α * r^2 := 
by 
  sorry

end central_angle_of_sector_l299_299104


namespace count_parallelograms_l299_299884

theorem count_parallelograms (ABC : Triangle) (n : ℕ) : 
    (three_times_comb n) = 3 * binomial (n+2) 4 :=
sorry

end count_parallelograms_l299_299884


namespace count_primes_between_10_and_20_l299_299624

-- Define the prime number property
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- List the integers between 10 and 20
def integers_between_10_and_20 : List ℕ :=
  [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

-- Define the problem as proving the number of prime numbers in the list
theorem count_primes_between_10_and_20 : 
  (integers_between_10_and_20.filter is_prime).length = 4 :=
by
  sorry

end count_primes_between_10_and_20_l299_299624


namespace total_percent_samples_candy_l299_299471

-- Define the conditions
def total_customers : ℝ := 100
def percent_caught : ℝ := 22
def percent_not_caught : ℝ := 8

-- Let P be the total percent of customers who sample the candy
noncomputable def total_percent_sample (P : ℝ) : Prop :=
  0.08 * P = P - 22

-- Proof statement: There exists a P such that total_percent_sample P and P = 23.91
theorem total_percent_samples_candy : ∃ P : ℝ, total_percent_sample P ∧ P ≈ 23.91 :=
by
  sorry

end total_percent_samples_candy_l299_299471


namespace sequence_2007th_term_is_85_l299_299011

noncomputable def sum_of_square_of_digits (n : ℕ) : ℕ :=
(n.digits 10).map (λ d, d * d).sum

noncomputable def sequence_term : ℕ → ℕ
| 0 := 2007
| (n+1) := sum_of_square_of_digits (sequence_term n)

theorem sequence_2007th_term_is_85 : sequence_term 2007 = 85 := 
sorry

end sequence_2007th_term_is_85_l299_299011


namespace wind_speed_l299_299098

theorem wind_speed (w : ℝ) (h : 420 / (253 + w) = 350 / (253 - w)) : w = 23 :=
by
  sorry

end wind_speed_l299_299098


namespace hyperbola_eccentricity_range_l299_299411

theorem hyperbola_eccentricity_range {a b : ℝ} (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b) :
  ∃ e : ℝ, e = (Real.sqrt (a^2 + b^2)) / a ∧ 1 < e ∧ e < Real.sqrt 2 :=
by
  -- Proof would go here
  sorry

end hyperbola_eccentricity_range_l299_299411


namespace gerald_price_l299_299965

-- Define the conditions provided in the problem

def price_hendricks := 200
def discount_percent := 20
def discount_ratio := 0.80 -- since 20% less means Hendricks paid 80% of what Gerald paid

-- Question to be answered: Prove that the price Gerald paid equals $250
-- P is what Gerald paid

theorem gerald_price (P : ℝ) (h : price_hendricks = discount_ratio * P) : P = 250 :=
by
  sorry

end gerald_price_l299_299965


namespace trajectory_M_on_line_segment_l299_299955

open Real EuclideanGeometry

noncomputable def F₁ : Point ℝ := (5, 0)
noncomputable def F₂ : Point ℝ := (-5, 0)

theorem trajectory_M_on_line_segment (M : Point ℝ) 
  (hF₁M : dist M F₁ + dist M F₂ = 10) :
  on_line_segment M F₁ F₂ :=
sorry

end trajectory_M_on_line_segment_l299_299955


namespace find_x_l299_299596

-- Define the operation "※" as given
def star (a b : ℕ) : ℚ := (a + 2 * b) / 3

-- Given that 6 ※ x = 22 / 3, prove that x = 8
theorem find_x : ∃ x : ℕ, star 6 x = 22 / 3 ↔ x = 8 :=
by
  sorry -- Proof not required

end find_x_l299_299596


namespace min_S_val_l299_299758

noncomputable def min_S (n : ℕ) :=
  1 - 1 / (2^(1/n:ℝ))

theorem min_S_val (n : ℕ) (x : Fin n → ℝ) (hx_sum : (∑ i, x i) = 1) (hx_pos: ∀ i, x i > 0) :
  let S := finset.max' (finset.univ.image (λ k, (x k) / (1 + ∑ i in finset.range (k + 1), x i))) sorry in
  S = min_S n := sorry

end min_S_val_l299_299758


namespace encrypted_result_of_3859_l299_299157

def unit_digit (n : ℤ) : ℤ := n % 10

def encrypt_digit (d : ℤ) : ℤ :=
  unit_digit (d^3 + 1)

def encrypt_number (n : ℤ) : ℤ :=
  let digits := [3, 8, 5, 9] in -- Assuming the digits of 3859 are directly extracted
  let encrypted_digits := digits.map encrypt_digit in
  encrypted_digits.foldl (λ acc x, acc * 10 + x) 0

theorem encrypted_result_of_3859 :
  encrypt_number 3859 = 8360 :=
by
  sorry

end encrypted_result_of_3859_l299_299157


namespace product_expression_eq_l299_299797

theorem product_expression_eq (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) * (xy + yz + zx)⁻¹ * ((xy)⁻¹ + (yz)⁻¹ + (zx)⁻¹) = x⁻² * y⁻² * z⁻² :=
by
  sorry

end product_expression_eq_l299_299797


namespace small_cone_altitude_l299_299093

noncomputable def area_of_circle (r : ℝ) := real.pi * r ^ 2

theorem small_cone_altitude
  (h₁ : ∀ r, area_of_circle r = 400 * real.pi → r = 20)
  (h₂ : ∀ r, area_of_circle r = 36 * real.pi → r = 6)
  (h_frustum : 30 = 2 / 3 * (2 / 3 * 45 + 30)) :
  30 = 2 / 3 * 45 → 15 = 1 / 3 * 45 :=
by
  sorry

end small_cone_altitude_l299_299093


namespace num_ways_Xiaoming_Xiaohong_diff_uni_l299_299457

theorem num_ways_Xiaoming_Xiaohong_diff_uni : 
  ∃ (students : Finset String), 
  ∃ universities : Finset String,
  ∃ num_applications : Finset (String × String) → ℕ,
  (students = {"Xiaoming", "Xiaohong", "C", "D"}) ∧ 
  (universities = {"UniversityA", "UniversityB"}) ∧
  (∀ s ∈ students, ∃ u ∈ universities, True) ∧  -- Each student applies to some university
  (∃ num_ways : ℕ, num_ways = 4 ∧
    num_applications {"UniA", "UniB"} = 2 ∧
    ((num_applications {"AC", "BD"} = num_ways) ∨ (num_applications {"AD", "BC"} = num_ways))) :=
sorry

end num_ways_Xiaoming_Xiaohong_diff_uni_l299_299457


namespace factor_2310_two_digit_numbers_l299_299627

theorem factor_2310_two_digit_numbers :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 2310 ∧ ∀ (c d : ℕ), 10 ≤ c ∧ c < 100 ∧ 10 ≤ d ∧ d < 100 ∧ c * d = 2310 → (c = a ∧ d = b) ∨ (c = b ∧ d = a) :=
by {
  sorry
}

end factor_2310_two_digit_numbers_l299_299627


namespace find_a_if_even_function_l299_299263

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2 * a^2 - a) * x + 1

theorem find_a_if_even_function (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 1 / 2 := by
  sorry

end find_a_if_even_function_l299_299263


namespace caesars_meal_cost_is_30_l299_299383

-- Define the charges for each hall
def caesars_room_rent : ℕ := 800
def caesars_meal_cost (C : ℕ) (guests : ℕ) : ℕ := C * guests
def venus_room_rent : ℕ := 500
def venus_meal_cost : ℕ := 35
def venus_total_cost (guests : ℕ) : ℕ := venus_room_rent + venus_meal_cost * guests

-- The number of guests when costs are the same
def guests : ℕ := 60

-- The total cost at Caesar's for 60 guests
def caesars_total_cost (C : ℕ) : ℕ := caesars_room_rent + caesars_meal_cost C guests

-- The fact that the costs are the same
def costs_are_equal (C : ℕ) : Prop := caesars_total_cost C = venus_total_cost guests

-- The theorem to prove
theorem caesars_meal_cost_is_30 : ∃ (C : ℕ), costs_are_equal C ∧ C = 30 :=
by 
  let C := 30
  have h1: caesars_total_cost C = venus_total_cost guests,
  {
    calc
      caesars_total_cost C
        = 800 + 60 * C : by rw [caesars_meal_cost, caesars_room_rent, guests]
    ... = 800 + 60 * 30 : by rw [C]
    ... = 800 + 1800 : by norm_num
    ... = 2600 : by norm_num
    ... = 500 + 2100: by norm_num
    ... = 500 + 60 * 35 : by rw [venus_meal_cost, guests]
    ... = venus_total_cost guests : by rw [venus_total_cost, venus_room_rent]
  },
  have h2: C = 30 := by refl,
  use C,
  exact ⟨h1, h2⟩

end caesars_meal_cost_is_30_l299_299383


namespace quadratic_function_properties_l299_299902

-- Definitions
def quadratic_function (m : ℝ) : ℝ → ℝ := λ x, m * x^2 - (4 * m + 1) * x + 3 * m + 3
def is_correct_1 (m : ℝ) (h : m < 0) : Prop :=
  ∀ x, x > 2 → quadratic_function(m) x ∈ (Ioi (quadratic_function(m) 2))

def passes_through_points (m : ℝ) : Prop :=
  quadratic_function(m) 1 = 2 ∧ quadratic_function(m) 3 = 0

def distance_less_than_2 (m : ℝ) (h : m < 0) : Prop :=
  ∃ x1 x2, (quadratic_function(m) x1 = 0 ∧ quadratic_function(m) x2 = 0) ∧ |x1 - x2| < 2

def ordinate_of_vertex (m : ℝ) (h : m < 0) : ℝ :=
  let a := m in
  let b := -(4 * m + 1) in
  let c := 3 * m + 3 in
  (4 * a * c - b^2) / (4 * a)

-- Proof Statements
theorem quadratic_function_properties (m : ℝ) (h : m < 0) :
  is_correct_1 m h ∧ passes_through_points m ∧ ¬distance_less_than_2 m h ∧ ordinate_of_vertex m h > 2 :=
sorry

end quadratic_function_properties_l299_299902


namespace find_center_of_circle_l299_299168

theorem find_center_of_circle (x y : ℝ) :
  4 * x^2 + 8 * x + 4 * y^2 - 24 * y + 16 = 0 →
  (x + 1)^2 + (y - 3)^2 = 6 :=
by
  intro h
  sorry

end find_center_of_circle_l299_299168


namespace sally_cherry_cost_l299_299723

noncomputable def peach_cost_after_coupon : ℝ := 12.32
noncomputable def coupon_value : ℝ := 3.0
noncomputable def total_spent : ℝ := 23.86

theorem sally_cherry_cost : ∃ (cherry_cost : ℝ), cherry_cost = 8.54 :=
by
  let peach_original_cost := peach_cost_after_coupon + coupon_value
  let cherry_cost := total_spent - peach_original_cost
  have h : cherry_cost = 8.54 := sorry
  use cherry_cost
  exact h

end sally_cherry_cost_l299_299723


namespace CE_eq_CA_l299_299850

variables {O1 O2 D A B C E : Point}
variables {r1 r2 : ℝ}

-- Defining that points D, A, and B belong to the respective circles O1 and O2
def is_tangent (O : Point) (r : ℝ) (P : Point) := (O - P).norm = r

-- Assuming the circles are tangent at D
def externally_tangent (O1 O2 D : Point) (r1 r2 : ℝ) :=
  is_tangent O1 r1 D ∧ is_tangent O2 r2 D

-- Tangent line AB is tangent to both circles
def common_external_tangent (O1 O2 D A B : Point) (r1 r2 : ℝ) :=
  is_tangent O1 r1 A ∧ is_tangent O2 r2 B

-- AO1 intersects O1 at C
def intersection_point (A O1 C : Point) (r1 : ℝ) :=
  is_tangent O1 r1 A ∧ (C - O1).norm = r1

-- Tangent CE at E for circle O2 from point C
def tangent_line (C E O2 : Point) (r2 : ℝ) :=
  is_tangent O2 r2 E ∧ (C - E).norm = (C - O2).norm + r2

-- Final theorem to prove CE = CA
theorem CE_eq_CA {O1 O2 D A B C E : Point} {r1 r2 : ℝ}
  (h1 : externally_tangent O1 O2 D r1 r2)
  (h2 : common_external_tangent O1 O2 D A B r1 r2)
  (h3 : intersection_point A O1 C r1)
  (h4 : tangent_line C E O2 r2) :
  (C - E).norm = (C - A).norm :=
sorry -- Proof goes here

end CE_eq_CA_l299_299850


namespace vector_condition_l299_299184

-- Define vectors a and b as non-zero vectors
variables (a b : ℝ^3) 

-- Define conditions: vectors a and b are non-zero
noncomputable def non_zero_vectors (a b : ℝ^3) := a ≠ 0 ∧ b ≠ 0

-- Define a parallel relationship between vectors
def parallel (a b : ℝ^3) : Prop := ∃ k : ℝ, a = k • b

-- Define vectors being antiparallel relationship
def antiparallel (a b : ℝ^3) : Prop := ∃ k : ℝ, a = -k • b ∧ k > 0

-- Define the Lean theorem statement
theorem vector_condition (h₀ : non_zero_vectors a b) : (a + b = 0) → parallel a b :=
by sorry

end vector_condition_l299_299184


namespace functional_relationship_optimizing_profit_l299_299832

-- Define the scope of the problem with conditions and proof statements

variables (x : ℝ) (y : ℝ)

-- Conditions
def price_condition := 44 ≤ x ∧ x ≤ 52
def sales_function := y = -10 * x + 740
def profit_function (x : ℝ) := -10 * x^2 + 1140 * x - 29600

-- Lean statement to prove the first part
theorem functional_relationship (h₁ : 44 ≤ x) (h₂ : x ≤ 52) : y = -10 * x + 740 := by
  sorry

-- Lean statement to prove the second part
theorem optimizing_profit (h₃ : 44 ≤ x) (h₄ : x ≤ 52) : (profit_function 52 = 2640 ∧ (∀ x, (44 ≤ x ∧ x ≤ 52) → profit_function x ≤ 2640)) := by
  sorry

end functional_relationship_optimizing_profit_l299_299832


namespace total_campers_rowing_and_hiking_l299_299480

def campers_morning_rowing : ℕ := 41
def campers_morning_hiking : ℕ := 4
def campers_afternoon_rowing : ℕ := 26

theorem total_campers_rowing_and_hiking :
  campers_morning_rowing + campers_morning_hiking + campers_afternoon_rowing = 71 :=
by
  -- We are skipping the proof since instructions specify only the statement is needed
  sorry

end total_campers_rowing_and_hiking_l299_299480


namespace solve_quadratic_l299_299419

theorem solve_quadratic (x : ℝ) : x^2 - 2*x = 0 ↔ (x = 0 ∨ x = 2) :=
by
  sorry

end solve_quadratic_l299_299419


namespace sqrt_n_and_cbrt_m_integers_l299_299017

theorem sqrt_n_and_cbrt_m_integers
  (m n : ℤ) (k : ℤ) : 
  (k = (Real.sqrt n : ℝ) + (Real.cbrt m : ℝ)) →
  (∃ a b : ℤ, a^2 = n ∧ b^3 = m) :=
by
  sorry

end sqrt_n_and_cbrt_m_integers_l299_299017


namespace ingrid_cookies_ratio_l299_299674

theorem ingrid_cookies_ratio (total_cookies : ℕ) (percent_baked_by_ingrid : ℚ)
  (h1 : total_cookies = 148)
  (h2 : percent_baked_by_ingrid = 31.524390243902438) :
  let ingrid_cookies := round (total_cookies * (percent_baked_by_ingrid / 100)) in
  ingrid_cookies = 47 ∧ (ingrid_cookies : ℚ) / total_cookies = 47 / 148 :=
by
  sorry

end ingrid_cookies_ratio_l299_299674


namespace find_system_of_equations_l299_299925

variables {x y : ℕ}

theorem find_system_of_equations 
  (h1 : x - y = 1) 
  (h2 : 10 * x + y - (10 * y + x) = 9) :
  x - y = 1 ∧ 10 * x + y - (10 * y + x) = 9 :=
by {
    exact ⟨h1, h2⟩,
    sorry
}

end find_system_of_equations_l299_299925


namespace intense_goblet_point_difference_l299_299997

theorem intense_goblet_point_difference :
  let teams := 10,
      games := teams * (teams - 1) / 2,
      max_points := games * 4,
      min_points := teams * (teams - 1),
      point_difference := max_points - min_points
  in point_difference = 90 :=
by {
  let teams := 10,
  let games := teams * (teams - 1) / 2,
  let max_points := games * 4,
  let min_points := teams * (teams - 1),
  let point_difference := max_points - min_points,
  show point_difference = 90,
  sorry
}

end intense_goblet_point_difference_l299_299997


namespace mountain_count_remainder_l299_299323

def p : ℕ := 2^16 + 1

def is_monotonic_bounded (a : ℕ → ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ 2^16 → (1 ≤ a i ∧ a i ≤ i)

def is_mountain (a : ℕ → ℕ) (k : ℕ) : Prop :=
  a k > a (k-1) ∧ a k > a (k+1)

def total_mountains (A : ℕ → ℕ → Prop) : ℕ :=
  ∑ a in {a : ℕ → ℕ | is_monotonic_bounded a}, ∑ k in {k | 2 ≤ k ∧ k ≤ 2^16 - 1}, if is_mountain a k then 1 else 0

theorem mountain_count_remainder :
  (total_mountains is_monotonic_bounded % p) = 49153 :=
sorry

end mountain_count_remainder_l299_299323


namespace mary_age_next_birthday_l299_299349

def mary_next_birthday_age (m s d : ℕ) : Prop :=
  1.25 * s = m ∧ s = 0.5 * d ∧ m + s + d = 42 → m + 1 = 14

theorem mary_age_next_birthday :
  ∃ (m s d : ℕ), mary_next_birthday_age m s d :=
by {
  sorry
}

end mary_age_next_birthday_l299_299349


namespace fractional_units_l299_299015

-- Define the mixed number and the smallest composite number
def mixed_number := 3 + 2/7
def smallest_composite := 4

-- To_struct fractional units of 3 2/7
theorem fractional_units (u : ℚ) (n : ℕ) (m : ℕ):
  u = 1/7 ∧ n = 23 ∧ m = 5 :=
by
  have h1 : u = 1 / 7 := sorry
  have h2 : mixed_number = 23 * u := sorry
  have h3 : smallest_composite - mixed_number = 5 * u := sorry
  have h4 : n = 23 := sorry
  have h5 : m = 5 := sorry
  exact ⟨h1, h4, h5⟩

end fractional_units_l299_299015


namespace billy_buys_bottle_l299_299866

-- Definitions of costs and volumes
def money : ℝ := 10
def cost1 : ℝ := 1
def volume1 : ℝ := 10
def cost2 : ℝ := 2
def volume2 : ℝ := 16
def cost3 : ℝ := 2.5
def volume3 : ℝ := 25
def cost4 : ℝ := 5
def volume4 : ℝ := 50
def cost5 : ℝ := 10
def volume5 : ℝ := 200

-- Statement of the proof problem
theorem billy_buys_bottle : ∃ b : ℕ, b = 1 ∧ cost5 = money := by 
  sorry

end billy_buys_bottle_l299_299866


namespace percentage_increase_in_expenditure_l299_299786

/-- Given conditions:
- The price of sugar increased by 32%
- The family's original monthly sugar consumption was 30 kg
- The family's new monthly sugar consumption is 25 kg
- The family's expenditure on sugar increased by 10%

Prove that the percentage increase in the family's expenditure on sugar is 10%. -/
theorem percentage_increase_in_expenditure (P : ℝ) :
  let initial_consumption := 30
  let new_consumption := 25
  let price_increase := 0.32
  let original_price := P
  let new_price := (1 + price_increase) * original_price
  let original_expenditure := initial_consumption * original_price
  let new_expenditure := new_consumption * new_price
  let expenditure_increase := new_expenditure - original_expenditure
  let percentage_increase := (expenditure_increase / original_expenditure) * 100
  percentage_increase = 10 := sorry

end percentage_increase_in_expenditure_l299_299786


namespace leading_coefficient_of_g_l299_299232

noncomputable def g (x : ℕ) : ℕ → ℕ := sorry

theorem leading_coefficient_of_g :
  (g ((x : ℕ) + 1) - g (x)) = 12 * (x : ℕ) + 2 → 
  has_leading_coefficient g 6 := 
sorry


end leading_coefficient_of_g_l299_299232


namespace six_digit_numbers_with_and_without_one_difference_l299_299509

theorem six_digit_numbers_with_and_without_one_difference :
  let total_numbers := Nat.choose 9 6 in
  let numbers_with_one := Nat.choose 8 5 in
  let numbers_without_one := total_numbers - numbers_with_one in
  numbers_with_one - numbers_without_one = 28 :=
by
  let total_numbers := Nat.choose 9 6
  let numbers_with_one := Nat.choose 8 5
  let numbers_without_one := total_numbers - numbers_with_one
  exact (numbers_with_one - numbers_without_one)
  sorry

end six_digit_numbers_with_and_without_one_difference_l299_299509


namespace gerald_price_l299_299964

-- Define the conditions provided in the problem

def price_hendricks := 200
def discount_percent := 20
def discount_ratio := 0.80 -- since 20% less means Hendricks paid 80% of what Gerald paid

-- Question to be answered: Prove that the price Gerald paid equals $250
-- P is what Gerald paid

theorem gerald_price (P : ℝ) (h : price_hendricks = discount_ratio * P) : P = 250 :=
by
  sorry

end gerald_price_l299_299964


namespace probability_one_card_each_suit_l299_299976

-- Define a standard deck size
def deck_size : ℕ := 52

-- Define the number of suits in a standard deck
def num_suits : ℕ := 4

-- Define the number of cards per suit in a standard deck
def cards_per_suit : ℕ := deck_size / num_suits

-- Define the total number of draws
def num_draws : ℕ := 4

-- Define a probability of drawing a card from a deck
def prob(d : ℕ) (n : ℕ) : ℝ := n / d

-- The condition that cards are drawn with replacement
def with_replacement := true

-- Main theorem statement
theorem probability_one_card_each_suit (h_deck : deck_size = 52) (h_suits : num_suits = 4)
    (h_cards_per_suit : cards_per_suit = 13) (h_draws : num_draws = 4)
    (h_replacement : with_replacement) :
    prob(deck_size, 39) * prob(deck_size, 26) * prob(deck_size, 13) = 3 / 32 := 
begin
    sorry
end

end probability_one_card_each_suit_l299_299976


namespace exists_two_numbers_sum_eq_53_l299_299194

theorem exists_two_numbers_sum_eq_53 (S : Finset ℕ) (hS_card : S.card = 53) (hS_sum : S.sum id ≤ 1990) : 
  ∃ a b ∈ S, a ≠ b ∧ a + b = 53 :=
by
  sorry

end exists_two_numbers_sum_eq_53_l299_299194


namespace yanni_money_left_in_cents_l299_299061

-- Define the constants based on the conditions
def initial_amount := 0.85
def mother_amount := 0.40
def found_amount := 0.50
def toy_cost := 1.60

-- Function to calculate the total amount
def total_amount := initial_amount + mother_amount + found_amount

-- Function to calculate the money left
def money_left := total_amount - toy_cost

-- Convert the remaining money from dollars to cents
def money_left_in_cents := money_left * 100

-- The theorem to prove
theorem yanni_money_left_in_cents : money_left_in_cents = 15 := by
  -- placeholder for proof, sorry used to skip the proof
  sorry

end yanni_money_left_in_cents_l299_299061


namespace incorrect_statements_proof_l299_299788

def incorrect_statements (α : ℝ) : Prop :=
α = 1 ∧ (¬ (0 < α ∧ α < π / 2) ∨ ¬ (π / 2 < α ∧ α < π))
∧ (∀ α : ℝ, (0 < α ∧ α < π / 2) → (0 < α / 3 ∧ α / 3 < π / 2))

theorem incorrect_statements_proof (α : ℝ) : incorrect_statements α :=
by
  -- α = 1 is in the first quadrant, not the second.
  unfold incorrect_statements
  split
  · exact 1
  · left
    split
    · linarith
    · linarith
  · intros α h1 h2
    linarith

end incorrect_statements_proof_l299_299788


namespace max_reciprocal_sum_l299_299196

theorem max_reciprocal_sum (n : ℕ) (h : 0 < n) :
  ∃ C : ℝ, (∀ S : multiset ℕ, (∀ x ∈ S, 1 < x) → S.sum (λ x, 1 / (x : ℝ)) < C →
  ∃ f : multiset ℕ →₀ ℝ, (∀ s ∈ f.support, s.sum (λ x, 1 / (x : ℝ)) < 1) ∧ f.support.card ≤ n) ∧
  C = (n + 1) / 2 :=
by sorry

end max_reciprocal_sum_l299_299196


namespace length_error_probability_l299_299595

noncomputable def normal_dist (μ σ : ℝ) : MeasureTheory.ProbMeasure ℝ :=
  MeasureTheory.ProbMeasure.fromDensity ((Real.gaussian μ σ).toDensity)

theorem length_error_probability :
  let μ := 0
  let σ := 3
  let P := normal_dist μ σ
  P.measure {x | 3 < x ∧ x < 6} = 0.1359 :=
by
  sorry

end length_error_probability_l299_299595


namespace wickets_before_last_match_l299_299096

theorem wickets_before_last_match 
  (initial_average : ℝ) 
  (wickets_last_match : ℕ) 
  (runs_last_match : ℝ) 
  (average_decrease : ℝ) 
  (W : ℕ) 
  (initial_average_eq : initial_average = 12.4)
  (wickets_last_match_eq : wickets_last_match = 7)
  (runs_last_match_eq : runs_last_match = 26)
  (average_decrease_eq : average_decrease = 0.4)
  (new_average_eq : initial_average - average_decrease = 12.0)
  (new_total_wickets_eq : W + wickets_last_match)
  (new_total_runs_eq : 12.4 * W + 26)
  (new_average_statement : (12.4 * W + 26) / (W + 7) = 12.0) : 
  W = 145 :=
by
  sorry

end wickets_before_last_match_l299_299096


namespace range_of_a_l299_299948

theorem range_of_a (a : ℝ) (h : ∀ x y : ℝ, 1 < x → x < y → y → f a x < f a y ) 
  : a ≤ 1 :=
sorry

end range_of_a_l299_299948


namespace find_line_equations_through_point_and_circle_l299_299706

-- Definitions of the conditions
def point_P : ℝ × ℝ := (-4, -3)
def center_circle : ℝ × ℝ := (-1, -2)
def radius_circle : ℝ := 5
def chord_length : ℝ := 8

-- Proof Statement: The desired equations of the lines
theorem find_line_equations_through_point_and_circle :
  (∃ k : ℝ, ∀ x y : ℝ, y = k * (x - (-4)) - 3 ↔ (x + 1)^2 + (y + 2)^2 = 25 ∧ (∀ x₁ x₂ : ℝ, 
    ((x₁ + 1) = (x₂ + 1)) → (y₁ = k * (x₁ - (-4)) - 3) → (y₂ = k * (x₂ - (-4)) - 3) 
    → dist (x₁, y₁) (x₂, y₂) = 8)) ∨
  (∀ x : ℝ, (x = -4 ↔ (x + 1)^2 + ((-3) + 2)^2 = 25 ∧ (∀ x₁ x₂ : ℝ, 
    (x₁ = -4) ∧ (x₂ = -4) → dist (x₁, -3) (x₂, -3) = 8)))
: (x = -4 ∨ 4 * x + 3 * y + 25 = 0) :=
sorry

end find_line_equations_through_point_and_circle_l299_299706


namespace workshop_male_workers_l299_299107

variables (F M : ℕ)

theorem workshop_male_workers :
  (M = F + 45) ∧ (M - 5 = 3 * F) → M = 65 :=
by
  intros h
  sorry

end workshop_male_workers_l299_299107


namespace a_sequence_b_sequence_sum_l299_299199

noncomputable def a (n : ℕ) : ℝ := (1 / 2) ^ n
noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range (n+1), a i

theorem a_sequence (n : ℕ) (h1 : ∀ n, a n + S n = 1) : a n = (1/2)^n :=
sorry

noncomputable def b (n : ℕ) : ℝ := 3 + Real.logBase 4 (a n)
noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range (n+1), abs (b i)

theorem b_sequence_sum (n : ℕ) (h2 : ∀ n, b n = 3 + Real.logBase 4 (a n)) : 
  T n = if n ≤ 6 then (n * (11 - n)) / 4 else (n^2 - 11 * n + 60) / 4 :=
sorry

end a_sequence_b_sequence_sum_l299_299199


namespace data_variance_l299_299740

def data : List ℝ := [9.8, 9.9, 10.1, 10, 10.2]

noncomputable def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : List ℝ) : ℝ :=
  (data.map (λ x => (x - mean data) ^ 2)).sum / data.length

theorem data_variance : variance data = 0.02 := by
  sorry

end data_variance_l299_299740


namespace prism_edge_length_l299_299041

theorem prism_edge_length (h_cube : ∀ (x : ℝ), x = 6 ∨ x = 6) 
  (h_edges : 12 * 6 = 72) : ∃ x, 12 * x = 72 := by
sorbus matplotlib.pyplot as plt import Matplotlib.pyplot as plt import matplotlib.pyplot as plt import matplotlib.pyplot as plt 

end prism_edge_length_l299_299041


namespace digits_of_3_pow_15_mul_5_pow_10_l299_299523

theorem digits_of_3_pow_15_mul_5_pow_10 :
  let digits_count (n : ℕ) : ℕ := n.to_string.length
  digits_count (3^15 * 5^10) = 14 :=
by
  -- Here we define the function to compute the number of digits and state our theorem
  let digits_count (n : ℕ) := n.to_string.length
  sorry

end digits_of_3_pow_15_mul_5_pow_10_l299_299523


namespace find_fx_l299_299908

theorem find_fx (f : ℝ → ℝ) (h : ∀ x : ℝ, f(x-1) = x^2 + 4x - 5) : 
  ∀ x : ℝ, f(x) = x^2 + 6x :=
sorry

end find_fx_l299_299908


namespace prove_min_value_of_expr_l299_299197

noncomputable def min_value_of_expr : ℝ :=
  let a : ℝ := sorry in
  let c : ℝ := sorry in
  have h1 : a > 0 := sorry,
  have h2 : c > 0 := sorry,
  have h3 : a * c = 1 := sorry,
  (a + 1) / c + (c + 1) / a

theorem prove_min_value_of_expr : min_value_of_expr = 4 :=
by
  sorry

end prove_min_value_of_expr_l299_299197


namespace probability_of_selecting_triangle_with_shaded_vertex_l299_299289

-- Definition of vertices and triangles
inductive Vertex
| A | B | C | D | E

open Vertex

def ShadedRegion : set Vertex := {D, E, C}

def Triangles : set (set Vertex) :=
  {{A, B, C}, {A, B, E}, {A, E, D}, {A, E, C}, {E, D, C}, {B, E, C}}

-- Function to decide if a triangle has at least one vertex in the shaded region
def has_vertex_in_shaded_region (triangle : set Vertex) : Prop :=
  ∃ v ∈ triangle, v ∈ ShadedRegion

-- Number of all triangles
def total_triangles := Triangles.to_finset.card

-- Number of triangles with at least one vertex in the shaded region
def triangles_with_shaded_vertex :=
  (Triangles.filter has_vertex_in_shaded_region).to_finset.card

-- The probability that a selected triangle has at least one vertex in the shaded region
def probability := (triangles_with_shaded_vertex : ℚ) / total_triangles

theorem probability_of_selecting_triangle_with_shaded_vertex :
  probability = 2 / 3 :=
sorry

end probability_of_selecting_triangle_with_shaded_vertex_l299_299289


namespace vertical_distance_is_22_l299_299106

noncomputable def total_vertical_distance : ℕ :=
  let number_of_discs := ((30 - 2) / 2) + 1 in
  let total_thickness_of_discs := number_of_discs * 1 in
  let total_space_between_discs := (number_of_discs - 1) * 0.5 in
  total_thickness_of_discs + total_space_between_discs

theorem vertical_distance_is_22 :
  total_vertical_distance = 22 := 
by
  sorry

end vertical_distance_is_22_l299_299106


namespace even_product_divisible_by_1947_l299_299461

theorem even_product_divisible_by_1947 (n : ℕ) (h_even : n % 2 = 0) :
  (∃ k: ℕ, 2 ≤ k ∧ k ≤ n / 2 ∧ 1947 ∣ (2 ^ k * k!)) → n ≥ 3894 :=
by
  sorry

end even_product_divisible_by_1947_l299_299461


namespace deductive_reasoning_is_optionC_l299_299057

-- Definitions of the options as conditions
def optionA : Prop :=
  ∀ (T₁ T₂ : Type) (P₁ : T₁ → Prop) (P₂ : T₂ → Prop), 
  (∀ x : T₁, P₁ x → ∃ y : T₂, P₂ y) →
  (∃ x : T₁, P₁ x) → (∃ y : T₂, P₂ y)

def optionB : Prop :=
  ∀ (n₁ n₂ : ℕ) (P : ℕ → Prop), 
  (P n₁ ∧ P n₂) → (∀ n, P n → n > 50)

def optionC : Prop :=
  ∀ (L₁ L₂ : Type) (P : Type → Prop) (A B : Type),
  (∀ l₁ l₂ : L₁, P l₁ ∧ P l₂ → A = B) →
  (∃ l₁ l₂ : L₁, P l₁ ∧ P l₂) → A = B

def optionD : Prop :=
  ∀ (a : ℕ → ℕ), 
  a 1 = 2 ∧ (∀ n ≥ 2, a n = 2 * a (n - 1) + 1) →
  ∃ f : ℕ → ℕ, ∀ n ≥ 1, a n = f n

-- The proof problem statement
theorem deductive_reasoning_is_optionC : optionC :=
sorry

end deductive_reasoning_is_optionC_l299_299057


namespace pie_eating_contest_l299_299431

theorem pie_eating_contest (B A S : ℕ) (h1 : A = B + 3) (h2 : S = 12) (h3 : B + A + S = 27) :
  S / B = 2 :=
by
  have h4 : B + (B + 3) + 12 = 27 from by rw [h1, h2]; exact h3
  have h5 : 2 * B + 15 = 27 from by linarith
  have h6 : 2 * B = 12 from by linarith
  have h7 : B = 6 from by linarith
  rw [h7, h2]; norm_num

end pie_eating_contest_l299_299431


namespace sum_of_new_sequence_l299_299430

open Nat

def seq1 (n : ℕ) : ℕ := 2 + 4 * n
def seq2 (n : ℕ) : ℕ := 2 + 6 * n

def commonSeq (m : ℕ) : ℕ := 2 + 12 * m

theorem sum_of_new_sequence : (finset.range 16).sum commonSeq = 1472 :=
by
  sorry

end sum_of_new_sequence_l299_299430


namespace min_unit_cubes_intersect_all_l299_299091

theorem min_unit_cubes_intersect_all (n : ℕ) : 
  let A_n := if n % 2 = 0 then n^2 / 2 else (n^2 + 1) / 2
  A_n = if n % 2 = 0 then n^2 / 2 else (n^2 + 1) / 2 :=
sorry

end min_unit_cubes_intersect_all_l299_299091


namespace max_rows_l299_299558

theorem max_rows (m : ℕ) : (∀ T : Matrix (Fin m) (Fin 8) (Fin 4), 
  ∀ i j : Fin m, ∀ k l : Fin 8, i ≠ j ∧ T i k = T j k ∧ T i l = T j l → k ≠ l) → m ≤ 28 :=
sorry

end max_rows_l299_299558


namespace option_A_correct_option_B_incorrect_option_C_correct_option_D_correct_l299_299737

-- Define the sequence according to the given rules
noncomputable def seq (k : ℕ) : ℕ → ℕ
| 1       := 1
| (n + 1) := if even (seq n) then seq n / 2 else seq n + k

-- Prove Option A
theorem option_A_correct (k : ℕ) (h : k = 5) : seq k 5 = 4 :=
sorry

-- Prove Option B's negation
theorem option_B_incorrect (k : ℕ) (n : ℕ) (hn : n > 5) : ¬ (seq k n ≠ 1) :=
sorry

-- Prove Option C
theorem option_C_correct (k : ℕ) (h : odd k) (n : ℕ) : seq k n ≤ 2 * k :=
sorry

-- Prove Option D
theorem option_D_correct (k : ℕ) (h : even k) (n : ℕ) : seq k n < seq k (n + 1) :=
sorry

end option_A_correct_option_B_incorrect_option_C_correct_option_D_correct_l299_299737


namespace theta_angle_through_point_l299_299980

theorem theta_angle_through_point :
  ∀ k : ℤ, ∃ θ : ℝ, 
  (∃ (x y : ℝ), x = - (sqrt 3) / 2 ∧ y = 1 / 2 ∧ 
  cos θ = x / sqrt (x^2 + y^2) ∧ sin θ = y / sqrt (x^2 + y^2)) → 
  θ = 2 * k * Real.pi + 5 / 6 * Real.pi :=
by
  sorry

end theta_angle_through_point_l299_299980


namespace evaluate_floor_ceil_l299_299892

theorem evaluate_floor_ceil :
  Int.floor (-3.75) + Int.ceil (34.25) = 31 := by
  have h1 : Int.floor (-3.75) = -4 := by sorry
  have h2 : Int.ceil (34.25) = 35 := by sorry
  rw [h1, h2]
  norm_num

end evaluate_floor_ceil_l299_299892


namespace sum_of_doubled_primes_lt_20_l299_299449

theorem sum_of_doubled_primes_lt_20 : 
  let primes := [2, 3, 5, 7, 11, 13, 17, 19]
  let doubled_primes := List.map (fun p => 2 * p) primes
  let sum_doubled_primes := List.sum doubled_primes
  in sum_doubled_primes = 154 := by
  sorry

end sum_of_doubled_primes_lt_20_l299_299449


namespace eccentricity_is_correct_l299_299357

-- Hyperbola definition and associated points
variables {a b c x₁ y₁ : ℝ}
variables {P : ℝ × ℝ} (hP : P = (x₁, y₁))
variables (F₂ : ℝ × ℝ) (hF₂ : F₂ = (c, 0))
variables {M : ℝ × ℝ} (hM : M = ((x₁ + c) / 2, y₁ / 2))

-- Definitions of some properties given in the problem
variables (h_hyperbola : (x₁^2 / a^2) - (y₁^2 / b^2) = 1)
variables (h_positive_a : 0 < a)
variables (h_positive_b : 0 < b)
variables (h_magnitude_eq : dist (0, 0) F₂ = dist F₂ M)
variables (h_dot_product : (c * ((x₁ - c) / 2)) = c^2 / 2)

-- Definition of eccentricity
noncomputable def eccentricity := c / a

-- Theorem statement
theorem eccentricity_is_correct :
  eccentricity a c = (1 + real.sqrt 3) / 2 :=
sorry

end eccentricity_is_correct_l299_299357


namespace part1_part2_l299_299939

-- Definitions and conditions
def parabola_standard_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def focus_distance (p t : ℝ) : Prop := (2 + p / 2) = 4 -- Converted condition for finding p
def line_intersection (k x y : ℝ) : Prop := y = k * x + 1 -- Equation of the line
def distinct_intersection (C : ℝ) (k : ℝ) : Prop := (k^2 * C^2 + (2 * k - 8) * C + 1) > 0 -- From delta > 0
def scalar_product (p : ℝ) (k x1 x2 y1 y2 : ℝ) : Prop := (2*x1*x2 - 4*(x1 + x2) + (k^2 + 1)*(1/k^2) + (k - 2)*(8 - 2 * k) / k^2 + 1) = 4 -- Scalar product condition
def parabola_with_focus_dist_eq := parabola_standard_eq 4
def line_eq := λ k, line_intersection k

-- Proving the two parts given the conditions
theorem part1 (t : ℝ) (h : focus_distance 4 t) : ∀ x y, parabola_with_focus_dist_eq x y := 
by
  intro x y
  exact (parabola_standard_eq 4 x y)

theorem part2 : ∀ (k x1 x2 y1 y2 : ℝ), 
  distinct_intersection 8 k -> scalar_product 4 k x1 x2 y1 y2 -> k = 5 / 4 := 
by
  intros k x1 x2 y1 y2 h_dist h_scalar
  sorry

end part1_part2_l299_299939


namespace gerald_paid_l299_299958

theorem gerald_paid (G : ℝ) (h : 0.8 * G = 200) : G = 250 :=
by
  sorry

end gerald_paid_l299_299958


namespace tan_pi_div_four_l299_299037

theorem tan_pi_div_four : Real.tan (π / 4) = 1 := by
  sorry

end tan_pi_div_four_l299_299037


namespace domain_of_rational_function_l299_299446

theorem domain_of_rational_function :
  { x : ℝ | x ≠ 7 } = set.univ \ {7} :=
by {
  sorry
}

end domain_of_rational_function_l299_299446


namespace total_distance_travelled_eight_boys_on_circle_l299_299543

noncomputable def distance_travelled_by_boys (radius : ℝ) : ℝ :=
  let n := 8
  let angle := 2 * Real.pi / n
  let distance_to_non_adjacent := 2 * radius * Real.sin (2 * angle / 2)
  n * (100 + 3 * distance_to_non_adjacent)

theorem total_distance_travelled_eight_boys_on_circle :
  distance_travelled_by_boys 50 = 800 + 1200 * Real.sqrt 2 :=
  by
    sorry

end total_distance_travelled_eight_boys_on_circle_l299_299543


namespace part1_part2_l299_299918

/-!
Definition of sets A, B, and C.
-/
def A (a : ℝ) : set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }
def B : set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def C : set ℝ := { x | x^2 + 2 * x - 8 = 0 }

/-!
Part 1: Prove if A ⊆ B, then the range of a is:
a < - (2 * real.sqrt 57) / 3 ∨ a > (2 * real.sqrt 57) / 3 ∨ a = 5
-/
theorem part1 (a : ℝ) (h : A a ⊆ B) :
  a < - (2 * real.sqrt 57) / 3 ∨ a > (2 * real.sqrt 57) / 3 ∨ a = 5 :=
sorry

/-!
Part 2: Prove if ∅ ⊂ (A ∩ B) and A ∩ C = ∅, then a = -2.
-/
theorem part2 {a : ℝ} (h1 : ∅ ⊂ (A a ∩ B)) (h2 : (A a ∩ C) = ∅) : a = -2 :=
sorry

end part1_part2_l299_299918


namespace equal_incircle_radii_and_ratio_l299_299267

theorem equal_incircle_radii_and_ratio (PQ QR PR : ℝ) (N P Q R : Point) (y : ℝ) 
  (h1 : dist P Q = 8) (h2 : dist Q R = 17) (h3 : dist P R = 15)
  (h4 : N ∈ line_segment P R) 
  (h5 : let I₁ = incenter (triangle P Q N),
             I₂ = incenter (triangle Q R N) in
        radius I₁ = radius I₂) :
  let PN := y,
      RN := 15 - y in
  y / (15 - y) = 3 / 11 := 
by 
  -- Proof Elided
  sorry

end equal_incircle_radii_and_ratio_l299_299267


namespace small_range_l299_299825

theorem small_range (x : Fin 6 → ℝ) 
  (h_mean : (∑ i, x i) / 6 = 15) 
  (h_median : (x 2 + x 3) / 2 = 18) :
  ∃ x_min x_max, 
    x_min = min (x 0) (min (x 1) (min (x 2) (min (x 3) (min (x 4) (x 5))))) ∧ 
    x_max = max (x 0) (max (x 1) (max (x 2) (max (x 3) (max (x 4) (x 5))))) ∧ 
    x_max - x_min = 7 := sorry

end small_range_l299_299825


namespace general_term_a_n_l299_299979

theorem general_term_a_n (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hSn : ∀ n, S n = (2/3) * a n + 1/3) :
  ∀ n, a n = (-2)^(n-1) :=
sorry

end general_term_a_n_l299_299979


namespace city_miles_count_l299_299132

-- Defining the variables used in the conditions
def miles_per_gallon_city : ℝ := 30
def miles_per_gallon_highway : ℝ := 40
def highway_miles : ℝ := 200
def cost_per_gallon : ℝ := 3
def total_cost : ℝ := 42

-- Required statement for the proof, statement to prove: count of city miles is 270
theorem city_miles_count : ∃ (C : ℝ), C = 270 ∧
  (total_cost / cost_per_gallon) = ((C / miles_per_gallon_city) + (highway_miles / miles_per_gallon_highway)) :=
by
  sorry

end city_miles_count_l299_299132


namespace rhombus_difference_l299_299321

theorem rhombus_difference (n : ℕ) (h : n > 3)
    (m : ℕ := 3 * (n - 1) * n / 2)
    (d : ℕ := 3 * (n - 3) * (n - 2) / 2) :
    m - d = 6 * n - 9 := by {
  -- Proof omitted
  sorry
}

end rhombus_difference_l299_299321


namespace max_profit_value_l299_299487

/-- The ex-factory price function definition -/
def ex_factory_price (x : ℕ) : ℝ :=
if 0 < x ∧ x ≤ 100 then 60
else if 100 < x ∧ x ≤ 500 then 62 - (1/50) * x
else 0 -- We add a fallback for other values outside of given range

/-- The profit function definition -/
def profit (x : ℕ) : ℝ :=
let P := ex_factory_price x in
(P - 40) * x

noncomputable def max_profit_x := 500

theorem max_profit_value :
  profit max_profit_x = 6000 :=
by
  -- The proof of this theorem will follow the solution steps and calculations provided in the solution
  sorry

end max_profit_value_l299_299487


namespace gerald_paid_l299_299961

theorem gerald_paid (G : ℝ) (h : 0.8 * G = 200) : G = 250 := by
  sorry

end gerald_paid_l299_299961


namespace evaluate_expression_l299_299780

theorem evaluate_expression : 
  (1 / 2 + ((2 / 3 * (3 / 8)) + 4) - (8 / 16)) = (17 / 4) :=
by
  sorry

end evaluate_expression_l299_299780


namespace find_omega_min_g_l299_299229

def f (ω x : ℝ) := sin (ω * x - π / 6) + sin (ω * x - π / 2)

theorem find_omega 
  (ω : ℝ) (h1 : 0 < ω)
  (h2 : ω < 3)
  (h3 : f ω (π / 6) = 0) :
  ω = 2 := by
  sorry

def g (x : ℝ) := √3 * sin (x - π / 12)

theorem min_g 
  (x : ℝ) 
  (h1 : x ∈ [-π/4, 3*π/4]) :
  ∃ y : ℝ, g(x) = y ∧ y = -3/2 := by
  sorry

end find_omega_min_g_l299_299229


namespace laborer_absent_days_l299_299815

theorem laborer_absent_days 
  (W A : ℕ) 
  (h1 : W + A = 25) 
  (h2 : 2 * W - 0.5 * A = 37.5) : 
  A = 5 := 
sorry

end laborer_absent_days_l299_299815


namespace fraction_of_darker_tiles_l299_299489

-- Define the problem conditions
def repeating_pattern_size : ℕ := 4

def corners_resemble (floor : ℕ → ℕ → bool) : Prop :=
  ∀ x y, (x < repeating_pattern_size) ∧ (y < repeating_pattern_size) → floor x y = floor (repeating_pattern_size - x - 1) (y)
                                  ∧ floor x y = floor x (repeating_pattern_size - y - 1)

-- Define the goal to prove
theorem fraction_of_darker_tiles (floor : ℕ → ℕ → bool) 
  (h_pattern : repeating_pattern_size = 4)
  (h_corners : corners_resemble floor) :
  let dark_tiles_count := (repeating_pattern_size * repeating_pattern_size * 3) / 4
  ∑ x in (finset.range repeating_pattern_size), ∑ y in (finset.range repeating_pattern_size), if floor x y then 1 else 0 = dark_tiles_count :=
sorry

end fraction_of_darker_tiles_l299_299489


namespace max_students_on_field_trip_l299_299392

theorem max_students_on_field_trip 
  (bus_cost : ℕ := 100)
  (bus_capacity : ℕ := 25)
  (student_admission_cost_high : ℕ := 10)
  (student_admission_cost_low : ℕ := 8)
  (discount_threshold : ℕ := 20)
  (teacher_cost : ℕ := 0)
  (budget : ℕ := 350) :
  max_students ≤ bus_capacity ↔ bus_cost + 
  (if max_students ≥ discount_threshold then max_students * student_admission_cost_low
  else max_students * student_admission_cost_high) 
   ≤ budget := 
sorry

end max_students_on_field_trip_l299_299392


namespace smallest_chi_value_l299_299318

noncomputable def binomial (n k : ℕ) : ℕ := nat.choose n k

def catalan (n : ℕ) : ℕ :=
  if h : n = 0 then 1
  else ∑ k in finset.range n, catalan k * catalan (n - 1 - k)

theorem smallest_chi_value (S : finset (ℝ × ℝ)) (h₁ : S.card = 16)
  (h₂ : ∀ (p₁ p₂ p₃ : ℝ × ℝ), p₁ ∈ S → p₂ ∈ S → p₃ ∈ S → p₁ ≠ p₂ → p₁ ≠ p₃ → collinear ({p₁, p₂, p₃} : set (ℝ × ℝ)) → false) :
  (catalan 8) = 1430 :=
by {
  sorry
}

end smallest_chi_value_l299_299318


namespace intersection_M_N_l299_299617

def M : Set ℤ := {-1, 0, 1}

def N : Set ℝ := {x | x * (x - 2) ≤ 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end intersection_M_N_l299_299617


namespace alice_score_l299_299652

variables (correct_answers wrong_answers unanswered_questions : ℕ)
variables (points_correct points_incorrect : ℚ)

def compute_score (correct_answers wrong_answers : ℕ) (points_correct points_incorrect : ℚ) : ℚ :=
    (correct_answers : ℚ) * points_correct + (wrong_answers : ℚ) * points_incorrect

theorem alice_score : 
    correct_answers = 15 → 
    wrong_answers = 5 → 
    unanswered_questions = 10 → 
    points_correct = 1 → 
    points_incorrect = -0.25 → 
    compute_score 15 5 1 (-0.25) = 13.75 := 
by intros; sorry

end alice_score_l299_299652


namespace square_value_is_10000_l299_299784
noncomputable def squareValue : Real := 6400000 / 400 / 1.6

theorem square_value_is_10000 : squareValue = 10000 :=
  by
  -- The proof is based on the provided steps, which will be omitted here.
  sorry

end square_value_is_10000_l299_299784


namespace perp_bisector_eq_l299_299583

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def slope (A B : ℝ × ℝ) : ℝ := (B.2 - A.2) / (B.1 - A.1)

def perp_bisector (A B : ℝ × ℝ) : String :=
  let mid := midpoint A B
  let k := slope A B
  let p_k := -1 / k
  s!"y - {mid.2} = {p_k} * (x - {mid.1})"

theorem perp_bisector_eq (A B : ℝ × ℝ) (hA : A = ⟨-1, 3⟩) (hB : B = ⟨3, -7⟩) :
  perp_bisector A B = "y + 2 = (2 / 5) * (x - 1)" :=
by
  rw [hA, hB]
  simp [perp_bisector, midpoint, slope]
  sorry

end perp_bisector_eq_l299_299583


namespace sum_of_3_consecutive_multiples_of_3_l299_299027

theorem sum_of_3_consecutive_multiples_of_3 (a b c : ℕ) (h₁ : a = b + 3) (h₂ : b = c + 3) (h₃ : a = 42) : a + b + c = 117 :=
by sorry

end sum_of_3_consecutive_multiples_of_3_l299_299027


namespace find_angles_in_range_l299_299972

theorem find_angles_in_range (x : ℝ) (h : 0 ≤ x ∧ x < real.pi) :
  (real.tan (4 * x - real.pi / 4) = 1) ↔
  (x = real.pi / 8 ∨ x = 3 * real.pi / 8 ∨ x = 5 * real.pi / 8 ∨ x = 7 * real.pi / 8) :=
sorry

end find_angles_in_range_l299_299972


namespace unique_representation_l299_299071

theorem unique_representation (A : ℤ) : 
  ∃ (n : ℕ) (a : ℕ → ℤ), 
  (A = ∑ i in finset.range n, a i * (2 ^ i)) ∧ 
  (∀ i, a i ∈ {0, 1, -1}) ∧ 
  (∀ i, i < n - 1 → a i * a (i + 1) = 0) ∧ 
  (∀ m : ℕ , (m ≠ n) → A ≠  ∑ i in finset.range m, a i * (2 ^ i)) :=
sorry

end unique_representation_l299_299071


namespace problem_l299_299597

noncomputable def geom_seq (a : ℕ → ℝ) :=
  ∃ q : ℝ, ∀ n, a (n + 1) = q * a n

theorem problem (a : ℕ → ℝ)
  (geom : geom_seq a)
  (h1 : a 2 * a 3 * a 4 = 64)
  (h2 : sqrt (a 6 * a 8) = 16) :
  (1 / 4) ^ (-2 : ℤ) * 2 ^ (-3 : ℤ) - (a 5) ^ (1 / 3) = 0 :=
by {
  sorry
}

end problem_l299_299597


namespace volume_of_figure_l299_299750

noncomputable def regular_prism_volume (a : ℝ) : ℝ :=
  -- Conditions
  let lateral_edge := a;
  let base_height := a;
  let prism_conditions := lateral_edge = base_height;
  
  -- Answer
  (9 * a^3 * real.sqrt 3) / 4

-- Statement of the problem
theorem volume_of_figure (a : ℝ) (h : a > 0) : 
  regular_prism_volume a = (9 * a^3 * real.sqrt 3) / 4 :=
sorry

end volume_of_figure_l299_299750


namespace intersection_of_sets_l299_299238

open Set

theorem intersection_of_sets (M N : Set ℕ) (hM : M = {1, 2, 3}) (hN : N = {2, 3, 4}) :
  M ∩ N = {2, 3} :=
by
  sorry

end intersection_of_sets_l299_299238


namespace sign_of_f_l299_299329

variables {a b c R r : ℝ}
variables {A B C : ℝ}
-- Conditions
-- C is the largest angle in triangle
-- R is the circumradius
-- r is the inradius

-- Define a + b - 2R - 2r
def f (a b c R r : ℝ) := a + b - 2 * R - 2 * r

theorem sign_of_f
  (h₁ : a ≤ b)
  (h₂ : b ≤ c)
  (h₃ : C = π - (A + B))
  (h₄ : a = 2 * R * sin A)
  (h₅ : b = 2 * R * sin B)
  (h₆ : r = 4 * R * sin (A / 2) * sin (B / 2) * sin (C / 2)) :
  (f a b c R r > 0 ↔ C < π / 2) ∧
  (f a b c R r = 0 ↔ C = π / 2) ∧
  (f a b c R r < 0 ↔ C > π / 2) :=
sorry

end sign_of_f_l299_299329


namespace minimum_value_l299_299359

def a (n : ℕ) : ℕ
def b (n : ℕ) : ℕ

axiom a_one : a 1 = 1
axiom b_one : b 1 = 1
axiom b_n_recurrence : ∀ n ≥ 2, b n = a n * b (n - 1) - 1 / 4

theorem minimum_value (m : ℕ) (hm : m > 0) : 
  4 * sqrt (∏ i in Finset.range m.succ, b (i + 1)) + 
  (Finset.sum (Finset.range m.succ) (λ k, 1 / ∏ i in Finset.range (k + 1), a (i + 1))) = 5 :=
sorry

end minimum_value_l299_299359


namespace condition_A_sufficient_not_necessary_condition_B_l299_299581

theorem condition_A_sufficient_not_necessary_condition_B {a b : ℝ} (hA : a > 1 ∧ b > 1) : 
  (a + b > 2 ∧ ab > 1) ∧ ¬∀ a b, (a + b > 2 ∧ ab > 1) → (a > 1 ∧ b > 1) :=
by
  sorry

end condition_A_sufficient_not_necessary_condition_B_l299_299581


namespace complete_square_form_l299_299868

theorem complete_square_form :
  ∀ x : ℝ, (3 * x^2 - 6 * x + 2 = 0) → (x - 1)^2 = (1 / 3) :=
by
  intro x h
  sorry

end complete_square_form_l299_299868


namespace range_of_a_plus_b_l299_299301

noncomputable def range_of_sum_of_sides (a b : ℝ) (c : ℝ) : Prop :=
  (2 < a + b ∧ a + b ≤ 4)

theorem range_of_a_plus_b
  (a b c : ℝ)
  (h1 : (2 * (b ^ 2 - (1/2) * a * b) = b ^ 2 + 4 - a ^ 2))
  (h2 : c = 2) :
  range_of_sum_of_sides a b c :=
by
  -- Proof would go here, but it's omitted as per the instructions.
  sorry

end range_of_a_plus_b_l299_299301


namespace find_m_max_f_eq_3_l299_299949

open Real

noncomputable def f (x m : ℝ) : ℝ := sqrt 3 * sin (2 * x) + 2 * (cos x)^2 + m

theorem find_m_max_f_eq_3 : ∃ m : ℝ, (∀ x ∈ Icc 0 (π / 2), f x m ≤ 3) ∧ (∃ x ∈ Icc 0 (π / 2), f x m = 3) := 
sorry

end find_m_max_f_eq_3_l299_299949


namespace clock_angle_at_5_15_l299_299870

noncomputable def hour_hand_angle (hours : ℕ) (minutes : ℕ) : ℕ :=
  (hours % 12) * 30 + minutes / 2

noncomputable def minute_hand_angle (minutes : ℕ) : ℕ :=
  minutes * 6

theorem clock_angle_at_5_15 :
  let hour_angle := hour_hand_angle 5 15 in
  let minute_angle := minute_hand_angle 15 in
  abs (hour_angle - minute_angle) = 67.5 := by
  sorry

end clock_angle_at_5_15_l299_299870


namespace weight_of_new_person_l299_299739

theorem weight_of_new_person :
  (∀ (weights : Fin 12 → ℝ) (new_weight : ℝ), 
   let old_avg := (∑ i, weights i) / 12 in
   let new_avg := (old_avg * 12 - 58 + new_weight) / 12 in
   new_avg = old_avg + 4 → new_weight = 106) :=
by intros weights new_weight old_avg new_avg h; sorry

end weight_of_new_person_l299_299739


namespace num_cows_on_farm_l299_299271

variables (D C S : ℕ)

def total_legs : ℕ := 8 * S + 2 * D + 4 * C
def total_heads : ℕ := D + C + S

theorem num_cows_on_farm
  (h1 : S = 2 * D)
  (h2 : total_legs D C S = 2 * total_heads D C S + 72)
  (h3 : D + C + S ≤ 40) :
  C = 30 :=
sorry

end num_cows_on_farm_l299_299271


namespace problem_solution_l299_299800

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 2 * cos (2 * x - π / 4)

theorem problem_solution :
  (∀ k : ℤ, ∀ x ∈ Icc (k * π - 3 * π / 8) (k * π + π / 8), 0 ≤ (2 * x - π / 4)) ∧
  (∀ x ∈ Icc (-π / 8) (π / 2), 
    (f x = -1 ↔ x = π / 2) ∧
    (f x = sqrt 2 ↔ x = π / 8)) :=
by
  sorry

end problem_solution_l299_299800


namespace range_of_a_for_monotonic_increase_l299_299748

def isMonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≤ f y

def piecewiseFunction (a : ℝ) : ℝ → ℝ
| x => if x ≤ 1 then (a - 2) * x - 1 else a^(x - 1)

theorem range_of_a_for_monotonic_increase 
  (a : ℝ) 
  (h : isMonotonicallyIncreasing (piecewiseFunction a)) : 
  2 < a ∧ a ≤ 4 := 
sorry

end range_of_a_for_monotonic_increase_l299_299748


namespace find_k_eccentricity_l299_299222

theorem find_k_eccentricity :
  ∀ (k : ℝ), 
  ∃ e : ℝ, e = 4 / 5 ∧ 
    ( (∃ a b : ℝ, a^2 = 9 ∧ b^2 = 4 + k ∧ e = (sqrt (5 - k)) / 3) ∨ 
      (∃ a b : ℝ, a^2 = 4 + k ∧ b^2 = 9 ∧ e = (sqrt (k - 5)) / sqrt (4 + k) )
    ) →
    (k = -19 / 25 ∨ k = 21) := 
by
  intro k
  use 4 / 5
  split
  . rfl
  . intro h
    sorry

end find_k_eccentricity_l299_299222


namespace incorrect_correlation_statement_l299_299787

/--
  The correlation coefficient measures the degree of linear correlation between two variables. 
  The linear correlation coefficient is a quantity whose absolute value is less than 1. 
  Furthermore, the larger its absolute value, the greater the degree of correlation.

  Let r be the sample correlation coefficient.

  We want to prove that the statement "D: |r| ≥ 1, and the closer |r| is to 1, the greater the degree of correlation" 
  is incorrect.
-/
theorem incorrect_correlation_statement (r : ℝ) (h1 : |r| ≤ 1) : ¬ (|r| ≥ 1) :=
by
  -- Proof steps go here
  sorry

end incorrect_correlation_statement_l299_299787


namespace isaac_ribbon_length_l299_299307

variable (part_length : ℝ) (total_length : ℝ := part_length * 6) (unused_length : ℝ := part_length * 2)

theorem isaac_ribbon_length
  (total_parts : ℕ := 6)
  (used_parts : ℕ := 4)
  (not_used_parts : ℕ := total_parts - used_parts)
  (not_used_length : Real := 10)
  (equal_parts : total_length / total_parts = part_length) :
  total_length = 30 := by
  sorry

end isaac_ribbon_length_l299_299307


namespace lorry_empty_weight_l299_299493

-- Define variables for the weights involved
variable (lw : ℕ)  -- weight of the lorry when empty
variable (bl : ℕ)  -- number of bags of apples
variable (bw : ℕ)  -- weight of each bag of apples
variable (total_weight : ℕ)  -- total loaded weight of the lorry

-- Given conditions
axiom lorry_loaded_weight : bl = 20 ∧ bw = 60 ∧ total_weight = 1700

-- The theorem we want to prove
theorem lorry_empty_weight : (∀ lw bw, total_weight - bl * bw = lw) → lw = 500 :=
by
  intro h
  rw [←h lw bw]
  sorry

end lorry_empty_weight_l299_299493


namespace value_of_S_l299_299547

def pseudocode_value : ℕ := 1
def increment (S I : ℕ) : ℕ := S + I

def loop_steps : ℕ :=
  let S := pseudocode_value
  let S := increment S 1
  let S := increment S 3
  let S := increment S 5
  let S := increment S 7
  S

theorem value_of_S : loop_steps = 17 :=
  by sorry

end value_of_S_l299_299547


namespace sum_of_leading_digits_of_roots_is_11_l299_299324

def M : ℕ := 888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888
-- other 450 8 digits omitted for brevity

def g (r : ℕ) (n : ℕ) : ℕ :=
  let root : ℚ := ↑n^(1.0 / r : ℚ)
  (root / 10^(root.floor.log10 : ℚ)).floor

theorem sum_of_leading_digits_of_roots_is_11 : 
  g 2 M + g 3 M + g 4 M + g 5 M + g 6 M + g 7 M = 11 :=
by
  sorry

end sum_of_leading_digits_of_roots_is_11_l299_299324


namespace committee_probability_l299_299738

theorem committee_probability :
  (∃ m n : ℕ, m = 24 ∧ n = 5 ∧ 14 > 0 ∧ 10 > 0 ∧
  (let boys := 14 in
   let girls := 10 in
   let com := m.choose n in
   let favorable := (boys.choose 3) * (girls.choose 2) +
                    (boys.choose 4) * (girls.choose 1) +
                    (boys.choose 5) in
   com = 42504 ∧ favorable = 28392 ∧
   (favorable : ℚ) / com = (7098 : ℚ) / 10626)) := sorry

end committee_probability_l299_299738


namespace children_selection_l299_299273

-- Conditions and definitions
def comb (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Proof problem statement
theorem children_selection : ∃ r : ℕ, comb 10 r = 210 ∧ r = 4 :=
by
  sorry

end children_selection_l299_299273


namespace binom_n_2_l299_299151

theorem binom_n_2 (n : ℕ) (h : 2 ≤ n) : nat.choose n 2 = n * (n - 1) / 2 := 
by sorry

end binom_n_2_l299_299151


namespace largest_integer_and_multiples_l299_299776

theorem largest_integer_and_multiples :
  ∃ (x : ℤ), ((7 - 3 * x > 23) ∧ (∀ y,  y ∈ (range (1:ℤ).succ 3).map (λ n, x^2 * n) ↔ y = 36 ∨ y = 72 ∨ y = 108)) ↔ (x = -6) := by
  sorry

end largest_integer_and_multiples_l299_299776


namespace samantha_saves_l299_299724

variables (fuel_eff_old fuel_cost_gas : ℝ) (new_efficiency_factor : ℝ := 1.3) (cost_reduction_factor : ℝ := 0.9)

def fuel_eff_new := new_efficiency_factor * fuel_eff_old
def cost_per_liter_biofuel := cost_reduction_factor * fuel_cost_gas

def cost_per_km_old := fuel_cost_gas * (1 / fuel_eff_old)
def cost_per_km_new := cost_per_liter_biofuel * (1 / fuel_eff_new)

def percent_saving : ℝ :=
  100 * (cost_per_km_old - cost_per_km_new) / cost_per_km_old

theorem samantha_saves : percent_saving fuel_eff_old fuel_cost_gas = 30.77 :=
  sorry

end samantha_saves_l299_299724


namespace product_of_distances_equal_l299_299481

/--
Given a \(2n\)-gon \(A_1, A_2, \ldots, A_{2n}\) inscribed in a circle and an arbitrary point \(M\) on the circle,
let \(p_1, p_2, \ldots, p_{2n}\) be the distances from \(M\) to the sides \(A_1A_2, A_2A_3, \ldots, A_{2n}A_1\).
Prove that \(p_1 p_3 \ldots p_{2n-1} = p_2 p_4 \ldots p_{2n}\).
-/
theorem product_of_distances_equal (n : ℕ) (A : fin (2 * n) → ℝ × ℝ) (M : ℝ × ℝ)
  (d : fin (2 * n) → ℝ) (h : ∀ i : fin (2 * n), d i = distance_from_point_to_side M (A i) (A (i + 1) % (2 * n))) :
  (finset.range (2 * n)).filter (λ i, odd i).prod d = (finset.range (2 * n)).filter (λ i, ¬ odd i).prod d := 
sorry

end product_of_distances_equal_l299_299481


namespace square_assembly_possible_l299_299129

theorem square_assembly_possible (Area1 Area2 Area3 : ℕ) (h1 : Area1 = 29) (h2 : Area2 = 18) (h3 : Area3 = 10) (h_total : Area1 + Area2 + Area3 = 57) : 
  ∃ s : ℝ, s^2 = 57 ∧ true :=
by
  sorry

end square_assembly_possible_l299_299129


namespace min_value_sum_inverse_squares_l299_299691

theorem min_value_sum_inverse_squares (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_sum : a + b + c = 3) :
    (1 / (a + b)^2) + (1 / (a + c)^2) + (1 / (b + c)^2) >= 3 / 2 :=
sorry

end min_value_sum_inverse_squares_l299_299691


namespace repeating_decimal_as_fraction_l299_299548

theorem repeating_decimal_as_fraction :
  let x := 0.134134134 -- definition of the repeating decimal
  in x = 134 / 999 :=
by
  sorry

end repeating_decimal_as_fraction_l299_299548


namespace f_is_int_for_all_naturals_f_divisible_by_m_for_all_naturals_l299_299073

-- Conditions and definitions for part a)
variable (q : ℕ) 
noncomputable def f (x : ℕ) := (c : ℤ) * q^x + 
  (a_n : ℤ) * x^n + 
  (a_(n-1) : ℤ) * x^(n-1) +
  -- ... -- Define all coefficients down to a0
  + (a_0 : ℤ)

-- Assuming integer values at specific points
axiom f_0_is_int : f q 0 ∈ ℤ
axiom f_1_is_int : f q 1 ∈ ℤ

-- Proof goal for part (a)
theorem f_is_int_for_all_naturals : ∀ x : ℕ, f q x ∈ ℤ := 
  sorry -- proof goes here

-- Extra conditions for part (b)
variable (m : ℕ)
axiom m_geq_1 : m ≥ 1
axiom f_mod_m_at_0 : f q 0 % m = 0
axiom f_mod_m_at_1 : f q 1 % m = 0
axiom f_mod_m_at_2 : f q 2 % m = 0
-- ... - Define all necessary conditions up to n+1
axiom f_mod_m_at_n1 : f q (n+1) % m = 0

-- Proof goal for part (b)
theorem f_divisible_by_m_for_all_naturals : ∀ x : ℕ, f q x % m = 0 := 
  sorry -- proof goes here

end f_is_int_for_all_naturals_f_divisible_by_m_for_all_naturals_l299_299073


namespace fraction_sum_log_eq_one_l299_299187

theorem fraction_sum_log_eq_one (m n : ℝ) (hm : 2^m = 10) (hn : 5^n = 10) : 
    (1/m) + (1/n) = 1 :=
by 
  sorry

end fraction_sum_log_eq_one_l299_299187


namespace division_proof_l299_299857

namespace DivisionProblem

def eight_digit_number : Type := 
  { n : ℕ // 10^7 ≤ n ∧ n < 10^8 }

def three_digit_number : Type :=
  { d : ℕ // 100 ≤ d ∧ d < 1000 }

noncomputable def quotient (N : eight_digit_number) (D : three_digit_number) : ℕ :=
  N.val / D.val

theorem division_proof (N : eight_digit_number) (D : three_digit_number) (hD : D.val = 142) : quotient N D = 70709 :=
by
  have hN : N.val = _ := sorry -- Here we assume the specific value for N, which will be filled in the proof
  rw [hD]
  sorry

end DivisionProblem

end division_proof_l299_299857


namespace non_working_games_count_l299_299351

-- Definitions based on conditions
def totalGames : ℕ := 16
def pricePerGame : ℕ := 7
def totalEarnings : ℕ := 56

-- Statement to prove
theorem non_working_games_count : 
  totalGames - (totalEarnings / pricePerGame) = 8 :=
by
  sorry

end non_working_games_count_l299_299351


namespace num_valid_paintings_l299_299516

def paint_faces_valid (faces : set ℕ) : Prop :=
  ¬(∃ a b ∈ faces, a + b = 7)

def valid_paintings (die_faces : finset ℕ) : finset (finset ℕ) :=
  (die_faces.powerset.filter (λ s, s.card = 3 ∧ paint_faces_valid s : Prop))

theorem num_valid_paintings : valid_paintings {1, 2, 3, 4, 5, 6}.card = 8 := 
sorry

end num_valid_paintings_l299_299516


namespace existence_of_xyz_l299_299922

theorem existence_of_xyz (n : ℕ) (hn_pos : 0 < n)
    (a b c : ℕ) (ha : 0 < a ∧ a ≤ 3 * n^2 + 4 * n) 
                (hb : 0 < b ∧ b ≤ 3 * n^2 + 4 * n) 
                (hc : 0 < c ∧ c ≤ 3 * n^2 + 4 * n) : 
  ∃ (x y z : ℤ), (|x| ≤ 2 * n) ∧ (|y| ≤ 2 * n) ∧ (|z| ≤ 2 * n) ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧ a * x + b * y + c * z = 0 := by
  sorry

end existence_of_xyz_l299_299922


namespace weight_of_one_bag_l299_299048

noncomputable def total_harvested_apples : ℕ := 405
noncomputable def apples_used_for_juice : ℕ := 90
noncomputable def apples_given_to_restaurant : ℕ := 60
noncomputable def total_revenue_from_bags : ℕ := 408
noncomputable def selling_price_per_bag : ℕ := 8

theorem weight_of_one_bag : 
  let apples_sold := total_harvested_apples - apples_used_for_juice - apples_given_to_restaurant,
      number_of_bags := total_revenue_from_bags / selling_price_per_bag
  in apples_sold / number_of_bags = 5 
:= by
  sorry

end weight_of_one_bag_l299_299048


namespace value_of_expression_l299_299190

-- Defining the conditions
variables (a b c : ℝ)
hypothesis ha : a ≠ 3
hypothesis hb : b ≠ 4
hypothesis hc : c ≠ 5

-- The Lean proof statement for the problem
theorem value_of_expression (ha : a ≠ 3) (hb : b ≠ 4) (hc : c ≠ 5) :
  (a-3) / (6-c) * (b-4) / (3-a) * (c-5) / (4-b) = -1 :=
by
  sorry

end value_of_expression_l299_299190


namespace melony_profit_l299_299350

theorem melony_profit (profit_3_shirts : ℝ)
  (profit_2_sandals : ℝ)
  (h1 : profit_3_shirts = 21)
  (h2 : profit_2_sandals = 4 * 21) : profit_3_shirts / 3 * 7 + profit_2_sandals / 2 * 3 = 175 := 
by 
  sorry

end melony_profit_l299_299350


namespace length_of_rectangle_l299_299755

theorem length_of_rectangle (s : ℝ) (l : ℝ) (h1 : 4 * s = 2 * l + 24) (h2 : (Real.pi * s) / 2 ≈ 21.99) : l ≈ 16 := by
  -- Proof is omitted as requested
  sorry

end length_of_rectangle_l299_299755


namespace alex_loan_difference_l299_299508

theorem alex_loan_difference :
  let P := (15000 : ℝ)
  let r1 := (0.08 : ℝ)
  let n := (2 : ℕ)
  let t := (12 : ℕ)
  let r2 := (0.09 : ℝ)
  
  -- Calculate the amount owed after 6 years with compound interest (first option)
  let A1_half := P * (1 + r1 / n)^(n * t / 2)
  let half_payment := A1_half / 2
  let remaining_balance := A1_half / 2
  let A1_final := remaining_balance * (1 + r1 / n)^(n * t / 2)
  
  -- Total payment for the first option
  let total1 := half_payment + A1_final
  
  -- Total payment for the second option (simple interest)
  let simple_interest := P * r2 * t
  let total2 := P + simple_interest
  
  -- Compute the positive difference
  let difference := abs (total1 - total2)
  
  difference = 24.59 :=
  by
  sorry

end alex_loan_difference_l299_299508


namespace indefinite_integral_correct_l299_299476

-- Define the integrand as a function
def integrand (x : ℝ) : ℝ := (x^3 + 6 * x^2 + 10 * x + 10) / ((x - 1) * (x + 2)^3)

-- Define the antiderivative function as the found solution
def antiderivative (x : ℝ) : ℝ := log (abs (x - 1)) + (1 / (x + 2)^2)

-- State that the indefinite integral of the integrand is the antiderivative plus a constant
theorem indefinite_integral_correct (C : ℝ) : 
  ∀ x, ∫ integrand x dx = antiderivative x + C :=
begin
  sorry
end

end indefinite_integral_correct_l299_299476


namespace sum_of_roots_of_given_quadratic_l299_299628

theorem sum_of_roots_of_given_quadratic : 
  ∀ x : ℝ, (x + 3) * (x - 5) = 19 → 
  let p : (ℝ → Prop) := λ x, (x+3)*(x-5) = 19 in
  ∀ {a:ℝ} {b:ℝ}, 
    ∀ h : ∃ x1 x2:ℝ, 
        p x1 ∧ p x2 ∧ 
        (x-x1)*(x-x2) = 0 ∧
        (∀ h1 h2 : x1 ≠ x2, 
          a * (x^2) + b * x - (19 + 3 * (5 + 3)) = 0) →
        -b/a = x1 + x2 
 := by 
  sorry

end sum_of_roots_of_given_quadratic_l299_299628


namespace best_ketchup_deal_l299_299863

def cost_per_ounce (price : ℝ) (ounces : ℝ) : ℝ := price / ounces

theorem best_ketchup_deal :
  let price_10oz := 1 in
  let ounces_10oz := 10 in
  let price_16oz := 2 in
  let ounces_16oz := 16 in
  let price_25oz := 2.5 in
  let ounces_25oz := 25 in
  let price_50oz := 5 in
  let ounces_50oz := 50 in
  let price_200oz := 10 in
  let ounces_200oz := 200 in
  let money := 10 in
  (∀ p o, cost_per_ounce p o ≥ cost_per_ounce price_200oz ounces_200oz) ∧ money = price_200oz :=
1
by
  sorry

end best_ketchup_deal_l299_299863


namespace basis_of_plane_l299_299219

noncomputable theory

variable {V : Type*} [add_comm_group V] [module ℝ V]

variables (a b : V)

-- Define the given conditions
def non_zero_vector (v : V) : Prop := v ≠ 0

def not_collinear (u v : V) : Prop := ∀ (k : ℝ), u ≠ k • v

-- Lean Statement
theorem basis_of_plane (ha : non_zero_vector a) (hb : non_zero_vector b) (h_not_collinear : not_collinear a b) :
  ∀ (u v : V), u = a + b → v = a - b → 
  ∀ (w : V), ∃ (k1 k2 : ℝ), w = k1 • u + k2 • v :=
by
  -- Define the vectors u and v
  intros u v hu hv w,
  -- TODO: prove that w can be written as a linear combination of u and v
  sorry

end basis_of_plane_l299_299219


namespace rectangle_area_k_value_l299_299023

theorem rectangle_area_k_value (d : ℝ) (length width : ℝ) (h1 : 5 * width = 2 * length) (h2 : d^2 = length^2 + width^2) :
  ∃ (k : ℝ), A = k * d^2 ∧ k = 10 / 29 :=
by
  sorry

end rectangle_area_k_value_l299_299023


namespace pat_oj_consumption_l299_299373

def initial_oj : ℚ := 3 / 4
def alex_fraction : ℚ := 1 / 2
def pat_fraction : ℚ := 1 / 3

theorem pat_oj_consumption : pat_fraction * (initial_oj * (1 - alex_fraction)) = 1 / 8 := by
  -- This will be the proof part which can be filled later
  sorry

end pat_oj_consumption_l299_299373


namespace segment_length_is_ten_l299_299882

-- Definition of the cube root function and the absolute value
def cube_root (x : ℝ) : ℝ := x^(1/3)

def absolute (x : ℝ) : ℝ := abs x

-- The prerequisites as conditions for the endpoints
def endpoints_satisfy (x : ℝ) : Prop := absolute (x - cube_root 27) = 5

-- Length of the segment determined by the endpoints
def segment_length (x1 x2 : ℝ) : ℝ := absolute (x2 - x1)

-- Theorem statement
theorem segment_length_is_ten : (∀ x, endpoints_satisfy x) → segment_length (-2) 8 = 10 :=
by
  intro h
  sorry

end segment_length_is_ten_l299_299882


namespace solve_exponential_equation_l299_299415

-- Define the problem as a Lean statement
theorem solve_exponential_equation (x : ℝ) : 
  (4^x - 2^x - 6 = 0) ↔ x = Real.log 3 / Real.log 2 := 
sorry

end solve_exponential_equation_l299_299415


namespace spider_legs_spider_legs_distinct_l299_299274

theorem spider_legs {n : ℕ} (h : n = 8) : 
  (∑ i in finset.range (n + 1), i) = 36 :=
by
  have h1 : finset.range (n + 1) = {0, 1, 2, 3, 4, 5, 6, 7, 8} := by simp [h]
  calc
    ∑ i in finset.range (n + 1), i
        = ∑ i in {0, 1, 2, 3, 4, 5, 6, 7, 8}, i : by rw h1
    ... = 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8   : by simp
    ... = 36                                 : by norm_num

-- To represent "distinct numbers from 1 to 8":
theorem spider_legs_distinct : 
  (∑ i in finset.range 9) - 0 = 36 :=
by
  calc
    (∑ i in finset.range 9) - 0
        = ∑ i in finset.range 9, i : by rw nat.sub_zero
    ... = 36                       : by norm_num

end spider_legs_spider_legs_distinct_l299_299274


namespace new_oranges_added_l299_299831

def initial_oranges : Nat := 31
def thrown_away_oranges : Nat := 9
def final_oranges : Nat := 60
def remaining_oranges : Nat := initial_oranges - thrown_away_oranges
def new_oranges (initial_oranges thrown_away_oranges final_oranges : Nat) : Nat := 
  final_oranges - (initial_oranges - thrown_away_oranges)

theorem new_oranges_added :
  new_oranges initial_oranges thrown_away_oranges final_oranges = 38 := by
  sorry

end new_oranges_added_l299_299831


namespace find_xy_l299_299534

noncomputable def star (a b c d : ℝ) : ℝ × ℝ :=
  (a * c + b * d, a * d + b * c)

theorem find_xy (a b x y : ℝ) (h : star a b x y = (a, b)) (h' : a^2 ≠ b^2) : (x, y) = (1, 0) :=
  sorry

end find_xy_l299_299534


namespace division_correct_l299_299140

theorem division_correct : 0.45 / 0.005 = 90 := by
  sorry

end division_correct_l299_299140


namespace gerald_paid_l299_299963

theorem gerald_paid (G : ℝ) (h : 0.8 * G = 200) : G = 250 := by
  sorry

end gerald_paid_l299_299963


namespace min_cos_C_l299_299986

-- Given definitions and conditions
variables (A B C : Type) [triangle : Triangle A B C]
#check Triangle -- check the type of triangle

-- Given dot product equation in the triangle
variable (h1 : (Triangle.dotProduct A B + 2 * Triangle.dotProduct B C) = 3 * Triangle.dotProduct C A)

-- Prove the minimum value of cos C
theorem min_cos_C (h1 : (Triangle.dotProduct A B + 2 * Triangle.dotProduct B C) = 3 * Triangle.dotProduct C A) : 
  Triangle.cosineC (Triangle.sideLength A C) (Triangle.sideLength B C) (Triangle.sideLength A B) = sqrt(2) / 3 := sorry

end min_cos_C_l299_299986


namespace correct_statements_l299_299148

theorem correct_statements:
  let y (x : ℝ) := sin (x / 2) + sqrt 3 * cos (x / 2) in
  let A := 120 in 
  let AB := 5 in
  let BC := 7 in
  (∀ x : ℝ, y x = 2 * sin (x / 2 + π / 3) → x = π / 3) ∧
  (∀ {a b A B : ℝ}, b = 2 * a * sin B → A = 30 ∨ A = 150 → false) ∧
  (area (sin A) = 15 * sqrt 3 / 4) ∧
  (sin 70 * cos 40 * cos 60 * cos 80 ≠ 1 / 8) :=
sorry

end correct_statements_l299_299148


namespace caleb_total_hamburgers_l299_299127

def total_hamburgers (total_spent single_cost double_cost double_burgers : ℕ → ℝ) : ℕ :=
  let total_cost := total_spent 64.50
  let cost_single := single_cost 1.0
  let cost_double := double_cost 1.5
  let num_double := double_burgers 29 
  let cost_double_total := num_double * cost_double
  let cost_single_total := total_cost - cost_double_total
  let num_single := cost_single_total / cost_single
  (num_single + num_double).to_nat

theorem caleb_total_hamburgers : total_hamburgers 64.50 1.0 1.5 29 = 50 := by
  sorry

end caleb_total_hamburgers_l299_299127


namespace fraction_irreducible_l299_299366

theorem fraction_irreducible (n : ℕ) : Nat.gcd (12 * n + 1) (30 * n + 1) = 1 :=
sorry

end fraction_irreducible_l299_299366


namespace square_assembly_possible_l299_299128

theorem square_assembly_possible (Area1 Area2 Area3 : ℕ) (h1 : Area1 = 29) (h2 : Area2 = 18) (h3 : Area3 = 10) (h_total : Area1 + Area2 + Area3 = 57) : 
  ∃ s : ℝ, s^2 = 57 ∧ true :=
by
  sorry

end square_assembly_possible_l299_299128


namespace simplest_quadratic_radical_among_choices_l299_299056

noncomputable def is_simplest_quadratic_radical (e : ℝ) : Prop :=
  e = real.sqrt 3

theorem simplest_quadratic_radical_among_choices :
  let a := real.sqrt (1 / 2)
  let b := real.sqrt 3
  let c := real.sqrt 8
  let d := real.sqrt 0.1
  is_simplest_quadratic_radical b :=
by
  sorry

end simplest_quadratic_radical_among_choices_l299_299056


namespace difference_between_length_and_breadth_l299_299428

theorem difference_between_length_and_breadth (L W : ℝ) (h1 : W = 1/2 * L) (h2 : L * W = 800) : L - W = 20 :=
by
  sorry

end difference_between_length_and_breadth_l299_299428


namespace initially_calculated_average_weight_l299_299003

theorem initially_calculated_average_weight (n : ℕ) (misread : ℕ) (correct : ℕ) (correct_avg : ℝ) :
  n = 20 → misread = 56 → correct = 60 → correct_avg = 58.6 → 
  let correct_total_weight := correct_avg * n in
  let misread_difference := correct - misread in
  let initial_total_weight := correct_total_weight - misread_difference in
  let initial_avg := initial_total_weight / n in
  initial_avg = 58.4 :=
by
  intros hn hmisread hcorrect hcorrect_avg
  dsimp only [correct_total_weight, misread_difference, initial_total_weight, initial_avg]
  rw [hn, hmisread, hcorrect]
  rw [hcorrect_avg]
  norm_num at *
  sorry

end initially_calculated_average_weight_l299_299003


namespace domain_of_f_l299_299443

def f (x : ℝ) : ℝ := (x^2 - 49) / (x - 7)

theorem domain_of_f :
  {x : ℝ | f x ≠ real.div_zero} = {x : ℝ | x ≠ 7} :=
by
  sorry

end domain_of_f_l299_299443


namespace calculation_correct_l299_299525

theorem calculation_correct : 
  sqrt 6 * (sqrt 2 - sqrt 3 + sqrt 6) - abs (3 * sqrt 2 - 6) = 2 * sqrt 3 := 
sorry

end calculation_correct_l299_299525


namespace interval_of_decrease_l299_299897

noncomputable def f (x : ℝ) : ℝ := ln x - x^2 + x

theorem interval_of_decrease : { x : ℝ | x > (1/2) } ⊆ { x : ℝ | 1/x - 2*x + 1 < 0 } :=
sorry

end interval_of_decrease_l299_299897


namespace find_measure_of_A_and_value_of_a_l299_299297

noncomputable def angle_A (a b c : ℝ) (C : ℝ) : ℝ :=
  if sqrt (3:ℝ) * c * cos C / a + sin C = sqrt (3:ℝ) * c then π / 3 else 0

noncomputable def side_a (b c : ℝ) (area: ℝ) : ℝ :=
  if sqrt (3:ℝ) = (1/2) * b * c * (sqrt (3:ℝ) / 2) then sqrt (13:ℝ) else 0

theorem find_measure_of_A_and_value_of_a
  (a b c : ℝ) (A C : ℝ) (area : ℝ)
  (h1 : sqrt (3:ℝ) * c * cos A + a * sin C = sqrt (3:ℝ) * c)
  (h2 : b + c = 5)
  (h3 : area = sqrt (3:ℝ)) :
  A = π / 3 ∧ a = sqrt (13:ℝ) := by
  sorry

end find_measure_of_A_and_value_of_a_l299_299297


namespace powderman_ran_when_heard_blast_l299_299823

noncomputable def distance_powderman_ran
  (time_for_blast : ℝ)   -- Time for the blast to take place
  (rate_powderman : ℝ)   -- Rate at which the powderman runs in yards per second
  (rate_sound : ℝ)   -- Rate at which the sound travels in feet per second
  (time_heard_blast : ℝ) : ℝ :=
  rate_powderman * time_heard_blast

theorem powderman_ran_when_heard_blast :
  ∀ (time_for_blast : ℝ) (rate_powderman : ℝ) (rate_sound : ℝ),
  time_for_blast = 45 → 
  rate_powderman = 10 →
  rate_sound = 1080 →
  ∃(time_heard_blast : ℝ),
  time_heard_blast = 46.29 →
  distance_powderman_ran time_for_blast rate_powderman rate_sound time_heard_blast / 3 ≈ 463 :=
by {
  sorry,
}

end powderman_ran_when_heard_blast_l299_299823


namespace audience_D_guessed_correctly_l299_299655

-- Definitions for contestant numbers
def Contestant : Type := ℕ

-- Definitions for audience members' guesses
def A_guesses (c : Contestant) : Prop := (c = 4 ∨ c = 5)
def B_guesses (c : Contestant) : Prop := (c ≠ 3)
def C_guesses (c : Contestant) : Prop := (c = 1 ∨ c = 2 ∨ c = 6)
def D_guesses (c : Contestant) : Prop := (c ≠ 4 ∨ c ≠ 5 ∨ c ≠ 6)

-- Declare the condition that only one of A, B, C, D guessed correctly
def exactly_one_guessed_correctly (c : Contestant) : Prop :=
  (A_guesses c ∧ ¬B_guesses c ∧ ¬C_guesses c ∧ ¬D_guesses c) ∨
  (¬A_guesses c ∧ B_guesses c ∧ ¬C_guesses c ∧ ¬D_guesses c) ∨
  (¬A_guesses c ∧ ¬B_guesses c ∧ C_guesses c ∧ ¬D_guesses c) ∨
  (¬A_guesses c ∧ ¬B_guesses c ∧ ¬C_guesses c ∧ D_guesses c)

-- Main theorem statement
theorem audience_D_guessed_correctly : ∃ c : Contestant, D_guesses c ∧ exactly_one_guessed_correctly c :=
  sorry

end audience_D_guessed_correctly_l299_299655


namespace minimum_value_of_f_l299_299609

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x + 12

theorem minimum_value_of_f :
  ∃ x ∈ set.Icc (-3 : ℝ) (3 : ℝ), ∀ y ∈ set.Icc (-3 : ℝ) (3 : ℝ), f x ≤ f y :=
sorry

end minimum_value_of_f_l299_299609


namespace work_done_in_lifting_satellite_l299_299152

-- Definitions for the conditions
def m : ℝ := 3000 -- mass in kg
def R₃ : ℝ := 6.38e6 -- radius of Earth in meters
def H : ℝ := 600e3 -- height in meters
def g : ℝ := 10 -- acceleration due to gravity in m/s^2

-- Proven theorem
theorem work_done_in_lifting_satellite :
  let A := (m * g * R₃^2) * ((1 / R₃) - (1 / (R₃ + H))) in
  A = 1.6452722063e10 :=
by
  sorry

end work_done_in_lifting_satellite_l299_299152


namespace exterior_angle_DEF_is_135_l299_299078

open EuclideanGeometry

noncomputable def exterior_angle_DEF : ℝ :=
  360 - 90 - 135

theorem exterior_angle_DEF_is_135
  (ABCDE : Type) (AEFGHIJK : Type)
  (coplanar : coplanar ABCDE AEFGHIJK)
  (AE : Type) (opposite_sides : opposite_sides ABCDE AEFGHIJK AE)
  (int_angle_square : interior_angle ABCDE = 90)
  (int_angle_octagon : interior_angle AEFGHIJK = 135) :
  exterior_angle_DEF = 135 :=
by simp [exterior_angle_DEF]

end exterior_angle_DEF_is_135_l299_299078


namespace grasshopper_proximity_to_zero_l299_299814

-- Definitions
variables {a b : ℝ} (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_irrational : ¬∃ (ratio : ℚ), b = a * ratio)

-- Theorem statement
theorem grasshopper_proximity_to_zero 
  (initial_position : ℝ) 
  (jump_right : Π (pos : ℝ), pos < -a → pos + a) 
  (jump_left : Π (pos : ℝ), -a / 2 ≤ pos ∧ pos ≤ b → pos - b)
  : ∃ (n : ℕ), abs (initial_position + n * (a - b)) < 10 ^ (-6) :=
by
  sorry

end grasshopper_proximity_to_zero_l299_299814


namespace gerald_paid_l299_299960

theorem gerald_paid (G : ℝ) (h : 0.8 * G = 200) : G = 250 :=
by
  sorry

end gerald_paid_l299_299960


namespace prism_volume_l299_299201

noncomputable def find_prism_volume (a b c : ℝ) : ℝ := (a * b * c) / (2 * sqrt 3)

theorem prism_volume :
  ∀ (a b c : ℝ),
  (√3 / 4 * a * b = √3 / 2) ∧ 
  (√3 / 4 * b * c = 2) ∧ 
  (√3 / 4 * c * a = 1) ∧ 
  (a = 1) ∧ (b = 2) ∧ (c = 4 / √3) →
  find_prism_volume a b c = (4 * sqrt(6)) / 3 :=
by
  intros a b c h
  sorry

end prism_volume_l299_299201


namespace a_is_minus_one_l299_299587

theorem a_is_minus_one (a : ℤ) (h1 : 2 * a + 1 < 0) (h2 : 2 + a > 0) : a = -1 := 
by
  sorry

end a_is_minus_one_l299_299587


namespace rectangular_garden_length_l299_299101

theorem rectangular_garden_length (w l : ℕ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 900) : l = 300 :=
by
  sorry

end rectangular_garden_length_l299_299101


namespace passengers_total_l299_299686

theorem passengers_total (a b : ℕ) (h1 : a = 14507) (h2 : b = 213) : a + b = 14720 :=
by
  rw [h1, h2]
  rfl

end passengers_total_l299_299686


namespace meeting_point_distance_proof_l299_299438

theorem meeting_point_distance_proof :
  ∃ x t : ℕ, (5 * t + (t * (7 + t) / 2) = 85) ∧ x = 9 :=
by
  sorry

end meeting_point_distance_proof_l299_299438


namespace coefficient_of_x5_in_expansion_l299_299005

theorem coefficient_of_x5_in_expansion : 
  ∀ (2x - sqrt(x))^8, coefficient_of_x5_in_expansion 8 (2 * x) (-sqrt x) = 112 := 
by
  sorry

end coefficient_of_x5_in_expansion_l299_299005


namespace Deepak_age_l299_299519

-- Define the current ages of Arun and Deepak
variable (A D : ℕ)

-- Define the conditions
def ratio_condition := A / D = 4 / 3
def future_age_condition := A + 6 = 26

-- Define the proof statement
theorem Deepak_age (h1 : ratio_condition A D) (h2 : future_age_condition A) : D = 15 :=
  sorry

end Deepak_age_l299_299519


namespace min_path_length_l299_299646

noncomputable def problem_statement : Prop :=
  let XY := 12
  let XZ := 8
  let angle_XYZ := 30
  let YP_PQ_QZ := by {
    -- Reflect Z across XY to get Z' and Y across XZ to get Y'.
    -- Use the Law of cosines in triangle XY'Z'.
    let cos_150 := -Real.sqrt 3 / 2
    let Y_prime_Z_prime := Real.sqrt (8^2 + 12^2 + 2 * 8 * 12 * cos_150)
    exact Y_prime_Z_prime
  }
  ∃ (P Q : Type), (YP_PQ_QZ = Real.sqrt (208 + 96 * Real.sqrt 3))

-- Goal is to prove the problem statement
theorem min_path_length : problem_statement := sorry

end min_path_length_l299_299646


namespace baseball_team_seating_l299_299991

-- Define blocks and constraints for the teams
def teams : Type := fin 3 -- Cubs, Red Sox, Yankees

-- Define the function to calculate factorial
def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the problem conditions
def blocks_arrangements := factorial 3 -- Arrangements of the blocks (3!)
def within_team_arrangements := factorial 3 -- Arrangements within each team (3!)

-- Calculate total arrangements
def total_arrangements := blocks_arrangements * (within_team_arrangements ^ 3)

-- Proof that the calculated arrangements equal the given answer 1296
theorem baseball_team_seating : total_arrangements = 1296 := 
by
  sorry

end baseball_team_seating_l299_299991


namespace sin_cos_property_l299_299946

theorem sin_cos_property 
  (α β : ℝ) 
  (h : (sin α)^6 / (sin β)^3 + (cos α)^6 / (cos β)^3 = 1) : 
  (cos β)^6 / (cos α)^3 + (sin β)^6 / (sin α)^3 = 1 := 
sorry

end sin_cos_property_l299_299946


namespace proof_F_2_f_3_l299_299693

def f (a : ℕ) : ℕ := a ^ 2 - 1

def F (a : ℕ) (b : ℕ) : ℕ := 3 * b ^ 2 + 2 * a

theorem proof_F_2_f_3 : F 2 (f 3) = 196 := by
  have h1 : f 3 = 3 ^ 2 - 1 := rfl
  rw [h1]
  have h2 : 3 ^ 2 - 1 = 8 := by norm_num
  rw [h2]
  exact rfl

end proof_F_2_f_3_l299_299693


namespace lambda_mu_sum_l299_299931

variable (A B C D E : Type) [AddCommGroup V] [Module ℝ V]
variables (V : Type) [add_comm_group V] [module ℝ V]

-- Definitions for the conditions
def parallelogram (AB CD AD BC : V) : Prop := AB + AD = AC ∧ BC + DA = AC
def AE_equation (AE EB : V) : Prop := AE = 2 • EB
def EC_decomposition (EC AB AD : V) (λ μ : ℝ) : Prop := EC = λ • AB + μ • AD

-- The theorem we want to prove
theorem lambda_mu_sum
  (AB CD AD BC AE EB EC AB AD : V)
  (h1 : parallelogram AB CD AD BC)
  (h2 : AE_equation AE EB)
  (h3 : EC_decomposition EC AB AD (λ : ℝ) (μ : ℝ)) :
  λ + μ = 4 / 3 := sorry

end lambda_mu_sum_l299_299931


namespace infinite_solutions_fractional_cubics_l299_299365

noncomputable def fractional_part (r : ℚ) : ℚ :=
r - r.floor

theorem infinite_solutions_fractional_cubics :
  ∃ (x y z : ℚ), (∀ r : ℚ, r.floor ≠ r) ∧ x^3 ≠ floor x^3 ∧ y^3 ≠ floor y^3 ∧ z^3 ≠ floor z^3 ∧
  (∀ k : ℤ, ∃ x y z : ℚ, fractional_part (x^3) + fractional_part (y^3) = fractional_part (z^3) ∧
  ∀ r : ℚ, r ≠ k) :=
sorry

end infinite_solutions_fractional_cubics_l299_299365


namespace recycling_drive_target_l299_299154

theorem recycling_drive_target :
  let section_A := 260
  let section_B := 290
  let section_C := 250
  let section_D := 270
  let section_E := 300
  let section_F := 310
  let section_G := 280
  let section_H := 265
  let additional_needed := 410
  let total_collected := section_A + section_B + section_C + section_D + section_E + section_F + section_G + section_H
  total_collected + additional_needed = 2635 :=
by
  let section_A := 260
  let section_B := 290
  let section_C := 250
  let section_D := 270
  let section_E := 300
  let section_F := 310
  let section_G := 280
  let section_H := 265
  let additional_needed := 410
  let total_collected := section_A + section_B + section_C + section_D + section_E + section_F + section_G + section_H
  show total_collected + additional_needed = 2635 from sorry

end recycling_drive_target_l299_299154


namespace arithmetic_seq_properties_l299_299598

noncomputable def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def c_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a n ^ 2 - a (n + 1) ^ 2

theorem arithmetic_seq_properties (a : ℕ → ℝ) (h_arith : arithmetic_seq a)
  (h_sum1 : ∑ i in finset.range 13, a (1 + 2 * i) = 130)
  (h_sum2 : ∑ i in finset.range 13, a (2 + 2 * i) = 117) :
  (∃ d : ℝ, c_seq a (n + 1) - c_seq a n = -2 * d ^ 2)
  ∧ ∃ d : ℝ, d = -1 
  ∧ ∃ b : ℕ → ℝ, b = λ n, 26 - n := 
sorry

end arithmetic_seq_properties_l299_299598


namespace employed_males_percentage_l299_299668

theorem employed_males_percentage (population : ℕ)
  (percent_employed : ℝ)
  (percent_employed_females : ℝ)
  (h_employed : percent_employed = 0.64)
  (h_females : percent_employed_females = 0.25) :
  let percent_employed_males := 1 - percent_employed_females in
  let percent_employed_males_population := percent_employed * percent_employed_males in
  percent_employed_males_population * 100 = 48 :=
by
  sorry

end employed_males_percentage_l299_299668


namespace total_steps_l299_299675

theorem total_steps (up_steps down_steps : ℕ) (h1 : up_steps = 567) (h2 : down_steps = 325) : up_steps + down_steps = 892 := by
  sorry

end total_steps_l299_299675


namespace karting_number_of_routes_l299_299659

theorem karting_number_of_routes :
  let M : ℕ → ℕ := λ n, Nat.fib (n + 1)
  in M 10 = 34 :=
by
  let M : ℕ → ℕ := λ n, Nat.fib (n + 1)
  show M 10 = 34
  exact Nat.fib_succ_succ 9

end karting_number_of_routes_l299_299659


namespace parallel_vectors_x_eq_one_l299_299243

/-- Given vectors a = (2x + 1, 3) and b = (2 - x, 1), prove that if they 
are parallel, then x = 1. -/
theorem parallel_vectors_x_eq_one (x : ℝ) :
  (∃ k : ℝ, (2 * x + 1) = k * (2 - x) ∧ 3 = k * 1) → x = 1 :=
by 
  sorry

end parallel_vectors_x_eq_one_l299_299243


namespace max_value_of_a_le_2_and_ge_neg_2_max_a_is_2_l299_299205

theorem max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : a ≤ 2 :=
by
  -- Proof omitted
  sorry

theorem le_2_and_ge_neg_2 (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : -2 ≤ a :=
by
  -- Proof omitted
  sorry

theorem max_a_is_2 (a : ℝ) (h3 : a ≤ 2) (h4 : -2 ≤ a) : a = 2 :=
by
  -- Proof omitted
  sorry

end max_value_of_a_le_2_and_ge_neg_2_max_a_is_2_l299_299205


namespace connected_graphs_bound_l299_299535

noncomputable def num_connected_graphs (n : ℕ) : ℕ := sorry
  
theorem connected_graphs_bound (n : ℕ) : 
  num_connected_graphs n ≥ (1/2) * 2^(n*(n-1)/2) := 
sorry

end connected_graphs_bound_l299_299535


namespace sum_of_first_seven_primes_with_units_digit_3_lt_150_l299_299175

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_less_than_150 (n : ℕ) : Prop :=
  n < 150

def first_seven_primes_with_units_digit_3 := [3, 13, 23, 43, 53, 73, 83]

theorem sum_of_first_seven_primes_with_units_digit_3_lt_150 :
  (has_units_digit_3 3) ∧ (is_less_than_150 3) ∧ (Prime 3) ∧
  (has_units_digit_3 13) ∧ (is_less_than_150 13) ∧ (Prime 13) ∧
  (has_units_digit_3 23) ∧ (is_less_than_150 23) ∧ (Prime 23) ∧
  (has_units_digit_3 43) ∧ (is_less_than_150 43) ∧ (Prime 43) ∧
  (has_units_digit_3 53) ∧ (is_less_than_150 53) ∧ (Prime 53) ∧
  (has_units_digit_3 73) ∧ (is_less_than_150 73) ∧ (Prime 73) ∧
  (has_units_digit_3 83) ∧ (is_less_than_150 83) ∧ (Prime 83) →
  (3 + 13 + 23 + 43 + 53 + 73 + 83 = 291) :=
by
  sorry

end sum_of_first_seven_primes_with_units_digit_3_lt_150_l299_299175


namespace stratified_sampling_male_employees_l299_299504

theorem stratified_sampling_male_employees
  (total_employees : ℕ) (female_employees : ℕ) (sample_size : ℕ)
  (h_total : total_employees = 120)
  (h_female : female_employees = 72)
  (h_sample : sample_size = 15) :
  let male_employees := total_employees - female_employees in
  let male_fraction := male_employees * sample_size / total_employees in
  male_fraction = 6 :=
by {
  let male_employees := total_employees - female_employees,
  have h_male : male_employees = 48 := by rw [h_total, h_female]; norm_num,
  let male_fraction := male_employees * sample_size / total_employees,
  have h_fraction : male_fraction = 6 := by rw [h_male, h_sample, h_total]; norm_num,
  exact h_fraction
}

end stratified_sampling_male_employees_l299_299504


namespace first_term_is_sqrt9_l299_299034

noncomputable def geometric_first_term (a r : ℝ) : ℝ :=
by
  have h1 : a * r^2 = 3 := by sorry
  have h2 : a * r^4 = 27 := by sorry
  have h3 : (a * r^4) / (a * r^2) = 27 / 3 := by sorry
  have h4 : r^2 = 9 := by sorry
  have h5 : r = 3 ∨ r = -3 := by sorry
  have h6 : (a * 9) = 3 := by sorry
  have h7 : a = 1/3 := by sorry
  exact a

theorem first_term_is_sqrt9 : geometric_first_term 3 9 = 3 :=
by
  sorry

end first_term_is_sqrt9_l299_299034


namespace sum_of_positive_divisors_200_l299_299421

theorem sum_of_positive_divisors_200 : 
  ∑ d in (Finset.filter (λ x, 200 % x = 0) (Finset.range 201)), d = 465 := 
by
  sorry

end sum_of_positive_divisors_200_l299_299421


namespace nice_table_greatest_k_l299_299496

open Matrix
open Finset

/-- Definition of a nice table -/
def is_nice (n k : ℕ) (M : Matrix (Fin k) (Fin n) (Fin 2)) :=
  ∀ (R₁ R₂: Finset (Fin k)), R₁.nonempty → R₂.nonempty → R₁.disjoint R₂ →
  ∃ S : Finset (Fin n), S.nonempty ∧
    (∀ r ∈ R₁, (∑ s in S, M r s) % 2 = 0) ∧
    (∀ r ∈ R₂, (∑ s in S, M r s) % 2 = 1)

/-- The greatest number of k such that there exists at least one nice k × n table is k = n -/
theorem nice_table_greatest_k (n : ℕ) :
  ∃ (M : Matrix (Fin n) (Fin n) (Fin 2)), is_nice n n M :=
sorry

end nice_table_greatest_k_l299_299496


namespace first_term_is_sqrt9_l299_299033

noncomputable def geometric_first_term (a r : ℝ) : ℝ :=
by
  have h1 : a * r^2 = 3 := by sorry
  have h2 : a * r^4 = 27 := by sorry
  have h3 : (a * r^4) / (a * r^2) = 27 / 3 := by sorry
  have h4 : r^2 = 9 := by sorry
  have h5 : r = 3 ∨ r = -3 := by sorry
  have h6 : (a * 9) = 3 := by sorry
  have h7 : a = 1/3 := by sorry
  exact a

theorem first_term_is_sqrt9 : geometric_first_term 3 9 = 3 :=
by
  sorry

end first_term_is_sqrt9_l299_299033


namespace inequality_proof_l299_299369

theorem inequality_proof (x : ℝ) (hx : x ≥ 1) : x^5 - 1 / x^4 ≥ 9 * (x - 1) := 
by sorry

end inequality_proof_l299_299369


namespace total_price_is_correct_l299_299771

def total_price_of_hats (total_hats : ℕ) (blue_hat_cost green_hat_cost : ℕ) (num_green_hats : ℕ) : ℕ :=
  let num_blue_hats := total_hats - num_green_hats
  let cost_green_hats := num_green_hats * green_hat_cost
  let cost_blue_hats := num_blue_hats * blue_hat_cost
  cost_green_hats + cost_blue_hats

theorem total_price_is_correct : total_price_of_hats 85 6 7 40 = 550 := 
  sorry

end total_price_is_correct_l299_299771


namespace boys_from_school_A_study_science_l299_299987

theorem boys_from_school_A_study_science (total_boys school_A_percent non_science_boys school_A_boys study_science_boys: ℕ) 
(h1 : total_boys = 300)
(h2 : school_A_percent = 20)
(h3 : non_science_boys = 42)
(h4 : school_A_boys = (school_A_percent * total_boys) / 100)
(h5 : study_science_boys = school_A_boys - non_science_boys) :
(study_science_boys * 100 / school_A_boys) = 30 :=
by
  sorry

end boys_from_school_A_study_science_l299_299987


namespace hyperbola_asymptote_angle_l299_299181

noncomputable def hyperbola_asymptote_ratio (a b : ℝ) (h : a > b) : ℝ :=
  1 / (-1 + real.sqrt 2)

theorem hyperbola_asymptote_angle (a b : ℝ) (h : a > b)
  (hyperbola_eq : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
  (asymptote_angle_45 : true) :
  a / b = hyperbola_asymptote_ratio a b h :=
sorry

end hyperbola_asymptote_angle_l299_299181


namespace exists_triangle_with_edges_l299_299578

variable {A B C D: Type}
variables (AB AC AD BC BD CD : ℝ)
variables (tetrahedron : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)

def x := AB * CD
def y := AC * BD
def z := AD * BC

theorem exists_triangle_with_edges :
  ∃ (x y z : ℝ), 
  ∃ (A B C D: Type),
  ∃ (AB AC AD BC BD CD : ℝ) (tetrahedron : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D),
  x = AB * CD ∧ y = AC * BD ∧ z = AD * BC → 
  (x + y > z ∧ y + z > x ∧ z + x > y) :=
by
  sorry

end exists_triangle_with_edges_l299_299578


namespace best_ketchup_deal_l299_299861

/-- Given different options of ketchup bottles with their respective prices and volumes as below:
 - Bottle 1: 10 oz for $1
 - Bottle 2: 16 oz for $2
 - Bottle 3: 25 oz for $2.5
 - Bottle 4: 50 oz for $5
 - Bottle 5: 200 oz for $10
And knowing that Billy's mom gives him $10 to spend entirely on ketchup,
prove that the best deal for Billy is to buy one bottle of the $10 ketchup which contains 200 ounces. -/
theorem best_ketchup_deal :
  let price := [1, 2, 2.5, 5, 10]
  let volume := [10, 16, 25, 50, 200]
  let cost_per_ounce := [0.1, 0.125, 0.1, 0.1, 0.05]
  ∃ i, (volume[i] = 200) ∧ (price[i] = 10) ∧ (∀ j, cost_per_ounce[i] ≤ cost_per_ounce[j]) ∧ (price.sum = 10) :=
by
  sorry

end best_ketchup_deal_l299_299861


namespace cost_price_computer_table_l299_299474

theorem cost_price_computer_table (CP SP : ℝ) (h1 : SP = 1.15 * CP) (h2 : SP = 6400) : CP = 5565.22 :=
by sorry

end cost_price_computer_table_l299_299474


namespace profit_24000_max_profit_l299_299085

variable (cost price sales_volume : ℕ)
variable (selling_price profit total_profit : ℕ)
variable (x : ℕ)

-- Conditions
def cost_of_shirt := 50
def sales_volume_eq : ℕ → ℕ := λ x, -20 * x + 2600
def desired_profit := 24000
def max_unit_profit := cost_of_shirt * 3 / 10

-- Proof Statements:
-- (1) Setting selling price to 70 should result in a profit of 24000 yuan
theorem profit_24000 (h : (x - cost_of_shirt) * (sales_volume_eq x) = desired_profit) : 
  x = 70 :=
by sorry

-- (2) Setting selling price to 65 within the profit constraints should maximize the profit, yielding 19500 yuan
theorem max_profit (h1 : 50 ≤ x ∧ x ≤ 65) 
                   (h2 : ∀ x, (x - cost_of_shirt) * (sales_volume_eq x) ≤ 19500) : 
  (x = 65 ∧ total_profit = 19500) :=
by sorry

end profit_24000_max_profit_l299_299085


namespace minimum_value_of_y_l299_299911

theorem minimum_value_of_y :
  ∀ (x : ℝ), x > 3 → let y := x + 1 / (x - 3) in y ≥ 5 :=
by
  sorry

end minimum_value_of_y_l299_299911


namespace probability_X_eq_4_l299_299103

-- Define the number of students and boys
def total_students := 15
def total_boys := 7
def selected_students := 10

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := n.choose k

-- Calculate the probability
def P_X_eq_4 := (binomial_coeff total_boys 4 * binomial_coeff (total_students - total_boys) 6) / binomial_coeff total_students selected_students

-- The statement to be proven
theorem probability_X_eq_4 :
  P_X_eq_4 = (binomial_coeff total_boys 4 * binomial_coeff (total_students - total_boys) 6) / binomial_coeff total_students selected_students := by
  sorry

end probability_X_eq_4_l299_299103


namespace number_of_valid_subsets_l299_299701

theorem number_of_valid_subsets (p : ℕ) (hp : Nat.Prime p) (oddp : p % 2 = 1) :
  let W := Finset.range (2 * p + 1)
  let valid_subsets := {A : Finset ℕ // A ⊆ W ∧ A.card = p ∧ (A.sum id) % p = 0}
  Finset.card valid_subsets = (1 / p * ((Finset.card (Finset.powersetLen p W)) - 2) + 2 : ℚ) := sorry

end number_of_valid_subsets_l299_299701


namespace vertical_asymptotes_sum_l299_299016

theorem vertical_asymptotes_sum : 
  let f := λ x : ℝ, 4 * x^2 + 6 * x + 3 in
  ∃ p q : ℝ, f p = 0 ∧ f q = 0 ∧ p + q = -1.75 :=
by 
  let f := λ x : ℝ, 4 * x^2 + 6 * x + 3
  use [-0.75, -1]
  sorry

end vertical_asymptotes_sum_l299_299016


namespace max_tourism_expenditure_l299_299742

noncomputable def p (x : ℕ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 12 then (1/2) * x * (x + 1) * (39 - 2 * x) else 0

noncomputable def q (x : ℕ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 6 then 35 - 2 * x
  else if 7 ≤ x ∧ x ≤ 12 then 160 / x
  else 0

noncomputable def f (x : ℕ) : ℝ :=
  -3 * x ^ 2 + 40 * x

noncomputable def g (x : ℕ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 6 then (-6 * x ^ 3 + 185 * x ^ 2 - 1400 * x)
  else if 7 ≤ x ∧ x ≤ 12 then -480 * x + 6400
  else 0

theorem max_tourism_expenditure :
  ∃ (x : ℕ), 1 ≤ x ∧ x ≤ 12 ∧ g x = 3125 * 10^4 ∧ x = 5 :=
begin
  sorry
end

end max_tourism_expenditure_l299_299742


namespace width_of_rect_prism_l299_299385

theorem width_of_rect_prism (w : ℝ) 
  (h : ℝ := 8) (l : ℝ := 5) (diagonal : ℝ := 17) 
  (h_diag : l^2 + w^2 + h^2 = diagonal^2) :
  w = 10 * Real.sqrt 2 :=
by
  sorry

end width_of_rect_prism_l299_299385


namespace charge_for_dozens_l299_299391

theorem charge_for_dozens
  (dozen : ℕ)
  (dan_buys : ℕ)
  (gus_buys : ℕ)
  (chris_buys_balls : ℕ)
  (total_balls : ℕ) 
  (balls_per_dozen : ℕ)
  (charges : ℕ) 
  (dan_buys = 5)
  (gus_buys = 2)
  (chris_buys_balls = 48)
  (total_balls = 132)
  (balls_per_dozen = 12)
  (charges = 30)
  : (dan_buys + gus_buys + chris_buys_balls / balls_per_dozen = total_balls / balls_per_dozen) → dozen = 11 :=
sorry

end charge_for_dozens_l299_299391


namespace solution_set_l299_299555

def inequality (x : ℝ) : Prop :=
  (x / (x - 1) + (x + 2) / (2 * x) ≥ 3)

theorem solution_set :
  ∀ (x : ℝ), x ≠ 0 ∧ x ≠ 1 → (inequality x ↔ x ∈ Set.Ioo 0 (1/3) ∪ Set.Ioo 1 2 ∪ Set.Ioo (1/3) (1/3) ∪ Set.Ioo 2 2) :=
begin
  sorry
end

end solution_set_l299_299555


namespace problem_a_l299_299802

theorem problem_a : (1038^2 % 1000) ≠ 4 := by
  sorry

end problem_a_l299_299802


namespace problem1_problem2_l299_299230

section
variable {x a : ℝ}

-- Definitions of the functions
def f (x : ℝ) : ℝ := |x + 1|
def g (x : ℝ) (a : ℝ) : ℝ := 2 * |x| + a

-- Problem 1
theorem problem1 (a : ℝ) (H : a = -1) : 
  ∀ x : ℝ, f x ≤ g x a ↔ (x ≤ -2/3 ∨ 2 ≤ x) :=
sorry

-- Problem 2
theorem problem2 (a : ℝ) : 
  (∃ x₀ : ℝ, f x₀ ≥ 1/2 * g x₀ a) → a ≤ 2 :=
sorry

end

end problem1_problem2_l299_299230


namespace initial_people_in_gym_l299_299858

variable (W A : ℕ)

theorem initial_people_in_gym (W A : ℕ) (h : W + A + 5 + 2 - 3 - 4 + 2 = 20) : W + A = 18 := by
  sorry

end initial_people_in_gym_l299_299858


namespace crayon_ratio_proof_l299_299342

-- Definitions
def billie_crayons : ℕ := 18
def bobbie_crayons : ℕ := 3 * billie_crayons
def lizzie_crayons : ℕ := 27

-- Theorem statement
theorem crayon_ratio_proof : lizzie_crayons : bobbie_crayons = 1 : 2 :=
by
  sorry

end crayon_ratio_proof_l299_299342


namespace transform_graph_to_left_by_pi_div_10_l299_299434

def graph_transformation (x : ℝ) : Prop :=
  ∀ y, y = 3 * sin (2 * x + (π / 5)) ↔ y = 3 * sin (2 * (x + (π / 10)))

theorem transform_graph_to_left_by_pi_div_10 :
  graph_transformation x :=
by
  sorry

end transform_graph_to_left_by_pi_div_10_l299_299434


namespace halloween_candy_l299_299565

theorem halloween_candy (katie_candy : ℕ) (sister_candy : ℕ) (remaining_candy : ℕ) (total_candy : ℕ) (eaten_candy : ℕ)
  (h1 : katie_candy = 10) 
  (h2 : sister_candy = 6) 
  (h3 : remaining_candy = 7) 
  (h4 : total_candy = katie_candy + sister_candy) 
  (h5 : eaten_candy = total_candy - remaining_candy) : 
  eaten_candy = 9 :=
by sorry

end halloween_candy_l299_299565


namespace favorite_numbers_l299_299712

-- Define what it means to be a favorite number
def is_favorite (n : ℕ) : Prop :=
  n * (n.digits 10).sum = 10 * n

-- Conditions given in the problem
def condition (a b c : ℕ) : Prop :=
  is_favorite a ∧ is_favorite b ∧ is_favorite c ∧ a * b * c = 71668

-- Problem statement
theorem favorite_numbers (a b c : ℕ) (h : condition a b c) :
  {a, b, c} = {19, 46, 82} :=
by 
  sorry -- proof to be completed

end favorite_numbers_l299_299712


namespace solve_quadratic_equation_l299_299417

theorem solve_quadratic_equation (x : ℝ) : x^2 - 2 * x = 0 ↔ x = 0 ∨ x = 2 :=
by
  sorry

end solve_quadratic_equation_l299_299417


namespace present_age_of_father_l299_299796

/-- The present age of the father is 3 years more than 3 times the age of his son, 
    and 3 years hence, the father's age will be 8 years more than twice the age of the son. 
    Prove that the present age of the father is 27 years. -/
theorem present_age_of_father (F S : ℕ) (h1 : F = 3 * S + 3) (h2 : F + 3 = 2 * (S + 3) + 8) : F = 27 :=
by
  sorry

end present_age_of_father_l299_299796


namespace largest_A_proof_smallest_A_proof_l299_299847

def is_coprime_with_12 (n : ℕ) : Prop := Nat.gcd n 12 = 1

def obtain_A_from_B (B : ℕ) : ℕ :=
  let b := B % 10
  let k := B / 10
  b * 10^7 + k

constant B : ℕ → Prop
constant A : ℕ → ℕ → Prop

noncomputable def largest_A : ℕ :=
  99999998

noncomputable def smallest_A : ℕ :=
  14444446

theorem largest_A_proof (B : ℕ) (h1 : B > 44444444) (h2 : is_coprime_with_12 B) :
  obtain_A_from_B B = largest_A :=
sorry

theorem smallest_A_proof (B : ℕ) (h1 : B > 44444444) (h2 : is_coprime_with_12 B) :
  obtain_A_from_B B = smallest_A :=
sorry

end largest_A_proof_smallest_A_proof_l299_299847


namespace negation_of_universal_statement_l299_299752

theorem negation_of_universal_statement :
  ¬ (∀ x : ℝ, x^2 ≤ 1) ↔ ∃ x : ℝ, x^2 > 1 :=
by
  sorry

end negation_of_universal_statement_l299_299752


namespace inclination_angle_and_slope_of_line_y_eq_2_l299_299049

theorem inclination_angle_and_slope_of_line_y_eq_2 :
  ∀ (line : ℕ → ℕ), (∀ x, line x = 2) → (inclination_angle line = 0 ∧ slope line = 0) :=
by
  intro line h
  sorry

end inclination_angle_and_slope_of_line_y_eq_2_l299_299049


namespace value_of_expression_l299_299075

theorem value_of_expression (n : ℕ) (a : ℝ) (h1 : 6 * 11 * n ≠ 0) (h2 : a ^ (2 * n) = 5) : 2 * a ^ (6 * n) - 4 = 246 :=
by
  sorry

end value_of_expression_l299_299075


namespace ken_kept_pencils_l299_299685

def ken_total_pencils := 50
def pencils_given_to_manny := 10
def pencils_given_to_nilo := pencils_given_to_manny + 10
def pencils_given_away := pencils_given_to_manny + pencils_given_to_nilo

theorem ken_kept_pencils : ken_total_pencils - pencils_given_away = 20 := by
  sorry

end ken_kept_pencils_l299_299685


namespace simplify_expression_l299_299375

theorem simplify_expression :
  (↑(Real.sqrt 648) / ↑(Real.sqrt 81) - ↑(Real.sqrt 245) / ↑(Real.sqrt 49)) = 2 * Real.sqrt 2 - Real.sqrt 5 := by
  -- proof omitted
  sorry

end simplify_expression_l299_299375


namespace tennis_tournament_rooms_l299_299277

noncomputable def min_rooms_needed (n : Nat) : Nat :=
  n + 1

theorem tennis_tournament_rooms (n : Nat) :
  ∀ (players_coaches_count : Nat) 
  (coaches_rooms players_rooms : Nat → Nat) 
  (friend_agreement enemy_residence : Nat → Nat → Bool),
  players_coaches_count = n ∧
  (∀ i, i < players_coaches_count → (friend_agreement i i = true) ∧ (enemy_residence i i = false)) →
  coaches_rooms players_coaches_count = players_rooms players_coaches_count →
  min_rooms_needed n = n + 1 :=
begin
  intros,
  sorry
end

end tennis_tournament_rooms_l299_299277


namespace minimum_value_expression_l299_299692

theorem minimum_value_expression (a b c: ℝ) (ha: a > 0) (hb: b > 0) (hc: c > 0) :
  min (sqrt ((a^2) / (b^2)) + sqrt ((b^2) / (c^2)) + sqrt ((c^2) / (a^2))) = 3 :=
by sorry

end minimum_value_expression_l299_299692


namespace cost_of_trip_l299_299768

variables (vaccines_count vaccines_cost_per_unit doctor_visit_cost total_payment insurance_rate : ℝ)
variables (trip_cost total_medical_cost insurance_coverage medical_cost_after_insurance : ℝ)

-- Given facts and conditions
def vaccines_cost := vaccines_count * vaccines_cost_per_unit
def total_medical_cost := vaccines_cost + doctor_visit_cost
def insurance_coverage := insurance_rate * total_medical_cost
def medical_cost_after_insurance := total_medical_cost - insurance_coverage
def trip_cost := total_payment - medical_cost_after_insurance

-- Prove that the cost of the trip is $1200
theorem cost_of_trip : 
  (vaccines_count = 10) →
  (vaccines_cost_per_unit = 45) →
  (doctor_visit_cost = 250) →
  (total_payment = 1340) →
  (insurance_rate = 0.80) →
  trip_cost = 1200 :=
begin
  sorry,
end

end cost_of_trip_l299_299768


namespace relationship_between_y_l299_299929

variable (f : ℝ → ℝ)
variable (A B C : ℝ × ℝ)

def is_on_graph (pt : ℝ × ℝ) (f : ℝ → ℝ) : Prop := pt.snd = f pt.fst

def quadratic_function (x : ℝ) : ℝ := (x - 2)^2 - 1

variables (y1 y2 y3 : ℝ)
variables hA : A = (4, y1)
variables hB : B = (Real.sqrt 2, y2)
variables hC : C = (-2, y3)

theorem relationship_between_y (h1 : is_on_graph A quadratic_function)
                                (h2 : is_on_graph B quadratic_function)
                                (h3 : is_on_graph C quadratic_function) :
  y3 > y1 ∧ y1 > y2 :=
by
  sorry

end relationship_between_y_l299_299929


namespace total_male_students_combined_l299_299019

/-- The number of first-year students is 695, of which 329 are female students. 
If the number of male second-year students is 254, prove that the number of male students in the first-year and second-year combined is 620. -/
theorem total_male_students_combined (first_year_students : ℕ) (female_first_year_students : ℕ) (male_second_year_students : ℕ) :
  first_year_students = 695 →
  female_first_year_students = 329 →
  male_second_year_students = 254 →
  (first_year_students - female_first_year_students + male_second_year_students) = 620 := by
  sorry

end total_male_students_combined_l299_299019


namespace favorite_numbers_product_71668_l299_299710

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the favorite number property
def is_favorite (n : ℕ) : Prop :=
  sum_of_digits n = 10

-- Define the given problem as a Lean theorem
theorem favorite_numbers_product_71668 (a b c : ℕ ) 
(hfa : is_favorite a) (hfb : is_favorite b) (hfc : is_favorite c) 
(hprod : a * b * c = 71668) : 
  {a, b, c} = {19, 46, 82} :=
by
  sorry

end favorite_numbers_product_71668_l299_299710


namespace mean_of_remaining_two_l299_299432

theorem mean_of_remaining_two (a b c d e : ℝ) (h : (a + b + c = 3 * 2010)) : 
  (a + b + c + d + e) / 5 = 2010 → (d + e) / 2 = 2011.5 :=
by
  sorry 

end mean_of_remaining_two_l299_299432


namespace largest_binomial_term_l299_299530

noncomputable def binomial_coefficient (n k : ℕ) : ℚ :=
if k > n then 0 else nat.factorial n / (nat.factorial k * nat.factorial (n - k))

noncomputable def A (k : ℕ) : ℚ :=
binomial_coefficient 500 k * (0.3 : ℚ) ^ k

theorem largest_binomial_term :
  (∀ k : ℕ, 0 ≤ k ∧ k ≤ 500 → A k ≤ A 125) ∧ (0 ≤ 125 ∧ 125 ≤ 500) :=
by
  sorry

end largest_binomial_term_l299_299530


namespace split_eight_nums_into_three_subsets_cannot_split_ten_nums_into_two_subsets_smallest_N_for_n_subsets_l299_299396

-- (a) Separation Problem
theorem split_eight_nums_into_three_subsets :
  ∃ A B C : Finset ℕ, (A ∪ B ∪ C = {1,2,3,4,5,6,7,8}) ∧ 
                   (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧ 
                   (A.sum = 12) ∧ (B.sum = 12) ∧ (C.sum = 12) :=
sorry

-- (b) Impossibility Problem
theorem cannot_split_ten_nums_into_two_subsets :
  ¬ ∃ A B : Finset ℕ, (A ∪ B = {1,2,3,4,5,6,7,8,9,10}) ∧ 
                     (A ∩ B = ∅) ∧
                     (A.sum = 27.5) ∧ (B.sum = 27.5) :=
sorry

-- (c) Minimum N Problem
theorem smallest_N_for_n_subsets (n : ℕ) (hn : n ≥ 2) :
  ∃ N : ℕ, (N = 2 * n - 1) ∧ (∃ (subsets : Finset (Finset ℕ)), 
    (∑ x in subsets, x.sum = (N * (N + 1)) / 2) ∧
    (∀ x ∈ subsets, x.sum = (N * (N + 1)) / (2 * n))) :=
sorry

end split_eight_nums_into_three_subsets_cannot_split_ten_nums_into_two_subsets_smallest_N_for_n_subsets_l299_299396


namespace Wednesday_earnings_l299_299082

variable (S W : ℝ)

def total_earnings := 4994.50
def difference := 1330.50

theorem Wednesday_earnings :
  W + S = total_earnings ∧ W = S - difference → W = 1832 :=
by
  sorry

end Wednesday_earnings_l299_299082


namespace distance_midpoint_AB_to_line_l299_299209

noncomputable def tan_alpha_eq_neg2 (α : ℝ) : Prop := Real.tan α = -2

noncomputable def focus_F (α : ℝ) : (ℝ × ℝ) := (-Real.sin α * Real.cos α, 0)

noncomputable def line_l_intersects_AB (x1 x2 : ℝ) : Prop :=
  x1 + x2 + (4 / 5) = 4 ∧ (x2 - x1).abs = 4

theorem distance_midpoint_AB_to_line (α x1 x2 : ℝ) :
  tan_alpha_eq_neg2 α →
  let F := focus_F α in
  line_l_intersects_AB x1 x2 →
  let midpoint_x := (x1 + x2) / 2 in
  (midpoint_x - (-1 / 2)).abs = 21 / 10 :=
by
  sorry

end distance_midpoint_AB_to_line_l299_299209


namespace total_profit_l299_299792

-- Definitions
def investment_a : ℝ := 45000
def investment_b : ℝ := 63000
def investment_c : ℝ := 72000
def c_share : ℝ := 24000

-- Theorem statement
theorem total_profit : (investment_a + investment_b + investment_c) * (c_share / investment_c) = 60000 := by
  sorry

end total_profit_l299_299792


namespace right_triangle_YZ_length_l299_299992

theorem right_triangle_YZ_length (X Y Z : Type) [metric_space X] 
  (XY XZ YZ : ℝ) (h : XZ = 10) (cos_X : real.cos X.to_real = 3/5) 
  (right_angle_at_Y : angle_is π/2 Y) 
  (XYZ : triangle_point_property XY XZ YZ): 
  YZ = 8 :=
by
  sorry

end right_triangle_YZ_length_l299_299992


namespace sum_of_W_and_Y_equals_eight_l299_299153

theorem sum_of_W_and_Y_equals_eight :
  ∃ (W X Y Z : ℕ), {W, X, Y, Z} = {2, 3, 5, 6} ∧ W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z ∧
  (W * X) / (Y * Z) + (Y / Z) = 3 ∧ W + Y = 8 :=
sorry

end sum_of_W_and_Y_equals_eight_l299_299153


namespace division_correct_l299_299139

theorem division_correct : 0.45 / 0.005 = 90 := by
  sorry

end division_correct_l299_299139


namespace analogy_uses_analogical_reasoning_l299_299452

-- Define the conditions for tangency and perpendicularity in 2D and 3D cases
axiom tangent_perpendicular_2D (C : Type) (L : Type) (T : C → L → Prop) (P : C → L → Prop) : 
  ∀ (c : C) (l : L) (t : T c l) (p : P c l), ⊥

axiom tangent_perpendicular_3D (S : Type) (P : Type) (T : S → P → Prop) (P : S → P → Prop) : 
  ∀ (s : S) (p : P) (t : T s p) (p : P s p), ⊥

-- Define the analogy reasoning type
inductive Reasoning 
| Analogical : Reasoning

-- State the proof goal
theorem analogy_uses_analogical_reasoning 
  (C : Type) (L : Type) (T_2D : C → L → Prop) (P_2D : C → L → Prop) 
  (S : Type) (P : Type) (T_3D : S → P → Prop) (P_3D : S → P → Prop) :
  tangent_perpendicular_2D C L T_2D P_2D → tangent_perpendicular_3D S P T_3D P_3D → Reasoning.Analogical :=
by 
  sorry

end analogy_uses_analogical_reasoning_l299_299452


namespace comic_book_issue_pages_l299_299769

theorem comic_book_issue_pages (total_pages: ℕ) 
  (speed_month1 speed_month2 speed_month3: ℕ) 
  (bonus_pages: ℕ) (issue1_2_pages: ℕ) 
  (issue3_pages: ℕ)
  (h1: total_pages = 220)
  (h2: speed_month1 = 5)
  (h3: speed_month2 = 4)
  (h4: speed_month3 = 4)
  (h5: issue3_pages = issue1_2_pages + 4)
  (h6: bonus_pages = 3)
  (h7: (issue1_2_pages + bonus_pages) + 
       (issue1_2_pages + bonus_pages) + 
       (issue3_pages + bonus_pages) = total_pages) : 
  issue1_2_pages = 69 := 
by 
  sorry

end comic_book_issue_pages_l299_299769


namespace angle_B_cos_A_l299_299302

theorem angle_B (a b c : ℝ) (C : ℝ) (h : b * cos C + b * sin C = a) : (B : ℝ) := sorry

theorem cos_A (a b c : ℝ) (C B : ℝ) (h1 : b * cos C + b * sin C = a) (h2 : B = π / 4) (h3 : BC = 1 / 4 * a) : cos A = - (√5) / 5 := sorry

end angle_B_cos_A_l299_299302


namespace power_function_properties_l299_299941

theorem power_function_properties (α : ℝ) (f : ℝ → ℝ)
  (h₁ : f(x) = x^α)
  (h₂ : f (1/8) = sqrt 2 / 4)
  (x₁ x₂ : ℝ) (hx : x₁ < x₂) :
  α = 1/2 → (x₁ * f x₁ < x₂ * f x₂) ∧ (f x₁ / x₁ > f x₂ / x₂) :=
by
  sorry

end power_function_properties_l299_299941


namespace number_of_solutions_l299_299020

theorem number_of_solutions (x y : ℤ) : (2 ^ (2 * x) - 3 ^ (2 * y) = 55) → (∃! (x y : ℤ), 2 ^ (2 * x) - 3 ^ (2 * y) = 55) :=
by
  sorry

end number_of_solutions_l299_299020


namespace line_b_perpendicular_to_c_is_necessary_and_sufficient_condition_l299_299936

-- Definitions of planes, lines, perpendicularity, and intersection points
variable {α β : Plane} {a b c : Line} {P : Point}

-- Hypotheses from the conditions
axiom h1 : α ⟂ β
axiom h2 : α ∩ β = c
axiom h3 : a ∈ α
axiom h4 : b ∈ β
axiom h5 : ¬(a ⟂ c)
axiom h6 : a ∩ b = P ∧ a ∩ c = P ∧ b ∩ c = P

-- The main theorem stating the necessary and sufficient condition
theorem line_b_perpendicular_to_c_is_necessary_and_sufficient_condition :
  (b ⟂ c) ↔ (b ⟂ a) := by
  sorry

end line_b_perpendicular_to_c_is_necessary_and_sufficient_condition_l299_299936


namespace ratio_boysGradeA_girlsGradeB_l299_299649

variable (S G B : ℕ)

-- Given conditions
axiom h1 : (1 / 3 : ℚ) * G = (1 / 4 : ℚ) * S
axiom h2 : S = B + G

-- Definitions based on conditions
def boys_in_GradeA (B : ℕ) := (2 / 5 : ℚ) * B
def girls_in_GradeB (G : ℕ) := (3 / 5 : ℚ) * G

-- The proof goal
theorem ratio_boysGradeA_girlsGradeB (S G B : ℕ) (h1 : (1 / 3 : ℚ) * G = (1 / 4 : ℚ) * S) (h2 : S = B + G) :
    boys_in_GradeA B / girls_in_GradeB G = 2 / 9 :=
by
  sorry

end ratio_boysGradeA_girlsGradeB_l299_299649


namespace six_digit_numbers_with_and_without_one_difference_l299_299510

theorem six_digit_numbers_with_and_without_one_difference :
  let total_numbers := Nat.choose 9 6 in
  let numbers_with_one := Nat.choose 8 5 in
  let numbers_without_one := total_numbers - numbers_with_one in
  numbers_with_one - numbers_without_one = 28 :=
by
  let total_numbers := Nat.choose 9 6
  let numbers_with_one := Nat.choose 8 5
  let numbers_without_one := total_numbers - numbers_with_one
  exact (numbers_with_one - numbers_without_one)
  sorry

end six_digit_numbers_with_and_without_one_difference_l299_299510


namespace palindrome_divisible_by_11_probability_zero_l299_299494

theorem palindrome_divisible_by_11_probability_zero :
  let palindromes := { n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 101 * a + 10 * b }
  let count_palindromes := card palindromes
  ∀ n ∈ palindromes, n % 11 ≠ 0 :=
by
  let palindromes := { n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 101 * a + 10 * b }
  let count_palindromes := cardinality palindromes
  suffices ∀ n ∈ palindromes, n % 11 ≠ 0 from
    (card (set.filter (λ n, n % 11 = 0) palindromes)) = 0
  sorry

end palindrome_divisible_by_11_probability_zero_l299_299494


namespace count_ordered_pairs_l299_299876

theorem count_ordered_pairs (x y : ℕ) (h₁ : 1 ≤ x) (h₂ : x < y) (h₃ : y ≤ 150) 
  (h₄ : ((x % 4 = y % 4) ∨ (x % 4 + y % 4 = 4)))
  (h₅ : (x * y) % 3 = 0) : 
  x * y ≠ 0 :=
begin
  sorry
end

noncomputable def approximate_number_of_pairs :=
  1225

end count_ordered_pairs_l299_299876


namespace isosceles_trapezoid_sides_l299_299656

theorem isosceles_trapezoid_sides (R : ℝ) (trapezoid : IsoscelesTrapezoid) 
  (circle_1_circle_2_tangent_circles : CirclesOfRadiusTangentInIsoscelesTrapezoid R trapezoid)
  (circle_centers_on_diagonals : CentersOnDiagonals trapezoid) :
  sides trapezoid = (2 * R * Real.sqrt 2, 2 * R * Real.sqrt 2, 2 * R * Real.sqrt 2, 2 * R * (2 + Real.sqrt 2)) :=
sorry

end isosceles_trapezoid_sides_l299_299656


namespace new_remainder_when_scaled_l299_299076

theorem new_remainder_when_scaled (a b c : ℕ) (h : a = b * c + 7) : (10 * a) % (10 * b) = 70 := by
  sorry

end new_remainder_when_scaled_l299_299076


namespace necessary_but_not_sufficient_condition_l299_299582
open Locale

variables {l m : Line} {α β : Plane}

def perp (l : Line) (p : Plane) : Prop := sorry
def subset (l : Line) (p : Plane) : Prop := sorry
def parallel (p₁ p₂ : Plane) : Prop := sorry

theorem necessary_but_not_sufficient_condition (h1 : perp l α) (h2 : subset m β) (h3 : perp l m) :
  ∃ (α : Plane) (β : Plane), parallel α β ∧ (perp l α → perp l β) ∧ (parallel α β → perp l β)  :=
sorry

end necessary_but_not_sufficient_condition_l299_299582


namespace new_solution_percentage_l299_299726

theorem new_solution_percentage 
  (initial_weight : ℝ) (evaporated_water : ℝ) (added_solution_weight : ℝ) 
  (percentage_X : ℝ) (percentage_water : ℝ)
  (total_initial_X : ℝ := initial_weight * percentage_X)
  (initial_water : ℝ := initial_weight * percentage_water)
  (post_evaporation_weight : ℝ := initial_weight - evaporated_water)
  (post_evaporation_X : ℝ := total_initial_X)
  (post_evaporation_water : ℝ := post_evaporation_weight - total_initial_X)
  (added_X : ℝ := added_solution_weight * percentage_X)
  (added_water : ℝ := added_solution_weight * percentage_water)
  (total_X : ℝ := post_evaporation_X + added_X)
  (total_water : ℝ := post_evaporation_water + added_water)
  (new_total_weight : ℝ := post_evaporation_weight + added_solution_weight) :
  (total_X / new_total_weight) * 100 = 41.25 := 
by {
  sorry
}

end new_solution_percentage_l299_299726


namespace total_students_appeared_l299_299995

theorem total_students_appeared (T : ℕ) 
    (h1 : 0.29 * T + 0.54 * T + 0.17 * T = T)
    (h2 : 0.17 * T = 51) : 
    T = 300 := 
sorry

end total_students_appeared_l299_299995


namespace find_first_term_geom_seq_l299_299747

noncomputable def first_term (a r : ℝ) := a

theorem find_first_term_geom_seq 
  (a r : ℝ) 
  (h1 : a * r ^ 3 = 720) 
  (h2 : a * r ^ 6 = 5040) : 
  first_term a r = 720 / 7 := 
sorry

end find_first_term_geom_seq_l299_299747


namespace fractions_order_l299_299454

theorem fractions_order:
  (20 / 15) < (25 / 18) ∧ (25 / 18) < (23 / 16) ∧ (23 / 16) < (21 / 14) :=
by
  sorry

end fractions_order_l299_299454


namespace best_ketchup_deal_l299_299860

/-- Given different options of ketchup bottles with their respective prices and volumes as below:
 - Bottle 1: 10 oz for $1
 - Bottle 2: 16 oz for $2
 - Bottle 3: 25 oz for $2.5
 - Bottle 4: 50 oz for $5
 - Bottle 5: 200 oz for $10
And knowing that Billy's mom gives him $10 to spend entirely on ketchup,
prove that the best deal for Billy is to buy one bottle of the $10 ketchup which contains 200 ounces. -/
theorem best_ketchup_deal :
  let price := [1, 2, 2.5, 5, 10]
  let volume := [10, 16, 25, 50, 200]
  let cost_per_ounce := [0.1, 0.125, 0.1, 0.1, 0.05]
  ∃ i, (volume[i] = 200) ∧ (price[i] = 10) ∧ (∀ j, cost_per_ounce[i] ≤ cost_per_ounce[j]) ∧ (price.sum = 10) :=
by
  sorry

end best_ketchup_deal_l299_299860


namespace repeating_decimal_fraction_equiv_l299_299050

noncomputable def repeating_decimal_to_fraction (x : ℚ) : Prop :=
  x = 0.4 + 37 / 990

theorem repeating_decimal_fraction_equiv : repeating_decimal_to_fraction (433 / 990) :=
by
  sorry

end repeating_decimal_fraction_equiv_l299_299050


namespace log_10_43_between_integers_l299_299036

theorem log_10_43_between_integers :
  ∃ a b : ℤ, (a = 1 ∧ b = 2 ∧ (a + b = 3)) ∧ 1 < real.log 43 / real.log 10 ∧ real.log 43 / real.log 10 < 2 :=
by
  sorry

end log_10_43_between_integers_l299_299036


namespace unimodular_sum_l299_299688

theorem unimodular_sum (n : ℕ) (h : 0 < n) (x : Fin n → ℝ) :
    ∑ (a : Fin n → ℝ) in (Finset.univ.filter (λ a, (∀ i, a i = 1 ∨ a i = -1))),
        (∏ i, a i) * (∑ i, a i * x i) ^ n = 2 ^ n * Nat.factorial n * ∏ i, x i :=
sorry

end unimodular_sum_l299_299688


namespace distribution_less_than_m_plus_g_l299_299989

theorem distribution_less_than_m_plus_g (m g : ℝ) (P : ℝ → ℝ → Prop) 
  (H1 : ∀ x, P x m → P x (g - m))
  (H2 : ∀ x, P x m → P x (m - g))
  (H3 : 0.68 = ∫ x in (m - g), (m + g), (distr x m g) dx) :
  ∫ x in (-(∞:ℝ)), (m + g), (distr x m g) dx = 0.84 :=
by
  sorry

end distribution_less_than_m_plus_g_l299_299989


namespace train_speed_160m_6sec_l299_299081

noncomputable def train_speed (distance time : ℕ) : ℚ :=
(distance : ℚ) / (time : ℚ)

theorem train_speed_160m_6sec : train_speed 160 6 = 26.67 :=
by
  simp [train_speed]
  norm_num
  sorry

end train_speed_160m_6sec_l299_299081


namespace solve_inequality_smallest_integer_solution_l299_299729

theorem solve_inequality (x : ℝ) : 
    (9 * x + 8) / 6 - x / 3 ≥ -1 ↔ x ≥ -2 := 
sorry

theorem smallest_integer_solution :
    ∃ (x : ℤ), (∃ (y : ℝ) (h₁ : y = x), 
    (9 * y + 8) / 6 - y / 3 ≥ -1) ∧ 
    ∀ (z : ℤ), ((∃ (w : ℝ) (h₂ : w = z), 
    (9 * w + 8) / 6 - w / 3 ≥ -1) → -2 ≤ z) :=
    ⟨-2, __, sorry⟩

end solve_inequality_smallest_integer_solution_l299_299729


namespace chord_length_equality_l299_299316

open EuclideanGeometry -- assuming this module exists to ease the necessary definitions

theorem chord_length_equality
  (Γ : Circle) 
  (B C : Γ.Point)
  (A : Point)
  (hA : A ∈ Circle.interior Γ)
  (hBAC : AcuteAngle B A C)
  (P : Point)
  (hACP : Isosceles (Triangle.mk A C P) ∧ RightAngle (Angle.mk A C P))
  (R : Point)
  (hABR : Isosceles (Triangle.mk A B R) ∧ RightAngle (Angle.mk A B R))
  (E F : Point)
  (hBAE : B, A, E ∈ Γ ∧ Line.mk B A ≈ Line.mk E A)
  (hCAF : C, A, F ∈ Γ ∧ Line.mk C A ≈ Line.mk F A)
  (X Y : Point)
  (hEPX : Line.mk E P ≈ Line.mk X P ∧ X ∈ Γ)
  (hFRY : Line.mk F R ≈ Line.mk Y R ∧ Y ∈ Γ) : 
  Length.mk B X = Length.mk C Y :=
sorry

end chord_length_equality_l299_299316


namespace sin_3x_over_4_period_l299_299778

noncomputable def sine_period (b : ℝ) : ℝ :=
  (2 * Real.pi) / b

theorem sin_3x_over_4_period :
  sine_period (3/4) = (8 * Real.pi) / 3 :=
by
  sorry

end sin_3x_over_4_period_l299_299778


namespace largest_and_smallest_A_l299_299840

noncomputable def is_coprime_with_12 (n : ℕ) : Prop := 
  Nat.gcd n 12 = 1

def problem_statement (A_max A_min : ℕ) : Prop :=
  ∃ B : ℕ, B > 44444444 ∧ is_coprime_with_12 B ∧
  (A_max = 9 * 10^7 + (B - 9) / 10) ∧
  (A_min = 1 * 10^7 + (B - 1) / 10)

theorem largest_and_smallest_A :
  problem_statement 99999998 14444446 := sorry

end largest_and_smallest_A_l299_299840


namespace rectangle_perimeter_l299_299018

-- Defining the conditions: the length is thrice the breadth, and the area is 432.
variable (breadth length perimeter : ℝ)
variable (h1 : length = 3 * breadth)
variable (h2 : breadth * length = 432)

-- We need to prove that the perimeter is 96.
theorem rectangle_perimeter (h1 : length = 3 * breadth) (h2 : breadth * length = 432) : 2 * length + 2 * breadth = 96 :=
begin
  sorry
end

end rectangle_perimeter_l299_299018


namespace fraction_of_jumbo_tiles_l299_299102

-- Definitions for conditions
variables (L W : ℝ) -- Length and width of regular tiles
variables (n : ℕ) -- Number of regular tiles
variables (m : ℕ) -- Number of jumbo tiles

-- Conditions
def condition1 : Prop := (n : ℝ) * (L * W) = 40 -- Regular tiles cover 40 square feet
def condition2 : Prop := (n : ℝ) * (L * W) + (m : ℝ) * (3 * L * W) = 220 -- Entire wall is 220 square feet
def condition3 : Prop := ∃ (k : ℝ), (m : ℝ) = k * (n : ℝ) ∧ k = 1.5 -- Relationship ratio between jumbo and regular tiles

-- Theorem to be proved
theorem fraction_of_jumbo_tiles (L W : ℝ) (n m : ℕ)
  (h1 : condition1 L W n)
  (h2 : condition2 L W n m)
  (h3 : condition3 n m) :
  (m : ℝ) / ((n : ℝ) + (m : ℝ)) = 3 / 5 :=
sorry

end fraction_of_jumbo_tiles_l299_299102


namespace count_difference_l299_299511

-- Given definitions
def count_six_digit_numbers_in_ascending_order_by_digits : ℕ := by
  -- Calculation using binomial coefficient
  exact Nat.choose 9 6

def count_six_digit_numbers_with_one : ℕ := by
  -- Calculation using binomial coefficient with fixed '1' in one position
  exact Nat.choose 8 5

def count_six_digit_numbers_without_one : ℕ := by
  -- Calculation subtracting with and without 1
  exact count_six_digit_numbers_in_ascending_order_by_digits - count_six_digit_numbers_with_one

-- Theorem to prove
theorem count_difference : 
  count_six_digit_numbers_with_one - count_six_digit_numbers_without_one = 28 :=
by
  sorry

end count_difference_l299_299511


namespace polygon_area_bound_l299_299111

theorem polygon_area_bound (n : ℕ) (h : n ≥ 3) {vertices : list (ℤ × ℤ)} 
  (hv : vertices.length = n) (integer_coords : ∀ v ∈ vertices, ∃ (x y : ℤ), v = (x, y))
  (convex : is_convex_polygon vertices) :
  ∃ A : ℚ, A ≥ (n : ℚ - 2) / 2 := 
sorry

end polygon_area_bound_l299_299111


namespace geometric_sequence_first_term_l299_299030

theorem geometric_sequence_first_term (a r : ℝ) 
  (h1 : a * r^2 = 3)
  (h2 : a * r^4 = 27) : 
  a = - (real.sqrt 9) / 9 :=
by
  sorry

end geometric_sequence_first_term_l299_299030


namespace arithmetic_and_geometric_sequences_l299_299999

noncomputable def a_n (n : ℕ) : ℕ :=
  3 * n

noncomputable def b_n (n : ℕ) : ℕ :=
  3 ^ (n - 1)

noncomputable def S_n (n : ℕ) : ℕ :=
  (n * (3 + 3 * n)) / 2

theorem arithmetic_and_geometric_sequences 
  (n : ℕ)
  (hn : n ≥ 1)
  (a1_eq : a_n 1 = 3)
  (b1_eq : b_n 1 = 1)
  (b2_plus_S2 : b_n 2 + S_n 2 = 12)
  (q_eq : (S_n 2) / (b_n 2) = 3) :
  (a_n n = 3 * n) ∧ (b_n n = 3 ^ (n - 1)) ∧
  (1 / 3 ≤ 1 / S_n 1 + 1 / S_n 2 + ... + 1 / S_n n < 2 / 3) :=
by
  sorry

end arithmetic_and_geometric_sequences_l299_299999


namespace angle_DE_CD_30_l299_299287

-- Define the conditions
def AB : Segment := Segment.mk A B
def C_on_AB : Point → Line → Prop := λ C AB, C ∈ AB
def vertical : Line → Line → Prop := λ CD AB, ∀ (x : Point), x ∈ CD → (x = C ∨ x ≠ C)
def angle : Line → Line → ℝ → Prop := λ l1 l2 α, ∃ β γ, β + γ = 180 ∧ α = β

-- Given conditions
variables {CD DE : Line} {C D E : Point} {x : ℝ}
variable (h1 : vertical CD AB)
variable (h2 : C_on_AB C AB)
variable (h3 : angle DE AB 60)
variable (h4 : angle CD DE x)

-- The required proof statement
theorem angle_DE_CD_30 : x = 30 :=
sorry

end angle_DE_CD_30_l299_299287


namespace basic_astrophysics_budget_percent_l299_299087

theorem basic_astrophysics_budget_percent
  (total_degrees : ℝ := 360)
  (astrophysics_degrees : ℝ := 108) :
  (astrophysics_degrees / total_degrees) * 100 = 30 := by
  sorry

end basic_astrophysics_budget_percent_l299_299087


namespace find_perimeter_of_triangle_parallel_lines_l299_299770

def triangle_perimeter (PQ QR PR l₁ l₂ l₃ : ℝ) : ℝ :=
  let s₁ := (PQ / PR) * (l₂ + QR)
  let s₂ := (PR / QR) * (l₁ + QR)
  let s₃ := (QR / PQ) * (l₃ + PQ)
  s₁ + s₂ + s₃

theorem find_perimeter_of_triangle_parallel_lines (PQ QR PR l₁ l₂ l₃ : ℝ) :
  PQ = 150 → QR = 250 → PR = 200 → l₁ = 65 → l₂ = 35 → l₃ = 25 →
  triangle_perimeter PQ QR PR l₁ l₂ l₃ = 555.42 :=
by
  intros
  sorry

end find_perimeter_of_triangle_parallel_lines_l299_299770


namespace solve_equation_l299_299026

theorem solve_equation (x : ℝ) (h : (3 * x) / (x + 1) = 9 / (x + 1)) : x = 3 :=
by sorry

end solve_equation_l299_299026


namespace triangle_ABC_is_right_triangle_l299_299299

theorem triangle_ABC_is_right_triangle (A B C : ℝ) (hA : A = 68) (hB : B = 22) :
  A + B + C = 180 → C = 90 :=
by
  intro hABC
  sorry

end triangle_ABC_is_right_triangle_l299_299299


namespace part_a_proof_part_b_proof_l299_299790

-- Definitions for Part (a)

variables {A B C M C_1 B_1 : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space M] [metric_space C_1] [metric_space B_1]
  
-- Assume properties of right triangles constructed externally on the sides of triangle ABC
variables (ABC1 : triangle B C A C1)
variables (AB1C : triangle B1 A C B)

-- External right triangles with specified angles
axiom ex_triangle_ABC1 : ∀ {ABC : triangle A B C}, ∀ {ϕ : angle}, right_triangle ABC1 ∧ angle.BAC = ϕ ∧ angle.C1 = pi/2
axiom ex_triangle_AB1C : ∀ {ABC : triangle A B C}, ∀ {ϕ : angle}, right_triangle AB1C ∧ angle.BCA = ϕ ∧ angle.B1 = pi/2

-- Midpoint M of BC
axiom M_midpoint : ∀ {A B C : point}, is_midpoint (M) B C 

-- Required to prove MB1 = MC1 and the specified angle relationship
theorem part_a_proof : ∀ {A B C M C1 B1 : point}, (is_midpoint M B C) → right_triangle ABC1 → (angle.BAC = ϕ) → right_triangle AB1C → (angle.BCA = ϕ) →  (dist M B1 = dist M C1) ∧ (angle B1 M C1 = 2 * ϕ) := 
sorry

-- Definitions for Part (b)

variables {A B C G : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space G]

-- External equilateral triangles on the sides of triangle ABC
variables (E1 E2 E3 : Type) [metric_space E1] [metric_space E2] [metric_space E3]

-- Centers of externally constructed equilateral triangles on the sides
axiom centers_equi_triangles : ∀ {A B C : point}, ∀ {E1 E2 E3 : point}, equilateral_triangle E1 E2 E3 ∧ is_center_tri E1 B C ∧ is_center_tri E2 A C ∧ is_center_tri E3 A B

-- Required to prove that the centers form an equilateral triangle coinciding with centroid
theorem part_b_proof : ∀ {A B C G : point}, equilateral_triangle E1 E2 E3 → (is_center_tri E1 B C) → (is_center_tri E2 A C) → (is_center_tri E3 A B) → equilateral_triangle (E1 E2 E3) ∧ centroid (triangle A B C) (E1 E2 E3) :=
sorry

end part_a_proof_part_b_proof_l299_299790


namespace max_sum_of_multiplication_table_l299_299409

def prime_numbers : list ℕ := [2, 3, 5, 7, 11, 13, 17]

def sum_of_primes (lst : list ℕ) : ℕ := lst.sum

theorem max_sum_of_multiplication_table :
  ∀ (a b c d e f g : ℕ), a ∈ prime_numbers → b ∈ prime_numbers → c ∈ prime_numbers → 
  d ∈ prime_numbers → e ∈ prime_numbers → f ∈ prime_numbers → g ∈ prime_numbers → 
  a ≠ b → a ≠ c → a ≠ d → a ≠ e → a ≠ f → a ≠ g →
  b ≠ c → b ≠ d → b ≠ e → b ≠ f → b ≠ g →
  c ≠ d → c ≠ e → c ≠ f → c ≠ g →
  d ≠ e → d ≠ f → d ≠ g →
  e ≠ f → e ≠ g → 
  f ≠ g →
  sum_of_primes [a, b, c, d, e, f, g] = 58 →
  (a + b + c + d) * (e + f + g) ≤ 841 :=
by sorry

end max_sum_of_multiplication_table_l299_299409


namespace prob_A_and_B_same_area_l299_299488

/-- We have 4 employees, including A and B, assigned to either the Food Exhibition Area 
or the Car Exhibition Area, with 2 employees in each area. What is the probability 
that A and B are assigned to the same area? --/
theorem prob_A_and_B_same_area : 
  let total_employees := 4 in
  let area1_num := 2 in
  let area2_num := 2 in
  let favorable_outcomes := 2 in
  let total_outcomes := 6 in
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 1 / 3 :=
by
  sorry

end prob_A_and_B_same_area_l299_299488


namespace integral_evaluation_l299_299546

theorem integral_evaluation : ∫ x in 0..1, (3 * x^2 + 1) = 2 := 
by 
  sorry

end integral_evaluation_l299_299546


namespace closest_vector_l299_299564

/-- Define vectors b and the parameterized vector w. -/
def b : ℝ^3 := ![5, 5, 6]

def w (s : ℝ) : ℝ^3 := ![3 + 8 * s, -2 + 6 * s, -4 - 2 * s]

/-- Define the direction vector v. -/
def v : ℝ^3 := ![8, 6, -2]

/-- state the main theorem to be proven in Lean 4 --/
theorem closest_vector (s : ℝ) : (w s - b) ⬝ v = 0 → s = 19 / 52 := by
  sorry

end closest_vector_l299_299564


namespace a_2019_lt_5_l299_299414

noncomputable def a : ℕ → ℝ
| 0 := 1
| (n+1) := (1 + a n + a n * b n) / (b n)

noncomputable def b : ℕ → ℝ
| 0 := 2
| (n+1) := (1 + b n + a n * b n) / (a n)

theorem a_2019_lt_5 : a 2019 < 5 := 
by {
  sorry
}

end a_2019_lt_5_l299_299414


namespace sequence_term_20_l299_299665

theorem sequence_term_20 :
  ∀ (a : ℕ → ℕ), (a 1 = 1) → (∀ n, a (n+1) = a n + 2) → (a 20 = 39) := by
  intros a h1 h2
  sorry

end sequence_term_20_l299_299665


namespace fraction_of_milk_in_first_cup_is_one_fourth_l299_299957

theorem fraction_of_milk_in_first_cup_is_one_fourth :
  ∀ (tea1 milk1 tea2 milk2 : ℚ),
  tea1 = 6 ∧ milk1 = 0 ∧ tea2 = 0 ∧ milk2 = 6 →
  let tea1_after_first_transfer := tea1 - 2 in
  let tea2_after_first_transfer := tea2 + 2 in
  let milk2_after_first_transfer := milk2 in
  let total_mix_cup2 := tea2_after_first_transfer + milk2_after_first_transfer in
  let tea2_ratio := tea2_after_first_transfer / total_mix_cup2 in
  let milk2_ratio := milk2_after_first_transfer / total_mix_cup2 in
  let transferred_mix := 2 in
  let tea_back := transferred_mix * tea2_ratio in
  let milk_back := transferred_mix * milk2_ratio in
  let tea1_final := tea1_after_first_transfer + tea_back in
  let milk1_final := milk_back in
  let total_final := tea1_final + milk1_final in
  milk1_final / total_final = 1/4 :=
begin
  intros tea1 milk1 tea2 milk2 h,
  let tea1_after_first_transfer := 4,
  let tea2_after_first_transfer := 2,
  let milk2_after_first_transfer := 6,
  let total_mix_cup2 := 8,
  let tea2_ratio := 2 / 8,
  let milk2_ratio := 6 / 8,
  let transferred_mix := 2,
  let tea_back := (2:ℚ) * (2 / 8),
  let milk_back := (2:ℚ) * (6 / 8),
  let tea1_final := 4 + tea_back,
  let milk1_final := milk_back,
  let total_final := tea1_final + milk1_final,
  rw [‹tea1_after_first_transfer = 4›, ‹tea2_after_first_transfer = 2›, ‹milk2_after_first_transfer = 6›, ‹total_mix_cup2 = 8›,
      ‹tea2_ratio = 2 / 8›, ‹milk2_ratio = 6 / 8›, ‹transferred_mix = 2›, ‹tea_back = (2 * (2 / 8))›, ‹milk_back = (2 * (6 / 8))›,
      ‹tea1_final = 4 + tea_back›, ‹milk1_final = milk_back›, ‹total_final = tea1_final + milk1_final›],
  norm_num,
  sorry
end

end fraction_of_milk_in_first_cup_is_one_fourth_l299_299957


namespace ratio_of_segments_l299_299687

open EuclideanGeometry

noncomputable def midpoint (A B : Point) : Point := sorry -- Assuming midpoint definition

theorem ratio_of_segments (A B C P O M : Point) 
  (hM_mid : M = midpoint A C)
  (hP_on_BC : isOnLine P B C)
  (hO_int: O = intersection (Line.mk A P) (Line.mk B M))
  (hBO_BP : distance O B = distance B P) :
  distance O M / distance P C = 1 / 2 :=
sorry

end ratio_of_segments_l299_299687


namespace caesars_meal_cost_is_30_l299_299382

-- Define the charges for each hall
def caesars_room_rent : ℕ := 800
def caesars_meal_cost (C : ℕ) (guests : ℕ) : ℕ := C * guests
def venus_room_rent : ℕ := 500
def venus_meal_cost : ℕ := 35
def venus_total_cost (guests : ℕ) : ℕ := venus_room_rent + venus_meal_cost * guests

-- The number of guests when costs are the same
def guests : ℕ := 60

-- The total cost at Caesar's for 60 guests
def caesars_total_cost (C : ℕ) : ℕ := caesars_room_rent + caesars_meal_cost C guests

-- The fact that the costs are the same
def costs_are_equal (C : ℕ) : Prop := caesars_total_cost C = venus_total_cost guests

-- The theorem to prove
theorem caesars_meal_cost_is_30 : ∃ (C : ℕ), costs_are_equal C ∧ C = 30 :=
by 
  let C := 30
  have h1: caesars_total_cost C = venus_total_cost guests,
  {
    calc
      caesars_total_cost C
        = 800 + 60 * C : by rw [caesars_meal_cost, caesars_room_rent, guests]
    ... = 800 + 60 * 30 : by rw [C]
    ... = 800 + 1800 : by norm_num
    ... = 2600 : by norm_num
    ... = 500 + 2100: by norm_num
    ... = 500 + 60 * 35 : by rw [venus_meal_cost, guests]
    ... = venus_total_cost guests : by rw [venus_total_cost, venus_room_rent]
  },
  have h2: C = 30 := by refl,
  use C,
  exact ⟨h1, h2⟩

end caesars_meal_cost_is_30_l299_299382


namespace value_of_a_minus_b_exp_2023_l299_299618

theorem value_of_a_minus_b_exp_2023 (a b : ℝ) 
  (h1 : ∀ x, x + a > 1 → x ∈ Ioo (-2 : ℝ) 3)
  (h2 : ∀ x, 2 * x - b < 2 → x ∈ Ioo (-2 : ℝ) 3) : 
  (a - b) ^ 2023 = -1 := 
sorry

end value_of_a_minus_b_exp_2023_l299_299618


namespace spinner_divisible_by_5_l299_299888

theorem spinner_divisible_by_5 :
  ∀ (digits : Fin 4 → Fin 3), (∃ k : ℕ, (∑ i, digits i * (10^i)) = k * 5) → False :=
by sorry

end spinner_divisible_by_5_l299_299888


namespace cone_height_calculation_l299_299944

def cone_height (l r : ℝ) : ℝ := sqrt (l^2 - r^2)

theorem cone_height_calculation : cone_height 10 5 = 5 * sqrt 3 :=
by
sorry

end cone_height_calculation_l299_299944


namespace min_area_tangents_l299_299336

theorem min_area_tangents (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f := λ x : ℝ, 3 - x^2
  let A := (a, f a)
  let B := (-b, f (-b))
  ∃ (area : ℝ), (∀ A B, area ≥ 8) := sorry

end min_area_tangents_l299_299336


namespace count_decreasing_digits_l299_299248

theorem count_decreasing_digits :
  let valid_numbers := {n | 100 ≤ n ∧ n ≤ 999 ∧
                         let h := n / 100,
                             t := (n / 10) % 10,
                             u := n % 10
                         in 1 ≤ h ∧ h ≤ 9 ∧
                            0 ≤ t ∧ t < h ∧
                            0 ≤ u ∧ u < t ∧
                            (h ≠ t ∧ h ≠ u ∧ t ≠ u)} in
  valid_numbers.card = 120 :=
by sorry

end count_decreasing_digits_l299_299248


namespace complex_num_in_second_quadrant_l299_299221

-- Definitions and conditions
def complex_num_z : ℂ := (i^2) / (1 + i)

-- The proof statement
theorem complex_num_in_second_quadrant : complex_num_z.im > 0 ∧ complex_num_z.re < 0 := by
  -- Proof would go here
  sorry

end complex_num_in_second_quadrant_l299_299221


namespace smallest_x_for_quadratic_l299_299783

theorem smallest_x_for_quadratic :
  ∃ x, 8 * x^2 - 38 * x + 35 = 0 ∧ (∀ y, 8 * y^2 - 38 * y + 35 = 0 → x ≤ y) ∧ x = 1.25 :=
by
  sorry

end smallest_x_for_quadratic_l299_299783


namespace solve_AF1_plus_BF1_l299_299206

-- Define the foci and properties of the ellipse
variable {F₁ F₂ : ℝ × ℝ}
def is_ellipse (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def foci (a b c : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := (⟨c, 0⟩, ⟨-c, 0⟩)

-- Given the specific ellipse and length |AB|
def a : ℝ := 4
def b : ℝ := 3
def c : ℝ := (a^2 - b^2)^.sqrt
def AB : ℝ := 5

-- Assuming A and B on the ellipse
variable {A B : ℝ × ℝ}
def on_ellipse (p : ℝ × ℝ) : Prop :=
  ∃ x y, p = (x, y) ∧ is_ellipse a b x y

-- Define the specific question to be proven in Lean
theorem solve_AF1_plus_BF1 :
  ∀ {F₁ F₂ A B : ℝ × ℝ},
    (F₁, F₂) = foci a b c →
    on_ellipse A →
    on_ellipse B →
    dist A B = AB →
    dist A F₁ + dist B F₁ = 11 :=
by
  intros
  sorry

end solve_AF1_plus_BF1_l299_299206


namespace profitable_year_exists_option2_more_economical_l299_299854

noncomputable def total_expenses (x : ℕ) : ℝ := 2 * (x:ℝ)^2 + 10 * x  

noncomputable def annual_income (x : ℕ) : ℝ := 50 * x  

def year_profitable (x : ℕ) : Prop := annual_income x > total_expenses x + 98 / 1000

theorem profitable_year_exists : ∃ x : ℕ, year_profitable x ∧ x = 3 := sorry

noncomputable def total_profit (x : ℕ) : ℝ := 
  50 * x - 2 * (x:ℝ)^2 + 10 * x - 98 / 1000 + if x = 10 then 8 else if x = 7 then 26 else 0

theorem option2_more_economical : 
  total_profit 10 = 110 ∧ total_profit 7 = 110 ∧ 7 < 10 :=
sorry

end profitable_year_exists_option2_more_economical_l299_299854


namespace minimum_bars_for_discount_l299_299810

variable (x : ℕ)

def cost_first_method(x : ℕ) : ℚ :=
  if x = 1 then 2 else 2 + 1.4 * (x - 1)

def cost_second_method(x : ℕ) : ℚ :=
  1.6 * x

theorem minimum_bars_for_discount :
  (2 + 1.4 * (x - 1) < 1.6 * x) → x = 4 :=
by
  intro h
  sorry

end minimum_bars_for_discount_l299_299810


namespace max_value_of_fx_l299_299606

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem max_value_of_fx : 
  ∃ M ∈ set.image f (set.Icc (-2 : ℝ) 2), ∀ y ∈ set.image f (set.Icc (-2 : ℝ) 2), y ≤ M :=
by 
  sorry

end max_value_of_fx_l299_299606


namespace domain_of_rational_function_l299_299445

theorem domain_of_rational_function :
  { x : ℝ | x ≠ 7 } = set.univ \ {7} :=
by {
  sorry
}

end domain_of_rational_function_l299_299445


namespace loss_per_meter_is_5_l299_299828

-- Define the conditions
def selling_price : ℕ := 18000
def cost_price_per_meter : ℕ := 50
def quantity : ℕ := 400

-- Define the statement to prove (question == answer given conditions)
theorem loss_per_meter_is_5 : 
  ((cost_price_per_meter * quantity - selling_price) / quantity) = 5 := 
by
  sorry

end loss_per_meter_is_5_l299_299828


namespace minimum_weights_set_l299_299072

/-- 
  Given a set of weights with the following properties:
  1. The set consists of 5 weights, each with a distinct weight.
  2. For any two weights, there are two other weights with the same total weight.
  
  Prove that the minimum number of weights that can be in this set is 13.
 -/
theorem minimum_weights_set (weights : set ℝ) 
  (h_distinct : weights.card = 5)
  (h_pairs : ∀ w1 w2 ∈ weights, ∃ w3 w4 ∈ weights, w1 + w2 = w3 + w4) :
  weights.card ≥ 13 :=
by sorry

end minimum_weights_set_l299_299072


namespace tangents_perpendicular_to_side_l299_299743

-- Define the vertices of the rectangle, intersection points, and tangents
variables (A B C D O P Q : Type)
  [metric_space O]

-- Conditions of the problem
def rectangle (A B C D : O) : Prop :=
  -- Definition of the rectangle with its properties
  sorry

def circle_center (O : O) (r : ℝ) : Prop := 
  -- Definition of the circle centered at O with radius r
  sorry

def tangents (A B C D : O) (O : O) : Prop := 
  -- Definition of the tangents from vertices to the circle
  sorry

def tangent_points (A B C D O P Q : O) : Prop := 
  -- Definition of the points P and Q created by tangents intersection
  sorry

-- Prove PQ is perpendicular to AB given the above conditions
theorem tangents_perpendicular_to_side {A B C D O P Q : O} :
  rectangle A B C D → 
  circle_center O (dist O A) → 
  tangents A B C D O → 
  tangent_points A B C D O P Q → 
  ∃ M N : O, PQ ⊥ AB :=
  sorry

end tangents_perpendicular_to_side_l299_299743


namespace glen_animals_total_impossible_l299_299760

theorem glen_animals_total_impossible (t : ℕ) :
  ¬ (∃ t : ℕ, 41 * t = 108) := sorry

end glen_animals_total_impossible_l299_299760


namespace launderette_machines_l299_299159

def quarters_per_machine := 80
def dimes_per_machine := 100
def total_income := 90
def quarter_value := 0.25
def dime_value := 0.10
def income_per_machine := (quarters_per_machine * quarter_value) + (dimes_per_machine * dime_value)
def num_machines := total_income / income_per_machine

theorem launderette_machines : num_machines = 3 := by
  sorry

end launderette_machines_l299_299159


namespace division_of_sums_and_products_l299_299773

theorem division_of_sums_and_products (a b c : ℕ) (h_a : a = 7) (h_b : b = 5) (h_c : c = 3) :
  (a^3 + b^3 + c^3) / (a^2 - a * b + b^2 - b * c + c^2) = 15 := by
  -- proofs go here
  sorry

end division_of_sums_and_products_l299_299773


namespace find_x_l299_299901

theorem find_x (x : ℝ) (h : 2^(x+1) * 8^(x-1) = 16^3) : x = 3.5 :=
by {
  sorry,
}

end find_x_l299_299901


namespace sequence_divisible_l299_299361

theorem sequence_divisible (n : ℕ) : 
  let a_n := 11^(n+2) + 12^(2n+1) in 
  a_n % 133 = 0 := 
by 
  sorry

end sequence_divisible_l299_299361


namespace simplify_expression_l299_299601

theorem simplify_expression (b : ℝ) (h1 : b ≠ 1) (h2 : b ≠ 1 / 2) :
  (1 / 2 - 1 / (1 + b / (1 - 2 * b))) = (3 * b - 1) / (2 * (1 - b)) :=
sorry

end simplify_expression_l299_299601


namespace figure_can_form_square_l299_299130

-- Define the problem statement in Lean
theorem figure_can_form_square (n : ℕ) (h : is_perfect_square n) : 
  ∃ (parts : list (set (ℕ × ℕ))), parts.length = 3 ∧ 
  (⋃ p in parts, p) = set.univ.filter (fun i => i.1 * i.2 < n) ∧ 
  (∃ k : ℕ, is_square k = some (sqrt n) ∧ 
  (⋃ p in parts, ∃ (x_shift y_shift : ℤ), (λ (i : ℕ × ℕ), (i.1 + x_shift, i.2 + y_shift)) '' p = set.univ.filter (fun i => i.1 < k ∧ i.2 < k))) :=
sorry

end figure_can_form_square_l299_299130


namespace no_real_solutions_l299_299536

theorem no_real_solutions : ∀ (x y : ℝ), ¬ (3 * x^2 + y^2 - 9 * x - 6 * y + 23 = 0) :=
by sorry

end no_real_solutions_l299_299536


namespace relay_team_l299_299268

-- Definitions based on the conditions given in the problem
def athlete := {ZhangMing, WangLiang, LiYang, ZhaoXu}
def legs := {FirstLeg, SecondLeg, ThirdLeg, FourthLeg}

-- Conditions
def condition1 (a: athlete → legs → Prop): Prop :=
  ∀ x, (a ZhangMing x) → ¬ (x = FirstLeg ∨ x = SecondLeg)

def condition2 (a: athlete → legs → Prop): Prop := 
  ∀ x, (a WangLiang x) → ¬ (x = FirstLeg ∨ x = FourthLeg)

def condition3 (a: athlete → legs → Prop): Prop := 
  ∀ x, (a LiYang x) → ¬ (x = FirstLeg ∨ x = FourthLeg)

def condition4 (a: athlete → legs → Prop): Prop :=
  ∀ x y, (a WangLiang SecondLeg = false) → (a ZhaoXu FirstLeg = false)

-- Problem statement
theorem relay_team (a : athlete → legs → Prop)
    (h1: condition1 a) (h2: condition2 a) (h3: condition3 a) (h4: condition4 a) : 
    a LiYang ThirdLeg := 
sorry

end relay_team_l299_299268


namespace find_number_of_machines_l299_299161

noncomputable def number_of_machines 
  (total_money : ℝ) 
  (quarters_per_machine : ℕ) 
  (quarter_value : ℝ) 
  (dimes_per_machine : ℕ) 
  (dime_value : ℝ) 
  (money_per_machine : ℝ) : ℕ := 
  total_money / money_per_machine

theorem find_number_of_machines 
  (total_money : ℝ := 90) 
  (quarters_per_machine : ℕ := 80) 
  (quarter_value : ℝ := 0.25) 
  (dimes_per_machine : ℕ := 100) 
  (dime_value : ℝ := 0.10) 
  (money_per_machine : ℝ := (quarters_per_machine * quarter_value) + (dimes_per_machine * dime_value)) : 
  number_of_machines total_money quarters_per_machine quarter_value dimes_per_machine dime_value money_per_machine = 3 := by 
  sorry

end find_number_of_machines_l299_299161


namespace maximum_m_b_l299_299348

noncomputable def m (b : ℕ) : ℕ :=
  Nat.find (λ m, ∀ n ≥ 0, b^n (b - 1) > 2018)

theorem maximum_m_b (b : ℕ) (hb : b ≥ 2) : ∃ m, m = 2188 ∧ ∀ n ≥ 0, b^(n+1) - b^n ≤ 2018 → m ≤ 2188 := 
by
  sorry

end maximum_m_b_l299_299348


namespace apple_tree_production_l299_299513

def first_year_production : ℕ := 40
def second_year_production (first_year_production : ℕ) : ℕ := 2 * first_year_production + 8
def third_year_production (second_year_production : ℕ) : ℕ := second_year_production - (second_year_production / 4)
def total_production (first_year_production second_year_production third_year_production : ℕ) : ℕ :=
    first_year_production + second_year_production + third_year_production

-- Proof statement
theorem apple_tree_production : total_production 40 88 66 = 194 := by
  sorry

end apple_tree_production_l299_299513


namespace sector_area_l299_299924

theorem sector_area (r θ : ℝ) (hr : r = 2) (hθ : θ = (45 : ℝ) * (Real.pi / 180)) : 
  (1 / 2) * r^2 * θ = Real.pi / 2 := 
by
  sorry

end sector_area_l299_299924


namespace max_soap_boxes_correct_l299_299094

def carton_width := 25
def carton_height := 48
def carton_length := 60

def soap_width := 8
def soap_height := 6
def soap_length := 5

def space_padding := 0.5

noncomputable def max_soap_boxes : ℕ :=
  let effective_soap_width := soap_width + 2 * space_padding
  let effective_soap_height := soap_height + 2 * space_padding
  let count_width := carton_width / effective_soap_width
  let count_height := carton_height / effective_soap_height
  let count_length := carton_length / soap_length
  (count_width.toNat) * (count_height.toNat) * (count_length.toNat)

theorem max_soap_boxes_correct :
  max_soap_boxes = 144 :=
  by
  sorry

end max_soap_boxes_correct_l299_299094


namespace probability_leftmost_green_on_second_purple_off_l299_299372

theorem probability_leftmost_green_on_second_purple_off :
  ∀ (lamp_colors : List Bool) (lamp_states : List Bool),
    lamp_colors.length = 8 →
    lamp_states.length = 8 →
    lamp_colors.count tt = 4 →
    lamp_states.count tt = 4 →
    lamp_colors.head = tt →
    lamp_states.head = tt →
    lamp_colors.nth 1 = some ff →
    lamp_states.nth 1 = some ff →
    (∃ n : ℚ, n = 4 / 49) :=
by
  intros _ _ _ _ _ _ _ _ _ _ _ _ _; exact ⟨4/49, rfl⟩


end probability_leftmost_green_on_second_purple_off_l299_299372


namespace f_monotonic_intervals_f_maximum_value_on_interval_l299_299604

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_monotonic_intervals :
  (∀ x : ℝ, 0 < x → x < Real.e → f x < f Real.e) ∧
  (∀ x : ℝ, x > Real.e → f x < f Real.e) :=
sorry

theorem f_maximum_value_on_interval (m : ℝ) (hm : m > 0) :
  ∃ x_max : ℝ, x_max ∈ (Set.Icc m (2 * m)) ∧
  (∀ x ∈ (Set.Icc m (2 * m)), f x ≤ f x_max) ∧
  ((m₁ : 0 < m ∧ m ≤ Real.e / 2 → x_max = 2 * m ∧ f x_max = (Real.log (2 * m)) / (2 * m)) ∨
   (m₂ : m ≥ Real.e → x_max = m ∧ f x_max = (Real.log m) / m) ∨
   (m₃ : Real.e / 2 < m ∧ m < Real.e → x_max = Real.e ∧ f x_max = 1 / Real.e)) :=
sorry

end f_monotonic_intervals_f_maximum_value_on_interval_l299_299604


namespace find_interior_angles_l299_299300

theorem find_interior_angles (A B C : ℝ) (h1 : B = A + 10) (h2 : C = B + 10) (h3 : A + B + C = 180) : 
  A = 50 ∧ B = 60 ∧ C = 70 := by
  sorry

end find_interior_angles_l299_299300


namespace arithmetic_mean_25_41_50_l299_299774

theorem arithmetic_mean_25_41_50 :
  (25 + 41 + 50) / 3 = 116 / 3 := by
  sorry

end arithmetic_mean_25_41_50_l299_299774


namespace tony_initial_amount_l299_299435

theorem tony_initial_amount : 
  ∀ (cheese_cost_per_pound beef_cost_per_pound cheese_pounds beef_pounds amount_left : ℕ), 
  cheese_cost_per_pound = 7 → 
  beef_cost_per_pound = 5 → 
  cheese_pounds = 3 → 
  beef_pounds = 1 → 
  amount_left = 61 → 
  let cheese_cost := cheese_cost_per_pound * cheese_pounds 
  let beef_cost :=  beef_cost_per_pound * beef_pounds 
  let total_cost := cheese_cost + beef_cost 
  ∃ initial_amount : ℕ, initial_amount = amount_left + total_cost ∧ initial_amount = 87 :=
by 
  intros cheese_cost_per_pound beef_cost_per_pound cheese_pounds beef_pounds amount_left
  intros h_cheese_cost_per_pound h_beef_cost_per_pound h_cheese_pounds h_beef_pounds h_amount_left
  let cheese_cost := cheese_cost_per_pound * cheese_pounds 
  let beef_cost := beef_cost_per_pound * beef_pounds 
  let total_cost := cheese_cost + beef_cost 
  use amount_left + total_cost 
  split 
  { 
    sorry
  }, 
  { 
    sorry 
  }

end tony_initial_amount_l299_299435


namespace carol_cheryl_same_color_probability_l299_299482

theorem carol_cheryl_same_color_probability :
  let total_ways := (∏ k in {9, 6, 3, 1}, choose 9 k)
  let favorable_ways := 3 * choose 3 3
  total_ways = 5040 ∧ favorable_ways = 3 ∧ 
  (favorable_ways : ℚ) / (total_ways : ℚ) = 1 / 1680 := 
by
  sorry

end carol_cheryl_same_color_probability_l299_299482


namespace prove_BA1_perp_AC1_calculate_volume_tetrahedron_l299_299303

structure Prism := 
  (A B C A1 B1 C1 : Point)
  (angle_ABC : ∠ ABC = 90°)
  (AA1 AC BC : ℝ)
  (projection_A1 : Point)
  (projection_A1_is_midpoint_D : projection_A1 = midpoint AC)

def BA1_perp_AC1 (P : Prism) : Prop :=
  ∃ (BA1 AC1 : Vector), BA1 ⊥ AC1

theorem prove_BA1_perp_AC1 (P : Prism) (h1 : P.angle_ABC = 90°)
  (h2 : P.AA1 = 2) (h3 : P.AC = 2) (h4 : P.BC = 2)
  (h5 : P.projection_A1 = midpoint P.AC) :
  BA1_perp_AC1 P :=
sorry

def volume_tetrahedron (P : Prism) : ℝ :=
  1 / 6 * volume_of_prism P

theorem calculate_volume_tetrahedron (P : Prism) (h1 : P.angle_ABC = 90°)
  (h2 : P.AA1 = 2) (h3 : P.AC = 2) (h4 : P.BC = 2)
  (h5 : P.projection_A1 = midpoint P.AC) :
  volume_tetrahedron P = 2 / 3 :=
sorry

end prove_BA1_perp_AC1_calculate_volume_tetrahedron_l299_299303


namespace compare_a_l299_299135

theorem compare_a (a : ℝ) (α : ℝ) (h1 : α = 0.00001)
  (h2 : 0.99999 = 1 - α) (h3 : 1.00001 = 1 + α) 
  (h4 : a = 0.99999 ^ 1.00001 * 1.00001 ^ 0.99999) : a < 1 :=
sorry

end compare_a_l299_299135


namespace complement_of_P_in_U_l299_299644

def U : Set ℤ := {-1, 0, 1, 2}
def P : Set ℤ := {x | -Real.sqrt 2 < x ∧ x < Real.sqrt 2}
def compl_U (P : Set ℤ) : Set ℤ := {x ∈ U | x ∉ P}

theorem complement_of_P_in_U : compl_U P = {2} :=
by
  sorry

end complement_of_P_in_U_l299_299644


namespace domain_f_l299_299593

noncomputable def a := 4
noncomputable def b := 2

def f (x : ℝ) : ℝ := sqrt (1 / b - log a (x - 1))

theorem domain_f : ∀ x : ℝ, (1 < x ∧ x ≤ 3) ↔ 0 ≤ 1 / b - log a (x - 1) ∧ 1 < x :=
begin
  assume x,
  -- simplification and proving omitted for clarity
  sorry
end

end domain_f_l299_299593


namespace heated_water_mass_is_correct_l299_299439

variables (P : ℝ) (t1 : ℝ) (deltaT : ℝ) (t2 : ℝ) (cB : ℝ)

-- Given values
def P_val := 500 -- Power in Watts
def t1_val := 60 -- Heating time in seconds
def deltaT_val := 2 -- Temperature increase in °C
def t2_val := 120 -- Cooling time in seconds
def cB_val := 4200 -- Specific heat capacity in J/kg·°C

-- The equation derived from the conditions
def mass (P t1 deltaT t2 cB : ℝ) : ℝ :=
  P * t1 / (cB * deltaT * (1 + t1 / t2))

theorem heated_water_mass_is_correct : 
  mass P_val t1_val deltaT_val t2_val cB_val = 2.38 :=
begin
  -- Begin proof
  -- values are substituted directly into the final theoretical mass equation
  sorry
end

end heated_water_mass_is_correct_l299_299439


namespace total_cost_of_books_l299_299803

theorem total_cost_of_books (total_children : ℕ) (n : ℕ) (extra_payment_per_child : ℕ) (cost : ℕ) :
  total_children = 12 →
  n = 2 →
  extra_payment_per_child = 10 →
  (total_children - n) * extra_payment_per_child = 100 →
  cost = 600 :=
by
  intros h1 h2 h3 h4
  sorry

end total_cost_of_books_l299_299803


namespace fraction_zero_implies_x_eq_one_l299_299981

theorem fraction_zero_implies_x_eq_one (x : ℝ) (h : (x - 1) / (x + 1) = 0) : x = 1 :=
sorry

end fraction_zero_implies_x_eq_one_l299_299981


namespace eq_relation_q_r_l299_299664

-- Define the angles in the context of the problem
variables {A B C D E F : Type}
variables {angle_BAC angle_BFD angle_ADE angle_FEC : ℝ}
variables (right_triangle_ABC : A → B → C → angle_BAC = 90)

-- Equilateral triangle DEF inscribed in ABC
variables (inscribed_equilateral_DEF : D → E → F)
variables (angle_BFD_eq_p : ∀ p : ℝ, angle_BFD = p)
variables (angle_ADE_eq_q : ∀ q : ℝ, angle_ADE = q)
variables (angle_FEC_eq_r : ∀ r : ℝ, angle_FEC = r)

-- Main statement to be proved
theorem eq_relation_q_r {p q r : ℝ} 
  (right_triangle_ABC : angle_BAC = 90)
  (angle_BFD : angle_BFD = 30 + q)
  (angle_FEC : angle_FEC = 120 - r) :
  q + r = 60 :=
sorry

end eq_relation_q_r_l299_299664


namespace midpoint_eq_l299_299896

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def midpoint (P Q : Point3D) : Point3D :=
  { x := (P.x + Q.x) / 2,
    y := (P.y + Q.y) / 2,
    z := (P.z + Q.z) / 2 }

def P : Point3D := { x := 1, y := 4, z := -3 }
def Q : Point3D := { x := 3, y := -2, z := 5 }

theorem midpoint_eq : midpoint P Q = { x := 2, y := 1, z := 1 } :=
by
  -- Proof placeholder
  sorry

end midpoint_eq_l299_299896


namespace distance_between_two_cars_l299_299475

-- Define the initial conditions and statement
theorem distance_between_two_cars :
  ∀ (initial_distance distance_car1 distance_car2 : ℕ),
  initial_distance = 113 →
  distance_car1 = 50 →
  distance_car2 = 35 →
  initial_distance - (distance_car1 + distance_car2) = 28 :=
by
  intros initial_distance distance_car1 distance_car2 h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end distance_between_two_cars_l299_299475


namespace parallel_segments_k_value_l299_299540

open Real

theorem parallel_segments_k_value :
  let A' := (-6, 0)
  let B' := (0, -6)
  let X' := (0, 12)
  ∃ k : ℝ,
  let Y' := (18, k)
  let m_ab := (B'.2 - A'.2) / (B'.1 - A'.1)
  let m_xy := (Y'.2 - X'.2) / (Y'.1 - X'.1)
  m_ab = m_xy → k = -6 :=
by
  sorry

end parallel_segments_k_value_l299_299540


namespace priya_speed_l299_299371

theorem priya_speed (Riya_speed Priya_speed : ℝ) (time_separation distance_separation : ℝ)
  (h1 : Riya_speed = 30) 
  (h2 : time_separation = 45 / 60) -- 45 minutes converted to hours
  (h3 : distance_separation = 60)
  : Priya_speed = 50 :=
sorry

end priya_speed_l299_299371


namespace product_of_last_two_digits_l299_299638

theorem product_of_last_two_digits (A B : ℕ) (h1 : A + B = 14) (h2 : B = 0 ∨ B = 5) : A * B = 45 :=
sorry

end product_of_last_two_digits_l299_299638


namespace smallest_odd_polygon_with_parallelograms_l299_299053

theorem smallest_odd_polygon_with_parallelograms :
  ∃ n, odd n ∧ (n ≥ 3) ∧ (∀ k, odd k ∧ k < n → ¬ can_be_divided_into_parallelograms k) ∧ can_be_divided_into_parallelograms n :=
sorry

def odd (n : ℕ) : Prop := n % 2 = 1

def can_be_divided_into_parallelograms (sides : ℕ) : Prop :=
  sorry

end smallest_odd_polygon_with_parallelograms_l299_299053


namespace complex_division_simplification_l299_299212

theorem complex_division_simplification (i : ℂ) (h_i : i * i = -1) : (1 - 3 * i) / (2 - i) = 1 - i := by
  sorry

end complex_division_simplification_l299_299212


namespace find_AC_l299_299517

noncomputable def find_AC' (BC' : ℝ) : ℝ := 
  real.cbrt 2

theorem find_AC'_correct (BC' : ℝ) (hBC' : BC' = 1) : find_AC' BC' = real.cbrt 2 := 
  by
    rw [hBC']
    sorry

end find_AC_l299_299517


namespace find_smallest_n_to_make_perfect_square_l299_299448

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem find_smallest_n_to_make_perfect_square :
  ∃ n : ℕ, n > 0 ∧ is_perfect_square (1008 * n) ∧ ∀ m : ℕ, m > 0 ∧ is_perfect_square (1008 * m) → n ≤ m :=
begin
  use 7,
  -- proof will be provided here
  sorry
end

end find_smallest_n_to_make_perfect_square_l299_299448


namespace can_obtain_6_9_14_cannot_obtain_2011_2012_2013_cannot_obtain_2011_2013_2015_l299_299754

-- Define a transformation on a list of three numbers where we replace the third number with a + b - 1
def transform (a b c d : ℕ) : ℕ := a + b - 1

-- Initial list of numbers: 2, 3, 4
def init_numbers : List ℕ := [2, 3, 4]

-- Targets to reach
def target1 : List ℕ := [6, 9, 14]
def target2 : List ℕ := [2011, 2012, 2013]
def target3 : List ℕ := [2011, 2013, 2015]

-- Theorems to prove
theorem can_obtain_6_9_14 : init_numbers ⟹* target1 :=
sorry

theorem cannot_obtain_2011_2012_2013 : ¬ (init_numbers ⟹* target2) :=
sorry

theorem cannot_obtain_2011_2013_2015 : ¬ (init_numbers ⟹* target3) :=
sorry

end can_obtain_6_9_14_cannot_obtain_2011_2012_2013_cannot_obtain_2011_2013_2015_l299_299754


namespace polynomial_root_condition_l299_299554

noncomputable def polynomial_q (q x : ℝ) : ℝ :=
  x^6 + 3 * q * x^4 + 3 * x^4 + 3 * q * x^2 + x^2 + 3 * q + 1

theorem polynomial_root_condition (q : ℝ) :
  (∃ x > 0, polynomial_q q x = 0) ↔ (q ≥ 3 / 2) :=
sorry

end polynomial_root_condition_l299_299554


namespace marker_distance_l299_299114

theorem marker_distance (k : ℝ) (h_pos : 0 < k) (h_dist : sqrt (16 + 16 * k^2) = 31) : 
  dist (7, 7 * k) (19, 19 * k) = 93 := 
by sorry

end marker_distance_l299_299114


namespace probability_is_correct_l299_299762

namespace ProbabilityProof

-- Define the labels on the balls
def balls : List ℕ := [1, 2, 2, 3, 4, 5]

-- Noncomputable because we use real number arithmetic
noncomputable def probability_sum_greater_than_7 : ℚ :=
  let total_combinations := (balls.choose 3).length
  let non_exceeding_combinations := (balls.choose 3).count (fun l => l.sum ≤ 7)
  1 - (non_exceeding_combinations / total_combinations)

theorem probability_is_correct :
  probability_sum_greater_than_7 = 7 / 10 :=
by
  simp [probability_sum_greater_than_7]
  sorry -- Proof omitted

end ProbabilityProof

end probability_is_correct_l299_299762


namespace correct_propositions_identification_l299_299614

theorem correct_propositions_identification (x y : ℝ) (h1 : x ≥ 0 ∧ y ≥ 0 → x * y ≥ 0)
    (h2 : ¬(x * y ≥ 0 → x ≥ 0 ∧ y ≥ 0))
    (h3 : ¬(¬(x ≥ 0 ∧ y ≥ 0 → x * y ≥ 0)))
    (h4 : (¬(x * y ≥ 0) → ¬(x ≥ 0) ∨ ¬(y ≥ 0))) :
  true :=
by
  -- Proof skipped
  sorry

end correct_propositions_identification_l299_299614


namespace cube_property_l299_299785

theorem cube_property (x : ℝ) (s : ℝ) 
  (h1 : s^3 = 8 * x)
  (h2 : 6 * s^2 = 4 * x) :
  x = 5400 :=
by
  sorry

end cube_property_l299_299785


namespace honda_total_production_l299_299967

theorem honda_total_production (second_shift_production : ℕ)
  (day_shift_mult : ℕ)
  (total_production : ℕ) :
  second_shift_production = 1100 →
  day_shift_mult = 4 →
  total_production = 5500 :=
by
  intros h_second_shift h_day_shift
  have day_shift_production : ℕ := day_shift_mult * second_shift_production
  rw [h_second_shift, h_day_shift] at day_shift_production
  have total : ℕ := day_shift_production + second_shift_production
  rw [h_second_shift, h_day_shift]
  sorry

end honda_total_production_l299_299967


namespace at_least_one_less_than_two_l299_299907

theorem at_least_one_less_than_two (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : 2 < a + b) :
  (1 + b) / a < 2 ∨ (1 + a) / b < 2 := 
by
  sorry

end at_least_one_less_than_two_l299_299907


namespace total_cost_of_shirts_l299_299677

theorem total_cost_of_shirts 
    (first_shirt_cost : ℤ)
    (second_shirt_cost : ℤ)
    (h1 : first_shirt_cost = 15)
    (h2 : first_shirt_cost = second_shirt_cost + 6) : 
    first_shirt_cost + second_shirt_cost = 24 := 
by
  sorry

end total_cost_of_shirts_l299_299677


namespace total_cookies_l299_299257

   -- Define the conditions
   def cookies_per_bag : ℕ := 41
   def number_of_bags : ℕ := 53

   -- Define the problem: Prove that the total number of cookies is 2173
   theorem total_cookies : cookies_per_bag * number_of_bags = 2173 :=
   by sorry
   
end total_cookies_l299_299257


namespace sum_of_roots_eq_six_l299_299255

variable (a b : ℝ)

theorem sum_of_roots_eq_six (h1 : a * (a - 6) = 7) (h2 : b * (b - 6) = 7) (h3 : a ≠ b) : a + b = 6 :=
sorry

end sum_of_roots_eq_six_l299_299255


namespace monotonic_intervals_find_f_max_l299_299602

noncomputable def f (x : ℝ) : ℝ := Real.log x / x

theorem monotonic_intervals :
  (∀ x, 0 < x → x < Real.exp 1 → 0 < (1 - Real.log x) / x^2) ∧
  (∀ x, x > Real.exp 1 → (1 - Real.log x) / x^2 < 0) :=
sorry

theorem find_f_max (m : ℝ) (h : m > 0) :
  if 0 < 2 * m ∧ 2 * m ≤ Real.exp 1 then f (2 * m) = Real.log (2 * m) / (2 * m)
  else if m ≥ Real.exp 1 then f m = Real.log m / m
  else f (Real.exp 1) = 1 / Real.exp 1 :=
sorry

end monotonic_intervals_find_f_max_l299_299602


namespace pairs_satisfied_condition_l299_299456

def set_A : Set ℕ := {1, 2, 3, 4, 5, 6, 10, 11, 12, 15, 20, 22, 30, 33, 44, 55, 60, 66, 110, 132, 165, 220, 330, 660}
def set_B : Set ℕ := {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72}

def is_valid_pair (a b : ℕ) := a ∈ set_A ∧ b ∈ set_B ∧ (a - b = 4)

def valid_pairs : Set (ℕ × ℕ) := 
  {(6, 2), (10, 6), (12, 8), (22, 18)}

theorem pairs_satisfied_condition :
  { (a, b) | is_valid_pair a b } = valid_pairs := 
sorry

end pairs_satisfied_condition_l299_299456


namespace complex_problem_third_quadrant_l299_299591

-- Define the imaginary unit
def i : ℂ := complex.I

-- Define the complex number problem
def complex_problem : Prop :=
  let z := (1 - i)^2 / (1 + i) in
  z = -1 - i ∧ -1 < 0 ∧ 1 < 0 -- This implies z is in the third quadrant

-- The theorem stating the equivalence of our problem to the third quadrant
theorem complex_problem_third_quadrant : complex_problem :=
sorry

end complex_problem_third_quadrant_l299_299591


namespace necessary_work_is_frequency_distribution_l299_299308

-- Definitions based on conditions
def students_count : ℕ := 800

def score_ranges := [
  { lower := 0, upper := 59 },
  { lower := 60, upper := 74 },
  { lower := 75, upper := 89 },
  { lower := 90, upper := 119 },
  { lower := 120, upper := 200 } -- Assuming scores can't be above 200
]

-- Proving the correct answer
theorem necessary_work_is_frequency_distribution 
  (student_count_800 : students_count = 800)
  (ranges_defined : ∀ r ∈ score_ranges, r.lower ≤ r.upper):
  (necessary_work_800_students_scoreranges = "Conduct a frequency distribution") :=
  sorry -- Proof is skipped

end necessary_work_is_frequency_distribution_l299_299308


namespace projection_AO_on_AB_is_2_l299_299657

-- Definitions for the problem's conditions
variables {O A B : Point}
variables (radius : ℝ)
-- Conditions
axiom centerO : O.center = true
axiom OnCircleA : dist O A = radius
axiom OnCircleB : dist O B = radius
axiom lengthAB : dist A B = 4

-- Theorem to prove: projection of AO on AB direction is 2
theorem projection_AO_on_AB_is_2 :
  let projection := dist O A * (dist O A / radius) * real.cos (angle A O B)
  projection = 2 :=
sorry

end projection_AO_on_AB_is_2_l299_299657


namespace find_k_l299_299234

-- Define the sequence {a_n} such that a_n = 63 / 2^n
def a : ℕ → ℝ
| n := 63 / 2 ^ n

-- Theorem statement
theorem find_k :
  ∃ k : ℕ, (∀ n : ℕ, n > 0 → 
              (∏ i in finset.range n, a (i + 1)) ≤ (∏ i in finset.range k, a (i + 1))) ∧ k = 5 :=
begin
  sorry
end

end find_k_l299_299234


namespace intersection_count_l299_299940

def f (x : ℝ) : ℝ := x^2

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

theorem intersection_count
  (f_is_periodic : is_periodic f 2)
  (f_def : ∀ x : ℝ, x ∈ set.Icc (-1) 1 → f x = x^2)
  : ∃ n : ℕ, n = 9 :=
sorry

end intersection_count_l299_299940


namespace evaluate_expression_l299_299894

theorem evaluate_expression (y : ℝ) : 
  let x : ℝ := 1 in 
  let expr : ℝ := x^(4 * y - 1) / (2 * (7⁻¹ + 4⁻¹)) in 
  expr = 14 / 11 := 
by
  -- Proof steps go here
  sorry

end evaluate_expression_l299_299894


namespace find_XZ_l299_299671

-- Definition of the triangle and given conditions
def triangle (X Y Z : Type) := 
  ∃ (XY YZ XZ : ℝ), 
  XY ≠ 0 ∧ YZ ≠ 0 ∧ XZ ≠ 0 ∧
  isRightTriangle (X Y Z) XY YZ XZ ∧
  angle X = 90 ∧
  YZ = 15 ∧
  tan Z = 3 * sin Z

-- Proof that XZ = 5 given the conditions
theorem find_XZ {X Y Z : Type} (XY YZ XZ : ℝ) (h : triangle X Y Z) : 
  XZ = 5 := 
sorry

end find_XZ_l299_299671


namespace sqrt_expression_cases_l299_299363

theorem sqrt_expression_cases (x : ℝ) (hx : x ≥ 1) :
  (x ≤ 2 → sqrt (x + 2 * sqrt (x - 1)) + sqrt (x - 2 * sqrt (x - 1)) = 2) ∧
  (x > 2 → sqrt (x + 2 * sqrt (x - 1)) + sqrt (x - 2 * sqrt (x - 1)) = 2 * sqrt (x - 1)) :=
by sorry

end sqrt_expression_cases_l299_299363


namespace butterfat_percentage_l299_299084

theorem butterfat_percentage (x : ℝ) : 
  let initial_milk_volume := 8
      initial_butterfat_percentage := 0.50
      added_milk_volume := 24
      final_milk_volume := initial_milk_volume + added_milk_volume
      final_butterfat_percentage := 0.20 in
  initial_milk_volume * initial_butterfat_percentage + added_milk_volume * (x / 100) = final_milk_volume * final_butterfat_percentage →
  x = 10 := 
by
  sorry

end butterfat_percentage_l299_299084


namespace tetrahedron_plane_centroid_l299_299334

-- Define the tetrahedron with opposite edges equal
structure Tetrahedron :=
(a b c : ℝ) -- lengths of the pairs of opposite edges
(h1 : a ≤ b)
(h2 : b ≤ c)

-- Definition to state a plane intersects the tetrahedron forming a quadrilateral with minimum perimeter
def plane_min_perimeter_condition (T : Tetrahedron) (P : ℝ × ℝ × ℝ) : Prop :=
∀ (Q : ℝ × ℝ × ℝ), (Q ≠ P) → -- Minimality condition for perimeter of quadrilateral
  let quadrilateral1 := intersection_with_tetrahedron T P,
      quadrilateral2 := intersection_with_tetrahedron T Q in
  quadrilateral1.perimeter ≤ quadrilateral2.perimeter

-- Find the locus of the centroid of quadrilaterals with the minimum perimeter
def centroid_locus_condition (T : Tetrahedron) (ℓ : LineSegment ℝ) : Prop :=
∀ (P : ℝ × ℝ × ℝ), plane_min_perimeter_condition T P →
  let quadrilateral := intersection_with_tetrahedron T P,
      centroid := quadrilateral.centroid in
  centroid ∈ ℓ

-- The final theorem
theorem tetrahedron_plane_centroid (T : Tetrahedron) (ℓ : LineSegment ℝ) :
  (plane_min_perimeter_condition T) ∧ (centroid_locus_condition T ℓ) := 
sorry -- proof omitted

end tetrahedron_plane_centroid_l299_299334


namespace power_equation_value_l299_299472

theorem power_equation_value (n : ℕ) (h : n = 20) : n ^ (n / 2) = 102400000000000000000 := by
  sorry

end power_equation_value_l299_299472


namespace no_group_of_ten_with_given_acquaintances_l299_299368

theorem no_group_of_ten_with_given_acquaintances :
  ¬∃ (G : SimpleGraph (Fin 10)),
  ∃ (f : Fin 10 → ℕ),
  (f = ![9, 9, 9, 8, 8, 8, 7, 6, 4, 4]) ∧
  (∀ v, Nat.card (G.neighborSet v) = f v) ∧
  Symmetric.G :=
sorry

end no_group_of_ten_with_given_acquaintances_l299_299368


namespace finite_matrices_l299_299322

noncomputable def is_extreme_point {n : ℕ} (x : Fin n → ℝ) : Prop :=
  (∃ i, ∀ j, x j = if i = j then 1 else 0) ∨ (∃ i, ∀ j, x j = if i = j then -1 else 0)

noncomputable def is_Hn {n : ℕ} (x : Fin n → ℝ) : Prop :=
  ∑ i, |x i| = 1

noncomputable def maps_to_Hn {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  ∀ x : Fin n → ℝ, is_Hn x → is_Hn (A.mulVec x)

theorem finite_matrices {n : ℕ} (hn : 2 ≤ n) :
  {A : Matrix (Fin n) (Fin n) ℝ // maps_to_Hn A}.toFinset.card = 2^n * n! :=
sorry

end finite_matrices_l299_299322


namespace inequality_smallest_integer_solution_l299_299730

theorem inequality_smallest_integer_solution (x : ℤ) :
    (9 * x + 8) / 6 - x / 3 ≥ -1 → x ≥ -2 := sorry

end inequality_smallest_integer_solution_l299_299730


namespace inequality_l299_299697

open Real

def f1 (x : ℝ) : ℝ := x^2

def f2 (x : ℝ) : ℝ := 2 * (x - x^2)

def f3 (x : ℝ) : ℝ := (1 / 3) * |sin (2 * π * x)|

def a (i : ℕ) : ℝ := i / 99

noncomputable def I (f : ℝ → ℝ) : ℝ :=
  ∑ i in Finset.range 99, |f (a (i + 1)) - f (a i)|

noncomputable def I1 : ℝ := I f1

noncomputable def I2 : ℝ := I f2

noncomputable def I3 : ℝ := I f3

theorem inequality : I2 < I1 ∧ I1 < I3 := by
  sorry

end inequality_l299_299697


namespace g_neither_even_nor_odd_l299_299306

noncomputable def g (x : ℝ) : ℝ := log10 (x - 2)

theorem g_neither_even_nor_odd :
  ¬((∀ x > 2, g x = g (-x)) ∨ (∀ x > 2, g (-x) = -g x)) :=
by sorry

end g_neither_even_nor_odd_l299_299306


namespace choir_arrangement_l299_299086

theorem choir_arrangement :
  let choir_members := 90 in
  let row_lengths := { x : ℕ | x ∣ choir_members ∧ 5 ≤ x ∧ x ≤ 25 } in
  row_lengths.card = 5 :=
by 
  let choir_members := 90 
  let row_lengths := { x : ℕ | x ∣ choir_members ∧ 5 ≤ x ∧ x ≤ 25 }
  sorry

end choir_arrangement_l299_299086


namespace mary_average_speed_l299_299708

noncomputable def trip_distance : ℝ := 1.5 + 1.5
noncomputable def trip_time_minutes : ℝ := 45 + 15
noncomputable def trip_time_hours : ℝ := trip_time_minutes / 60

theorem mary_average_speed :
  (trip_distance / trip_time_hours) = 3 := by
  sorry

end mary_average_speed_l299_299708


namespace eval_expr_at_2_l299_299055

def expr (x : ℝ) : ℝ := (3 * x + 4)^2

theorem eval_expr_at_2 : expr 2 = 100 :=
by sorry

end eval_expr_at_2_l299_299055


namespace number_of_valid_functions_valid_functions_are_two_l299_299560

noncomputable def count_valid_functions : ℕ := sorry

theorem number_of_valid_functions :
  count_valid_functions = 2 := sorry

structure RealFnProperty where
  func : ℝ → ℝ
  prop : ∀ x y z : ℝ, func (x^2 * y) + func (x * z) - func x * func (y * z) ≥ 2

def valid_functions_count : ℕ :=
  {f : RealFnProperty | true}.toFinset.card

theorem valid_functions_are_two :
  valid_functions_count = 2 := sorry

end number_of_valid_functions_valid_functions_are_two_l299_299560


namespace ellipse_min_distance_exists_l299_299223

def ellipse_distance_min (x y : ℝ) : Prop :=
  (x^2 / 25 + y^2 / 9 = 1) ∧ (∃ (d : ℝ), d = 15 / Real.sqrt 41 ∧ d = Real.abs (40 - (4 * x - 5 * y)) / Real.sqrt (4^2 + 5^2))

theorem ellipse_min_distance_exists :
  ∃ (x y : ℝ), ellipse_distance_min x y :=
by
  sorry

end ellipse_min_distance_exists_l299_299223


namespace three_digit_integers_count_l299_299623

/-- The number of different positive three-digit integers that can be formed using only 
the digits from the set {2, 3, 5, 5, 7, 7, 7}, where no digit is used more times than it 
appears in the set, is 43. -/
theorem three_digit_integers_count : 
  let digits := {2, 3, 5, 5, 7, 7, 7}
  in (Σ n in digits, Σ m in digits, Σ o in digits, (n ≠ m ∧ n ≠ o ∧ m ≠ o) 
      ∨ (n = m ∧ n ≠ o) 
      ∨ (n = o ∧ n ≠ m) 
      ∨ (m = n ∧ m ≠ o) 
      ∨ (m = o ∧ n ≠ m) 
      ∨ (o = n ∧ o ≠ m)
      ∨ (o = m ∧ o ≠ n)).fintype.card = 43 := 
sorry

end three_digit_integers_count_l299_299623


namespace monotonic_decreasing_interval_arcsin_l299_299399

noncomputable def g (x : ℝ) : ℝ := x^2 - 2*x

theorem monotonic_decreasing_interval_arcsin :
  ∀ x y : ℝ, (1 - real.sqrt 2) ≤ x ∧ x ≤ 1 → (1 - real.sqrt 2) ≤ y ∧ y ≤ 1 → 
  x ≤ y → arcsin (g y) ≤ arcsin (g x) := by
sorry

end monotonic_decreasing_interval_arcsin_l299_299399


namespace total_cost_of_shirts_is_24_l299_299684

-- Definitions based on conditions
def cost_first_shirt : ℕ := 15
def cost_difference : ℕ := 6
def cost_second_shirt : ℕ := cost_first_shirt - cost_difference

-- The proof problem statement: Calculate total cost given the conditions
theorem total_cost_of_shirts_is_24 : cost_first_shirt + cost_second_shshirt = 24 :=
by
  sorry

end total_cost_of_shirts_is_24_l299_299684


namespace polynomial_at_3_l299_299612

def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

theorem polynomial_at_3 : f 3 = 1641 := 
by
  -- proof would go here
  sorry

end polynomial_at_3_l299_299612


namespace surface_area_of_resulting_solid_l299_299856

-- Define the original cube dimensions
def original_cube_surface_area (s : ℕ) := 6 * s * s

-- Define the smaller cube dimensions to be cut
def small_cube_surface_area (s : ℕ) := 3 * s * s

-- Define the proof problem
theorem surface_area_of_resulting_solid :
  original_cube_surface_area 3 - small_cube_surface_area 1 - small_cube_surface_area 2 + (3 * 1 + 3 * 4) = 54 :=
by
  -- The actual proof is to be filled in here
  sorry

end surface_area_of_resulting_solid_l299_299856


namespace figure_can_form_square_l299_299131

-- Define the problem statement in Lean
theorem figure_can_form_square (n : ℕ) (h : is_perfect_square n) : 
  ∃ (parts : list (set (ℕ × ℕ))), parts.length = 3 ∧ 
  (⋃ p in parts, p) = set.univ.filter (fun i => i.1 * i.2 < n) ∧ 
  (∃ k : ℕ, is_square k = some (sqrt n) ∧ 
  (⋃ p in parts, ∃ (x_shift y_shift : ℤ), (λ (i : ℕ × ℕ), (i.1 + x_shift, i.2 + y_shift)) '' p = set.univ.filter (fun i => i.1 < k ∧ i.2 < k))) :=
sorry

end figure_can_form_square_l299_299131


namespace area_triangle_APQ_l299_299420

open Real

-- Define coordinates of points
def A := (0 : ℝ, 105 : ℝ)
def B := (105 : ℝ, 105 : ℝ)
def C := (105 : ℝ, 0 : ℝ)
def D := (0 : ℝ, 0 : ℝ)
def M := (105 : ℝ, 52.5 : ℝ)
def N := (105 : ℝ, 78.75 : ℝ)

-- Define equations of lines
noncomputable def BD (x : ℝ) := 105 - x
noncomputable def AM (x : ℝ) := - (x / 2) + 105
noncomputable def AN (x : ℝ) := - (x / 4) + 105

-- Define points of intersection
noncomputable def P := (70 : ℝ, 35 : ℝ)
noncomputable def Q := (42 : ℝ, 63 : ℝ)

-- Lean 4 statement for the area of triangle APQ being 1102.5
theorem area_triangle_APQ : 
  let area := 0.5 * abs (A.1 * (P.2 - Q.2) + P.1 * (Q.2 - A.2) + Q.1 * (A.2 - P.2))
  in area = 1102.5 := 
by
  sorry

end area_triangle_APQ_l299_299420


namespace trapezoid_MSNR_area_l299_299651

theorem trapezoid_MSNR_area (P Q R M N S : Points) (areaPQR : ℝ) (height : ℝ) (n : ℕ) (small_triangle_area : ℝ) :
  Similar (Triangle P Q R) (Triangle M N S) ∧ PQ = PR ∧ height_is_constant P QR ∧ n = 9 ∧ small_triangle_area = 1 ∧ areaPQR = 50 →
  trapezoid_area MSNR = 3850 / 81 :=
by
  sorry

end trapezoid_MSNR_area_l299_299651


namespace halve_population_time_l299_299464

/-
Define the conditions:
- Population of the country: 5000
- Emigration rate per 500 persons: 50.4
- Immigration rate per 500 persons: 15.4
- Net emigration rate per 500 persons (calculated from emigration and immigration)
- Calculation of time for population to halve based on net emigration rate
-/

def population : ℕ := 5000
def emigration_rate_per_500 : ℚ := 50.4
def immigration_rate_per_500 : ℚ := 15.4

def net_emigration_rate_per_500 : ℚ := emigration_rate_per_500 - immigration_rate_per_500

def net_emigration_rate : ℚ := net_emigration_rate_per_500 * (population / 500)

def final_population : ℕ := population / 2

noncomputable def time_to_half_population : ℚ :=
  (population - final_population) / net_emigration_rate

theorem halve_population_time :
  time_to_half_population ≈ 7.14 :=
by
  unfold time_to_half_population
  have h1 : (population / 2 : ℚ) = 2500 := by norm_num
  have h2 : net_emigration_rate = 350 := by
    unfold net_emigration_rate net_emigration_rate_per_500
    norm_num
  rw [h1, h2]
  norm_num
  sorry

end halve_population_time_l299_299464


namespace inequality_proof_l299_299332

noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

def a : ℕ → ℝ
| 0 := 1 / 2
| 1 := 3 / 4
| (n + 2) := f (a n) + f (a (n + 1))

theorem inequality_proof (n : ℕ) (hn : 0 < n) : 
  f (3 * 2 ^ (n - 1)) ≤ a (2 * n) ∧ a (2 * n) ≤ f (3 * 2 ^ (2 * n - 2)) := 
begin
  sorry
end

end inequality_proof_l299_299332


namespace product_positive_probability_l299_299043

theorem product_positive_probability : 
  let S := {-9, -1, 3, 6, -4}
  in (∃ A B ∈ S, A * B > 0 ∧ 
      ∃ (m n : ℕ),
      m = {x ∈ S | x < 0}.to_finset.card ∧ 
      n = {x ∈ S | x > 0}.to_finset.card ∧ 
      [choose m 2 + choose n 2, choose (m + n) 2] = [3, 10] ∧ 
      (choose m 2 + choose n 2) * 5 = (choose (m + n) 2) * 2 ) :=
begin
  sorry
end

end product_positive_probability_l299_299043


namespace proof_problem_l299_299927

-- Definition of the propositions based on the conditions
def prop1 : Prop := ∀ n : ℤ, (n % 10 = 0 ∨ n % 10 = 2 ∨ n % 10 = 4 ∨ n % 10 = 6 ∨ n % 10 = 8) → 2 ∣ n

def prop2 : Prop := ∃ x : rhombus, square x

def prop3 : Prop := ∃ x : ℝ, x > 0

def prop4 : Prop := ∀ x : ℝ, ∃ k : ℤ, 2 * x + 1 = 2 * k + 1

-- The theorem to prove that propositions 2 and 3 are existential
theorem proof_problem : prop2 ∧ prop3 :=
by 
  sorry

end proof_problem_l299_299927


namespace largest_and_smallest_A_l299_299841

noncomputable def is_coprime_with_12 (n : ℕ) : Prop := 
  Nat.gcd n 12 = 1

def problem_statement (A_max A_min : ℕ) : Prop :=
  ∃ B : ℕ, B > 44444444 ∧ is_coprime_with_12 B ∧
  (A_max = 9 * 10^7 + (B - 9) / 10) ∧
  (A_min = 1 * 10^7 + (B - 1) / 10)

theorem largest_and_smallest_A :
  problem_statement 99999998 14444446 := sorry

end largest_and_smallest_A_l299_299841


namespace number_of_ways_to_form_team_l299_299993

theorem number_of_ways_to_form_team (boys girls : ℕ) (select_boys select_girls : ℕ)
    (H_boys : boys = 7) (H_girls : girls = 9) (H_select_boys : select_boys = 2) (H_select_girls : select_girls = 3) :
    (Nat.choose boys select_boys) * (Nat.choose girls select_girls) = 1764 := by
  rw [H_boys, H_girls, H_select_boys, H_select_girls]
  sorry

end number_of_ways_to_form_team_l299_299993


namespace expression_value_l299_299573

theorem expression_value (a b c : ℕ) (h1 : 25^a * 5^(2*b) = 5^6) (h2 : 4^b / 4^c = 4) : a^2 + a * b + 3 * c = 6 := by
  sorry

end expression_value_l299_299573


namespace line_intersects_circle_l299_299150

noncomputable def circle_center := (3 : ℝ, -5 : ℝ)
noncomputable def circle_radius := 6 : ℝ

def line (x y : ℝ) := 4 * x - 3 * y - 2 = 0

def distance (x y : ℝ) : ℝ :=
  abs (4 * x - 3 * y - 2) / real.sqrt (4^2 + (-3)^2)

theorem line_intersects_circle : distance 3 (-5) < circle_radius :=
by {
  -- Let the distance formula be simplified
  let d := distance 3 (-5),
  -- Calculate distance
  have distance_calc : d = abs (4 * 3 - 3 * (-5) - 2) / real.sqrt (4^2 + (-3)^2),
  -- Simplify the expressions and distances
  calc
    d = |4 * 3 - 3 * (-5) - 2| / real.sqrt (4^2 + (-3)^2) : by sorry
    ... = 25 / 5 : by sorry
    ... = 5 : by sorry,
  -- End by showing 5 < 6
  show 5 < 6, from by sorry
}

end line_intersects_circle_l299_299150


namespace class_overall_score_l299_299485

def max_score : ℝ := 100
def percentage_study : ℝ := 0.4
def percentage_hygiene : ℝ := 0.25
def percentage_discipline : ℝ := 0.25
def percentage_activity : ℝ := 0.1

def score_study : ℝ := 85
def score_hygiene : ℝ := 90
def score_discipline : ℝ := 80
def score_activity : ℝ := 75

theorem class_overall_score :
  (score_study * percentage_study) +
  (score_hygiene * percentage_hygiene) +
  (score_discipline * percentage_discipline) +
  (score_activity * percentage_activity) = 84 :=
  by sorry

end class_overall_score_l299_299485


namespace sigma_sum_inequality_l299_299360

theorem sigma_sum_inequality (n : ℕ) (hn : 0 < n) : 
  ∑ i in Finset.range n, (Nat.sigma i.succ) / i.succ ≤ 2 * n :=
sorry

end sigma_sum_inequality_l299_299360


namespace direct_proportion_conditions_l299_299914

theorem direct_proportion_conditions (k b : ℝ) : 
  (y = (k - 4) * x + b → (k ≠ 4 ∧ b = 0)) ∧ ¬ (b ≠ 0 ∨ k ≠ 4) :=
sorry

end direct_proportion_conditions_l299_299914


namespace total_cost_of_shirts_l299_299678

theorem total_cost_of_shirts 
    (first_shirt_cost : ℤ)
    (second_shirt_cost : ℤ)
    (h1 : first_shirt_cost = 15)
    (h2 : first_shirt_cost = second_shirt_cost + 6) : 
    first_shirt_cost + second_shirt_cost = 24 := 
by
  sorry

end total_cost_of_shirts_l299_299678


namespace min_value_S1_4S2_l299_299932

def parabola : set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y^2 = x }
def focus : ℝ × ℝ := (1/4, 0)
def on_opposite_sides (A B : ℝ × ℝ) : Prop := A.2 * B.2 < 0  -- opposite sides of x-axis

theorem min_value_S1_4S2 (A B : ℝ × ℝ) (S1 S2 : ℝ)
  (hA : A ∈ parabola) (hB : B ∈ parabola)
  (h_op : on_opposite_sides A B)
  (dot_prod : A.1 * B.1 + A.2 * B.2 = 6)
  (S1_def : S1 = 1/2 * 3 * (A.2 - B.2))
  (S2_def : S2 = 1/2 * 1/4 * A.2) :
  S1 + 4 * S2 = 6 :=
sorry

end min_value_S1_4S2_l299_299932


namespace sqrt7_add_2_l299_299441

theorem sqrt7_add_2 (a b : ℝ) (h1 : a ∈ ℤ) (h2 : 0 < b) (h3 : b < 1) (h4 : sqrt 7 + 2 = a + b) : a - b = 6 - sqrt 7 :=
by sorry

end sqrt7_add_2_l299_299441


namespace balls_not_adjacent_probability_l299_299763

theorem balls_not_adjacent_probability : 
  (let outcomes := (Finset.pairs (Finset.range 8)).card in
  let adjacent_pairs := [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)] in
  let total_outcomes := outcomes in
  let adjacent_outcomes := adjacent_pairs.length in
  let non_adjacent_outcomes := total_outcomes - adjacent_outcomes in
  (non_adjacent_outcomes.to_rat / total_outcomes.to_rat) = 3 / 4) :=
by
  sorry

end balls_not_adjacent_probability_l299_299763


namespace prob_A_B_same_room_l299_299541

open Probability

theorem prob_A_B_same_room :
  let boys := {A, B, C, D, E}
  let rooms := {room1, room2, room3}
  (∃! (boys_assign : boys → rooms), ∀ b ∈ boys, ∃! r ∈ rooms, boys_assign b = r) →
  let total_arrangements := 90
  let favorable_arrangements := 18
  (favorable_arrangements / total_arrangements) = (1 / 5) :=
by
  sorry

end prob_A_B_same_room_l299_299541


namespace coeff_x2_expansion_l299_299004

theorem coeff_x2_expansion : 
  (let f := (1 + (1/x^2)) * (1 + x)^6 in 
  (f.coeff 2) = 30) :=
sorry

end coeff_x2_expansion_l299_299004


namespace monotonic_increasing_interval_l299_299400

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem monotonic_increasing_interval :
  ∃ (a b : ℝ), (2 < a ∧ b = ∞) ∧ ∀ x, 2 < x → (f' x) > 0 :=
by
  let f' (x : ℝ) : ℝ := (x - 2) * Real.exp x
  have h : ∀ x, 2 < x → f' x > 0 := sorry
  exact ⟨2, ∞, ⟨by trivial, rfl⟩, h⟩

end monotonic_increasing_interval_l299_299400


namespace arrangements_without_A_at_head_l299_299374

theorem arrangements_without_A_at_head :
  let people := {A, B, C, D, E}
  let total_arrangements := 5 * 4 * 3
  let arrangements_with_A_at_head := 4 * 3
  total_arrangements - arrangements_with_A_at_head = 48 :=
by
  -- Definition of sets and variables
  let people := {A, B, C, D, E}
  let total_arrangements := 5 * 4 * 3
  let arrangements_with_A_at_head := 4 * 3
  -- Compute the desired result
  have h : total_arrangements - arrangements_with_A_at_head = 48 := by rfl
  exact h

end arrangements_without_A_at_head_l299_299374


namespace main_problem_l299_299749

/-- Define the odd function f passing through A and B --/
def odd_function (a b c x : ℝ) : Prop := 
  ∀ x : ℝ, f (x) = ax^3 + bx^2 + cx ∧ f (-x) = -f (x) ∧ (b = 0) ∧ (f (-sqrt 2) = sqrt 2) ∧ (f 2sqrt 2 = 10sqrt 2)

/-- Define the expression for f(x) --/
def expression_f_x (f : ℝ → ℝ) : Prop :=
  f = λ x, x^3 - 3x

/-- Define the intervals of monotonicity --/
def intervals_of_monotonicity (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x) = x^3 - 3x) ∧ 
  (∀ x ∈ Ioo (-(1 : ℝ)) (1), deriv f x < 0) ∧ 
  (∀ x ∈ Iio (-(1 : ℝ)) ∪ Ioi (1), deriv f x > 0)

/-- Define the range of m for three distinct roots --/
def range_of_m (f : ℝ → ℝ) : Prop :=
  (f(-1) = 2) ∧ (f(1) = -2) ∧ (∀ m : ℝ, -2 < m ∧ m < 2 → (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ f x1 = -m ∧ f x2 = -m ∧ f x3 = -m))

/-- The main theorem combining all parts --/
theorem main_problem (a b c : ℝ) :
  odd_function a b c x →
  (∃ f : ℝ → ℝ, expression_f_x f ∧ intervals_of_monotonicity f ∧ range_of_m f) :=
sorry

end main_problem_l299_299749


namespace cole_drive_time_l299_299527

noncomputable def time_to_drive_to_work (D : ℝ) : ℝ :=
  D / 50

theorem cole_drive_time (D : ℝ) (h₁ : time_to_drive_to_work D + (D / 110) = 2) : time_to_drive_to_work D * 60 = 82.5 :=
by
  sorry

end cole_drive_time_l299_299527


namespace wristbands_per_person_l299_299988

theorem wristbands_per_person (total_wristbands : ℕ) (total_spectators : ℕ)
    (h1 : total_wristbands = 290) (h2 : total_spectators = 145) :
    (total_wristbands / total_spectators) = 2 := 
by
  rw [h1, h2]
  norm_num
  sorry

end wristbands_per_person_l299_299988


namespace cube_vertex_label_sum_impossible_l299_299889

theorem cube_vertex_label_sum_impossible :
  ∀ (cube_labels : Fin 8 → Finset ℕ), (∀ v, v ∈ cube_labels) →
  (∀ f : Fin 6, ∃ S : ℕ, (∑ v in cube_labels.filter (λ v, v.faces.contains f), v.val) = S) →
  ¬ ∃ arrangement : Fin 8, (cube_labels ∪ arrangement) :=
by
  sorry

end cube_vertex_label_sum_impossible_l299_299889


namespace white_squares_in_42nd_row_l299_299080

theorem white_squares_in_42nd_row :
  let white_start (n : ℕ) := ∀ k < n, if k % 2 = 0 then is_white k else is_black k
  let pattern (row n : ℕ) := 
    if n = 1 then 1 
    else pattern row (n - 1) + 2
  in
  white_start (42 : ℕ) ∧ 
  alternation_in_rows ∧
  pattern ∧ 
  (let N := (42 * 2) - 1
  in ((N + 1) / 2 = 42)) :=
sorry

end white_squares_in_42nd_row_l299_299080


namespace bill_age_l299_299118

theorem bill_age (C : ℕ) (h1 : ∀ B : ℕ, B = 2 * C - 1) (h2 : C + (2 * C - 1) = 26) : 
  ∃ B : ℕ, B = 17 := 
by
  sorry

end bill_age_l299_299118


namespace midpoints_on_straight_line_l299_299718

-- Define the standard form of a hyperbola
def hyperbola (a b x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the equation of a chord, where m is the slope and c is the y-intercept
def chord (m c x y : ℝ) : Prop := y = m * x + c

-- Given that the chords are parallel, they have the same slope m.
-- We need to prove that the midpoints of these chords lie on a straight line.
theorem midpoints_on_straight_line (a b m : ℝ) (hm : m ≠ 0) :
  ∃ k k' : ℝ, ∀ c : ℝ, ∀ x1 x2 y1 y2 : ℝ,
  hyperbola a b x1 y1 →
  hyperbola a b x2 y2 →
  chord m c x1 y1 →
  chord m c x2 y2 →
  let xm := (x1 + x2) / 2,
  xm = k * c + k' :=
sorry

end midpoints_on_straight_line_l299_299718


namespace arithmetic_geometric_mean_inequality_l299_299934

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : a < 0) (hb : b < 0) (hab : a ≠ b) :
  let A := (a + b) / 2
  let G := sqrt (a * b)
  in A < G :=
by
  sorry

end arithmetic_geometric_mean_inequality_l299_299934


namespace solve_inequality_smallest_integer_solution_l299_299728

theorem solve_inequality (x : ℝ) : 
    (9 * x + 8) / 6 - x / 3 ≥ -1 ↔ x ≥ -2 := 
sorry

theorem smallest_integer_solution :
    ∃ (x : ℤ), (∃ (y : ℝ) (h₁ : y = x), 
    (9 * y + 8) / 6 - y / 3 ≥ -1) ∧ 
    ∀ (z : ℤ), ((∃ (w : ℝ) (h₂ : w = z), 
    (9 * w + 8) / 6 - w / 3 ≥ -1) → -2 ≤ z) :=
    ⟨-2, __, sorry⟩

end solve_inequality_smallest_integer_solution_l299_299728


namespace smallest_five_digit_number_divisibility_l299_299781

-- Define the smallest 5-digit number satisfying the conditions
def isDivisibleBy (n m : ℕ) : Prop := m ∣ n

theorem smallest_five_digit_number_divisibility :
  ∃ (n : ℕ), isDivisibleBy n 15
          ∧ isDivisibleBy n (2^8)
          ∧ isDivisibleBy n 45
          ∧ isDivisibleBy n 54
          ∧ n >= 10000
          ∧ n < 100000
          ∧ n = 69120 :=
sorry

end smallest_five_digit_number_divisibility_l299_299781


namespace max_n_for_factored_poly_l299_299559

theorem max_n_for_factored_poly : 
  ∃ (n : ℤ), (∀ (A B : ℤ), 2 * B + A = n → A * B = 50) ∧ 
            (∀ (m : ℤ), (∀ (A B : ℤ), 2 * B + A = m → A * B = 50) → m ≤ 101) ∧ 
            n = 101 :=
by
  sorry

end max_n_for_factored_poly_l299_299559


namespace solution_set_l299_299694

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}

-- Given conditions
axiom differentiable_f : Differentiable ℝ f
axiom f_deriv : ∀ x, deriv f x = f' x
axiom f_at_3 : f 3 = 1
axiom inequality : ∀ x, 3 * f x + x * f' x > 1

-- Goal to prove
theorem solution_set :
  {x : ℝ | (x - 2017) ^ 3 * f (x - 2017) - 27 > 0} = {x | 2020 < x} :=
  sorry

end solution_set_l299_299694


namespace ratio_a_over_b_l299_299217

-- Definitions of conditions
def func (a b x : ℝ) : ℝ := a * x^2 + b
def derivative (a b x : ℝ) : ℝ := 2 * a * x

-- Given conditions
variables (a b : ℝ)
axiom tangent_slope : derivative a b 1 = 2
axiom point_on_graph : func a b 1 = 3

-- Statement to prove
theorem ratio_a_over_b : a / b = 1 / 2 :=
by sorry

end ratio_a_over_b_l299_299217


namespace tangent_line_iff_l299_299945

theorem tangent_line_iff (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 8 * y + 12 = 0 → ax + y + 2 * a = 0) ↔ a = -3 / 4 :=
by
  sorry

end tangent_line_iff_l299_299945


namespace tensor_sum_l299_299880

def tensor (m n : ℕ) : ℕ := m * m - n * n

theorem tensor_sum :
  (2.tensor 4) - (4.tensor 6) - (6.tensor 8) - (8.tensor 10) 
  - ⋯ - (96.tensor 98) - (98.tensor 100) = 9972 := 
sorry

end tensor_sum_l299_299880


namespace total_cost_of_shirts_is_24_l299_299683

-- Definitions based on conditions
def cost_first_shirt : ℕ := 15
def cost_difference : ℕ := 6
def cost_second_shirt : ℕ := cost_first_shirt - cost_difference

-- The proof problem statement: Calculate total cost given the conditions
theorem total_cost_of_shirts_is_24 : cost_first_shirt + cost_second_shshirt = 24 :=
by
  sorry

end total_cost_of_shirts_is_24_l299_299683


namespace positive_rational_achievable_l299_299827

open Set Finset

noncomputable theory

variable {A : Set ℕ} (h1 : ∀ n ∈ A, 2 * n ∈ A)
                             (h2 : ∀ (n : ℕ), ∃ m ∈ A, m % n = 0)
                             (h3 : ∀ (C : ℝ), C > 0 → ∃ B : Finset ℕ, (B ⊆ A.toFinset ∧ ∑ x in B, (1 / (x : ℝ)) > C))

theorem positive_rational_achievable :
  ∀ r : ℚ, r > 0 → ∃ B : Finset ℕ, (B ⊆ A.toFinset ∧ (∑ x in B, (1 / (x : ℝ)) = (r : ℝ))) :=
by
  sorry

end positive_rational_achievable_l299_299827


namespace solve_quadratic_equation_l299_299416

theorem solve_quadratic_equation (x : ℝ) : x^2 - 2 * x = 0 ↔ x = 0 ∨ x = 2 :=
by
  sorry

end solve_quadratic_equation_l299_299416


namespace triangle_AB_is_correct_l299_299851

theorem triangle_AB_is_correct
  (angle_B : ℕ) (angle_C : ℕ) (angle_BDC : ℕ)
  (BD CD : ℝ) :
  angle_B = 30 ∧ angle_C = 45 ∧ angle_BDC = 150 ∧ BD = 5 ∧ CD = 5 →
  let AB := 5 * Real.sqrt 3 in
  AB = 5 * Real.sqrt 3 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  exact Eq.refl (5 * Real.sqrt 3)

end triangle_AB_is_correct_l299_299851


namespace count_valid_n_l299_299249

theorem count_valid_n :
  ∃ (S : Finset ℕ), (∀ n ∈ S, 300 < n^2 ∧ n^2 < 1200 ∧ n % 3 = 0) ∧
                     S.card = 6 := sorry

end count_valid_n_l299_299249


namespace length_PQ_l299_299660

variables (P Q R S : Type)
variables [DecidableEq P] [DecidableEq Q] [DecidableEq R] [DecidableEq S]

structure Triangle (A B C : Type) :=
  (is_isosceles : Bool)
  (perimeter : ℝ)

variables
  (T1 : Triangle P Q R)
  (T2 : Triangle Q R S)
  (QR_length : ℝ)

theorem length_PQ (h1 : T1.is_isosceles)
    (h2 : T2.is_isosceles)
    (h3 : T2.perimeter = 24)
    (h4 : T1.perimeter = 27)
    (QR_length = 10) :
    let PQ_length := 8.5 in PQ_length = 8.5 :=
by
  sorry

end length_PQ_l299_299660


namespace log_sum_equality_l299_299124

noncomputable def log_base_5 (x : ℝ) := Real.log x / Real.log 5

theorem log_sum_equality :
  2 * log_base_5 10 + log_base_5 0.25 = 2 :=
by
  sorry -- proof goes here

end log_sum_equality_l299_299124


namespace contest_score_difference_l299_299275

noncomputable def percentage_scores : ℕ → ℕ → ℚ
| total_students score_60 => 0.20
| total_students score_75 => 0.40
| total_students score_85 => 0.25
| total_students score_95 => 0.15

noncomputable def mean_score (total_students : ℕ) : ℚ := 
  0.20 * 60 + 0.40 * 75 + 0.25 * 85 + 0.15 * 95

noncomputable def median_score (total_students : ℕ) : ℚ := 
  -- 50% of students score below 75 and 25% of students score exactly 75
  75

def score_difference (total_students : ℕ) : ℚ :=
  median_score total_students - mean_score total_students

theorem contest_score_difference : ∀ (total_students : ℕ), score_difference total_students = -2.5 :=
begin
  intro total_students,
  unfold score_difference,
  unfold mean_score,
  unfold median_score,
  norm_num
end

end contest_score_difference_l299_299275


namespace monotonically_increasing_interval_l299_299401

def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x < y → f x < f y

noncomputable def log_base_3 (x : ℝ) := log x / log 3

def f (x : ℝ) := log_base_3 (x^2 - 2 * x)

theorem monotonically_increasing_interval :
  is_monotonically_increasing f (set.Ioi 2) :=
sorry

end monotonically_increasing_interval_l299_299401


namespace floor_sqrt_24_squared_l299_299893

theorem floor_sqrt_24_squared : (⌊Real.sqrt 24⌋ : ℕ)^2 = 16 :=
by
  sorry

end floor_sqrt_24_squared_l299_299893


namespace shirts_total_cost_l299_299679

def shirt_cost_problem : Prop :=
  ∃ (first_shirt_cost second_shirt_cost total_cost : ℕ),
    first_shirt_cost = 15 ∧
    first_shirt_cost = second_shirt_cost + 6 ∧
    total_cost = first_shirt_cost + second_shirt_cost ∧
    total_cost = 24

theorem shirts_total_cost : shirt_cost_problem := by
  sorry

end shirts_total_cost_l299_299679


namespace find_number_of_machines_l299_299160

noncomputable def number_of_machines 
  (total_money : ℝ) 
  (quarters_per_machine : ℕ) 
  (quarter_value : ℝ) 
  (dimes_per_machine : ℕ) 
  (dime_value : ℝ) 
  (money_per_machine : ℝ) : ℕ := 
  total_money / money_per_machine

theorem find_number_of_machines 
  (total_money : ℝ := 90) 
  (quarters_per_machine : ℕ := 80) 
  (quarter_value : ℝ := 0.25) 
  (dimes_per_machine : ℕ := 100) 
  (dime_value : ℝ := 0.10) 
  (money_per_machine : ℝ := (quarters_per_machine * quarter_value) + (dimes_per_machine * dime_value)) : 
  number_of_machines total_money quarters_per_machine quarter_value dimes_per_machine dime_value money_per_machine = 3 := by 
  sorry

end find_number_of_machines_l299_299160


namespace find_tan_phi_l299_299325

open Matrix

noncomputable theory

variables (l : ℝ) (φ : ℝ) (h_l_pos : l > 0)

def D : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![l, 0], ![0, l]]

def R : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos φ,  Real.sin φ], ![-Real.sin φ, Real.cos φ]]

theorem find_tan_phi (hRD : R l φ ⬝ D l = ![![9, 12], ![-12, 9]]) : Real.tan φ = 4 / 3 := 
sorry

end find_tan_phi_l299_299325


namespace trajectory_eqn_exists_line_with_conditions_l299_299241

noncomputable def C₁ : ℝ × ℝ := (0, 1)
noncomputable def C₂ : ℝ × ℝ := (0, -1)

def equation_circle_C₁ (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1
def equation_circle_C₂ (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4
def slope_P_C₁ (x y : ℝ) : ℝ := (y - 1) / x
def slope_P_C₂ (x y : ℝ) : ℝ := (y + 1) / x

-- Prove the trajectory equation
theorem trajectory_eqn (x y : ℝ) (h1 : x ≠ 0)
  (cond : slope_P_C₁ x y * slope_P_C₂ x y = -1/2) : 
  x^2 / 2 + y^2 = 1 :=
sorry

-- Prove existence of line l that intersects the trajectory M at two distinct points 
-- and satisfies the given conditions about distances
theorem exists_line_with_conditions (C₁ : ℝ × ℝ := (0, 1)) :
  ∃ l : ℝ × ℝ → Prop, 
    (l (2, 0)) ∧
    ∀ C D : ℝ × ℝ, 
      (C ≠ D) → (l C) → (l D) →
      (equation_circle_C₁ C.1 C.2 ∧ equation_circle_C₁ D.1 D.2) →
      |C.1 - C₁.1| = |D.1 - C₁.1| :=
sorry


end trajectory_eqn_exists_line_with_conditions_l299_299241


namespace focus_of_parabola_l299_299393

theorem focus_of_parabola (x y : ℝ) : (x^2 + y = 0) → (0, -1/4) ∈ set_of_focus_of_parabola :=
by
  sorry

end focus_of_parabola_l299_299393


namespace billy_buys_bottle_l299_299867

-- Definitions of costs and volumes
def money : ℝ := 10
def cost1 : ℝ := 1
def volume1 : ℝ := 10
def cost2 : ℝ := 2
def volume2 : ℝ := 16
def cost3 : ℝ := 2.5
def volume3 : ℝ := 25
def cost4 : ℝ := 5
def volume4 : ℝ := 50
def cost5 : ℝ := 10
def volume5 : ℝ := 200

-- Statement of the proof problem
theorem billy_buys_bottle : ∃ b : ℕ, b = 1 ∧ cost5 = money := by 
  sorry

end billy_buys_bottle_l299_299867


namespace sum_inequality_l299_299916

open Real

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ (a * b / (a + b)) + (b * c / (b + c)) + (c * a / (c + a)) + 
             (1 / 2) * ((a * b / c) + (b * c / a) + (c * a / b)) :=
by
  sorry

end sum_inequality_l299_299916


namespace area_intersection_not_covered_by_triangle_l299_299498

theorem area_intersection_not_covered_by_triangle :
  let rect_length := 10
  let rect_width := 3
  let circle_radius := 5
  let tri_leg_1 := 6
  let tri_leg_2 := 8
  let tri_area := (1/2) * tri_leg_1 * tri_leg_2
in
(6 * pi - tri_area = 6 * pi - 24) :=
by
  sorry

end area_intersection_not_covered_by_triangle_l299_299498


namespace courtyard_paving_l299_299089

noncomputable def length_of_brick (L : ℕ) := L = 12

theorem courtyard_paving  (courtyard_length : ℕ) (courtyard_width : ℕ) 
                           (brick_width : ℕ) (total_bricks : ℕ) 
                           (H1 : courtyard_length = 18) (H2 : courtyard_width = 12) 
                           (H3 : brick_width = 6) (H4 : total_bricks = 30000) 
                           : length_of_brick 12 := 
by 
  sorry

end courtyard_paving_l299_299089


namespace find_omega_find_zeros_in_interval_l299_299225

noncomputable def f (ω : ℝ) (x : ℝ) := 2 * (Real.sin (ω * x - π / 6)) * (Real.cos (ω * x)) + 1 / 2

noncomputable def h (x : ℝ) := Real.sin (2 * x + π / 6)

noncomputable def g (x : ℝ) := Real.sin (x + π / 6)

theorem find_omega (ω : ℝ) (h₁ : ω > 0) (h₂ : ∀ x, f ω x = f ω (x + π)) : ω = 1 := 
sorry

theorem find_zeros_in_interval :
  let f' := λ x : ℝ, Real.sin (x + π / 6) in
  ∃ x ∈ Set.Icc (-π) π, f' x = 0 ∧
  ((x = -π / 6) ∨ (x = 5 * π / 6)) :=
sorry

end find_omega_find_zeros_in_interval_l299_299225


namespace solve_for_k_l299_299467

theorem solve_for_k :
  (∀ x : ℤ, (2 * x + 4 = 4 * (x - 2)) ↔ ( -x + 17 = 2 * x - 1 )) :=
by
  sorry

end solve_for_k_l299_299467


namespace problem_statement_l299_299741

-- Conditions translating directly from the problem:
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Definition of the base case for the factorial function
lemma factorial_15 : factorial 15 = 1307674368000 :=
  sorry

-- Representation condition for the ending digits of 15!
def ends_in_pattern (n : ℕ) (pattern : ℕ) : Prop :=
  ∃ k, n = k * 10^(pattern.digits / 10)

-- Lean statement for the mathematically equivalent problem
theorem problem_statement (X Y Z : ℕ) :
  (∃ Z, Z = 0 ∧ (factorial 15).digits = XXXYYY030000) →
  (X + Y + Z = 3) :=
begin
  sorry
end

end problem_statement_l299_299741


namespace find_x_l299_299975

theorem find_x (x : ℚ) (h : (3 * x + 4) / 5 = 15) : x = 71 / 3 :=
by
  sorry

end find_x_l299_299975


namespace radical_axis_shared_l299_299580

theorem radical_axis_shared (ABC : Triangle)
  (O1 : Circle) (HO1 : O1.is_circumcircle ABC)
  (O2 : Circle) (HO2 : O2.is_nine_point_circle ABC)
  (O3 : Circle) (H : Point) (G : Point) 
  (H_ortho : H.is_orthocenter ABC) (G_centroid : G.is_centroid ABC)
  (HO3 : O3 = Circle.diameter H G) : 
  have share_common_radical_axis : ∀ x : Point, x ∈ O1.radical_axis O2 → x ∈ O1.radical_axis O3 := sorry
  have all_radical_axes_equal : O1.radical_axis O2 = O1.radical_axis O3 := sorry
  in true := sorry 

end radical_axis_shared_l299_299580


namespace inequality_smallest_integer_solution_l299_299731

theorem inequality_smallest_integer_solution (x : ℤ) :
    (9 * x + 8) / 6 - x / 3 ≥ -1 → x ≥ -2 := sorry

end inequality_smallest_integer_solution_l299_299731


namespace constant_term_in_binomial_expansion_l299_299191

noncomputable def a : ℝ := ∫ x in -1..1, 5 * (x ^ (2 / 3))

theorem constant_term_in_binomial_expansion :
  let binomial_term := (Real.sqrt t - a / (6 * t)) ^ a in
  a = 6 → -- Since we derive a = 6 from the integral, it's included as a condition
  -- Solve: constant term is 15
  ((∑ r in Finset.range 7, Binomial.coeff 6 r * (Real.sqrt t)^(6-r) * (-a/(6 * t))^r) = 15) :=
begin
  sorry
end

end constant_term_in_binomial_expansion_l299_299191


namespace parabola_line_intersection_sum_x_l299_299095

theorem parabola_line_intersection_sum_x 
  (x1 x2 y1 y2 : ℝ)
  (h_intersect : ∃ l m, ∀ p, y = l * p + m → p ≠ 0)
  (h_parabola_A : y1^2 = 4 * x1)
  (h_parabola_B : y2^2 = 4 * x2)
  (h_focus_line : ∃ l, ∀ x, (l * x = (x1 + x2) /2))
  (h_AB : | AB | = 12) 
  : x1 + x2 = 10 := 
sorry

end parabola_line_intersection_sum_x_l299_299095


namespace new_cost_after_decrease_l299_299109

theorem new_cost_after_decrease (c : ℝ) (d : ℝ) (new_cost : ℝ) : 
  c = 10000.000000000002 →
  d = 0.56 →
  new_cost = 10000.000000000002 - 0.56 * 10000.000000000002 →
  new_cost = 4400.000000000001 :=
by
  intros hc hd hnew_cost
  have hc' : 10000.000000000002 - 0.56 * 10000.000000000002 = 4400.000000000001,
    by norm_num,
  rw [hc, hd] at hnew_cost,
  exact hnew_cost.trans hc'

end new_cost_after_decrease_l299_299109


namespace union_of_sets_l299_299702

-- Define the sets P and Q and the condition P ∩ Q = {0}
variable {a b : ℝ}
def P := {3, real.log 2 a}
def Q := {a, b}
def P_inter_Q := P ∩ Q = {0}
def P_union_Q := P ∪ Q

theorem union_of_sets (ha : real.log 2 a = 0) (hb : b = 0) : P_union_Q = {3, 0, 1} := 
  sorry

end union_of_sets_l299_299702


namespace cherries_initially_l299_299354

theorem cherries_initially (x : ℕ) (h₁ : x - 6 = 10) : x = 16 :=
by
  sorry

end cherries_initially_l299_299354


namespace shooting_star_trace_line_l299_299264

-- Mathematical definition: observing a shooting star
def shooting_star_observation (image: Type) : Prop :=
  ∃ (p : point) (l : line), p ∈ l

-- The problem statement we need to prove
theorem shooting_star_trace_line : shooting_star_observation (trace_image) := 
begin
  -- proof goes here
  sorry
end

end shooting_star_trace_line_l299_299264


namespace candy_bars_per_person_l299_299761

theorem candy_bars_per_person 
    (total_candy_bars : ℝ) 
    (number_of_people : ℝ)
    (h_total_candy_bars : total_candy_bars = 27.5) 
    (h_number_of_people : number_of_people = 8.3) : 
    total_candy_bars / number_of_people ≈ 3.313 :=
by sorry

end candy_bars_per_person_l299_299761


namespace cubic_yards_to_cubic_feet_l299_299622

theorem cubic_yards_to_cubic_feet :
  (5:ℚ) * ((1:ℚ) * (3:ℚ)^3) = 135 := 
by
  have h1 : (3:ℚ)^3 = 27 := by norm_num
  rw [←h1]
  norm_num

end cubic_yards_to_cubic_feet_l299_299622


namespace original_paint_intensity_l299_299378

theorem original_paint_intensity (I : ℝ)
  (H1 : let replaced_fraction := (2 : ℝ) / 3)
  (H2 : let replaced_solution := 20)
  (H3 : let new_mixture_intensity := 30)
  (H4 : (1-replaced_fraction) * I + replaced_fraction * replaced_solution = new_mixture_intensity) :
  I = 50 :=
sorry

end original_paint_intensity_l299_299378


namespace lunch_cost_is_1036_l299_299345

/-- Define the number of classes and number of students per class. -/
def thirdGradeClasses := 5
def studentsPerThirdGradeClass := 30

def fourthGradeClasses := 4
def studentsPerFourthGradeClass := 28

def fifthGradeClasses := 4
def studentsPerFifthGradeClass := 27

/-- Define the cost of food items for each student. -/
def costOfHamburger := 2.10
def costOfCarrots := 0.50
def costOfCookie := 0.20

/-- Summing up the number of students in each grade. -/
def totalStudents := 
  (thirdGradeClasses * studentsPerThirdGradeClass) + 
  (fourthGradeClasses * studentsPerFourthGradeClass) + 
  (fifthGradeClasses * studentsPerFifthGradeClass)

/-- Calculating the cost of one student's lunch. -/
def costPerStudent :=
  costOfHamburger + costOfCarrots + costOfCookie

/-- Calculating the total cost for all students. -/
def totalCost :=
  totalStudents * costPerStudent

/-- The theorem we need to prove: the total cost is $1,036. -/
theorem lunch_cost_is_1036 : totalCost = 1036 := 
by 
  have h1 : totalStudents = 370 := by sorry
  have h2 : costPerStudent = 2.80 := by sorry
  show totalCost = 370 * 2.80 from sorry

end lunch_cost_is_1036_l299_299345


namespace imaginary_part_of_z_is_sqrt2_div2_l299_299640

open Complex

noncomputable def z : ℂ := abs (1 - I) / (1 - I)

theorem imaginary_part_of_z_is_sqrt2_div2 : z.im = Real.sqrt 2 / 2 := by
  sorry

end imaginary_part_of_z_is_sqrt2_div2_l299_299640


namespace set_union_l299_299930

noncomputable def A : Set ℝ := { x | 1/9 < (1/3)^x ∧ (1/3)^x < 3}
noncomputable def B : Set ℝ := { x | Real.log x / Real.log 2 > 0 }

theorem set_union :
  (A ∪ B) = {x | -2 < x} := 
by
  sorry

end set_union_l299_299930


namespace complex_expression_equality_l299_299162

-- Define the basic complex number properties and operations.
def i : ℂ := Complex.I -- Define the imaginary unit

theorem complex_expression_equality (a b : ℤ) :
  (3 - 4 * i) * ((-4 + 2 * i) ^ 2) = -28 - 96 * i :=
by
  -- Syntactical proof placeholders
  sorry

end complex_expression_equality_l299_299162


namespace range_of_a_l299_299577

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then a + |x - 2|
  else x^2 - 2 * a * x + 2 * a

theorem range_of_a (a : ℝ) (x : ℝ) (h : ∀ x : ℝ, f a x ≥ 0) : -1 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l299_299577


namespace arithmetic_sequence_sum_l299_299690

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℤ) :
  (∀ n, a n = a 1 + (n - 1) * d) → 
  (∀ n, S n = n * (a 1 + a n) / 2) → 
  (a 3 + 4 = a 2 + a 7) → 
  S 11 = 44 :=
by 
  sorry

end arithmetic_sequence_sum_l299_299690


namespace rectangle_area_inscribed_semicircle_l299_299719

theorem rectangle_area_inscribed_semicircle
  (MO MG KO : ℝ) 
  (hMO : MO = 20) 
  (hMG : MG = 12) 
  (hKO : KO = 12) 
  (diameter := MG + MO + KO) 
  (r := diameter / 2) 
  (NP := Math.sqrt (MG * KO)) : 
  MO * NP = 240 := 
by 
  sorry

end rectangle_area_inscribed_semicircle_l299_299719


namespace subset_problem_l299_299616

theorem subset_problem (m : ℝ) : ({3, 2 * m - 1} : set ℝ) ⊆ ({-1, 3, m ^ 2} : set ℝ) → (m = 0 ∨ m = 1) :=
by
  sorry

end subset_problem_l299_299616


namespace number_of_correct_props_l299_299239

-- Define the basic entities: Lines and Planes
variable {Line : Type} [OrderedField Line]
variable {Plane : Type} [OrderedField Plane]

-- Define the propositions according to the conditions
def prop1 (m n : Line) (α : Plane) : Prop := (m ∥ n ∧ n ⊂ α) → (m ∥ α)
def prop2 (l m : Line) (α β : Plane) : Prop := (l ⊥ α ∧ m ⊥ β ∧ l ⊥ m) → (α ⊥ β)
def prop3 (l m n : Line) : Prop := (l ⊥ n ∧ m ⊥ n) → (l ∥ m)
def prop4 (m n : Line) (α β : Plane) : Prop := (α ⊥ β ∧ α ∩ β = m ∧ n ⊂ β ∧ n ⊥ m) → (n ⊥ α)

-- The theorem to be proved: number of correct propositions is two
theorem number_of_correct_props (m n l : Line) (α β : Plane) :
  (prop2 l m α β) ∧ (prop4 m n α β) ∧ ¬(prop1 m n α) ∧ ¬(prop3 l m n) → 2 :=
sorry

end number_of_correct_props_l299_299239


namespace range_of_a_if_x_in_A_necessary_for_x_in_B_l299_299188

variable (a : ℝ)
def A := {x : ℝ | x ≥ a}
def B := {x : ℝ | |x - 1| < 1}

theorem range_of_a_if_x_in_A_necessary_for_x_in_B (h : (∀ x, x ∈ B → x ∈ A) ∧ (∃ x, x ∈ A ∧ x ∉ B)) : a ≤ 0 :=
sorry

end range_of_a_if_x_in_A_necessary_for_x_in_B_l299_299188


namespace total_path_length_point_A_l299_299853

theorem total_path_length_point_A 
  (triangle₁ u : Type)
  [metric_space triangle₁]
  [has_measure triangle₁]
  [metric_space u]
  [has_measure u]
  (circumference_point : triangle₁ → u)
  (center_point : triangle₁ → u)
  (angle_ABC : triangle₁ → ℝ)
  (rotation_count : ℕ)
  (α : ℝ)
  (A B C : triangle₁)
  (path_length : ℝ)
  (h_center : B = center_point B)
  (h_angle_ABC : angle_ABC B = 2 * α)
  (h_circumference_A : circumference_point A ≠ B)
  (h_circumference_C : circumference_point C ≠ B)
  (h_alpha_range : 0 < α ∧ α < π / 3)
  (rotate : triangle₁ → ℝ → triangle₁)
  (h_rotate_A : ∀ B : triangle₁, rotate B 2 = circumference_point B)
  (h_rotate_B : ∀ C : triangle₁, rotate C 1 = circumference_point C)
  (h_rotate_C : ∀ A : triangle₁, rotate A 1 = circumference_point A)
  (S : ℝ) :
  S = 22 * π * (1 + sin α) - 66 * α :=
sorry

end total_path_length_point_A_l299_299853


namespace hexagonal_pyramid_volume_lateral_surface_area_l299_299757

theorem hexagonal_pyramid_volume_lateral_surface_area
  (a : ℝ) 
  (h_base_hexagon : RegularHexagon base a) 
  (h_congruent_diagonals : congruent_diagonal_sections pyramid) :
  (∃ V A : ℝ, 
    V = (3 * a ^ 3) / 4 ∧ 
    A = (3 * a ^ 2 * Real.sqrt 6) / 2) :=
begin
  use [(3 * a ^ 3) / 4, (3 * a ^ 2 * Real.sqrt 6) / 2],
  split;
  sorry, -- Placeholder for the proofs
end

end hexagonal_pyramid_volume_lateral_surface_area_l299_299757


namespace greatest_possible_perimeter_l299_299994

theorem greatest_possible_perimeter (a b c : ℕ) 
    (h₁ : a = 4 * b ∨ b = 4 * a ∨ c = 4 * a ∨ c = 4 * b)
    (h₂ : a = 18 ∨ b = 18 ∨ c = 18)
    (triangle_ineq : a + b > c ∧ a + c > b ∧ b + c > a) :
    a + b + c = 43 :=
by {
  sorry
}

end greatest_possible_perimeter_l299_299994


namespace letters_by_30_typists_in_1_hour_l299_299254

-- Definitions from the conditions
def lettersTypedByOneTypistIn20Minutes := 44 / 20

def lettersTypedBy30TypistsIn20Minutes := 30 * (lettersTypedByOneTypistIn20Minutes)

def conversionToHours := 3

-- Theorem statement
theorem letters_by_30_typists_in_1_hour : lettersTypedBy30TypistsIn20Minutes * conversionToHours = 198 := by
  sorry

end letters_by_30_typists_in_1_hour_l299_299254


namespace smallest_four_digit_number_l299_299883

noncomputable def smallest_four_digit_solution : ℕ := 1011

theorem smallest_four_digit_number (x : ℕ) (h1 : 5 * x ≡ 25 [MOD 20]) (h2 : 3 * x + 10 ≡ 19 [MOD 7]) (h3 : x + 3 ≡ 2 * x [MOD 12]) :
  x = smallest_four_digit_solution :=
by
  sorry

end smallest_four_digit_number_l299_299883


namespace even_cycle_exists_l299_299798

variable (A : Type) [AddGroup A]
variable (n : ℕ)
variable (P : set A)
variable (η : P → P → Prop)

-- Given conditions:
variable (n_ge_4 : n ≥ 4)
variable (no_three_collinear : ∀ (A1 A2 A3 : A), A1 ≠ A2 → A2 ≠ A3 → A1 ≠ A3 → ¬ (η A1 A2 ∧ η A2 A3 ∧ η A3 A1))
variable (connected_to_at_least_three : ∀ (A : A), (∃ (B C D : A), η A B ∧ η A C ∧ η A D))

-- Question: Exists 2k distinct points such that they are cyclically connected
noncomputable def exists_even_cycle (P : set A) (η : P → P → Prop) : Prop :=
∃ (k : ℕ) (X : Fin (2 * k) → A), 1 < k ∧ ∀ (i : Fin (2 * k)), η (X i) (X (i + 1) % (2 * k))

-- Main theorem statement
theorem even_cycle_exists :
  ∃ (X : Fin n → A),
  (∀ (i : Fin n), η (X i) (X (i + 1) % n)) →
  exists_even_cycle P η :=
sorry


end even_cycle_exists_l299_299798


namespace reduced_price_per_kg_l299_299463

theorem reduced_price_per_kg (P R : ℝ) (hR : R = 0.85 * P)
  (hEq : 800 / R - 800 / P = 5) : R = 24 :=
begin
  sorry
end

end reduced_price_per_kg_l299_299463


namespace left_faces_dots_l299_299429

structure Cube where
  face_3_dots : ℕ
  face_2_dots : ℕ
  face_1_dots : ℕ

-- Define a structure for the problem setting
structure Configuration where
  cubes : List Cube
  is_П_shape : Prop
  adjacent_faces_match : Prop

-- Example initialization (not required in the statement, but can be helpful)
def example_cube := Cube.mk 1 2 3
def example_configuration := {
  cubes := List.replicate 7 example_cube,
  is_П_shape := true,
  adjacent_faces_match := true
}

theorem left_faces_dots (A B C : ℕ) (config : Configuration) :
  config.is_П_shape ∧ config.adjacent_faces_match ∧
  ∃ (c : Cube), (c.face_2_dots = A) ∧ (c.face_2_dots = B) ∧ (c.face_3_dots = C) → 
  A = 2 ∧ B = 2 ∧ C = 3 := 
by 
  sorry

end left_faces_dots_l299_299429


namespace sanity_proof_l299_299477

-- Define the characters and their sanity status as propositions
variables (Griffin QuasiTurtle Lobster : Prop)

-- Conditions
axiom Lobster_thinks : (Griffin ∧ ¬QuasiTurtle ∧ ¬Lobster) ∨ (¬Griffin ∧ QuasiTurtle ∧ ¬Lobster) ∨ (¬Griffin ∧ ¬QuasiTurtle ∧ Lobster)
axiom QuasiTurtle_thinks : Griffin

-- Statement to prove
theorem sanity_proof : ¬Griffin ∧ ¬QuasiTurtle ∧ ¬Lobster :=
by {
  sorry
}

end sanity_proof_l299_299477


namespace seven_digit_numbers_count_seven_digit_numbers_with_even_together_seven_digit_numbers_with_even_and_odd_together_seven_digit_numbers_no_two_even_adjacent_l299_299186

theorem seven_digit_numbers_count :
  (∑ b in choose (finset.filter even (finset.range 1 10)), 3),
  (∑ b in choose (finset.filter odd (finset.range 1 10)), 4),
  (7.factorial) = 100800 :=
sorry

theorem seven_digit_numbers_with_even_together :
  (∑ b in choose (finset.filter even (finset.range 1 10)), 3),
  (∑ b in choose (finset.filter odd (finset.range 1 10)), 4),
  (5.factorial) * (3.factorial) = 14400 :=
sorry

theorem seven_digit_numbers_with_even_and_odd_together :
  (∑ b in choose (finset.filter even (finset.range 1 10)), 3),
  (∑ b in choose (finset.filter odd (finset.range 1 10)), 4),
  (3.factorial) * (4.factorial) * (2.factorial) = 5760 :=
sorry

theorem seven_digit_numbers_no_two_even_adjacent :
  (∑ b in choose (finset.filter even (finset.range 1 10)), 3),
  (∑ b in choose (finset.filter odd (finset.range 1 10)), 4),
  (4.factorial) * (choose 5 3) * (3.factorial) = 1440 :=
sorry

end seven_digit_numbers_count_seven_digit_numbers_with_even_together_seven_digit_numbers_with_even_and_odd_together_seven_digit_numbers_no_two_even_adjacent_l299_299186


namespace find_b2023_l299_299700

noncomputable def seq (n : ℕ) : ℝ
| 0       := 2 + Real.sqrt 5
| 1       := 12 + Real.sqrt 5
| n       := if n < 2 then 0 else seq (n - 1) * seq (n + 1)

theorem find_b2023 : seq 2023 = (4 + 10 * Real.sqrt 5) / 3 :=
sorry

end find_b2023_l299_299700


namespace midpoint_of_rotated_segment_l299_299799

variables (A A' B B' O : Point)
-- Assuming that Point is a predefined type to represent points

-- Definitions for the given conditions
axiom tangent_point : tangent A O B
axiom rotated_segment : rotate_about_center O AB A'B'

-- The statement to be proven
theorem midpoint_of_rotated_segment :
  midpoint (line_through A A') (line_segment B B') :=
begin
  sorry -- Proof goes here
end

end midpoint_of_rotated_segment_l299_299799


namespace journey_time_l299_299458

theorem journey_time
  (t_1 t_2 : ℝ)
  (h1 : t_1 + t_2 = 5)
  (h2 : 40 * t_1 + 60 * t_2 = 240) :
  t_1 = 3 :=
sorry

end journey_time_l299_299458


namespace distance_between_trees_l299_299990

theorem distance_between_trees (n : ℕ) (length_yard : ℝ) (h1 : n = 52) (h2 : length_yard = 1500) :
  length_yard / (n - 1) ≈ 29.41 := 
by 
  sorry

end distance_between_trees_l299_299990


namespace janet_spending_difference_l299_299309

-- Defining hourly rates and weekly hours for each type of lessons
def clarinet_hourly_rate := 40
def clarinet_weekly_hours := 3
def piano_hourly_rate := 28
def piano_weekly_hours := 5
def violin_hourly_rate := 35
def violin_weekly_hours := 2
def singing_hourly_rate := 45
def singing_weekly_hours := 1

-- Calculating weekly costs
def clarinet_weekly_cost := clarinet_hourly_rate * clarinet_weekly_hours
def piano_weekly_cost := piano_hourly_rate * piano_weekly_hours
def violin_weekly_cost := violin_hourly_rate * violin_weekly_hours
def singing_weekly_cost := singing_hourly_rate * singing_weekly_hours
def combined_weekly_cost := piano_weekly_cost + violin_weekly_cost + singing_weekly_cost

-- Calculating annual costs with 52 weeks in a year
def weeks_per_year := 52
def clarinet_annual_cost := clarinet_weekly_cost * weeks_per_year
def combined_annual_cost := combined_weekly_cost * weeks_per_year

-- Proving the final statement
theorem janet_spending_difference :
  combined_annual_cost - clarinet_annual_cost = 7020 := by sorry

end janet_spending_difference_l299_299309


namespace QD_value_l299_299436

theorem QD_value (DEF : Triangle) (Q : Point) (E F D : Point)
  (h_right_angle : DEF ∠ E = 90°)
  (h_QE : dist Q E = 8)
  (h_QF : dist Q F = 15)
  (h_angles : ∠ EQF = ∠ FQD ∧ ∠ FQD = ∠ DQE ∧ ∠ DQE = 120°) :
  dist Q D = 97 / 11 := 
sorry

end QD_value_l299_299436


namespace mans_rate_in_still_water_l299_299462

theorem mans_rate_in_still_water :
  let speed_with_stream := 6
  let speed_against_stream := 4
  man's rate in still water = (speed_with_stream + speed_against_stream) / 2 := (5 : ℝ) :=
by
  sorry

end mans_rate_in_still_water_l299_299462


namespace product_remainder_div_5_l299_299120

theorem product_remainder_div_5 :
  (1234 * 1567 * 1912) % 5 = 1 :=
by
  sorry

end product_remainder_div_5_l299_299120


namespace question_solution_l299_299341

noncomputable def segment_ratio : (ℝ × ℝ) :=
  let m := 7
  let n := 2
  let x := - (2 / (m - n))
  let y := 7 / (m - n)
  (x, y)

theorem question_solution : segment_ratio = (-2/5, 7/5) :=
  by
  -- prove that the pair (x, y) calculated using given m and n equals (-2/5, 7/5)
  sorry

end question_solution_l299_299341


namespace geometric_sum_S40_l299_299028

theorem geometric_sum_S40 (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : ∀ n, S n = (finset.range n).sum (λ k, a k))
  (h2 : S 10 = 10) 
  (h3 : S 30 = 70) : 
  S 40 = 150 := 
sorry

end geometric_sum_S40_l299_299028


namespace solve_expression_l299_299377

noncomputable def b : ℝ := 0.76 * 0.76 * 0.76

theorem solve_expression : 
  let numerator := b - 0.008
  let denominator := (0.76 * 0.76) + (0.76 * 0.2) + 0.04
  (numerator / denominator) ≈ 0.5601 :=
by
  let numerator := b - 0.008
  let denominator := (0.76 * 0.76) + (0.76 * 0.2) + 0.04
  have : b = 0.76 * 0.76 * 0.76 := by sorry
  show (numerator / denominator) ≈ 0.5601 from sorry

end solve_expression_l299_299377


namespace range_of_f_l299_299171

-- Definition of the function f(B)
def f (B : ℝ) : ℝ := 
  (sin B * (2 * cos B^2 + cos B^4 + 2 * sin B^2 + sin B^2 * cos B^2)) / 
  (tan B * (sec B - sin B * tan B))

-- The main theorem stating the range of f(B)
theorem range_of_f :
  (∀ B : ℝ, B ≠ (n * π / 2) → f B ∈ (2, 3)) :=
  sorry

end range_of_f_l299_299171


namespace average_seashells_correct_l299_299722

def total_seashells (sally : ℕ) (tom : ℕ) (jessica : ℕ) (alex : ℕ) : ℕ :=
  sally + tom + jessica + alex

def num_people : ℕ := 4

def average_seashells_per_person (total_seashells : ℕ) (num_people : ℕ) : ℚ :=
  total_seashells / num_people

theorem average_seashells_correct :
  let sally := 9
  let tom := 7
  let jessica := 5
  let alex := 12
  total_seashells sally tom jessica alex / num_people = 8.25 := 
by
  sorry

end average_seashells_correct_l299_299722


namespace correct_props_l299_299397

namespace ComplexNumberPropositions

open Complex

noncomputable def z : ℂ := 2 / (-1 + I)

def p1 : Prop := abs z = 2
def p2 : Prop := z^2 = 2 * I
def p3 : Prop := conj z = -1 + I
def p4 : Prop := z.im = -1

theorem correct_props : p2 ∧ p4 := by 
  have h1 : z = -1 - I := by
    rw [z, div_eq_mul_inv, inv_def, mul_comm (2 : ℂ)]
    simp
  have h2 : p2 := by 
    rw [p2, h1]
    norm_num
  have h3 : p4 := by
    rw [p4, h1]
    simp
  exact ⟨h2, h3⟩
end ComplexNumberPropositions

end correct_props_l299_299397


namespace max_reddish_bluish_sum_l299_299887

theorem max_reddish_bluish_sum (n : ℕ) :
  let x_y_max := if n % 2 = 1 then 2 * n - 2
                 else if n >= 8 then 2 * n - 4
                 else if n = 2 then 0
                 else if n = 4 then 5
                 else if n = 6 then 9
                 else 0
  in x_y_max = maximum_value_of_x_plus_y n :=
by {
  sorry
}

end max_reddish_bluish_sum_l299_299887


namespace sin_A_sin_C_eq_3_over_4_triangle_is_equilateral_l299_299928

variable {α : Type*}

-- Part 1
theorem sin_A_sin_C_eq_3_over_4
  (A B C : Real)
  (a b c : Real)
  (h1 : b ^ 2 = a * c)
  (h2 : (Real.cos (A - C)) + (Real.cos B) = 3 / 2) :
  Real.sin A * Real.sin C = 3 / 4 :=
sorry

-- Part 2
theorem triangle_is_equilateral
  (A B C : Real)
  (a b c : Real)
  (h1 : b ^ 2 = a * c)
  (h2 : (Real.cos (A - C)) + (Real.cos B) = 3 / 2) :
  A = B ∧ B = C :=
sorry

end sin_A_sin_C_eq_3_over_4_triangle_is_equilateral_l299_299928


namespace parallel_chords_l299_299315

def circle (P : Type) [metric_space P] :=
{ center : P // ∀ p, dist (center : P) p = constant } 

variables {P : Type} [metric_space P] {O A B C D E M N : P} 

-- Definition of midpoints
def midpoint (a b : P) := ∃ m, dist a m = dist m b ∧ dist a m = dist b m

def perp (a b : P) :=
∃ c, ∀ x, dist b x = dist x a + dist x c

-- Given conditions
variables (ch_ab : ∃ (O : P) (r : ℝ), ∀ x, dist O x = r) (ch_cd : ∃ (O : P) (r : ℝ), ∀ x, dist O x = r)
          (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
          (right_angle : ∃ E, dist A E = dist E B ∧ dist C E = dist E D ∧ E ≠ A ∧ E ≠ B ∧ E ≠ C ∧ E ≠ D)
          (midpoints : midpoint A C M ∧ midpoint B D N)
          (perpendicular : perp M N ∧ perp O E)
          (condition : ∃ O, ∀ x, dist O x = constant ∧ M ⊥ O ∧ N ⊥ O)

-- Conclusion to prove
theorem parallel_chords : AB // AD ∥ BC →
  sorry

end parallel_chords_l299_299315


namespace problem_statement_l299_299210

noncomputable def general_term (a : ℕ → ℕ) (n : ℕ) : Prop :=
a n = n

noncomputable def sum_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
∀ n, S n = (n * (n + 1)) / 2

noncomputable def b_def (S : ℕ → ℕ) (b : ℕ → ℚ) : Prop :=
∀ n, b n = (2 : ℚ) / (S n)

noncomputable def sum_b_first_n_terms (b : ℕ → ℚ) (T : ℕ → ℚ) : Prop :=
∀ n, T n = (4 * n) / (n + 1)

theorem problem_statement (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℚ) (T : ℕ → ℚ) :
  (∀ n, a n = 1 + (n - 1) * 1) →
  a 1 = 1 →
  (∀ n, a (n + 1) - a n ≠ 0) →
  a 3 ^ 2 = a 1 * a 9 →
  general_term a 1 →
  sum_first_n_terms a S →
  b_def S b →
  sum_b_first_n_terms b T :=
by
  intro arithmetic_seq
  intro a_1_eq_1
  intro non_zero_diff
  intro geometric_seq
  intro gen_term_cond
  intro sum_terms_cond
  intro b_def_cond
  intro sum_b_terms_cond
  -- The proof goes here.
  sorry

end problem_statement_l299_299210


namespace number_of_seedlings_l299_299526

theorem number_of_seedlings (packets : ℕ) (seeds_per_packet : ℕ) (h1 : packets = 60) (h2 : seeds_per_packet = 7) : packets * seeds_per_packet = 420 :=
by
  sorry

end number_of_seedlings_l299_299526


namespace pizza_area_increase_l299_299637

theorem pizza_area_increase (r : ℝ) :
  let radius_medium := r,
      radius_large := 1.5 * r,
      area (radius : ℝ) := Real.pi * radius^2,
      area_medium := area radius_medium,
      area_large := area radius_large in
  ((area_large - area_medium) / area_medium) * 100 = 125 :=
by
  sorry

end pizza_area_increase_l299_299637


namespace unique_two_scoop_sundaes_l299_299849

theorem unique_two_scoop_sundaes (n : ℕ) (hn : n = 8) : ∃ k, k = Nat.choose 8 2 :=
by
  use 28
  sorry

end unique_two_scoop_sundaes_l299_299849


namespace sum_sin_double_angles_eq_l299_299460

theorem sum_sin_double_angles_eq (
  α β γ : ℝ
) (h : α + β + γ = Real.pi) :
  Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ) = 
  4 * Real.sin α * Real.sin β * Real.sin γ :=
sorry

end sum_sin_double_angles_eq_l299_299460


namespace determine_value_of_x_l299_299633

theorem determine_value_of_x {b x : ℝ} (hb : 1 < b) (hx : 0 < x) 
  (h_eq : (4 * x)^(Real.logb b 2) = (5 * x)^(Real.logb b 3)) : 
  x = (4 / 5)^(Real.logb (3 / 2) b) :=
by
  sorry

end determine_value_of_x_l299_299633


namespace gray_areas_trees_l299_299812

-- Define the variables and conditions.
def total_trees (T : ℕ) : Prop :=
  T = 100

def white_area_trees (white_trees : ℕ) : Prop :=
  white_trees = 82

-- The theorem to be proven.
theorem gray_areas_trees (T : ℕ) (white_trees : ℕ) (gray_1 gray_2 total_gray : ℕ):
  total_trees T ∧ white_area_trees white_trees ∧
  gray_1 = T - white_trees ∧ gray_2 = gray_1 ∧ total_gray = gray_1 + gray_2 →
  total_gray = 26 :=
by
  intro h,
  cases h with hT h,
  cases h with hW h,
  cases h with hG1 h,
  cases h with hG2 hTotalGray,
  rw [hT, hW] at *,
  rw [hG1, hG2] at hTotalGray,
  exact hTotalGray

-- Initial placeholders
constant T : ℕ
constant white_trees : ℕ
constant gray_1 : ℕ
constant gray_2 : ℕ
constant total_gray : ℕ

end gray_areas_trees_l299_299812


namespace tarantula_perimeter_l299_299532

-- Define the rectangles and their positions
structure Rectangle :=
(width : ℝ)
(height : ℝ)

def rect1 : Rectangle := ⟨3, 5⟩
def rect2 : Rectangle := ⟨3, 5⟩
def rect3 : Rectangle := ⟨3, 5⟩

-- Define positions (vertical stack of rect1 and rect2, horizontal placement of rect3 at the midpoint)
def isVertical (r : Rectangle) : Prop := r.width = 3 ∧ r.height = 5
def isHorizontalMidpoint (r : Rectangle) : Prop := r.width = 5 ∧ r.height = 3

-- Theorem to prove the perimeter
theorem tarantula_perimeter :
  isVertical rect1 ∧ isVertical rect2 ∧ isHorizontalMidpoint rect3 →
  let perimeter := rect1.height + rect2.height + rect3.width + rect1.width in
  perimeter = 16 :=
begin
  intros h,
  by_cases h1 : isVertical rect1 ∧ isVertical rect2 ∧ isHorizontalMidpoint rect3,
  simp [isVertical, isHorizontalMidpoint] at *,
  unfold rect1 rect2 rect3 at *,
  have hPerimeter : rect1.height + rect2.height + rect3.width + rect1.width = 16,
  {
     simp,
     have h_rect1 : rect1.height = 5,
     have h_rect2 : rect2.height = 5,
     have h_rect3 : rect3.width = 5,
     unfold rect1 rect2 rect3 at *,
     simp at *,
     exact hPerimeter,
  },
  sorry,
end

end tarantula_perimeter_l299_299532


namespace domain_of_f_l299_299009

noncomputable def f (x : ℝ) : ℝ := real.sqrt (real.log x / real.log 3)

theorem domain_of_f :
  {x : ℝ | 0 < x} = {x : ℝ | 1 ≤ x} :=
by
  sorry

end domain_of_f_l299_299009


namespace relationship_between_m_and_n_l299_299216

theorem relationship_between_m_and_n (f : ℝ → ℝ) (a : ℝ) 
  (h_even : ∀ x : ℝ, f x = f (-x)) 
  (h_mono_inc : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y) 
  (m_def : f (-1) = f 1) 
  (n_def : f (a^2 + 2*a + 3) > f 1) :
  f (-1) < f (a^2 + 2*a + 3) := 
by 
  sorry

end relationship_between_m_and_n_l299_299216


namespace EF_parallel_AB_l299_299358

theorem EF_parallel_AB 
  (A B C D K L M N E F : Type*) 
  [has_point A] [has_point B] 
  [has_point C] [has_point D] 
  [has_point K] [has_point L] 
  [has_point M] [has_point N] 
  [has_point E] [has_point F] 
  [has_line (A, B)] [has_line (B, C)] 
  [has_line (C, D)] [has_line (D, A)] 
  [is_square (A, B, C, D)] 
  [is_square (K, L, M, N)] 
  [on_side K (A, B)] [on_side L (B, C)] 
  [on_side M (C, D)] [on_side N (D, A)] 
  [intersection (DK, NM) E] 
  [intersection (KC, LM) F] : 
  parallel EF AB := sorry

end EF_parallel_AB_l299_299358


namespace possible_values_for_abc_l299_299213

theorem possible_values_for_abc (a b c : ℝ)
  (h : ∀ x y z : ℤ, (a * x + b * y + c * z) ∣ (b * x + c * y + a * z)) :
  (a, b, c) = (1, 0, 0) ∨ (a, b, c) = (0, 1, 0) ∨ (a, b, c) = (0, 0, 1) ∨
  (a, b, c) = (-1, 0, 0) ∨ (a, b, c) = (0, -1, 0) ∨ (a, b, c) = (0, 0, -1) :=
sorry

end possible_values_for_abc_l299_299213


namespace inequality_holds_l299_299364

-- Define parameters for the problem
variables (p q x y z : ℝ) (n : ℕ)

-- Define the conditions on x, y, and z
def condition1 : Prop := y = x^n + p*x + q
def condition2 : Prop := z = y^n + p*y + q
def condition3 : Prop := x = z^n + p*z + q

-- Define the statement of the inequality
theorem inequality_holds (h1 : condition1 p q x y n) (h2 : condition2 p q y z n) (h3 : condition3 p q x z n):
  x^2 * y + y^2 * z + z^2 * x ≥ x^2 * z + y^2 * x + z^2 * y :=
sorry

end inequality_holds_l299_299364


namespace find_interest_rate_l299_299473

theorem find_interest_rate (P r : ℝ) 
  (h1 : 100 = P * (1 + 2 * r)) 
  (h2 : 200 = P * (1 + 6 * r)) : 
  r = 0.5 :=
sorry

end find_interest_rate_l299_299473


namespace find_matrix_find_original_curve_l299_299198
  
variables (a b x y : ℝ)

-- Given conditions:
def matrix_M (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![a, 1; 3, b]

def eigenvector : Matrix (Fin 2) (Fin 1) ℝ := !![1; -3]

def eigenvalue : ℝ := -1

-- Question 1: Find the matrix M
theorem find_matrix :
  (matrix_M 2 0) * eigenvector = eigenvalue • eigenvector :=
  sorry

-- Question 2: Find the equation of curve C
variables (x' y' : ℝ)

def M : Matrix (Fin 2) (Fin 2) ℝ := !![2, 1; 3, 0]

def transformed_curve (x y : ℝ) : Matrix (Fin 2) (Fin 1) ℝ := M ⬝ !![x; y]

theorem find_original_curve (x y x' y' : ℝ) :
  (transformed_curve x y = !![x'; y']) → (x' * y' = 1) → (6 * x ^ 2 + 3 * x * y = 1) :=
  sorry

end find_matrix_find_original_curve_l299_299198


namespace find_other_number_l299_299423

theorem find_other_number (a b : ℕ) (h1 : a + b = 62) (h2 : b - a = 12) (h3 : a = 25) : b = 37 :=
sorry

end find_other_number_l299_299423


namespace largest_A_smallest_A_l299_299844

noncomputable def is_coprime_with_12 (n : Nat) : Prop :=
  Nat.gcd n 12 = 1

noncomputable def rotated_number (n : Nat) : Option Nat :=
  if n < 10^7 then none else
  let b := n % 10
  let k := n / 10
  some (b * 10^7 + k)

noncomputable def satisfies_conditions (B : Nat) : Prop :=
  B > 44444444 ∧ is_coprime_with_12 B

theorem largest_A :
  ∃ (B : Nat), satisfies_conditions B ∧ rotated_number B = some 99999998 :=
sorry

theorem smallest_A :
  ∃ (B : Nat), satisfies_conditions B ∧ rotated_number B = some 14444446 :=
sorry

end largest_A_smallest_A_l299_299844


namespace ab_value_l299_299256

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a * b = 9 :=
by
  sorry

end ab_value_l299_299256


namespace minimum_positive_period_and_monotonicity_l299_299227

def f (φ x : ℝ) : ℝ := sqrt 3 * sin (2 * x + φ) + cos (2 * x + φ)

theorem minimum_positive_period_and_monotonicity (φ : ℝ) (hφ : |φ| < π / 2) 
  (h_sym : ∀ x, f φ x = f φ (-x)) :
  ∃ T > 0, T = π ∧ ∀ x ∈ Ioo (0 : ℝ) (π / 2), f φ x > f φ (x + π / 2) :=
sorry

end minimum_positive_period_and_monotonicity_l299_299227


namespace parabola_vertex_shift_l299_299406

def shift_parabola (x : ℝ) : ℝ := (x - 2)^2 + 3

theorem parabola_vertex_shift :
  let original_vertex := (1, 2) in
  let shifted_vertex := (3, 5) in
  (shift_parabola 1, shift_parabola 2 + 3) = shifted_vertex := 
sorry

end parabola_vertex_shift_l299_299406


namespace Tara_gas_tank_capacity_l299_299736

variable {C : ℝ} -- Capacity of the gas tank

theorem Tara_gas_tank_capacity 
  (condition1 : 3 * C + 3.5 * C + 4 * C + 4.5 * C = 180)
  : C = 12 :=
by {
  sorry
}

end Tara_gas_tank_capacity_l299_299736


namespace range_of_x_range_of_a_l299_299337

-- Definitions of propositions p and q
def p (a x : ℝ) := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) := (x - 3) / (x - 2) ≤ 0

-- Question 1
theorem range_of_x (a x : ℝ) : a = 1 → p a x ∧ q x → 2 < x ∧ x < 3 := by
  sorry

-- Question 2
theorem range_of_a (a : ℝ) : (∀ x, ¬p a x → ¬q x) → (∀ x, q x → p a x) → 1 < a ∧ a ≤ 2 := by
  sorry

end range_of_x_range_of_a_l299_299337


namespace math_competition_selection_l299_299384

def studentA_scores := [82, 81, 79, 78, 95, 88, 93, 84]
def studentB_scores := [92, 95, 80, 75, 83, 80, 90, 85]

-- Define function to calculate the percentile
def percentile (scores : List ℝ) (p : ℝ) : ℝ := 
  let sorted := scores.sorted
  let rank := (sorted.length : ℝ) * p
  -- rounding up if rank is not an integer
  sorted.getOrElse (rank.ceil.toNat - 1) 0 -- rank.ceil gives a integer whose least value is 1

-- Calculate Variance
def variance (scores : List ℝ) : ℝ :=
  let mean := ((scores.foldl (· + ·) 0) / (scores.length : ℝ))
  (scores.foldl (λ acc x => acc + (x - mean)^2) 0) / (scores.length : ℝ)

-- Main theorem combining all parts
theorem math_competition_selection :
  percentile studentA_scores 0.8 = 93 ∧
  ((studentA_scores.foldl (· + ·) 0) / (studentA_scores.length : ℝ) = 85) ∧
  ((studentB_scores.foldl (· + ·) 0) / (studentB_scores.length : ℝ) = 85) ∧
  (variance studentA_scores = 35.5) ∧
  (variance studentB_scores = 41) →
  "Select Student A for better stability"
:= by
  sorry

end math_competition_selection_l299_299384


namespace sin_x_plus_pi_increasing_l299_299881

noncomputable def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

theorem sin_x_plus_pi_increasing :
  ∀ (x : ℝ),
    (-π/2 ≤ x ∧ x ≤ π) →
    (∃ (a : ℝ) (b : ℝ), a = π/2 ∧ b = π ∧ is_increasing (λ x, Real.sin (x + π)) a b) :=
by
  intros x h
  use [π/2, π]
  split; try { refl }
  intro t
  sorry

end sin_x_plus_pi_increasing_l299_299881


namespace find_train_length_l299_299465

variable (L : ℝ) -- Let L be the length of the train in meters.

axiom train_crosses_170m_platform : ∀ L, (L + 170) / 15 = (L + 250) / 20

theorem find_train_length (L : ℝ) (h : train_crosses_170m_platform L) : L = 70 := by
  sorry

end find_train_length_l299_299465


namespace alok_age_proof_l299_299119

variable (A B C : ℕ)

theorem alok_age_proof (h1 : B = 6 * A) (h2 : B + 10 = 2 * (C + 10)) (h3 : C = 10) : A = 5 :=
by
  sorry

end alok_age_proof_l299_299119


namespace cone_unfolded_area_l299_299599

theorem cone_unfolded_area (l r : ℝ) (h_l : l = 4) (h_r : r = 3) : 
  π * r * l = 12 * π :=
by
  rw [h_l, h_r]
  norm_num
  simp

end cone_unfolded_area_l299_299599


namespace number_of_even_face_painted_cubes_is_24_l299_299507

/-- A wooden block is modeled by a 6x6x1 block of 1 inch cubes. -/
def wooden_block : List (List (List Nat)) := List.replicate 1 (List.replicate 6 (List.replicate 6 1))

/-- Given a block painted on all six sides, calculate the number of cubes that have an even number of painted faces. -/
def painted_faces (l w h : Nat) (painted_faces_per_cube : Nat -> Nat) : Nat :=
  List.sum (wooden_block.flatten.map painted_faces_per_cube)

def even_painted_faces_count : Nat :=
  painted_faces 6 6 1 (fun f => if (f = 2) then 1 else 0)

theorem number_of_even_face_painted_cubes_is_24 : even_painted_faces_count = 24 :=
  by
    sorry

end number_of_even_face_painted_cubes_is_24_l299_299507


namespace probability_complement_l299_299585

theorem probability_complement (A B : set α) (P : measure α) (h : P (A ∪ B) = 3/4) : 
  P (Aᶜ ∩ Bᶜ) = 1/4 :=
by sorry

end probability_complement_l299_299585


namespace equal_sum_of_volumes_l299_299390

variables {S A B C D O X : Type}
variables [affine_space S] [affine_space A] [affine_space B] [affine_space C] [affine_space D] [affine_space O]

/-- Given a pyramid S A B C D with base parallelogram A B C D and a point O inside the pyramid,
    the sum of the volumes of tetrahedrons O S A B and O S C D is equal to the sum of the volumes 
    of tetrahedrons O S B C and O S D A. -/
theorem equal_sum_of_volumes 
  (H1 : parallelogram A B C D)
  (H2 : inside O (pyramid S A B C D)) :
  volume (tetrahedron O S A B) + volume (tetrahedron O S C D) = 
  volume (tetrahedron O S B C) + volume (tetrahedron O S D A) := 
sorry

end equal_sum_of_volumes_l299_299390


namespace angle_ACB_45_degrees_l299_299670

-- Definitions of the conditions
variables (A B C D E F : Type)
variables [triangle A B C] -- A and B and C form a triangle
variables [collinear A B D] -- point D lies on line segment AB
variables [collinear B C E] -- point E lies on line segment BC
variables [intersecting_points A E C D F] -- AE and CD intersect at F

def AB_eq_3AC (A B C : Type) [triangle A B C] : Prop :=
  length A B = 3 * length A C

def angle_BAE_eq_angle_ACD (A B C D E : Type) [triangle A B C] [collinear A B D] [collinear B C E] : Prop :=
  ∠BAE = ∠ACD

def isosceles_right_triangle (C E F : Type) : Prop :=
  (triangle C E F) ∧ (∠CFE = 90°) ∧ (length C F = length E F)

-- Statement of the problem
theorem angle_ACB_45_degrees
  (A B C D E F : Type)
  [triangle ABC]
  [collinear ABD]
  [collinear BCE]
  [intersecting_points A E C D F]
  (h1: AB_eq_3AC A B C)
  (h2: angle_BAE_eq_angle_ACD A B C D E)
  (h3: isosceles_right_triangle C E F)
  : ∠ACB = 45° :=
sorry

end angle_ACB_45_degrees_l299_299670


namespace thought_number_and_appended_digit_l299_299100

theorem thought_number_and_appended_digit (x y : ℕ) (hx : x > 0) (hy : y ≤ 9):
  (10 * x + y - x^2 = 8 * x) ↔ (x = 2 ∧ y = 0) ∨ (x = 3 ∧ y = 3) ∨ (x = 4 ∧ y = 8) := sorry

end thought_number_and_appended_digit_l299_299100


namespace caesars_meal_cost_l299_299380

theorem caesars_meal_cost :
  ∃ C : ℝ, let caesars_total_cost := 800 + 60 * C,
               venus_total_cost := 500 + 60 * 35,
           in caesars_total_cost = venus_total_cost ∧ C = 30 :=
by {
  sorry
}

end caesars_meal_cost_l299_299380


namespace sum_of_digits_10pow97_minus_97_l299_299252

-- Define a function that computes the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main statement we want to prove
theorem sum_of_digits_10pow97_minus_97 :
  sum_of_digits (10^97 - 97) = 858 :=
by
  sorry

end sum_of_digits_10pow97_minus_97_l299_299252


namespace max_c_exists_l299_299576

-- Define the conditions in Lean
variables {α : Type*} [plane α]
variables {l : vector3 α} -- direction vector in plane α
variables (F : set α) [is_convex F] -- convex set F within plane α
variables {S : set α → ℝ} -- area function for set α
variables {Δ : set α} -- inscribed triangle Δ within F with one side parallel to l
variables {SΔ : ℝ} -- area of the inscribed triangle Δ

-- Maximum constant c such that for any convex set F in the plane, the area of the largest inscribed triangle Δ with one side parallel to l is at least c times the area of F
theorem max_c_exists (l : vector3 α) (F : set α) [is_convex F] (S : set α → ℝ) (Δ : set α) (SΔ : ℝ) :
  ∃ (c : ℝ), (0 < c) ∧ (∀ (F : set α) [is_convex F], ∃ (Δ : set α), 
  (∃ (SΔ : ℝ), (inscribed_triangle_with_side_parallel_to l F Δ) ∧ (SΔ = area Δ) ∧ (SΔ ≥ c * (area F)))) :=
exists.intro (3/8) sorry

end max_c_exists_l299_299576


namespace trig_identity_l299_299873

theorem trig_identity : sin^2 (π / 8) - cos^2 (π / 8) = - (real.sqrt 2) / 2 :=
by
  sorry

end trig_identity_l299_299873


namespace solve_circle_problem_l299_299648

noncomputable def circle_problem (A B C D O : Type*) [metric_space O] [circle O 1] 
  (h1 : parallel (chord A B) (chord C D))
  (h2 : length_chord A C = x)
  (h3 : length_chord C D = x)
  (h4 : length_chord D B = x)
  (h5 : length_chord A B = y)
  (h6 : angle A O C = 60) : Prop :=
  y * x = sqrt 3 / 2

theorem solve_circle_problem {A B C D O : Type*} [metric_space O] [circle O 1]
  (h1 : parallel (chord A B) (chord C D))
  (h2 : length_chord A C = x)
  (h3 : length_chord C D = x)
  (h4 : length_chord D B = x)
  (h5 : length_chord A B = y)
  (h6 : angle A O C = 60) :
  y * x = sqrt 3 / 2 :=
sorry

end solve_circle_problem_l299_299648


namespace range_of_m_l299_299917

variable (x m : ℝ)

def p := (x - 4) / 3)^2 ≤ 4
def q := x^2 - 2*x + 1 - m^2 ≤ 0

theorem range_of_m (m_pos : m > 0) (h : ¬((x - 4) / 3)^2 ≤ 4) → ¬(x^2 - 2*x + 1 - m^2 ≤ 0) :=
  sorry
  have h1: m ≥ 9, from sorry,
  h1

end range_of_m_l299_299917


namespace problem_proof_l299_299218

-- Definitions from the problem
def arithmetic_seq (a : ℕ → ℚ) := ∃ (d : ℚ), ∀ n, a n = a 0 + n * d
def sum_seq (a : ℕ → ℚ) (S : ℕ → ℚ) := ∀ n, S n = (finset.range (n + 1)).sum a
def ge_seq (b : ℕ → ℚ) (r : ℚ) := ∀ n, b (n + 1) = b n * r

-- Problem conditions
axiom a_seq (a : ℕ → ℚ) (S : ℕ → ℚ) :
  arithmetic_seq a ∧ sum_seq a S ∧ 
  (∀ n, a n ^ 2 = S (2 * n - 1))

axiom b_seq (b : ℕ → ℚ) : 
  b 1 = -1/2 ∧ (∀ n, 2 * b (n + 1) = b n - 1)

-- Problem to prove
theorem problem_proof :
  (∀ (a S : ℕ → ℚ), a_seq a S → ∀ n, a n = 2 * n - 1) ∧
  (∀ (b : ℕ → ℚ), b_seq b → ge_seq (λ n, b n + 1) (1 / 2)) ∧
  (∀ (a b S : ℕ → ℚ) (T : ℕ → ℚ), 
     a_seq a S → b_seq b → 
     (∀ i, T i = (finset.range (i + 1)).sum (λ i, a i * (b i + 1))) → 
     ∀ n, T n = 3 - (2 * n + 3) * (1 / 2) ^ n) :=
sorry

end problem_proof_l299_299218


namespace rectangle_width_l299_299427

theorem rectangle_width
  (L W : ℝ)
  (h1 : W = L + 2)
  (h2 : 2 * L + 2 * W = 16) :
  W = 5 :=
by
  sorry

end rectangle_width_l299_299427


namespace coin_flips_probability_exactly_5_heads_in_7_l299_299046

noncomputable def fair_coin_probability (n k : ℕ) : ℚ :=
(nat.choose n k) / (2^n) 

theorem coin_flips_probability_exactly_5_heads_in_7 :
  fair_coin_probability 7 5 = 21 / 128 :=
by
  -- execution of this proof will require the detailed calculation of choose (7, 5) which is 21 and that the total outcomes of 7 flips equals 128
  sorry

end coin_flips_probability_exactly_5_heads_in_7_l299_299046


namespace charge_difference_l299_299795

theorem charge_difference (cost_x cost_y : ℝ) (num_copies : ℕ) (hx : cost_x = 1.25) (hy : cost_y = 2.75) (hn : num_copies = 40) : 
  num_copies * cost_y - num_copies * cost_x = 60 := by
  sorry

end charge_difference_l299_299795


namespace money_distribution_problem_l299_299000

theorem money_distribution_problem :
  ∃ (n : ℕ), 3 * n + (n * (n - 1)) / 2 = 100 * n ∧ n = 195 :=
by
  use 195
  split
  sorry
  rfl

end money_distribution_problem_l299_299000


namespace equation_of_ellipse_area_of_OAPB_l299_299214

variables {a b x y : ℝ} {A B M P : Point}

def ellipse_eq (a b x y : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (a > b) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def midpoint (A B M : Point) : Prop := M = Point.midpoint A B

def slopes_product (A B : Point) (l OM : Line) : Prop := 
let k := (A.y - B.y) / (A.x - B.x) in
let k_OM := (M.y / M.x) in
k * k_OM = -1/4

theorem equation_of_ellipse (a b : ℝ) (A B M P : Point) (l OM : Line) (C := {p : Point | p.1^2 / a^2 + p.2^2 / b^2 = 1})
  (hC : ellipse_eq a b x y) (hM : midpoint A B M) (hP : extension_OM A B P) (hS : slopes_product A B l OM)
  (h_major_axis : 2 * a = 4) :
  ∃ a b, C = {p : Point | p.1^2 / 4 + p.2^2 = 1} :=
sorry

theorem area_of_OAPB (a b : ℝ) (A B M P : Point) (l : Line) (C := {p : Point | p.1^2 / a^2 + p.2^2 / b^2 = 1})
  (hC : ellipse_eq a b x y) (hM : midpoint A B M) (hP : extension_OM A B P) (hS : slopes_product A B l OM)
  (h_parallel : parallelogram O A P B) :
  area O A P B = (sqrt 3 / 2) * a * b :=
sorry

end equation_of_ellipse_area_of_OAPB_l299_299214


namespace trees_in_yard_l299_299269

theorem trees_in_yard (L d : ℕ) (hL : L = 250) (hd : d = 5) : 
  (L / d + 1) = 51 := by
  sorry

end trees_in_yard_l299_299269


namespace find_B_l299_299395

theorem find_B (N : ℕ) (A B : ℕ) (H1 : N = 757000000 + A * 10000 + B * 1000 + 384) (H2 : N % 357 = 0) : B = 5 :=
sorry

end find_B_l299_299395


namespace dot_product_abs_l299_299326

-- Definitions of the vectors and conditions
variables (u v : ℝ^3)
variables (h₁ : ‖u‖ = 3) (h₂ : ‖v‖ = 4) (h₃ : ‖u × v‖ = 6)

-- Statement of the problem to prove
theorem dot_product_abs (u v : ℝ^3) (h₁ : ‖u‖ = 3) (h₂ : ‖v‖ = 4) (h₃ : ‖u × v‖ = 6) :
  |u ⬝ v| = 6 * real.sqrt 3 :=
sorry

end dot_product_abs_l299_299326


namespace tan_minus_alpha_l299_299208

noncomputable def m : ℝ := 4 / 3

theorem tan_minus_alpha :
  (∀ α : ℝ, (sin α = 4 / 5) ∧ (m = 4 / 3) → tan (-α) = 4 / 3) :=
begin
  sorry
end

end tan_minus_alpha_l299_299208


namespace increasing_log_function_range_l299_299261

theorem increasing_log_function_range (a : ℝ) :
  (∀ x y : ℝ, (1 < x ∧ 1 < y ∧ x < y → log 2 (x^2 + a * x) < log 2 (y^2 + a * y))) →
  a ≥ -1 :=
sorry

end increasing_log_function_range_l299_299261


namespace no_real_solution_l299_299900

theorem no_real_solution :
  ¬ ∃ x : ℝ, sqrt (4 - 5 * x) = 9 - x :=
by
  sorry

end no_real_solution_l299_299900


namespace estimate_proportion_ge_31_5_l299_299499

theorem estimate_proportion_ge_31_5 (frequencies : List (Set.Ico ℝ ℝ) × ℕ)
  (total_samples : ℕ) (expected_proportion : ℝ) :
  frequencies = [((11.5, 15.5), 2), ((15.5, 19.5), 4), ((19.5, 23.5), 9), ((23.5, 27.5), 18),
                 ((27.5, 31.5), 11), ((31.5, 35.5), 12), ((35.5, 39.5), 7), ((39.5, 43.5), 3)] →
  total_samples = 66 →
  expected_proportion = (22 : ℝ) / 66 →
  ((frequencies.filter (λ (interval, count), interval.1.1 >= 31.5).map (λ (interval, count), count)).sum : ℝ) / total_samples = expected_proportion :=
by
  sorry

end estimate_proportion_ge_31_5_l299_299499


namespace gold_copper_alloy_ratio_l299_299244

theorem gold_copper_alloy_ratio {G C A : ℝ} (hC : C = 9) (hA : A = 18) (hG : 9 < G ∧ G < 18) :
  ∃ x : ℝ, 18 = x * G + (1 - x) * 9 :=
by
  sorry

end gold_copper_alloy_ratio_l299_299244


namespace maximal_N_with_property_l299_299074

-- Define the function f satisfying the given conditions and recurrence relation
def f : ℕ → ℕ
| 1       := 1
| (n + 1) := 2 * f n + 1

theorem maximal_N_with_property : f 10 = 1023 :=
by {
  -- Following the recurrence relation defined
  induction n with n ih,
  -- Base case
  case zero {
    rw [f, zero_add],
  }, 
  -- Induction step
  case succ {
    rw [f, ih, mul_add, mul_one, nat.succ_eq_add_one],
  },
  sorry
}

end maximal_N_with_property_l299_299074


namespace alcohol_percentage_solution_x_l299_299727

theorem alcohol_percentage_solution_x :
  ∃ (P : ℝ), 
  (∀ (vol_x vol_y : ℝ), vol_x = 50 → vol_y = 150 →
    ∀ (percent_y percent_new : ℝ), percent_y = 30 → percent_new = 25 →
      ((P / 100) * vol_x + (percent_y / 100) * vol_y) / (vol_x + vol_y) = percent_new) → P = 10 :=
by
  -- Given conditions
  let vol_x := 50
  let vol_y := 150
  let percent_y := 30
  let percent_new := 25

  -- The proof body should be here
  sorry

end alcohol_percentage_solution_x_l299_299727


namespace vertex_of_transformed_parabola_l299_299404

theorem vertex_of_transformed_parabola :
  let original_parabola := λ x : ℝ, x^2 - 2 * x + 3
  let transformed_parabola := λ x : ℝ, (x - 2)^2 + 5
  ∃ vertex : ℝ × ℝ, vertex = (3, 5) ∧ ∀ x : ℝ, transformed_parabola x = original_parabola (x - 2) + 3 := 
by
  sorry

end vertex_of_transformed_parabola_l299_299404


namespace proof_sin_theta_l299_299327

noncomputable def sin_theta : ℝ :=
  let d : ℝ × ℝ × ℝ := (4, 5, 7)
  let n : ℝ × ℝ × ℝ := (3, 4, -7)
  let dot_product : ℝ := d.1 * n.1 + d.2 * n.2 + d.3 * n.3
  let norm_d : ℝ := real.sqrt (d.1 ^ 2 + d.2 ^ 2 + d.3 ^ 2)
  let norm_n : ℝ := real.sqrt (n.1 ^ 2 + n.2 ^ 2 + n.3 ^ 2)
  |dot_product| / (norm_d * norm_n)

theorem proof_sin_theta : sin_theta = 17 / Real.sqrt 6660 := by
  sorry

end proof_sin_theta_l299_299327


namespace yanni_money_left_in_cents_l299_299062

-- Define the constants based on the conditions
def initial_amount := 0.85
def mother_amount := 0.40
def found_amount := 0.50
def toy_cost := 1.60

-- Function to calculate the total amount
def total_amount := initial_amount + mother_amount + found_amount

-- Function to calculate the money left
def money_left := total_amount - toy_cost

-- Convert the remaining money from dollars to cents
def money_left_in_cents := money_left * 100

-- The theorem to prove
theorem yanni_money_left_in_cents : money_left_in_cents = 15 := by
  -- placeholder for proof, sorry used to skip the proof
  sorry

end yanni_money_left_in_cents_l299_299062


namespace sequence_2007th_term_is_85_l299_299012

noncomputable def sum_of_square_of_digits (n : ℕ) : ℕ :=
(n.digits 10).map (λ d, d * d).sum

noncomputable def sequence_term : ℕ → ℕ
| 0 := 2007
| (n+1) := sum_of_square_of_digits (sequence_term n)

theorem sequence_2007th_term_is_85 : sequence_term 2007 = 85 := 
sorry

end sequence_2007th_term_is_85_l299_299012


namespace part_one_part_two_l299_299339

noncomputable def f (x b : ℝ) : ℝ := (x + b) * Real.log x

noncomputable def f' (x b : ℝ) : ℝ := Real.log x + b / x + 1

noncomputable def g (x a b : ℝ) : ℝ := Real.exp x * ((f x b) / (x + 1) - a)

theorem part_one (b : ℝ) : 
  (∀ x, x + 2 * ((f 1 b - f x b) / (1 - x)) = 0) → f' 1 b = 2 → b = 1 :=
by 
  intro h1 h2
  sorry

theorem part_two (a b : ℝ) : 
  (∀ x > 0, a ≠ 0 → ((Real.exp x * (Real.log x - a))' = Real.exp x * (1/x - a + Real.log x))) → 
  (∀ x > 0, ((Real.exp x * (Real.log x - a))' ≥ 0)) → a ≤ 1 :=
by 
  intro h1 h2
  sorry

end part_one_part_two_l299_299339


namespace sin_pow_cos_pow_eq_l299_299933

theorem sin_pow_cos_pow_eq (x : ℝ) (h : Real.sin x ^ 10 + Real.cos x ^ 10 = 11 / 36) : 
  Real.sin x ^ 14 + Real.cos x ^ 14 = 41 / 216 := by
  sorry

end sin_pow_cos_pow_eq_l299_299933


namespace sufficient_but_not_necessary_condition_l299_299586

variable {α : Type*} (A B : Set α)

theorem sufficient_but_not_necessary_condition (h₁ : A ∩ B = A) (h₂ : A ≠ B) :
  (∀ x, x ∈ A → x ∈ B) ∧ ¬(∀ x, x ∈ B → x ∈ A) :=
by 
  sorry

end sufficient_but_not_necessary_condition_l299_299586


namespace tangent_line_at_point_l299_299010

def tangent_line_equation (f : ℝ → ℝ) (slope : ℝ) (p : ℝ × ℝ) :=
  ∃ (a b c : ℝ), a * p.1 + b * p.2 + c = 0 ∧ a = slope ∧ p.2 = f p.1

noncomputable def curve (x : ℝ) : ℝ := x^3 + x + 1

theorem tangent_line_at_point : 
  tangent_line_equation curve 4 (1, 3) :=
sorry

end tangent_line_at_point_l299_299010


namespace solve_for_m_l299_299065

theorem solve_for_m (n : ℝ) (m : ℝ) (h : 21 * (m + n) + 21 = 21 * (-m + n) + 21) : m = 1 / 2 := 
sorry

end solve_for_m_l299_299065


namespace total_cost_of_shirts_l299_299676

theorem total_cost_of_shirts 
    (first_shirt_cost : ℤ)
    (second_shirt_cost : ℤ)
    (h1 : first_shirt_cost = 15)
    (h2 : first_shirt_cost = second_shirt_cost + 6) : 
    first_shirt_cost + second_shirt_cost = 24 := 
by
  sorry

end total_cost_of_shirts_l299_299676


namespace variance_is_2_l299_299200

variable {a : ℝ}
variable data_set : Fin₅ → ℝ

def x1 := 2
def x2 := 3
def x3 := a
def x4 := 5
def x5 := 6

def data_set (i : Fin₅) : ℝ :=
  match i with
  | 0 => x1
  | 1 => x2
  | 2 => x3
  | 3 => x4
  | 4 => x5

-- The average of the data set is 4
axiom average_condition : (data_set 0 + data_set 1 + data_set 2 + data_set 3 + data_set 4) / 5 = 4

-- The goal is to prove that the variance of the data set is 2
theorem variance_is_2 : 2 :=
  sorry

end variance_is_2_l299_299200


namespace mark_song_count_l299_299707

/-- Mark does a gig every other day for 2 weeks.
 - Each gig consists of a certain number of songs: 2 songs that are 5 minutes long each and 
   the last song being twice as long, which is 10 minutes.
 - The total time he played over the 7 gigs is 280 minutes.
 - Prove that the number of songs he plays at each gig is 7.
 -/
theorem mark_song_count :
  ∀ (gigs_count : ℕ) (total_time : ℕ) (fixed_set_time : ℕ) (gig_time : ℕ) (additional_song_time : ℕ),
  gigs_count = 7 →
  total_time = 280 →
  fixed_set_time = 20 →
  gig_time = 40 →
  additional_song_time = 5 →
  (total_time / gigs_count = gig_time) →
  (fixed_set_time + 4 * additional_song_time = gig_time) →
  ∃ n, n = 7 :=
by
  intros gigs_count total_time fixed_set_time gig_time additional_song_time
  intro h_gigs_count h_total_time h_fixed_set_time h_gig_time h_additional_song_time h_gig_time_calc h_songs_calc
  use 7
  sorry

end mark_song_count_l299_299707


namespace sue_final_answer_l299_299833

theorem sue_final_answer (x : ℕ) (h : x = 8) : (4 * ((3 * (x + 3)) - 2)) = 124 := 
by
  rw [h]
  norm_num
  sorry

end sue_final_answer_l299_299833


namespace locus_of_M_l299_299919

noncomputable def circle {α : Type*} [metric_space α] (O : α) (R : ℝ) := 
  { p : α | dist O p = R }

theorem locus_of_M {α : Type*} [metric_space α] 
  (O A B M : α) (R : ℝ) (h_out : dist O A > R) 
  (h_tangent : ¬∃ P : α, circle O R P ∧ circle P (dist P A) B) 
  (h_intersect : ∀ (T1 T2 : α → set α), T1 A ∈ tangent_bundle (circle O R) ∧ T2 B ∈ tangent_bundle (circle O R) → 
  ∃ M : α, M ∈ T1 A ∩ T2 B)
  : ∃ L : set α, L = { P : α | ∀ Q : α, P = midpoint O Q } ∧ M ∈ L := 
by 
  sorry

end locus_of_M_l299_299919


namespace parametric_two_rays_l299_299394

theorem parametric_two_rays (t : ℝ) : (x = t + 1 / t ∧ y = 2) → (x ≤ -2 ∨ x ≥ 2) := by
  sorry

end parametric_two_rays_l299_299394


namespace minimum_value_of_y_l299_299912

theorem minimum_value_of_y :
  ∀ (x : ℝ), x > 3 → let y := x + 1 / (x - 3) in y ≥ 5 :=
by
  sorry

end minimum_value_of_y_l299_299912


namespace x_plus_y_plus_z_equals_4_l299_299630

theorem x_plus_y_plus_z_equals_4 (x y z : ℝ) 
  (h1 : 2 * x + 3 * y + 4 * z = 10) 
  (h2 : y + 2 * z = 2) : 
  x + y + z = 4 :=
by
  sorry

end x_plus_y_plus_z_equals_4_l299_299630


namespace adam_more_cans_of_cat_food_l299_299108

theorem adam_more_cans_of_cat_food
    (cat_packages : ℕ) (cans_per_cat_package : ℕ)
    (dog_packages : ℕ) (cans_per_dog_package : ℕ)
    (h_cat_packages : cat_packages = 9)
    (h_cans_per_cat_package : cans_per_cat_package = 10)
    (h_dog_packages : dog_packages = 7)
    (h_cans_per_dog_package : cans_per_dog_package = 5):
    cat_packages * cans_per_cat_package - dog_packages * cans_per_dog_package = 55 :=
by
  rw [h_cat_packages, h_cans_per_cat_package, h_dog_packages, h_cans_per_dog_package]
  simp
  sorry

end adam_more_cans_of_cat_food_l299_299108


namespace abs_diff_squares_104_96_l299_299442

theorem abs_diff_squares_104_96 : 
  |104^2 - 96^2| = 1600 := by
  sorry

end abs_diff_squares_104_96_l299_299442


namespace pos_real_unique_solution_l299_299552

theorem pos_real_unique_solution (x : ℝ) (hx_pos : 0 < x) (h : (x - 3) / 8 = 5 / (x - 8)) : x = 16 :=
sorry

end pos_real_unique_solution_l299_299552


namespace problem_statement_l299_299635

-- Define constants and initial conditions
def c128 : ℕ := 2^7
def c32 : ℕ := 2^5

-- Prove the target statement
theorem problem_statement (y : ℚ) (h : c128^7 = c32^y) : 2^(-3*y) = (1 / 2^(147 / 5)) :=
by 
  sorry

end problem_statement_l299_299635


namespace rachelle_gpa_probability_l299_299370

noncomputable def grade_A_points : ℕ := 5
noncomputable def grade_B_points : ℕ := 4
noncomputable def grade_C_points : ℕ := 3
noncomputable def grade_D_points : ℕ := 2

noncomputable def total_classes : ℕ := 5

noncomputable def assert_GPA : Rational := 4

noncomputable def english_A_probability : Rational := 1 / 7
noncomputable def english_B_probability : Rational := 1 / 5
noncomputable def english_C_probability : Rational := 1 - (1 / 7 + 1 / 5)

noncomputable def history_B_probability : Rational := 1 / 3
noncomputable def history_C_probability : Rational := 1 / 6
noncomputable def history_D_probability : Rational := 1 - (1 / 3 + 1 / 6)

noncomputable def assured_A_points : ℕ := 15
noncomputable def required_points : ℕ := assert_GPA * total_classes

noncomputable def aim_points_from_remaining_classes : ℕ := required_points - assured_A_points

theorem rachelle_gpa_probability : 
    (english_A_probability 
     + english_B_probability * history_B_probability 
     + english_B_probability * history_C_probability) 
    = 17 / 70 := 
by
  sorry

end rachelle_gpa_probability_l299_299370


namespace find_x_l299_299165

theorem find_x (x : ℝ) :
  (9^x + 32^x) / (15^x + 24^x) = 8 / 5 -> x = Real.log 2 (Real.cbrt (4 / 5)) :=
by
  sorry

end find_x_l299_299165


namespace steve_berry_picking_l299_299732

theorem steve_berry_picking :
  ∀ (monday tuesday wednesday thursday : ℕ)
  (total_income : ℕ)
  (rate_per_pound : ℕ),
  rate_per_pound = 2 →
  monday = 8 →
  tuesday = 3 * monday →
  wednesday = 0 →
  total_income = 100 →
  thursday = 50 - (monday + tuesday) →
  thursday = 18 :=
by
  intros monday tuesday wednesday thursday total_income rate_per_pound
  intros h_rate h_monday h_tuesday h_wednesday h_total_income h_thursday
  rw [h_rate, h_monday, h_tuesday, h_wednesday, h_total_income, h_thursday]
  exact congrArg (50 -) h_monday
  sorry

end steve_berry_picking_l299_299732


namespace area_error_percent_l299_299068

theorem area_error_percent (L W : ℝ) (L_pos : 0 < L) (W_pos : 0 < W) :
  let A := L * W
  let A_measured := (1.05 * L) * (0.96 * W)
  let error_percent := ((A_measured - A) / A) * 100
  error_percent = 0.8 :=
by
  let A := L * W
  let A_measured := (1.05 * L) * (0.96 * W)
  let error := A_measured - A
  let error_percent := (error / A) * 100
  sorry

end area_error_percent_l299_299068


namespace average_honey_per_bee_per_day_l299_299650

-- Definitions based on conditions
def num_honey_bees : ℕ := 50
def honey_bee_days : ℕ := 35
def total_honey_produced : ℕ := 75
def expected_avg_honey_per_bee_per_day : ℝ := 2.14

-- Statement of the proof problem
theorem average_honey_per_bee_per_day :
  ((total_honey_produced : ℝ) / (num_honey_bees * honey_bee_days)) = expected_avg_honey_per_bee_per_day := by
  sorry

end average_honey_per_bee_per_day_l299_299650


namespace perpendicular_lines_a_value_l299_299645

theorem perpendicular_lines_a_value :
  ∀ (a : ℝ), (∀ x y : ℝ, 2 * x - y = 0) -> (∀ x y : ℝ, a * x - 2 * y - 1 = 0) ->    
  (∀ m1 m2 : ℝ, m1 = 2 -> m2 = a / 2 -> m1 * m2 = -1) -> a = -1 :=
sorry

end perpendicular_lines_a_value_l299_299645


namespace calculate_expression_l299_299984

theorem calculate_expression : 
  let x := 7.5
  let y := 2.5
  (x ^ y + Real.sqrt x + y ^ x) - (x ^ 2 + y ^ y + Real.sqrt y) = 679.2044 :=
by
  sorry

end calculate_expression_l299_299984


namespace favorite_numbers_l299_299711

-- Define what it means to be a favorite number
def is_favorite (n : ℕ) : Prop :=
  n * (n.digits 10).sum = 10 * n

-- Conditions given in the problem
def condition (a b c : ℕ) : Prop :=
  is_favorite a ∧ is_favorite b ∧ is_favorite c ∧ a * b * c = 71668

-- Problem statement
theorem favorite_numbers (a b c : ℕ) (h : condition a b c) :
  {a, b, c} = {19, 46, 82} :=
by 
  sorry -- proof to be completed

end favorite_numbers_l299_299711


namespace Marnie_can_make_9_bracelets_l299_299347

def number_of_beads : Nat :=
  (5 * 50) + (2 * 100)

def beads_per_bracelet : Nat := 50

def total_bracelets (total_beads : Nat) (beads_per_bracelet : Nat) : Nat :=
  total_beads / beads_per_bracelet

theorem Marnie_can_make_9_bracelets :
  total_bracelets number_of_beads beads_per_bracelet = 9 :=
by
  -- proof goes here
  sorry

end Marnie_can_make_9_bracelets_l299_299347


namespace shaded_area_ratio_l299_299816

theorem shaded_area_ratio (n : ℕ) (hn : n = 6) :
  let large_square_area := n * n
  let small_square_area := 1
  let shaded_area := 4 * small_square_area
  (shaded_area : ℚ) / large_square_area = 1 / 9 := 
by 
  -- Definition of variables
  let large_square_area := n * n
  let small_square_area := 1
  let shaded_area := 4 * small_square_area
  -- Calculation
  have h1 : large_square_area = 36 := by rw [hn]
  have h2 : shaded_area = 4 := rfl
  -- Desired ratio
  show (4 : ℚ) / 36 = 1 / 9
  sorry

end shaded_area_ratio_l299_299816


namespace arithmetic_sequence_ratio_l299_299619

-- Define conditions
def sum_ratios (A_n B_n : ℕ → ℚ) (n : ℕ) : Prop := (A_n n) / (B_n n) = (4 * n + 2) / (5 * n - 5)
def arithmetic_sequences (a_n b_n : ℕ → ℚ) : Prop :=
  ∃ A_n B_n : ℕ → ℚ,
    (∀ n, A_n n = n * (a_n 1) + (n * (n - 1) / 2) * (a_n 2 - a_n 1)) ∧
    (∀ n, B_n n = n * (b_n 1) + (n * (n - 1) / 2) * (b_n 2 - b_n 1)) ∧
    ∀ n, sum_ratios A_n B_n n

-- Theorem to be proven
theorem arithmetic_sequence_ratio
  (a_n b_n : ℕ → ℚ)
  (h : arithmetic_sequences a_n b_n) :
  (a_n 5 + a_n 13) / (b_n 5 + b_n 13) = 7 / 8 :=
sorry

end arithmetic_sequence_ratio_l299_299619


namespace twice_abs_difference_of_squares_is_4000_l299_299451

theorem twice_abs_difference_of_squares_is_4000 :
  2 * |(105:ℤ)^2 - (95:ℤ)^2| = 4000 :=
by sorry

end twice_abs_difference_of_squares_is_4000_l299_299451


namespace domain_of_g_range_of_g_l299_299192

noncomputable def f (x : ℝ) : ℝ := 1 + Real.log x / Real.log 2

noncomputable def g (x : ℝ) : ℝ := (f x) ^ 2 + f (x ^ 2)

theorem domain_of_g : set.Icc (1 : ℝ) 2 = { x : ℝ | 1 ≤ x ∧ x ≤ 2 } :=
by
  sorry

theorem range_of_g : set.Icc (2 : ℝ) 7 = { g (x : ℝ) | 1 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end domain_of_g_range_of_g_l299_299192


namespace soda_price_increase_l299_299836

theorem soda_price_increase (P : ℝ) (h1 : 1.5 * P = 6) : P = 4 :=
by
  -- Proof will be provided here
  sorry

end soda_price_increase_l299_299836


namespace fraction_addition_l299_299054

theorem fraction_addition : (1 / 3) + (5 / 12) = 3 / 4 := 
sorry

end fraction_addition_l299_299054


namespace sum_first_9_terms_is_9_l299_299658

-- Define an arithmetic sequence
def arithmetic_seq (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Given condition
def condition (a₁ d : ℝ) : Prop :=
  2 * arithmetic_seq a₁ d 3 + arithmetic_seq a₁ d 9 = 3

-- Sum of the first 9 terms of an arithmetic sequence
def sum_first_9_terms (a₁ d : ℝ) : ℝ :=
  9 * a₁ + (9 * 8 / 2) * d

-- Main theorem to prove
theorem sum_first_9_terms_is_9 (a₁ d : ℝ) (h : condition a₁ d) :
  sum_first_9_terms a₁ d = 9 :=
sorry

end sum_first_9_terms_is_9_l299_299658


namespace median_room_number_is_14_l299_299518

-- Define the initial set of rooms
def total_rooms : List ℕ := List.range' 1 32

-- Define the excluded rooms
def excluded_rooms : List ℕ := [15, 16, 30]

-- Define the remaining rooms after excluding the specified ones
def remaining_rooms : List ℕ := total_rooms.filter (λ r, r ∉ excluded_rooms)

-- Define a function to find the median of the list
def median (l : List ℕ) : ℕ := l[(l.length / 2) - 1]

-- Assert the required property
theorem median_room_number_is_14 : median remaining_rooms = 14 := by
  -- Here we would provide the proof, but we will leave it as 'sorry' to focus on the statement
  sorry

end median_room_number_is_14_l299_299518


namespace sin_cos_pi_twelve_identity_l299_299450

theorem sin_cos_pi_twelve_identity : 
  (sin (Real.pi / 12) * cos (Real.pi / 12)) = 1 / 4 :=
by
  sorry

end sin_cos_pi_twelve_identity_l299_299450


namespace solve_for_a_l299_299926

noncomputable def p (x : ℝ) : Prop := 1 / 4 < 2^x ∧ 2^x < 16
noncomputable def q (x a : ℝ) : Prop := (x + 2) * (x + a) < 0

theorem solve_for_a : 
  ∀ a : ℝ, (∃ x : ℝ, p x → q x a) ∧ (∃ x : ℝ, q x a → ¬p x) ↔ a < -4 :=
by sorry

end solve_for_a_l299_299926


namespace product_min_max_a_b_sums_l299_299426

def a_seq : List Int := List.range' (-32) 10 2  -- Arithmetic sequence from -32 to 8 (inclusive)
def b_seq : List Int := List.range' (-17) 15 2  -- Arithmetic sequence from -17 to 13 (inclusive)

lemma min_a : a_seq.min = some (-32) :=
by
  sorry

lemma max_a : a_seq.max = some 8 :=
by
  sorry

lemma min_b : b_seq.min = some (-17) :=
by
  sorry

lemma max_b : b_seq.max = some 13 :=
by
  sorry

theorem product_min_max_a_b_sums : 
  (a_seq.min.getD 0 + b_seq.min.getD 0) * (a_seq.max.getD 0 + b_seq.max.getD 0) = -1029 :=
by
  calc
    (a_seq.min.getD 0 + b_seq.min.getD 0) = -49 := by rw [min_a, min_b]; rfl
    (a_seq.max.getD 0 + b_seq.max.getD 0) = 21 := by rw [max_a, max_b]; rfl
    (-49) * 21 = -1029 := by norm_num

end product_min_max_a_b_sums_l299_299426


namespace minimum_distance_on_locus_l299_299240

noncomputable def minimum_distance (a b : ℝ) : ℝ := 
  Real.sqrt (a ^ 2 + b ^ 2) + Real.sqrt ((a - 5) ^ 2 + (b + 1) ^ 2)

theorem minimum_distance_on_locus : 
  ∀ (P : ℝ × ℝ), 
  let (a, b) := P in 
  (b = -1 / 2 * a + 5 / 2) → minimum_distance a b = Real.sqrt 34 := 
by 
  sorry

end minimum_distance_on_locus_l299_299240


namespace spadesuit_problem_l299_299185

def spadesuit (x y : ℝ) : ℝ := x - 1 / y

theorem spadesuit_problem :
  spadesuit 3 (spadesuit (5 / 2) 3) = 33 / 13 := by
  sorry

end spadesuit_problem_l299_299185


namespace joey_hourly_wage_l299_299312

def sneakers_cost : ℕ := 92
def mowing_earnings (lawns : ℕ) (rate : ℕ) : ℕ := lawns * rate
def selling_earnings (figures : ℕ) (rate : ℕ) : ℕ := figures * rate
def total_additional_earnings (mowing : ℕ) (selling : ℕ) : ℕ := mowing + selling
def remaining_amount (total_cost : ℕ) (earned : ℕ) : ℕ := total_cost - earned
def hourly_wage (remaining : ℕ) (hours : ℕ) : ℕ := remaining / hours

theorem joey_hourly_wage :
  let total_mowing := mowing_earnings 3 8
  let total_selling := selling_earnings 2 9
  let total_earned := total_additional_earnings total_mowing total_selling
  let remaining := remaining_amount sneakers_cost total_earned
  hourly_wage remaining 10 = 5 :=
by
  sorry

end joey_hourly_wage_l299_299312


namespace inequality_between_zero_and_alpha_l299_299193

variable (a b c α β : ℝ)
variable (h_a : 0 < a)
variable (h_roots : ∀ x, (ax^2 + (b - 1)x + c = 0) ↔ (x = α ∨ x = β))
variable (h_alpha_beta : 0 < α ∧ α < β)

theorem inequality_between_zero_and_alpha :
  ∀ x, 0 < x → x < α → x < (λ x, a * x^2 + b * x + c) x :=
by
  intros x h1 h2
  sorry

end inequality_between_zero_and_alpha_l299_299193


namespace problem_statement_l299_299311

/-- Define the sequence of numbers spoken by Jo and Blair. -/
def next_number (n : ℕ) : ℕ :=
if n % 2 = 1 then (n + 1) / 2 else n / 2

/-- Helper function to compute the 21st number said. -/
noncomputable def twenty_first_number : ℕ :=
(21 + 1) / 2

/-- Statement of the problem in Lean 4. -/
theorem problem_statement : twenty_first_number = 11 := by
  sorry

end problem_statement_l299_299311


namespace lcm_of_multiple_numbers_l299_299470

open Nat

def lcm (a b : ℕ) : ℕ := a * b / gcd a b

theorem lcm_of_multiple_numbers : lcm (lcm 12 16) (lcm 18 24) = 144 :=
by
  -- Definition of LCM
  have h1 : lcm 12 16 = 48 := sorry
  have h2 : lcm 18 24 = 72 := sorry
  have h3 : lcm 48 72 = 144 := sorry
  show  lcm (lcm 12 16) (lcm 18 24) = 144 from Eq.trans (lcm_comm _) $
  Eq.trans (lcm_assoc _) $
  Eq.trans h1 $
  Eq.trans h2 h3
  sorry

end lcm_of_multiple_numbers_l299_299470


namespace best_ketchup_deal_l299_299862

def cost_per_ounce (price : ℝ) (ounces : ℝ) : ℝ := price / ounces

theorem best_ketchup_deal :
  let price_10oz := 1 in
  let ounces_10oz := 10 in
  let price_16oz := 2 in
  let ounces_16oz := 16 in
  let price_25oz := 2.5 in
  let ounces_25oz := 25 in
  let price_50oz := 5 in
  let ounces_50oz := 50 in
  let price_200oz := 10 in
  let ounces_200oz := 200 in
  let money := 10 in
  (∀ p o, cost_per_ounce p o ≥ cost_per_ounce price_200oz ounces_200oz) ∧ money = price_200oz :=
1
by
  sorry

end best_ketchup_deal_l299_299862


namespace area_of_equilateral_triangle_l299_299278

theorem area_of_equilateral_triangle {A B C : Type} [MetricSpace A] 
  [MetricSpace B] [MetricSpace C] (AB AC BC : ℝ) (h1 : AB = 15) (h2 : AC = 15) (h3 : BC = 15) :
  let AD := sqrt (15^2 - (15/2)^2) 
      area := 0.5 * BC * AD
  in area = 97.5 :=
by
  sorry

end area_of_equilateral_triangle_l299_299278


namespace max_value_of_quadratic_on_interval_l299_299615

theorem max_value_of_quadratic_on_interval : 
  ∃ (x : ℝ), -2 ≤ x ∧ x ≤ 2 ∧ (∀ y, (∃ x, -2 ≤ x ∧ x ≤ 2 ∧ y = (x + 1)^2 - 4) → y ≤ 5) :=
sorry

end max_value_of_quadratic_on_interval_l299_299615


namespace sin_equals_exp_intersections_l299_299170

noncomputable def sin_fn (x : ℝ) : ℝ := Real.sin x
noncomputable def exp_fn (x : ℝ) : ℝ := (1 / 3) ^ x

def interval := Set.Ioo 0 (100 * Real.pi)

def num_solutions_eq (f g : ℝ → ℝ) (a b : ℝ) := 
  { x : ℝ | a < x ∧ x < b ∧ f x = g x }.to_finset.card

theorem sin_equals_exp_intersections : 
  num_solutions_eq sin_fn exp_fn 0 (100 * Real.pi) = 100 := sorry

end sin_equals_exp_intersections_l299_299170


namespace find_imaginary_part_l299_299937

def imaginary_part_of_z {z : ℂ} (hz : conj z * (2 + 3 * complex.i) = (2 - complex.i) ^ 2) : ℂ :=
  z.im

theorem find_imaginary_part (z : ℂ) (hz : conj z * (2 + 3 * complex.i) = (2 - complex.i) ^ 2) :
  imaginary_part_of_z hz = 17 / 13 :=
sorry

end find_imaginary_part_l299_299937


namespace cot_G_in_right_triangle_l299_299164

theorem cot_G_in_right_triangle (FH GH FG : ℕ) (hFHG : GH^2 + FH^2 = FG^2) (hFG : FG = 13) (hGH : GH = 12) : 
  Real.cot (Real.arctan (FH / GH)) = 12 / 5 := 
by
  -- using triangle condition
  have hFH : FH = 5 :=
    calc
      FH = sqrt (FG^2 - GH^2) : by sorry
      _  = sqrt (169 - 144)   : by sorry
      _  = sqrt 25           : by sorry
      _  = 5                 : by sorry
  -- calculating cotangent
  have h_tan_G : tan (Real.arctan (FH / GH)) = FH / GH :=
    by sorry
  show Real.cot (Real.arctan (FH / GH)) = 12 / 5
    from by sorry

end cot_G_in_right_triangle_l299_299164


namespace ancient_chinese_poem_l299_299998

theorem ancient_chinese_poem (x : ℕ) :
  (7 * x + 7 = 9 * (x - 1)) := by
  sorry

end ancient_chinese_poem_l299_299998


namespace annual_income_increase_l299_299386

variable (x y : ℝ)

-- Definitions of the conditions
def regression_line (x : ℝ) : ℝ := 0.254 * x + 0.321

-- The statement we want to prove
theorem annual_income_increase (x : ℝ) : regression_line (x + 1) - regression_line x = 0.254 := 
sorry

end annual_income_increase_l299_299386


namespace time_for_B_alone_to_complete_work_l299_299807

theorem time_for_B_alone_to_complete_work :
  (∃ (A B C : ℝ), A = 1 / 4 
  ∧ B + C = 1 / 3
  ∧ A + C = 1 / 2)
  → 1 / B = 12 :=
by
  sorry

end time_for_B_alone_to_complete_work_l299_299807


namespace nail_marks_and_rotations_l299_299506

theorem nail_marks_and_rotations (r_wheel r_circle : ℝ) (hw : r_wheel = 18) (hc : r_circle = 40) :
  let ω := 2 * Real.pi * r_wheel,
      Ω := 2 * Real.pi * r_circle,
      lcm_ω_Ω := Nat.lcm (80 : ℕ) (36 : ℕ)
  in 
    lcm_ω_Ω / (Nat.gcd (80 : ℕ) (36 : ℕ)) = 720
    ∧ (lcm_ω_Ω / 36) = 20
    ∧ (lcm_ω_Ω / 80) = 9 :=
by
  have ω := 36 * Real.pi,
  have Ω := 80 * Real.pi,
  have lcm_ω_Ω := Nat.lcm (80 : ℕ) (36 : ℕ),
  split,
  { rw [Nat.lcm, Nat.gcd],
    sorry },
  split,
  { have rotations := lcm_ω_Ω / 36,
    exact rotations },
  { have marks := lcm_ω_Ω / 80,
    exact marks }


end nail_marks_and_rotations_l299_299506


namespace find_m_value_l299_299613

theorem find_m_value (m : ℕ) (h₀ : m > 0) 
(h₁ : ∃ f : ℝ → ℝ, (∀ x, f x = x ^ (m^2 + m)) ∧ f (real.sqrt 2) = 2) : m = 1 :=
by
  sorry

end find_m_value_l299_299613


namespace number_of_guest_cars_l299_299886

-- Definitions and conditions
def total_wheels : ℕ := 48
def mother_car_wheels : ℕ := 4
def father_jeep_wheels : ℕ := 4
def wheels_per_car : ℕ := 4

-- Theorem statement
theorem number_of_guest_cars (total_wheels mother_car_wheels father_jeep_wheels wheels_per_car : ℕ) : ℕ :=
  (total_wheels - (mother_car_wheels + father_jeep_wheels)) / wheels_per_car

-- Specific instance for the problem
example : number_of_guest_cars 48 4 4 4 = 10 := 
by
  sorry

end number_of_guest_cars_l299_299886


namespace range_of_a_l299_299575

def proposition_p (a : ℝ) : Prop := a > 1
def proposition_q (a : ℝ) : Prop := 0 < a ∧ a < 4

theorem range_of_a
(a : ℝ)
(h1 : a > 0)
(h2 : ¬ proposition_p a)
(h3 : ¬ proposition_q a)
(h4 : proposition_p a ∨ proposition_q a) :
  (0 < a ∧ a ≤ 1) ∨ (4 ≤ a) :=
by sorry

end range_of_a_l299_299575


namespace valid_integers_count_eq_eight_l299_299066

noncomputable def count_valid_integers : ℕ :=
  (set.univ.filter (λ n : ℕ, n >= 10 ∧ n < 100 ∧
    let a := n / 10 in
    let b := n % 10 in
    10 * b + a = n + 9)).card

theorem valid_integers_count_eq_eight : count_valid_integers = 8 :=
by
  sorry

end valid_integers_count_eq_eight_l299_299066


namespace problem1_problem2_problem3_l299_299871

-- Sub-problem 1
theorem problem1 : -2^2 + 3^0 - (-(1 / 2))⁻¹ = -1 :=
by
  sorry

-- Sub-problem 2
theorem problem2 (x : ℝ) : (x + 2) * (x - 1) - 3 * x * (x + 1) = 4 * (x^2 + x - 1 / 2) :=
by
  sorry

-- Sub-problem 3
theorem problem3 (a : ℝ) : 2 * a^6 - a^2 * a^4 + (2 * a^4)^2 / a^4 = a^4 * (a^2 + 4) :=
by
  sorry

end problem1_problem2_problem3_l299_299871


namespace compute_length_CD_l299_299985

   noncomputable theory

   def length_CD := CD = 5 * Real.sqrt 2

   open Real

   variable (A B C D : Point)
   variable (AB BC AC : ℝ)
   variable (angle_ABC : ℝ)
   variable (is_perpendicular_to_AB_at_A : ∃ p : Point, is_perpendicular p B A)
   variable (is_perpendicular_to_BC_at_C : ∃ p : Point, is_perpendicular p B C)

   -- Define the initial conditions
   def initial_conditions : Prop :=
     AB = 2 ∧
     BC = 5 ∧
     angle_ABC = 135 * (π / 180) ∧ -- convert to radians
     (∃ p : Point, is_perpendicular_to_AB_at_A) ∧
     (∃ q : Point, is_perpendicular_to_BC_at_C)

   -- The theorem statement
   theorem compute_length_CD (cond : initial_conditions) : length_CD :=
   sorry
   
end compute_length_CD_l299_299985


namespace line_equation_l299_299744

noncomputable def inclination_angle (θ : ℝ) : ℝ :=
  Real.tan θ

def slope_intercept_form (m b : ℝ) : (ℝ → ℝ) :=
  λ x, m * x + b

theorem line_equation (θ : ℝ) (b : ℝ) (m : ℝ) :
  θ = 120 ∧ b = -2 ∧ m = -Real.sqrt 3 →
  slope_intercept_form m b = λ x, -Real.sqrt 3 * x - 2 := by
  intro h
  sorry

end line_equation_l299_299744


namespace find_a_l299_299952

noncomputable def tangent_condition (a : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), y₀ = x₀ + 1 ∧ y₀ = Real.log (x₀ + a) ∧ (1 : ℝ) = (1 / (x₀ + a))

theorem find_a : ∃ a : ℝ, tangent_condition a ∧ a = 2 :=
by
  sorry

end find_a_l299_299952


namespace focus_of_parabola_l299_299007

theorem focus_of_parabola (x y : ℝ) : 
  x = 4 * y^2 → (1/16 : ℝ, 0 : ℝ) = (1/16, 0) :=
by
  sorry

end focus_of_parabola_l299_299007


namespace problem_proof_l299_299714

variable (u v w : ℝ^3)

def squared_norm (x : ℝ^3) : ℝ := x.dot_product x

noncomputable def AG_squared := squared_norm (u + v + w)
noncomputable def BH_squared := squared_norm (u - v + w)
noncomputable def CE_squared := squared_norm (-u + v + w)
noncomputable def DF_squared := squared_norm (u + v - w)

def AB_squared := squared_norm v
def AD_squared := squared_norm w
def AE_squared := squared_norm u

theorem problem_proof : 
  (AG_squared u v w + BH_squared u v w + CE_squared u v w + DF_squared u v w) / (AB_squared v + AD_squared w + AE_squared u) = 4 := 
  by 
    sorry

end problem_proof_l299_299714


namespace find_n_l299_299636

theorem find_n (n : ℕ) (h_pos : n > 0) (a b : ℕ)
  (h_expansion : (x + 1 : ℤ)^n = x^n + ∑ i in finset.range (n - 3), (C n i) * x^i + ax^3 + bx^2 + n * x + 1)
  (h_ratio : a : b = 3 : 1) : n = 11 :=
by sorry

end find_n_l299_299636


namespace dodecagon_areas_l299_299661

-- Define the conditions and the given area for quadrilateral EFGM
variables (S_ijm S_igh S_jkl S_cdem S_abml S_efgm : ℝ)

def dodecagon_conditions := 
  ∀ (S : ℝ), 
    (⟶let S_ijm = 1 in
     let S_igh = 2 in
     let S_jkl = 2 in
     let S_cdem = 5 in
     let S_abml = 7 in
     let S_efgm = 7 in
    S_efgm = 7) 

theorem dodecagon_areas
  (h : dodecagon_conditions) :
  S_ijm = 1 ∧ 
  S_igh = 2 ∧ 
  S_jkl = 2 ∧ 
  S_cdem = 5 ∧ 
  S_abml = 7 :=
begin
  sorry
end

end dodecagon_areas_l299_299661


namespace find_angle_AKB_l299_299305

-- Definitions for point, angle, and triangle
structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A B C : Point)
(is_iso : dist A B = dist B C)  -- Isosceles triangle condition

-- Function to measure the distance between points
def dist (p1 p2 : Point) : ℝ :=
real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Conditions about the points and angles
variable (A B C K : Point)
variable (h_iso_triangle : Triangle A B C).is_iso
variable (h_eq_dist : dist C K = dist A B)
variable (h_angle_KAC : ∠ K A C = 30)

-- The theorem to prove
theorem find_angle_AKB :
  ∠ A K B = 150 := sorry

end find_angle_AKB_l299_299305


namespace fencing_cost_l299_299794

noncomputable def pi_approx : ℝ := 3.14159

theorem fencing_cost 
  (d : ℝ) (r : ℝ)
  (h_d : d = 20) 
  (h_r : r = 1.50) :
  abs (r * pi_approx * d - 94.25) < 1 :=
by
  -- Proof omitted
  sorry

end fencing_cost_l299_299794


namespace geometric_sequence_first_term_l299_299031

theorem geometric_sequence_first_term (a r : ℝ) 
  (h1 : a * r^2 = 3)
  (h2 : a * r^4 = 27) : 
  a = - (real.sqrt 9) / 9 :=
by
  sorry

end geometric_sequence_first_term_l299_299031


namespace caesars_meal_cost_l299_299381

theorem caesars_meal_cost :
  ∃ C : ℝ, let caesars_total_cost := 800 + 60 * C,
               venus_total_cost := 500 + 60 * 35,
           in caesars_total_cost = venus_total_cost ∧ C = 30 :=
by {
  sorry
}

end caesars_meal_cost_l299_299381


namespace spinner_final_direction_is_west_l299_299673

/-
Initially, a spinner points south. Chenille first moves it clockwise 3 1/2 revolutions
and then counterclockwise 2 1/4 revolutions. Prove that the spinner points west.
-/

def initial_direction : Prop := true -- Initially, the spinner points south

def clockwise_revolutions : ℚ := 7 / 2 -- 3 1/2 revolutions clockwise
def counterclockwise_revolutions : ℚ := 9 / 4 -- 2 1/4 revolutions counterclockwise

def net_movement : ℚ := 7 / 2 - 9 / 4

def fractional_part_of_net_movement : ℚ := net_movement - (⌊net_movement⌋ : ℚ)

def direction_after_movement : Prop :=
  fractional_part_of_net_movement = 1 / 4 -- Moving 1/4 revolution clockwise leads to west

theorem spinner_final_direction_is_west :
  initial_direction → dir_trans (south, net_movement) = west := sorry

end spinner_final_direction_is_west_l299_299673


namespace lawnmower_percentage_drop_l299_299520

theorem lawnmower_percentage_drop :
  ∀ (initial_value value_after_one_year value_after_six_months : ℝ)
    (percentage_drop_in_year : ℝ),
  initial_value = 100 →
  value_after_one_year = 60 →
  percentage_drop_in_year = 20 →
  value_after_one_year = (1 - percentage_drop_in_year / 100) * value_after_six_months →
  (initial_value - value_after_six_months) / initial_value * 100 = 25 :=
by
  intros initial_value value_after_one_year value_after_six_months percentage_drop_in_year
  intros h_initial h_value_after_one_year h_percentage_drop_in_year h_value_equation
  sorry

end lawnmower_percentage_drop_l299_299520


namespace odd_two_digit_numbers_count_l299_299625

theorem odd_two_digit_numbers_count :
  let digits := [0, 1, 2, 3, 4]
  let odd_digits := [1, 3]
  ∃ (count : ℕ), count = 6 ∧ 
    count = (∑ d in odd_digits, 
               ∑ t in digits, 
                  if t ≠ d ∧ t ≠ 0 then 1 else 0) :=
by
  let digits := [0, 1, 2, 3, 4]
  let odd_digits := [1, 3]
  let count := (∑ d in odd_digits, 
               ∑ t in digits, 
                  if t ≠ d ∧ t ≠ 0 then 1 else 0)
  use 6
  rw count
  sorry

end odd_two_digit_numbers_count_l299_299625


namespace min_expression_of_vectors_l299_299189

def vec (x y : ℝ) : ℝ × ℝ := (x, y)

theorem min_expression_of_vectors 
  (m n : ℝ) (λ : ℝ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : ∃ λ ≠ 0, vec (1 - m) 2 = λ • vec n 1) :
  m + 2n = 1 → ∃ c, c = 9 ∧ (1 / m + 2 / n) = c :=
sorry

end min_expression_of_vectors_l299_299189


namespace sock_pairs_count_l299_299251

theorem sock_pairs_count :
  let white := 8
  let brown := 6
  let blue := 4
  (nat.choose white 2 + nat.choose brown 2 + nat.choose blue 2) = 49 := by
  sorry

end sock_pairs_count_l299_299251


namespace bananas_indeterminate_l299_299721

namespace RubyBananaProblem

variables (number_of_candies : ℕ) (number_of_friends : ℕ) (candies_per_friend : ℕ)
           (number_of_bananas : Option ℕ)

-- Given conditions
def Ruby_has_36_candies := number_of_candies = 36
def Ruby_has_9_friends := number_of_friends = 9
def Each_friend_gets_4_candies := candies_per_friend = 4
def Can_distribute_candies := number_of_candies = number_of_friends * candies_per_friend

-- Mathematical statement
theorem bananas_indeterminate (h1 : Ruby_has_36_candies number_of_candies)
                              (h2 : Ruby_has_9_friends number_of_friends)
                              (h3 : Each_friend_gets_4_candies candies_per_friend)
                              (h4 : Can_distribute_candies number_of_candies number_of_friends candies_per_friend) :
  number_of_bananas = none :=
by
  sorry

end RubyBananaProblem

end bananas_indeterminate_l299_299721


namespace cos_diff_half_l299_299905

theorem cos_diff_half (α β : ℝ) 
  (h1 : Real.cos α + Real.cos β = 1 / 2)
  (h2 : Real.sin α + Real.sin β = Real.sqrt 3 / 2) :
  Real.cos (α - β) = -1 / 2 :=
by
  sorry

end cos_diff_half_l299_299905


namespace maximal_area_inscribed_quadrilateral_l299_299362

theorem maximal_area_inscribed_quadrilateral 
  (A B C D : ℝ)
  (p : ℝ)
  (h1 : 0 < p)
  (h2 : sin A + sin B + sin C + sin D = p) :
  exists quadrilateral ABCD, (circle_in_circle ABCD) → max_area AB (circle_in_circle ABCD) :=
begin
  sorry -- This proof is deferred; it establishes that a quadrilateral in which a circle can be inscribed has the maximal area.
end

end maximal_area_inscribed_quadrilateral_l299_299362


namespace perimeter_of_shaded_region_l299_299288

-- Definitions
def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
def arc_length (circumference : ℝ, angle_deg : ℝ) : ℝ := (angle_deg / 360) * circumference

-- Problem statement
theorem perimeter_of_shaded_region (circum := 48) :
  let r := circumference / (2 * Real.pi) in
  let arc_len := arc_length circum 60 in
  3 * arc_len = 24 :=
by
  -- sorry is used to skip the proof for now.
  sorry

end perimeter_of_shaded_region_l299_299288


namespace fractional_sum_equals_015025_l299_299869

theorem fractional_sum_equals_015025 :
  (2 / 20) + (8 / 200) + (3 / 300) + (5 / 40000) * 2 = 0.15025 := 
by
  sorry

end fractional_sum_equals_015025_l299_299869


namespace shorties_donut_today_l299_299282

theorem shorties_donut_today :
  let daily_eaters := 6
  let bi_daily_eaters := 8
  let yesterday_donuts := 11
  yesterday_donuts <= daily_eaters + bi_daily_eaters → -- Ensure only correct conditions are used, math prerequisites also apply
  (yesterday_donuts - daily_eaters <= bi_daily_eaters) →  -- Verify this Yesterday's donut should be from who eats daily and bi-daily
  daily_eaters + (bi_daily_eaters - (yesterday_donuts - daily_eaters)) = 9 := -- Condition to prove
  begin
    sorry -- Proof omitted as per instructions
  end

end shorties_donut_today_l299_299282


namespace sum_of_coefficients_binomial_expansion_l299_299121

theorem sum_of_coefficients_binomial_expansion : 
  (Polynomial.sum (\(n : Nat) x => (Polynomial.coeff ((Polynomial.C (1: ℚ) - Polynomial.X)^7) n) x)) = 0 :=
by
  sorry

end sum_of_coefficients_binomial_expansion_l299_299121


namespace range_of_f_on_interval_l299_299610

noncomputable def f : ℝ → ℝ := λ x, 2 * Real.sin (x / 3 + Real.pi / 6)

theorem range_of_f_on_interval :
  set.image f (set.Icc 0 (3 * Real.pi / 2)) = set.Icc 1 2 :=
sorry

end range_of_f_on_interval_l299_299610


namespace tetrahedron_four_edge_paths_l299_299766

theorem tetrahedron_four_edge_paths (P Q R S : Type) (graph : P × Q × R × S → Bool) (hP : ∀ x, graph (P, x)) (hQ : ∀ x, graph (Q, x)) (hR : ∀ x, graph (R, x)) (hS : ∀ x, graph (S, x)) :
  ∃! (paths : Set (List P × Q × R × S )), paths.card = 4 ∧ ∀ (path ∈ paths), path.length = 4 ∧ path.head = P ∧ path.last = R :=
by
  sorry

end tetrahedron_four_edge_paths_l299_299766


namespace rubiks_cube_repeats_l299_299144

theorem rubiks_cube_repeats (num_positions : ℕ) (H : num_positions = 43252003274489856000) 
  (moves : ℕ → ℕ) : 
  ∃ n, ∃ m, (∀ P, moves n = moves m → P = moves 0) :=
by
  sorry

end rubiks_cube_repeats_l299_299144


namespace sum_f_1_to_50_l299_299211

variable {f : ℝ → ℝ}

-- Definitions given in the conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)
def functional_symmetry (f : ℝ → ℝ) : Prop := ∀ x, f(1 - x) = f(1 + x)
def f_value (f : ℝ → ℝ) (n : ℝ) (value : ℝ) : Prop := f(n) = value

theorem sum_f_1_to_50 :
  odd_function f →
  functional_symmetry f →
  f_value f 1 2 →
  (f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 +
   f 11 + f 12 + f 13 + f 14 + f 15 + f 16 + f 17 + f 18 + f 19 + f 20 +
   f 21 + f 22 + f 23 + f 24 + f 25 + f 26 + f 27 + f 28 + f 29 + f 30 +
   f 31 + f 32 + f 33 + f 34 + f 35 + f 36 + f 37 + f 38 + f 39 + f 40 +
   f 41 + f 42 + f 43 + f 44 + f 45 + f 46 + f 47 + f 48 + f 49 + f 50) = 2 := by
  sorry

end sum_f_1_to_50_l299_299211


namespace largest_A_proof_smallest_A_proof_l299_299848

def is_coprime_with_12 (n : ℕ) : Prop := Nat.gcd n 12 = 1

def obtain_A_from_B (B : ℕ) : ℕ :=
  let b := B % 10
  let k := B / 10
  b * 10^7 + k

constant B : ℕ → Prop
constant A : ℕ → ℕ → Prop

noncomputable def largest_A : ℕ :=
  99999998

noncomputable def smallest_A : ℕ :=
  14444446

theorem largest_A_proof (B : ℕ) (h1 : B > 44444444) (h2 : is_coprime_with_12 B) :
  obtain_A_from_B B = largest_A :=
sorry

theorem smallest_A_proof (B : ℕ) (h1 : B > 44444444) (h2 : is_coprime_with_12 B) :
  obtain_A_from_B B = smallest_A :=
sorry

end largest_A_proof_smallest_A_proof_l299_299848


namespace unicorns_total_games_l299_299117

theorem unicorns_total_games (y x : ℕ) (h1 : x = 0.5 * y) (h2 : (x + 8) = 0.55 * (y + 11)) : (y + 11) = 50 := by
  sorry

end unicorns_total_games_l299_299117


namespace smallest_five_digit_number_divisible_by_primes_l299_299561

noncomputable def lcm_of_primes := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 2 3) 5) 7) 11

theorem smallest_five_digit_number_divisible_by_primes :
  ∃ n, 10000 ≤ n ∧ n ≤ 99999 ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ (11 ∣ n) ∧ n = 11550 :=
by
  let m := lcm_of_primes
  have h1 : m = 2310 := by norm_num [lcm_of_primes, Nat.lcm, Nat.lcm_assoc, Nat.lcm_comm]
  have h2 : 10000 / 2310 = 4.329004329 := by norm_num
  have h3 : ∃ x : ℕ, x = 5 := ⟨5, rfl⟩
  existsi 2310 * 5
  split
  { norm_num }
  split
  { norm_num }
  repeat { norm_num[HasDvd.dvd] }
  sorry  -- skip the proof

end smallest_five_digit_number_divisible_by_primes_l299_299561


namespace molecular_weight_of_compound_l299_299777

def atomic_weight (count : ℕ) (atomic_mass : ℝ) : ℝ :=
  count * atomic_mass

def molecular_weight (C_atom_count H_atom_count O_atom_count : ℕ)
  (C_atomic_weight H_atomic_weight O_atomic_weight : ℝ) : ℝ :=
  (atomic_weight C_atom_count C_atomic_weight) +
  (atomic_weight H_atom_count H_atomic_weight) +
  (atomic_weight O_atom_count O_atomic_weight)

theorem molecular_weight_of_compound :
  molecular_weight 3 6 1 12.01 1.008 16.00 = 58.078 :=
by
  sorry

end molecular_weight_of_compound_l299_299777


namespace ratio_of_heights_l299_299643

theorem ratio_of_heights (a b : ℝ) (area_ratio_is_9_4 : a / b = 9 / 4) :
  ∃ h₁ h₂ : ℝ, h₁ / h₂ = 3 / 2 :=
by
  sorry

end ratio_of_heights_l299_299643


namespace symmetric_circle_eq_common_chord_length_l299_299942

theorem symmetric_circle_eq (a b : ℝ) 
  (P_symmetric : ∀ (x y : ℝ), (x = a ∧ y = b) → (y + 1 = b + 1 ∧ x - 1 = a - 1))
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 - 6 * x - 2 * y = 0) :
  ∀ (x y : ℝ), (x-2)^2 + (y-2)^2 = 10 := 
sorry

theorem common_chord_length (x y : ℝ) 
  (circle_C : ∀ (x y : ℝ), x^2 + y^2 - 6 * x - 2 * y = 0)
  (circle_C_prime : ∀ (x y : ℝ), (x-2)^2 + (y-2)^2 = 10) :
  length_common_chord := 
  sqrt 38 :=
sorry

end symmetric_circle_eq_common_chord_length_l299_299942


namespace vertex_of_transformed_parabola_l299_299403

theorem vertex_of_transformed_parabola :
  let original_parabola := λ x : ℝ, x^2 - 2 * x + 3
  let transformed_parabola := λ x : ℝ, (x - 2)^2 + 5
  ∃ vertex : ℝ × ℝ, vertex = (3, 5) ∧ ∀ x : ℝ, transformed_parabola x = original_parabola (x - 2) + 3 := 
by
  sorry

end vertex_of_transformed_parabola_l299_299403


namespace correct_proposition_is_C_l299_299112

-- Defining the propositions as conditions
def propositionA (x : ℝ) : Prop := x > 3 → x > 5
def propositionB (x : ℝ) : Prop := x^2 = 1 → x = 1
def propositionC (a b c : ℝ) : Prop := ac^2 > bc^2 → a > b
def propositionD (α : ℝ) : Prop := (sin α = 1) → α = π / 2

-- Define the proof problem
theorem correct_proposition_is_C (a b c : ℝ) :
  (propositionA x = false) ∧
  (propositionB x = false) ∧
  (propositionC a b c = true) ∧
  (propositionD α = false) :=
by
  sorry

end correct_proposition_is_C_l299_299112


namespace find_f_of_f_at_2_l299_299695

def f (x : ℝ) : ℝ :=
  if x < 2 then 2 * Real.exp(x - 1) else Real.log 3 (x^2 - 1)

theorem find_f_of_f_at_2 :
  f (f 2) = 2 := by
  sorry

end find_f_of_f_at_2_l299_299695


namespace trig_identity_A_trig_identity_D_l299_299453

theorem trig_identity_A : 
  (Real.tan (25 * Real.pi / 180) + Real.tan (20 * Real.pi / 180) + Real.tan (25 * Real.pi / 180) * Real.tan (20 * Real.pi / 180) = 1) :=
by sorry

theorem trig_identity_D : 
  (1 / Real.sin (10 * Real.pi / 180) - Real.sqrt 3 / Real.cos (10 * Real.pi / 180) = 4) :=
by sorry

end trig_identity_A_trig_identity_D_l299_299453


namespace c_share_of_profit_l299_299793

theorem c_share_of_profit (a b c total_profit : ℕ) 
  (h₁ : a = 5000) (h₂ : b = 8000) (h₃ : c = 9000) (h₄ : total_profit = 88000) :
  c * total_profit / (a + b + c) = 36000 :=
by
  sorry

end c_share_of_profit_l299_299793


namespace abc_minus_def_l299_299180

def f (x y z : ℕ) : ℕ := 5^x * 2^y * 3^z

theorem abc_minus_def {a b c d e f : ℕ} (ha : a = d) (hb : b = e) (hc : c = f + 1) : 
  (100 * a + 10 * b + c) - (100 * d + 10 * e + f) = 1 :=
by
  -- Proof omitted
  sorry

end abc_minus_def_l299_299180


namespace log_sum_equality_l299_299125

noncomputable def log_base_5 (x : ℝ) := Real.log x / Real.log 5

theorem log_sum_equality :
  2 * log_base_5 10 + log_base_5 0.25 = 2 :=
by
  sorry -- proof goes here

end log_sum_equality_l299_299125


namespace max_distance_line_ellipse_l299_299817

theorem max_distance_line_ellipse :
  (∀ (t : ℝ), ¬(t^2 < 5) → true) →
  ∃ t : ℝ, -sqrt 5 < t ∧ t < sqrt 5 ∧
  (let x1 := -4 * t / 5 in
   let x2 := (4 * t^2 - 4) / 5 in
   real.sqrt 2 * real.sqrt ((x1 + x1) ^ 2 - 4 * x1 * x2) ≤ 4 * real.sqrt 10 / 5) :=
begin
  sorry
end

end max_distance_line_ellipse_l299_299817


namespace sum_first_m_terms_inequality_always_holds_l299_299951

noncomputable def f (x : ℝ) : ℝ := 1 / (4^x + 2)

-- Define the sequence {a_n}
noncomputable def a_n (n m : ℕ) : ℝ := f (n / m)

-- Define the sum S_m
noncomputable def S_m (m : ℕ) : ℝ := ∑ n in Finset.range m, a_n n m

theorem sum_first_m_terms (m : ℕ) : S_m m = (1 / 12) * (3 * m - 1) := sorry

theorem inequality_always_holds (m : ℕ) (a : ℝ) (h : ∀ m : ℕ, (a^m / S_m m) < (a^(m+1) / S_m (m+1))) : a > 5/2 := sorry

end sum_first_m_terms_inequality_always_holds_l299_299951


namespace part_i_l299_299077

theorem part_i (n : ℕ) (a : Fin (n+1) → ℤ) :
  ∃ (i j : Fin (n+1)), i ≠ j ∧ (a i - a j) % n = 0 := by
  sorry

end part_i_l299_299077


namespace div_add_sqrt_l299_299176

theorem div_add_sqrt :
  (Real.sqrt 1.21) / (Real.sqrt 0.81) + (Real.sqrt 0.81) / (Real.sqrt 0.49) = 158 / 63 :=
by
  have h1 : Real.sqrt 1.21 = 1.1 := by sorry
  have h2 : Real.sqrt 0.81 = 0.9 := by sorry
  have h3 : Real.sqrt 0.49 = 0.7 := by sorry
  calc
    (Real.sqrt 1.21) / (Real.sqrt 0.81) + (Real.sqrt 0.81) / (Real.sqrt 0.49)
      = 1.1 / 0.9 + 0.9 / 0.7 : by rw [h1, h2, h3]
    ... = 158 / 63 : by sorry

end div_add_sqrt_l299_299176


namespace xy_sum_to_one_l299_299584

-- Declare the points O, A, B, C as vectors in a vector space
variables (V : Type*) [AddCommGroup V] [Module ℝ V]
variables (O A B C : V)

-- Declare the scalars x and y
variables (x y : ℝ)

-- The conditions
-- 1. Points A, B, and C are collinear
-- 2. \vec{OC} = x \vec{OA} + y \vec{OB}
axiom collinear_ABC : ∃ (m : ℝ), C = m • A + (1 - m) • B
axiom oc_decomposition : C = x • A + y • B

-- The theorem to be proved
theorem xy_sum_to_one (h_collinear : collinear_ABC) (h_decomposition : oc_decomposition) : 
  x + y = 1 :=
sorry

end xy_sum_to_one_l299_299584


namespace binomial_coeff_l299_299262

-- Define the condition where the line x + ay - 1 = 0 is perpendicular to 2x - 4y + 3 = 0
def is_perpendicular (a : ℝ) : Prop :=
  let m₁ := -1 / a in
  let m₂ := 1 / 2 in
  m₁ * m₂ = -1

-- Define the main theorem stating the coefficient of x in the binomial expansion
theorem binomial_coeff (a : ℝ) (h : is_perpendicular a) : 
  let b := -(5.choose 3) * (1/2)^(5-3) * (-1)^3 in 
  (-b) = -(5.choose 3) * (1/2)^2 * (-1 : ℝ) := 
by sorry

end binomial_coeff_l299_299262


namespace number_of_correct_statements_l299_299666

-- Definitions of points M and their symmetric points
def M (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)
def M1 (a b c : ℝ) : ℝ × ℝ × ℝ := (a, -b, c)
def M2 (a b c : ℝ) : ℝ × ℝ × ℝ := (a, -b, -c)
def M3 (a b c : ℝ) : ℝ × ℝ × ℝ := (a, -b, c)
def M4 (a b c : ℝ) : ℝ × ℝ × ℝ := (-a, -b, -c)

-- Actual symmetric points
def actual_M1 (a b c : ℝ) : ℝ × ℝ × ℝ := (a, -b, -c)
def actual_M2 (a b c : ℝ) : ℝ × ℝ × ℝ := (-a, b, c)
def actual_M3 (a b c : ℝ) : ℝ × ℝ × ℝ := (-a, b, -c)
def actual_M4 (a b c : ℝ) : ℝ × ℝ × ℝ := (-a, -b, -c)

theorem number_of_correct_statements (a b c : ℝ) 
  (H1 : M1 a b c = actual_M1 a b c -> true)
  (H2 : M2 a b c = actual_M2 a b c -> true)
  (H3 : M3 a b c = actual_M3 a b c -> true)
  (H4 : M4 a b c = actual_M4 a b c -> true) : 
  (iff (M1 a b c = actual_M1 a b c) false) ∧ 
  (iff (M2 a b c = actual_M2 a b c) false) ∧ 
  (iff (M3 a b c = actual_M3 a b c) false) ∧ 
  (iff (M4 a b c = actual_M4 a b c) true) := 
by
  sorry

end number_of_correct_statements_l299_299666


namespace increase_to_restore_l299_299756

noncomputable def percentage_increase_to_restore (P : ℝ) : ℝ :=
  let reduced_price := 0.9 * P
  let restore_factor := P / reduced_price
  (restore_factor - 1) * 100

theorem increase_to_restore :
  percentage_increase_to_restore 100 = 100 / 9 :=
by
  sorry

end increase_to_restore_l299_299756


namespace Marnie_can_make_9_bracelets_l299_299346

def number_of_beads : Nat :=
  (5 * 50) + (2 * 100)

def beads_per_bracelet : Nat := 50

def total_bracelets (total_beads : Nat) (beads_per_bracelet : Nat) : Nat :=
  total_beads / beads_per_bracelet

theorem Marnie_can_make_9_bracelets :
  total_bracelets number_of_beads beads_per_bracelet = 9 :=
by
  -- proof goes here
  sorry

end Marnie_can_make_9_bracelets_l299_299346


namespace inverse_function_problem_l299_299696

theorem inverse_function_problem
  (f : ℝ → ℝ)
  (f_inv : ℝ → ℝ)
  (h₁ : ∀ x, f (f_inv x) = x)
  (h₂ : ∀ x, f_inv (f x) = x)
  (a b : ℝ)
  (h₃ : f_inv (a - 1) + f_inv (b - 1) = 1) :
  f (a * b) = 3 :=
by
  sorry

end inverse_function_problem_l299_299696


namespace number_of_ways_to_choose_subsets_l299_299500

-- Define the set S
def S : Finset ℕ := {0, 1, 2, 3, 4, 5}

-- Define the problem statement
theorem number_of_ways_to_choose_subsets : 
  (Finset.powersetLen 3 S).card * 8 / 2 = 80 := 
by 
  sorry

end number_of_ways_to_choose_subsets_l299_299500


namespace find_x_l299_299295

theorem find_x 
  (b : ℤ) (h_b : b = 0) 
  (a z y x w : ℤ)
  (h1 : z + a = 1)
  (h2 : y + z + a = 0)
  (h3 : x + y + z = a)
  (h4 : w + x + y = z)
  :
  x = 2 :=
by {
    sorry
}    

end find_x_l299_299295


namespace min_value_of_y_l299_299909

theorem min_value_of_y (x : ℝ) (h : x > 3) : y = x + 1/(x-3) → y ≥ 5 :=
sorry

end min_value_of_y_l299_299909


namespace compare_shaded_areas_l299_299879

-- Definitions of shaded areas for each square
def shaded_area_square_I : ℝ := 2 * (1 / 4)
def shaded_area_square_II : ℝ := 1 * (1 / 4)
def shaded_area_square_III : ℝ := 4 * (1 / 8)

-- Theorems to compare shaded areas
theorem compare_shaded_areas :
  shaded_area_square_I = shaded_area_square_III ∧
  shaded_area_square_I ≠ shaded_area_square_II ∧
  shaded_area_square_III ≠ shaded_area_square_II := 
by 
  sorry

end compare_shaded_areas_l299_299879


namespace part1_part2_l299_299611

def f (x : ℝ) : ℝ := abs (x - 5) + abs (x + 4)

theorem part1 (x : ℝ) : f x ≥ 12 ↔ x ≥ 13 / 2 ∨ x ≤ -11 / 2 :=
by
    sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x - 2 ^ (1 - 3 * a) - 1 ≥ 0) ↔ -2 / 3 ≤ a :=
by
    sorry

end part1_part2_l299_299611


namespace variance_of_sample_data_l299_299389

def sample_data : List ℝ := [9.4, 9.7, 9.8, 10.3, 10.8]

def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def variance (data : List ℝ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x - μ)^2)).sum / data.length

theorem variance_of_sample_data :
  variance sample_data = 0.244 := by
  sorry

end variance_of_sample_data_l299_299389


namespace floor_equality_iff_l299_299167

variable (x : ℝ)

theorem floor_equality_iff :
  (⌊3 * x + 4⌋ = ⌊5 * x - 1⌋) ↔
  (11 / 5 ≤ x ∧ x < 7 / 3) ∨
  (12 / 5 ≤ x ∧ x < 13 / 5) ∨
  (17 / 5 ≤ x ∧ x < 18 / 5) := by
  sorry

end floor_equality_iff_l299_299167


namespace sum_odd_numbers_eq_square_l299_299801

theorem sum_odd_numbers_eq_square (n : ℕ) (h : n > 0) : 
  (∑ k in range n, (2 * k + 1)) = n^2 :=
by
  sorry

end sum_odd_numbers_eq_square_l299_299801


namespace train_speed_proof_l299_299834

noncomputable def train_speed_kmh (length_train : ℝ) (time_crossing : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  let relative_speed := length_train / time_crossing
  let train_speed_ms := relative_speed - man_speed_ms
  train_speed_ms * (3600 / 1000)

theorem train_speed_proof :
  train_speed_kmh 150 8 7 = 60.5 :=
by
  sorry

end train_speed_proof_l299_299834


namespace evaluate_at_2_l299_299524

-- Define the polynomial function using Lean
def f (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 + x + 1

-- State the theorem that f(2) evaluates to 35 using Horner's method
theorem evaluate_at_2 : f 2 = 35 := by
  sorry

end evaluate_at_2_l299_299524


namespace equation_of_line_with_slope_tan60_y_intercept_minus1_l299_299745

noncomputable def tan_60 : ℝ := Real.tan (Real.pi / 3)

theorem equation_of_line_with_slope_tan60_y_intercept_minus1 :
  ∀ (x y : ℝ), (y = tan_60 * x - 1) ↔ (sqrt 3 * x - y - 1 = 0) :=
by
  intros x y
  split
  { intro h1
    sorry
  }
  { intro h2
    sorry
  }

end equation_of_line_with_slope_tan60_y_intercept_minus1_l299_299745


namespace triangle_ABCs_satisfies_identity_l299_299672

theorem triangle_ABCs_satisfies_identity (A B C : ℝ) (a b c S : ℝ) 
  (hA : A = 60) (hb : b = 1) (hS : S = Math.sqrt 3)
  (hArea : S = (1/2) * b * c * Real.sin A)
  (hCosineLaw : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) :
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = (2 * Math.sqrt 39) / 3 :=
begin
  sorry
end

end triangle_ABCs_satisfies_identity_l299_299672


namespace aluminum_carbonate_molecular_weight_l299_299447

theorem aluminum_carbonate_molecular_weight :
  let atomic_weight_Al := 26.98
  let atomic_weight_C := 12.01
  let atomic_weight_O := 16.00
  let num_Al := 2
  let num_C := 3
  let num_O := 9
  let total_weight := (num_Al * atomic_weight_Al) + (num_C * atomic_weight_C) + (num_O * atomic_weight_O)
in total_weight = 233.99 := 
sorry

end aluminum_carbonate_molecular_weight_l299_299447


namespace term_2007_in_sequence_is_4_l299_299014

-- Definition of the function to compute the sum of the squares of the digits of a number
def sum_of_squares_of_digits (n : Nat) : Nat := 
  n.digits.sum (λ d => d * d)

-- Definition of the sequence based on the given rules
def sequence : Nat → Nat
| 0 => 2007
| (n + 1) => sum_of_squares_of_digits (sequence n)

-- Theorem stating that the 2007th term in the sequence is 4
theorem term_2007_in_sequence_is_4 : sequence 2007 = 4 :=
  sorry -- Proof skipped

end term_2007_in_sequence_is_4_l299_299014


namespace leak_empties_tank_l299_299789

theorem leak_empties_tank :
  let P := 1 / 3 -- Filling rate by the pump in tanks per hour
  let combined_rate := 1 / 3.5 -- Effective filling rate with the leak in tanks per hour
  let L := P - combined_rate -- Leak emptying rate in tanks per hour
  L = 1 / 21 → 
  (1 / L) = 21 :=
by {
  intros,
  sorry
}

end leak_empties_tank_l299_299789


namespace polynomial_remainder_l299_299899

theorem polynomial_remainder (x : ℝ) : 
  polynomial.eval x (polynomial.X ^ 1010) % polynomial.mul (polynomial.X ^ 2 - 1) (polynomial.X + 1) = 1 :=
sorry

end polynomial_remainder_l299_299899


namespace percentage_of_white_chips_l299_299767

theorem percentage_of_white_chips (T : ℕ) (h1 : 3 = 10 * T / 100) (h2 : 12 = 12): (15 / T * 100) = 50 := by
  sorry

end percentage_of_white_chips_l299_299767


namespace hcf_third_fraction_l299_299052

theorem hcf_third_fraction (x y : ℕ) :
  let f1 := (2 : ℚ) / 3,
      f2 := (4 : ℚ) / 9,
      f3 := (x : ℚ) / y,
      hcf_fractions := (1 : ℚ) / 9
  in f1.gcd f2.gcd f3 = hcf_fractions → f3 = (2 : ℚ) / 9 :=
sorry

end hcf_third_fraction_l299_299052


namespace first_term_is_sqrt9_l299_299035

noncomputable def geometric_first_term (a r : ℝ) : ℝ :=
by
  have h1 : a * r^2 = 3 := by sorry
  have h2 : a * r^4 = 27 := by sorry
  have h3 : (a * r^4) / (a * r^2) = 27 / 3 := by sorry
  have h4 : r^2 = 9 := by sorry
  have h5 : r = 3 ∨ r = -3 := by sorry
  have h6 : (a * 9) = 3 := by sorry
  have h7 : a = 1/3 := by sorry
  exact a

theorem first_term_is_sqrt9 : geometric_first_term 3 9 = 3 :=
by
  sorry

end first_term_is_sqrt9_l299_299035


namespace min_x_coordinate_midpoint_l299_299921

theorem min_x_coordinate_midpoint (a b m : ℝ) (h1 : m > (2 * b^2) / a) :
  ∃ Mx : ℝ, Mx = (a * (m + 2 * a)) / (2 * sqrt (a^2 + b^2)) :=
sorry

end min_x_coordinate_midpoint_l299_299921


namespace functional_equation_implies_identity_l299_299716

theorem functional_equation_implies_identity 
  (f : ℝ → ℝ) 
  (hf : ∀ x y : ℝ, 0 < x → 0 < y → 
    f ((x + y) / 2) + f ((2 * x * y) / (x + y)) = f x + f y) 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  : 2 * f (Real.sqrt (x * y)) = f x + f y := sorry

end functional_equation_implies_identity_l299_299716


namespace math_problem_l299_299331

   noncomputable def x : ℝ := (3 + Real.sqrt 8)^100
   def n : ℤ := Int.floor x
   def f : ℝ := x - n

   theorem math_problem : x * (1 - f) = 1 := 
   by 
      -- x is defined as (3 + sqrt 8)^100
      -- n is defined as floor x
      -- f is defined as x - n
      sorry
   
end math_problem_l299_299331


namespace eccentricity_hyperbola_equations_l299_299195

noncomputable def hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1^2 / a^2 - p.2^2 / b^2 = 1}

variable (a b c e : ℝ)
variable (P Q F : ℝ × ℝ)
variable (chord_length : ℝ)

-- Conditions
axiom hyperbola_eq : ∀ (a b x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → Set (ℝ × ℝ)
axiom line_intersection : ∀ (a c x : ℝ), x = a^2 / c → ∃ y, y = (b / a) * x
axiom focus : ∃ F : ℝ × ℝ, F.1 = c ∧ F.2 = 0
axiom equilateral_triangle : c - a^2 / c = √3 * ab / c → b = √3 * a
axiom chord : line_chord_length y = ax + b → chord_length = b^2 * e^2 / a

-- Proof Problems
theorem eccentricity : ∀ (a b c : ℝ), a > 0 → b = √3 * a → eccentricity = √(1 + b^2 / a^2) → eccentricity = 2 := sorry

theorem hyperbola_equations : 
  ∀ (a b : ℝ), (y = ax + b) → chord_length = (b^2 * e^2 / a) → 
  (13 * a ^ 4 - 77 * a^2 + 102 = 0) → 
  (∃ a b, (a^2 = 2 ∧ b^2 = 6 ∧ hyperbola_eq a b) ∨ (a^2 = 51/13 ∧ b^2 = 153/13 ∧ hyperbola_eq a b)) := sorry

end eccentricity_hyperbola_equations_l299_299195


namespace football_team_progress_l299_299063

theorem football_team_progress :
  let loss := -5
  let gain := 13
  loss + gain = 8 := 
by
  let loss := -5
  let gain := 13
  show loss + gain = 8 from
  sorry

end football_team_progress_l299_299063


namespace limit_perimeters_eq_l299_299145

universe u

noncomputable def limit_perimeters (s : ℝ) : ℝ :=
  let a := 4 * s
  let r := 1 / 2
  a / (1 - r)

theorem limit_perimeters_eq (s : ℝ) : limit_perimeters s = 8 * s := by
  sorry

end limit_perimeters_eq_l299_299145


namespace slow_train_speed_l299_299410

/-- Given the conditions of two trains traveling towards each other and their meeting times,
     prove the speed of the slow train. -/
theorem slow_train_speed :
  let distance_AB := 901
  let slow_train_departure := 5 + 30 / 60 -- 5:30 AM in decimal hours
  let fast_train_departure := 9 + 30 / 60 -- 9:30 AM in decimal hours
  let meeting_time := 16 + 30 / 60 -- 4:30 PM in decimal hours
  let fast_train_speed := 58 -- speed in km/h
  let slow_train_time := meeting_time - slow_train_departure
  let fast_train_time := meeting_time - fast_train_departure
  let fast_train_distance := fast_train_speed * fast_train_time
  let slow_train_distance := distance_AB - fast_train_distance
  let slow_train_speed := slow_train_distance / slow_train_time
  slow_train_speed = 45 := sorry

end slow_train_speed_l299_299410


namespace number_of_terms_is_13_l299_299259

-- Define sum of first three terms
def sum_first_three (a d : ℤ) : ℤ := a + (a + d) + (a + 2 * d)

-- Define sum of last three terms when the number of terms is n
def sum_last_three (a d : ℤ) (n : ℕ) : ℤ := (a + (n - 3) * d) + (a + (n - 2) * d) + (a + (n - 1) * d)

-- Define sum of all terms in the sequence
def sum_all_terms (a d : ℤ) (n : ℕ) : ℤ := n / 2 * (2 * a + (n - 1) * d)

-- Given conditions
def condition_one (a d : ℤ) : Prop := sum_first_three a d = 34
def condition_two (a d : ℤ) (n : ℕ) : Prop := sum_last_three a d n = 146
def condition_three (a d : ℤ) (n : ℕ) : Prop := sum_all_terms a d n = 390

-- Theorem to prove that n = 13
theorem number_of_terms_is_13 (a d : ℤ) (n : ℕ) :
  condition_one a d →
  condition_two a d n →
  condition_three a d n →
  n = 13 :=
by sorry

end number_of_terms_is_13_l299_299259


namespace parts_of_water_in_original_solution_l299_299105

theorem parts_of_water_in_original_solution :
  ∀ (W : ℝ),
  (∀ (L W : ℝ), L = 7 →
    let new_total := L + W + 2.1428571428571423 in
    (L / new_total = 0.20) →
    W = 25.857142857142854) :=
begin
  sorry
end

end parts_of_water_in_original_solution_l299_299105


namespace find_angle_B_find_S_n_l299_299266

-- Part I

theorem find_angle_B 
  (a b c : ℝ)
  (C : ℝ := 2 * Real.pi / 3)
  (h1 : a^2 - (b - c)^2 = (2 - Real.sqrt 3) * b * c) 
  : ∃ B, B = Real.pi / 6 :=
sorry

-- Part II

theorem find_S_n 
  (a_n : ℕ → ℝ)
  (B : ℝ := Real.pi / 6)
  (h2 : (a_n 1) * Real.cos (2 * B) = 1)
  (h3 : {a_n 2, a_n 4, a_n 8} form_geometric_sequence)
  : ∃ (S_n : ℕ → ℝ), S_n = λ n, n.toReal / (n.toReal + 1) :=
sorry

end find_angle_B_find_S_n_l299_299266


namespace smallest_k_l299_299782

theorem smallest_k (k: ℕ) : k > 1 ∧ (k % 23 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) → k = 484 :=
sorry

end smallest_k_l299_299782


namespace log_equality_l299_299122

theorem log_equality : 2 * log 5 10 + log 5 0.25 = 2 := 
sorry

end log_equality_l299_299122


namespace vector_problem_l299_299983

def vec_add (u v : ℕ × ℕ) : ℕ × ℕ :=
  (u.1 + v.1, u.2 + v.2)

def vec_scalar_mult (c : ℕ) (v : ℕ × ℕ) : ℕ × ℕ :=
  (c * v.1, c * v.2)

def vec_sub (u v : ℕ × ℕ) : ℕ × ℕ :=
  (u.1 - v.1, u.2 - v.2)

theorem vector_problem : 
  let a := (1, 2) in 
  let b := (3, 4) in 
  vec_sub (vec_scalar_mult 2 a) b = (-1, 0) :=
by
  let a := (1, 2)
  let b := (3, 4)
  simp [vec_scalar_mult, vec_sub]
  sorry

end vector_problem_l299_299983


namespace statement_a_correct_statement_b_correct_l299_299293

open Real

theorem statement_a_correct (a b c : ℝ) (ha : a > b) (hc : c < 0) : a + c > b + c := by
  sorry

theorem statement_b_correct (a b : ℝ) (ha : a > b) (hb : b > 0) : (a + b) / 2 > Real.sqrt (a * b) := by
  sorry

end statement_a_correct_statement_b_correct_l299_299293


namespace ordered_triples_count_l299_299824

def is_similar (a b c : ℕ) (x y z : ℕ): Prop :=
  x * b = a * y ∧ y * c = b * z ∧ x * c = a * z

def valid_triple (a b c : ℕ) : Prop :=
  0 < a ∧ a ≤ b ∧ b ≤ c ∧ b = 1995 ∧ ∃ P' a b c. is_similar a b c x y z

theorem ordered_triples_count :
  let b := 1995
  ∃ n : ℕ, n = 24 ∧ ∀ (a c : ℕ), valid_triple a b c → n = 24 :=
sorry

end ordered_triples_count_l299_299824


namespace length_of_AB_parallel_to_x_l299_299594

-- Definitions for the coordinates of points A and B
def point (α : Type*) := prod α α
def A (a : ℝ) : point ℝ := (3, a + 3)
def B (a : ℝ) : point ℝ := (a, 4)

-- Condition that AB is parallel to the x-axis
def is_parallel_to_x_axis (a : ℝ) : Prop := (snd (A a)) = (snd (B a))

-- Function to calculate the length of segment AB
def length_of_AB (a : ℝ) : ℝ := abs (fst (A a) - fst (B a))

-- The theorem to verify the length of segment AB
theorem length_of_AB_parallel_to_x (a : ℝ) (h : is_parallel_to_x_axis a) : length_of_AB a = 2 := by
  sorry

end length_of_AB_parallel_to_x_l299_299594


namespace count_polynomials_l299_299878

open Nat

noncomputable def num_of_polynomials (deg : ℕ) (coeff_set : Finset ℕ) (value : ℕ) : ℕ :=
  let coeffs := coeff_set.toList,
  let solutions := { P : (Fin 4) → ℕ | 
                      ∀ i, P i ∈ coeffs ∧ 
                      P 0 + P 1 - P 2 - P 3 = -value },
  solutions.card

theorem count_polynomials : 
  num_of_polynomials 3 (Finset.range 10) 9 = 220 := 
  sorry

end count_polynomials_l299_299878


namespace parallel_distance_eq_zero_l299_299142

open Real

def a : ℝ × ℝ := (3, -4)
def b : ℝ × ℝ := (2, -1)
def d : ℝ × ℝ := (-1, 3)
def v : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def dot_prod (x y: ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2
def p : ℝ × ℝ := (dot_prod v d / dot_prod d d) * d

theorem parallel_distance_eq_zero (a b d v p : ℝ × ℝ) : (v - p) = (0, 0) := by
  sorry

end parallel_distance_eq_zero_l299_299142


namespace expression_evaluation_l299_299544

noncomputable def sum_odd_sequence : ℕ :=
  let n := (2025 - 5) / 2 + 1 in
  (n / 2) * (5 + 2025)

noncomputable def sum_even_sequence : ℕ :=
  let n := (2024 - 4) / 2 + 1 in
  (n / 2) * (4 + 2024)

noncomputable def sum_geometric_sequence : ℕ :=
  let n := 11 in
  2 * (2^n - 1)

noncomputable def evaluate_expression : ℕ :=
  sum_odd_sequence - sum_even_sequence + sum_geometric_sequence

theorem expression_evaluation : evaluate_expression = 5104 := by
  sorry

end expression_evaluation_l299_299544


namespace sum_inequality_of_pairwise_distinct_l299_299935

-- Define the necessary conditions
def is_pairwise_distinct (S : ℕ → ℕ) := ∀ (i j : ℕ), i ≠ j → S i ≠ S j

theorem sum_inequality_of_pairwise_distinct {a : ℕ → ℕ}
  (h1 : is_pairwise_distinct a) (n : ℕ) (h2 : 0 < n) : 
  (∑ k in Finset.range n + 1, (a (k+1) : ℝ) / ( (k+1) : ℝ)^2) ≥ (∑ k in Finset.range n + 1, 1 / ( (k+1) : ℝ) ) :=
sorry

end sum_inequality_of_pairwise_distinct_l299_299935


namespace domain_of_f_l299_299444

def f (x : ℝ) : ℝ := (x^2 - 49) / (x - 7)

theorem domain_of_f :
  {x : ℝ | f x ≠ real.div_zero} = {x : ℝ | x ≠ 7} :=
by
  sorry

end domain_of_f_l299_299444


namespace curve_cartesian_eq_min_value_intersection_l299_299996

theorem curve_cartesian_eq :
  (set_of (λ (p : ℝ × ℝ), ∃ (θ : ℝ), (p.1^2 + (p.2 - 3)^2 = 9))) = 
  (set_of (λ (p : ℝ × ℝ), ∃ (ρ θ : ℝ), (ρ = 6 * real.sin θ) ∧ (p.1 = ρ * real.cos θ) ∧ (p.2 = ρ * real.sin θ))) :=
sorry

theorem min_value_intersection (α : ℝ) (t : ℝ → ℝ) (P : ℝ × ℝ) (A B : ℝ × ℝ):
  (P = (1, 2)) →
  (A ≠ B) →
  (0 ≤ α ∧ α ≤ π) →
  (A = (1 + t(0) * real.cos α, 2 + t(0) * real.sin α)) →
  (B = (1 + t(1) * real.cos α, 2 + t(1) * real.sin α)) →
  (∀ t, (A.1^2 + (A.2 - 3)^2 = 9) ∧ (B.1^2 + (B.2 - 3)^2 = 9)) →
  (∃ t₁ t₂, (t₁ + t₂ = -2 * (real.cos α - real.sin α))
           ∧ (t₁ * t₂ = -7)
           ∧ (∀ P A B, |PA| + |PB| = sqrt ((t₁ + t₂)^2 - 4 * t₁ * t₂)))
           ∧ (min_value_intersection t₁ t₂ = 2 * real.sqrt 7 / 7)
:=
sorry

end curve_cartesian_eq_min_value_intersection_l299_299996


namespace find_odd_x_l299_299751

def is_odd (n : Int) : Prop := n % 2 = 1

def median (s : List Real) : Real :=
  let sorted := s.qsort (≤)
  let len := sorted.length
  if len % 2 = 0 then
    (sorted.get! (len / 2 - 1) + sorted.get! (len / 2)) / 2
  else
    sorted.get! (len / 2)

def mean (s : List Real) : Real :=
  s.sum / s.length

theorem find_odd_x (x : ℤ) (h1 : List.intro 18 34 50 x 16 22) 
(h2 : is_odd x) : 
median [18, 34, 50, x, 16, 22] = mean [18, 34, 50, x, 16, 22] - 7 ↔ x = 21 := by
  sorry

end find_odd_x_l299_299751


namespace trap_speed_independent_of_location_l299_299735

theorem trap_speed_independent_of_location 
  (h b a : ℝ) (v_mouse : ℝ) 
  (path_length : ℝ := Real.sqrt (a^2 + (3*h)^2)) 
  (T : ℝ := path_length / v_mouse) 
  (step_height : ℝ := h) 
  (v_trap : ℝ := step_height / T) 
  (h_val : h = 3) 
  (b_val : b = 1) 
  (a_val : a = 8) 
  (v_mouse_val : v_mouse = 17) : 
  v_trap = 8 := by
  sorry

end trap_speed_independent_of_location_l299_299735


namespace percentage_of_boys_l299_299486

variable (B : ℝ) -- percentage of boys (real number since it's a percentage).

theorem percentage_of_boys (h1 : ∀ b, b ∈ ℝ ∧ b = 80) -- Every boy gets an 80% score.
  (h2 : ∀ g, g ∈ ℝ ∧ g = 90) -- Every girl gets a 90% score.
  (h3 : ∀ avg, avg ∈ ℝ ∧ avg = 86) -- The class average is 86%.
  (h4 : ∀ B, 0 <= B ∧ B <= 100) -- Boys percentage is between 0% and 100%.
  (h5 : ∀ G, G = 100 - B)      -- Girls percentage is (100% - Boys percentage).
  : B = 40 := 
sorry

end percentage_of_boys_l299_299486


namespace ratio_area_squares_l299_299379

theorem ratio_area_squares 
  (A B C D Q R S T : ℝ)
  (len_ABCD : ℝ := 12)
  (SQRT : sq_root len_ABCD 2)
  (on_sides : Set) 
  (H1 : Q ∈ AB) 
  (H2 : AQ = 3 * QB) 
  (H3 : (12:ℝ) = 12)
  (H4: len_ABCD > 0):
  let ABCD_area := len_ABCD^2
  let QRST_area := (3√2 * (1/2/3 * len_ABCD))^2
  let ratio := QRST_area / ABCD_area
  ratio = 1 / 8 := by sorry

end ratio_area_squares_l299_299379


namespace plane_equation_l299_299495

noncomputable def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2 * s - 3 * t, 4 - 2 * s, 5 + s + 3 * t)

theorem plane_equation :
  ∃ (A B C D : ℤ),
    (∀ (s t : ℝ),
      let (x, y, z) := parametric_plane s t
      in A * x + B * y + C * z + D = 0) ∧
    A = 2 ∧ B = 3 ∧ C = 2 ∧ D = -26 ∧
    Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1 :=
begin
  sorry
end

end plane_equation_l299_299495


namespace number_of_convex_polygons_l299_299538

noncomputable def is_polygon_inscribed_in_unit_circle (sides : List ℝ) : Prop :=
  ∀ side ∈ sides, 
    ∃ n : ℤ, side = Real.sqrt n ∧ 
    all_sides_less_than_two sides ∧
    satisfies_triangle_inequality sides

theorem number_of_convex_polygons (n : ℕ) :
  (∃ (polygons : Finset (Finset ℝ)), 
    (∀ polygon ∈ polygons, 
       (polygon.card = n) ∧ 
       is_polygon_inscribed_in_unit_circle polygon ∧ 
       cons_rotations_reflection_as_same polygons) 
    ∧ polygons.card = 14) :=
sorry

end number_of_convex_polygons_l299_299538


namespace math_problem_l299_299634

theorem math_problem (x y : Int)
  (hx : x = 2 - 4 + 6)
  (hy : y = 1 - 3 + 5) :
  x - y = 1 :=
by
  sorry

end math_problem_l299_299634


namespace max_circle_radius_l299_299663

theorem max_circle_radius :
  ∃ (C : set (ℝ × ℝ)), (C (4, 0) ∧ C (-4, 0)) ∧ (∀ (x y : ℝ), C (x, y) → ∃ r, r = 4) :=
sorry

end max_circle_radius_l299_299663


namespace count_correct_derivatives_l299_299149

theorem count_correct_derivatives :
  (∀ x : ℝ, ∃ y : ℝ, 
    (y = ln 2 → deriv (λ z, ln 2) y = 0) ∧
    (y = sqrt x → deriv (λ z, sqrt z) y = 1 / (2 * sqrt y)) ∧
    (y = exp (-x) → deriv (λ z, exp (-z)) y = -exp (-y)) ∧
    (y = cos x → deriv (λ z, cos z) y = -sin y)) → 
  (number_of_correct_conclusions = 2) :=
by sorry

end count_correct_derivatives_l299_299149


namespace Paul_seashells_l299_299246

namespace SeashellProblem

variables (P L : ℕ)

def initial_total_seashells (H P L : ℕ) : Prop := H + P + L = 59

def final_total_seashells (H P L : ℕ) : Prop := H + P + L - L / 4 = 53

theorem Paul_seashells : 
  (initial_total_seashells 11 P L) → (final_total_seashells 11 P L) → P = 24 :=
by
  intros h_initial h_final
  sorry

end SeashellProblem

end Paul_seashells_l299_299246


namespace time_for_B_alone_to_complete_work_l299_299808

theorem time_for_B_alone_to_complete_work :
  (∃ (A B C : ℝ), A = 1 / 4 
  ∧ B + C = 1 / 3
  ∧ A + C = 1 / 2)
  → 1 / B = 12 :=
by
  sorry

end time_for_B_alone_to_complete_work_l299_299808


namespace center_rectangle_is_A_l299_299542

def Rectangle : Type := { w : ℕ, x : ℕ, y : ℕ, z : ℕ }

def rectA : Rectangle := ⟨8, 2, 9, 5⟩
def rectB : Rectangle := ⟨2, 1, 5, 8⟩
def rectC : Rectangle := ⟨6, 9, 4, 3⟩
def rectD : Rectangle := ⟨4, 6, 2, 9⟩
def rectE : Rectangle := ⟨9, 5, 6, 1⟩

def sum_wy (r : Rectangle) : ℕ := r.w + r.y

theorem center_rectangle_is_A :
  sum_wy rectA = max (max (sum_wy rectB) (sum_wy rectC)) (max (sum_wy rectD) (sum_wy rectE)) :=
by
  sorry

end center_rectangle_is_A_l299_299542


namespace inscribed_circle_radius_l299_299402

theorem inscribed_circle_radius:
  ∀ (b r: ℝ), 
  let perimeter := 5 * b,
  let area_triangle := (b^2 * (Real.sqrt 3)) / 2 in
  (perimeter: ℝ) = area_triangle →
  r = b * (Real.sqrt 3) / 10 :=
by
  intros b r perimeter area_triangle h_perimeter_area_eq
  -- these assumptions are necessary based on the given conditions
  have h_perimeter_eq: perimeter = 5 * b := by rfl,
  have h_area_triangle_eq: area_triangle = (b^2 * (Real.sqrt 3)) / 2 := by rfl,
  -- sorry to indicate unfinished proof.
  sorry

end inscribed_circle_radius_l299_299402


namespace german_mo_2016_problem_1_l299_299166

theorem german_mo_2016_problem_1 (a b : ℝ) :
  a^2 + b^2 = 25 ∧ 3 * (a + b) - a * b = 15 ↔
  (a = 0 ∧ b = 5) ∨ (a = 5 ∧ b = 0) ∨
  (a = 4 ∧ b = -3) ∨ (a = -3 ∧ b = 4) :=
sorry

end german_mo_2016_problem_1_l299_299166


namespace launderette_machines_l299_299158

def quarters_per_machine := 80
def dimes_per_machine := 100
def total_income := 90
def quarter_value := 0.25
def dime_value := 0.10
def income_per_machine := (quarters_per_machine * quarter_value) + (dimes_per_machine * dime_value)
def num_machines := total_income / income_per_machine

theorem launderette_machines : num_machines = 3 := by
  sorry

end launderette_machines_l299_299158


namespace ordering_of_exponentials_l299_299537

theorem ordering_of_exponentials :
  let A := 3^20
  let B := 6^10
  let C := 2^30
  B < A ∧ A < C :=
by
  -- Definitions and conditions
  have h1 : 6^10 = 3^10 * 2^10 := by sorry
  have h2 : 3^10 = 59049 := by sorry
  have h3 : 2^10 = 1024 := by sorry
  have h4 : 2^30 = (2^10)^3 := by sorry
  
  -- We know 3^20, 6^10, 2^30 by definition and conditions
  -- Comparison
  have h5 : 3^20 = (3^10)^2 := by sorry
  have h6 : 2^30 = 1024^3 := by sorry
  
  -- Combine to get results
  have h7 : (3^10)^2 > 6^10 := by sorry
  have h8 : 1024^3 > 6^10 := by sorry
  have h9 : 1024^3 > (3^10)^2 := by sorry

  exact ⟨h7, h9⟩

end ordering_of_exponentials_l299_299537


namespace min_people_for_three_shared_greek_signs_min_people_for_shared_greek_chinese_signs_l299_299791

-- Part (a)
theorem min_people_for_three_shared_greek_signs (num_greek_signs : ℕ) (num_people : ℕ) : 
  num_greek_signs = 12 → num_people = 25 → 
  ∃ s (h : s ⊆ finset.Ico 1 (num_greek_signs + 1)), 
  (∀ t ⊆ s, t.card > 2) :=
by
  intros,
  sorry

-- Part (b)
theorem min_people_for_shared_greek_chinese_signs (num_greek_signs : ℕ) (num_chinese_signs : ℕ) (num_people : ℕ) : 
  num_greek_signs = 12 → num_chinese_signs = 12 → num_people = 145 → 
  ∃ s (h : s ⊆ finset.Ico 1 (num_greek_signs*num_chinese_signs + 1)), 
  (∀ t ⊆ s, t.card > 1) :=
by
  intros,
  sorry

end min_people_for_three_shared_greek_signs_min_people_for_shared_greek_chinese_signs_l299_299791


namespace inscribed_ngon_theorem_l299_299408

noncomputable def inscribed_ngon_exists (n : ℕ) (a : Fin n → ℝ) : Prop :=
  (∀ i, 0 < a i) ∧ (∀ i, 2 * a i < (Finset.univ.sum a)) →
  ∃ (vertices : Fin n → ℝ × ℝ),
    -- condition to be vertices of an inscribed n-gon with side lengths a_1, ..., a_n
    ∀ i, dist (vertices i) (vertices ((i + 1) % n)) = a i

-- The theorem statement
theorem inscribed_ngon_theorem (n : ℕ) (a : Fin n → ℝ) :
  (∀ i, 0 < a i) → (∀ i, 2 * a i < (Finset.univ.sum a)) →
  ∃ (vertices : Fin n → ℝ × ℝ),
    ∀ i, dist (vertices i) (vertices ((i + 1) % n)) = a i := 
begin
  intros h_pos h_cond,
  sorry -- proof to be completed
end

end inscribed_ngon_theorem_l299_299408


namespace incorrect_option_B_l299_299514

noncomputable def Sn : ℕ → ℝ := sorry
-- S_n is the sum of the first n terms of the arithmetic sequence

axiom S5_S6 : Sn 5 < Sn 6
axiom S6_eq_S_gt_S8 : Sn 6 = Sn 7 ∧ Sn 7 > Sn 8

theorem incorrect_option_B : ¬ (Sn 9 < Sn 5) := sorry

end incorrect_option_B_l299_299514


namespace shirts_total_cost_l299_299681

def shirt_cost_problem : Prop :=
  ∃ (first_shirt_cost second_shirt_cost total_cost : ℕ),
    first_shirt_cost = 15 ∧
    first_shirt_cost = second_shirt_cost + 6 ∧
    total_cost = first_shirt_cost + second_shirt_cost ∧
    total_cost = 24

theorem shirts_total_cost : shirt_cost_problem := by
  sorry

end shirts_total_cost_l299_299681


namespace incorrect_statement_l299_299839

-- Define properties and statements related to triangles
variables (T1 T2 : Triangle)
variables (h1 : T1 ≅ T2 → T1.altitudes = T2.altitudes)
variables (h2 : T1 ≅ T2 → T1.area = T2.area)
variables (h3 : T1 ≅ T2 → T1.perimeter = T2.perimeter)

-- Define the statement we want to prove is incorrect
theorem incorrect_statement (p : T1.perimeter = T2.perimeter) : ¬ (T1 ≅ T2) :=
sorry

end incorrect_statement_l299_299839


namespace log_equality_l299_299123

theorem log_equality : 2 * log 5 10 + log 5 0.25 = 2 := 
sorry

end log_equality_l299_299123


namespace div_decimal_l299_299137

theorem div_decimal (a b : ℝ)  (h₁ : a = 0.45) (h₂ : b = 0.005):
  a / b = 90 :=
by {
  sorry
}

end div_decimal_l299_299137


namespace third_circle_radius_l299_299528

/--
Given a circle of radius 3 centered at A, and another circle of radius 6 centered at B,
where the two circles are externally tangent. A third circle is tangent to the first two circles
and to one of their common external tangents. Prove that the radius of this third circle is 3.
-/
theorem third_circle_radius (A B : Point) (r1 r2 : ℝ) (hA : Circle A r1) (hB : Circle B r2)
  (h_ext_tangent : externally_tangent hA hB) (r3 : ℝ) (hTangent1 : tangent hA (Circle ⟨A.x + A.y + 9, B.y + B.y, 0⟩ r3))
  (hTangent2 : tangent hB (Circle ⟨A.x - 3, B.y + 6, r3⟩)) :
  (r3 = 3) :=
  sorry

end third_circle_radius_l299_299528


namespace cricket_innings_l299_299811

theorem cricket_innings (n : ℕ) 
  (average_run : ℕ := 40) 
  (next_innings_run : ℕ := 84) 
  (new_average_run : ℕ := 44) :
  (40 * n + 84) / (n + 1) = 44 ↔ n = 10 := 
by
  sorry

end cricket_innings_l299_299811


namespace last_integer_in_sequence_l299_299413

noncomputable def sequence (n : ℕ) : ℝ :=
  2000000 * (1 / 5) ^ n

theorem last_integer_in_sequence : ∃ n : ℕ, sequence n = 128 ∧ ∀ m : ℕ, m > n → sequence m ∉ ℕ :=
by
  sorry

end last_integer_in_sequence_l299_299413


namespace largest_A_smallest_A_l299_299845

noncomputable def is_coprime_with_12 (n : Nat) : Prop :=
  Nat.gcd n 12 = 1

noncomputable def rotated_number (n : Nat) : Option Nat :=
  if n < 10^7 then none else
  let b := n % 10
  let k := n / 10
  some (b * 10^7 + k)

noncomputable def satisfies_conditions (B : Nat) : Prop :=
  B > 44444444 ∧ is_coprime_with_12 B

theorem largest_A :
  ∃ (B : Nat), satisfies_conditions B ∧ rotated_number B = some 99999998 :=
sorry

theorem smallest_A :
  ∃ (B : Nat), satisfies_conditions B ∧ rotated_number B = some 14444446 :=
sorry

end largest_A_smallest_A_l299_299845


namespace chess_tournament_participants_l299_299270

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 136) : n = 17 :=
by {
  sorry -- Proof will be here.
}

end chess_tournament_participants_l299_299270


namespace solve_for_x_l299_299550

theorem solve_for_x : ∀ {x : ℝ}, log 9 (3 * x - 15) = 5 / 2 → x = 86 :=
by
  intros x h
  have : 3 * x - 15 = 9 ^ (5 / 2) := by sorry
  have : 9 ^ (5 / 2) = 243 := by sorry
  have : 3 * x - 15 = 243 := by sorry
  have : 3 * x = 258 := by linarith
  have : x = 258 / 3 := by linarith
  have : x = 86 := by norm_num
  exact this

end solve_for_x_l299_299550


namespace at_least_1971_are_1_l299_299915

noncomputable def proof_problem : Prop :=
  ∃ (x : Fin 1981 → ℕ) (k : ℕ), 
    k ≤ 1970 ∧
    (x 0 > 1 → x 0 + x 1 + k = (∏ i, x i)) ∧
    ∀ i, i < 1981 → x i > 0 ∧
    ∃ (a : ℕ), a ∈ {2, 3, 4, 5, 6} ∧ (x 0 = a ∧ x 1 = (a + 1979) / (a - 1))

theorem at_least_1971_are_1 : proof_problem := by
  sorry

end at_least_1971_are_1_l299_299915


namespace four_digit_numbers_exist_l299_299247

theorem four_digit_numbers_exist :
  let even_digits := [2, 4, 6, 8] in
  let odd_digits := [1, 3, 5, 7, 9] in
  ∃ (f s t fo : ℕ),
    f ∈ even_digits ∧
    s ∈ odd_digits ∧
    t ∈ (finset.range 10).erase f ∧ t ≠ s ∧
    fo = (f + s) % 10 ∧ fo ≠ f ∧ fo ≠ s ∧ fo ≠ t ∧
    (finset.card $ do
      f' ← finset.univ.filter (λ x, x ∈ even_digits),
      s' ← finset.univ.filter (λ y, y ∈ odd_digits),
      t' ← finset.univ.filter (λ z, z ∉ [f', s']),
      fo' ← finset.univ.filter (λ w, w = (f' + s') % 10),
      if fo' ≠ f' ∧ fo' ≠ s' ∧ fo' ≠ t'
      then ([f', s', t', fo']).to_finset
      else finset.empty) = 160 :=
begin
  sorry -- Proof not required
end

end four_digit_numbers_exist_l299_299247


namespace compare_powers_l299_299136

theorem compare_powers:
  (2 ^ 2023) * (7 ^ 2023) < (3 ^ 2023) * (5 ^ 2023) :=
  sorry

end compare_powers_l299_299136


namespace best_ketchup_deal_l299_299859

/-- Given different options of ketchup bottles with their respective prices and volumes as below:
 - Bottle 1: 10 oz for $1
 - Bottle 2: 16 oz for $2
 - Bottle 3: 25 oz for $2.5
 - Bottle 4: 50 oz for $5
 - Bottle 5: 200 oz for $10
And knowing that Billy's mom gives him $10 to spend entirely on ketchup,
prove that the best deal for Billy is to buy one bottle of the $10 ketchup which contains 200 ounces. -/
theorem best_ketchup_deal :
  let price := [1, 2, 2.5, 5, 10]
  let volume := [10, 16, 25, 50, 200]
  let cost_per_ounce := [0.1, 0.125, 0.1, 0.1, 0.05]
  ∃ i, (volume[i] = 200) ∧ (price[i] = 10) ∧ (∀ j, cost_per_ounce[i] ≤ cost_per_ounce[j]) ∧ (price.sum = 10) :=
by
  sorry

end best_ketchup_deal_l299_299859


namespace tangent_lines_inequality_l299_299398

theorem tangent_lines_inequality (k k1 k2 b b1 b2 : ℝ)
  (h1 : k = - (b * b) / 4)
  (h2 : k1 = - (b1 * b1) / 4)
  (h3 : k2 = - (b2 * b2) / 4)
  (h4 : b = b1 + b2) :
  k ≥ 2 * (k1 + k2) := sorry

end tangent_lines_inequality_l299_299398


namespace num_ways_remove_11x11_from_2011x2011_l299_299250

theorem num_ways_remove_11x11_from_2011x2011 :
  let n := 2011
  let m := 11
  let valid_tiling_removals := (⌈(n - m + 1)^2 / 2⌉)
  valid_tiling_removals = 2_002_001 :=
sorry

end num_ways_remove_11x11_from_2011x2011_l299_299250


namespace frac_div_l299_299051

theorem frac_div : (3 / 7) / (4 / 5) = 15 / 28 := by
  sorry

end frac_div_l299_299051


namespace number_of_customers_before_lunch_rush_l299_299835

-- Defining the total number of customers during the lunch rush
def total_customers_during_lunch_rush : ℕ := 49 + 2

-- Defining the number of additional customers during the lunch rush
def additional_customers : ℕ := 12

-- Target statement to prove
theorem number_of_customers_before_lunch_rush : total_customers_during_lunch_rush - additional_customers = 39 :=
  by sorry

end number_of_customers_before_lunch_rush_l299_299835


namespace find_y_for_projection_l299_299022

theorem find_y_for_projection :
  ∃ y : ℝ, (∑ i, !([1, y, 5].nth i) *! ([4, -3, 2].nth i)) / (∑ i, !([4, -3, 2].nth i) *! ([4, -3, 2].nth i)) = 1 / 7 :=
sorry

end find_y_for_projection_l299_299022


namespace fraction_simplification_l299_299424

theorem fraction_simplification : 
  let m := 1
  let n := 210
  (1 + 210) = 211 := by
sory

end fraction_simplification_l299_299424


namespace find_m_of_g_l299_299566

theorem find_m_of_g (y : ℝ) :
  let g := λ y, Real.cot (y / 6) - Real.cot (y / 2)
  ∃ m, ∀ y, g y = (Real.sin (m * y)) / (Real.sin (y / 6) * Real.sin (y / 2)) :=
sorry

end find_m_of_g_l299_299566


namespace nice_functions_at_least_n_to_the_n_l299_299333

-- Definitions for the given conditions
variables {X : Type} {m n : ℕ}
variables (X_i : fin m → set X) [fintype X]
variables (n_ge_2 : n ≥ 2) (X_elements : fintype.card X = n)
variables (distinct_nonempty_subsets : ∀ i, X_i i ≠ ∅ ∧ ∀ j, i ≠ j → X_i i ≠ X_i j)

-- Definition for a "nice" function
def is_nice_function (f : X → fin (n+2)) := ∃ k : fin m, 
  (∀ i : fin m, i ≠ k → ∑ x in X_i k, f x > ∑ x in X_i i, f x)

-- The theorem statement
theorem nice_functions_at_least_n_to_the_n : 
  ∃ (nice_functions : finset (X → fin (n+2))), 
  (∀ f ∈ nice_functions, is_nice_function X_i f) ∧
  finset.card nice_functions ≥ n^n := 
sorry

end nice_functions_at_least_n_to_the_n_l299_299333


namespace range_of_a_l299_299590

noncomputable def f : ℝ → ℝ := sorry

-- Define the conditions
def condition_1 (x : ℝ) : Prop := true -- implicit in definition
def condition_2 (x : ℝ) : Prop := f(x + 1) = f(-(x + 2))
def condition_3 (x1 x2 : ℝ) : Prop := 
  x1 ∈ [0, +∞[ ∧ x2 ∈ [0, +∞[ ∧ x1 ≠ x2 → (f(x1) - f(x2)) / (x1 - x2) > -1

-- Given inequality we need to prove
def given_inequality (a : ℝ) : Prop := 
  f(a^2 - 1) + f(a - 1) + a^2 + a > 2

-- The final proof problem
theorem range_of_a (a : ℝ) :
  (∀ x, condition_1 x) →
  (∀ x, condition_2 x) →
  (∀ x1 x2, condition_3 x1 x2) →
  given_inequality a →
  a < -2 ∨ a > 1 := sorry

end range_of_a_l299_299590


namespace probability_f_x_gt_2_l299_299608

noncomputable def f (x : ℝ) : ℝ := 2 ^ x

theorem probability_f_x_gt_2 : 
  let interval := set.Icc (-2 : ℝ) (2 : ℝ) in
  let favorable_interval := set.Ioc (1 : ℝ) (2 : ℝ) in
  let total_length := (2:ℝ) - (-2:ℝ) in
  let favorable_length := (2:ℝ) - 1 in
  (favorable_length / total_length) = (1 / 4) :=
by sorry

end probability_f_x_gt_2_l299_299608


namespace hyperbola_properties_l299_299746

theorem hyperbola_properties :
  (∀ x y : ℝ, (x^2 / 2 - y^2 = 1) →
  (y = x * (sqrt 2 / 2) ∨ y = -x * (sqrt 2 / 2)) ∧
  (∀ a b c : ℝ, a = sqrt(2) ∧ b = 1 ∧ c = sqrt(a^2 + b^2) → c / a = sqrt(6) / 2)) :=
by 
  intro x y h
  split
  {
    sorry  -- Prove the equations of the asymptotes
  }
  {
    intro a b c ha hb hc
    rw ha at *
    rw hb at *
    rw hc at *
    sorry  -- Prove the eccentricity
  }

end hyperbola_properties_l299_299746


namespace max_sum_of_shaded_areas_l299_299115

theorem max_sum_of_shaded_areas (AB AC : ℝ) (hAB : AB = 2) (hAC : AC = 3) : 
  ∃ θ : ℝ, θ = real.pi / 2 ∧ 
  (9 * real.sin θ) = 9 :=
by
  use real.pi / 2
  split
  · exact rfl
  · simp [real.sin_pi_div_two]

end max_sum_of_shaded_areas_l299_299115


namespace tournament_participants_l299_299258

theorem tournament_participants (n : ℕ) (h : (n * (n - 1)) / 2 = 171) : n = 19 :=
by
  sorry

end tournament_participants_l299_299258


namespace compute_fraction_l299_299875

theorem compute_fraction : (1922^2 - 1913^2) / (1930^2 - 1905^2) = (9 : ℚ) / 25 := by
  sorry

end compute_fraction_l299_299875


namespace numerator_equals_denominator_l299_299177

theorem numerator_equals_denominator (x : ℝ) (h : 4 * x - 3 = 5 * x + 2) : x = -5 :=
  by
    sorry

end numerator_equals_denominator_l299_299177


namespace billy_buys_bottle_l299_299865

-- Definitions of costs and volumes
def money : ℝ := 10
def cost1 : ℝ := 1
def volume1 : ℝ := 10
def cost2 : ℝ := 2
def volume2 : ℝ := 16
def cost3 : ℝ := 2.5
def volume3 : ℝ := 25
def cost4 : ℝ := 5
def volume4 : ℝ := 50
def cost5 : ℝ := 10
def volume5 : ℝ := 200

-- Statement of the proof problem
theorem billy_buys_bottle : ∃ b : ℕ, b = 1 ∧ cost5 = money := by 
  sorry

end billy_buys_bottle_l299_299865


namespace coin_flips_probability_exactly_5_heads_in_7_l299_299047

noncomputable def fair_coin_probability (n k : ℕ) : ℚ :=
(nat.choose n k) / (2^n) 

theorem coin_flips_probability_exactly_5_heads_in_7 :
  fair_coin_probability 7 5 = 21 / 128 :=
by
  -- execution of this proof will require the detailed calculation of choose (7, 5) which is 21 and that the total outcomes of 7 flips equals 128
  sorry

end coin_flips_probability_exactly_5_heads_in_7_l299_299047


namespace alex_hours_per_week_l299_299837

theorem alex_hours_per_week
  (summer_earnings : ℕ)
  (summer_weeks : ℕ)
  (summer_hours_per_week : ℕ)
  (academic_year_weeks : ℕ)
  (academic_year_earnings : ℕ)
  (same_hourly_rate : Prop) :
  summer_earnings = 4000 →
  summer_weeks = 8 →
  summer_hours_per_week = 40 →
  academic_year_weeks = 32 →
  academic_year_earnings = 8000 →
  same_hourly_rate →
  (academic_year_earnings / ((summer_earnings : ℚ) / (summer_weeks * summer_hours_per_week)) / academic_year_weeks) = 20 :=
by
  sorry

end alex_hours_per_week_l299_299837


namespace count_negative_terms_sequence_increases_from_fourth_term_minimum_term_at_fourth_l299_299296

def sequence (n : ℕ) : ℤ := n*(n-8) - 20

-- 1. Prove that the sequence has 9 negative terms.
theorem count_negative_terms : ∃ n : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i < 10 → sequence i < 0 ∧ i < 10 :=
sorry

-- 2. Prove that the sequence starts to increase from the fourth term.
theorem sequence_increases_from_fourth_term : ∀ n : ℕ, n ≥ 4 → sequence (n+1) - sequence n > 0 :=
sorry

-- 3. Prove that the sequence has a minimum term at a₄ = -36.
theorem minimum_term_at_fourth : sequence 4 = -36 ∧ (∀ n : ℕ, n ≠ 4 → sequence n ≥ -36) :=
sorry

end count_negative_terms_sequence_increases_from_fourth_term_minimum_term_at_fourth_l299_299296


namespace solve_for_q_l299_299376

theorem solve_for_q (n m q: ℚ)
  (h1 : 3 / 4 = n / 88)
  (h2 : 3 / 4 = (m + n) / 100)
  (h3 : 3 / 4 = (q - m) / 150) :
  q = 121.5 :=
sorry

end solve_for_q_l299_299376


namespace goods_train_passing_time_l299_299820

/-- Given the speeds of a man's train, a goods train, and the length of the goods train,
    prove that the time it takes for the goods train to pass the man is 9 seconds. -/
theorem goods_train_passing_time 
    (speed_mans_train_kmph : ℝ := 70)
    (speed_goods_train_kmph : ℝ := 42)
    (length_goods_train_m : ℝ := 280) :
    let speed_mans_train_mps := speed_mans_train_kmph * (1000 / 3600)
    let speed_goods_train_mps := speed_goods_train_kmph * (1000 / 3600)
    let relative_speed_mps := speed_mans_train_mps + speed_goods_train_mps
    let time_seconds := length_goods_train_m / relative_speed_mps
    in time_seconds ≈ 9 :=
by
  sorry

end goods_train_passing_time_l299_299820


namespace angle_sum_proof_l299_299292

def angle_P : ℝ := 45
def angle_Q : ℝ := 80
def angle_R : ℝ := 40

theorem angle_sum_proof (a b : ℝ) (h₁ : m\angle P = angle_P) (h₂ : m\angle Q = angle_Q) (h₃ : m\angle R = angle_R) : a + b = 15 :=
sorry

end angle_sum_proof_l299_299292


namespace smallest_bob_number_l299_299110

theorem smallest_bob_number (b : ℕ) (h : ∀ p : ℕ, Prime p → p ∣ 30 → p ∣ b) : 30 ≤ b :=
by {
  sorry
}

end smallest_bob_number_l299_299110


namespace sequence_expression_l299_299600

-- Given conditions
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)
variable (h1 : ∀ n, S n = (1/4) * (a n + 1)^2)

-- Theorem statement
theorem sequence_expression (n : ℕ) : a n = 2 * n - 1 :=
sorry

end sequence_expression_l299_299600


namespace gerald_paid_l299_299959

theorem gerald_paid (G : ℝ) (h : 0.8 * G = 200) : G = 250 :=
by
  sorry

end gerald_paid_l299_299959


namespace weight_of_seventh_person_l299_299040

theorem weight_of_seventh_person (total_weight_six: Real) (total_weight_seven: Real) : Real :=
  (total_weight_six = 960) → (total_weight_seven = 1057) → (total_weight_seven - total_weight_six = 97)

end weight_of_seventh_person_l299_299040


namespace last_box_weight_l299_299310

theorem last_box_weight (a b c : ℕ) (h1 : a = 2) (h2 : b = 11) (h3 : a + b + c = 18) : c = 5 :=
by
  sorry

end last_box_weight_l299_299310


namespace pairs_with_green_shirts_l299_299654

theorem pairs_with_green_shirts (r g t p rr_pairs gg_pairs : ℕ)
  (h1 : r = 60)
  (h2 : g = 90)
  (h3 : t = 150)
  (h4 : p = 75)
  (h5 : rr_pairs = 28)
  : gg_pairs = 43 := 
sorry

end pairs_with_green_shirts_l299_299654


namespace sum_log_series_l299_299141

noncomputable def sum_series : ℝ :=
  ∑ k in Finset.range 62 \ Finset.singleton 0,
    Real.log2 (1 + 1 / (k+2)) * (Real.log (2) / Real.log (k+2)) * (Real.log (2) / Real.log (k+3))

theorem sum_log_series :
  sum_series = 5 / 6 :=
by
  sorry

end sum_log_series_l299_299141


namespace machines_work_together_time_l299_299469

theorem machines_work_together_time (rate1 rate2 : ℚ) (h1 : rate1 = 1 / 20) (h2 : rate2 = 1 / 30) :
  (1 / (rate1 + rate2)) = 12 :=
by
  sorry

end machines_work_together_time_l299_299469


namespace cricketer_average_score_l299_299245

variable {A : ℤ} -- A represents the average score after 18 innings

theorem cricketer_average_score
  (h1 : (19 * (A + 4) = 18 * A + 98)) :
  A + 4 = 26 := by
  sorry

end cricketer_average_score_l299_299245


namespace total_fish_l299_299340

def LillyFish : ℕ := 10
def RosyFish : ℕ := 8
def MaxFish : ℕ := 15

theorem total_fish : LillyFish + RosyFish + MaxFish = 33 := by
  sorry

end total_fish_l299_299340


namespace correct_statement_is_d_l299_299059

/-- A definition for all the conditions given in the problem --/
def very_small_real_form_set : Prop := false
def smallest_natural_number_is_one : Prop := false
def sets_equal : Prop := false
def empty_set_subset_of_any_set : Prop := true

/-- The main statement to be proven --/
theorem correct_statement_is_d : (very_small_real_form_set = false) ∧ 
                                 (smallest_natural_number_is_one = false) ∧ 
                                 (sets_equal = false) ∧ 
                                 (empty_set_subset_of_any_set = true) :=
by
  sorry

end correct_statement_is_d_l299_299059


namespace points_lie_on_hyperbola_l299_299570

variable {t : ℝ}

def x (t : ℝ) : ℝ := (2*t + 1)/t
def y (t : ℝ) : ℝ := (t - 2)/t

theorem points_lie_on_hyperbola (ht : t ≠ 0) : ∃ t : ℝ, (x t) + (y t) = 3 - 1/t :=
by
  sorry

end points_lie_on_hyperbola_l299_299570


namespace fraction_of_mixture_is_water_l299_299809

-- Define the mixture conditions and weights
def total_weight : ℝ := 120
def weight_of_sand : ℝ := total_weight / 5
def weight_of_gravel : ℝ := 6

-- Define the weight of water
def weight_of_water : ℝ := total_weight - (weight_of_sand + weight_of_gravel)

-- Fraction of the mixture that is water
def fraction_of_water : ℝ := weight_of_water / total_weight

-- The theorem we want to prove
theorem fraction_of_mixture_is_water : fraction_of_water = 3 / 4 :=
by
  -- Since we are skipping the proof, insert 'sorry' here
  sorry

end fraction_of_mixture_is_water_l299_299809


namespace common_chord_through_intersection_point_l299_299006

-- Define the trapezoid and the problem's conditions.
structure Trapezoid where
  A B C D : Point
  A_non_parallel_C : A ≠ C
  B_non_parallel_D : B ≠ D
  ⟨AB⟩ ∥ ⟨CD⟩ : ∥ (Line A B) (Line C D)

-- Define the points and circles involved.
structure GeometryProblem where
  trapezoid : Trapezoid
  circle1 : Circle (diameter (diagonal trapezoid))
  circle2 : Circle (diameter' (diagonal trapezoid))
  intersection_point : Point := line_intersection (non_parallel_side_1 trapezoid) (non_parallel_side_2 trapezoid)
  common_chord : Line := radical_axis circle1 circle2

theorem common_chord_through_intersection_point
  (P : GeometryProblem) :
  P.intersection_point ∈ P.common_chord :=

sorry

end common_chord_through_intersection_point_l299_299006


namespace range_of_H_l299_299779

def H (x : ℝ) : ℝ := |x + 3| - |x - 2|

theorem range_of_H : set.range H = {-1, 5} := 
by { sorry }

end range_of_H_l299_299779


namespace Gunther_free_time_left_l299_299956

def vacuuming_time := 45
def dusting_time := 60
def folding_laundry_time := 25
def mopping_time := 30
def cleaning_bathroom_time := 40
def wiping_windows_time := 15
def brushing_cats_time := 4 * 5
def washing_dishes_time := 20
def first_tasks_total_time := 2 * 60 + 30
def available_free_time := 5 * 60

theorem Gunther_free_time_left : 
  (available_free_time - 
   (vacuuming_time + dusting_time + folding_laundry_time + 
    mopping_time + cleaning_bathroom_time + 
    wiping_windows_time + brushing_cats_time + 
    washing_dishes_time) = 45) := 
by 
  sorry

end Gunther_free_time_left_l299_299956


namespace problem_equivalent_l299_299253

theorem problem_equivalent :
  ∀ m n : ℤ, |m - n| = n - m ∧ |m| = 4 ∧ |n| = 3 → m + n = -1 ∨ m + n = -7 :=
by
  intros m n h
  have h1 : |m - n| = n - m := h.1
  have h2 : |m| = 4 := h.2.1
  have h3 : |n| = 3 := h.2.2
  sorry

end problem_equivalent_l299_299253


namespace part_I_part_II_part_III_zeroes_part_III_range_l299_299226

noncomputable def f : ℝ → ℝ := sorry

def F (x : ℝ) : ℝ :=
if x < f(x) then 1
else if x = f(x) then 0
else -1

def F_2x_minus_1 (x : ℝ) : ℝ :=
F (2 * x - 1)

theorem part_I (x : ℝ) :
  (F_2x_minus_1 x = 1 ↔ x > 1) ∧
  (F_2x_minus_1 x = 0 ↔ x = 1) ∧
  (F_2x_minus_1 x = -1 ↔ x < 1) := sorry

theorem part_II (a : ℝ) :
  (∀ x, F (abs (x - a)) + F_2x_minus_1 x = 0) ↔ (a = 0 ∨ a = 2) := sorry

def h (x : ℝ) : ℝ :=
cos x * F (x + sin x)

theorem part_III_zeroes : 
  ∃! x ∈ set.Icc (Real.pi / 3) (4 * Real.pi / 3), h(x) = 0 :=
sorry

theorem part_III_range :
  let S := (Real.pi / 3), T := (4 * Real.pi / 3)
  (∀ x ∈ set.Icc S T, -1 < h(x) ∧ h(x) < 1) := sorry

end part_I_part_II_part_III_zeroes_part_III_range_l299_299226


namespace rope_cut_ratio_l299_299313

theorem rope_cut_ratio (H1 : Josh has 100 feet of rope)
                       (H2 : He cuts the rope in half)
                       (H3 : He takes one of the halves and cuts it into two equal parts)
                       (H4 : He takes one of the remaining pieces and cuts it into fifths)
                       (H5 : The length of the rope he's most recently cut is 5 feet long) :
                       ratio_of_second_cut_to_first_cut = 1 : 2 := 
by
  sorry

end rope_cut_ratio_l299_299313


namespace betty_age_l299_299874

variable (C A B : ℝ)

-- conditions
def Carol_five_times_Alice := C = 5 * A
def Alice_twelve_years_younger_than_Carol := A = C - 12
def Carol_twice_as_old_as_Betty := C = 2 * B

-- goal
theorem betty_age (hc1 : Carol_five_times_Alice C A)
                  (hc2 : Alice_twelve_years_younger_than_Carol C A)
                  (hc3 : Carol_twice_as_old_as_Betty C B) : B = 7.5 := 
  by
  sorry

end betty_age_l299_299874


namespace oliver_final_amount_l299_299353

def initial_amount : ℤ := 33
def spent : ℤ := 4
def received : ℤ := 32

def final_amount (initial_amount spent received : ℤ) : ℤ :=
  initial_amount - spent + received

theorem oliver_final_amount : final_amount initial_amount spent received = 61 := 
by sorry

end oliver_final_amount_l299_299353


namespace find_a5_l299_299202

-- Definitions (conditions)
variable {α : Type*} [AddGroup α] {a : ℕ → α}
def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d : α, ∀ n : ℕ, a (n+1) = a n + d

variable [IsArithmeticSequence a]

variable (h1 : a 2 + a 11 = 36)
variable (h2 : a 8 = 24)

-- Statement (proof problem)
theorem find_a5 : a 5 = 12 := by
  sorry


end find_a5_l299_299202


namespace perfect_square_of_custom_number_l299_299367

-- Define the problem: Proving that the number consisting of 1997 ones, 1998 twos followed by a 5 is a perfect square
theorem perfect_square_of_custom_number :
  ∃ k : ℕ, (let m := (∑ i in finset.range 1997, 10^i) * 10^1998 + (∑ i in finset.range 1998, 10^i) * 2 + 5 in m = (10 * k + 5)^2) :=
by
  sorry

end perfect_square_of_custom_number_l299_299367


namespace arithmetic_sequence_problem_l299_299285

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
∀ n, a n = a1 + (n - 1) * d

-- Given condition
def given_condition (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
3 * a 9 - a 15 - a 3 = 20

-- Question to prove
def question (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
2 * a 8 - a 7 = 20

-- Main theorem
theorem arithmetic_sequence_problem (a: ℕ → ℝ) (a1 d: ℝ):
  arithmetic_sequence a a1 d →
  given_condition a a1 d →
  question a a1 d :=
by
  sorry

end arithmetic_sequence_problem_l299_299285


namespace even_function_derivative_zero_l299_299973

variable {ℝ : Type*} [LinearOrderedField ℝ] {f : ℝ → ℝ}

theorem even_function_derivative_zero (h_even : ∀ x, f x = f (-x)) 
  (h_deriv : ∃ f', ∀ x, deriv f x = f' x) : deriv f 0 = 0 :=
by
  sorry

end even_function_derivative_zero_l299_299973


namespace lcm_of_numbers_is_750_l299_299978

-- Define the two numbers x and y
variables (x y : ℕ)

-- Given conditions as hypotheses
def product_of_numbers := 18750
def hcf_of_numbers := 25

-- The proof problem statement
theorem lcm_of_numbers_is_750 (h_product : x * y = product_of_numbers) 
                              (h_hcf : Nat.gcd x y = hcf_of_numbers) : Nat.lcm x y = 750 :=
by
  sorry

end lcm_of_numbers_is_750_l299_299978


namespace true_propositions_l299_299224

-- Define the propositions as hypotheses
variable (P1 P2 P3 P4 : Prop)

-- Given conditions as axioms
axiom h1 : P1 = (∀ l1 l2 p, (plane_line_parallel l1 p) ∧ (plane_line_parallel l2 p) → plane_parallel l1 l2)
axiom h2 : P2 = (∀ l p1 p2, (plane_line_perpendicular l p1) ∧ (plane_contains_line l p2) → plane_perpendicular p1 p2)
axiom h3 : P3 = (∀ l l1 l2, (line_perpendicular l1 l) ∧ (line_perpendicular l2 l) → line_parallel l1 l2)
axiom h4 : P4 = (∀ l p1 p2, plane_perpendicular p1 p2 → line_in_plane_not_perpendicular l p1 p2 → line_not_perpendicular l p2)

-- The proposition to prove
theorem true_propositions (hP2 : P2) (hP4 : P4) : 
  (P1 ∨ P2 ∨ P3 ∨ P4) → (¬P1 ∧ P2 ∧ ¬P3 ∧ P4) :=
by
  sorry

end true_propositions_l299_299224


namespace product_divisible_by_factorial_l299_299568

theorem product_divisible_by_factorial (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℕ, (∏ i in Finset.range(n), 9 * i + 3) = k * n! :=
by sorry

end product_divisible_by_factorial_l299_299568


namespace max_diagonal_sum_l299_299352

open Matrix

noncomputable def max_trace (M : Matrix ℕ ℕ ℝ) (p q : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i, M i i

theorem max_diagonal_sum (n : ℕ) (p q : ℕ → ℝ) (M : Matrix (Fin n) (Fin n) ℝ) :
  (∀ i, 0 ≤ p i) →
  (∀ j, 0 ≤ q j) →
  (∀ i, ∑ j, M i j = p i) →
  (∀ j, ∑ i, M i j = q j) →
  ∑ i, p i = ∑ j, q j →
  (∀ i j, 0 ≤ M i j) →
  max_trace M p q n = ∑ i in Fin.range n, min (p i) (q i) :=
sorry

end max_diagonal_sum_l299_299352


namespace seats_arrangements_l299_299572

-- Define the concept of a derangement (permutation where no element appears in its initial position) for 4 elements.
def derangement_count (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement_count (n - 1) + derangement_count (n - 2))

-- Ensure the derangement for 4 elements is calculable
def D4 := derangement_count 4

-- Define the theorem based on the given conditions
theorem seats_arrangements : ∀ (s : Fin 4 → (Fin 2 → Fin 4)), 
  (∃ f : Fin 4 → Fin 4, 
  ∀ i, 
  f i ≠ i) →
  4! * D4 * 2^4 = 3456 := 
by
  assume s f h,
  sorry

end seats_arrangements_l299_299572


namespace largest_integer_n_l299_299505

theorem largest_integer_n (total_crates : ℕ) (orange_counts : finset ℕ)
  (h1 : total_crates = 160) (h2 : orange_counts = finset.Icc 115 142) :
  ∃ n : ℕ, n = 6 ∧ ∀ c ∈ orange_counts, ∃ k : ℕ, k = total_crates / finset.card orange_counts 
          ∧ ∃ remainder : ℕ, remainder = total_crates % finset.card orange_counts
          ∧ k + (if 28 > 0 then 1 else 0) = n :=
by
  sorry

end largest_integer_n_l299_299505


namespace glasses_per_pitcher_l299_299134

def total_glasses : Nat := 30
def num_pitchers : Nat := 6

theorem glasses_per_pitcher : total_glasses / num_pitchers = 5 := by
  sorry

end glasses_per_pitcher_l299_299134


namespace square_side_length_of_right_triangle_l299_299720

def hypotenuse_length (a b : ℕ) : ℕ := (a^2 + b^2).nat_sqrt

theorem square_side_length_of_right_triangle (DE DF : ℕ) (s : ℚ) (hDE : DE = 7) 
(hDF : DF = 24) (square_on_hypotenuse : s = 525 / 96) :
  let EF := hypotenuse_length DE DF in
  EF * s / DE * s / DF = s := sorry

end square_side_length_of_right_triangle_l299_299720


namespace find_some_number_l299_299412

noncomputable def some_number : ℝ := 1000
def expr_approx (a b c d : ℝ) := (a * b) / c = d

theorem find_some_number :
  expr_approx 3.241 14 some_number 0.045374000000000005 :=
by sorry

end find_some_number_l299_299412


namespace lcm_of_proportion_l299_299895

noncomputable def A B C : ℕ := sorry

theorem lcm_of_proportion (A B C : ℕ) (h1 : A * 4 = B * 3) (h2 : B * 5 = C * 4) (h3 : Nat.gcd A (Nat.gcd B C) = 6) (h4 : ∃ k, Nat.lcm A (Nat.lcm B C) = 12 * k) :
  Nat.lcm A (Nat.lcm B C) = 360 :=
sorry

end lcm_of_proportion_l299_299895


namespace roster_representation_of_M_l299_299705

def M : Set ℚ := {x | ∃ m n : ℤ, x = m / n ∧ |m| < 2 ∧ 1 ≤ n ∧ n ≤ 3}

theorem roster_representation_of_M :
  M = {-1, -1/2, -1/3, 0, 1/2, 1/3} :=
by sorry

end roster_representation_of_M_l299_299705


namespace distance_focus_to_directrix_l299_299008

def parabola_distance_focus_to_directrix (p : ℕ) (y x : ℝ) (h : y^2 = 2 * p * x) : ℕ :=
  p

theorem distance_focus_to_directrix (h : ∀ (y x : ℝ), y^2 = 8 * x → y^2 = 2 * 4 * x) :
  parabola_distance_focus_to_directrix 4 y x (h y x) = 4 := by
  sorry

end distance_focus_to_directrix_l299_299008


namespace final_car_price_l299_299021

theorem final_car_price :
  let p₀ := 80000
  let d₁ := 0.30
  let d₂ := 0.25
  let d₃ := 0.20
  let d₄ := 0.15
  let d₅ := 0.10
  let d₆ := 0.05
  let p₁ := p₀ * (1 - d₁)
  let p₂ := p₁ * (1 - d₂)
  let p₃ := p₂ * (1 - d₂)
  let p₄ := p₃ * (1 - d₃)
  let p₵ := p₄ * (1 - d₄)
  let p₆ := p₵ * (1 - d₄)
  let p₇ := p₆ * (1 - d₅)
  let p₈ := p₇ * (1 - d₅)
  let p₉ := p₈ * (1 - d₆) in
  p₉ = 24418.80 := by
  sorry

end final_car_price_l299_299021


namespace hexagon_shaded_area_correct_l299_299653

theorem hexagon_shaded_area_correct :
  let side_length := 3
  let semicircle_radius := side_length / 2
  let central_circle_radius := 1
  let hexagon_area := (3 * Real.sqrt 3 / 2) * side_length ^ 2
  let semicircle_area := (π * (semicircle_radius ^ 2)) / 2
  let total_semicircle_area := 6 * semicircle_area
  let central_circle_area := π * (central_circle_radius ^ 2)
  let shaded_area := hexagon_area - (total_semicircle_area + central_circle_area)
  shaded_area = 13.5 * Real.sqrt 3 - 7.75 * π := by
  sorry

end hexagon_shaded_area_correct_l299_299653


namespace ratio_of_perimeters_is_correct_l299_299830

-- Define the conditions
def square_side : ℝ := 6
def folded_rectangle_width : ℝ := square_side / 2
def small_rectangle_width : ℝ := folded_rectangle_width / 3
def large_rectangle_width : ℝ := 2 * small_rectangle_width

-- Define the perimeters
def small_rectangle_perimeter : ℝ := 2 * (square_side + small_rectangle_width)
def large_rectangle_perimeter : ℝ := 2 * (square_side + large_rectangle_width)

-- Define the expected ratio of perimeters
def expected_ratio : ℝ := 7 / 8 

-- Write the theorem to prove
theorem ratio_of_perimeters_is_correct :
  small_rectangle_perimeter / large_rectangle_perimeter = expected_ratio := 
sorry

end ratio_of_perimeters_is_correct_l299_299830


namespace polynomial_characterization_l299_299182

noncomputable def satisfiesCondition (P : Polynomial ℤ) (n : ℕ) : Prop :=
  ∃ (pairs : Finset (ℕ × ℕ)), 
    (pairs.card ≤ 2021) ∧ 
    (∀ (a b : ℕ), a < b ∧ b ≤ n → ((abs(P.eval a) - abs(P.eval b)) % n = 0) → 
           ((a, b) ∈ pairs))

theorem polynomial_characterization :
  ∀ (P : Polynomial ℤ), (∀ n : ℕ, satisfiesCondition P n) →
  ∃ (c d : ℤ), P = c • Polynomial.X + Polynomial.C d ∧ 
               ( (c = 1 ∧ d ≥ -2022) ∨ 
                (c = -1 ∧ d ≤ 2022) ) :=
sorry

end polynomial_characterization_l299_299182


namespace mul_three_point_six_and_zero_point_twenty_five_l299_299521

theorem mul_three_point_six_and_zero_point_twenty_five : 3.6 * 0.25 = 0.9 := by 
  sorry

end mul_three_point_six_and_zero_point_twenty_five_l299_299521


namespace product_simplification_l299_299183

theorem product_simplification (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    (a + b + c)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹) * (a^2 + b^2 + c^2)⁻¹ * ((ab)⁻¹ + (bc)⁻¹ + (ca)⁻¹) = (a * b * c)⁻² := 
by
  sorry

end product_simplification_l299_299183


namespace range_of_h_l299_299172

noncomputable def h (t : ℝ) : ℝ := (t^2 + 1/2 * t) / (t^2 + 1)

theorem range_of_h :
  Set.range h = setOf (y | (-1/4 : ℝ) ≤ y ∧ y ≤ 1/4) := by
  sorry

end range_of_h_l299_299172


namespace cylinder_height_proof_l299_299113

noncomputable def cone_base_radius : ℝ := 15
noncomputable def cone_height : ℝ := 25
noncomputable def cylinder_base_radius : ℝ := 10
noncomputable def cylinder_water_height : ℝ := 18.75

theorem cylinder_height_proof :
  (1 / 3 * π * cone_base_radius^2 * cone_height) = π * cylinder_base_radius^2 * cylinder_water_height :=
by sorry

end cylinder_height_proof_l299_299113


namespace value_range_of_2_sin_x_minus_1_l299_299425

theorem value_range_of_2_sin_x_minus_1 :
  (∀ x : ℝ, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1) →
  (∀ y : ℝ, y = 2 * Real.sin y - 1 → -3 ≤ y ∧ y ≤ 1) :=
by
  sorry

end value_range_of_2_sin_x_minus_1_l299_299425


namespace find_percentage_l299_299079

theorem find_percentage (P : ℝ) : 
  (P / 100) * 100 - 40 = 30 → P = 70 :=
by
  intros h
  sorry

end find_percentage_l299_299079


namespace trajectory_is_ellipse_l299_299920

variable (z : ℂ) (i : ℂ := complex.I) (a b : ℂ)

def trajectory_of_point (z : ℂ) : Type :=
  { X : ℝ × ℝ // |z - i| + |z + i| = 3 }

theorem trajectory_is_ellipse (h : |z - i| + |z + i| = 3) :
  ∃ f1 f2 : ℝ × ℝ, 
    ∀ Z : trajectory_of_point z, 
      Z.val = f1 ∨ Z.val = f2 :=
sorry

end trajectory_is_ellipse_l299_299920


namespace asymptotic_minimal_eccentricity_l299_299557

noncomputable def hyperbola (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / m - y^2 / (m^2 + 4) = 1

noncomputable def eccentricity (m : ℝ) : ℝ :=
  Real.sqrt (m + 4 / m + 1)

theorem asymptotic_minimal_eccentricity :
  ∃ (m : ℝ), m = 2 ∧ hyperbola m x y → ∀ x y, y = 2 * x ∨ y = -2 * x :=
by
  sorry

end asymptotic_minimal_eccentricity_l299_299557


namespace average_rate_of_change_nonzero_l299_299286

-- Define the conditions related to the average rate of change.
variables {x0 : ℝ} {Δx : ℝ}

-- Define the statement to prove that in the definition of the average rate of change, Δx ≠ 0.
theorem average_rate_of_change_nonzero (h : Δx ≠ 0) : True :=
sorry  -- The proof is omitted as per instruction.

end average_rate_of_change_nonzero_l299_299286


namespace correct_log_values_l299_299437

variable (a b c : ℝ)

def log_values (x : ℝ) : Prop :=
  match x with
  | 0.021  => 2 * a + b + c - 3
  | 0.27   => 6 * a - 3 * b - 2
  | 1.5    => 3 * a - b + c - 1
  | 2.8    => 1 - 2 * a + 2 * b - c
  | 3      => 2 * a - b
  | 5      => a + c
  | 6      => 1 + a - b - c
  | 7      => 2 * b + c
  | 8      => 3 - 3 * a - 3 * c
  | 9      => 4 * a - 2 * b
  | 14     => 1 - c + 2 * b
  | _      => 0  -- assuming default is 0 for simplicity
  end

theorem correct_log_values :
  log_values a b c 7 = 2 * b + c ∧ log_values a b c 1.5 = 3 * a - b + c - 1 :=
by
  sorry

end correct_log_values_l299_299437


namespace intersection_point_on_circle_l299_299953

theorem intersection_point_on_circle :
  ∀ (m : ℝ) (x y : ℝ),
  (m * x - y = 0) → 
  (x + m * y - m - 2 = 0) → 
  (x - 1)^2 + (y - 1 / 2)^2 = 5 / 4 :=
by
  intros m x y h1 h2
  sorry

end intersection_point_on_circle_l299_299953


namespace gcd_polynomials_l299_299734

-- Given condition: a is an even multiple of 1009
def is_even_multiple_of_1009 (a : ℤ) : Prop :=
  ∃ k : ℤ, a = 2 * 1009 * k

-- Statement: gcd(2a^2 + 31a + 58, a + 15) = 1
theorem gcd_polynomials (a : ℤ) (ha : is_even_multiple_of_1009 a) :
  gcd (2 * a^2 + 31 * a + 58) (a + 15) = 1 := 
sorry

end gcd_polynomials_l299_299734


namespace largest_A_proof_smallest_A_proof_l299_299846

def is_coprime_with_12 (n : ℕ) : Prop := Nat.gcd n 12 = 1

def obtain_A_from_B (B : ℕ) : ℕ :=
  let b := B % 10
  let k := B / 10
  b * 10^7 + k

constant B : ℕ → Prop
constant A : ℕ → ℕ → Prop

noncomputable def largest_A : ℕ :=
  99999998

noncomputable def smallest_A : ℕ :=
  14444446

theorem largest_A_proof (B : ℕ) (h1 : B > 44444444) (h2 : is_coprime_with_12 B) :
  obtain_A_from_B B = largest_A :=
sorry

theorem smallest_A_proof (B : ℕ) (h1 : B > 44444444) (h2 : is_coprime_with_12 B) :
  obtain_A_from_B B = smallest_A :=
sorry

end largest_A_proof_smallest_A_proof_l299_299846


namespace extreme_value_at_one_is_minimum_l299_299607

noncomputable def f (x a : ℝ) : ℝ := (x^2 + 1) / (x + a)

theorem extreme_value_at_one_is_minimum (a : ℝ) (h_extreme : ∀ x ≠ (-a), deriv (λ x, (x^2 + 1) / (x + a)) x = 0 → x = 1) :
  ∃ y, f y a ≤ f 1 a ∧ ∀ z, z ≠ y → f y a < f z a :=
begin
  sorry
end

end extreme_value_at_one_is_minimum_l299_299607


namespace max_good_points_l299_299529

-- We define that we have seven lines in the plane.
def seven_lines (P : Type) [plane : Plane P] : Prop := 
  ∃ l1 l2 l3 l4 l5 l6 l7 : Line P, pairwise (≠) [l1, l2, l3, l4, l5, l6, l7]

-- A point is good if it lies on at least three lines.
def is_good_point (P : Type) [plane : Plane P] (pt : P) (l1 l2 l3 l4 l5 l6 l7 : Line P) : Prop :=
  3 ≤ {(l : Line P) | pt ∈ l ∧ (l = l1 ∨ l = l2 ∨ l = l3 ∨ l = l4 ∨ l = l5 ∨ l = l6 ∨ l = l7)}.card

-- The theorem to state the maximum number of good points.
theorem max_good_points (P : Type) [plane : Plane P] :
  seven_lines P → ∃ pts : Finset P, 
    (∀ pt ∈ pts, ∃ (l1 l2 l3 l4 l5 l6 l7 : Line P), is_good_point P pt l1 l2 l3 l4 l5 l6 l7) ∧
    pts.card ≤ 6 :=
sorry

end max_good_points_l299_299529


namespace dot_product_value_norm_sum_value_l299_299620

variable {α : Type*} [inner_product_space ℝ α]

def condition1 (a b : α) : Prop := ∥a∥ = 2
def condition2 (a b : α) : Prop := ∥b∥ = 1
def condition3 (a b : α) : Prop := ∥a - b∥ = 2

theorem dot_product_value (a b : α) (h1 : condition1 a b) (h2 : condition2 a b) (h3 : condition3 a b) : 
  ⟪a, b⟫ = 3/2 :=
by sorry

theorem norm_sum_value (a b : α) (h1 : condition1 a b) (h2 : condition2 a b) (h3 : condition3 a b) : 
  ∥a + b∥ = 2 * real.sqrt 2 :=
by sorry

end dot_product_value_norm_sum_value_l299_299620


namespace only_prime_satisfying_condition_l299_299553

theorem only_prime_satisfying_condition (p : ℕ) (h_prime : Prime p) : (Prime (p^2 + 14) ↔ p = 3) := 
by
  sorry

end only_prime_satisfying_condition_l299_299553


namespace triploid_fruit_fly_chromosome_periodicity_l299_299838

-- Define the conditions
def normal_chromosome_count (organism: Type) : ℕ := 8
def triploid_fruit_fly (organism: Type) : Prop := true
def XXY_sex_chromosome_composition (organism: Type) : Prop := true
def periodic_change (counts: List ℕ) : Prop := counts = [9, 18, 9]

-- State the theorem
theorem triploid_fruit_fly_chromosome_periodicity (organism: Type)
  (h1: triploid_fruit_fly organism) 
  (h2: XXY_sex_chromosome_composition organism)
  (h3: normal_chromosome_count organism = 8) : 
  periodic_change [9, 18, 9] :=
sorry

end triploid_fruit_fly_chromosome_periodicity_l299_299838


namespace pipe_B_cannot_fill_tank_l299_299502

noncomputable def fill_rate_C : ℝ := 1 / 80
noncomputable def fill_rate_B : ℝ := 2 * fill_rate_C
noncomputable def leak_rate : ℝ := 1 / 20
noncomputable def net_fill_rate_B : ℝ := fill_rate_B - leak_rate

theorem pipe_B_cannot_fill_tank : net_fill_rate_B < 0 :=
by 
  have : fill_rate_C = 1 / 80 := rfl
  have : fill_rate_B = 2 * (1 / 80) := by rw [fill_rate_C]; ring
  have : leak_rate = 1 / 20 := rfl
  have : net_fill_rate_B = (1 / 40) - (1 / 20) := by rw [fill_rate_B, leak_rate]; ring
  have : (1 / 40) - (1 / 20) = - (1 / 40) := by norm_num
  show - (1 / 40) < 0, by norm_num

#eval pipe_B_cannot_fill_tank

end pipe_B_cannot_fill_tank_l299_299502


namespace rectangle_divided_area_l299_299276

-- Define the parameters for the rectangle
def rectangle_sides (AD AB : ℝ) : Prop := AD = 4 ∧ AB = 2

-- Define the angle bisectors
def angle_bisectors (AK DL : ℝ → ℝ) : Prop := 
  ∀ (x : ℝ), (0 ≤ x) → (x ≤ AB) → (AK x = x / 2) ∧ (DL x = x / 2)

-- Have the condition about rectangle sides and angle bisectors and prove the division of area
theorem rectangle_divided_area (AD AB : ℝ) (AK DL : ℝ → ℝ) 
  (h_sides : rectangle_sides AD AB) (h_bisectors : angle_bisectors AK DL) :
  ∃ A1 A2 A3 : ℝ, A1 = 2 ∧ A2 = 2 ∧ A3 = 4 ∧ (A1 + A2 + A3 = AD * AB) :=
by
  sorry

end rectangle_divided_area_l299_299276


namespace function_parity_l299_299143

noncomputable def f : ℝ → ℝ := sorry

-- Condition: f satisfies the functional equation for all x, y in Real numbers
axiom functional_eqn (x y : ℝ) : f (x + y) + f (x - y) = 2 * f x * f y

-- Prove that the function could be either odd or even.
theorem function_parity : (∀ x, f (-x) = f x) ∨ (∀ x, f (-x) = -f x) := 
sorry

end function_parity_l299_299143


namespace unknown_cell_is_red_l299_299440

-- Definitions based on problem conditions
def Cube : Type := array 3 (array 3 (array 3 (option char)))

def initial_cube : Cube :=
  [[['K', none, 'K'], 
    ['C', 'K', 'C'], 
    ['K', 'C', 'K']],
   [['none', none, none], 
    [none, none, none], 
    [none, none, none]],
   [['K', none, 'K'], 
    ['C', 'K', 'C'], 
    ['K', 'C', 'K']]]

noncomputable def solve_cube (c : Cube) : option char :=
  c[0][0][1] -- Access the upper middle cell of the leftmost layer

-- Theorem stating the proof problem
theorem unknown_cell_is_red (c : Cube) (initial_conditions : c = initial_cube) :
  solve_cube c = some 'K' :=
by
  sorry

end unknown_cell_is_red_l299_299440


namespace number_of_divisible_by_3_or_5_l299_299970

open Finset

theorem number_of_divisible_by_3_or_5 :
  let S := {i | 1 ≤ i ∧ i ≤ 60}.toFinset in
  let A := S.filter (fun n => n % 3 = 0) in
  let B := S.filter (fun n => n % 5 = 0) in
  A.card + B.card - (A ∩ B).card = 28 :=
by
  let S := {i | 1 ≤ i ∧ i ≤ 60}.toFinset
  let A := S.filter (fun n => n % 3 = 0)
  let B := S.filter (fun n => n % 5 = 0)
  sorry

end number_of_divisible_by_3_or_5_l299_299970


namespace largest_r_plus_s_l299_299042

-- Defining the coordinates of X and Y given in the problem
def X : ℝ × ℝ := (10, 15)
def Y : ℝ × ℝ := (22, 16)

-- Area of triangle XYZ
def area : ℝ := 56

-- Coordinates of Z (r, s) which we need to find
variables (r s : ℝ)

-- Slope of the median to side XY
def slope : ℝ := -3

-- Defining the midpoint M and the condition on the slope of the median
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def median_slope_condition (r s : ℝ) : Prop :=
  let M := midpoint X Y in
  (s - M.2) = slope * (r - M.1)

-- Defining the area condition using the determinant method
def area_condition (r s : ℝ) : Prop :=
  let [x1, y1] := [X.1, X.2]
      [x2, y2] := [Y.1, Y.2] in
  (area * 2) = abs (-r + (12 * s) - 160)

-- The main theorem combining all conditions and giving the conclusion
theorem largest_r_plus_s :
  ∃ r s : ℝ, 
  median_slope_condition r s ∧ area_condition r s ∧ r + s = 36.96 :=
  sorry

end largest_r_plus_s_l299_299042


namespace lunch_cost_is_1036_l299_299344

/-- Define the number of classes and number of students per class. -/
def thirdGradeClasses := 5
def studentsPerThirdGradeClass := 30

def fourthGradeClasses := 4
def studentsPerFourthGradeClass := 28

def fifthGradeClasses := 4
def studentsPerFifthGradeClass := 27

/-- Define the cost of food items for each student. -/
def costOfHamburger := 2.10
def costOfCarrots := 0.50
def costOfCookie := 0.20

/-- Summing up the number of students in each grade. -/
def totalStudents := 
  (thirdGradeClasses * studentsPerThirdGradeClass) + 
  (fourthGradeClasses * studentsPerFourthGradeClass) + 
  (fifthGradeClasses * studentsPerFifthGradeClass)

/-- Calculating the cost of one student's lunch. -/
def costPerStudent :=
  costOfHamburger + costOfCarrots + costOfCookie

/-- Calculating the total cost for all students. -/
def totalCost :=
  totalStudents * costPerStudent

/-- The theorem we need to prove: the total cost is $1,036. -/
theorem lunch_cost_is_1036 : totalCost = 1036 := 
by 
  have h1 : totalStudents = 370 := by sorry
  have h2 : costPerStudent = 2.80 := by sorry
  show totalCost = 370 * 2.80 from sorry

end lunch_cost_is_1036_l299_299344


namespace gcd_b2_add_11b_add_28_b_add_6_eq_2_l299_299589

theorem gcd_b2_add_11b_add_28_b_add_6_eq_2 {b : ℤ} (h : ∃ k : ℤ, b = 1836 * k) : 
  Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 := 
by
  sorry

end gcd_b2_add_11b_add_28_b_add_6_eq_2_l299_299589


namespace path_inequality_l299_299974

theorem path_inequality
  (f : ℕ → ℕ → ℝ) :
  f 1 6 * f 2 5 * f 3 4 + f 1 5 * f 2 4 * f 3 6 + f 1 4 * f 2 6 * f 3 5 ≥
  f 1 6 * f 2 4 * f 3 5 + f 1 5 * f 2 6 * f 3 4 + f 1 4 * f 2 5 * f 3 6 :=
sorry

end path_inequality_l299_299974


namespace favorite_numbers_product_71668_l299_299709

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the favorite number property
def is_favorite (n : ℕ) : Prop :=
  sum_of_digits n = 10

-- Define the given problem as a Lean theorem
theorem favorite_numbers_product_71668 (a b c : ℕ ) 
(hfa : is_favorite a) (hfb : is_favorite b) (hfc : is_favorite c) 
(hprod : a * b * c = 71668) : 
  {a, b, c} = {19, 46, 82} :=
by
  sorry

end favorite_numbers_product_71668_l299_299709


namespace max_value_at_critical_points_or_endpoints_l299_299490

variable (f : ℝ → ℝ) (a b : ℝ)
variable (h_diff : ∀ x ∈ set.Icc a b, differentiable_at ℝ f x)
variable (h_cont : continuous_on f (set.Icc a b))
variable (h_interval : a < b)

theorem max_value_at_critical_points_or_endpoints :
  ∃ x ∈ set.Icc a b, (∀ y ∈ set.Icc a b, f y ≤ f x) :=
sorry

end max_value_at_critical_points_or_endpoints_l299_299490


namespace quadrant_identification_l299_299632

theorem quadrant_identification (α : ℝ) :
  (tan α < 0) ∧ (sin α > cos α) → (α > π / 2) ∧ (α < π) :=
by
  sorry

end quadrant_identification_l299_299632


namespace find_xy_integers_l299_299551

theorem find_xy_integers (x y : ℤ) (h : x^3 + 2 * x * y = 7) :
  (x, y) = (-7, -25) ∨ (x, y) = (-1, -4) ∨ (x, y) = (1, 3) ∨ (x, y) = (7, -24) :=
sorry

end find_xy_integers_l299_299551


namespace point_on_transformed_graph_l299_299943

theorem point_on_transformed_graph (g : ℝ → ℝ) (h : g 3 = 4) : 
  3 * (14 / 3) = 2 * g (3 * 1) + 6 ∧ 1 + 14 / 3 = 17 / 3 := 
by
  split
  . rw [h, mul_one]
    ring
  . exact rfl

end point_on_transformed_graph_l299_299943


namespace num_ways_l299_299531

-- Definitions of the sequences based on given recurrence relations
def a (n : ℕ) : ℕ
def b (n : ℕ) : ℕ
def c (n : ℕ) : ℕ

-- Initial conditions
axiom a0 : a 0 = 1
axiom b0 : b 0 = 0
axiom c0 : c 0 = 0

-- Recurrence relations
axiom rec_a (n : ℕ) : a (n + 1) = 2 * a n + b n
axiom rec_b (n : ℕ) : b (n + 1) = 2 * a n + 2 * b n + c n
axiom rec_c (n : ℕ) : c (n + 1) = 5 * a n + 4 * b n + c n

-- The theorem to be proven
theorem num_ways (n : ℕ) : a n = -- insert the correct formula or leave it as a proof goal
sorry

end num_ways_l299_299531


namespace students_play_neither_l299_299067

def total_students : ℕ := 35
def play_football : ℕ := 26
def play_tennis : ℕ := 20
def play_both : ℕ := 17

theorem students_play_neither : (total_students - (play_football + play_tennis - play_both)) = 6 := by
  sorry

end students_play_neither_l299_299067


namespace simplify_expression_calculate_difference_of_squares_l299_299872

section Problem1
variable (a b : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0)

theorem simplify_expression : ((-2 * a^2) ^ 2 * (-b^2)) / (4 * a^3 * b^2) = -a :=
by sorry
end Problem1

section Problem2

theorem calculate_difference_of_squares : 2023^2 - 2021 * 2025 = 4 :=
by sorry
end Problem2

end simplify_expression_calculate_difference_of_squares_l299_299872


namespace part1_part2_l299_299667

variables {V : Type*} [inner_product_space ℝ V]
variables {A B C D P Q : V}

-- Condition definitions
def midpoint (A B P : V) := P = (A + B) / 2
def perpendicular (u v : V) := ⟪u, v⟫ = 0
def equal_length (u v : V) := ∥u∥ = ∥v∥

theorem part1 (h1 : equal_length (A - B) (C - D)) (h2 : equal_length (A - D) (B - C))
  (hP : midpoint A C P) (hQ : midpoint B D Q) :
  perpendicular (Q - P) (C - A) ∧ perpendicular (Q - P) (D - B) :=
sorry

theorem part2 (h1 : perpendicular (Q - P) (C - A)) (h2 : perpendicular (Q - P) (D - B))
  (hP : midpoint A C P) (hQ : midpoint B D Q) :
  equal_length (A - B) (C - D) ∧ equal_length (A - D) (B - C) :=
sorry

end part1_part2_l299_299667


namespace planes_parallel_l299_299335

-- Required Definitions
variables {Line Plane : Type} 
-- Definitions for parallel and perpendicular relationships
variables (parallel : Line → Plane → Prop)
variables (perpendicular : Line → Plane → Prop)
variables (parallel_lines : Line → Line → Prop)

-- Given conditions
variable (m n : Line)
variable (alpha beta : Plane)
variable (h1 : perpendicular m alpha)
variable (h2 : perpendicular n beta)
variable (h3 : parallel_lines m n)

-- To be proven: alpha is parallel to beta
theorem planes_parallel :
    perpendicular m alpha →
    perpendicular n beta →
    parallel_lines m n →
    parallel alpha beta :=
sorry -- proof to be added later


end planes_parallel_l299_299335


namespace div_decimal_l299_299138

theorem div_decimal (a b : ℝ)  (h₁ : a = 0.45) (h₂ : b = 0.005):
  a / b = 90 :=
by {
  sorry
}

end div_decimal_l299_299138


namespace number_of_extreme_points_zero_l299_299753

noncomputable def f (x a : ℝ) : ℝ := x^3 + 3 * x^2 + 3 * x - a

theorem number_of_extreme_points_zero (a : ℝ) : 
  ∀ f, (f = λ x, x^3 + 3 * x^2 + 3 * x - a) → (∀ x, f' x ≤ 0) → f' = λ x, 3 * (x + 1)^2 → 
  ∀ x, (f' x ≠ 0) := 
begin
  sorry
end


end number_of_extreme_points_zero_l299_299753


namespace geometric_sequence_product_l299_299304

theorem geometric_sequence_product :
  ∃ a : ℕ → ℝ, 
    a 1 = 1 ∧ 
    a 5 = 16 ∧ 
    (∀ n, a (n + 1) = a n * r) ∧
    ∃ r : ℝ, 
      a 2 * a 3 * a 4 = 64 :=
by
  sorry

end geometric_sequence_product_l299_299304


namespace speed_in_still_water_l299_299819

theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) (h₁ : upstream_speed = 20) (h₂ : downstream_speed = 60) :
  (upstream_speed + downstream_speed) / 2 = 40 := by
  sorry

end speed_in_still_water_l299_299819


namespace probability_geq_six_l299_299272

def tetrahedral_faces : List ℕ := [1, 2, 3, 5]

def possible_pairs (faces : List ℕ) : List (ℕ × ℕ) :=
  List.bind faces (λ x, faces.map (λ y, (x, y)))

def favorable_pairs : List (ℕ × ℕ) :=
  possible_pairs tetrahedral_faces |>.filter (λ (x, y) => x + y ≥ 6)

def probability_event (total favorable : ℕ) : ℚ := favorable / total

theorem probability_geq_six :
  probability_event 16 (List.length favorable_pairs) = 1 / 2 := by
  sorry

end probability_geq_six_l299_299272


namespace probability_4a_plus_8b_units_digit_4_l299_299178

noncomputable def probability_units_digit_four (a b : ℕ) :=
  if (4 ^ a + 8 ^ b) % 10 = 4 then 1 else 0

theorem probability_4a_plus_8b_units_digit_4 : 
  ((∑ a in Finset.range 20 \ Finset.single 0, 
    ∑ b in Finset.range 20 \ Finset.single 0, 
      probability_units_digit_four a b) : ℚ) / (20 * 20) = 1 / 2 :=
by sorry

end probability_4a_plus_8b_units_digit_4_l299_299178


namespace area_of_triangle_gkl_l299_299290

-- Define the conditions as Lean definitions
def is_square (s : ℝ) : Prop := s ^ 2 = 2
def is_right_triangle (a b c : ℝ) : Prop := a ^ 2 + b ^ 2 = c ^ 2

-- Lean theorem statement for the proof problem
theorem area_of_triangle_gkl (s a : ℝ) (h_s : is_square s) (h_right : is_right_triangle a a s) :
  (1 / 2) * a ^ 2 = 1 / 2 :=
by 
  have h_s' : s = sqrt 2 := by sorry -- side length of the square
  have h_a : a = 1 := by sorry -- side lengths of the triangle
  have area : (1 / 2) * a ^ 2 = 1 / 2 := by sorry
  exact area

end area_of_triangle_gkl_l299_299290


namespace xz_in_right_triangle_l299_299298

theorem xz_in_right_triangle
  (X Y Z : Type)
  [metric_space X] [metric_space Y] [metric_space Z]
  (XY XZ YZ : Real) 
  (h1 : ∠ X Y Z = 90°)
  (h2 : YZ = 26)
  (h3 : tan ∠ Z = 3 * sin ∠ Z) 
  : XZ = 26 / 3 := 
sorry

end xz_in_right_triangle_l299_299298


namespace min_sum_distance_is_correct_l299_299231

-- Define the conditions
def line1 (x y : ℝ) : Prop := x - y + 5 = 0
def line2 (x : ℝ) : Prop := x + 4 = 0
def parabola (x y : ℝ) : Prop := y^2 = 16 * x
def focus : ℝ × ℝ := (4, 0)

-- Define the distance function from a point to a line
def distance_to_line (p : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  abs (A * p.1 + B * p.2 + C) / sqrt (A^2 + B^2)

noncomputable def min_distance_sum (P : ℝ × ℝ) (H : parabola P.1 P.2) : ℝ :=
  distance_to_line P 1 (-1) 5 + distance_to_line P 1 0 4

-- Define the problem statement
theorem min_sum_distance_is_correct : min_distance_sum focus sorry = 9 * sqrt 2 / 2 :=
sorry

end min_sum_distance_is_correct_l299_299231


namespace moles_of_NaHSO4_l299_299898

theorem moles_of_NaHSO4 (moles_NaOH moles_H2SO4 : ℕ) (h₁ : moles_NaOH = 2) (h₂ : moles_H2SO4 = 2) :
  ∃ (moles_NaHSO4 : ℕ), moles_NaHSO4 = 2 :=
by {
  use 2,
  sorry
}

end moles_of_NaHSO4_l299_299898


namespace time_for_B_to_complete_work_l299_299806

theorem time_for_B_to_complete_work 
  (A B C : ℝ)
  (h1 : A = 1 / 4) 
  (h2 : B + C = 1 / 3) 
  (h3 : A + C = 1 / 2) :
  1 / B = 12 :=
by
  -- Proof is omitted, as per instruction.
  sorry

end time_for_B_to_complete_work_l299_299806


namespace best_ketchup_deal_l299_299864

def cost_per_ounce (price : ℝ) (ounces : ℝ) : ℝ := price / ounces

theorem best_ketchup_deal :
  let price_10oz := 1 in
  let ounces_10oz := 10 in
  let price_16oz := 2 in
  let ounces_16oz := 16 in
  let price_25oz := 2.5 in
  let ounces_25oz := 25 in
  let price_50oz := 5 in
  let ounces_50oz := 50 in
  let price_200oz := 10 in
  let ounces_200oz := 200 in
  let money := 10 in
  (∀ p o, cost_per_ounce p o ≥ cost_per_ounce price_200oz ounces_200oz) ∧ money = price_200oz :=
1
by
  sorry

end best_ketchup_deal_l299_299864


namespace second_train_speed_l299_299503

theorem second_train_speed 
    (v : ℕ) -- speed of the second train
    (distance1 : ℕ := 30) -- distance covered by first train in 1 hour
    (meeting_distance : ℕ := 60) -- meeting point from Mumbai
    (speed1 : ℕ := 30) -- speed of the first train
    (time1 : ℕ := 1) -- time taken by first train to cover additional distance
    (time2 : ℕ := 1) -- time taken by second train to cover the meeting distance
    : (v = 60) :=
    begin 
        sorry
    end

end second_train_speed_l299_299503


namespace find_s_value_l299_299163

noncomputable def find_s : ℝ → Prop :=
  λ s : ℝ, (∃ m b : ℝ, m = (1- (-2)) / (0 - (-6)) ∧ b = 1 ∧ 
          (7 = m * s + b))

theorem find_s_value : ∃ s : ℝ, find_s s :=
begin
  use 12,
  unfold find_s,
  rw [mul_eq_mul_right_iff, sub_eq_sub_iff, add_eq_add_iff],
  use [(1 - (-2)) / (0 - (-6)), 1],
  split,
  { ring_nf, simp only [one_div], norm_num },
  { exact ⟨rfl, rfl⟩ }
end

end find_s_value_l299_299163


namespace overall_class_average_l299_299468

theorem overall_class_average :
  ∀ (class_avg_15 class_avg_50 class_avg_35 : ℝ) (p15 p50 p35 : ℝ),
  p15 = 0.15 → p50 = 0.50 → p35 = 0.35 →
  class_avg_15 = 100 → class_avg_50 = 78 → class_avg_35 = 63 →
  (p15 * class_avg_15 + p50 * class_avg_50 + p35 * class_avg_35).round = 76 :=
by
  intros class_avg_15 class_avg_50 class_avg_35 p15 p50 p35
  intros hp15 hp50 hp35 ha15 ha50 ha35
  rw [hp15, hp50, hp35, ha15, ha50, ha35]
  simp only [mul_add, Real.round]
  sorry

end overall_class_average_l299_299468


namespace horses_meet_time_sum_of_digits_12_l299_299877

theorem horses_meet_time :
  ∃ (T : ℕ), T > 0 ∧
  (∃ (horses : set ℕ) (h : horses ⊆ {1, 2, 3, 4, 5, 6, 7, 8}) | horses.card = 4 ∧
    ∀ k ∈ horses, T % k = 0) ∧ (T = 12)
  :=
begin
  sorry
end

theorem sum_of_digits_12 : 
  (1 + 2 = 3) := 
begin
  norm_num
end

end horses_meet_time_sum_of_digits_12_l299_299877


namespace milk_leftover_l299_299533

theorem milk_leftover 
  (total_milk : ℕ := 24)
  (kids_percent : ℝ := 0.80)
  (cooking_percent : ℝ := 0.60)
  (neighbor_percent : ℝ := 0.25)
  (husband_percent : ℝ := 0.06) :
  let milk_after_kids := total_milk * (1 - kids_percent)
  let milk_after_cooking := milk_after_kids * (1 - cooking_percent)
  let milk_after_neighbor := milk_after_cooking * (1 - neighbor_percent)
  let milk_after_husband := milk_after_neighbor * (1 - husband_percent)
  milk_after_husband = 1.3536 :=
by 
  -- skip the proof for simplicity
  sorry

end milk_leftover_l299_299533


namespace parabola_equation_l299_299938

theorem parabola_equation (P : ℝ × ℝ) :
  let d1 := dist P (-3, 0)
  let d2 := abs (P.1 - 2)
  (d1 = d2 + 1 ↔ P.2^2 = -12 * P.1) :=
by
  intro d1 d2
  sorry

end parabola_equation_l299_299938


namespace circle_area_increase_125_percent_l299_299642

-- The original radius and area of the circle
variable (r : ℝ)

-- New radius after a 50% increase
def new_radius := 1.5 * r

-- Areas corresponding to original and new radii
def original_area := Real.pi * r^2
def new_area := Real.pi * (new_radius r)^2

-- Calculation of the area increase
def area_increase := (new_area r) - (original_area r)

-- Percentage increase in area
def percentage_increase := (area_increase r) / (original_area r) * 100

-- The theorem to prove
theorem circle_area_increase_125_percent : (percentage_increase r) = 125 := 
by
  sorry

end circle_area_increase_125_percent_l299_299642


namespace probability_exactly_5_heads_in_7_flips_l299_299044

theorem probability_exactly_5_heads_in_7_flips : 
  let outcome_space := (finset.range 2).product (finset.range 2).product (finset.range 2).product (finset.range 2).product (finset.range 2).product (finset.range 2).product (finset.range 2),
      heads := λ (outcome : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ), (outcome.1 + outcome.2.1 + outcome.2.2.1 + outcome.2.2.2.1 + outcome.2.2.2.2.1 + outcome.2.2.2.2.2.1 + outcome.2.2.2.2.2.2) = 5 in
  (outcome_space.filter heads).card / outcome_space.card = 21 / 128 := 
by
  sorry

end probability_exactly_5_heads_in_7_flips_l299_299044


namespace decomposition_of_x_l299_299060

def vec3 := ℝ × ℝ × ℝ

def x : vec3 := (0, -8, 9)
def p : vec3 := (0, -2, 1)
def q : vec3 := (3, 1, -1)
def r : vec3 := (4, 0, 1)

theorem decomposition_of_x :
  (2 : ℝ) • p + (-4 : ℝ) • q + (3 : ℝ) • r = x :=
sorry

end decomposition_of_x_l299_299060


namespace cookies_problem_l299_299891

theorem cookies_problem (e : ℚ) 
  (Fiona_has : Fiona_has_cookies e) 
  (Greg_has : Greg_has_cookies e) 
  (Helen_has : Helen_has_cookies e) 
  (total_cookies : Total_cookies e) : 
  e = 45 / 17 := 
sorry

def Fiona_has_cookies (e : ℚ) : Prop := 
  3 * e

def Greg_has_cookies (e : ℚ) : Prop := 
  2 * (3 * e) = 6 * e

def Helen_has_cookies (e : ℚ) : Prop := 
  4 * (2 * (3 * e)) = 24 * e

def Total_cookies (e : ℚ) : Prop := 
  e + 3 * e + 6 * e + 24 * e = 90

end cookies_problem_l299_299891


namespace value_of_m_l299_299260

noncomputable def given_equation_is_quadratic (m : ℝ) : Prop :=
  ((m - 2) * (x^|m|) + 3 * m * x + 1 = 0) ∧ (|m| = 2) ∧ (m - 2 ≠ 0)

theorem value_of_m (m : ℝ) : given_equation_is_quadratic m → m = -2 :=
sorry

end value_of_m_l299_299260


namespace polar_to_cartesian_curvey2_2x_min_intersect_len_l299_299647

theorem polar_to_cartesian_curvey2_2x (θ ρ : ℝ) : 
  (ρ = Real.sec θ + Real.tan θ) -> (ρ * ρ * Real.sin θ * Real.sin θ = 2 * ρ * Real.cos θ) :=
sorry

theorem min_intersect_len (α : ℝ) (hα : 0 < α ∧ α < π) : 
  ∃ l A B, (line_l_intersects_curve_C_at_points l A B α) -> 
  (∃ ABmin, (∀ α, ABmin = distance_AB A B α) ∧ (∃ α, ABmin = 2)) :=
sorry

end polar_to_cartesian_curvey2_2x_min_intersect_len_l299_299647


namespace prove_correct_set_of_equations_l299_299662

-- Define variables x and y
variables (x y : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := 9 * x - y = 4
def condition2 : Prop := y - 6 * x = 5

-- Define the correct answer using these conditions
def correct_answer : Prop := 
  (9 * x - y = 4) ∧ (y - 6 * x = 5)

-- Statement to prove that the correct answer equals the conditions
theorem prove_correct_set_of_equations : condition1 ∧ condition2 ↔ correct_answer :=
by 
  -- the proof is omitted as per instructions
  sorry

end prove_correct_set_of_equations_l299_299662


namespace min_value_of_y_l299_299910

theorem min_value_of_y (x : ℝ) (h : x > 3) : y = x + 1/(x-3) → y ≥ 5 :=
sorry

end min_value_of_y_l299_299910


namespace systematic_sampling_correct_l299_299280

theorem systematic_sampling_correct :
  ∃ sequence : List ℕ, 
  (sequence.length = 6) ∧ 
  (∀ i, i < sequence.length → sequence.get i = 4 + i * 5) ∧ 
  (sequence = [4, 9, 14, 19, 24, 29]) :=
by {
  sorry
}

end systematic_sampling_correct_l299_299280


namespace value_of_a5_max_sum_first_n_value_l299_299579

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

def sum_first_n (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem value_of_a5 (a d a5 : ℤ) :
  a5 = 4 ↔ (2 * a + 4 * d) + (a + 4 * d) + (a + 8 * d) = (a + 5 * d) + 8 :=
  sorry

theorem max_sum_first_n_value (a d : ℤ) (n : ℕ) (max_n : ℕ) :
  a = 16 →
  d = -3 →
  (∀ i, sum_first_n a d i ≤ sum_first_n a d max_n) →
  max_n = 6 :=
  sorry

end value_of_a5_max_sum_first_n_value_l299_299579


namespace students_taking_geometry_or_science_but_not_both_l299_299549

def students_taking_both : ℕ := 15
def students_taking_geometry : ℕ := 30
def students_taking_science_only : ℕ := 18

theorem students_taking_geometry_or_science_but_not_both : students_taking_geometry - students_taking_both + students_taking_science_only = 33 := by
  sorry

end students_taking_geometry_or_science_but_not_both_l299_299549


namespace probability_point_above_parabola_l299_299699

noncomputable def set_of_integers : Set ℤ := {2, 3, 4, 5, 6, 7, 8, 9, 10}

def lies_above_parabola (a b : ℤ) (x : ℤ) : Prop :=
  b > a * x^2 - b * x

theorem probability_point_above_parabola :
  ∃! (p : ℚ), p = 2 / 81 ∧ 
    let points := { (a, b) | a ∈ set_of_integers ∧ b ∈ set_of_integers } in
    let valid_points := { (a, b) ∈ points | lies_above_parabola a b (a + 1) } in
    (∑ x in valid_points, 1) / (∑ x in points, 1) = p := 
begin
  sorry
end

end probability_point_above_parabola_l299_299699


namespace inequality_correct_l299_299588

variable {a b c : ℝ}

theorem inequality_correct (h : a * b < 0) : |a - c| ≤ |a - b| + |b - c| :=
sorry

end inequality_correct_l299_299588


namespace gerald_paid_l299_299962

theorem gerald_paid (G : ℝ) (h : 0.8 * G = 200) : G = 250 := by
  sorry

end gerald_paid_l299_299962


namespace choose_two_common_courses_l299_299982

theorem choose_two_common_courses :
  let courses : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
  let personA_choices : Finset Char := courses.erase 'A'
  let personB_choices : Finset Char := courses.erase 'F'
  -- Define the set of common choices that can be chosen by both persons
  let common_choices : Finset Char := personA_choices ∩ personB_choices
  -- The final count of the number of ways in which two people can choose exactly two courses in common
  (univ.filter (λ (p : Finset Char × Finset Char),
    p.1.card = 3 ∧ p.2.card = 3 ∧ (p.1 ∩ p.2).card = 2 ∧
    p.1 ⊆ personA_choices ∧ p.2 ⊆ personB_choices)).card = 42 :=
by 
  -- actual proof is omitted
  sorry

end choose_two_common_courses_l299_299982


namespace pounds_of_fudge_sold_l299_299483

variable (F : ℝ)
variable (price_fudge price_truffles price_pretzels total_revenue : ℝ)

def conditions := 
  price_fudge = 2.50 ∧
  price_truffles = 60 * 1.50 ∧
  price_pretzels = 36 * 2.00 ∧
  total_revenue = 212 ∧
  total_revenue = (price_fudge * F) + price_truffles + price_pretzels

theorem pounds_of_fudge_sold (F : ℝ) (price_fudge price_truffles price_pretzels total_revenue : ℝ) 
  (h : conditions F price_fudge price_truffles price_pretzels total_revenue ) :
  F = 20 :=
by
  sorry

end pounds_of_fudge_sold_l299_299483


namespace right_triangle_345_l299_299058

theorem right_triangle_345 :
  ∃ a b c : ℕ, a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2 :=
by {
  use [3, 4, 5],
  split; try {refl},
  split; try {refl},
  exact (nat.pow_two 3).trans (nat.add_right_inj (9 + 16)).mpr rfl
}

#print right_triangle_345 -- To verify correctness

end right_triangle_345_l299_299058


namespace infinite_geometric_series_sum_l299_299545

noncomputable def a : ℚ := 5 / 3
noncomputable def r : ℚ := -1 / 2

theorem infinite_geometric_series_sum : 
  ∑' (n : ℕ), a * r^n = 10 / 9 := 
by sorry

end infinite_geometric_series_sum_l299_299545


namespace sum_of_common_divisors_l299_299764

theorem sum_of_common_divisors :
  let common_divisors := {d | d ∣ 36 ∧ d ∣ 72 ∧ d ∣ 108 ∧ d ∣ 12 ∧ d ∣ 96 ∧ d ∣ 180 ∧ d > 0}
  let four_common_divisors := {1, 2, 3, 6}
  (common_divisors = four_common_divisors) → (finset.sum finset.univ id four_common_divisors) = 12 :=
by
  sorry

end sum_of_common_divisors_l299_299764


namespace divisibility_by_15_l299_299571

theorem divisibility_by_15 (n : ℕ) : 
  let num := 8 * 10000 + n * 1000 + 9 * 100 + 4 * 10 + 5
  in (num % 15 = 0) ↔ (n % 3 = 1) := 
by
  sorry

end divisibility_by_15_l299_299571


namespace smallest_rel_prime_to_180_l299_299562

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact nat.succ_pos' 6 },
  split,
  { show gcd 7 180 = 1, from by norm_num },
  { intros y hy,
    cases hy with hy1 hy2,
    rw ← nat.lt_succ_iff,
    exact nat.lt_of_succ_le_succ hy1 }
end

end smallest_rel_prime_to_180_l299_299562


namespace geometric_sequence_first_term_l299_299032

theorem geometric_sequence_first_term (a r : ℝ) 
  (h1 : a * r^2 = 3)
  (h2 : a * r^4 = 27) : 
  a = - (real.sqrt 9) / 9 :=
by
  sorry

end geometric_sequence_first_term_l299_299032


namespace convex_32gon_minimum_perimeter_l299_299479

noncomputable def minimum_perimeter_32gon_grid : ℝ :=
  4 + 4 * Real.sqrt 2 + 8 * Real.sqrt 5 + 8 * Real.sqrt 10 + 8 * Real.sqrt 13

theorem convex_32gon_minimum_perimeter :
  ∀ (v : Fin 32 → ℤ × ℤ),
    (∀ i, 
       (v i).fst ≠ (v ((i + 1) % 32)).fst ∨ 
       (v i).snd ≠ (v ((i + 1) % 32)).snd) ∧
    (∑ i, (v ((i + 1) % 32)).1 - (v i).1 = 0 ∧
     ∑ i, (v ((i + 1) % 32)).2 - (v i).2 = 0) ∧
    ∑ i, ((v ((i + 1) % 32)).1 - (v i).1) ^ 2 + 
        ((v ((i + 1) % 32)).2 - (v i).2) ^ 2 =
    (4 + 4 * Real.sqrt 2 + 8 * Real.sqrt 5 + 8 * Real.sqrt 10 + 8 * Real.sqrt 13)^2 :=
  sorry

end convex_32gon_minimum_perimeter_l299_299479


namespace men_in_second_group_l299_299804

noncomputable theory

variables {x m w : ℝ}

def condition1 (m w : ℝ) : Prop := 3 * m + 8 * w = x * m + 2 * w

def condition2 (m w : ℝ) : Prop := 3 * m + 2 * w = (4 / 7) * (3 * m + 8 * w)

theorem men_in_second_group (h1 : condition1 m w) (h2 : condition2 m w) : x = 6 :=
by sorry

end men_in_second_group_l299_299804


namespace revenue_from_full_price_tickets_l299_299826

theorem revenue_from_full_price_tickets (f h p : ℕ) (H1 : f + h = 150) (H2 : f * p + h * (p / 2) = 2450) : 
  f * p = 1150 :=
by 
  sorry

end revenue_from_full_price_tickets_l299_299826


namespace part_a_l299_299314

def A (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, f (x * y) = x * f y

theorem part_a (f : ℝ → ℝ) (h : A f) : ∀ x y : ℝ, f (x + y) = f x + f y :=
sorry

end part_a_l299_299314


namespace two_digit_sums_of_six_powers_of_two_l299_299969

def powers_of_2 : set ℕ := {1, 2, 4, 8, 16, 32, 64}

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem two_digit_sums_of_six_powers_of_two : 
    { n : ℕ // is_two_digit n ∧ ∃ s : set ℕ, s ⊆ powers_of_2 ∧ s.card = 6 ∧ s.sum = n }.card = 2 := 
by 
  sorry

end two_digit_sums_of_six_powers_of_two_l299_299969


namespace area_ΔABD_dist_CD_l299_299433

open Real

-- Definitions based on conditions
def Point := (ℝ × ℝ)  -- Assuming 2D coordinates for simplicity
def A : Point := (0, 0)
def B : Point := (sqrt 3, 0)
def C : Point := (x_C, y_C)  -- arbitrary point
def D : Point := (x_D, y_D)  -- arbitrary point

-- Angles and distances from conditions
def angle_BAC := 30 * pi / 180
def angle_DAC := 45 * pi / 180
def angle_ABD := 45 * pi / 180
def angle_DBC := 75 * pi / 180
def dist_AB := sqrt 3

-- Theorem 1: Area of triangle ABD
theorem area_ΔABD : 
    let BD := sqrt (6) + sqrt (2) / 2 in
    let area := (1/2) * dist_AB * BD * sin angle_ABD
    in area = (3 + sqrt 3) / 4 := 
by sorry

-- Theorem 2: Distance between points C and D
theorem dist_CD : 
    let BD := sqrt(6) + sqrt(2)/2 in
    let BC := sqrt 3 in
    let CD_sq := BC^2 + BD^2 - 2 * BC * BD * cos angle_DBC
    in sqrt CD_sq = sqrt 5 :=
by sorry

end area_ΔABD_dist_CD_l299_299433


namespace direction_and_distance_travel_records_total_fuel_consumption_number_of_times_passing_gas_station_l299_299484

-- Define the travel records as a list of integers
def travelRecords : List Int := [10, -8, 6, -13, 7, -12, 3, -1]

-- Define the fuel consumption rate
def fuelConsumptionRate : Float := 0.05

-- Define the gas station location in kilometers east of the guard post
def gasStationLocation : Int := 6

-- Statement 1: Prove the direction and distance of location A relative to the guard post
theorem direction_and_distance_travel_records :
  let totalDisplacement := travelRecords.sum;
  totalDisplacement < 0 → totalDisplacement.abs = 8 := by
  sorry

-- Statement 2: Prove the total fuel consumption for the patrol that day
theorem total_fuel_consumption :
  let totalDistanceTraveled := travelRecords.map Int.abs |>.sum;
  totalDistanceTraveled = 60 → totalDistanceTraveled * fuelConsumptionRate = 3 := by
  sorry

-- Statement 3: Prove the number of times the police officer passed the gas station
theorem number_of_times_passing_gas_station :
  let positions : List Int := travelRecords.scanl (· + ·) 0;
  (positions.enum.zip positions.tail.enum).count (λ ⟨(i, pos1), (j, pos2)⟩ => (pos1 <= gasStationLocation ∧ pos2 > gasStationLocation)
                                                     ∨ (pos1 > gasStationLocation ∧ pos2 <= gasStationLocation)) = 4 := by
  sorry

end direction_and_distance_travel_records_total_fuel_consumption_number_of_times_passing_gas_station_l299_299484


namespace difference_of_numbers_l299_299422

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 34800) (h2 : b % 25 = 0) (h3 : b / 100 = a) : b - a = 32112 := by
  sorry

end difference_of_numbers_l299_299422


namespace slope_of_line_l299_299501

-- Define the coordinates of the points
def point1 := (0, 2)
def point2 := (498, 998)

-- Define the slope calculation function
def slope (p1 p2 : ℕ × ℕ) : ℚ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Prove that the slope of the line passing through the given points is 2
theorem slope_of_line : slope point1 point2 = 2 := 
by
  -- The detailed steps of the proof are omitted
  sorry

end slope_of_line_l299_299501


namespace comb_plus_perm_eq_l299_299478

def comb (n r : ℕ) : ℕ := n.choose r
def perm (n r : ℕ) : ℕ := nat.desc_fac n r

theorem comb_plus_perm_eq : comb 15 3 + perm 15 2 = 665 :=
by 
  -- combination part: C(15, 3)
  -- permutation part: A(15, 2)
  -- final result: 665
  sorry

end comb_plus_perm_eq_l299_299478
