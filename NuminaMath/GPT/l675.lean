import MathLib
import Mathlib
import Mathlib.Algebra.AbsoluteValue
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Algebra.BigOperators.Finprod
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Definitions
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Quadratic
import Mathlib.Analysis.Calculus.Slope
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Modeq
import Mathlib.Data.List
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Perm.Basic
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.LinearAlgebra.Det
import Mathlib.NumberTheory.Basic
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Tactic

namespace peter_completes_remaining_work_in_14_days_l675_675630

-- Define the conditions and the theorem
variable (W : ℕ) (work_done : ℕ) (remaining_work : ℕ)

theorem peter_completes_remaining_work_in_14_days
  (h1 : Matt_and_Peter_rate = (W/20))
  (h2 : Peter_rate = (W/35))
  (h3 : Work_done_in_12_days = (12 * (W/20)))
  (h4 : Remaining_work = (W - (12 * (W/20))))
  : (remaining_work / Peter_rate)  = 14 := sorry

end peter_completes_remaining_work_in_14_days_l675_675630


namespace cos_17pi_over_4_l675_675421

theorem cos_17pi_over_4 : Real.cos (17 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end cos_17pi_over_4_l675_675421


namespace solve_ellipse_correct_m_l675_675883

noncomputable def ellipse_is_correct_m : Prop :=
  ∃ (m : ℝ), 
    (m > 6) ∧
    ((m - 2) - (10 - m) = 4) ∧
    (m = 8)

theorem solve_ellipse_correct_m : ellipse_is_correct_m :=
sorry

end solve_ellipse_correct_m_l675_675883


namespace relation_between_y_l675_675258

/-- Definition of the points on the parabola y = -(x-3)^2 - 4 --/
def pointA (y₁ : ℝ) : Prop := y₁ = -(1/4 - 3)^2 - 4
def pointB (y₂ : ℝ) : Prop := y₂ = -(1 - 3)^2 - 4
def pointC (y₃ : ℝ) : Prop := y₃ = -(4 - 3)^2 - 4 

/-- Relationship between y₁, y₂, y₃ for given points on the quadratic function --/
theorem relation_between_y (y₁ y₂ y₃ : ℝ) 
  (hA : pointA y₁)
  (hB : pointB y₂)
  (hC : pointC y₃) : 
  y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

end relation_between_y_l675_675258


namespace smallest_n_l675_675958

noncomputable def S (n : ℕ) := {k | 2 ≤ k ∧ k ≤ n}

def partition (n : ℕ) (A B : set ℕ) := A ∪ B = S n ∧ A ∩ B = ∅

def exists_ab_equals_c (A : set ℕ) : Prop :=
  ∃ a b c, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a * b = c

theorem smallest_n (n : ℕ) (h : n ≥ 2) :
  (∀ A B, partition n A B → (exists_ab_equals_c A ∨ exists_ab_equals_c B)) ↔ n = 16 :=
sorry

end smallest_n_l675_675958


namespace triangle_altitude_median_angle_bisector_outside_l675_675841

theorem triangle_altitude_median_angle_bisector_outside (A B C D E F : Type) [Triangle A B C] (is_altitude : is_altitude A D) (is_median : is_median A E) (is_angle_bisector : is_angle_bisector A F) : 
  (lies_outside_triangle A D ∨ lies_outside_triangle A E ∨ lies_outside_triangle A F ∨ lies_outside_triangle A B C) :=
sorry

end triangle_altitude_median_angle_bisector_outside_l675_675841


namespace length_of_EF_l675_675463

variables {A B C D E F : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variables (triangle_ABC : A) (triangle_DEF : D)
variables {AB DE BC EF : ℝ}

-- Given conditions
variable (sim_ABC_DEF : ∃ (A B C D E F : Type*) 
                          [metric_space A] [metric_space B] [metric_space C] 
                          [metric_space D] [metric_space E] [metric_space F],
                          ∀ (triangle_ABC : A) (triangle_DEF : D),
                          ∃ (AB DE BC EF : ℝ),
                          (triangle_ABC = triangle_DEF)
                        )
variable (ratio_AB_DE : AB / DE = 1 / 2)
variable (BC_length : BC = 2)

-- Prove that EF = 4
theorem length_of_EF (h1 : sim_ABC_DEF) (h2 : ratio_AB_DE) (h3 : BC_length) : EF = 4 :=
sorry

end length_of_EF_l675_675463


namespace seventh_number_in_sequence_l675_675397

-- Define the sequence pattern
def sequence (n : ℕ) : ℕ :=
  3 * 2 ^ n

-- Define the problem statement
theorem seventh_number_in_sequence : sequence 6 = 192 :=
by
  -- Proof omitted
  sorry

end seventh_number_in_sequence_l675_675397


namespace circle_area_l675_675260

open Real

/-- Points A = (4, 15) and B = (12, 9) lie on the circle ω. The tangent lines to ω at A and B
intersect at a point on the x-axis. Prove that the area of ω is 306π. -/
theorem circle_area
  (A B : ℝ × ℝ)
  (hA : A = (4, 15))
  (hB : B = (12, 9))
  (hTangentIntersect : ∃ C : ℝ × ℝ, C.2 = 0 ∧ tangent_point (circle_thru A B) A B C) :
  ∃ r : ℝ, pi * r ^ 2 = 306 * pi := 
sorry

end circle_area_l675_675260


namespace cover_plane_with_two_identical_squares_l675_675206

def plane : Type := ℝ × ℝ

noncomputable def tiling_with_unique_squares (squares : set (set plane)) : Prop :=
  ∃ (s1 s2 : set plane) (size : ℝ) (uniq_squares : set (set plane)),
    (s1 ≠ s2) ∧
    (∀ s ∈ uniq_squares, (∃ s' ∈ squares, s ≠ s' ∧ s ≠ s1 ∧ s ≠ s2 ∧ noncolliding_squares s s')) ∧
    (tile_plane squares) ∧
    (card {p ∈ squares | size_of_square p = size} = 2) ∧
    (∀ (p : set plane), p ∈ squares → (∃ size', size_of_square p = size' ∧ (p = s1 ∨ p = s2 ∨ p ∈ uniq_squares)))

theorem cover_plane_with_two_identical_squares (squares : set (set plane)) :
  tiling_with_unique_squares squares :=
sorry

end cover_plane_with_two_identical_squares_l675_675206


namespace remainder_of_M_l675_675608

def T : ℕ → ℕ := λ n, sorry -- (Function to generate the increasing sequence whose binary representation has exactly 9 ones)

def M : ℕ := T 1500

theorem remainder_of_M (hM: M = 33023) : M % 1500 = 23 :=
by {
  have : M = 33023 := hM,
  rw this,
  norm_num,
}

end remainder_of_M_l675_675608


namespace evaluate_propositions_l675_675119

variables (m n : Line) (α β : Plane)

-- Define parallel and perpendicular relations
def parallel (l : Line) (p : Plane) : Prop := ∃ k : Line, k ⊆ p ∧ parallel_lines l k
def perpendicular (l : Line) (p : Plane) : Prop := ∀ t : Line, t ⊆ p → perpendicular_lines l t

theorem evaluate_propositions :
  ¬ ((parallel m α ∧ parallel n α) → parallel m n) ∧
  ((parallel m α ∧ perpendicular n α) → perpendicular n m) ∧
  ((perpendicular m α ∧ parallel m β) → perpendicular_planes α β) :=
by
  -- Proposition 1 is incorrect
  sorry

  -- Proposition 2 is correct
  sorry

  -- Proposition 3 is correct
  sorry

end evaluate_propositions_l675_675119


namespace marble_solid_color_percentage_l675_675763

theorem marble_solid_color_percentage (a b : ℕ) (h1 : a = 5) (h2 : b = 85) : a + b = 90 := 
by
  sorry

end marble_solid_color_percentage_l675_675763


namespace find_b12_l675_675611

def sequence (b : ℕ → ℤ) := b 1 = 2 ∧ ∀ m n : ℕ, 0 < m → 0 < n → b (m + n) = b m + b n + m * n + 1

theorem find_b12 (b : ℕ → ℤ) (h : sequence b) : b 12 = 101 :=
sorry

end find_b12_l675_675611


namespace arrange_balls_l675_675205

/-- 
Define a type for balls to specify their identity.
We assume the balls are numbered from 1 to 9.
-/
def Balls : Type := Fin 9

/-- 
Define a property that states ball 5 and ball 7 can have only one neighbor, 
which must be ball 1 if they are included in the arrangement.
-/
def adjacent_to_one (b : Balls) (arr : List Balls) : Prop :=
  match b with
  | ⟨4, _⟩ => ∃ i, List.nth arr i = some b ∧ (List.nth arr (i - 1) = some (Fin.mk 0 sorry) ∨ List.nth arr (i + 1) = some (Fin.mk 0 sorry))
  | ⟨6, _⟩ => ∃ i, List.nth arr i = some b ∧ (List.nth arr (i - 1) = some (Fin.mk 0 sorry) ∨ List.nth arr (i + 1) = some (Fin.mk 0 sorry))
  | _ => true

/-- 
Define the main theorem. 
-/
theorem arrange_balls (arr : List Balls) :
  ¬(List.length arr = 9 ∧ (adjacent_to_one (Fin.mk 5 sorry) arr ∧ adjacent_to_one (Fin.mk 7 sorry) arr)) ∧
  ∃ arr' : List Balls, List.length arr' = 8 ∧ adjacent_to_one (Fin.mk 5 sorry) arr' := 
begin
  sorry
end

end arrange_balls_l675_675205


namespace find_a_l675_675873

theorem find_a (x y a : ℤ) (h1 : x = 1) (h2 : y = -1) (h3 : 2 * x - a * y = 3) : a = 1 := by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end find_a_l675_675873


namespace parallelogram_base_length_l675_675751

theorem parallelogram_base_length 
  (area : ℝ)
  (b h : ℝ)
  (h_area : area = 128)
  (h_altitude : h = 2 * b) 
  (h_area_eq : area = b * h) : 
  b = 8 :=
by
  -- Proof goes here
  sorry

end parallelogram_base_length_l675_675751


namespace line_slope_is_neg_half_l675_675698

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := x + 2 * y - 4 = 0

-- The main theorem to be proved
theorem line_slope_is_neg_half : ∀ (x y : ℝ), line_eq x y → (∃ m b : ℝ, y = m * x + b ∧ m = -1/2) := by
  sorry

end line_slope_is_neg_half_l675_675698


namespace f_f_neg_two_l675_675858

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then x^2 else Real.log x / Real.log 2

theorem f_f_neg_two : f (f (-2)) = 2 :=
by
  sorry

end f_f_neg_two_l675_675858


namespace B_M_N_collinear_l675_675038

theorem B_M_N_collinear 
  (A B C D E F M N : Point)
  (h_hex : regular_hexagon A B C D E F)
  (h_ACCE : diagonal A C ∧ diagonal C E)
  (h_M_div : divides M A C (real.sqrt 3 / 3))
  (h_N_div : divides N C E (real.sqrt 3 / 3)) :
  collinear B M N :=
begin
  sorry
end

end B_M_N_collinear_l675_675038


namespace integral_of_x_minus_x_squared_l675_675081

theorem integral_of_x_minus_x_squared :
  ∫ x in 0..1, (x - x^2) = 1 / 6 := 
by
  sorry

end integral_of_x_minus_x_squared_l675_675081


namespace max_x_squared_plus_y_squared_l675_675267

theorem max_x_squared_plus_y_squared (a x y : ℝ) 
  (h1 : x + y = a + 1)
  (h2 : xy = a^2 - 7a + 16) : x^2 + y^2 ≤ 32 := 
sorry

end max_x_squared_plus_y_squared_l675_675267


namespace inequality_proof_l675_675476

theorem inequality_proof (a b c : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b) (hc : 1 ≤ c) :
  (a + b + c) / 4 ≥ (sqrt (a * b - 1)) / (b + c) + (sqrt (b * c - 1)) / (c + a) + (sqrt (c * a - 1)) / (a + b) :=
by
  sorry

end inequality_proof_l675_675476


namespace range_of_b_l675_675182

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
if x > 0 then 
  (b - 3 / 2) * x + b - 1 
else 
  -x^2 + (2 - b) * x

theorem range_of_b (b : ℝ) :
  (∀ x y : ℝ, x ≤ y → f b x ≤ f b y) ↔ (3 / 2 < b ∧ b ≤ 2) :=
begin
  sorry
end

end range_of_b_l675_675182


namespace determine_digit_X_l675_675821

theorem determine_digit_X (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 9) (h : 510 / X = 10 * 4 + X + 2 * X) : X = 8 :=
sorry

end determine_digit_X_l675_675821


namespace probability_arithmetic_progression_correct_l675_675334

noncomputable def total_outcomes : ℕ := 6 * 6 * 6

noncomputable def prime_numbers_within_range : List ℕ := [2, 3, 5]

noncomputable def is_arithmetic_progression (nums : List ℕ) : Prop :=
nums.length = 3 ∧ (∀ i j, i < j ∧ j < 3 → (nums[j] - nums[i] == nums[1] - nums[0]))

noncomputable def favorable_outcomes : ℕ := 6  -- 3! for permutations of [2, 3, 5]

noncomputable def probability_arithmetic_progression : ℚ :=
(favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_arithmetic_progression_correct :
  probability_arithmetic_progression = 1 / 36 := 
begin
  -- the proof is not required
  sorry
end

end probability_arithmetic_progression_correct_l675_675334


namespace sum_sqrt_31_l675_675915

noncomputable theory

open Classical

-- Define the context and assumptions
variables (a b c : ℝ)
variables (h1 : a^2 + b^2 + c^2 = 64) (h2 : a*b + b*c + c*a = 30) (h3 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)

-- Define the theorem to be proven
theorem sum_sqrt_31 : a + b + c = 2 * Real.sqrt 31 :=
by sorry

end sum_sqrt_31_l675_675915


namespace sara_golf_balls_total_l675_675280

-- Define the conditions
def dozens := 16
def dozen_to_balls := 12

-- The final proof statement
theorem sara_golf_balls_total : dozens * dozen_to_balls = 192 :=
by
  sorry

end sara_golf_balls_total_l675_675280


namespace problem1_problem2_l675_675370

variable (a : ℝ) -- Declaring a as a real number

-- Proof statement for Problem 1
theorem problem1 : (a + 2) * (a - 2) = a^2 - 4 :=
sorry

-- Proof statement for Problem 2
theorem problem2 (h : a ≠ -2) : (a^2 - 4) / (a + 2) + 2 = a :=
sorry

end problem1_problem2_l675_675370


namespace price_per_crab_l675_675948

theorem price_per_crab (crab_baskets_per_week : ℕ) (crabs_per_basket : ℕ) (collections_per_week : ℕ) (total_money : ℕ) :
  crab_baskets_per_week = 3 → crabs_per_basket = 4 → collections_per_week = 2 → total_money = 72 →
  (total_money / (crab_baskets_per_week * crabs_per_basket * collections_per_week)) = 3 :=
begin
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  norm_num,
  sorry
end

end price_per_crab_l675_675948


namespace postman_pat_miles_walked_l675_675641

theorem postman_pat_miles_walked :
  let pedometer_resets := 50
  let steps_per_reset := 90000
  let final_steps := 30000
  let steps_per_mile := 1500
  let total_steps := pedometer_resets * steps_per_reset + final_steps
  let total_miles := total_steps / steps_per_mile
  abs ((3000 : ℤ) - total_miles) < abs ((3200 : ℤ) - total_miles) ∧
  abs ((3000 : ℤ) - total_miles) < abs ((3600 : ℤ) - total_miles) ∧
  abs ((3000 : ℤ) - total_miles) < abs ((3900 : ℤ) - total_miles) ∧
  abs ((3000 : ℤ) - total_miles) < abs ((4200 : ℤ) - total_miles) ∧
  abs ((3000 : ℤ) - total_miles) < abs ((4500 : ℤ) - total_miles)
  sorry

end postman_pat_miles_walked_l675_675641


namespace find_expression_for_a_n_l675_675020

noncomputable def a_sequence (a : ℕ → ℕ) :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 2^n

theorem find_expression_for_a_n (a : ℕ → ℕ) (h : a_sequence a) (initial : a 1 = 1) :
  ∀ n : ℕ, a (n + 1) = 2^(n + 1) - 1 :=
by
  sorry

end find_expression_for_a_n_l675_675020


namespace sqrt_nested_eq_l675_675444

theorem sqrt_nested_eq (y : ℝ) (hy : 0 ≤ y) :
  Real.sqrt (y * Real.sqrt (y * Real.sqrt (y * Real.sqrt y))) = y ^ (9 / 4) :=
by
  sorry

end sqrt_nested_eq_l675_675444


namespace symmetry_about_y_axis_l675_675352

theorem symmetry_about_y_axis :
    ∀ (x y : ℝ),
    (x^2 - y^2 = 1) ↔ ((-x)^2 - y^2 = 1) ∧
    ¬((x^2 - x + y^2 = 1) ↔ ((-x)^2 - (-x) + y^2 = 1)) ∧
    ¬((x^2y + xy^2 = 1) ↔ ((-x)^2y + (-x)y^2 = 1)) ∧
    ¬((x - y = 1) ↔ ((-x) - y = 1)) :=
by
    intros x y
    sorry

end symmetry_about_y_axis_l675_675352


namespace total_savings_l675_675360

def CompanyX.price : ℕ := 575
def CompanyX.surcharge_percent : ℕ := 4
def CompanyX.installation_charge : ℕ := 8250 / 100
def CompanyX.warranty_fee : ℕ := 125
def CompanyX.maintenance_fee : ℕ := 50

def CompanyY.price : ℕ := 530
def CompanyY.surcharge_percent : ℕ := 3
def CompanyY.installation_charge : ℕ := 9300 / 100
def CompanyY.warranty_fee : ℕ := 150
def CompanyY.maintenance_fee : ℕ := 40

def surcharge (price : ℕ) (percent : ℕ) : ℕ :=
  price * percent / 100

def total_charge (price surcharge installation warranty maintenance : ℕ) : ℕ :=
  price + surcharge + installation + warranty + maintenance

def CompanyX.total_charge : ℕ :=
  total_charge CompanyX.price (surcharge CompanyX.price CompanyX.surcharge_percent)
               CompanyX.installation_charge CompanyX.warranty_fee CompanyX.maintenance_fee

def CompanyY.total_charge : ℕ :=
  total_charge CompanyY.price (surcharge CompanyY.price CompanyY.surcharge_percent)
               CompanyY.installation_charge CompanyY.warranty_fee CompanyY.maintenance_fee

def savings : ℕ :=
  CompanyX.total_charge - CompanyY.total_charge

theorem total_savings : savings = 2660 / 100 := 
by 
  -- Total savings sould be calculated as 26.60 which can be computed as 2660 / 100
  sorry

end total_savings_l675_675360


namespace ellipse_standard_equation_hyperbola_equation_l675_675372

-- Ellipse problem
theorem ellipse_standard_equation (c e a b : ℝ) (h1 : 2 * c = 8) (h2 : e = 0.8) (h3 : e = c / a) 
  (h4 : b = Real.sqrt (a^2 - c^2)) : 
  (a = 5) → (b = 3) → (c = 4) → (e = 0.8) → 
  (std_eqn : (a^2 = 25) ∧ (b^2 = 9)) := 
by { sorry }

-- Hyperbola problem
theorem hyperbola_equation (x y m : ℝ) (h1 : ∀ k, (k ≠ 0) → (den_y : y^2 / (4 * k)) = (num_x : x^2 / (3 * k)))
  (h2 : -2 = m) : 
  (std_eqn : (x := 3, y := -2) ∧ x^2 / 6 - y^2 / 8 = 1) :=
by { sorry }

end ellipse_standard_equation_hyperbola_equation_l675_675372


namespace triangle_ABC_B_120_AC_sqrt19_AB_2_BC_3_l675_675550

theorem triangle_ABC_B_120_AC_sqrt19_AB_2_BC_3 :
  ∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C], (B = 120) → 
  (dist A C = sqrt 19) → (dist A B = 2) → (dist B C = 3).
sorry

end triangle_ABC_B_120_AC_sqrt19_AB_2_BC_3_l675_675550


namespace total_buses_needed_l675_675019

-- Definitions derived from conditions
def total_students : ℕ := 635
def full_bus_capacity : ℕ := 45
def reserved_bus_remaining_capacity : ℕ := 20

-- The total number of buses needed, proving the correct answer.
theorem total_buses_needed : 
  ∃ n : ℕ, n = (1 + Int.ceil (↑(total_students - reserved_bus_remaining_capacity) / full_bus_capacity)) ∧ n = 15 :=
by 
  sorry

end total_buses_needed_l675_675019


namespace union_of_M_and_N_l675_675896

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {x | 1 < x ∧ x < 3}

theorem union_of_M_and_N : M ∪ N = {x | -1 < x ∧ x < 3} := by
  sorry

end union_of_M_and_N_l675_675896


namespace comics_ordering_l675_675975

def numSpidermanComics := 6
def numArchieComics := 5
def numGarfieldComics := 4
def numBatmanComics := 3

def totalWaysOrderingComics : ℕ :=
  (numSpidermanComics.factorial *
   numArchieComics.factorial *
   numGarfieldComics.factorial *
   numBatmanComics.factorial *
   4.factorial)

theorem comics_ordering : totalWaysOrderingComics = 1_244_160 := by
  sorry

end comics_ordering_l675_675975


namespace sum_of_f_of_arithmetic_seq_l675_675376

noncomputable def f (x : ℝ) : ℝ := x^3 - 9 * x^2 + 20 * x - 4

def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def a : ℕ → ℝ
| 1 := _
| 2 := _
| 3 := _
...

theorem sum_of_f_of_arithmetic_seq (a : ℕ → ℝ) (h : arithmetic_seq a) (h₁ : a 5 = 3) :
  (f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) + f (a 7) + f (a 8) + f (a 9)) = 18 :=
sorry

end sum_of_f_of_arithmetic_seq_l675_675376


namespace distinct_values_S_when_k_equals_2_l675_675965

def i := Complex.I

noncomputable def S (k n : ℤ) : ℂ := i^(k * n) + i^-(k * n)

theorem distinct_values_S_when_k_equals_2 :
  ∀ k : ℤ, k = 2 → (∃ values : set ℂ, values = {x | ∃ n : ℤ, x = S k n} ∧ values.card = 2) :=
by
  sorry

end distinct_values_S_when_k_equals_2_l675_675965


namespace range_of_m_l675_675151

noncomputable def f (x m : ℝ) : ℝ := real.exp x - m * x + 1
noncomputable def f_deriv (x m : ℝ) : ℝ := real.exp x - m
noncomputable def is_vertical_to (y_prime : ℝ) : Prop := y_prime = -1 / real.exp 1

theorem range_of_m (m : ℝ) :
    (∃ x : ℝ, f_deriv x m = -1 / real.exp 1) →

    m > 1 / real.exp 1 :=
sorry

end range_of_m_l675_675151


namespace trigonometric_identity_l675_675064

theorem trigonometric_identity : (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 1 - Real.sqrt 3 :=
by
  sorry

end trigonometric_identity_l675_675064


namespace range_of_a_l675_675466

variable (a : ℝ)

def set_A : set ℝ := {x | x^2 - 2 * x - 3 ≥ 0}
def set_B (a : ℝ) : set ℝ :=
  if a = 0 then {0}
  else if a > 0 then {x | a < x ∧ x < 2 * a}
  else {x | 2 * a < x ∧ x < a}

theorem range_of_a (a : ℝ) :
  (set_B a ⊆ set_A) →
  a ≥ 3 ∨ a ≤ -1 :=
sorry

end range_of_a_l675_675466


namespace chess_club_not_playing_any_l675_675561

theorem chess_club_not_playing_any (total_members : ℕ) (chess_players : ℕ) (checkers_players : ℕ) (both_players : ℕ) 
  (h1 : total_members = 70) (h2 : chess_players = 45) (h3 : checkers_players = 38) (h4 : both_players = 25) : 
  total_members - (chess_players - both_players + checkers_players - both_players + both_players) = 12 := 
by sorry

end chess_club_not_playing_any_l675_675561


namespace number_of_squares_in_50th_ring_l675_675074

theorem number_of_squares_in_50th_ring :
  ∀ (n : ℕ), (n = 50) → (4 * n + 4 = 204) :=
by
  intro n
  intro h
  rw h
  rfl

end number_of_squares_in_50th_ring_l675_675074


namespace unique_intersecting_line_l675_675039

variables {Point Line Segment : Type}
variables (A B C D P Q R : Point)
variables (AB CD : Line)
variables (PQ : Segment)
variable (is_midpoint : Point → Point → Point → Prop)
variable (is_not_parallel : Line → Line → Prop)
variable (passes_through : Line → Point → Prop)
variable (intersects_at_midpoint : Line → Line → Point → Prop)

-- Hypotheses
variables (hP : is_midpoint P B C)
          (hQ : is_midpoint Q A D)
          (hParallel : ¬ is_not_parallel AB CD)
          
-- Theorem statement
theorem unique_intersecting_line (hR : R ∈ PQ) :
  ∃! (L : Line), passes_through L R ∧ intersects_at_midpoint L AB Q ∧ intersects_at_midpoint L CD Q :=
sorry

end unique_intersecting_line_l675_675039


namespace smallest_prime_factor_in_C_l675_675651

-- Let C be the set of given numbers
def C : Set ℕ := {70, 72, 75, 76, 78}

-- Define a function to find the smallest prime factor
def smallest_prime_factor (n : ℕ) : ℕ :=
  if 2 ∣ n then 2
  else if 3 ∣ n then 3
  else sorry -- Implementation for finding the smallest prime factor

-- The main theorem to prove
theorem smallest_prime_factor_in_C : ∀ n ∈ C, smallest_prime_factor n = 2 :=
by
  intros n hn
  cases hn with
  | inl h70 => rw h70; exact rfl
  | inr hn =>
    cases hn with
    | inl h72 => rw h72; exact rfl
    | inr hn =>
      cases hn with
      | inl h75 => rw h75; -- Provide justification for 75 here using the supporting smallest_prime_factor function
      | inr hn =>
        cases hn with
        | inl h76 => rw h76; exact rfl
        | inr hn =>
          rw hn; exact rfl

end smallest_prime_factor_in_C_l675_675651


namespace triangle_side_length_l675_675544

theorem triangle_side_length (A B C : Type) [euclidean_geometry A B C]
  (angle_B : B = 120 * π / 180)
  (AC : ℝ) (AB : ℝ) 
  (h₁ : AC = sqrt 19)
  (h₂ : AB = 2) :
  ∃ BC, BC = 3 :=
begin
  -- proof to be filled in
  sorry
end

end triangle_side_length_l675_675544


namespace value_of_x2017_l675_675486

-- Definitions and conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f (x) < f (y)

def arithmetic_sequence (x : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, x (n + 1) = x n + d

variables (f : ℝ → ℝ) (x : ℕ → ℝ)
variables (d : ℝ)
variable (h_odd : is_odd_function f)
variable (h_increasing : is_increasing_function f)
variable (h_arithmetic : arithmetic_sequence x 2)
variable (h_condition : f (x 7) + f (x 8) = 0)

-- Define the proof goal
theorem value_of_x2017 : x 2017 = 4019 :=
by
  sorry

end value_of_x2017_l675_675486


namespace initial_money_l675_675215

theorem initial_money (spent allowance total initial : ℕ) 
  (h1 : spent = 2) 
  (h2 : allowance = 26) 
  (h3 : total = 29) 
  (h4 : initial - spent + allowance = total) : 
  initial = 5 := 
by 
  sorry

end initial_money_l675_675215


namespace min_value_problem_l675_675468

theorem min_value_problem (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 57 * a + 88 * b + 125 * c ≥ 1148) :
  240 ≤ a^3 + b^3 + c^3 + 5 * a^2 + 5 * b^2 + 5 * c^2 :=
sorry

end min_value_problem_l675_675468


namespace polynomial_root_exists_l675_675964

noncomputable def exists_real_root (P Q : Polynomial ℝ) : Prop :=
  ∃ x : ℝ, P.eval (x - 1) = Q.eval (x + 1)

theorem polynomial_root_exists
  (P Q : Polynomial ℝ)
  (hP : P.degree = 2014)
  (hQ : Q.degree = 2014)
  (hmonicP : P.leading_coeff = 1)
  (hmonicQ : Q.leading_coeff = 1)
  (hneq : ∀ x : ℝ, P.eval x ≠ Q.eval x) : 
  exists_real_root P Q :=
  sorry

end polynomial_root_exists_l675_675964


namespace smallest_n_directed_graph_conditions_l675_675094

-- Define the conditions as Lean predicates

structure DirGraph (n : ℕ) :=
  (edges : fin n → fin n → Prop)
  (at_most_one_edge : ∀ (A B : fin n), edges A B → (∀ C, (C = B ∨ C ≠ B → ¬ edges A C)))
  (trans_edge : ∀ (A B : fin n), edges A B → ∃ C, edges A C ∧ edges C B)
  (exists_out_edge : ∀ (A : fin n), ∃ B, edges A B)

-- State the theorem ensuring the smallest n is 7.

theorem smallest_n_directed_graph_conditions : ∃ (n : ℕ), 
  (∀ (G : DirGraph n), (G.edges = (λ (i j : fin n), (i+1 % n = j ∨ i+2 % n = j ∨ i+4 % n = j))) ∧ n = 7) := 
sorry

end smallest_n_directed_graph_conditions_l675_675094


namespace train_length_l675_675403

theorem train_length 
  (bridge_length train_length time_seconds v : ℝ)
  (h1 : bridge_length = 300)
  (h2 : time_seconds = 36)
  (h3 : v = 40) :
  (train_length = v * time_seconds - bridge_length) →
  (train_length = 1140) := by
  -- solve in a few lines
  -- This proof is omitted for the purpose of this task
  sorry

end train_length_l675_675403


namespace smallest_nonneg_integer_divisible_by_4_l675_675347

theorem smallest_nonneg_integer_divisible_by_4 :
  ∃ n : ℕ, (7 * (n - 3)^5 - n^2 + 16 * n - 30) % 4 = 0 ∧ ∀ m : ℕ, m < n -> (7 * (m - 3)^5 - m^2 + 16 * m - 30) % 4 ≠ 0 :=
by
  use 1
  sorry

end smallest_nonneg_integer_divisible_by_4_l675_675347


namespace horse_revolutions_l675_675388

noncomputable def carousel_revolutions (r1 r2 d1 : ℝ) : ℝ :=
  (d1 * r1) / r2

theorem horse_revolutions :
  carousel_revolutions 30 10 40 = 120 :=
by
  sorry

end horse_revolutions_l675_675388


namespace triangle_interior_angle_ge_60_l675_675263

theorem triangle_interior_angle_ge_60 (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A < 60) (h3 : B < 60) (h4 : C < 60) : false := 
by
  sorry

end triangle_interior_angle_ge_60_l675_675263


namespace trigonometric_identity_l675_675059

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180))
  = (4 * Real.sin (10 * Real.pi / 180) / Real.sin (40 * Real.pi / 180)) :=
by sorry

end trigonometric_identity_l675_675059


namespace segment_to_axis_distance_l675_675396

noncomputable def distance_from_segment_to_axis (l : ℝ) (α : ℝ) : ℝ :=
  (l / 2) * real.sqrt (- real.cos (2 * α))

theorem segment_to_axis_distance (l α : ℝ)
  (hα : (π / 4) < α ∧ α < (3 * π / 4)) :
  distance_from_segment_to_axis l α = (l / 2) * real.sqrt (-real.cos (2 * α)) := 
by
  sorry

end segment_to_axis_distance_l675_675396


namespace find_m_l675_675536

theorem find_m (m : ℕ) (h : 10^(m-1) < 2^512 ∧ 2^512 < 10^m): 
  m = 155 :=
sorry

end find_m_l675_675536


namespace gcd_expression_l675_675439

theorem gcd_expression (a b c d : ℤ) : 
  gcd ((a - b) * (b - c) * (c - d) * (d - a) * (a - c) * (b - d)) 12 = 12 := 
by
  sorry

end gcd_expression_l675_675439


namespace range_of_c_l675_675853

theorem range_of_c (c : ℝ) (hc1 : c > 0) (hc2 : c ≠ 1) :
  (0 < c ∧ c ≤ 1/2) ∨ (c ≥ 1 ∧ c > 1/2) :=
begin
  -- Skipping proof
  sorry
end

end range_of_c_l675_675853


namespace boxes_filled_l675_675639

theorem boxes_filled (total_toys toys_per_box : ℕ) (h1 : toys_per_box = 8) (h2 : total_toys = 32) :
  total_toys / toys_per_box = 4 :=
by
  sorry

end boxes_filled_l675_675639


namespace greatest_divisor_of_remainders_l675_675450

theorem greatest_divisor_of_remainders :
  let d := Nat.gcd 13953 20831 in
  d = 1 :=
by
  let d := Nat.gcd 13953 20831
  show d = 1
  sorry

end greatest_divisor_of_remainders_l675_675450


namespace valid_seatings_l675_675195

variable (People : Type) [Fintype People] [DecidableEq People]

-- Define specific individuals
variables (John Wilma Paul Anna Bob Cindy : People)

-- Define the total number of people
variable (TenPeople : Fintype People)

-- Define the function that calculates the valid number of seatings
noncomputable def numberOfValidSeatings
    (h₁: {John, Wilma, Paul} ⊆ People) 
    (h₂: {Anna, Bob, Cindy} ⊆ People)
    (total_seating: Fintype People) : ℕ :=
    10.factorial 
    - 2 * (8.factorial * 3.factorial) 
    + (8.factorial * 3.factorial * 3.factorial)

-- Define the theorem with the answer
theorem valid_seatings (h₁: {John, Wilma, Paul} ⊆ People) (h₂: {Anna, Bob, Cindy} ⊆ People)
    (total_seating: Fintype People) :
    numberOfValidSeatings John Wilma Paul Anna Bob Cindy TenPeople = 4596480 :=
by sorry

end valid_seatings_l675_675195


namespace total_price_of_order_l675_675383

theorem total_price_of_order :
  let num_ice_cream_bars := 225
  let price_per_ice_cream_bar := 0.60
  let num_sundaes := 125
  let price_per_sundae := 0.52
  (num_ice_cream_bars * price_per_ice_cream_bar + num_sundaes * price_per_sundae) = 200 := 
by
  -- The proof steps go here
  sorry

end total_price_of_order_l675_675383


namespace exists_cycle_removal_preserves_strong_connectivity_l675_675234

variables {V : Type*} [fintype V] 

-- Definition of directed graph G with the provided conditions
structure directed_graph (V : Type*) :=
(adj : V → V → Prop)
(no_self_loops : ∀ v, ¬ adj v v) 
(min_out_deg : ∀ v, 2 ≤ cardinal.mk {w | adj v w}.to_finset)
(min_in_deg : ∀ v, 2 ≤ cardinal.mk {u | adj u v}.to_finset)
-- Strong connectivity condition
(strongly_connected : ∀ u v, u ≠ v → (u = v) ∨ (∃ p : list V, p.head? = some u ∧ p.last? = some v ∧ ∀ (a b : V), (a,b) ∈ p.zip (p.tail) → adj a b))

-- Prove that there exists a cycle whose removal keeps the graph strongly connected
theorem exists_cycle_removal_preserves_strong_connectivity {G : directed_graph V} :
  ∃ (c : list V), ∀ (G' : directed_graph V), (∀ u, u ∈ c → directed_graph.min_out_deg G' u = directed_graph.min_out_deg G u) ∧ 
  (∀ v, v ∉ c → directed_graph.min_in_deg G' v = directed_graph.min_in_deg G v) → 
  directed_graph.strongly_connected G' :=
sorry

end exists_cycle_removal_preserves_strong_connectivity_l675_675234


namespace sum_of_remaining_is_perfect_square_l675_675728

theorem sum_of_remaining_is_perfect_square (k : ℕ) (hk : 0 < k) :
  let n := 50 * k in
  let S_n := n * (n + 1) / 2 in
  let S_k := 25 * k * (k + 1) in
  let S_remaining := S_n - S_k in
  ∃ m : ℕ, S_remaining = m * m :=
by 
  sorry

end sum_of_remaining_is_perfect_square_l675_675728


namespace simplify_expression_l675_675281

-- Define the variables and the polynomials
variables (y : ℤ)

-- Define the expressions
def expr1 := (2 * y - 1) * (5 * y^12 - 3 * y^11 + y^9 - 4 * y^8)
def expr2 := 10 * y^13 - 11 * y^12 + 3 * y^11 + y^10 - 9 * y^9 + 4 * y^8

-- State the theorem
theorem simplify_expression : expr1 = expr2 := by
  sorry

end simplify_expression_l675_675281


namespace greatest_n_with_222_digits_of_2_l675_675454

def a_n (n : ℕ) : ℚ := (2 * 10^(n + 1) - 20 - 18 * n) / 81

theorem greatest_n_with_222_digits_of_2 : ∃ n : ℕ, 
  (a_n n).num_digits = 222 ∧ (∀ m : ℕ, (a_n m).num_digits = 222 → m ≤ n) ∧ n = 222 := 
sorry

end greatest_n_with_222_digits_of_2_l675_675454


namespace number_of_boys_l675_675284

theorem number_of_boys 
    (B : ℕ) 
    (total_boys_sticks : ℕ := 15 * B)
    (total_girls_sticks : ℕ := 12 * 12)
    (sticks_relation : total_girls_sticks = total_boys_sticks - 6) : 
    B = 10 :=
by
    sorry

end number_of_boys_l675_675284


namespace problem1_monotonic_decreasing_problem2_monotonic_decreasing_pos_problem2_monotonic_decreasing_neg_l675_675518

-- Problem 1: Monotonicity of f(x) = 1 - 3x on ℝ
theorem problem1_monotonic_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → (1 - 3 * x1) > (1 - 3 * x2) :=
by
  -- Proof (skipped)
  sorry

-- Problem 2: Monotonicity of g(x) = 1/x + 2 on (0, ∞) and (-∞, 0)
theorem problem2_monotonic_decreasing_pos : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → (1 / x1 + 2) > (1 / x2 + 2) :=
by
  -- Proof (skipped)
  sorry

theorem problem2_monotonic_decreasing_neg : ∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 < x2 → (1 / x1 + 2) > (1 / x2 + 2) :=
by
  -- Proof (skipped)
  sorry

end problem1_monotonic_decreasing_problem2_monotonic_decreasing_pos_problem2_monotonic_decreasing_neg_l675_675518


namespace total_handshakes_is_790_l675_675922

-- Define the context of the two groups:
def groupA : Finset ℕ := {1, 2, ..., 30} -- Set of 30 people knowing each other
def groupB : Finset ℕ := {31, 32, ..., 50} -- Set of 20 people knowing no one

noncomputable def total_handshakes : ℕ :=
  let handshakes_between : ℕ := 30 * 20 in  -- Handshakes between Group A and Group B
  let handshakes_within_B : ℕ := Nat.choose 20 2 in -- Handshakes within Group B (choose 2 from 20)
  handshakes_between + handshakes_within_B

-- The theorem to prove
theorem total_handshakes_is_790 : total_handshakes = 790 := by
  sorry

end total_handshakes_is_790_l675_675922


namespace second_cube_weight_correct_third_cube_weight_correct_l675_675772

variable (s : ℝ) (ρ : ℝ) (V1 : ℝ) (V2 : ℝ) (V3 : ℝ)
variable (W1 : ℝ) (W2 : ℝ) (W3 : ℝ)

/-- First cube conditions -/
def first_cube_conditions (s : ℝ) (W1 : ℝ) : Prop :=
  V1 = s^3 ∧ W1 = 8 ∧ ρ = W1 / V1

/-- Second cube weight -/
def second_cube_weight (s : ℝ) (ρ : ℝ) (V2 : ℝ) : ℝ :=
  ρ * V2

/-- Third cube weight -/
def third_cube_weight (s : ℝ) (ρ : ℝ) (V3 : ℝ) : ℝ :=
  ρ * V3

theorem second_cube_weight_correct (s : ℝ) (W1 : ℝ) (ρ : ℝ) (V1 : ℝ) (V2 : ℝ) :
  first_cube_conditions s W1 →
  V2 = (4 * s)^3 →
  second_cube_weight s ρ V2 = 512 :=
by
  intros h_cond h_volume
  sorry

theorem third_cube_weight_correct (s : ℝ) (W1 : ℝ) (ρ : ℝ) (V1 : ℝ) (V3 : ℝ) :
  first_cube_conditions s W1 →
  V3 = s * s * (2 * s) →
  third_cube_weight s ρ V3 = 16 :=
by
  intros h_cond h_volume
  sorry

end second_cube_weight_correct_third_cube_weight_correct_l675_675772


namespace almond_butter_servings_l675_675000

def convert_mixed_to_fraction (a b : ℤ) (n : ℕ) : ℚ :=
  (a * n + b) / n

def servings (total servings_fraction : ℚ) : ℚ :=
  total / servings_fraction

theorem almond_butter_servings :
  servings (convert_mixed_to_fraction 35 2 3) (convert_mixed_to_fraction 2 1 2) = 14 + 4 / 15 :=
by
  sorry

end almond_butter_servings_l675_675000


namespace domain_of_f_2x_minus_1_l675_675135

theorem domain_of_f_2x_minus_1 (f : ℝ → ℝ) :
  (∀ x, 1 ≤ x → x ≤ 3 → (x ^ 2 - 1) ∈ set.Ici 0 ∩ set.Iic 8) →
  ∀ x, (1/2 : ℝ) ≤ x → x ≤ (9/2 : ℝ) → (2 * x - 1) ∈ set.Ici 0 ∩ set.Iic 8 :=
sorry

end domain_of_f_2x_minus_1_l675_675135


namespace sum_of_intersections_l675_675136

theorem sum_of_intersections (f g : ℝ → ℝ) (m : ℕ) (x y : ℕ → ℝ) 
  (h1 : ∀ x, f (-x) = 4 - f x) 
  (h2 : ∀ x, g x = (x - 2) / (x - 1) + x / (x + 1)) 
  (h3 : ∀ i, 1 ≤ i ∧ i ≤ m → f (x i) = g (x i) ∧ y i = f (x i)) :
  (∑ i in Finset.range (m + 1), (x i + y i)) = 2 * m := 
by
  sorry

end sum_of_intersections_l675_675136


namespace lines_intersect_on_diagonal_l675_675259

variables {A B C D M N O : Type*} [AddGroup A] [VAdd A B] [AffineSpace A B]
variables [AddGroup C] [VAdd C D] [AffineSpace C D]

-- Definition of M and N as the midpoints of sides BC and CD respectively
def is_midpoint (M : A) (B C : A) : Prop := M = (1 / 2) • (B + C)
def is_midpoint (N : A) (C D : A) : Prop := N = (1 / 2) • (C + D)

-- Condition that ABCD is a parallelogram
def is_parallelogram (A B C D : A) : Prop :=
((B - A) + (D - C) = 0) ∧ ((C - B) + (D - A) = 0)

-- Prove that lines DM and BN intersect on the diagonal AC
theorem lines_intersect_on_diagonal
  (A B C D M N O : A)
  (h_parallelogram : is_parallelogram A B C D)
  (h_midpoint_M : is_midpoint M B C)
  (h_midpoint_N : is_midpoint N C D) :
  ∃ O : A, O ∈ line_segment A C ∧ O ∈ line DM ∧ O ∈ line BN := 
sorry

end lines_intersect_on_diagonal_l675_675259


namespace measure_angle_PQR_l675_675937
noncomputable theory

-- Definitions for points and angles.
variables (P Q R S : Type) [Geometry P Q R S]

-- Definitions for angles and isosceles triangles.
variables (angle_PQS angle_QRS angle_RQS angle_PSQ angle_PQR : ℝ)

-- Given conditions.
axiom h1 : RQ = RS -- ∆ QRS is isosceles.
axiom h2 : angle QRS = 60 -- given.
axiom h3 : PQ = PS -- ∆ PQS is isosceles.
axiom h4 : angle QPS = 30 -- given.

-- Goal statement proving the measure of ∠PQR.
-- Ensure the statement reflects proof of ∠PQR == 15° given the conditions.
theorem measure_angle_PQR : angle PQR = 15 :=
by
  sorry

end measure_angle_PQR_l675_675937


namespace normal_distribution_probability_mixed_products_probability_l675_675330

/-- Given that Z follows a normal distribution N(200, 150), and standard deviation is approximately 12.2, prove that P(187.8 ≤ Z ≤ 212.2) is approximately 0.6827 -/
theorem normal_distribution_probability :
  (∀ (Z : ℝ), Z ∼ Normal 200 150) →
  P(187.8 ≤ Z ∧ Z ≤ 212.2) = 0.6827 := sorry

/-- Given the qualification rates of 0.95, 0.90, and 0.80 for manufacturers A, B, and C respectively, 
     and the proportions 2:3:5, prove that the probability that a randomly selected product is qualified is 0.86 -/
theorem mixed_products_probability :
  let p_A := 0.95,
      p_B := 0.90,
      p_C := 0.80,
      P_B1 := 2 / (2 + 3 + 5),
      P_B2 := 3 / (2 + 3 + 5),
      P_B3 := 5 / (2 + 3 + 5)
  in P_B1 * p_A + P_B2 * p_B + P_B3 * p_C = 0.86 := sorry

end normal_distribution_probability_mixed_products_probability_l675_675330


namespace inverse_transformation_l675_675107

variable {n : ℕ} {p q : ℤ} 
variables (a : ℤ → ℂ) (b : ℤ → ℂ)

-- Definitions of epsilon
def epsilon (n : ℕ) : ℂ :=
  Complex.exp (Complex.I * Real.pi / n)

-- Given transformation for b_q
def b_q (q : ℤ) : ℂ :=
  (1 / Complex.sqrt (2 * n)) * ∑ p in finset.Icc (-n : ℤ) (n+1 : ℤ), 
  (a p) * (epsilon n)^(p * q)

-- Goal statement to prove
theorem inverse_transformation :
  a p = (1 / Complex.sqrt (2 * n)) * ∑ q in finset.Icc (-n : ℤ) (n+1 : ℤ), 
  (b q) * (epsilon n)^(-p * q) := sorry

end inverse_transformation_l675_675107


namespace find_x_value_l675_675524

theorem find_x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x / 3 = y^3) (h2 : x / 6 = 3 * y) : x = 18 * Real.sqrt 6 := 
by 
  sorry

end find_x_value_l675_675524


namespace vanessa_ribbon_length_l675_675675

theorem vanessa_ribbon_length
  (area : ℝ)
  (extra_ribbon : ℝ)
  (pi_approx : ℝ)
  (h_area : area = 154)
  (h_extra_ribbon : extra_ribbon = 2)
  (h_pi_approx : pi_approx = 22 / 7)
  : 
  let radius := Real.sqrt (area * 7 / pi_approx)
  let circumference := 2 * pi_approx * radius
  in circumference + extra_ribbon = 46 := 
by
  sorry

end vanessa_ribbon_length_l675_675675


namespace find_smallest_n_l675_675241

noncomputable def smallest_n (X : Finset ℕ) (n : ℕ) : Prop :=
  ∃ (subsets : Finset (Finset ℕ)) (h₁ : subsets.card = 15),
  (∀ (A : Finset (Finset ℕ)), A.card = 7 → A ⊆ subsets → (A.bUnion id).card ≥ n) →
  ∃ (B : Finset (Finset ℕ)), B.card = 3 ∧ B ⊆ subsets ∧ (B.bInter id).nonempty

theorem find_smallest_n : smallest_n (Finset.range 56) 41 :=
by
  sorry

end find_smallest_n_l675_675241


namespace sum_of_digits_N_sum_of_digits_5040_l675_675952

def lcm_1_9 : ℕ := Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))))

def N : ℕ := 2 * lcm_1_9

theorem sum_of_digits_N : Nat.digits 10 N = [5, 0, 4, 0] :=
by
  sorry

theorem sum_of_digits_5040 : Nat.digits.sum (Nat.digits 10 N) = 9 :=
by
  sorry


end sum_of_digits_N_sum_of_digits_5040_l675_675952


namespace student_D_most_stable_l675_675847

-- Define the variances for students A, B, C, and D
def SA_squared : ℝ := 2.1
def SB_squared : ℝ := 3.5
def SC_squared : ℝ := 9
def SD_squared : ℝ := 0.7

-- Theorem stating that student D has the most stable performance
theorem student_D_most_stable :
  SD_squared < SA_squared ∧ SD_squared < SB_squared ∧ SD_squared < SC_squared := by
  sorry

end student_D_most_stable_l675_675847


namespace cosine_central_angle_subtended_minor_arc_PS_l675_675856

noncomputable def circle_radius : ℝ := 8
noncomputable def chord_RS_length : ℝ := 10
noncomputable def half_chord_RS_length : ℝ := chord_RS_length / 2
noncomputable def chord_PQ_bisection_point_T : Prop := 
  ∃ P Q R S T : ℝ, 
    T = (P + Q) / 2 ∧ 
    chord_RS_length = R + S ∧ 
    R = S ∧ 
    R = half_chord_RS_length ∧ 
    P ∈ {x : ℝ | x^2 + y^2 = circle_radius^2} ∧ 
    (∃ PQ : ℝ, PQ = P ∧ PQ = Q)

open_locale classical

theorem cosine_central_angle_subtended_minor_arc_PS (P Q R S T : ℝ) 
  (h1 : T = (P + Q) / 2 ∧
        chord_RS_length = R + S ∧
        R = S ∧
        R = half_chord_RS_length ∧
        P ∈ {x : ℝ | x^2 + y^2 = circle_radius^2} ∧
        (∃ PQ : ℝ, PQ = P ∧ PQ = Q)) :
  ∃ m n : ℝ, m = real.sqrt 39 ∧ n = circle_radius ∧ m * n = 8 * real.sqrt 39 :=
by
  sorry

end cosine_central_angle_subtended_minor_arc_PS_l675_675856


namespace cost_of_magazine_l675_675753

theorem cost_of_magazine (B M : ℝ) 
  (h1 : 2 * B + 2 * M = 26) 
  (h2 : B + 3 * M = 27) : 
  M = 7 := 
by 
  sorry

end cost_of_magazine_l675_675753


namespace teams_passing_time_l675_675023

theorem teams_passing_time
  (length_team1 : ℕ)
  (speed_team1 : ℕ)
  (length_team2 : ℕ)
  (speed_team2 : ℕ)
  (h1 : length_team1 = 50)
  (h2 : speed_team1 = 3)
  (h3 : length_team2 = 60)
  (h4 : speed_team2 = 2)
  :
  (length_team1 + length_team2) / (speed_team1 + speed_team2) = 22 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end teams_passing_time_l675_675023


namespace find_x_value_l675_675199

theorem find_x_value:
  ∃ x : ℕ, (let a := 13 + x in
            let b := x + 11 in
            let c := 11 + 2 * x in
            a + c = 42 ∧ x = 6) :=
by
  sorry

end find_x_value_l675_675199


namespace cows_died_last_year_l675_675390

theorem cows_died_last_year (D : ℕ) (initial : ℕ) (sold : ℕ) (increased : ℕ) (bought : ℕ) (gift : ℕ) (current : ℕ)
  (h_initial : initial = 39)
  (h_sold : sold = 6)
  (h_increased : increased = 24)
  (h_bought : bought = 43)
  (h_gift : gift = 8)
  (h_current : current = 83) :
  initial - D - sold + increased + bought + gift = current → D = 31 := by
  intro h_eq
  have eq1 : 39 - D - 6 + 24 + 43 + 8 = 83 := by
    rw [h_initial, h_sold, h_increased, h_bought, h_gift, h_current]
    exact h_eq
  have eq2 : 39 + 24 + 43 + 8 - 6 - D = 83 := by
    simpa using eq1
  have eq3 : 114 - D = 83 := by
    norm_num at eq2
    exact eq2
  have eq4 : D = 114 - 83 := by
    linarith
  have eq5 : D = 31 := by
    norm_num at eq4
    linarith
  exact eq5

end cows_died_last_year_l675_675390


namespace john_draw_on_back_l675_675217

theorem john_draw_on_back (total_pictures front_pictures : ℕ) (h1 : total_pictures = 15) (h2 : front_pictures = 6) : total_pictures - front_pictures = 9 :=
  by
  sorry

end john_draw_on_back_l675_675217


namespace distinct_segments_four_points_l675_675359

theorem distinct_segments_four_points (A B C D : Point) 
  (h1 : collinear A B C D) : 
  number_of_segments {A, B, C, D} = 6 := 
sorry

end distinct_segments_four_points_l675_675359


namespace chromium_percentage_l675_675196

theorem chromium_percentage (c1 c2 : ℝ) (w1 w2 : ℝ) (percentage1 percentage2 : ℝ) : 
  percentage1 = 0.1 → 
  percentage2 = 0.08 → 
  w1 = 15 → 
  w2 = 35 → 
  (c1 = percentage1 * w1) → 
  (c2 = percentage2 * w2) → 
  (c1 + c2 = 4.3) → 
  ((w1 + w2) = 50) →
  ((c1 + c2) / (w1 + w2) * 100 = 8.6) := 
by 
  sorry

end chromium_percentage_l675_675196


namespace circle_line_intersection_l675_675855

/-- Given a circle \( k \) with center \( O \) and a line \( e \) that does not pass through \( O \),
construct the intersection points of line \( e \) and the circle \( k \) using only a compass. --/
theorem circle_line_intersection (O : Point) (k : Circle O) (e : Line) (A B : Point) 
  (hne : ¬ e.Contains O) : 
  let O' := Reflection.over_line e O in
  let k' := Circle O' (k.radius) in
  ∀ P : Point, k.Contains P ↔ (k.Contains P ∧ e.Contains P) :=
sorry

end circle_line_intersection_l675_675855


namespace triangle_AB_BE_l675_675981

theorem triangle_AB_BE (A B C D E F : Type) (triangle : Triangle A B C)
  (hE : E ∈ line_segment B C) (hF : F ∈ angle_bisector B D)
  (hEF_parallel_AC : parallel E F A C) (hAF_AD : dist A F = dist A D) :
  dist A B = dist B E :=
sorry

end triangle_AB_BE_l675_675981


namespace compute_product_l675_675624

theorem compute_product (s : ℂ) (h1 : s^7 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) * (s^6 - 1) = 10 :=
sorry

end compute_product_l675_675624


namespace remainder_mod_x_minus_1_remainder_mod_x_squared_minus_1_remainder_mod_x_squared_minus_1_explicit_remainder_mod_x_squared_minus_1_coeff_l675_675441

noncomputable def P (x : ℝ) : ℝ :=
  x + x^3 + x^9 + x^27 + x^81 + x^243

theorem remainder_mod_x_minus_1 : (P 1) = 6 := by
  sorry

theorem remainder_mod_x_squared_minus_1 (x : ℂ) (h1 : x = 1 ∨ x = -1) : 
  P x = (x^2 - 1) * ((P x) / (x^2 - 1)) + if x = 1 then 6 else -6 := by
  sorry

theorem remainder_mod_x_squared_minus_1_explicit : 
  P (1 : ℝ) = 6 ∧ P (-1 : ℝ) = -6 :=
  by sorry

theorem remainder_mod_x_squared_minus_1_coeff : 
  ∃ a b : ℝ, a * 1 + b = 6 ∧ -a * 1 + b = -6 :=
  by sorry

end remainder_mod_x_minus_1_remainder_mod_x_squared_minus_1_remainder_mod_x_squared_minus_1_explicit_remainder_mod_x_squared_minus_1_coeff_l675_675441


namespace solve_super_prime_dates_2007_l675_675437

def isSuperPrimeDate (month day : ℕ) : Prop :=
  Nat.Prime month ∧ Nat.Prime day ∧ 
  (month ∈ [2, 3, 5, 7, 11]) ∧ 
  (day ≤ match month with
          | 2 => 28
          | 3 => 31
          | 5 => 31
          | 7 => 31
          | 11 => 30
          | _ => 0
        end)

def numSuperPrimeDates2007 : ℕ := 
  (List.filter (λ day, isSuperPrimeDate 2 day) (List.range 29)).length + 
  (List.filter (λ day, isSuperPrimeDate 3 day) (List.range 32)).length +
  (List.filter (λ day, isSuperPrimeDate 5 day) (List.range 32)).length +
  (List.filter (λ day, isSuperPrimeDate 7 day) (List.range 32)).length +
  (List.filter (λ day, isSuperPrimeDate 11 day) (List.range 31)).length

theorem solve_super_prime_dates_2007 : numSuperPrimeDates2007 = 50 := 
  sorry

end solve_super_prime_dates_2007_l675_675437


namespace equal_distances_l675_675951

-- Define the parallelogram and its properties
variables {A B C D : Point}
variable (h_parallelogram : Parallelogram A B C D)

-- Define the squares constructed on each side
variables {C1 D1 : Point}
variable (h_square1 : Square A B C1 D1)
variables {A2 C2 : Point}
variable (h_square2 : Square B C D2 A2)
variables {A3 B3 : Point}
variable (h_square3 : Square C D A3 B3)
variables {C4 B4 : Point}
variable (h_square4 : Square D A B4 C4)

-- Define the centers of squares created by segments as sides
variables {O_A O_B O_C O_D : Point}
variable (h_square_OA : CenterSquare B4 D1 O_A)
variable (h_square_OB : CenterSquare C1 A2 O_B)
variable (h_square_OC : CenterSquare D2 B3 O_C)
variable (h_square_OD : CenterSquare A3 C4 O_D)

-- State the theorem to be proven
theorem equal_distances
  (h1 : Parallelogram A B C D)
  (h2 : Square A B C1 D1)
  (h3 : Square B C D2 A2)
  (h4 : Square C D A3 B3)
  (h5 : Square D A B4 C4)
  (h6 : CenterSquare B4 D1 O_A)
  (h7 : CenterSquare C1 A2 O_B)
  (h8 : CenterSquare D2 B3 O_C)
  (h9 : CenterSquare A3 C4 O_D) :
  dist A O_A = dist B O_B ∧ dist B O_B = dist C O_C ∧ dist C O_C = dist D O_D := 
sorry

end equal_distances_l675_675951


namespace sequence_T_mod_1500_l675_675603

theorem sequence_T_mod_1500 :
  let T := {n : ℕ | nat.popcount n = 9}.to_finset.sort (≤)
  (T \u 1500 - 1) % 1500 = 500 := by
  sorry

end sequence_T_mod_1500_l675_675603


namespace janes_total_score_l675_675565

theorem janes_total_score (correct incorrect unanswered : ℕ) (score_correct score_incorrect score_unanswered total_score : ℝ) :
  correct = 18 → 
  incorrect = 12 → 
  unanswered = 5 → 
  score_correct = 1 → 
  score_incorrect = 0.5 → 
  score_unanswered = 0 → 
  total_score = correct * score_correct - incorrect * score_incorrect - unanswered * score_unanswered →
  total_score = 12 := 
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6]
  exact h7
  sorry

end janes_total_score_l675_675565


namespace arithmetic_sequence_problem_l675_675232

variables (a_n b_n : ℕ → ℚ)
variables (S_n T_n : ℕ → ℚ)
variable (n : ℕ)

axiom sum_a_terms : ∀ n : ℕ, S_n n = n / 2 * (a_n 1 + a_n n)
axiom sum_b_terms : ∀ n : ℕ, T_n n = n / 2 * (b_n 1 + b_n n)
axiom given_fraction : ∀ n : ℕ, n > 0 → S_n n / T_n n = (2 * n + 1) / (4 * n - 2)

theorem arithmetic_sequence_problem : 
  (a_n 10) / (b_n 3 + b_n 18) + (a_n 11) / (b_n 6 + b_n 15) = 41 / 78 :=
sorry

end arithmetic_sequence_problem_l675_675232


namespace sum_in_base_c_to_base_7_l675_675970

noncomputable def base_c_to_base_10 (c n : ℕ) :  ℕ :=
  match n with
  | 13 => c + 3
  | 18 => c + 8
  | 17 => c + 7
  | 4563 => 4 * c^3 + 5 * c^2 + 6 * c + 3
  | _ => 0

theorem sum_in_base_c_to_base_7 : 
  ∀ (c : ℕ),
  (base_c_to_base_10 c 13 * base_c_to_base_10 c 18 * base_c_to_base_10 c 17) = base_c_to_base_10 c 4563 →
  let t_in_base_10 := base_c_to_base_10 c 13 + base_c_to_base_10 c 18 + base_c_to_base_10 c 17 in
  t_in_base_10 = 39 →
  (39 / 7) = 5 ∧ (39 % 7) = 4 :=
by
  intros
  sorry

end sum_in_base_c_to_base_7_l675_675970


namespace not_exists_odd_product_sum_l675_675324

theorem not_exists_odd_product_sum (a b : ℤ) : ¬ (a * b * (a + b) = 20182017) :=
sorry

end not_exists_odd_product_sum_l675_675324


namespace bigIsland_has_odd_counties_structure_l675_675366

-- Definitions based on conditions
def island (n : Nat) : Prop :=
  ∃ grid: List (List (Bool × Bool)), -- Represents the grid of counties with diagonal roads
  (∀ row ∈ grid, row.length = n) ∧   -- The grid has n fiefs in each row
  (grid.length = n) ∧                -- The grid has n fiefs in each column
  (∃ path: List (Nat × Nat),          -- There exists a closed path on the grid
    (length path = n * n) ∧           -- The path visits every county exactly once
    (∀ (i j : Nat), (i, j) ∈ path → (i, j) ∈ grid))

-- Theorem corresponding to the proof problem
theorem bigIsland_has_odd_counties_structure :
  ∃ n, n % 2 = 1 ∧ island 9 :=
by
  exists 9
  split
  -- Proof of 9 being an odd number
  { exact Nat.mod_eq_of_lt (Nat.zero_lt_succ (Nat.succ_pos 7)) }
  -- Proof of the island structure (we use sorry as a placeholder)
  { sorry }

end bigIsland_has_odd_counties_structure_l675_675366


namespace trigonometric_identity_l675_675066

theorem trigonometric_identity : (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 1 - Real.sqrt 3 :=
by
  sorry

end trigonometric_identity_l675_675066


namespace triangle_ABC_B_120_AC_sqrt19_AB_2_BC_3_l675_675552

theorem triangle_ABC_B_120_AC_sqrt19_AB_2_BC_3 :
  ∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C], (B = 120) → 
  (dist A C = sqrt 19) → (dist A B = 2) → (dist B C = 3).
sorry

end triangle_ABC_B_120_AC_sqrt19_AB_2_BC_3_l675_675552


namespace sally_remaining_cards_l675_675988

variable (total_cards : ℕ) (torn_cards : ℕ) (bought_cards : ℕ)

def intact_cards (total_cards : ℕ) (torn_cards : ℕ) : ℕ := total_cards - torn_cards
def remaining_cards (intact_cards : ℕ) (bought_cards : ℕ) : ℕ := intact_cards - bought_cards

theorem sally_remaining_cards :
  intact_cards 39 9 - 24 = 6 :=
by
  -- sorry for proof
  sorry

end sally_remaining_cards_l675_675988


namespace increase_in_share_l675_675778

/-- Given a total cost of a car, earnings from a car wash, number of friends initially sharing 
    the cost, and the number of friends after Brad leaves, prove the increase in the amount each 
    remaining friend has to pay. -/
theorem increase_in_share
  (total_cost : ℕ)
  (car_wash_earnings : ℕ)
  (initial_friends : ℕ)
  (remaining_friends : ℕ)
  (initial_share : ℕ := (total_cost - car_wash_earnings) / initial_friends)
  (remaining_share : ℕ := (total_cost - car_wash_earnings) / remaining_friends) :
  total_cost = 1700 →
  car_wash_earnings = 500 →
  initial_friends = 6 →
  remaining_friends = 5 →
  (remaining_share - initial_share = 40) :=
by
  intros h1 h2 h3 h4
  have h₀ : total_cost - car_wash_earnings = 1700 - 500 := by rw [h1, h2]
  have h₁ : initial_share = (1700 - 500) / 6 := by rw [h₀, h3]
  have h₂ : remaining_share = (1700 - 500) / 5 := by rw [h₀, h4]
  have h₃ : (1700 - 500) = 1200 := by norm_num
  have h₄ : initial_share = 1200 / 6 := by rw [h₃]
  have h₅ : remaining_share = 1200 / 5 := by rw [h₃]
  have h₆ : 1200 / 6 = 200 := by norm_num
  have h₇ : 1200 / 5 = 240 := by norm_num
  rw [h₄, h₅, h₆, h₇]
  norm_num
  sorry

end increase_in_share_l675_675778


namespace probability_two_sixes_l675_675083

theorem probability_two_sixes :
  let n := 15
  let p_six := 1 / 6
  let p_not_six := 5 / 6
  let k := 2
  let binomial_coeff := Nat.choose n k
  let probability := binomial_coeff * (p_six ^ k) * (p_not_six ^ (n - k))
  probability ≈ 0.158 := 
sorry

end probability_two_sixes_l675_675083


namespace find_a5_l675_675129

open_locale big_operators

-- Definitions and conditions
def geometric_seq (a : ℕ → ℝ) :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

variable {a : ℕ → ℝ}

def a3_eq_1 := a 3 = 1
def a7_eq_9 := a 7 = 9

theorem find_a5 (h : geometric_seq a) (h3 : a3_eq_1) (h7 : a7_eq_9) : a 5 = 3 :=
sorry

end find_a5_l675_675129


namespace sum_of_reversed_base_digits_l675_675095

theorem sum_of_reversed_base_digits :
  let digits_match (n : ℕ) :=
    let base4_digits := n.digits 4
    let base5_digits := n.digits 5
    base4_digits.reverse = base5_digits
  (∑ n in Finset.filter digits_match (Finset.range 100), n) = 6 := sorry

end sum_of_reversed_base_digits_l675_675095


namespace sum_of_positive_real_numbers_l675_675817

theorem sum_of_positive_real_numbers (S : set ℝ) (h : S = {x : ℝ | 0 < x ∧ x ≤ 5 ∧ x = (⌈x^2⌉ + ⌈x⌉ * ⌊x⌋) / (⌈x⌉ + ⌊x⌋)}) :
  ∑ x in S, x = 85 :=
by
  sorry

end sum_of_positive_real_numbers_l675_675817


namespace kevin_vanessa_age_multiple_l675_675594

theorem kevin_vanessa_age_multiple :
  ∃ x k : ℕ, (16 + x = k * (2 + x)) → (16 + x) / (2 + x) = 4.5 := 
sorry

end kevin_vanessa_age_multiple_l675_675594


namespace number_of_distinct_intersections_of_curves_l675_675092

theorem number_of_distinct_intersections_of_curves (x y : ℝ) :
  (∀ x y, x^2 - 4*y^2 = 4) ∧ (∀ x y, 4*x^2 + y^2 = 16) → 
  ∃ (x1 x2 : ℝ) (y1 y2 : ℝ), 
    ((x1, y1) ≠ (x2, y2)) ∧
    ((x1^2 - 4*y1^2 = 4) ∧ (4*x1^2 + y1^2 = 16)) ∧
    ((x2^2 - 4*y2^2 = 4) ∧ (4*x2^2 + y2^2 = 16)) ∧
    ∀ (x' y' : ℝ), 
      ((x'^2 - 4*y'^2 = 4) ∧ (4*x'^2 + y'^2 = 16)) → 
      ((x', y') = (x1, y1) ∨ (x', y') = (x2, y2)) := 
sorry

end number_of_distinct_intersections_of_curves_l675_675092


namespace collinear_M_O_N_l675_675221

section

variables {A B C D : Type} [EuclideanGeometry A B C D] 
variables (Γ : Circle A) (O : Point A B C)
variables (M N : Point A B) (mid_AC : M = midpoint A C) (mid_BD : N = midpoint B D)

theorem collinear_M_O_N
  (incircle : Γ.incircle A B C D)
  (center : Γ.center = O) :
  collinear M O N :=
sorry

end

end collinear_M_O_N_l675_675221


namespace probability_of_total_score_is_38_l675_675379

variable (n : ℕ := 3)
variable (p : ℚ := 1/2)
variable (q : ℚ := 1/2)
variable (score_red : ℕ := 2)
variable (score_black : ℕ := 1)

theorem probability_of_total_score_is_38 :
  ∑ (s : Finset (Fin 8)) (h : s.1 + s.2 = 5),
    (n.choose s.1) * p^s.1 * q^(n - s.1) = 3 / 8 :=
sorry

end probability_of_total_score_is_38_l675_675379


namespace jelly_bean_match_probability_l675_675027

theorem jelly_bean_match_probability :
  let abe_total := 4 
  let bob_total := 8
  let abe_green := 2 
  let abe_red := 1 
  let abe_blue := 1
  let bob_green := 3 
  let bob_yellow := 2 
  let bob_blue := 1 
  let bob_red := 2
  let prob_green := (abe_green / abe_total) * (bob_green / bob_total)
  let prob_blue := (abe_blue / abe_total) * (bob_blue / bob_total)
  let prob_red := (abe_red / abe_total) * (bob_red / bob_total) in
  prob_green + prob_blue + prob_red = 9 / 32 :=
by
  sorry

end jelly_bean_match_probability_l675_675027


namespace terry_age_proof_l675_675919

theorem terry_age_proof
  (nora_age : ℕ)
  (h1 : nora_age = 10)
  (terry_age_in_10_years : ℕ)
  (h2 : terry_age_in_10_years = 4 * nora_age)
  (nora_age_in_5_years : ℕ)
  (h3 : nora_age_in_5_years = nora_age + 5)
  (sam_age_in_5_years : ℕ)
  (h4 : sam_age_in_5_years = 2 * nora_age_in_5_years)
  (sam_current_age : ℕ)
  (h5 : sam_current_age = sam_age_in_5_years - 5)
  (terry_current_age : ℕ)
  (h6 : sam_current_age = terry_current_age + 6) :
  terry_current_age = 19 :=
by
  sorry

end terry_age_proof_l675_675919


namespace inequality_non_empty_solution_l675_675699

theorem inequality_non_empty_solution (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 2 * x + 1 < 0) → a ≤ 1 := sorry

end inequality_non_empty_solution_l675_675699


namespace trigonometric_identity_l675_675065

theorem trigonometric_identity : (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 1 - Real.sqrt 3 :=
by
  sorry

end trigonometric_identity_l675_675065


namespace r_g_7_l675_675961

-- Define the functions r and g according to the problem statement
def r (x : ℝ) : ℝ := Real.sqrt (5 * x + 2)
def g (x : ℝ) : ℝ := 7 - r x

-- The theorem stating the proof problem
theorem r_g_7 : r (g 7) = Real.sqrt (37 - 5 * Real.sqrt 37) := 
by
  sorry

end r_g_7_l675_675961


namespace average_middle_numbers_l675_675672

theorem average_middle_numbers (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
  (h7 : a + b + c + d = 28) (h8 : (max a (max b (max c d))) - (min a (min b (min c d))) = 21) :
  (if a ≤ b ∧ b ≤ c ∧ c ≤ d then (b + c) / 2 
   else if a ≤ b ∧ b ≤ d ∧ d ≤ c then (b + d) / 2
   else if a ≤ c ∧ c ≤ b ∧ b ≤ d then (c + b) / 2
   else if a ≤ c ∧ c ≤ d ∧ d ≤ b then (c + d) / 2
   else if a ≤ d ∧ d ≤ b ∧ b ≤ c then (d + b) / 2
   else if a ≤ d ∧ d ≤ c ∧ c ≤ b then (d + c) / 2
   else if b ≤ a ∧ a ≤ c ∧ c ≤ d then (a + c) / 2
   else if b ≤ a ∧ a ≤ d ∧ d ≤ c then (a + d) / 2
   else if b ≤ c ∧ c ≤ a ∧ a ≤ d then (c + a) / 2
   else if b ≤ c ∧ c ≤ d ∧ d ≤ a then (c + d) / 2
   else if b ≤ d ∧ d ≤ a ∧ a ≤ c then (d + a) / 2
   else if b ≤ d ∧ d ≤ c ∧ c ≤ a then (d + c) / 2
   else if c ≤ a ∧ a ≤ b ∧ b ≤ d then (a + b) / 2
   else if c ≤ a ∧ a ≤ d ∧ d ≤ b then (a + d) / 2
   else if c ≤ b ∧ b ≤ a ∧ a ≤ d then (b + a) / 2
   else if c ≤ b ∧ b ≤ d ∧ d ≤ a then (b + d) / 2
   else if c ≤ d ∧ d ≤ a ∧ a ≤ b then (d + a) / 2
   else if c ≤ d ∧ d ≤ b ∧ b ≤ a then (d + b) / 2
   else if d ≤ a ∧ a ≤ b ∧ b ≤ c then (a + b) / 2
   else if d ≤ a ∧ a ≤ c ∧ c ≤ b then (a + c) / 2
   else if d ≤ b ∧ b ≤ a ∧ a ≤ c then (b + a) / 2
   else if d ≤ b ∧ b ≤ c ∧ c ≤ a then (b + c) / 2
   else if d ≤ c ∧ c ≤ a ∧ a ≤ b then (c + a) / 2
   else if d ≤ c ∧ c ≤ b ∧ b ≤ a then (c + b) / 2
   else 0) = 2.5 :=
sorry

end average_middle_numbers_l675_675672


namespace draw_at_least_one_even_ball_l675_675373

theorem draw_at_least_one_even_ball:
  -- Let the total number of ordered draws of 4 balls from 15 balls
  let total_draws := 15 * 14 * 13 * 12
  -- Let the total number of ordered draws of 4 balls where all balls are odd (balls 1, 3, ..., 15)
  let odd_draws := 8 * 7 * 6 * 5
  -- The number of valid draws containing at least one even ball
  total_draws - odd_draws = 31080 :=
by
  sorry

end draw_at_least_one_even_ball_l675_675373


namespace pythagorean_triple_square_l675_675984

theorem pythagorean_triple_square (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pythagorean : a^2 + b^2 = c^2) : ∃ k : ℤ, k^2 = (c - a) * (c - b) / 2 := 
sorry

end pythagorean_triple_square_l675_675984


namespace integral_square_ge_n_sq_l675_675967

variables (n : ℕ) (f : ℝ → ℝ)
open interval_integral

-- Assuming n is a positive integer
-- And f is a continuous function satisfying the condition
def satisfies_condition (n : ℕ) (f : ℝ → ℝ) : Prop :=
  ∀ k : ℕ, k < n → ∫ x in 0..1, x ^ k * f x = 1

theorem integral_square_ge_n_sq (hpos : 0 < n) (hcont : continuous_on f (set.Icc 0 1)) 
  (hcond : satisfies_condition n f) : 
  ∫ x in 0..1, (f x)^2 ≥ n^2 :=
sorry

end integral_square_ge_n_sq_l675_675967


namespace minimum_area_for_rectangle_l675_675413

theorem minimum_area_for_rectangle 
(length width : ℝ) 
(h_length_min : length = 4 - 0.5) 
(h_width_min : width = 5 - 1) :
length * width = 14 := 
by 
  simp [h_length_min, h_width_min]
  sorry

end minimum_area_for_rectangle_l675_675413


namespace constant_term_expansion_l675_675298

theorem constant_term_expansion (x : ℝ) : 
  let general_term (r : ℕ) := (Nat.choose 6 r) * x^(6 - r) * (- (1 / sqrt x))^r in
  6 - (3 / 2 * 4) = 0 →
  ∃ r = 4, general_term 4 = 15 := by
  sorry

end constant_term_expansion_l675_675298


namespace possible_integer_values_for_diagonal_l675_675101

-- Define the vertices of the quadrilateral
variables A B C D : Type
variables (AB BC CD DA x : ℕ)

-- Given conditions as axioms
axiom AB_val : AB = 6
axiom BC_val : BC = 12
axiom CD_val : CD = 14
axiom DA_val : DA = 10
axiom x_AC : x = AC

-- Define the proof problem
theorem possible_integer_values_for_diagonal : 
  6 < x ∧ x < 18 → (set.count (finset.Ico 7 18)).card = 11 :=
begin
  sorry
end

end possible_integer_values_for_diagonal_l675_675101


namespace exists_natural_m_l675_675620

def n (m : ℕ) : ℕ := (Nat.factors m!).count 2

theorem exists_natural_m :
  ∃ (m : ℕ), m > 1990^(1990) ∧ m = 3^(1990) + n m := sorry

end exists_natural_m_l675_675620


namespace problem_expression_equal_272_l675_675070

theorem problem_expression_equal_272 :
  (∏ k in finset.range 21, (1 + 15) / k.succ) /
  (∏ k in finset.range 19, (1 + 17) / k.succ) = 272 :=
by
  sorry

end problem_expression_equal_272_l675_675070


namespace compute_trig_expr_l675_675055

theorem compute_trig_expr :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 2 * Real.sec (10 * Real.pi / 180) :=
by
  sorry

end compute_trig_expr_l675_675055


namespace kenneth_earnings_l675_675218

theorem kenneth_earnings (E : ℝ) (h1 : E - 0.1 * E = 405) : E = 450 :=
sorry

end kenneth_earnings_l675_675218


namespace find_value_of_k_l675_675577

noncomputable def line_equation (k : ℝ) : ℝ → ℝ := λ x, k * x + 3

def point_A (k : ℝ) : ℝ × ℝ := (-3 / k, 0)

def point_B : ℝ × ℝ := (0, 3)

def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem find_value_of_k (k : ℝ) (h : distance (point_A k) point_B = 5) : k = 3/4 ∨ k = -3/4 :=
by
  sorry

end find_value_of_k_l675_675577


namespace six_digit_number_divisible_by_eleven_l675_675262

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def reverse_digits (a b c : ℕ) : ℕ :=
  100 * c + 10 * b + a

def concatenate_reverse (a b c : ℕ) : ℕ :=
  100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a

theorem six_digit_number_divisible_by_eleven (a b c : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9)
  (h₃ : 0 ≤ b) (h₄ : b ≤ 9) (h₅ : 0 ≤ c) (h₆ : c ≤ 9) :
  11 ∣ concatenate_reverse a b c :=
by
  sorry

end six_digit_number_divisible_by_eleven_l675_675262


namespace Mrs_Wake_speed_l675_675633

theorem Mrs_Wake_speed :
  ∃ (r : ℝ), r = 37.5 ∧
    (∃ d t, (d = 30 * (t + 1 / 30) ∧ d = 50 * (t - 1 / 30)) ∧ (t = 2 / 15) ∧ (r = d / t)) :=
begin
  -- The exact proof steps will be added here
  sorry
end

end Mrs_Wake_speed_l675_675633


namespace smallest_degree_polynomial_with_roots_l675_675290

theorem smallest_degree_polynomial_with_roots :
  ∀ (f : Polynomial ℚ),
  f.is_root (3 - 2 * Real.sqrt 2) →
  f.is_root (-3 - 2 * Real.sqrt 2) →
  f.is_root (1 + Real.sqrt 7) →
  f.is_root (1 - Real.sqrt 7) →
  ∃ (g : Polynomial ℚ), g.degree = 6 ∧
  (3 - 2 * Real.sqrt 2) ∈ g.roots ∧
  (-3 - 2 * Real.sqrt 2) ∈ g.roots ∧
  (1 + Real.sqrt 7) ∈ g.roots ∧
  (1 - Real.sqrt 7) ∈ g.roots
  :=
by sorry

end smallest_degree_polynomial_with_roots_l675_675290


namespace leading_arrangements_count_l675_675329

theorem leading_arrangements_count :
  let teachers : Finset String := {"A", "B", "C", "D", "E"}
  let groups : Finset (Finset String) := {{"A", "C"}, {"A", "D"}, {"A", "E"}, {"B", "C"}, {"B", "D"}, {"B", "E"}, {"C", "D"}, {"C", "E"}, {"D", "E"}}
  (count_leading_arrangements teachers groups) = 54 := sorry

end leading_arrangements_count_l675_675329


namespace height_of_trapezoid_l675_675825

theorem height_of_trapezoid (m : ℝ) (h_m : m > 0) : 
  ∃ x : ℝ, x = m * (1 - real.sqrt 2 / 2) ∧ (area_of_trapezoid_with_parallel_line_dividing_triangle m x = area_of_triangle m / 2) :=
begin
  -- Dummy example for the area_of_trapezoid_with_parallel_line_dividing_triangle function
  sorry
end

end height_of_trapezoid_l675_675825


namespace truck_speed_in_kmph_l675_675793

-- Define the given constants
def distance_meters : ℝ := 600
def time_seconds : ℝ := 10
def meters_per_kilometer : ℝ := 1000
def seconds_per_hour : ℝ := 3600

-- Define the theorem to be proved
theorem truck_speed_in_kmph : 
  let speed_mps := distance_meters / time_seconds in 
  let speed_kmph := speed_mps * (seconds_per_hour / meters_per_kilometer) 
  in speed_kmph = 216 := 
by 
  let speed_mps := distance_meters / time_seconds 
  let speed_kmph := speed_mps * (seconds_per_hour / meters_per_kilometer) 
  have : speed_kmph = 216 := sorry
  exact this

end truck_speed_in_kmph_l675_675793


namespace factor_polynomial_l675_675830

-- Define the polynomial expression
def polynomial (x : ℝ) : ℝ := 60 * x + 45 + 9 * x ^ 2

-- Define the factored form of the polynomial
def factored_form (x : ℝ) : ℝ := 3 * (3 * x + 5) * (x + 3)

-- The statement of the problem to prove equivalence of the forms
theorem factor_polynomial : ∀ x : ℝ, polynomial x = factored_form x :=
by
  -- The actual proof is omitted and replaced by sorry
  sorry

end factor_polynomial_l675_675830


namespace customer_initial_amount_l675_675002

theorem customer_initial_amount (d c : ℕ) (h1 : c = 100 * d) (h2 : c = 2 * d) : d = 0 ∧ c = 0 := by
  sorry

end customer_initial_amount_l675_675002


namespace perpendicular_lines_slope_l675_675158

theorem perpendicular_lines_slope (m : ℝ) : 
  ((m ≠ -3) ∧ (m ≠ -5) ∧ 
  (- (m + 3) / 4 * - (2 / (m + 5)) = -1)) ↔ m = -13 / 3 := 
sorry

end perpendicular_lines_slope_l675_675158


namespace chess_move_parity_l675_675332

theorem chess_move_parity (m n x : ℤ) (a b : ℤ) 
  (moves : ℕ → ℤ × ℤ)
  (hmoves : ∀ i, moves i = (a + k * m, b + l * n) ∨ 
                          (a - k * m, b + l * n) ∨ 
                          (a + k * m, b - l * n) ∨ 
                          (a - k * m, b - l * n) ∨ 
                          (a + k * n, b + l * m) ∨ 
                          (a - k * n, b + l * m) ∨ 
                          (a + k * n, b - l * m) ∨ 
                          (a - k * n, b - l * m))
  (hx: (∑ i in Finset.range x, moves i) = (a, b)) :
  even x :=
begin
  sorry
end

end chess_move_parity_l675_675332


namespace distinct_positive_integer_sums_eq_odd_integer_sums_l675_675959

theorem distinct_positive_integer_sums_eq_odd_integer_sums (n : ℕ) (h: n > 0) :
  (number of ways to express n as sum of distinct positive integers) =
  (number of ways to express n as sum of positive odd integers) :=
sorry

end distinct_positive_integer_sums_eq_odd_integer_sums_l675_675959


namespace sequence_T_mod_1500_l675_675604

theorem sequence_T_mod_1500 :
  let T := {n : ℕ | nat.popcount n = 9}.to_finset.sort (≤)
  (T \u 1500 - 1) % 1500 = 500 := by
  sorry

end sequence_T_mod_1500_l675_675604


namespace trigonometric_identity_l675_675056

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180))
  = (4 * Real.sin (10 * Real.pi / 180) / Real.sin (40 * Real.pi / 180)) :=
by sorry

end trigonometric_identity_l675_675056


namespace tetrahedron_surface_area_l675_675671

theorem tetrahedron_surface_area (m : ℝ) (hm : m > 0)
  (h_square_area : ∃ (s : ℝ), s * s = m^2) :
  let a := 2 * m in
  4 * ((a * a * (Real.sqrt 3)) / 4) = 4 * m^2 * (Real.sqrt 3) :=
by
  let a := 2 * m
  have h_triangle_area : (a * a * (Real.sqrt 3)) / 4 = a^2 * Real.sqrt 3 / 4,
  sorry
  calc
    4 * (a^2 * Real.sqrt 3 / 4)
        = a^2 * Real.sqrt 3 : by ring
    ... = (2 * m)^2 * Real.sqrt 3 : by rw [a]
    ... = 4 * m^2 * Real.sqrt 3 : by ring

end tetrahedron_surface_area_l675_675671


namespace cut_piece_ratio_l675_675591

noncomputable def original_log_length : ℕ := 20
noncomputable def weight_per_foot : ℕ := 150
noncomputable def cut_piece_weight : ℕ := 1500

theorem cut_piece_ratio :
  (cut_piece_weight / weight_per_foot / original_log_length) = (1 / 2) := by
  sorry

end cut_piece_ratio_l675_675591


namespace max_third_side_triangle_l675_675669

theorem max_third_side_triangle
  (P Q R : ℝ)
  (s1 s2 : ℝ)
  (h1 : cos (3 * P) + cos (3 * Q) + cos (3 * R) = 1)
  (h2 : s1 = 12)
  (h3 : s2 = 15) :
  ∃ s3, s3 = 27 :=
by
  use 27
  sorry

end max_third_side_triangle_l675_675669


namespace solve_modulus_eq_five_l675_675283

open Complex

theorem solve_modulus_eq_five (z : ℂ) (hz : |z - 1| = 5) :
  ∃ x y : ℝ, z = x + y * Complex.I ∧
    (-4 ≤ x ∧ x ≤ 6) ∧
    (y = Real.sqrt (25 - (x - 1)^2) ∨ y = -Real.sqrt (25 - (x - 1)^2)) :=
by
  sorry

end solve_modulus_eq_five_l675_675283


namespace hyperbola_eccentricity_l675_675489

theorem hyperbola_eccentricity (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : ∃ x y : ℝ, y = (b/a) * x ∧ x = 2 ∧ y = sqrt 21) :
  let e := sqrt (1 + (b / a) ^ 2) in
  e = 5 / 2 :=
by
  sorry

end hyperbola_eccentricity_l675_675489


namespace even_function_phi_l675_675503

noncomputable def phi := (3 * Real.pi) / 2

theorem even_function_phi (phi_val : Real) (hphi : 0 ≤ phi_val ∧ phi_val ≤ 2 * Real.pi) :
  (∀ x, Real.sin ((x + phi) / 3) = Real.sin ((-x + phi) / 3)) ↔ phi_val = phi := by
  sorry

end even_function_phi_l675_675503


namespace range_of_m_l675_675183

theorem range_of_m (f : ℝ → ℝ) (h_mono : ∀ x y : ℝ, x < y → f(x) > f(y)) (m : ℝ) (h_cond : f (2 * m) > f (1 + m)) : m < 1 :=
sorry

end range_of_m_l675_675183


namespace value_at_2012_l675_675436

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f(x + T) = f(x)

theorem value_at_2012 (f : ℝ → ℝ) 
  (h_odd : odd_function f)
  (h_periodic : periodic_function f 4) : 
  f 2012 = 0 :=
sorry

end value_at_2012_l675_675436


namespace decreasing_function_ratios_l675_675207

noncomputable def f : ℝ → ℝ := sorry

theorem decreasing_function_ratios 
  (h₁: ∀ x > 0, ∀ y > 0, x ≠ y → (y * f x - x * f y) / (x - y) < 0) :
  let a := f (2^0.2) / 2^0.2 in
  let b := f (0.2^2) / 0.2^2 in
  let c := f (Real.log2 5) / Real.log2 5 in
  c < a ∧ a < b :=
by
  sorry

end decreasing_function_ratios_l675_675207


namespace sum_a_b_eq_negative_one_l675_675540

theorem sum_a_b_eq_negative_one 
  (a b : ℝ) 
  (h1 : ∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - a * x - b < 0)
  (h2 : ∀ x : ℝ, x^2 - a * x - b = 0 → x = 2 ∨ x = 3) :
  a + b = -1 := 
sorry

end sum_a_b_eq_negative_one_l675_675540


namespace part1_part2_l675_675935

-- Definitions for the curve C and line l parametric equations
def curve_C (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ, 2 * Real.sin θ)
def line_l (t : ℝ) : ℝ × ℝ := (t + Real.sqrt 3, 2 * t - 2 * Real.sqrt 3)

-- Problem 1: Prove that the distance |AB| is 40/17
noncomputable def distance_AB : ℝ := 
  let A := (x1, y1) in
  let B := (x2, y2) in
  let x1 := sorry in -- x-coordinate of A from intersection of curve C and line l
  let y1 := sorry in -- y-coordinate of A from intersection of curve C and line l
  let x2 := sorry in -- x-coordinate of B from intersection of curve C and line l
  let y2 := sorry in -- y-coordinate of B from intersection of curve C and line l
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem part1 : distance_AB = (40 / 17) := 
  sorry -- We would fill this in with a proof in Lean

-- Problem 2: Prove that the dot product of FA and FB is 44
def left_focus_F : ℝ × ℝ := (-2 * Real.sqrt 3, 0)  -- The left focus of the ellipse

def vector_FA (A : ℝ × ℝ) : ℝ × ℝ := (A.1 - left_focus_F.1, A.2 - left_focus_F.2)
def vector_FB (B : ℝ × ℝ) : ℝ × ℝ := (B.1 - left_focus_F.1, B.2 - left_focus_F.2)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

noncomputable def FA_dot_FB : ℝ :=
  let A := (x1, y1) in
  let B := (x2, y2) in
  dot_product (vector_FA A) (vector_FB B)

theorem part2 : FA_dot_FB = 44 := 
  sorry -- We would fill this in with a proof in Lean

end part1_part2_l675_675935


namespace necessary_sufficient_condition_l675_675084

theorem necessary_sufficient_condition (n : ℕ) (hn : n > 0) :
  (∃ x : ℚ, x^n + (2 + x)^n + (2 - x)^n = 0) ↔ n = 1 :=
sorry

end necessary_sufficient_condition_l675_675084


namespace students_at_start_of_year_l675_675569

theorem students_at_start_of_year (S : ℝ) (h1 : S + 46.0 = 56) : S = 10 :=
sorry

end students_at_start_of_year_l675_675569


namespace brian_total_commission_l675_675419

theorem brian_total_commission :
  let commission_rate := 0.02
  let house1 := 157000
  let house2 := 499000
  let house3 := 125000
  let total_sales := house1 + house2 + house3
  let total_commission := total_sales * commission_rate
  total_commission = 15620 := by
{
  sorry
}

end brian_total_commission_l675_675419


namespace sequence_inequality_l675_675318

noncomputable def a : ℕ → ℕ
| 0       := 1
| (n + 1) := 2 * a (n / 2) + 3 * a (n / 3) + 6 * a (n / 6)

theorem sequence_inequality (n : ℕ) : a n ≤ 10 * n^2 + 1 := 
by 
  sorry

end sequence_inequality_l675_675318


namespace percent_of_a_is_b_l675_675529

theorem percent_of_a_is_b (a b c : ℝ) (h1 : c = 0.30 * a) (h2 : c = 0.25 * b) : b = 1.2 * a :=
by
  -- proof 
  sorry

end percent_of_a_is_b_l675_675529


namespace similar_triangles_proportionalities_l675_675460

-- Definitions of the conditions as hypotheses
variables (A B C D E F : Type)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
variables (triangle_ABC_sim_triangle_DEF : SimilarTriangles A B C D E F)
variables (AB_DE_ratio : AB / DE = 1 / 2)
variables (BC_length : BC = 2)

-- Defining the hypothesis of similarity
def SimilarTriangles (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] : Prop :=
  ∀ (AB BC CA DE EF FD : ℝ), (AB / DE = BC / EF) ∧ (BC / EF = CA / FD) ∧ (CA / FD = AB / DE)

-- The proof statement
theorem similar_triangles_proportionalities (A B C D E F : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (triangle_ABC_sim_triangle_DEF : SimilarTriangles A B C D E F)
  (AB_DE_ratio : AB / DE = 1 / 2)
  (BC_length : BC = 2) : 
  EF = 4 := 
by sorry

end similar_triangles_proportionalities_l675_675460


namespace max_value_l675_675178

-- Definitions for the given conditions
def point_A := (3, 1)
def line_equation (m n : ℝ) := 3 * m + n + 1 = 0
def positive_product (m n : ℝ) := m * n > 0

-- The main statement to be proved
theorem max_value (m n : ℝ) (h1 : line_equation m n) (h2 : positive_product m n) : 
  (3 / m + 1 / n) ≤ -16 :=
sorry

end max_value_l675_675178


namespace total_surface_area_of_tower_l675_675652

def volume_to_side_length (v : ℕ) : ℕ :=
  nat.cbrt v

def surface_area (s : ℕ) : ℕ :=
  6 * s^2

def adjusted_surface_area (s : ℕ) : ℕ :=
  if s > 1 then surface_area s - s^2 else surface_area s

theorem total_surface_area_of_tower :
  let side_lengths := [7, 6, 5, 4, 3, 2, 1].map volume_to_side_length in
  let surface_areas := side_lengths.map adjusted_surface_area in
  surface_areas.sum = 701 :=
by
  sorry

end total_surface_area_of_tower_l675_675652


namespace length_CD_l675_675581

-- Geometrical setup
variables (A B C D : Point)
variables (AD BC : Segment)
variables (E : Point)
variables [Trapezoid Angle90]

-- Conditions
axiom trapezoid_ABCD : is_trapezoid A B C D
axiom AD_parallel_BC : is_parallel AD BC
axiom length_AD : length AD = 2
axiom length_BC : length BC = 1
axiom angle_ABD : angle A B D = 90

-- Proof statement
theorem length_CD :
  length CD = 1 :=
sorry

end length_CD_l675_675581


namespace triangle_side_length_l675_675543

theorem triangle_side_length (A B C : Type) [euclidean_geometry A B C]
  (angle_B : B = 120 * π / 180)
  (AC : ℝ) (AB : ℝ) 
  (h₁ : AC = sqrt 19)
  (h₂ : AB = 2) :
  ∃ BC, BC = 3 :=
begin
  -- proof to be filled in
  sorry
end

end triangle_side_length_l675_675543


namespace brian_stones_l675_675043

variable (W B : ℕ)
variable (total_stones : ℕ := 100)
variable (G : ℕ := 40)
variable (Gr : ℕ := 60)

theorem brian_stones :
  (W > B) →
  ((W + B = total_stones) ∧ (G + Gr = total_stones) ∧ (W = 60)) :=
by
  sorry

end brian_stones_l675_675043


namespace swimming_speed_in_still_water_l675_675012

variable (v : ℝ) -- the person's swimming speed in still water

-- Conditions
variable (water_speed : ℝ := 4) -- speed of the water
variable (time : ℝ := 2) -- time taken to swim 12 km against the current
variable (distance : ℝ := 12) -- distance swam against the current

theorem swimming_speed_in_still_water :
  (v - water_speed) = distance / time → v = 10 :=
by
  sorry

end swimming_speed_in_still_water_l675_675012


namespace liams_annual_income_l675_675320

def income_tax (I : ℝ) (q : ℝ) : ℝ :=
  if I ≤ 30000 then
    q * 0.01 * I
  else if I ≤ 50000 then
    q * 0.01 * 30000 + (q + 3) * 0.01 * (I - 30000)
  else
    q * 0.01 * 30000 + (q + 3) * 0.01 * 20000 + (q + 5) * 0.01 * (I - 50000)

theorem liams_annual_income (I q : ℝ) 
  (h1 : income_tax I q = (q + 0.35) * 0.01 * I)
  (h2 : q = 10) :
  I = 54000 :=
  sorry

end liams_annual_income_l675_675320


namespace eccentricity_range_l675_675869

-- Define a structure for an ellipse with given properties
structure Ellipse (a b : ℝ) := 
  (a_pos : a > 0)
  (b_pos : b > 0)
  (a_gt_b : a > b)

-- Define the eccentricity of the ellipse
def eccentricity (a b : ℝ) [Ellipse a b] : ℝ := (Real.sqrt (a^2 - b^2)) / a

-- Proof statement
theorem eccentricity_range {a b : ℝ} [h : Ellipse a b] (P : ℝ × ℝ)
  (hP : (P.1^2 / a^2 + P.2^2 / b^2 = 1)) 
  (hObtuse : let c := Real.sqrt (a^2 - b^2) in (P.1 + c) * (c - P.1) + P.2^2 < 0) : 
  (Real.sqrt 2 / 2) < eccentricity a b ∧ eccentricity a b < 1 :=
by
  sorry

end eccentricity_range_l675_675869


namespace inequality_sum_d_l675_675226

open Nat

noncomputable def a : ℕ → ℕ
| 0       => 0 -- not defined for zero, but included for completeness
| (n + 1) => n + 1

noncomputable def b : ℕ → ℕ
| 0       => 0 -- not defined for zero, but included for completeness
| (n + 1) => 2^(n + 1)

noncomputable def d (n : ℕ) : ℤ :=
a n * b n

lemma arithmetic_sequence_condition {n : ℕ} (h : n > 0) :
  ∀ m ≤ n, a (m + 1) = m + 1 :=
begin
  intros,
  exact rfl,
end

lemma geometric_sequence_condition {n : ℕ} (h : n > 0) :
  ∀ m ≤ n, b (m + 1) = 2^(m + 1) :=
begin
  intros,
  exact rfl,
end

theorem inequality_sum_d (n : ℕ) (h : n > 1) :
  ∑ i in range n, (1 : ℝ) / ((d (i + 2)) - (d (i + 1))) < 1/4 :=
begin
  sorry,
end

end inequality_sum_d_l675_675226


namespace perimeter_square_d_l675_675992

def perimeter_square (side: ℝ) : ℝ := 4 * side
def area_square (side: ℝ) : ℝ := side * side

def side_length (perimeter: ℝ) : ℝ := perimeter / 4

noncomputable def side_square_d (area_c : ℝ) : ℝ := Real.sqrt (area_c / 2)

theorem perimeter_square_d (pc: ℝ) (h₁ : pc = 32) (h₂ : area_square (side_length pc) / 2 = area_square (side_square_d (area_square (side_length pc)))) :
  perimeter_square (side_square_d (area_square (side_length pc))) = 16 * Real.sqrt 2 :=
by
  sorry

end perimeter_square_d_l675_675992


namespace problem_statement_l675_675309

noncomputable def solve_problem (x : ℝ) : ℝ :=
  if h: (log 2 (log 4 x) + log 4 (log 8 x) + log 8 (log 2 x) = 1) then 
    log 4 (log 2 x) + log 8 (log 4 x) + log 2 (log 8 x)
  else
    0 -- This else clause is just a fallback

theorem problem_statement (x : ℝ) 
  (h : log 2 (log 4 x) + log 4 (log 8 x) + log 8 (log 2 x) = 1) : 
  abs ((log 4 (log 2 x) + log 8 (log 4 x) + log 2 (log 8 x)) - 0.87) < 0.01 :=
sorry

end problem_statement_l675_675309


namespace evaluate_expression_l675_675828

theorem evaluate_expression : 64^(1/2 : ℝ) * 125^(1/3 : ℝ) * 16^(1/4 : ℝ) = 40 := by
  sorry

end evaluate_expression_l675_675828


namespace virginia_avg_rainfall_l675_675705

theorem virginia_avg_rainfall:
  let march := 3.79
  let april := 4.5
  let may := 3.95
  let june := 3.09
  let july := 4.67
  let total_rainfall := march + april + may + june + july
  let avg_rainfall := total_rainfall / 5
  avg_rainfall = 4 := by sorry

end virginia_avg_rainfall_l675_675705


namespace find_f_7_5_l675_675613

noncomputable def f (x : ℝ) : ℝ := -- Given conditions
  if 0 ≤ x ∧ x ≤ 1 then x
  else if x ≥ 2 then -f (x - 2)
  else -f (-x)

theorem find_f_7_5 : f 7.5 = -0.5 := by
  -- The proof goes here
  sorry

end find_f_7_5_l675_675613


namespace uphill_integers_divisible_by_15_count_l675_675077

def is_uphill (n : ℕ) : Prop := 
  let digits := (nat.digits 10 n).reverse
  digits = list.sort digits

def ends_in_5 (n : ℕ) : Prop :=
  n % 10 = 5

def is_divisible_by_15 (n : ℕ) : Prop := 
  n % 15 = 0

def valid_digit (d : ℕ) : Prop := 
  d ∈ {1, 2, 3, 4, 6}.to_finset

noncomputable def count_valid_uphill_integers : ℕ :=
  (finset.range 10000).filter (λ n, is_uphill n ∧ ends_in_5 n ∧ is_divisible_by_15 n ∧ 
    (nat.digits 10 n).reverse.take (nat.digits 10 n).length.pred.forall valid_digit).card

theorem uphill_integers_divisible_by_15_count :
  count_valid_uphill_integers = 13 :=
sorry

end uphill_integers_divisible_by_15_count_l675_675077


namespace unit_digit_sum_l675_675881

theorem unit_digit_sum :
  ∃ P, 
  (let seq := (List.range' 1 1998) ++ [1999] ++ (List.reverse (List.range' 1 1998))) in 
  let S := seq.sum in 
  P = S % 10 
  ∧ P = 1 :=
by
  sorry

end unit_digit_sum_l675_675881


namespace next_palindromic_prime_after_131_l675_675308

theorem next_palindromic_prime_after_131 : ∃ p : ℕ, p > 131 ∧ (nat.is_palindrome p) ∧ nat.prime p ∧ ∀ q, (nat.is_palindrome q) ∧ nat.prime q ∧ q > 131 → q ≥ p := by
  sorry

end next_palindromic_prime_after_131_l675_675308


namespace main_theorem_l675_675865

open Real

def ellipse_standard_equation (c a : ℝ) : Prop := 
  c = sqrt 3 ∧ a = 2 ∧ b = sqrt (a^2 - c^2) ∧ (b = 1) ∧ (x y : ℝ) ∧ 
  (x^2 / a^2) + y^2 = 1

def chord_length_theorem (slope : ℝ) (a b : ℝ) : Prop :=
  slope = 1 / 2 ∧ a^2 = 4 ∧ b^2 = 1 ∧
  ∀ b : ℝ, (0 ≤ b^2 ∧ b^2 ≤ 2) ∧
  |AB| = sqrt (10 - 5 * b^2) ∧ b = 0 → y = (1 / 2) * x ∧ 
  max_chord_length = sqrt 10

theorem main_theorem : ellipse_standard_equation (sqrt 3) 2 ∧ 
                        chord_length_theorem (1 / 2) 2 (sqrt (2^2 - (sqrt 3)^2)) :=
begin
  sorry
end

end main_theorem_l675_675865


namespace coefficient_of_x3_in_expansion_l675_675087

open Finset

noncomputable def coefficient_x3_term_expansion (a b c n : ℕ) : ℤ :=
  let term := (a * x^2 + b * x + c) ^ n
  term.coeff 3

theorem coefficient_of_x3_in_expansion :
  coefficient_x3_term_expansion 1 (-1) 1 10 = -210 :=
by sorry

end coefficient_of_x3_in_expansion_l675_675087


namespace length_AB_l675_675256
-- Importing a broad library to ensure necessary tools are available

-- Define the necessary conditions and the proof statement
theorem length_AB  (parabola : Set (ℝ × ℝ)) (focus : ℝ × ℝ) (directrix : ℝ → Prop) (M : ℝ × ℝ)
  (h1 : parabola = {p | p.2 ^ 2 = 4 * p.1})
  (h2 : focus = (1, 0))
  (h3 : directrix = λ x, x = -1)
  (h4 : M.1 = 3)
  (h5 : ∀ A B ∈ parabola, (A.1 + B.1) / 2 = M.1) : 
  (∀ A B ∈ parabola |AB| = 8) :=
begin
  sorry -- Proof is skipped as per instructions.
end

end length_AB_l675_675256


namespace f_value_at_three_halves_l675_675969

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd (x : ℝ) : f(x) + f(-x) = 0
axiom f_periodic (x : ℝ) : f(x + 2) = f(x)
axiom f_condition (x : ℝ) (h : 0 < x ∧ x < 1) : f(x) = x / 2

theorem f_value_at_three_halves : f (3 / 2) = -1 / 4 :=
by
  sorry

end f_value_at_three_halves_l675_675969


namespace median_of_64_consecutive_integers_l675_675707

theorem median_of_64_consecutive_integers (sum_of_integers : ℕ) (h : sum_of_integers = 2^12) : 
  (2^12 / 64) = 64 :=
by
  rw h
  norm_num
  sorry

end median_of_64_consecutive_integers_l675_675707


namespace name_tag_perimeter_l675_675392

variable (x y z : ℕ)
variable (A B C D E : ℕ)

-- Define the side lengths of the pentagon in terms of x, y, and z.
def sideA := 3 * x
def sideB := 5 * y
def sideC := 6 * z
def sideD := 4 * x
def sideE := 7 * y

-- Define the specific values for x, y, and z
def myX := 1
def myY := 2
def myZ := 3

-- Define the perimeter
def perimeter := sideA + sideB + sideC + sideD + sideE

-- Define the proof problem
theorem name_tag_perimeter :
  (perimeter by sorry) = 49 := 
  by
  let A := 3 * myX
  let B := 5 * myY
  let C := 6 * myZ
  let D := 4 * myX
  let E := 7 * myY
  sorry -- Proof not required

end name_tag_perimeter_l675_675392


namespace climb_stairs_l675_675051

-- Define f(n) as a function with given base cases and recursive relationship
def f : ℕ → ℕ
| 1 := 1
| 2 := 2
| (n+3) := f (n+2) + f (n+1) -- Here n+3 is used to offset 0-based indexing

-- The goal is to prove the equivalence
theorem climb_stairs (n : ℕ) : f n = 
  if n = 1 ∨ n = 2 then n else f (n-1) + f (n-2) :=
sorry


end climb_stairs_l675_675051


namespace lines_concurrent_l675_675617

-- Definitions of the points and conditions
variables {A B C H A1 B1 C1 : Point}
variable (triangle_ABC : Triangle A B C)
variable [is_acute_angled_triangle : IsAcuteAngledTriangle triangle_ABC]
variable [orthocenter : Orthocenter triangle_ABC H]
variable (circumcenter_BCH : Circumcenter (Triangle B C H) A1)
variable (circumcenter_CAH : Circumcenter (Triangle C A H) B1)
variable (circumcenter_ABH : Circumcenter (Triangle A B H) C1)

-- Lean 4 statement to prove the concurrency of lines AA1, BB1, and CC1
theorem lines_concurrent : Concurrent (Line A A1) (Line B B1) (Line C C1) :=
sorry

end lines_concurrent_l675_675617


namespace ab_minus_cd_value_l675_675289

-- Conditions definitions
variables {a b c d : ℝ}
variables (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d ≥ 0)
variables (h5 : a^2 + d^2 = 1) (h6 : b^2 + c^2 = 1) (h7 : ac + bd = 1/3)

-- Theorem statement
theorem ab_minus_cd_value : (ab - cd) = 2 * sqrt 2 / 3 :=
sorry

end ab_minus_cd_value_l675_675289


namespace number_of_neighborhoods_l675_675402

def street_lights_per_side : ℕ := 250
def roads_per_neighborhood : ℕ := 4
def total_street_lights : ℕ := 20000

theorem number_of_neighborhoods : 
  (total_street_lights / (2 * street_lights_per_side * roads_per_neighborhood)) = 10 :=
by
  -- proof to show that the number of neighborhoods is 10
  sorry

end number_of_neighborhoods_l675_675402


namespace conjugate_in_interval_l675_675643

noncomputable def quadratic_irrational (A D B : ℤ) : ℚ :=
  (A + Real.sqrt D) / B

noncomputable def conjugate_quadratic_irrational (A D B : ℤ) : ℚ :=
  (A - Real.sqrt D) / B

def is_periodic_continued_fraction (α : ℚ) : Prop :=
  -- This is a simplified version of the purely periodic continued fraction condition.
  ∃ a₀ a₁ ... aₙ : ℤ, α = GeneralizedContinuedFraction.of -- Using an appropriate definition or placeholder.

theorem conjugate_in_interval
  (A D B : ℤ)
  (hD_irrational : ¬ ∃ n : ℤ, n^2 = D) -- D is not a perfect square
  (hB_nonzero : B ≠ 0)
  (h_alpha_periodic : is_periodic_continued_fraction (quadratic_irrational A D B)) :
  -1 < conjugate_quadratic_irrational A D B ∧ conjugate_quadratic_irrational A D B < 0 := 
sorry -- Proof not required.

end conjugate_in_interval_l675_675643


namespace shape_of_triangle_formed_by_Z_A_B_is_isosceles_l675_675626

noncomputable def conjugate (z : ℂ) : ℂ := complex.conj z

def f (z : ℂ) : ℝ := abs ((z + 1) * (conjugate z - complex.i))

theorem shape_of_triangle_formed_by_Z_A_B_is_isosceles
    (z : ℂ) (A B : ℂ) (h1 : |z| = 1)
    (hA : A = -1 + 0 * complex.i)
    (hB : B = 0 - 1 * complex.i) :
    let Z := z in
    ∃ θ : ℝ, z = real.cos θ + complex.i * real.sin θ ∧
    (f z) = 2 + real.sqrt 2 ∧
    dist Z A = dist Z B := 
  sorry

end shape_of_triangle_formed_by_Z_A_B_is_isosceles_l675_675626


namespace hyperbola_equation_and_eccentricity_l675_675490

theorem hyperbola_equation_and_eccentricity :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = 5 ∧ 
    (∃ P : ℝ × ℝ, P = (sqrt 5, 4) ∧ 
    (0, 2) = ((-sqrt 5 + P.1) / 2, P.2 / 2) ∧ 
    ((5 / a^2) - (16 / b^2) = 1))) →
    (∃ e, (x^2 - (y^2 / 4) = 1) ∧ e = sqrt 5 := begin sorry end

end hyperbola_equation_and_eccentricity_l675_675490


namespace probability_of_40_cents_l675_675996

noncomputable def num_successful_outcomes : ℕ := 16 + 3

def total_outcomes : ℕ := 2 ^ 5

def probability_success : ℚ := num_successful_outcomes / total_outcomes

theorem probability_of_40_cents : probability_success = 19 / 32 := by
  unfold probability_success num_successful_outcomes total_outcomes
  norm_num
  sorry

end probability_of_40_cents_l675_675996


namespace jack_initial_money_l675_675209

-- Define the cost of one pair of socks
def cost_pair_socks : ℝ := 9.50

-- Define the cost of soccer shoes
def cost_soccer_shoes : ℝ := 92

-- Define the additional money Jack needs
def additional_money_needed : ℝ := 71

-- Define the total cost of two pairs of socks and one pair of soccer shoes
def total_cost : ℝ := 2 * cost_pair_socks + cost_soccer_shoes

-- Theorem to prove Jack's initial money
theorem jack_initial_money : ∃ m : ℝ, total_cost - additional_money_needed = 40 :=
by
  sorry

end jack_initial_money_l675_675209


namespace carol_initial_peanuts_l675_675429

variable initial_peanuts : ℕ
variable father_given : ℕ := 5
variable total_peanuts : ℕ := 7

theorem carol_initial_peanuts :
  initial_peanuts + father_given = total_peanuts → initial_peanuts = 2 :=
by
  sorry

end carol_initial_peanuts_l675_675429


namespace find_m_l675_675621

def numFactorsOf2 (k : ℕ) : ℕ :=
  k / 2 + k / 4 + k / 8 + k / 16 + k / 32 + k / 64 + k / 128 + k / 256

theorem find_m : ∃ m : ℕ, m > 1990 ^ 1990 ∧ m = 3 ^ 1990 + numFactorsOf2 m :=
by
  sorry

end find_m_l675_675621


namespace mashed_potatoes_suggestion_l675_675282

theorem mashed_potatoes_suggestion :
  ∃ M B, B = 42 ∧ M = B + 366 ∧ M = 408 :=
by
  use 408
  use 42
  split
  . rw nat.eq_refl
  split
  . rw nat.eq_refl
  . rw [nat.refl (42 + 366)]
  .
sorry

end mashed_potatoes_suggestion_l675_675282


namespace sin_alpha_correct_l675_675125

-- Definitions of the conditions
variable (α : Real) (hα : 0 < α ∧ α < π / 2)
variable (htan : ∃ x, (4 * x^2 + x - 3 = 0) ∧ (tan α = x))

-- Statement to be proven
theorem sin_alpha_correct : sin α = 3 / 5 :=
sorry

end sin_alpha_correct_l675_675125


namespace possible_values_of_n_l675_675523

noncomputable def quadratic_has_distinct_real_roots (n : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ (x^2 + n * x + 9 = 0) ∧ (y^2 + n * y + 9 = 0)

theorem possible_values_of_n (n : ℝ) :
  quadratic_has_distinct_real_roots n ↔ n ∈ (set.Iio (-6) ∪ set.Ioi 6) :=
sorry

end possible_values_of_n_l675_675523


namespace alex_buys_17_1_pounds_of_corn_l675_675076

-- Definitions based on conditions
def corn_cost_per_pound : ℝ := 1.20
def bean_cost_per_pound : ℝ := 0.50
def total_pounds : ℝ := 30
def total_cost : ℝ := 27.00

-- Define the variables
variables (c b : ℝ)

-- Theorem statement to prove the number of pounds of corn Alex buys
theorem alex_buys_17_1_pounds_of_corn (h1 : b + c = total_pounds) (h2 : bean_cost_per_pound * b + corn_cost_per_pound * c = total_cost) :
  c = 17.1 :=
sorry

end alex_buys_17_1_pounds_of_corn_l675_675076


namespace problem_statement_l675_675493

open Real

noncomputable def f (x : ℝ) : ℝ := 10^x

theorem problem_statement : f (log 2) * f (log 5) = 10 :=
by {
  -- Note: Proof is omitted as indicated in the procedure.
  sorry
}

end problem_statement_l675_675493


namespace max_red_squares_l675_675377

-- Define the conditions of the problem
def cube_dimensions : ℕ := 3
def total_faces : ℕ := 6
def squares_per_face : ℕ := cube_dimensions * cube_dimensions
def total_small_squares : ℕ := total_faces * squares_per_face

-- Define the colors
inductive Color
| red
| yellow
| blue

-- Define a function considering no two adjacent squares can have the same color
noncomputable def is_valid_coloring (coloring : ℕ → ℕ → ℕ → Color) : Prop :=
  ∀ x y z, (∀ dx dy dz, (dx = 0 ∧ dy = 0 ∧ dz ≠ 0) ∨ (dx = 0 ∧ dy ≠ 0 ∧ dz = 0) ∨ (dx ≠ 0 ∧ dy = 0 ∧ dz = 0) →
    x + dx < cube_dimensions ∧ y + dy < cube_dimensions ∧ z + dz < cube_dimensions →
    coloring x y z ≠ coloring (x + dx) (y + dy) (z + dz))

-- Define and prove the maximal number of red squares
theorem max_red_squares (coloring : ℕ → ℕ → ℕ → Color) (h : is_valid_coloring coloring) : 
  ∑ x in Finset.range cube_dimensions, ∑ y in Finset.range cube_dimensions, ∑ z in Finset.range cube_dimensions, 
  if coloring x y z = Color.red then 1 else 0 = 22 :=
sorry

end max_red_squares_l675_675377


namespace max_distance_C1_to_C2_l675_675497

def curve_C1_parametric (α : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos α, sin α)

def curve_C2_general (x y : ℝ) : Prop :=
  x + y = 8

theorem max_distance_C1_to_C2 :
  ∃ (P : ℝ × ℝ), (P = (√3 * cos (-π/6), sin (-π/6))) ∧ 
  (∀ (Q : ℝ × ℝ), (Q.1 = 6 - (sqrt 2 / 2) * Q.2) ∧ (Q.2 = 2 + (sqrt 2 / 2) * Q.1) →
  ∃ (d : ℝ), d = 5 * sqrt 2) ∧ 
  P = (-3/2, -1/2) :=
sorry

end max_distance_C1_to_C2_l675_675497


namespace bowls_initially_bought_l675_675784

theorem bowls_initially_bought 
  (x : ℕ) 
  (cost_per_bowl : ℕ := 13) 
  (revenue_per_bowl : ℕ := 17)
  (sold_bowls : ℕ := 108)
  (profit_percentage : ℝ := 23.88663967611336) 
  (approx_x : ℝ := 139) :
  (23.88663967611336 / 100) * (cost_per_bowl : ℝ) * (x : ℝ) = 
    (sold_bowls * revenue_per_bowl) - (sold_bowls * cost_per_bowl) → 
  abs ((x : ℝ) - approx_x) < 0.5 :=
by
  sorry

end bowls_initially_bought_l675_675784


namespace slope_of_line_eq_neg_four_thirds_l675_675346

variable {x y : ℝ}
variable (p₁ p₂ : ℝ × ℝ) (h₁ : 3 / p₁.1 + 4 / p₁.2 = 0) (h₂ : 3 / p₂.1 + 4 / p₂.2 = 0)

theorem slope_of_line_eq_neg_four_thirds 
  (hneq : p₁.1 ≠ p₂.1):
  (p₂.2 - p₁.2) / (p₂.1 - p₁.1) = -4 / 3 := 
sorry

end slope_of_line_eq_neg_four_thirds_l675_675346


namespace family_reunion_attendance_l675_675032

-- Define the conditions
def male_adults : ℕ := 100
def female_adults : ℕ := male_adults + 50
def total_adults : ℕ := male_adults + female_adults
def children : ℕ := 2 * total_adults

-- State the theorem to be proven
theorem family_reunion_attendance : 
  let total_people := total_adults + children in
  total_people = 750 :=
by 
  sorry

end family_reunion_attendance_l675_675032


namespace triangle_properties_l675_675585

open Real

variables (A B C a b c : ℝ) (triangle_obtuse triangle_right triangle_acute : Prop)

-- Declaration of properties 
def sin_gt (A B : ℝ) := sin A > sin B
def tan_product_lt (A C : ℝ) := tan A * tan C < 1
def cos_squared_eq (A B C : ℝ) := cos A ^ 2 + cos B ^ 2 - cos C ^ 2 = 1

theorem triangle_properties :
  (sin_gt A B → A > B) ∧
  (triangle_obtuse → tan_product_lt A C) ∧
  (cos_squared_eq A B C → triangle_right) :=
  by sorry

end triangle_properties_l675_675585


namespace sequence_properties_l675_675864

theorem sequence_properties (a : ℕ → ℕ) 
  (h : ∀ n, a 1 + ∑ i in finRange (n-1), 2^(i+1) * a (i+2) = n * 2^(n+1)) :
  (a 1 = 4) ∧
  (∑ i in finRange 10, a (i+1) ≠ 150) ∧
  (∑ i in finRange 11, (-1)^(i+1) * a (i+1) = -14) ∧
  (∑ i in finRange 16, Nat.abs (a (i+1) - 10) = 168) := by
  sorry

end sequence_properties_l675_675864


namespace hyperbola_eccentricity_l675_675197

variables {a b p : ℝ}

def hyperbola (x y : ℝ) (a b : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def parabola (x y : ℝ) (p : ℝ) : Prop :=
  (p > 0) ∧ (x^2 = 2 * p * y)

def e_correct (a b : ℝ) : Prop :=
  let e := (sqrt (a^2 + b^2)) / a in e = 3 / 2

theorem hyperbola_eccentricity :
  ∀ (a b p : ℝ),
  ∀ (O A B : ℝ × ℝ),
  hyperbola O.1 O.2 a b ∧ parabola A.1 A.2 p ∧ parabola B.1 B.2 p ∧ 
  (interaction and orthocenter conditions for O, A, B and the focus of C2 being p/2)
  ⭢ e_correct a b :=
sorry

end hyperbola_eccentricity_l675_675197


namespace other_number_l675_675709

theorem other_number (x : ℕ) (h : 27 + x = 62) : x = 35 :=
by
  sorry

end other_number_l675_675709


namespace parallel_tanx_and_fraction_perp_sinx_minus_cosx_l675_675162

variables (x : ℝ)

-- Definition for vector a and vector b
def vector_a := (1, cos x)
def vector_b := (1 / 3, sin x)

-- Condition that vector a is parallel to vector b
axiom parallel_condition (x : ℝ) : x ∈ Ioo 0 real.pi → vector_a x == 1 / (1 / 3) * vector_b x

-- Prove the values for tan x and (sin x + cos x) / (sin x - cos x)
theorem parallel_tanx_and_fraction (hx : x ∈ Ioo 0 real.pi) (hpar : parallel_condition x) :
  tan x = 1 / 3 ∧ (sin x + cos x) / (sin x - cos x) = -2 := by
  sorry

-- Definition for the dot product
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Condition that vector a is perpendicular to vector b
axiom perp_condition (x : ℝ) : x ∈ Ioo 0 real.pi → dot_product (vector_a x) (vector_b x) = 0

-- Prove the value for sin x - cos x
theorem perp_sinx_minus_cosx (hx : x ∈ Ioo 0 real.pi) (hperp : perp_condition x) :
  sin x - cos x = sqrt 15 / 3 := by
  sorry

end parallel_tanx_and_fraction_perp_sinx_minus_cosx_l675_675162


namespace probability_at_least_40_cents_heads_l675_675997

noncomputable def value_of_heads (p n d q h : Bool) : Real :=
  (if p then 0.01 else 0) + (if n then 0.05 else 0) + (if d then 0.10 else 0) + (if q then 0.25 else 0) + (if h then 0.50 else 0)

theorem probability_at_least_40_cents_heads :
  let outcomes := {p : Bool, n : Bool, d : Bool, q : Bool, h : Bool}
  let favorable := (outcomes.filter $ λ (o : outcomes), value_of_heads o.p o.n o.d o.q o.h >= 0.40).size
  favorable / (outcomes.size : Real) = 19 / 32 :=
by
  sorry

end probability_at_least_40_cents_heads_l675_675997


namespace sequence_T_mod_1500_l675_675605

theorem sequence_T_mod_1500 :
  let T := {n : ℕ | nat.popcount n = 9}.to_finset.sort (≤)
  (T \u 1500 - 1) % 1500 = 500 := by
  sorry

end sequence_T_mod_1500_l675_675605


namespace find_a_values_l675_675612

variables (a : ℝ) (P : ℝ → ℝ)
noncomputable def P : ℝ → ℝ := λ x, x^2 - 2 * a * x - a^2 - 3 / 4

theorem find_a_values (a : ℝ) :
  (∀ x ∈ (set.Icc 0 1), |P x| ≤ 1) ↔ (a ∈ set.Icc (-1 / 2 : ℝ) (sqrt 2 / 4 : ℝ)) :=
sorry

end find_a_values_l675_675612


namespace problem1_problem2_l675_675884

noncomputable def f (x : ℝ) : ℝ :=
  |x - 2| - |2 * x + 1|

theorem problem1 (x : ℝ) :
  f x ≤ 2 ↔ x ≤ -1 ∨ -1/3 ≤ x :=
sorry

theorem problem2 (a : ℝ) (b : ℝ) :
  (∀ x, |a + b| - |a - b| ≥ f x) → (a ≥ 5 / 4 ∨ a ≤ -5 / 4) :=
sorry

end problem1_problem2_l675_675884


namespace min_value_eccentricities_l675_675144

-- Definitions for conditions
variables (a1 b1 a2 b2 c : ℝ)
variables (F1 F2 P : ℝ × ℝ)
variables (e1 e2 : ℝ)

-- Condition specifications
def is_ellipse (a1 b1 : ℝ) (P : ℝ × ℝ) : Prop := 
  (P.1^2 / a1^2) + (P.2^2 / b1^2) = 1

def is_hyperbola (a2 b2 : ℝ) (P : ℝ × ℝ) : Prop := 
  (P.1^2 / a2^2) - (P.2^2 / b2^2) = 1

def same_foci (F1 F2 : ℝ × ℝ) (a1 a2 c : ℝ) : Prop :=
  sqrt(F1.1^2 + F1.2^2) = sqrt(F2.1^2 + F2.2^2) ∧ a1^2 + a2^2 = 2 * c^2

def perpendicular (P F1 F2 : ℝ × ℝ) : Prop := 
  (P.1 - F1.1)^2 + (P.2 - F1.2)^2 + (P.1 - F2.1)^2 + (P.2 - F2.2)^2 = 4 * c^2

def eccentricities (a1 a2 a b : ℝ) (e1 e2 c : ℝ) : Prop :=
  e1 = c / a1 ∧ e2 = c / a2

-- Lean statement for the problem
theorem min_value_eccentricities
  (h1 : is_ellipse a1 b1 P)
  (h2 : is_hyperbola a2 b2 P)
  (h3 : same_foci F1 F2 a1 a2 c)
  (h4 : perpendicular P F1 F2)
  (h5 : eccentricities a1 a2 a b e1 e2 c):
  4 * e1^2 + e2^2 = 9 / 2 := 
sorry

end min_value_eccentricities_l675_675144


namespace apple_price_relationship_l675_675315

theorem apple_price_relationship (x : ℕ) (y : ℕ) :
  (x = 1 → y = 8) ∧ 
  (x = 2 → y = 16) ∧ 
  (x = 3 → y = 24) ∧ 
  (x = 4 → y = 32) →
  y = 8 * x :=
by
  simpl
  intros
  sorry

end apple_price_relationship_l675_675315


namespace regression_decrease_l675_675862

theorem regression_decrease (x : ℝ) : 
  let y := 2 - 2.5 * x in
  let y_new := 2 - 2.5 * (x + 1) in
  y_new - y = -2.5 :=
by
  let y := 2 - 2.5 * x
  let y_new := 2 - 2.5 * (x + 1)
  calc
    y_new - y = (2 - 2.5 * (x + 1)) - (2 - 2.5 * x) : by rfl
           ... = -2.5 : by sorry

end regression_decrease_l675_675862


namespace difference_of_numbers_l675_675312

theorem difference_of_numbers (x y : ℝ) (h1 : x * y = 23) (h2 : x + y = 24) : |x - y| = 22 :=
sorry

end difference_of_numbers_l675_675312


namespace veranda_width_l675_675302

-- Defining the conditions as given in the problem
def room_length : ℝ := 21
def room_width : ℝ := 12
def veranda_area : ℝ := 148

-- The main statement to prove
theorem veranda_width :
  ∃ (w : ℝ), (21 + 2*w) * (12 + 2*w) - 21 * 12 = 148 ∧ w = 2 :=
by
  sorry

end veranda_width_l675_675302


namespace monitors_in_lunchroom_l675_675685

-- Definitions based on conditions:
def percentGirls := 0.40
def percentBoys := 0.60
def milkConsumedByGirls (G : ℕ) := 2 * G
def milkConsumedByBoys (B : ℕ) := 1 * B
def totalMilkConsumption := 168
def monitorToStudentRatio := 2 / 15

-- Main statement to prove:
theorem monitors_in_lunchroom (S G B M : ℕ) (h1 : G = percentGirls * S) (h2 : B = percentBoys * S)
  (h3 : milkConsumedByGirls G + milkConsumedByBoys B = totalMilkConsumption)
  (h4 : M / S = monitorToStudentRatio) : M = 16 :=
sorry

end monitors_in_lunchroom_l675_675685


namespace coefficient_of_x_in_binomial_expansion_l675_675105

theorem coefficient_of_x_in_binomial_expansion :
  let a := ∫ x in 1..(Real.exp 2), 1 / x
  (x : ℝ) =>
  (1 - 0) * a = 2 * a :=
  let a := ∫ x in 1..(Real.exp 2), 1 / x
  (0 : ℝ) =>
  a := 2
  (c : ℝ) => 
  c = - 80 := \int_{1}^{e^{2}} \frac{1}{x} dx :=
begin
  let a := ∫ x in 1 .. e ^ 2, 1 / x,
  have h : a = 2, sorry,
  have x : real := (x^{2} - \frac{a}{x})^{5}
  have coeff : -80, sorry,
end

end coefficient_of_x_in_binomial_expansion_l675_675105


namespace power_division_l675_675730

theorem power_division : 3^16 / 81^2 = 6561 := by
  have h1 : 81 = 3^4 := rfl
  have h2 : 81^2 = (3^4)^2 := by rw [h1]
  have h3 : 81^2 = 3^8 := by rw [pow_mul]
  rw [h2, h3]
  have h4 : 3^16 / 3^8 = 3^(16 - 8) := div_pow_same_base
  rw [h4]
  have h5 : 3^8 = 6561 := by
    -- You can use a calculator or computer for this step if you want to verify
    sorry
  rwa h5

end power_division_l675_675730


namespace find_elements_in_1_to_40_not_possible_27_elements_in_1_to_40_l675_675448

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n
  
def no_square_products (s : set ℕ) : Prop :=
  ∀ a b ∈ s, a ≠ b → ¬ is_perfect_square (a * b)

theorem find_elements_in_1_to_40 :
  ∃ s : set ℕ, s ⊆ { i | 1 ≤ i ∧ i ≤ 40 } ∧ no_square_products s ∧ s.card = 26 :=
sorry

theorem not_possible_27_elements_in_1_to_40 :
  ¬ ∃ s : set ℕ, s ⊆ { i | 1 ≤ i ∧ i ≤ 40 } ∧ no_square_products s ∧ s.card = 27 :=
sorry

end find_elements_in_1_to_40_not_possible_27_elements_in_1_to_40_l675_675448


namespace nail_pierces_one_not_other_l675_675650

open Set

-- Let R be the rectangular box
variable (R : Type) [rectangular_box R]

-- Define C1 and C2 as identical pieces of cardboard covering the bottom of R
variables (C1 C2 : R) (h_identical : C1 = C2)

-- Define the center point p of the bottom of the box R
variable (p : R)

-- Define the condition that C1 and C2 are overlapping and cover the bottom of R
def covers_bottom (C1 C2 : R) : Prop := 
  ∀ x, (x ∈ C1 ∪ C2) → (x ∈ R)

-- The hypothesized statement: there exists a configuration such that the nail pierces only one cardboard
theorem nail_pierces_one_not_other (R : Type) [rectangular_box R] (C1 C2 : R) 
  (h_identical : C1 = C2) (p : R) (h_cover : covers_bottom C1 C2) : 
  ∃ x ∈ C1, x ∉ C2 :=
sorry

end nail_pierces_one_not_other_l675_675650


namespace hyperbola_eq_from_conditions_l675_675005

-- Conditions of the problem
def hyperbola_center : Prop := ∃ (h : ℝ → ℝ → Prop), h 0 0
def hyperbola_eccentricity : Prop := ∃ e : ℝ, e = 2
def parabola_focus : Prop := ∃ p : ℝ × ℝ, p = (4, 0)
def parabola_equation : Prop := ∀ x y : ℝ, y^2 = 8 * x

-- Hyperbola equation to be proved
def hyperbola_equation : Prop := ∀ x y : ℝ, (x^2 / 4) - (y^2 / 12) = 1

-- Lean 4 theorem statement
theorem hyperbola_eq_from_conditions 
  (h_center : hyperbola_center) 
  (h_eccentricity : hyperbola_eccentricity) 
  (p_focus : parabola_focus) 
  (p_eq : parabola_equation) 
  : hyperbola_equation :=
by
  sorry

end hyperbola_eq_from_conditions_l675_675005


namespace magnitude_of_a_l675_675128

variable (a e : ℝ^3)
variable (norm_e : Real) (angle_ae : Real) (proj_ae : Real)

theorem magnitude_of_a 
  (h1 : norm_e = 1)
  (h2 : angle_ae = 120 * Real.pi / 180)
  (h3 : proj_ae = -2)
  (h4 : |e| = norm_e)
  (h5 : Real.cos angle_ae = -1/2)
  (h6 : (a.dot e) = proj_ae) : 
  |a| = 4 := by
  sorry

end magnitude_of_a_l675_675128


namespace sin_half_angle_product_lt_quarter_l675_675025

theorem sin_half_angle_product_lt_quarter (A B C : ℝ) (h : A + B + C = 180) :
    Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) < 1 / 4 := 
    sorry

end sin_half_angle_product_lt_quarter_l675_675025


namespace tower_total_surface_area_l675_675655

/-- Given seven cubes with volumes 1, 8, 27, 64, 125, 216, and 343 cubic units each, stacked vertically
    with volumes decreasing from bottom to top, compute their total surface area including the bottom. -/
theorem tower_total_surface_area :
  let volumes := [1, 8, 27, 64, 125, 216, 343]
  let side_lengths := volumes.map (fun v => v ^ (1 / 3))
  let surface_area (n : ℝ) (visible_faces : ℕ) := visible_faces * (n ^ 2)
  let total_surface_area := surface_area 7 5 + surface_area 6 4 + surface_area 5 4 + surface_area 4 4
                            + surface_area 3 4 + surface_area 2 4 + surface_area 1 5
  total_surface_area = 610 := sorry

end tower_total_surface_area_l675_675655


namespace units_digit_base8_l675_675047

theorem units_digit_base8 (a b n : ℕ) : 
  a = 256 → b = 72 → n = 8 → (a * b) % n = 0 :=
by
  intros ha hb hn
  rw [ha, hb, hn]
  have h : 256 * 72 = 18432 := rfl
  rw h
  exact Nat.mod_refl 8

end units_digit_base8_l675_675047


namespace tom_fruit_bowl_l675_675336

def initial_lemons (oranges lemons removed remaining : ℕ) : ℕ :=
  lemons

theorem tom_fruit_bowl (oranges removed remaining : ℕ) (L : ℕ) 
  (h_oranges : oranges = 3)
  (h_removed : removed = 3)
  (h_remaining : remaining = 6)
  (h_initial : oranges + L - removed = remaining) : 
  initial_lemons oranges L removed remaining = 6 :=
by
  -- Implement the proof here
  sorry

end tom_fruit_bowl_l675_675336


namespace alice_change_sum_l675_675798

theorem alice_change_sum :
  let amounts := {x : ℕ | x < 100 ∧ (x % 5 = 4 ∨ x % 10 = 6)}
  (∀ x ∈ amounts, x % 5 = 4 → x ∈ amounts) ∧
  (∀ x ∈ amounts, x % 10 = 6 → x ∈ amounts) →
  (amounts.sum id = 486) :=
by
  let amounts := {x : ℕ | x < 100 ∧ (x % 5 = 4 ∨ x % 10 = 6)}
  have h1 : ∀ x ∈ amounts, x % 5 = 4 → x ∈ amounts := sorry
  have h2 : ∀ x ∈ amounts, x % 10 = 6 → x ∈ amounts := sorry
  have sum_eq : (amounts.sum id = 486) := sorry
  exact ⟨h1, h2, sum_eq⟩

end alice_change_sum_l675_675798


namespace fg_of_3_is_83_l675_675176

def g (x : ℕ) : ℕ := x ^ 3
def f (x : ℕ) : ℕ := 3 * x + 2
theorem fg_of_3_is_83 : f (g 3) = 83 := by
  sorry

end fg_of_3_is_83_l675_675176


namespace sum_of_sides_le_twice_third_side_l675_675580

theorem sum_of_sides_le_twice_third_side 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : A + C = 2 * B) 
  (h2 : A + B + C = 180)
  (h3 : a / (Real.sin A) = b / (Real.sin B))
  (h4 : a / (Real.sin A) = c / (Real.sin C))
  (h5 : b / (Real.sin B) = c / (Real.sin C)) : 
  a + c ≤ 2 * b := 
by 
  sorry

end sum_of_sides_le_twice_third_side_l675_675580


namespace find_integer_less_than_neg3_l675_675034

theorem find_integer_less_than_neg3 (a b c d : Int) (h : a = -4 ∧ b = -2 ∧ c = 0 ∧ d = 3) :
  a < -3 ∧ ¬ (b < -3) ∧ ¬ (c < -3) ∧ ¬ (d < -3) :=
by
  cases h with
  | intro ha hb hc hd =>
  split
  · rewrite [ha]; exact Int.lt_neg3_of_negsucc
  split
  · intro; contradiction
  split
  · intro; contradiction
  · intro; contradiction

end find_integer_less_than_neg3_l675_675034


namespace remaining_cookie_percentage_l675_675640

theorem remaining_cookie_percentage : 
  let A := 0.70 in
  let B := (1 - A) / 3 in
  1 - (A + B) = 0.20 :=
by
  intros
  sorry

end remaining_cookie_percentage_l675_675640


namespace cos_sin_15_degree_identity_l675_675445

theorem cos_sin_15_degree_identity :
  cos 15 * cos 15 * cos 15 * cos 15 - sin 15 * sin 15 * sin 15 * sin 15 = (sqrt 3) / 2 := by
  sorry

end cos_sin_15_degree_identity_l675_675445


namespace find_x_l675_675498

-- Define the variables and key constants
variables (x y z a b c k : ℝ)
variables (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : k ≠ 0)

-- Define the given equations
def eq1 := (xy) / (x + y) = a
def eq2 := (xz) / (x + z) = b
def eq3 := (yz) / (y + z) = c
def eq4 := (xyz) / (x + y + z) = k

-- State the theorem to be proved
theorem find_x (h1 : eq1) (h2 : eq2) (h3 : eq3) (h4 : eq4) : 
  x = 2 * k * a * b / (a * b + b * c - a * c) :=
sorry

end find_x_l675_675498


namespace find_C_and_D_l675_675831

noncomputable def C : ℚ := 15 / 8
noncomputable def D : ℚ := 17 / 8

theorem find_C_and_D (x : ℚ) (h₁ : x ≠ 9) (h₂ : x ≠ -7) :
  (4 * x - 6) / ((x - 9) * (x + 7)) = C / (x - 9) + D / (x + 7) :=
by sorry

end find_C_and_D_l675_675831


namespace length_of_faster_train_583_33_l675_675726

open Real

noncomputable def length_of_faster_train (speed_A : ℝ) (speed_B : ℝ) (time_sec : ℝ) : ℝ :=
  let relative_speed := (speed_A + speed_B) * (5 / 18)
  in relative_speed * time_sec

theorem length_of_faster_train_583_33 :
  length_of_faster_train 60 80 15 = 583.33 :=
by
  sorry

end length_of_faster_train_583_33_l675_675726


namespace train_passing_time_l675_675004

variables (length_highspeed : ℝ) (length_regular : ℝ) (time_highspeed_observation : ℝ) (time_regular_observation : ℝ)

theorem train_passing_time (h1 : length_highspeed = 400)
                            (h2 : length_regular = 600)
                            (h3 : time_highspeed_observation = 3)
                            (h4 : time_regular_observation = 2) :
    (length_regular / time_highspeed_observation) * time_regular_observation = length_highspeed :=
by {
    rw [h1, h2, h3, h4],
    norm_num,
    sorry
}

end train_passing_time_l675_675004


namespace initial_books_count_l675_675712

theorem initial_books_count (books_taken books_left : ℕ) (h_taken : books_taken = 10) (h_left : books_left = 28) : ∃ N : ℕ, N = books_taken + books_left ∧ N = 38 :=
by
  use (books_taken + books_left)
  split
  {
    exact rfl
  }
  {
    rw [h_taken, h_left]
    exact rfl
  }

end initial_books_count_l675_675712


namespace n_prime_or_power_of_2_l675_675364

theorem n_prime_or_power_of_2 (n : ℕ) (a : ℕ → ℕ) (k : ℕ) 
  (h_n : n > 6) 
  (h_coprime : ∀ i, i < k → (a i).gcd n = 1) 
  (h_pos : ∀ i, i < k - 1 → a (i + 1) - a i > 0) : 
  nat.prime n ∨ ∃ m, n = 2 ^ m :=
sorry

end n_prime_or_power_of_2_l675_675364


namespace coins_fit_in_new_box_l675_675800

-- Definitions
def diameters_bound (d : ℕ) : Prop :=
  d ≤ 10

def box_fits (length width : ℕ) (fits : Prop) : Prop :=
  fits

-- Conditions
axiom coins_diameter_bound : ∀ (d : ℕ), diameters_bound d
axiom original_box_fits : box_fits 30 70 True

-- Proof statement
theorem coins_fit_in_new_box : box_fits 40 60 True :=
sorry

end coins_fit_in_new_box_l675_675800


namespace math_competition_question_1_math_competition_question_2_l675_675670

noncomputable def participant_score_probabilities : Prop :=
  let P1 := (3 / 5)^2 * (2 / 5)^2
  let P2 := 2 * (3 / 5) * (2 / 5)
  let P3 := 2 * (3 / 5) * (2 / 5)^2
  let P4 := (3 / 5)^2
  P1 + P2 + P3 + P4 = 208 / 625

noncomputable def winning_probabilities : Prop :=
  let P_100_or_more := (4 / 5)^8 * (3 / 5)^3 + 3 * (4 / 5)^8 * (3 / 5)^2 * (2 / 5) + 
                      (8 * (4 / 5)^7 * (1/5) * (3 / 5)^3 + 
                      28 * (4 / 5)^6 * (1/5)^2 * (3 / 5)^3)
  let winning_if_100_or_more := P_100_or_more * (9 / 10)
  let winning_if_less_100 := (1 - P_100_or_more) * (2 / 5)
  winning_if_100_or_more + winning_if_less_100 ≥ 1 / 2

theorem math_competition_question_1 : participant_score_probabilities :=
by sorry

theorem math_competition_question_2 : winning_probabilities :=
by sorry

end math_competition_question_1_math_competition_question_2_l675_675670


namespace Ramesh_investment_l675_675293

theorem Ramesh_investment (Suresh_investment : ℕ) (total_profit : ℕ) (Ramesh_profit_share : ℕ) : ℕ :=
  let Suresh_profit_share := total_profit - Ramesh_profit_share
  let gcd_val := 1125
  let profit_ratio_Suresh := Suresh_profit_share / gcd_val
  let profit_ratio_Ramesh := Ramesh_profit_share / gcd_val
  let ratio := profit_ratio_Suresh.to_real / profit_ratio_Ramesh.to_real
  let R := Suresh_investment / ratio
  if Suresh_investment = 24000 ∧ total_profit = 19000 ∧ Ramesh_profit_share = 11875 then
    R.to_nat = 42000 else sorry

-- Applying the theorem with given conditions
example : Ramesh_investment 24000 19000 11875 = 42000 := by
  apply rfl
  sorry

end Ramesh_investment_l675_675293


namespace trig_identity_l675_675062

theorem trig_identity :
  ∀ (θ : ℝ),
    θ = 70 * (π / 180) →
    (1 / Real.cos θ - (Real.sqrt 3) / Real.sin θ) = Real.sec (20 * (π / 180)) ^ 2 :=
by
  sorry

end trig_identity_l675_675062


namespace seq_general_term_l675_675484

-- Define the sequence according to the given conditions
def seq (a : ℕ+ → ℚ) : Prop := 
  a 1 = 1 ∧ ∀ n : ℕ+, a (n + 1) = n / (n + 1 : ℕ) * a n

-- The theorem statement: proving the general term
theorem seq_general_term (a : ℕ+ → ℚ) (h : seq a) : ∀ n : ℕ+, a n = 1 / n :=
by {
  sorry
}

end seq_general_term_l675_675484


namespace probability_at_most_one_red_light_l675_675562

def probability_of_no_red_light (p : ℚ) (n : ℕ) : ℚ := (1 - p) ^ n

def probability_of_exactly_one_red_light (p : ℚ) (n : ℕ) : ℚ :=
  (n.choose 1) * p ^ 1 * (1 - p) ^ (n - 1)

theorem probability_at_most_one_red_light (p : ℚ) (n : ℕ) (h : p = 1/3 ∧ n = 4) :
  probability_of_no_red_light p n + probability_of_exactly_one_red_light p n = 16 / 27 :=
by
  rw [h.1, h.2]
  sorry

end probability_at_most_one_red_light_l675_675562


namespace find_a_l675_675872

theorem find_a (x y a : ℤ) (h1 : x = 1) (h2 : y = -1) (h3 : 2 * x - a * y = 3) : a = 1 := by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end find_a_l675_675872


namespace determine_x_value_l675_675222

theorem determine_x_value
  (a d x : ℕ) (h1 : s a = x + 2)
  (h2 : s (a + d) = x^2 + 18)
  (h3 : s (a + 2 * d) = 2 * x^2 + 18) :
  x = 4 := by
  sorry

noncomputable def s (n : ℕ) : ℕ := n -- Placeholder for the sequence definition

end determine_x_value_l675_675222


namespace simplify_expression_l675_675731

theorem simplify_expression : 
  (1 / (64^(1/3))^9) * 8^6 = 1 := by 
  have h1 : 64 = 2^6 := by rfl
  have h2 : 8 = 2^3 := by rfl
  sorry

end simplify_expression_l675_675731


namespace lines_intersect_l675_675783

-- Define the parameterization of the first line
def line1 (t : ℝ) : ℝ × ℝ :=
  (1 + 2 * t, 1 - 3 * t)

-- Define the parameterization of the second line
def line2 (u : ℝ) : ℝ × ℝ :=
  (-1 + 5 * u, 2 - 2 * u)

-- Define the expected intersection point
def expected_intersection : ℝ × ℝ :=
  (9 / 11, 14 / 11)

-- Statement to prove that the lines intersect at the expected point
theorem lines_intersect :
  ∃ t u, line1 t = expected_intersection ∧ line2 u = expected_intersection :=
sorry

end lines_intersect_l675_675783


namespace geometric_progression_l675_675834

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

theorem geometric_progression (x : ℝ) (h1 : x > 0) 
                              (h2 : x - Real.floor x > 0) (h3 : x - Real.floor x < 1) 
                              (h4 : ∃ r : ℝ, (x - Real.floor x) * r = Real.floor x ∧ 
                                              (x - Real.floor x) * r^2 = x) :
  x = golden_ratio :=
  sorry

end geometric_progression_l675_675834


namespace initial_average_weight_l675_675711

theorem initial_average_weight
  (A : ℚ) -- Define A as a rational number since we are dealing with division 
  (h1 : 6 * A + 133 = 7 * 151) : -- Condition from the problem translated into an equation
  A = 154 := -- Statement we need to prove
by
  sorry -- Placeholder for the proof

end initial_average_weight_l675_675711


namespace maximum_marks_each_subject_l675_675791

theorem maximum_marks_each_subject:
  (∃ M : ℝ, 
    let total_obtained := 220 + 180 + 135 + 245 + 175 - 50 in
    let required_marks := total_obtained + 50 in
    let total_maximum := 5 * M in
    required_marks = 0.60 * total_maximum ∧
    M = 319) :=
begin
  sorry
end

end maximum_marks_each_subject_l675_675791


namespace largest_possible_b_l675_675357

theorem largest_possible_b 
  (a b : ℤ) 
  (h1 : 29 < a ∧ a < 41) 
  (h2 : b > 39) 
  (h3 : ∃ (A B : ℤ), (29 < A ∧ A < 41 ∧ 39 < B ∧ (A ≤ a ∧ a ≤ A)) ∧ (A / B - B / B = 0.4))
  : b = 75 := 
sorry

end largest_possible_b_l675_675357


namespace johns_out_of_pocket_expense_l675_675214

-- Define the conditions given in the problem
def old_system_cost : ℤ := 250
def old_system_trade_in_value : ℤ := (80 * old_system_cost) / 100
def new_system_initial_cost : ℤ := 600
def new_system_discount : ℤ := (25 * new_system_initial_cost) / 100
def new_system_final_cost : ℤ := new_system_initial_cost - new_system_discount

-- Define the amount of money that came out of John's pocket
def out_of_pocket_expense : ℤ := new_system_final_cost - old_system_trade_in_value

-- State the theorem that needs to be proven
theorem johns_out_of_pocket_expense : out_of_pocket_expense = 250 := by
  sorry

end johns_out_of_pocket_expense_l675_675214


namespace burger_calories_proof_l675_675216

def burrito_count := 10
def burrito_price := 6
def burrito_calories := 120
def burger_count := 5
def burger_price := 8
def additional_calories_per_dollar := 50

theorem burger_calories_proof :
  let 
    total_burrito_calories := burrito_count * burrito_calories,
    burrito_calories_per_dollar := total_burrito_calories / burrito_price,
    burger_calories_per_dollar := burrito_calories_per_dollar + additional_calories_per_dollar,
    total_burger_calories := burger_calories_per_dollar * burger_price,
    calories_per_burger := total_burger_calories / burger_count 
  in 
    calories_per_burger = 400 :=
by 
  sorry

end burger_calories_proof_l675_675216


namespace find_BC_l675_675266

-- Definition of the given triangle with provided sides and angle
structure Triangle :=
  (A B C : ℝ)  -- representing the sides of the triangle
  (angleA : ℝ) -- representing the angle ∠A

noncomputable def cos_60 : ℝ := real.cos (real.pi / 3)

lemma BC_squared {t : Triangle} (h1 : t.A = 3) (h2 : t.B = 4) (h3 : t.angleA = 60) : t.C^2 = t.A^2 + t.B^2 - 2 * t.A * t.B * cos_60 :=
by
  have angle_in_rad : t.angleA = real.pi / 3 := by norm_num [h3]
  rw [h1, h2, angle_in_rad]
  simp [cos_60]
  norm_num

noncomputable def BC {t : Triangle} (h1 : t.A = 3) (h2 : t.B = 4) (h3 : t.angleA = 60) : ℝ := real.sqrt (t.C^2)

theorem find_BC (t : Triangle) (h1 : t.A = 3) (h2 : t.B = 4) (h3 : t.angleA = 60) : BC t h1 h2 h3 = real.sqrt 13 :=
by
  unfold BC
  rw [BC_squared t h1 h2 h3]
  norm_num
  sorry

end find_BC_l675_675266


namespace journey_total_distance_l675_675167

theorem journey_total_distance :
  let speed1 := 40 -- in kmph
  let time1 := 3 -- in hours
  let speed2 := 60 -- in kmph
  let totalTime := 5 -- in hours
  let distance1 := speed1 * time1
  let time2 := totalTime - time1
  let distance2 := speed2 * time2
  let totalDistance := distance1 + distance2
  totalDistance = 240 := 
by
  sorry

end journey_total_distance_l675_675167


namespace remainder_of_M_l675_675606

def T : ℕ → ℕ := λ n, sorry -- (Function to generate the increasing sequence whose binary representation has exactly 9 ones)

def M : ℕ := T 1500

theorem remainder_of_M (hM: M = 33023) : M % 1500 = 23 :=
by {
  have : M = 33023 := hM,
  rw this,
  norm_num,
}

end remainder_of_M_l675_675606


namespace similar_triangles_proportionalities_l675_675461

-- Definitions of the conditions as hypotheses
variables (A B C D E F : Type)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
variables (triangle_ABC_sim_triangle_DEF : SimilarTriangles A B C D E F)
variables (AB_DE_ratio : AB / DE = 1 / 2)
variables (BC_length : BC = 2)

-- Defining the hypothesis of similarity
def SimilarTriangles (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] : Prop :=
  ∀ (AB BC CA DE EF FD : ℝ), (AB / DE = BC / EF) ∧ (BC / EF = CA / FD) ∧ (CA / FD = AB / DE)

-- The proof statement
theorem similar_triangles_proportionalities (A B C D E F : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (triangle_ABC_sim_triangle_DEF : SimilarTriangles A B C D E F)
  (AB_DE_ratio : AB / DE = 1 / 2)
  (BC_length : BC = 2) : 
  EF = 4 := 
by sorry

end similar_triangles_proportionalities_l675_675461


namespace degree_of_expression_l675_675499

open Polynomial

noncomputable def expr1 : Polynomial ℤ := (monomial 5 3 - monomial 3 2 + 4) * (monomial 12 2 - monomial 8 1 + monomial 6 5 - 15)
noncomputable def expr2 : Polynomial ℤ := (monomial 3 2 - 4) ^ 6
noncomputable def final_expr : Polynomial ℤ := expr1 - expr2

theorem degree_of_expression : degree final_expr = 18 := by
  sorry

end degree_of_expression_l675_675499


namespace general_term_formula_sum_inequality_l675_675111

variable {a_n : ℕ → ℚ} {S_n T_n : ℕ → ℚ}

-- Given conditions
def is_arithmetic_sequence (a_n : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n + d

def sum_first_n_terms (a_n : ℕ → ℚ) (S_n : ℕ → ℚ) : Prop :=
  ∀ n, S_n n = ∑ i in finset.range n, a_n i

def problem_conditions : Prop :=
  is_arithmetic_sequence a_n 2 ∧ a_n 2 = 4 ∧ sum_first_n_terms a_n S_n ∧ S_n 5 = 30

-- Statements to prove
theorem general_term_formula (h : problem_conditions) :
  ∀ n, a_n n = 2 * n :=
sorry

theorem sum_inequality (h : problem_conditions) :
  ∀ n, 1 / 8 ≤ T_n n ∧ T_n n < 1 / 4 :=
sorry

variable [problems_conditions]

def T_n (n : ℕ) : ℚ :=
  (1 / 4) * (1 - 1 / (n + 1))

example : general_term_formula problem_conditions := sorry
example : sum_inequality problem_conditions := sorry

end general_term_formula_sum_inequality_l675_675111


namespace triangle_angle_measure_l675_675584

theorem triangle_angle_measure
  (D E F : ℝ)
  (hD : D = 70)
  (hE : E = 2 * F + 18)
  (h_sum : D + E + F = 180) :
  F = 92 / 3 :=
by
  sorry

end triangle_angle_measure_l675_675584


namespace sin_double_angle_l675_675127

open Real 

theorem sin_double_angle (α β : ℝ) 
  (h1 : π / 2 < β) 
  (h2 : β < α) 
  (h3 : α < 3 * π / 4) 
  (h4 : cos (α - β) = 12 / 13) 
  (h5 : sin (α + β) = -3 / 5) : 
  sin (2 * α) = -56 / 65 := 
by 
  sorry

end sin_double_angle_l675_675127


namespace max_sales_l675_675009

noncomputable def f (t : ℕ) : ℕ := 30 - t

noncomputable def g (t : ℕ) : ℕ :=
  if t ≤ 10 then 2 * t + 40 else 15

noncomputable def S (t : ℕ) : ℕ :=
  f t * g t

theorem max_sales : (argmax S {1 ≤ t ∧ t ≤ 20}) = 5 ∧ S 5 = 1250 :=
by {
  -- The proof would go here
  sorry
}

end max_sales_l675_675009


namespace find_a_l675_675491

-- Definitions for the hyperbola and its eccentricity
def hyperbola_eq (a : ℝ) : Prop := a > 0 ∧ ∃ b : ℝ, b^2 = 3 ∧ ∃ e : ℝ, e = 2 ∧ 
  e = Real.sqrt (1 + b^2 / a^2)

-- The main theorem stating the value of 'a' given the conditions
theorem find_a (a : ℝ) (h : hyperbola_eq a) : a = 1 := 
by {
  sorry
}

end find_a_l675_675491


namespace unique_integer_sum_squares_l675_675833

theorem unique_integer_sum_squares (n : ℤ) (h : ∃ d1 d2 d3 d4 : ℕ, d1 * d2 * d3 * d4 = n ∧ n = d1*d1 + d2*d2 + d3*d3 + d4*d4) : n = 42 := 
sorry

end unique_integer_sum_squares_l675_675833


namespace volunteers_to_venues_l675_675311

theorem volunteers_to_venues :
  let n := 5 -- Number of volunteers
  let k := 3 -- Number of venues
  -- Number of ways to assign 5 volunteers to 3 venues such that each venue has at least one volunteer
  (3^5 - 3 * 2^5 = 147) := 
begin
  let n := 5,
  let k := 3,
  -- By inclusion-exclusion principle
  have h1 : 3^5 = 243 := by norm_num,
  have h2 : 3 * 2^5 = 96 := by norm_num,
  calc 
    3^5 - 3 * 2^5 : 243 - 96 := by sorry,
    243 - 96 = 147 := by norm_num,
end

end volunteers_to_venues_l675_675311


namespace meaningful_expression_l675_675335

theorem meaningful_expression (m : ℝ) : 
  (∃ (r : ℝ), r = (∑ (x : ℝ), (x > m) → (x = (m+1) ^ (1 / 3) / (m-1))) ↔ 
  (m ≠ 1) ) :=
by sorry

end meaningful_expression_l675_675335


namespace inequality_proof_l675_675852

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c) ≥ 1 / (a + b) + 1 / (b + c) + 1 / (c + a) :=
by
  sorry

end inequality_proof_l675_675852


namespace probability_at_least_40_cents_heads_l675_675999

noncomputable def value_of_heads (p n d q h : Bool) : Real :=
  (if p then 0.01 else 0) + (if n then 0.05 else 0) + (if d then 0.10 else 0) + (if q then 0.25 else 0) + (if h then 0.50 else 0)

theorem probability_at_least_40_cents_heads :
  let outcomes := {p : Bool, n : Bool, d : Bool, q : Bool, h : Bool}
  let favorable := (outcomes.filter $ λ (o : outcomes), value_of_heads o.p o.n o.d o.q o.h >= 0.40).size
  favorable / (outcomes.size : Real) = 19 / 32 :=
by
  sorry

end probability_at_least_40_cents_heads_l675_675999


namespace triangle_area_l675_675024

def line1 (x : ℝ) : ℝ := x + 4
def line2 (x : ℝ) : ℝ := -3 * x + 9
def line3 (x : ℝ) : ℝ := 2

theorem triangle_area :
  let P1 := (-2 : ℝ, 2)
  let P2 := (7 / 3 : ℝ, 2)
  let P3 := (5 / 4 : ℝ, 21 / 4)
  let base := abs (7 / 3 - (-2))
  let height := (21 / 4 - 2)
  let area := 1 / 2 * base * height
  area = 169 / 24 := 
by 
  sorry

end triangle_area_l675_675024


namespace rectangular_box_in_sphere_radius_l675_675394

theorem rectangular_box_in_sphere_radius (a b c s : ℝ) 
  (h1 : a + b + c = 40) 
  (h2 : 2 * a * b + 2 * b * c + 2 * a * c = 608) 
  (h3 : (2 * s)^2 = a^2 + b^2 + c^2) : 
  s = 16 * Real.sqrt 2 :=
by
  sorry

end rectangular_box_in_sphere_radius_l675_675394


namespace angle_at_3_15_l675_675078

-- Define the measurements and conditions
def hour_hand_position (hour min : ℕ) : ℝ := 
  30 * hour + 0.5 * min

def minute_hand_position (min : ℕ) : ℝ := 
  6 * min

def angle_between_hands (hour min : ℕ) : ℝ := 
  abs (minute_hand_position min - hour_hand_position hour min)

-- Theorem statement in Lean 4
theorem angle_at_3_15 : angle_between_hands 3 15 = 7.5 :=
by sorry

end angle_at_3_15_l675_675078


namespace union_A_B_inter_CR_A_B_range_of_a_l675_675121

open Set

section
variable {a : ℝ} {x : ℝ}

def A := { x | 3 ≤ x ∧ x < 7 }
def B := { x | x^2 - 12 * x + 20 < 0 }
def C := { x | x < a }
def CR_A := { x | x < 3 ∨ x ≥ 7 }

/- (1) -/
theorem union_A_B : A ∪ B = { x | 2 < x ∧ x < 10 } := sorry

theorem inter_CR_A_B : (CR_A ∩ B) = { x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) } := sorry

/- (2) -/
theorem range_of_a (h : A ⊆ C) : a ≥ 7 := sorry

end

end union_A_B_inter_CR_A_B_range_of_a_l675_675121


namespace tower_total_surface_area_l675_675654

/-- Given seven cubes with volumes 1, 8, 27, 64, 125, 216, and 343 cubic units each, stacked vertically
    with volumes decreasing from bottom to top, compute their total surface area including the bottom. -/
theorem tower_total_surface_area :
  let volumes := [1, 8, 27, 64, 125, 216, 343]
  let side_lengths := volumes.map (fun v => v ^ (1 / 3))
  let surface_area (n : ℝ) (visible_faces : ℕ) := visible_faces * (n ^ 2)
  let total_surface_area := surface_area 7 5 + surface_area 6 4 + surface_area 5 4 + surface_area 4 4
                            + surface_area 3 4 + surface_area 2 4 + surface_area 1 5
  total_surface_area = 610 := sorry

end tower_total_surface_area_l675_675654


namespace problem_l675_675410

open EuclideanGeometry

variables {A B C D P Q : Point} 

def conditions (ABC : Triangle) := 
  AC ≠ BC ∧ 
  ∃ D, D ∈ interior_triangle ABC ∧ 
  ∠ ADB = 90 + (1 / 2) * ∠ ACB ∧
  ∃ circumcircle_ABC : Circle, is_circumcircle circumcircle_ABC ABC ∧ 
  tangent_at circumcircle_ABC C intersects AB at P ∧
  ∃ circumcircle_ADC : Circle, is_circumcircle circumcircle_ADC (Triangle.mk A D C) ∧ 
  tangent_at circumcircle_ADC C intersects AD at Q

def to_prove (PQ : Line) : Prop :=
  bisects PQ (∠ BPC)

theorem problem (ABC : Triangle) (cond : conditions ABC) : 
  ∃ (PQ : Line), to_prove PQ :=
sorry

end problem_l675_675410


namespace shortest_distance_from_circle_to_line_l675_675319

open Real

def circle_center : ℝ × ℝ := (-2, 1)
def circle_radius : ℝ := 1
def line : ℝ × ℝ → Prop := λ p, p.2 = p.1 - 1

def point_to_line_distance (p : ℝ × ℝ) : ℝ :=
  abs ((-1) * p.1 + 1 * p.2 + 1) / sqrt (1 + 1)

def shortest_distance := point_to_line_distance circle_center - circle_radius

theorem shortest_distance_from_circle_to_line :
  shortest_distance = 2 * sqrt 2 - 1 :=
sorry

end shortest_distance_from_circle_to_line_l675_675319


namespace mixed_candy_price_l675_675400

noncomputable def price_per_pound (a b c : ℕ) (pa pb pc : ℝ) : ℝ :=
  (a * pa + b * pb + c * pc) / (a + b + c)

theorem mixed_candy_price :
  let a := 30
  let b := 15
  let c := 20
  let pa := 10.0
  let pb := 12.0
  let pc := 15.0
  price_per_pound a b c pa pb pc * 0.9 = 10.8 := by
  sorry

end mixed_candy_price_l675_675400


namespace find_two_digit_number_t_l675_675452

theorem find_two_digit_number_t (t : ℕ) (ht1 : 10 ≤ t) (ht2 : t ≤ 99) (ht3 : 13 * t % 100 = 52) : t = 12 := 
sorry

end find_two_digit_number_t_l675_675452


namespace correct_tile_for_b_l675_675080

structure Tile where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

def TileI : Tile := {top := 5, right := 3, bottom := 1, left := 6}
def TileII : Tile := {top := 2, right := 6, bottom := 3, left := 5}
def TileIII : Tile := {top := 6, right := 1, bottom := 4, left := 2}
def TileIV : Tile := {top := 4, right := 5, bottom := 2, left := 1}

def RectangleBTile := TileIII

theorem correct_tile_for_b : RectangleBTile = TileIII :=
  sorry

end correct_tile_for_b_l675_675080


namespace corresponding_angles_equal_l675_675272

-- Definition: Corresponding angles and their equality
def corresponding_angles (α β : ℝ) : Prop :=
  -- assuming definition of corresponding angles can be defined
  sorry

theorem corresponding_angles_equal {α β : ℝ} (h : corresponding_angles α β) : α = β :=
by
  -- the proof is provided in the problem statement
  sorry

end corresponding_angles_equal_l675_675272


namespace problem1_problem2_problem3_l675_675423

-- Problem 1
theorem problem1 (x : ℝ) : 
  (-3 * x^3)^2 - (4 * x^8) / (x^2) = 5 * x^6 :=
sorry

-- Problem 2
theorem problem2 (m n : ℝ) :
  (m - 2 * n) * (m + 2 * n) - (m - n)^2 = -5 * n^2 + 2 * m * n :=
sorry

-- Problem 3
theorem problem3 : 
  (| ((3 * 1) + (-2))^2 - (-2 * ((3 * 1) + (-2))) |) / (3 * 1) = 1 :=
sorry

end problem1_problem2_problem3_l675_675423


namespace inequalities_not_hold_l675_675098

theorem inequalities_not_hold (x y z a b c : ℝ) (h1 : x < a) (h2 : y < b) (h3 : z < c) : 
  ¬ (x * y + y * z + z * x < a * b + b * c + c * a) ∧ 
  ¬ (x^2 + y^2 + z^2 < a^2 + b^2 + c^2) ∧ 
  ¬ (x * y * z < a * b * c) := 
sorry

end inequalities_not_hold_l675_675098


namespace possible_values_of_n_l675_675522

noncomputable def quadratic_has_distinct_real_roots (n : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ (x^2 + n * x + 9 = 0) ∧ (y^2 + n * y + 9 = 0)

theorem possible_values_of_n (n : ℝ) :
  quadratic_has_distinct_real_roots n ↔ n ∈ (set.Iio (-6) ∪ set.Ioi 6) :=
sorry

end possible_values_of_n_l675_675522


namespace breakEvenBooks_l675_675398

theorem breakEvenBooks (FC VC_per_book SP : ℝ) (hFC : FC = 56430) (hVC : VC_per_book = 8.25) (hSP : SP = 21.75) :
  ∃ x : ℕ, FC + (VC_per_book * x) = SP * x ∧ x = 4180 :=
by {
  sorry
}

end breakEvenBooks_l675_675398


namespace equal_hexagon_areas_l675_675819

-- Definitions for Convex Hexagons and Parallelism
structure ConvexHexagon (P : Type) :=
(vertices : Fin 6 → P)

def parallel {P : Type} [AffineSpace ℝ P] (A B C D : P) : Prop :=
  ∃ u v : ℝ, ∀ k : ℝ, k • (A -ᵥ B) = u • (C -ᵥ D)

variables {P : Type} [AffineSpace ℝ P]

-- Conditions
variables (A B : ConvexHexagon P)
(h_interior : ∀ i : Fin 6, B.vertices i ∉ ⋂ j : Fin 6, closes_ball (A.vertices j) 0)
(h_parallel1 : parallel (A.vertices 0) (A.vertices 1) (B.vertices 0) (B.vertices 1))
(h_parallel2 : parallel (A.vertices 1) (A.vertices 2) (B.vertices 1) (B.vertices 2))
(h_parallel3 : parallel (A.vertices 2) (A.vertices 3) (B.vertices 2) (B.vertices 3))
(h_parallel4 : parallel (A.vertices 3) (A.vertices 4) (B.vertices 3) (B.vertices 4))
(h_parallel5 : parallel (A.vertices 4) (A.vertices 5) (B.vertices 4) (B.vertices 5))
(h_parallel6 : parallel (A.vertices 5) (A.vertices 0) (B.vertices 5) (B.vertices 0))

-- Question: Prove that the areas of hexagons formed by alternating vertices are equal
theorem equal_hexagon_areas :
  area ⟦⟨A.vertices 0, B.vertices 1, A.vertices 2, B.vertices 3, A.vertices 4, B.vertices 5⟩⟧ = area ⟦⟨B.vertices 0, A.vertices 1, B.vertices 2, A.vertices 3, B.vertices 4, A.vertices 5⟩⟧ :=
sorry

end equal_hexagon_areas_l675_675819


namespace constant_function_of_zero_derivative_l675_675916

theorem constant_function_of_zero_derivative
  {f : ℝ → ℝ}
  (h : ∀ x : ℝ, deriv f x = 0) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end constant_function_of_zero_derivative_l675_675916


namespace triangle_side_BC_l675_675548

theorem triangle_side_BC {A B C : Type} [inner_product_space ℝ A] 
  {b c BC: ℝ} 
  (hB : ∠ B = real.pi / 3) 
  (hAC : b = real.sqrt 19) 
  (hAB : c = 2) : 
  BC = 3 := 
sorry

end triangle_side_BC_l675_675548


namespace image_of_element_2_l675_675159

-- Define the mapping f and conditions
def f (x : ℕ) : ℕ := 2 * x + 1

-- Define the element and its image using f
def element_in_set_A : ℕ := 2
def image_in_set_B : ℕ := f element_in_set_A

-- The theorem to prove
theorem image_of_element_2 : image_in_set_B = 5 :=
by
  -- This is where the proof would go, but we omit it with sorry
  sorry

end image_of_element_2_l675_675159


namespace triangle_side_BC_l675_675547

theorem triangle_side_BC {A B C : Type} [inner_product_space ℝ A] 
  {b c BC: ℝ} 
  (hB : ∠ B = real.pi / 3) 
  (hAC : b = real.sqrt 19) 
  (hAB : c = 2) : 
  BC = 3 := 
sorry

end triangle_side_BC_l675_675547


namespace integral_half_circle_area_l675_675305

open Real

-- Define the integrand function
def integrand (a : ℝ) (x : ℝ) : ℝ := sqrt (a ^ 2 - x ^ 2)

noncomputable def integral_of_half_circle (a : ℝ) : ℝ :=
  ∫ x in -a..a, integrand a x

-- The theorem we want to prove
theorem integral_half_circle_area (a : ℝ) (ha : 0 < a) :
  integral_of_half_circle a = (1 / 2) * π * a ^ 2 := by
  sorry

end integral_half_circle_area_l675_675305


namespace maximize_profit_l675_675762

theorem maximize_profit (x : ℤ) (hx : 20 ≤ x ∧ x ≤ 30) :
  (∀ y, 20 ≤ y ∧ y ≤ 30 → ((y - 20) * (30 - y)) ≤ ((25 - 20) * (30 - 25))) := 
sorry

end maximize_profit_l675_675762


namespace f_is_even_l675_675371

section

variables (F f : ℝ → ℝ)
hypothesis (h1 : ∀ x, x ≠ 0 → F(x) = (x^3 - 2 * x) * f(x))
hypothesis (h2 : ∀ x, F(-x) = -F(x))
hypothesis (h3 : ¬ (∀ x, f(x) = 0))

theorem f_is_even : ∀ x, f(-x) = f(x) := 
by
  sorry

end

end f_is_even_l675_675371


namespace fg_of_3_is_83_l675_675175

def g (x : ℕ) : ℕ := x ^ 3
def f (x : ℕ) : ℕ := 3 * x + 2
theorem fg_of_3_is_83 : f (g 3) = 83 := by
  sorry

end fg_of_3_is_83_l675_675175


namespace problem_l675_675957

def g (x : ℕ) : ℕ := x^2 + 1
def f (x : ℕ) : ℕ := 3 * x - 2

theorem problem : f (g 3) = 28 := by
  sorry

end problem_l675_675957


namespace sum_of_possible_values_of_p_s_l675_675615

theorem sum_of_possible_values_of_p_s (p q r s : ℝ) 
  (h1 : |p - q| = 3) 
  (h2 : |q - r| = 4) 
  (h3 : |r - s| = 5) : 
  (finite (λ x, ∃ (s : ℝ), (|p - s| = x ∧ |q - p| = 3 ∧ |r - q| = 4 ∧ |s - r| = 5)).sum = 24) :=
sorry

end sum_of_possible_values_of_p_s_l675_675615


namespace discounted_price_correct_l675_675029

noncomputable def verify_discounted_price (P : ℝ) (d1 d2 d3 : ℝ) : ℝ :=
  let P1 := P * (1 - d1 / 100)
  let P2 := P1 * (1 - d2 / 100)
  P2 * (1 - d3 / 100)

theorem discounted_price_correct (P : ℝ) (d1 d2 d3 : ℝ) (C : ℝ) :
  P = 9649.12 →
  d1 = 20 → d2 = 10 → d3 = 5 →
  verify_discounted_price P d1 d2 d3 = C →
  C = 6600.09808 :=
by
  intros hP hd1 hd2 hd3 hC
  rw [hP, hd1, hd2, hd3] at hC
  exact hC

end discounted_price_correct_l675_675029


namespace diagonal_of_regular_octagon_l675_675090

open Real

noncomputable def diagonal_length_of_regular_octagon (side_length : ℝ) : ℝ :=
  sqrt (2 * side_length^2 * (1 - cos (3 * π / 4)))

theorem diagonal_of_regular_octagon 
  (s : ℝ) 
  (h : s = 15) : 
  diagonal_length_of_regular_octagon s = sqrt (900 + 225 * sqrt 2) := 
by 
  rw h 
  unfold diagonal_length_of_regular_octagon 
  simp 
  sorry -- The actual computation proof goes here

end diagonal_of_regular_octagon_l675_675090


namespace circles_in_triangle_l675_675224

theorem circles_in_triangle (A₁ A₂ A₃ : Type) (ω₁ ω₂ ω₃ ω₄ ω₅ ω₆ ω₇ : Type) 
  (h₁ : ω₁ passes_through A₁ A₂)
  (h₂ : ∀ k ∈ {2, 3, 4, 5, 6, 7}, ωₖ externally_tangent_to ωₖ₋₁ ∧ ωₖ passes_through Aₖ Aₖ₊₁)
  (h₃ : ∀ n ≥ 1, Aₙ₊₃ = Aₙ) :
  ω₇ = ω₁ :=
sorry

end circles_in_triangle_l675_675224


namespace segment_coverage_l675_675358

theorem segment_coverage (segments : list (ℝ × ℝ)) (h : ∀ t ∈ segments, 0 < t.2 - t.1) :
  ∃ (selected : list (ℝ × ℝ)), 
  (∀ s t ∈ selected, s ≠ t → s.2 ≤ t.1 ∨ t.2 ≤ s.1) ∧ 
  ∑ (s : ℝ × ℝ) in selected, s.2 - s.1 ≥ 1 / 2 :=
sorry

end segment_coverage_l675_675358


namespace bus_stoppage_time_l675_675446

theorem bus_stoppage_time (v1 v2 : ℝ) (h1 : v1 = 50) (h2 : v2 = 43) : (60 * (v1 - v2) / v1) = 8.4 :=
by
  rw [h1, h2]
  norm_num
  sorry

end bus_stoppage_time_l675_675446


namespace mrs_sheridan_final_cats_l675_675632

def initial_cats : ℝ := 17.5
def given_away_cats : ℝ := 6.2
def returned_cats : ℝ := 2.8
def additional_given_away_cats : ℝ := 1.3

theorem mrs_sheridan_final_cats : 
  initial_cats - given_away_cats + returned_cats - additional_given_away_cats = 12.8 :=
by
  sorry

end mrs_sheridan_final_cats_l675_675632


namespace calc_expression_l675_675350

theorem calc_expression : 2 * 0 * 1 + 1 = 1 :=
by
  sorry

end calc_expression_l675_675350


namespace constant_distance_l675_675225

-- Define the ellipse E and various points on it
variables {a b x0 y0 x1 y1 x2 y2 : ℝ}
variables (M : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ) (D : ℝ × ℝ) (P : ℝ × ℝ)

-- Definition of the elliptic curve and basic conditions
def ellipse (a b : ℝ) :=
  ∀ x y : ℝ, ((x^2 / a^2) + (y^2 / b^2) = 1)

-- Given conditions:
def conditions : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ (A = (-a, 0)) ∧ (B = (a, 0)) ∧
  (∃ k1 k2, k1 * k2 = -1/2)

-- Equation to be proved (1)
def ellipse_equation : Prop :=
  ellipse √2 1

-- Calculate the tangent lines intersection point P and establish orthogonality
def tangent_lines (x1 y1 x2 y2 : ℝ) :=
  ∀ P : ℝ × ℝ, (P = (2 * (y2 - y1) / (x1 * y2 - x2 * y1), (x1 - x2) / (x1 * y2 - x2 * y1)))

-- Condition that the vectors are orthogonal
def orthogonal_vectors (PC PD : ℝ × ℝ) :=
  PC.1 * PD.1 + PC.2 * PD.2 = 0

-- Distance from point P to origin
def distance_from_origin (P : ℝ × ℝ) :=
  sqrt (P.1^2 + P.2^2)

-- Proof that the problem's second condition holds with the given condition
theorem constant_distance (h : conditions) : distance_from_origin P = sqrt 3 :=
  sorry

end constant_distance_l675_675225


namespace a_perp_a_minus_b_l675_675513

noncomputable def a : ℝ × ℝ := (-2, 1)
noncomputable def b : ℝ × ℝ := (-1, 3)
noncomputable def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

theorem a_perp_a_minus_b : (a.1 * a_minus_b.1 + a.2 * a_minus_b.2) = 0 := by
  sorry

end a_perp_a_minus_b_l675_675513


namespace radius_of_circle_l675_675314

theorem radius_of_circle (x y : ℝ) : (x^2 + y^2 - 8*x = 0) → (∃ r, r = 4) :=
by
  intro h
  sorry

end radius_of_circle_l675_675314


namespace total_pastries_made_l675_675250

theorem total_pastries_made : 
  let lola_mini_cupcakes := 13
  let lulu_mini_cupcakes := 16
  let lola_pop_tarts := 10
  let lulu_pop_tarts := 12
  let lola_blueberry_pies := 8
  let lulu_blueberry_pies := 14
  in lola_mini_cupcakes + lulu_mini_cupcakes + lola_pop_tarts + lulu_pop_tarts + lola_blueberry_pies + lulu_blueberry_pies = 73
:= by
  sorry

end total_pastries_made_l675_675250


namespace total_questions_on_test_l675_675022

theorem total_questions_on_test (score correct_responses incorrect_responses total_questions : ℤ) :
  score = correct_responses - 2 * incorrect_responses ∧ score = 70 ∧ correct_responses = 90 → 
  total_questions = correct_responses + incorrect_responses → 
  total_questions = 100 := 
by 
  intro h; 
  cases h with hc hscore; cases hscore with hs hcorrect;
  rw [hs, hcorrect] at hc; 
  have hi : 2 * incorrect_responses = 20 := by linarith;
  have hinc : incorrect_responses = 10 := by linarith;
  have ht: total_questions = correct_responses + incorrect_responses := by linarith;
  rw hinc at ht; 
  rw hcorrect at ht; 
  linarith

end total_questions_on_test_l675_675022


namespace solve_for_v_l675_675610

open Matrix

noncomputable def a : Vector3 := ![2, -1, 1]
noncomputable def b : Vector3 := ![-1, 3, 0]
noncomputable def wanted_v : Vector3 := ![1, 2, 1]

theorem solve_for_v (v : Vector3) (h1 : v ×ₜ a = b ×ₜ a) (h2 : v ×ₜ b = a ×ₜ b) : 
  v = wanted_v := by
  sorry

end solve_for_v_l675_675610


namespace sine_double_angle_inequality_l675_675917

theorem sine_double_angle_inequality {α : ℝ} (hα1 : 0 < α) (hα2 : α < π / 4) : 
  Real.sin (2 * α) < 2 * Real.sin α :=
by
  sorry

end sine_double_angle_inequality_l675_675917


namespace problem_statement_l675_675244

variable (U S T : Set ℕ) -- Define the sets U, S, and T

-- Define the sets U, S, T as per the conditions
def U := {1, 2, 3, 4, 5, 6}
def S := {1, 4, 5}
def T := {2, 3, 4}

-- Define the complement of T with respect to U
def complement_U_T := {x ∈ U | x ∉ T}

-- State the theorem
theorem problem_statement : S ∩ complement_U_T = {1, 5} :=
by 
  sorry

end problem_statement_l675_675244


namespace a_3_value_S_n_value_l675_675863

noncomputable def a (n : ℕ) : ℚ :=
if n = 1 then 2 else (n + 1) / n

def b (n : ℕ) : ℚ :=
(a n) / (n + 1) ^ 2

def S (n : ℕ) : ℚ :=
∑ i in Finset.range (n + 1), b i

theorem a_3_value : a 3 = 4 / 3 := 
sorry

theorem S_n_value (n : ℕ) : S n = n / (n + 1) := 
sorry

end a_3_value_S_n_value_l675_675863


namespace area_ratio_triangle_l675_675582

/-- In triangle ABC, points G, H, and I are chosen on sides BC, CA, and AB, respectively, 
so that BG:GC = CH:HA = AI:IB = 2:1. Line segments AG, BH, and CI intersect at points X, 
Y, and Z. Prove that the area of triangle XYZ divided by the area of triangle ABC is 1/10. -/
theorem area_ratio_triangle (A B C G H I X Y Z : Point)
  (BG_GC : ratio (between B G C) = 2/1)
  (CH_HA : ratio (between C H A) = 2/1)
  (AI_IB : ratio (between A I B) = 2/1)
  (intersect_AGX : lies_on_line X (line_through A G))
  (intersect_BHY : lies_on_line Y (line_through B H))
  (intersect_CIZ : lies_on_line Z (line_through C I)) :
  area (triangle X Y Z) / area (triangle A B C) = 1 / 10 := 
sorry

end area_ratio_triangle_l675_675582


namespace holes_after_n_folds_l675_675790

/-- 
   After folding a square piece of paper in half 6 times and cutting a notch along each edge, 
   the number of holes in the middle of the unfolded paper is 4032.
-/
theorem holes_after_n_folds (n : ℕ) (h1 : n = 6) : 
  let k := 2 ^ n - 1 
  in k * 2 ^ n = 4032 := 
by 
  -- Definitions of internal calculations 
  sorry

end holes_after_n_folds_l675_675790


namespace isosceles_triangle_area_l675_675194

theorem isosceles_triangle_area
  (A B C D E : Type)
  [IsoscelesTriangle ABC] 
  (h1 : AB = BC)
  (h2 : BD ⊥ CE)
  (h3 : length(BD) = 15)
  (h4 : length(CE) = 18) :
  area(ABC) = 540 := by
  sorry

end isosceles_triangle_area_l675_675194


namespace count_valid_colorings_l675_675212

-- Grid has 3 rows and 3 columns.
def grid : Type := Fin 3 × Fin 3

-- Colors are represented by Fin 3 (3 different colors: 0, 1, 2).
def color := Fin 3

-- Define adjacency condition: two cells are adjacent if they share a side.
def adjacent (x y : grid) : Prop :=
  (x.1 = y.1 ∧ (x.2 = y.2 + 1 ∨ x.2 + 1 = y.2)) ∨
  (x.2 = y.2 ∧ (x.1 = y.1 + 1 ∨ x.1 + 1 = y.1))

-- Define valid coloring: no two adjacent cells share the same color.
def valid_coloring (f : grid → color) : Prop :=
  ∀ x y : grid, adjacent x y → f x ≠ f y

-- Proof that the number of valid colorings of the 3x3 grid with 3 colors is 9.
theorem count_valid_colorings : ∃ (n : ℕ), n = 9 ∧ ∀ f : grid → color, valid_coloring f → f ∈ valid_colorings n :=
sorry

end count_valid_colorings_l675_675212


namespace line_through_point_parallel_to_given_line_l675_675108

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Define the slope of the line that the new line is parallel to
def slope_of_given_line : ℝ := 2

-- Define the line equation form
def line_equation (slope : ℝ) (b : ℝ) (x : ℝ) : ℝ := slope * x + b

-- State the main theorem
theorem line_through_point_parallel_to_given_line :
  ∃ b : ℝ, (∀ x, line_equation slope_of_given_line b x = 2 * x) ∧ (line_equation slope_of_given_line b 1 = 2) :=
by
  use 0
  intro x
  -- Continuing the proof is omitted using sorry.
  sorry

end line_through_point_parallel_to_given_line_l675_675108


namespace expected_value_X_probability_X_eq_2_variance_X_variance_Y_l675_675494

section
open ProbabilityTheory

-- Random variable X follows binomial distribution B(9, 2/3)
variable (X : ℕ → ℝ) (hx : binomial 9 (2/3) X)

-- Random variable Y defined as
def Y := λ x, 2 * X x + 1

-- Expected value of X
theorem expected_value_X : E[X] = 6 :=
sorry

-- Probability P(X = 2)
theorem probability_X_eq_2 : P(X = 2) = (nat.choose 9 2) * (2/3)^2 * (1/3)^7 :=
sorry

-- Variance of X
theorem variance_X : variance[X] = 2 :=
sorry

-- Variance of Y
theorem variance_Y : variance[Y] = 8 :=
sorry

end

end expected_value_X_probability_X_eq_2_variance_X_variance_Y_l675_675494


namespace exists_natural_m_l675_675619

def n (m : ℕ) : ℕ := (Nat.factors m!).count 2

theorem exists_natural_m :
  ∃ (m : ℕ), m > 1990^(1990) ∧ m = 3^(1990) + n m := sorry

end exists_natural_m_l675_675619


namespace dealer_decision_is_mode_l675_675368

noncomputable def sales_A := 15
noncomputable def sales_B := 22
noncomputable def sales_C := 18
noncomputable def sales_D := 10

def is_mode (sales: List ℕ) (mode_value: ℕ) : Prop :=
  mode_value ∈ sales ∧ ∀ x ∈ sales, x ≤ mode_value

theorem dealer_decision_is_mode : 
  is_mode [sales_A, sales_B, sales_C, sales_D] sales_B :=
by
  sorry

end dealer_decision_is_mode_l675_675368


namespace find_temperature_l675_675521

theorem find_temperature :
  ∀ (C F : ℝ), C = (5 / 9) * (F - 32) ∧ C = 20 → F = 68 :=
by
  intros C F h
  cases h with h1 h2
  sorry

end find_temperature_l675_675521


namespace inequality_1_inequality_2_l675_675664

theorem inequality_1 (x : ℝ) : (2 * x^2 - 3 * x + 1 < 0) ↔ (1 / 2 < x ∧ x < 1) := 
by sorry

theorem inequality_2 (x : ℝ) (h : x ≠ -1) : (2 * x / (x + 1) ≥ 1) ↔ (x < -1 ∨ x ≥ 1) := 
by sorry

end inequality_1_inequality_2_l675_675664


namespace least_m_for_inequality_l675_675618

noncomputable def sequence (n : ℕ) : ℕ → ℕ 
| 1 := 1
| 2 := 2
| (n + 1) := (2 * n / (n + 1)) * sequence n - (n - 1 / (n + 1)) * sequence (n - 1)

theorem least_m_for_inequality :
  ∃ m : ℕ, ∀ n ≥ m, sequence n > (2 + 2009 / 2010) ∧ m = 4021 :=
sorry

end least_m_for_inequality_l675_675618


namespace ellipse_problem_l675_675475

theorem ellipse_problem : 
  let a := 2
  let b := sqrt 3
  let e := 1 / 2
  let M := {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}
  ∃ P : ℝ × ℝ, P = (4, 0) ∧ 
  ∀ A B C : ℝ × ℝ, 
    A ∉ {v : ℝ × ℝ | v.1^2 / a^2 + v.2^2 / b^2 = 1 ∧ v.1 ≠ 0 ∧ v.2 ≠ 0}
    ∧ C = (A.1, -A.2)
    ∧ B ∈ {v : ℝ × ℝ | (C.1 - 2) * (v.2 + C.2) = (C.2 + 2) * (v.1 - 2)}
    → 
      let PA := (A.1 - 4, A.2)
      let F2C := (A.1 - 1, -A.2)
      (range x1 : ℝ, -2 < x1 ∧ x1 < 2) 
      (range x1, (7/4 * (x1 - 10/7)^2 - 18/7)) :=
sorry

end ellipse_problem_l675_675475


namespace minimum_m_l675_675599

noncomputable def arithmetic_seq (a1 a13 : ℕ) (S9 : ℕ) : ℕ → ℕ :=
λ n, if a1 + a13 = 26 ∧ S9 = 81 then 2n - 1 else 0

def bn (a_n : ℕ) (n : ℕ) : ℚ :=
1 / ((a_n (n+1)) * (a_n (n+2)))

def Tn (b : ℕ → ℚ) (n : ℕ) : ℚ :=
finset.sum (finset.range n) b

theorem minimum_m (a1 a13 : ℕ) (S9 : ℕ) :
  let a_n := arithmetic_seq a1 a13 S9 in
  ∀ n : ℕ, 30 * (Tn (bn a_n) n) - 5 ≤ 0 :=
by
  assume a1 a13 S9,
  let a_n := arithmetic_seq a1 a13 S9,
  assume n,
  have h1 : a1 + a13 = 26, sorry,
  have h2 : S9 = 81, sorry,
  have ha_n : ∀ n, a_n n = 2n - 1, sorry,
  have hbn : ∀ n, bn a_n n = 1 / ((2 * (n + 1) - 1) * (2 * (n + 2) - 1)), sorry,
  have hTn : ∀ n, Tn (bn a_n) n = (n / (3 * (2 * n + 3))), sorry,
  have hineq : ∀ n, 30 * (n / (3 * (2 * n + 3))) - 5 ≤ 0, sorry,
  exact hineq

end minimum_m_l675_675599


namespace proposition_D_l675_675046

theorem proposition_D (a b c d : ℝ) (h1 : a < b) (h2 : c < d) : a + c < b + d :=
sorry

end proposition_D_l675_675046


namespace area_of_quadrilateral_ABCD_l675_675072

theorem area_of_quadrilateral_ABCD :
  (∀ (A B C D E : Type) (AE BE CE DE AB BC CD : ℝ),
    -- Conditions
    AE = 48 ∧
    ∠AEB = 30 ∧
    ∠BEC = 30 ∧
    ∠CED = 30 ∧
    -- Right-angled at E conditions
    (IsRightAngle (∠AEB)) ∧
    (IsRightAngle (∠BEC)) ∧
    (IsRightAngle (∠CED)) ∧
    -- Prove statement
    BE = AE / 2 ∧
    AB = BE * sqrt 3 ∧
    CE = BE / 2 ∧
    BC = CE * sqrt 3 ∧
    DE = CE / 2 ∧
    CD = DE * sqrt 3 ∧
    -- Areas of triangles
    (area_of_triangle (triangle ABE) = 1/2 * BE * AB) ∧
    (area_of_triangle (triangle BCE) = 1/2 * CE * BC) ∧
    (area_of_triangle (triangle CDE) = 1/2 * DE * CD) ∧
    -- Sum of areas
    (area_of_triangle (triangle ABE) + area_of_triangle (triangle BCE) + area_of_triangle (triangle CDE) = 378 * sqrt 3)) :=
sorry

end area_of_quadrilateral_ABCD_l675_675072


namespace necessary_but_not_sufficient_for_p_l675_675909

variable {p q r : Prop}

theorem necessary_but_not_sufficient_for_p 
  (h₁ : p → q) (h₂ : ¬ (q → p)) 
  (h₃ : q → r) (h₄ : ¬ (r → q)) 
  : (r → p) ∧ ¬ (p → r) :=
sorry

end necessary_but_not_sufficient_for_p_l675_675909


namespace smaller_third_angle_l675_675722

theorem smaller_third_angle (x y : ℕ) (h₁ : x = 64) 
  (h₂ : 2 * x + (x - y) = 180) : y = 12 :=
by
  sorry

end smaller_third_angle_l675_675722


namespace angles_of_triangle_l675_675185

theorem angles_of_triangle {A B C : ℝ}
  (h1 : 2 * (AB.vec ⬝ AC.vec) = real.sqrt 3 * (AB.len * AC.len))
  (h2 : real.sqrt 3 * (AB.len * AC.len) = 3 * BC.len^2) :
  (A = π / 6 ∧ (B = 2 * π / 3 ∧ C = π / 6) ∨ (B = π / 6 ∧ C = 2 * π / 3)) :=
sorry

end angles_of_triangle_l675_675185


namespace pipe_fill_time_l675_675768

theorem pipe_fill_time (A : ℝ) (hA : A > 0) 
    (rate_A : 1 / A)
    (rate_B : 1 / 12)
    (combined_rate : 1 / 24)
    (h : rate_A - rate_B = combined_rate) : A = 8 := sorry

end pipe_fill_time_l675_675768


namespace mod_of_1500th_number_with_9_ones_l675_675602

theorem mod_of_1500th_number_with_9_ones :
  let T := {n : ℕ | nat.popcount n = 9} in
  let M := (T.to_list.nth 1500).get_or_else 0 in
  M % 1500 = ?answer :=
by sorry

end mod_of_1500th_number_with_9_ones_l675_675602


namespace number_of_refills_l675_675526

variable (totalSpent costPerRefill : ℕ)
variable (h1 : totalSpent = 40)
variable (h2 : costPerRefill = 10)

theorem number_of_refills (h1 h2 : totalSpent = 40) (h2 : costPerRefill = 10) :
  totalSpent / costPerRefill = 4 := by
  sorry

end number_of_refills_l675_675526


namespace roots_of_polynomial_l675_675662

noncomputable def polynomial : ℝ[X] := X^4 - 9 * X^3 + 29.25 * X^2 - 40.28 * X + 19.4304

theorem roots_of_polynomial : 
  (polynomial.eval 1.1 polynomial = 0) ∧ 
  (polynomial.eval 2.4 polynomial = 0) ∧ 
  (polynomial.eval 2.3 polynomial = 0) ∧ 
  (polynomial.eval 3.2 polynomial = 0) :=
by 
  -- Here would be the proof, but for now we place sorry
  sorry

end roots_of_polynomial_l675_675662


namespace irreducible_fraction_l675_675645

-- Definition of gcd
def my_gcd (m n : Int) : Int :=
  gcd m n

-- Statement of the problem
theorem irreducible_fraction (a : Int) : my_gcd (a^3 + 2 * a) (a^4 + 3 * a^2 + 1) = 1 :=
by
  sorry

end irreducible_fraction_l675_675645


namespace fourth_person_height_l675_675713

theorem fourth_person_height 
  (H : ℕ) 
  (h_avg : (H + (H + 2) + (H + 4) + (H + 10)) / 4 = 79) : 
  H + 10 = 85 :=
by
  sorry

end fourth_person_height_l675_675713


namespace domain_of_f_l675_675299

noncomputable def f (x : ℝ) := sqrt (1 + x) + x / (1 - x)

theorem domain_of_f :
  {x : ℝ | (1 + x ≥ 0) ∧ (1 - x ≠ 0)} = {x : ℝ | (-1 ≤ x ∧ x < 1) ∨ (1 < x)} :=
by
  sorry

end domain_of_f_l675_675299


namespace solutions_of_system_l675_675514

theorem solutions_of_system :
  ∀ (x y : ℝ), (x - 2 * y = 1) ∧ (x^3 - 8 * y^3 - 6 * x * y = 1) ↔ y = (x - 1) / 2 :=
by
  -- Since this is a statement-only task, the detailed proof is omitted.
  -- Insert actual proof here.
  sorry

end solutions_of_system_l675_675514


namespace bird_families_flew_away_l675_675741

theorem bird_families_flew_away (num_africa : ℕ) (num_asia : ℕ) (h_africa : num_africa = 38) (h_asia : num_asia = 80) : num_africa + num_asia = 118 :=
by
  rw [h_africa, h_asia]
  exact rfl

end bird_families_flew_away_l675_675741


namespace brian_stones_l675_675044

variable (W B : ℕ)
variable (total_stones : ℕ := 100)
variable (G : ℕ := 40)
variable (Gr : ℕ := 60)

theorem brian_stones :
  (W > B) →
  ((W + B = total_stones) ∧ (G + Gr = total_stones) ∧ (W = 60)) :=
by
  sorry

end brian_stones_l675_675044


namespace intersection_points_distance_l675_675007

theorem intersection_points_distance (a b : ℝ) :
  (√(a^2 + 4 * (b + 2)) = √26) ∧ (√(a^2 + 4 * (b + 1)) = 3 * √2) → (√((2 - √3 - (2 + √3))^2) = 2 * √3) :=
by
  sorry

end intersection_points_distance_l675_675007


namespace beetle_walks_percentage_l675_675036

noncomputable def ant_distance_meters : ℝ := 1000
noncomputable def time_minutes : ℝ := 30
noncomputable def beetle_speed_kmh : ℝ := 1.8

theorem beetle_walks_percentage :
  let ant_distance_km := ant_distance_meters / 1000
  let time_hours := time_minutes / 60
  let ant_speed_kmh := ant_distance_km / time_hours
  let beetle_distance_km := beetle_speed_kmh * time_hours in
  (beetle_distance_km / ant_distance_km) * 100 = 90 :=
by
  sorry

end beetle_walks_percentage_l675_675036


namespace hyperbola_asymptotes_l675_675155

noncomputable def eccentricity_asymptotes (a b : ℝ) (h₁ : a > 0) (h₂ : b = Real.sqrt 15 * a) : Prop :=
  ∀ (x y : ℝ), (y = (Real.sqrt 15) * x) ∨ (y = -(Real.sqrt 15) * x)

theorem hyperbola_asymptotes (a : ℝ) (h₁ : a > 0) :
  eccentricity_asymptotes a (Real.sqrt 15 * a) h₁ (by simp) :=
sorry

end hyperbola_asymptotes_l675_675155


namespace cindy_correct_method_l675_675432

theorem cindy_correct_method (x : ℝ) (h : (x - 7) / 5 = 15) : (x - 5) / 7 = 11 := 
by
  sorry

end cindy_correct_method_l675_675432


namespace corresponding_angles_equal_l675_675271

-- Define what it means for two angles to be corresponding angles
def corresponding_angles (a b : ℝ) : Prop :=
  -- Hypothetical definition
  sorry

-- Lean 4 statement of the problem
theorem corresponding_angles_equal (a b : ℝ) (h : corresponding_angles a b) : a = b :=
by
  sorry

end corresponding_angles_equal_l675_675271


namespace maximum_avg_rate_of_change_l675_675145

def avg_rate_of_change (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (f b - f a) / (b - a)

theorem maximum_avg_rate_of_change :
  let f1 := (λ x : ℝ, x)
  let f2 := (λ x : ℝ, x^2)
  let f3 := (λ x : ℝ, x^3)
  let f4 := (λ x : ℝ, 1 / x)
  avg_rate_of_change f3 1 2 > avg_rate_of_change f1 1 2 ∧
  avg_rate_of_change f3 1 2 > avg_rate_of_change f2 1 2 ∧
  avg_rate_of_change f3 1 2 > avg_rate_of_change f4 1 2 := sorry

end maximum_avg_rate_of_change_l675_675145


namespace different_way_back_is_10_miles_farther_l675_675973

-- Define the conditions
def distance_to_destination : ℕ := 55
def time_at_destination : ℕ := 2 * 60 -- 2 hours in minutes
def time_per_mile : ℕ := 2 -- 2 minutes per mile
def total_tour_time : ℕ := 6 * 60 -- 6 hours in minutes

-- Lean statement to prove the problem
theorem different_way_back_is_10_miles_farther :
  let driving_time := total_tour_time - time_at_destination,
      total_miles_driven := driving_time / time_per_mile,
      distance_back := total_miles_driven - distance_to_destination in
  distance_back - distance_to_destination = 10 := sorry

end different_way_back_is_10_miles_farther_l675_675973


namespace solution_set_l675_675149

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 1 else -1

theorem solution_set :
  {x : ℝ | x + (x + 2) * f (x + 2) ≤ 5} = {x : ℝ | x ≤ 3 / 2} :=
by {
  sorry
}

end solution_set_l675_675149


namespace p_has_49_l675_675361

theorem p_has_49 (P : ℝ) (h : P = (2/7) * P + 35) : P = 49 :=
by
  sorry

end p_has_49_l675_675361


namespace arithmetic_sequence_index_l675_675697

theorem arithmetic_sequence_index {a : ℕ → ℕ} (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n + 3) (h₃ : a n = 2014) : n = 672 :=
by
  sorry

end arithmetic_sequence_index_l675_675697


namespace mountain_return_path_l675_675971

theorem mountain_return_path
  (lower_base_circ : ℝ) (upper_base_circ : ℝ) (incline_angle : ℝ)
  (distance_along_edge : ℝ) (lower_radius : ℝ) (upper_radius : ℝ) :
  lower_base_circ = 10 → 
  upper_base_circ = 9 → 
  incline_angle = 60 → 
  distance_along_edge = 3 → 
  lower_radius = 5 / real.pi →
  upper_radius = 9 / (2 * real.pi) →
  let height := (lower_radius - upper_radius) * real.tan (real.pi / 3) in
  let return_path := height * real.sqrt 3 / 2 in
  return_path = 5 * real.sqrt 3 / real.pi :=
sorry

end mountain_return_path_l675_675971


namespace number_of_girls_in_senior_year_l675_675409

theorem number_of_girls_in_senior_year
  (total_sample_size : ℕ) 
  (total_population : ℕ)
  (girls_less_than_boys : ℕ)
  (sampling_proportion : ℚ)
  (number_of_girls_in_sample : ℕ)
  (number_of_boys_in_sample : ℕ) :
  total_sample_size = 100 →
  girls_less_than_boys = 20 →
  sampling_proportion = (100 : ℚ) / (1200 : ℚ) →
  number_of_girls_in_sample = number_of_boys_in_sample - girls_less_than_boys →
  number_of_girls_in_sample + number_of_boys_in_sample = total_sample_size →
  number_of_girls_in_senior_year = 480 :=
by
  intro h_total_sample_size h_girls_less_than_boys h_sampling_proportion h_girls_sample_eq h_sample_eq_total
  -- Proof omitted
  sorry

end number_of_girls_in_senior_year_l675_675409


namespace not_possible_d_count_l675_675690

open Real

theorem not_possible_d_count (t s d : ℝ) (h1 : 3 * t - 4 * s = 1989) (h2 : t - s = d) (h3 : 4 * s > 0) :
  ∃ k : ℕ, k = 663 ∧ ∀ n : ℕ, 1 ≤ n ∧ n ≤ k → d ≠ n :=
by
  sorry

end not_possible_d_count_l675_675690


namespace cos_F_in_right_triangle_l675_675924

noncomputable def right_triangle_cosine (DE EF : ℝ) (h_triangle : DE > 0 ∧ EF > 0 ∧ DE^2 + (real.sqrt (EF^2 - DE^2))^2 = EF^2) : ℝ :=
let DF := real.sqrt (EF^2 - DE^2) in
DF / EF

theorem cos_F_in_right_triangle :
  right_triangle_cosine 5 13 (by split; norm_num; linarith) = 12 / 13 :=
begin
  sorry
end

end cos_F_in_right_triangle_l675_675924


namespace a_20_is_5_7_l675_675317

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then 6/7
  else
    let a_n := sequence (n - 1)
    if 0 ≤ a_n ∧ a_n < 1/2 then 2 * a_n
    else if 1/2 ≤ a_n ∧ a_n < 1 then 2 * a_n - 1
    else 0 -- included to ensure the function is well-defined but theoretically unreachable

theorem a_20_is_5_7 : sequence 20 = 5/7 :=
sorry

end a_20_is_5_7_l675_675317


namespace construct_tangent_sine_graph_l675_675636

theorem construct_tangent_sine_graph (α : ℝ) (hα_pos : 0 < α) (hα : α ≠ 0) :
  (∀x0 : ℝ, (0 < x0) ∧ (x0 < α) → 
  ∃ (P Q : ℝ × ℝ), (P.2 = sin P.1) ∧ (Q.2 = 0) ∧ 
  ∃ (line containing : ℝ → ℝ), (line containing is the tangent on y = sin x at P) :=
sorry

end construct_tangent_sine_graph_l675_675636


namespace no_inscribed_circle_in_given_pentagon_l675_675860

theorem no_inscribed_circle_in_given_pentagon : ¬ ∃ (a b c d e : ℝ), 
  a = 1 ∧ b = 2 ∧ c = 5 ∧ d = 6 ∧ e = 7 ∧ 
  (a + c < b + d + e ∧ 
   b + d < a + c + e ∧ 
   c + e < a + b + d ∧ 
   d + a < b + c + e ∧ 
   e + b < a + c + d) :=
begin
  sorry,
end

end no_inscribed_circle_in_given_pentagon_l675_675860


namespace monotonic_increasing_interval_l675_675307

theorem monotonic_increasing_interval (k : ℤ) : 
  ∀ x : ℝ, (k * real.pi + real.pi / 6 ≤ x ∧ x ≤ k * real.pi + 2 * real.pi / 3) ↔ 
  ∃ y : ℝ, y = 2 * real.sin (real.pi / 3 - 2 * x) ∧ monotone_increasing y := 
sorry

end monotonic_increasing_interval_l675_675307


namespace find_r_l675_675227

noncomputable def r_value (a b : ℝ) (h : a * b = 3) : ℝ :=
  let r := (a^2 + 1 / b^2) * (b^2 + 1 / a^2)
  r

theorem find_r (a b : ℝ) (h : a * b = 3) : r_value a b h = 100 / 9 := by
  sorry

end find_r_l675_675227


namespace exists_digit_combination_l675_675268

theorem exists_digit_combination (d1 d2 d3 d4 : ℕ) (H1 : 42 * (d1 * 10 + 8) = 2 * 1000 + d2 * 100 + d3 * 10 + d4) (H2: ∃ n: ℕ, n = 2 + d2 + d3 + d4 ∧ n % 2 = 1):
  d1 = 4 ∧ 42 * 48 = 2016 ∨ d1 = 6 ∧ 42 * 68 = 2856 :=
sorry

end exists_digit_combination_l675_675268


namespace find_m_values_l675_675086

open Nat

-- Define the product of the factorials of the first m odd natural numbers
def odd_factorials (m : ℕ) : ℕ :=
  ∏ i in (range m).map (λ n, 2 * n + 1), (fact i)

-- Define the factorial of the sum of the first m natural numbers
def sum_factorial (m : ℕ) : ℕ :=
  fact (m * (m + 1) / 2)

theorem find_m_values (m : ℕ) : (odd_factorials m = sum_factorial m) ↔ (m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4) := by
  sorry

end find_m_values_l675_675086


namespace solution_inequality_l675_675104

open Real

noncomputable def is_solution (θ : ℝ) : Prop :=
  θ ∈ Icc 0 (2 * π) ∧ cos θ ^ 5 - sin θ ^ 5 < 7 * (sin θ ^ 3 - cos θ ^ 3)

theorem solution_inequality (θ : ℝ) : is_solution θ ↔ θ ∈ Ioc (π / 4) (5 * π / 4) := by
  sorry

end solution_inequality_l675_675104


namespace sum_series_l675_675829

theorem sum_series {n : ℕ} (h : n = 1000) : 
  ∑ k in Finset.range n, 3 / ((k + 1) * (k + 3)) = 9 / 4 := 
by
  sorry

end sum_series_l675_675829


namespace henry_walks_distance_l675_675903

theorem henry_walks_distance : 
  ∃ (A B : ℝ), 
  A < B ∧ 
  (∃ (D α : ℝ), D = 3 ∧ α = 2 / 3 ∧ 
    ∀ ε > 0, ∃ N, ∀ n > N, abs (|A - B| - 3 / 2) < ε) := 
begin
  sorry
end

end henry_walks_distance_l675_675903


namespace sufficient_condition_for_vectors_l675_675163

variables {R : Type*} [InnerProductSpace ℝ R]

/-- Given two non-zero vectors a and b, proposition p: |a + b| = |a| + |b| implies proposition q: ∃ t ∈ ℝ, such that a = t • b. -/
theorem sufficient_condition_for_vectors (a b : R) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∥a + b∥ = ∥a∥ + ∥b∥) → (∃ t : ℝ, a = t • b) :=
sorry

end sufficient_condition_for_vectors_l675_675163


namespace find_ellipse_equation_existence_of_line_l675_675474

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

theorem find_ellipse_equation :
  ∃ a b : ℝ, ∃ (h1 : a > b > 0) (h2 : (a^2 - b^2) = 1),
  ellipse_equation 1 (3 / 2) → ellipse_equation x y :=
sorry

noncomputable def line_eq (k₁ x : ℝ) : ℝ :=
  k₁ * (x - 2) + 1

theorem existence_of_line :
  ∃ k₁ : ℝ,
  (∀ x (h : ellipse_equation x (line_eq k₁ x)),
  x ≠ 2) → (∃ l1 : (ℝ → ℝ), l1 ∙ l1 = 5 / 4) :=
sorry

end find_ellipse_equation_existence_of_line_l675_675474


namespace color_guard_team_row_length_l675_675297

theorem color_guard_team_row_length (n : ℕ) (p d : ℝ)
  (h_n : n = 40)
  (h_p : p = 0.4)
  (h_d : d = 0.5) :
  (n - 1) * d + n * p = 35.5 :=
by
  sorry

end color_guard_team_row_length_l675_675297


namespace initial_time_to_cover_distance_is_6_hours_l675_675405

theorem initial_time_to_cover_distance_is_6_hours
  (d : ℝ) (new_speed : ℝ) (h₁ : d = 288) (h₂ : new_speed = 32) :
  ∃ t : ℝ, (t > 0 ∧ d = 288 ∧ (32 * (3 / 2) * t) = d) := 
by
  use 6
  split
  sorry

end initial_time_to_cover_distance_is_6_hours_l675_675405


namespace vector_problem_l675_675900

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem vector_problem
  (x : ℝ)
  (a b c : ℝ × ℝ)
  (h₁ : a = (x, 2))
  (h₂ : b = (2, 1))
  (h₃ : c = (3, x))
  (h₄ : ∃ k : ℝ, a = (k * 2, k * 1)) :
  vector_magnitude (b.1 + c.1, b.2 + c.2) = 5 * real.sqrt 2 :=
by
  sorry

end vector_problem_l675_675900


namespace slope_of_line_l675_675470

-- Define parameters and conditions
variables (α t : ℝ)
def x (α t : ℝ) : ℝ := 2 + t * Real.cos α
def y (α t : ℝ) : ℝ := 1 + t * Real.sin α

-- Condition for the ellipse
def on_ellipse (x y : ℝ) : Prop := (x^2) / 16 + (y^2) / 4 = 1

-- Midpoint condition for M(2,1)
def midpoint_condition (α : ℝ) : Prop := ∀ t₁ t₂ : ℝ, (t₁ + t₂ = 0) → 
  (x α t₁ = 2 ∧ y α t₁ = 1) ∧ (x α t₂ = 2 ∧ y α t₂ = 1)

-- The goal is to prove the slope of the line
theorem slope_of_line (α : ℝ) (h_midpoint : midpoint_condition α) (h_ellipse : on_ellipse (x α t) (y α t)) :
  Real.tan α = -1 / 2 :=
by
  sorry

end slope_of_line_l675_675470


namespace sum_of_smallest_positive_solutions_l675_675839

theorem sum_of_smallest_positive_solutions :
  let f (x : ℝ) := x - Real.floor x
  let g (n : ℕ) := 1 / (n * n : ℝ) in
  (2 + 1/4) + (3 + 1/9) + (4 + 1/16) = 9 + 73/144 := by
  let x1 := 2 + 1/4
  let x2 := 3 + 1/9
  let x3 := 4 + 1/16
  -- Check the conditions inside each range
  have h1 : f x1 = g 2 := by sorry
  have h2 : f x2 = g 3 := by sorry
  have h3 : f x3 = g 4 := by sorry

  -- Calculate the sum
  have sum_solutions : x1 + x2 + x3 = 9 + 73/144 := by sorry
  exact sum_solutions

end sum_of_smallest_positive_solutions_l675_675839


namespace total_pastries_correct_l675_675251

def lola_minicupcakes : ℕ := 13
def lola_poptarts : ℕ := 10
def lola_blueberrypies : ℕ := 8

def lulu_minicupcakes : ℕ := 16
def lulu_poptarts : ℕ := 12
def lulu_blueberrypies : ℕ := 14

def total_pastries : ℕ :=
  lola_minicupcakes + lulu_minicupcakes +
  lola_poptarts + lulu_poptarts +
  lola_blueberrypies + lulu_blueberrypies

theorem total_pastries_correct : total_pastries = 73 := by
  calc
    total_pastries
        = (lola_minicupcakes + lulu_minicupcakes +
           lola_poptarts + lulu_poptarts +
           lola_blueberrypies + lulu_blueberrypies) : rfl
    ... = (13 + 16) + (10 + 12) + (8 + 14)             : by rfl
    ... = 29 + 22 + 22                                 : by rfl
    ... = 73                                           : by rfl

#eval total_pastries

end total_pastries_correct_l675_675251


namespace tan_expression_value_l675_675140

noncomputable def sequence_properties (a b : ℕ → ℝ) :=
  (a 0 * a 5 * a 10 = -3 * Real.sqrt 3) ∧
  (b 0 + b 5 + b 10 = 7 * Real.pi) ∧
  (∀ n, a (n + 1) = a n * a 1) ∧
  (∀ n, b (n + 1) = b n + (b 1 - b 0))

theorem tan_expression_value (a b : ℕ → ℝ) (h : sequence_properties a b) :
  Real.tan (b 2 + b 8) / (1 - a 3 * a 7) = -Real.sqrt 3 :=
sorry

end tan_expression_value_l675_675140


namespace Jack_received_8_letters_in_the_morning_l675_675945

variables (Lm La : ℕ)

theorem Jack_received_8_letters_in_the_morning
  (h1 : Lm = La + 1)
  (h2 : La = 7) :
  Lm = 8 :=
by {
  rw h2 at h1,
  rw h1,
  exact rfl,
}

end Jack_received_8_letters_in_the_morning_l675_675945


namespace train_speed_approx_72_km_hr_l675_675792

noncomputable def train_length : ℝ := 150
noncomputable def bridge_length : ℝ := 132
noncomputable def crossing_time : ℝ := 14.098872090232781
noncomputable def total_distance : ℝ := train_length + bridge_length
noncomputable def speed_m_s : ℝ := total_distance / crossing_time
noncomputable def conversion_factor : ℝ := 3.6
noncomputable def speed_km_hr : ℝ := speed_m_s * conversion_factor

theorem train_speed_approx_72_km_hr : abs (speed_km_hr - 72) < 0.01 :=
sorry

end train_speed_approx_72_km_hr_l675_675792


namespace scientific_notation_of_number_l675_675687

def is_scientific_notation (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ a ∧ a < 10 ∧ (∃ m : ℝ, m = a * 10^n)

def billion := 10^9

def number := 2.94 * billion

theorem scientific_notation_of_number : is_scientific_notation 2.94 8 :=
by
  sorry

end scientific_notation_of_number_l675_675687


namespace quadratic_has_distinct_real_roots_l675_675535

theorem quadratic_has_distinct_real_roots (a : ℝ) (h : a = -2) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 2 * x1 + 3 = 0 ∧ a * x2^2 + 2 * x2 + 3 = 0) :=
by
  sorry

end quadratic_has_distinct_real_roots_l675_675535


namespace probability_of_at_least_two_hits_l675_675785

namespace Probability

def binomial_coefficient (n k : ℕ) : ℕ :=
if k > n then 0 else nat.factorial n / (nat.factorial k * nat.factorial (n - k))

noncomputable def probability_of_at_least_two_hits_in_three_shots (p : ℝ) : ℝ :=
binomial_coefficient 3 2 * (p^2) * (1 - p) + binomial_coefficient 3 3 * (p^3)

theorem probability_of_at_least_two_hits {p : ℝ} (h : p = 0.6) : 
  probability_of_at_least_two_hits_in_three_shots p = 0.648 :=
by 
  sorry

end Probability

end probability_of_at_least_two_hits_l675_675785


namespace compute_factorial_of_binom_l675_675069

-- Define the binomial coefficient
def binom_coeff (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the main theorem
theorem compute_factorial_of_binom : Nat.factorial (binom_coeff 8 5) = Nat.factorial 56 :=
by
  have h: binom_coeff 8 5 = 56 := by sorry
  rw h
  rfl

end compute_factorial_of_binom_l675_675069


namespace probability_at_least_40_cents_heads_l675_675998

noncomputable def value_of_heads (p n d q h : Bool) : Real :=
  (if p then 0.01 else 0) + (if n then 0.05 else 0) + (if d then 0.10 else 0) + (if q then 0.25 else 0) + (if h then 0.50 else 0)

theorem probability_at_least_40_cents_heads :
  let outcomes := {p : Bool, n : Bool, d : Bool, q : Bool, h : Bool}
  let favorable := (outcomes.filter $ λ (o : outcomes), value_of_heads o.p o.n o.d o.q o.h >= 0.40).size
  favorable / (outcomes.size : Real) = 19 / 32 :=
by
  sorry

end probability_at_least_40_cents_heads_l675_675998


namespace original_apples_l675_675386

theorem original_apples (x : ℕ) (h : 0.40 * x = 560) : x = 1400 :=
by
  sorry

end original_apples_l675_675386


namespace find_a_l675_675871

theorem find_a (x y a : ℤ) (h₁ : x = 1) (h₂ : y = -1) (h₃ : 2 * x - a * y = 3) : a = 1 :=
sorry

end find_a_l675_675871


namespace tan_double_angle_l675_675103

theorem tan_double_angle 
  (α : ℝ) 
  (h : sin α + 2 * cos α = (sqrt 10) / 2) : 
  tan (2 * α) = -3 / 4 :=
sorry

end tan_double_angle_l675_675103


namespace monotonic_intervals_l675_675501

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 - x

theorem monotonic_intervals :
  (∀ x, x < 0 → DifferentiableAt ℝ f x ∧ f' x < 0) ∧
  (∀ x, x > 0 → DifferentiableAt ℝ f x ∧ f' x > 0) :=
by
  sorry

end monotonic_intervals_l675_675501


namespace matrix_condition_equiv_l675_675836

-- Definitions and conditions from the problem
def all_vectors (v : ℝ × ℝ) : Prop := v = (1, 0) ∨ v = (0, 1)

def satisfies_eq (M : Matrix (Fin 2) (Fin 2) ℝ) (v : ℝ × ℝ) : Prop :=
  ![![M 0 0, M 0 1], ![M 1 0, M 1 1]] ⬝ ![v.1, v.2] = ![-5 * v.1, -5 * v.2]

-- Proof statement
theorem matrix_condition_equiv :
  ∀ M : Matrix (Fin 2) (Fin 2) ℝ,
  (∀ v : ℝ × ℝ, all_vectors v → satisfies_eq M v) ↔
  M = ![![(-5 : ℝ), 0], ![0, -5]] :=
begin
  sorry -- Proof not required
end

end matrix_condition_equiv_l675_675836


namespace graphs_with_inverses_l675_675166

-- Define the graphs as per the given problem
inductive Graph
| A | B | C | D | E

-- Define the condition that we need to prove which graphs have inverses
def has_inverse : Graph → Prop
| Graph.A := false
| Graph.B := true
| Graph.C := true
| Graph.D := false
| Graph.E := false

-- Lean theorem to represent the problem statement and the correct answer
theorem graphs_with_inverses : 
  { g : Graph // has_inverse g } = {g : Graph // g = Graph.B ∨ g = Graph.C} :=
by sorry

end graphs_with_inverses_l675_675166


namespace least_number_subtracted_l675_675451

theorem least_number_subtracted (n : ℕ) (h : n = 2361) : 
  ∃ k, (n - k) % 23 = 0 ∧ k = 15 := 
by
  sorry

end least_number_subtracted_l675_675451


namespace max_n_sequence_l675_675566

theorem max_n_sequence :
  ∃ (a : Fin 9 → ℤ), (∀ i j, i < j → a i > a j) ∧ (Fin.sum_univ (fun i => a i) = 2520) ∧ 
    (∀ (k : Fin 8), a k = (1 : ℚ) / (k + 2) * (2520 - a k)) := by
  sorry

end max_n_sequence_l675_675566


namespace undelivered_briquettes_l675_675629

-- Define the conditions based on the problem
def total_briquettes (x : ℕ) : Prop :=
  ∃ d u : ℕ, d = (3 * x) / 8 + 50 ∧ u = x - d ∧ d = (5 * u) / 7

-- Define the final statement to prove
theorem undelivered_briquettes (x : ℕ) (h : total_briquettes x) : 
  ∃ u : ℕ, u = 700 := of Exists.intro 700 by sorry

end undelivered_briquettes_l675_675629


namespace homothety_center_l675_675510

open EuclideanGeometry

variables {P Q A B O : Point}
variables {ω1 ω2 : Circle}

/-- Given two circles ω1 and ω2 intersect at points A and B such that these two points are on the same side of line PQ, prove that O is the center of homothety that transforms circle ω1 to circle ω2 --/
theorem homothety_center
  (hA : ω1.contains A)
  (hB : ω1.contains B)
  (hA2 : ω2.contains A)
  (hB2 : ω2.contains B)
  (h_same_side : SameSide A B P Q)
  (h_line_same_side : ¬Collinear P Q A ∧ ¬Collinear P Q B) :
  is_center_of_homothety ω1 ω2 O :=
sorry

end homothety_center_l675_675510


namespace calculate_expression_l675_675963

noncomputable def z : ℂ := complex.cos (6 * real.pi / 11) + complex.sin (6 * real.pi / 11) * complex.I

theorem calculate_expression : 
  (\frac{z}{1 + z^2} + \frac{z^2}{1 + z^4} + \frac{z^3}{1 + z^6} = -2) 
  ∧ (z^11 = 1) :=
sorry

end calculate_expression_l675_675963


namespace sqrt_three_irrational_sqrt_three_non_repeating_non_terminating_decimal_l675_675822

theorem sqrt_three_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (p^2 = 3 * q^2) := sorry

theorem sqrt_three_non_repeating_non_terminating_decimal : ∀ (d : ℝ), d = real.sqrt 3 → irrational d := by
  intro d hd
  have hirr : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (p^2 = 3 * q^2) := sqrt_three_irrational
  rw hd at hirr
  exact real.irrational_sqrt_of_prime (by norm_num : 3.prime) hirr

end sqrt_three_irrational_sqrt_three_non_repeating_non_terminating_decimal_l675_675822


namespace circle_center_l675_675532

theorem circle_center (x y : ℝ) :
  x^2 + y^2 - 2*x + 4*y + 1 = 0 → (1, -2) = (1, -2) :=
by
  sorry

end circle_center_l675_675532


namespace triangle_expression_zero_l675_675541

theorem triangle_expression_zero (PQ PR QR : ℝ) (P Q R : ℝ) (hPQ : PQ = 8) (hPR : PR = 7) (hQR : QR = 5) (hSum : P + Q + R = 180) :
  (cos ((P - Q) / 2) / sin (R / 2)) - (sin ((P - Q) / 2) / cos (R / 2)) = 0 :=
by
  sorry

end triangle_expression_zero_l675_675541


namespace length_of_platform_l675_675743

-- Definitions for conditions
def train_length : ℕ := 300
def time_cross_platform : ℕ := 39
def time_cross_signal : ℕ := 12

-- Speed calculation
def train_speed := train_length / time_cross_signal

-- Total distance calculation while crossing the platform
def total_distance := train_speed * time_cross_platform

-- Length of the platform
def platform_length : ℕ := total_distance - train_length

-- Theorem stating the length of the platform
theorem length_of_platform :
  platform_length = 675 := by
  sorry

end length_of_platform_l675_675743


namespace relationship_l675_675968

-- Define natural numbers and geometric sequences
def is_geometric (a : ℕ → ℂ) : Prop :=
  ∃ r : ℂ, ∀ n : ℕ, a (n + 1) = r * a n

def is_geometric_odd (a : ℕ → ℂ) : Prop :=
  ∃ r : ℂ, ∀ k : ℕ, a (2 * k + 1 + 1) = r * a (2 * k + 1)

def is_geometric_even (a : ℕ → ℂ) : Prop :=
  ∃ r : ℂ, ∀ k : ℕ, a (2 * k + 2) = r * a (2 * k)

-- Proposition p: The original sequence is geometric
def p (a : ℕ → ℂ) : Prop := 
  is_geometric a

-- Proposition q: The odd and even subsequences are geometric
def q (a : ℕ → ℂ) : Prop := 
  is_geometric_odd a ∧ is_geometric_even a

-- The main theorem stating the relationship between p and q
theorem relationship (a : ℕ → ℂ) : p a → q a ∧ ¬(q a → p a) :=
begin
  sorry
end

end relationship_l675_675968


namespace radius_is_independent_variable_l675_675040

theorem radius_is_independent_variable 
  (r C : ℝ)
  (h : C = 2 * Real.pi * r) : 
  ∃ r_independent, r_independent = r := 
by
  sorry

end radius_is_independent_variable_l675_675040


namespace correct_derivatives_count_l675_675802

noncomputable def f1 (x : ℝ) : ℝ := sqrt x
noncomputable def f2 (x : ℝ) (a : ℝ) : ℝ := a^x
noncomputable def f3 (x : ℝ) : ℝ := sin (2 * x)
noncomputable def f4 (x : ℝ) : ℝ := 1 / (x + 1)

def derivative1_is_correct : Prop := deriv f1 = λ x, 1 / 2 * x^(-1 / 2)
def derivative2_is_correct : Prop := deriv (λ x, f2 x a) ≠ λ x, a^2 * log x
def derivative3_is_correct : Prop := deriv f3 ≠ λ x, cos (2 * x)
def derivative4_is_correct : Prop := deriv f4 ≠ λ x, 1 / (x + 1)

theorem correct_derivatives_count : (derivative1_is_correct ∧ ¬derivative2_is_correct ∧ ¬derivative3_is_correct ∧ ¬derivative4_is_correct) = true := 
  sorry

end correct_derivatives_count_l675_675802


namespace shifted_graph_eq_l675_675406

def f (x : ℝ) : ℝ := 3^(2 * x)

theorem shifted_graph_eq (x : ℝ) : f (x - 1) = 9^(x - 1) :=
by {
  unfold f,
  simp,
  sorry
}

end shifted_graph_eq_l675_675406


namespace range_of_f_l675_675693

noncomputable def f (x : ℝ) : ℝ := 1 + Real.log x / Real.log 2

theorem range_of_f : Set.range f = Set.Ici 3 :=
by
  sorry

end range_of_f_l675_675693


namespace triangle_BC_length_l675_675555

-- Define the triangle and given angles and side lengths
def Triangle (A B C : Type) := {
  angle_B : ℝ,
  side_AC : ℝ,
  side_AB : ℝ
}

-- Prove that the length of side BC is 3 given the conditions
theorem triangle_BC_length (A B C : Type)
  [Triangle A B C] (h₁ : A.angle_B = 120)
  (h₂ : A.side_AC = sqrt 19) (h₃ : A.side_AB = 2) :  
  ∃ (BC : ℝ), BC = 3 :=
by
  sorry

end triangle_BC_length_l675_675555


namespace non_empty_subsets_count_l675_675517

theorem non_empty_subsets_count :
  let S := {1, 2, 3, ..., 20}
  let count := (Σ k in finset.range(1, 11), nat.choose (20 - k + 1) k)
  count = 3164
  sorry

end non_empty_subsets_count_l675_675517


namespace root_in_interval_l675_675181

def f (x : ℝ) : ℝ := 2^x + x - 5

theorem root_in_interval (t : ℤ) (h : ∃ x : ℝ, (t : ℝ) < x ∧ x < (t + 1 : ℝ) ∧ f x = 0) : t = 1 :=
  sorry

end root_in_interval_l675_675181


namespace one_to_one_correspondence_l675_675985

noncomputable def bijective_mapping (plane : Type) [metric_space plane] [proper_space plane] (line : Type) [linear_ordered_field line] [topological_space_line_Ioi] :=
  ∃ (f : plane → line), bijective f ∧ ∀ (P : plane), P = f P 

theorem one_to_one_correspondence (plane : Type) [metric_space plane] [proper_space plane] (line : Type) [linear_ordered_field line] [topological_space_line_Ioi] (O : plane) (f : plane → line) :
  bijective_mapping plane line :=
sorry

end one_to_one_correspondence_l675_675985


namespace find_f_2015_l675_675886

def f (x : ℝ) : ℝ := 2015 * sin x + x^2015 + 2015 * tan x + 2015

theorem find_f_2015 : f 2015 = 2014 :=
by
  have h1 : f (-2015) = 2016 := sorry
  have h2 : f(x) - 2015 = 2015 * sin x + x^2015 + 2015 * tan x := sorry
  have h3 : ∀ x, f(x) - 2015 = -(f(-x) - 2015) := sorry
  -- Using the property of the odd function derived from the conditions
  have h4 : f(-2015) - 2015 = -(f(2015) - 2015) := by rw [h3, h1]
  have h5 : 2016 - 2015 = -(f(2015) - 2015) := sorry -- substitute h1 into h4
  have h6 : 1 = -(f(2015) - 2015) := by rw h5
  have h7 : f(2015) - 2015 = -1 := by rw h6
  have h8 : f(2015) = 2014 := by rw [h7, sub_eq_add_neg, add_neg_self]
  exact h8

end find_f_2015_l675_675886


namespace sum_is_even_probability_l675_675946

open ProbabilityTheory

-- Definitions of spinners and even sum condition
def SpinnerS := {1, 2, 4}
def SpinnerT := {1, 3, 5}
def SpinnerU := {2, 3, 6}

def isEven (n : ℕ) : Prop := n % 2 = 0

theorem sum_is_even_probability :
  (probability (λ (s t u : ℕ), s ∈ SpinnerS ∧ t ∈ SpinnerT ∧ u ∈ SpinnerU ∧ isEven (s + t + u))) = 2 / 9 := sorry

end sum_is_even_probability_l675_675946


namespace problem_statement_l675_675230

def q (x : ℤ) : ℤ := x^2010 + x^2009 + x^2008 + ... + x + 1
def divisor (x : ℤ) : ℤ := x^5 + x^4 + 2 * x^3 + 2 * x^2 + x + 1
def remainder (x : ℤ) : ℤ := q(x) % divisor(x)

theorem problem_statement (x : ℤ) :
  remainder(2010) % 1000 = 0 :=
sorry

end problem_statement_l675_675230


namespace find_f_100_l675_675680

theorem find_f_100 (f : ℝ → ℝ) (k : ℝ) (h_nonzero : k ≠ 0) 
(h_func : ∀ x y : ℝ, 0 < x → 0 < y → k * (x * f y - y * f x) = f (x / y)) : 
f 100 = 0 := 
by
  sorry

end find_f_100_l675_675680


namespace permutation_count_no_adjacent_l675_675808

theorem permutation_count_no_adjacent (digits : Finset ℕ) (symbols : Finset Char) :
  digits = {1, 2, 3} → symbols = {'+', '-'} → 
  (∃ (valid_permutations : Finset (List (ℕ ⊕ Char))), 
  (∀ l ∈ valid_permutations, no_adjacent_digits l digits) ∧ valid_permutations.card = 12) := by
  intros hDigits hSymbols
  -- not proving yet, just stating the theorem
  sorry

-- Helper definition to state that no two digits are adjacent in a list
def no_adjacent_digits {α : Type*} (l : List α) (digits : Finset α) : Prop :=
  ∀ (x y : α) (i j : ℕ), x ∈ digits → y ∈ digits → i < l.length → j < l.length → l.nth i = some x → l.nth j = some y → abs (i - j) ≠ 1

end permutation_count_no_adjacent_l675_675808


namespace find_base_k_l675_675748

theorem find_base_k : ∃ k : ℕ, 6 * k^2 + 6 * k + 4 = 340 ∧ k = 7 := 
by 
  sorry

end find_base_k_l675_675748


namespace calculate_product_sum_l675_675424

theorem calculate_product_sum :
  17 * (17/18) + 35 * (35/36) = 50 + 1/12 :=
by sorry

end calculate_product_sum_l675_675424


namespace last_digit_to_appear_mod9_l675_675818

def fib (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fib (n - 1) + fib (n - 2)

def fib_mod9 (n : ℕ) : ℕ :=
  (fib n) % 9

theorem last_digit_to_appear_mod9 :
  ∃ n : ℕ, ∀ m : ℕ, m < n → fib_mod9 m ≠ 0 ∧ fib_mod9 n = 0 :=
sorry

end last_digit_to_appear_mod9_l675_675818


namespace BP_bisected_at_AQ_l675_675110

variables {A B C M P Q : Type*}
variables [linear_ordered_field A] 

-- Definitions for midpoint and parallelogram
def is_midpoint (A M C : Point) : Prop := dist A M = dist M C

-- Assuming geometric properties
variable (hm : is_midpoint A M C)

-- Define AQ, BM, PQ parallel properties
variable (hP_on_BM : false)
variable (hQ : false)
variable (PQ_parallel_AB : false)
variable (CQ_parallel_BM : false)

theorem BP_bisected_at_AQ : midpoint (P, A) (Q, B) sorry : midpoint (P, A) (Q, B) := sorry

end BP_bisected_at_AQ_l675_675110


namespace balance_of_five_diamonds_l675_675457

variables (Δ ♦ • : ℕ)

def condition1 : Prop := 4 * Δ + 3 * ♦ = 15 * •
def condition2 : Prop := 2 * Δ = ♦ + 4 * •

theorem balance_of_five_diamonds (h1 : condition1 Δ ♦ •) (h2 : condition2 Δ ♦ •) : 5 * ♦ = 7 * • :=
by
  sorry

end balance_of_five_diamonds_l675_675457


namespace minimize_z_l675_675285

-- Define the function z
def z (x a b c d : ℝ) : ℝ :=
  (x - a)^2 + (x - b)^2 + c * (x - a) + d * (x - b)

-- Prove that the argument that minimizes z is (2(a+b) - (c+d)) / 4
theorem minimize_z (a b c d : ℝ) : 
  argmin (λ x, z x a b c d) = (2 * (a + b) - (c + d)) / 4 :=
by
  sorry -- Proof to be provided

end minimize_z_l675_675285


namespace probability_less_than_8000_miles_l675_675674

def cities : List String := ["Bangkok", "Cape Town", "Honolulu", "London"]

def distances : ℕ × String × String → ℕ
| (1, "Bangkok", "Cape Town") => 6300
| (2, "Bangkok", "Honolulu") => 6609
| (3, "Bangkok", "London") => 5944
| (4, "Cape Town", "Honolulu") => 11535
| (5, "Cape Town", "London") => 5989
| (6, "Honolulu", "London") => 7240
| _ => 0

theorem probability_less_than_8000_miles (h : List String := cities) (d : ℕ × String × String → ℕ := distances) :
  /- This calculates the probability that the distance between any two randomly chosen cities from the set 
  [Bangkok, Cape Town, Honolulu, London] is less than 8000 miles, which should be 5/6. -/
  let pairs : List (String × String) := [("Bangkok", "Cape Town"), ("Bangkok", "Honolulu"), ("Bangkok", "London"),
                                         ("Cape Town", "Honolulu"), ("Cape Town", "London"), ("Honolulu", "London")] in
  let count_less_than_8000 := (pairs.filter (λ (c : String × String), 
    match c with
    | ("Bangkok", "Cape Town") => d (1, "Bangkok", "Cape Town") < 8000
    | ("Bangkok", "Honolulu") => d (2, "Bangkok", "Honolulu") < 8000
    | ("Bangkok", "London") => d (3, "Bangkok", "London") < 8000
    | ("Cape Town", "Honolulu") => d (4, "Cape Town", "Honolulu") < 8000
    | ("Cape Town", "London") => d (5, "Cape Town", "London") < 8000
    | ("Honolulu", "London") => d (6, "Honolulu", "London") < 8000
    | _ => false
  )).length in
  (count_less_than_8000.toRational / pairs.length.toRational) = 5 / 6 :=
sorry

end probability_less_than_8000_miles_l675_675674


namespace quadrant_of_complex_number_l675_675893

theorem quadrant_of_complex_number
  (h : ∀ x : ℝ, 0 < x → (a^2 + a + 2)/x < 1/x^2 + 1) :
  ∃ a : ℝ, -1 < a ∧ a < 0 ∧ i^27 = -i :=
sorry

end quadrant_of_complex_number_l675_675893


namespace corresponding_angles_equal_l675_675274

-- Definition: Corresponding angles and their equality
def corresponding_angles (α β : ℝ) : Prop :=
  -- assuming definition of corresponding angles can be defined
  sorry

theorem corresponding_angles_equal {α β : ℝ} (h : corresponding_angles α β) : α = β :=
by
  -- the proof is provided in the problem statement
  sorry

end corresponding_angles_equal_l675_675274


namespace find_fraction_l675_675091

variable {N : ℕ}
variable {f : ℚ}

theorem find_fraction (h1 : N = 150) (h2 : N - f * N = 60) : f = 3/5 := by
  sorry

end find_fraction_l675_675091


namespace least_number_of_homeowners_l675_675559

theorem least_number_of_homeowners (total_members : ℕ) 
(num_men : ℕ) (num_women : ℕ) 
(homeowners_men : ℕ) (homeowners_women : ℕ) 
(h_total : total_members = 5000)
(h_men_women : num_men + num_women = total_members) 
(h_percentage_men : homeowners_men = 15 * num_men / 100)
(h_percentage_women : homeowners_women = 25 * num_women / 100):
  homeowners_men + homeowners_women = 4 :=
sorry

end least_number_of_homeowners_l675_675559


namespace variance_of_9_data_points_l675_675143

theorem variance_of_9_data_points 
  (data_points : Fin 8 → ℝ)
  (average_8 : (∑ i, data_points i) / 8 = 5)
  (variance_8 : (∑ i, (data_points i - 5) ^ 2) / 8 = 3)
  (new_point : ℝ)
  (new_point_is_5 : new_point = 5) :
  let data_points_9 := (data_points, new_point)
  in (∑ i, (if h : i.1 < 8 then data_points i else new_point) - 5) ^ 2 / 9 = 8 / 3 :=
by
  sorry

end variance_of_9_data_points_l675_675143


namespace solve_for_x_l675_675660

theorem solve_for_x (x: ℝ) (h: (x-3)^4 = 16): x = 5 := 
by
  sorry

end solve_for_x_l675_675660


namespace corresponding_angles_equal_l675_675269

-- Define what it means for two angles to be corresponding angles
def corresponding_angles (a b : ℝ) : Prop :=
  -- Hypothetical definition
  sorry

-- Lean 4 statement of the problem
theorem corresponding_angles_equal (a b : ℝ) (h : corresponding_angles a b) : a = b :=
by
  sorry

end corresponding_angles_equal_l675_675269


namespace qin_jiushao_l675_675339

def f (x : ℤ) : ℤ := 2 * x ^ 6 + 5 * x ^ 5 + 6 * x ^ 4 + 23 * x ^ 3 - 8 * x ^ 2 + 10 * x - 3

theorem qin_jiushao (x : ℤ) : x = -4 → 
  let v0 := 2 in
  let v1 := v0 * x + 5 in
  let v2 := v1 * x + 6 in
  let v3 := v2 * x + 23 in
  v3 = -49 :=
begin
  intro h,
  subst h,
  let v0 := 2,
  let v1 := v0 * -4 + 5,
  let v2 := v1 * -4 + 6,
  let v3 := v2 * -4 + 23,
  show v3 = -49,
  sorry
end

end qin_jiushao_l675_675339


namespace part_a_proof_part_b_proof_l675_675637

variables {A B C O C1 A1 B1 : Point}

-- Given conditions
variable (triangleABC : Triangle A B C)
variable (on_side_AB : Point_on_side C1 A B)
variable (on_side_BC : Point_on_side A1 B C)
variable (on_side_CA : Point_on_side B1 C A)
variable (intersect_at_O : Intersection (Line_through C C1) (Line_through A A1) (Line_through B B1) O)

-- Definitions for ratios
def ratio_CO_OC1 := dist C O / dist O C1
def ratio_CA1_A1B := dist C A1 / dist A1 B
def ratio_CB1_B1A := dist C B1 / dist B1 A
def ratio_AO_OA1 := dist A O / dist O A1
def ratio_BO_OB1 := dist B O / dist O B1

-- Part (a) Proof statement
theorem part_a_proof : ratio_CO_OC1 = ratio_CA1_A1B + ratio_CB1_B1A := sorry

-- Part (b) Proof statement
theorem part_b_proof : ratio_AO_OA1 * ratio_BO_OB1 * ratio_CO_OC1 ≥ 8 := sorry

end part_a_proof_part_b_proof_l675_675637


namespace extremum_condition_l675_675464

theorem extremum_condition (a b : ℝ) (f : ℝ → ℝ) (h₁ : f = λ x, x^3 + 3 * a * x^2 + b * x + a^2)
  (h₂ : (deriv f) (-1) = 0) (h₃ : f (-1) = 0) : a - b = -7 :=
sorry

end extremum_condition_l675_675464


namespace chloe_percent_of_two_dollars_l675_675431

theorem chloe_percent_of_two_dollars :
  let total_cents := 2 + 5 + 10 + 50 in
  let two_dollars := 200 in
  (total_cents: ℝ) / (two_dollars: ℝ) * 100 = 33.5 :=
by
  sorry

end chloe_percent_of_two_dollars_l675_675431


namespace quiz_answer_keys_count_l675_675931

noncomputable def count_answer_keys : ℕ :=
  (Nat.choose 10 5) * (Nat.factorial 6)

theorem quiz_answer_keys_count :
  count_answer_keys = 181440 := 
by
  -- Proof is skipped, using sorry
  sorry

end quiz_answer_keys_count_l675_675931


namespace goods_train_length_is_470_l675_675003

noncomputable section

def speed_kmph := 72
def platform_length := 250
def crossing_time := 36

def speed_mps := speed_kmph * 5 / 18
def distance_covered := speed_mps * crossing_time

def length_of_train := distance_covered - platform_length

theorem goods_train_length_is_470 :
  length_of_train = 470 :=
by
  sorry

end goods_train_length_is_470_l675_675003


namespace max_profit_l675_675769

theorem max_profit (x y : ℝ) (h1 : L1 = -x^2 + 21 * x) (h2 : L2 = 2 * y) (h3 : x + y = 15) : 
  max_profit L1 L2 = 120 := 
by sorry

end max_profit_l675_675769


namespace remainder_of_sum_mod_8_l675_675170

theorem remainder_of_sum_mod_8 (x y z : ℤ) 
  (h1 : x * y * z ≡ 1 [MOD 8]) 
  (h2 : 3 * z ≡ 5 [MOD 8])
  (h3 : 7 * y ≡ 4 + y [MOD 8]) : 
  (x + y + z) % 8 = 1 := 
by 
  sorry

end remainder_of_sum_mod_8_l675_675170


namespace value_of_C_l675_675801

theorem value_of_C (
  roots : Fin₅ → ℤ
) : (∀ i, 0 < roots i) ∧ (∑ i, roots i = 15) ∧ (∏ i : Fin₅, z - roots i = z^5 - 15*z^4 + A*z^3 + B*z^2 + C*z + 81) → C = -405 :=
sorry

end value_of_C_l675_675801


namespace profit_growth_equation_l675_675341

noncomputable def profitApril : ℝ := 250000
noncomputable def profitJune : ℝ := 360000
noncomputable def averageMonthlyGrowth (x : ℝ) : ℝ := 25 * (1 + x) * (1 + x)

theorem profit_growth_equation (x : ℝ) :
  averageMonthlyGrowth x = 36 * 10000 ↔ 25 * (1 + x)^2 = 36 :=
by
  sorry

end profit_growth_equation_l675_675341


namespace total_pastries_made_l675_675249

theorem total_pastries_made : 
  let lola_mini_cupcakes := 13
  let lulu_mini_cupcakes := 16
  let lola_pop_tarts := 10
  let lulu_pop_tarts := 12
  let lola_blueberry_pies := 8
  let lulu_blueberry_pies := 14
  in lola_mini_cupcakes + lulu_mini_cupcakes + lola_pop_tarts + lulu_pop_tarts + lola_blueberry_pies + lulu_blueberry_pies = 73
:= by
  sorry

end total_pastries_made_l675_675249


namespace wire_cut_min_segments_l675_675435

theorem wire_cut_min_segments (n : ℕ) (h : n ≥ 3) 
  (segments : ℕ → ℝ) 
  (wire_len : ∑ i in Finset.range n, segments i = 100)
  (min_len : ∀ i, i < n → segments i ≥ 10)
  (triangle_condition : 
    ∀ (a b c : ℕ), a < n → b < n → c < n → a ≠ b → b ≠ c → a ≠ c → 
    ¬(segments a + segments b > segments c ∧ 
      segments b + segments c > segments a ∧ 
      segments c + segments a > segments b) → 
    false) :
  n ≥ 5 :=
sorry

end wire_cut_min_segments_l675_675435


namespace gcd_2_l675_675835

-- Define the two numbers obtained from the conditions.
def n : ℕ := 3589 - 23
def m : ℕ := 5273 - 41

-- State that the GCD of n and m is 2.
theorem gcd_2 : Nat.gcd n m = 2 := by
  sorry

end gcd_2_l675_675835


namespace length_QR_l675_675938

-- Define the points and segments in the triangle
variables {P Q R : Type} [RightTriangle P Q R]

-- Define the lengths of segments and the cosine condition
variable (QP : ℝ) (h1 : QP = 16)
variable (cos_Q : ℝ) (h2 : cos_Q = 0.5)

-- Define the goal
theorem length_QR : QR = 32 :=
by
-- Proof implementation here
sorry

end length_QR_l675_675938


namespace paving_cost_correct_l675_675389

noncomputable def total_paving_cost (length_rect : ℝ) (width_rect : ℝ) (radius_semi : ℝ)
(rate_rect : ℝ) (rate_semi : ℝ) : ℝ :=
(length_rect * width_rect * rate_rect) + (0.5 * real.pi * radius_semi^2 * rate_semi)

theorem paving_cost_correct :
  total_paving_cost 6.5 4.75 3.25 800 950 = 40488.05 :=
begin
  sorry
end

end paving_cost_correct_l675_675389


namespace triangle_ABC_is_obtuse_l675_675943

open Real

-- Define the given conditions
variables (A B : ℝ) (a b : ℝ)
hypothesis h_B_eq_2A : B = 2 * A
hypothesis h_a_eq_1 : a = 1
hypothesis h_b_eq_4_over_3 : b = 4 / 3

-- The theorem to be proven
theorem triangle_ABC_is_obtuse (A B a b : ℝ)
  (h_B_eq_2A : B = 2 * A)
  (h_a_eq_1 : a = 1)
  (h_b_eq_4_over_3 : b = 4 / 3) : 
  ∃ (C : ℝ), C > 0 ∧ C < π - A - B ∧ (B > π / 2) :=
by 
  sorry

end triangle_ABC_is_obtuse_l675_675943


namespace pigeonhole_sum_l675_675427

theorem pigeonhole_sum {n : ℕ} :
  ¬ ∃ (M : matrix (fin n) (fin n) ℤ),
    (∀ i j, M i j ∈ {-1, 0, 1}) ∧
    (∀ i j, M i j = 0 ∨ M i j = 1 ∨ M i j = -1) ∧
    let row_sum (i : fin n) := ∑ j, M i j in
    let col_sum (j : fin n) := ∑ i, M i j in
    let diag1_sum := ∑ i, M i i in
    let diag2_sum := ∑ i, M i (fin.reverse i) in
    function.injective row_sum ∧
    function.injective col_sum ∧
    row_sum diag1_sum ∧
    row_sum diag2_sum := 
sorry

end pigeonhole_sum_l675_675427


namespace circles_intersect_at_m_eq_2_no_m_such_that_circles_contain_each_other_l675_675867

noncomputable def circle₁ (m : ℝ) : set (ℝ × ℝ) := 
  {p | (p.1 ^ 2 + p.2 ^ 2 - 2 * m * p.1 + 4 * p.2 + m ^ 2 - 5 = 0)}

noncomputable def circle₂ : set (ℝ × ℝ) := 
  {p | (p.1 ^ 2 + p.2 ^ 2 + 4 * p.1 = 0)}

theorem circles_intersect_at_m_eq_2 : 
  ∃ p, p ∈ circle₁ 2 ∧ p ∈ circle₂ := 
sorry

theorem no_m_such_that_circles_contain_each_other : 
  ∀ m : ℝ, ¬ (∀ p ∈ circle₁ m, p ∈ circle₂ ∧ ∀ q ∈ circle₂, q ∈ circle₁ m) := 
sorry

end circles_intersect_at_m_eq_2_no_m_such_that_circles_contain_each_other_l675_675867


namespace additional_payment_each_friend_l675_675780

theorem additional_payment_each_friend (initial_cost : ℕ) (earned_amount : ℕ) (total_friends : ℕ) (final_friends : ℕ) 
(h_initial_cost : initial_cost = 1700) (h_earned_amount : earned_amount = 500) 
(h_total_friends : total_friends = 6) (h_final_friends : final_friends = 5) : 
  ((initial_cost - earned_amount) / total_friends) / final_friends = 40 :=
sorry

end additional_payment_each_friend_l675_675780


namespace suraj_innings_count_l675_675292

theorem suraj_innings_count
  (A : ℕ := 24)  -- average before the last innings
  (new_average : ℕ := 28)  -- Suraj’s average after the last innings
  (last_score : ℕ := 92)  -- Suraj’s score in the last innings
  (avg_increase : ℕ := 4)  -- the increase in average after the last innings
  (n : ℕ)  -- number of innings before the last one
  (h_avg : A + avg_increase = new_average)  -- A + 4 = 28
  (h_eqn : n * A + last_score = (n + 1) * new_average) :  -- n * 24 + 92 = (n + 1) * 28
  n = 16 :=
by {
  sorry
}

end suraj_innings_count_l675_675292


namespace log_a_b_conditions_l675_675519

open Real

theorem log_a_b_conditions (a b : ℝ) (h0 : 0 < b) (h1 : b < 1) (h2 : log a b < 1) : (0 < a ∧ a < b) ∨ (a > 1) :=
  sorry

end log_a_b_conditions_l675_675519


namespace maximize_angle_A_CB_l675_675306

theorem maximize_angle_A_CB :
  ∃ C : ℝ × ℝ, C = (2, 4) ∧
  (C ∈ {p : ℝ × ℝ | p.2 = 2 * p.1}) ∧
  ∀ C', C' ∈ {p : ℝ × ℝ | p.2 = 2 * p.1} → 
    ∠A C B ≤ ∠A (2, 4) B :=
begin
  -- Define points A and B
  let A : ℝ × ℝ := (2, 2),
  let B : ℝ × ℝ := (6, 2),
  -- Define the line c
  let line_c := {p : ℝ × ℝ | p.2 = 2 * p.1},
  
  -- Define the tangent point C
  let C : ℝ × ℝ := (2, 4),
  -- Assume C lies on the line c
  have C_on_line : C ∈ line_c,
  { simp [C, line_c] },

  -- State the existence of point C
  use C,
  
  -- Split the conjunction
  split,
  { refl },
  split,
  { exact C_on_line },
  
  -- Consider any other point C' on the line and compare angles
  intros C' C'_on_line,
  -- Here, you'd provide the detailed proof comparing angles (skipping with sorry for now)
  sorry,
end

end maximize_angle_A_CB_l675_675306


namespace vampire_count_l675_675666

variable (V : ℕ)

axiom initial_vampires (h1 : ∀ v : ℕ, 36 * v = 72) : V = 2

theorem vampire_count : ∃ V : ℕ, 36 * V = 72 ∧ V = 2 :=
begin
  have init_vamps := initial_vampires (λ v, begin
    sorry,
  end),
  use 2,
  split,
  { exact calc
      36 * 2 = 72 : by norm_num },
  { exact init_vamps }
end

end vampire_count_l675_675666


namespace triangle_ratio_DE_EF_l675_675940

theorem triangle_ratio_DE_EF
  (X Y Z D E F : Type)
  (XD DY YE EZ DE EF : ℝ)
  (h1 : XD / DY = 4 / 1)
  (h2 : YE / EZ = 4 / 1)
  (h3 : DE + EF = XD + YE) : 
  DE / EF = 1 / 4 := 
sorry

end triangle_ratio_DE_EF_l675_675940


namespace transform_graph_of_g_to_f_l675_675717

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - Real.sqrt 3 * Real.sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x) + 1

theorem transform_graph_of_g_to_f :
  ∀ (x : ℝ), f x = g (x + (5 * Real.pi) / 12) :=
by
  sorry

end transform_graph_of_g_to_f_l675_675717


namespace total_travel_distance_when_hits_ground_fifth_time_l675_675758

def rebounded_height (h : ℝ) : ℝ := 0.4 * h

noncomputable def total_distance : ℝ :=
  let h₀ := 150 in
  let h₁ := rebounded_height h₀ in
  let h₂ := rebounded_height h₁ in
  let h₃ := rebounded_height h₂ in
  let h₄ := rebounded_height h₃ in
  h₀ + h₁ + h₁ + h₂ + h₂ + h₃ + h₃ + h₄ + h₄ + h₄

theorem total_travel_distance_when_hits_ground_fifth_time :
  total_distance = 344.88 :=
sorry

end total_travel_distance_when_hits_ground_fifth_time_l675_675758


namespace three_planes_six_parts_l675_675715

-- Lean statement to express the problem
theorem three_planes_six_parts (P1 P2 P3 : Plane) (h : dividesIntoParts P1 P2 P3 6) : 
  intersectionLines P1 P2 P3 = 1 ∨ intersectionLines P1 P2 P3 = 2 := 
sorry

end three_planes_six_parts_l675_675715


namespace analytical_expression_of_f_range_m_condition_l675_675116

def f (x : ℝ) : ℝ :=
if x < 0 then -x ^ 2 - 2 * x + 3
else if x = 0 then 0
else x ^ 2 - 2 * x - 3

theorem analytical_expression_of_f :
  ∀ x : ℝ, f x = if x < 0 then -x ^ 2 - 2 * x + 3
        else if x = 0 then 0
        else x ^ 2 - 2 * x - 3 := by
  -- proof would go here
  sorry

theorem range_m_condition :
  ∃ x1 x2 ∈ set.Icc (-4:ℝ) 4, ∀ m : ℝ, |f x1 - f x2| ≥ m → m ≤ 10 :=
by
  -- proof would go here
  sorry

end analytical_expression_of_f_range_m_condition_l675_675116


namespace remainder_of_T_mod_1000_l675_675609

def T : ℤ :=
  ∑ n in finset.range 2011, (-1)^n * nat.choose 6006 (3 * n + 1)

theorem remainder_of_T_mod_1000 : T % 1000 = 973 :=
  sorry

end remainder_of_T_mod_1000_l675_675609


namespace triangle_angle_side_relation_l675_675264

theorem triangle_angle_side_relation (A B C : ℝ) (a b c : ℝ) : 
  (A + B + C = π) → (cos C = 0 ↔ a^2 + b^2 = c^2) :=
by
  sorry

end triangle_angle_side_relation_l675_675264


namespace paint_time_for_two_people_l675_675659

/-- 
Proof Problem Statement: Prove that it would take 12 hours for two people to paint the house
given that six people can paint it in 4 hours, assuming everyone works at the same rate.
--/
theorem paint_time_for_two_people 
  (h1 : 6 * 4 = 24) 
  (h2 : ∀ (n : ℕ) (t : ℕ), n * t = 24 → t = 24 / n) : 
  2 * 12 = 24 :=
sorry

end paint_time_for_two_people_l675_675659


namespace problem_solution_l675_675408

def is_increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ ⦃a b⦄, a ∈ I → b ∈ I → a < b → f a < f b

def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
∀ x, f x = f (x + T)

def satisfies_conditions (f : ℝ → ℝ) : Prop :=
is_increasing_on f (set.Ioo 0 (π / 2)) ∧ is_even_function f ∧ has_period f π

theorem problem_solution :
  satisfies_conditions (λ x, abs (sin x)) :=
by sorry

end problem_solution_l675_675408


namespace price_per_dozen_proof_l675_675678

def price_per_dozen (P : ℝ) : Prop :=
  let first_doughnut_price := 1
  let total_doughnuts := 48
  let total_spent := 24
  let remaining_doughnuts := total_doughnuts - 1
  let dozens := remaining_doughnuts / 12
  let remaining_money := total_spent - first_doughnut_price
  (dozens * P = remaining_money) ∧ (P = 276 / 47) ∧ (P ≈ 5.87)

theorem price_per_dozen_proof (P : ℝ) : price_per_dozen P :=
  sorry

end price_per_dozen_proof_l675_675678


namespace sum_of_cn_l675_675496

-- Definitions according to conditions
def Sn (n : ℕ) : ℝ := sorry -- sum of the first n terms of the arithmetic sequence {a_n}
def Tn (n : ℕ) : ℝ := sorry -- sum of the first n terms of the sequence {b_n}
def an (n : ℕ) : ℝ := 3 * n
def bn (n : ℕ) : ℝ := 3 * 2^(n-1)
def cn (n : ℕ) : ℝ := if n % 2 = 1 then bn n else an n

-- Main theorem statement
theorem sum_of_cn (n : ℕ) : 
  ∑ i in finset.range n, cn (i + 1) = 
  if n % 2 = 0 then 
    2 ^ n - 1 + (3 / 4) * n^2 + (3 / 2) * n 
  else 
    2^(n + 1) + (3 / 4) * n^2 - 7 / 4 := 
sorry

end sum_of_cn_l675_675496


namespace average_rainfall_is_4_l675_675704

namespace VirginiaRainfall

def march_rainfall : ℝ := 3.79
def april_rainfall : ℝ := 4.5
def may_rainfall : ℝ := 3.95
def june_rainfall : ℝ := 3.09
def july_rainfall : ℝ := 4.67

theorem average_rainfall_is_4 :
  (march_rainfall + april_rainfall + may_rainfall + june_rainfall + july_rainfall) / 5 = 4 := by
  sorry

end VirginiaRainfall

end average_rainfall_is_4_l675_675704


namespace lives_after_bonus_l675_675340

variable (X Y Z : ℕ)

theorem lives_after_bonus (X Y Z : ℕ) : (X - Y + 3 * Z) = (X - Y + 3 * Z) :=
sorry

end lives_after_bonus_l675_675340


namespace initial_money_l675_675786

theorem initial_money (M : ℝ) 
  (clothes_expense : M / 3) 
  (remaining_after_clothes : M - clothes_expense)
  (food_expense : remaining_after_clothes / 5)
  (remaining_after_food : remaining_after_clothes - food_expense)
  (travel_expense : remaining_after_food / 4)
  (remaining_after_travel : remaining_after_food - travel_expense)
  (final_amount : remaining_after_travel = 200) : 
  M = 500 := 
sorry

end initial_money_l675_675786


namespace mod_of_1500th_number_with_9_ones_l675_675601

theorem mod_of_1500th_number_with_9_ones :
  let T := {n : ℕ | nat.popcount n = 9} in
  let M := (T.to_list.nth 1500).get_or_else 0 in
  M % 1500 = ?answer :=
by sorry

end mod_of_1500th_number_with_9_ones_l675_675601


namespace triangle_BC_length_l675_675557

-- Define the triangle and given angles and side lengths
def Triangle (A B C : Type) := {
  angle_B : ℝ,
  side_AC : ℝ,
  side_AB : ℝ
}

-- Prove that the length of side BC is 3 given the conditions
theorem triangle_BC_length (A B C : Type)
  [Triangle A B C] (h₁ : A.angle_B = 120)
  (h₂ : A.side_AC = sqrt 19) (h₃ : A.side_AB = 2) :  
  ∃ (BC : ℝ), BC = 3 :=
by
  sorry

end triangle_BC_length_l675_675557


namespace matrix_condition_even_iff_l675_675846

theorem matrix_condition_even_iff (n : ℕ) (n_pos : 0 < n) :
  (∃ (A : Matrix (Fin n) (Fin n) ℤ),
    (∀ i j,
      A i j = -1 ∨ A i j = 0 ∨ A i j = 1)
    ∧
    (Finset.univ.image (λ i, (Finset.univ.sum (λ j, A i j))).disjoint 
      (Finset.univ.image (λ j, (Finset.univ.sum (λ i, A i j)))))
  ) ↔ Even n := 
sorry

end matrix_condition_even_iff_l675_675846


namespace line_through_P_parallel_to_AB_circumcircle_of_triangle_OAB_l675_675898

-- Define the points A, B and P
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (0, 2)
def P : ℝ × ℝ := (2, 3)
def O : ℝ × ℝ := (0, 0)

-- Define the functions and theorems for the problem
theorem line_through_P_parallel_to_AB :
  ∃ k b : ℝ, ∀ x y : ℝ, ((y = k * x + b) ↔ (x + 2 * y - 8 = 0)) :=
sorry

theorem circumcircle_of_triangle_OAB :
  ∃ cx cy r : ℝ, (cx, cy) = (2, 1) ∧ r^2 = 5 ∧ ∀ x y : ℝ, ((x - cx)^2 + (y - cy)^2 = r^2) ↔ ((x - 2)^2 + (y - 1)^2 = 5) :=
sorry

end line_through_P_parallel_to_AB_circumcircle_of_triangle_OAB_l675_675898


namespace trigonometric_identity_l675_675067

theorem trigonometric_identity : (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 1 - Real.sqrt 3 :=
by
  sorry

end trigonometric_identity_l675_675067


namespace surface_area_circumscribed_sphere_l675_675471

theorem surface_area_circumscribed_sphere (a : ℝ) (h : a = 2) : 
  4 * Real.pi * (√6 / 4) ^ 2 = 6 * Real.pi :=
by
  sorry

end surface_area_circumscribed_sphere_l675_675471


namespace remaining_grass_area_l675_675767

theorem remaining_grass_area 
  (d : ℝ) (r : ℝ) (path_width : ℝ) (center_to_edge : ℝ) 
  (h1 : d = 16) (h2 : r = d / 2) (h3 : path_width = 4) (h4 : center_to_edge = 2)
  (h5 : center_to_edge + path_width = r) :
  let total_area := π * r^2 in
  let remaining_grass_area := total_area - (π * (r - path_width)^2) + (π * path_width * center_to_edge) in
  remaining_grass_area = 56 * π + 16 :=
by
  sorry

end remaining_grass_area_l675_675767


namespace rosie_pie_count_l675_675987

def total_apples : ℕ := 40
def initial_apples_required : ℕ := 3
def apples_per_pie : ℕ := 5

theorem rosie_pie_count : (total_apples - initial_apples_required) / apples_per_pie = 7 :=
by
  sorry

end rosie_pie_count_l675_675987


namespace sqrt_radicand_range_l675_675180

theorem sqrt_radicand_range (x : ℝ) :
  (∃ r : ℝ, r = real.sqrt (2023 - x)) ↔ x ≤ 2023 :=
by
  sorry

end sqrt_radicand_range_l675_675180


namespace max_sum_first_n_terms_l675_675130

open Nat

-- This encapsulates the arithmetic sequence properties
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Prove that the maximum sum of the first n terms is reached at n = 5
theorem max_sum_first_n_terms
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : is_arithmetic_sequence a d)
  (h2 : a 1 > 0)
  (h3 : d < 0)
  (h4 : 10 * (a 1 + a 10) / 2 = 0) :
  ∃ n, n = 5 ∧ ∀ m, m ≠ 5 → (sum (list.range m).map a) ≤ (sum (list.range 5).map a) :=
sorry

end max_sum_first_n_terms_l675_675130


namespace female_students_proportion_and_count_l675_675635

noncomputable def num_students : ℕ := 30
noncomputable def num_male_students : ℕ := 8
noncomputable def overall_avg_score : ℚ := 90
noncomputable def male_avg_scores : (ℚ × ℚ × ℚ) := (87, 95, 89)
noncomputable def female_avg_scores : (ℚ × ℚ × ℚ) := (92, 94, 91)
noncomputable def avg_attendance_alg_geom : ℚ := 0.85
noncomputable def avg_attendance_calc : ℚ := 0.89

theorem female_students_proportion_and_count :
  ∃ (F : ℕ), F = num_students - num_male_students ∧ (F / num_students : ℚ) = 11 / 15 :=
by
  sorry

end female_students_proportion_and_count_l675_675635


namespace factory_earnings_l675_675187

def Machine := Type

variables (hours_per_day : Machine → ℕ) (production_rate : Machine → ℕ)
          (selling_price : Machine → ℕ)

variables (m1 m2 m3 m4 : Machine)

-- Conditions
def hours_per_day_m1 : hours_per_day m1 = 23 := sorry
def hours_per_day_m2 : hours_per_day m2 = 23 := sorry
def hours_per_day_m3 : hours_per_day m3 = 23 := sorry
def hours_per_day_m4 : hours_per_day m4 = 12 := sorry

def production_rate_m1 : production_rate m1 = 2 := sorry
def production_rate_m2 : production_rate m2 = 2 := sorry
def production_rate_m3 : production_rate m3 = 3 := sorry
def production_rate_m4 : production_rate m4 = 3 := sorry

def selling_price_m1 : selling_price m1 = 50 := sorry
def selling_price_m2 : selling_price m2 = 60 := sorry
def selling_price_m3 : selling_price m3 = 50 := sorry
def selling_price_m4 : selling_price m4 = 60 := sorry

-- Proof that total earnings in one day are $10670
theorem factory_earnings :
  (hours_per_day m1 * production_rate m1 * selling_price m1 +
   hours_per_day m2 * production_rate m2 * selling_price m2 +
   hours_per_day m3 * production_rate m3 * selling_price m3 +
   hours_per_day m4 * production_rate m4 * selling_price m4) = 10670 :=
by {
  rw [hours_per_day_m1, production_rate_m1, selling_price_m1,
      hours_per_day_m2, production_rate_m2, selling_price_m2,
      hours_per_day_m3, production_rate_m3, selling_price_m3,
      hours_per_day_m4, production_rate_m4, selling_price_m4],
  norm_num,
}

end factory_earnings_l675_675187


namespace final_ball_is_white_l675_675923

def initial_black_balls : ℕ := 2007
def initial_white_balls : ℕ := 2007

def operation (black : ℕ) (white : ℕ) : ℕ × ℕ :=
  if black > 0 ∧ white > 0 then (black - 1, white) -- different colors operation
  else if black ≥ 2 then (black - 2, white) -- same color (both black) operation
  else (black, white - 2) -- same color (both white) operation

def invariant (white : ℕ) : Prop :=
  white % 2 = 1

theorem final_ball_is_white :
  ∀ (black white : ℕ),
    black = initial_black_balls → white = initial_white_balls →
    (∃ black white, (black, white) = iterate operation (initial_black_balls, initial_white_balls) (initial_black_balls + initial_white_balls - 1) ∧ white = 1) :=
by sorry

end final_ball_is_white_l675_675923


namespace minimize_abs_expression_l675_675737

theorem minimize_abs_expression {x : ℝ} : 
  ((|x - 2|) + 3) ≥ ((|2 - 2|) + 3) := 
sorry

end minimize_abs_expression_l675_675737


namespace curve_C_cartesian_equation_line_intersects_curve_range_l675_675574

-- Definition of the polar and Cartesian equations
def polar_equation (θ : ℝ) : ℝ := (4 / (4 * real.sin θ ^ 2 + real.cos θ ^ 2))
def cartesian_equation (x y : ℝ) : Prop := (x ^ 2 / 4 + y ^ 2 = 1)

-- Parametric equation of line l and point P
def line_l (t α : ℝ) : ℝ × ℝ := (-1 + t * real.cos α, 1/2 + t * real.sin α)
def point_P : ℝ × ℝ := (-1, 1/2)

-- Theorem to prove the Cartesian equation of curve C
theorem curve_C_cartesian_equation :
  ∀ θ : ℝ, ∃ ρ : ℝ, ρ^2 = polar_equation θ ∧ ∀ x y, x = ρ * real.cos θ ∧ y = ρ * real.sin θ → cartesian_equation x y :=
by
  sorry

-- Theorem to prove the range of values for |PA| * |PB|
theorem line_intersects_curve_range (α : ℝ) :
  ∃ (tA tB : ℝ), ∀ P A B, P = point_P ∧ A = line_l tA α ∧ B = line_l tB α ∧ 
  (A ≠ B) ∧ (tA ≠ tB) → abs ((A.1 - P.1)^2 + (A.2 - P.2)^2) * abs ((B.1 - P.1)^2 + (B.2 - P.2)^2) ∈ set.Icc (1/2 : ℝ) 2 :=
by
  sorry

end curve_C_cartesian_equation_line_intersects_curve_range_l675_675574


namespace neither_necessary_nor_sufficient_l675_675483

theorem neither_necessary_nor_sufficient (a b : ℝ) : (((1/2)^a < (1/2)^b) ↔ (a^2 > b^2)) = false :=
by
  sorry

end neither_necessary_nor_sufficient_l675_675483


namespace sum_of_B_elements_eq_neg2_l675_675894

def A : Set ℝ := {2, 0, 1, 4}

def B : Set ℝ := {k | k ∈ (ℝ) ∧ (k^2 - 2) ∈ A ∧ (k - 2) ∉ A}

noncomputable def sum_of_elements_in_B : ℝ :=
  set.sum (id ∈ B)

theorem sum_of_B_elements_eq_neg2 : sum_of_elements_in_B = -2 := 
  sorry

end sum_of_B_elements_eq_neg2_l675_675894


namespace sequence_f_formula_l675_675316

def sequence_a : ℕ → ℚ 
| 1 := 0
| 2 := 1
| (n+3) := (1/2) * (n+3) * (sequence_a (n+2)) 
            + (1/2) * (n+3) * (n+2) * (sequence_a (n+1)) 
            + (-1)^(n+3) * (1 - (n+3)/2)

def f_n (n : ℕ) : ℚ := 
  ∑ k in finset.range n, (k+1) * (nat.choose n k) * (sequence_a (n - k))

theorem sequence_f_formula (n : ℕ) : 
  f_n n = 2 * nat.factorial n - (n + 1) := 
sorry

end sequence_f_formula_l675_675316


namespace trig_identity_l675_675060

theorem trig_identity :
  ∀ (θ : ℝ),
    θ = 70 * (π / 180) →
    (1 / Real.cos θ - (Real.sqrt 3) / Real.sin θ) = Real.sec (20 * (π / 180)) ^ 2 :=
by
  sorry

end trig_identity_l675_675060


namespace zephyr_halfway_orbit_distance_l675_675691

-- Define the parameters of the problem.
def perihelion_distance : ℝ := 3
def aphelion_distance : ℝ := 15

-- Statement: When Zephyr is halfway through its orbit, its distance from the star is 9 AU.
theorem zephyr_halfway_orbit_distance :
  (perihelion_distance + aphelion_distance) / 2 = 9 := 
by {
  simp [perihelion_distance, aphelion_distance],
  sorry
}

end zephyr_halfway_orbit_distance_l675_675691


namespace sphere_surface_area_from_volume_l675_675328

theorem sphere_surface_area_from_volume 
  (V : ℝ) (h : V = 72 * Real.pi) :
  ∃ (A : ℝ), A = 36 * Real.pi * 2^(2/3) :=
by
  sorry

end sphere_surface_area_from_volume_l675_675328


namespace monkeys_truth_both_times_l675_675188

variables (x y z m n : ℕ)
variables (animals monkey tiger fox : set ℕ)

-- Definitions of the conditions
def tiger_truthful (a : ℕ) : Prop := a ∈ tiger
def fox_lying (a : ℕ) : Prop := a ∈ fox
def monkey_can_lie_or_truth (a : ℕ) : Prop := a ∈ monkey

-- Stating the problem
theorem monkeys_truth_both_times :
  ∃ m, m = 76 →
    let tigers := 100 in
    let foxes := 100 in
    let monkeys := 100 in
    let total_groups := 100 in
    let yes_to_tigers := 138 in
    let yes_to_foxes := 188 in
    (∀ a, a ∈ animals → 
      ((a ∈ tiger ∧ tiger_truthful a) ∨ 
       (a ∈ fox ∧ fox_lying a) ∨ 
       (a ∈ monkey ∧ monkey_can_lie_or_truth a))) ∧
    (total_groups = tigers + foxes + monkeys) ∧ 
    (yes_to_tigers = 138) ∧ 
    (yes_to_foxes = 188) → m = 76 := 
begin
  sorry
end

end monkeys_truth_both_times_l675_675188


namespace quadrilateral_equal_area_division_l675_675625

noncomputable def intersection_point (A B C D : ℝ × ℝ) :=
  (23 / 6 : ℝ, 7.5)

theorem quadrilateral_equal_area_division :
  let A := (1,1) : ℝ × ℝ in
  let B := (2,4) : ℝ × ℝ in
  let C := (5,4) : ℝ × ℝ in
  let D := (6,1) : ℝ × ℝ in
  let I := intersection_point A B C D in
  I = ⟨ 23/6, 7.5 ⟩ ∧ (23 + 6 + 15 + 2 = 46) :=
by sorry

end quadrilateral_equal_area_division_l675_675625


namespace tan_alpha_tan_beta_l675_675814

-- Define the points P, Q, and R
def pointP (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
def pointQ (β : ℝ) : ℝ × ℝ := (Real.cos β, Real.sin β)
def pointR (α : ℝ) : ℝ × ℝ := (Real.cos α, -Real.sin α)

-- Define the cosine similarity function
def cosine_similarity (A B : ℝ × ℝ) : ℝ :=
  (A.1 * B.1 + A.2 * B.2) / (Real.sqrt (A.1 ^ 2 + A.2 ^ 2) * Real.sqrt (B.1 ^ 2 + B.2 ^ 2))

-- Conditions in the problem
axiom cosine_distance_PQ : ∀ α β : ℝ, 1 - cosine_similarity (pointP α) (pointQ β) = 1 / 3
axiom cosine_distance_QR : ∀ α β : ℝ, 1 - cosine_similarity (pointQ β) (pointR α) = 1 / 2

-- The main theorem to prove
theorem tan_alpha_tan_beta (α β : ℝ) : Real.tan α * Real.tan β = 1 / 7 :=
  sorry

end tan_alpha_tan_beta_l675_675814


namespace angle_bisector_slope_l675_675294

theorem angle_bisector_slope
  (m₁ m₂ : ℝ) (h₁ : m₁ = 2) (h₂ : m₂ = -1) (k : ℝ)
  (h_k : k = (m₁ + m₂ + Real.sqrt ((m₁ - m₂)^2 + 4)) / 2) :
  k = (1 + Real.sqrt 13) / 2 :=
by
  rw [h₁, h₂] at h_k
  sorry

end angle_bisector_slope_l675_675294


namespace find_S6_div_S12_l675_675247

noncomputable def arithmetic_sequence_sum (a: ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (a 1 + a n)

variable (a : ℕ → ℝ)

def S (n : ℕ) : ℝ := arithmetic_sequence_sum a n

axiom condition_1 : S 3 / S 6 = 1 / 3

theorem find_S6_div_S12 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (condition_1 : S 3 / S 6 = 1 / 3) :
  S 6 / S 12 = 3 / 10 := by
  sorry

end find_S6_div_S12_l675_675247


namespace toys_cost_price_gain_l675_675782

theorem toys_cost_price_gain (selling_price : ℕ) (cost_price_per_toy : ℕ) (num_toys : ℕ)
    (total_cost_price : ℕ) (gain : ℕ) (x : ℕ) :
    selling_price = 21000 →
    cost_price_per_toy = 1000 →
    num_toys = 18 →
    total_cost_price = num_toys * cost_price_per_toy →
    gain = selling_price - total_cost_price →
    x = gain / cost_price_per_toy →
    x = 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3] at *
  sorry

end toys_cost_price_gain_l675_675782


namespace problem_I_problem_II_l675_675131

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Condition: sides opposite the angles A, B, C
-- Condition: sin^2 B = 2 sin A sin C
axiom condition1 : sin B ^ 2 = 2 * sin A * sin C

-- Problem (I): If a = b, then show cos B = 1/4
theorem problem_I (h : a = b) : cos B = 1 / 4 := 
by sorry

-- Problem (II): If B = 90 degrees and a = √2, then show the area of triangle ABC is 1
theorem problem_II (hB : B = π / 2) (ha : a = real.sqrt 2) : 1 / 2 * a * c = 1 := 
by sorry

end problem_I_problem_II_l675_675131


namespace sum_of_corner_rectangles_area_l675_675399

theorem sum_of_corner_rectangles_area
  (side_len : ℝ)
  (square_intersections : ℝ → ℝ → ℝ)
  (central_width : ℝ)
  (central_height : ℝ) :
  side_len = 100 →
  central_width = 40 →
  central_height = 60 →
  let x := side_len - central_width,
      y := side_len - central_height in
  let A_bl := x * y,
      A_br := (central_width - x) * y,
      A_tl := x * (central_height - y),
      A_tr := (central_width - x) * (central_height - y) in
  A_bl + A_br + A_tl + A_tr = 2400 :=
begin
  intros h_side h_width h_height,
  let x := 100 - 40,
  let y := 100 - 60,
  let A_bl := x * y,
  let A_br := (60 - x) * y,
  let A_tl := x * (40 - y),
  let A_tr := (60 - x) * (40 - y),
  show A_bl + A_br + A_tl + A_tr = 2400,
  sorry
end

end sum_of_corner_rectangles_area_l675_675399


namespace problem1_proof_problem2_proof_l675_675811

-- Definition and Statement for Problem 1
noncomputable def problem1_lhs : ℚ := 
  real.sqrt 48 / real.sqrt 3 - real.sqrt (1 / 2) * real.sqrt 12 + real.sqrt 24

noncomputable def problem1_rhs : ℚ := 
  4 + real.sqrt 6

theorem problem1_proof : problem1_lhs = problem1_rhs := 
  by
  sorry

-- Definition and Statement for Problem 2
noncomputable def problem2_lhs : ℚ := 
  (real.sqrt 20 + real.sqrt 5) / real.sqrt 5 - real.sqrt 27 * real.sqrt 3 + (real.sqrt 3 + 1) * (real.sqrt 3 - 1)

noncomputable def problem2_rhs : ℚ := 
  -4

theorem problem2_proof : problem2_lhs = problem2_rhs := 
  by
  sorry

end problem1_proof_problem2_proof_l675_675811


namespace logarithm_inequality_problem_l675_675850

theorem logarithm_inequality_problem (a m n : ℝ) (ha : 0 < a) (ha1 : a < 1) (hlog : logarithm a m < logarithm a n) (hlog0 : logarithm a n < 0) :
  1 < n ∧ n < m := by
  sorry

end logarithm_inequality_problem_l675_675850


namespace int_999_column_is_C_l675_675799

def column_of_int (n : ℕ) : String :=
  let m := n - 2
  match (m / 7 % 2, m % 7) with
  | (0, 0) => "A"
  | (0, 1) => "B"
  | (0, 2) => "C"
  | (0, 3) => "D"
  | (0, 4) => "E"
  | (0, 5) => "F"
  | (0, 6) => "G"
  | (1, 0) => "G"
  | (1, 1) => "F"
  | (1, 2) => "E"
  | (1, 3) => "D"
  | (1, 4) => "C"
  | (1, 5) => "B"
  | (1, 6) => "A"
  | _      => "Invalid"

theorem int_999_column_is_C : column_of_int 999 = "C" := by
  sorry

end int_999_column_is_C_l675_675799


namespace fraction_product_identity_l675_675422

theorem fraction_product_identity :
  (2 / 3) * (5 / 7) * (11 / 13) * (17 / 19) = 1870 / 5187 :=
by
  have numerator :=  (2 * 5 * 11 * 17 : ℤ) -- numerators multiplication
  have denominator := (3 * 7 * 13 * 19 : ℤ) -- denominators multiplication
  rw [numerator, denominator]
  sorry

end fraction_product_identity_l675_675422


namespace number_of_correct_propositions_l675_675118

def prop1 (A B : Point) (α : Plane) : Prop :=
  (segment A B ∈ α) → ¬ (line A B ∈ α)

def prop2 (p1 p2 : Plane) (P : Point) : Prop :=
  (P ∈ p1 ∧ P ∈ p2) → (∃ L : Line, ∀ Q : Point, (Q ∈ p1 ∧ Q ∈ p2) ↔ (Q ∈ L))

def prop3 (L1 L2 L3 : Line) : Prop :=
  (parallel L1 L2 ∧ parallel L2 L3 ∧ parallel L1 L3) → coplanar L1 L2 L3

def prop4 (p1 p2 : Plane) (P1 P2 P3 : Point) : Prop :=
  (P1 ≠ P2 ∧ P2 ≠ P3 ∧ P1 ≠ P3 ∧ P1 ∈ p1 ∧ P2 ∈ p1 ∧ P3 ∈ p1 ∧ P1 ∈ p2 ∧ P2 ∈ p2 ∧ P3 ∈ p2) → p1 = p2

theorem number_of_correct_propositions :
  ∃ (n : ℕ), n = 1 ∧
    (¬ (∀ A B : Point, ∀ α : Plane, prop1 A B α)) ∧
    (∀ p1 p2 : Plane, ∀ P : Point, prop2 p1 p2 P) ∧
    (¬ (∀ L1 L2 L3 : Line, prop3 L1 L2 L3)) ∧
    (¬ (∀ p1 p2 : Plane, ∀ P1 P2 P3 : Point, prop4 p1 p2 P1 P2 P3))
:= 
sorry

end number_of_correct_propositions_l675_675118


namespace triangle_ABC_B_120_AC_sqrt19_AB_2_BC_3_l675_675551

theorem triangle_ABC_B_120_AC_sqrt19_AB_2_BC_3 :
  ∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C], (B = 120) → 
  (dist A C = sqrt 19) → (dist A B = 2) → (dist B C = 3).
sorry

end triangle_ABC_B_120_AC_sqrt19_AB_2_BC_3_l675_675551


namespace find_locus_of_P_l675_675477

theorem find_locus_of_P:
  ∃ x y: ℝ, (x - 1)^2 + y^2 = 9 ∧ y ≠ 0 ∧
          ((x + 2)^2 + y^2 + (x - 4)^2 + y^2 = 36) :=
sorry

end find_locus_of_P_l675_675477


namespace smallest_possible_S_l675_675736

theorem smallest_possible_S (n : ℕ) (h1 : 6 * n ≥ 500) (h2 : ∀ (k : ℕ), 6 * k < 500 → k < n) : 
  let S := 7 * n - 500 in
  S = 88 := by
  sorry

end smallest_possible_S_l675_675736


namespace total_salaries_l675_675694

variables (A_salary B_salary : ℝ)

def A_salary := 3750
def A_saving_rate := 0.05
def B_saving_rate := 0.15
def A_saving := A_salary * A_saving_rate
def B_saving := B_salary * B_saving_rate

theorem total_salaries : A_saving = B_saving → A_salary + B_salary = 5000 :=
by
  assume h : A_saving = B_saving
  have h1 : A_salary * 0.05 = B_salary * 0.15, from h
  sorry

end total_salaries_l675_675694


namespace solve_AE_in_quadrilateral_l675_675434

noncomputable def AE_in_quadrilateral (AB CD AC : ℝ) (E: ℝ) (area_eq: {AED BEC : ℝ}): ℝ := 
  let AB := 10
  let CD := 15
  let AC := 17
  let area_eq := 2 / 3
  let EC := (17 * 3) / 5
  let AE := (2 / 3) * EC
  AE

theorem solve_AE_in_quadrilateral (AB CD AC : ℝ) (E: ℝ) (area_eq: {AED BEC : ℝ}):
  AB = 10 → CD = 15 → AC = 17 → area_eq = 2 / 3 → AE_in_quadrilateral AB CD AC E area_eq = 6.8 :=
by
  intros
  simp [AE_in_quadrilateral, *]
  sorry

end solve_AE_in_quadrilateral_l675_675434


namespace price_per_gaming_chair_l675_675220

theorem price_per_gaming_chair 
  (P : ℝ)
  (price_per_organizer : ℝ := 78)
  (num_organizers : ℕ := 3)
  (num_chairs : ℕ := 2)
  (total_paid : ℝ := 420)
  (delivery_fee_rate : ℝ := 0.05) 
  (cost_organizers : ℝ := num_organizers * price_per_organizer)
  (cost_gaming_chairs : ℝ := num_chairs * P)
  (total_sales : ℝ := cost_organizers + cost_gaming_chairs)
  (delivery_fee : ℝ := delivery_fee_rate * total_sales) :
  total_paid = total_sales + delivery_fee → P = 83 := 
sorry

end price_per_gaming_chair_l675_675220


namespace inequality_system_solution_l675_675701

theorem inequality_system_solution (x : ℝ) : x + 1 > 0 → x - 3 > 0 → x > 3 :=
by
  intros h1 h2
  sorry

end inequality_system_solution_l675_675701


namespace equation_one_solutions_equation_two_solutions_l675_675663

theorem equation_one_solutions (x : ℝ) : x^2 + 2 * x - 8 = 0 ↔ x = -4 ∨ x = 2 := 
by {
  sorry
}

theorem equation_two_solutions (x : ℝ) : x * (x - 2) = x - 2 ↔ x = 2 ∨ x = 1 := 
by {
  sorry
}

end equation_one_solutions_equation_two_solutions_l675_675663


namespace black_area_remaining_l675_675075

theorem black_area_remaining :
  let initial_black : ℚ := 1 / 2
  let black_after_each_change (black_area : ℚ) : ℚ := 3 / 4 * black_area
  let black_after_n_changes (n : ℕ) (black_area : ℚ) : ℚ := (3 / 4)^n * black_area
  in black_after_n_changes 6 initial_black = 729 / 8192 :=
by 
  sorry

end black_area_remaining_l675_675075


namespace remainder_sum_abc_mod8_l675_675906

theorem remainder_sum_abc_mod8 (a b c : ℕ) (h₁ : 1 ≤ a ∧ a < 8) (h₂ : 1 ≤ b ∧ b < 8) (h₃ : 1 ≤ c ∧ c < 8) 
  (h₄ : a * b * c ≡ 1 [MOD 8]) (h₅ : 4 * b * c ≡ 3 [MOD 8]) (h₆ : 5 * b ≡ 3 + b [MOD 8]) :
  (a + b + c) % 8 = 2 := 
sorry

end remainder_sum_abc_mod8_l675_675906


namespace minimum_value_of_2x_plus_y_l675_675538

theorem minimum_value_of_2x_plus_y (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x^2 + 2 * x * y - 1 = 0) : 
  ∃ (m : ℝ), m = 2 * x + y ∧ m = sqrt 3 := 
sorry

end minimum_value_of_2x_plus_y_l675_675538


namespace original_agreed_amount_l675_675008

theorem original_agreed_amount (months: ℕ) (cash: ℚ) (uniform_price: ℚ) (received_total: ℚ) (full_year: ℚ) :
  months = 9 →
  cash = 300 →
  uniform_price = 300 →
  received_total = 600 →
  full_year = (12: ℚ) →
  ((months / full_year) * (cash + uniform_price) = received_total) →
  cash + uniform_price = 800 := 
by
  intros h_months h_cash h_uniform h_received h_year h_proportion
  sorry

end original_agreed_amount_l675_675008


namespace infinite_k_same_value_l675_675597

-- Define a predicate indicating that the function is completely multiplicative
def is_completely_multiplicative {G : Type*} [Group G] (f : ℕ+ → G) : Prop :=
  ∀ (m n : ℕ+), f(m * n) = f(m) * f(n)

theorem infinite_k_same_value {G : Type*} [Finite G] [Abelian G]
  (f : ℕ+ → G) (h_mul : is_completely_multiplicative f) :
  ∃∞ (k : ℕ+), f k = f (k + 1) :=
sorry

end infinite_k_same_value_l675_675597


namespace find_number_satisfying_condition_l675_675351

theorem find_number_satisfying_condition :
  ∃ x : ℝ, x ≥ 0 ∧ (sqrt x / 5 = 5) → x = 625 :=
by
  intro h
  cases h with x hx
  cases hx with hx1 hx2
  rw [sqrt_eq_iff_sq_eq, mul_self_div_self (eq.trans (eq.symm (zero_lt_iff.mpr zero_ne_one)))] at hx2
  apply hx2
  apply zero_le
  apply zero_lt_iff.mpr
  apply not_even_one
  sorry

end find_number_satisfying_condition_l675_675351


namespace rate_of_current_l675_675745

variable (c : ℝ)
def effective_speed_downstream (c : ℝ) : ℝ := 4.5 + c
def effective_speed_upstream (c : ℝ) : ℝ := 4.5 - c

theorem rate_of_current
  (h1 : ∀ d : ℝ, d / (4.5 - c) = 2 * (d / (4.5 + c)))
  : c = 1.5 :=
by
  sorry

end rate_of_current_l675_675745


namespace slope_of_line_l675_675575

open Real

-- Given parametric equations for curve C
def C (θ : ℝ) : ℝ × ℝ :=
  (2 * cos θ, 4 * sin θ)

-- Given parametric equations for line l
def l (t α : ℝ) : ℝ × ℝ :=
  (1 + t * cos α, 2 + t * sin α)

-- Condition: midpoint of the intersection segment is (1, 2)
def is_midpoint_of_intersection (α : ℝ) : Prop :=
  let t1 := -2 / cos α
  let t2 := 2 / cos α
  1 = (1 + (t1 + t2) * cos α / 2) ∧
  2 = (2 + (t1 + t2) * sin α / 2)

-- Main theorem to be proven
theorem slope_of_line (α : ℝ) (h : is_midpoint_of_intersection α) : tan α = -2 :=
  sorry

end slope_of_line_l675_675575


namespace polynomial_has_real_root_l675_675179

theorem polynomial_has_real_root
  (a b c d e : ℝ)
  (h : ∃ r : ℝ, ax^2 + (c - b)x + (e - d) = 0 ∧ r > 1) :
  ∃ x : ℝ, ax^4 + bx^3 + cx^2 + dx + e = 0 :=
by
  sorry

end polynomial_has_real_root_l675_675179


namespace continuous_sum_m_l675_675243

noncomputable def g : ℝ → ℝ → ℝ
| x, m => if x < m then x^2 + 4 else 3 * x + 6

theorem continuous_sum_m :
  ∀ m1 m2 : ℝ, (∀ m : ℝ, (g m m1 = g m m2) → g m (m1 + m2) = g m m1 + g m m2) →
  m1 + m2 = 3 :=
sorry

end continuous_sum_m_l675_675243


namespace corresponding_angles_equal_l675_675270

-- Define what it means for two angles to be corresponding angles
def corresponding_angles (a b : ℝ) : Prop :=
  -- Hypothetical definition
  sorry

-- Lean 4 statement of the problem
theorem corresponding_angles_equal (a b : ℝ) (h : corresponding_angles a b) : a = b :=
by
  sorry

end corresponding_angles_equal_l675_675270


namespace chef_bought_kilograms_of_almonds_l675_675765

def total_weight_of_nuts : ℝ := 0.52
def weight_of_pecans : ℝ := 0.38
def weight_of_almonds : ℝ := total_weight_of_nuts - weight_of_pecans

theorem chef_bought_kilograms_of_almonds : weight_of_almonds = 0.14 := by
  sorry

end chef_bought_kilograms_of_almonds_l675_675765


namespace newspaper_pages_l675_675165

theorem newspaper_pages (p : ℕ) (h₁ : p >= 21) (h₂ : 8•2 - 1 ≤ p) (h₃ : p ≤ 8•3) : p = 28 :=
sorry

end newspaper_pages_l675_675165


namespace total_pastries_correct_l675_675252

def lola_minicupcakes : ℕ := 13
def lola_poptarts : ℕ := 10
def lola_blueberrypies : ℕ := 8

def lulu_minicupcakes : ℕ := 16
def lulu_poptarts : ℕ := 12
def lulu_blueberrypies : ℕ := 14

def total_pastries : ℕ :=
  lola_minicupcakes + lulu_minicupcakes +
  lola_poptarts + lulu_poptarts +
  lola_blueberrypies + lulu_blueberrypies

theorem total_pastries_correct : total_pastries = 73 := by
  calc
    total_pastries
        = (lola_minicupcakes + lulu_minicupcakes +
           lola_poptarts + lulu_poptarts +
           lola_blueberrypies + lulu_blueberrypies) : rfl
    ... = (13 + 16) + (10 + 12) + (8 + 14)             : by rfl
    ... = 29 + 22 + 22                                 : by rfl
    ... = 73                                           : by rfl

#eval total_pastries

end total_pastries_correct_l675_675252


namespace probability_of_special_number_l675_675011

open Set

/-- Define the set of numbers from 40 to 999 -/
def S : Set ℕ := {n | 40 ≤ n ∧ n ≤ 999}

/-- Define what it means for a number to be either less than 60 or a multiple of 10 -/
def is_special (n : ℕ) : Prop := n < 60 ∨ n % 10 = 0

/-- Define the event of selecting a special number from the set S -/
def special_event : Set ℕ := {n ∈ S | is_special n}

/-- Define the total number of elements in S -/
def total_elements : ℕ := S.toFinset.card

/-- Define the number of special elements in S -/
def special_elements : ℕ := special_event.toFinset.card

/-- Define the probability as a rational number -/
def probability : ℚ := special_elements / total_elements

/-- The probability of selecting a special number from set S is 19/160 -/
theorem probability_of_special_number : probability = 19 / 160 := 
by sorry

end probability_of_special_number_l675_675011


namespace train_speed_l675_675747

theorem train_speed (length_of_train : ℝ) (time_to_cross : ℝ) (speed : ℝ) 
  (h1 : length_of_train = 200) (h2 : time_to_cross = 5) : speed = 40 :=
by
  have h3 : speed = length_of_train / time_to_cross := sorry
  rw [h1, h2] at h3
  exact h3

end train_speed_l675_675747


namespace isosceles_triangle_of_equal_bisectors_l675_675586

variable {α β : ℝ}
variables {a b c : ℝ}
variables {A B C : Type}

-- Conditions from the problem:
def bisector_length_a (b c : ℝ) (α : ℝ) : ℝ :=
  (2 * b * c / (b + c)) * Real.cos (α / 2)

def bisector_length_b (a c : ℝ) (β : ℝ) : ℝ :=
  (2 * a * c / (a + c)) * Real.cos (β / 2)

theorem isosceles_triangle_of_equal_bisectors
  (a b c : ℝ) (α β : ℝ)
  (ha_gt_hb : a > b)
  (h_cos_ineq : Real.cos (α / 2) > Real.cos (β / 2))
  (h_bisectors_equal : bisector_length_a b c α = bisector_length_b a c β) :
  a = b :=
by
  sorry

end isosceles_triangle_of_equal_bisectors_l675_675586


namespace remainder_of_M_l675_675607

def T : ℕ → ℕ := λ n, sorry -- (Function to generate the increasing sequence whose binary representation has exactly 9 ones)

def M : ℕ := T 1500

theorem remainder_of_M (hM: M = 33023) : M % 1500 = 23 :=
by {
  have : M = 33023 := hM,
  rw this,
  norm_num,
}

end remainder_of_M_l675_675607


namespace area_of_triangle_FYG_l675_675337

theorem area_of_triangle_FYG (EF GH : ℝ) (area_trap : ℝ) (h : ℝ) : 
    (EF = 24) → (GH = 40) → (area_trap = 384) → 
    (h = 12) → (∃ FYG, FYG = 76.8 ∧ triangle_area FYG) := 
by
  sorry

end area_of_triangle_FYG_l675_675337


namespace arcsin_sine_2000_deg_eq_neg20_l675_675809

-- Define the necessary angles and properties
def principal_range (x : ℝ) : Prop := -90 ≤ x ∧ x ≤ 90

def coterminal (x y : ℝ) : Prop := ∃ (k : ℤ), y = x + 360 * k

def decompose (x : ℝ) : ℝ := x - 360 * ⌊x / 360⌋

-- Lean statement to prove
theorem arcsin_sine_2000_deg_eq_neg20 :
  ∀ (x : ℝ), decompose x = 2000 → principal_range (arcsin (sin x)) → arcsin (sin 2000) = -20 :=
by
  intro x hx hp
  sorry

end arcsin_sine_2000_deg_eq_neg20_l675_675809


namespace notebooks_to_achieve_goal_l675_675776

noncomputable def cost_price_per_notebook : ℝ := 15 / 4
noncomputable def selling_price_per_notebook : ℝ := 25 / 6
noncomputable def profit_per_notebook : ℝ := selling_price_per_notebook - cost_price_per_notebook
noncomputable def desired_profit : ℝ := 40
noncomputable def notebooks_needed_to_achieve_goal : ℝ := desired_profit / profit_per_notebook

theorem notebooks_to_achieve_goal :
  notebooks_needed_to_achieve_goal ≈ 96 := 
begin
  sorry
end

end notebooks_to_achieve_goal_l675_675776


namespace quadrilateral_parallelogram_l675_675950

open Function

noncomputable theory

variables {P : Type} [AffinePlane P]
variables {A B C D E F G H O : P}

theorem quadrilateral_parallelogram
  (ABCD_convex : ConvexQuadrilateral A B C D)
  (E_mid_AB : Midpoint E A B)
  (F_mid_BC : Midpoint F B C)
  (G_mid_CD : Midpoint G C D)
  (H_mid_DA : Midpoint H D A)
  (conc_AC_BD_EG_FH : Concurrent (LineThrough A C) (LineThrough B D) (LineThrough E G) (LineThrough F H) O)
  : Parallelogram A B C D :=
sorry

end quadrilateral_parallelogram_l675_675950


namespace coefficient_of_x4_in_product_l675_675343

-- Define the polynomials f(x) and g(x)
def f (x : ℚ) := 3 * x^5 - 4 * x^4 + 2 * x^3 + x - 2
def g (x : ℚ) := 2 * x^3 + x^2 - 3

-- Statement of the proof problem in Lean 4
theorem coefficient_of_x4_in_product :
  (polynomial.coeff ((f : polynomial ℚ) * (g : polynomial ℚ)) 4) = 12 :=
sorry

end coefficient_of_x4_in_product_l675_675343


namespace cos_600_eq_neg_one_half_l675_675810

theorem cos_600_eq_neg_one_half : real.cos (600 * real.pi / 180) = -1 / 2 := 
by sorry

end cos_600_eq_neg_one_half_l675_675810


namespace triangle_ABC_B_120_AC_sqrt19_AB_2_BC_3_l675_675553

theorem triangle_ABC_B_120_AC_sqrt19_AB_2_BC_3 :
  ∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C], (B = 120) → 
  (dist A C = sqrt 19) → (dist A B = 2) → (dist B C = 3).
sorry

end triangle_ABC_B_120_AC_sqrt19_AB_2_BC_3_l675_675553


namespace fg_of_3_eq_83_l675_675174

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 3 * x + 2

theorem fg_of_3_eq_83 : f (g 3) = 83 := by
  sorry

end fg_of_3_eq_83_l675_675174


namespace vector_addition_mn_l675_675509

theorem vector_addition_mn :
  ∀ (m n : ℤ), 
  (∀ (x y : ℤ × ℤ), (x = (m, 5) ∧ y = (4, n) ∧ (4 - m, n - 5) = (7, 6))) → 
  m + n = 8 :=
by
  intros m n h
  cases h with ab h
  cases h with ac bc_eq
  cases ab with ab_x ab_y
  cases ac with ac_x ac_y
  cases bc_eq with bc_x bc_y
  sorry

end vector_addition_mn_l675_675509


namespace range_of_a_extremum_points_l675_675502

noncomputable def f (a x : ℝ) := x * Real.log x - a * Real.exp x

theorem range_of_a_extremum_points :
  ∀ a, (0 < a ∧ a < 1 / Real.exp 1) →
    ∃ x1 x2, x1 ≠ x2 ∧ (Real.deriv (f a)) x1 = 0 ∧ (Real.deriv (f a)) x2 = 0 :=
by
  intros
  sorry

end range_of_a_extremum_points_l675_675502


namespace triangle_side_length_l675_675542

theorem triangle_side_length (A B C : Type) [euclidean_geometry A B C]
  (angle_B : B = 120 * π / 180)
  (AC : ℝ) (AB : ℝ) 
  (h₁ : AC = sqrt 19)
  (h₂ : AB = 2) :
  ∃ BC, BC = 3 :=
begin
  -- proof to be filled in
  sorry
end

end triangle_side_length_l675_675542


namespace distance_from_focus_l675_675912

theorem distance_from_focus (x y : ℝ) (h_parabola : y^2 = 8 * x) (h_x : x = 1) : 
  let F : ℝ × ℝ := (2, 0)
  in dist (x, y) F = 3 :=
by
  sorry

end distance_from_focus_l675_675912


namespace parallel_transitive_l675_675646

-- Definitions of the conditions
variables {α : Type*} [EuclideanGeometry α] -- Assume a type α with Euclidean geometry

/-- A theorem stating that if two lines are parallel to a third line, they are parallel to each other. -/
theorem parallel_transitive (a b c : Line α) (h_ac : Parallel a c) (h_bc : Parallel b c) : Parallel a b :=
by
  sorry

end parallel_transitive_l675_675646


namespace rectangle_equation_l675_675579

-- Given points in the problem, we define the coordinates
def A : ℝ × ℝ := (5, 5)
def B : ℝ × ℝ := (9, 2)
def C (a : ℝ) : ℝ × ℝ := (a, 13)
def D (b : ℝ) : ℝ × ℝ := (15, b)

-- We need to prove that a - b = 1 given the conditions
theorem rectangle_equation (a b : ℝ) (h1 : C a = (a, 13)) (h2 : D b = (15, b)) (h3 : 15 - a = 4) (h4 : 13 - b = 3) : 
     a - b = 1 := 
sorry

end rectangle_equation_l675_675579


namespace tank_holds_21_liters_l675_675781

def tank_capacity (S L : ℝ) : Prop :=
  (L = 2 * S + 3) ∧
  (L = 4) ∧
  (2 * S + 5 * L = 21)

theorem tank_holds_21_liters :
  ∃ S L : ℝ, tank_capacity S L :=
by
  use 1/2, 4
  unfold tank_capacity
  simp
  sorry

end tank_holds_21_liters_l675_675781


namespace angle_ordering_l675_675595

variable {α : Type*} [LinearOrder α] [Field α] [Real α]

variables (a b c u : α)

def triangle_ABC (angle_A angle_B angle_C : α) :=
  ∃ (BC CA AB : α), a = BC ∧ b = CA ∧ c = AB ∧
  ∠A = angle_A ∧ ∠B = angle_B ∧ ∠C = angle_C

def triangle_DEF (angle_D angle_E angle_F : α) :=
  ∃ (EF FD DE : α), sqrt a * sqrt u = EF ∧ sqrt b * sqrt u = FD ∧ sqrt c * sqrt u = DE ∧
  ∠D = angle_D ∧ ∠E = angle_E ∧ ∠F = angle_F

theorem angle_ordering
  (h1: ∠A > ∠B) 
  (h2: ∠B > ∠C) 
  (h3: triangle_ABC ∠A ∠B ∠C)
  (h4: triangle_DEF ∠D ∠E ∠F):
  ∠A > ∠D ∧ ∠D > ∠E ∧ ∠E > ∠F ∧ ∠F > ∠C :=
sorry

end angle_ordering_l675_675595


namespace part1_B_value_part2_max_value_l675_675203

theorem part1_B_value (A B C a b c: ℝ) (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) (h3: 0 < C ∧ C < π) (h4 : A + B + C = π) 
(h5 : b = 2 * sqrt 3) (h6 : 2 * a - c = 2 * b * cos C) : 
B = π / 3 := 
sorry

theorem part2_max_value (A B C a b c: ℝ) (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) (h3: 0 < C ∧ C < π) (h4 : A + B + C = π) 
(h5 : b = 2 * sqrt 3) (h6 : 2 * a - c = 2 * b * cos C) : 
∃ x, (3 * a + 2 * c = x) ∧ x ≤ 4 * sqrt 19 := 
sorry

end part1_B_value_part2_max_value_l675_675203


namespace finite_ring_identity_l675_675949

variable {A : Type} [Ring A] [Fintype A]
variables (a b : A)

theorem finite_ring_identity (h : (ab - 1) * b = 0) : b * (ab - 1) = 0 :=
sorry

end finite_ring_identity_l675_675949


namespace evaluate_expression_l675_675443

theorem evaluate_expression (a : ℚ) (h : a = 4 / 3) : 
  (6 * a ^ 2 - 15 * a + 5) * (3 * a - 4) = 0 := by
  sorry

end evaluate_expression_l675_675443


namespace find_m_l675_675622

def numFactorsOf2 (k : ℕ) : ℕ :=
  k / 2 + k / 4 + k / 8 + k / 16 + k / 32 + k / 64 + k / 128 + k / 256

theorem find_m : ∃ m : ℕ, m > 1990 ^ 1990 ∧ m = 3 ^ 1990 + numFactorsOf2 m :=
by
  sorry

end find_m_l675_675622


namespace hyperbola_properties_l675_675469

theorem hyperbola_properties
  (a b : ℝ) (h1 : a^2 = 25) (h2 : b^2 = 9)
  (sum_eccentricities : ℝ) (h_sum : sum_eccentricities = 14 / 5)
  (focus_distance : ℝ) (h_focus : focus_distance = 4) :
  let e_ellipse := sqrt (h_focus^2 / a^2),
      e_hyperbola := sum_eccentricities - e_ellipse,
      m := a / e_hyperbola,
      n := sqrt (focus_distance^2 - m^2) in
  e_hyperbola = 2 ∧ (m = 2) ∧ (n^2 = 12) ∧ (∀ x y : ℝ, (x^2 / m^2 - y^2 / n^2 = 1) ↔ (x^2 / 4 - y^2 / 12 = 1)) :=
by
  sorry

end hyperbola_properties_l675_675469


namespace rose_paid_147_6_l675_675648

-- Define the conditions
def marked_price : ℝ := 180
def discount_percentage : ℝ := 18

-- Function to calculate the discount amount
def discount_amount (marked_price discount_percentage : ℝ) : ℝ :=
  (discount_percentage / 100) * marked_price

-- Function to calculate the final price after discount
def price_after_discount (marked_price discount_amount : ℝ) : ℝ :=
  marked_price - discount_amount

-- The theorem to prove
theorem rose_paid_147_6 :
  price_after_discount marked_price (discount_amount marked_price discount_percentage) = 147.6 :=
by
  sorry

end rose_paid_147_6_l675_675648


namespace single_elimination_games_needed_l675_675927

theorem single_elimination_games_needed (n : ℕ) (n_pos : n > 0) :
  (number_of_games_needed : ℕ) = n - 1 :=
by
  sorry

end single_elimination_games_needed_l675_675927


namespace area_of_yard_l675_675848

theorem area_of_yard (L W : ℕ) (h1 : L = 40) (h2 : L + 2 * W = 64) : L * W = 480 := by
  sorry

end area_of_yard_l675_675848


namespace hexagon_area_half_triangle_area_l675_675983

theorem hexagon_area_half_triangle_area
  {A B C A_1 B_1 C_1 P Q R O : Point} 
  (hABC_acute : triangle_acute ABC)
  (hA1_midpoint : midpoint A_1 B C)
  (hB1_midpoint : midpoint B_1 C A)
  (hC1_midpoint : midpoint C_1 A B)
  (hPerpendicular_A1 : perpendicular A_1 OppSide1)
  (hPerpendicular_B1 : perpendicular B_1 OppSide2)
  (hPerpendicular_C1 : perpendicular C_1 OppSide3) :
  area (hexagon A_1 B_1 C_1 P Q R) = (1 / 2) * area (triangle ABC) :=
begin
  sorry
end

end hexagon_area_half_triangle_area_l675_675983


namespace probability_of_40_cents_l675_675994

noncomputable def num_successful_outcomes : ℕ := 16 + 3

def total_outcomes : ℕ := 2 ^ 5

def probability_success : ℚ := num_successful_outcomes / total_outcomes

theorem probability_of_40_cents : probability_success = 19 / 32 := by
  unfold probability_success num_successful_outcomes total_outcomes
  norm_num
  sorry

end probability_of_40_cents_l675_675994


namespace tangent_line_at_x1_g_extremum_range_max_lambda_l675_675467

noncomputable def f (x λ : ℝ) : ℝ := exp x - exp 1 * x - λ * (x * log x - x + 1)
noncomputable def g (x λ : ℝ) : ℝ := (deriv (λ x, exp x - (exp 1 : ℝ) * x - λ * (x * log x - x + 1))) x

theorem tangent_line_at_x1 (λ : ℝ) : (f 1 λ) = 0 ∧ (deriv (λ x, f x λ)) 1 = 0 := 
sorry

theorem g_extremum_range (λ : ℝ) : (∃ x > 0, g x λ = 0) ↔ (0 < λ) := 
sorry

theorem max_lambda (λ : ℝ) : (∀ x ≥ 1, f x λ ≥ 0) ↔ (λ ≤ exp 1) := 
sorry

end tangent_line_at_x1_g_extremum_range_max_lambda_l675_675467


namespace positional_relationship_l675_675488

open_locale classical

variables {α : Type*} [ordered_field α]

-- Definitions for the problem conditions

def is_parallel_to_plane {α : Type*} [add_comm_group α] [vector_space α (euclidean_space)] (l : affine_subspace α) (p : affine_subspace α) : Prop := sorry
def is_within_plane {α : Type*} [add_comm_group α] [vector_space α (euclidean_space)] (l : affine_subspace α) (p : affine_subspace α) : Prop := sorry
def are_parallel {α : Type*} [add_comm_group α] [vector_space α (euclidean_space)] (l₁ l₂ : affine_subspace α) : Prop := sorry
def are_skew {α : Type*} [add_comm_group α] [vector_space α (euclidean_space)] (l₁ l₂ : affine_subspace α) : Prop := sorry

-- Given conditions
variables (a b : affine_subspace α) (α' : affine_subspace α)

-- Theorem statement
theorem positional_relationship (h1 : is_parallel_to_plane a α') (h2 : is_within_plane b α') : are_parallel a b ∨ are_skew a b :=
sorry

end positional_relationship_l675_675488


namespace condition_for_equation_l675_675676

theorem condition_for_equation (x : ℝ) : 
  (sqrt((3 - x) / (x - 1)) = sqrt(3 - x) / sqrt(x - 1)) ↔ (1 < x ∧ x ≤ 3) :=
by
  sorry

end condition_for_equation_l675_675676


namespace problem_solution_l675_675953

noncomputable theory

open BigOperators

def P := Nat.lcm ⟨20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40⟩

def Q := Nat.lcm (P :: [41, 42, 43, 44, 45, 46, 47, 48, 50])

theorem problem_solution :
  Q / P = 82821 :=
by
  -- proof goes here
  sorry

end problem_solution_l675_675953


namespace range_of_m_l675_675245

def prop_p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 + y^2 - 2*x - 4*y + m = 0
def prop_q (m : ℝ) : Prop := ∃ (x y : ℝ), (x^2) / (m-6) - (y^2) / (m+3) = 1

theorem range_of_m (m : ℝ) : ¬ (prop_p m ∧ prop_q m) → m ≥ -3 :=
sorry

end range_of_m_l675_675245


namespace solve_for_x_l675_675661

-- Defining the problem conditions
def condition (x : ℝ) : Prop :=
  log 3 x + log (3^2) x = 5

-- The proof problem statement
theorem solve_for_x (x : ℝ) (h : condition x) : x = 3^(10/3) :=
by
sory

end solve_for_x_l675_675661


namespace axis_of_symmetry_of_shifted_function_l675_675989

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem axis_of_symmetry_of_shifted_function :
  (∃ x : ℝ, g x = 1 ∧ x = Real.pi / 12) :=
by
  sorry

end axis_of_symmetry_of_shifted_function_l675_675989


namespace sin_tan_identity_of_cos_eq_tan_identity_l675_675168

open Real

variable (α : ℝ)
variable (hα : α ∈ Ioo 0 π)   -- α is in the interval (0, π)
variable (hcos : cos (2 * α) = 2 * cos (α + π / 4))

theorem sin_tan_identity_of_cos_eq_tan_identity : 
  sin (2 * α) = 1 ∧ tan α = 1 :=
by
  sorry

end sin_tan_identity_of_cos_eq_tan_identity_l675_675168


namespace design_is_stable_l675_675296

-- Definitions based on conditions
def is_triangular (structure : Type) : Prop := 
  structure = 'triangular structure'

-- Definitions based on the question
def utilizes_stability_of_triangles (design : Type) : Prop := 
  design = 'stable design'

-- The mathematically equivalent proof problem
theorem design_is_stable
  (crane_base : Type) (powerline_support : Type) (bicycle_bracket : Type)
  (h_crane : is_triangular crane_base)
  (h_powerline : is_triangular powerline_support)
  (h_bicycle : is_triangular bicycle_bracket) : 
  utilizes_stability_of_triangles crane_base ∧ 
  utilizes_stability_of_triangles powerline_support ∧ 
  utilizes_stability_of_triangles bicycle_bracket :=
sorry

end design_is_stable_l675_675296


namespace line_intersection_points_l675_675415

def line_intersects_axes (x y : ℝ) : Prop :=
  (4 * y - 5 * x = 20)

theorem line_intersection_points :
  ∃ p1 p2, line_intersects_axes p1.1 p1.2 ∧ line_intersects_axes p2.1 p2.2 ∧
    (p1 = (-4, 0) ∧ p2 = (0, 5)) :=
by
  sorry

end line_intersection_points_l675_675415


namespace trigonometric_identity_l675_675057

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180))
  = (4 * Real.sin (10 * Real.pi / 180) / Real.sin (40 * Real.pi / 180)) :=
by sorry

end trigonometric_identity_l675_675057


namespace speed_of_current_l675_675702

-- Definitions
def speed_boat_still_water := 60
def speed_downstream := 77
def speed_upstream := 43

-- Theorem statement
theorem speed_of_current : ∃ x, speed_boat_still_water + x = speed_downstream ∧ speed_boat_still_water - x = speed_upstream ∧ x = 17 :=
by
  unfold speed_boat_still_water speed_downstream speed_upstream
  sorry

end speed_of_current_l675_675702


namespace fraction_of_subsets_l675_675954

theorem fraction_of_subsets (S T : ℕ) (hS : S = 2^10) (hT : T = Nat.choose 10 3) :
    (T:ℚ) / (S:ℚ) = 15 / 128 :=
by sorry

end fraction_of_subsets_l675_675954


namespace exists_x_satisfies_primary_but_not_secondary_l675_675085

theorem exists_x_satisfies_primary_but_not_secondary :
  ∃ (x : ℝ), (3 * x + 7 = 22) ∧ (¬ (2 * x + 1 = 9)) :=
by
  use 5  -- Declare x to be 5
  split  -- We want to prove two statements: primary condition and negation of secondary condition
  {
    -- Prove the primary condition
    calc
      3 * 5 + 7 = 15 + 7 : by ring
             ... = 22    : by ring
  }
  {
    -- Prove the negation of the secondary condition
    intro h
    have : 2 * 5 + 1 = 11 := by ring
    contradiction
  }

end exists_x_satisfies_primary_but_not_secondary_l675_675085


namespace brian_stones_l675_675045

variable (W B : ℕ)
variable (total_stones : ℕ := 100)
variable (G : ℕ := 40)
variable (Gr : ℕ := 60)

theorem brian_stones :
  (W > B) →
  ((W + B = total_stones) ∧ (G + Gr = total_stones) ∧ (W = 60)) :=
by
  sorry

end brian_stones_l675_675045


namespace max_sum_table_l675_675079

theorem max_sum_table (m n : ℕ) (T : matrix (fin (mn + 1)) (fin (mn + 1)) ℝ)
  (hT1 : ∀ i j, T i j ∈ set.Icc (0 : ℝ) 1)
  (hT2 : ∀ i j, (∀ (a b : fin n), T ((i * n) + a) ((j * n) + b)) = n) :
  ∑ (i : fin (mn + 1)) (j : fin (mn + 1)), T i j ≤ (mn + 1) * (m + 1) :=
sorry

end max_sum_table_l675_675079


namespace find_particular_number_l675_675028

variable (x : ℝ)

theorem find_particular_number (h : 0.46 + x = 0.72) : x = 0.26 :=
sorry

end find_particular_number_l675_675028


namespace samantha_cost_crate_l675_675279

variable (cost_per_egg : ℝ) (total_eggs remaining_eggs sold_eggs : ℕ)

-- Conditions
def total_eggs := 30
def remaining_eggs := 5
def sold_eggs := total_eggs - remaining_eggs
def cost_per_egg := 0.20

-- Prove that the cost of the crate of eggs is 5 dollars
theorem samantha_cost_crate : (sold_eggs : ℝ) * cost_per_egg = 5 := 
by 
  sorry

end samantha_cost_crate_l675_675279


namespace total_instruments_l675_675813

-- Definitions of the quantities of musical instruments owned by each individual according to the conditions.
def Charlie_flutes := 1
def Charlie_horns := 2
def Charlie_harps := 1
def Charlie_drums := 1  -- 1 set of drums
def Charlie_drum_pieces := 5

def Carli_flutes := 2 * Charlie_flutes
def Carli_horns := 1 / 2 * Charlie_horns
def Carli_harps := 0
def Carli_drums := 3 * Charlie_drums
def Carli_drum_pieces := 3 * Charlie_drum_pieces

def Nick_flutes := Charlie_flutes + Carli_flutes
def Nick_horns := Charlie_horns - Carli_horns
def Nick_harps := 0
def Nick_drums := 4 * (Charlie_drums + Carli_drums)
def Nick_drum_pieces := 4 * (Charlie_drum_pieces + Carli_drum_pieces)

-- Total number of instruments owned by each individual.
def total_Charlie := Charlie_flutes + Charlie_horns + Charlie_harps + Charlie_drums
def total_Carli := Carli_flutes + Carli_horns + Carli_harps + Carli_drums
def total_Nick := Nick_flutes + Nick_horns + Nick_harps + Nick_drums

-- Prove the total number of instruments owned by Charlie, Carli, and Nick is 19.
theorem total_instruments : total_Charlie + total_Carli + total_Nick = 19 :=
by
  -- Charlie's total instruments: 5
  have h_charlie : total_Charlie = 5 := by sorry
  -- Carli's total instruments: 6
  have h_carli : total_Carli = 6 := by sorry
  -- Nick's total instruments: 8
  have h_nick : total_Nick = 8 := by sorry
  -- Summing the totals
  calc
    total_Charlie + total_Carli + total_Nick
    = 5 + 6 + 8 : by rw [h_charlie, h_carli, h_nick]
    = 19 : by norm_num

end total_instruments_l675_675813


namespace solve_for_A_l675_675172

def diamond (A B : ℝ) := 4 * A + 3 * B + 7

theorem solve_for_A : diamond A 5 = 71 → A = 12.25 := by
  intro h
  unfold diamond at h
  sorry

end solve_for_A_l675_675172


namespace jakes_friend_purchases_l675_675210

-- Definitions for given conditions
def feeding_allowance : Real := 4
def fraction_given : Real := 1 / 4
def price_per_candy : Real := 0.20
def discount : Real := 0.15
def exchange_rate : Real := 0.85

-- Derived conditions
def amount_given := feeding_allowance * fraction_given
def discounted_price_per_candy := price_per_candy * (1 - discount)
def candies_bought := floor (amount_given / discounted_price_per_candy)
def amount_in_euros := amount_given * exchange_rate

-- Theorem statement
theorem jakes_friend_purchases :
    candies_bought = 5 ∧ amount_in_euros = 0.85 := by 
    sorry

end jakes_friend_purchases_l675_675210


namespace smallest_n_sqrt_inequality_l675_675734

theorem smallest_n_sqrt_inequality : ∃ n : ℕ, ∀ k : ℕ, k < n ↔ k < 10001 ∧ (sqrt (n : ℝ) - sqrt (n - 1 : ℝ) < 0.005) := 
  sorry

end smallest_n_sqrt_inequality_l675_675734


namespace sum_of_numbers_facing_up_is_4_probability_l675_675177

-- Definition of a uniform dice with faces numbered 1 to 6
def dice_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Definition of the sample space when the dice is thrown twice
def sample_space : Finset (ℕ × ℕ) := Finset.product dice_faces dice_faces

-- Definition of the event where the sum of the numbers is 4
def event_sum_4 : Finset (ℕ × ℕ) := sample_space.filter (fun pair => pair.1 + pair.2 = 4)

-- The number of favorable outcomes
def favorable_outcomes : ℕ := event_sum_4.card

-- The total number of possible outcomes
def total_outcomes : ℕ := sample_space.card

-- The probability of the event
def probability_event_sum_4 : ℚ := favorable_outcomes / total_outcomes

theorem sum_of_numbers_facing_up_is_4_probability :
  probability_event_sum_4 = 1 / 12 :=
by
  sorry

end sum_of_numbers_facing_up_is_4_probability_l675_675177


namespace inheritance_amount_l675_675219

-- Definitions based on conditions given
def inheritance (x : ℝ) : Prop :=
  let federal_tax := 0.25 * x
  let remaining_after_federal := x - federal_tax
  let state_tax := 0.15 * remaining_after_federal
  let total_tax := federal_tax + state_tax
  total_tax = 15000

-- The statement to be proven
theorem inheritance_amount (x : ℝ) (hx : inheritance x) : x = 41379 :=
by
  -- Proof goes here
  sorry

end inheritance_amount_l675_675219


namespace scientific_notation_of_0point0000025_l675_675932

theorem scientific_notation_of_0point0000025 : ∃ (a : ℝ) (n : ℤ), 0.0000025 = a * 10 ^ n ∧ a = 2.5 ∧ n = -6 :=
by {
  sorry
}

end scientific_notation_of_0point0000025_l675_675932


namespace hiring_probabilities_l675_675333

-- Define the candidates and their abilities
inductive Candidate : Type
| Strong
| Moderate
| Weak

open Candidate

-- Define the ordering rule and hiring rule
def interviewOrders : List (Candidate × Candidate × Candidate) :=
  [(Strong, Moderate, Weak), (Strong, Weak, Moderate), 
   (Moderate, Strong, Weak), (Moderate, Weak, Strong),
   (Weak, Strong, Moderate), (Weak, Moderate, Strong)]

def hiresStrong (order : Candidate × Candidate × Candidate) : Bool :=
  match order with
  | (Moderate, Strong, Weak) => true
  | (Moderate, Weak, Strong) => true
  | (Weak, Strong, Moderate) => true
  | _ => false

def hiresModerate (order : Candidate × Candidate × Candidate) : Bool :=
  match order with
  | (Strong, Weak, Moderate) => true
  | (Weak, Moderate, Strong) => true
  | _ => false

-- The main theorem to be proved
theorem hiring_probabilities :
  let orders := interviewOrders
  let p := (orders.filter hiresStrong).length / orders.length
  let q := (orders.filter hiresModerate).length / orders.length
  p = 1 / 2 ∧ q = 1 / 3 := by
  sorry

end hiring_probabilities_l675_675333


namespace condition_concatenation_of_rational_sides_l675_675770

theorem condition_concatenation_of_rational_sides {P : Finset (Finset ℚ)} :
  P.Card > 2 →
  (∀ s ∈ P, s ∈ (Finset.range P.Card → ℝ) → (∀ i j ∈ s, i ≠ j → rational (dist ℝ i j)) →
  ∀ s' ∈ (Finset.range (P.Card - 2) → ℝ), 
  (∀ i j ∈ s', i ≠ j → rational (dist ℝ i j))) := 
begin
  sorry,
end

end condition_concatenation_of_rational_sides_l675_675770


namespace num_boys_is_20_l675_675375

-- Definitions according to conditions
def total_people : ℕ := 31
def Agi_boys : ℕ := 7
def next_girl_boys (m : ℕ) : ℕ := m + 1
def Marta_boys (n : ℕ) : ℕ := n - 3

-- Translate to a mathematically equivalent proof problem
theorem num_boys_is_20 :
  ∃ n k : ℕ, 
    let Agi_boys := 7 in
    let Marta_boys := n - 3 in
    k = n - 9 ∧
    k + n = total_people ∧
    Marta_boys = k + 6 ∧
    n = 20 := 
by {
  sorry
}

end num_boys_is_20_l675_675375


namespace karl_net_receipt_l675_675978

-- Definitions of the conditions in the problem
def hans_initial_sales : ℕ := 25
def refund_to_return : ℕ := 5
def hans_spent_on_candies : ℕ := 3
def total_refund_given : ℕ := 2

-- Define the proof problem
theorem karl_net_receipt : hans_initial_sales - refund_to_return = 20 :=
by
  calc hans_initial_sales - refund_to_return
    = 25 - 5 : rfl
    ... = 20 : rfl

end karl_net_receipt_l675_675978


namespace yarn_length_proof_l675_675682

def green_length := 156
def total_length := 632

noncomputable def red_length (x : ℕ) := green_length * x + 8

theorem yarn_length_proof (x : ℕ) (green_length_eq : green_length = 156)
  (total_length_eq : green_length + red_length x = 632) : x = 3 :=
by {
  sorry
}

end yarn_length_proof_l675_675682


namespace find_d_l675_675242

theorem find_d (d x y : ℝ) (H1 : x - 2 * y = 5) (H2 : d * x + y = 6) (H3 : x > 0) (H4 : y > 0) :
  -1 / 2 < d ∧ d < 6 / 5 :=
by
  sorry

end find_d_l675_675242


namespace opposite_face_to_p_l675_675918

variable (faces : Type) [Fintype faces] [DecidableEq faces] 
variable (p q r t : faces)
variable (adjacent : faces → faces → Bool)

-- Define the adjacency relationships
def is_top (f : faces) : Bool := f = p
def is_right_adj_to (f : faces) : Bool := adjacent p q
def is_left_adj_to (f : faces) : Bool := adjacent p r

-- State the theorem
theorem opposite_face_to_p : is_top p ∧ is_right_adj_to q ∧ is_left_adj_to r → t ≠ q ∧ t ≠ r :=
by sorry

end opposite_face_to_p_l675_675918


namespace find_a_l675_675146

theorem find_a (f : ℤ → ℤ) (h1 : ∀ (x : ℤ), f (2 * x + 1) = 3 * x + 2) (h2 : f a = 2) : a = 1 := by
sorry

end find_a_l675_675146


namespace range_of_a_l675_675150

noncomputable def log_function (a : ℝ) (x : ℝ) : ℝ :=
  if a > 0 then real.logb a (3 * x^2 - 2 * a * x) else 0

def decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≥ f y

theorem range_of_a :
  (∀ x ∈ set.Icc (1/2 : ℝ) 1, decreasing_on_interval (log_function a) (1/2) 1) →
  0 < a ∧ a < (3 / 4) :=
by
  sorry

end range_of_a_l675_675150


namespace geometric_sequence_problem_l675_675200

noncomputable def a_n (q : ℝ) (a₁ : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

def b_n (a : ℕ → ℝ) (n : ℕ) : ℝ := Real.log2 (6 / a (2 * n + 1))

def C_n (b : ℕ → ℝ) (n : ℕ) : ℝ := 1 / (b n * b (n + 1))

theorem geometric_sequence_problem :
  let a := a_n (-1/2) 6 in
  let C (n : ℕ) := let b := b_n a in C_n b n in
  a 3 = 3/2 ∧ ∑ i in Finset.range 3, a i ≤ 9/2 →
  (∀ n, a n = 3/2 ∨ a n = 6 * (-1/2)^(n-1)) ∧ 
  ∀ n, ∑ i in Finset.range n, C i < 1/4 :=
begin
  intros a C h,
  sorry
end

end geometric_sequence_problem_l675_675200


namespace total_commission_l675_675417

-- Define the commission rate
def commission_rate : ℝ := 0.02

-- Define the sale prices of the three houses
def sale_price1 : ℝ := 157000
def sale_price2 : ℝ := 499000
def sale_price3 : ℝ := 125000

-- Total commission calculation
theorem total_commission :
  (commission_rate * sale_price1 + commission_rate * sale_price2 + commission_rate * sale_price3) = 15620 := 
by
  sorry

end total_commission_l675_675417


namespace brian_total_commission_l675_675420

theorem brian_total_commission :
  let commission_rate := 0.02
  let house1 := 157000
  let house2 := 499000
  let house3 := 125000
  let total_sales := house1 + house2 + house3
  let total_commission := total_sales * commission_rate
  total_commission = 15620 := by
{
  sorry
}

end brian_total_commission_l675_675420


namespace question_l675_675642

-- Let x and y be real numbers.
variables (x y : ℝ)

-- Proposition A: x + y ≠ 8
def PropA : Prop := x + y ≠ 8

-- Proposition B: x ≠ 2 ∨ y ≠ 6
def PropB : Prop := x ≠ 2 ∨ y ≠ 6

-- We need to prove that PropA is a sufficient but not necessary condition for PropB.
theorem question : (PropA x y → PropB x y) ∧ ¬ (PropB x y → PropA x y) :=
sorry

end question_l675_675642


namespace cannot_be_20182017_l675_675322

theorem cannot_be_20182017 (a b : ℤ) (h : a * b * (a + b) = 20182017) : False :=
by
  sorry

end cannot_be_20182017_l675_675322


namespace find_a_l675_675286

theorem find_a (f g : ℝ → ℝ) (a : ℝ) 
  (h₁ : ∀ x, f x = 2 * x / 3 + 4) 
  (h₂ : ∀ x, g x = 5 - 2 * x) 
  (h₃ : f (g a) = 7) : 
  a = 1 / 4 := 
sorry

end find_a_l675_675286


namespace brads_speed_l675_675631

theorem brads_speed
  (maxwell_speed : ℝ)
  (distance_between_homes : ℝ)
  (maxwell_distance_traveled : ℝ)
  (meeting_time_equal : Prop)
  (t : ℝ)
  (maxwell_time : t = maxwell_distance_traveled / maxwell_speed)
  (brad_time : t = (distance_between_homes - maxwell_distance_traveled) / (distance_between_homes / t - meeting_time_equal)) :
  (distance_between_homes - maxwell_distance_traveled) / t = 6 :=
by
  have no_use := meeting_time_equal
  sorry

end brads_speed_l675_675631


namespace circumference_of_shaded_area_l675_675928

-- Condition definitions given in a)
def side_length : ℝ := 2
def radius : ℝ := 2
def pi_value : ℝ := 3.141

-- The theorem statement proving the desired circumference of the shaded area
theorem circumference_of_shaded_area :
  let circle_circumference := 2 * pi_value * radius
  in circle_circumference / 3 = 4.188 := by
  sorry

end circumference_of_shaded_area_l675_675928


namespace angle_between_tangents_equal_l675_675026

-- Given Definitions
structure Quadrilateral :=
(A B C D : Point)

def passes_through (C : Circle) (P Q R : Point) : Prop := -- Definition of circle passing through points
  P ∈ C ∧ Q ∈ C ∧ R ∈ C

-- Hypotheses
axiom rhombus (ABCD : Quadrilateral) : Prop

-- Circles passing through specific points
axiom CA (A B C D : Point) : Circle
axiom CB (A B C D : Point) : Circle
axiom CC (A B C D : Point) : Circle
axiom CD (A B C D : Point) : Circle

-- Prove the required equality of angles
theorem angle_between_tangents_equal (A B C D : Point) (h1 : rhombus ⟨A, B, C, D⟩)
  (hCA : passes_through (CA A B C D) B C D)
  (hCB : passes_through (CB A B C D) A C D)
  (hCC : passes_through (CC A B C D) A B D)
  (hCD : passes_through (CD A B C D) A B C) :
  angle_between_tangents (CA A B C D) (CC A B C D) B = angle_between_tangents (CB A B C D) (CD A B C D) A :=
sorry

end angle_between_tangents_equal_l675_675026


namespace ratio_area_triangle_quadrilateral_l675_675124

section
  universe u
  variables {α : Type u} [RealInnerProductSpace α] (A B C D E F : α) (DA : ℝ) (DB : ℝ)
  
  def is_midpoint (M X Y : α) := 2 • M = X + Y 
  def on_segment (F A D : α) (r : ℝ) := (1 - r) • A + r • D = F

  theorem ratio_area_triangle_quadrilateral
    (h_square : ∃ (l : ℝ), ∀ {P Q : α}, P - Q = l ∨ P - Q = -l)
    (hE : is_midpoint E B D)
    (hF : on_segment F A D (1/3)) :
    (area (triangle E D F) / area (quadrilateral A B E F) = 1 / 5) :=
  by
    sorry
end

end ratio_area_triangle_quadrilateral_l675_675124


namespace red_bank_amount_l675_675714

-- Definitions of initial amounts and daily increments
def initial_red := 8000
def increment_red := 300
def initial_yellow := 5000
def increment_yellow := 500

-- Suppose d is the number of days when the amounts in both banks are equal
variable d : ℕ 
-- Definitions of the final amounts in the banks after d days
def red_amount := initial_red + d * increment_red
def yellow_amount := initial_yellow + d * increment_yellow

-- The equation when the amounts are equal
def equal_amounts := red_amount = yellow_amount

theorem red_bank_amount (h : equal_amounts) : red_amount = 12500 := by
  show red_amount = 12500 from
  -- using given conditions and solving
  sorry

end red_bank_amount_l675_675714


namespace each_group_has_two_bananas_l675_675295

theorem each_group_has_two_bananas (G T : ℕ) (hG : G = 196) (hT : T = 392) : T / G = 2 :=
by
  sorry

end each_group_has_two_bananas_l675_675295


namespace compute_trig_expr_l675_675054

theorem compute_trig_expr :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 2 * Real.sec (10 * Real.pi / 180) :=
by
  sorry

end compute_trig_expr_l675_675054


namespace corresponding_angles_equal_l675_675273

-- Definition: Corresponding angles and their equality
def corresponding_angles (α β : ℝ) : Prop :=
  -- assuming definition of corresponding angles can be defined
  sorry

theorem corresponding_angles_equal {α β : ℝ} (h : corresponding_angles α β) : α = β :=
by
  -- the proof is provided in the problem statement
  sorry

end corresponding_angles_equal_l675_675273


namespace smallest_polynomial_degree_rational_roots_l675_675291

theorem smallest_polynomial_degree_rational_roots :
  ∃ (p : Polynomial ℚ), 
    Polynomial.degree p = 8 ∧
    p ≠ 0 ∧
    (p.root 3 - sqrt 8) ∧
    (p.root 5 + sqrt 11) ∧
    (p.root 16 - 2 * sqrt 10) ∧
    (p.root (-sqrt 3)) :=
begin
  sorry
end

end smallest_polynomial_degree_rational_roots_l675_675291


namespace even_odd_function_value_l675_675137

theorem even_odd_function_value 
  (f g : ℝ → ℝ) 
  (h_even : ∀ x, f (-x) = f x)
  (h_odd : ∀ x, g (-x) = - g x)
  (h_eqn : ∀ x, f x + g x = x^3 + x^2 + 1) :
  f 1 - g 1 = 1 := 
by {
  sorry
}

end even_odd_function_value_l675_675137


namespace average_payment_l675_675744

-- Each condition from part a) is used as a definition here
variable (n : Nat) (p1 p2 first_payment remaining_payment : Nat)

-- Conditions given in natural language
def payments_every_year : Prop :=
  n = 52 ∧
  first_payment = 410 ∧
  remaining_payment = first_payment + 65 ∧
  p1 = 8 * first_payment ∧
  p2 = 44 * remaining_payment ∧
  p2 = 44 * (first_payment + 65) ∧
  p1 + p2 = 24180

-- The theorem to prove based on the conditions
theorem average_payment 
  (h : payments_every_year n p1 p2 first_payment remaining_payment) 
  : (p1 + p2) / n = 465 := 
sorry  -- Proof is omitted intentionally

end average_payment_l675_675744


namespace equal_segments_l675_675204

noncomputable theory

variables {A B C M N : Point}
variables {circumcircle : Circle}
variables {triangle : Triangle A B C}

-- Definitions based on conditions
def tangent_at_point (circle : Circle) (P : Point) : Line := sorry

def is_tangent (circle : Circle) (line : Line) (P : Point) : Prop := 
  line = tangent_at_point circle P

def intersects_at_point (l1 l2 : Line) (P : Point) : Prop :=
  l1 ≠ l2 ∧ P ∈ l1 ∧ P ∈ l2

def parallel (l1 l2 : Line) : Prop := sorry

def line_through_points (P Q : Point) : Line := sorry

def line_parallel_through_point (parallel_to : Line) (through : Point) : Line := sorry

-- Given conditions
def M_property (A B C M : Point) : Prop :=
  let circumcircle := circumcircle_through A B C in
  let tangent_B := tangent_at_point circumcircle B in
  let tangent_C := tangent_at_point circumcircle C in
  intersects_at_point tangent_B tangent_C M

def N_property (A B C M N : Point) : Prop :=
  let AB := line_through_points A B in
  let AC := line_through_points A C in
  let parallel_to_AB := line_parallel_through_point AB M in
  intersects_at_point parallel_to_AB AC N

-- Prove that AN = BN
theorem equal_segments (A B C M N : Point) 
  (h1 : M_property A B C M)
  (h2 : N_property A B C M N) : 
  segment_length A N = segment_length B N :=
sorry

end equal_segments_l675_675204


namespace mod_of_1500th_number_with_9_ones_l675_675600

theorem mod_of_1500th_number_with_9_ones :
  let T := {n : ℕ | nat.popcount n = 9} in
  let M := (T.to_list.nth 1500).get_or_else 0 in
  M % 1500 = ?answer :=
by sorry

end mod_of_1500th_number_with_9_ones_l675_675600


namespace literature_books_cost_more_l675_675695

theorem literature_books_cost_more :
  let num_books := 45
  let literature_cost_per_book := 7
  let technology_cost_per_book := 5
  (num_books * literature_cost_per_book) - (num_books * technology_cost_per_book) = 90 :=
by
  sorry

end literature_books_cost_more_l675_675695


namespace range_f_example_increasing_f_example_l675_675154

-- Definition of the function f(x) with general 'a' condition
def f (a x : ℝ) : ℝ := log (2⁻¹) (a^x - 3)

noncomputable def range_f_on_interval (x : ℝ) : set ℝ := 
  {f 2 x | 2 < x}

theorem range_f_example : range_f_on_interval ∈ (-∞, 0) :=
by sorry

noncomputable def increasing_f_interval (a x : ℝ) : Prop :=
  0 < a ∧ a ≤ (sqrt 3 / 3) → 
  ∀ x1 x2, x1 < x2 → x1 < -2 ∧ x2 < -2 → f a x1 < f a x2

theorem increasing_f_example : increasing_f_interval :=
by sorry

end range_f_example_increasing_f_example_l675_675154


namespace probability_all_evens_before_any_odd_l675_675774

/-- The probability of rolling every even number (2, 4, 6, and 8) at least once before the first
  occurrence of an odd number (1, 3, 5, 7) when rolling a fair 8-sided die is 1 / 210. -/
theorem probability_all_evens_before_any_odd : 
  (∑ n in set.Ico 5 (set.Ioo (nat.pos 0) ∞), 
    (1 / (2 ^ n)) * (4 * ((1 / 4) ^ (n - 1)) - 6 * ((1 / 2) ^ (n - 1)) + 4 * ((3 / 4) ^ (n - 1)) - 1)) = (1 / 210) :=
sorry

end probability_all_evens_before_any_odd_l675_675774


namespace Cheryl_needed_square_yards_first_type_l675_675304

noncomputable def square_yards_first_type (x : ℝ) : Prop :=
  let second_type_material : ℝ := 1 / 8
  let unused_material : ℝ := 4 / 18
  let used_material : ℝ := 0.125
  let total_material_bought : ℝ := used_material + unused_material in
  x = total_material_bought - second_type_material

theorem Cheryl_needed_square_yards_first_type :
  ∃ x, square_yards_first_type x ∧ x = 0.2222 :=
by
  sorry

end Cheryl_needed_square_yards_first_type_l675_675304


namespace total_arms_collected_l675_675812

-- Define the conditions as parameters
def arms_of_starfish := 7 * 5
def arms_of_seastar := 14

-- Define the theorem to prove total arms
theorem total_arms_collected : arms_of_starfish + arms_of_seastar = 49 := by
  sorry

end total_arms_collected_l675_675812


namespace sector_area_half_triangle_area_l675_675936

theorem sector_area_half_triangle_area (θ : Real) (r : Real) (hθ1 : 0 < θ) (hθ2 : θ < π / 3) :
    2 * θ = Real.tan θ := by
  sorry

end sector_area_half_triangle_area_l675_675936


namespace rectangle_area_l675_675684

theorem rectangle_area (L B r s : ℝ) (h1 : L = 5 * r)
                       (h2 : r = s)
                       (h3 : s^2 = 16)
                       (h4 : B = 11) :
  (L * B = 220) :=
by
  sorry

end rectangle_area_l675_675684


namespace pyramid_intersection_max_sides_l675_675015

-- Define the maximum number of sides of intersection polygon
def max_sides_intersection (n : ℕ) : ℕ :=
  n-1 + (n-1) / 2

-- Formalize the problem statement
theorem pyramid_intersection_max_sides (n : ℕ) (h : 0 < n) :
  ∃ m, m = max_sides_intersection n ∧ m = n-1 + (n-1) / 2 :=
by
  use (n-1 + (n-1) / 2)
  split
  . refl
  . refl

-- Example usage
example : pyramid_intersection_max_sides 5 := by sorry

end pyramid_intersection_max_sides_l675_675015


namespace lowest_deg_polyf_l675_675093

theorem lowest_deg_polyf (f : ℕ × ℕ → ℕ) 
(h1 : ∀ x y, f (x, y) + f (y, x) = 0)
(h2 : ∀ x y, f (x, x + y) + f (y, x + y) = 0) :
  ∃ a b c : ℤ, f = λ ⟨x, y⟩, (x - y) * x * y * (x + y) * (2 * x^2 - 5 * x * y + 2 * y^2) :=
sorry

end lowest_deg_polyf_l675_675093


namespace perimeter_of_pentagon_l675_675198

variable (x : ℝ)

theorem perimeter_of_pentagon (h1 : ∀ (PQ RS QR SP : ℝ), PQ = 30 → QR = 30 → RS = 30 → SP = 30 → PQ + QR + RS + SP = 120)
                             (h2 : ∀ (PZ ZS SP : ℝ), SP = 30 → PZ + ZS + SP = 2x → PZ + ZS = 2x - 30) :
  ∀ (PQ RS QR SP PZ ZS : ℝ), PQ = 30 → QR = 30 → RS = 30 → SP = 30 → PZ + ZS = 2x - 30 →
  PQ + QR + RS + PZ + ZS = 60 + 2x := 
by
  intros PQ RS QR SP PZ ZS hPQ hQR hRS hSP hPZ_ZS
  calc
  PQ + QR + RS + PZ + ZS = 30 + 30 + 30 + (2x - 30) : by rw [hPQ, hQR, hRS, hPZ_ZS]
  ... = 60 + 2x : by linarith

#check perimeter_of_pentagon

end perimeter_of_pentagon_l675_675198


namespace probability_A_wins_3_1_l675_675725

/-- The probability that player A wins a best-of-five match with a score of 3:1,
    given that the probability of A winning each game is 2/3, is 8/27. -/
theorem probability_A_wins_3_1 (p : ℝ) (h : p = 2/3) : 
  (3.choose 1) * p^3 * (1 - p) = 8/27 :=
by
  rw [h]
  -- Below is placeholder proof, which should be replaced while working further in Lean.
  sorry

end probability_A_wins_3_1_l675_675725


namespace isosceles_triangle_angle_condition_l675_675528

theorem isosceles_triangle_angle_condition (A B C : ℝ) (h_iso : A = B) (h_angle_eq : A = 2 * C ∨ C = 2 * A) :
    (A = 45 ∨ A = 72) ∧ (B = 45 ∨ B = 72) :=
by
  -- Given isosceles triangle properties.
  sorry

end isosceles_triangle_angle_condition_l675_675528


namespace real_solutions_in_interval_l675_675449

noncomputable def problem_statement (x : ℝ) : Prop :=
  (x + 1 > 0) ∧ 
  (x ≠ -1) ∧
  (x^2 / (x + 1 - Real.sqrt (x + 1))^2 < (x^2 + 3 * x + 18) / (x + 1)^2)
  
theorem real_solutions_in_interval (x : ℝ) (h : problem_statement x) : -1 < x ∧ x < 3 :=
sorry

end real_solutions_in_interval_l675_675449


namespace arccos_cos_7_l675_675068

noncomputable def arccos_cos_7_eq_7_minus_2pi : Prop :=
  ∃ x : ℝ, x = 7 - 2 * Real.pi ∧ Real.arccos (Real.cos 7) = x

theorem arccos_cos_7 :
  arccos_cos_7_eq_7_minus_2pi :=
by
  sorry

end arccos_cos_7_l675_675068


namespace min_perimeter_l675_675719

-- Define the triangle and its properties
structure Triangle (ABC : Triangle) where
  AB : ℕ
  AC : ℕ
  BC : ℕ
  AB_eq_AC : AB = AC
  BI : ℝ
  BI_eq : BI = 6

-- Prove that the smallest possible perimeter of the triangle is 56
theorem min_perimeter (ABC : Triangle) (h : ABC.AB_eq_AC) (h' : ABC.BI_eq) :
  2 * (ABC.AB + ABC.BC / 2) = 56 :=
sorry

end min_perimeter_l675_675719


namespace selected_number_in_first_group_is_7_l675_675010

def N : ℕ := 800
def k : ℕ := 50
def interval : ℕ := N / k
def selected_number : ℕ := 39
def second_group_start : ℕ := 33
def second_group_end : ℕ := 48

theorem selected_number_in_first_group_is_7 
  (h1 : interval = 16)
  (h2 : selected_number ≥ second_group_start ∧ selected_number ≤ second_group_end)
  (h3 : ∃ n, selected_number = second_group_start + interval * n - 1) :
  selected_number % interval = 7 :=
sorry

end selected_number_in_first_group_is_7_l675_675010


namespace quotient_remainder_increase_l675_675355

theorem quotient_remainder_increase (a b q r q' r' : ℕ) (hb : b ≠ 0) 
    (h1 : a = b * q + r) (h2 : 0 ≤ r) (h3 : r < b) (h4 : 3 * a = 3 * b * q' + r') 
    (h5 : 0 ≤ r') (h6 : r' < 3 * b) :
    q' = q ∧ r' = 3 * r := by
  sorry

end quotient_remainder_increase_l675_675355


namespace triangle_side_length_l675_675545

theorem triangle_side_length (A B C : Type) [euclidean_geometry A B C]
  (angle_B : B = 120 * π / 180)
  (AC : ℝ) (AB : ℝ) 
  (h₁ : AC = sqrt 19)
  (h₂ : AB = 2) :
  ∃ BC, BC = 3 :=
begin
  -- proof to be filled in
  sorry
end

end triangle_side_length_l675_675545


namespace pipe_fill_time_solution_l675_675013

noncomputable def pipe_fill_time_without_leak : ℝ := 
  let T := 6  -- the correct answer proved
  T  -- returning T

theorem pipe_fill_time_solution (T : ℝ) :
  let pipe_with_leak_time := 8
  let leak_empty_time := 24
  (1 / T - 1 / leak_empty_time = 1 / pipe_with_leak_time) → 
  T = 6 :=
by
  intro h
  have h := calc
    1 / T - 1 / 24 = 1 / 8 : h
  sorry

end pipe_fill_time_solution_l675_675013


namespace sum_of_two_relatively_prime_integers_l675_675313

theorem sum_of_two_relatively_prime_integers (x y : ℕ) : 0 < x ∧ x < 30 ∧ 0 < y ∧ y < 30 ∧
  gcd x y = 1 ∧ x * y + x + y = 119 ∧ x + y = 20 :=
by
  sorry

end sum_of_two_relatively_prime_integers_l675_675313


namespace find_m_l675_675576

theorem find_m (m : ℝ) :
  (∃ x ∈ set.Icc (-2 : ℝ) 4, |x| ≤ m) ∧ (set.filter (λ x, |x| ≤ m) (set.Icc (-2 : ℝ) 4)).card / (set.Icc (-2 : ℝ) 4).card = 1 / 3 → m = 1 :=
sorry

end find_m_l675_675576


namespace lipstick_ratio_l675_675980

-- Definitions based on conditions
variables (total_students blue_lipstick red_lipstick colored_lipstick : ℕ)
variables (one_fifth_red_wore_blue : blue_lipstick = red_lipstick / 5)
variables (one_quarter_colored_wore_red : red_lipstick = colored_lipstick / 4)

-- Conditions explicitly given
theorem lipstick_ratio (h1 : total_students = 200) (h2 : blue_lipstick = 5) 
  (h3 : one_fifth_red_wore_blue) (h4 : one_quarter_colored_wore_red) :
  colored_lipstick / total_students = 1 / 2 :=
begin
  sorry
end

end lipstick_ratio_l675_675980


namespace problem_expression_value_l675_675623

theorem problem_expression_value (x : ℤ) (h : x = -2023) :
    abs (abs (abs x - x) - abs x) - x^2 = -4094506 :=
by
  rw h
  sorry

end problem_expression_value_l675_675623


namespace find_smaller_number_l675_675982

theorem find_smaller_number {x y : ℤ} 
  (h1 : y = 2 * x - 3) 
  (h2 : x + y = 39) : 
  x = 14 :=
begin
  sorry
end

end find_smaller_number_l675_675982


namespace teacher_age_l675_675752

theorem teacher_age (avg_age_students : ℕ) (num_students : ℕ) (avg_age_students_teacher : ℕ) (num_total : ℕ) (teacher_age : ℕ) (h1 : avg_age_students = 15) (h2 : num_students = 30) (h3 : avg_age_students_teacher = 16) (h4 : num_total = 31) : teacher_age = 46 :=
by
  -- Assume the conditions are pre-provided as true    
  have h5 : num_students * avg_age_students = 450 := by sorry
  have h6 : num_total * avg_age_students_teacher = 496 := by sorry
  have h7 : teacher_age = (num_total * avg_age_students_teacher) - (num_students * avg_age_students) := by sorry
  exact h7.trans Eq.refl 46

end teacher_age_l675_675752


namespace difference_of_numbers_l675_675325

theorem difference_of_numbers :
  ∃ (a b : ℕ), a + b = 36400 ∧ b = 100 * a ∧ b - a = 35640 :=
by
  sorry

end difference_of_numbers_l675_675325


namespace dot_product_l675_675082

-- Definitions/Conditions
def is_parabola (f : ℝ × ℝ) (A : ℝ × ℝ) (directrix_dist : ℝ) := 
  f.1 = 1/2 ∧ f.2 = 0 ∧ ((A.2)^2 = 2 * A.1) ∧ (directrix_dist = A.1)

-- Prove that given conditions, the dot product is as required.
theorem dot_product (F A B : ℝ × ℝ) (h1 : is_parabola F A F.1)
  (h2 : B.1 = F.1 + 4 * (A.1 - F.1) ∧ B.2 = F.2 + 4 * (A.2 - F.2)) :
  let FA := (A.1 - F.1, A.2 - F.2) in
  let FB := (B.1 - F.1, B.2 - F.2) in
  (FA.1 * FB.1 + FA.2 * FB.2) = 9 / 4 :=
by {
  sorry
}

end dot_product_l675_675082


namespace mia_stamp_arrangements_l675_675974

def num_arrangements_of_stamps : ℕ := 117

theorem mia_stamp_arrangements :
  let stamps := ([1], [2, 2], [6, 6, 6, 6, 6, 6] : list ℕ) in
  (arrangements stamps 15 = num_arrangements_of_stamps) :=
sorry

end mia_stamp_arrangements_l675_675974


namespace smallest_prime_that_is_6_more_than_perfect_square_and_9_less_than_next_perfect_square_l675_675735

theorem smallest_prime_that_is_6_more_than_perfect_square_and_9_less_than_next_perfect_square :
  ∃ p : ℕ, Prime p ∧ (∃ k m : ℤ, k^2 = p - 6 ∧ m^2 = p + 9 ∧ m^2 - k^2 = 15) ∧ p = 127 :=
sorry

end smallest_prime_that_is_6_more_than_perfect_square_and_9_less_than_next_perfect_square_l675_675735


namespace other_community_boys_l675_675925

theorem other_community_boys (total_boys : ℕ) (perc_muslims perc_hindus perc_sikhs : ℕ) :
  total_boys = 850 →
  perc_muslims = 46 →
  perc_hindus = 28 →
  perc_sikhs = 10 →
  let perc_other := 100 - (perc_muslims + perc_hindus + perc_sikhs) in
  let number_other := (perc_other * total_boys) / 100 in
  number_other = 136 :=
by
  intros
  sorry

end other_community_boys_l675_675925


namespace number_transformation_impossible_l675_675255

theorem number_transformation_impossible :
  ¬(∃ (sequence : list ℕ), 
    sequence = [2, 3, 4, 5, 6, 7, 8, 9, 10, 2012] ∧ 
    (list.sum sequence = 2066) ∧ 
    (initial_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) ∧ 
    ∀ x y, x ∈ initial_list → y ∈ initial_list → 
    ∃ seq', (seq' = (initial_list.erase x).erase y) ∧ 
    (list.sum seq' = list.sum initial_list + 2))
  :=
sorry

end number_transformation_impossible_l675_675255


namespace range_of_m_l675_675890

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 3^(-x) - 1 else real.sqrt x

theorem range_of_m {m : ℝ} (h : f m > 2) : m < -1 ∨ m > 4 :=
by sorry

end range_of_m_l675_675890


namespace collinear_vector_is_optionC_l675_675851

-- Define the original vector a
def a : Vector ℝ 2 := ![3, 2]

-- Define the four options as vectors
def optionA : Vector ℝ 2 := ![-2, -3]
def optionB : Vector ℝ 2 := ![6, -4]
def optionC : Vector ℝ 2 := ![6, 4]
def optionD : Vector ℝ 2 := ![-3, 2]

-- Define the proposition that needs to be proved
theorem collinear_vector_is_optionC : 
  (∃ k : ℝ, optionC = k • a) ∧ 
  ¬(∃ k : ℝ, optionA = k • a) ∧ 
  ¬(∃ k : ℝ, optionB = k • a) ∧ 
  ¬(∃ k : ℝ, optionD = k • a) := 
by 
  sorry

end collinear_vector_is_optionC_l675_675851


namespace how_many_football_tickets_l675_675530

variable (cost_movie_ticket : ℕ) (num_movie_tickets : ℕ) 
variable (total_amount : ℕ)
variable (multiplier : ℕ)

-- Conditions
def cost_movie_tickets := num_movie_tickets * cost_movie_ticket
def cost_football_ticket := cost_movie_tickets / multiplier
def amount_football_tickets (num_football_tickets : ℕ) := num_football_tickets * cost_football_ticket

-- Given Conditions
constant h1 : num_movie_tickets = 8
constant h2 : cost_movie_ticket = 30
constant h3 : multiplier = 2
constant h4 : total_amount = 840

-- Conclusion to Prove
theorem how_many_football_tickets : ∃ n, total_amount = cost_movie_tickets + amount_football_tickets n ∧ n = 5 := by
  sorry

end how_many_football_tickets_l675_675530


namespace count_pairs_sum_multiple_of_five_l675_675570

theorem count_pairs_sum_multiple_of_five :
  let S := finset.range (2005 + 1)
  ∃ n : ℕ, n = finset.card {ab ∈ S.product S | ab.1 ≠ ab.2 ∧ (ab.1 + ab.2) % 5 = 0} ∧ n = 401802 :=
begin
  let S := finset.range (2005 + 1),
  use finset.card {ab ∈ S.product S | ab.1 ≠ ab.2 ∧ (ab.1 + ab.2) % 5 = 0},
  sorry
end

end count_pairs_sum_multiple_of_five_l675_675570


namespace geo_seq_product_l675_675189

theorem geo_seq_product (a : ℕ → ℝ) (r : ℝ) (h_pos : ∀ n, 0 < a n) 
  (h_geo : ∀ n, a (n + 1) = a n * r) 
  (h_roots : a 1 ^ 2 - 10 * a 1 + 16 = 0) 
  (h_root19 : a 19 ^ 2 - 10 * a 19 + 16 = 0) : 
  a 8 * a 10 * a 12 = 64 :=
by
  sorry

end geo_seq_product_l675_675189


namespace largest_possible_median_l675_675732

theorem largest_possible_median (x : ℤ) (h : x < 0) : 
  let S := {x, 2 * x, 4, 1, 7}.to_finset in
  S.sort (≤) ![2] = 1 := sorry

end largest_possible_median_l675_675732


namespace range_h_l675_675440

noncomputable def h (t : ℝ) : ℝ := (t^2 - 1/2 * t) / (t^2 + 2)

theorem range_h : set.range h = {1 / 8} :=
by sorry

end range_h_l675_675440


namespace GCD_area_proof_l675_675986

noncomputable def area_GCD {ABCD : Type*} (A B C D E F G : ABCD) : Real :=
  let BC := dist B C
  let AB := 1.5 * BC
  let area_ABCD := 300
  let E_on_BC := BE / EC = 1 / 2
  let F_mid_AE := dist F (A + E) / 2
  let G_mid_DE := dist G (D + E) / 2
  let area_BEGF := 54
  let area_ABCD_check := AB * BC = area_ABCD

  if area_ABCD_check then  -- Ensure proper area condition
  let total_area := rectangle_area A B C D
  let AED_area := 1 / 2 * total_area
  let FEG_area := 1 / 4 * AED_area
  let BFE_area := area_BEGF - FEG_area
  let ABE_area := 3 * BFE_area
  let ECD_area := total_area - (AED_area + ABE_area)
  let GCD_area := 1 / 2 * ECD_area
  
  in if GCD_area = 50.25 then GCD_area else 0  -- Return result or 0 if mismatch

theorem GCD_area_proof (A B C D E F G : ABCD)
  (h1 : dist B C * 1.5 * dist B C = 300) -- Rectangle ABCD has area 300
  (h2 : BE / EC = 1 / 2)  -- BE : EC = 1 : 2
  (h3 : dist F (A + E) / 2)  -- F is midpoint of AE
  (h4 : dist G (D + E) / 2)  -- G is midpoint of DE
  (h5 : area of quadrilateral BEGF = 54) :
  area_GCD A B C D E F G = 50.25 := sorry

end GCD_area_proof_l675_675986


namespace game_show_total_possible_guesses_correct_l675_675387

noncomputable def total_possible_guesses (digits : List ℕ) : ℕ :=
  let total_digits := digits.length
  let arrangements := Nat.factorial total_digits /
                      (Nat.factorial 4 * Nat.factorial 2 * Nat.factorial 3)
  let partitions_of_digits := Nat.choose (total_digits - 1) 2
  arrangements * partitions_of_digits

def main : ℕ :=
  let digits := [1, 1, 1, 1, 2, 2, 3, 3, 3]
  total_possible_guesses digits

theorem game_show_total_possible_guesses_correct :
  main = 35280 :=
by
  sorry

end game_show_total_possible_guesses_correct_l675_675387


namespace total_cans_in_display_l675_675929

-- Definitions and conditions
def first_term : ℕ := 30
def second_term : ℕ := 27
def nth_term : ℕ := 3
def common_difference : ℕ := second_term - first_term

-- Statement of the problem
theorem total_cans_in_display : 
  ∃ (n : ℕ), nth_term = first_term + (n - 1) * common_difference ∧
  (2 * 165 = n * (first_term + nth_term)) :=
by
  sorry

end total_cans_in_display_l675_675929


namespace benny_kids_l675_675041

theorem benny_kids (total_money : ℕ) (cost_per_apple : ℕ) (apples_per_kid : ℕ) (total_apples : ℕ) (kids : ℕ) :
  total_money = 360 →
  cost_per_apple = 4 →
  apples_per_kid = 5 →
  total_apples = total_money / cost_per_apple →
  kids = total_apples / apples_per_kid →
  kids = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end benny_kids_l675_675041


namespace range_of_x_l675_675115

theorem range_of_x {f : ℝ → ℝ} (h_even : ∀ x, f x = f (-x)) 
  (h_mono_dec : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)
  (h_f2 : f 2 = 0)
  (h_pos : ∀ x, f (x - 1) > 0) : 
  ∀ x, -1 < x ∧ x < 3 ↔ f (x - 1) > 0 :=
sorry

end range_of_x_l675_675115


namespace inequality_abs_l675_675891

noncomputable def f (x : ℝ) : ℝ := abs (x - 1/2) + abs (x + 1/2)

def M : Set ℝ := { x | -1 < x ∧ x < 1 }

theorem inequality_abs (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : |a + b| < |1 + a * b| := 
by
  sorry

end inequality_abs_l675_675891


namespace expected_value_X_probability_X_eq_2_variance_X_variance_Y_l675_675495

section
open ProbabilityTheory

-- Random variable X follows binomial distribution B(9, 2/3)
variable (X : ℕ → ℝ) (hx : binomial 9 (2/3) X)

-- Random variable Y defined as
def Y := λ x, 2 * X x + 1

-- Expected value of X
theorem expected_value_X : E[X] = 6 :=
sorry

-- Probability P(X = 2)
theorem probability_X_eq_2 : P(X = 2) = (nat.choose 9 2) * (2/3)^2 * (1/3)^7 :=
sorry

-- Variance of X
theorem variance_X : variance[X] = 2 :=
sorry

-- Variance of Y
theorem variance_Y : variance[Y] = 8 :=
sorry

end

end expected_value_X_probability_X_eq_2_variance_X_variance_Y_l675_675495


namespace find_r_value_l675_675231

theorem find_r_value (n : ℕ) (r s : ℕ) (h_s : s = 2^n - 1) (h_r : r = 3^s - s) (h_n : n = 3) : r = 2180 :=
by
  sorry

end find_r_value_l675_675231


namespace angle_between_line_and_plane_l675_675123

theorem angle_between_line_and_plane (A : ℝ × ℝ × ℝ) (p : ℝ × ℝ × ℝ) (PA : ℝ × ℝ × ℝ)
  (n : ℝ × ℝ × ℝ) (h1 : A ∈ α) (h2 : p ∉ α) 
  (hPA : PA = (-√3/2, 1/2, √2)) 
  (hn : n = (0, -1/2, -√2)) :
  let θ := real.arcsin ((3 * real.sqrt 11) / 11) in θ = 60 :=
by
  sorry

end angle_between_line_and_plane_l675_675123


namespace value_of_expression_l675_675171

theorem value_of_expression {x y z w : ℝ} (h1 : 4 * x * z + y * w = 3) (h2 : x * w + y * z = 6) :
  (2 * x + y) * (2 * z + w) = 15 :=
by
  sorry

end value_of_expression_l675_675171


namespace angle_BTA_eq_angle_CTD_l675_675854

theorem angle_BTA_eq_angle_CTD 
  (Γ : Circle)
  (B C : Point)
  (incircle : Circle)
  (A K T D : Point)
  (hBC_on_Γ : B ∈ Γ ∧ C ∈ Γ)
  (hK_touches_incircle : ∃ p : Point, p ∈ Γ ∧ Circle.touches Γ incircle p)
  (hT_on_ext_AK : AK inter Γ = {A, T})
  (hD_incircle_touches_BC : Circle.touches incircle (BC segment) D) :
  ∠BTA = ∠CTD := 
by
  sorry

end angle_BTA_eq_angle_CTD_l675_675854


namespace arithmetic_sequence_initial_term_l675_675876

theorem arithmetic_sequence_initial_term (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum : ∀ n, S n = n * (a 1 + n * d / 2))
  (h_product : a 2 * a 3 = a 4 * a 5)
  (h_sum_9 : S 9 = 27)
  (h_d_nonzero : d ≠ 0) :
  a 1 = -5 :=
sorry

end arithmetic_sequence_initial_term_l675_675876


namespace virginia_avg_rainfall_l675_675706

theorem virginia_avg_rainfall:
  let march := 3.79
  let april := 4.5
  let may := 3.95
  let june := 3.09
  let july := 4.67
  let total_rainfall := march + april + may + june + july
  let avg_rainfall := total_rainfall / 5
  avg_rainfall = 4 := by sorry

end virginia_avg_rainfall_l675_675706


namespace perimeter_of_region_l675_675395

theorem perimeter_of_region {s : ℝ} (h1 : s = 2 / Real.pi) : 
  let r := s / 2,
      C := 2 * Real.pi * (r / Real.pi),
      semi_circumference := C / 2,
      total_perimeter := 4 * semi_circumference
  in total_perimeter = 4 :=
by 
  let r := s / 2,
      C := 2 * Real.pi * (r / Real.pi),
      semi_circumference := C / 2,
      total_perimeter := 4 * semi_circumference;
  sorry

end perimeter_of_region_l675_675395


namespace BD_minus_ED_equals_OE_BD_plus_ED_equals_O₁E_l675_675193

-- Given an isosceles triangle ABC with AB = AC
variables {A B C E D O O₁ : Point}

-- Conditions: ABC is isosceles, AE is the altitude, BD intersects AE at D, 
-- O is the incenter, O₁ is the excenter both touching BC at E.
axiom isosceles_triangle (h₁ : triangle.is_isosceles A B C) (h₂ : B ≠ C) : AB = AC
axiom altitude (h₃ : altitude A E)
axiom perpendicular (h₄ : perpendicular_from B AB D)
axiom intersection (h₅ : AE ∩ BD = D)
axiom incenter (h₆ : incenter_triangle O A B C)
axiom excenter (h₇ : excenter_triangle O₁ A B C)
axiom touch_same_point (h₈ : incircle_touches_externally_at E)

-- Desired Propositions to prove:
theorem BD_minus_ED_equals_OE (h : isosceles_triangle h₁ h₂) (h₃ : altitude A E) (h₄ : perpendicular_from B AB D) 
  (h₅ : AE ∩ BD = D) (h₆ : incenter_triangle O A B C) (h₇ : excenter_triangle O₁ A B C) (h₈ : incircle_touches_externally_at E) : 
  BD - ED = OE :=
sorry

theorem BD_plus_ED_equals_O₁E (h : isosceles_triangle h₁ h₂) (h₃ : altitude A E) (h₄ : perpendicular_from B AB D) 
  (h₅ : AE ∩ BD = D) (h₆ : incenter_triangle O A B C) (h₇ : excenter_triangle O₁ A B C) (h₈ : incircle_touches_externally_at E) : 
  BD + ED = O₁ E :=
sorry

end BD_minus_ED_equals_OE_BD_plus_ED_equals_O₁E_l675_675193


namespace range_of_a_l675_675878

noncomputable def p (x : ℝ) : Prop := (3*x - 1)/(x - 2) ≤ 1
noncomputable def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) < 0

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, ¬ q x a) → (¬ ∃ x : ℝ, ¬ p x) → -1/2 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l675_675878


namespace continuity_of_f_l675_675363

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then exp (3 * x) else a + 5 * x

theorem continuity_of_f (a : ℝ) : (∀ x : ℝ, continuous_at (f x a) x) ↔ a = 1 := by
  sorry

end continuity_of_f_l675_675363


namespace ratio_closest_to_one_l675_675300

-- Define the entrance fee for adults and children.
def adult_fee : ℕ := 20
def child_fee : ℕ := 15

-- Define the total collected amount.
def total_collected : ℕ := 2400

-- Define the number of adults and children.
variables (a c : ℕ)

-- The main theorem to prove:
theorem ratio_closest_to_one 
  (h1 : a > 0) -- at least one adult
  (h2 : c > 0) -- at least one child
  (h3 : adult_fee * a + child_fee * c = total_collected) : 
  a / (c : ℚ) = 69 / 68 := 
sorry

end ratio_closest_to_one_l675_675300


namespace ellipse_and_area_l675_675866

-- Conditions
def conditions (a b : ℝ) (c : ℝ) :=
  a = 2 * Real.sqrt 3 ∧ c = 3 ∧ b^2 = a^2 - c^2

-- Equation of the ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 12 + y^2 / 3 = 1

-- Area of PMN
def area_PMN (m : ℝ) : ℝ :=
  2 * Real.sqrt 3 * Real.sqrt ((m^2 + 1) / (m^2 + 4)^2)

-- Maximum area evaluation
def max_area := 1

theorem ellipse_and_area :
  ∃ (a b : ℝ), ∃ (c : ℝ), conditions a b c ∧ 
  ellipse_equation x y ∧ 
  (∀ (m : ℝ), area_PMN m ≤ 1) ∧ 
  (∀ (m : ℝ), area_PMN m = 1 ↔ m = Real.sqrt 2 ∨ m = -Real.sqrt 2)
:=
sorry

end ellipse_and_area_l675_675866


namespace circle_product_divisor_l675_675374

theorem circle_product_divisor:
  ∀ (a : Fin 2015 → ℕ), 
  (∀ i, Nat.gcd (a i).val (a ((i + 1) % 2015)).val = abs ((a i).val - (a ((i + 1) % 2015)).val) ) →
  (∃ k : ℕ, k = 3 * 2^1009 ∧ ∀ i, k ∣ (a i)) := 
sorry

end circle_product_divisor_l675_675374


namespace star_polygon_n_l675_675638

-- Define the given conditions
variables (n : ℕ)

def congruent_edges (n : ℕ) : Prop := -- Simple closed polygon with 2n congruent edges
  ∀ i j, i ≠ j → edge i = edge j

def congruent_angles_A (A : ℕ → ℝ) : Prop := -- Angles A1, A2, ..., An are congruent
  ∀ i j, i ≠ j → A i = A j

def congruent_angles_B (B : ℕ → ℝ) : Prop := -- Angles B1, B2, ..., Bn are congruent
  ∀ i j, i ≠ j → B i = B j

def acute_angle_relation (A B : ℕ → ℝ) : Prop := -- Angle at A1 is 15° less than the angle at B1
  A 1 = B 1 - 15

theorem star_polygon_n (n : ℕ) (A B : ℕ → ℝ) 
  (hc_edges : congruent_edges n)
  (hc_angles_A : congruent_angles_A A)
  (hc_angles_B : congruent_angles_B B)
  (hc_relation : acute_angle_relation A B) :
  n = 24 := by
  sorry

end star_polygon_n_l675_675638


namespace number_of_symmetric_subsets_l675_675508

def has_integer_solutions (m : ℤ) : Prop :=
  ∃ x y : ℤ, x * y = -36 ∧ x + y = -m

def M : Set ℤ :=
  {m | has_integer_solutions m}

def is_symmetric_subset (A : Set ℤ) : Prop :=
  A ⊆ M ∧ ∀ a ∈ A, -a ∈ A

theorem number_of_symmetric_subsets :
  (∃ A : Set ℤ, is_symmetric_subset A ∧ A ≠ ∅) →
  (∃ n : ℕ, n = 31) :=
by
  sorry

end number_of_symmetric_subsets_l675_675508


namespace degree_h_is_5_l675_675229

noncomputable def f (x : ℝ) : ℝ := -5 * x^5 + 2 * x^4 + 7 * x - 8

theorem degree_h_is_5
  (h : ℝ → ℝ)
  (h_deg : Nat)
  (hf_h_deg_3 : degree (f - h) = 3)
  (hf : f = -5 * X^5 + 2 * X^4 + 7 * X - 8) :
  h_deg = 5 :=
sorry

end degree_h_is_5_l675_675229


namespace spade_5_7_8_l675_675844

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_5_7_8 : spade 5 (spade 7 8) = -200 :=
by
  sorry

end spade_5_7_8_l675_675844


namespace chrysanthemums_arrangement_l675_675414

theorem chrysanthemums_arrangement :
  let varieties := ['A', 'B', 'C', 'D', 'E', 'F']
  let is_same_side (A B C : char) (perm : List char) := 
    (perm.indexOf A < perm.indexOf C ∧ perm.indexOf B < perm.indexOf C) ∨ 
    (perm.indexOf A > perm.indexOf C ∧ perm.indexOf B > perm.indexOf C)
  (list.length 
    (list.filter 
      (λ perm, is_same_side 'A' 'B' 'C' perm) 
      (list.perm.quotient varieties)))
  = 480 := sorry

end chrysanthemums_arrangement_l675_675414


namespace hyperbola_equation_l675_675859

-- Lean 4 statement
theorem hyperbola_equation (a b : ℝ) (hpos_a : a > 0) (hpos_b : b > 0)
    (length_imag_axis : 2 * b = 2)
    (asymptote : ∃ (k : ℝ), ∀ x : ℝ, y = k * x ↔ y = (1 / 2) * x) :
  (x y : ℝ) → (x^2 / a^2) - (y^2 / b^2) = 1 ↔ (x^2 / 4) - (y^2 / 1) = 1 :=
by 
  intros
  sorry

end hyperbola_equation_l675_675859


namespace f_of_zero_eq_two_f_is_even_f_quadratic_relation_f_of_double_l675_675857

variable (f : ℝ → ℝ)
variable (hf1 : ∀ x y : ℝ, f(x + y) * f(x - y) = f(x) + f(y))
variable (hf0 : f(0) ≠ 0)

theorem f_of_zero_eq_two : f(0) = 2 := sorry

theorem f_is_even : ∀ x : ℝ, f(x) = f(-x) := sorry

theorem f_quadratic_relation : ∀ x : ℝ, (f(x))^2 - f(x) - 2 = 0 := sorry

theorem f_of_double : ∀ x : ℝ, f(2 * x) = f(x) := sorry

end f_of_zero_eq_two_f_is_even_f_quadratic_relation_f_of_double_l675_675857


namespace sum_first_2016_terms_l675_675161

theorem sum_first_2016_terms (a : ℕ → ℝ) 
  (h_next : ∀ n, a (n+1) = (n + 2) / n * a n) 
  (h_init : a 1 = 2) : 
  ∑ n in Finset.range 2016, 1 / a (n + 1) = 2016 / 2017 := 
by 
  sorry

end sum_first_2016_terms_l675_675161


namespace decreasing_interval_of_g_l675_675303

-- Definitions of the conditions
def f (x : ℝ) := sqrt 3 * sin (x / 2) - cos (x / 2)

def g (x : ℝ) := f (x - 2 * π / 3)

-- The proof problem to be solved
theorem decreasing_interval_of_g :
  ∃ (a b : ℝ), (a = -π/2) ∧ (b = -π/4) ∧ (∀ x y, a < x ∧ x < y ∧ y < b → g x > g y) :=
sorry

end decreasing_interval_of_g_l675_675303


namespace quadratic_with_given_properties_l675_675433

noncomputable def quadratic_function (a b c : ℝ) : (ℝ → ℝ) :=
  λ x, a*x^2 + b*x + c

theorem quadratic_with_given_properties :
  ∃ (a b c : ℝ), quadratic_function a b c = (λ x, -2.5*x^2 + 15*x - 12.5) ∧
    (quadratic_function a b c 1 = 0) ∧
    (quadratic_function a b c 5 = 0) ∧
    (quadratic_function a b c 3 = 10) :=
by
  sorry

end quadratic_with_given_properties_l675_675433


namespace shoe_sizes_mode_and_median_l675_675795

theorem shoe_sizes_mode_and_median :
  let sizes := [(34, 2), (35, 5), (36, 10), (37, 2), (38, 1)] in
  ∃ mode median : ℕ,
  mode = 36 ∧ median = 36 :=
by
  let sizes : List (ℕ × ℕ) := [(34, 2), (35, 5), (36, 10), (37, 2), (38, 1)]
  let mode := 36
  let median := 36
  use [mode, median]
  split
  ·
    unfold mode
    trivial
  ·
    unfold median
    trivial

end shoe_sizes_mode_and_median_l675_675795


namespace not_sufficient_nor_necessary_l675_675955

theorem not_sufficient_nor_necessary (a b : ℝ) : ¬((a > b) → (a^2 > b^2)) ∧ ¬((a^2 > b^2) → (a > b)) := 
by 
  sorry

end not_sufficient_nor_necessary_l675_675955


namespace complex_number_is_in_second_quadrant_l675_675520

variables (A B : ℝ) 

-- Conditions
variable (hA : 0 < A ∧ A < π/2)
variable (hB : 0 < B ∧ B < π/2)
variable (hSum : A + B + (π/2 - A - B) = π)

-- Complex number definition
def z : ℂ := (complex.cos B - complex.sin A) + complex.i * (complex.sin B - complex.cos A)

-- The main statement to prove
theorem complex_number_is_in_second_quadrant 
  (hA : 0 < A ∧ A < π/2)
  (hB : 0 < B ∧ B < π/2)
  (hSum : A + B + (π/2 - A - B) = π) : 
  z A B ∈ {z : ℂ | z.re < 0 ∧ 0 < z.im} :=
sorry

end complex_number_is_in_second_quadrant_l675_675520


namespace ratio_of_circumradii_independent_l675_675756

theorem ratio_of_circumradii_independent {A B C M : Type*}
  [euclidean_geometry A] [euclidean_geometry B]
  [euclidean_geometry C] [euclidean_geometry M]
  (h: collinear {A, C, M}) :
  ∀ M1 M2 : euclidean_geometry.point,
  (between A M1 C) → (between A M2 C) →
  (circumradius (triangle A B M1) / circumradius (triangle B C M1)) =
  (circumradius (triangle A B M2) / circumradius (triangle B C M2)) :=
begin
  sorry
end

end ratio_of_circumradii_independent_l675_675756


namespace problem1_part1_problem1_part2_problem2_part1_problem2_part2_l675_675897

open Set

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | -2 < x ∧ x < 5 }
def B : Set ℝ := { x | -1 ≤ x - 1 ∧ x - 1 ≤ 2 }

theorem problem1_part1 : A ∪ B = { x | -2 < x ∧ x < 5 } := sorry
theorem problem1_part2 : A ∩ B = { x | 0 ≤ x ∧ x ≤ 3 } := sorry

def B_c : Set ℝ := { x | x < 0 ∨ 3 < x }

theorem problem2_part1 : A ∪ B_c = U := sorry
theorem problem2_part2 : A ∩ B_c = { x | (-2 < x ∧ x < 0) ∨ (3 < x ∧ x < 5) } := sorry

end problem1_part1_problem1_part2_problem2_part1_problem2_part2_l675_675897


namespace equal_wear_tires_l675_675385

theorem equal_wear_tires (t D d : ℕ) (h1 : t = 7) (h2 : D = 42000) (h3 : t * d = 6 * D) : d = 36000 :=
by
  sorry

end equal_wear_tires_l675_675385


namespace sum_of_x_intercepts_l675_675261

theorem sum_of_x_intercepts (c d : ℕ) (pos_c : c > 0) (pos_d : d > 0) (cd_eq_eight : c * d = 8) :
    ∑ x in {(c, d) | c * d = 8}.map (λ ⟨c, d⟩, -(2 : ℚ) / c), x = -(15 / 4) :=
by
  sorry

end sum_of_x_intercepts_l675_675261


namespace range_of_a_l675_675880

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
by sorry

end range_of_a_l675_675880


namespace probability_of_40_cents_l675_675995

noncomputable def num_successful_outcomes : ℕ := 16 + 3

def total_outcomes : ℕ := 2 ^ 5

def probability_success : ℚ := num_successful_outcomes / total_outcomes

theorem probability_of_40_cents : probability_success = 19 / 32 := by
  unfold probability_success num_successful_outcomes total_outcomes
  norm_num
  sorry

end probability_of_40_cents_l675_675995


namespace tickets_left_l675_675806

theorem tickets_left (initial_tickets used_tickets tickets_left : ℕ) 
  (h1 : initial_tickets = 127) 
  (h2 : used_tickets = 84) : 
  tickets_left = initial_tickets - used_tickets := 
by
  sorry

end tickets_left_l675_675806


namespace largest_b_l675_675966

theorem largest_b (b : ℝ) (h : (3 * b + 4) * (b - 3) = 9 * b) : 
  b ≤ (4 + 4 * Real.sqrt 5) / 6 :=
begin
  sorry
end

end largest_b_l675_675966


namespace find_values_of_a_and_b_l675_675102

theorem find_values_of_a_and_b (a b : ℚ) (h1 : 4 * a + 2 * b = 92) (h2 : 6 * a - 4 * b = 60) : 
  a = 122 / 7 ∧ b = 78 / 7 :=
by {
  sorry
}

end find_values_of_a_and_b_l675_675102


namespace circumference_of_base_l675_675773

-- Define the height of the cylinder in zhang and convert it to chi
def height_zhang : ℝ := 1.3
def height_chi : ℝ := height_zhang * 10

-- Define the volume of the stored grain in hu and convert it to cubic chi
def volume_hu : ℝ := 2000
def volume_cubic_chi : ℝ := volume_hu * 1.62

-- Define the approximate value of pi
def pi_approx : ℝ := 3

-- Given radius in terms of chi solving the equation: 3r^2 * height_chi = 3240
def radius_chi : ℝ := 9

-- Compute the circumference in chi and convert it to zhang
def circumference_chi := 2 * pi_approx * radius_chi
def circumference_zhang := circumference_chi / 10

-- The statement to prove
theorem circumference_of_base : circumference_zhang = 5.4 := by
  sorry

end circumference_of_base_l675_675773


namespace arrange_numbers_divisible_l675_675048

open List

theorem arrange_numbers_divisible :
  ∃ (σ : List ℕ), σ = [7, 1, 8, 4, 10, 6, 9, 3, 2, 5] ∧
  ∀ (i : ℕ) (h : i < length σ), σ.nthLe i h ∣ (sum (take i σ)) :=
by
  use [7, 1, 8, 4, 10, 6, 9, 3, 2, 5]
  intro i h
  cases i
  case nat.zero => simp [nat.dvd_refl]
  case nat.succ i =>
    simp
    sorry

end arrange_numbers_divisible_l675_675048


namespace alice_commute_distance_l675_675796

noncomputable def office_distance_commute (commute_time_regular commute_time_holiday : ℝ) (speed_increase : ℝ) : ℝ := 
  let v := commute_time_regular * ((commute_time_regular + speed_increase) / commute_time_holiday - speed_increase)
  commute_time_regular * v

theorem alice_commute_distance : 
  office_distance_commute 0.5 0.3 12 = 9 := 
sorry

end alice_commute_distance_l675_675796


namespace scenario_a_scenario_b_l675_675849

-- Define the chessboard and the removal function
def is_adjacent (x1 y1 x2 y2 : ℕ) : Prop :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y1 + 1 = y2)) ∨ (y1 = y2 ∧ (x1 = x2 + 1 ∨ x1 + 1 = x2))

def is_square (x y : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 8 ∧ 1 ≤ y ∧ y ≤ 8

-- Define a Hamiltonian path on the chessboard
inductive HamiltonianPath : (ℕ × ℕ) → (ℕ → (ℕ × ℕ)) → ℕ → Prop
| empty : Π (start : ℕ × ℕ) (path : ℕ → (ℕ × ℕ)), HamiltonianPath start path 0
| step : Π (start : ℕ × ℕ) (path : ℕ → (ℕ × ℕ)) (n : ℕ),
    is_adjacent (path n).1 (path n).2 (path (n+1)).1 (path (n+1)).2 →
    HamiltonianPath start path n →
    (is_square (path (n + 1)).1 (path (n + 1)).2 ∧ ¬ (∃ m < n + 1, path m = path (n + 1))) →
    HamiltonianPath start path (n + 1)

-- State the main theorems
theorem scenario_a : 
  ∃ (path : ℕ → (ℕ × ℕ)),
    HamiltonianPath (3, 2) path 62 ∧
    (∀ n, path n ≠ (2, 2)) := sorry

theorem scenario_b :
  ¬ ∃ (path : ℕ → (ℕ × ℕ)),
    HamiltonianPath (3, 2) path 61 ∧
    (∀ n, path n ≠ (2, 2) ∧ path n ≠ (7, 7)) := sorry

end scenario_a_scenario_b_l675_675849


namespace problem_statement_l675_675755

theorem problem_statement : (-1:ℤ) ^ 4 - (2 - (-3:ℤ) ^ 2) = 6 := by
  sorry  -- Proof will be provided separately

end problem_statement_l675_675755


namespace sequence_is_geometric_progression_l675_675803

noncomputable def seq (n : ℕ≥3) := ℕ → ℝ
def cond (x : seq) : Prop :=
  ∀ n ≥ 3, 
  (∑ i in finset.range (n - 1), (x i) ^ 2) * (∑ i in finset.range n \ {0}, (x i) ^ 2)
  = (∑ i in finset.range (n - 1), x i * x (i + 1)) ^ 2

def geometric_progression (x : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, x (n + 1) = r * x n

theorem sequence_is_geometric_progression (x : seq) (h₁ : ∀ n, x n ≠ 0) (h₂ : cond x) : 
  geometric_progression x :=
  sorry

end sequence_is_geometric_progression_l675_675803


namespace find_y_value_l675_675628

theorem find_y_value : 
  let x := 88 + 0.30 * 88 in
  let y := x - 0.15 * x in
  y = 97.24 :=
by
  let x := 88 + 0.30 * 88
  let y := x - 0.15 * x
  exact (by calc
    y = x - 0.15 * x   : by rfl
    ... = 114.4 - 0.15 * 114.4   : by ring
    ... = 97.24   : by norm_num
  )

end find_y_value_l675_675628


namespace find_QT_l675_675568

-- Define the convex quadrilateral PQRS with given conditions
variables (P Q R S T : Type)
variables [NonemptyType P] [NonemptyType Q] [NonemptyType R] [NonemptyType S]
variables (RS PQ : ℝ)
variables (perpendicular_RS_PQ : RS ⟂ PQ)
variables (perpendicular_PQ_SR : PQ ⟂ SR)
variables (RS_len : RS = 52)
variables (PQ_len : PQ = 34)
variables (PT_len : PT = 15)

-- Define the proof problem
theorem find_QT 
  (perpendicular_RS_PQ : perpendicular RS PQ)
  (perpendicular_PQ_SR : perpendicular PQ SR)
  (RS_len : RS = 52)
  (PQ_len : PQ = 34)
  (PT_len : PT = 15) :
  QT = 36 := sorry

end find_QT_l675_675568


namespace count_distinct_ordered_pairs_reciprocal_sum_eq_one_sixth_l675_675516

theorem count_distinct_ordered_pairs_reciprocal_sum_eq_one_sixth :
  (finset.univ.product finset.univ).filter (λ p : ℕ × ℕ, p.1 > 0 ∧ p.2 > 0 ∧ (1 / p.1 + 1 / p.2 : ℚ) = 1 / 6).card = 9 :=
by {
  sorry
}

end count_distinct_ordered_pairs_reciprocal_sum_eq_one_sixth_l675_675516


namespace calculate_value_expr_l675_675875

noncomputable def value_expr (e : ℝ) : ℝ :=
  2^(1/4) * 8^0.25 + Real.log10 (1/100) + 2 * Real.log (real.sqrt e)

theorem calculate_value_expr (e : ℝ) (he : e = Real.exp 1) :
  value_expr e = 1 :=
by
  sorry

end calculate_value_expr_l675_675875


namespace max_value_sin_y_cos_x_l675_675689

noncomputable section

-- Given conditions
variables (x y : ℝ)
axiom cond1 : cos y + sin x + cos (3 * x) = 0
axiom cond2 : sin (2 * y) - sin (2 * x) = cos (4 * x) + cos (2 * x)

-- Function to determine the value of sin y + cos x
def value (x y : ℝ) : ℝ := sin y + cos x

-- Proof statement
theorem max_value_sin_y_cos_x :
  ∃ x y, cond1 x y ∧ cond2 x y ∧ value x y = 1 + (sqrt (2 + sqrt 2)) / 2 :=
sorry

end max_value_sin_y_cos_x_l675_675689


namespace period_cos_3x_l675_675733

theorem period_cos_3x (T : ℝ) (hCos : ∀ x : ℝ, real.cos x = real.cos (x + 2 * real.pi)) :
  (∀ x : ℝ, real.cos (3 * x) = real.cos (3 * x + T)) ↔ T = 2 * real.pi / 3 :=
by
  sorry

end period_cos_3x_l675_675733


namespace sqrt_value_l675_675874

theorem sqrt_value (h : Real.sqrt 100.4004 = 10.02) : Real.sqrt 1.004004 = 1.002 := 
by
  sorry

end sqrt_value_l675_675874


namespace range_of_x_l675_675152

noncomputable def f (x : ℝ) : ℝ := x^3 + x

theorem range_of_x (x m : ℝ) (hx : x > -2 ∧ x < 2/3) (hm : m ≥ -2 ∧ m ≤ 2) :
    f (m * x - 2) + f x < 0 := sorry

end range_of_x_l675_675152


namespace drone_height_l675_675453

theorem drone_height (r s h : ℝ) 
  (h_distance_RS : r^2 + s^2 = 160^2)
  (h_DR : h^2 + r^2 = 170^2) 
  (h_DS : h^2 + s^2 = 150^2) : 
  h = 30 * Real.sqrt 43 :=
by 
  sorry

end drone_height_l675_675453


namespace next_number_after_48_eighth_number_in_sequence_two_thousand_thirteenth_number_l675_675972

-- Problem: Next number after 48 in the sequence
theorem next_number_after_48 (x : ℕ) (h₁ : x % 3 = 0) (h₂ : (x + 1) = 64) : x = 63 := sorry

-- Problem: Eighth number in the sequence
theorem eighth_number_in_sequence (n : ℕ) 
  (h₁ : ∀ k, (k + 1) % 3 = 0 → 3 * n <= (k + 1) * (k + 1) ∧ (k + 1) * (k + 1) < 3 * (n + 1))
  (h₂ : (n : ℤ) = 8) : n = 168 := sorry

-- Problem: 2013th number in the sequence
theorem two_thousand_thirteenth_number (n : ℕ) 
  (h₁ : ∀ k, (k + 1) % 3 = 0 → 3 * n <= (k + 1) * (k + 1) ∧ (k + 1) * (k + 1) < 3 * (n + 1))
  (h₂ : (n : ℤ) = 2013) : n = 9120399 := sorry

end next_number_after_48_eighth_number_in_sequence_two_thousand_thirteenth_number_l675_675972


namespace initial_price_of_cereal_l675_675692

variable P : ℝ

theorem initial_price_of_cereal :
  (20 * (P - 24) = 1600) → P = 104 :=
by
  sorry

end initial_price_of_cereal_l675_675692


namespace total_weight_of_mixture_l675_675761

variable (W : ℝ)

-- Conditions
-- Sand: 2/7 of the mixture is sand
def sand := (2/7 : ℝ) * W
-- Water: 3/7 of the mixture is water
def water := (3/7 : ℝ) * W
-- Gravel: 1/14 of the mixture is gravel
def gravel := (1/14 : ℝ) * W
-- Cement: 1/14 of the mixture is cement
def cement := (1/14 : ℝ) * W
-- Crushed stones: remaining 12 pounds
def crush_stones := 12

-- Statement to prove
theorem total_weight_of_mixture : sand W + water W + gravel W + cement W + crush_stones = W := by
  sorry

end total_weight_of_mixture_l675_675761


namespace four_digit_number_l675_675342

def digit_constraint (A B C D : ℕ) : Prop :=
  A = B / 3 ∧ C = A + B ∧ D = 3 * B

theorem four_digit_number 
  (A B C D : ℕ) 
  (h₁ : A = B / 3) 
  (h₂ : C = A + B) 
  (h₃ : D = 3 * B)
  (hA_digit : A < 10) 
  (hB_digit : B < 10)
  (hC_digit : C < 10)
  (hD_digit : D < 10) :
  1000 * A + 100 * B + 10 * C + D = 1349 := 
sorry

end four_digit_number_l675_675342


namespace standard_eq_circle_through_fixed_point_l675_675114

-- Conditions
def a (b : ℝ) := (2 : ℝ)
def b : ℝ := 1
def c := sqrt 3
def e := c / a b

def ellipse_eq (x y : ℝ) := (x^2 / 4 + y^2 / 1 = 1)

def A := (-2, 0)
def B := (2, 0)
def F := (sqrt 3, 0)

-- Definitions involving P, PA, PB, S, T:
def PA_slope (P : ℝ × ℝ) := 
(P.2 / (P.1 + 2))

def PB_slope (P : ℝ × ℝ) :=
(P.2 / (P.1 - 2))

def S (P : ℝ × ℝ) := (3, 5 * PA_slope P)
def T (P : ℝ × ℝ) := (3, -(1 / (4 * PA_slope P)))

-- Standard equation proof
theorem standard_eq (x y : ℝ) (a b : ℝ) (P : ℝ × ℝ) :
    a > b ∧ b > 0 ∧ e = sqrt 3 / 2 ∧ F = (sqrt 3, 0) ∧
    ellipse_eq x y ∧ P ≠ A ∧ P ≠ B →
    ellipse_eq x y = (x^2 / 4 + y^2 / 1 = 1) := sorry

-- Circle through fixed point proof
theorem circle_through_fixed_point (x y : ℝ) (P : ℝ × ℝ) :
    a > b > 0 → 
    e = sqrt 3 / 2 → 
    F = (sqrt 3, 0) →
    ellipse_eq x y →
    (let S := S P; let T := T P in 
    ∀ y = 0, ∃ x, x = 3 ± sqrt 5 / 2) := sorry

end standard_eq_circle_through_fixed_point_l675_675114


namespace state_A_selection_percentage_l675_675921

theorem state_A_selection_percentage
  (candidates_A : ℕ)
  (candidates_B : ℕ)
  (x : ℕ)
  (selected_B_ratio : ℚ)
  (extra_B : ℕ)
  (h1 : candidates_A = 7900)
  (h2 : candidates_B = 7900)
  (h3 : selected_B_ratio = 0.07)
  (h4 : extra_B = 79)
  (h5 : 7900 * (7 / 100) + 79 = 7900 * (x / 100) + 79) :
  x = 7 := by
  sorry

end state_A_selection_percentage_l675_675921


namespace fg_of_3_eq_83_l675_675173

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 3 * x + 2

theorem fg_of_3_eq_83 : f (g 3) = 83 := by
  sorry

end fg_of_3_eq_83_l675_675173


namespace chameleon_distance_inequality_l675_675237

def is_chameleon (n : ℕ) (s : String) : Prop :=
  s.length = 3 * n ∧ s.count 'a' = n ∧ s.count 'b' = n ∧ s.count 'c' = n

def swap (s : String) (i : ℕ) : String :=
  if i < s.length - 1 then
    let c1 := s.get i
    let c2 := s.get (i+1)
    s.set i c2 |>.set (i+1) c1
  else s

def distance (X Y : String) : ℕ := sorry -- actual calculation of distance omitted

theorem chameleon_distance_inequality (n : ℕ) (X : String) (hX : is_chameleon n X) :
  ∃ Y : String, is_chameleon n Y ∧ distance X Y ≥ 3 * n^2 / 2 :=
sorry

end chameleon_distance_inequality_l675_675237


namespace Ann_age_is_39_l675_675416

def current_ages (A B : ℕ) : Prop :=
  A + B = 52 ∧ (B = 2 * B - A / 3) ∧ (A = 3 * B)

theorem Ann_age_is_39 : ∃ A B : ℕ, current_ages A B ∧ A = 39 :=
by
  sorry

end Ann_age_is_39_l675_675416


namespace coprime_count_f_100_3_l675_675843

def f (n k : ℕ) : ℕ :=
  let m := n / k
  (Finset.range m).filter (λ x, Nat.coprime (x+1) n).card

theorem coprime_count_f_100_3 : f 100 3 = 14 := by
  sorry

end coprime_count_f_100_3_l675_675843


namespace trigonometric_identity_l675_675058

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180))
  = (4 * Real.sin (10 * Real.pi / 180) / Real.sin (40 * Real.pi / 180)) :=
by sorry

end trigonometric_identity_l675_675058


namespace f_neg_one_l675_675679

-- Assume the function f : ℝ → ℝ
variable (f : ℝ → ℝ)

-- Conditions
-- 1. f(x) is odd: f(-x) = -f(x) for all x ∈ ℝ
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

-- 2. f(x) = 2^x for all x > 0
axiom f_pos : ∀ x : ℝ, x > 0 → f x = 2^x

-- Proof statement to be filled
theorem f_neg_one : f (-1) = -2 := 
by
  sorry

end f_neg_one_l675_675679


namespace lines_perpendicular_l675_675914

theorem lines_perpendicular (a : ℝ) :
  (∃ (x y : ℝ), (2 * x - a * y - 1 = 0) ∧ (x = 1 ∧ y = 1)) → 
  ((2: ℝ) * (-1/2: ℝ) = -1) :=
by
  intro h
  have ha : a = 1 := by sorry
  simp [ha]
  rw [← mul_assoc]
  norm_num
  exact h.elim (λ x hx, hx.elim (λ y hy, hy.2.symm ▸ hy.1.symm ▸ rfl))

end lines_perpendicular_l675_675914


namespace conditional_prob_good_air_quality_l675_675030

def prob_good_any_day : ℝ := 0.75
def prob_good_two_days : ℝ := 0.6
def prob_good_following_day_given_current_good : ℝ := prob_good_two_days / prob_good_any_day

theorem conditional_prob_good_air_quality :
  prob_good_following_day_given_current_good = 0.8 :=
by {
  unfold prob_good_following_day_given_current_good,
  norm_num,
  sorry -- To ignore the proof and focus on the statement
}

end conditional_prob_good_air_quality_l675_675030


namespace minimize_y_l675_675838

noncomputable def y (x : ℝ) : ℝ := 
  Real.tan (x + Real.pi / 3) - Real.tan (x + Real.pi / 4) + Real.sin (x + Real.pi / 4)

theorem minimize_y : ∃ x : ℝ, -Real.pi / 2 ≤ x ∧ x ≤ -Real.pi / 4 ∧ y x = 1 := by
  let z := -x - Real.pi / 4
  have h₁ : Real.pi / 4 ≤ z := sorry
  have h₂ : z ≤ Real.pi / 2 := sorry
  have h₃ : Real.pi / 2 ≤ 2 * z := sorry
  have h₄ : 2 * z ≤ Real.pi := sorry
  have hy : y x = 2 * Real.cos z / Real.sin z + Real.sin z := sorry
  have h_min : z = Real.pi / 2 := sorry -- Minimum occurs at z = π/2
  exact exists.intro (-Real.pi / 2) (and.intro (by linarith) (and.intro (by linarith) (by linarith)))

end minimize_y_l675_675838


namespace max_c_value_l675_675120

variable {a b c : ℝ}

theorem max_c_value (h1 : 2 * (a + b) = a * b) (h2 : a + b + c = a * b * c) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  c ≤ 8 / 15 :=
sorry

end max_c_value_l675_675120


namespace find_q_l675_675805

theorem find_q :
  let a := 2^5 * 3^3 * 5 * 7
  let b := 15! / a^2
  let ab := a * b
  let q := ab / 15!
  q = 4 :=
by
  sorry

end find_q_l675_675805


namespace table_rearrangement_possible_l675_675558

def composite (n : ℕ) : Prop :=
  n ≥ 4 ∧ ∃ m k : ℕ, 2 ≤ m ∧ 2 ≤ k ∧ m * k = n

theorem table_rearrangement_possible :
  ∃ (swap_moves : list (ℕ × ℕ)) (h_len : swap_moves.length ≤ 35),
    ∀ (table : ℕ → ℕ → ℕ)
    (table_cond : ∀ i j, 0 ≤ i ∧ i < 10 → 0 ≤ j ∧ j < 10 → (table i j) ≠ 0 ∧ (table i j) ≤ 100 ∧ (∃! k, 1 ≤ k ∧ k ≤ 100 ∧ (table i j) = k)),
      (let new_table := foldl (λ t (p : ℕ × ℕ), let (i, j) := p in swap t i j) table swap_moves in
        ∀ i j, 0 ≤ i ∧ i < 10 ∧ 0 ≤ j ∧ j < 10 →
               (i < 9 → composite (new_table i j + new_table (i + 1) j)) ∧
               (j < 9 → composite (new_table i j + new_table i (j + 1)))
      ) := sorry

end table_rearrangement_possible_l675_675558


namespace functional_equation_solution_l675_675832

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x + f y) = f x + y) :
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = -x) :=
sorry

end functional_equation_solution_l675_675832


namespace pictures_per_album_l675_675729

theorem pictures_per_album (phone_pics camera_pics albums : ℕ) (h_phone : phone_pics = 22) (h_camera : camera_pics = 2) (h_albums : albums = 4) (h_total_pics : phone_pics + camera_pics = 24) : (phone_pics + camera_pics) / albums = 6 :=
by
  sorry

end pictures_per_album_l675_675729


namespace g_diff_l675_675097

-- Define the function σ(n), the sum of all positive divisors of n
def σ(n : ℕ) : ℕ :=
  ∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d

-- Define the function g(n) as σ(n) / n
def g(n : ℕ) : ℚ :=
  σ(n) / n

-- State the final proof goal: g(432) - g(216) = 5 / 54
theorem g_diff : g(432) - g(216) = (5 : ℚ) / 54 :=
  sorry

end g_diff_l675_675097


namespace max_min_distance_inequality_l675_675465

theorem max_min_distance_inequality (n : ℕ) (D d : ℝ) (h1 : d > 0) 
    (exists_points : ∃ (points : Fin n → ℝ × ℝ), 
      (∀ i j : Fin n, i ≠ j → dist (points i) (points j) ≥ d) 
      ∧ (∀ i j : Fin n, dist (points i) (points j) ≤ D)) : 
    D / d > (Real.sqrt (n * Real.pi)) / 2 - 1 := 
  sorry

end max_min_distance_inequality_l675_675465


namespace sarah_math_homework_pages_l675_675649

theorem sarah_math_homework_pages (x : ℕ) 
  (h1 : ∀ page, 4 * page = 4 * 6 + 4 * x)
  (h2 : 40 = 4 * 6 + 4 * x) : 
  x = 4 :=
by 
  sorry

end sarah_math_homework_pages_l675_675649


namespace cosQ_is_0_point_4_QP_is_12_prove_QR_30_l675_675668

noncomputable def find_QR (Q : Real) (QP : Real) : Real :=
  let cosQ := 0.4
  let QR := QP / cosQ
  QR

theorem cosQ_is_0_point_4_QP_is_12_prove_QR_30 :
  find_QR 0.4 12 = 30 :=
by
  sorry

end cosQ_is_0_point_4_QP_is_12_prove_QR_30_l675_675668


namespace angle_between_vectors_l675_675899

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

theorem angle_between_vectors (ha : ∥a∥ = 1) (hb : ∥b∥ = real.sqrt 2) (h_perp : inner a (a + b) = 0) :
  real.angle a b = 3 * real.pi / 4 :=
begin
  sorry
end

end angle_between_vectors_l675_675899


namespace quadrilateral_areas_equal_l675_675616

structure Point := (x y : ℝ)
structure Quadrilateral := (A B C D P : Point) (convex : Prop)

noncomputable def midpoint (P Q : Point) : Point := 
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

def diagonals_intersect_at_midpoints (Q : Quadrilateral) : Prop :=
  let AC_mid := midpoint Q.A Q.C in
  let BD_mid := midpoint Q.B Q.D in
  AC_mid = BD_mid ∧ Q.P = AC_mid

def equal_triangle_areas (Q : Quadrilateral) : Prop :=
  let area (P1 P2 P3 : Point) := abs ((P1.x * (P2.y - P3.y) + P2.x * (P3.y - P1.y) + P3.x * (P1.y - P2.y)) / 2) in
  area Q.P Q.A Q.B = area Q.P Q.B Q.C ∧
  area Q.P Q.B Q.C = area Q.P Q.C Q.D ∧
  area Q.P Q.C Q.D = area Q.P Q.D Q.A

theorem quadrilateral_areas_equal (Q : Quadrilateral)
  (h1 : Q.convex)
  (h2 : diagonals_intersect_at_midpoints Q) :
  equal_triangle_areas Q :=
sorry

end quadrilateral_areas_equal_l675_675616


namespace alicia_candies_problem_l675_675407

theorem alicia_candies_problem :
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ (n % 9 = 7) ∧ (n % 7 = 5) ∧ n = 124 :=
by
  sorry

end alicia_candies_problem_l675_675407


namespace relative_cumulative_product_4027_l675_675228

theorem relative_cumulative_product_4027
  (a : ℕ → ℝ)
  (ht_pos : ∀ n, 0 < a n)
  (T : ℕ → ℝ)
  (hT_2013 : T 2013 = ∏ i in finset.range 2013, a i)
  (h_relative_2013 : ∑ i in finset.range 2013, real.log (T (i + 1)) = 2013) :
  (∑ i in finset.range 2014, real.log ((λ n, if n = 0 then 10 else 10 * T n) i)) = 4027 :=
begin
  sorry
end

end relative_cumulative_product_4027_l675_675228


namespace line_parallel_to_y_axis_l675_675534

theorem line_parallel_to_y_axis (a b : ℝ) (h : ∀ x y : ℝ, a * x + b * y + 1 = 0 → b = 0):
  a ≠ 0 ∧ b = 0 :=
sorry

end line_parallel_to_y_axis_l675_675534


namespace determine_k_for_quadratic_eq_l675_675820

theorem determine_k_for_quadratic_eq {k : ℝ} :
  (∀ r s : ℝ, 3 * r^2 + 5 * r + k = 0 ∧ 3 * s^2 + 5 * s + k = 0 →
    (|r + s| = r^2 + s^2)) ↔ k = -10/3 := by
sorry

end determine_k_for_quadratic_eq_l675_675820


namespace minimum_faulty_lights_l675_675920

def is_neighbor (a b : ℕ × ℕ) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 = b.2 - 1)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 = b.1 - 1))

def faulty_light (grid : ℕ × ℕ → bool) (pos : ℕ × ℕ) : Prop :=
  even (Finset.card (Finset.filter (λ p, grid p) (Finset.univ.filter (is_neighbor pos))))

def faulty_light_count (grid : ℕ × ℕ → bool) : ℕ :=
  Finset.card (Finset.filter (faulty_light grid) (Finset.range 2017 ×ˢ Finset.range 2017))

theorem minimum_faulty_lights : ∃ grid : (ℕ × ℕ → bool), (faulty_light_count grid = 1) :=
sorry

end minimum_faulty_lights_l675_675920


namespace polynomial_reduction_l675_675014

theorem polynomial_reduction (a : Fin 17 → ℕ) (h : ∀ z, ((∏ i in Finset.range 16, (1 - z^((i + 1) : ℕ))^(a i)) % z^17) = (1 - z % z^17)) : 
    a 15 = 1 :=
sorry

end polynomial_reduction_l675_675014


namespace parallel_lines_l675_675139

theorem parallel_lines (a : ℝ) : (∀ x y : ℝ, 2 * x + a * y + 1 = 0 ↔ x - 4 * y - 1 = 0) → a = -8 :=
by
  intro h -- Introduce the hypothesis that lines are parallel
  sorry -- Skip the proof

end parallel_lines_l675_675139


namespace prob_l675_675911

theorem prob (f : ℝ → ℝ) (h1 : ∀ x, f(x) + deriv f x > 1) (h2 : f(0) = 4) :
  ∀ x, x > 0 → f(x) > 3 / Real.exp x + 1 := 
by 
  sorry

end prob_l675_675911


namespace meeting_time_l675_675683

/-
The jogging track circumference: 726 meters.
Suresh's speed: 4.5 km/hr.
His wife's speed: 3.75 km/hr.
Prove that the time it takes for them to meet for the first time is approximately 5.28 minutes.
-/

open Nat

noncomputable def joggingTrackCircumference : ℝ := 726
noncomputable def sureshSpeed : ℝ := 4.5 * 1000 / 60
noncomputable def wifeSpeed : ℝ := 3.75 * 1000 / 60
noncomputable def relativeSpeed : ℝ := sureshSpeed + wifeSpeed

theorem meeting_time :
  joggingTrackCircumference / relativeSpeed ≈ 5.28 :=
by
  sorry

end meeting_time_l675_675683


namespace sum_of_solutions_of_abs_equation_l675_675442

theorem sum_of_solutions_of_abs_equation : ∑ (y : ℝ) in {y : ℝ | |3 * y - 6| = 3 * |y + 2|}, y = 0 :=
by
  -- Definitions and conditions derived from the problem
  sorry

end sum_of_solutions_of_abs_equation_l675_675442


namespace solution_l675_675879

noncomputable def X_dist : ℕ → ℚ
| 0 := 1/2
| 1 := 1/4
| 2 := 1/4
| _ := 0

theorem solution:
  (∀ x, 0 ≤ X_dist x ∧ X_dist x ≤ 1) ∧
  (X_dist 0 + X_dist 1 + X_dist 2 = 1) ∧
  (let a := X_dist 1 in a = 1/4) ∧
  (let E := 0 * 1/2 + 1 * 1/4 + 2 * 1/4 in E = 3/4) ∧
  (let D := 1/2 * (-3/4)^2 + 1/4 * (1/4)^2 + 1/4 * (5/4)^2 in D = 11/16) :=
begin
  sorry
end

end solution_l675_675879


namespace perimeter_inequality_l675_675596

-- defining the angles, points, and the perimeters in Lean
variables {A B C D E F : Type} [Point A] [Point B] [Point C] [Point D] [Point E] [Point F] 
variables (ABC : Triangle A B C) (DEF : Triangle D E F)
variables (p p1 : ℝ)

-- given conditions: angles at A and C, positioning of points, and specific angle conditions
axiom angle_A : angle A = 30
axiom angle_C : angle C = 30
axiom D_on_AB : on_line AB D
axiom E_on_BC : on_line BC E
axiom F_on_CA : on_line CA F
axiom angle_BFD_60 : angle B F D = 60
axiom angle_BFE_60 : angle B F E = 60

-- prove that perimeter ABC is less than or equal to twice perimeter DEF
theorem perimeter_inequality : p ≤ 2 * p1 := sorry

end perimeter_inequality_l675_675596


namespace tan_B_tan_C_equals_one_l675_675942

variables {A B C H D : Type*} [Real]

-- Definitions based on given conditions
def is_triangle (A B C : Type*) : Prop := sorry
def is_orthocenter (H : Type*) (A B C : Type*) : Prop := sorry
def is_altitude (BD : Type*) (B D : Type*) : Prop := sorry

-- Values given in the conditions
lemma length_HD : length (segment H D) = 4 := sorry
lemma length_HB : length (segment H B) = 10 := sorry
lemma length_BD : length (segment B D) = 14 := sorry

-- The goal statement
theorem tan_B_tan_C_equals_one :
  is_triangle A B C →
  is_orthocenter H A B C →
  is_altitude BD B D →
  length (segment H D) = 4 →
  length (segment H B) = 10 →
  length (segment B D) = 14 →
  tan_angle B * tan_angle C = 1 := by
  sorry

end tan_B_tan_C_equals_one_l675_675942


namespace correct_answer_of_john_l675_675592

theorem correct_answer_of_john (x : ℝ) (h : 5 * x + 4 = 104) : (x + 5) / 4 = 6.25 :=
by
  sorry

end correct_answer_of_john_l675_675592


namespace grabbed_books_l675_675904

-- Definitions from conditions
def initial_books : ℕ := 99
def boxed_books : ℕ := 3 * 15
def room_books : ℕ := 21
def table_books : ℕ := 4
def kitchen_books : ℕ := 18
def current_books : ℕ := 23

-- Proof statement
theorem grabbed_books : (boxed_books + room_books + table_books + kitchen_books = initial_books - (23 - current_books)) → true := sorry

end grabbed_books_l675_675904


namespace oleg_can_win_l675_675979

theorem oleg_can_win (n : ℕ) (a : fin n → ℕ) (hn : 1 < n)
  (hp : ∀ i : fin n, a i < Nat.prime n) :
  ∃ x y : fin n, x ≠ y ∧ a x = a y :=
by
  sorry

end oleg_can_win_l675_675979


namespace trig_identity_l675_675061

theorem trig_identity :
  ∀ (θ : ℝ),
    θ = 70 * (π / 180) →
    (1 / Real.cos θ - (Real.sqrt 3) / Real.sin θ) = Real.sec (20 * (π / 180)) ^ 2 :=
by
  sorry

end trig_identity_l675_675061


namespace problem_conditions_l675_675248

theorem problem_conditions (x: ℝ) (h1: |x - 2.5| < 1.5) 
                           (h2: (∃ k: ℤ, k^2 ≤ x^2 + x + 1 < (k + 1)^2 → false)) 
                           (h3: ∃ (n : ℤ), x = n) 
                           (h4: log x 10 > 2):
                           ¬ (|x - 2.5| < 1.5 ∧ 
                              (∃ k: ℤ, k^2 ≤ x^2 + x + 1 < (k + 1)^2 → false) ∧ 
                              log x 10 > 2 ∧ 
                              ∃ (n : ℤ), x = n) → 
                           (∃ (condition_not_met: "x is an integer"), condition_not_met) := 
  sorry

end problem_conditions_l675_675248


namespace organization_has_10_members_l675_675560

theorem organization_has_10_members
  (committees : Finset ℕ) -- representing the set of 5 committees
  (members : Type) -- representing the members of the organization
  (belongs_to : members → Finset ℕ) -- function mapping each member to a pair of committees
  (H1 : committees.card = 5)
  (H2 : ∀ m : members, (belongs_to m).card = 2) -- each member belongs to exactly two committees
  (H3 : ∀ c1 c2 : ℕ, (c1 ∈ committees) → (c2 ∈ committees) → c1 ≠ c2 → (Finset.filter (λ m, (belongs_to m).contains c1 ∧ (belongs_to m).contains c2) Finset.univ).card = 1) -- each pair of committees has exactly one unique member in common
  : Finset.card (Finset.univ : Finset members) = 10 := sorry

end organization_has_10_members_l675_675560


namespace coplanar_vectors_x_value_l675_675512

open Matrix

-- Definitions based on conditions
def vec_a (x : ℝ) : Vector := ![1, x, 2]
def vec_b : Vector := ![0, 1, 2]
def vec_c : Vector := ![1, 0, 0]

-- Statement of the problem in Lean
theorem coplanar_vectors_x_value (x : ℝ) 
  (coplanar : det ![vec_a x, vec_b, vec_c] = 0) : x = -1 :=
  by sorry

end coplanar_vectors_x_value_l675_675512


namespace problem_statement_l675_675895

noncomputable def A : Set ℝ := {x | x^2 - 5 * x - 6 = 0}
noncomputable def B : Set ℝ := {x | x^2 - x - 6 = 0}
def U : Set ℝ := {-2, -1, 0, 3, 6}

def C (S : Set ℝ) : Set ℝ := U \ S

theorem problem_statement :
  (A ∪ B = {-2, -1, 3, 6}) ∧
  (A ∩ B = ∅) ∧
  ((C A) ∩ (C B) = {0}) :=
by
  sorry

end problem_statement_l675_675895


namespace trig_max_value_l675_675837

theorem trig_max_value 
  (α β γ δ ε : ℝ) :
  (cos α * sin β + cos β * sin γ + cos γ * sin δ + cos δ * sin ε + cos ε * sin α ≤ 5) 
  ∧ 
  (∃ α' β' γ' δ' ε' : ℝ, cos α' * sin β' + cos β' * sin γ' + cos γ' * sin δ' + cos δ' * sin ε' + cos ε' * sin α' = 5) :=
sorry

end trig_max_value_l675_675837


namespace main_theorem_l675_675941

noncomputable def problem_1 : Prop :=
  ∃ (A : ℝ × ℝ),
    (let C := (4, -1) in
    let median_BC := λ x y : ℝ, 3 * x - y - 1 = 0 in
    let AC := λ x y : ℝ, y + 1 = -(x - 4) in
    AC (fst A) (snd A) = 0 ∧ median_BC (fst A) (snd A) = 0) ∧
    A = (1, 2)

noncomputable def problem_2 : Prop :=
  ∀ (a b : ℝ),
    (a > 0 ∧ b > 0 ∧ (1 / a + 2 / b = 1)) →
    let S := (1 / 2) * a * b in
    S ≥ 4 ∧ (∃ (hole : S = 4))

theorem main_theorem : problem_1 ∧ problem_2 :=
  sorry

end main_theorem_l675_675941


namespace domain_of_f_l675_675089

noncomputable def f (x : ℝ) : ℝ := Math.tan (Real.arccos (Real.sin x))

theorem domain_of_f :
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi}ᶜ = {x : ℝ | ∃ k : ℤ, x ≠ k * Real.pi} :=
by
  sorry

end domain_of_f_l675_675089


namespace number_of_correct_statements_is_2_l675_675647

def line : Type := sorry  -- Placeholder for the definition of a line
def plane : Type := sorry  -- Placeholder for the definition of a plane

variables (m n : line) (α β : plane)

-- Conditions for the problem
def condition1 := (m ∥ α) ∧ (n ∥ β) ∧ (α ∥ β)
def condition2 := (m ⟂ α) ∧ (n ⟂ β) ∧ (α ⟂ β)
def condition3 := (m ⟂ α) ∧ (n ∥ β) ∧ (α ∥ β)
def condition4 := (m ∥ α) ∧ (n ⟂ β) ∧ (α ⟂ β)

-- Definitions for the statements' conclusions
def statement1 := ∀ (m n : line) (α β : plane), condition1 → (m ∥ n)
def statement2 := ∀ (m n : line) (α β : plane), condition2 → (m ⟂ n)
def statement3 := ∀ (m n : line) (α β : plane), condition3 → (m ⟂ n)
def statement4 := ∀ (m n : line) (α β : plane), condition4 → (m ∥ n)

-- Problem statement: Prove the number of correct statements is 2
theorem number_of_correct_statements_is_2 : 
  (¬ statement1 m n α β) ∧
  (statement2 m n α β) ∧
  (statement3 m n α β) ∧
  (¬ statement4 m n α β) → 2 := 
sorry

end number_of_correct_statements_is_2_l675_675647


namespace birthday_count_l675_675696

theorem birthday_count (N : ℕ) (P : ℝ) (days : ℕ) (hN : N = 1200) (hP1 : P = 1 / 365 ∨ P = 1 / 366) 
  (hdays : days = 365 ∨ days = 366) : 
  N * P = 4 :=
by
  sorry

end birthday_count_l675_675696


namespace no_extreme_points_range_a_l675_675099

theorem no_extreme_points_range_a {a : ℝ} :
  (∀ x : ℝ, let f := λ x, x^3 + a*x^2 + 7*a*x in
           (∀ y : ℝ, 3*y^2 + 2*a*y + 7*a ≥ 0)) →
  0 ≤ a ∧ a ≤ 21 :=
by
  sorry

end no_extreme_points_range_a_l675_675099


namespace negation_of_ln_gt_one_l675_675507

open Real

theorem negation_of_ln_gt_one :
  ¬ (∀ x : ℝ, ln x > 1) ↔ ∃ x : ℝ, ln x ≤ 1 :=
by
  -- skeleton for the proof
  sorry

end negation_of_ln_gt_one_l675_675507


namespace time_for_completion_l675_675356

noncomputable def efficiency_b : ℕ := 100

noncomputable def efficiency_a := 130

noncomputable def total_work := efficiency_a * 23

noncomputable def combined_efficiency := efficiency_a + efficiency_b

noncomputable def time_taken := total_work / combined_efficiency

theorem time_for_completion (h1 : efficiency_a = 130)
                           (h2 : efficiency_b = 100)
                           (h3 : total_work = 2990)
                           (h4 : combined_efficiency = 230) :
  time_taken = 13 := by
  sorry

end time_for_completion_l675_675356


namespace x_pow_2n_mod_2nplus2_l675_675238

theorem x_pow_2n_mod_2nplus2 (x n : ℤ) (h_odd : Odd n) (h_ge_1 : n ≥ 1) : 
  x^(2^n) ≡ 1 [ZMOD 2^(n+2)] :=
sorry

end x_pow_2n_mod_2nplus2_l675_675238


namespace find_alphas_l675_675148

def increasing_function (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, x < y → f x < f y

def possible_alphas := {-1, 1/2, 1, 2, 3}

theorem find_alphas (α : ℝ) (hα : α ∈ possible_alphas) :
  (∀ x : ℝ, x ≠ 0 → differentiable_at ℝ (λ x, x^α) x) →
  (α ≠ -1 ∧ α ≠ 1/2 ∧ α ≠ 2) →
  increasing_function (λ x, x^α) ↔ (α = 1 ∨ α = 3) :=
by
  sorry

end find_alphas_l675_675148


namespace distance_between_points_l675_675344

-- Define the two points
structure Point where
  x : ℝ
  y : ℝ

def point1 : Point := { x := -5, y := 3 }
def point2 : Point := { x := 3, y := 6 }

-- Function to calculate the distance between two points
def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Statement proving the distance between the points is sqrt(73)
theorem distance_between_points :
  distance point1 point2 = real.sqrt 73 := by
  sorry

end distance_between_points_l675_675344


namespace cot_B_plus_cot_C_eq_2_sqrt3_div_3_l675_675583

noncomputable def triangle_ABC (A B C D : Type) [EuclideanGeometry] : Prop :=
  ∃ (AD BC : Line), 
    isAngleBisector AD ∧ makesAngle AD BC (30 : ℝ)

theorem cot_B_plus_cot_C_eq_2_sqrt3_div_3 
  {A B C D : Type} [EuclideanGeometry]
  (h : triangle_ABC A B C D) : 
  ∃ (cot_B cot_C: ℝ), cot_B + cot_C = 2 * sqrt 3 / 3 := 
sorry

end cot_B_plus_cot_C_eq_2_sqrt3_div_3_l675_675583


namespace correct_observation_value_l675_675686

theorem correct_observation_value :
  ∀ (n : ℕ) (mean wrong_corr correct_corr : ℝ) (total_sum corrected_total_sum : ℝ),
    n = 50 →
    mean = 36 →
    wrong_corr = 47 →
    correct_corr = 36.02 →
    total_sum = n * mean →
    corrected_total_sum = n * correct_corr →
    ∃ x : ℝ, total_sum - wrong_corr + x = corrected_total_sum :=
by
  intros n mean wrong_corr correct_corr total_sum corrected_total_sum
  intro hn hmean hwrong hcorr hsum hcsum
  use corrected_total_sum - (total_sum - wrong_corr)
  rw [hsum, hcsum, hn, hmean, hcorr, hwrong]
  norm_num
  sorry

end correct_observation_value_l675_675686


namespace second_player_wins_l675_675724

noncomputable def distance (x1 y1 x2 y2 : ℕ) : ℕ :=
  (abs (x1 - x2) + abs (y1 - y2))

def valid_move (previous_move : ℕ) (current_move : ℕ) : Prop :=
  current_move > previous_move

def symmetric_move (i j : ℕ) : Prop :=
  1 ≤ i ∧ i ≤ 10 ∧ 1 ≤ j ∧ j ≤ 10 ∧ i ≠ 5 ∧ j ≠ 5 ∧ distance 5 5 i j < distance 5 5 (10-i) (10-j)

theorem second_player_wins :
  ∀ (moves : list (ℕ × ℕ)), 
  (∀ i, i < moves.length → symmetric_move (moves.nth i).fst (moves.nth i).snd) →
  (∀ i, i < moves.length - 1 → valid_move (distance (moves.nth i).fst (moves.nth i).snd (moves.nth (i + 1)).fst (moves.nth (i + 1)).snd) 
                                     (distance (moves.nth (i + 1)).fst (moves.nth (i + 1)).snd (moves.nth (i + 2)).fst (moves.nth (i + 2)).snd)) →
  second_player_wins :=
sorry

end second_player_wins_l675_675724


namespace domain_of_g_l675_675438

def g (x : ℝ) := (x - 3) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_of_g :
  {x : ℝ | ∃ y, g y = g x} = {x : ℝ | x < 2 ∨ 3 < x} :=
by
  sorry

end domain_of_g_l675_675438


namespace rep_decimal_to_fraction_l675_675977

theorem rep_decimal_to_fraction :
  (0.2 + 0.1 * (1/3) = 7/30) :=
by
  have h0 : 0.2 = 1/5 := by sorry
  have h1 : 0.1 = 1/10 := by sorry
  have h2 : 0.\overline{3} = 1/3 := by sorry
  calc
    0.2 + 0.1 * (1/3)
    = 1/5 + 1/10 * 1/3  : by rw [h0, h1, h2]
    ... = 1/5 + 1/30 : by norm_num
    ... = 6/30 + 1/30 : by norm_num
    ... = 7/30 : by norm_num

end rep_decimal_to_fraction_l675_675977


namespace failed_marks_l675_675021

/-- Given that a student needs 45% of 400 marks to pass, if the student got 150 marks, 
    prove that the student failed by 30 marks. -/
theorem failed_marks (maximum_marks : ℕ) (percentage_to_pass : ℝ) (marks_obtained : ℕ) :
  percentage_to_pass = 0.45 →
  maximum_marks = 400 →
  marks_obtained = 150 →
  let passing_marks := (percentage_to_pass * maximum_marks) in
  let marks_failed_by := passing_marks - marks_obtained in
  marks_failed_by = 30 :=
by {
  intros perc pass max_marks obt_marks,
  let passing_marks := (0.45 * 400) : ℝ,
  let marks_failed_by := passing_marks - 150,
  have h_pass : passing_marks = 180, by sorry,
  have h_fail : marks_failed_by = 30, by sorry,
  exact h_fail,
}

end failed_marks_l675_675021


namespace bisectors_adjacent_angles_perpendicular_bisectors_vertical_angles_straight_line_l675_675265

-- Part 1: Proving that the bisectors of two adjacent angles are perpendicular
theorem bisectors_adjacent_angles_perpendicular (x : ℝ) :
  let a := x,
      b := 180 - x,
      bisector_a := a / 2,
      bisector_b := b / 2
  in bisector_a + bisector_b = 90 :=
by
  sorry

-- Part 2: Proving that the bisectors of two vertical angles lie on one straight line
theorem bisectors_vertical_angles_straight_line (a b : ℝ) :
  let bisector_a := a / 2,
      bisector_b := b / 2,
      vertical_angles_equal := a = b,
      sum_vertical_angles := a + b = 180
  in bisector_a + bisector_b = 180 :=
by
  sorry

end bisectors_adjacent_angles_perpendicular_bisectors_vertical_angles_straight_line_l675_675265


namespace points_on_circle_l675_675233

-- Define the points A, B, C, D.
variables {A B C D : Type}

-- Define the orthocenters P, Q, R.
variables (orthocenter : Type → Type)
variables (P : orthocenter B C D)
variables (Q : orthocenter C A D)
variables (R : orthocenter A B D)

-- Hypothesize the given conditions.
def no_three_collinear : Prop := 
  ∀ (X Y Z : Type), X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X → ∃ l : Set Type, (X ∉ l) ∧ (Y ∉ l) ∧ (Z ∉ l)

def D_not_orthocenter_ABC (h : A) (h' : C) : Prop :=
  ¬orthocenter A B C = D

def distinct_concurrent {A B C D : Type} {P Q R : Type} : Prop :=
  ∃ T : Type, ∀ (l₁ l₂ : Set Type), T ∈ l₁ ∧ T ∈ l₂ → l₁ ≠ l₂ → (∀ (T' : Type), T' ≠ T → ∃ l : Set Type, T' ∈ l)

-- The main theorem matching the equivalent math proof problem.
theorem points_on_circle 
  (h1 : no_three_collinear A B C D)
  (h2 : D_not_orthocenter_ABC A D)
  (h3 : ∀ P Q R, (P = orthocenter B C D) ∧ (Q = orthocenter C A D) ∧ (R = orthocenter A B D))
  (h4 : distinct_concurrent AP BQ CR) :
  ∃ O : Type, (∀ (X : Type), X ∈ {A, B, C, D} → ∃ r : ℝ, dist O X = r) :=
sorry

end points_on_circle_l675_675233


namespace mouse_away_from_cheese_l675_675393

theorem mouse_away_from_cheese:
  ∃ a b : ℝ, a = 3 ∧ b = 3 ∧ (a + b = 6) ∧
  ∀ x y : ℝ, (y = -3 * x + 12) → 
  ∀ (a y₀ : ℝ), y₀ = (1/3) * a + 11 →
  (a, b) = (3, 3) :=
by
  sorry

end mouse_away_from_cheese_l675_675393


namespace sin_theta_of_point_on_terminal_side_l675_675142

theorem sin_theta_of_point_on_terminal_side (a : ℝ) (h : a ≠ 0) : 
  ∃ θ : ℝ, (sin θ = sqrt 2 / 2 ∨ sin θ = -sqrt 2 / 2) ∧ 
  (cos θ = sqrt 2 / 2 ∨ cos θ = -sqrt 2 / 2) :=
by 
  sorry

end sin_theta_of_point_on_terminal_side_l675_675142


namespace pq_eq_neg72_l675_675614

theorem pq_eq_neg72 {p q : ℝ} (h : ∀ x, (x - 7) * (3 * x + 11) = x ^ 2 - 20 * x + 63 →
(p = x ∨ q = x) ∧ p ≠ q) : 
(p + 2) * (q + 2) = -72 :=
sorry

end pq_eq_neg72_l675_675614


namespace product_of_B_coordinates_l675_675478

theorem product_of_B_coordinates (M A B : Point) 
  (hM : M = (2, 5))
  (hA : A = (3, 1))
  (hM_is_midpoint : M = ((fst A + fst B) / 2, (snd A + snd B) / 2)) :
  (fst B) * (snd B) = 9 := 
by
  sorry

end product_of_B_coordinates_l675_675478


namespace farmer_brown_additional_cost_l675_675049

-- Definitions for the conditions
def originalQuantity : ℕ := 10
def originalPricePerBale : ℕ := 15
def newPricePerBale : ℕ := 18
def newQuantity : ℕ := 2 * originalQuantity

-- Definition for the target equation (additional cost)
def additionalCost : ℕ := (newQuantity * newPricePerBale) - (originalQuantity * originalPricePerBale)

-- Theorem stating the problem voiced in Lean 4
theorem farmer_brown_additional_cost : additionalCost = 210 :=
by {
  sorry
}

end farmer_brown_additional_cost_l675_675049


namespace axis_of_symmetry_circle_l675_675533

theorem axis_of_symmetry_circle (a : ℝ) : 
  (2 * a + 0 - 1 = 0) ↔ (a = 1 / 2) :=
by
  sorry

end axis_of_symmetry_circle_l675_675533


namespace problem_proof_1_problem_proof_2_l675_675754

variable (α β γ : ℝ)
variable (s s₁ s₂ s₃ t : ℝ)

theorem problem_proof_1 (h1 : t = s * s₁ * Real.tan (α / 2))
                        (h2 : t = s * s₂ * Real.tan (β / 2))
                        (h3 : t = s * s₃ * Real.tan (γ / 2)) :
    t = s * s₁ * Real.tan (α / 2) ∧
    t = s * s₂ * Real.tan (β / 2) ∧
    t = s * s₃ * Real.tan (γ / 2) := by
  sorry

theorem problem_proof_2 (h1 : t = s₂ * s₃ * Real.cot (α / 2))
                        (h2 : t = s₁ * s₃ * Real.cot (β / 2))
                        (h3 : t = s₁ * s₂ * Real.cot (γ / 2)) :
    t = s₂ * s₃ * Real.cot (α / 2) ∧
    t = s₁ * s₃ * Real.cot (β / 2) ∧
    t = s₁ * s₂ * Real.cot (γ / 2) := by
  sorry

end problem_proof_1_problem_proof_2_l675_675754


namespace min_value_of_linear_combination_of_variables_l675_675537

-- Define the conditions that x and y are positive numbers and satisfy the equation x + 3y = 5xy
def conditions (x y : ℝ) : Prop :=
  0 < x ∧ 0 < y ∧ x + 3 * y = 5 * x * y

-- State the theorem that the minimum value of 3x + 4y given the conditions is 5
theorem min_value_of_linear_combination_of_variables (x y : ℝ) (h: conditions x y) : 3 * x + 4 * y ≥ 5 :=
by 
  sorry

end min_value_of_linear_combination_of_variables_l675_675537


namespace max_possible_k_l675_675459

theorem max_possible_k :
  ∃ k : ℕ, from_set_of_integers(3009) k (λ (a_i b_i : ℕ), a_i < b_i) ∧ 
                  (∀ (i ≠ j), (a_i, b_i) ≠ (a_j, b_j)) ∧ 
                  (∀ (i ≠ j), (a_i + b_i) ≠ (a_j + b_j)) ∧
                  (∀ i, a_i + b_i ≤ 3009) ∧
                  k = 1504 :=
begin
  sorry
end

end max_possible_k_l675_675459


namespace slope_of_dividing_line_l675_675723

/-- 
  Prove that given two circles of radius 4 centered at (0, 20) and (6, 12),
  and a line passing through (4, 0) that divides the areas of the circles equally,
  the absolute value of the slope of this line is 4/3.
-/
theorem slope_of_dividing_line : 
  ∀ (r : ℝ) (p₁ p₂ : ℝ × ℝ) (l₁ : ℝ × ℝ) (m : ℝ),
     r = 4 → p₁ = (0, 20) → p₂ = (6, 12) → l₁ = (4, 0) →
     |m| = 4 / 3 →
     (∀ x₁ y₁ x₂ y₂, 
       dist_from_line (m, -4 * m) (x₁, y₁) = dist_from_line (m, -4 * m) (x₂, y₂)) :=
begin
  assume r p₁ p₂ l₁ m hr hp₁ hp₂ hl₁ h_abs_m,
  sorry -- Proof skipped
end

/-- Compute the perpendicular distance of (p, q) from the line y = mx + b -/
def dist_from_line (line : ℝ × ℝ) (point : ℝ × ℝ) : ℝ :=
  let (m, b) := line in
  let (x, y) := point in
  abs (m * x - y + b) / sqrt (m ^ 2 + 1)

end slope_of_dividing_line_l675_675723


namespace total_four_digit_numbers_l675_675458

def combination (n k : ℕ) : ℕ := n.choose k
def arrangement (n k : ℕ) : ℕ := n.perm k

def count_four_digit_numbers : ℕ :=
  let c1 := combination 5 2
  let c2 := combination 4 2
  let c2_with_0 := combination 3 1 * combination 4 1
  let a := arrangement 4 4
  let a_without_0 := arrangement 3 3
  c1 * (c2 * a + c2_with_0 * a_without_0)

theorem total_four_digit_numbers : count_four_digit_numbers = 1260 := by
  sorry

end total_four_digit_numbers_l675_675458


namespace string_length_proof_l675_675766

noncomputable def string_length (circumference height loops : ℝ) : ℝ := 
  let height_per_loop := height / loops
  let hypotenuse := Math.sqrt (height_per_loop^2 + circumference^2)
  loops * hypotenuse

theorem string_length_proof :
  string_length 6 20 5 = 10 * Real.sqrt 13 :=
by
  rw [string_length]
  have height_per_loop := 20 / 5
  rw [Real.sqrt_eq_rpow, Real.sqrt_eq_rpow, Real.sqrt_eq_rpow]
  calc
    5 * Real.sqrt ((height_per_loop ^ 2) + 6 ^ 2)
    = 5 * Real.sqrt ((4 : ℝ) ^ 2 + 6 ^ 2)  : by {norm_num, exact congr_arg2 _ rfl rfl}
    = 5 * Real.sqrt (16 + 36)              : by {norm_num}
    = 5 * Real.sqrt (52)                   : by {norm_num}
    = 5 * Real.sqrt (4 * 13)               : by {norm_num}
    = 5 * (2 * Real.sqrt 13)               : by {rw Real.sqrt_mul, norm_num}
    = 10 * Real.sqrt 13                    : by ring
  
  sorry -- placeholder for the proof steps

end string_length_proof_l675_675766


namespace area_of_triangle_is_sum_of_roots_l675_675050

noncomputable def radius := 5
noncomputable def a := 675
noncomputable def b := 612.5

theorem area_of_triangle_is_sum_of_roots :
  let ω_1 := circle (0, 0) radius,
      ω_2 := circle (10, 0) radius,
      ω_3 := circle (5, 5 * Real.sqrt 3) radius,
      Q_1 := (5, -5 * Real.sqrt 3 / 3),
      Q_2 := (-5, -5 * Real.sqrt 3 / 3),
      Q_3 := (0, 10 * Real.sqrt 3 / 3) in
  are_points_of_triangle (* Points lie and form equilateral triangle with specified conditions. *)
  tangent_to_circles Q_1 Q_2 Q_3 ω_1 ω_2 ω_3 → 
  area_of_triangle Q_1 Q_2 Q_3 = Real.sqrt a + Real.sqrt b :=
sorry

end area_of_triangle_is_sum_of_roots_l675_675050


namespace collinear_K_L_T_l675_675367

-- Define the points and conditions given in the problem
variables {A B C B1 K L T : Point}
variable [triangle ABC]
variable [Circumcircle ABC]

-- Conditions
variable (isAngleBisector : AngleBisector B1 B A C)
variable (perpendicular_B1K_BC : Perpendicular B1 K B C)
variable (K_on_arcBC : OnSmallerArc K B C)
variable (perpendicular_BL_AK : Perpendicular B L A K)
variable (L_on_AC : OnLineSegment L A C)
variable (T_on_arcAC : OnArc T A C)

-- Prove that K, L, T are collinear
theorem collinear_K_L_T
  (isAngleBisector : AngleBisector B1 B A C)
  (perpendicular_B1K_BC : Perpendicular B1 K B C)
  (K_on_arcBC : OnSmallerArc K B C)
  (perpendicular_BL_AK : Perpendicular B L A K)
  (L_on_AC : OnLineSegment L A C)
  (T_on_arcAC : OnArc T A C) :
  Collinear K L T :=
sorry

end collinear_K_L_T_l675_675367


namespace smallest_k_for_polynomial_identity_l675_675960

/-- Let n be a positive even integer.
    There exists a positive integer k such that k = f(x)(x+1)^n + g(x)(x^n + 1),
    where f(x) and g(x) are polynomials with integer coefficients.
    If k_0 denotes the smallest such k, express k_0 as a function of n. -/

theorem smallest_k_for_polynomial_identity (n : ℕ) (hn_even : n % 2 = 0) :
  ∃ (k : ℕ) (f g : polynomial ℤ),
  k = f * (polynomial.X + 1) ^ n + g * (polynomial.X ^ n + 1) ∧
  ∀ k' (f' g' : polynomial ℤ),
  (k' = f' * (polynomial.X + 1) ^ n + g' * (polynomial.X ^ n + 1) → k' ≥ k) ∧
  k = 2 ^ (nat.log 2 n).ceil :=
sorry

end smallest_k_for_polynomial_identity_l675_675960


namespace problem_statement_l675_675473

open BigOperators

-- Definition of the sequence {a_n} based on given conditions
def a (n : ℕ) : ℕ := 2 * n + 1

-- Sum of the first n terms in the sequence {a_n}
def S (n : ℕ) : ℕ := n^2 + 2 * n

-- Definition of sequence {b_n} based on {a_n}
def b (n : ℕ) : ℝ := 1 / (a n)^2 - 1

-- Sum of the first n terms of sequence {b_n}
def T (n : ℕ) : ℝ :=
  (1 / 4) * ∑ i in Finset.range n, (1 : ℝ) / ↑i * (i + 1)

-- The problem's main statement
theorem problem_statement (n : ℕ) : 
  (∀ n, a 3 = 7 ∧ a 5 + a 7 = 26) →
  (∀ n, S n = n^2 + 2 * n) →
  (∀ n, T n = (n : ℝ) / (4 * (n + 1))) := by
  sorry

end problem_statement_l675_675473


namespace find_a_l675_675934

theorem find_a (a : ℝ) :
  (∃ l : ℝ → ℝ, (∀ x, l x = -(1 / 2) * x + (3 / 2)) ∧ (∀ x y, (x + 1)^2 + (y - 2)^2 = 5 → l (1) = 1)) ∧
  (∀ x : ℝ, ax + y - 1 = 0 → ∃ l : ℝ → ℝ, l x = -a) ∧
  (∀ x, -a = - (1 / 2))
  → a = 1 / 2 :=
begin
  sorry
end

end find_a_l675_675934


namespace polynomials_satisfy_eq_l675_675823

theorem polynomials_satisfy_eq (x y z w : ℝ) (P Q : ℝ → ℝ → ℝ → ℝ → ℝ) :
  (P = λ x y z w,  x y + x^2 - z + w ∨ P = λ x y z w, -(x y + x^2 - z + w)) ∧
  (Q = λ x y z w, x + y ∨ Q = λ x y z w, -(x + y)) ∧
  ((x * y + z + w)^2 - (x^2 - 2 * z) * (y^2 - 2 * w)) =
  (P x y z w)^2 - (x^2 - 2 * z) * (Q x y z w)^2 :=
sorry

end polynomials_satisfy_eq_l675_675823


namespace simplify_expression_l675_675738

theorem simplify_expression (a : ℝ) : 
  ( (a^(16 / 8))^(1 / 4) )^3 * ( (a^(16 / 4))^(1 / 8) )^3 = a^3 := by
  sorry

end simplify_expression_l675_675738


namespace percent_calculation_l675_675525

theorem percent_calculation (x : ℝ) (h : 0.40 * x = 160) : 0.30 * x = 120 :=
by
  sorry

end percent_calculation_l675_675525


namespace project_scientists_total_l675_675257

def total_scientists (S : ℕ) : Prop :=
  S / 2 + S / 5 + 21 = S

theorem project_scientists_total : ∃ S, total_scientists S ∧ S = 70 :=
by
  existsi 70
  unfold total_scientists
  sorry

end project_scientists_total_l675_675257


namespace total_surface_area_of_tower_l675_675653

def volume_to_side_length (v : ℕ) : ℕ :=
  nat.cbrt v

def surface_area (s : ℕ) : ℕ :=
  6 * s^2

def adjusted_surface_area (s : ℕ) : ℕ :=
  if s > 1 then surface_area s - s^2 else surface_area s

theorem total_surface_area_of_tower :
  let side_lengths := [7, 6, 5, 4, 3, 2, 1].map volume_to_side_length in
  let surface_areas := side_lengths.map adjusted_surface_area in
  surface_areas.sum = 701 :=
by
  sorry

end total_surface_area_of_tower_l675_675653


namespace pq_perpendicular_rs_l675_675563

variables {A B C D M P Q R S : Type*}
[metric_space A] [metric_space B] [metric_space C] [metric_space D] 
[metric_space M] [metric_space P] [metric_space Q] [metric_space R] [metric_space S]
[convex_quadrilateral A B C D]  -- Assumes ABCD is a convex quadrilateral

def intersect_diagonals (A B C D M : Type*) : Prop :=
∀ (A B C D M : Type*), is_diagonal_intersection A B C D M

def centroid (X Y Z G : Type*) [metric_space G] : Prop :=
is_centroid_triangle X Y Z G

def orthocenter (X Y Z H : Type*) [metric_space H] : Prop :=
is_orthocenter_triangle X Y Z H

theorem pq_perpendicular_rs
  (h1: intersect_diagonals A B C D M)
  (h2: centroid A M D P)
  (h3: centroid C M B Q)
  (h4: orthocenter D M C R)
  (h5: orthocenter M A B S):
  PQ ⊥ RS :=
sorry

end pq_perpendicular_rs_l675_675563


namespace k_value_range_l675_675492

-- Definitions
def f (x : ℝ) (k : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- The theorem we are interested in
theorem k_value_range (k : ℝ) (h : ∀ x₁ x₂ : ℝ, (x₁ > 5 → x₂ > 5 → f x₁ k ≤ f x₂ k) ∨ (x₁ > 5 → x₂ > 5 → f x₁ k ≥ f x₂ k)) :
  k ≥ 40 :=
sorry

end k_value_range_l675_675492


namespace center_of_hyperbola_l675_675088

-- Define the given equation of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  ((3 * y - 6)^2 / 8^2) - ((4 * x - 5)^2 / 3^2) = 1

-- Prove that the center of the hyperbola is (5 / 4, 2)
theorem center_of_hyperbola :
  (∃ h k : ℝ, h = 5 / 4 ∧ k = 2 ∧ ∀ x y : ℝ, hyperbola_eq x y ↔ ((y - k)^2 / (8 / 3)^2 - (x - h)^2 / (3 / 4)^2 = 1)) :=
sorry

end center_of_hyperbola_l675_675088


namespace molly_total_swim_l675_675253

variable (meters_saturday : ℕ) (meters_sunday : ℕ)

theorem molly_total_swim (h1 : meters_saturday = 45) (h2 : meters_sunday = 28) : meters_saturday + meters_sunday = 73 := by
  sorry

end molly_total_swim_l675_675253


namespace woody_writing_time_l675_675588

open Real

theorem woody_writing_time (W : ℝ) 
  (h1 : ∃ n : ℝ, n * 12 = W * 12 + 3) 
  (h2 : 12 * W + (12 * W + 3) = 39) :
  W = 1.5 :=
by sorry

end woody_writing_time_l675_675588


namespace riddles_ratio_l675_675208

theorem riddles_ratio (Josh_riddles : ℕ) (Ivory_riddles : ℕ) (Taso_riddles : ℕ) 
  (h1 : Josh_riddles = 8) 
  (h2 : Ivory_riddles = Josh_riddles + 4) 
  (h3 : Taso_riddles = 24) : 
  Taso_riddles / Ivory_riddles = 2 := 
by sorry

end riddles_ratio_l675_675208


namespace geom_seq_b_lambda_range_l675_675472

noncomputable def a : ℕ → ℕ
| 1     := 2
| (n+1) := if n % 2 = 0 then a n + 1 else 2 * a n

def b (n : ℕ) : ℕ := a (2 * n) + 1

def c : ℕ → ℚ
| 1     := 3 / 2
| (n+1) := c n - (Real.log (a (2 * n - 1) + 2)).toRat / b n

theorem geom_seq_b : ∀ n : ℕ, b (n + 1) = 2 * b n := 
sorry

theorem lambda_range (λ : ℝ) (h : ∀ n : ℕ, λ + (3 * n + 13) / (2 ^ n) ≥ 4 * (c n).toReal) : 
λ ∈ Icc (1 / 64) ⊤ := 
sorry

end geom_seq_b_lambda_range_l675_675472


namespace num_typing_orders_l675_675191

theorem num_typing_orders (k : ℕ) (S : finset ℕ) :
  S ⊆ {1, 2, 3, 4, 5, 6, 7, 8} ∧ (9 ∉ S) ∧ (10 ∉ S) →
  ∑ k in finset.range 9, (nat.choose 8 k * (k + 2)) = 1280 := 
sorry

end num_typing_orders_l675_675191


namespace floor_square_eq_l675_675369

noncomputable def recurrence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a n / n + n / a n

theorem floor_square_eq {a : ℕ → ℝ} (h : ∀ n : ℕ, a (n + 1) = recurrence a n) :
  ∀ n : ℕ, n ≥ 4 → ⌊(a n) ^ 2⌋ = n := by
  intro n hn
  sorry

end floor_square_eq_l675_675369


namespace value_of_a2017_l675_675138

noncomputable def f : ℝ → ℝ := sorry

def sequence_a : ℕ → ℝ
| 0 := f 1
| (n+1) := if n > 0 then 2 * (sequence_a n) + 1 else f 1

theorem value_of_a2017 : sequence_a 2016 = 2^2016 - 1 :=
by sorry

end value_of_a2017_l675_675138


namespace probability_red_or_black_probability_red_black_or_white_l675_675564

-- We define the probabilities of events A, B, and C
def P_A : ℚ := 5 / 12
def P_B : ℚ := 1 / 3
def P_C : ℚ := 1 / 6

-- Define the probability of event D for completeness
def P_D : ℚ := 1 / 12

-- 1. Statement for the probability of drawing a red or black ball (P(A ⋃ B))
theorem probability_red_or_black :
  (P_A + P_B = 3 / 4) :=
by
  sorry

-- 2. Statement for the probability of drawing a red, black, or white ball (P(A ⋃ B ⋃ C))
theorem probability_red_black_or_white :
  (P_A + P_B + P_C = 11 / 12) :=
by
  sorry

end probability_red_or_black_probability_red_black_or_white_l675_675564


namespace product_positive_intervals_l675_675456

theorem product_positive_intervals (x : ℝ) : 
  ((x > -3) ∧ (x < -1) ∨ (x > 1)) ↔ (x+1) * (x-1) * (x+3) > 0 :=
begin
  sorry
end

end product_positive_intervals_l675_675456


namespace HarrysMondaySpeed_l675_675902

/-- Define the speeds for various days -/
def S_Friday : ℕ := 24
def increase_60 (S_Thursday : ℕ) : Prop := 1.60 * S_Thursday = S_Friday
def increase_50 (S_Monday S_Thursday : ℕ) : Prop := 1.50 * S_Monday = S_Thursday

theorem HarrysMondaySpeed : ∃ S_Monday : ℕ, increase_60 (S_Friday / 1.60) ∧ increase_50 S_Monday (S_Friday / 1.60) ∧ S_Monday = 10 := 
sorry

end HarrysMondaySpeed_l675_675902


namespace trig_identity_l675_675063

theorem trig_identity :
  ∀ (θ : ℝ),
    θ = 70 * (π / 180) →
    (1 / Real.cos θ - (Real.sqrt 3) / Real.sin θ) = Real.sec (20 * (π / 180)) ^ 2 :=
by
  sorry

end trig_identity_l675_675063


namespace equilateral_triangle_ad_eq_be_l675_675718

open_locale euclidean_geometry

theorem equilateral_triangle_ad_eq_be 
  (A B C D E : Point) 
  (hABC : equilateral_triangle A B C)
  (hD : collinear A B D ∧ between B A D)
  (hE : collinear C B E ∧ between B C E)
  (hCD_DE : dist C D = dist D E) :
  dist A D = dist B E :=
begin
  -- Proof goes here
  sorry
end

end equilateral_triangle_ad_eq_be_l675_675718


namespace exists_infinitely_many_n_with_exactly_i_sum_of_cubes_l675_675657

theorem exists_infinitely_many_n_with_exactly_i_sum_of_cubes (i : ℕ):
  i ∈ {1, 2, 3} →
  ∃ (A : ℕ → Prop), (∀ n, A n → ∃ᶠ n in at_top, A n) ∧
    ∀ n, A n → (nat.sum_of_three_cubes n :
    count_terms (λ m, nat.sum_of_three_cubes m) [n, n + 2, n + 28] = i) :=
sorry

end exists_infinitely_many_n_with_exactly_i_sum_of_cubes_l675_675657


namespace solution_set_l675_675515

theorem solution_set (x y : ℝ) : 
  x^5 - 10 * x^3 * y^2 + 5 * x * y^4 = 0 ↔ 
  x = 0 
  ∨ y = x / Real.sqrt (5 + 2 * Real.sqrt 5) 
  ∨ y = x / Real.sqrt (5 - 2 * Real.sqrt 5) 
  ∨ y = -x / Real.sqrt (5 + 2 * Real.sqrt 5) 
  ∨ y = -x / Real.sqrt (5 - 2 * Real.sqrt 5) := 
by
  sorry

end solution_set_l675_675515


namespace farmer_crops_saved_l675_675775

noncomputable def average_corn_per_row := (10 + 14) / 2
noncomputable def average_potato_per_row := (35 + 45) / 2
noncomputable def average_wheat_per_row := (55 + 65) / 2

noncomputable def avg_reduction_corn := (40 + 60 + 25) / 3 / 100
noncomputable def avg_reduction_potato := (50 + 30 + 60) / 3 / 100
noncomputable def avg_reduction_wheat := (20 + 55 + 35) / 3 / 100

noncomputable def saved_corn_per_row := average_corn_per_row * (1 - avg_reduction_corn)
noncomputable def saved_potato_per_row := average_potato_per_row * (1 - avg_reduction_potato)
noncomputable def saved_wheat_per_row := average_wheat_per_row * (1 - avg_reduction_wheat)

def rows_corn := 30
def rows_potato := 24
def rows_wheat := 36

noncomputable def total_saved_corn := saved_corn_per_row * rows_corn
noncomputable def total_saved_potatoes := saved_potato_per_row * rows_potato
noncomputable def total_saved_wheat := saved_wheat_per_row * rows_wheat

noncomputable def total_crops_saved := total_saved_corn + total_saved_potatoes + total_saved_wheat

theorem farmer_crops_saved : total_crops_saved = 2090 := by
  sorry

end farmer_crops_saved_l675_675775


namespace prove_K_equals_D_prove_L_from_third_arc_l675_675365

variable {ABC : Triangle}
variable {A B C : Point}
variable {a b c : ℝ}
variable (AB_eq_c : dist A B = c)
variable (BC_eq_a : dist B C = a)
variable (CA_eq_b : dist C A = b)

-- Additional point definitions
variable {D E F G H J K : Point}
variable {L : Point}
variable (AD_eq_x : dist A D = x)

-- Conditions for arcs marking points
variable (arc_1 : dist A E = dist A D ∧ E = arc_with_radius D A A E)
variable (arc_2 : dist C E = dist C F ∧ F = arc_with_radius D C C E)
variable (arc_3 : dist B F = dist B G ∧ G = arc_with_radius D B B F)
variable (arc_4 : dist A G = dist A H ∧ H = arc_with_radius D A A G)
variable (arc_5 : dist C H = dist C J ∧ J = arc_with_radius D C C H)
variable (arc_6 : dist B J = dist B K ∧ K = arc_with_radius D B B J)

-- Midpoint of segment DG
variable (D_to_G : D = segment A B G)
variable (midpoint_L : L = midpoint D G)

-- Theorem statements
theorem prove_K_equals_D :
  K = D := sorry

theorem prove_L_from_third_arc :
  L = midpoint_segment_L_after_third_arc :=
  sorry

end prove_K_equals_D_prove_L_from_third_arc_l675_675365


namespace bricks_lay_calculation_l675_675907

theorem bricks_lay_calculation (b c d : ℕ) (h1 : 0 < c) (h2 : 0 < d) : 
  ∃ y : ℕ, y = (b * (b + d) * (c + d))/(c * d) :=
sorry

end bricks_lay_calculation_l675_675907


namespace family_reunion_attendance_l675_675031

-- Define the conditions
def male_adults : ℕ := 100
def female_adults : ℕ := male_adults + 50
def total_adults : ℕ := male_adults + female_adults
def children : ℕ := 2 * total_adults

-- State the theorem to be proven
theorem family_reunion_attendance : 
  let total_people := total_adults + children in
  total_people = 750 :=
by 
  sorry

end family_reunion_attendance_l675_675031


namespace arithmetic_sequence_general_formula_sum_of_first_n_terms_l675_675481

theorem arithmetic_sequence_general_formula (a : ℕ → ℚ) (h1 : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h2 : a 2 * a 2 - 5 * a 2 + 6 = 0) (h3 : a 4 * a 4 - 5 * a 4 + 6 = 0) :
  ∀ n, a n = (n + 2) / 2 := by
  sorry

theorem sum_of_first_n_terms (b : ℕ → ℚ) (h : ∀ n, b n = ((n + 2) / 2) / 2^(n - 1))
  (h1 : ∀ n, T n = ∑ i in finset.range n, b i) :
  T n = 4 - (n + 4) / 2^n := by
  sorry

end arithmetic_sequence_general_formula_sum_of_first_n_terms_l675_675481


namespace value_of_m_l675_675184

theorem value_of_m (m : ℝ) :
  let p := (m * x^2 - 3 * x) * (x^2 - 2 * x - 1) in
  (∀ x : ℝ, ∀ c : ℝ, (((m * x^2 - 3 * x) * (x^2 - 2 * x - 1)).coeff 3) = 0) →
  m = -3/2 :=
by
  sorry

end value_of_m_l675_675184


namespace calculate_g_16_l675_675681

noncomputable def a : ℝ := sorry
noncomputable def m : ℝ := (1 / 2 : ℝ)

-- Conditions
axiom a_pos : a > 0
axiom a_ne_one : a ≠ 1
axiom fixed_point : ∀ x, x = 2 → log a (x - 1) + 4 = 4
axiom power_function_point : ∀ x, (x = 4) → x^m = 2

-- Question proof
theorem calculate_g_16 : (16:ℝ)^m = 4 := by
  -- Proof would go here
  sorry

end calculate_g_16_l675_675681


namespace distribution_ways_l675_675037

theorem distribution_ways (n : ℕ) :
  let pieces := {1, 2, ..., 2*n} in
  let red_box := set.univ in     -- assume universally defined boxes
  let blue_box := set.univ in    -- assume universally defined boxes
  (∀ k : ℕ, (1 ≤ k ∧ k ≤ n) → k ∈ pieces ∧ 2*k ∈ pieces ∧ 
  (k ∈ red_box ↔ 2*k ∈ blue_box)) →
  (number_of_ways = 2 ^ n) :=
sorry

end distribution_ways_l675_675037


namespace line_PQ_passes_through_fixed_point_l675_675506

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 2 = 1

-- Define the conditions for points P and Q on the hyperbola
def on_hyperbola (P Q : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2 ∧ hyperbola Q.1 Q.2

-- Define the condition for perpendicular lines, given points A, P, and Q
def perpendicular (A P Q : ℝ × ℝ) : Prop :=
  ((P.2 - A.2) / (P.1 - A.1)) * ((Q.2 - A.2) / (Q.1 - A.1)) = -1

-- Define the main theorem to prove
theorem line_PQ_passes_through_fixed_point :
  ∀ (P Q : ℝ × ℝ), on_hyperbola P Q → perpendicular ⟨-1, 0⟩ P Q →
    ∃ (b : ℝ), ∀ (y : ℝ), (P.1 = y * P.2 + b ∨ Q.1 = y * Q.2 + b) → (b = 3) :=
by
  sorry

end line_PQ_passes_through_fixed_point_l675_675506


namespace find_a_l675_675885

def f (a x : ℝ) : ℝ := 
if x < 1 then 2^x + 1 else x^2 + a*x

theorem find_a :
  let a := 2 in 
  (f a (f a 0) = 4 * a) := by
  sorry

end find_a_l675_675885


namespace triangle_right_angle_AB_solution_l675_675202

theorem triangle_right_angle_AB_solution (AC BC AB : ℝ) (hAC : AC = 6) (hBC : BC = 8) :
  (AC^2 + BC^2 = AB^2 ∨ AB^2 + AC^2 = BC^2) ↔ (AB = 10 ∨ AB = 2 * Real.sqrt 7) :=
by
  sorry

end triangle_right_angle_AB_solution_l675_675202


namespace find_a_l675_675870

theorem find_a (x y a : ℤ) (h₁ : x = 1) (h₂ : y = -1) (h₃ : 2 * x - a * y = 3) : a = 1 :=
sorry

end find_a_l675_675870


namespace find_a9_l675_675169

noncomputable def polynomial_coefficients : Prop :=
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℤ),
  ∀ (x : ℤ),
    x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 =
    a₀ + a₁ * (1 + x) + a₂ * (1 + x)^2 + a₃ * (1 + x)^3 + a₄ * (1 + x)^4 + 
    a₅ * (1 + x)^5 + a₆ * (1 + x)^6 + a₇ * (1 + x)^7 + a₈ * (1 + x)^8 + 
    a₉ * (1 + x)^9 + a₁₀ * (1 + x)^10

theorem find_a9 (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℤ) (h : polynomial_coefficients) : a₉ = -9 := by
  sorry

end find_a9_l675_675169


namespace origin_inside_circle_range_l675_675134

theorem origin_inside_circle_range (m : ℝ) :
  ((0 - m)^2 + (0 + m)^2 < 8) → (-2 < m ∧ m < 2) :=
by
  intros h
  sorry

end origin_inside_circle_range_l675_675134


namespace max_value_of_n_l675_675112

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (S_sum : ∀ n, S n = n * (a 1 + a n) / 2)
variable (S_2015_pos : S 2015 > 0)
variable (S_2016_neg : S 2016 < 0)

theorem max_value_of_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (S_sum : ∀ n, S n = n * (a 1 + a n) / 2)
  (S_2015_pos : S 2015 > 0)
  (S_2016_neg : S 2016 < 0) : 
  ∃ n, n = 1008 ∧ ∀ m, S m < S n := 
sorry

end max_value_of_n_l675_675112


namespace probability_of_sum_5_l675_675278

namespace DieProblem

-- Define the sample space for a single die roll
def roll_outcomes : Finset (ℕ × ℕ) := (Finset.finRange 6).product (Finset.finRange 6)

-- Define event A: the sum of the rolls is 5
def event_A (outcome : ℕ × ℕ) : Prop := outcome.fst + outcome.snd = 5

-- Finset of outcomes where event_A holds
def outcomes_A : Finset (ℕ × ℕ) := roll_outcomes.filter event_A

-- Probability of event_A
def probability_A : ℚ := (outcomes_A.card : ℚ) / (roll_outcomes.card : ℚ)

theorem probability_of_sum_5 :
  probability_A = 1 / 9 := by
  -- Proof goes here
  sorry

end DieProblem

end probability_of_sum_5_l675_675278


namespace carol_additional_miles_carol_total_fuel_cost_l675_675430

structure RoadTrip where
  total_miles : ℕ
  highway_mileage : ℕ
  city_mileage : ℕ
  city_miles : ℕ
  total_gallons : ℕ
  gas_price : ℝ

def carol_trip : RoadTrip := {
  total_miles := 450,
  highway_mileage := 30,
  city_mileage := 25,
  city_miles := 50,
  total_gallons := 18,
  gas_price := 3.50
}

theorem carol_additional_miles : 
  let highway_miles := carol_trip.total_miles - carol_trip.city_miles,
      gallons_highway := highway_miles / carol_trip.highway_mileage,
      gallons_city := carol_trip.city_miles / carol_trip.city_mileage,
      total_gallons_used := gallons_highway + gallons_city,
      remaining_gallons := carol_trip.total_gallons - total_gallons_used,
      additional_miles := remaining_gallons * carol_trip.highway_mileage 
  in additional_miles = 80.1 :=
sorry

theorem carol_total_fuel_cost : 
  let total_cost := carol_trip.total_gallons * carol_trip.gas_price
  in total_cost = 63 :=
sorry

end carol_additional_miles_carol_total_fuel_cost_l675_675430


namespace unit_digit_of_product_eight_consecutive_l675_675349

theorem unit_digit_of_product_eight_consecutive (n : ℕ) : 
  (∏ i in finset.range 8, n + i) % 10 = 0 :=
sorry

end unit_digit_of_product_eight_consecutive_l675_675349


namespace least_n_value_l675_675126

open Nat

theorem least_n_value (n : ℕ) (h : 1 / (n * (n + 1)) < 1 / 15) : n = 4 :=
sorry

end least_n_value_l675_675126


namespace assign_tasks_l675_675634

theorem assign_tasks (n : ℕ) (cook : ℕ) (set_table : ℕ) (clean : ℕ) 
  (h_n : n = 5) (h_cook : cook = 2) (h_set_table : set_table = 1) (h_clean : clean = 2) :
  choose n cook * choose (n - cook) set_table * choose (n - cook - set_table) clean = 30 :=
by {
  rw [h_n, h_cook, h_set_table, h_clean],
  norm_num,
  sorry -- Placeholder for proof steps
}

end assign_tasks_l675_675634


namespace max_volume_triangular_prism_l675_675673

theorem max_volume_triangular_prism :
  (∀ (ABC : Triangle) (S : Point), 
    ABC.is_equilateral →
    (S.distance_to ABC.A = 1) →
    (A.projection_onto SBC = ABC.orthocenter) →
    (volume (prism S ABC) ≤ 1 / 6)) :=
sorry

end max_volume_triangular_prism_l675_675673


namespace hyperbola_intersection_theorem_l675_675892

-- Step a): Definitions of given conditions

def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧  ( (x^2) / (a^2) - (y^2) / (b^2) = 1 )

def focus_and_asymptotes (x y : ℝ) (a b : ℝ) : Prop :=
  ((2: ℝ), (0: ℝ)) = (x, y) ∧ (sqrt 3 = b / a)

-- Using information from provided problem and solution

def hyperbola_proof :=
  ∃ (a b : ℝ), hyperbola a b 1 0 ∧ focus_and_asymptotes 2 0 a b

-- Step c): Translate proof goal to Lean 4 statement

lemma hyperbola_theorem : hyperbola_proof :=
begin
  unfold hyperbola hyperbola_proof,
  use 1,
  use real.sqrt 3,
  split,
  sorry, -- Prove the hyperbola equation
  sorry  -- Prove focus and asymptotes
end

-- For the second part, definitions based on part (2) conditions

def line (P Q : ℝ × ℝ) : Prop := (∃ k : ℝ, Q.2 = k * (Q.1 - P.1))

def condition_1 (M A B: ℝ × ℝ) (P Q : ℝ × ℝ): Prop := (M.1, M.2) = (A.1 + B.1)/2, (A.2 + B.2)/2

def condition_2 (PQ AB: ℝ × ℝ): Prop := ∃ k: ℝ, PQ = k*AB

-- Remaining condition to be proven
def condition_3 (A B M: ℝ × ℝ) : Prop := dist A M = dist B M

theorem hyperbola_intersection_theorem (P Q A B M : ℝ × ℝ)
  (h1: condition_1 M A B P Q) 
  (h2: condition_2 P Q A B):
  condition_3 A B M :=
begin
  sorry
end

end hyperbola_intersection_theorem_l675_675892


namespace find_a10_of_arithmetic_sequence_l675_675480

theorem find_a10_of_arithmetic_sequence (a : ℕ → ℚ)
  (h_seq : ∀ n : ℕ, ∃ d : ℚ, ∀ m : ℕ, a (n + m + 1) = a (n + m) + d)
  (h_a1 : a 1 = 1)
  (h_a4 : a 4 = 4) :
  a 10 = -4 / 5 :=
sorry

end find_a10_of_arithmetic_sequence_l675_675480


namespace jennifer_tanks_l675_675947

theorem jennifer_tanks (initial_tanks : ℕ) (fish_per_initial_tank : ℕ) (total_fish_needed : ℕ) 
  (additional_tanks : ℕ) (fish_per_additional_tank : ℕ) 
  (initial_calculation : initial_tanks = 3) (fish_per_initial_calculation : fish_per_initial_tank = 15)
  (total_fish_calculation : total_fish_needed = 75) (additional_tanks_calculation : additional_tanks = 3) :
  initial_tanks * fish_per_initial_tank + additional_tanks * fish_per_additional_tank = total_fish_needed 
  → fish_per_additional_tank = 10 := 
by sorry

end jennifer_tanks_l675_675947


namespace problem_1_problem_2_problem_3_l675_675845

-- Define first-order difference sequence
def delta_seq (a : ℕ → ℤ) (n : ℕ) : ℤ := a (n + 1) - a n

-- Define k-th order difference sequence
def delta_k (a : ℕ → ℤ) : Π (k n : ℕ), ℤ
| 0, n   => a n
| k+1, n => delta_seq (delta_k a k) n

-- Sequence given by a_n = n^2 + n
def a (n : ℕ) : ℤ := n^2 + n

-- Define the necessary conditions
axiom a_init (n : ℕ) : a n = n^2 + n

-- Formal statements of the problems as Lean theorems
theorem problem_1 : 
  delta_seq a 0 = 2 ∧ 
  (∀ n : ℕ, delta_seq a n - delta_seq a (n+1) = 0) := sorry

theorem problem_2 (a : ℕ → ℤ) (h1: a 1 = 1) (h2: ∀ n : ℕ, delta_k a 2 n  - delta_seq a (n+1) + a n  = -2^n) : 
  ∀ n : ℕ, a n = n * 2^(n-1) := sorry

theorem problem_3 (a : ℕ → ℤ) (h2: ∀ n : ℕ, a n = n * 2 ^ (n - 1)) : 
  ∃ b : ℕ → ℕ, ∀ n : ℕ, (Σ i in finset.range (n+1), b (i+1) * nat.choose n i) = n * 2 ^ (n - 1) := sorry

end problem_1_problem_2_problem_3_l675_675845


namespace alicia_local_tax_deduction_l675_675033

theorem alicia_local_tax_deduction :
  ∀ (hourly_wage_dollars : ℕ) (tax_rate : ℚ),
    hourly_wage_dollars = 25 ∧ tax_rate = 0.02 →
    ∃ (tax_deduction_cents : ℕ), tax_deduction_cents = 50 :=
by
  intros hourly_wage_dollars tax_rate h,
  cases h with wage_eq tax_rate_eq,
  use 50,
  rw [wage_eq, tax_rate_eq],
  -- Continue the proof steps here
  sorry

end alicia_local_tax_deduction_l675_675033


namespace container_and_ball_volume_proof_l675_675331

noncomputable def cylinder_volume (r h : ℝ) : ℝ :=
  π * r^2 * h

noncomputable def sphere_volume (r : ℝ) : ℝ :=
  (4 / 3) * π * r^3

theorem container_and_ball_volume_proof 
    (cylinder_radius : ℝ)
    (cylinder_height : ℝ)
    (ball_radius : ℝ)
    (cylinder_vol : ℝ)
    (ball_vol : ℝ) 
    (h₁ : cylinder_radius = 5)
    (h₂ : cylinder_height = 2)
    (h₃ : ball_radius = 3)
    (h₄ : cylinder_vol = cylinder_volume cylinder_radius cylinder_height)
    (h₅ : ball_vol = sphere_volume ball_radius) : 
  cylinder_vol = 50 * π ∧ ball_vol = 36 * π := 
by
  sorry

end container_and_ball_volume_proof_l675_675331


namespace subtraction_correct_l675_675667

theorem subtraction_correct : 900000009000 - 123456789123 = 776543220777 :=
by
  -- Placeholder proof to ensure it compiles
  sorry

end subtraction_correct_l675_675667


namespace no_such_polynomial_exists_l675_675658

-- Define the necessary conditions
def is_real_polynomial (P : ℝ[X]) : Prop := true
def positive_integer (n : ℤ) : Prop := n > 0

-- Main statement
theorem no_such_polynomial_exists :
  ¬∃ P : ℝ[X], (∀ n : ℤ, positive_integer n → P.eval n = real.cbrt (n^2 + 1)) :=
sorry

end no_such_polynomial_exists_l675_675658


namespace increase_in_share_l675_675777

/-- Given a total cost of a car, earnings from a car wash, number of friends initially sharing 
    the cost, and the number of friends after Brad leaves, prove the increase in the amount each 
    remaining friend has to pay. -/
theorem increase_in_share
  (total_cost : ℕ)
  (car_wash_earnings : ℕ)
  (initial_friends : ℕ)
  (remaining_friends : ℕ)
  (initial_share : ℕ := (total_cost - car_wash_earnings) / initial_friends)
  (remaining_share : ℕ := (total_cost - car_wash_earnings) / remaining_friends) :
  total_cost = 1700 →
  car_wash_earnings = 500 →
  initial_friends = 6 →
  remaining_friends = 5 →
  (remaining_share - initial_share = 40) :=
by
  intros h1 h2 h3 h4
  have h₀ : total_cost - car_wash_earnings = 1700 - 500 := by rw [h1, h2]
  have h₁ : initial_share = (1700 - 500) / 6 := by rw [h₀, h3]
  have h₂ : remaining_share = (1700 - 500) / 5 := by rw [h₀, h4]
  have h₃ : (1700 - 500) = 1200 := by norm_num
  have h₄ : initial_share = 1200 / 6 := by rw [h₃]
  have h₅ : remaining_share = 1200 / 5 := by rw [h₃]
  have h₆ : 1200 / 6 = 200 := by norm_num
  have h₇ : 1200 / 5 = 240 := by norm_num
  rw [h₄, h₅, h₆, h₇]
  norm_num
  sorry

end increase_in_share_l675_675777


namespace additional_payment_each_friend_l675_675779

theorem additional_payment_each_friend (initial_cost : ℕ) (earned_amount : ℕ) (total_friends : ℕ) (final_friends : ℕ) 
(h_initial_cost : initial_cost = 1700) (h_earned_amount : earned_amount = 500) 
(h_total_friends : total_friends = 6) (h_final_friends : final_friends = 5) : 
  ((initial_cost - earned_amount) / total_friends) / final_friends = 40 :=
sorry

end additional_payment_each_friend_l675_675779


namespace find_f_of_2_l675_675504

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x - b

theorem find_f_of_2 (a b : ℝ) (h_pos : 0 < a)
  (h1 : ∀ x : ℝ, a * f x a b - b = 4 * x - 3)
  : f 2 a b = 3 := 
sorry

end find_f_of_2_l675_675504


namespace ice_cream_flavors_l675_675905

theorem ice_cream_flavors :
  ∃ n : ℕ, n = (Nat.choose 7 2) ∧ n = 21 :=
by
  use Nat.choose 7 2
  split
  · rfl
  sorry

end ice_cream_flavors_l675_675905


namespace angle_between_a_b_is_pi_over_4_l675_675133

variables {V : Type*} [inner_product_space ℝ V]

-- Definitions corresponding to conditions
variables (a b : V)
hypothesis (ha : ∥a∥ = 1)
hypothesis (hb : ∥b∥ = real.sqrt 2)
hypothesis (h_perp : inner_product_space.inner a (a - b) = 0)

-- The correct answer to be proven
theorem angle_between_a_b_is_pi_over_4 : real.arccos (inner_product_space.inner a b / (∥a∥ * ∥b∥)) = real.pi / 4 :=
sorry

end angle_between_a_b_is_pi_over_4_l675_675133


namespace find_average_pages_per_day_l675_675362

theorem find_average_pages_per_day :
    ∀ (total_pages first_five_days_pages remaining_days : ℕ), total_pages = 400 ∧ first_five_days_pages = 100 ∧ remaining_days = 5 →
    (total_pages - first_five_days_pages) / remaining_days = 60 :=
begin
    -- let's denote the given values
    intros total_pages first_five_days_pages remaining_days h,
    -- split the conjunction
    cases h with h1 h2,
    cases h2 with h3 h4,
    -- replace the given constants
    rw [h1, h3, h4],
    -- simplifying
    norm_num,
end

end find_average_pages_per_day_l675_675362


namespace triangle_BC_length_l675_675556

-- Define the triangle and given angles and side lengths
def Triangle (A B C : Type) := {
  angle_B : ℝ,
  side_AC : ℝ,
  side_AB : ℝ
}

-- Prove that the length of side BC is 3 given the conditions
theorem triangle_BC_length (A B C : Type)
  [Triangle A B C] (h₁ : A.angle_B = 120)
  (h₂ : A.side_AC = sqrt 19) (h₃ : A.side_AB = 2) :  
  ∃ (BC : ℝ), BC = 3 :=
by
  sorry

end triangle_BC_length_l675_675556


namespace corresponding_angles_equal_l675_675276

variable {α : Type}
variables (A B : α) [angle : has_measure α (angle_measure α)]
variables (h : A.is_corresponding_with B)

theorem corresponding_angles_equal (A B : α) [angle A] [angle B] :
  A.is_corresponding_with B → A = B :=
by
  sorry

end corresponding_angles_equal_l675_675276


namespace min_value_of_expression_l675_675153

theorem min_value_of_expression (a : ℝ) (h₀ : a > 0)
  (x₁ x₂ : ℝ)
  (h₁ : x₁ + x₂ = 4 * a)
  (h₂ : x₁ * x₂ = a * a) :
  x₁ + x₂ + a / (x₁ * x₂) = 4 :=
sorry

end min_value_of_expression_l675_675153


namespace find_zero_range_l675_675147

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  k * 4^x - k * 2^(x + 1) - 4 * (k + 5)

theorem find_zero_range (k : ℝ) :
  (∃ x ∈ set.Icc (0 : ℝ) 2, f x k = 0) :=
sorry

end find_zero_range_l675_675147


namespace parabola_eq_l675_675573

theorem parabola_eq 
  (h1 : ∃ a : ℝ, ∀ x y : ℝ, y^2 = a * x ∧ y^2 = y^2 ∧ x = x)
  (h2 : ∀ x y : ℝ, (x = 0 ∧ y = 0) → (0 = 0))
  (h3 : (2:ℝ, 4:ℝ) ∈ { p : ℝ × ℝ | ∃ a : ℝ, (p.2)^2 = a * p.1 })
:  ∃ a : ℝ, a = 8 ∧ ∀ x y : ℝ, y^2 = a * x :=
by
  sorry

end parabola_eq_l675_675573


namespace double_burger_cost_l675_675426

theorem double_burger_cost :
  ∃ (D : ℝ), 
  let single_burger_cost := 1.00 in
  let total_burgers := 50 in
  let double_burgers := 33 in
  let single_burgers := total_burgers - double_burgers in
  let total_cost := 66.50 in
  let single_burgers_cost := single_burgers * single_burger_cost in
  let double_burgers_cost := double_burgers * D in
  single_burgers_cost + double_burgers_cost = total_cost ∧ D = 1.50 :=
begin
  sorry
end

end double_burger_cost_l675_675426


namespace cosine_beta_l675_675122

-- Let α and β be real numbers satisfying the given conditions.
variables {α β : ℝ}
-- The conditions as specified in the problem.
variables (h1 : 0 < α ∧ α < π / 2 ∧ π / 2 < β ∧ β < π) 
variables (h2 : cos α = 3 / 5) 
variables (h3 : sin (α + β) = -3 / 5)

-- The statement that needs to be proven.
theorem cosine_beta (h1 : 0 < α ∧ α < π / 2 ∧ π / 2 < β ∧ β < π)
  (h2 : cos α = 3 / 5)
  (h3 : sin (α + β) = -3 / 5) : cos β = -24 / 25 := 
sorry

end cosine_beta_l675_675122


namespace find_x_perpendicular_l675_675511

-- Define vectors a and b
def a : ℝ × ℝ × ℝ := (-3, 2, 5)
def b (x : ℝ) : ℝ × ℝ × ℝ := (1, x, -1)

-- Define the perpendicular condition
def perpendicular (a b : ℝ × ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0

-- State the problem
theorem find_x_perpendicular (x : ℝ) (h : perpendicular a (b x)) : x = 4 := 
  sorry

end find_x_perpendicular_l675_675511


namespace at_least_one_not_less_than_six_l675_675132

-- Definitions for the conditions.
variables {a b c : ℝ}
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- The proof statement.
theorem at_least_one_not_less_than_six :
  (a + 4 / b) < 6 ∧ (b + 9 / c) < 6 ∧ (c + 16 / a) < 6 → false :=
by
  sorry

end at_least_one_not_less_than_six_l675_675132


namespace find_extrema_l675_675487

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

noncomputable def f (x : ℝ) : ℝ :=
  log_base (1/2) (2 * x) * log_base (1/2) (4 * x)

theorem find_extrema :
  (∀ x, x ∈ Icc (1/8 : ℝ) 1 → (-1/4 : ℝ) ≤ f x) ∧ 
  (∃ x, x ∈ Icc (1/8 : ℝ) 1 ∧ f x = -1/4) ∧ 
  (∀ x, x ∈ Icc (1/8 : ℝ) 1 → f x ≤ 2) ∧ 
  (∃ x, x ∈ Icc (1/8 : ℝ) 1 ∧ f x = 2) :=
by
  sorry

end find_extrema_l675_675487


namespace percent_voters_for_A_l675_675186

-- Definitions from conditions
def total_voters : ℕ := 100
def percent_democrats : ℝ := 0.70
def percent_republicans : ℝ := 0.30
def percent_dems_for_A : ℝ := 0.80
def percent_reps_for_A : ℝ := 0.30

-- Calculations based on definitions
def num_democrats := total_voters * percent_democrats
def num_republicans := total_voters * percent_republicans
def dems_for_A := num_democrats * percent_dems_for_A
def reps_for_A := num_republicans * percent_reps_for_A
def total_for_A := dems_for_A + reps_for_A

-- Proof problem statement
theorem percent_voters_for_A : (total_for_A / total_voters) * 100 = 65 :=
by
  sorry

end percent_voters_for_A_l675_675186


namespace min_dot_prod_OX_coords_cos_angle_AXB_l675_675201

-- Definitions of the given vectors
def OA : ℝ × ℝ := (1, 7)
def OB : ℝ × ℝ := (5, 1)
def OP : ℝ × ℝ := (2, 1)

-- 1. Prove the coordinates of OX when XA · XB is minimized are (4, 2)
theorem min_dot_prod_OX_coords : ∃ (x y : ℝ), x = 4 ∧ y = 2 ∧
  (let OX := (x, y)
   let XA := (1 - x, 7 - y)
   let XB := (5 - x, 1 - y)
   ((XA.1 * XB.1) + (XA.2 * XB.2)) = min (((OA.1 - x) * (OB.1 - x)) + ((OA.2 - y) * (OB.2 - y)))) :=
sorry

-- 2. Prove the value of cos(∠AXB) when OX = (4, 2) is -4√17/17
theorem cos_angle_AXB :
  let x := 4
  let y := 2
  let OX := (x, y)
  let XA := (1 - x, 7 - y)
  let XB := (5 - x, 1 - y)
  let dot_prod := (XA.1 * XB.1) + (XA.2 * XB.2)
  let XA_mag := Real.sqrt ((XA.1)^2 + (XA.2)^2)
  let XB_mag := Real.sqrt ((XB.1)^2 + (XB.2)^2)
  (dot_prod / (XA_mag * XB_mag)) = - (4 * (Real.sqrt 17) / 17) :=
sorry

end min_dot_prod_OX_coords_cos_angle_AXB_l675_675201


namespace passing_mark_is_200_l675_675382

variable (T : ℝ) (PassingMark : ℝ)

-- Conditions
def condition1 : Prop := 0.30 * T + 50 = PassingMark
def condition2 : Prop := 0.45 * T - 25 = PassingMark

-- Proof statement
theorem passing_mark_is_200 (h1 : condition1 T PassingMark) (h2 : condition2 T PassingMark) :
  PassingMark = 200 :=
by
  -- Proving T first based on the equations
  have h : 0.30 * T + 50 = 0.45 * T - 25 := by rw [h1, h2]
  have T_calc : T = 500 := by
    simp at h
    linarith
  -- Using T=500 to conclude PassingMark
  rw [T_calc] at h1
  exact h1.solver

end passing_mark_is_200_l675_675382


namespace maxWhitePieces_l675_675411

-- Definitions corresponding to problem conditions
def initialPieces : List Piece := [B, B, B, B, W]

inductive Piece
| W : Piece
| B : Piece

def transform (pieces : List Piece) : List Piece :=
  pieces.zip pieces.rotateLeft 1 >>= λ 
  | (Piece.B, Piece.B) => [Piece.B, Piece.W, Piece.B]
  | (Piece.W, Piece.W) => [Piece.B, Piece.W, Piece.B]
  | _ => [Piece.B, Piece.B, Piece.B]

noncomputable def iterateTransform : List Piece → ℕ → List Piece 
| pieces, 0 => pieces
| pieces, n + 1 => iterateTransform (transform pieces) n

theorem maxWhitePieces (initial : List Piece) : (initial = initialPieces) →
  (∀ n, (iterateTransform initial n).count Piece.W ≤ 3) :=
by
  sorry

end maxWhitePieces_l675_675411


namespace max_rock_value_l675_675354

def rock_value (weight_5 : Nat) (weight_4 : Nat) (weight_1 : Nat) : Nat :=
  14 * weight_5 + 11 * weight_4 + 2 * weight_1

def total_weight (weight_5 : Nat) (weight_4 : Nat) (weight_1 : Nat) : Nat :=
  5 * weight_5 + 4 * weight_4 + 1 * weight_1

theorem max_rock_value : ∃ (weight_5 weight_4 weight_1 : Nat), 
  total_weight weight_5 weight_4 weight_1 ≤ 18 ∧ 
  rock_value weight_5 weight_4 weight_1 = 50 :=
by
  -- We need to find suitable weight_5, weight_4, and weight_1.
  use 2, 2, 0 -- Example values
  apply And.intro
  -- Prove the total weight condition
  show total_weight 2 2 0 ≤ 18
  sorry
  -- Prove the value condition
  show rock_value 2 2 0 = 50
  sorry

end max_rock_value_l675_675354


namespace max_eccentricity_l675_675006

theorem max_eccentricity (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : ∀ x y, x^2/a^2 - y^2/b^2 = 1 → (3 * b - b * x ^ 2 / a ^ 2 >= b) ) : 
  ∃ e : ℝ, e = 3 ∧ eccentricity e = 3 := 
sorry

end max_eccentricity_l675_675006


namespace probability_divisible_by_25_l675_675539

theorem probability_divisible_by_25 :
  (let digits := [1, 1, 3, 5, 5, 5, 9] in
   let total_permutations := (7.factorial / (3.factorial)) in
   let valid_arrangements := (5.factorial / (2.factorial)) in
   valid_arrangements / total_permutations = 1 / 14) :=
sorry

end probability_divisible_by_25_l675_675539


namespace pyramid_volume_alpha_range_l675_675190

noncomputable def volume_pyramid (S α : ℝ) : ℝ :=
  (16 * S * Real.cot α * Real.sqrt(2 * S * Real.sin(α - π / 6) * Real.sin(α + π / 6))) / (3 * Real.sin (2 * α))

theorem pyramid_volume (S α : ℝ) (hS : 0 < S) (hα1 : π / 6 < α) (hα2 : α < π / 2) :
  volume_pyramid S α = (16 * S * Real.cot α * Real.sqrt(2 * S * Real.sin(α - π / 6) * Real.sin(α + π / 6))) / (3 * Real.sin (2 * α)) :=
sorry

theorem alpha_range (α : ℝ) (hα : π / 6 < α ∧ α < π / 2) :
  π / 6 < α ∧ α < π / 2 :=
begin
  exact hα,
end

end pyramid_volume_alpha_range_l675_675190


namespace biff_hourly_earnings_l675_675807

theorem biff_hourly_earnings:
  let ticket_cost := 11
  let drinks_snacks_cost := 3
  let headphones_cost := 16
  let wifi_cost_per_hour := 2
  let bus_ride_hours := 3
  let total_non_wifi_expenses := ticket_cost + drinks_snacks_cost + headphones_cost
  let total_wifi_cost := bus_ride_hours * wifi_cost_per_hour
  let total_expenses := total_non_wifi_expenses + total_wifi_cost
  ∀ (x : ℝ), 3 * x = total_expenses → x = 12 :=
by sorry -- Proof skipped

end biff_hourly_earnings_l675_675807


namespace intersection_perpendiculars_circumcircle_l675_675192

open_locale classical
noncomputable theory

variables {A B C D I J K M N : Type} [inscribed_quadrilateral A B C D]
variables [h1 : BC = CD] [h2 : AB ≠ AD]
variables [incenter_triangle ABC I] [incenter_triangle ACD J]
variables [K_on_AC : point_on_line_segment K A C] [IK_eq_JK : IK = JK]
variables [incenter_triangle AIK M] [incenter_triangle AJK N]

theorem intersection_perpendiculars_circumcircle :
  let l1 := perpendicular_through_point CD D,
      l2 := perpendicular_through_point KI I in
  ∃ P, intersection l1 l2 P ∧ on_circumcircle P MAN :=
sorry

end intersection_perpendiculars_circumcircle_l675_675192


namespace gold_coins_percentage_l675_675804

-- Definitions for conditions
def percent_beads : Float := 0.30
def percent_sculptures : Float := 0.10
def percent_silver_coins : Float := 0.30

-- Definitions derived from conditions
def percent_coins : Float := 1.0 - percent_beads - percent_sculptures
def percent_gold_coins_among_coins : Float := 1.0 - percent_silver_coins

-- Theorem statement
theorem gold_coins_percentage : percent_gold_coins_among_coins * percent_coins = 0.42 :=
by
sorry

end gold_coins_percentage_l675_675804


namespace money_given_to_cashier_l675_675001

theorem money_given_to_cashier (regular_ticket_cost : ℕ) (discount : ℕ) 
  (age1 : ℕ) (age2 : ℕ) (change : ℕ) 
  (h1 : regular_ticket_cost = 109)
  (h2 : discount = 5)
  (h3 : age1 = 6)
  (h4 : age2 = 10)
  (h5 : change = 74)
  (h6 : age1 < 12)
  (h7 : age2 < 12) :
  regular_ticket_cost + regular_ticket_cost + (regular_ticket_cost - discount) + (regular_ticket_cost - discount) + change = 500 :=
by
  sorry

end money_given_to_cashier_l675_675001


namespace average_age_of_girls_correct_l675_675926

noncomputable def average_age_of_girls (total_students : ℕ) (average_age_boys : ℝ) 
  (average_age_school : ℝ) (num_girls : ℕ) : ℝ := 
  let num_boys := total_students - num_girls
  let total_age_boys := num_boys * average_age_boys
  let total_age_school := total_students * average_age_school
  (total_age_school - total_age_boys) / num_girls

theorem average_age_of_girls_correct :
  ∀ (G : ℝ), 
    average_age_of_girls 652 12 11.75 163 = G → 
    G ≈ 10.94 :=
sorry

end average_age_of_girls_correct_l675_675926


namespace range_of_a_l675_675861

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then x^2 + 2 * a else -x

theorem range_of_a (a : ℝ) (h : a < 0) (hf : f a (1 - a) ≥ f a (1 + a)) : -2 ≤ a ∧ a ≤ -1 :=
  sorry

end range_of_a_l675_675861


namespace watch_correct_time_l675_675412

-- Conditions
def initial_time_slow : ℕ := 4 -- minutes slow at 8:00 AM
def final_time_fast : ℕ := 6 -- minutes fast at 4:00 PM
def total_time_interval : ℕ := 480 -- total time interval in minutes from 8:00 AM to 4:00 PM
def rate_of_time_gain : ℚ := (initial_time_slow + final_time_fast) / total_time_interval

-- Statement to prove
theorem watch_correct_time : 
  ∃ t : ℕ, t = 11 * 60 + 12 ∧ 
  ((8 * 60 + t) * rate_of_time_gain = 4) := 
sorry

end watch_correct_time_l675_675412


namespace probability_BD_greater_than_4sqrt2_l675_675720

def isoscelesRightTriangle (A B C : Point) : Prop :=
  ∠ACB = 90 ∧ Isosceles (A B C) ∧ AB = 10

def probabilityBDGreater (P : Point) (B : Point) (D : Point) : ℝ :=
  let prob := (5 * sqrt 2) - (2 * sqrt 17)
  prob / (5 * sqrt 2)

theorem probability_BD_greater_than_4sqrt2 
  {A B C P D : Point}
  (HABC : isoscelesRightTriangle A B C)
  (HP_in_triangle : InTriangle ABC P)
  (HBD : B != D) :
  probabilityBDGreater P B D = (5 * sqrt 2 - 2 * sqrt 17) / (5 * sqrt 2) :=
by
  sorry

end probability_BD_greater_than_4sqrt2_l675_675720


namespace monic_polynomial_degree_3_with_conditions_l675_675757

noncomputable def f (x : ℝ) : ℝ := x^3 + x^2 + 2 * x + 3

theorem monic_polynomial_degree_3_with_conditions :
  (f 0 = 3) ∧ (f 2 = 19) ∧ (degree (C 1 * X^3 + C 1 * X^2 + C 2 * X + C 3) = 3) :=
begin
  split,
  { -- Proof of f(0) = 3
    sorry },
  split,
  { -- Proof of f(2) = 19
    sorry },
  { -- Proof of monic polynomial of degree 3
    sorry },
end

end monic_polynomial_degree_3_with_conditions_l675_675757


namespace travel_A_to_D_l675_675815

-- Definitions for the number of roads between each pair of cities
def roads_A_to_B : ℕ := 3
def roads_A_to_C : ℕ := 1
def roads_B_to_C : ℕ := 2
def roads_B_to_D : ℕ := 1
def roads_C_to_D : ℕ := 3

-- Theorem stating the total number of ways to travel from A to D visiting each city exactly once
theorem travel_A_to_D : roads_A_to_B * roads_B_to_C * roads_C_to_D + roads_A_to_C * roads_B_to_C * roads_B_to_D = 20 :=
by
  -- Formal proof goes here
  sorry

end travel_A_to_D_l675_675815


namespace exists_positive_integer_N_l675_675236

theorem exists_positive_integer_N 
  (m : fin 2013 → ℕ)
  (hm_gt_one : ∀ i, 1 < m i)
  (hp_rel_prime : ∀ i j, i ≠ j → Nat.coprime (m i) (m j))
  (A : fin 2013 → set ℕ)
  (hA : ∀ i, A i ⊆ finset.range (m i) \ {0}) :
  ∃ N : ℕ,
    N ≤ finset.prod finset.univ (λ i, 2 * (A i).to_finset.card + 1) ∧
    ∀ i a, a ∈ A i → ¬(m i) ∣ (N - a) := 
by { sorry }

end exists_positive_integer_N_l675_675236


namespace max_area_of_2m_wire_l675_675527

theorem max_area_of_2m_wire (P : ℝ) (l w : ℝ) (a : ℝ) :
  P = 2 → 2 * (l + w) = 2 → (a = l * w → l + w = 1) → max l w = (1 / 4) :=
by
  sorry

end max_area_of_2m_wire_l675_675527


namespace plane_speed_against_tailwind_l675_675787

variable (V_p V_w V_gw V_gaw : ℝ)

theorem plane_speed_against_tailwind :
  ∀ (V_gw V_w : ℝ), V_gw = 460 → V_w = 75 → 
  let V_p := V_gw - V_w in 
  let V_gaw := V_p - V_w in 
  V_gaw = 310 :=
by 
  intros V_gw V_w h1 h2
  rw [h1, h2]
  let V_p := 460 - 75
  let V_gaw := V_p - 75
  have h3 : V_p = 385 := by norm_num
  subst h3
  have h4 : V_gaw = 310 := by norm_num
  exact h4

end plane_speed_against_tailwind_l675_675787


namespace intervals_sum_ge_k_squared_l675_675235

noncomputable def interval_length (I : Set ℝ) : ℝ := sorry

theorem intervals_sum_ge_k_squared (k : ℕ) (h_k : 1 ≤ k)
  (I : Fin k → Set ℝ) 
  (h_I : ∀ i, ∃ a b, a < b ∧ I i = Set.Icc a b) :
  (∑ i j in Finset.univ.product Finset.univ, 
    if (Set.Nonempty (I i ∩ I j)) then 1 / interval_length (I i ∪ I j) else 0) ≥ k^2 :=
by
  sorry

end intervals_sum_ge_k_squared_l675_675235


namespace mode_of_combined_set_l675_675338

theorem mode_of_combined_set 
  (x y : ℕ)
  (h1 : (3 + x + 2 * y + 5) / 4 = 6)
  (h2 : (x + 6 + y) / 3 = 6) :
  multiset.mode ({3, x, 2 * y, 5, x, 6, y} : multiset ℕ) = 8 := by
  sorry

end mode_of_combined_set_l675_675338


namespace problem_f_even_function_l675_675500

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.cos (2 * x + φ)

theorem problem_f_even_function (φ : ℝ) (h : ∀ x : ℝ, f x φ ≤ f 1 φ) : ∀ x : ℝ, f (x + 1) φ = f (-(x + 1)) φ :=
begin
  -- Proof is omitted
  sorry
end

end problem_f_even_function_l675_675500


namespace triangle_side_BC_l675_675549

theorem triangle_side_BC {A B C : Type} [inner_product_space ℝ A] 
  {b c BC: ℝ} 
  (hB : ∠ B = real.pi / 3) 
  (hAC : b = real.sqrt 19) 
  (hAB : c = 2) : 
  BC = 3 := 
sorry

end triangle_side_BC_l675_675549


namespace math_problem_correct_answers_l675_675740

theorem math_problem_correct_answers :
  ¬ (∀ x, x ≠ -1 → 
      (x ∈ Ioo (-∞) (-1) ∨ x ∈ Ioo (-1) ∞) → 
      (differentiable ℝ (λ x, (x+2)/(x+1)) x) ∧ 
      (deriv (λ x, (x+2)/(x+1)) x < 0)) ∧
  (∀ a : ℝ, a > 0 ∧ a ≠ 1 → 
      (f : ℝ → ℝ,
       f = λ x, a^(x-2023) + 1, 
       f 2023 = 2)) ∧
  (¬ (∃ x : ℝ, x^2 + a*x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 + a*x + 1 > 0)) ∧
  (¬ (∀ a : ℝ, (2 < a)  →
      (∀ x, (x ∈ Ioo (-∞) 1) →
       (differentiable ℝ (λ x, 2^(-x^2 + a*x)) x) ∧
       (deriv (λ x, 2^(-x^2 + a*x)) x > 0)))) := sorry

end math_problem_correct_answers_l675_675740


namespace problem_solution_l675_675160

noncomputable def sequence (a b : ℤ) : ℕ → ℤ
| 1       := a
| 2       := b
| (n + 1) := sequence b (sequence (n - 1) - sequence n)

theorem problem_solution (a b : ℤ) :
  let x : ℕ → ℤ := sequence a b
  in x 100 = -a ∧ (∑ i in finset.range 100, x (i + 1)) = 2 * b - a :=
begin
  sorry
end

end problem_solution_l675_675160


namespace theater_adult_charge_l675_675842

theorem theater_adult_charge :
  ∃ A : ℝ, let children_charge := 4.50, total_receipts := 405, children := 48 in
  let adults := children - 20 in
  let children_receipts := children * children_charge in
  let adults_receipts := adults * A in
  children_receipts + adults_receipts = total_receipts ∧ 
  A = 6.75 :=
begin
  sorry
end

end theater_adult_charge_l675_675842


namespace jackson_champagne_cost_l675_675589

def hotTubGallons : ℕ := 40
def quartsPerGallon : ℕ := 4
def bottlesPerQuart : ℕ := 1
def costPerBottle : ℕ := 50
def discount : ℕ := 20

theorem jackson_champagne_cost :
  let neededQuarts := hotTubGallons * quartsPerGallon;
  let totalCost := neededQuarts * costPerBottle;
  let discountAmount := (totalCost * discount) / 100;
  let finalCost := totalCost - discountAmount
  in finalCost = 6400 :=
by 
  sorry

end jackson_champagne_cost_l675_675589


namespace part1_part2_l675_675157

noncomputable theory
open Real

def line_equation (k : ℝ) : ℝ × ℝ → Prop :=
  λ p, k * p.1 - 2 * p.2 - 3 + k = 0

def not_in_second_quadrant (l : ℝ × ℝ → Prop) : Prop :=
  ¬ ∃ x y : ℝ, x < 0 ∧ y > 0 ∧ l (x, y)

theorem part1 (k : ℝ) :
  not_in_second_quadrant (line_equation k) ↔ 0 ≤ k ∧ k ≤ 3 :=
sorry

def area_triangle (A B : ℝ × ℝ) : ℝ :=
  1 / 2 * |A.1 * B.2|

theorem part2 (k : ℝ) :
  let A := (3 - k) / k;
  let B := (3 - k) / 2;
  area_triangle (A, 0) (0, B) = 4 →
  k = -1 → ∀ (x y : ℝ), (line_equation k (x, y)) ↔ x + 2 * y + 4 = 0 :=
sorry

end part1_part2_l675_675157


namespace right_triangle_correct_set_l675_675353

-- Problem: Which of the following sets of numbers can be the side lengths of a right triangle?
-- Conditions: We will check using the Pythagorean theorem for sets A, B, C, D

def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

def set_A := (0.1 : ℝ, 0.2 : ℝ, 0.3 : ℝ)
def set_B := (1 : ℝ, 1 : ℝ, 2 : ℝ)
def set_C := (10 : ℝ, 24 : ℝ, 26 : ℝ)
def set_D := (3^2 : ℝ, 4^2 : ℝ, 5^2 : ℝ)

theorem right_triangle_correct_set :
  is_right_triangle set_C.1 set_C.2 set_C.3 →
  ¬ (is_right_triangle set_A.1 set_A.2 set_A.3) →
  ¬ (is_right_triangle set_B.1 set_B.2 set_B.3) →
  ¬ (is_right_triangle set_D.1 set_D.2 set_D.3)
  := by
  sorry

end right_triangle_correct_set_l675_675353


namespace estimate_white_balls_l675_675930

-- Statements for conditions
variables (black_balls white_balls : ℕ)
variables (draws : ℕ := 40)
variables (black_draws : ℕ := 10)

-- Define total white draws
def white_draws := draws - black_draws

-- Ratio of black to white draws
def draw_ratio := black_draws / white_draws

-- Given condition on known draws
def black_ball_count := 4
def known_draw_ratio := 1 / 3

-- Lean 4 statement to prove the number of white balls
theorem estimate_white_balls (h : black_ball_count / white_balls = known_draw_ratio) : white_balls = 12 :=
sorry -- Proof omitted

end estimate_white_balls_l675_675930


namespace find_r_l675_675240

variable {x y r k : ℝ}

theorem find_r (h1 : y^2 + 4 * y + 4 + Real.sqrt (x + y + k) = 0)
               (h2 : r = |x * y|) :
    r = 2 :=
by
  sorry

end find_r_l675_675240


namespace julias_debt_l675_675593

-- Define the earnings per hour
def earning_per_hour (n : ℕ) : ℕ :=
  match n % 6 with
  | 0 => 6
  | 1 => 2
  | 2 => 4
  | 3 => 6
  | 4 => 2
  | 5 => 4
  | _ => 0 -- should not happen due to %6

-- Calculate total earnings for a given number of hours
def total_earnings (hours : ℕ) : ℕ :=
  (List.range hours).sum_map earning_per_hour

-- Main theorem statement
theorem julias_debt : total_earnings 45 = 180 := by
  sorry

end julias_debt_l675_675593


namespace problem_statement_l675_675117

noncomputable def f : ℝ → ℝ
| x := if x ∈ set.Ioo (-3/2) 0 then log (2 * x + 7) / log 2 else 0 -- define the function partially

theorem problem_statement :
  (∀ x : ℝ, f (-x) = - f x) ∧
  (∀ x : ℝ, f (3 - x) + f x = 0) ∧
  (∀ x ∈ set.Ioo (-3/2 : ℝ) 0, f x = log (2 * x + 7) / log 2) →
  f 2017 = - log 5 / log 2 :=
by
  intros h_odd h_func h_log
  sorry -- The proof is omitted as per the instructions

end problem_statement_l675_675117


namespace cos_alpha_value_l675_675840

-- Define the point P(4, -3)
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define the point P
def P : Point := { x := 4, y := -3 }

-- Define the radius r based on point P's coordinates.
def r : ℤ := 5  -- r = √(4^2 + (-3)^2) = √(16 + 9) = √25 = 5

-- Define the trigonometric function cos(α)
def cos_alpha (P : Point) (r : ℤ) : ℚ :=
  P.x.toRat / r.toRat

-- The theorem to be proved:
theorem cos_alpha_value : cos_alpha P r = 4/5 :=
by
  sorry

end cos_alpha_value_l675_675840


namespace valid_2021_digit_even_numbers_l675_675933

def count_valid_sequences (n : ℕ) : ℕ :=
  if n = 2021 then
    8.4 * (3^2019).toReal - 5 * (2^2019).toReal + 1
  else
    0

theorem valid_2021_digit_even_numbers :
  count_valid_sequences 2021 = 8.4 * (3^2019).toReal - 5 * (2^2019).toReal + 1 := by
  sorry

end valid_2021_digit_even_numbers_l675_675933


namespace function_sum_l675_675888

def f (x : ℝ) : ℝ := (2^(x + 1)) / (2^x + 1)

theorem function_sum :
  f (-1/3) + f (-1/2) + f (-1) + f 0 + f 1 + f (1/2) + f (1/3) = 7 :=
by
  -- Using the property that f(x) + f(-x) = 2 and the given conditions
  have h0 : f 0 = 1 := by sorry
  have h1 : ∀ x : ℝ, f x + f (-x) = 2 := by sorry
  
  let x := -1/3
  let y := -1/2
  let z := -1
  calc f x + f y + f z + f 0 + f (-z) + f (-y) + f (-x)
      = f x + f (-x) + f y + f (-y) + f z + f (-z) + f 0 : by sorry
  ... = 2 + 2 + 2 + f 0 : by { simp [h1], sorry }
  ... = 7 : by { simp [h0], sorry }

end function_sum_l675_675888


namespace sum_1_over_2011_array_l675_675378

theorem sum_1_over_2011_array (n : ℕ) :
  let m := 2011 * 2011 in
  let denom := (2 * 2011 - 1) * (2011 - 1) in
  let sum_fraction := m / denom in
  (m + denom) % 2011 = 0 :=
by {
  sorry
}

end sum_1_over_2011_array_l675_675378


namespace parabola_circle_tangent_l675_675677

noncomputable def p_sol : ℝ :=
  let p := 4
  in p

theorem parabola_circle_tangent (p : ℝ) (hp : p > 0) (h_dir : ∀ x, x = -p/2)
  (h_tangent : ∀ x y, x^2 + y^2 + 2*x = 0 → |-1 + p/2| = 1) : p = 4 :=
by
  sorry

end parabola_circle_tangent_l675_675677


namespace new_paint_intensity_l675_675991

variable (V : ℝ)  -- V is the volume of the original 50% intensity red paint.
variable (I₁ I₂ : ℝ)  -- I₁ is the intensity of the original paint, I₂ is the intensity of the replaced paint.
variable (f : ℝ)  -- f is the fraction of the original paint being replaced.

-- Assume given conditions
axiom intensity_original : I₁ = 0.5
axiom intensity_new : I₂ = 0.25
axiom fraction_replaced : f = 0.8

-- Prove that the new intensity is 30%
theorem new_paint_intensity :
  (f * I₂ + (1 - f) * I₁) = 0.3 := 
by 
  -- This is the main theorem we want to prove
  sorry

end new_paint_intensity_l675_675991


namespace running_speed_l675_675727

variables (w t_w t_r : ℝ)

-- Given conditions
def walking_speed : w = 8 := sorry
def walking_time_hours : t_w = 4.75 := sorry
def running_time_hours : t_r = 2 := sorry

-- Prove the man's running speed
theorem running_speed (w t_w t_r : ℝ) 
  (H1 : w = 8) 
  (H2 : t_w = 4.75) 
  (H3 : t_r = 2) : 
  (w * t_w) / t_r = 19 := 
sorry

end running_speed_l675_675727


namespace find_m_l675_675908

theorem find_m (m : ℝ) (h₁: 0 < m) (h₂: ∀ p q : ℝ × ℝ, p = (m, 4) → q = (2, m) → ∃ s : ℝ, s = m^2 ∧ ((q.2 - p.2) / (q.1 - p.1)) = s) : m = 2 :=
by
  sorry

end find_m_l675_675908


namespace cannot_be_20182017_l675_675321

theorem cannot_be_20182017 (a b : ℤ) (h : a * b * (a + b) = 20182017) : False :=
by
  sorry

end cannot_be_20182017_l675_675321


namespace min_max_sums_are_correct_l675_675381

def coin_values : List ℕ := [1, 1, 1, 5, 10, 10, 25, 50]

def pair_sums (l : List ℕ) : List ℕ :=
  l.product l |>.filter (λ (x : ℕ × ℕ), x.fst ≠ x.snd) |>.map (λ (x : ℕ × ℕ), x.fst + x.snd)

def min_sum (l : List ℕ) : ℕ := (pair_sums l).minimum' sorry
def max_sum (l : List ℕ) : ℕ := (pair_sums l).maximum' sorry

theorem min_max_sums_are_correct : min_sum coin_values = 2 ∧ max_sum coin_values = 75 :=
by
  sorry

end min_max_sums_are_correct_l675_675381


namespace hannahs_brothers_age_l675_675901

-- Define the conditions
variables (S : ℕ) (age_of_hannah : ℕ)
variable (brothers_are_same_age : ℕ)
variable [fact (age_of_hannah = 48)]
variable [fact (age_of_hannah = 2 * S)]
variable [fact (brothers_are_same_age = S / 3)]

-- Math proof statement: Each brother's age is 8
theorem hannahs_brothers_age :
  brothers_are_same_age = 8 :=
sorry

end hannahs_brothers_age_l675_675901


namespace tan_div_sub_tan_div_eq_l675_675962

variables {x y : ℝ}

theorem tan_div_sub_tan_div_eq (h1 : (sin x / cos y) - (sin y / cos x) = 2) 
                               (h2 : (cos x / sin y) - (cos y / sin x) = 3) : 
  (tan x / tan y) - (tan y / tan x) = (2 / 3) :=
by 
  sorry

end tan_div_sub_tan_div_eq_l675_675962


namespace minimum_points_to_guarantee_win_l675_675656

theorem minimum_points_to_guarantee_win :
  ∃ (points : ℕ), points = 16 ∧
    ∀ (other_points : ℕ),
    let possible_scores := [18, 16, 14, 12, 10, 8, 6, 4, 2] in
    (other_points ∈ possible_scores ∧ other_points < 16) → points > other_points :=
by
  sorry

end minimum_points_to_guarantee_win_l675_675656


namespace g_value_at_2002_l675_675485

-- Define the function f on ℝ
variable (f : ℝ → ℝ)

-- Conditions given in the problem
axiom f_one : f 1 = 1
axiom f_inequality_5 : ∀ x : ℝ, f (x + 5) ≥ f x + 5
axiom f_inequality_1 : ∀ x : ℝ, f (x + 1) ≤ f x + 1

-- Define the function g based on f
def g (x : ℝ) : ℝ := f x + 1 - x

-- The goal is to prove that g 2002 = 1
theorem g_value_at_2002 : g 2002 = 1 :=
sorry

end g_value_at_2002_l675_675485


namespace min_abs_diff_2002_l675_675310

theorem min_abs_diff_2002 : ∃ (a1 a2 ... am b1 b2 ... bn : ℕ),
  2002 = (nat.factorial a1 * nat.factorial a2 * ... * nat.factorial am) /
          (nat.factorial b1 * nat.factorial b2 * ... * nat.factorial bn) ∧
  a1 ≥ a2 ∧ ... ∧ a_m ∧
  b1 ≥ b2 ∧ ... ∧ b_n ∧
  a1 + b1 = minimized =>
  abs (a1 - b1) = 2 :=
sorry

end min_abs_diff_2002_l675_675310


namespace emily_sixth_score_l675_675827

theorem emily_sixth_score:
  ∀ (s₁ s₂ s₃ s₄ s₅ sᵣ : ℕ),
  s₁ = 88 →
  s₂ = 90 →
  s₃ = 85 →
  s₄ = 92 →
  s₅ = 97 →
  (s₁ + s₂ + s₃ + s₄ + s₅ + sᵣ) / 6 = 91 →
  sᵣ = 94 :=
by intros s₁ s₂ s₃ s₄ s₅ sᵣ h₁ h₂ h₃ h₄ h₅ h₆;
   rw [h₁, h₂, h₃, h₄, h₅] at h₆;
   sorry

end emily_sixth_score_l675_675827


namespace corresponding_angles_equal_l675_675275

variable {α : Type}
variables (A B : α) [angle : has_measure α (angle_measure α)]
variables (h : A.is_corresponding_with B)

theorem corresponding_angles_equal (A B : α) [angle A] [angle B] :
  A.is_corresponding_with B → A = B :=
by
  sorry

end corresponding_angles_equal_l675_675275


namespace total_surface_area_space_l675_675018

theorem total_surface_area_space (h r1 : ℝ) (h_cond : h = 8) (r1_cond : r1 = 3) : 
  (2 * π * (r1 + 1) * h - 2 * π * r1 * h) = 16 * π := 
by
  sorry

end total_surface_area_space_l675_675018


namespace space_diagonal_of_rectangular_solid_l675_675016

theorem space_diagonal_of_rectangular_solid :
  ∀ (l w h : ℕ), l = 12 → w = 4 → h = 3 → (sqrt (l^2 + w^2 + h^2) = 13) :=
by
  intros l w h hl hw hh
  rw [hl, hw, hh]
  show sqrt (12^2 + 4^2 + 3^2) = 13
  sorry -- Proof steps skipped

end space_diagonal_of_rectangular_solid_l675_675016


namespace total_commission_l675_675418

-- Define the commission rate
def commission_rate : ℝ := 0.02

-- Define the sale prices of the three houses
def sale_price1 : ℝ := 157000
def sale_price2 : ℝ := 499000
def sale_price3 : ℝ := 125000

-- Total commission calculation
theorem total_commission :
  (commission_rate * sale_price1 + commission_rate * sale_price2 + commission_rate * sale_price3) = 15620 := 
by
  sorry

end total_commission_l675_675418


namespace financial_loss_example_l675_675944

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ := 
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := 
  P * (1 + r * t)

theorem financial_loss_example :
  let P := 10000
  let r1 := 0.06
  let r2 := 0.05
  let t := 3
  let n := 4
  let A1 := compound_interest P r1 n t
  let A2 := simple_interest P r2 t
  abs (A1 - A2 - 456.18) < 0.01 := by
    sorry

end financial_loss_example_l675_675944


namespace middle_three_cards_sum_l675_675797

theorem middle_three_cards_sum :
  ∃ (G Y : List ℕ), 
  G = [1, 2, 3, 4, 5, 6] ∧
  Y = [4, 5, 6, 7, 8] ∧
  (∀ (i : ℕ), i < 11 → 
    (if even i 
    then (i < 11 ∧ (Y.get! (i / 2)).factorOf (G.get! (i - 1) / 2)) 
    else (i < 11 ∧ (G.get! (i / 2)).factorOf (Y.get! (i - 1) / 2))) ) ∧
  (∀ (a b : ℕ), (a < 11 ∧ b < 11 ∧ y % 2 ≠ 0 → (a.get! y + b.get! y = odd) ) ) ∧
  (∃ (arrangement : List ℕ), ( G ++ Y = arrangement  ) ∧   
  (arrangement.get! ((arrangement.length / 2) - 1) + arrangement.get! (arrangement.length / 2) + (arrangement.get! ((arrangement.length / 2) + 1 ) = 14) )
  sorry

end middle_three_cards_sum_l675_675797


namespace total_amount_on_July_2005_l675_675976

theorem total_amount_on_July_2005 (a r : ℝ) (H : 0 ≤ r) : 
  ∑ k in finset.range 5, a * (1 + r)^(k+1) = a / r * ((1 + r)^6 - (1 + r)) :=
sorry

end total_amount_on_July_2005_l675_675976


namespace percentage_decrease_l675_675910

-- Definitions based on the conditions
def original_salary (S : ℝ) : Prop :=
  1.15 * S = 575

def decreased_salary (S : ℝ) : Prop :=
  S * (1 - x / 100) = 560

-- The problem statement: Prove that the percentage decrease is 12%
theorem percentage_decrease (S : ℝ) (x : ℝ) (h1 : original_salary S) (h2 : decreased_salary S) : x = 12 := by
  sorry

end percentage_decrease_l675_675910


namespace fraction_of_sums_l675_675113

variables {a : ℕ → ℝ} -- Define the arithmetic sequence a_n
variable {S : ℕ → ℝ} -- Define S_n as the sum of the first n terms

-- Define arithmetic sequence condition
def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, (a (n + m) = a n + a m)

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(n * (a 1 + a n)) / 2

-- Given conditions
variables (h1 : is_arithmetic_seq a)
variable (h2 : a 8 = 2 * a 3)
variable (h3 : a 8 = 2)

-- Theorem to prove
theorem fraction_of_sums :
  (sum_first_n_terms a 15) / (sum_first_n_terms a 5) = 6 :=
sorry

end fraction_of_sums_l675_675113


namespace monotonicity_f_interval_extrema_f_l675_675889

noncomputable def f (a x : ℝ) := (a + 1/a) * Real.log x + 1/x - x

-- Specifying the monotonicity of f(x) on the interval (0, 1) when a > 1
theorem monotonicity_f_interval (a x : ℝ) (h : 1 < a) (hx : 0 < x) (hx1 : x < 1) :
  let fp := (-(1/x - a)*(1/x - 1/a)) in
  if hx2 : x < 1/a then fp < 0 else fp > 0 :=
sorry

-- Specifying the extrema of f(x) when a > 0
theorem extrema_f (a : ℝ) (h : 0 < a) :
  if h1 : a < 1 then
    let min_val := (a + 1/a) * Real.log a + 1/a - a in
    let max_val := -(a + 1/a) * Real.log a + a - 1/a in
    True -- here we'd assert the characteristics of the local min and max, skipping exact formalism
  else if h2 : a = 1 then
    f a x <= f a y ∀ x y ∈ ℝ, 0 < x ∧ 0 < y ∧ x < y
  else
    let min_val := -(a + 1/a) * Real.log a + a - 1/a in
    let max_val := (a + 1/a) * Real.log a + 1/a - a in
    True -- similarly, skipping exact formalism for now
  :=
sorry

end monotonicity_f_interval_extrema_f_l675_675889


namespace triangle_perimeter_l675_675749

-- Define the side lengths
def a : ℕ := 7
def b : ℕ := 10
def c : ℕ := 15

-- Define the perimeter
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- Statement of the proof problem
theorem triangle_perimeter : perimeter 7 10 15 = 32 := by
  sorry

end triangle_perimeter_l675_675749


namespace product_of_chords_is_correct_l675_675598

noncomputable def omega := Complex.exp (2 * Real.pi * Complex.I / 18)
def a := (3 : ℝ)
def b := (-3 : ℝ)
def c (k : ℕ) := 3 * omega ^ k

def ac_length (k : ℕ) := 3 * Complex.abs (1 - omega ^ k)
def bc_length (k : ℕ) := 3 * Complex.abs (1 - omega ^ (k + 9))

def ac_prod := ∏ k in Finset.range 8, ac_length (k + 1)
def bc_prod := ∏ k in Finset.range 8, bc_length (k + 1)
def prod_of_all_chords := ac_prod * bc_prod

theorem product_of_chords_is_correct :
  prod_of_all_chords = 387420489 := 
  sorry

end product_of_chords_is_correct_l675_675598


namespace min_value_expression_l675_675106

noncomputable def find_min_val (a b c : ℝ) : ℝ :=
  (ac/b) + (c/(ab)) - (c/2) + (sqrt 5)/(c-2)

theorem min_value_expression (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 2) (h_sum : a + b = 2) : 
  find_min_val a b c = sqrt 10 + sqrt 5 :=
sorry

end min_value_expression_l675_675106


namespace area_of_polar_curve_l675_675578

theorem area_of_polar_curve (f : ℝ → ℝ) (h : ∀ θ, f θ = 2 * Real.cos θ) : 
  (∫ θ in 0..(2 * Real.pi), (f θ)^2 / 2) = Real.pi :=
by
  -- Integral calculation and proof omitted
  sorry

end area_of_polar_curve_l675_675578


namespace find_larger_number_l675_675708

-- Definitions based on the conditions
variables (x y : ℕ)

-- Main theorem
theorem find_larger_number (h1 : x + y = 50) (h2 : x - y = 10) : x = 30 :=
by
  sorry

end find_larger_number_l675_675708


namespace foldable_point_area_l675_675993

theorem foldable_point_area (AB BC : ℝ) (C_angle : ℝ) (hAB : AB = 24) (hBC : BC = 48) (hC_angle : C_angle = 90) :
  ∃ q r s : ℕ, (s ≠ 0 ∧ ∀ (p : ℕ), p * p | s → p = 1) ∧ ((area : ℝ) = q * Real.pi - r * Real.sqrt s) ∧ (q + r + s = 411) := 
begin
  sorry
end

end foldable_point_area_l675_675993


namespace distance_of_points_in_polar_coordinates_l675_675223

theorem distance_of_points_in_polar_coordinates
  (A : Real × Real) (B : Real × Real) (θ1 θ2 : Real)
  (hA : A = (5, θ1)) (hB : B = (12, θ2))
  (hθ : θ1 - θ2 = Real.pi / 2) : 
  dist (5 * Real.cos θ1, 5 * Real.sin θ1) (12 * Real.cos θ2, 12 * Real.sin θ2) = 13 := 
by sorry

end distance_of_points_in_polar_coordinates_l675_675223


namespace train_length_l675_675404

noncomputable def length_of_train (t : ℝ) (v_train_kmh : ℝ) (v_man_kmh : ℝ) : ℝ :=
  let v_relative_kmh := v_train_kmh - v_man_kmh
  let v_relative_ms := v_relative_kmh * 1000 / 3600
  v_relative_ms * t

theorem train_length : length_of_train 30.99752019838413 80 8 = 619.9504039676826 := 
  by simp [length_of_train]; sorry

end train_length_l675_675404


namespace triangle_side_BC_l675_675546

theorem triangle_side_BC {A B C : Type} [inner_product_space ℝ A] 
  {b c BC: ℝ} 
  (hB : ∠ B = real.pi / 3) 
  (hAC : b = real.sqrt 19) 
  (hAB : c = 2) : 
  BC = 3 := 
sorry

end triangle_side_BC_l675_675546


namespace valid_t_range_for_f_l675_675913

theorem valid_t_range_for_f :
  (∀ x : ℝ, |x + 1| + |x - t| ≥ 2015) ↔ t ∈ (Set.Iic (-2016) ∪ Set.Ici 2014) := 
sorry

end valid_t_range_for_f_l675_675913


namespace probability_two_aces_probability_two_aces_after_two_kings_l675_675990

-- Define the necessary sets and counts
def total_cards : ℕ := 12
def aces : ℕ := 4
def kings : ℕ := 4
def queens : ℕ := 4

-- Calculate combinations
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Problem 1
theorem probability_two_aces : 
  (C total_cards 2) = 66 ∧ (C aces 2) = 6 → (6 : ℚ) / (66 : ℚ) = (1 : ℚ) / (11 : ℚ) := by
  intros h
  have h1 : C total_cards 2 = 66 := h.left
  have h2 : C aces 2 = 6 := h.right
  rw [h2, h1]
  norm_num

-- Define remaining cards after drawing 2 Kings
def remaining_cards : ℕ := total_cards - 2 

-- Problem 2
theorem probability_two_aces_after_two_kings :
  (C remaining_cards 2) = 45 ∧ (C aces 2) = 6 → (6 : ℚ) / (45 : ℚ) = (2 : ℚ) / (15 : ℚ) := by
  intros h
  have h1 : C remaining_cards 2 = 45 := h.left
  have h2 : C aces 2 = 6 := h.right
  rw [h2, h1]
  norm_num


end probability_two_aces_probability_two_aces_after_two_kings_l675_675990


namespace dice_product_probability_l675_675716

theorem dice_product_probability :
  ∃ p : ℚ, p = 1 / 24 ∧ (let outcomes : finset (ℕ × ℕ × ℕ) :=
    { (a, b, c) | 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 }).count
    (λ (abc : ℕ × ℕ × ℕ), abc.1 * abc.2 * abc.3 = 72) / (6 * 6 * 6)  = p :=
by { sorry }

end dice_product_probability_l675_675716


namespace correct_proposition_is_3_l675_675035

/-
Among the following propositions:
1. If \( m > 0 \), then the equation \( x^2 - x + m = 0 \) has real roots.
2. The converse of the proposition "If \( x > 1, y > 1 \), then \( x + y > 2 \)".
3. The negation of "For all \( x \in \{ x \mid -2 < x < 4 \} \), \( |x-2| < 3 \)".
4. \(\Delta > 0\) is a necessary and sufficient condition for a quadratic equation \( ax^2 + bx + c = 0 \) to have one positive root and one negative root.

The true propositions are \(\boxed{3}\).
-/

theorem correct_proposition_is_3 :
  (∀ m > 0, (¬ (∃ x : ℝ, x^2 - x + m = 0))) ∧
  (¬ (∀ x y : ℝ, (x + y > 2) → (x > 1 ∧ y > 1))) ∧
  (∃ x : ℝ, (-2 < x ∧ x < 4) ∧ (|x - 2| ≥ 3)) ∧
  (¬ (∀ Δ > 0, ∀ a b c : ℝ, (a ≠ 0) → (∃ x1 x2 : ℝ, (a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) → (x1 * x2 < 0)))) :=
begin
  sorry -- Proof steps are not required; hence, we use sorry to skip.
end

end correct_proposition_is_3_l675_675035


namespace tank_filling_rate_l675_675384

theorem tank_filling_rate
    (V_i : ℕ) (L1_rate : ℕ) (L1_time : ℕ) (L2_rate : ℕ) (L2_time : ℕ)
    (M : ℕ) (T_f : ℕ) : ℕ :=
begin
  let total_loss := (L1_rate * L1_time) + (L2_rate * L2_time),
  let final_volume := V_i - total_loss,
  let current_volume := V_i - M,
  let added_volume := current_volume - final_volume,
  exact added_volume / T_f
end

example (V_i : ℕ) (L1_rate : ℕ) (L1_time : ℕ) (L2_rate : ℕ) (L2_time : ℕ)
    (M : ℕ) (T_f : ℕ) : tank_filling_rate V_i L1_rate L1_time L2_rate L2_time M T_f = 40000 :=
by {
  sorry -- Definition of variables from the problem statement and proof steps
}

end tank_filling_rate_l675_675384


namespace smallest_n_l675_675824

theorem smallest_n (n : ℕ) (h : n ≥ 2) : 
  (∃ m : ℕ, m * m = (n + 1) * (2 * n + 1) / 6) ↔ n = 337 :=
by
  sorry

end smallest_n_l675_675824


namespace locus_of_point_M_l675_675788

def Point := (ℝ × ℝ)

def F1 : Point := (0, -5)
def F2 : Point := (0, 5)

def distance (p1 p2 : Point) : ℝ :=
  (Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2))

def on_line_segment (p : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (p = ((1 - t) * F1.1 + t * F2.1, (1 - t) * F1.2 + t * F2.2))

theorem locus_of_point_M (M : Point) (h : distance M F1 + distance M F2 = 10) : on_line_segment M :=
sorry

end locus_of_point_M_l675_675788


namespace basketball_properties_l675_675380

noncomputable def radius (C : ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

noncomputable def volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem basketball_properties (C : ℝ) (hC : C = 29.5) :
  let r := radius C in
  surface_area r = 1081.5625 / Real.pi ∧
  volume r = 3203.609375 / (3 * Real.pi^2) :=
suffices h_r : radius 29.5 = 29.5 / (2 * Real.pi), by
suffices h_s : surface_area (radius 29.5) = 1081.5625 / Real.pi, by
suffices h_v : volume (radius 29.5) = 3203.609375 / (3 * Real.pi^2), by
  ⟨h_s, h_v⟩
sorry

end basketball_properties_l675_675380


namespace carl_savings_correct_l675_675428

theorem carl_savings_correct (x : ℝ) :
  (let savings := 6 * x in
   let after_bills := savings - (1 / 3) * savings in
   let final_savings := after_bills + 70 in
   final_savings = 170) →
   x = 25 :=
by
  intro h
  have savings_eq : 6 * x = 6 * x := by rfl
  have after_bills_eq : 6 * x - (1 / 3) * (6 * x) = 4 * x := by
    rw [mul_comm (1 / 3) (6 * x)]
    rw mul_assoc
    rw mul_comm (1 / 3) 6
    rw ← mul_assoc
    simp
  have final_savings_eq : 4 * x + 70 = 170 := h
  have aux : 4 * x = 170 - 70 := by
    rw final_savings_eq
    norm_num
  have x_eq_25 : x = (170 - 70) / 4 := by
    rw aux
    norm_num
  rw x_eq_25
  norm_num
  sorry

end carl_savings_correct_l675_675428


namespace domain_of_sqrt_and_cbrt_domain_of_function_l675_675345

def domain_of_f : Set Real :=
  {x | x ≥ 5}

theorem domain_of_sqrt_and_cbrt (x : Real) :
  (x - 5 ≥ 0) ∧ true → x ∈ domain_of_f :=
by
  intros h
  cases h with h₁ h₂
  exact h₁

-- To state the domain result formally we can phrase it as
theorem domain_of_function :
  ∀ x : Real, x ∈ domain_of_f ↔ (x - 5 ≥ 0) :=
by
  intro x
  apply Iff.intro
  · intro h
    exact h
  
  · intro h
    exact h

end domain_of_sqrt_and_cbrt_domain_of_function_l675_675345


namespace corresponding_angles_equal_l675_675277

variable {α : Type}
variables (A B : α) [angle : has_measure α (angle_measure α)]
variables (h : A.is_corresponding_with B)

theorem corresponding_angles_equal (A B : α) [angle A] [angle B] :
  A.is_corresponding_with B → A = B :=
by
  sorry

end corresponding_angles_equal_l675_675277


namespace blackbirds_in_each_tree_l675_675794

variable (B : ℕ)

def conditions : Prop :=
  (∃ B, 7 * B + 13 = 34)

theorem blackbirds_in_each_tree (h : conditions B) : B = 3 :=
  sorry

end blackbirds_in_each_tree_l675_675794


namespace symmetry_point_l675_675882

noncomputable def g (x : ℝ) : ℝ := Math.sin (2 * x + (6:ℝ))

theorem symmetry_point :
  ∃ φ : ℝ, φ > 0 ∧ (∀ x, g (x) = Math.sin (2 * (x + φ) + (π / 6))) ∧ (g (-π / 12) = 1) ∧
  (∀ x, g (x) = g ((4 * π / 3) - x)) :=
sorry

end symmetry_point_l675_675882


namespace students_no_scholarship_percentage_l675_675750

theorem students_no_scholarship_percentage (x : ℝ) (h : x > 0) :
    let total_students := 11 * x
    let boys_no_scholarship := 3.75 * x
    let girls_no_scholarship := 4.8 * x
    let students_no_scholarship := boys_no_scholarship + girls_no_scholarship
    let percentage_no_scholarship := (students_no_scholarship / total_students) * 100
    percentage_no_scholarship ≈ 77.73 :=
        by
          let total_students := 11 * x
          let boys_no_scholarship := 3.75 * x
          let girls_no_scholarship := 4.8 * x
          let students_no_scholarship := boys_no_scholarship + girls_no_scholarship
          let percentage_no_scholarship := (students_no_scholarship / total_students) * 100
          have h1 : students_no_scholarship = 8.55 * x := by ring
          have h2 : total_students = 11 * x := by ring
          have h3 : percentage_no_scholarship = (8.55 * x) / (11 * x) * 100 := by rw [h1, h2]
          have h4 : percentage_no_scholarship = 8.55 / 11 * 100 := by field_simp at h3; exact h3
          have h5 : 8.55 / 11 * 100 ≈ 77.73 := sorry
          exact h5

end students_no_scholarship_percentage_l675_675750


namespace ratio_of_equilateral_triangles_l675_675721

noncomputable def median (s : ℝ) : ℝ :=
  (√3 / 2) * s

noncomputable def circumradius (s : ℝ) : ℝ :=
  (s * √3) / 3

theorem ratio_of_equilateral_triangles
  (s t : ℝ) (hs : s > 0) (ht : t > 0)
  (MX := median s) (MY := median t)
  (RX := circumradius s) (RY := circumradius t) :
  (s + 2 * MX + RX) / (t + 2 * MY + RY) = s / t :=
by
  sorry

end ratio_of_equilateral_triangles_l675_675721


namespace bert_spent_at_dry_cleaners_l675_675042

theorem bert_spent_at_dry_cleaners
  (initial_amount : ℕ) (hardware_store_fraction : ℚ) (remaining_after_hardware : ℕ)
  (remaining_after_cleaners : ℚ) (grocery_store_fraction : ℚ) (final_amount : ℕ)
  (H1 : initial_amount = 44) (H2 : hardware_store_fraction = 1/4)
  (H3 : remaining_after_hardware = initial_amount - (initial_amount * hardware_store_fraction).to_nat)
  (H4 : grocery_store_fraction = 1/2)
  (H5 : final_amount = remaining_after_cleaners - (remaining_after_cleaners * grocery_store_fraction))
  (H6 : final_amount = 12) :
  ∃ X : ℕ, remaining_after_cleaners = remaining_after_hardware - X ∧ X = 9 :=
by
  sorry

end bert_spent_at_dry_cleaners_l675_675042


namespace compute_trig_expr_l675_675053

theorem compute_trig_expr :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 2 * Real.sec (10 * Real.pi / 180) :=
by
  sorry

end compute_trig_expr_l675_675053


namespace average_rainfall_is_4_l675_675703

namespace VirginiaRainfall

def march_rainfall : ℝ := 3.79
def april_rainfall : ℝ := 4.5
def may_rainfall : ℝ := 3.95
def june_rainfall : ℝ := 3.09
def july_rainfall : ℝ := 4.67

theorem average_rainfall_is_4 :
  (march_rainfall + april_rainfall + may_rainfall + june_rainfall + july_rainfall) / 5 = 4 := by
  sorry

end VirginiaRainfall

end average_rainfall_is_4_l675_675703


namespace find_a_l675_675627

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then -x else x^2

theorem find_a (a : ℝ) (h : f a = 4) : a = -4 ∨ a = 2 :=
by
  sorry

end find_a_l675_675627


namespace find_m_plus_n_l675_675877

theorem find_m_plus_n (x : ℝ) (m n : ℕ) (h₁ : (1 + Real.sin x) / (Real.cos x) = 22 / 7) 
                      (h₂ : (1 + Real.cos x) / (Real.sin x) = m / n) :
                      m + n = 44 := by
  sorry

end find_m_plus_n_l675_675877


namespace max_value_h_l675_675288

-- Define the functions f and g with the given ranges
def f (x : ℝ) : ℝ := sorry
def g (x : ℝ) : ℝ := sorry

-- Define h as the sum of f and g
def h (x : ℝ) : ℝ := f(x) + g(x)

-- Conditions on the ranges of f and g
axiom range_f : ∀ x : ℝ, -7 ≤ f(x) ∧ f(x) ≤ 4
axiom range_g : ∀ x : ℝ, -3 ≤ g(x) ∧ g(x) ≤ 2

-- The theorem to prove
theorem max_value_h : ∃ y : ℝ, (h(y) = 6) :=
  by
  sorry

end max_value_h_l675_675288


namespace smallest_n_satisfying_l675_675071

def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2

def satisfied_sum (n : ℕ) : Prop :=
  (Finset.range (n + 1)).sum (λ k, log_base_2 (1 + 1 / (3 ^ (3 ^ k)))) ≥ 1 + log_base_2 (2023 / 2024)

theorem smallest_n_satisfying : ∃ n : ℕ, satisfied_sum n ∧ ∀ m : ℕ, m < n → ¬satisfied_sum m :=
sorry

end smallest_n_satisfying_l675_675071


namespace solution_set_of_inequality_l675_675700

theorem solution_set_of_inequality (x : ℝ) (hx : x ≥ 2) :
  (sqrt (log 2 x - 1) + (1 / 2) * log 4 (x ^ 3) + 2 > 0) ↔ (2 ≤ x ∧ x < 4) :=
sorry

end solution_set_of_inequality_l675_675700


namespace select_non_coplanar_points_l675_675401

def Point : Type := sorry  -- Placeholder for the type of points in our tetrahedron

def vertices : Fin 4 → Point := sorry  -- A function mapping each of the 4 numbers to a vertex
def midpoints : Fin 6 → Point := sorry  -- A function mapping each of the 6 numbers to a midpoint

def is_non_coplanar (p1 p2 p3 p4 : Point) : Prop :=
  sorry  -- Predicate to check if four points are non-coplanar

theorem select_non_coplanar_points :
  (∑ (p1 : Fin 10), ∑ (p2 : Fin 10), ∑ (p3 : Fin 10), ∑ (p4 : Fin 10),
  (p1 < p2) → (p2 < p3) → (p3 < p4) → is_non_coplanar (fin_or_embed p1) (fin_or_embed p2) (fin_or_embed p3) (fin_or_embed p4)) = 141 :=
sorry  -- Proof not provided

-- Helper function to map Fin 10 to the correct Point, considering both vertices and midpoints
def fin_or_embed (i : Fin 10) : Point :=
  if h : i.val < 4 then vertices ⟨i.val,h⟩ else midpoints ⟨i.val - 4, by linarith⟩

end select_non_coplanar_points_l675_675401


namespace sphere_volume_increase_l675_675531

theorem sphere_volume_increase (R : ℝ) (h : 0 < R) :
  let great_circle_area := π * R^2
  let new_great_circle_area := 2 * great_circle_area
  let new_radius := R * sqrt 2
  let initial_volume := (4 * π * R^3) / 3
  let new_volume := (4 * π * (new_radius)^3) / 3
  new_volume / initial_volume = 2 * sqrt 2 :=
by sorry

end sphere_volume_increase_l675_675531


namespace range_of_f_max_value_of_f_l675_675505

section
  variable (a : ℝ) (x : ℝ)
  variable (h_a : a > 1) 

  def f (x : ℝ) := 1 - 2 * (a ^ x) - (a ^ (2 * x))

  theorem range_of_f : Set.range (f a) = {y : ℝ | y < 1} :=
  sorry

  variable (h_min_f : ∀ x ∈ Icc (-2 : ℝ) (1 : ℝ), f a x ≥ -7)

  theorem max_value_of_f : ∃ y, y = max (f a x) :=
  sorry
end

end range_of_f_max_value_of_f_l675_675505


namespace factorize_expression_l675_675447

theorem factorize_expression : 989 * 1001 * 1007 + 320 = 991 * 997 * 1009 := 
by sorry

end factorize_expression_l675_675447


namespace tank_base_length_width_difference_l675_675789

variable (w l h : ℝ)

theorem tank_base_length_width_difference :
  (l = 5 * w) →
  (h = (1/2) * w) →
  (l * w * h = 3600) →
  (|l - w - 45.24| < 0.01) := 
by
  sorry

end tank_base_length_width_difference_l675_675789


namespace not_exists_odd_product_sum_l675_675323

theorem not_exists_odd_product_sum (a b : ℤ) : ¬ (a * b * (a + b) = 20182017) :=
sorry

end not_exists_odd_product_sum_l675_675323


namespace problem_l675_675956

open Real

noncomputable def g (x : ℝ) := sorry

theorem problem (m t : ℕ)
  (h1 : ∀ x y : ℝ, g(x) * g(y) - g(x * y) = 2 * x + 2 * y)
  (h2 : m = {y : ℝ | ∃ x : ℝ, g(x) = y}.card)
  (h3 : t = ∑ y in ({y : ℝ | ∃ x : ℝ, g(x) = y}).to_finset, y) :
  m * t = 3 := 
sorry

end problem_l675_675956


namespace graph_passes_through_all_quadrants_l675_675887

def f (a x : ℝ) : ℝ := (1 / 3) * a * x^3 + (1 / 2) * a * x^2 - 2 * a * x + 2 * a + 1

theorem graph_passes_through_all_quadrants (a : ℝ) :
  (-6 / 5 < a ∧ a < -3 / 16) ↔ (f a (-2) * f a 1 < 0) :=
sorry

end graph_passes_through_all_quadrants_l675_675887


namespace compute_trig_expr_l675_675052

theorem compute_trig_expr :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 2 * Real.sec (10 * Real.pi / 180) :=
by
  sorry

end compute_trig_expr_l675_675052


namespace c_plus_2d_eq_neg59_l675_675301

theorem c_plus_2d_eq_neg59 (c d : ℤ) (h1 : (5 * (X : ℤ[X]) + c) * (5 * X + d) = 25 * X^2 - 135 * X - 150) : c + 2 * d = -59 :=
sorry

end c_plus_2d_eq_neg59_l675_675301


namespace basketball_team_win_rate_l675_675759

theorem basketball_team_win_rate
  (games_played_initial : ℕ)
  (games_won_initial : ℕ)
  (remaining_games : ℕ)
  (total_games : ℕ)
  (target_win_percentage : ℚ)
  (correct_answer : ℕ) :
  games_played_initial = 55 →
  games_won_initial = 45 →
  remaining_games = 35 →
  total_games = games_played_initial + remaining_games →
  target_win_percentage = 0.75 →
  correct_answer = 23 →
  ∃ x : ℕ, x ≥ correct_answer ∧ (games_won_initial + x) / total_games.to_rat = target_win_percentage :=
by
  intros h1 h2 h3 h4 h5 h6
  have : (games_won_initial + correct_answer) / total_games.to_rat = target_win_percentage := sorry
  use correct_answer
  split
  · exact le_refl _
  · exact this

end basketball_team_win_rate_l675_675759


namespace inequality_solution_l675_675665

theorem inequality_solution (x : ℝ) : 
  (6 * x^2 + 9 * x - 48) / ((3 * x + 5) * (x - 2)) < 0 ↔ x ∈ Ioo (-4 : ℝ) (-5 / 3 : ℝ) :=
sorry

end inequality_solution_l675_675665


namespace magnitude_of_sum_of_perpendicular_vectors_l675_675164

def p : ℝ × ℝ := (1, 2)
def q (x : ℝ) : ℝ × ℝ := (x, 3)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem magnitude_of_sum_of_perpendicular_vectors (x : ℝ) (h : dot_product p (q x) = 0) :
  |(p.1 + (q x).1, p.2 + (q x).2)| = 5 * real.sqrt 2 := by
  sorry

end magnitude_of_sum_of_perpendicular_vectors_l675_675164


namespace parking_space_area_l675_675746

theorem parking_space_area (L W : ℕ) (h1 : L = 9) (h2 : 2 * W + L = 37) : L * W = 126 :=
by
  -- Proof omitted.
  sorry

end parking_space_area_l675_675746


namespace cover_plane_with_1988_circles_l675_675587

theorem cover_plane_with_1988_circles :
  ∃ (families : ℕ → ℕ → bool), 
  (∀ x y : ℤ, (Σ k, families k (x, y)) = 1988) := sorry

end cover_plane_with_1988_circles_l675_675587


namespace triangle_perimeter_l675_675156

theorem triangle_perimeter (a b x : ℝ) (hx : (x - 6) * (x - 10) = 0) (h1 : a = 4) (h2 : b = 6) 
  (h_triangle_ineq : 4 + 6 > x ∧ 4 + x > 6 ∧ 6 + x > 4) : a + b + x = 16 :=
by
  have root6 : x = 6 ∨ x = 10 := by {
    have discrim_eq_zero : (x - 6) * (x - 10) = 0 := hx
    rw mul_eq_zero at discrim_eq_zero
    exact discrim_eq_zero
  }
  cases root6 with
  | inl h6 =>
    rw h6
    exact calc
      a + b + 6 = 4 + 6 + 6 : by rw [h1, h2]
           ... = 16 : by linarith
  | inr h10 =>
    exfalso
    have triangle_ineq := h_triangle_ineq
    rw h10 at triangle_ineq
    linarith

end triangle_perimeter_l675_675156


namespace cardinality_of_A_max_A_40_l675_675455

-- Part (a) definition and theorem
def A (a : ℝ) : set ℕ :=
  {n | ∃ k : ℕ, n^2 + a * n = k^2}

theorem cardinality_of_A (a : ℝ) : (finite (A a)) ↔ a ≠ 0 :=
by sorry

-- Part (b) definition and theorem
def A_40 : set ℕ :=
  {n | ∃ k : ℕ, n^2 + 40 * n = k^2}

theorem max_A_40 : ∃ n : ℕ, n ∈ A_40 ∧ (∀ m : ℕ, m ∈ A_40 → m ≤ n) ∧ n = 380 :=
by sorry

end cardinality_of_A_max_A_40_l675_675455


namespace find_x_l675_675326

theorem find_x : 
  ∃ x : ℝ, 4.7 * 13.26 + 4.7 * 9.43 + 4.7 * x = 470 ∧ x = 77.31 :=
by
  sorry

end find_x_l675_675326


namespace length_of_EF_l675_675462

variables {A B C D E F : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variables (triangle_ABC : A) (triangle_DEF : D)
variables {AB DE BC EF : ℝ}

-- Given conditions
variable (sim_ABC_DEF : ∃ (A B C D E F : Type*) 
                          [metric_space A] [metric_space B] [metric_space C] 
                          [metric_space D] [metric_space E] [metric_space F],
                          ∀ (triangle_ABC : A) (triangle_DEF : D),
                          ∃ (AB DE BC EF : ℝ),
                          (triangle_ABC = triangle_DEF)
                        )
variable (ratio_AB_DE : AB / DE = 1 / 2)
variable (BC_length : BC = 2)

-- Prove that EF = 4
theorem length_of_EF (h1 : sim_ABC_DEF) (h2 : ratio_AB_DE) (h3 : BC_length) : EF = 4 :=
sorry

end length_of_EF_l675_675462


namespace limit_interested_is_four_l675_675141

-- Define the function f and its derivative at x0
variable (f : ℝ → ℝ)
variable (x0 : ℝ)
variable (y0 : ℝ)
variable (f' : ℝ → ℝ)

-- Given that the tangent line at (x0, y0) is y = 2x + 1
axiom tangent_line : ∀ (x : ℝ), (f(x) - y0 = 2 * (x - x0))

-- Define the derivative at x0
axiom deriv_f_at_x0 : f' x0 = 2

-- Define the limit we are interested in
noncomputable def limit_interested (f : ℝ → ℝ) (x0 : ℝ) : ℝ :=
  lim (λ Δx, (f(x0) - f(x0 - 2 * Δx)) / Δx)

-- Problem statement
theorem limit_interested_is_four : limit_interested f x0 = 4 := sorry

end limit_interested_is_four_l675_675141


namespace negation_of_exists_statement_l675_675688

theorem negation_of_exists_statement :
  (¬ (∃ x_0 : ℝ, x_0^2 > 0)) ↔ (∀ x : ℝ, x^2 ≤ 0) :=
begin
  sorry
end

end negation_of_exists_statement_l675_675688


namespace infinite_solutions_eq_l675_675644

/-
Proving that the equation x - y + z = 1 has infinite solutions under the conditions:
1. x, y, z are distinct positive integers.
2. The product of any two numbers is divisible by the third one.
-/
theorem infinite_solutions_eq (x y z : ℕ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x) 
(h4 : ∃ m n k : ℕ, x = m * n ∧ y = n * k ∧ z = m * k)
(h5 : (x*y) % z = 0) (h6 : (y*z) % x = 0) (h7 : (z*x) % y = 0) : 
∃ (m : ℕ), x - y + z = 1 ∧ x > 0 ∧ y > 0 ∧ z > 0 :=
by sorry

end infinite_solutions_eq_l675_675644


namespace nina_money_l675_675254

-- Definitions based on the problem's conditions
def original_widgets := 15
def reduced_widgets := 25
def price_reduction := 5

-- The statement
theorem nina_money : 
  ∃ (W : ℝ), 15 * W = 25 * (W - 5) ∧ 15 * W = 187.5 :=
by
  sorry

end nina_money_l675_675254


namespace value_of_a_l675_675482

noncomputable def f (x : ℝ) : ℝ := x^2 + 10
noncomputable def g (x : ℝ) : ℝ := x^2 - 5

theorem value_of_a (a : ℝ) (h₁ : a > 0) (h₂ : f (g a) = 18) :
  a = Real.sqrt (5 + 2 * Real.sqrt 2) ∨ a = Real.sqrt (5 - 2 * Real.sqrt 2) := 
by
  sorry

end value_of_a_l675_675482


namespace hundredth_term_of_sequence_l675_675073

theorem hundredth_term_of_sequence : 
  ∃ (seq : ℕ → ℕ), 
    (∀ n, seq n = ∑ i in (finset.range n).filter (λ x, nat.bit_test n x), 3^i) ∧ 
    seq 99 = 981 :=
sorry

end hundredth_term_of_sequence_l675_675073


namespace call_duration_equal_costs_l675_675764

def planA_cost (x : ℝ) : ℝ := 
  if x <= 6 then 0.60 else 0.60 + (x - 6) * 0.06

def planB_cost (x : ℝ) : ℝ := 
  x * 0.08

theorem call_duration_equal_costs (x : ℝ) : 
  planA_cost x = planB_cost x ↔ x = 12 := by
  sorry

end call_duration_equal_costs_l675_675764


namespace find_vertex_coordinates_l675_675327

theorem find_vertex_coordinates :
  ∃ (h k : ℝ), (∀ (x : ℝ), -(x + 2) ^ 2 + 6 = -(x - h) ^ 2 + k) ∧ (h, k) = (-2, 6) :=
by
  use (-2, 6)
  -- proof is omitted
  sorry

end find_vertex_coordinates_l675_675327


namespace find_alpha_l675_675479

open Real

def alpha_is_acute (α : ℝ) : Prop := 0 < α ∧ α < π / 2

theorem find_alpha (α : ℝ) (h1 : alpha_is_acute α) (h2 : sin (α - 10 * (pi / 180)) = sqrt 3 / 2) : α = 70 * (pi / 180) :=
sorry

end find_alpha_l675_675479


namespace jane_usable_A4_sheets_l675_675590

-- Define conditions
def total_brown_A4 := 28
def less_than_70_intact_brown_A4 := 3
def total_usable_brown_A4 := total_brown_A4 - less_than_70_intact_brown_A4

def total_yellow_A4 := 18
def less_than_70_intact_yellow_A4 := 5
def damaged_stuck_together_yellow_A4 := (less_than_70_intact_yellow_A4 / 2).toInt
def total_usable_yellow_A4 := total_yellow_A4 - less_than_70_intact_yellow_A4 + damaged_stuck_together_yellow_A4

def total_yellow_A3 := 9
def less_than_70_intact_yellow_A3 := 2
def damaged_stuck_together_yellow_A3 := (less_than_70_intact_yellow_A3 / 2).toInt
def total_usable_yellow_A3 := total_yellow_A3 - less_than_70_intact_yellow_A3 + damaged_stuck_together_yellow_A3

def total_pink_A2 := 10
def less_than_70_intact_pink_A2 := 2
def damaged_stuck_together_pink_A2 := (less_than_70_intact_pink_A2 / 2).toInt
def total_usable_pink_A2 := total_pink_A2 - less_than_70_intact_pink_A2 + damaged_stuck_together_pink_A2

-- Define the statement to prove
theorem jane_usable_A4_sheets :
  total_usable_brown_A4 + total_usable_yellow_A4 = 40 :=
by
  sorry

end jane_usable_A4_sheets_l675_675590


namespace triangle_BC_length_l675_675554

-- Define the triangle and given angles and side lengths
def Triangle (A B C : Type) := {
  angle_B : ℝ,
  side_AC : ℝ,
  side_AB : ℝ
}

-- Prove that the length of side BC is 3 given the conditions
theorem triangle_BC_length (A B C : Type)
  [Triangle A B C] (h₁ : A.angle_B = 120)
  (h₂ : A.side_AC = sqrt 19) (h₃ : A.side_AB = 2) :  
  ∃ (BC : ℝ), BC = 3 :=
by
  sorry

end triangle_BC_length_l675_675554


namespace bricks_required_for_courtyard_l675_675771

/-- 
A courtyard is 45 meters long and 25 meters broad needs to be paved with bricks of 
dimensions 15 cm by 7 cm. What will be the total number of bricks required?
-/
theorem bricks_required_for_courtyard 
  (courtyard_length : ℕ) (courtyard_width : ℕ)
  (brick_length : ℕ) (brick_width : ℕ)
  (H1 : courtyard_length = 4500) (H2 : courtyard_width = 2500)
  (H3 : brick_length = 15) (H4 : brick_width = 7) :
  let courtyard_area_cm : ℕ := courtyard_length * courtyard_width
  let brick_area_cm : ℕ := brick_length * brick_width
  let total_bricks : ℕ := (courtyard_area_cm + brick_area_cm - 1) / brick_area_cm
  total_bricks = 107143 := by
  sorry

end bricks_required_for_courtyard_l675_675771


namespace geometric_sequence_max_value_log_l675_675109

noncomputable def max_value_log_expression (a : ℕ → ℝ) :=
  log 2 (a 1) - log 2 (1 / (a 2)) + log 2 (a 9) - log 2 (1 / (a 10))

theorem geometric_sequence_max_value_log
  (a : ℕ → ℝ) (r : ℝ) (h_geo : ∀ n, a (n + 1) = a n * r)
  (h_pos : ∀ n, a n > 0)
  (h_sum_a3_a8 : a 3 + a 8 = 4) :
  max_value_log_expression a ≤ 4 :=
sorry

end geometric_sequence_max_value_log_l675_675109


namespace sum_first_49_odd_numbers_l675_675348

theorem sum_first_49_odd_numbers : (49^2 = 2401) :=
by
  sorry

end sum_first_49_odd_numbers_l675_675348


namespace determine_n_l675_675239

noncomputable def p (x : ℕ) : ℝ := sorry  -- Polynomial function (definition abstracted away)

def poly_deg (p : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ x, degree p = 3 * n

def condition1 (p : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ m, m ∈ (List.range (n + 1)).map (λ x, 3 * x) → p m = 2

def condition2 (p : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ m, m ∈ (List.range (n)).map (λ x, 3 * x + 1) → p m = 1

def condition3 (p : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ m, m ∈ (List.range (n)).map (λ x, 3 * x + 2) → p m = 0

def condition4 (p : ℕ → ℝ) (n : ℕ) : Prop :=
  p (3 * n + 1) = 730

theorem determine_n :
  ∃ n : ℕ, n = 4 ∧
    (poly_deg p n) ∧
    (condition1 p n) ∧
    (condition2 p n) ∧
    (condition3 p n) ∧
    (condition4 p n) := sorry

end determine_n_l675_675239


namespace birthday_pizza_problem_l675_675100

theorem birthday_pizza_problem (m : ℕ) (h1 : m > 11) (h2 : 55 % m = 0) : 10 + 55 / m = 13 := by
  sorry

end birthday_pizza_problem_l675_675100


namespace multiple_of_6_and_9_false_statements_l675_675287

theorem multiple_of_6_and_9_false_statements (c d : ℤ) (hc : ∃ m : ℤ, c = 6 * m) (hd : ∃ n : ℤ, d = 9 * n) :
  ¬ ((even (c + d)) ∧ (∃ k : ℤ, c + d = 6 * k) ∧ (∃ l : ℤ, c + d = 9 * l) ∧ (∀ l : ℤ, c + d ≠ 9 * l)) :=
by
  sorry

end multiple_of_6_and_9_false_statements_l675_675287


namespace area_of_enclosed_region_l675_675572

noncomputable def f (a x : ℝ) : ℝ := a * sin (a * x) + cos (a * x)
noncomputable def g (a : ℝ) : ℝ := real.sqrt (a^2 + 1)

theorem area_of_enclosed_region (a : ℝ) (h : a > 0) : 
  ∫ x in (0 : ℝ)..(2 * real.pi / a), f a x - g a = 2 * real.pi * real.sqrt (a^2 + 1) / a :=
sorry

end area_of_enclosed_region_l675_675572


namespace sum_digits_10_pow_1001_minus_9_l675_675096

theorem sum_digits_10_pow_1001_minus_9 : 
  ∑ i in (10 ^ 1001 - 9).digits, i = 9001 := 
by
  sorry

end sum_digits_10_pow_1001_minus_9_l675_675096


namespace final_number_is_correct_l675_675391

def initial_number := 9
def doubled_number (x : ℕ) := x * 2
def added_number (x : ℕ) := x + 13
def trebled_number (x : ℕ) := x * 3

theorem final_number_is_correct : trebled_number (added_number (doubled_number initial_number)) = 93 := by
  sorry

end final_number_is_correct_l675_675391


namespace volume_of_sphere_l675_675571

theorem volume_of_sphere
  (r : ℝ) (V : ℝ)
  (h₁ : r = 1/3)
  (h₂ : 2 * r = (16/9 * V)^(1/3)) :
  V = 1/6 :=
  sorry

end volume_of_sphere_l675_675571


namespace mow_time_approx_1_09_l675_675211

def mow_lawn_time (length width : ℝ) (mower_width mower_overlap : ℝ) (mowing_rate : ℝ) : ℝ := 
  let effective_swath_width := (mower_width - mower_overlap) / 12
  let number_of_strips := width / effective_swath_width
  let total_distance := number_of_strips * length
  total_distance / mowing_rate

theorem mow_time_approx_1_09 :
  ∀ (length width mower_width mower_overlap mowing_rate : ℝ),
    length = 100 ∧ width = 120 ∧ mower_width = 30 ∧ mower_overlap = 6 ∧ mowing_rate = 5500 →
      abs (mow_lawn_time length width mower_width mower_overlap mowing_rate - 1.09) < 0.01 :=
by
  intros length width mower_width mower_overlap mowing_rate h_cond
  rw [mow_lawn_time]
  cases h_cond with h_length h_rest
  cases h_rest with h_width h_rest
  cases h_rest with h_mower_width h_rest
  cases h_rest with h_mower_overlap h_mowing_rate
  sorry

end mow_time_approx_1_09_l675_675211


namespace evaluate_expression_l675_675816

theorem evaluate_expression :
  (1/2)^(-5) + log 2 2 + log 2 5 = 33 :=
  sorry

end evaluate_expression_l675_675816


namespace bulbs_arrangement_l675_675710

theorem bulbs_arrangement :
  let blue_bulbs := 5
  let red_bulbs := 8
  let white_bulbs := 11
  let total_non_white_bulbs := blue_bulbs + red_bulbs
  let total_gaps := total_non_white_bulbs + 1
  (Nat.choose 13 5) * (Nat.choose total_gaps white_bulbs) = 468468 :=
by
  sorry

end bulbs_arrangement_l675_675710


namespace intersect_at_least_10_circles_l675_675567

-- Define the problem conditions and the theorem
theorem intersect_at_least_10_circles
  (centers : Fin 100 → (ℝ × ℝ))
  (h_radius : ∀ i, ∃ r, r = 1)
  (h_area : ∀ (i j k : Fin 100), 
    1 / 2 * |centers i].1 * (centers j].2 - centers k].2) +
             centers j].1 * (centers k].2 - centers i].2) +
             centers k].1 * (centers i].2 - centers j].2)| ≤ 100) :
  ∃ (line : ℝ × ℝ), 
    card {c | ∃ i, centers i = c ∧ line_intersects_circle c 1 line} ≥ 10 :=
  sorry

end intersect_at_least_10_circles_l675_675567


namespace box_volume_condition_l675_675742

theorem box_volume_condition : 
  {x : ℕ | (x + 3) * (x - 3) * (x^2 + 9) < 500 ∧ x > 3}.finite.card = 2 :=
by
  sorry

end box_volume_condition_l675_675742


namespace regular_prism_volume_l675_675017

noncomputable def volume_of_prism (R : ℝ) (h : ℝ) (r : ℝ) (V : ℝ) : Prop :=
  let A := (sqrt 3 / 4) * r^2
  V = A * h

theorem regular_prism_volume
  (R h r V : ℝ)
  (K_midpoint : ∀ (CC1 : ℝ), K = CC1 / 2)
  (C1D_diameter : C1D = 2 * R)
  (DK_distance : DK = 2)
  (DA_distance : DA = sqrt 6)
  (h_eq : h^2 = 4 * R^2 - 4)
  (r_eq : 2 * R^2 = r^2 + 4)
  (volume_eq : V = 2) :
  volume_of_prism R h r V :=
by
  sorry

end regular_prism_volume_l675_675017


namespace cost_per_minute_advertising_l675_675826

theorem cost_per_minute_advertising :
  ∀ (n : ℕ) (m : ℕ) (total_cost : ℕ),
  (n = 5) →  
  (m = 3) →  
  (total_cost = 60000) →  
  (total_cost / (n * m) = 4000) :=
by
  intros n m total_cost h_n h_m h_cost
  rw [h_n, h_m, h_cost]
  norm_num
  sorry

end cost_per_minute_advertising_l675_675826


namespace cos_B_equals_3_over_4_l675_675939

theorem cos_B_equals_3_over_4
  (a b c : ℝ) 
  (h1 : c = 2 * a) 
  (h2 : b^2 = a * c) : 
  real.cos (real.acos ((a^2 + c^2 - b^2) / (2 * a * c))) = 3 / 4 := 
by 
  -- sorry is added to skip the proof
  sorry

end cos_B_equals_3_over_4_l675_675939


namespace curve_C2_symmetry_l675_675246

noncomputable def curve_C (a b c : ℝ) (h : a ≠ 0) := ∀ x : ℝ, y = a * x^2 + b * x + c

theorem curve_C2_symmetry (a b c x : ℝ) (h : a ≠ 0) : ∀ x : ℝ, (∃ y : ℝ, y = a * x^2 + b * x + c) → (∃ y2 : ℝ, y2 = -a * x^2 + b * x - c) :=
by
  sorry

end curve_C2_symmetry_l675_675246


namespace initial_water_amount_l675_675760

theorem initial_water_amount (W : ℝ) (h1 : 0.006 * 50 = 0.03 * W) : W = 10 :=
by
  -- Proof steps would go here
  sorry

end initial_water_amount_l675_675760


namespace part1_inequality_l675_675868

noncomputable def f (x : ℝ) : ℝ := x - 2
noncomputable def g (x m : ℝ) : ℝ := x^2 - 2 * m * x + 4

theorem part1_inequality (m : ℝ) : (∀ x : ℝ, g x m > f x) ↔ (m ∈ Set.Ioo (-Real.sqrt 6 - (1/2)) (Real.sqrt 6 - (1/2))) :=
sorry

end part1_inequality_l675_675868


namespace find_decreasing_function_l675_675739

noncomputable def is_decreasing {α : Type*} [preorder α] (f : α → α) (s : set α) : Prop :=
∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f y < f x

theorem find_decreasing_function :
  let I := set.Ioi (0 : ℝ)
  let f1 : ℝ → ℝ := λ x, x + 1
  let f2 : ℝ → ℝ := λ x, log (1/2 : ℝ) x
  let f3 : ℝ → ℝ := λ x, 2^x
  let f4 : ℝ → ℝ := λ x, -(x - 1)^2
  is_decreasing f2 I ∧
  ¬is_decreasing f1 I ∧
  ¬is_decreasing f3 I ∧
  ¬is_decreasing f4 I := by
  sorry

end find_decreasing_function_l675_675739


namespace Joan_pays_139_20_l675_675213

noncomputable def JKL : Type := ℝ × ℝ × ℝ

def conditions (J K L : ℝ) : Prop :=
  J + K + L = 600 ∧
  2 * J = K + 74 ∧
  L = K + 52

theorem Joan_pays_139_20 (J K L : ℝ) (h : conditions J K L) : J = 139.20 :=
by
  sorry

end Joan_pays_139_20_l675_675213


namespace power_of_power_negative_fraction_power_l675_675425

-- First proof problem: (x^2)^3 = x^6
theorem power_of_power (x : ℝ) : (x^2)^3 = x^6 :=
  sorry

-- Second proof problem: (-1/2)^(-2) = 4
theorem negative_fraction_power : (- (1/2 : ℝ))^(-2) = 4 :=
  sorry

end power_of_power_negative_fraction_power_l675_675425
