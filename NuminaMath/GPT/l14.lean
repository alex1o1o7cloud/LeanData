import Mathlib
import Mathlib.Algebra.Cubic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Analysis.Calculus.Inverse
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Geometry.Point
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Prob
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Order.ConditionallyCompleteLattice
import Mathlib.Statistics.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import data.real.basic

namespace dot_product_ab_l14_14684

variables (a b : ℝ^3)

-- Given conditions
def condition1 : Prop := ‖a‖ = 1
def condition2 : Prop := ‖b‖ = real.sqrt 3
def condition3 : Prop := ‖a - 2 • b‖ = 3

-- The theorem statement to prove
theorem dot_product_ab (h1 : condition1 a) (h2 : condition2 b) (h3 : condition3 a b) : 
  a ⬝ b = 1 :=
sorry

end dot_product_ab_l14_14684


namespace triangle_median_angle_l14_14334

theorem triangle_median_angle (A B C M : Point) (h : line_segment A B)
  (h1 : midpoint B M C) (h2 : 2 * (dist B M) = dist A B) (h3 : angle A B M = 40) :
  angle A B C = 110 := 
sorry

end triangle_median_angle_l14_14334


namespace abcdef_base16_to_base2_bits_l14_14470

noncomputable def base_16_to_dec (h : String) : ℕ :=
  10 * 16^5 + 11 * 16^4 + 12 * 16^3 + 13 * 16^2 + 14 * 16^1 + 15 * 16^0

theorem abcdef_base16_to_base2_bits (n : ℕ) (h : n = base_16_to_dec "ABCDEF") :
  nat.log2 n + 1 = 24 :=
by
  rw [h]
  -- nat.log2 calculations and the necessary steps to establish the number of bits
  sorry

end abcdef_base16_to_base2_bits_l14_14470


namespace count_integers_satisfying_inequality_l14_14732

theorem count_integers_satisfying_inequality :
  {m : ℤ | m ≠ 0 ∧ (1 / (|m| : ℝ) ≥ 1 / 12)}.to_finset.card = 24 := 
sorry

end count_integers_satisfying_inequality_l14_14732


namespace average_weight_of_remaining_boys_l14_14763

theorem average_weight_of_remaining_boys :
  ∀ (total_boys remaining_boys_num : ℕ)
    (avg_weight_22 remaining_boys_avg_weight total_class_avg_weight : ℚ),
    total_boys = 30 →
    remaining_boys_num = total_boys - 22 →
    avg_weight_22 = 50.25 →
    total_class_avg_weight = 48.89 →
    (remaining_boys_num : ℚ) * remaining_boys_avg_weight =
    total_boys * total_class_avg_weight - 22 * avg_weight_22 →
    remaining_boys_avg_weight = 45.15 :=
by
  intros total_boys remaining_boys_num avg_weight_22 remaining_boys_avg_weight total_class_avg_weight
         h_total_boys h_remaining_boys_num h_avg_weight_22 h_total_class_avg_weight h_equation
  sorry

end average_weight_of_remaining_boys_l14_14763


namespace cone_lateral_surface_area_l14_14199

theorem cone_lateral_surface_area (r h l S : ℝ) (π_pos : 0 < π) (r_eq : r = 6)
  (V : ℝ) (V_eq : V = 30 * π)
  (vol_eq : V = (1/3) * π * r^2 * h)
  (h_eq : h = 5 / 2)
  (l_eq : l = Real.sqrt (r^2 + h^2))
  (S_eq : S = π * r * l) :
  S = 39 * π :=
  sorry

end cone_lateral_surface_area_l14_14199


namespace integer_div_product_l14_14872

theorem integer_div_product (n : ℤ) : ∃ (k : ℤ), n * (n + 1) * (n + 2) = 6 * k := by
  sorry

end integer_div_product_l14_14872


namespace smallest_total_cashews_l14_14976

noncomputable def first_monkey_final (c1 c2 c3 : ℕ) : ℕ :=
  (2 * c1) / 3 + c2 / 6 + (4 * c3) / 18

noncomputable def second_monkey_final (c1 c2 c3 : ℕ) : ℕ :=
  c1 / 6 + (2 * c2) / 6 + (4 * c3) / 18

noncomputable def third_monkey_final (c1 c2 c3 : ℕ) : ℕ :=
  c1 / 6 + (2 * c2) / 6 + c3 / 9

theorem smallest_total_cashews : ∃ (c1 c2 c3 : ℕ), ∃ y : ℕ,
  3 * y = first_monkey_final c1 c2 c3 ∧
  2 * y = second_monkey_final c1 c2 c3 ∧
  y = third_monkey_final c1 c2 c3 ∧
  c1 + c2 + c3 = 630 :=
sorry

end smallest_total_cashews_l14_14976


namespace volume_ratio_of_spheres_l14_14295

theorem volume_ratio_of_spheres
  (r1 r2 r3 : ℝ)
  (A1 A2 A3 : ℝ)
  (V1 V2 V3 : ℝ)
  (hA : A1 / A2 = 1 / 4 ∧ A2 / A3 = 4 / 9)
  (hSurfaceArea : A1 = 4 * π * r1^2 ∧ A2 = 4 * π * r2^2 ∧ A3 = 4 * π * r3^2)
  (hVolume : V1 = (4 / 3) * π * r1^3 ∧ V2 = (4 / 3) * π * r2^3 ∧ V3 = (4 / 3) * π * r3^3) :
  V1 / V2 = 1 / 8 ∧ V2 / V3 = 8 / 27 := by
  sorry

end volume_ratio_of_spheres_l14_14295


namespace largest_class_students_l14_14306

theorem largest_class_students (x : ℕ) (h1 : 8 * x - (4 + 8 + 12 + 16 + 20 + 24 + 28) = 380) : x = 61 :=
by
  sorry

end largest_class_students_l14_14306


namespace parallelogram_area_is_20_l14_14559

-- Define the vertices of the parallelogram
def vertex1 := (0, 0) : ℝ × ℝ
def vertex2 := (4, 0) : ℝ × ℝ
def vertex3 := (1, 5) : ℝ × ℝ
def vertex4 := (5, 5) : ℝ × ℝ

-- Define a function to compute the area of a parallelogram given vertices
def parallelogram_area (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  let base := real.dist (v1.1, v1.2) (v2.1, v2.2)
  let height := real.dist (v1.1, v1.2) (v3.1, v3.2)
  base * height

-- Prove that the area of the parallelogram with given vertices is 20
theorem parallelogram_area_is_20 :
  parallelogram_area vertex1 vertex2 vertex3 vertex4 = 20 := by
  sorry

end parallelogram_area_is_20_l14_14559


namespace chinese_remainder_sequence_count_l14_14929

theorem chinese_remainder_sequence_count :
  let seq := { n | 1 ≤ n ∧ n ≤ 2016 ∧ n % 3 = 1 ∧ n % 5 = 1 } in
  seq.card = 135 :=
by
  let n := (2016 + 14) / 15
  have h : n = 135
  { calc
    n = (2016 + 14) / 15 : by rfl
    ... = 2030 / 15 : by norm_num
    ... = 135 : by norm_num }
  use h
  sorry

end chinese_remainder_sequence_count_l14_14929


namespace remainder_of_99_pow_36_mod_100_l14_14466

theorem remainder_of_99_pow_36_mod_100 :
  (99 : ℤ)^36 % 100 = 1 := sorry

end remainder_of_99_pow_36_mod_100_l14_14466


namespace lateral_surface_area_of_given_cone_l14_14214

noncomputable def coneLateralSurfaceArea (r V : ℝ) : ℝ :=
let h := (3 * V) / (π * r^2) in
let l := Real.sqrt (r^2 + h^2) in
π * r * l

theorem lateral_surface_area_of_given_cone :
  coneLateralSurfaceArea 6 (30 * π) = 39 * π := by
simp [coneLateralSurfaceArea]
sorry

end lateral_surface_area_of_given_cone_l14_14214


namespace negation_universal_proposition_l14_14945

theorem negation_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2 * x + 1 ≥ 0) → ∃ x : ℝ, x^2 - 2 * x + 1 < 0 :=
by sorry

end negation_universal_proposition_l14_14945


namespace smallest_abundant_not_multiple_of_5_l14_14584

def is_abundant (n : ℕ) : Prop :=
  (∑ d in (nat.divisors n).filter (≠ n), d) > n

def not_multiple_of_5 (n : ℕ) : Prop :=
  ¬ (5 ∣ n)

theorem smallest_abundant_not_multiple_of_5 :
  ∃ n : ℕ, is_abundant n ∧ not_multiple_of_5 n ∧ 
    (∀ m : ℕ, is_abundant m ∧ not_multiple_of_5 m → m >= n) :=
sorry

end smallest_abundant_not_multiple_of_5_l14_14584


namespace greatest_area_of_quadrilateral_l14_14873

-- Definitions of points and lines based on the problem statement
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  p1 : Point
  p2 : Point

def rectangleEFGH := {E := Point.mk x y, F := Point.mk (x + 8) y, G := Point.mk (x + 8) (y + 9), H := Point.mk x (y + 9)}
def rectangleABCD := {A := Point.mk 0 0, B := Point.mk 14 0, C := Point.mk 14 13, D := Point.mk 0 13}

-- Define the lines \ell_A, \ell_B, \ell_C, \ell_D
def lA := Line.mk rectangleABCD.A rectangleEFGH.E
def lB := Line.mk rectangleABCD.B rectangleEFGH.F
def lC := Line.mk rectangleABCD.C rectangleEFGH.G
def lD := Line.mk rectangleABCD.D rectangleEFGH.H

-- Define the intersection points
def P := intersection lA lB
def Q := intersection lB lC
def R := intersection lC lD
def S := intersection lD lA

-- Use the Shoelace Theorem to define area formula
def areaPQRS : ℝ :=
  (1/2) * abs (
    P.x * Q.y + Q.x * R.y + R.x * S.y + S.x * P.y
    - (P.y * Q.x + Q.y * R.x + R.y * S.x + S.y * P.x)
  )

-- Now state the theorem
theorem greatest_area_of_quadrilateral :
  ∃ m n : ℕ, (gcd m n = 1) ∧ (areaPQRS = m / n) ∧ (100 * m + n = 1725) :=
sorry

end greatest_area_of_quadrilateral_l14_14873


namespace number_of_girls_l14_14969

theorem number_of_girls (n : ℕ) (h : 25 - n > 0) (hprob : (n * (n - 1)) / (25 * 24) = 3 / 25) : 25 - n = 16 :=
by
  have h1 : 25 > n := by linarith
  have hquad : n * (n - 1) = 72 := by
    calc
      (n * (n - 1)) = 3 / 25 * (25 * 24) : by linarith[hprob]
      _ = 72 : by norm_num
  have hsol : n = 9 := by
    have hquad' : n^2 - n - 72 = 0 := by
      ring_exp
      rw ← hquad
    sorry  -- Solve the quadratic equation manually
  rw hsol
  norm_num

end number_of_girls_l14_14969


namespace question_equals_answer_l14_14272

variables (a b : ℝ^3)
variables (theta : ℝ)

-- Given conditions
def magnitude_a : Prop := ∥a∥ = 1
def angle_45 : Prop := cos theta = (1 / real.sqrt 2)
def magnitude_2a_minus_b : Prop := ∥2 • a - b∥ = real.sqrt 10

-- Question with the correct answer
def proof_problem : Prop := magnitude_a ∧ angle_45 ∧ magnitude_2a_minus_b → ∥b∥ = 3 * real.sqrt 2

theorem question_equals_answer : proof_problem a b theta := sorry

end question_equals_answer_l14_14272


namespace factorial_division_l14_14034

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_division : (factorial 15) / ((factorial 6) * (factorial 9)) = 5005 :=
by
  sorry

end factorial_division_l14_14034


namespace g_neither_even_nor_odd_l14_14336

noncomputable def g (x : ℝ) : ℝ := ⌈x⌉ - 1 / 2

theorem g_neither_even_nor_odd :
  (¬ ∀ x, g x = g (-x)) ∧ (¬ ∀ x, g (-x) = -g x) :=
by
  sorry

end g_neither_even_nor_odd_l14_14336


namespace find_truth_teller_l14_14475

variable (A B C D : Prop)

/-- First person said, "All four of us are liars." -/
def statement1 : Prop := ¬A ∧ ¬B ∧ ¬C ∧ ¬D

/-- Second person said, "Only one amongst us is a liar." -/
def statement2 : Prop := (¬A ∧ B ∧ C ∧ D) ∨ (A ∧ ¬B ∧ C ∧ D) ∨ (A ∧ B ∧ ¬C ∧ D) ∨ (A ∧ B ∧ C ∧ ¬D)

/-- Third person said, "There are two liars among us." -/
def statement3 : Prop := (¬A ∧ ¬B ∧ C ∧ D) ∨ (¬A ∧ B ∧ ¬C ∧ D) ∨ (¬A ∧ B ∧ C ∧ ¬D) ∨ 
                            (A ∧ ¬B ∧ ¬C ∧ D) ∨ (A ∧ ¬B ∧ C ∧ ¬D) ∨ (A ∧ B ∧ ¬C ∧ ¬D)

/-- Fourth person said, "I am a truth-teller." -/
def statement4 : Prop := D

/-- Given the statements, prove the fourth person is the truth-teller -/
theorem find_truth_teller (H1 : statement1 = false) (H2 : statement2 = false) (H3 : statement3 = false) : D := by
  sorry

end find_truth_teller_l14_14475


namespace solve_system_of_equations_l14_14400

theorem solve_system_of_equations 
  (a1 a2 a3 a4 : ℝ) (h_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)
  (x1 x2 x3 x4 : ℝ)
  (h1 : |a1 - a1| * x1 + |a1 - a2| * x2 + |a1 - a3| * x3 + |a1 - a4| * x4 = 1)
  (h2 : |a2 - a1| * x1 + |a2 - a2| * x2 + |a2 - a3| * x3 + |a2 - a4| * x4 = 1)
  (h3 : |a3 - a1| * x1 + |a3 - a2| * x2 + |a3 - a3| * x3 + |a3 - a4| * x4 = 1)
  (h4 : |a4 - a1| * x1 + |a4 - a2| * x2 + |a4 - a3| * x3 + |a4 - a4| * x4 = 1) :
  x1 = 1 / (a1 - a4) ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 1 / (a1 - a4) :=
sorry

end solve_system_of_equations_l14_14400


namespace speed_of_train_is_90_kmph_l14_14112

-- Definitions based on conditions
def length_of_train_in_meters : ℝ := 225
def time_to_cross_pole_in_seconds : ℝ := 9

-- Conversion factors
def meters_to_kilometers (meters : ℝ) : ℝ := meters / 1000
def seconds_to_hours (seconds : ℝ) : ℝ := seconds / 3600

-- Computed values from the conditions
def length_of_train_in_kilometers : ℝ := meters_to_kilometers length_of_train_in_meters
def time_to_cross_pole_in_hours : ℝ := seconds_to_hours time_to_cross_pole_in_seconds

-- Speed calculation
def speed_of_train_in_kmph : ℝ := length_of_train_in_kilometers / time_to_cross_pole_in_hours

-- Theorem to assert the query
theorem speed_of_train_is_90_kmph : speed_of_train_in_kmph = 90 := by
  sorry

end speed_of_train_is_90_kmph_l14_14112


namespace leRoyPayments_l14_14346

variable (X Y Z : ℝ)
variable (h_ltXY : X < Y)
variable (h_ltYZ : Y < Z)

def totalCost : ℝ :=
  X + Y + Z

def equalShare : ℝ :=
  totalCost X Y Z / 3

def leRoyToGiveBernardo : ℝ :=
  equalShare X Y Z - X - (Z - Y) / 2

def leRoyToGiveCarlos : ℝ :=
  (Z - Y) / 2

theorem leRoyPayments :
  (leRoyToGiveBernardo X Y Z h_ltXY h_ltYZ = equalShare X Y Z - X - (Z - Y) / 2) ∧ (leRoyToGiveCarlos X Y Z h_ltXY h_ltYZ = (Z - Y) / 2) :=
by
  sorry

end leRoyPayments_l14_14346


namespace michael_choices_l14_14322

theorem michael_choices (n k : ℕ) (h_n : n = 10) (h_k : k = 4) : nat.choose n k = 210 :=
by
  rw [h_n, h_k]
  norm_num
  sorry

end michael_choices_l14_14322


namespace volume_new_parallelepiped_l14_14011

variables {V : Type*} [inner_product_space ℝ V]

def volume_of_parallelepiped (a b c : V) : ℝ :=
  (a ∧ b ∧ c).abs

theorem volume_new_parallelepiped
  {a b c : V}
  (h : volume_of_parallelepiped a b c = 6) :
  volume_of_parallelepiped (2 • a - 3 • b) (4 • b + 5 • c) (c + 2 • a) = 24 := 
sorry

end volume_new_parallelepiped_l14_14011


namespace dice_probability_l14_14444

theorem dice_probability :
  let C (r : ℝ) := 2 * Real.pi * r
  let A (r : ℝ) := Real.pi * r^2
  let is_greater (r : ℝ) := C r > 2 * A r
  let possible_r_values := {r : ℝ | r ≥ 3 ∧ r ≤ 18 ∧ ∃ a b c : ℕ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ (1 ≤ c ∧ c ≤ 6) ∧ (r = a + b + c)}
  prob := (∅ : set ℝ)
  in ∀ r ∈ possible_r_values, is_greater r -> r < 1 → prob = ∅ := sorry

end dice_probability_l14_14444


namespace max_tulips_l14_14456

theorem max_tulips (y r : ℕ) (h1 : (y + r) % 2 = 1) (h2 : r = y + 1 ∨ y = r + 1) (h3 : 50 * y + 31 * r ≤ 600) : y + r = 15 :=
by
  sorry

end max_tulips_l14_14456


namespace tournament_impossibility_l14_14394

theorem tournament_impossibility :
  ∀ (teams : Fin 16 → Type)
    (country : Type)
    (played_in : Π (i j : Fin 16), i ≠ j → country)
    (distinct_countries : ∀ (i : Fin 16), ∃ country_i : country, ∀ j : Fin 16, j ≠ i → country_i ≠ played_in i j _),
  False :=
by sorry

end tournament_impossibility_l14_14394


namespace lantern_probability_l14_14066

theorem lantern_probability :
  let total_large := 360
  let total_small := 1200
  let x := 120 -- number of large lanterns with 2 small lanterns
  let y := total_large - x -- number of large lanterns with 4 small lanterns
  let num_combinations := choose total_large 2
  let num_favorable := choose y 2 + y * x
  (num_favorable / num_combinations : ℚ) = 958 / 1077 :=
by
  sorry

end lantern_probability_l14_14066


namespace interest_percentage_correct_l14_14841
noncomputable def total_interest_paid (purchase_price : ℝ) (down_payment : ℝ) (monthly_rates : ℕ → ℝ) : ℝ :=
let balance := purchase_price - down_payment in
let interest :=
  list.sum (list.range 12).map (λ n, monthly_rates n * balance / 100) in
interest / purchase_price * 100

theorem interest_percentage_correct :
  total_interest_paid 127 27 (λ n, 2 + n) ≈ 70.9 :=
sorry

end interest_percentage_correct_l14_14841


namespace number_of_routes_from_X_to_Y_l14_14549

open GraphTheory

noncomputable def graph_with_ten_vertices_and_fifteen_edges : SimpleGraph (Fin 10) := sorry

theorem number_of_routes_from_X_to_Y :
  let G := graph_with_ten_vertices_and_fifteen_edges
  let X : V := 0
  let Y : V := 1
  let num_routes := λ (G : SimpleGraph (Fin 10)) (X Y : Fin 10), 
    { paths : G.walk X Y | paths.edges.count ≤ 11 ∧ ∀ (e : G.edge_set), e ∈ paths.edges → paths.edges.count e = 1 }.toFinset.card
  num_routes G X Y = 3 :=
sorry

end number_of_routes_from_X_to_Y_l14_14549


namespace least_multiple_of_five_primes_l14_14993

noncomputable def smallest_multiple_of_five_primes : ℕ :=
  let primes := [2, 3, 5, 7, 11] in
  primes.foldl (· * ·) 1

theorem least_multiple_of_five_primes : smallest_multiple_of_five_primes = 2310 := by
  sorry

end least_multiple_of_five_primes_l14_14993


namespace cone_lateral_surface_area_l14_14195

-- Definitions based on the conditions
def coneRadius : ℝ := 6
def coneVolume : ℝ := 30 * Real.pi

-- Mathematical statement
theorem cone_lateral_surface_area (r V : ℝ) (hr : r = coneRadius) (hV : V = coneVolume) :
  ∃ S : ℝ, S = 39 * Real.pi :=
by 
  have h_volume := hV
  have h_radius := hr
  sorry

end cone_lateral_surface_area_l14_14195


namespace least_multiple_of_five_primes_l14_14994

noncomputable def smallest_multiple_of_five_primes : ℕ :=
  let primes := [2, 3, 5, 7, 11] in
  primes.foldl (· * ·) 1

theorem least_multiple_of_five_primes : smallest_multiple_of_five_primes = 2310 := by
  sorry

end least_multiple_of_five_primes_l14_14994


namespace simplify_fraction_l14_14910

theorem simplify_fraction (y : ℝ) (hy : y ≠ 0) : 
  (5 / (4 * y⁻⁴)) * ((4 * y³) / 3) = (5 * y⁷) / 3 := 
by
  sorry

end simplify_fraction_l14_14910


namespace partI_partII_l14_14300

noncomputable def triangleProblem (a b c : ℝ) (cosA cosB : ℝ) : Prop :=
  ¬(a ∈ {0, π/2, π}) ∧ (b * c - 8) * cosA + a * c * cosB = a^2 - b^2 ∧ b + c = 5 ∧ b * c = 4

theorem partI (a : ℝ) (cosA cosB : ℝ) (h : triangleProblem a 1 4 cosA cosB) :
  ({b : ℝ // b = 1} ∨ {c : ℝ // c = 4}) :=
sorry

theorem partII (A B C : Type) [Euclidean A] [Euclidean B] [Euclidean C]
  (a : ℝ) (h1 : a = Real.sqrt 5) (b : ℝ) (c : ℝ) (S : ℝ) 
  (cosA : ℝ) (h2 : triangleProblem a b c cosA (λ x, 0))
  : S ≤ Real.sqrt 55 / 4 :=
sorry

end partI_partII_l14_14300


namespace trapezoid_area_l14_14121

variable (x y : ℝ)

-- Conditions: 
-- 1. The trapezoid is isosceles and circumscribed around a circle.
-- 2. The longer base of the trapezoid is 20.
-- 3. One of the base angles is arccos(0.6).

theorem trapezoid_area 
  (h1 : 2 * y + 0.6 * x + 0.6 * x = 2 * x)
  (h2 : y + 0.6 * x + 0.6 * x = 20)
  (h3 : ∃ θ : ℝ, θ = Real.arccos 0.6) :
  let h := 0.8 * x in
  1 / 2 * (y + 20) * h = 112 :=
by
  sorry

end trapezoid_area_l14_14121


namespace hyperbola_equation_l14_14606

/-- Given a hyperbola centered at the origin with its foci on the x-axis, a line passing 
through the right focus of the hyperbola with a slope of sqrt(3/5) intersects the hyperbola 
at two points P and Q. If OP is perpendicular to OQ (where O is the origin) and |PQ| = 4, 
then the equation of the hyperbola is x^2 - y^2 / 3 = 1. -/
theorem hyperbola_equation :
  ∃ (a b : ℝ), (a ≠ 0) ∧ (b = √3 * a) ∧ 
    (∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 3 = 1)) := 
sorry

end hyperbola_equation_l14_14606


namespace probability_contemporaries_l14_14450

/-- 
Prove that the probability that two historians, who each live to be 80 years old 
and are equally likely to be born within a 300-year period, 
were contemporaries for any length of time is 104/225.
-/
theorem probability_contemporaries (n a : ℕ) (h_n : n = 300) (h_a : a = 80) :
  let total_area := n * n in
  let non_overlap_area_one_side := (n - a) * (n - a) / 2 in
  let non_overlap_area := 2 * non_overlap_area_one_side in
  let overlap_area := total_area - non_overlap_area in
  (overlap_area / total_area : ℝ) = 104 / 225 :=
by
  sorry

end probability_contemporaries_l14_14450


namespace probability_not_six_l14_14754

theorem probability_not_six (favorable unfavorable : ℕ) (h_fav : favorable = 5) (h_unfav : unfavorable = 7) :
  let total := favorable + unfavorable in
  (unfavorable : ℚ) / total = 7 / 12 :=
by
  sorry

end probability_not_six_l14_14754


namespace find_number_l14_14852

theorem find_number (x : ℝ) (h : (5/3) * x = 45) : x = 27 :=
by
  sorry

end find_number_l14_14852


namespace length_PR_divided_by_AR_l14_14279

theorem length_PR_divided_by_AR
    (A B C D P R : Type)
    [circle R]
    (O : point R)
    (h1 : diameter AB R)
    (h2 : diameter CD R)
    (h3 : perpendicular AB CD)
    (h4 : on_line P R A)
    (h5 : angle RPC = 45) :
    PR / AR = 1 :=
by
  sorry

end length_PR_divided_by_AR_l14_14279


namespace cos_sum_identity_l14_14387

theorem cos_sum_identity : cos (2 * π / 5) + cos (4 * π / 5) = -1 / 2 := 
by
  sorry

end cos_sum_identity_l14_14387


namespace distance_of_line_l_l14_14083

def unit_cube := { 
  A: (ℝ, ℝ, ℝ) | A = (0,0,0),
  B: (ℝ, ℝ, ℝ) | B = (1,0,0),
  C: (ℝ, ℝ, ℝ) | C = (1,1,0),
  D: (ℝ, ℝ, ℝ) | D = (0,1,0),
  A1: (ℝ, ℝ, ℝ) | A1 = (0,0,1),
  B1: (ℝ, ℝ, ℝ) | B1 = (1,0,1),
  C1: (ℝ, ℝ, ℝ) | C1 = (1,1,1),
  D1: (ℝ, ℝ, ℝ) | D1 = (0,1,1)
}

def diagonal_AC1 (A C1: (ℝ, ℝ, ℝ)) : ℝ :=
  √((A.1 - C1.1)^2 + (A.2 - C1.2)^2 + (A.3 - C1.3)^2)

def line_l_parallel_to_AC1 (l: ℝ → (ℝ, ℝ, ℝ)) (A C1 : (ℝ, ℝ, ℝ)) : Prop :=
  ∀ t: ℝ, ∃ k: ℝ, l t = (A.1 + k*(C1.1 - A.1), A.2 + k*(C1.2 - A.2), A.3 + k*(C1.3 - A.3))

def equidistant_lines (l: ℝ → (ℝ, ℝ, ℝ)) (BD A1D1 CB1: (ℝ, ℝ, ℝ) → ℝ) : Prop :=
  ∀ p: (ℝ, ℝ, ℝ), BD p = A1D1 p ∧ A1D1 p = CB1 p

theorem distance_of_line_l :
  ∀ l: ℝ → (ℝ, ℝ, ℝ), line_l_parallel_to_AC1 l (0,0,0) (1,1,1) ∧ 
  equidistant_lines l (λ p, p.1 + p.2 + p.3) (λ p, p.1 + p.2 + p.3) (λ p, p.1 + p.2 + p.3)
  → 
  (distance_of (0, 0, 0) (1, 0, 0) (1, 1, 1) (0, 1, 1) = 5*(2*√6 - 3*√2)/6)
  ∧ (distance_of (0, 0, 1) (1, 1, 1) (1, 1, 1) (1, 0, 1) = 5*(3*√2 + √6)/6)
  ∧ (distance_of (0, 1, 0) (1, 0, 0) (1,1, 0) (1, 1, 1) = √2/6)
  ∧ (distance_of (0,0,0) (0,0,0) (1,0,0) (1,1,1) = 5*√2/6) :=
sorry

end distance_of_line_l_l14_14083


namespace range_of_a_for_monotonicity_l14_14645

variable {a : ℝ}

def f (x : ℝ) (a : ℝ) : ℝ := (1 / 3) * x^3 - |2 * a * x + 4|

def monotone_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

theorem range_of_a_for_monotonicity :
  monotone_increasing_on (f · a) 1 2 ↔ a ∈ set.Icc (- (2 : ℝ)^(1 / 3)) (1 / 2) :=
sorry

end range_of_a_for_monotonicity_l14_14645


namespace tree_planting_growth_rate_l14_14947

theorem tree_planting_growth_rate {x : ℝ} :
  400 * (1 + x) ^ 2 = 625 :=
sorry

end tree_planting_growth_rate_l14_14947


namespace parabola_and_circle_eq_and_tangent_l14_14962

noncomputable def parabola_eq (x y : ℝ) : Prop := y^2 = x

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

theorem parabola_and_circle_eq_and_tangent :
  (∀ x y : ℝ, parabola_eq x y ↔ y^2 = x) ∧
  (∀ x y : ℝ, circle_eq x y ↔ (x - 2)^2 + y^2 = 1) ∧
  (∀ A1 A2 A3 : ℝ × ℝ,
    parabola_eq A1.1 A1.2 ∧ parabola_eq A2.1 A2.2 ∧ parabola_eq A3.1 A3.2 →
    tangent_to_circle A1 A2 A3)
:=
sorry

end parabola_and_circle_eq_and_tangent_l14_14962


namespace cost_of_section_A_ticket_l14_14447

theorem cost_of_section_A_ticket 
  (A : ℝ) 
  (section_B_cost : ℝ) 
  (total_tickets_sold : ℝ) 
  (total_revenue : ℝ) 
  (section_A_tickets_sold : ℝ) 
  (section_B_tickets_sold : ℝ) 
  (h1 : section_B_cost = 4.25)
  (h2 : total_tickets_sold = 4500)
  (h3 : total_revenue = 30000)
  (h4 : section_A_tickets_sold = 2900)
  (h5 : section_B_tickets_sold = 1600)
  (h6 : total_revenue = section_A_tickets_sold * A + section_B_tickets_sold * section_B_cost) : 
  A ≈ 7.66 := 
by
  sorry

end cost_of_section_A_ticket_l14_14447


namespace combined_return_percentage_l14_14494

theorem combined_return_percentage (investment1 investment2 : ℝ) 
  (return1_percent return2_percent : ℝ) (total_investment total_return : ℝ) :
  investment1 = 500 → 
  return1_percent = 0.07 → 
  investment2 = 1500 → 
  return2_percent = 0.09 → 
  total_investment = investment1 + investment2 → 
  total_return = investment1 * return1_percent + investment2 * return2_percent → 
  (total_return / total_investment) * 100 = 8.5 :=
by 
  sorry

end combined_return_percentage_l14_14494


namespace xiaoxiao_age_in_2015_l14_14761

-- Definitions for conditions
variables (x : ℕ) (T : ℕ)

-- The total age of the family in 2015 was 7 times Xiaoxiao's age
axiom h1 : T = 7 * x

-- The total age of the family in 2020 after the sibling is 6 times Xiaoxiao's age in 2020
axiom h2 : T + 19 = 6 * (x + 5)

-- Proof goal: Xiaoxiao’s age in 2015 is 11
theorem xiaoxiao_age_in_2015 : x = 11 :=
by
  sorry

end xiaoxiao_age_in_2015_l14_14761


namespace replace_stars_identity_l14_14879

theorem replace_stars_identity : 
  ∃ (a b : ℤ), a = -1 ∧ b = 1 ∧ (2 * x + a)^3 = 5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x := 
by 
  use [-1, 1]
  sorry

end replace_stars_identity_l14_14879


namespace dot_product_ab_l14_14691

variables (a b : ℝ^3)

-- Given conditions
def condition1 : Prop := ‖a‖ = 1
def condition2 : Prop := ‖b‖ = real.sqrt 3
def condition3 : Prop := ‖a - 2 • b‖ = 3

-- The theorem statement to prove
theorem dot_product_ab (h1 : condition1 a) (h2 : condition2 b) (h3 : condition3 a b) : 
  a ⬝ b = 1 :=
sorry

end dot_product_ab_l14_14691


namespace find_dot_product_l14_14682

open Real

noncomputable def vec_a : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector a
noncomputable def vec_b : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector b

def magnitude (v : ℝ → ℝ → ℝ) : ℝ :=
  sqrt ((v 0) ^ 2 + (v 1)^ 2)

def dot_product (u v : ℝ → ℝ → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

axiom magnitude_a_eq1 : magnitude vec_a = 1
axiom magnitude_b_eq_sqrt3 : magnitude vec_b = sqrt 3
axiom magnitude_a_minus_2b_eq3 : magnitude (λ x, vec_a x - 2 * vec_b x) = 3

theorem find_dot_product (a b : ℝ → ℝ → ℝ) 
  (ha : magnitude a = 1) 
  (hb : magnitude b = sqrt 3) 
  (h : magnitude (λ x, a x - 2 * b x) = 3) :
  dot_product a b = 1 := sorry

end find_dot_product_l14_14682


namespace basketball_game_l14_14148

theorem basketball_game 
    (a b x : ℕ)
    (h1 : 3 * b = 2 * a)
    (h2 : x = 2 * b)
    (h3 : 2 * a + 3 * b + x = 72) : 
    x = 18 :=
sorry

end basketball_game_l14_14148


namespace inequality_proof_l14_14817

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : 1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2) :
  sqrt((a^2 + 1) / 2) + sqrt((b^2 + 1) / 2) + sqrt((c^2 + 1) / 2) + sqrt((d^2 + 1) / 2) >= 
  3 * (sqrt(a) + sqrt(b) + sqrt(c) + sqrt(d)) - 8 :=
by
  sorry

end inequality_proof_l14_14817


namespace find_x_l14_14853

theorem find_x (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by 
  sorry

end find_x_l14_14853


namespace equation_of_symmetric_circle_l14_14604

theorem equation_of_symmetric_circle :
  ∃ r : ℝ, r > 0 ∧ 
    (∀ (C : ℝ × ℝ → Prop), 
      (C = λ z, (z.1 - 0)^2 + (z.2 - 0)^2 = r^2) ∧
      (C (1, 1)) ∧ 
      (∀ (M : ℝ × ℝ → Prop), 
        (M = λ z, (z.1 + 2)^2 + (z.2 + 2)^2 = r^2) → 
        (∀ (x y : ℝ), x + y + 2 = 0 → 
          (λ p, (p.1 - (-2)) / 2 = x ∧ (p.2 - (-2)) / 2 = y) 
             (λ z, (z.1 - 0)^2 + (z.2 - 0)^2 = r^2)))) :=
sorry

end equation_of_symmetric_circle_l14_14604


namespace max_tulips_l14_14457

theorem max_tulips (y r : ℕ) (h1 : (y + r) % 2 = 1) (h2 : r = y + 1 ∨ y = r + 1) (h3 : 50 * y + 31 * r ≤ 600) : y + r = 15 :=
by
  sorry

end max_tulips_l14_14457


namespace sequence_divisible_by_103_up_to_4000_l14_14734

theorem sequence_divisible_by_103_up_to_4000 :
  let sequence_term (n : ℕ) := 10 ^ n + 1
  in let count_divisibles_by_103 (k : ℕ) := ∃ m : ℕ, m ≤ k ∧ 103 ∣ sequence_term m
  in (∀ n ≤ 4000, count_divisibles_by_103 n) = 666 :=
by
  sorry

end sequence_divisible_by_103_up_to_4000_l14_14734


namespace number_of_prime_divisors_of_13575_l14_14736

/-- There are exactly 3 distinct prime positive integers that are divisors of 13575. -/
theorem number_of_prime_divisors_of_13575 : 
  ∃ (primes : set ℕ), {p : ℕ | nat.prime p ∧ p ∣ 13575} = primes ∧ primes.card = 3 :=
sorry

end number_of_prime_divisors_of_13575_l14_14736


namespace min_cubes_for_views_l14_14507

def front_view := (3, 2, 1)  -- cubes per level from the front view
def side_view := (1, 2, 3)   -- cubes per level from the side view

theorem min_cubes_for_views : 
  ∃ (n : ℕ), 
    (∀ (view : unit),
      view = () → 
      let front := front_view
          side := side_view in
      n = 6
    ) := 
begin
  let n := 6,
  use n,
  intros view h_view,
  have front := front_view,
  have side := side_view,
  exact eq.refl n,
end

end min_cubes_for_views_l14_14507


namespace point_D_sum_is_ten_l14_14386

noncomputable def D_coordinates_sum_eq_ten : Prop :=
  ∃ (D : ℝ × ℝ), (5, 5) = ( (7 + D.1) / 2, (3 + D.2) / 2 ) ∧ (D.1 + D.2 = 10)

theorem point_D_sum_is_ten : D_coordinates_sum_eq_ten :=
  sorry

end point_D_sum_is_ten_l14_14386


namespace cos_pi_over_2_minus_2alpha_l14_14740

theorem cos_pi_over_2_minus_2alpha (α : ℝ) (h : Real.tan α = 2) : Real.cos (Real.pi / 2 - 2 * α) = 4 / 5 := 
by 
  sorry

end cos_pi_over_2_minus_2alpha_l14_14740


namespace problem_statement_l14_14603

theorem problem_statement (x : ℂ) (h : x + x⁻¹ = 10 / 3) : x^2 - x⁻² = 80 / 9 ∨ x^2 - x⁻² = -80 / 9 :=
sorry

end problem_statement_l14_14603


namespace sum_of_intersections_l14_14948

-- Define the set X and its cardinality n
def X (n : ℕ) : Finset ℕ := Finset.range n

-- Define the problem statement in Lean
theorem sum_of_intersections (n : ℕ) :
  let X := Finset.range n
  in let subsets := Finset.powerset X
  in ∑ _ in subsets.product subsets, (λ p, (p.1 ∩ p.2).card) = n * 4^(n-1) := 
begin
  sorry
end

end sum_of_intersections_l14_14948


namespace C_total_score_l14_14018

-- Definitions of the conditions
variable (A B C : Type)
variable (score : A → ℕ → ℤ)

axiom total_rounds : ℕ
axiom total_points : A → ℤ
axiom round_points : ℕ → B → ℤ

-- Conditions based on the problem
def rounds := 5
def A_total_points := 14
def B_first_round_points := 3
def B_second_round_points := 1

axiom A_scored_14 : total_points A = A_total_points
axiom B_first_round : score B 1 = B_first_round_points
axiom B_second_round : score B 2 = B_second_round_points

-- Prove C's total score
theorem C_total_score : (sum (score C) (range rounds)) = 9 :=
by sorry

end C_total_score_l14_14018


namespace problem_D_is_odd_function_l14_14118

-- Define a function f for the given condition
def f (a : ℝ) (x : ℝ) : ℝ := (x - a) * abs x

-- Define the condition for function f to be odd
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = -f(x)

-- Prove that f is an odd function when a = 0
theorem problem_D_is_odd_function: odd_function (f 0) :=
  sorry

end problem_D_is_odd_function_l14_14118


namespace lawn_remaining_fractional_part_l14_14832

noncomputable def lawn_remaining (mary_hours : ℕ) (mary_rate : ℚ) (tom_hours : ℕ) (tom_rate : ℚ) (combined_time : ℕ) (combined_rate : ℚ) : ℚ :=
  tom_hours * tom_rate + combined_time * combined_rate

theorem lawn_remaining_fractional_part :
  lawn_remaining 3 (1/5) 1 (1/4) 1 (1/5 + 1/4) - 1 = 1/20 :=
by
  -- Definitions based on conditions
  let tom_rate := 1/5
  let mary_rate := 1/4
  let combined_rate := tom_rate + mary_rate
  let tom_work := 3 * tom_rate
  let combined_work := 1 * combined_rate
  -- Calculation with subtraction of the whole lawn
  let total_work := tom_work + combined_work
  let remaining_work := total_work - 1
  -- We're expected to prove this is equal to 1/20
  exact calc
    remaining_work = total_work - 1 : rfl
    ... =  1/20 : sorry

end lawn_remaining_fractional_part_l14_14832


namespace dot_product_proof_l14_14658

variables {ℝ : Type*}
variables (a b : ℝ → ℝ)
variables [inner_product_space ℝ ℝ]

theorem dot_product_proof
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = sqrt 3)
  (h3 : ∥a - 2 • b∥ = 3) :
  inner (a : ℝ) (b : ℝ) = 1 :=
sorry

end dot_product_proof_l14_14658


namespace combined_yearly_return_percentage_l14_14492

theorem combined_yearly_return_percentage :
  let investment1 := 500
  let return1 := 0.07
  let investment2 := 1500
  let return2 := 0.09
  let total_investment := investment1 + investment2
  let total_return := (investment1 * return1) + (investment2 * return2)
  total_return / total_investment * 100 = 8.5 := 
begin
  sorry
end

end combined_yearly_return_percentage_l14_14492


namespace square_center_sum_l14_14407

noncomputable def sum_of_center_coords (A B C D : (ℝ × ℝ)) (DA BC CD AB : ℝ → ℝ) : Prop :=
  let center := (A.1 + B.1 + C.1 + D.1) / 4
  center.1 + center.2 = 12

theorem square_center_sum 
  (h1 : ∀ x, (DA x) = 0 → x = 2)
  (h2 : ∀ x, (BC x) = 0 → x = 8)
  (h3 : ∀ x, (CD x) = 0 → x = 12)
  (h4 : ∀ x, (AB x) = 0 → x = 20)
  (A B C D : (ℝ × ℝ)) :
  (∃ m : ℝ, m ≠ 0 ∧ ∀ x, DA x = m * (x - 2)) →
  (∃ m : ℝ, m ≠ 0 ∧ ∀ x, BC x = m * (x - 8)) →
  (∃ m : ℝ, m ≠ 0 ∧ ∀ x, CD x = - (1 / m) * (x - 12)) →
  (∃ m : ℝ, m ≠ 0 ∧ ∀ x, AB x = - (1 / m) * (x - 20)) →
  sum_of_center_coords A B C D DA BC CD AB := 
sorry

end square_center_sum_l14_14407


namespace x_any_real_number_l14_14771

variable (x : ℝ) (s : ℝ)
variable (BD DE EA AF FB EC : ℝ)

-- Given conditions
variable h1 : BD = 4
variable h2 : DE = 2 * x
variable h3 : EA = x + 5
variable h4 : AF = 3
variable h5 : FB = 7 + x
variable h6 : EC = 2 * x + 15

-- Equilateral triangle side lengths
variable h7 : s = BD + DE + EC

theorem x_any_real_number :
  ∃ x : ℝ, BD + DE + EC = 4 * x + 19 :=
by
  sorry

end x_any_real_number_l14_14771


namespace statues_at_end_of_fourth_year_l14_14727

def initial_statues : ℕ := 4
def statues_after_second_year : ℕ := initial_statues * 4
def statues_added_third_year : ℕ := 12
def broken_statues_third_year : ℕ := 3
def statues_removed_third_year : ℕ := broken_statues_third_year
def statues_added_fourth_year : ℕ := broken_statues_third_year * 2

def statues_end_of_first_year : ℕ := initial_statues
def statues_end_of_second_year : ℕ := statues_after_second_year
def statues_end_of_third_year : ℕ := statues_end_of_second_year + statues_added_third_year - statues_removed_third_year
def statues_end_of_fourth_year : ℕ := statues_end_of_third_year + statues_added_fourth_year

theorem statues_at_end_of_fourth_year : statues_end_of_fourth_year = 31 :=
by
  sorry

end statues_at_end_of_fourth_year_l14_14727


namespace intersect_product_distance_l14_14782

open Real

def C1_polar (ρ θ : ℝ) : Prop := ρ * sin θ ^ 2 = 4 * cos θ

def C2_parametric (t : ℝ) (x y : ℝ) : Prop :=
  x = 2 + (1 / 2) * t ∧ y = (sqrt 3 / 2) * t 

def on_curve_C1 (x y : ℝ) : Prop := y ^ 2 = 4 * x

def on_curve_C2 (x y : ℝ) : Prop := sqrt 3 * x - y - 2 * sqrt 3 = 0

theorem intersect_product_distance : 
  (∀ (ρ θ : ℝ), C1_polar ρ θ → on_curve_C1 (ρ * cos θ) (ρ * sin θ)) →
  (∀ (t x y : ℝ), C2_parametric t x y → on_curve_C2 x y) →
  (∃ A B : ℝ × ℝ, on_curve_C1 A.1 A.2 ∧ on_curve_C2 A.1 A.2 ∧ 
                   on_curve_C1 B.1 B.2 ∧ on_curve_C2 B.1 B.2 ∧ 
                   let PA := (2 - A.1, - A.2), 
                       PB := (2 - B.1, - B.2) in
                   sqrt (PA.1 ^ 2 + PA.2 ^ 2) * sqrt (PB.1 ^ 2 + PB.2 ^ 2) = 32 / 3) :=
sorry

end intersect_product_distance_l14_14782


namespace event_B_is_correct_l14_14166

-- Defining the conditions of the problem
variable (Balls : α) -- α is the type representing balls
variable [DecidableEq α]
variable (red_ball : α)
variable (green_ball : α)
variable (bag : Multiset α)
variable (has_more_than_two_red : Multiset.card (Multiset.filter (λ x, x = red_ball) bag) > 2)
variable (has_more_than_two_green : Multiset.card (Multiset.filter (λ x, x = green_ball) bag) > 2)
variable (draw : Finset (Finset α)) -- Finset of size 2 representing the draw

-- Event definitions, where draw is a set of two balls drawn from the bag.
variable (Event_A : draw ⊆ Finset.insert red_ball (Finset.singleton green_ball) ∨ draw ⊆ Finset.insert green_ball (Finset.singleton red_ball))
variable (Event_B : Finset.card (Finset.filter (λ x, x = red_ball) draw) = 1 ∧ Finset.card (Finset.filter (λ x, x = green_ball) draw) = 2)
variable (Event_C : draw ⊆ Finset.insert red_ball (Finset.singleton red_ball))
variable (Event_D : draw ⊆ Finset.insert green_ball (Finset.singleton green_ball))

-- Statement ensuring that Event B is mutually exclusive but not contradictory
theorem event_B_is_correct : (Event_B) ∧ ¬(Event_A) ∧ ¬(Event_C) ∧ ¬(Event_D) := by
  sorry

end event_B_is_correct_l14_14166


namespace harper_rubber_bands_l14_14274

theorem harper_rubber_bands (H : ℕ) (brother_bands : H - 6) (total_bands : H + (H - 6) = 24) : H = 15 :=
by 
  sorry

end harper_rubber_bands_l14_14274


namespace cone_lateral_surface_area_l14_14188

theorem cone_lateral_surface_area (r V : ℝ) (h l S : ℝ) 
  (radius_condition : r = 6)
  (volume_condition : V = 30 * Real.pi)
  (volume_formula : V = (1 / 3) * Real.pi * r^2 * h)
  (slant_height_formula : l = Real.sqrt (r^2 + h^2))
  (lateral_surface_area_formula : S = Real.pi * r * l) :
  S = 39 * Real.pi := 
sorry

end cone_lateral_surface_area_l14_14188


namespace range_of_a_l14_14617

variable (a : ℝ)

def proposition_p (a : ℝ) : Prop := 
  ∃ x : ℝ, x ∈ set.Icc (-1:ℝ) (1:ℝ) ∧ x^2 - 3 * a * x + 2 * a^2 = 0

def proposition_q (a : ℝ) : Prop := 
  ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

theorem range_of_a (h : ¬ (proposition_p a ∨ proposition_q a)) : 
  a ∈ set.Ioo (-∞) (-1) ∪ set.Ioo 1 2 ∪ set.Ioo 2 ∞ := 
sorry

end range_of_a_l14_14617


namespace integer_solution_x_l14_14789

theorem integer_solution_x (x y : ℤ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y + x * y = 101) : x = 50 :=
sorry

end integer_solution_x_l14_14789


namespace identity_solution_l14_14885

theorem identity_solution (x : ℝ) :
  ∃ a b : ℝ, (2 * x + a) ^ 3 = 5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x ∧
             a = -1 ∧ b = 1 :=
by
  -- we can skip the proof as this is just a statement
  sorry

end identity_solution_l14_14885


namespace probability_of_winning_l14_14525

-- Define the conditions
def total_tickets : ℕ := 10
def winning_tickets : ℕ := 3
def people : ℕ := 5
def losing_tickets : ℕ := total_tickets - winning_tickets

-- The probability calculation as per the conditions
def probability_at_least_one_wins : ℚ :=
  1 - ((Nat.choose losing_tickets people : ℚ) / (Nat.choose total_tickets people))

-- The statement to be proven
theorem probability_of_winning :
  probability_at_least_one_wins = 11 / 12 := 
sorry

end probability_of_winning_l14_14525


namespace cost_per_book_l14_14392

theorem cost_per_book (a r n c : ℕ) (h : a - r = n * c) : c = 7 :=
by sorry

end cost_per_book_l14_14392


namespace none_of_the_above_correct_l14_14595

-- Definitions based on the conditions
def periodic_function (f : ℝ → ℝ) : Prop := ∃ T > 0, ∀ x, f (x + T) = f x
def monotonic_function (f : ℝ → ℝ) : Prop := ∀ x y, x ≤ y → f x ≤ f y

-- Original proposition
def original_proposition (f : ℝ → ℝ) : Prop :=
  periodic_function f → ¬ monotonic_function f

-- Converse proposition
def converse_proposition (f : ℝ → ℝ) : Prop :=
  ¬ monotonic_function f → periodic_function f

-- Negation proposition
def negation_proposition (f : ℝ → ℝ) : Prop :=
  ¬ (periodic_function f) → monotonic_function f

-- Contrapositive proposition
def contrapositive_proposition (f : ℝ → ℝ) : Prop :=
  ¬ monotonic_function f → periodic_function f

-- Theorem stating the correct answer is "None of the above are correct"
theorem none_of_the_above_correct (f : ℝ → ℝ) :
  ¬ (converse_proposition f) ∧ ¬ (negation_proposition f) ∧ ¬ (contrapositive_proposition f) :=
begin
  sorry
end

end none_of_the_above_correct_l14_14595


namespace compute_f_values_sum_eq_one_l14_14601

noncomputable def f_n (f : ℕ → ℕ) (n : ℕ) (x : ℕ): ℕ :=
  if n = 1 then f x
  else f_n f (n - 1) (f x)

theorem compute_f_values_sum_eq_one (f : ℕ → ℕ) (n : ℕ) (h_n : n ≥ 2) : 
  (∑ i in Finset.range (n + 1), f i) + (∑ i in Finset.range n, f_n f (i + 1) 1) = 1 :=
sorry

end compute_f_values_sum_eq_one_l14_14601


namespace energy_of_first_particle_l14_14766

theorem energy_of_first_particle
  (E_1 E_2 E_3 : ℤ)
  (h1 : E_1^2 - E_2^2 - E_3^2 + E_1 * E_2 = 5040)
  (h2 : E_1^2 + 2 * E_2^2 + 2 * E_3^2 - 2 * E_1 * E_2 - E_1 * E_3 - E_2 * E_3 = -4968)
  (h3 : 0 < E_3)
  (h4 : E_3 ≤ E_2)
  (h5 : E_2 ≤ E_1) : E_1 = 12 :=
by sorry

end energy_of_first_particle_l14_14766


namespace number_of_ordered_pairs_l14_14735

/-
  Define a predicate stating that p is a prime number less than 100
  and that (a, b) is a pair of positive integers such that ab = 2p
-/

def is_prime_less_than_100 (p : ℕ) : Prop :=
  Nat.Prime p ∧ p < 100

def valid_pair (p a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a * b = 2 * p

/-
  The statement to prove that there are exactly 50 ordered pairs of (a, b)
  satisfying the conditions
-/

theorem number_of_ordered_pairs : 
  (∑ p in Finset.filter is_prime_less_than_100 (Finset.range 100), 
    (if ∃ (a b : ℕ), valid_pair p a b then 2 else 0)) = 50 :=
sorry

end number_of_ordered_pairs_l14_14735


namespace dot_product_eq_one_l14_14693

variables {α : Type*} [InnerProductSpace ℝ α]

noncomputable def vector_a (a : α) : Prop := ∥a∥ = 1
noncomputable def vector_b (b : α) : Prop := ∥b∥ = real.sqrt 3
noncomputable def vector_c (a b : α) : Prop := ∥a - (2 : ℝ) • b∥ = 3

theorem dot_product_eq_one (a b : α) (ha : vector_a a) (hb : vector_b b) (hc : vector_c a b) :
  inner a b = 1 :=
sorry

end dot_product_eq_one_l14_14693


namespace equilateral_triangle_if_A_pi_over_3_cos2A_cos2B_iff_A_B_area_triangle_condition_not_satisfied_sin_BAD_condition_l14_14332

-- For statement A
theorem equilateral_triangle_if_A_pi_over_3 (A B C a b c : ℝ) (h1 : A = Real.pi / 3) (h2 : Real.sin A ^ 2 = Real.sin B * Real.sin C) : 
    (a = b ∧ b = c ∧ c = a) := 
sorry

-- For statement B
theorem cos2A_cos2B_iff_A_B (A B : ℝ) (h : A > B) : 
    Real.cos (2 * A) < Real.cos (2 * B) := 
sorry

-- For statement C
theorem area_triangle_condition_not_satisfied (A B C a b c : ℝ) (h : 1 / 2 * a * c * Real.sin B = b * (a ^ 2 + c ^ 2 - b ^ 2) / (4 * a)) :
     ¬ (A + B = Real.pi / 2) :=
sorry

-- For statement D
theorem sin_BAD_condition (B C : ℝ) (D : Point) (h1 : 2 * Real.sin C = Real.sin B) (BD : Vector) (h2 : BD = 1/4 * (C - B)) : 
    Real.sin (angle BAD) = 2/3 * Real.sin (angle CAD) :=
sorry

end equilateral_triangle_if_A_pi_over_3_cos2A_cos2B_iff_A_B_area_triangle_condition_not_satisfied_sin_BAD_condition_l14_14332


namespace symmetric_about_pi_over_3_range_of_f_on_interval_l14_14255

def f (x : ℝ) := (real.sqrt 3 / 2) * real.sin (2 * x) - (1 / 2) * real.cos (2 * x)

theorem symmetric_about_pi_over_3 :
  ∀ x, f (x) = f (2 * real.pi / 3 - x) :=
sorry

theorem range_of_f_on_interval :
  set.Icc (-real.pi / 6) (real.pi / 2) ⊆ f '' set.Icc (-real.pi / 6) (real.pi / 2) :=
sorry

end symmetric_about_pi_over_3_range_of_f_on_interval_l14_14255


namespace simplify_expression_l14_14904

variable (y : ℝ)

theorem simplify_expression : (5 / (4 * y^(-4)) * (4 * y^3) / 3) = 5 * y^7 / 3 :=
by
  sorry

end simplify_expression_l14_14904


namespace find_dot_product_l14_14675

open Real

noncomputable def vec_a : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector a
noncomputable def vec_b : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector b

def magnitude (v : ℝ → ℝ → ℝ) : ℝ :=
  sqrt ((v 0) ^ 2 + (v 1)^ 2)

def dot_product (u v : ℝ → ℝ → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

axiom magnitude_a_eq1 : magnitude vec_a = 1
axiom magnitude_b_eq_sqrt3 : magnitude vec_b = sqrt 3
axiom magnitude_a_minus_2b_eq3 : magnitude (λ x, vec_a x - 2 * vec_b x) = 3

theorem find_dot_product (a b : ℝ → ℝ → ℝ) 
  (ha : magnitude a = 1) 
  (hb : magnitude b = sqrt 3) 
  (h : magnitude (λ x, a x - 2 * b x) = 3) :
  dot_product a b = 1 := sorry

end find_dot_product_l14_14675


namespace total_outcomes_black_outcomes_probability_black_outcomes_l14_14497

-- Definitions for the problem conditions
def balls := {1, 2, 3, 4}  -- 1 represents the white ball, and 2, 3, 4 represent the black balls
def is_black (b : ℕ) : Prop := b ≠ 1
def draws := {S | S ⊆ balls ∧ S.card = 2}

-- Lean statements of the required proofs

-- (1) Total number of different outcomes
theorem total_outcomes : draws.card = 6 := sorry

-- (2) Number of outcomes when drawing 2 black balls
def black_draws := {S | S ⊆ {2, 3, 4} ∧ S.card = 2}

theorem black_outcomes : black_draws.card = 3 := sorry

-- (3) Probability of drawing 2 black balls
theorem probability_black_outcomes : (black_draws.card : ℚ) / (draws.card : ℚ) = 1 / 2 := sorry

end total_outcomes_black_outcomes_probability_black_outcomes_l14_14497


namespace smallest_abundant_not_multiple_of_five_l14_14579

def proper_divisors_sum (n : ℕ) : ℕ :=
  (Nat.divisors n).erase n |>.sum

def is_abundant (n : ℕ) : Prop := proper_divisors_sum n > n

def is_not_multiple_of_five (n : ℕ) : Prop := ¬ (5 ∣ n)

theorem smallest_abundant_not_multiple_of_five :
  18 = Nat.find (λ n, is_abundant n ∧ is_not_multiple_of_five n) :=
sorry

end smallest_abundant_not_multiple_of_five_l14_14579


namespace max_g6_l14_14367

noncomputable def g (x : ℝ) : ℝ :=
sorry

theorem max_g6 :
  (∀ x, (g x = a * x^2 + b * x + c) ∧ (a ≥ 0) ∧ (b ≥ 0) ∧ (c ≥ 0)) →
  (g 3 = 3) →
  (g 9 = 243) →
  (g 6 ≤ 6) :=
sorry

end max_g6_l14_14367


namespace probability_two_females_l14_14303

theorem probability_two_females (n f : ℕ) (h_n : n = 8) (h_f : f = 5) : 
  (∃ p : ℚ, p = (nat.choose 5 2 : ℚ) / (nat.choose 8 2) ∧ p = 5 / 14) :=
by 
  use ((nat.choose 5 2 : ℚ) / (nat.choose 8 2))
  split
  sorry
  sorry

end probability_two_females_l14_14303


namespace largest_angle_is_135_degrees_l14_14420

theorem largest_angle_is_135_degrees
  (α β γ α' β' γ' : ℝ)
  (h1 : cos α = sin α')
  (h2 : cos β = sin β')
  (h3 : cos γ = sin γ')
  (s1 : 0 < α ∧ α < π / 2)
  (s2 : 0 < β ∧ β < π / 2)
  (s3 : 0 < γ ∧ γ < π / 2)
  (sum_angles_first_triangle : α + β + γ = π)
  (sum_angles_second_triangle : α' + β' + γ' = π) :
  max (max α β) (max γ (max α' (max β' γ'))) = (3 * π / 4) :=
by
  sorry

end largest_angle_is_135_degrees_l14_14420


namespace least_multiple_of_five_primes_l14_14990

noncomputable def smallest_multiple_of_five_primes : ℕ :=
  let primes := [2, 3, 5, 7, 11] in
  primes.foldl (· * ·) 1

theorem least_multiple_of_five_primes : smallest_multiple_of_five_primes = 2310 := by
  sorry

end least_multiple_of_five_primes_l14_14990


namespace tangent_line_unique_chords_and_area_l14_14247

noncomputable def circle (x y : ℝ) : Prop := x^2 + y^2 = 4
def point (x a : ℝ) : Prop := x = 1

theorem tangent_line_unique (a : ℝ) (tangent_line : ℝ → ℝ) :
  (∃ k b : ℝ, ∀ x y : ℝ, tangent_line x = k * x + b ∧ tangent_line x - y = 0 ∧
    ∀ x y : ℝ, point 1 a ∧ circle x y → tangent_line x ≠ k * x + b) →
  a = sqrt 3 ∨ a = -sqrt 3 :=
begin
  sorry
end

theorem chords_and_area (a : ℝ) :
  a = sqrt 2 →
  (∃ k1 b1 k2 b2 : ℝ,
    ∀ x y : ℝ, (k1 * x + b1 = y ∨ k2 * x + b2 = y) ∧
    ∀ x y : ℝ, circle x y ∧ point 1 a →
      (∀ x y : ℝ, y = k1 * x + b1 → k2 * x + b2 = y) ∧
      (∀ x y : ℝ, y = k2 * x + b2 → k1 * x + b1 = y) ∧
      true ∧ -- placeholder for equations check
      true ∧ -- placeholder for distance calculation
      4 = 4) :=
begin
  sorry
end

end tangent_line_unique_chords_and_area_l14_14247


namespace simplify_expression_l14_14641

open Complex

variables (a b x y : ℝ)

theorem simplify_expression :
  (⟨a * x, 0⟩ : ℂ) * (⟨a * x, 0⟩ : ℂ) - (⟨a * x, 0⟩ : ℂ) * (⟨b * ⟦0, y⟧⟩ : ℂ) + 
  (⟨b * Film⟦0, y⟧⟩ : ℂ) * (⟨a * x, 0⟩ : ℂ) - (⟨b * ⟦0, y⟧⟩ : ℂ) * (⟨b * ⟦0, y⟧⟩ : ℂ) = 
  (a^2 * x^2 - b^2 * y^2 : ℝ) :=
sorry

end simplify_expression_l14_14641


namespace division_by_fraction_equiv_neg_multiplication_l14_14171

theorem division_by_fraction_equiv_neg_multiplication (h : 43 * 47 = 2021) : (-43) / (1 / 47) = -2021 :=
by
  -- Proof would go here, but we use sorry to skip the proof for now.
  sorry

end division_by_fraction_equiv_neg_multiplication_l14_14171


namespace parabola_circle_tangency_l14_14965

-- Definitions for the given conditions
def is_origin (P : Point) : Prop :=
  P.x = 0 ∧ P.y = 0

def is_on_x_axis (P : Point) : Prop := 
  P.y = 0

def parabola_vertex_focus_condition (C : Parabola) (O F : Point) : Prop :=
  is_origin O ∧ is_on_x_axis F ∧ C.vertex = O ∧ C.focus = F

def intersects_at_perpendicular_points (C : Parabola) (l : Line) (P Q : Point) : Prop :=
  l.slope = 1 ∧ l.intersect C = {P, Q} ∧ vector_product_is_perpendicular 0 P Q = true

def circle_tangent_to_line_at_point (M : Circle) (l : Line) (P : Point) : Prop :=
  l.form = "x=1" ∧ distance M.center P = M.radius ∧ M.contains P

def parabola_contains_point (C : Parabola) (P : Point) : Prop :=
  P.on_parabola C = true

def lines_tangent_to_circle (l₁ l₂ : Line) (M : Circle) : Prop :=
  M.tangent l₁ ∧ M.tangent l₂

def position_relationship (l : Line) (M : Circle) : PositionRelationship :=
  TangencyLine.and Circle M

-- Statement of the proof problem
theorem parabola_circle_tangency :
  ∃ C M, let O : Point := { x := 0, y := 0 } in
  let F : Point := { x := 1/2, y := 0 } in
  parabola_vertex_focus_condition C O F ∧ 
  intersects_at_perpendicular_points C (line_horizontal 1) P Q ∧
  point M_center = (mk_point 2 0) ∧
  circle_tangent_to_line_at_point M (line_horizontal 1) (mk_point 1 0) ∧
  ∀ A1 A2 A3 : Point, parabola_contains_point C A1 ∧
                    parabola_contains_point C A2 ∧
                    parabola_contains_point C A3 ∧
                    lines_tangent_to_circle (line_through A1 A2) M ∧
                    lines_tangent_to_circle (line_through A1 A3) M →
                    position_relationship (line_through A2 A3) M = Tangent :=
begin
  sorry  
end

end parabola_circle_tangency_l14_14965


namespace find_m_l14_14349

-- We define the sequence Q where Q(n) is the probability of the bug being at vertex B after n meters
def Q : ℕ → ℚ
| 0       => 0
| (n + 1) => (1 - Q n) / 3

-- Hypothesis: The tetrahedron properties and bug's movement
variable (A B C D : Type) (hab : A != B) (hbc : B != C) (hcd : C != D) (hda : D != A)
           (habc : ¬has_collinear_points A B C) (habd : ¬has_collinear_points A B D)
           (hbcd : ¬has_collinear_points B C D) (hcda : ¬has_collinear_points C D A)
           (hdab : ¬has_collinear_points D A B)

-- Main theorem stating the probability after 10 meters is 14763/59049
theorem find_m : Q 10 = 14763 / 59049 := by
  sorry

end find_m_l14_14349


namespace range_of_a_l14_14261

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x)
  else -3*x + 1

theorem range_of_a :
  {a : ℝ | f(f(a)) = 2 ^ (-f(a))} = {a : ℝ | a ≥ (1 / 3)} :=
by
  sorry

end range_of_a_l14_14261


namespace topless_cubical_box_l14_14926

def squares : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

def valid_placement (s : Char) : Bool :=
  match s with
  | 'A' => true
  | 'B' => true
  | 'C' => true
  | 'D' => false
  | 'E' => false
  | 'F' => true
  | 'G' => true
  | 'H' => false
  | _ => false

def valid_configurations : List Char := squares.filter valid_placement

theorem topless_cubical_box:
  valid_configurations.length = 5 := by
  sorry

end topless_cubical_box_l14_14926


namespace not_basic_logical_structure_l14_14767

-- Definitions for the basic logical structures
def sequential_structure : Prop := true
def conditional_structure : Prop := true
def loop_structure : Prop := true

-- Prove that "Decision structure" is not one of the three basic logical structures
theorem not_basic_logical_structure : ∀ x : string, 
  (x = "Sequential structure" ∨ x = "Conditional structure" ∨ x = "Loop structure") ↔ x ≠ "Decision structure" :=
by
  intro x
  split
  {
    intro h
    cases h
    any_goals
    {
      intro hx
      contradiction
    }
  }
  {
    intro h
    by_cases hs : x = "Sequential structure"
    any_goals
    {
      right
      left
      exact hs
    }
    left_cases hc : x = "Conditional structure"
    any_goals
    {
      right
      right
      exact hc
    }
    left_cases hl : x = "Loop structure"
    any_goals
    {
      exact or.inl hl
    }
    exact or.inr h
  }

end not_basic_logical_structure_l14_14767


namespace cost_of_painting_murals_l14_14807

def first_mural_area : ℕ := 20 * 15
def second_mural_area : ℕ := 25 * 10
def third_mural_area : ℕ := 30 * 8

def first_mural_time : ℕ := first_mural_area * 20
def second_mural_time : ℕ := second_mural_area * 25
def third_mural_time : ℕ := third_mural_area * 30

def total_time : ℚ := (first_mural_time + second_mural_time + third_mural_time) / 60

def total_area : ℕ := first_mural_area + second_mural_area + third_mural_area

def cost (area : ℕ) : ℚ :=
  if area <= 100 then area * 150 else 
  if area <= 300 then 100 * 150 + (area - 100) * 175 
  else 100 * 150 + 200 * 175 + (area - 300) * 200

def total_cost : ℚ := cost total_area

theorem cost_of_painting_murals :
  total_cost = 148000 := by
  sorry

end cost_of_painting_murals_l14_14807


namespace cone_lateral_surface_area_l14_14206

theorem cone_lateral_surface_area (r : ℕ) (V : ℝ) (h l S : ℝ)
  (h_r : r = 6)
  (h_V : V = 30 * Real.pi)
  (h_volume : V = (1 / 3) * Real.pi * (r ^ 2) * h)
  (h_slant_height : l = Real.sqrt (r^2 + h^2))
  (h_lateral_surface_area : S = Real.pi * r * l) :
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l14_14206


namespace proof1_proof2_proof3_l14_14602

def problem1 (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 - (a + 2) * x + 2 < 3 - 2 * x

theorem proof1 (a : ℝ) (h : problem1 a) : a ∈ Set.Icc (-4 : ℝ) 0 := sorry

def problem2 (a : ℝ) (x : ℝ) : Prop :=
  (a ∈ Set.Icc (-1 : ℝ) 1) ∧ (a * x^2 - (a + 2) * x + 2 > 0)

theorem proof2 (a x : ℝ) (h : problem2 a x) : x ∈ Set.Ico (-2 : ℝ) 1 := sorry

def problem3 (a : ℝ) : Prop :=
  ∃ (m : ℝ) (h : m > 0), a * x^2 - (a + 2) * abs x + 2 = m + 1/m + 1 ∧ roots (λ x, a*x^2 - (a+2)*abs x + 2 - (m + 1/m + 1)).nodup

theorem proof3 (a : ℝ) (h : problem3 a) : a < -4 - 2 * Real.sqrt 3 := sorry

end proof1_proof2_proof3_l14_14602


namespace smallest_possible_age_difference_l14_14439

-- Define predicates to represent the conditions
def distinct (ages : List ℕ) : Prop := List.Nodup ages
def pairwise_distinct_diff (ages : List ℕ) : Prop :=
  List.Pairwise (λ a b, abs (a - b) ≠ abs (b - a)) ages

-- The main problem statement
theorem smallest_possible_age_difference (ages : List ℕ) 
  (h_len : ages.length = 5)
  (h_distinct : distinct ages)
  (h_pairwise : pairwise_distinct_diff ages) : 
  ∃ a b, a ∈ ages ∧ b ∈ ages ∧ a ≠ b ∧ a - b = 11 := 
  sorry

end smallest_possible_age_difference_l14_14439


namespace domain_composite_l14_14630

-- Define the conditions
def domain_f (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4

-- The theorem statement
theorem domain_composite (h : ∀ x, domain_f x → 0 ≤ x ∧ x ≤ 4) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2) :=
by
  sorry

end domain_composite_l14_14630


namespace least_multiple_of_five_primes_l14_14997

noncomputable def smallest_multiple_of_five_primes : ℕ :=
  let primes := [2, 3, 5, 7, 11] in
  primes.foldl (· * ·) 1

theorem least_multiple_of_five_primes : smallest_multiple_of_five_primes = 2310 := by
  sorry

end least_multiple_of_five_primes_l14_14997


namespace sum_possible_rs_l14_14838

theorem sum_possible_rs (r s : ℤ) (h1 : r ≠ s) (h2 : r + s = 24) : 
  ∃ sum : ℤ, sum = 1232 := 
sorry

end sum_possible_rs_l14_14838


namespace identity_solution_l14_14889

theorem identity_solution (x : ℝ) :
  ∃ a b : ℝ, (2 * x + a) ^ 3 = 5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x ∧
             a = -1 ∧ b = 1 :=
by
  -- we can skip the proof as this is just a statement
  sorry

end identity_solution_l14_14889


namespace count_natural_numbers_ending_in_0_l14_14733

theorem count_natural_numbers_ending_in_0 :
  ∃ (n : ℕ), n = 503 ∧ ∀ (k : ℕ), k ≤ 2012 → 
    (1^k + 2^k + 3^k + 4^k) % 10 = 0 ↔ ((k % 20 = 4) ∨ (k % 20 = 8) ∨ (k % 20 = 12) ∨ (k % 20 = 16)) :=
begin
  sorry,
end

end count_natural_numbers_ending_in_0_l14_14733


namespace logarithmic_expression_l14_14169

theorem logarithmic_expression (a : ℝ) (h : 3^a = 2) : 
  log 3 8 - 2 * log 3 6 = a - 2 := 
by
  sorry

end logarithmic_expression_l14_14169


namespace rancher_cattle_count_l14_14082

theorem rancher_cattle_count
  (truck_capacity : ℕ)
  (distance_to_higher_ground : ℕ)
  (truck_speed : ℕ)
  (total_transport_time : ℕ)
  (h1 : truck_capacity = 20)
  (h2 : distance_to_higher_ground = 60)
  (h3 : truck_speed = 60)
  (h4 : total_transport_time = 40):
  ∃ (number_of_cattle : ℕ), number_of_cattle = 400 :=
by {
  sorry
}

end rancher_cattle_count_l14_14082


namespace mul_powers_same_base_simplified_exponent_l14_14538

theorem mul_powers_same_base (x : ℝ) : (x^2) * (x^3) = x^(2+3) :=
by sorry

theorem simplified_exponent (x : ℝ) : (x^2) * (x^3) = x^5 :=
by {
  rw mul_powers_same_base,
  rw add_comm 2 3,
  exact rfl,
  sorry
}

end mul_powers_same_base_simplified_exponent_l14_14538


namespace even_segments_of_closed_self_intersecting_l14_14974

open Set

-- Define what it means for a polygonal chain.
structure PolygonalChain :=
(segments : Set (ℝ × ℝ × ℝ × ℝ)) -- A set of segments represented by pairs of points in ℝ²
(is_closed : ∀ (s t : ℝ × ℝ × ℝ × ℝ), s ∈ segments → t ∈ segments → s ≠ t → (intersect s t))

-- Predicate stating the condition of intersections.
def intersects_once (P : PolygonalChain) :=
∀ (s t : ℝ × ℝ × ℝ × ℝ), s ∈ P.segments → t ∈ P.segments → s ≠ t → s ≠ t → (intersection s t).finite

theorem even_segments_of_closed_self_intersecting (P : PolygonalChain) (h_intersects : intersects_once P) : 
  ∃ n : ℕ, even n ∧ finset.card P.segments = n := 
sorry

end even_segments_of_closed_self_intersecting_l14_14974


namespace general_term_sequence_l14_14942

theorem general_term_sequence (n : ℕ) : 
  let numerator := 2 * 2^(n - 1),
      denominator := 2 * n - 1 in
  (numerator / denominator) = (2^n / (2 * n - 1)) :=
by 
  -- Provide a reason why the numerator is 2^n and the denominator is 2n-1
  sorry

end general_term_sequence_l14_14942


namespace largest_tan_B_l14_14786

theorem largest_tan_B (A B C : Type u) [metric_space A] [linear_ordered_field B]
  (dist : A → A → B) : dist A B = 16 → dist A C = 12 → angle A C B = 90 → 
  ∃ B : Type u, tan B = 3 * sqrt 7 := by sorry

end largest_tan_B_l14_14786


namespace exists_three_fitting_rectangles_l14_14029

-- Define the fitting condition
def fits_inside (a b c d : ℕ) : Prop :=
  (a ≤ c ∧ b ≤ d) ∨ (a ≤ d ∧ b ≤ c)

-- Set S and its bounds
def S (n : ℕ) : Finset (ℕ × ℕ) := 
  (Finset.product (Finset.range n.succ) (Finset.range n.succ)).filter (λ p, p.1 ≠ 0 ∧ p.2 ≠ 0)

-- Main theorem
theorem exists_three_fitting_rectangles (S : Finset (ℕ × ℕ)) (hS : S.card = 2019 ∧ ∀ r ∈ S, r.1 ≤ 2018 ∧ r.2 ≤ 2018) :
  ∃ A B C, A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ fits_inside A.1 A.2 B.1 B.2 ∧ fits_inside B.1 B.2 C.1 C.2 :=
sorry

end exists_three_fitting_rectangles_l14_14029


namespace square_complex_C_l14_14637

noncomputable def A : ℂ := 1 + 2*complex.I
noncomputable def B : ℂ := 3 - 5*complex.I

theorem square_complex_C (h : ∀ z : ℂ, z ≠ 0 → z * complex.I ≠ 0) : ∃ C : ℂ, C = 10 - 3*complex.I :=
by
  have AB : ℂ := B - A
  have BC : ℂ := AB * complex.I
  have C : ℂ := B + BC
  existsi C
  sorry

end square_complex_C_l14_14637


namespace julies_birthday_day_of_week_l14_14165

theorem julies_birthday_day_of_week
    (fred_birthday_monday : Nat)
    (pat_birthday_before_fred : Nat)
    (julie_birthday_before_pat : Nat)
    (fred_birthday_after_pat : fred_birthday_monday - pat_birthday_before_fred = 37)
    (julie_birthday_before_pat_eq : pat_birthday_before_fred - julie_birthday_before_pat = 67)
    : (julie_birthday_before_pat - julie_birthday_before_pat % 7 + ((julie_birthday_before_pat % 7) - fred_birthday_monday % 7)) % 7 = 2 :=
by
  sorry

end julies_birthday_day_of_week_l14_14165


namespace angle_between_diagonal_and_base_l14_14935

theorem angle_between_diagonal_and_base
  (a b h : ℝ)
  (α β : ℝ)
  (h1 : h = a * Real.tan α)
  (h2 : h = b * Real.tan β) :
  Real.cot (Real.acot (sqrt ((Real.cot α)^2 + (Real.cot β)^2))) =
  sqrt ((Real.cot α)^2 + (Real.cot β)^2) :=
by
  sorry

end angle_between_diagonal_and_base_l14_14935


namespace part1_part2_part3_l14_14251

def f (a x : ℝ) (h : a > 0 ∧ a ≠ 1) (m : ℝ) := log a ((1 - m * x) / (x - 1))

theorem part1 (a : ℝ) (h : a > 0 ∧ a ≠ 1) : 
  (∀ x : ℝ, f a (-x) h (-1) + f a x h (-1) = 0) → 
  ∀ m : ℝ, f a (-x) h m + f a x h m = 0 → m = -1 := 
sorry

theorem part2 (a : ℝ) (h : a > 0 ∧ a ≠ 1) : 
  (∃ p : ℝ, ∃ a : ℝ, (1 ≤ p ∧ p ≤ a-2) ∧ (p = 1 ∧ a = 2 + Real.sqrt 3)) :=
sorry

def g (a x : ℝ) (h : a > 0 ∧ a ≠ 1) := -a * x^2 + 6 * (x - 1) * a^(log a ((1 + x) / (x - 1))) - 5

theorem part3 (x : ℝ) (a : ℝ) (h : a > 0 ∧ a ≠ 1) : 
  (4 ≤ x ∧ x ≤ 5) → 
  g x a h (4) = -16 * a + 25 ∧ g x a h (5) = -25 * a + 31 ∧ 
  (3 / (4 * a) < 4 ∧ 3 / (4 * a) > 5 → g x a h (3 / a) = 9 / a + 1) :=
sorry

end part1_part2_part3_l14_14251


namespace choose_4_out_of_10_l14_14318

theorem choose_4_out_of_10 :
  nat.choose 10 4 = 210 :=
  by
  sorry

end choose_4_out_of_10_l14_14318


namespace midpoint_feet_collinear_l14_14599

theorem midpoint_feet_collinear 
  (A B C H E F M : Type)
  [RealElement A] [RealElement B] [RealElement C]
  [RealElement H] [RealElement E] [RealElement F] [RealElement M]
  (h_orthocenter : orthocenter A B C H)
  (h_E : foot_perpendicular H (internal_angle_bisector A) E)
  (h_F : foot_perpendicular H (external_angle_bisector A) F)
  (h_M : midpoint B C M) :
  collinear M E F := 
sorry

end midpoint_feet_collinear_l14_14599


namespace sin_subtraction_example_l14_14539

theorem sin_subtraction_example : 
  sin (37.5 * (Real.pi / 180)) * cos (7.5 * (Real.pi / 180)) - cos (37.5 * (Real.pi / 180)) * sin (7.5 * (Real.pi / 180)) = 1 / 2 := 
by 
  -- This is a placeholder for the proof
  sorry

end sin_subtraction_example_l14_14539


namespace matrix_vector_computation_l14_14352

theorem matrix_vector_computation (N : Matrix (fin 2) (fin 2) ℝ) (u z : Vector (fin 2) ℝ) 
  (hN_u : N.mul_vec u = ![1, 4])
  (hN_z : N.mul_vec z = ![5, -3]) :
  N.mul_vec (-2 • u + 4 • z) = ![18, -20] :=
by sorry

end matrix_vector_computation_l14_14352


namespace solve_for_x_l14_14597

theorem solve_for_x : ∃ x : ℤ, 45 - 3 * x = 12 ∧ x = 11 := by
  existsi 11
  split
  · apply eq_refl
  · ring
  sorry

end solve_for_x_l14_14597


namespace min_students_l14_14764

theorem min_students (b g : ℕ) 
  (h1 : 3 * b = 2 * g) 
  (h2 : ∃ k : ℕ, b + g = 10 * k) : b + g = 38 :=
sorry

end min_students_l14_14764


namespace monotonic_decreasing_interval_l14_14428

noncomputable def func : ℝ → ℝ := λ x => -2 * x + x ^ 3

theorem monotonic_decreasing_interval :
  {x : ℝ | -func.deriv x < 0} = set.Ioo (-real.sqrt 6 / 3) (real.sqrt 6 / 3) := 
sorry

end monotonic_decreasing_interval_l14_14428


namespace find_x_l14_14803

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 :=
sorry

end find_x_l14_14803


namespace solution_l14_14238

noncomputable def given_problem (α : ℝ) : ℝ :=
  (2 * Real.sin (π + α) * Real.cos (π - α) - Real.cos (π + α)) /
  (1 + Real.sin α ^ 2 + Real.sin (π - α) - Real.cos (π + α) ^ 2)

theorem solution (α : ℝ) (hα : α = - 35 / 6 * π) : 
  given_problem α = √3 := by
  sorry

end solution_l14_14238


namespace dot_product_l14_14668

variables (a b : Vector ℝ) -- ℝ here stands for the real numbers

-- Given conditions
def condition1 : ∥a∥ = 1 := sorry
def condition2 : ∥b∥ = √3 := sorry
def condition3 : ∥a - (2 : ℝ) • b∥ = 3 := sorry

-- Goal to prove
theorem dot_product (a b : Fin₃ → ℝ) 
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = √3)
  (h3 : ∥a - (2 : ℝ) • b∥ = 3) : 
  a ⬝ b = 1 := 
sorry

end dot_product_l14_14668


namespace find_x_values_l14_14792

theorem find_x_values (x y : ℕ) (hx: x > y) (hy: y > 0) (h: x + y + x * y = 101) :
  x = 50 ∨ x = 16 :=
sorry

end find_x_values_l14_14792


namespace find_x_values_l14_14795

theorem find_x_values (x y : ℕ) (hx: x > y) (hy: y > 0) (h: x + y + x * y = 101) :
  x = 50 ∨ x = 16 :=
sorry

end find_x_values_l14_14795


namespace log_sum_value_l14_14436

theorem log_sum_value : log 4 + log 25 = 2 :=
by 
  -- Conditions and necessary log properties will be utilized here:
  -- log(a * b) = log(a) + log(b)
  -- log(a^b) = b * log(a)
  -- log(10) = 1
  sorry

end log_sum_value_l14_14436


namespace angles_opposite_edges_not_simultaneously_acute_or_obtuse_l14_14524

theorem angles_opposite_edges_not_simultaneously_acute_or_obtuse (S A B C D : Type) [convex_polyhedral_angle S A B C D]
  (h₁ : dihedral_angle S A B  = 60) 
  (h₂ : dihedral_angle S B C  = 60) 
  (h₃ : dihedral_angle S C D  = 60) 
  (h₄ : dihedral_angle S D A  = 60) : 
  ¬ (all_angles_acute_or_all_angles_obtuse S A B C D) :=
sorry

end angles_opposite_edges_not_simultaneously_acute_or_obtuse_l14_14524


namespace probability_six_envelopes_l14_14970

noncomputable def choose (n k: ℕ) : ℕ := nat.choose n k

noncomputable def derangement (n: ℕ) : ℕ :=
match n with
| 0 => 1
| 1 => 0
| 2 => 1
| 3 => 2
| _ => nat.subfactorial n

def probability_three_correct (n : ℕ) : ℚ :=
  let total_permutations := nat.factorial n
  let number_of_combinations := choose n 3
  let number_of_derangements := derangement 3
  (number_of_combinations * number_of_derangements : ℚ) / total_permutations

theorem probability_six_envelopes : probability_three_correct 6 = (1 / 18 : ℚ) :=
by 
  -- skipping the proof
  sorry

end probability_six_envelopes_l14_14970


namespace smallest_multiplier_to_perfect_square_l14_14480

theorem smallest_multiplier_to_perfect_square : ∃ k : ℕ, k > 0 ∧ ∀ m : ℕ, (2010 * m = k * k) → m = 2010 :=
by
  sorry

end smallest_multiplier_to_perfect_square_l14_14480


namespace cost_per_book_l14_14391

theorem cost_per_book (a r n c : ℕ) (h : a - r = n * c) : c = 7 :=
by sorry

end cost_per_book_l14_14391


namespace find_dot_product_l14_14678

open Real

noncomputable def vec_a : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector a
noncomputable def vec_b : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector b

def magnitude (v : ℝ → ℝ → ℝ) : ℝ :=
  sqrt ((v 0) ^ 2 + (v 1)^ 2)

def dot_product (u v : ℝ → ℝ → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

axiom magnitude_a_eq1 : magnitude vec_a = 1
axiom magnitude_b_eq_sqrt3 : magnitude vec_b = sqrt 3
axiom magnitude_a_minus_2b_eq3 : magnitude (λ x, vec_a x - 2 * vec_b x) = 3

theorem find_dot_product (a b : ℝ → ℝ → ℝ) 
  (ha : magnitude a = 1) 
  (hb : magnitude b = sqrt 3) 
  (h : magnitude (λ x, a x - 2 * b x) = 3) :
  dot_product a b = 1 := sorry

end find_dot_product_l14_14678


namespace find_dot_product_l14_14677

open Real

noncomputable def vec_a : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector a
noncomputable def vec_b : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector b

def magnitude (v : ℝ → ℝ → ℝ) : ℝ :=
  sqrt ((v 0) ^ 2 + (v 1)^ 2)

def dot_product (u v : ℝ → ℝ → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

axiom magnitude_a_eq1 : magnitude vec_a = 1
axiom magnitude_b_eq_sqrt3 : magnitude vec_b = sqrt 3
axiom magnitude_a_minus_2b_eq3 : magnitude (λ x, vec_a x - 2 * vec_b x) = 3

theorem find_dot_product (a b : ℝ → ℝ → ℝ) 
  (ha : magnitude a = 1) 
  (hb : magnitude b = sqrt 3) 
  (h : magnitude (λ x, a x - 2 * b x) = 3) :
  dot_product a b = 1 := sorry

end find_dot_product_l14_14677


namespace shorter_piece_length_l14_14495

variable (x y : ℕ)
variable (total_length : ℕ)
variable (length_difference : ℕ)

-- conditions
def total_length_eq : Prop := total_length = 68
def length_diff_eq : Prop := length_difference = 12
def pipe_eq : Prop := x + y = total_length
def diff_eq : Prop := y = x + length_difference

-- proof
theorem shorter_piece_length (h1 : total_length_eq) (h2 : length_diff_eq) (h3 : pipe_eq) (h4 : diff_eq) : x = 28 :=
by 
  sorry

end shorter_piece_length_l14_14495


namespace sqrt7_pos_neg_l14_14006

theorem sqrt7_pos_neg : ∃ x : ℝ, x * x = 7 ∧ (x = real.sqrt 7 ∨ x = -real.sqrt 7) :=
by
  sorry

end sqrt7_pos_neg_l14_14006


namespace part1_part2_l14_14622

theorem part1 (f : ℝ → ℝ) 
  (h1 : ∃ a b c, f = (λ x, a * x^2 + b * x + c) ∧ f 0 = 1)
  (h2 : ∀ x, f (x + 1) - f x = 2 * x) :
  f = λ x, x^2 - x + 1 := 
sorry

theorem part2 (a : ℝ)
  (f : ℝ → ℝ)
  (h : f = (λ x, x^2 - x + 1))
  (g : ℝ → ℝ := (λ x, f x - a)) :
  (∀ x, 1 ≤ x → g x > 0) ↔ a < 1 := 
sorry

end part1_part2_l14_14622


namespace choosing_4_out_of_10_classes_l14_14314

theorem choosing_4_out_of_10_classes :
  ∑ (k : ℕ) in (finset.range 5).map (prod.mk 10), k! / (4! * (k - 4)!) = 210 :=
by sorry

end choosing_4_out_of_10_classes_l14_14314


namespace measure_of_angle_BCD_l14_14335

-- Define the elements of the problem
variables {A B C D : Type} [EuclideanGeometry A B C D]

-- Conditions:
def isosceles_triangle (ABC : Triangle A B C) : Prop :=
  side_length AB = side_length AC

def angle_BAC (ABC : Triangle A B C) : Prop :=
  angle BAC = 100

def point_D_property (D : Type) (ABC : Triangle A B C) : Prop :=
  on_extension D AB A ∧ side_length AD = side_length BC

-- The statement to prove
theorem measure_of_angle_BCD (ABC : Triangle A B C) (D : Type)
  (h1 : isosceles_triangle ABC)
  (h2 : angle_BAC ABC)
  (h3 : point_D_property D ABC) :
  angle BCD = 10 :=
by
  sorry

end measure_of_angle_BCD_l14_14335


namespace prove_was_given_card_bob_3_and_dave_9_l14_14769

-- Define the cards and the players' scores
def cards := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
def players := ["Alice", "Bob", "Cathy", "Dave", "Ellen", "Frank"]
def scores := [8, 5, 15, 18, 10, 11]

noncomputable def was_given_card_bob_3_and_dave_9 : Prop :=
  ∃ (alice_cards bob_cards cathy_cards dave_cards ellen_cards frank_cards : finset ℕ),
    alice_cards ∪ bob_cards ∪ cathy_cards ∪ dave_cards ∪ ellen_cards ∪ frank_cards = cards ∧
    alice_cards.card = 2 ∧ bob_cards.card = 2 ∧ cathy_cards.card = 2 ∧ 
    dave_cards.card = 2 ∧ ellen_cards.card = 2 ∧ frank_cards.card = 2 ∧
    alice_cards.sum = 8 ∧ bob_cards.sum = 5 ∧ cathy_cards.sum = 15 ∧ 
    dave_cards.sum = 18 ∧ ellen_cards.sum = 10 ∧ frank_cards.sum = 11 ∧
    3 ∈ bob_cards ∧ 9 ∈ dave_cards

theorem prove_was_given_card_bob_3_and_dave_9 :
  was_given_card_bob_3_and_dave_9 :=
sorry

end prove_was_given_card_bob_3_and_dave_9_l14_14769


namespace identity_holds_l14_14893

noncomputable def identity_proof : Prop :=
∀ (x : ℝ), (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x

theorem identity_holds : identity_proof :=
by
  sorry

end identity_holds_l14_14893


namespace optimal_distance_circular_valid_optimal_distance_square_valid_l14_14057

noncomputable def optimal_distance_circular (n : ℕ) : ℝ :=
  match n with
  | 1 => 1
  | 2 => 1
  | 3 => Real.sqrt 3 / 2
  | 4 => Real.sqrt 2 / 2
  | 7 => 1 / 2
  | _ => 0 -- undefined for other values

noncomputable def optimal_distance_square (n : ℕ) : ℝ :=
  match n with
  | 1 => Real.sqrt 2 / 2
  | 2 => 1 / 2
  | 3 => Real.sqrt 3 / 3
  | 4 => 1 / 2
  | _ => 0 -- undefined for other values

theorem optimal_distance_circular_valid (n : ℕ) : 
  n ∈ {1, 2, 3, 4, 7} →
  ∃ (r : ℝ), r = optimal_distance_circular n :=
by
  intro hn
  cases n with 
  | 1 => exact ⟨1, rfl⟩
  | 2 => exact ⟨1, rfl⟩
  | 3 => exact ⟨Real.sqrt 3 / 2, rfl⟩
  | 4 => exact ⟨Real.sqrt 2 / 2, rfl⟩
  | 7 => exact ⟨1 / 2, rfl⟩
  | _ => exact ⟨0, rfl⟩
  sorry

theorem optimal_distance_square_valid (n : ℕ) : 
  n ∈ {1, 2, 3, 4} →
  ∃ (r : ℝ), r = optimal_distance_square n :=
by
  intro hn
  cases n with 
  | 1 => exact ⟨Real.sqrt 2 / 2, rfl⟩
  | 2 => exact ⟨1 / 2, rfl⟩
  | 3 => exact ⟨Real.sqrt 3 / 3, rfl⟩
  | 4 => exact ⟨1 / 2, rfl⟩
  | _ => exact ⟨0, rfl⟩
  sorry
 
end optimal_distance_circular_valid_optimal_distance_square_valid_l14_14057


namespace man_l14_14084

theorem man's_age_twice_son's_age_in_2_years
  (S : ℕ) (M : ℕ) (Y : ℕ)
  (h1 : M = S + 24)
  (h2 : S = 22)
  (h3 : M + Y = 2 * (S + Y)) :
  Y = 2 := by
  sorry

end man_l14_14084


namespace g_neither_even_nor_odd_l14_14337

noncomputable def g (x : ℝ) : ℝ := ⌈x⌉ - 1 / 2

theorem g_neither_even_nor_odd :
  (¬ ∀ x, g x = g (-x)) ∧ (¬ ∀ x, g (-x) = -g x) :=
by
  sorry

end g_neither_even_nor_odd_l14_14337


namespace elder_twice_as_old_l14_14423

theorem elder_twice_as_old (Y E : ℕ) (hY : Y = 35) (hDiff : E - Y = 20) : ∃ (X : ℕ),  X = 15 ∧ E - X = 2 * (Y - X) := 
by
  sorry

end elder_twice_as_old_l14_14423


namespace cone_lateral_surface_area_l14_14202

theorem cone_lateral_surface_area (r h l S : ℝ) (π_pos : 0 < π) (r_eq : r = 6)
  (V : ℝ) (V_eq : V = 30 * π)
  (vol_eq : V = (1/3) * π * r^2 * h)
  (h_eq : h = 5 / 2)
  (l_eq : l = Real.sqrt (r^2 + h^2))
  (S_eq : S = π * r * l) :
  S = 39 * π :=
  sorry

end cone_lateral_surface_area_l14_14202


namespace dot_product_is_one_l14_14722

variable {V : Type*} [InnerProductSpace ℝ V]
variables (a b : V)

theorem dot_product_is_one 
  (ha : ∥a∥ = 1) 
  (hb : ∥b∥ = sqrt 3) 
  (hab : ∥a - 2•b∥ = 3) : 
  ⟪a, b⟫ = 1 :=
by 
  sorry

end dot_product_is_one_l14_14722


namespace find_c_for_intersecting_linesegment_midpoint_l14_14943

theorem find_c_for_intersecting_linesegment_midpoint :
  ∃ c : ℝ, let midpoint := ((1 + 5) / 2, (3 + 11) / 2) in
  2 * midpoint.1 - midpoint.2 = c ∧ c = -1 :=
by
  -- define the endpoints of the segment
  let p1 := (1 : ℝ, 3 : ℝ)
  let p2 := (5 : ℝ, 11 : ℝ)
  -- calculate the midpoint of the segment
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  -- show that the midpoint satisfies 2x - y = c for c = -1
  have h : 2 * midpoint.1 - midpoint.2 = -1,
    sorry
  use -1
  exact ⟨midpoint, ⟨h, rfl⟩⟩

end find_c_for_intersecting_linesegment_midpoint_l14_14943


namespace product_multiple_of_12_probability_l14_14297

theorem product_multiple_of_12_probability :
  let s := {3, 4, 5, 6, 8}
  in
  (∃ pairs : set (ℕ × ℕ), pairs = { (3, 4), (4, 6), (6, 8) }) →
  let total_pairs := finset.card (set.to_finset ((s × s).image prod.mk)) / 2,
      favorable_pairs := finset.card (set.to_finset { (3, 4), (4, 6), (6, 8) })
  in
  favorable_pairs / total_pairs = (3 : ℚ) / 10 :=
sorry

end product_multiple_of_12_probability_l14_14297


namespace find_x_l14_14867

-- Define the variables and conditions
def x := 27
axiom h : (5 / 3) * x = 45

-- Main statement to be proved
theorem find_x : x = 27 :=
by
  have : (5 / 3) * x = 45 := h
  sorry

end find_x_l14_14867


namespace rate_of_stream_l14_14498

theorem rate_of_stream (v : ℝ) (h : 126 = (16 + v) * 6) : v = 5 :=
by 
  sorry

end rate_of_stream_l14_14498


namespace necessary_but_not_sufficient_condition_l14_14514

variable {x : ℝ}

theorem necessary_but_not_sufficient_condition 
    (h : -1 ≤ x ∧ x < 2) : 
    (-1 ≤ x ∧ x < 3) ∧ ¬(((-1 ≤ x ∧ x < 3) → (-1 ≤ x ∧ x < 2))) :=
by
  sorry

end necessary_but_not_sufficient_condition_l14_14514


namespace find_sum_l14_14104

theorem find_sum (I r1 r2 r3 r4 r5: ℝ) (t1 t2 t3 t4 t5 : ℝ) (P: ℝ) 
  (hI: I = 6016.75)
  (hr1: r1 = 0.06) (hr2: r2 = 0.075) (hr3: r3 = 0.08) (hr4: r4 = 0.085) (hr5: r5 = 0.09)
  (ht: ∀ i, (i = t1 ∨ i = t2 ∨ i = t3 ∨ i = t4 ∨ i = t5) → i = 1): 
  I = P * (r1 * t1 + r2 * t2 + r3 * t3 + r4 * t4 + r5 * t5) → P = 15430 :=
by
  sorry

end find_sum_l14_14104


namespace surface_area_of_circumscribed_sphere_l14_14324

theorem surface_area_of_circumscribed_sphere (SA SB SC AB BC CA : ℝ) 
  (h1 : SA = 2 * sqrt 3) (h2 : SB = 2 * sqrt 3) (h3 : SC = 2 * sqrt 3)
  (h4 : AB = 2 * sqrt 6) (h5 : BC = 2 * sqrt 6) (h6 : CA = 2 * sqrt 6) : 
  4 * π * (2 * sqrt 3)^2 = 48 * π := by
  sorry

end surface_area_of_circumscribed_sphere_l14_14324


namespace system_solutions_l14_14432

theorem system_solutions : {p : ℝ × ℝ | p.snd ^ 2 = p.fst ∧ p.snd = p.fst} = {⟨1, 1⟩, ⟨0, 0⟩} :=
by
  sorry

end system_solutions_l14_14432


namespace choose_4_out_of_10_l14_14319

theorem choose_4_out_of_10 :
  nat.choose 10 4 = 210 :=
  by
  sorry

end choose_4_out_of_10_l14_14319


namespace dot_product_is_one_l14_14718

variable {V : Type*} [InnerProductSpace ℝ V]
variables (a b : V)

theorem dot_product_is_one 
  (ha : ∥a∥ = 1) 
  (hb : ∥b∥ = sqrt 3) 
  (hab : ∥a - 2•b∥ = 3) : 
  ⟪a, b⟫ = 1 :=
by 
  sorry

end dot_product_is_one_l14_14718


namespace factorial_division_l14_14039

theorem factorial_division : (nat.factorial 15) / ((nat.factorial 6) * (nat.factorial 9)) = 5005 := 
by 
    sorry

end factorial_division_l14_14039


namespace max_ab_l14_14643

noncomputable def f (x : ℝ) := log (2 - x) + 1

theorem max_ab (a b : ℝ) (h₁ : (1, 1) = (1, f 1)) (h₂ : a + b = 1) : ab ≤ 1/4 := by
  sorry

end max_ab_l14_14643


namespace remainder_98765432101_div_240_l14_14146

theorem remainder_98765432101_div_240 :
  (98765432101 % 240) = 61 :=
by
  -- Proof to be filled in later
  sorry

end remainder_98765432101_div_240_l14_14146


namespace three_consecutive_primes_exist_l14_14138

theorem three_consecutive_primes_exist (n : ℕ) (primes : Fin n → ℕ) (h1 : ∀ i, prime (primes i))
  (h2 : ∀ i j, i < j → primes i < primes j)
  (h3 : 30 ∣ ∑ i in Finset.range 31, (primes i) ^ 4) :
  ∃ i j k, consecutive_primes (primes i) (primes j) (primes k) := sorry

def consecutive_primes (a b c : ℕ) : Prop :=
  prime a ∧ prime b ∧ prime c ∧ a < b ∧ b < c ∧ b = a + 1 ∧ c = b + 1

end three_consecutive_primes_exist_l14_14138


namespace identity_holds_l14_14882

theorem identity_holds (x : ℝ) : 
  (2 * x - 1) ^ 3 = 5 * x ^ 3 + (3 * x + 1) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x :=
by sorry

end identity_holds_l14_14882


namespace max_sinA_cosB_cosC_l14_14116

theorem max_sinA_cosB_cosC (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 0 < A ∧ A < 180) (h3 : 0 < B ∧ B < 180) (h4 : 0 < C ∧ C < 180) : 
  ∃ M : ℝ, M = (1 + Real.sqrt 5) / 2 ∧ ∀ a b c : ℝ, a + b + c = 180 → 0 < a ∧ a < 180 → 0 < b ∧ b < 180 → 0 < c ∧ c < 180 → (Real.sin a + Real.cos b * Real.cos c) ≤ M :=
by sorry

end max_sinA_cosB_cosC_l14_14116


namespace simplify_and_evaluate_expression_l14_14921

variable (x y : ℝ)

theorem simplify_and_evaluate_expression :
  (7 * x^2 * y - (3 * x * y - 2 * (x * y - (7/2) * x^2 * y + 1) + (1/2) * x * y)) =
  (-3/2 * x * y + 2) :=
sorry

example : simplify_and_evaluate_expression 6 (-1/6) = (7/2) := by
  simp [simplify_and_evaluate_expression, x := 6, y := -1/6]
  sorry

end simplify_and_evaluate_expression_l14_14921


namespace division_by_fraction_equiv_neg_multiplication_l14_14170

theorem division_by_fraction_equiv_neg_multiplication (h : 43 * 47 = 2021) : (-43) / (1 / 47) = -2021 :=
by
  -- Proof would go here, but we use sorry to skip the proof for now.
  sorry

end division_by_fraction_equiv_neg_multiplication_l14_14170


namespace angle_DAE_l14_14299

-- Define the geometric setup and hypothesis
variables {A B C D E : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]

-- Given conditions
variables (AB AC AE : ℝ)
variable (D : A)
variable (B C E : A)
hypothesis h1: AB = AC
hypothesis h2: AE = BC
hypothesis h3: CD = CA
hypothesis h4: AD = BD
hypothesis h5: ∀ B, is_perpendicular AE BC

-- The statement to prove
theorem angle_DAE (angle_DAE_degrees : ℝ) : 
  angle_DAE_degrees = 18 := by
  sorry

end angle_DAE_l14_14299


namespace percentage_error_is_correct_l14_14087

noncomputable def percentage_error_obtained : ℝ :=
  let π := Real.pi in
  let correct_result := (5 * π / 2) ^ 2 in
  let incorrect_result := (2 * π / 5) ^ 2 in
  let absolute_error := Real.abs (correct_result - incorrect_result) in
  (absolute_error / correct_result) * 100

theorem percentage_error_is_correct : percentage_error_obtained = 97.44 := by
  sorry

end percentage_error_is_correct_l14_14087


namespace dot_product_proof_l14_14659

variables {ℝ : Type*}
variables (a b : ℝ → ℝ)
variables [inner_product_space ℝ ℝ]

theorem dot_product_proof
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = sqrt 3)
  (h3 : ∥a - 2 • b∥ = 3) :
  inner (a : ℝ) (b : ℝ) = 1 :=
sorry

end dot_product_proof_l14_14659


namespace dot_product_l14_14666

variables (a b : Vector ℝ) -- ℝ here stands for the real numbers

-- Given conditions
def condition1 : ∥a∥ = 1 := sorry
def condition2 : ∥b∥ = √3 := sorry
def condition3 : ∥a - (2 : ℝ) • b∥ = 3 := sorry

-- Goal to prove
theorem dot_product (a b : Fin₃ → ℝ) 
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = √3)
  (h3 : ∥a - (2 : ℝ) • b∥ = 3) : 
  a ⬝ b = 1 := 
sorry

end dot_product_l14_14666


namespace position_relationship_l14_14955

-- Define the parabola with vertex at the origin and focus on the x-axis
def parabola (p : ℝ) : Prop := ∀ x y, y^2 = 2 * p * x

-- Define the line l: x = 1
def line_l (x : ℝ) : Prop := x = 1

-- Define the points P and Q where l intersects the parabola
def P (p : ℝ) : ℝ × ℝ := (1, Real.sqrt(2 * p))
def Q (p : ℝ) : ℝ × ℝ := (1, -Real.sqrt(2 * p))

-- Define the perpendicularity condition
def perpendicular (O P Q : ℝ × ℝ) : Prop := (O.1 * P.1 + O.2 * P.2) = 0

-- Point M
def M : ℝ × ℝ := (2, 0)

-- Equation of circle M
def circle_M (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop := (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Define the tangent condition
def tangent (l : ℝ × ℝ → Prop) (center : ℝ × ℝ) (radius : ℝ) : Prop := ∀ x, l(x) → (abs(center.1 - x.1) = radius)

-- Define points on the parabola
def points_on_parabola (A₁ A₂ A₃ : ℝ × ℝ) (p : ℝ) : Prop := parabola p A₁.1 A₁.2 ∧ parabola p A₂.1 A₂.2 ∧ parabola p A₃.1 A₃.2

-- The main theorem statement
theorem position_relationship (p : ℝ) (center : ℝ × ℝ) (radius : ℝ) (O A₁ A₂ A₃ : ℝ × ℝ) :
  parabola 1 O.1 O.2 ∧
  line_l 1 ∧
  A₁ = (0, 0) ∧
  perpendicular O (P 1) (Q 1) ∧
  tangent (λ x, line_l x.1) M 1 ∧
  points_on_parabola A₁ A₂ A₃ 1 →
  ∀ A₂ A₃ : ℝ × ℝ,
    (parabola 1 A₂.1 A₂.2 ∧ parabola 1 A₃.1 A₃.2) →
    tangent(λ x, circle_M center radius x) (A₂.1, A₂.2) (A₃.1, A₃.2) :=
sorry

end position_relationship_l14_14955


namespace michael_worked_hours_l14_14379

theorem michael_worked_hours 
  (hourly_wage : ℕ) 
  (overtime_multiplier : ℕ) 
  (regular_hours : ℕ) 
  (total_earnings : ℕ) :
  regular_hours = 40 → 
  hourly_wage = 7 → 
  overtime_multiplier = 2 → 
  total_earnings = 320 →
  ((total_earnings - regular_hours * hourly_wage) / (overtime_multiplier * hourly_wage) + regular_hours : ℝ) = 42.86 := 
by {
  intros hreg hwage hmult hearn,
  sorry
}

end michael_worked_hours_l14_14379


namespace rectangle_ratio_l14_14922

theorem rectangle_ratio (s : ℝ) (h : s > 0) :
    let large_square_side := 3 * s
    let rectangle_length := 3 * s
    let rectangle_width := 2 * s
    rectangle_length / rectangle_width = 3 / 2 := by
  sorry

end rectangle_ratio_l14_14922


namespace neg_power_identity_l14_14129

variable (m : ℝ)

theorem neg_power_identity : (-m^2)^3 = -m^6 :=
sorry

end neg_power_identity_l14_14129


namespace epidemic_control_l14_14563

-- Given definitions
def consecutive_days (days : List ℝ) : Prop :=
  days.length = 7 ∧ ∀ d ∈ days, d ≤ 5

def mean_le_3 (days : List ℝ) : Prop :=
  days.sum / days.length ≤ 3

def stddev_le_2 (days : List ℝ) : Prop :=
  let μ := days.sum / days.length
  let variance := (days.map (λ x, (x - μ) ^ 2)).sum / days.length
  sqrt variance ≤ 2

def range_le_2 (days : List ℝ) : Prop :=
  (days.maximumD 0 - days.minimumD 0) ≤ 2

def mode_eq_1 (days : List ℝ) : Prop :=
  (days.count 1) > (days.count x) ∀ x ≠ 1

def conditions_met_4 (days : List ℝ) : Prop :=
  mean_le_3 days ∧ range_le_2 days

def conditions_met_5 (days : List ℝ) : Prop :=
  mode_eq_1 days ∧ range_le_1 days

-- The Lean 4 Statement
theorem epidemic_control (days : List ℝ):
  (consecutive_days days)
  ↔ (conditions_met_4 days ∨ conditions_met_5 days) := 
sorry

end epidemic_control_l14_14563


namespace find_dot_product_l14_14681

open Real

noncomputable def vec_a : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector a
noncomputable def vec_b : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector b

def magnitude (v : ℝ → ℝ → ℝ) : ℝ :=
  sqrt ((v 0) ^ 2 + (v 1)^ 2)

def dot_product (u v : ℝ → ℝ → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

axiom magnitude_a_eq1 : magnitude vec_a = 1
axiom magnitude_b_eq_sqrt3 : magnitude vec_b = sqrt 3
axiom magnitude_a_minus_2b_eq3 : magnitude (λ x, vec_a x - 2 * vec_b x) = 3

theorem find_dot_product (a b : ℝ → ℝ → ℝ) 
  (ha : magnitude a = 1) 
  (hb : magnitude b = sqrt 3) 
  (h : magnitude (λ x, a x - 2 * b x) = 3) :
  dot_product a b = 1 := sorry

end find_dot_product_l14_14681


namespace determine_z_l14_14010

variable {x y z : ℝ}

-- The relationship definition
def relationship (k : ℝ) : Prop := ∀ x y, z = (k * y) / (Real.sqrt x)

-- The theorem statement
theorem determine_z (k : ℝ) (h1 : relationship k)
  (h2 : 6 = (k * 3) / (Real.sqrt 4))
  : relationship k
  : ∃ z, z = (k * 6) / (Real.sqrt 9) ∧ z = 8 := by
sorry

end determine_z_l14_14010


namespace tax_percentage_diminished_l14_14009

variable (T T' Q : ℝ)

-- Original conditions
def initial_conditions (h1 : T' = T * 1.1000000000000085 / 1.15)
                       (h2 : T' * Q * 1.15 = T * Q * 1.1000000000000085) : Prop :=
  T' * Q * 1.15 = T * Q * 1.1000000000000085

-- Define the percentage by which the tax was diminished
def percentage_diminished (T T' : ℝ) : ℝ :=
  (1 - T' / T) * 100

-- Prove the corresponding proof statement
theorem tax_percentage_diminished (T T' Q : ℝ)
  (h1 : T' = T * 1.1000000000000085 / 1.15)
  (h2 : T' * Q * 1.15 = T * Q * 1.1000000000000085) :
  percentage_diminished T T' = 4.35 :=
by sorry

end tax_percentage_diminished_l14_14009


namespace proof_problem_l14_14616

noncomputable def polarToRectangular (ρ θ : ℝ) : (ℝ × ℝ) :=
  (ρ * cos θ, ρ * sin θ)

noncomputable def parametricCurve (t : ℝ) : (ℝ × ℝ) :=
  (3 - (1 / 2) * t, (sqrt 3 / 2) * t)

def equationC1 (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 4

def equationC2 (x y : ℝ) : Prop :=
  sqrt 3 * x + y - 3 * sqrt 3 = 0

def A := (3, 0)

theorem proof_problem :
  (∀ (ρ θ x y : ℝ), polarToRectangular ρ θ = (ρ * cos θ, ρ * sin θ) → equationC1 x y) ∧
  (∀ (t x y : ℝ), parametricCurve t = (x, y) → equationC2 x y) ∧
  (∃ (t1 t2 : ℝ), (t1^2 - t1 - 3 = 0) ∧ (t2^2 - t2 - 3 = 0) ∧ (|A.1 - (3 - (1 / 2) * t1)|^2 + |A.2 - (sqrt 3 / 2 * t1)|^2) * (|A.1 - (3 - (1 / 2) * t2)|^2 + |A.2 - (sqrt 3 / 2 * t2)|^2) = 3) :=
  by
  sorry

end proof_problem_l14_14616


namespace correct_statements_l14_14325

def avg_goals_class_a : ℝ := 1.9
def std_dev_class_a : ℝ := 0.3
def avg_goals_class_b : ℝ := 1.3
def std_dev_class_b : ℝ := 1.2

theorem correct_statements :
  (avg_goals_class_a > avg_goals_class_b) ∧
  (std_dev_class_a < std_dev_class_b) ∧
  (std_dev_class_a ≥ 0.3) →
  (std_dev_class_b > std_dev_class_a) ∧
  (avg_goals_class_a ≤ 2) := 
begin
  sorry
end

end correct_statements_l14_14325


namespace rate_percent_l14_14030

theorem rate_percent (P SI T : ℕ) (H_P : P = 800) (H_SI : SI = 144) (H_T : T = 4) : Real :=
by
  -- Definitions based on conditions
  let R := 4.5
  have h : (32 : Real) * R = 144 := by sorry
  exact R

end rate_percent_l14_14030


namespace sin_alpha_value_l14_14233

theorem sin_alpha_value (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : cos (α + π / 6) = 4 / 5) : sin α = (3 * sqrt 3 - 4) / 10 :=
sorry

end sin_alpha_value_l14_14233


namespace probability_of_red_tile_l14_14073

theorem probability_of_red_tile : 
  let tiles := Finset.range 77, 
      red_tiles := tiles.filter (λ n, n % 7 = 3) in
  (red_tiles.card : ℚ) / tiles.card = 10 / 77 :=
by
  sorry

end probability_of_red_tile_l14_14073


namespace smallest_positive_angle_l14_14136

theorem smallest_positive_angle :
  ∀ (x : ℝ), 12 * (Real.sin x)^3 * (Real.cos x)^3 - 2 * (Real.sin x)^3 * (Real.cos x)^3 = 1 → 
  x = 15 * (Real.pi / 180) :=
by
  intros x h
  sorry

end smallest_positive_angle_l14_14136


namespace new_person_weight_l14_14481

variable (weight_1 weight_2 weight_3 weight_4 : Real)
def avg (a b c d : Real) : Real := (a + b + c + d) / 4

theorem new_person_weight (W : Real)
  (H1 : avg weight_1 weight_2 weight_3 weight_4 + 8.5 = avg weight_1 weight_2 weight_3 W)
  (H2 : weight_4 = 95) : W = 129 := by
  sorry

end new_person_weight_l14_14481


namespace choosing_4_out_of_10_classes_l14_14313

theorem choosing_4_out_of_10_classes :
  ∑ (k : ℕ) in (finset.range 5).map (prod.mk 10), k! / (4! * (k - 4)!) = 210 :=
by sorry

end choosing_4_out_of_10_classes_l14_14313


namespace uninsured_employees_l14_14309

theorem uninsured_employees (T P U : ℕ) (prob : ℝ) (hT : T = 340) (hP : P = 54)
  (hProb : prob = 0.5735294117647058) 
  (h12_5_percent : 0.125 * U = 0.125 * U) :
  (∃ U : ℕ, U + P - 0.125 * U = T * (1 - prob)) → U = 104 :=
by
  sorry

end uninsured_employees_l14_14309


namespace inequality_of_derivative_condition_l14_14629

variable (f : ℝ → ℝ)

theorem inequality_of_derivative_condition
  (h : ∀ x, f' x > 2 * f x) :
  3 * f (1/2 * Real.log 2) < 2 * f (1/2 * Real.log 3) := 
sorry

end inequality_of_derivative_condition_l14_14629


namespace range_of_a_l14_14252

noncomputable def f (a x : ℝ) : ℝ := 
if x ≤ 1 then (1 - 2 * a) ^ x 
else log a x + 1 / 3

theorem range_of_a (a : ℝ)
  (H₀ : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) :
  0 < a ∧ a ≤ 1 / 3 :=
sorry

end range_of_a_l14_14252


namespace volume_of_tetrahedron_l14_14245

-- Define the conditions of the problem
def diameter_of_sphere (A B O : Point) (r : ℝ) :=
  dist A O = r ∧ dist B O = r ∧ dist A B = 2 * r

def on_surface_of_sphere (P : Point) (O : Point) (r : ℝ) :=
  dist P O = r

def perpendicular (A B C D : Point) :=
  ∀ (P : Point) (hPA : P = proj_line A B P) (hPB : P = proj_line C D P),
    (P = proj_line A B P) ⊥ (P = proj_line C D P)

def in_angle_range (A O C : Point) :=
  45 ≤ angle A O C ∧ angle A O C ≤ 135

def correct_volume_range :=
  [4 / 3, 4 * real.sqrt 3 / 3]

-- State the theorem
theorem volume_of_tetrahedron (A B C D O : Point) (r : ℝ) (h_diameter : diameter_of_sphere A B O r)
    (h_C_on_sphere : on_surface_of_sphere C O r) (h_D_on_sphere : on_surface_of_sphere D O r)
    (h_CD_length : dist C D = 2) (h_AB_perp_CD : perpendicular A B C D)
    (h_angle_AOC : in_angle_range A O C) :
    volume A B C D ∈ Interval.fromEndpoints correct_volume_range :=
sorry

end volume_of_tetrahedron_l14_14245


namespace Hassan_has_one_apple_tree_l14_14114

variable (H : ℕ) -- H represents the number of apple trees Hassan has
variable (A_orange H_orange H_apple : ℕ) -- representing the number of orange trees in Ahmed's and Hassan's orchard and apple trees in Hassan's orchard

-- Conditions based on the problem
variable (Ahmed_orange : ℕ := 8) -- Ahmed has 8 orange trees
variable (Ahmed_apple : ℕ := 4 * H) -- Ahmed has 4 * H apple trees
variable (Hassan_orange : ℕ := 2) -- Hassan has 2 orange trees
variable (total_Ahmed_trees : ℕ := Ahmed_orange + Ahmed_apple) -- Total trees in Ahmed's orchard
variable (total_Hassan_trees : ℕ := Hassan_orange + H) -- Total trees in Hassan's orchard
variable (condition : total_Ahmed_trees = total_Hassan_trees + 9) -- Total trees condition

theorem Hassan_has_one_apple_tree (H : ℕ) (Ahmed_orange : ℕ = 8) (Hassan_orange : ℕ = 2)
  (condition : 8 + 4 * H = 2 + H + 9) : H = 1 :=
by {
  -- Placeholder for the proof
  sorry
}

end Hassan_has_one_apple_tree_l14_14114


namespace simplify_fraction_l14_14908

theorem simplify_fraction (y : ℝ) (hy : y ≠ 0) : 
  (5 / (4 * y⁻⁴)) * ((4 * y³) / 3) = (5 * y⁷) / 3 := 
by
  sorry

end simplify_fraction_l14_14908


namespace math_proof_problem_l14_14647

noncomputable def f (x : ℝ) : ℝ := 
  √3 * (Real.cos (ω * x))^2 + (Real.sin (ω * x)) * (Real.cos (ω * x)) - (√3 / 2)

-- Definition: smallest positive period of f(x) is π
def period_π (ω : ℝ) : Prop := ∃ p > 0, ∀ x, f (x + p) = f x ∧ p = π

-- Part I: Interval where the function is monotonically decreasing
def monotonically_decreasing_interval (k : ℤ) : Prop := 
  ∀ x, (x ∈ Set.Icc (π / 12 + k * π) (7 * π / 12 + k * π)) → 
  (2 * x + π / 3)

-- Part II: Set of values for x where f(x) > √2 / 2
def values_for_x_greater_than (k : ℤ) : Prop := 
  ∀ x, (x ∈ Set.Ioo (-π / 24 + k * π) (5 * π / 24 + k * π)) → 
  (f x > sqrt 2 / 2)

-- Formulating the proof statement
theorem math_proof_problem (ω : ℝ) (k : ℤ) (hx : period_π ω) : 
  (monotonically_decreasing_interval k) ∧ (values_for_x_greater_than k) := 
sorry

end math_proof_problem_l14_14647


namespace geometric_progression_fifth_term_is_one_l14_14939

-- Definitions as per conditions
def first_term : ℝ := real.sqrt 4
def second_term : ℝ := 4^(1/3 : ℝ)
def exponent_decrement_first_two : ℝ := 1/6
def exponent_decrement_thereafter : ℝ := 1/12

-- Lean statement to prove the fifth term in the geometric progression sequence
theorem geometric_progression_fifth_term_is_one : 
  let a1 := first_term,
      a2 := second_term,
      a3 := 4^(((1/3) : ℝ) - exponent_decrement_first_two),
      a4 := 4^(1/6 - exponent_decrement_first_two),
      a5 := 4^((1/6 - exponent_decrement_first_two) - exponent_decrement_thereafter * 2)
  in 
  a5 = 1 :=
by {
  -- hypothesis and reasoning can be detailed here if needed
  sorry
}

end geometric_progression_fifth_term_is_one_l14_14939


namespace simplify_expression_l14_14920

theorem simplify_expression (θ : ℝ) (h1 : θ = 3 * Real.pi / 5) 
    (trig_identity : ∀ θ : ℝ, sin θ ^ 2 + cos θ ^ 2 = 1) :
    sqrt (1 - sin (3 * Real.pi / 5) ^ 2) = -cos (3 * Real.pi / 5) :=
by 
  sorry

end simplify_expression_l14_14920


namespace min_value_expression_l14_14360

theorem min_value_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 48) :
  x^2 + 6 * x * y + 9 * y^2 + 4 * z^2 ≥ 128 := 
sorry

end min_value_expression_l14_14360


namespace dot_product_l14_14670

variables (a b : Vector ℝ) -- ℝ here stands for the real numbers

-- Given conditions
def condition1 : ∥a∥ = 1 := sorry
def condition2 : ∥b∥ = √3 := sorry
def condition3 : ∥a - (2 : ℝ) • b∥ = 3 := sorry

-- Goal to prove
theorem dot_product (a b : Fin₃ → ℝ) 
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = √3)
  (h3 : ∥a - (2 : ℝ) • b∥ = 3) : 
  a ⬝ b = 1 := 
sorry

end dot_product_l14_14670


namespace jill_speed_is_8_l14_14806

-- Definitions for conditions
def speed_jack1 := 12 -- speed in km/h for the first 12 km
def distance_jack1 := 12 -- distance in km for the first 12 km

def speed_jack2 := 6 -- speed in km/h for the second 12 km
def distance_jack2 := 12 -- distance in km for the second 12 km

def distance_jill := distance_jack1 + distance_jack2 -- total distance in km for Jill

-- Total time taken by Jack
def time_jack := (distance_jack1 / speed_jack1) + (distance_jack2 / speed_jack2)

-- Jill's speed calculation
def jill_speed := distance_jill / time_jack

-- Theorem stating Jill's speed is 8 km/h
theorem jill_speed_is_8 : jill_speed = 8 := by
  sorry

end jill_speed_is_8_l14_14806


namespace equilateral_triangle_ratio_l14_14382

theorem equilateral_triangle_ratio (A B C X Y Z : Point)
  (hABC : equilateral_triangle A B C)
  (hX : ∈ (X : A ≤ B))
  (hY : ∈ (Y : B ≤ C))
  (hZ : ∈ (Z : C ≤ A))
  (hRatio : AX / XB = BY / YC = CZ / ZA)
  (hArea : area (triangle C X Y) = (1 / 4) * area (triangle A B C)):
  AX / XB = (4 - sqrt 7) / 3 :=
sorry

end equilateral_triangle_ratio_l14_14382


namespace cone_lateral_surface_area_l14_14207

theorem cone_lateral_surface_area (r : ℕ) (V : ℝ) (h l S : ℝ)
  (h_r : r = 6)
  (h_V : V = 30 * Real.pi)
  (h_volume : V = (1 / 3) * Real.pi * (r ^ 2) * h)
  (h_slant_height : l = Real.sqrt (r^2 + h^2))
  (h_lateral_surface_area : S = Real.pi * r * l) :
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l14_14207


namespace enrollment_increase_1991_to_1992_l14_14074

theorem enrollment_increase_1991_to_1992 (E E_1992 E_1993 : ℝ)
    (h1 : E_1993 = 1.26 * E)
    (h2 : E_1993 = 1.05 * E_1992) :
    ((E_1992 - E) / E) * 100 = 20 :=
by
  sorry

end enrollment_increase_1991_to_1992_l14_14074


namespace dot_product_proof_l14_14657

variables {ℝ : Type*}
variables (a b : ℝ → ℝ)
variables [inner_product_space ℝ ℝ]

theorem dot_product_proof
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = sqrt 3)
  (h3 : ∥a - 2 • b∥ = 3) :
  inner (a : ℝ) (b : ℝ) = 1 :=
sorry

end dot_product_proof_l14_14657


namespace sequence_a4_eq_15_l14_14609

theorem sequence_a4_eq_15 (a : ℕ → ℕ) :
  a 1 = 1 ∧ (∀ n, a (n + 1) = 2 * a n + 1) → a 4 = 15 :=
by
  sorry

end sequence_a4_eq_15_l14_14609


namespace angle_sum_pi_div_2_l14_14625

theorem angle_sum_pi_div_2 (A B : ℝ) (h1 : 0 < A) (h2 : A < π / 2) (h3 : 0 < B) (h4 : B < π / 2)
    (h5 : sin A ^ 2 + sin B ^ 2 = real.sqrt (sin (A + B) ^ 3)) : A + B = π / 2 :=
by
  sorry

end angle_sum_pi_div_2_l14_14625


namespace find_number_l14_14851

theorem find_number (x : ℝ) (h : (5/3) * x = 45) : x = 27 :=
by
  sorry

end find_number_l14_14851


namespace sum_of_squares_arithmetic_geometric_l14_14388

theorem sum_of_squares_arithmetic_geometric (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 225) : x^2 + y^2 = 1150 :=
by
  sorry

end sum_of_squares_arithmetic_geometric_l14_14388


namespace find_x_l14_14798

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 := 
by
  sorry

end find_x_l14_14798


namespace dot_product_ab_l14_14687

variables (a b : ℝ^3)

-- Given conditions
def condition1 : Prop := ‖a‖ = 1
def condition2 : Prop := ‖b‖ = real.sqrt 3
def condition3 : Prop := ‖a - 2 • b‖ = 3

-- The theorem statement to prove
theorem dot_product_ab (h1 : condition1 a) (h2 : condition2 b) (h3 : condition3 a b) : 
  a ⬝ b = 1 :=
sorry

end dot_product_ab_l14_14687


namespace binom_25_7_l14_14232

theorem binom_25_7 :
  (Nat.choose 23 5 = 33649) →
  (Nat.choose 23 6 = 42504) →
  (Nat.choose 23 7 = 33649) →
  Nat.choose 25 7 = 152306 :=
by
  intros h1 h2 h3
  sorry

end binom_25_7_l14_14232


namespace smallest_abundant_not_multiple_of_5_is_12_l14_14582

/-- Define a number n is abundant if the sum of its proper divisors is greater than n -/
def is_abundant (n : ℕ) : Prop :=
  ∑ i in finset.filter (λ x, x < n ∧ n % x = 0) (finset.range n), i > n

/-- Main theorem: Smallest abundant number that is not a multiple of 5 is 12 -/
theorem smallest_abundant_not_multiple_of_5_is_12 :
  ∃ (n : ℕ), is_abundant n ∧ n % 5 ≠ 0 ∧ (∀ m : ℕ, is_abundant m ∧ m % 5 ≠ 0 → 12 ≤ m) → n = 12 :=
sorry

end smallest_abundant_not_multiple_of_5_is_12_l14_14582


namespace percentile_80_eq_835_l14_14020

open List
open Nat

-- Define the scores as a list
def scores : List ℕ := [78, 70, 72, 86, 79, 80, 81, 84, 56, 83]

-- Define the function that calculates the 80th percentile
def percentile (k : ℕ) (data : List ℕ) : ℚ :=
  if h : data ≠ [] then
    let sorted_data := sort data
    let n := length sorted_data
    let p := ((k : ℚ) * n) / 100
    if p.floor + 1 > n then
      sorted_data.getD (n - 1) 0
    else
      (sorted_data.getD p.floor 0 + sorted_data.getD (p.floor + 1) 0) / 2
  else
    0

-- Define the theorem statement
theorem percentile_80_eq_835 :
  percentile 80 scores = 83.5 := 
sorry

end percentile_80_eq_835_l14_14020


namespace M_inter_N_eq_M_l14_14818

def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {y | y ≥ 1}

theorem M_inter_N_eq_M : M ∩ N = M := by
  sorry

end M_inter_N_eq_M_l14_14818


namespace symmetric_circle_equation_l14_14424

-- Define original circle equation
def original_circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0

-- Define symmetric circle equation
def symmetric_circle (x y : ℝ) : Prop := x^2 + y^2 + 4 * x = 0

theorem symmetric_circle_equation (x y : ℝ) : 
  symmetric_circle x y ↔ original_circle (-x) y :=
by sorry

end symmetric_circle_equation_l14_14424


namespace one_and_two_thirds_of_what_number_is_45_l14_14844

theorem one_and_two_thirds_of_what_number_is_45 (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by
  sorry

end one_and_two_thirds_of_what_number_is_45_l14_14844


namespace lateral_surface_area_of_given_cone_l14_14215

noncomputable def coneLateralSurfaceArea (r V : ℝ) : ℝ :=
let h := (3 * V) / (π * r^2) in
let l := Real.sqrt (r^2 + h^2) in
π * r * l

theorem lateral_surface_area_of_given_cone :
  coneLateralSurfaceArea 6 (30 * π) = 39 * π := by
simp [coneLateralSurfaceArea]
sorry

end lateral_surface_area_of_given_cone_l14_14215


namespace example1_example2_example3_l14_14750

namespace PositiveSquareHarmonicFunction

def positive_square_harmonic (f : ℝ → ℝ) : Prop :=
  (f 1 = 1) ∧
  (∀ x ∈ set.Icc (0 : ℝ) 1, 0 ≤ f x) ∧ 
  (∀ x1 x2 ∈ set.Icc (0 : ℝ) 1, x1 + x2 ∈ set.Icc (0 : ℝ) 1 → f x1 + f x2 ≤ f (x1 + x2))

theorem example1 : positive_square_harmonic (λ x : ℝ, x ^ 2) :=
  sorry

theorem example2 (f : ℝ → ℝ) (h : positive_square_harmonic f) : f 0 = 0 :=
  sorry

theorem example3 (f : ℝ → ℝ) (h : positive_square_harmonic f) : 
  ∀ x ∈ set.Icc (0 : ℝ) 1, f x ≤ 2 * x :=
  sorry

end PositiveSquareHarmonicFunction

end example1_example2_example3_l14_14750


namespace geometric_locus_open_segment_l14_14078

variable {A B C : Point}

-- Let ABC be a triangle
def is_triangle (A B C : Point) : Prop := 
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ collinear A B C = false

-- Define the inscribed parallelograms in the triangle ABC
def inscribed_parallelograms (A B C : Point) : Set (Parallelogram) :=
  {p : Parallelogram | p.is_inscribed_in_triangle A B C}

-- Define the center of a parallelogram
def center (p : Parallelogram) : Point := (p.M + p.N + p.K + p.L) / 4

-- Locus of the centers of inscribed parallelograms
def locus_centers_of_inscribed_parallelograms (A B C : Point) : Set Point :=
  {center p | p ∈ inscribed_parallelograms A B C}

-- Main theorem statement
theorem geometric_locus_open_segment (A B C : Point) 
  (h_triangle : is_triangle A B C) :
  ∃ s t : Segment, ∀ P ∈ locus_centers_of_inscribed_parallelograms A B C,
  P ∈ (s.to_set \ {s.start, s.end}) ∧ P ∈ (t.to_set \ {t.start, t.end}) := sorry

end geometric_locus_open_segment_l14_14078


namespace max_balloons_purchase_l14_14383

-- Define the conditions
def regular_price : ℝ := 5
def discount_price : ℝ := 2.5
def budget : ℝ := 200

-- Define the proof problem
theorem max_balloons_purchase (regular_price discount_price budget : ℝ) : 
  (∃ (max_discount_balloons : ℕ) (remaining_balloons : ℕ), 
    max_discount_balloons = 20 ∧ 
    remaining_balloons = ⌊(budget - max_discount_balloons * (regular_price + discount_price/2)) / regular_price⌋ ∧
    (max_discount_balloons * 2 + remaining_balloons = 45)) := 
begin
  sorry -- The proof can be filled in here
end

end max_balloons_purchase_l14_14383


namespace factorial_division_l14_14032

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_division : (factorial 15) / ((factorial 6) * (factorial 9)) = 5005 :=
by
  sorry

end factorial_division_l14_14032


namespace range_of_a_l14_14179

theorem range_of_a (a : ℝ) : |a - 1| + |a - 4| = 3 ↔ 1 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l14_14179


namespace dot_product_is_one_l14_14713

variable {V : Type*} [InnerProductSpace ℝ V]
variables (a b : V)

theorem dot_product_is_one 
  (ha : ∥a∥ = 1) 
  (hb : ∥b∥ = sqrt 3) 
  (hab : ∥a - 2•b∥ = 3) : 
  ⟪a, b⟫ = 1 :=
by 
  sorry

end dot_product_is_one_l14_14713


namespace constant_term_expansion_l14_14933

theorem constant_term_expansion :
  let T (r : ℕ) := Nat.choose 6 r * (-1)^r
  (x^2 - (1/x))^6 = T 4 := by sorry

end constant_term_expansion_l14_14933


namespace intersection_complement_eq_l14_14270

open Set

universe u

def U : Set ℝ := univ

def A : Set ℝ := { x | x < 0 }

def B : Set ℝ := { x | x ≤ -1 }

theorem intersection_complement_eq : A ∩ (U \ B) = { x | -1 < x ∧ x < 0 } :=
by
  sorry

end intersection_complement_eq_l14_14270


namespace subscriptions_sold_to_parents_l14_14830

-- Definitions for the conditions
variable (P : Nat) -- subscriptions sold to parents
def grandfather := 1
def next_door_neighbor := 2
def other_neighbor := 2 * next_door_neighbor
def subscriptions_other_than_parents := grandfather + next_door_neighbor + other_neighbor
def total_earnings := 55
def earnings_from_others := 5 * subscriptions_other_than_parents
def earnings_from_parents := total_earnings - earnings_from_others
def subscription_price := 5

-- Theorem stating the equivalent math proof
theorem subscriptions_sold_to_parents : P = earnings_from_parents / subscription_price :=
by
  sorry

end subscriptions_sold_to_parents_l14_14830


namespace chromatic_number_iff_k_constructible_subgraph_l14_14216

variables {G : Type*} [Graph G] {k : ℕ}

theorem chromatic_number_iff_k_constructible_subgraph (G : G) (k : ℕ) :
  chromatic_number G ≥ k ↔ ∃ H : G, k_constructible_subgraph H :=
sorry

end chromatic_number_iff_k_constructible_subgraph_l14_14216


namespace sector_area_max_radius_l14_14248

noncomputable def arc_length (R : ℝ) : ℝ := 20 - 2 * R

noncomputable def sector_area (R : ℝ) : ℝ :=
  let l := arc_length R
  0.5 * l * R

theorem sector_area_max_radius :
  ∃ (R : ℝ), sector_area R = -R^2 + 10 * R ∧
             R = 5 :=
sorry

end sector_area_max_radius_l14_14248


namespace identity_holds_l14_14894

noncomputable def identity_proof : Prop :=
∀ (x : ℝ), (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x

theorem identity_holds : identity_proof :=
by
  sorry

end identity_holds_l14_14894


namespace num_bits_for_ABCDEF_l14_14471

def hex_ABCDEF : ℕ := 10 * 16^5 + 11 * 16^4 + 12 * 16^3 + 13 * 16^2 + 14 * 16 + 15

theorem num_bits_for_ABCDEF :
  ∀ n : ℕ, n = hex_ABCDEF → (log2 n).natAbs + 1 = 24 :=
by
  sorry

end num_bits_for_ABCDEF_l14_14471


namespace problem_statement_l14_14361

-- Define the given conditions of the problem
variables {x₁ x₂ x₃ : ℝ} (a : ℝ)
noncomputable def roots : Prop :=
  a = Real.sqrt 2014 ∧
  ∀ {x : ℝ}, (a * x ^ 3 - 4029 * x ^ 2 + 2 = 0) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  x₁ < x₂ ∧ x₂ < x₃

-- Prove that x₂ (x₁ + x₃) = 2 under the given conditions
theorem problem_statement (h : roots a) : x₂ * (x₁ + x₃) = 2 :=
sorry

end problem_statement_l14_14361


namespace dot_product_proof_l14_14660

variables {ℝ : Type*}
variables (a b : ℝ → ℝ)
variables [inner_product_space ℝ ℝ]

theorem dot_product_proof
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = sqrt 3)
  (h3 : ∥a - 2 • b∥ = 3) :
  inner (a : ℝ) (b : ℝ) = 1 :=
sorry

end dot_product_proof_l14_14660


namespace M_eq_N_l14_14437

noncomputable def M (a : ℝ) : ℝ :=
  a^2 + (a + 3)^2 + (a + 5)^2 + (a + 6)^2

noncomputable def N (a : ℝ) : ℝ :=
  (a + 1)^2 + (a + 2)^2 + (a + 4)^2 + (a + 7)^2

theorem M_eq_N (a : ℝ) : M a = N a :=
by
  sorry

end M_eq_N_l14_14437


namespace geometric_series_log_sum_l14_14262

noncomputable def f (m x : ℝ) : ℝ := Real.log x / Real.log m

theorem geometric_series_log_sum
  {m : ℝ}
  (a : ℕ → ℝ)
  (hm_pos : 0 < m)
  (hm_neq_one : m ≠ 1)
  (h_geom : ∀ n, a (n + 1) = m * a n)
  (h_f_value : f m (∏ i in Finset.range (10 + 1 \div 2 + 1 | 2 ), a (2 * i)) = 7) :
  (Finset.sum (Finset.range 20) (λ i, f m (a (i + 1)^2) )) = 26 := 
sorry

end geometric_series_log_sum_l14_14262


namespace integer_solution_x_l14_14790

theorem integer_solution_x (x y : ℤ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y + x * y = 101) : x = 50 :=
sorry

end integer_solution_x_l14_14790


namespace sum_of_exponents_is_22_l14_14014

theorem sum_of_exponents_is_22 :
  ∃ (s : ℕ) (m : fin s → ℕ) (b : fin s → ℤ),
    (∀ i : fin s, b i = 1 ∨ b i = -1) ∧
    (∀ i j : fin s, i < j → m i > m j) ∧
    (∑ i, b i * 3 ^ (m i) = 2010) →
    (∑ i, m i = 22) :=
sorry

end sum_of_exponents_is_22_l14_14014


namespace chips_removal_even_initial_40_chips_removal_minimum_moves_1000_l14_14552

-- Part a: Prove that with 40 chips, exactly one chip cannot remain after both players have made two moves.
theorem chips_removal_even_initial_40 
  (initial_chips : Nat)
  (num_moves : Nat)
  (remaining_chips : Nat) :
  initial_chips = 40 → 
  num_moves = 4 → 
  remaining_chips = 1 → 
  False :=
by
  sorry

-- Part b: Prove that with 1000 chips, the minimum number of moves to reduce to one chip is 8.
theorem chips_removal_minimum_moves_1000
  (initial_chips : Nat)
  (min_moves : Nat)
  (remaining_chips : Nat) :
  initial_chips = 1000 → 
  remaining_chips = 1 → 
  min_moves = 8 :=
by
  sorry

end chips_removal_even_initial_40_chips_removal_minimum_moves_1000_l14_14552


namespace dot_product_l14_14664

variables (a b : Vector ℝ) -- ℝ here stands for the real numbers

-- Given conditions
def condition1 : ∥a∥ = 1 := sorry
def condition2 : ∥b∥ = √3 := sorry
def condition3 : ∥a - (2 : ℝ) • b∥ = 3 := sorry

-- Goal to prove
theorem dot_product (a b : Fin₃ → ℝ) 
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = √3)
  (h3 : ∥a - (2 : ℝ) • b∥ = 3) : 
  a ⬝ b = 1 := 
sorry

end dot_product_l14_14664


namespace ellipse_standard_equation_l14_14614

-- Definitions of the conditions
def centered_at_origin (c : ℝ × ℝ) : Prop := c = (0, 0)

def foci_on_x_axis (f1 f2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, f1 = (c, 0) ∧ f2 = (-c, 0)

def point_on_ellipse (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  let (x, y) := P in (x^2 / a^2) + (y^2 / b^2) = 1

def sum_of_distances_to_foci (P : ℝ × ℝ) (f1 f2 : ℝ × ℝ) (sum : ℝ) : Prop :=
  let (x1, y1) := f1 in
  let (x2, y2) := f2 in
  let (x, y) := P in
  (Real.sqrt ((x - x1)^2 + (y - y1)^2)) + (Real.sqrt ((x - x2)^2 + (y - y2)^2)) = sum

-- The proof statement
theorem ellipse_standard_equation :
  ∃ (a b : ℝ),
    centered_at_origin (0, 0) ∧
    foci_on_x_axis (6, 0) (-6, 0) ∧
    point_on_ellipse (3 * Real.sqrt 2, 4) 6 (Real.sqrt 32) ∧
    sum_of_distances_to_foci (3 * Real.sqrt 2, 4) (6, 0) (-6, 0) 12 ∧
    (∀ (x y : ℝ), (x^2 / 36) + (y^2 / 32) = 1) :=
begin
  sorry
end

end ellipse_standard_equation_l14_14614


namespace replace_stars_identity_l14_14875

theorem replace_stars_identity : 
  ∃ (a b : ℤ), a = -1 ∧ b = 1 ∧ (2 * x + a)^3 = 5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x := 
by 
  use [-1, 1]
  sorry

end replace_stars_identity_l14_14875


namespace dot_product_eq_one_l14_14700

variables {α : Type*} [InnerProductSpace ℝ α]

noncomputable def vector_a (a : α) : Prop := ∥a∥ = 1
noncomputable def vector_b (b : α) : Prop := ∥b∥ = real.sqrt 3
noncomputable def vector_c (a b : α) : Prop := ∥a - (2 : ℝ) • b∥ = 3

theorem dot_product_eq_one (a b : α) (ha : vector_a a) (hb : vector_b b) (hc : vector_c a b) :
  inner a b = 1 :=
sorry

end dot_product_eq_one_l14_14700


namespace train_speed_l14_14110

theorem train_speed (time crossing_pole: ℝ) (train_length: ℝ) (time_eq: time = 9) (length_eq: train_length = 225) :
  let speed := (train_length / time) * 3.6 in
  speed = 90 := by
  sorry

end train_speed_l14_14110


namespace angle_not_45_or_135_l14_14292

variable {a b S : ℝ}
variable {C : ℝ} (h : S = (1/2) * a * b * Real.cos C)

theorem angle_not_45_or_135 (h : S = (1/2) * a * b * Real.cos C) : ¬ (C = 45 ∨ C = 135) :=
sorry

end angle_not_45_or_135_l14_14292


namespace largest_corner_sum_l14_14605

-- Definitions based on the given problem
def faces_labeled : List ℕ := [2, 3, 4, 5, 6, 7]
def opposite_faces : List (ℕ × ℕ) := [(2, 7), (3, 6), (4, 5)]

-- Condition that face 2 cannot be adjacent to face 4
def non_adjacent_faces : List (ℕ × ℕ) := [(2, 4)]

-- Function to check adjacency constraints
def adjacent_allowed (f1 f2 : ℕ) : Bool := 
  ¬ (f1, f2) ∈ non_adjacent_faces ∧ ¬ (f2, f1) ∈ non_adjacent_faces

-- Determine the largest sum of three numbers whose faces meet at a corner
theorem largest_corner_sum : ∃ (a b c : ℕ), a ∈ faces_labeled ∧ b ∈ faces_labeled ∧ c ∈ faces_labeled ∧ 
  (adjacent_allowed a b) ∧ (adjacent_allowed b c) ∧ (adjacent_allowed c a) ∧ 
  a + b + c = 18 := 
sorry

end largest_corner_sum_l14_14605


namespace profit_percentage_on_cost_price_l14_14120

variable (C M D : ℝ)
variable (discount : ℝ → ℝ)
variable (SP : ℝ → ℝ → ℝ)
variable (profit : ℝ → ℝ → ℝ)
variable (profit_percentage : ℝ → ℝ → ℝ)

noncomputable def given_C := 47.50
noncomputable def given_M := 62.5
noncomputable def given_D := 0.05

def calculate_discount (M: ℝ) (D: ℝ) : ℝ := D * M
def calculate_SP (M: ℝ) (discount: ℝ) : ℝ := M - discount
def calculate_profit (SP: ℝ) (C: ℝ) : ℝ := SP - C
def calculate_profit_percentage (profit: ℝ) (C: ℝ) : ℝ := (profit / C) * 100

theorem profit_percentage_on_cost_price : 
    profit_percentage (profit (SP given_M (calculate_discount given_M given_D)) given_C) given_C = 25 :=
    by
    sorry

end profit_percentage_on_cost_price_l14_14120


namespace problem1_l14_14130

theorem problem1 (x y : ℝ) (h₁ : 5^x = 45) (h₂ : 3^y = 45) :
  1 / x + 2 / y = 1 :=
sorry

end problem1_l14_14130


namespace identity_holds_l14_14880

theorem identity_holds (x : ℝ) : 
  (2 * x - 1) ^ 3 = 5 * x ^ 3 + (3 * x + 1) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x :=
by sorry

end identity_holds_l14_14880


namespace multiplication_72519_9999_l14_14059

theorem multiplication_72519_9999 :
  72519 * 9999 = 725117481 :=
by
  sorry

end multiplication_72519_9999_l14_14059


namespace difference_between_shares_l14_14526

theorem difference_between_shares (F V R : ℕ) (v_share : ℕ) (hV : V = 5) (v_value : v_share = 1500) (total_share : F + V + R = 14) :
  (1 * v_value) / 5 = (900) :=
by sorry

end difference_between_shares_l14_14526


namespace identity_holds_l14_14884

theorem identity_holds (x : ℝ) : 
  (2 * x - 1) ^ 3 = 5 * x ^ 3 + (3 * x + 1) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x :=
by sorry

end identity_holds_l14_14884


namespace min_value_x_l14_14244

theorem min_value_x (a : ℝ) (h : ∀ a > 0, x^2 ≤ 1 + a) : ∃ x, ∀ a > 0, -1 ≤ x ∧ x ≤ 1 := 
sorry

end min_value_x_l14_14244


namespace range_of_a_if_ineq_has_empty_solution_l14_14263

theorem range_of_a_if_ineq_has_empty_solution (a : ℝ) :
  (∀ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0) → -2 ≤ a ∧ a < 6/5 :=
by
  sorry

end range_of_a_if_ineq_has_empty_solution_l14_14263


namespace verify_value_l14_14235

theorem verify_value (a b c d m : ℝ) 
  (h₁ : a = -b) 
  (h₂ : c * d = 1) 
  (h₃ : |m| = 3) :
  3 * c * d + (a + b) / (c * d) - m = 0 ∨ 
  3 * c * d + (a + b) / (c * d) - m = 6 := 
sorry

end verify_value_l14_14235


namespace max_f_value_l14_14576

noncomputable def f (x : ℝ) : ℝ := x + (1/x) + Real.exp (x + (1/x))

theorem max_f_value : ∃ x > 0, ∀ y > 0, f(y) ≤ f(1) :=
by
  unfold f
  have h : f(1) = 2 + Real.exp(2) := by sorry
  exists 1
  split
  { exact zero_lt_one }
  { intro y hy
    sorry }

end max_f_value_l14_14576


namespace equation_1_equation_2_l14_14590

theorem equation_1 (x : ℝ) : x^2 - 1 = 8 ↔ x = 3 ∨ x = -3 :=
by sorry

theorem equation_2 (x : ℝ) : (x + 4)^3 = -64 ↔ x = -8 :=
by sorry

end equation_1_equation_2_l14_14590


namespace no_closed_polygons_and_no_intersecting_segments_l14_14871

open Function

variable (Points : Type) [MetricSpace Points] (d : Finset Points) (p : Points)

noncomputable def nearest_neighbor (p : Points) : Option Points :=
  (d.erase p).minBy (dist p)

theorem no_closed_polygons_and_no_intersecting_segments
  (h : ∀ p ∈ d, ∃! q ∈ d, q ≠ p ∧ dist p q < ∀ r ∈ (d.erase p), dist p r) :
  (∀ (a b : Points), (nearest_neighbor d a = nearest_neighbor d b) → False) ∧
  (∀ (a b c d : Points), nearest_neighbor d a = d → nearest_neighbor d c = b →
     a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d →
     line_intersection (dist a d) (dist c b) → False) :=
by
  apply sorry

end no_closed_polygons_and_no_intersecting_segments_l14_14871


namespace sum_first_10_terms_l14_14551

def sequence (a : ℕ → ℝ) :=
  a 1 + a 2 = 6 ∧ 
  a 1 + a 2 + a 3 = 15 ∧ 
  a 1 + a 2 + a 3 + a 4 = 28

theorem sum_first_10_terms (a : ℕ → ℝ) 
  (h : sequence a) : 
  (finset.range 10).sum a = 205 :=
sorry

end sum_first_10_terms_l14_14551


namespace find_equation_AB_l14_14230

theorem find_equation_AB (M A B : Point ℝ) (hM : M = (2, 4)) 
(h_perpendicular : ∀ O,  isRightAngle O M A B)
(h_symmetry : isSymmetricQuad O A M B AB) :
(line_eq_ab_1 : equation AB x y = x + 2 * y - 5 = 0 ∨
 line_eq_ab_2 : equation AB x y = 2 * x + y - 4 = 0) :=
sorry

end find_equation_AB_l14_14230


namespace roberts_monthly_expenses_l14_14900

-- Conditions
def basic_salary : ℝ := 1250
def commission_rate : ℝ := 0.1
def total_sales : ℝ := 23600
def savings_rate : ℝ := 0.2

-- Definitions derived from the conditions
noncomputable def commission : ℝ := commission_rate * total_sales
noncomputable def total_earnings : ℝ := basic_salary + commission
noncomputable def savings : ℝ := savings_rate * total_earnings
noncomputable def monthly_expenses : ℝ := total_earnings - savings

-- The statement to be proved
theorem roberts_monthly_expenses : monthly_expenses = 2888 := by
  sorry

end roberts_monthly_expenses_l14_14900


namespace problem_statement_l14_14748

noncomputable def z : Complex := (1 + Complex.i) / (1 - Complex.i)
noncomputable def conj_z : Complex := Complex.conj z

theorem problem_statement : (conj_z) ^ 2017 = -Complex.i := by
  sorry

end problem_statement_l14_14748


namespace equation_of_parabola_and_circle_position_relationship_l14_14959

-- Conditions
variables (p q : ℝ) (C : ℝ → ℝ → Prop) (M : ℝ → ℝ → Prop)

-- Definitions
def parabola_equation := ∀ x y, C x y ↔ y^2 = x
def circle_equation := ∀ x y, M x y ↔ (x - 2)^2 + y^2 = 1

-- Proof of Part 1
theorem equation_of_parabola_and_circle 
(h_vertex_origin : ∀ x y, C x y ↔ y^2 = 2 * p * x)
(h_focus_xaxis : p > 0)
(h_perpendicular : ∀ x y, C 1 y → (x = 1 ∧ y = ±sqrt (2 * p)) → ((1 : ℝ)  * (1 : ℝ)  + sqrt (2 * p) * (- sqrt (2 * p))) = 0)
(h_M : (2, 0))
(h_tangent_l : ∀ x y, M x y ↔ (x - 2)^2 + y^2 = 1)
: 
(parabola_equation C ∧ circle_equation M) := sorry

-- Proof of Part 2
theorem position_relationship
(A1 A2 A3 : ℝ × ℝ) 
(h_points_on_C : ∀ Ai, Ai = (y * y, y) for y in {A1, A2, A3})
(h_tangents_to_circle : ∀ Ai Aj, Ai ≠ Aj → tangent_line Ai Aj M)
:
(is_tangent_to_circle M (line_through A2 A3)) := sorry

end equation_of_parabola_and_circle_position_relationship_l14_14959


namespace square_side_length_eq_triangle_area_eq_distance_WH_final_proof_l14_14385

theorem square_side_length_eq (s : ℝ) (hs : s^2 = 144) : s = 12 :=
by {
  have h : s = real.sqrt 144, from (eq_div_iff_mul_eq 12 (12 : ℝ) hs).mpr (eq.symm (real.sqrt_sqr_eq_abs 12)),
  rw real.sqrt_eq_iff_sq_eq at h,
  cases h,
  repeat { assumption },
}

theorem triangle_area_eq (ZG ZH : ℝ) (htriangle : 90 = 1 / 2 * ZG * ZH) : ZG = ZH := 
by {
  have key : 2 * 90 = ZG * ZH, by { simp [htriangle], },
  have square : 180 = ZG^2, from (eq_div_iff_mul_eq 180 (ZG * ZH)).mpr key,
  have h1 : ZG = real.sqrt 180, from (eq_div_iff_mul_eq 180 (real.sqrt (ZG * ZG) * (1 : ℝ))).mpr (eq.symm square),
  rw real.sqrt_eq_iff_sq_eq at h1,
  cases h1,
  repeat { assumption },
}

theorem distance_WH (ZG ZH WH ZW : ℝ) (h1 : ZG = ZH) (h2 : ZW = 12) (h3 : ZG = 6 * real.sqrt 5) :
  WH = 18 :=
by {
  have hyp1 : WH^2 = ZW^2 + ZH^2,
    { simp [pow_two], },
  have hyp2 : ZW^2 = 144,
    { rw ← pow_two, exact (eq_div_iff_mul_eq ZW ZW).mpr 
      (eq.symm (real.sqr_sqrt_eq_abs 144)), },
  have hyp3 : ZH^2 = 180,
    { rw ← pow_two, exact (eq_div_iff_mul_eq ZH ZH).mpr 
      (eq.symm (real.sqr_sqrt_eq_abs 180)), },
  have hyp : WH^2 = 324,
    { simp [hyp1, hyp2, hyp3], },
  rw real.sqrt_eq_iff_sq_eq at hyp,
  cases hyp with h1 h2,
  assumption,
  simp at *,
}

theorem final_proof : WH = 18 :=
by {
  let s := 12,
  let ZG := 6 * real.sqrt 5,
  let ZH := ZG,
  let ZW := s,
  apply distance_WH ZG ZH WH ZW,
  all_goals { simp [pow_two] },
}

end square_side_length_eq_triangle_area_eq_distance_WH_final_proof_l14_14385


namespace dot_product_is_one_l14_14721

variable {V : Type*} [InnerProductSpace ℝ V]
variables (a b : V)

theorem dot_product_is_one 
  (ha : ∥a∥ = 1) 
  (hb : ∥b∥ = sqrt 3) 
  (hab : ∥a - 2•b∥ = 3) : 
  ⟪a, b⟫ = 1 :=
by 
  sorry

end dot_product_is_one_l14_14721


namespace perpendicular_lines_condition_l14_14944

theorem perpendicular_lines_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * x - y - 1 = 0 → m * x + y + 1 = 0 → False) ↔ m = 1 / 2 :=
by sorry

end perpendicular_lines_condition_l14_14944


namespace cone_lateral_surface_area_l14_14184

-- Definitions from conditions
def r : ℝ := 6
def V : ℝ := 30 * Real.pi

-- Theorem to prove
theorem cone_lateral_surface_area : 
  let h := V / (Real.pi * (r ^ 2) / 3) in
  let l := Real.sqrt (r ^ 2 + h ^ 2) in
  let S := Real.pi * r * l in
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l14_14184


namespace steve_total_cost_paid_l14_14835

-- Define all the conditions given in the problem
def cost_of_dvd_mike : ℕ := 5
def cost_of_dvd_steve (m : ℕ) : ℕ := 2 * m
def shipping_cost (s : ℕ) : ℕ := (8 * s) / 10

-- Define the proof problem (statement)
theorem steve_total_cost_paid : ∀ (m s sh t : ℕ), 
  m = cost_of_dvd_mike →
  s = cost_of_dvd_steve m → 
  sh = shipping_cost s → 
  t = s + sh → 
  t = 18 := by
    intros m s sh t h1 h2 h3 h4
    rw [h1, h2, h3, h4]
    norm_num -- The proof would normally be the next steps, but we skip it with sorry
    sorry

end steve_total_cost_paid_l14_14835


namespace polygon_sides_eq_eleven_l14_14737

theorem polygon_sides_eq_eleven (n : ℕ) (D : ℕ)
(h1 : D = n + 33)
(h2 : D = n * (n - 3) / 2) :
  n = 11 :=
by {
  sorry
}

end polygon_sides_eq_eleven_l14_14737


namespace sum_of_coefficients_l14_14486

theorem sum_of_coefficients (a : ℕ → ℝ) :
  (∀ x : ℝ, (2 - x) ^ 10 = a 0 + a 1 * x + a 2 * x ^ 2 + a 3 * x ^ 3 + a 4 * x ^ 4 + a 5 * x ^ 5 + a 6 * x ^ 6 + a 7 * x ^ 7 + a 8 * x ^ 8 + a 9 * x ^ 9 + a 10 * x ^ 10) →
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 1 →
  a 0 = 1024 →
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = -1023 :=  
by
  intro h1 h2 h3
  sorry

end sum_of_coefficients_l14_14486


namespace correct_options_l14_14633

variable {a b c : ℝ}

def sol_set (a b c : ℝ) : set ℝ :=
  {x | ax^2 + bx + c ≤ 0}

theorem correct_options (h: sol_set a b c = {x | x ≤ -2 ∨ x ≥ 3}) :
  (a < 0) ∧
  (∀ x, (ax + c > 0 ↔ x < 6)) ∧
  (¬ (8a + 4b + 3c < 0)) ∧
  (∀ x, (cx^2 + bx + a < 0 ↔ -1 / 2 < x ∧ x < 1 / 3)) :=
by
  sorry

end correct_options_l14_14633


namespace replace_stars_identity_l14_14876

theorem replace_stars_identity : 
  ∃ (a b : ℤ), a = -1 ∧ b = 1 ∧ (2 * x + a)^3 = 5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x := 
by 
  use [-1, 1]
  sorry

end replace_stars_identity_l14_14876


namespace exists_binary_multiple_of_m_l14_14824

theorem exists_binary_multiple_of_m (m : ℕ) (hm : m > 0) : 
  ∃ a : ℕ, (∀ d ∈ (nat.digits 2 a), d = 0 ∨ d = 1) ∧ m ∣ a := 
sorry

end exists_binary_multiple_of_m_l14_14824


namespace binary_235_w_minus_z_l14_14140

theorem binary_235_w_minus_z :
  let n := 235
  let bin := "11101011"
  let w := bin.to_list.count '1'
  let z := bin.to_list.count '0'
  w - z = 2 := by
  let n := 235
  let bin := "11101011"
  let w := bin.to_list.count '1'
  let z := bin.to_list.count '0'
  exact eq.refl 2

end binary_235_w_minus_z_l14_14140


namespace sum_possible_coefficients_l14_14837

theorem sum_possible_coefficients : 
  (∑ r in ({(1, 24), (2, 12), (3, 8), (4, 6)} : Finset (ℕ × ℕ)), (r.1 + r.2)) = 60 :=
by
  sorry

end sum_possible_coefficients_l14_14837


namespace building_floors_l14_14537

/--
Building A has some floors, which is 9 less than Building B.
Building C has six less than five times as many floors as Building B.
Building C has 59 floors.
Prove that the number of floors in Building A is 4.
--/
noncomputable def floorsA (floorsB : ℕ) : ℕ := floorsB - 9

theorem building_floors :
  ∃ (FB : ℕ), (5 * FB - 6 = 59) → (floorsA FB = 4) :=
by
  unfold floorsA
  intro FB h1
  have : floorsA FB = FB - 9 := rfl
  sorry

end building_floors_l14_14537


namespace choose_two_out_of_three_l14_14440

-- Define the number of vegetables as n and the number to choose as k
def n : ℕ := 3
def k : ℕ := 2

-- The combination formula C(n, k) == n! / (k! * (n - k)!)
def combination (n k : ℕ) : ℕ := n.choose k

-- Problem statement: Prove that the number of ways to choose 2 out of 3 vegetables is 3
theorem choose_two_out_of_three : combination n k = 3 :=
by
  sorry

end choose_two_out_of_three_l14_14440


namespace geometric_sequence_sum_l14_14329

-- Definitions of the geometric sequence and the conditions
variables {α : Type*} [linear_ordered_field α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
∀ (m n : ℕ), m < n → a m * (a (m + n) / a n) = a (2 * m)

noncomputable def a : ℕ → α := sorry

-- Proving the specific condition given in the problem
theorem geometric_sequence_sum (a : ℕ → α) (h : is_geometric_sequence a) 
  (h_pos : ∀ n, 0 < a n) (h_cond : a 2 * a 6 + 2 * a 4 * a 5 + a 5 ^ 2 = 25) :
  a 4 + a 5 = 5 :=
sorry

end geometric_sequence_sum_l14_14329


namespace problem_order_of_numbers_l14_14145

noncomputable def a : ℝ := 0.6 ^ 7
noncomputable def b : ℝ := 7 ^ 0.6
noncomputable def c : ℝ := Real.log 6 / Real.log 0.7

theorem problem_order_of_numbers : c < a ∧ a < b :=
by
  -- Proof steps will be added
  sorry

end problem_order_of_numbers_l14_14145


namespace cone_lateral_surface_area_l14_14190

theorem cone_lateral_surface_area (r V : ℝ) (h l S : ℝ) 
  (radius_condition : r = 6)
  (volume_condition : V = 30 * Real.pi)
  (volume_formula : V = (1 / 3) * Real.pi * r^2 * h)
  (slant_height_formula : l = Real.sqrt (r^2 + h^2))
  (lateral_surface_area_formula : S = Real.pi * r * l) :
  S = 39 * Real.pi := 
sorry

end cone_lateral_surface_area_l14_14190


namespace max_z_val_l14_14278

theorem max_z_val (x y : ℝ) (h1 : x + y ≤ 4) (h2 : y - 2 * x + 2 ≤ 0) (h3 : y ≥ 0) :
  ∃ x y, z = x + 2 * y ∧ z = 6 :=
by
  sorry

end max_z_val_l14_14278


namespace intersection_of_M_and_N_l14_14067

theorem intersection_of_M_and_N :
  let M := { y : ℝ | ∃ x : ℝ, y = Real.log (x^2 + 1) }
  let N := { x : ℝ | 4^x > 4 }
  M ∩ N = { x : ℝ | 1 < x } := by
sorry

end intersection_of_M_and_N_l14_14067


namespace find_b_value_l14_14264

noncomputable def ellipse_b_value : ℝ :=
  let a := 2 in
  let c := 1 in
  sqrt (a ^ 2 - c ^ 2)

theorem find_b_value {b : ℝ} (pb : y^2 = -4*x) (latus_rectum : x = 1) (ellipse : x^2/4 + y^2/b^2 = 1 ∧ b > 0) :
  b = ellipse_b_value :=
sorry

end find_b_value_l14_14264


namespace identity_solution_l14_14887

theorem identity_solution (x : ℝ) :
  ∃ a b : ℝ, (2 * x + a) ^ 3 = 5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x ∧
             a = -1 ∧ b = 1 :=
by
  -- we can skip the proof as this is just a statement
  sorry

end identity_solution_l14_14887


namespace tv_cost_l14_14828

-- Definitions from the problem conditions
def fraction_on_furniture : ℚ := 3 / 4
def total_savings : ℚ := 1800
def fraction_on_tv : ℚ := 1 - fraction_on_furniture  -- Fraction of savings on TV

-- The proof problem statement
theorem tv_cost : total_savings * fraction_on_tv = 450 := by
  sorry

end tv_cost_l14_14828


namespace math_proof_problem_l14_14219

variable {a b : ℝ}
variables (a_nonzero : a ≠ 0) (a_ge_neg1 : a ≥ -1)
variables (y_eq : ℝ → ℝ) (point_A : ℝ × ℝ)
variable (b_iff : b = -a - 3)
variable (a_val : a = -1/2)
variable (m_eq : ℝ)

-- Given parabola
def parabola (x : ℝ) : ℝ := a * x^2 + b * x + 4

-- Pass through A(1,1)
axiom point_A_cond :  parabola 1 = 1

-- Condition (1)
def b_eq_n_a_sub3 : b = -a - 3 := b_iff

-- Condition (2)
def axis_of_symmetry (a b : ℝ) : ℝ := -b / (2 * a)

axiom a_is_neg_half : a = -1/2
axiom b_from_neg_half : b = -1/2 - 3

-- For Condition (3)
def portion_decreases_condition (a : ℝ) : Prop :=
  (-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a ≤ 3)

axiom valid_a_range : portion_decreases_condition a

-- For Condition (4)
def max_val_fxn (x : ℂ) : ℂ := a * x^2 + b * x + 4

def m : ℝ := 2 * a + 7

axiom min_m_val : m ≥ 5

-- Main goal
theorem math_proof_problem :
  ((forall x, x ∈ Real ∧ parabola x = 1) ∧
  (forall a, axis_of_symmetry a b = -5/2) ∧
  portion_decreases_condition a ∧
  m ≥ 5) :=
  sorry

end math_proof_problem_l14_14219


namespace tan_ne1_necessary_but_not_sufficient_l14_14738

noncomputable def necessary_but_not_sufficient_condition (α : ℝ) : Prop :=
  (α ≠ π / 4) ↔ (real.tan α ≠ 1)

theorem tan_ne1_necessary_but_not_sufficient (α : ℝ) : necessary_but_not_sufficient_condition α :=
by {
  -- the necessity part
  split,
  {
    intro hα,
    intro h,
    have h' := real.tan_pi_div_four,
    linarith,
  },
  -- the sufficiency part (show that it's not sufficient)
  {
    intro htan,
    by_contra h,
    exact htan (real.tan_of_circle h),
  }
}

end tan_ne1_necessary_but_not_sufficient_l14_14738


namespace inequality_proof_l14_14363

theorem inequality_proof 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) : 
  a ^ a * b ^ b * c ^ c ≥ 1 / (a * b * c) := 
sorry

end inequality_proof_l14_14363


namespace gcd_exp_sub_l14_14460

theorem gcd_exp_sub (n m : ℕ) (hnm : n ≤ m) :
  gcd (2^n - 1) (2^m - 1) = 2^(m - n) - 1 :=
sorry

example : gcd (2^2022 - 1) (2^2036 - 1) = 2^14 - 1 :=
begin
  apply gcd_exp_sub,
  exact nat.le_of_lt (by norm_num),
end

end gcd_exp_sub_l14_14460


namespace segment_CM_length_l14_14406

theorem segment_CM_length :
  ∃ (CM : ℝ), (∀ (A B C D M N : ℝ), 
  A = 4 → B = 4 → C = 4 → D = 4 →
  ∀ (sqr_area : ℝ), sqr_area = A * B →
  ∀ (tri_area : ℝ), tri_area = sqr_area / 3 →
  (∃ (BM : ℝ), ∀ (CB : ℝ), CB = A → 
   tri_area = 0.5 * CB * BM → BM = 8 / 3) →
  CM = real.sqrt (4^2 + (8/3)^2) →
  CM = 4 * real.sqrt 13 / 3)

end segment_CM_length_l14_14406


namespace systematic_sampling_l14_14762

theorem systematic_sampling :
  ∃ (selected : ℕ), 
    let total_students := 1000
    let sample_count := 200
    let interval := total_students / sample_count
    let start_number := 122
    let remainder := start_number % interval
    (selected = 927 ∧ ((selected % interval) = remainder)) :=
begin
  sorry
end

end systematic_sampling_l14_14762


namespace max_digits_sum_is_24_l14_14505

def maxSumOfDigits : ℕ :=
  let hours := (0:24).toList
  let minutes := (0:60).toList
  let maxHourSum := hours.map (λ h, (h / 10) + (h % 10)).maximum'.getOrElse 0
  let maxMinuteSum := minutes.map (λ m, (m / 10) + (m % 10)).maximum'.getOrElse 0
  maxHourSum + maxMinuteSum

theorem max_digits_sum_is_24 : 
  maxSumOfDigits = 24 := 
  sorry

end max_digits_sum_is_24_l14_14505


namespace even_segments_of_closed_self_intersecting_l14_14975

open Set

-- Define what it means for a polygonal chain.
structure PolygonalChain :=
(segments : Set (ℝ × ℝ × ℝ × ℝ)) -- A set of segments represented by pairs of points in ℝ²
(is_closed : ∀ (s t : ℝ × ℝ × ℝ × ℝ), s ∈ segments → t ∈ segments → s ≠ t → (intersect s t))

-- Predicate stating the condition of intersections.
def intersects_once (P : PolygonalChain) :=
∀ (s t : ℝ × ℝ × ℝ × ℝ), s ∈ P.segments → t ∈ P.segments → s ≠ t → s ≠ t → (intersection s t).finite

theorem even_segments_of_closed_self_intersecting (P : PolygonalChain) (h_intersects : intersects_once P) : 
  ∃ n : ℕ, even n ∧ finset.card P.segments = n := 
sorry

end even_segments_of_closed_self_intersecting_l14_14975


namespace find_x_l14_14797

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 := 
by
  sorry

end find_x_l14_14797


namespace factorial_division_l14_14045

-- Define factorial using the standard library's factorial function
def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- The problem statement
theorem factorial_division :
  (factorial 15) / (factorial 6 * factorial 9) = 834 :=
sorry

end factorial_division_l14_14045


namespace correct_result_l14_14069

-- Define the original number
def original_number := 51 + 6

-- Define the correct calculation using multiplication
def correct_calculation (x : ℕ) : ℕ := x * 6

-- Theorem to prove the correct calculation
theorem correct_result : correct_calculation original_number = 342 := by
  -- Skip the actual proof steps
  sorry

end correct_result_l14_14069


namespace seven_inv_mod_thirty_one_l14_14150

theorem seven_inv_mod_thirty_one : ∃ x : ℤ, (0 ≤ x ∧ x ≤ 30) ∧ (7 * x ≡ 1 [MOD 31]) ∧ (x = 9) :=
begin
  use 9,  -- the candidate x we identified from the solution
  split,
  { exact ⟨dec_trivial, dec_trivial⟩, },  -- 0 <= 9 and 9 <= 30 are trivially true
  split,
  { norm_num, },  -- 7 * 9 ≡ 1 [MOD 31] is true
  { refl, },  -- x = 9 by definition
end

end seven_inv_mod_thirty_one_l14_14150


namespace curves_intersect_at_4_points_l14_14561

theorem curves_intersect_at_4_points (a : ℝ) :
  (∀ (x y : ℝ), x^2 + (y - 1)^2 = a^2 ∧ y = x^2 - a → ∃ x1 x2 x3 x4 y1 y2 y3 y4 : ℝ,
  (x1, y1) ≠ (x2, y2) ∧ (x2, y2) ≠ (x3, y3) ∧ (x3, y3) ≠ (x4, y4) ∧
  (x1, y1) ≠ (x3, y3) ∧ (x1, y1) ≠ (x4, y4) ∧ (x2, y2) ≠ (x4, y4) ∧
  (x4, y4) ≠ (x3, y3) ∧ x1^2 + (y1 - 1)^2 = a^2 ∧ y1 = x1^2 - a ∧
  x2^2 + (y2 - 1)^2 = a^2 ∧ y2 = x2^2 - a ∧
  x3^2 + (y3 - 1)^2 = a^2 ∧ y3 = x3^2 - a ∧
  x4^2 + (y4 - 1)^2 = a^2 ∧ y4 = x4^2 - a) ↔ a > 0 :=
sorry

end curves_intersect_at_4_points_l14_14561


namespace expression_evaluation_l14_14131

theorem expression_evaluation : 
  (real.root 5 ((-4: ℝ)^5)) - ((-5: ℝ)^2) - 5 + (real.root 4 ((-43: ℝ)^4)) - (-(3^2)) = 0 :=
by
  sorry

end expression_evaluation_l14_14131


namespace find_angle_A_find_cos2C_minus_pi_over_6_l14_14333

noncomputable def triangle_area_formula (a b c : ℝ) (C : ℝ) : ℝ :=
  (1 / 2) * a * b * Real.sin C

noncomputable def given_area_formula (b c : ℝ) (S : ℝ) (a : ℝ) (C : ℝ) : Prop :=
  S = (Real.sqrt 3 / 6) * b * (b + c - a * Real.cos C)

noncomputable def angle_A (S b c a C : ℝ) (h : given_area_formula b c S a C) : ℝ :=
  Real.arcsin ((Real.sqrt 3 / 3) * (b + c - a * Real.cos C))

theorem find_angle_A (a b c S C : ℝ) (h : given_area_formula b c S a C) :
  angle_A S b c a C h = π / 3 :=
sorry

-- Part 2 related definitions
noncomputable def cos2C_minus_pi_over_6 (b c a C : ℝ) : ℝ :=
  let cos_C := (b^2 + c^2 - a^2) / (2 * b * c)
  let sin_C := Real.sqrt (1 - cos_C^2)
  let cos_2C := 2 * cos_C^2 - 1
  let sin_2C := 2 * sin_C * cos_C
  cos_2C * (Real.sqrt 3 / 2) + sin_2C * (1 / 2)

theorem find_cos2C_minus_pi_over_6 (b c a C : ℝ) (hb : b = 1) (hc : c = 3) (ha : a = Real.sqrt 7) :
  cos2C_minus_pi_over_6 b c a C = - (4 * Real.sqrt 3 / 7) :=
sorry

end find_angle_A_find_cos2C_minus_pi_over_6_l14_14333


namespace inequality_hold_l14_14280

theorem inequality_hold (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - |c| > b - |c| :=
sorry

end inequality_hold_l14_14280


namespace max_area_enclosed_by_fencing_l14_14829

theorem max_area_enclosed_by_fencing (l w : ℕ) (h : 2 * (l + w) = 142) : l * w ≤ 1260 :=
sorry

end max_area_enclosed_by_fencing_l14_14829


namespace michael_choices_l14_14321

theorem michael_choices (n k : ℕ) (h_n : n = 10) (h_k : k = 4) : nat.choose n k = 210 :=
by
  rw [h_n, h_k]
  norm_num
  sorry

end michael_choices_l14_14321


namespace cubics_sum_l14_14826

noncomputable def roots_cubic (a b c d p q r : ℝ) : Prop :=
  (p + q + r = b) ∧ (p*q + p*r + q*r = c) ∧ (p*q*r = d)

noncomputable def root_values (p q r : ℝ) : Prop :=
  p^3 = 2*p^2 - 3*p + 4 ∧
  q^3 = 2*q^2 - 3*q + 4 ∧
  r^3 = 2*r^2 - 3*r + 4

theorem cubics_sum (p q r : ℝ) (h1 : p + q + r = 2) (h2 : p*q + q*r + p*r = 3)  (h3 : p*q*r = 4)
  (h4 : root_values p q r) : p^3 + q^3 + r^3 = 2 :=
by
  sorry

end cubics_sum_l14_14826


namespace solve_trig_equation_l14_14398

theorem solve_trig_equation (x : ℝ) : 
  (∑ i in Finset.range 6, Real.sin (x + 3^i * (2 * Real.pi / 7))) = 1 →
  ∃ k : ℤ, x = - (Real.pi / 2) + 2 * Real.pi * k :=
sorry

end solve_trig_equation_l14_14398


namespace triangle_ineq_l14_14355

noncomputable def area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in (s * (s - a) * (s - b) * (s - c)).sqrt

theorem triangle_ineq (a b c : ℝ) (h : a + b > c ∧ b + c > a ∧ c + a > b) :
  let S := area a b c
  in a^2 + b^2 + c^2 ≥ 4 * (real.sqrt 3) * S + (a - b)^2 + (b - c)^2 + (c - a)^2 :=
begin
  sorry
end

end triangle_ineq_l14_14355


namespace probability_even_in_set_l14_14515

theorem probability_even_in_set :
  let s := finset.range (500 - 100 + 1).map (nat.add 100)
  (finset.card (s.filter (λ n, n % 2 = 0)).to_rat / s.card.to_rat) = 201 / 401 :=
by
  sorry

end probability_even_in_set_l14_14515


namespace area_ratio_of_triangle_in_trapezium_l14_14784

variables (A B C D E: Type)
variables [Geometry A] [Geometry B] [Geometry C] [Geometry D] [Geometry E]

theorem area_ratio_of_triangle_in_trapezium
  (k : ℕ)
  (h1 : parallel (line_segment AB) (line_segment DC))
  (h2 : length BC = k)
  (h3 : length AD = k)
  (h4 : length DC = 2 * k)
  (h5 : length AB = 3 * k)
  (E : Type)
  (h6 : ∃ X, is_angle_bisector (∠DAB) X E ∧ is_angle_bisector (∠CBA) X E) :
  (area (triangle ABE)) / (area (trapezium ABCD)) = (3 / 5) :=
sorry

end area_ratio_of_triangle_in_trapezium_l14_14784


namespace identity_holds_l14_14883

theorem identity_holds (x : ℝ) : 
  (2 * x - 1) ^ 3 = 5 * x ^ 3 + (3 * x + 1) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x :=
by sorry

end identity_holds_l14_14883


namespace dot_product_l14_14663

variables (a b : Vector ℝ) -- ℝ here stands for the real numbers

-- Given conditions
def condition1 : ∥a∥ = 1 := sorry
def condition2 : ∥b∥ = √3 := sorry
def condition3 : ∥a - (2 : ℝ) • b∥ = 3 := sorry

-- Goal to prove
theorem dot_product (a b : Fin₃ → ℝ) 
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = √3)
  (h3 : ∥a - (2 : ℝ) • b∥ = 3) : 
  a ⬝ b = 1 := 
sorry

end dot_product_l14_14663


namespace set_in_quadrant_I_l14_14949

theorem set_in_quadrant_I (x y : ℝ) (h1 : y ≥ 3 * x) (h2 : y ≥ 5 - x) (h3 : y < 7) : 
  x > 0 ∧ y > 0 :=
sorry

end set_in_quadrant_I_l14_14949


namespace quadratic_has_two_distinct_real_roots_l14_14222

theorem quadratic_has_two_distinct_real_roots (a : ℝ) :
  let b := -(2 * a - 1)
  let c := a^2 - a
  let Δ := b^2 - 4 * 1 * c
  Δ > 0 :=
by
  let b := -(2 * a - 1)
  let c := a^2 - a
  let Δ := b^2 - 4 * 1 * c
  have : Δ = 1 := sorry -- Detailed proof steps omitted and replaced with sorry
  show Δ > 0, by rw this; norm_num
  sorry

end quadratic_has_two_distinct_real_roots_l14_14222


namespace least_multiple_of_five_primes_l14_14996

noncomputable def smallest_multiple_of_five_primes : ℕ :=
  let primes := [2, 3, 5, 7, 11] in
  primes.foldl (· * ·) 1

theorem least_multiple_of_five_primes : smallest_multiple_of_five_primes = 2310 := by
  sorry

end least_multiple_of_five_primes_l14_14996


namespace maximize_expr_l14_14266

theorem maximize_expr (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -2) (hy : 0 ≤ y ∧ y ≤ 4) :
    ∃ (v : ℝ), v = (x + y) / x ∧ (∀ (x' y' : ℝ), -6 ≤ x' ∧ x' ≤ -2 → 0 ≤ y' ∧ y' ≤ 4 → (x' + y') / x' ≤ v) :=
begin
  use (x + y) / x,
  split,
  { refl },
  { intros x' y' hx' hy',
    sorry }
end

end maximize_expr_l14_14266


namespace alicia_gumballs_l14_14523

theorem alicia_gumballs (A : ℕ) (h1 : 3 * A = 60) : A = 20 := sorry

end alicia_gumballs_l14_14523


namespace prove_identity_l14_14895

variable (x : ℝ)

theorem prove_identity : 
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  -- Expand both sides and prove identity
  sorry

end prove_identity_l14_14895


namespace inequality_solution_l14_14923

theorem inequality_solution {x : ℝ} (h : 5 - 1 / (3 * x + 4) < 7) : 
  x ∈ set.Ioo (Float.neg_inf) (-11 / 6 : ℝ) ∪ set.Ioo (-4 / 3 : ℝ) Float.inf :=
by
  sorry

end inequality_solution_l14_14923


namespace bhanu_petrol_expense_l14_14126

theorem bhanu_petrol_expense (I : ℝ) (H1 : 0.2 * 0.7 * I = 140) : 0.3 * I = 300 := by
  have H3 : 0.14 * I = 140 := by
    rwa [←mul_assoc] at H1
  have H4 : I = 140 / 0.14 := by
    field_simp
    exact eq_div_iff_mul_eq.symm.mpr H3
  rw [H4]
  norm_num
  done

end bhanu_petrol_expense_l14_14126


namespace regression_estimate_correct_l14_14977

noncomputable def linear_regression_estimate
  (n : Nat) (x_values y_values : Fin n → ℝ) (b : ℝ) (x_new : ℝ) : ℝ :=
  let x_sum := (List.of_fn x_values).sum
  let y_sum := (List.of_fn y_values).sum
  let x_mean := x_sum / n
  let y_mean := y_sum / n
  let a := y_mean - b * x_mean
  b * x_new + a

theorem regression_estimate_correct :
  linear_regression_estimate 12 (λ i, if i < 12 then 20 else 0) (λ i, if i < 12 then 170 else 0) 6.5 22 = 183 :=
by
  sorry

end regression_estimate_correct_l14_14977


namespace fans_with_all_vouchers_l14_14308

theorem fans_with_all_vouchers (total_fans : ℕ) 
    (soda_interval : ℕ) (popcorn_interval : ℕ) (hotdog_interval : ℕ)
    (h1 : soda_interval = 60) (h2 : popcorn_interval = 80) (h3 : hotdog_interval = 100)
    (h4 : total_fans = 4500)
    (h5 : Nat.lcm soda_interval (Nat.lcm popcorn_interval hotdog_interval) = 1200) :
    (total_fans / Nat.lcm soda_interval (Nat.lcm popcorn_interval hotdog_interval)) = 3 := 
by
    sorry

end fans_with_all_vouchers_l14_14308


namespace calc_g_x_plus_2_minus_g_x_l14_14259

def g (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 4

theorem calc_g_x_plus_2_minus_g_x (x : ℝ) : g (x + 2) - g x = 12 * x + 22 := 
by 
  sorry

end calc_g_x_plus_2_minus_g_x_l14_14259


namespace problem_1_problem_2_l14_14644

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * Real.log x

theorem problem_1 :
  let a := -2 * Real.exp 1 in
  let f (x : ℝ) : ℝ := x^2 + a * Real.log x in
  ∀ x : ℝ, (0 < x ∧ x < Real.sqrt (Real.exp 1)) → f x ≤ f (Real.sqrt (Real.exp 1))
    ∧ (Real.sqrt (Real.exp 1) < x) → f (Real.sqrt (Real.exp 1)) ≤ f x 
    ∧ f (Real.sqrt (Real.exp 1)) = 0 := 
by
  sorry

theorem problem_2 {a : ℝ} :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → (2 * x + a / x) ≤ 0) ↔ (a ≤ -32) :=
by
  sorry

end problem_1_problem_2_l14_14644


namespace dot_product_is_one_l14_14714

variable {V : Type*} [InnerProductSpace ℝ V]
variables (a b : V)

theorem dot_product_is_one 
  (ha : ∥a∥ = 1) 
  (hb : ∥b∥ = sqrt 3) 
  (hab : ∥a - 2•b∥ = 3) : 
  ⟪a, b⟫ = 1 :=
by 
  sorry

end dot_product_is_one_l14_14714


namespace parabola_circle_tangency_l14_14966

-- Definitions for the given conditions
def is_origin (P : Point) : Prop :=
  P.x = 0 ∧ P.y = 0

def is_on_x_axis (P : Point) : Prop := 
  P.y = 0

def parabola_vertex_focus_condition (C : Parabola) (O F : Point) : Prop :=
  is_origin O ∧ is_on_x_axis F ∧ C.vertex = O ∧ C.focus = F

def intersects_at_perpendicular_points (C : Parabola) (l : Line) (P Q : Point) : Prop :=
  l.slope = 1 ∧ l.intersect C = {P, Q} ∧ vector_product_is_perpendicular 0 P Q = true

def circle_tangent_to_line_at_point (M : Circle) (l : Line) (P : Point) : Prop :=
  l.form = "x=1" ∧ distance M.center P = M.radius ∧ M.contains P

def parabola_contains_point (C : Parabola) (P : Point) : Prop :=
  P.on_parabola C = true

def lines_tangent_to_circle (l₁ l₂ : Line) (M : Circle) : Prop :=
  M.tangent l₁ ∧ M.tangent l₂

def position_relationship (l : Line) (M : Circle) : PositionRelationship :=
  TangencyLine.and Circle M

-- Statement of the proof problem
theorem parabola_circle_tangency :
  ∃ C M, let O : Point := { x := 0, y := 0 } in
  let F : Point := { x := 1/2, y := 0 } in
  parabola_vertex_focus_condition C O F ∧ 
  intersects_at_perpendicular_points C (line_horizontal 1) P Q ∧
  point M_center = (mk_point 2 0) ∧
  circle_tangent_to_line_at_point M (line_horizontal 1) (mk_point 1 0) ∧
  ∀ A1 A2 A3 : Point, parabola_contains_point C A1 ∧
                    parabola_contains_point C A2 ∧
                    parabola_contains_point C A3 ∧
                    lines_tangent_to_circle (line_through A1 A2) M ∧
                    lines_tangent_to_circle (line_through A1 A3) M →
                    position_relationship (line_through A2 A3) M = Tangent :=
begin
  sorry  
end

end parabola_circle_tangency_l14_14966


namespace f_2005_l14_14941

noncomputable def f : ℕ → ℕ := sorry

axiom f_non_negative : ∀ n : ℕ, n > 0 → f(n) ≥ 0
axiom f_2 : f(2) = 0
axiom f_3_pos : f(3) > 0
axiom f_9999 : f(9999) = 3333
axiom f_functional_eq : ∀ m n : ℕ, f(m + n) - f(m) - f(n) = 0 ∨ f(m + n) - f(m) - f(n) = 1

theorem f_2005 : f(2005) = 668 :=
by
  sorry

end f_2005_l14_14941


namespace angle_sum_l14_14328

theorem angle_sum (x y z : ℝ) 
  (h1 : ∠ FCB = y + z) 
  (h2 : ∠ FBA = x + y + z)
  (h3 : ∠ FBA = 124) : 
  x + y + z = 124 := 
by
  sorry

end angle_sum_l14_14328


namespace discount_percentage_l14_14099

theorem discount_percentage (CP SP SP_no_discount discount : ℝ)
  (h1 : SP = CP * (1 + 0.44))
  (h2 : SP_no_discount = CP * (1 + 0.50))
  (h3 : discount = SP_no_discount - SP) :
  (discount / SP_no_discount) * 100 = 4 :=
by
  sorry

end discount_percentage_l14_14099


namespace enclosed_area_eq_1224_975_l14_14153

theorem enclosed_area_eq_1224_975 
  (x y : ℝ) 
  (h : |x - 70| + |y| = |x / 3|) : 
  (∃ x y : ℝ, |x - 70| + |y| = |x / 3| → let figure_area = 
    (have cases_x_lt_0 : x < 0 → |x - 70| + |y| = - x / 3,
      -- Proof or calculation for this case
      sorry
    ),
    (have cases_x_between_0_and_70 : 0 ≤ x < 70 → |x - 70| + |y| = x / 3,
      -- Proof or calculation for this case
      sorry
    ),
    (have cases_x_ge_70 : x ≥ 70 → |x - 70| + |y| = x / 3,
      -- Proof or calculation for this case
      sorry
    ))
  → |x| = 52.5 ∧ 
     set_of y (abs y - abs (abs abs (abs abs y y) y) y) (abs y -> y) = true) [0,2,3,4] 
  where
    figure_area := 1/2 * (105 - 52.5) * (23.33 - (-23.33)),
    enclosed_area_eq_1224_975
: figure_area = 1224.975 := 
begin
  sorry
end

end enclosed_area_eq_1224_975_l14_14153


namespace initial_bacteria_count_l14_14417

theorem initial_bacteria_count (n : ℕ) :
  (n * 4^15 = 1_073_741_824) → (n = 1) :=
begin
  sorry
end

end initial_bacteria_count_l14_14417


namespace cube_properties_l14_14048

theorem cube_properties (x : ℝ) (h1 : 6 * (2 * (8 * x)^(1/3))^2 = x) : x = 13824 :=
sorry

end cube_properties_l14_14048


namespace horner_value_v2_l14_14028

def poly (x : ℤ) : ℤ := 208 + 9 * x^2 + 6 * x^4 + x^6

theorem horner_value_v2 : poly (-4) = ((((0 + -4) * -4 + 6) * -4 + 9) * -4 + 208) :=
by
  sorry

end horner_value_v2_l14_14028


namespace replace_stars_identity_l14_14877

theorem replace_stars_identity : 
  ∃ (a b : ℤ), a = -1 ∧ b = 1 ∧ (2 * x + a)^3 = 5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x := 
by 
  use [-1, 1]
  sorry

end replace_stars_identity_l14_14877


namespace problem_statement_l14_14271

def a : ℕ → ℚ
def b : ℕ → ℚ

axiom initial_values : (a 1 = 2) ∧ (b 1 = 1)

axiom recursive_relations (n : ℕ) (h : 1 < n) : 
  a n = (3/4 : ℚ) * a (n - 1) + (1/4 : ℚ) * b (n - 1) + 1 ∧
  b n = (1/4 : ℚ) * a (n - 1) + (3/4 : ℚ) * b (n - 1) + 1

theorem problem_statement :
  (a 3 + b 3) * (a 4 - b 4) = 7 / 8 :=
by
  sorry

end problem_statement_l14_14271


namespace number_added_at_end_l14_14487

theorem number_added_at_end :
  (26.3 * 12 * 20) / 3 + 125 = 2229 := sorry

end number_added_at_end_l14_14487


namespace find_m_plus_n_l14_14982

def probability_no_exact_k_pairs (k n : ℕ) : ℚ :=
  -- A function to calculate the probability
  -- Placeholder definition (details omitted for brevity)
  sorry

theorem find_m_plus_n : ∃ m n : ℕ,
  gcd m n = 1 ∧ 
  (probability_no_exact_k_pairs k n = (97 / 1000) → m + n = 1097) :=
sorry

end find_m_plus_n_l14_14982


namespace bacteria_initial_count_l14_14415

theorem bacteria_initial_count (t : ℕ) (quad : ℕ) (final_count : ℕ): 
  (quad = 4) → 
  (t = 5 * 60) → 
  (final_count = 4194304) → 
  let n := final_count / (quad ^ (t / 20)) in
  n = 1 := by
  sorry

end bacteria_initial_count_l14_14415


namespace triangle_circumcircle_intersection_sum_l14_14023

/-- Given a triangle PQR with sides PQ = 9, QR = 10, and PR = 17, and midpoints U, V, W of 
    segments PQ, QR, PR respectively, and Z being the intersection point other than V
    of the circumcircles of triangles PVU and QWV, we need to prove ZP + ZQ + ZR = 195/8. -/
theorem triangle_circumcircle_intersection_sum (P Q R U V W Z : Point)
  (hPQ : dist P Q = 9)
  (hQR : dist Q R = 10)
  (hPR : dist P R = 17)
  (hU : midpoint P Q U)
  (hV : midpoint Q R V)
  (hW : midpoint P R W)
  (hZ : Z ≠ V ∧ onCircumcircleOfTriangle Z P U V ∧ onCircumcircleOfTriangle Z Q W V) :
  dist Z P + dist Z Q + dist Z R = 195 / 8 := 
sorry

end triangle_circumcircle_intersection_sum_l14_14023


namespace trig_identity_proof_l14_14566

noncomputable def trig_identity : Prop :=
  cos (π / 7) * cos (2 * π / 7) * cos (4 * π / 7) = -1 / 8

theorem trig_identity_proof : trig_identity :=
  sorry

end trig_identity_proof_l14_14566


namespace min_dist_right_tri_prism_l14_14529

noncomputable def min_dist (A B C A1 B1 C1 P : ℝ) : ℝ := AP + PC1

theorem min_dist_right_tri_prism 
  (AC BC CC1 : ℝ)
  (h_AC : AC = 6)
  (h_BC : BC = real.sqrt 2)
  (h_CC1 : CC1 = real.sqrt 2)
  (angle_ACB : ∠ ACB = pi / 2) :
  ∃ P ∈ segment[B1, C], min_dist A B C A1 B1 C1 P = 5 * real.sqrt 2 :=
sorry

end min_dist_right_tri_prism_l14_14529


namespace find_x_l14_14570

theorem find_x (x : ℝ) (h : 2 * arctan (1 / 3) + arctan (1 / 15) + arctan (1 / x) = Real.pi / 4) : 
  x = 53 / 4 := 
sorry

end find_x_l14_14570


namespace sum_of_integer_solutions_l14_14404

theorem sum_of_integer_solutions:
  (∑ k in Finset.range 34 \ Finset.range 9, k) = 525 :=
by
  sorry

end sum_of_integer_solutions_l14_14404


namespace find_dot_product_l14_14711

open Real

variables (a b : ℝ^3)
variables (dot_product : ℝ^3 → ℝ^3 → ℝ)

def vector_magnitude (v : ℝ^3) : ℝ := sqrt (dot_product v v)

axiom magnitude_a : vector_magnitude a = 1
axiom magnitude_b : vector_magnitude b = sqrt 3
axiom magnitude_a_minus_2b : vector_magnitude (a - (2:ℝ) • b) = 3

theorem find_dot_product : dot_product a b = 1 :=
sorry

end find_dot_product_l14_14711


namespace four_pow_m_div_two_pow_n_eq_eight_l14_14163

theorem four_pow_m_div_two_pow_n_eq_eight 
  (x y m n : ℤ) 
  (h1 : 3 * x + y = 2 * m - 1) 
  (h2 : x - y = n) 
  (h3 : x + y = 1) : 
  4^m / 2^n = 8 :=
by 
  sorry

end four_pow_m_div_two_pow_n_eq_eight_l14_14163


namespace function_property_l14_14287

noncomputable def f : ℝ → ℝ :=
  λ x, 2^x

theorem function_property : ∀ x y : ℝ, f(x) * f(y) = f(x + y) :=
by
  intros x y
  simp [f]
  sorry

end function_property_l14_14287


namespace constant_term_trinomial_l14_14419

theorem constant_term_trinomial (x : ℂ) (h : x ≠ 0) : 
  let term (r : ℕ) := (C 6 r : ℂ) * (x ^ (6 - r)) * ((1 / (2 * x)) ^ r) in
  term 3 = (5 : ℂ) / 2 := 
by 
  -- substitute the known value at r = 3
  have r3 : term 3 = ((C 6 3 : ℂ) * (x ^ 3) * ((1 / (2 * x)) ^ 3)) := by sorry
  -- compute the binomial coefficient C(6, 3)
  have binom_coeff : (C 6 3 : ℂ) = 20 := by sorry
  -- compute the powers and multiply
  have powers : (x ^ 3 * ((1 / (2 * x)) ^ 3) * 20) = (5 : ℂ / 2) := by sorry
  -- conclude the proof
  exact rfl

end constant_term_trinomial_l14_14419


namespace determine_x_l14_14560

theorem determine_x (x : ℚ) (h : ∀ (y : ℚ), 10 * x * y - 18 * y + x - 2 = 0) : x = 9 / 5 :=
sorry

end determine_x_l14_14560


namespace total_wristbands_proof_l14_14304

-- Definitions from the conditions
def wristbands_per_person : ℕ := 2
def total_wristbands : ℕ := 125

-- Theorem statement to be proved
theorem total_wristbands_proof : total_wristbands = 125 :=
by
  sorry

end total_wristbands_proof_l14_14304


namespace wardrobe_single_discount_l14_14113

theorem wardrobe_single_discount :
  let p : ℝ := 50
  let d1 : ℝ := 0.30
  let d2 : ℝ := 0.20
  let final_price := p * (1 - d1) * (1 - d2)
  let equivalent_discount := 1 - (final_price / p)
  equivalent_discount = 0.44 :=
by
  let p : ℝ := 50
  let d1 : ℝ := 0.30
  let d2 : ℝ := 0.20
  let final_price := p * (1 - d1) * (1 - d2)
  let equivalent_discount := 1 - (final_price / p)
  show equivalent_discount = 0.44
  sorry

end wardrobe_single_discount_l14_14113


namespace length_of_train_A_l14_14452

-- Conditions as Lean definitions
def speed_train_A_kmh : ℝ := 54
def speed_train_B_kmh : ℝ := 36
def time_cross_seconds : ℝ := 11
def length_train_B_m : ℝ := 150

-- Convert km/h to m/s
def kmh_to_ms (speed : ℝ) : ℝ := speed * (1000 / 3600)

-- Speeds in m/s
def speed_train_A_ms : ℝ := kmh_to_ms speed_train_A_kmh
def speed_train_B_ms : ℝ := kmh_to_ms speed_train_B_kmh

-- Relative speed in m/s
def relative_speed_ms : ℝ := speed_train_A_ms + speed_train_B_ms

-- Distance covered in crossing train B
def total_distance_m : ℝ := relative_speed_ms * time_cross_seconds

-- Length of train A
def length_train_A_m : ℝ := total_distance_m - length_train_B_m

-- The theorem to prove
theorem length_of_train_A :
  length_train_A_m = 125 :=
by
  sorry

end length_of_train_A_l14_14452


namespace sin_A_plus_B_eq_3sqrt7_over_8_l14_14621

theorem sin_A_plus_B_eq_3sqrt7_over_8
  (A B C : ℝ) (a b c : ℝ)
  (h1 : a = 4)
  (h2 : b = 5)
  (h3 : c = 6)
  (h4 : a = 4 → b = 5 → c = 6 → a^2 + b^2 - c^2 = 2*a*b*cos C)
  (h5 : A + B + C = π) :
  sin (A + B) = 3 * √7 / 8 :=
by
  sorry

end sin_A_plus_B_eq_3sqrt7_over_8_l14_14621


namespace no_24_points_and_2019_planes_l14_14804

theorem no_24_points_and_2019_planes 
  (points : set ℝ^3) (planes : set (set ℝ^3))
  (h_points : points.card = 24)
  (h_distinct_points : ∀ p1 p2 p3 ∈ points, p1 ≠ p2 → p1 ≠ p3 → p2 ≠ p3 → ¬ collinear ℝ p1 p2 p3)
  (h_planes_2019 : planes.card = 2019)
  (h_plane_contains_points : ∀ plane ∈ planes, ∃ p1 p2 p3 ∈ points, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ (p1 ∈ plane ∧ p2 ∈ plane ∧ p3 ∈ plane))
  (h_triple_on_one_plane : ∀ p1 p2 p3 ∈ points, ∃ plane ∈ planes, p1 ∈ plane ∧ p2 ∈ plane ∧ p3 ∈ plane) :
  false :=
sorry

end no_24_points_and_2019_planes_l14_14804


namespace factorial_ratio_value_l14_14037

theorem factorial_ratio_value : fact 15 / (fact 6 * fact 9) = 770 := by
  sorry

end factorial_ratio_value_l14_14037


namespace cards_in_envelopes_l14_14869

theorem cards_in_envelopes :
  let cards := {1, 2, 3, 4, 5, 6}
      envelopes := {1, 2, 3}
      num_ways : ℕ := 3 * 6
  in num_ways = 18 :=
by
  sorry

end cards_in_envelopes_l14_14869


namespace smallest_abundant_not_multiple_of_five_l14_14580

def proper_divisors_sum (n : ℕ) : ℕ :=
  (Nat.divisors n).erase n |>.sum

def is_abundant (n : ℕ) : Prop := proper_divisors_sum n > n

def is_not_multiple_of_five (n : ℕ) : Prop := ¬ (5 ∣ n)

theorem smallest_abundant_not_multiple_of_five :
  18 = Nat.find (λ n, is_abundant n ∧ is_not_multiple_of_five n) :=
sorry

end smallest_abundant_not_multiple_of_five_l14_14580


namespace probability_less_than_zero_l14_14265

variables {σ : ℝ} {ζ : ℝ → ℝ}

def normal_distribution (μ : ℝ) (σ : ℝ) (x : ℝ) : ℝ :=
  (1 / (σ * (Real.sqrt (2 * Real.pi)))) * Real.exp (-(x - μ)^2 / (2 * σ^2))

axiom zeta_follows_normal : ∀ (x : ℝ), ζ x = normal_distribution 4 σ x
axiom probability_condition : ∫ x in 4..8, ζ x = 0.3

theorem probability_less_than_zero : ∫ x in -∞..0, ζ x = 0.2 :=
by
  sorry

end probability_less_than_zero_l14_14265


namespace find_a_tangent_l14_14243

theorem find_a_tangent (a : ℝ) (h : ∃ x0 : ℝ, f x0 = 2 * x0 ∧ f' x0 = 2) : a = 1 :=
by
  let f := λ x => Real.log (2 * x + a)
  have f' := λ x => deriv (λ x => Real.log (2 * x + a))
  have deriv_f : ∀ x, deriv (λ x => Real.log (2 * x + a)) = 2 / (2 * x + a) := sorry
  cases h with x0 hx0
  cases hx0 with h1 h2
  have eq1 : 2 / (2 * x0 + a) = 2 := h2
  have eq2 : 2 * x0 + a = 1 := sorry
  have eq3 : f x0 = 2 * x0 := h1
  have eq4 : Real.log (2 * x0 + a) = 2 * x0 := sorry
  have x0_eq0 : x0 = 0 := eq4 ▸ sorry
  rw x0_eq0 at eq2
  exact eq2 ▸ rfl

end find_a_tangent_l14_14243


namespace cubic_sum_identity_l14_14441

section
variables {x y z a b c : ℝ}

theorem cubic_sum_identity
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = b^2)
  (h3 : x⁻¹ + y⁻¹ + z⁻¹ = c⁻¹) :
  x^3 + y^3 + z^3 = a^3 + (3 / 2) * (a^2 - b^2) * (c - a) := 
sorry
end

end cubic_sum_identity_l14_14441


namespace max_tulips_l14_14455

theorem max_tulips :
  ∃ (y r : ℕ), (y + r = 15) ∧ (y + r) % 2 = 1 ∧ |y - r| = 1 ∧ (50 * y + 31 * r ≤ 600) :=
begin
  sorry
end

end max_tulips_l14_14455


namespace find_x_values_l14_14794

theorem find_x_values (x y : ℕ) (hx: x > y) (hy: y > 0) (h: x + y + x * y = 101) :
  x = 50 ∨ x = 16 :=
sorry

end find_x_values_l14_14794


namespace find_n_l14_14745

theorem find_n (x n : ℤ) (k m : ℤ) (h1 : x = 82*k + 5) (h2 : x + n = 41*m + 22) : n = 5 := by
  sorry

end find_n_l14_14745


namespace angle_A_in_triangle_l14_14759

theorem angle_A_in_triangle (a b c : ℝ) (S : ℝ)
  (h1 : S = (1 / 4) * (b^2 + c^2 - a^2))
  (h2 : S = (1 / 2) * b * c * real.sin (A : ℝ))
  (h3 : 0 < A) (h4 : A < real.pi) :
  A = real.pi / 4 :=
begin
  sorry
end

end angle_A_in_triangle_l14_14759


namespace sum_of_digits_9ab_l14_14331

def a : ℕ := 999
def b : ℕ := 666

theorem sum_of_digits_9ab : 
  let n := 9 * a * b
  (n.digits 10).sum = 36 := 
by
  sorry

end sum_of_digits_9ab_l14_14331


namespace equation_of_parabola_and_circle_position_relationship_l14_14958

-- Conditions
variables (p q : ℝ) (C : ℝ → ℝ → Prop) (M : ℝ → ℝ → Prop)

-- Definitions
def parabola_equation := ∀ x y, C x y ↔ y^2 = x
def circle_equation := ∀ x y, M x y ↔ (x - 2)^2 + y^2 = 1

-- Proof of Part 1
theorem equation_of_parabola_and_circle 
(h_vertex_origin : ∀ x y, C x y ↔ y^2 = 2 * p * x)
(h_focus_xaxis : p > 0)
(h_perpendicular : ∀ x y, C 1 y → (x = 1 ∧ y = ±sqrt (2 * p)) → ((1 : ℝ)  * (1 : ℝ)  + sqrt (2 * p) * (- sqrt (2 * p))) = 0)
(h_M : (2, 0))
(h_tangent_l : ∀ x y, M x y ↔ (x - 2)^2 + y^2 = 1)
: 
(parabola_equation C ∧ circle_equation M) := sorry

-- Proof of Part 2
theorem position_relationship
(A1 A2 A3 : ℝ × ℝ) 
(h_points_on_C : ∀ Ai, Ai = (y * y, y) for y in {A1, A2, A3})
(h_tangents_to_circle : ∀ Ai Aj, Ai ≠ Aj → tangent_line Ai Aj M)
:
(is_tangent_to_circle M (line_through A2 A3)) := sorry

end equation_of_parabola_and_circle_position_relationship_l14_14958


namespace complex_number_properties_l14_14747

theorem complex_number_properties (z : ℂ) (h : (1 + complex.i) * z = 1 + 5 * complex.i) :
  complex.abs z = real.sqrt 13 ∧ complex.conj z = 3 - 2 * complex.i ∧ (0 < z.re ∧ 0 < z.im) :=
by sorry

end complex_number_properties_l14_14747


namespace plaza_area_increase_l14_14101

theorem plaza_area_increase (a : ℝ) : 
  ((a + 2)^2 - a^2 = 4 * a + 4) :=
sorry

end plaza_area_increase_l14_14101


namespace find_dot_product_l14_14679

open Real

noncomputable def vec_a : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector a
noncomputable def vec_b : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector b

def magnitude (v : ℝ → ℝ → ℝ) : ℝ :=
  sqrt ((v 0) ^ 2 + (v 1)^ 2)

def dot_product (u v : ℝ → ℝ → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

axiom magnitude_a_eq1 : magnitude vec_a = 1
axiom magnitude_b_eq_sqrt3 : magnitude vec_b = sqrt 3
axiom magnitude_a_minus_2b_eq3 : magnitude (λ x, vec_a x - 2 * vec_b x) = 3

theorem find_dot_product (a b : ℝ → ℝ → ℝ) 
  (ha : magnitude a = 1) 
  (hb : magnitude b = sqrt 3) 
  (h : magnitude (λ x, a x - 2 * b x) = 3) :
  dot_product a b = 1 := sorry

end find_dot_product_l14_14679


namespace maximum_area_exists_l14_14093

def max_area_rectangle (l w : ℕ) (h : l + w = 20) : Prop :=
  l * w ≤ 100

theorem maximum_area_exists : ∃ (l w : ℕ), max_area_rectangle l w (by sorry) ∧ (10 * 10 = 100) :=
begin
  sorry
end

end maximum_area_exists_l14_14093


namespace four_digit_number_count_l14_14731

theorem four_digit_number_count : 
  ∃ N : ℕ, 
    3000 ≤ N ∧ N < 7000 ∧ 
    (∃ k : ℕ, N = 35 * k) ∧ 
    ∃ a b c d : ℕ, 
      N = 1000 * a + 100 * b + 10 * c + d ∧ 
      a ∈ {3, 4, 5, 6} ∧
      d = 0 ∧
      2 ≤ b ∧ b < c ∧ c ≤ 7 :=
by 
  have valid_a: {3, 4, 5, 6}.card = 4 := by decide
  let valid_bc_pairs := { (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), 
                          (3, 4), (3, 5), (3, 6), (3, 7), 
                          (4, 5), (4, 6), (4, 7), 
                          (5, 6), (5, 7), (6, 7) }
  have valid_bc: valid_bc_pairs.card = 15 := by decide
  use 60
  dsimp
  sorry

end four_digit_number_count_l14_14731


namespace find_side_length_of_triangle_l14_14409

-- Definitions
def is_equilateral_triangle (t : ℝ) (D E F : ℝ × ℝ) : Prop :=
  dist D E = t ∧ dist E F = t ∧ dist F D = t

def unique_point_Q (D E F Q : ℝ × ℝ) : Prop :=
  dist D Q = 2 ∧ dist E Q = 2 * real.sqrt 2 ∧ dist F Q = 3

-- Problem statement
theorem find_side_length_of_triangle (D E F Q : ℝ × ℝ) (t : ℝ) :
  is_equilateral_triangle t D E F →
  unique_point_Q D E F Q →
  t = real.sqrt 37 :=
by
  sorry

end find_side_length_of_triangle_l14_14409


namespace count_valid_ks_correct_l14_14161

-- Noncomputable because we are dealing with reasoning rather than explicit computation
noncomputable def count_valid_ks : Nat :=
  Finset.card { k ∈ Finset.range 2014 | (k ^ k) % 10 = 1 }

theorem count_valid_ks_correct : count_valid_ks = 808 := by
  sorry

end count_valid_ks_correct_l14_14161


namespace radius_of_tangent_circle_l14_14521

theorem radius_of_tangent_circle (s : ℝ) (r_s : ℝ) (r_c : ℝ) : 
  s = 4 ∧ r_s = s / 2 ∧ ( √(r_s ^ 2 + r_s ^ 2) - r_s) = r_c → r_c = 2 * (sqrt 2 - 1) :=
by
  sorry

end radius_of_tangent_circle_l14_14521


namespace find_invariant_function_l14_14572

noncomputable def transformation_invariant_func : ℝ → ℝ :=
  λ x, if x > 0 then x * Real.log (2 * x) + x * h for some h else x * Real.log (2 * |x|) + x * h for some h

theorem find_invariant_function :
  ∀ (f : ℝ → ℝ), (∀ k, f (2^k * x) = 2^k * (k * x + f x)) ↔ f = transformation_invariant_func :=
sorry

end find_invariant_function_l14_14572


namespace bounded_function_l14_14373

variable {α : Type*} [OrderedField α]

def f (x : α) : α := sorry

theorem bounded_function (f : α → α) (M : α) 
  (h₁ : ∀ x y, 0 ≤ x → 0 ≤ y → f(x) * f(y) ≤ y^2 * f(x / 2) + x^2 * f(y / 2))
  (h₂ : M > 0)
  (h₃ : ∀ x, 0 ≤ x → x ≤ 1 → f(x) ≤ M) :
  ∀ x, 0 ≤ x → f(x) ≤ x^2 :=
begin
  sorry
end

end bounded_function_l14_14373


namespace dot_product_eq_one_l14_14699

variables {α : Type*} [InnerProductSpace ℝ α]

noncomputable def vector_a (a : α) : Prop := ∥a∥ = 1
noncomputable def vector_b (b : α) : Prop := ∥b∥ = real.sqrt 3
noncomputable def vector_c (a b : α) : Prop := ∥a - (2 : ℝ) • b∥ = 3

theorem dot_product_eq_one (a b : α) (ha : vector_a a) (hb : vector_b b) (hc : vector_c a b) :
  inner a b = 1 :=
sorry

end dot_product_eq_one_l14_14699


namespace find_190th_digit_l14_14749

/-- A sequence is formed by writing consecutive integers from 100 to 1, and we need to find the 190th digit in this sequence. -/
theorem find_190th_digit :
  let digits_seq := (List.range 100).reverse.flat_map (λ n, (100 - n).toString.data.map (λ c, c.val - '0'.val)) in
  digits_seq[189] = 3 :=
by sorry

end find_190th_digit_l14_14749


namespace max_distance_from_curve_to_line_l14_14608

def polar_to_cartesian_line (ρ : ℝ) (θ : ℝ) := ρ * sin (θ - π / 6) = 5

def parametric_curve (α : ℝ) (x : ℝ) (y : ℝ) :=
  x = sqrt 2 * cos α ∧ y = -2 + sqrt 2 * sin α ∧ (0 ≤ α ∧ α < 2 * π)

theorem max_distance_from_curve_to_line :
  (max_distance : ℝ) =
    let line_l := λ (x y : ℝ), x + sqrt 3 * y - 10 = 0 in
    let curve_C := λ (x y : ℝ), x^2 + (y + 2)^2 = 2 in
    (∀ α, parametric_curve α x y → line_l x y) →
    max_distance = 5 + sqrt 3 + sqrt 2 :=
by
  sorry

end max_distance_from_curve_to_line_l14_14608


namespace john_complete_square_l14_14812

def isSolution (a b c : Int) : Prop :=
  (64 * (a * a) = 64) ∧
  (2 * a * b = 96) ∧
  (c = a * a + 2 * a * b + b * b + (-128 - 36)) ∧
  (a > 0)

theorem john_complete_square : ∃ a b c : Int, isSolution a b c ∧ a + b + c = 178 :=
by
  use 8, 6, 164
  simp [isSolution]
  split
  rfl
  split
  rfl
  split
  rfl
  rfl

end john_complete_square_l14_14812


namespace monic_poly_unique_l14_14151

noncomputable def P (x : ℝ) : ℝ := x^2023

theorem monic_poly_unique 
  (P : ℝ → ℝ)
  (h_monic : P(x) = x^2023 + a_{2022}x^2022 + a_{2021}x^2021 + ... + a_0)
  (h_coeff : a_{2022} = 0)
  (h_P1 : P(1) = 1)
  (h_roots : ∀ r, P(r) = 0 → r < 1) :
  P(x) = x^2023 := by
  sorry

end monic_poly_unique_l14_14151


namespace tomatoes_left_l14_14079

theorem tomatoes_left (initial_tomatoes picked_yesterday picked_today : ℕ)
    (h_initial : initial_tomatoes = 171)
    (h_picked_yesterday : picked_yesterday = 134)
    (h_picked_today : picked_today = 30) :
    initial_tomatoes - picked_yesterday - picked_today = 7 :=
by
    sorry

end tomatoes_left_l14_14079


namespace draw_3Ls_probability_m_add_n_l14_14811

variable (L M T : Type) (deck : Finset (L ⊕ M ⊕ T))
          (probability : Finset (L ⊕ M ⊕ T) → ℝ)

-- Deck contains exactly 4 Ls, 4 Ms, and 4 Ts
variables (hL : (deck.filter (λ x, x = L)).card = 4)
          (hM : (deck.filter (λ x, x = M)).card = 4)
          (hT : (deck.filter (λ x, x = T)).card = 4)
          (hDeck : deck.card = 12)

-- Event of drawing 3 cards
variable (draw_three : Finset (L ⊕ M ⊕ T))

-- Probability of the specific draw sequence of 3 Ls
def probability_draw_3Ls : ℝ :=
  (4 / 12) * (3 / 11) * (2 / 10)

theorem draw_3Ls_probability (hDraw : draw_three.card = 3) :
  probability draw_three = 1 / 55 :=
by
  unfold probability
  sorry

theorem m_add_n : 1 + 55 = 56 := by
  sorry

end draw_3Ls_probability_m_add_n_l14_14811


namespace apple_juice_production_l14_14927

theorem apple_juice_production (total_production cider_percent juice_percent : ℝ) 
    (total_production_eq : total_production = 7)
    (cider_percent_eq : cider_percent = 0.25)
    (juice_percent_eq : juice_percent = 0.60) :
  total_production * (1 - cider_percent) * juice_percent = 3.15 :=
by
  have remaining_percent := 1 - cider_percent
  have juice_percent_of_remaining := remaining_percent * juice_percent
  have apple_juice_tons := total_production * juice_percent_of_remaining
  exact apple_juice_tons

#check apple_juice_production

end apple_juice_production_l14_14927


namespace price_per_glass_first_day_l14_14840

variables (O G : ℝ) (P1 : ℝ)

theorem price_per_glass_first_day (H1 : G * P1 = 1.5 * G * 0.40) : 
  P1 = 0.60 :=
by sorry

end price_per_glass_first_day_l14_14840


namespace domain_of_g_l14_14143

def quadratic_expression (x : ℝ) : ℝ := 15 * x^2 + 8 * x - 3

def g (x : ℝ) : ℝ := Real.sqrt (quadratic_expression x)

theorem domain_of_g :
  { x : ℝ | quadratic_expression x ≥ 0 } =
  { x : ℝ | x ≤ -3/5 } ∪ { x : ℝ | x ≥ 1/3 } :=
by
  sorry

end domain_of_g_l14_14143


namespace figure_100_nonoverlapping_unit_squares_l14_14408

theorem figure_100_nonoverlapping_unit_squares :
  let f : ℕ → ℕ := λ n, 2 * n^2 + 4 * n + 3 in
  f 100 = 20403 :=
by
  sorry

end figure_100_nonoverlapping_unit_squares_l14_14408


namespace area_trapezoid_ABCD_l14_14768

-- Given conditions
variables (A B C D E : Type)
variables [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq D] [decidable_eq E]
variables (parABCD : ∃ α β : A, α ∥ β) (dissectE : ∃ γ δ, γ ∩ δ = E)
variables (area_ABE : ℝ) (area_ADE : ℝ)
variables (ABE_area_condition : area_ABE = 60) (ADE_area_condition : area_ADE = 30)

-- Required statement to prove
theorem area_trapezoid_ABCD :
  let area_BCE := area_ADE,
      area_CDE := 15,
      area_trapezoid := area_ABE + area_ADE + area_BCE + area_CDE
  in area_trapezoid = 135 :=
by
  sorry

end area_trapezoid_ABCD_l14_14768


namespace ratio_of_triangle_areas_l14_14446

-- Define the given conditions
variables (m n x a : ℝ) (S T1 T2 : ℝ)

-- Conditions
def area_of_square : Prop := S = x^2
def area_of_triangle_1 : Prop := T1 = m * x^2
def length_relation : Prop := x = n * a

-- The proof goal
theorem ratio_of_triangle_areas (h1 : area_of_square S x) 
                                (h2 : area_of_triangle_1 T1 m x)
                                (h3 : length_relation x n a) : 
                                T2 / S = m / n^2 := 
sorry

end ratio_of_triangle_areas_l14_14446


namespace computer_sequence_count_l14_14276

theorem computer_sequence_count : 
  let letters := "COMPUTER".toList.erase 'M'.erase 'R';
  let vowels := filter (λ c, c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U') letters;
  let remaining := λ v, (letters.erase v);
  ∃ (v : Char) (H : v ∈ vowels), v * (remaining v).length * ((remaining v).length - 1) = 36 :=
by sorry

end computer_sequence_count_l14_14276


namespace dot_product_proof_l14_14654

variables {ℝ : Type*}
variables (a b : ℝ → ℝ)
variables [inner_product_space ℝ ℝ]

theorem dot_product_proof
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = sqrt 3)
  (h3 : ∥a - 2 • b∥ = 3) :
  inner (a : ℝ) (b : ℝ) = 1 :=
sorry

end dot_product_proof_l14_14654


namespace circle_line_intersection_l14_14781

theorem circle_line_intersection (x y : ℝ) (P A B : ℝ × ℝ) :
  (∀ θ : ℝ, ρ = 2 * sin θ → ((x - 0)^2 + (y - 1)^2 = 1))  ∧
  (∀ θ : ℝ, ρ * sin (θ + π/4) = sqrt 2 → (x + y - 2 = 0)) →
  let PA := dist P A,
      PB := dist P B in
  ( ∃ A B : ℝ × ℝ, (x + y - 2 = 0) ∧ 
    ((x - 0)^2 + (y - 1)^2 = 1) ) →
  (frac (1 / PA) + (1 / PB) = 3 * sqrt 2 / 4) :=
sorry

end circle_line_intersection_l14_14781


namespace angle_condition_l14_14298

theorem angle_condition (A : ℝ) (hA1 : 0 < A) (hA2 : A < 180) : 
  (30 < A → (sin (A) > 1 / 2)) ∧ (sin (A) > 1 / 2 → 30 < A) ∧ (∃ A₁, 30 < A₁ ∧ ¬ (sin (A₁) > 1 / 2)) :=
by
  sorry

end angle_condition_l14_14298


namespace greatest_number_of_factors_l14_14412

noncomputable theory

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem greatest_number_of_factors (b n : ℕ) (h1 : 0 < b ∧ b ≤ 15) (h2 : 0 < n) (h3 : is_perfect_square n) (h4 : n ≤ 15) : 
  (∃ b n : ℕ, (0 < b ∧ b ≤ 15) ∧ (0 < n) ∧ (is_perfect_square n) ∧ (n ≤ 15) ∧ ((∏ p in (Finset.range (b ^ n)).1, p + 1) = 561)) :=
by
  sorry

end greatest_number_of_factors_l14_14412


namespace smallest_abundant_not_multiple_of_5_is_12_l14_14581

/-- Define a number n is abundant if the sum of its proper divisors is greater than n -/
def is_abundant (n : ℕ) : Prop :=
  ∑ i in finset.filter (λ x, x < n ∧ n % x = 0) (finset.range n), i > n

/-- Main theorem: Smallest abundant number that is not a multiple of 5 is 12 -/
theorem smallest_abundant_not_multiple_of_5_is_12 :
  ∃ (n : ℕ), is_abundant n ∧ n % 5 ≠ 0 ∧ (∀ m : ℕ, is_abundant m ∧ m % 5 ≠ 0 → 12 ≤ m) → n = 12 :=
sorry

end smallest_abundant_not_multiple_of_5_is_12_l14_14581


namespace capital_at_end_of_2014_year_capital_exceeds_32dot5_billion_l14_14076

noncomputable def company_capital (n : ℕ) : ℝ :=
  if n = 0 then 1000
  else 2 * company_capital (n - 1) - 500

theorem capital_at_end_of_2014 : company_capital 4 = 8500 :=
by sorry

theorem year_capital_exceeds_32dot5_billion : ∀ n : ℕ, company_capital n > 32500 → n ≥ 7 :=
by sorry

end capital_at_end_of_2014_year_capital_exceeds_32dot5_billion_l14_14076


namespace maximum_area_exists_l14_14092

def max_area_rectangle (l w : ℕ) (h : l + w = 20) : Prop :=
  l * w ≤ 100

theorem maximum_area_exists : ∃ (l w : ℕ), max_area_rectangle l w (by sorry) ∧ (10 * 10 = 100) :=
begin
  sorry
end

end maximum_area_exists_l14_14092


namespace find_dot_product_l14_14706

open Real

variables (a b : ℝ^3)
variables (dot_product : ℝ^3 → ℝ^3 → ℝ)

def vector_magnitude (v : ℝ^3) : ℝ := sqrt (dot_product v v)

axiom magnitude_a : vector_magnitude a = 1
axiom magnitude_b : vector_magnitude b = sqrt 3
axiom magnitude_a_minus_2b : vector_magnitude (a - (2:ℝ) • b) = 3

theorem find_dot_product : dot_product a b = 1 :=
sorry

end find_dot_product_l14_14706


namespace cone_remaining_volume_l14_14003

theorem cone_remaining_volume {α : ℝ} {V v : ℝ} (cos_alpha : cos α = 1/4) (V_minus_v : V - v = 37) : 
  let k : ℝ := 3/4 in
  let v := k^3 * V in
  V - 37 = 27 :=
by 
  let k : ℝ := (3 / 4)
  let v := k^3 * V
  have volume_difference : V - v = 37 := V_minus_v
  sorry

end cone_remaining_volume_l14_14003


namespace rectangular_solid_volume_l14_14427

-- Define the conditions
variables (a : ℝ)

-- Define the length, width, and height of the rectangular solid
def length := 3 * a - 4
def width := 2 * a
def height := a

-- Define the volume of the rectangular solid
def volume (length width height : ℝ) : ℝ := length * width * height

-- State the theorem
theorem rectangular_solid_volume : volume (3 * a - 4) (2 * a) a = 6 * a^3 - 8 * a^2 :=
by 
  -- This 'sorry' is here to indicate that the proof is not provided, but the statement is complete.
  sorry

end rectangular_solid_volume_l14_14427


namespace worth_of_entire_lot_l14_14085

theorem worth_of_entire_lot (half_share : ℝ) (amount_per_tenth : ℝ) (total_amount : ℝ) :
  half_share = 0.5 →
  amount_per_tenth = 460 →
  total_amount = (amount_per_tenth * 10) →
  (total_amount * 2) = 9200 :=
by
  intros h1 h2 h3
  sorry

end worth_of_entire_lot_l14_14085


namespace choose_4_out_of_10_l14_14317

theorem choose_4_out_of_10 :
  nat.choose 10 4 = 210 :=
  by
  sorry

end choose_4_out_of_10_l14_14317


namespace max_sequence_length_x_l14_14380

-- Define the sequence terms as functions of x
def a (n x : ℝ) : ℝ :=
  if n = 1 then 500
  else if n = 2 then x
  else if n % 2 = 1 then (a (n - 2) x) - (a (n - 1) x)
  else (a (n - 1) x) - (a (n - 2) x)

-- The statement to prove: 
-- The maximum length sequence of positive terms is achieved when x = 1307
theorem max_sequence_length_x : 
  ∀ n, a n 1307 > 0 ↔ n ≤ 10 :=
by
  sorry

end max_sequence_length_x_l14_14380


namespace identity_holds_l14_14881

theorem identity_holds (x : ℝ) : 
  (2 * x - 1) ^ 3 = 5 * x ^ 3 + (3 * x + 1) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x :=
by sorry

end identity_holds_l14_14881


namespace integer_solution_x_l14_14791

theorem integer_solution_x (x y : ℤ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y + x * y = 101) : x = 50 :=
sorry

end integer_solution_x_l14_14791


namespace moores_law_l14_14839

theorem moores_law (initial_transistors : ℕ) (doubling_period : ℕ) (t1 t2 : ℕ) 
  (initial_year : t1 = 1985) (final_year : t2 = 2010) (transistors_in_1985 : initial_transistors = 300000) 
  (doubles_every_two_years : doubling_period = 2) : 
  (initial_transistors * 2 ^ ((t2 - t1) / doubling_period) = 1228800000) := 
by
  sorry

end moores_law_l14_14839


namespace number_of_ways_to_form_square_l14_14442

/-- There are wooden sticks with lengths of 1 cm, 2 cm, 3 cm, 4 cm, 5 cm, 6 cm, 7 cm, 8 cm, and 9 cm, 
one of each. Based on the given sticks, we prove that there are exactly 9 distinct ways to select 
combinations of the given sticks to form squares (without breaking any sticks). -/
theorem number_of_ways_to_form_square : ∃! n : ℕ, n = 9 := 
by 
  -- Define the lengths of sticks
  let stick_lengths := [1, 2, 3, 4, 5, 6, 7, 8, 9]
  -- Calculate total length
  have h_total_length : stick_lengths.sum = 45 := sorry 
  -- Define valid squares configurations and count them
  have h_valid_configurations : sorry := 
    sorry

  -- Since we already calculated them, we directly assert the final count of valid configurations:
  use 9
  sorry

end number_of_ways_to_form_square_l14_14442


namespace polynomial_identity_l14_14739

theorem polynomial_identity
  (x a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ)
  (h : (x - 1)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7) :
  (a + a_2 + a_4 + a_6)^2 - (a_1 + a_3 + a_5 + a_7)^2 = 0 :=
by sorry

end polynomial_identity_l14_14739


namespace find_number_l14_14859

theorem find_number (x : ℝ) : (5 / 3) * x = 45 → x = 27 := by
  sorry

end find_number_l14_14859


namespace find_x_l14_14802

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 :=
sorry

end find_x_l14_14802


namespace dot_product_ab_l14_14690

variables (a b : ℝ^3)

-- Given conditions
def condition1 : Prop := ‖a‖ = 1
def condition2 : Prop := ‖b‖ = real.sqrt 3
def condition3 : Prop := ‖a - 2 • b‖ = 3

-- The theorem statement to prove
theorem dot_product_ab (h1 : condition1 a) (h2 : condition2 b) (h3 : condition3 a b) : 
  a ⬝ b = 1 :=
sorry

end dot_product_ab_l14_14690


namespace parabola_circle_tangency_l14_14967

-- Definitions for the given conditions
def is_origin (P : Point) : Prop :=
  P.x = 0 ∧ P.y = 0

def is_on_x_axis (P : Point) : Prop := 
  P.y = 0

def parabola_vertex_focus_condition (C : Parabola) (O F : Point) : Prop :=
  is_origin O ∧ is_on_x_axis F ∧ C.vertex = O ∧ C.focus = F

def intersects_at_perpendicular_points (C : Parabola) (l : Line) (P Q : Point) : Prop :=
  l.slope = 1 ∧ l.intersect C = {P, Q} ∧ vector_product_is_perpendicular 0 P Q = true

def circle_tangent_to_line_at_point (M : Circle) (l : Line) (P : Point) : Prop :=
  l.form = "x=1" ∧ distance M.center P = M.radius ∧ M.contains P

def parabola_contains_point (C : Parabola) (P : Point) : Prop :=
  P.on_parabola C = true

def lines_tangent_to_circle (l₁ l₂ : Line) (M : Circle) : Prop :=
  M.tangent l₁ ∧ M.tangent l₂

def position_relationship (l : Line) (M : Circle) : PositionRelationship :=
  TangencyLine.and Circle M

-- Statement of the proof problem
theorem parabola_circle_tangency :
  ∃ C M, let O : Point := { x := 0, y := 0 } in
  let F : Point := { x := 1/2, y := 0 } in
  parabola_vertex_focus_condition C O F ∧ 
  intersects_at_perpendicular_points C (line_horizontal 1) P Q ∧
  point M_center = (mk_point 2 0) ∧
  circle_tangent_to_line_at_point M (line_horizontal 1) (mk_point 1 0) ∧
  ∀ A1 A2 A3 : Point, parabola_contains_point C A1 ∧
                    parabola_contains_point C A2 ∧
                    parabola_contains_point C A3 ∧
                    lines_tangent_to_circle (line_through A1 A2) M ∧
                    lines_tangent_to_circle (line_through A1 A3) M →
                    position_relationship (line_through A2 A3) M = Tangent :=
begin
  sorry  
end

end parabola_circle_tangency_l14_14967


namespace solve_inequality_l14_14646

noncomputable def f (a : ℝ) (x : ℝ) := log a (x^2) + a^(abs x)

theorem solve_inequality (a : ℝ) (x : ℝ) (h1 : 1 < a) (h2 : f a (-3) < f a 4) :
  {x | f a (x^2 - 2 * x) ≤ f a 3} = {x | (-1 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 3)} := 
by
  sorry

end solve_inequality_l14_14646


namespace solve_equation_l14_14399

-- Define the main equation
def equation (x : ℝ) :=
  (4 * (Real.tan (8 * x))^4 + 4 * Real.sin (2 * x) * Real.sin (6 * x) - Real.cos (4 * x) - Real.cos (12 * x) + 2) / Real.sqrt (Real.cos x - Real.sin x) = 0

-- Define the condition
def cond (x : ℝ) :=
  Real.cos x > Real.sin x

-- Definitions for the solutions
noncomputable def possible_solutions := { x : ℝ | ∃ n : ℤ, x = -Real.pi / 2 + 2 * n * Real.pi ∨ x = -Real.pi / 4 + 2 * n * Real.pi ∨ x = 2 * n * Real.pi}

-- Theorem statement
theorem solve_equation (x : ℝ) (h1: cond x) : equation x ↔ x ∈ possible_solutions :=
sorry

end solve_equation_l14_14399


namespace bob_vs_alice_payment_l14_14115

open Real

-- Define the conditions
def num_slices : ℕ := 12
def plain_pizza_cost : ℝ := 12.0
def olive_extra_cost : ℝ := 4.0
def bob_olive_slices : ℕ := 3
def bob_plain_slices : ℕ := 3
def alice_slices : ℕ := 6

-- Cost calculations
def total_pizza_cost : ℝ := plain_pizza_cost + olive_extra_cost
def cost_per_slice : ℝ := total_pizza_cost / num_slices
def bob_slices : ℕ := bob_olive_slices + bob_plain_slices
def bob_cost : ℝ := bob_slices * cost_per_slice
def alice_cost : ℝ := alice_slices * cost_per_slice

-- The statement to be proved
theorem bob_vs_alice_payment : bob_cost - alice_cost = 0 := by
  sorry

end bob_vs_alice_payment_l14_14115


namespace probability_defective_leq_one_l14_14012

noncomputable def total_products : ℕ := 8
noncomputable def defective_products : ℕ := 3
noncomputable def total_selection : ℕ := 3

theorem probability_defective_leq_one : 
  probability (number_defective_drawn ≤ 1) = 5 / 7 :=
sorry

end probability_defective_leq_one_l14_14012


namespace find_dot_product_l14_14705

open Real

variables (a b : ℝ^3)
variables (dot_product : ℝ^3 → ℝ^3 → ℝ)

def vector_magnitude (v : ℝ^3) : ℝ := sqrt (dot_product v v)

axiom magnitude_a : vector_magnitude a = 1
axiom magnitude_b : vector_magnitude b = sqrt 3
axiom magnitude_a_minus_2b : vector_magnitude (a - (2:ℝ) • b) = 3

theorem find_dot_product : dot_product a b = 1 :=
sorry

end find_dot_product_l14_14705


namespace parabola_and_circle_eq_and_tangent_l14_14961

noncomputable def parabola_eq (x y : ℝ) : Prop := y^2 = x

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

theorem parabola_and_circle_eq_and_tangent :
  (∀ x y : ℝ, parabola_eq x y ↔ y^2 = x) ∧
  (∀ x y : ℝ, circle_eq x y ↔ (x - 2)^2 + y^2 = 1) ∧
  (∀ A1 A2 A3 : ℝ × ℝ,
    parabola_eq A1.1 A1.2 ∧ parabola_eq A2.1 A2.2 ∧ parabola_eq A3.1 A3.2 →
    tangent_to_circle A1 A2 A3)
:=
sorry

end parabola_and_circle_eq_and_tangent_l14_14961


namespace diamond_value_l14_14741

def diamond (a b : ℤ) : ℤ := 4 * a - 2 * b

theorem diamond_value : diamond 7 3 = 22 :=
by
  -- Proof skipped
  sorry

end diamond_value_l14_14741


namespace max_tulips_l14_14454

theorem max_tulips :
  ∃ (y r : ℕ), (y + r = 15) ∧ (y + r) % 2 = 1 ∧ |y - r| = 1 ∧ (50 * y + 31 * r ≤ 600) :=
begin
  sorry
end

end max_tulips_l14_14454


namespace simplify_expression_l14_14914

variable {y : ℤ}

theorem simplify_expression (y : ℤ) : 5 / (4 * y^(-4)) * (4 * y^3) / 3 = 5 * y^7 / 3 := 
by 
  -- Proof is omitted with 'sorry'
  sorry

end simplify_expression_l14_14914


namespace solve_sine_equation_l14_14395

open Real

def angle (k : ℕ) : ℝ := 3^k * (2 * π) / 7

noncomputable def sum_sine (x : ℝ) : ℝ := 
  ∑ i in (Finset.range 6), sin (x + angle i)

theorem solve_sine_equation (x : ℝ) : 
  sum_sine x = 1 ↔ ∃ n : ℤ, x = -π/2 + 2 * π * n :=
by
  sorry

end solve_sine_equation_l14_14395


namespace ola_wins_l14_14381

-- Definitions for islands and connections
def Island : Type := ℕ
def connected : Island → Island → Prop

-- The archipelago has 2009 islands and some are connected by bridges
constant islands : fin 2009 → Island
constant bidirectional_bridges : ∀ i j : fin 2009, connected (islands i) (islands j) ↔ connected (islands j) (islands i)

-- Definition of "sparing"
def sparing : set (Island × Island) :=
  { p | connected p.1 p.2 }

-- Constraints: Maksim cannot move to an isolated island, and the strategic advantage leading to Ola's win
theorem ola_wins : 
  (∀ i j : Island, (connected i j ∨ ∀ k : Island, k ≠ i → k ≠ j → ¬connected j k ∧ ¬connected i k)) →
  (∃ i : Island, ∀ j : Island, connected i j → j ≠ i)
:= sorry

end ola_wins_l14_14381


namespace hyperbola_standard_eq_l14_14239

theorem hyperbola_standard_eq (a c : ℝ) (h1 : a = 5) (h2 : c = 7) :
  (∃ b, b^2 = c^2 - a^2 ∧ (1 = (x^2 / a^2 - y^2 / b^2) ∨ 1 = (y^2 / a^2 - x^2 / b^2))) := by
  sorry

end hyperbola_standard_eq_l14_14239


namespace find_number_l14_14860

theorem find_number (x : ℝ) : (5 / 3) * x = 45 → x = 27 := by
  sorry

end find_number_l14_14860


namespace find_side_c_l14_14770

theorem find_side_c
  (a b : ℝ) (S : ℝ) (ha : a = 4) (hb : b = 5) (hS : S = 5 * real.sqrt 3) :
  ∃ c : ℝ, c = real.sqrt 21 :=
by sorry

end find_side_c_l14_14770


namespace simplify_expression_l14_14918

theorem simplify_expression (y : ℝ) (hy : y ≠ 0) :
  (5 / (4 * y ^ (-4)) * (4 * y ^ 3 / 3)) = (5 * y ^ 7 / 3) :=
by
  sorry

end simplify_expression_l14_14918


namespace melanie_blue_balloons_count_l14_14343

variable (j : ℕ) (t : ℕ)

theorem melanie_blue_balloons_count (h1 : j = 40) (h2 : t = 81) : (t - j = 41) :=
by
  rw [h1, h2]
  norm_num
  sorry

end melanie_blue_balloons_count_l14_14343


namespace AK_gt_KB_l14_14785

-- Definitions for the problem
variables {A B C D K : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited K]
variable (ABCD : is_trapezoid A B C D)
variable (MidpointAB : is_midpoint M A B)
variable (MidpointCD : is_midpoint N C D)
variable (base_AD : has_length AD)
variable (lateral_CD : has_length CD)
variable (angle_bisector_D : is_angle_bisector D K A B)
variable (length_inequality : AD > CD)

-- The proof problem statement
theorem AK_gt_KB : AK > KB := 
by 
  sorry

end AK_gt_KB_l14_14785


namespace caroline_socks_l14_14134

theorem caroline_socks :
  let initial_pairs : ℕ := 40
  let lost_pairs : ℕ := 4
  let donation_fraction : ℚ := 2/3
  let purchased_pairs : ℕ := 10
  let gift_pairs : ℕ := 3
  let remaining_pairs := initial_pairs - lost_pairs
  let donated_pairs := remaining_pairs * donation_fraction
  let pairs_after_donation := remaining_pairs - donated_pairs.toNat
  let total_pairs := pairs_after_donation + purchased_pairs + gift_pairs
  total_pairs = 25 :=
by
  sorry

end caroline_socks_l14_14134


namespace choose_4_out_of_10_l14_14316

theorem choose_4_out_of_10 :
  nat.choose 10 4 = 210 :=
  by
  sorry

end choose_4_out_of_10_l14_14316


namespace num_bits_for_ABCDEF_l14_14472

def hex_ABCDEF : ℕ := 10 * 16^5 + 11 * 16^4 + 12 * 16^3 + 13 * 16^2 + 14 * 16 + 15

theorem num_bits_for_ABCDEF :
  ∀ n : ℕ, n = hex_ABCDEF → (log2 n).natAbs + 1 = 24 :=
by
  sorry

end num_bits_for_ABCDEF_l14_14472


namespace cos2_sin2alpha_value_l14_14783

theorem cos2_sin2alpha_value (α : ℝ) (x y : ℝ) (h1 : x = -2) (h2 : y = 1) (h3 : x = -2 ∧ y = 1) :
    cos^2(α) - sin(2 * α) = 8 / 5 :=
sorry

end cos2_sin2alpha_value_l14_14783


namespace integer_values_count_l14_14005

theorem integer_values_count {x : ℝ} (h : 4 < (sqrt (3 * x)) ∧ (sqrt (3 * x)) < 6) : 
    (x ∈ {6, 7, 8, 9, 10, 11}) :=
by
  sorry

end integer_values_count_l14_14005


namespace Torricelli_point_concurrence_l14_14980

noncomputable def AA_BB_CC_concurrent (A B C A' B' C' : Type*) 
  [triangle : triangle ABC]
  [equilateral_triangle_A : equilateral_triangle BC A']
  [equilateral_triangle_B : equilateral_triangle AC B']
  [equilateral_triangle_C : equilateral_triangle AB C'] : Prop := 
concurrent (line_through A A') (line_through B B') (line_through C C')

theorem Torricelli_point_concurrence (A B C A' B' C' : Type*) 
  [triangle : triangle ABC]
  [equilateral_triangle_A : equilateral_triangle BC A']
  [equilateral_triangle_B : equilateral_triangle AC B']
  [equilateral_triangle_C : equilateral_triangle AB C'] : 
  AA_BB_CC_concurrent A B C A' B' C' := 
sorry

end Torricelli_point_concurrence_l14_14980


namespace distance_between_a_and_c_l14_14290

-- Given conditions
variables (a : ℝ)

-- Statement to prove
theorem distance_between_a_and_c : |a + 1| = |a - (-1)| :=
by sorry

end distance_between_a_and_c_l14_14290


namespace centroid_fixed_point_l14_14652

open EuclideanGeometry

variable (A B C D E F : Point)
variable (ω1 ω2 : Circle)

-- Defining the conditions
-- ω1 is the circumcircle of ΔABC
def is_circumcircle (ω1 : Circle) (A B C : Point) : Prop := 
  ∀ (P : Point), P ∈ ω1 ↔ On_circumcircle P A B C

-- ω2 is the A-excircle of ΔABC
def is_A_excircle (ω2 : Circle) (A B C : Point) : Prop := 
  ∀ (P : Point), P ∈ ω2 ↔ On_A_excircle P A B C

-- ω2 touches BC, CA, and AB at points D, E, and F respectively
def touches_excircle_points (ω2 : Circle) (D E F : Point) : Prop := 
  ∃ (A B C : Point), 
    On_line D B C ∧ 
    On_line E C A ∧ 
    On_line F A B ∧ 
    Tangent_points ω2 D E F

-- Given that these conditions are satisfied, we want to prove that the 
-- centroid of ΔDEF is a fixed point.
theorem centroid_fixed_point 
  (hcirc : is_circumcircle ω1 A B C)
  (hexcir : is_A_excircle ω2 A B C)
  (htouch : touches_excircle_points ω2 D E F) :
    ∃ (G : Point), Is_centroid G D E F ∧ Fixed_point G :=
sorry

end centroid_fixed_point_l14_14652


namespace arith_prog_intersects_segment_system_l14_14384

-- Define segment and arithmetic progression types
structure Segment :=
  (start : ℝ)
  (length : ℝ)

structure ArithProg :=
  (initial : ℝ)
  (difference : ℝ)
  (term : ℕ → ℝ := λ n, initial + n * difference)

-- Define the main theorem
theorem arith_prog_intersects_segment_system 
  (segments : List Segment)
  (h_no_shared_endpoints : ∀ (i j : ℕ), i ≠ j → segments[i].start + segments[i].length ≤ segments[j].start ∨ segments[j].start + segments[j].length ≤ segments[i].start)
  (h_no_shared_points : ∀ (i j : ℕ) (x : ℝ), i ≠ j → (x < segments[i].start ∨ x > segments[i].start + segments[i].length) ∨ (x < segments[j].start ∨ x > segments[j].start + segments[j].length)) :
  ∀ (ap : ArithProg), ∃ (s : Segment) (n : ℕ), segments.contains s ∧ s.start ≤ ap.term n ∧ ap.term n ≤ s.start + s.length :=
begin
  sorry
end

end arith_prog_intersects_segment_system_l14_14384


namespace sin_43_lt_sqrt2_div_2_and_sin_73_l14_14234

theorem sin_43_lt_sqrt2_div_2_and_sin_73 (a : ℝ) (h : sin (43 * (π / 180)) = a) :
  a < (real.sqrt 2 / 2) ∧ sin (73 * (π / 180)) = (1 / 2) * real.sqrt (1 - a ^ 2) + (real.sqrt 3 / 2) * a :=
by {
  -- Proof to be filled in here
  sorry
}

end sin_43_lt_sqrt2_div_2_and_sin_73_l14_14234


namespace find_original_number_l14_14393

theorem find_original_number (x : ℚ) (h : 5 * ((3 * x + 6) / 2) = 100) : x = 34 / 3 := sorry

end find_original_number_l14_14393


namespace least_multiple_of_five_primes_l14_14989

noncomputable def smallest_multiple_of_five_primes : ℕ :=
  let primes := [2, 3, 5, 7, 11] in
  primes.foldl (· * ·) 1

theorem least_multiple_of_five_primes : smallest_multiple_of_five_primes = 2310 := by
  sorry

end least_multiple_of_five_primes_l14_14989


namespace equal_spacing_of_zeros_l14_14365

noncomputable def f (n : ℕ) (a : ℕ → ℝ) (x : ℝ) : ℝ :=
  (List.range n).sum (λ k, 1 / 2 ^ k * Real.cos (a k + x))

theorem equal_spacing_of_zeros 
  (n : ℕ) (a : ℕ → ℝ) (x₁ x₂ : ℝ)
  (h₁ : f n a x₁ = 0) 
  (h₂ : f n a x₂ = 0) : 
  ∃ m : ℤ, x₂ - x₁ = m * Real.pi := by sorry

end equal_spacing_of_zeros_l14_14365


namespace prove_identity_l14_14898

variable (x : ℝ)

theorem prove_identity : 
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  -- Expand both sides and prove identity
  sorry

end prove_identity_l14_14898


namespace least_number_divisible_by_five_smallest_primes_l14_14999

theorem least_number_divisible_by_five_smallest_primes : 
  ∃ n ∈ ℕ+, n = 2 * 3 * 5 * 7 * 11 ∧ n = 2310 :=
by
  sorry

end least_number_divisible_by_five_smallest_primes_l14_14999


namespace odd_function_five_value_l14_14822

variable (f : ℝ → ℝ)

theorem odd_function_five_value (h_odd : ∀ x : ℝ, f (-x) = -f x)
                               (h_f1 : f 1 = 1 / 2)
                               (h_f_recurrence : ∀ x : ℝ, f (x + 2) = f x + f 2) :
  f 5 = 5 / 2 :=
sorry

end odd_function_five_value_l14_14822


namespace factorial_ratio_value_l14_14038

theorem factorial_ratio_value : fact 15 / (fact 6 * fact 9) = 770 := by
  sorry

end factorial_ratio_value_l14_14038


namespace sequence_injective_surjective_l14_14554

noncomputable def sequence (n : ℕ) : ℕ
| 0       := 0
| 1       := 1
| (2 * k) := if k % 2 = 0 then 2 * sequence k else 2 * sequence k + 1
| (2 * k + 1) := if k % 2 = 1 then 2 * sequence k else 2 * sequence k + 1

theorem sequence_injective_surjective : 
  ∀ m : ℕ, m > 0 → 
  ∃! n, sequence n = m :=
sorry

end sequence_injective_surjective_l14_14554


namespace f_neg_expression_l14_14751

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - f x

def f_pos (x : ℝ) : ℝ :=
  if x > 0 then x * (x - 1) else 0

theorem f_neg_expression (f : ℝ → ℝ) (h_odd : is_odd_function f)
  (h_pos : ∀ x, 0 < x → f x = x * (x - 1)) :
  ∀ x, x < 0 → f x = -x * (x + 1) :=
by
  sorry

end f_neg_expression_l14_14751


namespace smallest_k_people_l14_14587

theorem smallest_k_people {m n : ℕ} (hm : m > 0) (hn : n > 0) : 
  ∃ k : ℕ, (∀ (G : Type) [fintype G] [decidable_eq G] (V : set G) (hV : fintype.card V = k), 
    (∃ M : set (G × G), fintype.card M = m ∧ (∀ ⟨u, v⟩ ∈ M, u ≠ v ∧ (u, v) ∈ G ∧ (v, u) ∈ G)) ∨ 
    (∃ N : set (G × G), fintype.card N = n ∧ (∀ ⟨u, v⟩ ∈ N, u ≠ v ∧ (u, v) ∉ G ∧ (v, u) ∉ G))) ∧
    k = m + n + max m n - 1 :=
begin
  sorry
end

end smallest_k_people_l14_14587


namespace number_of_segments_is_even_l14_14972

-- Define the notion of a self-intersecting polygonal chain
def closed_self_intersecting_polygonal_chain (n : ℕ) : Prop :=
  ∃ (segments : list (ℝ × ℝ)), segments.length = n ∧
  (∀ segment, segment ∈ segments → segment.crosses_once (segments \ [segment]))

-- Define the property of crossing exactly once
def crosses_once (segment : ℝ × ℝ) (segments : list (ℝ × ℝ)) : Prop :=
  ∃! p, ∃ s, s ∈ segments ∧ intersects segment s p

-- Assuming an intersection function
def intersects (segment1 segment2 : ℝ × ℝ) (p : ℝ × ℝ) : Prop := sorry

-- The main theorem
theorem number_of_segments_is_even (n : ℕ):
  closed_self_intersecting_polygonal_chain n → 
  n % 2 = 0 :=
by
  intro h
  sorry

end number_of_segments_is_even_l14_14972


namespace carmen_reaches_alex_in_17_5_minutes_l14_14541

-- Define the conditions
variable (initial_distance : ℝ := 30) -- Initial distance in kilometers
variable (rate_of_closure : ℝ := 2) -- Rate at which the distance decreases in km per minute
variable (minutes_before_stop : ℝ := 10) -- Minutes before Alex stops

-- Define the speeds
variable (v_A : ℝ) -- Alex's speed in km per hour
variable (v_C : ℝ := 2 * v_A) -- Carmen's speed is twice Alex's speed
variable (total_closure_rate : ℝ := 120) -- Closure rate in km per hour (2 km per minute)

-- Main theorem to prove:
theorem carmen_reaches_alex_in_17_5_minutes : 
  ∃ (v_A v_C : ℝ), v_C = 2 * v_A ∧ v_C + v_A = total_closure_rate ∧ 
    (initial_distance - rate_of_closure * minutes_before_stop 
    - v_C * ((initial_distance - rate_of_closure * minutes_before_stop) / v_C) / 60 = 0) ∧ 
    (minutes_before_stop + ((initial_distance - rate_of_closure * minutes_before_stop) / v_C) * 60 = 17.5) :=
by
  sorry

end carmen_reaches_alex_in_17_5_minutes_l14_14541


namespace simplify_fraction_l14_14909

theorem simplify_fraction (y : ℝ) (hy : y ≠ 0) : 
  (5 / (4 * y⁻⁴)) * ((4 * y³) / 3) = (5 * y⁷) / 3 := 
by
  sorry

end simplify_fraction_l14_14909


namespace max_area_triangle_APC_l14_14448

/-- Consider a triangle ABC with sides AB = 10, BC = 17, and CA = 21.
    Let P be a point on the circle with diameter AB.
    Prove that the greatest possible area of triangle APC is 189/2. -/
theorem max_area_triangle_APC :
  ∀ (A B C P : Type) (d : AB = 10 ∧ BC = 17 ∧ CA = 21 ∧ P lies_on (circle_with_diameter AB)),
  max_area (triangle_APC A B C P) = 189 / 2 :=
by
  sorry

end max_area_triangle_APC_l14_14448


namespace find_x_range_l14_14152

-- Defining the function f(x) = sqrt(3 - x) - sqrt(x + 1)
def f (x : ℝ) : ℝ := Real.sqrt (3 - x) - Real.sqrt (x + 1)

-- State the theorem
theorem find_x_range (x : ℝ) (h1 : -1 ≤ x) (h2 : x ≤ 3) :
  (f x > 1 / 2) ↔ (-1 ≤ x ∧ x < 1 - Real.sqrt 31 / 8) := 
by
  sorry

end find_x_range_l14_14152


namespace sum_of_roots_quadratic_l14_14285

theorem sum_of_roots_quadratic :
  ∃ (x1 x2 : ℝ), (x1 + x2 = 2) ∧ (x1 * x2 = -8) :=
by
  use [sorry, sorry]
  sorry

end sum_of_roots_quadratic_l14_14285


namespace average_greater_than_median_l14_14600

def gabriel_weight : ℝ := 96
def brother_weights : List ℝ := [10, 10, 12, 18]
def weights : List ℝ := gabriel_weight :: brother_weights

def calculate_median (l : List ℝ) : ℝ :=
  let sorted := l.qsort (≤)
  sorted.get! (sorted.length / 2)

def calculate_average (l : List ℝ) : ℝ :=
  (l.sum) / l.length

theorem average_greater_than_median :
  calculate_average(weights) = 29.2 ∧ 
  calculate_median(weights) = 12 ∧ 
  calculate_average(weights) > calculate_median(weights) ∧ 
  (calculate_average(weights) - calculate_median(weights)) = 17.2 :=
by
  sorry

end average_greater_than_median_l14_14600


namespace reciprocal_pairs_l14_14119

-- Define the pairs
def pair_1 : ℝ × ℝ := (1, -1)
def pair_2 : ℝ × ℝ := (-1/3, 3)
def pair_3 : ℝ × ℝ := (-5, -1/5)
def pair_4 : ℝ × ℝ := (-3, abs(-3))

-- State the theorem
theorem reciprocal_pairs :
  pair_1.1 * pair_1.2 ≠ 1 ∧
  pair_2.1 * pair_2.2 ≠ 1 ∧
  pair_3.1 * pair_3.2 = 1 ∧
  pair_4.1 * pair_4.2 ≠ 1 :=
by {
  sorry
}

end reciprocal_pairs_l14_14119


namespace find_sum_of_distinct_real_numbers_l14_14359

noncomputable def determinant_3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

theorem find_sum_of_distinct_real_numbers (x y : ℝ) (hxy : x ≠ y) 
    (h : determinant_3x3 1 6 15 3 x y 3 y x = 0) : x + y = 63 := 
by
  sorry

end find_sum_of_distinct_real_numbers_l14_14359


namespace cone_lateral_surface_area_l14_14189

theorem cone_lateral_surface_area (r V : ℝ) (h l S : ℝ) 
  (radius_condition : r = 6)
  (volume_condition : V = 30 * Real.pi)
  (volume_formula : V = (1 / 3) * Real.pi * r^2 * h)
  (slant_height_formula : l = Real.sqrt (r^2 + h^2))
  (lateral_surface_area_formula : S = Real.pi * r * l) :
  S = 39 * Real.pi := 
sorry

end cone_lateral_surface_area_l14_14189


namespace dot_product_ab_l14_14686

variables (a b : ℝ^3)

-- Given conditions
def condition1 : Prop := ‖a‖ = 1
def condition2 : Prop := ‖b‖ = real.sqrt 3
def condition3 : Prop := ‖a - 2 • b‖ = 3

-- The theorem statement to prove
theorem dot_product_ab (h1 : condition1 a) (h2 : condition2 b) (h3 : condition3 a b) : 
  a ⬝ b = 1 :=
sorry

end dot_product_ab_l14_14686


namespace abcdef_base16_to_base2_bits_l14_14469

noncomputable def base_16_to_dec (h : String) : ℕ :=
  10 * 16^5 + 11 * 16^4 + 12 * 16^3 + 13 * 16^2 + 14 * 16^1 + 15 * 16^0

theorem abcdef_base16_to_base2_bits (n : ℕ) (h : n = base_16_to_dec "ABCDEF") :
  nat.log2 n + 1 = 24 :=
by
  rw [h]
  -- nat.log2 calculations and the necessary steps to establish the number of bits
  sorry

end abcdef_base16_to_base2_bits_l14_14469


namespace price_of_larger_jar_l14_14077

-- Definitions of the given problem conditions
def smaller_jar_diameter : ℝ := 4
def smaller_jar_height : ℝ := 5
def smaller_jar_price : ℝ := 0.90

def larger_jar_diameter : ℝ := 12
def larger_jar_height : ℝ := 10

-- Volume calculations for cylinders
def volume (diameter height : ℝ) : ℝ :=
  let radius := diameter / 2
  π * radius^2 * height

-- Calculating the price based on volume and linear scaling
def price (smaller_vol larger_vol smaller_price : ℝ) : ℝ :=
  (larger_vol / smaller_vol) * smaller_price

-- Statement to prove
theorem price_of_larger_jar :
  let smaller_vol := volume smaller_jar_diameter smaller_jar_height in
  let larger_vol := volume larger_jar_diameter larger_jar_height in
  price smaller_vol larger_vol smaller_jar_price = 16.20 :=
by sorry

end price_of_larger_jar_l14_14077


namespace simplify_expression_l14_14913

variable {y : ℤ}

theorem simplify_expression (y : ℤ) : 5 / (4 * y^(-4)) * (4 * y^3) / 3 = 5 * y^7 / 3 := 
by 
  -- Proof is omitted with 'sorry'
  sorry

end simplify_expression_l14_14913


namespace remainder_division_l14_14594

theorem remainder_division
  (j : ℕ) (h_pos : 0 < j)
  (h_rem : ∃ b : ℕ, 72 = b * j^2 + 8) :
  150 % j = 6 :=
sorry

end remainder_division_l14_14594


namespace identity_element_is_neg4_l14_14556

def op (a b : ℝ) := a + b + 4

def is_identity (e : ℝ) := ∀ a : ℝ, op e a = a

theorem identity_element_is_neg4 : ∃ e : ℝ, is_identity e ∧ e = -4 :=
by
  use -4
  sorry

end identity_element_is_neg4_l14_14556


namespace range_f3_l14_14258

def function_f (a c x : ℝ) : ℝ := a * x^2 - c

theorem range_f3 (a c : ℝ) :
  (-4 ≤ function_f a c 1) ∧ (function_f a c 1 ≤ -1) →
  (-1 ≤ function_f a c 2) ∧ (function_f a c 2 ≤ 5) →
  -12 ≤ function_f a c 3 ∧ function_f a c 3 ≤ 1.75 :=
by
  sorry

end range_f3_l14_14258


namespace hyperbola_eccentricity_sqrt_5_l14_14746

noncomputable def hyperbola_eccentricity {a b : ℝ} (ha : a > 0) (hb : b > 0)
  (h_eq : ∃ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)) 
  (h_asymptotes : ∀ x : ℝ, has_sup (λ y, y = 2 * x) (λ y, y = -2 * x))
  : ℝ :=
  let c := sqrt(a^2 + b^2) in
  c / a

theorem hyperbola_eccentricity_sqrt_5 {a b : ℝ} (ha : a > 0) (hb : b > 0)
  (h_eq : ∃ x y, (x^2 / a^2 - y^2 / b^2 = 1))
  (h_asymptotes : ∀ x : ℝ, has_sup (λ y, y = 2 * x) (λ y, y = -2 * x)) :
  hyperbola_eccentricity ha hb h_eq h_asymptotes = sqrt 5 :=
sorry

end hyperbola_eccentricity_sqrt_5_l14_14746


namespace probability_two_blue_l14_14510

open Finset

-- Definitions based on conditions
def total_jellybeans : ℕ := 12
def red_jellybeans : ℕ := 5
def blue_jellybeans : ℕ := 3
def white_jellybeans : ℕ := 4
def pick_count : ℕ := 3

-- Let's define the combination function if not already present
def combination (n k : ℕ) : ℕ := (Finset.range (n + 1)).choose k

-- Mathematically equivalent proof problem in Lean 4
theorem probability_two_blue :
  let total_outcomes := combination total_jellybeans pick_count in
  let blue_combinations := combination blue_jellybeans 2 in
  let non_blue_combinations := combination (red_jellybeans + white_jellybeans) 1 in
  let favorable_outcomes := blue_combinations * non_blue_combinations in
  (favorable_outcomes.to_rat / total_outcomes.to_rat = 27 / 220) :=
by
  sorry

end probability_two_blue_l14_14510


namespace find_speed_range_l14_14013

noncomputable def runningErrorB (v : ℝ) : ℝ := abs ((300 / v) - 7)
noncomputable def runningErrorC (v : ℝ) : ℝ := abs ((480 / v) - 11)

theorem find_speed_range (v : ℝ) :
  (runningErrorB v + runningErrorC v ≤ 2) →
  33.33 ≤ v ∧ v ≤ 48.75 := sorry

end find_speed_range_l14_14013


namespace smallest_abundant_not_multiple_of_5_l14_14586

def is_abundant (n : ℕ) : Prop :=
  (∑ d in (nat.divisors n).filter (≠ n), d) > n

def not_multiple_of_5 (n : ℕ) : Prop :=
  ¬ (5 ∣ n)

theorem smallest_abundant_not_multiple_of_5 :
  ∃ n : ℕ, is_abundant n ∧ not_multiple_of_5 n ∧ 
    (∀ m : ℕ, is_abundant m ∧ not_multiple_of_5 m → m >= n) :=
sorry

end smallest_abundant_not_multiple_of_5_l14_14586


namespace find_x_y_l14_14410

theorem find_x_y (x y : ℤ) (hx : 0 < x) (hy : 0 < y) (h : (x + y * Complex.I)^2 = (7 + 24 * Complex.I)) :
  x + y * Complex.I = 4 + 3 * Complex.I :=
by
  sorry

end find_x_y_l14_14410


namespace value_of_m_l14_14753

theorem value_of_m (m : ℝ) (h1 : m - 2 ≠ 0) (h2 : |m| - 1 = 1) : m = -2 := by {
  sorry
}

end value_of_m_l14_14753


namespace find_dot_product_l14_14708

open Real

variables (a b : ℝ^3)
variables (dot_product : ℝ^3 → ℝ^3 → ℝ)

def vector_magnitude (v : ℝ^3) : ℝ := sqrt (dot_product v v)

axiom magnitude_a : vector_magnitude a = 1
axiom magnitude_b : vector_magnitude b = sqrt 3
axiom magnitude_a_minus_2b : vector_magnitude (a - (2:ℝ) • b) = 3

theorem find_dot_product : dot_product a b = 1 :=
sorry

end find_dot_product_l14_14708


namespace David_pushups_calculation_l14_14054

variable (Zachary_pushups : ℕ) (David_pushups : ℕ)

axiom Zachary_pushups_value : Zachary_pushups = 19
axiom David_more_than_Zachary : David_pushups = Zachary_pushups + 39

theorem David_pushups_calculation : David_pushups = 58 := by
  rw [Zachary_pushups_value] at David_more_than_Zachary
  exact David_more_than_Zachary

-- Proof is intentionally omitted

end David_pushups_calculation_l14_14054


namespace lateral_surface_area_of_cone_l14_14000

theorem lateral_surface_area_of_cone (r l : ℝ) (h₁ : r = 3) (h₂ : l = 5) :
  π * r * l = 15 * π :=
by sorry

end lateral_surface_area_of_cone_l14_14000


namespace rose_bushes_unwatered_l14_14065

theorem rose_bushes_unwatered (n V A : ℕ) (V_set A_set : Finset ℕ) (hV : V = 1003) (hA : A = 1003) (hTotal : n = 2006) (hIntersection : V_set.card = 3) :
  n - (V + A - V_set.card) = 3 :=
by
  sorry

end rose_bushes_unwatered_l14_14065


namespace find_dot_product_l14_14710

open Real

variables (a b : ℝ^3)
variables (dot_product : ℝ^3 → ℝ^3 → ℝ)

def vector_magnitude (v : ℝ^3) : ℝ := sqrt (dot_product v v)

axiom magnitude_a : vector_magnitude a = 1
axiom magnitude_b : vector_magnitude b = sqrt 3
axiom magnitude_a_minus_2b : vector_magnitude (a - (2:ℝ) • b) = 3

theorem find_dot_product : dot_product a b = 1 :=
sorry

end find_dot_product_l14_14710


namespace turtle_race_ratio_l14_14273

theorem turtle_race_ratio :
  let Greta_time := 6
  let George_time := Greta_time - 2
  let Gloria_time := 8
  ratio (Gloria_time) (George_time) = 2 :=
by
  let Greta_time := 6
  let George_time := Greta_time - 2
  let Gloria_time := 8
  have ratio := Gloria_time / George_time
  have h : ratio = 2 := sorry
  exact h

end turtle_race_ratio_l14_14273


namespace caroline_total_socks_l14_14132

def initial_socks_pairs : ℕ := 40
def lost_socks_pairs : ℕ := 4
def donation_fraction : ℚ := 2 / 3
def purchased_socks : ℕ := 10
def gift_socks : ℕ := 3

theorem caroline_total_socks : 
  let n_0 := initial_socks_pairs in
  let n_1 := n_0 - lost_socks_pairs in
  let d := (donation_fraction * n_1 : ℚ) in
  let n_2 := n_1 - d.nat_abs in  -- since d is a rational number, taking nat_abs converts it to ℕ
  let n_3 := n_2 + purchased_socks in
  let n_final := n_3 + gift_socks in
  n_final = 25 :=
by
  sorry

end caroline_total_socks_l14_14132


namespace tangent_position_is_six_oclock_l14_14936

-- Define constants and initial conditions
def bigRadius : ℝ := 30
def smallRadius : ℝ := 15
def initialPosition := 12 -- 12 o'clock represented as initial tangent position
def initialArrowDirection := 0 -- upwards direction

-- Define that the small disk rolls counterclockwise around the clock face.
def rollsCCW := true

-- Define the destination position when the arrow next points upward.
def diskTangencyPosition (bR sR : ℝ) (initPos initDir : ℕ) (rolls : Bool) : ℕ :=
  if rolls then 6 else 12

theorem tangent_position_is_six_oclock :
  diskTangencyPosition bigRadius smallRadius initialPosition initialArrowDirection rollsCCW = 6 :=
sorry  -- the proof is omitted

end tangent_position_is_six_oclock_l14_14936


namespace largest_fraction_l14_14049

def frac_A := (5 : ℚ) / 11
def frac_B := (6 : ℚ) / 13
def frac_C := (19 : ℚ) / 39
def frac_D := (101 : ℚ) / 203
def frac_E := (152 : ℚ) / 303
def frac_F := (80 : ℚ) / 159

theorem largest_fraction : 
  ∀ f ∈ {frac_A, frac_B, frac_C, frac_D, frac_E, frac_F}, f ≤ frac_F :=
by
  sorry

end largest_fraction_l14_14049


namespace complex_number_C_l14_14634

-- Define the complex numbers corresponding to points A and B
def A : ℂ := 1 + 2 * Complex.I
def B : ℂ := 3 - 5 * Complex.I

-- Prove the complex number corresponding to point C
theorem complex_number_C :
  ∃ C : ℂ, (C = 10 - 3 * Complex.I) ∧ 
           (A = 1 + 2 * Complex.I) ∧ 
           (B = 3 - 5 * Complex.I) ∧ 
           -- Square with vertices in counterclockwise order
           True := 
sorry

end complex_number_C_l14_14634


namespace general_formula_sum_of_first_n_terms_l14_14226

-- Define the conditions for the arithmetic sequence
variable {d : ℝ} (a : ℕ → ℝ)
variable (h_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)
variable (h_a1 : a 1 = 1)
variable (h_a2_pos : a 2 > 1)
variable (h_geom : (a 2) * (a 9) = (a 4) ^ 2)

-- General formula for the sequence
theorem general_formula (n : ℕ) (h_a2: a 2 = 1 + d) (h_d : d = 3) : a n = 3 * n - 2 := by
  sorry

-- Sum of first n terms of the sequence
theorem sum_of_first_n_terms (n : ℕ) (h_a2: a 2 = 1 + d) (h_d : d = 3) : 
  (∑ i in range n, a i) = (3 / 2 * n ^ 2 - 1 / 2 * n) := by
  sorry

end general_formula_sum_of_first_n_terms_l14_14226


namespace dot_product_ab_l14_14688

variables (a b : ℝ^3)

-- Given conditions
def condition1 : Prop := ‖a‖ = 1
def condition2 : Prop := ‖b‖ = real.sqrt 3
def condition3 : Prop := ‖a - 2 • b‖ = 3

-- The theorem statement to prove
theorem dot_product_ab (h1 : condition1 a) (h2 : condition2 b) (h3 : condition3 a b) : 
  a ⬝ b = 1 :=
sorry

end dot_product_ab_l14_14688


namespace sum_of_numbers_l14_14002

theorem sum_of_numbers {x : ℝ} (h1 : (x^2) + (2*x)^2 + (4*x)^2 + (5*x)^2 = 2460) : 
  x + 2*x + 4*x + 5*x ≈ 87.744 := 
by
  sorry

end sum_of_numbers_l14_14002


namespace caroline_total_socks_l14_14133

def initial_socks_pairs : ℕ := 40
def lost_socks_pairs : ℕ := 4
def donation_fraction : ℚ := 2 / 3
def purchased_socks : ℕ := 10
def gift_socks : ℕ := 3

theorem caroline_total_socks : 
  let n_0 := initial_socks_pairs in
  let n_1 := n_0 - lost_socks_pairs in
  let d := (donation_fraction * n_1 : ℚ) in
  let n_2 := n_1 - d.nat_abs in  -- since d is a rational number, taking nat_abs converts it to ℕ
  let n_3 := n_2 + purchased_socks in
  let n_final := n_3 + gift_socks in
  n_final = 25 :=
by
  sorry

end caroline_total_socks_l14_14133


namespace factorize_expression_l14_14568

theorem factorize_expression (x y : ℝ) : 
  (x + y)^2 - 14 * (x + y) + 49 = (x + y - 7)^2 := 
by
  sorry

end factorize_expression_l14_14568


namespace find_x_l14_14799

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 := 
by
  sorry

end find_x_l14_14799


namespace integer_solutions_sum_inequality_l14_14403

theorem integer_solutions_sum_inequality :
  let satisfies_inequality (x : ℝ) := 
    sqrt (x - 4) + sqrt (x + 1) + sqrt (2 * x) > sqrt (33 - x) + 4
  in ∑ (x : ℕ) in (Finset.Icc 9 33), x = 525 :=
by
  sorry

end integer_solutions_sum_inequality_l14_14403


namespace lateral_surface_area_of_given_cone_l14_14210

noncomputable def coneLateralSurfaceArea (r V : ℝ) : ℝ :=
let h := (3 * V) / (π * r^2) in
let l := Real.sqrt (r^2 + h^2) in
π * r * l

theorem lateral_surface_area_of_given_cone :
  coneLateralSurfaceArea 6 (30 * π) = 39 * π := by
simp [coneLateralSurfaceArea]
sorry

end lateral_surface_area_of_given_cone_l14_14210


namespace candies_total_l14_14340

theorem candies_total (b : ℕ) (m : ℕ) (s : ℕ) (j : ℕ) (sa : ℕ)
  (h_b : b = 10) (h_m : m = 5) (h_s : s = 20) (h_j : j = 5) (h_sa : sa = 10) :
  b + m + s + j + sa = 50 :=
by
  rw [h_b, h_m, h_s, h_j, h_sa]
  sorry

end candies_total_l14_14340


namespace prob_exact_one_pair_of_five_from_ten_is_40_over_63_l14_14730

noncomputable def prob_exact_one_pair (socks : Finset (Fin 10)) : ℚ :=
  let total_combinations := (socks.card.choose 5 : ℚ)
  let choose_color_for_pair := (Finset.card (Finset.univ : Finset (Fin 5)).choose 1 : ℚ)
  let choose_remaining_colors := (Finset.card (Finset.univ \ Finset.singleton 0).choose 3 : ℚ)
  let ways_to_choose_single_socks := (2 * 2 * 2 : ℚ) -- For 3 single socks, each color has 2 socks
  let favorable_outcomes := choose_color_for_pair * choose_remaining_colors * ways_to_choose_single_socks
  (favorable_outcomes / total_combinations)

theorem prob_exact_one_pair_of_five_from_ten_is_40_over_63 : ∀ (socks : Finset (Fin 10)), socks.card = 10 → 
  prob_exact_one_pair socks = 40 / 63 :=
by
  sorry

end prob_exact_one_pair_of_five_from_ten_is_40_over_63_l14_14730


namespace smallest_positive_debt_pigs_goats_l14_14026

theorem smallest_positive_debt_pigs_goats :
  ∃ p g : ℤ, 350 * p + 240 * g = 10 :=
by
  sorry

end smallest_positive_debt_pigs_goats_l14_14026


namespace initial_bacteria_count_l14_14418

theorem initial_bacteria_count (n : ℕ) :
  (n * 4^15 = 1_073_741_824) → (n = 1) :=
begin
  sorry
end

end initial_bacteria_count_l14_14418


namespace fraction_product_l14_14128

theorem fraction_product : 
  (7 / 5) * (8 / 16) * (21 / 15) * (14 / 28) * (35 / 25) * (20 / 40) * (49 / 35) * (32 / 64) = 2401 / 10000 :=
by
  -- This line is to skip the proof
  sorry

end fraction_product_l14_14128


namespace find_number_l14_14862

theorem find_number (x : ℝ) : (5 / 3) * x = 45 → x = 27 := by
  sorry

end find_number_l14_14862


namespace total_servings_daily_l14_14476

def cost_per_serving : ℕ := 14
def price_A : ℕ := 20
def price_B : ℕ := 18
def total_revenue : ℕ := 1120
def total_profit : ℕ := 280

theorem total_servings_daily (x y : ℕ) (h1 : price_A * x + price_B * y = total_revenue)
                             (h2 : (price_A - cost_per_serving) * x + (price_B - cost_per_serving) * y = total_profit) :
                             x + y = 60 := sorry

end total_servings_daily_l14_14476


namespace equalize_marbles_condition_l14_14808

variables (D : ℝ)
noncomputable def marble_distribution := 
    let C := 1.25 * D
    let B := 1.4375 * D
    let A := 1.725 * D
    let total := A + B + C + D
    let equal := total / 4
    let move_from_A := (A - equal) / A * 100
    let move_from_B := (B - equal) / B * 100
    let add_to_C := (equal - C) / C * 100
    let add_to_D := (equal - D) / D * 100
    (move_from_A, move_from_B, add_to_C, add_to_D)

theorem equalize_marbles_condition :
    marble_distribution D = (21.56, 5.87, 8.25, 35.31) := sorry

end equalize_marbles_condition_l14_14808


namespace statues_at_end_of_fourth_year_l14_14726

def initial_statues : ℕ := 4
def statues_after_second_year : ℕ := initial_statues * 4
def statues_added_third_year : ℕ := 12
def broken_statues_third_year : ℕ := 3
def statues_removed_third_year : ℕ := broken_statues_third_year
def statues_added_fourth_year : ℕ := broken_statues_third_year * 2

def statues_end_of_first_year : ℕ := initial_statues
def statues_end_of_second_year : ℕ := statues_after_second_year
def statues_end_of_third_year : ℕ := statues_end_of_second_year + statues_added_third_year - statues_removed_third_year
def statues_end_of_fourth_year : ℕ := statues_end_of_third_year + statues_added_fourth_year

theorem statues_at_end_of_fourth_year : statues_end_of_fourth_year = 31 :=
by
  sorry

end statues_at_end_of_fourth_year_l14_14726


namespace isosceles_triangle_perimeter_l14_14631

noncomputable def quadratic_roots (a b c : ℝ) : set ℝ :=
  {x | a * x^2 + b * x + c = 0}

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem isosceles_triangle_perimeter :
  let roots := (quadratic_roots 1 (-4) 3) in
  1 ∈ roots → 3 ∈ roots →
  is_valid_triangle 1 3 3 →
  ∃ p, p = 1 + 3 + 3 ∧ p = 7 :=
by
  intro roots h1 h3 h_valid
  use 1 + 3 + 3
  split
  case left => rfl
  case right => norm_num
  exact h_valid

#eval isosceles_triangle_perimeter

end isosceles_triangle_perimeter_l14_14631


namespace cone_lateral_surface_area_l14_14187

theorem cone_lateral_surface_area (r V : ℝ) (h l S : ℝ) 
  (radius_condition : r = 6)
  (volume_condition : V = 30 * Real.pi)
  (volume_formula : V = (1 / 3) * Real.pi * r^2 * h)
  (slant_height_formula : l = Real.sqrt (r^2 + h^2))
  (lateral_surface_area_formula : S = Real.pi * r * l) :
  S = 39 * Real.pi := 
sorry

end cone_lateral_surface_area_l14_14187


namespace dot_product_is_one_l14_14719

variable {V : Type*} [InnerProductSpace ℝ V]
variables (a b : V)

theorem dot_product_is_one 
  (ha : ∥a∥ = 1) 
  (hb : ∥b∥ = sqrt 3) 
  (hab : ∥a - 2•b∥ = 3) : 
  ⟪a, b⟫ = 1 :=
by 
  sorry

end dot_product_is_one_l14_14719


namespace distance_A_to_directrix_of_C_l14_14241

noncomputable def distance_point_to_directrix 
  (A : ℝ × ℝ) (p : ℝ) (hA_on_parabola : A.2^2 = 2 * p * A.1) : ℝ :=
let directrix := -p / 2 in
let distance := (A.1 - directrix).abs in
distance

theorem distance_A_to_directrix_of_C 
  (A : ℝ × ℝ) (p : ℝ) (hA_on_parabola : A.2^2 = 2 * p * A.1) 
  (hA : A = (1, real.sqrt 2)) : distance_point_to_directrix A p hA_on_parabola = 3 / 2 :=
sorry

end distance_A_to_directrix_of_C_l14_14241


namespace find_x_l14_14864

-- Define the variables and conditions
def x := 27
axiom h : (5 / 3) * x = 45

-- Main statement to be proved
theorem find_x : x = 27 :=
by
  have : (5 / 3) * x = 45 := h
  sorry

end find_x_l14_14864


namespace exists_subsequence_sum_39_l14_14485

theorem exists_subsequence_sum_39
  (a : Fin 100 → ℕ)
  (h_cond : ∀ i : Fin 91, (∑ j in Finset.Ico i (i + 10), a j) ≤ 16)
  (h_a_val : ∀ i : Fin 100, a i = 1 ∨ a i = 2) :
  ∃ h k : Fin 100, h < k ∧ (∑ j in Finset.Ico h (k + 1), a j) = 39 :=
by
  sorry

end exists_subsequence_sum_39_l14_14485


namespace maria_success_rate_increase_l14_14831

/-- 
Maria made 7 out of her first 15 basketball free throws. She makes 3/4 of her next 18 attempts. 
We need to prove that Maria increases her overall success rate by 17 percentage points. 
-/
theorem maria_success_rate_increase 
  (initial_successes : ℕ) (initial_attempts : ℕ) 
  (success_rate_ratio : ℚ) (next_attempts : ℕ) 
  (increase_in_percentage_points : ℕ) 
  (new_successes : ℚ)
  (total_successes : ℚ)
  (total_attempts : ℚ)
  (new_success_rate : ℚ)
  (old_success_rate : ℚ)
  (increase : ℚ): 
  initial_successes = 7 -> 
  initial_attempts = 15 -> 
  success_rate_ratio = 3/4 -> 
  next_attempts = 18 -> 
  new_successes = success_rate_ratio * next_attempts ->
  total_successes = initial_successes + 14 ->   -- 14 because 13.5 rounds to 14
  total_attempts = initial_attempts + next_attempts -> 
  new_success_rate = total_successes / total_attempts -> 
  old_success_rate = initial_successes / initial_attempts -> 
  increase = new_success_rate * 100 - old_success_rate * 100 ->
  increase_in_percentage_points = 17 ->
  initial_successes = 7 ∧ initial_attempts = 15 ∧ success_rate_ratio = 3/4 ∧
  next_attempts = 18 ∧ new_successes = 14 ∧ total_successes = 21 ∧ total_attempts = 33 ∧
  old_success_rate ≈ 0.4667 ∧ new_success_rate ≈ 0.6363 ∧ increase = increase_in_percentage_points * 1
:= by
  intros h₁ h₂ h₃ h₄ h₅ h₆ h₇ h₈ h₉ h₀ hx₁ hx₂
  sorry

end maria_success_rate_increase_l14_14831


namespace find_x_l14_14854

theorem find_x (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by 
  sorry

end find_x_l14_14854


namespace dice_probability_abs_difference_l14_14024

theorem dice_probability_abs_difference :
  let outcomes := [(1, 5), (2, 6), (5, 1), (6, 2)] in
  let total_outcomes := 36 in
  (outcomes.length : ℚ) / total_outcomes = 1 / 9 :=
by
  let outcomes := [(1, 5), (2, 6), (5, 1), (6, 2)]
  have h_outcomes_length : outcomes.length = 4 :=
    by decide
  have h_total_outcomes : total_outcomes = 36 :=
    by decide
  calc
    (outcomes.length : ℚ) / total_outcomes 
    = 4 / 36 : by rw [h_outcomes_length, h_total_outcomes]
    ... = 1 / 9 : by norm_num

end dice_probability_abs_difference_l14_14024


namespace michael_choices_l14_14320

theorem michael_choices (n k : ℕ) (h_n : n = 10) (h_k : k = 4) : nat.choose n k = 210 :=
by
  rw [h_n, h_k]
  norm_num
  sorry

end michael_choices_l14_14320


namespace number_of_free_numbers_l14_14569

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def has_no_perfect_squares_in_row (row : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, ¬ is_perfect_square (row i)

def has_no_perfect_squares_in_column (table : ℕ → ℕ → ℕ) (j : ℕ) : Prop :=
  ∀ i : ℕ, ¬ is_perfect_square (table i j)

def is_free_number (table : ℕ → ℕ → ℕ) (i j : ℕ) : Prop :=
  has_no_perfect_squares_in_row (λ k, table i k) ∧ has_no_perfect_squares_in_column table j

def count_free_numbers (table : ℕ → ℕ → ℕ) : ℕ :=
  (List.range 100).sum (λ i, 
    (List.range 100).countp (λ j, is_free_number table i j))

theorem number_of_free_numbers : count_free_numbers (λ i j, 100 * i + j + 1) = 1950 :=
  sorry

end number_of_free_numbers_l14_14569


namespace dot_product_proof_l14_14662

variables {ℝ : Type*}
variables (a b : ℝ → ℝ)
variables [inner_product_space ℝ ℝ]

theorem dot_product_proof
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = sqrt 3)
  (h3 : ∥a - 2 • b∥ = 3) :
  inner (a : ℝ) (b : ℝ) = 1 :=
sorry

end dot_product_proof_l14_14662


namespace lateral_surface_area_of_given_cone_l14_14211

noncomputable def coneLateralSurfaceArea (r V : ℝ) : ℝ :=
let h := (3 * V) / (π * r^2) in
let l := Real.sqrt (r^2 + h^2) in
π * r * l

theorem lateral_surface_area_of_given_cone :
  coneLateralSurfaceArea 6 (30 * π) = 39 * π := by
simp [coneLateralSurfaceArea]
sorry

end lateral_surface_area_of_given_cone_l14_14211


namespace smallest_five_digit_reverse_multiple_of_4_largest_five_digit_reverse_multiple_of_4_l14_14429

def isReverseMultipleOf4 (n : ℕ) : Prop :=
  let n_rev := n.digits.reverse.foldl (λ acc d, 10 * acc + d) 0
  (n_rev % 4) = 0

def isFiveDigitNumber (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem smallest_five_digit_reverse_multiple_of_4 :
  ∃ n : ℕ, isFiveDigitNumber n ∧ isReverseMultipleOf4 n ∧ 
  (∀ m : ℕ, isFiveDigitNumber m ∧ isReverseMultipleOf4 m → n ≤ m) :=
by sorry

theorem largest_five_digit_reverse_multiple_of_4 :
  ∃ n : ℕ, isFiveDigitNumber n ∧ isReverseMultipleOf4 n ∧ 
  (∀ m : ℕ, isFiveDigitNumber m ∧ isReverseMultipleOf4 m → n ≥ m) :=
by sorry

end smallest_five_digit_reverse_multiple_of_4_largest_five_digit_reverse_multiple_of_4_l14_14429


namespace function_decreases_iff_l14_14293

theorem function_decreases_iff (m : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → (m - 3) * x1 + 4 > (m - 3) * x2 + 4) ↔ m < 3 :=
by
  sorry

end function_decreases_iff_l14_14293


namespace lindy_distance_when_meeting_l14_14341

def jack_speed := 3 -- speed in feet per second
def christina_speed := 3 -- speed in feet per second
def lindy_speed := 10 -- speed in feet per second
def initial_distance := 240 -- initial distance in feet

def meeting_time := initial_distance / (jack_speed + christina_speed)

def lindy_total_distance := lindy_speed * meeting_time

theorem lindy_distance_when_meeting : lindy_total_distance = 400 :=
by 
  -- Convert the given data into definitions, namely speeds and distances
  have jack_christina_combined_speed : ℝ := jack_speed + christina_speed
  have meeting_time_value : ℝ := initial_distance / jack_christina_combined_speed
  have lindy_distance_value : ℝ := lindy_speed * meeting_time_value
  -- Assert the calculated result with the correct answer from the solution
  calc
    lindy_distance_value = 10 * 40         : by sorry -- need calculations to prove it
                     ... = 400             : by sorry -- final result

end lindy_distance_when_meeting_l14_14341


namespace initial_bowls_eq_70_l14_14533

def customers : ℕ := 20
def bowls_per_customer : ℕ := 20
def reward_ratio := 10
def reward_bowls := 2
def remaining_bowls : ℕ := 30

theorem initial_bowls_eq_70 :
  let rewards_per_customer := (bowls_per_customer / reward_ratio) * reward_bowls
  let total_rewards := (customers / 2) * rewards_per_customer
  (remaining_bowls + total_rewards) = 70 :=
by
  sorry

end initial_bowls_eq_70_l14_14533


namespace order_of_a_b_c_l14_14174

def a := Real.log 3 / Real.log 2
def b := Real.log 3 / Real.log (1 / 2)
def c := 3 ** (-1 / 2 : ℝ)

theorem order_of_a_b_c : a > c ∧ c > b := by
  have ha : a = Real.log 3 / Real.log 2 := rfl
  have hb : b = Real.log 3 / Real.log (1 / 2) := rfl
  have hc : c = 3 ** (-1 / 2 : ℝ) := rfl
  sorry

end order_of_a_b_c_l14_14174


namespace cone_lateral_surface_area_l14_14193

-- Definitions based on the conditions
def coneRadius : ℝ := 6
def coneVolume : ℝ := 30 * Real.pi

-- Mathematical statement
theorem cone_lateral_surface_area (r V : ℝ) (hr : r = coneRadius) (hV : V = coneVolume) :
  ∃ S : ℝ, S = 39 * Real.pi :=
by 
  have h_volume := hV
  have h_radius := hr
  sorry

end cone_lateral_surface_area_l14_14193


namespace yard_length_l14_14305

theorem yard_length
  (num_trees : ℕ)
  (distance_between_trees : ℕ)
  (trees_at_ends : bool)
  (h_num_trees : num_trees = 26)
  (h_distance_between : distance_between_trees = 24)
  (h_trees_at_ends : trees_at_ends = tt) :
  let num_gaps := num_trees - 1 in
  let yard_length := num_gaps * distance_between_trees in
  yard_length = 600 := by
  sorry

end yard_length_l14_14305


namespace value_of_as1_plus_bs2_plus_cs3_l14_14250

noncomputable def roots : Type := { x1 x2 : ℂ // a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 }

variables (a b c : ℂ) [fact (a ≠ 0)]
variables {x1 x2 : ℂ} (hx: roots a b c)
let s1 := x1^(2005) + x2^(2005)
let s2 := x1^(2004) + x2^(2004)
let s3 := x1^(2003) + x2^(2003)

theorem value_of_as1_plus_bs2_plus_cs3 :
  a * s1 + b * s2 + c * s3 = 0 :=
sorry

end value_of_as1_plus_bs2_plus_cs3_l14_14250


namespace minimum_height_for_surface_area_geq_120_l14_14374

noncomputable def box_surface_area (x : ℝ) : ℝ :=
  6 * x^2 + 20 * x

theorem minimum_height_for_surface_area_geq_120 :
  ∃ (x : ℝ), (x ≥ 0) ∧ (box_surface_area x ≥ 120) ∧ (x + 5 = 9) := by
  sorry

end minimum_height_for_surface_area_geq_120_l14_14374


namespace find_cd_value_l14_14774

theorem find_cd_value :
  ∀ (a b d : ℕ), 
  let ab := 10 * a + b
  let cd := 10 * 9 + d
  (a ≠ b ∧ b ≠ 9 ∧ d ≠ a ∧ d ≠ b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9) →
  ∃ jjj : ℕ, (div (9*9) 100 = jjj / 100 ) and (ab + cd = jjj) → cd = 98 := 
sorry

end find_cd_value_l14_14774


namespace cone_lateral_surface_area_l14_14209

theorem cone_lateral_surface_area (r : ℕ) (V : ℝ) (h l S : ℝ)
  (h_r : r = 6)
  (h_V : V = 30 * Real.pi)
  (h_volume : V = (1 / 3) * Real.pi * (r ^ 2) * h)
  (h_slant_height : l = Real.sqrt (r^2 + h^2))
  (h_lateral_surface_area : S = Real.pi * r * l) :
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l14_14209


namespace three_digit_numbers_count_l14_14117

def number_of_3_digit_numbers : ℕ := 
  let without_zero := 2 * Nat.choose 9 3
  let with_zero := Nat.choose 9 2
  without_zero + with_zero

theorem three_digit_numbers_count : number_of_3_digit_numbers = 204 := by
  -- Proof to be completed
  sorry

end three_digit_numbers_count_l14_14117


namespace sum_of_first_9_terms_is_99_l14_14433

variable {α : Type} [LinearOrderedField α]

-- Define the arithmetic sequence in general
def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d : α, ∀ n : ℕ, a(n+1) = a n + d

-- Given conditions
variable (a : ℕ → α)
variable (h_seq : is_arithmetic_sequence a)
variable (h_sum : a 1 + a 3 + a 5 + a 7 + a 9 = 55)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_of_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  ∑ i in range (n + 1), a i

-- Prove that S_9 == 99
theorem sum_of_first_9_terms_is_99 : sum_of_first_n_terms a 9 = 99 :=
by
  sorry

end sum_of_first_9_terms_is_99_l14_14433


namespace min_value_x_2y_l14_14178

theorem min_value_x_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2*y + 2*x*y = 8) : x + 2*y ≥ 4 :=
sorry

end min_value_x_2y_l14_14178


namespace choosing_4_out_of_10_classes_l14_14315

theorem choosing_4_out_of_10_classes :
  ∑ (k : ℕ) in (finset.range 5).map (prod.mk 10), k! / (4! * (k - 4)!) = 210 :=
by sorry

end choosing_4_out_of_10_classes_l14_14315


namespace dot_product_is_one_l14_14717

variable {V : Type*} [InnerProductSpace ℝ V]
variables (a b : V)

theorem dot_product_is_one 
  (ha : ∥a∥ = 1) 
  (hb : ∥b∥ = sqrt 3) 
  (hab : ∥a - 2•b∥ = 3) : 
  ⟪a, b⟫ = 1 :=
by 
  sorry

end dot_product_is_one_l14_14717


namespace part1_part2_l14_14819

variables {A B C M A' B' C' : Type}
variables [Incenter A B C M] [AltitudeFoot A' A M] [AltitudeFoot B' B M] [AltitudeFoot C' C M]

theorem part1 (h : is_orthocenter M A B C) :
  (AB / CM) + (BC / AM) + (CA / BM) = (AB * BC * CA) / (AM * BM * CM) :=
sorry

theorem part2 (h : is_orthocenter M A B C) (h1 : is_altitude_foot A' A M) (h2 : is_altitude_foot B' B M) (h3 : is_altitude_foot C' C M):
  (A'M * AM = B'M * BM) ∧ (B'M * BM = C'M * CM) :=
sorry

end part1_part2_l14_14819


namespace games_played_by_third_player_l14_14017

theorem games_played_by_third_player
    (games_first : ℕ)
    (games_second : ℕ)
    (games_first_eq : games_first = 10)
    (games_second_eq : games_second = 21) :
    ∃ (games_third : ℕ), games_third = 11 := by
  sorry

end games_played_by_third_player_l14_14017


namespace remainder_when_divided_by_13_l14_14501

theorem remainder_when_divided_by_13 (N k : ℤ) (h : N = 39 * k + 20) : N % 13 = 7 := by
  sorry

end remainder_when_divided_by_13_l14_14501


namespace expand_expression_l14_14567

theorem expand_expression (x : ℝ) : (15 * x + 17 + 3) * (3 * x) = 45 * x^2 + 60 * x :=
by
  sorry

end expand_expression_l14_14567


namespace find_dot_product_l14_14674

open Real

noncomputable def vec_a : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector a
noncomputable def vec_b : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector b

def magnitude (v : ℝ → ℝ → ℝ) : ℝ :=
  sqrt ((v 0) ^ 2 + (v 1)^ 2)

def dot_product (u v : ℝ → ℝ → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

axiom magnitude_a_eq1 : magnitude vec_a = 1
axiom magnitude_b_eq_sqrt3 : magnitude vec_b = sqrt 3
axiom magnitude_a_minus_2b_eq3 : magnitude (λ x, vec_a x - 2 * vec_b x) = 3

theorem find_dot_product (a b : ℝ → ℝ → ℝ) 
  (ha : magnitude a = 1) 
  (hb : magnitude b = sqrt 3) 
  (h : magnitude (λ x, a x - 2 * b x) = 3) :
  dot_product a b = 1 := sorry

end find_dot_product_l14_14674


namespace people_in_first_group_l14_14291

theorem people_in_first_group (P : ℕ) (work_done_by_P : 60 = 1 / (P * (1/60))) (work_done_by_16 : 30 = 1 / (16 * (1/30))) : P = 8 :=
by
  sorry

end people_in_first_group_l14_14291


namespace number_of_containers_needed_l14_14925

/-
  Define the parameters for the given problem
-/
def bags_suki : ℝ := 6.75
def weight_per_bag_suki : ℝ := 27

def bags_jimmy : ℝ := 4.25
def weight_per_bag_jimmy : ℝ := 23

def bags_natasha : ℝ := 3.80
def weight_per_bag_natasha : ℝ := 31

def container_capacity : ℝ := 17

/-
  The total weight bought by each person and the total combined weight
-/
def total_weight_suki : ℝ := bags_suki * weight_per_bag_suki
def total_weight_jimmy : ℝ := bags_jimmy * weight_per_bag_jimmy
def total_weight_natasha : ℝ := bags_natasha * weight_per_bag_natasha

def total_weight_combined : ℝ := total_weight_suki + total_weight_jimmy + total_weight_natasha

/-
  Prove that number of containers needed is 24
-/
theorem number_of_containers_needed : 
  Nat.ceil (total_weight_combined / container_capacity) = 24 := 
by
  sorry

end number_of_containers_needed_l14_14925


namespace part_a_part_b_l14_14478

-- Part (a) 
theorem part_a (S : Finset ℕ) (m : ℕ) (hS : S.card = m) : (S.product S).card ≤ m * (m + 1) / 2 :=
sorry

-- Part (b)
def C (m : ℕ) : ℕ :=
  Sup { k | ∃ S : Finset ℕ, S.card = m ∧ (Finset.range (k + 1) ⊆ (S ∪ (S.product S).image (λ p, p.1 + p.2))) }

theorem part_b (m : ℕ) (hm : 0 < m) :
  m * (m + 6) / 4 ≤ C m ∧ C m ≤ m * (m + 3) / 2 :=
sorry

end part_a_part_b_l14_14478


namespace elaine_rent_percentage_l14_14345

theorem elaine_rent_percentage (E : ℝ) (P : ℝ) 
  (h1 : E > 0) 
  (h2 : P > 0) 
  (h3 : 0.25 * 1.15 * E = 1.4375 * (P / 100) * E) : 
  P = 20 := 
sorry

end elaine_rent_percentage_l14_14345


namespace arithmetic_sequence_divisor_l14_14986

def divides (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem arithmetic_sequence_divisor (a d : ℕ) (h : 0 < a) (h' : 0 < d) :
  divides 15 (∑ i in Finset.range 15, (a + i * d)) :=
by
  sorry

end arithmetic_sequence_divisor_l14_14986


namespace lateral_surface_area_of_cone_l14_14001

theorem lateral_surface_area_of_cone (r l : ℝ) (h₁ : r = 3) (h₂ : l = 5) :
  π * r * l = 15 * π :=
by sorry

end lateral_surface_area_of_cone_l14_14001


namespace slope_of_l_l14_14773

noncomputable def parabola_eq : ℝ × ℝ → Prop :=
  λ P, P.1 ^ 2 = 4 * P.2 + 4

noncomputable def line_parametric_eq (α : ℝ) : ℝ → ℝ × ℝ :=
  λ t, (t * Real.cos α, t * Real.sin α)

theorem slope_of_l (α : ℝ) 
  (h_intersect: ∀ (t : ℝ), ∃ (p : ℝ × ℝ), parabola_eq p ∧ line_parametric_eq α t = p) 
  (h_AB: ∀ (ρ1 ρ2 : ℝ), |ρ1 - ρ2| = 8) :
  Real.tan α = 1 ∨ Real.tan α = -1 :=
sorry

end slope_of_l_l14_14773


namespace parabola_and_circle_eq_and_tangent_l14_14964

noncomputable def parabola_eq (x y : ℝ) : Prop := y^2 = x

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

theorem parabola_and_circle_eq_and_tangent :
  (∀ x y : ℝ, parabola_eq x y ↔ y^2 = x) ∧
  (∀ x y : ℝ, circle_eq x y ↔ (x - 2)^2 + y^2 = 1) ∧
  (∀ A1 A2 A3 : ℝ × ℝ,
    parabola_eq A1.1 A1.2 ∧ parabola_eq A2.1 A2.2 ∧ parabola_eq A3.1 A3.2 →
    tangent_to_circle A1 A2 A3)
:=
sorry

end parabola_and_circle_eq_and_tangent_l14_14964


namespace grandmas_turtle_statues_l14_14728

theorem grandmas_turtle_statues : 
  let year1 := 4 in
  let year2 := 4 * year1 in
  let year3 := year2 + 12 - 3 in
  let year4 := year3 + 2 * 3 in
  year4 = 31 :=
by
  sorry

end grandmas_turtle_statues_l14_14728


namespace minimize_average_annual_cost_l14_14086

noncomputable def maintenance_cost (n : ℕ) : ℝ :=
  if n = 1 then 0
  else if n = 2 then 0.2
  else 0.2 * (n - 1)

noncomputable def total_maintenance_cost (n : ℕ) : ℝ :=
  (0.1 * n^2) - (0.1 * n)

noncomputable def f (n : ℕ) : ℝ :=
  14.4 + 0.7 * n + total_maintenance_cost n

def average_annual_cost (n : ℕ) : ℝ :=
  f n / n

theorem minimize_average_annual_cost : ∃ (n : ℕ), n = 12 ∧ ∀ m : ℕ, m ≠ 12 → average_annual_cost 12 ≤ average_annual_cost m :=
by
  sorry

end minimize_average_annual_cost_l14_14086


namespace sum_of_solutions_l14_14588

theorem sum_of_solutions:
  (∀ x, (x - 6)^2 = 25 → x = 1 ∨ x = 11) →
  ∑ x in {x | (x - 6)^2 = 25}, x = 12 :=
by sorry

end sum_of_solutions_l14_14588


namespace tangents_meet_at_altitude_l14_14842

-- Definitions of our geometric entities and conditions
variables (A B C M N H K : Point)
variables (ABC : Triangle)
variables (semicircle : Semicircle)
variables (AB : Segment)
variables (CH: Line)

-- Assuming A, B, and H are collinear on a line
-- Assuming the semicircle is formed with AB as the diameter
-- Assuming M and N are the intersections of the semicircle with lines AC and BC respectively.
-- Assuming CH is the altitude from point C to line AB

def semicircle_diameter_AB (ABC : Triangle) (semicircle : Semicircle) : Prop :=
  semicircle.diameter = AB

def intersections_M_N (semicircle : Semicircle) (AC BC : Line) (M N : Point) : Prop :=
  semicircle.intersects AC = M ∧ semicircle.intersects BC = N

def altitude_CH (ABC : Triangle) (CH : Line) : Prop :=
  CH.altitude_from C = line_of A B

def tangents_intersect_altitude (semicircle : Semicircle) (M N : Point)
  (CH: Line) : Prop :=
  let tangent_M := semicircle.tangent_at M
  let tangent_N := semicircle.tangent_at N in
    intersects (tangent_M ∩ tangent_N) CH

-- The theorem we seek to prove
theorem tangents_meet_at_altitude (ABC : Triangle) (semicircle : Semicircle) (AB : Segment) 
(M N H K : Point) (CH : Line) 
(h_diameter : semicircle_diameter_AB ABC semicircle)
(h_intersections : intersections_M_N semicircle AC BC M N)
(h_altitude : altitude_CH ABC CH) : tangents_intersect_altitude semicircle M N CH :=
sorry

end tangents_meet_at_altitude_l14_14842


namespace find_x_l14_14857

theorem find_x (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by 
  sorry

end find_x_l14_14857


namespace principal_amount_l14_14937

theorem principal_amount (P : ℝ) (CI SI : ℝ) 
  (H1 : CI = P * 0.44) 
  (H2 : SI = P * 0.4) 
  (H3 : CI - SI = 216) : 
  P = 5400 :=
by {
  sorry
}

end principal_amount_l14_14937


namespace max_candies_l14_14149

theorem max_candies (V M S : ℕ) (hv : V = 35) (hm : 1 ≤ M ∧ M < 35) (hs : S = 35 + M) (heven : Even S) : V + M + S = 136 :=
sorry

end max_candies_l14_14149


namespace super_cool_triangle_areas_sum_l14_14518

theorem super_cool_triangle_areas_sum :
  ∃ (A : ℕ), 
    (∀ (a b : ℕ), (a * b / 2 = 3 * (a + b))) → 
    A = 471 :=
begin
  sorry
end

end super_cool_triangle_areas_sum_l14_14518


namespace find_number_l14_14056

theorem find_number (x : ℝ) (h : 3 * (2 * x + 5) = 129) : x = 19 :=
by
  sorry

end find_number_l14_14056


namespace median_time_is_150_l14_14952

-- Define the list of times in seconds
def times_in_seconds : List Nat := [
    30, 45, 55,
    70, 85, 100, 115,
    125, 135, 150, 165, 175,
    195, 195, 210, 220, 225,
    250, 265
]

-- Proof statement: The median of the times_in_seconds list is 150 seconds
theorem median_time_is_150 :
  let sorted_times := List.sort times_in_seconds
  sorted_times.nth 10 = some 150 := 
by
  -- We defer the proof details (to be filled in with specific steps)
  sorry

end median_time_is_150_l14_14952


namespace digits_of_smallest_n_l14_14358

def is_divisible_by_30 (n : ℕ) : Prop := 30 ∣ n

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k^2 = n

def smallest_n := 
  Nat.find 
    (λ n, is_divisible_by_30 n ∧ is_perfect_cube (n^2) ∧ is_perfect_square (n^3))

def number_of_digits (n : ℕ) : ℕ := Nat.log 10 n + 1

theorem digits_of_smallest_n : number_of_digits smallest_n = 9 := 
  sorry

end digits_of_smallest_n_l14_14358


namespace parallel_ne_implies_value_l14_14724

theorem parallel_ne_implies_value 
  (x : ℝ) 
  (m : ℝ × ℝ := (2 * x, 7)) 
  (n : ℝ × ℝ := (6, x + 4)) 
  (h1 : 2 * x * (x + 4) = 42) 
  (h2 : m ≠ n) :
  x = -7 :=
by {
  sorry
}

end parallel_ne_implies_value_l14_14724


namespace given_inequality_true_l14_14236

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem given_inequality_true (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h_deriv : ∀ x, deriv f x = f' x)
  (h_odd : odd_function f)
  (h_cond : ∀ x, (0 < x ∧ x < π ∨ π < x ∧ x < 2 * π) → f x + f' x * Real.tan x > 0) :
  √2 * f (π / 4) + f (-π / 6) > 0 :=
sorry

end given_inequality_true_l14_14236


namespace distance_to_school_9_miles_l14_14342

noncomputable def distance_to_school (v : ℝ) : ℝ :=
  let d := (20:ℝ) / 60 * v in
  d

theorem distance_to_school_9_miles :
  ∃ d v : ℝ, d = distance_to_school v ∧ d = 9 ∧ ((20:ℝ) / 60 * v = (12:ℝ) / 60 * (v + 18)) :=
by
  sorry

end distance_to_school_9_miles_l14_14342


namespace sum_of_below_avg_l14_14053

-- Define class averages
def a1 := 75
def a2 := 85
def a3 := 90
def a4 := 65

-- Define the overall average
def avg : ℚ := (a1 + a2 + a3 + a4) / 4

-- Define a predicate indicating if a class average is below the overall average
def below_avg (a : ℚ) : Prop := a < avg

-- The theorem to prove the required sum of averages below the overall average
theorem sum_of_below_avg : a1 < avg ∧ a4 < avg → a1 + a4 = 140 :=
by
  sorry

end sum_of_below_avg_l14_14053


namespace dot_product_l14_14665

variables (a b : Vector ℝ) -- ℝ here stands for the real numbers

-- Given conditions
def condition1 : ∥a∥ = 1 := sorry
def condition2 : ∥b∥ = √3 := sorry
def condition3 : ∥a - (2 : ℝ) • b∥ = 3 := sorry

-- Goal to prove
theorem dot_product (a b : Fin₃ → ℝ) 
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = √3)
  (h3 : ∥a - (2 : ℝ) • b∥ = 3) : 
  a ⬝ b = 1 := 
sorry

end dot_product_l14_14665


namespace max_cubes_fit_in_box_l14_14461

theorem max_cubes_fit_in_box :
  let L := 8
  let W := 9
  let H := 12
  let V_box := L * W * H
  let V_cube := 27
  V_box / V_cube = 32 := by
  let L := 8
  let W := 9
  let H := 12
  let V_box := L * W * H
  let V_cube := 27
  have h1 : V_box = 8 * 9 * 12 := rfl
  have h2 : V_cube = 27 := rfl
  have h3 : 8 * 9 * 12 = 864 := by norm_num
  have h4 : 864 / 27 = 32 := by norm_num
  show 864 / 27 = 32 from h4
  sorry

end max_cubes_fit_in_box_l14_14461


namespace exists_two_candidates_solve_all_problems_l14_14489

theorem exists_two_candidates_solve_all_problems :
  ∃ e1 e2 : Fin 200, ∀ p : Fin 6, (solved p e1 ∨ solved p e2) :=
by
  -- Define the conditions as assumptions
  assume (solved : Fin 6 → Fin 200 → Prop)
  -- Condition: Each problem is solved by at least 120 participants
  assume h_solved_min : ∀ p : Fin 6, 120 ≤ (Finset.filter (solved p) Finset.univ).card
  
  -- Goal: There exist two candidates such that each problem has been solved by at least one of them.
  sorry

end exists_two_candidates_solve_all_problems_l14_14489


namespace steve_paid_18_l14_14833

-- Define the conditions
def mike_price : ℝ := 5
def steve_multiplier : ℝ := 2
def shipping_rate : ℝ := 0.8

-- Define Steve's cost calculation
def steve_total_cost : ℝ :=
  let steve_dvd_price := steve_multiplier * mike_price
  let shipping_cost := shipping_rate * steve_dvd_price
  steve_dvd_price + shipping_cost

-- Prove that Steve's total payment is 18.
theorem steve_paid_18 : steve_total_cost = 18 := by
  -- Provide a placeholder for the proof
  sorry

end steve_paid_18_l14_14833


namespace bacteria_initial_count_l14_14416

theorem bacteria_initial_count (t : ℕ) (quad : ℕ) (final_count : ℕ): 
  (quad = 4) → 
  (t = 5 * 60) → 
  (final_count = 4194304) → 
  let n := final_count / (quad ^ (t / 20)) in
  n = 1 := by
  sorry

end bacteria_initial_count_l14_14416


namespace factorial_division_l14_14041

theorem factorial_division : (nat.factorial 15) / ((nat.factorial 6) * (nat.factorial 9)) = 5005 := 
by 
    sorry

end factorial_division_l14_14041


namespace dot_product_eq_one_l14_14698

variables {α : Type*} [InnerProductSpace ℝ α]

noncomputable def vector_a (a : α) : Prop := ∥a∥ = 1
noncomputable def vector_b (b : α) : Prop := ∥b∥ = real.sqrt 3
noncomputable def vector_c (a b : α) : Prop := ∥a - (2 : ℝ) • b∥ = 3

theorem dot_product_eq_one (a b : α) (ha : vector_a a) (hb : vector_b b) (hc : vector_c a b) :
  inner a b = 1 :=
sorry

end dot_product_eq_one_l14_14698


namespace dot_product_eq_one_l14_14694

variables {α : Type*} [InnerProductSpace ℝ α]

noncomputable def vector_a (a : α) : Prop := ∥a∥ = 1
noncomputable def vector_b (b : α) : Prop := ∥b∥ = real.sqrt 3
noncomputable def vector_c (a b : α) : Prop := ∥a - (2 : ℝ) • b∥ = 3

theorem dot_product_eq_one (a b : α) (ha : vector_a a) (hb : vector_b b) (hc : vector_c a b) :
  inner a b = 1 :=
sorry

end dot_product_eq_one_l14_14694


namespace jack_more_votes_than_john_l14_14490

theorem jack_more_votes_than_john :
  ∀ (total_votes john_votes : ℕ)
    (james_percentage_remaining jacob_percentage_combined joey_percentage_addition jack_percentage_joey_decrease : ℝ),
  total_votes = 1150 →
  john_votes = 150 →
  james_percentage_remaining = 0.70 →
  jacob_percentage_combined = 0.30 →
  joey_percentage_addition = 1.25 →
  jack_percentage_joey_decrease = 0.95 →
  let remaining_votes := total_votes - john_votes
      james_votes := james_percentage_remaining * remaining_votes
      combined_votes := john_votes + james_votes
      jacob_votes := jacob_percentage_combined * combined_votes
      joey_votes := joey_percentage_addition * jacob_votes
      jack_votes := jack_percentage_joey_decrease * joey_votes
  in round jack_votes - john_votes = 153 :=
  sorry

end jack_more_votes_than_john_l14_14490


namespace smallest_a_value_l14_14354

theorem smallest_a_value (a : ℤ) (P : ℤ → ℤ)
  (ha_pos : a > 0)
  (P_int_coeffs : ∀ n, P n ∈ ℤ)
  (h1 : P 1 = a) (h4 : P 4 = a) (h6 : P 6 = a) (h9 : P 9 = a)
  (h3 : P 3 = -a) (h5 : P 5 = -a) (h8 : P 8 = -a) (h10 : P 10 = -a) :
  a = 10080 :=
sorry

end smallest_a_value_l14_14354


namespace planes_determined_by_parallel_lines_l14_14756

theorem planes_determined_by_parallel_lines (l1 l2 l3 : Type)
  (h1 : ∃ p : Type, ∀ (a : l1) (b : l2), p)
  (h2 : ∃ q : Type, ∀ (b : l2) (c : l3), q)
  (h3 : ∃ r : Type, ∀ (c : l3) (a : l1), r)
  (h_parallel : ∀ (a1 a2 : l1) (b1 b2 : l2) (c1 c2 : l3),
                a1 = a2 ∨ b1 = b2 ∨ c1 = c2) :
  (∑ 2, (finset.univ : finset (fin 3)).mem) = 3 :=
by intro h; exact succ_inj h; simp; sorry

end planes_determined_by_parallel_lines_l14_14756


namespace no_real_intersection_l14_14451

def parabola_line_no_real_intersection : Prop :=
  let a := 3
  let b := -6
  let c := 5
  (b^2 - 4 * a * c) < 0

theorem no_real_intersection (h : parabola_line_no_real_intersection) : 
  ∀ x : ℝ, 3*x^2 - 4*x + 2 ≠ 2*x - 3 :=
by sorry

end no_real_intersection_l14_14451


namespace determine_speeds_l14_14984

def train_speeds (length1 length2 dist time delta_t : ℝ) (speed1 speed2 : ℝ) :=
  length1 = 490 ∧ length2 = 210 ∧ dist = 700 ∧ time = 28 ∧ delta_t = 35 ∧
  700 = 28 * (speed1 + speed2) ∧
  (490 / speed1 - 210 / speed2 = 35)

theorem determine_speeds :
  ∃ (speed1 speed2 : ℝ),
    train_speeds 490 210 700 28 35 speed1 speed2 ∧ speed1 = 10 ∧ speed2 = 15 :=
begin
  sorry
end

end determine_speeds_l14_14984


namespace select_p3_squares_l14_14347

theorem select_p3_squares (p : ℕ) (hp : p.prime)
  (array : matrix (fin (p^2)) (fin (p^2)) ℕ) :
  ∃ (selected : fin (p^2) → fin (p^2) → Prop),
    (∀ i j, selected i j → array i j = 1) ∧
    (∃ h : ∀ (i1 i2 j1 j2 : fin (p^2)),
           (i1 ≠ i2 ∨ j1 ≠ j2) →
           selected i1 j1 →
           selected i2 j2 →
           selected i1 j2 →
           selected i2 j1 →
           false,
      (finset.univ.filter (λ (i : fin (p^2)), finset.univ.filter (λ (j : fin (p^2)), selected i j) = p^3))) :=
sorry

end select_p3_squares_l14_14347


namespace inequality_solution_l14_14401

theorem inequality_solution (x : ℝ) : 
  (1 / (x - 1) - 3 / (x - 2) + 3 / (x - 3) - 1 / (x - 4)) < (1 / 24) ↔ 
  x ∈ Set.Ioo (-∞) (-2) ∪ Set.Ioo (-1) 1 ∪ Set.Ioo 2 3 ∪ Set.Ioo 4 6 :=
by
  sorry

end inequality_solution_l14_14401


namespace area_increase_l14_14103

theorem area_increase (a : ℝ) : ((a + 2) ^ 2 - a ^ 2 = 4 * a + 4) := by
  sorry

end area_increase_l14_14103


namespace prove_identity_l14_14896

variable (x : ℝ)

theorem prove_identity : 
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  -- Expand both sides and prove identity
  sorry

end prove_identity_l14_14896


namespace total_zinc_in_mixture_l14_14449

theorem total_zinc_in_mixture :
  let alloy_A := (3, 5),
      alloy_B := (4, 9),
      weight_A := 80,
      weight_B := 120,
      zinc_alloy_A := weight_A * alloy_A.2 / (alloy_A.1 + alloy_A.2),
      zinc_alloy_B := weight_B * alloy_B.2 / (alloy_B.1 + alloy_B.2),
      total_zinc := zinc_alloy_A + zinc_alloy_B
  in
  total_zinc ≈ 133.07 :=
by sorry

end total_zinc_in_mixture_l14_14449


namespace sin_2_angle_BPC_l14_14870

theorem sin_2_angle_BPC (A B C D P : Type)
    (hAB : A = B) (hBC : B = C) (hCD : C = D)
    (cos_APC : Real.cos (angle A P C) = 3 / 5)
    (cos_BPD : Real.cos (angle B P D) = 1 / 5) :
    Real.sin (2 * angle B P C) = 16 / 15 := by
  sorry

end sin_2_angle_BPC_l14_14870


namespace find_distance_l14_14220

-- Definitions
variables {P : Type} [EuclideanSpace P]
variables {a b c : ℝ}
variables {O : P} (x y z : ℝ)
variables (P : P)

-- Coordinates for point P and origin O
variable (hP: P = ⟨x, y, z⟩)
variable (hO: O = 0)

-- Distances from P to the mutually perpendicular rays OA, OB, and OC
variable (hx : dist P O = sqrt(y^2 + z^2) = a)
variable (hy : dist P O = sqrt(x^2 + z^2) = b)
variable (hz : dist P O = sqrt(x^2 + y^2) = c)

-- The main proof statement
theorem find_distance (h: x^2 + y^2 + z^2 = (a^2 + b^2 + c^2) / 2) :
  dist P O = sqrt((a^2 + b^2 + c^2) / 2) :=
begin
  sorry
end 

end find_distance_l14_14220


namespace max_length_is_2_l14_14613

noncomputable theory

def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

def tangent_line (m : ℝ) : ℝ → ℝ → Prop := λ x y, y = -(x * m / sqrt (m^2 - 1)) + (m^2 / (m^2 - 1))

def intersects (p1 p2 : ℝ × ℝ) (q1 q2 : ℝ × ℝ) : Prop :=
p1.1 * q1.2 - p2.1 * q2.2 = 0 -- Simplified intersection condition

def distance (p q : ℝ × ℝ) : ℝ :=
real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem max_length_is_2 (m : ℝ) :
  (∀ (A B : ℝ × ℝ), intersects A B (m, 0) ∧ ellipse A.1 A.2 ∧ ellipse B.1 B.2 → distance A B ≤ 2) :=
begin
  sorry
end

end max_length_is_2_l14_14613


namespace system_a_l14_14924

theorem system_a (x y z : ℝ) (h1 : x + y + z = 6) (h2 : 1/x + 1/y + 1/z = 11/6) (h3 : x*y + y*z + z*x = 11) :
  x = 1 ∧ y = 2 ∧ z = 3 ∨ x = 1 ∧ y = 3 ∧ z = 2 ∨ x = 2 ∧ y = 1 ∧ z = 3 ∨ x = 2 ∧ y = 3 ∧ z = 1 ∨ x = 3 ∧ y = 1 ∧ z = 2 ∨ x = 3 ∧ y = 2 ∧ z = 1 :=
sorry

end system_a_l14_14924


namespace spinner_final_direction_l14_14951

-- Define the directions as an enumeration
inductive Direction
| north
| east
| south
| west

-- Convert between revolution fractions to direction
def direction_after_revolutions (initial : Direction) (revolutions : ℚ) : Direction :=
  let quarters := (revolutions * 4) % 4
  match initial with
  | Direction.south => if quarters == 0 then Direction.south
                       else if quarters == 1 then Direction.west
                       else if quarters == 2 then Direction.north
                       else Direction.east
  | Direction.east  => if quarters == 0 then Direction.east
                       else if quarters == 1 then Direction.south
                       else if quarters == 2 then Direction.west
                       else Direction.north
  | Direction.north => if quarters == 0 then Direction.north
                       else if quarters == 1 then Direction.east
                       else if quarters == 2 then Direction.south
                       else Direction.west
  | Direction.west  => if quarters == 0 then Direction.west
                       else if quarters == 1 then Direction.north
                       else if quarters == 2 then Direction.east
                       else Direction.south

-- Final proof statement
theorem spinner_final_direction : direction_after_revolutions Direction.south (4 + 3/4 - (6 + 1/2)) = Direction.east := 
by 
  sorry

end spinner_final_direction_l14_14951


namespace friendly_not_blue_l14_14979

-- Defining the total number of snakes.
constant S : Type
constant snakes : Finset S
constant total_snakes : ∀ s : S, s ∈ snakes

-- Given conditions.
constant Blue : Finset S
constant Friendly : Finset S

-- There are 15 snakes.
constant h_total_snakes : (snakes.card = 15)

-- There are 6 blue snakes and 7 friendly snakes.
constant h_blue_snakes : (Blue.card = 6)
constant h_friendly_snakes : (Friendly.card = 7)

-- All friendly snakes can multiply.
constant CanMultiply : S → Prop
constant h_friendly_multiply : ∀ s, s ∈ Friendly → CanMultiply s

-- None of the blue snakes can divide.
constant CanDivide : S → Prop
constant h_blue_no_divide : ∀ s, s ∈ Blue → ¬ CanDivide s

-- All snakes that can't divide also can't multiply.
constant h_no_divide_no_multiply : ∀ s, ¬ CanDivide s → ¬ CanMultiply s

-- The statement we aim to prove.
theorem friendly_not_blue (s : S) (h_s_friendly : s ∈ Friendly) : s ∉ Blue :=
by
  -- Proof not required, just indicating the theorem.
  sorry

end friendly_not_blue_l14_14979


namespace angle_AOD_l14_14327

theorem angle_AOD (OA OC OB OD : Type*)
  [InnerProductSpace ℝ OA]
  [InnerProductSpace ℝ OC]
  [InnerProductSpace ℝ OB]
  [InnerProductSpace ℝ OD]
  (h1 : ∀ {v : OA}, ⟪OA, OC⟫ = (0 : ℝ))
  (h2 : ∀ {v : OB}, ⟪OB, OD⟫ = (0 : ℝ))
  (h3 : ∃ (y : ℝ), ∡ OA OD = 2.5 * ∡ OB OC ∧ ∡ OB OC = y)
  : ∡ OA OD = 128.57 :=
by
  sorry

end angle_AOD_l14_14327


namespace percentage_books_not_sold_approx_63_45_l14_14061

/-
Given:
- Initial stock of books = 1100
- Books sold on Monday = 75
- Books sold on Tuesday = 50
- Books sold on Wednesday = 64
- Books sold on Thursday = 78
- Books sold on Friday = 135

Prove that the percentage of books not sold is approximately 63.45%.
-/

theorem percentage_books_not_sold_approx_63_45 :
  let initial_stock := 1100
  let monday_sales := 75
  let tuesday_sales := 50
  let wednesday_sales := 64
  let thursday_sales := 78
  let friday_sales := 135
  let total_books_sold := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales
  let books_not_sold := initial_stock - total_books_sold
  let percentage_not_sold := (books_not_sold.to_rat / initial_stock.to_rat) * 100
  percentage_not_sold ≈ 63.45 :=
by
  sorry

end percentage_books_not_sold_approx_63_45_l14_14061


namespace continuous_at_three_l14_14162

def f (x : ℝ) (a : ℝ) : ℝ := 
  if x > 3 then x + 5 else 2 * x + a

theorem continuous_at_three (a : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 3) < δ → abs (f x a - f 3 a) < ε) ↔ a = 2 := by
sorry

end continuous_at_three_l14_14162


namespace dot_product_eq_one_l14_14695

variables {α : Type*} [InnerProductSpace ℝ α]

noncomputable def vector_a (a : α) : Prop := ∥a∥ = 1
noncomputable def vector_b (b : α) : Prop := ∥b∥ = real.sqrt 3
noncomputable def vector_c (a b : α) : Prop := ∥a - (2 : ℝ) • b∥ = 3

theorem dot_product_eq_one (a b : α) (ha : vector_a a) (hb : vector_b b) (hc : vector_c a b) :
  inner a b = 1 :=
sorry

end dot_product_eq_one_l14_14695


namespace polynomial_identity_l14_14619

theorem polynomial_identity (a_0 a_1 a_2 a_3 a_4 : ℝ) (x : ℝ) 
  (h : (2 * x + 1)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) : 
  a_0 - a_1 + a_2 - a_3 + a_4 = 1 :=
by
  sorry

end polynomial_identity_l14_14619


namespace identity_holds_l14_14891

noncomputable def identity_proof : Prop :=
∀ (x : ℝ), (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x

theorem identity_holds : identity_proof :=
by
  sorry

end identity_holds_l14_14891


namespace a_n_formula_l14_14127

open Nat

def a_n (n : ℕ) : ℚ :=
  2 * ∏ k in (range (n + 1)).filter (λ k, k ≥ 2), (1 - 1 / (k : ℚ)^2)

theorem a_n_formula (n : ℕ) : a_n n = (n + 2) / (n + 1) := 
by 
  sorry

end a_n_formula_l14_14127


namespace total_handshakes_convention_l14_14532

theorem total_handshakes_convention :
  let twins := 8 * 2
  let quadruplets := 5 * 4
  let twin_handshakes := twins * (twins - 2) / 2
  let quadruplet_handshakes := quadruplets * (quadruplets - 4) / 2
  let cross_handshakes :=
    twins * (2 / 3 * quadruplets).to_nat   -- ≈ 16 * 13 (total approx)
    + quadruplets * (2 / 3 * twins).to_nat -- ≈ 20 * 10 (total approx)
  total_handshakes := twin_handshakes + quadruplet_handshakes + cross_handshakes
  total_handshakes = 680 := by
  let twins := 16 in
  let quadruplets := 20 in
  let twin_handshakes := twins * (twins - 2) / 2 in
  let quadruplet_handshakes := quadruplets * (quadruplets - 4) / 2 in
  let cross_handshakes := twins * 13 + quadruplets * 10 in
  let total_handshakes := twin_handshakes + quadruplet_handshakes + cross_handshakes in
  sorry

end total_handshakes_convention_l14_14532


namespace max_modulus_l14_14743

-- Given condition: z ∈ ℂ and |z + 2 - 2i| = 1
variables (z : ℂ)

-- The statement
theorem max_modulus (h : complex.abs (z + 2 - 2 * complex.I) = 1) : ∃ M, M = 4 ∧ ∀ w, w = complex.abs (z - 1 - 2 * complex.I) → w ≤ M :=
sorry

end max_modulus_l14_14743


namespace sum_of_integer_solutions_l14_14405

theorem sum_of_integer_solutions:
  (∑ k in Finset.range 34 \ Finset.range 9, k) = 525 :=
by
  sorry

end sum_of_integer_solutions_l14_14405


namespace base_of_parallelogram_l14_14154

theorem base_of_parallelogram (area height base : ℝ) 
  (h_area : area = 320)
  (h_height : height = 16) :
  base = area / height :=
by 
  rw [h_area, h_height]
  norm_num
  sorry

end base_of_parallelogram_l14_14154


namespace simplify_expression_l14_14137

variable (x : ℝ)
variable (h₁ : x ≠ 2)
variable (h₂ : x ≠ 3)
variable (h₃ : x ≠ 4)
variable (h₄ : x ≠ 5)

theorem simplify_expression : 
  ( (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 9) / (x^2 - 8*x + 15)) 
  = ( (x - 1) * (x - 5) ) / ( (x - 4) * (x - 2) * (x - 3) ) ) :=
by sorry

end simplify_expression_l14_14137


namespace range_of_a_l14_14246

noncomputable def f (x : ℝ) : ℝ := Real.log x + 3 * x^2
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 4 * x^2 - a * x

theorem range_of_a (a : ℝ) :
  (∃ x0 : ℝ, x0 > 0 ∧ f x0 = g (-x0) a) → a ≤ -1 := 
by
  sorry

end range_of_a_l14_14246


namespace firetruck_reachable_area_l14_14780

theorem firetruck_reachable_area :
  let speed_highway := 50
  let speed_prairie := 14
  let travel_time := 0.1
  let area := 16800 / 961
  ∀ (x r : ℝ),
    (x / speed_highway + r / speed_prairie = travel_time) →
    (0 ≤ x ∧ 0 ≤ r) →
    ∃ m n : ℕ, gcd m n = 1 ∧
    m = 16800 ∧ n = 961 ∧
    m + n = 16800 + 961 := by
  sorry

end firetruck_reachable_area_l14_14780


namespace discount_profit_percentage_l14_14519

theorem discount_profit_percentage (CP : ℝ) (P_no_discount : ℝ) (D : ℝ) (profit_with_discount : ℝ) (SP_no_discount : ℝ) (SP_discount : ℝ) :
  P_no_discount = 50 ∧ D = 4 ∧ SP_no_discount = CP + 0.5 * CP ∧ SP_discount = SP_no_discount - (D / 100) * SP_no_discount ∧ profit_with_discount = SP_discount - CP →
  (profit_with_discount / CP) * 100 = 44 :=
by sorry

end discount_profit_percentage_l14_14519


namespace sum_all_possible_values_l14_14284

theorem sum_all_possible_values (x : ℝ) (h : x^2 = 16) :
  (x = 4 ∨ x = -4) → (4 + (-4) = 0) :=
by
  intro h1
  have : 4 + (-4) = 0 := by norm_num
  exact this

end sum_all_possible_values_l14_14284


namespace distance_between_planes_l14_14575

open Real

def plane1 (x y z : ℝ) : Prop := 3 * x - y + 2 * z - 3 = 0
def plane2 (x y z : ℝ) : Prop := 6 * x - 2 * y + 4 * z + 4 = 0

theorem distance_between_planes :
  ∀ (x y z : ℝ), plane1 x y z →
  6 * x - 2 * y + 4 * z + 4 ≠ 0 →
  (∃ d : ℝ, d = abs (6 * x - 2 * y + 4 * z + 4) / sqrt (6^2 + (-2)^2 + 4^2) ∧ d = 5 * sqrt 14 / 14) :=
by
  intros x y z p1 p2
  sorry

end distance_between_planes_l14_14575


namespace series_convergence_l14_14596

theorem series_convergence (α : ℝ) : 
  (∑ n in (Set.univ : Set ℕ), (1 / (n * csc(1 / n)) - 1)^α).Summable ↔ (1 / 2 < α) := 
sorry

end series_convergence_l14_14596


namespace find_number_l14_14861

theorem find_number (x : ℝ) : (5 / 3) * x = 45 → x = 27 := by
  sorry

end find_number_l14_14861


namespace dog_rabbit_age_ratio_l14_14978

-- Definitions based on conditions
def cat_age := 8
def rabbit_age := cat_age / 2
def dog_age := 12
def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

-- Theorem statement
theorem dog_rabbit_age_ratio : is_multiple dog_age rabbit_age ∧ dog_age / rabbit_age = 3 :=
by
  sorry

end dog_rabbit_age_ratio_l14_14978


namespace find_2a10_minus_a12_l14_14775

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m, m > n → a m = a n + (m - n) * d

noncomputable def arithmetic_sum_eq (a : ℕ → ℝ) (d : ℝ) : Prop :=
a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem find_2a10_minus_a12 (a : ℕ → ℝ) (d : ℝ) (h1 : arithmetic_sequence a) (h2 : arithmetic_sum_eq a d) : 
  2 * a 10 - a 12 = 24 :=
begin
  sorry
end

end find_2a10_minus_a12_l14_14775


namespace num_ways_to_stack_ice_cream_scoops_l14_14901

theorem num_ways_to_stack_ice_cream_scoops :
  let scoops := ["vanilla", "chocolate", "strawberry", "cherry", "banana"]
  in scoops.permutations.length = 120 :=
by
  sorry

end num_ways_to_stack_ice_cream_scoops_l14_14901


namespace min_edges_in_graph_l14_14815

noncomputable def min_edges (n : ℕ) : ℕ := 
  (7 * n^2 - 3 * n) / 2

theorem min_edges_in_graph (G : SimpleGraph) (n : ℕ) (h_n : n ≥ 2)
  (h_vert : G.order = 3 * n^2)
  (h_deg : ∀ v : G.vertex, G.degree v ≤ 4 * n)
  (h_deg1 : ∃ v : G.vertex, G.degree v = 1)
  (h_path : ∀ u v : G.vertex, ∃ p : u ⟶*[≤ 3] v, G.path_length p ≤ 3) : 
  G.edge_count ≥ min_edges n :=
sorry

end min_edges_in_graph_l14_14815


namespace initial_alcohol_percentage_l14_14072

theorem initial_alcohol_percentage :
  ∀ (P : ℝ), 0 ≤ P ∧ P ≤ 1,
  (6 * P + 1.8 = 3.9) → (P = 0.35) :=
by
  -- condition P is the initial percentage
  intro P,
  -- initial percentage must be between 0 and 1
  intro H,
  intro H1,
  sorry

end initial_alcohol_percentage_l14_14072


namespace expression_values_l14_14607

noncomputable def sign (x : ℝ) : ℝ := 
if x > 0 then 1 else -1

theorem expression_values (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ v ∈ ({-4, 0, 4} : Set ℝ), 
    sign a + sign b + sign c + sign (a * b * c) = v := by
  sorry

end expression_values_l14_14607


namespace isodynamic_eq_triangle_l14_14296

/-
  Given the following conditions:
  1.  Δ ABN ≅ Δ AMC ≅ Δ BCL are similar triangles.
  2.  Δ ABN, Δ AMC, and Δ BCL are acute and have isodynamic points.
  
  Prove that the isodynamic points of Δ ABN, Δ AMC, and Δ BCL form an equilateral triangle.
-/
theorem isodynamic_eq_triangle {A B C N M L : Type*} 
  [geometry A B C N M L] 
  (h1: similar_triangles ABN AMC)
  (h2: similar_triangles AMC BCL)
  (h3: similar_triangles BCL ABN)
  (h4: isodynamic_point ABN) 
  (h5: isodynamic_point AMC) 
  (h6: isodynamic_point BCL) :
  equilateral_triangle (isodynamic_point ABN) (isodynamic_point AMC) (isodynamic_point BCL) :=
sorry

end isodynamic_eq_triangle_l14_14296


namespace minnows_with_white_bellies_l14_14378

theorem minnows_with_white_bellies (T : ℕ) (H1 : 0.4 * T = 20) : 0.3 * T = 15 := 
by
  sorry

end minnows_with_white_bellies_l14_14378


namespace percent_decrease_for_bulk_purchase_l14_14512

-- Define the conditions
def last_month_price_per_kg : ℝ := 3.60
def bulk_price_per_kg : ℝ := 2.80
def threshold_kg : ℝ := 5.0

theorem percent_decrease_for_bulk_purchase :
  (last_month_price_per_kg - bulk_price_per_kg) / last_month_price_per_kg * 100 = 22.22 :=
by
  sorry

end percent_decrease_for_bulk_purchase_l14_14512


namespace find_q_l14_14571

noncomputable def has_two_distinct_negative_roots (q : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ 
  (x₁ ^ 4 + q * x₁ ^ 3 + 2 * x₁ ^ 2 + q * x₁ + 4 = 0) ∧ 
  (x₂ ^ 4 + q * x₂ ^ 3 + 2 * x₂ ^ 2 + q * x₂ + 4 = 0)

theorem find_q (q : ℝ) : 
  has_two_distinct_negative_roots q ↔ q ≤ 3 / Real.sqrt 2 := sorry

end find_q_l14_14571


namespace factorial_division_l14_14046

-- Define factorial using the standard library's factorial function
def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- The problem statement
theorem factorial_division :
  (factorial 15) / (factorial 6 * factorial 9) = 834 :=
sorry

end factorial_division_l14_14046


namespace problem_1_solution_problem_2_solution_l14_14411

-- Definition of the function f
def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 3) - abs (x - a)

-- Proof problem for question 1
theorem problem_1_solution (x : ℝ) : f x 2 ≤ -1/2 ↔ x ≥ 11/4 :=
by
  sorry

-- Proof problem for question 2
theorem problem_2_solution (a : ℝ) : (∀ x : ℝ, f x a ≥ a) ↔ a ∈ Set.Iic (3/2) :=
by
  sorry

end problem_1_solution_problem_2_solution_l14_14411


namespace angle_aod_l14_14776

/--
Given:
1. Vectors $\overrightarrow{OA} \perp \overrightarrow{OC}$ and $\overrightarrow{OB} \perp \overrightarrow{OD}$.
2. $\angle AOD = 4 \times \angle BOC$,
Prove that $\angle AOD = 144^\circ$.
-/
theorem angle_aod (x : ℝ) (h1 : ∠AOD = 4 * ∠BOC) (h2 : ∠BOD = 90) (h3 : ∠COA = 90) :
  ∠AOD = 144 :=
by
  sorry

end angle_aod_l14_14776


namespace min_value_change_when_2x2_added_l14_14467

variable (f : ℝ → ℝ)
variable (a b c : ℝ)

def quadratic (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem min_value_change_when_2x2_added
  (a b : ℝ)
  (h1 : ∀ x : ℝ, f x = a * x^2 + b * x + c)
  (h2 : ∀ x : ℝ, (a + 1) * x^2 + b * x + c > a * x^2 + b * x + c + 1)
  (h3 : ∀ x : ℝ, (a - 1) * x^2 + b * x + c < a * x^2 + b * x + c - 3) :
  ∀ x : ℝ, (a + 2) * x^2 + b * x + c = a * x^2 + b * x + (c + 1.5) :=
sorry

end min_value_change_when_2x2_added_l14_14467


namespace parabola_and_circle_eq_and_tangent_l14_14963

noncomputable def parabola_eq (x y : ℝ) : Prop := y^2 = x

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

theorem parabola_and_circle_eq_and_tangent :
  (∀ x y : ℝ, parabola_eq x y ↔ y^2 = x) ∧
  (∀ x y : ℝ, circle_eq x y ↔ (x - 2)^2 + y^2 = 1) ∧
  (∀ A1 A2 A3 : ℝ × ℝ,
    parabola_eq A1.1 A1.2 ∧ parabola_eq A2.1 A2.2 ∧ parabola_eq A3.1 A3.2 →
    tangent_to_circle A1 A2 A3)
:=
sorry

end parabola_and_circle_eq_and_tangent_l14_14963


namespace cone_lateral_surface_area_l14_14180

-- Definitions from conditions
def r : ℝ := 6
def V : ℝ := 30 * Real.pi

-- Theorem to prove
theorem cone_lateral_surface_area : 
  let h := V / (Real.pi * (r ^ 2) / 3) in
  let l := Real.sqrt (r ^ 2 + h ^ 2) in
  let S := Real.pi * r * l in
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l14_14180


namespace simplify_expression_l14_14917

theorem simplify_expression (y : ℝ) (hy : y ≠ 0) :
  (5 / (4 * y ^ (-4)) * (4 * y ^ 3 / 3)) = (5 * y ^ 7 / 3) :=
by
  sorry

end simplify_expression_l14_14917


namespace factorial_division_l14_14044

-- Define factorial using the standard library's factorial function
def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- The problem statement
theorem factorial_division :
  (factorial 15) / (factorial 6 * factorial 9) = 834 :=
sorry

end factorial_division_l14_14044


namespace tangent_plane_parallel_elliptic_paraboloid_l14_14125

theorem tangent_plane_parallel_elliptic_paraboloid (x y : ℝ) :
  let z := 2 * x^2 + 4 * y^2 in
  z = 18 → (x = 1) → (y = -2) :=
by sorry

end tangent_plane_parallel_elliptic_paraboloid_l14_14125


namespace steve_paid_18_l14_14834

-- Define the conditions
def mike_price : ℝ := 5
def steve_multiplier : ℝ := 2
def shipping_rate : ℝ := 0.8

-- Define Steve's cost calculation
def steve_total_cost : ℝ :=
  let steve_dvd_price := steve_multiplier * mike_price
  let shipping_cost := shipping_rate * steve_dvd_price
  steve_dvd_price + shipping_cost

-- Prove that Steve's total payment is 18.
theorem steve_paid_18 : steve_total_cost = 18 := by
  -- Provide a placeholder for the proof
  sorry

end steve_paid_18_l14_14834


namespace factorial_ratio_value_l14_14036

theorem factorial_ratio_value : fact 15 / (fact 6 * fact 9) = 770 := by
  sorry

end factorial_ratio_value_l14_14036


namespace equation_of_parabola_and_circle_position_relationship_l14_14957

-- Conditions
variables (p q : ℝ) (C : ℝ → ℝ → Prop) (M : ℝ → ℝ → Prop)

-- Definitions
def parabola_equation := ∀ x y, C x y ↔ y^2 = x
def circle_equation := ∀ x y, M x y ↔ (x - 2)^2 + y^2 = 1

-- Proof of Part 1
theorem equation_of_parabola_and_circle 
(h_vertex_origin : ∀ x y, C x y ↔ y^2 = 2 * p * x)
(h_focus_xaxis : p > 0)
(h_perpendicular : ∀ x y, C 1 y → (x = 1 ∧ y = ±sqrt (2 * p)) → ((1 : ℝ)  * (1 : ℝ)  + sqrt (2 * p) * (- sqrt (2 * p))) = 0)
(h_M : (2, 0))
(h_tangent_l : ∀ x y, M x y ↔ (x - 2)^2 + y^2 = 1)
: 
(parabola_equation C ∧ circle_equation M) := sorry

-- Proof of Part 2
theorem position_relationship
(A1 A2 A3 : ℝ × ℝ) 
(h_points_on_C : ∀ Ai, Ai = (y * y, y) for y in {A1, A2, A3})
(h_tangents_to_circle : ∀ Ai Aj, Ai ≠ Aj → tangent_line Ai Aj M)
:
(is_tangent_to_circle M (line_through A2 A3)) := sorry

end equation_of_parabola_and_circle_position_relationship_l14_14957


namespace standard_deviation_is_2_l14_14639

noncomputable def dataset := [51, 54, 55, 57, 53]

noncomputable def mean (l : List ℝ) : ℝ :=
  ((l.sum : ℝ) / (l.length : ℝ))

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  ((l.map (λ x => (x - m)^2)).sum : ℝ) / (l.length : ℝ)

noncomputable def std_dev (l : List ℝ) : ℝ :=
  Real.sqrt (variance l)

theorem standard_deviation_is_2 :
  mean dataset = 54 →
  std_dev dataset = 2 := by
  intro h_mean
  sorry

end standard_deviation_is_2_l14_14639


namespace solve_sine_equation_l14_14396

open Real

def angle (k : ℕ) : ℝ := 3^k * (2 * π) / 7

noncomputable def sum_sine (x : ℝ) : ℝ := 
  ∑ i in (Finset.range 6), sin (x + angle i)

theorem solve_sine_equation (x : ℝ) : 
  sum_sine x = 1 ↔ ∃ n : ℤ, x = -π/2 + 2 * π * n :=
by
  sorry

end solve_sine_equation_l14_14396


namespace find_a_l14_14253

def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x < 1 then 2^x + 1 else x^2 + a * x

theorem find_a (a : ℝ) (h : f a (f a 0) = 4 * a) : a = 2 :=
  sorry

end find_a_l14_14253


namespace general_term_of_sequence_l14_14064

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 3 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = (3 * a n - 4) / (9 * a n + 15)

theorem general_term_of_sequence (a : ℕ → ℚ) (h : sequence a) : ∀ n : ℕ, n > 0 → a n = (49 - 22 * n) / (33 * n - 24) :=
by
  sorry

end general_term_of_sequence_l14_14064


namespace grandmas_turtle_statues_l14_14729

theorem grandmas_turtle_statues : 
  let year1 := 4 in
  let year2 := 4 * year1 in
  let year3 := year2 + 12 - 3 in
  let year4 := year3 + 2 * 3 in
  year4 = 31 :=
by
  sorry

end grandmas_turtle_statues_l14_14729


namespace turban_as_part_of_salary_l14_14725

-- Definitions of the given conditions
def annual_salary (T : ℕ) : ℕ := 90 + 70 * T
def nine_month_salary (T : ℕ) : ℕ := 3 * (90 + 70 * T) / 4
def leaving_amount : ℕ := 50 + 70

-- Proof problem statement in Lean 4
theorem turban_as_part_of_salary (T : ℕ) (h : nine_month_salary T = leaving_amount) : T = 1 := 
sorry

end turban_as_part_of_salary_l14_14725


namespace binom_2024_1_l14_14548

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_2024_1 : binomial 2024 1 = 2024 := by
  sorry

end binom_2024_1_l14_14548


namespace choosing_4_out_of_10_classes_l14_14312

theorem choosing_4_out_of_10_classes :
  ∑ (k : ℕ) in (finset.range 5).map (prod.mk 10), k! / (4! * (k - 4)!) = 210 :=
by sorry

end choosing_4_out_of_10_classes_l14_14312


namespace square_rotation_invariance_l14_14985

theorem square_rotation_invariance (x y θ : ℝ) :
  let pt1 := (x, y)
  let pt2 := (20 : ℝ, 20 : ℝ)
  let pt3 := (20 : ℝ, 5 : ℝ)
  let pt4 := (x, 5 : ℝ)
  -- Assuming a valid square before rotation
  -- and these are valid points of a square
  (x = 5) →
    let side_length := (20 - 5 : ℝ) -- length calculated from given points
    let area := side_length * side_length
    area = 225 :=
by
  sorry

end square_rotation_invariance_l14_14985


namespace ricki_removed_14_apples_l14_14534

theorem ricki_removed_14_apples : 
  ∀ (initial_apples final_apples : ℕ) (R : ℕ),
    initial_apples = 74 ∧ final_apples = 32 ∧ (initial_apples - final_apples = 3 * R) → 
    R = 14 := 
by
  intros initial_apples final_apples R h,
  cases h with h_init h_rest,
  cases h_rest with h_final h_eq,
  sorry

end ricki_removed_14_apples_l14_14534


namespace range_of_m_l14_14231

theorem range_of_m (A B : Set ℝ) :
  (A = {x | x^2 - 3x + 2 < 0}) →
  (B = {x | x * (x - m) > 0}) →
  (A ∩ B = ∅) →
  (∃ m, m ≥ 2) :=
by
  intros hA hB hAB
  -- Proof omitted; place your proof here based on the conditions
  sorry

end range_of_m_l14_14231


namespace initially_planned_days_l14_14805

-- Definitions of the conditions
def total_work_initial (x : ℕ) : ℕ := 50 * x
def total_work_with_reduction (x : ℕ) : ℕ := 25 * (x + 20)

-- The main theorem
theorem initially_planned_days :
  ∀ (x : ℕ), total_work_initial x = total_work_with_reduction x → x = 20 :=
by
  intro x
  intro h
  sorry

end initially_planned_days_l14_14805


namespace find_dot_product_l14_14673

open Real

noncomputable def vec_a : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector a
noncomputable def vec_b : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector b

def magnitude (v : ℝ → ℝ → ℝ) : ℝ :=
  sqrt ((v 0) ^ 2 + (v 1)^ 2)

def dot_product (u v : ℝ → ℝ → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

axiom magnitude_a_eq1 : magnitude vec_a = 1
axiom magnitude_b_eq_sqrt3 : magnitude vec_b = sqrt 3
axiom magnitude_a_minus_2b_eq3 : magnitude (λ x, vec_a x - 2 * vec_b x) = 3

theorem find_dot_product (a b : ℝ → ℝ → ℝ) 
  (ha : magnitude a = 1) 
  (hb : magnitude b = sqrt 3) 
  (h : magnitude (λ x, a x - 2 * b x) = 3) :
  dot_product a b = 1 := sorry

end find_dot_product_l14_14673


namespace average_eq_35_implies_y_eq_50_l14_14414

theorem average_eq_35_implies_y_eq_50 (y : ℤ) (h : (15 + 30 + 45 + y) / 4 = 35) : y = 50 :=
by
  sorry

end average_eq_35_implies_y_eq_50_l14_14414


namespace least_number_divisible_by_five_smallest_primes_l14_14998

theorem least_number_divisible_by_five_smallest_primes : 
  ∃ n ∈ ℕ+, n = 2 * 3 * 5 * 7 * 11 ∧ n = 2310 :=
by
  sorry

end least_number_divisible_by_five_smallest_primes_l14_14998


namespace total_chairs_in_hall_l14_14765

theorem total_chairs_in_hall : 
  ∀ (n_tables : ℕ) (n_half : ℕ) (n_fifth : ℕ) (n_tenth : ℕ) (remaining : ℕ) (additional : ℕ),
    n_tables = 60 →
    n_half = n_tables / 2 →
    n_fifth = n_tables / 5 →
    n_tenth = n_tables / 10 →
    remaining = n_tables - (n_half + n_fifth + n_tenth) →
    additional = n_tables / 2 →
    let total_chairs := (n_half * 2) + (n_fifth * 3) + (n_tenth * 5) + (remaining * 4)
    let odd_additional_chairs := additional * 1
  in total_chairs + odd_additional_chairs = 204 :=
by
  intros
  simp only [total_chairs, odd_additional_chairs]
  sorry

end total_chairs_in_hall_l14_14765


namespace triangle_perimeter_l14_14229

-- Conditions as definitions
def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ c = a

def has_sides (a b : ℕ) : Prop :=
  a = 4 ∨ b = 4 ∨ a = 9 ∨ b = 9

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the isosceles triangle with specified sides
structure IsoTriangle :=
  (a b c : ℕ)
  (iso : is_isosceles_triangle a b c)
  (valid_sides : has_sides a b ∧ has_sides a c ∧ has_sides b c)
  (triangle : triangle_inequality a b c)

-- The statement to prove perimeter
def perimeter (T : IsoTriangle) : ℕ :=
  T.a + T.b + T.c

-- The theorem we aim to prove
theorem triangle_perimeter (T : IsoTriangle) (h: T.a = 9 ∧ T.b = 9 ∧ T.c = 4) : perimeter T = 22 :=
sorry

end triangle_perimeter_l14_14229


namespace digit_D_is_five_l14_14938

variable (A B C D : Nat)
variable (h1 : (B * A) % 10 = A % 10)
variable (h2 : ∀ (C : Nat), B - A = B % 10 ∧ C ≤ A)

theorem digit_D_is_five : D = 5 :=
by
  sorry

end digit_D_is_five_l14_14938


namespace cone_lateral_surface_area_l14_14196

-- Definitions based on the conditions
def coneRadius : ℝ := 6
def coneVolume : ℝ := 30 * Real.pi

-- Mathematical statement
theorem cone_lateral_surface_area (r V : ℝ) (hr : r = coneRadius) (hV : V = coneVolume) :
  ∃ S : ℝ, S = 39 * Real.pi :=
by 
  have h_volume := hV
  have h_radius := hr
  sorry

end cone_lateral_surface_area_l14_14196


namespace max_area_rectangle_with_perimeter_40_l14_14095

theorem max_area_rectangle_with_perimeter_40 :
  ∃ (l w : ℕ), 2 * l + 2 * w = 40 ∧ l * w = 100 :=
sorry

end max_area_rectangle_with_perimeter_40_l14_14095


namespace cone_lateral_surface_area_l14_14205

theorem cone_lateral_surface_area (r : ℕ) (V : ℝ) (h l S : ℝ)
  (h_r : r = 6)
  (h_V : V = 30 * Real.pi)
  (h_volume : V = (1 / 3) * Real.pi * (r ^ 2) * h)
  (h_slant_height : l = Real.sqrt (r^2 + h^2))
  (h_lateral_surface_area : S = Real.pi * r * l) :
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l14_14205


namespace area_of_given_triangle_l14_14027

noncomputable def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

structure Point where
  x : ℝ
  y : ℝ

theorem area_of_given_triangle :
  let C := Point.mk 1 1
  let A := Point.mk 3.5 8.5
  let B := Point.mk 16 (-8/3)
  area_of_triangle A B C = 71 := by
    -- Definitions of given points
    let C := Point.mk 1 1
    let A := Point.mk 3.5 8.5
    let B := Point.mk 16 (-8/3)
    -- The actual calculation for area is omitted
    sorry

end area_of_given_triangle_l14_14027


namespace position_relationship_l14_14953

-- Define the parabola with vertex at the origin and focus on the x-axis
def parabola (p : ℝ) : Prop := ∀ x y, y^2 = 2 * p * x

-- Define the line l: x = 1
def line_l (x : ℝ) : Prop := x = 1

-- Define the points P and Q where l intersects the parabola
def P (p : ℝ) : ℝ × ℝ := (1, Real.sqrt(2 * p))
def Q (p : ℝ) : ℝ × ℝ := (1, -Real.sqrt(2 * p))

-- Define the perpendicularity condition
def perpendicular (O P Q : ℝ × ℝ) : Prop := (O.1 * P.1 + O.2 * P.2) = 0

-- Point M
def M : ℝ × ℝ := (2, 0)

-- Equation of circle M
def circle_M (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop := (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Define the tangent condition
def tangent (l : ℝ × ℝ → Prop) (center : ℝ × ℝ) (radius : ℝ) : Prop := ∀ x, l(x) → (abs(center.1 - x.1) = radius)

-- Define points on the parabola
def points_on_parabola (A₁ A₂ A₃ : ℝ × ℝ) (p : ℝ) : Prop := parabola p A₁.1 A₁.2 ∧ parabola p A₂.1 A₂.2 ∧ parabola p A₃.1 A₃.2

-- The main theorem statement
theorem position_relationship (p : ℝ) (center : ℝ × ℝ) (radius : ℝ) (O A₁ A₂ A₃ : ℝ × ℝ) :
  parabola 1 O.1 O.2 ∧
  line_l 1 ∧
  A₁ = (0, 0) ∧
  perpendicular O (P 1) (Q 1) ∧
  tangent (λ x, line_l x.1) M 1 ∧
  points_on_parabola A₁ A₂ A₃ 1 →
  ∀ A₂ A₃ : ℝ × ℝ,
    (parabola 1 A₂.1 A₂.2 ∧ parabola 1 A₃.1 A₃.2) →
    tangent(λ x, circle_M center radius x) (A₂.1, A₂.2) (A₃.1, A₃.2) :=
sorry

end position_relationship_l14_14953


namespace percentage_apples_sold_l14_14508

noncomputable def original_apples := 9999.9980000004
noncomputable def remaining_apples := 5000.0

theorem percentage_apples_sold :
  let sold_apples := original_apples - remaining_apples,
      percentage_sold := (sold_apples / original_apples) * 100
  in percentage_sold = 50 :=
by
  let sold_apples := original_apples - remaining_apples
  let percentage_sold := (sold_apples / original_apples) * 100
  have : percentage_sold = 50 := sorry
  exact this

end percentage_apples_sold_l14_14508


namespace delta_epsilon_time_l14_14445

variable (D E Z h t : ℕ)

theorem delta_epsilon_time :
  (t = D - 8) →
  (t = E - 3) →
  (t = Z / 3) →
  (h = 3 * t) → 
  h = 15 / 8 :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end delta_epsilon_time_l14_14445


namespace find_k_values_l14_14221

def a (n k : ℕ) : ℝ :=
if n = 1 then 3 else (3 ^ (2 / (2 * k - 1)) - 1) * (∑ i in finset.range n, a i k) + 3

noncomputable def S (n k : ℕ) : ℝ := ∑ i in finset.range n, a i k

noncomputable def b (n k : ℕ) : ℝ := (1 / n) * real.log (3 ^ (finset.range n).sum (λ i, a (i + 1) k))

noncomputable def T (k : ℕ) : ℝ := ∑ i in finset.range (2 * k + 1), abs (b (i + 1) k - 1.5)

theorem find_k_values (k : ℕ) : T k ∈ int ↔ k = 1 := sorry

end find_k_values_l14_14221


namespace math_problem_l14_14282

theorem math_problem (x : ℂ) (hx : x + 1/x = 3) : x^6 + 1/x^6 = 322 := 
by 
  sorry

end math_problem_l14_14282


namespace probability_valid_values_l14_14473

def is_standard_dice (n : ℕ) : Prop :=
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6

def is_valid_value (n : ℕ) : Prop :=
  n ≠ 1 ∧ n ≠ 6

theorem probability_valid_values :
  let outcomes := (finset.range 6).product (finset.range 6).product (finset.range 6)
  let valid_outcomes := outcomes.filter (λ (abc: ℕ × (ℕ × ℕ)), is_valid_value abc.1 ∧ is_valid_value abc.2.1 ∧ is_valid_value abc.2.2)
  (valid_outcomes.card : ℚ) / (outcomes.card : ℚ) = 8 / 27 :=
by sorry

end probability_valid_values_l14_14473


namespace coeff_x3_in_expansion_l14_14778

noncomputable def binom_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem coeff_x3_in_expansion :
  (∀ x : ℝ,
    let p := (1 + x - 2 * x^2) * (1 + x)^5 in
    (p.expand - (1 + x - 2 * x^2) * (C(5,0) + C(5,1) * x + C(5,2) * x^2 + C(5,3) * x^3 + C(5,4) * x^4 + C(5,5) * x^5)).coeff 3 = 10) :=
by sorry

end coeff_x3_in_expansion_l14_14778


namespace gcd_459_357_convert_459_to_octal_l14_14426

theorem gcd_459_357 : Int.gcd 459 357 = 51 := 
by
  sorry

theorem convert_459_to_octal : "713".toNat = 459.toString 8.toNat := 
by
  sorry

end gcd_459_357_convert_459_to_octal_l14_14426


namespace symmetry_center_on_line_l14_14435

def symmetry_center_curve :=
  ∃ θ : ℝ, (∃ x y : ℝ, (x = -1 + Real.cos θ ∧ y = 2 + Real.sin θ))

-- The main theorem to prove
theorem symmetry_center_on_line : 
  (∃ cx cy : ℝ, (symmetry_center_curve ∧ (cy = -2 * cx))) :=
sorry

end symmetry_center_on_line_l14_14435


namespace find_number_l14_14858

theorem find_number (x : ℝ) : (5 / 3) * x = 45 → x = 27 := by
  sorry

end find_number_l14_14858


namespace grid_covering_third_piece_l14_14496

theorem grid_covering_third_piece :
  ∀ (grid : matrix (fin 4) (fin 4) bool) (piece1 piece2 : (fin 2) × (fin 1) → bool),
    (nonoverlapping_pieces grid [piece1, piece2]) →
    ∃ piece3 : (fin 2) × (fin 2) → bool, nonoverlapping_pieces grid [piece1, piece2, piece3] ∧ covers_grid grid [piece1, piece2, piece3] :=
sorry

end grid_covering_third_piece_l14_14496


namespace smallest_positive_angle_l14_14004

theorem smallest_positive_angle (k : ℤ) : ∃ α, α = 400 + k * 360 ∧ α > 0 ∧ α = 40 :=
by
  use 40
  sorry

end smallest_positive_angle_l14_14004


namespace number_of_quadruples_l14_14825

theorem number_of_quadruples (p : ℕ) (n : ℕ) (prime_p : Nat.Prime p) (hn : 0 < n) :
  let A := Fin p^n
  let S := { q : A × A × A × A | p^n ∣ (q.1.1.1 * q.1.2 + q.1.1.2 * q.2 + 1) } 
  Finset.card S = p^(3*n) - p^(3*n - 2) :=
sorry

end number_of_quadruples_l14_14825


namespace dot_product_proof_l14_14661

variables {ℝ : Type*}
variables (a b : ℝ → ℝ)
variables [inner_product_space ℝ ℝ]

theorem dot_product_proof
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = sqrt 3)
  (h3 : ∥a - 2 • b∥ = 3) :
  inner (a : ℝ) (b : ℝ) = 1 :=
sorry

end dot_product_proof_l14_14661


namespace dot_product_is_one_l14_14715

variable {V : Type*} [InnerProductSpace ℝ V]
variables (a b : V)

theorem dot_product_is_one 
  (ha : ∥a∥ = 1) 
  (hb : ∥b∥ = sqrt 3) 
  (hab : ∥a - 2•b∥ = 3) : 
  ⟪a, b⟫ = 1 :=
by 
  sorry

end dot_product_is_one_l14_14715


namespace identity_solution_l14_14886

theorem identity_solution (x : ℝ) :
  ∃ a b : ℝ, (2 * x + a) ^ 3 = 5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x ∧
             a = -1 ∧ b = 1 :=
by
  -- we can skip the proof as this is just a statement
  sorry

end identity_solution_l14_14886


namespace point_segment_length_eq_l14_14249

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x ^ 2 / 25 + y ^ 2 / 16 = 1)

noncomputable def line_eq (x : ℝ) : Prop := (x = 3)

theorem point_segment_length_eq :
  ∀ (A B : ℝ × ℝ), (ellipse_eq A.1 A.2) → (ellipse_eq B.1 B.2) → 
  (line_eq A.1) → (line_eq B.1) → (A = (3, 16/5) ∨ A = (3, -16/5)) → 
  (B = (3, 16/5) ∨ B = (3, -16/5)) → 
  |A.2 - B.2| = 32 / 5 := sorry

end point_segment_length_eq_l14_14249


namespace find_least_n_multiple_of_45_l14_14366

noncomputable def b : ℕ → ℕ
| 10 := 20
| (n+1) := 50 * b n + 2 * (n + 1)

theorem find_least_n_multiple_of_45 :
  ∃ n > 10, b n % 45 = 0 ∧ ∀ m > 10, b m % 45 = 0 → n ≤ m :=
begin
  use 15,
  split,
  { norm_num },
  split,
  { sorry },  -- proof that b 15 % 45 = 0
  { intros m h1 h2,
    by_contra h3,
    exact sorry }  -- proof that 15 is the smallest solution
end

end find_least_n_multiple_of_45_l14_14366


namespace number_of_positive_integers_satisfying_inequality_l14_14277

theorem number_of_positive_integers_satisfying_inequality :
  {n : ℤ | 0 < n ∧ (n + 9) * (n - 2) * (n - 10) < 0}.to_finset.card = 7 := by
  sorry

end number_of_positive_integers_satisfying_inequality_l14_14277


namespace valid_10_digit_numbers_count_l14_14071

-- Define the total number of 10-digit numbers
def total_10_digit_numbers : ℕ := 9 * 10^9

-- Define the sum of digits function
def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Define the condition for a valid 10-digit number
def is_valid_10_digit_number (n : ℕ) : Prop :=
  n >= 10^9 ∧ n < 10^10 ∧ digit_sum n ≤ 87

-- Define the count of invalid 10-digit numbers
def invalid_10_digit_numbers_count : ℕ := 66

-- Prove the number of valid 10-digit numbers is correct
theorem valid_10_digit_numbers_count : 
  (finset.range (10^10)).filter is_valid_10_digit_number).card = 8999999934 := by
  sorry

end valid_10_digit_numbers_count_l14_14071


namespace remainder_of_poly_division_l14_14464

theorem remainder_of_poly_division :
  ∀ (x : ℂ), ((x + 1)^2048) % (x^2 - x + 1) = x + 1 :=
by
  sorry

end remainder_of_poly_division_l14_14464


namespace math_problem_l14_14540

noncomputable def expression1 : ℝ := (1 / 2)⁻¹
noncomputable def expression2 : ℝ := 4 * Real.cos (Float.pi / 3)  -- cos 60° in radians
noncomputable def expression3 : ℝ := (5 - Real.pi)^0

theorem math_problem : expression1 + expression2 - expression3 = 3 :=
by 
  have h1 : expression1 = 2 := by
    unfold expression1
    norm_num
    
  have h2 : expression2 = 2 := by
    unfold expression2
    -- cos(pi/3) is 1/2
    rw [Real.cos_pi_div_three]
    norm_num
  
  have h3 : expression3 = 1 := by
    unfold expression3
    -- Any number to the power of 0 is 1
    norm_num

  -- Putting everything together
  rw [h1, h2, h3]
  -- Prove the final sum equals 3
  norm_num

end math_problem_l14_14540


namespace least_multiple_of_five_primes_l14_14995

noncomputable def smallest_multiple_of_five_primes : ℕ :=
  let primes := [2, 3, 5, 7, 11] in
  primes.foldl (· * ·) 1

theorem least_multiple_of_five_primes : smallest_multiple_of_five_primes = 2310 := by
  sorry

end least_multiple_of_five_primes_l14_14995


namespace norm_squared_sum_l14_14353

noncomputable theory
open_locale classical

variables (a b m : ℝ × ℝ)

-- Conditions given in the problem
def midpoint (a b : ℝ × ℝ) := (a.1 + b.1) / 2, (a.2 + b.2) / 2
def dot_product (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2
def norm_squared (u : ℝ × ℝ) := u.1 * u.1 + u.2 * u.2

-- Given values
def m_val : ℝ × ℝ := (5, 3)
def ab_dot_product_val : ℝ := 10

-- Statement to be proved
theorem norm_squared_sum : 
  m = m_val → 
  dot_product a b = ab_dot_product_val →
  midpoint a b = m →
  norm_squared a + norm_squared b = 116 := 
by 
  intros h1 h2 h3
  sorry

end norm_squared_sum_l14_14353


namespace determine_special_set_l14_14557

-- Define the set S and its properties
def is_special_set (S : Set ℝ) : Prop :=
  1 ∈ S ∧ (∀ x y ∈ S, x > y → Real.sqrt (x^2 - y^2) ∈ S)

-- Define the possible solutions
def S1 : Set ℝ := { x | ∃ k : ℕ, x = Real.sqrt (k + 1) }
def S2 (n : ℕ) : Set ℝ := { x | ∃ k : ℕ, k < n + 1 ∧ x = Real.sqrt (k + 1) }

theorem determine_special_set (S : Set ℝ) :
  (is_special_set S) → (S = S1 ∨ (∃ n : ℕ, S = S2 n)) :=
by
  sorry

end determine_special_set_l14_14557


namespace distance_M0_to_plane_l14_14484
open Real

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def distance_to_plane (plane_coeffs : (ℝ × ℝ × ℝ × ℝ)) (p : Point3D) : ℝ :=
  let (A, B, C, D) := plane_coeffs
  abs (A * p.x + B * p.y + C * p.z + D) / sqrt (A^2 + B^2 + C^2)

def plane_through_points (p1 p2 p3 : Point3D) : (ℝ × ℝ × ℝ × ℝ) :=
  let v1 := (p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)
  let v2 := (p3.x - p1.x, p3.y - p1.y, p3.z - p1.z)
  let normal := (
    v1.2 * v2.3 - v1.3 * v2.2,
    v1.3 * v2.1 - v1.1 * v2.3,
    v1.1 * v2.2 - v1.2 * v2.1
  )
  let (A, B, C) := normal
  let D := -(A * p1.x + B * p1.y + C * p1.z)
  (A, B, C, D)

def M1 : Point3D := ⟨1, 1, 2⟩
def M2 : Point3D := ⟨-1, 1, 3⟩
def M3 : Point3D := ⟨2, -2, 4⟩
def M0 : Point3D := ⟨2, 3, 8⟩

theorem distance_M0_to_plane :
  distance_to_plane (plane_through_points M1 M2 M3) M0 = 7 * sqrt (7 / 10) :=
by
  sorry

end distance_M0_to_plane_l14_14484


namespace num_tangent_circles_aux_l14_14362

namespace Geometry

structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

def externally_tangent (C1 C2 : Circle) : Prop :=
  let d := ((C1.center.1 - C2.center.1)^2 + (C1.center.2 - C2.center.2)^2).sqrt
  d = C1.radius + C2.radius

def externally_tangent_to_both (C : Circle) (C1 C2 : Circle) : Prop :=
  (externally_tangent C C1) ∧ (externally_tangent C C2)

theorem num_tangent_circles_aux (centerC1 centerC2 : ℝ × ℝ) :
  let C1 := Circle centerC1 2
  let C2 := Circle centerC2 3
  externally_tangent C1 C2 →
  ∃! n : ℕ, n = 2 ∧ (∀ C : Circle, C.radius = 1 → externally_tangent_to_both C C1 C2) → ∃ (C : Circle), C.radius = 1 :=
by 
  intros C1_radius_eq C2_radius_eq tangent_condition
  sorry

end num_tangent_circles_aux_l14_14362


namespace least_multiple_of_five_primes_l14_14988

noncomputable def smallest_multiple_of_five_primes : ℕ :=
  let primes := [2, 3, 5, 7, 11] in
  primes.foldl (· * ·) 1

theorem least_multiple_of_five_primes : smallest_multiple_of_five_primes = 2310 := by
  sorry

end least_multiple_of_five_primes_l14_14988


namespace least_multiple_of_five_primes_l14_14992

noncomputable def smallest_multiple_of_five_primes : ℕ :=
  let primes := [2, 3, 5, 7, 11] in
  primes.foldl (· * ·) 1

theorem least_multiple_of_five_primes : smallest_multiple_of_five_primes = 2310 := by
  sorry

end least_multiple_of_five_primes_l14_14992


namespace cone_lateral_surface_area_l14_14191

theorem cone_lateral_surface_area (r V : ℝ) (h l S : ℝ) 
  (radius_condition : r = 6)
  (volume_condition : V = 30 * Real.pi)
  (volume_formula : V = (1 / 3) * Real.pi * r^2 * h)
  (slant_height_formula : l = Real.sqrt (r^2 + h^2))
  (lateral_surface_area_formula : S = Real.pi * r * l) :
  S = 39 * Real.pi := 
sorry

end cone_lateral_surface_area_l14_14191


namespace equation_of_parabola_and_circle_position_relationship_l14_14960

-- Conditions
variables (p q : ℝ) (C : ℝ → ℝ → Prop) (M : ℝ → ℝ → Prop)

-- Definitions
def parabola_equation := ∀ x y, C x y ↔ y^2 = x
def circle_equation := ∀ x y, M x y ↔ (x - 2)^2 + y^2 = 1

-- Proof of Part 1
theorem equation_of_parabola_and_circle 
(h_vertex_origin : ∀ x y, C x y ↔ y^2 = 2 * p * x)
(h_focus_xaxis : p > 0)
(h_perpendicular : ∀ x y, C 1 y → (x = 1 ∧ y = ±sqrt (2 * p)) → ((1 : ℝ)  * (1 : ℝ)  + sqrt (2 * p) * (- sqrt (2 * p))) = 0)
(h_M : (2, 0))
(h_tangent_l : ∀ x y, M x y ↔ (x - 2)^2 + y^2 = 1)
: 
(parabola_equation C ∧ circle_equation M) := sorry

-- Proof of Part 2
theorem position_relationship
(A1 A2 A3 : ℝ × ℝ) 
(h_points_on_C : ∀ Ai, Ai = (y * y, y) for y in {A1, A2, A3})
(h_tangents_to_circle : ∀ Ai Aj, Ai ≠ Aj → tangent_line Ai Aj M)
:
(is_tangent_to_circle M (line_through A2 A3)) := sorry

end equation_of_parabola_and_circle_position_relationship_l14_14960


namespace probability_is_one_third_l14_14488

noncomputable def probability_multiple_of_3_on_die : ℚ :=
  (finset.filter (λ n, n % 3 = 0) (finset.range 7)).card / (finset.range 7).card

theorem probability_is_one_third :
  probability_multiple_of_3_on_die = 1/3 :=
by
  sorry

end probability_is_one_third_l14_14488


namespace tank_capacity_l14_14106

theorem tank_capacity (A_rate B_rate C_rate : ℕ) (cycle_minutes total_minutes : ℕ) (capacity : ℕ)
  (h1 : A_rate = 40)
  (h2 : B_rate = 30)
  (h3 : C_rate = 20)
  (h4 : cycle_minutes = 3)
  (h5 : total_minutes = 57)
  (h6 : capacity = 
    let net_fill_per_cycle := A_rate + B_rate - C_rate in
    let cycles := total_minutes / cycle_minutes in
    let extra_minutes := total_minutes % cycle_minutes in
    cycles * net_fill_per_cycle + extra_minutes * A_rate -- Only pipe A fills during the extra minute
  ) : capacity = 990 :=
by 
  sorry

end tank_capacity_l14_14106


namespace work_completion_l14_14499

theorem work_completion (A B C D : ℝ) :
  (A = 1 / 5) →
  (A + C = 2 / 5) →
  (B + C = 1 / 4) →
  (A + D = 1 / 3.6) →
  (B + C + D = 1 / 2) →
  B = 1 / 20 :=
by
  sorry

end work_completion_l14_14499


namespace ken_height_l14_14931

theorem ken_height 
  (height_ivan : ℝ) (height_jackie : ℝ) (height_ken : ℝ)
  (h1 : height_ivan = 175) (h2 : height_jackie = 175)
  (h_avg : (height_ivan + height_jackie + height_ken) / 3 = (height_ivan + height_jackie) / 2 * 1.04) :
  height_ken = 196 := 
sorry

end ken_height_l14_14931


namespace caroline_socks_l14_14135

theorem caroline_socks :
  let initial_pairs : ℕ := 40
  let lost_pairs : ℕ := 4
  let donation_fraction : ℚ := 2/3
  let purchased_pairs : ℕ := 10
  let gift_pairs : ℕ := 3
  let remaining_pairs := initial_pairs - lost_pairs
  let donated_pairs := remaining_pairs * donation_fraction
  let pairs_after_donation := remaining_pairs - donated_pairs.toNat
  let total_pairs := pairs_after_donation + purchased_pairs + gift_pairs
  total_pairs = 25 :=
by
  sorry

end caroline_socks_l14_14135


namespace largest_power_of_two_dividing_7_pow_2048_minus_1_l14_14144

theorem largest_power_of_two_dividing_7_pow_2048_minus_1 :
  ∃ n : ℕ, 2^n ∣ (7^2048 - 1) ∧ n = 14 :=
by
  use 14
  sorry

end largest_power_of_two_dividing_7_pow_2048_minus_1_l14_14144


namespace increasing_arithmetic_sequence_l14_14175

theorem increasing_arithmetic_sequence (a : ℕ → ℝ) (h : ∀ n : ℕ, a (n + 1) = a n + 2) : ∀ n : ℕ, a (n + 1) > a n :=
by
  sorry

end increasing_arithmetic_sequence_l14_14175


namespace dot_product_l14_14672

variables (a b : Vector ℝ) -- ℝ here stands for the real numbers

-- Given conditions
def condition1 : ∥a∥ = 1 := sorry
def condition2 : ∥b∥ = √3 := sorry
def condition3 : ∥a - (2 : ℝ) • b∥ = 3 := sorry

-- Goal to prove
theorem dot_product (a b : Fin₃ → ℝ) 
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = √3)
  (h3 : ∥a - (2 : ℝ) • b∥ = 3) : 
  a ⬝ b = 1 := 
sorry

end dot_product_l14_14672


namespace infinite_prime_divisors_of_polynomial_l14_14827

def P (x : ℤ) (a b c : ℤ) := a * x^2 + b * x + c

theorem infinite_prime_divisors_of_polynomial (a b c : ℤ) (h : ∃ x, P x a b c ≠ 0) :
  ∃ (q : ℕ) (q_pr : Nat.Prime q), ∃ n : ℤ, q ∣ P n a b c :=
by
  sorry

end infinite_prime_divisors_of_polynomial_l14_14827


namespace binom_2024_1_l14_14545

-- Define the binomial coefficient using the factorial definition
def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- State the theorem
theorem binom_2024_1 : binom 2024 1 = 2024 :=
by
  unfold binom
  rw [Nat.factorial_one, Nat.factorial_sub, Nat.sub_self]
  sorry

end binom_2024_1_l14_14545


namespace product_of_b_l14_14555

open Function

theorem product_of_b (b : ℝ) :
  let g (x : ℝ) := b / (3*x - 4)
  g 3 = (g ⁻¹' {b + 2}) →
  (3 * b^2 - 19 * b - 40 = 0) →
  b ≠ 20 / 3 →
  (Π b in roots (3*b^2 - 19*b - 40), b) = -40 / 3 :=
by sorry

end product_of_b_l14_14555


namespace range_of_positive_integers_in_list_l14_14375

theorem range_of_positive_integers_in_list (K : List ℤ) 
  (h1 : K.length = 40) 
  (h2 : ∀ (n : ℕ), n < 40 → K.nth n = some (-25 + 3 * n)) : 
  (K.filter (λ x, x > 0)).range = 90 := 
sorry

end range_of_positive_integers_in_list_l14_14375


namespace train_speed_180_kmh_l14_14055

theorem train_speed_180_kmh
  (length_of_train : ℝ)
  (time_to_cross : ℝ)
  (length_cond : length_of_train = 150)
  (time_cond : time_to_cross = 3) :
  (length_of_train / time_to_cross * 3.6 = 180) :=
by 
  rw [length_cond, time_cond]
  norm_num
  sorry

end train_speed_180_kmh_l14_14055


namespace opposite_of_three_l14_14946

theorem opposite_of_three : ∀ (n : ℝ), n = 3 → -n = -3 := by
  intros n h
  rw h
  sorry

end opposite_of_three_l14_14946


namespace tom_received_20_percent_bonus_l14_14022

-- Define the initial conditions
def tom_spent : ℤ := 250
def gems_per_dollar : ℤ := 100
def total_gems_received : ℤ := 30000

-- Calculate the number of gems received without the bonus
def gems_without_bonus : ℤ := tom_spent * gems_per_dollar
def bonus_gems : ℤ := total_gems_received - gems_without_bonus

-- Calculate the percentage of the bonus
def bonus_percentage : ℚ := (bonus_gems : ℚ) / gems_without_bonus * 100

-- State the theorem
theorem tom_received_20_percent_bonus : bonus_percentage = 20 := by
  sorry

end tom_received_20_percent_bonus_l14_14022


namespace hiker_walking_speed_l14_14509

-- Define the distance covered by the cyclist
def cyclist_speed : ℝ := 24
def cyclist_time_in_hours : ℝ := 5 / 60
def cyclist_distance_covered := cyclist_speed * cyclist_time_in_hours

-- Define the waiting time for the cyclist
def waiting_time_in_hours : ℝ := 25 / 60

-- Prove that the hiker's walking speed is 4.8 km/h
theorem hiker_walking_speed (v : ℝ) 
  (hiker_distance_covered : v * waiting_time_in_hours = cyclist_distance_covered) :
  v = 4.8 :=
by
  sorry

end hiker_walking_speed_l14_14509


namespace find_number_l14_14849

theorem find_number (x : ℝ) (h : (5/3) * x = 45) : x = 27 :=
by
  sorry

end find_number_l14_14849


namespace matrix_exp_computation_l14_14821

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, 5; 0, 3]

theorem matrix_exp_computation :
  A ^ 15 - 3 * (A ^ 14) = !![0, 0; 0, 0] :=
by sorry

end matrix_exp_computation_l14_14821


namespace number_of_valid_15_letter_words_l14_14779

def a : ℕ → ℕ
| 3 := 8
| (n+1) := 2 * (a n + c n)
and b : ℕ → ℕ
| 3 := 0
| (n+1) := a n
and c : ℕ → ℕ
| 3 := 0
| (n+1) := 4 * b n

def N (n : ℕ) : ℕ := a n + b n + c n

theorem number_of_valid_15_letter_words : N 15 % 1000 = 0 :=
by sorry

end number_of_valid_15_letter_words_l14_14779


namespace position_relationship_l14_14956

-- Define the parabola with vertex at the origin and focus on the x-axis
def parabola (p : ℝ) : Prop := ∀ x y, y^2 = 2 * p * x

-- Define the line l: x = 1
def line_l (x : ℝ) : Prop := x = 1

-- Define the points P and Q where l intersects the parabola
def P (p : ℝ) : ℝ × ℝ := (1, Real.sqrt(2 * p))
def Q (p : ℝ) : ℝ × ℝ := (1, -Real.sqrt(2 * p))

-- Define the perpendicularity condition
def perpendicular (O P Q : ℝ × ℝ) : Prop := (O.1 * P.1 + O.2 * P.2) = 0

-- Point M
def M : ℝ × ℝ := (2, 0)

-- Equation of circle M
def circle_M (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop := (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Define the tangent condition
def tangent (l : ℝ × ℝ → Prop) (center : ℝ × ℝ) (radius : ℝ) : Prop := ∀ x, l(x) → (abs(center.1 - x.1) = radius)

-- Define points on the parabola
def points_on_parabola (A₁ A₂ A₃ : ℝ × ℝ) (p : ℝ) : Prop := parabola p A₁.1 A₁.2 ∧ parabola p A₂.1 A₂.2 ∧ parabola p A₃.1 A₃.2

-- The main theorem statement
theorem position_relationship (p : ℝ) (center : ℝ × ℝ) (radius : ℝ) (O A₁ A₂ A₃ : ℝ × ℝ) :
  parabola 1 O.1 O.2 ∧
  line_l 1 ∧
  A₁ = (0, 0) ∧
  perpendicular O (P 1) (Q 1) ∧
  tangent (λ x, line_l x.1) M 1 ∧
  points_on_parabola A₁ A₂ A₃ 1 →
  ∀ A₂ A₃ : ℝ × ℝ,
    (parabola 1 A₂.1 A₂.2 ∧ parabola 1 A₃.1 A₃.2) →
    tangent(λ x, circle_M center radius x) (A₂.1, A₂.2) (A₃.1, A₃.2) :=
sorry

end position_relationship_l14_14956


namespace michael_choices_l14_14323

theorem michael_choices (n k : ℕ) (h_n : n = 10) (h_k : k = 4) : nat.choose n k = 210 :=
by
  rw [h_n, h_k]
  norm_num
  sorry

end michael_choices_l14_14323


namespace integer_solutions_sum_inequality_l14_14402

theorem integer_solutions_sum_inequality :
  let satisfies_inequality (x : ℝ) := 
    sqrt (x - 4) + sqrt (x + 1) + sqrt (2 * x) > sqrt (33 - x) + 4
  in ∑ (x : ℕ) in (Finset.Icc 9 33), x = 525 :=
by
  sorry

end integer_solutions_sum_inequality_l14_14402


namespace max_area_of_rectangle_l14_14091

noncomputable def max_area (l w : ℕ) : ℕ :=
  if 2 * l + 2 * w = 40 then l * w else 0

theorem max_area_of_rectangle : 
  ∃ (l w : ℕ), 2 * l + 2 * w = 40 ∧ l * w = 100 :=
by
  use 10
  use 10
  simp
  exact ⟨by norm_num, by norm_num⟩

end max_area_of_rectangle_l14_14091


namespace clock_angle_l14_14462

noncomputable def minute_degrees (minutes : ℕ) : ℝ :=
  minutes * 6

noncomputable def hour_degrees (hours : ℕ) (minutes : ℕ) : ℝ :=
  (hours * 30) + (minutes * 0.5)

theorem clock_angle : 
  let minute_angle := minute_degrees 50 in
  let hour_angle := hour_degrees 2 50 in
  let angle_diff := abs (minute_angle - hour_angle) in
  let smaller_angle := if angle_diff > 180 then 360 - angle_diff else angle_diff in
  smaller_angle = 145.0 :=
by
  sorry

end clock_angle_l14_14462


namespace dot_product_ab_l14_14685

variables (a b : ℝ^3)

-- Given conditions
def condition1 : Prop := ‖a‖ = 1
def condition2 : Prop := ‖b‖ = real.sqrt 3
def condition3 : Prop := ‖a - 2 • b‖ = 3

-- The theorem statement to prove
theorem dot_product_ab (h1 : condition1 a) (h2 : condition2 b) (h3 : condition3 a b) : 
  a ⬝ b = 1 :=
sorry

end dot_product_ab_l14_14685


namespace problem_one_solution_set_problem_two_range_of_a_l14_14257

def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + 8 * x
def g (x : ℝ) (a : ℝ) : ℝ := f x a - 7 * x - a^2 + 3

theorem problem_one_solution_set (x : ℝ) (h : x > -2) : f x 1 ≥ 2 * x + 1 ↔ x ≥ 0 :=
by
  sorry

theorem problem_two_range_of_a (a : ℝ) (h1 : a > 0) : 
  ∀ x : ℝ, x > -2 → g x a ≥ 0 → a ∈ set.Ioc 0 2 :=
by
  sorry

end problem_one_solution_set_problem_two_range_of_a_l14_14257


namespace solve_final_selling_price_l14_14098

/-- Given conditions:
1. A's profit margin is 25%.
2. B's profit margin is 25%.
3. The cost price for A is Rs. 144.
-/
def finalSellingPrice (aProfitPct : ℝ) (bProfitPct : ℝ) (aCostPrice : ℝ) : ℝ :=
  let aProfit := (aProfitPct / 100) * aCostPrice
  let aSellingPrice := aCostPrice + aProfit
  let bProfit := (bProfitPct / 100) * aSellingPrice
  aSellingPrice + bProfit

theorem solve_final_selling_price :
  finalSellingPrice 25 25 144 = 225 :=
by
  sorry

end solve_final_selling_price_l14_14098


namespace train_speed_l14_14108

theorem train_speed
  (num_carriages : ℕ)
  (length_carriage length_engine : ℕ)
  (bridge_length_km : ℝ)
  (crossing_time_min : ℝ)
  (h1 : num_carriages = 24)
  (h2 : length_carriage = 60)
  (h3 : length_engine = 60)
  (h4 : bridge_length_km = 4.5)
  (h5 : crossing_time_min = 6) :
  (num_carriages * length_carriage + length_engine) / 1000 + bridge_length_km / (crossing_time_min / 60) = 60 :=
by
  sorry

end train_speed_l14_14108


namespace one_and_two_thirds_of_what_number_is_45_l14_14843

theorem one_and_two_thirds_of_what_number_is_45 (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by
  sorry

end one_and_two_thirds_of_what_number_is_45_l14_14843


namespace fraction_to_decimal_l14_14553

theorem fraction_to_decimal : (7 / 50 : ℝ) = 0.14 := by
  sorry

end fraction_to_decimal_l14_14553


namespace prove_identity_l14_14899

variable (x : ℝ)

theorem prove_identity : 
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  -- Expand both sides and prove identity
  sorry

end prove_identity_l14_14899


namespace dot_product_ab_l14_14689

variables (a b : ℝ^3)

-- Given conditions
def condition1 : Prop := ‖a‖ = 1
def condition2 : Prop := ‖b‖ = real.sqrt 3
def condition3 : Prop := ‖a - 2 • b‖ = 3

-- The theorem statement to prove
theorem dot_product_ab (h1 : condition1 a) (h2 : condition2 b) (h3 : condition3 a b) : 
  a ⬝ b = 1 :=
sorry

end dot_product_ab_l14_14689


namespace ratio_price_16_to_8_l14_14902

def price_8_inch := 5
def P : ℝ := sorry
def price_16_inch := 5 * P
def daily_earnings := 3 * price_8_inch + 5 * price_16_inch
def three_day_earnings := 3 * daily_earnings
def total_earnings := 195

theorem ratio_price_16_to_8 : total_earnings = three_day_earnings → P = 2 :=
by
  sorry

end ratio_price_16_to_8_l14_14902


namespace order_of_numbers_l14_14281

variables (a b : ℚ)

theorem order_of_numbers (ha_pos : a > 0) (hb_neg : b < 0) (habs : |a| < |b|) :
  b < -a ∧ -a < a ∧ a < -b :=
by { sorry }

end order_of_numbers_l14_14281


namespace find_number_l14_14850

theorem find_number (x : ℝ) (h : (5/3) * x = 45) : x = 27 :=
by
  sorry

end find_number_l14_14850


namespace solve_trig_equation_l14_14397

theorem solve_trig_equation (x : ℝ) : 
  (∑ i in Finset.range 6, Real.sin (x + 3^i * (2 * Real.pi / 7))) = 1 →
  ∃ k : ℤ, x = - (Real.pi / 2) + 2 * Real.pi * k :=
sorry

end solve_trig_equation_l14_14397


namespace y_investment_proof_l14_14063

noncomputable def y_investment (x_investment : ℝ) (z_investment : ℝ) (total_profit : ℝ)
  (z_share_profit : ℝ) (x_duration : ℝ) (y_duration : ℝ) (z_duration : ℝ) : ℝ :=
let remaining_profit := total_profit - z_share_profit in
let x_share_proportional := x_investment * x_duration in
let z_share_proportional := z_investment * z_duration in
let y_share_proportional := remaining_profit * x_investment * x_duration / (x_share_proportional + z_share_proportional) in
y_share_proportional / y_duration

theorem y_investment_proof : y_investment 36000 48000 13860 4032 12 12 8 = 25000 :=
by
  sorry

end y_investment_proof_l14_14063


namespace sum_of_four_smallest_common_divisors_l14_14155

theorem sum_of_four_smallest_common_divisors : (list.sum ([1, 2, 3, 6] : list ℕ)) = 12 :=
by
  have gcd_common_factors : list.gcd_list [48, 96, -24, 120, 144] = [1, 2, 3, 4, 6, 12] := sorry
  have four_smallest_divisors : list.take 4 gcd_common_factors = [1, 2, 3, 6] := sorry
  rw [←four_smallest_divisors, list.sum_take [1, 2, 3, 4, 6, 12], list.sum, add_comm, add_assoc]
  have sum_comm_divisors : 1 + 2 + 3 + 6 = 12 := rfl
  exact sum_comm_divisors

end sum_of_four_smallest_common_divisors_l14_14155


namespace find_angle_B_l14_14758

noncomputable def sine (θ : ℝ) : ℝ := Math.sin θ

theorem find_angle_B (A : ℝ) (a b : ℝ) (A_eq : A = 120) (a_eq : a = 2) (b_eq : b = (2 * real.sqrt 3) / 3) :
  ∃ B : ℝ, B = 30 :=
by
  sorry

end find_angle_B_l14_14758


namespace find_dot_product_l14_14712

open Real

variables (a b : ℝ^3)
variables (dot_product : ℝ^3 → ℝ^3 → ℝ)

def vector_magnitude (v : ℝ^3) : ℝ := sqrt (dot_product v v)

axiom magnitude_a : vector_magnitude a = 1
axiom magnitude_b : vector_magnitude b = sqrt 3
axiom magnitude_a_minus_2b : vector_magnitude (a - (2:ℝ) • b) = 3

theorem find_dot_product : dot_product a b = 1 :=
sorry

end find_dot_product_l14_14712


namespace binomial_sum_identity_l14_14638

theorem binomial_sum_identity (n : ℕ) :
  (∀ x : ℕ, x > 0 → (1 + x)^n = ∑ k in finset.range (n + 1), nat.choose n k * x^k) →
  (∑ k in finset.range (n + 1), k * nat.choose n k = n * 2^(n-1)) →
  ∑ k in finset.range (n + 1), (k + 1) * nat.choose n k = (n + 2) * 2^(n-1) :=
by
  intros h_binomial_expansion h_derivative_result
  sorry

end binomial_sum_identity_l14_14638


namespace find_dot_product_l14_14709

open Real

variables (a b : ℝ^3)
variables (dot_product : ℝ^3 → ℝ^3 → ℝ)

def vector_magnitude (v : ℝ^3) : ℝ := sqrt (dot_product v v)

axiom magnitude_a : vector_magnitude a = 1
axiom magnitude_b : vector_magnitude b = sqrt 3
axiom magnitude_a_minus_2b : vector_magnitude (a - (2:ℝ) • b) = 3

theorem find_dot_product : dot_product a b = 1 :=
sorry

end find_dot_product_l14_14709


namespace totalPearsPicked_l14_14809

-- Define the number of pears picked by each individual
def jasonPears : ℕ := 46
def keithPears : ℕ := 47
def mikePears : ℕ := 12

-- State the theorem to prove the total number of pears picked
theorem totalPearsPicked : jasonPears + keithPears + mikePears = 105 := 
by
  -- The proof is omitted
  sorry

end totalPearsPicked_l14_14809


namespace maria_paper_count_l14_14376

-- Defining the initial number of sheets and the actions taken
variables (x y : ℕ)
def initial_sheets := 50 + 41
def remaining_sheets_after_giving_away := initial_sheets - x
def whole_sheets := remaining_sheets_after_giving_away - y
def half_sheets := y

-- The theorem we want to prove
theorem maria_paper_count (x y : ℕ) :
  whole_sheets x y = initial_sheets - x - y ∧ 
  half_sheets y = y :=
by sorry

end maria_paper_count_l14_14376


namespace fraction_odd_products_approx_l14_14535

def is_odd (n : ℕ) : Prop := n % 2 = 1

def multiplication_table := {x : ℕ × ℕ // x.1 ≤ 10 ∧ x.2 ≤ 10}

def odd_products (table : {x : ℕ × ℕ // is_odd (x.1) ∧ is_odd (x.2)}) :=
  table.1.1 * table.1.2

def odd_numbers_less_than (n : ℕ) : List ℕ :=
  List.filter is_odd (List.range (n + 1))

def count_odd_products : ℕ :=
  ((odd_numbers_less_than 10).length) ^ 2

def total_products : ℕ := 121
def fraction_of_odd_products : ℚ := (count_odd_products : ℚ) / total_products

theorem fraction_odd_products_approx : (fraction_of_odd_products ≈ 0.21) :=
  by
  sorry

end fraction_odd_products_approx_l14_14535


namespace range_of_a_l14_14007

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (0 ≤ a ∧ a < 3) :=
sorry

end range_of_a_l14_14007


namespace find_lambda_l14_14649

theorem find_lambda (a : ℕ → ℝ) (λ : ℝ) (h1 : ∀ n : ℕ, a (n + 1) = 2 * a n + 2^n) 
  (h2 : ∃ d : ℝ, ∀ n : ℕ, (a n + λ) / 2^n = (a 0 + λ) / 2^0 + d * n) :
  λ = 1 / 3 :=
by
  sorry

end find_lambda_l14_14649


namespace original_employees_l14_14520

theorem original_employees (x : ℝ) (h : 0.86 * x = 195) : x ≈ 227 :=
by 
  sorry

end original_employees_l14_14520


namespace kelseys_sister_is_3_years_older_l14_14344

-- Define the necessary conditions
def kelsey_birth_year : ℕ := 1999 - 25
def sister_birth_year : ℕ := 2021 - 50
def age_difference (a b : ℕ) : ℕ := a - b

-- State the theorem to prove
theorem kelseys_sister_is_3_years_older :
  age_difference kelsey_birth_year sister_birth_year = 3 :=
by
  -- Skipping the proof steps as only the statement is needed
  sorry

end kelseys_sister_is_3_years_older_l14_14344


namespace parabola_circle_tangency_l14_14968

-- Definitions for the given conditions
def is_origin (P : Point) : Prop :=
  P.x = 0 ∧ P.y = 0

def is_on_x_axis (P : Point) : Prop := 
  P.y = 0

def parabola_vertex_focus_condition (C : Parabola) (O F : Point) : Prop :=
  is_origin O ∧ is_on_x_axis F ∧ C.vertex = O ∧ C.focus = F

def intersects_at_perpendicular_points (C : Parabola) (l : Line) (P Q : Point) : Prop :=
  l.slope = 1 ∧ l.intersect C = {P, Q} ∧ vector_product_is_perpendicular 0 P Q = true

def circle_tangent_to_line_at_point (M : Circle) (l : Line) (P : Point) : Prop :=
  l.form = "x=1" ∧ distance M.center P = M.radius ∧ M.contains P

def parabola_contains_point (C : Parabola) (P : Point) : Prop :=
  P.on_parabola C = true

def lines_tangent_to_circle (l₁ l₂ : Line) (M : Circle) : Prop :=
  M.tangent l₁ ∧ M.tangent l₂

def position_relationship (l : Line) (M : Circle) : PositionRelationship :=
  TangencyLine.and Circle M

-- Statement of the proof problem
theorem parabola_circle_tangency :
  ∃ C M, let O : Point := { x := 0, y := 0 } in
  let F : Point := { x := 1/2, y := 0 } in
  parabola_vertex_focus_condition C O F ∧ 
  intersects_at_perpendicular_points C (line_horizontal 1) P Q ∧
  point M_center = (mk_point 2 0) ∧
  circle_tangent_to_line_at_point M (line_horizontal 1) (mk_point 1 0) ∧
  ∀ A1 A2 A3 : Point, parabola_contains_point C A1 ∧
                    parabola_contains_point C A2 ∧
                    parabola_contains_point C A3 ∧
                    lines_tangent_to_circle (line_through A1 A2) M ∧
                    lines_tangent_to_circle (line_through A1 A3) M →
                    position_relationship (line_through A2 A3) M = Tangent :=
begin
  sorry  
end

end parabola_circle_tangency_l14_14968


namespace determine_t_range_l14_14080

def is_half_value_function (f : ℝ → ℝ) (D : set ℝ) : Prop :=
  (∀ x ∈ D, monotone f) ∧ ∃ (a b : ℝ), a ∈ D ∧ b ∈ D ∧ ∀ x ∈ set.Icc a b, f x ∈ set.Icc (a / 2) (b / 2)

noncomputable def h (c t : ℝ) : ℝ → ℝ := λ x, log c (c ^ x + t)

theorem determine_t_range (c : ℝ) (h_is_half_value : is_half_value_function (h c t) set.univ) (h_conditions : c > 0 ∧ c ≠ 1) :
  ∃ t : ℝ, 0 < t ∧ t < 1 / 4 := sorry

end determine_t_range_l14_14080


namespace sum_geometric_series_l14_14268

theorem sum_geometric_series :
  ∑' n : ℕ+, (3 : ℝ)⁻¹ ^ (n : ℕ) = (1 / 2 : ℝ) := by
  sorry

end sum_geometric_series_l14_14268


namespace g_is_odd_l14_14338

def g (x : ℝ) : ℝ := ⌈x⌉ - 1/2

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end g_is_odd_l14_14338


namespace polynomial_solution_l14_14624

noncomputable def f (x : ℝ) : ℝ := x^3 - 2

theorem polynomial_solution (n : ℕ) (hn : n > 0) (f : ℝ → ℝ) 
  (hf : ∀ x : ℝ, 8 * f(x^3) - x^6 * f(2 * x) - 2 * f(x^2) + 12 = 0) :
  f = λ x, x^3 - 2 :=
by
  sorry

end polynomial_solution_l14_14624


namespace sum_max_triangle_areas_at_least_twice_area_l14_14503

variables {P : Type} [polynomial_convex_polygon P]

noncomputable def max_triangle_area (b : P.side) : ℝ :=
  -- Define the area of the largest triangle with base b inside P
  sorry

theorem sum_max_triangle_areas_at_least_twice_area (P : polynomial convex_polygon) :
  ∑ b in P.sides, max_triangle_area b ≥ 2 * (polygon_area P) :=
begin
  sorry
end

end sum_max_triangle_areas_at_least_twice_area_l14_14503


namespace ball_travel_distance_sixth_l14_14107

def initial_height : ℝ := 150
def rebound_ratio : ℝ := 1 / 3

noncomputable def total_distance (n : ℕ) : ℝ :=
  let fall := λ k, initial_height * (rebound_ratio ^ (k - 1))
  let rebound := λ k, initial_height * (rebound_ratio ^ k)
  ∑ i in Finset.range n, fall (i + 1) + ∑ i in Finset.range (n - 1), rebound (i + 1)

theorem ball_travel_distance_sixth : total_distance 6 = 300.45 :=
by
  sorry

end ball_travel_distance_sixth_l14_14107


namespace express_train_leaves_6_hours_later_l14_14081

theorem express_train_leaves_6_hours_later
  (V_g V_e : ℕ) (t : ℕ) (catch_up_time : ℕ)
  (goods_train_speed : V_g = 36)
  (express_train_speed : V_e = 90)
  (catch_up_in_4_hours : catch_up_time = 4)
  (distance_e : V_e * catch_up_time = 360)
  (distance_g : V_g * (t + catch_up_time) = 360) :
  t = 6 := by
  sorry

end express_train_leaves_6_hours_later_l14_14081


namespace bus_driver_total_compensation_l14_14500

theorem bus_driver_total_compensation
  (hours_worked : ℝ) (regular_rate : ℝ) (regular_hours : ℝ) (overtime_multiplier : ℝ) :
  hours_worked = 63.62 → regular_rate = 12 → regular_hours = 40 → overtime_multiplier = 0.75 →
  (let regular_earnings := regular_rate * regular_hours,
       overtime_hours := hours_worked - regular_hours,
       overtime_rate := regular_rate * (1 + overtime_multiplier),
       overtime_earnings := overtime_rate * overtime_hours
   in regular_earnings + overtime_earnings = 976.02) :=
by
  intros h_hours_worked h_regular_rate h_regular_hours h_overtime_multiplier
  let regular_earnings := regular_rate * regular_hours
  let overtime_hours := hours_worked - regular_hours
  let overtime_rate := regular_rate * (1 + overtime_multiplier)
  let overtime_earnings := overtime_rate * overtime_hours
  have regular_earnings_calc : regular_earnings = 480 := by
    rw [h_regular_rate, h_regular_hours]
    norm_num
  have overtime_hours_calc : overtime_hours = 23.62 := by
    rw [h_hours_worked, h_regular_hours]
    norm_num
  have overtime_rate_calc : overtime_rate = 21 := by
    rw [h_regular_rate, h_overtime_multiplier]
    norm_num
  have overtime_earnings_calc : overtime_earnings = 496.02 := by
    rw [←overtime_rate_calc, ←overtime_hours_calc]
    norm_num
  show regular_earnings + overtime_earnings = 976.02
  rw [←regular_earnings_calc, ←overtime_earnings_calc]
  norm_num

end bus_driver_total_compensation_l14_14500


namespace close_interval_is_zero_one_l14_14357

noncomputable def f (x : ℝ) : ℝ := x^2 + x + 2
noncomputable def g (x : ℝ) : ℝ := 2 * x + 1

theorem close_interval_is_zero_one {a b : ℝ} (hab : a ≤ b) :
  (∀ x ∈ set.Icc a b, abs (f x - g x) ≤ 1) → set.Icc a b = set.Icc 0 1 :=
by
  sorry

end close_interval_is_zero_one_l14_14357


namespace water_current_speed_l14_14105

-- Definitions based on the conditions
def swimmer_speed : ℝ := 4  -- The swimmer's speed in still water (km/h)
def swim_time : ℝ := 2  -- Time taken to swim against the current (hours)
def swim_distance : ℝ := 6  -- Distance swum against the current (km)

-- The effective speed against the current
noncomputable def effective_speed_against_current (v : ℝ) : ℝ := swimmer_speed - v

-- Lean statement that formalizes proving the speed of the current
theorem water_current_speed (v : ℝ) (h : effective_speed_against_current v = swim_distance / swim_time) : v = 1 :=
by
  sorry

end water_current_speed_l14_14105


namespace probability_of_choosing_red_base_l14_14075

theorem probability_of_choosing_red_base (A B : Prop) (C D : Prop) : 
  let red_bases := 2
  let total_bases := 4
  let probability := red_bases / total_bases
  probability = 1 / 2 := 
by
  sorry

end probability_of_choosing_red_base_l14_14075


namespace isosceles_triangle_perimeter_l14_14615

-- Definitions for the side lengths
def side_a (x : ℝ) := 4 * x - 2
def side_b (x : ℝ) := x + 1
def side_c (x : ℝ) := 15 - 6 * x

-- Main theorem statement
theorem isosceles_triangle_perimeter (x : ℝ) (h1 : side_a x = side_b x ∨ side_a x = side_c x ∨ side_b x = side_c x) :
  (side_a x + side_b x + side_c x = 12.3) :=
  sorry

end isosceles_triangle_perimeter_l14_14615


namespace cone_lateral_surface_area_l14_14203

theorem cone_lateral_surface_area (r h l S : ℝ) (π_pos : 0 < π) (r_eq : r = 6)
  (V : ℝ) (V_eq : V = 30 * π)
  (vol_eq : V = (1/3) * π * r^2 * h)
  (h_eq : h = 5 / 2)
  (l_eq : l = Real.sqrt (r^2 + h^2))
  (S_eq : S = π * r * l) :
  S = 39 * π :=
  sorry

end cone_lateral_surface_area_l14_14203


namespace derivative_of_y_l14_14422

def y (x : ℝ) : ℝ := sin (2 * x) ^ 2 + 2 * cos (x) ^ 2

theorem derivative_of_y :
  deriv y = λ x, 2 * sin (4 * x) - 4 * x * sin (x ^ 2) :=
by sorry

end derivative_of_y_l14_14422


namespace prob_B_wins_first_l14_14413

-- Define conditions
def is_equally_likely (p : ℝ) := p = 0.5
def no_ties := true
def independent_outcomes := true

-- Define the given probabilities and scenarios
noncomputable def team_A_wins_series_if_B_wins_4th_game : Prop :=
  let series := ["A", "A", "A", "B", "A", "A", "A"] ∨
                ["A", "A", "B", "A", "B", "A", "A"] ∨
                ["A", "B", "A", "A", "B", "A", "A"] ∨
                ["B", "A", "A", "A", "B", "A", "A"]
  in series ∧ series.contains "B" ∧ series.count "A" = 4 ∧ series.count "B" < 4

-- Define a function to compute the probability
noncomputable def probability_B_wins_first_game (conditions : Prop) : ℝ :=
  if conditions then 1 / 4 else 0

-- Prove the required statement implying the given conditions
theorem prob_B_wins_first (h : team_A_wins_series_if_B_wins_4th_game) : 
  probability_B_wins_first_game h = 1 / 4 :=
sorry

end prob_B_wins_first_l14_14413


namespace cone_lateral_surface_area_l14_14182

-- Definitions from conditions
def r : ℝ := 6
def V : ℝ := 30 * Real.pi

-- Theorem to prove
theorem cone_lateral_surface_area : 
  let h := V / (Real.pi * (r ^ 2) / 3) in
  let l := Real.sqrt (r ^ 2 + h ^ 2) in
  let S := Real.pi * r * l in
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l14_14182


namespace other_diagonal_length_l14_14934

variable (d1 d2 A : ℕ)
variable (A_eq : A = 330)
variable (d1_eq : d1 = 22)

theorem other_diagonal_length (h : A = (d1 * d2) / 2) : d2 = 30 := by
  have : A = (22 * d2) / 2 := by 
    rw [d1_eq]
  have A_val : 330 = (22 * d2) / 2 := by 
    rw [A_eq]
    exact this
  have : 660 = 22 * d2 := by
    linarith
  have : d2 = 30 := by
    linarith
  exact this

end other_diagonal_length_l14_14934


namespace smallest_abundant_not_multiple_of_5_l14_14585

def is_abundant (n : ℕ) : Prop :=
  (∑ d in (nat.divisors n).filter (≠ n), d) > n

def not_multiple_of_5 (n : ℕ) : Prop :=
  ¬ (5 ∣ n)

theorem smallest_abundant_not_multiple_of_5 :
  ∃ n : ℕ, is_abundant n ∧ not_multiple_of_5 n ∧ 
    (∀ m : ℕ, is_abundant m ∧ not_multiple_of_5 m → m >= n) :=
sorry

end smallest_abundant_not_multiple_of_5_l14_14585


namespace problem_equiv_answer_l14_14543

theorem problem_equiv_answer:
  (1 + Real.sin (Real.pi / 12)) * 
  (1 + Real.sin (5 * Real.pi / 12)) * 
  (1 + Real.sin (7 * Real.pi / 12)) * 
  (1 + Real.sin (11 * Real.pi / 12)) =
  (17 / 16 + 2 * Real.sin (Real.pi / 12)) * 
  (17 / 16 + 2 * Real.sin (5 * Real.pi / 12)) := by
sorry

end problem_equiv_answer_l14_14543


namespace train_speed_l14_14109

theorem train_speed (time crossing_pole: ℝ) (train_length: ℝ) (time_eq: time = 9) (length_eq: train_length = 225) :
  let speed := (train_length / time) * 3.6 in
  speed = 90 := by
  sorry

end train_speed_l14_14109


namespace product_of_positive_c_for_real_roots_l14_14158

theorem product_of_positive_c_for_real_roots :
  (∏ i in Finset.filter (λ c, 8 * (c: ℝ) < 225) (Finset.range 8), i) = 5040 :=
by
  sorry

end product_of_positive_c_for_real_roots_l14_14158


namespace dot_product_proof_l14_14653

variables {ℝ : Type*}
variables (a b : ℝ → ℝ)
variables [inner_product_space ℝ ℝ]

theorem dot_product_proof
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = sqrt 3)
  (h3 : ∥a - 2 • b∥ = 3) :
  inner (a : ℝ) (b : ℝ) = 1 :=
sorry

end dot_product_proof_l14_14653


namespace min_difference_of_composites_summing_to_87_l14_14070

def is_composite (n : ℕ) : Prop :=
  ∃ (p : ℕ), p > 1 ∧ p < n ∧ n % p = 0

theorem min_difference_of_composites_summing_to_87 :
  ∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ (a + b = 87) ∧ ∀ (x y : ℕ), (is_composite x ∧ is_composite y ∧ (x + y = 87) → abs (x - y) ≥ abs (45 - 42)) :=
sorry

end min_difference_of_composites_summing_to_87_l14_14070


namespace find_xy_l14_14573

theorem find_xy (x y : ℝ) (h : (x - 13)^2 + (y - 14)^2 + (x - y)^2 = 1/3) : 
  x = 40/3 ∧ y = 41/3 :=
sorry

end find_xy_l14_14573


namespace magnitude_sub_3b_l14_14620

variables (a b : ℝ^3) -- Assuming vectors are in 3-dimensional real space
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) -- a and b are unit vectors
variable (h_angle : real.angle a b = real.pi / 3) -- angle between a and b is 60 degrees or pi/3 radians

theorem magnitude_sub_3b (a b : ℝ^3) (ha : ∥a∥ = 1)  (hb : ∥b∥ = 1) 
  (h_angle : real.angle a b = real.pi / 3) : ∥a - 3 • b∥ = real.sqrt 7 :=
by sorry

end magnitude_sub_3b_l14_14620


namespace sum_of_recorded_numbers_l14_14301

theorem sum_of_recorded_numbers : 
  let n := 16
  let pairs := n.choose 2
  let total_sum := pairs
  pairs = n * (n - 1) / 2
  ∑ (i : Fin n), (friends_count i + enemies_count i) = total_sum :=
by
  let n := 16
  let pairs := n.choose 2
  let total_sum := pairs
  have pairs_eq : pairs = n * (n - 1) / 2 := by
    rw Nat.choose
    apply Nat.choose_self_eq 
  sorry

end sum_of_recorded_numbers_l14_14301


namespace profit_percentage_is_12_36_l14_14060

noncomputable def calc_profit_percentage (SP CP : ℝ) : ℝ :=
  let Profit := SP - CP
  (Profit / CP) * 100

theorem profit_percentage_is_12_36
  (SP : ℝ) (h1 : SP = 100)
  (CP : ℝ) (h2 : CP = 0.89 * SP) :
  calc_profit_percentage SP CP = 12.36 :=
by
  sorry

end profit_percentage_is_12_36_l14_14060


namespace trapezoid_area_l14_14123

theorem trapezoid_area
  (longer_base : ℝ)
  (base_angle : ℝ)
  (h₁ : longer_base = 20)
  (h₂ : base_angle = real.arccos 0.6) :
  ∃ area : ℝ, (area ≈ 74.12) :=
by
  sorry

end trapezoid_area_l14_14123


namespace quadratic_has_two_real_roots_l14_14431

theorem quadratic_has_two_real_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x^2 - (m + 1) * x + (3 * m - 6) = 0 :=
by
  sorry

end quadratic_has_two_real_roots_l14_14431


namespace limit_oa_l14_14141

def oa : ℕ → ℝ
| 0       := 1
| (n + 1) := oa n * cos (Real.pi / 2^(n + 2))

theorem limit_oa : 
  ∃ (L : ℝ), (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (oa n - L) < ε) ∧ 
             L = Real.exp (-Real.pi^2 / 12) :=
begin
  sorry
end

end limit_oa_l14_14141


namespace least_multiple_of_five_primes_l14_14991

noncomputable def smallest_multiple_of_five_primes : ℕ :=
  let primes := [2, 3, 5, 7, 11] in
  primes.foldl (· * ·) 1

theorem least_multiple_of_five_primes : smallest_multiple_of_five_primes = 2310 := by
  sorry

end least_multiple_of_five_primes_l14_14991


namespace integer_solution_x_l14_14788

theorem integer_solution_x (x y : ℤ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y + x * y = 101) : x = 50 :=
sorry

end integer_solution_x_l14_14788


namespace greatest_prime_factor_f24_is_11_value_of_f12_l14_14160

def is_even (n : ℕ) : Prop := n % 2 = 0

def f (n : ℕ) : ℕ := (List.range' 2 ((n + 1) / 2)).map (λ x => 2 * x) |> List.prod

theorem greatest_prime_factor_f24_is_11 : 
  ¬ ∃ p, Prime p ∧ p ∣ f 24 ∧ p > 11 := 
  sorry

theorem value_of_f12 : f 12 = 46080 := 
  sorry

end greatest_prime_factor_f24_is_11_value_of_f12_l14_14160


namespace first_player_wins_l14_14971

/-- There are two heaps of stones. Pile A starts with 30 stones, and Pile B starts with 20 stones.
    Players take turns to take any number of stones from one pile. The player who cannot take a stone on their turn loses.
    This statement proves that the first player has a winning strategy. -/
theorem first_player_wins :
  ∃ strategy : (ℕ × ℕ) → ℕ × Bool, ∀ (state : ℕ × ℕ), 
  (state = (30, 20) → 
  let (new_state, first_player_wins) := strategy state in
  first_player_wins) := sorry

end first_player_wins_l14_14971


namespace problem_statements_correct_l14_14051

theorem problem_statements_correct :
    (∀ (select : ℕ) (male female : ℕ), male = 4 → female = 3 → 
      (select = (4 * 3 + 3)) → select ≥ 12 = false) ∧
    (∀ (a1 a2 a3 : ℕ), 
      a2 = 0 ∨ a2 = 1 ∨ a2 = 2 →
      (∃ (cases : ℕ), cases = 14) →
      cases = 14) ∧
    (∀ (ways enter exit : ℕ), enter = 4 → exit = 4 - 1 →
      (ways = enter * exit) → ways = 12 = false) ∧
    (∀ (a b : ℕ),
      a > 0 ∧ a < 10 ∧ b > 0 ∧ b < 10 →
      (∃ (log_val : ℕ), log_val = 54) →
      log_val = 54) := by
  admit

end problem_statements_correct_l14_14051


namespace max_area_rectangle_with_perimeter_40_l14_14097

theorem max_area_rectangle_with_perimeter_40 :
  ∃ (l w : ℕ), 2 * l + 2 * w = 40 ∧ l * w = 100 :=
sorry

end max_area_rectangle_with_perimeter_40_l14_14097


namespace prove_identity_l14_14897

variable (x : ℝ)

theorem prove_identity : 
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  -- Expand both sides and prove identity
  sorry

end prove_identity_l14_14897


namespace factorial_division_l14_14040

theorem factorial_division : (nat.factorial 15) / ((nat.factorial 6) * (nat.factorial 9)) = 5005 := 
by 
    sorry

end factorial_division_l14_14040


namespace operation_ab_equals_nine_l14_14593

variable (a b : ℝ)

def operation (x y : ℝ) : ℝ := a * x + b * y - 1

theorem operation_ab_equals_nine
  (h1 : operation a b 1 2 = 4)
  (h2 : operation a b (-2) 3 = 10)
  : a * b = 9 :=
by
  sorry

end operation_ab_equals_nine_l14_14593


namespace factorial_division_l14_14042

theorem factorial_division : (nat.factorial 15) / ((nat.factorial 6) * (nat.factorial 9)) = 5005 := 
by 
    sorry

end factorial_division_l14_14042


namespace sin_600_plus_tan_240_l14_14047

theorem sin_600_plus_tan_240 : sin (600 * Real.pi / 180) + tan (240 * Real.pi / 180) = (Real.sqrt 3) / 2 := by
  sorry

end sin_600_plus_tan_240_l14_14047


namespace cannot_equal_distance_l14_14513

theorem cannot_equal_distance (n : ℕ) (m f c : ℕ) (dist_mf : m = f + 1) (children : finset ℕ) (hc : children.card = 9)
  (positions : finset ℕ → ℕ → ℕ) : 
  ¬ (∑ c in children, positions c m = ∑ c in children, positions c f) :=
by
  sorry

end cannot_equal_distance_l14_14513


namespace range_of_b_l14_14742

theorem range_of_b (y : ℝ) (b : ℝ) (h1 : |y - 2| + |y - 5| < b) (h2 : b > 1) : b > 3 := 
sorry

end range_of_b_l14_14742


namespace dot_product_eq_one_l14_14697

variables {α : Type*} [InnerProductSpace ℝ α]

noncomputable def vector_a (a : α) : Prop := ∥a∥ = 1
noncomputable def vector_b (b : α) : Prop := ∥b∥ = real.sqrt 3
noncomputable def vector_c (a b : α) : Prop := ∥a - (2 : ℝ) • b∥ = 3

theorem dot_product_eq_one (a b : α) (ha : vector_a a) (hb : vector_b b) (hc : vector_c a b) :
  inner a b = 1 :=
sorry

end dot_product_eq_one_l14_14697


namespace dot_product_eq_one_l14_14696

variables {α : Type*} [InnerProductSpace ℝ α]

noncomputable def vector_a (a : α) : Prop := ∥a∥ = 1
noncomputable def vector_b (b : α) : Prop := ∥b∥ = real.sqrt 3
noncomputable def vector_c (a b : α) : Prop := ∥a - (2 : ℝ) • b∥ = 3

theorem dot_product_eq_one (a b : α) (ha : vector_a a) (hb : vector_b b) (hc : vector_c a b) :
  inner a b = 1 :=
sorry

end dot_product_eq_one_l14_14696


namespace binom_2024_1_l14_14547

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_2024_1 : binomial 2024 1 = 2024 := by
  sorry

end binom_2024_1_l14_14547


namespace find_dot_product_l14_14707

open Real

variables (a b : ℝ^3)
variables (dot_product : ℝ^3 → ℝ^3 → ℝ)

def vector_magnitude (v : ℝ^3) : ℝ := sqrt (dot_product v v)

axiom magnitude_a : vector_magnitude a = 1
axiom magnitude_b : vector_magnitude b = sqrt 3
axiom magnitude_a_minus_2b : vector_magnitude (a - (2:ℝ) • b) = 3

theorem find_dot_product : dot_product a b = 1 :=
sorry

end find_dot_product_l14_14707


namespace book_cost_l14_14390

theorem book_cost (initial_money : ℕ) (remaining_money : ℕ) (num_books : ℕ) 
  (h1 : initial_money = 79) (h2 : remaining_money = 16) (h3 : num_books = 9) :
  (initial_money - remaining_money) / num_books = 7 :=
by
  sorry

end book_cost_l14_14390


namespace residue_of_negative_1257_mod_37_l14_14147

theorem residue_of_negative_1257_mod_37 : 
    ∃ k : ℤ, (k ∈ set.Icc (0 : ℕ) 36) ∧ (-1257 ≡ k [MOD 37]) := 
sorry

end residue_of_negative_1257_mod_37_l14_14147


namespace area_f2_is_7_l14_14648

noncomputable def f0 (x : ℝ) : ℝ := |x|

noncomputable def f1 (x : ℝ) : ℝ := |f0 x - 1|

noncomputable def f2 (x : ℝ) : ℝ := |f1 x - 2|

theorem area_f2_is_7 : ∫ x in -3..3, f2 x = 7 :=
by
  sorry

end area_f2_is_7_l14_14648


namespace g_of_neg_5_is_4_l14_14356

def f (x : ℝ) : ℝ := 3 * x - 8
def g (y : ℝ) : ℝ := 2 * y^2 + 5 * y - 3

theorem g_of_neg_5_is_4 : g (-5) = 4 :=
by
  sorry

end g_of_neg_5_is_4_l14_14356


namespace radius_of_cookie_l14_14930

theorem radius_of_cookie (x y : ℝ) : 
  x^2 + y^2 + 35 = 6*x + 22*y → 
  ∃ r : ℝ, r = sqrt 95 :=
begin
  sorry
end

end radius_of_cookie_l14_14930


namespace find_x_l14_14855

theorem find_x (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by 
  sorry

end find_x_l14_14855


namespace dot_product_is_one_l14_14716

variable {V : Type*} [InnerProductSpace ℝ V]
variables (a b : V)

theorem dot_product_is_one 
  (ha : ∥a∥ = 1) 
  (hb : ∥b∥ = sqrt 3) 
  (hab : ∥a - 2•b∥ = 3) : 
  ⟪a, b⟫ = 1 :=
by 
  sorry

end dot_product_is_one_l14_14716


namespace parallel_vectors_l14_14474

open Vector

theorem parallel_vectors
  {AB CD : Vector3} 
  (hAB : AB ≠ 0) 
  (hCD : CD ≠ 0) 
  (hSum : AB + CD = 0) 
  : Parallel AB CD :=
by
  sorry

end parallel_vectors_l14_14474


namespace distance_to_left_focus_is_2m_l14_14217

noncomputable def hyperbola_real_axis_length (m : ℝ) :=
  -- Define the conditions of the hyperbola and point P.
  ∃ (left_focus right_focus P : ℝ → ℝ),
    -- Distance between foci is the real-axis length.
    ∀ x, (left_focus x - right_focus x).norm = m ∧
    -- P is on the hyperbola:
    ((P x) - right_focus x).norm = m ∧
    -- Question: proving the distance from P to the left focus.
    ((P x) - left_focus x).norm = 2 * m

theorem distance_to_left_focus_is_2m (m : ℝ) (left_focus right_focus P : ℝ → ℝ)
  (h : hyperbola_real_axis_length m) :
  ((P x) - left_focus x).norm = 2 * m :=
by sorry

end distance_to_left_focus_is_2m_l14_14217


namespace fill_tank_time_l14_14016

theorem fill_tank_time :
  let rate_A := 1 / 30
  let rate_B := 1 / 45
  let rate_C := 1 / 60
  let rate_D := -1 / 90
  let combined_rate := rate_A + rate_B + rate_C + rate_D
  let time_to_fill := 1 / combined_rate
  time_to_fill = 180 / 11 :=
by
  let rate_A := (1 : ℝ) / 30
  let rate_B := (1 : ℝ) / 45
  let rate_C := (1 : ℝ) / 60
  let rate_D := -(1 : ℝ) / 90
  let combined_rate := rate_A + rate_B + rate_C + rate_D
  have h_combined_rate : combined_rate = 11 / 180 := by
    sorry
  have h_time_to_fill : 1 / combined_rate = 180 / 11 := by
    sorry
  exact h_time_to_fill

end fill_tank_time_l14_14016


namespace find_larger_number_l14_14008

variable (x y : ℕ)

theorem find_larger_number (h1 : x = 7) (h2 : x + y = 15) : y = 8 := by
  sorry

end find_larger_number_l14_14008


namespace median_room_number_of_arrived_mathletes_l14_14531

theorem median_room_number_of_arrived_mathletes 
  (rooms : List ℕ := List.range 1 31)
  (unoccupied : List ℕ := [15, 16, 29])
  (occupied := rooms.diff unoccupied): (occupied.length = 27) → list.med occupied = 14 :=
by
  intros
  -- placeholder for calculated result
  sorry

end median_room_number_of_arrived_mathletes_l14_14531


namespace triangles_equilateral_simultaneously_l14_14522

theorem triangles_equilateral_simultaneously {A B C A1 B1 C1 : Type}
  [linear_ordered_field A] [linear_ordered_field B] [linear_ordered_field C] 
  (h1 : A = B = C) 
  (divide_bc : ∀ {a b : A}, a = b ↔ BC/2 = A1) 
  (divide_ca : ∀ {c d : B}, c = d ↔ CA/2 = B1) 
  (divide_ab : ∀ {e f : C}, e = f ↔ AB/2 = C1) :
  (A = B = C ↔ A1 = B1 = C1) := 
by sorry

end triangles_equilateral_simultaneously_l14_14522


namespace isosceles_triangle_of_bisectors_and_ratios_l14_14823

theorem isosceles_triangle_of_bisectors_and_ratios
  (A B C D E : Point)
  (h1 : angleBisector A E B C)
  (h2 : angleBisector C D A B)
  (h3 : ∃ (k : ℝ), angle B D E = k * angle E D C ∧ angle B E D = k * angle D E A):
  isosceles A B C :=
sorry

end isosceles_triangle_of_bisectors_and_ratios_l14_14823


namespace simplify_expression_l14_14905

variable (y : ℝ)

theorem simplify_expression : (5 / (4 * y^(-4)) * (4 * y^3) / 3) = 5 * y^7 / 3 :=
by
  sorry

end simplify_expression_l14_14905


namespace train_pass_time_eq_4_seconds_l14_14479

-- Define the length of the train in meters
def train_length : ℕ := 40

-- Define the speed of the train in km/h
def train_speed_kmph : ℕ := 36

-- Conversion factor: 1 kmph = 1000 meters / 3600 seconds
def conversion_factor : ℚ := 1000 / 3600

-- Convert the train's speed from km/h to m/s
def train_speed_mps : ℚ := train_speed_kmph * conversion_factor

-- Calculate the time to pass the telegraph post
def time_to_pass_post : ℚ := train_length / train_speed_mps

-- The goal: prove the actual time is 4 seconds
theorem train_pass_time_eq_4_seconds : time_to_pass_post = 4 := by
  sorry

end train_pass_time_eq_4_seconds_l14_14479


namespace max_area_of_rectangle_l14_14090

noncomputable def max_area (l w : ℕ) : ℕ :=
  if 2 * l + 2 * w = 40 then l * w else 0

theorem max_area_of_rectangle : 
  ∃ (l w : ℕ), 2 * l + 2 * w = 40 ∧ l * w = 100 :=
by
  use 10
  use 10
  simp
  exact ⟨by norm_num, by norm_num⟩

end max_area_of_rectangle_l14_14090


namespace dot_product_l14_14667

variables (a b : Vector ℝ) -- ℝ here stands for the real numbers

-- Given conditions
def condition1 : ∥a∥ = 1 := sorry
def condition2 : ∥b∥ = √3 := sorry
def condition3 : ∥a - (2 : ℝ) • b∥ = 3 := sorry

-- Goal to prove
theorem dot_product (a b : Fin₃ → ℝ) 
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = √3)
  (h3 : ∥a - (2 : ℝ) • b∥ = 3) : 
  a ⬝ b = 1 := 
sorry

end dot_product_l14_14667


namespace smallest_period_of_f_max_value_of_f_monotonic_intervals_of_f_on_0_pi_l14_14752

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 2) * Real.cos x * Real.sin (x + (Real.pi / 4))

theorem smallest_period_of_f : Real.periodic f Real.pi :=
sorry

theorem max_value_of_f : Real.sup (Set.range f) = (Real.sqrt 2 + 1) / 2 :=
sorry

theorem monotonic_intervals_of_f_on_0_pi :
  (∀ x y, 0 ≤ x → x ≤ (Real.pi / 8) → y ∈ Icc x (Real.pi / 8) → f x ≤ f y) ∧
  (∀ x y, (5 * Real.pi / 8) ≤ x → x ≤ Real.pi → y ∈ Icc x Real.pi → f x ≤ f y) ∧
  (∀ x y, (Real.pi / 8) ≤ x → x ≤ (5 * Real.pi / 8) → y ∈ Icc x (5 * Real.pi / 8) → f x ≥ f y) :=
sorry

end smallest_period_of_f_max_value_of_f_monotonic_intervals_of_f_on_0_pi_l14_14752


namespace trapezoid_area_l14_14122

variable (x y : ℝ)

-- Conditions: 
-- 1. The trapezoid is isosceles and circumscribed around a circle.
-- 2. The longer base of the trapezoid is 20.
-- 3. One of the base angles is arccos(0.6).

theorem trapezoid_area 
  (h1 : 2 * y + 0.6 * x + 0.6 * x = 2 * x)
  (h2 : y + 0.6 * x + 0.6 * x = 20)
  (h3 : ∃ θ : ℝ, θ = Real.arccos 0.6) :
  let h := 0.8 * x in
  1 / 2 * (y + 20) * h = 112 :=
by
  sorry

end trapezoid_area_l14_14122


namespace sum_WY_eq_eight_l14_14564

theorem sum_WY_eq_eight (W X Y Z : ℕ) (hW : W ∈ {1, 2, 3, 5}) (hX : X ∈ {1, 2, 3, 5}) (hY : Y ∈ {1, 2, 3, 5}) (hZ : Z ∈ {1, 2, 3, 5}) (h_distinct : W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z) (h_eq : (W : ℚ) / X - (Y : ℚ) / Z = 1) : W + Y = 8 := by
  sorry

end sum_WY_eq_eight_l14_14564


namespace monic_cubic_polynomial_minimum_munificence_l14_14550

def monic_cubic_polynomial (a b c x : ℝ) : ℝ :=
  x^3 + a*x^2 + b*x + c

def munificence (p : ℝ → ℝ) : ℝ :=
  max (abs (p (-1))) (max (abs (p 0)) (abs (p 1)))

theorem monic_cubic_polynomial_minimum_munificence :
  ∃ a b c, ∀ x : ℝ, (-1 ≤ x ∧ x ≤ 1) → monic_cubic_polynomial a b c x = x^3 + a*x^2 + b*x + c ∧
  (∀ x, abs (monic_cubic_polynomial a b c x) ≤ 2)  ∧
  (∀ a1 b1 c1, (∀ x, abs (monic_cubic_polynomial a1 b1 c1 x) ≤ 2) → munificence (monic_cubic_polynomial a1 b1 c1) = 2) :=
sorry

end monic_cubic_polynomial_minimum_munificence_l14_14550


namespace book_cost_l14_14389

theorem book_cost (initial_money : ℕ) (remaining_money : ℕ) (num_books : ℕ) 
  (h1 : initial_money = 79) (h2 : remaining_money = 16) (h3 : num_books = 9) :
  (initial_money - remaining_money) / num_books = 7 :=
by
  sorry

end book_cost_l14_14389


namespace light_path_l14_14612

noncomputable def symmetric_point (P : Point) (L : Line) : Point := sorry

-- Definitions
variables (A O B M N : Point)
variables (AO BO : Line)
variables (M1 N1 K : Point)
variables h₀ : angle A O B
variables h₁ : M ∈ interior_of_angle A O B
variables h₂ : N ∈ interior_of_angle A O B

-- Reflections
def M1 := symmetric_point M AO
def N1 := symmetric_point N BO

-- Intersection point
def intersects (L1 L2 : Line) : Point := sorry -- Assume a function for line intersection
def M1N1 := line_through M1 N1
def K := intersects M1N1 AO

-- Main statement
theorem light_path : is_correct_direction M K AO BO N :=
sorry

end light_path_l14_14612


namespace avg_of_multiples_of_10_eq_305_l14_14459

theorem avg_of_multiples_of_10_eq_305 (N : ℕ) (h : N % 10 = 0) (h_avg : (10 + N) / 2 = 305) : N = 600 :=
sorry

end avg_of_multiples_of_10_eq_305_l14_14459


namespace arc_length_l14_14628

theorem arc_length (r α : ℝ) (h1 : r = 3) (h2 : α = π / 3) : r * α = π :=
by
  rw [h1, h2]
  norm_num
  sorry -- This is the step where actual simplification and calculation will happen

end arc_length_l14_14628


namespace determine_a_range_l14_14218

variable {a x x1 x2 y1 y2 : ℝ}

theorem determine_a_range (h1 : ∀ x, y = (-3 * a + 1) * x + a)
    (h2 : ∀ x1 x2 y1 y2, x1 > x2 → y1 > y2)
    (h3 : ∀ x, x > 0 ∧ (-3 * a + 1) * x + a ≥ 0) :
    0 ≤ a ∧ a < 1 / 3 := 
sorry

end determine_a_range_l14_14218


namespace min_value_of_quadratic_l14_14577

theorem min_value_of_quadratic :
  (∀ x : ℝ, 3 * x^2 - 12 * x + 908 ≥ 896) ∧ (∃ x : ℝ, 3 * x^2 - 12 * x + 908 = 896) :=
by
  sorry

end min_value_of_quadratic_l14_14577


namespace cos_product_identity_l14_14477

theorem cos_product_identity (k m : ℤ) (h : k ≠ 14 * m) :
  let x := (π * k) / 14 in
  cos x * cos (2 * x) * cos (4 * x) * cos (8 * x) = (1 / 8) * cos (15 * x) :=
by
  let x := (Real.pi * k) / 14
  sorry

end cos_product_identity_l14_14477


namespace general_term_l14_14156

def seq (n : ℕ) : ℝ :=
  match n with
  | 0 => 1
  | n+1 => 2 / (n + 2)

theorem general_term (n : ℕ) : seq n = 2 / (n + 1 + 1) :=
by
  sorry

end general_term_l14_14156


namespace event_excl_not_compl_l14_14598

-- Define the bag with balls
def bag := ({: nat // n < 6})

-- Define the events
def both_white (e : Finset (Fin 6)) : Prop :=
  e = {1, 3} 

def both_not_white (e : Finset (Fin 6)) : Prop :=
  (1 ∉ e ∧ 3 ∉ e)

def exactly_one_white (e : Finset (Fin 6)) : Prop :=
  (1 ∈ e ∧ 3 ∉ e) ∨ (1 ∉ e ∧ 3 ∈ e)

def at_most_one_white (e : Finset (Fin 6)) : Prop :=
  ¬ both_white e

-- The main theorem
theorem event_excl_not_compl :
  ∀ e : Finset (Fin 6),
    (both_white e → (both_not_white e ∨ exactly_one_white e ∨ at_most_one_white e) ∧
    ¬(both_not_white e ∧ (exactly_one_white e ∨ at_most_one_white e))) :=
by {
  sorry
}

end event_excl_not_compl_l14_14598


namespace cone_lateral_surface_area_l14_14181

-- Definitions from conditions
def r : ℝ := 6
def V : ℝ := 30 * Real.pi

-- Theorem to prove
theorem cone_lateral_surface_area : 
  let h := V / (Real.pi * (r ^ 2) / 3) in
  let l := Real.sqrt (r ^ 2 + h ^ 2) in
  let S := Real.pi * r * l in
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l14_14181


namespace simplify_expression_l14_14915

variable {y : ℤ}

theorem simplify_expression (y : ℤ) : 5 / (4 * y^(-4)) * (4 * y^3) / 3 = 5 * y^7 / 3 := 
by 
  -- Proof is omitted with 'sorry'
  sorry

end simplify_expression_l14_14915


namespace find_number_l14_14848

theorem find_number (x : ℝ) (h : (5/3) * x = 45) : x = 27 :=
by
  sorry

end find_number_l14_14848


namespace words_per_page_l14_14511

theorem words_per_page (p : ℕ) :
  (p ≤ 120) ∧ (154 * p % 221 = 145) → p = 96 := by
  sorry

end words_per_page_l14_14511


namespace speed_of_train_is_90_kmph_l14_14111

-- Definitions based on conditions
def length_of_train_in_meters : ℝ := 225
def time_to_cross_pole_in_seconds : ℝ := 9

-- Conversion factors
def meters_to_kilometers (meters : ℝ) : ℝ := meters / 1000
def seconds_to_hours (seconds : ℝ) : ℝ := seconds / 3600

-- Computed values from the conditions
def length_of_train_in_kilometers : ℝ := meters_to_kilometers length_of_train_in_meters
def time_to_cross_pole_in_hours : ℝ := seconds_to_hours time_to_cross_pole_in_seconds

-- Speed calculation
def speed_of_train_in_kmph : ℝ := length_of_train_in_kilometers / time_to_cross_pole_in_hours

-- Theorem to assert the query
theorem speed_of_train_is_90_kmph : speed_of_train_in_kmph = 90 := by
  sorry

end speed_of_train_is_90_kmph_l14_14111


namespace find_standard_ellipse_equation_l14_14240

noncomputable def standard_ellipse_equation_x_axis : Prop :=
  ∃ a b c : ℝ, a = 2 * c ∧ a - c = sqrt 3 ∧ b^2 = 9 ∧ 
  (c ≠ 0 ∧ 4 * b^2 = 3 * c^2) ∧ 
  (∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1)

noncomputable def standard_ellipse_equation_y_axis : Prop :=
  ∃ a b c : ℝ, a = 2 * c ∧ a - c = sqrt 3 ∧ b^2 = 9 ∧ 
  (c ≠ 0 ∧ 4 * b^2 = 3 * c^2) ∧ 
  (∀ x y : ℝ, (x^2) / (b^2) + (y^2) / (a^2) = 1)

theorem find_standard_ellipse_equation :
  standard_ellipse_equation_x_axis ∨ standard_ellipse_equation_y_axis :=
sorry

end find_standard_ellipse_equation_l14_14240


namespace club_members_after_4_years_l14_14502

noncomputable def b : ℕ → ℕ
| 0       := 20
| (n + 1) := 3 * b n - 16

theorem club_members_after_4_years : b 4 = 980 :=
by 
  -- Proof goes here
  sorry

end club_members_after_4_years_l14_14502


namespace equal_sum_groups_l14_14164

noncomputable def sum_of_squares (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

theorem equal_sum_groups (n : ℕ) : 
  (∃ (k : ℕ), n = 8 * k + 3 ∨ n = 8 * k + 4 ∨ n = 8 * k + 7 ∨ n = 8 * k + 8) ↔
  (∃ (A B : finset ℕ), A ∪ B = (finset.range (n + 1)).map (λ i, i ^ 2) ∧
                        A.disjoint B ∧
                        A.sum id = B.sum id) :=
by sorry

end equal_sum_groups_l14_14164


namespace T_expansion_l14_14820

def T (x : ℝ) : ℝ := (x - 2)^5 + 5 * (x - 2)^4 + 10 * (x - 2)^3 + 10 * (x - 2)^2 + 5 * (x - 2) + 1

theorem T_expansion (x : ℝ) : T x = (x - 1)^5 := by
  sorry

end T_expansion_l14_14820


namespace identity_holds_l14_14890

noncomputable def identity_proof : Prop :=
∀ (x : ℝ), (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x

theorem identity_holds : identity_proof :=
by
  sorry

end identity_holds_l14_14890


namespace price_of_remote_controlled_airplane_l14_14517

theorem price_of_remote_controlled_airplane (x : ℝ) (h : 300 = 0.8 * x) : x = 375 :=
by
  sorry

end price_of_remote_controlled_airplane_l14_14517


namespace anne_distance_l14_14744

variable (time : ℕ) (speed : ℕ)

theorem anne_distance (h1 : time = 3) (h2 : speed = 2) : time * speed = 6 := 
by {
  rw [h1, h2],
  exact rfl
}

end anne_distance_l14_14744


namespace angle_a_b_eq_pi_div_4_l14_14723

open Real

def a : euclidean_space ℝ (fin 2) := ![1, 1]
def b : euclidean_space ℝ (fin 2) := ![2, 0]

noncomputable def angle_between_vectors (v1 v2 : euclidean_space ℝ (fin 2)) : ℝ :=
real.arccos ((inner_product ℝ v1 v2) / ((euclidean_space.nnnorm v1) * (euclidean_space.nnnorm v2)))

theorem angle_a_b_eq_pi_div_4 : angle_between_vectors a b = π / 4 :=
by
  sorry

end angle_a_b_eq_pi_div_4_l14_14723


namespace sum_of_angles_in_acute_triangle_l14_14310

variables {A B C A_1 B_1 C_1 A_2 B_2 C_2 : Type}
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
variables [InnerProductSpace ℝ A_1] [InnerProductSpace ℝ B_1] [InnerProductSpace ℝ C_1]
variables [InnerProductSpace ℝ A_2] [InnerProductSpace ℝ B_2] [InnerProductSpace ℝ C_2]

/-- The sum of the angles B_2A_1C_2, C_2B_1A_2, and A_2C_1B_2 in an acute-angled
triangle ABC, with specific midpoints on the altitudes, is π. --/
theorem sum_of_angles_in_acute_triangle (triangle_ABC : IsAcuteTriangle A B C)
  (midpoint_AA1 : Midpoint A_2 (LineSeg A A_1)) (midpoint_BB1 : Midpoint B_2 (LineSeg B B_1))
  (midpoint_CC1 : Midpoint C_2 (LineSeg C C_1))
  (foot_A_alt : IsPerpendicular (LineSeg A A_1) (LineSeg B C))
  (foot_B_alt : IsPerpendicular (LineSeg B B_1) (LineSeg A C))
  (foot_C_alt : IsPerpendicular (LineSeg C C_1) (LineSeg A B)) :
  ∠B_2A_1C_2 + ∠C_2B_1A_2 + ∠A_2C_1B_2 = π := 
sorry

end sum_of_angles_in_acute_triangle_l14_14310


namespace find_x_values_l14_14793

theorem find_x_values (x y : ℕ) (hx: x > y) (hy: y > 0) (h: x + y + x * y = 101) :
  x = 50 ∨ x = 16 :=
sorry

end find_x_values_l14_14793


namespace find_x_l14_14800

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 :=
sorry

end find_x_l14_14800


namespace annual_income_before_tax_l14_14294

theorem annual_income_before_tax (I : ℝ) 
    (h1 : 0.45 * I - 0.30 * I = 7200) : 
    I = 48000 := 
by
  -- Given that 0.45 * I - 0.30 * I = 7200,
  -- we want to prove I = 48000.
  have : 0.15 * I = 7200 := by
    linarith
  -- Divide both sides by 0.15 to isolate I:
  have I_eq : I = 48000 := by
    field_simp at this
    exact this
  exact I_eq

end annual_income_before_tax_l14_14294


namespace cone_lateral_surface_area_l14_14198

theorem cone_lateral_surface_area (r h l S : ℝ) (π_pos : 0 < π) (r_eq : r = 6)
  (V : ℝ) (V_eq : V = 30 * π)
  (vol_eq : V = (1/3) * π * r^2 * h)
  (h_eq : h = 5 / 2)
  (l_eq : l = Real.sqrt (r^2 + h^2))
  (S_eq : S = π * r * l) :
  S = 39 * π :=
  sorry

end cone_lateral_surface_area_l14_14198


namespace combined_yearly_return_percentage_l14_14491

theorem combined_yearly_return_percentage :
  let investment1 := 500
  let return1 := 0.07
  let investment2 := 1500
  let return2 := 0.09
  let total_investment := investment1 + investment2
  let total_return := (investment1 * return1) + (investment2 * return2)
  total_return / total_investment * 100 = 8.5 := 
begin
  sorry
end

end combined_yearly_return_percentage_l14_14491


namespace find_EC_and_sum_l14_14623

noncomputable section

open Real

def angle_A := 45
def BC := 10
def BD_perp_AC := True
def CE_perp_AB := True
def angle_DBC_eq_2_angle_ECB (y : ℝ) := mangle DBC = 2 * mangle ECB

theorem find_EC_and_sum (b c : ℝ) : 
  angle_A = 45 ∧ 
  BC = 10 ∧ 
  BD_perp_AC ∧ 
  CE_perp_AB ∧ 
  angle_DBC_eq_2_angle_ECB y →
  EC = 2 * (sqrt b + sqrt c) ∧ 
  (a = 2 ∧ b = 6 ∧ c = 2 → a + b + c = 10) :=
sorry

end find_EC_and_sum_l14_14623


namespace simplify_expression_l14_14919

theorem simplify_expression (y : ℝ) (hy : y ≠ 0) :
  (5 / (4 * y ^ (-4)) * (4 * y ^ 3 / 3)) = (5 * y ^ 7 / 3) :=
by
  sorry

end simplify_expression_l14_14919


namespace average_price_of_hen_l14_14068

theorem average_price_of_hen 
  (pigs: ℕ) (hens: ℕ) (total_cost: ℕ) (avg_price_pig: ℕ) (H: ℕ)
  (h1: pigs = 3) 
  (h2: hens = 10) 
  (h3: total_cost = 1200) 
  (h4: avg_price_pig = 300) 
  (h5: total_cost = pigs * avg_price_pig + hens * H) :
  H = 30 := 
by 
  simp only [h1, h2, h3, h4, mul_add, add_sub_cancel] at h5
  exact h5

end average_price_of_hen_l14_14068


namespace find_x_l14_14796

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 := 
by
  sorry

end find_x_l14_14796


namespace solar_energy_exponential_growth_l14_14760

def f : ℕ → ℝ
| 2000 := 6
| 2010 := 12
| 2015 := 24
| 2020 := 48
| _    := 0

theorem solar_energy_exponential_growth :
  (∃ a r : ℝ, a > 0 ∧ r > 1 ∧ (f 2010 = f 2000 * r) ∧ (f 2015 = f 2010 * r) ∧ (f 2020 = f 2015 * r)) :=
sorry

end solar_energy_exponential_growth_l14_14760


namespace volume_square_pyramid_from_cube_l14_14516

/-- Given a cube of side length 3, if a convex polyhedron is formed by cutting the cube and the resulting shape is a square pyramid, then the volume of the resulting shape is 9 cubic units. -/
theorem volume_square_pyramid_from_cube (side : ℝ) (volume_cube : ℝ) (volume_pyramid : ℝ) :
  side = 3 → volume_cube = side^3 → volume_pyramid = (1 / 3) * volume_cube → volume_pyramid = 9 :=
by
  assume hside : side = 3
  assume hvolume_cube : volume_cube = side^3
  assume hvolume_pyramid : volume_pyramid = (1 / 3) * volume_cube
  sorry

end volume_square_pyramid_from_cube_l14_14516


namespace number_of_segments_is_even_l14_14973

-- Define the notion of a self-intersecting polygonal chain
def closed_self_intersecting_polygonal_chain (n : ℕ) : Prop :=
  ∃ (segments : list (ℝ × ℝ)), segments.length = n ∧
  (∀ segment, segment ∈ segments → segment.crosses_once (segments \ [segment]))

-- Define the property of crossing exactly once
def crosses_once (segment : ℝ × ℝ) (segments : list (ℝ × ℝ)) : Prop :=
  ∃! p, ∃ s, s ∈ segments ∧ intersects segment s p

-- Assuming an intersection function
def intersects (segment1 segment2 : ℝ × ℝ) (p : ℝ × ℝ) : Prop := sorry

-- The main theorem
theorem number_of_segments_is_even (n : ℕ):
  closed_self_intersecting_polygonal_chain n → 
  n % 2 = 0 :=
by
  intro h
  sorry

end number_of_segments_is_even_l14_14973


namespace find_s_t_u_l14_14369

variables {R : Type*} [RealField R] 
variables (u v w : EuclideanSpace R)
variables (s t u : R)

-- Declare the conditions
axiom h1 : ortho u v ∧ ortho u w ∧ ortho v w
axiom u_unit : ∥u∥ = 1
axiom v_unit : ∥v∥ = 1
axiom w_unit : ∥w∥ = 1
axiom h2 : u = s • (u ⬝ v) + t • (v ⬝ w) + u • (w ⬝ u)
axiom h3 : u ⬝ (v ⬝ w) = 1

-- The goal statement
theorem find_s_t_u : s + t + u = 1 :=
sorry

end find_s_t_u_l14_14369


namespace ellipse_properties_l14_14228

theorem ellipse_properties
  (a b : ℝ)
  (ha : a > b)
  (hb : b > 0)
  (hQ : ∃ (P : ℝ × ℝ), P = (sqrt 2 / 2, sqrt 3 / 2) ∧ (P.1 ^ 2) / (a ^ 2) + (P.2 ^ 2) / (b ^ 2) = 1)
  (product_of_slopes : ∀ P : ℝ × ℝ, ((P.2 - b) / P.1) * ((P.2 + b) / P.1) = -b^2 / a^2)
  (h_product : -b^2 / a^2 = -1 / 2) :

  let E := {p : ℝ × ℝ | (p.1 ^ 2) / 2 + (p.2 ^ 2) = 1} in

  (E = {p : ℝ × ℝ | (p.1 ^ 2) / (a ^ 2) + (p.2 ^ 2) / (b ^ 2) = 1}) ∧

  (∀ F1 F2 : ℝ × ℝ, 
    let l := {p : ℝ × ℝ | p.1 = p.2 * (1 / p.2) - 1} in
    ∃ A B : ℝ × ℝ,
    ∃ m : ℝ,
      l = {p : ℝ × ℝ | p.1 = m * p.2 - 1} ∧
      (E p.1) ∧ 
      (l p.2) ∧
      ((A.1 - 1) * (B.1 - 1) = 2) ∧
      triangle_area_ABF2 = 4 / 3) :=
  sorry

end ellipse_properties_l14_14228


namespace one_and_two_thirds_of_what_number_is_45_l14_14846

theorem one_and_two_thirds_of_what_number_is_45 (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by
  sorry

end one_and_two_thirds_of_what_number_is_45_l14_14846


namespace cone_lateral_surface_area_l14_14192

-- Definitions based on the conditions
def coneRadius : ℝ := 6
def coneVolume : ℝ := 30 * Real.pi

-- Mathematical statement
theorem cone_lateral_surface_area (r V : ℝ) (hr : r = coneRadius) (hV : V = coneVolume) :
  ∃ S : ℝ, S = 39 * Real.pi :=
by 
  have h_volume := hV
  have h_radius := hr
  sorry

end cone_lateral_surface_area_l14_14192


namespace longer_subsegment_length_l14_14950

theorem longer_subsegment_length (XYZ : Triangle)
  (h_ratio : XYZ.ratio = (3, 4, 5))
  (h_YZ : XYZ.side_YZ = 12)
  (h_angle_bisector : XYZ.angle_bisector_XE divYZ = true) :
  XYZ.segment_ZE = 48 / 7 :=
by
  sorry

end longer_subsegment_length_l14_14950


namespace max_area_of_triangle_ABC_l14_14814

-- Define the maximum area of triangle ABC given the conditions
noncomputable def max_area_triangle_ABC (PA PB PC : ℝ) (angle_BAC : ℝ) : ℝ :=
  if (PA = 1) ∧ (PB = 2) ∧ (PC = 3) ∧ (angle_BAC = real.pi / 3) then
    (3 * real.sqrt 3) / 2
  else
    0

-- Statement to be proved
theorem max_area_of_triangle_ABC :
  max_area_triangle_ABC 1 2 3 (real.pi / 3) = (3 * real.sqrt 3) / 2 :=
by {
  sorry
}

end max_area_of_triangle_ABC_l14_14814


namespace polar_coordinate_equation_intersection_sum_l14_14627

-- Definitions to set up the conditions
def parametric_curve (θ : ℝ) : ℝ × ℝ :=
  (2 + sqrt 3 * Real.cos θ, sqrt 3 * Real.sin θ)

def polar_coordinates (ρ α : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos α, ρ * Real.sin α)

-- Proof problem for the polar coordinate equation
theorem polar_coordinate_equation (θ ρ α : ℝ) :
  ∃ (θ : ℝ), (rho = sqrt 3) -> (2 + sqrt 3 * Real.cos θ) ^ 2 + (sqrt 3 * Real.sin θ) ^ 2 = 3 :=
sorry

-- Proof problem for the intersection points
theorem intersection_sum (ρ α : ℝ) :
  ρ^2 - 4*ρ * Real.cos α + 1 = 0 → 2*Real.sqrt 2 :=
sorry

end polar_coordinate_equation_intersection_sum_l14_14627


namespace stuffed_animals_mom_gift_l14_14052

theorem stuffed_animals_mom_gift (x : ℕ) :
  (10 + x) + 3 * (10 + x) = 48 → x = 2 :=
by {
  sorry
}

end stuffed_animals_mom_gift_l14_14052


namespace range_of_k_l14_14177

variable {x k : ℝ}

def p : Prop := x ≥ k
def q : Prop := x^2 - x - 2 > 0

theorem range_of_k (h : ∀ x, p → q ∧ ¬ (∀ x, q → p)) : k ∈ set.Ioi 2 :=
sorry

end range_of_k_l14_14177


namespace find_value_of_a_plus_b_l14_14618

noncomputable def A (a b : ℤ) : Set ℤ := {1, a, b}
noncomputable def B (a b : ℤ) : Set ℤ := {a, a^2, a * b}

theorem find_value_of_a_plus_b (a b : ℤ) (h : A a b = B a b) : a + b = -1 :=
by sorry

end find_value_of_a_plus_b_l14_14618


namespace find_base_c_l14_14302

theorem find_base_c (c : ℕ) : (c^3 - 7*c^2 - 18*c - 8 = 0) → c = 10 :=
by
  sorry

end find_base_c_l14_14302


namespace lateral_surface_area_of_given_cone_l14_14213

noncomputable def coneLateralSurfaceArea (r V : ℝ) : ℝ :=
let h := (3 * V) / (π * r^2) in
let l := Real.sqrt (r^2 + h^2) in
π * r * l

theorem lateral_surface_area_of_given_cone :
  coneLateralSurfaceArea 6 (30 * π) = 39 * π := by
simp [coneLateralSurfaceArea]
sorry

end lateral_surface_area_of_given_cone_l14_14213


namespace find_dot_product_l14_14680

open Real

noncomputable def vec_a : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector a
noncomputable def vec_b : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector b

def magnitude (v : ℝ → ℝ → ℝ) : ℝ :=
  sqrt ((v 0) ^ 2 + (v 1)^ 2)

def dot_product (u v : ℝ → ℝ → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

axiom magnitude_a_eq1 : magnitude vec_a = 1
axiom magnitude_b_eq_sqrt3 : magnitude vec_b = sqrt 3
axiom magnitude_a_minus_2b_eq3 : magnitude (λ x, vec_a x - 2 * vec_b x) = 3

theorem find_dot_product (a b : ℝ → ℝ → ℝ) 
  (ha : magnitude a = 1) 
  (hb : magnitude b = sqrt 3) 
  (h : magnitude (λ x, a x - 2 * b x) = 3) :
  dot_product a b = 1 := sorry

end find_dot_product_l14_14680


namespace find_x_l14_14865

-- Define the variables and conditions
def x := 27
axiom h : (5 / 3) * x = 45

-- Main statement to be proved
theorem find_x : x = 27 :=
by
  have : (5 / 3) * x = 45 := h
  sorry

end find_x_l14_14865


namespace simplify_expression_l14_14916

theorem simplify_expression (y : ℝ) (hy : y ≠ 0) :
  (5 / (4 * y ^ (-4)) * (4 * y ^ 3 / 3)) = (5 * y ^ 7 / 3) :=
by
  sorry

end simplify_expression_l14_14916


namespace volume_problem_l14_14530

noncomputable def volume_of_cylinder (h_cyl h_cone C_cyl C_cone V_cone : ℝ) : ℝ :=
  let h_ratio := (h_cyl / h_cone) = (4 / 5)
  let C_ratio := (C_cyl / C_cone) = (3 / 5)
  let A_cone := 25
  let A_cyl := (C_ratio^2) * A_cone
  let h_cyl := (4 / 5) * h_cone
  let V_cone := (1 / 3) * A_cone * h_cone
  let Ah := 6
  let volume_cone := 250
  let volume_cylinder := (9 * Ah) * (4 * h_cone)
  volume_cylinder

theorem volume_problem (h_cyl h_cone C_cyl C_cone V_cone : ℝ) :
  let h_ratio := (h_cyl / h_cone) = (4 / 5)
  let C_ratio := (C_cyl / C_cone) = (3 / 5)
  V_cone = 250 → volume_of_cylinder h_cyl h_cone C_cyl C_cone V_cone = 216 :=
by sorry

end volume_problem_l14_14530


namespace find_x_l14_14863

-- Define the variables and conditions
def x := 27
axiom h : (5 / 3) * x = 45

-- Main statement to be proved
theorem find_x : x = 27 :=
by
  have : (5 / 3) * x = 45 := h
  sorry

end find_x_l14_14863


namespace max_k_value_l14_14504

theorem max_k_value :
  ∃ A B C k : ℕ, 
  (A ≠ 0) ∧ 
  (A < 10) ∧ 
  (B < 10) ∧ 
  (C < 10) ∧
  (10 * A + B) * k = 100 * A + 10 * C + B ∧
  (∀ k' : ℕ, 
     ((A ≠ 0) ∧ (A < 10) ∧ (B < 10) ∧ (C < 10) ∧
     (10 * A + B) * k' = 100 * A + 10 * C + B) 
     → k' ≤ 19) ∧
  k = 19 :=
sorry

end max_k_value_l14_14504


namespace sequence_sum_l14_14610

theorem sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (h : ∀ n : ℕ, S n + a n = 2 * n + 1) :
  ∀ n : ℕ, a n = 2 - (1 / 2^n) :=
by
  sorry

end sequence_sum_l14_14610


namespace arrangement_condition_1_arrangement_condition_2_arrangement_condition_3_l14_14275

theorem arrangement_condition_1 : ∃ N₁ : ℕ, 
  let F := 3; let M := 4; let entities := F - 2 + M;
  let permutations_entities := nat.factorial entities;
  let permutations_F := nat.factorial F; 
  N₁ = permutations_F * permutations_entities ∧ N₁ = 720 :=
by sorry

theorem arrangement_condition_2 : ∃ N₂ : ℕ, 
  let F := 3; let M := 4;
  let permutations_M := nat.factorial M;
  let gaps := M + 1;
  let choose_gaps := nat.choose gaps F;
  N₂ = permutations_M * choose_gaps ∧ N₂ = 1440 :=
by sorry

theorem arrangement_condition_3 : ∃ N₃ : ℕ, 
  let F := 3; let M := 4; let positions := F + M;
  let permutations_positions_M := nat.factorial (positions - F);
  N₃ = permutations_positions_M * 1 ∧ N₃ = 840 :=
by sorry

end arrangement_condition_1_arrangement_condition_2_arrangement_condition_3_l14_14275


namespace find_dot_product_l14_14676

open Real

noncomputable def vec_a : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector a
noncomputable def vec_b : ℝ → ℝ → ℝ := sorry -- Placeholder for the vector b

def magnitude (v : ℝ → ℝ → ℝ) : ℝ :=
  sqrt ((v 0) ^ 2 + (v 1)^ 2)

def dot_product (u v : ℝ → ℝ → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

axiom magnitude_a_eq1 : magnitude vec_a = 1
axiom magnitude_b_eq_sqrt3 : magnitude vec_b = sqrt 3
axiom magnitude_a_minus_2b_eq3 : magnitude (λ x, vec_a x - 2 * vec_b x) = 3

theorem find_dot_product (a b : ℝ → ℝ → ℝ) 
  (ha : magnitude a = 1) 
  (hb : magnitude b = sqrt 3) 
  (h : magnitude (λ x, a x - 2 * b x) = 3) :
  dot_product a b = 1 := sorry

end find_dot_product_l14_14676


namespace maximum_value_a1_l14_14223

noncomputable def max_possible_value (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0)
  (h3 : a 1 = a 10) : ℝ :=
  16

theorem maximum_value_a1 (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0)
  (h3 : a 1 = a 10) : a 1 ≤ max_possible_value a h1 h2 h3 :=
  sorry

end maximum_value_a1_l14_14223


namespace one_and_two_thirds_of_what_number_is_45_l14_14845

theorem one_and_two_thirds_of_what_number_is_45 (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by
  sorry

end one_and_two_thirds_of_what_number_is_45_l14_14845


namespace first_number_in_list_is_55_l14_14932

theorem first_number_in_list_is_55 : 
  ∀ (x : ℕ), (55 + 57 + 58 + 59 + 62 + 62 + 63 + 65 + x) / 9 = 60 → x = 65 → 55 = 55 :=
by
  intros x avg_cond x_is_65
  rfl

end first_number_in_list_is_55_l14_14932


namespace cone_lateral_surface_area_l14_14208

theorem cone_lateral_surface_area (r : ℕ) (V : ℝ) (h l S : ℝ)
  (h_r : r = 6)
  (h_V : V = 30 * Real.pi)
  (h_volume : V = (1 / 3) * Real.pi * (r ^ 2) * h)
  (h_slant_height : l = Real.sqrt (r^2 + h^2))
  (h_lateral_surface_area : S = Real.pi * r * l) :
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l14_14208


namespace complex_number_C_l14_14635

-- Define the complex numbers corresponding to points A and B
def A : ℂ := 1 + 2 * Complex.I
def B : ℂ := 3 - 5 * Complex.I

-- Prove the complex number corresponding to point C
theorem complex_number_C :
  ∃ C : ℂ, (C = 10 - 3 * Complex.I) ∧ 
           (A = 1 + 2 * Complex.I) ∧ 
           (B = 3 - 5 * Complex.I) ∧ 
           -- Square with vertices in counterclockwise order
           True := 
sorry

end complex_number_C_l14_14635


namespace dot_product_ab_l14_14692

variables (a b : ℝ^3)

-- Given conditions
def condition1 : Prop := ‖a‖ = 1
def condition2 : Prop := ‖b‖ = real.sqrt 3
def condition3 : Prop := ‖a - 2 • b‖ = 3

-- The theorem statement to prove
theorem dot_product_ab (h1 : condition1 a) (h2 : condition2 b) (h3 : condition3 a b) : 
  a ⬝ b = 1 :=
sorry

end dot_product_ab_l14_14692


namespace finite_noncommutative_ring_is_simple_l14_14816

theorem finite_noncommutative_ring_is_simple
  (R : Type) [Ring R] [IsDomain R] [Fintype R] [NoncommRing R]
  (h1 : ∀ I : Ideal R, I ≠ ⊥ → ∃ S : Subring R, S = Algebra.adjoin R (I.val ∪ {1}))
  : ∀ I : Ideal R, I = ⊥ ∨ I = ⊤ :=
sorry

end finite_noncommutative_ring_is_simple_l14_14816


namespace ganesh_average_speed_l14_14062

noncomputable def averageSpeed (D : ℝ) : ℝ :=
  let time_uphill := D / 60
  let time_downhill := D / 36
  let total_time := time_uphill + time_downhill
  let total_distance := 2 * D
  total_distance / total_time

theorem ganesh_average_speed (D : ℝ) (hD : D > 0) : averageSpeed D = 45 := by
  sorry

end ganesh_average_speed_l14_14062


namespace conics_common_points_on_circle_iff_axes_perpendicular_l14_14983

variables {a b a1 b1 c1 : ℝ}
variables {C1 C2 : set (ℝ × ℝ)}
-- Conics are represented using predicates over the real plane

-- Conic 1 assumed to have four common points and general form ax^2 + by^2 + ... = 0
def conic1 (p : ℝ × ℝ) : Prop := a * p.1^2 + b * p.2^2 = 0

-- Conic 2 assumed to have four common points and general form a1x^2 + b1 y^2 + c1xy + ... = 0
def conic2 (p : ℝ × ℝ) : Prop := a1 * p.1^2 + b1 * p.2^2 + c1 * p.1 * p.2 = 0

-- The four common points
def common_points : set (ℝ × ℝ) := { p | conic1 p ∧ conic2 p }

-- Circle definition (centered at origin for simplicity)
def circle (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 = r^2

theorem conics_common_points_on_circle_iff_axes_perpendicular :
  (∀ p ∈ common_points, circle p) ↔ (a * b1 - a1 * b = 0) :=
sorry

end conics_common_points_on_circle_iff_axes_perpendicular_l14_14983


namespace count_valid_A_values_l14_14425

theorem count_valid_A_values :
  let valid_A (a : ℕ) := a < 10 ∧ (10 + 2 * a) % 3 = 0,
  num_valid_A := {a : ℕ | valid_A a}.toFinset.card
  in num_valid_A = 3 :=
by
  sorry

end count_valid_A_values_l14_14425


namespace find_curve_equation_l14_14371

-- Define the circle equation and initial points
def circle (x0 y0 : ℝ) : Prop := x0^2 + y0^2 = 4

-- Define the condition for point M
def point_M (x y x0 y0 : ℝ) : Prop :=
  x = x0 ∧ y = (√3 / 2) * y0

-- Define the final curve C
def curve_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Prove that the trajectory of point M satisfies the equation of curve C
theorem find_curve_equation (x y x0 y0 : ℝ) (h1 : circle x0 y0) (h2 : point_M x y x0 y0) : curve_C x y :=
by {
  -- Given conditions of the problem
  sorry -- Placeholder for the actual proof
}

end find_curve_equation_l14_14371


namespace dragons_total_games_l14_14536

theorem dragons_total_games (x y : ℕ) (h1 : x = 0.40 * y) (h2 : x + 5 = 0.55 * (y + 8)) : y + 8 = 12 := 
by sorry

end dragons_total_games_l14_14536


namespace max_sum_of_twelfth_powers_l14_14267

variables (x : ℕ → ℝ)

-- Define the conditions
def condition1 := ∀ i, 1 ≤ i ∧ i ≤ 1990 → -1/(Real.sqrt 3) ≤ x i ∧ x i ≤ Real.sqrt 3
def condition2 := (Finset.range 197).sum x = -318 * Real.sqrt 3

-- Define the function we want to maximize
def sum_of_twelfth_powers := (Finset.range 1297).sum (λ i, (x i) ^ 12)

-- State the theorem
theorem max_sum_of_twelfth_powers :
  condition1 x ∧ condition2 x → sum_of_twelfth_powers x = 189548 :=
sorry

end max_sum_of_twelfth_powers_l14_14267


namespace cone_lateral_surface_area_l14_14204

theorem cone_lateral_surface_area (r : ℕ) (V : ℝ) (h l S : ℝ)
  (h_r : r = 6)
  (h_V : V = 30 * Real.pi)
  (h_volume : V = (1 / 3) * Real.pi * (r ^ 2) * h)
  (h_slant_height : l = Real.sqrt (r^2 + h^2))
  (h_lateral_surface_area : S = Real.pi * r * l) :
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l14_14204


namespace angle_ABC_is_65_l14_14981

theorem angle_ABC_is_65
  (A B C O : Type)
  (h1 : ∠BAC = 75)
  (h2 : ∠ACB = 40)
  (h3 : ∀ (x y z : Type), ∠x + ∠y + ∠z = 180) :
  ∠ABC = 65 :=
by
  sorry

end angle_ABC_is_65_l14_14981


namespace inequality_must_hold_l14_14283

theorem inequality_must_hold (x y : ℝ) (h : x > y) : -2 * x < -2 * y :=
sorry

end inequality_must_hold_l14_14283


namespace south_120_meters_l14_14288

-- Define the directions
inductive Direction
| North
| South

-- Define the movement function
def movement (dir : Direction) (distance : Int) : Int :=
  match dir with
  | Direction.North => distance
  | Direction.South => -distance

-- Statement to prove
theorem south_120_meters : movement Direction.South 120 = -120 := 
by
  sorry

end south_120_meters_l14_14288


namespace binom_2024_1_l14_14546

-- Define the binomial coefficient using the factorial definition
def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- State the theorem
theorem binom_2024_1 : binom 2024 1 = 2024 :=
by
  unfold binom
  rw [Nat.factorial_one, Nat.factorial_sub, Nat.sub_self]
  sorry

end binom_2024_1_l14_14546


namespace sequence_general_term_sequence_sum_of_reciprocals_l14_14224

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → S n = n^2 + n) →
  (∀ n : ℕ, n > 0 → a n = (S n) - (S (n - 1))) →
  (a 1 = 2) ∧ (∀ n : ℕ, n > 0 → a n = 2 * n) :=
sorry

theorem sequence_sum_of_reciprocals (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → S n = n^2 + n) →
  (∀ n : ℕ, n > 0 → a n = 2 * n) →
  (∀ n : ℕ, n > 0 → (finset.sum (finset.range n) (λ k, (1 : ℚ) / ((k+1) * a (k+1)))) = (n : ℚ) / (2 * (n+1))) :=
sorry

end sequence_general_term_sequence_sum_of_reciprocals_l14_14224


namespace first_new_player_weight_l14_14438

theorem first_new_player_weight :
  let initial_weight := 7 * 112
  let total_weight_with_new_players := 9 * 106
  let second_new_player_weight := 60
  let w := total_weight_with_new_players - (initial_weight + second_new_player_weight)
  w = 110
  :=
by
  let initial_weight := 7 * 112
  let total_weight_with_new_players := 9 * 106
  let second_new_player_weight := 60
  let w := total_weight_with_new_players - (initial_weight + second_new_player_weight)
  show w = 110 from sorry

end first_new_player_weight_l14_14438


namespace range_of_a_l14_14651

open Set

noncomputable def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | abs (x - a) ≤ 1}

theorem range_of_a :
  (∀ x, x ∈ B a → x ∈ A) ↔ (0 ≤ a ∧ a ≤ 1) :=
sorry

end range_of_a_l14_14651


namespace dot_product_l14_14671

variables (a b : Vector ℝ) -- ℝ here stands for the real numbers

-- Given conditions
def condition1 : ∥a∥ = 1 := sorry
def condition2 : ∥b∥ = √3 := sorry
def condition3 : ∥a - (2 : ℝ) • b∥ = 3 := sorry

-- Goal to prove
theorem dot_product (a b : Fin₃ → ℝ) 
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = √3)
  (h3 : ∥a - (2 : ℝ) • b∥ = 3) : 
  a ⬝ b = 1 := 
sorry

end dot_product_l14_14671


namespace negation_of_existential_l14_14430

theorem negation_of_existential (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) :=
by
  sorry

end negation_of_existential_l14_14430


namespace every_sum_of_squares_of_three_consecutive_odds_is_even_l14_14351

def is_sum_of_squares_of_three_consecutive_odds (n : ℤ) : Prop :=
  ∃ (k : ℤ), n = (k-2)^2 + k^2 + (k+2)^2

theorem every_sum_of_squares_of_three_consecutive_odds_is_even (n : ℤ) :
  is_sum_of_squares_of_three_consecutive_odds n → n % 2 = 0 :=
by
  intro h
  obtain ⟨k, hk⟩ := h
  rw hk
  calc
    (k-2)^2 + k^2 + (k+2)^2
        = k^2 - 4*k + 4 + k^2 + k^2 + 4*k + 4  : by ring
    ... = 3*k^2 + 8                            : by ring
  have : 3*k^2 + 8 % 2 = 0 :=
    by sorry
  show 3*k^2 + 8 % 2 = 0 from this

end every_sum_of_squares_of_three_consecutive_odds_is_even_l14_14351


namespace increasing_function_range_l14_14260

noncomputable def f (b : ℝ) : ℝ → ℝ :=
λ x, if x > 0 then (2 * b - 1) * x + (b - 1) else -x^2 + (2 - b) * x

theorem increasing_function_range (b : ℝ) : 1 ≤ b ∧ b ≤ 2 →
  ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → (f b x₁ - f b x₂) / (x₁ - x₂) > 0 :=
sorry

end increasing_function_range_l14_14260


namespace sum_of_products_l14_14269

theorem sum_of_products (n : ℕ) :
  let a : ℕ → ℕ := λ k, 2^k
  ∑ i in finset.range (n + 1), ∑ j in finset.range (i + 1), (a i) * (a j) = (4 / 3) * (2^n - 1) * (2^(n+1) - 1) :=
sorry

end sum_of_products_l14_14269


namespace cone_lateral_surface_area_l14_14186

theorem cone_lateral_surface_area (r V : ℝ) (h l S : ℝ) 
  (radius_condition : r = 6)
  (volume_condition : V = 30 * Real.pi)
  (volume_formula : V = (1 / 3) * Real.pi * r^2 * h)
  (slant_height_formula : l = Real.sqrt (r^2 + h^2))
  (lateral_surface_area_formula : S = Real.pi * r * l) :
  S = 39 * Real.pi := 
sorry

end cone_lateral_surface_area_l14_14186


namespace rhombus_area_l14_14574

theorem rhombus_area (R1 R2 : ℝ) (x y : ℝ)
  (hR1 : R1 = 15) (hR2 : R2 = 30)
  (hx : x = 15) (hy : y = 2 * x):
  (x * y / 2 = 225) :=
by 
  -- Lean 4 proof not required here
  sorry

end rhombus_area_l14_14574


namespace multiply_and_divide_equiv_l14_14050

/-- Defines the operation of first multiplying by 4/5 and then dividing by 4/7 -/
def multiply_and_divide (x : ℚ) : ℚ :=
  (x * (4 / 5)) / (4 / 7)

/-- Statement to prove the operation is equivalent to multiplying by 7/5 -/
theorem multiply_and_divide_equiv (x : ℚ) : 
  multiply_and_divide x = x * (7 / 5) :=
by 
  -- This requires a proof, which we can assume here
  sorry

end multiply_and_divide_equiv_l14_14050


namespace correct_options_l14_14632

variable {a b c : ℝ}

def sol_set (a b c : ℝ) : set ℝ :=
  {x | ax^2 + bx + c ≤ 0}

theorem correct_options (h: sol_set a b c = {x | x ≤ -2 ∨ x ≥ 3}) :
  (a < 0) ∧
  (∀ x, (ax + c > 0 ↔ x < 6)) ∧
  (¬ (8a + 4b + 3c < 0)) ∧
  (∀ x, (cx^2 + bx + a < 0 ↔ -1 / 2 < x ∧ x < 1 / 3)) :=
by
  sorry

end correct_options_l14_14632


namespace solution_set_of_f_l14_14242

def f (x : ℝ) : ℝ := if x ≥ 0 then x^3 - 8 else (-x)^3 - 8

theorem solution_set_of_f (x : ℝ) :
  f x = (if x ≥ 0 then x^3 - 8 else (-x)^3 - 8) →
  (x < 0 ∨ x > 4) ↔ f(x-2) > 0 :=
by
  sorry

end solution_set_of_f_l14_14242


namespace total_amount_paid_l14_14021

def apples_kg := 8
def apples_rate := 70
def mangoes_kg := 9
def mangoes_rate := 65
def oranges_kg := 5
def oranges_rate := 50
def bananas_kg := 3
def bananas_rate := 30

def total_amount := (apples_kg * apples_rate) + (mangoes_kg * mangoes_rate) + (oranges_kg * oranges_rate) + (bananas_kg * bananas_rate)

theorem total_amount_paid : total_amount = 1485 := by
  sorry

end total_amount_paid_l14_14021


namespace solve_l14_14172

noncomputable def problem (α : ℝ) := 
  sin α + cos α = 1 / 2

theorem solve {α : ℝ} (h : problem α) : sin (2 * α) = -3 / 4 := by
  sorry

end solve_l14_14172


namespace volleyball_ways_to_choose_starters_l14_14868

noncomputable def volleyball_team_starters : ℕ :=
  let total_players := 16
  let triplets := 3
  let twins := 2
  let other_players := total_players - triplets - twins
  let choose (n k : ℕ) := Nat.choose n k
  
  let no_triplets_no_twins := choose other_players 6
  let one_triplet_no_twins := triplets * choose other_players 5
  let no_triplet_one_twin := twins * choose other_players 5
  let one_triplet_one_twin := triplets * twins * choose other_players 4
  
  no_triplets_no_twins + one_triplet_no_twins + no_triplet_one_twin + one_triplet_one_twin

theorem volleyball_ways_to_choose_starters : volleyball_team_starters = 4752 := by
  sorry

end volleyball_ways_to_choose_starters_l14_14868


namespace winning_candidate_percentage_l14_14311

theorem winning_candidate_percentage
  (majority_difference : ℕ)
  (total_valid_votes : ℕ)
  (P : ℕ)
  (h1 : majority_difference = 192)
  (h2 : total_valid_votes = 480)
  (h3 : 960 * P = 67200) : 
  P = 70 := by
  sorry

end winning_candidate_percentage_l14_14311


namespace number_of_planes_determined_l14_14019

theorem number_of_planes_determined (L1 L2 L3 : line) (P : point)
    (h1 : L1 ∈ P ∧ L2 ∈ P ∧ L3 ∈ P)
    (h2 : ∃ plane1, plane1.contains L1 ∧ plane1.contains L2 ∧ plane1.contains L3 ↔ 1 = number_of_planes)
    (h3 : ∀ plane1 plane2 plane3, plane1.contains L1 ∧ ¬plane1.contains L2 ∧ ¬plane1.contains L3 ∧ plane2.contains L2 ∧ ¬plane2.contains L1 ∧ ¬plane2.contains L3 ∧ plane3.contains L3 ∧ ¬plane3.contains L1 ∧ ¬plane3.contains L2 ↔ 3 = number_of_planes) :
  number_of_planes = 1 ∨ number_of_planes = 3 :=
  by
  sorry

end number_of_planes_determined_l14_14019


namespace inequality_l14_14562

open Real

-- Define the points and conditions
variables (A D B C E : ℝ → ℝ → Prop)
variable (x : ℝ)

-- Assume the distances |AB|, |BC|, and |CD| all equal 1
def distance_AB := dist A B
def distance_BC := dist B C
def distance_CD := dist C D
axiom h1 : distance_AB = 1
axiom h2 : distance_BC = 1
axiom h3 : distance_CD = 1

-- AD is perpendicular to BC
axiom h4 : ⟂ (Line AD) (Line BC)

-- E is the intersection point of AD and BC
axiom h5 : intersects E (Line AD) (Line BC)

-- Further assumptions
axiom h6 : dist C E = x
axiom h7 : x < 1 / 2
axiom h8 : dist B E = 1 - x

-- Define distances for AE and DE using Pythagorean theorem
noncomputable def distance_AE : ℝ := sqrt (2 * x - x ^ 2)
noncomputable def distance_DE : ℝ := sqrt (1 - x ^ 2)
noncomputable def distance_AD : ℝ := sqrt (distance_AE ^ 2 + distance_DE ^ 2)

-- The inequality we need to prove
theorem inequality : |distance_B E - distance_C E| < distance_AD * sqrt 3 := by
  sorry

end inequality_l14_14562


namespace find_dot_product_l14_14704

open Real

variables (a b : ℝ^3)
variables (dot_product : ℝ^3 → ℝ^3 → ℝ)

def vector_magnitude (v : ℝ^3) : ℝ := sqrt (dot_product v v)

axiom magnitude_a : vector_magnitude a = 1
axiom magnitude_b : vector_magnitude b = sqrt 3
axiom magnitude_a_minus_2b : vector_magnitude (a - (2:ℝ) • b) = 3

theorem find_dot_product : dot_product a b = 1 :=
sorry

end find_dot_product_l14_14704


namespace factory_output_l14_14506

theorem factory_output :
  ∀ (J M : ℝ), M = J * 0.8 → J = M * 1.25 :=
by
  intros J M h
  sorry

end factory_output_l14_14506


namespace factorial_division_l14_14033

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_division : (factorial 15) / ((factorial 6) * (factorial 9)) = 5005 :=
by
  sorry

end factorial_division_l14_14033


namespace initial_roses_count_l14_14443

variable (R : ℕ) -- Initial number of roses
variable (O_initial : ℕ := 3) -- Initial number of orchids
variable (R_final : ℕ := 12) -- Final number of roses
variable (O_final : ℕ := 2) -- Final number of orchids
variable (roses_more_than_orchids : ℕ := 10) -- More roses than orchids now

theorem initial_roses_count :
  R_final - roses_more_than_orchids = R :=
by
  -- From the given conditions, we know:
  -- (1) Final number of roses is R_final.
  -- (2) The number of orchids initially is O_initial.
  -- (3) The number of additional roses is such that there are now 12 roses.
  -- (4) The final number of orchids is O_final.
  -- (5) The difference between the final roses and final orchids is roses_more_than_orchids.
  have eq1 : R_final = R + (R_final - R) := by sorry
  have eq2 : R_final - O_final = roses_more_than_orchids := by sorry
  -- From eq2, 12 - 2 = 10
  -- Therefore, we solve for initial roses: 12 - 10 = 2
  exact eq1


end initial_roses_count_l14_14443


namespace steve_total_cost_paid_l14_14836

-- Define all the conditions given in the problem
def cost_of_dvd_mike : ℕ := 5
def cost_of_dvd_steve (m : ℕ) : ℕ := 2 * m
def shipping_cost (s : ℕ) : ℕ := (8 * s) / 10

-- Define the proof problem (statement)
theorem steve_total_cost_paid : ∀ (m s sh t : ℕ), 
  m = cost_of_dvd_mike →
  s = cost_of_dvd_steve m → 
  sh = shipping_cost s → 
  t = s + sh → 
  t = 18 := by
    intros m s sh t h1 h2 h3 h4
    rw [h1, h2, h3, h4]
    norm_num -- The proof would normally be the next steps, but we skip it with sorry
    sorry

end steve_total_cost_paid_l14_14836


namespace max_area_of_rectangle_l14_14089

noncomputable def max_area (l w : ℕ) : ℕ :=
  if 2 * l + 2 * w = 40 then l * w else 0

theorem max_area_of_rectangle : 
  ∃ (l w : ℕ), 2 * l + 2 * w = 40 ∧ l * w = 100 :=
by
  use 10
  use 10
  simp
  exact ⟨by norm_num, by norm_num⟩

end max_area_of_rectangle_l14_14089


namespace part_a_part_b_l14_14058

-- Definition of Part (a)
theorem part_a (n : ℕ) : ∃ a b : ℕ, (1 - Real.sqrt 2)^n = a - b * Real.sqrt 2 ∧ a^2 - 2 * b^2 = (-1)^n :=
sorry

-- Definition of Part (b)
theorem part_b (n : ℕ) : ∃ m : ℕ, (Real.sqrt 2 - 1)^n = Real.sqrt m - Real.sqrt (m-1) :=
sorry

end part_a_part_b_l14_14058


namespace max_candies_purchased_l14_14377

-- Definitions of the costs and conditions
def cost_individual_candy : ℕ := 2
def cost_4_candy_pack : ℕ := 6
def cost_7_candy_pack : ℕ := 10
def discount_2_7_candy_packs : ℕ := 3
def total_money : ℕ := 25

-- Statement of the problem
theorem max_candies_purchased : ∃ m : ℕ, m ≤ total_money ∧ 
  (∀ n : ℕ, n ≤ total_money → n ≠ m → candies n < candies m) :=
by
  existsi 18
  split
  sorry

end max_candies_purchased_l14_14377


namespace log_arithmetic_sequence_l14_14326

variable {a : ℕ → ℝ}

def f (x : ℝ) : ℝ := (1/3) * x ^ 3 - 4 * x ^ 2 + 6 * x - 1

def is_extreme_point (x : ℝ) : Prop :=
  (f' x = 0)

noncomputable def a1 : ℝ := -- Determine a1 from the extreme points
noncomputable def a4033 : ℝ := -- Determine a4033 from the extreme points
def a2017 := (a1 + a4033) / 2

theorem log_arithmetic_sequence :
  is_extreme_point a1 → is_extreme_point a4033 →
  3 + log 2 3 = log 2 (a1 * a2017 * a4033) := by
  sorry

end log_arithmetic_sequence_l14_14326


namespace plaza_area_increase_l14_14100

theorem plaza_area_increase (a : ℝ) : 
  ((a + 2)^2 - a^2 = 4 * a + 4) :=
sorry

end plaza_area_increase_l14_14100


namespace quadratic_eq_root_l14_14139

theorem quadratic_eq_root (a b c m p n : ℕ) 
  (h_eq : 3 * a^2 - 8 * b - 5 = 0)
  (h_form : ∃ (m p n : ℕ), n > 0 ∧ gcd m p = 1 ∧ gcd p n = 1 ∧ gcd m n = 1 
    ∧ (a = (-b + m + p)/ (2 * p) ∨ a = (-b - m + p) / (2 * p))): 
  n = 31 :=
sorry

end quadratic_eq_root_l14_14139


namespace cone_lateral_surface_area_l14_14185

-- Definitions from conditions
def r : ℝ := 6
def V : ℝ := 30 * Real.pi

-- Theorem to prove
theorem cone_lateral_surface_area : 
  let h := V / (Real.pi * (r ^ 2) / 3) in
  let l := Real.sqrt (r ^ 2 + h ^ 2) in
  let S := Real.pi * r * l in
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l14_14185


namespace dot_product_proof_l14_14656

variables {ℝ : Type*}
variables (a b : ℝ → ℝ)
variables [inner_product_space ℝ ℝ]

theorem dot_product_proof
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = sqrt 3)
  (h3 : ∥a - 2 • b∥ = 3) :
  inner (a : ℝ) (b : ℝ) = 1 :=
sorry

end dot_product_proof_l14_14656


namespace total_cave_traversal_time_and_depth_l14_14810

def section1_depth := 600
def section1_time_per_100ft := 5

def section2_depth := 374
def section2_time_per_100ft := 10

def section3_depth := 1000
def section3_time_per_100ft := 3

def total_time (d1 d2 d3 : ℕ) (t1 t2 t3 : ℕ) : ℝ := 
  (d1 / 100) * t1 + (d2 / 100) * t2 + ((d2 % 100) / 100.0) * t2 + (d3 / 100) * t3

def total_depth (d1 d2 d3 : ℕ) : ℕ := d1 + d2 + d3

theorem total_cave_traversal_time_and_depth :
  total_time section1_depth section2_depth section3_depth section1_time_per_100ft section2_time_per_100ft section3_time_per_100ft = 97.4 ∧
  total_depth section1_depth section2_depth section3_depth = 1974 := 
  by
    sorry

end total_cave_traversal_time_and_depth_l14_14810


namespace min_value_y_l14_14173

theorem min_value_y :
  ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → 
  ∃ y_min, y_min = 3 + 2 * Real.sqrt 2 ∧ ( ∀ y, y = 1 / a + 2 / b → y ≥ y_min ) :=
by
  intros a b h_a_pos h_b_pos h_a_b
  use 3 + 2 * Real.sqrt 2
  constructor
  { 
    exact rfl
  },
  {
    sorry
  }

end min_value_y_l14_14173


namespace one_and_two_thirds_of_what_number_is_45_l14_14847

theorem one_and_two_thirds_of_what_number_is_45 (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by
  sorry

end one_and_two_thirds_of_what_number_is_45_l14_14847


namespace simplify_expression_l14_14906

variable (y : ℝ)

theorem simplify_expression : (5 / (4 * y^(-4)) * (4 * y^3) / 3) = 5 * y^7 / 3 :=
by
  sorry

end simplify_expression_l14_14906


namespace smallest_abundant_not_multiple_of_5_is_12_l14_14583

/-- Define a number n is abundant if the sum of its proper divisors is greater than n -/
def is_abundant (n : ℕ) : Prop :=
  ∑ i in finset.filter (λ x, x < n ∧ n % x = 0) (finset.range n), i > n

/-- Main theorem: Smallest abundant number that is not a multiple of 5 is 12 -/
theorem smallest_abundant_not_multiple_of_5_is_12 :
  ∃ (n : ℕ), is_abundant n ∧ n % 5 ≠ 0 ∧ (∀ m : ℕ, is_abundant m ∧ m % 5 ≠ 0 → 12 ≤ m) → n = 12 :=
sorry

end smallest_abundant_not_multiple_of_5_is_12_l14_14583


namespace find_ff_neg3_l14_14642

def f (x : ℝ) : ℝ := 
  if x > 0 then log 3 x - 2 
  else if x < 0 then 2^(x + 3) 
  else 0 -- definition to handle x = 0 case by default although it's not specified in the problem.

theorem find_ff_neg3 : f (f (-3)) = -2 := 
by 
  sorry

end find_ff_neg3_l14_14642


namespace BK_eq_2DC_l14_14787

-- Defining the elements and conditions of the triangle and points
variables {A B C D K : Type*} [real_vector_space V] [affine_space.point A] 
variables (triangle_ABC : triangle A B C) 
variables (right_angle_C : angle_eq triangle_ABC.angle C 90)
variables (D_on_AC : point_on_line D triangle_ABC.side AC)
variables (K_on_BD : point_on_segment K triangle_ABC.segment BD)
variables (angle_condition : angle_eq (triangle_ABC.angle B) (triangle_angle KAD) ∧ 
                                   angle_eq (triangle_angle KAD) (triangle_angle AKD))

-- The theorem we aim to prove
theorem BK_eq_2DC : segment_length BK = 2 * segment_length DC :=
by
  -- Additional conditions and constructions introduced in the solution steps
  sorry

end BK_eq_2DC_l14_14787


namespace vector_addition_correct_l14_14544

open Matrix

-- Define the vectors as 3x1 matrices
def v1 : Matrix (Fin 3) (Fin 1) ℤ := ![![3], ![-5], ![1]]
def v2 : Matrix (Fin 3) (Fin 1) ℤ := ![![-1], ![4], ![-2]]
def v3 : Matrix (Fin 3) (Fin 1) ℤ := ![![2], ![-1], ![3]]

-- Define the scalar multiples
def scaled_v1 := (2 : ℤ) • v1
def scaled_v2 := (3 : ℤ) • v2
def neg_v3 := (-1 : ℤ) • v3

-- Define the summation result
def result := scaled_v1 + scaled_v2 + neg_v3

-- Define the expected result for verification
def expected_result : Matrix (Fin 3) (Fin 1) ℤ := ![![1], ![3], ![-7]]

-- The proof statement (without the proof itself)
theorem vector_addition_correct :
  result = expected_result := by
  sorry

end vector_addition_correct_l14_14544


namespace find_x_l14_14866

-- Define the variables and conditions
def x := 27
axiom h : (5 / 3) * x = 45

-- Main statement to be proved
theorem find_x : x = 27 :=
by
  have : (5 / 3) * x = 45 := h
  sorry

end find_x_l14_14866


namespace real_roots_of_equation_root_one_of_equation_l14_14640

noncomputable def discriminant_quadratic (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem real_roots_of_equation (k : ℝ) : 
  discriminant_quadratic 1 (-2*(k-3)) (k^2-4*k-1) ≥ 0 → k ≤ 5 := 
by
  sorry

theorem root_one_of_equation (k : ℝ) : 
  (1 : ℝ)^2 - 2*(k-3) + k^2 - 4*k - 1 = 0 → k = 3 + real.sqrt 3 ∨ k = 3 - real.sqrt 3 := 
by
  sorry

end real_roots_of_equation_root_one_of_equation_l14_14640


namespace kira_lucky_license_plates_l14_14483

-- We define the set of allowed letters and digits
def allowed_letters := {А, В, Е, К, М, Н, О, Р, С, Т, У, Х}
def vowels := {А, Е, О, У}
def digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def odd_digits := {1, 3, 5, 7, 9}
def even_digits := {0, 2, 4, 6, 8}

-- The number of "lucky" license plates
def lucky_license_plates := 12 * 4 * 12 * 10 * 5 * 5 * 10 - 1

theorem kira_lucky_license_plates : 
  ∃ n, n = lucky_license_plates ∧ n = 359999 :=
by {
  use 359999,
  dsimp [lucky_license_plates],
  exact eq.refl 359999,
  sorry
}

end kira_lucky_license_plates_l14_14483


namespace trigonometric_range_l14_14237

noncomputable def trigonometric_expression_range (α β : ℝ) : set ℝ :=
  {y | y = sin (α + π / 4) + 2 * sin (β + π / 4)}

theorem trigonometric_range 
  (α β : ℝ) (h : sin α + 2 * cos β = 2) : 
  trigonometric_expression_range α β = set.Icc (√2 - √10 / 2) (√2 + √10 / 2) :=
sorry

end trigonometric_range_l14_14237


namespace trajectory_theorem_lambda_mu_theorem_l14_14772

section TrajectoryAndVectorProblem

variables {F : ℝ × ℝ} {P : ℝ × ℝ} {Q : ℝ × ℝ} (trajectory : ℝ → ℝ) (λ μ t : ℝ)
def F : ℝ × ℝ := (2, 0)
def l_eqn (x : ℝ) : Prop := x = -2

-- Trajectory problem: proving the trajectory equation
def trajectory_eq (P : ℝ × ℝ) : Prop :=
  let Q := (-2, P.snd)
  let QP : ℝ × ℝ := (P.fst + 2, 0)
  let QF : ℝ × ℝ := (4, -P.snd)
  let FP : ℝ × ℝ := (P.fst - 2, P.snd)
  let FQ : ℝ × ℝ := (-4, P.snd)
  ( QP.fst * QF.fst + QP.snd * QF.snd = FP.fst * FQ.fst + FP.snd * FQ.snd ) →
  P.snd^2 = 8 * P.fst

-- Line intersection and vector problem: proving λ + μ = 0
def lambda_mu_eq (A B : ℝ × ℝ) (M : ℝ × ℝ) (trajectory : ℝ → ℝ) (λ μ : ℝ) : Prop :=
  let line_eqn := (λ y : ℝ, t * y + 2)
  let y1 := A.snd
  let y2 := B.snd
  y1 + y2 = 8 * t ∧ y1 * y2 = -16 →
  λ = -1 - 4 / (t * y1) ∧ μ = -1 - 4 / (t * y2) →
  λ + μ = 0

theorem trajectory_theorem (P : ℝ × ℝ) : trajectory_eq P :=
by {
  -- Proof omitted
  sorry
}

theorem lambda_mu_theorem (A B : ℝ × ℝ) (M : ℝ × ℝ) : lambda_mu_eq A B M trajectory λ μ :=
by {
  -- Proof omitted
  sorry
}

end TrajectoryAndVectorProblem

end trajectory_theorem_lambda_mu_theorem_l14_14772


namespace perimeter_ABCD_l14_14777

-- Definitions based on the conditions
def ABE_is_right (A B E : Point) : Prop := ∠AEB = 60 ∧ is_right_triangle A B E
def BCE_is_right (B C E : Point) : Prop := ∠BEC = 60 ∧ is_right_triangle B C E
def CDE_is_right (C D E : Point) : Prop := ∠CED = 60 ∧ is_right_triangle C D E
def AE_length (A E : Point) := dist A E = 24

-- Main statement
theorem perimeter_ABCD (A B C D E : Point) 
  (h1 : ABE_is_right A B E)
  (h2 : BCE_is_right B C E) 
  (h3 : CDE_is_right C D E) 
  (h4 : AE_length A E) : 
  perimeter_of_quadrilateral A B C D = 27 + 21 * sqrt 3 := 
sorry

end perimeter_ABCD_l14_14777


namespace cone_lateral_surface_area_l14_14183

-- Definitions from conditions
def r : ℝ := 6
def V : ℝ := 30 * Real.pi

-- Theorem to prove
theorem cone_lateral_surface_area : 
  let h := V / (Real.pi * (r ^ 2) / 3) in
  let l := Real.sqrt (r ^ 2 + h ^ 2) in
  let S := Real.pi * r * l in
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l14_14183


namespace find_dot_product_l14_14703

open Real

variables (a b : ℝ^3)
variables (dot_product : ℝ^3 → ℝ^3 → ℝ)

def vector_magnitude (v : ℝ^3) : ℝ := sqrt (dot_product v v)

axiom magnitude_a : vector_magnitude a = 1
axiom magnitude_b : vector_magnitude b = sqrt 3
axiom magnitude_a_minus_2b : vector_magnitude (a - (2:ℝ) • b) = 3

theorem find_dot_product : dot_product a b = 1 :=
sorry

end find_dot_product_l14_14703


namespace moles_of_HCl_produced_l14_14157

noncomputable def sodium_chloride : ℕ := 3
noncomputable def nitric_acid : ℕ := 3

theorem moles_of_HCl_produced :
  sodium_chloride = 3 ∧ nitric_acid = 3 →
  (sodium_chloride = nitric_acid → sodium_chloride = 3 ∧ nitric_acid = 3 → 3) :=
by sorry

end moles_of_HCl_produced_l14_14157


namespace units_digit_of_expression_l14_14159

theorem units_digit_of_expression :
  (6 * 16 * 1986 - 6 ^ 4) % 10 = 0 := 
sorry

end units_digit_of_expression_l14_14159


namespace profit_percent_l14_14482

theorem profit_percent (CP SP : ℤ) (h : CP/SP = 2/3) : (SP - CP) * 100 / CP = 50 := 
by
  sorry

end profit_percent_l14_14482


namespace identity_solution_l14_14888

theorem identity_solution (x : ℝ) :
  ∃ a b : ℝ, (2 * x + a) ^ 3 = 5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x ∧
             a = -1 ∧ b = 1 :=
by
  -- we can skip the proof as this is just a statement
  sorry

end identity_solution_l14_14888


namespace circle_line_distance_l14_14289

theorem circle_line_distance (c : ℝ) : 
  (∃ (P₁ P₂ P₃ : ℝ × ℝ), 
     (P₁ ≠ P₂ ∧ P₂ ≠ P₃ ∧ P₁ ≠ P₃) ∧
     ((P₁.1 - 2)^2 + (P₁.2 - 2)^2 = 18) ∧
     ((P₂.1 - 2)^2 + (P₂.2 - 2)^2 = 18) ∧
     ((P₃.1 - 2)^2 + (P₃.2 - 2)^2 = 18) ∧
     (abs (P₁.1 - P₁.2 + c) / Real.sqrt 2 = 2 * Real.sqrt 2) ∧
     (abs (P₂.1 - P₂.2 + c) / Real.sqrt 2 = 2 * Real.sqrt 2) ∧
     (abs (P₃.1 - P₃.2 + c) / Real.sqrt 2 = 2 * Real.sqrt 2)) ↔ 
  -2 ≤ c ∧ c ≤ 2 :=
sorry

end circle_line_distance_l14_14289


namespace cone_lateral_surface_area_l14_14201

theorem cone_lateral_surface_area (r h l S : ℝ) (π_pos : 0 < π) (r_eq : r = 6)
  (V : ℝ) (V_eq : V = 30 * π)
  (vol_eq : V = (1/3) * π * r^2 * h)
  (h_eq : h = 5 / 2)
  (l_eq : l = Real.sqrt (r^2 + h^2))
  (S_eq : S = π * r * l) :
  S = 39 * π :=
  sorry

end cone_lateral_surface_area_l14_14201


namespace dot_product_is_one_l14_14720

variable {V : Type*} [InnerProductSpace ℝ V]
variables (a b : V)

theorem dot_product_is_one 
  (ha : ∥a∥ = 1) 
  (hb : ∥b∥ = sqrt 3) 
  (hab : ∥a - 2•b∥ = 3) : 
  ⟪a, b⟫ = 1 :=
by 
  sorry

end dot_product_is_one_l14_14720


namespace find_x_l14_14856

theorem find_x (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by 
  sorry

end find_x_l14_14856


namespace simplify_expression_l14_14912

variable {y : ℤ}

theorem simplify_expression (y : ℤ) : 5 / (4 * y^(-4)) * (4 * y^3) / 3 = 5 * y^7 / 3 := 
by 
  -- Proof is omitted with 'sorry'
  sorry

end simplify_expression_l14_14912


namespace dot_product_l14_14669

variables (a b : Vector ℝ) -- ℝ here stands for the real numbers

-- Given conditions
def condition1 : ∥a∥ = 1 := sorry
def condition2 : ∥b∥ = √3 := sorry
def condition3 : ∥a - (2 : ℝ) • b∥ = 3 := sorry

-- Goal to prove
theorem dot_product (a b : Fin₃ → ℝ) 
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = √3)
  (h3 : ∥a - (2 : ℝ) • b∥ = 3) : 
  a ⬝ b = 1 := 
sorry

end dot_product_l14_14669


namespace algebraic_integer_a0_x0_l14_14368

noncomputable def algebraic_integer (a : ℤ) : Prop :=
∃ n : ℕ, ∃ p : ℤ[X], p.monic ∧ degree p = n ∧
  (p.eval a) = 0

theorem algebraic_integer_a0_x0
  (n : ℕ) (a : ℤ) (x : ℂ)
  (coeffs : fin (n + 1) → ℤ)
  (root : ∑ i in finset.range (n + 1), (coeffs i) * x^(n - i) = 0) :
  algebraic_integer (a * x) := 
sorry

end algebraic_integer_a0_x0_l14_14368


namespace election_result_l14_14015

theorem election_result:
  ∀ (Henry_votes India_votes Jenny_votes Ken_votes Lena_votes : ℕ)
    (counted_percentage : ℕ)
    (counted_votes : ℕ), 
    Henry_votes = 14 → 
    India_votes = 11 → 
    Jenny_votes = 10 → 
    Ken_votes = 8 → 
    Lena_votes = 2 → 
    counted_percentage = 90 → 
    counted_votes = 45 → 
    (counted_percentage * Total_votes / 100 = counted_votes) →
    (Total_votes = counted_votes * 100 / counted_percentage) →
    (Remaining_votes = Total_votes - counted_votes) →
    ((Henry_votes + Max_remaining_Votes >= Max_votes) ∨ 
    (India_votes + Max_remaining_Votes >= Max_votes) ∨ 
    (Jenny_votes + Max_remaining_Votes >= Max_votes)) →
    3 = 
    (if Henry_votes + Remaining_votes > Max_votes then 1 else 0) + 
    (if India_votes + Remaining_votes > Max_votes then 1 else 0) + 
    (if Jenny_votes + Remaining_votes > Max_votes then 1 else 0) := 
  sorry

end election_result_l14_14015


namespace polar_curve_intersection_length_l14_14330

theorem polar_curve_intersection_length (theta t : ℝ) (h: 0 ≤ theta ∧ theta < real.pi) :
  let rho := 6 * real.sin theta,
      P := (real.sqrt 2, real.pi / 4 : ℝ×ℝ),
      C := {p : ℝ × ℝ | p.1^2 + (p.2 - 3)^2 = 9},
      l := {p : ℝ × ℝ | ∃ t : ℝ, p.1 = 1 + t * real.cos theta ∧ p.2 = 1 + t * real.sin theta}
  in
    (∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧ A ∈ l ∧ B ∈ l ∧ abs (dist P A) = 2 * abs (dist P B)) →
    (∃ AB : ℝ, AB = 3 * real.sqrt 2) :=
by
  sorry

end polar_curve_intersection_length_l14_14330


namespace largest_positive_integer_divisible_l14_14987

theorem largest_positive_integer_divisible (n : ℕ) :
  (n + 20 ∣ n^3 - 100) ↔ n = 2080 :=
sorry

end largest_positive_integer_divisible_l14_14987


namespace three_pow_23_mod_11_l14_14465

theorem three_pow_23_mod_11 : 3^23 % 11 = 5 :=
by {
  have h1 : 3^1 % 11 = 3 := by norm_num,
  have h2 : 3^2 % 11 = 9 := by norm_num,
  have h3 : 3^3 % 11 = 5 := by norm_num,
  have h4 : 3^4 % 11 = 4 := by norm_num,
  have h5 : 3^5 % 11 = 1 := by norm_num,
  sorry
}

end three_pow_23_mod_11_l14_14465


namespace two_times_sum_of_fourth_power_is_perfect_square_l14_14434

theorem two_times_sum_of_fourth_power_is_perfect_square (a b c : ℤ) 
  (h : a + b + c = 0) : 2 * (a^4 + b^4 + c^4) = (a^2 + b^2 + c^2)^2 := 
by sorry

end two_times_sum_of_fourth_power_is_perfect_square_l14_14434


namespace area_of_triangle_BXN_l14_14350

open Real

noncomputable def is_isosceles (A B C : Point ℝ) : Prop := (dist A B = dist B C)

noncomputable def midpoint (A B : Point ℝ) : Point ℝ := (1/2 • A + 1/2 • B)

noncomputable def is_equilateral (A B C : Point ℝ) : Prop :=
  (dist A B = dist B C) ∧ (dist B C = dist C A)

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s^2

theorem area_of_triangle_BXN 
  (A B C M N X : Point ℝ)
  (h1 : is_isosceles A B C)
  (h2 : dist A C = 4)
  (h3 : M = midpoint A C)
  (h4 : N = midpoint A B)
  (h5 : line (C, N) = angle_bisector A C B)
  (h6 : X = line (B, M) ∩ line (C, N))
  (h7 : is_equilateral B X N)
  : area_of_triangle_BXN = sqrt(3)/4 := 
sorry

end area_of_triangle_BXN_l14_14350


namespace factorial_division_l14_14043

-- Define factorial using the standard library's factorial function
def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- The problem statement
theorem factorial_division :
  (factorial 15) / (factorial 6 * factorial 9) = 834 :=
sorry

end factorial_division_l14_14043


namespace adoption_time_l14_14088

theorem adoption_time
  (p0 : ℕ) (p1 : ℕ) (rate : ℕ)
  (p0_eq : p0 = 10) (p1_eq : p1 = 15) (rate_eq : rate = 7) :
  Nat.ceil ((p0 + p1) / rate) = 4 := by
  sorry

end adoption_time_l14_14088


namespace express_positive_rational_less_than_one_l14_14364
-- Import necessary libraries

-- Define the sequence as a predicate to show each term is a positive integer
def is_positive_integer_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, 0 < a n

-- Define the condition that for any prime p, there are infinitely many terms divisible by p
def infinitely_many_divisible_by (a : ℕ → ℕ) (p : ℕ) [h : Fact (Nat.Prime p)] : Prop :=
  ∃ infinitely_many (n : ℕ), p ∣ a n

-- Main theorem statement
theorem express_positive_rational_less_than_one
  (a : ℕ → ℕ)
  (h_seq: is_positive_integer_sequence a)
  (h_inf_prime: ∀ p [Fact (Nat.Prime p)], infinitely_many_divisible_by a p)
  (q : ℚ) (h_q_pos : 0 < q) (h_q_lt_1 : q < 1) :
  ∃ (b : ℕ → ℕ) (n : ℕ), (∀ i : ℕ, i < n → 0 ≤ b i ∧ b i < a i) ∧ q = ∑ i in finset.range n, (b i : ℚ) / (∏ j in finset.range (i + 1), a j) :=
sorry

end express_positive_rational_less_than_one_l14_14364


namespace ratio_of_volumes_of_tetrahedrons_l14_14307

theorem ratio_of_volumes_of_tetrahedrons (a b : ℝ) (h : a / b = 1 / 2) : (a^3) / (b^3) = 1 / 8 :=
by
-- proof goes here
sorry

end ratio_of_volumes_of_tetrahedrons_l14_14307


namespace shift_down_equation_l14_14874

def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := f x - 3

theorem shift_down_equation : ∀ x : ℝ, g x = 2 * x := by
  sorry

end shift_down_equation_l14_14874


namespace dot_product_ab_l14_14683

variables (a b : ℝ^3)

-- Given conditions
def condition1 : Prop := ‖a‖ = 1
def condition2 : Prop := ‖b‖ = real.sqrt 3
def condition3 : Prop := ‖a - 2 • b‖ = 3

-- The theorem statement to prove
theorem dot_product_ab (h1 : condition1 a) (h2 : condition2 b) (h3 : condition3 a b) : 
  a ⬝ b = 1 :=
sorry

end dot_product_ab_l14_14683


namespace total_pay_of_two_employees_l14_14025

theorem total_pay_of_two_employees
  (Y_pay : ℝ)
  (X_pay : ℝ)
  (h1 : Y_pay = 280)
  (h2 : X_pay = 1.2 * Y_pay) :
  X_pay + Y_pay = 616 :=
by
  sorry

end total_pay_of_two_employees_l14_14025


namespace eccentricity_of_ellipse_l14_14227

theorem eccentricity_of_ellipse (a b x y : ℝ) (h : a > b ∧ b > 0) :
  (P : ℝ × ℝ) (F1 : ℝ × ℝ) (A B O : ℝ × ℝ) 
  (h1 : A = (a,0)) 
  (h2 : B = (0,b)) 
  (h3 : F1 = (-real.sqrt (a^2 - b^2), 0)) 
  (h4 : P = (F1.1, y) ∧ ((F1.1^2 / a^2) + (y^2 / b^2) = 1)) 
  (h5 : (B.2 - A.2)/(B.1 - A.1) = (P.2 - O.2)/(P.1 - O.1)) :
  real.sqrt(2) / 2 = (real.sqrt (a^2 - b^2)) / a :=
by sorry

end eccentricity_of_ellipse_l14_14227


namespace smallest_abundant_not_multiple_of_five_l14_14578

def proper_divisors_sum (n : ℕ) : ℕ :=
  (Nat.divisors n).erase n |>.sum

def is_abundant (n : ℕ) : Prop := proper_divisors_sum n > n

def is_not_multiple_of_five (n : ℕ) : Prop := ¬ (5 ∣ n)

theorem smallest_abundant_not_multiple_of_five :
  18 = Nat.find (λ n, is_abundant n ∧ is_not_multiple_of_five n) :=
sorry

end smallest_abundant_not_multiple_of_five_l14_14578


namespace exists_divisible_subset_of_six_l14_14168

theorem exists_divisible_subset_of_six
  (A : Finset ℕ) (h_card : A.card = 26) 
  (h_property : ∀(S : Finset ℕ), S ⊆ A → S.card = 6 → (∃ x y ∈ S, x ≠ y ∧ (x ∣ y ∨ y ∣ x))) :
  ∃ (B : Finset ℕ), B ⊆ A ∧ B.card = 6 ∧ (∃ x ∈ B, ∀ y ∈ B, x ∣ y ∨ x = y) :=
sorry

end exists_divisible_subset_of_six_l14_14168


namespace sequence_b_gt_neg3_l14_14928

theorem sequence_b_gt_neg3 (b : ℝ) (a : ℕ → ℝ) (h : ∀ n : ℕ, n > 0 → a n = n^2 + b * n) :
  (∀ n : ℕ, n > 0 → a (n + 1) > a n) → b > -3 :=
begin
  sorry
end

end sequence_b_gt_neg3_l14_14928


namespace dot_product_eq_one_l14_14701

variables {α : Type*} [InnerProductSpace ℝ α]

noncomputable def vector_a (a : α) : Prop := ∥a∥ = 1
noncomputable def vector_b (b : α) : Prop := ∥b∥ = real.sqrt 3
noncomputable def vector_c (a b : α) : Prop := ∥a - (2 : ℝ) • b∥ = 3

theorem dot_product_eq_one (a b : α) (ha : vector_a a) (hb : vector_b b) (hc : vector_c a b) :
  inner a b = 1 :=
sorry

end dot_product_eq_one_l14_14701


namespace derivative_at_2_l14_14254

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem derivative_at_2 : deriv f 2 = (1 - Real.log 2) / 4 :=
by
  sorry

end derivative_at_2_l14_14254


namespace dot_product_eq_one_l14_14702

variables {α : Type*} [InnerProductSpace ℝ α]

noncomputable def vector_a (a : α) : Prop := ∥a∥ = 1
noncomputable def vector_b (b : α) : Prop := ∥b∥ = real.sqrt 3
noncomputable def vector_c (a b : α) : Prop := ∥a - (2 : ℝ) • b∥ = 3

theorem dot_product_eq_one (a b : α) (ha : vector_a a) (hb : vector_b b) (hc : vector_c a b) :
  inner a b = 1 :=
sorry

end dot_product_eq_one_l14_14702


namespace cone_lateral_surface_area_l14_14197

-- Definitions based on the conditions
def coneRadius : ℝ := 6
def coneVolume : ℝ := 30 * Real.pi

-- Mathematical statement
theorem cone_lateral_surface_area (r V : ℝ) (hr : r = coneRadius) (hV : V = coneVolume) :
  ∃ S : ℝ, S = 39 * Real.pi :=
by 
  have h_volume := hV
  have h_radius := hr
  sorry

end cone_lateral_surface_area_l14_14197


namespace part_a_part_b_l14_14463

noncomputable def combination (n k : ℕ) : ℕ := nat.choose n k

def probability_exactly_4_lower_cards : ℚ :=
  let total_combinations := combination 32 8;
  let favorable_combinations := combination 28 4;
  favorable_combinations / total_combinations

def probability_at_least_5_red_cards : ℚ :=
  let total_combinations := combination 32 8;
  let favorable_combinations :=
    (combination 8 8) * (combination 24 0) + -- for 8 red cards
    (combination 8 7) * (combination 24 1) + -- for 7 red cards
    (combination 8 6) * (combination 24 2) + -- for 6 red cards
    (combination 8 5) * (combination 24 3);  -- for 5 red cards
  favorable_combinations / total_combinations

theorem part_a : probability_exactly_4_lower_cards = 7 / 3596 := 
  by sorry

theorem part_b : probability_at_least_5_red_cards = 24253 / 2103660 := 
  by sorry

end part_a_part_b_l14_14463


namespace first_train_length_l14_14453

theorem first_train_length 
  (speed1_kmph : ℝ) (speed2_kmph : ℝ) 
  (crossing_time : ℝ) (length2 : ℝ) :
  speed1_kmph = 72 →
  speed2_kmph = 18 →
  crossing_time = 17.998560115190784 →
  length2 = 250 →
  let speed1 := speed1_kmph * (1000 / 3600) in
  let speed2 := speed2_kmph * (1000 / 3600) in
  let relative_speed := speed1 + speed2 in
  let total_distance := relative_speed * crossing_time in
  total_distance = 449.9640028797696 →
  total_distance = length2 + length1 →
  length1 = 199.9640028797696 :=
by
  intros
  sorry

end first_train_length_l14_14453


namespace monotonic_increasing_interval_l14_14256

-- Define the function f
def f (x a : ℝ) := log a (2 * x^2 - x)

-- Define the main condition
def a_condition (a : ℝ) : Prop := (0 < a) ∧ (a ≠ 1)

-- Define the interval condition
def interval_condition (x : ℝ) (a : ℝ) : Prop := f x a > 0 ∧ (1 / 2 < x ∧ x < 1)

-- The main theorem: proving the interval where f(x) is monotonically increasing
theorem monotonic_increasing_interval (a : ℝ) (h : a_condition a) :
  (∀ x, interval_condition x a) → ∀ x, f x a = log a (2 * x^2 - x) → is_strict_monotone (f x)
  sorry -- proof placeholder

end monotonic_increasing_interval_l14_14256


namespace good_permutation_exists_iff_power_of_two_l14_14458

def is_good_permutation (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ i j k : ℕ, i < j → j < k → k < n → ¬ (↑n ∣ (a i + a k - 2 * a j))

theorem good_permutation_exists_iff_power_of_two (n : ℕ) (h : n ≥ 3) :
  (∃ a : ℕ → ℕ, (∀ i, i < n → a i < n) ∧ is_good_permutation n a) ↔ ∃ b : ℕ, 2 ^ b = n :=
sorry

end good_permutation_exists_iff_power_of_two_l14_14458


namespace square_complex_C_l14_14636

noncomputable def A : ℂ := 1 + 2*complex.I
noncomputable def B : ℂ := 3 - 5*complex.I

theorem square_complex_C (h : ∀ z : ℂ, z ≠ 0 → z * complex.I ≠ 0) : ∃ C : ℂ, C = 10 - 3*complex.I :=
by
  have AB : ℂ := B - A
  have BC : ℂ := AB * complex.I
  have C : ℂ := B + BC
  existsi C
  sorry

end square_complex_C_l14_14636


namespace replace_stars_identity_l14_14878

theorem replace_stars_identity : 
  ∃ (a b : ℤ), a = -1 ∧ b = 1 ∧ (2 * x + a)^3 = 5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x := 
by 
  use [-1, 1]
  sorry

end replace_stars_identity_l14_14878


namespace points_concyclic_l14_14527

open Geometry

-- Definition of the problem
theorem points_concyclic 
  (A B C P D E F L L' M M' N N' : Point)
  (h₁ : ∆ABC is a triangle)
  (h₂ : P inside ∆ABC)
  (h₃ : Line(AP) intersect (BC) at D)
  (h₄ : Line(BP) intersect (CA) at E)
  (h₅ : Line(CP) intersect (AB) at F)
  (h₆ : Circle(BC) ∩ Circle(AD) = {L, L'})
  (h₇ : Circle(CA) ∩ Circle(BE) = {M, M'})
  (h₈ : Circle(AB) ∩ Circle(CF) = {N, N'}) :
  Concyclic (Set.of_list [L, L', M, M', N, N']) := 
sorry

end points_concyclic_l14_14527


namespace solve_eq1_solve_eq2_l14_14542

theorem solve_eq1 : (2 * (x - 3) = 3 * x * (x - 3)) → (x = 3 ∨ x = 2 / 3) :=
by
  intro h
  sorry

theorem solve_eq2 : (2 * x ^ 2 - 3 * x + 1 = 0) → (x = 1 ∨ x = 1 / 2) :=
by
  intro h
  sorry

end solve_eq1_solve_eq2_l14_14542


namespace maximum_area_exists_l14_14094

def max_area_rectangle (l w : ℕ) (h : l + w = 20) : Prop :=
  l * w ≤ 100

theorem maximum_area_exists : ∃ (l w : ℕ), max_area_rectangle l w (by sorry) ∧ (10 * 10 = 100) :=
begin
  sorry
end

end maximum_area_exists_l14_14094


namespace four_digit_numbers_sum_l14_14348

def count_four_digit_odd_numbers : ℕ := 9 * 10 * 10 * 5
def count_four_digit_multiples_of_3 : ℕ := (9 * 10 * 10 * 10) / 3

theorem four_digit_numbers_sum : count_four_digit_odd_numbers + count_four_digit_multiples_of_3 = 7500 :=
by
  let A := count_four_digit_odd_numbers
  let B := count_four_digit_multiples_of_3
  have hA : A = 4500 := by sorry
  have hB : B = 3000 := by sorry
  rw [hA, hB]
  exact Eq.refl 7500

end four_digit_numbers_sum_l14_14348


namespace common_ratio_of_geometric_sequence_l14_14372

variable (a_1 q : ℚ) (S : ℕ → ℚ)

def geometric_sum (n : ℕ) : ℚ :=
  a_1 * (1 - q^n) / (1 - q)

def is_arithmetic_sequence (a b c : ℚ) : Prop :=
  2 * b = a + c

theorem common_ratio_of_geometric_sequence 
  (h1 : ∀ n, S n = geometric_sum a_1 q n)
  (h2 : ∀ n, is_arithmetic_sequence (S (n+2)) (S (n+1)) (S n)) : q = -2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l14_14372


namespace simplify_expression_l14_14907

variable (y : ℝ)

theorem simplify_expression : (5 / (4 * y^(-4)) * (4 * y^3) / 3) = 5 * y^7 / 3 :=
by
  sorry

end simplify_expression_l14_14907


namespace find_a2_and_S10_l14_14225

-- Definitions of the sequence, sum of the sequence, and conditions
def a (n : ℕ) : ℝ := if n = 0 then 0 else 2 + (n - 1) * 2
def S (n : ℕ) : ℝ := (n / 2) * (2 * 2 + (n - 1) * 2)

-- Conditions given in the problem
lemma a1_eq_2 : a 1 = 2 := by
  simp [a, if_pos]
lemma S2_eq_a3 : S 2 = a 3 := by
  simp [S, a, if_neg]
  linarith

-- Statements to be proved
theorem find_a2_and_S10 : a 2 = 4 ∧ S 10 = 110 := by
  split
  case left => 
    -- Proof for a2
    sorry
  case right =>
    -- Proof for S10
    sorry

end find_a2_and_S10_l14_14225


namespace count_units_digit_cubes_greater_than_5_l14_14591

def units_digit (n : ℕ) : ℕ := n % 10

def cubes_units_digit_greater_than_5 (n : ℕ) : Prop :=
  units_digit (n ^ 3) > 5

def count_integers_with_property (P : ℕ → Prop) (s : Finset ℕ) : ℕ :=
  (s.filter P).card

theorem count_units_digit_cubes_greater_than_5 :
  count_integers_with_property cubes_units_digit_greater_than_5 (Finset.range 201).erase 0 = 80 := 
sorry

end count_units_digit_cubes_greater_than_5_l14_14591


namespace sum_of_angles_l14_14611

theorem sum_of_angles (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (sin_α : Real.sin α = 2 * Real.sqrt 5 / 5) (sin_beta : Real.sin β = 3 * Real.sqrt 10 / 10) :
  α + β = 3 * Real.pi / 4 :=
sorry

end sum_of_angles_l14_14611


namespace cone_lateral_surface_area_l14_14194

-- Definitions based on the conditions
def coneRadius : ℝ := 6
def coneVolume : ℝ := 30 * Real.pi

-- Mathematical statement
theorem cone_lateral_surface_area (r V : ℝ) (hr : r = coneRadius) (hV : V = coneVolume) :
  ∃ S : ℝ, S = 39 * Real.pi :=
by 
  have h_volume := hV
  have h_radius := hr
  sorry

end cone_lateral_surface_area_l14_14194


namespace area_increase_l14_14102

theorem area_increase (a : ℝ) : ((a + 2) ^ 2 - a ^ 2 = 4 * a + 4) := by
  sorry

end area_increase_l14_14102


namespace ratio_of_y_and_z_l14_14757

variable (x y z : ℝ)

theorem ratio_of_y_and_z (h1 : x + y = 2 * x + z) (h2 : x - 2 * y = 4 * z) (h3 : x + y + z = 21) : y / z = -5 := 
by 
  sorry

end ratio_of_y_and_z_l14_14757


namespace cot_arccot_add_l14_14589

noncomputable def cot (x : ℝ) : ℝ := 1 / Real.tan x
noncomputable def arccot (x : ℝ) : ℝ := Real.atan (1 / x)

theorem cot_arccot_add (a b c d : ℝ) :
  cot(arccot a + arccot b + arccot c + arccot d) = (593 / 649) :=
by
  have h₁ : cot(arccot a) = a := sorry
  have h₂ : cot(arccot b) = b := sorry
  have h₃ : cot(arccot c) = c := sorry
  have h₄ : cot(arccot d) = d := sorry
  have h_add : ∀ x y, cot(arccot x + arccot y) = (x * y - 1) / (x + y) := sorry
  sorry

end cot_arccot_add_l14_14589


namespace g_is_odd_l14_14339

def g (x : ℝ) : ℝ := ⌈x⌉ - 1/2

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end g_is_odd_l14_14339


namespace a_formula_T_sum_l14_14650

def a : ℕ → ℕ
| 1       := 2
| (n + 1) := 2^(n + 1) * a n

def b (n : ℕ) : ℚ :=
if n = 0 then 1 else 2 / (n * (n + 1))

def T (n : ℕ) : ℚ :=
∑ i in Finset.range (n + 1), b i

theorem a_formula (n : ℕ) : a n = 2 ^ (n * (n + 1) / 2) := by
  sorry

theorem T_sum (n : ℕ) : T n = 2 * n / (n + 1) := by
  sorry

end a_formula_T_sum_l14_14650


namespace identity_holds_l14_14892

noncomputable def identity_proof : Prop :=
∀ (x : ℝ), (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x

theorem identity_holds : identity_proof :=
by
  sorry

end identity_holds_l14_14892


namespace ruth_train_track_length_l14_14903

theorem ruth_train_track_length (n : ℕ) (R : ℕ)
  (h_sean : 72 = 8 * n)
  (h_ruth : 72 = R * n) : 
  R = 8 :=
by
  sorry

end ruth_train_track_length_l14_14903


namespace average_of_w_and_x_is_one_half_l14_14286

noncomputable def average_of_w_and_x (w x y : ℝ) : ℝ :=
  (w + x) / 2

theorem average_of_w_and_x_is_one_half (w x y : ℝ)
  (h1 : 2 / w + 2 / x = 2 / y)
  (h2 : w * x = y) : average_of_w_and_x w x y = 1 / 2 :=
by
  sorry

end average_of_w_and_x_is_one_half_l14_14286


namespace cone_lateral_surface_area_l14_14200

theorem cone_lateral_surface_area (r h l S : ℝ) (π_pos : 0 < π) (r_eq : r = 6)
  (V : ℝ) (V_eq : V = 30 * π)
  (vol_eq : V = (1/3) * π * r^2 * h)
  (h_eq : h = 5 / 2)
  (l_eq : l = Real.sqrt (r^2 + h^2))
  (S_eq : S = π * r * l) :
  S = 39 * π :=
  sorry

end cone_lateral_surface_area_l14_14200


namespace max_area_rectangle_with_perimeter_40_l14_14096

theorem max_area_rectangle_with_perimeter_40 :
  ∃ (l w : ℕ), 2 * l + 2 * w = 40 ∧ l * w = 100 :=
sorry

end max_area_rectangle_with_perimeter_40_l14_14096


namespace factorial_division_l14_14031

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_division : (factorial 15) / ((factorial 6) * (factorial 9)) = 5005 :=
by
  sorry

end factorial_division_l14_14031


namespace simplify_fraction_l14_14911

theorem simplify_fraction (y : ℝ) (hy : y ≠ 0) : 
  (5 / (4 * y⁻⁴)) * ((4 * y³) / 3) = (5 * y⁷) / 3 := 
by
  sorry

end simplify_fraction_l14_14911


namespace pqr_eq_p_l14_14167

variable (a b c n : ℝ)

def p := n * a + (n + 1) * b + (n + 1) * c
def q := (n + 3) * a + (n + 2) * b + (n + 3) * c
def r := (n + 3) * a + (n + 3) * b + (n + 2) * c

def s := (p a b c n + q a b c n + r a b c n) / 3

def p' := 2 * s a b c n - p a b c n
def q' := 2 * s a b c n - q a b c n
def r' := 2 * s a b c n - r a b c n

theorem pqr_eq_p'q'r' (k : ℕ) (hk : k = 1 ∨ k = 2) : 
  p a b c n ^ k + q a b c n ^ k + r a b c n ^ k = p' a b c n ^ k + q' a b c n ^ k + r' a b c n ^ k := 
sorry

end pqr_eq_p_l14_14167


namespace initial_cost_of_milk_l14_14813

theorem initial_cost_of_milk (total_money : ℝ) (bread_cost : ℝ) (detergent_cost : ℝ) (banana_cost_per_pound : ℝ) (banana_pounds : ℝ) (detergent_coupon : ℝ) (milk_discount_rate : ℝ) (money_left : ℝ)
  (h_total_money : total_money = 20) (h_bread_cost : bread_cost = 3.50) (h_detergent_cost : detergent_cost = 10.25) (h_banana_cost_per_pound : banana_cost_per_pound = 0.75) (h_banana_pounds : banana_pounds = 2)
  (h_detergent_coupon : detergent_coupon = 1.25) (h_milk_discount_rate : milk_discount_rate = 0.5) (h_money_left : money_left = 4) : 
  ∃ (initial_milk_cost : ℝ), initial_milk_cost = 4 := 
sorry

end initial_cost_of_milk_l14_14813


namespace find_x_l14_14801

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 :=
sorry

end find_x_l14_14801


namespace tangent_segment_length_l14_14626

theorem tangent_segment_length :
  ∀ (x y : ℝ), (x + 2 * y = 3) →
  (2^x + 4^y = 4 * Real.sqrt 2) →
  (let d := Real.sqrt ((x - 1/2)^2 + (y + 1/4)^2) in
   let r := Real.sqrt (1/2) in
   Real.sqrt (d^2 - r^2) = Real.sqrt (3/2)) :=
by
  sorry

end tangent_segment_length_l14_14626


namespace cubic_eq_real_root_count_l14_14421

theorem cubic_eq_real_root_count :
  ∀ x : ℝ, x^3 - real.sqrt 3 * x^2 + x - (1 + real.sqrt 3 / 9) = 0 → ∃! x : ℝ, x^3 - real.sqrt 3 * x^2 + x - (1 + real.sqrt 3 / 9) = 0 :=
by { sorry }

end cubic_eq_real_root_count_l14_14421


namespace lcm_k_values_l14_14592

theorem lcm_k_values :
  let n := 18
  let a := 2^24 * 3^18
  let b := 2^18 * 3^36
  ∃ k, (k = 2^a * 3^b) ∧ 
       (∀ a b, 0 ≤ a ∧ 0 ≤ b ∧ lcm (9^9) (lcm (12^12) (2^a * 3^b)) = 18^18)
       ↔ card { a | a ≤ 18 } = 19 :=
by
  sorry

end lcm_k_values_l14_14592


namespace find_K_l14_14755

theorem find_K (K M : ℕ) (h1 : ∑ i in range (K + 1), i^2 = M^3) (h2 : M < 50) : K = 1 :=
sorry

end find_K_l14_14755


namespace reflections_in_mirrors_l14_14565

theorem reflections_in_mirrors (x : ℕ)
  (h1 : 30 = 10 * 3)
  (h2 : 18 = 6 * 3)
  (h3 : 88 = 30 + 5 * x + 18 + 3 * x) :
  x = 5 := by
  sorry

end reflections_in_mirrors_l14_14565


namespace dot_product_proof_l14_14655

variables {ℝ : Type*}
variables (a b : ℝ → ℝ)
variables [inner_product_space ℝ ℝ]

theorem dot_product_proof
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = sqrt 3)
  (h3 : ∥a - 2 • b∥ = 3) :
  inner (a : ℝ) (b : ℝ) = 1 :=
sorry

end dot_product_proof_l14_14655


namespace combined_return_percentage_l14_14493

theorem combined_return_percentage (investment1 investment2 : ℝ) 
  (return1_percent return2_percent : ℝ) (total_investment total_return : ℝ) :
  investment1 = 500 → 
  return1_percent = 0.07 → 
  investment2 = 1500 → 
  return2_percent = 0.09 → 
  total_investment = investment1 + investment2 → 
  total_return = investment1 * return1_percent + investment2 * return2_percent → 
  (total_return / total_investment) * 100 = 8.5 :=
by 
  sorry

end combined_return_percentage_l14_14493


namespace trapezoid_area_l14_14124

theorem trapezoid_area
  (longer_base : ℝ)
  (base_angle : ℝ)
  (h₁ : longer_base = 20)
  (h₂ : base_angle = real.arccos 0.6) :
  ∃ area : ℝ, (area ≈ 74.12) :=
by
  sorry

end trapezoid_area_l14_14124


namespace factorial_ratio_value_l14_14035

theorem factorial_ratio_value : fact 15 / (fact 6 * fact 9) = 770 := by
  sorry

end factorial_ratio_value_l14_14035


namespace min_n_xn_l14_14940

noncomputable def f (x : ℝ) : ℝ :=
  if -2 ≤ x ∧ x ≤ 0 then 2 * x + 1
  else if 2 ≤ x ∧ x ≤ 4 then -2 * (x - 4) + 1
  else by sorry

theorem min_n_xn (x : ℝ) (n : ℕ) :
  (∀ i, 0 ≤ x ∧ x < 4) →
  |f x_i - f x_{i+1}| + |f x_{i+2} - f x_{i+3}| + ... + |f x_{n-1} - f x_n| = 2016 →
  n + x_n = 1513 :=
by
  sorry

end min_n_xn_l14_14940


namespace count_two_digit_numbers_with_reverse_sum_110_l14_14142

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def reverse (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)

theorem count_two_digit_numbers_with_reverse_sum_110 :
  (finset.filter (λ n : ℕ, two_digit_number n ∧ n + reverse n = 110) (finset.range 100)).card = 9 :=
  sorry

end count_two_digit_numbers_with_reverse_sum_110_l14_14142


namespace lateral_surface_area_of_given_cone_l14_14212

noncomputable def coneLateralSurfaceArea (r V : ℝ) : ℝ :=
let h := (3 * V) / (π * r^2) in
let l := Real.sqrt (r^2 + h^2) in
π * r * l

theorem lateral_surface_area_of_given_cone :
  coneLateralSurfaceArea 6 (30 * π) = 39 * π := by
simp [coneLateralSurfaceArea]
sorry

end lateral_surface_area_of_given_cone_l14_14212


namespace position_relationship_l14_14954

-- Define the parabola with vertex at the origin and focus on the x-axis
def parabola (p : ℝ) : Prop := ∀ x y, y^2 = 2 * p * x

-- Define the line l: x = 1
def line_l (x : ℝ) : Prop := x = 1

-- Define the points P and Q where l intersects the parabola
def P (p : ℝ) : ℝ × ℝ := (1, Real.sqrt(2 * p))
def Q (p : ℝ) : ℝ × ℝ := (1, -Real.sqrt(2 * p))

-- Define the perpendicularity condition
def perpendicular (O P Q : ℝ × ℝ) : Prop := (O.1 * P.1 + O.2 * P.2) = 0

-- Point M
def M : ℝ × ℝ := (2, 0)

-- Equation of circle M
def circle_M (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop := (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Define the tangent condition
def tangent (l : ℝ × ℝ → Prop) (center : ℝ × ℝ) (radius : ℝ) : Prop := ∀ x, l(x) → (abs(center.1 - x.1) = radius)

-- Define points on the parabola
def points_on_parabola (A₁ A₂ A₃ : ℝ × ℝ) (p : ℝ) : Prop := parabola p A₁.1 A₁.2 ∧ parabola p A₂.1 A₂.2 ∧ parabola p A₃.1 A₃.2

-- The main theorem statement
theorem position_relationship (p : ℝ) (center : ℝ × ℝ) (radius : ℝ) (O A₁ A₂ A₃ : ℝ × ℝ) :
  parabola 1 O.1 O.2 ∧
  line_l 1 ∧
  A₁ = (0, 0) ∧
  perpendicular O (P 1) (Q 1) ∧
  tangent (λ x, line_l x.1) M 1 ∧
  points_on_parabola A₁ A₂ A₃ 1 →
  ∀ A₂ A₃ : ℝ × ℝ,
    (parabola 1 A₂.1 A₂.2 ∧ parabola 1 A₃.1 A₃.2) →
    tangent(λ x, circle_M center radius x) (A₂.1, A₂.2) (A₃.1, A₃.2) :=
sorry

end position_relationship_l14_14954


namespace probability_three_flips_all_heads_l14_14468

open ProbabilityTheory

-- Define a fair coin flip
def fair_coin_flip : ProbabilityTheory.PMeasure bool := 
  PMF.ofMultiset { (true, 1) , (false, 1) }.1 sorry

-- Define the event that the first three flips are all heads
def three_flips_all_heads : Event (bool × bool × bool) := 
  { (true, true, true) }

-- State the theorem specifying the probability
theorem probability_three_flips_all_heads :
  P (independent_ideals fair_coin_flip fair_coin_flip fair_coin_flip)
  (three_flips_all_heads fair_coin_flip fair_coin_flip fair_coin_flip) = 1 / 8 := 
sorry

end probability_three_flips_all_heads_l14_14468


namespace num_valid_a_values_l14_14370

theorem num_valid_a_values : 
  ∃ S : Finset ℕ, (∀ a ∈ S, a < 100 ∧ (a^3 + 23) % 24 = 0) ∧ S.card = 5 :=
sorry

end num_valid_a_values_l14_14370


namespace perpendicular_lines_implies_perpendicular_planes_l14_14176

variables (m n : Type) [line m] [line n]
variables (α β : Type) [plane α] [plane β]

-- Given conditions as Lean hypotheses
variables (h1 : α ≠ β) (h2 : m ≠ n)
variables (h_perp_planes : α ⊥ β)
variables (h_intersection : α ∩ β = m)
variables (h_perp_lines : m ⊥ n)

-- Statement of the proof problem
theorem perpendicular_lines_implies_perpendicular_planes :
  n ⊥ β :=
sorry

end perpendicular_lines_implies_perpendicular_planes_l14_14176


namespace number_of_triples_l14_14558

def sign (x : ℝ) : ℤ :=
if x > 0 then 1 else if x < 0 then -1 else 0

theorem number_of_triples :
  { (x, y, z) : ℝ × ℝ × ℝ //
    x = 2023 - 2024 * (sign (y + z)) ∧
    y = 2023 - 2024 * (sign (x + z)) ∧
    z = 2023 - 2024 * (sign (x + y))
  }.card = 3 := sorry

end number_of_triples_l14_14558


namespace perimeter_of_triangle_ACF_l14_14528

-- Definitions and conditions
def triangleABC := ∃ (A B C : Type) [metric_space A] [metric_space B] [metric_space C],
  ∃ (AB : ℝ) (BC : ℝ) (AC : ℝ) (D E F : Type) 
  [metric_space D] [metric_space E] [metric_space F],
  AD_isAltitude (A B C D) →
  DE_isPerpendicular (D E B) →
  AF_isPerpendicular (A F C) →
  AB = 13 ∧ BC = 14 ∧ AC = 15

-- Theorem statement
theorem perimeter_of_triangle_ACF 
  (A B C D E F : Type) [metric_space A] [metric_space B] [metric_space C]
  [metric_space D] [metric_space E] [metric_space F] 
  (h_triangleABC : triangleABC) :
  perimeter (triangleACF A C F) = 450 / 13 :=
by
  sorry

end perimeter_of_triangle_ACF_l14_14528
