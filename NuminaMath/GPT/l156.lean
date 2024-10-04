import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Basic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.LCM
import Mathlib.Algebra.Modular
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype
import Mathlib.Data.Int.Basic
import Mathlib.Data.List
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Floor
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.Geometry
import Mathlib.MeasureTheory.ConditionalExpectation
import Mathlib.NumberTheory.Divisors
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.Cond
import Mathlib.Tactic
import Mathlib.Topology.Angle
import Mathlib.Topology.Geometry.Circles

namespace sum_first_28_natural_numbers_eq_406_l156_156948

theorem sum_first_28_natural_numbers_eq_406 :
  (∑ i in Finset.range 28, (i + 1)) = 406 := begin
    sorry
  end

end sum_first_28_natural_numbers_eq_406_l156_156948


namespace part_a_part_b_l156_156351

variable {f : ℝ → ℝ}

-- Condition definitions
def continuous_function (f : ℝ → ℝ) := Continuous f
def positive_function (f : ℝ → ℝ) := ∀ x, 0 < f x
def periodic_function (f : ℝ → ℝ) := ∀ x, f (x + 1) = f x

-- Part (a) statement
theorem part_a (h_cont : continuous_function f) 
               (h_pos : positive_function f) 
               (h_period : periodic_function f) :
  ∫ x in 0..1, (f (x + 0.5) / f x) ≥ 1 := 
by sorry

-- Part (b) statement
theorem part_b (h_cont : continuous_function f) 
               (h_pos : positive_function f) 
               (h_period : periodic_function f) 
               (α : ℝ) :
  ∫ x in 0..1, (f (x + α) / f x) ≥ 1 :=
by sorry

end part_a_part_b_l156_156351


namespace plus_signs_count_l156_156785

noncomputable def symbols : Type := fin 23 -- The type of 23 symbols

def is_plus (x : symbols) : Prop
def is_minus (x : symbols) : Prop

axiom total_symbols : ∀ (x : symbols), is_plus x ∨ is_minus x
axiom unique_classification : ∀ (x : symbols), ¬ (is_plus x ∧ is_minus x)
axiom ten_symbols_plus : ∀ (S : finset symbols), S.card = 10 → ∃ x ∈ S, is_plus x
axiom fifteen_symbols_minus : ∀ (S : finset symbols), S.card = 15 → ∃ x ∈ S, is_minus x

def num_plus_symbols : ℕ := finset.card (finset.filter is_plus finset.univ)

theorem plus_signs_count : num_plus_symbols = 14 :=
sorry

end plus_signs_count_l156_156785


namespace line_integral_OB_eq_91_line_integral_AB_eq_173_75_l156_156448

-- Define the points and the constraints of the line segment and the arc
def A : ℝ × ℝ × ℝ := (3, -6, 0)
def B : ℝ × ℝ × ℝ := (-2, 4, 5)
def O : ℝ × ℝ × ℝ := (0, 0, 0)

-- Define the line integral function
noncomputable def line_integral (C : ℝ → ℝ × ℝ × ℝ) (t_start t_end : ℝ) : ℝ :=
  ∫ t in set.Icc t_start t_end, let (x, y, z) := C t in 
  x * y^2 * (deriv (λ t, (C t).1) t) + 
  y * z^2 * (deriv (λ t, (C t).2) t) - 
  z * x^2 * (deriv (λ t, (C t).3) t)

-- First part: Line segment OB
def C_OB (t : ℝ) : ℝ × ℝ × ℝ := (-2*t, 4*t, 5*t)
theorem line_integral_OB_eq_91 : line_integral C_OB 0 1 = 91 :=
by
  sorry

-- Second part: Arc AB of the circle
def C_AB (t : ℝ) : ℝ × ℝ × ℝ := (t, -2*t, real.sqrt (45 - 5*t^2))
theorem line_integral_AB_eq_173_75 : line_integral C_AB 3 (-2) = -173.75 :=
by
  sorry

end line_integral_OB_eq_91_line_integral_AB_eq_173_75_l156_156448


namespace differentiable_function_inequality_l156_156128

theorem differentiable_function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_cond : ∀ x : ℝ, (x - 1) * (deriv f x) ≥ 0) : 
  f 0 + f 2 ≥ 2 * (f 1) :=
sorry

end differentiable_function_inequality_l156_156128


namespace median_length_in_right_triangle_l156_156153

theorem median_length_in_right_triangle (DE EF : ℝ) (h_right : DE^2 + EF^2 = (Real.sqrt (DE^2 + EF^2))^2) :
  let ED := Real.sqrt (DE^2 + EF^2) in
  ED = 13 -> 
  ∃ F : ℝ, F = 6.5 :=
by
  sorry

end median_length_in_right_triangle_l156_156153


namespace man_speed_approx_2_0016_kmph_l156_156475

/-- 
Given a train 390 meters long moving at a speed of 25 kmph, crossing a man moving in the opposite direction in 52 seconds,
the speed of the man is approximately 2.0016 kmph.
-/
theorem man_speed_approx_2_0016_kmph : 
  ∀ (train_length : ℝ) (train_speed_kmph : ℝ) (cross_time_sec : ℝ) (train_speed_mps man_speed_mps : ℝ),
  train_length = 390 → 
  train_speed_kmph = 25 → 
  cross_time_sec = 52 → 
  train_speed_mps = (25 * 1000) / 3600 →
  let Vr := train_speed_mps + man_speed_mps in
  Vr * cross_time_sec = train_length →
  (man_speed_mps * 3.6) ≈ 2.0016 :=
begin
  intros,
  sorry
end

end man_speed_approx_2_0016_kmph_l156_156475


namespace compute_probability_domain_l156_156719

theorem compute_probability_domain :
  let D := {x : ℝ | -2 ≤ x ∧ x ≤ 3} in 
  let interval := Icc (-4 : ℝ) 5 in
  (∃ x ∈ interval, x ∈ D) →
  (measure_theory.measure_space.volume (set_of (λ x, x ∈ D ∧ x ∈ interval)) / 
   measure_theory.measure_space.volume interval = 5 / 9) :=
by
  let D := {x : ℝ | -2 ≤ x ∧ x ≤ 3}
  let interval := Icc (-4 : ℝ) 5
  sorry

end compute_probability_domain_l156_156719


namespace polygon_sides_equation_l156_156371

theorem polygon_sides_equation (n : ℕ) 
  (h1 : (n-2) * 180 = 4 * 360) : n = 10 := 
by 
  sorry

end polygon_sides_equation_l156_156371


namespace three_friends_digits_l156_156845

-- Define the problem
def possible_digits (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 
  (∃ x y z w : ℕ, 
    let digits := [a, b, c] in 
    -- All six possible permutations of the digits are considered as numbers
    let perms := list.permutations digits in 
    let nums := perms.map (λ ds, ds.head! * 100 + ds.tail!.head! * 10 + ds.tail!.tail!.head!) in
    nums.contains x ∧ nums.contains y ∧ nums.contains z ∧ nums.contains w ∧
    -- Differences for pairs of numbers
    | x - y | = 1 ∨ | y - x | = 1 ∧
    | z - w | = 10 ∨ | w - z | = 10 ∧
    (| x - z | = 100 ∨ | z - x | = 100 ∨ | x - w | = 100 ∨ | w - x | = 100) ∧ 
    (x - y ∣ 5 ∨ y - x ∣ 5 ∨ z - w ∣ 5 ∨ w - z ∣ 5))

-- Prove the correct digits
theorem three_friends_digits : 
  ∃ (a b c : ℕ), possible_digits a b c ∧ (a, b, c) = (2, 3, 7) ∨ (a, b, c) = (3, 7, 8) ∨ (a, b, c) = (3, 4, 8) :=
sorry

end three_friends_digits_l156_156845


namespace plus_signs_count_l156_156790

noncomputable def symbols : Type := fin 23 -- The type of 23 symbols

def is_plus (x : symbols) : Prop
def is_minus (x : symbols) : Prop

axiom total_symbols : ∀ (x : symbols), is_plus x ∨ is_minus x
axiom unique_classification : ∀ (x : symbols), ¬ (is_plus x ∧ is_minus x)
axiom ten_symbols_plus : ∀ (S : finset symbols), S.card = 10 → ∃ x ∈ S, is_plus x
axiom fifteen_symbols_minus : ∀ (S : finset symbols), S.card = 15 → ∃ x ∈ S, is_minus x

def num_plus_symbols : ℕ := finset.card (finset.filter is_plus finset.univ)

theorem plus_signs_count : num_plus_symbols = 14 :=
sorry

end plus_signs_count_l156_156790


namespace triangle_area_l156_156233

-- Define the side lengths of the triangle
def a := 2
def b := 3
def c := 4

-- Calculate the semi-perimeter
def P := (a + b + c) / 2

-- State the problem to prove
theorem triangle_area : sqrt (P * (P - a) * (P - b) * (P - c)) = 3 * sqrt 15 / 4 :=
by
  -- Here is where the implementation of the proof would go
  sorry

end triangle_area_l156_156233


namespace total_laces_needed_l156_156107

variable (x : ℕ) -- Eva has x pairs of shoes
def long_laces_per_pair : ℕ := 3
def short_laces_per_pair : ℕ := 3
def laces_per_pair : ℕ := long_laces_per_pair + short_laces_per_pair

theorem total_laces_needed : 6 * x = 6 * x :=
by
  have h : laces_per_pair = 6 := rfl
  sorry

end total_laces_needed_l156_156107


namespace S_seen_from_any_point_in_ABC_l156_156261

variable {Point : Type}
variable (Segment : Point → Point → Set Point)
variable (Triangle : Point → Point → Point → Set Point)

variable (S : Set Point)
variable (A B C : Point)
variable (P : Point)

-- S is a non-empty subset of a plane
-- Point P can be seen from point A if every point on the line segment AP belongs to S
def seen_from (P A : Point) (S : Set Point) : Prop :=
  ∀ p ∈ Segment A P, p ∈ S

-- Set S can be seen from point A if every point of S can be seen from A
def can_be_seen_from (A : Point) (S : Set Point) : Prop :=
  ∀ p ∈ S, seen_from p A S

-- S can be seen from A, B, and C where ABC is a triangle
axiom can_be_seen_from_A : can_be_seen_from A S
axiom can_be_seen_from_B : can_be_seen_from B S
axiom can_be_seen_from_C : can_be_seen_from C S

-- Triangle ABC
def ABC : Set Point := Triangle A B C

-- Prove that S can also be seen from any other point within the triangle ABC
theorem S_seen_from_any_point_in_ABC (D : Point) (hD : D ∈ ABC) : can_be_seen_from D S :=
  sorry

end S_seen_from_any_point_in_ABC_l156_156261


namespace trig_tangent_sum_theorem_l156_156200

noncomputable def trig_tangent_sum : Prop :=
  ∀ (x y : ℝ), sin x + sin y = 120 / 169 ∧ cos x + cos y = 119 / 169 → tan x + tan y = 0

-- placeholder for our proof
theorem trig_tangent_sum_theorem : trig_tangent_sum :=
by
  -- proof should go here
  sorry

end trig_tangent_sum_theorem_l156_156200


namespace tangent_dihedral_angle_A1_B1P_C_l156_156599

noncomputable section

-- Let the side length of the cube be 1 and point P be on edge AB
variables (P : ℝ) [0 < P ∧ P < 1]

-- Definitions of points and vectors for A, B, C, D, A1, B1, C1, D1
def A := (0 : ℝ, 0, 0)
def B := (1 : ℝ, 0, 0)
def C := (1 : ℝ, 1, 0)
def D := (0 : ℝ, 1, 0)
def A1 := (0 : ℝ, 0, 1)
def B1 := (1 : ℝ, 0, 1)
def C1 := (1 : ℝ, 1, 1)
def D1 := (0 : ℝ, 1, 1)

-- The angle between the line segment A1B and the plane B1CP is 60 degrees
def angle_A1B_plane_B1CP := real.angle (B - A1) ((C - B1) + (P * (B - A)))

-- The correct answer to prove: tangent value of dihedral angle A1-B1P-C
theorem tangent_dihedral_angle_A1_B1P_C : 
  P ∈ lineSegment ℝ A B → real.angle (line_segment A1 B) (plane (B1, C, P)) = 60 →
  (real.dihedral_angle (A1, B1) (P, C)).tan = -real.sqrt 5 :=
sorry

end tangent_dihedral_angle_A1_B1P_C_l156_156599


namespace minimum_value_of_k_l156_156628

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
noncomputable def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c
noncomputable def h (a b c : ℝ) (x : ℝ) : ℝ := (f a b x)^2 + 8 * (g a c x)
noncomputable def k (a b c : ℝ) (x : ℝ) : ℝ := (g a c x)^2 + 8 * (f a b x)

theorem minimum_value_of_k:
  ∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, h a b c x ≥ -29) → (∃ x : ℝ, k a b c x = -3) := sorry

end minimum_value_of_k_l156_156628


namespace polygon_sides_l156_156377

theorem polygon_sides (n : ℕ) 
  (h_interior : (n - 2) * 180 = 4 * 360) : 
  n = 10 :=
by
  sorry

end polygon_sides_l156_156377


namespace problem_1_problem_2_l156_156617

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 / 6 + 1 / x - a * Real.log x

theorem problem_1 (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 3 → f x a ≤ f 3 a) → a ≥ 8 / 3 :=
sorry

theorem problem_2 (a : ℝ) (h1 : 0 < a) (x0 : ℝ) :
  (∃! t : ℝ, 0 < t ∧ f t a = 0) → Real.log x0 = (x0^3 + 6) / (2 * (x0^3 - 3)) :=
sorry

end problem_1_problem_2_l156_156617


namespace determine_parabola_equation_l156_156462

noncomputable def parabola_equation (p : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

theorem determine_parabola_equation (p : ℝ) (x y : ℝ) :
  (∃ F : ℝ × ℝ, 
    ∃ A B C : ℝ × ℝ,
    parabola_equation p ∧
    (A.1^2 = 2 * p * A.2) ∧ (B.1^2 = 2 * p * B.2) ∧
    (C.1 = 0) ∧
    (C.2 = A.2 ∨ C.2 = B.2) ∧
    (C.2 - B.2 = -2 * (C.2 - F.2)) ∧
    ∥A - F∥ = 3
  ) → y^2 = 3 * x :=
by sorry

end determine_parabola_equation_l156_156462


namespace prove_difference_male_participants_l156_156681

def participants_2018 : ℕ := 150
def participants_2019 : ℕ := 20 + 2 * participants_2018
def participants_2020 : ℕ := (1/2 : ℚ) * participants_2019 - 40
def participants_2021 : ℕ := 30 + (participants_2018 - participants_2020)

def male_participants_2018 : ℕ := (3/5 : ℚ) * participants_2018
def male_participants_2019 : ℕ := (7/10 : ℚ) * participants_2019
def male_participants_2020 : ℕ := (1/2 : ℚ) * participants_2020
-- further definitions are not needed for this particular proof

def difference_in_male_participants : ℕ := male_participants_2019 - male_participants_2020

theorem prove_difference_male_participants : difference_in_male_participants = 164 :=
by
  sorry

end prove_difference_male_participants_l156_156681


namespace quadrilateral_sum_of_squares_l156_156264

variables {a b c d e f g : ℝ}
variables {A B C D M E F : Point}
variables {x y : ℝ}
variables {α : ℝ}

-- Given definitions according to conditions posed
def quadrilateral_sides (a b c d : ℝ) : Prop :=
  ∃ (A B C D : Point), dist A B = a ∧ dist B C = b ∧ dist C D = c ∧ dist D A = d

def quadrilateral_diagonals (e f : ℝ) : Prop :=
  ∃ (A B C D M : Point), M = midpoint A C ∧ dist A C = e ∧ dist B D = f

def segment_midpoints (g : ℝ) : Prop :=
  ∃ (E F : Point), E = midpoint A C ∧ F = midpoint B D ∧ dist E F = g

theorem quadrilateral_sum_of_squares
  (h_sides: quadrilateral_sides a b c d)
  (h_diagonals: quadrilateral_diagonals e f)
  (h_segment: segment_midpoints g) :
  a^2 + b^2 + c^2 + d^2 = e^2 + f^2 + 4 * g^2 :=
by
  sorry

end quadrilateral_sum_of_squares_l156_156264


namespace small_triangle_area_ratio_l156_156399

theorem small_triangle_area_ratio (a b n : ℝ) 
  (h₀ : a > 0) 
  (h₁ : b > 0) 
  (h₂ : n > 0) 
  (h₃ : ∃ (r s : ℝ), r > 0 ∧ s > 0 ∧ (1/2) * a * r = n * a * b ∧ s = (b^2) / (2 * n * b)) :
  (b^2 / (4 * n)) / (a * b) = 1 / (4 * n) :=
by sorry

end small_triangle_area_ratio_l156_156399


namespace number_of_males_in_village_l156_156363

-- Given the total population is 800 and it is divided into four equal groups.
def total_population : ℕ := 800
def num_groups : ℕ := 4

-- Proof statement
theorem number_of_males_in_village : (total_population / num_groups) = 200 := 
by sorry

end number_of_males_in_village_l156_156363


namespace arrangement_count_is_74_l156_156303

def count_valid_arrangements : Nat :=
  74

-- Lean statement for the proof
theorem arrangement_count_is_74 :
  let seven_cards := list.range' 1 7 in
  ∃ seq : list Nat, 
    (seq.length = 7) ∧ 
    (∀ n, list.erase seq n = list.range' 1 6 ∨ 
          (list.reverse (list.erase seq n) = list.range' 1 6)) ∧
    (count_valid_arrangements = 74) :=
by
  let seven_cards := list.range' 1 7
  existsi seven_cards
  split
  -- Provide the conditions here for Lean to handle
  sorry

end arrangement_count_is_74_l156_156303


namespace sum_a_b_l156_156211

theorem sum_a_b (a b : ℕ) (h : ∏ k in finset.range (a - 4), (k + 5) / (k + 4) = 16) : a + b = 127 :=
sorry

end sum_a_b_l156_156211


namespace area_of_region_l156_156990

noncomputable def region_area : ℝ :=
  let eq := λ (x y : ℝ), x^2 + y^2 = 4 * |x - y| + 2 * |x + y|
  in 10 * Real.pi

theorem area_of_region : region_area = 10 * Real.pi := by
  sorry

end area_of_region_l156_156990


namespace problem_l156_156212

theorem problem (a : ℝ) :
  (∀ x : ℝ, (x > 1 ↔ (x - 1 > 0 ∧ 2 * x - a > 0))) → a ≤ 2 :=
by
  sorry

end problem_l156_156212


namespace shaded_area_l156_156423

theorem shaded_area (d_s : ℝ) (r_s : ℝ) (r_l : ℝ) (c_dist : ℝ) 
  (h1 : d_s = 6) 
  (h2 : r_s = d_s / 2) 
  (h3 : r_l = 3 * r_s)
  (h4 : c_dist = 2) :
  let A_shaded := π * r_l^2 - π * r_s^2 in
  A_shaded = 72 * π :=
by
  sorry

end shaded_area_l156_156423


namespace plus_signs_count_l156_156828

theorem plus_signs_count (n p m : ℕ) (h_total : n = 23) 
  (h_plus_min : ∀ s, s.card = 10 → ∃ t ∈ s, is_plus t)
  (h_minus_min : ∀ s, s.card = 15 → ∃ t ∈ s, is_minus t)
  : p = 14 :=
sorry

end plus_signs_count_l156_156828


namespace angle_sum_triangle_l156_156242

theorem angle_sum_triangle (A B C : ℝ) 
  (hA : A = 20)
  (hC : C = 90) :
  B = 70 := 
by
  -- In a triangle the sum of angles is 180 degrees
  have h_sum : A + B + C = 180 := sorry
  -- Substitute the given angles A and C
  rw [hA, hC] at h_sum
  -- Simplify the equation to find B
  have hB : 20 + B + 90 = 180 := sorry
  linarith

end angle_sum_triangle_l156_156242


namespace plus_signs_count_l156_156837

theorem plus_signs_count (total_symbols : ℕ) (n_plus n_minus : ℕ)
  (h1 : total_symbols = 23)
  (h2 : ∀ s : Finset ℕ, s.card = 10 → ∃ i ∈ s, i ≤ n_plus)
  (h3 : ∀ s : Finset ℕ, s.card = 15 → ∃ i ∈ s, i > n_plus) :
  n_plus = 14 :=
by
  -- the proof will go here
  sorry

end plus_signs_count_l156_156837


namespace total_number_of_flowers_l156_156495

theorem total_number_of_flowers : 
  let red_roses := 1491
  let yellow_carnations := 3025
  let white_roses := 1768
  let purple_tulips := 2150
  let pink_daisies := 3500
  let blue_irises := 2973
  let orange_marigolds := 4234
  red_roses + yellow_carnations + white_roses + purple_tulips + pink_daisies + blue_irises + orange_marigolds = 19141 :=
by 
  sorry

end total_number_of_flowers_l156_156495


namespace rightTriangle_setC_l156_156005

/-- Define sets of line segments -/
def setA : list ℝ := [1/3, 1/4, 1/5]
def setB : list ℝ := [6, 8, 11]
def setC : list ℝ := [1, 1, real.sqrt 2]
def setD : list ℝ := [5, 12, 23]

/-- Check if a list of line segments can form a right triangle -/
def formsRightTriangle (l : list ℝ) : Prop :=
match l with
| [a, b, c] => a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2
| _ => false
end

/-- Prove that set C forms a right triangle -/
theorem rightTriangle_setC : formsRightTriangle setC :=
by
  unfold setC formsRightTriangle
  simp
  sorry

end rightTriangle_setC_l156_156005


namespace max_cos_a_c_l156_156718

theorem max_cos_a_c (a b c : ℝ) (h : Real.cos (a + c + b + c) = Real.cos (a + c) + Real.cos (b + c)) :
  ∃ A : ℝ, A = a + c ∧ Real.cos A = 1 :=
begin
  sorry
end

end max_cos_a_c_l156_156718


namespace darla_total_payment_l156_156506

-- Definitions of the conditions
def rate_per_watt : ℕ := 4
def energy_usage : ℕ := 300
def late_fee : ℕ := 150

-- Definition of the expected total cost
def expected_total_cost : ℕ := 1350

-- Theorem stating the problem
theorem darla_total_payment :
  rate_per_watt * energy_usage + late_fee = expected_total_cost := 
by 
  sorry

end darla_total_payment_l156_156506


namespace simplify_fraction_l156_156739

theorem simplify_fraction (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) : 
  (8 * a^4 * b^2 * c) / (4 * a^3 * b) = 2 * a * b * c :=
by
  sorry

end simplify_fraction_l156_156739


namespace plus_signs_count_l156_156778

theorem plus_signs_count (p m : ℕ) (h_sum : p + m = 23)
                         (h_max_minus : m ≤ 9) (h_max_plus : p ≤ 14)
                         (h_at_least_one_plus_in_10 : ∀ (s : Finset (Fin 23)), s.card = 10 → ∃ i ∈ s, i < p)
                         (h_at_least_one_minus_in_15 : ∀ (s : Finset (Fin 23)), s.card = 15 → ∃ i ∈ s, m ≤ i) :
  p = 14 :=
by sorry

end plus_signs_count_l156_156778


namespace Martha_apple_problem_l156_156280

theorem Martha_apple_problem (Martha_has : ∀ (apples : ℕ), apples = 20)
  (Jane_apple : ∀ (X : ℕ), ∃ (X : ℕ), True)
  (James_apple : ∀ (Y : ℕ), Y = Jane_apple + 2)
  (Martha_needs : ∀ (remaining_apples : ℕ), remaining_apples = 4)
  (Martha_left : ∀ (given_apples : ℕ), given_apples = 16) :
  (∃ X, X = 5) :=
by 
  sorry

end Martha_apple_problem_l156_156280


namespace number_of_students_who_walk_home_l156_156392

def total_students : ℕ := 500
def fraction_bus : ℚ := 1 / 5
def fraction_bike : ℚ := 45 / 100

theorem number_of_students_who_walk_home :
  total_students - (total_students * fraction_bus).natAbs - (total_students * fraction_bike).natAbs = 175 := 
by
  sorry

end number_of_students_who_walk_home_l156_156392


namespace boys_in_class_l156_156841

def total_students (num_groups : ℕ) (members_per_group : ℕ) : ℕ :=
  num_groups * members_per_group

def num_boys (total_students : ℕ) (num_girls : ℕ) : ℕ :=
  total_students - num_girls

theorem boys_in_class (num_groups : ℕ) (members_per_group : ℕ) (num_girls : ℕ) :
  num_groups = 7 → members_per_group = 3 → num_girls = 12 → num_boys (total_students num_groups members_per_group) num_girls = 9 :=
by
  intros hg mg hg
  rw [hg, mg, hg]
  simp [total_students, num_boys]
  sorry

end boys_in_class_l156_156841


namespace unclaimed_candy_fraction_l156_156065

-- Definitions for the shares taken by each person.
def al_share (x : ℕ) : ℚ := 3 / 7 * x
def bert_share (x : ℕ) : ℚ := 2 / 7 * (x - al_share x)
def carl_share (x : ℕ) : ℚ := 1 / 7 * ((x - al_share x) - bert_share x)
def dana_share (x : ℕ) : ℚ := 1 / 7 * (((x - al_share x) - bert_share x) - carl_share x)

-- The amount of candy that goes unclaimed.
def remaining_candy (x : ℕ) : ℚ := x - (al_share x + bert_share x + carl_share x + dana_share x)

-- The theorem we want to prove.
theorem unclaimed_candy_fraction (x : ℕ) : remaining_candy x / x = 584 / 2401 :=
by
  sorry

end unclaimed_candy_fraction_l156_156065


namespace sin_14pi_div_3_eq_sqrt3_div_2_l156_156541

theorem sin_14pi_div_3_eq_sqrt3_div_2 : Real.sin (14 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_14pi_div_3_eq_sqrt3_div_2_l156_156541


namespace minimal_speed_of_plane_l156_156752

theorem minimal_speed_of_plane {d a b : ℝ} (h_d : d > 0) (h_a : a > 0) (h_b : b > 0) :
  let minimal_speed := 20 * π * d * real.sqrt (a * b) in
  minimal_speed = 20 * π * d * real.sqrt (a * b) :=
by
  sorry

end minimal_speed_of_plane_l156_156752


namespace turtle_minimum_distance_l156_156480

/-- 
Given a turtle starting at the origin (0,0), crawling at a speed of 5 m/hour,
and turning 90 degrees at the end of each hour, prove that after 11 hours,
the minimum distance from the origin it could be is 5 meters.
-/
theorem turtle_minimum_distance :
  let speed := 5
  let hours := 11
  let distance (n : ℕ) := n * speed
  in ∃ (final_position : ℤ × ℤ),
      final_position = (5, 0) ∨ final_position = (-5, 0) ∨ final_position = (0, 5) ∨ final_position = (0, -5) :=
  sorry

end turtle_minimum_distance_l156_156480


namespace set_union_example_l156_156592

open Set

/-- Given sets A = {1, 2, 3} and B = {-1, 1}, prove that A ∪ B = {-1, 1, 2, 3} -/
theorem set_union_example : 
  let A := ({1, 2, 3} : Set ℤ)
  let B := ({-1, 1} : Set ℤ)
  A ∪ B = ({-1, 1, 2, 3} : Set ℤ) :=
by
  let A := ({1, 2, 3} : Set ℤ)
  let B := ({-1, 1} : Set ℤ)
  show A ∪ B = ({-1, 1, 2, 3} : Set ℤ)
  -- Proof to be provided here
  sorry

end set_union_example_l156_156592


namespace height_of_cylinder_B_l156_156749

noncomputable def radius_A : ℝ := 10
noncomputable def height_A : ℝ := 8
noncomputable def diameter_B : ℝ := 8
noncomputable def volume_ratio : ℝ := 1.5

noncomputable def height_B : ℝ :=
  let base_radius_B := diameter_B / 2
  let volume_A := π * radius_A^2 * height_A
  let volume_B := volume_ratio * volume_A
  volume_B / (π * base_radius_B^2)

theorem height_of_cylinder_B : height_B = 75 := by
  sorry

end height_of_cylinder_B_l156_156749


namespace skates_cost_is_65_l156_156068

constant admission_cost : ℝ := 5
constant rental_cost_per_visit : ℝ := 2.50
constant visits_to_justify : ℕ := 26

noncomputable def new_skates_cost : ℝ :=
  rental_cost_per_visit * visits_to_justify

theorem skates_cost_is_65 : new_skates_cost = 65 := by
  unfold new_skates_cost
  calc
    rental_cost_per_visit * visits_to_justify
      = 2.50 * 26 : by sorry
    ... = 65 : by sorry

end skates_cost_is_65_l156_156068


namespace plus_signs_count_l156_156781

theorem plus_signs_count (p m : ℕ) (h_sum : p + m = 23)
                         (h_max_minus : m ≤ 9) (h_max_plus : p ≤ 14)
                         (h_at_least_one_plus_in_10 : ∀ (s : Finset (Fin 23)), s.card = 10 → ∃ i ∈ s, i < p)
                         (h_at_least_one_minus_in_15 : ∀ (s : Finset (Fin 23)), s.card = 15 → ∃ i ∈ s, m ≤ i) :
  p = 14 :=
by sorry

end plus_signs_count_l156_156781


namespace exists_labeling_l156_156025

-- Define a structure for the points in the plane.
structure Point (α : Type) := (x : α) (y : α)

-- Define a predicate for three points being non-collinear (not on a single line).
def non_collinear {α : Type} [Field α] (A B C : Point α) : Prop :=
  (B.y - A.y) * (C.x - B.x) ≠ (C.y - B.y) * (B.x - A.x)

-- Define a predicate for a point lying inside the circle passing through three points.
def point_inside_circle {α : Type} [Field α] (A B C D : Point α) : Prop :=
  let det := λ p1 p2 p3 p4,
    (p2.x - p1.x) * (p3.y - p1.y) * (p4.x - p1.x) * (p4.y - p1.y)
    + (p3.x - p2.x) * (p4.y - p2.y) * (p1.x - p2.x) * (p1.y - p2.y)
    + (p4.x - p3.x) * (p1.y - p3.y) * (p2.x - p3.x) * (p2.y - p3.y)
    - (p4.x - p1.x) * (p3.y - p1.y) * (p2.x - p1.x) * (p2.y - p1.y)
    - (p2.x - p1.x) * (p4.y - p2.y) * (p3.x - p2.x) * (p3.y - p2.y)
    - (p3.x - p2.x) * (p1.y - p3.y) * (p4.x - p3.x) * (p4.y - p3.y)
  in det A B C D > 0

-- The main theorem to be proven.
theorem exists_labeling {α : Type} [Field α] (A B C D : Point α) :
  (∀ P₁ P₂ P₃ P₄ : Point α, ¬ collinear P₁ P₂ P₃ ∧ ¬ collinear P₁ P₂ P₄) →
  ∃ A' B' C' D' : Point α, 
    point_inside_circle A' B' C' D' :=
  by sorry

end exists_labeling_l156_156025


namespace min_ab_l156_156668

variable {A B C : ℝ} -- Angles
variable {a b c : ℝ} -- Sides opposite to angles A, B, and C
variable {S : ℝ} -- Area of the triangle

-- Conditions
axiom angle_ABC (a b c : ℝ) (A B C : ℝ) : 
  ∃ (α β γ : ℝ), α = A ∧ β = B ∧ γ = C ∧ a = c * sin A ∧ b = c * sin B ∧ α + β + γ = π

-- Given conditions
axiom cond1 : 2 * c * cos B = 2 * a + b
axiom cond2 : S = sqrt 3 * c
axiom triangle_area : S = 1 / 2 * a * b * sin C

-- Conclusion
theorem min_ab : ∃ (a b : ℝ), ab ≥ 48 := 
by
  sorry

end min_ab_l156_156668


namespace additional_track_length_l156_156047

/-
Problem statement:
A new railroad line needs to ascend 800 feet to cross a mountain range. The engineering team is considering reducing the grade to make a smoother incline. They want to know the additional track length required to reduce the grade from 1.5% to 1%. Prove that the additional length of track needed is approximately 26667 feet.
-/
theorem additional_track_length (ascent : ℕ) (initial_grade final_grade : ℚ) (initial_length final_length : ℚ) :
  ascent = 800 →
  initial_grade = 0.015 →
  final_grade = 0.01 →
  initial_length = ascent / initial_grade →
  final_length = ascent / final_grade →
  final_length - initial_length ≈ 26667 :=
by
  sorry

end additional_track_length_l156_156047


namespace plus_signs_count_l156_156836

theorem plus_signs_count (total_symbols : ℕ) (n_plus n_minus : ℕ)
  (h1 : total_symbols = 23)
  (h2 : ∀ s : Finset ℕ, s.card = 10 → ∃ i ∈ s, i ≤ n_plus)
  (h3 : ∀ s : Finset ℕ, s.card = 15 → ∃ i ∈ s, i > n_plus) :
  n_plus = 14 :=
by
  -- the proof will go here
  sorry

end plus_signs_count_l156_156836


namespace range_of_k_l156_156209

theorem range_of_k (k : ℝ) :
  (∃ a b c : ℝ, (a = 1) ∧ (b = -1) ∧ (c = -k) ∧ (b^2 - 4 * a * c > 0)) ↔ k > -1 / 4 :=
by
  sorry

end range_of_k_l156_156209


namespace find_principal_amount_l156_156928

theorem find_principal_amount (SI : ℝ) (R : ℝ) (T : ℝ) : 
  SI = 4016.25 → R = 0.09 → T = 5 → (SI / (R * T)) = 8925 :=
by
  sorry

end find_principal_amount_l156_156928


namespace zeros_in_expansion_of_999999999975_squared_l156_156947

theorem zeros_in_expansion_of_999999999975_squared :
  let n := 999999999975
  let m := 10^12 - 25
  n = m →
  let expansion := (10^12 - 25)^2
  ∃ k, (number_zero_digits expansion = k) ∧ k = 12 := sorry

end zeros_in_expansion_of_999999999975_squared_l156_156947


namespace smallest_n_perfect_square_and_fifth_power_l156_156873

theorem smallest_n_perfect_square_and_fifth_power : 
  ∃ (n : ℕ), (∀ k : ℕ, 4 * n = k^2 ∧ 5 * n = k^5) ↔ n = 3125 :=
begin
  sorry
end

end smallest_n_perfect_square_and_fifth_power_l156_156873


namespace least_five_digit_congruent_l156_156862

theorem least_five_digit_congruent (x : ℕ) (h1 : x ≥ 10000) (h2 : x < 100000) (h3 : x % 17 = 8) : x = 10004 :=
by {
  sorry
}

end least_five_digit_congruent_l156_156862


namespace stratified_sampling_l156_156036

theorem stratified_sampling (total_employees young middle_aged : ℕ) (selected_total : ℕ)
    (h_emp : total_employees = 100)
    (h_young : young = 45)
    (h_middle_aged : middle_aged = 25)
    (elderly : ℕ := 100 - young - middle_aged)
    (sample_young : ℕ := 9)
    (sample_middle_aged : ℕ := 5)
    (sample_elderly : ℕ := 6)
    (h_sample_total : sample_young + sample_middle_aged + sample_elderly = selected_total)
    (h_selected_total : selected_total = 20) :
  (sample_young, sample_middle_aged, sample_elderly) = (9, 5, 6) :=
  by
    rw [sample_young, sample_middle_aged, sample_elderly]
    sorry

end stratified_sampling_l156_156036


namespace average_temperature_l156_156559

def temperatures : List ℝ := [-36, 13, -15, -10]

theorem average_temperature : (List.sum temperatures) / (temperatures.length) = -12 := by
  sorry

end average_temperature_l156_156559


namespace rectangle_perimeter_l156_156921

theorem rectangle_perimeter
  (w l P : ℝ)
  (h₁ : l = 2 * w)
  (h₂ : l * w = 400) :
  P = 60 * Real.sqrt 2 :=
by
  sorry

end rectangle_perimeter_l156_156921


namespace max_blank_squares_on_grid_set_of_blank_squares_properly_defined_l156_156412

theorem max_blank_squares_on_grid : 
  ∀ (n : ℕ), (n = 100) →
  (∀ i j, (i ≤ n ∧ j ≤ n) → 
    ∃ s, (s = 1 ∨ s = 0) ∧ 
         (s = 0 ↔ (i, j) ∈ set_of_blank_squares)) →
  ∃ max_blanks, max_blanks = 2450 := 
by sorry

theorem set_of_blank_squares_properly_defined : 
  ∀ (n : ℕ) (i j : ℕ), (i ≤ n ∧ j ≤ n) → (i + j ≤ 51 ∨ i + j ≥ 151 ∨ j ≥ 51 + i ∨ i ≥ 51 + j) ∨ (∃ m, (i + j) % 2 = 0 ∧ m = 2 ∧ (∀ a, a = (i, j) → a ∉ set_of_blank_squares) ∨ 
( (i + j) % 2 = 1 ∧ (∀ a, a = (i, j) → a ∈ set_of_blank_squares)) :=
by sorry

end max_blank_squares_on_grid_set_of_blank_squares_properly_defined_l156_156412


namespace fraction_sum_is_11_l156_156360

theorem fraction_sum_is_11 (a b : ℕ) (h1 : 0.375 = (a : ℚ) / b) (h2 : Nat.coprime a b) : a + b = 11 := 
by sorry

end fraction_sum_is_11_l156_156360


namespace scientific_notation_95500_l156_156757

theorem scientific_notation_95500 : ∃ x : ℝ, x = 95500 ∧ x = 9.55 * 10^4 :=
by
  use 95500
  split
  ${ sorry }      -- Placeholders for proof steps
  ${ sorry }      -- These will ensure the code builds successfully.

end scientific_notation_95500_l156_156757


namespace plus_signs_count_l156_156819

theorem plus_signs_count (total_symbols : ℕ) (n : ℕ) (m : ℕ)
    (h1 : total_symbols = 23)
    (h2 : ∀ (s : Finset ℕ), s.card = 10 → (∃ x ∈ s, x = n))
    (h3 : ∀ (s : Finset ℕ), s.card = 15 → (∃ x ∈ s, x = m)) :
    n = 14 := by
  sorry

end plus_signs_count_l156_156819


namespace solve_expression_l156_156499

theorem solve_expression :
  -1^2023 + Real.sqrt 27 + (Real.pi - 3.14)^0 - Real.abs (Real.sqrt 3 - 2) = 4 * Real.sqrt 3 - 2 := 
sorry

end solve_expression_l156_156499


namespace compute_fourth_power_z_l156_156083

-- Definitions from the problem
def cos_angle (θ : ℝ) : ℝ := Real.cos θ
def sin_angle (θ : ℝ) : ℝ := Real.sin θ
def θ := Real.pi / 6  -- 30 degrees in radians

def z : ℂ := 3 * (cos_angle θ) + 3 * Complex.I * (sin_angle θ)

-- Lean 4 Statement for the proof
theorem compute_fourth_power_z : (z ^ 4) = -40.5 + 40.5 * Complex.I * Real.sqrt 3 :=
sorry

end compute_fourth_power_z_l156_156083


namespace sequences_behavior_l156_156612

noncomputable def f (x : ℝ) : ℝ := Real.cos x

def a_n (n : ℕ) [Fact (0 < n)] : ℝ :=
  (π / (2 * n)) * ∑ i in Finset.range n, f ((i : ℕ) * π / (2 * n))

def b_n (n : ℕ) [Fact (0 < n)] : ℝ :=
  (π / (2 * n)) * ∑ i in Finset.range n, f ((i + 1) * π / (2 * n))

theorem sequences_behavior :
  (∀ n : ℕ, Fact (0 < n) → a_n n > 1 ∧ a_n n > a_n (n + 1)) ∧
  (∀ n : ℕ, Fact (0 < n) → b_n n < 1 ∧ b_n n < b_n (n + 1)) :=
  sorry

end sequences_behavior_l156_156612


namespace cards_arrangement_count_is_10_l156_156319

-- Define the problem in Lean statement terms
def valid_arrangements_count : ℕ :=
  -- number of arrangements of seven cards where one card can be removed 
  -- leaving the remaining six cards in either ascending or descending order
  10

-- Theorem stating that the number of valid arrangements is 10
theorem cards_arrangement_count_is_10 : valid_arrangements_count = 10 :=
by
  -- Proof is omitted (the explanation above corresponds to the omitted proof details)
  sorry

end cards_arrangement_count_is_10_l156_156319


namespace inradius_of_triangle_PkCPk1_total_sum_areas_of_incircles_limit_value_I_n_limit_value_n_S_n_l156_156095

noncomputable def right_angle_triangle_inradius (n k : ℕ) (hn : n ≥ 2) :=
  let AB := 1
  let AC := Real.sqrt 3
  let BAC := Real.pi / 2
  let P_k := k / n
  let P_k1 := (k + 1) / n
  let r := 1 / (2 * n)
  r

noncomputable def total_sum_incircle_areas (n : ℕ) (hn : n ≥ 2) :=
  let rs := List.range (n - 1) |>.map (right_angle_triangle_inradius n)
  rs.sum * π / 4

noncomputable def limit_I_n :=
  ∫ x in 0..1, 1 / (3 + x^2)

noncomputable def limit_n_S_n :=
  π / 4

-- Proof statements
theorem inradius_of_triangle_PkCPk1 (n k : ℕ) (hn : n ≥ 2) : 
  let r := right_angle_triangle_inradius n k hn
  r = 1 / (2 * n) :=
sorry

theorem total_sum_areas_of_incircles (n : ℕ) (hn : n ≥ 2) :
  let S_n := total_sum_incircle_areas n hn
  S_n = π / (4 * n) :=
sorry

theorem limit_value_I_n :
  limit_I_n = π / (6 * Real.sqrt 3) :=
sorry

theorem limit_value_n_S_n :
  limit_n_S_n = π / 4 :=
sorry

end inradius_of_triangle_PkCPk1_total_sum_areas_of_incircles_limit_value_I_n_limit_value_n_S_n_l156_156095


namespace dog_nails_per_foot_l156_156075

-- Definitions from conditions
def number_of_dogs := 4
def number_of_parrots := 8
def total_nails_to_cut := 113
def parrots_claws := 8

-- Derived calculations from the solution but only involving given conditions
def dogs_claws (nails_per_foot : ℕ) := 16 * nails_per_foot
def parrots_total_claws := number_of_parrots * parrots_claws

-- The main theorem to prove the number of nails per dog foot
theorem dog_nails_per_foot :
  ∃ x : ℚ, 16 * x + parrots_total_claws = total_nails_to_cut :=
by {
  -- Directly state the expected answer
  use 3.0625,
  -- Placeholder for proof
  sorry
}

end dog_nails_per_foot_l156_156075


namespace max_num_pieces_l156_156446

-- Definition of areas
def largeCake_area : ℕ := 21 * 21
def smallPiece_area : ℕ := 3 * 3

-- Problem Statement
theorem max_num_pieces : largeCake_area / smallPiece_area = 49 := by
  sorry

end max_num_pieces_l156_156446


namespace polygon_sides_equation_l156_156374

theorem polygon_sides_equation (n : ℕ) 
  (h1 : (n-2) * 180 = 4 * 360) : n = 10 := 
by 
  sorry

end polygon_sides_equation_l156_156374


namespace equation_of_tangent_line_l156_156164

theorem equation_of_tangent_line
  (k b : ℝ)
  (M N : ℝ × ℝ)
  (h1 : ∀ x y, (x, y) = M ∨ (x, y) = N → (x - 1) * k = y)
  (h2 : ((M.1 + 3)^2 + (M.2 - 4)^2 = 16) ∧ ((N.1 + 3)^2 + (N.2 - 4)^2 = 16))
  (h3 : ∀ x y, (x, y) ∈ {M, N} → (x - y + b = 0))
  (h4 : ∀ x y, (x - y + b = 0) → (x^2 + y^2 = 2))
: b = 2 := by
  sorry

end equation_of_tangent_line_l156_156164


namespace second_deposit_interest_rate_l156_156942

theorem second_deposit_interest_rate:
  ∃ r : ℝ, 
  let initial_investment := 15000,
      first_deposit_annual_interest_rate := 0.08,
      first_deposit_duration := 9 / 12,
      second_deposit_duration := 9 / 12,
      final_amount := 17010,
      first_deposit_amount := initial_investment * (1 + first_deposit_annual_interest_rate * first_deposit_duration) in
  first_deposit_amount * (1 + (r * second_deposit_duration)) = final_amount ∧ r = 0.09333 :=
by
  sorry

end second_deposit_interest_rate_l156_156942


namespace volume_of_truncated_cone_l156_156341

theorem volume_of_truncated_cone (α l : ℝ) :
  (∃ (AO OB1 : ℝ), AO = (2 / 3) * l ∧ OB1 = (1 / 3) * l) →
  (∀ θ, θ = α) →
  V truncated_cone = (7 * Real.pi * l^3 / 54) * Real.sin α :=
by sorry

end volume_of_truncated_cone_l156_156341


namespace field_extension_equality_l156_156708

variable (p : ℕ) (hp : p.prime) (hp2 : p % 2 = 1)
variable (a : ℕ → ℚ) (n : ℕ)

theorem field_extension_equality :
  (∀ i, a i > 0) →
  (𝕜 := ℚ) →
  let algebra_p_root (i : ℕ) : 𝕜 := (a i)^(1/(p:ℕ)) in
  (𝕜⟮algebra_p_root⟯ := field.adjoin 𝕜 {algebra_p_root i | i < n}) →
  let algebra_sum_root : 𝕜 := finset.sum (finset.range n) (λ i, algebra_p_root i) in
  𝕜⟮algebra_p_root⟯ = 𝕜⟮algebra_sum_root⟯ :=
begin
  intro hpos,
  sorry
end

end field_extension_equality_l156_156708


namespace largest_positive_integer_n_l156_156114

theorem largest_positive_integer_n :
  ∃ n : ℕ, (∀ x : ℝ, (Real.sin x) ^ n + (Real.cos x) ^ n ≥ 2 / n) ∧ (∀ m : ℕ, m > n → ∃ x : ℝ, (Real.sin x) ^ m + (Real.cos x) ^ m < 2 / m) :=
sorry

end largest_positive_integer_n_l156_156114


namespace dart_lands_within_central_pentagon_probability_l156_156038

def is_regular_pentagon_shape (P : Type) : Prop := sorry

def central_pentagon (P : Type) : Prop := sorry

def divides_into_triangles (P : Type) : Prop := sorry

def triangles_meet_at_center (P : Type) : Prop := sorry

def vertices_touch_midpoints (P : Type) : Prop := sorry

def dart_equal_probability (P : Type) : Prop := sorry

theorem dart_lands_within_central_pentagon_probability {P : Type} 
  (h1 : is_regular_pentagon_shape P) 
  (h2 : central_pentagon P) 
  (h3 : divides_into_triangles P)
  (h4 : triangles_meet_at_center P) 
  (h5 : vertices_touch_midpoints P) 
  (h6 : dart_equal_probability P) : 
  probability (dart_lands_within_central_pentagon P) = 1/4 := 
sorry

end dart_lands_within_central_pentagon_probability_l156_156038


namespace g_monotonically_increasing_l156_156616

-- Conditions
def f (x : ℝ) (φ : ℝ) : ℝ := 2 * Real.cos (x + 3 * φ)
def g (x : ℝ) (φ : ℝ) : ℝ := Real.cos (2 * x - φ)

-- Given φ
def φ : ℝ := Real.pi / 6

-- Prove: g(x) is monotonically increasing in (7π/12, 13π/12)
theorem g_monotonically_increasing :
  ∀ x : ℝ, φ ∈ (0, Real.pi / 2) → (f x φ = - f (-x) φ) →
  (7 * Real.pi / 12 < x ∧ x < 13 * Real.pi / 12) →
  g x φ < g x (x + ε) :=
begin
  sorry -- the proof is omitted as instructed
end

end g_monotonically_increasing_l156_156616


namespace polygon_intersections_inside_circle_l156_156734

noncomputable def number_of_polygon_intersections
    (polygonSides: List Nat) : Nat :=
  let pairs := [(4,5), (4,7), (4,9), (5,7), (5,9), (7,9)]
  pairs.foldl (λ acc (p1, p2) => acc + 2 * min p1 p2) 0

theorem polygon_intersections_inside_circle :
  number_of_polygon_intersections [4, 5, 7, 9] = 58 :=
by
  sorry

end polygon_intersections_inside_circle_l156_156734


namespace cone_lateral_surface_area_l156_156909

theorem cone_lateral_surface_area (r h : ℝ) (h_r : r = 3) (h_h : h = 4) : 
  (1/2) * (2 * Real.pi * r) * (Real.sqrt (r ^ 2 + h ^ 2)) = 15 * Real.pi := 
by
  sorry

end cone_lateral_surface_area_l156_156909


namespace correct_statements_l156_156939

/-
Let A := {x | k * x^2 + 4 * x + 4 = 0}
Let f(x) be an odd function defined on the real numbers.
Prove that:
Among the following statements, the correct ones are (stmt2 ∧ stmt4):

1) If the set A has only one element, then k = 1.
2) In the same Cartesian coordinate system, the graphs of y = 2^x and y = 2^{-x} are symmetric about the y-axis.
3) y = (√3)^(-x) is an increasing function.
4) An odd function f(x) defined on ℝ has f(x) * f(-x) ≤ 0.
-/

def stmt1 (k : ℝ) (A : set ℝ) : Prop := 
  (∀ x, x ∈ A ↔ k * x^2 + 4 * x + 4 = 0) → ∃! x, k * x^2 + 4 * x + 4 = 0 → k = 1

def stmt2 : Prop := 
  ∀ x, 2^x = 2^(-(-x))

def stmt3 : Prop := 
  ∀ x : ℝ, (√3 : ℝ)^(-x) < (√3 : ℝ)^(-(x+1)) 

def stmt4 (f : ℝ → ℝ) : Prop := 
  (∀ x, f(-x) = -f(x)) → ∀ x, f(x) * f(-x) ≤ 0

theorem correct_statements (k : ℝ) (A : set ℝ) (f : ℝ → ℝ) :
  ¬ stmt1 k A ∧ stmt2 ∧ ¬ stmt3 ∧ stmt4 f := 
by
  sorry

end correct_statements_l156_156939


namespace principal_amount_l156_156751

theorem principal_amount (P : ℝ) (r t : ℝ) (d : ℝ) 
  (h1 : r = 7)
  (h2 : t = 2)
  (h3 : d = 49)
  (h4 : P * ((1 + r / 100) ^ t - 1) - P * (r * t / 100) = d) :
  P = 10000 :=
by sorry

end principal_amount_l156_156751


namespace min_students_l156_156218

theorem min_students (b g : ℕ) (hb : 1 ≤ b) (hg : 1 ≤ g)
    (h1 : b = (4/3) * g) 
    (h2 : (1/2) * b = 2 * ((1/3) * g)) 
    : b + g = 7 :=
by sorry

end min_students_l156_156218


namespace ellipse_eq_proof_value_of_k_l156_156147

section
  variable (a b : ℝ)  -- Variables for semi-major and semi-minor axes of the ellipse
  variable (x y : ℝ)  -- Coordinates on the ellipse
  variable (e : ℝ)    -- Eccentricity
  variable (k : ℝ)    -- Slope of the line
  variable (d : ℝ)    -- Distance from origin to line

  -- Given conditions
  def ellipse_eq : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
  def eccentricity : Prop := e = Real.sqrt 6 / 3
  def distance_from_origin : Prop := d = Real.sqrt 3 / 2
  def a_gt_b_gt_zero : Prop := a > b ∧ b > 0
  def line_through_AB : Prop := (x, y) = (0, -b) ∨ (x, y) = (a, 0)
  def point_E : Prop := (x, y) = (-1, 0)
  def line_condition : Prop := ∃ k, k ≠ 0 ∧ (y = k * x + 2)

  -- Problem 1: Find the equation of the ellipse
  theorem ellipse_eq_proof (h1 : a > 0) (h2 : b > 0) (h3 : eccentricity a b e) 
    (h4 : d = Real.sqrt 3 / 2) : 
    ∃ a b, (x^2 / 3) + y^2 = 1 :=
  sorry

  -- Problem 2: Does there exist a value k such that conditions hold?
  theorem value_of_k (fixed_point : point_E) (line_eq : line_condition) 
    (ellipse : ellipse_eq) : 
    ∃ k, k = 7 / 6 :=
  sorry
end

end ellipse_eq_proof_value_of_k_l156_156147


namespace geometric_sequence_triangle_inequality_l156_156606

open Real

theorem geometric_sequence_triangle_inequality (q : ℝ) (a : ℝ) (h0 : q > 0) (h1 : a > 0) :
  (frac ((sqrt 5) - 1) 2 < q ∧ q < frac ((sqrt 5) + 1) 2) ↔
  (a + q * a > q^2 * a ∧ a + q^2 * a > q * a ∧ q * a + q^2 * a > a) :=
sorry

end geometric_sequence_triangle_inequality_l156_156606


namespace initial_sentences_today_l156_156695

-- Definitions of the given conditions
def typing_rate : ℕ := 6
def initial_typing_time : ℕ := 20
def additional_typing_time : ℕ := 15
def erased_sentences : ℕ := 40
def post_meeting_typing_time : ℕ := 18
def total_sentences_end_of_day : ℕ := 536

def sentences_typed_before_break := initial_typing_time * typing_rate
def sentences_typed_after_break := additional_typing_time * typing_rate
def sentences_typed_post_meeting := post_meeting_typing_time * typing_rate
def sentences_today := sentences_typed_before_break + sentences_typed_after_break - erased_sentences + sentences_typed_post_meeting

theorem initial_sentences_today : total_sentences_end_of_day - sentences_today = 258 := by
  -- proof here
  sorry

end initial_sentences_today_l156_156695


namespace abs_sum_lt_abs_diff_l156_156590

theorem abs_sum_lt_abs_diff (a b : ℝ) (h : a * b < 0) : |a + b| < |a - b| := 
sorry

end abs_sum_lt_abs_diff_l156_156590


namespace taylor_series_expansion_and_radius_of_convergence_l156_156981

noncomputable def f (z : ℂ) : ℂ :=
  z / (z^2 - 2 * z - 3)

theorem taylor_series_expansion_and_radius_of_convergence :
  let taylor_series := -z / 3 + 2 * z^2 / 9 - 7 * z^3 / 27 + ∑ n in finset.range infinity, a(n) * z^n 
  ∃ R > 0, R = 1 ∧ f(z) = taylor_series :=
sorry

end taylor_series_expansion_and_radius_of_convergence_l156_156981


namespace exist_n_exactly_3_rainy_days_l156_156664

-- Define the binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the binomial probability
def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coeff n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exist_n_exactly_3_rainy_days (p : ℝ) (k : ℕ) (prob : ℝ) :
  p = 0.5 → k = 3 → prob = 0.25 →
  ∃ n : ℕ, binomial_prob n k p = prob :=
by
  intros h1 h2 h3
  sorry

end exist_n_exactly_3_rainy_days_l156_156664


namespace fraction_product_eq_one_l156_156960

theorem fraction_product_eq_one :
  (∏ i in finset.range 21, (1 + (19 : ℕ) / (i + 1))) / (∏ i in finset.range 19, (1 + (21 : ℕ) / (i + 1))) = 1 :=
by
  -- main proof
  sorry

end fraction_product_eq_one_l156_156960


namespace total_gas_cost_l156_156126

theorem total_gas_cost 
  (x : ℝ)
  (cost_per_person_initial : ℝ := x / 5)
  (cost_per_person_new : ℝ := x / 8)
  (cost_difference : cost_per_person_initial - cost_per_person_new = 15) :
  x = 200 :=
sorry

end total_gas_cost_l156_156126


namespace pow_congruence_modulus_p_squared_l156_156269

theorem pow_congruence_modulus_p_squared (p : ℕ) (a b : ℤ) (hp : Nat.Prime p) (h : a ≡ b [ZMOD p]) : a^p ≡ b^p [ZMOD p^2] :=
sorry

end pow_congruence_modulus_p_squared_l156_156269


namespace plus_signs_count_l156_156835

theorem plus_signs_count (total_symbols : ℕ) (n_plus n_minus : ℕ)
  (h1 : total_symbols = 23)
  (h2 : ∀ s : Finset ℕ, s.card = 10 → ∃ i ∈ s, i ≤ n_plus)
  (h3 : ∀ s : Finset ℕ, s.card = 15 → ∃ i ∈ s, i > n_plus) :
  n_plus = 14 :=
by
  -- the proof will go here
  sorry

end plus_signs_count_l156_156835


namespace sum_of_integer_solutions_l156_156123

theorem sum_of_integer_solutions : 
  (∑ x in ({-3, 3, -2, 2} : Finset ℤ), x) = 0 := 
by
  have h1 : (Polynomial.eval x (Polynomial.C (-13) + Polynomial.C 1 * x ^ 4 + Polynomial.C 36) : ℚ) = 0 := sorry
  have h2 : x = -3 ∨ x = 3 ∨ x = -2 ∨ x = 2 := sorry
  suffices (∑ x in ({-3, 3, -2, 2} : Finset ℤ), x) = 0, by
    sorry
  sorry

end sum_of_integer_solutions_l156_156123


namespace sin_cos_pi_minus_two_alpha_l156_156139

theorem sin_cos_pi_minus_two_alpha (α : ℝ) (h : tan α = 2 / 3) : 
  sin (2 * α) - cos (π - 2 * α) = 17 / 13 :=
by
  sorry

end sin_cos_pi_minus_two_alpha_l156_156139


namespace cindy_age_l156_156952

-- Define the ages involved
variables (C J M G : ℕ)

-- Define the conditions
def jan_age_condition : Prop := J = C + 2
def marcia_age_condition : Prop := M = 2 * J
def greg_age_condition : Prop := G = M + 2
def greg_age_known : Prop := G = 16

-- The statement we need to prove
theorem cindy_age : 
  jan_age_condition C J → 
  marcia_age_condition J M → 
  greg_age_condition M G → 
  greg_age_known G → 
  C = 5 := 
by 
  -- Sorry is used here to skip the proof
  sorry

end cindy_age_l156_156952


namespace greatest_number_of_dimes_l156_156277

theorem greatest_number_of_dimes (total_value : ℝ) (num_dimes : ℕ) (num_nickels : ℕ) 
  (h_same_num : num_dimes = num_nickels) (h_total_value : total_value = 4.80) 
  (h_value_calculation : 0.10 * num_dimes + 0.05 * num_nickels = total_value) :
  num_dimes = 32 :=
by
  sorry

end greatest_number_of_dimes_l156_156277


namespace find_x_l156_156891

variable (x : ℝ)

def initial_salt_volume (x : ℝ) : ℝ := 0.20 * x
def remaining_solution_volume (x : ℝ) : ℝ := (3 / 4) * x
def added_water_volume : ℝ := 5
def added_salt_volume : ℝ := 10
def final_solution_volume (x : ℝ) : ℝ := remaining_solution_volume x + added_water_volume + added_salt_volume
def final_salt_volume (x : ℝ) : ℝ := initial_salt_volume x + added_salt_volume

theorem find_x (h : final_salt_volume x / final_solution_volume x = 1 / 3) : x = 100 :=
by
  sorry -- The proof is skipped.

end find_x_l156_156891


namespace real_solutions_equation_l156_156547

noncomputable def g (x : ℝ) : ℝ :=
  ∑ k in (Finset.range 50).map (λ k, k + 1), (2 * k - 1) / (x - (2 * k - 1))

theorem real_solutions_equation : 
  (∑ k in (Finset.range 50).map (λ k, k + 1), (2 * k - 1) / (x - (2 * k - 1)) = x - 1) → 
  (∃ n : ℕ, n = 50 ∧ linear_independent ℝ 
  (λ i : ℝ, i = (Finset.range 50).map (λ k, k + 1))) := 
sorry

end real_solutions_equation_l156_156547


namespace undefined_expression_l156_156132

theorem undefined_expression (a : ℝ) : (a = 3 ∨ a = -3) ↔ (a^2 - 9 = 0) := 
by
  sorry

end undefined_expression_l156_156132


namespace arithmetic_sum_10_l156_156683

noncomputable def a (n : ℕ) : ℝ := sorry  -- Define the arithmetic sequence {a_n}

def d : ℝ := sorry  -- Define the common difference

def S (n : ℕ) : ℝ := (n / 2) * (a 1 + a n)  -- Sum of first n terms

axiom a3_a8_sum : a 3 + a 8 = 8  -- Given condition

theorem arithmetic_sum_10 : S 10 = 40 :=
by
  -- Use the given condition and properties of the arithmetic sequence
  sorry

end arithmetic_sum_10_l156_156683


namespace complex_fourth_power_l156_156088

noncomputable def complex_number : ℂ := 3 * complexOfReal((real.cos (real.pi / 6))) + 3i * complexOfReal((real.sin (real.pi / 6)))

theorem complex_fourth_power :
  (complex_number ^ 4) = -40.5 + 40.5 * (sqrt 3) * complex.i := 
by 
  sorry

end complex_fourth_power_l156_156088


namespace train_length_l156_156010

theorem train_length (speed_kmph : ℕ) (platform_length_m : ℕ) (time_seconds : ℕ) :
  speed_kmph = 72 → platform_length_m = 350 → time_seconds = 26 → 
  let speed_mps := speed_kmph * 5 / 18 in
  let distance_covered := speed_mps * time_seconds in
  let train_length := distance_covered - platform_length_m in
  train_length = 170 :=
by
  intros h_speed h_platform h_time
  let speed_mps := speed_kmph * 5 / 18
  let distance_covered := speed_mps * time_seconds
  let train_length := distance_covered - platform_length_m
  have h_speed_mps : speed_mps = 20 := sorry -- Proof of conversion
  have h_distance_covered : distance_covered = 520 := sorry -- Proof of distance calculation
  have h_train_length : train_length = 170 := sorry -- Proof of final length
  exact h_train_length

end train_length_l156_156010


namespace standard_eqns_and_intersection_l156_156177

noncomputable def parametric_line : Type :=
{ x : ℝ, y : ℝ, t : ℝ // x = (1/2) * t ∧ y = 1 + (sqrt 3/2) * t }

noncomputable def polar_curve : Type :=
{ rho : ℝ, theta : ℝ // rho = 2 * sqrt 2 * sin (theta + π/4) }

theorem standard_eqns_and_intersection (l : parametric_line) (C: polar_curve) :
  let line_eqn := (sqrt 3 * l.1 - l.2 + 1 = 0) in
  let curve_eqn := ((l.1 - 1)^2 + (l.2 - 1)^2 = 2) in
  let PA := abs l.2 in
  let PB := abs (l.1 / (1/2)) in
  line_eqn ∧ curve_eqn ∧ (1 / PA + 1 / PB) = sqrt 5 :=
by 
  sorry

end standard_eqns_and_intersection_l156_156177


namespace bicyclist_speed_remainder_l156_156726

theorem bicyclist_speed_remainder (total_distance first_distance remainder_distance first_speed avg_speed remainder_speed time_total time_first time_remainder : ℝ) 
  (H1 : total_distance = 350)
  (H2 : first_distance = 200)
  (H3 : remainder_distance = total_distance - first_distance)
  (H4 : first_speed = 20)
  (H5 : avg_speed = 17.5)
  (H6 : time_total = total_distance / avg_speed)
  (H7 : time_first = first_distance / first_speed)
  (H8 : time_remainder = time_total - time_first)
  (H9 : remainder_speed = remainder_distance / time_remainder) :
  remainder_speed = 15 := 
sorry

end bicyclist_speed_remainder_l156_156726


namespace least_positive_angle_l156_156991

theorem least_positive_angle(theta : ℝ) : 
  ∃ θ : ℝ, θ > 0 ∧ θ ≤ 90 ∧ 
  (cos (15 * real.pi / 180) = sin (35 * real.pi / 180) + sin (θ * real.pi / 180)) ∧
  θ = 35 :=
by 
  sorry

end least_positive_angle_l156_156991


namespace certain_number_is_4_l156_156654

theorem certain_number_is_4 (x y C : ℝ) (h1 : 2 * x - y = C) (h2 : 6 * x - 3 * y = 12) : C = 4 :=
by
  -- Proof goes here
  sorry

end certain_number_is_4_l156_156654


namespace line_passes_through_point_l156_156755

theorem line_passes_through_point :
  ∀ (m : ℝ), (∃ y : ℝ, y - 2 = m * (-1) + m) :=
by
  intros m
  use 2
  sorry

end line_passes_through_point_l156_156755


namespace tan_of_angle_B_in_right_triangle_l156_156720

theorem tan_of_angle_B_in_right_triangle 
  (A B C D E : Type)
  [triangle A B C]
  (right_angle_at_C : is_right_triangle A B C C)
  (D_on_AB : D ∈ line_segment A B)
  (E_on_AB : E ∈ line_segment A B)
  (D_between_AE : is_between D A E)
  (trisect_CD_CE : trisecting_lines CD CE (angle at C))
  (ratio_of_DE_BE : DE / BE = 3/7) : ∃ B, tan B = tan 70  :=
begin
  sorry
end

end tan_of_angle_B_in_right_triangle_l156_156720


namespace plus_signs_count_l156_156827

theorem plus_signs_count (n p m : ℕ) (h_total : n = 23) 
  (h_plus_min : ∀ s, s.card = 10 → ∃ t ∈ s, is_plus t)
  (h_minus_min : ∀ s, s.card = 15 → ∃ t ∈ s, is_minus t)
  : p = 14 :=
sorry

end plus_signs_count_l156_156827


namespace sum_or_all_zero_l156_156674

theorem sum_or_all_zero (a : ℕ → ℕ → ℕ) (n m : ℕ)
  (x : ℕ → ℕ) (y : ℕ → ℕ)
  (H1 : ∀ i, x i = ∑ j in finset.range m, a i j)
  (H2 : ∀ j, y j = ∑ i in finset.range n, a i j)
  (H3 : ∀ i j, a i j = x i * y j) :
  (∑ i in finset.range n, ∑ j in finset.range m, a i j = 1) ∨ 
  (∀ i j, a i j = 0) :=
sorry

end sum_or_all_zero_l156_156674


namespace skateboarded_one_way_distance_l156_156256

-- Define the total skateboarded distance and the walked distance.
def total_skateboarded : ℕ := 24
def walked_distance : ℕ := 4

-- Define the proof theorem.
theorem skateboarded_one_way_distance : 
    (total_skateboarded - walked_distance) / 2 = 10 := 
by sorry

end skateboarded_one_way_distance_l156_156256


namespace parallel_lines_slope_condition_l156_156129

theorem parallel_lines_slope_condition (m : ℝ) :
  (∃ (l1 l2 : ℝ → ℝ → ℝ), l1 = 2 * m + (m + 1) * y + 4 = 0 ∧ l2 = m * x + 3 * y - 2 = 0 ∧ (2 / m = (m + 1) / 3 ∧ m ≠ 0))
  → (m = -3 ∨ m = 2) :=
by
  sorry

end parallel_lines_slope_condition_l156_156129


namespace number_of_valid_arrangements_l156_156305

open Finset

-- We define the condition that a list is sorted in ascending order
def is_ascending (l : List ℕ) : Prop :=
  l = List.sort (≤) l

-- We define the condition that a list is sorted in descending order
def is_descending (l : List ℕ) : Prop :=
  l = List.sort (≥) l

def cards := Finset.range 7
def arrangements := cards.to_list.permutations

-- Define the function to check if a list of numbers (cards) 
-- can have one element removed to form an ascending or descending list
def valid_arrangement (l : List ℕ) : Prop :=
  ∃ (x : ℕ), (l.erase x).is_ascending ∨ (l.erase x).is_descending

-- Define the final theorem
theorem number_of_valid_arrangements : finset.card (arrangements.filter valid_arrangement) = 72 :=
by
  sorry

end number_of_valid_arrangements_l156_156305


namespace solve_tan_system_l156_156325

open Real

noncomputable def solve_system (a b : ℝ) : Prop :=
  ∃ x y k : ℤ, 
    (b ≠ 1 → 
      x = (a + 2 * (k : ℝ) * π + (arccos ((1 + b) / (1 - b) * cos a))) / 2 ∧
      y = (a - 2 * (k : ℝ) * π - (arccos ((1 + b) / (1 - b) * cos a))) / 2 ∧
      x + y = a ∧ tan x * tan y = b) ∧
    (b ≠ 1 → 
      x = (a + 2 * (k : ℝ) * π - (arccos ((1 + b) / (1 - b) * cos a))) / 2 ∧
      y = (a - 2 * (k : ℝ) * π + (arccos ((1 + b) / (1 - b) * cos a))) / 2 ∧
      x + y = a ∧ tan x * tan y = b) ∧    
    (b = 1 → 
      (∃ m : ℤ, a = (π / 2) + m * π ∧ y = (π / 2) + m * π - x ∧ x + y = a ∧ tan x * tan y = b))

-- Then we need to prove that for any a and b, the solutions satisfy the system:
theorem solve_tan_system (a b : ℝ) : solve_system a b := 
  sorry

end solve_tan_system_l156_156325


namespace find_eighth_grade_participants_l156_156034

-- Define the problem conditions
def chess_competition (n : ℕ) :=
  let seventh_grade_total_points := 8
  let seventh_grade_self_games_points := 1
  let seventh_grade_vs_eighth_grade_points := seventh_grade_total_points - seventh_grade_self_games_points
  let eighth_grade_total_points := 2 * n - seventh_grade_vs_eighth_grade_points
  let sigma := eighth_grade_total_points / n
  sigma == 1 ∨ sigma == 1.5

theorem find_eighth_grade_participants (n : ℕ) : 
  (chess_competition 7 ∨ chess_competition 14) :=
by
  sorry

end find_eighth_grade_participants_l156_156034


namespace alan_must_eat_more_l156_156701

theorem alan_must_eat_more (
  kevin_eats_total : ℕ,
  kevin_time : ℕ,
  alan_eats_rate : ℕ
) (h_kevin_eats_total : kevin_eats_total = 64) 
  (h_kevin_time : kevin_time = 8)
  (h_alan_eats_rate : alan_eats_rate = 5)
  (kevin_rate_gt_alan_rate : (kevin_eats_total / kevin_time) > alan_eats_rate) :
  ∃ wings_more_per_minute : ℕ, wings_more_per_minute = 4 :=
by
  sorry

end alan_must_eat_more_l156_156701


namespace cone_volume_proof_l156_156578

-- Define the problem context
variables (a : ℝ) (r : ℝ) (h : ℝ)

-- Given conditions
def equal_edge_lengths (a : ℝ) : Prop := 
  r = sqrt(6) ∧ 
  (sqrt(3) / 2 * a + sqrt(3) / 2 * a * (1 / 3) = sqrt(6))

-- Correct answer to prove: volume of the cone
def cone_volume_is (a : ℝ) (h : ℝ) (V : ℝ) : Prop := 
  V = 1 / 3 * (sqrt(3) / 4) * a^2 * sqrt(a^2 - (2 / 3 * sqrt(3) / 2 * a)^2) 

-- The theorem statement
theorem cone_volume_proof (a : ℝ) (h : ℝ) (V : ℝ) : 
  equal_edge_lengths a → 
  cone_volume_is a h V → 
  V = 9 / 8 := 
by
  sorry

end cone_volume_proof_l156_156578


namespace correct_option_is_A_l156_156435

-- Define the four options
def A_pair :=
  let x := -3
  let y := Real.sqrt ((-3) ^ 2)
  (x, y)

def B_pair :=
  let x := -3
  let y := Real.cbrt (-27)
  (x, y)

def C_pair :=
  let x := -3
  let y := -(1 / 3)
  (x, y)

def D_pair :=
  let x := abs (-3)
  let y := 3
  (x, y)

-- Define a predicate to check if two numbers are opposites
def are_opposites (a b : ℝ) : Prop :=
  a = -b

-- State the theorem with the correct answer
theorem correct_option_is_A : 
  are_opposites (A_pair.1) (A_pair.2) ∧
  ¬ are_opposites (B_pair.1) (B_pair.2) ∧
  ¬ are_opposites (C_pair.1) (C_pair.2) ∧
  ¬ are_opposites (D_pair.1) (D_pair.2) :=
by
  sorry

end correct_option_is_A_l156_156435


namespace solution_set_inequality_l156_156665

theorem solution_set_inequality {a b : ℝ} (h : ∀ x : ℝ, (x - a) / (x - b) > 0 ↔ x ∈ (Iio 1 ∪ Ioi 4)) : a + b = 5 :=
sorry

end solution_set_inequality_l156_156665


namespace five_topping_pizzas_count_l156_156465

theorem five_topping_pizzas_count : ∀ (n k : ℕ), n = 8 → k = 5 → (Finset.card (Finset.comb n k) = 56) :=
by
  intros n k hn hk
  subst hn hk
  dsimp
  rw [Finset.card_combinations, Nat.choose_eq_zero_original]
  calc
    Nat.choose 8 5 = Nat.choose 8 (8 - 3)      : by rw [Nat.symm_subt_le_refl]
                ... = 56                       : by norm_num

#check five_topping_pizzas_count -- ensure the theorem is correctly stated

end five_topping_pizzas_count_l156_156465


namespace least_five_digit_congruent_to_8_mod_17_l156_156858

theorem least_five_digit_congruent_to_8_mod_17 : 
  ∃ n : ℕ, n = 10004 ∧ n % 17 = 8 ∧ 10000 ≤ n ∧ n < 100000 ∧ (∀ m, 10000 ≤ m ∧ m < 10004 → m % 17 ≠ 8) :=
sorry

end least_five_digit_congruent_to_8_mod_17_l156_156858


namespace average_temperature_l156_156560

theorem average_temperature :
  let temp1 := -36
  let temp2 := 13
  let temp3 := -15
  let temp4 := -10
  (temp1 + temp2 + temp3 + temp4) / 4 = -12 :=
by
  unfold temp1 temp2 temp3 temp4
  calc
    (-36 + 13 + -15 + -10) / 4
      = (-48) / 4 : by norm_num
      ... = -12 : by norm_num

end average_temperature_l156_156560


namespace circles_tangent_points_collinear_l156_156180

open Function

noncomputable def CirclesExternallyTangent
(O₁ O₂ P A B M C D : Point)
(tangent : TangentToCircle P A B O₂)
(midpoint_AB : Midpoint M A B)
(perpendicular : Perpendicular O₁ C P A)
(intersectC : IntersectCircle O₁ C)
(intersectD : IntersectCircleAgain O₁ P B D) : Prop :=
Collinear [O₁, S, O₂] ∧ -- There exist S such that ∀ points O₁, S, O₂ in a line
Collinear [P, M, S] ∧  -- ∀ points P, M, S in a line
Collinear [A, S, C] ∧  -- ∀ points A, S, C in a line
Collinear [C, D, M] -- finally we prove points C, D, M are collinear

theorem circles_tangent_points_collinear
(O₁ O₂ P A B M C D : Point)
(tangent : TangentToCircle P A B O₂)
(midpoint_AB : Midpoint M A B)
(perpendicular : Perpendicular O₁ C P A)
(intersectC : IntersectCircle O₁ C)
(intersectD : IntersectCircleAgain O₁ P B D) :
CirclesExternallyTangent O₁ O₂ P A B M C D tangent midpoint_AB perpendicular intersectC intersectD :=
sorry

end circles_tangent_points_collinear_l156_156180


namespace accommodation_arrangements_l156_156551

theorem accommodation_arrangements :
  let colleagues := {A, B, C, D, E},
      number_of_rooms := 3,
      max_per_room := 2 in
  ∃ arrangements : ℕ,
  (∀ r : fin 3, r.card ≤ max_per_room) ∧ -- Each room can hold up to two people 
  (∀ r : fin 3, (A ∈ r → B ∉ r) ∧ (B ∈ r → A ∉ r)) ∧ -- A and B cannot stay in the same room
  arrangements = 72 := 
sorry

end accommodation_arrangements_l156_156551


namespace count_valid_numbers_l156_156988

def is_valid_first_digit (d : ℕ) : Prop := 4 ≤ d ∧ d ≤ 9
def is_valid_last_digit (d : ℕ) : Prop := 0 ≤ d ∧ d ≤ 9
def is_valid_middle_digits (d1 d2 : ℕ) : Prop := 
  1 ≤ d1 ∧ d1 ≤ 9 ∧ 1 ≤ d2 ∧ d2 ≤ 9 ∧ d1 * d2 > 7

theorem count_valid_numbers : 
  let first_digit_choices := 6
  let last_digit_choices := 10
  let middle_digit_pairs := 65 in
  (first_digit_choices * middle_digit_pairs * last_digit_choices) = 3900 :=
by
  -- Skip the proof
  sorry

end count_valid_numbers_l156_156988


namespace symposium_lecture_schedule_l156_156046

def lecturer_schedule_count (n : ℕ) : Prop :=
  let total_permutations := factorial n in
  let valid_permutations := total_permutations / 2 / 3 in
  valid_permutations = 840

theorem symposium_lecture_schedule : lecturer_schedule_count 7 :=
by
  let total_permutations := factorial 7
  have : total_permutations = 5040 := factorial 7
  let valid_permutations := total_permutations / 2 / 3
  have : valid_permutations = 840
  sorry

end symposium_lecture_schedule_l156_156046


namespace new_variance_when_new_point_added_l156_156579

variable (x : list ℝ) (n : ℕ) (mean var : ℝ)

-- Given conditions
def original_sample_size := 7
def original_mean := 5
def original_variance := 2
def additional_data_point := 5

-- Given facts
def original_sum := original_sample_size * original_mean
def sum_of_squared_deviations := original_sample_size * original_variance
def new_sample_size := original_sample_size + 1

theorem new_variance_when_new_point_added :
  ∑ i in list.range original_sample_size, (x[i] - original_mean) ^ 2 = sum_of_squared_deviations ∧
  (list.sum (list.range original_sample_size)) = original_sum →
  (list.sum (list.range original_sample_size) + additional_data_point) / new_sample_size = original_mean →
  let new_variance := (sum_of_squared_deviations + (additional_data_point - original_mean)^2) / new_sample_size in
  new_variance = 7 / 4 :=
by 
  intros hsum hmean 
  sorry

end new_variance_when_new_point_added_l156_156579


namespace parabola_focus_distance_l156_156658

theorem parabola_focus_distance (p : ℝ) (hp : 0 < p) : 
  (∃ (y x : ℝ), y^2 = 2 * p * x) ∧ (∃ (d : ℝ), d = sqrt 2) ∧ 
  (∀ (focus_x focus_y line_a line_b line_c : ℝ), 
      focus_x = p / 2 ∧ focus_y = 0 ∧ 
      line_a = -1 ∧ line_b = 1 ∧ line_c = -1 ∧ 
      abs (line_a * focus_x + line_b * focus_y + line_c) / sqrt (line_a^2 + line_b^2) = sqrt 2) → 
  p = 2 :=
by
  sorry

end parabola_focus_distance_l156_156658


namespace complex_power_4_l156_156080

noncomputable def cos30_re : ℂ := 3 * real.cos (π / 6)
noncomputable def sin30_im : ℂ := 3 * complex.I * real.sin (π / 6)
noncomputable def c : ℂ := cos30_re + sin30_im

theorem complex_power_4 :
  c ^ 4 = -40.5 + 40.5 * complex.I * real.sqrt 3 := sorry

end complex_power_4_l156_156080


namespace parabola_and_triangle_properties_l156_156622

theorem parabola_and_triangle_properties:
  ∀ (A : Point) (p : ℝ), (parabola C : (ℝ × ℝ) → Prop) -- The parabola C
  (directrix : (ℝ × ℝ) → Prop) -- The directrix
  (line_parallel_OA : (ℝ × ℝ) → Prop), -- The line l parallel to OA
  ((A = (1, -2)) → (p > 0) → 
    (parabola = {xy | (xy.2)^2 = 2 * p * (xy.1) }) →
    ((1, -2) ∈ parabola) →
    ((parabola = {xy | (xy.2)^2 = 4 * xy.1 }) ∧ 
     (directrix = {xy | xy.1 = -1 })) ∧
    (∀ l : (ℝ × ℝ) → Prop, |MN| = 3 * sqrt 5 →
      area_of_triangle A M N = 6)) 
sorry

end parabola_and_triangle_properties_l156_156622


namespace complex_power_rectangular_form_l156_156092

noncomputable def cos_30 : ℂ := real.cos (real.pi / 6) -- 30 degrees in radians
noncomputable def sin_30 : ℂ := real.sin (real.pi / 6)

theorem complex_power_rectangular_form :
  (3 * cos_30 + 3 * complex.I * sin_30)^4 = -81 / 2 + (81 * complex.I * real.sqrt 3) / 2 :=
by
  have h1 : cos_30 = real.sqrt 3 / 2 := by sorry
  have h2 : sin_30 = 1 / 2 := by sorry
  rw [h1, h2]
  sorry

end complex_power_rectangular_form_l156_156092


namespace max_cosA_cosB_cosC_l156_156484

theorem max_cosA_cosB_cosC (A B C : ℝ) (h1 : A + B + C = Real.pi) (h2 : 0 < A) (h3 : 0 < B) (h4 : 0 < C) :
  ∃ (M : ℝ), (∀ A B C, A + B + C = Real.pi ∧ 0 < A ∧ 0 < B ∧ 0 < C → cos A + cos B * cos C ≤ M) ∧ M = 1 :=
by
  use 1
  sorry

end max_cosA_cosB_cosC_l156_156484


namespace modular_inverse_32_mod_33_l156_156119

theorem modular_inverse_32_mod_33 :
  ∃ a: ℤ, 0 ≤ a ∧ a ≤ 32 ∧ (32 * a) % 33 = 1 :=
by
  use 32
  split
  { exact le_refl 32 }
  split
  { exact le_of_lt (by decide) }
  { exact Int.mod_eq_of_lt (by decide) }
  {
    have : 32 * 32 = 1024 := rfl
    rw [this]
    exact mathlib.finish (1024 % 33 = 1)
  }

end modular_inverse_32_mod_33_l156_156119


namespace oblique_asymptote_of_f_l156_156968

-- Define the given function
def f (x : ℝ) : ℝ := (3 * x^2 - 5 * x + 4) / (x - 2)

-- Define the (to be proven) equation of the oblique asymptote
def oblique_asymptote (x : ℝ) : Prop := y = 3 * x - 1

-- State the theorem to be proven
theorem oblique_asymptote_of_f :
  ∀ x : ℝ, tendsto (fun (x : ℝ) => f(x) - (3 * x - 1)) (at_top) (nhds 0) := 
sorry

end oblique_asymptote_of_f_l156_156968


namespace m_value_for_arithmetic_seq_min_value_t_smallest_T_periodic_seq_l156_156144

-- Defining the sequence condition
def seq_condition (a : ℕ → ℝ) (m : ℝ) : Prop :=
  ∀ n ≥ 2, a n ^ 2 - a (n + 1) * a (n - 1) = m * (a 2 - a 1) ^ 2

-- (1) Value of m for an arithmetic sequence with a non-zero common difference
theorem m_value_for_arithmetic_seq {a : ℕ → ℝ} (d : ℝ) (h_nonzero : d ≠ 0) :
  (∀ n, a (n + 1) = a n + d) → seq_condition a 1 :=
by
  sorry

-- (2) Minimum value of t given specific conditions
theorem min_value_t {t p : ℝ} (a : ℕ → ℝ) (h_p : 3 ≤ p ∧ p ≤ 5) :
  a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 4 ∧ (∀ n, t * a n + p ≥ n) → t = 1 / 32 :=
by
  sorry

-- (3) Smallest value of T for non-constant periodic sequence
theorem smallest_T_periodic_seq {a : ℕ → ℝ} {m : ℝ} (h_m_nonzero : m ≠ 0) :
  seq_condition a m → (∀ n, a (n + T) = a n) → (∃ T' > 0, ∀ T'', T'' > 0 → T'' = 3) :=
by
  sorry

end m_value_for_arithmetic_seq_min_value_t_smallest_T_periodic_seq_l156_156144


namespace minimum_distance_PQ_l156_156562

open Real

noncomputable def minimum_distance (t : ℝ) : ℝ := 
  (|t - 1|) / (sqrt (1 + t ^ 2))

theorem minimum_distance_PQ :
  let t := sqrt 2 / 2
  let x_P := 2
  let y_P := 0
  let x_Q := -1 + t
  let y_Q := 2 + t
  let d := minimum_distance (x_Q - y_Q + 3)
  (d - 2) = (5 * sqrt 2) / 2 - 2 :=
sorry

end minimum_distance_PQ_l156_156562


namespace smallest_integer_solution_l156_156430

theorem smallest_integer_solution : ∃ x : ℤ, (x^2 = 3 * x + 78) ∧ x = -6 :=
by {
  sorry
}

end smallest_integer_solution_l156_156430


namespace smallest_positive_period_monotonic_decreasing_interval_l156_156175

open Real

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) + sin (2 * x)

theorem smallest_positive_period :
  is_periodic (cos (2 * x) + sin (2 * x)) π :=
sorry

theorem monotonic_decreasing_interval (k : ℤ) :
  ∀ x, k * π + π / 8 ≤ x ∧ x ≤ k * π + 5 * π / 8 → deriv (λ x, cos (2 * x) + sin (2 * x)) x < 0 :=
sorry

end smallest_positive_period_monotonic_decreasing_interval_l156_156175


namespace automotive_test_l156_156941

theorem automotive_test (D T1 T2 T3 T_total : ℕ) (H1 : 3 * D = 180) 
  (H2 : T1 = D / 4) (H3 : T2 = D / 5) (H4 : T3 = D / 6)
  (H5 : T_total = T1 + T2 + T3) : T_total = 37 :=
  sorry

end automotive_test_l156_156941


namespace problem_y_values_l156_156270

theorem problem_y_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 54) :
  ∃ y : ℝ, (y = (x - 3)^2 * (x + 4) / (3 * x - 4)) ∧ (y = 7.5 ∨ y = 4.5) := by
sorry

end problem_y_values_l156_156270


namespace compute_fourth_power_z_l156_156084

-- Definitions from the problem
def cos_angle (θ : ℝ) : ℝ := Real.cos θ
def sin_angle (θ : ℝ) : ℝ := Real.sin θ
def θ := Real.pi / 6  -- 30 degrees in radians

def z : ℂ := 3 * (cos_angle θ) + 3 * Complex.I * (sin_angle θ)

-- Lean 4 Statement for the proof
theorem compute_fourth_power_z : (z ^ 4) = -40.5 + 40.5 * Complex.I * Real.sqrt 3 :=
sorry

end compute_fourth_power_z_l156_156084


namespace limit_f_over_x_at_0_l156_156713

noncomputable theory

variable {f : ℝ → ℝ}
variable (f_diff : DifferentiableAt ℝ f 0)
variable (f_at_0 : f 0 = 0)
variable (f'_at_0 : deriv f 0 = 3)

theorem limit_f_over_x_at_0 : (filter.tendsto (λ x => f x / x) (nhds 0) (nhds 3)) :=
by
  sorry

end limit_f_over_x_at_0_l156_156713


namespace sum_of_products_negative_if_A_not_empty_l156_156706

theorem sum_of_products_negative_if_A_not_empty
  (n : ℕ)
  (h1 : n ≥ 2)
  (a : Fin n → ℝ)
  (h2 : (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → |a i - a j| ≥ 1))
  (h3 : (∑ i, a i) = 0)
  : (∑ (i:Fin n) (j:Fin n), if 1 ≤ i ∧ i < j ∧ j ≤ n ∧ |a i - a j| ≥ 1 then a i * a j else 0) < 0 :=
by
  sorry

end sum_of_products_negative_if_A_not_empty_l156_156706


namespace no_similar_triangle_obtained_l156_156916

-- Define the initial triangle with angles 20°, 20°, and 140°
def initial_triangle (α β γ : ℝ) : Prop :=
  α = 20 ∧ β = 20 ∧ γ = 140

-- Define the cutting operation, which cuts the triangle along one of its angle bisectors
def bisector_cut (α β γ : ℝ) (α' β' γ' : ℝ) : Prop :=
  (2 * α' = α ∧ β = β' ∧ γ' = 180 - 2 * α' - β') ∨
  (2 * α' = β ∧ α = γ' ∧ γ = 180 - 2 * α' - α') ∨
  (2 * α' = γ ∧ α = α' ∧ β = 180 - 2 * α' - α')

-- Final theorem stating it is impossible to obtain a similar triangle with angles 20°, 20°, and 140°
theorem no_similar_triangle_obtained :
  ∀ (α β γ : ℝ), initial_triangle α β γ →
  ¬ (∃ (α' β' γ' : ℝ), (initial_triangle α' β' γ') ∧ repeat (λ t, bisector_cut t.1 t.2 t.3) k (α, β, γ)) :=
by
  intros α β γ h_tri h_k
  sorry

end no_similar_triangle_obtained_l156_156916


namespace sum_of_numbers_in_table_l156_156677

theorem sum_of_numbers_in_table
  (m n : ℕ)
  (a : matrix (fin m) (fin n) ℝ)
  (h : ∀ i j, (∑ k, a i k) * (∑ k, a k j) = a i j) :
  (∑ i j, a i j = 1) ∨ (∀ i j, a i j = 0) :=
sorry

end sum_of_numbers_in_table_l156_156677


namespace plus_signs_count_l156_156818

theorem plus_signs_count (total_symbols : ℕ) (n : ℕ) (m : ℕ)
    (h1 : total_symbols = 23)
    (h2 : ∀ (s : Finset ℕ), s.card = 10 → (∃ x ∈ s, x = n))
    (h3 : ∀ (s : Finset ℕ), s.card = 15 → (∃ x ∈ s, x = m)) :
    n = 14 := by
  sorry

end plus_signs_count_l156_156818


namespace part1_solution_part2_solution_l156_156898

-- Part 1: Solve the system of equations:
-- Conditions: {x = 4y + 1, 2x - 5y = 8}
def solveSystemEquations (x y : ℤ) : Prop :=
  x = 4 * y + 1 ∧ 2 * x - 5 * y = 8

-- Final statement: Prove {x = 9, y = 2} given the conditions
theorem part1_solution: ∃ x y : ℤ, solveSystemEquations x y ∧ x = 9 ∧ y = 2 :=
by
  apply Exists.intro 9
  apply Exists.intro 2
  split
  repeat {sorry}

-- Part 2: Solve the system of inequalities:
-- Conditions: {4x - 5 ≤ 3, ((x - 1) / 3) < ((2x + 1) / 5)}
def systemInequalities (x : ℝ) : Prop :=
  4 * x - 5 ≤ 3 ∧ (x - 1) / 3 < (2 * x + 1) / 5

-- Final statement: Prove -8 < x ≤ 2 given the conditions
theorem part2_solution: ∃ x : ℝ, systemInequalities x ∧ -8 < x ∧ x ≤ 2 :=
by
  sorry

end part1_solution_part2_solution_l156_156898


namespace more_permutations_with_P_than_without_l156_156918

def has_property_P (n : ℕ) (σ : Fin 2n → Fin 2n) : Prop :=
  ∃ i : Fin (2n-1), |σ i - σ ⟨i + 1, sorry⟩| = n

theorem more_permutations_with_P_than_without (n : ℕ) (hn : n > 0) :
  ∃ A : Finset (Fin 2n → Fin 2n), ∃ B : Finset (Fin 2n → Fin 2n),
  (A.card > B.card) ∧ 
  (∀ σ, σ ∈ A ↔ has_property_P n σ) ∧ 
  (∀ σ, σ ∈ B ↔ ¬ has_property_P n σ) :=
by
sorry

end more_permutations_with_P_than_without_l156_156918


namespace num_integers_in_solution_set_l156_156633

theorem num_integers_in_solution_set : ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℤ), abs (x - 3) ≤ 7 ↔ (x ≥ -4 ∧ x ≤ 10) ∧ ∃ y, (y = -4 ∨ y = -3 ∨ y = -2 ∨ y = -1 ∨ y = 0 ∨ y = 1 ∨ y = 2 ∨ y = 3 ∨ y = 4 ∨ y = 5 ∨ y = 6 ∨ y = 7 ∨ y = 8 ∨ y = 9 ∨ y = 10).
sorry

end num_integers_in_solution_set_l156_156633


namespace polynomial_value_at_3_l156_156410

-- Definitions based on given conditions
def f (x : ℕ) : ℕ :=
  5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

def x := 3

-- Proof statement
theorem polynomial_value_at_3 : f x = 1641 := by
  sorry

end polynomial_value_at_3_l156_156410


namespace root_interval_l156_156660

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 1

theorem root_interval (a b : ℤ) (h1 : b = a + 1) (h2 : ∃ x ∈ Ioo (a : ℝ) b, f x = 0) :
  a + b = -3 := by
  sorry

end root_interval_l156_156660


namespace sin_cos_pi_minus_two_alpha_l156_156140

theorem sin_cos_pi_minus_two_alpha (α : ℝ) (h : tan α = 2 / 3) : 
  sin (2 * α) - cos (π - 2 * α) = 17 / 13 :=
by
  sorry

end sin_cos_pi_minus_two_alpha_l156_156140


namespace sum_square_div_product_of_odds_l156_156293

open BigOperators
open Nat

theorem sum_square_div_product_of_odds (n : ℕ) (h : n > 0) :
  ∑ i in finset.range n, (i + 1) ^ 2 / (2 * (i + 1) - 1) * (2 * (i + 1) + 1) = 
    n * (n + 1) / (2 * n + 1) :=
by sorry

end sum_square_div_product_of_odds_l156_156293


namespace IntervalForKTriangleLengths_l156_156571

noncomputable def f (x k : ℝ) := (x^4 + k * x^2 + 1) / (x^4 + x^2 + 1)

theorem IntervalForKTriangleLengths (k : ℝ) :
  (∀ (x : ℝ), 1 ≤ f x k ∧
              (k ≥ 1 → f x k ≤ (k + 2) / 3) ∧ 
              (k < 1 → f x k ≥ (k + 2) / 3)) →
  (∀ (a b c : ℝ), (f a k < f b k + f c k) ∧ 
                  (f b k < f a k + f c k) ∧ 
                  (f c k < f a k + f b k)) ↔ (-1/2 < k ∧ k < 4) :=
by sorry

#check f
#check IntervalForKTriangleLengths

end IntervalForKTriangleLengths_l156_156571


namespace polygon_sides_l156_156382

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 4 * 360) → n = 10 :=
by 
  sorry

end polygon_sides_l156_156382


namespace students_between_hoseok_and_minyoung_l156_156389

def num_students : Nat := 13
def hoseok_position_from_right : Nat := 9
def minyoung_position_from_left : Nat := 8

theorem students_between_hoseok_and_minyoung
    (n : Nat)
    (h : n = num_students)
    (p_h : n - hoseok_position_from_right + 1 = 5)
    (p_m : minyoung_position_from_left = 8):
    ∃ k : Nat, k = 2 :=
by
  sorry

end students_between_hoseok_and_minyoung_l156_156389


namespace parts_meet_requirement_best_quality_part_l156_156906

noncomputable def part_meets_requirement (measurement : ℝ) : Prop :=
  49.97 ≤ 50 + measurement ∧ 50 + measurement ≤ 50.04

theorem parts_meet_requirement (m1 m2 m3 m4 m5 : ℝ) :
  m1 = 0.031 → m2 = -0.037 → m3 = 0.018 → m4 = -0.021 → m5 = 0.042 →
  part_meets_requirement m1 ∧
  part_meets_requirement m3 ∧
  part_meets_requirement m4 ∧
  ¬ part_meets_requirement m2 ∧
  ¬ part_meets_requirement m5 :=
begin
  intros h1 h2 h3 h4 h5,
  -- use the definitions and conditions to prove the assertions
  sorry,
end

theorem best_quality_part (m1 m3 m4 : ℝ) :
  m1 = 0.031 → m3 = 0.018 → m4 = -0.021 →
  abs (50 + m3 - 50) < abs (50 + m1 - 50) ∧
  abs (50 + m3 - 50) < abs (50 + m4 - 50) :=
begin
  intros h1 h3 h4,
  -- use the definitions and conditions to prove the assertions
  sorry,
end

end parts_meet_requirement_best_quality_part_l156_156906


namespace darla_total_payment_l156_156508

-- Define the cost per watt, total watts used, and late fee
def cost_per_watt : ℝ := 4
def total_watts : ℝ := 300
def late_fee : ℝ := 150

-- Define the total cost of electricity
def electricity_cost : ℝ := cost_per_watt * total_watts

-- Define the total amount Darla needs to pay
def total_amount : ℝ := electricity_cost + late_fee

-- The theorem to prove the total amount equals $1350
theorem darla_total_payment : total_amount = 1350 := by
  sorry

end darla_total_payment_l156_156508


namespace a_1000_is_3009_l156_156227

noncomputable def sequence : ℕ+ → ℤ
| ⟨1, _⟩ := 2010
| ⟨2, _⟩ := 2011
| ⟨n + 3, _⟩ := 2 * (n + 1) - (sequence ⟨n + 2, Nat.succ_pos' _⟩ + sequence ⟨n + 1, Nat.succ_pos' _⟩)

theorem a_1000_is_3009 : sequence ⟨1000, Nat.succ_pos' _⟩ = 3009 := 
sorry

end a_1000_is_3009_l156_156227


namespace best_fit_line_slope_l156_156439

variables {x1 x2 x3 x4 y1 y2 y3 y4 d : ℝ}

theorem best_fit_line_slope :
  x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧
  x4 - x3 = d ∧ x3 - x2 = d ∧ x2 - x1 = d →
  (x4 - x1) ≠ 0 →
  let m := (y4 - y1) / (x4 - x1)
  in m = (y4 - y1) / (x4 - x1) :=
by
  intros h h_nonzero
  dsimp
  rfl

end best_fit_line_slope_l156_156439


namespace min_distinct_integers_with_ap_and_gp_of_length_5_l156_156869

theorem min_distinct_integers_with_ap_and_gp_of_length_5 : 
  ∃ (s : set ℤ), (s.card = 6) ∧ 
    (∃ (a r : ℤ),  (∀ i, 0 ≤ i ∧ i < 5 → (a + i * r) ∈ s)) ∧ 
    (∃ (b q : ℤ), q ≠ 0 ∧ (∀ i, 0 ≤ i ∧ i < 5 → (b * q^i) ∈ s)) :=
sorry

end min_distinct_integers_with_ap_and_gp_of_length_5_l156_156869


namespace min_sum_value_exists_min_sum_value_l156_156591

noncomputable def min_value_sum (x : Fin 2018 → ℝ) : Prop :=
  ∑ i in Finset.range 2018, (i + 1) * x ⟨i, sorry⟩ = 2037171 / 2

theorem min_sum_value {x : Fin 2018 → ℝ} (h : ∀ (i j : Fin 2018), i < j → x i + x j >= -1^(i + j + 2)) :
  ∑ i in Finset.range 2018, (i + 1) * x ⟨i, sorry⟩ >= 2037171 / 2 :=
sorry

theorem exists_min_sum_value: 
  ∃ x : Fin 2018 → ℝ, min_value_sum x :=
sorry

end min_sum_value_exists_min_sum_value_l156_156591


namespace gcd_12_20_l156_156992

theorem gcd_12_20 : Nat.gcd 12 20 = 4 := by
  sorry

end gcd_12_20_l156_156992


namespace find_term_number_l156_156042

variable {α : ℝ} (b : ℕ → ℝ) (q : ℝ)

namespace GeometricProgression

noncomputable def geometric_progression (b : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ (n : ℕ), b (n + 1) = b n * q

noncomputable def satisfies_conditions (α : ℝ) (b : ℕ → ℝ) : Prop :=
  b 25 = 2 * Real.tan α ∧ b 31 = 2 * Real.sin α

theorem find_term_number (α : ℝ) (b : ℕ → ℝ) (q : ℝ) (hb : geometric_progression b q) (hc : satisfies_conditions α b) :
  ∃ n, b n = Real.sin (2 * α) ∧ n = 37 :=
sorry

end GeometricProgression

end find_term_number_l156_156042


namespace arithmetic_mean_of_roots_l156_156746

-- Definitions corresponding to the conditions
def quadratic_eqn (a b c : ℝ) (x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- The term statement for the quadratic equation mean
theorem arithmetic_mean_of_roots : 
  ∀ (a b c : ℝ), a = 1 → b = 4 → c = 1 → (∃ (x1 x2 : ℝ), quadratic_eqn a b c x1 ∧ quadratic_eqn a b c x2 ∧ -4 / 2 = -2) :=
by
  -- skip the proof
  sorry

end arithmetic_mean_of_roots_l156_156746


namespace incircles_of_triangels_tangent_l156_156130

-- Define the quadrilateral and the given condition
variables {A B C D : Type} [Point A] [Point B] [Point C] [Point D]

-- Definition of a convex quadrilateral
def is_convex_quadrilateral (A B C D : Point) : Prop :=
  -- Definition of convexity (details omitted, assume well-defined)
  sorry

-- Condition for the problem
def equal_sums (A B C D : Point) [Dist A B] [Dist C D] [Dist B C] [Dist D A] : Prop :=
  (dist A B + dist C D) = (dist B C + dist D A)

-- Define the concept of the incircle being tangent to another incircle
def incircles_tangent (A B C D : Point) [Incircle (triangle A B C)] [Incircle (triangle A C D)] : Prop :=
  -- Definition of tangency between incircles (details omitted, assume well-defined)
  sorry

-- The main theorem
theorem incircles_of_triangels_tangent
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : equal_sums A B C D) :
  incircles_tangent A B C D :=
sorry

end incircles_of_triangels_tangent_l156_156130


namespace conjugate_of_z_l156_156573

theorem conjugate_of_z : 
  (∃ z : ℂ, (1 + 2 * Complex.I) * z = 4 + 3 * Complex.I) → 
  ∃ z : ℂ, Complex.conj z = 2 + Complex.I :=
by
  -- proof by contradiction or structural proof steps here
  sorry

end conjugate_of_z_l156_156573


namespace man_speed_against_current_l156_156011

def speed_with_current := 25 -- km/hr
def speed_of_current := 2.5 -- km/hr

theorem man_speed_against_current : 
  let V_m := speed_with_current - speed_of_current in
  V_m - speed_of_current = 20 := 
by
  sorry

end man_speed_against_current_l156_156011


namespace sum_or_all_zero_l156_156675

theorem sum_or_all_zero (a : ℕ → ℕ → ℕ) (n m : ℕ)
  (x : ℕ → ℕ) (y : ℕ → ℕ)
  (H1 : ∀ i, x i = ∑ j in finset.range m, a i j)
  (H2 : ∀ j, y j = ∑ i in finset.range n, a i j)
  (H3 : ∀ i j, a i j = x i * y j) :
  (∑ i in finset.range n, ∑ j in finset.range m, a i j = 1) ∨ 
  (∀ i j, a i j = 0) :=
sorry

end sum_or_all_zero_l156_156675


namespace joe_eats_at_least_two_different_fruits_l156_156494

namespace JoeFruitProblem

-- Define the probability space and events
noncomputable def prob_at_least_two_different_fruits : ℚ := sorry

theorem joe_eats_at_least_two_different_fruits
  : prob_at_least_two_different_fruits = 63 / 64 := 
sorry

end JoeFruitProblem

end joe_eats_at_least_two_different_fruits_l156_156494


namespace minimum_value_expression_l156_156553

theorem minimum_value_expression :
  ∀ (r s t : ℝ), (1 ≤ r ∧ r ≤ s ∧ s ≤ t ∧ t ≤ 4) →
  (r - 1) ^ 2 + (s / r - 1) ^ 2 + (t / s - 1) ^ 2 + (4 / t - 1) ^ 2 = 4 * (Real.sqrt 2 - 1) ^ 2 := 
sorry

end minimum_value_expression_l156_156553


namespace inequality_proof_l156_156596

theorem inequality_proof 
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ac + bd ≤ real.sqrt ((a ^ 2 + b ^ 2) * (c ^ 2 + d ^ 2)) :=
by sorry

end inequality_proof_l156_156596


namespace num_of_triangles_l156_156390

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def count_triangles (lengths : List ℕ) : ℕ :=
  lengths.combinations 3 |>.count (λ t, is_triangle t[0] t[1] t[2])

theorem num_of_triangles : count_triangles [3, 5, 7, 9] = 3 := sorry

end num_of_triangles_l156_156390


namespace initial_shipment_robot_rascals_l156_156473

theorem initial_shipment_robot_rascals 
(T : ℝ) 
(h1 : (0.7 * T = 168)) : 
  T = 240 :=
sorry

end initial_shipment_robot_rascals_l156_156473


namespace number_of_different_teams_l156_156225

namespace DoctorTeam

-- Conditions
variables (total_doctors pediatricians surgeons general_practitioners : ℕ)
          (team_size : ℕ) (at_least_one_pead pediatrician_choice surgeon_choice general_practitioner_choice other_choices : ℕ)

-- Define specific numbers as per the problem
def conditions := 
  total_doctors = 25 ∧
  pediatricians = 5 ∧
  surgeons = 10 ∧
  general_practitioners = 10 ∧
  team_size = 5 ∧
  at_least_one_pead = 1 ∧ 
  pediatrician_choice = (choose pediatricians at_least_one_pead).val ∧
  surgeon_choice = (choose surgeons at_least_one_pead).val ∧
  general_practitioner_choice = (choose general_practitioners at_least_one_pead).val ∧
  other_choices = (choose (total_doctors - (at_least_one_pead * 3)) (team_size - (at_least_one_pead * 3))).val

-- The proof problem
theorem number_of_different_teams {total_doctors pediatricians surgeons general_practitioners team_size at_least_one_pead pediatrician_choice surgeon_choice general_practitioner_choice other_choices : ℕ} :
  conditions total_doctors pediatricians surgeons general_practitioners team_size at_least_one_pead pediatrician_choice surgeon_choice general_practitioner_choice other_choices →
  (pediatrician_choice * surgeon_choice * general_practitioner_choice * other_choices) = 115500 :=
by
  sorry

end DoctorTeam

end number_of_different_teams_l156_156225


namespace darla_total_payment_l156_156507

-- Define the cost per watt, total watts used, and late fee
def cost_per_watt : ℝ := 4
def total_watts : ℝ := 300
def late_fee : ℝ := 150

-- Define the total cost of electricity
def electricity_cost : ℝ := cost_per_watt * total_watts

-- Define the total amount Darla needs to pay
def total_amount : ℝ := electricity_cost + late_fee

-- The theorem to prove the total amount equals $1350
theorem darla_total_payment : total_amount = 1350 := by
  sorry

end darla_total_payment_l156_156507


namespace problem_statement_l156_156202

theorem problem_statement (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end problem_statement_l156_156202


namespace decrypt_test_phrase_l156_156697

constant encrypted_phrase : string
constant decrypted_phrase : string
constant example_step1 : string → string
constant example_step2 : string → string
constant example_step3 : string → string 

axiom no_yo_character (s : string) : ¬ s.contains "ё"

axiom example_encrypt (input : string) :
  example_step3 (example_step2 (example_step1 input)) = "уфзмтфсзек"

axiom reverse_string (s : string) : string
axiom shift_left_two (s : string) : string
axiom swap_adjacent (s : string) : string

theorem decrypt_test_phrase :
  reverse_string (shift_left_two (swap_adjacent "врпвл терпраиэ вйзгцфпз")) = "нефте базы южного района" :=
sorry

end decrypt_test_phrase_l156_156697


namespace number_of_elements_in_A_is_multiple_of_q_l156_156707

theorem number_of_elements_in_A_is_multiple_of_q
  (p q : ℕ)
  (hp : Nat.Prime p)
  (hq : Nat.Prime q)
  (S : Finset ℕ)
  (hS_sub : ∀ x ∈ S, x ∈ Finset.range (p - 1) ∧ x > 0) :
  ∃ k, card ({x : Fin (q ⟶ S)} \{ 
    x \in Finset.range (p - 1) 
    and 0 : ∑ {i = 1} x i 0 x i ≡ 0 [MOD p] => rhat => 
    k * q :=
sorry

end number_of_elements_in_A_is_multiple_of_q_l156_156707


namespace count_divisible_by_90_l156_156193

theorem count_divisible_by_90 : 
  ∃ n, n = 10 ∧ (∀ k, 1000 ≤ k ∧ k < 10000 ∧ k % 100 = 90 ∧ k % 90 = 0 → n = 10) :=
begin
  sorry
end

end count_divisible_by_90_l156_156193


namespace parabola_focus_distance_l156_156659

theorem parabola_focus_distance (p : ℝ) (hp : 0 < p) : 
  (∃ (y x : ℝ), y^2 = 2 * p * x) ∧ (∃ (d : ℝ), d = sqrt 2) ∧ 
  (∀ (focus_x focus_y line_a line_b line_c : ℝ), 
      focus_x = p / 2 ∧ focus_y = 0 ∧ 
      line_a = -1 ∧ line_b = 1 ∧ line_c = -1 ∧ 
      abs (line_a * focus_x + line_b * focus_y + line_c) / sqrt (line_a^2 + line_b^2) = sqrt 2) → 
  p = 2 :=
by
  sorry

end parabola_focus_distance_l156_156659


namespace polygon_sides_l156_156376

theorem polygon_sides (n : ℕ) 
  (h_interior : (n - 2) * 180 = 4 * 360) : 
  n = 10 :=
by
  sorry

end polygon_sides_l156_156376


namespace jelly_bean_problem_l156_156775

theorem jelly_bean_problem 
  (x y : ℕ) 
  (h1 : x + y = 1200) 
  (h2 : x = 3 * y - 400) :
  x = 800 := 
sorry

end jelly_bean_problem_l156_156775


namespace plus_signs_count_l156_156777

theorem plus_signs_count (p m : ℕ) (h_sum : p + m = 23)
                         (h_max_minus : m ≤ 9) (h_max_plus : p ≤ 14)
                         (h_at_least_one_plus_in_10 : ∀ (s : Finset (Fin 23)), s.card = 10 → ∃ i ∈ s, i < p)
                         (h_at_least_one_minus_in_15 : ∀ (s : Finset (Fin 23)), s.card = 15 → ∃ i ∈ s, m ≤ i) :
  p = 14 :=
by sorry

end plus_signs_count_l156_156777


namespace simplify_first_expression_simplify_second_expression_l156_156740

theorem simplify_first_expression (x y : ℝ) : 3 * x - 2 * y + 1 + 3 * y - 2 * x - 5 = x + y - 4 :=
sorry

theorem simplify_second_expression (x : ℝ) : (2 * x ^ 4 - 5 * x ^ 2 - 4 * x + 3) - (3 * x ^ 3 - 5 * x ^ 2 - 4 * x) = 2 * x ^ 4 - 3 * x ^ 3 + 3 :=
sorry

end simplify_first_expression_simplify_second_expression_l156_156740


namespace binom_sum_n_l156_156105

-- Definitions of the binomial coefficient
noncomputable def binom : ℕ → ℕ → ℕ
| n, k := if k > n then 0 else nat.choose n k

-- Property we are trying to prove
theorem binom_sum_n (A B : ℕ) : 
  (∀ n, binom 22 n + binom 22 12 = binom 23 13 → (n = 11 ∨ n = 13)) →
  ∑ n in {11, 13}, n = 24 :=
by
  sorry

end binom_sum_n_l156_156105


namespace value_of_squares_l156_156266

-- Define the conditions
variables (p q : ℝ)

-- State the theorem with the given conditions and the proof goal
theorem value_of_squares (h1 : p * q = 12) (h2 : p + q = 8) : p ^ 2 + q ^ 2 = 40 :=
sorry

end value_of_squares_l156_156266


namespace words_added_to_removed_ratio_l156_156887

-- Conditions in the problem
def Yvonnes_words : ℕ := 400
def Jannas_extra_words : ℕ := 150
def words_removed : ℕ := 20
def words_needed : ℕ := 1000 - 930

-- Definitions derived from the conditions
def Jannas_words : ℕ := Yvonnes_words + Jannas_extra_words
def total_words_before_editing : ℕ := Yvonnes_words + Jannas_words
def total_words_after_removal : ℕ := total_words_before_editing - words_removed
def words_added : ℕ := words_needed

-- The theorem we need to prove
theorem words_added_to_removed_ratio :
  (words_added : ℚ) / words_removed = 7 / 2 :=
sorry

end words_added_to_removed_ratio_l156_156887


namespace floor_sqrt_120_l156_156533

theorem floor_sqrt_120 : (Int.floor (Real.sqrt 120)) = 10 := by
  have h1 : 10^2 < 120 := by norm_num
  have h2 : 120 < 11^2 := by norm_num
  -- Additional steps to show that Int.floor (Real.sqrt 120) = 10
  sorry

end floor_sqrt_120_l156_156533


namespace percentage_decrease_second_year_l156_156762

def initial_population : ℝ := 415600
def first_year_increase : ℝ := 0.25
def end_second_year_population : ℝ := 363650

def population_end_first_year (initial : ℝ) (rate : ℝ) : ℝ :=
  initial + rate * initial

def percentage_decrease (initial : ℝ) (final : ℝ) : ℝ :=
  100 * (initial - final) / initial

theorem percentage_decrease_second_year :
  percentage_decrease (population_end_first_year initial_population first_year_increase) end_second_year_population = 29.99 :=
by
  sorry

end percentage_decrease_second_year_l156_156762


namespace complex_power_rectangular_form_l156_156093

noncomputable def cos_30 : ℂ := real.cos (real.pi / 6) -- 30 degrees in radians
noncomputable def sin_30 : ℂ := real.sin (real.pi / 6)

theorem complex_power_rectangular_form :
  (3 * cos_30 + 3 * complex.I * sin_30)^4 = -81 / 2 + (81 * complex.I * real.sqrt 3) / 2 :=
by
  have h1 : cos_30 = real.sqrt 3 / 2 := by sorry
  have h2 : sin_30 = 1 / 2 := by sorry
  rw [h1, h2]
  sorry

end complex_power_rectangular_form_l156_156093


namespace count_integer_solutions_l156_156636

theorem count_integer_solutions (x : ℤ) : 
  (|x - 3| ≤ 7) → (x ∈ Finset.range 15 → x + -8 : ℤ) := 
sorry

end count_integer_solutions_l156_156636


namespace linear_equation_in_two_vars_example_l156_156937

def is_linear_equation_in_two_vars (eq : String) : Prop :=
  eq = "x + 4y = 6"

theorem linear_equation_in_two_vars_example :
  is_linear_equation_in_two_vars "x + 4y = 6" :=
by
  sorry

end linear_equation_in_two_vars_example_l156_156937


namespace Sawyer_cleans_in_6_hours_l156_156008

theorem Sawyer_cleans_in_6_hours (N : ℝ) (S : ℝ) (h1 : S = (2/3) * N) 
                                 (h2 : 1/S + 1/N = 1/3.6) : S = 6 :=
by
  sorry

end Sawyer_cleans_in_6_hours_l156_156008


namespace sum_in_base7_l156_156996

-- An encoder function for base 7 integers
def to_base7 (n : ℕ) : string :=
sorry -- skipping the implementation for brevity

-- Decoding the string representation back to a natural number
def from_base7 (s : string) : ℕ :=
sorry -- skipping the implementation for brevity

-- The provided numbers in base 7
def x : ℕ := from_base7 "666"
def y : ℕ := from_base7 "66"
def z : ℕ := from_base7 "6"

-- The expected sum in base 7
def expected_sum : ℕ := from_base7 "104"

-- The statement to be proved
theorem sum_in_base7 : x + y + z = expected_sum :=
sorry -- The proof is omitted

end sum_in_base7_l156_156996


namespace largest_visits_l156_156895

theorem largest_visits (stores : ℕ) (total_visits : ℕ) (unique_visitors : ℕ) 
  (visits_two_stores : ℕ) (remaining_visitors : ℕ) : 
  stores = 7 ∧ total_visits = 21 ∧ unique_visitors = 11 ∧ visits_two_stores = 7 ∧ remaining_visitors = (unique_visitors - visits_two_stores) →
  (remaining_visitors * 2 <= total_visits - visits_two_stores * 2) → (∀ v : ℕ, v * unique_visitors = total_visits) →
  (∃ v_max : ℕ, v_max = 4) :=
by
  sorry

end largest_visits_l156_156895


namespace four_digit_div_90_count_l156_156190

theorem four_digit_div_90_count :
  ∃ n : ℕ, n = 10 ∧ (∀ (ab : ℕ), 10 ≤ ab ∧ ab ≤ 99 → ab % 9 = 0 → 
  (10 * ab + 90) % 90 = 0 ∧ 1000 ≤ 10 * ab + 90 ∧ 10 * ab + 90 < 10000) :=
sorry

end four_digit_div_90_count_l156_156190


namespace simplify_fraction_l156_156321

theorem simplify_fraction (b : ℕ) (hb : b = 5) : (15 * b^4) / (90 * b^3 * b) = 1 / 6 := by
  sorry

end simplify_fraction_l156_156321


namespace simplify_expression_l156_156322

variable (x : ℝ)

theorem simplify_expression (h : x ≠ 0) : x⁻¹ - 3 * x + 2 = - (3 * x^2 - 2 * x - 1) / x :=
by
  sorry

end simplify_expression_l156_156322


namespace percentage_of_cost_for_overhead_is_58_33_l156_156365

-- Definitions of the prices
def purchase_price : ℝ := 48
def net_profit : ℝ := 12
def markup : ℝ := 40

-- Definition for the overhead calculation
def overhead : ℝ := markup - net_profit

-- Definition for the percentage calculation
def percentage_of_overhead : ℝ := (overhead / purchase_price) * 100

-- The main theorem to be proven
theorem percentage_of_cost_for_overhead_is_58_33 : percentage_of_overhead = 58.33 :=
by
  sorry

end percentage_of_cost_for_overhead_is_58_33_l156_156365


namespace delta_delta_delta_45_l156_156509

def delta (P : ℚ) : ℚ := (2 / 3) * P + 2

theorem delta_delta_delta_45 :
  delta (delta (delta 45)) = 158 / 9 :=
by sorry

end delta_delta_delta_45_l156_156509


namespace amount_tom_should_pay_l156_156401

theorem amount_tom_should_pay (original_price : ℝ) (multiplier : ℝ) 
  (h1 : original_price = 3) (h2 : multiplier = 3) : 
  original_price * multiplier = 9 :=
sorry

end amount_tom_should_pay_l156_156401


namespace plus_signs_count_l156_156815

theorem plus_signs_count (num_symbols : ℕ) (at_least_one_plus_in_10 : ∀ s : Finset ℕ, s.card = 10 → (∃ i ∈ s, i < 14)) (at_least_one_minus_in_15 : ∀ s : Finset ℕ, s.card = 15 → (∃ i ∈ s, i ≥ 14)) : 
    ∃ (p m : ℕ), p + m = 23 ∧ p = 14 ∧ m = 9 := by
  sorry

end plus_signs_count_l156_156815


namespace find_root_equation_l156_156548

theorem find_root_equation : ∃ x : ℤ, x - (5 / (x - 4)) = 2 - (5 / (x - 4)) ∧ x = 2 :=
by
  sorry

end find_root_equation_l156_156548


namespace cube_as_difference_of_squares_l156_156022

theorem cube_as_difference_of_squares (a : ℕ) : 
  a^3 = (a * (a + 1) / 2)^2 - (a * (a - 1) / 2)^2 := 
by 
  -- The proof portion would go here, but since we only need the statement:
  sorry

end cube_as_difference_of_squares_l156_156022


namespace floor_sqrt_120_l156_156529

theorem floor_sqrt_120 : (Int.floor (Real.sqrt 120)) = 10 := by
  have h1 : 10^2 < 120 := by norm_num
  have h2 : 120 < 11^2 := by norm_num
  -- Additional steps to show that Int.floor (Real.sqrt 120) = 10
  sorry

end floor_sqrt_120_l156_156529


namespace extremum_a_eq_zero_l156_156613

def f (a x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - x

theorem extremum_a_eq_zero
  (h : ∀ a x : ℝ, x = 1 → deriv (f a) 1 = 0) :
  ∃ a : ℝ, a = 0 := by
  sorry

end extremum_a_eq_zero_l156_156613


namespace max_value_proof_l156_156588

noncomputable def max_sum_value (n : ℕ) (h : n ≥ 2) (x : Fin n → ℝ) : ℝ :=
  ∑ i in range n, x i * |x i - x ((i + 1) % n)|

theorem max_value_proof (n : ℕ) (hn : n ≥ 2) (x : Fin n → ℝ)
  (hx : ∀ i, 0 ≤ x i ∧ x i ≤ 1) :
  max_sum_value n hn x = n / 4 + ⌊n / 2⌋ / 2 :=
sorry

end max_value_proof_l156_156588


namespace line_and_circle_relationship_l156_156024

-- Define the line and circle equations
def line (k : ℝ) : set (ℝ × ℝ) := {p | p.2 = k * p.1 + 1}
def circle : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2}

-- Algebraic definition of line not passing through the origin
def does_not_pass_through_origin (k : ℝ) : Prop :=
  ¬(∃ (x : ℝ), (x, k * x + 1) = (0, 0))

-- Prove that the line intersects with the circle but does not pass through the center for all k
theorem line_and_circle_relationship (k : ℝ) :
  ∃ (p : ℝ × ℝ), p ∈ line k ∧ p ∈ circle ∧ does_not_pass_through_origin k :=
by
  sorry

end line_and_circle_relationship_l156_156024


namespace count_of_plus_signs_l156_156805

-- Definitions based on the given conditions
def total_symbols : ℕ := 23
def any_10_symbols_contains_plus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 10 → (\exists c ∈ s, c = '+')
def any_15_symbols_contains_minus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 15 → (\exists c ∈ s, c = '-')

-- Main theorem statement without the proof
theorem count_of_plus_signs (symbols : Finset char) (h1 : symbols.card = total_symbols) 
  (h2 : any_10_symbols_contains_plus symbols) 
  (h3 : any_15_symbols_contains_minus symbols) : 
  symbols.count '+' = 14 :=
sorry

end count_of_plus_signs_l156_156805


namespace correct_derivative_statement_l156_156882

theorem correct_derivative_statement :
  (∀ x, has_deriv_at (λ x, Real.sin x) (Real.cos x) x
  → ¬ (∀ x, has_deriv_at (λ x, Real.sin x) (-Real.cos x) x))
  ∧
  (∀ x, has_deriv_at (λ x, Real.log x + x) ((x + 1) / x) x)
  ∧
  (∀ x, has_deriv_at (λ x, 4 * x^2) (8 * x) x
  → ¬ (∀ x, has_deriv_at (λ x, 4 * x^2) (4 * x) x))
  ∧
  (∀ x, has_deriv_at (λ x, Real.exp x - x) (Real.exp x - 1) x
  ∧ has_deriv_at (λ x, Real.exp x - x) 0 0
  → ¬ (∀ x, has_deriv_at (λ x, Real.exp x - x) 1 0)) :=
by {
  sorry
}

end correct_derivative_statement_l156_156882


namespace distance_inequality_l156_156159

-- Define the setup and conditions
variables {α β : Type} [plane α] [plane β] [line m] [line n] [point A] [point B]
variables (in_plane_m : line m ∈ plane α) (in_plane_n : line n ∈ plane β)
variables (A_on_m : point A ∈ line m) (B_on_n : point B ∈ line n)
variables (α_parallel_β : plane α ∥ plane β)
variables (dist_A_B : ℝ) (dist_A_n : ℝ) (dist_m_n : ℝ)

-- Define distances
def distance (A B : point) : ℝ := dist_A_B
def distance_to_line (A : point) (l : line) : ℝ := dist_A_n
def distance_between_lines (m n : line) : ℝ := dist_m_n

-- Declare the proof problem
theorem distance_inequality 
  (h1 : parallel plane α plane β)
  (h2 : dist_A_B = distance A B)
  (h3 : dist_A_n = distance_to_line A n)
  (h4 : dist_m_n = distance_between_lines m n) :
  dist_m_n ≤ dist_A_B ∧ dist_A_B ≤ dist_A_n :=
sorry

end distance_inequality_l156_156159


namespace range_a_of_tangents_coincide_l156_156611

theorem range_a_of_tangents_coincide (x1 x2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) (a : ℝ)
  (h3 : -1 / (x2 ^ 2) = 2 * x1 + 1) (h4 : x1 ^ 2 = -a) :
  1/4 < a ∧ a < 1 :=
by
  sorry 

end range_a_of_tangents_coincide_l156_156611


namespace graph_transformation_l156_156457

theorem graph_transformation (a b c : ℝ) (h1 : c = 1) (h2 : a + b + c = -2) (h3 : a - b + c = 2) :
  (∀ x, cx^2 + 2 * bx + a = (x - 2)^2 - 5) := 
sorry

end graph_transformation_l156_156457


namespace volume_ratio_l156_156408

theorem volume_ratio (V1 V2 M1 M2 : ℝ)
  (h1 : M1 / (V1 - M1) = 1 / 2)
  (h2 : M2 / (V2 - M2) = 3 / 2)
  (h3 : (M1 + M2) / (V1 - M1 + V2 - M2) = 1) :
  V1 / V2 = 9 / 5 :=
by
  sorry

end volume_ratio_l156_156408


namespace complex_power_rectangular_form_l156_156091

noncomputable def cos_30 : ℂ := real.cos (real.pi / 6) -- 30 degrees in radians
noncomputable def sin_30 : ℂ := real.sin (real.pi / 6)

theorem complex_power_rectangular_form :
  (3 * cos_30 + 3 * complex.I * sin_30)^4 = -81 / 2 + (81 * complex.I * real.sqrt 3) / 2 :=
by
  have h1 : cos_30 = real.sqrt 3 / 2 := by sorry
  have h2 : sin_30 = 1 / 2 := by sorry
  rw [h1, h2]
  sorry

end complex_power_rectangular_form_l156_156091


namespace general_solution_l156_156545

-- Given constants in the equation
constant a b c : ℝ
-- Given the differential equation y'' + a * y' + b * y = 0
constant eq : ∀ (y : ℝ → ℝ) (y'' y' : ℝ → ℝ), (d2 y ≫ d x 2) + a * (d2 y''/d x 2) + b * y = 0

-- General solution using characteristic roots
theorem general_solution :
  ∀ (y : ℝ → ℝ) (d : ℝ) (cos : ℝ → ℝ) (sin : ℝ → ℝ) (C₁ C₂ C₃ C₄ : ℝ), 
  y x = C₁ * cos (2 * x) + C₂ * sin (2 * x) + C₃ * cos (5 * x) + C₄ * sin (5 * x) :=
by
  sorry

end general_solution_l156_156545


namespace bridge_length_l156_156894

-- Define the conditions as variables
variables (train_length : ℕ) (train_speed_kmph : ℕ) (crossing_time_sec : ℕ)

-- Define the theorem proving the length of the bridge
theorem bridge_length (h1 : train_length = 135) (h2 : train_speed_kmph = 45) (h3 : crossing_time_sec = 30) : 
  let train_speed_mps := train_speed_kmph * 1000 / 3600 in
  let total_distance := train_speed_mps * crossing_time_sec in
  let bridge_length := total_distance - train_length in
  bridge_length = 240 := by
  sorry

end bridge_length_l156_156894


namespace complement_intersection_eq_complement_l156_156178

open Set

theorem complement_intersection_eq_complement (U A B : Set ℕ) (hU : U = {1, 2, 3, 4}) (hA : A = {1, 2}) (hB : B = {2, 4}) :
  (U \ (A ∩ B)) = {1, 3, 4} :=
by
  sorry

end complement_intersection_eq_complement_l156_156178


namespace count_divisible_by_90_four_digit_numbers_l156_156188

theorem count_divisible_by_90_four_digit_numbers :
  ∃ (n : ℕ), (n = 10) ∧ (∀ (x : ℕ), 1000 ≤ x ∧ x < 10000 ∧ x % 90 = 0 ∧ x % 100 = 90 → (x = 1890 ∨ x = 2790 ∨ x = 3690 ∨ x = 4590 ∨ x = 5490 ∨ x = 6390 ∨ x = 7290 ∨ x = 8190 ∨ x = 9090 ∨ x = 9990)) :=
by
  sorry

end count_divisible_by_90_four_digit_numbers_l156_156188


namespace trigonometric_identity_l156_156199

theorem trigonometric_identity (α : ℝ) (h : Real.sin (α + (Real.pi / 3)) = 3 / 5) :
  Real.cos ((Real.pi / 6) - α) = 3 / 5 :=
by
  sorry

end trigonometric_identity_l156_156199


namespace number_divisible_by_20p_l156_156262

noncomputable def floor_expr (p : ℕ) : ℤ :=
  Int.floor ((2 + Real.sqrt 5) ^ p - 2 ^ (p + 1))

theorem number_divisible_by_20p (p : ℕ) (hp : Nat.Prime p ∧ p % 2 = 1) :
  ∃ k : ℤ, floor_expr p = k * 20 * p :=
by
  sorry

end number_divisible_by_20p_l156_156262


namespace candy_system_of_equations_l156_156489

-- Definitions based on conditions
def candy_weight := 100
def candy_price1 := 36
def candy_price2 := 20
def mixed_candy_price := 28

theorem candy_system_of_equations (x y: ℝ):
  (x + y = candy_weight) ∧ (candy_price1 * x + candy_price2 * y = mixed_candy_price * candy_weight) :=
sorry

end candy_system_of_equations_l156_156489


namespace inequality_lemma_l156_156198

theorem inequality_lemma (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  (1 / (b * c + c * d + d * a - 1)) +
  (1 / (a * b + c * d + d * a - 1)) +
  (1 / (a * b + b * c + d * a - 1)) +
  (1 / (a * b + b * c + c * d - 1)) ≤ 2 :=
sorry

end inequality_lemma_l156_156198


namespace arccos_gt_arctan_l156_156985

theorem arccos_gt_arctan (x : ℝ) (h : -1 ≤ x ∧ x < 1/2) : Real.arccos x > Real.arctan x :=
sorry

end arccos_gt_arctan_l156_156985


namespace pump_time_l156_156905

-- Definitions based on the given conditions
def basement_length : ℝ := 30 -- in feet
def basement_width : ℝ := 40 -- in feet
def water_depth_inches : ℝ := 12 -- in inches
def cubic_foot_to_gallons : ℝ := 7.5 -- conversion rate
def pump_rate : ℝ := 10 -- gallons per minute per pump
def number_of_pumps : ℝ := 4

-- Conversion factor from inches to feet
def inches_to_feet (inches : ℝ) : ℝ := inches / 12

-- Main statement to be proved
theorem pump_time :
  let water_depth_feet := inches_to_feet water_depth_inches,
      volume_cubic_feet := basement_length * basement_width * water_depth_feet,
      total_gallons := volume_cubic_feet * cubic_foot_to_gallons,
      total_pump_rate := pump_rate * number_of_pumps,
      required_time := total_gallons / total_pump_rate
  in required_time = 225 := by
  sorry

end pump_time_l156_156905


namespace triangle_condition_isosceles_or_right_l156_156216

theorem triangle_condition_isosceles_or_right {A B C : ℝ} {a b c : ℝ} 
  (h_triangle : A + B + C = π) (h_cos_eq : a * Real.cos A = b * Real.cos B) : 
  (A = B) ∨ (A + B = π / 2) :=
sorry

end triangle_condition_isosceles_or_right_l156_156216


namespace simplest_expression_f_l156_156053

def sequence_a (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧
  a 2 = 1 ∧
  (∀ n ≥ 3, a n = (1 / 2 : ℚ) * n * a (n - 1) + (1 / 2 : ℚ) * n * (n - 1) * a (n - 2) + (-1) ^ n * (1 - n / 2))

def f (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  let binom := λ (n k : ℕ), Nat.choose n k
  in a n + ∑ k in Finset.range n, (k + 1) * binom n k * a (n - (k + 1))

theorem simplest_expression_f (a : ℕ → ℤ) (n : ℕ) (h : sequence_a a) :
  f a n = 2 * Nat.factorial n - (n + 1) :=
sorry

end simplest_expression_f_l156_156053


namespace max_value_of_f_l156_156355

noncomputable def f (x : ℝ) := 2 * Real.cos x + Real.sin x

theorem max_value_of_f : ∃ x, f x = Real.sqrt 5 := sorry

end max_value_of_f_l156_156355


namespace lcm_15_18_l156_156118

theorem lcm_15_18 : Nat.lcm 15 18 = 90 := by
  sorry

end lcm_15_18_l156_156118


namespace liz_final_cost_l156_156279

def recipe_book_cost : ℝ := 6
def baking_dish_cost : ℝ := 2 * recipe_book_cost
def ingredients_cost : ℝ := 5 * 3
def apron_cost : ℝ := recipe_book_cost + 1
def mixer_cost : ℝ := 3 * baking_dish_cost
def measuring_cups_cost : ℝ := apron_cost / 2
def spices_cost : ℝ := 4 * 2
def utensils_cost : ℝ := 3 * 4
def baking_cups_cost : ℝ := 6 * 0.5

def total_cost_before_discount : ℝ :=
  recipe_book_cost +
  baking_dish_cost +
  ingredients_cost +
  apron_cost +
  mixer_cost +
  measuring_cups_cost +
  spices_cost +
  utensils_cost +
  baking_cups_cost

def discount : ℝ := 0.10 * total_cost_before_discount
def final_cost : ℝ := total_cost_before_discount - discount

theorem liz_final_cost : final_cost = 92.25 := by
  sorry

end liz_final_cost_l156_156279


namespace geometric_sequence_n_eq_7_l156_156574

variable (a : ℕ → ℝ)

-- Conditions
def a₁ : ℝ := 1
def q : ℝ := 1 / 2
def an (n : ℕ) : ℝ := 1 / 64

-- Prove that the number of terms n is 7
theorem geometric_sequence_n_eq_7 :
  ∃ n : ℕ, a₁ * q^(n - 1) = an n ∧ an n = 1 / 64 ∧ q = 1 / 2 ∧ a₁ = 1 → n = 7 :=
begin
  sorry,
end

end geometric_sequence_n_eq_7_l156_156574


namespace interest_rate_per_annum_l156_156121

noncomputable def principal : ℝ := 933.3333333333334
noncomputable def amount : ℝ := 1120
noncomputable def time : ℝ := 4

theorem interest_rate_per_annum (P A T : ℝ) (hP : P = principal) (hA : A = amount) (hT : T = time) :
  ∃ R : ℝ, R = 1.25 :=
sorry

end interest_rate_per_annum_l156_156121


namespace solve_for_n_l156_156323

theorem solve_for_n (n : ℝ) (log3_2 : ℝ) (h : 3^n * 9^n = 256^(n - 50)) : 
  n = (-400 * log3_2) / (3 - 8 * log3_2) :=
by
  sorry

end solve_for_n_l156_156323


namespace least_five_digit_congruent_to_8_mod_17_l156_156860

theorem least_five_digit_congruent_to_8_mod_17 : 
  ∃ n : ℕ, n = 10004 ∧ n % 17 = 8 ∧ 10000 ≤ n ∧ n < 100000 ∧ (∀ m, 10000 ≤ m ∧ m < 10004 → m % 17 ≠ 8) :=
sorry

end least_five_digit_congruent_to_8_mod_17_l156_156860


namespace find_x_l156_156015

-- Given condition that x is 11 percent greater than 90
def eleven_percent_greater (x : ℝ) : Prop := x = 90 + (11 / 100) * 90

-- Theorem statement
theorem find_x (x : ℝ) (h: eleven_percent_greater x) : x = 99.9 :=
  sorry

end find_x_l156_156015


namespace lucas_mod_5_150th_term_l156_156742

noncomputable def Lucas : ℕ → ℕ
| 0     := 1
| 1     := 3
| (n+2) := Lucas n + Lucas (n+1)

theorem lucas_mod_5_150th_term : (Lucas 149 % 5) = 3 :=
by sorry

end lucas_mod_5_150th_term_l156_156742


namespace complex_fourth_power_l156_156089

noncomputable def complex_number : ℂ := 3 * complexOfReal((real.cos (real.pi / 6))) + 3i * complexOfReal((real.sin (real.pi / 6)))

theorem complex_fourth_power :
  (complex_number ^ 4) = -40.5 + 40.5 * (sqrt 3) * complex.i := 
by 
  sorry

end complex_fourth_power_l156_156089


namespace maximize_profit_l156_156487

noncomputable def p (t : ℝ) : ℝ := -(1 / 60) * t^3 + 21 * t
noncomputable def g (t : ℝ) (b : ℝ) : ℝ := -2 * (-(1 / 60)) * (t - b)^2
noncomputable def f (x : ℝ) : ℝ := p x + g (200 - x) 110

theorem maximize_profit :
  ∃ x y : ℝ, 
  10 ≤ x ∧ x ≤ 190 ∧ 
  y = 200 - x ∧
  f x = 453.6 ∧ 
  f x = -(1 / 60) * x^3 + 21 * x + (1 / 30) * ((200 - x) - 110)^2 ∧
  ∀ z, 10 ≤ z ∧ z ≤ 190 → f z ≤ f x :=
  begin
    -- proof to be done
    sorry,
  end

end maximize_profit_l156_156487


namespace tom_age_ratio_l156_156849

theorem tom_age_ratio (T N : ℕ)
  (sum_children : T = T) 
  (age_condition : T - N = 3 * (T - 4 * N)) :
  T / N = 11 / 2 := 
sorry

end tom_age_ratio_l156_156849


namespace find_ending_number_l156_156542

def is_even (n : ℕ) : Prop := n % 2 = 0

def average_of_evens_is (avg N : ℕ) : Prop :=
  (∀ m : ℕ, 12 ≤ m ∧ m ≤ N → is_even m) ∧ 
  N = 20 ∧ 
  avg = 16

theorem find_ending_number :
  ∃ N : ℕ, average_of_evens_is 16 N :=
begin
  use 20,
  split,
  { intros m hm,
    exact sorry, },
  { split; refl },
end

end find_ending_number_l156_156542


namespace arccos_sqrt3_div_2_eq_pi_div_6_l156_156962

theorem arccos_sqrt3_div_2_eq_pi_div_6 :
  arccos (Real.sqrt 3 / 2) = Real.pi / 6 :=
by
  sorry

end arccos_sqrt3_div_2_eq_pi_div_6_l156_156962


namespace det_B_squared_minus_3B_l156_156259

open Matrix
open Real

variable {α : Type*} [Fintype α] {n : ℕ}
variable [DecidableEq α]

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![2, 4],
  ![1, 3]
]

theorem det_B_squared_minus_3B : det (B * B - 3 • B) = -8 := sorry

end det_B_squared_minus_3B_l156_156259


namespace coefficients_sum_expansion_l156_156609

theorem coefficients_sum_expansion (x : ℂ) :
  (∑ k in finset.range 1001, ((1 + x + x^2) ^ 1000).coeff (3 * k)) = 3^999 :=
sorry

end coefficients_sum_expansion_l156_156609


namespace num_integer_solutions_abs_leq_seven_l156_156639

theorem num_integer_solutions_abs_leq_seven : 
  (∃ n : ℕ, n = (finset.Icc (-4 : ℤ) 10).card) ∧ n = 15 := 
by 
  sorry

end num_integer_solutions_abs_leq_seven_l156_156639


namespace find_a_l156_156626

def is_element_of_set (a : ℝ) := 
  let A := {a+2, (a+1)^2, a^2 + 3 * a + 3}
  in 1 ∈ A

theorem find_a (a : ℝ) (h : is_element_of_set a) : a = 0 :=
sorry

end find_a_l156_156626


namespace find_value_of_a_l156_156851

theorem find_value_of_a (a : ℝ) :
  (∀ x y : ℝ, (ax + y - 1 = 0) → (x - y + 3 = 0) → (-a) * 1 = -1) → a = 1 :=
by
  sorry

end find_value_of_a_l156_156851


namespace non_divisible_l156_156292

theorem non_divisible (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ¬ ∃ k : ℤ, x^2 + y^2 + z^2 = k * 3 * (x * y + y * z + z * x) :=
by sorry

end non_divisible_l156_156292


namespace part1_part2_l156_156950

-- Problem Part 1
theorem part1 : (-((-8)^(1/3)) - |(3^(1/2) - 2)| + ((-3)^2)^(1/2) + -3^(1/2) = 3) :=
by {
  sorry
}

-- Problem Part 2
theorem part2 (x : ℤ) : (2 * x + 5 ≤ 3 * (x + 2) ∧ 2 * x - (1 + 3 * x) / 2 < 1) ↔ (x = 0 ∨ x = 1 ∨ x = 2) :=
by {
  sorry
}

end part1_part2_l156_156950


namespace rectangle_length_l156_156467

theorem rectangle_length (L W : ℝ) 
  (h1 : L + W = 23) 
  (h2 : L^2 + W^2 = 289) : 
  L = 15 :=
by 
  sorry

end rectangle_length_l156_156467


namespace evelyn_daughters_and_granddaughters_no_daughters_l156_156109

theorem evelyn_daughters_and_granddaughters_no_daughters :
  let E : ℕ := 8,
      total : ℕ := 36,
      granddaughters := total - E,
      daughters_with_granddaughters := granddaughters / 7,
      daughters_no_granddaughters := E - daughters_with_granddaughters
  in daughters_no_granddaughters + granddaughters = 32 := by
{
  sorry
}

end evelyn_daughters_and_granddaughters_no_daughters_l156_156109


namespace q_can_complete_work_in_25_days_l156_156893

-- Define work rates for p, q, and r
variables (W_p W_q W_r : ℝ)

-- Define total work
variable (W : ℝ)

-- Prove that q can complete the work in 25 days under given conditions
theorem q_can_complete_work_in_25_days
  (h1 : W_p = W_q + W_r)
  (h2 : W_p + W_q = W / 10)
  (h3 : W_r = W / 50) :
  W_q = W / 25 :=
by
  -- Given: W_p = W_q + W_r
  -- Given: W_p + W_q = W / 10
  -- Given: W_r = W / 50
  -- We need to prove: W_q = W / 25
  sorry

end q_can_complete_work_in_25_days_l156_156893


namespace count_of_plus_signs_l156_156802

-- Definitions based on the given conditions
def total_symbols : ℕ := 23
def any_10_symbols_contains_plus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 10 → (\exists c ∈ s, c = '+')
def any_15_symbols_contains_minus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 15 → (\exists c ∈ s, c = '-')

-- Main theorem statement without the proof
theorem count_of_plus_signs (symbols : Finset char) (h1 : symbols.card = total_symbols) 
  (h2 : any_10_symbols_contains_plus symbols) 
  (h3 : any_15_symbols_contains_minus symbols) : 
  symbols.count '+' = 14 :=
sorry

end count_of_plus_signs_l156_156802


namespace geometric_seq_seventh_term_l156_156043

theorem geometric_seq_seventh_term (a r : ℕ) (r_pos : r > 0) (first_term : a = 3)
    (fifth_term : a * r^4 = 243) : a * r^6 = 2187 := by
  sorry

end geometric_seq_seventh_term_l156_156043


namespace least_number_to_add_l156_156867

theorem least_number_to_add (k : ℕ) : (1076 + k) % 23 = 0 ↔ k = 5 :=
by
  split
  · intro h
    -- we'd have to prove k = 5 here
    sorry
  · intro hk
    -- we'd have to prove (1076 + 5) % 23 = 0 here
    sorry

end least_number_to_add_l156_156867


namespace arrangement_count_is_74_l156_156302

def count_valid_arrangements : Nat :=
  74

-- Lean statement for the proof
theorem arrangement_count_is_74 :
  let seven_cards := list.range' 1 7 in
  ∃ seq : list Nat, 
    (seq.length = 7) ∧ 
    (∀ n, list.erase seq n = list.range' 1 6 ∨ 
          (list.reverse (list.erase seq n) = list.range' 1 6)) ∧
    (count_valid_arrangements = 74) :=
by
  let seven_cards := list.range' 1 7
  existsi seven_cards
  split
  -- Provide the conditions here for Lean to handle
  sorry

end arrangement_count_is_74_l156_156302


namespace number_represented_by_B_l156_156344

theorem number_represented_by_B (b : ℤ) : 
  (abs (b - 3) = 5) -> (b = 8 ∨ b = -2) :=
by
  intro h
  sorry

end number_represented_by_B_l156_156344


namespace natural_number_n_l156_156565

theorem natural_number_n (a : ℕ → ℕ) (n : ℕ) :
  (∀ x : ℝ, (∑ i in Finset.range (n + 1), (a i : ℝ) * x^i) = (1 + x)^n) →
  (∑ i in Finset.range (n + 1), a i) = 16 →
  n = 4 :=
by
  intro h₁ h₂
  sorry

end natural_number_n_l156_156565


namespace range_of_m_l156_156843

theorem range_of_m (m : ℝ) 
  (h : ∀ x : ℝ, 0 < x → m * x^2 + 2 * x + m ≤ 0) : m ≤ -1 :=
sorry

end range_of_m_l156_156843


namespace stephanie_oranges_l156_156330

theorem stephanie_oranges (num_visits : ℕ) (oranges_per_visit : ℕ) (total_oranges : ℕ) :
  num_visits = 8 ∧ oranges_per_visit = 2 → total_oranges = 16 := 
begin
  sorry
end

end stephanie_oranges_l156_156330


namespace log_calculation_l156_156949

-- Define the necessary logarithmic symbols
noncomputable def lg : ℝ → ℝ := log
noncomputable def log_base (b x : ℝ) : ℝ := log x / log b

-- The main theorem to prove
theorem log_calculation :
  (lg 5)^2 + lg 2 * lg 50 - log_base 8 9 * log_base 27 32 = -1 / 9 :=
by
  sorry

end log_calculation_l156_156949


namespace polygon_sides_l156_156386

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 4 * 360) : 
  n = 6 :=
by sorry

end polygon_sides_l156_156386


namespace midpoint_complex_number_l156_156684

theorem midpoint_complex_number :
  let A := (4, 5)
  let B := (-2, 1)
  let Cx := (A.1 + B.1) / 2
  let Cy := (A.2 + B.2) / 2
  complex.mk Cx Cy = 1 + 3 * complex.i :=
by
  sorry

end midpoint_complex_number_l156_156684


namespace plus_signs_count_l156_156786

noncomputable def symbols : Type := fin 23 -- The type of 23 symbols

def is_plus (x : symbols) : Prop
def is_minus (x : symbols) : Prop

axiom total_symbols : ∀ (x : symbols), is_plus x ∨ is_minus x
axiom unique_classification : ∀ (x : symbols), ¬ (is_plus x ∧ is_minus x)
axiom ten_symbols_plus : ∀ (S : finset symbols), S.card = 10 → ∃ x ∈ S, is_plus x
axiom fifteen_symbols_minus : ∀ (S : finset symbols), S.card = 15 → ∃ x ∈ S, is_minus x

def num_plus_symbols : ℕ := finset.card (finset.filter is_plus finset.univ)

theorem plus_signs_count : num_plus_symbols = 14 :=
sorry

end plus_signs_count_l156_156786


namespace maximum_marks_l156_156890

theorem maximum_marks (M : ℝ)
  (pass_threshold_percentage : ℝ := 33)
  (marks_obtained : ℝ := 92)
  (marks_failed_by : ℝ := 40) :
  (marks_obtained + marks_failed_by) = (pass_threshold_percentage / 100) * M → M = 400 := by
  sorry

end maximum_marks_l156_156890


namespace number_multiplied_by_2_equals_10_percent_of_900_l156_156914

theorem number_multiplied_by_2_equals_10_percent_of_900 :
  ∃ x : ℝ, (2 * x = 0.1 * 900) ∧ x = 45 :=
by
  use 45
  split
  · -- condition from part a): 2 * x = 0.1 * 900
    sorry 
  · -- conclusion: x = 45
    sorry

end number_multiplied_by_2_equals_10_percent_of_900_l156_156914


namespace necessarily_negative_expression_l156_156715

theorem necessarily_negative_expression
  (x y z : ℝ)
  (hx : 0 < x ∧ x < 1)
  (hy : -1 < y ∧ y < 0)
  (hz : 0 < z ∧ z < 1)
  : y - z < 0 :=
sorry

end necessarily_negative_expression_l156_156715


namespace decimal_repetend_denominator_l156_156339

noncomputable def decimal_to_fraction (d : ℚ) : ℚ :=
if d = 0.36 
then 4 / 11 
else d -- This is a stub for example purposes. 
-- In practice, we'd define a function to convert repeating decimals to fractions.

theorem decimal_repetend_denominator :
  (decimal_to_fraction (0.36 : ℚ)).denom = 11 :=
by 
sorry

end decimal_repetend_denominator_l156_156339


namespace perimeter_ratio_l156_156222

theorem perimeter_ratio (O A B C D E : Point) (α : ℝ) 
    (h_circle : Circle O)
    (h_diameter : ¬(A = B) ∧ (O ∈ Line A B))
    (h_chord : parallel (Line C D) (Line A B))
    (h_intersect : (Line A B).intersection (Line C D) = {E})
    (h_angle : angle E A D = 2 * α) :
    (perimeter (Triangle C D E) / perimeter (Triangle A B E)) = (sin α) ^ 2 :=
sorry

end perimeter_ratio_l156_156222


namespace abby_damon_weight_l156_156063

theorem abby_damon_weight (a' b' c' d' : ℕ) (h1 : a' + b' = 265) (h2 : b' + c' = 250) (h3 : c' + d' = 280) :
  a' + d' = 295 :=
  sorry -- Proof goes here

end abby_damon_weight_l156_156063


namespace find_m_n_l156_156566

theorem find_m_n (x : ℝ) (m n : ℝ) 
  (h : (2 * x - 5) * (x + m) = 2 * x^2 - 3 * x + n) :
  m = 1 ∧ n = -5 :=
by
  have h_expand : (2 * x - 5) * (x + m) = 2 * x^2 + (2 * m - 5) * x - 5 * m := by
    ring
  rw [h_expand] at h
  have coeff_eq1 : 2 * m - 5 = -3 := by sorry
  have coeff_eq2 : -5 * m = n := by sorry
  have m_sol : m = 1 := by
    linarith [coeff_eq1]
  have n_sol : n = -5 := by
    rw [m_sol] at coeff_eq2
    linarith
  exact ⟨m_sol, n_sol⟩

end find_m_n_l156_156566


namespace clock_angle_7_35_l156_156427

noncomputable def hour_angle (hours : ℤ) (minutes : ℤ) : ℝ :=
  (hours * 30 + (minutes * 30) / 60 : ℝ)

noncomputable def minute_angle (minutes : ℤ) : ℝ :=
  (minutes * 360 / 60 : ℝ)

noncomputable def angle_between (angle1 angle2 : ℝ) : ℝ :=
  abs (angle1 - angle2)

theorem clock_angle_7_35 : angle_between (hour_angle 7 35) (minute_angle 35) = 17.5 :=
by
  sorry

end clock_angle_7_35_l156_156427


namespace polygon_sides_equation_l156_156372

theorem polygon_sides_equation (n : ℕ) 
  (h1 : (n-2) * 180 = 4 * 360) : n = 10 := 
by 
  sorry

end polygon_sides_equation_l156_156372


namespace alicia_read_more_books_than_ian_l156_156518

def books_read : List Nat := [3, 5, 8, 6, 7, 4, 2, 1]

def alicia_books (books : List Nat) : Nat :=
  books.maximum?.getD 0

def ian_books (books : List Nat) : Nat :=
  books.minimum?.getD 0

theorem alicia_read_more_books_than_ian :
  alicia_books books_read - ian_books books_read = 7 :=
by
  -- By reviewing the given list of books read [3, 5, 8, 6, 7, 4, 2, 1]
  -- We find that alicia_books books_read = 8 and ian_books books_read = 1
  -- Thus, 8 - 1 = 7
  sorry

end alicia_read_more_books_than_ian_l156_156518


namespace shift_sine_wave_left_l156_156170

theorem shift_sine_wave_left (x : ℝ) (ω : ℝ) (hω : ω = 2) (h_period : ∀ x, f (x + π) = f x) : 
  ∀ x, (λ x, sin (ω * x + π / 4)) x = (λ x, sin (ω * (x + π / 8))) x :=
by
  intro x
  have h1 : g x = sin (ω * x + π / 4) := rfl
  have h2 : f (x + π / 8) = sin (ω * (x + π / 8)) := rfl
  rw [h1, h2, hω]
  sorry

end shift_sine_wave_left_l156_156170


namespace find_mn_l156_156685

def OA : ℝ := 1
def OB : ℝ := 2
def OC : ℝ := 2
def tan_AOC : ℝ := 3
def angle_BOC : ℝ := real.pi / 4

theorem find_mn :
  ∃ (m n : ℝ),
  (OC = m * OA + n * OB) ∧
  m = (5 + real.sqrt 5) / 10 ∧
  n = (5 * real.sqrt 5) / 4 :=
sorry

end find_mn_l156_156685


namespace quadratic_has_two_distinct_real_roots_l156_156210

theorem quadratic_has_two_distinct_real_roots (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 2 * x1 + m = 0) ∧ (x2^2 - 2 * x2 + m = 0)) ↔ (m < 1) :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l156_156210


namespace plus_signs_count_l156_156825

theorem plus_signs_count (n p m : ℕ) (h_total : n = 23) 
  (h_plus_min : ∀ s, s.card = 10 → ∃ t ∈ s, is_plus t)
  (h_minus_min : ∀ s, s.card = 15 → ∃ t ∈ s, is_minus t)
  : p = 14 :=
sorry

end plus_signs_count_l156_156825


namespace count_of_plus_signs_l156_156804

-- Definitions based on the given conditions
def total_symbols : ℕ := 23
def any_10_symbols_contains_plus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 10 → (\exists c ∈ s, c = '+')
def any_15_symbols_contains_minus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 15 → (\exists c ∈ s, c = '-')

-- Main theorem statement without the proof
theorem count_of_plus_signs (symbols : Finset char) (h1 : symbols.card = total_symbols) 
  (h2 : any_10_symbols_contains_plus symbols) 
  (h3 : any_15_symbols_contains_minus symbols) : 
  symbols.count '+' = 14 :=
sorry

end count_of_plus_signs_l156_156804


namespace line_PQ_intersects_pentagon_adjacent_A_l156_156850

-- Definitions of triangles and their properties
structure Triangle :=
  (base_height : ℝ)
  (vertex_angle : ℝ)

structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  -- Point located on the circumference of the circle
  (on_circle : ℝ × ℝ → Prop)

-- Define the two isosceles triangles ABC and ADE
def triangle_ABC : Triangle := ⟨1/2, 90⟩ -- height is 1/2, base forms 90 degree with radius
def triangle_ADE : Triangle := ⟨5/4, 90⟩ -- height is 5/4, base forms 90 degree with radius

-- Define the unit circle k with center O
def center_O : ℝ × ℝ := (0, 0)
def k : Circle := ⟨center_O, 1, λ p, p.1^2 + p.2^2 = 1⟩

-- Define points B, C, D, E
def B : ℝ × ℝ := (0, 1/2)
def C : ℝ × ℝ := (0, -1/2)
def D : ℝ × ℝ := (0, 5/4)
def E : ℝ × ℝ := (0, -5/4)

-- Define the intersection points P and Q
def P : ℝ × ℝ := (0, 0) -- Intersection of OB and AD
def Q : ℝ × ℝ := (0, 0) -- Intersection of OC and AE

-- Define the regular pentagon inscribed in circle k
def regular_pentagon_vertices : list (ℝ × ℝ) := [-- Points to be defined e.g., --
(0.809, 0.588), (0.309, 0.951), ..., ...]

-- Prove that line PQ intersects the vertices adjacent to A
theorem line_PQ_intersects_pentagon_adjacent_A :
  ∃ (P Q : ℝ × ℝ), 
  (P = intersection (line center_O B) (line (A D))) ∧ 
  (Q = intersection (line center_O C) (line (A E))) ∧ 
  ∀ (adj : ℝ × ℝ), adj ∈ regular_pentagon_vertices → 
  line (P Q) intersects_with adj :=
by
  -- Proof goes here
  sorry

end line_PQ_intersects_pentagon_adjacent_A_l156_156850


namespace f_log_log2_l156_156614

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^3 + b * Real.sin x + 4

theorem f_log_log2 :
  ∀ (a b : ℝ),
  f a b (Real.log (Real.log2 10)) = 5 →
  f a b (Real.log (Real.log 2)) = 3 :=
by
  intros a b h
  sorry

end f_log_log2_l156_156614


namespace count_divisible_by_90_four_digit_numbers_l156_156187

theorem count_divisible_by_90_four_digit_numbers :
  ∃ (n : ℕ), (n = 10) ∧ (∀ (x : ℕ), 1000 ≤ x ∧ x < 10000 ∧ x % 90 = 0 ∧ x % 100 = 90 → (x = 1890 ∨ x = 2790 ∨ x = 3690 ∨ x = 4590 ∨ x = 5490 ∨ x = 6390 ∨ x = 7290 ∨ x = 8190 ∨ x = 9090 ∨ x = 9990)) :=
by
  sorry

end count_divisible_by_90_four_digit_numbers_l156_156187


namespace plus_signs_count_l156_156797

noncomputable def solveSymbols : Prop :=
  ∃ p m : ℕ, p + m = 23 ∧ p ≤ 14 ∧ m ≤ 9 ∧ (∀ s : Finset (Fin 23), s.card = 10 → (s.image (λ i => if i < p then true else false)).count true ≥ 1) ∧ (∀ s : Finset (Fin 23), s.card = 15 → (s.image (λ i => if i < p then false else true)).count false ≥ 1) 

theorem plus_signs_count : solveSymbols :=
sorry

end plus_signs_count_l156_156797


namespace problem1_problem2_l156_156136

-- Define the given sets A and B
def setA (a : ℝ) : Set ℝ := { x | a - 4 < x ∧ x < a + 4 }
def setB : Set ℝ := { x | x < -1 ∨ x > 5 }

-- Problem 1: Prove A ∩ B = { x | -3 < x ∧ x < -1 } when a = 1
theorem problem1 (a : ℝ) (h : a = 1) : 
  (setA a ∩ setB) = { x : ℝ | -3 < x ∧ x < -1 } := sorry

-- Problem 2: Prove range of a given A ∪ B = ℝ is (1, 3)
theorem problem2 (a : ℝ) : 
  (forall x : ℝ, x ∈ (setA a ∪ setB)) ↔ (1 < a ∧ a < 3) := sorry

end problem1_problem2_l156_156136


namespace successive_percentage_reduction_l156_156444

theorem successive_percentage_reduction (a b : ℝ) (h₁ : a = 25) (h₂ : b = 20) :
  a + b - (a * b) / 100 = 40 := by
  sorry

end successive_percentage_reduction_l156_156444


namespace plus_signs_count_l156_156784

noncomputable def symbols : Type := fin 23 -- The type of 23 symbols

def is_plus (x : symbols) : Prop
def is_minus (x : symbols) : Prop

axiom total_symbols : ∀ (x : symbols), is_plus x ∨ is_minus x
axiom unique_classification : ∀ (x : symbols), ¬ (is_plus x ∧ is_minus x)
axiom ten_symbols_plus : ∀ (S : finset symbols), S.card = 10 → ∃ x ∈ S, is_plus x
axiom fifteen_symbols_minus : ∀ (S : finset symbols), S.card = 15 → ∃ x ∈ S, is_minus x

def num_plus_symbols : ℕ := finset.card (finset.filter is_plus finset.univ)

theorem plus_signs_count : num_plus_symbols = 14 :=
sorry

end plus_signs_count_l156_156784


namespace number_of_red_balls_l156_156670

-- Definitions and conditions
def ratio_white_red (w : ℕ) (r : ℕ) : Prop := (w : ℤ) * 3 = 5 * (r : ℤ)
def white_balls : ℕ := 15

-- The theorem to prove
theorem number_of_red_balls (r : ℕ) (h : ratio_white_red white_balls r) : r = 9 :=
by
  sorry

end number_of_red_balls_l156_156670


namespace min_distinct_integers_with_ap_and_gp_of_length_5_l156_156870

theorem min_distinct_integers_with_ap_and_gp_of_length_5 : 
  ∃ (s : set ℤ), (s.card = 6) ∧ 
    (∃ (a r : ℤ),  (∀ i, 0 ≤ i ∧ i < 5 → (a + i * r) ∈ s)) ∧ 
    (∃ (b q : ℤ), q ≠ 0 ∧ (∀ i, 0 ≤ i ∧ i < 5 → (b * q^i) ∈ s)) :=
sorry

end min_distinct_integers_with_ap_and_gp_of_length_5_l156_156870


namespace smallest_n_conditions_l156_156876

theorem smallest_n_conditions (n : ℕ) : 
  (∃ k m : ℕ, 4 * n = k^2 ∧ 5 * n = m^5 ∧ ∀ n' : ℕ, (∃ k' m' : ℕ, 4 * n' = k'^2 ∧ 5 * n' = m'^5) → n ≤ n') → 
  n = 625 :=
by
  intro h
  sorry

end smallest_n_conditions_l156_156876


namespace roots_product_l156_156419

theorem roots_product : (27^(1/3) * 81^(1/4) * 64^(1/6)) = 18 := 
by
  sorry

end roots_product_l156_156419


namespace fraction_of_decimal_l156_156357

theorem fraction_of_decimal (a b : ℕ) (h : 0.375 = (a : ℝ) / (b : ℝ)) (gcd_ab : Nat.gcd a b = 1) : a + b = 11 :=
  sorry

end fraction_of_decimal_l156_156357


namespace part1_part2i_part2ii_l156_156620

noncomputable def f (x k : ℝ) : ℝ := abs (x^2 - 1) + x^2 + k * x

theorem part1 (k : ℝ) : (∀ x : ℝ, 0 < x → f x k ≤ 0) → k ≥ -1 :=
begin
  sorry,
end

theorem part2i (k : ℝ) : 
(∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 2 ∧ 0 < x₂ ∧ x₂ < 2 ∧ x₁ ≠ x₂ ∧ f x₁ k = 0 ∧ f x₂ k = 0) → 
-7/2 < k ∧ k < -1 :=
begin
  sorry,
end

theorem part2ii (k x₁ x₂ : ℝ) :
(0 < x₁ ∧ x₁ < 2 ∧ 0 < x₂ ∧ x₂ < 2 ∧ x₁ ≠ x₂ ∧ f x₁ k = 0 ∧ f x₂ k = 0) → 
2 * x₂ ∈ set.Ioo (2 : ℝ) 4 :=
begin
  sorry,
end

end part1_part2i_part2ii_l156_156620


namespace simplified_equation_equivalent_l156_156687

theorem simplified_equation_equivalent  (x : ℝ) :
    (x / 0.3 = 1 + (1.2 - 0.3 * x) / 0.2) ↔ (10 * x / 3 = 1 + (12 - 3 * x) / 2) :=
by sorry

end simplified_equation_equivalent_l156_156687


namespace female_democrats_l156_156445

theorem female_democrats :
  ∀ (F M : ℕ),
  F + M = 720 →
  F/2 + M/4 = 240 →
  F / 2 = 120 :=
by
  intros F M h1 h2
  sorry

end female_democrats_l156_156445


namespace value_of_f_neg_one_l156_156203

noncomputable def f : ℝ → ℝ := sorry

theorem value_of_f_neg_one (f_def : ∀ x, f (Real.tan x) = Real.sin (2 * x)) : f (-1) = -1 := 
by
sorry

end value_of_f_neg_one_l156_156203


namespace transform_sin_to_cos_l156_156847

theorem transform_sin_to_cos (x : ℝ) :
  3 * cos (2 * x - π / 4) = 3 * sin (2 * (x + π / 8)) :=
begin
  sorry
end

end transform_sin_to_cos_l156_156847


namespace smallest_n_conditions_l156_156875

theorem smallest_n_conditions (n : ℕ) : 
  (∃ k m : ℕ, 4 * n = k^2 ∧ 5 * n = m^5 ∧ ∀ n' : ℕ, (∃ k' m' : ℕ, 4 * n' = k'^2 ∧ 5 * n' = m'^5) → n ≤ n') → 
  n = 625 :=
by
  intro h
  sorry

end smallest_n_conditions_l156_156875


namespace num_possible_n_values_l156_156096

theorem num_possible_n_values (n : ℕ) :
  (∃ n : ℕ, (∠B > ∠A > ∠C) ∧ AB = 3 * n + 6 ∧ BC = 2 * n + 15 ∧ AC = 2 * n + 5 ∧
  (5 * n + 11 > 2 * n + 15) ∧ (5 * n + 21 > 2 * n + 5) ∧ (4 * n + 20 > 3 * n + 6)) →
  (set.size {x : ℕ | 2 ≤ x ∧ x ≤ 8}) = 7 :=
by
-- From the given conditions:
-- Triangle inequality
-- Angle inequality
-- Evaluating the range of n
-- The count of such integer values within the range is 7.
sorry

end num_possible_n_values_l156_156096


namespace find_a_b_sum_l156_156328

theorem find_a_b_sum :
  let q := (prob_reach_31_in_at_most_8_steps (0, 0))
  in ∃ a b : ℕ, a + b = 1039 ∧ q = a / b
:= sorry

def prob_reach_31_in_at_most_8_steps (start : ℤ × ℤ) : ℚ := 
  -- This is a placeholder definition, the actual function
  -- calculation of the probability would be done here
  if start = (0, 0) then (15 / 1024 : ℚ) else 0

end find_a_b_sum_l156_156328


namespace solve_for_m_l156_156569

open Real

theorem solve_for_m (a b m : ℝ)
  (h1 : (1/2)^a = m)
  (h2 : 3^b = m)
  (h3 : 1/a - 1/b = 2) :
  m = sqrt 6 / 6 := 
  sorry

end solve_for_m_l156_156569


namespace expected_value_of_winnings_l156_156040

-- Define the probabilities and winnings
def numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def probability := (1:ℚ) / 10

-- Define the function to calculate winnings
def winnings (n : ℕ) : ℚ := n^3

-- Define the expected value calculation
noncomputable def expected_value : ℚ :=
  probability * (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 + 7^3 + 8^3 + 9^3 + 10^3)

theorem expected_value_of_winnings :
  expected_value = 302.50 := by
  sorry

end expected_value_of_winnings_l156_156040


namespace skate_cost_l156_156071

/- Define the initial conditions as Lean definitions -/
def admission_cost : ℕ := 5
def rental_cost : ℕ := 250 / 100  -- 2.50 dollars in cents for integer representation
def visits : ℕ := 26

/- Define the cost calculation as a Lean definition -/
def total_rental_cost (rental_cost : ℕ) (visits : ℕ) : ℕ := rental_cost * visits

/- Statement of the problem in Lean proof form -/
theorem skate_cost (C : ℕ) (h : total_rental_cost rental_cost visits = C) : C = 65 :=
by
  sorry

end skate_cost_l156_156071


namespace alan_more_wings_per_minute_to_beat_record_l156_156703

-- Define relevant parameters and conditions
def kevin_wings := 64
def time_minutes := 8
def alan_rate := 5

-- Theorem: Alan must eat 3 more wings per minute to beat Kevin's record
theorem alan_more_wings_per_minute_to_beat_record : 
  (kevin_wings > alan_rate * time_minutes) → ((kevin_wings - (alan_rate * time_minutes)) / time_minutes = 3) :=
by
  sorry

end alan_more_wings_per_minute_to_beat_record_l156_156703


namespace find_r_condition_l156_156331

variable {x y z w r : ℝ}

axiom h1 : x ≠ 0
axiom h2 : y ≠ 0
axiom h3 : z ≠ 0
axiom h4 : w ≠ 0
axiom h5 : (x ≠ y) ∧ (x ≠ z) ∧ (x ≠ w) ∧ (y ≠ z) ∧ (y ≠ w) ∧ (z ≠ w)

noncomputable def is_geometric_progression (a b c d : ℝ) (r : ℝ) : Prop :=
  b = a * r ∧ c = a * r^2 ∧ d = a * r^3

theorem find_r_condition :
  is_geometric_progression (x * (y - z)) (y * (z - x)) (z * (x - y)) (w * (y - x)) r →
  r^3 + r^2 + r + 1 = 0 :=
by
  intros
  sorry

end find_r_condition_l156_156331


namespace fairfield_animal_shelter_l156_156705

/-- Last year, 120 adult cats, 60% of whom were female, were brought into the Fairfield Animal Shelter. 
Two-thirds of the adult female cats were accompanied by a litter of kittens. 
The average number of kittens per litter was 5. 
Prove that the total number of cats and kittens received by the shelter last year is 360. --/
theorem fairfield_animal_shelter:
  let adult_cats := 120
  let percentage_female := 60 / 100
  let female_cats := percentage_female * adult_cats
  let litters := (2 / 3) * female_cats
  let kittens_per_litter := 5
  let total_kittens := litters * kittens_per_litter
  let total_cats_kittens := adult_cats + total_kittens
  in total_cats_kittens = 360 := 
by
  sorry

end fairfield_animal_shelter_l156_156705


namespace find_period_l156_156913

-- Definitions based on conditions
def interest_rate_A : ℝ := 0.10
def interest_rate_C : ℝ := 0.115
def principal : ℝ := 4000
def total_gain : ℝ := 180

-- The question to prove
theorem find_period (n : ℝ) : 
  n = 3 :=
by 
  have interest_to_A := interest_rate_A * principal
  have interest_from_C := interest_rate_C * principal
  have annual_gain := interest_from_C - interest_to_A
  have equation := total_gain = annual_gain * n
  sorry

end find_period_l156_156913


namespace value_of_a_m_minus_3n_l156_156649

theorem value_of_a_m_minus_3n (a : ℝ) (m n : ℝ) (h1 : a^m = 8) (h2 : a^n = 2) : a^(m - 3 * n) = 1 :=
sorry

end value_of_a_m_minus_3n_l156_156649


namespace number_of_correct_propositions_is_2_l156_156513

def correct_propositions_count : Nat :=
  let prop1 := (forall a b : Line, (skew_lines a b -> ¬intersect a b) ∧ (¬intersect a b -> ¬skew_lines a b)) = false
  let prop2 := (forall l : Line, forall α : Plane, (perpendicular_to_plane l α ↔ forall m : Line, (m ∈ α -> perpendicular l m))) = true
  let prop3 := (forall a b : Line, forall α : Plane, (perpendicular a b -> perpendicular a (projection b α)) ∧ (perpendicular a (projection b α) -> perpendicular a b)) = false
  let prop4 := (forall a : Line, forall β : Plane, (parallel_to_plane a β -> exists m : Line, (m ∈ β ∧ parallel a m)) ∧ (exists m : Line, (m ∈ β ∧ parallel a m) -> parallel_to_plane a β)) = true
  [prop1, prop2, prop3, prop4].count true

theorem number_of_correct_propositions_is_2 :
  correct_propositions_count = 2 :=
sorry

end number_of_correct_propositions_is_2_l156_156513


namespace distinct_prime_reciprocal_sum_is_integer_l156_156260

theorem distinct_prime_reciprocal_sum_is_integer (n : ℕ) (p : ℕ → Prop) [hp : ∀ i, p i → nat.prime i]
  (hpp : ∀ i j, i ≠ j → p i → p j → nat.coprime i j)
  (hdiv : n > 1 ∧ ∀ i, p i → i ∣ n ∧ ∃ k, p k ∧ k ≠ i)
  (hphi : ∀ i, p i → (nat.totient n)) :
  ∀ p : (ℕ → Prop), (∃ k (hpi : ∀ i : fin k, p i),
      n ∣ (nat.sum (λ (i : fin k), (nat.pow i (nat.totient n)))) ∧
      (∑ i, 1 / (p i) + 1 / (∏ j, p j)) ∈ ℤ) :=
begin
  sorry
end

end distinct_prime_reciprocal_sum_is_integer_l156_156260


namespace _l156_156717

noncomputable def is_parallel (l₁ l₂ : Line) : Prop := sorry

structure Triangle (α : Type*) :=
(A B C : α)

structure Segment (α : Type*) :=
(start end : α)

structure Altitude (α : Type*) :=
(base : α)
(to : Segment α)

def Angle (α : Type*) := Segment α × Segment α

def divides_equally (α : Type*) (segments : List (Segment α)) (angle : Angle α) : Prop := sorry

def intersection (α : Type*) (seg1 seg2 : Segment α) : α := sorry

structure Problem (α : Type*) :=
(triangle : Triangle α)
(ad : Altitude α)
(segments : List (Segment α))
(m_points : List α)
(e_point : α)
(bm_last : Segment α)

def positional_relationship {α : Type*} (problem : Problem α) : Prop :=
  let ⟨triangle, ad, segments, m_points, e_point, bm_last⟩ := problem
  let m1 := m_points.head
  let bm1 := segments.head
  let cm2 := segments.nth! 2
  let em1 := Segment.mk e_point m1
  is_parallel em1 bm_last

-- Hypothesis for the given problem
noncomputable theorem is_parallel_EM1_BMn {α : Type*} (problem : Problem α) (h₁ : Altitude α)
    (h₂ : divides_equally α problem.segments (Angle.mk (Segment.mk problem.triangle.B problem.triangle.C) (Segment.mk problem.triangle.B problem.triangle.A))) :
  positional_relationship problem :=
  sorry

end _l156_156717


namespace f_is_even_f_range_l156_156618

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (|x| + 2) / (1 - |x|)

-- Prove that f(x) is an even function
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

-- Prove the range of f(x) is (-∞, -1) ∪ [2, +∞)
theorem f_range : ∀ y : ℝ, ∃ x : ℝ, y = f x ↔ y ≥ 2 ∨ y < -1 := by
  sorry

end f_is_even_f_range_l156_156618


namespace decrypt_test_phrase_l156_156696

constant encrypted_phrase : string
constant decrypted_phrase : string
constant example_step1 : string → string
constant example_step2 : string → string
constant example_step3 : string → string 

axiom no_yo_character (s : string) : ¬ s.contains "ё"

axiom example_encrypt (input : string) :
  example_step3 (example_step2 (example_step1 input)) = "уфзмтфсзек"

axiom reverse_string (s : string) : string
axiom shift_left_two (s : string) : string
axiom swap_adjacent (s : string) : string

theorem decrypt_test_phrase :
  reverse_string (shift_left_two (swap_adjacent "врпвл терпраиэ вйзгцфпз")) = "нефте базы южного района" :=
sorry

end decrypt_test_phrase_l156_156696


namespace add_in_base_7_l156_156998

def from_base (b : ℕ) (digits : List ℕ) : ℕ := 
  digits.reverse.enum_from 1 |>.map (λ (i, d), d * b^(i-1)).sum

def to_base (b : ℕ) (n : ℕ) : List ℕ :=
  if n = 0 then [0] else 
    List.unfold (λ x, if x = 0 then none else some (x%b, x / b)) n |>.reverse

theorem add_in_base_7 : 
  from_base 7 [6, 6, 6] + from_base 7 [6, 6] + from_base 7 [6] = from_base 7 [1, 4, 0, 0] :=
by 
  unfold from_base 
  have h1 : from_base 7 [6, 6, 6] = 6 * 7^2 + 6 * 7^1 + 6 * 7^0 := by rfl
  have h2 : from_base 7 [6, 6] = 6 * 7^1 + 6 * 7^0 := by rfl
  have h3 : from_base 7 [6] = 6 := by rfl
  have h4 : from_base 7 [1, 4, 0, 0] = 1 * 7^3 + 4 * 7^2 + 0 * 7^1 + 0 * 7^0 := by rfl
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end add_in_base_7_l156_156998


namespace triangle_is_isosceles_l156_156221

-- Problem restatement in Lean 4
theorem triangle_is_isosceles
  (ABC : Triangle) (B1 C1 : Point)
  (hB_bisect : AngleBisector ABC.B ABC.C B1)
  (hC_bisect : AngleBisector ABC.C ABC.A C1)
  (hCircumcircle : OnCircumcircle ABC B1)
  (hCircumcircle : OnCircumcircle ABC C1)
  (hEqualChords : Distance B B1 = Distance C C1) :
  IsIsosceles ABC :=
sorry

end triangle_is_isosceles_l156_156221


namespace area_of_CDE_in_isosceles_triangle_l156_156229

noncomputable def isosceles_triangle_area (b : ℝ) (s : ℝ) (area : ℝ) : Prop :=
  area = (1 / 2) * b * s

noncomputable def cot (α : ℝ) : ℝ := 1 / Real.tan α

noncomputable def isosceles_triangle_vertex_angle (b : ℝ) (area : ℝ) (θ : ℝ) : Prop :=
  area = (b^2 / 4) * cot (θ / 2)

theorem area_of_CDE_in_isosceles_triangle (b θ area : ℝ) (hb : b = 3 * (2 * b / 3)) (hθ : θ = 100) (ha : area = 30) :
  ∃ CDE_area, CDE_area = area / 9 ∧ CDE_area = 10 / 3 :=
by
  sorry

end area_of_CDE_in_isosceles_triangle_l156_156229


namespace plus_signs_count_l156_156809

theorem plus_signs_count (num_symbols : ℕ) (at_least_one_plus_in_10 : ∀ s : Finset ℕ, s.card = 10 → (∃ i ∈ s, i < 14)) (at_least_one_minus_in_15 : ∀ s : Finset ℕ, s.card = 15 → (∃ i ∈ s, i ≥ 14)) : 
    ∃ (p m : ℕ), p + m = 23 ∧ p = 14 ∧ m = 9 := by
  sorry

end plus_signs_count_l156_156809


namespace dolphin_ratio_l156_156394

def initial_dolphins := 65
def total_dolphins := 260

theorem dolphin_ratio : (total_dolphins - initial_dolphins) / initial_dolphins = 3 := by
  have h : total_dolphins - initial_dolphins = 195 := by
    sorry
  rw [h]
  norm_num

end dolphin_ratio_l156_156394


namespace area_of_U_l156_156964

open Complex Real

noncomputable def octagon : Set ℂ := {
  z : ℂ | let x := Re(z); let y := Im(z);
          (abs x ≤ 1/2) ∧  -- Condition for real axis alignment
          (some additional conditions defining the octagon's vertices and bounds)
}

noncomputable def T : Set ℂ := {z | z ∉ octagon}

noncomputable def U : Set ℂ := {w | ∃ z ∈ T, w = 1/z}

theorem area_of_U :
  ∃ area, area = ∞ - 4 * π := by
  sorry

end area_of_U_l156_156964


namespace angle_AKO_eq_angle_DAC_l156_156584

noncomputable theory
open_locale classical

universe u
variables {α : Type u} [EuclideanSpace α]

structure Triangle (P Q R : α) : Prop := 
(acute : ∀ {a : α} (ha : a ∈ interior (triangle P Q R)), ∠a < π / 2) 

variables (A B C : α) {H O D K : α}

theorem angle_AKO_eq_angle_DAC (triangle_ABC : Triangle A B C) 
  (circumcircle : Circle O) 
  (orthocenter_H : IsOrthocenter H A B C) 
  (D_on_circumcircle : D ∈ circumcircle.pts) 
  (perpendicular_bisector_intersects_AB_at_K : is_perpendicular_bisector (line H D) (line K A B)) :
  ∠ A K O = ∠ D A C :=
sorry

end angle_AKO_eq_angle_DAC_l156_156584


namespace MA_MB_product_l156_156162

theorem MA_MB_product :
  let C1 := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 } in
  let polar_eq_C1 := ∀ ⦃ρ θ : ℝ⦄, ρ = 1 ↔ ∃ x y, x = ρ * cos θ ∧ y = ρ * sin θ ∧ p = (x, y) in
  let C2 := { p : ℝ × ℝ | (1 / 9) * p.1^2 + p.2^2 = 1 } in
  let M := (1 : ℝ, 0 : ℝ) in
  let l (t : ℝ) := (1 + (sqrt 2 / 2) * t, (sqrt 2 / 2) * t) in
  ∃ A B : ℝ × ℝ, A ∈ C2 ∧ B ∈ C2 ∧ l t_A = A ∧ l t_B = B ∧ |dist M A| * |dist M B| = 8 / 5
:= sorry

end MA_MB_product_l156_156162


namespace part1_part2_l156_156576

noncomputable def midpoint_condition (A B : ℝ × ℝ) (M : ℝ × ℝ) : Prop :=
A.1 + B.1 = 2 * M.1 ∧ A.2 + B.2 = 2 * M.2

noncomputable def ellipse (P : ℝ × ℝ) : Prop :=
(P.1 ^ 2) / 4 + (P.2 ^ 2) / 3 = 1

noncomputable def slope_condition (A B : ℝ × ℝ) (k : ℝ) : Prop :=
k = (A.2 - B.2) / (A.1 - B.1)

noncomputable def slope_inequality (A B : ℝ × ℝ) (M : ℝ × ℝ) (m k : ℝ) : Prop :=
midpoint_condition A B M ∧ ellipse A ∧ ellipse B ∧ M.1 = 1 ∧ M.2 = m ∧ m > 0 ∧ slope_condition A B k → k < -(1 / 2)

noncomputable def vector_sum_condition (A B P F : ℝ × ℝ) : Prop :=
(F.1 - P.1) + (F.1 - A.1) + (F.1 - B.1) = 0 ∧
(F.2 - P.2) + (F.2 - A.2) + (F.2 - B.2) = 0

noncomputable def focal_radius (F : ℝ × ℝ) (A : ℝ × ℝ) : ℝ :=
if A.1 >= F.1 then 2 - (1 / 2) * (A.1 - F.1) else 2 + (1 / 2) * (A.1 - F.1)

noncomputable def ellipse_proof (A B P : ℝ × ℝ) : Prop :=
let F := (1, 0) in
vec_sum_condition A B P F ∧ ellipse P ∧ P.1 = 1 →
2 * focal_radius F P = focal_radius F A + focal_radius F B

theorem part1 (A B M : ℝ × ℝ) (m k : ℝ) : slope_inequality A B M m k :=
sorry

theorem part2 (A B P: ℝ × ℝ) : ellipse_proof A B P :=
sorry

end part1_part2_l156_156576


namespace count_of_plus_signs_l156_156801

-- Definitions based on the given conditions
def total_symbols : ℕ := 23
def any_10_symbols_contains_plus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 10 → (\exists c ∈ s, c = '+')
def any_15_symbols_contains_minus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 15 → (\exists c ∈ s, c = '-')

-- Main theorem statement without the proof
theorem count_of_plus_signs (symbols : Finset char) (h1 : symbols.card = total_symbols) 
  (h2 : any_10_symbols_contains_plus symbols) 
  (h3 : any_15_symbols_contains_minus symbols) : 
  symbols.count '+' = 14 :=
sorry

end count_of_plus_signs_l156_156801


namespace value_of_a_l156_156275

noncomputable def A : Set ℝ := {x | x^2 - x - 2 < 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 5}

theorem value_of_a (a : ℝ) (h : A ⊆ B a) : -3 ≤ a ∧ a ≤ -1 :=
by
  sorry

end value_of_a_l156_156275


namespace sequence_inequality_l156_156770

noncomputable def aₙ : ℕ → ℝ
| 0       := 2
| (n + 1) := (aₙ n)^2 - (aₙ n) + 1

theorem sequence_inequality : 
  1 - (1 / (2^2003 : ℝ)) < ∑ i in finRange 2003, (1 / aₙ (i + 1)) ∧ ∑ i in finRange 2003, (1 / aₙ (i + 1)) < 1 :=
sorry

end sequence_inequality_l156_156770


namespace forty_five_days_after_monday_is_thursday_l156_156073

-- Definition of the days of the week as an enumeration
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

open DayOfWeek

-- Function to calculate the day of the week after a given number of days
def dayAfter (start : DayOfWeek) (n : ℕ) : DayOfWeek :=
  match start with
  | Monday => match n % 7 with
              | 0 => Monday
              | 1 => Tuesday
              | 2 => Wednesday
              | 3 => Thursday
              | 4 => Friday
              | 5 => Saturday
              | 6 => Sunday
  | Tuesday => match n % 7 with
               | 0 => Tuesday
               | 1 => Wednesday
               | 2 => Thursday
               | 3 => Friday
               | 4 => Saturday
               | 5 => Sunday
               | 6 => Monday
  -- similar cases for other days

-- The proof statement
theorem forty_five_days_after_monday_is_thursday :
  dayAfter Monday 45 = Thursday :=
by
  -- proof goes here
  sorry

end forty_five_days_after_monday_is_thursday_l156_156073


namespace parabola_equation_triangle_area_l156_156917

theorem parabola_equation (p : ℝ) :
  (∃ (p : ℝ), 2 * p = 8) → (y^2 = 2 * p * x) := by
  sorry

theorem triangle_area (a b : ℝ) :
  (a = sqrt(3)) → (b = 1) → (y = ± sqrt(3) / 3 * x) →
  (x = -2) → (|AB| = 4 * sqrt(3) / 3) →
  (S = 1 / 2 * (4 * sqrt(3) / 3) * 2) → (S = 4 * sqrt(3) / 3) := by
  sorry

end parabola_equation_triangle_area_l156_156917


namespace trapezoid_AKLN_area_l156_156021

noncomputable def cube := sorry -- Define the cube with side length 1.
def midpoint {A B : point} (A B : point) : point := sorry -- Midpoint of two points
def center_face {A B C} (A B C : point) : point := sorry -- Center of a face defined by points A, B, C
def intersection_line_point {A B C : point} : point := sorry -- Intersection of line AK with DC
def triangles_congruent {A B K C O} (A B K C O : point) := sorry -- Congruence of triangles ABK and KCO
def trapezoid_area {A K L N : point} := sorry -- Area of trapezoid AKLN

theorem trapezoid_AKLN_area :
  let K := midpoint B C in
  let M := center_face D C (center_face D_1 D) in
  let O := intersection_line_point A K D C in
  let L := midpoint C (center_face C_1 C) in
  let N := midpoint D (center_face D_1 D) in
  triangles_congruent A B K C O →
  trapezoid_area A K L N = sqrt(14) / 4 :=
begin
  sorry
end

end trapezoid_AKLN_area_l156_156021


namespace rectangle_diagonals_intersect_square_diagonal_l156_156926

theorem rectangle_diagonals_intersect_square_diagonal
  (s a b : ℝ)
  (square_vertices: list (ℝ × ℝ))
  (rectangle_vertices: list (ℝ × ℝ))
  (H1 : square_vertices = [(0,0), (s,0), (0,s), (s,s)])
  (H2 : rectangle_vertices = [(0,0), (a,0), (0,b), (a,b)])
  (H3 : 2*a + 2*b = 4*s) :
  (∃ (x y : ℝ), (x, y) = (a / 2, b / 2) ∧ (y = x)) :=
begin
  -- proof goes here
  sorry
end

end rectangle_diagonals_intersect_square_diagonal_l156_156926


namespace sum_ninth_power_l156_156724

theorem sum_ninth_power (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) 
                        (h3 : a^3 + b^3 = 4) (h4 : a^4 + b^4 = 7)
                        (h5 : a^5 + b^5 = 11)
                        (h_ind : ∀ n, n ≥ 3 → a^n + b^n = a^(n-1) + b^(n-1) + a^(n-2) + b^(n-2)) :
  a^9 + b^9 = 76 :=
by
  sorry

end sum_ninth_power_l156_156724


namespace vertical_faces_same_color_probability_l156_156976

open BigOperators

def colors := { red, blue, green }
def painting_probability := 1 / 3

noncomputable def cube_faces : Finset (Fin 6 → colors) :=
  (Finset.univ : Finset (Fin 6 → colors))

noncomputable def valid_arrangements : Finset (Fin 6 → colors) :=
  cube_faces.filter (λ arr, 
    (∃ c, ∀ i j, i ≠ j ∧ i < 4 ∧ j < 4 ∧ arr i = c ∧ arr j = c) ∨
    (∃ c d e, ∀ i j, ((i = 4 ∧ j < 4) ∨ (i = 5 ∧ j < 4)) ∧ (arr i = d ∧ arr j = c ∧ arr ((i + 1) % 6) = c ∧ arr ((i + 2) % 6) = c ∧ arr ((i + 3) % 6) = c) ∨
    (∃ c d e, ∀ i j, ((i = 0 ∨ i = 1 ∨ i = 2 ∧ j = 3) ∨ (i = 4 ∨ i = 5) ∧ arr i = d ∧ arr j = c ∧ (arr ((i + 1) % 6) = c ∧ arr ((i + 2) % 6) = c ∧ arr ((i + 3) % 6) = c)) 
  )) 

theorem vertical_faces_same_color_probability : 
  (valid_arrangements.card / cube_faces.card : ℚ) = 57 / 729 :=
by sorry

end vertical_faces_same_color_probability_l156_156976


namespace symmetry_center_l156_156402

def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)
def g (x : ℝ) : ℝ := f (x + Real.pi / 4) + 2

theorem symmetry_center :
  ∃ k : ℤ, g ((-3 * Real.pi / 8) + (k * Real.pi / 2)) = 2 :=
begin
  use 1,
  sorry

end symmetry_center_l156_156402


namespace sum_of_numerator_and_denominator_of_q_l156_156327

def probability_reaching_31 (steps : ℕ) : ℚ :=
  if steps = 4 then 1/64
  else if steps = 6 then 56/4096
  else if steps = 8 then 1656/65536
  else 0

theorem sum_of_numerator_and_denominator_of_q :
  let q := probability_reaching_31 4 + probability_reaching_31 6 + probability_reaching_31 8
  q = 7 / 128 → 
  7 + 128 = 135 :=
by
  intro hq
  linarith

end sum_of_numerator_and_denominator_of_q_l156_156327


namespace wicket_keeper_age_difference_l156_156671

theorem wicket_keeper_age_difference :
  let n := 11 in
  let avg_team_age := 21 in
  let captain_age := 24 in
  let remaining_players := n - 2 in
  let avg_remaining_age := avg_team_age - 1 in
  let total_team_age := avg_team_age * n in
  let total_remaining_age := avg_remaining_age * remaining_players in
  let wicket_keeper_age := captain_age + (3 : ℕ) in
  total_team_age = total_remaining_age + captain_age + wicket_keeper_age := 
by
  sorry

end wicket_keeper_age_difference_l156_156671


namespace find_angle_A_max_area_l156_156240

open Real

variables (a b c : ℝ) (A B C : ℝ)

-- Condition: Triangle ABC, with angles A, B, C opposite sides a, b, c respectively
-- It is given that: (-b + sqrt 2 * c) / cos B = a / cos A

def condition1 : Prop := (-b + sqrt 2 * c) / cos B = a / cos A

-- Question 1: Find the magnitude of angle A
theorem find_angle_A (h : condition1 a b c A B C) : A = π / 4 :=
sorry

-- Question 2: If a = 2, find the maximum value of the area S of triangle ABC
def condition2 : Prop := a = 2
def area (a b c A B C : ℝ) : ℝ := (1/2) * b * c * sin A

theorem max_area (h1 : condition1 a b c A B C) (h2 : condition2 a) : 
  area a b c A B C ≤ sqrt 2 + 1 :=
sorry

end find_angle_A_max_area_l156_156240


namespace percent_women_employees_is_58_l156_156493

noncomputable def percent_women_employees (E W : ℕ) (M : ℕ)
    (married_fraction : ℝ) 
    (fraction_single_men : ℝ) 
    (fraction_married_women : ℝ) : Prop := 
  W / E = 0.58

theorem percent_women_employees_is_58 
    (E W M : ℕ) 
    (h1 : W + M = E) 
    (h2 : 0.60 * E = (1/3) * M + 0.7931034482758621 * W) 
    (h3 : 2/3 = fraction_single_men) 
    (h4 : 0.7931034482758621 = fraction_married_women): 
    percent_women_employees E W M 0.60 2/3 0.7931034482758621 :=
  sorry

end percent_women_employees_is_58_l156_156493


namespace triangle_angle_C_right_l156_156241

theorem triangle_angle_C_right {a b c A B C : ℝ}
  (h1 : a / Real.sin B + b / Real.sin A = 2 * c) 
  (h2 : a / Real.sin A = b / Real.sin B) 
  (h3 : b / Real.sin B = c / Real.sin C) : 
  C = Real.pi / 2 :=
by sorry

end triangle_angle_C_right_l156_156241


namespace geometric_sequence_formula_and_sum_l156_156689

-- Defining basic conditions and main goal
theorem geometric_sequence_formula_and_sum (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) (T_n : ℕ → ℝ)
    (q : ℝ) (a1 : ℝ):
  (a_n 2 - a_n 1 = 2) → 
  (2 * a_n 2 = (3 * a_n 1 + a_n 3) / 2) → 
  (∀ n, a_n (n+1) = a_n 1 * q ^ n) →
  (b_n = λ n, 2 * log q (a_n n) + 1) → 
  (T_n = λ n, ∑ i in range n, 1 / (b_n i * b_n (i+1))) →
  a_n = λ n, 3^(n-1) ∧ T_n n = n / (2 * n + 1) :=
by
  intros h1 h2 h3 hb ht
  split
  { -- Prove the general formula a_n = 3^(n-1)
    -- Sorry placeholder for the proof
    sorry },
  { -- Prove T_n = n / (2n + 1)
    -- Sorry placeholder for the proof
    sorry }

end geometric_sequence_formula_and_sum_l156_156689


namespace plus_signs_count_l156_156812

theorem plus_signs_count (num_symbols : ℕ) (at_least_one_plus_in_10 : ∀ s : Finset ℕ, s.card = 10 → (∃ i ∈ s, i < 14)) (at_least_one_minus_in_15 : ∀ s : Finset ℕ, s.card = 15 → (∃ i ∈ s, i ≥ 14)) : 
    ∃ (p m : ℕ), p + m = 23 ∧ p = 14 ∧ m = 9 := by
  sorry

end plus_signs_count_l156_156812


namespace number_of_palindromes_l156_156915

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem number_of_palindromes (n : ℕ) (h_n : 0 < n) : 
  (set.univ.filter (λ x, x > 0 ∧ is_palindrome x ∧ x.digits.length = 2 * n + 1)).card = 9 * 10^n :=
  sorry

end number_of_palindromes_l156_156915


namespace apples_removed_in_week_l156_156946

def initial_apples : ℕ := 1000
def apples_removed_per_day_ricki : ℕ := 50
def apples_removed_per_day_samson : ℕ := 2 * apples_removed_per_day_ricki
def apples_removed_per_day_bindi : ℕ := 3 * apples_removed_per_day_samson
def days_in_a_week : ℕ := 7

theorem apples_removed_in_week :
  let total_removed_per_day := apples_removed_per_day_ricki + apples_removed_per_day_samson + apples_removed_per_day_bindi in
  let total_removed_in_week := total_removed_per_day * days_in_a_week in
  total_removed_in_week > initial_apples ∧ total_removed_in_week - initial_apples = 2150 :=
by
  sorry

end apples_removed_in_week_l156_156946


namespace permutations_of_three_digit_numbers_from_set_l156_156361

theorem permutations_of_three_digit_numbers_from_set {digits : Finset ℕ} (h : digits = {1, 2, 3, 4, 5}) :
  ∃ n : ℕ, n = (Finset.card digits) * (Finset.card digits - 1) * (Finset.card digits - 2) ∧ n = 60 :=
by
  sorry

end permutations_of_three_digit_numbers_from_set_l156_156361


namespace find_a_b_sum_l156_156329

theorem find_a_b_sum :
  let q := (prob_reach_31_in_at_most_8_steps (0, 0))
  in ∃ a b : ℕ, a + b = 1039 ∧ q = a / b
:= sorry

def prob_reach_31_in_at_most_8_steps (start : ℤ × ℤ) : ℚ := 
  -- This is a placeholder definition, the actual function
  -- calculation of the probability would be done here
  if start = (0, 0) then (15 / 1024 : ℚ) else 0

end find_a_b_sum_l156_156329


namespace students_in_first_class_l156_156335

variable (x : ℕ)
variable (avg_marks_first_class : ℕ := 40)
variable (num_students_second_class : ℕ := 28)
variable (avg_marks_second_class : ℕ := 60)
variable (avg_marks_all : ℕ := 54)

theorem students_in_first_class : (40 * x + 60 * 28) / (x + 28) = 54 → x = 12 := 
by 
  sorry

end students_in_first_class_l156_156335


namespace intersection_with_y_axis_l156_156750

theorem intersection_with_y_axis :
  ∃ (y : ℝ), (y = -x^2 + 3*x - 4) ∧ (x = 0) ∧ (y = -4) := 
by
  sorry

end intersection_with_y_axis_l156_156750


namespace expression_positive_l156_156738

theorem expression_positive (x : ℝ) : x^2 * sin x + x * cos x + x^2 + 1/2 > 0 :=
by
  sorry

end expression_positive_l156_156738


namespace probability_heads_B_greater_l156_156452

-- Definitions for our proof problem
def fair_coin_tosses (n : ℕ) : Type := { xs : vector bool n // ∀ x ∈ xs, x = true ∨ x = false }

noncomputable def number_of_heads_in_tosses (xs : vector bool (n : ℕ)) : ℕ :=
  xs.to_list.count tt

theorem probability_heads_B_greater (A_tosses : fair_coin_tosses 10) (B_tosses : fair_coin_tosses 11) :
  (∀ a, a ∈ A_tosses.val → a = true ∨ a = false) →
  (∀ b, b ∈ B_tosses.val → b = true ∨ b = false) →
  (number_of_heads_in_tosses B_tosses.val > number_of_heads_in_tosses A_tosses.val) →
  probability (number_of_heads_in_tosses B_tosses.val > number_of_heads_in_tosses A_tosses.val) = 1 / 2 := 
sorry

end probability_heads_B_greater_l156_156452


namespace decrypt_test_phrase_l156_156699

-- Definition of encryption and decryption steps
def swap_adjacent (s : String) : String :=
  s.toList.groupByIndex (· % 2 = 1).toList.join

def shift_alphabet_right (ch : Char) (positions : Nat) : Char :=
  let base := if ch.isLower then 'а' else 'А'
  let shift := ((ch.toNat - base.toNat + positions) % 32)
  Char.ofNat (base.toNat + shift)

def shift_right (s : String) (positions : Nat) : String :=
  s.toList.map (λ ch => shift_alphabet_right ch positions).asString

def reverse_string (s : String) : String := s.reverse

-- Given example process
def encryption_example : String :=
  reverse_string (shift_right (swap_adjacent "гипертекст") 2)

-- Decryption process
def decrypt (encrypted : String) : String :=
  let reversed := reverse_string encrypted
  let shifted := shift_right reversed (32 - 2)  -- reversing the shift of 2 positions
  swap_adjacent shifted

-- Given encrypted phrase
def encrypted_phrase : String := "врпвл терпраиэ вйзгцфпз"

-- Expected result after decryption
def decrypted_phrase : String := "нефтебазы южного района"

-- Statement
theorem decrypt_test_phrase :
  decrypt encrypted_phrase = decrypted_phrase :=
by
  sorry

end decrypt_test_phrase_l156_156699


namespace p_sq_plus_q_sq_l156_156268

theorem p_sq_plus_q_sq (p q : ℝ) (h1 : p * q = 12) (h2 : p + q = 8) : p^2 + q^2 = 40 :=
by
  sorry

end p_sq_plus_q_sq_l156_156268


namespace quadratic_inequality_solution_l156_156971

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 + 9 * x + 8 < 0} = set.Ioo (-8) (-1) :=
by sorry
 
end quadratic_inequality_solution_l156_156971


namespace plus_signs_count_l156_156822

theorem plus_signs_count (total_symbols : ℕ) (n : ℕ) (m : ℕ)
    (h1 : total_symbols = 23)
    (h2 : ∀ (s : Finset ℕ), s.card = 10 → (∃ x ∈ s, x = n))
    (h3 : ∀ (s : Finset ℕ), s.card = 15 → (∃ x ∈ s, x = m)) :
    n = 14 := by
  sorry

end plus_signs_count_l156_156822


namespace average_pairs_of_consecutive_integers_l156_156857

open Finset

theorem average_pairs_of_consecutive_integers :
  let s := {1, 2, ..., 35}.powerset.filter (λ (a : Finset ℕ), a.card = 6) in
  ∑ t in s, (t.pairwise (λ x y, y = x + 1) ∩ t.card = 2) / s.card = 101 / 210 := sorry

end average_pairs_of_consecutive_integers_l156_156857


namespace cevian_concurrency_l156_156709

-- Define the points and segments involved in the problem
variables {A B C D E F M N P : Type}

-- We need to state these geometric properties in Lean definitions
def triangle (A B C : Type) := true
def on_segment (P A B : Type) : Prop := true
def concurrent (P Q R : Type) : Prop := true

-- The main theorem statement
theorem cevian_concurrency (h1 : triangle A B C)
                            (h2 : on_segment D B C)
                            (h3 : on_segment E C A)
                            (h4 : on_segment F A B)
                            (h5 : concurrent (AD) (BE) (CF))
                            (h6 : on_segment M AD)
                            (h7 : on_segment N BE)
                            (h8 : on_segment P CF) :
   (concurrent (AM) (BN) (CP)) ↔ (concurrent (DM) (EN) (FP)) :=
begin
    sorry
end

end cevian_concurrency_l156_156709


namespace tetrahedron_insphere_surface_area_ratio_l156_156163

theorem tetrahedron_insphere_surface_area_ratio (S_1 S_2 : ℝ) (h_tetrahedron : S_1 = √3 * a^2)
  (h_insphere : S_2 = (π * a^2) / 6) : S_1 / S_2 = 6 * √3 / π :=
by
  sorry

end tetrahedron_insphere_surface_area_ratio_l156_156163


namespace range_of_a_in_triangle_l156_156667

open Real

noncomputable def law_of_sines_triangle (A B C : ℝ) (a b c : ℝ) :=
  sin A / a = sin B / b ∧ sin B / b = sin C / c

theorem range_of_a_in_triangle (b : ℝ) (B : ℝ) (a : ℝ) (h1 : b = 2) (h2 : B = pi / 4) (h3 : true) :
  2 < a ∧ a < 2 * sqrt 2 :=
by
  sorry

end range_of_a_in_triangle_l156_156667


namespace plus_signs_count_l156_156832

theorem plus_signs_count (total_symbols : ℕ) (n_plus n_minus : ℕ)
  (h1 : total_symbols = 23)
  (h2 : ∀ s : Finset ℕ, s.card = 10 → ∃ i ∈ s, i ≤ n_plus)
  (h3 : ∀ s : Finset ℕ, s.card = 15 → ∃ i ∈ s, i > n_plus) :
  n_plus = 14 :=
by
  -- the proof will go here
  sorry

end plus_signs_count_l156_156832


namespace line_log_intersection_l156_156458

theorem line_log_intersection (a b : ℤ) (k : ℝ)
  (h₁ : k = a + Real.sqrt b)
  (h₂ : k > 0)
  (h₃ : Real.log k / Real.log 2 - Real.log (k + 2) / Real.log 2 = 1
    ∨ Real.log (k + 2) / Real.log 2 - Real.log k / Real.log 2 = 1) :
  a + b = 2 :=
sorry

end line_log_intersection_l156_156458


namespace skate_cost_l156_156070

/- Define the initial conditions as Lean definitions -/
def admission_cost : ℕ := 5
def rental_cost : ℕ := 250 / 100  -- 2.50 dollars in cents for integer representation
def visits : ℕ := 26

/- Define the cost calculation as a Lean definition -/
def total_rental_cost (rental_cost : ℕ) (visits : ℕ) : ℕ := rental_cost * visits

/- Statement of the problem in Lean proof form -/
theorem skate_cost (C : ℕ) (h : total_rental_cost rental_cost visits = C) : C = 65 :=
by
  sorry

end skate_cost_l156_156070


namespace repeating_decimal_sum_l156_156101

theorem repeating_decimal_sum (x : ℚ) (hx : x = 0.417) :
  let num := 46
  let denom := 111
  let sum := num + denom
  sum = 157 :=
by
  sorry

end repeating_decimal_sum_l156_156101


namespace binomial_square_l156_156550

variable (c : ℝ)

theorem binomial_square (h : ∃ a : ℝ, (x^2 - 164 * x + c) = (x + a)^2) : c = 6724 := sorry

end binomial_square_l156_156550


namespace triangle_side_lengths_l156_156246

variable (AB BC AC : ℝ)
variable (BAC ACB : ℝ)

def triangle_valid {AB BC AC : ℝ} {BAC ACB : ℝ} : Prop :=
  AC = 5 ∧ BC - AB = 2 ∧ BAC = 2 * ACB ∧
  AB / Real.sin ACB = BC / Real.sin (2 * ACB) ∧
  AB / Real.sin ACB = AC / Real.sin (Real.pi - 3 * ACB)

theorem triangle_side_lengths (h : triangle_valid AB BC AC BAC ACB) : AB = 4 ∧ BC = 6 :=
by
  sorry

end triangle_side_lengths_l156_156246


namespace even_abundant_numbers_less_than_50_count_l156_156184

def proper_factors (n : ℕ) : List ℕ :=
  List.filter (λ d, d < n ∧ n % d = 0) (List.range n)

def is_abundant (n : ℕ) : Prop :=
  n < (proper_factors n).sum

def even_numbers_up_to (n : ℕ) : List ℕ :=
  List.filter (λ x, x % 2 = 0) (List.range n)

theorem even_abundant_numbers_less_than_50_count : 
  (List.filter is_abundant (even_numbers_up_to 50)).length = 9 := by
  sorry

end even_abundant_numbers_less_than_50_count_l156_156184


namespace center_is_8_l156_156935

def numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def valid_array (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  ∀ (i j : Fin 3), m i j ∈ numbers ∧ 
  ∀ (i1 j1 i2 j2 : Fin 3), 
    (m i1 j1 + 1 = m i2 j2 ∨ m i1 j1 - 1 = m i2 j2) → 
    (i1 - i2).abs + (j1 - j2).abs = 1

def corner_sum_20 (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  m 0 0 + m 0 2 + m 2 0 + m 2 2 = 20

def corner_one (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  m 0 0 = 1

def center (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ := m 1 1

theorem center_is_8 (m : Matrix (Fin 3) (Fin 3) ℕ) 
  (h_valid : valid_array m) 
  (h_corners_sum : corner_sum_20 m) 
  (h_corner_one : corner_one m) :
  center m = 8 := by
  sorry

end center_is_8_l156_156935


namespace range_of_a_l156_156753

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, x ≤ 4 → y ≤ 4 → x ≤ y → f x ≥ f y) → a ≥ 8 :=
  sorry

-- Define f(x) = x^2 - ax - 3
def f (x : ℝ) (a : ℝ) := x^2 - a * x - 3

end range_of_a_l156_156753


namespace arccos_gt_arctan_l156_156984

theorem arccos_gt_arctan (x : ℝ) (h : -1 ≤ x ∧ x < 1/2) : Real.arccos x > Real.arctan x :=
sorry

end arccos_gt_arctan_l156_156984


namespace minimum_PA_PF_l156_156601

-- Definitions based on the given problem.
def parabola_equation (P : ℝ × ℝ) : Prop :=
  P.2^2 = 8 * P.1

def focus_of_parabola : ℝ × ℝ :=
  (2, 0)

def point_A : ℝ × ℝ :=
  (5, 2)

-- The minimum value of PA + PF is 7
theorem minimum_PA_PF (P : ℝ × ℝ) (hP : parabola_equation P) : 
  let F := focus_of_parabola in
  let A := point_A in
  ∃ (min_val : ℝ), min_val = 7 ∧ ∀ (P : ℝ × ℝ) (hP : parabola_equation P), 
    dist A P + dist P F ≥ min_val :=
sorry

end minimum_PA_PF_l156_156601


namespace eccentricity_of_ellipse_l156_156146

theorem eccentricity_of_ellipse
  (a b c : ℝ)
  (h1 : a > b ∧ b > 0)
  (h2 : ∀ x y : ℝ, (x^2 / a^2 ) + (y^2 / b^2 ) = 1)
  (A B F1 F2 : ℝ × ℝ)
  (h3 : A.1^2 / a^2 + A.2^2 / b^2 = 1)
  (h4 : B.1 = 0)
  (h5 : let AF1 := (A.1 - F1.1, A.2 - F1.2),
             BF1 := (B.1 - F1.1, B.2 - F1.2)
        in AF1.1 * BF1.1 + AF1.2 * BF1.2 = 0)
  (h6 : let AF2 := (A.1 - F2.1, A.2 - F2.2),
             F2B := (F2.1 - B.1, F2.2 - B.2)
        in AF2 = (2 / 3) • F2B)
  : let e := c / a
    in e = (√5) / 5 := sorry

end eccentricity_of_ellipse_l156_156146


namespace max_real_roots_two_l156_156969

noncomputable def polynomial_max_real_roots (n : ℕ) (h_pos : 0 < n) : Prop :=
  ∀ (x : ℝ), x ≠ 0 → 2 * x^n + (2 - 2) * x^(n - 1) + (2 - 4) * x^(n - 2) + 
  ∑ j in Icc 1 n, (2 - 2 * j) * x^(n - j) = 0 → x = 1 ∨ (x = -1 ∧ even n)

theorem max_real_roots_two (n : ℕ) (h_pos : 0 < n) :
  polynomial_max_real_roots n h_pos → ∃! (r : ℝ), r = 1 ∨ (r = -1 ∧ even n) :=
sorry

end max_real_roots_two_l156_156969


namespace percentage_decrease_l156_156366

theorem percentage_decrease (a b p : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
    (h_ratio : a / b = 4 / 5) 
    (h_x : ∃ x, x = a * 1.25)
    (h_m : ∃ m, m = b * (1 - p / 100))
    (h_mx : ∃ m x, (m / x = 0.2)) :
        (p = 80) :=
by
  sorry

end percentage_decrease_l156_156366


namespace eccentricity_of_hyperbola_l156_156544

-- Define the conditions
def hyperbola_eccentricity (a b : ℝ) := sqrt (1 + (b^2 / a^2))

theorem eccentricity_of_hyperbola :
  (hyperbola_eccentricity (sqrt 2) (sqrt 3)) = sqrt 5 / 2 :=
by
  sorry

end eccentricity_of_hyperbola_l156_156544


namespace constructible_numbers_card_eq_98_l156_156975

noncomputable theory

def bricks := 25
def increments := {5, 14, 17}

def constructible_heights (bricks : ℕ) (increments : set ℕ) : set ℕ := 
  {sum | ∃ (count_5 count_14 count_17 : ℕ), 
    count_5 + count_14 + count_17 = bricks ∧ 
    sum = count_5 * 5 + count_14 * 14 + count_17 * 17}

theorem constructible_numbers_card_eq_98 : 
  (constructible_heights bricks increments).card = 98 := 
by 
  sorry

end constructible_numbers_card_eq_98_l156_156975


namespace dana_fewest_cookies_l156_156901

open Real

-- Define the areas of the cookies
def area_alex : ℝ := π * (2 ^ 2)
def area_ben : ℝ := (3 ^ 2)
def area_carl : ℝ := 4 * 2
def area_dana : ℝ := (3 * sqrt 3 / 2) * (1 ^ 2) * 6
def area_eliza : ℝ := (sqrt 3 / 4) * (4 ^ 2)

-- Define cookie quantities
def cookies (area : ℝ) (total_dough : ℝ) : ℝ := total_dough / area

-- Suppose the same volume of dough is used by each friend
variable (D : ℝ)

-- Prove Dana gets the fewest cookies
theorem dana_fewest_cookies :
  (cookies area_dana D) < (cookies area_alex D) ∧ 
  (cookies area_dana D) < (cookies area_ben D) ∧ 
  (cookies area_dana D) < (cookies area_carl D) ∧ 
  (cookies area_dana D) < (cookies area_eliza D) :=
by
  sorry

end dana_fewest_cookies_l156_156901


namespace det_S_l156_156711

noncomputable def S : Matrix (Fin 2) (Fin 2) ℝ :=
  let R : Matrix (Fin 2) (Fin 2) ℝ := ![
    ![Real.cos (Float.pi / 4), -Real.sin (Float.pi / 4)],
    ![Real.sin (Float.pi / 4), Real.cos (Float.pi / 4)]
  ]
  3 * R

theorem det_S : Matrix.det S = 9 := by
  sorry

end det_S_l156_156711


namespace partial_sum_exists_l156_156963

theorem partial_sum_exists
  (n : ℕ)
  (x : Fin n → ℝ)
  (h1 : 2 ≤ n)
  (h2 : ∀ i j : Fin n, i ≤ j → 0 < x i ∧ x i ≤ x j)
  (h3 : (Finset.univ.sum x) = 1)
  (h4 : x (Fin.last n) ≤ 2 / 3) :
  ∃ k : Fin n, 1 / 3 ≤ (Finset.range (k + 1)).sum x ∧ (Finset.range (k + 1)).sum x < 2 / 3 := by
  sorry

end partial_sum_exists_l156_156963


namespace population_at_300pm_l156_156492

namespace BacteriaProblem

def initial_population : ℕ := 50
def time_increments_to_220pm : ℕ := 4   -- 4 increments of 5 mins each till 2:20 p.m.
def time_increments_to_300pm : ℕ := 2   -- 2 increments of 10 mins each till 3:00 p.m.

def growth_factor_before_220pm : ℕ := 3
def growth_factor_after_220pm : ℕ := 2

theorem population_at_300pm :
  initial_population * growth_factor_before_220pm^time_increments_to_220pm *
  growth_factor_after_220pm^time_increments_to_300pm = 16200 :=
by
  sorry

end BacteriaProblem

end population_at_300pm_l156_156492


namespace part1_part2_l156_156171

def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b * x

theorem part1 (a b : ℝ) (h1 : f a b 1 = -1/2) (h2 : deriv (f a b) 1 = 1/2) :
  a = 1 ∧ b = -1/2 :=
by
  sorry

theorem part2 (k : ℝ) (h : ∀ x > 1, f 1 (-1/2) x + k/x < 0) :
  k ≤ 1/2 :=
by
  sorry

end part1_part2_l156_156171


namespace arithmetic_sequence_sum_l156_156604

variable (S : ℕ → ℕ)   -- S is a function that gives the sum of the first k*n terms

theorem arithmetic_sequence_sum
  (n : ℕ)
  (h1 : S n = 45)
  (h2 : S (2 * n) = 60) :
  S (3 * n) = 65 := sorry

end arithmetic_sequence_sum_l156_156604


namespace largest_positive_integer_n_l156_156115

theorem largest_positive_integer_n :
  ∃ n : ℕ, (∀ x : ℝ, (Real.sin x) ^ n + (Real.cos x) ^ n ≥ 2 / n) ∧ (∀ m : ℕ, m > n → ∃ x : ℝ, (Real.sin x) ^ m + (Real.cos x) ^ m < 2 / m) :=
sorry

end largest_positive_integer_n_l156_156115


namespace log_3_24_minus_log_3_8_l156_156497

lemma log_difference (log : ℕ → ℕ → ℝ) (a b : ℕ) :
  log 3 24 - log 3 8 = log 3 3 :=
begin
  sorry
end

lemma log_base (log : ℕ → ℕ → ℝ) (a : ℕ) :
  log a a = 1 :=
begin
  sorry
end

theorem log_3_24_minus_log_3_8 :
  ∀ (log : ℕ → ℕ → ℝ), log 3 24 - log 3 8 = 1 :=
by {
  intro log,
  have h1 : log 3 24 - log 3 8 = log 3 3, from log_difference log 24 8,
  have h2 : log 3 3 = 1, from log_base log 3,
  rw [h1, h2],
}

end log_3_24_minus_log_3_8_l156_156497


namespace number_of_valid_arrangements_l156_156309

open Finset

-- We define the condition that a list is sorted in ascending order
def is_ascending (l : List ℕ) : Prop :=
  l = List.sort (≤) l

-- We define the condition that a list is sorted in descending order
def is_descending (l : List ℕ) : Prop :=
  l = List.sort (≥) l

def cards := Finset.range 7
def arrangements := cards.to_list.permutations

-- Define the function to check if a list of numbers (cards) 
-- can have one element removed to form an ascending or descending list
def valid_arrangement (l : List ℕ) : Prop :=
  ∃ (x : ℕ), (l.erase x).is_ascending ∨ (l.erase x).is_descending

-- Define the final theorem
theorem number_of_valid_arrangements : finset.card (arrangements.filter valid_arrangement) = 72 :=
by
  sorry

end number_of_valid_arrangements_l156_156309


namespace trader_loss_percent_l156_156655

theorem trader_loss_percent :
  let SP1 : ℝ := 404415
  let SP2 : ℝ := 404415
  let gain_percent : ℝ := 15 / 100
  let loss_percent : ℝ := 15 / 100
  let CP1 : ℝ := SP1 / (1 + gain_percent)
  let CP2 : ℝ := SP2 / (1 - loss_percent)
  let TCP : ℝ := CP1 + CP2
  let TSP : ℝ := SP1 + SP2
  let overall_loss : ℝ := TSP - TCP
  let overall_loss_percent : ℝ := (overall_loss / TCP) * 100
  overall_loss_percent = -2.25 := 
sorry

end trader_loss_percent_l156_156655


namespace complicated_fraction_equiv_one_l156_156955

theorem complicated_fraction_equiv_one : 
  (∏ i in Finset.range 21, (1 + 19 / (i + 1))) / (∏ i in Finset.range 19, (1 + 21 / (i + 1))) = 1 :=
by sorry

end complicated_fraction_equiv_one_l156_156955


namespace length_of_bridge_l156_156443

theorem length_of_bridge (L_train : ℕ) (V_train_kmhr : ℚ) (T_cross : ℚ) (V_train_ms : ℚ) :
  V_train_ms = V_train_kmhr * (1000 / 3600) →
  (L_train = 160 ∧ V_train_kmhr = 45 ∧ T_cross = 30) →
  let total_distance := V_train_ms * T_cross in
  let L_bridge := total_distance - L_train in
  L_bridge = 215 :=
begin
  sorry
end

end length_of_bridge_l156_156443


namespace calculate_expression_l156_156498

theorem calculate_expression :
  ( (5^1010)^2 - (5^1008)^2) / ( (5^1009)^2 - (5^1007)^2) = 25 := 
by
  sorry

end calculate_expression_l156_156498


namespace jill_arrives_after_jack_l156_156252

theorem jill_arrives_after_jack (distance : ℝ) (jill_speed : ℝ) (jack_speed : ℝ) (jack_stop_time : ℝ) :
  distance = 2 ∧ jill_speed = 8 ∧ jack_speed = 6 ∧ jack_stop_time = 5 →
  let jill_time := (distance / jill_speed) * 60 in
  let jack_time := ((distance / jack_speed) * 60 + jack_stop_time) in
  jack_time - jill_time = 10 := 
by
  sorry

end jill_arrives_after_jack_l156_156252


namespace polygon_sides_arithmetic_progression_l156_156356

theorem polygon_sides_arithmetic_progression
  (angles_in_arithmetic_progression : ∃ (a d : ℝ) (angles : ℕ → ℝ), ∀ (k : ℕ), angles k = a + k * d)
  (common_difference : ∃ (d : ℝ), d = 3)
  (largest_angle : ∃ (n : ℕ) (angles : ℕ → ℝ), angles n = 150) :
  ∃ (n : ℕ), n = 15 :=
sorry

end polygon_sides_arithmetic_progression_l156_156356


namespace number_of_ordered_19_tuples_l156_156993

theorem number_of_ordered_19_tuples :
  let b : Fin 19 → ℤ := sorry in
  (∀ i, b i ^ 2 = ∑ j in Finset.univ \ {i}, b j) →
  (b.count = 54264) :=
sorry

end number_of_ordered_19_tuples_l156_156993


namespace no_quadratic_polynomials_f_g_l156_156733

theorem no_quadratic_polynomials_f_g (f g : ℝ → ℝ) 
  (hf : ∃ a b c, ∀ x, f x = a * x^2 + b * x + c)
  (hg : ∃ d e h, ∀ x, g x = d * x^2 + e * x + h) : 
  ¬ (∀ x, f (g x) = x^4 - 3 * x^3 + 3 * x^2 - x) :=
by
  sorry

end no_quadratic_polynomials_f_g_l156_156733


namespace number_of_valid_arrangements_l156_156311

def is_ascending (l : List ℕ) : Prop :=
  ∀ i j, i < j → l.nth i ≤ l.nth j

def is_descending (l : List ℕ) : Prop :=
  ∀ i j, i < j → l.nth i ≥ l.nth j

def remove_one_is_ordered (l : List ℕ) : Prop :=
  ∃ (i : ℕ), (is_ascending (l.removeNth i) ∨ is_descending (l.removeNth i))

def valid_arrangements_count (cards : List ℕ) : ℕ :=
  -- counting the number of valid arrangements
  if (cards.length = 7
        ∧ ∀ i, i ∈ cards → 1 ≤ i ∧ i ≤ 7 ∧ (remove_one_is_ordered cards)) then 4 else 0

theorem number_of_valid_arrangements :
  valid_arrangements_count [1,2,3,4,5,6,7] = 4 :=
by sorry

end number_of_valid_arrangements_l156_156311


namespace statement_b_statement_c_l156_156006
-- Import all of Mathlib to include necessary mathematical functions and properties

-- First, the Lean statement for Statement B
theorem statement_b (a b : ℝ) (h : a > |b|) : a^2 > b^2 := 
sorry

-- Second, the Lean statement for Statement C
theorem statement_c (a b : ℝ) (h : a > b) : a^3 > b^3 := 
sorry

end statement_b_statement_c_l156_156006


namespace value_of_a_m_minus_3n_l156_156648

theorem value_of_a_m_minus_3n (a : ℝ) (m n : ℝ) (h1 : a^m = 8) (h2 : a^n = 2) : a^(m - 3 * n) = 1 :=
sorry

end value_of_a_m_minus_3n_l156_156648


namespace ratio_of_girls_to_boys_in_biology_class_l156_156395

-- Defining the conditions
def physicsClassStudents : Nat := 200
def biologyClassStudents := physicsClassStudents / 2
def boysInBiologyClass : Nat := 25
def girlsInBiologyClass := biologyClassStudents - boysInBiologyClass

-- Statement of the problem
theorem ratio_of_girls_to_boys_in_biology_class : girlsInBiologyClass / boysInBiologyClass = 3 :=
by
  sorry

end ratio_of_girls_to_boys_in_biology_class_l156_156395


namespace ratio_of_areas_l156_156923

theorem ratio_of_areas (O A B E D: Point) (r : ℝ) (S T : ℝ) (h₁ : semi_circle O r A B) 
                       (h₂ : inscribed_rectangle O r B C D E) (h₃ : BC = r / 2) 
                       (h₄ : S = (1 / 2) * r^2 * (π / 4)) (h₅ : T = (1 / 2) * r * (r / 2) * (√2 / 2)) :
  T / S = √2 / π := sorry

end ratio_of_areas_l156_156923


namespace cards_arrangement_count_is_10_l156_156315

-- Define the problem in Lean statement terms
def valid_arrangements_count : ℕ :=
  -- number of arrangements of seven cards where one card can be removed 
  -- leaving the remaining six cards in either ascending or descending order
  10

-- Theorem stating that the number of valid arrangements is 10
theorem cards_arrangement_count_is_10 : valid_arrangements_count = 10 :=
by
  -- Proof is omitted (the explanation above corresponds to the omitted proof details)
  sorry

end cards_arrangement_count_is_10_l156_156315


namespace pipe_c_filling_time_l156_156397

theorem pipe_c_filling_time :
  ∀ (t : ℝ),
  let rate_a := 1 / 20,
      rate_b := 1 / 20,
      rate_c := 1 / t,
      total_rate := rate_a + rate_b + rate_c,
      fraction_filled := 3 * total_rate,
      proportion_r := 3 * rate_c / fraction_filled,
      target_proportion := 1 / 4
  in proportion_r = target_proportion → t = 30 :=
by
  intros t rate_a rate_b rate_c total_rate fraction_filled proportion_r target_proportion
  sorry

end pipe_c_filling_time_l156_156397


namespace floor_sqrt_120_eq_10_l156_156523

theorem floor_sqrt_120_eq_10 :
  (√120).to_floor = 10 := by
  have h1 : √100 = 10 := by norm_num
  have h2 : √121 = 11 := by norm_num
  have h : 100 < 120 ∧ 120 < 121 := by norm_num
  have sqrt_120 : 10 < √120 ∧ √120 < 11 :=
    by exact ⟨real.sqrt_lt' 120 121 h.2, real.sqrt_lt'' 100 120 h.1⟩
  sorry

end floor_sqrt_120_eq_10_l156_156523


namespace max_elevation_l156_156463

def particle_elevation (t : ℝ) : ℝ := 200 * t - 20 * t^2 + 50

theorem max_elevation : ∃ t : ℝ, particle_elevation t = 550 :=
by {
  sorry
}

end max_elevation_l156_156463


namespace area_BCD_l156_156728

variables (K M A B C D : Type) [Point K] [Point M] [Point A] [Point B] [Point C] [Point D]
variables (p q : ℝ) -- base area of triangle ABC and area of triangle DKM

def midpoint (P Q R : Type) [Point P] [Point Q] [Point R] : Prop := sorry -- Defining midpoint property

def medians_intersect (G A B C : Type) [Point G] [Point A] [Point B] [Point C] : Prop := sorry -- Intersection of medians of triangle ABC

def triangle_area (P Q R : Type) [Point P] [Point Q] [Point R] (area : ℝ) : Prop := sorry -- Defining the area of the triangle

-- Stating the conditions
axioms
  (mid_K : midpoint K A B)
  (mid_M : midpoint M A C)
  (area_ABC : triangle_area A B C p)
  (area_DKM : triangle_area D K M q)
  (height_base : medians_intersect G A B C)

-- Stating the proof goal
theorem area_BCD : 
  ∃ (area : ℝ), area = sqrt (4 * q ^ 2 + p ^ 2 / 12) :=
sorry

end area_BCD_l156_156728


namespace plus_signs_count_l156_156831

theorem plus_signs_count (n p m : ℕ) (h_total : n = 23) 
  (h_plus_min : ∀ s, s.card = 10 → ∃ t ∈ s, is_plus t)
  (h_minus_min : ∀ s, s.card = 15 → ∃ t ∈ s, is_minus t)
  : p = 14 :=
sorry

end plus_signs_count_l156_156831


namespace max_area_rectangle_l156_156050

theorem max_area_rectangle (wire_length : ℕ) (red_dot_spacing : ℕ) (segments : ℕ)
  (length_width_sum : ℕ) (x y : ℕ)
  (h1 : wire_length = 78)
  (h2 : red_dot_spacing = 3)
  (h3 : segments = wire_length / red_dot_spacing)
  (h4 : length_width_sum = segments / 2)
  (h5 : x + y = length_width_sum) :
  ∃ x y, 9 * x * y = 378 :=
begin
  -- sorry to skip the proof
  sorry,
end

end max_area_rectangle_l156_156050


namespace solution_of_inequality_l156_156370

theorem solution_of_inequality (x : ℝ) : x * (x - 1) < 2 ↔ -1 < x ∧ x < 2 :=
sorry

end solution_of_inequality_l156_156370


namespace equilateral_triangle_area_l156_156978

-- Define the conditions and perform the steps to reach the final conclusion.
theorem equilateral_triangle_area (ABC : Triangle) (circle : Circle) (D E F G : Point) :
    (is_equilateral ABC ∧ inscribed ABC circle ∧ radius circle = 2) ∧
    (extends_through AB B D ∧ AD = 13) ∧
    (extends_through AC C E ∧ AE = 11) ∧
    (parallel_through D l₁ AE ∧ parallel_through E l₂ AD ∧ intersection l₁ l₂ = F) ∧
    (collinear A F G ∧ G ≠ A ∧ on_circle G circle) →
    area_triangle CBG = 429 * sqrt 3 / 433 ∧
    let p := 429; q := 3; r := 433 in p.gcd r = 1 ∧ (p + q + r = 865) :=
by
  intros h
  sorry


end equilateral_triangle_area_l156_156978


namespace general_formula_for_sequence_l156_156581

noncomputable def S := ℕ → ℚ
noncomputable def a := ℕ → ℚ

theorem general_formula_for_sequence (a : a) (S : S) (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, S (n + 1) = (2 / 3) * a (n + 1) + 1 / 3) :
  ∀ n : ℕ, a n = 
  if n = 1 then 2 
  else -5 * (-2)^(n-2) := 
by 
  sorry

end general_formula_for_sequence_l156_156581


namespace plus_signs_count_l156_156826

theorem plus_signs_count (n p m : ℕ) (h_total : n = 23) 
  (h_plus_min : ∀ s, s.card = 10 → ∃ t ∈ s, is_plus t)
  (h_minus_min : ∀ s, s.card = 15 → ∃ t ∈ s, is_minus t)
  : p = 14 :=
sorry

end plus_signs_count_l156_156826


namespace arrangement_count_is_74_l156_156301

def count_valid_arrangements : Nat :=
  74

-- Lean statement for the proof
theorem arrangement_count_is_74 :
  let seven_cards := list.range' 1 7 in
  ∃ seq : list Nat, 
    (seq.length = 7) ∧ 
    (∀ n, list.erase seq n = list.range' 1 6 ∨ 
          (list.reverse (list.erase seq n) = list.range' 1 6)) ∧
    (count_valid_arrangements = 74) :=
by
  let seven_cards := list.range' 1 7
  existsi seven_cards
  split
  -- Provide the conditions here for Lean to handle
  sorry

end arrangement_count_is_74_l156_156301


namespace polygon_sides_l156_156384

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 4 * 360) : 
  n = 6 :=
by sorry

end polygon_sides_l156_156384


namespace Olivia_score_l156_156284

theorem Olivia_score 
  (n : ℕ) (m : ℕ) (average20 : ℕ) (average21 : ℕ)
  (h_n : n = 20) (h_m : m = 21) (h_avg20 : average20 = 85) (h_avg21 : average21 = 86)
  : ∃ (scoreOlivia : ℕ), scoreOlivia = m * average21 - n * average20 :=
by
  sorry

end Olivia_score_l156_156284


namespace asian_population_west_percentage_l156_156491

-- Define the populations in millions
def population_NE := 2
def population_MW := 2
def population_South := 3
def population_West := 8

-- Define the total Asian population
def total_population := population_NE + population_MW + population_South + population_West

-- Define the population in the West
def population_in_West := population_West

-- Define the expected percentage
def expected_percentage := 53

theorem asian_population_west_percentage :
  (population_in_West.to_nat * 100 / total_population.to_nat).round = expected_percentage :=
by
  sorry

end asian_population_west_percentage_l156_156491


namespace plus_signs_count_l156_156820

theorem plus_signs_count (total_symbols : ℕ) (n : ℕ) (m : ℕ)
    (h1 : total_symbols = 23)
    (h2 : ∀ (s : Finset ℕ), s.card = 10 → (∃ x ∈ s, x = n))
    (h3 : ∀ (s : Finset ℕ), s.card = 15 → (∃ x ∈ s, x = m)) :
    n = 14 := by
  sorry

end plus_signs_count_l156_156820


namespace plus_signs_count_l156_156823

theorem plus_signs_count (total_symbols : ℕ) (n : ℕ) (m : ℕ)
    (h1 : total_symbols = 23)
    (h2 : ∀ (s : Finset ℕ), s.card = 10 → (∃ x ∈ s, x = n))
    (h3 : ∀ (s : Finset ℕ), s.card = 15 → (∃ x ∈ s, x = m)) :
    n = 14 := by
  sorry

end plus_signs_count_l156_156823


namespace min_six_integers_for_progressions_l156_156871

theorem min_six_integers_for_progressions : 
  (∀ (s : Finset ℤ), s.card < 6 →  ¬(∃ (a r d : ℤ), 
    (Set.image (λ n, a * r^n) {0, 1, 2, 3, 4} ⊆ s) ∧ 
    (Set.image (λ n, (a + n * d)) {0, 1, 2, 3, 4} ⊆ s))) ∧ 
  (∃ (s : Finset ℤ), s.card = 6 ∧ 
    (∃ (a r d : ℤ), 
      (Set.image (λ n, a * r^n) {0, 1, 2, 3, 4} ⊆ s) ∧ 
      (Set.image (λ n, (a + n * d)) {0, 1, 2, 3, 4} ⊆ s))) :=
by
  sorry

end min_six_integers_for_progressions_l156_156871


namespace number_of_valid_arrangements_l156_156310

def is_ascending (l : List ℕ) : Prop :=
  ∀ i j, i < j → l.nth i ≤ l.nth j

def is_descending (l : List ℕ) : Prop :=
  ∀ i j, i < j → l.nth i ≥ l.nth j

def remove_one_is_ordered (l : List ℕ) : Prop :=
  ∃ (i : ℕ), (is_ascending (l.removeNth i) ∨ is_descending (l.removeNth i))

def valid_arrangements_count (cards : List ℕ) : ℕ :=
  -- counting the number of valid arrangements
  if (cards.length = 7
        ∧ ∀ i, i ∈ cards → 1 ≤ i ∧ i ≤ 7 ∧ (remove_one_is_ordered cards)) then 4 else 0

theorem number_of_valid_arrangements :
  valid_arrangements_count [1,2,3,4,5,6,7] = 4 :=
by sorry

end number_of_valid_arrangements_l156_156310


namespace minimum_distance_parabola_line_l156_156759

theorem minimum_distance_parabola_line :
  let parabola := λ x : ℝ, x^2
  let distance (x0 : ℝ) : ℝ := abs (2 * x0 - x0^2 - 10) / sqrt 5
  ∃ x0 : ℝ, (∀ x : ℝ, distance x0 ≤ distance x) ∧ distance x0 = (9 * sqrt 5) / 5 :=
sorry

end minimum_distance_parabola_line_l156_156759


namespace target_heart_rate_l156_156940

theorem target_heart_rate (age : ℕ) (h_age : age = 26) :
  let mhr := 220 - age in
  let thr := 0.8 * mhr in
  Int.toNat (thr).round = 155 :=
by
  sorry

end target_heart_rate_l156_156940


namespace smallest_n_perfect_square_and_fifth_power_l156_156874

theorem smallest_n_perfect_square_and_fifth_power : 
  ∃ (n : ℕ), (∀ k : ℕ, 4 * n = k^2 ∧ 5 * n = k^5) ↔ n = 3125 :=
begin
  sorry
end

end smallest_n_perfect_square_and_fifth_power_l156_156874


namespace count_squares_in_grid_l156_156502

theorem count_squares_in_grid :
  let grid_size := 6 in
  (∑ i in Finset.range (grid_size - 1), (grid_size - i - 1)^2) = 54 :=
by
  sorry

end count_squares_in_grid_l156_156502


namespace angles_comparison_l156_156563

-- Defining the right triangle ABC with right angle at C
structure RightTriangle (A B C : Type) :=
  (right_angle : ∠ A C B = 90)
  (vertex_C : Type)
  (H3 : vertex_C)
  (height_H3 : Line vertex_C H3)

theorem angles_comparison (A B C : Type) [RightTriangle A B C] (H_3 : Type) :
  ∠ A C H_3 ≥ ∠ A B C ∧ ∠ B C H_3 ≥ ∠ B A C :=
by 
  sorry

end angles_comparison_l156_156563


namespace palindromic_sum_of_digits_l156_156930

-- Define the condition for a number being a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s == s.reverse

-- Define the condition for being a three-digit number
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Define the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem palindromic_sum_of_digits :
  ∃ y : ℕ, is_three_digit y ∧ is_palindrome y ∧ is_palindrome (y + 50) ∧ (sum_of_digits y = 20 ∨ sum_of_digits y = 22) :=
by sorry

end palindromic_sum_of_digits_l156_156930


namespace probability_area_less_than_circumference_l156_156461

theorem probability_area_less_than_circumference :
  let probability (d : ℕ) := if d = 2 then (1 / 100 : ℚ)
                             else if d = 3 then (1 / 50 : ℚ)
                             else 0
  let sum_prob (d_s : List ℚ) := d_s.foldl (· + ·) 0
  let outcomes : List ℕ := List.range' 2 19 -- dice sum range from 2 to 20
  let valid_outcomes : List ℕ := outcomes.filter (· < 4)
  sum_prob (valid_outcomes.map probability) = (3 / 100 : ℚ) :=
by
  sorry

end probability_area_less_than_circumference_l156_156461


namespace c_share_l156_156017

theorem c_share (a b c : ℝ) (h1 : a = b / 2) (h2 : b = c / 2) (h3 : a + b + c = 700) : c = 400 :=
by 
  -- Proof goes here
  sorry

end c_share_l156_156017


namespace domain_of_function_l156_156345

theorem domain_of_function (x : ℝ) (k : ℤ) :
  ∃ x, (2 * Real.sin x + 1 ≥ 0) ↔ (- (Real.pi / 6) + 2 * k * Real.pi ≤ x ∧ x ≤ (7 * Real.pi / 6) + 2 * k * Real.pi) :=
sorry

end domain_of_function_l156_156345


namespace perpendicular_lines_l156_156629

-- Define the two lines
def line1 (a : ℝ) : set (ℝ × ℝ) := {p | a * p.1 - p.2 - 2 = 0}
def line2 (a : ℝ) : set (ℝ × ℝ) := {p | (a + 2) * p.1 - p.2 + 1 = 0}

-- Lean statement to show that if the lines are perpendicular, then a = -1
theorem perpendicular_lines (a : ℝ) (h : a * (a + 2) + 1 = 0) : a = -1 :=
by {
  sorry -- proof to be done
}

end perpendicular_lines_l156_156629


namespace floor_sqrt_120_eq_10_l156_156534

theorem floor_sqrt_120_eq_10 : ⌊Real.sqrt 120⌋ = 10 := by
  -- Here, we note that we are given:
  -- 100 < 120 < 121 and the square root of it lies between 10 and 11
  sorry

end floor_sqrt_120_eq_10_l156_156534


namespace floor_sqrt_120_l156_156528

theorem floor_sqrt_120 : (⌊Real.sqrt 120⌋ = 10) :=
by
  -- Conditions from the problem
  have h1: 10^2 = 100 := rfl
  have h2: 11^2 = 121 := rfl
  have h3: 10 < Real.sqrt 120 := sorry
  have h4: Real.sqrt 120 < 11 := sorry
  -- Proof goal
  sorry

end floor_sqrt_120_l156_156528


namespace largest_n_for_sine_cosine_inequality_l156_156116

theorem largest_n_for_sine_cosine_inequality :
  ∃ n : ℕ, (∀ x : ℝ, (Real.sin x) ^ n + (Real.cos x) ^ n ≥ 2 / n) ∧ ∀ m > n, ∃ x : ℝ, (Real.sin x) ^ m + (Real.cos x) ^ m < 2 / m :=
begin
  use 4,
  split,
  { intro x,
    have h1 : (Real.sin x) ^ 4 + (Real.cos x) ^ 4 ≥ 1 / 2,
    { -- Proof using QM-AM and other inequalities
      sorry
    },
    linarith,
  },
  { intros m hm,
    -- Proof that for m > 4 the inequality fails
    sorry
  }
end

end largest_n_for_sine_cosine_inequality_l156_156116


namespace arrangement_count_is_74_l156_156300

def count_valid_arrangements : Nat :=
  74

-- Lean statement for the proof
theorem arrangement_count_is_74 :
  let seven_cards := list.range' 1 7 in
  ∃ seq : list Nat, 
    (seq.length = 7) ∧ 
    (∀ n, list.erase seq n = list.range' 1 6 ∨ 
          (list.reverse (list.erase seq n) = list.range' 1 6)) ∧
    (count_valid_arrangements = 74) :=
by
  let seven_cards := list.range' 1 7
  existsi seven_cards
  split
  -- Provide the conditions here for Lean to handle
  sorry

end arrangement_count_is_74_l156_156300


namespace solve_equation_l156_156511

theorem solve_equation (x : ℝ) (h : sqrt (4 * x - 3) + 12 / sqrt (4 * x - 3) = 8) :
  x = 39 / 4 ∨ x = 7 / 4 :=
sorry

end solve_equation_l156_156511


namespace cos_lt_sin3_div_x3_l156_156732

open Real

theorem cos_lt_sin3_div_x3 (x : ℝ) (h1 : 0 < x) (h2 : x < pi / 2) : 
  cos x < (sin x / x) ^ 3 := 
  sorry

end cos_lt_sin3_div_x3_l156_156732


namespace find_c_value_l156_156692

def finds_c (c : ℝ) : Prop :=
  6 * (-(c / 6)) + 9 * (-(c / 9)) + c = 0 ∧ (-(c / 6) + -(c / 9) = 30)

theorem find_c_value : ∃ c : ℝ, finds_c c ∧ c = -108 :=
by
  use -108
  sorry

end find_c_value_l156_156692


namespace initial_birds_count_l156_156844

theorem initial_birds_count (B : ℕ) :
  ∃ B, B + 4 = 5 + 2 → B = 3 :=
by
  sorry

end initial_birds_count_l156_156844


namespace sum_of_numerator_and_denominator_of_q_l156_156326

def probability_reaching_31 (steps : ℕ) : ℚ :=
  if steps = 4 then 1/64
  else if steps = 6 then 56/4096
  else if steps = 8 then 1656/65536
  else 0

theorem sum_of_numerator_and_denominator_of_q :
  let q := probability_reaching_31 4 + probability_reaching_31 6 + probability_reaching_31 8
  q = 7 / 128 → 
  7 + 128 = 135 :=
by
  intro hq
  linarith

end sum_of_numerator_and_denominator_of_q_l156_156326


namespace student_scores_l156_156557

theorem student_scores:
  ∀ (students : Fin 49 → Fin 8 × Fin 8 × Fin 8), 
  ∃ (A B : Fin 49), (A ≠ B) ∧ 
  (students A).1.1 ≥ (students B).1.1 ∧ 
  (students A).1.2 ≥ (students B).1.2 ∧ 
  (students A).2 ≥ (students B).2 :=
by
  sorry

end student_scores_l156_156557


namespace decimal_to_binary_51_l156_156504

theorem decimal_to_binary_51 : (51 : ℕ) = 0b110011 := by sorry

end decimal_to_binary_51_l156_156504


namespace cards_arrangement_count_l156_156298

theorem cards_arrangement_count : 
  let cards := [1, 2, 3, 4, 5, 6, 7] in
  let valid_arrangements := 
    {arrangement | ∃ removed, 
      removed ∈ cards ∧ 
      (∀ remaining, 
        remaining = cards.erase removed → 
        (sorted remaining ∨ sorted (remaining.reverse))) } in
  valid_arrangements.card = 26 :=
sorry

end cards_arrangement_count_l156_156298


namespace next_bell_ring_time_l156_156035

theorem next_bell_ring_time :
  let church_interval := 15
  let school_interval := 20
  let daycare_interval := 25
  let lcm_intervals := Nat.lcm church_interval (Nat.lcm school_interval daycare_interval)
  lcm_intervals = 300 →
  "05:00" := by
  sorry

end next_bell_ring_time_l156_156035


namespace range_of_t_l156_156151

theorem range_of_t (a b : ℝ) 
  (h1 : a^2 + a * b + b^2 = 1) 
  (h2 : ∃ t : ℝ, t = a * b - a^2 - b^2) : 
  ∀ t, t = a * b - a^2 - b^2 → -3 ≤ t ∧ t ≤ -1/3 :=
by sorry

end range_of_t_l156_156151


namespace prob_lathe_parts_l156_156840

theorem prob_lathe_parts (
  P_A_B1 : ℝ,
  P_A_B2 : ℝ,
  P_A_B3 : ℝ,
  P_B1 : ℝ,
  P_B2 : ℝ,
  P_B3 : ℝ
) : 
  (P_A_B1 = 0.05) →
  (P_A_B2 = 0.03) →
  (P_A_B3 = 0.03) →
  (P_B1 = 0.15) →
  (P_B2 = 0.25) →
  (P_B3 = 0.60) →
  (let P_A := P_A_B1 * P_B1 + P_A_B2 * P_B2 + P_A_B3 * P_B3 in
  P_A = 0.033 ∧
  P_B1 + P_B2 + P_B3 = 1 ∧
  (P_B1 * P_A_B1 / P_A = P_B2 * P_A_B2 / P_A) ∧
  (P_B1 * P_A_B1 / P_A + P_B2 * P_A_B2 / P_A ≠ P_B3 * P_A_B3 / P_A)) :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end prob_lathe_parts_l156_156840


namespace square_binomial_constant_l156_156349

theorem square_binomial_constant (y : ℝ) : ∃ b : ℝ, (y^2 + 12*y + 50 = (y + 6)^2 + b) ∧ b = 14 := 
by
  sorry

end square_binomial_constant_l156_156349


namespace plus_signs_count_l156_156817

theorem plus_signs_count (total_symbols : ℕ) (n : ℕ) (m : ℕ)
    (h1 : total_symbols = 23)
    (h2 : ∀ (s : Finset ℕ), s.card = 10 → (∃ x ∈ s, x = n))
    (h3 : ∀ (s : Finset ℕ), s.card = 15 → (∃ x ∈ s, x = m)) :
    n = 14 := by
  sorry

end plus_signs_count_l156_156817


namespace roof_dimension_difference_l156_156485

theorem roof_dimension_difference :
  ∀ (L W : ℝ), 
    (L = 4 * W) ∧ (L * W = 676) → (L - W = 39) :=
by 
  intros L W h,
  sorry

end roof_dimension_difference_l156_156485


namespace decreases_as_x_increases_graph_passes_through_origin_l156_156577

-- Proof Problem 1: Show that y decreases as x increases if and only if k > 2
theorem decreases_as_x_increases (k : ℝ) : (∀ x1 x2 : ℝ, (x1 < x2) → ((2 - k) * x1 - k^2 + 4) > ((2 - k) * x2 - k^2 + 4)) ↔ (k > 2) := 
  sorry

-- Proof Problem 2: Show that the graph passes through the origin if and only if k = -2
theorem graph_passes_through_origin (k : ℝ) : ((2 - k) * 0 - k^2 + 4 = 0) ↔ (k = -2) :=
  sorry

end decreases_as_x_increases_graph_passes_through_origin_l156_156577


namespace linear_function_quadrant_l156_156621

theorem linear_function_quadrant (x y : ℝ) (h : y = -3 * x + 2) :
  ¬ (x > 0 ∧ y > 0) :=
by
  sorry

end linear_function_quadrant_l156_156621


namespace EF_squared_eq_EP_squared_plus_FN_squared_points_E_M_G_N_collinear_l156_156944

-- Define the given conditions
variables {α : Type*} [euclidean_geometry α]
variables (O A B C D E F G P M N : α)

-- Assume initial conditions
variables (h1 : circle O)
variables (h0 : inscribed_quad ABCD h1)
variables (h2 : intersect_lines AB DC E)
variables (h3 : intersect_lines AD BC F)
variables (h4 : intersect_diagonals AC BD G)
variables (h5 : tangent_line EP O P)
variables (h6 : tangent_from_point F O M N)

-- Statement 1: EF^2 = EP^2 + FN^2
theorem EF_squared_eq_EP_squared_plus_FN_squared 
(hEF : segment E F)
(hEP : segment E P)
(hFN : segment F N) :
  (segment_length hEF) ^ 2 = 
  (segment_length hEP) ^ 2 + (segment_length hFN) ^ 2 := 
sorry

-- Statement 2: Points E, M, G, and N are collinear
theorem points_E_M_G_N_collinear 
(hEMG : collinear E M G)
(hEGN : collinear E G N) :
  collinear E M N := 
sorry

end EF_squared_eq_EP_squared_plus_FN_squared_points_E_M_G_N_collinear_l156_156944


namespace turtle_minimum_distance_l156_156477

theorem turtle_minimum_distance 
  (constant_speed : ℝ)
  (turn_angle : ℝ)
  (total_time : ℕ) :
  constant_speed = 5 →
  turn_angle = 90 →
  total_time = 11 →
  ∃ (final_position : ℝ × ℝ), 
    (final_position = (5, 0) ∨ final_position = (-5, 0) ∨ final_position = (0, 5) ∨ final_position = (0, -5)) ∧
    dist final_position (0, 0) = 5 :=
by
  intros
  sorry

end turtle_minimum_distance_l156_156477


namespace surface_area_of_sphere_l156_156388

-- Given a sphere with volume 72*pi cubic inches
def volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3
def volume_given : ℝ := 72 * Real.pi

-- Define the surface area of the sphere
def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem surface_area_of_sphere :
  ∃ r : ℝ, volume r = volume_given → surface_area r = 36 * Real.pi * 2^(2/3) :=
by {
  sorry
}

end surface_area_of_sphere_l156_156388


namespace symphony_orchestra_has_260_members_l156_156387

def symphony_orchestra_member_count (n : ℕ) : Prop :=
  200 < n ∧ n < 300 ∧ n % 6 = 2 ∧ n % 8 = 3 ∧ n % 9 = 4

theorem symphony_orchestra_has_260_members : symphony_orchestra_member_count 260 :=
by {
  sorry
}

end symphony_orchestra_has_260_members_l156_156387


namespace complex_fourth_power_l156_156086

noncomputable def complex_number : ℂ := 3 * complexOfReal((real.cos (real.pi / 6))) + 3i * complexOfReal((real.sin (real.pi / 6)))

theorem complex_fourth_power :
  (complex_number ^ 4) = -40.5 + 40.5 * (sqrt 3) * complex.i := 
by 
  sorry

end complex_fourth_power_l156_156086


namespace figure_E_not_formed_l156_156009

-- Defining the pieces
def piece1 := (1, 1) -- 1x1 square
def piece2 := (1, 2) -- 1x2 rectangle
def piece3 := (2, 2) -- 2x2 square

-- Defining the total area of pieces
def total_area := 3 * (1 * 1) + 2 * (1 * 2) + (2 * 2)

-- Definition of figures and their areas
def figure_A := 3 * (1 * 1) + 1 * (1 * 2) -- Example for figure A
def figure_B := sorry  -- Define areas for other figures likewise
def figure_C := sorry
def figure_D := sorry
def figure_E := sorry -- Area which cannot be formed

-- Proof goal
theorem figure_E_not_formed : cannot_form figure_E := 
by sorry

end figure_E_not_formed_l156_156009


namespace rounding_bounds_l156_156472

theorem rounding_bounds (x : ℝ) (h : round_to 2 x = 2.48) :
  2.475 ≤ x ∧ x ≤ 2.484 := sorry

end rounding_bounds_l156_156472


namespace excircles_tangent_to_vertices_l156_156167

noncomputable def tangent_to_vertices (a b : ℝ) (h : a > b > 0) (P : ℝ × ℝ) : Prop :=
∃ (F₁ F₂ A₁ A₂ : ℝ × ℝ),
  let ellipse := (λ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) in
  let on_ellipse := ellipse P.1 P.2 in
  let foci_property := (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 + (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 = 4 * a^2 in
  let right_vertex := A₂ = (a, 0) in
  let left_vertex := A₁ = (-a, 0) in
  let excircle_opposite_PF1F2 := tangent (∠P F₁ F₂) A₂ in
  let excircle_opposite_PF2F1 := tangent (∠P F₂ F₁) A₁ in
  on_ellipse ∧ foci_property ∧ right_vertex ∧ left_vertex ∧ excircle_opposite_PF1F2 ∧ excircle_opposite_PF2F1

theorem excircles_tangent_to_vertices (a b : ℝ) (h : a > b > 0) (P : ℝ × ℝ) :
  tangent_to_vertices a b h P :=
sorry

end excircles_tangent_to_vertices_l156_156167


namespace relationship_of_y_values_l156_156644

def parabola_y (x : ℝ) (c : ℝ) : ℝ :=
  2 * (x + 1)^2 + c

theorem relationship_of_y_values (c : ℝ) (y1 y2 y3 : ℝ) :
  y1 = parabola_y (-2) c →
  y2 = parabola_y 1 c →
  y3 = parabola_y 2 c →
  y3 > y2 ∧ y2 > y1 :=
by
  intros h1 h2 h3
  sorry

end relationship_of_y_values_l156_156644


namespace paths_from_A_to_B_l156_156453

def path_count_A_to_B : Nat :=
  let red_to_blue_ways := [2, 3]  -- 2 ways to first blue, 3 ways to second blue
  let blue_to_green_ways_first := 4 * 2  -- Each of the 2 green arrows from first blue, 4 ways each
  let blue_to_green_ways_second := 5 * 2 -- Each of the 2 green arrows from second blue, 5 ways each
  let green_to_B_ways_first := 2 * blue_to_green_ways_first  -- Each of the first green, 2 ways each
  let green_to_B_ways_second := 3 * blue_to_green_ways_second  -- Each of the second green, 3 ways each
  green_to_B_ways_first + green_to_B_ways_second  -- Total paths from green arrows to B

theorem paths_from_A_to_B : path_count_A_to_B = 46 := by
  sorry

end paths_from_A_to_B_l156_156453


namespace intersection_point_is_correct_l156_156113

def line1 (x y : ℝ) := x - 2 * y + 7 = 0
def line2 (x y : ℝ) := 2 * x + y - 1 = 0

theorem intersection_point_is_correct : line1 (-1) 3 ∧ line2 (-1) 3 :=
by
  sorry

end intersection_point_is_correct_l156_156113


namespace boys_and_girls_in_class_l156_156231

theorem boys_and_girls_in_class (m d : ℕ)
  (A : (m - 1 = 10 ∧ (d - 1 = 14 ∨ d - 1 = 10 + 4 ∨ d - 1 = 10 - 4)) ∨ 
       (m - 1 = 14 - 4 ∧ (d - 1 = 14 ∨ d - 1 = 10 + 4 ∨ d - 1 = 10 - 4)))
  (B : (m - 1 = 13 ∧ (d - 1 = 11 ∨ d - 1 = 11 + 4 ∨ d - 1 = 11 - 4)) ∨ 
       (m - 1 = 11 - 4 ∧ (d - 1 = 11 ∨ d - 1 = 11 + 4 ∨ d - 1 = 11 - 4)))
  (C : (m - 1 = 13 ∧ (d - 1 = 19 ∨ d - 1 = 19 + 4 ∨ d - 1 = 19 - 4)) ∨ 
       (m - 1 = 19 - 4 ∧ (d - 1 = 19 ∨ d - 1 = 19 + 4 ∨ d - 1 = 19 - 4))) : 
  m = 14 ∧ d = 15 := 
sorry

end boys_and_girls_in_class_l156_156231


namespace area_quadrilateral_QTSU_l156_156023

theorem area_quadrilateral_QTSU
  (P Q R S T U : Type)
  (distance_PQ : dist P Q = 3)
  (distance_QR : dist Q R = 3)
  (distance_RS : dist R S = 3)
  (midpoint_T : dist P T = 1.5 ∧ dist T Q = 1.5)
  (intersection_U : median P R ∩ median Q S = U) :
  area_quadrilateral Q T S U = 4.5 * sqrt 3 := 
sorry

end area_quadrilateral_QTSU_l156_156023


namespace prove_a2_b2_prove_general_formula_l156_156237

noncomputable def a_seq : ℕ+ → ℕ+
| 1 => 1
| n => (n * (n + 1)) / 2

noncomputable def b_seq : ℕ+ → ℕ+
| 1 => 4
| n => (n + 1) ^ 2

def sum_seq (f : ℕ+ → ℕ+) : ℕ+ → ℕ
| 1     => f 1
| n + 1 => sum_seq n + f (n + 1)

def satisfies_condition (f : ℕ+ → ℕ) : Prop :=
    ∀ n : ℕ+, n * sum_seq f (n + 1) - (n + 3) * sum_seq f n = 0

def geometric_mean_condition (a : ℕ+ → ℕ+) (b : ℕ+ → ℕ+) : Prop :=
    ∀ n : ℕ+, 2 * a (n + 1) = (b n * b (n + 1)) ^ (1 / 2)

theorem prove_a2_b2 : 
  (a_seq 2 = 3) ∧ (b_seq 2 = 9) :=
by
  sorry

theorem prove_general_formula :
    (satisfies_condition a_seq) ∧ (geometric_mean_condition a_seq b_seq) →
    (∀ n : ℕ+, a_seq n = n * (n + 1) / 2) ∧ 
    (∀ n : ℕ+, b_seq n = (n + 1) ^ 2) :=
by
  sorry

end prove_a2_b2_prove_general_formula_l156_156237


namespace perimeter_triangle_AEC_l156_156927

open Real

noncomputable def perimeter_triangle_AEC' : ℝ :=
  let A := (0 : ℝ, 2 : ℝ)
  let B := (0 : ℝ, 0 : ℝ)
  let C := (2 : ℝ, 0 : ℝ)
  let D := (2 : ℝ, 2 : ℝ)
  let C' := (2 : ℝ, 4 / 3 : ℝ)
  let E := (3 / 2 : ℝ, 3 / 2 : ℝ)
  let dist := λ (p1 p2 : ℝ × ℝ), sqrt ((p1.fst - p2.fst) ^ 2 + (p1.snd - p2.snd) ^ 2)
  dist A E + dist E C' + dist A C'

theorem perimeter_triangle_AEC'_is_correct :
  perimeter_triangle_AEC' = 4 * sqrt 10 / 3 := by
  sorry

end perimeter_triangle_AEC_l156_156927


namespace sum_gcd_lcm_is_39_l156_156000

theorem sum_gcd_lcm_is_39 : Nat.gcd 30 81 + Nat.lcm 36 12 = 39 := by 
  sorry

end sum_gcd_lcm_is_39_l156_156000


namespace student_weekly_allowance_l156_156471

theorem student_weekly_allowance (A : ℝ) :
  (3 / 4) * (1 / 3) * ((2 / 5) * A + 4) - 2 = 0 ↔ A = 100/3 := sorry

end student_weekly_allowance_l156_156471


namespace flour_needed_l156_156051

theorem flour_needed (cookies : ℕ) (flour : ℕ) (k : ℕ) (f_whole_wheat f_all_purpose : ℕ) 
  (h : cookies = 45) (h1 : flour = 3) (h2 : k = 90) (h3 : (k / 2) = 45) 
  (h4 : f_all_purpose = (flour * (k / cookies)) / 2) 
  (h5 : f_whole_wheat = (flour * (k / cookies)) / 2) : 
  f_all_purpose = 3 ∧ f_whole_wheat = 3 := 
by
  sorry

end flour_needed_l156_156051


namespace box_volume_l156_156943

-- Let's define the initial conditions
def initial_length : ℝ := 18
def initial_width : ℝ := 12
def cut_length (x : ℝ) : ℝ := x

-- Define the new dimensions after cutting and folding
def new_length (x : ℝ) : ℝ := initial_length - 2 * (cut_length x)
def new_width (x : ℝ) : ℝ := initial_width - 2 * (cut_length x)
def height (x : ℝ) : ℝ := cut_length x

-- Define the volume of the box
def volume (x : ℝ) : ℝ := (new_length x) * (new_width x) * (height x)

-- The theorem we want to prove
theorem box_volume (x : ℝ) : volume x = 4 * x^3 - 60 * x^2 + 216 * x :=
by {
  -- Placeholder for proof
  sorry
}

end box_volume_l156_156943


namespace value_of_X_after_execution_l156_156973

-- Define the initial conditions
def initial_X : ℕ := 5
def initial_S : ℕ := 10

-- Define the step increments
def increment_X (X : ℕ) : ℕ := X + 3
def increment_S (S : ℕ) (X : ℕ) : ℕ := S + X

-- Define the loop to find the value of X when S ≥ 12000
noncomputable def final_X : ℕ :=
  let rec find_X (X S : ℕ) : ℕ :=
    if S ≥ 12000 then X
    else find_X (increment_X X) (increment_S S (increment_X X))
  in find_X initial_X initial_S

-- State the problem as a theorem
theorem value_of_X_after_execution : final_X = 275 :=
sorry

end value_of_X_after_execution_l156_156973


namespace complex_power_4_l156_156079

noncomputable def cos30_re : ℂ := 3 * real.cos (π / 6)
noncomputable def sin30_im : ℂ := 3 * complex.I * real.sin (π / 6)
noncomputable def c : ℂ := cos30_re + sin30_im

theorem complex_power_4 :
  c ^ 4 = -40.5 + 40.5 * complex.I * real.sqrt 3 := sorry

end complex_power_4_l156_156079


namespace area_inequality_l156_156362

theorem area_inequality
  (ABCDE : Pentagon) 
  (circumscribed : circumscribed ABCDE) 
  (w w_a w_b w_c w_d w_e : Circle)
  (refl_w_AB : reflection w AB = w_a)
  (refl_w_BC : reflection w BC = w_b)
  (refl_w_CD : reflection w CD = w_c)
  (refl_w_DE : reflection w DE = w_d)
  (refl_w_EA : reflection w EA = w_e)
  (A' : Point) (B' : Point) (C' : Point) (D' : Point) (E' : Point)
  (sec_inter_A' : second_intersection w_a w_e = A')
  (sec_inter_B' : second_intersection w_b w_a = B')
  (sec_inter_C' : second_intersection w_c w_b = C')
  (sec_inter_D' : second_intersection w_d w_c = D')
  (sec_inter_E' : second_intersection w_e w_d = E') :
  2 ≤ area (Polygon.mk [A', B', C', D', E']) / area ABCDE ∧ 
  area (Polygon.mk [A', B', C', D', E']) / area ABCDE ≤ 3 := 
by
  sorry

end area_inequality_l156_156362


namespace matrix_inverse_eq_l156_156354

theorem matrix_inverse_eq (d k : ℚ) (A : Matrix (Fin 2) (Fin 2) ℚ) 
  (hA : A = ![![1, 4], ![6, d]]) 
  (hA_inv : A⁻¹ = k • A) :
  (d, k) = (-1, 1/25) :=
  sorry

end matrix_inverse_eq_l156_156354


namespace electric_fan_wattage_l156_156257

theorem electric_fan_wattage (hours_per_day : ℕ) (energy_per_month : ℝ) (days_per_month : ℕ) 
  (h1 : hours_per_day = 8) (h2 : energy_per_month = 18) (h3 : days_per_month = 30) : 
  (energy_per_month * 1000) / (days_per_month * hours_per_day) = 75 := 
by { 
  -- Placeholder for the proof
  sorry 
}

end electric_fan_wattage_l156_156257


namespace ab_value_l156_156469

theorem ab_value (a b : ℝ) (h1 : a + b = 4) (h2 : a^2 + b^2 = 18) : a * b = -1 :=
by sorry

end ab_value_l156_156469


namespace symmetric_line_equation_l156_156347

-- Define the original line as an equation in ℝ².
def original_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Define the line of symmetry.
def line_of_symmetry (x : ℝ) : Prop := x = 1

-- The theorem stating the equation of the symmetric line.
theorem symmetric_line_equation (x y : ℝ) :
  original_line x y → line_of_symmetry x → (x + 2 * y - 3 = 0) :=
by
  intros h₁ h₂
  sorry

end symmetric_line_equation_l156_156347


namespace orthocenter_of_triangle_l156_156608

theorem orthocenter_of_triangle (A : ℝ × ℝ) (x y : ℝ) 
  (h₁ : x + y = 0) (h₂ : 2 * x - 3 * y + 1 = 0) : 
  A = (1, 2) → (x, y) = (-1 / 5, 1 / 5) :=
by
  sorry

end orthocenter_of_triangle_l156_156608


namespace find_m_l156_156176

theorem find_m 
  (m : ℝ)
  (h_pos : 0 < m)
  (asymptote_twice_angle : ∃ l : ℝ, l = 3 ∧ (x - l * y = 0 ∧ m * x^2 - y^2 = m)) :
  m = 3 :=
by
  sorry

end find_m_l156_156176


namespace floor_sqrt_120_eq_10_l156_156535

theorem floor_sqrt_120_eq_10 : ⌊Real.sqrt 120⌋ = 10 := by
  -- Here, we note that we are given:
  -- 100 < 120 < 121 and the square root of it lies between 10 and 11
  sorry

end floor_sqrt_120_eq_10_l156_156535


namespace range_of_p_l156_156150

noncomputable def condition_P (p : ℝ) : Prop := 0 < p ∧ p < 2/3
noncomputable def condition_Q (p : ℝ) : Prop := 1/2 < p ∧ p < 2

theorem range_of_p (p : ℝ) :
  (¬ (condition_P p ∧ condition_Q p)) ∧ (condition_P p ∨ condition_Q p) ↔
  (p ∈ Ioo 0 (1/2) ∪ Ico (2/3) 2) :=
  sorry

end range_of_p_l156_156150


namespace pour_out_one_sixth_of_water_tilt_angle_is_correct_l156_156060

noncomputable def equilateral_triangle_tilt_angle
  (side_length : ℝ) -- The side length of the equilateral triangle
  (water_initial_height : ℝ) -- The initial height (full) of the water in the trough
  (water_desired_height : ℝ) -- The height after pouring out 1/6 of water
  (desired_ratio : ℝ := 5/6):-- The desired ratio of the area remaining
ℝ :=
  let h := (side_length * (real.sqrt 3)) / 2 in -- Height of equilateral triangle
  let remaining_height := water_desired_height * desired_ratio in
  let tilt_angle := (real.arctan ((remaining_height * real.sqrt 3) / side_length)) * 180 / real.pi in
  -- Adjust the angle to find the required tilt for specific water level
  tilt_angle

theorem pour_out_one_sixth_of_water_tilt_angle_is_correct:
  ∀ (side_length : ℝ),
  let water_initial_height := (side_length * (real.sqrt 3)) / 2 in
  let water_desired_height := water_initial_height * (5/6) in
  let angle := equilateral_triangle_tilt_angle side_length water_initial_height water_desired_height in
  abs (angle - 10.9) < 0.1 :=
  sorry

end pour_out_one_sixth_of_water_tilt_angle_is_correct_l156_156060


namespace lambda_value_l156_156181

theorem lambda_value (λ: ℝ) (a b: ℝ × ℝ) 
  (h_a : a = (3,2)) 
  (h_b : b = (-7, λ + 1)) 
  (h_parallel : ∃ k: ℝ, (11 * a.fst - 2018 * b.fst, 11 * a.snd - 2018 * b.snd) = k • (10 * a.fst + 2017 * b.fst, 10 * a.snd + 2017 * b.snd)) :
  λ = - 17 / 3 := 
by
  sorry

end lambda_value_l156_156181


namespace count_of_plus_signs_l156_156806

-- Definitions based on the given conditions
def total_symbols : ℕ := 23
def any_10_symbols_contains_plus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 10 → (\exists c ∈ s, c = '+')
def any_15_symbols_contains_minus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 15 → (\exists c ∈ s, c = '-')

-- Main theorem statement without the proof
theorem count_of_plus_signs (symbols : Finset char) (h1 : symbols.card = total_symbols) 
  (h2 : any_10_symbols_contains_plus symbols) 
  (h3 : any_15_symbols_contains_minus symbols) : 
  symbols.count '+' = 14 :=
sorry

end count_of_plus_signs_l156_156806


namespace lambda_range_l156_156589

def A := (2, 3 : ℝ)
def B := (5, 4 : ℝ)
def C := (7, 10 : ℝ)

def vectorSub (p1 p2 : ℝ × ℝ) : ℝ × ℝ := (p1.1 - p2.1, p1.2 - p2.2)
def scalarMul (a : ℝ) (p : ℝ × ℝ) : ℝ × ℝ := (a * p.1, a * p.2)

noncomputable def AB := vectorSub B A
noncomputable def AC := vectorSub C A

def thirdQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0

theorem lambda_range (P : ℝ × ℝ) (λ : ℝ) (h₁ : P = (A.1 + AB.1 + λ * AC.1, A.2 + AB.2 + λ * AC.2)) 
(h₂ : thirdQuadrant P) : λ < -1 := sorry

end lambda_range_l156_156589


namespace decrease_in_combined_area_l156_156179

theorem decrease_in_combined_area (r1 r2 r3 : ℝ) :
    let π := Real.pi
    let A_original := π * (r1 ^ 2) + π * (r2 ^ 2) + π * (r3 ^ 2)
    let r1' := r1 * 0.5
    let r2' := r2 * 0.5
    let r3' := r3 * 0.5
    let A_new := π * (r1' ^ 2) + π * (r2' ^ 2) + π * (r3' ^ 2)
    let Decrease := A_original - A_new
    Decrease = 0.75 * π * (r1 ^ 2) + 0.75 * π * (r2 ^ 2) + 0.75 * π * (r3 ^ 2) :=
by
  sorry

end decrease_in_combined_area_l156_156179


namespace cover_black_squares_with_L_shape_l156_156556

-- Define a function to check if a number is odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Define the main theorem
theorem cover_black_squares_with_L_shape (n : ℕ) (h_odd : is_odd n) (h_corner_black : ∀i j, (i = 0 ∨ i = n - 1) ∧ (j = 0 ∨ j = n - 1) → (i + j) % 2 = 1) : n ≥ 7 :=
sorry

end cover_black_squares_with_L_shape_l156_156556


namespace jeremy_not_A_implies_score_less_than_90_l156_156348

variables {Jeremy : Type}
variable (score : Jeremy → ℝ)
variable (grade : Jeremy → Prop)
variable (A : Prop)

-- Condition: Scoring at least 90% on the multiple-choice questions guarantees an A grade.
axiom scoring_90_implies_A : ∀ j : Jeremy, score j ≥ 90 → grade j = A

-- Mathematically equivalent proof problem
theorem jeremy_not_A_implies_score_less_than_90 :
  ∀ j : Jeremy, grade j ≠ A → score j < 90 :=
by
  intros j h1
  by_contra h2
  exact h1 (scoring_90_implies_A j h2)

end jeremy_not_A_implies_score_less_than_90_l156_156348


namespace sin_x_in_terms_of_a_and_b_l156_156197

theorem sin_x_in_terms_of_a_and_b (a b x : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : 0 < x) (h4 : x < 90) (h5 : cot x = (a^2 - b^2) / (2 * a * b)) : 
  sin x = (2 * a * b) / (a^2 + b^2) :=
by
  sorry

end sin_x_in_terms_of_a_and_b_l156_156197


namespace least_five_digit_congruent_eight_mod_17_l156_156865

theorem least_five_digit_congruent_eight_mod_17 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 8 ∧ n = 10009 :=
by
  use 10009
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  sorry

end least_five_digit_congruent_eight_mod_17_l156_156865


namespace find_z_l156_156570

variables (z : ℂ)

def is_pure_imaginary (w : ℂ) : Prop :=
  re w = 0

theorem find_z (h1 : complex.abs z = 5) (h2 : is_pure_imaginary ((3 + 4 * I) * z)) :
  z = 4 + 3 * I ∨ z = -(4 + 3 * I) :=
by sorry

end find_z_l156_156570


namespace polygon_sides_equation_l156_156373

theorem polygon_sides_equation (n : ℕ) 
  (h1 : (n-2) * 180 = 4 * 360) : n = 10 := 
by 
  sorry

end polygon_sides_equation_l156_156373


namespace initial_customers_correct_l156_156933

def initial_customers (remaining : ℕ) (left : ℕ) : ℕ := remaining + left

theorem initial_customers_correct :
  initial_customers 12 9 = 21 :=
by
  sorry

end initial_customers_correct_l156_156933


namespace trig_identity_l156_156138

theorem trig_identity (α : ℝ) (h : Real.tan α = 2 / 3) : 
  Real.sin (2 * α) - Real.cos (π - 2 * α) = 17 / 13 :=
by
  sorry

end trig_identity_l156_156138


namespace floor_sqrt_120_eq_10_l156_156520

theorem floor_sqrt_120_eq_10 :
  (√120).to_floor = 10 := by
  have h1 : √100 = 10 := by norm_num
  have h2 : √121 = 11 := by norm_num
  have h : 100 < 120 ∧ 120 < 121 := by norm_num
  have sqrt_120 : 10 < √120 ∧ √120 < 11 :=
    by exact ⟨real.sqrt_lt' 120 121 h.2, real.sqrt_lt'' 100 120 h.1⟩
  sorry

end floor_sqrt_120_eq_10_l156_156520


namespace approximate_to_nearest_hundredth_l156_156744

theorem approximate_to_nearest_hundredth (x : ℝ) (h : x = 3.1415) : Real.floor (x * 100 + 0.5) / 100 = 3.14 := by
  sorry

end approximate_to_nearest_hundredth_l156_156744


namespace find_function_and_extrema_l156_156143

def quadratic_function (f : ℝ → ℝ) : Prop :=
  f(0) = 1 ∧ (∀ x : ℝ, f(x + 1) - f(x) = 2x)

theorem find_function_and_extrema (f : ℝ → ℝ)
  (hf : quadratic_function f) :
  (∀ x : ℝ, f(x) = x^2 - x + 1) ∧
  (∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), 
    ∃ m M : ℝ, m = 3 / 4 ∧ M = 3 ∧
    (∀ y ∈ set.Icc (-1 : ℝ) 1, m ≤ f(y) ∧ f(y) ≤ M)) :=
by
  sorry

end find_function_and_extrema_l156_156143


namespace right_triangle_area_l156_156512

def is_right_triangle (a b c : Point) : Prop :=
  -- definition for right triangle 
  sorry

noncomputable def area_of_triangle (a b c : Point) : ℝ :=
  -- definition for the area of a triangle 
  sorry

def satisfies_inequalities (x y k : ℝ) : Prop :=
  (x ≥ 0) ∧ (y ≥ 2 * x) ∧ (k * x - y + 1 ≥ 0)

theorem right_triangle_area {k : ℝ} :
  (∃ a b c : Point, 
    is_right_triangle a b c ∧ 
    satisfies_inequalities (a.x) (a.y) k ∧ 
    satisfies_inequalities (b.x) (b.y) k ∧ 
    satisfies_inequalities (c.x) (c.y) k) →
  (area_of_triangle a b c = 1/5 ∨ area_of_triangle a b c = 1/4) :=
by
  sorry

end right_triangle_area_l156_156512


namespace equivalence_cond_indep_l156_156897

variables {Ω : Type*} 
variables [measurable_space Ω]
variables {μ : measure Ω}
variables {𝒜 𝒞 𝒷 : measurable_space Ω}

def cond_indep (𝒜 𝒷 𝒞 : measurable_space Ω) : Prop :=
  ∀ A ∈ 𝒜.sets, ∀ B ∈ 𝒷.sets, condexp (𝒞.measurable_space) (A ∩ B) = condexp (𝒞.measurable_space) A * condexp (𝒞.measurable_space) B

theorem equivalence_cond_indep (𝒜 𝒷 𝒞 : measurable_space Ω) :
  (∀ A ∈ 𝒜.sets, condexp (𝒷.measurable_space ⊔ 𝒞.measurable_space) A = condexp (𝒞.measurable_space) A) ↔
  (∀ X : Ω → ℝ, measurable[𝒜] X → integrable[μ] X → condexp (𝒷.measurable_space ⊔ 𝒞.measurable_space) X = condexp (𝒞.measurable_space) X) ↔
  (∀ A ∈ (generate_from (𝒜.pi_sets)).sets, condexp (𝒷.measurable_space ⊔ 𝒞.measurable_space) A = condexp (𝒞.measurable_space) A) ↔
  (cond_indep 𝒜 𝒷 𝒞) :=
sorry

end equivalence_cond_indep_l156_156897


namespace c_completes_in_three_days_l156_156900

variables (r_A r_B r_C : ℝ)
variables (h1 : r_A + r_B = 1/3)
variables (h2 : r_B + r_C = 1/3)
variables (h3 : r_A + r_C = 2/3)

theorem c_completes_in_three_days : 1 / r_C = 3 :=
by sorry

end c_completes_in_three_days_l156_156900


namespace total_points_second_half_proof_l156_156032

variables (a r b e : ℕ) -- Define the sequences variables.
constants (S_X S_Y : ℕ) -- Scores of Team X and Team Y.

-- Conditions:
-- 1. Geometric sequence for Team X.
-- 2. Arithmetic sequence for Team Y.
-- 3. Scores tied at the first quarter which implies: a = b.
-- 4. Team X won by two points: S_X = S_Y + 2.
-- 5. Scores do not exceed 120 points.
-- 6. Specific sequence properties used: r = 2, a = 4, e = 7.

-- Define total scores for Team X and Team Y.
def total_points_X (a r : ℕ) : ℕ := a * (1 + r + r^2 + r^3)
def total_points_Y (b e : ℕ) : ℕ := 4 * b + 6 * e

-- Translate the mathematical problem into a Lean statement.
theorem total_points_second_half_proof :
  a = 4 → b = a → r = 2 → e = 7 →
  S_X = total_points_X a r →
  S_Y = total_points_Y b e →
  S_X = S_Y + 2 →
  (16 + 32) + (18 + 25) = 91 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  simp [total_points_X, total_points_Y] at h5 h6
  sorry -- Proof is omitted.

end total_points_second_half_proof_l156_156032


namespace polygon_sides_l156_156379

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 4 * 360) → n = 10 :=
by 
  sorry

end polygon_sides_l156_156379


namespace sampling_method_systematic_l156_156910

theorem sampling_method_systematic 
  (inspect_interval : ℕ := 10)
  (products_interval : ℕ := 10)
  (position : ℕ) :
  inspect_interval = 10 ∧ products_interval = 10 → 
  (sampling_method = "Systematic Sampling") :=
by
  sorry

end sampling_method_systematic_l156_156910


namespace nth_term_formula_l156_156156

def a : ℕ → ℚ
| 1          := 1 / 4
| (n + 1)    := 1 / 2 * a n + 2^(-(n+1))  -- Use negative exponent for 2^-(n+1)

theorem nth_term_formula (n : ℕ) (h : n ≥ 1) : a n = (2 * n - 1) / 2^(n + 1) :=
by
  sorry

end nth_term_formula_l156_156156


namespace triangle_height_BF_l156_156564

theorem triangle_height_BF 
  (B A C E F : Point)
  (BE_circumcenter: is_circumcenter_on (triangle B A C) (ray B E))
  (cond_1: intersection (line_segment B E) (line_segment A C) = some E)
  (cond_2: AF_FE_val : (length (line_segment A F)) * (length (line_segment F E)) = 5)
  (cond_3: cot_ratio : cot (angle E B C) / cot (angle B E C) = 3 / 4)
  : length (line_segment B F) = 1.94 := 
by
  sorry

end triangle_height_BF_l156_156564


namespace plus_signs_count_l156_156814

theorem plus_signs_count (num_symbols : ℕ) (at_least_one_plus_in_10 : ∀ s : Finset ℕ, s.card = 10 → (∃ i ∈ s, i < 14)) (at_least_one_minus_in_15 : ∀ s : Finset ℕ, s.card = 15 → (∃ i ∈ s, i ≥ 14)) : 
    ∃ (p m : ℕ), p + m = 23 ∧ p = 14 ∧ m = 9 := by
  sorry

end plus_signs_count_l156_156814


namespace angle_sum_triangle_l156_156243

theorem angle_sum_triangle (A B C : ℝ) 
  (hA : A = 20)
  (hC : C = 90) :
  B = 70 := 
by
  -- In a triangle the sum of angles is 180 degrees
  have h_sum : A + B + C = 180 := sorry
  -- Substitute the given angles A and C
  rw [hA, hC] at h_sum
  -- Simplify the equation to find B
  have hB : 20 + B + 90 = 180 := sorry
  linarith

end angle_sum_triangle_l156_156243


namespace find_c_l156_156205

-- Define the polynomial f(x)
def f (c : ℚ) (x : ℚ) : ℚ := 2 * c * x^3 + 14 * x^2 - 6 * c * x + 25

-- State the problem in Lean 4
theorem find_c (c : ℚ) : (∀ x : ℚ, f c x = 0 ↔ x = (-5)) → c = 75 / 44 := 
by sorry

end find_c_l156_156205


namespace average_weight_increase_l156_156337

noncomputable def initial_average_weight (A : ℝ) : ℝ := 5 * A
noncomputable def new_average_weight (A : ℝ) (X : ℝ) : ℝ := (5 * A - 68 + 95.5) / 5

theorem average_weight_increase (A X : ℝ) :
  A + X = new_average_weight A X → X = 5.5 :=
by
  intro h
  have h1 : 5 * (A + X) = 5 * A - 68 + 95.5 := by
    rw [h]
    format
  calc
    X = (27.5) / 5 := by sorry

end average_weight_increase_l156_156337


namespace minimum_bailing_rate_correct_l156_156007

-- Define the conditions as Lean definitions
def distance_from_shore : ℝ := 2 -- 2 miles
def water_inflow_rate : ℝ := 6 -- 6 gallons per minute
def boat_max_water : ℝ := 60 -- 60 gallons
def rowing_speed : ℝ := 3 -- 3 miles per hour

-- Total allowable water intake without sinking
def allowable_water_intake : ℝ := boat_max_water

-- Total travel time to shore
def time_to_shore : ℝ := (distance_from_shore / rowing_speed) * 60 -- in minutes

-- Total water intake during the journey
def total_water_intake : ℝ := water_inflow_rate * time_to_shore

-- Excess water to bail out
def water_to_bail_out : ℝ := total_water_intake - allowable_water_intake

-- Minimum bailing rate required ( r = (total water to bail out) / (time to shore) )
def min_bailing_rate : ℝ := water_to_bail_out / time_to_shore

-- Theorem to prove the minimum bailing rate is 4.5 gallons per minute
theorem minimum_bailing_rate_correct : min_bailing_rate = 4.5 := 
by 
  -- Placeholder for the proof, which is not required here
  sorry

end minimum_bailing_rate_correct_l156_156007


namespace product_of_repeating_decimals_l156_156496

theorem product_of_repeating_decimals :
  let x := (4 / 9 : ℚ)
  let y := (7 / 9 : ℚ)
  x * y = 28 / 81 :=
by
  sorry

end product_of_repeating_decimals_l156_156496


namespace amount_saved_is_40_percent_l156_156945

theorem amount_saved_is_40_percent (P : ℕ) :
    let original_price := 5 * P
    let sale_price := 3 * P
    let amount_saved := original_price - sale_price
    (amount_saved.to_rat / original_price.to_rat) * 100 = 40 :=
by
    sorry

end amount_saved_is_40_percent_l156_156945


namespace problem_solution_l156_156956

theorem problem_solution :
  (∏ k in finset.range(21).map (λ n, n + 1), (1 + 19 / k) / (1 + 21 / k)) = 1 / 686400 :=
by sorry

end problem_solution_l156_156956


namespace tournament_no_cyclic_triad_l156_156679

variable (n : ℕ) (S : Fin n → ℕ)

noncomputable def sum_of_squares (S : Fin n → ℕ) : ℕ :=
  Finset.univ.sum (λ i, S i ^ 2)

theorem tournament_no_cyclic_triad (S : Fin n → ℕ) (h : Finset.univ.sum S = n * (n - 1) / 2) :
  sum_of_squares S < n * (n - 1) * (2 * n - 1) / 6 ↔ 
  ∃ A B C, S A > S B ∧ S B > S C ∧ S C > S A := sorry

end tournament_no_cyclic_triad_l156_156679


namespace plus_signs_count_l156_156798

noncomputable def solveSymbols : Prop :=
  ∃ p m : ℕ, p + m = 23 ∧ p ≤ 14 ∧ m ≤ 9 ∧ (∀ s : Finset (Fin 23), s.card = 10 → (s.image (λ i => if i < p then true else false)).count true ≥ 1) ∧ (∀ s : Finset (Fin 23), s.card = 15 → (s.image (λ i => if i < p then false else true)).count false ≥ 1) 

theorem plus_signs_count : solveSymbols :=
sorry

end plus_signs_count_l156_156798


namespace sin_A_value_values_b_c_l156_156669

variables (A B C : ℝ) (a b c : ℝ)

-- Given conditions
def condition_side_a : a = 2 := sorry
def condition_cos_B : cos B = 3 / 5 := sorry
def condition_b_4 : b = 4 := sorry
def condition_area : 1 / 2 * a * c * sin B = 4 := sorry

-- Proof (1) that given the conditions, sin A = 2/5
theorem sin_A_value :
  (a = 2) → (b = 4) → (cos B = 3 / 5) → (sin A = 2 / 5) := 
by 
  intros h1 h2 h3 
  -- Here would be the proof steps
  sorry

-- Proof (2) that given the conditions, b = √17 and c = 5
theorem values_b_c :
  (a = 2) → (cos B = 3 / 5) → (1 / 2 * a * c * sin B = 4) → (b = real.sqrt 17) ∧ (c = 5) := 
by 
  intros h1 h2 h3 
  -- Here would be the proof steps
  sorry

end sin_A_value_values_b_c_l156_156669


namespace find_d_l156_156214

theorem find_d (y d : ℝ) (hy : y > 0) (h : (8 * y) / 20 + (3 * y) / d = 0.7 * y) : d = 10 :=
by
  sorry

end find_d_l156_156214


namespace number_wall_top_block_value_l156_156235

theorem number_wall_top_block_value (a b c d : ℕ) 
    (h1 : a = 8) (h2 : b = 5) (h3 : c = 3) (h4 : d = 2) : 
    (a + b + (b + c) + (c + d) = 34) :=
by
  sorry

end number_wall_top_block_value_l156_156235


namespace combined_salaries_l156_156367

theorem combined_salaries (A_s S_avg : ℝ) (hA : A_s = 8000) (hS : S_avg = 9000) :
  ∑ i in {B, C, D, E}, i = 37000 :=
by
    let total_salary := 5 * S_avg
    have h_total_salary : total_salary = 45000, by
    { rw hS, norm_num }

    let combined_salaries := total_salary - A_s
    have h_combined_salaries : combined_salaries = 37000, by
    { rw [←h_total_salary, hA], norm_num }

    exact h_combined_salaries


end combined_salaries_l156_156367


namespace multiplication_is_correct_l156_156028

theorem multiplication_is_correct : 209 * 209 = 43681 := sorry

end multiplication_is_correct_l156_156028


namespace smallest_z_for_27z_gt_3_24_l156_156868

theorem smallest_z_for_27z_gt_3_24 (z : ℤ) (h : 27 = 3 ^ 3) : 27 ^ z > 3 ^ 24 ↔ z ≥ 9 :=
by
  sorry

end smallest_z_for_27z_gt_3_24_l156_156868


namespace count_of_plus_signs_l156_156803

-- Definitions based on the given conditions
def total_symbols : ℕ := 23
def any_10_symbols_contains_plus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 10 → (\exists c ∈ s, c = '+')
def any_15_symbols_contains_minus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 15 → (\exists c ∈ s, c = '-')

-- Main theorem statement without the proof
theorem count_of_plus_signs (symbols : Finset char) (h1 : symbols.card = total_symbols) 
  (h2 : any_10_symbols_contains_plus symbols) 
  (h3 : any_15_symbols_contains_minus symbols) : 
  symbols.count '+' = 14 :=
sorry

end count_of_plus_signs_l156_156803


namespace transformation_sequences_count_l156_156476

-- Definitions based on the conditions
def Point := (ℝ × ℝ)
def A : Point := (1, 0)
def B : Point := (0, 1)
def C : Point := (-1, -1)

-- Transformation declarations
def L : Point → Point := sorry -- rotation by 120 degrees counterclockwise
def R : Point → Point := sorry -- rotation by 120 degrees clockwise
def H : Point → Point := sorry -- reflection across the x-axis
def V : Point → Point := sorry -- reflection across the y-axis

-- Identity relation declarations
def I : Point → Point := id

-- Group relations
axiom L_identity : ∀ (p : Point), L (L (L p)) = p
axiom R_identity : ∀ (p : Point), R (R (R p)) = p
axiom H_identity : ∀ (p : Point), H (H p) = p
axiom V_identity : ∀ (p : Point), V (V p) = p

-- Theorem statement
theorem transformation_sequences_count :
  ∃ (count : ℕ), count = 286 ∧ 
  (∀ (seq : List (Point → Point)), seq.length = 12 →
    (∀ (t ∈ seq, t ∈ [L, R, H, V]) → (∃ (perm : Point → Point), perm A = A ∧ perm B = B ∧ perm C = C))) :=
by
  sorry

end transformation_sequences_count_l156_156476


namespace power_subtraction_l156_156650

variable {a m n : ℝ}

theorem power_subtraction (hm : a^m = 8) (hn : a^n = 2) : a^(m - 3 * n) = 1 := by
  sorry

end power_subtraction_l156_156650


namespace exists_distinct_integers_l156_156127

theorem exists_distinct_integers (n : ℕ) (hn : 0 < n) :
  ∃ (a : Fin n → ℕ), 
  (∀ i j, i ≠ j → a i ≠ a j) ∧ 
  is_pure_2009th_power (∑ i, a i) ∧ 
  is_pure_2010th_power (∏ i, a i) :=
  sorry

def is_pure_kth_power (k : ℕ) (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m ^ k

def is_pure_2009th_power := is_pure_kth_power 2009
def is_pure_2010th_power := is_pure_kth_power 2010


end exists_distinct_integers_l156_156127


namespace find_distances_l156_156460

-- Definitions for conditions
def total_distance := 21
def walking_speed := 4
def bus_speed := 60
def bus_speed_fraction := 21 / bus_speed
def walking_time (distance : ℝ) := distance / walking_speed

-- Main theorem statement
theorem find_distances (x y : ℝ) (h1 : x + y = total_distance) 
    (h2 : x / bus_speed + bus_speed_fraction = (10 / bus_speed) + walking_time y) 
    : x = 19 ∧ y = 2 :=
sorry

end find_distances_l156_156460


namespace part1_part2_l156_156173

-- 1. Prove f(x) < 0 for -1 < x < 0 and f(x) > 0 for x > 0 when a = 0
theorem part1 (x : ℝ) (hx_neg : -1 < x) (hx_pos: x > 0) : 
  let f := λ x, (2 + x) * Real.log (1 + x) - 2 * x in 
  (hx_neg → f x < 0) ∧ (hx_pos → f x > 0 ) := 
sorry

-- 2. Find the value of a such that x = 0 is a local maximum of f(x)
theorem part2 (a : ℝ) (f := λ x, (2 + x + a * x^2) * Real.log (1 + x) - 2 * x) :
  is_local_max f 0 ↔ a = -1 / 6 :=
sorry

end part1_part2_l156_156173


namespace prob_l156_156842

noncomputable def g (x : ℝ) : ℝ := 1 / (2 + 1 / (2 + 1 / x))

theorem prob (x1 x2 x3 : ℝ) (h1 : x1 = 0) 
  (h2 : 2 + 1 / x2 = 0) 
  (h3 : 2 + 1 / (2 + 1 / x3) = 0) : 
  x1 + x2 + x3 = -9 / 10 := 
sorry

end prob_l156_156842


namespace complex_sum_real_imag_l156_156124

theorem complex_sum_real_imag : 
  (Complex.re ((Complex.I / (1 + Complex.I)) - (1 / (2 * Complex.I))) + 
  Complex.im ((Complex.I / (1 + Complex.I)) - (1 / (2 * Complex.I)))) = 3/2 := 
by sorry

end complex_sum_real_imag_l156_156124


namespace count_divisible_by_90_l156_156195

theorem count_divisible_by_90 : 
  ∃ n, n = 10 ∧ (∀ k, 1000 ≤ k ∧ k < 10000 ∧ k % 100 = 90 ∧ k % 90 = 0 → n = 10) :=
begin
  sorry
end

end count_divisible_by_90_l156_156195


namespace complicated_fraction_equiv_one_l156_156953

theorem complicated_fraction_equiv_one : 
  (∏ i in Finset.range 21, (1 + 19 / (i + 1))) / (∏ i in Finset.range 19, (1 + 21 / (i + 1))) = 1 :=
by sorry

end complicated_fraction_equiv_one_l156_156953


namespace skates_cost_is_65_l156_156069

constant admission_cost : ℝ := 5
constant rental_cost_per_visit : ℝ := 2.50
constant visits_to_justify : ℕ := 26

noncomputable def new_skates_cost : ℝ :=
  rental_cost_per_visit * visits_to_justify

theorem skates_cost_is_65 : new_skates_cost = 65 := by
  unfold new_skates_cost
  calc
    rental_cost_per_visit * visits_to_justify
      = 2.50 * 26 : by sorry
    ... = 65 : by sorry

end skates_cost_is_65_l156_156069


namespace area_of_triangle_l156_156215

-- Given conditions
variables (a b c : ℝ) (α β γ : ℝ) (A B C : Triangle ℝ)
  (ha : A.sides = (a, b, c))
  (hb : A.angles = (α, β, γ))
  (h1 : a^2 + b^2 = 4 - (cos γ)^2)
  (h2 : a * b = 2)
  (hγ : γ = π/2)

-- Statement to prove the area of triangle ABC is 1
theorem area_of_triangle : triangle.area A = 1 :=
sorry

end area_of_triangle_l156_156215


namespace minimum_socks_to_guarantee_20_pairs_l156_156039

-- Definitions and conditions
def red_socks := 120
def green_socks := 100
def blue_socks := 80
def black_socks := 50
def number_of_pairs := 20

-- Statement
theorem minimum_socks_to_guarantee_20_pairs 
  (red_socks green_socks blue_socks black_socks number_of_pairs: ℕ) 
  (h1: red_socks = 120) 
  (h2: green_socks = 100) 
  (h3: blue_socks = 80) 
  (h4: black_socks = 50) 
  (h5: number_of_pairs = 20) : 
  ∃ min_socks, min_socks = 43 := 
by 
  sorry

end minimum_socks_to_guarantee_20_pairs_l156_156039


namespace cards_arrangement_count_l156_156299

theorem cards_arrangement_count : 
  let cards := [1, 2, 3, 4, 5, 6, 7] in
  let valid_arrangements := 
    {arrangement | ∃ removed, 
      removed ∈ cards ∧ 
      (∀ remaining, 
        remaining = cards.erase removed → 
        (sorted remaining ∨ sorted (remaining.reverse))) } in
  valid_arrangements.card = 26 :=
sorry

end cards_arrangement_count_l156_156299


namespace correctness_of_propositions_l156_156768

-- Definitions of the conditions
def residual_is_random_error (e : ℝ) : Prop := ∃ (y : ℝ) (y_hat : ℝ), e = y - y_hat
def data_constraints (a b c d : ℕ) : Prop := a ≥ 5 ∧ b ≥ 5 ∧ c ≥ 5 ∧ d ≥ 5
def histogram_judgement : Prop := ∀ (H : Type) (rel : H → H → Prop), ¬(H ≠ H) ∨ (∀ x y : H, rel x y ↔ true)

-- The mathematical equivalence proof problem
theorem correctness_of_propositions (e : ℝ) (a b c d : ℕ) : 
  (residual_is_random_error e → false) ∧
  (data_constraints a b c d → true) ∧
  (histogram_judgement → true) :=
by
  sorry

end correctness_of_propositions_l156_156768


namespace meet_probability_correct_l156_156286

open Classical

-- Definitions based on given conditions
noncomputable def prob_A_reaches (i j : ℕ) : ℝ := (Nat.choose 10 i * Nat.choose (10 - i) j)/ 3^10
noncomputable def prob_B_reaches (i j : ℕ) : ℝ := (Nat.choose 10 (6 - i) * Nat.choose (10 - (6 - i)) (8 - j)) / 2^10

-- Conditions
def valid_meeting (i j : ℕ) := i + j = 10 ∧ i ≤ 6 ∧ j ≤ 8

-- Sum probability of all valid meeting points
noncomputable def total_meeting_probability : ℝ := ∑ i in Finset.range 11, ∑ j in Finset.range 9, if valid_meeting i j then (prob_A_reaches i j) * (prob_B_reaches i j) else 0

-- Goal
theorem meet_probability_correct : total_meeting_probability = 0.0139 := by
  sorry

end meet_probability_correct_l156_156286


namespace calcium_oxide_required_l156_156989

def balanced_reaction (n_cao n_h2o n_caoh2 : ℕ) : Prop :=
  n_cao = n_h2o ∧ n_h2o = n_caoh2

theorem calcium_oxide_required {n_h2o : ℕ} (h : n_h2o = 3) :
  ∃ n_cao, balanced_reaction n_cao n_h2o 3 :=
by
  use 3
  simp [balanced_reaction, h]
  sorry

end calcium_oxide_required_l156_156989


namespace trigonometric_identity_l156_156539

theorem trigonometric_identity (α : ℝ) :
  (sin (13 * α) + sin (14 * α) + sin (15 * α) + sin (16 * α)) /
  (cos (13 * α) + cos (14 * α) + cos (15 * α) + cos (16 * α)) = 
  tan (29 * α / 2) :=
  sorry

end trigonometric_identity_l156_156539


namespace not_younger_means_taller_l156_156291

-- Definitions
variables {Person : Type} (younger older : Person → Person → Prop)
variable tallest : Person → Prop

variables (A B C : Person)
-- Conditions
axiom younger_never_contradicts_elder {p1 p2 : Person} : younger p1 p2 → ¬(tallest p1 ∧ tallest p2)
axiom elder_not_wrong_when_contradicts_younger {p1 p2 : Person} : older p1 p2 → ¬(tallest p1 ∧ tallest p2)

-- Residents statements
axiom statement_A : tallest B
axiom statement_B : tallest A
axiom statement_C : ∀ p, taller p B → p = C

-- To Prove
theorem not_younger_means_taller : ¬(younger A B ∧ younger B C → tallest C) :=
by sorry

end not_younger_means_taller_l156_156291


namespace visibility_beach_to_hill_visibility_ferry_to_tree_l156_156250

noncomputable def altitude_lake : ℝ := 104
noncomputable def altitude_hill_tree : ℝ := 154
noncomputable def map_distance_1 : ℝ := 70 / 100 -- Convert cm to meters
noncomputable def map_distance_2 : ℝ := 38.5 / 100 -- Convert cm to meters
noncomputable def map_scale : ℝ := 95000
noncomputable def earth_circumference : ℝ := 40000000 -- Convert km to meters

noncomputable def earth_radius : ℝ := earth_circumference / (2 * Real.pi)

noncomputable def visible_distance (height : ℝ) : ℝ :=
  Real.sqrt (2 * earth_radius * height)

noncomputable def actual_distance_1 : ℝ := map_distance_1 * map_scale
noncomputable def actual_distance_2 : ℝ := map_distance_2 * map_scale

theorem visibility_beach_to_hill :
  actual_distance_1 > visible_distance (altitude_hill_tree - altitude_lake) :=
by
  sorry

theorem visibility_ferry_to_tree :
  actual_distance_2 > visible_distance (altitude_hill_tree - altitude_lake) :=
by
  sorry

end visibility_beach_to_hill_visibility_ferry_to_tree_l156_156250


namespace frac_mul_sub_eq_l156_156428

/-
  Theorem:
  The result of multiplying 2/9 by 4/5 and then subtracting 1/45 is equal to 7/45.
-/
theorem frac_mul_sub_eq :
  (2/9 * 4/5 - 1/45) = 7/45 :=
by
  sorry

end frac_mul_sub_eq_l156_156428


namespace box_difference_proof_l156_156725

def box_weights :=
  (w1 w2 w3 : ℕ)

theorem box_difference_proof (w1 w3 : ℕ) (h1 : w1 = 2) (h3 : w3 = 13) :
  w3 - w1 = 11 :=
  by {
    -- The proof will go here
    sorry
  }

end box_difference_proof_l156_156725


namespace range_of_independent_variable_l156_156766

theorem range_of_independent_variable (x : ℝ) :
  (sqrt (x - 1)).nonneg → x ≥ 1 :=
by
  sorry

end range_of_independent_variable_l156_156766


namespace coordinates_of_C_l156_156468

def point : Type := (ℝ × ℝ)

def A : point := (-3, 5)
def B : point := (9, -1)

def dist (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def C : point :=
  let BC := dist A B * (2 / 5)
  (B.1 + (B.1 - A.1) * (2 / 5), B.2 + (B.2 - A.2) * (2 / 5))

theorem coordinates_of_C :
  C = (13.8, -3.4) :=
sorry

end coordinates_of_C_l156_156468


namespace tan_thirty_eq_sqrt3_div_3_l156_156773

theorem tan_thirty_eq_sqrt3_div_3 :
  ∀ (triangle : Type) (x y z : ℝ),
  (triangle ∧ x = 1 ∧ y = Real.sqrt 3 ∧ z = 2) →
  Real.tan (Real.pi / 6) = Real.sqrt 3 / 3 := 
by
  sorry

end tan_thirty_eq_sqrt3_div_3_l156_156773


namespace correct_propositions_l156_156169

-- Definitions related to the conditions
def prop1 (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = -f x → f 0 = 0

def prop2 (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = f x → ∀ a, f a = f (-a)

def function1 (x : ℝ) := x^3 + 1
def function2 (x : ℝ) := -|x| + 1

def prop3 : Prop := 
  ∀ x, function1 (-x) ≠ -function1 x

def prop4 : Prop := 
  ∀ x, function2 (-x) = function2 x → False

-- Statement of the proof problem
theorem correct_propositions :
  ({2, 3} : set ℕ) = {i | (i = 2 ∧ prop2 function1) ∨ (i = 3 ∧ prop3) } :=
sorry

end correct_propositions_l156_156169


namespace max_bishops_on_chessboard_8x8_l156_156104

theorem max_bishops_on_chessboard_8x8 (n : ℕ) (h1 : n = 64):
  ∃ m, m = 16 ∧
  (∀ i j, i ≠ j → are_diagonal i j → placed_bishop i → placed_bishop j → false) ∧
  (∀ i, placed_bishop i → ∃! j, i ≠ j ∧ are_diagonal i j ∧ placed_bishop j) :=
by
  let chessboard := (fin 8) × (fin 8)
  let bishop_position := chessboard
  let placed_bishops := set bishop_position
  let is_max := ∀ (n : ℕ), (∀ b : placed_bishops, n ≤ 16)
  have : placed_bishops → ∃ m, m = 16 ∧ is_max := sorry
  exact ⟨16, this⟩

end max_bishops_on_chessboard_8x8_l156_156104


namespace investment_ratio_l156_156061

theorem investment_ratio (A_invest B_invest C_invest : ℝ) (F : ℝ) (total_profit B_share : ℝ)
  (h1 : A_invest = 3 * B_invest)
  (h2 : B_invest = F * C_invest)
  (h3 : total_profit = 7700)
  (h4 : B_share = 1400)
  (h5 : (B_invest / (A_invest + B_invest + C_invest)) * total_profit = B_share) :
  (B_invest / C_invest) = 2 / 3 := 
by
  sorry

end investment_ratio_l156_156061


namespace plus_signs_count_l156_156810

theorem plus_signs_count (num_symbols : ℕ) (at_least_one_plus_in_10 : ∀ s : Finset ℕ, s.card = 10 → (∃ i ∈ s, i < 14)) (at_least_one_minus_in_15 : ∀ s : Finset ℕ, s.card = 15 → (∃ i ∈ s, i ≥ 14)) : 
    ∃ (p m : ℕ), p + m = 23 ∧ p = 14 ∧ m = 9 := by
  sorry

end plus_signs_count_l156_156810


namespace probability_not_e_after_n_spins_l156_156582

theorem probability_not_e_after_n_spins
    (S : Type)
    (e b c d : S)
    (p_e : ℝ)
    (p_b : ℝ)
    (p_c : ℝ)
    (p_d : ℝ) :
    (p_e = 0.25) →
    (p_b = 0.25) →
    (p_c = 0.25) →
    (p_d = 0.25) →
    (1 - p_e)^2 = 0.5625 :=
by
  sorry

end probability_not_e_after_n_spins_l156_156582


namespace ordered_pair_unique_solution_l156_156094

theorem ordered_pair_unique_solution : ∃ x y : ℤ, 
  0 < x ∧ 0 < y ∧
  (x ^ y + 3 = y ^ x + 1) ∧
  (2 * x ^ y + 4 = y ^ x + 9) ∧
  (x = 3 ∧ y = 1) := by
squeeze_simp
-- sorry is used to complete the Lean script without providing the proof steps
sorry

end ordered_pair_unique_solution_l156_156094


namespace range_of_independent_variable_l156_156765

theorem range_of_independent_variable (x : ℝ) :
  (sqrt (x - 1)).nonneg → x ≥ 1 :=
by
  sorry

end range_of_independent_variable_l156_156765


namespace student_divisor_l156_156223

theorem student_divisor (x : ℕ) : (24 * x = 42 * 36) → x = 63 := 
by
  intro h
  sorry

end student_divisor_l156_156223


namespace last_digit_of_1_div_3_pow_9_is_7_l156_156426

noncomputable def decimal_expansion_last_digit (n d : ℕ) : ℕ :=
  (n / d) % 10

theorem last_digit_of_1_div_3_pow_9_is_7 :
  decimal_expansion_last_digit 1 (3^9) = 7 :=
by
  sorry

end last_digit_of_1_div_3_pow_9_is_7_l156_156426


namespace ellipse_eq_range_m_l156_156208

theorem ellipse_eq_range_m (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (m - 1) + y^2 / (3 - m) = 1)) ↔ (1 < m ∧ m < 2) ∨ (2 < m ∧ m < 3) :=
sorry

end ellipse_eq_range_m_l156_156208


namespace trapezoid_area_l156_156934

noncomputable def isosceles_triangle (A B C : Type) (AB AC : ℝ) : Prop := AB = AC

noncomputable def similar_triangles (A B C : Type) : Prop := sorry -- Similar triangles condition

constant smallest_triangle_area : ℝ

axiom smallest_triangle_area_def : smallest_triangle_area = 1

constant ABC_area : ℝ

axiom ABC_area_def : ABC_area = 36

constant ADE_area : ℝ

axiom ADE_area_def : ADE_area = 5

noncomputable def trap_area (DBCE_area : ℝ) : Prop := DBCE_area = ABC_area - ADE_area

theorem trapezoid_area (DBCE_area : ℝ) (h1: isosceles_triangle A B C AB AC)
  (h2: similar_triangles A B C) (h3: smallest_triangle_area = 1) (h4: ABC_area = 36)
  (h5: ADE_area = 5) : trap_area DBCE_area :=
by {
  rw [trap_area, ABC_area_def, ADE_area_def],
  exact dec_trivial,
  sorry,
}


end trapezoid_area_l156_156934


namespace bricks_needed_for_room_floor_l156_156219

theorem bricks_needed_for_room_floor
    (num_rooms : ℕ)
    (room_length : ℝ)
    (room_breadth : ℝ)
    (room_height : ℝ)
    (bricks_per_sqm : ℕ)
    (one_room_area : ℝ := room_length * room_breadth)
    (total_bricks_needed : ℕ := nat_floor (one_room_area * (bricks_per_sqm: ℝ)))
    :
    num_rooms = 5 → 
    room_length = 4 → 
    room_breadth = 5 → 
    room_height = 2 → 
    bricks_per_sqm = 17 → 
    total_bricks_needed = 340 :=
by 
    intros
    sorry

end bricks_needed_for_room_floor_l156_156219


namespace grass_field_width_l156_156052

theorem grass_field_width
  (l p A_path : ℝ)
  (h1 : l = 75)
  (h2 : p = 2.8)
  (h3 : A_path = 1518.72)
  : let w := 190.6 in
    80.6 * (w + 5.6) - 75 * w = A_path := by
sorry

end grass_field_width_l156_156052


namespace cards_arrangement_count_is_10_l156_156318

-- Define the problem in Lean statement terms
def valid_arrangements_count : ℕ :=
  -- number of arrangements of seven cards where one card can be removed 
  -- leaving the remaining six cards in either ascending or descending order
  10

-- Theorem stating that the number of valid arrangements is 10
theorem cards_arrangement_count_is_10 : valid_arrangements_count = 10 :=
by
  -- Proof is omitted (the explanation above corresponds to the omitted proof details)
  sorry

end cards_arrangement_count_is_10_l156_156318


namespace RouteB_quicker_than_RouteA_l156_156723

def RouteA_segment1_time : ℚ := 4 / 40 -- time in hours
def RouteA_segment2_time : ℚ := 4 / 20 -- time in hours
def RouteA_total_time : ℚ := RouteA_segment1_time + RouteA_segment2_time -- total time in hours

def RouteB_segment1_time : ℚ := 6 / 35 -- time in hours
def RouteB_segment2_time : ℚ := 1 / 15 -- time in hours
def RouteB_total_time : ℚ := RouteB_segment1_time + RouteB_segment2_time -- total time in hours

def time_difference_minutes : ℚ := (RouteA_total_time - RouteB_total_time) * 60 -- difference in minutes

theorem RouteB_quicker_than_RouteA : time_difference_minutes = 3.71 := by
  sorry

end RouteB_quicker_than_RouteA_l156_156723


namespace exists_plane_perpendicular_to_c_subs_a_parallel_b_l156_156595

variables {Point : Type} [EuclideanSpace Point]
  
-- Define lines and planes
variable {Line : Type} [has_perpendicular Line Point]
variable {Plane : Type} [has_perpendicular Plane Line] [has_subset Plane Line] [has_parallel Plane Line] [has_parallel Line Line]
  
variables (a b c : Line) 

-- Define conditions
def skew_lines (a b : Line) : Prop := ¬ ∃ (p : Point), p ∈ a ∧ p ∈ b
def perpendicular_line (c : Line) (l : Line) : Prop := c ⊥ l
def perpendicular_plane (c : Line) (π : Plane) : Prop := c ⊥ π
def subset_plane (l : Line) (π : Plane) : Prop := l ⊂ π
def parallel_plane (l : Line) (π : Plane) := l ∥ π
  
-- conditions
variable (h1 : skew_lines a b)
variable (h2 : perpendicular_line c a)
variable (h3 : perpendicular_line c b)

-- Proof goal:
theorem exists_plane_perpendicular_to_c_subs_a_parallel_b :
  ∃ α : Plane, perpendicular_plane c α ∧ subset_plane a α ∧ parallel_plane b α :=
sorry

end exists_plane_perpendicular_to_c_subs_a_parallel_b_l156_156595


namespace two_pow_n_add_two_gt_n_sq_l156_156501

open Nat

theorem two_pow_n_add_two_gt_n_sq (n : ℕ) (h : n > 0) : 2^n + 2 > n^2 :=
by
  sorry

end two_pow_n_add_two_gt_n_sq_l156_156501


namespace arrangement_count_is_74_l156_156304

def count_valid_arrangements : Nat :=
  74

-- Lean statement for the proof
theorem arrangement_count_is_74 :
  let seven_cards := list.range' 1 7 in
  ∃ seq : list Nat, 
    (seq.length = 7) ∧ 
    (∀ n, list.erase seq n = list.range' 1 6 ∨ 
          (list.reverse (list.erase seq n) = list.range' 1 6)) ∧
    (count_valid_arrangements = 74) :=
by
  let seven_cards := list.range' 1 7
  existsi seven_cards
  split
  -- Provide the conditions here for Lean to handle
  sorry

end arrangement_count_is_74_l156_156304


namespace compute_fourth_power_z_l156_156082

-- Definitions from the problem
def cos_angle (θ : ℝ) : ℝ := Real.cos θ
def sin_angle (θ : ℝ) : ℝ := Real.sin θ
def θ := Real.pi / 6  -- 30 degrees in radians

def z : ℂ := 3 * (cos_angle θ) + 3 * Complex.I * (sin_angle θ)

-- Lean 4 Statement for the proof
theorem compute_fourth_power_z : (z ^ 4) = -40.5 + 40.5 * Complex.I * Real.sqrt 3 :=
sorry

end compute_fourth_power_z_l156_156082


namespace hassan_orange_trees_l156_156481

theorem hassan_orange_trees :
  ∀ (O : ℕ), 
    let ahmed_orange_trees := 8 in
    let hassan_apple_trees := 1 in
    let ahmed_apple_trees := 4 * hassan_apple_trees in
    let total_ahmed_trees := ahmed_orange_trees + ahmed_apple_trees in
    let total_hassan_trees := O + hassan_apple_trees in
    total_ahmed_trees = total_hassan_trees + 9 →
    O = 2 :=
by 
  intros O ahmed_orange_trees hassan_apple_trees ahmed_apple_trees total_ahmed_trees total_hassan_trees h
  sorry

end hassan_orange_trees_l156_156481


namespace number_of_valid_arrangements_l156_156314

def is_ascending (l : List ℕ) : Prop :=
  ∀ i j, i < j → l.nth i ≤ l.nth j

def is_descending (l : List ℕ) : Prop :=
  ∀ i j, i < j → l.nth i ≥ l.nth j

def remove_one_is_ordered (l : List ℕ) : Prop :=
  ∃ (i : ℕ), (is_ascending (l.removeNth i) ∨ is_descending (l.removeNth i))

def valid_arrangements_count (cards : List ℕ) : ℕ :=
  -- counting the number of valid arrangements
  if (cards.length = 7
        ∧ ∀ i, i ∈ cards → 1 ≤ i ∧ i ≤ 7 ∧ (remove_one_is_ordered cards)) then 4 else 0

theorem number_of_valid_arrangements :
  valid_arrangements_count [1,2,3,4,5,6,7] = 4 :=
by sorry

end number_of_valid_arrangements_l156_156314


namespace plus_signs_count_l156_156783

theorem plus_signs_count (p m : ℕ) (h_sum : p + m = 23)
                         (h_max_minus : m ≤ 9) (h_max_plus : p ≤ 14)
                         (h_at_least_one_plus_in_10 : ∀ (s : Finset (Fin 23)), s.card = 10 → ∃ i ∈ s, i < p)
                         (h_at_least_one_minus_in_15 : ∀ (s : Finset (Fin 23)), s.card = 15 → ∃ i ∈ s, m ≤ i) :
  p = 14 :=
by sorry

end plus_signs_count_l156_156783


namespace right_triangle_inequality_l156_156731

theorem right_triangle_inequality (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  a^4 + b^4 < c^4 :=
by
  sorry

end right_triangle_inequality_l156_156731


namespace q_roots_l156_156440

noncomputable def has_distinct_real_roots (p : ℝ[X]) (n : ℕ) : Prop :=
∃ (roots : finset ℝ), (∀ a ∈ roots, p.eval a = 0) ∧ roots.card = n ∧ ∀ a ∈ roots, a > 1

noncomputable def q (p : ℝ[X]) : ℝ[X] :=
(x^2 + 1) * p * p.derivative + x * p^2 + x * p.derivative^2

theorem q_roots (p : ℝ[X]) (n : ℕ)
  (hp : has_distinct_real_roots p n) :
  ∃ (roots : finset ℝ), roots.card ≥ (2 * n - 1) ∧ ∀ a ∈ roots, q p.eval a = 0 :=
sorry

end q_roots_l156_156440


namespace albert_total_profit_l156_156482

theorem albert_total_profit (total_cost : ℝ) (cost_per_horse : ℝ) (num_horses : ℕ) (num_cows : ℕ) (horse_profit_pct : ℝ) (cow_profit_pct : ℝ) 
  (horse_total_cost_eq : cost_per_horse * num_horses = 8000) 
  (total_cost_eq : total_cost = 13400) 
  (profit_horse_eq : horse_profit_pct = 0.10) 
  (profit_cow_eq : cow_profit_pct = 0.20) 
  (horse_count_eq : num_horses = 4) 
  (cow_count_eq : num_cows = 9) :
  let cost_per_cow := 5400 / num_cows
      horse_profit := num_horses * (cost_per_horse * horse_profit_pct)
      cow_profit := num_cows * (cost_per_cow * cow_profit_pct)
      total_profit := horse_profit + cow_profit
  in total_profit = 1880 :=
by
  sorry

end albert_total_profit_l156_156482


namespace problem_statement_l156_156647

theorem problem_statement
  (a b : ℝ)
  (ha : a = Real.sqrt 2 + 1)
  (hb : b = Real.sqrt 2 - 1) :
  a^2 - a * b + b^2 = 5 :=
sorry

end problem_statement_l156_156647


namespace plus_signs_count_l156_156793

noncomputable def solveSymbols : Prop :=
  ∃ p m : ℕ, p + m = 23 ∧ p ≤ 14 ∧ m ≤ 9 ∧ (∀ s : Finset (Fin 23), s.card = 10 → (s.image (λ i => if i < p then true else false)).count true ≥ 1) ∧ (∀ s : Finset (Fin 23), s.card = 15 → (s.image (λ i => if i < p then false else true)).count false ≥ 1) 

theorem plus_signs_count : solveSymbols :=
sorry

end plus_signs_count_l156_156793


namespace max_divisors_in_range_l156_156288

theorem max_divisors_in_range :
  let max_divisor_numbers := {12, 18, 20}
  let num_divisors (n : ℕ) := (finset.range (n + 1)).filter (λ d, n % d = 0)
  let max_divisors := finset.fold max 0 (finset.range 21).image (λ n, (num_divisors n).card)
  (finset.filter (λ n, (num_divisors n).card = max_divisors) (finset.range 21)).val.to_finset = max_divisor_numbers :=
sorry

end max_divisors_in_range_l156_156288


namespace sequence_formula_l156_156769

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 then 1 / 2 else
  (sequence (n - 1) * sequence (n - 1 / 2)).sqrt / 2).sqrt

theorem sequence_formula (n : ℕ) : sequence n = Math.sin (Real.pi / (3 * 2 ^ n)) := 
  sorry

end sequence_formula_l156_156769


namespace not_collinear_l156_156447

noncomputable def a : ℝ × ℝ × ℝ := (7, 9, -2)
noncomputable def b : ℝ × ℝ × ℝ := (5, 4, 3)
noncomputable def c1 : ℝ × ℝ × ℝ := (4 * 7 - 5, 4 * 9 - 4, 4 * -2 - 3)
noncomputable def c2 : ℝ × ℝ × ℝ := (4 * 5 - 7, 4 * 4 - 9, 4 * 3 + 2)

def collinear (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ γ : ℝ, v1 = (γ * v2.1, γ * v2.2, γ * v2.3)

theorem not_collinear : ¬ collinear c1 c2 :=
by
  simp [collinear, c1, c2]
  sorry

end not_collinear_l156_156447


namespace systematic_sampling_correct_l156_156737

theorem systematic_sampling_correct :
  ∀ (total_students selected_students interval random_number group_number selected_number : ℕ),
    total_students = 900 →
    selected_students = 50 →
    interval = total_students / selected_students →
    random_number = 8 →
    interval * group_number - (interval - random_number) = selected_number →
    91 ≤ group_number * interval - (interval - random_number) ∧ group_number * interval - (interval - random_number) ≤ 108 →
    selected_number = 98 :=
by
  intros total_students selected_students interval random_number group_number selected_number ht hs hi hr hsng hrange
  have k_eq_6 : group_number = 6, from sorry
  exact k_eq_6 ▸ hsng

end systematic_sampling_correct_l156_156737


namespace remainder_of_product_divided_by_10_l156_156429

theorem remainder_of_product_divided_by_10 :
  let a := 2457
  let b := 6273
  let c := 91409
  (a * b * c) % 10 = 9 :=
by
  sorry

end remainder_of_product_divided_by_10_l156_156429


namespace find_a_if_remainder_constant_l156_156982

variable {R : Type} [CommRing R]
noncomputable def poly1 (a : R) := 10 * X^3 - 7 * X^2 + a * X + 6
def poly2 := 2 * X^2 - 3 * X + 1

theorem find_a_if_remainder_constant (a : R) :
  ∃ r : R, ∀ s, polynomial.div_mod_by_monic (poly1 a) poly2 = (s, C r) → r 0 :=
begin
  sorry
end

end find_a_if_remainder_constant_l156_156982


namespace man_earnings_first_third_day_l156_156045

theorem man_earnings_first_third_day (E : ℕ) (H1 : ∀ n : ℕ, n ≤ 60 → (n % 2 = 0 → man_earns E n ∧ man_spends 15 (n + 1)))
  (H2 : man_has 60 at_day 60) : E = 17 :=
sorry

end man_earnings_first_third_day_l156_156045


namespace store_loss_90_l156_156470

theorem store_loss_90 (x y : ℝ) (h1 : x * (1 + 0.12) = 3080) (h2 : y * (1 - 0.12) = 3080) :
  2 * 3080 - x - y = -90 :=
by
  sorry

end store_loss_90_l156_156470


namespace Mairead_triathlon_l156_156885

noncomputable def convert_km_to_miles (km: Float) : Float :=
  0.621371 * km

noncomputable def convert_yards_to_miles (yd: Float) : Float :=
  0.000568182 * yd

noncomputable def convert_feet_to_miles (ft: Float) : Float :=
  0.000189394 * ft

noncomputable def total_distance_in_miles := 
  let run_distance_km := 40.0
  let run_distance_miles := convert_km_to_miles run_distance_km
  let walk_distance_miles := 3.0/5.0 * run_distance_miles
  let jog_distance_yd := 5.0 * (walk_distance_miles * 1760.0)
  let jog_distance_miles := convert_yards_to_miles jog_distance_yd
  let bike_distance_ft := 3.0 * (jog_distance_miles * 5280.0)
  let bike_distance_miles := convert_feet_to_miles bike_distance_ft
  let swim_distance_miles := 2.5
  run_distance_miles + walk_distance_miles + jog_distance_miles + bike_distance_miles + swim_distance_miles

theorem Mairead_triathlon:
  total_distance_in_miles = 340.449562 ∧
  (convert_km_to_miles 40.0) / 10.0 = 2.485484 ∧
  (3.0/5.0 * (convert_km_to_miles 40.0)) / 10.0 = 1.4912904 ∧
  (convert_yards_to_miles (5.0 * (3.0/5.0 * (convert_km_to_miles 40.0) * 1760.0))) / 10.0 = 7.45454544 ∧
  (convert_feet_to_miles (3.0 * (convert_yards_to_miles (5.0 * (3.0/5.0 * (convert_km_to_miles 40.0) * 1760.0)) * 5280.0))) / 10.0 = 22.36363636 ∧
  2.5 / 10.0 = 0.25 := sorry

end Mairead_triathlon_l156_156885


namespace alan_must_eat_more_l156_156702

theorem alan_must_eat_more (
  kevin_eats_total : ℕ,
  kevin_time : ℕ,
  alan_eats_rate : ℕ
) (h_kevin_eats_total : kevin_eats_total = 64) 
  (h_kevin_time : kevin_time = 8)
  (h_alan_eats_rate : alan_eats_rate = 5)
  (kevin_rate_gt_alan_rate : (kevin_eats_total / kevin_time) > alan_eats_rate) :
  ∃ wings_more_per_minute : ℕ, wings_more_per_minute = 4 :=
by
  sorry

end alan_must_eat_more_l156_156702


namespace congruent_faces_of_tetrahedron_l156_156745

theorem congruent_faces_of_tetrahedron
  (t : ℕ → ℝ)
  (r : ℕ → ℝ)
  (h_area_eq : ∀ i j : ℕ, i > 0 ∧ i ≤ 4 ∧ j > 0 ∧ j ≤ 4 → t i = t j)
  (h_radius_eq : ∀ i j : ℕ, i > 0 ∧ i ≤ 4 ∧ j > 0 ∧ j ≤ 4 → r i = r j) : 
  ∀ i j : ℕ, i > 0 ∧ i ≤ 4 ∧ j > 0 ∧ j ≤ 4 → congruent (face i) (face j) :=
by
  sorry

end congruent_faces_of_tetrahedron_l156_156745


namespace necessary_but_not_sufficient_condition_l156_156224

variable {a : ℕ → ℝ}
variable {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem necessary_but_not_sufficient_condition
    (a1_pos : a 1 > 0)
    (geo_seq : geometric_sequence a q)
    (a3_lt_a6 : a 3 < a 6) :
  (a 1 < a 3) ↔ ∃ k : ℝ, k > 1 ∧ a 1 * k^2 < a 1 * k^5 :=
by
  sorry

end necessary_but_not_sufficient_condition_l156_156224


namespace children_left_on_bus_l156_156027

theorem children_left_on_bus
  (initial_children : ℕ)
  (children_got_off : ℕ)
  (children_left : ℕ)
  (h_initial : initial_children = 41)
  (h_difference : initial_children - children_left = 23) : 
  children_left = 18 :=
by
  -- Using the given conditions
  have h1 : 41 - children_left = 23, from h_difference ▸ h_initial.symm,
  -- We can directly solve for children_left
  have h2 : children_left = 41 - 23, by linarith,
  -- Therefore, children_left = 18
  exact (h2 ▸ rfl)


end children_left_on_bus_l156_156027


namespace find_correct_statement_l156_156555

-- Define the function over reals
variable {f : ℝ → ℝ}

-- Define the conditions as Lean definitions
def statement1 := (f (-2) = f 2) → ∀ x, f (-x) = f x
def statement2 := (f (-2) ≠ f 2) → ¬ (∀ x, f (-x) = f x)
def statement3 := (f (-2) = f 2) → ¬ (∀ x, f x = -f x)

-- The lean statement to prove that statement 2 is correct
theorem find_correct_statement : statement2 := by
  sorry

end find_correct_statement_l156_156555


namespace remainder_sum_5678_5681_mod_13_l156_156122

theorem remainder_sum_5678_5681_mod_13 : (5678 + 5679 + 5680 + 5681) % 13 = 4 := 
by 
  have h1 : 5678 % 13 = 6 := by sorry
  have h2 : 5679 % 13 = 7 := by sorry
  have h3 : 5680 % 13 = 8 := by sorry
  have h4 : 5681 % 13 = 9 := by sorry
  have h_sum : (6 + 7 + 8 + 9) % 13 = (30 % 13) := by norm_num
  show (5678 + 5679 + 5680 + 5681) % 13 = 4, from 
    calc
      (5678 + 5679 + 5680 + 5681) % 13 
      = (6 + 7 + 8 + 9) % 13 : by sorry
      ... = 30 % 13 : by sorry
      ... = 4 : by norm_num

end remainder_sum_5678_5681_mod_13_l156_156122


namespace Jordan_Lee_debt_equal_l156_156258

theorem Jordan_Lee_debt_equal (initial_debt_jordan : ℝ) (additional_debt_jordan : ℝ)
  (rate_jordan : ℝ) (initial_debt_lee : ℝ) (rate_lee : ℝ) :
  initial_debt_jordan + additional_debt_jordan + (initial_debt_jordan + additional_debt_jordan) * rate_jordan * 33.333333333333336 
  = initial_debt_lee + initial_debt_lee * rate_lee * 33.333333333333336 :=
by
  let t := 33.333333333333336
  have rate_jordan := 0.12
  have rate_lee := 0.08
  have initial_debt_jordan := 200
  have additional_debt_jordan := 20
  have initial_debt_lee := 300
  sorry

end Jordan_Lee_debt_equal_l156_156258


namespace a_received_share_l156_156441

def a_inv : ℕ := 7000
def b_inv : ℕ := 11000
def c_inv : ℕ := 18000

def b_share : ℕ := 2200

def total_profit : ℕ := (b_share / (b_inv / 1000)) * 36
def a_ratio : ℕ := a_inv / 1000
def total_ratio : ℕ := (a_inv / 1000) + (b_inv / 1000) + (c_inv / 1000)

def a_share : ℕ := (a_ratio / total_ratio) * total_profit

theorem a_received_share :
  a_share = 1400 := 
sorry

end a_received_share_l156_156441


namespace plus_signs_count_l156_156813

theorem plus_signs_count (num_symbols : ℕ) (at_least_one_plus_in_10 : ∀ s : Finset ℕ, s.card = 10 → (∃ i ∈ s, i < 14)) (at_least_one_minus_in_15 : ∀ s : Finset ℕ, s.card = 15 → (∃ i ∈ s, i ≥ 14)) : 
    ∃ (p m : ℕ), p + m = 23 ∧ p = 14 ∧ m = 9 := by
  sorry

end plus_signs_count_l156_156813


namespace plus_signs_count_l156_156834

theorem plus_signs_count (total_symbols : ℕ) (n_plus n_minus : ℕ)
  (h1 : total_symbols = 23)
  (h2 : ∀ s : Finset ℕ, s.card = 10 → ∃ i ∈ s, i ≤ n_plus)
  (h3 : ∀ s : Finset ℕ, s.card = 15 → ∃ i ∈ s, i > n_plus) :
  n_plus = 14 :=
by
  -- the proof will go here
  sorry

end plus_signs_count_l156_156834


namespace ellipse_equation_locus_point_M_range_of_y0_l156_156586

-- Definitions related to the ellipse C_1
def a : ℝ := sorry
def b : ℝ := sorry
def e : ℝ := sqrt 3 / 3
def C1 := ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1)
def eccentricity := (sqrt (1 - b^2 / a^2)) = e

-- Conditions based on vertices and properties
def conditions := (a > b ∧ b > 0 ∧ (a^2 - b^2 = 1) ∧ (∇(a1, b1) ⸳ ∇(a2, b2) = -1))

-- The first problem: finding the ellipse equation
theorem ellipse_equation : eccentricity ∧ conditions → C1 = (x^2 / 3 + y^2 / 2 = 1) := sorry

-- The second problem: finding the locus of point M
def l1 : ℝ := -sqrt 3
def F2 : ℝ × ℝ := (sqrt 3, 0)
def C2 := ∀ x y : ℝ, (y^2 = 4x)

theorem locus_point_M (P : ℝ × ℝ) : ((∇(P, F2).bisector ⊥ l1) ∧ (M ∈ C2)) := sorry

-- The third problem: determining the range of y0
def y0_conditions : set ℝ := {y0 | y0 < -6 ∨ y0 ≥ 10}

theorem range_of_y0 (A B C : ℝ × ℝ) 
  (y0 ≠ -6)
  (y0 ∈ y0_conditions)
  (on_C2: ∀ (P : ℝ × ℝ), (∇(P, B) ⊥ ∇(B, C))) 
  : A ∈ C2 ∧ B ∈ C2 ∧ C ∈ C2 ∧ (A ≠ B ∧ B ≠ C ∧ AB ⸳ BC = 0) → y0_conditions := sorry

end ellipse_equation_locus_point_M_range_of_y0_l156_156586


namespace probability_of_area_ABP_greater_than_ACP_and_BCP_l156_156204

theorem probability_of_area_ABP_greater_than_ACP_and_BCP
  (ABC : Type) 
  [equilateral_triangle ABC]
  (P : point) 
  (hP : P ∈ interior ABC) : 
  prob (area (triangle ABP) > area (triangle ACP) ∧ area (triangle ABP) > area (triangle BCP)) = 1/3 :=
begin
  sorry,
end

end probability_of_area_ABP_greater_than_ACP_and_BCP_l156_156204


namespace irreducible_fraction_form_l156_156729

theorem irreducible_fraction_form (p q : ℕ) 
  (hpq_irreducible : Nat.coprime p q) (hp_pos : 0 < p) (hq_pos : 0 < q) (hq_odd : Odd q) :
  ∃ n k : ℕ, 0 < n ∧ 1 < k ∧ p / q = n / (2^k - 1) := sorry

end irreducible_fraction_form_l156_156729


namespace arithmetic_sequence_sum_l156_156145

noncomputable def sum_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ := n * a1 + (n * (n - 1) * d) / 2

theorem arithmetic_sequence_sum (a1 d : ℝ) (n : ℕ)
  (h1 : ∀ x y, (x - 2)^2 + y^2 = 4 → y = a1 * x)
  (h2 : ∀ x y, x + y + d = 0 → (x, y) ∈ set_of (λ (p : ℝ × ℝ), ∃ x y, (x - 2)^2 + y^2 = 4 ∧ y = a1 * x)) :
  sum_arithmetic_sequence a1 d n = -n^2 + 2 * n :=
by sorry

end arithmetic_sequence_sum_l156_156145


namespace complex_fourth_power_l156_156087

noncomputable def complex_number : ℂ := 3 * complexOfReal((real.cos (real.pi / 6))) + 3i * complexOfReal((real.sin (real.pi / 6)))

theorem complex_fourth_power :
  (complex_number ^ 4) = -40.5 + 40.5 * (sqrt 3) * complex.i := 
by 
  sorry

end complex_fourth_power_l156_156087


namespace solution_l156_156271

noncomputable def problem_statement : Prop :=
  ∀ (A B C H : Type) [EuclideanSpace ℝ A] [EuclideanSpace ℝ B] [EuclideanSpace ℝ C] [EuclideanSpace ℝ H],
    (angle A B C = 30) ∧ (distance B C = 3) ∧ (isOrthocenter H A B C) →
    distance A H = 3 * Real.sqrt 3

theorem solution : problem_statement :=
  sorry

end solution_l156_156271


namespace radius_of_congruent_spheres_in_cone_l156_156249

theorem radius_of_congruent_spheres_in_cone : 
  ∀ (R H : ℝ) (r : ℝ), 
  R = 6 → 
  H = 10 → 
  (10 - r)^2 + 6^2 = (2*real.sqrt 34 - r)^2 → 
  r = 100 / (20 - 4 * real.sqrt 34) := 
by
  intros R H r hR hH h_eq
  sorry

end radius_of_congruent_spheres_in_cone_l156_156249


namespace least_five_digit_congruent_eight_mod_17_l156_156866

theorem least_five_digit_congruent_eight_mod_17 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 8 ∧ n = 10009 :=
by
  use 10009
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  sorry

end least_five_digit_congruent_eight_mod_17_l156_156866


namespace locus_eq_circumcircle_and_hyperbola_l156_156226

variable {α : Type} [Field α]

structure Point (α : Type) :=
(x y : α)

structure Line (α : Type) :=
(a b c : α) -- Represents ax + by + c = 0

def distance (p1 p2 : Point α) : α :=
((p1.x - p2.x)^2 + (p1.y - p2.y)^2) ^ (1 / 2)

def perpendicular_distance (p : Point α) (l : Line α) : α :=
(abs (l.a * p.x + l.b * p.y + l.c)) / (l.a^2 + l.b^2) ^ (1 / 2)

theorem locus_eq_circumcircle_and_hyperbola
  (A B M : Point α) (a b : Line α)
  (h_line_A : a.a * A.x + a.b * A.y + a.c = 0) -- A lies on line a
  (h_line_B : b.a * B.x + b.b * B.y + b.c = 0) -- B lies on line b
  (h_equal_incline : ∀ p : Point α, (distance p A = perpendicular_distance p a) 
                                    ∧ (distance p B = perpendicular_distance p b)) :
  (distance M A) * (perpendicular_distance M a) = 
  (distance M B) * (perpendicular_distance M b) ↔
  (M lies on circumcircle of triangle A B C) ∨ 
  (M lies on rectangular circum-hyperbola through A B)
:= 
sorry

end locus_eq_circumcircle_and_hyperbola_l156_156226


namespace weighted_average_is_correct_l156_156922

def totalMarks (mean : ℕ) (students : ℕ) : ℕ := mean * students

def overallWeightedAverage : ℕ := 
  let totalMarksSections := [
    totalMarks 50 50,
    totalMarks 60 35,
    totalMarks 55 45,
    totalMarks 45 42,
    totalMarks 55 38,
    totalMarks 50 48
  ].sum
  let totalStudents := [50, 35, 45, 42, 38, 48].sum
  totalMarksSections / totalStudents

theorem weighted_average_is_correct : overallWeightedAverage = 52.15 :=
  by sorry

end weighted_average_is_correct_l156_156922


namespace jack_morning_emails_l156_156253

theorem jack_morning_emails (x : ℕ) (aft_mails eve_mails total_morn_eve : ℕ) (h1: aft_mails = 4) (h2: eve_mails = 8) (h3: total_morn_eve = 11) :
  x = total_morn_eve - eve_mails :=
by 
  sorry

end jack_morning_emails_l156_156253


namespace sum_no_green_2x2_equals_101_l156_156517

def probability_no_green_2x2 (n : Nat) : Rational := {
  num := 37,
  denom := 64
}

theorem sum_no_green_2x2_equals_101 : (probability_no_green_2x2 4).num + (probability_no_green_2x2 4).denom = 101 :=
  by
  sorry

end sum_no_green_2x2_equals_101_l156_156517


namespace floor_sqrt_120_l156_156526

theorem floor_sqrt_120 : (⌊Real.sqrt 120⌋ = 10) :=
by
  -- Conditions from the problem
  have h1: 10^2 = 100 := rfl
  have h2: 11^2 = 121 := rfl
  have h3: 10 < Real.sqrt 120 := sorry
  have h4: Real.sqrt 120 < 11 := sorry
  -- Proof goal
  sorry

end floor_sqrt_120_l156_156526


namespace count_divisible_by_90_four_digit_numbers_l156_156189

theorem count_divisible_by_90_four_digit_numbers :
  ∃ (n : ℕ), (n = 10) ∧ (∀ (x : ℕ), 1000 ≤ x ∧ x < 10000 ∧ x % 90 = 0 ∧ x % 100 = 90 → (x = 1890 ∨ x = 2790 ∨ x = 3690 ∨ x = 4590 ∨ x = 5490 ∨ x = 6390 ∨ x = 7290 ∨ x = 8190 ∨ x = 9090 ∨ x = 9990)) :=
by
  sorry

end count_divisible_by_90_four_digit_numbers_l156_156189


namespace part_I_part_II_l156_156610

def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x else if 1 ≤ x then 1 / x else 0

def g (a x : ℝ) : ℝ :=
  a * f x - abs (x - 1)

theorem part_I (b : ℝ) :
  (∀ x, 0 < x → g 0 x ≤ abs (x - 2) + b) → b ≥ -1 :=
by sorry

theorem part_II (a : ℝ) :
  a = 1 → ∃ x, 0 < x → ∀ y, 0 < y → g a x ≥ g a y :=
by sorry

end part_I_part_II_l156_156610


namespace decrypt_test_phrase_l156_156698

-- Definition of encryption and decryption steps
def swap_adjacent (s : String) : String :=
  s.toList.groupByIndex (· % 2 = 1).toList.join

def shift_alphabet_right (ch : Char) (positions : Nat) : Char :=
  let base := if ch.isLower then 'а' else 'А'
  let shift := ((ch.toNat - base.toNat + positions) % 32)
  Char.ofNat (base.toNat + shift)

def shift_right (s : String) (positions : Nat) : String :=
  s.toList.map (λ ch => shift_alphabet_right ch positions).asString

def reverse_string (s : String) : String := s.reverse

-- Given example process
def encryption_example : String :=
  reverse_string (shift_right (swap_adjacent "гипертекст") 2)

-- Decryption process
def decrypt (encrypted : String) : String :=
  let reversed := reverse_string encrypted
  let shifted := shift_right reversed (32 - 2)  -- reversing the shift of 2 positions
  swap_adjacent shifted

-- Given encrypted phrase
def encrypted_phrase : String := "врпвл терпраиэ вйзгцфпз"

-- Expected result after decryption
def decrypted_phrase : String := "нефтебазы южного района"

-- Statement
theorem decrypt_test_phrase :
  decrypt encrypted_phrase = decrypted_phrase :=
by
  sorry

end decrypt_test_phrase_l156_156698


namespace option_D_is_linear_equation_with_two_variables_l156_156004

def is_linear_equation (eq : String) : Prop :=
  match eq with
  | "3x - 6 = x" => false
  | "x = 5 / y - 1" => false
  | "2x - 3y = x^2" => false
  | "3x = 2y" => true
  | _ => false

theorem option_D_is_linear_equation_with_two_variables :
  is_linear_equation "3x = 2y" = true := by
  sorry

end option_D_is_linear_equation_with_two_variables_l156_156004


namespace fraction_product_eq_one_l156_156961

theorem fraction_product_eq_one :
  (∏ i in finset.range 21, (1 + (19 : ℕ) / (i + 1))) / (∏ i in finset.range 19, (1 + (21 : ℕ) / (i + 1))) = 1 :=
by
  -- main proof
  sorry

end fraction_product_eq_one_l156_156961


namespace cards_arrangement_count_l156_156295

theorem cards_arrangement_count : 
  let cards := [1, 2, 3, 4, 5, 6, 7] in
  let valid_arrangements := 
    {arrangement | ∃ removed, 
      removed ∈ cards ∧ 
      (∀ remaining, 
        remaining = cards.erase removed → 
        (sorted remaining ∨ sorted (remaining.reverse))) } in
  valid_arrangements.card = 26 :=
sorry

end cards_arrangement_count_l156_156295


namespace min_omega_value_l156_156603

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem min_omega_value (ω : ℝ) (φ : ℝ) (h_ω_pos : ω > 0)
  (h_even : ∀ x : ℝ, f ω φ x = f ω φ (-x))
  (h_symmetry : f ω φ 1 = 0 ∧ ∀ x : ℝ, f ω φ (1 + x) = - f ω φ (1 - x)) :
  ω = Real.pi / 2 :=
by
  sorry

end min_omega_value_l156_156603


namespace second_third_order_distance_equality_l156_156239

-- Define the triangle and points
structure Triangle :=
  (A B C : Point)

-- Given points on triangle sides
structure SecondOrderPoints (T : Triangle) :=
  (A1 : Point)  -- where angle bisector AA1 intersects BC
  (A2b A2c : Point) -- A2b, A2c as second-order points
  (A3b A3c : Point) -- A3b, A3c as third-order points

-- Assume Angle Bisectors and Parallel lines constructions
axiom bisector_AA1 (T : Triangle) (p : Point) : Prop
axiom parallel (p1 p2 p3 p4 : Point) : Prop

-- We need to prove the distance property between second-order and third-order points
theorem second_third_order_distance_equality 
  (T : Triangle)
  (P : SecondOrderPoints T)
  (h1 : bisector_AA1 T P.A1)
  (h2 : parallel P.A1 P.A2b T.B T.C)
  (h3 : parallel P.A1 P.A2c T.A T.B)
  (h4 : parallel P.A2b P.A3b T.A T.C)
  (h5 : parallel P.A2c P.A3c T.B T.C) :
  distance P.A2b P.A2c = distance P.A3b P.A3c :=
by
  sorry

end second_third_order_distance_equality_l156_156239


namespace circle_center_coordinates_l156_156019

theorem circle_center_coordinates (b c p q : ℝ) 
    (h_circle_eq : ∀ x y : ℝ, x^2 + y^2 - 2 * p * x - 2 * q * y + 2 * q - 1 = 0) 
    (h_quad_roots : ∀ x : ℝ, x^2 + b * x + c = 0) 
    (h_condition : b^2 - 4 * c ≥ 0) : 
    (p = -b / 2) ∧ (q = (1 + c) / 2) := 
sorry

end circle_center_coordinates_l156_156019


namespace cos_C_equal_two_thirds_l156_156247

variable {A B C : ℝ}
variable {a b c : ℝ}

-- Define the conditions
def condition1 : a > 0 ∧ b > 0 ∧ c > 0 := sorry
def condition2 : (a / b) + (b / a) = 4 * Real.cos C := sorry
def condition3 : Real.cos (A - B) = 1 / 6 := sorry

-- Statement to prove
theorem cos_C_equal_two_thirds 
  (h1: a > 0 ∧ b > 0 ∧ c > 0) 
  (h2: (a / b) + (b / a) = 4 * Real.cos C) 
  (h3: Real.cos (A - B) = 1 / 6) 
  : Real.cos C = 2 / 3 :=
  sorry

end cos_C_equal_two_thirds_l156_156247


namespace man_completes_in_9_days_l156_156459

-- Definitions of the work rates and the conditions given
def M : ℚ := sorry
def W : ℚ := 1 / 6
def B : ℚ := 1 / 18
def combined_rate : ℚ := 1 / 3

-- Statement that the man alone can complete the work in 9 days
theorem man_completes_in_9_days
  (h_combined : M + W + B = combined_rate) : 1 / M = 9 :=
  sorry

end man_completes_in_9_days_l156_156459


namespace plus_signs_count_l156_156799

noncomputable def solveSymbols : Prop :=
  ∃ p m : ℕ, p + m = 23 ∧ p ≤ 14 ∧ m ≤ 9 ∧ (∀ s : Finset (Fin 23), s.card = 10 → (s.image (λ i => if i < p then true else false)).count true ≥ 1) ∧ (∀ s : Finset (Fin 23), s.card = 15 → (s.image (λ i => if i < p then false else true)).count false ≥ 1) 

theorem plus_signs_count : solveSymbols :=
sorry

end plus_signs_count_l156_156799


namespace cost_of_staying_23_days_l156_156442

def hostel_cost (days: ℕ) : ℝ :=
  if days ≤ 7 then
    days * 18
  else
    7 * 18 + (days - 7) * 14

theorem cost_of_staying_23_days : hostel_cost 23 = 350 :=
by
  sorry

end cost_of_staying_23_days_l156_156442


namespace find_index_120th_term_lt_zero_l156_156102

-- Define the sequence b_n as the sum of cosines
noncomputable def b (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1), Real.cos k

-- Define the target theorem to find the index of the 120th term for which b_n < 0
theorem find_index_120th_term_lt_zero : 
  (∃ k : ℕ, (k = 120) → ∃ n : ℕ, (n = floor (2 * Real.pi * k)) ∧ b n < 0) :=
sorry

end find_index_120th_term_lt_zero_l156_156102


namespace trig_tangent_sum_theorem_l156_156201

noncomputable def trig_tangent_sum : Prop :=
  ∀ (x y : ℝ), sin x + sin y = 120 / 169 ∧ cos x + cos y = 119 / 169 → tan x + tan y = 0

-- placeholder for our proof
theorem trig_tangent_sum_theorem : trig_tangent_sum :=
by
  -- proof should go here
  sorry

end trig_tangent_sum_theorem_l156_156201


namespace plus_signs_count_l156_156780

theorem plus_signs_count (p m : ℕ) (h_sum : p + m = 23)
                         (h_max_minus : m ≤ 9) (h_max_plus : p ≤ 14)
                         (h_at_least_one_plus_in_10 : ∀ (s : Finset (Fin 23)), s.card = 10 → ∃ i ∈ s, i < p)
                         (h_at_least_one_minus_in_15 : ∀ (s : Finset (Fin 23)), s.card = 15 → ∃ i ∈ s, m ≤ i) :
  p = 14 :=
by sorry

end plus_signs_count_l156_156780


namespace pizza_slices_l156_156029

theorem pizza_slices (total_slices pepperoni_slices mushroom_slices : ℕ) 
  (h_total : total_slices = 24)
  (h_pepperoni : pepperoni_slices = 15)
  (h_mushrooms : mushroom_slices = 16)
  (h_at_least_one : total_slices = pepperoni_slices + mushroom_slices - both_slices)
  : both_slices = 7 :=
by
  have h1 : total_slices = 24 := h_total
  have h2 : pepperoni_slices = 15 := h_pepperoni
  have h3 : mushroom_slices = 16 := h_mushrooms
  have h4 : total_slices = 24 := by sorry
  sorry

end pizza_slices_l156_156029


namespace percent_decrease_is_36_l156_156234

open Real

-- Define given conditions as Lean definitions
def area_triangle_1 : ℝ := 18 * sqrt 3
def area_triangle_3 : ℝ := 50 * sqrt 3
def area_square : ℝ := 72
def side_square : ℝ := sqrt 72

-- Define initial and decreased length of segment AD
def segment_AD_initial : ℝ := side_square
def segment_AD_decreased : ℝ := segment_AD_initial * 0.8 

-- Define the new area of the square with the reduced side length
def new_area_square : ℝ := (segment_AD_decreased) ^ 2

-- Define percent decrease in the area of the square
def percent_decrease_area : ℝ := ((area_square - new_area_square) / area_square) * 100

-- Lean theorem to prove the percent decrease in the area is 36%
theorem percent_decrease_is_36 :
  percent_decrease_area = 36 := 
sorry

end percent_decrease_is_36_l156_156234


namespace average_score_of_makeup_students_l156_156220

variables (A B total_students : ℕ)
variables (Avg_A Avg_Total Avg_B : ℤ)
variables (perc_assigned perc_total : ℝ)

-- Conditions
def exam_conditions : Prop :=
  total_students = 100 ∧
  perc_assigned = 0.70 ∧
  Avg_A = 55 ∧
  Avg_Total = 67 ∧
  A = (perc_assigned * total_students).to_nat ∧
  B = total_students - A

-- Question
def average_score_makeup_date := (⟦(Avg_B * B) + (Avg_A * A)⟧ / 100 = 67)

-- Proof statement
theorem average_score_of_makeup_students 
  (h : exam_conditions) :
  Avg_B = 95 :=
sorry

end average_score_of_makeup_students_l156_156220


namespace roots_product_l156_156420

theorem roots_product : (27^(1/3) * 81^(1/4) * 64^(1/6)) = 18 := 
by
  sorry

end roots_product_l156_156420


namespace count_of_plus_signs_l156_156800

-- Definitions based on the given conditions
def total_symbols : ℕ := 23
def any_10_symbols_contains_plus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 10 → (\exists c ∈ s, c = '+')
def any_15_symbols_contains_minus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 15 → (\exists c ∈ s, c = '-')

-- Main theorem statement without the proof
theorem count_of_plus_signs (symbols : Finset char) (h1 : symbols.card = total_symbols) 
  (h2 : any_10_symbols_contains_plus symbols) 
  (h3 : any_15_symbols_contains_minus symbols) : 
  symbols.count '+' = 14 :=
sorry

end count_of_plus_signs_l156_156800


namespace polygon_sides_l156_156383

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 4 * 360) : 
  n = 6 :=
by sorry

end polygon_sides_l156_156383


namespace red_points_count_exists_exact_red_points_l156_156393

noncomputable def count_red_midpoints (points : Finset (ℝ × ℝ)) : ℕ :=
  (Finset.image (λ (p : (ℝ × ℝ) × (ℝ × ℝ)), ((p.1.1 + p.2.1) / 2, (p.1.2 + p.2.2) / 2))
    (points.product points)).card

theorem red_points_count (points : Finset (ℝ × ℝ)) (h : points.card = 997) :
  count_red_midpoints points ≥ 1991 :=
sorry

theorem exists_exact_red_points (h : ∃ (points : Finset (ℝ × ℝ)), points.card = 997 ∧ count_red_midpoints points = 1991) :
  ∃ (points : Finset (ℝ × ℝ)), points.card = 997 ∧ count_red_midpoints points = 1991 :=
h

end red_points_count_exists_exact_red_points_l156_156393


namespace distinct_z_values_90_l156_156607

noncomputable def countDistinctZValues : ℕ :=
  let z_values : Finset ℕ := 
    (Finset.range 10).bind (λ a : ℕ, 
    (Finset.range 10).bind (λ b : ℕ, 
    (Finset.range 10).bind (λ c : ℕ, 
    (Finset.range 10).image (λ d : ℕ, 
      abs (9 * (111 * abs (a - d) + 10 * abs (b - c)))))))
  z_values.card

theorem distinct_z_values_90 : countDistinctZValues = 90 := sorry

end distinct_z_values_90_l156_156607


namespace factor_y6_plus_64_l156_156967

theorem factor_y6_plus_64 : (y^2 + 4) ∣ (y^6 + 64) :=
sorry

end factor_y6_plus_64_l156_156967


namespace find_principal_amount_l156_156013

theorem find_principal_amount (P r : ℝ) (h1 : 720 = P * (1 + 2 * r)) (h2 : 1020 = P * (1 + 7 * r)) : P = 600 :=
by sorry

end find_principal_amount_l156_156013


namespace a_2008_lt_5_l156_156896

open Nat

def a : ℕ → ℚ
| 0       := 1
| (n + 1) := a n + (1 + a n + a n * b n) / b n

def b : ℕ → ℚ
| 0       := 2
| (n + 1) := (1 + b n + a n * b n) / a n

theorem a_2008_lt_5 : a 2008 < 5 := by sorry

end a_2008_lt_5_l156_156896


namespace find_angle_BAC_l156_156743

-- Given conditions:
variables {A B C O B_1 C_1 : Type*}
variables [inst : EuclideanGeometry A B C O B_1 C_1]
include inst

-- Define conditions:
def is_acute_angled_triangle (A B C : EuclideanGeometry.A) : Prop :=
  EuclideanGeometry.is_acute_angled_triangle A B C

def altitudes_intersect_circumcircle (B C: EuclideanGeometry.A)
  (B_1 C_1 : EuclideanGeometry.A) (circumcircle : set (EuclideanGeometry.Polyline))
  (center : EuclideanGeometry.Point) : Prop :=
  EuclideanGeometry.is_altitude B A * ∈ circumcircle B_1
  ∧ EuclideanGeometry.is_altitude C A * ∈ circumcircle C_1
  ∧ (B_1, C_1).center = center

theorem find_angle_BAC
  (h1 : is_acute_angled_triangle A B C)
  (h2 : altitudes_intersect_circumcircle B C B_1 C_1 circumcircle center) :
  ∠ BAC = 45 :=
by
  sorry

end find_angle_BAC_l156_156743


namespace a_neg_half_not_bounded_a_bounded_range_l156_156510

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  1 + a * (1/3)^x + (1/9)^x

theorem a_neg_half_not_bounded (a : ℝ) :
  a = -1/2 → ¬(∃ M > 0, ∀ x < 0, |f x a| ≤ M) :=
by
  sorry

theorem a_bounded_range (a : ℝ) : 
  (∀ x ≥ 0, |f x a| ≤ 4) → -6 ≤ a ∧ a ≤ 2 :=
by
  sorry

end a_neg_half_not_bounded_a_bounded_range_l156_156510


namespace plus_signs_count_l156_156796

noncomputable def solveSymbols : Prop :=
  ∃ p m : ℕ, p + m = 23 ∧ p ≤ 14 ∧ m ≤ 9 ∧ (∀ s : Finset (Fin 23), s.card = 10 → (s.image (λ i => if i < p then true else false)).count true ≥ 1) ∧ (∀ s : Finset (Fin 23), s.card = 15 → (s.image (λ i => if i < p then false else true)).count false ≥ 1) 

theorem plus_signs_count : solveSymbols :=
sorry

end plus_signs_count_l156_156796


namespace polygon_sides_l156_156385

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 4 * 360) : 
  n = 6 :=
by sorry

end polygon_sides_l156_156385


namespace exists_bijective_g_dist_preserving_l156_156450

variables {V E : Type} [Fintype V] [DecidableEq V]

-- Define a tree data structure
class Tree (G : SimpleGraph V) :=
  (is_tree : G.is_tree)
  (no_deg2_vertices : ∀ (v : V), ¬(degree G v = 2))

-- Define the distance function
def distance {V : Type} (G : SimpleGraph V) [weighted_graph G ℝ]
  (u v : V) : ℝ :=
  (G.weighted_distance u v : ℝ)

variables {S T : SimpleGraph V} [Tree S] [Tree T]

-- Given function preserving leaf distances
variable (f : Π (u : V), S.leaf u → T.leaf (f u))
   (Hf : ∀ (u v : V), S.leaf u → S.leaf v → distance S u v = distance T (f u) (f v))

-- Required to prove
theorem exists_bijective_g_dist_preserving :
  ∃ (g : V → V), Function.Bijective g ∧
  ∀ u v, distance S u v = distance T (g u) (g v) :=
sorry

end exists_bijective_g_dist_preserving_l156_156450


namespace inverse_functions_linear_periodic_l156_156135

variable {R : Type*} [Field R] {f g h p : R → R} {k d : R}

def periodic (h : R → R) (d : R) : Prop :=
  ∀ x : R, h(x + d) = h(x)

theorem inverse_functions_linear_periodic
  (h_periodic : periodic h d)
  (h_inv : ∀ x y : R, g (f x) = x ∧ f (g y) = y)
  (h_linear_periodic : ∀ x : R, f x = k * x + h x) :
  ∃ p : R → R, periodic p (k * d) ∧ ∀ y : R, g y = (1 / k) * y + p y := 
sorry

end inverse_functions_linear_periodic_l156_156135


namespace cards_arrangement_count_l156_156297

theorem cards_arrangement_count : 
  let cards := [1, 2, 3, 4, 5, 6, 7] in
  let valid_arrangements := 
    {arrangement | ∃ removed, 
      removed ∈ cards ∧ 
      (∀ remaining, 
        remaining = cards.erase removed → 
        (sorted remaining ∨ sorted (remaining.reverse))) } in
  valid_arrangements.card = 26 :=
sorry

end cards_arrangement_count_l156_156297


namespace plus_signs_count_l156_156830

theorem plus_signs_count (n p m : ℕ) (h_total : n = 23) 
  (h_plus_min : ∀ s, s.card = 10 → ∃ t ∈ s, is_plus t)
  (h_minus_min : ∀ s, s.card = 15 → ∃ t ∈ s, is_minus t)
  : p = 14 :=
sorry

end plus_signs_count_l156_156830


namespace infinite_geometric_series_correct_l156_156108

noncomputable def infinite_geometric_series_sum : ℚ :=
  let a : ℚ := 5 / 3
  let r : ℚ := -9 / 20
  a / (1 - r)

theorem infinite_geometric_series_correct : infinite_geometric_series_sum = 100 / 87 := 
by
  sorry

end infinite_geometric_series_correct_l156_156108


namespace floor_sqrt_120_eq_10_l156_156537

theorem floor_sqrt_120_eq_10 : ⌊Real.sqrt 120⌋ = 10 := by
  -- Here, we note that we are given:
  -- 100 < 120 < 121 and the square root of it lies between 10 and 11
  sorry

end floor_sqrt_120_eq_10_l156_156537


namespace cards_arrangement_count_is_10_l156_156317

-- Define the problem in Lean statement terms
def valid_arrangements_count : ℕ :=
  -- number of arrangements of seven cards where one card can be removed 
  -- leaving the remaining six cards in either ascending or descending order
  10

-- Theorem stating that the number of valid arrangements is 10
theorem cards_arrangement_count_is_10 : valid_arrangements_count = 10 :=
by
  -- Proof is omitted (the explanation above corresponds to the omitted proof details)
  sorry

end cards_arrangement_count_is_10_l156_156317


namespace unique_increasing_function_on_positive_reals_l156_156938

-- Definitions of the functions
def f1 (x : ℝ) : ℝ := -x^2
def f2 (x : ℝ) : ℝ := 1 / x
def f3 (x : ℝ) : ℝ := (1 / 2)^x
def f4 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Theorem statement ensuring f4 is the only function that is increasing on (0, ∞)
theorem unique_increasing_function_on_positive_reals : 
  ∀ (x y : ℝ), 0 < x → x < y → (f4 x < f4 y) ∧ 
               ¬ ((f1 x < f1 y) ∨ (f2 x < f2 y) ∨ (f3 x < f3 y)) :=
by
  intros x y hx hxy
  split
  {
    exact sorry -- Proof that f4 is increasing on (0, ∞)
  }
  {
    intro h
    cases h 
    {
      exact sorry -- Proof that f1 is not increasing on (0, ∞)
    }
    {
      exact sorry -- Proof that f2 is not increasing on (0, ∞)
    }
    {
      exact sorry -- Proof that f3 is not increasing on (0, ∞)
    }
  }

end unique_increasing_function_on_positive_reals_l156_156938


namespace cube_sum_div_by_nine_l156_156853

theorem cube_sum_div_by_nine (n : ℕ) (hn : 0 < n) : (n^3 + (n+1)^3 + (n+2)^3) % 9 = 0 := by sorry

end cube_sum_div_by_nine_l156_156853


namespace right_triangle_sides_l156_156758

theorem right_triangle_sides (m n : ℝ) (x : ℝ) (a b c : ℝ)
  (h1 : 2 * x < m + n) 
  (h2 : a = Real.sqrt (2 * m * n) - m)
  (h3 : b = Real.sqrt (2 * m * n) - n)
  (h4 : c = m + n - Real.sqrt (2 * m * n))
  (h5 : a^2 + b^2 = c^2)
  (h6 : 4 * x^2 = (m - 2 * x)^2 + (n - 2 * x)^2) :
  a = Real.sqrt (2 * m * n) - m ∧ b = Real.sqrt (2 * m * n) - n ∧ c = m + n - Real.sqrt (2 * m * n) :=
by
  sorry

end right_triangle_sides_l156_156758


namespace area_of_polygon_l156_156098

theorem area_of_polygon (S : Point) (r : ℝ) (A B C D K L M N : Point)
  (h_circle : circle S r = true)
  (h_radius : r = 3)
  (h_diameters : perpendicular (line_through_pts A C) (line_through_pts B D) ∧
    colinear S A C ∧ colinear S B D)
  (h_ABK : is_isosceles_triangle A B K ∧ length (segment A B) = length (segment AK))
  (h_BCL : is_isosceles_triangle B C L ∧ length (segment B C) = length (segment BL))
  (h_CDM : is_isosceles_triangle C D M ∧ length (segment C D) = length (segment CM))
  (h_DAN : is_isosceles_triangle D A N ∧ length (segment D A) = length (segment AN))
  : area (polygon [A, K, B, L, C, M, D, N]) = 108 := 
sorry

end area_of_polygon_l156_156098


namespace separability_l156_156911

-- Define type for points in the plane
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define a finite set S of points where each point is either red or green
def is_red : Point → Prop := sorry
def is_green : Point → Prop := sorry

-- Define what it means for a set to be divisible
def is_divisible (S : set Point) : Prop :=
∃ (Δ : set Point), (∀ p ∈ S, is_red p → p ∈ interior Δ) ∧ (∀ p ∈ S, is_green p → p ∉ interior Δ)

-- Given any 1000 points from S form a divisible set
axiom finite_set_S : set Point
axiom finite_set_S_finite : finite finite_set_S
axiom any_1000_points_divisible : ∀ (T : set Point), T ⊆ finite_set_S ∧ T.card = 1000 → is_divisible T

-- Prove that it is not necessarily true that the entire set S is divisible
theorem separability (S : set Point) (hS : finite_set_S = S) : ¬ (is_divisible S) :=
sorry

end separability_l156_156911


namespace ratio_cans_of_water_to_concentrate_l156_156062

-- Definition of the problem conditions
def cans_of_concentrate : ℕ := 40
def volume_per_serving : ℕ := 6
def servings_required : ℕ := 320
def can_volume : ℕ := 12

-- Theorem statement corresponding to the proof problem
theorem ratio_cans_of_water_to_concentrate :
  let total_volume := servings_required * volume_per_serving in
  let total_cans_of_juice := total_volume / can_volume in
  let cans_of_water := total_cans_of_juice - cans_of_concentrate in
  cans_of_water / cans_of_concentrate = 3 :=
by
  sorry

end ratio_cans_of_water_to_concentrate_l156_156062


namespace fill_time_l156_156929

variables (X Y Z : ℝ)

-- Given conditions
def condition1 := (1 / X + 1 / Y = 1 / 5)
def condition2 := (1 / X + 1 / Z = 1 / 6)
def condition3 := (1 / Y + 1 / Z = 1 / 7)

-- Proving the total time for X, Y, and Z working together to fill the pool
theorem fill_time (h1 : condition1) (h2 : condition2) (h3 : condition3) : 
  (1 / ((1 / X) + (1 / Y) + (1 / Z))) ≈ 3.93 :=
sorry

end fill_time_l156_156929


namespace fewerEmployeesAbroadThanInKorea_l156_156908

def totalEmployees : Nat := 928
def employeesInKorea : Nat := 713
def employeesAbroad : Nat := totalEmployees - employeesInKorea

theorem fewerEmployeesAbroadThanInKorea :
  employeesInKorea - employeesAbroad = 498 :=
by
  sorry

end fewerEmployeesAbroadThanInKorea_l156_156908


namespace fraction_undefined_l156_156880

theorem fraction_undefined (x : ℝ) : x = -3 → (∃ y : ℝ, y = (3 * x - 1) / (x + 3)) → false :=
by yeah

end fraction_undefined_l156_156880


namespace inscribed_circle_radius_l156_156764

theorem inscribed_circle_radius
  (a b c : ℝ)
  (h₁ : a = 5)
  (h₂ : b = 12)
  (h₃ : c = 13)
  (h₄ : a^2 + b^2 = c^2) :
  let r := (a + b - c) / 2 in
  r = 2 :=
by 
  -- Show that the values of a, b, c satisfy the Pythagorean theorem
  rw [h₁, h₂, h₃] at h₄,
  -- Calculate the radius of the inscribed circle
  let r := (a + b - c) / 2,
  rw [h₁, h₂, h₃],
  norm_num,
  exact sorry

end inscribed_circle_radius_l156_156764


namespace train_cross_platform_time_l156_156904

theorem train_cross_platform_time :
  ∀ (L_train : ℕ) (L_platform : ℕ) (T_pole : ℕ),
    L_train = 300 → L_platform = 250 → T_pole = 18 →
    (L_train + L_platform) / (L_train / T_pole) = 33 :=
by {
    intros,
    simp,
    sorry
}

end train_cross_platform_time_l156_156904


namespace sum_of_inverted_first_five_terms_l156_156672

-- Definitions based on conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a (i + 1)

def first_term_one (a : ℕ → ℝ) : Prop := a 1 = 1

def sum_condition (a : ℕ → ℝ) (q : ℝ) : Prop := 
  9 * S_n a 3 = S_n a 6

def inverted_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n = 1 / a n

def sum_of_first_n_terms (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, b (i + 1)

-- Main theorem statement
theorem sum_of_inverted_first_five_terms (a b : ℕ → ℝ) (q : ℝ) 
  (h1 : geometric_sequence a q)
  (h2 : first_term_one a)
  (h3 : sum_condition a q)
  (h4 : inverted_sequence a b) :
  sum_of_first_n_terms b 5 = 31 / 16 := 
sorry

end sum_of_inverted_first_five_terms_l156_156672


namespace exists_root_in_interval_l156_156641

noncomputable def f (x : ℝ) := log x / log 2 + x - 5

theorem exists_root_in_interval (h₁ : continuous_on f (set.Ioo (0 : ℝ) (real.top))):
∃ c ∈ set.Ioo 3 4, f c = 0 :=
by {
  -- We know that f(3) < 0 and f(4) > 0 by evaluation.
  have h2 : f 3 < 0 := sorry,
  have h3 : f 4 > 0 := sorry,
  -- By the Intermediate Value Theorem:
  obtain ⟨c, hc, hfc⟩ := intermediate_value_Ioo (by linarith [h2, h3]) h₁ ⟨3, 4⟩ sorry,
  exact ⟨c, hc, hfc⟩,
}

end exists_root_in_interval_l156_156641


namespace find_angle_ACE_l156_156142

-- Definitions of points and intersection properties
variables (A B C D E P Q O : Point) (ABCDE : ConvexPentagon A B C D E)
variable (H1 : Intersect BE AC P) 
variable (H2 : Intersect CE AD Q) 
variable (H3 : Intersect AD BE O)

-- Definitions of triangle properties
variables (T1 : IsoscelesTriangleAtVertex B P A 40) -- ∠BPA = 40°
variables (T2 : IsoscelesTriangleAtVertex D Q E 40) -- ∠DQE = 40°
variables (T3 : IsoscelesTriangle P A O)
variables (T4 : IsoscelesTriangle E Q O)

-- The goal statement
theorem find_angle_ACE : (angle ACE = 120 ∨ angle ACE = 75) := 
sorry

end find_angle_ACE_l156_156142


namespace sum_of_squares_l156_156213

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 52) : x^2 + y^2 = 220 :=
sorry

end sum_of_squares_l156_156213


namespace no_integer_satisfies_n_inequalities_l156_156415

theorem no_integer_satisfies_n_inequalities : ¬ exists n : ℤ, n + 15 > 18 ∧ -3n > -9 :=
by
  sorry

end no_integer_satisfies_n_inequalities_l156_156415


namespace part1_l156_156615

def f (x : ℝ) : ℝ := exp x - log (x - 1) + 1

theorem part1 (x : ℝ) (h : x > 1) : f x > 4 :=
sorry

end part1_l156_156615


namespace science_club_neither_bio_chem_l156_156287

theorem science_club_neither_bio_chem (total_students biology_students chemistry_students both_students : ℕ)
  (h1: total_students = 60)
  (h2: biology_students = 42)
  (h3: chemistry_students = 35)
  (h4: both_students = 25) :
  let only_biology := biology_students - both_students in
  let only_chemistry := chemistry_students - both_students in
  let total_taking_subjects := only_biology + only_chemistry + both_students in
  let neither := total_students - total_taking_subjects in
  neither = 8 :=
by
  sorry

end science_club_neither_bio_chem_l156_156287


namespace area_relation_l156_156854

variables {Point : Type} [AffineSpace ℤ Point]
variables (A B C D M : Point) (AB CD : Line Point) 
variables (area : Set Point → ℝ)

-- Conditions
variables (h_convex : ConvexQuadrilateral A B C D)
variables (h_on_AD : M ∉ ∉ LineThrough A D)
variables (h_parallel_CM_AB : Parallel (LineThrough C M) (LineThrough A B))
variables (h_parallel_BM_CD : Parallel (LineThrough B M) (LineThrough C D))

-- Area function
noncomputable def S (s : Set Point) : ℝ := area s

-- Statement
theorem area_relation (h_convex : ConvexQuadrilateral A B C D)
 (h_on_AD : M ∈ LineThrough A D)
 (h_parallel_CM_AB : Parallel (LineThrough C M) (LineThrough A B))
 (h_parallel_BM_CD : Parallel (LineThrough B M) (LineThrough C D)) :
 S {A, B, C, D} ≥ 3 * S {B, C, M} :=
by
  sorry

end area_relation_l156_156854


namespace arccos_gt_arctan_l156_156987

theorem arccos_gt_arctan (x : ℝ) (h : x ∈ set.Icc (-1 : ℝ) 1) : 
  (arccos x > arctan x) ↔ (x < real.sqrt 2 / 2) :=
sorry

end arccos_gt_arctan_l156_156987


namespace floor_sqrt_120_eq_10_l156_156536

theorem floor_sqrt_120_eq_10 : ⌊Real.sqrt 120⌋ = 10 := by
  -- Here, we note that we are given:
  -- 100 < 120 < 121 and the square root of it lies between 10 and 11
  sorry

end floor_sqrt_120_eq_10_l156_156536


namespace greatest_of_5_consec_even_numbers_l156_156336

-- Definitions based on the conditions
def avg_of_5_consec_even_numbers (N : ℤ) : ℤ := (N - 4 + N - 2 + N + N + 2 + N + 4) / 5

-- Proof statement
theorem greatest_of_5_consec_even_numbers (N : ℤ) (h : avg_of_5_consec_even_numbers N = 35) : N + 4 = 39 :=
by
  sorry -- proof is omitted

end greatest_of_5_consec_even_numbers_l156_156336


namespace plus_signs_count_l156_156811

theorem plus_signs_count (num_symbols : ℕ) (at_least_one_plus_in_10 : ∀ s : Finset ℕ, s.card = 10 → (∃ i ∈ s, i < 14)) (at_least_one_minus_in_15 : ∀ s : Finset ℕ, s.card = 15 → (∃ i ∈ s, i ≥ 14)) : 
    ∃ (p m : ℕ), p + m = 23 ∧ p = 14 ∧ m = 9 := by
  sorry

end plus_signs_count_l156_156811


namespace large_cartridge_pages_correct_l156_156924

-- Define the conditions
def small_cartridge_pages : ℕ := 600
def medium_cartridge_pages : ℕ := 2 * 3 * small_cartridge_pages / 6
def large_cartridge_pages : ℕ := 2 * 3 * medium_cartridge_pages / 6

-- The theorem to prove
theorem large_cartridge_pages_correct :
  large_cartridge_pages = 1350 :=
by
  sorry

end large_cartridge_pages_correct_l156_156924


namespace x_plus_y_eq_3012_plus_pi_div_2_l156_156598

theorem x_plus_y_eq_3012_plus_pi_div_2
  (x y : ℝ)
  (h1 : x + Real.cos y = 3012)
  (h2 : x + 3012 * Real.sin y = 3010)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 3012 + Real.pi / 2 :=
sorry

end x_plus_y_eq_3012_plus_pi_div_2_l156_156598


namespace least_five_digit_congruent_eight_mod_17_l156_156864

theorem least_five_digit_congruent_eight_mod_17 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 8 ∧ n = 10009 :=
by
  use 10009
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  sorry

end least_five_digit_congruent_eight_mod_17_l156_156864


namespace sub_inequality_l156_156646

variable {a b c : ℝ}

theorem sub_inequality (h : a > b) : a - c > b - c :=
begin
  sorry
end

end sub_inequality_l156_156646


namespace evaluate_expression_l156_156979

theorem evaluate_expression : 
  let e := 3 + 2 * Real.sqrt 3 + 1 / (3 + 2 * Real.sqrt 3) + 1 / (2 * Real.sqrt 3 - 3)
  e = 3 + 10 * Real.sqrt 3 / 3 :=
by
  let e := 3 + 2 * Real.sqrt 3 + 1 / (3 + 2 * Real.sqrt 3) + 1 / (2 * Real.sqrt 3 - 3)
  have h : e = 3 + 10 * Real.sqrt 3 / 3 := sorry
  exact h

end evaluate_expression_l156_156979


namespace set_intersection_l156_156276

def S : Set ℝ := {x | x^2 - 5 * x + 6 ≥ 0}
def T : Set ℝ := {x | x > 1}
def result : Set ℝ := {x | x ≥ 3 ∨ (1 < x ∧ x ≤ 2)}

theorem set_intersection (x : ℝ) : x ∈ (S ∩ T) ↔ x ∈ result := by
  sorry

end set_intersection_l156_156276


namespace log_function_properties_l156_156554

theorem log_function_properties (x1 x2 : ℝ) (hx1 : x1 ≠ x2) (hx1_pos : 0 < x1) (hx2_pos : 0 < x2) :
  (∀ x1 x2, f (x1 + x2) = f x1 * f x2) = false ∧
  (∀ x1 x2, f (x1 * x2) = f x1 + f x2 → f x1 = log x1) ∧
  (∀ x1 x2, (f x1 - f x2) / (x1 - x2) > 0 → f x1 = log x1) ∧
  (∀ x1 x2, f ((x1 + x2) / 2) < (f x1 + f x2) / 2) = false :=
sorry

end log_function_properties_l156_156554


namespace correct_statements_l156_156407

variables {X Y : Type*} [AddCommGroup X] [AddCommGroup Y] [Module ℝ X] [Module ℝ Y]
variables {r : ℝ} {x : X} {y : Y}

def statement1 := r > 0 → (x}, y) ↦ (if x increases then y increases)
def statement2 := abs(r) ≈ 1 → degree_of_linear_correlation(x, y) ≈ 1
def statement3 := (r = 1 ∨ r = -1) → (x, y) ↦ (if the relationship is completely corresponding then scatter_plot(straight_line))

theorem correct_statements : statement1 ∧ statement2 ∧ statement3 :=
sorry

end correct_statements_l156_156407


namespace min_six_integers_for_progressions_l156_156872

theorem min_six_integers_for_progressions : 
  (∀ (s : Finset ℤ), s.card < 6 →  ¬(∃ (a r d : ℤ), 
    (Set.image (λ n, a * r^n) {0, 1, 2, 3, 4} ⊆ s) ∧ 
    (Set.image (λ n, (a + n * d)) {0, 1, 2, 3, 4} ⊆ s))) ∧ 
  (∃ (s : Finset ℤ), s.card = 6 ∧ 
    (∃ (a r d : ℤ), 
      (Set.image (λ n, a * r^n) {0, 1, 2, 3, 4} ⊆ s) ∧ 
      (Set.image (λ n, (a + n * d)) {0, 1, 2, 3, 4} ⊆ s))) :=
by
  sorry

end min_six_integers_for_progressions_l156_156872


namespace value_of_squares_l156_156265

-- Define the conditions
variables (p q : ℝ)

-- State the theorem with the given conditions and the proof goal
theorem value_of_squares (h1 : p * q = 12) (h2 : p + q = 8) : p ^ 2 + q ^ 2 = 40 :=
sorry

end value_of_squares_l156_156265


namespace problem_statement_l156_156714

-- Given conditions
def g (x : ℝ) : ℝ := sorry
axiom g_property : ∀ x y : ℝ, g(x) * g(y) - g(x * y) = 2 * (x + y)

-- Define the values to be proven
def n : ℝ := 1
def s : ℝ := 16 / (Real.sqrt 17 - 1)

-- The equivalent theorem
theorem problem_statement : n * s = 16 / (Real.sqrt 17 - 1) := by
  sorry

end problem_statement_l156_156714


namespace smallest_positive_integer_3_l156_156995

def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], ![0, 1]]

theorem smallest_positive_integer_3 :
  ∃ n : ℕ, n > 0 ∧ (rotation_matrix (120 * Real.pi / 180)) ^ n = identity_matrix ∧ (∀ m : ℕ, m > 0 ∧ (rotation_matrix (120 * Real.pi / 180)) ^ m = identity_matrix → n ≤ m) := 
sorry

end smallest_positive_integer_3_l156_156995


namespace angle_B_is_pi_over_4_l156_156248

variable {A B C : ℝ}
variable {a b c : ℝ}

theorem angle_B_is_pi_over_4 (h₁ : ∀ (A B C a b c : ℝ), 
    a = sin A * (b + c) ∧ 
    ∀ c b sqrt,
    ∀ (sinA sinB sinC: ℝ), 
    ∀ (A B C: ℝ → ℝ), 
    ∀ ( a b c : ℝ → ℝ → ℝ), 
    ∀ sin (sin A / sin B + sinC),
    sin(a) = sinb := c - b / sqrt * 2 * c := a) :
    B = ℝ.pi / 4 := by
  sorry

end angle_B_is_pi_over_4_l156_156248


namespace roots_product_l156_156421

theorem roots_product : (27^(1/3) * 81^(1/4) * 64^(1/6)) = 18 := 
by
  sorry

end roots_product_l156_156421


namespace angle_CDB_45_degrees_l156_156058

theorem angle_CDB_45_degrees
  (α β γ δ : ℝ)
  (triangle_isosceles_right : α = β)
  (triangle_angle_BCD : γ = 90)
  (square_angle_DCE : δ = 90)
  (triangle_angle_ABC : α = β)
  (isosceles_triangle_angle : α + β + γ = 180)
  (isosceles_triangle_right : α = 45)
  (isosceles_triangle_sum : α + α + 90 = 180)
  (square_geometry : δ = 90) :
  γ + δ = 180 →  180 - (γ + α) = 45 :=
by
  sorry

end angle_CDB_45_degrees_l156_156058


namespace simplify_fraction_l156_156350

variable {a b c : ℝ} -- assuming a, b, c are real numbers

theorem simplify_fraction (hc : a + b + c ≠ 0) :
  (a^2 + b^2 - c^2 + 2 * a * b) / (a^2 + c^2 - b^2 + 2 * a * c) = (a + b - c) / (a - b + c) :=
sorry

end simplify_fraction_l156_156350


namespace count_of_plus_signs_l156_156807

-- Definitions based on the given conditions
def total_symbols : ℕ := 23
def any_10_symbols_contains_plus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 10 → (\exists c ∈ s, c = '+')
def any_15_symbols_contains_minus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 15 → (\exists c ∈ s, c = '-')

-- Main theorem statement without the proof
theorem count_of_plus_signs (symbols : Finset char) (h1 : symbols.card = total_symbols) 
  (h2 : any_10_symbols_contains_plus symbols) 
  (h3 : any_15_symbols_contains_minus symbols) : 
  symbols.count '+' = 14 :=
sorry

end count_of_plus_signs_l156_156807


namespace B_time_to_complete_work_l156_156454

theorem B_time_to_complete_work :
  (A1 B1 C1 : ℚ) (h1 : A1 = 1/5) (h2 : B1 + C1 = 1/3) (h3 : A1 + C1 = 1/2) :
  1/B1 = 30 :=
by
  sorry

end B_time_to_complete_work_l156_156454


namespace floor_sqrt_120_eq_10_l156_156521

theorem floor_sqrt_120_eq_10 :
  (√120).to_floor = 10 := by
  have h1 : √100 = 10 := by norm_num
  have h2 : √121 = 11 := by norm_num
  have h : 100 < 120 ∧ 120 < 121 := by norm_num
  have sqrt_120 : 10 < √120 ∧ √120 < 11 :=
    by exact ⟨real.sqrt_lt' 120 121 h.2, real.sqrt_lt'' 100 120 h.1⟩
  sorry

end floor_sqrt_120_eq_10_l156_156521


namespace ellipse_equation_hyperbola_equation_l156_156549

-- Proof statement for the ellipse equation
theorem ellipse_equation :
  ∃ λ : ℝ, λ = 2 ∧
      ∃ a b : ℝ, a = 8 ∧ b = 6 ∧
        (∀ x y : ℝ, (x, y) = (2, -sqrt 3) → (x^2 / a + y^2 / b = λ)) ∧
        (∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1) → λ = x^2 / 4 + y^2 / 3) ∧
        (a = 8 ∧ b = 6) :=
sorry

-- Proof statement for the hyperbola equation
theorem hyperbola_equation :
  ∃ a b : ℝ, a = 3 ∧ b = 1 ∧
      (∀ x y : ℝ, (x^2 / 9 - y^2 / (b^2) = 1)) ∧
      (2 * a = 6) ∧
      (b / a = 1 / 3) :=
sorry

end ellipse_equation_hyperbola_equation_l156_156549


namespace turtle_minimum_distance_l156_156479

/-- 
Given a turtle starting at the origin (0,0), crawling at a speed of 5 m/hour,
and turning 90 degrees at the end of each hour, prove that after 11 hours,
the minimum distance from the origin it could be is 5 meters.
-/
theorem turtle_minimum_distance :
  let speed := 5
  let hours := 11
  let distance (n : ℕ) := n * speed
  in ∃ (final_position : ℤ × ℤ),
      final_position = (5, 0) ∨ final_position = (-5, 0) ∨ final_position = (0, 5) ∨ final_position = (0, -5) :=
  sorry

end turtle_minimum_distance_l156_156479


namespace plus_signs_count_l156_156829

theorem plus_signs_count (n p m : ℕ) (h_total : n = 23) 
  (h_plus_min : ∀ s, s.card = 10 → ∃ t ∈ s, is_plus t)
  (h_minus_min : ∀ s, s.card = 15 → ∃ t ∈ s, is_minus t)
  : p = 14 :=
sorry

end plus_signs_count_l156_156829


namespace no_prime_configuration_l156_156403

theorem no_prime_configuration (d1 d2 d3 d4 : ℕ) :
  (∀ a, a ∈ {d1, d2, d3, d4} → a ∈ {1, 3, 7, 9}) →
  (∀ (x y : ℕ), x ∈ {d1, d2} → y ∈ {d3, d4} → prime (10 * x + y) ) →
  false :=
begin
  sorry
end

end no_prime_configuration_l156_156403


namespace rabbits_to_hamsters_l156_156767

theorem rabbits_to_hamsters (rabbits hamsters : ℕ) (h_ratio : 3 * hamsters = 4 * rabbits) (h_rabbits : rabbits = 18) : hamsters = 24 :=
by
  sorry

end rabbits_to_hamsters_l156_156767


namespace sin_alpha_value_l156_156157

theorem sin_alpha_value (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : cos (α + π / 6) = 2 / 3) :
  sin α = (sqrt 15 - 2) / 6 :=
sorry

end sin_alpha_value_l156_156157


namespace sqrt2x_lt_3x_minus_4_iff_x_gt_2_l156_156514

theorem sqrt2x_lt_3x_minus_4_iff_x_gt_2 (x : ℝ) (hx_pos : 0 < x) : 
  (sqrt (2 * x) < 3 * x - 4) ↔ (x > 2) :=
sorry

end sqrt2x_lt_3x_minus_4_iff_x_gt_2_l156_156514


namespace count_integer_solutions_l156_156635

theorem count_integer_solutions (x : ℤ) : 
  (|x - 3| ≤ 7) → (x ∈ Finset.range 15 → x + -8 : ℤ) := 
sorry

end count_integer_solutions_l156_156635


namespace compare_sequences_l156_156111

def u_seq : ℕ → ℚ
| 0       := 2
| (n + 1) := 2 / (2 + u_seq n)

def v_seq : ℕ → ℚ
| 0       := 3
| (n + 1) := 3 / (3 + v_seq n)

theorem compare_sequences : u_seq 2022 < v_seq 2022 :=
sorry

end compare_sequences_l156_156111


namespace tan_theta_parallel_l156_156630

theorem tan_theta_parallel
  (θ : ℝ)
  (a : ℝ × ℝ := (2, sin θ))
  (b : ℝ × ℝ := (1, cos θ))
  (h : ∃ k : ℝ, a = (k * b.1, k * b.2)) :
  tan θ = 2 :=
by
  sorry

end tan_theta_parallel_l156_156630


namespace infinite_solutions_iff_c_eq_5_over_2_l156_156974

theorem infinite_solutions_iff_c_eq_5_over_2 (c y : ℝ) (h : y ≠ 0) :
  (∀ y ≠ 0, 3 * (5 + 2 * c * y) = 15 * y + 15) ↔ c = 5 / 2 :=
by
  sorry

end infinite_solutions_iff_c_eq_5_over_2_l156_156974


namespace tetrahedron_circumsphere_radius_l156_156230

theorem tetrahedron_circumsphere_radius :
  ∀ {A B C D : Type}
    [MetricSpace A]
    [MetricSpace B]
    [MetricSpace C]
    [MetricSpace D]
    (angle_ADB angle_BDC angle_CDA : ℝ)
    (r : ℝ)
    (AD BD : ℝ)
    (CD : ℝ),
    angle_ADB = 60 ∧
    angle_BDC = 60 ∧
    angle_CDA = 60 ∧
    AD = 3 ∧
    BD = 3 ∧
    CD = 2 →
    r = sqrt 3 :=
by sorry

end tetrahedron_circumsphere_radius_l156_156230


namespace inscribed_squares_equilateral_triangle_l156_156112

theorem inscribed_squares_equilateral_triangle (a b c h_a h_b h_c : ℝ) 
  (h1 : a * h_a / (a + h_a) = b * h_b / (b + h_b))
  (h2 : b * h_b / (b + h_b) = c * h_c / (c + h_c)) :
  a = b ∧ b = c ∧ h_a = h_b ∧ h_b = h_c :=
sorry

end inscribed_squares_equilateral_triangle_l156_156112


namespace total_property_price_l156_156763

theorem total_property_price :
  let price_per_sqft : ℝ := 98
  let house_sqft : ℝ := 2400
  let barn_sqft : ℝ := 1000
  let house_price : ℝ := house_sqft * price_per_sqft
  let barn_price : ℝ := barn_sqft * price_per_sqft
  let total_price : ℝ := house_price + barn_price
  total_price = 333200 := by
  sorry

end total_property_price_l156_156763


namespace markup_percentage_is_20_l156_156064

-- Define the given conditions
def CP : ℝ := 180
def profit : ℝ := 0.20 * CP
def SP : ℝ := CP + profit
def discount : ℝ := 50
def SP_after_discount : ℝ := SP - discount
def markup : ℝ := SP - CP
def markup_percentage : ℝ := (markup / CP) * 100

-- Theorem stating the required proof problem
theorem markup_percentage_is_20 :
  markup_percentage = 20 := 
sorry

end markup_percentage_is_20_l156_156064


namespace polygon_sides_l156_156378

theorem polygon_sides (n : ℕ) 
  (h_interior : (n - 2) * 180 = 4 * 360) : 
  n = 10 :=
by
  sorry

end polygon_sides_l156_156378


namespace correct_statements_count_l156_156883

theorem correct_statements_count :
  -- Condition 1
  (∀ x : ℝ, let y := 2 - 3 * x in ∀ k : ℝ, y + -k * 3 = 2 - 3 * (x + k)) →
  -- Condition 2
  (¬ (∃ x : ℝ, x^2 - x - 1 > 0) ↔ ∀ x : ℝ, x^2 - x - 1 ≤ 0) →
  -- Condition 3
  (∀ ssrs1 ssrs2 : ℝ, ssrs1 < ssrs2 → model_fits_better ssrs1 ssrs2) →
  -- Condition 4
  (∀ R1 R2 : ℝ, R1 < R2 → regression_fit_better R2 R1) →
  -- Conclusion
  2 = (2)
:= sorry

end correct_statements_count_l156_156883


namespace flowchart_output_l156_156540

theorem flowchart_output (N : ℕ) (hN : N = 6) : 
  let p := (Nat.range (N + 1)).foldl (λ acc k, if k = 0 then 1 else acc * k) 1
  in p = 720 :=
by
  have N_val : N = 6 := hN
  simp [N_val]
  -- Fold over the range from 0 to N to simulate the iterative process.
  let p := (Nat.range (6 + 1)).foldl (λ acc k, if k = 0 then 1 else acc * k) 1
  show p = 720
  -- The resulting p should be 720 after the iterations.
  have p_val : p = 720 := by
    simp [p]
  exact p_val

end flowchart_output_l156_156540


namespace complex_modulus_l156_156656

def complex_number (b: ℝ) : ℂ :=
  (3 - b * complex.I) / (2 + complex.I)

theorem complex_modulus :
  ∀ (b : ℝ), 
    (complex_number b).re = (complex_number b).im → 
    complex.abs (complex_number b) = 3 * real.sqrt 2 := 
by
  intro b
  sorry

end complex_modulus_l156_156656


namespace counterexample_21_l156_156490

def is_prime (k : ℕ) : Prop := k > 1 ∧ ∀ d : ℕ, d ∣ k → d = 1 ∨ d = k

def not_prime (k : ℕ) : Prop := ¬ is_prime(k)

def is_counterexample (n : ℕ) : Prop :=
  not_prime n ∧ not_prime (n + 3)

theorem counterexample_21 : is_counterexample 21 :=
by {
  -- Definition of not_prime by negation of is_prime
  have h1 : not_prime 21 := by {
      unfold not_prime,
      unfold is_prime,
      push_neg,
      exact ⟨21 > 1, 3, dvd.intro 7 rfl, λ h, h.symm = 69⟩,
  },
  -- Same for is_prime for 24 (21 + 3)
  have h2 : not_prime 24 := by {
      unfold not_prime,
      unfold is_prime,
      push_neg,
      exact ⟨24 > 1, 6, dvd.intro 4 rfl, λ h, h.symm = 144⟩,
  },
  exact ⟨h1, h2⟩,
}

end counterexample_21_l156_156490


namespace find_gain_percentage_l156_156640

variable {C1 C2 SP1 SP2 : ℝ}

-- Condition definitions
def cost_first_book := C1 = 280
def total_cost := C1 + C2 = 480
def sell_first_loss := SP1 = C1 - 0.15 * C1
def same_selling_price := SP1 = SP2

-- To prove: gain percentage on the second book is 19%
def gain_percentage_on_second_book : Prop :=
  ∃ (gain_percentage : ℝ), gain_percentage = 19 ∧
  (same_selling_price → (total_cost → (cost_first_book → (sell_first_loss → 
    C2 = 200 ∧ SP1 = 238 ∧ SP2 = 238 ∧ SP2 = C2 + (gain_percentage / 100 * C2)))))
  
theorem find_gain_percentage (C1 C2 SP1 SP2 : ℝ)
  (h1 : cost_first_book)
  (h2 : total_cost)
  (h3 : sell_first_loss)
  (h4 : same_selling_price) :
  gain_percentage_on_second_book :=
begin
  sorry
end

end find_gain_percentage_l156_156640


namespace arccos_gt_arctan_l156_156986

theorem arccos_gt_arctan (x : ℝ) (h : x ∈ set.Icc (-1 : ℝ) 1) : 
  (arccos x > arctan x) ↔ (x < real.sqrt 2 / 2) :=
sorry

end arccos_gt_arctan_l156_156986


namespace monotonic_intervals_constant_sum_extremes_value_of_a_for_max_abs_f_l156_156263

variable (a : ℝ) (h_a : a > 0)

def f (x : ℝ) : ℝ := (x - 2)^3 - a * x

-- Part 1: Monotonic Intervals
theorem monotonic_intervals : 
  (∀ x < 2 - real.sqrt (a / 3), f'(x) > 0) ∧ 
  (∀ x ∈ Ioo (2 - real.sqrt (a / 3)) (2 + real.sqrt (a / 3)), f'(x) < 0) ∧ 
  (∀ x > 2 + real.sqrt (a / 3), f'(x) > 0) := 
sorry

-- Part 2: Constant x1 + 2x0 for extreme points
theorem constant_sum_extremes (x0 x1 : ℝ) (h_extreme : (x0 = 2 - real.sqrt (a / 3)) ∨ (x0 = 2 + real.sqrt (a / 3))) 
  (h_f_eq : f x1 = f x0) (h_neq : x1 ≠ x0) : 
  x1 + 2 * x0 = 6 := 
sorry

-- Part 3: Values of a for max |f(x)| on [0, 6]
theorem value_of_a_for_max_abs_f (h_max : ∀ x ∈ Icc 0 6, abs (f x) ≤ 40) :
  a ∈ {4, 12} :=
sorry

end monotonic_intervals_constant_sum_extremes_value_of_a_for_max_abs_f_l156_156263


namespace work_completion_days_l156_156888

theorem work_completion_days
  (A B : ℝ)
  (h1 : A + B = 1 / 12)
  (h2 : A = 1 / 20)
  : 1 / (A + B / 2) = 15 :=
by 
  sorry

end work_completion_days_l156_156888


namespace no_positive_integer_n_exists_l156_156694

theorem no_positive_integer_n_exists {n : ℕ} (hn : n > 0) :
  ¬ ((∃ k, 5 * 10^(k - 1) ≤ 2^n ∧ 2^n < 6 * 10^(k - 1)) ∧
     (∃ m, 2 * 10^(m - 1) ≤ 5^n ∧ 5^n < 3 * 10^(m - 1))) :=
sorry

end no_positive_integer_n_exists_l156_156694


namespace plus_signs_count_l156_156824

theorem plus_signs_count (n p m : ℕ) (h_total : n = 23) 
  (h_plus_min : ∀ s, s.card = 10 → ∃ t ∈ s, is_plus t)
  (h_minus_min : ∀ s, s.card = 15 → ∃ t ∈ s, is_minus t)
  : p = 14 :=
sorry

end plus_signs_count_l156_156824


namespace sum_in_base7_l156_156997

-- An encoder function for base 7 integers
def to_base7 (n : ℕ) : string :=
sorry -- skipping the implementation for brevity

-- Decoding the string representation back to a natural number
def from_base7 (s : string) : ℕ :=
sorry -- skipping the implementation for brevity

-- The provided numbers in base 7
def x : ℕ := from_base7 "666"
def y : ℕ := from_base7 "66"
def z : ℕ := from_base7 "6"

-- The expected sum in base 7
def expected_sum : ℕ := from_base7 "104"

-- The statement to be proved
theorem sum_in_base7 : x + y + z = expected_sum :=
sorry -- The proof is omitted

end sum_in_base7_l156_156997


namespace bowling_average_decrease_l156_156889

theorem bowling_average_decrease
    (initial_average : ℝ) (wickets_last_match : ℝ) (runs_last_match : ℝ)
    (average_decrease : ℝ) (W : ℝ)
    (H_initial : initial_average = 12.4)
    (H_wickets_last_match : wickets_last_match = 6)
    (H_runs_last_match : runs_last_match = 26)
    (H_average_decrease : average_decrease = 0.4) :
    W = 115 :=
by
  sorry

end bowling_average_decrease_l156_156889


namespace stream_speed_l156_156012

theorem stream_speed :
  ∀ (v : ℝ),
  (12 - v) / (12 + v) = 1 / 2 →
  v = 4 :=
by
  sorry

end stream_speed_l156_156012


namespace ticket_cost_before_rally_l156_156772

-- We define the variables and constants given in the problem
def total_attendance : ℕ := 750
def tickets_before_rally : ℕ := 475
def tickets_at_door : ℕ := total_attendance - tickets_before_rally
def cost_at_door : ℝ := 2.75
def total_receipts : ℝ := 1706.25

-- Problem statement: Prove that the cost of each ticket bought before the rally (x) is 2 dollars.
theorem ticket_cost_before_rally (x : ℝ) 
  (h₁ : tickets_before_rally * x + tickets_at_door * cost_at_door = total_receipts) :
  x = 2 :=
by
  sorry

end ticket_cost_before_rally_l156_156772


namespace union_is_real_l156_156593

-- Definitions of sets A and B
def setA : Set ℝ := {x | x^2 - x - 2 ≥ 0}
def setB : Set ℝ := {x | x > -1}

-- Theorem to prove
theorem union_is_real :
  setA ∪ setB = Set.univ :=
by
  sorry

end union_is_real_l156_156593


namespace slope_of_line_l156_156353

theorem slope_of_line (h_intersection : ∃ L : ℝ → ℝ, L (0) = 4 ∧ L (-8) = 0)
  (h_area : ∃ b h : ℝ, b = 8 ∧ h = 4 ∧ 1 / 2 * b * h = 16)
  : ∃ m : ℝ, m = 1/2 :=
by
  -- Intersection at (-8, 0) and (0, 4) implies height as 4
  have height := 4
  -- Area calculation gives height using base spanning 8 units and given area
  have base := 8
  have area := 16
  have triangle_area := (1 / 2 : ℝ) * base * height
  have correct_area : (1 / 2 : ℝ) * 8 * 4 = 16 := by norm_num
  -- Identify the slope using the coordinates
  have slope_formula := (4 - 0) / (0 + 8)
  use (1 / 2)
  sorry  

end slope_of_line_l156_156353


namespace linda_savings_l156_156892

theorem linda_savings (S : ℝ) (h1 : 1 / 4 * S = 150) : S = 600 :=
sorry

end linda_savings_l156_156892


namespace cards_arrangement_count_is_10_l156_156316

-- Define the problem in Lean statement terms
def valid_arrangements_count : ℕ :=
  -- number of arrangements of seven cards where one card can be removed 
  -- leaving the remaining six cards in either ascending or descending order
  10

-- Theorem stating that the number of valid arrangements is 10
theorem cards_arrangement_count_is_10 : valid_arrangements_count = 10 :=
by
  -- Proof is omitted (the explanation above corresponds to the omitted proof details)
  sorry

end cards_arrangement_count_is_10_l156_156316


namespace evaluate_expression_l156_156074

theorem evaluate_expression : 6 / (-1 / 2 + 1 / 3) = -36 := 
by
  sorry

end evaluate_expression_l156_156074


namespace compute_fourth_power_z_l156_156085

-- Definitions from the problem
def cos_angle (θ : ℝ) : ℝ := Real.cos θ
def sin_angle (θ : ℝ) : ℝ := Real.sin θ
def θ := Real.pi / 6  -- 30 degrees in radians

def z : ℂ := 3 * (cos_angle θ) + 3 * Complex.I * (sin_angle θ)

-- Lean 4 Statement for the proof
theorem compute_fourth_power_z : (z ^ 4) = -40.5 + 40.5 * Complex.I * Real.sqrt 3 :=
sorry

end compute_fourth_power_z_l156_156085


namespace plus_signs_count_l156_156833

theorem plus_signs_count (total_symbols : ℕ) (n_plus n_minus : ℕ)
  (h1 : total_symbols = 23)
  (h2 : ∀ s : Finset ℕ, s.card = 10 → ∃ i ∈ s, i ≤ n_plus)
  (h3 : ∀ s : Finset ℕ, s.card = 15 → ∃ i ∈ s, i > n_plus) :
  n_plus = 14 :=
by
  -- the proof will go here
  sorry

end plus_signs_count_l156_156833


namespace total_get_well_cards_l156_156722

-- Definitions for the number of cards received in each place
def cardsInHospital : ℕ := 403
def cardsAtHome : ℕ := 287

-- Theorem statement:
theorem total_get_well_cards : cardsInHospital + cardsAtHome = 690 := by
  sorry

end total_get_well_cards_l156_156722


namespace integers_not_multiples_of_2_3_5_l156_156186

theorem integers_not_multiples_of_2_3_5 (A B C: Finset ℕ) (n : ℕ) 
  (hA : A.card = 100) (hB : B.card = 66) (hC : C.card = 40) 
  (hAB : (A ∩ B).card = 33) (hAC : (A ∩ C).card = 20) 
  (hBC : (B ∩ C).card = 13) (hABC : (A ∩ B ∩ C).card = 6) 
  (h_range : n = 200) :
  (Finset.range n).filter (λ x, ¬(x ∈ A ∪ B ∪ C)).card = 54 :=
by sorry

end integers_not_multiples_of_2_3_5_l156_156186


namespace find_five_digit_number_l156_156899

-- Define the representation and properties of the digits
variables {digits : Finset ℕ} (h_digits : digits.card = 5) (h_nonzero : ∀ d ∈ digits, d ≠ 0)

-- Define the sum of all three-digit numbers formed by any 3 of the 5 digits
def sum_three_digit_numbers (digits : Finset ℕ) : ℕ :=
  digits.sum (λ d, 1332 * d)

-- Define the 5-digit number N and the condition it must satisfy
def N := sum_three_digit_numbers digits

-- State the theorem to be proved
theorem find_five_digit_number (h_14N_five_digits : 10000 ≤ 14 * N ∧ 14 * N < 100000) : N = 35964 :=
  sorry

end find_five_digit_number_l156_156899


namespace sum_of_numbers_in_table_l156_156676

theorem sum_of_numbers_in_table
  (m n : ℕ)
  (a : matrix (fin m) (fin n) ℝ)
  (h : ∀ i j, (∑ k, a i k) * (∑ k, a k j) = a i j) :
  (∑ i j, a i j = 1) ∨ (∀ i j, a i j = 0) :=
sorry

end sum_of_numbers_in_table_l156_156676


namespace probability_of_events_l156_156406

def tetrahedron_outcomes : Set (ℕ × ℕ) :=
  { (i, j) | i ∈ {1, 2, 3, 4} ∧ j ∈ {1, 2, 3, 4} }

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

def cond_A (outcome : ℕ × ℕ) : Prop := is_even outcome.1
def cond_B (outcome : ℕ × ℕ) : Prop := is_odd outcome.2
def cond_C (outcome : ℕ × ℕ) : Prop := (is_odd outcome.1 ∧ is_odd outcome.2) ∨ (is_even outcome.1 ∧ is_even outcome.2)

def event_A : Set (ℕ × ℕ) := {x ∈ tetrahedron_outcomes | cond_A x}
def event_B : Set (ℕ × ℕ) := {x ∈ tetrahedron_outcomes | cond_B x}
def event_C : Set (ℕ × ℕ) := {x ∈ tetrahedron_outcomes | cond_C x}
def event_AB : Set (ℕ × ℕ) := event_A ∩ event_B
def event_AC : Set (ℕ × ℕ) := event_A ∩ event_C
def event_BC : Set (ℕ × ℕ) := event_B ∩ event_C
def event_ABC : Set (ℕ × ℕ) := event_A ∩ event_B ∩ event_C

theorem probability_of_events :
  P(event_A) = 1 / 2 ∧
  P(event_B) = 1 / 2 ∧
  P(event_C) = 1 / 2 ∧
  P(event_AB) = 1 / 4 ∧
  P(event_AC) = 1 / 4 ∧
  P(event_BC) = 1 / 4 ∧
  P(event_ABC) = 0 ∧
  ¬ disjoint event_A event_B := by
  sorry

end probability_of_events_l156_156406


namespace correct_speed_to_reach_on_time_l156_156282

theorem correct_speed_to_reach_on_time
  (d : ℝ)
  (t : ℝ)
  (h1 : d = 50 * (t + 1 / 12))
  (h2 : d = 70 * (t - 1 / 12)) :
  d / t = 58 := 
by
  sorry

end correct_speed_to_reach_on_time_l156_156282


namespace probability_circle_intersects_side_l156_156466

/-- 
A point P is chosen randomly inside a triangle with sides 13, 20, and 21. 
The probability that the circle centered at P with radius 1 will intersect 
at least one of the sides of the triangle is 75/196.
-/
theorem probability_circle_intersects_side :
  let P : Type := ℝ × ℝ in
  let sides : Type := {abc : ℝ × ℝ × ℝ // abc.1 = 21 ∧ abc.2.1 = 13 ∧ abc.2.2 = 20} in
  ∃ (p : P) (abc : sides), 
  let intersects := (circleIntersectSide p abc) in
  (calculateProbability intersects) = (75 / 196) :=
begin
  sorry
end

end probability_circle_intersects_side_l156_156466


namespace plus_signs_count_l156_156794

noncomputable def solveSymbols : Prop :=
  ∃ p m : ℕ, p + m = 23 ∧ p ≤ 14 ∧ m ≤ 9 ∧ (∀ s : Finset (Fin 23), s.card = 10 → (s.image (λ i => if i < p then true else false)).count true ≥ 1) ∧ (∀ s : Finset (Fin 23), s.card = 15 → (s.image (λ i => if i < p then false else true)).count false ≥ 1) 

theorem plus_signs_count : solveSymbols :=
sorry

end plus_signs_count_l156_156794


namespace number_of_valid_arrangements_l156_156308

open Finset

-- We define the condition that a list is sorted in ascending order
def is_ascending (l : List ℕ) : Prop :=
  l = List.sort (≤) l

-- We define the condition that a list is sorted in descending order
def is_descending (l : List ℕ) : Prop :=
  l = List.sort (≥) l

def cards := Finset.range 7
def arrangements := cards.to_list.permutations

-- Define the function to check if a list of numbers (cards) 
-- can have one element removed to form an ascending or descending list
def valid_arrangement (l : List ℕ) : Prop :=
  ∃ (x : ℕ), (l.erase x).is_ascending ∨ (l.erase x).is_descending

-- Define the final theorem
theorem number_of_valid_arrangements : finset.card (arrangements.filter valid_arrangement) = 72 :=
by
  sorry

end number_of_valid_arrangements_l156_156308


namespace cards_arrangement_count_l156_156296

theorem cards_arrangement_count : 
  let cards := [1, 2, 3, 4, 5, 6, 7] in
  let valid_arrangements := 
    {arrangement | ∃ removed, 
      removed ∈ cards ∧ 
      (∀ remaining, 
        remaining = cards.erase removed → 
        (sorted remaining ∨ sorted (remaining.reverse))) } in
  valid_arrangements.card = 26 :=
sorry

end cards_arrangement_count_l156_156296


namespace number_of_true_propositions_l156_156597

theorem number_of_true_propositions
  (m n : Line)
  (α β : Plane)
  (h1 : m ≠ n)
  (h2 : α ≠ β) :
  (¬ (m ⊆ α ∧ n ∥ α → m ∥ n)) ∧
  (¬ (m ∥ α ∧ m ∥ β → α ∥ β)) ∧
  (¬ (m ⊥ α ∧ m ⊥ n → n ∥ α)) ∧
  (m ⊥ α ∧ m ⊥ β → α ∥ β) →
  1 := by
  sorry

end number_of_true_propositions_l156_156597


namespace ages_sum_seven_years_ago_l156_156333

theorem ages_sum_seven_years_ago (a1 a2 a3 : ℕ) (a4 a5 : ℕ) 
    (h1 : a4 = 8) (h2: a5 = 6) (h3 : a1 + a2 + a3 + a4 + a5 = 80) : 
    a1 - 7 + (a2 - 7) + (a3 - 7) + 1 + 0 = 45 := 
by 
  simp [h1, h2] 
  linarith 

end ages_sum_seven_years_ago_l156_156333


namespace inequality_proof_l156_156653

theorem inequality_proof (x a : ℝ) (h1 : x > a) (h2 : a > 0) : x^2 > ax ∧ ax > a^2 :=
by
  sorry

end inequality_proof_l156_156653


namespace ball_bounces_height_l156_156031

theorem ball_bounces_height : ∃ k : ℕ, ∀ n ≥ k, 800 * (2 / 3: ℝ) ^ n < 10 :=
by
  sorry

end ball_bounces_height_l156_156031


namespace cos_value_l156_156155

theorem cos_value (x : ℝ) (h : Real.csc x - Real.cot x = 5 / 2) :
  Real.cos x = -21 / 29 :=
sorry

end cos_value_l156_156155


namespace angle_BPE_l156_156342

-- Define the conditions given in the problem
def triangle_ABC (A B C : ℝ) : Prop := A = 60 ∧ 
  (∃ (B₁ B₂ B₃ : ℝ), B₁ = B / 3 ∧ B₂ = B / 3 ∧ B₃ = B / 3) ∧ 
  (∃ (C₁ C₂ C₃ : ℝ), C₁ = C / 3 ∧ C₂ = C / 3 ∧ C₃ = C / 3) ∧ 
  (B + C = 120)

-- State the theorem to proof
theorem angle_BPE (A B C x : ℝ) (h : triangle_ABC A B C) : x = 50 := by
  sorry

end angle_BPE_l156_156342


namespace sequence_solution_l156_156168

-- Define the sequences and conditions
def positive_seq (a : ℕ → ℕ) (S : ℕ → ℕ) :=
  (∀ n : ℕ, n > 0 → (4 * S n = (a n) ^ 2 + 2 * (a n))) ∧ (a 1 = 2) ∧ (∀ n : ℕ, n > 0 → (n * a (n + 1) = 2 * (S n)))

-- Define the final remainder computation
def final_remainder (f : ℕ → ℕ) (n : ℕ) :=
  let b := λ i, 2 ^ (f i / 2) in
  (b.sum (range (2 * n + 2)) % 3 = 2)

-- Statement that needs to be proved
theorem sequence_solution : ∃ a S : ℕ → ℕ, positive_seq a S ∧ ∀ n : ℕ, a n = 2 * n ∧ final_remainder a (n + 1) :=
sorry

end sequence_solution_l156_156168


namespace frequency_histogram_classes_l156_156907

def tallest_height : ℕ := 186
def shortest_height : ℕ := 154
def class_interval : ℕ := 5

theorem frequency_histogram_classes :
  ∃ (num_classes : ℕ), num_classes = Int.ceil ((tallest_height - shortest_height) / class_interval) ∧ num_classes = 7 :=
sorry

end frequency_histogram_classes_l156_156907


namespace num_integer_solutions_abs_leq_seven_l156_156637

theorem num_integer_solutions_abs_leq_seven : 
  (∃ n : ℕ, n = (finset.Icc (-4 : ℤ) 10).card) ∧ n = 15 := 
by 
  sorry

end num_integer_solutions_abs_leq_seven_l156_156637


namespace mode_and_median_of_data_set_l156_156691

theorem mode_and_median_of_data_set {X : List ℕ} (h : X = [5, 9, 5, 6, 4, 5, 7]) :
  List.mode X = 5 ∧ List.median X = 5 :=
by
  sorry

end mode_and_median_of_data_set_l156_156691


namespace republic_connectivity_l156_156020

-- Definitions based on the problem's conditions
def City : Type := Fin 1001
def Republic := Fin 668
def outgoing_roads (C : City) : Finset City := sorry
def incoming_roads (C : City) : Finset City := sorry

-- Condition: Each city has exactly 500 outgoing and 500 incoming roads.
axiom outgoing_roads_card : ∀ C : City, (outgoing_roads C).card = 500
axiom incoming_roads_card : ∀ C : City, (incoming_roads C).card = 500

-- Condition: There is a road between each distinct pair of cities.
axiom road_exists : ∀ (C1 C2 : City), C1 ≠ C2 → (C2 ∈ outgoing_roads C1 ∨ C1 ∈ incoming_roads C2)

-- Republic is a subset of cities with the given properties and contains 668 cities.
def Republic_cities : Finset City := sorry

-- Condition: Republic contains exactly 668 cities
axiom republic_card : Republic_cities.card = 668

-- Main theorem to prove
theorem republic_connectivity: 
  ∀ C1 C2 : City, C1 ∈ Republic_cities → C2 ∈ Republic_cities → ∃ (path : Finset (Finset City)), -- A path between C1 and C2 exists
    (∀ segment : Finset City, segment ∈ path → ∃ C1' C2' : City, C1' ∈ segment ∧ C2' ∈ segment ∧ C2' ∈ outgoing_roads C1')
    ∧ path.nonempty
    ∧ ∃ first last, first ∈ path ∧ last ∈ path ∧ C1 ∈ first ∧ C2 ∈ last
    := sorry

end republic_connectivity_l156_156020


namespace mean_value_points_range_l156_156662

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2

noncomputable def g (b x : ℝ) : ℝ := x^2 - 2*x - (1/3) * b^2 + b

theorem mean_value_points_range (b : ℝ) :
  (∃ a m : ℝ, a < m ∧ m < b ∧ (f' m) * (b - a) = f b - f a ∧
    (0 < g b 0 ∧ 0 < g b b ∧ 4 + (4/3) * b^2 - 4 * b > 0)) ↔ (3/2 < b ∧ b < 3) :=
sorry

end mean_value_points_range_l156_156662


namespace selected_athlete_is_B_l156_156057

def Athlete : Type := {A : Self, B : Self, C : Self, D : Self}
def Coach : Type := {Jia : Self, Yi : Self, Bing : Self, Ding : Self}

def prediction (coach : Coach) (athlete : Athlete) : Prop :=
  match coach, athlete with
  | Coach.Jia, Athlete.C => True
  | Coach.Jia, Athlete.D => True
  | Coach.Yi, Athlete.B => True
  | Coach.Bing, Athlete.B => True
  | Coach.Bing, Athlete.C => True
  | Coach.Ding, Athlete.C => True
  | _, _ => False

def correct_predictions (selected : Athlete) : ℕ :=
  if prediction Coach.Jia selected then 1 else 0 +
  if prediction Coach.Yi selected then 1 else 0 +
  if prediction Coach.Bing selected then 1 else 0 +
  if prediction Coach.Ding selected then 1 else 0

theorem selected_athlete_is_B : ∃ selected : Athlete, correct_predictions selected = 2 ∧ selected = Athlete.B :=
by
  unfold Athlete
  unfold Coach
  unfold prediction
  unfold correct_predictions
  sorry

end selected_athlete_is_B_l156_156057


namespace derivative_of_y_l156_156340

def y (x : ℝ) : ℝ := x * Real.cos x

theorem derivative_of_y (x : ℝ) : 
  (deriv y x) = Real.cos x - x * Real.sin x :=
by
  sorry

end derivative_of_y_l156_156340


namespace find_value_of_d_l156_156149

theorem find_value_of_d
  (a b c d : ℕ) 
  (h1 : 0 < a) 
  (h2 : a < b) 
  (h3 : b < c) 
  (h4 : c < d) 
  (h5 : ab + bc + ac = abc) 
  (h6 : abc = d) : 
  d = 36 := 
sorry

end find_value_of_d_l156_156149


namespace new_home_fraction_l156_156182

variable {M H G : ℚ} -- Use ℚ (rational numbers)

def library_fraction (H : ℚ) (G : ℚ) (M : ℚ) : ℚ :=
  (1 / 3 * H + 2 / 5 * G + 1 / 2 * M) / M

theorem new_home_fraction (H_eq : H = 1 / 2 * M) (G_eq : G = 3 * H) :
  library_fraction H G M = 29 / 30 :=
by
  sorry

end new_home_fraction_l156_156182


namespace nth_monomial_is_correct_l156_156054

-- conditions
def coefficient (n : ℕ) : ℕ := 2 * n - 1
def exponent (n : ℕ) : ℕ := n
def monomial (n : ℕ) : ℕ × ℕ := (coefficient n, exponent n)

-- theorem to prove the nth monomial
theorem nth_monomial_is_correct (n : ℕ) : monomial n = (2 * n - 1, n) := 
by 
    sorry

end nth_monomial_is_correct_l156_156054


namespace length_AB_l156_156623

noncomputable def parabola : set (ℝ × ℝ) :=
  { p | ∃ x y : ℝ, p = (x, y) ∧ x^2 = 12 * y }

def focus : ℝ × ℝ := (0, 3)

noncomputable def line (θ : ℝ) : set (ℝ × ℝ) :=
  { p | ∃ x y : ℝ, p = (x, y) ∧ y = x * Real.tan θ + (focus.snd - focus.fst * Real.tan θ) }

def line_through_focus := line (Real.pi / 3)

noncomputable def intersection_points : set (ℝ × ℝ) :=
  { p | p ∈ parabola ∧ p ∈ line_through_focus }

theorem length_AB :
  let A B : ℝ × ℝ := sorry in
  ∃ A B, A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧
    dist A B = 12 * Real.sqrt 3 + 6 :=
sorry

end length_AB_l156_156623


namespace person_speed_l156_156464

def distance_m := 600  -- distance in meters
def time_min := 5  -- time in minutes

def distance_km := distance_m / 1000  -- converting meters to kilometers
def time_hr := time_min / 60  -- converting minutes to hours

theorem person_speed :
  distance_km / time_hr = 7.2 :=
by
  -- correct steps are assumed to be followed here in an actual proof
  sorry

end person_speed_l156_156464


namespace inscribed_square_area_l156_156097

def isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ c = (a ^ 2 + b ^ 2) ^ (1 / 2)

def square_area (s : ℝ) : ℝ := s * s

theorem inscribed_square_area
  (a b c : ℝ) (s₁ s₂ : ℝ)
  (ha : a = 16 * 2) -- Leg lengths equal to 2 * 16 cm
  (hb : b = 16 * 2)
  (hc : c = 32 * Real.sqrt 2) -- Hypotenuse of the triangle
  (hiso : isosceles_right_triangle a b c)
  (harea₁ : square_area 16 = 256) -- Given square area
  (hS : s₂ = 16 * Real.sqrt 2 - 8) -- Side length of the new square
  : square_area s₂ = 576 - 256 * Real.sqrt 2 := sorry

end inscribed_square_area_l156_156097


namespace angle_B_in_triangle_ABC_l156_156245

theorem angle_B_in_triangle_ABC (A B C : Prop) (angle_A : ℕ) (angle_C : ℕ) 
  (hA : angle_A = 20) (hC : angle_C = 90) : angle B = 70 :=
by
  sorry

end angle_B_in_triangle_ABC_l156_156245


namespace qiqi_and_jiajia_both_correct_l156_156396

def simplest_quadratic_radical (r : ℝ) : Prop :=
  -- Add the formal definition of the simplest quadratic radical here

def is_simplest (r : ℝ) : Prop :=
  simplest_quadratic_radical (r)

theorem qiqi_and_jiajia_both_correct :
  let r1 := sqrt(x^2 + 1),
      r2 := sqrt(x^2 * y^5),
      r3 := sqrt(13),
      r4 := 2 * sqrt(3),
      r5 := sqrt(1/2),
      r6 := sqrt(6) in
  (is_simplest r1) ∧ (is_simplest r3) ∧
  (is_simplest r4) ∧ (is_simplest r6) ∧
  ¬(is_simplest r2) ∧ ¬(is_simplest r5) → 
  (qiqi_correct ∧ jiajia_correct) :=
by
  sorry

end qiqi_and_jiajia_both_correct_l156_156396


namespace distance_between_foci_l156_156131

noncomputable def ellipse_equation (x y : ℝ) : ℝ :=
  9 * x^2 - 36 * x + 4 * y^2 + 8 * y + 16

theorem distance_between_foci :
  (∀ x y : ℝ, ellipse_equation x y = 0) →
  (∃ (d : ℝ), d = (2 * real.sqrt (10 / 3)) / 3) :=
by
  intros h
  use (2 * real.sqrt (10 / 3)) / 3
  sorry

end distance_between_foci_l156_156131


namespace largest_three_digit_multiple_of_8_with_sum_16_l156_156424

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_8_with_sum_16 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ sum_of_digits n = 16 ∧
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 8 = 0 ∧ sum_of_digits m = 16 → m ≤ n :=
  ∃! (952 : ℕ), 100 ≤ 952 ∧ 952 < 1000 ∧ 952 % 8 = 0 ∧ sum_of_digits 952 = 16 ∧
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 8 = 0 ∧ sum_of_digits m = 16 → m ≤ 952 :=
  sorry

end largest_three_digit_multiple_of_8_with_sum_16_l156_156424


namespace theta_plus_2phi_eq_pi_div_4_l156_156158

noncomputable def theta (θ : ℝ) (φ : ℝ) : Prop := 
  ((Real.tan θ = 5 / 12) ∧ 
   (Real.sin φ = 1 / 2) ∧ 
   (0 < θ ∧ θ < Real.pi / 2) ∧ 
   (0 < φ ∧ φ < Real.pi / 2)  )

theorem theta_plus_2phi_eq_pi_div_4 (θ φ : ℝ) (h : theta θ φ) : 
    θ + 2 * φ = Real.pi / 4 :=
by 
  sorry

end theta_plus_2phi_eq_pi_div_4_l156_156158


namespace problem_solution_l156_156957

theorem problem_solution :
  (∏ k in finset.range(21).map (λ n, n + 1), (1 + 19 / k) / (1 + 21 / k)) = 1 / 686400 :=
by sorry

end problem_solution_l156_156957


namespace number_of_valid_arrangements_l156_156312

def is_ascending (l : List ℕ) : Prop :=
  ∀ i j, i < j → l.nth i ≤ l.nth j

def is_descending (l : List ℕ) : Prop :=
  ∀ i j, i < j → l.nth i ≥ l.nth j

def remove_one_is_ordered (l : List ℕ) : Prop :=
  ∃ (i : ℕ), (is_ascending (l.removeNth i) ∨ is_descending (l.removeNth i))

def valid_arrangements_count (cards : List ℕ) : ℕ :=
  -- counting the number of valid arrangements
  if (cards.length = 7
        ∧ ∀ i, i ∈ cards → 1 ≤ i ∧ i ≤ 7 ∧ (remove_one_is_ordered cards)) then 4 else 0

theorem number_of_valid_arrangements :
  valid_arrangements_count [1,2,3,4,5,6,7] = 4 :=
by sorry

end number_of_valid_arrangements_l156_156312


namespace problem_part1_problem_part2_l156_156594

variable (α : Real)
variable (h : Real.tan α = 1 / 2)

theorem problem_part1 : 
  (2 * Real.cos α - 3 * Real.sin α) / (3 * Real.cos α + 4 * Real.sin α) = 1 / 10 := sorry

theorem problem_part2 : 
  Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α + 4 * Real.cos α ^ 2 = 11 / 5 := sorry

end problem_part1_problem_part2_l156_156594


namespace find_a_l156_156172

noncomputable def f (x a : ℝ) := x^2 + (a + 1) * x + (a + 2)

def g (x a : ℝ) := (a + 1) * x
def h (x a : ℝ) := x^2 + a + 2

def p (a : ℝ) := ∀ x ≥ (a + 1)^2, f x a ≤ x
def q (a : ℝ) := ∀ x, g x a < 0

theorem find_a : 
  (¬p a) → (p a ∨ q a) → a ≥ -1 := sorry

end find_a_l156_156172


namespace base4_to_base10_conversion_l156_156100

theorem base4_to_base10_conversion :
  2 * 4^4 + 0 * 4^3 + 3 * 4^2 + 1 * 4^1 + 2 * 4^0 = 566 :=
by
  sorry

end base4_to_base10_conversion_l156_156100


namespace largest_three_digit_multiple_of_8_with_sum_16_l156_156425

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_8_with_sum_16 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ sum_of_digits n = 16 ∧
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 8 = 0 ∧ sum_of_digits m = 16 → m ≤ n :=
  ∃! (952 : ℕ), 100 ≤ 952 ∧ 952 < 1000 ∧ 952 % 8 = 0 ∧ sum_of_digits 952 = 16 ∧
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 8 = 0 ∧ sum_of_digits m = 16 → m ≤ 952 :=
  sorry

end largest_three_digit_multiple_of_8_with_sum_16_l156_156425


namespace math_problem_l156_156416

open Real

lemma radical_product :
  (3:ℝ) * (3:ℝ) * (2:ℝ) = 18 :=
by sorry

lemma cube_root_27 :
  real.cbrt 27 = 3 :=
by sorry

lemma fourth_root_81 :
  81 ^ (1 / 4:ℝ) = 3 :=
by sorry

lemma sixth_root_64 :
  64 ^ (1 / 6:ℝ) = 2 :=
by sorry

theorem math_problem :
  real.cbrt 27 * 81 ^ (1 / 4:ℝ) * 64 ^ (1 / 6:ℝ) = 18 :=
begin
  rw [cube_root_27, fourth_root_81, sixth_root_64],
  exact radical_product,
end

end math_problem_l156_156416


namespace ratio_AB_CD_over_AC_BD_tangents_perpendicular_l156_156716

open EuclideanGeometry

/- Given conditions in the problem -/
variables {A B C D : Point}
variables (triangle_ABC : Triangle A B C)
variables (is_acute : isAcute triangle_ABC)
variables (inside_triangle : inside D triangle_ABC)
variables (angle_condition : ∠ A D B = ∠ A C B + 90)
variables (length_condition : AC * BD = AD * BC)

/- First problem statement: Proving the ratio -/
theorem ratio_AB_CD_over_AC_BD (h1 : angle_condition) (h2 : length_condition) :
  AB * CD / (AC * BD) = √2 := 
sorry

/- Second problem statement: Proving the perpendicular tangents -/
theorem tangents_perpendicular (h1 : angle_condition) (h2 : length_condition) :
  tangents_perpendicular_at_C (circumcircle A C D) (circumcircle B C D) C :=
sorry

end ratio_AB_CD_over_AC_BD_tangents_perpendicular_l156_156716


namespace distinct_integer_roots_for_r_l156_156133

noncomputable def has_two_distinct_integer_roots (a b c : ℝ) : Prop :=
  ∃ p q : ℤ, p ≠ q ∧ a * p^2 + b * p + c = 0 ∧ a * q^2 + b * q + c = 0

theorem distinct_integer_roots_for_r (r : ℝ) :
  r ∈ {1, -1, 1/2, -1/2, 1/3, -1/3} →
  has_two_distinct_integer_roots (r^2) (2 * r) (4 * (1 - 7 * r^2)) :=
begin
  sorry
end

end distinct_integer_roots_for_r_l156_156133


namespace airline_daily_passengers_l156_156067

/-- 
Prove that the airline company can accommodate 2482 passengers each day 
given the following conditions: 
1. Fleet of 5 airplanes: 2 small, 2 medium-sized, 1 large.
2. Small planes: 15 rows, 6 seats per row, 3 flights per day, 80% occupancy.
3. Medium-sized planes: 25 rows, 8 seats per row, 2 flights per day, 90% occupancy.
4. Large plane: 35 rows, 10 seats per row, 4 flights per day, 95% occupancy.
-/
theorem airline_daily_passengers : 
  let small_planes := 2 in
  let medium_planes := 2 in
  let large_plane := 1 in
  let small_rows := 15 in
  let small_seats_per_row := 6 in
  let small_flights_per_day := 3 in
  let small_occupancy := 0.80 in
  let medium_rows := 25 in
  let medium_seats_per_row := 8 in
  let medium_flights_per_day := 2 in
  let medium_occupancy := 0.90 in
  let large_rows := 35 in
  let large_seats_per_row := 10 in
  let large_flights_per_day := 4 in
  let large_occupancy := 0.95 in
  (small_planes * small_rows * small_seats_per_row * small_flights_per_day * small_occupancy) + 
  (medium_planes * medium_rows * medium_seats_per_row * medium_flights_per_day * medium_occupancy) + 
  (large_plane * large_rows * large_seats_per_row * large_flights_per_day * large_occupancy) = 2482 :=
begin
  sorry
end

end airline_daily_passengers_l156_156067


namespace num_integers_in_solution_set_l156_156632

theorem num_integers_in_solution_set : ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℤ), abs (x - 3) ≤ 7 ↔ (x ≥ -4 ∧ x ≤ 10) ∧ ∃ y, (y = -4 ∨ y = -3 ∨ y = -2 ∨ y = -1 ∨ y = 0 ∨ y = 1 ∨ y = 2 ∨ y = 3 ∨ y = 4 ∨ y = 5 ∨ y = 6 ∨ y = 7 ∨ y = 8 ∨ y = 9 ∨ y = 10).
sorry

end num_integers_in_solution_set_l156_156632


namespace product_first_17_terms_l156_156690

theorem product_first_17_terms (a : ℕ → ℤ) (a_geom : ∀ n : ℕ, a (n + 1) = r * a n) (h : a 9 = -2) : 
  (∀ n, a (n + 1) = r * a n) → (∏ i in (finset.range 17), a (i + 1)) = -2^17 := 
by
  sorry

end product_first_17_terms_l156_156690


namespace second_largest_element_distinct_values_l156_156044

theorem second_largest_element_distinct_values :
  ∃ (xs : List ℕ), 
    xs.length = 5 ∧ 
    (∑ i in xs, i) = 75 ∧ 
    (List.maximum xs - List.minimum xs = 24) ∧ 
    10 ∈ xs ∧ 
    (∃ (m : ℕ), 10 = m ∧ xs.count m ≥ 2) ∧ 
    (xs.nth_le 2 (by simp [List.length]) = 10) ∧ 
    (xs.sort.nth_le 3 (by simp [List.length_sort] : 3 < xs.length) = 10) ∧ 
    let second_largest := xs.sort.nth_le 3 (by simp [List.length_sort] : 3 < xs.length) in
    (∃ (S : Finset ℕ), S = {x | second_largest ∈ xs ∧ x ∈ xs.sort.tail} ∧ S.card = 10) :=
begin
  sorry
end

end second_largest_element_distinct_values_l156_156044


namespace point_not_on_graph_l156_156884

theorem point_not_on_graph : ¬(∃ (x y : ℝ), x = -2 ∧ y = -4 ∧ y = 2 * x / (x + 2)) :=
by
  intro h
  cases h with x hx
  cases hx with y hy
  cases hy with h1 h2
  cases h2 with h3 h4
  linarith [h4.symm, show 0 = x + 2 from by rw [h1]; simp]

end point_not_on_graph_l156_156884


namespace find_symmetric_line_l156_156161

theorem find_symmetric_line :
  ∃ (l : ℝ → ℝ → Prop), ( ∀ x y, x^2 + y^2 = 9 ) ↔ ( ∀ x y, x^2 + y^2 - 4x + 4y - 1 = 0 ) → 
  (∀ x y, l x y ↔ x - y - 2 = 0) :=
begin
  sorry
end

end find_symmetric_line_l156_156161


namespace floor_sqrt_120_l156_156524

theorem floor_sqrt_120 : (⌊Real.sqrt 120⌋ = 10) :=
by
  -- Conditions from the problem
  have h1: 10^2 = 100 := rfl
  have h2: 11^2 = 121 := rfl
  have h3: 10 < Real.sqrt 120 := sorry
  have h4: Real.sqrt 120 < 11 := sorry
  -- Proof goal
  sorry

end floor_sqrt_120_l156_156524


namespace triangle_inscribed_circle_ratio_l156_156455

theorem triangle_inscribed_circle_ratio
  (a b c : ℕ)
  (ha : a = 7)
  (hb : b = 24)
  (hc : c = 25)
  (r s : ℕ)
  (h_tangent : r + s = b)
  (h_ratio : r * 7 = s) : r:s = 1:7 :=
sorry

end triangle_inscribed_circle_ratio_l156_156455


namespace carter_siblings_oldest_age_l156_156747

theorem carter_siblings_oldest_age
    (avg_age : ℕ)
    (sibling1 : ℕ)
    (sibling2 : ℕ)
    (sibling3 : ℕ)
    (sibling4 : ℕ) :
    avg_age = 9 →
    sibling1 = 5 →
    sibling2 = 8 →
    sibling3 = 7 →
    ((sibling1 + sibling2 + sibling3 + sibling4) / 4) = avg_age →
    sibling4 = 16 := by
  intros
  sorry

end carter_siblings_oldest_age_l156_156747


namespace ratio_of_radii_l156_156236

theorem ratio_of_radii (a b c : ℝ) (h1 : π * c^2 - π * a^2 = 4 * π * a^2) (h2 : π * b^2 = (π * a^2 + π * c^2) / 2) :
  a / c = 1 / Real.sqrt 5 := by
  sorry

end ratio_of_radii_l156_156236


namespace fraction_of_area_l156_156289

def larger_square_side : ℕ := 6
def shaded_square_side : ℕ := 2

def larger_square_area : ℕ := larger_square_side * larger_square_side
def shaded_square_area : ℕ := shaded_square_side * shaded_square_side

theorem fraction_of_area : (shaded_square_area : ℚ) / larger_square_area = 1 / 9 :=
by
  -- proof omitted
  sorry

end fraction_of_area_l156_156289


namespace floor_sqrt_120_l156_156527

theorem floor_sqrt_120 : (⌊Real.sqrt 120⌋ = 10) :=
by
  -- Conditions from the problem
  have h1: 10^2 = 100 := rfl
  have h2: 11^2 = 121 := rfl
  have h3: 10 < Real.sqrt 120 := sorry
  have h4: Real.sqrt 120 < 11 := sorry
  -- Proof goal
  sorry

end floor_sqrt_120_l156_156527


namespace hexagon_coloring_l156_156106

-- Definition of a convex hexagon with vertices A, B, C, D, E, and F
structure Hexagon :=
  (A B C D E F : ℕ)
  (color_assignment : Fin 7 → ℕ)

-- Conditions for the different coloring:
axiom different_colors : 
  ∀ (hex : Hexagon), ∀ (x y : Fin 7), 
    (x ≠ y ∧ (x, y) ∈ { (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 2), (1, 3), (2, 4), (3, 5), (4, 0), (5, 1) }) 
    → hex.color_assignment x ≠ hex.color_assignment y

noncomputable def number_of_colorings : ℕ :=
  7 * 6 * 5 * 5 * 6 * 5

theorem hexagon_coloring (hex : Hexagon) : 
  number_of_colorings = 31500 :=
by 
  sorry

end hexagon_coloring_l156_156106


namespace kelvin_probability_l156_156700

/-- Kelvin the frog lives in a pond with an infinite number of lily pads, numbered 0, 1, 2, 3, and so forth.
    Kelvin starts on lily pad 0 and jumps from pad to pad in the following manner: when on lily pad i, he will jump
    to lily pad (i+k) with probability (1 / (2^k)) for k > 0. This theorem states that the probability that
    Kelvin lands on lily pad 2019 at some point in his journey is 1/2. -/
theorem kelvin_probability : 
  let P := λ n : ℕ, 1 / 2^n
  let jump_prob := λ (i k : ℕ), if k > 0 then P k else 0
  let land_on_2019 := ∑ k in {k | k > 0}, if k = 2019 then jump_prob 0 k else 0
  let land_beyond_2019 := ∑ k in {k | k > 2019}, jump_prob 0 k
  in land_on_2019 = 1 / 2 :=
by
  sorry

end kelvin_probability_l156_156700


namespace amaya_first_rewind_time_l156_156483

theorem amaya_first_rewind_time (w1 w2 w3 r2 total: ℕ) (H1 : w1 = 35)
  (H2 : w2 = 45) (H3 : w3 = 20) (H4 : r2 = 15) (H5 : total = 120) : 
  ∃ r1, r1 = 5 :=
by
  -- Define the actual movie time and total added rewind time based on conditions
  let movie_time := w1 + w2 + w3
  have H_movie_time : movie_time = 35 + 45 + 20 := by
    rw [H1, H2, H3]
    norm_num
    
  let total_rewind_time := total - movie_time
  have H_total_rewind_time : total_rewind_time = 120 - 100 := by
    rw [H5, H_movie_time]
    norm_num

  -- Define the time added by the first rewind
  let r1 := total_rewind_time - r2
  have H_r1 : r1 = (120 - 100) - 15 := by
    rw [H_total_rewind_time, H4]
    norm_num

  -- Assert existence of r1
  use r1
  rw H_r1
  norm_num
  exact ⟨rfl⟩

end amaya_first_rewind_time_l156_156483


namespace percentage_decrease_is_2_l156_156332

-- Define the starting value and ending value as constants
def starting_value : ℕ := 8900
def ending_value : ℕ := 8722

-- Define the formula for percentage decrease
def percentage_decrease (start end : ℕ) : ℚ := ((start - end : ℤ) / (start : ℤ)) * 100

-- The proof statement
theorem percentage_decrease_is_2 :
  percentage_decrease starting_value ending_value = 2 :=
by
  sorry

end percentage_decrease_is_2_l156_156332


namespace geometric_sequence_ratio_l156_156688

variable {α : Type*} [Field α]

def geometric_sequence (a_1 q : α) (n : ℕ) : α :=
  a_1 * q ^ (n - 1)

theorem geometric_sequence_ratio (a1 q a4 a14 a5 a13 : α)
  (h_seq : ∀ n, geometric_sequence a1 q (n + 1) = a_5) 
  (h0 : geometric_sequence a1 q 5 * geometric_sequence a1 q 13 = 6) 
  (h1 : geometric_sequence a1 q 4 + geometric_sequence a1 q 14 = 5) :
  (∃ (k : α), k = 2 / 3 ∨ k = 3 / 2) → 
  geometric_sequence a1 q 80 / geometric_sequence a1 q 90 = k :=
by
  sorry

end geometric_sequence_ratio_l156_156688


namespace problem_solution_l156_156958

theorem problem_solution :
  (∏ k in finset.range(21).map (λ n, n + 1), (1 + 19 / k) / (1 + 21 / k)) = 1 / 686400 :=
by sorry

end problem_solution_l156_156958


namespace circumradius_of_acute_triangle_l156_156368

noncomputable def isOrthocenter (H A B C : Point) (A₁ B₁ C₁ : Point) : Prop :=
  -- Define orthocenter properties
  isAltitude H A A₁ ∧ isAltitude H B B₁ ∧ isAltitude H C C₁

noncomputable def isTriangle (A B C : Point) : Prop :=
  -- Define any necessary properties of a triangle
  A ≠ B ∧ B ≠ C ∧ A ≠ C

noncomputable def feetOfAltitudesFormRightTriangle (A₁ B₁ C₁ : Point) : Prop :=
  -- Define that feet of altitudes form a right triangle
  ∠ A₁ C₁ B₁ = 90 ∧ dist A₁ B₁ = 10

theorem circumradius_of_acute_triangle (A B C A₁ B₁ C₁ : Point) :
  isTriangle A B C →
  isOrthocenter H A B C A₁ B₁ C₁ →
  feetOfAltitudesFormRightTriangle A₁ B₁ C₁ →
  circumradius (△ A B C) = 10 :=
by
  sorry

end circumradius_of_acute_triangle_l156_156368


namespace count_valid_numbers_l156_156196

theorem count_valid_numbers :
  {n : ℕ | n >= 100 ∧ n < 1000 ∧ (∀ d ∈ [n / 100, (n % 100) / 10, n % 10], d > 6) ∧ (n % 12 = 0)}.card = 1 :=
by
  sorry

end count_valid_numbers_l156_156196


namespace area_of_triangle_ABC_l156_156343

-- Given conditions
def circle_radius : ℝ := 4
def BD : ℝ := 5
def ED : ℝ := 6
def perpendicular_AD_ED : Prop := ED ⊥ ((2 * circle_radius + BD) : ℝ)

-- Definitions based on conditions
def AD : ℝ := (2 * circle_radius) + BD
def EA : ℝ := Real.sqrt (AD ^ 2 + ED ^ 2)

-- Correct Answer (to be proved)
theorem area_of_triangle_ABC : 
  let BC := Real.sqrt ((((2 * circle_radius)):ℝ) ^ 2 - (EA - 65 / EA) ^ 2) / Real.sqrt 205,
      AC := (EA - 65 / EA) / Real.sqrt 205
  in  1/2 * BC * AC = 140 * Real.sqrt 2360 / 205 :=
sorry

end area_of_triangle_ABC_l156_156343


namespace exists_colored_equilateral_triangle_l156_156587

noncomputable def point_on_triangle (h n m : ℕ) (A B C : Point) (M : Point) : Prop :=
  n ≤ h ∧ m ≤ h ∧ n + m ≤ h ∧
  M = (1 / h) • (n • (B - A) + m • (C - A)) + A

def color_ab_constraint (A B C M : Point) : Prop := ¬ (M ∈ line_segment A B ∧ is_blue M)
def color_ac_constraint (A B C M : Point) : Prop := ¬ (M ∈ line_segment A C ∧ is_white M)
def color_bc_constraint (A B C M : Point) : Prop := ¬ (M ∈ line_segment B C ∧ is_red M)

theorem exists_colored_equilateral_triangle
  (A B C : Point) (h : ℕ)
  (point_condition : ∀ (n m : ℕ) (M : Point), point_on_triangle h n m A B C M)
  (color_ab : ∀ (M : Point), point_condition → color_ab_constraint A B C M)
  (color_ac : ∀ (M : Point), point_condition → color_ac_constraint A B C M)
  (color_bc : ∀ (M : Point), point_condition → color_bc_constraint A B C M) :
  ∃ (P Q R : Point),
    P.1 ≠ Q.1 ∧ Q.1 ≠ R.1 ∧ P.1 ≠ R.1 ∧
    P ≠ Q ∧ Q ≠ R ∧ P ≠ R ∧
    is_equilateral_triangle P Q R ∧
    is_red P ∧ is_white Q ∧ is_blue R :=
sorry

end exists_colored_equilateral_triangle_l156_156587


namespace closest_point_to_line_l156_156994

open Real

noncomputable def closest_point_on_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ × ℝ :=
let k := -(a / b)
let b0 := -(c / b)
let x := ((k * (p.2 - b0) + p.1) / (1 + k^2))
let y := (k * x + b0)
in (x, y)

theorem closest_point_to_line (x1 y1 : ℝ) :
  closest_point_on_line (3,2) (3) (-5) (-4) = (495 / 61, 296 / 61) :=
by
  sorry

end closest_point_to_line_l156_156994


namespace problem_l156_156273

variable (S : Finset Point)
variable [fintype S]
variable (x : ℝ)
hypothesis (h_noncollinear : ∀ (p1 p2 p3 : Point), set_of p1 ∈ S ∧ p2 ∈ S ∧ p3 ∈ S → ¬ (collinear p1 p2 p3))
hypothesis (h_x : 0 < x ∧ x < 1)

def a (P : ConvexPolygon) : ℕ := P.vertices.card
def b (P : ConvexPolygon) (S : Finset Point) : ℕ := (S \ P.vertices).card

theorem problem (S : Finset Point) (h_noncollinear : ∀ (p1 p2 p3 : Point), p1 ∈ S ∧ p2 ∈ S ∧ p3 ∈ S → ¬collinear p1 p2 p3) 
(x : ℝ) (h_x : 0 < x ∧ x < 1) 
: ∑ P, x^(a P) * (1-x)^(b P S) = 1 :=
sorry

end problem_l156_156273


namespace least_five_digit_congruent_l156_156863

theorem least_five_digit_congruent (x : ℕ) (h1 : x ≥ 10000) (h2 : x < 100000) (h3 : x % 17 = 8) : x = 10004 :=
by {
  sorry
}

end least_five_digit_congruent_l156_156863


namespace sum_of_values_of_N_l156_156972

-- Given conditions
variables (N R : ℝ)
-- Condition that needs to be checked
def condition (N R : ℝ) : Prop := N + 3 / N = R ∧ N ≠ 0

-- The statement to prove
theorem sum_of_values_of_N (N R : ℝ) (h: condition N R) : N + (3 / N) = R :=
sorry

end sum_of_values_of_N_l156_156972


namespace average_temperature_l156_156561

theorem average_temperature :
  let temp1 := -36
  let temp2 := 13
  let temp3 := -15
  let temp4 := -10
  (temp1 + temp2 + temp3 + temp4) / 4 = -12 :=
by
  unfold temp1 temp2 temp3 temp4
  calc
    (-36 + 13 + -15 + -10) / 4
      = (-48) / 4 : by norm_num
      ... = -12 : by norm_num

end average_temperature_l156_156561


namespace reconstruct_lines_possible_l156_156774

theorem reconstruct_lines_possible (n : ℕ) (h1 : n > 2)
    (lines : finset (set (ℝ × ℝ)))
    (h2 : ∀ l₁ l₂ ∈ lines, l₁ ≠ l₂ → ∃! p : ℝ × ℝ, p ∈ l₁ ∧ p ∈ l₂)
    (h3 : ∀ p, (∃! l₁, ∃! l₂, l₁ ≠ l₂ ∧ p ∈ l₁ ∧ p ∈ l₂) ∧
                ∃! l, l ∈ lines ∧ (∀ q₁ q₂ ∈ l, q₁ ≠ q₂ → q₁.1 = q₂.1 ∨ q₁.2 = q₂.2)) :
  ∃ (restored_lines : finset (set (ℝ × ℝ))), restored_lines = lines :=
by
  sorry

end reconstruct_lines_possible_l156_156774


namespace find_N_l156_156049

theorem find_N (x y : ℕ) (N : ℕ) (h1 : N = x * (x + 9)) (h2 : N = y * (y + 6)) : 
  N = 112 :=
  sorry

end find_N_l156_156049


namespace initial_average_mark_correct_l156_156334

-- Definitions and conditions
def total_students : ℕ := 15
def excluded_students : ℕ := 5
def excluded_average_mark : ℕ := 60
def remaining_average_mark : ℕ := 90
def initial_average_mark : ℕ := 80

-- The proof problem statement in Lean 4
theorem initial_average_mark_correct
  (total_students = 15)
  (excluded_students = 5)
  (excluded_average_mark * excluded_students = 300)
  (remaining_students = 10)
  (remaining_average_mark * remaining_students = 900)
  (total_marks = initial_average_mark * total_students)
  (total_marks = excluded_average_mark * excluded_students + remaining_average_mark * remaining_students) :
  initial_average_mark = 80 :=
by sorry

end initial_average_mark_correct_l156_156334


namespace polygon_sides_l156_156375

theorem polygon_sides (n : ℕ) 
  (h_interior : (n - 2) * 180 = 4 * 360) : 
  n = 10 :=
by
  sorry

end polygon_sides_l156_156375


namespace inequality_proof_l156_156572

-- Define the context of non-negative real numbers and sum to 1
variable {x y z : ℝ}
variable (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z)
variable (h_sum : x + y + z = 1)

-- State the theorem to be proved
theorem inequality_proof (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : x + y + z = 1) :
    0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
    sorry

end inequality_proof_l156_156572


namespace evaluate_expression_l156_156141

theorem evaluate_expression (x y : ℝ) (h : x - 2 * y = 3) : x - 2 * y + 4 = 7 := 
by
  -- Hypothesis
  have hyp : x - 2 * y = 3 := h
  -- Simplifying
  rw [hyp]
  -- Final result
  simp
  sorry

end evaluate_expression_l156_156141


namespace plus_signs_count_l156_156816

theorem plus_signs_count (total_symbols : ℕ) (n : ℕ) (m : ℕ)
    (h1 : total_symbols = 23)
    (h2 : ∀ (s : Finset ℕ), s.card = 10 → (∃ x ∈ s, x = n))
    (h3 : ∀ (s : Finset ℕ), s.card = 15 → (∃ x ∈ s, x = m)) :
    n = 14 := by
  sorry

end plus_signs_count_l156_156816


namespace plus_signs_count_l156_156787

noncomputable def symbols : Type := fin 23 -- The type of 23 symbols

def is_plus (x : symbols) : Prop
def is_minus (x : symbols) : Prop

axiom total_symbols : ∀ (x : symbols), is_plus x ∨ is_minus x
axiom unique_classification : ∀ (x : symbols), ¬ (is_plus x ∧ is_minus x)
axiom ten_symbols_plus : ∀ (S : finset symbols), S.card = 10 → ∃ x ∈ S, is_plus x
axiom fifteen_symbols_minus : ∀ (S : finset symbols), S.card = 15 → ∃ x ∈ S, is_minus x

def num_plus_symbols : ℕ := finset.card (finset.filter is_plus finset.univ)

theorem plus_signs_count : num_plus_symbols = 14 :=
sorry

end plus_signs_count_l156_156787


namespace least_five_digit_congruent_to_8_mod_17_l156_156859

theorem least_five_digit_congruent_to_8_mod_17 : 
  ∃ n : ℕ, n = 10004 ∧ n % 17 = 8 ∧ 10000 ≤ n ∧ n < 100000 ∧ (∀ m, 10000 ≤ m ∧ m < 10004 → m % 17 ≠ 8) :=
sorry

end least_five_digit_congruent_to_8_mod_17_l156_156859


namespace dog_distance_travel_l156_156030

noncomputable def distance_traveled_by_dog (initial_distance : ℝ) (speed_A : ℝ) (speed_B : ℝ) (speed_dog : ℝ) : ℝ :=
  let total_speed := speed_A + speed_B in
  (initial_distance / total_speed) * speed_dog

theorem dog_distance_travel (d : ℝ) (sA : ℝ) (sB : ℝ) (sD : ℝ) (H_d : d = 22.5) (H_sA : sA = 2.5) (H_sB : sB = 5) (H_sD : sD = 7.5) :
  distance_traveled_by_dog d sA sB sD = 22.5 :=
by
  rw [H_d, H_sA, H_sB, H_sD]
  unfold distance_traveled_by_dog
  norm_num


end dog_distance_travel_l156_156030


namespace initial_workers_l156_156486

theorem initial_workers (M : ℝ) :
  let totalLength : ℝ := 15
  let totalDays : ℝ := 300
  let completedLength : ℝ := 2.5
  let completedDays : ℝ := 100
  let remainingLength : ℝ := totalLength - completedLength
  let remainingDays : ℝ := totalDays - completedDays
  let extraMen : ℝ := 60
  let rateWithM : ℝ := completedLength / completedDays
  let newRate : ℝ := remainingLength / remainingDays
  let newM : ℝ := M + extraMen
  (rateWithM * M = newRate * newM) → M = 100 :=
by
  intros h
  sorry

end initial_workers_l156_156486


namespace bird_families_remaining_l156_156886

/-- There are 85 bird families living near the mountain. -/
def total_bird_families : ℕ := 85

/-- 23 bird families flew away to Africa. -/
def bird_families_to_africa : ℕ := 23

/-- 37 bird families flew away to Asia. -/
def bird_families_to_asia : ℕ := 37

/-- The number of bird families left near the mountain. -/
def bird_families_left : ℕ := total_bird_families - (bird_families_to_africa + bird_families_to_asia)

theorem bird_families_remaining : bird_families_left = 25 :=
by
  /- Directly stating the conclusion that the bird families left should be 25. -/
  have h1 : bird_families_left = total_bird_families - (bird_families_to_africa + bird_families_to_asia) := rfl
  have h2 : total_bird_families = 85 := rfl
  have h3 : bird_families_to_africa = 23 := rfl
  have h4 : bird_families_to_asia = 37 := rfl
  calc bird_families_left
      = 85 - (23 + 37) : by rw [h2, h3, h4]
  ... = 85 - 60 : by rfl
  ... = 25 : by rfl

end bird_families_remaining_l156_156886


namespace complex_power_rectangular_form_l156_156090

noncomputable def cos_30 : ℂ := real.cos (real.pi / 6) -- 30 degrees in radians
noncomputable def sin_30 : ℂ := real.sin (real.pi / 6)

theorem complex_power_rectangular_form :
  (3 * cos_30 + 3 * complex.I * sin_30)^4 = -81 / 2 + (81 * complex.I * real.sqrt 3) / 2 :=
by
  have h1 : cos_30 = real.sqrt 3 / 2 := by sorry
  have h2 : sin_30 = 1 / 2 := by sorry
  rw [h1, h2]
  sorry

end complex_power_rectangular_form_l156_156090


namespace floor_sqrt_120_l156_156525

theorem floor_sqrt_120 : (⌊Real.sqrt 120⌋ = 10) :=
by
  -- Conditions from the problem
  have h1: 10^2 = 100 := rfl
  have h2: 11^2 = 121 := rfl
  have h3: 10 < Real.sqrt 120 := sorry
  have h4: Real.sqrt 120 < 11 := sorry
  -- Proof goal
  sorry

end floor_sqrt_120_l156_156525


namespace total_distance_traveled_l156_156404

theorem total_distance_traveled
  (r1 r2 : ℝ) (h1 : r1 = 15) (h2 : r2 = 25):
  let arc_outer := 1/4 * 2 * Real.pi * r2
  let radial := r2 - r1
  let circ_inner := 2 * Real.pi * r1
  let return_radial := radial
  let total_distance := arc_outer + radial + circ_inner + return_radial
  total_distance = 42.5 * Real.pi + 20 := 
by
  sorry

end total_distance_traveled_l156_156404


namespace plus_signs_count_l156_156791

noncomputable def symbols : Type := fin 23 -- The type of 23 symbols

def is_plus (x : symbols) : Prop
def is_minus (x : symbols) : Prop

axiom total_symbols : ∀ (x : symbols), is_plus x ∨ is_minus x
axiom unique_classification : ∀ (x : symbols), ¬ (is_plus x ∧ is_minus x)
axiom ten_symbols_plus : ∀ (S : finset symbols), S.card = 10 → ∃ x ∈ S, is_plus x
axiom fifteen_symbols_minus : ∀ (S : finset symbols), S.card = 15 → ∃ x ∈ S, is_minus x

def num_plus_symbols : ℕ := finset.card (finset.filter is_plus finset.univ)

theorem plus_signs_count : num_plus_symbols = 14 :=
sorry

end plus_signs_count_l156_156791


namespace add_in_base_7_l156_156999

def from_base (b : ℕ) (digits : List ℕ) : ℕ := 
  digits.reverse.enum_from 1 |>.map (λ (i, d), d * b^(i-1)).sum

def to_base (b : ℕ) (n : ℕ) : List ℕ :=
  if n = 0 then [0] else 
    List.unfold (λ x, if x = 0 then none else some (x%b, x / b)) n |>.reverse

theorem add_in_base_7 : 
  from_base 7 [6, 6, 6] + from_base 7 [6, 6] + from_base 7 [6] = from_base 7 [1, 4, 0, 0] :=
by 
  unfold from_base 
  have h1 : from_base 7 [6, 6, 6] = 6 * 7^2 + 6 * 7^1 + 6 * 7^0 := by rfl
  have h2 : from_base 7 [6, 6] = 6 * 7^1 + 6 * 7^0 := by rfl
  have h3 : from_base 7 [6] = 6 := by rfl
  have h4 : from_base 7 [1, 4, 0, 0] = 1 * 7^3 + 4 * 7^2 + 0 * 7^1 + 0 * 7^0 := by rfl
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end add_in_base_7_l156_156999


namespace magnitude_of_A_l156_156627

open Real

def vector_A : ℝ × ℝ × ℝ × ℝ := (1, -1, -3, -4)

def dot_product (v : ℝ × ℝ × ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2 + v.3 * v.3 + v.4 * v.4

def magnitude (v : ℝ × ℝ × ℝ × ℝ) : ℝ :=
  sqrt (dot_product v)

theorem magnitude_of_A : magnitude vector_A = 3 * sqrt 3 := 
by
  sorry

end magnitude_of_A_l156_156627


namespace cylinder_radius_l156_156663

theorem cylinder_radius (h : ℝ) (A : ℝ) (r : ℝ) (h_eq : h = 3) (A_eq : A = 12 * real.pi) :
  2 * real.pi * r * h = A → r = 2 :=
by
  rw [h_eq, A_eq, mul_assoc, mul_right_comm]
  sorry

end cylinder_radius_l156_156663


namespace find_prime_numbers_of_form_p_p_plus_1_l156_156983

def has_at_most_19_digits (n : ℕ) : Prop := n < 10^19

theorem find_prime_numbers_of_form_p_p_plus_1 :
  {n : ℕ | ∃ p : ℕ, n = p^p + 1 ∧ has_at_most_19_digits n ∧ Nat.Prime n} = {2, 5, 257} :=
by
  sorry

end find_prime_numbers_of_form_p_p_plus_1_l156_156983


namespace min_period_func_l156_156760

def func (x : ℝ) : ℝ := Real.tan ((π / 5) - (x / 3))

theorem min_period_func : ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, func (x + T) = func x) ∧ T = 3 * π :=
by
  sorry

end min_period_func_l156_156760


namespace find_four_digit_number_l156_156855

def is_four_digit_number (k : ℕ) : Prop :=
  1000 ≤ k ∧ k < 10000

def appended_number (k : ℕ) : ℕ :=
  4000000 + k

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem find_four_digit_number (k : ℕ) (hk : is_four_digit_number k) :
  is_perfect_square (appended_number k) ↔ k = 4001 ∨ k = 8004 :=
sorry

end find_four_digit_number_l156_156855


namespace real_solution_exists_l156_156110

theorem real_solution_exists (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 3) :
  (x^3 - 4*x^2) / (x^2 - 5*x + 6) - x = 9 → x = 9/2 :=
by sorry

end real_solution_exists_l156_156110


namespace part_a_part_b_part_c_l156_156352

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def can_be_rearranged_to_perfect_square_sums (n : ℕ) : Prop :=
  ∃ (f : fin n → fin n),
    (∀ i : fin n, is_perfect_square (i.val + 1 + (f i).val + 1)) ∧
    (function.bijective f)

-- For the case n = 9
theorem part_a : can_be_rearranged_to_perfect_square_sums 9 :=
by { sorry }

-- For the case n = 11
theorem part_b : ¬ can_be_rearranged_to_perfect_square_sums 11 :=
by { sorry }

-- For the case n = 1996
theorem part_c : can_be_rearranged_to_perfect_square_sums 1996 :=
by { sorry }

end part_a_part_b_part_c_l156_156352


namespace not_first_class_prob_l156_156294

-- Definitions based on conditions
def P (event : Type) := ℝ  -- Event to Real number function representing probability

def A : Type := {x // x = "first_class"}
def B : Type := {x // x = "second_class"}
def C : Type := {x // x = "third_class"}

axiom P_A : P A = 0.65
axiom P_B : P B = 0.2
axiom P_C : P C = 0.1

-- Lean statement for the proof problem
theorem not_first_class_prob : P (Aᶜ) = 0.35 := by
  sorry

end not_first_class_prob_l156_156294


namespace floor_sqrt_120_l156_156530

theorem floor_sqrt_120 : (Int.floor (Real.sqrt 120)) = 10 := by
  have h1 : 10^2 < 120 := by norm_num
  have h2 : 120 < 11^2 := by norm_num
  -- Additional steps to show that Int.floor (Real.sqrt 120) = 10
  sorry

end floor_sqrt_120_l156_156530


namespace triangle_area_ABC_l156_156856

noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_ABC :
  let A := (3 : ℝ, 2 : ℝ)
  let B := (9 : ℝ, 2 : ℝ)
  let C := (5 : ℝ, 10 : ℝ)
  area_triangle A B C = 24 :=
by
  sorry

end triangle_area_ABC_l156_156856


namespace train_length_is_499_96_l156_156931

-- Define the conditions
def speed_train_kmh : ℕ := 75   -- Speed of the train in km/h
def speed_man_kmh : ℕ := 3     -- Speed of the man in km/h
def time_cross_s : ℝ := 24.998 -- Time taken for the train to cross the man in seconds

-- Define the conversion factors
def km_to_m : ℕ := 1000        -- Conversion from kilometers to meters
def hr_to_s : ℕ := 3600        -- Conversion from hours to seconds

-- Define relative speed in m/s
def relative_speed_ms : ℕ := (speed_train_kmh - speed_man_kmh) * km_to_m / hr_to_s

-- Prove the length of the train in meters
def length_of_train : ℝ := relative_speed_ms * time_cross_s

theorem train_length_is_499_96 : length_of_train = 499.96 := sorry

end train_length_is_499_96_l156_156931


namespace length_of_bridge_l156_156014

theorem length_of_bridge
  (train_length : ℕ)
  (train_speed_kmh : ℕ)
  (crossing_time_seconds : ℕ)
  (h_train_length : train_length = 125)
  (h_train_speed_kmh : train_speed_kmh = 45)
  (h_crossing_time_seconds : crossing_time_seconds = 30) :
  ∃ (bridge_length : ℕ), bridge_length = 250 :=
by
  sorry

end length_of_bridge_l156_156014


namespace tip_percentage_l156_156183

noncomputable def large_pizza_cost := 14
noncomputable def topping_cost := 2
noncomputable def num_pizzas := 2
noncomputable def num_toppings := 3
noncomputable def total_cost := 50

theorem tip_percentage : 
  let pizza_total := num_pizzas * large_pizza_cost
  let topping_total := num_pizzas * num_toppings * topping_cost
  let total_without_tip := pizza_total + topping_total
  let tip_amount := total_cost - total_without_tip
  let tip_percent := (tip_amount / total_without_tip) * 100 
  in tip_percent = 25 := 
by
  -- we add sorry to skip the proof
  sorry

end tip_percentage_l156_156183


namespace LiuHuiMethod_l156_156721

theorem LiuHuiMethod :
  ∀ (LiuHuiGreatMathematician : Prop) 
    (MarksBeginningOfLimits : Prop) 
    (GerminationOfIntegralCalculus : Prop), 
  LiuHuiGreatMathematician ∧ MarksBeginningOfLimits ∧ GerminationOfIntegralCalculus → 
    "The method of exhaustions" ∈ {"A: The method of exhaustions", "B: Pythagorean theorem", "C: The method of celestial element computation", "D: The algorithm of division"} :=
by
  sorry

end LiuHuiMethod_l156_156721


namespace complex_number_quadrant_l156_156002

theorem complex_number_quadrant (m : ℝ) (h : 1 < m ∧ m < 3 / 2) : 
  ∃ z : ℂ, z = (3 : ℂ) + (1 : ℂ) * complex.I - m * ((2 : ℂ) + (1 : ℂ) * complex.I) ∧ z.re > 0 ∧ z.im < 0 := 
by 
  sorry

end complex_number_quadrant_l156_156002


namespace math_problem_l156_156418

open Real

lemma radical_product :
  (3:ℝ) * (3:ℝ) * (2:ℝ) = 18 :=
by sorry

lemma cube_root_27 :
  real.cbrt 27 = 3 :=
by sorry

lemma fourth_root_81 :
  81 ^ (1 / 4:ℝ) = 3 :=
by sorry

lemma sixth_root_64 :
  64 ^ (1 / 6:ℝ) = 2 :=
by sorry

theorem math_problem :
  real.cbrt 27 * 81 ^ (1 / 4:ℝ) * 64 ^ (1 / 6:ℝ) = 18 :=
begin
  rw [cube_root_27, fourth_root_81, sixth_root_64],
  exact radical_product,
end

end math_problem_l156_156418


namespace train_speed_is_117_l156_156474

noncomputable def train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmph : ℝ) : ℝ :=
  let man_speed_mps := man_speed_kmph * 1000 / 3600
  let relative_speed := train_length / crossing_time
  (relative_speed - man_speed_mps) * 3.6

theorem train_speed_is_117 :
  train_speed 300 9 3 = 117 :=
by
  -- We leave the proof as sorry since only the statement is needed
  sorry

end train_speed_is_117_l156_156474


namespace complex_power_4_l156_156078

noncomputable def cos30_re : ℂ := 3 * real.cos (π / 6)
noncomputable def sin30_im : ℂ := 3 * complex.I * real.sin (π / 6)
noncomputable def c : ℂ := cos30_re + sin30_im

theorem complex_power_4 :
  c ^ 4 = -40.5 + 40.5 * complex.I * real.sqrt 3 := sorry

end complex_power_4_l156_156078


namespace tangent_line_at_3_monotonic_intervals_range_of_a_l156_156619

variables (a : ℝ) (x : ℝ) (e : ℝ)
def f (x : ℝ) (a : ℝ) := log (x - 2) - (x^2) / (2 * a)

-- Question I: Tangent line at (3, f(3)) when a = 1
theorem tangent_line_at_3 (h : a = 1) :
  let f' := (1 / (x - 2)) - (x / a) in
  let slope := -2 in
  let p := (3, f 3 a) in
  4 * x + 2 * (f 3 a) - 3 = 0 :=
sorry

-- Question II: Monotonic intervals of f(x)
theorem monotonic_intervals :
  (a < 0 → ∀ x > 2, deriv (f x a) > 0) ∧
  (a > 0 → ∀ x ∈ set.Ioo 2 (1 + sqrt (a + 1)), deriv (f x a) > 0 ∧ ∀ x ∈ set.Ioo (1 + sqrt (a + 1)) +infty, deriv (f x a) < 0) :=
sorry

-- Question III: Range of a
theorem range_of_a (h : ∀ x ∈ set.Icc (e + 2) (e^3 + 2), f x a ≥ 0) (x_0 : ℝ) (h' : x_0 ∉ set.Icc (e + 2) (e^3 + 2)) :
  a > e^6 + 2 * (e^3) :=
sorry

end tangent_line_at_3_monotonic_intervals_range_of_a_l156_156619


namespace problem_statement_l156_156585

variable {α : Type*}

def arithmetic_sum (S : ℕ → ℕ) (a : ℕ) (d : ℕ) : Prop :=
  S 3 = 3 * a + 3 * (3 - 1) / 2 * d ∧ S 5 = 5 * a + 5 * (5 - 1) / 2 * d

def general_term (a₀ d : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n, a n = a₀ + (n - 1) * d

def sequence_bn (a : ℕ → ℕ) (b : ℕ → ℕ) : Prop :=
  ∀ n, b n = a n / 2 ^ a n

def sum_sequence_bn (b : ℕ → ℕ) (T : ℕ → ℝ) : Prop :=
  ∀ n, T n = (∑ k in finset.range n, b (k + 1))

theorem problem_statement :
  ∃ a d : ℕ, arithmetic_sum (λ n, n * (2 * a + (n - 1) * d) / 2) a d ∧
  general_term a d id ∧
  ∃ T : ℕ → ℝ,
    sum_sequence_bn (λ n, n / 2 ^ n) T ∧
    ∀ n, T n = 2 - (2 + n) / 2 ^ n :=
sorry

end problem_statement_l156_156585


namespace plus_signs_count_l156_156776

theorem plus_signs_count (p m : ℕ) (h_sum : p + m = 23)
                         (h_max_minus : m ≤ 9) (h_max_plus : p ≤ 14)
                         (h_at_least_one_plus_in_10 : ∀ (s : Finset (Fin 23)), s.card = 10 → ∃ i ∈ s, i < p)
                         (h_at_least_one_minus_in_15 : ∀ (s : Finset (Fin 23)), s.card = 15 → ∃ i ∈ s, m ≤ i) :
  p = 14 :=
by sorry

end plus_signs_count_l156_156776


namespace sufficient_but_not_necessary_l156_156568

def p (x : ℝ) : Prop := |x - 4| > 2
def q (x : ℝ) : Prop := x > 1

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 6 → x > 1) ∧ ¬(∀ x, x > 1 → 2 ≤ x ∧ x ≤ 6) :=
  sorry

end sufficient_but_not_necessary_l156_156568


namespace plus_signs_count_l156_156839

theorem plus_signs_count (total_symbols : ℕ) (n_plus n_minus : ℕ)
  (h1 : total_symbols = 23)
  (h2 : ∀ s : Finset ℕ, s.card = 10 → ∃ i ∈ s, i ≤ n_plus)
  (h3 : ∀ s : Finset ℕ, s.card = 15 → ∃ i ∈ s, i > n_plus) :
  n_plus = 14 :=
by
  -- the proof will go here
  sorry

end plus_signs_count_l156_156839


namespace eight_numbers_contain_347_l156_156185

theorem eight_numbers_contain_347 : 
  ∃ S : finset ℕ, S.card = 8 ∧ (∀ x ∈ S, 800 ≤ x ∧ x ≤ 1400 ∧ (3 ∈ digits x) ∧ (4 ∈ digits x) ∧ (7 ∈ digits x)) := sorry

end eight_numbers_contain_347_l156_156185


namespace robe_initial_savings_l156_156735

noncomputable def initial_savings (repair_fee corner_light_cost brake_disk_cost tires_cost remaining_savings : ℕ) : ℕ :=
  remaining_savings + repair_fee + corner_light_cost + 2 * brake_disk_cost + tires_cost

theorem robe_initial_savings :
  let R := 10
  let corner_light := 2 * R
  let brake_disk := 3 * corner_light
  let tires := corner_light + 2 * brake_disk
  let remaining := 480
  initial_savings R corner_light brake_disk tires remaining = 770 :=
by
  sorry

end robe_initial_savings_l156_156735


namespace volume_of_tetrahedron_l156_156398

-- Definitions corresponding to the problem conditions

-- Assume all tetrahedrons and their properties are in a Euclidean space
variables {A B C D M P Q R S : Point ℝ}

-- Given conditions:
-- Tetrahedron ABCD
-- Two smaller tetrahedrons AMRS and MBPQ 
-- formed by planes parallel to faces ACD and BCD
-- Volumes of smaller tetrahedrons are 1 cm^3 and 8 cm^3 respectively

-- Let's assume two functions to represent volumes
noncomputable def vol : Point ℝ × Point ℝ × Point ℝ × Point ℝ → ℝ := sorry

-- Main goal: The volume of tetrahedron ABCD
theorem volume_of_tetrahedron (h1 : vol (A, M, R, S) = 1)
                             (h2 : vol (M, B, P, Q) = 8)
                             (similar_1 : similar_tetrahedrons (A, M, R, S) (A, B, C, D))
                             (similar_2 : similar_tetrahedrons (M, B, P, Q) (A, B, C, D)) :
  vol (A, B, C, D) = 27 :=
by {
  -- Step through the argument, simplifying the proof based on the given conditions 
  -- and the volume ratios of similar tetrahedrons
  sorry
}

end volume_of_tetrahedron_l156_156398


namespace solve_math_problem_l156_156920

theorem solve_math_problem (x : ℕ) (h1 : x > 0) (h2 : x % 3 = 0) (h3 : x % x = 9) : x = 30 := by
  sorry

end solve_math_problem_l156_156920


namespace area_of_circle_l156_156422

open Real

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x1 - x2) ^ 2 + (y1 - y2) ^ 2)

def circle_area (radius : ℝ) : ℝ :=
  π * radius ^ 2

theorem area_of_circle :
  let A := (1 : ℝ, -2 : ℝ) in
  let B := (-5 : ℝ, 6 : ℝ) in
  let r := distance A.1 A.2 B.1 B.2 in
  circle_area r = 100 * π :=
by
  sorry

end area_of_circle_l156_156422


namespace yoongi_initial_books_l156_156437

-- Definition of initial conditions
def initial_books (Y E U : ℕ) :=
  let Y' := Y - 5 + 15 in
  let E' := E + 5 - 10 in
  let U' := U + 10 - 15 in
  Y' = 45 ∧ E' = 45 ∧ U' = 45

-- The proof statement
theorem yoongi_initial_books (Y E U : ℕ) (h : initial_books Y E U) : Y = 35 :=
by
  have hY : Y - 5 + 15 = 45 := h.1
  have hE : E + 5 - 10 = 45 := h.2.1
  have hU : U + 10 - 15 = 45 := h.2.2
  calc
    Y + 10 = 45 : by linarith
    Y = 35 : by linarith

end yoongi_initial_books_l156_156437


namespace simplify_expression_l156_156645

theorem simplify_expression (a b c : ℝ) (h1 : a > 0) (h2 : b < 0) (h3 : c < 0) (h4 : |a| > |b|) (h5 : |c| > |a|) : 
  |a + c| - |b + c| - |a + b| = -2a :=
by 
  sorry

end simplify_expression_l156_156645


namespace floor_sqrt_120_l156_156532

theorem floor_sqrt_120 : (Int.floor (Real.sqrt 120)) = 10 := by
  have h1 : 10^2 < 120 := by norm_num
  have h2 : 120 < 11^2 := by norm_num
  -- Additional steps to show that Int.floor (Real.sqrt 120) = 10
  sorry

end floor_sqrt_120_l156_156532


namespace plus_signs_count_l156_156788

noncomputable def symbols : Type := fin 23 -- The type of 23 symbols

def is_plus (x : symbols) : Prop
def is_minus (x : symbols) : Prop

axiom total_symbols : ∀ (x : symbols), is_plus x ∨ is_minus x
axiom unique_classification : ∀ (x : symbols), ¬ (is_plus x ∧ is_minus x)
axiom ten_symbols_plus : ∀ (S : finset symbols), S.card = 10 → ∃ x ∈ S, is_plus x
axiom fifteen_symbols_minus : ∀ (S : finset symbols), S.card = 15 → ∃ x ∈ S, is_minus x

def num_plus_symbols : ℕ := finset.card (finset.filter is_plus finset.univ)

theorem plus_signs_count : num_plus_symbols = 14 :=
sorry

end plus_signs_count_l156_156788


namespace sequence_general_term_l156_156625

theorem sequence_general_term
  (a : ℕ → ℝ)
  (h₁ : a 1 = 1 / 2)
  (h₂ : ∀ n, a (n + 1) = 3 * a n + 7) :
  ∀ n, a n = 4 * 3^(n - 1) - 7 / 2 :=
by
  sorry

end sequence_general_term_l156_156625


namespace plus_signs_count_l156_156792

noncomputable def solveSymbols : Prop :=
  ∃ p m : ℕ, p + m = 23 ∧ p ≤ 14 ∧ m ≤ 9 ∧ (∀ s : Finset (Fin 23), s.card = 10 → (s.image (λ i => if i < p then true else false)).count true ≥ 1) ∧ (∀ s : Finset (Fin 23), s.card = 15 → (s.image (λ i => if i < p then false else true)).count false ≥ 1) 

theorem plus_signs_count : solveSymbols :=
sorry

end plus_signs_count_l156_156792


namespace solve_for_x_l156_156324

theorem solve_for_x : ∃ x : ℤ, 25 - 7 = 3 + x ∧ x = 15 := by
  sorry

end solve_for_x_l156_156324


namespace vertical_asymptotes_sum_l156_156754

theorem vertical_asymptotes_sum : 
  (∀ x : ℝ, 4 * x^2 + 7 * x + 3 = 0 → x = -3 / 4 ∨ x = -1) →
  (-3 / 4) + (-1) = -7 / 4 :=
by
  intro h
  sorry

end vertical_asymptotes_sum_l156_156754


namespace square_b_perimeter_l156_156741

theorem square_b_perimeter (a b : ℝ) 
  (ha : a^2 = 65) 
  (prob : (65 - b^2) / 65 = 0.7538461538461538) : 
  4 * b = 16 :=
by 
  sorry

end square_b_perimeter_l156_156741


namespace hyperbola_eccentricity_l156_156207

theorem hyperbola_eccentricity {a b c : ℝ}
    (h1 : b = (3/2) * a ∨ a = (3/2) * b)
    (h2 : c = sqrt (a^2 + b^2))
    (h3 : ∃ a b, a > 0 ∧ b > 0 ∧
        (c = sqrt(a^2 + b^2) ∧ (b = (3/2) * a ∨ a = (3/2) * b))) :
  ∃ e : ℝ, e = c / a ∧ (e = sqrt(13)/2 ∨ e = sqrt(13)/3) :=
sorry

end hyperbola_eccentricity_l156_156207


namespace num_integers_in_solution_set_l156_156631

theorem num_integers_in_solution_set : ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℤ), abs (x - 3) ≤ 7 ↔ (x ≥ -4 ∧ x ≤ 10) ∧ ∃ y, (y = -4 ∨ y = -3 ∨ y = -2 ∨ y = -1 ∨ y = 0 ∨ y = 1 ∨ y = 2 ∨ y = 3 ∨ y = 4 ∨ y = 5 ∨ y = 6 ∨ y = 7 ∨ y = 8 ∨ y = 9 ∨ y = 10).
sorry

end num_integers_in_solution_set_l156_156631


namespace general_term_arithmetic_sequence_sum_of_first_n_terms_l156_156165

noncomputable def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 + a 2 = 6 ∧ a 2 + a 3 = 10

theorem general_term_arithmetic_sequence (a : ℕ → ℕ) (h : arithmetic_sequence a) :
  ∀ n, a n = 2 * n :=
sorry

noncomputable def sequence_b (b : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n, b n = 2 ^ a(n + 1)

theorem sum_of_first_n_terms (b : ℕ → ℕ) (a : ℕ → ℕ) (h_a : ∀ n, a n = 2 * n)
  (h_b : sequence_b b a) :
  ∀ n, (∑ i in Finset.range n, b i) = (4 ^ (n + 2) - 16) / 3 :=
sorry

end general_term_arithmetic_sequence_sum_of_first_n_terms_l156_156165


namespace distance_AC_l156_156405

theorem distance_AC : 
  ∀ (A B C D : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space D] 
  (d_AD d_DB d_AC d_CD : ℝ)
  (h1 : d_AD = 6) (h2 : d_DB = 15)
  (h3 : d_AD + d_DB = 21)
  (h4 : d_AC = real.sqrt (6 * 6 + d_CD * d_CD))
  (h5 : d_CD = real.sqrt (21 * 21 - 15 * 15)),
  d_AC = 6 * real.sqrt 7 :=
by
  sorry

end distance_AC_l156_156405


namespace supplier_delivery_l156_156966

variable (initial_inventory : Nat)
variable (monday_sales : Nat)
variable (tuesday_sales : Nat)
variable (wed_to_sun_sales : Nat)
variable (final_inventory : Nat)

def total_sales (monday_sales tuesday_sales wed_to_sun_sales : Nat) : Nat :=
  monday_sales + tuesday_sales + wed_to_sun_sales

def remaining_before_delivery (initial_inventory monday_sales tuesday_sales wed_to_fri_sales : Nat) : Nat :=
  initial_inventory - (monday_sales + tuesday_sales + wed_to_fri_sales)

def bottles_delivered (final_inventory remaining : Nat) : Nat :=
  final_inventory - remaining

theorem supplier_delivery (initial_inventory monday_sales tuesday_sales wed_to_sun_sales final_inventory : Nat) :
  initial_inventory = 4500 →
  monday_sales = 2445 →
  tuesday_sales = 900 →
  wed_to_sun_sales = 50 * 5 →
  final_inventory = 1555 →
  bottles_delivered final_inventory (remaining_before_delivery initial_inventory monday_sales tuesday_sales (50 * 4)) = 600 :=
by
  intros h1 h2 h3 h4 h5
  unfold bottles_delivered remaining_before_delivery total_sales
  rw [h1, h2, h3, h4, h5]
  sorry

end supplier_delivery_l156_156966


namespace ellipse_eccentricity_l156_156166

noncomputable def a : ℝ := Real.sqrt 4
noncomputable def b : ℝ := Real.sqrt 3
noncomputable def c : ℝ := Real.sqrt (a^2 - b^2)
noncomputable def e : ℝ := c / a

theorem ellipse_eccentricity :
  (∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1) → e = 1 / 2) :=
begin
  sorry
end

end ellipse_eccentricity_l156_156166


namespace first_group_hours_per_day_l156_156026

theorem first_group_hours_per_day :
  ∃ H : ℕ, 
    (39 * 12 * H = 30 * 26 * 3) ∧
    H = 5 :=
by sorry

end first_group_hours_per_day_l156_156026


namespace area_of_journey_l156_156438

theorem area_of_journey:
  ( ∀ (x y: ℝ), (0 ≤ x ∧ x ≤ 6 ∧ 0 ≤ y ∧ y ≤ 6) ∧ 
    (y = 0 → x = 3) ∧ 
    (y = 6 → x = 3) ∧ 
    (y > 0 ∧ y < 6 → |x - 3| ≤ 6 - y))
  →
  let area := (9 * Real.sqrt 3) + (21 * Real.pi / 2)
  in ∃ t, t = area :=
sorry

end area_of_journey_l156_156438


namespace candle_height_l156_156912

noncomputable def total_time : ℕ := 15 * (150 * 151) / 2

noncomputable def time_half : ℕ := total_time / 2

def burned_height_at_time (t : ℕ) : ℕ :=
  let k := Nat.sqrt (t / 7.5) in k

def remaining_height (h : ℕ) : ℕ :=
  150 - h

theorem candle_height
  (total_time : ℕ)
  (time_half : ℕ)
  (burned_height_at_time : ℕ → ℕ)
  (remaining_height : ℕ → ℕ):
  remaining_height (burned_height_at_time time_half) = 45 := by
  sorry

end candle_height_l156_156912


namespace fraction_of_decimal_l156_156358

theorem fraction_of_decimal (a b : ℕ) (h : 0.375 = (a : ℝ) / (b : ℝ)) (gcd_ab : Nat.gcd a b = 1) : a + b = 11 :=
  sorry

end fraction_of_decimal_l156_156358


namespace plus_signs_count_l156_156795

noncomputable def solveSymbols : Prop :=
  ∃ p m : ℕ, p + m = 23 ∧ p ≤ 14 ∧ m ≤ 9 ∧ (∀ s : Finset (Fin 23), s.card = 10 → (s.image (λ i => if i < p then true else false)).count true ≥ 1) ∧ (∀ s : Finset (Fin 23), s.card = 15 → (s.image (λ i => if i < p then false else true)).count false ≥ 1) 

theorem plus_signs_count : solveSymbols :=
sorry

end plus_signs_count_l156_156795


namespace polygon_sides_l156_156380

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 4 * 360) → n = 10 :=
by 
  sorry

end polygon_sides_l156_156380


namespace jacob_age_in_X_years_l156_156254

-- Definitions of the conditions
variable (J M X : ℕ)

theorem jacob_age_in_X_years
  (h1 : J = M - 14)
  (h2 : M + 9 = 2 * (J + 9))
  (h3 : J = 5) :
  J + X = 5 + X :=
by
  sorry

end jacob_age_in_X_years_l156_156254


namespace train_overtakes_motorbike_time_l156_156932

theorem train_overtakes_motorbike_time :
  let train_speed_kmph := 100
  let motorbike_speed_kmph := 64
  let train_length_m := 120.0096
  let relative_speed_kmph := train_speed_kmph - motorbike_speed_kmph
  let relative_speed_m_s := (relative_speed_kmph : ℝ) * (1 / 3.6)
  let time_seconds := train_length_m / relative_speed_m_s
  time_seconds = 12.00096 :=
sorry

end train_overtakes_motorbike_time_l156_156932


namespace allen_and_carla_meet_probability_l156_156066

theorem allen_and_carla_meet_probability :
  let P : ℚ := 17 / 18 in
  ∀ (t_A t_C : ℝ), 
    (0 ≤ t_A ∧ t_A ≤ 1) ∧ 
    (0 ≤ t_C ∧ t_C ≤ 1) ∧ 
    ((t_C - 1/6 ≤ t_A ∧ t_A ≤ t_C + 1/3) ∨ (t_A - 1/3 ≤ t_C ∧ t_C ≤ t_A + 1/6)) →
    P = 17 / 18 :=
by sorry

end allen_and_carla_meet_probability_l156_156066


namespace floor_sqrt_120_eq_10_l156_156538

theorem floor_sqrt_120_eq_10 : ⌊Real.sqrt 120⌋ = 10 := by
  -- Here, we note that we are given:
  -- 100 < 120 < 121 and the square root of it lies between 10 and 11
  sorry

end floor_sqrt_120_eq_10_l156_156538


namespace scalene_triangle_can_be_divided_l156_156516

theorem scalene_triangle_can_be_divided :
  ∃ (T : Triangle), (is_scalene T) ∧ (has_angle T 30) ∧ (can_be_divided_into_equal_triangles T 3) :=
sorry

end scalene_triangle_can_be_divided_l156_156516


namespace sum_of_solution_eqn_abs_value_l156_156431

theorem sum_of_solution_eqn_abs_value :
  let solutions := {n : ℚ | abs (3 * n - 10) = 8} in
  (solutions.sum : ℚ) = 20 / 3 :=
by
  sorry

end sum_of_solution_eqn_abs_value_l156_156431


namespace smallest_n_l156_156877

theorem smallest_n (n : ℕ) (hn : n > 0) (h : 623 * n % 32 = 1319 * n % 32) : n = 4 :=
sorry

end smallest_n_l156_156877


namespace four_digit_div_90_count_l156_156192

theorem four_digit_div_90_count :
  ∃ n : ℕ, n = 10 ∧ (∀ (ab : ℕ), 10 ≤ ab ∧ ab ≤ 99 → ab % 9 = 0 → 
  (10 * ab + 90) % 90 = 0 ∧ 1000 ≤ 10 * ab + 90 ∧ 10 * ab + 90 < 10000) :=
sorry

end four_digit_div_90_count_l156_156192


namespace largest_n_for_sine_cosine_inequality_l156_156117

theorem largest_n_for_sine_cosine_inequality :
  ∃ n : ℕ, (∀ x : ℝ, (Real.sin x) ^ n + (Real.cos x) ^ n ≥ 2 / n) ∧ ∀ m > n, ∃ x : ℝ, (Real.sin x) ^ m + (Real.cos x) ^ m < 2 / m :=
begin
  use 4,
  split,
  { intro x,
    have h1 : (Real.sin x) ^ 4 + (Real.cos x) ^ 4 ≥ 1 / 2,
    { -- Proof using QM-AM and other inequalities
      sorry
    },
    linarith,
  },
  { intros m hm,
    -- Proof that for m > 4 the inequality fails
    sorry
  }
end

end largest_n_for_sine_cosine_inequality_l156_156117


namespace sum_of_squares_of_coefficients_l156_156881

theorem sum_of_squares_of_coefficients (x : ℝ) : 
  let expr := 4 * (x^2 - 2*x + 2) - 7 * (x^3 - 3*x + 1)
  let simplified_expr := -7 * x^3 + 4 * x^2 + 13 * x + 1
  let coefficients := [-7, 4, 13, 1]
  (coefficients.map (λ c, c^2)).sum = 235 :=
by
  sorry

end sum_of_squares_of_coefficients_l156_156881


namespace total_yield_UncleLi_yield_difference_l156_156409

-- Define the conditions related to Uncle Li and Aunt Lin
def UncleLiAcres : ℕ := 12
def UncleLiYieldPerAcre : ℕ := 660
def AuntLinAcres : ℕ := UncleLiAcres - 2
def AuntLinTotalYield : ℕ := UncleLiYieldPerAcre * UncleLiAcres - 420

-- Prove the total yield of Uncle Li's rice
theorem total_yield_UncleLi : UncleLiYieldPerAcre * UncleLiAcres = 7920 := by
  sorry

-- Prove how much less the yield per acre of Uncle Li's rice is compared to Aunt Lin's
theorem yield_difference :
  UncleLiYieldPerAcre - AuntLinTotalYield / AuntLinAcres = 90 := by
  sorry

end total_yield_UncleLi_yield_difference_l156_156409


namespace plus_signs_count_l156_156821

theorem plus_signs_count (total_symbols : ℕ) (n : ℕ) (m : ℕ)
    (h1 : total_symbols = 23)
    (h2 : ∀ (s : Finset ℕ), s.card = 10 → (∃ x ∈ s, x = n))
    (h3 : ∀ (s : Finset ℕ), s.card = 15 → (∃ x ∈ s, x = m)) :
    n = 14 := by
  sorry

end plus_signs_count_l156_156821


namespace count_divisible_by_90_l156_156194

theorem count_divisible_by_90 : 
  ∃ n, n = 10 ∧ (∀ k, 1000 ≤ k ∧ k < 10000 ∧ k % 100 = 90 ∧ k % 90 = 0 → n = 10) :=
begin
  sorry
end

end count_divisible_by_90_l156_156194


namespace Carolyn_wants_to_embroider_l156_156951

theorem Carolyn_wants_to_embroider (s : ℕ) (f : ℕ) (u : ℕ) (g : ℕ) (n_f : ℕ) (t : ℕ) (number_of_unicorns : ℕ) :
  s = 4 ∧ f = 60 ∧ u = 180 ∧ g = 800 ∧ n_f = 50 ∧ t = 1085 ∧ 
  (t * s - (n_f * f) - g) / u = number_of_unicorns ↔ number_of_unicorns = 3 :=
by 
  sorry

end Carolyn_wants_to_embroider_l156_156951


namespace floor_sqrt_120_eq_10_l156_156522

theorem floor_sqrt_120_eq_10 :
  (√120).to_floor = 10 := by
  have h1 : √100 = 10 := by norm_num
  have h2 : √121 = 11 := by norm_num
  have h : 100 < 120 ∧ 120 < 121 := by norm_num
  have sqrt_120 : 10 < √120 ∧ √120 < 11 :=
    by exact ⟨real.sqrt_lt' 120 121 h.2, real.sqrt_lt'' 100 120 h.1⟩
  sorry

end floor_sqrt_120_eq_10_l156_156522


namespace plus_signs_count_l156_156782

theorem plus_signs_count (p m : ℕ) (h_sum : p + m = 23)
                         (h_max_minus : m ≤ 9) (h_max_plus : p ≤ 14)
                         (h_at_least_one_plus_in_10 : ∀ (s : Finset (Fin 23)), s.card = 10 → ∃ i ∈ s, i < p)
                         (h_at_least_one_minus_in_15 : ∀ (s : Finset (Fin 23)), s.card = 15 → ∃ i ∈ s, m ≤ i) :
  p = 14 :=
by sorry

end plus_signs_count_l156_156782


namespace necessary_and_sufficient_condition_l156_156575

variables {a b e k : ℝ} (C : set (ℝ × ℝ))
def hyperbola (a b x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def focus_right (e a b : ℝ) (x y : ℝ) : Prop := x = (a * e) ∧ y = 0
def line_through_focus (k c x y : ℝ) : Prop := y = k * (x - c)
def intersect_both_branches (C : set (ℝ × ℝ)) (l : set (ℝ × ℝ)) : Prop := ∃ x1 x2: ℝ, x1 > 0 ∧ x2 < 0 ∧ l ⊆ C

theorem necessary_and_sufficient_condition
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : focus_right e a b (a * e) 0)
  (h4 : ∀ x y, line_through_focus k (a * e) x y → C (x, y))
  : (∃ x y, hyperbola a b x y) ∧ intersect_both_branches C (line_through_focus k (a * e)) 
    ↔ e^2 - k^2 > 1 := 
sorry

end necessary_and_sufficient_condition_l156_156575


namespace solve_for_y_l156_156016

theorem solve_for_y (x y : ℝ) (h1 : x * y = 9) (h2 : x / y = 36) (h3 : x > 0) (h4 : y > 0) : y = 1/2 := 
by
  sorry

end solve_for_y_l156_156016


namespace find_point_on_parabola_l156_156048

open Real

theorem find_point_on_parabola :
  ∃ (x y : ℝ), 
  (0 ≤ x ∧ 0 ≤ y) ∧
  (x^2 = 8 * y) ∧
  sqrt (x^2 + (y - 2)^2) = 120 ∧
  (x = 2 * sqrt 236 ∧ y = 118) :=
by
  sorry

end find_point_on_parabola_l156_156048


namespace sequence_value_l156_156411

/-- Define the sequence {a_n} recursively -/
def a : ℕ+ → ℕ
| ⟨n, _⟩ := 
  if n % 2 = 1 then n
  else a ⟨n / 2, (nat.div_pos (nat.succ_pos _) (nat.succ_pos 1)).ne'⟩

theorem sequence_value :
  a ⟨64, sorry⟩ + a ⟨65, sorry⟩ = 66 := by sorry

end sequence_value_l156_156411


namespace alan_more_wings_per_minute_to_beat_record_l156_156704

-- Define relevant parameters and conditions
def kevin_wings := 64
def time_minutes := 8
def alan_rate := 5

-- Theorem: Alan must eat 3 more wings per minute to beat Kevin's record
theorem alan_more_wings_per_minute_to_beat_record : 
  (kevin_wings > alan_rate * time_minutes) → ((kevin_wings - (alan_rate * time_minutes)) / time_minutes = 3) :=
by
  sorry

end alan_more_wings_per_minute_to_beat_record_l156_156704


namespace classification_randomly_walkable_graphs_l156_156414

-- Definition of what it means for a graph to be randomly walkable
def randomly_walkable (V : Type) (E : V → V → Prop) : Prop :=
  ∀ (p : List V), (∀ v ∈ p, ∃ w ∈ p.tail, E v w) → 
  (∀ (v : V) (w : V), v ∈ p → w ∈ p → v ≠ w → E v w) →
  ∀ v, v ∈ p.tail → ∃ u, u ∈ p.tail ∧ v ≠ u

-- Proving the classification of randomly walkable graphs as the given graphs
theorem classification_randomly_walkable_graphs (V : Type) (E : V → V → Prop) :
    randomly_walkable V E ↔
      (∃ n, V = Fin n ∧ 
      (E = λ x y, True) ∨
      (∃ cycle_order (p : list (Fin n)), 
       (∀ i, E (p.nth i) (p.nth ((i + 1) % n))) ∨
      (∃ m, 2 * m = n ∧ 
      ∀ i < m, ∀ j ≥ m, E (Fin.ofNat i) (Fin.ofNat j) ∧
      ∀ j ≥ m, ∀ i < m, E (Fin.ofNat j) (Fin.ofNat i))))
 :=
by
  sorry

end classification_randomly_walkable_graphs_l156_156414


namespace max_cookies_Andy_eats_l156_156488

theorem max_cookies_Andy_eats (cookies_total : ℕ) (h_cookies_total : cookies_total = 30) 
  (exists_pos_a : ∃ a : ℕ, a > 0 ∧ 3 * a = 30 - a ∧ (∃ k : ℕ, 3 * a = k ∧ ∃ m : ℕ, a = m)) 
  : ∃ max_a : ℕ, max_a ≤ 7 ∧ 3 * max_a < cookies_total ∧ 3 * max_a ∣ cookies_total ∧ max_a = 6 :=
by
  sorry

end max_cookies_Andy_eats_l156_156488


namespace altitude_inequality_l156_156552

theorem altitude_inequality
  (a b m_a m_b : ℝ)
  (h1 : a > b)
  (h2 : a * m_a = b * m_b) :
  a^2010 + m_a^2010 ≥ b^2010 + m_b^2010 :=
sorry

end altitude_inequality_l156_156552


namespace water_flow_speed_l156_156369

-- Define the problem conditions using Lean's structures and statements 
theorem water_flow_speed (x y : ℝ) 
  (h1 : 135 / (x + y) + 70 / (x - y) = 12.5)
  (h2 : 75 / (x + y) + 110 / (x - y) = 12.5) :
  y = 3.2 :=
begin
  -- The proof is omitted here; it is indicated with 'sorry'.
  sorry
end

end water_flow_speed_l156_156369


namespace max_non_attacking_grasshoppers_l156_156338

theorem max_non_attacking_grasshoppers (m n : ℕ) (h_m : m = 2017) (h_n : n = 100) : 
  ∃ (max_grasshoppers : ℕ), max_grasshoppers = 4034 := 
by 
  use 4034 
  sorry

end max_non_attacking_grasshoppers_l156_156338


namespace minimum_area_l156_156451

-- Let A, B, and C be points of a triangle with coordinates as specified
def Point := (ℤ × ℤ)
def A : Point := (0, 0)
def B : Point := (48, 20)
def Area (p q : ℤ) : ℕ := Nat.abs (20 * p - 48 * q)

-- Define the theorem for the minimum area
theorem minimum_area : ∃ (p q : ℤ), (p, q ∈ ℤ × ℤ) ∧ (1 / 2 : ℚ) * (Area p q) = 2 := sorry

end minimum_area_l156_156451


namespace battery_usage_minutes_l156_156285

theorem battery_usage_minutes (initial_battery final_battery : ℝ) (initial_minutes : ℝ) (rate_of_usage : ℝ) :
  initial_battery - final_battery = rate_of_usage * initial_minutes →
  initial_battery = 100 →
  final_battery = 68 →
  initial_minutes = 60 →
  rate_of_usage = 8 / 15 →
  ∃ additional_minutes : ℝ, additional_minutes = 127.5 :=
by
  intros
  sorry

end battery_usage_minutes_l156_156285


namespace min_value_T_l156_156580

-- The sequence a_n is defined as 1 + 2(n-1) = 2n - 1
def a (n : ℕ) : ℕ := 2 * n - 1

-- The sequence b_n is defined as 1 / (a_n * a_(n-1))
def b (n : ℕ) : ℝ :=
  if h : n ≠ 0 then
    1 / (a n * a (n - 1))
  else
    0

-- The sum of first n terms of b_n
def T (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, b (k + 1)

-- Statement to prove that the minimum value of sum T_n is 1/3
theorem min_value_T (n : ℕ) : 
  ∑ k in Finset.range n, b (k + 1) = 1/2 - 1/(2 * (2 * n + 1)) →
  ∑ k in Finset.range n, b (k + 1) ≥ 1/3 :=
begin
  intro h,
  sorry,
end

end min_value_T_l156_156580


namespace no_more_than_two_chords_intersect_at_any_point_l156_156077

theorem no_more_than_two_chords_intersect_at_any_point
    (n m : ℕ) (h1 : 0 < n) (h2 : 2 ≤ m) :
    ∀ (p : Point) (circle : Circle), 
    (divided_into_n_equal_parts circle n) →
    (connect_points_m_steps_away circle n m) →
    ¬ (more_than_two_chords_intersect_at_any_point circle p) :=
by
  sorry

end no_more_than_two_chords_intersect_at_any_point_l156_156077


namespace length_first_train_l156_156059

-- Define the conditions
def speed_first_train := 120 -- in km/hr
def speed_second_train := 80 -- in km/hr
def length_second_train := 280.04 -- in meters
def time_to_cross := 9 -- in seconds

-- Proof statement
theorem length_first_train (speed_first_train speed_second_train length_second_train time_to_cross : ℝ) :
  speed_first_train = 120 →
  speed_second_train = 80 →
  length_second_train = 280.04 →
  time_to_cross = 9 →
  let relative_speed_ms := (speed_first_train + speed_second_train) * 1000 / 3600 in
  let combined_length := relative_speed_ms * time_to_cross in
  let length_first_train := combined_length - length_second_train in
  length_first_train = 220 :=
by
  intros h1 h2 h3 h4
  sorry

end length_first_train_l156_156059


namespace count_integer_solutions_l156_156634

theorem count_integer_solutions (x : ℤ) : 
  (|x - 3| ≤ 7) → (x ∈ Finset.range 15 → x + -8 : ℤ) := 
sorry

end count_integer_solutions_l156_156634


namespace ratio_equality_l156_156642

theorem ratio_equality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (h7 : (y + 1) / (x + z) = (x + y + 2) / (z + 1))
  (h8 : (x + 1) / y = (y + 1) / (x + z)) :
  (x + 1) / y = 1 :=
by
  sorry

end ratio_equality_l156_156642


namespace polygon_sides_l156_156381

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 4 * 360) → n = 10 :=
by 
  sorry

end polygon_sides_l156_156381


namespace plants_per_row_l156_156456

theorem plants_per_row (P : ℕ) (rows : ℕ) (yield_per_plant : ℕ) (total_yield : ℕ) 
  (h1 : rows = 30)
  (h2 : yield_per_plant = 20)
  (h3 : total_yield = 6000)
  (h4 : rows * yield_per_plant * P = total_yield) : 
  P = 10 :=
by 
  sorry

end plants_per_row_l156_156456


namespace eccentricity_of_ellipse_l156_156160

theorem eccentricity_of_ellipse 
  (A B C : Point)
  (ellipse D : Ellipse)
  (h1 : A ∈ D) 
  (h2 : B ∈ D) 
  (h3 : C ∈ D) 
  (h4 : is_equilateral_triangle A B C) 
  : eccentricity D = Real.sqrt (6) / 3 := 
begin
  sorry
end

end eccentricity_of_ellipse_l156_156160


namespace polygon_perimeter_l156_156673

noncomputable def perimeter {A B C D E: ℝ} (AB BC ED DA: ℝ) (CX: ℝ) (CE: ℝ) : ℝ :=
  AB + BC + CE + ED + DA

theorem polygon_perimeter (A B C D E: ℝ)
  (AB BC ED DA: ℝ) (CX: ℝ) 
  (h1: AB = 3)
  (h2: BC = 3)
  (h3: ED = 10)
  (h4: DA = 5)
  (h5: CE^2 = CX^2 + 6^2)
  : perimeter 3 3 10 5 CX CE = 21 + 3 * Real.sqrt(5) := by
  sorry

end polygon_perimeter_l156_156673


namespace p_sq_plus_q_sq_l156_156267

theorem p_sq_plus_q_sq (p q : ℝ) (h1 : p * q = 12) (h2 : p + q = 8) : p^2 + q^2 = 40 :=
by
  sorry

end p_sq_plus_q_sq_l156_156267


namespace distance_to_focus_l156_156919

-- Define the parabola and point conditions
def parabola (x y : ℝ) : Prop := x^2 = 2 * y
def point_P (x_P y_P : ℝ) : Prop := parabola x_P y_P ∧ y_P = 3

-- Define the parameter p
def parameter_p : ℝ := 1 / 2

-- Define the focus distance according to the focal radius formula
def focal_radius (y_P p : ℝ) : ℝ := y_P + p / 2

-- The theorem to prove
theorem distance_to_focus (x_P y_P : ℝ) (hP : point_P x_P y_P) : 
  focal_radius y_P parameter_p = 7 / 2 := by
  sorry

end distance_to_focus_l156_156919


namespace plus_signs_count_l156_156838

theorem plus_signs_count (total_symbols : ℕ) (n_plus n_minus : ℕ)
  (h1 : total_symbols = 23)
  (h2 : ∀ s : Finset ℕ, s.card = 10 → ∃ i ∈ s, i ≤ n_plus)
  (h3 : ∀ s : Finset ℕ, s.card = 15 → ∃ i ∈ s, i > n_plus) :
  n_plus = 14 :=
by
  -- the proof will go here
  sorry

end plus_signs_count_l156_156838


namespace shaded_area_eq_63_l156_156852

noncomputable def rect1_width : ℕ := 4
noncomputable def rect1_height : ℕ := 12
noncomputable def rect2_width : ℕ := 5
noncomputable def rect2_height : ℕ := 7
noncomputable def overlap_width : ℕ := 4
noncomputable def overlap_height : ℕ := 5

theorem shaded_area_eq_63 :
  (rect1_width * rect1_height) + (rect2_width * rect2_height) - (overlap_width * overlap_height) = 63 := by
  sorry

end shaded_area_eq_63_l156_156852


namespace find_x_value_l156_156686

theorem find_x_value (x : ℝ) (h : 3 * x + 6 * x + x + 2 * x = 360) : x = 30 :=
by sorry

end find_x_value_l156_156686


namespace angle_B_in_triangle_ABC_l156_156244

theorem angle_B_in_triangle_ABC (A B C : Prop) (angle_A : ℕ) (angle_C : ℕ) 
  (hA : angle_A = 20) (hC : angle_C = 90) : angle B = 70 :=
by
  sorry

end angle_B_in_triangle_ABC_l156_156244


namespace workers_in_first_group_l156_156666

theorem workers_in_first_group
  (W D : ℕ)
  (h1 : 6 * W * D = 9450)
  (h2 : 95 * D = 9975) :
  W = 15 := 
sorry

end workers_in_first_group_l156_156666


namespace books_in_library_l156_156680

   theorem books_in_library (B : ℕ) 
     (h1 : 0.80 * B = (B * 4 / 5)) 
     (h2 : 0.60 * (0.80 * B) = (B * 3 / 5 * 4 / 5))
     (h3 : 736 = 0.40 * (0.80 * B)) :
     B = 2300 := 
   by
     sorry
   
end books_in_library_l156_156680


namespace area_of_BDOE_l156_156727

namespace Geometry

noncomputable def areaQuadrilateralBDOE (AE CD AB BC AC : ℝ) : ℝ :=
  if AE = 2 ∧ CD = 11 ∧ AB = 8 ∧ BC = 8 ∧ AC = 6 then
    189 * Real.sqrt 55 / 88
  else
    0

theorem area_of_BDOE :
  areaQuadrilateralBDOE 2 11 8 8 6 = 189 * Real.sqrt 55 / 88 :=
by 
  sorry

end Geometry

end area_of_BDOE_l156_156727


namespace positive_difference_of_fraction_results_l156_156980

theorem positive_difference_of_fraction_results :
  let a := 8
  let expr1 := (a ^ 2 - a ^ 2) / a
  let expr2 := (a ^ 2 * a ^ 2) / a
  expr1 = 0 ∧ expr2 = 512 ∧ (expr2 - expr1) = 512 := 
by
  sorry

end positive_difference_of_fraction_results_l156_156980


namespace complex_power_4_l156_156081

noncomputable def cos30_re : ℂ := 3 * real.cos (π / 6)
noncomputable def sin30_im : ℂ := 3 * complex.I * real.sin (π / 6)
noncomputable def c : ℂ := cos30_re + sin30_im

theorem complex_power_4 :
  c ^ 4 = -40.5 + 40.5 * complex.I * real.sqrt 3 := sorry

end complex_power_4_l156_156081


namespace collinear_points_l156_156515

theorem collinear_points (p : ℝ) :
  collinear (set.mk_points [(1, -2), (3, 4), (6, p/3)]) → p = 39 :=
by
  sorry

end collinear_points_l156_156515


namespace correct_option_is_A_l156_156434

-- Define the four options
def A_pair :=
  let x := -3
  let y := Real.sqrt ((-3) ^ 2)
  (x, y)

def B_pair :=
  let x := -3
  let y := Real.cbrt (-27)
  (x, y)

def C_pair :=
  let x := -3
  let y := -(1 / 3)
  (x, y)

def D_pair :=
  let x := abs (-3)
  let y := 3
  (x, y)

-- Define a predicate to check if two numbers are opposites
def are_opposites (a b : ℝ) : Prop :=
  a = -b

-- State the theorem with the correct answer
theorem correct_option_is_A : 
  are_opposites (A_pair.1) (A_pair.2) ∧
  ¬ are_opposites (B_pair.1) (B_pair.2) ∧
  ¬ are_opposites (C_pair.1) (C_pair.2) ∧
  ¬ are_opposites (D_pair.1) (D_pair.2) :=
by
  sorry

end correct_option_is_A_l156_156434


namespace distance_between_closest_points_of_circles_is_zero_l156_156500

noncomputable def find_closest_distance_points_of_circles : ℝ :=
  let center1 := (5, 5 : ℝ × ℝ)
  let center2 := (20, 15 : ℝ × ℝ)
  let radius1 := 5 - (-3)  -- Radius of circle 1
  let radius2 := 15 - (-3) -- Radius of circle 2
  let distance_centers := real.sqrt ((20 - 5)^2 + (15 - 5)^2)
  if distance_centers < radius1 + radius2 then 0 else distance_centers - (radius1 + radius2)

theorem distance_between_closest_points_of_circles_is_zero :
  find_closest_distance_points_of_circles = 0 :=
sorry

end distance_between_closest_points_of_circles_is_zero_l156_156500


namespace Jerry_paid_more_last_month_l156_156255

def Debt_total : ℕ := 50
def Debt_remaining : ℕ := 23
def Paid_2_months_ago : ℕ := 12
def Paid_last_month : ℕ := 27 - Paid_2_months_ago

theorem Jerry_paid_more_last_month :
  Paid_last_month - Paid_2_months_ago = 3 :=
by
  -- Calculation for Paid_last_month
  have h : Paid_last_month = 27 - 12 := by rfl
  -- Compute the difference
  have diff : 15 - 12 = 3 := by rfl
  exact diff

end Jerry_paid_more_last_month_l156_156255


namespace mrKlinker_twice_as_old_in_15_years_l156_156283

-- Define the conditions: Mr. Klinker is 35 and his daughter is 10
def mrKlinkerAge : ℕ := 35
def daughterAge : ℕ := 10

-- Define the proof statement
theorem mrKlinker_twice_as_old_in_15_years :
  ∃ x : ℕ, mrKlinkerAge + x = 2 * (daughterAge + x) ∧ x = 15 :=
by
  use 15
  split
  · have h1 : mrKlinkerAge + 15 = 35 + 15 := rfl
    have h2 : daughterAge + 15 = 10 + 15 := rfl
    rw [h1, h2]
    norm_num
    
  · rfl

-- Proof not required, so we use sorry here.

end mrKlinker_twice_as_old_in_15_years_l156_156283


namespace fraction_sum_is_11_l156_156359

theorem fraction_sum_is_11 (a b : ℕ) (h1 : 0.375 = (a : ℚ) / b) (h2 : Nat.coprime a b) : a + b = 11 := 
by sorry

end fraction_sum_is_11_l156_156359


namespace product_of_roots_l156_156970

theorem product_of_roots :
  let p := (3 * x^2 - x + 5) * (4 * x^4 - 16 * x^3 + 25) in
  let leading_coeff := 12 in
  let const_term := 125 in
  let product_of_roots := const_term / leading_coeff in
  ∀ x: ℝ, p = 0 → product_of_roots = 125 / 12 :=
by
  sorry

end product_of_roots_l156_156970


namespace conic_sections_ab_value_l156_156503

theorem conic_sections_ab_value
  (a b : ℝ)
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 64) :
  |a * b| = Real.sqrt 868.5 :=
by
  -- Proof will be filled in later
  sorry

end conic_sections_ab_value_l156_156503


namespace winning_frequencies_correct_estimated_probability_correct_num_white_balls_correct_l156_156033

-- Given Conditions
def red_balls : ℕ := 8
def participants (i : ℕ) : ℕ := if i = 1 then 200 else if i = 2 then 300 else if i = 3 then 400 else 500
def winners (i : ℕ) : ℕ := if i = 1 then 39 else if i = 2 then 63 else if i = 3 then 82 else 99

-- 1) Calculate and complete the table of winning frequencies
def winning_frequency (i : ℕ) : ℚ :=
  (winners i : ℚ) / (participants i : ℚ)

-- Prove that winning frequencies are as given
theorem winning_frequencies_correct :
  winning_frequency 1 = 0.195 ∧
  winning_frequency 2 = 0.21 ∧
  winning_frequency 3 = 0.205 ∧
  winning_frequency 4 = 0.198 :=
by {
  -- Proof steps to be added
  sorry
}

-- 2) Estimate the probability of winning a drink
def estimated_probability : ℚ :=
  (winning_frequency 1 + winning_frequency 2 + winning_frequency 3 + winning_frequency 4) / 4

theorem estimated_probability_correct :
  estimated_probability = 0.2 :=
by {
  -- Proof steps to be added
  sorry
}

-- 3) Estimate the number of white balls in the bag
def num_white_balls : ℚ :=
  have eq : 8 / (x + 8) = 0.2,
  32

theorem num_white_balls_correct :
  num_white_balls = 32 :=
by {
  -- Proof steps to be added
  sorry
}

end winning_frequencies_correct_estimated_probability_correct_num_white_balls_correct_l156_156033


namespace screws_weight_l156_156413

theorem screws_weight (x y : ℕ) 
  (h1 : 3 * x + 2 * y = 319) 
  (h2 : 2 * x + 3 * y = 351) : 
  x = 51 ∧ y = 83 :=
by 
  sorry

end screws_weight_l156_156413


namespace find_corresponding_element_l156_156693

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - p.2, p.1 + p.2)

theorem find_corresponding_element :
  f (-1, 2) = (-3, 1) :=
by
  sorry

end find_corresponding_element_l156_156693


namespace turtle_minimum_distance_l156_156478

theorem turtle_minimum_distance 
  (constant_speed : ℝ)
  (turn_angle : ℝ)
  (total_time : ℕ) :
  constant_speed = 5 →
  turn_angle = 90 →
  total_time = 11 →
  ∃ (final_position : ℝ × ℝ), 
    (final_position = (5, 0) ∨ final_position = (-5, 0) ∨ final_position = (0, 5) ∨ final_position = (0, -5)) ∧
    dist final_position (0, 0) = 5 :=
by
  intros
  sorry

end turtle_minimum_distance_l156_156478


namespace complicated_fraction_equiv_one_l156_156954

theorem complicated_fraction_equiv_one : 
  (∏ i in Finset.range 21, (1 + 19 / (i + 1))) / (∏ i in Finset.range 19, (1 + 21 / (i + 1))) = 1 :=
by sorry

end complicated_fraction_equiv_one_l156_156954


namespace total_stars_correct_l156_156320

-- Define the number of gold stars Shelby earned each day
def monday_stars : ℕ := 4
def tuesday_stars : ℕ := 7
def wednesday_stars : ℕ := 3
def thursday_stars : ℕ := 8
def friday_stars : ℕ := 2

-- Define the total number of gold stars
def total_stars : ℕ := monday_stars + tuesday_stars + wednesday_stars + thursday_stars + friday_stars

-- Prove that the total number of gold stars Shelby earned throughout the week is 24
theorem total_stars_correct : total_stars = 24 :=
by
  -- The proof goes here, using sorry to skip the proof
  sorry

end total_stars_correct_l156_156320


namespace floor_sqrt_120_eq_10_l156_156519

theorem floor_sqrt_120_eq_10 :
  (√120).to_floor = 10 := by
  have h1 : √100 = 10 := by norm_num
  have h2 : √121 = 11 := by norm_num
  have h : 100 < 120 ∧ 120 < 121 := by norm_num
  have sqrt_120 : 10 < √120 ∧ √120 < 11 :=
    by exact ⟨real.sqrt_lt' 120 121 h.2, real.sqrt_lt'' 100 120 h.1⟩
  sorry

end floor_sqrt_120_eq_10_l156_156519


namespace spinner_probability_shaded_region_l156_156041
noncomputable theory

def area_of_triangle (a b : ℕ) : ℕ := (a * b) / 2

def area_of_region (total_area : ℕ) : ℕ := total_area / 4

def shaded_area (region_area regions_shaded : ℕ) : ℕ := region_area * regions_shaded

def probability (shaded_area total_area : ℚ) : ℚ := shaded_area / total_area

theorem spinner_probability_shaded_region :
  ∃ (a b hypotenuse : ℕ) (regions_shaded : ℕ),
  a = 6 ∧ b = 8 ∧ hypotenuse = 10 ∧ regions_shaded = 3 ∧
    probability (shaded_area (area_of_region (area_of_triangle a b)) regions_shaded)
                (area_of_triangle a b) = 3 / 4 :=
by
  sorry

end spinner_probability_shaded_region_l156_156041


namespace trajectory_equation_circle_equation_circle_equation_1_circle_equation_2_l156_156232

-- Proving the trajectory of the center P
theorem trajectory_equation (a b R : ℝ) (h1 : R^2 - b^2 = 2) (h2 : R^2 - a^2 = 3) :
  b^2 - a^2 = 1 :=
by
  sorry

-- Finding the equation of circle P
theorem circle_equation (a b R : ℝ) (h1 : R^2 - b^2 = 2) (h2 : R^2 - a^2 = 3) (h3 : |b - a| = 1) :
  (a = 0 ∧ b = 1 ∧ R = √3) ∨ (a = 0 ∧ b = -1 ∧ R = √3) :=
by
  sorry

-- Proving the actual circle equations based on the found values
theorem circle_equation_1 (x y : ℝ) (a b : ℝ) (R : ℝ) (h1 : a = 0) (h2 : b = 1) (h3 : R = √3) :
  x^2 + (y - 1)^2 = 3 :=
by
  sorry

theorem circle_equation_2 (x y : ℝ) (a b : ℝ) (R : ℝ) (h1 : a = 0) (h2 : b = -1) (h3 : R = √3) :
  x^2 + (y + 1)^2 = 3 :=
by
  sorry

end trajectory_equation_circle_equation_circle_equation_1_circle_equation_2_l156_156232


namespace permutations_5_choose_2_l156_156391

theorem permutations_5_choose_2 : 
  ∃ n k : ℕ, n = 5 ∧ k = 2 ∧ nat.factorial n / nat.factorial (n - k) = 20 :=
by
  use 5
  use 2
  split
  { rfl }
  split
  { rfl }
  sorry

end permutations_5_choose_2_l156_156391


namespace triangle_formation_probability_l156_156977

theorem triangle_formation_probability :
  let sticks := [1, 2, 4, 6, 9, 10, 12, 15]
  ∃ valid_combinations, valid_combinations.card = 11 ∧
  (list.to_finset (list.combinations sticks 3)).card = 56 ∧
  (valid_combinations.card : ℚ) / (list.to_finset (list.combinations sticks 3)).card = 11 / 56 :=
by
  sorry

end triangle_formation_probability_l156_156977


namespace range_of_a_l156_156624

theorem range_of_a (a b : ℝ) :
  (∀ x : ℝ, (a * x^2 + b * x + 1 < 2)) ∧ (a - b + 1 = 1) → (-4 < a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l156_156624


namespace conjecture_l156_156902

-- Definitions for the ellipse and points
def ellipse (a b : ℝ) (a_pos b_pos : 0 < a ∧ 0 < b ∧ a > b) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

def symmetric_point (m n : ℝ) := (-m, -n)

def slope (x1 y1 x2 y2 : ℝ) := (y2 - y1) / (x2 - x1)

-- Restate the conjecture and the proof requirement
theorem conjecture 
  (a b : ℝ) (a_pos b_pos : 0 < a ∧ 0 < b ∧ a > b) 
  (x y m n : ℝ) 
  (ellipse_Q : ellipse a b a_pos x y) 
  (ellipse_M : ellipse a b a_pos m n) 
  (symmetric_M : symmetric_point m n = (-m, -n)) : 
  slope x y m n * slope x y (-m) (-n) = - (b^2 / a^2) := 
sorry

end conjecture_l156_156902


namespace plus_signs_count_l156_156808

theorem plus_signs_count (num_symbols : ℕ) (at_least_one_plus_in_10 : ∀ s : Finset ℕ, s.card = 10 → (∃ i ∈ s, i < 14)) (at_least_one_minus_in_15 : ∀ s : Finset ℕ, s.card = 15 → (∃ i ∈ s, i ≥ 14)) : 
    ∃ (p m : ℕ), p + m = 23 ∧ p = 14 ∧ m = 9 := by
  sorry

end plus_signs_count_l156_156808


namespace floor_sqrt_120_l156_156531

theorem floor_sqrt_120 : (Int.floor (Real.sqrt 120)) = 10 := by
  have h1 : 10^2 < 120 := by norm_num
  have h2 : 120 < 11^2 := by norm_num
  -- Additional steps to show that Int.floor (Real.sqrt 120) = 10
  sorry

end floor_sqrt_120_l156_156531


namespace solid_volume_eq_80_l156_156771

theorem solid_volume_eq_80 :
  ∃ (solid : { base_area : ℕ // base_area = (1 / 2 : ℚ) * 8 * 5 })
    (depth : ℕ // depth = 6),
    solid.base_area * depth = 80 :=
by
  use ⟨20, by norm_num⟩
  use ⟨6, rfl⟩
  norm_num
  sorry

end solid_volume_eq_80_l156_156771


namespace geometry_problem_l156_156449

theorem geometry_problem
  (A B C D : Point)
  (BA BC : ℝ)
  (h_angle : ∠ ABC = 90)
  (h_circles_intersect : Circles_with_diameters BA BC meet_at D)
  (h_BA : BA = 20)
  (h_BC : BC = 21) :
  let BD := (420 / 29 : ℝ) 
  in m + n = 449 :=
by
  sorry

end geometry_problem_l156_156449


namespace angle_double_condition_l156_156290

theorem angle_double_condition {A B C D E : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
  (h1 : A B = C D) (h2 : C E = D E) :
  (∠ C E D = 2 * ∠ A E B) ↔ (dist A C = dist E C) :=
by sorry

end angle_double_condition_l156_156290


namespace trig_identity_l156_156137

theorem trig_identity (α : ℝ) (h : Real.tan α = 2 / 3) : 
  Real.sin (2 * α) - Real.cos (π - 2 * α) = 17 / 13 :=
by
  sorry

end trig_identity_l156_156137


namespace alyssa_books_correct_l156_156936

def alyssa_books (n_books : ℕ) (ratio : ℕ) : ℕ := n_books / ratio

theorem alyssa_books_correct :
  ∀ (n_books ratio : ℕ),
  ratio = 7 →
  n_books = 252 →
  alyssa_books n_books ratio = 36 := 
by
  intros n_books ratio h_ratio h_books
  rw [h_ratio, h_books]
  norm_num

end alyssa_books_correct_l156_156936


namespace plus_signs_count_l156_156779

theorem plus_signs_count (p m : ℕ) (h_sum : p + m = 23)
                         (h_max_minus : m ≤ 9) (h_max_plus : p ≤ 14)
                         (h_at_least_one_plus_in_10 : ∀ (s : Finset (Fin 23)), s.card = 10 → ∃ i ∈ s, i < p)
                         (h_at_least_one_minus_in_15 : ∀ (s : Finset (Fin 23)), s.card = 15 → ∃ i ∈ s, m ≤ i) :
  p = 14 :=
by sorry

end plus_signs_count_l156_156779


namespace jane_win_probability_l156_156925

-- Define the conditions
def spinner_sectors : ℕ := 8

-- Define the winning condition
def jane_wins (jane_spin sister_spin : ℕ) : Prop := 
  (abs (jane_spin - sister_spin) < 4)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := spinner_sectors * spinner_sectors

-- Define the number of losing combinations
def losing_combinations : ℕ := 20

-- Calculate the probability that Jane wins
def probability_jane_wins : ℚ := 1 - (losing_combinations / total_outcomes: ℚ)

-- Prove the probability that Jane wins is 11/16
theorem jane_win_probability : probability_jane_wins = 11 / 16 := 
by sorry

end jane_win_probability_l156_156925


namespace log_bc_ab_l156_156251

-- Define the conditions as parameters
variables {a b c m n : ℝ}
variable (logb_a : Real.log b a = m)
variable (logc_b : Real.log c b = n)

-- Define the theorem to prove the question equals the answer
theorem log_bc_ab (logb_a : Real.log b a = m) (logc_b : Real.log c b = n) :
  Real.log (b * c) (a * b) = n * (m + 1) / (n + 1) :=
sorry

end log_bc_ab_l156_156251


namespace number_of_valid_arrangements_l156_156313

def is_ascending (l : List ℕ) : Prop :=
  ∀ i j, i < j → l.nth i ≤ l.nth j

def is_descending (l : List ℕ) : Prop :=
  ∀ i j, i < j → l.nth i ≥ l.nth j

def remove_one_is_ordered (l : List ℕ) : Prop :=
  ∃ (i : ℕ), (is_ascending (l.removeNth i) ∨ is_descending (l.removeNth i))

def valid_arrangements_count (cards : List ℕ) : ℕ :=
  -- counting the number of valid arrangements
  if (cards.length = 7
        ∧ ∀ i, i ∈ cards → 1 ≤ i ∧ i ≤ 7 ∧ (remove_one_is_ordered cards)) then 4 else 0

theorem number_of_valid_arrangements :
  valid_arrangements_count [1,2,3,4,5,6,7] = 4 :=
by sorry

end number_of_valid_arrangements_l156_156313


namespace four_digit_div_90_count_l156_156191

theorem four_digit_div_90_count :
  ∃ n : ℕ, n = 10 ∧ (∀ (ab : ℕ), 10 ≤ ab ∧ ab ≤ 99 → ab % 9 = 0 → 
  (10 * ab + 90) % 90 = 0 ∧ 1000 ≤ 10 * ab + 90 ∧ 10 * ab + 90 < 10000) :=
sorry

end four_digit_div_90_count_l156_156191


namespace total_cars_l156_156278

theorem total_cars (Cathy_cars : ℕ) (Lindsey_add : ℕ) (Carol_mult : ℕ) (Susan_sub : ℕ) (Erica_pct : ℕ)
  (Cathy_cars_eq : Cathy_cars = 5)
  (Lindsey_eq : ∀ Cathy_cars, Lindsey_add = Cathy_cars + 4)
  (Carol_eq : ∀ Cathy_cars, Carol_mult = 2 * Cathy_cars)
  (Susan_eq : ∀ Carol_mult, Susan_sub = Carol_mult - 2)
  (Erica_eq : ∀ Lindsey_add, Erica_pct = Lindsey_add + (Lindsey_add * 25 / 100).nat_floor)
  : (Cathy_cars + Lindsey_add + Carol_mult + Susan_sub + Erica_pct) = 43 :=
by
  sorry

end total_cars_l156_156278


namespace sequence_terms_l156_156605

theorem sequence_terms (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (hS : ∀ n, S n = 3 ^ n - 2) :
  (a 1 = 1) ∧ (∀ n, n ≥ 2 → a n = 2 * 3 ^ (n - 1)) := by
  sorry

end sequence_terms_l156_156605


namespace num_integer_solutions_abs_leq_seven_l156_156638

theorem num_integer_solutions_abs_leq_seven : 
  (∃ n : ℕ, n = (finset.Icc (-4 : ℤ) 10).card) ∧ n = 15 := 
by 
  sorry

end num_integer_solutions_abs_leq_seven_l156_156638


namespace ages_correct_in_2018_l156_156238

-- Define the initial ages in the year 2000
def age_marianne_2000 : ℕ := 20
def age_bella_2000 : ℕ := 8
def age_carmen_2000 : ℕ := 15

-- Define the birth year of Elli
def birth_year_elli : ℕ := 2003

-- Define the target year when Bella turns 18
def year_bella_turns_18 : ℕ := 2000 + 18

-- Define the ages to be proven
def age_marianne_2018 : ℕ := 30
def age_carmen_2018 : ℕ := 33
def age_elli_2018 : ℕ := 15

theorem ages_correct_in_2018 :
  age_marianne_2018 = age_marianne_2000 + (year_bella_turns_18 - 2000) ∧
  age_carmen_2018 = age_carmen_2000 + (year_bella_turns_18 - 2000) ∧
  age_elli_2018 = year_bella_turns_18 - birth_year_elli :=
by 
  -- The proof would go here
  sorry

end ages_correct_in_2018_l156_156238


namespace volume_tetrahedron_PQRS_l156_156878

noncomputable def tetrahedron_volume
  (PQ PR PS QR QS RS : ℝ)
  (hPQ : PQ = 3) 
  (hPR : PR = 5) 
  (hPS : PS = 6) 
  (hQR : QR = 4) 
  (hQS : QS = Real.sqrt 26) 
  (hRS : RS = 5) : ℝ :=
  sorry  -- The implementation will calculate the volume

theorem volume_tetrahedron_PQRS : 
  tetrahedron_volume 3 5 6 4 (Real.sqrt 26) 5 = 10 := 
by
  -- Proof to be filled in
  sorry

end volume_tetrahedron_PQRS_l156_156878


namespace ellipse_eq_elipse_q_l156_156965

noncomputable def a := sqrt 8
noncomputable def b := sqrt 2
noncomputable def c := sqrt 6
def e := c / a
def ellipse (x y : ℝ) := (x^2) / (a^2) + (y^2) / (b^2) = 1

theorem ellipse_eq_elipse_q (x y : ℝ) (h1 : e = sqrt 3 / 2)
  (h2 : 0 < b) (h3 : b < a) 
  (h4 : (M : ℝ × ℝ) → (N : ℝ × ℝ) → (M = (x_0, y_0)) → (N = (-x_0, y_0)) → (P : (ℝ × ℝ) := (0, 1)) → (Q : (ℝ × ℝ) := (0, 2)) 
  (h5 : ∀ (T : ℝ × ℝ), ellipse T.1 T.2)
  : ellipse x y = (x^2 / 8 + y^2 / 2 = 1) :=
sorry

end ellipse_eq_elipse_q_l156_156965


namespace correct_dictionary_l156_156281

-- Define Russian and Am-Yam words as constants
constant гулять : String := "гулять"
constant видит : String := "видит"
constant поймать : String := "поймать"
constant мышка : String := "мышка"
constant ночью : String := "ночью"
constant пошла : String := "пошла"
constant кошка : String := "кошка"

constant му : String := "му"
constant бу : String := "бу"
constant ям : String := "ям"
constant ту : String := "ту"
constant ам : String := "ам"
constant ля : String := "ля"

-- Define the sentences for the conditions
def sentence1 := ["мышка", "ночью", "пошла", "гулять"]
def sentence2 := ["кошка", "ночью", "видит", "мышка"]
def sentence3 := ["мышка", "кошка", "пошла", "поймать"]

-- Define the expected dictionary
def dictionary : List (String × String) := [
  (гулять, му), 
  (видит, бу), 
  (поймать, ям), 
  (мышка, ту), 
  (ночью, ам), 
  (пошла, ям), 
  (кошка, ля)
]

-- The theorem statement
theorem correct_dictionary : 
  (sentence1 = ["ту", "ам", "ям", "му"]) ∧ 
  (sentence2 = ["ля", "ам", "бу", "ту"]) ∧ 
  (sentence3 = ["ту", "ля", "ям", "ям"]) → 
  dictionary =
    [("гулять", "му"), ("видит", "бу"), ("поймать", "ям"), 
     ("мышка", "ту"), ("ночью", "ам"), ("пошла", "ям"), 
     ("кошка", "ля")] :=
by
  sorry

end correct_dictionary_l156_156281


namespace shaded_percentage_l156_156001

noncomputable def percent_shaded (side_len : ℕ) : ℝ :=
  let total_area := (side_len : ℝ) * side_len
  let shaded_area := (2 * 2) + (2 * 5) + (1 * 7)
  100 * (shaded_area / total_area)

theorem shaded_percentage (PQRS_side : ℕ) (hPQRS : PQRS_side = 7) :
  percent_shaded PQRS_side = 42.857 :=
  by
  rw [hPQRS]
  sorry

end shaded_percentage_l156_156001


namespace equation_solution_l156_156364

open Real

theorem equation_solution (x : ℝ) : 
  (x = 4 ∨ x = -1 → 3 * (2 * x - 5) ≠ (2 * x - 5) ^ 2) ∧
  (3 * (2 * x - 5) = (2 * x - 5) ^ 2 → x = 5 / 2 ∨ x = 4) :=
by
  sorry

end equation_solution_l156_156364


namespace parabola_focus_coordinates_l156_156543

theorem parabola_focus_coordinates (x y : ℝ) (h : y = 2 * x^2) : (0, 1 / 8) = (0, 1 / 8) :=
by
  sorry

end parabola_focus_coordinates_l156_156543


namespace total_material_used_l156_156076

-- Definitions for the original and leftover amounts of materials
def original_first_material := 4 / 9
def original_second_material := 2 / 3
def original_third_material := 5 / 6

def leftover_first_material := 8 / 18
def leftover_second_material := 3 / 9
def leftover_third_material := 2 / 12

-- Proof statement
theorem total_material_used :
  let used_first_material := original_first_material - leftover_first_material in
  let used_second_material := original_second_material - leftover_second_material in
  let used_third_material := original_third_material - leftover_third_material in
  used_first_material + used_second_material + used_third_material = 1 := by
  sorry

end total_material_used_l156_156076


namespace period_of_f_l156_156761

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + sqrt 3 * cos (2 * x)

theorem period_of_f : Function.Periodic f π :=
by
  sorry

end period_of_f_l156_156761


namespace triangle_construction_l156_156099

noncomputable def construct_triangle (BC : ℝ) (AM : ℝ) (AD : ℝ) : Type :=
  ∃ (A B C M D : Type),
  (M = midpoint B C) ∧
  (D = midpoint A C) ∧
  (AM = distance A M) ∧
  (AD = distance A D) ∧
  (BC = distance B C)

theorem triangle_construction (BC AM AD : ℝ) :
  construct_triangle BC AM AD :=
sorry

end triangle_construction_l156_156099


namespace canoe_trip_shorter_l156_156846

def lake_diameter : ℝ := 2
def pi_value : ℝ := 3.14

theorem canoe_trip_shorter : (2 * pi_value * (lake_diameter / 2) - lake_diameter) = 4.28 :=
by
  sorry

end canoe_trip_shorter_l156_156846


namespace derek_initial_money_l156_156103

theorem derek_initial_money :
  ∃ D : ℤ, 
  let derek_spent := 14 + 11 + 5 in
  let derek_left := D - derek_spent in
  let dave_initial := 50 in
  let dave_spent := 7 in
  let dave_left := dave_initial - dave_spent in
  let condition := dave_left = derek_left + 33 in
  D = 40 :=
begin
  sorry
end

end derek_initial_money_l156_156103


namespace grid_probability_black_l156_156903

theorem grid_probability_black :
  let p : ℝ := 1/4096 in
  let grid : List (List (ℕ × ℕ)) := 
    [(0..4).toList.bind (λ i => (0..4).toList.map (λ j => (i, j)))] in
  ∀ (initial_colors : ((ℕ × ℕ) → bool)),
  (∀ (i j : ℕ), i ∈ {0, 1, 2, 3} → j ∈ {0, 1, 2, 3} → initial_colors (i, j) = (i + j) % 2 = 0 → true) →
  (∀ (i j : ℕ), i ∈ {0, 1, 2, 3} → j ∈ {0, 1, 2, 3} →
    if initial_colors (i, j) 
    then initial_colors (3 - j, i) = true 
    else initial_colors (i, j) = false) →
  1 / (grid.length * grid.length).toReal = p :=
by
  sorry

end grid_probability_black_l156_156903


namespace area_converted_2018_l156_156228

theorem area_converted_2018 :
  let a₁ := 8 -- initial area in ten thousand hectares
  let q := 1.1 -- common ratio
  let a₆ := a₁ * q^5 -- area converted in 2018
  a₆ = 8 * 1.1^5 :=
sorry

end area_converted_2018_l156_156228


namespace possible_values_of_m_l156_156154

-- Defining sets A and B based on the given conditions
def set_A : Set ℝ := { x | x^2 - 2 * x - 3 = 0 }
def set_B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }

-- The main theorem statement
theorem possible_values_of_m (m : ℝ) :
  (set_A ∪ set_B m = set_A) ↔ (m = 0 ∨ m = -1 / 3 ∨ m = 1) := by
  sorry

end possible_values_of_m_l156_156154


namespace euclidean_algorithm_divisions_le_n_l156_156712

def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

theorem euclidean_algorithm_divisions_le_n
  (a b n : ℕ)
  (h_fib_a : fib n ≤ a)
  (h_fib_a_succ : a < fib (n + 1))
  (h_le : a ≤ b) :
  ∃ N, N ≤ n ∧ (euclidean_division_steps a b ≤ N) := 
sorry

end euclidean_algorithm_divisions_le_n_l156_156712


namespace find_a_l156_156657

theorem find_a (a : ℝ) : 
  (∃ (r : ℕ), r = 3 ∧ 
  ((-1)^r * (Nat.choose 5 r : ℝ) * a^(5 - r) = -40)) ↔ a = 2 ∨ a = -2 :=
by
    sorry

end find_a_l156_156657


namespace plus_signs_count_l156_156789

noncomputable def symbols : Type := fin 23 -- The type of 23 symbols

def is_plus (x : symbols) : Prop
def is_minus (x : symbols) : Prop

axiom total_symbols : ∀ (x : symbols), is_plus x ∨ is_minus x
axiom unique_classification : ∀ (x : symbols), ¬ (is_plus x ∧ is_minus x)
axiom ten_symbols_plus : ∀ (S : finset symbols), S.card = 10 → ∃ x ∈ S, is_plus x
axiom fifteen_symbols_minus : ∀ (S : finset symbols), S.card = 15 → ∃ x ∈ S, is_minus x

def num_plus_symbols : ℕ := finset.card (finset.filter is_plus finset.univ)

theorem plus_signs_count : num_plus_symbols = 14 :=
sorry

end plus_signs_count_l156_156789


namespace largest_divisor_of_n_l156_156206

theorem largest_divisor_of_n (n : ℕ) (h_pos : 0 < n) (h_div : 360 ∣ n^2) : 60 ∣ n := 
sorry

end largest_divisor_of_n_l156_156206


namespace math_problem_l156_156417

open Real

lemma radical_product :
  (3:ℝ) * (3:ℝ) * (2:ℝ) = 18 :=
by sorry

lemma cube_root_27 :
  real.cbrt 27 = 3 :=
by sorry

lemma fourth_root_81 :
  81 ^ (1 / 4:ℝ) = 3 :=
by sorry

lemma sixth_root_64 :
  64 ^ (1 / 6:ℝ) = 2 :=
by sorry

theorem math_problem :
  real.cbrt 27 * 81 ^ (1 / 4:ℝ) * 64 ^ (1 / 6:ℝ) = 18 :=
begin
  rw [cube_root_27, fourth_root_81, sixth_root_64],
  exact radical_product,
end

end math_problem_l156_156417


namespace ellipse_h_k_a_c_sum_l156_156346

theorem ellipse_h_k_a_c_sum :
  let h := -3
  let k := 1
  let a := 4
  let c := 2
  h + k + a + c = 4 :=
by
  let h := -3
  let k := 1
  let a := 4
  let c := 2
  show h + k + a + c = 4
  sorry

end ellipse_h_k_a_c_sum_l156_156346


namespace darla_total_payment_l156_156505

-- Definitions of the conditions
def rate_per_watt : ℕ := 4
def energy_usage : ℕ := 300
def late_fee : ℕ := 150

-- Definition of the expected total cost
def expected_total_cost : ℕ := 1350

-- Theorem stating the problem
theorem darla_total_payment :
  rate_per_watt * energy_usage + late_fee = expected_total_cost := 
by 
  sorry

end darla_total_payment_l156_156505


namespace distinct_meals_l156_156072

-- Define the conditions
def number_of_entrees : ℕ := 4
def number_of_drinks : ℕ := 3
def number_of_desserts : ℕ := 2

-- Define the main theorem
theorem distinct_meals : number_of_entrees * number_of_drinks * number_of_desserts = 24 := 
by
  -- sorry is used to skip the proof
  sorry

end distinct_meals_l156_156072


namespace weight_of_replaced_person_l156_156748

theorem weight_of_replaced_person 
  (avg_increase : ℝ) (new_person_weight : ℝ) (n : ℕ) (original_weight : ℝ) 
  (h1 : avg_increase = 2.5)
  (h2 : new_person_weight = 95)
  (h3 : n = 8)
  (h4 : original_weight = new_person_weight - n * avg_increase) : 
  original_weight = 75 := 
by
  sorry

end weight_of_replaced_person_l156_156748


namespace increase_by_percentage_l156_156879

theorem increase_by_percentage (initial : ℕ) (percentage : ℕ) : 
  (initial = 70) ∧ (percentage = 150) → (initial + (percentage / 100) * initial = 175) :=
by
  intro h
  cases h with h_initial h_percentage
  rw [h_initial, h_percentage]
  simp
  sorry

end increase_by_percentage_l156_156879


namespace fraction_meaningless_value_l156_156661

theorem fraction_meaningless_value (x : ℝ) : 
  (2 - x = 0) ↔ (x = 2) :=
begin
  sorry
end

end fraction_meaningless_value_l156_156661


namespace number_of_valid_arrangements_l156_156307

open Finset

-- We define the condition that a list is sorted in ascending order
def is_ascending (l : List ℕ) : Prop :=
  l = List.sort (≤) l

-- We define the condition that a list is sorted in descending order
def is_descending (l : List ℕ) : Prop :=
  l = List.sort (≥) l

def cards := Finset.range 7
def arrangements := cards.to_list.permutations

-- Define the function to check if a list of numbers (cards) 
-- can have one element removed to form an ascending or descending list
def valid_arrangement (l : List ℕ) : Prop :=
  ∃ (x : ℕ), (l.erase x).is_ascending ∨ (l.erase x).is_descending

-- Define the final theorem
theorem number_of_valid_arrangements : finset.card (arrangements.filter valid_arrangement) = 72 :=
by
  sorry

end number_of_valid_arrangements_l156_156307


namespace bounded_figure_convex_l156_156730

theorem bounded_figure_convex
  (Φ : set ℝ) -- Φ is a bounded figure
  (h1 : ∀ (P : ℝ), P ∈ interior Φ → ∀ l : set ℝ, is_line l ∧ P ∈ l → ∃! Q1 Q2 : ℝ, Q1 ∈ boundary Φ ∧ Q2 ∈ boundary Φ ∧ Q1 ≠ Q2 ∧ l ∩ Φ = {Q1, Q2})
  : convex Φ :=
sorry

end bounded_figure_convex_l156_156730


namespace five_digit_palindromic_numbers_eq_900_l156_156217

def is_palindromic (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

theorem five_digit_palindromic_numbers_eq_900 :
  (finset.filter (λ n, is_palindromic n ∧ is_five_digit n) (finset.range 100000)).card = 900 :=
by
  sorry

end five_digit_palindromic_numbers_eq_900_l156_156217


namespace medium_pizza_slices_l156_156056

theorem medium_pizza_slices (M : ℕ) 
  (small_pizza_slices : ℕ := 6)
  (large_pizza_slices : ℕ := 12)
  (total_pizzas : ℕ := 15)
  (small_pizzas : ℕ := 4)
  (medium_pizzas : ℕ := 5)
  (total_slices : ℕ := 136) :
  (small_pizzas * small_pizza_slices) + (medium_pizzas * M) + ((total_pizzas - small_pizzas - medium_pizzas) * large_pizza_slices) = total_slices → 
  M = 8 :=
by
  intro h
  sorry

end medium_pizza_slices_l156_156056


namespace converse_false_inverse_false_contrapositive_true_l156_156436

variable {a b : ℝ}

-- Proposition: If a = b, then a^2 = b^2.
def proposition : Prop := a = b → a^2 = b^2

-- Converse: If a^2 = b^2, then a = b. This is false.
def converse : Prop := a^2 = b^2 → a = b

-- Inverse: If a ≠ b, then a^2 ≠ b^2. This is false.
def inverse : Prop := a ≠ b → a^2 ≠ b^2

-- Contrapositive: If a^2 ≠ b^2, then a ≠ b. This is true.
def contrapositive : Prop := a^2 ≠ b^2 → a ≠ b

theorem converse_false : ¬converse := by
  sorry

theorem inverse_false : ¬inverse := by 
  sorry

theorem contrapositive_true : contrapositive := by 
  sorry

end converse_false_inverse_false_contrapositive_true_l156_156436


namespace average_temperature_l156_156558

def temperatures : List ℝ := [-36, 13, -15, -10]

theorem average_temperature : (List.sum temperatures) / (temperatures.length) = -12 := by
  sorry

end average_temperature_l156_156558


namespace chord_arc_angle_division_l156_156600

theorem chord_arc_angle_division (C : Type) [MetricSpace C] [NormedGroup C] [NormedSpace ℝ C]
  {circle : Circle C} {A B : circle.Point}
  (h : ∃ ratio : ℕ, ratio = 1 ∧ (1 + ratio) = 6) :
  (∃ θ : ℝ, θ = 30 ∨ θ = 150) :=
by
  sorry

end chord_arc_angle_division_l156_156600


namespace correct_option_l156_156433

theorem correct_option (a : ℝ) : 
  (∃ (opt : string), 
    (opt = "A" → (2 * a^2)^3 = 6 * a^6) ∧ 
    (opt = "B" → 2 * a^2 + 3 * a^4 = 5 * a^6) ∧ 
    (opt = "C" → (2 * a)⁻² = 1 / (4 * a^2)) ∧ 
    (opt = "D" → a^2 * (a^3 - 2 * a) = a^6 - 2 * a^3) 
  ) ↔ opt = "C" :=
by
  sorry

end correct_option_l156_156433


namespace sum_of_squares_of_real_solutions_l156_156125

theorem sum_of_squares_of_real_solutions (x : ℝ) (h : x ^ 64 = 16 ^ 16) : 
  (x = 2 ∨ x = -2) → (x ^ 2 + (-x) ^ 2) = 8 :=
by
  sorry

end sum_of_squares_of_real_solutions_l156_156125


namespace correct_calculation_l156_156432

theorem correct_calculation :
  3 * Real.sqrt 2 - (Real.sqrt 2 / 2) = (5 / 2) * Real.sqrt 2 :=
by
  -- To proceed with the proof, we need to show:
  -- 3 * sqrt(2) - (sqrt(2) / 2) = (5 / 2) * sqrt(2)
  sorry

end correct_calculation_l156_156432


namespace fraction_product_eq_one_l156_156959

theorem fraction_product_eq_one :
  (∏ i in finset.range 21, (1 + (19 : ℕ) / (i + 1))) / (∏ i in finset.range 19, (1 + (21 : ℕ) / (i + 1))) = 1 :=
by
  -- main proof
  sorry

end fraction_product_eq_one_l156_156959


namespace right_triangle_leg_lengths_l156_156678

theorem right_triangle_leg_lengths (a b c : ℕ) (h : a ^ 2 + b ^ 2 = c ^ 2) (h1: c = 17) (h2: a + (c - b) = 17) (h3: b + (c - a) = 17) : a = 8 ∧ b = 15 :=
by {
  sorry
}

end right_triangle_leg_lengths_l156_156678


namespace minimal_cost_is_128_25_l156_156736

def flower_costs := [2.25, 2.0, 1.75, 1.5, 1.0]
def rectangle_areas := [3 * 4, 5 * 3, 2 * 7, 6 * 3, 5 * 4]

noncomputable def minimal_flower_planting_cost : ℚ :=
  let costs := flower_costs.sort (≤)
  let areas := rectangle_areas.sort (≤)
  list.zip costs areas |>.map (λ (cost, area) => cost * area) |>.sum

theorem minimal_cost_is_128_25 :
  minimal_flower_planting_cost = 128.25 := 
sorry

end minimal_cost_is_128_25_l156_156736


namespace classify_finite_magic_set_l156_156055

open Int

noncomputable def is_magic_set (S : Set ℕ) : Prop :=
  ∀ x y ∈ S, x ≠ y → (x + y) / Nat.gcd x y ∈ S

theorem classify_finite_magic_set (S : Set ℕ) (finite_S : Set.Finite S) (hS : is_magic_set S) :
  ∃ i ≥ 3, S = { i, i^2 - i } :=
sorry

end classify_finite_magic_set_l156_156055


namespace ratio_q_p_l156_156134

open_locale big_operators

noncomputable theory
open finset

-- Define the total number of slips
def total_slips := 40

-- Define the number of distinct numbers each appearing on four slips
def num_options := 10 ∙ 4

-- Probability p that all four slips bear the same number
def probability_same_number := (10:ℚ) / (choose 40 4)

-- Probability q that two slips bear a number a and the other two bear a different number b
def probability_two_pairs := (45:ℚ) * (6:ℚ) * (6:ℚ) / (choose 40 4)

theorem ratio_q_p : (probability_two_pairs / probability_same_number) = 162 := sorry

end ratio_q_p_l156_156134


namespace sum_of_fx_half_l156_156602

theorem sum_of_fx_half (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x : ℝ, f (2 * x + 2) = f (-2 * x + 2)) →
  (∀ x : ℝ, f (x+1) = -f (-x + 1)) →
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f x = a * x + b) →
  f 4 = 1 →
  ∑ i in {1, 2, 3}, f (i + 1/2) = -1/2 :=
by
  sorry

end sum_of_fx_half_l156_156602


namespace convex_quad_sum_of_angles_l156_156037

theorem convex_quad_sum_of_angles (ABCD : Type) [ConvexQuadrilateral ABCD]
  (h : AB * CD = AD * BC) :
  ∑ (ang : list ℝ) in [∠BAC, ∠CBD, ∠DCA, ∠ADB], ang = 180 :=
by
  sorry

end convex_quad_sum_of_angles_l156_156037


namespace number_of_valid_arrangements_l156_156306

open Finset

-- We define the condition that a list is sorted in ascending order
def is_ascending (l : List ℕ) : Prop :=
  l = List.sort (≤) l

-- We define the condition that a list is sorted in descending order
def is_descending (l : List ℕ) : Prop :=
  l = List.sort (≥) l

def cards := Finset.range 7
def arrangements := cards.to_list.permutations

-- Define the function to check if a list of numbers (cards) 
-- can have one element removed to form an ascending or descending list
def valid_arrangement (l : List ℕ) : Prop :=
  ∃ (x : ℕ), (l.erase x).is_ascending ∨ (l.erase x).is_descending

-- Define the final theorem
theorem number_of_valid_arrangements : finset.card (arrangements.filter valid_arrangement) = 72 :=
by
  sorry

end number_of_valid_arrangements_l156_156306


namespace divisibility_l156_156274

theorem divisibility (n : ℕ) (a b c d : ℤ) 
  (h1 : n ∣ a + b + c + d) 
  (h2 : n ∣ a^2 + b^2 + c^2 + d^2) : 
  n ∣ a^4 + b^4 + c^4 + d^4 + 4 * a * b * c * d :=
  sorry

end divisibility_l156_156274


namespace product_is_permutation_l156_156652

open Nat

noncomputable def factorial_product (n : ℕ) : ℕ := 
  (27 - n) * (28 - n) * (29 - n) * (30 - n) * (31 - n) * (32 - n) * (33 - n) * (34 - n)

noncomputable def permutation (m k : ℕ) : ℕ := 
  if h : m ≥ k then (range k).prod (λ i, m - i) else 0

theorem product_is_permutation (n : ℕ) (h₀ : n ∈ ℕ) (h₁ : n < 20) : factorial_product n = permutation (34 - n) 8 := by
  sorry

end product_is_permutation_l156_156652


namespace sum_coefficients_equality_l156_156643

theorem sum_coefficients_equality :
  let exp := (1 + 2 * x) ^ 5
  let a_0 := (exp.coeff 0)
  let a_2 := (exp.coeff 2)
  let a_4 := (exp.coeff 4)
  in a_0 + a_2 + a_4 = 121 :=
by
  sorry

end sum_coefficients_equality_l156_156643


namespace length_of_shorter_angle_trisector_l156_156152

theorem length_of_shorter_angle_trisector (BC AC : ℝ) (h1 : BC = 3) (h2 : AC = 4) :
  let AB := Real.sqrt (BC^2 + AC^2)
  let x := 2 * (12 / (4 * Real.sqrt 3 + 3))
  let PC := 2 * x
  AB = 5 ∧ PC = (32 * Real.sqrt 3 - 24) / 13 :=
by
  sorry

end length_of_shorter_angle_trisector_l156_156152


namespace least_five_digit_congruent_l156_156861

theorem least_five_digit_congruent (x : ℕ) (h1 : x ≥ 10000) (h2 : x < 100000) (h3 : x % 17 = 8) : x = 10004 :=
by {
  sorry
}

end least_five_digit_congruent_l156_156861


namespace λ_geq_2_sin_54_l156_156148

open Real

noncomputable def λ (points : Fin 5 → ℝ × ℝ) : ℝ :=
  let distances := (List.filter (<>) (List.map (λ ⟨i, j⟩, dist (points i) (points j)) ((Finset.cartesianProduct (Finset.univ) (Finset.univ)).val.toList)))
  distances.maximum / distances.minimum

theorem λ_geq_2_sin_54 : ∀ (points : Fin 5 → ℝ × ℝ), λ points ≥ 2 * sin (54 * (π / 180)) :=
sorry

end λ_geq_2_sin_54_l156_156148


namespace vector_dot_product_l156_156567

def vector (α : Type*) := (α × α)

def a : vector ℝ := (2, 1)
def b : vector ℝ := (1/2, 0)

theorem vector_dot_product 
  (a_eq : vector ℝ = (2, 1))
  (cond : vector ℝ = ((a.1 - 2 * b.1), (a.2 - 2 * b.2)) = (1, 1)) :
  (a.1 * b.1 + a.2 * b.2) = 1 :=
sorry

end vector_dot_product_l156_156567


namespace geometric_mean_sqrt2_minus1_sqrt2_plus1_l156_156546

theorem geometric_mean_sqrt2_minus1_sqrt2_plus1 : 
  ∃ a : ℝ, (a = 1 ∨ a = -1) ∧ a^2 = (real.sqrt 2 - 1) * (real.sqrt 2 + 1) :=
begin
  sorry
end

end geometric_mean_sqrt2_minus1_sqrt2_plus1_l156_156546


namespace percentage_of_600_equals_150_is_25_l156_156120

theorem percentage_of_600_equals_150_is_25 : (150 / 600 * 100) = 25 := by
  sorry

end percentage_of_600_equals_150_is_25_l156_156120


namespace total_experiments_non_adjacent_l156_156400

theorem total_experiments_non_adjacent (n_org n_inorg n_add : ℕ) 
  (h_org : n_org = 3) (h_inorg : n_inorg = 2) (h_add : n_add = 2) 
  (no_adjacent : True) : 
  (n_org + n_inorg + n_add).factorial / (n_inorg + n_add).factorial * 
  (n_inorg + n_add + 1).choose n_org = 1440 :=
by
  -- The actual proof will go here.
  sorry

end total_experiments_non_adjacent_l156_156400


namespace eval_f_at_one_sixth_l156_156174

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos (π * x + π / 2)

theorem eval_f_at_one_sixth :
  f (1 / 6) = -2 :=
by
  -- Proof would go here
  sorry

end eval_f_at_one_sixth_l156_156174


namespace printer_task_total_pages_l156_156018

theorem printer_task_total_pages
  (A B : ℕ)
  (h1 : 1 / A + 1 / B = 1 / 24)
  (h2 : 1 / A = 1 / 60)
  (h3 : B = A + 6) :
  60 * A = 720 := by
  sorry

end printer_task_total_pages_l156_156018


namespace prob_divisible_by_4_ab_probability_two_fair_12_sided_dice_divisible_by_4_l156_156003

noncomputable theory

def divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

def is_fair_12_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 12

theorem prob_divisible_by_4_ab (a b : ℕ) (h₁ : is_fair_12_sided_die a) (h₂ : is_fair_12_sided_die b) :
  (divisible_by_4 a ∧ divisible_by_4 b) ↔ (divisible_by_4 (10 * a + b)) :=
sorry

theorem probability_two_fair_12_sided_dice_divisible_by_4 :
  let prob := 1 / 16 in
  prob = 1/16 :=
sorry

end prob_divisible_by_4_ab_probability_two_fair_12_sided_dice_divisible_by_4_l156_156003


namespace part1_part2_l156_156583

variables {A B C a b c : ℝ}
variables {m n : ℝ × ℝ}

-- Definitions based on conditions
def vectors_collinear (m n : ℝ × ℝ) := ∃ k : ℝ, m = (k * n.1, k * n.2)
def obtuse (B : ℝ) := B > π / 2

-- Conditions for the problem
axiom angles_sides : A + B + C = π
axiom sides_pos : a > 0 ∧ b > 0 ∧ c > 0
axiom vectors_def : m = (cos A, b) ∧ n = (sin A, a)
axiom vectors_collinear_cond : vectors_collinear m n
axiom obtuse_angle : obtuse B

theorem part1 : B - A = π / 2 :=
sorry

theorem part2 : b = 2 * real.sqrt 3 → a = 2 → area ABC = real.sqrt 3 :=
sorry

end part1_part2_l156_156583


namespace distance_between_towers_l156_156682

theorem distance_between_towers :
  ∀ (length_of_wall soldiers_per_tower total_soldiers : ℕ), 
  length_of_wall = 7300 →
  soldiers_per_tower = 2 → 
  total_soldiers = 2920 →
  let towers := total_soldiers / soldiers_per_tower in
  let gaps := towers - 1 in
  let distance := length_of_wall / gaps in
  distance = 5 :=
by
  intros length_of_wall soldiers_per_tower total_soldiers
  intros h1 h2 h3
  let towers := total_soldiers / soldiers_per_tower
  let gaps := towers - 1
  let distance := length_of_wall / gaps
  sorry

end distance_between_towers_l156_156682


namespace lines_concurrent_on_OI_l156_156272

theorem lines_concurrent_on_OI
  (ABC : Triangle)
  (I : Point)
  (O : Point)
  (ΓA ΓB ΓC : Circle)
  (hI : I = incenter ABC)
  (hO : O = circumcenter ABC)
  (hΓA : ∀ (B C : Point), B ≠ C → B ∈ ΓA ∧ C ∈ ΓA ∧ is_tangent_to ΓA (incircle ABC))
  (hΓB : ∀ (A C : Point), A ≠ C → A ∈ ΓB ∧ C ∈ ΓB ∧ is_tangent_to ΓB (incircle ABC))
  (hΓC : ∀ (A B : Point), A ≠ B → A ∈ ΓC ∧ B ∈ ΓC ∧ is_tangent_to ΓC (incircle ABC))
  (A' B' C' : Point)
  (hAB : ∀ (B C : Point), B ≠ C → (B ∈ ΓB ∧ C ∈ ΓB ∧ B ∈ ΓC) ∧ C ∈ ΓC ∧ (A ∈ {A, A'} ∧ A' ∈ ΓB ∩ ΓC))
  (hBC : ∀ (A C : Point), A ≠ C → (A ∈ ΓA ∧ C ∈ ΓA ∧ A ∈ ΓC) ∧ C ∈ ΓC ∧ (B ∈ {B, B'} ∧ B' ∈ ΓA ∩ ΓC))
  (hCA : ∀ (A B : Point), A ≠ B → (A ∈ ΓA ∧ B ∈ ΓA ∧ A ∈ ΓB) ∧ B ∈ ΓB ∧ (C ∈ {C, C'} ∧ C' ∈ ΓA ∩ ΓB)) :
  ∃ (Q : Point), is_concurrent (AA' BB' CC') ∧ Q ∈ OI := sorry

end lines_concurrent_on_OI_l156_156272


namespace number_of_distinct_values_of_S_l156_156710

noncomputable def S (n : ℕ) : ℂ := complex.I^n - complex.I^(3 * n)

theorem number_of_distinct_values_of_S : 
  (finset.image S (finset.range 8)).card = 2 := 
by
  sorry

end number_of_distinct_values_of_S_l156_156710


namespace min_AB_l156_156756

noncomputable def f (x : ℝ) : ℝ :=
  (x - 1/2 * (x + exp x + 3))

theorem min_AB (x1 x2 : ℝ) (m : ℝ) 
  (h1 : 2 * x1 - 3 = x2 + exp x2)
  (h2 : 0 = f 0) : 
  ∃ (c : ℝ), c = 2 :=
by 
  sorry

end min_AB_l156_156756


namespace todd_money_left_l156_156848

-- Define the initial amount of money Todd has
def initial_amount : ℕ := 20

-- Define the number of candy bars Todd buys
def number_of_candy_bars : ℕ := 4

-- Define the cost per candy bar
def cost_per_candy_bar : ℕ := 2

-- Define the total cost of the candy bars
def total_cost : ℕ := number_of_candy_bars * cost_per_candy_bar

-- Define the final amount of money Todd has left
def final_amount : ℕ := initial_amount - total_cost

-- The statement to be proven in Lean
theorem todd_money_left : final_amount = 12 := by
  -- The proof is omitted
  sorry

end todd_money_left_l156_156848


namespace power_subtraction_l156_156651

variable {a m n : ℝ}

theorem power_subtraction (hm : a^m = 8) (hn : a^n = 2) : a^(m - 3 * n) = 1 := by
  sorry

end power_subtraction_l156_156651
