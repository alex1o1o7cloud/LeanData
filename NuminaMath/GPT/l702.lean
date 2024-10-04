import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.Field.Power
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.InnerProductSpace.Projection
import Mathlib.Analysis.SpecialFunctions.Pi
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Fib
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Arithmetic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Sequence.Fibonacci
import Mathlib.Data.Set.Basic
import Mathlib.Data.ZMod.Basic
import Mathlib.Geometry.Angle
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Geometry.Reflection
import Mathlib.NumberTheory.Divisors
import Mathlib.NumberTheory.Prime.Basic
import Mathlib.NumberTheory.PythagoreanTriples
import Mathlib.Probability.Distribution
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.SetTheory.Cardinal.Basic
import Mathlib.SetTheory.Set.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.EuclideanSpace.Basic

namespace parabola_focus_distance_l702_702829

theorem parabola_focus_distance (C : Set (ℝ × ℝ))
  (hC : ∀ x y, (y^2 = x) → (x, y) ∈ C)
  (F : ℝ × ℝ)
  (hF : F = (1/4, 0))
  (A : ℝ × ℝ)
  (hA : A = (x0, y0) ∧ (y0^2 = x0 ∧ (x0, y0) ∈ C))
  (hAF : dist A F = (5/4) * x0) :
  x0 = 1 :=
sorry

end parabola_focus_distance_l702_702829


namespace problem_a_l702_702384

theorem problem_a (k l m : ℝ) : 
  (k + l + m) ^ 2 >= 3 * (k * l + l * m + m * k) :=
by sorry

end problem_a_l702_702384


namespace problem_condition_l702_702510

variable {f : ℝ → ℝ}

theorem problem_condition (h_diff : Differentiable ℝ f) (h_ineq : ∀ x : ℝ, f x < iteratedDeriv 2 f x) : 
  e^2019 * f (-2019) < f 0 ∧ f 2019 > e^2019 * f 0 :=
by
  sorry

end problem_condition_l702_702510


namespace road_length_in_km_l702_702304

/-- The actual length of the road in kilometers is 7.5, given the scale of 1:50000 
    and the map length of 15 cm. -/

theorem road_length_in_km (s : ℕ) (map_length_cm : ℕ) (actual_length_cm : ℕ) (actual_length_km : ℝ) 
  (h_scale : s = 50000) (h_map_length : map_length_cm = 15) (h_conversion : actual_length_km = actual_length_cm / 100000) :
  actual_length_km = 7.5 :=
  sorry

end road_length_in_km_l702_702304


namespace closest_point_on_plane_l702_702125

noncomputable def closest_point (A : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let t := 21 / 38 in
  (2 + 5 * t, -1 + 3 * t, 4 - 2 * t)

theorem closest_point_on_plane (A : ℝ × ℝ × ℝ) (P : ℝ × ℝ × ℝ) :
  A = (2, -1, 4) →
  P = closest_point A →
  P = \left(\frac{181}{38}, \frac{25}{38}, \frac{110}{38}\right) :=
by
  intro hA hP
  simp only [closest_point, hA, hP]
  sorry

end closest_point_on_plane_l702_702125


namespace sequence_irrational_count_l702_702199

theorem sequence_irrational_count (x : ℕ → ℝ) (h : x 1 > 0)
  (rec : ∀ n : ℕ, x (n + 1) = sqrt 5 * x n + 2 * sqrt (x n ^ 2 + 1)) :
  ∃ S : finset ℕ, S.card ≥ 672 ∧ ∀ n ∈ S, irrational (x n) :=
by
  sorry

end sequence_irrational_count_l702_702199


namespace magnitude_of_z_l702_702975

theorem magnitude_of_z (z : ℂ) 
  (h : z * (2 * complex.I) = complex.abs z ^ 2 + 1) : 
  complex.abs z = 1 := 
sorry

end magnitude_of_z_l702_702975


namespace triangles_equiv_area_l702_702490

-- Definitions based on given conditions

-- Define the relevant geometric entities and properties
variables (A B C D E F H Q R U V M N : Type)
variable (circumcircle_of_ABC : Circle)
variable (circumcircle_of_DEF : Circle)
variable [IsAltitude A B C D E F] -- Assuming a typeclass that marks AD, BE, CF as altitudes.
variable [IsOrthocenter H A B C] -- Assuming a typeclass that marks H as the orthocenter.
variable [LiesOn Q circumcircle_of_ABC]
variable [Perpendicular QR BC]
variable [Parallel AQ (line_through R)]
variable [LiesOn U circumcircle_of_DEF]
variable [LiesOn V circumcircle_of_DEF]
variable [Perpendicular AM RV]
variable [Perpendicular HN RV]

-- Lean theorem statement
theorem triangles_equiv_area :
  area_of_triangle AM V = area_of_triangle HN V := by
  sorry

end triangles_equiv_area_l702_702490


namespace flag_arrangement_modulo_1000_l702_702668

theorem flag_arrangement_modulo_1000 :
  let red_flags := 8
  let white_flags := 8
  let black_flags := 1
  let total_flags := red_flags + white_flags + black_flags
  let number_of_gaps := total_flags + 1
  let valid_arrangements := (Nat.choose number_of_gaps white_flags) * (number_of_gaps - 2)
  valid_arrangements % 1000 = 315 :=
by
  sorry

end flag_arrangement_modulo_1000_l702_702668


namespace arithmetic_seq_sum_20_terms_l702_702241

variable (a d : ℝ)
variable (S20 : ℝ)
variable (sum_first_three sum_last_three : ℝ)

def arithmetic_seq_first_three_sum (a d : ℝ) : ℝ :=
  a + (a + d) + (a + 2 * d)

def arithmetic_seq_last_three_sum (a d : ℝ) : ℝ :=
  (a + 17 * d) + (a + 18 * d) + (a + 19 * d)

def arithmetic_seq_sum (a d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a + (n - 1) * d)

theorem arithmetic_seq_sum_20_terms 
  (h1 : arithmetic_seq_first_three_sum a d = 15)
  (h2 : arithmetic_seq_last_three_sum a d = 12) : 
  arithmetic_seq_sum a d 20 = 90 :=
sorry

end arithmetic_seq_sum_20_terms_l702_702241


namespace problem_1_problem_2_l702_702845

noncomputable def triangle_inequality (a b c : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

theorem problem_1 (a b c : ℝ) 
  (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (angle_bisectors : triangle_inequality a b c) 
  (h_internal_bisectors : ∀ D E F : ℝ, DE = DF): 
  (a / (b + c) = b / (c + a) + c / (a + b)) :=
sorry

theorem problem_2 (a b c : ℝ) 
  (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (angle_bisectors : triangle_inequality a b c) 
  (h_internal_bisectors : ∀ D E F : ℝ, DE = DF): 
  (∠ABC > (real.pi/2)) :=
sorry

end problem_1_problem_2_l702_702845


namespace rectangular_field_length_l702_702017

   theorem rectangular_field_length (w l : ℝ) 
     (h1 : l = 2 * w)
     (h2 : 64 = 8 * 8)
     (h3 : 64 = (1/72) * (l * w)) :
     l = 96 :=
   sorry
   
end rectangular_field_length_l702_702017


namespace proposition_1_not_proposition_2_proposition_3_not_proposition_4_l702_702914

theorem proposition_1 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 - b^2 = 1) : a - b < 1 := sorry

theorem not_proposition_2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/b - 1/a = 1) : ¬ (a - b < 1) := sorry

theorem proposition_3 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : Math.exp a - Math.exp b = 1) : a - b < 1 := sorry

theorem not_proposition_4 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : Real.log a - Real.log b = 1) : ¬ (a - b < 1) := sorry

end proposition_1_not_proposition_2_proposition_3_not_proposition_4_l702_702914


namespace problem_statement_l702_702502

-- Define the sequence and conditions.
def x_seq : ℕ → ℝ
| 0     := a
| 1     := b
| (n+2) := x_seq (n+1) - x_seq n

-- Define the partial sum \(S_n\).
def S (n : ℕ) := (Finset.range n).sum x_seq

-- State the problem in Lean: proving \(x_{100} = -a\) and \(S_{100} = 2b - a\).
theorem problem_statement : x_seq 99 = -a ∧ S 100 = 2 * b - a :=
sorry

end problem_statement_l702_702502


namespace cube_inequality_l702_702139

theorem cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 :=
by
  sorry

end cube_inequality_l702_702139


namespace circle_intersection_line_eq_b_l702_702411

theorem circle_intersection_line_eq_b :
  let center1 : ℝ × ℝ := (-5, 3)
  let radius1 : ℝ := 10
  let center2 : ℝ × ℝ := (4, -6)
  let radius2 : ℝ := 11
  ∃ (b : ℝ), (∀ x y : ℝ,
    ((x + center1.fst)^2 + (y - center1.snd)^2 = radius1^2) ∧
    ((x - center2.fst)^2 + (y + center2.snd)^2 = radius2^2) →
    (x - y) = b) ∧ b = -17 / 9 :=
begin
  sorry
end

end circle_intersection_line_eq_b_l702_702411


namespace calculate_total_money_l702_702857

noncomputable def cost_per_gumdrop : ℕ := 4
noncomputable def number_of_gumdrops : ℕ := 20
noncomputable def total_money : ℕ := 80

theorem calculate_total_money : 
  cost_per_gumdrop * number_of_gumdrops = total_money := 
by
  sorry

end calculate_total_money_l702_702857


namespace minimum_value_l702_702603

theorem minimum_value (p q r s t u v w : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
    (ht : 0 < t) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) (h₁ : p * q * r * s = 16) (h₂ : t * u * v * w = 25) :
    (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 40 := 
sorry

end minimum_value_l702_702603


namespace largest_first_term_geometric_progression_l702_702427

noncomputable def geometric_progression_exists (d : ℝ) : Prop :=
  ∃ (a : ℝ), a = 5 ∧ (a + d + 3) / a = (a + 2 * d + 15) / (a + d + 3)

theorem largest_first_term_geometric_progression : ∀ (d : ℝ), 
  d^2 + 6 * d - 36 = 0 → 
  ∃ (a : ℝ), a = 5 ∧ geometric_progression_exists d ∧ a = 5 ∧ 
  ∀ (a' : ℝ), geometric_progression_exists d → a' ≤ a :=
by intros d h; sorry

end largest_first_term_geometric_progression_l702_702427


namespace cannot_determine_perp_a_b_l702_702823

-- Definitions and conditions
variables {a b : Line} {α β : Plane}
variables (h1 : parallel α a) (h2 : parallel b β) (h3 : perp a β)

theorem cannot_determine_perp_a_b :
  ¬ (determine_perpendicular a b h1 h2 h3) := sorry

end cannot_determine_perp_a_b_l702_702823


namespace equilateral_triangle_cd_value_l702_702991

theorem equilateral_triangle_cd_value (c d : ℝ) :
  ((0,0), (c,17), (d,43)).is_equilateral →
  (cd = -1689 / 24) :=
sorry

end equilateral_triangle_cd_value_l702_702991


namespace speed_of_boat_in_still_water_l702_702693

variable (b s : ℝ)

theorem speed_of_boat_in_still_water :
  (b + s = 16) ∧ (b - s = 6) → b = 11 :=
by
  intro h
  cases h with h1 h2
  sorry

end speed_of_boat_in_still_water_l702_702693


namespace age_of_15th_student_l702_702016

open Nat

theorem age_of_15th_student :
  ∀ (x : ℕ),
    (∀ (ages : list ℕ), ages.length = 15 →
       list.sum ages / 15 = 15 →
         (∀ (ages1 : list ℕ), ages1.length = 5 →
          list.sum ages1 / 5 = 14 →
            (∀ (ages2 : list ℕ), ages2.length = 9 →
             list.sum ages2 / 9 = 16 →
                (ages = ages1 ++ ages2 ++ [x] → x = 11)))) :=
by
  intros x ages len_ages avg_ages ages1 len_ages1 avg_ages1 ages2 len_ages2 avg_ages2 sum_ages_eq
  sorry

end age_of_15th_student_l702_702016


namespace min_value_expr_l702_702921

theorem min_value_expr (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (hxyz : x * y * z = 1) :
  2 * x^2 + 8 * x * y + 6 * y^2 + 16 * y * z + 3 * z^2 ≥ 24 :=
by
  sorry

end min_value_expr_l702_702921


namespace last_two_digits_7_pow_2018_l702_702678

theorem last_two_digits_7_pow_2018 : 
  (7 ^ 2018) % 100 = 49 := 
sorry

end last_two_digits_7_pow_2018_l702_702678


namespace total_weeds_correct_l702_702961

def tuesday : ℕ := 25
def wednesday : ℕ := 3 * tuesday
def thursday : ℕ := wednesday / 5
def friday : ℕ := thursday - 10
def total_weeds : ℕ := tuesday + wednesday + thursday + friday

theorem total_weeds_correct : total_weeds = 120 :=
by
  sorry

end total_weeds_correct_l702_702961


namespace cupcakes_left_l702_702953

def total_cupcakes : ℕ := 40
def students_class_1 : ℕ := 18
def students_class_2 : ℕ := 16
def additional_individuals : ℕ := 4

theorem cupcakes_left (total_cupcakes students_class_1 students_class_2 additional_individuals : ℕ) :
  total_cupcakes - students_class_1 - students_class_2 - additional_individuals = 2 :=
by
  have h1 : total_cupcakes - students_class_1 = 22 := by sorry
  have h2 : 22 - students_class_2 = 6 := by sorry
  have h3 : 6 - additional_individuals = 2 := by sorry
  exact h3

end cupcakes_left_l702_702953


namespace rounding_sum_eq_one_third_probability_l702_702642

noncomputable def rounding_sum_probability : ℝ :=
  (λ (total : ℝ) => 
    let round := (λ (x : ℝ) => if x < 0.5 then 0 else if x < 1.5 then 1 else if x < 2.5 then 2 else 3)
    let interval := (λ (start : ℝ) (end_ : ℝ) => end_ - start)
    let sum_conditions := [((0.5,1.5), 3), ((1.5,2.5), 2)]
    let total_length := 3

    let valid_intervals := sum_conditions.map (λ p => interval (p.fst.fst) (p.fst.snd))
    let total_valid_interval := List.sum valid_intervals
    total_valid_interval / total_length
  ) 3

theorem rounding_sum_eq_one_third_probability : rounding_sum_probability = 2 / 3 := by sorry

end rounding_sum_eq_one_third_probability_l702_702642


namespace find_n_l702_702725

theorem find_n 
  (molecular_weight : ℕ)
  (atomic_weight_Al : ℕ)
  (weight_OH : ℕ)
  (n : ℕ) 
  (h₀ : molecular_weight = 78)
  (h₁ : atomic_weight_Al = 27) 
  (h₂ : weight_OH = 17)
  (h₃ : molecular_weight = atomic_weight_Al + n * weight_OH) : 
  n = 3 := 
by 
  -- the proof is omitted
  sorry

end find_n_l702_702725


namespace statement_A_statement_B_statement_C_statement_D_l702_702686

def f (x : ℝ) : ℝ := x^(1/3)^(3 : ℝ)
def g (t : ℝ) : ℝ := t

theorem statement_A :
  ∀ (x : ℝ),
  f x = g x := 
by sorry

theorem statement_B :
  ¬ (∀ (f : ℝ → ℝ) (h : ∀ x, f (-x) = -f x),
  f 0 = 0) := 
by sorry

theorem statement_C :
  ¬ (∀ (f : ℝ → ℝ) (x1 x2 : ℝ),
  (f x1 ≠ f x2) → (x1 ≠ x2)) := 
by sorry

theorem statement_D :
  ¬ (∀ x y : ℝ, 
  (0 < x → 0 < y → f x < f y)) := 
by sorry

end statement_A_statement_B_statement_C_statement_D_l702_702686


namespace polynomial_factorization_l702_702009

-- Define the polynomial and its factorized form
def polynomial (x : ℝ) : ℝ := x^2 - 4*x + 4
def factorized_form (x : ℝ) : ℝ := (x - 2)^2

-- The theorem stating that the polynomial equals its factorized form
theorem polynomial_factorization (x : ℝ) : polynomial x = factorized_form x :=
by {
  sorry -- Proof skipped
}

end polynomial_factorization_l702_702009


namespace divisibility_condition_l702_702475

-- Define the problem conditions and the theorem
theorem divisibility_condition (n : ℕ) (h_pos : 0 < n) : 
  (n ∣ Nat.lcm (Finset.range (n - 1)).val) ↔ (¬ (Nat.prime n) ∧ ∀ k : ℕ, k ≥ 2 → (n ≠ k ^ 2 ∧ n ≠ k ^ 3 ∧ n ≠ k ^ 4 ∧ ...)) :=
sorry

end divisibility_condition_l702_702475


namespace canteen_distances_l702_702035

theorem canteen_distances 
  (B G C : ℝ)
  (hB : B = 600)
  (hBG : G = 800)
  (hBC_eq_2x : ∃ x, C = 2 * x ∧ B = G + x + x) :
  G = 800 / 3 :=
by
  sorry

end canteen_distances_l702_702035


namespace value_of_x_eq_eight_l702_702803

theorem value_of_x_eq_eight (x : ℝ) (h1 : 0 < x) (h2 : x * (floor x) = 48) : x = 8 :=
sorry

end value_of_x_eq_eight_l702_702803


namespace min_value_abs_expr_l702_702140

noncomputable def minExpr (a b : ℝ) : ℝ :=
  |a + b| + |(1 / (a + 1)) - b|

theorem min_value_abs_expr (a b : ℝ) (h₁ : a ≠ -1) : minExpr a b ≥ 1 ∧ (minExpr a b = 1 ↔ a = 0) :=
by
  sorry

end min_value_abs_expr_l702_702140


namespace probability_getting_wet_l702_702414

theorem probability_getting_wet :
  (∀ (P_rain P_no_rain : ℝ), P_rain = 0.5 ∧ P_no_rain = 0.5) →
  (∀ (P_tents_on_time P_tents_not_on_time : ℝ), P_tents_on_time = 0.5 ∧ P_tents_not_on_time = 0.5) →
  ∃ (P_getting_wet : ℝ), P_getting_wet = 0.25 :=
by
  intro h_rain h_tents
  use 0.25
  sorry

end probability_getting_wet_l702_702414


namespace salary_ratio_l702_702718

theorem salary_ratio (R S A : ℝ) (h_R : R = 25600) (h_A : A = 192000 / 12) (h_ratio : 0.10 * R = 0.08 * S) : S / A = 2 :=
by
  -- Definitions based on the conditions
  have h_R_val : R = 25600 := h_R,
  have h_A_val : A = 192000 / 12 := h_A,
  have h_S_val : S = 2560 / 0.08 := by 
    calc S = 2560 / 0.08 : by linarith [h_ratio],
  have h_SA_val : S / A = 2 := by
    calc S / A = (2560 / 0.08) / (192000 / 12) : by linarith [h_S_val, h_A_val]
             ... = 2 : by norm_num,
  exact h_SA_val

end salary_ratio_l702_702718


namespace arc_length_is_correct_l702_702519

-- Define the radius and central angle as given
def radius := 16
def central_angle := 2

-- Define the arc length calculation
def arc_length (r : ℕ) (α : ℕ) := α * r

-- The theorem stating the mathematically equivalent proof problem
theorem arc_length_is_correct : arc_length radius central_angle = 32 :=
by sorry

end arc_length_is_correct_l702_702519


namespace no_triangle_front_view_cylinder_l702_702869

theorem no_triangle_front_view_cylinder :
  ∀ (G : Type) (front_view_triangle : G → Prop),
  ¬ front_view_triangle Cylinder :=
by
  sorry

end no_triangle_front_view_cylinder_l702_702869


namespace range_of_a_l702_702505

variable {α : Type*} [LinearOrder α] [OrderedField α]

def p (x : α) : Prop := 1 / (x - 1) < 1

def q (x a : α) : Prop := x^2 + (a - 1) * x - a > 0

theorem range_of_a (a : Real) :
  (∀ x : Real, p x → q x a) ∧ (∃ x : Real, q x a ∧ ¬p x) →
  -2 < a ∧ a ≤ -1 :=
sorry

end range_of_a_l702_702505


namespace translation_correct_l702_702532

-- Define the points in the Cartesian coordinate system
structure Point where
  x : ℤ
  y : ℤ

-- Given points A and B
def A : Point := { x := -1, y := 0 }
def B : Point := { x := 1, y := 2 }

-- Translated point A' (A₁)
def A₁ : Point := { x := 2, y := -1 }

-- Define the translation applied to a point
def translate (p : Point) (v : Point) : Point :=
  { x := p.x + v.x, y := p.y + v.y }

-- Calculate the translation vector from A to A'
def translationVector : Point :=
  { x := A₁.x - A.x, y := A₁.y - A.y }

-- Define the expected point B' (B₁)
def B₁ : Point := { x := 4, y := 1 }

-- Theorem statement
theorem translation_correct :
  translate B translationVector = B₁ :=
by
  -- proof goes here
  sorry

end translation_correct_l702_702532


namespace janice_final_higher_than_christine_first_l702_702452

structure Throws :=
  (christine_first : ℕ)
  (janice_difference : ℕ)
  (christine_second_inc : ℕ)
  (janice_multiplier : ℕ)
  (christine_third_inc : ℕ)
  (highest_throw : ℕ)

variables (T : Throws)
  (janice_second_throw : ℕ := (T.christine_first - T.janice_difference) * T.janice_multiplier)
  (christine_second_throw : ℕ := T.christine_first + T.christine_second_inc)
  (christine_third_throw : ℕ := christine_second_throw + T.christine_third_inc)

noncomputable def janice_final_throw : ℕ := T.highest_throw

theorem janice_final_higher_than_christine_first :
  janice_final_throw T - T.christine_first = 17 :=
by
  sorry

-- Conditions setup
def myThrows : Throws :=
{ christine_first := 20,
  janice_difference := 4,
  christine_second_inc := 10,
  janice_multiplier := 2,
  christine_third_inc := 4,
  highest_throw := 37 }

example : janice_final_higher_than_christine_first myThrows := by
  sorry

end janice_final_higher_than_christine_first_l702_702452


namespace discarded_sevens_discarded_number_of_7s_last_remaining_card_l702_702706

def total_cards : ℕ := 288 * 7

def discard_steps (k : ℕ) : ℕ := total_cards - 6 * k

def cycles_needed_to_301_cards := sorry -- Calculation needed here

def remaining_card_after_steps (k : ℕ) : ℕ := sorry -- Calculation for remaining card

theorem discarded_sevens (total_cards remaining_cards: ℕ) (cycles : ℕ) : ℕ :=
  let num_groups := 288 in
  let num_sevens_per_group := 1 in
  let total_sevens := num_groups * num_sevens_per_group in
  total_sevens - (2016 - remaining_cards) / 6 -- utilize some integer division

-- Formalizing the actual part of lean import setup
theorem discarded_number_of_7s (total_remaining: ℕ) : 301 → discarded_sevens 2016 301 285 = 244 :=
by sorry

-- Setting up final card and group calculation
theorem last_remaining_card : Π (total_cards: ℕ) (k : ℕ), (remaining_card_after_steps k = sorry)  :=
by sorry

end discarded_sevens_discarded_number_of_7s_last_remaining_card_l702_702706


namespace arithmetic_geo_sequence_l702_702560

theorem arithmetic_geo_sequence (a_n : ℕ → ℤ) (d : ℤ) (n : ℕ) :
  (∀ n, a_n = 10 + (n - 1) * d) ∧ 
  (d = -1 ∨ d = 4) ∧ 
  (n ≤ 11 → (∑ i in finset.range n, |a_n i|) = -↑n * (n : ℤ) / 2 + 21 * n / 2) ∧ 
  (n ≥ 12 → (∑ i in finset.range n, |a_n i|) = ↑n * (n : ℤ) / 2 - 21 * n / 2 + 110) :=
by
  sorry

end arithmetic_geo_sequence_l702_702560


namespace true_statements_l702_702161

variables (m n : Line) (α β : Plane)

def line_parallel_to_plane (m : Line) (α : Plane) : Prop := sorry
def plane_parallel_to_plane (α β : Plane) : Prop := sorry
def line_perpendicular_to_plane (m : Line) (α : Plane) : Prop := sorry
def line_contained_in_plane (m : Line) (α : Plane) : Prop := sorry
def line_parallel_to_line (m n : Line) : Prop := sorry

theorem true_statements (h1 : line_parallel_to_plane m α)
                        (h2 : plane_parallel_to_plane α β)
                        (h3 : line_contained_in_plane m α)
                        (h4 : line_contained_in_plane n β)
                        (h5 : line_perpendicular_to_plane m α)
                        (h6 : line_perpendicular_to_plane n β)
                        (h7 : line_parallel_to_line m n) :
                        (line_parallel_to_plane m α → ∃ p : Plane, line_parallel_to_line m p ∧ p = α ∨ p ≠ α) ∧
                        (plane_parallel_to_plane α β ∧ line_contained_in_plane m α ∧ line_contained_in_plane n β → ¬ line_parallel_to_line m n) ∧
                        (line_parallel_to_line m n ∧ line_perpendicular_to_plane m α ∧ line_perpendicular_to_plane n β → plane_parallel_to_plane α β) ∧
                        (plane_parallel_to_plane α β ∧ line_contained_in_plane m α → line_parallel_to_plane m β) :=
by sorry

end true_statements_l702_702161


namespace num_factors_M_l702_702280

def M : ℕ := 57^4 + 4 * 57^3 + 6 * 57^2 + 4 * 57 + 1

theorem num_factors_M : (nat.factors M).length = 25 := 
sorry

end num_factors_M_l702_702280


namespace problem_inequality_l702_702814

noncomputable def u : ℕ → ℝ
| 1       := 1
| (n + 1) := 1 / ((Finset.range n).sum u) ^ 2

noncomputable def S (n : ℕ) : ℝ :=
(Finset.range n).sum u

theorem problem_inequality (n : ℕ) (hn : n ≥ 2) :
  (Real.cbrt (3 * n + 2) ≤ S n) ∧ (S n ≤ Real.cbrt (3 * n + 2) + 1 / Real.cbrt (3 * n + 2)) :=
sorry

end problem_inequality_l702_702814


namespace part_1_part_2_part_3_l702_702185

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2 - a) * Real.log x + (1 / x) + 2 * a * x

theorem part_1 {x : ℝ} (hx : 0 < x) : f x 0 = 2 * Real.log x + (1 / x) := 
by
  unfold f
  sorry

theorem part_2 {x : ℝ} (hx : 0 < x) (a : ℝ) (ha : a < 0) :
  (a < -2 → 
    (0 < x ∧ x < -1 / a ∨ x > 1 / 2 → (f x a)) < 0 ∧ (-1 / a < x ∧ x < 1 / 2 → (f x a)) > 0) ∧
  (a = -2 → ∀ x, (f x a) ≤ 0) ∧
  (-2 < a ∧ a < 0 → 
    (0 < x ∧ x < 1 / 2 ∨ x > -1 / a → (f x a) < 0) ∧ 
    (1 / 2 < x ∧ x < -1 / a → (f x a) > 0)) := 
by
   sorry

theorem part_3 {x₁ x₂ : ℝ} {a m : ℝ} (h₁ : a < -2) (h₂ : 1 ≤ x₁ ∧ x₁ ≤ 3) (h₃ : 1 ≤ x₂ ∧ x₂ ≤ 3) :
  |f x₁ a - f x₂ a| < (m + Real.log 3) * a - 2 * Real.log 3 → m ≤ -13/3 := 
by
  sorry

end part_1_part_2_part_3_l702_702185


namespace sqrt_inequality_iff_l702_702738

theorem sqrt_inequality_iff (y : ℝ) (hy : 0 < y) : (sqrt y < 3 * y) ↔ (y > 1 / 9) :=
sorry

end sqrt_inequality_iff_l702_702738


namespace course_selection_probability_l702_702573

-- Required conditions
variables (subjects : Finset String)
variables [DecidableEq String]
variables (P G C B : String)
variables (H : subjects = {P, G, C, B})
variables (combinations : Finset (Finset String))
variables (H_comb : combinations = subjects.powerset.filter (λ s, s.card = 2))

-- Main theorem
theorem course_selection_probability 
  (H_PG : P ∈ subjects ∧ G ∈ subjects ∧ C ∈ subjects ∧ B ∈ subjects)
  (H_equal_prob : ∀ s ∈ combinations, Prob s = 1 / combinations.card) :
  let event_A := combinations.filter (λ s, P ∈ s ∨ G ∈ s) in
  Prob event_A = 5 / 6 := 
by
  -- The statement constructs the problem as guided, the proof is not required
  sorry

end course_selection_probability_l702_702573


namespace au_tribe_max_words_correct_l702_702571

def au_tribe_max_words : ℕ :=
  let total_sequences := 2^14 - 2
  let valid_sequences := 2^14 - 2^7
  valid_sequences

theorem au_tribe_max_words_correct : 
  ∃ n : ℕ, au_tribe_max_words = n ∧ n = 16056 :=
by
  use 16056
  split
  · rfl
  · sorry

end au_tribe_max_words_correct_l702_702571


namespace range_of_m_l702_702146

theorem range_of_m (m : ℝ) (h : 1 < (8 - m) / (m - 5)) : 5 < m ∧ m < 13 / 2 :=
sorry

end range_of_m_l702_702146


namespace same_speed_for_all_spheres_l702_702443

-- Definitions of given parameters and conditions
def mass : Type := ℝ
def radius : Type := ℝ
def height : Type := ℝ
def g : ℝ := 9.8 -- acceleration due to gravity; assume standard value for simplicity

def sphere := {m : mass, R : radius}
def energy_conservation (s : sphere) (h : height) : ℝ :=
  sqrt((10 * g * h) / 7)

-- Theorem stating all spheres will have the same speed under given conditions
theorem same_speed_for_all_spheres
  (s1 s2 s3 s4 : sphere)
  (h : height) :
  energy_conservation s1 h = energy_conservation s2 h ∧
  energy_conservation s1 h = energy_conservation s3 h ∧
  energy_conservation s1 h = energy_conservation s4 h :=
by
  sorry

end same_speed_for_all_spheres_l702_702443


namespace sin_theta_value_l702_702221

theorem sin_theta_value (θ : ℝ) (h1 : 0 < θ) (h2 : θ < (π / 2))
  (h3 : 1 + sin θ = 2 * cos θ) : sin θ = 3 / 5 := 
sorry

end sin_theta_value_l702_702221


namespace find_z_l702_702345

theorem find_z
  (z : ℝ)
  (proj_eq : (let v := (2, -1, z : ℝ × ℝ × ℝ),
                   u := (1, 4, 2 : ℝ × ℝ × ℝ),
                   proj := ((v.1 * u.1 + v.2 * u.2 + v.3 * u.3) / (u.1 * u.1 + u.2 * u.2 + u.3 * u.3)) in
               proj * u = (3/21) * u))
  : z = (5 / 2) := 
by
  sorry

end find_z_l702_702345


namespace lines_intersect_intersection_point_on_ellipse_l702_702932

variables {k1 k2 : ℝ}
def l1 (x : ℝ) : ℝ := k1 * x + 1
def l2 (x : ℝ) : ℝ := k2 * x - 1

theorem lines_intersect (h : k1 * k2 + 2 = 0) : ∃ x y : ℝ, l1 x = y ∧ l2 x = y :=
by {
    sorry 
}

theorem intersection_point_on_ellipse (h : k1 * k2 + 2 = 0) : 
  ∃ x y : ℝ, l1 x = y ∧ l2 x = y ∧ 2 * x^2 + y^2 = 1 :=
by {
    sorry 
}

end lines_intersect_intersection_point_on_ellipse_l702_702932


namespace george_total_socks_l702_702494

-- Define the initial number of socks George had
def initial_socks : ℝ := 28.0

-- Define the number of socks he bought
def bought_socks : ℝ := 36.0

-- Define the number of socks his Dad gave him
def given_socks : ℝ := 4.0

-- Define the number of total socks
def total_socks : ℝ := initial_socks + bought_socks + given_socks

-- State the theorem we want to prove
theorem george_total_socks : total_socks = 68.0 :=
by
  sorry

end george_total_socks_l702_702494


namespace gumball_water_wednesday_l702_702206

theorem gumball_water_wednesday :
  ∀ (total_weekly_water monday_thursday_saturday_water tuesday_friday_sunday_water : ℕ),
  total_weekly_water = 60 →
  monday_thursday_saturday_water = 9 →
  tuesday_friday_sunday_water = 8 →
  total_weekly_water - (monday_thursday_saturday_water * 3 + tuesday_friday_sunday_water * 3) = 9 :=
by
  intros total_weekly_water monday_thursday_saturday_water tuesday_friday_sunday_water
  sorry

end gumball_water_wednesday_l702_702206


namespace scientific_notation_of_0_0000000005_l702_702617

theorem scientific_notation_of_0_0000000005 : 0.0000000005 = 5 * 10^(-10) :=
by {
  sorry
}

end scientific_notation_of_0_0000000005_l702_702617


namespace volume_of_solid_l702_702565

noncomputable def volumeOfSolidOfRevolution : ℝ :=
  ∫ y in 0..1, π * (Real.exp y)

theorem volume_of_solid :
  volumeOfSolidOfRevolution = π * (Real.exp 1 - 1) :=
by
  sorry

end volume_of_solid_l702_702565


namespace divisibility_condition_l702_702121

theorem divisibility_condition (x : ℂ) (k n p : ℕ) (h : n + 1 = k * (p + 1)) : 
  (x^n + x^(n-1) + ... + x + 1) ∣ (x^p + x^(p-1) + ... + x + 1) := 
sorry

end divisibility_condition_l702_702121


namespace no_real_solution_for_quadratic_eq_l702_702324

theorem no_real_solution_for_quadratic_eq (y : ℝ) :
  (8 * y^2 + 155 * y + 3) / (4 * y + 45) = 4 * y + 3 →  (¬ ∃ y : ℝ, (8 * y^2 + 37 * y + 33/2 = 0)) :=
by
  sorry

end no_real_solution_for_quadratic_eq_l702_702324


namespace sarah_marriage_age_l702_702468

theorem sarah_marriage_age : 
  let name_length := 5 in
  let current_age := 9 in
  let twice_age := 2 * current_age in
  name_length + twice_age = 23 :=
by
  let name_length := 5
  let current_age := 9
  let twice_age := 2 * current_age
  show name_length + twice_age = 23
  sorry

end sarah_marriage_age_l702_702468


namespace expand_polynomial_l702_702101

theorem expand_polynomial :
  (2 * t^2 - 3 * t + 2) * (-3 * t^2 + t - 5) = -6 * t^4 + 11 * t^3 - 19 * t^2 + 17 * t - 10 :=
by sorry

end expand_polynomial_l702_702101


namespace problem2_4_are_true_l702_702182

theorem problem2_4_are_true :
  (∀ x > 0, x^2 + 1 ≤ 3 * x →
  (∀ (x1 x2 : ℝ), (exp ((x1 + x2) / 2) ≤ (exp x1 + exp x2) / 2)) →
  ¬(∀ x : ℝ, exp (-x + 2) = exp (x - 2)) →
  (∀ (a : ℝ), (a = 1) → (cos (2 * a * x)) :=
    (∃ (a : ℝ), a = 1 → cos (2 * a * x) = cos(2 * x) ))
: ∃ (p : ℕ), p = 2 ∨ p = 4 :=
begin
  sorry
end

end problem2_4_are_true_l702_702182


namespace find_B_find_area_l702_702232

variables {A B C : ℝ} {a b c : ℝ}

-- Conditions provided in the problem
def condition1 := a = b * Real.cos C + Real.sqrt 3 * c * Real.sin B
def condition2 := b = 2
def condition3 := a = Real.sqrt 3 * c

-- First proof problem: Find B
theorem find_B (h1 : condition1) (h2 : 0 < B) (h3 : B < Real.pi) : B = Real.pi / 6 := by
  sorry

-- Second proof problem: Find area of triangle ABC
theorem find_area (h1 : condition1) (h2 : condition2) (h3 : condition3) : 
  let s := 1/2 * a * c * Real.sin B in
  s = Real.sqrt 3 := by
  sorry

end find_B_find_area_l702_702232


namespace cyclic_quadrilateral_sum_of_angles_l702_702555

theorem cyclic_quadrilateral_sum_of_angles
  (EFGH : Type)
  [cyclic_quadrilateral EFGH]
  (α β γ δ : ℝ) -- α = ∠EGH, β = central angle sum for EF and FH, γ = next vertex central angle, δ = next vertex central angle
  (h₁ : α = 50)
  (h₂ : β = 200)
  (h₃ : γ = 100) :
  α / 2 + (β / 2) = 100 := sorry

end cyclic_quadrilateral_sum_of_angles_l702_702555


namespace boxes_with_neither_l702_702762

def total_boxes : ℕ := 15
def boxes_with_crayons : ℕ := 9
def boxes_with_markers : ℕ := 6
def boxes_with_both : ℕ := 4

theorem boxes_with_neither : total_boxes - (boxes_with_crayons + boxes_with_markers - boxes_with_both) = 4 := by
  sorry

end boxes_with_neither_l702_702762


namespace loop_structure_and_body_l702_702887

theorem loop_structure_and_body (alg_cond : Prop) (repeated_step : Prop) :
  alg_cond → repeated_step → (structure eq to "loop structure") ∧ (step eq to "loop body") :=
by
  intro alg_cond
  intro repeated_step
  sorry

end loop_structure_and_body_l702_702887


namespace max_minus_min_example_l702_702357

theorem max_minus_min_example : 
  let s := {53, 98, 69, 84}
  in (Finset.max s) - (Finset.min s) = 45 := 
by
  sorry

end max_minus_min_example_l702_702357


namespace evaluate_expression_correct_l702_702078

noncomputable def evaluate_expression :=
  abs (-1) - ((-3.14 + Real.pi) ^ 0) + (2 ^ (-1 : ℤ)) + (Real.cos (Real.pi / 6)) ^ 2

theorem evaluate_expression_correct : evaluate_expression = 5 / 4 := by sorry

end evaluate_expression_correct_l702_702078


namespace multiplication_subtraction_difference_l702_702004

theorem multiplication_subtraction_difference (x n : ℕ) (h₁ : x = 5) (h₂ : 3 * x = (16 - x) + n) : n = 4 :=
by
  -- Proof will go here
  sorry

end multiplication_subtraction_difference_l702_702004


namespace wedge_volume_correct_l702_702730

noncomputable def volume_wedge (d: ℝ) (h: ℝ) (θ: ℝ) : ℝ :=
  let r := d / 2
  let V_cylinder := π * r^2 * h
  (θ / 360) * V_cylinder

theorem wedge_volume_correct :
  volume_wedge 16 16 60 = 128 * π :=
by
  sorry

end wedge_volume_correct_l702_702730


namespace initial_volume_is_40_l702_702410

theorem initial_volume_is_40
  (V : ℝ)
  (initial_concentration : V > 0 ∧ initial_concentration = 0.05)  
  (added_alcohol : 4.5) 
  (added_water : 5.5)
  (final_concentration : (0.05 * V + 4.5) = 0.13 * (V + 10)) :
  V = 40 :=
sorry

end initial_volume_is_40_l702_702410


namespace nine_chapters_math_art_l702_702708

theorem nine_chapters_math_art (x y : ℕ) 
  (h1 : x = 3 * (y - 2)) 
  (h2 : x - 9 = 2 * y) :
  (x = 3 * (y - 2)) ∧ (x - 9 = 2 * y) :=
by
  split
  case left => exact h1
  case right => exact h2

end nine_chapters_math_art_l702_702708


namespace problem_statement_l702_702491

-- Define the given expression
def expr (x : ℝ) : ℝ := (x - 10 * x^2 + 25 * x^3) / (8 - x^3)

-- Define the condition for nonnegativity
def nonnegative_interval (x : ℝ) : Prop :=
  expr x ≥ 0

-- The theorem statement for the equivalent proof problem
theorem problem_statement : ∀ (x : ℝ), 0 ≤ x ∧ x < 2 ↔ nonnegative_interval x :=
by
  sorry

end problem_statement_l702_702491


namespace simplify_expression_l702_702636

theorem simplify_expression 
  (a b x y : ℝ) :
  let expr := (3 * b * x * (a^3 * x^3 + 3 * a^2 * y^2 + 2 * b^2 * y^2) 
               + 2 * a * y * (2 * a^2 * x^2 + 3 * b^2 * x^2 + b^3 * y^3)) 
               / (3 * b * x + 2 * a * y) in
  expr = a^3 * x^3 + 3 * a^2 * x * y + 2 * b^2 * y^2 := 
by
  sorry

end simplify_expression_l702_702636


namespace slope_of_tangent_at_point_l702_702660

theorem slope_of_tangent_at_point (x y : ℝ) (h_curve : y = x^3 - 2 * x + 4) (h_point : (x, y) = (1, 3)) :
  deriv (λ x : ℝ, x^3 - 2 * x + 4) 1 = 1 :=
by {
  sorry
}

end slope_of_tangent_at_point_l702_702660


namespace number_of_students_l702_702997

-- Definitions based on conditions
def candy_bar_cost : ℝ := 2
def chips_cost : ℝ := 0.5
def total_cost_per_student : ℝ := candy_bar_cost + 2 * chips_cost
def total_amount : ℝ := 15

-- Statement to prove
theorem number_of_students : (total_amount / total_cost_per_student) = 5 :=
by
  sorry

end number_of_students_l702_702997


namespace inequality_proof_l702_702949

theorem inequality_proof
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a * b + b * c + c * a = 1) :
  (a + b) / Real.sqrt (a * b * (1 - a * b)) + 
  (b + c) / Real.sqrt (b * c * (1 - b * c)) + 
  (c + a) / Real.sqrt (c * a * (1 - c * a)) ≤ Real.sqrt 2 / (a * b * c) :=
sorry

end inequality_proof_l702_702949


namespace nellie_final_legos_l702_702303

-- Define the conditions
def original_legos : ℕ := 380
def lost_legos : ℕ := 57
def given_away_legos : ℕ := 24

-- The total legos Nellie has now
def remaining_legos (original lost given_away : ℕ) : ℕ := original - lost - given_away

-- Prove that given the conditions, Nellie has 299 legos left
theorem nellie_final_legos : remaining_legos original_legos lost_legos given_away_legos = 299 := by
  sorry

end nellie_final_legos_l702_702303


namespace cos_value_l702_702497

theorem cos_value (α : ℝ) (h : Real.sin (π / 5 - α) = 1 / 3) : 
  Real.cos (2 * α + 3 * π / 5) = -7 / 9 := by
  sorry

end cos_value_l702_702497


namespace problem_l702_702173

theorem problem (S : ℝ) (N : ℝ)
  (h1 : S = (1 / (2! * 17!)) + (1 / (3! * 16!)) + (1 / (4! * 15!)) + (1 / (5! * 14!)) + (1 / (6! * 13!)) + (1 / (7! * 12!)) + (1 / (8! * 11!)) + (1 / (9! * 10!)))
  (h2 : N = 18! * S) : 
  ⌊N / 100⌋ = 137 := 
sorry

end problem_l702_702173


namespace abs_diff_eq_3div2_l702_702831

theorem abs_diff_eq_3div2 
  (m n : ℝ)
  (h : ∀ x : ℝ, (x^2 - m*x + 2) * (x^2 - n*x + 2) = 0) 
  (h_geom_seq : ∃ p : ℝ, 
      ∀ (x1 x2 x3 x4 : ℝ), 
        {x1, x2, x3, x4}.to_finset.card = 4 ∧ 
        x1 = 1/2 ∧ 
        x2 = 1/2 * p ∧
        x3 = 1/2 * p^2 ∧ 
        x4 = 1/2 * p^3 ∧ 
        x1 * x2 = 2 ∧ 
        x3 * x4 = 2) : 
  |m - n| = 3/2 :=
by
  sorry

end abs_diff_eq_3div2_l702_702831


namespace smallest_repeating_block_digits_l702_702209

theorem smallest_repeating_block_digits (n d : ℕ) (h1: n = 9) (h2: d = 11) (h3: ((n : ℚ) / d) = 0.81):
  ∃ r, nat.length r = 2 ∧ ∀ k : ℕ, (n / d) = 0 * 10^k + r :=
sorry

end smallest_repeating_block_digits_l702_702209


namespace tangent_product_constant_l702_702395

open Real

noncomputable theory

variable {P F1 F2 : Point}

def is_on_ellipse (P : Point) (F1 F2 : Point) (a : ℝ) :=
  dist P F1 + dist P F2 = a

def is_not_major_axis_endpoint (P : Point) (F1 F2 : Point) := 
  -- definition to indicate that P is not an endpoint of the major axis
  sorry

def angle {A B C : Point} := 
  -- definition to compute angle at B formed by line segments BA and BC
  sorry 

def tangent_half_angle_identity (α : ℝ) :=
  tan (α / 2)

theorem tangent_product_constant
  (a b c: ℝ) 
  (h_ellipse : is_on_ellipse P F1 F2 (a + b)) 
  (h_not_major_axis_ep : is_not_major_axis_endpoint P F1 F2) :
  tangent_half_angle_identity (angle P F1 F2) * 
  tangent_half_angle_identity (angle P F2 F1) = (a + b - c) / (a + b + c) :=
sorry

end tangent_product_constant_l702_702395


namespace polynomials_exist_l702_702130

noncomputable theory

variable (n : ℕ)

-- Define the polynomials P and Q with integer coefficients
def P : (Fin n → ℝ) → ℝ := sorry
def Q : (Fin n → ℝ) → ℝ := sorry

theorem polynomials_exist :
  ∃ P Q : (Fin n → ℝ) → ℝ, (P ≠ 0) ∧ (Q ≠ 0) ∧
    ∀ (x : Fin n → ℝ), (sum univ (λ i, x i) * P x = Q (λ i, (x i)^2)) :=
sorry

end polynomials_exist_l702_702130


namespace dot_product_of_vectors_l702_702839

theorem dot_product_of_vectors
  (A B C : Type)
  [metric_space A]
  (dist : A → A → ℝ)
  (h1 : dist A B = 7)
  (h2 : dist B C = 5)
  (h3 : dist C A = 6) :
  let u := dist A B
  let v := dist B C
  let w := dist C A
  ∃ d : ℝ, d = -19 := by
sorry

end dot_product_of_vectors_l702_702839


namespace valid_parameterizations_l702_702340

-- Definition for the line equation
def line_eq (x y : ℝ) : Prop := y = 3 * x + 5

-- Definitions for each option
def option_A (t : ℝ) : Prop :=
  let p := (0 + 3 * t, 5 + t) in line_eq p.1 p.2

def option_B (t : ℝ) : Prop :=
  let p := (-5/3 + t, 3 * t) in line_eq p.1 p.2

def option_C (t : ℝ) : Prop :=
  let p := (1 + 9 * t, 8 + 3 * t) in line_eq p.1 p.2

def option_D (t : ℝ) : Prop :=
  let p := (2 + 2 * t, 11 + 3 * t) in line_eq p.1 p.2

def option_E (t : ℝ) : Prop :=
  let p := (-5 + t, 3 * t) in line_eq p.1 p.2

theorem valid_parameterizations :
  (option_B ∧ option_E) :=
by
  sorry

end valid_parameterizations_l702_702340


namespace power_function_properties_l702_702828

theorem power_function_properties (α : ℝ) (h : (3 : ℝ) ^ α = 27) :
  (α = 3) →
  (∀ x : ℝ, (x ^ α) = x ^ 3) ∧
  (∀ x : ℝ, x ^ α = -(((-x) ^ α))) ∧
  (∀ x y : ℝ, x < y → x ^ α < y ^ α) ∧
  (∀ y : ℝ, ∃ x : ℝ, x ^ α = y) :=
by
  sorry

end power_function_properties_l702_702828


namespace range_of_a_l702_702188

theorem range_of_a
  (a : ℝ)
  (f : ℝ → ℝ := λ x, (1/3) * x^3 + x^2 + a * x)
  (g : ℝ → ℝ := λ x, 1 / Real.exp x)
  (h : ∀ x1 ∈ Icc (1 / 2 : ℝ) 2, ∃ x2 ∈ Icc (1 / 2 : ℝ) 2, f'' x1 ≤ g x2) :
  a ≤ (Real.sqrt Real.exp 1) / Real.exp 1 - 8 :=
by
  sorry

end range_of_a_l702_702188


namespace exist_linearly_dependent_bases_l702_702906

open LinearAlgebra

variables (n : ℕ)
variables (A B C : Subspace ℝ (Fin n))
variables (A_basis : Basis (Fin n) ℝ A)
variables (B_basis : Basis (Fin n) ℝ B)
variables (C_basis : Basis (Fin n) ℝ C)

theorem exist_linearly_dependent_bases 
  (A_inter_B : A ⊓ B = ⊥) 
  (B_inter_C : B ⊓ C = ⊥)
  (C_inter_A : C ⊓ A = ⊥) :
  ∃ (a : Fin n → A) (b : Fin n → B) (c : Fin n → C),
    (∀ i : Fin n, a i + b i = c i) ∧
    (Basis (Fin n) ℝ (Submodule.span ℝ (Set.range a))) ∧
    (Basis (Fin n) ℝ (Submodule.span ℝ (Set.range b))) ∧
    (Basis (Fin n) ℝ (Submodule.span ℝ (Set.range c))) :=
sorry

end exist_linearly_dependent_bases_l702_702906


namespace seq_x_is_perfect_square_l702_702993

def seq_x : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := 14 * seq_x (n+1) - seq_x n - 4

theorem seq_x_is_perfect_square (n : ℕ) (h : n ≥ 1) : ∃ k : ℕ, seq_x n = k^2 := by
  sorry

end seq_x_is_perfect_square_l702_702993


namespace polynomial_symmetry_center_l702_702588

theorem polynomial_symmetry_center (P : ℝ → ℝ) 
  (hP_poly : ∃ n : ℕ, ∀ m : ℝ, P m = sum (range n) (λ i, (c i) * m^i))
  (hP_pairs : ∀ k : ℕ, ∃ m n : ℤ, P m + P n = 0) :
  ∃ c : ℝ, ∀ x : ℝ, P (c - x) = -P x := 
sorry

end polynomial_symmetry_center_l702_702588


namespace percent_diff_areas_l702_702296

def expected_diameter : ℝ := 30
def actual_diameter   : ℝ := 33

def expected_area (d : ℝ) : ℝ :=
  Real.pi * (d / 2) ^ 2

def actual_area (d : ℝ) : ℝ :=
  Real.pi * (d / 2) ^ 2

def percent_difference (expected actual : ℝ) : ℝ :=
  ((actual - expected) / expected) * 100

theorem percent_diff_areas : 
  percent_difference (expected_area expected_diameter) (actual_area actual_diameter) = 21 := 
by
  simp [expected_area, actual_area, percent_difference]
  sorry

end percent_diff_areas_l702_702296


namespace smallest_n_integer_price_l702_702433

theorem smallest_n_integer_price (p : ℚ) (h : ∃ x : ℕ, p = x ∧ 1.06 * p = n) : n = 53 :=
sorry

end smallest_n_integer_price_l702_702433


namespace gcd_of_three_numbers_l702_702122

theorem gcd_of_three_numbers : Nat.gcd 16434 (Nat.gcd 24651 43002) = 3 := by
  sorry

end gcd_of_three_numbers_l702_702122


namespace tray_height_l702_702430

-- Definitions based on conditions in the problem
def side_length : ℝ := 50
def corner_distance : ℝ := real.sqrt 5
def cut_angle : ℝ := 45

-- Height of the tray calculation based on the defined conditions and given the angle and distance
theorem tray_height : (corner_distance * real.sqrt 2 / 2) = real.sqrt 10 / 2 :=
by sorry

end tray_height_l702_702430


namespace ratio_of_distances_rational_l702_702634

theorem ratio_of_distances_rational (S : Finset ℝ) (k : ℝ) (hk : ∀ (x y a b : ℝ), x ∈ S → y ∈ S → a ∈ S → b ∈ S → 
  (abs(x - y) < k → ∃ (z w : ℝ), z ∈ S ∧ w ∈ S ∧ (abs(z - w) = abs(a - b))) → 
  (x ≠ y ∧ a ≠ b → ∃ q : ℚ, (abs(x - y) / abs(a - b)) = q)) :
  ∀ (a b : ℝ), a ∈ S → b ∈ S → ∃ q : ℚ, (abs(a - b) / k) = q :=
begin
  sorry
end

end ratio_of_distances_rational_l702_702634


namespace number_of_item_B_l702_702964

theorem number_of_item_B
    (x y z : ℕ)
    (total_items total_cost : ℕ)
    (hx_price : 1 ≤ x ∧ x ≤ 100)
    (hy_price : 1 ≤ y ∧ y ≤ 100)
    (hz_price : 1 ≤ z ∧ z ≤ 100)
    (h_total_items : total_items = 100)
    (h_total_cost : total_cost = 100)
    (h_price_equation : (x / 8) + 10 * y = z)
    (h_item_equation : x + y + (total_items - (x + y)) = total_items)
    : total_items - (x + y) = 21 :=
sorry

end number_of_item_B_l702_702964


namespace max_circle_radius_l702_702923

theorem max_circle_radius (Y : ℝ) : 
  let Z := 10 * real.sqrt 3 - 15 in
  ∃ r : ℝ, 3 * (r * r) ≤ Y  ∧ r = real.sqrt Z :=
sorry

end max_circle_radius_l702_702923


namespace regular_polygon_interior_angle_160_l702_702470

theorem regular_polygon_interior_angle_160 (n : ℕ) (h : 160 * n = 180 * (n - 2)) : n = 18 :=
by {
  sorry
}

end regular_polygon_interior_angle_160_l702_702470


namespace degree_of_polynomial_is_nine_l702_702777

def expr1 := λ (x : ℝ), x^7
def expr2 := λ (x : ℝ), x^2 + x^(-2)
def expr3 := λ (x : ℝ), 1 + 3*x^(-1) + 5*x^(-3)

def mult_expr := λ (x : ℝ), expr1 x * expr2 x * expr3 x

def polynomial_degree (f : ℝ → ℝ) : ℕ :=
  if hf : ∃ (n : ℕ), ∃ x, f x = x^n then classical.some hf else 0

theorem degree_of_polynomial_is_nine : polynomial_degree mult_expr = 9 := 
sorry

end degree_of_polynomial_is_nine_l702_702777


namespace coordinates_of_C_l702_702736

theorem coordinates_of_C : 
  ∃ C : ℝ × ℝ, 
    let A := (1, 3) in
    let B := (7, -1) in
    let AB := (B.1 - A.1, B.2 - A.2) in
    let BC := (AB.1 / 2, AB.2 / 2) in
    let C_expected := (B.1 + BC.1, B.2 + BC.2) in
    C_expected = (10, -3) :=
by
  sorry

end coordinates_of_C_l702_702736


namespace triangle_PQR_area_l702_702255

/-

Define the points P, Q, and R.
Define a function to calculate the area of a triangle given three points.
Then write a theorem to state that the area of triangle PQR is 12.

-/

structure Point where
  x : ℕ
  y : ℕ

def P : Point := ⟨2, 6⟩
def Q : Point := ⟨2, 2⟩
def R : Point := ⟨8, 5⟩

def area (A B C : Point) : ℚ :=
  abs ((A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2)

theorem triangle_PQR_area : area P Q R = 12 := by
  /- 
    The proof should involve calculating the area using the given points.
   -/
  sorry

end triangle_PQR_area_l702_702255


namespace arithmetic_sequence_sum_l702_702567

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Definition of the sum of the first n terms
def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

-- Problem statement in Lean 4
theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (S : ℕ → ℤ)
  (h1 : arithmetic_sequence a)
  (h2 : sum_first_n_terms a S)
  (h3 : S 9 = a 4 + a 5 + a 6 + 66) :
  a 2 + a 8 = 22 := by
  sorry

end arithmetic_sequence_sum_l702_702567


namespace interest_rate_per_annum_l702_702298

def principal : ℝ := 5396.103896103896
def total_amount : ℝ := 8310
def time_period : ℕ := 9

theorem interest_rate_per_annum :
  let interest_rate := (total_amount - principal) * 100 / (principal * time_period)
  interest_rate ≈ 6 := by
  sorry

end interest_rate_per_annum_l702_702298


namespace rational_root_even_coeff_l702_702309

theorem rational_root_even_coeff (a b c : ℤ) (h : a ≠ 0) (rational_root : ∃ (p q : ℤ), q ≠ 0 ∧ (p, q).coprime ∧ (a * p^2 + b * p * q + c * q^2 = 0)) :
  (¬(∀ x, x ∈ {a, b, c} → ¬(2 ∣ x))) → (∃ x, x ∈ {a, b, c} ∧ 2 ∣ x) :=
by
  intro h2
  sorry

end rational_root_even_coeff_l702_702309


namespace path_length_cube_dot_l702_702728

-- Define the edge length of the cube
def edge_length : ℝ := 2

-- Define the distance of the dot from the center of the top face
def dot_distance_from_center : ℝ := 0.5

-- Define the number of complete rolls
def complete_rolls : ℕ := 2

-- Calculate the constant c such that the path length of the dot is c * π
theorem path_length_cube_dot : ∃ c : ℝ, dot_distance_from_center = 2.236 :=
by
  sorry

end path_length_cube_dot_l702_702728


namespace number_of_balls_condition_l702_702406

theorem number_of_balls_condition (X : ℕ) (h1 : 25 - 20 = X - 25) : X = 30 :=
by
  sorry

end number_of_balls_condition_l702_702406


namespace distance_between_intersections_l702_702937

theorem distance_between_intersections (x y : ℝ) (h₁ : x = y^4)
  (h₂ : x + y^2 = 2) : 
  let p1 := (1, 1)
  in let p2 := (1, -1)
  in dist (p1 : ℝ × ℝ) p2 = 2 := by
  sorry

end distance_between_intersections_l702_702937


namespace absolute_sum_of_b_is_22176_l702_702598

theorem absolute_sum_of_b_is_22176 :
  let S := ∑ b in (Finset.filter (λ b : ℤ, ∃ r s : ℤ, r + s = -b ∧ r * s = 504 * b) (Finset.range 100000)), b
  in |S| = 22176 := 
by
  sorry

end absolute_sum_of_b_is_22176_l702_702598


namespace num_lines_through_A_with_equal_intercepts_l702_702212

def pointA := (1 : ℝ, 2 : ℝ)

def is_line_with_equal_intercepts (l : ℝ × ℝ → Prop) : Prop :=
∀ (x y : ℝ), l (x, y) ↔ (∃ (a : ℝ), (x = a ∨ y = a ∨ x = -a ∨ y = -a))

theorem num_lines_through_A_with_equal_intercepts :
  (∃ l₁ l₂ l₃ : ℝ × ℝ → Prop,
    l₁ pointA ∧ l₂ pointA ∧ l₃ pointA ∧
    l₁ ≠ l₂ ∧ l₁ ≠ l₃ ∧ l₂ ≠ l₃ ∧
    is_line_with_equal_intercepts l₁ ∧ is_line_with_equal_intercepts l₂ ∧ is_line_with_equal_intercepts l₃) :=
sorry

end num_lines_through_A_with_equal_intercepts_l702_702212


namespace smallest_possible_value_l702_702539

theorem smallest_possible_value (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (⌊(a + b + c) / d⌋ + ⌊(a + b + d) / c⌋ + ⌊(a + c + d) / b⌋ + ⌊(b + c + d) / a⌋) ≥ 8 :=
sorry

end smallest_possible_value_l702_702539


namespace add_and_subtract_base9_l702_702749

def base9_to_nat (digits: List ℕ) : ℕ :=
  digits.reverse.map_with_index (λ i d => d * 9 ^ i).sum

noncomputable def from_base9 (n: ℕ) : List ℕ :=
  let rec aux (n : ℕ) (acc : List ℕ) : List ℕ :=
    if h : 0 < n then
      aux (n / 9) ((n % 9) :: acc)
    else
      acc.reverse
  aux n []

theorem add_and_subtract_base9 :
  let a := base9_to_nat [2, 3, 6, 5]
  let b := base9_to_nat [1, 4, 8, 4]
  let c := base9_to_nat [7, 8, 2]
  let d := base9_to_nat [6, 7, 1]
  let result := base9_to_nat [4, 1, 7, 0]
  from_base9 ((a + b + c) - d) = [4, 1, 7, 0] :=
by
  sorry

end add_and_subtract_base9_l702_702749


namespace same_solution_eq_l702_702530

theorem same_solution_eq (a b : ℤ) (x y : ℤ) 
  (h₁ : 4 * x + 3 * y = 11)
  (h₂ : a * x + b * y = -2)
  (h₃ : 3 * x - 5 * y = 1)
  (h₄ : b * x - a * y = 6) :
  (a + b) ^ 2023 = 0 := by
  sorry

end same_solution_eq_l702_702530


namespace solution_set_of_inequality_l702_702444

def isOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def isDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

def isIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

theorem solution_set_of_inequality (f : ℝ → ℝ) :
  isOdd f →
  f (-4) = 0 →
  isDecreasing f 0 3 →
  isIncreasing f 3 ∞ →
  {x : ℝ | (x^2 - 4) * f x < 0} = {x : ℝ | x < -4} ∪ {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 2 < x ∧ x < 4} :=
by
  intros h1 h2 h3 h4
  sorry

end solution_set_of_inequality_l702_702444


namespace sum_of_d_mu_M_l702_702615

def dates : List ℕ := List.append (List.replicate 24 [1, 2..28])
  (List.append (List.replicate 22 [29, 30]) (List.replicate 14 31)) 

def n : ℕ := 730

def d : ℚ := 14.5
def mu : ℚ := 11476 / 730
def M : ℕ := 29

theorem sum_of_d_mu_M : (d + mu + M) = 59.22 := by
  have mu_approx : mu ≈ 15.72 := sorry
  show 14.5 + 15.72 + 29 = 59.22
  sorry

end sum_of_d_mu_M_l702_702615


namespace ratio_of_steps_l702_702069

-- Defining the conditions of the problem
def andrew_steps : ℕ := 150
def jeffrey_steps : ℕ := 200

-- Stating the theorem that we need to prove
theorem ratio_of_steps : andrew_steps / Nat.gcd andrew_steps jeffrey_steps = 3 ∧ jeffrey_steps / Nat.gcd andrew_steps jeffrey_steps = 4 :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_steps_l702_702069


namespace sum_first_10_terms_c_l702_702174

-- Define natural numbers as positive integers (excluding zero)
def Nat∗ := { n : ℕ // n > 0 }

-- Sequences a_n and b_n defined as arithmetic progressions with common difference of 1.
def a (a1 : Nat∗) (n : ℕ) : ℕ := a1.val + n - 1
def b (b1 : Nat∗) (n : ℕ) : ℕ := b1.val + n - 1

-- Define c_n as a_{b_n}
def c (a1 b1 : Nat∗) (n : ℕ) : ℕ := a a1 (b b1 n)

-- Define the first term of sequence c_n
def c_first_term (a1 b1 : Nat∗) : ℕ := c a1 b1 1

-- Define the sum of the first 'n' terms of an arithmetic sequence
def sum_arithmetic_seq (a1 : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2*a1 + (n - 1)*d) / 2

-- Problem Statement: Given the conditions, the sum of the first 10 terms of sequence c_n is 85.
theorem sum_first_10_terms_c (a1 b1 : Nat∗) (h₁ : a1.val + b1.val = 5) :
  sum_arithmetic_seq (c_first_term a1 b1) 1 10 = 85 :=
sorry

end sum_first_10_terms_c_l702_702174


namespace probability_of_F_l702_702413

-- Definitions for the probabilities of regions D, E, and the total probability
def P_D : ℚ := 3 / 8
def P_E : ℚ := 1 / 4
def total_probability : ℚ := 1

-- The hypothesis
lemma total_probability_eq_one : P_D + P_E + (1 - P_D - P_E) = total_probability :=
by
  simp [P_D, P_E, total_probability]

-- The goal is to prove this statement
theorem probability_of_F : 1 - P_D - P_E = 3 / 8 :=
by
  -- Using the total_probability_eq_one hypothesis
  have h := total_probability_eq_one
  -- This is a structured approach where verification using hypothesis and simplification can be done
  sorry

end probability_of_F_l702_702413


namespace initial_experts_unique_l702_702638

variable (x : ℕ)

def initial_experts_condition_1 : Prop :=
  ∀ (x : ℕ), (1 / (24 * x)) * 24 = 1

def initial_experts_condition_2 : Prop :=
  ∀ (x : ℕ), (1 / (18 * (x + 1))) * 18 = 1

theorem initial_experts_unique (x : ℕ) :
  initial_experts_condition_1 x → initial_experts_condition_2 x → x = 3 :=
by
  intros
  sorry

end initial_experts_unique_l702_702638


namespace sum_of_divisors_of_24_l702_702003

theorem sum_of_divisors_of_24 : ∑ d in {1, 2, 3, 4, 6, 8, 12, 24}, d = 60 := by
  sorry

end sum_of_divisors_of_24_l702_702003


namespace projection_of_a_onto_b_l702_702533

variable (a b : ℝ^3)
variable (ha : ‖a‖ = 1)
variable (hb : ‖b‖ = 1)
variable (h : ‖a - 2 • b‖ = Real.sqrt 7)

theorem projection_of_a_onto_b :
  ((⟪a, b⟫ / ‖b‖^2) • b) = (-1/2 : ℝ) • b :=
by
suffices h₀ : ⟪a, b⟫ = -1/2
from sorry
sorry

end projection_of_a_onto_b_l702_702533


namespace leading_coeff_of_100_degree_polynomial_l702_702311

open Polynomial

-- Define the Fibonacci sequence (using an existing library)
def fib : ℕ → ℤ
| 0       := 0
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

-- Define the problem statement
theorem leading_coeff_of_100_degree_polynomial (p : Polynomial ℤ)
  (h_deg : p.degree = 100)
  (h_vals : ∀ n, 1 ≤ n → n ≤ 102 → p.eval n = fib n) :
  p.leading_coeff = 1 / (100.factorial : ℤ) := 
sorry

end leading_coeff_of_100_degree_polynomial_l702_702311


namespace tractor_growth_rate_and_annual_production_l702_702042

theorem tractor_growth_rate_and_annual_production
  (oct_production : ℕ)
  (additional_production : ℕ)
  (annual_increase_percent : ℚ)
  (monthly_growth_rate : ℚ)
  (original_annual_production : ℕ) :
  oct_production = 1000 ∧
  additional_production = 2310 ∧
  annual_increase_percent = 0.21 ∧
  (monthly_growth_rate = 0.1 ∧
   original_annual_production = 1910) ↔
  (1000 * (1 + monthly_growth_rate) + 1000 * ((1 + monthly_growth_rate)^2) = additional_production ∧
   1.21 * original_annual_production = additional_production / 1.21) :=
by
  sorry

end tractor_growth_rate_and_annual_production_l702_702042


namespace multiply_fractions_l702_702766

theorem multiply_fractions :
  (2 : ℚ) / 3 * 5 / 7 * 11 / 13 = 110 / 273 :=
by
  sorry

end multiply_fractions_l702_702766


namespace eigenvalues_of_2x2_matrix_l702_702104

theorem eigenvalues_of_2x2_matrix :
  ∃ (k : ℝ), (k = 3 + 4 * Real.sqrt 6 ∨ k = 3 - 4 * Real.sqrt 6) ∧
            ∃ (v : ℝ × ℝ), v ≠ (0, 0) ∧
            ((3 : ℝ) * v.1 + 4 * v.2 = k * v.1 ∧ (6 : ℝ) * v.1 + 3 * v.2 = k * v.2) :=
begin
  sorry
end

end eigenvalues_of_2x2_matrix_l702_702104


namespace counties_under_50k_perc_l702_702051

def percentage (s: String) : ℝ := match s with
  | "20k_to_49k" => 45
  | "less_than_20k" => 30
  | _ => 0

theorem counties_under_50k_perc : percentage "20k_to_49k" + percentage "less_than_20k" = 75 := by
  sorry

end counties_under_50k_perc_l702_702051


namespace winning_probability_range_l702_702674

theorem winning_probability_range (a1 : ℝ) (hprob : ∑ p in ({(4*a1 - 36, a1+6, a1/4 + 18, a1+12) : set ℝ}), 1/4 * (ite (p > a1) 1 0)) = 3/4) :
  a1 ≤ 12 ∨ 24 ≤ a1 :=
  sorry

end winning_probability_range_l702_702674


namespace gobblenian_words_count_l702_702619

noncomputable def gobblenian_possible_words (alphabet_size : ℕ) (max_word_length : ℕ) : ℕ :=
  ∑ i in Finset.range max_word_length.succ, alphabet_size ^ i

theorem gobblenian_words_count :
  gobblenian_possible_words 4 4 = 340 :=
by
  -- The proof would be added here, but it's omitted for this task.
  sorry

end gobblenian_words_count_l702_702619


namespace little_john_initial_money_l702_702935

theorem little_john_initial_money :
  let spent := 1.05
  let given_to_each_friend := 2 * 1.00
  let total_spent := spent + given_to_each_friend
  let left := 2.05
  total_spent + left = 5.10 :=
by
  let spent := 1.05
  let given_to_each_friend := 2 * 1.00
  let total_spent := spent + given_to_each_friend
  let left := 2.05
  show total_spent + left = 5.10
  sorry

end little_john_initial_money_l702_702935


namespace magnitude_of_z_l702_702330

noncomputable def i : ℂ := complex.i

noncomputable def z : ℂ := (1 - 2 * i) * (3 + i)

theorem magnitude_of_z : complex.abs z = 5 * real.sqrt 2 :=
by {
  sorry
}

end magnitude_of_z_l702_702330


namespace part1_part2_l702_702184

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x + 3

theorem part1 (a : ℝ) : (∀ x : ℝ, f (1 - x) a = f (1 + x) a) → a = 2 :=
sorry

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := Real.log 2 x + m

theorem part2 (m : ℝ) : (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 4 → x₂ ∈ Set.Icc 1 4 → 
  let f_fixed := (λ (x : ℝ), x^2 - 2 * x + 3)
  f_fixed x₁ > g x₂ m) → m < 2 :=
sorry

end part1_part2_l702_702184


namespace sphere_surface_area_of_inscribed_cuboid_l702_702057

theorem sphere_surface_area_of_inscribed_cuboid
  (l w h : ℝ)
  (hl : l = 1)
  (hw : w = sqrt 6)
  (hh : h = 3)
  (d : ℝ)
  (hd : d = sqrt (l^2 + w^2 + h^2)) :
  4 * π * (d / 2)^2 = 16 * π :=
by
  sorry

end sphere_surface_area_of_inscribed_cuboid_l702_702057


namespace sales_difference_greatest_in_june_l702_702977

def percentage_difference (D B : ℕ) : ℚ :=
  if B = 0 then 0 else (↑(max D B - min D B) / ↑(min D B)) * 100

def january : ℕ × ℕ := (8, 5)
def february : ℕ × ℕ := (10, 5)
def march : ℕ × ℕ := (8, 8)
def april : ℕ × ℕ := (4, 8)
def may : ℕ × ℕ := (5, 10)
def june : ℕ × ℕ := (3, 9)

noncomputable
def greatest_percentage_difference_month : String :=
  let jan_diff := percentage_difference january.1 january.2
  let feb_diff := percentage_difference february.1 february.2
  let mar_diff := percentage_difference march.1 march.2
  let apr_diff := percentage_difference april.1 april.2
  let may_diff := percentage_difference may.1 may.2
  let jun_diff := percentage_difference june.1 june.2
  if max jan_diff (max feb_diff (max mar_diff (max apr_diff (max may_diff jun_diff)))) == jun_diff
  then "June" else "Not June"
  
theorem sales_difference_greatest_in_june : greatest_percentage_difference_month = "June" :=
  by sorry

end sales_difference_greatest_in_june_l702_702977


namespace simplify_expression_l702_702321

theorem simplify_expression :
  (3 * Real.sqrt 10) / (Real.sqrt 5 + 2) = 15 * Real.sqrt 2 - 6 * Real.sqrt 10 := 
by
  sorry

end simplify_expression_l702_702321


namespace cube_volume_after_cylinder_removal_l702_702039

noncomputable def side_length : ℝ := 6
noncomputable def radius : ℝ := 3
noncomputable def cube_volume : ℝ := side_length^3
noncomputable def cylinder_volume : ℝ := π * radius^2 * side_length
noncomputable def remaining_volume : ℝ := cube_volume - cylinder_volume

theorem cube_volume_after_cylinder_removal :
  remaining_volume = 216 - 54 * π :=
by 
  -- Placeholder for the actual proof
  sorry

end cube_volume_after_cylinder_removal_l702_702039


namespace coordinates_with_respect_to_origin_l702_702248

def point_coordinates (x y : ℤ) : ℤ × ℤ :=
  (x, y)

def origin : ℤ × ℤ :=
  (0, 0)

theorem coordinates_with_respect_to_origin :
  point_coordinates 2 (-3) = (2, -3) := by
  -- placeholder proof
  sorry

end coordinates_with_respect_to_origin_l702_702248


namespace frisbee_league_committees_l702_702881

theorem frisbee_league_committees (teams_num members_num host_committee non_host_committee : ℕ) :
  teams_num = 5 → members_num = 7 → host_committee = 4 → non_host_committee = 3 →
  ∑ (host in finset.range teams_num), 
    choose members_num host_committee * (choose members_num non_host_committee) ^ (teams_num - 1) = 262609375 :=
by
  intros
  have main_calculation : (choose members_num host_committee) * (choose members_num non_host_committee) ^ (teams_num - 1) = 35 * 35^4 := by
    sorry
  exact mul_comm.main_calculation teams_num = 262609375

end frisbee_league_committees_l702_702881


namespace balance_difference_correct_l702_702070

def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

def angela_balance : ℝ :=
  compound_interest 5000 0.05 4 15

def bob_balance : ℝ :=
  simple_interest 7000 0.06 15

def balance_difference : ℝ :=
  bob_balance - angela_balance

theorem balance_difference_correct :
  abs balance_difference = 2732 :=
by
  sorry

end balance_difference_correct_l702_702070


namespace sum_first_five_terms_eq_ninety_three_l702_702508

variable (a : ℕ → ℕ)

-- Definitions
def geometric_sequence (a : ℕ → ℕ) : Prop :=
∀ n m : ℕ, a (n + m) = a n * a m

variables (a1 : ℕ) (a2 : ℕ) (a4 : ℕ)
variables (S : ℕ → ℕ)

-- Conditions
axiom a1_value : a1 = 3
axiom a2a4_value : a2 * a4 = 144

-- Question: Prove S_5 = 93
theorem sum_first_five_terms_eq_ninety_three
    (h1 : geometric_sequence a)
    (h2 : a 1 = a1)
    (h3 : a 2 = a2)
    (h4 : a 4 = a4)
    (Sn_def : S 5 = (a1 * (1 - (2:ℕ)^5)) / (1 - 2)) :
  S 5 = 93 :=
sorry

end sum_first_five_terms_eq_ninety_three_l702_702508


namespace alphametic_puzzle_solution_l702_702971

theorem alphametic_puzzle_solution :
  ∃ (K O L A V D : ℕ),
    K ≠ 0 ∧
    K ≤ 4 ∧
    O = 9 ∧
    A = 0 ∧
    K ≠ O ∧ K ≠ L ∧ K ≠ A ∧ K ≠ V ∧ K ≠ D ∧
    O ≠ L ∧ O ≠ A ∧ O ≠ V ∧ O ≠ D ∧
    L ≠ A ∧ L ≠ V ∧ L ≠ D ∧
    A ≠ V ∧ A ≠ D ∧
    V ≠ D ∧
    1000 * K + 100 * O + 10 * K + A +
    1000 * K + 100 * O + 10 * L + A =
    1000 * V + 100 * O + 10 * D + A :=
by
  use 3, 9, 8, 0, 7, 1
  repeat { split }
  · exact Nat.succ_pos' _
  · exact Nat.le_refl _
  · rfl
  · rfl
  all_goals { simp }
  · ring
sorry

end alphametic_puzzle_solution_l702_702971


namespace least_alpha_condition_l702_702592

variables {a b α : ℝ}

theorem least_alpha_condition (a_gt_1 : a > 1) (b_gt_0 : b > 0) : 
  ∀ x, (x ≥ α) → (a + b) ^ x ≥ a ^ x + b ↔ α = 1 :=
by
  sorry

end least_alpha_condition_l702_702592


namespace parallel_lines_m_eq_one_l702_702548

theorem parallel_lines_m_eq_one (m : ℝ) :
  (∀ x y : ℝ, 2 * x + m * y + 8 = 0 ∧ (m + 1) * x + y + (m - 2) = 0 → m = 1) :=
by
  intro x y h
  let L1_slope := -2 / m
  let L2_slope := -(m + 1)
  have h_slope : L1_slope = L2_slope := sorry
  have m_positive : m = 1 := sorry
  exact m_positive

end parallel_lines_m_eq_one_l702_702548


namespace quadratic_coefficients_l702_702346

theorem quadratic_coefficients (x : ℝ) :
  ∃ a b c : ℝ, (a * x^2 + b * x + c = 0) ∧ a = 2 ∧ b = -6 ∧ c = -9 :=
by
  use 2, -6, -9
  split
  . show 2 * x^2 - 6 * x - 9 = 0
    sorry
  . constructor
    . refl
    . constructor
      . refl
      . refl

end quadratic_coefficients_l702_702346


namespace douglas_won_38_percent_in_y_l702_702015

def douglas_votes_in_county_y (T : ℝ) (V : ℝ) : ℝ :=
  let total_votes := 0.54 * T
  let votes_x := 0.62 * (2 / 3 * T)
  let P := (total_votes - votes_x) / (1 / 3 * T)
  P

theorem douglas_won_38_percent_in_y (T : ℝ) (V : ℝ) (T_pos : 0 < T) (V_pos : 0 < V) : 
  douglas_votes_in_county_y T V = 0.3801 :=
by
  sorry

end douglas_won_38_percent_in_y_l702_702015


namespace mark_less_than_kate_and_laura_l702_702941

theorem mark_less_than_kate_and_laura (K : ℝ) (h : K + 2 * K + 3 * K + 4.5 * K = 360) :
  let Pat := 2 * K
  let Mark := 3 * K
  let Laura := 4.5 * K
  let Combined := K + Laura
  Mark - Combined = -85.72 :=
sorry

end mark_less_than_kate_and_laura_l702_702941


namespace percentage_increase_indeterminate_l702_702613

/-- Define the variables as needed for the problem -/
variables (T : ℝ) (rate_A : ℝ) (rate_Q : ℝ)

/-- Assumption: Machine A produces 8.00000000000001 sprockets per hour -/
def rate_of_Machine_A : ℝ := 8.00000000000001

/-- Definition: The rate of Machine Q based on time T to produce 880 sprockets -/
def rate_of_Machine_Q (T : ℝ) : ℝ := 880 / T

/-- The percentage increase formula -/
def percentage_increase (rate_A rate_Q : ℝ) : ℝ :=
  ((rate_Q - rate_A) / rate_A) * 100

/-- The main theorem to state that without additional information, 
    the percentage increase cannot be determined -/
theorem percentage_increase_indeterminate (T : ℝ) (rate_A : ℝ) :
  rate_A = rate_of_Machine_A → ¬ (∃ rate_Q : ℝ, rate_Q = rate_of_Machine_Q T) :=
by {
  intros h1 h2,
  sorry
}

end percentage_increase_indeterminate_l702_702613


namespace nellie_final_legos_l702_702302

-- Define the conditions
def original_legos : ℕ := 380
def lost_legos : ℕ := 57
def given_away_legos : ℕ := 24

-- The total legos Nellie has now
def remaining_legos (original lost given_away : ℕ) : ℕ := original - lost - given_away

-- Prove that given the conditions, Nellie has 299 legos left
theorem nellie_final_legos : remaining_legos original_legos lost_legos given_away_legos = 299 := by
  sorry

end nellie_final_legos_l702_702302


namespace find_q_zero_l702_702915

theorem find_q_zero
  (p q r : ℝ → ℝ)  -- Define p, q, r as functions from ℝ to ℝ (since they are polynomials)
  (h1 : ∀ x, r x = p x * q x + 2)  -- Condition 1: r(x) = p(x) * q(x) + 2
  (h2 : p 0 = 6)                   -- Condition 2: constant term of p(x) is 6
  (h3 : r 0 = 5)                   -- Condition 3: constant term of r(x) is 5
  : q 0 = 1 / 2 :=                 -- Conclusion: q(0) = 1/2
sorry

end find_q_zero_l702_702915


namespace distance_points_l702_702800

theorem distance_points :
  let x1 := 1
  let y1 := 2
  let x2 := -3
  let y2 := -4
  sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 2 * sqrt 13 :=
by
  sorry

end distance_points_l702_702800


namespace minimal_m_correct_l702_702128

-- We define the functions and the conditions stated in the problem.
def vp (p : ℕ) (N : ℕ) : ℕ :=
  if N = 0 then 0 else Nat.find_greatest (fun k => p^k ∣ N)

-- We need noncomputable context because of factorial and vp not being structurally recursive.
noncomputable def minimal_m (n : ℕ) (p : ℕ) (prime_p : Nat.Prime p) (positive_n : 0 < n) : ℕ := 
  n + vp p (Nat.factorial n)

-- The main theorem we are going to prove
theorem minimal_m_correct (n : ℕ) (p : ℕ) (prime_p : Nat.Prime p) (positive_n : 0 < n) :
    ∀ (f : ℕ → ℕ) (h : ∀ k : ℕ, ∃ k' : ℕ, vp p (f k) < vp p (f k') ∧ vp p (f k') ≤ vp p (f k) + minimal_m n p prime_p positive_n), 
    minimal_m n p prime_p positive_n = n + vp p (Nat.factorial n) :=
by sorry

end minimal_m_correct_l702_702128


namespace find_inverse_of_f_l702_702218

theorem find_inverse_of_f :
  (∃ x, (3 * x^3 + 9 = 93) ∧ x = (28)^(1/3)) :=
begin
  sorry
end

end find_inverse_of_f_l702_702218


namespace exists_k_for_A_mul_v_eq_k_mul_v_l702_702110

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 6, 3]

theorem exists_k_for_A_mul_v_eq_k_mul_v (v : Fin 2 → ℝ) (h : v ≠ 0) :
  (∃ k : ℝ, A.mul_vec v = k • v) →
  k = 3 + 2 * real.sqrt 6 ∨ k = 3 - 2 * real.sqrt 6 :=
by
  sorry

end exists_k_for_A_mul_v_eq_k_mul_v_l702_702110


namespace price_of_car_is_20000_l702_702628

-- Given conditions for Quincy's car purchase
variable (down_payment : ℕ) (monthly_payment : ℕ) (loan_term_years : ℕ) (price_of_car : ℕ)
variable (annual_interest_rate : ℕ)

-- Definitions based on the conditions
def total_number_of_payments : ℕ := loan_term_years * 12
def total_amount_paid : ℕ := monthly_payment * total_number_of_payments
def final_price_of_car : ℕ := down_payment + total_amount_paid

-- Specific values given in the problem
theorem price_of_car_is_20000
  (down_payment := 5000)
  (monthly_payment := 250)
  (loan_term_years := 5)
  (annual_interest_rate := 4) :
  final_price_of_car down_payment monthly_payment loan_term_years = 20000 := by
  -- To be proved: the price of the car is $20,000
  sorry

end price_of_car_is_20000_l702_702628


namespace find_k_l702_702516

theorem find_k (k : ℕ) (hk : k > 0) (h_coeff : 15 * k^4 < 120) : k = 1 := 
by 
  sorry

end find_k_l702_702516


namespace find_angle_BCD_l702_702879

-- Defining the given conditions in the problem
def angleA : ℝ := 100
def angleD : ℝ := 120
def angleE : ℝ := 80
def angleABC : ℝ := 140
def pentagonInteriorAngleSum : ℝ := 540

-- Statement: Prove that the measure of ∠ BCD is 100 degrees given the conditions
theorem find_angle_BCD (h1 : angleA = 100) (h2 : angleD = 120) (h3 : angleE = 80) 
                       (h4 : angleABC = 140) (h5 : pentagonInteriorAngleSum = 540) :
    (angleBCD : ℝ) = 100 :=
sorry

end find_angle_BCD_l702_702879


namespace joan_spent_on_jacket_l702_702902

def total_spent : ℝ := 42.33
def shorts_spent : ℝ := 15.00
def shirt_spent : ℝ := 12.51
def jacket_spent : ℝ := 14.82

theorem joan_spent_on_jacket :
  total_spent - shorts_spent - shirt_spent = jacket_spent :=
by
  sorry

end joan_spent_on_jacket_l702_702902


namespace construct_equilateral_triangle_l702_702676

-- Define the angles
def angleA := 40
def angleB := 70
def angleC := 70

-- Problem statement
theorem construct_equilateral_triangle : 
  ∃ (triangle : Type) (A B C : triangle) (angleA angleB angleC : ℝ), 
    angleA = 40 ∧ angleB = 70 ∧ angleC = 70 ∧
    -- Add the condition that we can construct an equilateral triangle
    (∃ (equilateral_triangle : Type) (X Y Z : equilateral_triangle), 
      ∠X + ∠Y + ∠Z = 180 ∧ ∠X = ∠Y ∧ ∠Y = ∠Z ∧ ∠X = 60) :=
sorry

end construct_equilateral_triangle_l702_702676


namespace binary_to_decimal_1010101_l702_702781

def bin_to_dec (bin : List ℕ) (len : ℕ): ℕ :=
  List.foldl (λ acc (digit, idx) => acc + digit * 2^idx) 0 (List.zip bin (List.range len))

theorem binary_to_decimal_1010101 : bin_to_dec [1, 0, 1, 0, 1, 0, 1] 7 = 85 :=
by
  simp [bin_to_dec, List.range, List.zip]
  -- Detailed computation can be omitted and sorry used here if necessary
  sorry

end binary_to_decimal_1010101_l702_702781


namespace area_of_triangle_XYZ_l702_702604

theorem area_of_triangle_XYZ (AX BY CZ : ℝ)
  (hAX : AX = 6) 
  (hBY : BY = 7) 
  (hCZ : CZ = 8) :
  let XY := CZ,
      YZ := AX,
      ZX := BY,
      s := (XY + YZ + ZX) / 2,
      area := real.sqrt (s * (s - XY) * (s - YZ) * (s - ZX))
  in area = 21 * real.sqrt 15 / 4 := 
by 
  sorry

end area_of_triangle_XYZ_l702_702604


namespace clock_angle_150_at_5pm_l702_702761

theorem clock_angle_150_at_5pm :
  (∀ t : ℕ, (t = 5) ↔ (∀ θ : ℝ, θ = 150 → θ = (30 * t))) := sorry

end clock_angle_150_at_5pm_l702_702761


namespace sum_of_coefficients_l702_702649

noncomputable def simplify (x : ℝ) : ℝ := 
  (x^3 + 11 * x^2 + 38 * x + 40) / (x + 3)

theorem sum_of_coefficients : 
  (∀ x : ℝ, (x ≠ -3) → (simplify x = x^2 + 8 * x + 14)) ∧
  (1 + 8 + 14 + -3 = 20) :=
by      
  sorry

end sum_of_coefficients_l702_702649


namespace a_greater_than_b_iff_exists_f_l702_702591

variable {a b : ℝ} (f : ℝ → ℝ)

-- Assumptions that a and b are greater than 1
def conditions := a > 1 ∧ b > 1

-- Definition of g and h functions and properties
def g (x : ℝ) : ℝ := f (a ^ x) - x
def h (x : ℝ) : ℝ := f (b ^ x) - x

-- Proof that g is increasing
def is_increasing {x : ℝ} : Prop := ∀ x y, x < y → g f x < g f y

-- Proof that h is decreasing
def is_decreasing {x : ℝ} : Prop := ∀ x y, x < y → h f x > h f y

theorem a_greater_than_b_iff_exists_f :
  conditions a b →
  (a > b ↔ ∃ f : ℝ → ℝ, is_increasing f ∧ is_decreasing f) :=
by
  introv cond
  sorry

end a_greater_than_b_iff_exists_f_l702_702591


namespace days_y_worked_l702_702699

theorem days_y_worked (W d : ℕ) : 
  ( ∀ (x y : ℕ), (1 / 20 : ℚ) = x / W ∧ (1 / 15 : ℚ) = y / W ∧ (d * y + 8 * x = W) -> d = 9 ) :=
by {
  intros x y h,
  sorry
}

end days_y_worked_l702_702699


namespace arithmetic_sequence_properties_l702_702157

noncomputable def arithmetic_sequence (n : ℕ) : ℕ :=
  4 * n - 3

noncomputable def sum_of_first_n_terms (n : ℕ) : ℕ :=
  2 * n^2 - n

noncomputable def sum_of_reciprocal_sequence (n : ℕ) : ℝ :=
  n / (4 * n + 1)

theorem arithmetic_sequence_properties :
  (arithmetic_sequence 3 = 9) →
  (arithmetic_sequence 8 = 29) →
  (∀ n, arithmetic_sequence n = 4 * n - 3) ∧
  (∀ n, sum_of_first_n_terms n = 2 * n^2 - n) ∧
  (∀ n, sum_of_reciprocal_sequence n = n / (4 * n + 1)) :=
by
  sorry

end arithmetic_sequence_properties_l702_702157


namespace sequence_term_l702_702350

noncomputable def a (n : ℕ) : ℚ :=
if n = 0 then 0 else (2 * n - 1) / 2^(n - 1)

theorem sequence_term (n : ℕ) :
  (∑ i in Finset.range n, 2^i * a (i + 1)) = n^2 → a n = (2 * n - 1) / 2^(n - 1) :=
by {
  sorry
}

end sequence_term_l702_702350


namespace part_I_part_II_part_III_l702_702517

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 2) - (1 / (2^x + 1))

theorem part_I :
  ∃ a : ℝ, ∀ x : ℝ, f x = a - (1 / (2^x + 1)) → a = (1 / 2) :=
by sorry

theorem part_II :
  ∀ y : ℝ, y = f x → (-1 / 2) < y ∧ y < (1 / 2) :=
by sorry

theorem part_III :
  ∀ m n : ℝ, m + n ≠ 0 → (f m + f n) / (m^3 + n^3) > f 0 :=
by sorry

end part_I_part_II_part_III_l702_702517


namespace ellen_bought_chairs_l702_702100

-- Define the conditions
def cost_per_chair : ℕ := 15
def total_amount_spent : ℕ := 180

-- State the theorem to be proven
theorem ellen_bought_chairs :
  (total_amount_spent / cost_per_chair = 12) := 
sorry

end ellen_bought_chairs_l702_702100


namespace ratio_of_segments_l702_702946

-- Points D and E are located on the diagonals AB₁ and CA₁ of the lateral faces of the prism ABC A₁ B₁ C₁
-- such that the lines DE and BC₁ are parallel. We need to find the ratio of DE to BC₁.

theorem ratio_of_segments (A B C A1 B1 C1 D E : Type)
  (on_diagAB1D : D ∈ (segment A B1))
  (on_diagCA1E : E ∈ (segment C A1))
  (parallel_DE_BC1 : parallel (segment D E) (segment B C1)) : 
  segment_ratio (segment D E) (segment B C1) = 1 / 2 :=
sorry

end ratio_of_segments_l702_702946


namespace coordinates_with_respect_to_origin_l702_702251

theorem coordinates_with_respect_to_origin (P : ℝ × ℝ) (h : P = (2, -3)) : P = (2, -3) :=
by
  sorry

end coordinates_with_respect_to_origin_l702_702251


namespace find_costs_compare_options_l702_702037

-- Definitions and theorems
def cost1 (x y : ℕ) : Prop := 2 * x + 4 * y = 350
def cost2 (x y : ℕ) : Prop := 6 * x + 3 * y = 420

def optionACost (m : ℕ) : ℕ := 70 * m + 35 * (80 - 2 * m)
def optionBCost (m : ℕ) : ℕ := (8 * (35 * m + 2800)) / 10

theorem find_costs (x y : ℕ) : 
  cost1 x y ∧ cost2 x y → (x = 35 ∧ y = 70) :=
by sorry

theorem compare_options (m : ℕ) (h : m < 41) : 
  if m < 20 then optionBCost m < optionACost m else 
  if m = 20 then optionBCost m = optionACost m 
  else optionBCost m > optionACost m :=
by sorry

end find_costs_compare_options_l702_702037


namespace simplify_neg_cube_square_l702_702969

theorem simplify_neg_cube_square (a : ℝ) : (-a^3)^2 = a^6 :=
by
  sorry

end simplify_neg_cube_square_l702_702969


namespace dividend_eq_160_l702_702985

theorem dividend_eq_160 (divisor quotient remainder : ℕ) (h1 : divisor = 17) (h2 : quotient = 9) (h3 : remainder = 7) : 
  divisor * quotient + remainder = 160 :=
by
  rw [h1, h2, h3]
  norm_num

end dividend_eq_160_l702_702985


namespace cannot_be_rhombus_l702_702561

variables {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (AB AD CD BC : ℝ)

def is_rhombus (ABCD : Type) [quadrilateral ABCD] : Prop :=
  AB = CD ∧ CD = BC ∧ BC = AD ∧ AD = AB

theorem cannot_be_rhombus
  (ABCD_quad : quadrilateral ABCD)
  (AB_parallel_CD : ABCD_quad.parallel AB CD)
  (AB_eq_AD : ABCD_quad.equal_lengths AB AD)
  (AB_eq_BC : ABCD_quad.equal_lengths AB BC) :
  ¬ is_rhombus ABCD :=
sorry

end cannot_be_rhombus_l702_702561


namespace marbles_remaining_l702_702939

theorem marbles_remaining 
  (initial_remaining : ℕ := 400)
  (num_customers : ℕ := 20)
  (marbles_per_customer : ℕ := 15) :
  initial_remaining - (num_customers * marbles_per_customer) = 100 :=
by
  sorry

end marbles_remaining_l702_702939


namespace branches_on_main_stem_l702_702098

theorem branches_on_main_stem (x : ℕ) (h : 1 + x + x^2 = 57) : x = 7 :=
  sorry

end branches_on_main_stem_l702_702098


namespace geom_ineq_distance_sum_min_value_expr_l702_702712

-- Problem (1)
theorem geom_ineq_distance_sum (x y : ℝ) :
  sqrt(x^2 + y^2) + sqrt((x - 1)^2 + y^2) + sqrt(x^2 + (y - 1)^2) + sqrt((x - 1)^2 + (y - 1)^2) ≥ 2 * sqrt 2 := sorry

-- Problem (2)
theorem min_value_expr (a b : ℝ) (h1 : |a| ≤ sqrt 2) (h2 : b > 0) :
  (a - b)^2 + (sqrt (2 - a^2) - 9/b)^2 ≥ 8 := sorry

end geom_ineq_distance_sum_min_value_expr_l702_702712


namespace total_cost_of_sandwiches_and_sodas_l702_702020

theorem total_cost_of_sandwiches_and_sodas :
  let price_per_sandwich := 1.49
  let quantity_of_sandwiches := 2
  let price_per_soda := 0.87
  let quantity_of_sodas := 4
  let total_cost := (quantity_of_sandwiches * price_per_sandwich) + (quantity_of_sodas * price_per_soda)
  total_cost = 6.46 :=
by
  let price_per_sandwich := 1.49
  let quantity_of_sandwiches := 2
  let price_per_soda := 0.87
  let quantity_of_sodas := 4
  let total_cost := (quantity_of_sandwiches * price_per_sandwich) + (quantity_of_sodas * price_per_soda)
  show total_cost = 6.46
  sorry

end total_cost_of_sandwiches_and_sodas_l702_702020


namespace series_solution_l702_702460

noncomputable def series_sum : ℝ :=
  ∑' (n : ℕ) in set.Ici 3, ∑' (k : ℕ) in set.Icc 2 (n - 1), k^2 / 3^(n + k)

theorem series_solution : 
  | series_sum - (3 / 14) | < 1e-4 :=
by
  sorry

end series_solution_l702_702460


namespace hyperbola_eccentricity_l702_702825

theorem hyperbola_eccentricity
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (h1 : c^2 = a^2 + b^2)
  (h2 : b^2 = (1/3) * a^2) :
  real.sqrt ((a^2 + b^2) / a^2) = 2 * real.sqrt (1 / 3) :=
by
  sorry

end hyperbola_eccentricity_l702_702825


namespace props_correct_l702_702834

theorem props_correct :
  (∀ {A B C : ℝ}, sin (2 * A) = sin (2 * B) → (A = B ∨ 2 * A + 2 * B = π) → False) ∧
  (∀ {A B C : ℝ}, sin A = cos B → (A = π / 2 + B ∨ A = π / 2 - B ∨ A + B = π / 2) → False) ∧
  (∀ {A B C : ℝ}, cos A * cos B * cos C < 0 → (A > π / 2 ∨ B > π / 2 ∨ C > π / 2)) ∧
  (∀ {A B C : ℝ}, cos (A - B) * cos (B - C) * cos (C - A) = 1 → (A = π / 3 ∧ B = π / 3 ∧ C = π / 3)) := by
  sorry

end props_correct_l702_702834


namespace sum_of_squares_of_factors_of_72_l702_702685

theorem sum_of_squares_of_factors_of_72 : 
  let factors := [1, 2, 4, 8, 3, 6, 12, 9, 18, 24, 36, 72] in
  (∑ f in factors, f^2) = 7735 :=
by sorry

end sum_of_squares_of_factors_of_72_l702_702685


namespace original_number_of_laborers_l702_702726

theorem original_number_of_laborers 
(L : ℕ) (h1 : L * 15 = (L - 5) * 20) : L = 15 :=
sorry

end original_number_of_laborers_l702_702726


namespace opposite_of_negative_six_is_six_l702_702990

theorem opposite_of_negative_six_is_six : ∀ (x : ℤ), (-6 + x = 0) → x = 6 :=
by
  intro x hx
  sorry

end opposite_of_negative_six_is_six_l702_702990


namespace find_original_cost_price_l702_702440

variable (C : ℝ)

-- Conditions
def first_discount (C : ℝ) : ℝ := 0.95 * C
def second_discount (C : ℝ) : ℝ := 0.9215 * C
def loss_price (C : ℝ) : ℝ := 0.90 * C
def gain_price_before_tax (C : ℝ) : ℝ := 1.08 * C
def gain_price_after_tax (C : ℝ) : ℝ := 1.20 * C

-- Prove that original cost price is 1800
theorem find_original_cost_price 
  (h1 : first_discount C = loss_price C)
  (h2 : gain_price_after_tax C - loss_price C = 540) : 
  C = 1800 := 
sorry

end find_original_cost_price_l702_702440


namespace parabola_intersection_area_l702_702501

-- Given conditions
structure Parabola :=
  (vertex : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (point_on_parabola : ℝ × ℝ)

def line_eq := fun (y : ℝ) => y = y - 4

-- Standard equation of the parabola 
def parabola_eq (p : ℝ) := fun (y x : ℝ) => y^2 = 2 * p * x

-- The proof problem
theorem parabola_intersection_area :
  ∃ p : ℝ, ∃ A B : ℝ × ℝ, ∃ area : ℝ,
  (vertex = (0, 0)) →
  (focus = (p, 0)) →
  (point_on_parabola = (1, 2)) →
  parabola_eq 2 = fun (y x : ℝ) => y^2 = 4 * x →
  line_eq y → 
  A = (x1, y1) ∧ B = (x2, y2) → 
  area = 16 * real.sqrt 5 := sorry

end parabola_intersection_area_l702_702501


namespace find_expression_maximize_profit_l702_702437

-- First, define the conditions for the function f and its properties
def f (x : ℝ) (a : ℝ) : ℝ := (a / (x - 4)) + 10 * (x - 7)^2

-- Define the main problem statement for the first part
theorem find_expression (f6_eq_15 : f 6 15 = 15) : f x 10 = (10 / (x - 4)) + 10 * (x - 7)^2 :=
by
  sorry

-- Define the profit function
def h (x : ℝ) : ℝ := (x - 4) * f x 10

-- Define the second part; the maximization problem
theorem maximize_profit (hx_derivative: ∀ (x : ℝ), h' x = 30 * x^2 - 360 * x + 1050) 
  (x_interval: 4 < x < 7) : x = 5 :=
by
  have h' := hx_derivative 5
  sorry

end find_expression_maximize_profit_l702_702437


namespace greatest_third_term_of_arithmetic_sequence_l702_702353

theorem greatest_third_term_of_arithmetic_sequence (a d : ℕ) (h : 4 * a + 6 * d = 46) : a + 2 * d ≤ 15 :=
sorry

end greatest_third_term_of_arithmetic_sequence_l702_702353


namespace area_ADEC_l702_702894

noncomputable def area_of_quadrilateral (A B C D E : ℝ) : ℝ := 
  let AD := 12
  let AC := 18
  let AB := 24
  let BC := 6 * Real.sqrt 7
  let area_ABC := 54 * Real.sqrt 7
  let area_BDE := 31 * Real.sqrt 7
  let area_DEC := 18 * Real.sqrt 7
  area_ABC - area_BDE + area_DEC

theorem area_ADEC :
  ∃ A B C D E : ℝ, ∠C = 90 ∧
  AD = 12 ∧
  DE ⊥ AB ∧
  AB = 24 ∧
  AC = 18 ∧
  midpoint D C E → 
  area_of_quadrilateral A B C D E = 41 * Real.sqrt 7 := sorry

end area_ADEC_l702_702894


namespace probability_of_exactly_one_second_class_product_l702_702752

-- Definitions based on the conditions provided
def total_products := 100
def first_class_products := 90
def second_class_products := 10
def selected_products := 4

-- Calculation of the probability
noncomputable def probability : ℚ :=
  (Nat.choose 10 1 * Nat.choose 90 3) / Nat.choose 100 4

-- Statement to prove that the probability is 0.30
theorem probability_of_exactly_one_second_class_product : 
  probability = 0.30 := by
  sorry

end probability_of_exactly_one_second_class_product_l702_702752


namespace ratio_BL_LC_of_square_l702_702587

theorem ratio_BL_LC_of_square
  (A B C D K L : ℝ)
  (h_square : ∃ s : ℝ, A = (0,0) ∧ B = (s, 0) ∧ C = (s, s) ∧ D = (0, s))
  (h_AK : |K - A| = 3)
  (h_KB : |B - K| = 2)
  (h_L : L ∈ segment B C)
  (h_dist : distance K (line_through D L) = 3) :
  |B - L| / |L - C| = 8 / 7 := sorry

end ratio_BL_LC_of_square_l702_702587


namespace circle_divides_sides_in_ratio_l702_702982

theorem circle_divides_sides_in_ratio
  (a : ℝ) (trapezoid : Type)
  (BC AD MN CK KD CD : ℝ)
  (ratio_bases : 3/2 = AD / BC)
  (circle_diameter : AD = 6 * a)
  (smaller_base_segment : MN = BC / 2)
  (CK_KD_sum : CK + KD = CD)
  (smaller_base_half : BC = 4 * a)
  (larger_base : AD = 6 * a)
  (CK : CK = a)
  (KD : KD = 2 * a)
  (non_parallel_side : CD = 3 * a) :
  CK / KD = 1 / 2 :=
by 
  sorry

end circle_divides_sides_in_ratio_l702_702982


namespace find_d_l702_702359

-- Defining the basic points and their corresponding conditions
structure Point (α : Type) :=
(x : α) (y : α) (z : α)

def a : Point ℝ := ⟨1, 0, 1⟩
def b : Point ℝ := ⟨0, 1, 0⟩
def c : Point ℝ := ⟨0, 1, 1⟩

-- introducing k as a positive integer
variables (k : ℤ) (hk : k > 0 ∧ k ≠ 6 ∧ k ≠ 1)

def d (k : ℤ) : Point ℝ := ⟨k*d, k*d, -d⟩ where d := -(k / (k-1))

-- The proof statement
theorem find_d (k : ℤ) (hk : k > 0 ∧ k ≠ 6 ∧ k ≠ 1) :
∃ d: ℝ, d = - (k / (k-1)) :=
sorry

end find_d_l702_702359


namespace power_of_p_in_product_l702_702544
open Nat

-- Conditions: p and q are distinct primes
axiom p : ℕ
axiom q : ℕ
axiom p_prime : Prime p
axiom q_prime : Prime q
axiom p_ne_q : p ≠ q

-- Condition: The product \( p^n \cdot q^6 \) has 28 divisors
axiom num_divisors (n : ℕ) : (n + 1) * (6 + 1) = 28

-- The goal: Prove that \( n = 3 \)
theorem power_of_p_in_product : ∃ n : ℕ, num_divisors n ∧ n = 3 :=
by
  existsi 3
  split
  sorry

end power_of_p_in_product_l702_702544


namespace count_triangles_with_same_area_as_shaded_l702_702998

def is_noncollinear (a b c : ℕ × ℕ) : Prop :=
  (a.1 ≠ b.1 ∨ a.2 ≠ b.2) ∧
  (b.1 ≠ c.1 ∨ b.2 ≠ c.2) ∧
  (a.1 ≠ c.1 ∨ a.2 ≠ c.2) ∧
  (a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2) ≠ 0)

noncomputable def triangles_with_area (points : list (ℕ × ℕ)) (area: ℕ) :=
  { t : set (ℕ × ℕ) // t.card = 3 ∧ ∀ (a b c ∈ t), is_noncollinear a b c ∧
    1/2 * abs (a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2)) = area }

theorem count_triangles_with_same_area_as_shaded {points : list (ℕ × ℕ)} (area : ℕ) :
  points = [
      (0, 0), (1, 0), (2, 0), (3, 0),
      (0, 1), (1, 1), (2, 1), (3, 1),
      (0, 2), (1, 2), (2, 2), (3, 2),
      (0, 3), (1, 3), (2, 3), (3, 3)
    ] ∧ area = 1 →
  (∃ triangles : finset (finset (ℕ × ℕ)),
    triangles.card = 48 ∧ ∀ t ∈ triangles, ∃ s, triangles_with_area points area s) :=
by
  sorry

end count_triangles_with_same_area_as_shaded_l702_702998


namespace duration_of_each_turn_l702_702689

-- Definitions based on conditions
def Wa := 1 / 4
def Wb := 1 / 12

-- Define the duration of each turn as T
def T : ℝ := 1 -- This is the correct answer we proved

-- Given conditions
def total_work_done := 6 * Wa + 6 * Wb

-- Lean statement to prove 
theorem duration_of_each_turn : T = 1 := by
  -- According to conditions, the total work done by a and b should equal the whole work
  have h1 : 3 * Wa + 3 * Wb = 1 := by sorry
  -- Let's conclude that T = 1
  sorry

end duration_of_each_turn_l702_702689


namespace min_total_cost_at_n_equals_1_l702_702972

-- Define the conditions and parameters
variables (a : ℕ) -- The total construction area
variables (n : ℕ) -- The number of floors

-- Definitions based on the given problem conditions
def land_expropriation_cost : ℕ := 2388 * a
def construction_cost (n : ℕ) : ℕ :=
  if n = 0 then 0 else if n = 1 then 455 * a else (455 * n * a + 30 * (n-2) * (n-1) / 2 * a)

-- Total cost including land expropriation and construction costs
def total_cost (n : ℕ) : ℕ := land_expropriation_cost a + construction_cost a n

-- The minimum total cost occurs at n = 1
theorem min_total_cost_at_n_equals_1 :
  ∃ n, n = 1 ∧ total_cost a n = 2788 * a :=
by sorry

end min_total_cost_at_n_equals_1_l702_702972


namespace probability_final_ball_white_l702_702033

theorem probability_final_ball_white
  (p q : ℕ) :
  let final_probability : ℕ → ℕ
  | 0        := 0
  | (n + 1)  := 1
  in final_probability p = if p % 2 = 0 then 0 else 1 :=
by
  sorry

end probability_final_ball_white_l702_702033


namespace value_of_Q_l702_702605

theorem value_of_Q (n : ℕ) (h : n = 2010) :
  let Q := ∏ k in Finset.range n, (1 + 1 / (k + 2)) in
  Q = 2011 / 2 :=
by
  have h2 : Q = ∏ k in Finset.range n, (1 + 1 / (k + 2)) := rfl
  rw h at h2
  rw Finset.prod_range_succ at h2
  sorry

end value_of_Q_l702_702605


namespace problem_solution_l702_702773

-- Definitions for conditions
def cos30 : ℂ := Real.cos (Real.pi / 6)
def sin30 : ℂ := Real.sin (Real.pi / 6)
def complex_number : ℂ := 2 * cos30 + 2 * complex.I * sin30

-- The main theorem
theorem problem_solution : (complex_number ^ 10) = (512 - 512 * complex.I * Real.sqrt 3) :=
by
  -- Proof skipped
  sorry

end problem_solution_l702_702773


namespace pyramid_volume_l702_702424

-- Define the base of the pyramid
def base_length : ℝ := 7
def base_width : ℝ := 9

-- Define the length of the edges from the apex to the corners
def edge_length : ℝ := 15

-- Define the volume of the pyramid
noncomputable def volume : ℝ := (1 / 3) * (base_length * base_width) * real.sqrt(192.5)

-- Theorem to prove the volume is as calculated
theorem pyramid_volume : volume = 21 * real.sqrt(192.5) := by
  sorry

end pyramid_volume_l702_702424


namespace remove_seven_magazines_l702_702664

noncomputable def magazines_cover (table_area : ℝ) (areas : Fin 15 → ℝ) : Prop :=
  (∀ i, 0 ≤ areas i) ∧ (sum univ areas = table_area)

theorem remove_seven_magazines 
  (areas : Fin 15 → ℝ)
  (h : magazines_cover 1 areas) :
  ∃ (remaining : Finset (Fin 15)), remaining.card = 8 ∧ (sum remaining areas) ≥ (8 / 15) :=
sorry

end remove_seven_magazines_l702_702664


namespace units_digit_L_L15_l702_702641

def lucas : ℕ → ℕ 
| 0 := 2
| 1 := 1
| n + 2 := lucas (n + 1) + lucas n

theorem units_digit_L_L15 : (lucas (lucas 15) % 10) = 7 := 
by
  sorry

end units_digit_L_L15_l702_702641


namespace number_of_possible_orders_l702_702073

-- Define the total number of bowlers participating in the playoff
def num_bowlers : ℕ := 6

-- Define the number of games
def num_games : ℕ := 5

-- Define the number of possible outcomes per game
def outcomes_per_game : ℕ := 2

-- Prove the total number of possible orders for bowlers to receive prizes
theorem number_of_possible_orders : (outcomes_per_game ^ num_games) = 32 :=
by sorry

end number_of_possible_orders_l702_702073


namespace parallelogram_base_length_l702_702052

theorem parallelogram_base_length (area height : ℝ) (h_area : area = 108) (h_height : height = 9) : 
  ∃ base : ℝ, base = area / height ∧ base = 12 := 
  by sorry

end parallelogram_base_length_l702_702052


namespace bonnie_roark_wire_ratio_l702_702763

theorem bonnie_roark_wire_ratio :
  let bonnie_wire_length := 12 * 8
  let bonnie_cube_volume := 8 ^ 3
  let roark_cube_volume := 2
  let roark_edge_length := 1.5
  let roark_cube_edge_count := 12
  let num_roark_cubes := bonnie_cube_volume / roark_cube_volume
  let roark_wire_per_cube := roark_cube_edge_count * roark_edge_length
  let roark_total_wire := num_roark_cubes * roark_wire_per_cube
  bonnie_wire_length / roark_total_wire = 1 / 48 :=
  by
  sorry

end bonnie_roark_wire_ratio_l702_702763


namespace perfect_number_phi_power_of_two_l702_702103

-- Define a perfect number
def isPerfectNumber (n : ℕ) : Prop :=
  (∑ d in (Finset.filter (λ d, d ∣ n) (Finset.range (n + 1))), d) = 2 * n

-- Define a power of 2
def isPowerOfTwo (m : ℕ) : Prop :=
  ∃ k : ℕ, m = 2 ^ k

-- Main theorem
theorem perfect_number_phi_power_of_two (n : ℕ) (hn : isPerfectNumber n) (phi_power_two : isPowerOfTwo (Nat.totient n)) : n = 6 :=
sorry

end perfect_number_phi_power_of_two_l702_702103


namespace proof_problem_l702_702093

def smallest_greater (m : ℝ) : ℤ :=
  ⌈m⌉

def largest_not_greater (m : ℝ) : ℤ :=
  ⌊m⌋

theorem proof_problem (x y : ℝ) 
  (h1 : 3 * ↑(largest_not_greater x) + 2 * y = 27)
  (h2 : 2 * x - largest_not_greater y = 28) :
  largest_not_greater x + smallest_greater y = 8 :=
sorry

end proof_problem_l702_702093


namespace lattice_points_on_hyperbola_l702_702211

-- Condition of the problem: defining the hyperbola equation
def hyperbola (x y : ℤ) : Prop := x^2 - 4 * y^2 = 3000^2

-- The theorem stating the mathematically equivalent proof problem
theorem lattice_points_on_hyperbola : 
  {p : ℤ × ℤ | hyperbola p.1 p.2}.to_finset.card = 120 :=
begin
  sorry
end

end lattice_points_on_hyperbola_l702_702211


namespace number_of_type_one_triplets_l702_702027

-- Define the number of teams
def n_teams : ℕ := 15

-- Define the number of matches each team won
def matches_won_per_team : ℕ := 7

-- Define the total number of triplets
def total_triplets := Nat.choose n_teams 3

-- Define the number of triplets where each team in the trio won one match
def desired_triplets : ℕ := 140

-- Theorem statement
theorem number_of_type_one_triplets : total_triplets = 455 ∧ desired_triplets = 140 :=
  sorry

end number_of_type_one_triplets_l702_702027


namespace find_c_l702_702278

def p (x : ℝ) := 4 * x - 9
def q (x : ℝ) (c : ℝ) := 5 * x - c

theorem find_c : ∃ (c : ℝ), p (q 3 c) = 14 ∧ c = 9.25 :=
by
  sorry

end find_c_l702_702278


namespace smallest_nk_2_to_k_l702_702488

noncomputable def smallest_nk (k : ℕ) : ℕ :=
  Inf {n : ℕ | ∃ (A : fin k → matrix (fin n) (fin n) ℝ)
              (h1 : ∀ i, (A i) * (A i) = 0)
              (h2 : ∀ i j, (A i) * (A j) = (A j) * (A i)),
                matrix.mul (A 0) (matrix.mul (A 1) ... (A (k-1))) ≠ 0 }

theorem smallest_nk_2_to_k (k : ℕ) : smallest_nk k = 2^k :=
sorry

end smallest_nk_2_to_k_l702_702488


namespace sum_of_numbers_in_100th_bracket_l702_702792

theorem sum_of_numbers_in_100th_bracket:
  let sequence (n : ℕ) := 2 * n + 1 in
  let groups (k : ℕ) := (finset.range (k + 1)).sum in
  let elements := (finset.range 14).image (λ n, 2 * ((finset.range n).sum) + 1) in
  finset.sum elements = 2856 :=
by
  sorry

end sum_of_numbers_in_100th_bracket_l702_702792


namespace solve_f_half_solve_f_general_gt1_solve_f_general_eq1_l702_702379

-- Define f(x) when a = 1/2
def f_half (x : ℝ) : ℝ := x^2 - (5/2) * x + 10

-- Prove the inequality and the solution set for a = 1/2
theorem solve_f_half : 
  (∀ x : ℝ, f_half x ≤ 0 ↔ (1/2 : ℝ) ≤ x ∧ x ≤ 2) :=
by sorry

-- Define general form of f(x)
def f_general (a x : ℝ) : ℝ := (x - 1/a) * (x - a)

-- Prove the solution set for a > 1
theorem solve_f_general_gt1 {a : ℝ} (h : a > 1) : 
  (∀ x : ℝ, f_general a x ≤ 0 ↔ 1/a ≤ x ∧ x ≤ a) :=
by sorry

-- Prove the solution for a = 1
theorem solve_f_general_eq1 : 
  (∀ x : ℝ, f_general 1 x ≤ 0 ↔ x = 1) :=
by sorry

end solve_f_half_solve_f_general_gt1_solve_f_general_eq1_l702_702379


namespace mean_median_mode_relation_l702_702758

/-- Define the list of catches Ashley recorded on her outings -/
def catches : List ℕ := [1, 2, 2, 3, 5, 5, 5, 1, 0, 4, 4, 6]

/-- Mean, median, and mode of Ashley's catches satisfy the relationship mean < median < mode -/
theorem mean_median_mode_relation : 
  let mean := (catches.sum.toRat / catches.length) 
  let median := ((catches.sorted.get! (catches.length / 2 - 1) + catches.sorted.get! (catches.length / 2)).toRat / 2)
  let mode := catches.mode.getD 0
  mean < median ∧ median < mode := 
sorry

end mean_median_mode_relation_l702_702758


namespace hyperbola_eccentricity_is_sqrt3_l702_702196

-- Definition of the hyperbola passing through point M(2, 2)
def hyperbola_passes_through (m : ℝ) : Prop :=
  (2^2 / 2) - (2^2 / m) = 1

-- Definition of eccentricity
def eccentricity (a b : ℝ) : ℝ :=
  let c := Real.sqrt(a^2 + b^2)
  c / a

theorem hyperbola_eccentricity_is_sqrt3 (m : ℝ)
  (h : hyperbola_passes_through m) :
  eccentricity (Real.sqrt 2) (Real.sqrt m) = Real.sqrt 3 :=
by {
  sorry
}

end hyperbola_eccentricity_is_sqrt3_l702_702196


namespace compute_sin_2theta_l702_702164

-- Define the conditions from the problem
def condition (θ : ℝ) : Prop :=
  2^(-2 + 2 * Real.sin θ) + 3 = 2^(1 / 2 + Real.sin θ)

-- State the theorem to prove the desired result
theorem compute_sin_2theta (θ : ℝ) (h : condition θ) : Real.sin (2 * θ) = (3 * Real.sqrt 7) / 8 :=
sorry

end compute_sin_2theta_l702_702164


namespace parallelogram_area_right_triangle_area_l702_702981

theorem parallelogram_area (base : ℕ) (height : ℕ) (h_base : base = 16) (h_height : height = 25) :
  base * height = 400 :=
by
  rw [h_base, h_height]
  simp

theorem right_triangle_area (side1 side2 : ℕ) (h_side1 : side1 = 3) (h_side2 : side2 = 4) :
  (side1 * side2) / 2 = 6 :=
by
  rw [h_side1, h_side2]
  norm_num

end parallelogram_area_right_triangle_area_l702_702981


namespace triangle_area_l702_702611

variables {U V W : Type} [MetricSpace U] [MetricSpace V] [MetricSpace W]

structure Point (α : Type) [MetricSpace α] :=
  (x : ℝ)
  (y : ℝ)

def distance (P Q : Point ℝ) : ℝ :=
  real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

def line (P Q: Point ℝ): (Point ℝ) → Prop :=
  λ R, ∃ t: ℝ, R.x = P.x + t * (Q.x - P.x) ∧ R.y = P.y + t * (Q.y - P.y)


theorem triangle_area {U V W G: Point ℝ}
  (h1: distance U W = 26)
  (h2: line U W G = G.y = G.x - 1)
  (h3: line V W G = G.y = 2 * G.x) :
  let area := ((U.x * V.y + V.x * W.y + W.x * U.y) - (U.y * V.x + V.y * W.x + W.y * U.x)) / 2 in
  abs(area) = 52 :=
by
  sorry

end triangle_area_l702_702611


namespace remainder_ones_more_than_zeros_l702_702596

def countOnesMoreThanZeros(n: ℕ): ℕ := 
  (List.range (n + 1)).count (λ k, (k.bits).count (λ b, b = 1) > (k.bits).count (λ b, b = 0))

theorem remainder_ones_more_than_zeros :
  let M := countOnesMoreThanZeros 2010
  M % 1000 = 162 :=
by
  sorry

end remainder_ones_more_than_zeros_l702_702596


namespace size_of_angle_C_l702_702819

theorem size_of_angle_C 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a = 5) 
  (h2 : b + c = 2 * a) 
  (h3 : 3 * Real.sin A = 5 * Real.sin B) : 
  C = 2 * Real.pi / 3 := 
sorry

end size_of_angle_C_l702_702819


namespace water_on_wednesday_l702_702205

-- Define the total water intake for the week.
def total_water : ℕ := 60

-- Define the water intake amounts for specific days.
def water_on_mon_thu_sat : ℕ := 9
def water_on_tue_fri_sun : ℕ := 8

-- Define the number of days for each intake.
def days_mon_thu_sat : ℕ := 3
def days_tue_fri_sun : ℕ := 3

-- Define the water intake calculated for specific groups of days.
def total_water_mon_thu_sat := water_on_mon_thu_sat * days_mon_thu_sat
def total_water_tue_fri_sun := water_on_tue_fri_sun * days_tue_fri_sun

-- Define the total water intake for these days combined.
def total_water_other_days := total_water_mon_thu_sat + total_water_tue_fri_sun

-- Define the water intake for Wednesday, which we need to prove to be 9 liters.
theorem water_on_wednesday : total_water - total_water_other_days = 9 := by
  -- Proof omitted.
  sorry

end water_on_wednesday_l702_702205


namespace cos_2theta_minus_2pi_over_3_equals_neg_7_over_8_l702_702203

variables (θ : Real)

def m := (2 * Real.sqrt 3, Real.cos θ)
def n := (Real.sin θ, 2)

theorem cos_2theta_minus_2pi_over_3_equals_neg_7_over_8 
  (h : m θ • n θ = 1) : Real.cos (2 * θ - 2 * Real.pi / 3) = -7 / 8 :=
by
  sorry

end cos_2theta_minus_2pi_over_3_equals_neg_7_over_8_l702_702203


namespace smallest_num_eggs_l702_702012

-- Define conditions regarding the number of containers and eggs
def conditions (c : ℕ) : Prop := 15 * c - 6 > 150

-- Define the smallest number of eggs
def num_eggs (c : ℕ) : ℕ := 15 * c - 6

-- Theorem: Prove that the smallest number of eggs is 159
theorem smallest_num_eggs : ∃ c : ℕ, conditions c ∧ num_eggs c = 159 := by
  use 11
  simp [conditions, num_eggs]
  linarith


end smallest_num_eggs_l702_702012


namespace inverse_function_undefined_at_2_l702_702862

def f (x : ℝ) : ℝ := (2 * x - 6) / (x - 5)

noncomputable def f_inv (y : ℝ) : ℝ := (6 - 5 * y) / (2 - y)

theorem inverse_function_undefined_at_2 : f_inv 2 = 0 := by
  sorry

end inverse_function_undefined_at_2_l702_702862


namespace dot_product_zero_l702_702514

variables {V : Type*} [inner_product_space ℝ V]

theorem dot_product_zero {A B C O : V}
  (hOcirc : dist O A = dist O B ∧ dist O B = dist O C ∧ dist O C = dist O A)
  (hAO : O = (1 / 2 : ℝ) • (B + C)) :
  (A - B) ⬝ (A - C) = 0 :=
sorry

end dot_product_zero_l702_702514


namespace machine_production_time_difference_undetermined_l702_702936

theorem machine_production_time_difference_undetermined :
  ∀ (machineP_machineQ_440_hours_diff : ℝ)
    (machineQ_production_rate : ℝ)
    (machineA_production_rate : ℝ),
    machineA_production_rate = 4.000000000000005 →
    machineQ_production_rate = machineA_production_rate * 1.1 →
    machineP_machineQ_440_hours_diff > 0 →
    machineQ_production_rate * machineP_machineQ_440_hours_diff = 440 →
    ∃ machineP_production_rate, 
    ¬(∃ hours_diff : ℝ, hours_diff = 440 / machineP_production_rate - 440 / machineQ_production_rate) := sorry

end machine_production_time_difference_undetermined_l702_702936


namespace choose_model_l702_702365

theorem choose_model :
  ∀ (x y : ℝ),
    (x = 1 → y = 2 → (y = x^2 + 1 ∧ y = 3 * x - 1)) ∧
    (x = 2 → y = 5 → (y = x^2 + 1 ∧ y = 3 * x - 1)) →
    (let yA := 3^2 + 1 in
     let yB := 3 * 3 - 1 in
     |10.2 - yA| < |10.2 - yB| →
     "Model A")
by
  intros x y h,
  funext,
  ring,
  sorry

end choose_model_l702_702365


namespace period_is_pi_center_of_symmetry_l702_702806

def f (x : Real) : Real :=
  sin (2 * x) - cos (2 * x)

theorem period_is_pi : ∀ x : Real, f (x + π) = f x := by
  sorry

theorem center_of_symmetry : f (π / 4) = 0 := by
  sorry

end period_is_pi_center_of_symmetry_l702_702806


namespace intervals_of_monotonicity_range_of_m_product_ln_n_lt_inv_n_l702_702190

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
a * real.log x - a * x - 3

theorem intervals_of_monotonicity (a : ℝ) :
  (a > 0 → (∀ x > 0, x < 1 → deriv (λ x, f x a) x > 0) ∧ (∀ x > 1, deriv (λ x, f x a) x < 0)) ∧
  (a < 0 → (∀ x > 1, deriv (λ x, f x a) x > 0) ∧ (∀ x > 0, x < 1, deriv (λ x, f x a) x < 0)) ∧
  (a = 0 → ∀ x > 0, deriv (λ x, f x a) x = 0) :=
sorry

theorem range_of_m (m : ℝ) :
  (-37 / 3 < m ∧ m < -9) :=
sorry

theorem product_ln_n_lt_inv_n (n : ℕ) (hn : 2 ≤ n) :
  (∏ i in finset.range (n + 1).filter (λ i, 2 <= i), real.log i / i) < (1 / n) :=
sorry

end intervals_of_monotonicity_range_of_m_product_ln_n_lt_inv_n_l702_702190


namespace range_of_m_l702_702835

noncomputable def f (x m : ℝ) : ℝ := x^2 + m*x + 1

theorem range_of_m (m : ℝ): (∃ x₀ : ℝ, x₀ > 0 ∧ f x₀ m < 0) → m < -2 :=
by
  assume h : ∃ x₀ : ℝ, x₀ > 0 ∧ f x₀ m < 0
  sorry

end range_of_m_l702_702835


namespace f_1_eq_0_range_x_l702_702930

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined_on_Rstar : ∀ x : ℝ, ¬ (x = 0) → f x = sorry
axiom f_4_eq_1 : f 4 = 1
axiom f_mult : ∀ (x₁ x₂ : ℝ), ¬ (x₁ = 0) → ¬ (x₂ = 0) → f (x₁ * x₂) = f x₁ + f x₂
axiom f_increasing : ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f x₁ < f x₂

theorem f_1_eq_0 : f 1 = 0 := sorry

theorem range_x (x : ℝ) : f (3 * x + 1) + f (2 * x - 6) ≤ 3 → 3 < x ∧ x ≤ 5 := sorry

end f_1_eq_0_range_x_l702_702930


namespace probability_sum_divisible_by_five_l702_702229

theorem probability_sum_divisible_by_five :
  let outcomes := { (d₁, d₂) | d₁ ∈ Finset.range (6 + 1) ∧ d₂ ∈ Finset.range (6 + 1) }
  let favorable_outcomes := outcomes.filter (λ (d : ℕ × ℕ), (d.1 + d.2) % 5 = 0)
  (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ) = 7 / 36 :=
by
  let outcomes := { (d₁, d₂) | d₁ ∈ Finset.range (6 + 1) ∧ d₂ ∈ Finset.range (6 + 1) }
  let favorable_outcomes := outcomes.filter (λ (d : ℕ × ℕ), (d.1 + d.2) % 5 = 0)
  have h₁ : outcomes.card = 36 := sorry
  have h₂ : favorable_outcomes.card = 7 := sorry
  have h₃ : (7 : ℚ) / 36 = 7 / 36 := sorry
  rw ←h₁
  rw ←h₂
  exact h₃

end probability_sum_divisible_by_five_l702_702229


namespace dot_product_calculation_l702_702849

def vector := (ℤ × ℤ)

def dot_product (v1 v2 : vector) : ℤ :=
  v1.1 * v2.1 + v1.2 * v2.2

def a : vector := (1, 3)
def b : vector := (-1, 2)

def scalar_mult (c : ℤ) (v : vector) : vector :=
  (c * v.1, c * v.2)

def vector_add (v1 v2 : vector) : vector :=
  (v1.1 + v2.1, v1.2 + v2.2)

theorem dot_product_calculation :
  dot_product (vector_add (scalar_mult 2 a) b) b = 15 := by
  sorry

end dot_product_calculation_l702_702849


namespace magnitude_z_is_sqrt_13_l702_702171

def z := (5 + complex.i) / (1 - complex.i)

theorem magnitude_z_is_sqrt_13 : complex.abs z = real.sqrt 13 := 
by sorry

end magnitude_z_is_sqrt_13_l702_702171


namespace incorrect_statement_B_l702_702489

variable {x : ℝ}

def y (x : ℝ) : ℝ := 2 / x

theorem incorrect_statement_B (hx : x > 0) : ¬ (∀ x1 x2 : ℝ, x1 > x2 → y x1 > y x2) :=
by {
  intro h,
  specialize h (x + 1) x,
  rw [y, y],
  simp,
  sorry
}

end incorrect_statement_B_l702_702489


namespace rotate_point_l702_702564

theorem rotate_point (A : ℝ × ℝ) :
  A = (-1/2, (real.sqrt 3)/2) →
  let OA′ := (real.cos (2*real.pi/3 - real.pi/2), real.sin (2*real.pi/3 - real.pi/2))
  in OA′ = (real.sqrt 3 / 2, 1 / 2) :=
begin
  intro h,
  simp [h],
  sorry
end

end rotate_point_l702_702564


namespace coin_difference_l702_702942

-- Definitions based on problem conditions
def denominations : List ℕ := [5, 10, 25, 50]
def amount_owed : ℕ := 55

-- Proof statement
theorem coin_difference :
  let min_coins := 1 + 1 -- one 50-cent coin and one 5-cent coin
  let max_coins := 11 -- eleven 5-cent coins
  max_coins - min_coins = 9 :=
by
  -- Proof details skipped
  sorry

end coin_difference_l702_702942


namespace average_speed_approx_43_29_mph_l702_702408

noncomputable def kmh_to_mph (x : ℝ) : ℝ := x / 1.60934

noncomputable def average_speed (total_distance : ℝ) (speeds : List ℝ) (distances : List ℝ): ℝ :=
  let times := List.map2 (λ d s => d / s) distances speeds
  total_distance / (List.sum times)

theorem average_speed_approx_43_29_mph :
  let distances := [75.0, 75.0, 75.0, 75.0]
  let speeds := [75.0, kmh_to_mph 45.0, kmh_to_mph 50.0, 90.0]
  abs (average_speed 300.0 speeds distances - 43.29) < 0.01 := 
by
  unfold distances speeds average_speed kmh_to_mph
  sorry

end average_speed_approx_43_29_mph_l702_702408


namespace g_54_l702_702283

def g : ℕ → ℤ := sorry

axiom g_multiplicative (x y : ℕ) (hx : x > 0) (hy : y > 0) : g (x * y) = g x + g y
axiom g_6 : g 6 = 10
axiom g_18 : g 18 = 14

theorem g_54 : g 54 = 18 := by
  sorry

end g_54_l702_702283


namespace coeff_x2_term_in_expansion_l702_702119

theorem coeff_x2_term_in_expansion :
  let p1 := (3 : ℤ) * X^2 - (2 : ℤ) * X + (5 : ℤ)
  let p2 := -(4 : ℤ) * X^2 + (3 : ℤ) * X + (6 : ℤ)
  coeff (p1 * p2) 2 = -8 :=
by
  sorry

end coeff_x2_term_in_expansion_l702_702119


namespace concyclic_roots_l702_702529

/-- Given the real-coefficient equations x^2 - 2x + 2 = 0 and x^2 + 2mx + 1 = 0,
    prove that the four distinct roots are concyclic in the complex plane 
    if and only if -1 < m < 1 or m = -3/2. -/
theorem concyclic_roots (m : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℂ, (x₁ ^ 2 - 2 * x₁ + 2 = 0) ∧ (x₂ ^ 2 - 2 * x₂ + 2 = 0) ∧
    (x₃ ^ 2 + 2 * m * x₃ + 1 = 0) ∧ (x₄ ^ 2 + 2 * m * x₄ + 1 = 0) ∧ 
    (concyclic (fin.to_set [x₁, x₂, x₃, x₄]))) ↔ 
    (-1 < m ∧ m < 1 ∨ m = -3 / 2) :=
sorry

end concyclic_roots_l702_702529


namespace compute_expression_at_x_eq_3_l702_702084

theorem compute_expression_at_x_eq_3 :
  (let x := 3 in (x^8 + 18 * x^4 + 81) / (x^4 + 9)) = 90 :=
by
  sorry

end compute_expression_at_x_eq_3_l702_702084


namespace new_ratio_of_liquids_l702_702046

-- Define initial conditions
def initial_ratio := 4 / 1
def initial_liquid_A := 32
def replaced_volume := 20

-- Theorem statement proving the new ratio after replacing 20 L of the mixture with liquid B
theorem new_ratio_of_liquids :
  let initial_total_mixture := initial_liquid_A * 5 / 4 in -- Total mixture initially
  let final_liquid_A := (initial_liquid_A - (replaced_volume * 4 / 5)) in -- Remaining Liquid A
  let final_liquid_B := ((initial_total_mixture - initial_liquid_A) - (replaced_volume * 1 / 5)) + replaced_volume in -- Remaining + Added Liquid B
  final_liquid_A / final_liquid_B = 2 / 3 :=
by {
  sorry
}

end new_ratio_of_liquids_l702_702046


namespace sarah_marry_age_l702_702466

/-- Sarah is 9 years old. -/
def Sarah_age : ℕ := 9

/-- Sarah's name has 5 letters. -/
def Sarah_name_length : ℕ := 5

/-- The game's rule is to add the number of letters in the player's name 
    to twice the player's age. -/
def game_rule (name_length age : ℕ) : ℕ :=
  name_length + 2 * age

/-- Prove that Sarah will get married at the age of 23. -/
theorem sarah_marry_age : game_rule Sarah_name_length Sarah_age = 23 := 
  sorry

end sarah_marry_age_l702_702466


namespace option2_cheaper_for_50_students_find_students_for_equal_costs_l702_702554

theorem option2_cheaper_for_50_students :
  let ticket_price := 30
  let discount_1 := 0.8
  let discount_2 := 0.9
  let free_tickets := 6
  let students := 50
  (students * ticket_price * discount_1 > (students - free_tickets) * ticket_price * discount_2) := by
  trivial -- Insert proper validation.

theorem find_students_for_equal_costs :
  let ticket_price := 30
  let discount_1 := 0.8
  let discount_2 := 0.9
  let free_tickets := 6
  ∃ x : ℕ, x > 40 ∧ x * ticket_price * discount_1 = (x - free_tickets) * ticket_price * discount_2 := by
  existsi (54 : ℕ)
  sorry -- Proof to show that x = 54 satisfies the given equation.

end option2_cheaper_for_50_students_find_students_for_equal_costs_l702_702554


namespace smallest_b_l702_702486

theorem smallest_b (b r s : ℕ) (h₁ : ∃ r s, (x^2 + b * x + 2016 = (x + r) * (x + s)) ∧ r > 0 ∧ s > 0 ∧ 8 ∣ s) : b = 260 :=
begin
  sorry
end

end smallest_b_l702_702486


namespace math_proof_problem_l702_702176

variable {a b : ℝ} (f : ℝ → ℝ)

-- Given conditions
def condition1 := ∀ x : ℝ, -1 < x ∧ x < 1 → f x = (a*x + b) / (x^2 + 1)
def condition2 := ∀ x : ℝ, f (-x) = -f x
def condition3 := f (1/2) = 2/5

-- To prove
def goal1 := ∀ x, f x = x / (x^2 + 1)
def goal2 := ∀ x, -1 < x ∧ x < 1 → ( (1 - x) * (1 + x) / ( (x^2 + 1) * (x^2 + 1) )  > 0)
def goal3 := ∀ x, 0 < x ∧ x < 1/3 → f(2*x - 1) + f(x) < 0

theorem math_proof_problem :
  (condition1 f a b) → (condition2 f) → (condition3 f) → (goal1 f) ∧ (goal2 f) ∧ (goal3 f) := by
  sorry

end math_proof_problem_l702_702176


namespace max_sectional_area_of_cone_l702_702518

noncomputable def sectional_area (l θ : ℝ) : ℝ := (1/2) * l^2 * sin θ

theorem max_sectional_area_of_cone :
  ∃ θ : ℝ, θ = π / 3 → ∃ l : ℝ, l = 3 → ∃ S : ℝ, S = 9 / 2 :=
begin
  sorry
end

end max_sectional_area_of_cone_l702_702518


namespace true_proposition_l702_702162

def p : Prop := ∀ x : ℝ, x < 1 → log (1/2) x < 0
def q : Prop := ∃ x : ℝ, x^2 ≥ 2 * x

theorem true_proposition : p ∨ q := by sorry

end true_proposition_l702_702162


namespace molecular_weight_N2O3_l702_702476

variable (atomic_weight_N : ℝ) (atomic_weight_O : ℝ)
variable (n_N_atoms : ℝ) (n_O_atoms : ℝ)
variable (expected_molecular_weight : ℝ)

theorem molecular_weight_N2O3 :
  atomic_weight_N = 14.01 →
  atomic_weight_O = 16.00 →
  n_N_atoms = 2 →
  n_O_atoms = 3 →
  expected_molecular_weight = 76.02 →
  (n_N_atoms * atomic_weight_N + n_O_atoms * atomic_weight_O = expected_molecular_weight) :=
by
  intros
  sorry

end molecular_weight_N2O3_l702_702476


namespace fixed_point_tangent_circle_l702_702329

theorem fixed_point_tangent_circle (x y a b t : ℝ) :
  (x ^ 2 + (y - 2) ^ 2 = 16) ∧ (a * 0 + b * 2 - 12 = 0) ∧ (y = -6) ∧ 
  (t * x - 8 * y = 0) → 
  (0, 0) = (0, 0) :=
by 
  sorry

end fixed_point_tangent_circle_l702_702329


namespace PR_QS_intersection_ratios_l702_702021

theorem PR_QS_intersection_ratios (A B C D P Q R S N : Type) 
(h1 : BP_ratio : AB = 1 : 3) 
(h2 : CR_ratio : CD = 1 : 3) 
(h3 : AS_ratio : AD = 1 : 4) 
(h4 : BQ_ratio : BC = 1 : 4) 
(hP : P ∈ lineSegment A B)
(hQ : Q ∈ lineSegment B C)
(hR : R ∈ lineSegment C D)
(hS : S ∈ lineSegment D A)
(hN : N ∈ (lineSegment P R) ∩ (lineSegment Q S)) :
  divides_ratio PR N 1 3 ∧ divides_ratio QS N 1 2 :=
sorry

end PR_QS_intersection_ratios_l702_702021


namespace tape_pieces_needed_l702_702727

-- Define the setup: cube edge length and tape width
def edge_length (n : ℕ) : ℕ := n
def tape_width : ℕ := 1

-- Define the statement we want to prove
theorem tape_pieces_needed (n : ℕ) (h₁ : edge_length n > 0) : 2 * n = 2 * (edge_length n) :=
  by
  sorry

end tape_pieces_needed_l702_702727


namespace increasing_interval_l702_702802

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * x^2

theorem increasing_interval :
  ∃ a b : ℝ, (0 < a) ∧ (a < b) ∧ (b = 1/2) ∧ (∀ x : ℝ, a < x ∧ x < b → (deriv f x > 0)) :=
by
  sorry

end increasing_interval_l702_702802


namespace f_odd_f_inequality_solution_l702_702192

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 ((1 + x) / (1 - x))

theorem f_odd: 
  ∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = - f x := 
by
  sorry

theorem f_inequality_solution:
  { x : ℝ // -1 < x ∧ x < 1 ∧ f x < -1 } = { x : ℝ // -1 < x ∧ x < -1/3 } := 
by 
  sorry

end f_odd_f_inequality_solution_l702_702192


namespace longest_constant_interval_length_l702_702274

theorem longest_constant_interval_length : 
  let f (x : ℝ) := ∑ i in finset.range (2014 + 1), |x - (i : ℝ)| in
  ∃ a b : ℝ, f a = f b ∧ ∀ x ∈ Icc a b, f x = f a ∧ b - a = 1 :=
begin
  sorry
end

end longest_constant_interval_length_l702_702274


namespace perimeter_of_trapezoid_l702_702259

theorem perimeter_of_trapezoid (EF GH FG EH : ℝ) 
  (h_parallel : EF = GH) 
  (distance : ℝ) 
  (EG : ℝ)
  (height : ℝ)
  (h_distance : distance = 5) 
  (h_EG : EG = 20) 
  (h_height : height = 6)
  (h_FG : FG = 15)
  (h_EH : EH = 25) :
  EF = real.sqrt 61 →
  GH = real.sqrt 61 →
  EF + FG + GH + EH = 40 + 2 * real.sqrt 61 := 
by 
  intros h1 h2 
  sorry

end perimeter_of_trapezoid_l702_702259


namespace total_amount_correct_l702_702099

-- Definition of the conditions
def initial_deposit : ℝ := 2000
def annual_interest_rate : ℝ := 0.05
def total_years : ℕ := 6
def factor (n : ℕ) : ℝ := (1 + annual_interest_rate) ^ n
def final_factor : ℝ := factor 7 -- Given that 1.05^7 = 1.41
def total_amount_withdrawn : ℝ := 14400
def deposits : ℕ → ℝ
| 0 => initial_deposit
| n+1 => initial_deposit

-- Proof statement to be constructed
theorem total_amount_correct :
  ∑ i in range total_years, deposits i * factor (total_years - i + 1) = total_amount_withdrawn := by
  sorry

end total_amount_correct_l702_702099


namespace sum_of_numbers_with_7_divisors_l702_702341

-- Definition of a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Condition based on problem constraints
def has_7_divisors (n : ℕ) : Prop :=
  ∃ p, is_prime p ∧ n = p^6

-- Define the set of numbers satisfying the condition
def numbers_with_7_divisors (N : ℕ) : List ℕ :=
  List.filter (λ n, has_7_divisors n) (List.range (N + 1))

-- Lean theorem statement
theorem sum_of_numbers_with_7_divisors (N : ℕ) (h : N = 1000) :
  (numbers_with_7_divisors N).sum = 793 :=
by
  -- Placeholder for the actual proof
  sorry

end sum_of_numbers_with_7_divisors_l702_702341


namespace max_angle_MPN_x_coord_l702_702889

variables {x y : ℝ}
variables {M : ℝ × ℝ} {N : ℝ × ℝ}

-- Define points M and N
def M : ℝ × ℝ := (-1, 2)
def N : ℝ × ℝ := (1, 4)

-- Define the condition that P is on the positive half of the x-axis
def P (x : ℝ) : ℝ × ℝ := if x > 0 then (x, 0) else (0, 0)

theorem max_angle_MPN_x_coord :
  (let E := (x, y),
   let P := P x,
   (x + 1)^2 + (y - 2)^2 = y^2 ∧
   y^2 = (x - 1)^2 + (y - 4)^2) → (x = 1) :=
by {
  sorry
}

end max_angle_MPN_x_coord_l702_702889


namespace possible_values_of_n_l702_702897

-- Define the conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5 * x

def arithmetic_sequence (a_1 d : ℝ) (n : ℕ) : Prop :=
∀ i j : ℕ, 0 < i ∧ i ≤ n ∧ 0 < j ∧ j ≤ n ∧ i ≠ j →
  ∃ k : ℕ, k * d = (a i - a j)

-- Define the problem statement
theorem possible_values_of_n
  (x y : ℝ)
  (inside_circle : circle_eq x y)
  (a_1 d : ℝ)
  (d_real : d ∈ ℝ)
  (n : ℕ)
  (arithmetic_chords : arithmetic_sequence a_1 d n) :
  n ∈ {4, 5, 6, 7} :=
  sorry

end possible_values_of_n_l702_702897


namespace prime_factor_mod4_1_l702_702273

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ 
  a 2 = 2 ∧ 
  ∀ n ≥ 3, a n = 2 * a (n-1) + a (n-2)

theorem prime_factor_mod4_1 {a : ℕ → ℕ} (h : sequence a) :
  ∀ n ≥ 5, ∃ p : ℕ, prime p ∧ p ∣ a n ∧ p % 4 = 1 :=
by
  -- The actual proof would go here
  sorry

end prime_factor_mod4_1_l702_702273


namespace remaining_gift_card_value_correct_l702_702264

def initial_best_buy := 5
def initial_target := 3
def initial_walmart := 7
def initial_amazon := 2

def value_best_buy := 500
def value_target := 250
def value_walmart := 100
def value_amazon := 1000

def sent_best_buy := 1
def sent_walmart := 2
def sent_amazon := 1

def remaining_dollars : Nat :=
  (initial_best_buy - sent_best_buy) * value_best_buy +
  initial_target * value_target +
  (initial_walmart - sent_walmart) * value_walmart +
  (initial_amazon - sent_amazon) * value_amazon

theorem remaining_gift_card_value_correct : remaining_dollars = 4250 :=
  sorry

end remaining_gift_card_value_correct_l702_702264


namespace jill_food_percentage_l702_702940

theorem jill_food_percentage (total_amount : ℝ) (tax_rate_clothing tax_rate_other_items spent_clothing_rate spent_other_rate spent_total_tax_rate : ℝ) : 
  spent_clothing_rate = 0.5 →
  spent_other_rate = 0.25 →
  tax_rate_clothing = 0.1 →
  tax_rate_other_items = 0.2 →
  spent_total_tax_rate = 0.1 →
  (spent_clothing_rate * tax_rate_clothing * total_amount) + (spent_other_rate * tax_rate_other_items * total_amount) = spent_total_tax_rate * total_amount →
  (1 - spent_clothing_rate - spent_other_rate) * total_amount / total_amount = 0.25 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end jill_food_percentage_l702_702940


namespace factor_expression_l702_702457

theorem factor_expression :
  let expr := (20 * x^3 + 100 * x - 10) - (-5 * x^3 + 5 * x - 10)
  expr = 5 * x * (5 * x^2 + 19) :=
by {
  let term1 := 20 * x^3 + 100 * x - 10,
  let term2 := -5 * x^3 + 5 * x - 10,
  have h : expr = term1 - term2,
  sorry
}

end factor_expression_l702_702457


namespace remainder_of_product_l702_702767

theorem remainder_of_product (a b c : ℕ) : 
  (a % 10 = 3) → (b % 10 = 1) → (c % 10 = 5) → 
  ((a * b * c) % 10 = 5) :=
by 
  assume ha : a % 10 = 3
  assume hb : b % 10 = 1
  assume hc : c % 10 = 5
  sorry

end remainder_of_product_l702_702767


namespace exists_divisible_by_2021_l702_702275

def concatenated_number (n m : ℕ) : ℕ := 
  -- This function should concatenate the digits from n to m inclusively
  sorry

theorem exists_divisible_by_2021 : ∃ (n m : ℕ), n > m ∧ m ≥ 1 ∧ 2021 ∣ concatenated_number n m :=
by
  sorry

end exists_divisible_by_2021_l702_702275


namespace polygon_perimeter_even_l702_702750

theorem polygon_perimeter_even (n : ℕ)
    (vertices : Fin n → (ℤ × ℤ))
    (side_length : Fin n → ℕ)
    (hc1 : ∀ i : Fin n, let i' := (i + 1) % n in
                        side_length i = Int.natAbs (fst (vertices i) - fst (vertices i')) +
                                        Int.natAbs (snd (vertices i) - snd (vertices i')))
    (hc2 : ∀ i : Fin n, let i' := (i + 1) % n in
                        (side_length i) % 2 = 0 ∨ (side_length i) % 2 = 1) :
    ∃ k : ℕ, k % 2 = 0 ∧ k = (Finset.univ.sum fun i : Fin n => side_length i) := 
sorry

end polygon_perimeter_even_l702_702750


namespace f_is_odd_no_a_f_not_odd_f_eq_neg_x_all_a_exists_a_f_eq_neg_x_l702_702183

-- Define the function f
def f (x a : ℝ) : ℝ := x + a / x

-- Conclusion ①: Prove that f(x) is odd for all a ∈ ℝ
theorem f_is_odd (a : ℝ) : ∀ x : ℝ, f (-x) a = - f x a := by
  intro x
  -- Proof here
  sorry

-- Conclusion ②: Prove that there does NOT exist an a such that f(x) is not odd
theorem no_a_f_not_odd : ¬ ∃ a : ℝ, ∃ x : ℝ, f (-x) a ≠ - f x a := by
  -- Proof here
  sorry

-- Conclusion ③: Prove that for all a ∈ ℝ, the equation f(x) = -x has real roots (Incorrect)
theorem f_eq_neg_x_all_a (a : ℝ) : ¬ ∀ x : ℝ, f x a = -x → ∃ (x : ℝ), f x a = -x := by
  -- Proof here
  sorry

-- Conclusion ④: Prove that there exists an a such that f(x) = -x has real roots
theorem exists_a_f_eq_neg_x : ∃ a : ℝ, ∃ x : ℝ, f x a = -x := by
  -- Proof here
  sorry

end f_is_odd_no_a_f_not_odd_f_eq_neg_x_all_a_exists_a_f_eq_neg_x_l702_702183


namespace broken_line_length_l702_702370

noncomputable def length_of_broken_line (AB : ℝ) (angle : ℝ) (n : ℕ) : ℝ :=
  if AB = 1 ∧ angle = 45 ∧ n > 0 then
    Real.sqrt 2
  else
    0

theorem broken_line_length (AB : ℝ) (angle : ℝ) (n : ℕ) (h1 : AB = 1) (h2 : angle = 45) (h3 : n > 0) :
  length_of_broken_line AB angle n = Real.sqrt 2 :=
by
  unfold length_of_broken_line
  rw [if_pos ⟨h1, h2, h3⟩]
  rfl

end broken_line_length_l702_702370


namespace average_speed_additional_hours_l702_702407

variables (v : ℝ) -- Define the average speed v for the additional hours

def Distance1 : ℝ := 35 * 4
def AdditionalTime : ℝ := 6 - 4
def TotalDistance : ℝ := 38 * 6 

theorem average_speed_additional_hours : 
  TotalDistance = Distance1 + v * AdditionalTime → 
  v = 44 :=
begin
  sorry
end

end average_speed_additional_hours_l702_702407


namespace actual_income_of_P_is_correct_l702_702980

-- Given conditions
def average_income_PQ (P_i Q_i : ℝ) : ℝ :=
  (0.85 * P_i - 2000 + 0.90 * Q_i - 2500) / 2

def average_income_QR (Q_i R_i : ℝ) : ℝ :=
  (0.90 * Q_i - 2500 + 0.88 * R_i - 3000) / 2

def average_income_PR (P_i R_i : ℝ) : ℝ :=
  (0.85 * P_i - 2000 + 0.88 * R_i - 3000) / 2

axiom avg_PQ : average_income_PQ 7058.82 (Q_i : ℝ) = 5050
axiom avg_QR : average_income_QR (Q_i : ℝ) (R_i : ℝ) = 6250
axiom avg_PR : average_income_PR 7058.82 (R_i : ℝ) = 5200

-- Proof statement
theorem actual_income_of_P_is_correct:
  ∃ P_i Q_i R_i : ℝ, 
  average_income_PQ P_i Q_i = 5050 ∧
  average_income_QR Q_i R_i = 6250 ∧
  average_income_PR P_i R_i = 5200 ∧
  P_i = 7058.82 :=
sorry

end actual_income_of_P_is_correct_l702_702980


namespace rectangle_area_increase_43_75_percent_l702_702640

theorem rectangle_area_increase_43_75_percent (k : ℝ) :
  let l := 3 * k,
      w := 2 * k,
      l_new := 1.25 * l,
      w_new := 1.15 * w,
      A_original := l * w,
      A_new := l_new * w_new,
      Delta_A := A_new - A_original
  in (Delta_A / A_original) * 100 = 43.75 := 
by 
  let l := 3 * k
  let w := 2 * k
  let l_new := 1.25 * l
  let w_new := 1.15 * w
  let A_original := l * w
  let A_new := l_new * w_new
  let Delta_A := A_new - A_original
  sorry

end rectangle_area_increase_43_75_percent_l702_702640


namespace expected_draws_with_digit_8_l702_702394

noncomputable def containsDigit8Count (n : ℕ) : Nat :=
  let tPlace := if (n ≥ 80 && n <= 89) then 10 else 0
  let uPlace := if (n % 10 == 8) then 1 else 0
  tPlace + uPlace

theorem expected_draws_with_digit_8 (n : ℕ) (h : 1 ≤ n ∧ n ≤ 90) :
  let containsDigit8 := containsDigit8Count 18
  let totalDraws := choose 90 5
  let N3 := choose 18 3 * choose 72 2
  let N4 := choose 18 4 * choose 72 1
  let N5 := choose 18 5
  let expectedDraws := (N3 + N4 + N5) / totalDraws
  100 * expectedDraws ≈ 5.3 :=
by 
  sorry

end expected_draws_with_digit_8_l702_702394


namespace x_coordinate_incenter_l702_702054

theorem x_coordinate_incenter (x y : ℝ)
  (h1 : abs y = abs (x + y - 1) / real.sqrt (1^2 + 1^2))
  (h2 : abs x = abs (x + y - 1) / real.sqrt (1^2 + 1^2))
  : x = 1 / 2 :=
sorry

end x_coordinate_incenter_l702_702054


namespace solve_problem_l702_702817

noncomputable def geometric_sequence_formula (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a n = a 0 * q^n

noncomputable def geometric_sequence_conditions_and_arith_mean (a : ℕ → ℝ) (q : ℝ) :=
  q > 1 ∧
  a 1 + a 2 + a 3 = 28 ∧
  2 * (a 2 + 2) = a 1 + a 3

noncomputable def general_formula_is_correct : Prop :=
  ∀ (a : ℕ → ℝ) (q : ℝ),
  geometric_sequence_conditions_and_arith_mean a q →
  (a = nat.succ → 2^(nat.succ) := λ n ≥ a n = 2^n)

noncomputable def sequence_bn (a : ℕ → ℝ) : ℕ → ℝ :=
  λ n, log (sqrt 2) (a n) - 3

noncomputable def sn_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, sequence_bn a i

noncomputable def maximum_positive_integer_exists : Prop :=
  ∀ (a : ℕ → ℝ),
  general_formula_is_correct a 2 →
  ∃ n : ℕ, n < 13 ∧ sn_sum a n < 143

theorem solve_problem :
  general_formula_is_correct ∧ maximum_positive_integer_exists :=
by
  split
  { -- Proof for general_formula_is_correct
    sorry },
  { -- Proof for maximum_positive_integer_exists
    sorry }

end solve_problem_l702_702817


namespace part1_part2_l702_702149

variable {α : Type*} [linear_ordered_field α]

-- Part (1)
theorem part1 (a_n: ℕ → α) (k : α) (h_pos : ∀ n, 0 < a_n n)
  (h_rec : ∀ n, a_n (n+1)^2 = a_n n * a_n (n+2) + k)
  (h_k_zero : k = 0)
  (h_arith : 8 * a_n 2 + a_n 4 = 6 * a_n 3) :
  ∃ q, q > 0 ∧ a_n 2 / a_n 1 = q :=
sorry

-- Part (2)
theorem part2 (a_n : ℕ → α) (k a b : α) (h_pos : ∀ n, 0 < a_n n)
  (h_rec : ∀ n, a_n (n+1)^2 = a_n n * a_n (n+2) + k)
  (h_init : a_n 1 = a ∧ a_n 2 = b) :
  ∃ λ, ∀ n, a_n n + a_n (n+2) = λ * a_n (n+1) ∧ λ = (a^2 + b^2 - k) / (a * b) :=
sorry

end part1_part2_l702_702149


namespace b_n_is_geometric_a_n_b_n_relation_l702_702847

-- Define the sequences a_n and b_n
variables {α : Type*} [field α]
variables (a b : ℕ → α)

-- Given conditions: Arithmetic sequence a_n and the equation
axiom arith_seq {a : ℕ → α} {a1 : α} (h : ∀ n, a n = n * a1)

-- The main equation given in the problem
axiom main_eq (h : ∀ n, 0 < n → a 1 * b n + ∑ i in finset.range (n - 1), a (i + 2) * b (n - i - 1) = 2 ^ (n + 1) - n - 2)

-- Problem (1): Prove that b_n is a geometric sequence
theorem b_n_is_geometric (a1 : α)
  (h_arith : ∀ n, a n = n * a1)
  (h_main : ∀ n, 0 < n → a 1 * b n + ∑ i in finset.range (n - 1), a (i + 2) * b (n - i - 1) = 2 ^ (n + 1) - n - 2) :
  ∃ q : α, ∀ n, 0 < n → b (n + 1) = q * b n :=
begin
  sorry
end

-- Problem (2): Prove that a_n * b_n = n * 2 ^ (n - 1)
theorem a_n_b_n_relation (a1 b1 : α)
  (h_arith : ∀ n, a n = n * a1)
  (h_geo : ∃ q : α, ∀ n, 0 < n → b (n + 1) = q * b n)
  (h_main : ∀ n, 0 < n → a 1 * b n + ∑ i in finset.range (n - 1), a (i + 2) * b (n - i - 1) = 2 ^ (n + 1) - n - 2) :
  ∀ n, 0 < n → a n * b n = n * 2 ^ (n - 1) :=
begin
  sorry
end

end b_n_is_geometric_a_n_b_n_relation_l702_702847


namespace proof_f_f_2008_eq_2008_l702_702462

-- Define the function f
axiom f : ℝ → ℝ

-- The conditions given in the problem
axiom odd_f : ∀ x, f (-x) = -f x
axiom periodic_f : ∀ x, f (x + 6) = f x
axiom f_at_4 : f 4 = -2008

-- The goal to prove
theorem proof_f_f_2008_eq_2008 : f (f 2008) = 2008 :=
by
  sorry

end proof_f_f_2008_eq_2008_l702_702462


namespace domain_f_x_plus_1_f_x_minus_1_l702_702547

theorem domain_f_x_plus_1_f_x_minus_1 {f : ℝ → ℝ} (h : ∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, y = f x) :
  (∀ x, (∀ y, y = f (x+1) ∨ y = f (x-1)) ↔ x = 1) := 
begin
  sorry
end

end domain_f_x_plus_1_f_x_minus_1_l702_702547


namespace bells_toll_together_in_9272_seconds_l702_702076

def intervals : List ℕ := [13, 17, 21, 26, 34, 39]

theorem bells_toll_together_in_9272_seconds :
  intervals.lcm = 9272 := 
sorry

end bells_toll_together_in_9272_seconds_l702_702076


namespace flooring_sq_ft_per_box_l702_702670

/-- The problem statement converted into a Lean theorem -/
theorem flooring_sq_ft_per_box
  (living_room_length : ℕ)
  (living_room_width : ℕ)
  (flooring_installed : ℕ)
  (additional_boxes : ℕ)
  (correct_answer : ℕ) 
  (h1 : living_room_length = 16)
  (h2 : living_room_width = 20)
  (h3 : flooring_installed = 250)
  (h4 : additional_boxes = 7)
  (h5 : correct_answer = 10) :
  
  (living_room_length * living_room_width - flooring_installed) / additional_boxes = correct_answer :=
by 
  sorry

end flooring_sq_ft_per_box_l702_702670


namespace equal_real_roots_real_roots_l702_702520

noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b * b - 4 * a * c

theorem equal_real_roots (m : ℝ) :
  let eq := 2 * (m + 1) * x^2 + 4 * m * x + 3 * m - 2 = 0;
  (quadratic_discriminant (2 * (m + 1)) (4 * m) (3 * m - 2) = 0) ↔ (m = -2 ∨ m = 1) := 
by 
  sorry

theorem real_roots (m : ℝ) (x : ℝ) :
  let eq := 2 * (m + 1) * x^2 + 4 * m * x + 3 * m - 2 = 0;
  eq →
  (m = -1 → x = -5 / 4) ∧ 
  (m ≠ -1 → x = (-2 * m + sqrt (-2 * m^2 - 2 * m + 4)) / (2 * (m + 1)) ∨
            x = (-2 * m - sqrt (-2 * m^2 - 2 * m + 4)) / (2 * (m+1))) := 
by 
  sorry

end equal_real_roots_real_roots_l702_702520


namespace sin_angle_GAD_correct_l702_702711

noncomputable def sin_angle_GAD (s : ℝ) :=
  let A : EuclideanSpace ℝ (Fin 3) := ⟨0, 0, 0⟩
  let D : EuclideanSpace ℝ (Fin 3) := ⟨1, 0, 0⟩
  let G : EuclideanSpace ℝ (Fin 3) := ⟨1, 0, 1⟩
  let AG := dist A G
  let DG := dist D G
  Real.sin (angle (AG -ᵥ A) (G -ᵥ A)) := (DG / AG)

theorem sin_angle_GAD_correct :
  sin_angle_GAD 1 = (Real.sqrt 3) / 3 := 
sorry

end sin_angle_GAD_correct_l702_702711


namespace total_tourists_904_l702_702044

theorem total_tourists_904 :
  let trips := [120, 118, 116, 114, 112, 110, 108, 106] in
  let total_tourists := trips.sum in
  total_tourists = 904 := by
  let trips := [120, 118, 116, 114, 112, 110, 108, 106]
  let total_tourists := trips.sum
  show total_tourists = 904
  sorry

end total_tourists_904_l702_702044


namespace mark_height_feet_l702_702614

theorem mark_height_feet
  (mark_height_inches : ℕ)
  (mike_height_feet : ℕ)
  (mike_height_inches : ℕ)
  (mike_taller_than_mark : ℕ)
  (foot_in_inches : ℕ)
  (mark_height_eq : mark_height_inches = 3)
  (mike_height_eq : mike_height_feet * foot_in_inches + mike_height_inches = 73)
  (mike_taller_eq : mike_height_feet * foot_in_inches + mike_height_inches = mark_height_inches + mike_taller_than_mark)
  (foot_in_inches_eq : foot_in_inches = 12) :
  mark_height_inches = 63 ∧ mark_height_inches / foot_in_inches = 5 := by
sorry

end mark_height_feet_l702_702614


namespace exists_point_viewing_all_sides_at_angle_l702_702155

theorem exists_point_viewing_all_sides_at_angle (A B C : Type) [Nonempty A] [Nonempty B] [Nonempty C]
  (angleA angleB angleC φ : ℝ)
  (h1 : angleA < φ) (h2 : angleB < φ) (h3 : angleC < φ) (hφ : φ < 2 * π / 3) :
  ∃ (P : Type), ∀ (side : Type), (side ∈ {A, B, C} ) → measurable_from_point P side φ :=
begin
  sorry
end

end exists_point_viewing_all_sides_at_angle_l702_702155


namespace no_adjacent_birch_or_maple_probability_l702_702734

open Nat

theorem no_adjacent_birch_or_maple_probability :
  let total_trees := 4 + 5 + 6
  let m_trees := 4
  let o_trees := 5
  let b_trees := 6
  let total_arrangements := fact total_trees / (fact m_trees * fact o_trees * fact b_trees)
  let valid_arrangements := 35
  let probability := valid_arrangements / total_arrangements
  let simplified_probability := probability.num / probability.denom
  simplified_probability = (7, 166320) → 7 + 166320 = 166327 := by
  sorry

end no_adjacent_birch_or_maple_probability_l702_702734


namespace angles_in_football_l702_702677

-- Assumptions
variable {edge_len : ℝ} (h1 : edge_len = 1) 

-- Definitions for the football structure
structure Football :=
  (regular_pentagon : Type)
  (regular_hexagon : Type)
  (unit_edge_length : ∀ (p : regular_pentagon) 
                       (h : regular_hexagon), edge_len = 1)
  (surround_pentagon_with_hexagons : ∀ (p : regular_pentagon),
      (∃ hexagons : list regular_hexagon, hexagons.length = 5 ∧ 
                                                ∀ h ∈ hexagons, convex (h ∪ p)))

-- Definitions for the angles
def angle_between_hexagons (F : Football) : ℝ := sorry
def angle_hexagon_pentagon (F : Football) : ℝ := sorry

-- Proof statement
theorem angles_in_football (F : Football) :
  ∃ α β : ℝ,
  α = angle_between_hexagons F ∧ β = angle_hexagon_pentagon F :=
sorry

end angles_in_football_l702_702677


namespace greatest_increase_year_l702_702778

open List

def annual_revenue : List (ℕ × ℝ) := [(2005, 2), (2006, 2.4), (2007, 3), (2008, 3.25), (2009, 5.5),
                                       (2010, 5.75), (2011, 5.8), (2012, 6), (2013, 5.4), (2014, 3.5)]

def revenue_changes (revenues : List (ℕ × ℝ)) : List (ℕ × ℝ) :=
  revenues.tail.zip revenues |>.map (λ ⟨(y1, r1), (y2, r2)⟩ => (y2, r2 - r1))

def max_revenue_change_year (changes : List (ℕ × ℝ)) : ℕ :=
  changes.maximumBy (λ ⟨_, delta⟩ => delta) |>.map Prod.fst |>.getD 2005

theorem greatest_increase_year :
  max_revenue_change_year (revenue_changes annual_revenue) = 2009 :=
  sorry

end greatest_increase_year_l702_702778


namespace necessary_condition_for_inequality_not_sufficient_condition_for_inequality_l702_702391

theorem necessary_condition_for_inequality (x : ℝ) :
  (|x - 1| < 2) → (x(x-3) < 0) :=
by sorry

theorem not_sufficient_condition_for_inequality (x : ℝ) :
  ¬(|x - 1| < 2) ∨ ¬(x(x-3) < 0) :=
by sorry

end necessary_condition_for_inequality_not_sufficient_condition_for_inequality_l702_702391


namespace relationship_among_abcdef_l702_702168

variables (a b c d : ℝ)

def log_base := 0.3
def a_def := Real.log 2 / Real.log log_base
def b_def := Real.log 3 / Real.log log_base
def c_def := Real.exp (0.3 * Real.log 2)
def d_def := log_base ^ 2

theorem relationship_among_abcdef (log_base_ne_zero : log_base ≠ 0) (log_base_lt_one : log_base < 1) :
  b < a ∧ a < 0 ∧ 0 < d ∧ d < 1 ∧ 1 < c ∧ b < d ∧ a < d := 
by
  rw [a_def, b_def, c_def, d_def]
  sorry

end relationship_among_abcdef_l702_702168


namespace geometric_series_sum_l702_702448

theorem geometric_series_sum :
  ∑ i in finset.range 7, (1 / 2 : ℚ) ^ (i+1) = 127 / 128 :=
by
  sorry

end geometric_series_sum_l702_702448


namespace four_digit_greater_than_three_digit_l702_702733

theorem four_digit_greater_than_three_digit (n m : ℕ) (h₁ : 1000 ≤ n ∧ n ≤ 9999) (h₂ : 100 ≤ m ∧ m ≤ 999) : n > m :=
sorry

end four_digit_greater_than_three_digit_l702_702733


namespace compare_y1_y2_l702_702504

noncomputable def quadratic (x : ℝ) : ℝ := -x^2 + 2

theorem compare_y1_y2 :
  let y1 := quadratic 1
  let y2 := quadratic 3
  y1 > y2 :=
by
  let y1 := quadratic 1
  let y2 := quadratic 3
  sorry

end compare_y1_y2_l702_702504


namespace g_3_2_plus_g_3_5_l702_702904

def g (x y : ℚ) : ℚ :=
  if x + y ≤ 5 then (x * y - x + 3) / (3 * x) else (x * y - y - 3) / (-3 * y)

theorem g_3_2_plus_g_3_5 : g 3 2 + g 3 5 = 1/5 := by
  sorry

end g_3_2_plus_g_3_5_l702_702904


namespace part1_profit_150_part2_max_profit_l702_702432

noncomputable def profit_function (x : ℝ) : ℝ :=
  (x - 20) * (-2 * x + 80)

theorem part1_profit_150 (x : ℝ) :
  ((x - 20) * (-2 * x + 80) = 150) → x = 25 := sorry

theorem part2_max_profit (x : ℝ) :
  (∀ y : ℝ, y ∈ set.Icc 20 28 → profit_function y ≤ profit_function x) → x = 28 ∧ profit_function x = 192 := sorry

end part1_profit_150_part2_max_profit_l702_702432


namespace min_n_value_l702_702922

def S : Set ℕ := {1, 2, 3, 4}

def has_property (seq : List ℕ) : Prop :=
  ∀ B ⊆ S, B.Nonempty → ∃ (i : ℕ), list.take (B.card) (list.drop i seq) = B

theorem min_n_value : ∃ (n : ℕ), (∀ seq : List ℕ, seq.length = n → has_property seq) ∧ n = 8 := by
  sorry

end min_n_value_l702_702922


namespace area_of_section_distance_from_D_to_plane_angle_between_SD_and_plane_l702_702979

-- Define the given conditions
def apothem (SA : ℝ) : Prop := SA = 2
def angle_with_base (θ : ℝ) : Prop := θ = Real.arctan (Real.sqrt 2)
def point_ratios (AE EB AF FD SK KC : ℝ) : Prop :=
(AE / EB = 1 / 2) ∧ (AF / FD = 1 / 2) ∧ (SK / KC = 1 / 2)

-- Define the questions as proofs to be resolved
theorem area_of_section {SA θ AE EB AF FD SK KC : ℝ}
  (h₁ : apothem SA) (h₂ : angle_with_base θ) (h₃ : point_ratios AE EB AF FD SK KC) :
  let area := (14/9) * Real.sqrt (5/3) in
  area = (14/9) * Real.sqrt (5/3) := 
sorry

theorem distance_from_D_to_plane {SA θ AE EB AF FD SK KC : ℝ}
  (h₁ : apothem SA) (h₂ : angle_with_base θ) (h₃ : point_ratios AE EB AF FD SK KC) :
  let distance := 4 / (3 * Real.sqrt 5) in
  distance = 4 / (3 * Real.sqrt 5) :=
sorry

theorem angle_between_SD_and_plane {SA θ AE EB AF FD SK KC : ℝ}
  (h₁ : apothem SA) (h₂ : angle_with_base θ) (h₃ : point_ratios AE EB AF FD SK KC) :
  let angle := Real.arcsin (3 / 5) in
  angle = Real.arcsin (3 / 5) :=
sorry

end area_of_section_distance_from_D_to_plane_angle_between_SD_and_plane_l702_702979


namespace proof_inequalities_l702_702809

variables {E : Type*} [linear_ordered_field E] 
variables (a b c d : E)

theorem proof_inequalities :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 ∧
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a :=
by sorry

end proof_inequalities_l702_702809


namespace orthogonal_vectors_l702_702787

theorem orthogonal_vectors:
  (in_list : list ℝ) (in_list = [(-1:ℝ), x, 2]) → 
  x : ℝ,
  (3 * (-1) + (-1) * x + 4 * 2) = 0 → x = 5 := by
sorry

end orthogonal_vectors_l702_702787


namespace multiply_fractions_l702_702765

theorem multiply_fractions :
  (2 : ℚ) / 3 * 5 / 7 * 11 / 13 = 110 / 273 :=
by
  sorry

end multiply_fractions_l702_702765


namespace range_of_a_l702_702549

theorem range_of_a (a : ℝ) :
  (∃ x₀ : ℝ, 0 < x₀ ∧ ax₀ - log x₀ < 0) → a < 1 / Real.exp 1 :=
sorry

end range_of_a_l702_702549


namespace number_of_days_B_find_daily_costs_fastest_completion_time_l702_702364

-- Define constants and conditions for the problem
variables (project_part_A project_part_total : ℕ)
def conditions_part1 := 
  (project_part_A : ℚ := 1 / 3) ∧ (days_A : ℕ := 30) ∧ (days_work_together : ℕ := 15) ∧ (days_B : ℕ := 30)

theorem number_of_days_B (project_part_B : ℚ) :
  ∃ (days_B : ℕ), (30 * (1 / 90 + 1 / days_B)) + project_part_B = 1 :=
sorry

-- Define variables for part 2
variables (cost_A cost_B total_cost_1 total_cost_2 : ℕ)

def conditions_part2 := 
  (4 * cost_A + 3 * cost_B = 42000) ∧ (5 * cost_A + 6 * cost_B = 75000)

theorem find_daily_costs :
  ∃ (cost_A cost_B : ℕ), 4 * cost_A + 3 * cost_B = 42000 ∧ 5 * cost_A + 6 * cost_B = 75000 :=
sorry

-- Define variables for part 3
variables (a b : ℕ)
def conditions_part3 :=
  (project_completion : ℚ := a / 90 + b / 30 = 1) ∧ (total_labor_cost : ℕ := 3000 * a + 10000 * b ≤ 280000)

theorem fastest_completion_time (total_days : ℕ) :
  ∃ (a b total_days : ℕ), (total_days = a + b) ∧ (a / 90 + b / 30 = 1) ∧ (3000 * a + 10000 * b ≤ 280000) ∧ total_days ≤ 70 :=
sorry

end number_of_days_B_find_daily_costs_fastest_completion_time_l702_702364


namespace Nellie_legos_l702_702300

def initial_legos : ℕ := 380
def lost_legos : ℕ := 57
def given_legos : ℕ := 24

def remaining_legos : ℕ := initial_legos - lost_legos - given_legos

theorem Nellie_legos : remaining_legos = 299 := by
  sorry

end Nellie_legos_l702_702300


namespace distance_from_A_to_plane_l702_702223

def vec := ℝ × ℝ × ℝ

def normal_vector : vec := (1, 2, 2)
def point_A : vec := (1, 0, 2)
def point_B : vec := (0, -1, 4)
def point_A_notin_plane (A : vec) (n : vec) (B : vec) : Prop :=
  let n_x, n_y, n_z := n
  let B_x, B_y, B_z := B
  let A_x, A_y, A_z := A
  (A_x * n_x + A_y * n_y + A_z * n_z) ≠ (B_x * n_x + B_y * n_y + B_z * n_z)
def point_B_in_plane (B : vec) (n : vec) : Prop :=
  let n_x, n_y, n_z := n
  let B_x, B_y, B_z := B
  (B_x * n_x + B_y * n_y + B_z * n_z) = 0

noncomputable def distance_point_to_plane (A B n : vec) : ℝ :=
  let (a_x, a_y, a_z) := A
  let (b_x, b_y, b_z) := B
  let (n_x, n_y, n_z) := n
  let ba := (a_x - b_x, a_y - b_y, a_z - b_z)
  let (ba_x, ba_y, ba_z) := ba
  float.abs(ba_x * n_x + ba_y * n_y + ba_z * n_z) / real.sqrt (n_x ^ 2 + n_y ^ 2 + n_z ^ 2)

theorem distance_from_A_to_plane (A : vec) (B : vec) (n : vec) 
  (hA : point_A_notin_plane A n B) 
  (hB : point_B_in_plane B n) :
  distance_point_to_plane A B n = 1 / 3 :=
by
  sorry

end distance_from_A_to_plane_l702_702223


namespace brooke_added_balloons_l702_702446

-- Definitions stemming from the conditions
def initial_balloons_brooke : Nat := 12
def added_balloons_brooke (x : Nat) : Nat := x
def initial_balloons_tracy : Nat := 6
def added_balloons_tracy : Nat := 24
def total_balloons_tracy : Nat := initial_balloons_tracy + added_balloons_tracy
def final_balloons_tracy : Nat := total_balloons_tracy / 2
def total_balloons (x : Nat) : Nat := initial_balloons_brooke + added_balloons_brooke x + final_balloons_tracy

-- Mathematical proof problem
theorem brooke_added_balloons (x : Nat) :
  total_balloons x = 35 → x = 8 := by
  sorry

end brooke_added_balloons_l702_702446


namespace find_polynomials_l702_702102

noncomputable def satisfies_condition (f : ℤ[X]) : Prop :=
  ∀ (p : ℕ), p.prime → p % 2 = 1 → f.eval (p : ℤ) ∣ ((p - 3)! + (p + 1) / 2 : ℤ)

theorem find_polynomials :
  ∀ f : ℤ[X], satisfies_condition f →
    (f = 1)
  ∨ (f = -1)
  ∨ (∃ (c : ℤ), c = 1 ∨ c = -1 ∧ f = C c * X) :=
by
  intros
  sorry

end find_polynomials_l702_702102


namespace factor_expression_l702_702458

theorem factor_expression :
  let expr := (20 * x^3 + 100 * x - 10) - (-5 * x^3 + 5 * x - 10)
  expr = 5 * x * (5 * x^2 + 19) :=
by {
  let term1 := 20 * x^3 + 100 * x - 10,
  let term2 := -5 * x^3 + 5 * x - 10,
  have h : expr = term1 - term2,
  sorry
}

end factor_expression_l702_702458


namespace red_ballpoint_pens_count_l702_702034

theorem red_ballpoint_pens_count (R B : ℕ) (h1: R + B = 240) (h2: B = R - 2) : R = 121 :=
by
  sorry

end red_ballpoint_pens_count_l702_702034


namespace Lori_has_300_beanie_babies_l702_702295

def beanie_babies (S L : ℕ) : Prop :=
  (L = 15 * S) ∧ (S + 15 * S = 320)

theorem Lori_has_300_beanie_babies (S L : ℕ) (h : beanie_babies S L) : L = 300 :=
by
  cases h with h1 h2
  have : 16 * S = 320 := by linarith
  have S_eq : S = 20 := by linarith
  rw [S_eq] at h1
  linarith

end Lori_has_300_beanie_babies_l702_702295


namespace minimize_expression_l702_702593

theorem minimize_expression :
  (∀ (a b : ℝ), (2^(a + b) + 8) * (3^a + 3^b) ≤ v * (12^(a - 1) + 12^(b - 1) - 2^(a + b - 1)) + w) →
  ∃ (v w : ℝ), 128 * v^2 + w^2 = 62208 :=
begin
  sorry
end

end minimize_expression_l702_702593


namespace CB_eq_MN_l702_702589

open EuclideanGeometry
variable {A B C N M : Point}

-- Conditions
def condition1 (A : Point) (angle_A : Angle) : Prop := angle_A = 60
def condition2 (A B N : Point) : Prop := N ∈ perpendicular_bisector A B
def condition3 (A C M : Point) : Prop := M ∈ perpendicular_bisector A C

-- Main Statement
theorem CB_eq_MN (ABC : Triangle) (h1: condition1 A (Angle A B C))
(h2: condition2 A B N) (h3: condition3 A C M) : dist C B = dist M N := by
  sorry

end CB_eq_MN_l702_702589


namespace triangle_ABC_area_l702_702671

theorem triangle_ABC_area : 
  ∃ (A B C : (ℝ × ℝ)),
  (A = (0, 1)) ∧ (B = (4, 0)) ∧ (C = (2, 5)) ∧ 
  is_rectangle ((0, 0)) ((4, 0)) ((4, 5)) ((0, 5)) ∧ 
  area_triangle A B C = 9 := 
by
  sorry

end triangle_ABC_area_l702_702671


namespace steps_to_one_seventh_remain_l702_702063

-- Conditions
def fraction_remaining_after_steps (n : ℕ) : ℚ :=
  (finset.range n).fold (λ acc k, acc * ((k+1)/(k+2):ℚ)) (2/3:ℚ)

theorem steps_to_one_seventh_remain : fraction_remaining_after_steps 12 = (1/7:ℚ) :=
sorry

end steps_to_one_seventh_remain_l702_702063


namespace constant_term_in_expansion_l702_702645

-- Define the binomial expansion general term
def binomial_general_term (x : ℤ) (r : ℕ) : ℤ :=
  (-2)^r * 3^(5 - r) * (Nat.choose 5 r) * x^(10 - 5 * r)

-- Define the condition for the specific r that makes the exponent of x zero
def condition (r : ℕ) : Prop :=
  10 - 5 * r = 0

-- Define the constant term calculation
def const_term : ℤ :=
  4 * 27 * (Nat.choose 5 2)

-- Theorem statement
theorem constant_term_in_expansion : const_term = 1080 :=
by 
  -- The proof is omitted
  sorry

end constant_term_in_expansion_l702_702645


namespace cupcakes_left_over_l702_702950

def total_cupcakes := 40
def ms_delmont_class := 18
def mrs_donnelly_class := 16
def ms_delmont := 1
def mrs_donnelly := 1
def school_nurse := 1
def school_principal := 1

def total_given_away := ms_delmont_class + mrs_donnelly_class + ms_delmont + mrs_donnelly + school_nurse + school_principal

theorem cupcakes_left_over : total_cupcakes - total_given_away = 2 := by
  sorry

end cupcakes_left_over_l702_702950


namespace eta_expectation_and_variance_l702_702842

noncomputable def ξ : Type := sorry

def η := 5 * ξ

theorem eta_expectation_and_variance :
  E(η) = 25 / 2 ∧ D(η) = 125 / 4 := by
sorry

end eta_expectation_and_variance_l702_702842


namespace find_t_l702_702895

noncomputable def a_n (n : ℕ) : ℝ := 1 * (2 : ℝ)^(n-1)

noncomputable def S_3n (n : ℕ) : ℝ := (1 - (2 : ℝ)^(3 * n)) / (1 - 2)

noncomputable def a_n_cubed (n : ℕ) : ℝ := (a_n n)^3

noncomputable def T_n (n : ℕ) : ℝ := (1 - (a_n_cubed 2)^n) / (1 - (a_n_cubed 2))

theorem find_t (n : ℕ) : S_3n n = 7 * T_n n :=
by
  sorry

end find_t_l702_702895


namespace exchange_doubloons_increase_l702_702023

theorem exchange_doubloons_increase (s d : ℝ) (hs : s > 0) :
  let pistoles := Real.round (d / s) in
  let final_doubloons := Real.round (pistoles * s) in
  (final_doubloons > d) →
  ∃ d > 0, ∃ s > 0, s ≠ 1 ∧
  let pistoles := Real.round (d / s) in
  let final_doubloons := Real.round (pistoles * s) in
  final_doubloons > d :=
sorry

end exchange_doubloons_increase_l702_702023


namespace sum_of_reciprocals_lt_2_5_l702_702144

open Nat

theorem sum_of_reciprocals_lt_2_5 :
  ∀ (n : ℕ) (a : ℕ → ℕ), n = 38 → (∀ i, i < n → ∃ k, a i ∣ 10^k) →
    (∑ i in Finset.range n, (1 : ℚ) / a i) < 2.5 :=
by
  sorry

end sum_of_reciprocals_lt_2_5_l702_702144


namespace cos_inequality_part1_cos_inequality_part2_l702_702310

theorem cos_inequality_part1 (x : ℝ) (h : 0 ≤ x ∧ x ≤ π / 2) : 
  1 - cos x ≤ (x^2) / 2 := 
by 
  sorry

theorem cos_inequality_part2 (x : ℝ) (h : 0 ≤ x ∧ x ≤ π / 2) : 
  x * cos x ≤ sin x ∧ sin x ≤ x * cos (x / 2) := 
by 
  sorry

end cos_inequality_part1_cos_inequality_part2_l702_702310


namespace unique_root_k_values_l702_702178

theorem unique_root_k_values (k : ℝ) :
  (∃ x : ℝ, (kx^2 - 8x + 16 = 0) ∧ ∀ y : ℝ, (kx^2 - 8x + 16 = 0) → y = x) ↔ (k = 0 ∨ k = 1) :=
by
  sorry

end unique_root_k_values_l702_702178


namespace triangle_area_three_times_l702_702154

variables {A B C D E F G H K : Type*} [metric_space A] [inner_product_space ℝ A]
variables (triangle_ABC : triangle A B C)
variables (square_ABDE : square A B D E) (square_CAFG : square C A F G) (square_BCHK : square B C H K)
variables (EF GH KD : line_segment)

theorem triangle_area_three_times (h1 : square_on_side AB square_ABDE)
                                  (h2 : square_on_side CA square_CAFG)
                                  (h3 : square_on_side BC square_BCHK)
                                  (h4 : forms_triangle EF GH KD) :
    area (triangle EF GH KD) = 3 * area (triangle ABC) :=
sorry

end triangle_area_three_times_l702_702154


namespace sum_of_smallest_prime_factors_of_360_l702_702378

theorem sum_of_smallest_prime_factors_of_360 : 
  (let n := 360 in 
    let prime_factors := [2, 3, 5] in  -- List of distinct prime factors
    prime_factors.nth_le 0 (by decide) + prime_factors.nth_le 1 (by decide) = 5) := 
by
  -- Establish hypothesis n = 360
  let n := 360
  -- Establish the list of prime factors of n (360)
  let prime_factors := [2, 3, 5]
  -- Sum the two smallest factors (index 0 and index 1) and assert equality with 5
  show prime_factors.nth_le 0 (by decide) + prime_factors.nth_le 1 (by decide) = 5
  exact calc
    prime_factors.nth_le 0 (by decide) + prime_factors.nth_le 1 (by decide)
          = 2 + 3 : by simp
    ... = 5 : by norm_num

end sum_of_smallest_prime_factors_of_360_l702_702378


namespace bus_tour_total_sales_l702_702421

noncomputable def total_sales (total_tickets sold_senior_tickets : Nat) (cost_senior_ticket cost_regular_ticket : Nat) : Nat :=
  let sold_regular_tickets := total_tickets - sold_senior_tickets
  let sales_senior := sold_senior_tickets * cost_senior_ticket
  let sales_regular := sold_regular_tickets * cost_regular_ticket
  sales_senior + sales_regular

theorem bus_tour_total_sales :
  total_sales 65 24 10 15 = 855 := by
    sorry

end bus_tour_total_sales_l702_702421


namespace compute_fraction_when_x_is_3_l702_702086

theorem compute_fraction_when_x_is_3 :
  let x := 3 in
  (x^8 + 18 * x^4 + 81) / (x^4 + 9) = 90 := 
by
  let x := 3
  sorry

end compute_fraction_when_x_is_3_l702_702086


namespace problem_l702_702288

open Real

theorem problem (x y : ℝ) (hx : 1 < x) (hy : 1 < y) 
  (h : (log 2 x)^3 + (log 3 y)^3 + 9 = 9 * (log 2 x) * (log 3 y)) :
  x^(√3) + y^(√3) = 35 := 
by sorry

end problem_l702_702288


namespace equal_sides_of_equilateral_convex_ngon_l702_702775

theorem equal_sides_of_equilateral_convex_ngon {n : ℕ} (h_n : n ≥ 3)
  (convex_ngon : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → angle A_i = angle A_(i+1))
  (side_condition : ∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ n → a_i ≥ a_j) :
  ∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → a_i = a_j := 
sorry

end equal_sides_of_equilateral_convex_ngon_l702_702775


namespace roots_triple_relation_l702_702786

theorem roots_triple_relation (a b c : ℤ) (α β : ℤ)
    (h_quad : a ≠ 0)
    (h_roots : α + β = -b / a)
    (h_prod : α * β = c / a)
    (h_triple : β = 3 * α) :
    3 * b^2 = 16 * a * c :=
sorry

end roots_triple_relation_l702_702786


namespace find_a1_l702_702150

theorem find_a1
  (a : ℕ → ℝ)
  (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 1)
  (h_rec : ∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7} → a (n + 1) = a n * (n / (n + 2))) :
  a 1 = 9 / 16 :=
by
  sorry

end find_a1_l702_702150


namespace centers_not_marked_points_l702_702635

-- Points marked on the plane
def marked_points : Type := Set Point

-- Not all points lie on a single straight line
def not_collinear (points : marked_points) : Prop :=
  ∃ p₁ p₂ p₃ ∈ points, ¬ collinear p₁ p₂ p₃

-- A circle is circumscribed around each triangle with vertices at the marked points
def circumscribed_circles (points : marked_points) : ∀ (p₁ p₂ p₃ ∈ points), ∃ (O : Point), circle_circumscribed_around O p₁ p₂ p₃

-- Centers of all these circumcircles cannot be the marked points themselves
theorem centers_not_marked_points (points : marked_points) :
  not_collinear points → (∀ (p₁ p₂ p₃ ∈ points), ∃ (O : Point), circle_circumscribed_around O p₁ p₂ p₃) → (∃ O ∈ points, (circle_center O points) = false) :=
by
  sorry

end centers_not_marked_points_l702_702635


namespace rhombus_intersection_FD_length_l702_702607

theorem rhombus_intersection_FD_length :
  ∀ (A B C D E F : Type) [metric_space A] [metric_space B] [metric_space C]
  [metric_space D] [metric_space E] [metric_space F]
  (d_AB : dist A B = 10)
  (d_AD : dist A D = 10)
  (d_AC : dist A C = 16)
  (angle_ABC : ∠A B C = 60)
  (d_DE : dist D E = 6)
  (BE_intersects_AD_at_F : line_through B E ∩ line_through A D = {F}),
  dist F D = 3.75 := sorry

end rhombus_intersection_FD_length_l702_702607


namespace subtracted_number_correct_sum_l702_702956

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldr (· + ·) 0

theorem subtracted_number_correct_sum :
  let x := 10^37 + 3 * 10^36
  sum_digits (10^38 - x) = 330 :=
by {
  let x := 10^37 + 3 * 10^36,
  have : sum_digits (10^38 - x) = 330,
  {
    sorry,
  },
  exact this,
}

end subtracted_number_correct_sum_l702_702956


namespace four_digit_integers_count_l702_702853

def first_two_digit_options : Finset ℕ := {2, 3, 6}
def last_two_digit_options_set1 : Finset ℕ := {5, 7, 8}
def last_two_digit_options_set2 : Finset ℕ := {0, 1, 9}

noncomputable def count_valid_4_digit_integers : ℕ :=
  let valid_first_two_digits := (first_two_digit_options.product first_two_digit_options).filter (λ p, true) -- 9 options
  let valid_last_two_digits_set1 := (last_two_digit_options_set1.product (last_two_digit_options_set1.filter (≠ p.1))).filter (λ p, true) -- 6 options
  let valid_last_two_digits_set2 := (last_two_digit_options_set2.product (last_two_digit_options_set2.filter (≠ p.1))).filter (λ p, true) -- 6 options
  let valid_last_two_digits := valid_last_two_digits_set1 ∪ valid_last_two_digits_set2 -- 12 options
  (valid_first_two_digits.product valid_last_two_digits).filter (λ p, (p.1.1 + p.1.2 + p.2.1 + p.2.2) % 3 = 0).card -- 36 valid numbers
  
theorem four_digit_integers_count : count_valid_4_digit_integers = 36 := by
  sorry

end four_digit_integers_count_l702_702853


namespace tangential_quadrilateral_conditions_l702_702647

variables {A B C D K L : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace K] [MetricSpace L]
variables (AB : ℝ) (BC : ℝ) (CD : ℝ) (AD : ℝ)
variables (BK : ℝ) (BL : ℝ) (DK : ℝ) (DL : ℝ)
variables (AK : ℝ) (CL : ℝ) (AL : ℝ) (CK : ℝ)
variables (K_intersection : ∃ (K : Type), ∃ (H1 : MetricSpace.inter A B K) (H2 : MetricSpace.inter C D K), true)
variables (L_intersection : ∃ (L : Type), ∃ (H1 : MetricSpace.inter A D L) (H2 : MetricSpace.inter B C L), true)
variables (segments_intersect : ∃ (P : Type), ∃ (H1 : MetricSpace.inter B L P) (H2 : MetricSpace.inter D K P), true)

theorem tangential_quadrilateral_conditions :
  (AB + CD = BC + AD ∨ BK + BL = DK + DL ∨ AK + CL = AL + CK) → 
  (AB + CD = BC + AD ∧ BK + BL = DK + DL ∧ AK + CL = AL + CK) :=
by
  sorry

end tangential_quadrilateral_conditions_l702_702647


namespace find_x_in_coconut_grove_l702_702014

theorem find_x_in_coconut_grove
  (x : ℕ)
  (h1 : (x + 2) * 30 + x * 120 + (x - 2) * 180 = 300 * x)
  (h2 : 3 * x ≠ 0) :
  x = 10 :=
by
  sorry

end find_x_in_coconut_grove_l702_702014


namespace quadratic_roots_l702_702180

theorem quadratic_roots (m : ℝ) (h_eq : ∃ α β : ℝ, (α + β = -4) ∧ (α * β = m) ∧ (|α - β| = 2)) : m = 5 :=
sorry

end quadratic_roots_l702_702180


namespace speed_of_man_l702_702036

noncomputable theory

/-- Given a bullet train of length 120 meters running at a speed of 50 km/h, 
passing a man running in the opposite direction in 8 seconds. 
Prove that the speed of the man is 4 km/h. -/
theorem speed_of_man (Ltrain : ℝ) (Vtrain : ℝ) (T : ℝ) (Vm : ℝ) 
  (h_length : Ltrain = 120) 
  (h_speed : Vtrain = 50) 
  (h_time : T = 8) 
  (h_distance : Ltrain = Vtrain * 1000 / 3600 * T)
  : Vm = 4 := 
sorry

end speed_of_man_l702_702036


namespace bookshop_total_books_l702_702405

theorem bookshop_total_books (T : ℕ) (h1 : 0.70 * T = 210) : T = 300 :=
by sorry

end bookshop_total_books_l702_702405


namespace sqrt32_plus_4sqrt_half_minus_sqrt18_l702_702079

theorem sqrt32_plus_4sqrt_half_minus_sqrt18 :
  (Real.sqrt 32 + 4 * Real.sqrt (1/2) - Real.sqrt 18) = 3 * Real.sqrt 2 :=
sorry

end sqrt32_plus_4sqrt_half_minus_sqrt18_l702_702079


namespace negation_proof_l702_702655

theorem negation_proof :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end negation_proof_l702_702655


namespace experts_expectation_100_games_envelope_5_probability_l702_702568

/-- 
Problem (a): Prove that the mathematical expectation of the number of points 
scored by the experts team over 100 games is 465.
--/
theorem experts_expectation_100_games 
  (num_envelopes : ℕ) 
  (total_points : ℕ) 
  (prob : ℚ)
  (E : ℕ → ℚ):
  num_envelopes = 13 → 
  total_points = 6 → 
  prob = 1 / 2 → 
  (E total_points) = 465 / 100 :=
  sorry

/-- 
Problem (b): Prove that the probability that envelope number 5 
will be played in the next game is approximately 0.715.
--/
theorem envelope_5_probability 
  (num_envelopes : ℕ) 
  (prob_draw : ℚ) 
  (approx_prob : ℚ):
  num_envelopes = 13 → 
  (1 / num_envelopes).natAbs.dist approx_prob < 0.01 :=
  sorry

end experts_expectation_100_games_envelope_5_probability_l702_702568


namespace exists_k_for_A_mul_v_eq_k_mul_v_l702_702111

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 6, 3]

theorem exists_k_for_A_mul_v_eq_k_mul_v (v : Fin 2 → ℝ) (h : v ≠ 0) :
  (∃ k : ℝ, A.mul_vec v = k • v) →
  k = 3 + 2 * real.sqrt 6 ∨ k = 3 - 2 * real.sqrt 6 :=
by
  sorry

end exists_k_for_A_mul_v_eq_k_mul_v_l702_702111


namespace sum_of_values_of_z_l702_702924

def f (x : ℝ) : ℝ := x^2 + 2 * x + 3

theorem sum_of_values_of_z (h : ∀ z : ℝ, f (2 * z) = 11 → (z = 1/2 ∨ z = -1)) :
  ∑ z in {1/2, -1}, z = -1/2 :=
by {
  sorry
}

end sum_of_values_of_z_l702_702924


namespace matrix_eigenvalue_problem_l702_702115

theorem matrix_eigenvalue_problem (k : ℝ) (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) :
  ((3*x + 4*y = k*x) ∧ (6*x + 3*y = k*y)) → k = 3 :=
by
  sorry

end matrix_eigenvalue_problem_l702_702115


namespace cherry_tomatoes_weight_l702_702875

def kilogram_to_grams (kg : ℕ) : ℕ := kg * 1000

theorem cherry_tomatoes_weight (kg_tomatoes : ℕ) (extra_tomatoes_g : ℕ) : kg_tomatoes = 2 → extra_tomatoes_g = 560 → kilogram_to_grams kg_tomatoes + extra_tomatoes_g = 2560 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end cherry_tomatoes_weight_l702_702875


namespace sarah_marriage_age_l702_702469

theorem sarah_marriage_age : 
  let name_length := 5 in
  let current_age := 9 in
  let twice_age := 2 * current_age in
  name_length + twice_age = 23 :=
by
  let name_length := 5
  let current_age := 9
  let twice_age := 2 * current_age
  show name_length + twice_age = 23
  sorry

end sarah_marriage_age_l702_702469


namespace find_values_of_expression_l702_702343

theorem find_values_of_expression (a b : ℝ) 
  (h : (2 * a) / (a + b) + b / (a - b) = 2) : 
  (∃ x : ℝ, x = (3 * a - b) / (a + 5 * b) ∧ (x = 3 ∨ x = 1)) :=
by 
  sorry

end find_values_of_expression_l702_702343


namespace log_5_x_eq_neg_1_point_7228_l702_702219

-- Define the conditions
def x : ℝ := (Real.logBase 4 2)^(Real.logBase 2 16)

-- Target proof statement
theorem log_5_x_eq_neg_1_point_7228 : Real.logBase 5 x = -1.7228 :=
by
  -- Add actual proof here
  sorry

end log_5_x_eq_neg_1_point_7228_l702_702219


namespace eccentricity_of_ellipse_l702_702905

theorem eccentricity_of_ellipse 
  (F₁ F₂ X Y : Point) 
  (is_ellipse : ∀ P : Point, P ∈ ellipse F₁ F₂ d ↔ distance P F₁ + distance P F₂ = d)
  (is_parabola : ∀ P : Point, P ∈ parabola F₁ F₂ ↔ distance P F₂ = distance P (directrix F₁ F₂))
  (X_Y_on_intersection : X ∈ ellipse F₁ F₂ d ∧ X ∈ parabola F₁ F₂ ∧ Y ∈ ellipse F₁ F₂ d ∧ Y ∈ parabola F₁ F₂)
  (tangents_intersect_on_directrix : intersect (tangent F₁ X) (tangent F₂ Y) ∈ directrix F₁ F₂) : 
  eccentricity F₁ F₂ d = (2 + sqrt 13) / 9 := 
sorry

end eccentricity_of_ellipse_l702_702905


namespace matrix_eigenvalue_problem_l702_702114

theorem matrix_eigenvalue_problem (k : ℝ) (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) :
  ((3*x + 4*y = k*x) ∧ (6*x + 3*y = k*y)) → k = 3 :=
by
  sorry

end matrix_eigenvalue_problem_l702_702114


namespace problem_true_l702_702929

def P : Prop := ∃ x_0 ∈ set.Ioi (0 : ℝ), x_0 + 1 / x_0 > 3
def q : Prop := ∀ x ∈ set.Ioi (2 : ℝ), x^2 > 2^x

theorem problem_true : P ∧ ¬q :=
by
  sorry

end problem_true_l702_702929


namespace weighted_average_remaining_two_l702_702643

theorem weighted_average_remaining_two (avg_10 : ℝ) (avg_2 : ℝ) (avg_3 : ℝ) (avg_3_next : ℝ) :
  avg_10 = 4.25 ∧ avg_2 = 3.4 ∧ avg_3 = 3.85 ∧ avg_3_next = 4.7 →
  (42.5 - (2 * 3.4 + 3 * 3.85 + 3 * 4.7)) / 2 = 5.025 :=
by
  intros
  sorry

end weighted_average_remaining_two_l702_702643


namespace pure_imaginary_a_value_l702_702546

theorem pure_imaginary_a_value (a : ℝ) (h : (∀ z : ℂ, (z = (a + complex.i) / (1 + 2 * complex.i)) → complex.re z = 0)) : a = -2 :=
sorry

end pure_imaginary_a_value_l702_702546


namespace abs_log_eq_condition_l702_702542

theorem abs_log_eq_condition {x y : ℝ} (h : |x - log y| = x + log y) : x * (y - 1) = 0 := 
sorry

end abs_log_eq_condition_l702_702542


namespace water_on_wednesday_l702_702204

-- Define the total water intake for the week.
def total_water : ℕ := 60

-- Define the water intake amounts for specific days.
def water_on_mon_thu_sat : ℕ := 9
def water_on_tue_fri_sun : ℕ := 8

-- Define the number of days for each intake.
def days_mon_thu_sat : ℕ := 3
def days_tue_fri_sun : ℕ := 3

-- Define the water intake calculated for specific groups of days.
def total_water_mon_thu_sat := water_on_mon_thu_sat * days_mon_thu_sat
def total_water_tue_fri_sun := water_on_tue_fri_sun * days_tue_fri_sun

-- Define the total water intake for these days combined.
def total_water_other_days := total_water_mon_thu_sat + total_water_tue_fri_sun

-- Define the water intake for Wednesday, which we need to prove to be 9 liters.
theorem water_on_wednesday : total_water - total_water_other_days = 9 := by
  -- Proof omitted.
  sorry

end water_on_wednesday_l702_702204


namespace polygon_area_l702_702557

noncomputable def section_1_area (n₁ sides_perimeter₁ : ℕ) :=
  let side_length₁ := sides_perimeter₁ / n₁
  in n₁ / 4 * (side_length₁ * side_length₁)

noncomputable def section_2_area (n₂ sides_perimeter₂ : ℕ) :=
  let side_length₂ := sides_perimeter₂ / n₂
  in n₂ / 4 * (side_length₂ * side_length₂)

theorem polygon_area : 
  ∀ (n₁ n₂ sides_perimeter₁ sides_perimeter₂ : ℕ), 
  n₁ = 16 → sides_perimeter₁ = 32 → n₂ = 20 → sides_perimeter₂ = 40 → 
  section_1_area n₁ sides_perimeter₁ + section_2_area n₂ sides_perimeter₂ = 56 := by
  intros 
  simp [section_1_area, section_2_area]
  sorry

end polygon_area_l702_702557


namespace rd_sum_4281_rd_sum_formula_rd_sum_count_3883_count_self_equal_rd_sum_l702_702127

-- Define the digit constraints and the RD sum function
def is_digit (n : ℕ) : Prop := n < 10
def is_nonzero_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

def rd_sum (A B C D : ℕ) : ℕ :=
  let abcd := 1000 * A + 100 * B + 10 * C + D
  let dcba := 1000 * D + 100 * C + 10 * B + A
  abcd + dcba

-- Problem (a)
theorem rd_sum_4281 : rd_sum 4 2 8 1 = 6105 := sorry

-- Problem (b)
theorem rd_sum_formula (A B C D : ℕ) (hA : is_nonzero_digit A) (hD : is_nonzero_digit D) :
  ∃ m n, m = 1001 ∧ n = 110 ∧ rd_sum A B C D = m * (A + D) + n * (B + C) :=
  sorry

-- Problem (c)
theorem rd_sum_count_3883 :
  ∃ n, n = 18 ∧ ∃ (A B C D : ℕ), is_nonzero_digit A ∧ is_digit B ∧ is_digit C ∧ is_nonzero_digit D ∧ rd_sum A B C D = 3883 :=
  sorry

-- Problem (d)
theorem count_self_equal_rd_sum : 
  ∃ n, n = 143 ∧ ∀ (A B C D : ℕ), is_nonzero_digit A ∧ is_digit B ∧ is_digit C ∧ is_nonzero_digit D → (1001 * (A + D) + 110 * (B + C) ≤ 9999 → (1000 * A + 100 * B + 10 * C + D = rd_sum A B C D → 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ D ∧ D ≤ 9)) :=
  sorry

end rd_sum_4281_rd_sum_formula_rd_sum_count_3883_count_self_equal_rd_sum_l702_702127


namespace compare_values_l702_702496

noncomputable def log3 := Real.log 0.5 / Real.log 3
noncomputable def log03 := Real.log 0.2 / Real.log 0.3
noncomputable def exp5 := 0.5 ^ 0.3

theorem compare_values : log03 > exp5 ∧ exp5 > log3 := by
  sorry

end compare_values_l702_702496


namespace simplify_sqrt_l702_702966

theorem simplify_sqrt (m : ℝ) (h : m < 1) : (m - 1) * real.sqrt (-1 / (m - 1)) = - real.sqrt (1 - m) :=
by
  sorry

end simplify_sqrt_l702_702966


namespace does_not_represent_right_triangle_l702_702579

/-- In triangle ABC, the sides opposite to angles A, B, and C are a, b, and c respectively. Given:
  - a:b:c = 6:8:10
  - ∠A:∠B:∠C = 1:1:3
  - a^2 + c^2 = b^2
  - ∠A + ∠B = ∠C

Prove that the condition ∠A:∠B:∠C = 1:1:3 does not represent a right triangle ABC. -/
theorem does_not_represent_right_triangle
  (a b c : ℝ) (A B C : ℝ)
  (h1 : a / b = 6 / 8 ∧ b / c = 8 / 10)
  (h2 : A / B = 1 / 1 ∧ B / C = 1 / 3)
  (h3 : a^2 + c^2 = b^2)
  (h4 : A + B = C) :
  ¬ (B = 90) :=
sorry

end does_not_represent_right_triangle_l702_702579


namespace count_multiples_6_or_8_not_both_l702_702856

theorem count_multiples_6_or_8_not_both (bound : ℕ) (h1 : bound = 201) :
  let multiples_6 := { k | k ∈ finset.range bound ∧ k % 6 = 0 }
  let multiples_8 := { k | k ∈ finset.range bound ∧ k % 8 = 0 }
  let multiples_24 := { k | k ∈ finset.range bound ∧ k % 24 = 0 }
  let count_6 := multiples_6.card
  let count_8 := multiples_8.card
  let count_24 := multiples_24.card
  count_6 - count_24 + count_8 - count_24 = 42 :=
by
  sorry

end count_multiples_6_or_8_not_both_l702_702856


namespace find_p1_plus_q1_l702_702779

noncomputable def p (x : ℤ) := x^4 + 14 * x^2 + 1
noncomputable def q (x : ℤ) := x^4 - 14 * x^2 + 1

theorem find_p1_plus_q1 :
  (p 1) + (q 1) = 4 :=
sorry

end find_p1_plus_q1_l702_702779


namespace smallest_positive_period_minimum_value_on_interval_l702_702187

def f (x : ℝ) : ℝ := cos x * sin x - sqrt 3 * (cos x)^2 + (sqrt 3) / 2

theorem smallest_positive_period : ∀ x : ℝ, f (x + π) = f x := sorry

theorem minimum_value_on_interval : 
  ∃ x ∈ Icc (-(π / 4)) (π / 4), ∀ y ∈ Icc (-(π / 4)) (π / 4), f x ≤ f y ∧ f x = -1 ∧ x = -(π / 12) := sorry

end smallest_positive_period_minimum_value_on_interval_l702_702187


namespace exists_N_n_l702_702910

noncomputable def g (x : ℝ) : ℝ := (3 * x + 4) / (x + 3)

theorem exists_N_n (T : set ℝ) (x : ℝ) (hx : x ≥ 1)
  (T_def : ∀ y, y ∈ T ↔ ∃ x, x ≥ 1 ∧ g(x) = y) :
  ∃ N n, N = 3 ∧ n = 7 / 4 ∧
         (n ∈ T) ∧ (N ∉ T) :=
by
  let N := 3
  let n := 7 / 4
  have hN_not_mem_T: N ∉ T := sorry
  have hn_mem_T: n ∈ T := sorry
  exact ⟨N, n, rfl, rfl, hn_mem_T, hN_not_mem_T⟩

end exists_N_n_l702_702910


namespace probability_encounter_sequence_10_100_1000_l702_702291

open Classical

noncomputable def transition_probability (n : ℕ) : ℕ := 
  if n = 10 then 10 else 
  if n = 100 then 100 else
  if n = 1000 then 1000 else 2019000000

theorem probability_encounter_sequence_10_100_1000 (n : ℕ) (H : n = 2019) :
  transition_probability n = 2019000000 :=
by
  cases H
  simp [transition_probability]
  sorry

end probability_encounter_sequence_10_100_1000_l702_702291


namespace proof_l702_702785

def num_values_b_passing_through_vertex : Prop :=
  let p : ℝ → ℝ := λ x => x^2 - 2 * (x * x)
  let l : ℝ → ℝ := λ x => x + x
  let vertex : ℝ × ℝ := (0, -2 * (0 * 0))
  let b_eq (b : ℝ) : Prop := b = -(2 * b^2)
  (Set.card (Set.of {b : ℝ | b_eq b})) = 2

theorem proof : num_values_b_passing_through_vertex :=
by
  -- proof goes here
  sorry

end proof_l702_702785


namespace circle_radius_tangent_to_circumcircles_l702_702585

noncomputable def circumradius (a b c : ℝ) : ℝ :=
  (a * b * c) / (4 * (Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))))

theorem circle_radius_tangent_to_circumcircles (AB BC CA : ℝ) (H : Point) 
  (h_AB : AB = 13) (h_BC : BC = 14) (h_CA : CA = 15) : 
  (radius : ℝ) = 65 / 16 :=
by
  sorry

end circle_radius_tangent_to_circumcircles_l702_702585


namespace areas_of_triangles_l702_702934

open EuclideanGeometry

variables {A B C D E F M P R S L N K : Point ℝ}

-- Given 
-- Intersecting points:
axiom h1 : Line_thru B C ∩ Line_thru E D = P
axiom h2 : Line_thru E D ∩ Line_thru A F = R
axiom h3 : Line_thru A F ∩ Line_thru D C = S
axiom h4 : Line_thru A B ∩ Line_thru C D = L
axiom h5 : Line_thru C D ∩ Line_thru E F = N
axiom h6 : Line_thru E F ∩ Line_thru A B = K

-- Equilateral triangles:
axiom h7 : EquilateralTriangle [K, L, N]
axiom h8 : EquilateralTriangle [S, R, P]
axiom h9 : CongruentTriangle [K, L, N] [S, R, P]

-- Sum of distances:
axiom h10 : (distance_to_line M B C) + (distance_to_line M E D) + (distance_to_line M A F) = 
            (distance_to_line M A B) + (distance_to_line M C D) + (distance_to_line M A F)

-- Sum of areas:
axiom h11 : area A M B + area C M D + area F M E + area B M C + area D M E + area A M E = (1/2) * area A B C D E F
axiom h12 : area A M B + area E M D = area B M C + area M E F = area C M D + area A M F = (1/3) * area A B C D E F

-- Prove:
theorem areas_of_triangles : 
  area M F E = 6 ∧ 
  area B M C = 6 ∧ 
  area D M E = 9 ∧ 
  area A M F = 3 :=
sorry

end areas_of_triangles_l702_702934


namespace coordinates_with_respect_to_origin_l702_702249

def point_coordinates (x y : ℤ) : ℤ × ℤ :=
  (x, y)

def origin : ℤ × ℤ :=
  (0, 0)

theorem coordinates_with_respect_to_origin :
  point_coordinates 2 (-3) = (2, -3) := by
  -- placeholder proof
  sorry

end coordinates_with_respect_to_origin_l702_702249


namespace total_children_l702_702667

-- Definitions for the conditions in the problem
def boys : ℕ := 19
def girls : ℕ := 41

-- Theorem stating the total number of children is 60
theorem total_children : boys + girls = 60 :=
by
  -- calculation done to show steps, but not necessary for the final statement
  sorry

end total_children_l702_702667


namespace quadrilateral_area_is_one_fifth_l702_702885

variables (A B C D M N : Point)
variables (parallelogram : A B C D)
variables (mid_M_CD : midpoint M C D)
variables (mid_N_BC : midpoint N B C)

theorem quadrilateral_area_is_one_fifth (S : ℝ):
  area (Quadrilateral A M B N) = (1/5) * area (Parallelogram A B C D) :=
sorry

end quadrilateral_area_is_one_fifth_l702_702885


namespace measure_angle_AFE_is_110_l702_702563

-- Definitions of the given conditions
variables (A B C D E F : Type)
variables [Square ABCD]
variables [PointOn AD F]
variables [Angle CDE : 140]
variables [EquivSegments DE DF]

-- Theorem: The measure of angle AFE is 110 degrees
theorem measure_angle_AFE_is_110 :
  measure_angle A F E = 110 :=
by
  sorry

end measure_angle_AFE_is_110_l702_702563


namespace KP_is_2sqrt33_l702_702412

-- Definitions based on conditions
variables (K P M F B : Type) [MetricSpace K]
variables [MetricSpace P] [MetricSpace M] [MetricSpace F] [MetricSpace B] 
variables {x y : ℝ} (h1 : dist K F / dist F M = 3) (h2 : dist P B / dist B M = 6/5)
  (h3 : dist B F = sqrt 15)

-- The formal statement to be proven
theorem KP_is_2sqrt33 : dist K P = 2 * sqrt 33 :=
  sorry

end KP_is_2sqrt33_l702_702412


namespace value_range_f_l702_702355

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + 2 * Real.cos x - Real.sin (2 * x) + 1

theorem value_range_f :
  ∀ x ∈ Set.Ico (-(5 * Real.pi) / 12) (Real.pi / 3), 
  f x ∈ Set.Icc ((3 : ℝ) / 2 - Real.sqrt 2) 3 :=
by
  sorry

end value_range_f_l702_702355


namespace routes_from_P_to_R_l702_702368

theorem routes_from_P_to_R : 
  let routes (x y : string) : Nat :=
    if (x = "P" ∧ y = "Q") ∨ (x = "P" ∧ y = "S") then 1
    else if (x = "Q" ∧ y = "R") ∨ (x = "Q" ∧ y = "T") then 1
    else if (x = "S" ∧ y = "R") ∨ (x = "T" ∧ y = "R") then 1
    else 0 in
  routes "P" "Q" + routes "Q" "R" + routes "Q" "T" + routes "P" "S" + routes "S" "R" = 3 := 
by 
  sorry

end routes_from_P_to_R_l702_702368


namespace imaginary_part_of_z_l702_702179

section complex_number

open Complex

def z : ℂ := (2 + Complex.i) / ((1 + Complex.i)^2)

theorem imaginary_part_of_z : z.im = -1 := 
  sorry

end complex_number

end imaginary_part_of_z_l702_702179


namespace tennis_tournament_ways_l702_702637

theorem tennis_tournament_ways :
  ∃ (f : Fin 6 → Fin 6), (∀ (i j : Fin 6), i ≠ j → (f i > f j ↔ f i = 5 ∧ f j = 4 ∨ f i = 4 ∧ f j = 3 ∨ f i = 3 ∧ f j = 2 ∨ f i = 2 ∧ f j = 1 ∨ f i = 1 ∧ f j = 0)) ∧ 
  (∃! (perm : Fin 6 → Fin 6), 
    ∀ i j, i ≠ j → (perm i > perm j ↔ perm i = 5 ∧ perm j = 4 ∨ perm i = 4 ∧ perm j = 3 ∨ perm i = 3 ∧ perm j = 2 ∨ perm i = 2 ∧ perm j = 1 ∨ perm i = 1 ∧ perm j = 0))
  → ∃! (p : Fin 6 → Fin 6), ∀ k, p k ∈ {0, 1, 2, 3, 4, 5} ∧ (Bijective p) :=
by
  sorry

end tennis_tournament_ways_l702_702637


namespace sin_cos_ratio_l702_702920

namespace TrigProof

theorem sin_cos_ratio (x y : ℝ) (h1 : sin x / sin y = 4) (h2 : cos x / cos y = 1 / 3) :
  (sin (2 * x) / sin (2 * y)) + (cos (2 * x) / cos (2 * y)) = -29 / 3 := by
  sorry

end TrigProof

end sin_cos_ratio_l702_702920


namespace sufficiency_and_necessity_of_p_and_q_l702_702160

noncomputable def p : Prop := ∀ k, k = Real.sqrt 3
noncomputable def q : Prop := ∀ k, ∃ y x, y = k * x + 2 ∧ x^2 + y^2 = 1

theorem sufficiency_and_necessity_of_p_and_q : (p → q) ∧ (¬ (q → p)) := by
  sorry

end sufficiency_and_necessity_of_p_and_q_l702_702160


namespace parabola_vertex_l702_702646

theorem parabola_vertex (x y : ℝ) : 
  (∀ x y, y^2 - 8*y + 4*x = 12 → (x, y) = (7, 4)) :=
by
  intros x y h
  sorry

end parabola_vertex_l702_702646


namespace exists_k_for_A_mul_v_eq_k_mul_v_l702_702109

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 6, 3]

theorem exists_k_for_A_mul_v_eq_k_mul_v (v : Fin 2 → ℝ) (h : v ≠ 0) :
  (∃ k : ℝ, A.mul_vec v = k • v) →
  k = 3 + 2 * real.sqrt 6 ∨ k = 3 - 2 * real.sqrt 6 :=
by
  sorry

end exists_k_for_A_mul_v_eq_k_mul_v_l702_702109


namespace log_15_20_eq_l702_702165

noncomputable def problem (a b : ℝ) (c₁ : 2 ^ a = 3) (c₂ : Real.log 5 / Real.log 3 = b) : ℝ :=
Real.log 20 / Real.log 15

theorem log_15_20_eq (a b : ℝ) (c₁ : 2 ^ a = 3) (c₂ : Real.log 5 / Real.log 3 = b) :
  problem a b c₁ c₂ = (2 + a * b) / (a + a * b) :=
by
  sorry

end log_15_20_eq_l702_702165


namespace find_a_l702_702233

theorem find_a
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : 2 * (b * Real.cos A + a * Real.cos B) = c^2)
  (h2 : b = 3)
  (h3 : 3 * Real.cos A = 1) :
  a = 3 :=
sorry

end find_a_l702_702233


namespace range_x_y_l702_702138

variable (x y : ℝ)

theorem range_x_y (hx : 60 < x ∧ x < 84) (hy : 28 < y ∧ y < 33) : 
  27 < x - y ∧ x - y < 56 :=
sorry

end range_x_y_l702_702138


namespace coordinates_of_point_l702_702245

theorem coordinates_of_point : 
  ∀ (x y : ℝ), (x, y) = (2, -3) → (x, y) = (2, -3) := 
by 
  intros x y h 
  exact h

end coordinates_of_point_l702_702245


namespace sqrt_solution_l702_702145

-- Given Conditions
variables {f : ℝ → ℝ}
variables {a : ℝ}

-- Condition 1: f(x) is continuous for x in ℝ⁺
variable (h_cont : ∀ x : ℝ, 0 ≤ x → continuous_at f x)

-- Condition 2: f(0) = 0
variable (h_zero : f 0 = 0)

-- Condition 3: f^2(x+y) ≥ f^2(x) + f^2(y) for all x, y ∈ ℝ⁺
variable (h_ineq : ∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → (f (x + y))^2 ≥ (f x)^2 + (f y)^2)

-- To Prove: f(x) = a * sqrt(x) where a > 0
theorem sqrt_solution : (∃ a : ℝ, 0 < a ∧ ∀ x : ℝ, 0 ≤ x → f x = a * real.sqrt x) :=
sorry

end sqrt_solution_l702_702145


namespace value_of_x_at_0_6_l702_702194

def x (p q : ℝ) : ℝ :=
  if 0 ≤ p ∧ p ≤ 0.5 then 1
  else if 0.5 < p ∧ p ≤ 1 then q / p
  else 0

theorem value_of_x_at_0_6 (q : ℝ) (h_q : q = 2) : x 0.6 q = 2 / 3 := 
by
  have h_p : 0.5 < 0.6 ∧ 0.6 ≤ 1 := ⟨by linarith, by linarith⟩
  rw [x, if_neg (by linarith), if_pos h_p]
  rw h_q
  norm_num
  sorry

end value_of_x_at_0_6_l702_702194


namespace mass_percentage_Al_in_AlPO4_l702_702482

-- Definitions of the molar masses
def molar_mass_Al : Float := 26.98
def molar_mass_P : Float := 30.97
def molar_mass_O : Float := 16.00
def molar_mass_AlPO4 : Float := molar_mass_Al + molar_mass_P + (4 * molar_mass_O)

-- Definition of the mass percentage function
def mass_percentage_Al : Float := (molar_mass_Al / molar_mass_AlPO4) * 100

-- Theorem stating the mass percentage of Al in AlPO₄ is approximately 22.12%
theorem mass_percentage_Al_in_AlPO4 : abs (mass_percentage_Al - 22.12) < 0.01 := 
by sorry

end mass_percentage_Al_in_AlPO4_l702_702482


namespace evaluate_series_l702_702472

-- Define the series S
noncomputable def S : ℝ := ∑' n : ℕ, (n + 1) / (3 ^ (n + 1))

-- Lean statement to show the evaluated series
theorem evaluate_series : (3:ℝ)^S = (3:ℝ)^(3 / 4) :=
by
  -- The proof is omitted
  sorry

end evaluate_series_l702_702472


namespace coordinates_with_respect_to_origin_l702_702250

theorem coordinates_with_respect_to_origin (P : ℝ × ℝ) (h : P = (2, -3)) : P = (2, -3) :=
by
  sorry

end coordinates_with_respect_to_origin_l702_702250


namespace referee_matches_inequality_l702_702392

theorem referee_matches_inequality (k : ℕ) (Hk : k > 0) :
  let num_players := 2 * k
  let total_matches := (num_players * (num_players - 1)) / 2
  ¬ ∃ n : ℕ, (total_matches = n * num_players) := 
by {
  let num_players := 2 * k,
  let total_matches := (num_players * (num_players - 1)) / 2,
  intro h,
  rcases h with ⟨n, hn⟩,
  have : 2 * n * num_players = num_players * (num_players - 1),
  { rw [← hn, mul_assoc, mul_comm _ num_players, ← mul_assoc] },
  have : 2 * n = num_players - 1 / num_players,
  { rw [mul_comm 2 n, ← mul_div_assoc _ 2 (num_players : ℚ), mul_comm _ (2 : ℚ), mul_div_cancel _ (by norm_cast; linarith)] }
  norm_cast at this,
  sorry
}

end referee_matches_inequality_l702_702392


namespace find_W_l702_702059

noncomputable def volumeOutsideCylinder (r_cylinder r_sphere : ℝ) : ℝ :=
  let h := 2 * Real.sqrt (r_sphere^2 - r_cylinder^2)
  let V_sphere := (4 / 3) * Real.pi * r_sphere^3
  let V_cylinder := Real.pi * r_cylinder^2 * h
  V_sphere - V_cylinder

theorem find_W : 
  volumeOutsideCylinder 4 7 = (1372 / 3 - 32 * Real.sqrt 33) * Real.pi :=
by
  sorry

end find_W_l702_702059


namespace greatest_integer_solution_l702_702681

theorem greatest_integer_solution (n : ℤ) (h : n^2 - 13 * n + 36 ≤ 0) : n ≤ 9 :=
by
  sorry

end greatest_integer_solution_l702_702681


namespace factorization_correct_l702_702007

theorem factorization_correct :
  ∀ (x y : ℝ), 
    (¬ ( (y - 1) * (y + 1) = y^2 - 1 ) ) ∧
    (¬ ( x^2 * y + x * y^2 - 1 = x * y * (x + y) - 1 ) ) ∧
    (¬ ( (x - 2) * (x - 3) = (3 - x) * (2 - x) ) ) ∧
    ( x^2 - 4 * x + 4 = (x - 2)^2 ) :=
by
  intros x y
  repeat { constructor }
  all_goals { sorry }

end factorization_correct_l702_702007


namespace sum_of_specific_non_palindromes_l702_702131

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def reverse_number (n : ℕ) : ℕ :=
  (n.digits 10).reverse.foldr (λ d acc, acc * 10 + d) 0

def becomes_palindrome_in_steps (n : ℕ) (steps : ℕ) : Prop :=
  (λ (m, k), k = steps ∧ is_palindrome m) ((nat.rec_on steps n (λ _ acc, acc + reverse_number acc)), steps)

theorem sum_of_specific_non_palindromes : 
  ∑ n in finset.filter (λ n, ¬ is_palindrome n ∧ 10 ≤ n ∧ n < 100 ∧ becomes_palindrome_in_steps n 5) (finset.range 100), n = 0 :=
by
  sorry

end sum_of_specific_non_palindromes_l702_702131


namespace standard_equation_of_ellipse_min_length_MN_l702_702521

-- Define the given conditions
variables {a b c : ℝ} (x y : ℝ)
def ellipse := (x^2) / (a^2) + (y^2) / (b^2) = 1
def a_gt_b_gt_0 := a > b ∧ b > 0
def max_distance_to_focus := a + c = 3 ∧ c / a = 1/2
def eccentricity_root := ∃ e, (2 * e^2 - 5 * e + 2 = 0 ∧ e > 0 ∧ e < 1)

-- Define the proof problems
theorem standard_equation_of_ellipse (hx : ellipse x y) (hab : a_gt_b_gt_0) (hmf : max_distance_to_focus) (her : eccentricity_root) :
    a = 2 ∧ b = sqrt 3 := sorry

theorem min_length_MN (P : ℝ × ℝ) (hx : ellipse x y) (hab : a_gt_b_gt_0) (hmf : max_distance_to_focus) (her : eccentricity_root) :
    ∃ M N : ℝ × ℝ, M.1 = 4 ∧ N.1 = 4 ∧ min_length P M N := sorry

end standard_equation_of_ellipse_min_length_MN_l702_702521


namespace slices_per_person_l702_702958

theorem slices_per_person (total_slices : ℕ) (total_people : ℕ) (h_slices : total_slices = 12) (h_people : total_people = 3) :
  total_slices / total_people = 4 :=
by
  sorry

end slices_per_person_l702_702958


namespace cylinder_volume_existence_l702_702729

theorem cylinder_volume_existence (a : ℝ) (h : a > 0) : 
    ∃ (V : ℝ), (V = (a^3) / π ∨ V = (a^3) / (2 * π)) := 
by
  use (a^3) / π
  use (a^3) / (2 * π)
  sorry

end cylinder_volume_existence_l702_702729


namespace distance_to_nearest_lattice_point_l702_702055

noncomputable def sqrt (x : ℝ) := Real.sqrt x

theorem distance_to_nearest_lattice_point :
  ∃ d : ℝ, (∃ (p : ℝ), p = 7 / 16 / 2560000 * 2500 * 2500 * 4 / π) ∧ 
  sqrt (7 / 16 / π) = 0.4 :=
by sorry

end distance_to_nearest_lattice_point_l702_702055


namespace find_inradius_of_scalene_triangle_l702_702600

noncomputable def side_a := 32
noncomputable def side_b := 40
noncomputable def side_c := 24
noncomputable def ic := 18
noncomputable def expected_inradius := 2 * Real.sqrt 17

theorem find_inradius_of_scalene_triangle (a b c : ℝ) (h : a = side_a) (h1 : b = side_b) (h2 : c = side_c) (ic_length : ℝ) (h3: ic_length = ic) : (Real.sqrt (ic_length ^ 2 - (b - ((a + b - c) / 2)) ^ 2)) = expected_inradius :=
by
  sorry

end find_inradius_of_scalene_triangle_l702_702600


namespace prop_2_l702_702143

variables (m n : Plane → Prop) (α β γ : Plane)

def perpendicular (m : Line) (α : Plane) : Prop :=
  -- define perpendicular relationship between line and plane
  sorry

def parallel (m : Line) (n : Line) : Prop :=
  -- define parallel relationship between two lines
  sorry

-- The proof of proposition (2) converted into Lean 4 statement
theorem prop_2 (hm₁ : perpendicular m α) (hn₁ : perpendicular n α) : parallel m n :=
  sorry

end prop_2_l702_702143


namespace ratio_perimeter_triangle_l702_702562

theorem ratio_perimeter_triangle
  (P Q R S : Type) [real_inner_product_space ℝ P]
  (PQ PR QR PS RS : ℝ)
  (hypotenuse : PQ ^ 2 + QR ^ 2 = PR ^ 2)
  (PQ_eq : PQ = 9)
  (QR_eq : QR = 40)
  (perimeter_SAS : 2 * 40 + PR = 121)
  (PR_eq : PR = 41) :
  (perimeter_SAS / PR) = 121 / 41 :=
by sorry

end ratio_perimeter_triangle_l702_702562


namespace min_value_seq_ratio_l702_702843

-- Define the sequence {a_n} based on the given recurrence relation and initial condition
def seq (n : ℕ) : ℕ := 
  if n = 0 then 0 -- Handling the case when n is 0, though sequence starts from n=1
  else n^2 - n + 15

-- Prove the minimum value of (a_n / n) is 27/4
theorem min_value_seq_ratio : 
  ∃ n : ℕ, n > 0 ∧ seq n / n = 27 / 4 :=
by
  sorry

end min_value_seq_ratio_l702_702843


namespace bottles_used_during_second_game_l702_702409

theorem bottles_used_during_second_game :
  let total_bottles := 10 * 20,
      remaining_after_first_game := total_bottles - 70,
      remaining_after_second_game := 20 in
  remaining_after_first_game - remaining_after_second_game = 110 := by
  sorry

end bottles_used_during_second_game_l702_702409


namespace second_caterer_cheaper_l702_702621

theorem second_caterer_cheaper (x : ℕ) :
  ∀ x, (250 + 14 * x - (if x >= 50 then 50 else 0) < 120 + 18 * x) → x ≥ 50 :=
begin
  intro x,
  simp,
  split_ifs,
  { exact nat.le_of_lt (lt_of_le_of_lt trivial (by linarith)) },
  { linarith }
end

end second_caterer_cheaper_l702_702621


namespace sum_of_squares_at_least_n_squared_l702_702926

def is_snake (length: ℕ) (cells: set (ℕ × ℕ)) : Prop :=
  ∃ k, cells = { (i, j) | i = k ∧ j ≥ 1 ∧ j ≤ length } ∨ cells = { (i, j) | j = k ∧ i ≥ 1 ∧ i ≤ length }

noncomputable def sum_of_squares (lengths: list ℕ) : ℕ :=
  lengths.foldl (λ acc l, acc + l * l) 0

theorem sum_of_squares_at_least_n_squared
  (n: ℕ)
  (p: set (ℕ × ℕ) → Prop)
  (a: set (ℕ × ℕ) → Prop)
  (pain_condition: ∀ (x: set (ℕ × ℕ)), p x ∨ a x ∧ ∃ s ⊆ {1, 2, ..., n} × {1, 2, ..., n}, x = s)
  (disjoint_condition: ∀ x y, x ≠ y → ¬(p x ∧ p y) ∧ ¬(a x ∧ a y) ∧ ¬(p x ∧ a y))
  (adjacency_condition: ∀ (x: set (ℕ × ℕ)), (∃ i j, (i, j) ∈ x ∧ (i + 1, j) ∈ x ∧ a x) ∨ (∃ i j, (i, j) ∈ x ∧ (i, j + 1) ∈ x ∧ p x))
  (anaconda_next_to_python: ∀ (i j: ℕ), p { (i, j) | i = i ∧ j ≥ 1 ∧ j ≤ n } → a { (i, j) | j = j ∧ i ≥ 1 ∧ i ≤ n })
  (python_next_to_anaconda: ∀ (i j: ℕ), a { (i, j) | j = j ∧ i ≥ 1 ∧ i ≤ n } → p { (i, j) | i = i ∧ j ≥ 1 ∧ j ≤ n })
  (sum_squares_ge_n_sq: ∀ lengths: list ℕ, sum_of_squares lengths ≥ n * n)
  : sum_of_squares (list.map (λ x, x.cardinality) (filter (λ x, p x ∨ a x) (list.powerset (nat.powerset (λ x, true))))) ≥ n * n  :=
sorry

end sum_of_squares_at_least_n_squared_l702_702926


namespace sum_first_seven_terms_l702_702156

variables {a : ℕ → ℝ} (d a1 : ℝ)

def arithmetic_seq (n : ℕ) : ℝ := a1 + (n - 1) * d

def sum_first_n_terms (n : ℕ) : ℝ := n * a1 + (n * (n - 1) / 2) * d

theorem sum_first_seven_terms (h : arithmetic_seq 2 a1 d + arithmetic_seq 3 a1 d + arithmetic_seq 4 a1 d = 12) :
  sum_first_n_terms 7 a1 d = 28 :=
sorry

end sum_first_seven_terms_l702_702156


namespace pooja_speed_proof_l702_702957

-- Definition for Roja's speed
def Roja_speed : ℝ := 7

-- Definition for time
def time : ℝ := 4

-- Definition for distance between them after the given time
def distance : ℝ := 40

-- Definition for Pooja's speed, which is to be proved as 3 km/hr
def Pooja_speed : ℝ := 3

-- Lean 4 statement to prove Pooja's speed
theorem pooja_speed_proof :
  ∃ (v : ℝ), (7 + v) * 4 = 40 ∧ v = 3 :=
by
  use Pooja_speed
  split
  { sorry }
  { sorry }

end pooja_speed_proof_l702_702957


namespace matrix_vector_combination_l702_702608

open Matrix

variables {R : Type*} [Field R] 
variables {n m : Type*} [Fintype n] [Fintype m] [DecidableEq n] [DecidableEq m]
variables (M : Matrix n m R) (v w u : Matrix m (Fin 1) R)

def v_value : Matrix m (Fin 1) R := ![![2], ![6]]
def w_value : Matrix m (Fin 1) R := ![![3], ![-5]]
def u_value : Matrix m (Fin 1) R := ![![-1], ![4]]

def Mv := M * v = v_value
def Mw := M * w = w_value
def Mu := M * u = u_value

theorem matrix_vector_combination :
  M * (2 * v - w + 4 * u) = ![![ -3], ![ 33]] :=
by
  sorry

end matrix_vector_combination_l702_702608


namespace evaluation_l702_702968

noncomputable def simplify_and_evaluate (x : ℝ) : ℝ :=
  (x + 1) / x / (x - 1 / x)

theorem evaluation (x : ℝ) (h : x = Real.sqrt 3 + 1) : simplify_and_evaluate x = Real.sqrt 3 / 3 :=
by
  rw [simplify_and_evaluate]
  rw [h]
  -- here we would proceed with the simplifications and substitutions as done in the solution
  sorry

end evaluation_l702_702968


namespace net_rate_of_pay_is_25_l702_702040

-- Define the conditions 
variables (hours : ℕ) (speed : ℕ) (efficiency : ℕ)
variables (pay_per_mile : ℝ) (cost_per_gallon : ℝ)
variables (total_distance : ℕ) (gas_used : ℕ)
variables (total_earnings : ℝ) (total_cost : ℝ) (net_earnings : ℝ) (net_rate_of_pay : ℝ)

-- Assume the given conditions are as stated in the problem
axiom hrs : hours = 3
axiom spd : speed = 50
axiom eff : efficiency = 25
axiom ppm : pay_per_mile = 0.60
axiom cpg : cost_per_gallon = 2.50

-- Assuming intermediate computations
axiom distance_calc : total_distance = speed * hours
axiom gas_calc : gas_used = total_distance / efficiency
axiom earnings_calc : total_earnings = pay_per_mile * total_distance
axiom cost_calc : total_cost = cost_per_gallon * gas_used
axiom net_earnings_calc : net_earnings = total_earnings - total_cost
axiom pay_rate_calc : net_rate_of_pay = net_earnings / hours

-- Proving the final result
theorem net_rate_of_pay_is_25 :
  net_rate_of_pay = 25 :=
by
  -- Proof goes here
  sorry

end net_rate_of_pay_is_25_l702_702040


namespace maximum_N_prime_l702_702804

noncomputable def J (k : ℕ) : ℕ := 10^(k + 4) + 128

-- Function to count the number of factors of 2 in the prime factorization of a number
noncomputable def N' (n : ℕ) : ℕ :=
  if n = 0 then 0 else (nat.find_greatest (λ m, 2^m ∣ n) (nat.log 2 n)) + 1

theorem maximum_N_prime (k : ℕ) (h : k > 0) : 
  ∃ M, ∀ k > 0, N' (J k) ≤ M ∧ M = 12 := sorry

end maximum_N_prime_l702_702804


namespace balance_scale_difference_l702_702306

theorem balance_scale_difference (a b : ℕ) (α β : ℕ) (hα : 0 ≤ α ∧ α ≤ 20) (hβ : 0 ≤ β ∧ β ≤ 20) :
  abs ((a + b + β) - (a + α + b)) ≤ 20 :=
by
  -- The difference is computed as |β - α|. 
  -- Given 0 ≤ α, β ≤ 20, |β - α| must be ≤ 20.
  sorry

end balance_scale_difference_l702_702306


namespace vector_dot_product_proof_l702_702202

def vector_dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem vector_dot_product_proof : 
  let a := (-1, 2)
  let b := (2, 3)
  vector_dot_product a (a.1 - b.1, a.2 - b.2) = 1 :=
by {
  sorry
}

end vector_dot_product_proof_l702_702202


namespace fill_time_no_leak_is_5_l702_702746

-- Conditions
def fill_time_without_leak : ℕ := T
def fill_time_with_leak : ℕ := T + 1
def leak_empty_time : ℕ := 30

-- Filling rates
def rate_filling_without_leak (T : ℕ) : ℚ := 1 / T
def rate_leak : ℚ := 1 / 30
def effective_rate_with_leak (T : ℕ) : ℚ := rate_filling_without_leak T - rate_leak
def effective_rate_fill_with_leak (T : ℕ) : ℚ := 1 / (T + 1)

-- Proof Statement
theorem fill_time_no_leak_is_5 (T : ℕ) :
  effective_rate_with_leak T = effective_rate_fill_with_leak T → T = 5 :=
by
  sorry

end fill_time_no_leak_is_5_l702_702746


namespace rthea_dna_sequences_l702_702714

/-- 
Rthea, a distant planet, is home to creatures whose DNA consists of two distinguishable strands of 
bases with a fixed orientation. Each base is one of the letters H, M, N, T, and each strand consists 
of a sequence of five bases, thus forming five pairs. Due to the chemical properties of the bases, 
each pair must consist of distinct bases. Also, the bases H and M cannot appear next to each other 
on the same strand; the same is true for N and T. How many possible DNA sequences are there on Rthea?
-/
theorem rthea_dna_sequences : 
  let bases := ["H", "M", "N", "T"]
  let valid_sequence (seq : List String) : Prop :=
        (∀ i j, seq.nth i = some "H" → seq.nth (i + 1) = some "M" → false)
     ∧ (∀ i j, seq.nth i = some "N" → seq.nth (i + 1) = some "T" → false)
  ∃ (count : ℕ), count = 1259712 :=
begin
  sorry
end

end rthea_dna_sequences_l702_702714


namespace square_side_length_and_area_l702_702061

theorem square_side_length_and_area (rect_side1 rect_side2 : ℝ) (h1 : rect_side1 = 10) (h2 : rect_side2 = 7) : 
  let p := 2 * (rect_side1 + rect_side2) in
  let s := p / 4 in 
  s = 8.5 ∧ s^2 = 72.25 := 
by
  sorry

end square_side_length_and_area_l702_702061


namespace general_integral_l702_702480

theorem general_integral (x y : ℝ) (C : ℝ) :
  ( ∃ y' : ℝ → ℝ,
    (∀ x y : ℝ, y' x = (x^2 + 3*x*y - y^2) / (3*x^2 - 2*x*y)) )  →
    (∀ x y : ℝ, 
      3 * real.arctan (y / x) = real.log (C * (x^2 + y^2) / |x|)
  ) :=
sorry

end general_integral_l702_702480


namespace reflection_proof_l702_702058

def vec2 (x y : ℝ) := ⟨x, y⟩

variable (u v w : ℝ × ℝ) -- Conditions
variable (reflection : (ℝ × ℝ) → (ℝ × ℝ)) -- The reflection function

-- Define the specific vectors from the problem
def u := vec2 2 (-3)
def v := vec2 8 3
def w := vec2 5 1

-- Define the midpoint condition used in the reflection logic 
def midpoint (a b : ℝ × ℝ) : ℝ × ℝ := ⟨(a.1 + b.1) / 2, (a.2 + b.2) / 2⟩

theorem reflection_proof : midpoint u v = ⟨5, 0⟩ ∧ reflection u = v → reflection w = ⟨1, 5⟩ :=
by
  intro h
  sorry -- proof to be completed

end reflection_proof_l702_702058


namespace monotonicity_l702_702335

-- Define f(x) and conditions
def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f x = f (-x)
axiom periodic_f : ∀ x : ℝ, f x = f (2 - x)
axiom decreasing_f : ∀ x y : ℝ, 1 ≤ x → x ≤ y → y ≤ 2 → f(x) ≥ f(y)

-- Theorem: monotonicity of f on given intervals
theorem monotonicity : 
  (∀ x y : ℝ, -2 ≤ x → x ≤ y → y ≤ -1 → f(x) ≤ f(y)) ∧
  (∀ x y : ℝ, 3 ≤ x → x ≤ y → y ≤ 4 → f(x) ≥ f(y)) :=
by sorry

end monotonicity_l702_702335


namespace correct_number_of_conclusions_l702_702317

theorem correct_number_of_conclusions :
  ∃ a : ℝ, (|a|=0→∃a : ℝ, (|a|≠0→2 = 2 ∧ b: = ∃a : ℝ, (|a|=1/4→\2 ≠2)) :=
begin
  sorry
end

end correct_number_of_conclusions_l702_702317


namespace arrangement_without_adjacent_girls_arrangement_with_constraints_arrangement_with_boys_together_arrangement_with_ordered_abc_l702_702234

-- Proof Problem (I)
theorem arrangement_without_adjacent_girls (boys girls : ℕ) (boys = 4) (girls = 2) 
: ∃ (arrangements : ℕ), arrangements = 480 := 
by sorry

-- Proof Problem (II)
theorem arrangement_with_constraints (total_individuals : ℕ) (boys girls : ℕ) (A B : ℕ)
(total_individuals = 6) (boys = 4) (girls = 2) 
(A_not_left_end B_not_right_end : Prop) 
: ∃ (arrangements : ℕ), arrangements = 504 := 
by sorry

-- Proof Problem (III)
theorem arrangement_with_boys_together (boys girls : ℕ) (boys = 4) (girls = 2) 
: ∃ (arrangements : ℕ), arrangements = 144 := 
by sorry

-- Proof Problem (IV)
theorem arrangement_with_ordered_abc (total_individuals : ℕ) (A B C : ℕ)
(total_individuals = 6)
: ∃ (arrangements : ℕ), arrangements = 120 := 
by sorry

end arrangement_without_adjacent_girls_arrangement_with_constraints_arrangement_with_boys_together_arrangement_with_ordered_abc_l702_702234


namespace no_perfect_square_in_base_131_l702_702210

theorem no_perfect_square_in_base_131 : ∀ n ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12}, 
    ¬ ∃ k : ℤ, n^2 + 3*n + 1 = k^2 := by
  sorry

end no_perfect_square_in_base_131_l702_702210


namespace max_value_of_f_l702_702653

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_of_f : ∃ x > 0, (∀ y > 0, f y ≤ f x) ∧ f x = 1 / Real.exp 1 :=
by
  use Real.exp 1
  split
  . exact Real.exp_pos 1
  . split
    . intros y hy
      sorry -- proof of (∀ y > 0, f y ≤ f x)
    . rw [f, Real.log_exp]
      exact one_div (Real.exp 1)

end max_value_of_f_l702_702653


namespace jo_and_max_sum_difference_l702_702266

def sum_arithmetic_series (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_5 (n : ℕ) : ℕ := 
  if n % 5 < 3 then n - (n % 5) else n + (5 - n % 5)

def max_rounded_sum (n : ℕ) : ℕ :=
  (finset.range (n + 1)).sum (λ x, round_to_nearest_5 x)

theorem jo_and_max_sum_difference : 
  |sum_arithmetic_series 60 - max_rounded_sum 60| = 1650 :=
begin
  sorry
end

end jo_and_max_sum_difference_l702_702266


namespace g_cases_g_minimum_value_l702_702989

def f (x : ℝ) : ℝ := x^2 - 4 * x - 4

def g (t : ℝ) : ℝ :=
  if t < 1 then t^2 - 2 * t - 7
  else if 1 ≤ t ∧ t ≤ 2 then -8
  else t^2 - 4 * t - 4

theorem g_cases (t : ℝ) :
  (t < 1 → g(t) = t^2 - 2 * t - 7) ∧
  (1 ≤ t ∧ t ≤ 2 → g(t) = -8) ∧
  (t > 2 → g(t) = t^2 - 4 * t - 4) :=
sorry

theorem g_minimum_value : ∀ t : ℝ, g(t) ≥ -8 :=
sorry

end g_cases_g_minimum_value_l702_702989


namespace polynomial_factorization_l702_702334

theorem polynomial_factorization :
  ∀ (a b c : ℝ),
    a * (b - c) ^ 4 + b * (c - a) ^ 4 + c * (a - b) ^ 4 =
    (a - b) * (b - c) * (c - a) * (a + b + c) :=
  by
    intro a b c
    sorry

end polynomial_factorization_l702_702334


namespace coda_password_combinations_l702_702770

open BigOperators

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13 ∨ n = 17 ∨ n = 19 
  ∨ n = 23 ∨ n = 29

def is_power_of_two (n : ℕ) : Prop :=
  n = 2 ∨ n = 4 ∨ n = 8 ∨ n = 16

def is_multiple_of_three (n : ℕ) : Prop :=
  n % 3 = 0 ∧ n ≥ 1 ∧ n ≤ 30

def count_primes : ℕ :=
  10
def count_powers_of_two : ℕ :=
  4
def count_multiples_of_three : ℕ :=
  10

theorem coda_password_combinations : count_primes * count_powers_of_two * count_multiples_of_three = 400 := by
  sorry

end coda_password_combinations_l702_702770


namespace square_free_of_not_dividing_any_square_free_does_not_imply_prime_l702_702907

theorem square_free_of_not_dividing_any (n : ℕ) (h1 : n > 2)
    (h2 : ∀ k : ℕ, 2 ≤ k ∧ k < n → ¬(n ∣ k^n - 1)) : square_free n :=
sorry

theorem square_free_does_not_imply_prime (n : ℕ) (h_square_free : square_free n) : 
  n = 6 ∨ ¬(∀ m : ℕ, 1 < m → m < n → ¬prime m) :=
sorry

end square_free_of_not_dividing_any_square_free_does_not_imply_prime_l702_702907


namespace major_axis_length_l702_702053

theorem major_axis_length (r : ℝ) (h₁ : r = 3)
  (minor_axis : ℝ) (major_axis : ℝ)
  (h₂ : minor_axis = 2 * r)
  (h₃ : major_axis = minor_axis * 1.60) :
  major_axis = 9.6 :=
by
  have h_minor : minor_axis = 6 := by rw [h₁, h₂]; norm_num
  have h_major : major_axis = 6 * 1.60 := by rw [h_minor, h₃]
  rw [h_major]; norm_num

end major_axis_length_l702_702053


namespace sara_spent_on_rented_movie_l702_702618

def total_spent_on_movies : ℝ := 36.78
def spent_on_tickets : ℝ := 2 * 10.62
def spent_on_bought_movie : ℝ := 13.95

theorem sara_spent_on_rented_movie : 
  (total_spent_on_movies - spent_on_tickets - spent_on_bought_movie = 1.59) := 
by sorry

end sara_spent_on_rented_movie_l702_702618


namespace chef_dressing_total_volume_l702_702721

theorem chef_dressing_total_volume :
  ∀ (V1 V2 : ℕ) (P1 P2 : ℕ) (total_amount : ℕ),
    V1 = 128 →
    V2 = 128 →
    P1 = 8 →
    P2 = 13 →
    total_amount = V1 + V2 →
    total_amount = 256 :=
by
  intros V1 V2 P1 P2 total_amount hV1 hV2 hP1 hP2 h_total
  rw [hV1, hV2, add_comm, add_comm] at h_total
  exact h_total

end chef_dressing_total_volume_l702_702721


namespace cost_hour_excess_is_1point75_l702_702332

noncomputable def cost_per_hour_excess (x : ℝ) : Prop :=
  let total_hours := 9
  let initial_cost := 15
  let excess_hours := total_hours - 2
  let total_cost := initial_cost + excess_hours * x
  let average_cost_per_hour := 3.0277777777777777
  (total_cost / total_hours) = average_cost_per_hour

theorem cost_hour_excess_is_1point75 : cost_per_hour_excess 1.75 :=
by
  sorry

end cost_hour_excess_is_1point75_l702_702332


namespace selection_count_l702_702760

theorem selection_count :
  (Nat.choose 6 3) * (Nat.choose 5 2) = 200 := 
sorry

end selection_count_l702_702760


namespace a_interval_l702_702908

-- Define the function F
def F (x y a : ℝ) : ℝ := x + y - a * (2 * real.sqrt (3 * x * y) + x)

-- Define the condition for a
theorem a_interval (a : ℝ) (x0 : ℝ) (hx0 : 0 < x0) (h : F x0 3 a = 3) : 0 < a ∧ a < 1 :=
by
  -- Proof will be inserted here
  sorry

end a_interval_l702_702908


namespace tangent_lines_from_point_to_circle_l702_702135

theorem tangent_lines_from_point_to_circle : 
  ∀ (P : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ), 
  P = (2, 3) → C = (1, 1) → r = 1 → 
  (∃ k : ℝ, ((3 : ℝ) * P.1 - (4 : ℝ) * P.2 + 6 = 0) ∨ (P.1 = 2)) :=
by
  intros P C r hP hC hr
  sorry

end tangent_lines_from_point_to_circle_l702_702135


namespace area_of_regular_octagon_l702_702312

-- Define a regular octagon with given diagonals
structure RegularOctagon where
  d_max : ℝ  -- length of the longest diagonal
  d_min : ℝ  -- length of the shortest diagonal

-- Theorem stating that the area of the regular octagon
-- is the product of its longest and shortest diagonals
theorem area_of_regular_octagon (O : RegularOctagon) : 
  let A := O.d_max * O.d_min
  A = O.d_max * O.d_min :=
by
  -- Proof to be filled in
  sorry

end area_of_regular_octagon_l702_702312


namespace combined_shape_area_l702_702442

theorem combined_shape_area (s : ℝ) (h : s = 4) : 
  let A_square := s * s,
      A_triangle := (real.sqrt 3 / 4) * s * s,
      A_total := A_square + A_triangle
  in A_total = 16 + 4 * real.sqrt 3 := 
by
  sorry

end combined_shape_area_l702_702442


namespace coffee_shop_distance_l702_702404

theorem coffee_shop_distance (resort_distance mall_distance : ℝ) 
  (coffee_dist : ℝ)
  (h_resort_distance : resort_distance = 400) 
  (h_mall_distance : mall_distance = 700)
  (h_equidistant : ∀ S, (S - resort_distance) ^ 2 + resort_distance ^ 2 = S ^ 2 ∧ 
  (mall_distance - S) ^ 2 + resort_distance ^ 2 = S ^ 2 → coffee_dist = S):
  coffee_dist = 464 := 
sorry

end coffee_shop_distance_l702_702404


namespace sum_first_seven_arithmetic_l702_702891

theorem sum_first_seven_arithmetic (a : ℕ) (d : ℕ) (h : a + 3 * d = 3) :
    let a1 := a
    let a2 := a + d
    let a3 := a + 2 * d
    let a4 := a + 3 * d
    let a5 := a + 4 * d
    let a6 := a + 5 * d
    let a7 := a + 6 * d
    a1 + a2 + a3 + a4 + a5 + a6 + a7 = 21 :=
by
  sorry

end sum_first_seven_arithmetic_l702_702891


namespace sum_b_c_d_eq_four_l702_702088

noncomputable def a : ℕ → ℝ
| n := 2 ⌊ (n : ℝ) ^ (1/3) ⌋ + 2

theorem sum_b_c_d_eq_four : ∃ b c d : ℤ, (∀ n : ℕ, n > 0 → a n = b * ⌊ (n : ℝ)^(1/3) ⌋ + d) ∧ b + c + d = 4 :=
by
  -- Definitions of b, c, and d
  use [2, 0, 2]
  -- Proof that the sequence holds for given b, c, and d
  split
  { intro n
    intro hn
    -- Using the given sequence and equation
    sorry
  }
  -- Sum of b, c, and d equals 4
  calc
    2 + 0 + 2 = 4 : by norm_num

end sum_b_c_d_eq_four_l702_702088


namespace max_min_diff_half_dollars_l702_702944

-- Definitions based only on conditions
variables (a c d : ℕ)

-- Conditions:
def condition1 : Prop := a + c + d = 60
def condition2 : Prop := 5 * a + 25 * c + 50 * d = 1000

-- The mathematically equivalent proof statement
theorem max_min_diff_half_dollars : condition1 a c d → condition2 a c d → (∃ d_min d_max : ℕ, d_min = 0 ∧ d_max = 15 ∧ d_max - d_min = 15) :=
by
  intros
  sorry

end max_min_diff_half_dollars_l702_702944


namespace round_trip_ticket_percent_l702_702694

variable (P R : ℕ)

/--
  Condition 1: 40% of the passengers held round-trip tickets and took their cars aboard the ship.
  Condition 2: 50% of the passengers with round-trip tickets did not take their cars aboard the ship.
  Question: What percent of the ship’s passengers held round-trip tickets?
-/
theorem round_trip_ticket_percent (h1 : 0.4 * P = 0.5 * R) : R = 0.8 * P := 
sorry

end round_trip_ticket_percent_l702_702694


namespace josh_spent_after_drink_l702_702268

theorem josh_spent_after_drink
  (initial_amount : ℝ)
  (drink_cost : ℝ)
  (remaining_amount : ℝ)
  (amount_left_after_all_spending : ℝ) :
  initial_amount = 9 →
  drink_cost = 1.75 →
  remaining_amount = 6 →
  amount_left_after_all_spending = initial_amount - drink_cost - 1.25 →
  9 - 1.75 - 1.25 = 6 :=
begin
  sorry
end

end josh_spent_after_drink_l702_702268


namespace triangle_ABC_AX_perpendicular_BC_l702_702663

theorem triangle_ABC_AX_perpendicular_BC :
  ∀ (A B C X : Type) 
    (angle_A : ℕ) (angle_B : ℕ)
    (angle_XBA : ℕ) (angle_XCA : ℕ),
    angle_A = 40 ∧ angle_B = 60 ∧ 
    angle_XBA = 20 ∧ angle_XCA = 10 →
    is_perpendicular (line_through A X) (line_through B C) :=
begin
  intros A B C X angle_A angle_B angle_XBA angle_XCA h,
  sorry
end

end triangle_ABC_AX_perpendicular_BC_l702_702663


namespace floor_divisibility_l702_702626

theorem floor_divisibility (n : ℕ) :
  let x := (1 + Real.sqrt 3) ^ (2 * n + 1)
  in ⌊x⌋ % 2^(n + 1) = 0 ∧ ⌊x⌋ % 2^(n + 2) ≠ 0 :=
by
  let x := (1 + Real.sqrt 3) ^ (2 * n + 1)
  sorry

end floor_divisibility_l702_702626


namespace line_perpendicular_to_plane_l702_702827

-- Define the direction vector of line l
def a : ℝ × ℝ × ℝ := (1, 1, 2)

-- Define the normal vector of plane α
def n : ℝ × ℝ × ℝ := (2, 2, 4)

-- Define a line is perpendicular to a plane if the direction vector of the line is parallel to the normal vector of the plane
def is_perpendicular (a n : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, n = (k • a)

-- Theorem statement
theorem line_perpendicular_to_plane (h : n = 2 • a) : is_perpendicular a n :=
by
  simp [is_perpendicular]
  exists 2
  exact h

end line_perpendicular_to_plane_l702_702827


namespace infinitely_many_solutions_l702_702984

def sum_of_first_n (x : ℕ) : ℕ := x * (x + 1) / 2

theorem infinitely_many_solutions : 
  ∃ (f : ℕ → ℕ × ℕ), 
  ∀ k : ℕ, 
  let (x_k, y_k) := f k in 
  sum_of_first_n x_k = y_k * y_k := 
sorry

end infinitely_many_solutions_l702_702984


namespace obtuse_triangle_AC_l702_702884

-- Assuming triangle type and notation for obtuse triangle ABC with given lengths and area
variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Condition: Area of triangle ABC is 1/2
-- AB length is 1, BC length is sqrt(2)
-- We need to prove AC equals sqrt(5)

noncomputable def area (a b c : ℝ) : ℝ :=
  (1 / 2) * a * b -- simplified for this example

theorem obtuse_triangle_AC :
  ∀ (AB BC AC : ℝ),
  area AB BC AC = 1 / 2 ∧
  AB = 1 ∧
  BC = real.sqrt 2 ∧
  ∃ B,
  ∠B = 1 / 2 ∧ -- ∠B is obtuse
  ∠A + ∠B + ∠C = π →
  AC = real.sqrt 5 :=
by sorry

end obtuse_triangle_AC_l702_702884


namespace matrix_eigenvalue_problem_l702_702116

theorem matrix_eigenvalue_problem (k : ℝ) (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) :
  ((3*x + 4*y = k*x) ∧ (6*x + 3*y = k*y)) → k = 3 :=
by
  sorry

end matrix_eigenvalue_problem_l702_702116


namespace show_revenue_l702_702742

theorem show_revenue (a b t : ℕ) (c : ℕ → ℕ → ℕ) (total_revenue : ℕ → ℤ) :
  a = 200 →
  b = 3 * a →
  t = 25 →
  total_revenue a = (a + b) * t →
  total_revenue a = 20000 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end show_revenue_l702_702742


namespace clothing_removed_l702_702348

theorem clothing_removed :
  ∃ (B C E B_ratio C_ratio B_new_ratio C_new) (B_to_C_new_ratio : ℚ),
  (B_ratio = 7 ∧ C_ratio = 4 ∧ E = 12 ∧ B_new_ratio = 14 ∧ C_new = 4) ∧
  (B = (7 / 3) * E) ∧
  (C = (4 / 3) * E) ∧
  (C_after_removal = 8) ∧
  (B_new_ratio / 2 = B_to_C_new_ratio) ∧
  (B_to_C_new_ratio = B / C_after_removal) →
  (C - C_after_removal) = 8 :=
begin
  sorry
end

end clothing_removed_l702_702348


namespace questionnaire_visitors_l702_702018

noncomputable def total_visitors :=
  let V := 600
  let E := (3 / 4) * V
  V

theorem questionnaire_visitors:
  ∃ (V : ℕ), V = 600 ∧
  (∀ (E : ℕ), E = (3 / 4) * V ∧ E + 150 = V) :=
by
    use 600
    sorry

end questionnaire_visitors_l702_702018


namespace satisfying_lines_l702_702381

theorem satisfying_lines (x y : ℝ) : (y^2 - 2*y = x^2 + 2*x) ↔ (y = x + 2 ∨ y = -x) :=
by
  sorry

end satisfying_lines_l702_702381


namespace jar_total_value_l702_702047

def total_value_in_jar (p n q : ℕ) (total_coins : ℕ) (value : ℝ) : Prop :=
  p + n + q = total_coins ∧
  n = 3 * p ∧
  q = 4 * n ∧
  value = p * 0.01 + n * 0.05 + q * 0.25

theorem jar_total_value (p : ℕ) (h₁ : 16 * p = 240) : 
  ∃ value, total_value_in_jar p (3 * p) (12 * p) 240 value ∧ value = 47.4 :=
by
  sorry

end jar_total_value_l702_702047


namespace BC_length_correct_l702_702892

variables {A B C D : Type}
variables [inner_product_space ℝ P] (A B C D : P)

noncomputable def BC_length (A B C D : P) : ℝ :=
  let CD : ℝ := 20
  let AC : ℝ := CD * 2
  let AB : ℝ := AC / 2.5
  sqrt (AC ^ 2 + AB ^ 2)

theorem BC_length_correct (h1 : ∥C - D∥ = 20) (h2 : angle D C A = π / 2) (h3 : (A - C).angle (B - A) = π / 2) (h4 : tan (angle D C A) = 2) (h5 : tan (angle A B A) = 2.5) :
  BC_length A B C D = sqrt 1856 :=
begin
  sorry
end

end BC_length_correct_l702_702892


namespace frog_five_jumps_within_one_point_five_meters_l702_702417

noncomputable def frog_jump_probability : ℚ :=
  1 / 8

theorem frog_five_jumps_within_one_point_five_meters :
  ∀ (u : ℕ → ℝ × ℝ), 
    (∀ i, i < 5 → ∥u i∥ = 1) →
    (random_walk_dist u 0 4 ≤ 1.5) →
    (probability (random_walk_within_distance u 5 1.5) = frog_jump_probability) :=
sorry

end frog_five_jumps_within_one_point_five_meters_l702_702417


namespace not_geometric_sequence_range_x_y_l702_702351

noncomputable def a (n : ℕ) : ℕ :=
  if n = 0 then 2 else 2^(n + 1)

def is_geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, ∀ n : ℕ, a (n + 1) = r * a n

theorem not_geometric_sequence : ¬ is_geometric_sequence a :=
sorry

noncomputable def b (n : ℕ) : ℕ :=
  if n = 0 then 1 else n + 1

theorem range_x_y (x y : ℝ) :
  (∀ n : ℕ, x < (Finset.range n).sum (λ k, 1 / (b k * b (k + 1))) ∧
  (Finset.range n).sum (λ k, 1 / (b k * b (k + 1))) < y) →
  x < 1 / 3 ∧ y ≥ 2 / 3 :=
sorry

end not_geometric_sequence_range_x_y_l702_702351


namespace Janet_pages_per_day_l702_702220

variable (J : ℕ)

-- Conditions
def belinda_pages_per_day : ℕ := 30
def janet_extra_pages_per_6_weeks : ℕ := 2100
def days_in_6_weeks : ℕ := 42

-- Prove that Janet reads 80 pages a day
theorem Janet_pages_per_day (h : J * days_in_6_weeks = (belinda_pages_per_day * days_in_6_weeks) + janet_extra_pages_per_6_weeks) : J = 80 := 
by sorry

end Janet_pages_per_day_l702_702220


namespace work_days_l702_702695

theorem work_days (p_can : ℕ → ℝ) (q_can : ℕ → ℝ) (together_can: ℕ → ℝ) :
  (together_can 6 = 1) ∧ (q_can 10 = 1) → (1 / (p_can x) + 1 / (q_can 10) = 1 / (together_can 6)) → (x = 15) :=
by
  sorry

end work_days_l702_702695


namespace minimum_days_to_plant_100_trees_l702_702739

theorem minimum_days_to_plant_100_trees :
  ∃ n : ℕ+, (2 * (2 ^ n - 1)) / (2 - 1) ≥ 100 ∧
           ∀ k : ℕ+, k < n → (2 * (2 ^ k - 1)) / (2 - 1) < 100 :=
begin
  -- The proof is omitted as requested
  sorry
end

end minimum_days_to_plant_100_trees_l702_702739


namespace debate_team_group_size_l702_702349

theorem debate_team_group_size (boys girls groups : ℕ) (h_boys : boys = 11) (h_girls : girls = 45) (h_groups : groups = 8) : 
  (boys + girls) / groups = 7 := by
  sorry

end debate_team_group_size_l702_702349


namespace sum_of_roots_eq_zero_l702_702999

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * Real.pi * Real.tan x)

theorem sum_of_roots_eq_zero :
  ∀ x ∈ Ioo (-Real.pi / 2) (Real.pi / 2),
  f x = 0 → (∑ x in (roots f (Ioo (-Real.pi / 2) (Real.pi / 2))), x) = 0 := 
sorry

end sum_of_roots_eq_zero_l702_702999


namespace swimming_pool_width_l702_702056

theorem swimming_pool_width
    (area : ℕ)
    (length : ℕ)
    (h1 : area = 30)
    (h2 : length = 10) 
    : (area / length = 3) := 
by 
    simp [h1, h2]
    exact (30 / 10)

end swimming_pool_width_l702_702056


namespace a_83_a_exp_B_exp_l702_702815

noncomputable def a (i j : ℕ) : ℚ :=
  if i > 0 ∧ j > 0 ∧ i ≥ j then (i / 4) * (1 / 2)^(j - 1) else 0

theorem a_83 : a 8 3 = 1 / 2 := 
by
  sorry

theorem a_exp (i j : ℕ) (h : i > 0 ∧ j > 0 ∧ i ≥ j) : a i j = (i / 4) * (1 / 2)^(j - 1) := 
by
  sorry

noncomputable def A (n : ℕ) : ℚ :=
  if n > 0 then (n / 2) - (n / 2^(n + 1)) else 0

noncomputable def B (m : ℕ) : ℚ :=
  ∑ i in finset.range (m + 1), A i

theorem B_exp (m : ℕ) : B m = (m^2 + m - 4) / 4 + (m + 2) / 2^(m + 1) := 
by
  sorry

end a_83_a_exp_B_exp_l702_702815


namespace star_is_addition_l702_702719

variable {α : Type} [AddCommGroup α]

-- Define the binary operation star
variable (star : α → α → α)

-- Define the condition given in the problem
axiom star_condition : ∀ (a b c : α), star (star a b) c = a + b + c

-- Prove that star is the same as usual addition
theorem star_is_addition : ∀ (a b : α), star a b = a + b :=
  sorry

end star_is_addition_l702_702719


namespace length_of_AB_l702_702254

theorem length_of_AB (AB BC CD : ℝ) 
  (h1 : CD = 15) 
  (h2 : tan B = 2.4) 
  (h3 : tan D = 1.2)
  (h4 : BC = CD * tan D) 
  (h5 : AB = BC / tan B) : 
  AB = 7.5 := 
by 
  sorry

end length_of_AB_l702_702254


namespace arithmetic_seq_problem_l702_702928

theorem arithmetic_seq_problem
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, a (n + 1) - a n = d)
  (h2 : d > 0)
  (h3 : a 1 + a 2 + a 3 = 15)
  (h4 : a 1 * a 2 * a 3 = 80) :
  a 11 + a 12 + a 13 = 105 :=
sorry

end arithmetic_seq_problem_l702_702928


namespace olympiad_divisors_l702_702976

theorem olympiad_divisors :
  {n : ℕ | n > 0 ∧ n ∣ (1998 + n)} = {n : ℕ | n > 0 ∧ n ∣ 1998} :=
by {
  sorry
}

end olympiad_divisors_l702_702976


namespace coeff_sum_eq_l702_702859

noncomputable def a : ℤ := (1 : ℤ)
noncomputable def a1 : ℤ := 14 * (2 : ℤ)^13 - 2*a + 1*a2  -- this will eventually be derived based on x=1 substitution etc.
-- Define all coefficients up to a14.
noncomputable def a2 : ℤ := 1  -- place holder

-- Similarly, all other a_3 to a_14 would be defined.

theorem coeff_sum_eq :
  let a1 := 14 * (2 : ℤ)^13 - 2*a + 1*a2 in -- subsitution will be further broken down in definitions
  a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 + 6 * a6 + ... 13 * a13 + 14 * a14 = 7 * (2 : ℤ)^14 :=
  sorry

end coeff_sum_eq_l702_702859


namespace parameterization_of_line_l702_702652

theorem parameterization_of_line (t : ℝ) (g : ℝ → ℝ) 
  (h : ∀ t, (g t - 10) / 2 = t ) :
  g t = 5 * t + 10 := by
  sorry

end parameterization_of_line_l702_702652


namespace vertical_asymptote_at_neg2_l702_702788

-- Given the function definition
def f (x : ℝ) : ℝ := (x^2 + 6*x + 9) / (x + 2)

-- Proof obligation: There is a vertical asymptote at x = -2
theorem vertical_asymptote_at_neg2 : 
  (∃ (x : ℝ), f x = (x^2 + 6*x + 9) / (x + 2) ∧ x = -2) ∧ 
  ((x + 2) = 0 ∧ (x^2 + 6*x + 9) ≠ 0) :=
sorry

end vertical_asymptote_at_neg2_l702_702788


namespace range_of_f_div_a_l702_702873

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a*x^2 + b*x + c

theorem range_of_f_div_a (a b c : ℝ) (h1 : 0 < a)
  (h2 : 1 < -b / (2 * a))
  (h3 : -b / (2 * a) < 2)
  (h4 : a + b + c ≥ 0)
  (h5 : 4 * a + 2 * b + c ≥ 0)
  (h6 : 4 * a * c - b^2 < 0) :
  ∃ (y : ℝ), y ∈ set.Ico 0 1 ∧ y = (1 + b / a + c / a) :=
sorry

end range_of_f_div_a_l702_702873


namespace cover_cube_with_rectangles_l702_702492

theorem cover_cube_with_rectangles (n : ℕ) : 
  (∃ (cover : (Σ (x y z : ℕ), x = 2 ∧ y = 2 ∧ z = 2 → Prop) → Prop), 
  (∀ rect, rect ∈ cover → (rect.borders.count (λ r, r ∈ cover) = 5))) ↔ (n % 2 = 0) :=
sorry

end cover_cube_with_rectangles_l702_702492


namespace number_of_integer_pairs_l702_702124

theorem number_of_integer_pairs (m n : ℤ) (h : m + n = 2 * m * n) :
  ({(m, n) ∈ (ℤ × ℤ) | m + n = 2 * m * n}.card = 2) :=
sorry

end number_of_integer_pairs_l702_702124


namespace triangle_incenter_inequality_l702_702909

theorem triangle_incenter_inequality {A B C I P : Point}
  (h_incenter : is_incenter I A B C)
  (h_angle_eq : ∠PBA + ∠PCA = ∠PBC + ∠PCB) :
  AP ≥ AI ∧ (AP = AI ↔ P = I) := 
sorry

end triangle_incenter_inequality_l702_702909


namespace area_of_regular_octagon_l702_702313

-- Define a regular octagon with given diagonals
structure RegularOctagon where
  d_max : ℝ  -- length of the longest diagonal
  d_min : ℝ  -- length of the shortest diagonal

-- Theorem stating that the area of the regular octagon
-- is the product of its longest and shortest diagonals
theorem area_of_regular_octagon (O : RegularOctagon) : 
  let A := O.d_max * O.d_min
  A = O.d_max * O.d_min :=
by
  -- Proof to be filled in
  sorry

end area_of_regular_octagon_l702_702313


namespace unoccupied_volume_l702_702672

noncomputable def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * π * r^2 * h

theorem unoccupied_volume :
  let r_cylinder := 8
  let h_cylinder := 30
  let r_cone := 8
  let h_cone := 10 in
  let V_cylinder := cylinder_volume r_cylinder h_cylinder in
  let V_cone := cone_volume r_cone h_cone in
  let V_total_cones := 2 * V_cone in
  V_cylinder - V_total_cones = (4480 * π) / 3 :=
by
  sorry

end unoccupied_volume_l702_702672


namespace number_of_men_in_club_l702_702415

variables (M W : ℕ)

theorem number_of_men_in_club 
  (h1 : M + W = 30) 
  (h2 : (1 / 3 : ℝ) * W + M = 18) : 
  M = 12 := 
sorry

end number_of_men_in_club_l702_702415


namespace sum_interest_l702_702062

noncomputable def simple_interest (P : ℝ) (R : ℝ) := P * R * 3 / 100

theorem sum_interest (P R : ℝ) (h : simple_interest P (R + 1) - simple_interest P R = 75) : P = 2500 :=
by
  sorry

end sum_interest_l702_702062


namespace visited_neither_l702_702236

theorem visited_neither (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) 
  (h1 : total = 100) 
  (h2 : iceland = 55) 
  (h3 : norway = 43) 
  (h4 : both = 61) : 
  (total - (iceland + norway - both)) = 63 := 
by 
  sorry

end visited_neither_l702_702236


namespace max_size_of_irrational_set_l702_702741

theorem max_size_of_irrational_set (S : Set ℝ) (hS : ∀ T ⊆ S, T.card = 5 → ∃ a b ∈ T, ¬(a + b).is_rational)
  : S.card ≤ 8 :=
sorry

end max_size_of_irrational_set_l702_702741


namespace num_three_letter_initials_l702_702208

theorem num_three_letter_initials : 
  let vowels := ['A', 'E', 'I']
  let consonants := ['B', 'C', 'D', 'F', 'G', 'H', 'J']
  let total_vowels := 3
  let total_consonants := 7
  let choose_1_vowel := total_vowels
  let choose_2_consonants := Nat.choose total_consonants 2
  let arrangements := Nat.factorial 3
  choose_1_vowel * choose_2_consonants * arrangements = 378 :=
by
  let vowels := ['A', 'E', 'I']
  let consonants := ['B', 'C', 'D', 'F', 'G', 'H', 'J']
  let total_vowels := 3
  let total_consonants := 7
  let choose_1_vowel := total_vowels
  let choose_2_consonants := Nat.choose total_consonants 2
  let arrangements := Nat.factorial 3
  have h_vowels : choose_1_vowel = 3 := by rfl
  have h_consonants : choose_2_consonants = 21 := by
    calc
      choose_2_consonants
      _ = Nat.choose 7 2 : by rfl
      _ = 21 : by norm_num
  have h_arrangements : arrangements = 6 := by
    calc
      arrangements
      _ = Nat.factorial 3 : by rfl
      _ = 6 : by norm_num
  calc
    choose_1_vowel * choose_2_consonants * arrangements
    _ = 3 * 21 * 6 : by rw [h_vowels, h_consonants, h_arrangements]
    _ = 378 : by norm_num

end num_three_letter_initials_l702_702208


namespace min_points_to_guarantee_win_l702_702556

theorem min_points_to_guarantee_win (P Q R S: ℕ) (bonus: ℕ) :
    (P = 6 ∨ P = 4 ∨ P = 2) ∧ (Q = 6 ∨ Q = 4 ∨ Q = 2) ∧ 
    (R = 6 ∨ R = 4 ∨ R = 2) ∧ (S = 6 ∨ S = 4 ∨ S = 2) →
    (bonus = 3 ↔ ((P = 6 ∧ Q = 4 ∧ R = 2) ∨ (P = 6 ∧ Q = 2 ∧ R = 4) ∨ 
                   (P = 4 ∧ Q = 6 ∧ R = 2) ∨ (P = 4 ∧ Q = 2 ∧ R = 6) ∨ 
                   (P = 2 ∧ Q = 6 ∧ R = 4) ∨ (P = 2 ∧ Q = 4 ∧ R = 6))) →
    (P + Q + R + S + bonus ≥ 24) :=
by sorry

end min_points_to_guarantee_win_l702_702556


namespace a_older_than_b_l702_702420

theorem a_older_than_b (A B : ℕ) (h1 : B = 36) (h2 : A + 10 = 2 * (B - 10)) : A - B = 6 :=
  sorry

end a_older_than_b_l702_702420


namespace complete_packages_count_l702_702852

-- Definitions for initial and remaining items
def initial_cupcakes : ℕ := 20
def initial_cookies : ℕ := 15
def ate_cupcakes : ℕ := 11
def ate_cookies : ℝ := 7.5

-- Definitions for package contents
def cupcakes_per_package : ℕ := 3
def cookies_per_package : ℕ := 2

-- Definitions for remaining items
def remaining_cupcakes : ℕ := initial_cupcakes - ate_cupcakes
def remaining_cookies : ℝ := initial_cookies - ate_cookies

-- Proof statement
theorem complete_packages_count : min (remaining_cupcakes / cupcakes_per_package) (remaining_cookies / cookies_per_package) = 3 :=
by
  sorry

end complete_packages_count_l702_702852


namespace find_principal_amount_l702_702120

theorem find_principal_amount
  (CI : ℝ := 2522.0000000000036)
  (r : ℝ := 0.20)
  (t : ℝ := 9 / 12)
  (n : ℝ := 4)
  (A : ℝ := CI + P)
  (P : ℝ) :
  let total_interest := (1 + (r / n))^(n * t)
  CI + P = P * total_interest → 
  P ≈ 16000 :=
by
  sorry 

end find_principal_amount_l702_702120


namespace log9_y_l702_702222

theorem log9_y (y : ℝ) (h : y = (Real.log 3 / Real.log 27) ^ (Real.log 81 / Real.log 3)) : Real.log 9 y = -2 := 
sorry

end log9_y_l702_702222


namespace reporters_percentage_l702_702798

theorem reporters_percentage (total_reporters : ℕ) (h1 : 35% of total_reporters cover local politics)
  (h2 : 50% of total_reporters do not cover politics) :
  30% of reporters who cover politics do not cover local politics in country X :=
sorry

end reporters_percentage_l702_702798


namespace symmetry_about_y_axis_l702_702651

def f (x : ℝ) : ℝ := log 2 (abs x)

theorem symmetry_about_y_axis : ∀ (x : ℝ), f (-x) = f x :=
by
  intros x
  sorry

end symmetry_about_y_axis_l702_702651


namespace average_of_three_numbers_l702_702624

theorem average_of_three_numbers (x : ℝ) 
  (h1 : 2 * x = 2x)
  (h2 : 2 * x / 3 = (2 * x) / 3)
  (h3 : 2 * x - (2 * x / 3) = 96) :
  (2 * x + x + (2 * x / 3)) / 3 = 88 := 
sorry

end average_of_three_numbers_l702_702624


namespace eigenvalues_of_2x2_matrix_l702_702108

theorem eigenvalues_of_2x2_matrix :
  ∃ (k : ℝ), (k = 3 + 4 * Real.sqrt 6 ∨ k = 3 - 4 * Real.sqrt 6) ∧
            ∃ (v : ℝ × ℝ), v ≠ (0, 0) ∧
            ((3 : ℝ) * v.1 + 4 * v.2 = k * v.1 ∧ (6 : ℝ) * v.1 + 3 * v.2 = k * v.2) :=
begin
  sorry
end

end eigenvalues_of_2x2_matrix_l702_702108


namespace problem1_problem2_l702_702772

-- Problem (1)
theorem problem1 : 2^(-1:ℤ) - real.sqrt 3 * real.tan (real.pi/3) + (real.pi - 2011)^0 + abs ((-1)/(2:ℚ)) = -1 := by
  have h_tan60 : real.tan (real.pi / 3) = real.sqrt 3 := sorry
  sorry

-- Problem (2)
theorem problem2 (x : ℝ) : x * (x + 1) - (x + 2) * (x - 2) = x + 4 := by
  sorry

end problem1_problem2_l702_702772


namespace ratio_sheep_to_horses_l702_702445

theorem ratio_sheep_to_horses 
  (sheep : ℕ) 
  (horse_food_per_day_per_horse : ℕ) 
  (total_horse_food_per_day : ℕ) 
  (number_of_sheep : ℕ)
  (number_of_horses : ℕ)
  (h1 : horse_food_per_day_per_horse = 230) 
  (h2 : total_horse_food_per_day = 12880) 
  (h3 : number_of_sheep = 8) 
  (h4 : total_horse_food_per_day = horse_food_per_day_per_horse * number_of_horses) 
  : number_of_sheep / number_of_horses = 1 / 7 :=
by
  rw [h1, h2, h3, h4]
  sorry

end ratio_sheep_to_horses_l702_702445


namespace angle_in_second_quadrant_l702_702217

theorem angle_in_second_quadrant (α : ℝ) :
  (sin α * tan α < 0) ∧ (0 < sin α + cos α ∧ sin α + cos α < 1) →
  π/2 < α ∧ α < π :=
by
  sorry

end angle_in_second_quadrant_l702_702217


namespace eigenvalues_of_2x2_matrix_l702_702106

theorem eigenvalues_of_2x2_matrix :
  ∃ (k : ℝ), (k = 3 + 4 * Real.sqrt 6 ∨ k = 3 - 4 * Real.sqrt 6) ∧
            ∃ (v : ℝ × ℝ), v ≠ (0, 0) ∧
            ((3 : ℝ) * v.1 + 4 * v.2 = k * v.1 ∧ (6 : ℝ) * v.1 + 3 * v.2 = k * v.2) :=
begin
  sorry
end

end eigenvalues_of_2x2_matrix_l702_702106


namespace reflection_angle_sum_l702_702586

open EuclideanGeometry

theorem reflection_angle_sum (ABC : Triangle) (J : Point) (K : Point) (E F : Point) :
  is_excenter_J A ABC ->
  reflection J (line_through B C) = K ->
  (E ∈ line_through B J) ∧ (F ∈ line_through C J) ->
  (\<angle EAB = 90) ∧ (\<angle CAF = 90) ->
  (\<angle FKE + \<angle FJE = 180) :=
  sorry

end reflection_angle_sum_l702_702586


namespace coordinates_of_point_l702_702244

theorem coordinates_of_point : 
  ∀ (x y : ℝ), (x, y) = (2, -3) → (x, y) = (2, -3) := 
by 
  intros x y h 
  exact h

end coordinates_of_point_l702_702244


namespace coordinates_with_respect_to_origin_l702_702247

def point_coordinates (x y : ℤ) : ℤ × ℤ :=
  (x, y)

def origin : ℤ × ℤ :=
  (0, 0)

theorem coordinates_with_respect_to_origin :
  point_coordinates 2 (-3) = (2, -3) := by
  -- placeholder proof
  sorry

end coordinates_with_respect_to_origin_l702_702247


namespace roll_combinations_l702_702403

theorem roll_combinations (total_rolls kinds: ℕ) (at_least_two: ℕ):
  kinds = 4 → total_rolls = 10 → at_least_two = 2 →
  (nat.choose (2 + 4 - 1) (4 - 1)) = 10 := 
by 
  intro hkinds htotal hatleast
  rw [htotal, hkinds, hatleast]
  norm_num
  decide


end roll_combinations_l702_702403


namespace kendall_tau_significance_test_l702_702558

noncomputable def T_critical (n : ℕ) (z_critical : ℝ) : ℝ :=
  z_critical * Real.sqrt (2 * (2 * n + 5) / (9 * n * (n - 1)))

-- Define the conditions
def n : ℕ := 10
def tau : ℝ := 0.47
def alpha : ℝ := 0.05
def z_critical : ℝ := 1.96 -- Pre-determined critical value from standard normal distribution

-- Calculate the critical test statistic
def T_critical_value : ℝ := T_critical n z_critical

-- The proof statement
theorem kendall_tau_significance_test : tau ≤ T_critical_value := 
sorry

end kendall_tau_significance_test_l702_702558


namespace sum_of_distances_is_63_l702_702599

variable (a b c d : ℝ)

def parabola (x : ℝ) : ℝ := x^2

def directrix : ℝ := -1 / 4

-- Given three points of intersection
def intersection_points : List (ℝ × ℝ) := [(-4, parabolatt (-4)), (1, parabola 1), (6, parabola 6), (d, parabola d)]

-- Define distances from the given points to the directrix
def distance_to_directrix (y : ℝ) : ℝ := abs (y + directrix)

def sum_of_distances (points : List (ℝ × ℝ)) : ℝ :=
  points.foldr (λ p acc => acc + distance_to_directrix (prod.snd p)) 0

theorem sum_of_distances_is_63 (h : a + b + c + d = 0) :
  sum_of_distances [(-4, parabola (-4)), (1, parabola 1), (6, parabola 6), (d, parabola d)] = 63 := 
sorry

end sum_of_distances_is_63_l702_702599


namespace miles_run_by_harriet_l702_702126

def miles_run_by_all_runners := 285
def miles_run_by_katarina := 51
def miles_run_by_adriana := 74
def miles_run_by_tomas_tyler_harriet (total_run: ℝ) := (total_run - (miles_run_by_katarina + miles_run_by_adriana))

theorem miles_run_by_harriet : (miles_run_by_tomas_tyler_harriet miles_run_by_all_runners) / 3 = 53.33 := by
  sorry

end miles_run_by_harriet_l702_702126


namespace book_number_combinations_l702_702759

theorem book_number_combinations :
  let first_char_possibilities := 2 in
  let last_two_chars_possibilities := 3 in
  let total_combinations := first_char_possibilities * last_two_chars_possibilities * last_two_chars_possibilities in
  total_combinations = 18 :=
by
  sorry

end book_number_combinations_l702_702759


namespace selection_methods_l702_702753

/-- Type definition for the workers -/
inductive Worker
  | PliersOnly  : Worker
  | CarOnly     : Worker
  | Both        : Worker

/-- Conditions -/
def num_workers : ℕ := 11
def num_pliers_only : ℕ := 5
def num_car_only : ℕ := 4
def num_both : ℕ := 2
def pliers_needed : ℕ := 4
def car_needed : ℕ := 4

/-- Main statement -/
theorem selection_methods : 
  (num_pliers_only + num_car_only + num_both = num_workers) → 
  (num_pliers_only = 5) → 
  (num_car_only = 4) → 
  (num_both = 2) → 
  (pliers_needed = 4) → 
  (car_needed = 4) → 
  ∃ n : ℕ, n = 185 := 
by 
  sorry -- Proof Skipped

end selection_methods_l702_702753


namespace correct_product_l702_702318

-- Assuming a function reverse_digits to reverse digits of a number
def reverse_digits (n : ℕ) : ℕ := 
  let digits := n.digits 10
  digits.reverse.foldl (λ acc d, acc * 10 + d) 0

theorem correct_product (a b : ℕ) (h : 100 ≤ a ∧ a < 1000)
  (b_pos : b > 0)
  (h_erroneous : b * (reverse_digits a - 3) = 245) :
  a * b = 5810 :=
sorry

end correct_product_l702_702318


namespace inscribed_circle_radius_square_l702_702723

variable {r : ℝ}

/- Conditions -/
def ER := 21
def RF := 28
def GS := 40
def SH := 32

theorem inscribed_circle_radius_square :
  let r_sq := r * r in
  (ER + RF) * (GS + SH) = ((ER + RF) * r_sq - ER * RF * SH * GS) :=
by
  sorry

end inscribed_circle_radius_square_l702_702723


namespace fourth_quarter_points_sum_l702_702237

variable (a d b j : ℕ)

-- Conditions from the problem
def halftime_tied : Prop := 2 * a + d = 2 * b
def wildcats_won_by_four : Prop := 4 * a + 6 * d = 4 * b - 3 * j + 4

-- The proof goal to be established
theorem fourth_quarter_points_sum
  (h1 : halftime_tied a d b)
  (h2 : wildcats_won_by_four a d b j) :
  (a + 3 * d) + (b - 2 * j) = 28 :=
sorry

end fourth_quarter_points_sum_l702_702237


namespace angles_in_quadrilateral_l702_702867

theorem angles_in_quadrilateral (A B C D : ℝ)
    (h : A / B = 1 / 3 ∧ B / C = 3 / 5 ∧ C / D = 5 / 6)
    (sum_angles : A + B + C + D = 360) :
    A = 24 ∧ D = 144 := 
by
    sorry

end angles_in_quadrilateral_l702_702867


namespace part_I_solution_set_part_II_min_value_l702_702525

-- Define the function f
def f (x a : ℝ) := 2*|x + 1| - |x - a|

-- Part I: Prove the solution set of f(x) ≥ 0 when a = 2
theorem part_I_solution_set (x : ℝ) :
  f x 2 ≥ 0 ↔ x ≤ -4 ∨ x ≥ 0 :=
sorry

-- Define the function g
def g (x a : ℝ) := f x a + 3*|x - a|

-- Part II: Prove the minimum value of m + n given t = 4 when a = 1
theorem part_II_min_value (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, g x 1 ≥ 4) → (2/m + 1/(2*n) = 4) → m + n = 9/8 :=
sorry

end part_I_solution_set_part_II_min_value_l702_702525


namespace part_one_l702_702837

theorem part_one (m : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = m * Real.exp x - x - 2) :
  (∀ x : ℝ, f x > 0) → m > Real.exp 1 :=
sorry

end part_one_l702_702837


namespace cupcakes_left_over_l702_702951

def total_cupcakes := 40
def ms_delmont_class := 18
def mrs_donnelly_class := 16
def ms_delmont := 1
def mrs_donnelly := 1
def school_nurse := 1
def school_principal := 1

def total_given_away := ms_delmont_class + mrs_donnelly_class + ms_delmont + mrs_donnelly + school_nurse + school_principal

theorem cupcakes_left_over : total_cupcakes - total_given_away = 2 := by
  sorry

end cupcakes_left_over_l702_702951


namespace f_odd_when_a_zero_f_neither_even_nor_odd_when_a_ne_zero_range_f_a_eq_4_l702_702193

def f (x a : ℝ) : ℝ := x * abs (x + a)

-- Definition and proof that f(x) is an odd function when a = 0
theorem f_odd_when_a_zero : ∀ (x : ℝ), f (-x) 0 = - f x 0 := by
  sorry

-- Definition and proof that f(x) is neither even nor odd when a ≠ 0
theorem f_neither_even_nor_odd_when_a_ne_zero : ∀ (x a : ℝ), a ≠ 0 → (f (-x) a ≠ - f x a ∧ f (-x) a ≠ f x a) := by
  sorry

-- Definition and proof for the range of f(x) in the interval [-4,1] when a = 4
theorem range_f_a_eq_4 : Set.Icc (-4.0) 1.0 ⊆ Set.image (λ x : ℝ, f x 4) (Set.Icc (-4.0) 1.0) := by
  sorry

end f_odd_when_a_zero_f_neither_even_nor_odd_when_a_ne_zero_range_f_a_eq_4_l702_702193


namespace total_clothes_washed_l702_702769

def number_of_clothing_items (Cally Danny Emily shared_socks : ℕ) : ℕ :=
  Cally + Danny + Emily + shared_socks

theorem total_clothes_washed :
  let Cally_clothes := (10 + 5 + 7 + 6 + 3)
  let Danny_clothes := (6 + 8 + 10 + 6 + 4)
  let Emily_clothes := (8 + 6 + 9 + 5 + 2)
  let shared_socks := (3 + 2)
  number_of_clothing_items Cally_clothes Danny_clothes Emily_clothes shared_socks = 100 :=
by
  sorry

end total_clothes_washed_l702_702769


namespace trigonometric_identity_l702_702399

theorem trigonometric_identity : 
  (Real.sin (42 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) - 
   Real.cos (138 * Real.pi / 180) * Real.cos (72 * Real.pi / 180)) = 
  (Real.sqrt 3 / 2) :=
by
  sorry

end trigonometric_identity_l702_702399


namespace time_to_cross_bridge_l702_702385

-- Definitions of the parameters given in the problem
def length_of_train : ℝ := 110     -- in meters
def speed_of_train : ℝ := 72 * (1000 / 3600) -- conversion from km/hr to m/s
def length_of_bridge : ℝ := 138    -- in meters

-- Theorem that needs to be proven
theorem time_to_cross_bridge : 
  (length_of_train + length_of_bridge) / speed_of_train = 12.4 :=
by
  -- The proof will go here
  sorry

end time_to_cross_bridge_l702_702385


namespace palindrome_divisible_by_11_probability_l702_702732

theorem palindrome_divisible_by_11_probability :
  (∃ N : ℕ, N > 0 ∧
    N = (Finset.card (Finset.filter
      (λ x : ℤ × ℤ × ℤ, let ⟨a, b, c⟩ := x in
        1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ (2 * a + 9 * b + c) % 11 = 0)
      (Finset.product (Finset.range 9).to_finset (Finset.product (Finset.range 10).to_finset (Finset.range 10).to_finset))) ∧
    (↑N : ℚ) / 900 = 1 / 5) :=
sorry

end palindrome_divisible_by_11_probability_l702_702732


namespace Delta_15_xDelta_eq_neg_15_l702_702129

-- Definitions of the operations based on conditions
def xDelta (x : ℝ) : ℝ := 9 - x
def Delta (x : ℝ) : ℝ := x - 9

-- Statement that we need to prove
theorem Delta_15_xDelta_eq_neg_15 : Delta (xDelta 15) = -15 :=
by
  -- The proof will go here
  sorry

end Delta_15_xDelta_eq_neg_15_l702_702129


namespace graph_passes_through_2_2_l702_702987

-- Given conditions as Lean definitions
variables {a x y : ℝ}
variable (h_a_pos : a > 0)
variable (h_a_ne_one : a ≠ 1)

-- Definition of the function
def f (x : ℝ) : ℝ := a^(x-2) + Real.logb a (x-1) + 1

-- Statement to prove that the graph of the function passes through (2, 2)
theorem graph_passes_through_2_2 : f a 2 = 2 :=
by
  rw [f]
  norm_num
  sorry

end graph_passes_through_2_2_l702_702987


namespace angles_equality_l702_702595

variables {A B C P Q D E : Type} [point : is_point A] [point : is_point B] [point : is_point C] [point : is_point P] [point : is_point Q] [point : is_point D] [point : is_point E]
variables {angle : is_angle P B A} {angle' : is_angle P C A}

-- Given conditions
def interior_point (A B C P : Type) [is_point A] [is_point B] [is_point C] [is_point P] : Prop :=
∃ (P : Type), is_point P ∧ inside_triangle A B C P

def equal_angles (angle : is_angle P B A) (angle' : is_angle P C A) : Prop := 
angle = angle'

def parallelogram (P Q B C : Type) : Prop := 
quadrilateral P Q B C ∧ parallel Q C P B

-- Main theorem statement
theorem angles_equality (A B C P Q D E : Type) [is_point A] [is_point B] [is_point C] [is_point P] [is_point Q] [is_point D] [is_point E]
    (h1 : interior_point A B C P) 
    (h2 : equal_angles (is_angle P B A) (is_angle P C A))
    (h3 : parallelogram P Q B C) :
  is_angle C A Q = is_angle P A B :=
sorry

end angles_equality_l702_702595


namespace eccentricity_squared_l702_702195

-- Define the hyperbola and its properties
variables (a b c e : ℝ) (x₁ y₁ x₂ y₂ : ℝ)

-- Define the hyperbola equation and conditions
def hyperbola_eq (a b x y : ℝ) := (x^2)/(a^2) - (y^2)/(b^2) = 1

def midpoint_eq (x₁ y₁ x₂ y₂ : ℝ) := x₁ + x₂ = -4 ∧ y₁ + y₂ = 2

def slope_eq (a b c : ℝ) := -b / c = (b^2 * (-4)) / (a^2 * 2)

-- Define the proof
theorem eccentricity_squared :
  a > 0 → b > 0 → hyperbola_eq a b x₁ y₁ → hyperbola_eq a b x₂ y₂ → midpoint_eq x₁ y₁ x₂ y₂ →
  slope_eq a b c → c^2 = a^2 + b^2 → (e = c / a) → e^2 = (Real.sqrt 2 + 1) / 2 :=
by
  intro ha hb h1 h2 h3 h4 h5 he
  sorry

end eccentricity_squared_l702_702195


namespace median_of_circumscribed_trapezoid_l702_702439

theorem median_of_circumscribed_trapezoid (a b c d : ℝ) (h1 : a + b + c + d = 12) (h2 : a + b = c + d) : (a + b) / 2 = 3 :=
by
  sorry

end median_of_circumscribed_trapezoid_l702_702439


namespace percentage_of_students_passed_l702_702883

theorem percentage_of_students_passed
  (students_failed : ℕ)
  (total_students : ℕ)
  (H_failed : students_failed = 260)
  (H_total : total_students = 400)
  (passed := total_students - students_failed) :
  (passed * 100 / total_students : ℝ) = 35 := 
by
  -- proof steps would go here
  sorry

end percentage_of_students_passed_l702_702883


namespace square_area_increase_l702_702388

variable (s : ℝ)

theorem square_area_increase (h : s > 0) : 
  let s_new := 1.30 * s
  let A_original := s^2
  let A_new := s_new^2
  let percentage_increase := ((A_new - A_original) / A_original) * 100
  percentage_increase = 69 := by
sorry

end square_area_increase_l702_702388


namespace power_modulo_l702_702606

theorem power_modulo {a : ℤ} : a^561 ≡ a [ZMOD 561] :=
sorry

end power_modulo_l702_702606


namespace radius_of_k3_l702_702363

variable (k₁ k₂ : Circle) (r1 r2 : ℝ)
variable (E P1 P2 : Point)
variable (lineThrough : Line)
variable (k₃ : Circle)
variable [externally_tangent k₁ k₂ at E]
variable [line_intersects_circles lineThrough k₁ k₂ at P1 P2]

theorem radius_of_k3 :
  circle_passes_through k₃ P1 P2 ∧ tangent_to k₃ k₁ →
  radius k₃ = radius k₁ + radius k₂ :=
by
  sorry

end radius_of_k3_l702_702363


namespace cone_to_cylinder_volume_ratio_l702_702485

noncomputable def cylinder_volume (r : ℝ) (h : ℝ) : ℝ :=
  π * r^2 * h

noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ :=
  (1 / 3) * π * r^2 * h

theorem cone_to_cylinder_volume_ratio :
  ∀ (r h_cylinder : ℝ), 
    r = 8 → h_cylinder = 24 →
    let h_cone := 0.75 * h_cylinder in
    (cone_volume r h_cone / cylinder_volume r h_cylinder) = 1 / 4 :=
by {
  intros r h_cylinder hr hh,
  rw [hr, hh],
  let h_cone := 0.75 * h_cylinder,
  sorry
}

end cone_to_cylinder_volume_ratio_l702_702485


namespace regression_passes_through_none_l702_702658

theorem regression_passes_through_none (b a x y : ℝ) (h₀ : (0, 0) ≠ (0*b + a, 0))
                                     (h₁ : (x, 0) ≠ (x*b + a, 0))
                                     (h₂ : (x, y) ≠ (x*b + a, y)) : 
                                     ¬ ((0, 0) = (0*b + a, 0) ∨ (x, 0) = (x*b + a, 0) ∨ (x, y) = (x*b + a, y)) :=
by sorry

end regression_passes_through_none_l702_702658


namespace intersecting_diagonals_of_parallelogram_l702_702625

theorem intersecting_diagonals_of_parallelogram (A C : ℝ × ℝ) (hA : A = (2, -3)) (hC : C = (14, 9)) :
    ∃ M : ℝ × ℝ, M = (8, 3) ∧ M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) :=
by {
  sorry
}

end intersecting_diagonals_of_parallelogram_l702_702625


namespace economist_winning_strategy_l702_702700

-- Conditions setup
variables {n a b x1 x2 y1 y2 : ℕ}

-- Definitions according to the conditions
def valid_initial_division (n a b : ℕ) : Prop :=
  n > 4 ∧ n % 2 = 1 ∧ 2 ≤ a ∧ 2 ≤ b ∧ a + b = n ∧ a < b

def valid_further_division (a b x1 x2 y1 y2 : ℕ) : Prop :=
  x1 + x2 = a ∧ x1 ≥ 1 ∧ x2 ≥ 1 ∧ y1 + y2 = b ∧ y1 ≥ 1 ∧ y2 ≥ 1 ∧ x1 ≤ x2 ∧ y1 ≤ y2

-- Methods defined: Assumptions about which parts the economist takes
def method_1 (x1 x2 y1 y2 : ℕ) : ℕ :=
  max x2 y2 + min x1 y1

def method_2 (x1 x2 y1 y2 : ℕ) : ℕ :=
  (x1 + y1) / 2 + (x2 + y2) / 2

def method_3 (x1 x2 y1 y2 : ℕ) : ℕ :=
  max (method_1 x1 x2 y1 y2 - 1) (method_2 x1 x2 y1 y2 - 1) + 1

-- The statement to prove that the economist would choose method 1
theorem economist_winning_strategy :
  ∀ n a b x1 x2 y1 y2,
    valid_initial_division n a b →
    valid_further_division a b x1 x2 y1 y2 →
    n > 4 → n % 2 = 1 →
    (method_1 x1 x2 y1 y2) > (method_2 x1 x2 y1 y2) →
    (method_1 x1 x2 y1 y2) > (method_3 x1 x2 y1 y2) →
    method_1 x1 x2 y1 y2 = max (method_1 x1 x2 y1 y2) (method_2 x1 x2 y1 y2) :=
by
  -- Placeholder for the actual proof
  sorry

end economist_winning_strategy_l702_702700


namespace polynomial_remainder_l702_702376

theorem polynomial_remainder (y : ℝ) : 
  let a := 3 ^ 50 - 2 ^ 50
  let b := 2 ^ 50 - 2 * 3 ^ 50 + 2 ^ 51
  (y ^ 50) % (y ^ 2 - 5 * y + 6) = a * y + b :=
by
  sorry

end polynomial_remainder_l702_702376


namespace length_AE_is_8_over_3_l702_702325

-- Define the square ABCD with side length 4 units
def squareABCD : Prop :=
  ∀ (A B C D : ℝ × ℝ), dist A B = 4 ∧ dist B C = 4 ∧ dist C D = 4 ∧ dist D A = 4 ∧
     dist A C = dist B D

-- Define the point E on side AB and the reflection F of E over diagonal AC
def pointEonAB_and_reflectionF (A B E F : ℝ × ℝ) : Prop :=
  ∃ (x : ℝ), E = (x, 0) ∧ B = (4, 0) ∧ A = (0, 0) ∧
  dist (4 - x, 4) F = 0

-- Define the conditions about distance DF and AE
def conditionDF_and_AE (A D F E : ℝ × ℝ) : Prop :=
  dist D F = 2 * dist A E

-- The main theorem combining all conditions
theorem length_AE_is_8_over_3 (A B C D E F : ℝ × ℝ) 
  (h1 : squareABCD) (h2 : pointEonAB_and_reflectionF A B E F) (h3 : conditionDF_and_AE A D F E) :
  ∃ x : ℝ, dist A E = x ∧ x = 8 / 3 :=
begin
  sorry
end

end length_AE_is_8_over_3_l702_702325


namespace incircle_radius_l702_702669

-- The centers of the circles
variables (O₁ O₂ O₃ K M N : Type*) [metric_space O₁] [metric_space O₂] [metric_space O₃]

-- Conditions: Circles with radii 1, 2, and 3
axiom radius_O₁ : dist O₁ O₁ = 1
axiom radius_O₂ : dist O₂ O₂ = 2
axiom radius_O₃ : dist O₃ O₃ = 3

-- Distance conditions between centers due to external tangency
axiom dist_O₁O₂ : dist O₁ O₂ = 3
axiom dist_O₁O₃ : dist O₁ O₃ = 4
axiom dist_O₂O₃ : dist O₂ O₃ = 5

-- The points of tangency
axiom tangency_K : dist O₁ K = 1 ∧ dist O₂ K = 2
axiom tangency_M : dist O₂ M = 2 ∧ dist O₃ M = 3
axiom tangency_N : dist O₃ N = 3 ∧ dist O₁ N = 1

-- Goal: Prove the radius of the circle passing through K, M, and N is 1
theorem incircle_radius : ∃ (r : ℝ), (r = 1) :=
sorry

end incircle_radius_l702_702669


namespace water_ounces_in_sport_drink_l702_702387

def standard_flavoring := 1
def standard_cornsyrup := 12
def standard_water := 30

def sport_flavoring_cornsyrup_ratio := 3 * standard_flavoring / standard_cornsyrup
def sport_flavoring_water_ratio := (1/2) * standard_flavoring / standard_water

def sport_flavoring := 1 -- normalized
def sport_cornsyrup := sport_flavoring / sport_flavoring_cornsyrup_ratio
def sport_water := sport_flavoring / sport_flavoring_water_ratio

axiom sport_cornsyrup_ounces : ℝ := 2
theorem water_ounces_in_sport_drink : 
  sport_cornsyrup_ounces * sport_water / sport_cornsyrup = 120 := 
sorry

end water_ounces_in_sport_drink_l702_702387


namespace right_triangle_hypotenuse_l702_702882

theorem right_triangle_hypotenuse (a b : ℝ) (m_a m_b : ℝ)
    (h1 : m_a = Real.sqrt (b^2 + (a / 2)^2))
    (h2 : m_b = Real.sqrt (a^2 + (b / 2)^2))
    (h3 : m_a = Real.sqrt 30)
    (h4 : m_b = 6) :
  Real.sqrt (4 * (a^2 + b^2)) = 2 * Real.sqrt 52.8 :=
by
  sorry

end right_triangle_hypotenuse_l702_702882


namespace greatest_vertical_distance_is_correct_l702_702134
noncomputable def rotated_square_max_height : ℝ := Real.sqrt 2 / 2

theorem greatest_vertical_distance_is_correct :
  ∀ (squares : ℕ → ℝ × ℝ)
    (H1 : ∀ n, n < 4 → squares n = (n, 0))
    (H2 : ∀ p, sqdist (p - squares 2) = 1)
    (H3 : ∀ p, sqdist (p - (1, 1)) = 1),
  rotated_square_max_height = Real.sqrt 2 / 2 :=
by sorry

end greatest_vertical_distance_is_correct_l702_702134


namespace greatest_n_leq_inequality_l702_702683

theorem greatest_n_leq_inequality : ∃ n : ℤ, (n^2 - 13 * n + 36 ≤ 0) ∧ ∀ m : ℤ, (m^2 - 13 * m + 36 ≤ 0) → m ≤ n := 
by
  existsi (9 : ℤ)
  split
  {
    -- Validate that 9 satisfies the inequality
    sorry
  }
  {
    -- Show for any m, if m satisfies the inequality, it must be less than or equals to 9
    intro m
    intro hm
    -- prove m <= 9
    sorry
  }

end greatest_n_leq_inequality_l702_702683


namespace min_value_f_l702_702123

def f (x : ℝ) : ℝ := x ^ 2 / (x - 3)

theorem min_value_f : ∃ (m : ℝ), m = 12 ∧ ∀ (x : ℝ), x > 3 → f(x) ≥ m := 
by
  sorry

end min_value_f_l702_702123


namespace log_expression_simplification_l702_702833

open Real

theorem log_expression_simplification (p q r s t z : ℝ) :
  log (p / q) + log (q / r) + log (r / s) - log (p * t / (s * z)) = log (z / t) :=
  sorry

end log_expression_simplification_l702_702833


namespace infinite_negative_terms_in_sequence_l702_702290

noncomputable def P (r q p : ℝ) (x : ℝ) : ℝ := r * x^3 + q * x^2 + p * x + 1

noncomputable def sequence (p q r : ℝ) : ℕ → ℝ
| 0     := 1
| 1     := -p
| 2     := p^2 - q
| (n+3) := -p * (sequence p q r (n + 2)) - q * (sequence p q r (n + 1)) - r * (sequence p q r n)
| _     := sorry -- Required to avoid non-exhaustive pattern warning; filled with sorry for now.

theorem infinite_negative_terms_in_sequence {r q p : ℝ} (hr : r > 0)
  (hroot : ∃ x : ℝ, P r q p x = 0 ∧ ∀ y : ℝ, P r q p y = 0 → x = y) :
  ∃ᶠ n in at_top, (sequence p q r n) < 0 :=
sorry

end infinite_negative_terms_in_sequence_l702_702290


namespace fraction_of_basic_stereos_l702_702771

theorem fraction_of_basic_stereos (B D : ℕ) (T : ℕ) 
  (h1 : 1.6 * T = 8 / 5 * T)
  (h2 : D * (1.6 * T) = 0.5 * (B * T + D * (1.6 * T)))
  (h3 : B = 1.6 * D) 
  (hT_pos : 0 < T) :
  B / (B + D) = 8 / 13 :=
by 
  sorry

end fraction_of_basic_stereos_l702_702771


namespace complex_number_properties_l702_702499

open Complex

-- Defining the imaginary unit
def i : ℂ := Complex.I

-- Given conditions in Lean: \( z \) satisfies \( z(2+i) = i^{10} \)
def satisfies_condition (z : ℂ) : Prop :=
  z * (2 + i) = i^10

-- Theorem stating the required proofs
theorem complex_number_properties (z : ℂ) (hc : satisfies_condition z) :
  Complex.abs z = Real.sqrt 5 / 5 ∧ 
  (z.re < 0 ∧ z.im > 0) := by
  -- Placeholders for the proof steps
  sorry

end complex_number_properties_l702_702499


namespace friends_positions_l702_702965

variables (inside_triangle : ℕ → Prop) (inside_square : ℕ → Prop) (inside_circle : ℕ → Prop)

def Ana := 6
def Bento := 7
def Celina := 3
def Diana := 5
def Elisa := 4
def Fábio := 1
def Guilherme := 2

theorem friends_positions :
  ¬(inside_triangle Ana ∨ inside_square Ana ∨ inside_circle Ana) ∧
  (inside_triangle Bento ∨ inside_square Bento ∨ inside_circle Bento) ∧
  ¬(inside_triangle Bento ∧ inside_square Bento ∧ inside_circle Bento) ∧
  (inside_triangle Celina ∧ inside_square Celina ∧ inside_circle Celina) ∧
  (inside_triangle Diana ∧ ¬inside_square Diana) ∧
  (inside_triangle Elisa ∧ inside_circle Elisa ∧ ¬inside_square Elisa) ∧
  ¬((inside_triangle Fábio ∨ inside_square Fábio) ∧ inside_circle Fábio) ∧
  inside_circle Guilherme :=
sorry

end friends_positions_l702_702965


namespace f_k_minus_e_k_eq_minus_e_sq_l702_702836

noncomputable def f (x : ℝ) : ℝ := abs(log(abs(x - 1)))

variable {x1 x2 x3 x4 m : ℝ}

axiom zero_points : {x1 x2 x3 x4 : ℝ // f x1 = m ∧ f x2 = m ∧ f x3 = m ∧ f x4 = m}
def k : ℝ := 1 / x1 + 1 / x2 + 1 / x3 + 1 / x4

theorem f_k_minus_e_k_eq_minus_e_sq :
  f k - real.exp k = -real.exp 2 := sorry

end f_k_minus_e_k_eq_minus_e_sq_l702_702836


namespace intersection_points_eq_ten_l702_702774

def fractional_part (x : ℝ) : ℝ := x - (Int.floor x)

def circle_eq (x y : ℝ) : Prop := (fractional_part x)^2 + y^2 = 2 * (fractional_part x)
def line_eq (x y : ℝ) : Prop := y = (1 / 3) * x

theorem intersection_points_eq_ten : 
  (∃ S : set (ℝ × ℝ), 
    ∀ p ∈ S, circle_eq p.1 p.2 ∧ line_eq p.1 p.2 ∧ 
    S.finite ∧ S.card = 10) :=
sorry

end intersection_points_eq_ten_l702_702774


namespace problem_lemma_l702_702597

theorem problem_lemma (n : ℕ) (h : n = 2007) :
  (∏ k in (finset.range (n - 3)).map (finset.Nat.optionEquivFin).symm, 
   2 * (1 - 1 / (k + 4 : ℝ))) = 3 * 2 ^ 2004 / 2007 :=
by {
  sorry
}

end problem_lemma_l702_702597


namespace angle_sum_of_regular_triangle_and_pentagon_l702_702256

noncomputable def regular_polygon_interior_angle (n : ℕ) : ℝ :=
180 * (n - 2) / n

theorem angle_sum_of_regular_triangle_and_pentagon (WXY WXZ WXP : ℝ)
  (h1 : WXY = regular_polygon_interior_angle 3)
  (h2 : WXZ = regular_polygon_interior_angle 3)
  (h3 : WXP = regular_polygon_interior_angle 5) :
  WXY + WXZ + WXP = 228 :=
by
  rw [h1, h2, h3]
  simp [regular_polygon_interior_angle]
  norm_num
  sorry

end angle_sum_of_regular_triangle_and_pentagon_l702_702256


namespace range_of_a_l702_702858

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 - a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
by
  sorry

end range_of_a_l702_702858


namespace find_natural_number_l702_702360

theorem find_natural_number :
  ∃ x : ℕ, (∀ d1 d2 : ℕ, d1 ∣ x → d2 ∣ x → d1 < d2 → d2 - d1 = 4) ∧
           (∀ d1 d2 : ℕ, d1 ∣ x → d2 ∣ x → d1 < d2 → x - d2 = 308) ∧
           x = 385 :=
by
  sorry

end find_natural_number_l702_702360


namespace probability_of_two_red_balls_l702_702013

-- Definitions of quantities
def total_balls := 11
def red_balls := 3
def blue_balls := 4 
def green_balls := 4 
def balls_picked := 2

-- Theorem statement
theorem probability_of_two_red_balls :
  ((red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1) / balls_picked)) = 3 / 55 :=
by
  sorry

end probability_of_two_red_balls_l702_702013


namespace g_2002_eq_one_l702_702141

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := f(x) + 1 - x

axiom f_one : f 1 = 1
axiom f_ineq1 (x : ℝ) : f (x + 5) ≥ f x + 5
axiom f_ineq2 (x : ℝ) : f (x + 1) ≤ f x + 1

theorem g_2002_eq_one : g 2002 = 1 :=
by
  sorry

end g_2002_eq_one_l702_702141


namespace expression_eq_neg_one_l702_702464

theorem expression_eq_neg_one (a b y : ℝ) (h1 : a ≠ 0) (h2 : b ≠ a) (h3 : y ≠ a) (h4 : y ≠ -a) :
  ( ( (a + b) / (a + y) + y / (a - y) ) / ( (y + b) / (a + y) - a / (a - y) ) = -1 ) ↔ ( y = a - b ) := 
sorry

end expression_eq_neg_one_l702_702464


namespace find_side_c_l702_702578

theorem find_side_c (a C S : ℝ) (ha : a = 3) (hC : C = 120) (hS : S = (15 * Real.sqrt 3) / 4) : 
  ∃ (c : ℝ), c = 7 :=
by
  sorry

end find_side_c_l702_702578


namespace julians_girls_percentage_l702_702269

theorem julians_girls_percentage
  (julian_friends : ℕ)
  (julian_boys_percentage : ℚ)
  (boyd_friends : ℕ)
  (boyd_girls_multiple : ℕ)
  (boyd_boys_percentage : ℚ)
  (h_julian_friends : julian_friends = 80)
  (h_julian_boys_percentage : julian_boys_percentage = 0.60)
  (h_boyd_friends : boyd_friends = 100)
  (h_boyd_girls_multiple : boyd_girls_multiple = 2)
  (h_boyd_boys_percentage : boyd_boys_percentage = 0.36)
  : ((julian_friends * (1 - julian_boys_percentage)) / julian_friends) * 100 = 40 := by
    sorry

end julians_girls_percentage_l702_702269


namespace part1_part2_l702_702498

-- Define the function f(x)
def f (x : ℝ) (b c : ℝ) : ℝ :=
  2 * x ^ 2 + b * x + c

-- The first part: proving the explicit form given the conditions
theorem part1 : 
  (∀ x, f x b c < 0 ↔ 0 < x ∧ x < 5) → (b = -10 ∧ c = 0) :=
by
  sorry

-- The second part: find the range of t given f(x) and the inequality condition
theorem part2 (t : ℝ) :
  (∀ x ∈ set.Icc (-1 : ℝ) 1, f x -10 0 + t ≤ 2) → t ≤ -10 :=
by
  sorry

end part1_part2_l702_702498


namespace time_to_cross_l702_702389

def train_speed1 := 60 -- speed of train 1 in km/h
def train_speed2 := 90 -- speed of train 2 in km/h
def train_length1 := 1.10 -- length of train 1 in km
def train_length2 := 0.9 -- length of train 2 in km
def relative_speed := train_speed1 + train_speed2 / 3.6 -- relative speed in m/s (60+90 km/h to m/s conversion)
def total_length := (train_length1 + train_length2) * 1000 -- total length in m
def expected_time := 47.96 -- expected time in seconds

theorem time_to_cross : (total_length / relative_speed) ≈ expected_time :=
by
  sorry

end time_to_cross_l702_702389


namespace compute_fraction_when_x_is_3_l702_702085

theorem compute_fraction_when_x_is_3 :
  let x := 3 in
  (x^8 + 18 * x^4 + 81) / (x^4 + 9) = 90 := 
by
  let x := 3
  sorry

end compute_fraction_when_x_is_3_l702_702085


namespace ratio_of_segments_in_triangle_l702_702580

theorem ratio_of_segments_in_triangle
  (A B C D E : Type)  -- Types of the vertices
  (a : ℝ)  -- Angle A
  (hC : CD A B C)
  (hB : BE A B C)
  (H1 : ∠ A = a)
  (H2 : is_altitude C D A B)
  (H3 : is_altitude B E A C) :
  ∃ (DE BC : ℝ), DE / BC = abs (cos a) :=
by
  sorry

end ratio_of_segments_in_triangle_l702_702580


namespace monotonic_increasing_interval_l702_702654

noncomputable def function_is_monotonic_in_increasing_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

noncomputable def logarithmic_function := λ (x : ℝ), Real.logb (1/2) (x^2 - 2 * x - 3)

theorem monotonic_increasing_interval:
  function_is_monotonic_in_increasing_interval (λ (x:ℝ), Real.logb (1/2) (x^2 - 2 * x - 3)) :=
sorry

end monotonic_increasing_interval_l702_702654


namespace locus_of_O_is_perpendicular_bisector_of_AX_l702_702024

noncomputable def circumcenter (A B C : Point) : Point := sorry -- Placeholder definition for circumcenter

noncomputable def perpendicular_bisector (A B : Point) : Set Point := sorry -- Placeholder for perpendicular bisector

-- Define the points and conditions
axiom (A B C D O O₁ O₂ : Point)
axiom (ABC_acute : ∀ (A B C : Point), Triangle A B C → Acute (Triangle A B C))
axiom (D_on_BC : on_line_segment D B C)
axiom (circumcenter_ABD : O₁ = circumcenter A B D)
axiom (circumcenter_ACD : O₂ = circumcenter A C D)
axiom (circumcenter_AO₁O₂ : O = circumcenter A O₁ O₂)

-- The proof statement
theorem locus_of_O_is_perpendicular_bisector_of_AX :
  locus O = perpendicular_bisector A X := sorry

end locus_of_O_is_perpendicular_bisector_of_AX_l702_702024


namespace find_phi_l702_702025

theorem find_phi (φ : ℝ) (h0 : 0 ≤ φ) (h1 : φ < 2 * Real.pi) :
  (∀ x : ℝ, cos x - √3 * sin x = 2 * sin (x + φ)) → φ = 5 * Real.pi / 6 :=
by
  sorry

end find_phi_l702_702025


namespace economist_wins_by_choosing_method_1_l702_702702

variable (n : ℕ) (h_odd : n % 2 = 1) (h_greater_than_4 : n > 4)

-- Condition for Step 1: Lawyer divides coins into a and b
variable (a b : ℕ) (h_a_b : a + b = n) (h_a_2 : a ≥ 2) (h_b_2 : b ≥ 2) (h_a_lt_b : a < b)

-- Condition for Step 2: Economist divides a into x1 and x2, and b into y1 and y2
variable (x1 x2 y1 y2 : ℕ)
variable (h_x1_x2 : x1 + x2 = a) (h_y1_y2 : y1 + y2 = b)
variable (h_x1_1 : x1 ≥ 1) (h_x2_1 : x2 ≥ 1) (h_y1_1 : y1 ≥ 1) (h_y2_1 : y2 ≥ 1)
variable (h_x1_le_x2 : x1 ≤ x2) (h_y1_le_y2 : y1 ≤ y2)

-- Method 1: Economist takes largest and smallest parts

-- Method 2: Economist takes both middle parts

-- Method 3: Economist chooses method 1 or 2 and gives one coin to the lawyer

theorem economist_wins_by_choosing_method_1 :
  economist_strategy n = method1 :=
sorry

end economist_wins_by_choosing_method_1_l702_702702


namespace binary_111_is_7_l702_702373

def binary_to_decimal (b0 b1 b2 : ℕ) : ℕ :=
  b0 * (2^0) + b1 * (2^1) + b2 * (2^2)

theorem binary_111_is_7 : binary_to_decimal 1 1 1 = 7 :=
by
  -- We will provide the proof here.
  sorry

end binary_111_is_7_l702_702373


namespace exists_AB_for_tan_sum_l702_702022

noncomputable def a (k : ℕ) : ℝ :=
  Real.tan k * Real.tan (k - 1)

-- Theorem statement
theorem exists_AB_for_tan_sum :
  ∃ A B : ℝ, ∀ n : ℕ, (∑ k in Finset.range n, a k) = A * Real.tan n + B * n :=
sorry

end exists_AB_for_tan_sum_l702_702022


namespace part1_part2_l702_702811

-- Definitions of y1 and y2 based on given conditions
def y1 (x : ℝ) : ℝ := -x + 3
def y2 (x : ℝ) : ℝ := 2 + x

-- Prove for x such that y1 = y2
theorem part1 (x : ℝ) : y1 x = y2 x ↔ x = 1 / 2 := by
  sorry

-- Prove for x such that y1 = 2y2 + 5
theorem part2 (x : ℝ) : y1 x = 2 * y2 x + 5 ↔ x = -2 := by
  sorry

end part1_part2_l702_702811


namespace range_of_2_pow_x_l702_702541

theorem range_of_2_pow_x :
  (∀ (x : ℝ), 2^(x^2 + 1) ≤ (1/4)^(x - 2)) → (∀ (x : ℝ), ∃ (y : ℝ), y = 2^x ∧ (1/8 ≤ y ∧ y ≤ 2)) :=
by sorry

end range_of_2_pow_x_l702_702541


namespace sequence_sum_zero_l702_702096

theorem sequence_sum_zero :
  let sequence := List.range 2000 |>.map (1+·) -- Generates the list from 1 to 2000
  let groups := sequence.grouped 4           -- Groups the sequence in sets of four
  ∑(group : List Int) in groups, ∑(n, group) = 0 :=    -- Sum of each group equals 0
by
  sorry

end sequence_sum_zero_l702_702096


namespace altitudes_theorem_l702_702751

-- Definition of the geometric properties
structure triangle (A B C P Q H : Type) :=
(angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) 
(HP HQ BP PC AQ QC : ℝ)
(height_A : ℝ) (height_B : ℝ) (height_C : ℝ)
(intersect : A -> B -> C -> P -> Q -> H -> Prop)
(acute : angle_A < 90 ∧ angle_B < 90 ∧ angle_C < 90)

-- Given values for HP and HQ
def given_values :=
(HP : ℝ) := 6
(HQ : ℝ) := 3

-- Main theorem to prove the required equation
theorem altitudes_theorem (A B C P Q H : Type) (t : triangle A B C P Q H) : 
  t.HP = 6 → t.HQ = 3 → 
  (t.BP * t.PC + t.AQ * t.QC) = (48 / 7 * t.height_C + 45) :=
by
  sorry

end altitudes_theorem_l702_702751


namespace problem_equiv_l702_702661

theorem problem_equiv 
  (even_sum : ∑ i in finset.range 30, 2 * (i + 1) = 930) 
  (odd_sum : ∑ i in finset.range 5, (186 + 2 * (i - 2)) = 930) 
  : 186 + 4 = 190 :=
by
  sorry

end problem_equiv_l702_702661


namespace complex_conjugate_product_l702_702285

-- Here we define our condition that the magnitude of w is 15
variable (w : ℂ) (hw : Complex.abs w = 15)

-- We now state what we need to prove, which is w * conjugate(w) = 225
theorem complex_conjugate_product : w * Complex.conj w = 225 :=
by 
  sorry

end complex_conjugate_product_l702_702285


namespace square_ratio_l702_702431

theorem square_ratio :
  ∃ (x y : ℝ),
    (∃ Δ₁ Δ₂ : triangle ℝ,
      Δ₁.sides = (5, 12, 13) ∧ 
      Δ₁.is_right_triangle ∧ 
      Δ₁.has_inscribed_square x ∧ 
      Δ₂.sides = (6, 8, 10) ∧ 
      Δ₂.is_right_triangle ∧ 
      Δ₂.has_inscribed_square y) ∧
    (x / y = 37 / 35) :=
by 
  sorry

end square_ratio_l702_702431


namespace red_button_probability_l702_702582

-- Definitions of the initial state
def initial_red_buttons : ℕ := 8
def initial_blue_buttons : ℕ := 12
def total_buttons := initial_red_buttons + initial_blue_buttons

-- Condition of removal and remaining buttons
def removed_buttons := total_buttons - (5 / 8 : ℚ) * total_buttons

-- Equal number of red and blue buttons removed
def removed_red_buttons := removed_buttons / 2
def removed_blue_buttons := removed_buttons / 2

-- State after removal
def remaining_red_buttons := initial_red_buttons - removed_red_buttons
def remaining_blue_buttons := initial_blue_buttons - removed_blue_buttons

-- Jars after removal
def jar_X := remaining_red_buttons + remaining_blue_buttons
def jar_Y := removed_red_buttons + removed_blue_buttons

-- Probability calculations
def probability_red_X : ℚ := remaining_red_buttons / jar_X
def probability_red_Y : ℚ := removed_red_buttons / jar_Y

-- Final probability
def final_probability : ℚ := probability_red_X * probability_red_Y

theorem red_button_probability :
  final_probability = 4 / 25 := 
  sorry

end red_button_probability_l702_702582


namespace inequality_proof_l702_702864

-- Given x in the interval (e^{-1}, 1)
variable (x : ℝ) (h1 : exp(-1) < x) (h2 : x < 1)

-- Definitions of a, b, c
def a := Real.log x
def b := 2 * Real.log x
def c := Real.log x ^ 3

theorem inequality_proof : b < a ∧ a < c := by
  have hln : Real.log x < 0 := sorry
  have ha : a = Real.log x := rfl
  have hb : b = 2 * Real.log x := rfl
  have hc : c = Real.log x ^ 3 := rfl
  sorry

end inequality_proof_l702_702864


namespace money_per_card_l702_702265

theorem money_per_card (n_grandchildren : ℕ) (cards_per_grandchild : ℕ) (total_money_given : ℕ) 
  (h1 : n_grandchildren = 3) (h2 : cards_per_grandchild = 2) (h3 : total_money_given = 480) : 
  total_money_given / (n_grandchildren * cards_per_grandchild) = 80 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end money_per_card_l702_702265


namespace expression_value_l702_702438

theorem expression_value :
  (6^2 - 3^2)^4 = 531441 := by
  -- Proof steps were omitted
  sorry

end expression_value_l702_702438


namespace sarah_marry_age_l702_702467

/-- Sarah is 9 years old. -/
def Sarah_age : ℕ := 9

/-- Sarah's name has 5 letters. -/
def Sarah_name_length : ℕ := 5

/-- The game's rule is to add the number of letters in the player's name 
    to twice the player's age. -/
def game_rule (name_length age : ℕ) : ℕ :=
  name_length + 2 * age

/-- Prove that Sarah will get married at the age of 23. -/
theorem sarah_marry_age : game_rule Sarah_name_length Sarah_age = 23 := 
  sorry

end sarah_marry_age_l702_702467


namespace parallel_line_slope_l702_702000

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℚ, m = 1 / 2 :=
by {
  use 1 / 2,
  sorry
}

end parallel_line_slope_l702_702000


namespace number_of_people_in_group_l702_702644

theorem number_of_people_in_group
  (n : ℕ) -- number of people in the group
  (W : ℝ) -- total weight of the group before the new person arrives
  (h1 : (93 - 65) = 28) -- weight difference condition
  (h2 : 3.5 * n = 28) -- increase in average weight condition
  : n = 8 := 
by
  -- Proof goes here
  sorry

end number_of_people_in_group_l702_702644


namespace distance_from_tangency_to_tangent_l702_702362

theorem distance_from_tangency_to_tangent 
  (R r : ℝ)
  (hR : R = 3)
  (hr : r = 1)
  (externally_tangent : true) :
  ∃ d : ℝ, (d = 0 ∨ d = 7/3) :=
by
  sorry

end distance_from_tangency_to_tangent_l702_702362


namespace cost_of_article_l702_702386

theorem cost_of_article (C G : ℝ) (h1 : C + G = 348) (h2 : C + 1.05 * G = 350) : C = 308 :=
by
  sorry

end cost_of_article_l702_702386


namespace anthony_solve_l702_702757

def completing_square (a b c : ℤ) : ℤ :=
  let d := Int.sqrt a
  let e := b / (2 * d)
  let f := (d * e * e - c)
  d + e + f

theorem anthony_solve (d e f : ℤ) (h_d_pos : d > 0)
  (h_eqn : 25 * d * d + 30 * d * e - 72 = 0)
  (h_form : (d * x + e)^2 = f) : 
  d + e + f = 89 :=
by
  have d : ℤ := 5
  have e : ℤ := 3
  have f : ℤ := 81
  sorry

end anthony_solve_l702_702757


namespace percentage_of_red_non_honda_cars_correct_l702_702697

noncomputable def percentage_of_red_non_honda_cars 
  (total_cars : ℕ) (honda_cars : ℕ) (percentage_red_honda : ℝ) (percentage_red_total : ℝ) : ℝ :=
  let non_honda_cars := total_cars - honda_cars
  let red_honda_cars := percentage_red_honda * honda_cars
  let total_red_cars := percentage_red_total * total_cars
  let red_non_honda_cars := total_red_cars - red_honda_cars
  (red_non_honda_cars / non_honda_cars) * 100

#eval percentage_of_red_non_honda_cars 900 500 0.9 0.6 -- Expected output: 22.5

-- The theorem statement asserting the above definition gives the correct answer
theorem percentage_of_red_non_honda_cars_correct :
  percentage_of_red_non_honda_cars 900 500 0.9 0.6 = 22.5 := 
by
  -- The proof is not required as per the instruction
  sorry

end percentage_of_red_non_honda_cars_correct_l702_702697


namespace range_of_x_max_y_over_x_l702_702159

-- Define the circle and point P(x,y) on the circle
def CircleEquation (x y : ℝ) : Prop := (x - 4)^2 + (y - 3)^2 = 9

theorem range_of_x (x y : ℝ) (h : CircleEquation x y) : 1 ≤ x ∧ x ≤ 7 :=
sorry

theorem max_y_over_x (x y : ℝ) (h : CircleEquation x y) : ∀ k : ℝ, (k = y / x) → 0 ≤ k ∧ k ≤ (24 / 7) :=
sorry

end range_of_x_max_y_over_x_l702_702159


namespace find_rate_percent_l702_702390

-- Definitions
def simpleInterest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Given conditions
def principal : ℕ := 900
def time : ℕ := 4
def simpleInterestValue : ℕ := 160

-- Rate percent
theorem find_rate_percent : 
  ∃ R : ℕ, simpleInterest principal R time = simpleInterestValue :=
by
  sorry

end find_rate_percent_l702_702390


namespace solve_for_a_and_monotonic_interval_prove_inequality_l702_702189

noncomputable def f (x : ℝ) : ℝ := Real.log (2^(2*x) + 2^x - 2) / Real.log 2  -- log base 2

theorem solve_for_a_and_monotonic_interval (a : ℝ) (h1 : a = 2) (h2 : ∀ x > 0, f(x) = 2) :
  a = 2 ∧ ∀ x : ℝ, x > 0 → Function.Monotone f :=
sorry

theorem prove_inequality (x : ℝ) (h1 : x > 0) (h2 : x < Real.log 3 / Real.log 2) :
  f(x + 1) - f(x) > 2 :=
sorry

end solve_for_a_and_monotonic_interval_prove_inequality_l702_702189


namespace find_m_and_a_l702_702841

-- Definitions based on given conditions
def is_power_function (f : ℝ → ℝ) (α : ℝ) := ∀ x, f x = x ^ α
def passes_through_points (f : ℝ → ℝ) (p1 p2 : ℝ × ℝ) := f p1.fst = p1.snd ∧ f p2.fst = p2.snd
def log_function_max_min_diff (g : ℝ → ℝ) (a : ℝ) (interval : Set ℝ) := 
  let g_min := set.minimal g (set.range (λ x, x ∈ interval))
  let g_max := set.maximal g (set.range (λ x, x ∈ interval))
  g_max - g_min = 1

-- Assertions to be proven based on the above conditions
theorem find_m_and_a :
  ∀ (f : ℝ → ℝ) (α m a : ℝ),
    is_power_function f α →
    passes_through_points f (8, m) (9, 3) →
    (∃ m, m = 2 * real.sqrt 2) ∧
    (∀ g : ℝ → ℝ, (g = λ x, real.log a (f x)) →
      log_function_max_min_diff g a {x | 16 ≤ x ∧ x ≤ 36} →
      (a = 2/3 ∨ a = 3/2)) :=
by
  sorry

end find_m_and_a_l702_702841


namespace problem_solution_l702_702506

theorem problem_solution (x y z : ℝ) (h1 : 2 * x - y - 2 * z - 6 = 0) (h2 : x^2 + y^2 + z^2 ≤ 4) :
  2 * x + y + z = 2 / 3 := 
by 
  sorry

end problem_solution_l702_702506


namespace suff_not_nec_condition_converse_not_needed_l702_702824

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ ≤ f x₂

theorem suff_not_nec_condition (f : ℝ → ℝ) :
  (is_monotonically_increasing f) → (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ < f x₂) :=
by
  intro h
  -- existence proof skipped
  sorry

theorem converse_not_needed (f : ℝ → ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ < f x₂) → ¬(is_monotonically_increasing f) :=
by
  -- counterexample proof skipped
  sorry

lemma sufficient_but_not_necessary_condition (f : ℝ → ℝ) : 
  (is_monotonically_increasing f) ↔ (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ < f x₂) :=
by
  apply iff.intro
  . exact suff_not_nec_condition f
  . exact converse_not_needed f

end suff_not_nec_condition_converse_not_needed_l702_702824


namespace isosceles_triangle_base_l702_702338

noncomputable def base_of_isosceles_triangle
  (height_to_base : ℝ)
  (height_to_side : ℝ)
  (is_isosceles : Bool) : ℝ :=
if is_isosceles then 7.5 else 0

theorem isosceles_triangle_base :
  base_of_isosceles_triangle 5 6 true = 7.5 :=
by
  -- The proof would go here, just placeholder for now
  sorry

end isosceles_triangle_base_l702_702338


namespace A₀_value_l702_702305

noncomputable def A₀ (a₁ a₂ a₃ : ℝ) (A₂ A₄ : ℕ → ℝ) : ℝ := sorry
noncomputable def A₂ (b₁ b₂ b₃ b₄ : ℝ) (A₀ A₂ A₄ A₆ : ℕ → ℝ) : ℝ := sorry
noncomputable def A₂j (c₁ c₂ c₃ c₄ : ℝ) (A_2j_minus_2 A_2j A_2j_plus_2 A_2j_plus_4 : ℕ → ℝ) : ℝ := sorry

theorem A₀_value (a₁ a₂ a₃ b₁ b₂ b₃ b₄ c₁ c₂ c₃ c₄ : ℝ) (A₂ A₄ A₆ : ℕ → ℝ) :
  (∑ k, A₄ (2 * k) = 1) →
  (A₀ = a₁ * A₀ + a₂ * A₂ + a₃ * A₄) →
  (A₂ = b₁ * A₀ + b₂ * A₂ + b₃ * A₄ + b₄ * A₆) →
  ∀ j ≥ 2, A₂j j = c₁ * A₂j (j-1) + c₂ * A₂j j + c₃ * A₂j (j+1) + c₄ * A₂j (j+2) →
  A₀ = (sqrt 5 - 1) / 2 :=
sorry

end A₀_value_l702_702305


namespace sarah_total_weeds_l702_702962

theorem sarah_total_weeds :
  let tuesday_weeds := 25
  let wednesday_weeds := 3 * tuesday_weeds
  let thursday_weeds := wednesday_weeds / 5
  let friday_weeds := thursday_weeds - 10
  tuesday_weeds + wednesday_weeds + thursday_weeds + friday_weeds = 120 :=
by
  intros
  let tuesday_weeds := 25
  let wednesday_weeds := 3 * tuesday_weeds
  let thursday_weeds := wednesday_weeds / 5
  let friday_weeds := thursday_weeds - 10
  sorry

end sarah_total_weeds_l702_702962


namespace subsets_with_perfect_square_intersections_l702_702844

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def valid_intersections (S : finset (finset ℕ)) : Prop :=
  ∀ (A B : finset ℕ), A ∈ S → B ∈ S → is_perfect_square (finset.card (A ∩ B))

noncomputable def exists_subsets (X : finset ℕ) : Prop := 
  ∃ S : finset (finset ℕ), finset.card S ≥ 1111 ∧ valid_intersections S

theorem subsets_with_perfect_square_intersections : 
  exists_subsets (finset.range 101) :=
sorry

end subsets_with_perfect_square_intersections_l702_702844


namespace matrix_eigenvalue_problem_l702_702117

theorem matrix_eigenvalue_problem (k : ℝ) (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) :
  ((3*x + 4*y = k*x) ∧ (6*x + 3*y = k*y)) → k = 3 :=
by
  sorry

end matrix_eigenvalue_problem_l702_702117


namespace calculate_result_l702_702764

theorem calculate_result : (2.5 - 0.3) * 0.25 = 0.55 :=
by
  norm_num

end calculate_result_l702_702764


namespace find_profit_percentage_l702_702872

theorem find_profit_percentage (h : (m + 8) / (1 - 0.08) = m + 10) : m = 15 := sorry

end find_profit_percentage_l702_702872


namespace sum_of_a_and_b_l702_702509

theorem sum_of_a_and_b
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : (∃ (c d e : ℝ), {a, b, -2} = {c, d, e} ∧ (c - d = d - e ∨ d / c = e / d ∧ b = d ∧ -2 = e ∧ a = c)) ∨ (c, d, e : ℝ), {c, d, e} = {a, -2, b} ∧ (c - d = d - e ∨ d / c = e / d ∧ b = d ∧ -2 = e ∧ a = c)))
  : a + b = 8 :=
sorry

end sum_of_a_and_b_l702_702509


namespace compute_product_l702_702358

variables (x1 y1 x2 y2 x3 y3 : ℝ)

def condition1 (x y : ℝ) : Prop :=
  x^3 - 3 * x * y^2 = 2017

def condition2 (x y : ℝ) : Prop :=
  y^3 - 3 * x^2 * y = 2016

theorem compute_product :
  condition1 x1 y1 → condition2 x1 y1 →
  condition1 x2 y2 → condition2 x2 y2 →
  condition1 x3 y3 → condition2 x3 y3 →
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 1008 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end compute_product_l702_702358


namespace amount_paid_l702_702633

theorem amount_paid (lemonade_price_per_cup sandwich_price_per_item change_received : ℝ) 
    (num_lemonades num_sandwiches : ℕ)
    (h1 : lemonade_price_per_cup = 2) 
    (h2 : sandwich_price_per_item = 2.50) 
    (h3 : change_received = 11) 
    (h4 : num_lemonades = 2) 
    (h5 : num_sandwiches = 2) : 
    (lemonade_price_per_cup * num_lemonades + sandwich_price_per_item * num_sandwiches + change_received = 20) :=
by
  sorry

end amount_paid_l702_702633


namespace jace_travel_distance_l702_702899

-- Define the conditions
def speed : ℕ := 60 -- miles per hour
def time1 : ℕ := 4 -- hours
def time2 : ℕ := 9 -- hours

-- State the theorem
theorem jace_travel_distance : (speed * time1 + speed * time2) = 780 := 
by 
  have distance_first_period := speed * time1
  have distance_second_period := speed * time2
  have total_distance := distance_first_period + distance_second_period
  show (speed * time1 + speed * time2) = 780, -- Concluding the proof objective
  sorry

end jace_travel_distance_l702_702899


namespace midpoint_product_of_distances_intersection_l702_702888

section proof_problem

variables {t : ℝ} {x y : ℝ}

def curve_C1 (t : ℝ) : (ℝ × ℝ) :=
(x = 4 * t^2, y = 4 * t)

def curve_C2 (θ ρ : ℝ) : Prop :=
ρ * cos (θ + π / 4) = sqrt 2 / 2

theorem midpoint_product_of_distances_intersection (A B P : ℝ × ℝ) :
  let x1 := 3 + sqrt 5
  let y1 := 2 * (sqrt 3 - 1)
  let x2 := 3 - sqrt 5
  let y2 := -2 * (sqrt 3 - 1)
  ∃θ ρ, 
    (curve_C2 θ ρ → ∃x y, (curve_C1 x y ↔ ((x - P.1)^2 + (y - P.2)^2 = 16))) :=
sorry

end proof_problem

end midpoint_product_of_distances_intersection_l702_702888


namespace unique_function_satisfying_conditions_l702_702590

section
  variable (f : ℚ → ℚ)

  def satisfies_conditions (f : ℚ → ℚ) : Prop :=
    (∃ a : ℚ, f a ∉ ℤ) ∧
    (∀ x y : ℚ, f (x + y) - f x - f y ∈ ℤ ∧ f (x * y) - f x * f y ∈ ℤ)

  theorem unique_function_satisfying_conditions
    (f : ℚ → ℚ)
    (hf : satisfies_conditions f) :
    f = id :=
  sorry
end

end unique_function_satisfying_conditions_l702_702590


namespace compounding_frequency_two_l702_702959

noncomputable def compound_interest_amount (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem compounding_frequency_two :
  ∃ n : ℕ, n = 2 ∧ compound_interest_amount 3000 0.10 n 1 = 3307.5 :=
by
  use 2
  split
  { refl }
  { sorry }

end compounding_frequency_two_l702_702959


namespace no_perfect_power_in_sequence_l702_702994

def sequence : ℕ → ℕ
| 0       => 2
| 1       => 12
| (n + 2) => 6 * (sequence (n + 1)) - (sequence n)

theorem no_perfect_power_in_sequence :
  ∀ n, ¬ ∃ k m : ℕ, m > 1 ∧ sequence n = k ^ m := sorry

end no_perfect_power_in_sequence_l702_702994


namespace parabola_equation_l702_702986

theorem parabola_equation
  (k : ℝ)
  (axis_parallel_x : ∀ x, axis_parallel_x = x)
  (vertex : ∃ y, vertex = (0, y) ∧ y = 2)
  (focus_y : ∃ y, focus_y = 2)
  (passes_through_point : ∀ x y, passes_through_point (2, 6)) :
  
  ∃ a b c d e f : ℤ,  y^2 - 8 * x - 4 * y + 4 = 0 ∧ c > 0 ∧ gcd a b c d e f = 1 := by
  sorry

end parabola_equation_l702_702986


namespace ratio_squirrels_to_raccoons_l702_702900

def animals_total : ℕ := 84
def raccoons : ℕ := 12
def squirrels : ℕ := animals_total - raccoons

theorem ratio_squirrels_to_raccoons : (squirrels : ℚ) / raccoons = 6 :=
by
  sorry

end ratio_squirrels_to_raccoons_l702_702900


namespace find_BD_in_triangle_l702_702260

theorem find_BD_in_triangle (A B C D : Type)
  (distance_AC : Float) (distance_BC : Float)
  (distance_AD : Float) (distance_CD : Float)
  (hAC : distance_AC = 10)
  (hBC : distance_BC = 10)
  (hAD : distance_AD = 12)
  (hCD : distance_CD = 5) :
  ∃ (BD : Float), BD = 6.85435 :=
by 
  sorry

end find_BD_in_triangle_l702_702260


namespace ratio_Sachin_Rahul_l702_702319

-- Definitions: Sachin's age (S) is 63, and Sachin is younger than Rahul by 18 years.
def Sachin_age : ℕ := 63
def Rahul_age : ℕ := Sachin_age + 18

-- The problem: Prove the ratio of Sachin's age to Rahul's age is 7/9.
theorem ratio_Sachin_Rahul : (Sachin_age : ℚ) / (Rahul_age : ℚ) = 7 / 9 :=
by 
  -- The proof will go here, but we are skipping the proof as per the instructions.
  sorry

end ratio_Sachin_Rahul_l702_702319


namespace max_value_of_f_at_a_eq_2_tangent_slopes_reciprocal_bounds_l702_702527

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x - a * (x - 1)
noncomputable def g (x : ℝ) : ℝ := Real.exp x

-- First proof problem: proving the maximum of f(x) when a = 2
theorem max_value_of_f_at_a_eq_2 : 
  ∃ x : ℝ, f x 2 = 1 - Real.log 2 ∧ ∀ y : ℝ, y ≠ x → f y 2 ≤ f x 2 := sorry

-- Second proof problem: proving the inequality involving a and the tangent lines
theorem tangent_slopes_reciprocal_bounds (a : ℝ) (h : a ≠ 0) : 
  (∃ l1 l2 : ℝ → ℝ, 
    (∀ x, l1 x = (log x - a * (x - 1)) ∧ l1 0 = 0) ∧ (∀ x, l2 x = Real.exp x ∧ l2 0 = 0) ∧ 
    (∀ x, differentiable_at ℝ l1 x ∧ differentiable_at ℝ l2 x) ∧
    (∃ x1 x2 : ℝ, l1' x1 * l2' x2 = 1)) → 
  (Real.exp 1 - 1) / Real.exp 1 < a ∧ a < (Real.exp (2 : ℝ) - 1) / Real.exp 1 := sorry

end max_value_of_f_at_a_eq_2_tangent_slopes_reciprocal_bounds_l702_702527


namespace longest_side_of_triangle_l702_702228

theorem longest_side_of_triangle (a b c : ℝ) (h_deg_ratio: (a / b = 1 / 2) ∧ (a / c = 1 / 3) ∧ (a + b + c = 180))
  (shortest_side : ℝ) (h_shortest : shortest_side = 5) :
  let h₃₀₆₀₉₀ := 30 * (pi / 180) in
  let h₆₀ := 60 * (pi / 180) in
  let h₉₀ := 90 * (pi / 180) in
  let shortest := shortest_side in
  let longest := 2 * shortest in
  longest = 10 :=
by
  sorry

end longest_side_of_triangle_l702_702228


namespace simplify_expression_l702_702967

theorem simplify_expression (x : ℝ) : 5 * x + 7 * x - 3 * x = 9 * x :=
by
  sorry

end simplify_expression_l702_702967


namespace line_point_relation_l702_702293

theorem line_point_relation (x1 y1 x2 y2 a1 b1 c1 a2 b2 c2 : ℝ)
  (h1 : a1 * x1 + b1 * y1 = c1)
  (h2 : a2 * x2 + b2 * y2 = c2)
  (h3 : a1 + b1 = c1)
  (h4 : a2 + b2 = 2 * c2)
  (h5 : dist (x1, y1) (x2, y2) ≥ (Real.sqrt 2) / 2) :
  c1 / a1 + a2 / c2 = 3 := 
sorry

end line_point_relation_l702_702293


namespace monochromatic_triangle_probability_l702_702471

open Classical

noncomputable def hexagon_edges : ℕ := 15  -- Total number of edges in K_6

-- Probability a single triangle is not monochromatic
noncomputable def prob_not_monochromatic : ℝ := 3/4

-- Probability that at least one triangle in K_6 is monochromatic
noncomputable def prob_monochromatic_triangle : ℝ := 1 - (prob_not_monochromatic)^20

theorem monochromatic_triangle_probability : 
  prob_monochromatic_triangle ≈ 0.99683 := 
by 
  -- The use of approximation here is abstract; in practice, you would detail the proof steps.
  sorry

end monochromatic_triangle_probability_l702_702471


namespace find_missing_number_l702_702257

noncomputable def missing_number_in_proportion : ℝ :=
  let x := 2 in
  x

theorem find_missing_number (x : ℝ) : 
  (x / 5) = ((4 / 3) / 3.333333333333333) ↔ x = 2 :=
by
  sorry

end find_missing_number_l702_702257


namespace exists_integer_coordinates_l702_702898

theorem exists_integer_coordinates :
  ∃ (x y : ℤ), (x^2 + y^2) = 2 * 2017^2 + 2 * 2018^2 :=
by
  sorry

end exists_integer_coordinates_l702_702898


namespace parallel_line_slope_l702_702001

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℚ, m = 1 / 2 :=
by {
  use 1 / 2,
  sorry
}

end parallel_line_slope_l702_702001


namespace digit_B_in_4B52B_divisible_by_9_l702_702648

theorem digit_B_in_4B52B_divisible_by_9 (B : ℕ) (h : (2 * B + 11) % 9 = 0) : B = 8 :=
by {
  sorry
}

end digit_B_in_4B52B_divisible_by_9_l702_702648


namespace find_x_l702_702513

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 80) : x = 26 :=
by 
  sorry

end find_x_l702_702513


namespace Deepak_investment_approx_l702_702068

noncomputable def Deepak_investment (total_profit : ℝ) (Deepak_share : ℝ) (Anand_investment : ℝ) : ℝ := 
  (Anand_investment * Deepak_share) / (total_profit - Deepak_share)

theorem Deepak_investment_approx : 
  Deepak_investment 1380 810.28 2250 ≈ 3200 :=
sorry

end Deepak_investment_approx_l702_702068


namespace tangent_line_at_a_eq_1_range_of_a_l702_702142

noncomputable def f (x a : ℝ) := (2 - a) * x - 2 * (1 + Real.log x) + a
noncomputable def g (x : ℝ) := (Real.exp 1) * x / (Real.exp x)
noncomputable def h (x a x0 : ℝ) := (2 - a) * (x - 1) - (Real.exp (1 - x0)) * x0

theorem tangent_line_at_a_eq_1 : 
  (f 1 1 = 0) ∧ 
  (∂ x, f x 1)' = λ x, (1 - 2 / x) ∧ 
  (∂ x, f x 1)' 1 = -1 ∧ 
  ∃ (y : ℝ), y = -1, true := 
sorry

theorem range_of_a :
  (∀ x0 : ℝ, x0 ∈ Icc 0 (Real.exp 1) → (∀ x : ℝ, x ∈ Icc 0 (Real.exp 2) → h x a x0 = 0 → f x = g x0)) →
  a ≤ 2 - 5 / (Real.exp 2 - 1) :=
sorry

end tangent_line_at_a_eq_1_range_of_a_l702_702142


namespace find_RU_l702_702261

theorem find_RU
  (D E F T S R U : Type)
  [metric_space D] [metric_space E] [metric_space F]
  [metric_space T] [metric_space S] [metric_space R] [metric_space U]
  [has_dist D] [has_dist E] [has_dist F]
  (DE DF EF : ℝ)
  (DE_eq : DE = 130) (DF_eq : DF = 110) (EF_eq : EF = 140)
  (is_angle_bisector_D : ∃ T, angle_bisector D E F T)
  (is_angle_bisector_E : ∃ S, angle_bisector E F D S)
  (is_perpendicular_F_ES : ∃ R, perpendicular F E S R)
  (is_perpendicular_F_DT : ∃ U, perpendicular F D T U) :
  dist R U = 60 := sorry

end find_RU_l702_702261


namespace cafe_purchase_max_items_l702_702230

theorem cafe_purchase_max_items (total_money sandwich_cost soft_drink_cost : ℝ) (total_money_pos sandwich_cost_pos soft_drink_cost_pos : total_money > 0 ∧ sandwich_cost > 0 ∧ soft_drink_cost > 0) :
    total_money = 40 ∧ sandwich_cost = 5 ∧ soft_drink_cost = 1.50 →
    ∃ s d : ℕ, s + d = 10 ∧ total_money = sandwich_cost * s + soft_drink_cost * d :=
by
  sorry

end cafe_purchase_max_items_l702_702230


namespace george_shots_l702_702495

theorem george_shots (initial_shots made_initial: ℕ) (additional_shots made_additional: ℕ)
  (initial_percentage final_percentage: ℝ) :
  initial_shots = 30 →
  made_initial = (0.60 * initial_shots).to_nat →
  additional_shots = 10 →
  final_percentage = 0.63 →
  ((made_initial + made_additional) / (initial_shots + additional_shots) : ℝ) = final_percentage →
  made_additional = 7 :=
by
  sorry

end george_shots_l702_702495


namespace product_first_two_terms_l702_702995

-- Define general parameters
variable (a_7 : ℕ) (d : ℕ)

-- The conditions
def condition1 := a_7 = 25
def condition2 := d = 3

-- Define the first term of the arithmetic sequence
def a₁ := (25 - 6 * 3)

-- Define the second term
def a₂ := (25 - 6 * 3) + 3

-- Prove the product of the first two terms of the arithmetic sequence
theorem product_first_two_terms (h1 : a_7 = 25) (h2 : d = 3) : (a₁ 25 3) * (a₂ 25 3) = 70 :=
by
  sorry

end product_first_two_terms_l702_702995


namespace minimum_days_on_duty_l702_702367

theorem minimum_days_on_duty (m n k : ℕ) (days : ℕ) 
  (h1 : 9 * m + 10 * n = 33 * k) 
  (h2 : ∀ b : ℕ, b mod 3 = 0) : 
  days = 7 := 
sorry

end minimum_days_on_duty_l702_702367


namespace g_inequality_l702_702191

def g (x : ℝ) : ℝ := Real.exp (1 + x^2) - 1 / (1 + x^2) + abs x

theorem g_inequality {x : ℝ} (h : g (x - 1) > g (3 * x + 1)) : -1 < x ∧ x < 0 :=
by
  sorry

end g_inequality_l702_702191


namespace paula_friends_count_l702_702943

theorem paula_friends_count :
  ∀ (candies_owned: ℕ) (candies_bought: ℕ) (candies_per_friend: ℕ),
  candies_owned = 20 →
  candies_bought = 4 →
  candies_per_friend = 4 →
  (candies_owned + candies_bought) / candies_per_friend = 6 :=
by
  intros candies_owned candies_bought candies_per_friend h_owned h_bought h_per_friend
  rw [h_owned, h_bought, h_per_friend]
  norm_num
  sorry

end paula_friends_count_l702_702943


namespace number_of_BMWs_sold_l702_702731

-- Defining the percentages of Mercedes, Toyota, and Acura cars sold
def percentageMercedes : ℕ := 18
def percentageToyota  : ℕ := 25
def percentageAcura   : ℕ := 15

-- Defining the total number of cars sold
def totalCars : ℕ := 250

-- The theorem to be proved
theorem number_of_BMWs_sold : (totalCars * (100 - (percentageMercedes + percentageToyota + percentageAcura)) / 100) = 105 := by
  sorry -- Proof to be filled in later

end number_of_BMWs_sold_l702_702731


namespace middle_number_is_11_l702_702662

theorem middle_number_is_11 (a b c : ℕ) (h1 : a + b = 18) (h2 : a + c = 22) (h3 : b + c = 26) (h4 : c - a = 10) :
  b = 11 :=
by
  sorry

end middle_number_is_11_l702_702662


namespace hypotenuse_equality_l702_702610

open Real

noncomputable def log_8_125 : ℝ := log 125 / log 8
noncomputable def log_4_125 : ℝ := log 125 / log 4

def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

noncomputable def h : ℝ := hypotenuse log_8_125 log_4_125

theorem hypotenuse_equality : 16^h = 5^20 :=
by
  -- Bring in the conditions defined earlier
  have log_8_125_eval : log_8_125 = log 5 / 3 := sorry
  have log_4_125_eval : log_4_125 = 1.5 * log 5 := sorry
  
  -- Calculate h based on hypotenuse
  have h_eq : h = 5 * log 5 := sorry

  -- Final computation for 16^h
  calc
    16^h = 2^(4 * h) : by sorry
        ... = 2^(4 * (5 * log 5)) : by rw h_eq
        ... = 2^(20 * log 5) : by sorry
        ... = 5^20 : by sorry

end hypotenuse_equality_l702_702610


namespace celsius_to_fahrenheit_l702_702369

theorem celsius_to_fahrenheit (C : ℝ) : (C * 9 / 5) + 32 = 122 ↔ C = 50 := 
by
  split
  · intro h
    have := congr_arg (λ x, (x - 32) * 5 / 9) h
    simp at this
    exact this.symm
  · intro h
    simp [h]
    norm_num
sorry

end celsius_to_fahrenheit_l702_702369


namespace expectation_of_eta_l702_702818

noncomputable def b_expectation (n : ℕ) (p : ℚ) : ℚ :=
n * p

noncomputable def eta_expectation (ξ : ℚ) : ℚ :=
2 * ξ - 1

theorem expectation_of_eta :
  ∀ (ξ: ℚ), ξ = 5 * (1 / 3) → eta_expectation ξ = 7 / 3 :=
by
  assume ξ hξ,
  rw [hξ],
  have h1 : ξ = 5 / 3 := by sorry,
  rw [h1],
  simp [eta_expectation],
  sorry

end expectation_of_eta_l702_702818


namespace max_triangle_area_l702_702996

noncomputable def maxArea (a b c A B C : ℝ) : ℝ :=
  1/2 * a * c * Real.sin B

theorem max_triangle_area
  (a b c A B C : ℝ)
  (habc : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_angles : A + B + C = Real.pi)
  (h_side_angles : ∀ {x y z X Y Z : ℝ}, x^2 = y^2 + z^2 - 2 * y * z * Real.cos X → x = sqrt (y^2 + z^2 - 2 * y * z * Real.cos X))
  (hLawOfCosines : a = sqrt (b^2 + c^2 - 2 * b * c * Real.cos A)∧ b = sqrt (a^2 + c^2 - 2 * a * c * Real.cos B) ∧ c = sqrt (a^2 + b^2 - 2 * a * b * Real.cos C))
  (h_sum_sides: a + c = 4)
  (h_identity: Real.sin A * (1 + Real.cos B) = (2 - Real.cos A) * Real.sin B) :
  maxArea a b c A B C = sqrt 3 :=
sorry

end max_triangle_area_l702_702996


namespace circles_on_parabola_tangent_to_line_pass_through_fixed_point_l702_702722

theorem circles_on_parabola_tangent_to_line_pass_through_fixed_point :
  ∀ (x y : ℝ), (y^2 = 8 * x) ∧ (abs (x + 2) = sqrt ((x - 2)^2 + y^2)) → ∃ p : ℝ × ℝ, p = (2, 0) :=
by
  -- the conditions and start of the proof definition
  intros x y h
  sorry

end circles_on_parabola_tangent_to_line_pass_through_fixed_point_l702_702722


namespace leaves_fall_total_l702_702238

theorem leaves_fall_total : 
  let planned_cherry_trees := 7 
  let planned_maple_trees := 5 
  let actual_cherry_trees := 2 * planned_cherry_trees
  let actual_maple_trees := 3 * planned_maple_trees
  let leaves_per_cherry_tree := 100
  let leaves_per_maple_tree := 150
  actual_cherry_trees * leaves_per_cherry_tree + actual_maple_trees * leaves_per_maple_tree = 3650 :=
by
  let planned_cherry_trees := 7 
  let planned_maple_trees := 5 
  let actual_cherry_trees := 2 * planned_cherry_trees
  let actual_maple_trees := 3 * planned_maple_trees
  let leaves_per_cherry_tree := 100
  let leaves_per_maple_tree := 150
  sorry

end leaves_fall_total_l702_702238


namespace range_of_k_l702_702868

theorem range_of_k (k : ℝ) : 
  (∃ a b : ℝ, x^2 + ky^2 = 2 ∧ a^2 = 2/k ∧ b^2 = 2 ∧ a > b) → 0 < k ∧ k < 1 :=
by {
  sorry
}

end range_of_k_l702_702868


namespace cos_squared_alpha_plus_pi_over_4_l702_702808

theorem cos_squared_alpha_plus_pi_over_4 (α : ℝ) (h : sin (2 * α) = 2 / 3) :
  cos ^ 2 (α + π / 4) = 1 / 6 :=
by sorry

end cos_squared_alpha_plus_pi_over_4_l702_702808


namespace max_value_f_l702_702483

noncomputable def f (x : ℝ) : ℝ :=
  sqrt (x + 20) + sqrt (20 - x) + sqrt (2 * x) + sqrt (30 - x)

theorem max_value_f : ∃ x ∈ set.Icc (0 : ℝ) 20, f x = sqrt 630 := by
  sorry

end max_value_f_l702_702483


namespace proof_ac_plus_bd_l702_702861

theorem proof_ac_plus_bd (a b c d : ℝ)
  (h1 : a + b + c = 10)
  (h2 : a + b + d = -6)
  (h3 : a + c + d = 0)
  (h4 : b + c + d = 15) :
  ac + bd = -130.111 := 
by
  sorry

end proof_ac_plus_bd_l702_702861


namespace least_possible_value_of_p_and_q_l702_702224

theorem least_possible_value_of_p_and_q 
  (p q : ℕ) 
  (h1 : p > 1) 
  (h2 : q > 1) 
  (h3 : 15 * (p + 1) = 29 * (q + 1)) : 
  p + q = 45 := 
sorry -- proof to be filled in

end least_possible_value_of_p_and_q_l702_702224


namespace max_savings_after_ticket_split_l702_702297

-- Definitions by conditions
def saved_base8 : ℕ := 0b5273 -- Max's savings in base 8
def ticket_cost_base10 : ℕ := 1500 -- Cost of the airline ticket in base 10

-- Theorem statement, proving that the remaining amount of money is 1247 dollars
theorem max_savings_after_ticket_split :
  (saved_base8 * 8^3 + saved_base8 * 8^2 + saved_base8 * 8 + saved_base8) - ticket_cost_base10 = 1247 :=
sorry

end max_savings_after_ticket_split_l702_702297


namespace max_S_value_proof_l702_702973

noncomputable def max_value_S (n : ℕ) (a : Fin n → ℝ) : ℝ :=
  (∑ i, (i : ℕ).pow 2 * a i) * (∑ i, a i / (i : ℕ)) ^ 2

theorem max_S_value_proof (n : ℕ) (a : Fin n → ℝ) (h1 : n ≥ 2) (h2 : ∀ i, 0 ≤ a i) (h3 : ∑ i, a i = 1) :
  max_value_S n a ≤ 4 * (n^2 + n + 1)^3 / (27 * n^2 * (n + 1)^2) :=
sorry

end max_S_value_proof_l702_702973


namespace convert_to_spherical_l702_702461

-- Define the given rectangular coordinates
def point_rect := (4 * Real.sqrt 2, -4, 4)

-- Define the conversion from rectangular coordinates to spherical coordinates
def spherical_coords := (8, 7 * Real.pi / 4, Real.pi / 3)

-- Define the conditions for spherical coordinates
def valid_spherical_coords (ρ θ φ : ℝ) : Prop := 
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi

-- State the theorem
theorem convert_to_spherical : 
  ∃ (ρ θ φ : ℝ), valid_spherical_coords ρ θ φ ∧ 
  (point_rect = (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)) ∧ 
  (ρ, θ, φ) = spherical_coords :=
by
  sorry

end convert_to_spherical_l702_702461


namespace mean_median_mode_l702_702089

theorem mean_median_mode (s : List ℕ) (h_s : s = [1, 1, 2, 2, 2, 5, 5, 5, 5, 7]) : 
  let mean := (s.sum.toReal / s.length) in
  let median := ((s.nthLe (s.length / 2 - 1) (by linarith) + s.nthLe (s.length / 2) (by linarith)).toReal / 2) in
  let mode := 5 in
  median = 3.5 ∧ mean = 3.5 ∧ mean < mode := 
by
  sorry

end mean_median_mode_l702_702089


namespace part1_part2_l702_702398

variable (a b m n : ℝ) (h_a : a > 0) (h_b : b > 0) (h_m : m > 0) (h_n : n > 0)

theorem part1 :
  (m^2/a + n^2/b ≥ (m+n)^2/(a+b))
  := sorry

variable (x : ℝ) (hx : 0 < x) (hx2 : x < 1/2)

theorem part2 :
  (∏ (2/x + 9/(1-2*x)) = 25 ∧ x = 1/5)
  := sorry

end part1_part2_l702_702398


namespace count_special_leap_years_l702_702422

def is_special_leap_year (y : ℕ) : Prop :=
  (y % 100 = 0) ∧ ((y % 800 = 200) ∨ (y % 800 = 600))

def is_in_range (y : ℕ) : Prop := 
  (2000 ≤ y ∧ y ≤ 5000)

def valid_years : ℕ := 
  Nat.card {y : ℕ // is_special_leap_year y ∧ is_in_range y}.1

theorem count_special_leap_years : valid_years = 9 :=
by
  -- The proof will be provided here
  sorry

end count_special_leap_years_l702_702422


namespace quadratic_roots_max_value_l702_702239

theorem quadratic_roots_max_value (t q u₁ u₂ : ℝ)
  (h1 : u₁ + u₂ = t)
  (h2 : u₁ * u₂ = q)
  (h3 : u₁ + u₂ = u₁^2 + u₂^2)
  (h4 : u₁ + u₂ = u₁^4 + u₂^4) :
  (1 / u₁^2009 + 1 / u₂^2009) ≤ 2 :=
sorry

-- Explaination: 
-- This theorem states that given the conditions on the roots u₁ and u₂ of the quadratic equation, 
-- the maximum possible value of the expression (1 / u₁^2009 + 1 / u₂^2009) is 2.

end quadratic_roots_max_value_l702_702239


namespace gumball_water_wednesday_l702_702207

theorem gumball_water_wednesday :
  ∀ (total_weekly_water monday_thursday_saturday_water tuesday_friday_sunday_water : ℕ),
  total_weekly_water = 60 →
  monday_thursday_saturday_water = 9 →
  tuesday_friday_sunday_water = 8 →
  total_weekly_water - (monday_thursday_saturday_water * 3 + tuesday_friday_sunday_water * 3) = 9 :=
by
  intros total_weekly_water monday_thursday_saturday_water tuesday_friday_sunday_water
  sorry

end gumball_water_wednesday_l702_702207


namespace solve_for_x_l702_702323

-- Definitions of conditions
def sqrt_81_as_3sq : ℝ := (81 : ℝ)^(1/2)  -- sqrt(81)
def sqrt_81_as_3sq_simplified : ℝ := (3^4 : ℝ)^(1/2)  -- equivalent to (3^2) since 81 = 3^4

-- Theorem and goal statement
theorem solve_for_x :
  sqrt_81_as_3sq = sqrt_81_as_3sq_simplified →
  (3 : ℝ)^(3 * (2/3)) = sqrt_81_as_3sq :=
by
  -- Placeholder for the proof
  sorry

end solve_for_x_l702_702323


namespace coincide_rest_days_in_first_500_days_l702_702441

-- Definitions for cycles
def Al_cycle := 6
def Barb_cycle := 6
def seminar_cycle := 15
def total_days := 500

-- Functions that determine rest days
def Al_rest_days (n : ℕ) : Prop := n % Al_cycle = 4 ∨ n % Al_cycle = 5
def Barb_rest_days (n : ℕ) : Prop := n % Barb_cycle = 5

def is_seminar_day (n : ℕ) : Prop := n % seminar_cycle = 0

-- Main problem statement
theorem coincide_rest_days_in_first_500_days :
  (∑ n in Finset.range total_days, (¬is_seminar_day n ∧ Al_rest_days n ∧ Barb_rest_days n).toIndicator) = 16 := sorry

end coincide_rest_days_in_first_500_days_l702_702441


namespace commute_distance_l702_702066

noncomputable def distance_to_work (total_time : ℕ) (speed_to_work : ℕ) (speed_to_home : ℕ) : ℕ :=
  let d := (speed_to_work * speed_to_home * total_time) / (speed_to_work + speed_to_home)
  d

-- Given conditions
def speed_to_work : ℕ := 45
def speed_to_home : ℕ := 30
def total_time : ℕ := 1

-- Proof problem statement
theorem commute_distance : distance_to_work total_time speed_to_work speed_to_home = 18 :=
by
  sorry

end commute_distance_l702_702066


namespace factor_expr_l702_702455

variable (x : ℝ)

def expr : ℝ := (20 * x ^ 3 + 100 * x - 10) - (-5 * x ^ 3 + 5 * x - 10)

theorem factor_expr :
  expr x = 5 * x * (5 * x ^ 2 + 19) :=
by
  sorry

end factor_expr_l702_702455


namespace number_of_proper_subsets_l702_702830

-- Define the universal set A and prove that the number of proper subsets of A is 6
theorem number_of_proper_subsets (A : Finset ℕ) (h : A = {0, 1, 2}) : A.card = 3 ∧ (A.powerset.card - 2) = 6 :=
by
  sorry

end number_of_proper_subsets_l702_702830


namespace theater_balcony_seat_cost_l702_702064

theorem theater_balcony_seat_cost :
  ∀ (O B : ℕ),
    -- Tickets in orchestra cost $12
    ∀ orchestra_ticket_cost : ℕ,
    orchestra_ticket_cost = 12 →
    -- Total tickets sold = 360
    O + (O + 140) = 360 →
    -- Total cost of all tickets = 3320
    O * 12 + (O + 140) * B = 3320 →
    -- The cost of a balcony seat
    B = 8 := 
by
  intros O B orchestra_ticket_cost Hcost Htotal_tickets Htotal_cost,
  sorry

end theater_balcony_seat_cost_l702_702064


namespace find_interval_l702_702870

def f (x : ℝ) : ℝ := -x^2

-- Assumptions
variables (a b : ℝ)
hypothesis ha : a ≤ b
hypothesis hf_min : ∀ x ∈ set.Icc a b, f x ≥ 2 * a
hypothesis hf_max : ∀ x ∈ set.Icc a b, f x ≤ 2 * b

-- Goal: prove the interval is [1, 3]
theorem find_interval : [a, b] = [1, 3] :=
by
  sorry

end find_interval_l702_702870


namespace text_messages_December_l702_702583

-- Definitions of the number of text messages sent each month
def text_messages_November := 1
def text_messages_January := 4
def text_messages_February := 8
def doubling_pattern (a b : ℕ) : Prop := b = 2 * a

-- Prove that Jared sent 2 text messages in December
theorem text_messages_December : ∃ x : ℕ, 
  doubling_pattern text_messages_November x ∧ 
  doubling_pattern x text_messages_January ∧ 
  doubling_pattern text_messages_January text_messages_February ∧ 
  x = 2 :=
by
  sorry

end text_messages_December_l702_702583


namespace count_valid_n_l702_702754

theorem count_valid_n:
  ( ∃ f: ℕ → ℕ, ∀ n, (0 < n ∧ n < 2012 → 7 ∣ (2^n - n^2) ↔ 7 ∣ (f n)) ∧ f 2012 = 576) → 
  ∃ valid_n_count: ℕ, valid_n_count = 576 := 
sorry

end count_valid_n_l702_702754


namespace variance_of_numbers_l702_702826

noncomputable def variance (s : List ℕ) : ℚ :=
  let mean := (s.sum : ℚ) / s.length
  let sqDiffs := s.map (λ n => (n - mean) ^ 2)
  sqDiffs.sum / s.length

def avg_is_34 (s : List ℕ) : Prop := (s.sum : ℚ) / s.length = 34

theorem variance_of_numbers (x : ℕ) 
  (h : avg_is_34 [31, 38, 34, 35, x]) : variance [31, 38, 34, 35, x] = 6 := 
by
  sorry

end variance_of_numbers_l702_702826


namespace matrix_eigenvalue_problem_l702_702118

theorem matrix_eigenvalue_problem (k : ℝ) (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) :
  ((3*x + 4*y = k*x) ∧ (6*x + 3*y = k*y)) → k = 3 :=
by
  sorry

end matrix_eigenvalue_problem_l702_702118


namespace factor_expr_l702_702456

variable (x : ℝ)

def expr : ℝ := (20 * x ^ 3 + 100 * x - 10) - (-5 * x ^ 3 + 5 * x - 10)

theorem factor_expr :
  expr x = 5 * x * (5 * x ^ 2 + 19) :=
by
  sorry

end factor_expr_l702_702456


namespace determine_x1_l702_702776

theorem determine_x1
  (x1 x2 x3 x4 : ℝ)
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1/3) :
  x1 = 4/5 :=
by
  sorry

end determine_x1_l702_702776


namespace exponent_calculation_l702_702540

theorem exponent_calculation (a m n : ℝ) (h1 : a^m = 3) (h2 : a^n = 2) : 
  a^(2 * m - 3 * n) = 9 / 8 := 
by
  sorry

end exponent_calculation_l702_702540


namespace economist_wins_by_choosing_method_1_l702_702703

variable (n : ℕ) (h_odd : n % 2 = 1) (h_greater_than_4 : n > 4)

-- Condition for Step 1: Lawyer divides coins into a and b
variable (a b : ℕ) (h_a_b : a + b = n) (h_a_2 : a ≥ 2) (h_b_2 : b ≥ 2) (h_a_lt_b : a < b)

-- Condition for Step 2: Economist divides a into x1 and x2, and b into y1 and y2
variable (x1 x2 y1 y2 : ℕ)
variable (h_x1_x2 : x1 + x2 = a) (h_y1_y2 : y1 + y2 = b)
variable (h_x1_1 : x1 ≥ 1) (h_x2_1 : x2 ≥ 1) (h_y1_1 : y1 ≥ 1) (h_y2_1 : y2 ≥ 1)
variable (h_x1_le_x2 : x1 ≤ x2) (h_y1_le_y2 : y1 ≤ y2)

-- Method 1: Economist takes largest and smallest parts

-- Method 2: Economist takes both middle parts

-- Method 3: Economist chooses method 1 or 2 and gives one coin to the lawyer

theorem economist_wins_by_choosing_method_1 :
  economist_strategy n = method1 :=
sorry

end economist_wins_by_choosing_method_1_l702_702703


namespace income_M_l702_702328

variable (M N O : ℝ)

theorem income_M (h1 : (M + N) / 2 = 5050) 
                  (h2 : (N + O) / 2 = 6250) 
                  (h3 : (M + O) / 2 = 5200) : 
                  M = 2666.67 := 
by 
  sorry

end income_M_l702_702328


namespace exists_k_for_A_mul_v_eq_k_mul_v_l702_702113

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 6, 3]

theorem exists_k_for_A_mul_v_eq_k_mul_v (v : Fin 2 → ℝ) (h : v ≠ 0) :
  (∃ k : ℝ, A.mul_vec v = k • v) →
  k = 3 + 2 * real.sqrt 6 ∨ k = 3 - 2 * real.sqrt 6 :=
by
  sorry

end exists_k_for_A_mul_v_eq_k_mul_v_l702_702113


namespace find_missing_number_l702_702716

theorem find_missing_number (x : ℝ) (h : 1 / ((1 / 0.03) + (1 / x)) = 0.02775) : abs (x - 0.370) < 0.001 := by
  sorry

end find_missing_number_l702_702716


namespace factor_expression_l702_702459

theorem factor_expression :
  let expr := (20 * x^3 + 100 * x - 10) - (-5 * x^3 + 5 * x - 10)
  expr = 5 * x * (5 * x^2 + 19) :=
by {
  let term1 := 20 * x^3 + 100 * x - 10,
  let term2 := -5 * x^3 + 5 * x - 10,
  have h : expr = term1 - term2,
  sorry
}

end factor_expression_l702_702459


namespace find_BC_length_l702_702893

namespace TrapezoidProblem

variables {A B C D : Point}
variables (AB CD AC : Line)

noncomputable def length_of_CD := 10
noncomputable def tan_D := 2
noncomputable def tan_B := 1.25

axiom AB_parallel_CD : Parallel AB CD
axiom AC_perpendicular_CD : Perpendicular AC CD
axiom AC_from_A_to_C : From A C AC
axiom CD_from_C_to_D : From C D CD

noncomputable def length_of_BC : ℝ := 4 * Real.sqrt 41

theorem find_BC_length (h1 : Parallel AB CD) 
                       (h2 : Perpendicular AC CD)
                       (h3 : From A C AC)
                       (h4 : From C D CD) 
                       (h5 : length_of_CD = 10) 
                       (h6 : tan_D = 2) 
                       (h7 : tan_B = 1.25) : 
                       length_of_BC = 4 * Real.sqrt 41 :=
sorry

end TrapezoidProblem

end find_BC_length_l702_702893


namespace triangle_BD_length_l702_702575

theorem triangle_BD_length 
  (A B C D : Type) 
  (hAC : AC = 8) 
  (hBC : BC = 8) 
  (hAD : AD = 6) 
  (hCD : CD = 5) : BD = 6 :=
  sorry

end triangle_BD_length_l702_702575


namespace median_of_data_set_with_mode_l702_702871

theorem median_of_data_set_with_mode :
  ∃ x : ℝ, let data_set := [3, 4, 5, 6, x, 8] in
  (list.mode data_set = 4) →
  list.median data_set = 4.5 :=
sorry

end median_of_data_set_with_mode_l702_702871


namespace cone_curved_surface_area_l702_702692

theorem cone_curved_surface_area (r l : ℝ) (π : ℝ) (h_r : r = 35) (h_l : l = 30) (h_π : π ≈ 3.14159) : 
    π * r * l ≈ 3299.34 :=
by {
  rw [h_r, h_l, h_π],
  norm_num,
  sorry
}

end cone_curved_surface_area_l702_702692


namespace least_distinct_values_theorem_l702_702737

noncomputable def least_distinct_values (n : ℕ) (mode_count : ℕ) (total_elements : ℕ) :=
  ∀ (d : ℕ), (mode_count = 12) → (list.length l = 2520) →
  (mode_num ∈ l) → (count mode_num l = 12) →
  ∃ x : ℕ, x = 229

theorem least_distinct_values_theorem : least_distinct_values 2520 12 :=
by sorry

end least_distinct_values_theorem_l702_702737


namespace regular_octagon_area_l702_702314

theorem regular_octagon_area : ∀ (R : ℝ),
  let d_b := 2 * R,
      d_c := 2 * R * Real.sin (Real.pi / 8)
  in R^2 * 4 * Real.sin (Real.pi / 8) = d_b * d_c :=
by
  intros R d_b d_c
  dsimp [d_b, d_c]
  sorry

end regular_octagon_area_l702_702314


namespace hotel_room_assignment_even_hotel_room_assignment_odd_l702_702878

def smallest_n_even (k : ℕ) (m : ℕ) (h1 : k = 2 * m) : ℕ :=
  100 * (m + 1)

def smallest_n_odd (k : ℕ) (m : ℕ) (h1 : k = 2 * m + 1) : ℕ :=
  100 * (m + 1) + 1

theorem hotel_room_assignment_even (k m : ℕ) (h1 : k = 2 * m) :
  ∃ n, n = smallest_n_even k m h1 ∧ n >= 100 :=
  by
  sorry

theorem hotel_room_assignment_odd (k m : ℕ) (h1 : k = 2 * m + 1) :
  ∃ n, n = smallest_n_odd k m h1 ∧ n >= 100 :=
  by
  sorry

end hotel_room_assignment_even_hotel_room_assignment_odd_l702_702878


namespace length_of_XY_l702_702235

theorem length_of_XY {α : Type} [inner_product_space ℝ α] [normed_group α] [normed_space ℝ α]
  (O A B Y X : α)
  (radius : ℝ)
  (h1 : dist O A = radius)
  (h2 : dist O B = radius)
  (h3 : ∠ A O B = π / 2)
  (OY_perpendicular : ∃ Y : α, (∠ O Y A = π / 2) ∧ (∃ X : α, collinear ℝ ({O, Y, X}) ∧ dist O X = (radius * real.sqrt 2) / 2))
  (radius_val : radius = 15) :
  dist Y X = 15 * (1 - real.sqrt 2 / 2) :=
sorry

end length_of_XY_l702_702235


namespace complex_conjugate_product_l702_702284

-- Here we define our condition that the magnitude of w is 15
variable (w : ℂ) (hw : Complex.abs w = 15)

-- We now state what we need to prove, which is w * conjugate(w) = 225
theorem complex_conjugate_product : w * Complex.conj w = 225 :=
by 
  sorry

end complex_conjugate_product_l702_702284


namespace range_a_l702_702821

variable (a : ℝ)

def p := (∀ x : ℝ, x^2 + x + a > 0)
def q := ∃ x y : ℝ, x^2 - 2 * a * x + 1 ≤ y

theorem range_a :
  ({a : ℝ | (p a ∧ ¬q a) ∨ (¬p a ∧ q a)} = {a : ℝ | a < -1} ∪ {a : ℝ | 1 / 4 < a ∧ a < 1}) := 
by
  sorry

end range_a_l702_702821


namespace can_construct_building_l702_702876

-- Definitions for height and distance of existing buildings
variable (height : Type)
variable (distance : Type)

-- Define the dominance condition
def dominant (A B : (height × distance)) : Prop :=
  | A.1 - B.1 | ≤ | A.2 - B.2 |

-- Define the condition that no two buildings are dominant to each other
def no_dominance (buildings : List (height × distance)) : Prop :=
  ∀ (A B : (height × distance)), A ∈ buildings → B ∈ buildings → A ≠ B → ¬dominant A B

-- The main statement
theorem can_construct_building (buildings: List (height × distance)) (x_S : distance) :
  no_dominance buildings →
  (∃ h_S : height, ∀ (A : (height × distance)), A ∈ buildings →
    | h_S - A.1 | > | x_S - A.2 | ) :=
begin
  sorry
end

end can_construct_building_l702_702876


namespace pyramid_dihedral_angle_l702_702657

noncomputable def dihedral_angle (k : ℝ) (h1 : 0 < k) (h2 : k < real.sqrt 3) : ℝ :=
2 * real.arcsin (k * real.sqrt 3 / 3)

theorem pyramid_dihedral_angle (k : ℝ) (h1 : 0 < k) (h2 : k < real.sqrt 3) :
  dihedral_angle k h1 h2 = 2 * real.arcsin (k * real.sqrt 3 / 3) :=
sorry

end pyramid_dihedral_angle_l702_702657


namespace problem_l702_702913

open_locale big_operators

variables {a b : ℕ → ℝ} {S T : ℕ → ℝ}

def arithmetic_seq (seq : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, seq (n + 1) = seq n + d

def sum_first_n_terms (seq : ℕ → ℝ) (n : ℕ) :=
  ∑ i in finset.range n, seq (i + 1)

theorem problem 
  (ha : arithmetic_seq a)
  (hb : arithmetic_seq b)
  (hS : ∀ n : ℕ, S n = sum_first_n_terms a n)
  (hT : ∀ n : ℕ, T n = sum_first_n_terms b n)
  (hRatio : ∀ n : ℕ, S n / T n = (3 * n - 1) / (n + 3)) :
  a 8 / (b 5 + b 11) = 11 / 9 :=
sorry

end problem_l702_702913


namespace limit_sequence_numerator_denominator_l702_702449

noncomputable def limit_sequence_expression : ℝ := 
  (λ n : ℕ, (n * (5 * n^2)^(1/3) + (9 * n^8 + 1)^(1/4)) / ((n + sqrt n) * sqrt (7 - n + n^2)))

theorem limit_sequence_numerator_denominator :
  tendsto limit_sequence_expression at_top (𝓝 (sqrt 3)) :=
begin
  sorry
end

end limit_sequence_numerator_denominator_l702_702449


namespace tan_values_l702_702581

theorem tan_values (x y : ℝ) (h1 : (cos x - sin x) / sin y = (2 * sqrt 2 / 5) * tan ((x + y) / 2))
  (h2 : (sin x + cos x) / cos y = - (5 / sqrt 2) * cot ((x + y) / 2)) :
  ∃ k : ℤ, tan (x + y) = -1 ∨ tan (x + y) = 20 / 21 ∨ tan (x + y) = -20 / 21 :=
sorry

end tan_values_l702_702581


namespace Eddie_number_divisibility_l702_702795

theorem Eddie_number_divisibility (n: ℕ) (h₁: n = 40) (h₂: n % 5 = 0): n % 2 = 0 := 
by
  sorry

end Eddie_number_divisibility_l702_702795


namespace least_gumballs_to_ensure_four_of_same_color_l702_702045

theorem least_gumballs_to_ensure_four_of_same_color 
  (red_gumballs white_gumballs blue_gumballs : ℕ) 
  (h_red : red_gumballs = 8)
  (h_white : white_gumballs = 10)
  (h_blue : blue_gumballs = 6) 
  : ∃ n, n = 10 ∧ (∀ m, m < 10 → ∃ (r w b : ℕ), r + w + b = m ∧ r ≤ 3 ∧ w ≤ 3 ∧ b ≤ 3) :=
by
  use 10
  split
  · refl
  · intros m h_m
    sorry

end least_gumballs_to_ensure_four_of_same_color_l702_702045


namespace limit_a_n_over_n_to_d_l702_702903

noncomputable def F (x : ℝ) : ℝ := 1 / ((2 - x - x^5) ^ 2011)

def a_n : ℕ → ℝ := sorry  -- This represents the coefficients in the power series expansion of F(x)

theorem limit_a_n_over_n_to_d (c : ℝ) (d : ℝ) (hF : ∀ (x : ℝ), F(x) = ∑ n in (Finset.range ?m_1), a_n n * x^n) (h_cd : c = 1 / (6^2011 * 2010!) ∧ d = 2010):
  (tendsto (λ (n : ℕ), a_n n / n^d) at_top (𝓝 c)) :=
begin
  sorry
end

end limit_a_n_over_n_to_d_l702_702903


namespace product_of_eccentricities_eq_one_l702_702512

-- Define the conditions
variable (m n : ℝ)
variable (F_1 F_2 P : ℝ × ℝ)
variable (m_pos : 0 < m)
variable (n_pos : 0 < n)
variable (ellipse_condition : ∀ x y : ℝ, (x, y) ∈ M ↔ x^2 / m^2 + y^2 / 2 = 1)
variable (hyperbola_condition : ∀ x y : ℝ, (x, y) ∈ N ↔ x^2 / n^2 - y^2 = 1)
variable (common_foci_condition : F_1 ≠ F_2 ∧ F_1 = F_2)
variable (common_point_condition : (P ∈ M ∧ P ∈ N) ∧ (P.fst = F_1.fst ∧ P.snd = F_2.snd))

-- Define the theorem
theorem product_of_eccentricities_eq_one 
  (ellipse_eccentricity : ℝ := sqrt(1 - (2 / m)^2))
  (hyperbola_eccentricity : ℝ := sqrt(1 + (1 / n)^2)) :
  (ellipse_eccentricity * hyperbola_eccentricity = 1) :=
sorry

end product_of_eccentricities_eq_one_l702_702512


namespace polynomial_factorization_l702_702008

-- Define the polynomial and its factorized form
def polynomial (x : ℝ) : ℝ := x^2 - 4*x + 4
def factorized_form (x : ℝ) : ℝ := (x - 2)^2

-- The theorem stating that the polynomial equals its factorized form
theorem polynomial_factorization (x : ℝ) : polynomial x = factorized_form x :=
by {
  sorry -- Proof skipped
}

end polynomial_factorization_l702_702008


namespace complex_conjugate_product_correct_l702_702287

noncomputable def complex_conjugate_product (w : ℂ) (h : abs w = 15) : ℂ :=
  w * conj w

theorem complex_conjugate_product_correct (w : ℂ) (h : abs w = 15) : complex_conjugate_product w h = 225 :=
by
  -- Proof omitted
  sorry

end complex_conjugate_product_correct_l702_702287


namespace complex_square_eq_l702_702170

-- Define the problem conditions and proof statement
theorem complex_square_eq (a b : ℝ) (h : (a + b * Complex.i)^2 = 3 + 4 * Complex.i) : a * b = 2 :=
  sorry -- Proof will be provided here

end complex_square_eq_l702_702170


namespace different_set_l702_702067

-- Define the sets
def set1 : Set ℝ := { x | x = 1 }
def set2 : Set ℝ := { y | (y - 1)^2 = 0 }
def set3 : Set (Set ℝ) := {1}
def set4 : Set ℝ := { 1 }

-- Main theorem statement to prove Set ③ is different from the other three sets
theorem different_set :
  set3 ≠ set1 ∧ set3 ≠ set2 ∧ set3 ≠ set4 :=
by
  sorry

end different_set_l702_702067


namespace smallest_positive_x_floor_value_l702_702276

def g (x : ℝ) : ℝ := 2 * Real.sin x - Real.cos x + 5 * Real.cot x

theorem smallest_positive_x_floor_value :
  ∃ s > 0, g s = 0 ∧ ⌊s⌋ = 4 :=
sorry

end smallest_positive_x_floor_value_l702_702276


namespace cone_vertex_angle_l702_702426

theorem cone_vertex_angle (θ : ℝ) :
  let A_axial := (1 / 2) * base * height,
      A_max := (1 / 2) * base * height * sin θ in
  (2 * A_axial = A_max) → θ = π / 6 →
  cone_vertex_angle = 2 * θ :=
sorry

end cone_vertex_angle_l702_702426


namespace round_4_65_to_nearest_tenth_l702_702631

theorem round_4_65_to_nearest_tenth : 
  (round (nearest (10^(-1))) 4.65) = 4.7 := 
by 
  sorry

end round_4_65_to_nearest_tenth_l702_702631


namespace Nellie_legos_l702_702301

def initial_legos : ℕ := 380
def lost_legos : ℕ := 57
def given_legos : ℕ := 24

def remaining_legos : ℕ := initial_legos - lost_legos - given_legos

theorem Nellie_legos : remaining_legos = 299 := by
  sorry

end Nellie_legos_l702_702301


namespace domain_of_f_l702_702783

def f (x : ℝ) : ℝ := log (x + 1) / sqrt (2 * x - 3)

theorem domain_of_f :
  { x : ℝ | x + 1 > 0 ∧ 2 * x - 3 > 0 } = { x : ℝ | x > 3 / 2 } :=
by
  -- The proof goes here
  sorry

end domain_of_f_l702_702783


namespace tan_half_sum_eq_third_l702_702916

theorem tan_half_sum_eq_third
  (x y : ℝ)
  (h1 : Real.cos x + Real.cos y = 3/5)
  (h2 : Real.sin x + Real.sin y = 1/5) :
  Real.tan ((x + y) / 2) = 1/3 :=
by sorry

end tan_half_sum_eq_third_l702_702916


namespace sqrt_144_div_6_l702_702698

theorem sqrt_144_div_6 : real.sqrt 144 / 6 = 2 := sorry

end sqrt_144_div_6_l702_702698


namespace part_a_part_b_l702_702201

-- Define the set based on given conditions
def M (a b : ℕ) : set ℕ := 
  {z | ∃ x y : ℕ, z = a * x + b * y}

-- Define the greatest integer that does not belong to the set M
def frobenius_number (a b : ℕ) : ℕ := a * b - a - b

-- Main hypotheses
variables {a b : ℕ}

-- Given: a and b are coprime natural numbers
-- Goal (a): Prove that frobenius_number a b is the greatest integer that does not belong to M a b
theorem part_a (h_coprime : Nat.coprime a b) : 
  ∀ z ∉ M a b, z ≤ frobenius_number a b := 
sorry

-- Goal (b): Prove that for any integer n, one of n or frobenius_number a b - n belongs to M a b and the other does not
theorem part_b (h_coprime : Nat.coprime a b) (n : ℤ) : 
  (n ∈ M a b ∧ frobenius_number a b - n ∉ M a b) ∨ (n ∉ M a b ∧ frobenius_number a b - n ∈ M a b) := 
sorry

end part_a_part_b_l702_702201


namespace tangent_line_at_1_ln_le_x_minus_1_l702_702524

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 1

theorem tangent_line_at_1 :
  let A := (1, f 1) in
  ∃ m b : ℝ, m = 0 ∧ b = f 1 ∧ (∀ x : ℝ, f x = m * (x - 1) + b → y = 0) :=
sorry

theorem ln_le_x_minus_1 (x : ℝ) (h : 0 < x) : Real.log x ≤ x - 1 := 
by
  sorry

end tangent_line_at_1_ln_le_x_minus_1_l702_702524


namespace fraction_exponent_rule_l702_702447

theorem fraction_exponent_rule (a b n : ℕ) (hab : a = 5) (hbb : b = 7) (hnb : n = 6) :
  (a / b : ℚ)^n = 15625 / 117649 :=
by
  rw [hab, hbb, hnb]
  norm_num
  sorry

end fraction_exponent_rule_l702_702447


namespace find_valera_car_l702_702675

def num_cars := 15
def first_pass_time := 28
def total_pass_time := 60
def sasha_car := 3

theorem find_valera_car : 
  ∃ (valera_car : ℕ), 
    (num_cars = 15) ∧ 
    (first_pass_time = 28) ∧ 
    (total_pass_time = 60) ∧ 
    (sasha_car = 3) ∧ 
    valera_car = 12 :=
by
  use 12
  simp
  split
  { refl }
  split
  { refl }
  split
  { refl }
  split
  { refl }
  { refl }

end find_valera_car_l702_702675


namespace range_of_m_l702_702337

theorem range_of_m (f : ℝ → ℝ) (g : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = x ^ (log 3 / log (sqrt 3))) →
  g = λ x => f x + 1 →
  Set.range (g ∘ (fun x => Icc m 2)) = Set.Icc 1 5 →
  -2 ≤ m ∧ m ≤ 0 :=
by
  intro pow_func_def g_def range_cond
  sorry

end range_of_m_l702_702337


namespace fib_mod_13_multiples_count_l702_702463

noncomputable def fib_mod (n : ℕ) : ℕ := Nat.fib n % 13

def is_multiple_of_13 (n : ℕ) : Prop := fib_mod n = 0

def count_fib_multiples_of_13 (upper_bound : ℕ) : ℕ :=
  Nat.length (List.filter is_multiple_of_13 (List.range (upper_bound + 1)))

theorem fib_mod_13_multiples_count :
  count_fib_multiples_of_13 100 = 15 :=
sorry

end fib_mod_13_multiples_count_l702_702463


namespace hair_cut_first_day_l702_702796

theorem hair_cut_first_day 
  (total_hair_cut : ℝ) 
  (hair_cut_second_day : ℝ) 
  (h_total : total_hair_cut = 0.875) 
  (h_second : hair_cut_second_day = 0.5) : 
  total_hair_cut - hair_cut_second_day = 0.375 := 
  by
  simp [h_total, h_second]
  sorry

end hair_cut_first_day_l702_702796


namespace nat_numbers_square_minus_one_power_of_prime_l702_702133

def is_power_of_prime (x : ℕ) : Prop :=
  ∃ (p : ℕ), Nat.Prime p ∧ ∃ (k : ℕ), x = p ^ k

theorem nat_numbers_square_minus_one_power_of_prime (n : ℕ) (hn : 1 ≤ n) :
  is_power_of_prime (n ^ 2 - 1) ↔ (n = 2 ∨ n = 3) := by
  sorry

end nat_numbers_square_minus_one_power_of_prime_l702_702133


namespace sum_of_digits_largest_n_is_13_l702_702602

-- Define the necessary conditions
def single_digit_primes : List ℕ := [2, 3, 5, 7]

def is_valid_prime_combination (d e : ℕ) : Prop := 
  d ∈ single_digit_primes ∧ 
  e ∈ single_digit_primes ∧ 
  d < e ∧ 
  Prime (d^2 + e^2)

def product_three_primes (d e : ℕ) : ℕ := d * e * (d^2 + e^2)

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

noncomputable def largest_n : ℕ := 
  (single_digit_primes.product single_digit_primes).filter (λ p, is_valid_prime_combination p.1 p.2)
  |>.map (λ p, product_three_primes p.1 p.2)
  |>.maximum.get_or_else 0

theorem sum_of_digits_largest_n_is_13 : sum_of_digits largest_n = 13 := by
  sorry

end sum_of_digits_largest_n_is_13_l702_702602


namespace beijing_time_conversion_l702_702072

-- Define the conversion conditions
def new_clock_hours_in_day : Nat := 10
def new_clock_minutes_per_hour : Nat := 100
def new_clock_time_at_5_beijing_time : Nat := 12 * 60  -- converting 12 noon to minutes


-- Define the problem to prove the corresponding Beijing time 
theorem beijing_time_conversion :
  new_clock_minutes_per_hour * 5 = 500 → 
  new_clock_time_at_5_beijing_time = 720 →
  (720 + 175 * 1.44) = 4 * 60 + 12 :=
by
  intros h1 h2
  sorry

end beijing_time_conversion_l702_702072


namespace factor_expr_l702_702454

variable (x : ℝ)

def expr : ℝ := (20 * x ^ 3 + 100 * x - 10) - (-5 * x ^ 3 + 5 * x - 10)

theorem factor_expr :
  expr x = 5 * x * (5 * x ^ 2 + 19) :=
by
  sorry

end factor_expr_l702_702454


namespace right_triangle_integer_segments_l702_702630

noncomputable def hypotenuse_length (a b : ℕ) : ℝ :=
  Real.sqrt (a^2 + b^2)

noncomputable def count_integer_segments (hypotenuse_altitude : ℝ) (leg1 leg2 : ℕ) : ℕ :=
  let max_len := max leg1 leg2
  let min_len := min leg1 (Real.toInt ⟨floor hypotenuse_altitude, sorry⟩)
  max_len - min_len + 1 + max_len - min_len

theorem right_triangle_integer_segments :
  ∃ (n : ℕ), n = 16 ∧ n = count_integer_segments (600 / hypotenuse_length 24 25) 24 25 :=
begin
  use 16,
  split,
  { refl },
  { sorry }
end

end right_triangle_integer_segments_l702_702630


namespace angle_BDC_proof_l702_702877

noncomputable def angle_sum_triangle (angle_A angle_B angle_C : ℝ) : Prop :=
  angle_A + angle_B + angle_C = 180

-- Given conditions
def angle_A : ℝ := 70
def angle_E : ℝ := 50
def angle_C : ℝ := 40

-- The problem of proving that angle_BDC = 20 degrees
theorem angle_BDC_proof (A E C BDC : ℝ) 
  (hA : A = angle_A)
  (hE : E = angle_E)
  (hC : C = angle_C) :
  BDC = 20 :=
  sorry

end angle_BDC_proof_l702_702877


namespace real_roots_for_all_k_S_equals_2_implies_k_2_l702_702805

-- Definitions from conditions
def quadratic_eq (k : ℝ) (x : ℝ) := (k-1) * x^2 + 2 * k * x + 2

def S (x1 x2 : ℝ) := (x2 / x1) + (x1 / x2) + x1 + x2

-- Theorem statements
theorem real_roots_for_all_k (k : ℝ) : ∃ x1 x2 : ℝ, quadratic_eq k x1 = 0 ∧ quadratic_eq k x2 = 0 :=
by sorry

theorem S_equals_2_implies_k_2 (k x1 x2 : ℝ) (h1 : quadratic_eq k x1 = 0) (h2 : quadratic_eq k x2 = 0) 
(h3 : S x1 x2 = 2) : k = 2 :=
by sorry

end real_roots_for_all_k_S_equals_2_implies_k_2_l702_702805


namespace horizontal_asymptote_crossing_l702_702807

theorem horizontal_asymptote_crossing (x : ℝ) : 
  (g x = 3) → x = 15 / 7 :=
by 
  let g := λ x : ℝ, (3 * x ^ 2 - 8 * x - 9) / (x ^ 2 - 5 * x + 2)
  sorry

end horizontal_asymptote_crossing_l702_702807


namespace required_weekly_hours_approx_27_l702_702537

noncomputable def planned_hours_per_week : ℝ := 25
noncomputable def planned_weeks : ℝ := 15
noncomputable def total_amount : ℝ := 4500
noncomputable def sick_weeks : ℝ := 3
noncomputable def increased_wage_weeks : ℝ := 5
noncomputable def wage_increase_factor : ℝ := 1.5 -- 50%

-- Normal hourly wage
noncomputable def normal_hourly_wage : ℝ := total_amount / (planned_hours_per_week * planned_weeks)

-- Increased hourly wage
noncomputable def increased_hourly_wage : ℝ := normal_hourly_wage * wage_increase_factor

-- Earnings in the last 5 weeks at increased wage
noncomputable def earnings_in_last_5_weeks : ℝ := increased_hourly_wage * planned_hours_per_week * increased_wage_weeks

-- Amount needed before the wage increase
noncomputable def amount_needed_before_wage_increase : ℝ := total_amount - earnings_in_last_5_weeks

-- We have 7 weeks before the wage increase
noncomputable def weeks_before_increase : ℝ := planned_weeks - sick_weeks - increased_wage_weeks

-- New required weekly hours before wage increase
noncomputable def required_weekly_hours : ℝ := amount_needed_before_wage_increase / (normal_hourly_wage * weeks_before_increase)

theorem required_weekly_hours_approx_27 :
  abs (required_weekly_hours - 27) < 1 :=
sorry

end required_weekly_hours_approx_27_l702_702537


namespace students_meet_time_l702_702366

theorem students_meet_time :
  ∀ (distance rate1 rate2 : ℝ),
    distance = 350 ∧ rate1 = 1.6 ∧ rate2 = 1.9 →
    distance / (rate1 + rate2) = 100 := by
  sorry

end students_meet_time_l702_702366


namespace compute_expression_at_x_eq_3_l702_702083

theorem compute_expression_at_x_eq_3 :
  (let x := 3 in (x^8 + 18 * x^4 + 81) / (x^4 + 9)) = 90 :=
by
  sorry

end compute_expression_at_x_eq_3_l702_702083


namespace vendor_apples_sold_l702_702744

theorem vendor_apples_sold (x : ℝ) (h : 0.15 * (1 - x / 100) + 0.50 * (1 - x / 100) * 0.85 = 0.23) : x = 60 :=
sorry

end vendor_apples_sold_l702_702744


namespace perimeter_KLH_eq_diameter_ABC_l702_702258

open Real

variables {A B C K L H : Point} -- Declaring the points of the triangles.
variables {circumcircle : Circle A B C} -- Declaring the circumcircle of triangle ABC.
variables {O : Point} -- Declaring the circumcenter of triangle ABC.
variables {perimeter_KLH diameter_circumcircle_ABC : ℝ}

-- Given conditions
axiom angle_ACB_45_deg : ∠ACB = 45
axiom center_O : circumcenter circumcircle = O
axiom orthocenter_H : orthocenter A B C = H
axiom perp_line_O : ∃ line, line.contains O ∧ line.perp CO ∧ line.intersects AC = K ∧ line.intersects BC = L

-- Problem statement
theorem perimeter_KLH_eq_diameter_ABC : perimeter_KLH = 2 * (diameter circumcircle) := sorry

end perimeter_KLH_eq_diameter_ABC_l702_702258


namespace part1_part2_l702_702522

noncomputable def f (x m : ℝ) := |x + 1| + |m - x|

theorem part1 (x : ℝ) : (f x 3) ≥ 6 ↔ (x ≤ -2 ∨ x ≥ 4) :=
by sorry

theorem part2 (m : ℝ) : (∀ x, f x m ≥ 8) ↔ (m ≥ 7 ∨ m ≤ -9) :=
by sorry

end part1_part2_l702_702522


namespace probability_at_least_one_3_is_fraction_l702_702041

-- Definitions
def is_fair_eight_sided_dice (d: ℕ) : Prop := d ∈ {1, 2, 3, 4, 5, 6, 7, 8}

noncomputable def probability_of_sum_condition_met 
(X1 X2 X3 X4: ℕ) (condition: X1 + X2 + X3 = X4) : ℝ := 
  if is_fair_eight_sided_dice X1 ∧ 
     is_fair_eight_sided_dice X2 ∧ 
     is_fair_eight_sided_dice X3 ∧ 
     is_fair_eight_sided_dice X4 ∧ 
     condition
  then 1 else 0

noncomputable def probability_of_at_least_one_3 (X1 X2 X3 X4: ℕ) (condition: X1 + X2 + X3 = X4) : ℝ := 
  if 3 ∈ {X1, X2, X3, X4} then 1 else 0

-- Main statement to prove
theorem probability_at_least_one_3_is_fraction (X1 X2 X3 X4: ℕ) 
(condition: X1 + X2 + X3 = X4) : 
  (probability_of_at_least_one_3 X1 X2 X3 X4 condition) / 
  (probability_of_sum_condition_met X1 X2 X3 X4 condition) = (detail of \frac{F}{T}) :=
sorry

end probability_at_least_one_3_is_fraction_l702_702041


namespace probability_f_x_le_3_l702_702526

noncomputable def f (x : ℝ) : ℝ := (2^x) + 2

theorem probability_f_x_le_3 :
  (∀ x ∈ (-3:ℝ)..3, f x ≤ 3) → ((1 / 2) : ℝ) :=
begin
  let bound : ℝ := (3 - (-3)),
  let favorable_length : ℝ := (0 - (-3)),
  have prob := favorable_length / bound,
  calc prob : ℝ = 1 / 2,
  sorry
end

end probability_f_x_le_3_l702_702526


namespace distinct_real_roots_implies_derivative_positive_l702_702838

variable (a c : ℝ) (f : ℝ → ℝ) (x1 x2 : ℝ)

-- Define the function
def f (x : ℝ) : ℝ := x^2 - (a - 2) * x - a * log x

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := (2 * x - a) * (x + 1) / x

-- The main theorem statement
theorem distinct_real_roots_implies_derivative_positive 
  (h_distinct : x1 ≠ x2)
  (h_domain : 0 < x1)
  (h_domain2 : 0 < x2)
  (h_order : x1 < x2)
  (h_eq_cx1 : f x1 = c)
  (h_eq_cx2 : f x2 = c) :
  f' ((x1 + x2) / 2) > 0 := sorry

end distinct_real_roots_implies_derivative_positive_l702_702838


namespace valid_course_combinations_l702_702419

def total_courses : ℕ := 7
def english_course : ℕ := 1
def math_courses : ℕ := 3
def program_courses : ℕ := 4

def valid_combinations : ℕ :=
  nat.choose (total_courses - english_course) (program_courses - english_course) -
  nat.choose (total_courses - english_course - math_courses) (program_courses - english_course)

theorem valid_course_combinations : valid_combinations = 16 :=
by sorry

end valid_course_combinations_l702_702419


namespace probability_divisibility_l702_702925

open BigOperators

noncomputable def S := {d ∈ Finset.range (36^7 + 1) | (36^7) % d = 0}

theorem probability_divisibility (a1 a2 a3 : ℕ) (h1 : a1 ∈ S) (h2 : a2 ∈ S) (h3 : a3 ∈ S) :
  let P := (3136 : ℤ) / ((15 : ℕ)^3 * (15: ℕ)^3)
  ∑ x in S, ∑ y in S, ∑ z in S, (x ∣ y ∧ y ∣ z) = P * S.card^3 :=
sorry

end probability_divisibility_l702_702925


namespace vector_A0_A2_coordinates_periodic_f_l702_702393

/-- Defining point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defining symmetry of a point with respect to another point -/
def symmetric (A P : Point) : Point :=
  ⟨2 * P.x - A.x, 2 * P.y - A.y⟩

/-- Defining the points -/
def P (n : ℕ) : Point :=
  ⟨n, 2^n⟩

/-- Defining vector from A0 to A2 given conditions -/
def vector_A0_A2 (A0 : Point) : Point :=
  let A1 := symmetric A0 (P 1)
  let A2 := symmetric A1 (P 2)
  ⟨A2.x - A0.x, A2.y - A0.y⟩

/-- Defining the periodic function f(x) for x in (1, 4] -/
noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Ioo 0 3 then log x
  else log (x - 1)

-- Skipping the proofs with sorry
theorem vector_A0_A2_coordinates (A0 : Point) : vector_A0_A2 A0 = ⟨2, 4⟩ := sorry

theorem periodic_f : (∀ x ∈ Ioo 0 3, f x = log x) ∧ (∀ x ∈ Ioo 1 4, f x = log (x - 1)) := sorry

end vector_A0_A2_coordinates_periodic_f_l702_702393


namespace dye_jobs_scheduled_l702_702955

noncomputable def revenue_from_haircuts (n : ℕ) : ℕ := n * 30
noncomputable def revenue_from_perms (n : ℕ) : ℕ := n * 40
noncomputable def revenue_from_dye_jobs (n : ℕ) : ℕ := n * (60 - 10)
noncomputable def total_revenue (haircuts perms dye_jobs : ℕ) (tips : ℕ) : ℕ :=
  revenue_from_haircuts haircuts + revenue_from_perms perms + revenue_from_dye_jobs dye_jobs + tips

theorem dye_jobs_scheduled : 
  (total_revenue 4 1 dye_jobs 50 = 310) → (dye_jobs = 2) := 
by
  sorry

end dye_jobs_scheduled_l702_702955


namespace center_coordinates_sum_l702_702326

-- Define the quadrant and points
def is_in_first_quadrant (p : ℝ × ℝ) : Prop := p.1 ≥ 0 ∧ p.2 ≥ 0
def points := [(4, 0), (6, 0), (9, 0), (15, 0)]

-- Define the lines through the points
def line_DA (x : ℝ) (n : ℝ) : ℝ := n * (x - 4)
def line_CB (x : ℝ) (n : ℝ) : ℝ := n * (x - 6)
def line_AB (x : ℝ) (n : ℝ) : ℝ := - (1 / n) * (x - 9)
def line_DC (x : ℝ) (n : ℝ) : ℝ := - (1 / n) * (x - 15)

-- Define the condition for the points being on the lines
def point_on_line (line : ℝ → ℝ) (p : ℝ × ℝ) : Prop := p.2 = line p.1

-- Define the condition that the square lies in the first quadrant
def square_in_first_quadrant :=
  is_in_first_quadrant (4, 0) ∧
  is_in_first_quadrant (6, 0) ∧
  is_in_first_quadrant (9, 0) ∧
  is_in_first_quadrant (15, 0)

-- State the theorem
theorem center_coordinates_sum :
  square_in_first_quadrant →
  point_on_line (line_DA 4) (4, 0) →
  point_on_line (line_CB 6) (6, 0) →
  point_on_line (line_AB 9) (9, 0) →
  point_on_line (line_DC 15) (15, 0) →
  let n := 3 in
  let midpoint_1 := (10, 0) in
  let midpoint_2 := (12, 0) in
  let center_x := 7.8 in
  let center_y := -6.6 in
  center_x + center_y = 1.2 :=
sorry


end center_coordinates_sum_l702_702326


namespace economist_winning_strategy_l702_702701

-- Conditions setup
variables {n a b x1 x2 y1 y2 : ℕ}

-- Definitions according to the conditions
def valid_initial_division (n a b : ℕ) : Prop :=
  n > 4 ∧ n % 2 = 1 ∧ 2 ≤ a ∧ 2 ≤ b ∧ a + b = n ∧ a < b

def valid_further_division (a b x1 x2 y1 y2 : ℕ) : Prop :=
  x1 + x2 = a ∧ x1 ≥ 1 ∧ x2 ≥ 1 ∧ y1 + y2 = b ∧ y1 ≥ 1 ∧ y2 ≥ 1 ∧ x1 ≤ x2 ∧ y1 ≤ y2

-- Methods defined: Assumptions about which parts the economist takes
def method_1 (x1 x2 y1 y2 : ℕ) : ℕ :=
  max x2 y2 + min x1 y1

def method_2 (x1 x2 y1 y2 : ℕ) : ℕ :=
  (x1 + y1) / 2 + (x2 + y2) / 2

def method_3 (x1 x2 y1 y2 : ℕ) : ℕ :=
  max (method_1 x1 x2 y1 y2 - 1) (method_2 x1 x2 y1 y2 - 1) + 1

-- The statement to prove that the economist would choose method 1
theorem economist_winning_strategy :
  ∀ n a b x1 x2 y1 y2,
    valid_initial_division n a b →
    valid_further_division a b x1 x2 y1 y2 →
    n > 4 → n % 2 = 1 →
    (method_1 x1 x2 y1 y2) > (method_2 x1 x2 y1 y2) →
    (method_1 x1 x2 y1 y2) > (method_3 x1 x2 y1 y2) →
    method_1 x1 x2 y1 y2 = max (method_1 x1 x2 y1 y2) (method_2 x1 x2 y1 y2) :=
by
  -- Placeholder for the actual proof
  sorry

end economist_winning_strategy_l702_702701


namespace calculate_factorial_expression_l702_702077

theorem calculate_factorial_expression : (13.factorial - 12.factorial + 11.factorial) / 10.factorial = 1595 := by
  sorry -- The proof will be constructed here

end calculate_factorial_expression_l702_702077


namespace find_d_l702_702074

theorem find_d (a b c d x : ℝ)
  (h1 : ∀ x, 2 ≤ a * (Real.cos (b * x + c)) + d ∧ a * (Real.cos (b * x + c)) + d ≤ 4)
  (h2 : Real.cos (b * 0 + c) = 1) :
  d = 3 :=
sorry

end find_d_l702_702074


namespace min_moves_to_equalize_l702_702320

-- Definition of the arrangement of coins in the boxes
def initial_coins : List ℕ := [9, 13, 10, 20, 5, 17, 18]

-- We need to prove a specific goal about the number of moves
theorem min_moves_to_equalize :
  ∃ m : ℕ, min_moves_to_equalize_coins initial_coins 7 m ∧ m = 22 :=
sorry

end min_moves_to_equalize_l702_702320


namespace max_stamps_l702_702227

theorem max_stamps (price_lt_100 : ℕ) (price_gt_100 : ℕ) 
  (price_amount : ℕ) (h1 : price_lt_100 = 45) (h2 : price_gt_100 = 40)
  (h3 : price_amount = 5000) : 
  max_stamps_50_dollars : ℕ :=
by {
  have n1_ineq := (5000 : ℕ) / 45,
  have n2_ineq := (5000 : ℕ) / 40,
  exact 125
}

end max_stamps_l702_702227


namespace number_of_carbon_atoms_l702_702724

noncomputable def molecular_weight := 78
noncomputable def hydrogen_weight_per_atom := 1
noncomputable def carbon_weight_per_atom := 12
noncomputable def num_hydrogen_atoms := 6

def weight_hydrogen_atoms := num_hydrogen_atoms * hydrogen_weight_per_atom
def weight_carbon_atoms := molecular_weight - weight_hydrogen_atoms
def num_carbon_atoms := weight_carbon_atoms / carbon_weight_per_atom

theorem number_of_carbon_atoms :
  num_carbon_atoms = 6 := 
sorry

end number_of_carbon_atoms_l702_702724


namespace intersection_complement_l702_702931

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3, 4, 5}

-- Define set M
def M : Set ℕ := {0, 3, 5}

-- Define set N
def N : Set ℕ := {1, 4, 5}

-- Define the complement of N in U
def complement_U_N : Set ℕ := U \ N

-- The main theorem statement
theorem intersection_complement : M ∩ complement_U_N = {0, 3} :=
by
  -- The proof would go here
  sorry

end intersection_complement_l702_702931


namespace volume_of_square_truncated_pyramid_l702_702790

theorem volume_of_square_truncated_pyramid (H B b : ℝ) (h_eq : H = 6) (B_eq : B = 16) (b_eq : b = 4) :
  (H / 3) * (B + b + real.sqrt (B * b)) = 56 :=
by {
  sorry
}

end volume_of_square_truncated_pyramid_l702_702790


namespace x_less_than_y_by_35_percent_l702_702550

noncomputable def percentage_difference (x y : ℝ) : ℝ :=
  ((y / x) - 1) * 100

theorem x_less_than_y_by_35_percent (x y : ℝ) (h : y = 1.5384615384615385 * x) :
  percentage_difference x y = 53.846153846153854 :=
by
  sorry

end x_less_than_y_by_35_percent_l702_702550


namespace equivalence_of_conditions_l702_702948

-- Definitions based on conditions in the problem
def gcd (a b : ℕ) : ℕ := Nat.gcd a b
def is_prime (p : ℕ) : Prop := Nat.Prime p
def v_p (p n : ℕ) : ℕ := Nat.findGreatestDivisible p n

-- The actual theorem statement
theorem equivalence_of_conditions (n : ℕ) (h1 : n > 1) :
  (∀ (m : ℕ), (h2 : m < n) → gcd n (Nat.div (n - m) (gcd n m)) = 1) ↔
  (∃ (p : ℕ), is_prime p ∧ (∀ (m : ℕ), (h2 : m < n) → (Nat.div (v_p p m) m) < (Nat.div (v_p p n) n))) := 
by 
  sorry

end equivalence_of_conditions_l702_702948


namespace last_digit_sum_of_squares_to_2012_l702_702374

theorem last_digit_sum_of_squares_to_2012 : 
  (∑ i in Finset.range 2013, (i * i) % 10) % 10 = 0 := 
by
  sorry

end last_digit_sum_of_squares_to_2012_l702_702374


namespace ratio_length_to_breadth_l702_702988

theorem ratio_length_to_breadth (l b : ℕ) (h1 : b = 14) (h2 : l * b = 588) : l / b = 3 :=
by
  sorry

end ratio_length_to_breadth_l702_702988


namespace measure_of_angle_C_perimeter_of_triangle_ABC_l702_702262

-- Define the necessary context and conditions
variables (A B C : ℝ) (a b c : ℝ)
variables (h_cond : a * cos B + b * cos A = 2 * c * cos C)
variables (h_area : (1 / 2) * a * b * sin C = 2 * sqrt 3)
variables (h_c : c = 2 * sqrt 3)

-- Part (1): Measure of angle C
theorem measure_of_angle_C :
  C = π / 3 :=
by sorry

-- Part (2): Perimeter of triangle ABC
theorem perimeter_of_triangle_ABC :
  a + b + c = 6 + 2 * sqrt 3 :=
by sorry

end measure_of_angle_C_perimeter_of_triangle_ABC_l702_702262


namespace circle_tangent_area_l702_702081

noncomputable def circle_tangent_area_problem 
  (radiusA radiusB radiusC : ℝ) (tangent_midpoint : Bool) : ℝ :=
  if (radiusA = 1 ∧ radiusB = 1 ∧ radiusC = 2 ∧ tangent_midpoint) then 
    (4 * Real.pi) - (2 * Real.pi) 
  else 0

theorem circle_tangent_area (radiusA radiusB radiusC : ℝ) (tangent_midpoint : Bool) :
  radiusA = 1 → radiusB = 1 → radiusC = 2 → tangent_midpoint = true → 
  circle_tangent_area_problem radiusA radiusB radiusC tangent_midpoint = 2 * Real.pi :=
by
  intros
  simp [circle_tangent_area_problem]
  split_ifs
  · sorry
  · sorry

end circle_tangent_area_l702_702081


namespace range_of_F_l702_702515

noncomputable def f_M (M : set ℝ) (x : ℝ) : ℝ :=
  if x ∈ M then 1 else 0

noncomputable def F (A B : set ℝ) (x : ℝ) : ℝ :=
  (f_M (A ∪ B) x + 1) / (f_M A x + f_M B x + 2)

theorem range_of_F (A B : set ℝ) (hA : A.nonempty) (hB : B.nonempty) (hAB : A ∩ B = ∅) :
  set.range (F A B) = {1 / 2, 2 / 3} :=
sorry

end range_of_F_l702_702515


namespace count_odd_numbers_with_distinct_digits_l702_702755

def number_of_odd_numbers_with_distinct_digits (n : ℕ) (h₁ : n ≤ 10000) (h₂ : n % 2 = 1) (h₃ : Function.Injective (Nat.digits 10 n)) : Prop :=
  ∀ m : ℕ, m = 2605

theorem count_odd_numbers_with_distinct_digits : number_of_odd_numbers_with_distinct_digits 10000 :=
by
  intro m
  sorry

end count_odd_numbers_with_distinct_digits_l702_702755


namespace question_1_question_2_question_3_l702_702696

variables (p q : Prop)

-- Conditions given in the problem
axiom h_p : p
axiom h_q : q

-- Proof of the first question: p or q
theorem question_1 : p ∨ q := by
  exact (Or.inl h_p) -- Since p is true, p ∨ q is trivially true

-- Proof of the second question: p and q
theorem question_2 : p ∧ q := by
  apply And.intro h_p h_q -- Both p and q are true, so p ∧ q is true

-- Proof of the third question: not p
theorem question_3 : ¬p = False := by
  have : p := h_p -- Given that p is true 
  exact (False.intro this) -- ¬p would lead to contradiction, so it is false

#sorries replaced with the corresponding necessary proof to avoid

end question_1_question_2_question_3_l702_702696


namespace alex_final_bill_l702_702030

def original_bill : ℝ := 500
def first_late_charge (bill : ℝ) : ℝ := bill * 1.02
def final_bill (bill : ℝ) : ℝ := first_late_charge bill * 1.03

theorem alex_final_bill : final_bill original_bill = 525.30 :=
by sorry

end alex_final_bill_l702_702030


namespace maximize_side_area_of_cylinder_l702_702038

noncomputable def radius_of_cylinder (x : ℝ) : ℝ :=
  (6 - x) / 3

noncomputable def side_area_of_cylinder (x : ℝ) : ℝ :=
  2 * Real.pi * (radius_of_cylinder x) * x

theorem maximize_side_area_of_cylinder :
  ∃ x : ℝ, (0 < x ∧ x < 6) ∧ (∀ y : ℝ, (0 < y ∧ y < 6) → (side_area_of_cylinder y ≤ side_area_of_cylinder x)) ∧ x = 3 :=
by
  sorry

end maximize_side_area_of_cylinder_l702_702038


namespace school_desk_purchase_solution_l702_702720

noncomputable def school_desk_purchase_problem : Prop :=
  ∃ (a b : ℕ) (x y : ℕ),
  (200 = x + y) ∧
  (a = b - 40) ∧
  (3 * a + 5 * b = 1640) ∧
  (180 * x + 220 * y ≤ 40880) ∧
  (x ≤ 2 * (200 - x) / 3) ∧
  (a = 180 ∧ b = 220) ∧
  (x = 78 ∨ x = 79 ∨ x = 80) ∧
  (min_cost := min [(180 * 78 + 220 * (200 - 78)),
                    (180 * 79 + 220 * (200 - 79)),
                    (180 * 80 + 220 * (200 - 80))] 40800)


theorem school_desk_purchase_solution : school_desk_purchase_problem :=
begin
  sorry
end

end school_desk_purchase_solution_l702_702720


namespace goldfish_cost_discrete_points_l702_702535

def goldfish_cost (n : ℕ) : ℝ :=
  0.25 * n + 5

theorem goldfish_cost_discrete_points :
  ∀ n : ℕ, 5 ≤ n ∧ n ≤ 20 → ∃ k : ℕ, goldfish_cost n = goldfish_cost k ∧ 5 ≤ k ∧ k ≤ 20 :=
by sorry

end goldfish_cost_discrete_points_l702_702535


namespace no_tangential_points_of_odd_polynomial_l702_702270

noncomputable def P (x : ℝ) : ℝ := sorry -- Define a nonzero polynomial P with appropriate properties as per conditions

theorem no_tangential_points_of_odd_polynomial (n : ℕ) (h1 : n > 1) (h2 : ∀ i, (0:ℝ) ≤ P(i))
  (h3 : ∀ x, P(-x) = -P(x)) -- P(x) is odd
  (h4 : ∃ (A : fin n → ℝ × ℝ), function.injective A ∧
    (∀ i, let Ai := A ⟨i.1, nat.lt_of_lt_pred i.2⟩ in 
    let Aj := A ⟨(i.1 + 1) % n, sorry⟩ in 
    (Ai.2 = (deriv P Ai.1) * (Aj.1 - Ai.1) + P(Ai.1)) ∧ -- tangent to G at Ai passes through Aj
    (∃ (Ai Aj : ℝ × ℝ), Ai ≠ Aj ∧ Ai ∈ A ∧ Aj ∈ A ∧ ((deriv P Ai.1) = (P Aj.1 - P Ai.1) / (Aj.1 - Ai.1))))): False :=
begin
  sorry,
end

end no_tangential_points_of_odd_polynomial_l702_702270


namespace number_of_rods_in_one_mile_l702_702167

theorem number_of_rods_in_one_mile :
  (1 : ℤ) * 6 * 60 = 360 :=
by
  sorry

end number_of_rods_in_one_mile_l702_702167


namespace no_two_triangles_in_sequence_are_similar_l702_702559

structure Triangle :=
  (alpha beta gamma : ℝ)
  (scalene : alpha < beta ∧ beta < gamma)

def anglesOfNewTriangle (t : Triangle) : Triangle :=
  { alpha := (t.beta + t.gamma) / 2,
    beta := (t.alpha + t.gamma) / 2,
    gamma := (t.alpha + t.beta) / 2,
    scalene := sorry } -- proof of scalene property for new triangle not needed for the statement

def nthTriangle (t : Triangle) (n : ℕ) : Triangle :=
  nat.iterate anglesOfNewTriangle n t

theorem no_two_triangles_in_sequence_are_similar 
  (t : Triangle) (h : t.scalene) :
  ∀ (m n : ℕ), m ≠ n → nthTriangle t m ≠ nthTriangle t n :=
sorry

end no_two_triangles_in_sequence_are_similar_l702_702559


namespace projections_perpendicular_projections_perpendicular_converse_l702_702627

noncomputable def orthogonal_projection_plane (P : Type*) [inner_product_space ℝ P] (U : submodule ℝ P) [finite_dimensional ℝ U] :=
  orthogonal_projection (↑U)

variable {P : Type*} [inner_product_space ℝ P] [finite_dimensional ℝ P]
variables (a b : P) (γ : submodule ℝ P)

-- Conditions:
-- a is parallel to the plane γ
-- b is perpendicular to a
-- b is not perpendicular to the plane γ

variables [fact (finite_dimensional.finrank ℝ γ = 2)]
-- Assuming γ is a 2-dimensional subspace (a plane)
theorem projections_perpendicular (ha : a ∈ γᗮ) (hb : ∀ x ∈ γ, ⟪b, x⟫ = 0 → ⟪b, a⟫ = 0) :
  let a' := orthogonal_projection γ a
      b' := orthogonal_projection γ b
  in ⟪a', b'⟫ = 0 :=
sorry

-- Converse Statement:
theorem projections_perpendicular_converse (ha : a ∈ γᗮ) :
  let a' := orthogonal_projection γ a
      b' := orthogonal_projection γ b
  in ⟪a', b'⟫ = 0 → ∀ x ∈ γ, ⟪b, x⟫ = 0 → ⟪b, a⟫ = 0 :=
sorry

end projections_perpendicular_projections_perpendicular_converse_l702_702627


namespace sin2x_cos2x_value_l702_702917

theorem sin2x_cos2x_value (x y : ℝ) (h₁ : sin x / sin y = 4) (h₂ : cos x / cos y = 1 / 3) :
  sin (2 * x) / sin (2 * y) + cos (2 * x) / cos (2 * y) = 61 / 129 :=
by
  sorry

end sin2x_cos2x_value_l702_702917


namespace regular_octagon_area_l702_702315

theorem regular_octagon_area : ∀ (R : ℝ),
  let d_b := 2 * R,
      d_c := 2 * R * Real.sin (Real.pi / 8)
  in R^2 * 4 * Real.sin (Real.pi / 8) = d_b * d_c :=
by
  intros R d_b d_c
  dsimp [d_b, d_c]
  sorry

end regular_octagon_area_l702_702315


namespace find_a1_l702_702153

variable {a : ℕ → ℝ}

-- Define the conditions given in the problem
def sequence_condition_1 : Prop := (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 1)

def sequence_condition_2 : Prop := ∀ n : ℕ, 1 ≤ n ∧ n ≤ 7 → (a (n + 1) / a n = n / (n + 2))

-- The goal is to prove that a_1 = 9/16
theorem find_a1 (h1 : sequence_condition_1) (h2 : sequence_condition_2) : a 1 = 9 / 16 :=
by
  sorry

end find_a1_l702_702153


namespace find_parallel_line_find_perpendicular_line_l702_702801

noncomputable def point_M : ℝ × ℝ := (-1, 2)

def is_line (A B C : ℝ) (P : ℝ × ℝ) : Prop := A * P.1 + B * P.2 + C = 0
def is_parallel (A B C : ℝ) : Prop := ∃ c : ℝ, 2 * A + B = 0 ∧ is_line 2 1 c (0, -5)
def is_perpendicular (A B C : ℝ) : Prop := A * 2 + B = 0 ∧ is_line 2 1 -5 (0, 1)

theorem find_parallel_line : ∃ c, is_line 2 1 c point_M := sorry
theorem find_perpendicular_line : ∃ c, is_line 1 (-2) c point_M := sorry

end find_parallel_line_find_perpendicular_line_l702_702801


namespace S_11_eq_22_l702_702860

variable {S : ℕ → ℕ}

-- Condition: given that S_8 - S_3 = 10
axiom h : S 8 - S 3 = 10

-- Proof goal: we want to show that S_11 = 22
theorem S_11_eq_22 : S 11 = 22 :=
by
  sorry

end S_11_eq_22_l702_702860


namespace Li_age_is_12_l702_702688

-- Given conditions:
def Zhang_twice_Li (Li: ℕ) : ℕ := 2 * Li
def Jung_older_Zhang (Zhang: ℕ) : ℕ := Zhang + 2
def Jung_age := 26

-- Proof problem:
theorem Li_age_is_12 : ∃ Li: ℕ, Jung_older_Zhang (Zhang_twice_Li Li) = Jung_age ∧ Li = 12 :=
by
  sorry

end Li_age_is_12_l702_702688


namespace prove_functions_same_l702_702791

theorem prove_functions_same (u v : ℝ) (huv : u = v) : 
  (u > 1) → (v > 1) → (Real.sqrt ((u + 1) / (u - 1)) = Real.sqrt ((v + 1) / (v - 1))) :=
by
  sorry

end prove_functions_same_l702_702791


namespace largest_number_after_removals_l702_702136

theorem largest_number_after_removals : 
  ∀ (original : ℕ), original = 2946835107 → 
  let n := 5 in
  ∃ (resultant : ℕ), resultant = 98517 ∧ remove_digits original n = resultant :=
by
  sorry

end largest_number_after_removals_l702_702136


namespace neither_probability_l702_702543

-- Definitions of the probabilities P(A), P(B), and P(A ∩ B)
def P_A : ℝ := 0.63
def P_B : ℝ := 0.49
def P_A_and_B : ℝ := 0.32

-- Definition stating the probability of neither event
theorem neither_probability :
  (1 - (P_A + P_B - P_A_and_B)) = 0.20 := 
sorry

end neither_probability_l702_702543


namespace inequality_abc_l702_702289

theorem inequality_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := 
sorry

end inequality_abc_l702_702289


namespace coin_toss_probability_l702_702344

theorem coin_toss_probability:
  (P_heads : ℚ) (P_tails : ℚ) 
  (h : P_heads = 1/4) : 
  P_tails = 3/4 :=
by
  sorry

end coin_toss_probability_l702_702344


namespace product_of_solutions_l702_702484

theorem product_of_solutions : 
  (∀ y : ℝ, |y| = 3 * (|y| - 2) → y = 3 ∨ y = -3) → 
  (∃ y1 y2 : ℝ, (|y1| = 3 ∧ |y2| = 3 ∧ y1 * y2 = -9)) := 
  by
    intro h
    have solution1 : y1 = 3 := sorry
    have solution2 : y2 = -3 := sorry
    use [solution1, solution2]
    split
    · -- y1 = 3
      sorry
    · split
      · -- y2 = 3
        sorry
      · -- product is -9
        sorry

end product_of_solutions_l702_702484


namespace exists_triang_and_square_le_50_l702_702687

def is_triang_num (n : ℕ) : Prop := ∃ m : ℕ, n = m * (m + 1) / 2
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem exists_triang_and_square_le_50 : ∃ n : ℕ, n ≤ 50 ∧ is_triang_num n ∧ is_perfect_square n :=
by
  sorry

end exists_triang_and_square_le_50_l702_702687


namespace a7_is_1_S2022_is_4718_l702_702756

def harmonious_progressive (a : ℕ → ℕ) : Prop :=
  ∀ p q : ℕ, p > 0 → q > 0 → a p = a q → a (p + 1) = a (q + 1)

variables (a : ℕ → ℕ) (S : ℕ → ℕ)

axiom harmonious_seq : harmonious_progressive a
axiom a1 : a 1 = 1
axiom a2 : a 2 = 2
axiom a4 : a 4 = 1
axiom a6_plus_a8 : a 6 + a 8 = 6

theorem a7_is_1 : a 7 = 1 := sorry

theorem S2022_is_4718 : S 2022 = 4718 := sorry

end a7_is_1_S2022_is_4718_l702_702756


namespace perpendicular_line_l702_702832

noncomputable def line_eq (A B C : ℝ) (x y : ℝ) : Prop :=
  A * x + B * y + C = 0

theorem perpendicular_line (A B C : ℝ) :
  (∃ l : ℝ → ℝ, ∀ x y : ℝ, line_eq A B C x y → (A ≠ 0 ∨ B ≠ 0) → B * x - A * y + C = 0) :=
by
  use λ x y, B * x - A * y + C = 0
  intros x y hl hneq
  sorry

end perpendicular_line_l702_702832


namespace point_on_ellipse_l702_702397

noncomputable def ellipse_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  let d1 := ((x - F1.1)^2 + (y - F1.2)^2).sqrt
  let d2 := ((x - F2.1)^2 + (y - F2.2)^2).sqrt
  x^2 + 4 * y^2 = 16 ∧ d1 = 7

theorem point_on_ellipse (P F1 F2 : ℝ × ℝ)
  (h : ellipse_condition P F1 F2) : 
  let x := P.1
  let y := P.2
  let d2 := ((x - F2.1)^2 + (y - F2.2)^2).sqrt
  d2 = 1 :=
sorry

end point_on_ellipse_l702_702397


namespace evaluate_expression_l702_702691

theorem evaluate_expression : | 5 - 8 * (3 - 12) | - | 5 - 11 | = 71 :=
by {
  -- proof steps would go here
  sorry
}

end evaluate_expression_l702_702691


namespace intersection_eq_l702_702609

open Set

def S : Set ℝ := { y | ∃ x : ℝ, y = 3^x }
def T : Set ℝ := { y | ∃ x : ℝ, y = x^2 + 1 }

theorem intersection_eq :
  S ∩ T = T := by
  sorry

end intersection_eq_l702_702609


namespace question_l702_702534

-- Given conditions and definitions

noncomputable def is_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem question (θ : ℝ) (h : is_parallel (2, sin θ) (1, cos θ)) :
  (sin θ)^2 / (1 + (cos θ)^2) = 2 / 3 :=
by
  sorry

end question_l702_702534


namespace partition_square_l702_702091

theorem partition_square (sq : set (ℝ × ℝ)) (shaded : set (ℝ × ℝ))
  (h_sq : sq = { p | 0 ≤ p.1 ∧ p.1 < 4 ∧ 0 ≤ p.2 ∧ p.2 < 4 })
  (h_shaded : shaded = { p | p = (0.5, 0.5) ∨ p = (2.5, 0.5) ∨ p = (0.5, 2.5) ∨ p = (2.5, 2.5) }) :
  ∃ parts : set (set (ℝ × ℝ)), (∀ part ∈ parts, ∃ f : (ℝ × ℝ) ≃ (ℝ × ℝ), function.bijective f ∧ f '' part = part ∧ part ⊆ sq ∧ part.inter shaded = {p | p ∈ shaded ∧ part p}) ∧ parts.card = 4 ∧ (∀ p1 ∈ parts, ∀ p2 ∈ parts, p1 ≠ p2 → disjoint p1 p2) :=
sorry

end partition_square_l702_702091


namespace final_selling_price_l702_702060

-- Define the conditions in Lean
def cost_price_A : ℝ := 150
def profit_A_rate : ℝ := 0.20
def profit_B_rate : ℝ := 0.25

-- Define the function to calculate selling price based on cost price and profit rate
def selling_price (cost_price : ℝ) (profit_rate : ℝ) : ℝ :=
  cost_price + (profit_rate * cost_price)

-- The theorem to be proved
theorem final_selling_price :
  selling_price (selling_price cost_price_A profit_A_rate) profit_B_rate = 225 :=
by
  -- The proof is omitted
  sorry

end final_selling_price_l702_702060


namespace apples_left_proof_l702_702616

noncomputable def apples_left (apples_picked : ℕ) (apples_eaten : ℕ) (pears_picked : ℕ) : ℕ :=
  let apples_disappeared := pears_picked / 3 in
  apples_picked - apples_eaten - apples_disappeared

theorem apples_left_proof :
  let mike_picked := 12 in
  let nancy_ate := 7 in
  let keith_picked_apples := 6 in
  let keith_picked_pears := 4 in
  let christine_picked_apples := 10 in
  let christine_picked_pears := 3 in
  let greg_ate := 9 in 
  let peaches_picked := 14 in
  let plums_picked := 7 in
  let total_apples_picked := mike_picked + keith_picked_apples + christine_picked_apples in
  let total_apples_eaten := nancy_ate + greg_ate in
  let total_pears_picked := keith_picked_pears + christine_picked_pears in
  apples_left total_apples_picked total_apples_eaten total_pears_picked = 10 :=
by 
  sorry

end apples_left_proof_l702_702616


namespace point_on_curve_l702_702181

theorem point_on_curve (x y : ℤ) (h : x = -1 ∧ y = 2) : x^2 - x * y + y - 5 = 0 := by
  obtain ⟨hx, hy⟩ := h
  rw [hx, hy]
  norm_num
  sorry

end point_on_curve_l702_702181


namespace max_value_2ab_2bc_2cd_2da_l702_702342

theorem max_value_2ab_2bc_2cd_2da {a b c d : ℕ} :
  (a = 2 ∨ a = 3 ∨ a = 5 ∨ a = 7) ∧
  (b = 2 ∨ b = 3 ∨ b = 5 ∨ b = 7) ∧
  (c = 2 ∨ c = 3 ∨ c = 5 ∨ c = 7) ∧
  (d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7) ∧
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧
  (b ≠ c) ∧ (b ≠ d) ∧
  (c ≠ d)
  → 2 * (a * b + b * c + c * d + d * a) ≤ 144 :=
by
  sorry

end max_value_2ab_2bc_2cd_2da_l702_702342


namespace castiel_fraction_sausages_ate_l702_702938

theorem castiel_fraction_sausages_ate :
  ∀ (total_sausages monday_fraction tuesday_fraction friday_remainder : ℕ),
    total_sausages = 600 →
    monday_fraction = 2 / 5 →
    tuesday_fraction = 1 / 2 →
    friday_remainder = 45 →
    (let monday_eaten := monday_fraction * total_sausages in
     let after_monday := total_sausages - monday_eaten in
     let tuesday_eaten := tuesday_fraction * after_monday in
     let after_tuesday := after_monday - tuesday_eaten in
     let friday_eaten := after_tuesday - friday_remainder in
     friday_eaten / after_tuesday = 3 / 4) :=
begin
  sorry
end

end castiel_fraction_sausages_ate_l702_702938


namespace card_ge_two_n_plus_two_l702_702281

variable (F : Finset ℤ)
variable (n : ℕ+)

-- Hypotheses
hypothesis H1 : ∀ x ∈ F, ∃ (y z : ℤ), y ∈ F ∧ z ∈ F ∧ x = y + z
hypothesis H2 : ∀ (k : ℕ) (h : 1 ≤ k ∧ k ≤ n) (x : Fin k → ℤ), (∀ i, x i ∈ F) → Finset.univ.sum x ≠ 0

theorem card_ge_two_n_plus_two : F.card ≥ 2 * n + 2 :=
sorry

end card_ge_two_n_plus_two_l702_702281


namespace number_of_palindromes_divisible_by_4_l702_702854

theorem number_of_palindromes_divisible_by_4 :
  let palindromes := {x : ℕ | 100 ≤ x ∧ x < 1000 ∧ 
                      ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ x = 101 * a + 10 * b ∧
                      (2 * a + b) % 4 = 0} in 
  palindromes.card = 22 :=
by
  sorry

end number_of_palindromes_divisible_by_4_l702_702854


namespace find_first_number_l702_702354

theorem find_first_number (x : ℕ) (h1 : x + 35 = 62) : x = 27 := by
  sorry

end find_first_number_l702_702354


namespace range_of_a_l702_702163

def A (a : ℝ) : Set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ x^2 + a * x - y + 2 = 0}
def B : Set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ 2 * x - y + 1 = 0 ∧ x > 0}

theorem range_of_a (a : ℝ) : (∃ p, p ∈ A a ∧ p ∈ B) ↔ a ∈ Set.Iic 0 := by
  sorry

end range_of_a_l702_702163


namespace max_points_top_three_teams_l702_702240

open Finset

theorem max_points_top_three_teams : 
  ∀ (teams : Finset ℕ) (tournament : set (ℕ × ℕ)) 
  (score : ℕ → ℕ) (p : ℕ),
  teams.card = 6 →
  (∀ (a b : ℕ), (a, b) ∈ tournament → (b, a) ∈ tournament) →
  (∀ (a : ℕ), a ∈ teams → score a ≤ 24) →
  (∀ (a b : ℕ), a ≠ b → (score a = 3 * (card {x | ((a, x) ∈ tournament ∧ x ≠ a) ∨ ((x, a) ∈ tournament ∧ x ≠ a)}) ∨ 
  score a = 1 * (card {x | ((a, x) ∈ tournament ∧ x ≠ a) ∨ ((x, a) ∈ tournament ∧ x ≠ a)}) ∨ 
  score a = 0 * (card {x | ((a, x) ∈ tournament ∧ x ≠ a) ∨ ((x, a) ∈ tournament ∧ x ≠ a)}))) →
  (∀ (a b c : ℕ), a ∈ teams → b ∈ teams → c ∈ teams → a ≠ b → b ≠ c → a ≠ c → 
    score a = p ∧ 
    score b = p ∧ 
    score c = p ∧ 
    3 * 30 = 90 ∧ 
    (score a + score b + score c) ≤ 3 * 24) →
  ∃ (p : ℕ), p = 24 := 
sorry

end max_points_top_three_teams_l702_702240


namespace polynomial_root_condition_l702_702707

theorem polynomial_root_condition (f : ℤ[X]) (h_coeffs : ∀ i, |f.coeff i| ≤ 5000000)
  (h_roots : ∀ n : ℤ, 1 ≤ n ∧ n ≤ 20 → ∃ x : ℤ, f.eval x = n * x) : 
  f.eval 0 = 0 :=
by
  sorry

end polynomial_root_condition_l702_702707


namespace f_monotonically_decreasing_l702_702784

def f (x : ℝ) : ℝ := x^2 - 2 * Real.log x

theorem f_monotonically_decreasing : ∃ I : Set ℝ, I = Set.Ioc 0 1 ∧ ∀ x ∈ I, ∀ y ∈ I, x ≤ y → f y ≤ f x :=
by
  refine ⟨Set.Ioc 0 1, ⟨rfl, _⟩⟩
  sorry

end f_monotonically_decreasing_l702_702784


namespace combination_lock_l702_702416

theorem combination_lock :
  (∃ (n_1 n_2 n_3 : ℕ), 
    n_1 ≥ 0 ∧ n_1 ≤ 39 ∧
    n_2 ≥ 0 ∧ n_2 ≤ 39 ∧
    n_3 ≥ 0 ∧ n_3 ≤ 39 ∧ 
    n_1 % 4 = n_3 % 4 ∧ 
    n_2 % 4 = (n_1 + 2) % 4) →
  ∃ (count : ℕ), count = 4000 :=
by
  sorry

end combination_lock_l702_702416


namespace multiple_of_distance_l702_702316

namespace WalkProof

variable (H R M : ℕ)

/-- Rajesh walked 10 kilometers less than a certain multiple of the distance that Hiro walked. 
    Together they walked 25 kilometers. Rajesh walked 18 kilometers. 
    Prove that the multiple of the distance Hiro walked that Rajesh walked less than is 4. -/
theorem multiple_of_distance (h1 : R = M * H - 10) 
                             (h2 : H + R = 25)
                             (h3 : R = 18) :
                             M = 4 :=
by
  sorry

end WalkProof

end multiple_of_distance_l702_702316


namespace sum_of_divisors_of_180_l702_702768

-- Define the number in question
def n : ℕ := 180

-- Define the prime factorization of the number (for reference, not used directly in Lean)
def prime_factorization_of_180 : Prop := n = 2^2 * 3^2 * 5

-- Statement of the problem
theorem sum_of_divisors_of_180 : ∑ d in divisors 180, d = 546 := by
  -- Proof would go here
  sorry

end sum_of_divisors_of_180_l702_702768


namespace int_modulo_l702_702679

theorem int_modulo (n : ℤ) (h1 : 0 ≤ n) (h2 : n < 17) (h3 : 38574 ≡ n [ZMOD 17]) : n = 1 :=
by
  sorry

end int_modulo_l702_702679


namespace greatest_integer_solution_l702_702682

theorem greatest_integer_solution (n : ℤ) (h : n^2 - 13 * n + 36 ≤ 0) : n ≤ 9 :=
by
  sorry

end greatest_integer_solution_l702_702682


namespace liquid_X_percentage_in_solution_B_l702_702612

theorem liquid_X_percentage_in_solution_B:
  let solution_a_pct := 0.008 in
  let solution_a_weight := 400 in
  let solution_b_weight := 700 in
  let mixture_pct := 0.0158 in
  let total_weight := solution_a_weight + solution_b_weight in
  let mixture_liquid_X := total_weight * mixture_pct in
  let liquid_X_in_a := solution_a_weight * solution_a_pct in
  let P := (mixture_liquid_X - liquid_X_in_a) / solution_b_weight in
  (P * 100 ≈ 2.03) :=
by
  sorry

end liquid_X_percentage_in_solution_B_l702_702612


namespace greatest_n_leq_inequality_l702_702684

theorem greatest_n_leq_inequality : ∃ n : ℤ, (n^2 - 13 * n + 36 ≤ 0) ∧ ∀ m : ℤ, (m^2 - 13 * m + 36 ≤ 0) → m ≤ n := 
by
  existsi (9 : ℤ)
  split
  {
    -- Validate that 9 satisfies the inequality
    sorry
  }
  {
    -- Show for any m, if m satisfies the inequality, it must be less than or equals to 9
    intro m
    intro hm
    -- prove m <= 9
    sorry
  }

end greatest_n_leq_inequality_l702_702684


namespace probability_of_both_heads_and_tails_l702_702380

-- Define a function to count the number of heads
def count_heads {α : Type} [DecidableEq α] (l : List α) : ℕ :=
  l.count (λ x => x = "H")

-- Define a function to count the number of tails
def count_tails {α : Type} [DecidableEq α] (l : List α) : ℕ :=
  l.count (λ x => x = "T")

-- Define what it means to have both heads and tails
def has_both_heads_and_tails (l : List String) : Prop :=
  count_heads l > 0 ∧ count_tails l > 0

-- All possible outcomes when three coins are tossed
def outcomes : List (List String) :=
  [['H', 'H', 'H'], ['H', 'H', 'T'], ['H', 'T', 'H'], ['T', 'H', 'H'], 
   ['T', 'T', 'H'], ['T', 'H', 'T'], ['H', 'T', 'T'], ['T', 'T', 'T']]

-- Number of favorable outcomes
def favorable_outcomes : List (List String) :=
  outcomes.filter has_both_heads_and_tails

theorem probability_of_both_heads_and_tails 
  (total : ℕ := outcomes.length) 
  (favorable : ℕ := favorable_outcomes.length) :
  (favorable : ℚ) / total = 3 / 4 :=
by
  have total_eq : total = 8 := by simp [total, outcomes]
  have favorable_eq : favorable = 6 := by simp [favorable, favorable_outcomes, has_both_heads_and_tails, count_heads, count_tails]
  rw [total_eq, favorable_eq]
  norm_num
  sorry

end probability_of_both_heads_and_tails_l702_702380


namespace range_of_m_l702_702198

noncomputable def eccentricity (m : ℝ) : ℝ := (Real.sqrt (5 + m)) / (Real.sqrt 5)

def prop_p (m : ℝ) : Prop := ∃ e ∈ Ioo (Real.sqrt 6 / 2) (Real.sqrt 2), e = eccentricity m

def prop_q (m : ℝ) : Prop := 3 < m ∧ m < 9

theorem range_of_m (m : ℝ) (h_pq : (prop_p m ∨ prop_q m) ∧ ¬(prop_p m ∧ prop_q m)) :
  (2.5 < m ∧ m ≤ 3) ∨ (5 ≤ m ∧ m < 9) :=
sorry

end range_of_m_l702_702198


namespace cupcakes_left_l702_702952

def total_cupcakes : ℕ := 40
def students_class_1 : ℕ := 18
def students_class_2 : ℕ := 16
def additional_individuals : ℕ := 4

theorem cupcakes_left (total_cupcakes students_class_1 students_class_2 additional_individuals : ℕ) :
  total_cupcakes - students_class_1 - students_class_2 - additional_individuals = 2 :=
by
  have h1 : total_cupcakes - students_class_1 = 22 := by sorry
  have h2 : 22 - students_class_2 = 6 := by sorry
  have h3 : 6 - additional_individuals = 2 := by sorry
  exact h3

end cupcakes_left_l702_702952


namespace find_f_and_max_profit_l702_702434

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a / (x - 4)) + 10 * (x - 7) ^ 2

def a_value (f : ℝ → ℝ) : ℝ := 10

def f_formula (x : ℝ) : ℝ := (10 / (x - 4)) + 10 * (x - 7) ^ 2

theorem find_f_and_max_profit:
  (∀ x, (4 < x ∧ x < 7) → ∃ a, f a x = f_formula x)
  ∧ ((∃ x, (4 < x ∧ x < 7) ∧ ∀ y, (4 < y ∧ y < 7) → (x ≠ y → h x > h y))
  ∧ (∃ c, (x = 4) ∧ (h x c = 50)
  :=
by {
  sorry
}

end find_f_and_max_profit_l702_702434


namespace how_many_men_have_all_conditions_l702_702019

theorem how_many_men_have_all_conditions (total_men married_men tv_owners radio_owners ac_owners: ℕ) 
    (h1 : total_men = 100)
    (h2 : married_men = 85)
    (h3 : tv_owners = 75)
    (h4 : radio_owners = 85)
    (h5 : ac_owners = 70) :
    ∃ (x : ℕ), x ≤ married_men ∧ x ≤ ac_owners ∧ x ≤ min tv_owners radio_owners ∧ x ≤ 70 := 
begin
    -- sorry to skip the proof
    sorry
end

end how_many_men_have_all_conditions_l702_702019


namespace total_tickets_needed_l702_702308

-- Definitions representing the conditions
def rides_go_karts : ℕ := 1
def cost_per_go_kart_ride : ℕ := 4
def rides_bumper_cars : ℕ := 4
def cost_per_bumper_car_ride : ℕ := 5

-- Calculate the total tickets needed
def total_tickets : ℕ := rides_go_karts * cost_per_go_kart_ride + rides_bumper_cars * cost_per_bumper_car_ride

-- The theorem stating the main proof problem
theorem total_tickets_needed : total_tickets = 24 := by
  -- Proof steps should go here, but we use sorry to skip the proof
  sorry

end total_tickets_needed_l702_702308


namespace trisect_ratio_l702_702576

noncomputable def trisect_angle (A B C F G : Point) (t : Triangle) : Prop :=
  t.has_vertex A ∧
  (t.has_vertex B ∧ t.has_vertex C) ∧
  ∃ (AF AG : Line),
    (AF ∈ t.angles A) ∧
    (AG ∈ t.angles A) ∧
    ∠BAC = 3 * ∠BAF ∧
    ∠BAF = ∠FAG ∧
    AF.meets BC F ∧
    AG.meets BC G

theorem trisect_ratio (A B C F G : Point) (t : Triangle)
  (Htrisect : trisect_angle A B C F G t)
  (Hmeet : AF.meets BC F ∧ AG.meets BC G):
  BF / GC = (AB * AF) / (AG * AC) :=
begin
  sorry
end

end trisect_ratio_l702_702576


namespace smallest_number_last_four_digits_l702_702277

theorem smallest_number_last_four_digits :
  ∃ (n : ℕ), (n % 10 = 4 ∨ n % 10 = 6 ∨ n % 10 = 9) ∧
             ((n / 10) % 10 = 4 ∨ (n / 10) % 10 = 6 ∨ (n / 10) % 10 = 9) ∧
             ((n / 100) % 10 = 4 ∨ (n / 100) % 10 = 6 ∨ (n / 100) % 10 = 9) ∧
             ((n / 1000) % 10 = 4 ∨ (n / 1000) % 10 = 6 ∨ (n / 1000) % 10 = 9) ∧
             (n % 2 = 0) ∧ (n % 3 = 0) ∧ (n % 9 = 0) ∧
             (∃ n4 n6 n9, n4 > 0 ∧ n6 > 0 ∧ n9 > 0 ∧
              n4 + n6 + n9 = 4 ∧
              ∀ (k : ℕ), (n / 10^k) % 10 = 4 → n4 = n4 - 1 ∧
                         (n / 10^k) % 10 = 6 → n6 = n6 - 1 ∧
                         (n / 10^k) % 10 = 9 → n9 = n9 - 1) ∧
             (n % 10000 = 4699) :=
proof
  sorry

end smallest_number_last_four_digits_l702_702277


namespace lemons_needed_and_cost_l702_702710

theorem lemons_needed_and_cost (lemons_per_gallon : ℕ) (cost_per_lemon : ℝ) : 
  lemons_per_gallon = 36 / 60 ∧ cost_per_lemon = 0.50 → 
  (let lemons_needed := 15 * lemons_per_gallon / 1 in
  lemons_needed = 9 ∧ lemons_needed * cost_per_lemon = 4.50) := 
by
  sorry

end lemons_needed_and_cost_l702_702710


namespace lewis_weekly_rent_l702_702294

theorem lewis_weekly_rent (total_rent_paid : ℕ) (number_of_weeks : ℕ) (weekly_rent : ℤ)
  (h1 : total_rent_paid = 527292)
  (h2 : number_of_weeks = 1359) :
  weekly_rent ≈ 388 :=
by
  sorry

end lewis_weekly_rent_l702_702294


namespace nh3_oxidation_mass_l702_702451

theorem nh3_oxidation_mass
  (initial_volume : ℚ)
  (initial_cl2_percentage : ℚ)
  (initial_n2_percentage : ℚ)
  (escaped_volume : ℚ)
  (escaped_cl2_percentage : ℚ)
  (escaped_n2_percentage : ℚ)
  (molar_volume : ℚ)
  (cl2_molar_mass : ℚ)
  (nh3_molar_mass : ℚ) :
  initial_volume = 1.12 →
  initial_cl2_percentage = 0.9 →
  initial_n2_percentage = 0.1 →
  escaped_volume = 0.672 →
  escaped_cl2_percentage = 0.5 →
  escaped_n2_percentage = 0.5 →
  molar_volume = 22.4 →
  cl2_molar_mass = 71 →
  nh3_molar_mass = 17 →
  ∃ (mass_nh3_oxidized : ℚ),
    mass_nh3_oxidized = 0.34 := 
by {
  sorry
}

end nh3_oxidation_mass_l702_702451


namespace solution_added_volume_l702_702026

theorem solution_added_volume (amount_solution : ℝ) :
  (∃ x : ℝ, 1 + x = amount_solution + 1 ∧ 
  0.33 * x = 0.2475 * (x + 1)) → 
  amount_solution = 3 :=
by
  intro h,
  cases h with x hx,
  cases hx with hx1 hx2,
  have h : amount_solution = x :=
    by linarith,
  rw [h] at *,
  have : 0.33 * x = 0.2475 * (x + 1) :=
    by exact hx2,
  sorry

end solution_added_volume_l702_702026


namespace find_a_l702_702639

def f (x : ℝ) : ℝ := x / 3 + 4
def g (x : ℝ) : ℝ := 7 - x

theorem find_a (a : ℝ) (h : f (g a) = 6) : a = 1 := by
  sorry

end find_a_l702_702639


namespace problem_statement_l702_702200

variables {x y z w p q : Prop}

theorem problem_statement (h1 : x = y → z ≠ w) (h2 : z = w → p ≠ q) : x ≠ y → p ≠ q :=
by
  sorry

end problem_statement_l702_702200


namespace sixteen_pow_five_eq_four_pow_p_l702_702213

theorem sixteen_pow_five_eq_four_pow_p (p : ℕ) (h : 16^5 = 4^p) : p = 10 := 
  sorry

end sixteen_pow_five_eq_four_pow_p_l702_702213


namespace find_t_l702_702215

variable (ω α t ω₀ θ : ℝ)

-- Define the given conditions as hypotheses
def condition1 := ω = α * t + ω₀
def condition2 := θ = (1 / 2) * α * t^2 + ω₀ * t

-- State the goal as an equality
theorem find_t (h1 : condition1) (h2 : condition2) : t = (2 * θ) / (ω + ω₀) :=
sorry

end find_t_l702_702215


namespace equivalent_n_is_p_minus_one_l702_702474

theorem equivalent_n_is_p_minus_one (n : ℕ) (h_pos : n > 0) : 
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → (n.factorial / (i.factorial * (n - i + 1).factorial) : ℚ).denom = 1) 
  ↔ (∃ p : ℕ, p.prime ∧ n = p - 1) :=
begin
  sorry
end

end equivalent_n_is_p_minus_one_l702_702474


namespace inequality_holds_for_all_x_l702_702465

theorem inequality_holds_for_all_x (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x ^ 2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Icc (-2 : ℝ) 2 :=
sorry

end inequality_holds_for_all_x_l702_702465


namespace log_sum_simplification_l702_702322

theorem log_sum_simplification : log 10 4 + log 10 25 = 2 :=
by sorry

end log_sum_simplification_l702_702322


namespace angle_BHM_90_deg_l702_702574

variables {P Q R S A B C D M H : Type} [AffineSpace ℝ P]

/-- Points A, B, C, D are the midpoints of sides PQ, QR, RS, SP respectively. -/
def midpoints (P Q R S A B C D : P) : Prop :=
(PQ midpoints A) ∧
(QR midpoints B) ∧
(RS midpoints C) ∧
(SP midpoints D)

/-- Point M is the midpoint of CD -/
def midpoint_CD (C D M : P) : Prop :=
(M pictures in_middle_of CD)

/-- Point H is on line AM such that HC = BC. -/
def H_condition (A M H C B : P) [Line ℝ A M] : Prop :=
(Point_on_line AM H) ∧
(distance H C = distance B C)

/-- Theorem to prove ∠BHM = 90°. -/
theorem angle_BHM_90_deg 
  (all_points : (P Q R S A B C D M H : P))
  (h_midpoints : midpoints P Q R S A B C D)
  (h_midpoint_CD : midpoint_CD C D M)
  (h_H_condition : H_condition A M H C B) :
  angle B H M = 90 :=
sorry

end angle_BHM_90_deg_l702_702574


namespace find_x_in_interval_l702_702282

def fn (n : ℕ) (x : ℝ) : ℝ := (Real.sin x)^n + (Real.cos x)^n

theorem find_x_in_interval :
  {x : ℝ | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 4 * fn 6 x - 3 * fn 4 x = fn 2 x}.finite.to_finset.card = 5 :=
  by
  sorry

end find_x_in_interval_l702_702282


namespace f_50_eq_68_l702_702092

noncomputable def f : ℕ → ℕ
| x := if hx : Real.logb 3 (x : ℝ) ∈ ℤ then Real.logb 3 (x : ℝ).to_nat
       else 2 + f (x + 1)

theorem f_50_eq_68 : f 50 = 68 := by
  sorry

end f_50_eq_68_l702_702092


namespace find_sides_of_triangle_l702_702347

noncomputable def right_triangle_sides 
  (ρ : ℝ) (α : ℝ) 
  (hρ : ρ = 10) (hα : α = 23 + 14/60 * 1/3600 * 360) : 
  ℝ × ℝ × ℝ :=
let a := ρ * (1 + real.cot ((33 + 23/60) * real.pi / 180))
let b := ρ * (1 + real.cot ((11 + 37/60) * real.pi / 180))
let c := ρ * (real.cot ((33 + 23/60) * real.pi / 180) + real.cot ((11 + 37/60) * real.pi / 180))
in (a, b, c)

theorem find_sides_of_triangle : 
  ∀ (a b c : ℝ), 
  ∀ (ρ : ℝ) (α : ℝ)
  (hρ : ρ = 10) (hα : α = 23 + 14 / 60 * 1 / 3600 * 360), 
  right_triangle_sides ρ α hρ hα = (a, b, c) → 
  a = 25.18 ∧ b = 58.65 ∧ c = 63.82 := 
  by
  intros a b c ρ α hρ hα
  intro h
  rw right_triangle_sides at h
  sorry

end find_sides_of_triangle_l702_702347


namespace circle_equation_midpoint_trajectory_equation_l702_702500

open Real

noncomputable def circle_center {C : ℝ × ℝ} (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop) := 
  ∃ r : ℝ, ∀ P, (P = C → l C) ∧ (l = (λ x, fst x - snd x + 1 = 0)) ∧
  (distance A C = r) ∧ (distance B C = r)

theorem circle_equation : 
  circle_center (-3, -2) (1,1) (2,-2) (λ x, fst x - snd x + 1 = 0) → 
  ∀ x y : ℝ, (x + 3) ^ 2 + (y + 2) ^ 2 = 25 := 
by 
  sorry

noncomputable def midpoint_trajectory {A C M : ℝ × ℝ} (A C : ℝ × ℝ) := 
  ∀ x y : ℝ, (x + 1) ^ 2 + (y + 1) ^ 2 = 5

theorem midpoint_trajectory_equation : 
  (midpoint_trajectory (1, 0) (-3, -2)) :=
by 
  sorry

end circle_equation_midpoint_trajectory_equation_l702_702500


namespace next_two_series_numbers_l702_702538

theorem next_two_series_numbers :
  ∀ (a : ℕ → ℤ), a 1 = 2 → a 2 = 3 →
    (∀ n, 3 ≤ n → a n = a (n - 1) + a (n - 2) - 5) →
    a 7 = -26 ∧ a 8 = -45 :=
by
  intros a h1 h2 h3
  sorry

end next_two_series_numbers_l702_702538


namespace characterisation_of_set_S_l702_702094
-- Set up the imports

-- Define the problem conditions and statement
theorem characterisation_of_set_S (S : set (ℝ × ℝ)) :
  (∀ (A B C D : ℝ × ℝ), A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D → 
    (∃ (circle : set (ℝ × ℝ)), {A, B, C, D} ⊆ circle) ∨ (∃ (line : set (ℝ × ℝ)), ∃ X Y Z ∈ {A, B, C, D}, X ≠ Y ≠ Z ≠ X ∧ {X, Y, Z} ⊆ line)) →
  (∃ (P : set (ℝ × ℝ)), S ⊆ P) ∨ 
  (∃ (Q : set (ℝ × ℝ)), S ⊆ Q ∧ 
    (∃ E ∈ S, E ∉ Q)) ∨ 
  (∃ (A B C D : ℝ × ℝ), {A, B, C, D} ⊆ S ∧
    (∃ (diag_intersect : ℝ × ℝ), diag_intersect ∈ S ∧ 
      (∃ (diagonals : list (set (ℝ × ℝ))), ∀ X Y ∈ {A, B, C, D}, X ≠ Y → X ∈ A ∪ B → Y ∈ C ∪ D → diag_intersect ∈ diagonals)
    ∨
    (∃ (side_intersect : ℝ × ℝ), side_intersect ∈ S ∧ 
      (∃ (sides : list (set (ℝ × ℝ))), ∀ U V W ∈ {A, B, C, D}, U ≠ V ≠ W ≠ U → U ∈ A ∪ B ∧ V ∈ B ∪ C ∧ W ∈ C ∪ D → side_intersect ∈ sides)
    ))
) :=
sorry

end characterisation_of_set_S_l702_702094


namespace domain_of_f_expression_of_f_min_max_of_f_l702_702717

-- Problem 1
theorem domain_of_f : 
  ∀ x, (4 - x ≥ 0) ∧ (x - 1 ≠ 0) ↔ (x ≤ 4 ∧ x ≠ 1) := 
sorry

-- Problem 2
theorem expression_of_f (f : ℝ → ℝ) : 
  (∀ x, f (x - 1) = x^2 + 2 * x + 3) → (∀ x, f x = x^2 + 4 * x + 6) := 
sorry

-- Problem 3
theorem min_max_of_f :
  ∃ (min_val max_val : ℝ), 
  min_val = (∀ x ∈ set.Icc 0 3, f x) ∧
  max_val = (∃ y ∈ set.Icc 0 3, f y = 6) ∧ 
  min_val = (∃ z ∈ set.Icc 0 3, f z = 2) :=
sorry

end domain_of_f_expression_of_f_min_max_of_f_l702_702717


namespace quadrilateral_proof_l702_702425

-- Define the quadrilateral
variables {A B C D O : Type*} 
variables [inhabited O] [Π (P : O), P → Point P]

-- Define the incircle centered at O
def is_circumscribed (A B C D : Point O) (O : Point O) : Prop :=
  ∀ P, inscribed_around_circle P A B C D O

-- Conditions for a quadrilateral and incircle
variables [H1 : pairwise_non_parallel A B C D]
variables [H2 : is_circumscribed A B C D O]

-- Define the intersection of the midlines
def intersection_of_midlines (A B C D : Point O) (O : Point O) : Prop :=
  ∃ X Y, midpoint X A B ∧ midpoint Y C D ∧ intersect_midlines X Y = O

-- Theorem statement
theorem quadrilateral_proof (H1 : pairwise_non_parallel A B C D)
(H2 : is_circumscribed A B C D O) :
intersection_of_midlines A B C D O ↔ OA * OC = OB * OD := 
sorry -- Proof is skipped

end quadrilateral_proof_l702_702425


namespace find_a1_l702_702151

theorem find_a1
  (a : ℕ → ℝ)
  (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 1)
  (h_rec : ∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7} → a (n + 1) = a n * (n / (n + 2))) :
  a 1 = 9 / 16 :=
by
  sorry

end find_a1_l702_702151


namespace circle_center_sum_l702_702487

theorem circle_center_sum (h k : ℝ) :
  (∀ x y : ℝ, (x - h) ^ 2 + (y - k) ^ 2 = x ^ 2 + y ^ 2 - 6 * x - 8 * y + 38) → h + k = 7 :=
by sorry

end circle_center_sum_l702_702487


namespace factorial_ends_with_base_16_zeroes_15_l702_702536

theorem factorial_ends_with_base_16_zeroes_15 : 
  (number_of_zeroes_in_base (factorial 15) 16) = 2 := 
sorry

/--
 This auxiliary definition calculates the number of trailing zeroes when a given number is written in a specified base.
-/
def number_of_zeroes_in_base (n : ℕ) (base: ℕ) : ℕ := sorry

end factorial_ends_with_base_16_zeroes_15_l702_702536


namespace outfit_combinations_l702_702974

theorem outfit_combinations :
  let num_shirts := 5
  let num_pants := 4
  let num_ties := 6 -- 5 ties + no tie option
  let num_belts := 3 -- 2 belts + no belt option
  num_shirts * num_pants * num_ties * num_belts = 360 :=
by
  let num_shirts := 5
  let num_pants := 4
  let num_ties := 6
  let num_belts := 3
  show num_shirts * num_pants * num_ties * num_belts = 360
  sorry

end outfit_combinations_l702_702974


namespace possible_pairs_l702_702665

theorem possible_pairs (n k : ℕ) (h_n : n ≥ 2) (h_total_score : ∀ i, i < n → ∀ j, j < k → (∑ d in finset.range k, (finset.range n).sum ((λ s, if s = i then 1 else if s = j then 2 else if s = i+j then 3 else 0)) = 26)) : (n, k) = (25, 2) ∨ (n, k) = (12, 4) ∨ (n, k) = (3, 13) :=
by sorry

end possible_pairs_l702_702665


namespace trajectory_of_Q_existence_of_M_l702_702846

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := (x + 2) ^ 2 + y ^ 2 = 81 / 16
def C2 (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 1 / 16

-- Define the conditions about circle Q
def is_tangent_to_both (Q : ℝ → ℝ → Prop) : Prop :=
  ∃ r : ℝ, (∀ x y : ℝ, Q x y → (x + 2)^2 + y^2 = (r + 9/4)^2) ∧ (∀ x y : ℝ, Q x y → (x - 2)^2 + y^2 = (r + 1/4)^2)

-- Prove the trajectory of the center of Q
theorem trajectory_of_Q (Q : ℝ → ℝ → Prop) (h : is_tangent_to_both Q) :
  ∀ x y : ℝ, Q x y ↔ (x^2 - y^2 / 3 = 1 ∧ x ≥ 1) :=
sorry

-- Prove the existence and coordinates of M
theorem existence_of_M (M : ℝ) (Q : ℝ → ℝ → Prop) (h : is_tangent_to_both Q) :
  ∃ x y : ℝ, (x, y) = (-1, 0) ∧ (∀ x0 y0 : ℝ, Q x0 y0 → ((-y0 / (x0 - 2) = 2 * (y0 / (x0 - M)) / (1 - (y0 / (x0 - M))^2)) ↔ M = -1)) :=
sorry

end trajectory_of_Q_existence_of_M_l702_702846


namespace ratio_bianca_meg_l702_702901

-- Define initial conditions from the problem statement
def initial_money (h: ℝ) := 2 * h
def given_to_meg := 8
def money_left := 54

-- Use the given condition half of Jerome's initial money was $43
def half_initial_money := 43

-- Define the amount given away and to Bianca
def total_given_away (initial: ℝ) (left: ℝ) := initial - left
def given_to_bianca (total_given: ℝ) (given_meg: ℝ) := total_given - given_meg

-- Define the ratio statement
def ratio (a b: ℝ) := a / b

-- The main theorem to state the ratio of the amount given to Bianca to the amount given to Meg
theorem ratio_bianca_meg : 
  ratio (given_to_bianca (total_given_away (initial_money half_initial_money) money_left) given_to_meg) given_to_meg = 3 := 
by 
  sorry

end ratio_bianca_meg_l702_702901


namespace total_weeds_correct_l702_702960

def tuesday : ℕ := 25
def wednesday : ℕ := 3 * tuesday
def thursday : ℕ := wednesday / 5
def friday : ℕ := thursday - 10
def total_weeds : ℕ := tuesday + wednesday + thursday + friday

theorem total_weeds_correct : total_weeds = 120 :=
by
  sorry

end total_weeds_correct_l702_702960


namespace find_a1_l702_702152

variable {a : ℕ → ℝ}

-- Define the conditions given in the problem
def sequence_condition_1 : Prop := (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 1)

def sequence_condition_2 : Prop := ∀ n : ℕ, 1 ≤ n ∧ n ≤ 7 → (a (n + 1) / a n = n / (n + 2))

-- The goal is to prove that a_1 = 9/16
theorem find_a1 (h1 : sequence_condition_1) (h2 : sequence_condition_2) : a 1 = 9 / 16 :=
by
  sorry

end find_a1_l702_702152


namespace max_number_valid_words_l702_702569

def max_num_words (a u : Char) : Nat :=
  let N := 2^1 + 2^2 + 2^3 + ... + 2^13          -- Total sequences of length 1 to 13
  let S := 2^7 + 2^8 + 2^9 + 2^10 + 2^11 + 2^12 + 2^13  -- Total valid sequences of length 7 to 13
  N - S

theorem max_number_valid_words : max_num_words 'a' 'u' = 16056 :=
sorry

end max_number_valid_words_l702_702569


namespace squared_expression_l702_702709

variable {x y : ℝ}

theorem squared_expression (x y : ℝ) : (-3 * x^2 * y)^2 = 9 * x^4 * y^2 :=
  by
  sorry

end squared_expression_l702_702709


namespace measure_angle_DSO_l702_702896

-- Define the angles and properties of the triangle and segment
def triangle_DOG (A B C : Type) := ∃ (D G O : A), 
  ∃ (angle_DGO angle_DOG angle_GOD : ℝ),
  angle_DGO = angle_DOG ∧ angle_GOD = 45 ∧ 
  (∃ (S : B), ∃ (bisects : Prop), bisects ∧ 
    ∃ (angle_DSO : ℝ),
      (2 * angle_DOG = 180 - angle_GOD) ∧
      angle_DOG = angle_DGO ∧
      angle_DOG / 2 = angle_DOS ∧
      180 - angle_DSO - (angle_DOS + angle_DOG / 2) = angle_DSO)

-- Prove that the measure of ∠DSO is 78.75 degrees
theorem measure_angle_DSO {A B : Type} (D G O : A) (S : B) 
  (angle_DGO angle_DOG angle_GOD angle_DSO angle_DOS : ℝ) 
  (h1 : angle_DGO = angle_DOG) 
  (h2 : angle_GOD = 45) 
  (h3 : ∃ (bisects : Prop), bisects ∧ angle_DOS = angle_DOG / 2) 
  (h4 : 2 * angle_DOG = 180 - angle_GOD) 
  (h5 : 180 - angle_DSO - (angle_DOS + angle_DOG / 2) = angle_DSO)
  : angle_DSO = 78.75 :=
sorry

end measure_angle_DSO_l702_702896


namespace range_of_a_in_fourth_quadrant_l702_702545

-- Definition of the problem in Lean 4
theorem range_of_a_in_fourth_quadrant (a : ℝ) (h1 : a > 0) (h2 : a - 2 < 0) : 
  0 < a ∧ a < 2 :=
by
  split
  · exact h1
  · linarith [h2]

end range_of_a_in_fourth_quadrant_l702_702545


namespace max_number_valid_words_l702_702570

def max_num_words (a u : Char) : Nat :=
  let N := 2^1 + 2^2 + 2^3 + ... + 2^13          -- Total sequences of length 1 to 13
  let S := 2^7 + 2^8 + 2^9 + 2^10 + 2^11 + 2^12 + 2^13  -- Total valid sequences of length 7 to 13
  N - S

theorem max_number_valid_words : max_num_words 'a' 'u' = 16056 :=
sorry

end max_number_valid_words_l702_702570


namespace find_m_n_sum_l702_702049

def P : ℕ × ℕ → ℚ
| (0, 0)       := 1
| (x+1, 0)     := 0
| (0, y+1)     := 0
| (x+1, y+1)   := (1 / 3) * P (x, y+1) + (1 / 3) * P (x+1, y) + (1 / 3) * P (x, y)

theorem find_m_n_sum : ∃ m n : ℕ, P (5, 5) = m / 3^n ∧ m % 3 ≠ 0 ∧ m + n = 6 := sorry

end find_m_n_sum_l702_702049


namespace range_of_a_l702_702523

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then (1/2 : ℝ) * x - 1 else 1 / x

theorem range_of_a (a : ℝ) : f a > a ↔ a < -1 :=
sorry

end range_of_a_l702_702523


namespace derivative_of_f_l702_702704

-- Define the function y
def f (x : ℝ) : ℝ := (x * sin x) ^ (8 * log (x * sin x))

-- State the theorem to prove the derivative
theorem derivative_of_f (x : ℝ) (hx : x ≠ 0 ∧ sin x ≠ 0) : 
  deriv f x = (16 * (x * sin x) ^ (8 * log (x * sin x)) * log (x * sin x) * (1 + x * cot x)) / x := 
by 
  sorry

end derivative_of_f_l702_702704


namespace circle_equation_from_tangents_and_parabola_l702_702840

/-- Given the parabola x^2 = 4y and the point H(1, -1), prove that the equation of the circle with the segment AB as its diameter, where A and B are points of intersection between the parabola and the tangent lines through H, is (x - 1)^2 + (y - 3/2)^2 = 25/4. -/
theorem circle_equation_from_tangents_and_parabola (x y : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ, x1^2 = 4 * y1 ∧ x2^2 = 4 * y2 ∧
      (x1 - 2 * y1 + 2 = 0 ∧ x2 - 2 * y2 + 2 = 0) ∧
      (1 - 2 * (-1) + 2 = 0 ∧ (x1^2 - 0) + (x2^2 - 0 = 5))
  </} ></-- +
  (x - 1)^2 + (y - (3/2))^2 = 25 / 4 :=}


end circle_equation_from_tangents_and_parabola_l702_702840


namespace number_of_goats_l702_702043

theorem number_of_goats : ∃ (G C P : ℕ), P = 2 * C ∧ C = G + 4 ∧ G + C + P = 56 ∧ G = 11 :=
by {
  use [11, 15, 30],
  split, {
    -- P = 2C
    exact rfl,
  },
  split, {
    -- C = G + 4
    exact rfl,
  },
  split, {
    -- G + C + P = 56
    exact rfl,
  },
  -- G = 11
  exact rfl
}

end number_of_goats_l702_702043


namespace max_area_of_trapezoid_l702_702620

theorem max_area_of_trapezoid :
  ∃ B C D : ℝ,
  (B = 1 ∧ C = 1 ∧ D = 1) → 
  (∃ a : ℝ, a > 1 ∧ 
    (area_of_trapezoid 1 1 1 a = 3 * Real.sqrt 3 / 4)) :=
by
intros h
sorry

end max_area_of_trapezoid_l702_702620


namespace prob_one_daily_necessity_max_m_for_favorable_promotion_l702_702793

section
variables {c h d : ℕ}

-- Condition: mall has 2 kinds of clothing, 2 kinds of home appliances, and 3 kinds of daily necessities
def total_items : ℕ := (c + h + d)
def total_clothing : ℕ := 2
def total_home_appliances : ℕ := 2
def total_daily_necessities : ℕ := 3

-- Question 1: Calculate the probability that at least one of the 3 chosen items is a daily necessity
def prob_at_least_one_daily_necessity (c h d : ℕ) :=
  1 - (nat.choose (c + h) 3) / (nat.choose (c + h + d) 3)

theorem prob_one_daily_necessity (c h d : ℕ) (h_c : c = 2) (h_h : h = 2) (h_d : d = 3) :
  prob_at_least_one_daily_necessity c h d = 31 / 35 := sorry

-- Question 2: Maximum m for promotion scheme to be favorable for the mall
def expected_cash_prize (m : ℝ) : ℝ :=
  0 * (1/8) + m * (3/8) + 2 * m * (3/8) + 3 * m * (1/8)

def max_m_condition (m : ℝ) : Prop :=
  expected_cash_prize m ≤ 150

theorem max_m_for_favorable_promotion (m : ℝ) : max_m_condition m ↔ m ≤ 100 := sorry

end

end prob_one_daily_necessity_max_m_for_favorable_promotion_l702_702793


namespace area_of_rhombus_of_roots_l702_702095

noncomputable def roots (p : Polynomial ℂ) : set ℂ := {z : ℂ | Polynomial.eval z p = 0}

def is_rhombus (s : set ℂ) : Prop :=
∃ w x y z : ℂ, s = {w, x, y, z} ∧ 
  (∀ p q ∈ s, (p - q).abs = (w - x).abs ∨ (p - q).abs = (w - y).abs) ∧
  ((w - x).abs = (y - z).abs ∧ (x - y).abs = (w - z).abs)

theorem area_of_rhombus_of_roots :
  let p := Polynomial.C (1 - 4*ⅈ) + Polynomial.X * (Polynomial.C (10 - ⅈ) + 
            Polynomial.X * (Polynomial.C (5 + 5*ⅈ) + Polynomial.X * (Polynomial.C 4*ⅈ + 
            Polynomial.X))) in
  is_rhombus (roots p) →
  ∃ A : ℝ, A = 2 * Real.sqrt 10 :=
by
  sorry

end area_of_rhombus_of_roots_l702_702095


namespace diagonals_of_square_equal_l702_702333

theorem diagonals_of_square_equal
  (H1 : ∀ (R : Type) [rect : Rectangle R], diagonals R)
  (H2 : ∀ (S : Type) [sq : Square S], Rectangle S) :
  ∀ (S : Type) [sq : Square S], diagonals S :=
by
  sorry

end diagonals_of_square_equal_l702_702333


namespace complex_on_real_axis_l702_702812

theorem complex_on_real_axis (a : ℝ) : ∃ (z : ℂ), z = (a - complex.I) * (1 + complex.I) ∧ ∀ (z : ℂ), z.im = 0 → a = 1 := 
by
  sorry

end complex_on_real_axis_l702_702812


namespace oranges_in_bin_l702_702743

theorem oranges_in_bin (initial_oranges : ℕ) (thrown_away : ℕ) (new_oranges : ℕ)
  (h1 : initial_oranges = 5)
  (h2 : thrown_away = 2)
  (h3 : new_oranges = 28) : 
  initial_oranges - thrown_away + new_oranges = 31 :=
by
  rw [h1, h2, h3]
  norm_num


end oranges_in_bin_l702_702743


namespace g_difference_l702_702863

-- Define the function g(n)
def g (n : ℤ) : ℚ := (1/2 : ℚ) * n^2 * (n + 3)

-- State the theorem
theorem g_difference (s : ℤ) : g s - g (s - 1) = (1/2 : ℚ) * (3 * s - 2) := by
  sorry

end g_difference_l702_702863


namespace au_tribe_max_words_correct_l702_702572

def au_tribe_max_words : ℕ :=
  let total_sequences := 2^14 - 2
  let valid_sequences := 2^14 - 2^7
  valid_sequences

theorem au_tribe_max_words_correct : 
  ∃ n : ℕ, au_tribe_max_words = n ∧ n = 16056 :=
by
  use 16056
  split
  · rfl
  · sorry

end au_tribe_max_words_correct_l702_702572


namespace couscous_dishes_l702_702402

def dishes (a b c d : ℕ) : ℕ := (a + b + c) / d

theorem couscous_dishes :
  dishes 7 13 45 5 = 13 :=
by
  unfold dishes
  sorry

end couscous_dishes_l702_702402


namespace couscous_dishes_l702_702401

def dishes (a b c d : ℕ) : ℕ := (a + b + c) / d

theorem couscous_dishes :
  dishes 7 13 45 5 = 13 :=
by
  unfold dishes
  sorry

end couscous_dishes_l702_702401


namespace max_intersections_of_circle_and_two_lines_l702_702032

theorem max_intersections_of_circle_and_two_lines (C : set (ℝ × ℝ)) (L1 L2 : set (ℝ × ℝ)) 
  (hC : ∀ p : ℝ × ℝ, p ∈ C → (p.1 - a)^2 + (p.2 - b)^2 = r^2)
  (hL1 : ∀ p : ℝ × ℝ, p ∈ L1 → a₁ * p.1 + b₁ * p.2 = c₁)
  (hL2 : ∀ p : ℝ × ℝ, p ∈ L2 → a₂ * p.1 + b₂ * p.2 = c₂)
  (hL1_dist : L1 ≠ ∅)
  (hL2_dist : L2 ≠ ∅)
  (hL1L2_distinct: ∀ p : ℝ × ℝ, p ∈ L1 ∧ p ∈ L2 → false):
  ∃ n : ℕ, n = 5 ∧ 
    (∀ L : set (ℝ × ℝ), 
    ((∀ p : ℝ × ℝ, p ∈ L → (∃ L', L' = L1 ∨ L' = L2) ∧ 
      (p ∈ C → L ∈ {L1, L2})) → 
          (∀ p : ℝ × ℝ, p ∈ C ∪ L1 ∪ L2 → p ∈ C ∩ L1 ∨ p ∈ C ∩ L2 ∨ p ∈ L1 ∩ L2) → 
              card ((C ∩ L1) ∪ (C ∩ L2) ∪ (L1 ∩ L2)) = n)) := sorry

end max_intersections_of_circle_and_two_lines_l702_702032


namespace solution_of_system_l702_702789

noncomputable def system_of_equations (x y : ℝ) :=
  x = 1.12 * y + 52.8 ∧ x = y + 50

theorem solution_of_system : 
  ∃ (x y : ℝ), system_of_equations x y ∧ y = -23.33 ∧ x = 26.67 :=
by
  sorry

end solution_of_system_l702_702789


namespace product_of_possible_values_of_N_l702_702071

theorem product_of_possible_values_of_N :
  ∀ (M L : ℝ), 
    -- Initial conditions
    (M = L + N) →
    -- Temperature changes by 8:00 PM
    (M₈₀₀₀ = M - 10) ∧ (L₈₀₀₀ = L + 6) →
    -- Difference of temperatures at 8:00 PM is 3 degrees
    (|M₈₀₀₀ - L₈₀₀₀| = 3) →
    -- Product of all possible values for N
    let possible_values : set ℝ := {n | n - 16 = 3 ∨ n - 16 = -3} in
    ∃ n1 n2 : ℝ, n1 ∈ possible_values ∧ n2 ∈ possible_values ∧ (n1 ≠ n2 ∧ n1 * n2 = 247) :=
by
  -- We define the variables M and L
  assume M L,
  -- Introduce the initial conditions and the temperature at 8:00 AM in Minneapolis
  assume h1 : M = L + N,
  -- Time Passed to 8:00 PM
  assume h2 : (M₈₀₀₀ = M - 10) ∧ (L₈₀₀₀ = L + 6),
  -- The difference is 3 degrees
  assume h3 : |M₈₀₀₀ - L₈₀₀₀| = 3,
  -- We define the possible solutions
  have n1 : ℝ := 19,
  have n2 : ℝ := 13,
  -- We return the result
  use [19, 13],
  split,
  { 
    -- Conditions for n1
    show n1 ∈ possible_values,
    exact or.inl (by norm_num),
  },
  split,
  { 
    -- Conditions for n2
    show n2 ∈ possible_values,
    exact or.inr (by norm_num),
  },
  -- Show that the product is indeed 247
  show n1 ≠ n2 ∧ n1 * n2 = 247, 
  exact ⟨by norm_num, by norm_num⟩ ⟩

end product_of_possible_values_of_N_l702_702071


namespace right_triangle_solution_l702_702886

-- Definitions in terms of the given conditions
def a := Real.sqrt 5
def b := Real.sqrt 15

-- The proof problem statement
theorem right_triangle_solution :
  ∃ (c : ℝ) (A B : ℝ),
    c = 2 * Real.sqrt 5 ∧
    A = 30 ∧
    B = 60 ∧
    c^2 = a^2 + b^2 ∧
    Real.tan (Real.ofDegrees A) = a / b ∧
    A + B = 90 := 
by
  sorry

end right_triangle_solution_l702_702886


namespace ab_value_l702_702673

theorem ab_value (a b : ℝ) (h1 : a + b = 8) (h2 : a^3 + b^3 = 172) : ab = 85 / 6 := 
by
  sorry

end ab_value_l702_702673


namespace proof_problem_l702_702010

/-- Definitions of the given sets and statements --/
def setA : Set (Set ℕ) := { {0} }
def setB : Set ℕ := {0, 1, 2}
def setC : Set ℕ := {2, 1, 0}
def emptySet := (∅ : Set (Set ℕ)) -- empty set of sets containing ℕ
def singletonEmptySet : Set (Set (Set ℕ)) := { ∅ }
def empty := (∅ : Set ℕ) -- empty set containing ℕ
def singletonEmpty := { empty }
def setAB : Set (Set ℕ) := { {0}, {1} }
def singleTuple := { (0, 1) }
def setD : Set (Set (ℕ × ℕ)) := { (0, 1) }

/-- Prove the combination of conditions over which sets are in which other sets--/
theorem proof_problem :
  ¬ (setA ⊆ setB) ∧
  (setB = setC) ∧
  (empty ∈ singletonEmpty) ∧
  ¬ (setAB = setD) :=
by
  sorry

end proof_problem_l702_702010


namespace lambda_geq_44_l702_702271

open Real

noncomputable def manhattan_distance (P₁ P₂ : ℝ × ℝ) : ℝ :=
  |P₁.1 - P₂.1| + |P₁.2 - P₂.2|

theorem lambda_geq_44 :
  ∀ (points : Fin 2023 → ℝ × ℝ)
    (h_distinct : Function.Injective points)
    (h_noncollinear : ∀ {i j : Fin 2023},
      i ≠ j → manhattan_distance (points i) (points j) ≠ 0),
  let distances := Finset.univ.image (λ p : Fin 2023 × Fin 2023,
      if p.1 ≠ p.2 then some (manhattan_distance (points p.1) (points p.2)) else none) in
  let min_distance := distances.min' (sorry : distances.Nonempty) in
  let max_distance := distances.max' (sorry : distances.Nonempty) in
  max_distance / min_distance ≥ 44 :=
sorry

end lambda_geq_44_l702_702271


namespace sin2x_cos2x_value_l702_702918

theorem sin2x_cos2x_value (x y : ℝ) (h₁ : sin x / sin y = 4) (h₂ : cos x / cos y = 1 / 3) :
  sin (2 * x) / sin (2 * y) + cos (2 * x) / cos (2 * y) = 61 / 129 :=
by
  sorry

end sin2x_cos2x_value_l702_702918


namespace count_numeral_nines_in_1_to_100_l702_702428

def count_numeral_nines (n : Nat) : Nat :=
  let num_list := List.range (n + 1)
  num_list.foldl (fun acc num => acc + (String.toList (num.repr)).count (fun ch => ch = '9')) 0

theorem count_numeral_nines_in_1_to_100 : count_numeral_nines 100 = 20 := 
by
  -- Prepare the proof here, using the given conditions and correct answer
  sorry

end count_numeral_nines_in_1_to_100_l702_702428


namespace exists_linear_function_second_quadrant_l702_702945

theorem exists_linear_function_second_quadrant (k b : ℝ) (h1 : k > 0) (h2 : b > 0) :
  ∃ (f : ℝ → ℝ), (∀ x, f x = k * x + b) ∧ (∀ x, x < 0 → f x > 0) :=
by
  -- Prove there exists a linear function of the form f(x) = kx + b with given conditions
  -- Skip the proof for now
  sorry

end exists_linear_function_second_quadrant_l702_702945


namespace exists_k_for_A_mul_v_eq_k_mul_v_l702_702112

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 6, 3]

theorem exists_k_for_A_mul_v_eq_k_mul_v (v : Fin 2 → ℝ) (h : v ≠ 0) :
  (∃ k : ℝ, A.mul_vec v = k • v) →
  k = 3 + 2 * real.sqrt 6 ∨ k = 3 - 2 * real.sqrt 6 :=
by
  sorry

end exists_k_for_A_mul_v_eq_k_mul_v_l702_702112


namespace find_a_b_find_extremum_of_g_l702_702169

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a*x^2 + b*x
noncomputable def f' (x : ℝ) (a b : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Proving values of a and b
theorem find_a_b (a b : ℝ) (H1 : f'(1, a, b) = 0) (H2 : f'(-1, a, b) = 0) :
    a = 0 ∧ b = -3 := 
sorry

noncomputable def g' (x : ℝ) (a b : ℝ) : ℝ := f x a b + 2

-- Proving the extremum point of g
theorem find_extremum_of_g (a b : ℝ) (h_a : a = 0) (h_b : b = -3) :
    ∃ x : ℝ, g' x a b = 0 ∧ (∀ y, g' y a b < 0 → y < x) ∧ (∀ z, g' z a b > 0 → z > x) :=
 sorry

end find_a_b_find_extremum_of_g_l702_702169


namespace vector_sum_magnitude_eq_2_or_5_l702_702850

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := 1
noncomputable def c : ℝ := 3
def equal_angles (θ : ℝ) := θ = 120 ∨ θ = 0

theorem vector_sum_magnitude_eq_2_or_5
  (a_mag : ℝ := a)
  (b_mag : ℝ := b)
  (c_mag : ℝ := c)
  (θ : ℝ)
  (Hθ : equal_angles θ) :
  (|a_mag| = 1) ∧ (|b_mag| = 1) ∧ (|c_mag| = 3) →
  (|a_mag + b_mag + c_mag| = 2 ∨ |a_mag + b_mag + c_mag| = 5) :=
by
  sorry

end vector_sum_magnitude_eq_2_or_5_l702_702850


namespace probability_digit_three_in_repeating_block_l702_702622

theorem probability_digit_three_in_repeating_block :
  let repeating_block := "615384" in
  let num_threes := repeating_block.to_list.filter (λ digit => digit = '3').length in
  let block_length := repeating_block.to_list.length in
  num_threes / block_length = (1 : ℚ) / 6 :=
by
  sorry

end probability_digit_three_in_repeating_block_l702_702622


namespace minimum_value_of_g_squared_minus_3f_l702_702848

noncomputable def f (x : ℝ) : ℝ := a * x + b
noncomputable def g (x : ℝ) : ℝ := a * x + c

theorem minimum_value_of_g_squared_minus_3f : 
  ∀ (a b c : ℝ), 
  (∃ x : ℝ, (f x)^2 - 3 * g x = 11 / 2) → 
  (∃ y : ℝ, (g y)^2 - 3 * f y = -10) :=
by sorry

end minimum_value_of_g_squared_minus_3f_l702_702848


namespace round_4_65_to_nearest_tenth_l702_702632

theorem round_4_65_to_nearest_tenth : 
  (round (nearest (10^(-1))) 4.65) = 4.7 := 
by 
  sorry

end round_4_65_to_nearest_tenth_l702_702632


namespace find_f_and_max_profit_l702_702435

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a / (x - 4)) + 10 * (x - 7) ^ 2

def a_value (f : ℝ → ℝ) : ℝ := 10

def f_formula (x : ℝ) : ℝ := (10 / (x - 4)) + 10 * (x - 7) ^ 2

theorem find_f_and_max_profit:
  (∀ x, (4 < x ∧ x < 7) → ∃ a, f a x = f_formula x)
  ∧ ((∃ x, (4 < x ∧ x < 7) ∧ ∀ y, (4 < y ∧ y < 7) → (x ≠ y → h x > h y))
  ∧ (∃ c, (x = 4) ∧ (h x c = 50)
  :=
by {
  sorry
}

end find_f_and_max_profit_l702_702435


namespace exists_function_f_l702_702263

theorem exists_function_f (f : ℤ → ℤ) 
  (h1 : ∀ n : ℤ, f(f(n)) = n^2) 
  : ∃ f : ℤ → ℤ, ∀ n : ℤ, f(f(n)) = n^2 :=
sorry

end exists_function_f_l702_702263


namespace circle_passing_through_MAB_tangent_slope_at_P_circle_with_diameter_AB_passes_through_M_l702_702528

-- Define the parabola C: x^2 = 4y
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

-- Define the line l: y = -1
def line (y : ℝ) : Prop := y = -1

-- Define a point M on the line l
def point_on_line (M : ℝ × ℝ) : Prop := line M.snd

-- Define tangents MA and MB to the parabola through M
def tangent (M A : ℝ × ℝ) : Prop := parabola A.fst A.snd ∧ ∃ k : ℝ, line (k * M.fst - 1)

-- Proof statements

-- (1) Prove the equation of the circle passing through M, A, and B when M = (0, -1)
theorem circle_passing_through_MAB (A B : ℝ × ℝ) (M : ℝ × ℝ) (hM : M = (0, -1)) :
  tangent M A ∧ tangent M B → ∀ x y : ℝ, (x^2 + (y-1)^2 = 4) :=
sorry

-- (2) Prove the slope of the tangent line at point P(x0, y0) is k = 1/2 * x0, for any point P on C
theorem tangent_slope_at_P (P : ℝ × ℝ) (hP : parabola P.fst P.snd) :
  ∃ k : ℝ, k = 1 / 2 * P.fst :=
sorry

-- (3) Prove that the circle with diameter AB always passes through point M
theorem circle_with_diameter_AB_passes_through_M (A B M : ℝ × ℝ) :
  tangent M A ∧ tangent M B → ∀ x y : ℝ, 
  (x - M.fst)^2 + (y - M.snd)^2 = 
  ((A.fst + B.fst) / 2 - M.fst)^2 + ((A.snd + B.snd) / 2 - M.snd)^2 → 
  circle_passing_through_MAB A B M sorry :=
sorry

end circle_passing_through_MAB_tangent_slope_at_P_circle_with_diameter_AB_passes_through_M_l702_702528


namespace find_x_values_l702_702584

noncomputable def integral_of_abs_diff (a b : ℝ) : ℝ :=
∫ t in 0..a, |t - b|

theorem find_x_values (c : ℝ) :
  {x : ℝ | integral_of_abs_diff c x = integral_of_abs_diff x c} = {c / 3, c, 3 * c} := sorry

end find_x_values_l702_702584


namespace smallest_k_l702_702705

variable (n : ℕ) (M : Set ℕ) 

noncomputable def contains_subset_with_sum (A : Set ℕ) :=
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b + c + d = 4 * n + 1

theorem smallest_k (h : ∀ A ⊆ M, Set.card A = k → contains_subset_with_sum n A) :
  k = n + 3 :=
sorry

end smallest_k_l702_702705


namespace coordinates_of_point_l702_702246

theorem coordinates_of_point : 
  ∀ (x y : ℝ), (x, y) = (2, -3) → (x, y) = (2, -3) := 
by 
  intros x y h 
  exact h

end coordinates_of_point_l702_702246


namespace function_symmetric_about_origin_l702_702336

theorem function_symmetric_about_origin (x : ℝ) : 
  let f := λ x : ℝ, 2 * x ^ 3
  in f (-x) = -f x :=
by
  let f := λ x : ℝ, 2 * x ^ 3
  sorry

end function_symmetric_about_origin_l702_702336


namespace geometric_sequence_sum_l702_702650

variable {a : ℕ → ℕ}

-- Defining the geometric sequence and the conditions
def is_geometric_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def condition1 (a : ℕ → ℕ) : Prop :=
  a 1 = 3

def condition2 (a : ℕ → ℕ) : Prop :=
  a 1 + a 3 + a 5 = 21

-- The main theorem
theorem geometric_sequence_sum (a : ℕ → ℕ) (q : ℕ) 
  (h1 : condition1 a) (h2: condition2 a) (hq : is_geometric_sequence a q) : 
  a 3 + a 5 + a 7 = 42 := 
sorry

end geometric_sequence_sum_l702_702650


namespace factorization_correct_l702_702006

theorem factorization_correct :
  ∀ (x y : ℝ), 
    (¬ ( (y - 1) * (y + 1) = y^2 - 1 ) ) ∧
    (¬ ( x^2 * y + x * y^2 - 1 = x * y * (x + y) - 1 ) ) ∧
    (¬ ( (x - 2) * (x - 3) = (3 - x) * (2 - x) ) ) ∧
    ( x^2 - 4 * x + 4 = (x - 2)^2 ) :=
by
  intros x y
  repeat { constructor }
  all_goals { sorry }

end factorization_correct_l702_702006


namespace number_of_subsets_l702_702656

theorem number_of_subsets (x y : Type) :  ∃ s : Finset (Finset Type), s.card = 4 := 
sorry

end number_of_subsets_l702_702656


namespace find_x_when_z_64_l702_702383

-- Defining the conditions
def directly_proportional (x y : ℝ) : Prop := ∃ m : ℝ, x = m * y^3
def inversely_proportional (y z : ℝ) : Prop := ∃ n : ℝ, y = n / z^2

theorem find_x_when_z_64 (x y z : ℝ) (m n : ℝ) (k : ℝ) (h1 : directly_proportional x y) 
    (h2 : inversely_proportional y z) (h3 : z = 64) (h4 : x = 8) (h5 : z = 16) : x = 1/256 := 
  sorry

end find_x_when_z_64_l702_702383


namespace min_value_of_fraction_sum_l702_702810

noncomputable def min_fraction_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 * x + y = 1) : ℝ :=
  1 / x + 4 / y

theorem min_value_of_fraction_sum : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 4 * x + y = 1 ∧ min_fraction_sum x y (by assumption) (by assumption) (by assumption) = 16 :=
sorry

end min_value_of_fraction_sum_l702_702810


namespace triangle_angle_C_triangle_min_prod_ab_l702_702231

theorem triangle_angle_C 
  {a b c : ℝ} 
  (h₁: 2 * c * (cos B) = 2 * a + b)
  (h₂: S = sqrt 3 * (a + b)) :
  C = 2 * pi / 3 :=
  sorry

theorem triangle_min_prod_ab 
  {a b c : ℝ}
  (h₁: 2 * c * (cos B) = 2 * a + b)
  (h₂: S = sqrt 3 * (a + b)) :
  ∃ ab_min, ab_min = 64 ∧ ab_min ≤ a * b :=
  sorry

end triangle_angle_C_triangle_min_prod_ab_l702_702231


namespace ratio_of_triangles_l702_702594

/-- Define vertices for a tetrahedron -/
variables {A B C D T : Point}

/-- Assume tetrahedron property -/
axiom tetrahedron : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

/-- Define intersection line and point -/
axiom intersection_line : ∃ f_a : Line, 
  is_intersection_line f_a (plane_through A) (plane_through B) (plane_through C) T

axiom intersection_point : T ∈ face B C D ∧ T ∈ f_a

/-- Area function for triangles -/
noncomputable def area (u v w : Point) : ℝ := sorry

/-- Theorem statement that matches the areas ratio -/
theorem ratio_of_triangles (htbc : triangle T B C) (htcd : triangle T C D) (htdb : triangle T D B)
  (habc : triangle A B C) (hacd : triangle A C D) (hadb : triangle A D B) :
  (area T B C) / (area A B C) = (area T C D) / (area A C D) ∧
  (area T C D) / (area A C D) = (area T D B) / (area A D B) :=
sorry

end ratio_of_triangles_l702_702594


namespace scientific_notation_of_100000_l702_702874

theorem scientific_notation_of_100000 :
  100000 = 1 * 10^5 :=
by sorry

end scientific_notation_of_100000_l702_702874


namespace nearest_integer_sum_roots_l702_702816

noncomputable def f (x : ℝ) : ℝ := sorry

theorem nearest_integer_sum_roots (f : ℝ → ℝ)
  (H1 : ∀ x : ℝ, x ≠ 0 → 2 * f x + f (1 / x) = 6 * x + 3)
  (H2 : ∀ x : ℝ, x ≠ 0 → f x = 2023) :
  let T := ∑ x in {x : ℝ | x ≠ 0 ∧ f x = 2023}, x in 
  |⌊ T + 0.5 ⌋| = 506 := 
by 
  have : false := sorry
  sorry

end nearest_integer_sum_roots_l702_702816


namespace total_tickets_needed_l702_702307

-- Definitions representing the conditions
def rides_go_karts : ℕ := 1
def cost_per_go_kart_ride : ℕ := 4
def rides_bumper_cars : ℕ := 4
def cost_per_bumper_car_ride : ℕ := 5

-- Calculate the total tickets needed
def total_tickets : ℕ := rides_go_karts * cost_per_go_kart_ride + rides_bumper_cars * cost_per_bumper_car_ride

-- The theorem stating the main proof problem
theorem total_tickets_needed : total_tickets = 24 := by
  -- Proof steps should go here, but we use sorry to skip the proof
  sorry

end total_tickets_needed_l702_702307


namespace first_cosine_theorem_for_trihedral_second_cosine_theorem_for_trihedral_l702_702396

variables {α β γ A B C : ℝ}

-- Given conditions
-- α, β, and γ are the plane angles of a trihedral angle
-- A, B, and C are the corresponding dihedral angles

-- Part (a): First cosine theorem for a trihedral angle
theorem first_cosine_theorem_for_trihedral 
  (h1 : (cos α) = (cos β) * (cos γ) + (sin β) * (sin γ) * (cos A)) : 
  cos(α) = cos(β) * cos(γ) + sin(β) * sin(γ) * cos(A) := 
begin
  sorry
end

-- Part (b): Second cosine theorem for a trihedral angle
theorem second_cosine_theorem_for_trihedral 
  (h2 : (cos A) = - (cos B) * (cos C) + (sin B) * (sin C) * (cos α)) : 
  cos(A) = - cos(B) * cos(C) + sin(B) * sin(C) * cos(α) := 
begin
  sorry
end

end first_cosine_theorem_for_trihedral_second_cosine_theorem_for_trihedral_l702_702396


namespace log_sum_tan_degrees_l702_702797

theorem log_sum_tan_degrees : 
  (∑ i in finset.range 44, real.log10 (real.tan (real.pi * (i + 1) / 180))) = 0 := 
by sorry

end log_sum_tan_degrees_l702_702797


namespace sarah_total_weeds_l702_702963

theorem sarah_total_weeds :
  let tuesday_weeds := 25
  let wednesday_weeds := 3 * tuesday_weeds
  let thursday_weeds := wednesday_weeds / 5
  let friday_weeds := thursday_weeds - 10
  tuesday_weeds + wednesday_weeds + thursday_weeds + friday_weeds = 120 :=
by
  intros
  let tuesday_weeds := 25
  let wednesday_weeds := 3 * tuesday_weeds
  let thursday_weeds := wednesday_weeds / 5
  let friday_weeds := thursday_weeds - 10
  sorry

end sarah_total_weeds_l702_702963


namespace eigenvalues_of_2x2_matrix_l702_702105

theorem eigenvalues_of_2x2_matrix :
  ∃ (k : ℝ), (k = 3 + 4 * Real.sqrt 6 ∨ k = 3 - 4 * Real.sqrt 6) ∧
            ∃ (v : ℝ × ℝ), v ≠ (0, 0) ∧
            ((3 : ℝ) * v.1 + 4 * v.2 = k * v.1 ∧ (6 : ℝ) * v.1 + 3 * v.2 = k * v.2) :=
begin
  sorry
end

end eigenvalues_of_2x2_matrix_l702_702105


namespace truncated_polyhedron_edges_and_vertices_l702_702890

theorem truncated_polyhedron_edges_and_vertices (polyhedron : Type) [convex_polyhedron polyhedron] 
  (h : edges polyhedron = 100) : 
  vertices (truncate polyhedron) = 200 ∧ edges (truncate polyhedron) = 300 := 
by
  sorry

end truncated_polyhedron_edges_and_vertices_l702_702890


namespace sin_cos_ratio_l702_702919

namespace TrigProof

theorem sin_cos_ratio (x y : ℝ) (h1 : sin x / sin y = 4) (h2 : cos x / cos y = 1 / 3) :
  (sin (2 * x) / sin (2 * y)) + (cos (2 * x) / cos (2 * y)) = -29 / 3 := by
  sorry

end TrigProof

end sin_cos_ratio_l702_702919


namespace equal_volume_partition_of_regular_tetrahedron_equal_volume_partition_of_tetrahedron_l702_702690

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Tetrahedron where
  A : Point
  B : Point
  C : Point
  D : Point

def midpoint (P Q : Point) : Point :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2, (P.z + Q.z) / 2⟩

def Plane (M N : Point) : Prop := sorry -- Define the plane passing through points M and N

def is_regular (T : Tetrahedron) : Prop := sorry -- Define a regular tetrahedron property

theorem equal_volume_partition_of_regular_tetrahedron (T : Tetrahedron) (hR : is_regular T) :
  ∃ M N : Point, (∃ P Q : Point, (P ≠ Q) ∧ (P, Q ∈ {T.A, T.B, T.C, T.D}) ∧ (M = midpoint P Q) 
  ∧ ∃ R S : Point, (R ≠ S) ∧ (R, S ∈ {T.A, T.B, T.C, T.D}) ∧ (N = midpoint R S) ∧ (Plane M N)) 
  ∧ (Volume_partition T M N):
  sorry

theorem equal_volume_partition_of_tetrahedron (T : Tetrahedron) :
  ∃ M N : Point, (∃ P Q : Point, (P ≠ Q) ∧ (P, Q ∈ {T.A, T.B, T.C, T.D}) ∧ (M = midpoint P Q) 
  ∧ ∃ R S : Point, (R ≠ S) ∧ (R, S ∈ {T.A, T.B, T.C, T.D}) ∧ (N = midpoint R S) ∧ (Plane M N)) 
  ∧ (Volume_partition T M N):
  sorry

end equal_volume_partition_of_regular_tetrahedron_equal_volume_partition_of_tetrahedron_l702_702690


namespace row_col_sum_equality_l702_702242

-- Definitions based on the conditions given in the problem
def row_condition (M : Matrix (Fin n) (Fin n) ℝ) (a : ℝ) : Prop :=
  ∀ i : Fin n, let row := λ j, M i j in
    let ⟨x, y⟩ := max_two_sums row in
      x + y = a

def col_condition (M : Matrix (Fin n) (Fin n) ℝ) (b : ℝ) : Prop :=
  ∀ j : Fin n, let col := λ i, M i j in
    let ⟨x, y⟩ := max_two_sums col in
      x + y = b

-- Function to get the two largest sums in a sequence (generalized for both row and column conditions)
def max_two_sums {α : Type*} [LinearOrder α] (seq : Fin n → α) : α × α :=
  let sorted := seq.to_list.sort (· ≤ ·) in
    (sorted.reverse.nth_le 0 sorry, sorted.reverse.nth_le 1 sorry)

theorem row_col_sum_equality (M : Matrix (Fin n) (Fin n) ℝ) (a b : ℝ) :
  row_condition M a → col_condition M b → a = b := sorry

end row_col_sum_equality_l702_702242


namespace needed_fraction_to_make_99th_digit_four_l702_702371

-- Given the conditions
def repeating_sequence_of_three_elevens : ℕ → ℝ
| n := if n % 2 = 0 then 2 else 7

def decimal_digit (r : ℝ) (n : ℕ) : ℕ := sorry -- define a function that calculates the nth digit after the decimal point of r

-- Main statement
theorem needed_fraction_to_make_99th_digit_four (x : ℝ) :
    decimal_digit ((3 / 11) + x) 99 = 4 := 
sorry

end needed_fraction_to_make_99th_digit_four_l702_702371


namespace ratio_divide_area_l702_702531

variables {a b c d z : ℝ}

-- Define the conditions
def is_trapezoid (ABCD : Quadrilateral) : Prop :=
  ABCD.bc ∥ ABCD.ad

def line_parallel (XY BC AD : Segment) : Prop :=
  XY ∥ BC ∧ XY ∥ AD

def equal_perimeters (AXYD XBCY : Trapezoid) : Prop :=
  AX + AY + YD + DA = XB + BC + CY + YX

-- Objective to prove the ratio
theorem ratio_divide_area (ABCD : Trapezoid) (XY : Segment) (AB = a)  (BC = b) (CD = c) (DA = d) (XY = z) :
  ∀ (XY BC AD : Segment), 
  is_trapezoid ABCD →
  line_parallel XY BC AD →
  equal_perimeters (AXYD) (XBCY) →
  z = (d + b) / 2 + (d - b)^2 / (2 * (a + c)) :=
sorry

end ratio_divide_area_l702_702531


namespace area_smaller_part_l702_702735

theorem area_smaller_part (A B : ℝ) (h₁ : A + B = 500) (h₂ : B - A = (A + B) / 10) : A = 225 :=
by sorry

end area_smaller_part_l702_702735


namespace coordinates_with_respect_to_origin_l702_702252

theorem coordinates_with_respect_to_origin (P : ℝ × ℝ) (h : P = (2, -3)) : P = (2, -3) :=
by
  sorry

end coordinates_with_respect_to_origin_l702_702252


namespace num_valid_tuples_l702_702503

noncomputable def count_valid_tuples : ℕ :=
  let N := 9
  let sum := 100
  let valid_tuple (a : Fin N → ℕ) : Prop :=
    ∀ (i j k : Fin N), i < j → j < k → (∃ l : Fin N, l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ a i + a j + a k + a l = sum)
  ∑ t in Finset.univ.filter (valid_tuple), 1

theorem num_valid_tuples : count_valid_tuples = 2017 := sorry

end num_valid_tuples_l702_702503


namespace parabola_equation_and_minimum_hk_l702_702197

def parabola (p : ℝ) (p_pos : p > 0) := ∀ x y : ℝ, y^2 = 2 * p * x

def line (a : ℝ) := ∀ x : ℝ, x = a

def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) := 
    (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem parabola_equation_and_minimum_hk
  (p : ℝ) (p_pos : p > 0) 
  (M N : ℝ × ℝ)
  (intersect_M : M.1 = 4 ∧ M.2 = sqrt (2 * p * 4))
  (intersect_N : N.1 = 4 ∧ N.2 = - sqrt (2 * p * 4))
  (O : ℝ × ℝ := (0, 0))
  (area_OMN : triangle_area O.1 O.2 M.1 M.2 N.1 N.2 = 8 * sqrt 6) :

  (∀ x y : ℝ, y^2 = 6 * x) ∧
  let E := λ a : ℝ, (a > 0) → 
    ∀ (A B C D : ℝ × ℝ)
    (H : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
    (K : ℝ × ℝ := ((C.1 + D.1) / 2, (C.2 + D.2) / 2))
    (perpendicular_AB_CD : ((A.2 - B.2) / (A.1 - B.1)) * ((C.2 - D.2) / (C.1 - D.1)) = -1)  
    (parabola_points : parabola p p_pos) 
  , min_value (dist H K) = 6 
 := sorry

end parabola_equation_and_minimum_hk_l702_702197


namespace businesses_brandon_can_apply_to_l702_702933

-- Definitions of the given conditions in the problem
variables (x y : ℕ)

-- Define the total, fired, and quit businesses
def total_businesses : ℕ := 72
def fired_businesses : ℕ := 36
def quit_businesses : ℕ := 24

-- Define the unique businesses Brandon can still apply to, considering common businesses and reapplications
def businesses_can_apply_to : ℕ := (12 + x) + y

-- The theorem to prove
theorem businesses_brandon_can_apply_to (x y : ℕ) : businesses_can_apply_to x y = 12 + x + y := by
  unfold businesses_can_apply_to
  sorry

end businesses_brandon_can_apply_to_l702_702933


namespace triangle_ratio_l702_702577

theorem triangle_ratio (
  (ABC : Triangle)
  (A' B' E : Point)
  (A'_line : PointOnSegment A' CA)
  (B'_line : PointOnSegment B' CB)
  (AA'_eq_BB' : dist A A' = dist B B')
  (A'B'_intersect_AB : IntersectsLine A'B' AB at E)
) :
  dist A' E / dist E B' = dist C B / dist C A := 
sorry

end triangle_ratio_l702_702577


namespace find_initial_period_l702_702659

theorem find_initial_period (P : ℝ) (T : ℝ) 
  (h1 : 1680 = (P * 4 * T) / 100)
  (h2 : 1680 = (P * 5 * 4) / 100) 
  : T = 5 := 
by 
  sorry

end find_initial_period_l702_702659


namespace greatest_four_digit_n_l702_702680

theorem greatest_four_digit_n :
  ∃ (n : ℕ), (1000 ≤ n ∧ n ≤ 9999) ∧ (∃ m : ℕ, n + 1 = m^2) ∧ ¬(n! % (n * (n + 1) / 2) = 0) ∧ n = 9999 :=
by sorry

end greatest_four_digit_n_l702_702680


namespace find_m_n_sum_l702_702050

def P : ℕ × ℕ → ℚ
| (0, 0)       := 1
| (x+1, 0)     := 0
| (0, y+1)     := 0
| (x+1, y+1)   := (1 / 3) * P (x, y+1) + (1 / 3) * P (x+1, y) + (1 / 3) * P (x, y)

theorem find_m_n_sum : ∃ m n : ℕ, P (5, 5) = m / 3^n ∧ m % 3 ≠ 0 ∧ m + n = 6 := sorry

end find_m_n_sum_l702_702050


namespace max_value_of_sin_l702_702375

theorem max_value_of_sin (x : ℝ) : (2 * Real.sin x) ≤ 2 :=
by
  -- this theorem directly implies that 2sin(x) has a maximum value of 2.
  sorry

end max_value_of_sin_l702_702375


namespace express_x_in_terms_of_y_l702_702137

theorem express_x_in_terms_of_y (x y : ℝ) (h : 3 * x - 4 * y = 8) : x = (4 * y + 8) / 3 :=
sorry

end express_x_in_terms_of_y_l702_702137


namespace sharon_distance_l702_702097

noncomputable def usual_speed (x : ℝ) := x / 180
noncomputable def reduced_speed (x : ℝ) := (x / 180) - 0.5

theorem sharon_distance (x : ℝ) (h : 300 = (x / 2) / usual_speed x + (x / 2) / reduced_speed x) : x = 157.5 :=
by sorry

end sharon_distance_l702_702097


namespace range_of_omega_l702_702186

theorem range_of_omega {ω : ℝ} (hω : ω > 0) :
  (∀ x ∈ set.Icc 0 π, 2 * sin (ω * x + π / 4) = 0 ↔ x = 0 ∨ x = (4 * π) / ω) ↔ 
  ω ∈ set.Ico (11 / 4 : ℝ) (15 / 4 : ℝ) :=
sorry

end range_of_omega_l702_702186


namespace gcd_decomposition_l702_702813

open Polynomial

noncomputable def f : Polynomial ℚ := 4 * X ^ 4 - 2 * X ^ 3 - 16 * X ^ 2 + 5 * X + 9
noncomputable def g : Polynomial ℚ := 2 * X ^ 3 - X ^ 2 - 5 * X + 4

theorem gcd_decomposition :
  ∃ (u v : Polynomial ℚ), u * f + v * g = X - 1 :=
sorry

end gcd_decomposition_l702_702813


namespace greatest_integer_not_exceeding_l702_702748

noncomputable def shadow_cube : ℝ :=
  let e := 2 in -- edge length of cube in cm
  let a_cube := e * e in -- area of cube's base
  let a_shadow := 200 in -- shadow area excluding area beneath the cube
  let a_total_shadow := a_cube + a_shadow in -- total shadow area
  let s := real.sqrt a_total_shadow in -- side length of shadow square
  let x := s / e in -- light height using similar triangles
  1000 * x -- scaling the height by 1000

theorem greatest_integer_not_exceeding : Int.floor shadow_cube = 14282 := sorry

end greatest_integer_not_exceeding_l702_702748


namespace quad_with_four_axes_of_symmetry_is_square_l702_702947

theorem quad_with_four_axes_of_symmetry_is_square
  (Q : Type) [quadrilateral Q]
  (h : has_four_axes_of_symmetry Q) :
  is_square Q :=
sorry

end quad_with_four_axes_of_symmetry_is_square_l702_702947


namespace smallest_triangle_perimeter_l702_702377

theorem smallest_triangle_perimeter : ∃ (a b c : ℕ), a = 3 ∧ b = a + 1 ∧ c = b + 1 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a + b + c = 12 := by
  sorry

end smallest_triangle_perimeter_l702_702377


namespace JillSavingsPercentage_l702_702794

noncomputable def JillNetMonthlySalary : ℝ := 3600
noncomputable def JillDiscretionaryIncome : ℝ := JillNetMonthlySalary / 5
noncomputable def VacationFundPercentage : ℝ := 0.30
noncomputable def EatingOutSocializingPercentage : ℝ := 0.35
noncomputable def GiftsCharityExpense : ℝ := 108

theorem JillSavingsPercentage :
  let discretionaryIncome := JillDiscretionaryIncome in
  let expenses := (VacationFundPercentage * discretionaryIncome) + (EatingOutSocializingPercentage * discretionaryIncome) + GiftsCharityExpense in
  let savings := discretionaryIncome - expenses in
  let savingsPercentage := (savings / discretionaryIncome) * 100 in
  savingsPercentage = 20 :=
by
  let discretionaryIncome := JillDiscretionaryIncome
  let expenses := (VacationFundPercentage * discretionaryIncome) + (EatingOutSocializingPercentage * discretionaryIncome) + GiftsCharityExpense
  let savings := discretionaryIncome - expenses
  let savingsPercentage := (savings / discretionaryIncome) * 100
  suffices h : savingsPercentage = 20 by exact h
  sorry

end JillSavingsPercentage_l702_702794


namespace general_formula_an_sum_bn_l702_702148

-- Define the sequences a_n and b_n
def a (n : ℕ) : ℕ := 4^(n-1)
def b (n : ℕ) : ℕ := 3*n - 2

-- Define sum of first n terms S_n and T_n
def Sn (n : ℕ) : ℕ := (List.range n).sum (λ i => a (i+1))
def Tn (n : ℕ) : ℕ := (List.range n).sum (λ i => b (i+1))

-- Conditions
axiom a1 : a 1 = 1
axiom aSn : ∀ n, 3 * Sn n = a (n + 1) - 1
axiom a2_eq_b2 : a 2 = b 2
axiom T4_eq_1_plus_S3 : Tn 4 = 1 + Sn 3

-- Goal: Prove general formula and the sum for b_n
theorem general_formula_an : ∀ n, a n = 4^(n-1) := by {
  sorry -- proof omitted
}

theorem sum_bn : (List.range 10).sum (λ n => 1 / (b n * b (n + 1))) = 10 / 31 := by {
  sorry -- proof omitted
}

end general_formula_an_sum_bn_l702_702148


namespace range_of_a_l702_702225

-- Definitions as per the conditions
def f (a : ℝ) (x : ℝ) : ℝ := 
  Real.log x + a * x + 1 / x

def derivative_f (a : ℝ) (x : ℝ) : ℝ :=
  1 / x + a - 1 / x ^ 2

def is_monotonically_increasing_on (f : ℝ -> ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x ≤ y → f x ≤ f y

-- The proof statement
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x → derivative_f a x ≥ 0) ↔ (a ∈ set.Iic (-1/4) ∪ set.Ici 0) :=
sorry

end range_of_a_l702_702225


namespace bike_cost_l702_702080

theorem bike_cost (h1: 8 > 0) (h2: 35 > 0) (weeks_in_month: ℕ := 4) (saved: ℕ := 720):
  let hourly_wage := 8
  let weekly_hours := 35
  let weekly_earnings := weekly_hours * hourly_wage
  let monthly_earnings := weekly_earnings * weeks_in_month
  let cost_of_bike := monthly_earnings - saved
  cost_of_bike = 400 :=
by
  sorry

end bike_cost_l702_702080


namespace volcano_explosion_percentage_l702_702048

theorem volcano_explosion_percentage :
  let initial_volcanoes := 200
  let first_two_months_explosions := initial_volcanoes * 20 / 100
  let after_first_two_months := initial_volcanoes - first_two_months_explosions
  let midyear_explosions := after_first_two_months * 40 / 100
  let after_midyear := after_first_two_months - midyear_explosions
  let final_intact := 48
  let endyear_explosions := after_midyear - final_intact
  let percentage_exploded_at_end := (endyear_explosions * 100) / after_midyear
  in percentage_exploded_at_end = 50 :=
by
  sorry

end volcano_explosion_percentage_l702_702048


namespace correct_conclusion_l702_702005

theorem correct_conclusion :
  ¬ (-(-3)^2 = 9) ∧
  ¬ (-6 / 6 * (1 / 6) = -6) ∧
  ((-3)^2 * abs (-1/3) = 3) ∧
  ¬ (3^2 / 2 = 9 / 4) :=
by
  sorry

end correct_conclusion_l702_702005


namespace simplify_log_expression_eq_2_trig_expression_eq_neg_sqrt3_plus1_div_3_l702_702713

noncomputable def simplify_log_expression : ℝ :=
  log 27 (1/3)^(1/2) + log 10 25 + log 10 4 + 7^(-log 7 2) + (-0.98)^0

theorem simplify_log_expression_eq_2 : simplify_log_expression = 2 :=
  by sorry

variables (α : ℝ) (P : ℝ × ℝ)
def point_on_terminal_side (α : ℝ) (P : ℝ × ℝ) : Prop :=
  P.1 = real.sqrt 2 ∧ P.2 = -real.sqrt 6

theorem trig_expression_eq_neg_sqrt3_plus1_div_3
  (h : point_on_terminal_side α (real.sqrt 2, -real.sqrt 6)) :
  (cos (π / 2 + α) * cos (2π - α) + sin (-α - π / 2) * cos (π - α)) / (sin (π + α) * cos (π / 2 - α))
  = - (real.sqrt 3 + 1) / 3 :=
  by sorry

end simplify_log_expression_eq_2_trig_expression_eq_neg_sqrt3_plus1_div_3_l702_702713


namespace square_diagonal_l702_702372

theorem square_diagonal (A : ℝ) (s : ℝ) (d : ℝ) (hA : A = 338) (hs : s^2 = A) (hd : d^2 = 2 * s^2) : d = 26 :=
by
  -- Proof goes here
  sorry

end square_diagonal_l702_702372


namespace set_of_values_l702_702214

theorem set_of_values (a : ℝ) (h : 2 ∉ {x : ℝ | x - a < 0}) : a ≤ 2 := 
sorry

end set_of_values_l702_702214


namespace collinear_A_B_D_collinear_B_D_F_k_l702_702912

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]

-- Conditions
variables (e1 e2 : V)
variables (h_non_collinear : ¬ collinear ({e1, e2} : set V))
variables (AB CB CD : V)
variables (h1 : AB = 2 • e1 - 8 • e2)
variables (h2 : CB = e1 + 3 • e2)
variables (h3 : CD = 2 • e1 - e2)

-- The first proof problem
theorem collinear_A_B_D (A B D : V) (h_AB : AB = 2 • (e1 - 4 • e2)) : collinear ({A, B, D} : set V) :=
sorry

-- Additional condition for the second part
variables (BF : V)
variables (h_BF : BF = 3 • e1 - k • e2)
variables (h_collinear_B_D_F : collinear ({B, D, F} : set V))

-- The second proof problem
theorem collinear_B_D_F_k (A B D F : V) (k : ℝ)
(hk : BF = 3 • e1 - k • e2) (h_AB_D : collinear ({A, B, D} : set V))
: k = 12 :=
sorry

end collinear_A_B_D_collinear_B_D_F_k_l702_702912


namespace area_of_triangle_PTU_l702_702629

open Real

noncomputable def length_diagonal (pq qr : ℝ) : ℝ :=
  sqrt (pq ^ 2 + qr ^ 2)
  
noncomputable def midpoint_length (diag : ℝ) : ℝ :=
  diag / 2

noncomputable def find_pu (pq : ℝ) : ℝ :=
  pq / 2

noncomputable def find_tu (pt pu : ℝ) : ℝ :=
  sqrt (pt ^ 2 - pu ^ 2)

noncomputable def area_triangle (base height : ℝ) : ℝ :=
  1 / 2 * base * height

theorem area_of_triangle_PTU :
  ∀ (PQ QR : ℝ), 
  PQ = 10 → QR = 24 →
  let PR := length_diagonal PQ QR,
      PT := midpoint_length PR,
      PU := find_pu PQ,
      TU := find_tu PT PU
  in area_triangle PU TU = 30 :=
by
  intros PQ QR hPQ hQR
  let PR := length_diagonal PQ QR
  let PT := midpoint_length PR
  let PU := find_pu PQ
  let TU := find_tu PT PU
  have hPQ: PQ = 10 := hPQ
  have hQR: QR = 24 := hQR
  sorry

end area_of_triangle_PTU_l702_702629


namespace problem_solution_l702_702740

def b : ℕ → ℕ
| 0       := 1  -- not given in the problem but harmless base case
| 1       := 2
| 2       := 4
| 3       := 7
| (n+4) := b (n+3) + b (n+2) + b (n+1)

lemma gcd_rel_prime (a b : ℕ): Nat.gcd a b = 1 ↔ ∀ d, d ∣ a ∧ d ∣ b → d = 1 :=
by sorry

theorem problem_solution : ∑ i in Finset.range 2, i ≠ 0 → (let m := 10609, n := 32768 in
  Nat.gcd m n = 1 ∧ m + n = 43377) :=
by
  sorry

end problem_solution_l702_702740


namespace ducks_killed_is_20_l702_702473

variable (x : ℕ)

def killed_ducks_per_year (x : ℕ) : Prop :=
  let initial_flock := 100
  let annual_births := 30
  let years := 5
  let additional_flock := 150
  let final_flock := 300
  initial_flock + years * (annual_births - x) + additional_flock = final_flock

theorem ducks_killed_is_20 : killed_ducks_per_year 20 :=
by
  sorry

end ducks_killed_is_20_l702_702473


namespace complex_conjugate_first_quadrant_l702_702172

theorem complex_conjugate_first_quadrant (z : ℂ) (h : z * complex.I = 2 + complex.I) :
    (z.conj.re > 0) ∧ (z.conj.im > 0) :=
sorry

end complex_conjugate_first_quadrant_l702_702172


namespace isosceles_triangle_perimeter_l702_702822

noncomputable def x : ℝ := 4
noncomputable def y : ℝ := 8

theorem isosceles_triangle_perimeter (x y : ℝ) (h : |x - 4| + (y - 8)^2 = 0)
        (hx : x = 4) (hy : y = 8) : x + y + y = 20 :=
by
  rw [hx, hy]
  norm_num
  sorry

end isosceles_triangle_perimeter_l702_702822


namespace eigenvalues_of_2x2_matrix_l702_702107

theorem eigenvalues_of_2x2_matrix :
  ∃ (k : ℝ), (k = 3 + 4 * Real.sqrt 6 ∨ k = 3 - 4 * Real.sqrt 6) ∧
            ∃ (v : ℝ × ℝ), v ≠ (0, 0) ∧
            ((3 : ℝ) * v.1 + 4 * v.2 = k * v.1 ∧ (6 : ℝ) * v.1 + 3 * v.2 = k * v.2) :=
begin
  sorry
end

end eigenvalues_of_2x2_matrix_l702_702107


namespace coefficient_x3_binomial_expansion_l702_702983

theorem coefficient_x3_binomial_expansion : 
  let coeff := ∑ r in (finset.range 10), (∑ x in (finset.range (r + 1)), ((-1) * (1 / 2)^x * x^r)) 
  in coeff = -15 :=
by 
  let term_3 := binom 10 3 * (-1/2)^3
  have coeff_x3 : term_3 = -15, by { sorry }
  exact coeff_x3

end coefficient_x3_binomial_expansion_l702_702983


namespace polynomial_multiplication_l702_702299

noncomputable def multiply_polynomials (a b : ℤ) :=
  let p1 := 3 * a ^ 4 - 7 * b ^ 3
  let p2 := 9 * a ^ 8 + 21 * a ^ 4 * b ^ 3 + 49 * b ^ 6 + 6 * a ^ 2 * b ^ 2
  let result := 27 * a ^ 12 + 18 * a ^ 6 * b ^ 2 - 42 * a ^ 2 * b ^ 5 - 343 * b ^ 9
  p1 * p2 = result

-- The main statement to prove
theorem polynomial_multiplication (a b : ℤ) : multiply_polynomials a b :=
by
  sorry

end polynomial_multiplication_l702_702299


namespace stephanie_falls_l702_702327

theorem stephanie_falls 
  (steven_falls : ℕ := 3)
  (sonya_falls : ℕ := 6)
  (h1 : sonya_falls = 6)
  (h2 : ∃ S : ℕ, sonya_falls = (S / 2) - 2 ∧ S > steven_falls) :
  ∃ S : ℕ, S - steven_falls = 13 :=
by
  sorry

end stephanie_falls_l702_702327


namespace total_cost_of_backpack_and_pencil_case_l702_702423

-- Definitions based on the given conditions
def pencil_case_price : ℕ := 8
def backpack_price : ℕ := 5 * pencil_case_price

-- Statement of the proof problem
theorem total_cost_of_backpack_and_pencil_case : 
  pencil_case_price + backpack_price = 48 :=
by
  -- Skip the proof
  sorry

end total_cost_of_backpack_and_pencil_case_l702_702423


namespace smaller_number_of_two_digits_product_3774_l702_702992

theorem smaller_number_of_two_digits_product_3774 (a b : ℕ) (ha : 9 < a ∧ a < 100) (hb : 9 < b ∧ b < 100) (h : a * b = 3774) : a = 51 ∨ b = 51 :=
by
  sorry

end smaller_number_of_two_digits_product_3774_l702_702992


namespace locus_of_midpoints_annulus_l702_702481

variable {S₁ S₂ : Type} [Circle S₁] [Circle S₂] {O₁ O₂ : Point}
variable {r₁ r₂ : Real} -- Radii of the circles S₁ and S₂

-- Assuming S₁ lies outside S₂ and both are non-intersecting
axiom S₁_outside_S₂ : ¬(S₁ ∩ S₂ ≠ ∅) ∧ ¬(O₂ ∈ S₁) ∧ (dist O₁ O₂ ≥ r₁ + r₂)

def locus_of_midpoints (M : Point) : Prop :=
  ∃ (A₁ ∈ S₁) (A₂ ∈ S₂), M = midpoint A₁ A₂

theorem locus_of_midpoints_annulus :
  ∀ (S₁ S₂ : Type) [Circle S₁] [Circle S₂] (O₁ O₂ : Point) (r₁ r₂ : Real),
    (¬(S₁ ∩ S₂ ≠ ∅) ∧ ¬(O₂ ∈ S₁) ∧ (dist O₁ O₂ ≥ r₁ + r₂)) →
    ∀ (M : Point),
      locus_of_midpoints M →
      ((dist (midpoint O₁ O₂) M = (r₁ - r₂) / 2) ∨ (dist (midpoint O₁ O₂) M = (r₁ + r₂) / 2)) :=
sorry

end locus_of_midpoints_annulus_l702_702481


namespace find_value_l702_702623

variable (N : ℝ)

def condition : Prop := (1 / 4) * (1 / 3) * (2 / 5) * N = 16

theorem find_value (h : condition N) : (1 / 3) * (2 / 5) * N = 64 :=
sorry

end find_value_l702_702623


namespace find_initial_percentage_l702_702031

def initial_alcohol_percentage (P : ℝ) : Prop :=
  let original_solution := 40
  let added_alcohol := 2.5
  let added_water := 7.5
  let final_solution := original_solution + added_alcohol + added_water
  let final_alcohol_percentage := 9
  let initial_alcohol_content := (P / 100) * original_solution
  let final_alcohol_content := (final_alcohol_percentage / 100) * final_solution
  in initial_alcohol_content + added_alcohol = final_alcohol_content

theorem find_initial_percentage (P : ℝ) : initial_alcohol_percentage P → P = 5 :=
by
  intros h
  sorry

end find_initial_percentage_l702_702031


namespace six_points_conic_l702_702820

theorem six_points_conic
  (A B C D E F G H I : Point)
  (on_AB_D : IsOnSegment A B D) (on_AB_E : IsOnSegment A B E)
  (on_BC_F : IsOnSegment B C F) (on_BC_G : IsOnSegment B C G)
  (on_CA_H : IsOnSegment C A H) (on_CA_I : IsOnSegment C A I)
  (segment_ratio : (ratio AD DB * ratio BG GC * ratio CI IA 
                    = ratio BE EA * ratio AH HC * ratio CF FB)) :
  ConicSection D E F G H I := 
by
  sorry

end six_points_conic_l702_702820


namespace seating_arrangements_l702_702666

theorem seating_arrangements : 
  ∃ (seats students : ℕ), seats = 7 ∧ students = 4 ∧ (∃ count : ℕ, count = 480 ∧
  let ⟨students_arrangements, gap_inserts⟩ := (factorial students, factorial (seats - students - 1)) in
  students_arrangements * gap_inserts = count) :=
sorry

end seating_arrangements_l702_702666


namespace probability_same_color_given_first_red_l702_702552

-- Definitions of events
def event_A (draw1 : ℕ) : Prop := draw1 = 1 -- Event A: the first ball drawn is red (drawing 1 means the first ball is red)

def event_B (draw1 draw2 : ℕ) : Prop := -- Event B: the two balls drawn are of the same color
  (draw1 = 1 ∧ draw2 = 1) ∨ (draw1 = 2 ∧ draw2 = 2)

-- Given probabilities
def P_A : ℚ := 2 / 5
def P_AB : ℚ := (2 / 5) * (1 / 4)

-- The conditional probability P(B|A)
def P_B_given_A : ℚ := P_AB / P_A

theorem probability_same_color_given_first_red : P_B_given_A = 1 / 4 := 
by 
  unfold P_B_given_A P_A P_AB
  sorry

end probability_same_color_given_first_red_l702_702552


namespace intersection_points_l702_702566

noncomputable def polar_intersection : Set (ℝ × ℝ) :=
  { (ρ, θ) |
    (∃ t, (2 + (1/2)*t = ρ * Real.cos θ ∧ (sqrt 3)/2 * t = ρ * Real.sin θ)) ∧
    (ρ = 4 * Real.cos θ) }

theorem intersection_points :
  polar_intersection = { (2, 5*π/3), (2 * sqrt 3, π/6) } :=
by
  sorry

end intersection_points_l702_702566


namespace arithmetic_sequence_n_15_l702_702253

theorem arithmetic_sequence_n_15 (a : ℕ → ℤ) (n : ℕ)
  (h₁ : a 3 = 5)
  (h₂ : a 2 + a 5 = 12)
  (h₃ : a n = 29) :
  n = 15 :=
sorry

end arithmetic_sequence_n_15_l702_702253


namespace fruit_problem_l702_702418

theorem fruit_problem
  (A B C : ℕ)
  (hA : A = 4) 
  (hB : B = 6) 
  (hC : C = 12) :
  ∃ x : ℕ, 1 = x / 2 := 
by
  sorry

end fruit_problem_l702_702418


namespace integer_satisfies_inequality_count_l702_702855

theorem integer_satisfies_inequality_count : 
  {m : ℤ | 1 ≤ |m| ∧ |m| ≤ 5}.card = 9 :=
by
  sorry

end integer_satisfies_inequality_count_l702_702855


namespace total_fish_correct_l702_702011

theorem total_fish_correct :
  (yesterday_fish : ℕ) (yesterday_payment today_payment : ℕ)
  (today_additional_payment : ℕ) 
  (h1 : yesterday_fish = 10) 
  (h2 : yesterday_payment = 3000) 
  (h3 : today_additional_payment = 6000) 
  (h4 : today_payment = yesterday_payment + today_additional_payment) :
  yesterday_fish + (today_payment / (yesterday_payment / yesterday_fish)) = 40 :=
by
  sorry

end total_fish_correct_l702_702011


namespace spherical_to_rectangular_coordinates_l702_702090

-- Definitions of the conditions
def ρ : ℝ := 10
def θ : ℝ := 5 * Real.pi / 4
def φ : ℝ := Real.pi / 4

-- The target rectangular coordinates to be proved
def x : ℝ := ρ * Real.sin φ * Real.cos θ
def y : ℝ := ρ * Real.sin φ * Real.sin θ
def z : ℝ := ρ * Real.cos φ

theorem spherical_to_rectangular_coordinates :
  (x, y, z) = (-5, -5, 5 * Real.sqrt 2) := by
  sorry

end spherical_to_rectangular_coordinates_l702_702090


namespace max_cut_length_l702_702029

theorem max_cut_length (board_size : ℕ) (total_pieces : ℕ) 
  (area_each : ℕ) 
  (total_area : ℕ)
  (total_perimeter : ℕ)
  (initial_perimeter : ℕ)
  (max_possible_length : ℕ)
  (h1 : board_size = 30) 
  (h2 : total_pieces = 225)
  (h3 : area_each = 4)
  (h4 : total_area = board_size * board_size)
  (h5 : total_perimeter = total_pieces * 10)
  (h6 : initial_perimeter = 4 * board_size)
  (h7 : max_possible_length = (total_perimeter - initial_perimeter) / 2) :
  max_possible_length = 1065 :=
by 
  -- Here, we do not include the proof as per the instructions
  sorry

end max_cut_length_l702_702029


namespace randy_biscuits_l702_702954

theorem randy_biscuits (F : ℕ) (initial_biscuits mother_biscuits brother_ate remaining_biscuits : ℕ) 
  (h_initial : initial_biscuits = 32)
  (h_mother : mother_biscuits = 15)
  (h_brother : brother_ate = 20)
  (h_remaining : remaining_biscuits = 40)
  : ((initial_biscuits + mother_biscuits + F) - brother_ate) = remaining_biscuits → F = 13 := 
by
  intros h_eq
  sorry

end randy_biscuits_l702_702954


namespace copper_production_is_correct_l702_702075

-- Define the percentages of copper production for each mine
def percentage_copper_mine_a : ℝ := 0.05
def percentage_copper_mine_b : ℝ := 0.10
def percentage_copper_mine_c : ℝ := 0.15

-- Define the daily production of each mine in tons
def daily_production_mine_a : ℕ := 3000
def daily_production_mine_b : ℕ := 4000
def daily_production_mine_c : ℕ := 3500

-- Define the total copper produced from all mines
def total_copper_produced : ℝ :=
  percentage_copper_mine_a * daily_production_mine_a +
  percentage_copper_mine_b * daily_production_mine_b +
  percentage_copper_mine_c * daily_production_mine_c

-- Prove that the total daily copper production is 1075 tons
theorem copper_production_is_correct :
  total_copper_produced = 1075 := 
sorry

end copper_production_is_correct_l702_702075


namespace find_f_neg_9_over_2_l702_702511

noncomputable def f : ℝ → ℝ
| x => if 0 ≤ x ∧ x ≤ 1 then 2^x else sorry

theorem find_f_neg_9_over_2
  (hf_even : ∀ x : ℝ, f (-x) = f x)
  (hf_periodic : ∀ x : ℝ, f (x + 2) = f x)
  (hf_definition : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → f x = 2^x) :
  f (-9 / 2) = Real.sqrt 2 := by
  sorry

end find_f_neg_9_over_2_l702_702511


namespace one_fourth_of_8_point_4_is_21_over_10_l702_702477

theorem one_fourth_of_8_point_4_is_21_over_10 : (8.4 / 4 : ℚ) = 21 / 10 := 
by
  sorry

end one_fourth_of_8_point_4_is_21_over_10_l702_702477


namespace inclination_angle_of_line_l702_702339

-- Definitions drawn from the condition.
def line_equation (x y : ℝ) := x - y + 1 = 0

-- The statement of the theorem (equivalent proof problem).
theorem inclination_angle_of_line : ∀ x y : ℝ, line_equation x y → θ = π / 4 :=
sorry

end inclination_angle_of_line_l702_702339


namespace x_varies_as_z_raised_to_n_power_l702_702866

noncomputable def x_varies_as_cube_of_y (k y : ℝ) : ℝ := k * y ^ 3
noncomputable def y_varies_as_cube_root_of_z (j z : ℝ) : ℝ := j * z ^ (1/3 : ℝ)

theorem x_varies_as_z_raised_to_n_power (k j z : ℝ) :
  ∃ n : ℝ, x_varies_as_cube_of_y k (y_varies_as_cube_root_of_z j z) = (k * j^3) * z ^ n ∧ n = 1 :=
by
  sorry

end x_varies_as_z_raised_to_n_power_l702_702866


namespace wheel_diameter_l702_702747

noncomputable def pi_approx : ℝ := 3.14159

def circumference (distance : ℝ) (revolutions : ℝ) : ℝ :=
  distance / revolutions

def diameter (C : ℝ) (pi : ℝ) : ℝ :=
  C / pi

theorem wheel_diameter (distance : ℝ) (revolutions : ℝ) (expected_diameter : ℝ) :
  distance = 1056 →
  revolutions = 12.010919017288444 →
  expected_diameter = 27.99 →
  let C := circumference distance revolutions in
  diameter C pi_approx ≈ expected_diameter :=
by
  intros
  sorry -- Proof to be filled in

end wheel_diameter_l702_702747


namespace geometric_statements_correct_l702_702382

/-- Mathematical definitions and properties -/
def equal_angles_are_vertical (angle1 angle2 : ℝ) : Prop := 
∃ (l1 l2 : ℝ), l1 ≠ l2 ∧ angle1 = angle2 

def unique_parallel_line (point line : ℝ) : Prop := 
∃! l, l ≠ line ∧ l ∋ point ∧ l ∥ line 

def perpendicular_segment_shortest (point line : ℝ) : Prop := 
∀ (p : ℝ), (p ∈ line) → dist point (orthogonal_projection line point) <= dist point p

def unique_perpendicular_line (point line : ℝ) : Prop := 
∃! l, l ≠ line ∧ l ⊥ line ∧ l ∋ point 

/-- Proof problem statement -/
theorem geometric_statements_correct :
  (equal_angles_are_vertical angle1 angle2 → false) ∧ 
  (unique_parallel_line point line → false) ∧
  (perpendicular_segment_shortest point line) ∧ 
  (unique_perpendicular_line point line → false) :=
sorry

end geometric_statements_correct_l702_702382


namespace couscous_dishes_l702_702400

def dishes (a b c d : ℕ) : ℕ := (a + b + c) / d

theorem couscous_dishes :
  dishes 7 13 45 5 = 13 :=
by
  unfold dishes
  sorry

end couscous_dishes_l702_702400


namespace sum_primes_reversed_l702_702002

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def reverse_digits (n : ℕ) : ℕ := 
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

def valid_primes : List ℕ := [31, 37, 71, 73]

theorem sum_primes_reversed :
  (∀ p ∈ valid_primes, 20 < p ∧ p < 80 ∧ is_prime p ∧ is_prime (reverse_digits p)) ∧
  (valid_primes.sum = 212) :=
by
  sorry

end sum_primes_reversed_l702_702002


namespace quadratic_relationship_l702_702177

variable (y_1 y_2 y_3 : ℝ)

-- Conditions
def vertex := (-2, 1)
def opens_downwards := true
def intersects_x_axis_at_two_points := true
def passes_through_points := [(1, y_1), (-1, y_2), (-4, y_3)]

-- Proof statement
theorem quadratic_relationship : y_1 < y_3 ∧ y_3 < y_2 :=
  sorry

end quadratic_relationship_l702_702177


namespace minimize_and_maximize_r3_l702_702147

-- define a regular pentagon as a set of points in a plane
structure Pentagon :=
  (vertices : Fin 5 → Point)
  (is_regular : ∀ i, dist (vertices i) (vertices (i + 1) % 5) = dist (vertices 0) (vertices 1))

-- define a point in the plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- define distance from a point to a line (side of the pentagon)
def dist_point_to_line (p : Point) (line : Set Point) : ℝ := sorry

-- define the positions that we are interested in
def is_vertex (p : Point) (pentagon : Pentagon) : Prop := sorry
def is_midpoint_of_side (p : Point) (pentagon : Pentagon) : Prop := sorry

-- define the distance ranking of point M to the sides of the pentagon
def distance_ranking (M : Point) (pentagon : Pentagon) : Fin 5 → ℝ :=
  λ i, dist_point_to_line M (line_through (pentagon.vertices i) (pentagon.vertices ((i + 1) % 5)))

-- define the 3rd smallest distance
def r3 (M : Point) (pentagon : Pentagon) : ℝ :=
  (list.fin_range 5).map (distance_ranking M pentagon)).sort (≤)!.nth_le 3 sorry

-- theorem to prove
theorem minimize_and_maximize_r3 (pentagon : Pentagon) (M : Point) :
  (is_vertex M pentagon → r_3 M pentagon = max_value)
  ∧ (is_midpoint_of_side M pentagon → r_3 M pentagon = min_value) :=
begin
  sorry,
end

end minimize_and_maximize_r3_l702_702147


namespace max_value_of_expression_l702_702175

variables {ℝ : Type*} [linear_ordered_field ℝ]
variables (a b : ℝ → ℝ) (x y : ℝ)

noncomputable def cos_angle := -1 / 2
noncomputable def vector_magnitude (v : ℝ → ℝ) := 2
def vector_c := λ (x y : ℝ), x • a + y • b
noncomputable def magnitude_c (x y : ℝ) := 2 * real.sqrt (x^2 + y^2 - x * y)

theorem max_value_of_expression (hx : x ≠ 0) :
  (∀ (a b : ℝ → ℝ)
    (h1 : ∀ v, vector_magnitude v = 2)
    (h2 : ∀ a b, (a • (1) + b • (1)) * cos_angle = -2),
    (vector_c x y = x • a + y • b) →
    (h3 : magnitude_c x y = 2 * real.sqrt (x^2 + y^2 - x * y)) →
    (max (abs (x / magnitude_c x y)) = abs (sqrt (3) / 3))) :=
by sorry

end max_value_of_expression_l702_702175


namespace choose_8_points_arc_length_not_3_or_8_l702_702028

def points_on_circle (n : ℕ) : set ℕ :=
{ x | x < n }

def is_valid_selection (selection : set ℕ) : Prop :=
  ∀ x y ∈ selection, x ≠ y → (x - y).nat_abs ≠ 3 ∧ (x - y).nat_abs ≠ 8

def count_valid_selections : ℕ :=
  set.count {s : set ℕ | s ⊆ points_on_circle 24 ∧ s.card = 8 ∧ is_valid_selection s}

theorem choose_8_points_arc_length_not_3_or_8 :
  count_valid_selections = 258 :=
sorry

end choose_8_points_arc_length_not_3_or_8_l702_702028


namespace compare_cosine_ratios_l702_702082

theorem compare_cosine_ratios :
  let θ1 := 2016
  let θ2 := 2017
  let θ3 := 2018
  let θ4 := 2019
  real.cos (θ1 % 360) / real.cos (θ2 % 360) < real.cos (θ3 % 360) / real.cos (θ4 % 360) :=
by
  sorry

end compare_cosine_ratios_l702_702082


namespace max_pages_copied_l702_702782

theorem max_pages_copied 
  (cost_per_page : ℕ → ℤ)
  (total_budget : ℤ) 
  (discount_threshold : ℕ) 
  (discount : ℤ)
  (target_pages : ℕ) :
  ∀ (n : ℕ), n ≥ discount_threshold → 
    cost_per_page n = 3.5 ∧ total_budget = 2500 ∧ discount_threshold = 400 ∧ discount = 5 →
    (cost_per_page n * n - discount ≤ total_budget ∧ n = 715)
    :=
begin
  intros n h_threshold h_assumptions,
  sorry
end

end max_pages_copied_l702_702782


namespace reflection_square_is_identity_l702_702911

theorem reflection_square_is_identity (R : Matrix (Fin 2) (Fin 2) ℝ)
  (hR : R.mul_vec (Vector.cons 4 (Vector.cons 2 Vector.nil)) =
                 R.mul_vec (R.mul_vec (Vector.cons 4 (Vector.cons 2 Vector.nil)))) :
  R ⬝ R = Matrix.identity _ := by
  sorry

end reflection_square_is_identity_l702_702911


namespace vector_subtraction_l702_702216

/- Definition of given vectors OA and OB -/
def OA : ℝ × ℝ := (2, 8)
def OB : ℝ × ℝ := (-7, 2)

/- Statement to prove that AB = OB - OA is (-9, -6) -/
theorem vector_subtraction : 
  let AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)
  in AB = (-9, -6) :=
by 
  -- Proof will be added here
  sorry

end vector_subtraction_l702_702216


namespace inequality_proof_l702_702279

theorem inequality_proof (a b c : ℝ) (ha1 : 0 ≤ a) (ha2 : a ≤ 1) (hb1 : 0 ≤ b) (hb2 : b ≤ 1) (hc1 : 0 ≤ c) (hc2 : c ≤ 1) :
  (a / (b + c + 1) + b / (a + c + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1) :=
by
  sorry

end inequality_proof_l702_702279


namespace average_percentage_score_is_71_l702_702880

-- Define the number of students.
def number_of_students : ℕ := 150

-- Define the scores and their corresponding frequencies.
def scores_and_frequencies : List (ℕ × ℕ) :=
  [(100, 10), (95, 20), (85, 45), (75, 30), (65, 25), (55, 15), (45, 5)]

-- Define the total points scored by all students.
def total_points_scored : ℕ := 
  scores_and_frequencies.foldl (λ acc pair => acc + pair.1 * pair.2) 0

-- Define the average percentage score.
def average_score : ℚ := total_points_scored / number_of_students

-- Statement of the proof problem.
theorem average_percentage_score_is_71 :
  average_score = 71.0 := by
  sorry

end average_percentage_score_is_71_l702_702880


namespace circle_yaxis_intersection_sum_l702_702453

theorem circle_yaxis_intersection_sum {
  center_x center_y : ℝ,
  radius : ℝ
} (h_center : center_x = -8) (h_center_y : center_y = 5) (h_radius : radius = 15) :
  let c_eq := (λ x y : ℝ, (x + 8)^2 + (y - 5)^2 = 225)
  let intersection_y_coords := λ y : ℝ, (0 + 8)^2 + (y - 5)^2 = 225
  ∃ y1 y2 : ℝ, intersection_y_coords y1 ∧ intersection_y_coords y2 ∧ y1 + y2 = 10 :=
sorry

end circle_yaxis_intersection_sum_l702_702453


namespace calc_modulus_power_l702_702799

theorem calc_modulus_power (a b : ℂ) (n : ℕ) :
  abs (a + b * complex.I) ^ n = abs a ^ n + abs b ^ n -> 
  a = 2 -> b = -3 -> n = 5 ->
  abs ((2 - 3 * complex.I) ^ 5) = 13 ^ (5 / 2) :=
by
  intro h_prop
  intro h_a
  intro h_b
  intro h_n
  rw [h_a, h_b, h_n]
  rw complex.abs_pow
  have h : abs (2 - 3 * complex.I) = real.sqrt 13 := sorry
  rw h
  have h_exp : (real.sqrt 13) ^ 5 = 13 ^ (5 / 2) := sorry
  rw h_exp
  sorry

end calc_modulus_power_l702_702799


namespace find_expression_maximize_profit_l702_702436

-- First, define the conditions for the function f and its properties
def f (x : ℝ) (a : ℝ) : ℝ := (a / (x - 4)) + 10 * (x - 7)^2

-- Define the main problem statement for the first part
theorem find_expression (f6_eq_15 : f 6 15 = 15) : f x 10 = (10 / (x - 4)) + 10 * (x - 7)^2 :=
by
  sorry

-- Define the profit function
def h (x : ℝ) : ℝ := (x - 4) * f x 10

-- Define the second part; the maximization problem
theorem maximize_profit (hx_derivative: ∀ (x : ℝ), h' x = 30 * x^2 - 360 * x + 1050) 
  (x_interval: 4 < x < 7) : x = 5 :=
by
  have h' := hx_derivative 5
  sorry

end find_expression_maximize_profit_l702_702436


namespace concurrency_of_cevians_l702_702715

-- Define basic entities: points and triangle
variables {A B C H A1 B1 E F : Type}
variable [inst : EuclideanGeometry]

-- Given conditions as definitions/variables
def is_right_angle (AB C : EuclideanGeometry.Point) : Prop := 
  EuclideanGeometry.angle ABC = 90

def is_altitude (C H : EuclideanGeometry.Point) (ABC : EuclideanGeometry.Triangle) : Prop :=
  EuclideanGeometry.is_perpendicular C H ABC

def is_bisector (P Q R : EuclideanGeometry.Point) : Prop :=
  EuclideanGeometry.is_angle_bisector Q P R

def is_midpoint (P M Q : EuclideanGeometry.Point) : Prop :=
  EuclideanGeometry.midpoint P M Q

axiom euclidean_geometry : EuclideanGeometry

open euclidean_geometry

-- Problem statement:
theorem concurrency_of_cevians 
  (ABC : Triangle)
  (H : Point)
  (A1 B1 : Point)
  (E F : Point) 
  (h_right : is_right_angle A B C)
  (h_altitude : is_altitude C H ABC)
  (h_bisector_ha1 : is_bisector C H A1)
  (h_bisector_hb1 : is_bisector A H B1)
  (h_midpoint_e : is_midpoint H E B1)
  (h_midpoint_f : is_midpoint H F A1) :
  meets_on_angle_bisector A E B F (angle_bisector A C B) :=
sorry

end concurrency_of_cevians_l702_702715


namespace equivalent_determinant_conditions_l702_702292

variables {n k : ℕ} (A : fin k → fin n → fin n → ℝ)
variable [fact(n ≥ 2)]

-- Define symmetry for the matrix A
def is_symmetric (A : fin n → fin n → ℝ) : Prop :=
  ∀ i j, A i j = A j i

-- Define the determinant condition for the sum of A_i^2
def sum_A_i_squared_det_zero (A : fin k → fin n → fin n → ℝ) : Prop :=
  matrix.det (∑ i, matrix.mul A[i] A[i]) = 0

-- Define the determinant condition for the sum of A_i B_i for any B_i
def sum_A_i_B_i_det_zero (A : fin k → fin n → fin n → ℝ) : Prop :=
  ∀ (B : fin k → fin n → fin n → ℝ), matrix.det (∑ i, matrix.mul A[i] B[i]) = 0

theorem equivalent_determinant_conditions
  (h_symm : ∀ i, is_symmetric (A i)) :
  sum_A_i_squared_det_zero A ↔ sum_A_i_B_i_det_zero A :=
sorry

end equivalent_determinant_conditions_l702_702292


namespace compare_shaded_areas_l702_702780

def shaded_area_square_I (a : ℝ) : ℝ := (a^2) * (1 / 4)
def shaded_area_square_II (a : ℝ) : ℝ := (a^2) * (1 / 2)
def shaded_area_square_III (a : ℝ) : ℝ := (a^2) * (1 / 2)

theorem compare_shaded_areas (a : ℝ) (ha : a > 0) :
  shaded_area_square_II a = shaded_area_square_III a ∧ 
  (shaded_area_square_II a > shaded_area_square_I a) :=
by
  unfold shaded_area_square_I shaded_area_square_II shaded_area_square_III
  sorry

end compare_shaded_areas_l702_702780


namespace pants_cost_l702_702493

-- The types for costs
def cost_pants (P : ℝ) := P
def cost_shirt (P : ℝ) := 2 * P
def cost_tie (P : ℝ) := (2 / 5) * (2 * P)
def cost_socks := 3.0

-- Each student needs to spend $355 in total.
def total_expenditure (P : ℝ) := 5 * P + 5 * (2 * P) + 5 * ((2 / 5) * (2 * P)) + 5 * 3

-- Given condition and proof statement
theorem pants_cost : ∃ P : ℝ, total_expenditure P = 355 ∧ P = 20 :=
by
  sorry

end pants_cost_l702_702493


namespace marbles_problem_l702_702553

theorem marbles_problem :
  let red_marbles := 20
  let green_marbles := 3 * red_marbles
  let yellow_marbles := 0.20 * green_marbles
  let total_marbles := green_marbles + 3 * green_marbles
  total_marbles - (red_marbles + green_marbles + yellow_marbles) = 148 := by
  sorry

end marbles_problem_l702_702553


namespace complex_conjugate_product_correct_l702_702286

noncomputable def complex_conjugate_product (w : ℂ) (h : abs w = 15) : ℂ :=
  w * conj w

theorem complex_conjugate_product_correct (w : ℂ) (h : abs w = 15) : complex_conjugate_product w h = 225 :=
by
  -- Proof omitted
  sorry

end complex_conjugate_product_correct_l702_702286


namespace sumata_family_total_miles_l702_702978

theorem sumata_family_total_miles
  (days : ℝ) (miles_per_day : ℝ)
  (h1 : days = 5.0)
  (h2 : miles_per_day = 250) : 
  miles_per_day * days = 1250 := 
by
  sorry

end sumata_family_total_miles_l702_702978


namespace one_fourth_of_8_point_4_is_21_over_10_l702_702478

theorem one_fourth_of_8_point_4_is_21_over_10 : (8.4 / 4 : ℚ) = 21 / 10 := 
by
  sorry

end one_fourth_of_8_point_4_is_21_over_10_l702_702478


namespace solve_equation_l702_702970

theorem solve_equation (x : ℝ) (h : x ≠ -2) : (x = -1/2) ↔ (x / (x + 2) + 1 = 1 / (x + 2)) :=
by
  sorry

end solve_equation_l702_702970


namespace smallest_positive_x_max_f_l702_702087

def f (x : ℝ) : ℝ := Real.sin (x / 5) + Real.sin (x / 7)

theorem smallest_positive_x_max_f :
  ∃ (x : ℝ), x > 0 ∧ (∀ y : ℝ, y > 0 → f y ≤ f x) ∧ x = 5850 :=
sorry

end smallest_positive_x_max_f_l702_702087


namespace ab_perpendicular_cd_l702_702243

variables {V : Type*} [inner_product_space ℝ V]

-- Define points A, B, C, D
variables (A B C D : V)

-- Define the conditions provided in the problem.
variables (h1 : inner_product A C = 0)
variables (h2 : inner_product A D = 0)
variables (h3 : inner_product B D = 0)
variables (h4 : inner_product B C = 0)

-- Prove that AB is perpendicular to CD
theorem ab_perpendicular_cd (h1 : inner_product A C = 0) (h2 : inner_product A D = 0) (h3 : inner_product B D = 0) (h4 : inner_product B C = 0) : inner_product B C = 0 :=
begin
  sorry
end

end ab_perpendicular_cd_l702_702243


namespace parabola_vertex_coordinates_l702_702331

theorem parabola_vertex_coordinates :
  ∃ (h k : ℝ), (∀ (x : ℝ), (y = (x - h)^2 + k) = (y = (x-1)^2 + 2)) ∧ h = 1 ∧ k = 2 :=
by
  sorry

end parabola_vertex_coordinates_l702_702331


namespace range_of_x_l702_702158

noncomputable def f (x : ℝ) : ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom monotone_increasing (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : a ≤ b) : f a ≤ f b

theorem range_of_x (x : ℝ) (h : f (1 - x) < f 2) : -1 < x ∧ x < 3 :=
begin
  have : |1 - x| < 2, {
    sorry
  },
  split; linarith
end

end range_of_x_l702_702158


namespace dress_shirt_cost_l702_702267

theorem dress_shirt_cost :
  ∃ x : ℝ, (3 * x + 0.10 * (3 * x) = 66) ∧ x = 20 :=
begin
  existsi 20,
  split,
  {
    -- This is the calculation part which proves 3 shirts + 10% tax equals 66
    have h₁ : 3 * (20 : ℝ) = 60 := by norm_num,
    have h₂ : 0.10 * 60 = 6 := by norm_num,
    have h₃ : 60 + 6 = 66 := by norm_num,
    exact h₃,
  },
  {
    -- This part is just the proof of the solution's correctness
    refl,
  }
end

end dress_shirt_cost_l702_702267


namespace total_weight_is_correct_l702_702356

-- Define the weight of apples
def weight_of_apples : ℕ := 240

-- Define the multiplier for pears
def pears_multiplier : ℕ := 3

-- Define the weight of pears
def weight_of_pears := pears_multiplier * weight_of_apples

-- Define the total weight of apples and pears
def total_weight : ℕ := weight_of_apples + weight_of_pears

-- The theorem that states the total weight calculation
theorem total_weight_is_correct : total_weight = 960 := by
  sorry

end total_weight_is_correct_l702_702356


namespace solve_ab_sum_l702_702865

theorem solve_ab_sum (x a b : ℝ) (ha : ℕ) (hb : ℕ)
  (h1 : a = ha)
  (h2 : b = hb)
  (h3 : x = a + Real.sqrt b)
  (h4 : x^2 + 3 * x + 3 / x + 1 / x^2 = 26) :
  (ha + hb = 5) :=
sorry

end solve_ab_sum_l702_702865


namespace maximum_value_of_expression_l702_702927

noncomputable def max_value (x y z : ℝ) : ℝ :=
  sqrt (3 * x ^ 2 + 3) + sqrt (3 * y ^ 2 + 6) + sqrt (3 * z ^ 2 + 3)

theorem maximum_value_of_expression (x y z : ℝ)
  (h1 : x + y + z = 2)
  (h2 : x ≥ -1)
  (h3 : y ≥ -2)
  (h4 : z ≥ -1) :
  max_value x y z ≤ 4 * sqrt 3 :=
sorry

end maximum_value_of_expression_l702_702927


namespace total_games_in_season_l702_702429

theorem total_games_in_season :
  let division1_teams := 5 in
  let division2_teams := 6 in
  let intra_division_games (n : ℕ) := (n * (n - 1) * 3) / 2 in
  let inter_division_games (n m : ℕ) := n * m * 2 in
  intra_division_games division1_teams + intra_division_games division2_teams + inter_division_games division1_teams division2_teams = 135 :=
by
  sorry

end total_games_in_season_l702_702429


namespace trig_expression_identity_l702_702507

theorem trig_expression_identity (a : ℝ) (h : 2 * Real.sin a = 3 * Real.cos a) : 
  (4 * Real.sin a + Real.cos a) / (5 * Real.sin a - 2 * Real.cos a) = 14 / 11 :=
by
  sorry

end trig_expression_identity_l702_702507


namespace purely_imaginary_second_quadrant_l702_702132

-- Define the conditions for purely imaginary.
def condition1 (m : ℝ) := (m^2 - m - 6 ≠ 0) ∧ (m^2 - 5m + 6 = 0)

-- Define the condition for second quadrant.
def condition2 (m : ℝ) := (m^2 - 5m + 6 < 0) ∧ (m^2 - m - 6 > 0)

-- Prove that if z is purely imaginary, then m = -2.
theorem purely_imaginary (m : ℝ) (h1 : condition1 m) : m = -2 :=
sorry

-- Prove that if z lies in the second quadrant, then m < -3 or -2 < m < 2 or m > 3.
theorem second_quadrant (m : ℝ) (h2 : condition2 m) : m < -3 ∨ (-2 < m ∧ m < 2) ∨ m > 3 :=
sorry

end purely_imaginary_second_quadrant_l702_702132


namespace chocolate_bars_in_large_box_l702_702745

theorem chocolate_bars_in_large_box:
  let num_small_boxes := 150 in
  let num_chocolates_per_box := 37 in
  num_small_boxes * num_chocolates_per_box = 5550 :=
by
  sorry

end chocolate_bars_in_large_box_l702_702745


namespace top_8_by_median_l702_702551

theorem top_8_by_median (scores : Fin 16 → ℝ) (unique_scores : ∀ i j, scores i = scores j → i = j) 
  (self_score : ℝ) :
  (self_score > (Finset.median (Finset.univ.image scores)).val → ∃ i, scores i = self_score) ∧
  (self_score ≤ (Finset.median (Finset.univ.image scores)).val → ¬ ∃ i, scores i = self_score → ∃ i, scores i = self_score ∧ i < 8 ) :=
sorry

end top_8_by_median_l702_702551


namespace arithmetic_sequence_sum_ratio_l702_702166

variable {a_n : ℕ → ℝ}

-- Sum of the first n terms of an arithmetic sequence
def S (n : ℕ) (a_n : ℕ → ℝ) : ℝ := n * (a_n 1 + a_n n) / 2

theorem arithmetic_sequence_sum_ratio (h : a_n 5 = 5 * a_n 3) : S 9 a_n / S 5 a_n = 9 :=
by
  sorry

end arithmetic_sequence_sum_ratio_l702_702166


namespace option_A_option_B_option_C_option_D_l702_702851

-- Definitions for vectors a and b
def vector_a (m : ℝ) := (m, 2 * m, 2)
def vector_b (m : ℝ) := (2 * m - 5, -m, -1)

-- Function to calculate dot product
def dot_product (v w : ℝ × ℝ × ℝ) := v.1 * w.1 + v.2 * w.2 + v.3 * w.3

-- Function to calculate the magnitude of a vector
def magnitude (v : ℝ × ℝ × ℝ) := real.sqrt(v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

-- Prove the candidate's option A
theorem option_A (m : ℝ) : (vector_a m).1 * (vector_b m).2 = 2 * (vector_a m).2 * -m ∧ (vector_a m).3 * -1 = 2 ∧ m = 2 :=
    sorry

-- Prove the candidate's option B
theorem option_B (m : ℝ) : dot_product (vector_a m) (vector_b m) = 0 ↔ m = -2 / 5 :=
    sorry

-- Prove the candidate's option C
theorem option_C : ∃ (m : ℝ), magnitude (vector_a m) = 2 :=
    exists.intro 0 (by simp [vector_a, magnitude, real.sqrt_pow_two])

-- Prove the candidate's option D
theorem option_D : ∀ (m : ℝ), magnitude (vector_a m) < 4 :=
    sorry

end option_A_option_B_option_C_option_D_l702_702851


namespace geom_series_sum_l702_702450

theorem geom_series_sum : 
  let a := (1 : ℚ) / 5;
  let r := -(1 : ℚ) / 2;
  let n := 6 in (a * (1 - r^n) / (1 - r)) = (21 : ℚ) / 160 :=
by
  sorry

end geom_series_sum_l702_702450


namespace boundary_of_set_T_is_quadrilateral_l702_702601

def set_T (a x y : ℝ) : Prop :=
  (a ≤ x ∧ x ≤ 3 * a) ∧
  (a ≤ y ∧ y ≤ 3 * a) ∧
  (x + y ≤ 4 * a) ∧
  (x + 2 * a ≥ y) ∧
  (y + 2 * a ≥ x)

theorem boundary_of_set_T_is_quadrilateral (a : ℝ) (ha : a > 0) : 
  ∃ (sides : ℕ), sides = 4 ∧ 
  (∃ (points : set (ℝ × ℝ)), set_T a ∧ (set.T.points = list.to_set [(a,a), (a,3*a), (3*a,a), (3 * a, 3 * a)])) :=
  sorry -- Proof omitted

end boundary_of_set_T_is_quadrilateral_l702_702601


namespace ratio_between_second_and_third_l702_702352

noncomputable def ratio_second_third : ℚ := sorry

theorem ratio_between_second_and_third (A B C : ℕ) (h₁ : A + B + C = 98) (h₂ : A * 3 = B * 2) (h₃ : B = 30) :
  ratio_second_third = 5 / 8 := sorry

end ratio_between_second_and_third_l702_702352


namespace triangle_subdivision_l702_702065

theorem triangle_subdivision (Δ : Triangle) (hΔ : ∀ angle ∈ Δ.angles, angle ≤ 120) 
(subdiv : Subdivision Δ) : 
  ∃ Δ' ∈ subdiv.triangles, ∀ angle ∈ Δ'.angles, angle ≤ 120 := 
sorry

end triangle_subdivision_l702_702065


namespace area_of_smaller_base_of_truncated_cone_l702_702361

-- Define the radii of the bases of the cones
def r1 : ℝ := 10
def r2 : ℝ := 15
def r3 : ℝ := 15

-- Define the radius of the smaller base of the truncated cone
def smaller_base_radius : ℝ := 2

-- Function to compute the area of a circle based on its radius
def area_of_circle (r : ℝ) : ℝ := real.pi * r ^ 2

-- Theorem stating the area of the smaller base of the truncated cone
theorem area_of_smaller_base_of_truncated_cone
  (h1 : r1 = 10)
  (h2 : r2 = 15)
  (h3 : r3 = 15)
  (h4 : smaller_base_radius = 2) :
  area_of_circle smaller_base_radius = 4 * real.pi :=
by
  sorry

end area_of_smaller_base_of_truncated_cone_l702_702361


namespace one_fourth_of_8_point_4_is_21_over_10_l702_702479

theorem one_fourth_of_8_point_4_is_21_over_10 : (8.4 / 4 : ℚ) = 21 / 10 := 
by
  sorry

end one_fourth_of_8_point_4_is_21_over_10_l702_702479


namespace problem_statement_l702_702272

def sign (x : ℝ) : Int :=
  if x ≥ 0 then 1 else -1

theorem problem_statement {n : ℕ} (hn : n % 2 = 1) :
  ∃ (a : Fin n → Fin n → Int) (b : Fin n → Int),
  (∀ i, b i ∈ ({1, -1} : Set Int)) ∧ 
  (∀ i j, a i j ∈ ({1, -1} : Set Int)) ∧ 
  (∀ (x : Fin n → Int) (hx : ∀ i, x i ∈ ({1, -1} : Set Int)),
    let y := fun i => sign (Finset.univ.sum (fun j => a i j * x j)),
        z := sign (Finset.univ.sum (fun i => y i * b i))
    in z = Finset.univ.prod x) :=
sorry

end problem_statement_l702_702272


namespace value_of_expression_l702_702226

theorem value_of_expression
  (m n : ℝ)
  (h1 : n = -2 * m + 3) :
  4 * m + 2 * n + 1 = 7 :=
sorry

end value_of_expression_l702_702226
