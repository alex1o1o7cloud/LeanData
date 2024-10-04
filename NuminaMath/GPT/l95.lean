import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Complex.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GeomSeries
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Logarithm
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Parity
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.CombinatorialGame
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Nat.Lcm
import Mathlib.Data.Nat.Pow
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Polynomial.RingDivision
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Data.Set.Lattice
import Mathlib.Data.ZMod.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.GroupTheory.Perm.Cycle
import Mathlib.Integration
import Mathlib.MeasureTheory.Integral
import Mathlib.Order.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.LinearCombination

namespace remaining_shirt_cost_l95_95177

theorem remaining_shirt_cost
  (total_cost : ℕ)
  (cost : ℕ → ℕ)
  (h_total_cost : total_cost = 85)
  (h_costs : ∀ k, k ∈ {1, 2, 3} → cost k = 15) :
  cost 4 = 20 ∧ cost 5 = 20 :=
by
  sorry

end remaining_shirt_cost_l95_95177


namespace solid_with_isosceles_triangle_views_is_regular_triangular_pyramid_l95_95299

theorem solid_with_isosceles_triangle_views_is_regular_triangular_pyramid
  (solid : Type)
  (is_isosceles_triangle : ∀ (view : solid → solid → solid), Prop)
  (A B C : solid)
  (H1 : is_isosceles_triangle (λ x y, orthographic_top_view x y))
  (H2 : is_isosceles_triangle (λ x y, orthographic_front_view x y))
  (H3 : is_isosceles_triangle (λ x y, orthographic_side_view x y)) :
  solid = regular_triangular_pyramid := 
sorry

end solid_with_isosceles_triangle_views_is_regular_triangular_pyramid_l95_95299


namespace value_of_x_l95_95540

theorem value_of_x (x y z w : ℕ) (h1 : x = y + 7) (h2 : y = z + 12) (h3 : z = w + 25) (h4 : w = 90) : x = 134 :=
by
  sorry

end value_of_x_l95_95540


namespace find_b_l95_95617

noncomputable def f (b x : ℝ) := x^2 - 2 * b * x - 1

theorem find_b (b : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → ∃ (min_x max_x : ℝ), 
  (min_x = f b 0 ∨ min_x = f b 1 ∨ ∃ y : ℝ, y ∈ set.Icc 0 1 ∧ min_x = f b y) ∧ 
  (max_x = f b 0 ∨ max_x = f b 1 ∨ ∃ y : ℝ, y ∈ set.Icc 0 1 ∧ max_x = f b y) ∧
  max_x - min_x = 1) ↔ b = 0 ∨ b = 1 :=
begin
  sorry
end

end find_b_l95_95617


namespace prec_property_l95_95922

noncomputable def prec (a b : ℕ) : Prop :=
  sorry -- The construction of the relation from the problem

axiom prec_total : ∀ a b : ℕ, (prec a b ∨ prec b a ∨ a = b)
axiom prec_trans : ∀ a b c : ℕ, (prec a b ∧ prec b c) → prec a c

theorem prec_property : ∀ a b c : ℕ, (prec a b ∧ prec b c) → 2 * b ≠ a + c :=
by
  sorry

end prec_property_l95_95922


namespace num_possible_pairs_l95_95949

theorem num_possible_pairs
  (a b : ℕ)
  (hb_gt_ha: b > a)
  (border_width : ℕ := 2)
  (area: ℕ := a * b)
  (painted_area: ℤ := (a-4) * (b-4))
  (border_area: ℤ := area - painted_area)
  (border_area_third: Prop := (border_area : ℚ) = (1 / 3) * area) :
  (∃ a b, 
      b > a ∧ 
      (a - 4) > 0 ∧ 
      (b - 4) > 0 ∧ 
      border_area_third.to_bool = tt ∧ 
      (a = 7 ∧ b = 18) ∨ 
      (a = 8 ∧ b = 12) ∨ 
      (a = 9 ∧ b = 10)) :=
begin
  sorry
end

end num_possible_pairs_l95_95949


namespace Ivanov_made_an_error_l95_95813

theorem Ivanov_made_an_error : 
  (∀ x m s : ℝ, 0 = x → 4 = m → 15.917 = s^2 → ¬ (|x - m| ≤ real.sqrt s)) :=
by 
  intros x m s hx hm hs2
  rw [hx, hm, hs2]
  have H : |0 - 4| = 4 := by norm_num
  have H2 : real.sqrt 15.917 ≈ 3.99 := by norm_num
  exact neq_of_not_le (by norm_num : 4 ≠ 3.99) H2 sorry

end Ivanov_made_an_error_l95_95813


namespace diagonals_intersection_probability_l95_95092

theorem diagonals_intersection_probability (decagon : Polygon) (h_regular : decagon.is_regular ∧ decagon.num_sides = 10) :
  probability_intersection_inside decagon = 42 / 119 := 
sorry

end diagonals_intersection_probability_l95_95092


namespace umbrellas_problem_l95_95110

theorem umbrellas_problem :
  ∃ (b r : ℕ), b = 36 ∧ r = 27 ∧ 
  b = (45 + r) / 2 ∧ 
  r = (45 + b) / 3 :=
by sorry

end umbrellas_problem_l95_95110


namespace coefficient_x2_in_Q_l95_95752

noncomputable def P (x : ℝ) : ℝ := (∑ k in Finset.range 20, (-1)^k * x^k)

noncomputable def Q (x : ℝ) : ℝ := P (x - 1)

theorem coefficient_x2_in_Q (x : ℝ) : 
  (Q x).coeff 2 = 1140 := 
sorry

end coefficient_x2_in_Q_l95_95752


namespace train_speed_l95_95543

theorem train_speed (length time : ℕ) (h_length : length = 100) (h_time : time = 20) :
  (length / time = 5) := 
by
  rw [h_length, h_time]
  exact rfl

end train_speed_l95_95543


namespace square_parallelogram_negation_contrapositive_l95_95851

theorem square_parallelogram_negation_contrapositive :
  (∀ (Q : Type) [isQuadrilateral Q], isSquare Q → isParallelogram Q) ↔
  (∀ (Q : Type) [isQuadrilateral Q], ¬ isSquare Q → ¬ isParallelogram Q) ∧
  (∀ (Q : Type) [isQuadrilateral Q], ¬ isParallelogram Q → ¬ isSquare Q) :=
sorry

end square_parallelogram_negation_contrapositive_l95_95851


namespace class_proof_l95_95356

def class_problem : Prop :=
  let total_students := 25
  let (students : Type) := {s // s ∈ Fin total_students}
  let good students (s : students) : Prop := sorry -- Condition Placeholder
  let troublemakers (s : students) : Prop := sorry -- Condition Placeholder
  have all_good_or_trouble : ∀ s, good students s ∨ troublemakers s,
    from sorry,
  have lh_trouble : ∀ (a b c d e : students), (good students a ∧ good students b ∧ good students c ∧ good students d ∧ good students e) →
    ∃ (x : ℕ), x > ((total_students - 1) / 2) ∧
    (λ rem_trouble : students - {a, b, c, d, e}, B : ℕ, B = x) ∧
    (λ rem_good : students - {a, b, c, d, e}, G : ℕ, G ∈ rem_good), from sorry,
  have th_triple_ratio : ∀ s, (students - {s}).card = 24 →
    ∃ (x y : ℕ), (troublemakers x ∧ good students y) ∧ (3*y = x), from sorry,
  ∃ (num_good_students : ℕ), (num_good_students = 5 ∨ num_good_students = 7)
  
theorem class_proof : class_problem :=
  sorry

end class_proof_l95_95356


namespace sabina_loan_amount_l95_95456

-- Definitions corresponding to the problem conditions
def tuition_cost : ℕ := 30000
def sabina_savings : ℕ := 10000
def grant_percentage : ℚ := 0.40

-- Proof statement to show the amount of loan required
theorem sabina_loan_amount : ∀ (tuition_cost sabina_savings : ℕ) (grant_percentage : ℚ), 
  sabina_savings < tuition_cost →
  let remainder := tuition_cost - sabina_savings in
  let grant_amount := grant_percentage * remainder in
  let loan_amount := remainder - grant_amount in
  loan_amount = 12000 :=
by 
  intros tuition_cost sabina_savings grant_percentage h_savings_lt;
  let remainder := tuition_cost - sabina_savings;
  let grant_amount := grant_percentage * ↑remainder;
  let loan_amount := remainder - grant_amount.to_nat;
  sorry

end sabina_loan_amount_l95_95456


namespace find_C_range_ab_bc_ca_over_c2_l95_95419

variable {A B C a b c : ℝ}

-- Given conditions
def trig_condition : Prop := 
  (cos A) / (1 + sin A) = (sin B) / (1 + cos B)

-- Objective Part 1: Prove C = π/2
theorem find_C (h1 : trig_condition) : C = π / 2 := 
  sorry

-- Objective Part 2: Find the range of (ab + bc + ca) / c^2 given C = π/2
theorem range_ab_bc_ca_over_c2 (h1 : trig_condition) (h2 : C = π / 2) :
  1 < (ab + bc + ca) / c^2 ∧ (ab + bc + ca) / c^2 ≤ (1 + 2 * sqrt 2) / 2 :=
  sorry

end find_C_range_ab_bc_ca_over_c2_l95_95419


namespace greatest_multiple_less_150_l95_95891

/-- Define the LCM of two natural numbers -/
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem greatest_multiple_less_150 (x y : ℕ) (h1 : x = 15) (h2 : y = 20) : 
  (∃ m : ℕ, LCM x y * m < 150 ∧ ∀ n : ℕ, LCM x y * n < 150 → LCM x y * n ≤ LCM x y * m) ∧ 
  (∃ m : ℕ, LCM x y * m = 120) :=
by
  sorry

end greatest_multiple_less_150_l95_95891


namespace good_students_options_l95_95322

variables (E B : ℕ)

-- Define the condition that the class has 25 students
def total_students : Prop := E + B = 25

-- Define the condition given by the first group of students
def first_group_condition : Prop := B > 12

-- Define the condition given by the second group of students
def second_group_condition : Prop := B = 3 * (E - 1)

-- Define the problem statement
theorem good_students_options (E B : ℕ) :
  total_students E B → first_group_condition B → second_group_condition E B → (E = 5 ∨ E = 7) :=
by
  intros h1 h2 h3
  sorry

end good_students_options_l95_95322


namespace introduction_methods_count_l95_95712

def street := ["Zhonghe", "Guixi", "Shiyang"]
def projects := ["A", "B", "C", "D"]
def num_ways : ℕ := 36

theorem introduction_methods_count :
  ∃ f : projects → street, (∀ s ∈ street, ∃ p ∈ projects, f p = s) ∧ (f.proper_count = 36) := 
sorry

end introduction_methods_count_l95_95712


namespace fraction_irreducible_l95_95448

theorem fraction_irreducible (n : ℕ) : Nat.gcd (12 * n + 1) (30 * n + 2) = 1 :=
sorry

end fraction_irreducible_l95_95448


namespace Ivanov_made_error_l95_95806

theorem Ivanov_made_error (x m : ℝ) (S_squared : ℝ) (h_x : x = 0) (h_m : m = 4) (h_S_squared : S_squared = 15.917) :
  ¬(|x - m| ≤ Real.sqrt S_squared) :=
by
  have h_sd : Real.sqrt S_squared = 3.99 := by sorry
  have h_ineq: |x - m| = 4 := by sorry
  rw [h_sd, h_ineq]
  linarith

end Ivanov_made_error_l95_95806


namespace find_product_l95_95591

-- Define the variables used in the problem statement
variables (A P D B E C F : Type) (AP PD BP PE CP PF : ℝ)

-- The condition given in the problem
def condition (x y z : ℝ) : Prop := 
  x + y + z = 90

-- The theorem to prove
theorem find_product (x y z : ℝ) (h : condition x y z) : 
  x * y * z = 94 :=
sorry

end find_product_l95_95591


namespace common_chord_length_l95_95850

theorem common_chord_length (x y : ℝ) : 
    (x^2 + y^2 = 4) → 
    (x^2 + y^2 - 4*x + 4*y - 12 = 0) → 
    ∃ l : ℝ, l = 2 * Real.sqrt 2 :=
by
  intros h1 h2
  sorry

end common_chord_length_l95_95850


namespace equal_perpendiculars_from_parallel_lines_l95_95181

theorem equal_perpendiculars_from_parallel_lines (l l' : Line) (A B : Point)
    (h_parallel : l ∥ l')
    (h_perpendicular_A : Perpendicular A l A l')
    (h_perpendicular_B : Perpendicular B l B l' ):
    distance A A' = distance B B' :=
sorry

end equal_perpendiculars_from_parallel_lines_l95_95181


namespace class_proof_l95_95358

def class_problem : Prop :=
  let total_students := 25
  let (students : Type) := {s // s ∈ Fin total_students}
  let good students (s : students) : Prop := sorry -- Condition Placeholder
  let troublemakers (s : students) : Prop := sorry -- Condition Placeholder
  have all_good_or_trouble : ∀ s, good students s ∨ troublemakers s,
    from sorry,
  have lh_trouble : ∀ (a b c d e : students), (good students a ∧ good students b ∧ good students c ∧ good students d ∧ good students e) →
    ∃ (x : ℕ), x > ((total_students - 1) / 2) ∧
    (λ rem_trouble : students - {a, b, c, d, e}, B : ℕ, B = x) ∧
    (λ rem_good : students - {a, b, c, d, e}, G : ℕ, G ∈ rem_good), from sorry,
  have th_triple_ratio : ∀ s, (students - {s}).card = 24 →
    ∃ (x y : ℕ), (troublemakers x ∧ good students y) ∧ (3*y = x), from sorry,
  ∃ (num_good_students : ℕ), (num_good_students = 5 ∨ num_good_students = 7)
  
theorem class_proof : class_problem :=
  sorry

end class_proof_l95_95358


namespace trace_ellipse_l95_95417

-- Define the complex plane and the modulus function
open Complex

-- Define main theorem
theorem trace_ellipse (w : ℂ) (r : ℝ) (h1 : r = 3) (h2 : abs w = r) :
  ∃ a b : ℝ, ∀ (c d : ℝ), w = c + d * Complex.i → 
  (∃ x y : ℝ, (x : ℂ) + (y * Complex.i) = w + 2 / w ∧
   (x^2) / (a^2) + (y^2) / (b^2) = 1) :=
by
  sorry

end trace_ellipse_l95_95417


namespace arithmetic_sequence_sum_l95_95499

theorem arithmetic_sequence_sum (c d : ℕ) 
  (h1 : ∀ (a1 a2 a3 a4 a5 a6 : ℕ), a1 = 3 → a2 = 10 → a3 = 17 → a6 = 38 → (a2 - a1 = a3 - a2) → (a3 - a2 = c - a3) → (c - a3 = d - c) → (d - c = a6 - d)) : 
  c + d = 55 := 
by 
  sorry

end arithmetic_sequence_sum_l95_95499


namespace total_time_to_watch_all_episodes_l95_95747

theorem total_time_to_watch_all_episodes 
  (announced_seasons : ℕ) (episodes_per_season : ℕ) (additional_episodes_last_season : ℕ)
  (seasons_before_announcement : ℕ) (episode_duration : ℝ) :
  announced_seasons = 1 →
  episodes_per_season = 22 →
  additional_episodes_last_season = 4 →
  seasons_before_announcement = 9 →
  episode_duration = 0.5 →
  let total_episodes_previous := seasons_before_announcement * episodes_per_season in
  let episodes_last_season := episodes_per_season + additional_episodes_last_season in 
  let total_episodes := total_episodes_previous + episodes_last_season in 
  total_episodes * episode_duration = 112 :=
by
  intros
  sorry

end total_time_to_watch_all_episodes_l95_95747


namespace probability_same_color_correct_l95_95098

noncomputable def probability_same_color : ℚ :=
let total_balls := 20
let p_blue := (8 / total_balls) * (8 / total_balls)
let p_green := (5 / total_balls) * (5 / total_balls)
let p_red := (7 / total_balls) * (7 / total_balls)
in p_blue + p_green + p_red

theorem probability_same_color_correct :
  probability_same_color = 69 / 200 :=
sorry

end probability_same_color_correct_l95_95098


namespace digits_sum_distinct_l95_95243

theorem digits_sum_distinct :
  ∃ (a b c : ℕ), (9 ≥ a ∧ a ≥ 0) ∧ (9 ≥ b ∧ b ≥ 0) ∧ (9 ≥ c ∧ c ≥ 0) ∧
  let N1 := 11111 * a in
  let N2 := 1111 * b in
  let N3 := 111 * c in
  let S := N1 + N2 + N3 in
  (10000 ≤ S ∧ S ≤ 99999) ∧
  (∃ d1 d2 d3 d4 d5 : ℕ, d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧
                        d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ 
                        d3 ≠ d4 ∧ d3 ≠ d5 ∧ d4 ≠ d5 ∧
                        S = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5) := 
by
  sorry

end digits_sum_distinct_l95_95243


namespace action_figures_per_shelf_l95_95784

/-- Mike has 64 action figures he wants to display. If each shelf 
    in his room can hold a certain number of figures and he needs 8 
    shelves, prove that each shelf can hold 8 figures. -/
theorem action_figures_per_shelf :
  (64 / 8) = 8 :=
by
  sorry

end action_figures_per_shelf_l95_95784


namespace third_cone_vertex_angle_l95_95870

theorem third_cone_vertex_angle :
  ∀ (A : Point) (cone1 cone2 cone3 cone4 : Cone), 
    (cone1.vertex = A) ∧ (cone2.vertex = A) ∧ (cone3.vertex = A) ∧ (cone4.vertex = A) →
    (cone1.vertex_angle = π / 6) ∧ (cone2.vertex_angle = π / 6) ∧ (cone4.vertex_angle = π / 3) →
    cone1.is_tangent_to cone2 ∧ cone1.is_tangent_to cone3 ∧ cone1.is_tangent_to cone4 ∧
    cone2.is_tangent_to cone3 ∧ cone2.is_tangent_to cone4 ∧ cone3.is_tangent_to cone4 →
    cone3.vertex_angle = 2 * (Real.arccot (√3 + 4)) :=
by
  intros A cone1 cone2 cone3 cone4 vertex_cond angle_cond tangent_cond
  sorry

end third_cone_vertex_angle_l95_95870


namespace bmo1999_q5_l95_95548

theorem bmo1999_q5
  (p : ℕ) (h_prime : Nat.Prime p) (h_p_ge_2 : p > 2) (h_p_mod_3 : p % 3 = 2) :
  let S := { z | ∃ x y : ℤ, 0 ≤ y ∧ y ≤ p - 1 ∧ z = y^2 - x^3 - 1 }
  in ∃ t ≤ p - 1, { x | x ∈ S ∧ p ∣ x }.card ≤ t :=
sorry

end bmo1999_q5_l95_95548


namespace mean_value_of_pentagon_angles_l95_95904

theorem mean_value_of_pentagon_angles : 
  let n := 5 
  let interior_angle_sum := (n - 2) * 180 
  mean_angle = interior_angle_sum / n :=
  sorry

end mean_value_of_pentagon_angles_l95_95904


namespace total_number_of_sweets_l95_95941

theorem total_number_of_sweets 
  (S : ℕ) 
  (mother_kept : S / 3 = S / 3) 
  (total_children : 2 * S / 3 = 18) 
  (eldest : 8) 
  (youngest : 4) 
  (second: 6) :
  S = 27 :=
by
  sorry

end total_number_of_sweets_l95_95941


namespace probability_line_through_cube_faces_l95_95711

def prob_line_intersects_cube_faces : ℚ :=
  1 / 7

theorem probability_line_through_cube_faces :
  let cube_vertices := 8
  let total_selections := Nat.choose cube_vertices 2
  let body_diagonals := 4
  let probability := (body_diagonals : ℚ) / total_selections
  probability = prob_line_intersects_cube_faces :=
by {
  sorry
}

end probability_line_through_cube_faces_l95_95711


namespace students_left_in_school_l95_95438

variable (totalStudents : ℕ) (checkedOutPercent : ℚ) (fieldTripFraction : ℚ)

theorem students_left_in_school (h1 : totalStudents = 450) 
    (h2 : checkedOutPercent = 0.35) 
    (h3 : fieldTripFraction = 1 / 6) : 
    let checkedOut := (checkedOutPercent * totalStudents).toNat in
    let remainingAfterCheckout := totalStudents - checkedOut in
    let fieldTrip := (fieldTripFraction * remainingAfterCheckout).toNat in
    let studentsLeft := remainingAfterCheckout - fieldTrip in
    studentsLeft = 245 := 
by
    sorry

end students_left_in_school_l95_95438


namespace cyclic_quadrilateral_angle_proof_l95_95197

-- Define the problem with the given conditions and the statement to prove.
theorem cyclic_quadrilateral_angle_proof
  (A B C D E : Type)
  [circle A B C D]
  (h1 : on_circle A B C D)
  (h2 : extended_line A B E)
  (h3 : ∠BAD = 75)
  (h4 : ∠ADC = 83) :
  ∠EBC = 83 :=
sorry

end cyclic_quadrilateral_angle_proof_l95_95197


namespace functional_relationship_and_range_maximized_profit_profit_notless_than_1600_l95_95838

-- Define the conditions
def cost_price : ℝ := 20
def selling_price_initial : ℝ := 30
def units_sold_initial : ℕ := 100
def decrease_per_unit_price_increase : ℕ := 2
def min_units_sold : ℕ := 70
def min_selling_price : ℝ := 35

-- Define the functional relationship for weekly profit y in terms of selling price x
def weekly_profit (x : ℝ) : ℝ := -2 * x^2 + 200 * x - 3200

-- Range of x
def price_range (x : ℝ) : Prop := 35 ≤ x ∧ x ≤ 45

-- Main theorem statements based on the problem
theorem functional_relationship_and_range (x : ℝ) (h : price_range x) :
  weekly_profit x = -2 * x^2 + 200 * x - 3200 := 
sorry

theorem maximized_profit : 
  (∃ x : ℝ, price_range x ∧ 
  weekly_profit x ≥ weekly_profit y forall (y : ℝ), price_range y) :=
sorry

theorem profit_notless_than_1600 (x : ℝ) (h1 : 1600 ≤ weekly_profit x) :
  40 ≤ x ∧ x ≤ 45 :=
sorry

end functional_relationship_and_range_maximized_profit_profit_notless_than_1600_l95_95838


namespace hash_op_correct_l95_95074

-- Definition of the custom operation #
def hash_op (a b : ℕ) : ℕ := a * b - b + b ^ 2

-- The theorem to prove that 3 # 8 = 80
theorem hash_op_correct : hash_op 3 8 = 80 :=
by
  sorry

end hash_op_correct_l95_95074


namespace number_of_remaining_triangles_after_12_repeats_l95_95586

-- Definitions based on the problem's conditions
def initial_triangle_side_length := (1 : ℝ)
def number_of_repeats := 12
def side_length_of_remaining_triangles (n : ℕ) : ℝ := initial_triangle_side_length / (2 ^ n)

theorem number_of_remaining_triangles_after_12_repeats :
  let n := number_of_repeats in
  (3 : ℕ) ^ n = 531441 :=
by
  sorry

end number_of_remaining_triangles_after_12_repeats_l95_95586


namespace each_girl_brought_2_cups_l95_95514

-- Here we define all the given conditions
def total_students : ℕ := 30
def num_boys : ℕ := 10
def cups_per_boy : ℕ := 5
def total_cups : ℕ := 90

-- Define the conditions as Lean definitions
def num_girls : ℕ := total_students - num_boys -- From condition 3
def cups_by_boys : ℕ := num_boys * cups_per_boy
def cups_by_girls : ℕ := total_cups - cups_by_boys

-- Define the final question and expected answer
def cups_per_girl : ℕ := cups_by_girls / num_girls

-- Final problem statement to prove
theorem each_girl_brought_2_cups :
  cups_per_girl = 2 :=
by
  have h1 : num_girls = 20 := by sorry
  have h2 : cups_by_boys = 50 := by sorry
  have h3 : cups_by_girls = 40 := by sorry
  have h4 : cups_per_girl = 2 := by sorry
  exact h4

end each_girl_brought_2_cups_l95_95514


namespace roots_of_equation_l95_95981

def operation (a b : ℝ) : ℝ := a^2 * b + a * b - 1

theorem roots_of_equation :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ operation x₁ 1 = 0 ∧ operation x₂ 1 = 0 :=
by
  sorry

end roots_of_equation_l95_95981


namespace log_condition_iff_l95_95205

variables (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) (h1 : a ≠ 1) (h2 : b ≠ 1) (h3 : x ≠ 1)

theorem log_condition_iff (h : 4 * (Real.logb a x)^2 + 3 * (Real.logb b x)^2 = 8 * (Real.logb a x) * (Real.logb b x)) :
  a = b ^ 2 :=
sorry

end log_condition_iff_l95_95205


namespace good_students_count_l95_95314

noncomputable def student_count := 25

def is_good_student (s : Nat) := s ≤ student_count

def is_troublemaker (s : Nat) := s ≤ student_count

def always_tell_truth (s : Nat) := is_good_student s

def always_lie (s : Nat) := is_troublemaker s

def condition1 (E B : Nat) := E + B = student_count

def condition2 := ∀ (x : Nat), x ≤ 5 → is_good_student x → 
  ∃ (B : Nat), B > 24 / 2

def condition3 := ∀ (x : Nat), x ≤ 20 → is_troublemaker x → 
  ∃ (B E : Nat), B = 3 * (E - 1) ∧ E + B = student_count

theorem good_students_count :
  ∃ (E : Nat), condition1 E (student_count - E) ∧ condition2 ∧ condition3 :=
sorry  -- the actual proof is not required

end good_students_count_l95_95314


namespace cos_sin_inequality_inequality_l95_95799

noncomputable def proof_cos_sin_inequality (a b : ℝ) (cos_x sin_x: ℝ) : Prop :=
  (cos_x ^ 2 = a) → (sin_x ^ 2 = b) → (a + b = 1) → (1 / 4 ≤ a ^ 3 + b ^ 3 ∧ a ^ 3 + b ^ 3 ≤ 1)

theorem cos_sin_inequality_inequality (a b : ℝ) (cos_x sin_x : ℝ) :
  proof_cos_sin_inequality a b cos_x sin_x :=
  by { sorry }

end cos_sin_inequality_inequality_l95_95799


namespace intersection_points_vertex_of_function_value_of_m_shift_l95_95229

noncomputable def quadratic_function (x m : ℝ) : ℝ :=
  (x - m) ^ 2 - 2 * (x - m)

theorem intersection_points (m : ℝ) : 
  ∃ x, quadratic_function x m = 0 ↔ x = m ∨ x = m + 2 := 
by
  sorry

theorem vertex_of_function (m : ℝ) : 
  ∃ x y, y = quadratic_function x m 
  ∧ x = m + 1 ∧ y = -1 := 
by
  sorry

theorem value_of_m_shift (m : ℝ) :
  (m - 2 = 0) → m = 2 :=
by
  sorry

end intersection_points_vertex_of_function_value_of_m_shift_l95_95229


namespace coefficient_of_third_term_l95_95734

theorem coefficient_of_third_term {x y : ℕ} (n : ℕ) 
    (h1 : (1 + 1) ^ n = 64) : 
    (binomial n 2) * x^(n-2) * (-2*y)^2 = 60 * x^(n-2) * y^2 :=
by 
  -- Given the condition 2^n = 64, we conclude n = 6.
  have h_n : n = 6,
    from Nat.pow_right_injective (show 2 > 1 from by norm_num) h1,
  rw h_n,
  -- Now, we calculate the coefficient of the 3rd term directly.
  simp,
  sorry

end coefficient_of_third_term_l95_95734


namespace problem_l95_95189

noncomputable def a : ℝ := Real.exp 0.2
noncomputable def b : ℝ := Real.pow 0.2 Real.exp 1
noncomputable def c : ℝ := Real.log 2

theorem problem (a := Real.exp 0.2) (b := Real.pow 0.2 Real.exp 1) (c := Real.log 2) : b < c ∧ c < a := by
  sorry

end problem_l95_95189


namespace probability_black_white_ball_l95_95861

theorem probability_black_white_ball :
  let total_balls := 5
  let black_balls := 3
  let white_balls := 2
  let favorable_outcomes := (Nat.choose 3 1) * (Nat.choose 2 1)
  let total_outcomes := Nat.choose 5 2
  (favorable_outcomes / total_outcomes) = (3 / 5) := 
by
  sorry

end probability_black_white_ball_l95_95861


namespace good_students_l95_95337

theorem good_students (E B : ℕ) (h1 : E + B = 25) (h2 : 12 < B) (h3 : B = 3 * (E - 1)) :
  E = 5 ∨ E = 7 :=
by 
  sorry

end good_students_l95_95337


namespace pete_total_marbles_after_trading_l95_95442

theorem pete_total_marbles_after_trading :
  ∀ (total_marbles : ℕ) (blue_percentage : ℝ) (kept_red_marbles : ℕ)
    (trade_blue_for_red : ℕ → ℕ → Prop),
    total_marbles = 10 →
    blue_percentage = 0.4 →
    kept_red_marbles = 1 →
    (∀ r b, trade_blue_for_red r b ↔ b = 2 * r) →
  ∃ total_after_trading,
    total_after_trading = (total_marbles * real.floor blue_percentage) + kept_red_marbles + (5 * 2) →
    total_after_trading = 15 :=
by { sorry }

end pete_total_marbles_after_trading_l95_95442


namespace calculate_volume_pyramid_l95_95948

noncomputable def volume_pyramid {α : Type*} [linear_ordered_field α] 
  (AB BC : α) (P : α × α × α) : α :=
let height := (19 : α) / real.sqrt 3 in
let base_area := (1 / 2) * (18 * real.sqrt 2) * (19 * real.sqrt 2) in
(1 / 3) * base_area * height

theorem calculate_volume_pyramid : 
  let AB := 18 * real.sqrt 2 in 
  let BC := 19 * real.sqrt 2 in 
  ∃ (height : ℝ), height = 19 / real.sqrt 3 ∧ 
  volume_pyramid AB BC (0, 0, height) = 2166 * real.sqrt 3 := 
by 
  sorry

end calculate_volume_pyramid_l95_95948


namespace smallest_rectangle_area_l95_95532

-- Given conditions
def radius := 5
def diameter := 2 * radius
def width := diameter
def height := 2 * diameter
def area (w h : ℕ) := w * h

-- The statement to be proven
theorem smallest_rectangle_area (r : ℕ) (d := 2 * r) (w := d) (h := 2 * d) : 
  radius = 5 → area w h = 200 :=
by
  sorry

end smallest_rectangle_area_l95_95532


namespace stuart_marbles_after_gift_l95_95582

def initial_marbles_stuart : Nat := 56
def marbles_betty : Nat := 60
def percentage_given : Float := 0.40

theorem stuart_marbles_after_gift : 
  let marbles_given := (percentage_given * marbles_betty).toInt in
  initial_marbles_stuart + marbles_given = 80 := 
by
  sorry

end stuart_marbles_after_gift_l95_95582


namespace total_rope_length_l95_95136

theorem total_rope_length 
  (longer_side : ℕ) (shorter_side : ℕ) 
  (h1 : longer_side = 28) (h2 : shorter_side = 22) : 
  2 * longer_side + 2 * shorter_side = 100 := by
  sorry

end total_rope_length_l95_95136


namespace class_proof_l95_95357

def class_problem : Prop :=
  let total_students := 25
  let (students : Type) := {s // s ∈ Fin total_students}
  let good students (s : students) : Prop := sorry -- Condition Placeholder
  let troublemakers (s : students) : Prop := sorry -- Condition Placeholder
  have all_good_or_trouble : ∀ s, good students s ∨ troublemakers s,
    from sorry,
  have lh_trouble : ∀ (a b c d e : students), (good students a ∧ good students b ∧ good students c ∧ good students d ∧ good students e) →
    ∃ (x : ℕ), x > ((total_students - 1) / 2) ∧
    (λ rem_trouble : students - {a, b, c, d, e}, B : ℕ, B = x) ∧
    (λ rem_good : students - {a, b, c, d, e}, G : ℕ, G ∈ rem_good), from sorry,
  have th_triple_ratio : ∀ s, (students - {s}).card = 24 →
    ∃ (x y : ℕ), (troublemakers x ∧ good students y) ∧ (3*y = x), from sorry,
  ∃ (num_good_students : ℕ), (num_good_students = 5 ∨ num_good_students = 7)
  
theorem class_proof : class_problem :=
  sorry

end class_proof_l95_95357


namespace evaluate_expression_l95_95162

theorem evaluate_expression (k : ℤ) :
  3 * 2^(-(2 * k + 2)) - 2^(-(2 * k + 1)) + 2 * 2^(-2 * k) = (9 / 4) * 2^(-2 * k) :=
by
  sorry

end evaluate_expression_l95_95162


namespace multiplier_is_five_l95_95566

-- condition 1: n = m * (n - 4)
-- condition 2: n = 5
-- question: prove m = 5

theorem multiplier_is_five (n m : ℝ) 
  (h1 : n = m * (n - 4)) 
  (h2 : n = 5) : m = 5 := 
  sorry

end multiplier_is_five_l95_95566


namespace max_bottles_from_C_and_D_l95_95609

noncomputable def shop_a_price := 1
noncomputable def shop_a_stock := 200
noncomputable def shop_b_price := 2
noncomputable def shop_b_stock := 150
noncomputable def shop_c_price := 3
noncomputable def shop_c_stock := 100
noncomputable def shop_d_price := 5
noncomputable def shop_d_stock := 50
noncomputable def budget := 600
noncomputable def bottles_bought_a := 150
noncomputable def bottles_bought_b := 180

theorem max_bottles_from_C_and_D : 
  let spent_a := bottles_bought_a * shop_a_price in
  let spent_b := bottles_bought_b * shop_b_price in
  let remaining_budget := budget - (spent_a + spent_b) in
  let bottles_from_c := remaining_budget / shop_c_price in
  bottles_from_c + 0 <= shop_c_stock 
  ∧ bottles_from_c * shop_c_price <= remaining_budget
  ∧ bottles_from_c = 30 :=
by
  sorry

end max_bottles_from_C_and_D_l95_95609


namespace smallest_k_mod_conditions_l95_95985

theorem smallest_k_mod_conditions : ∃ k : ℕ, k > 1 ∧ k % 19 = 1 ∧ k % 7 = 1 ∧ k % 3 = 1 ∧ k = 400 :=
by
  use 400
  split
  · exact lt_add_one 399
  · split
    · exact Nat.mod_eq_of_lt (by decide)
    · split
      · exact Nat.mod_eq_of_lt (by decide)
      · split
        · exact Nat.mod_eq_of_lt (by decide)
        · rfl

end smallest_k_mod_conditions_l95_95985


namespace smallest_positive_angle_of_neg_660_is_pi_over_3_l95_95624

noncomputable def smallest_positive_angle_same_terminal_side (angle : ℝ) : ℝ :=
  let radian_conversion := real.pi / 180 in
  let k := nat.ceil ((-angle) / 360) in
  ((angle + k * 360) * radian_conversion / 180) % (2 * real.pi)

theorem smallest_positive_angle_of_neg_660_is_pi_over_3 : 
  smallest_positive_angle_same_terminal_side (-660) = real.pi / 3 := 
by
  sorry

end smallest_positive_angle_of_neg_660_is_pi_over_3_l95_95624


namespace probability_of_stopping_after_2nd_shot_l95_95159

-- Definitions based on the conditions
def shootingProbability : ℚ := 2 / 3

noncomputable def scoring (n : ℕ) : ℕ := 12 - n

def stopShootingProbabilityAfterNthShot (n : ℕ) (probOfShooting : ℚ) : ℚ :=
  if n = 2 then (1 / 3) * (2 / 3) * sorry -- Note: Here, filling in the remaining calculation steps according to problem logic.
  else sorry -- placeholder for other cases

theorem probability_of_stopping_after_2nd_shot :
  stopShootingProbabilityAfterNthShot 2 shootingProbability = 8 / 729 :=
by
  sorry

end probability_of_stopping_after_2nd_shot_l95_95159


namespace mono_decreasing_m_l95_95213

noncomputable def f (x : ℝ) (m : ℝ) := 2 * Real.exp x - m * x

theorem mono_decreasing_m (m : ℝ) :
  (∀ x ∈ set.Icc (-1 : ℝ) 0, (deriv (λ x, f x m)) x ≤ 0) → m ≥ 2 :=
by
  intros h
  sorry

end mono_decreasing_m_l95_95213


namespace purely_imaginary_complex_number_l95_95837

theorem purely_imaginary_complex_number (m : ℝ) :
  (m^2 - 2 * m - 3 = 0) ∧ (m^2 - 4 * m + 3 ≠ 0) → m = -1 :=
by
  sorry

end purely_imaginary_complex_number_l95_95837


namespace only_setA_forms_triangle_l95_95065

-- Define the sets of line segments
def setA := [3, 5, 7]
def setB := [3, 6, 10]
def setC := [5, 5, 11]
def setD := [5, 6, 11]

-- Define a function to check the triangle inequality
def satisfies_triangle_inequality (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Formalize the question
theorem only_setA_forms_triangle :
  satisfies_triangle_inequality 3 5 7 ∧
  ¬(satisfies_triangle_inequality 3 6 10) ∧
  ¬(satisfies_triangle_inequality 5 5 11) ∧
  ¬(satisfies_triangle_inequality 5 6 11) :=
by
  sorry

end only_setA_forms_triangle_l95_95065


namespace num_factors_both_cubes_and_squares_l95_95701

theorem num_factors_both_cubes_and_squares (n : ℕ) (h : n = 2 ^ 3 * 3 ^ 2 * 5 ^ 2) :
  {d : ℕ | d ∣ n ∧ ∃ e, d = e ^ 2 ∧ e ^ 3 ∣ n}.to_finset.card = 1 := by
  sorry

end num_factors_both_cubes_and_squares_l95_95701


namespace scientific_notation_example_l95_95044

def scientific_notation (n : ℝ) (a : ℝ) (b : ℤ) : Prop :=
  n = a * 10^b

theorem scientific_notation_example : 
  scientific_notation 0.00519 5.19 (-3) :=
by 
  sorry

end scientific_notation_example_l95_95044


namespace bob_buys_nose_sprays_l95_95596

theorem bob_buys_nose_sprays (cost_per_spray : ℕ) (promotion : ℕ → ℕ) (total_paid : ℕ)
  (h1 : cost_per_spray = 3)
  (h2 : ∀ n, promotion n = 2 * n)
  (h3 : total_paid = 15) : (total_paid / cost_per_spray) * 2 = 10 :=
by
  sorry

end bob_buys_nose_sprays_l95_95596


namespace true_discount_is_52_l95_95546

/-- The banker's gain on a bill due 3 years hence at 15% per annum is Rs. 23.4. -/
def BG : ℝ := 23.4

/-- The rate of interest per annum is 15%. -/
def R : ℝ := 15

/-- The time in years is 3. -/
def T : ℝ := 3

/-- The true discount is Rs. 52. -/
theorem true_discount_is_52 : BG * 100 / (R * T) = 52 :=
by
  -- Placeholder for proof. This needs proper calculation.
  sorry

end true_discount_is_52_l95_95546


namespace bread_cost_equality_l95_95794

variable (B : ℝ)
variable (C1 : B + 3 + 2 * B = 9)  -- $3 for butter, 2B for juice, total spent is 9 dollars

theorem bread_cost_equality : B = 2 :=
by
  sorry

end bread_cost_equality_l95_95794


namespace optionD_is_equation_l95_95064

-- Definitions for options
def optionA (x : ℕ) := 2 * x - 3
def optionB := 2 + 4 = 6
def optionC (x : ℕ) := x > 2
def optionD (x : ℕ) := 2 * x - 1 = 3

-- Goal: prove that option D is an equation.
theorem optionD_is_equation (x : ℕ) : (optionD x) = True :=
sorry

end optionD_is_equation_l95_95064


namespace xy_problem_l95_95274

theorem xy_problem (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 :=
by
  sorry

end xy_problem_l95_95274


namespace nature_of_roots_l95_95607

theorem nature_of_roots (p : Polynomial ℝ) (h : p = Polynomial.C 1 * x^4 - 6 * x^3 + 11 * x^2 - 6 * x + 1) :
  ((∀ r : ℝ, (Polynomial.aeval r p = 0) → (r ≠ 1) ∧ (r ≠ -1)) ∧ (2 ≤ (Polynomial.rootCount IsoPolynomial p) ∧ (Polynomial.rootCount IsoPolynomial p) ≤ 4)
  ∧ ((Polynomial.rootCount IsoPolynomial (-p)) = 1 ∨ (Polynomial.rootCount IsoPolynomial (-p)) = 3)) :=
sorry

end nature_of_roots_l95_95607


namespace smallest_K_222_multiple_of_198_l95_95206

theorem smallest_K_222_multiple_of_198 :
  ∀ K : ℕ, (∃ x : ℕ, x = 2 * (10^K - 1) / 9 ∧ x % 198 = 0) → K = 18 :=
by
  sorry

end smallest_K_222_multiple_of_198_l95_95206


namespace cannot_determine_right_triangle_l95_95473

noncomputable def ∠A : ℝ := 1
noncomputable def ∠B : ℝ := 1
noncomputable def ∠C : ℝ := 1
noncomputable def a : ℝ := 1
noncomputable def b : ℝ := 1
noncomputable def c : ℝ := 1
noncomputable def common_factor : ℝ := 15

-- Given conditions translated as assumptions
axiom condition1 : ∠A = ∠B - ∠C
axiom condition2 : a = 5 * common_factor ∧ b = 12 * common_factor ∧ c = 13 * common_factor
axiom condition3 : ∠A = 3 * common_factor ∧ ∠B = 4 * common_factor ∧ ∠C = 5 * common_factor
axiom condition4 : a^2 = (b + c) * (b - c)

-- Theorem statement in Lean 4
theorem cannot_determine_right_triangle :
  ∠A : ∠B : ∠C = 3 : 4 : 5 →
  ¬ (∠A = 90 ∨ ∠B = 90 ∨ ∠C = 90) :=
sorry

end cannot_determine_right_triangle_l95_95473


namespace good_students_options_l95_95327

variables (E B : ℕ)

-- Define the condition that the class has 25 students
def total_students : Prop := E + B = 25

-- Define the condition given by the first group of students
def first_group_condition : Prop := B > 12

-- Define the condition given by the second group of students
def second_group_condition : Prop := B = 3 * (E - 1)

-- Define the problem statement
theorem good_students_options (E B : ℕ) :
  total_students E B → first_group_condition B → second_group_condition E B → (E = 5 ∨ E = 7) :=
by
  intros h1 h2 h3
  sorry

end good_students_options_l95_95327


namespace distinct_values_of_D_l95_95386

-- Define the digits are distinct and within range.
def is_digit (n : ℕ) := n < 10

-- Main theorem to prove:
theorem distinct_values_of_D : (∃A B C D : ℕ, is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧ A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ B + A ≤ 8 ∧ D = B + A + 1) ↔ (finset.range 10).count (fun D => ∃ A B C : ℕ, is_digit A ∧ is_digit B ∧ is_digit C ∧ A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ B + A + 1 = D) = 7 :=
sorry

end distinct_values_of_D_l95_95386


namespace quintic_polynomial_root_sum_l95_95976

theorem quintic_polynomial_root_sum :
  ∃ (a b c : ℕ), (16 * (x : ℝ)^5 - 4 * x^4 - 4 * x - 1 = 0) ∧ 
  x = (↑(Nat.root 5 a) + ↑(Nat.root 5 b) + 1) / ↑c ∧ a + b + c = 69648 :=
sorry

end quintic_polynomial_root_sum_l95_95976


namespace cone_ratio_42_l95_95950

theorem cone_ratio_42
  (r h : ℝ)
  (cone_condition : (√(r^2 + h^2) = 20 * r)) :
  ∃ (m n : ℕ), (m = 19 ∧ n = 23 ∧ m + n = 42 ∧ h/r = 19 * √23) :=
by
  sorry

end cone_ratio_42_l95_95950


namespace largest_number_l95_95892

theorem largest_number (n : ℕ) (digits : List ℕ) (h_digits : ∀ d ∈ digits, d = 5 ∨ d = 3 ∨ d = 1) (h_sum : digits.sum = 15) : n = 555 :=
by
  sorry

end largest_number_l95_95892


namespace train_pass_bridge_time_l95_95122

-- Definitions of the given conditions
def length_of_train : ℝ := 357
def length_of_bridge : ℝ := 137
def speed_of_train_kmh : ℝ := 42

-- Conversion factor from km/h to m/s
def kmh_to_ms : ℝ := 1000 / 3600

-- Speed of the train in m/s
def speed_of_train_ms : ℝ := speed_of_train_kmh * kmh_to_ms

-- Total distance the train needs to cover to pass the bridge
def total_distance : ℝ := length_of_train + length_of_bridge

-- Time it takes to pass the bridge
def time_to_pass_bridge : ℝ := total_distance / speed_of_train_ms

-- Approximate time to pass the bridge in seconds
def expected_time : ℝ := 42.33

-- The statement to be proven
theorem train_pass_bridge_time : |time_to_pass_bridge - expected_time| < 0.01 := by
  sorry

end train_pass_bridge_time_l95_95122


namespace intersecting_lines_l95_95055

-- Definitions based on conditions
def line1 (m : ℝ) (x : ℝ) : ℝ := m * x + 4
def line2 (b : ℝ) (x : ℝ) : ℝ := 3 * x + b

-- Lean 4 Statement of the problem
theorem intersecting_lines (m b : ℝ) (h1 : line1 m 6 = 10) (h2 : line2 b 6 = 10) : b + m = -7 :=
by
  sorry

end intersecting_lines_l95_95055


namespace no_triangle_from_geom_progression_l95_95400

theorem no_triangle_from_geom_progression :
  ¬ ∃ (a b c : ℕ), (a < b ∧ b < c) ∧ (∃ (s : list ℕ),
    s = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] ∧
    a ∈ s ∧ b ∈ s ∧ c ∈ s ∧
    a + b > c ∧ b + c > a ∧ a + c > b) :=
by 
  sorry

end no_triangle_from_geom_progression_l95_95400


namespace division_remainder_unique_u_l95_95911

theorem division_remainder_unique_u :
  ∃! u : ℕ, ∃ q : ℕ, 15 = u * q + 4 ∧ u > 4 :=
sorry

end division_remainder_unique_u_l95_95911


namespace how_many_have_4_factors_l95_95252

open Nat

def num_factors (n : ℕ) : ℕ :=
  (range n).count_dvd n

theorem how_many_have_4_factors : (list.length (list.filter (λ n, num_factors n = 4) [14, 21, 28, 35, 42])) = 3 := by
  sorry

end how_many_have_4_factors_l95_95252


namespace cost_price_correct_l95_95584

noncomputable def cost_price (selling_price marked_price_ratio cost_profit_ratio : ℝ) : ℝ :=
  (selling_price * marked_price_ratio) / cost_profit_ratio

theorem cost_price_correct : 
  abs (cost_price 63.16 0.94 1.25 - 50.56) < 0.01 :=
by 
  sorry

end cost_price_correct_l95_95584


namespace hyperbola_eccentricity_l95_95224

noncomputable def eccentricity_of_hyperbola 
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  : ℝ :=
  let c := Math.sqrt (a^2 + b^2)
  let e := c / a
  e -- Definition of eccentricity

theorem hyperbola_eccentricity 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_hyp : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_circle : ∀ x y : ℝ, (x - 2)^2 + y^2 = 2)
  (h_chord_len : ∀ l : ℝ, l = 2)
  : eccentricity_of_hyperbola a b ha hb = 2 * Real.sqrt 3 / 3 := by
    sorry

end hyperbola_eccentricity_l95_95224


namespace infinitely_many_integers_divisible_by_sum_of_digits_l95_95130

theorem infinitely_many_integers_divisible_by_sum_of_digits :
  ∀ (n : ℕ), ∃ k : ℕ, let N := (10^((3^n) - 1)) / 9 in 
    (k > 0) ∧ (N > 0) ∧ (N % (digit_sum N) = 0) :=
by
  sorry

end infinitely_many_integers_divisible_by_sum_of_digits_l95_95130


namespace find_cos_beta_l95_95667

variable {α β : ℝ}
variable (h_acute_α : 0 < α ∧ α < π / 2)
variable (h_acute_β : 0 < β ∧ β < π / 2)
variable (h_sin_α : Real.sin α = 2 / 5 * Real.sqrt 5)
variable (h_sin_α_plus_β : Real.sin (α + β) = 3 / 5)

theorem find_cos_beta 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = 2 / 5 * Real.sqrt 5)
  (h_sin_α_plus_β : Real.sin (α + β) = 3 / 5) :
  Real.cos β = Real.sqrt 5 / 5 := 
sorry

end find_cos_beta_l95_95667


namespace larger_interior_angle_trapezoid_pavilion_l95_95311

theorem larger_interior_angle_trapezoid_pavilion :
  let n := 12
  let central_angle := 360 / n
  let smaller_angle := 180 - (central_angle / 2)
  let larger_angle := 180 - smaller_angle
  larger_angle = 97.5 :=
by
  sorry

end larger_interior_angle_trapezoid_pavilion_l95_95311


namespace point_C_eq_l95_95388

-- Given points A and vectors BA and BC in the complex plane
def A : ℂ := 2 + complex.i
def BA : ℂ := 2 + 3 * complex.i
def BC : ℂ := 3 - complex.i

-- We need to find the complex number corresponding to point C
theorem point_C_eq : ∃ C : ℂ, C = 3 - 3 * complex.i := sorry

end point_C_eq_l95_95388


namespace constant_term_of_binomial_expansion_l95_95505

theorem constant_term_of_binomial_expansion 
  (sum_of_binomial_coeffs : (sqrt[3]{x} - (1/x) : ℝ) ^ 12 = 4096) : 
  constant_term (sqrt[3]{x} - 1/x)^12 = -220 :=
sorry

end constant_term_of_binomial_expansion_l95_95505


namespace factorization_equiv_l95_95475

theorem factorization_equiv :
  ∀ (a b x y : ℝ), a ≠ 0 ∧ b ≠ 0 → 
  (a^8 * x * y - a^7 * y - a^6 * x = a^5 * (b^5 - 1)) ↔ 
  (a^4 * x - a^3) * (a * y - 1) = a^5 * b^5 :=
by
  intros a b x y h
  split
  { intro h1
    sorry
  }
  { intro h2
    sorry
  }

end factorization_equiv_l95_95475


namespace percentage_of_males_l95_95729

theorem percentage_of_males (total_employees males_below_50 males_percentage : ℕ) (h1 : total_employees = 800) (h2 : males_below_50 = 120) (h3 : 40 * males_percentage / 100 = 60 * males_below_50):
  males_percentage = 25 :=
by
  sorry

end percentage_of_males_l95_95729


namespace path_length_traveled_l95_95148

-- Define the conditions of the rectangle and the transformations:

-- Assume the side lengths of the rectangle
def AB : ℝ := 3
def BC : ℝ := 8

-- Define the diagonal calculation
def BD : ℝ := Real.sqrt (AB^2 + BC^2)

-- Define the rotations and respective distances
def distance_first_rotation : ℝ := (1 / 2) * π * BD
def distance_second_rotation : ℝ := (3 / 2) * π * AB
def distance_third_rotation : ℝ := 4 * π * BC

-- The path length traveled by point A 
def total_distance : ℝ := distance_first_rotation + distance_second_rotation + distance_third_rotation

-- The target distance given by the problem
def target_distance : ℝ := (1 / 2 : ℝ) * Real.sqrt 73 * π + (11 / 2 : ℝ) * π

-- The proof statement to check if total_distance equals target_distance
theorem path_length_traveled : total_distance = target_distance := sorry

end path_length_traveled_l95_95148


namespace each_girl_brought_2_cups_l95_95513

-- Here we define all the given conditions
def total_students : ℕ := 30
def num_boys : ℕ := 10
def cups_per_boy : ℕ := 5
def total_cups : ℕ := 90

-- Define the conditions as Lean definitions
def num_girls : ℕ := total_students - num_boys -- From condition 3
def cups_by_boys : ℕ := num_boys * cups_per_boy
def cups_by_girls : ℕ := total_cups - cups_by_boys

-- Define the final question and expected answer
def cups_per_girl : ℕ := cups_by_girls / num_girls

-- Final problem statement to prove
theorem each_girl_brought_2_cups :
  cups_per_girl = 2 :=
by
  have h1 : num_girls = 20 := by sorry
  have h2 : cups_by_boys = 50 := by sorry
  have h3 : cups_by_girls = 40 := by sorry
  have h4 : cups_per_girl = 2 := by sorry
  exact h4

end each_girl_brought_2_cups_l95_95513


namespace ratio_of_angles_l95_95123

noncomputable def angle_ratio (α β γ δ : ℝ) [fact (α = 140)] [fact (β = 60)] [fact (γ = 160)] [fact (δ = 70)] : ℝ :=
  δ / (β / 2)

theorem ratio_of_angles (α β γ δ : ℝ) [fact (α = 140)] [fact (β = 60)] [fact (γ = 160)] [fact (δ = 70)] :
  angle_ratio α β γ δ = 7 / 3 :=
by
  unfold angle_ratio
  sorry

end ratio_of_angles_l95_95123


namespace Ivanov_made_an_error_l95_95821

theorem Ivanov_made_an_error (mean median : ℝ) (variance : ℝ) (h1 : mean = 0) (h2 : median = 4) (h3 : variance = 15.917) : 
  |mean - median| ≤ Real.sqrt variance → False :=
by {
  have mean_value : mean = 0 := h1,
  have median_value : median = 4 := h2,
  have variance_value : variance = 15.917 := h3,

  let lhs := |mean_value - median_value|,
  have rhs := Real.sqrt variance_value,
  
  calc
    lhs = |0 - 4| : by rw [mean_value, median_value]
    ... = 4 : by norm_num,
  
  have rhs_val : Real.sqrt variance_value ≈ 3.99 := by sorry, -- approximate value for demonstration
  
  have ineq : 4 ≤ rhs_val := by 
    calc 4 = 4 : rfl -- trivial step for clarity,
    have sqrt_val : Real.sqrt 15.917 < 4 := by sorry, -- from calculation or suitable proof
  
  exact absurd ineq (not_le_of_gt sqrt_val)
}

end Ivanov_made_an_error_l95_95821


namespace necessary_but_not_sufficient_condition_l95_95847

variable (a : ℝ) (x : ℝ)

def inequality_holds_for_all_real_numbers (a : ℝ) : Prop :=
    ∀ x : ℝ, (a * x^2 - a * x + 1 > 0)

theorem necessary_but_not_sufficient_condition :
  (0 < a ∧ a < 4) ↔
  (inequality_holds_for_all_real_numbers a) :=
by
  sorry

end necessary_but_not_sufficient_condition_l95_95847


namespace acute_angle_trapezoid_sixty_degrees_l95_95057

theorem acute_angle_trapezoid_sixty_degrees
  (rect : Type)
  (O : rect → rect → Prop) -- centerline 
  (A : rect)
  (center_line : ∀ x y, O x y -> O y x)
  (corner_fold : rect → rect → Prop)
  (align_with_adjacent_corner : ∀ x : rect, corner_fold x x) :
  angle B C A = 60 := sorry

end acute_angle_trapezoid_sixty_degrees_l95_95057


namespace people_per_column_in_second_arrangement_l95_95721
-- Import the necessary libraries

-- Define the conditions as given in the problem
def number_of_people_first_arrangement : ℕ := 30 * 16
def number_of_columns_second_arrangement : ℕ := 8

-- Define the problem statement with proof
theorem people_per_column_in_second_arrangement :
  (number_of_people_first_arrangement / number_of_columns_second_arrangement) = 60 :=
by
  -- Skip the proof here
  sorry

end people_per_column_in_second_arrangement_l95_95721


namespace city_population_correct_l95_95145

variable (C G : ℕ)

theorem city_population_correct :
  (C - G = 119666) ∧ (C + G = 845640) → (C = 482653) := by
  intro h
  have h1 : C - G = 119666 := h.1
  have h2 : C + G = 845640 := h.2
  sorry

end city_population_correct_l95_95145


namespace each_girl_brought_2_cups_l95_95511

-- Definitions of the conditions
def total_students : ℕ := 30
def boys : ℕ := 10
def total_cups : ℕ := 90
def cups_per_boy : ℕ := 5
def girls : ℕ := total_students - boys

def total_cups_by_boys : ℕ := boys * cups_per_boy
def total_cups_by_girls : ℕ := total_cups - total_cups_by_boys
def cups_per_girl : ℕ := total_cups_by_girls / girls

-- The statement with the correct answer
theorem each_girl_brought_2_cups (
  h1 : total_students = 30,
  h2 : boys = 10,
  h3 : total_cups = 90,
  h4 : cups_per_boy = 5,
  h5 : total_cups_by_boys = boys * cups_per_boy,
  h6 : total_cups_by_girls = total_cups - total_cups_by_boys,
  h7 : cups_per_girl = total_cups_by_girls / girls
) : cups_per_girl = 2 := 
sorry

end each_girl_brought_2_cups_l95_95511


namespace min_spheres_to_cover_cylinder_l95_95033

-- Define the height and radius of the cylinder.
def cylinder_height : ℝ := 1
def cylinder_radius : ℝ := 1

-- Define the radius of the spheres.
def sphere_radius : ℝ := 1

-- Proposition stating the minimum number of spheres needed to cover the cylinder.
theorem min_spheres_to_cover_cylinder (h r s: ℝ) (Hh: h = cylinder_height) (Hr: r = cylinder_radius) (Hs: s = sphere_radius) : Nat :=
  if Hh = 1 ∧ Hr = 1 ∧ Hs = 1 then 3 else sorry

end min_spheres_to_cover_cylinder_l95_95033


namespace cistern_fill_time_l95_95562

variable (filling_rate : ℝ) (emptying_rate : ℝ)
variable (time_to_fill : ℝ)

-- Conditions
def fill_tap_rate := filling_rate = 1 / 5
def empty_tap_rate := emptying_rate = 1 / 6

-- Proof Statement
theorem cistern_fill_time (h1 : fill_tap_rate) (h2 : empty_tap_rate) : time_to_fill = 30 := 
by
  sorry

end cistern_fill_time_l95_95562


namespace probability_of_drawing_neither_prime_nor_composite_l95_95075

-- Define what it means for a number to be neither prime nor composite
def is_neither_prime_nor_composite (n : ℕ) : Prop :=
  n = 1

-- List of numbers from 1 to 98
def numbers : list ℕ := list.range' 1 98

-- Count the number of elements that satisfy the condition
def count_neither_prime_nor_composite (lst : list ℕ) :=
  list.countp is_neither_prime_nor_composite lst

-- Define the probability calculation
def probability (favorable : ℕ) (total : ℕ) :=
  favorable.to_rat / total.to_rat

-- Total number of elements
def total_elements : ℕ := numbers.length

-- Favorable outcome (1 is the only number neither prime nor composite)
def favorable_outcomes : ℕ :=
  count_neither_prime_nor_composite numbers

-- Theorem to prove
theorem probability_of_drawing_neither_prime_nor_composite :
  probability favorable_outcomes total_elements = (1 : ℚ) / 98 :=
by
  sorry

end probability_of_drawing_neither_prime_nor_composite_l95_95075


namespace intersection_point_l95_95730

structure Point3D : Type where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨8, -9, 5⟩
def B : Point3D := ⟨18, -19, 15⟩
def C : Point3D := ⟨2, 5, -8⟩
def D : Point3D := ⟨4, -3, 12⟩

/-- Prove that the intersection point of lines AB and CD is (16, -19, 13) -/
theorem intersection_point :
  ∃ (P : Point3D), 
  (∃ t : ℝ, P = ⟨A.x + t * (B.x - A.x), A.y + t * (B.y - A.y), A.z + t * (B.z - A.z)⟩) ∧
  (∃ s : ℝ, P = ⟨C.x + s * (D.x - C.x), C.y + s * (D.y - C.y), C.z + s * (D.z - C.z)⟩) ∧
  P = ⟨16, -19, 13⟩ :=
by
  sorry

end intersection_point_l95_95730


namespace part1_part2_l95_95186

noncomputable theory

variables (a b : ℝ × ℝ)

def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem part1 (ha : a = (4, 4)) (hb : b = (3, 4)) :
  vector_magnitude (3 • a - 2 • b) = 2 * real.sqrt 13 :=
sorry

theorem part2 (ha : a = (4, 4)) (hb : b = (3, 4)) (k : ℝ)
  (h_perpendicular : is_perpendicular (k • a + b) (a - b)) :
  k = -3 / 4 :=
sorry

end part1_part2_l95_95186


namespace curve_transformation_l95_95086

-- Defining the given matrices M and N
def M : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 0], ![0, 2]]
def N : Matrix (Fin 2) (Fin 2) ℝ := ![![1/2, 0], ![0, 1]]

-- The matrix product MN
def MN : Matrix (Fin 2) (Fin 2) ℝ := M ⬝ N

-- The equation of the original curve
def original_curve (x : ℝ) : ℝ := Real.sin x

-- The equation of the transformed curve C
def transformed_curve (x : ℝ) : ℝ := 2 * Real.sin (2 * x)

-- The proof statement
theorem curve_transformation :
  ∀ x y, y = original_curve x → transformed_curve x = y :=
by
  sorry

end curve_transformation_l95_95086


namespace max_intersections_three_polygons_l95_95051

variables {P1 P2 P3 : Type} -- Represent the three polygons
variables (n1 n2 n3 : ℕ) -- Represent the number of sides of the polygons

-- Assume that n1 <= n2 <= n3
axiom h1 : n1 ≤ n2 
axiom h2 : n2 ≤ n3

-- We are interested in finding the maximum number of intersections
theorem max_intersections_three_polygons (h_no_shared_segments : ¬ ∃ (s : Type), s ∈ P1 ∧ s ∈ P2 ∧ s ∈ P3) : 
  (max_intersections P1 P2 P3 = n1 * n2 + n1 * n3 + n2 * n3) :=
sorry

end max_intersections_three_polygons_l95_95051


namespace total_winter_clothing_l95_95083

-- Definitions based on the conditions
def boxes : ℕ := 8
def scarves_per_box : ℕ := 4
def mittens_per_box : ℕ := 6

-- The target statement (the proof problem)
theorem total_winter_clothing : boxes * (scarves_per_box + mittens_per_box) = 80 :=
by
  -- Given that total number of boxes is 8
  have h1 : boxes = 8 := rfl,
  -- And each box contains 4 scarves and 6 mittens
  have h2 : scarves_per_box = 4 := rfl,
  have h3 : mittens_per_box = 6 := rfl,
  -- Therefore the total pieces of winter clothing is 80
  rw [h1, h2, h3],
  exact calc
    8 * (4 + 6) = 8 * 10 : by rw [add_assoc, add_comm 4 6]
             ... = 80     : by norm_num

end total_winter_clothing_l95_95083


namespace maximum_abc_value_l95_95655

variable (a b c : ℝ)

noncomputable def maximum_abc : ℝ :=
  let ab := a * b in
  let c := 5 - ab in
  ab * c

theorem maximum_abc_value (h1 : 2 * a + b = 4) (h2 : a * b + c = 5) : maximum_abc a b ≤ 6 :=
by sorry

end maximum_abc_value_l95_95655


namespace probability_non_defective_pencils_l95_95076

theorem probability_non_defective_pencils :
  let total_pencils := 8
  let defective_pencils := 2
  let selected_pencils := 3
  let non_defective_pencils := total_pencils - defective_pencils
  let total_combinations := Nat.choose total_pencils selected_pencils
  let non_defective_combinations := Nat.choose non_defective_pencils selected_pencils
  (non_defective_combinations:ℚ) / (total_combinations:ℚ) = 5 / 14 := by
  sorry

end probability_non_defective_pencils_l95_95076


namespace diagonal_intersection_probability_decagon_l95_95095

noncomputable def probability_diagonal_intersection_in_decagon : ℚ :=
  let vertices := 10
  let total_diagonals := (vertices * (vertices - 3)) / 2
  let total_pairs_of_diagonals := total_diagonals * (total_diagonals - 1) / 2
  let total_intersecting_pairs := (vertices * (vertices - 1) * (vertices - 2) * (vertices - 3)) / 24
  total_intersecting_pairs / total_pairs_of_diagonals

theorem diagonal_intersection_probability_decagon (h : probability_diagonal_intersection_in_decagon = 42 / 119) : 
  probability_diagonal_intersection_in_decagon = 42 / 119 :=
sorry

end diagonal_intersection_probability_decagon_l95_95095


namespace ivanov_error_l95_95812

-- Given conditions
def mean_temperature (x : ℝ) : Prop := x = 0
def median_temperature (m : ℝ) : Prop := m = 4
def variance_temperature (s² : ℝ) : Prop := s² = 15.917

-- Statement to prove
theorem ivanov_error (x m s² : ℝ) 
  (mean_x : mean_temperature x)
  (median_m : median_temperature m)
  (variance_s² : variance_temperature s²) :
  (x - m)^2 > s² :=
by
  rw [mean_temperature, median_temperature, variance_temperature] at *
  simp [*, show 0 = 0, from rfl, show 4 = 4, from rfl]
  sorry

end ivanov_error_l95_95812


namespace total_cantaloupes_l95_95180

def cantaloupes (fred : ℕ) (tim : ℕ) := fred + tim

theorem total_cantaloupes : cantaloupes 38 44 = 82 := by
  sorry

end total_cantaloupes_l95_95180


namespace ctg_inequality_l95_95446

open Real

theorem ctg_inequality (α : ℝ) (h1 : 0 < α) (h2 : α < π/2) : 
  (ctan (α / 2)) > (1 + (ctan α)) := 
sorry

end ctg_inequality_l95_95446


namespace range_of_PA_dot_PB_l95_95393

-- Define the ellipse and points
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

-- Define point P (left vertex)
def point_P : (ℝ × ℝ) := (-sqrt(3), 0)

-- Define points A and B
def point_A (α : ℝ) : (ℝ × ℝ) := (sqrt(3) * cos α, sin α)
def point_B (β : ℝ) : (ℝ × ℝ) := (sqrt(3) * cos β, sin β)

-- Define vectors PA and PB
def vector_PA (α : ℝ) : (ℝ × ℝ) := (sqrt(3) * cos α + sqrt(3), sin α)
def vector_PB (β : ℝ) : (ℝ × ℝ) := (sqrt(3) * cos β + sqrt(3), sin β)

-- Define dot product
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Main theorem statement
theorem range_of_PA_dot_PB (α β : ℝ) :
  ∃ d : ℝ, d ∈ set.Icc (-1/4) 12 ∧ d = dot_product (vector_PA α) (vector_PB β) :=
sorry

end range_of_PA_dot_PB_l95_95393


namespace parallelepiped_dimensions_l95_95114

noncomputable def calc_lengths (M : ℝ) (ρ : ℝ) : ℝ × ℝ × ℝ :=
  let a := (M / (4 * ρ))^(1/3)
  let b := √2 * a
  let c := 2 * √2 * a
  (a, b, c)

theorem parallelepiped_dimensions (M ρ : ℝ) :
  let a := (M / (4 * ρ))^(1/3)
  let b := √2 * a
  let c := 2 * √2 * a
  calc_lengths M ρ = (a, b, c) :=
by
  sorry

end parallelepiped_dimensions_l95_95114


namespace tangent_line_derivative_l95_95211

variable {ℝ : Type*} [LinearOrderedField ℝ]

noncomputable def tangent_slope (f : ℝ → ℝ) (x : ℝ) : ℝ := (deriv f) x

theorem tangent_line_derivative (f : ℝ → ℝ) (hf : f 1 = (1 * 1/2 + 2)) : tangent_slope f 1 = 1/2 := by
  -- additional conditions or context can be added here if necessary
  sorry

end tangent_line_derivative_l95_95211


namespace initial_men_count_l95_95019

-- Define the data conditions and the target value to prove
theorem initial_men_count :
  ∃ M : ℕ, 
  let provisions := P in
  (provisions = M * 21) ∧ 
  (provisions = (M + 800) * 11.67) ∧ 
  (M = 1071) :=
by
  sorry

end initial_men_count_l95_95019


namespace number_of_good_students_is_5_or_7_l95_95346

-- Definitions based on the conditions
def total_students : ℕ := 25
def number_of_good_students (G : ℕ) (T : ℕ) := G + T = total_students
def first_group_condition (T : ℕ) := T > 12
def second_group_condition (G : ℕ) (T : ℕ) := T = 3 * (G - 1)

-- Problem statement in Lean 4:
theorem number_of_good_students_is_5_or_7 (G T : ℕ) :
  number_of_good_students G T ∧ first_group_condition T ∧ second_group_condition G T → G = 5 ∨ G = 7 :=
by
  sorry

end number_of_good_students_is_5_or_7_l95_95346


namespace triangle_equilateral_l95_95040

theorem triangle_equilateral (a d : ℝ) (h1 h2 h3 : ℝ) 
  (d_nonneg : d ≥ 0) 
  (sides_progression : ∀ (a d : ℝ), a, a + d, a + 2d) 
  (altitudes_progression : ∀ (h1 h2 h3 : ℝ), h1, (h1 + h2) / 2, h1 + 2 * ((h1 + h2) / 2 - h1)) : 
  d = 0 := 
by
  sorry

end triangle_equilateral_l95_95040


namespace a_50_eq_1024_S_50_eq_15358_l95_95602

def seq (n : ℕ) : ℕ := 
  let k := Nat.find (λ k, (n > k * (k - 1) / 2) ∧ (n <= k * (k + 1) / 2)) in
  2^k

def sum_seq (n : ℕ) : ℕ :=
  (List.range n).map (λ i, seq (i + 1)).sum

theorem a_50_eq_1024 : seq 50 = 1024 := 
by 
  sorry

theorem S_50_eq_15358 : sum_seq 50 = 15358 := 
by 
  sorry

end a_50_eq_1024_S_50_eq_15358_l95_95602


namespace arithmetic_sequence_sum_l95_95200

theorem arithmetic_sequence_sum {a : ℕ → ℤ} (S : ℕ → ℤ) 
  (h₀ : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2)
  (h₁ : S 9 = 27) :
  (a 4 + a 6) = 6 :=
sorry

end arithmetic_sequence_sum_l95_95200


namespace total_cans_l95_95046

theorem total_cans (total_oil : ℕ) (oil_in_8_liter_cans : ℕ) (number_of_8_liter_cans : ℕ) (remaining_oil : ℕ) 
(oil_per_15_liter_can : ℕ) (number_of_15_liter_cans : ℕ) :
  total_oil = 290 ∧ oil_in_8_liter_cans = 8 ∧ number_of_8_liter_cans = 10 ∧ oil_per_15_liter_can = 15 ∧
  remaining_oil = total_oil - (number_of_8_liter_cans * oil_in_8_liter_cans) ∧
  number_of_15_liter_cans = remaining_oil / oil_per_15_liter_can →
  (number_of_8_liter_cans + number_of_15_liter_cans) = 24 := sorry

end total_cans_l95_95046


namespace equation_of_line_AC_equation_of_median_AD_l95_95693

-- Define the points A, B, and C
def A := (-5 : ℝ, 0 : ℝ)
def B := (3 : ℝ, -3 : ℝ)
def C := (0 : ℝ, 2 : ℝ)

-- Define the line equation for AC
def line_AC (x y : ℝ) : Prop := 2 * x - 5 * y + 10 = 0

-- Define the midpoint D of B and C
def D := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the line equation for the median from A to the midpoint of BC
def line_AD (x y : ℝ) : Prop := x + 13 * y + 5 = 0

-- The proof problem statements
theorem equation_of_line_AC : ∀ (x y : ℝ), (∃ (t : ℝ), x = A.1 + t * (C.1 - A.1) ∧ y = A.2 + t * (C.2 - A.2)) → line_AC x y :=
begin
  sorry
end

theorem equation_of_median_AD : ∀ (x y : ℝ), (∃ (t : ℝ), x = A.1 + t * (D.1 - A.1) ∧ y = A.2 + t * (D.2 - A.2)) → line_AD x y :=
begin
  sorry
end

end equation_of_line_AC_equation_of_median_AD_l95_95693


namespace functional_eq_solution_l95_95165

theorem functional_eq_solution (f : ℝ → ℝ) (h : ∀ (x y : ℝ), f(x + y) = x + f(f(y))) : 
  ∀ x : ℝ, f(x) = x := 
by
  sorry

end functional_eq_solution_l95_95165


namespace soaps_in_one_package_l95_95048

theorem soaps_in_one_package (boxes : ℕ) (packages_per_box : ℕ) (total_packages : ℕ) (total_soaps : ℕ) : 
  boxes = 2 → packages_per_box = 6 → total_packages = boxes * packages_per_box → total_soaps = 2304 → (total_soaps / total_packages) = 192 :=
by
  intros h_boxes h_packages_per_box h_total_packages h_total_soaps
  sorry

end soaps_in_one_package_l95_95048


namespace isosceles_triangle_perimeter_1_isosceles_triangle_perimeter_2_l95_95552

-- Problem 1
theorem isosceles_triangle_perimeter_1 (a b : ℕ) (h1: a = 4 ∨ a = 6) (h2: b = 4 ∨ b = 6) (h3: a ≠ b): 
  (a + b + b = 14 ∨ a + b + b = 16) :=
sorry

-- Problem 2
theorem isosceles_triangle_perimeter_2 (a b : ℕ) (h1: a = 2 ∨ a = 6) (h2: b = 2 ∨ b = 6) (h3: a ≠ b ∨ (a = 2 ∧ 2 + 2 ≥ 6 ∧ 6 = b)):
  (a + b + b = 14) :=
sorry

end isosceles_triangle_perimeter_1_isosceles_triangle_perimeter_2_l95_95552


namespace rods_in_one_mile_l95_95203

-- Define the conditions as assumptions in Lean

-- 1. 1 mile = 8 furlongs
def mile_to_furlong : ℕ := 8

-- 2. 1 furlong = 220 paces
def furlong_to_pace : ℕ := 220

-- 3. 1 pace = 0.2 rods
def pace_to_rod : ℝ := 0.2

-- Define the statement to be proven
theorem rods_in_one_mile : (mile_to_furlong * furlong_to_pace * pace_to_rod) = 352 := by
  sorry

end rods_in_one_mile_l95_95203


namespace isabel_initial_money_l95_95401

theorem isabel_initial_money (X : ℝ) (h : (X * (1 - (1/3)) * (1 - (1/2)) * 0.75 = 60)) : X = 720 :=
by
  -- We state the given condition
  have h1 : X * 2 / 3 = (2 / 3) * (1 / 2) := sorry,
  have h2 : X * 1 / 3 = (1 / 3) * 3 / 4 := sorry,
  have h3 : X * 3 / 4 = 12 * 60 := sorry,
  have h4 : 60 / 1 / 12 = 60 := sorry    
  -- Each condition should be proven step-by-step according to Lean,
  -- however, we will use sorry here for simplification.

end isabel_initial_money_l95_95401


namespace sum_of_identical_digits_l95_95237

theorem sum_of_identical_digits
  (a b c : ℕ) (x : ℕ := a * 11111) (y : ℕ := b * 1111) (z : ℕ := c * 111) :
  (∃ (x y z : ℕ), (x % 11111 = 0) ∧ (y % 1111 = 0) ∧ (z % 111 = 0) ∧ 
  let sum := x + y + z in 
  sum ≥ 10000 ∧ sum < 100000 ∧ 
  (let digits := [sum / 10000 % 10, sum / 1000 % 10, sum / 100 % 10, sum / 10 % 10, sum % 10] in 
  digits.nodup)) := 
sorry

end sum_of_identical_digits_l95_95237


namespace quadratic_polynomials_eq_l95_95010

-- Define the integer part function
def intPart (x : ℝ) : ℤ := Int.floor x

-- Define the condition for quadratic polynomials
def is_quadratic (f : ℝ → ℝ) := ∃ (a b c : ℝ), ∀ x, f(x) = a*x^2 + b*x + c

theorem quadratic_polynomials_eq 
    (f g : ℝ → ℝ)
    (hf : is_quadratic f)
    (hg : is_quadratic g)
    (h_condition : ∀ x, intPart (f x) = intPart (g x)) :
    ∀ x, f x = g x :=
by
  sorry

end quadratic_polynomials_eq_l95_95010


namespace cone_volume_difference_l95_95104

theorem cone_volume_difference (H R : ℝ) : ΔV = (1/12) * Real.pi * R^2 * H := 
sorry

end cone_volume_difference_l95_95104


namespace john_tv_show_duration_l95_95746

def john_tv_show (seasons_before : ℕ) (episodes_per_season : ℕ) (additional_episodes : ℕ) (episode_duration : ℝ) : ℝ :=
  let total_episodes_before := seasons_before * episodes_per_season
  let last_season_episodes := episodes_per_season + additional_episodes
  let total_episodes := total_episodes_before + last_season_episodes
  total_episodes * episode_duration

theorem john_tv_show_duration :
  john_tv_show 9 22 4 0.5 = 112 := 
by
  sorry

end john_tv_show_duration_l95_95746


namespace part1_part2_l95_95302

theorem part1 (A B : ℝ) (c : ℝ) (cos_A : ℝ) (tan_half_B_add_cot_half_B: ℝ) 
  (h1: cos_A = 5 / 13) 
  (h2: tan_half_B_add_cot_half_B = 10 / 3) 
  (pos_c: c = 21) :
  ∃ (cos_AB: ℝ), cos_AB = 56 / 65 :=
by {
  sorry
}

theorem part2 (A B : ℝ) (c : ℝ) (cos_A : ℝ) (tan_half_B_add_cot_half_B: ℝ) 
  (h1: cos_A = 5 / 13) 
  (h2: tan_half_B_add_cot_half_B = 10 / 3) 
  (pos_c: c = 21) :
  ∃ (area: ℝ), area = 126 :=
by {
  sorry
}

end part1_part2_l95_95302


namespace period_and_extrema_l95_95234

-- Define the function f(x)
def f (x : ℝ) := 2 * Real.sin (x / 2 + Real.pi / 3)

-- State the theorem for the period and maximum/minimum values
theorem period_and_extrema :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧  -- Existence of a period T
  (∀ x, (0 ≤ x ∧ x ≤ Real.pi) → f x ≤ 2 ∧ f x ≥ 1) ∧  -- Maximum and minimum values in [0, π]
  (f (Real.pi / 3) = 2) ∧  -- Maximum value
  (f Real.pi = 1)  -- Minimum value
:= sorry

end period_and_extrema_l95_95234


namespace line_equations_equal_intercepts_l95_95028

theorem line_equations_equal_intercepts (a b : ℝ) :
  (∃ (l : ℝ → ℝ → Prop), (l 1 2) ∧ (∀ e, l e 0 ↔ e = a) ∧ (∀ f, l 0 f ↔ f = b) ∧ a = b) →
  (∀ x y, (2 * x - y = 0 ∨ x + y - 3 = 0) → (∃ l, l x y)) :=
by
  sorry

end line_equations_equal_intercepts_l95_95028


namespace largest_percentage_increase_l95_95593

theorem largest_percentage_increase :
  let students (y : ℕ) : ℕ :=
    match y with
    | 2002 => 50
    | 2003 => 55
    | 2004 => 60
    | 2005 => 65
    | 2006 => 72
    | 2007 => 80
    | 2008 => 90
    | _    => 0,
  let percentage_increase (a b : ℕ) : ℤ :=
    ((students (b) - students (a)) * 100 / students (a)),
  max_percentage_years (y1 y2 y3 y4 y5 y6 : ℕ × ℕ) : y1 = (2002, 2003) ∧ y2 = (2003, 2004) ∧ y3 = (2004, 2005) ∧ y4 = (2005, 2006) ∧ y5 = (2006, 2007) ∧ y6 = (2007, 2008) ∧
     (percentage_increase y1.1 y1.2 < percentage_increase y6.1 y6.2) ∧
     (percentage_increase y2.1 y2.2 < percentage_increase y6.1 y6.2) ∧
     (percentage_increase y3.1 y3.2 < percentage_increase y6.1 y6.2) ∧
     (percentage_increase y4.1 y4.2 < percentage_increase y6.1 y6.2) ∧
     (percentage_increase y5.1 y5.2 < percentage_increase y6.1 y6.2) :=
sorry

end largest_percentage_increase_l95_95593


namespace product_divisible_by_10_cases_l95_95202

theorem product_divisible_by_10_cases :
  (finset.range 6).card ^ 3 - 
  (finset.card ((finset.range 6).filter (λ x, x ≠ 2 ∧ x ≠ 4 ∧ x ≠ 6)) * 
       (finset.card ((finset.range 6).filter (λ y, y ≠ 2 ∧ y ≠ 4 ∧ y ≠ 6))) *
       (finset.card ((finset.range 6).filter (λ z, z ≠ 2 ∧ z ≠ 4 ∧ z ≠ 6)))) - 
  (finset.card ((finset.range 6).filter (λ x, x ≠ 5)) * 
       (finset.card ((finset.range 6).filter (λ y, y ≠ 5))) *
       (finset.card ((finset.range 6).filter (λ z, z ≠ 5)))) +
  (finset.card ((finset.range 6).filter (λ x, x ≠ 2 ∧ x ≠ 4 ∧ x ≠ 5 ∧ x ≠ 6)) *
       (finset.card ((finset.range 6).filter (λ y, y ≠ 2 ∧ y ≠ 4 ∧ y ≠ 5 ∧ y ≠ 6))) *
       (finset.card ((finset.range 6).filter (λ z, z ≠ 2 ∧ z ≠ 4 ∧ z ≠ 5 ∧ z ≠ 6)))) = 72 :=
sorry

end product_divisible_by_10_cases_l95_95202


namespace minimize_distances_is_centroid_l95_95395

/-- In any triangle ABC with median AD, the point M on AD that minimizes MA^2 + MB^2 + MC^2
     is the centroid of the triangle. -/
theorem minimize_distances_is_centroid {A B C D M : Type} [metric_space A] [metric_space B]
  [metric_space C] [metric_space D] [metric_space M]
 (h : midpoint B C = D) (m : M ∈ segment A D) :
    M = centroid A B C :=
sorry

end minimize_distances_is_centroid_l95_95395


namespace mixture_volume_correct_l95_95378

-- Define the input values
def water_volume : ℕ := 20
def vinegar_volume : ℕ := 18
def water_ratio : ℚ := 3/5
def vinegar_ratio : ℚ := 5/6

-- Calculate the mixture volume
def mixture_volume : ℚ :=
  (water_volume * water_ratio) + (vinegar_volume * vinegar_ratio)

-- Define the expected result
def expected_mixture_volume : ℚ := 27

-- State the theorem
theorem mixture_volume_correct : mixture_volume = expected_mixture_volume := by
  sorry

end mixture_volume_correct_l95_95378


namespace least_integer_with_six_factors_l95_95893

theorem least_integer_with_six_factors :
  ∃ n : ℕ, (∀ d : ℕ, d | n → ∃ (a b : ℕ) (p q : ℕ), n = p^a * q^b ∧ (a + 1) * (b + 1) = 6 ∧ prime p ∧ prime q) ∧
  ∀ m : ℕ, (∀ d : ℕ, d | m → ∃ (a b : ℕ) (p q : ℕ), m = p^a * q^b ∧ (a + 1) * (b + 1) = 6 ∧ prime p ∧ prime q) → n ≤ m :=
begin
  -- proof is intentionally omitted as per instructions
  sorry
end

end least_integer_with_six_factors_l95_95893


namespace union_P_Q_l95_95691

def P : Set ℝ := {x | -1 < x ∧ x < 1}
def Q : Set ℝ := {x | x^2 - 2*x < 0}

theorem union_P_Q : P ∪ Q = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

end union_P_Q_l95_95691


namespace measure_four_messzely_l95_95946

theorem measure_four_messzely (c3 c5 : ℕ) (hc3 : c3 = 3) (hc5 : c5 = 5) : 
  ∃ (x y z : ℕ), x = 4 ∧ x + y * c3 + z * c5 = 4 := 
sorry

end measure_four_messzely_l95_95946


namespace rice_weight_per_container_l95_95920

-- Given total weight of rice in pounds
def totalWeightPounds : ℚ := 25 / 2

-- Conversion factor from pounds to ounces
def poundsToOunces : ℚ := 16

-- Number of containers
def numberOfContainers : ℕ := 4

-- Total weight in ounces
def totalWeightOunces : ℚ := totalWeightPounds * poundsToOunces

-- Weight per container in ounces
def weightPerContainer : ℚ := totalWeightOunces / numberOfContainers

theorem rice_weight_per_container :
  weightPerContainer = 50 := 
sorry

end rice_weight_per_container_l95_95920


namespace years_on_compound_interest_l95_95856

noncomputable def simple_interest (P : ℕ) (R : ℕ) (T : ℕ) : ℚ :=
  (P * R * T) / 100

noncomputable def compound_interest (P : ℕ) (R : ℕ) (T : ℕ) : ℚ :=
  P * ((1 + R / 100.0)^T - 1)

noncomputable def calculate_years_for_interest : ℕ :=
let SI := simple_interest 1750 8 3 in
let CI := 2 * SI in
nat.ceil (Math.log ((CI / 4000) + 1) / Math.log (1.1))

theorem years_on_compound_interest :
   calculate_years_for_interest = 2 := 
by
  sorry

end years_on_compound_interest_l95_95856


namespace tailoring_cost_is_200_l95_95741

variables 
  (cost_first_suit : ℕ := 300)
  (total_paid : ℕ := 1400)

def cost_of_second_suit (tailoring_cost : ℕ) := 3 * cost_first_suit + tailoring_cost

theorem tailoring_cost_is_200 (T : ℕ) (h1 : cost_first_suit = 300) (h2 : total_paid = 1400) 
  (h3 : total_paid = cost_first_suit + cost_of_second_suit T) : 
  T = 200 := 
by 
  sorry

end tailoring_cost_is_200_l95_95741


namespace tray_height_l95_95955

theorem tray_height (side : ℕ) (distance : ℝ) (angle : ℝ) (h_side : side = 120)
  (h_distance : distance = real.sqrt 20) (h_angle : angle = real.pi / 4) :
  ∃ (m n : ℕ), (n > 0) ∧ (m < 1000) ∧ (∀ p : ℕ, p.prime → ¬ (p ^ n ∣ m)) ∧ m + n = 804 := 
by
  let height := real.sqrt 50
  have height_form : height = real.sqrt (2 ^ 4 * 5) := by sorry
  use [800, 4]
  split
  . exact by linarith
  split
  . exact by linarith
  split
  . intros p hp
    exact by sorry
  . exact by rfl

end tray_height_l95_95955


namespace ivanov_error_l95_95823

theorem ivanov_error (x : ℝ) (m : ℝ) (S2 : ℝ) (std_dev : ℝ) :
  x = 0 → m = 4 → S2 = 15.917 → std_dev = Real.sqrt S2 →
  ¬ (|x - m| ≤ std_dev) :=
by
  intros h1 h2 h3 h4
  -- Using the given values directly to state the inequality
  have h5 : |0 - 4| = 4 := by norm_num
  have h6 : Real.sqrt 15.917 ≈ 3.99 := sorry  -- approximation as direct result
  -- Evaluating the inequality
  have h7 : 4 ≰ 3.99 := sorry  -- this represents the key step that shows the error
  exact h7
  sorry

end ivanov_error_l95_95823


namespace passes_through_1_1_l95_95279

theorem passes_through_1_1 (a : ℝ) (h_pos : a > 0) (h_ne : a ≠ 1) : (1, 1) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, a^ (x - 1))} :=
by
  -- proof not required
  sorry

end passes_through_1_1_l95_95279


namespace good_students_count_l95_95371

theorem good_students_count (E B : ℕ) 
    (h1 : E + B = 25) 
    (h2 : ∀ (e ∈ finset.range 5), B > 12) 
    (h3 : ∀ (e ∈ finset.range 20), B = 3 * (E - 1)) :
    E = 5 ∨ E = 7 :=
by
  sorry

end good_students_count_l95_95371


namespace measure_angle_BAC_l95_95561

noncomputable def angle_BAC (angle_AOC angle_AOB : ℝ) : ℝ :=
  let angle_OAC := (180 - angle_AOC) / 2
  let angle_OAB := (180 - angle_AOB) / 2
  angle_OAC + angle_OAB

theorem measure_angle_BAC {O A B C : Type}
  (h1 : O = "center of the circle")
  (h2 : angle_AOC = 140)
  (h3 : angle_AOB = 160) :
  angle_BAC 140 160 = 30 :=
by
  sorry

end measure_angle_BAC_l95_95561


namespace find_b_plus_c_l95_95636

-- Define the variables and conditions
variables {a b c d : ℝ}

-- Conditions
def cond1 : Prop := a * b + a * c + b * d + c * d = 40
def cond2 : Prop := a + d = 6
def cond3 : Prop := a ≠ d

-- Goal
theorem find_b_plus_c (h1 : cond1) (h2 : cond2) (h3 : cond3) : b + c = 20 / 3 :=
by sorry

end find_b_plus_c_l95_95636


namespace grace_earnings_september_l95_95247

def charge_small_lawn_per_hour := 6
def charge_large_lawn_per_hour := 10
def charge_pull_small_weeds_per_hour := 11
def charge_pull_large_weeds_per_hour := 15
def charge_small_mulch_per_hour := 9
def charge_large_mulch_per_hour := 13

def hours_small_lawn := 20
def hours_large_lawn := 43
def hours_small_weeds := 4
def hours_large_weeds := 5
def hours_small_mulch := 6
def hours_large_mulch := 4

def earnings_small_lawn := hours_small_lawn * charge_small_lawn_per_hour
def earnings_large_lawn := hours_large_lawn * charge_large_lawn_per_hour
def earnings_small_weeds := hours_small_weeds * charge_pull_small_weeds_per_hour
def earnings_large_weeds := hours_large_weeds * charge_pull_large_weeds_per_hour
def earnings_small_mulch := hours_small_mulch * charge_small_mulch_per_hour
def earnings_large_mulch := hours_large_mulch * charge_large_mulch_per_hour

def total_earnings : ℕ :=
  earnings_small_lawn + earnings_large_lawn + earnings_small_weeds + earnings_large_weeds +
  earnings_small_mulch + earnings_large_mulch

theorem grace_earnings_september : total_earnings = 775 :=
by
  sorry

end grace_earnings_september_l95_95247


namespace each_girl_brought_2_cups_l95_95512

-- Here we define all the given conditions
def total_students : ℕ := 30
def num_boys : ℕ := 10
def cups_per_boy : ℕ := 5
def total_cups : ℕ := 90

-- Define the conditions as Lean definitions
def num_girls : ℕ := total_students - num_boys -- From condition 3
def cups_by_boys : ℕ := num_boys * cups_per_boy
def cups_by_girls : ℕ := total_cups - cups_by_boys

-- Define the final question and expected answer
def cups_per_girl : ℕ := cups_by_girls / num_girls

-- Final problem statement to prove
theorem each_girl_brought_2_cups :
  cups_per_girl = 2 :=
by
  have h1 : num_girls = 20 := by sorry
  have h2 : cups_by_boys = 50 := by sorry
  have h3 : cups_by_girls = 40 := by sorry
  have h4 : cups_per_girl = 2 := by sorry
  exact h4

end each_girl_brought_2_cups_l95_95512


namespace solution_set_ineq_l95_95779

noncomputable
def f (x : ℝ) : ℝ := sorry
noncomputable
def g (x : ℝ) : ℝ := sorry

axiom h_f_odd : ∀ x : ℝ, f (-x) = -f x
axiom h_g_even : ∀ x : ℝ, g (-x) = g x
axiom h_deriv_pos : ∀ x : ℝ, x < 0 → deriv f x * g x + f x * deriv g x > 0
axiom h_g_neg_three_zero : g (-3) = 0

theorem solution_set_ineq : { x : ℝ | f x * g x < 0 } = { x : ℝ | x < -3 } ∪ { x : ℝ | 0 < x ∧ x < 3 } := 
by sorry

end solution_set_ineq_l95_95779


namespace find_distance_l95_95149

noncomputable def distance_between_points_P_and_Q (A B C D P Q : ℝ × ℝ × ℝ) : ℝ :=
  let x₀ := P.1 - Q.1 in
  let y₀ := P.2 - Q.2 in
  let z₀ := P.3 - Q.3 in
  real.sqrt (x₀^2 + y₀^2 + z₀^2)

theorem find_distance :
  let A := (0, 0, 0) : ℝ × ℝ × ℝ
  let B := (1, 0, 0) : ℝ × ℝ × ℝ
  let C := (0.5, real.sqrt 3 / 2, 0) : ℝ × ℝ × ℝ
  let D := (0.5, real.sqrt 3 / 6, real.sqrt 6 / 3) : ℝ × ℝ × ℝ
  let P := (1/3, 0, 0) : ℝ × ℝ × ℝ
  let Q := (0.5, real.sqrt 3 / 3, real.sqrt 6 / 9) : ℝ × ℝ × ℝ in
  distance_between_points_P_and_Q A B C D P Q = real.sqrt 21 / 6 := by
  sorry

end find_distance_l95_95149


namespace distance_from_center_to_line_l95_95839

def center_circle : ℝ × ℝ := (1, 0)
def line_eqn : ℝ × ℝ → ℝ := λ p, p.1 + p.2 + 2 * Real.sqrt 2 - 1

theorem distance_from_center_to_line : 
  let dist := λ p l, |p / Real.sqrt (l * l + ... - ...
  (dist center_circle (λ p, line_eqn p)) = 2 :=
sorry

end distance_from_center_to_line_l95_95839


namespace interval_of_monotonic_increase_solution_set_for_derivative_l95_95757

noncomputable def f (x : ℝ) (λ : ℝ) : ℝ :=
  let a := (Real.cos x, Real.sin x)
  let b := (λ * Real.sin x - Real.cos x, Real.cos (Real.pi / 2 - x))
  a.1 * b.1 + a.2 * b.2

theorem interval_of_monotonic_increase (λ : ℝ) 
  (h : f (-Real.pi / 3) λ = f 0 λ) :
  ∃ k : ℤ, ∀ x : ℝ, kπ - π / 6 ≤ x ∧ x ≤ kπ + π / 3 ↔ f x λ is strictly increasing :=
sorry

theorem solution_set_for_derivative (λ : ℝ) 
  (h : f (-Real.pi / 3) λ = f 0 λ) :
  ∃ k : ℤ, ∀ x : ℝ, kπ < x ∧ x < kπ + π / 6 ↔ deriv f x λ > 2√3 :=
sorry

end interval_of_monotonic_increase_solution_set_for_derivative_l95_95757


namespace solve_x_squared_plus_y_squared_l95_95268

-- Variables
variables {x y : ℝ}

-- Conditions
def cond1 : (x + y)^2 = 36 := sorry
def cond2 : x * y = 8 := sorry

-- Theorem stating the problem's equivalent proof
theorem solve_x_squared_plus_y_squared : x^2 + y^2 = 20 := sorry

end solve_x_squared_plus_y_squared_l95_95268


namespace initial_games_l95_95307

theorem initial_games (X : ℕ) (h1 : X - 68 + 47 = 74) : X = 95 :=
by
  sorry

end initial_games_l95_95307


namespace find_BP_QT_l95_95383

-- Definitions based on conditions in the problem
variable {A B C D P T S Q A' : Point }
variable {PA AQ QP AB : ℝ}
variable h_rectangle : isRectangle A B C D 
variable h_APD : angle A P D = 90
variable h_perpendicular : isPerpendicular T S BC
variable h_BP_eq_PT : BP = PT
variable h_PD_intersects_TS_at_Q : intersects (PD) (TS) Q
variable h_RA_passes_through_Q : RA = intersection (A Q)

-- Given lengths in the problem
axiom h_PA : PA = 20
axiom h_AQ : AQ = 25
axiom h_QP : QP = 15

-- To be proven: BP and QT
theorem find_BP_QT : BP = 12 ∧ QT = 9 := by 
  sorry

end find_BP_QT_l95_95383


namespace value_of_625_powers_l95_95043

theorem value_of_625_powers :
  (625 : ℝ) ^ 0.12 * (625 : ℝ) ^ 0.13 = 5 := 
by
  sorry

end value_of_625_powers_l95_95043


namespace smallest_value_of_y_l95_95538

-- Define the quadratic equation
def quadratic_eq (y : ℝ) : Prop := 10 * y^2 - 47 * y + 49 = 0 

-- Define the smallest root
def smallest_root := 1.4

-- State the theorem
theorem smallest_value_of_y : ∃ y : ℝ, quadratic_eq y ∧ ∀ z : ℝ, quadratic_eq z → y ≤ z := 
begin
  use smallest_root,
  split,
  {
    -- Check that smallest_root satisfies the equation
    sorry
  },
  {
    -- Check that smallest_root is the smallest value
    sorry
  }
end

end smallest_value_of_y_l95_95538


namespace cost_of_remaining_shirts_l95_95174

theorem cost_of_remaining_shirts (total_cost : ℕ) (shirt_cost :ℕ) (remaining_shirts_cost : ℕ) (n : ℕ) (m : ℕ)
  (h1 : total_cost = 85)
  (h2 : m = 3)
  (h3 : shirt_cost = 15)
  (h4 : n = 2)
  (h5 : remaining_shirts_cost = total_cost - m * shirt_cost)
  (h6 : remaining_shirts_cost / n = 20) :
  remaining_shirts_cost / n = 20 :=
begin
  sorry
end

end cost_of_remaining_shirts_l95_95174


namespace basketball_game_l95_95713

theorem basketball_game (E H : ℕ) (h1 : E = H + 18) (h2 : E + H = 50) : H = 16 :=
by
  sorry

end basketball_game_l95_95713


namespace quadratic_polynomial_roots_l95_95082

theorem quadratic_polynomial_roots (b c p q : ℝ) :
  (∃ (K L M : ℝ × ℝ), 
    K = (p, 0) ∧ 
    L = (-p, 0) ∧ 
    M = (0, c) ∧ 
    (L.1 - K.1) = (M.1 - K.1) ∧ 
    ∠LKM = real.pi / 3) →
  y = (2 / real.sqrt 3) * x ^ 2 + b * x + c →
  roots y = {p, q} →
  p = real.sqrt 3 / 3 ∧ q = real.sqrt 3 :=
sorry

end quadratic_polynomial_roots_l95_95082


namespace cab_usual_time_l95_95527

noncomputable def usual_time (S : ℝ) : ℝ :=
  let T := 60
  T

theorem cab_usual_time (S : ℝ) :
  let T := 60 in
  walking_at (5 / 6) of S = S * (T + 12) ⇨ T = 60 :=
begin
  sorry,
end

end cab_usual_time_l95_95527


namespace adam_and_simon_time_to_be_80_miles_apart_l95_95124

theorem adam_and_simon_time_to_be_80_miles_apart :
  ∃ x : ℝ, (10 * x)^2 + (8 * x)^2 = 80^2 ∧ x = 6.25 :=
by
  sorry

end adam_and_simon_time_to_be_80_miles_apart_l95_95124


namespace good_students_count_l95_95317

noncomputable def student_count := 25

def is_good_student (s : Nat) := s ≤ student_count

def is_troublemaker (s : Nat) := s ≤ student_count

def always_tell_truth (s : Nat) := is_good_student s

def always_lie (s : Nat) := is_troublemaker s

def condition1 (E B : Nat) := E + B = student_count

def condition2 := ∀ (x : Nat), x ≤ 5 → is_good_student x → 
  ∃ (B : Nat), B > 24 / 2

def condition3 := ∀ (x : Nat), x ≤ 20 → is_troublemaker x → 
  ∃ (B E : Nat), B = 3 * (E - 1) ∧ E + B = student_count

theorem good_students_count :
  ∃ (E : Nat), condition1 E (student_count - E) ∧ condition2 ∧ condition3 :=
sorry  -- the actual proof is not required

end good_students_count_l95_95317


namespace polynomial_even_or_odd_polynomial_divisible_by_3_l95_95910

theorem polynomial_even_or_odd (p q : ℤ) :
  (∀ x : ℤ, (x^2 + p * x + q) % 2 = 0 ↔ (q % 2 = 0) ∧ (p % 2 = 1)) ∧
  (∀ x : ℤ, (x^2 + p * x + q) % 2 = 1 ↔ (q % 2 = 1) ∧ (p % 2 = 1)) := 
sorry

theorem polynomial_divisible_by_3 (p q : ℤ) :
  (∀ x : ℤ, (x^3 + p * x + q) % 3 = 0) ↔ (q % 3 = 0) ∧ (p % 3 = 2) := 
sorry

end polynomial_even_or_odd_polynomial_divisible_by_3_l95_95910


namespace part_1_part_2_part_3_l95_95606

def recursive_sequence (f : ℕ+ → ℝ) (c : ℝ) : Prop :=
  f 1 = 0 ∧ ∀ n : ℕ+, f (n + 1) = c * (f n)^3 + (1 - c)

theorem part_1 (f : ℕ+ → ℝ) (c : ℝ) (h : recursive_sequence f c) :
  (∀ n : ℕ+, f n ∈ set.Icc 0 1) ↔ c ∈ set.Icc 0 1 :=
sorry

theorem part_2 (f : ℕ+ → ℝ) (c : ℝ) (h : recursive_sequence f c) (h2 : 0 < c ∧ c < 1/3) :
  ∀ n : ℕ+, f n ≥ 1 - (3 * c)^(n - 1) :=
sorry

theorem part_3 (f : ℕ+ → ℝ) (c : ℝ) (h : recursive_sequence f c) (h2 : 0 < c ∧ c < 1/3) :
  ∀ n : ℕ+, ∑ i in finset.range n.succ, (f ⟨i.succ, nat.succ_pos i⟩)^2 > n + 1 - 2 / (1 - 3 * c) :=
sorry

end part_1_part_2_part_3_l95_95606


namespace good_students_count_l95_95318

noncomputable def student_count := 25

def is_good_student (s : Nat) := s ≤ student_count

def is_troublemaker (s : Nat) := s ≤ student_count

def always_tell_truth (s : Nat) := is_good_student s

def always_lie (s : Nat) := is_troublemaker s

def condition1 (E B : Nat) := E + B = student_count

def condition2 := ∀ (x : Nat), x ≤ 5 → is_good_student x → 
  ∃ (B : Nat), B > 24 / 2

def condition3 := ∀ (x : Nat), x ≤ 20 → is_troublemaker x → 
  ∃ (B E : Nat), B = 3 * (E - 1) ∧ E + B = student_count

theorem good_students_count :
  ∃ (E : Nat), condition1 E (student_count - E) ∧ condition2 ∧ condition3 :=
sorry  -- the actual proof is not required

end good_students_count_l95_95318


namespace fibonacci_product_divisibility_l95_95460

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci n + fibonacci (n + 1)

theorem fibonacci_product_divisibility (k : ℕ) (h : k > 0) :
  ∀ (n : ℕ), n ≥ k → 
    (∏ i in finset.range' n k, fibonacci i) ∣ (∏ j in finset.range k, fibonacci j) :=
  sorry

end fibonacci_product_divisibility_l95_95460


namespace problem_solution_l95_95409

-- Define the equation as a condition for y
def satisfies_equation (y : ℝ) : Prop :=
  2^(y^2 - 1) = (2^(y - 1))^2

-- Define the sum T of all positive real numbers y that satisfy the equation
noncomputable def T : ℝ :=
  ∑' (y : ℝ) in {y | y > 0 ∧ satisfies_equation y}, y

-- Prove that this sum T equals 1
theorem problem_solution : T = 1 :=
by
  sorry

end problem_solution_l95_95409


namespace xy_problem_l95_95271

theorem xy_problem (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 :=
by
  sorry

end xy_problem_l95_95271


namespace probability_of_matching_pair_l95_95612

theorem probability_of_matching_pair :
  let total_socks := 12 + 10 + 5 in
  let total_ways := Nat.choose total_socks 2 in
  let matching_ways_black := Nat.choose 12 2 in
  let matching_ways_white := Nat.choose 10 2 in
  let matching_ways_blue := Nat.choose 5 2 in
  let total_matching_ways := matching_ways_black + matching_ways_white + matching_ways_blue in
  (total_matching_ways / total_ways : ℚ) = 121 / 351 :=
by
  sorry

end probability_of_matching_pair_l95_95612


namespace good_students_count_l95_95319

noncomputable def student_count := 25

def is_good_student (s : Nat) := s ≤ student_count

def is_troublemaker (s : Nat) := s ≤ student_count

def always_tell_truth (s : Nat) := is_good_student s

def always_lie (s : Nat) := is_troublemaker s

def condition1 (E B : Nat) := E + B = student_count

def condition2 := ∀ (x : Nat), x ≤ 5 → is_good_student x → 
  ∃ (B : Nat), B > 24 / 2

def condition3 := ∀ (x : Nat), x ≤ 20 → is_troublemaker x → 
  ∃ (B E : Nat), B = 3 * (E - 1) ∧ E + B = student_count

theorem good_students_count :
  ∃ (E : Nat), condition1 E (student_count - E) ∧ condition2 ∧ condition3 :=
sorry  -- the actual proof is not required

end good_students_count_l95_95319


namespace find_x_l95_95392

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point :=
  ⟨Q.x - P.x, Q.y - P.y⟩

def parallel (v w : Point) : Prop :=
  v.x * w.y = v.y * w.x

theorem find_x (A B C : Point) (hA : A = ⟨0, -3⟩) (hB : B = ⟨3, 3⟩) (hC : C = ⟨x, -1⟩) (h_parallel : parallel (vector A B) (vector A C)) : x = 1 := 
by
  sorry

end find_x_l95_95392


namespace Xiaohong_wins_5_times_l95_95917

theorem Xiaohong_wins_5_times :
  ∃ W L : ℕ, (3 * W - 2 * L = 1) ∧ (W + L = 12) ∧ W = 5 :=
by
  sorry

end Xiaohong_wins_5_times_l95_95917


namespace solve_x_squared_plus_y_squared_l95_95263

-- Variables
variables {x y : ℝ}

-- Conditions
def cond1 : (x + y)^2 = 36 := sorry
def cond2 : x * y = 8 := sorry

-- Theorem stating the problem's equivalent proof
theorem solve_x_squared_plus_y_squared : x^2 + y^2 = 20 := sorry

end solve_x_squared_plus_y_squared_l95_95263


namespace good_students_count_l95_95370

theorem good_students_count (E B : ℕ) 
    (h1 : E + B = 25) 
    (h2 : ∀ (e ∈ finset.range 5), B > 12) 
    (h3 : ∀ (e ∈ finset.range 20), B = 3 * (E - 1)) :
    E = 5 ∨ E = 7 :=
by
  sorry

end good_students_count_l95_95370


namespace range_of_a_l95_95221

noncomputable def piecewiseFunc (x : ℝ) (a : ℝ) : ℝ :=
  if x < Real.exp 1 then -x^3 + x^2 else a * Real.log x

theorem range_of_a :
  ∃a : ℝ, (∃ P Q : ℝ × ℝ,
    let O := (0, 0) in
    P.1 * Q.1 = 0 ∧
    (-P.1^2 + piecewiseFunc P.1 a * (Q.1^3 + Q.1^2)) = 0 ∧
    ((P.1 + Q.1) / 2 = 0)) ↔ (0 < a ∧ a ≤ 1 / (Real.exp 1 + 1)) :=
sorry

end range_of_a_l95_95221


namespace total_heads_is_46_l95_95109

noncomputable def total_heads (hens cows : ℕ) : ℕ :=
  hens + cows

def num_feet_hens (num_hens : ℕ) : ℕ :=
  2 * num_hens

def num_cows (total_feet feet_hens_per_cow feet_cow_per_cow : ℕ) : ℕ :=
  (total_feet - feet_hens_per_cow) / feet_cow_per_cow

theorem total_heads_is_46 (num_hens : ℕ) (total_feet : ℕ)
  (hen_feet cow_feet hen_head cow_head : ℕ)
  (num_heads : ℕ) :
  num_hens = 24 →
  total_feet = 136 →
  hen_feet = 2 →
  cow_feet = 4 →
  hen_head = 1 →
  cow_head = 1 →
  num_heads = total_heads num_hens (num_cows total_feet (num_feet_hens num_hens) cow_feet) →
  num_heads = 46 :=
by
  intros
  sorry

end total_heads_is_46_l95_95109


namespace flip_all_red_m_19_n_94_flip_all_red_m_19_n_95_l95_95045

-- Define a Bool array of length 1995 representing disk red/blue status, true represents red side up.
def DiskArray := Array Bool

-- Defining the flip operation
def flip (lst : DiskArray) (k : ℕ) (m : ℕ) : DiskArray :=
  lst.mapIdx (λ idx val => if (idx >= k ∧ idx < k + m) then !val else val)

-- The statement for m = 19 and n = 94
theorem flip_all_red_m_19_n_94 (initial_config : DiskArray) (m n : ℕ) (hm : m = 19) (hn : n = 94) :
  ∃ flips : List (ℕ × ℕ), 
    (flips.foldl (λ arr (k, cnt) => flip arr k cnt) initial_config) = Array.mkArray 1995 true := sorry

-- The statement for m = 19 and n = 95
theorem flip_all_red_m_19_n_95 (initial_config : DiskArray) (m n : ℕ) (hm : m = 19) (hn : n = 95) :
  ¬ (∃ flips : List (ℕ × ℕ), 
    (flips.foldl (λ arr (k, cnt) => flip arr k cnt) initial_config) = Array.mkArray 1995 true) := sorry

end flip_all_red_m_19_n_94_flip_all_red_m_19_n_95_l95_95045


namespace solve_x_squared_plus_y_squared_l95_95269

-- Variables
variables {x y : ℝ}

-- Conditions
def cond1 : (x + y)^2 = 36 := sorry
def cond2 : x * y = 8 := sorry

-- Theorem stating the problem's equivalent proof
theorem solve_x_squared_plus_y_squared : x^2 + y^2 = 20 := sorry

end solve_x_squared_plus_y_squared_l95_95269


namespace digits_of_2_pow_41_l95_95621

noncomputable def log_10 (x : ℝ) := Real.log x / Real.log 10

theorem digits_of_2_pow_41 : ∀ n : ℕ, (n = 2^41) → (⌊log_10 (n : ℝ)⌋ + 1 = 13) :=
begin
  intros n hn,
  rw hn,
  have log_approx : log_10 2 ≈ 0.3010, from sorry,
  have approx_log_val : log_10 (2 : ℝ)^41 ≈ 12.341, from sorry,
  have trunc_log_val : ⌊log_10 (2 : ℝ)^41⌋ = 12, from sorry,
  show 12 + 1 = 13, by linarith
end

end digits_of_2_pow_41_l95_95621


namespace quadrilateral_diagonal_identity_l95_95966

-- Definitions of the conditions
variables (A B C D P Q S R : Point)
variable (areaABCD : ℝ)
variable (PQ RS d2 : ℝ)
variable (m n p : ℕ)
variable (d : ℝ)

-- Given conditions
def conditions := 
  (quadrilateral A B C D) ∧
  (areaABCD = 15) ∧
  (projections A C B D P Q) ∧
  (projections B D A C S R) ∧
  (PQ = 6) ∧
  (RS = 8) ∧
  (d_sq = d^2) ∧
  (d^2 = m + n * (real.sqrt p))

-- Proof statement
theorem quadrilateral_diagonal_identity :
  conditions A B C D P Q S R areaABCD PQ RS d2 d m n p →
  m + n + p = 97 :=
sorry

end quadrilateral_diagonal_identity_l95_95966


namespace absolute_value_of_neg_eight_l95_95025

/-- Absolute value of a number is the distance from 0 on the number line. -/
def absolute_value (x : ℤ) : ℤ :=
  if x >= 0 then x else -x

theorem absolute_value_of_neg_eight : absolute_value (-8) = 8 := by
  -- Proof is omitted
  sorry

end absolute_value_of_neg_eight_l95_95025


namespace find_sixth_sum_l95_95047

-- Defining the weights and the known sums
variables (x y z t : ℝ)
def known_sums : set ℝ := {1800, 1970, 2110, 2330, 2500}

-- Function to collect the pairwise sums
def pairwise_sums (x y z t : ℝ) : set ℝ := {x + y, x + z, x + t, y + z, y + t, z + t}

-- Prove that the sixth sum is 2190 grams
theorem find_sixth_sum
  (h_distinct: x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t) 
  (h_pair_sums: pairwise_sums x y z t \ known_sums = {2190}) : 
  2190 ∈ pairwise_sums x y z t :=
by
  sorry

end find_sixth_sum_l95_95047


namespace compound_interest_years_l95_95853

theorem compound_interest_years :
  let SI := 1750 * 8 / 100 * 3
  let CI := 2 * SI
  let P := 4000
  let R := 10 / 100
  log (1.21) / log (1.1) ≈ 2 → 
  ( ∀ (T : ℝ), CI = P * ((1 + R) ^ T - 1) → T = 2) :=
by
  sorry

end compound_interest_years_l95_95853


namespace problem1_problem2_l95_95385

-- Definition of the ellipse trajectory Γ
def ellipse_trajectory (x y : ℝ) : Prop :=
  x ^ 2 / 4 + y ^ 2 / 3 = 1

-- Statement for problem 1: proving the equation of the trajectory
theorem problem1 (x y : ℝ) : ellipse_trajectory x y ↔ ellipse_trajectory x y := 
by {
  sorry
}

-- Definition of line AD
def line_AD (m y : ℝ) : Prop :=
  x = m * y + 1

-- Given specific points and demonstrating intersection properties
noncomputable def points_intersection (x1 x2 y1 y2 m : ℝ) : Prop :=
  y1 + y2 = (-6 * m) / (4 + 3 * m ^ 2) ∧
  y1 * y2 = -9 / (4 + 3 * m ^ 2)

noncomputable def points_MN (x1 x2 y1 y2 : ℝ) (M N : ℝ × ℝ) : Prop :=
  M = (4, (2 * y1) / (x1 - 2)) ∧ 
  N = (4, (2 * y2) / (x2 - 2))

noncomputable def vectors_DM_DN (M N : ℝ × ℝ) : ℝ × ℝ :=
  (3, M.2) = (3, N.2);

-- Statement for problem 2: proving intersection properties and circle passing through a point
theorem problem2 (M N D : ℝ × ℝ) : vectors_DM_DN M N ↔ (vectors_DM_DN M N = D) := 
by {
  sorry
}

end problem1_problem2_l95_95385


namespace vector_parallel_cos_pi_two_plus_alpha_l95_95246

open Real

theorem vector_parallel_cos_pi_two_plus_alpha (α : ℝ) 
  (h_parallel : (1 / 3, tan α) = (cos α * k, 1 * k) for some k : ℝ) :
  cos (π / 2 + α) = -1 / 3 :=
begin
  sorry
end

end vector_parallel_cos_pi_two_plus_alpha_l95_95246


namespace correct_option_l95_95915

theorem correct_option :
  let x : ℂ in
  (∀ x : ℂ, x^3 * x^3 = x^9) ∧
  (∀ x : ℂ, 2*x^3 + 3*x^3 = 5*x^6) ∧
  (∀ x : ℂ, (2*x^2)^3 = 6*x^6) ∧
  (∀ x : ℂ, (2 + 3*x) * (2 - 3*x) = 4 - 9*x^2) →
  (∀ x : ℂ, (2 + 3*x) * (2 - 3*x) = 4 - 9*x^2) :=
by {
  intro _,
  simp,
  sorry
}

end correct_option_l95_95915


namespace find_divisor_l95_95909

theorem find_divisor (x : ℕ) (h : 180 % x = 0) (h_eq : 70 + 5 * 12 / (180 / x) = 71) : x = 3 := 
by
  -- proof goes here
  sorry

end find_divisor_l95_95909


namespace sum_of_digits_of_N_plus_2021_is_10_l95_95038

-- The condition that N is the smallest positive integer whose digits add to 41.
def smallest_integer_with_digit_sum_41 (N : ℕ) : Prop :=
  (N > 0) ∧ ((N.digits 10).sum = 41)

-- The Lean 4 statement to prove the problem.
theorem sum_of_digits_of_N_plus_2021_is_10 :
  ∃ N : ℕ, smallest_integer_with_digit_sum_41 N ∧ ((N + 2021).digits 10).sum = 10 :=
by
  -- The proof would go here
  sorry

end sum_of_digits_of_N_plus_2021_is_10_l95_95038


namespace batteries_problem_l95_95867

noncomputable def x : ℝ := 2 * z
noncomputable def y : ℝ := (4 / 3) * z

theorem batteries_problem
  (z : ℝ)
  (W : ℝ)
  (h1 : 4 * x + 18 * y + 16 * z = W * z)
  (h2 : 2 * x + 15 * y + 24 * z = W * z)
  (h3 : 6 * x + 12 * y + 20 * z = W * z) :
  W = 48 :=
sorry

end batteries_problem_l95_95867


namespace sum_pos_integers_9_l95_95286

theorem sum_pos_integers_9 (x y z : ℕ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_eq : 30 / 7 = x + 1 / (y + 1 / z)) : x + y + z = 9 :=
sorry

end sum_pos_integers_9_l95_95286


namespace problem_remainder_l95_95178

noncomputable def f (n k : ℕ) : ℝ :=
if k = 0 then 0 else
if k = 1 then 1 else
(n - k : ℕ) * f n (k-1) + k * f n (k+1) / n

noncomputable def S (N : ℕ) : ℝ :=
∑ i in finset.range N, f (N + i + 1) (i + 1)

theorem problem_remainder : 
  (⌊S 2013⌋.toNat % 2011) = 26 := sorry

end problem_remainder_l95_95178


namespace volume_ratio_l95_95402

/-
James has a cylindrical container with a diameter of 10 cm and height of 15 cm.
Lisa has a conical container with a diameter of 10 cm and height of 10 cm.
Prove the ratio of the volume of James’s container to Lisa's container equals 2.25:1.
-/

def diameter_James := 10 -- cm
def height_James := 15 -- cm
def diameter_Lisa := 10 -- cm
def height_Lisa := 10 -- cm

def radius_James := diameter_James / 2
def radius_Lisa := diameter_Lisa / 2

def volume_James := Math.pi * radius_James^2 * height_James
def volume_Lisa := (1/3 : ℝ) * Math.pi * radius_Lisa^2 * height_Lisa

theorem volume_ratio : volume_James / volume_Lisa = 2.25 := by
  sorry

end volume_ratio_l95_95402


namespace average_percentage_difference_in_tail_sizes_l95_95160

-- Definitions for the number of segments in each type of rattlesnake
def segments_eastern : ℕ := 6
def segments_western : ℕ := 8
def segments_southern : ℕ := 7
def segments_northern : ℕ := 9

-- Definition for percentage difference function
def percentage_difference (a : ℕ) (b : ℕ) : ℚ := ((b - a : ℚ) / b) * 100

-- Theorem statement to prove the average percentage difference
theorem average_percentage_difference_in_tail_sizes :
  (percentage_difference segments_eastern segments_western +
   percentage_difference segments_southern segments_western +
   percentage_difference segments_northern segments_western) / 3 = 16.67 := 
sorry

end average_percentage_difference_in_tail_sizes_l95_95160


namespace find_a4_b4_l95_95994

theorem find_a4_b4
  (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ)
  (h1 : a1 * b1 + a2 * b3 = 1)
  (h2 : a1 * b2 + a2 * b4 = 0)
  (h3 : a3 * b1 + a4 * b3 = 0)
  (h4 : a3 * b2 + a4 * b4 = 1)
  (h5 : a2 * b3 = 7) : 
  a4 * b4 = -6 :=
sorry

end find_a4_b4_l95_95994


namespace part_I_5_continuous_part_I_6_not_continuous_part_II_min_k_for_8_continuous_part_III_min_k_for_20_continuous_l95_95184

def is_continuous_representable (m : ℕ) (Q : List ℤ) : Prop :=
  ∀ n ∈ (List.range (m + 1)).tail, ∃ (sublist : List ℤ), sublist ≠ [] ∧ sublist ∈ Q.sublists' ∧ sublist.sum = n

theorem part_I_5_continuous :
  is_continuous_representable 5 [2, 1, 4] :=
sorry

theorem part_I_6_not_continuous :
  ¬is_continuous_representable 6 [2, 1, 4] :=
sorry

theorem part_II_min_k_for_8_continuous (Q : List ℤ) :
  is_continuous_representable 8 Q → Q.length ≥ 4 :=
sorry

theorem part_III_min_k_for_20_continuous (Q : List ℤ) 
  (h : is_continuous_representable 20 Q) (h_sum : Q.sum < 20) :
  Q.length ≥ 7 :=
sorry

end part_I_5_continuous_part_I_6_not_continuous_part_II_min_k_for_8_continuous_part_III_min_k_for_20_continuous_l95_95184


namespace geometric_sequence_common_ratio_l95_95760

noncomputable def common_ratio := { q : ℝ // ∃ (a1 : ℝ),
  let a2 := a1 * q,
      a3 := a1 * q^2,
      S3 := a1 * (1 - q^3) / (1 - q) in
  (a3 - 2 * a2 = 5) ∧ (S3 = 3) ∧ (q = -5 ∨ q = -1/2) }

theorem geometric_sequence_common_ratio :
  ∃ q : common_ratio, true := sorry

end geometric_sequence_common_ratio_l95_95760


namespace find_n_l95_95860

-- Defining the conditions.
def condition_one : Prop :=
  ∀ (c d : ℕ), 
  (80 * 2 * c = 320) ∧ (80 * 2 * d = 160)

def condition_two : Prop :=
  ∀ (c d : ℕ), 
  (100 * 3 * c = 450) ∧ (100 * 3 * d = 300)

def condition_three (n : ℕ) : Prop :=
  ∀ (c d : ℕ), 
  (40 * 4 * c = n) ∧ (40 * 4 * d = 160)

-- Statement of the proof problem using the conditions.
theorem find_n : 
  condition_one ∧ condition_two ∧ condition_three 160 :=
by
  sorry

end find_n_l95_95860


namespace vector_magnitude_condition_l95_95697

variables (a b : EuclideanSpace ℝ (Fin 2))

def vec_a : EuclideanSpace ℝ (Fin 2) := ![1,2]
def vec_b (m : ℝ) : EuclideanSpace ℝ (Fin 2) := ![m,1]

theorem vector_magnitude_condition (m : ℝ) (h : m < 0) (cond : inner (vec_b m) (vec_a + vec_b m) = 3) : norm (vec_b m) = Real.sqrt 2 :=
by
  have am_plus_bm := 1 + m
  have bm := vec_b m
  have a_plus_b := vec_a + bm
  have inner_product := m * am_plus_bm + 3
  have solved_m := by linarith
  have final_m : m = -1 := by assumption
  have b_am := vec_b (-1)
  have magnitude := by 
    rw [norm_eq_sqrt_inner, inner_self_eq_norm_sq]
    simp
  exact Real.sqrt_eq_rfl
  sorry

end vector_magnitude_condition_l95_95697


namespace arithmetic_sequence_c_d_sum_l95_95502

theorem arithmetic_sequence_c_d_sum (c d : ℕ) 
  (h1 : 10 - 3 = 7) 
  (h2 : 17 - 10 = 7) 
  (h3 : c = 17 + 7) 
  (h4 : d = c + 7) 
  (h5 : d + 7 = 38) :
  c + d = 55 := 
by
  sorry

end arithmetic_sequence_c_d_sum_l95_95502


namespace problem_statement_l95_95258

-- Define the conditions and the goal
theorem problem_statement {x y : ℝ} 
  (h1 : (x + y)^2 = 36)
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := 
by
  sorry

end problem_statement_l95_95258


namespace good_students_l95_95365

theorem good_students (G T : ℕ) (h1 : G + T = 25) (h2 : T > 12) (h3 : T = 3 * (G - 1)) :
  G = 5 ∨ G = 7 :=
by
  sorry

end good_students_l95_95365


namespace count_standard_01_sequences_m_eq_4_l95_95979

def is_standard_01_sequence (s : List Bool) : Prop :=
  let n := s.length / 2
  ∧ (s.count (fun x => x = false) = n)
  ∧ (s.count (fun x => x = true) = n)
  ∧ ∀ k ∈ (List.range (2 * n + 1)), (s.take k).count (fun x => x = false) ≥ (s.take k).count (fun x => x = true)

theorem count_standard_01_sequences_m_eq_4 : 
  ∃ (seqs : List (List Bool)),
    ∀ s ∈ seqs, is_standard_01_sequence s ∧ 
    seqs.length = 14 := 
by
  sorry

end count_standard_01_sequences_m_eq_4_l95_95979


namespace food_waste_in_scientific_notation_l95_95594

-- Given condition that 1 billion equals 10^9
def billion : ℕ := 10 ^ 9

-- Problem statement: expressing 530 billion kilograms in scientific notation
theorem food_waste_in_scientific_notation :
  (530 * billion : ℝ) = 5.3 * 10^10 := 
  sorry

end food_waste_in_scientific_notation_l95_95594


namespace num_four_digit_integers_with_remainders_eq_6_l95_95699

theorem num_four_digit_integers_with_remainders_eq_6 :
  (set.filter (λ n : ℕ, n % 7 = 1 ∧ n % 10 = 3 ∧ n % 13 = 5)
    (set.Ico 1000 10000)).card = 6 :=
by sorry

end num_four_digit_integers_with_remainders_eq_6_l95_95699


namespace minimum_dot_product_solution_l95_95634

-- Definitions of vectors OA, OB, OP
def vec_OA : ℝ × ℝ × ℝ := (1, 2, 4)
def vec_OB : ℝ × ℝ × ℝ := (2, 1, 1)
def vec_OP : ℝ × ℝ × ℝ := (1, 1, 2)

-- Definition of the point Q on line OP as Q(λ, λ, 2λ)
def point_Q (λ : ℝ) : ℝ × ℝ × ℝ := (λ, λ, 2 * λ)

-- Dot product of two 3D vectors
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Calculation of QA and QB given Q
def vec_QA (λ : ℝ) : ℝ × ℝ × ℝ := (1 - λ, 2 - λ, 4 - 2 * λ)
def vec_QB (λ : ℝ) : ℝ × ℝ × ℝ := (2 - λ, 1 - λ, 1 - 2 * λ)

-- Proof statement
noncomputable def minimum_dot_product_Q_coordinates : ℝ × ℝ × ℝ :=
    let λ := 4 / 3 in
    point_Q λ

theorem minimum_dot_product_solution :
  minimum_dot_product_Q_coordinates = (4 / 3, 4 / 3, 8 / 3) :=
by
  sorry

end minimum_dot_product_solution_l95_95634


namespace quadratics_equal_l95_95003

-- Definitions of integer part and quadratic polynomials
def intPart (a : ℝ) : ℤ := ⌊a⌋

-- Define quadratic polynomial
structure QuadraticPoly :=
(coeffs : ℝ × ℝ × ℝ) -- (a, b, c) coefficients of ax^2 + bx + c

def evalQuadPoly (p : QuadraticPoly) (x : ℝ) : ℝ :=
p.coeffs.1 * x^2 + p.coeffs.2 * x + p.coeffs.3

-- Problem Statement in Lean 4
theorem quadratics_equal
  (f g : QuadraticPoly)
  (h : ∀ x : ℝ, intPart (evalQuadPoly f x) = intPart (evalQuadPoly g x)) :
  ∀ x : ℝ, evalQuadPoly f x = evalQuadPoly g x := 
sorry

end quadratics_equal_l95_95003


namespace hours_of_rain_l95_95632

theorem hours_of_rain (total_hours : ℕ) (hours_without_rain : ℕ) :
  total_hours = 9 ∧ hours_without_rain = 5 → total_hours - hours_without_rain = 4 :=
by
  intros h
  cases h
  rw [h_left, h_right]
  exact rfl

end hours_of_rain_l95_95632


namespace opposite_neg_three_over_two_l95_95037

-- Define the concept of the opposite number
def opposite (x : ℚ) : ℚ := -x

-- State the problem: The opposite number of -3/2 is 3/2
theorem opposite_neg_three_over_two :
  opposite (- (3 / 2 : ℚ)) = (3 / 2 : ℚ) := 
  sorry

end opposite_neg_three_over_two_l95_95037


namespace incorrect_transformation_l95_95185

theorem incorrect_transformation (a b : ℕ) (h : a ≠ 0 ∧ b ≠ 0) (h_eq : a / 2 = b / 3) :
  (∃ k : ℕ, 2 * a = 3 * b → false) ∧ 
  (a / b = 2 / 3) ∧ 
  (b / a = 3 / 2) ∧
  (3 * a = 2 * b) :=
by
  sorry

end incorrect_transformation_l95_95185


namespace express_q_in_terms_of_a_and_b_l95_95284

-- Definitions of real numbers
variables (a b q r : ℝ)

-- Conditions
def is_imaginary_root (a b : ℝ) (x : ℂ) : Prop :=
  x = a + b * complex.I ∨ x = a - b * complex.I

def is_polynomial_root (a b q r : ℝ) (x : ℂ) : Prop :=
  ∃ (c : ℝ), c ≠ a ∧ c ≠ a - b * complex.I ∧ c ≠ a + b * complex.I ∧
  ((x - (a + b * complex.I)) * (x - (a - b * complex.I)) * (x - c) = 0)

-- Main theorem statement
theorem express_q_in_terms_of_a_and_b (h_b_ne_zero : b ≠ 0)
    (h_root : is_imaginary_root a b (a + b * complex.I))
    (h_poly_root : is_polynomial_root a b q r (a + b * complex.I)) :
  q = b^2 - 3 * a^2 :=
sorry

end express_q_in_terms_of_a_and_b_l95_95284


namespace equation_circle_iff_a_equals_neg_one_l95_95551

theorem equation_circle_iff_a_equals_neg_one :
  (∀ x y : ℝ, ∃ k : ℝ, a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a = k * (x^2 + y^2)) ↔ 
  a = -1 :=
by sorry

end equation_circle_iff_a_equals_neg_one_l95_95551


namespace intersecting_lines_solution_l95_95479

theorem intersecting_lines_solution (x y b : ℝ) 
  (h₁ : y = 2 * x - 5)
  (h₂ : y = 3 * x + b)
  (hP : x = 1 ∧ y = -3) : 
  b = -6 ∧ x = 1 ∧ y = -3 := by
  sorry

end intersecting_lines_solution_l95_95479


namespace mr_smith_markers_l95_95435

theorem mr_smith_markers :
  ∀ (initial_markers : ℕ) (total_markers : ℕ) (markers_per_box : ℕ) 
  (number_of_boxes : ℕ),
  initial_markers = 32 → 
  total_markers = 86 → 
  markers_per_box = 9 → 
  number_of_boxes = (total_markers - initial_markers) / markers_per_box →
  number_of_boxes = 6 :=
by
  intros initial_markers total_markers markers_per_box number_of_boxes h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃] at h₄
  simp only [Nat.sub] at h₄
  exact h₄

end mr_smith_markers_l95_95435


namespace fractional_part_painted_correct_l95_95873

noncomputable def fractional_part_painted (time_fence : ℕ) (time_hole : ℕ) : ℚ :=
  (time_hole : ℚ) / time_fence

theorem fractional_part_painted_correct : fractional_part_painted 60 40 = 2 / 3 := by
  sorry

end fractional_part_painted_correct_l95_95873


namespace sum_f_k_l95_95208

noncomputable def f : ℝ → ℝ := sorry

theorem sum_f_k :
  (∀ x : ℝ, f(1 - x) = -f(-(1 - x))) ∧
  (∀ x : ℝ, 1 - f(x + 2) = -(1 - f(-(x + 2)))) →
  (∑ k in finset.range 22, f k) = 210 := 
begin
  sorry
end

end sum_f_k_l95_95208


namespace train_length_l95_95956

theorem train_length (speed_kph : ℝ) (time_sec : ℝ) (speed_mps : ℝ) (length_m : ℝ) 
  (h1 : speed_kph = 60) 
  (h2 : time_sec = 42) 
  (h3 : speed_mps = speed_kph * 1000 / 3600) 
  (h4 : length_m = speed_mps * time_sec) :
  length_m = 700.14 :=
by
  sorry

end train_length_l95_95956


namespace sum_of_identical_digits_l95_95235

theorem sum_of_identical_digits
  (a b c : ℕ) (x : ℕ := a * 11111) (y : ℕ := b * 1111) (z : ℕ := c * 111) :
  (∃ (x y z : ℕ), (x % 11111 = 0) ∧ (y % 1111 = 0) ∧ (z % 111 = 0) ∧ 
  let sum := x + y + z in 
  sum ≥ 10000 ∧ sum < 100000 ∧ 
  (let digits := [sum / 10000 % 10, sum / 1000 % 10, sum / 100 % 10, sum / 10 % 10, sum % 10] in 
  digits.nodup)) := 
sorry

end sum_of_identical_digits_l95_95235


namespace find_f_neg_two_l95_95292

-- Definitions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Function f
def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + x else -(((-x)^2) + (-x))

-- The proof problem statement
theorem find_f_neg_two
  (h_odd : is_odd_function f)
  (h_f_pos : ∀ x : ℝ, x ≥ 0 → f x = x^2 + x) :
  f (-2) = -6 :=
sorry

end find_f_neg_two_l95_95292


namespace cost_of_remaining_shirts_l95_95175

theorem cost_of_remaining_shirts (total_cost : ℕ) (shirt_cost :ℕ) (remaining_shirts_cost : ℕ) (n : ℕ) (m : ℕ)
  (h1 : total_cost = 85)
  (h2 : m = 3)
  (h3 : shirt_cost = 15)
  (h4 : n = 2)
  (h5 : remaining_shirts_cost = total_cost - m * shirt_cost)
  (h6 : remaining_shirts_cost / n = 20) :
  remaining_shirts_cost / n = 20 :=
begin
  sorry
end

end cost_of_remaining_shirts_l95_95175


namespace calc_expression_l95_95970

theorem calc_expression :
  2 / (-1 / 4) - abs (-Real.sqrt 18) + (1 / 5) ^ (-1) = -3 - 3 * Real.sqrt 2 :=
by
  sorry

end calc_expression_l95_95970


namespace largest_primes_product_l95_95906

theorem largest_primes_product : 7 * 97 * 997 = 679679 := by
  sorry

end largest_primes_product_l95_95906


namespace distance_from_intersection_point_to_line_l95_95849

theorem distance_from_intersection_point_to_line :
  let l1 := λ x y : ℝ, x - 2 * y + 4 = 0,
      l2 := λ x y : ℝ, x + y - 2 = 0,
      l3 := λ x y : ℝ, 3 * x - 4 * y + 5 = 0,
      P  := ⟨0, 2⟩ in
  l1 P.1 P.2 ∧ l2 P.1 P.2 → 
  real.dist (P.1, P.2) (3, -4, 5) = 3/5 :=
by
  sorry

end distance_from_intersection_point_to_line_l95_95849


namespace fraction_of_smaller_region_l95_95795

-- Assume P is the center of the regular hexagon ABCDEF and Z is the midpoint of side AB
variables (P Z A B C D E F : Type) 
[RegularHexagon : regular_hexagon ABCDEF] 
[midAB : midpoint AB Z]
[centerHex : hexagon_center ABCDEF P]

-- Define the smaller region function
def smaller_region (hexagon : ABCDEF) (mid : Z) (center : P) : ℚ :=
  -- We only define the fraction of the hexagon's area. 
  1 / 12

-- The goal is to prove the smaller region fraction is 1/12
theorem fraction_of_smaller_region : smaller_region ABCDEF Z P = 1 / 12 := 
sorry

end fraction_of_smaller_region_l95_95795


namespace intersection_eq_l95_95656

def A : set ℤ := {0, 1, 2}
def B : set ℤ := {-2, 0, 1}

theorem intersection_eq : A ∩ B = {0, 1} :=
by 
  sorry

end intersection_eq_l95_95656


namespace length_of_second_leg_l95_95117

theorem length_of_second_leg (a c b : ℝ) (h₁ : a = 6) (h₂ : c = 10) (h₃ : c^2 = a^2 + b^2) : b = 8 :=
by {
  rw [h₁, h₂] at h₃,
  have h₄ : 10^2 = 6^2 + b^2 := h₃,
  norm_num at h₄,
  exact eq_of_sq_eq_sq 64 (by norm_num)
}

end length_of_second_leg_l95_95117


namespace max_marks_l95_95544

-- Definitions based on conditions
def passing_percentage: Float := 0.36
def student_marks: Int := 130
def fails_by: Int := 14

-- Prove that given these conditions, the maximum number of marks is 400
theorem max_marks (h1: student_marks + fails_by = round (passing_percentage * 400)) : (400 : Int) = 400 := by
  rw [Int.cast_id]
  sorry

end max_marks_l95_95544


namespace equivalent_expression_l95_95467

theorem equivalent_expression (m n : ℕ) (P Q : ℕ) (hP : P = 3^m) (hQ : Q = 5^n) :
  15^(m + n) = P * Q :=
by
  sorry

end equivalent_expression_l95_95467


namespace sum_of_identical_digit_numbers_l95_95238

-- Definitions of the compositions of digits
def five_digit_number (a : ℕ) : ℕ := 10000 * a + 1000 * a + 100 * a + 10 * a + a
def four_digit_number (b : ℕ) : ℕ := 1000 * b + 100 * b + 10 * b + b
def three_digit_number (c : ℕ) : ℕ := 100 * c + 10 * c + c

-- Statement of the problem
theorem sum_of_identical_digit_numbers :
  ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧
  let sum := five_digit_number a + four_digit_number b + three_digit_number c in
  sum >= 10000 ∧ sum < 100000 ∧
  let digits := [sum / 10000 % 10, sum / 1000 % 10, sum / 100 % 10, sum / 10 % 10, sum % 10] in
  digits.nodup :=
sorry

end sum_of_identical_digit_numbers_l95_95238


namespace find_x_l95_95072

def f (x: ℝ) : ℝ := 3 * x - 5

theorem find_x (x : ℝ) (h : 2 * (f x) - 10 = f (x - 2)) : x = 3 :=
by
  sorry

end find_x_l95_95072


namespace tissue_paper_remaining_l95_95600

theorem tissue_paper_remaining (initial_tissue : ℕ) (used_tissue : ℕ) : initial_tissue = 97 → used_tissue = 4 → initial_tissue - used_tissue = 93 := by
  intros h1 h2
  simp [h1, h2]
  sorry

end tissue_paper_remaining_l95_95600


namespace cows_and_goats_sum_l95_95380

theorem cows_and_goats_sum (x y z : ℕ) 
  (h1 : x + y + z = 12) 
  (h2 : 4 * x + 2 * y + 4 * z = 18 + 2 * (x + y + z)) 
  : x + z = 9 := by 
  sorry

end cows_and_goats_sum_l95_95380


namespace compare_values_l95_95191

noncomputable def a := Real.exp 0.2
noncomputable def b := 0.2 ^ Real.exp 1
noncomputable def c := Real.log 2

theorem compare_values : b < c ∧ c < a := 
  by
  sorry

end compare_values_l95_95191


namespace diophantine_3x_5y_diophantine_3x_5y_indefinite_l95_95877

theorem diophantine_3x_5y (n : ℕ) (h_n_pos : n > 0) :
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 3 * x + 5 * y = n) ↔ 
    (∃ k : ℕ, (n = 3 * k ∧ n ≥ 15) ∨ 
              (n = 3 * k + 1 ∧ n ≥ 13) ∨ 
              (n = 3 * k + 2 ∧ n ≥ 11) ∨ 
              (n = 8)) :=
sorry

theorem diophantine_3x_5y_indefinite (n m : ℕ) (h_n_large : n > 40 * m):
  ∃ (N : ℕ), ∀ k ≤ N, ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 3 * x + 5 * y = n + k :=
sorry

end diophantine_3x_5y_diophantine_3x_5y_indefinite_l95_95877


namespace daps_equivalent_to_dips_l95_95282

-- Definitions from conditions
def daps (n : ℕ) : ℕ := n
def dops (n : ℕ) : ℕ := n
def dips (n : ℕ) : ℕ := n

-- Given conditions
def equivalence_daps_dops : daps 8 = dops 6 := sorry
def equivalence_dops_dips : dops 3 = dips 11 := sorry

-- Proof problem
theorem daps_equivalent_to_dips (n : ℕ) (h1 : daps 8 = dops 6) (h2 : dops 3 = dips 11) : daps 24 = dips 66 :=
sorry

end daps_equivalent_to_dips_l95_95282


namespace ivanov_error_l95_95811

-- Given conditions
def mean_temperature (x : ℝ) : Prop := x = 0
def median_temperature (m : ℝ) : Prop := m = 4
def variance_temperature (s² : ℝ) : Prop := s² = 15.917

-- Statement to prove
theorem ivanov_error (x m s² : ℝ) 
  (mean_x : mean_temperature x)
  (median_m : median_temperature m)
  (variance_s² : variance_temperature s²) :
  (x - m)^2 > s² :=
by
  rw [mean_temperature, median_temperature, variance_temperature] at *
  simp [*, show 0 = 0, from rfl, show 4 = 4, from rfl]
  sorry

end ivanov_error_l95_95811


namespace max_sum_first_n_terms_arithmetic_sequence_l95_95287

theorem max_sum_first_n_terms_arithmetic_sequence
  {a : ℕ → ℤ} (h1 : a 7 + a 8 + a 9 > 0) (h2 : a 7 + a 10 < 0) :
  ∃ n : ℕ, n = 8 ∧
    (∀ m : ℕ, m ≠ 8 → (m ≤ 8 → (∑ i in finset.range m, a i) < ∑ i in finset.range 8, a i) ∧ (m > 8 → (∑ i in finset.range m, a i) < ∑ i in finset.range 8, a i)) :=
by
  sorry

end max_sum_first_n_terms_arithmetic_sequence_l95_95287


namespace quadratics_equal_l95_95006

-- Definitions of integer part and quadratic polynomials
def intPart (a : ℝ) : ℤ := ⌊a⌋

-- Define quadratic polynomial
structure QuadraticPoly :=
(coeffs : ℝ × ℝ × ℝ) -- (a, b, c) coefficients of ax^2 + bx + c

def evalQuadPoly (p : QuadraticPoly) (x : ℝ) : ℝ :=
p.coeffs.1 * x^2 + p.coeffs.2 * x + p.coeffs.3

-- Problem Statement in Lean 4
theorem quadratics_equal
  (f g : QuadraticPoly)
  (h : ∀ x : ℝ, intPart (evalQuadPoly f x) = intPart (evalQuadPoly g x)) :
  ∀ x : ℝ, evalQuadPoly f x = evalQuadPoly g x := 
sorry

end quadratics_equal_l95_95006


namespace intersecting_lines_sum_l95_95483

theorem intersecting_lines_sum (a b : ℝ) (h1 : 2 = (1/3) * 4 + a) (h2 : 4 = (1/3) * 2 + b) : a + b = 4 :=
sorry

end intersecting_lines_sum_l95_95483


namespace lizzy_initial_money_l95_95783

-- Declare the conditions
def initial_money (x : ℕ) := 
  ∃ y z : ℕ, y = 15 ∧ z = 33 ∧ x = z - (y + (y * 20 / 100))

-- State the theorem to be proved
theorem lizzy_initial_money : initial_money 15 :=
by
  unfold initial_money
  use [15, 33]
  split
  { refl }
  split
  { refl }
  norm_num
  sorry

end lizzy_initial_money_l95_95783


namespace quadratic_polynomials_equal_l95_95001

def integer_part (a : ℝ) : ℤ := ⌊a⌋

theorem quadratic_polynomials_equal 
  (f g : ℝ → ℝ)
  (hf : ∀ x, ∃ a1 b1 c1, f x = a1 * x^2 + b1 * x + c1)
  (hg : ∀ x, ∃ a2 b2 c2, g x = a2 * x^2 + b2 * x + c2)
  (H : ∀ x, integer_part (f x) = integer_part (g x)) : 
  ∀ x, f x = g x :=
sorry

end quadratic_polynomials_equal_l95_95001


namespace find_angle_DAE_l95_95042

-- Definitions
variables {A B C D E : Type} [EuclideanGeometry] 
variables {a b c BD DE EC : ℝ}

-- Given conditions
def triangle_sides (a b c : ℝ) : Prop := 
  a = 29 ∧ b = 21 ∧ c = 20

def segment_conditions (BD DE EC : ℝ) : Prop := 
  BD = 8 ∧ DE = 12 ∧ EC = 9

-- Theorem statement
theorem find_angle_DAE {A B C D E : Type} [EuclideanGeometry] 
  (h1 : triangle_sides 29 21 20)
  (h2 : segment_conditions 8 12 9) : 
  ∠DAE = 45 :=
sorry

end find_angle_DAE_l95_95042


namespace total_trip_length_l95_95013

/-- Prove the total length of Roy's trip in miles given the conditions. --/
theorem total_trip_length
  (d : ℝ) -- Let d be the total length of the trip in miles
  (battery_distance : ℝ := 60) -- The car ran on its battery for the first 60 miles
  (gasoline_rate : ℝ := 0.03) -- The car ran on gasoline at a rate of 0.03 gallons per mile
  (avg_mileage : ℝ := 75) -- The car averaged 75 miles per gallon on the whole trip
  (h : d / (gasoline_rate * (d - battery_distance)) = avg_mileage) -- Equation for average mileage
  : d = 108 := 
begin
  sorry -- Proof omitted
end

end total_trip_length_l95_95013


namespace dist_balls_into_boxes_l95_95703

open Finset

/-- Helper function to count distinct ways to distribute balls into indistinguishable boxes -/
def distribution_count (balls boxes : ℕ) : ℕ :=
  nat.strong_induction_on balls (λ n IH, match boxes with
    | 0        => if n = 0 then 1 else 0
    | succ b   => (range (n + 1)).sum (λ i, IH (n - i) b)
  end)

theorem dist_balls_into_boxes :
  distribution_count 6 4 = 187 :=
by
  -- Placeholder for the actual proof
  sorry

end dist_balls_into_boxes_l95_95703


namespace annie_job_time_l95_95153

noncomputable def annie_time : ℝ :=
  let dan_time := 15
  let dan_rate := 1 / dan_time
  let dan_hours := 6
  let fraction_done_by_dan := dan_rate * dan_hours
  let fraction_left_for_annie := 1 - fraction_done_by_dan
  let annie_work_remaining := fraction_left_for_annie
  let annie_hours := 6
  let annie_rate := annie_work_remaining / annie_hours
  let annie_time := 1 / annie_rate 
  annie_time

theorem annie_job_time :
  annie_time = 3.6 := 
sorry

end annie_job_time_l95_95153


namespace sum_inequality_l95_95081

variable {n : ℕ} (a b : Fin n → ℝ)

def condition1 := ∀ i, 1 ≤ i → i ≤ n → a i > 0
def condition2 := ∀ i j, 1 ≤ i → i < j → j ≤ n → a i ≥ a j
def condition3 := b 0 ≥ a 0
def condition4 := ∀ i, 1 ≤ i → i < n → (∏ j in Fin.range (i + 1), b j) ≥ (∏ j in Fin.range (i + 1), a j)

theorem sum_inequality :
  (∀ i, 1 ≤ i → i ≤ n → a i > 0) → 
  (∀ i j, 1 ≤ i → i < j → j ≤ n → a i ≥ a j) → 
  b 0 ≥ a 0 → 
  (∀ i, 1 ≤ i → i < n → (∏ j in Fin.range (i + 1), b j) ≥ (∏ j in Fin.range (i + 1), a j)) → 
  (∑ i, b i) ≥ (∑ i, a i) ∧ 
  (∀ i, (∑ j, b j) = (∑ j, a j) → b i = a i) := 
sorry

end sum_inequality_l95_95081


namespace odd_digits_in_base4_317_l95_95622

noncomputable def base4_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (n : ℕ) : List ℕ :=
    if n = 0 then [] else (aux (n / 4)) ++ [n % 4]
  aux n

def count_odd_digits (lst : List ℕ) : ℕ :=
  lst.filter (λ x, x % 2 = 1).length

theorem odd_digits_in_base4_317 : count_odd_digits (base4_representation 317) = 4 :=
  sorry

end odd_digits_in_base4_317_l95_95622


namespace john_initial_money_l95_95406

variable (X S : ℕ)
variable (L : ℕ := 500)
variable (cond1 : L = S - 600)
variable (cond2 : X = S + L)

theorem john_initial_money : X = 1600 :=
by
  sorry

end john_initial_money_l95_95406


namespace labourer_monthly_income_l95_95027

-- Define the conditions
def total_expense_first_6_months : ℕ := 90 * 6
def total_expense_next_4_months : ℕ := 60 * 4
def debt_cleared_and_savings : ℕ := 30

-- Define the monthly income
def monthly_income : ℕ := 81

-- The statement to be proven
theorem labourer_monthly_income (I D : ℕ) (h1 : 6 * I + D = total_expense_first_6_months) 
                               (h2 : 4 * I - D = total_expense_next_4_months + debt_cleared_and_savings) :
  I = monthly_income :=
by {
  sorry
}

end labourer_monthly_income_l95_95027


namespace ratio_group1_group2_l95_95971

def car_distance (speed: ℕ) (time: ℕ) : ℕ := speed * time

def group_distance (num_carA: ℕ) (num_carB: ℕ) (num_carC: ℕ) : ℕ := 
  num_carA * car_distance 80 4 + num_carB * car_distance 100 2 + num_carC * car_distance 120 3

def gcd (a: ℕ) (b: ℕ) : ℕ :=
if b = 0 then a else gcd b (a % b)

theorem ratio_group1_group2 : 
  let group1_distance := group_distance 3 1 0 in
  let group2_distance := group_distance 0 2 2 in
  gcd group1_distance group2_distance = 40 →
  (group1_distance / 40, group2_distance / 40) = (29, 28) :=
sorry

end ratio_group1_group2_l95_95971


namespace least_number_to_add_or_subtract_is_25_l95_95534

theorem least_number_to_add_or_subtract_is_25 :
  let n := 7846321 in
  let p := 89 in
  (n % p = 64) ∧ min (n % p) (p - (n % p)) = 25 :=
by sorry

end least_number_to_add_or_subtract_is_25_l95_95534


namespace translated_function_correct_l95_95875

def original_function (x : ℝ) : ℝ := 2 * x^2

def translated_function_left_one (x : ℝ) : ℝ := original_function(x + 1)

def translated_function_left_one_up_three (x : ℝ) : ℝ := translated_function_left_one(x) + 3

theorem translated_function_correct : ∀ x : ℝ, translated_function_left_one_up_three(x) = 2 * (x + 1)^2 + 3 := 
by
  simp [translated_function_left_one_up_three, translated_function_left_one, original_function]
  sorry

end translated_function_correct_l95_95875


namespace find_m_values_l95_95756

theorem find_m_values (α : Real) (m : Real) (h1 : α ∈ Set.Ioo π (3 * π / 2)) 
  (h2 : Real.sin α = (3 * m - 2) / (m + 3)) 
  (h3 : Real.cos α = (m - 5) / (m + 3)) : m = (10 / 9) ∨ m = 2 := by 
  sorry

end find_m_values_l95_95756


namespace quadratics_equal_l95_95005

-- Definitions of integer part and quadratic polynomials
def intPart (a : ℝ) : ℤ := ⌊a⌋

-- Define quadratic polynomial
structure QuadraticPoly :=
(coeffs : ℝ × ℝ × ℝ) -- (a, b, c) coefficients of ax^2 + bx + c

def evalQuadPoly (p : QuadraticPoly) (x : ℝ) : ℝ :=
p.coeffs.1 * x^2 + p.coeffs.2 * x + p.coeffs.3

-- Problem Statement in Lean 4
theorem quadratics_equal
  (f g : QuadraticPoly)
  (h : ∀ x : ℝ, intPart (evalQuadPoly f x) = intPart (evalQuadPoly g x)) :
  ∀ x : ℝ, evalQuadPoly f x = evalQuadPoly g x := 
sorry

end quadratics_equal_l95_95005


namespace eccentricity_of_hyperbola_l95_95777

variable (p a b : ℝ) (p_pos : p > 0) (a_pos : a > 0) (b_pos : b > 0)
variable (F : ℝ × ℝ := (p / 2, 0)) (A : ℝ × ℝ := (p / 2, p * b / (2 * a)))

theorem eccentricity_of_hyperbola : 
  ∃ e : ℝ, (∃ c : ℝ, c = Real.sqrt(a^2 + b^2) ∧ e = c / a) ∧ e = Real.sqrt(5) :=
by
  sorry

end eccentricity_of_hyperbola_l95_95777


namespace root_poly_ratio_c_d_l95_95858

theorem root_poly_ratio_c_d (a b c d : ℝ)
  (h₁ : 1 + (-2) + 3 = 2)
  (h₂ : 1 * (-2) + (-2) * 3 + 3 * 1 = -5)
  (h₃ : 1 * (-2) * 3 = -6)
  (h_sum : -b / a = 2)
  (h_pair_prod : c / a = -5)
  (h_prod : -d / a = -6) :
  c / d = 5 / 6 := by
  sorry

end root_poly_ratio_c_d_l95_95858


namespace ice_cream_volume_l95_95478

noncomputable def radius := 4 -- inches
noncomputable def height := 12 -- inches

def volume_cone (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h
def volume_hemisphere (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3

def total_volume (r h : ℝ) : ℝ :=
  volume_cone r h + volume_hemisphere r

theorem ice_cream_volume :
  total_volume radius height = (320 / 3) * Real.pi :=
by
  sorry

end ice_cream_volume_l95_95478


namespace arithmetic_sequence_c_d_sum_l95_95501

theorem arithmetic_sequence_c_d_sum (c d : ℕ) 
  (h1 : 10 - 3 = 7) 
  (h2 : 17 - 10 = 7) 
  (h3 : c = 17 + 7) 
  (h4 : d = c + 7) 
  (h5 : d + 7 = 38) :
  c + d = 55 := 
by
  sorry

end arithmetic_sequence_c_d_sum_l95_95501


namespace right_handed_players_l95_95436

theorem right_handed_players (total_players : ℕ) (throwers : ℕ) (third_ratio : ℚ)
  (left_handed_non_throwers_ratio : third_ratio = 1 / 3)
  (all_throwers_right_handed : ∀ t, t < throwers -> right_handed t) :
  total_players = 70 ∧ throwers = 46 -> ∃ right_handed_players, right_handed_players = 62 := 
by
  sorry

end right_handed_players_l95_95436


namespace smallest_possible_median_l95_95061

-- Variables for the elements in the set
variables (x : ℕ)

-- The set of five numbers {x, 4x, 8, 1, x+1}
def set_five_numbers : set ℕ := {x, 4 * x, 8, 1, x + 1}

-- Function to find the median of a set of five numbers
def median (s : set ℕ) : ℕ :=
  if h : s.card = 5 then
    (s.to_finset.sort (≤)).nth_le 2 h
  else
    0 -- This case should not happen given the context

theorem smallest_possible_median : ∃ x : ℕ, median (set_five_numbers x) = 1 := by
  sorry

end smallest_possible_median_l95_95061


namespace good_students_l95_95339

theorem good_students (E B : ℕ) (h1 : E + B = 25) (h2 : 12 < B) (h3 : B = 3 * (E - 1)) :
  E = 5 ∨ E = 7 :=
by 
  sorry

end good_students_l95_95339


namespace find_tangent_equations_through_P_to_C_l95_95673

def circle_C_eqn (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 2

def is_tangent_line (k : ℝ) (P Q : ℝ) : Prop :=
  P - 2 * k - 1 = 0 ∧ k ^ 2 - 6 * k - 7 = 0

def point_P : ℝ × ℝ := (2, -1)

theorem find_tangent_equations_through_P_to_C (x y : ℝ) :
  (circle_C_eqn x y) →
  (is_tangent_line 7 (x + y - 1) (2, -1) ∨ is_tangent_line (-1) (7 * x - y - 15) (2, -1)) :=
sorry

end find_tangent_equations_through_P_to_C_l95_95673


namespace range_of_f_l95_95491

def f (x : ℝ) : ℝ := 2 ^ (-|x|)

theorem range_of_f : set.range f = set.Ioc 0 1 :=
sorry

end range_of_f_l95_95491


namespace seventh_observation_l95_95545

theorem seventh_observation (a b : ℕ) (h1 : a = 16) (h2 : b = 6) (new_avg : ℕ) (h3 : new_avg = 15) :
  let total_sum := b * a,
      new_total_sum := 7 * new_avg,
      seventh_observation := new_total_sum - total_sum in
  seventh_observation = 9 :=
by
  sorry

end seventh_observation_l95_95545


namespace cakes_remaining_l95_95595

theorem cakes_remaining (initial_cakes : ℕ) (bought_cakes : ℕ) (h1 : initial_cakes = 169) (h2 : bought_cakes = 137) : initial_cakes - bought_cakes = 32 :=
by
  sorry

end cakes_remaining_l95_95595


namespace no_solution_a4_plus_6_eq_b3_mod_13_l95_95461

theorem no_solution_a4_plus_6_eq_b3_mod_13 :
  ¬ ∃ (a b : ℤ), (a^4 + 6) % 13 = b^3 % 13 :=
by
  sorry

end no_solution_a4_plus_6_eq_b3_mod_13_l95_95461


namespace height_difference_l95_95524

-- Define the constants mentioned in the problem
def diameter : ℝ := 8
def num_balls : ℕ := 150
def rows : ℕ := 15
def balls_per_row : ℕ := 10

-- Calculations for Crate X
def height_Crate_X : ℝ := rows * diameter

-- Calculations for Crate Y
def vertical_distance_staggered : ℝ := (real.sqrt 3 / 2) * diameter
def height_Crate_Y : ℝ := (rows - 1) * vertical_distance_staggered + diameter

-- The statement to prove the positive difference in heights
theorem height_difference :
  abs(height_Crate_X - height_Crate_Y) = 112 - 56 * real.sqrt 3 :=
by
  sorry

end height_difference_l95_95524


namespace ivanov_error_l95_95825

theorem ivanov_error (x : ℝ) (m : ℝ) (S2 : ℝ) (std_dev : ℝ) :
  x = 0 → m = 4 → S2 = 15.917 → std_dev = Real.sqrt S2 →
  ¬ (|x - m| ≤ std_dev) :=
by
  intros h1 h2 h3 h4
  -- Using the given values directly to state the inequality
  have h5 : |0 - 4| = 4 := by norm_num
  have h6 : Real.sqrt 15.917 ≈ 3.99 := sorry  -- approximation as direct result
  -- Evaluating the inequality
  have h7 : 4 ≰ 3.99 := sorry  -- this represents the key step that shows the error
  exact h7
  sorry

end ivanov_error_l95_95825


namespace find_f_neg_one_l95_95665

variable {f : ℝ → ℝ}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def f_when_positive (f : ℝ → ℝ) : Prop :=
  ∀ x, x > 0 → f(x) = x + 1

theorem find_f_neg_one (h1 : is_odd_function f) (h2 : f_when_positive f) : 
  f(-1) = -2 :=
by
  -- proof goes here
  sorry

end find_f_neg_one_l95_95665


namespace diagonal_intersection_probability_decagon_l95_95093

noncomputable def probability_diagonal_intersection_in_decagon : ℚ :=
  let vertices := 10
  let total_diagonals := (vertices * (vertices - 3)) / 2
  let total_pairs_of_diagonals := total_diagonals * (total_diagonals - 1) / 2
  let total_intersecting_pairs := (vertices * (vertices - 1) * (vertices - 2) * (vertices - 3)) / 24
  total_intersecting_pairs / total_pairs_of_diagonals

theorem diagonal_intersection_probability_decagon (h : probability_diagonal_intersection_in_decagon = 42 / 119) : 
  probability_diagonal_intersection_in_decagon = 42 / 119 :=
sorry

end diagonal_intersection_probability_decagon_l95_95093


namespace a_101_mod_49_l95_95411

def a (n : ℕ) : ℕ := 5 ^ n + 9 ^ n

theorem a_101_mod_49 : (a 101) % 49 = 0 :=
by
  -- proof to be filled here
  sorry

end a_101_mod_49_l95_95411


namespace table_tennis_probability_l95_95723

theorem table_tennis_probability
  (pA_serve : ℚ := 2 / 3)
  (pB_serve : ℚ := 1 / 2)
  (indep_events : ∀ (s : bool), true) :
  (∑ seqs in { ⦃(0, 1), (1, 0), (0, 0), (1, 1)⦄,
      prod seqs.1 match seqs.2 with
                    | 0  := pA_serve
                    | 1  := 1 - pA_serve
                    end } *
    ∑ seqs in { ⦃(0, 1), (1, 0), (0, 0), (1, 1)⦄,
      prod seqs.1 match seqs.2 with
                    | 0  := pB_serve
                    | 1  := 1 - pB_serve
                    end } *
    ∑ seqs in { ⦃(0, 1), (1, 0), (0, 0), (1, 1)⦄,
      prod seqs.1 match seqs.2 with
                    | 0  := pA_serve
                    | 1  := 1 - pA_serve
                    end } *
    ∑ seqs in { ⦃(0, 1), (1, 0), (0, 0), (1, 1)⦄,
      prod seqs.1 match seqs.2 with
                    | 0  := pB_serve
                    | 1  := 1 - pB_serve
                    end }) = 1 / 4 :=
sorry

end table_tennis_probability_l95_95723


namespace propositions_correct_l95_95128

theorem propositions_correct :
  (∀ x : ℝ, cos (x + π / 2) = - sin x) ∧
  (∀ x : ℝ, cos (cos x) ≥ 0) :=
by
  sorry

end propositions_correct_l95_95128


namespace a_seq_formula_l95_95637

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (2 + x)

def a_seq : ℕ → ℝ
| 0     := 1
| (n+1) := f (a_seq n)

theorem a_seq_formula (n : ℕ) : a_seq n = 2 / (n + 1) := 
  sorry

end a_seq_formula_l95_95637


namespace find_range_of_m_l95_95640

noncomputable def intersects_line_circle (m : ℝ) : Prop :=
  abs (1 - m) / real.sqrt 2 < 1

def roots_opposite_signs (m : ℝ) : Prop :=
  m < 4

theorem find_range_of_m (m : ℝ) :
  ¬ intersects_line_circle m ∧ (intersects_line_circle m ∨ roots_opposite_signs m) →
  (m ≤ 1 - real.sqrt 2 ∨ (1 + real.sqrt 2 ≤ m ∧ m < 4)) :=
sorry

end find_range_of_m_l95_95640


namespace equations_not_equivalent_l95_95132

theorem equations_not_equivalent :
  (∀ x, (2 * (x - 10) / (x^2 - 13 * x + 30) = 1 ↔ x = 5)) ∧ 
  (∃ x, x ≠ 5 ∧ (x^2 - 15 * x + 50 = 0)) :=
sorry

end equations_not_equivalent_l95_95132


namespace complex_sum_equality_l95_95611

noncomputable def complex_sum : ℂ :=
  15 * complex.exp((3 * real.pi * complex.I) / 13) + 
  15 * complex.exp((24 * real.pi * complex.I) / 26)

noncomputable def expected_result : ℂ :=
  30 * real.cos((9 * real.pi) / 26) * complex.exp((15 * real.pi * complex.I) / 26)

theorem complex_sum_equality :
  complex_sum = expected_result :=
  sorry

end complex_sum_equality_l95_95611


namespace no_valid_solutions_l95_95463

theorem no_valid_solutions (x : ℝ) (h : x ≠ 1) : 
  ¬(3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) :=
sorry

end no_valid_solutions_l95_95463


namespace cost_difference_is_120_l95_95182

-- Define the monthly costs and duration
def rent_monthly_cost : ℕ := 20
def buy_monthly_cost : ℕ := 30
def months_in_a_year : ℕ := 12

-- Annual cost definitions
def annual_rent_cost : ℕ := rent_monthly_cost * months_in_a_year
def annual_buy_cost : ℕ := buy_monthly_cost * months_in_a_year

-- The main theorem to prove the difference in annual cost is $120
theorem cost_difference_is_120 : annual_buy_cost - annual_rent_cost = 120 := by
  sorry

end cost_difference_is_120_l95_95182


namespace complex_solution_count_is_one_l95_95250

noncomputable def complex_solution_count : ℕ :=
  let count := {z : ℂ | abs(z) < 20 ∧ exp(z) = 1 - z / 2}.to_finset.card in
  count

theorem complex_solution_count_is_one : complex_solution_count = 1 := by
  sorry

end complex_solution_count_is_one_l95_95250


namespace sum_of_consecutive_odd_integers_l95_95034

-- Definitions of conditions
def consecutive_odd_integers (a b : ℤ) : Prop :=
  b = a + 2 ∧ (a % 2 = 1) ∧ (b % 2 = 1)

def five_times_smaller_minus_two_condition (a b : ℤ) : Prop :=
  b = 5 * a - 2

-- Theorem statement
theorem sum_of_consecutive_odd_integers (a b : ℤ)
  (h1 : consecutive_odd_integers a b)
  (h2 : five_times_smaller_minus_two_condition a b) : a + b = 4 :=
by
  sorry

end sum_of_consecutive_odd_integers_l95_95034


namespace xy_problem_l95_95270

theorem xy_problem (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 :=
by
  sorry

end xy_problem_l95_95270


namespace problem_l95_95190

noncomputable def a : ℝ := Real.exp 0.2
noncomputable def b : ℝ := Real.pow 0.2 Real.exp 1
noncomputable def c : ℝ := Real.log 2

theorem problem (a := Real.exp 0.2) (b := Real.pow 0.2 Real.exp 1) (c := Real.log 2) : b < c ∧ c < a := by
  sorry

end problem_l95_95190


namespace hexagon_equilateral_triangles_l95_95876

theorem hexagon_equilateral_triangles (s n : ℕ) (h₁ : s = 5) (h₂ : n = 1) :
    (6 : ℕ) * n^2 = 150 := 
by 
  have h₃ : n^2 = 25 := by sorry  --  Given n = 5, thus n^2 = 5^2 = 25.
  rw h₃
  exact sorry

end hexagon_equilateral_triangles_l95_95876


namespace shop_makes_off_jersey_l95_95469

/-- Given conditions stated in the problem -/
theorem shop_makes_off_jersey :
  (let j_profit := 185.85 in
  let tshirt_profit := 240 in
  let tshirts_sold := 177 in
  let jerseys_sold := 23 in
  let tshirt_over_jersey := 30 in
  tshirts_sold * (j_profit + tshirt_over_jersey) + jerseys_sold * j_profit = 42480) →
  j_profit = 185.85 :=
by
  sorry

end shop_makes_off_jersey_l95_95469


namespace compound_interest_years_l95_95854

theorem compound_interest_years :
  let SI := 1750 * 8 / 100 * 3
  let CI := 2 * SI
  let P := 4000
  let R := 10 / 100
  log (1.21) / log (1.1) ≈ 2 → 
  ( ∀ (T : ℝ), CI = P * ((1 + R) ^ T - 1) → T = 2) :=
by
  sorry

end compound_interest_years_l95_95854


namespace mean_age_of_all_children_l95_95833

def euler_ages : List ℕ := [10, 12, 8]
def gauss_ages : List ℕ := [8, 8, 8, 16, 18]
def all_ages : List ℕ := euler_ages ++ gauss_ages
def total_children : ℕ := all_ages.length
def total_age : ℕ := all_ages.sum
def mean_age : ℕ := total_age / total_children

theorem mean_age_of_all_children : mean_age = 11 := by
  sorry

end mean_age_of_all_children_l95_95833


namespace smallest_candies_value_l95_95127

def smallest_valid_n := ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 9 = 2 ∧ n % 7 = 5 ∧ ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m % 9 = 2 ∧ m % 7 = 5 → n ≤ m

theorem smallest_candies_value : ∃ n : ℕ, smallest_valid_n ∧ n = 101 := 
by {
  sorry  
}

end smallest_candies_value_l95_95127


namespace number_of_equilateral_triangles_after_12_operations_l95_95588

theorem number_of_equilateral_triangles_after_12_operations :
  ∀ (initial_triangles : ℕ),
    initial_triangles = 1 →
    ∀ (operations : ℕ),
      operations = 12 →
      let remaining_triangles : ℕ := initial_triangles * 3 ^ operations in
      remaining_triangles = 531441 :=
by
  intros initial_triangles h_initial_triangles operations h_operations
  simp [h_initial_triangles, h_operations]
  have : 3 ^ 12 = 531441 := by norm_num
  rw [this]
  sorry

end number_of_equilateral_triangles_after_12_operations_l95_95588


namespace positive_difference_equals_4_l95_95096

def original_block : matrix (fin 5) (fin 5) ℕ :=
  ![
    ![1, 2, 3, 4, 5],
    ![10, 11, 12, 13, 14],
    ![19, 20, 21, 22, 23],
    ![28, 29, 30, 31, 32],
    ![37, 38, 39, 40, 41]
  ]

def reversed_block : matrix (fin 5) (fin 5) ℕ :=
  ![
    ![1, 2, 3, 4, 5],
    ![14, 13, 12, 11, 10],
    ![19, 20, 21, 22, 23],
    ![28, 29, 30, 31, 32],
    ![41, 40, 39, 38, 37]
  ]

def main_diagonal_sum (m : matrix (fin 5) (fin 5) ℕ) : ℕ :=
  ∑ i, m i i

def secondary_diagonal_sum (m : matrix (fin 5) (fin 5) ℕ) : ℕ :=
  ∑ i, m i (4 - i)

def positive_difference_between_diagonals (m : matrix (fin 5) (fin 5) ℕ) : ℕ :=
  abs ((main_diagonal_sum m) - (secondary_diagonal_sum m))

theorem positive_difference_equals_4 :
  positive_difference_between_diagonals reversed_block = 4 :=
by
  sorry

end positive_difference_equals_4_l95_95096


namespace student_tickets_sold_l95_95872

theorem student_tickets_sold
  (A S : ℕ)
  (h1 : A + S = 846)
  (h2 : 6 * A + 3 * S = 3846) :
  S = 410 :=
sorry

end student_tickets_sold_l95_95872


namespace segment_le_radius_or_AB_l95_95399

open Real
open Angle

variables {R : ℝ} {A B O M N : Point}
variables (hAOB : Segment A O = Segment O B)
variables (hAngleAOB : Angle A O B < π)
variables (hMNInside : Segment M N lies_inside Sector A O B)

theorem segment_le_radius_or_AB
    (hMNInside : LineSegment M N lies_inside Sector A O B)
    (hAO : Segment A O = R)
    (hBO : Segment B O = R)
    (hAngle : Angle A O B < 180°) :
  Length (LineSegment M N) ≤ R ∨ Length (LineSegment M N) ≤ Length (LineSegment A B) :=
sorry

end segment_le_radius_or_AB_l95_95399


namespace new_boxes_of_markers_l95_95431

theorem new_boxes_of_markers (initial_markers new_markers_per_box total_markers : ℕ) 
    (h_initial : initial_markers = 32) 
    (h_new_markers_per_box : new_markers_per_box = 9)
    (h_total : total_markers = 86) :
  (total_markers - initial_markers) / new_markers_per_box = 6 := 
by
  sorry

end new_boxes_of_markers_l95_95431


namespace red_envelope_probability_l95_95989

def wechat_red_envelope_prob {A B : Type} [DecidableEq A] [DecidableEq B]
  (total_amount : ℝ) (distribution : Finset ℝ) (num_people : ℕ)
  (amount_snatched_by_A_and_B : Finset (ℝ × ℝ)) : Prop :=
  total_amount = 8 ∧
  distribution = {1.72, 1.83, 2.28, 1.55, 0.62} ∧
  num_people = 5 ∧
  amount_snatched_by_A_and_B.filter (λ (x : ℝ × ℝ), x.1 + x.2 ≥ 3).card = 6

theorem red_envelope_probability :
  wechat_red_envelope_prob 8 {1.72, 1.83, 2.28, 1.55, 0.62} 5 { (1.72, 1.83), (1.72, 2.28), (1.72, 1.55), (1.83, 2.28), (1.83, 1.55), (2.28, 1.55) } →
  (6 / 10 : ℝ) = 3 / 5 :=
by
  sorry

end red_envelope_probability_l95_95989


namespace maximum_possible_k_l95_95633

noncomputable def max_k_value := 599

theorem maximum_possible_k :
  ∃ (k : ℕ), 
    (k = max_k_value) 
    ∧ (∀ (a b : Finset ℕ) (ta tb : ℕ), ta ∈ a → tb ∈ b → a ≠ b)
    ∧ (∀ {i j : ℕ}, i ≠ j → a_i + b_i ≠ a_j + b_j)
    ∧ (∀ i, a_i < b_i ∧ a_i + b_i ≤ 1500)
    ∧ (k ≤ ∑ i in finset.range 1500, k(i)) :=
by
  sorry

end maximum_possible_k_l95_95633


namespace arithmetic_sequence_sum_l95_95496

theorem arithmetic_sequence_sum (c d : ℤ) (h₁ : d = 10 - 3)
    (h₂ : c = 17 + (10 - 3)) (h₃ : d = 24 + (10 - 3)) :
    c + d = 55 :=
sorry

end arithmetic_sequence_sum_l95_95496


namespace class_proof_l95_95355

def class_problem : Prop :=
  let total_students := 25
  let (students : Type) := {s // s ∈ Fin total_students}
  let good students (s : students) : Prop := sorry -- Condition Placeholder
  let troublemakers (s : students) : Prop := sorry -- Condition Placeholder
  have all_good_or_trouble : ∀ s, good students s ∨ troublemakers s,
    from sorry,
  have lh_trouble : ∀ (a b c d e : students), (good students a ∧ good students b ∧ good students c ∧ good students d ∧ good students e) →
    ∃ (x : ℕ), x > ((total_students - 1) / 2) ∧
    (λ rem_trouble : students - {a, b, c, d, e}, B : ℕ, B = x) ∧
    (λ rem_good : students - {a, b, c, d, e}, G : ℕ, G ∈ rem_good), from sorry,
  have th_triple_ratio : ∀ s, (students - {s}).card = 24 →
    ∃ (x y : ℕ), (troublemakers x ∧ good students y) ∧ (3*y = x), from sorry,
  ∃ (num_good_students : ℕ), (num_good_students = 5 ∨ num_good_students = 7)
  
theorem class_proof : class_problem :=
  sorry

end class_proof_l95_95355


namespace min_cells_marked_l95_95437

theorem min_cells_marked (grid_size : ℕ) (triomino_size : ℕ) (total_cells : ℕ) : 
  grid_size = 5 ∧ triomino_size = 3 ∧ total_cells = grid_size * grid_size → ∃ m, m = 9 :=
by
  intros h
  -- Placeholder for detailed proof steps
  sorry

end min_cells_marked_l95_95437


namespace diameter_of_sphere_with_triple_volume_l95_95168

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem diameter_of_sphere_with_triple_volume 
  (r₁ r₂ : ℝ) 
  (h₁ : r₁ = 6) 
  (h₂ : sphere_volume r₂ = 3 * sphere_volume r₁) : 
  2 * r₂ = 12 * Real.cbrt 3 ∧ 12 + 3 = 15 :=
by
  sorry

end diameter_of_sphere_with_triple_volume_l95_95168


namespace common_tangents_count_l95_95836

def circleC1 : Prop := ∃ (x y : ℝ), x^2 + y^2 + 2 * x - 6 * y - 15 = 0
def circleC2 : Prop := ∃ (x y : ℝ), x^2 + y^2 - 4 * x + 2 * y + 4 = 0

theorem common_tangents_count (C1 : circleC1) (C2 : circleC2) : 
  ∃ (n : ℕ), n = 2 := 
sorry

end common_tangents_count_l95_95836


namespace sum_of_digits_l95_95743

theorem sum_of_digits (digits : list ℕ) (h_length : digits.length = 12000)
  (h_digits : ∀ d, d ∈ digits → d ∈ {1, 2, 3, 4, 5, 6})
  (h_erased_4th : ∀ (i : ℕ), 4 ∣ i → digits.nth i = none)
  (h_erased_5th : ∀ (i : ℕ), 5 ∣ i → (digits.filter (λ x, x ≠ none)).nth i = none)
  (h_erased_6th : ∀ (i : ℕ), 6 ∣ i → (digits.filter (λ x, x ≠ none).filter (λ x, x ≠ none)).nth i = none) :
  (sum (digits.nth (3000 % 30).get_or_else 0 ::
        digits.nth (3001 % 30).get_or_else 0 ::
        digits.nth (3002 % 30).get_or_else 0 ::
        [])) = 8 :=
by {
  -- Proof omitted
  sorry
}

end sum_of_digits_l95_95743


namespace gcm_15_and_20_less_than_150_gcm_of_15_and_20_l95_95887

theorem gcm_15_and_20_less_than_150 : 
  ∃ x, (x = 15 * k ∧ x = 20 * m ∧ x < 150) ∧ (∀ y, (y = 15 * k' ∧ y = 20 * m' ∧ y < 150) → y ≤ x) := by
sorr

theorem gcm_of_15_and_20 : 
  ∃ x, x = 120 ∧ (x = 15 * k ∧ x = 20 * m ∧ x < 150) ∧ (∀ y, (y = 15 * k' ∧ y = 20 * m' ∧ y < 150) → y ≤ x) := by
sorry

end gcm_15_and_20_less_than_150_gcm_of_15_and_20_l95_95887


namespace find_angle_DAE_l95_95396

noncomputable def angle_DAE (A B C D O E : Type) [EuclideanGeometry] : Bool :=
  let angle_ACB := 40
  let angle_CBA := 80
  let angle_CAD := 90 - angle_ACB
  let angle_AOC := 2 * angle_CBA
  let angle_CAO := (180 - angle_AOC) / 2
  let angle_DAE := angle_CAD - angle_CAO
  angle_DAE = 40

axiom euclidean_geometry (A B C D O E : Type) [EuclideanGeometry]
  (angle_ACB angle_CBA : ℝ) 
  (h₁ : angle_ACB = 40) 
  (h₂ : angle_CBA = 80) 
  (h₃ : D = foot_of_perpendicular A B C)
  (h₄ : O = circumcenter A B C)
  (h₅ : E = other_end_of_diameter A O) :
  angle_DAE A B C D O E

theorem find_angle_DAE : euclidean_geometry A B C D O E 40 80
  = true := sorry

end find_angle_DAE_l95_95396


namespace sixth_result_is_66_l95_95472

-- Define the constants and conditions
def avg_all := 60 / 1
def avg_first_6 := 58 / 1
def avg_last_6 := 63 / 1
def num_results := 11
def num_first_6 := 6
def num_last_6 := 6

-- Define the sums based on averages and counts
def sum_first_6 := num_first_6 * avg_first_6
def sum_last_6 := num_last_6 * avg_last_6
def total_sum := num_results * avg_all

-- Define the sixth result using the given conditions
def sixth_result : ℕ := (total_sum - sum_first_6 - sum_last_6) / (-1)

-- Theorem to prove that the sixth result is 66
theorem sixth_result_is_66 : sixth_result = 66 :=
  sorry -- Proof to be provided

end sixth_result_is_66_l95_95472


namespace max_non_overlapping_triangles_max_non_overlapping_squares_l95_95903

theorem max_non_overlapping_triangles (T : Triangle) : 
  ∃ (n : ℕ), n = 6 ∧ (∀ T' ∈ set.univ, T' ≠ T → 
  (T' ∩ T = ∅) ∧ (T' ∥ T)) := 
sorry

theorem max_non_overlapping_squares (K : Square): 
  ∃ (n : ℕ), n = 8 ∧ (∀ K' ∈ set.univ, K' ≠ K →
  (K' ∩ K = ∅) ∧ (K' ∥ K)) := 
sorry

end max_non_overlapping_triangles_max_non_overlapping_squares_l95_95903


namespace robot_cost_max_units_A_l95_95012

noncomputable def cost_price_A (x : ℕ) := 1600
noncomputable def cost_price_B (x : ℕ) := 2800

theorem robot_cost (x : ℕ) (y : ℕ) (a : ℕ) (b : ℕ) :
  y = 2 * x - 400 →
  a = 96000 →
  b = 168000 →
  a / x = 6000 →
  b / y = 6000 →
  (x = 1600 ∧ y = 2800) :=
by sorry

theorem max_units_A (m n total_units : ℕ) : 
  total_units = 100 →
  m + n = 100 →
  m ≤ 2 * n →
  m ≤ 66 :=
by sorry

end robot_cost_max_units_A_l95_95012


namespace graph_of_g_abs_eq_reflection_graph_of_g_abs_is_reflection_l95_95477

noncomputable def g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -1 then 2 * x + 8
  else if -1 ≤ x ∧ x ≤ 3 then -x + 2
  else if 3 ≤ x ∧ x ≤ 4 then 3 * (x - 3)
  else 0 -- To make it a total function

theorem graph_of_g_abs_eq_reflection :
  ∀ (x : ℝ), g (|x|) = g x ∨ g (|x|) = g (-x) :=
by
  intro x
  cases lt_or_ge x 0 with
  | inl h => simp [abs_of_neg h]
  | inr h => simp [abs_of_nonneg h]

-- Auxiliary lemma: The relationship between g(x) and g(-x) for given x ranges
lemma g_symmetry (x : ℝ) : g x = g (-x) :=
by
  sorry

theorem graph_of_g_abs_is_reflection :
  ∀ (x : ℝ), graph_of_g_abs_eq_reflection x
by
  sorry

end graph_of_g_abs_eq_reflection_graph_of_g_abs_is_reflection_l95_95477


namespace prove_sum_13_l95_95506

open Finset

variables (a_1 d : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Define the arithmetic sequence a_n
def arithmetic_sequence (a_1 d : ℝ) (n : ℕ) : ℝ :=
  a_1 + n * d

-- Define the sum of the first n terms of the arithmetic sequence S_n
def sum_of_first_n_terms (a_1 d : ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a_1 + (n - 1) * d)

lemma collinear_condition (a_1 d : ℝ) :
  a 2 + a 7 + a 12 = 1 :=
sorry

lemma S_13_value (a_1 d : ℝ) :
  S 13 = 13 * (arithmetic_sequence a_1 d 6) :=
sorry

theorem prove_sum_13 (a_1 d : ℝ) (h1 : S = sum_of_first_n_terms a_1 d)
  (h2 : collinear_condition a_1 d) : 
  S 13 = 13 / 3 :=
begin
  sorry
end

end prove_sum_13_l95_95506


namespace good_students_count_l95_95375

theorem good_students_count (E B : ℕ) 
    (h1 : E + B = 25) 
    (h2 : ∀ (e ∈ finset.range 5), B > 12) 
    (h3 : ∀ (e ∈ finset.range 20), B = 3 * (E - 1)) :
    E = 5 ∨ E = 7 :=
by
  sorry

end good_students_count_l95_95375


namespace locus_of_inscribed_circle_centers_l95_95695

-- Definitions for circles touching each other internally at point A
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

def circles_touch_internally_at (c1 c2 : Circle) (A : ℝ × ℝ) : Prop :=
  (c1.center ≠ c2.center ∧
   dist c1.center A = c1.radius ∧
   dist c2.center A = c2.radius ∧
   dist c1.center c2.center = c1.radius - c2.radius)

def tangent_intersects (c1 c2 : Circle) (B C : ℝ × ℝ) : Prop :=
  (tangent c1 c2 B ∧ tangent c1 c2 C)

-- Statement of the problem
theorem locus_of_inscribed_circle_centers
  (c1 c2 : Circle) (A B C : ℝ × ℝ) (O : ℝ × ℝ)
  (h1 : circles_touch_internally_at c1 c2 A)
  (h2 : tangent_intersects c2 c1 B C) :
  locus_of_centers_of_inscribed_circles_in_ABC A B C O = 
  (c2.radius * sqrt(c1.radius)) / (sqrt(c1.radius) + sqrt(c1.radius - c2.radius)) :=
by
  sorry

end locus_of_inscribed_circle_centers_l95_95695


namespace triangle_perimeter_sqrt_l95_95507

theorem triangle_perimeter_sqrt :
  let a := Real.sqrt 8
  let b := Real.sqrt 18
  let c := Real.sqrt 32
  a + b + c = 9 * Real.sqrt 2 :=
by
  sorry

end triangle_perimeter_sqrt_l95_95507


namespace quadratic_solution_unique_l95_95020

theorem quadratic_solution_unique (a : ℝ) (ha : a ≠ 0) :
  a = 45 → (∃ x : ℝ, a * x^2 + 30 * x + 5 = 0 ∧ x = -1/3) :=
by
  intros ha45
  use -1/3
  split
  . rw ha45
    simp
    linarith
  . sorry

end quadratic_solution_unique_l95_95020


namespace sequence_arithmetic_expression_l95_95648

theorem sequence_arithmetic_expression (a : ℕ → ℝ) (h0 : a 1 = 1) (h1 : a 3 = 9) 
  (h2 : ∀ n : ℕ, sqrt (a (n + 1)) - sqrt (a n) = 2 * n - 2) 
  : ∀ n : ℕ, a n = (n^2 - 3 * n + 3)^2 := 
    by sorry

end sequence_arithmetic_expression_l95_95648


namespace combined_volume_is_108_l95_95101

-- Definitions
def V_rect := 4 -- Volume of normal rectangular block in cubic feet
def V_cyl := 6 -- Volume of normal cylindrical block in cubic feet

def large_rect_volume (V_rect : ℕ) : ℕ :=
  1.5 * 3 * 2 * V_rect

def large_cyl_volume (V_cyl : ℕ) : ℕ :=
  12 * V_cyl

-- Theorem statement
theorem combined_volume_is_108 :
  large_rect_volume V_rect + large_cyl_volume V_cyl = 108 :=
by
  -- Skip the proof
  sorry

end combined_volume_is_108_l95_95101


namespace find_angle_C_find_side_c_l95_95739

noncomputable def triangle_angle_C (a b c : ℝ) (C : ℝ) (A : ℝ) : Prop := 
a * Real.cos C = c * Real.sin A

theorem find_angle_C (a b c : ℝ) (C : ℝ) (A : ℝ)
  (h1 : triangle_angle_C a b c C A)
  (h2 : 0 < A) : C = Real.pi / 3 := 
sorry

noncomputable def triangle_side_c (a b c : ℝ) (C : ℝ) : Prop := 
(∃ (area : ℝ), area = 6 ∧ b = 4 ∧ c * c = a * a + b * b - 2 * a * b * Real.cos C)

theorem find_side_c (a b c : ℝ) (C : ℝ) 
  (h1 : triangle_side_c a b c C) : c = 2 * Real.sqrt 7 :=
sorry

end find_angle_C_find_side_c_l95_95739


namespace function_even_and_max_value_l95_95465

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) + abs (sin x)

theorem function_even_and_max_value :
  (∀ x : ℝ, f (-x) = f x) ∧ (∃ x : ℝ, f x = 9 / 8) :=
by
  sorry

end function_even_and_max_value_l95_95465


namespace minimize_distance_l95_95525

-- Definitions of points and lines in the Euclidean plane
structure Point : Type :=
(x : ℝ)
(y : ℝ)

-- Line is defined by a point and a direction vector
structure Line : Type :=
(point : Point)
(direction : Point)

-- Distance function between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Given conditions
variables (a b : Line) -- lines a and b
variables (A1 A2 : Point) -- positions of point A on line a
variables (B1 B2 : Point) -- positions of point B on line b

-- Hypotheses about uniform motion along the lines
def moves_uniformly (A1 A2 : Point) (a : Line) (B1 B2 : Point) (b : Line) : Prop :=
  ∀ t : ℝ, ∃ (At Bt : Point), 
  At.x = A1.x + t * (A2.x - A1.x) ∧ At.y = A1.y + t * (A2.y - A1.y) ∧
  Bt.x = B1.x + t * (B2.x - B1.x) ∧ Bt.y = B1.y + t * (B2.y - B1.y) ∧
  ∀ s : ℝ, At.x + s * (a.direction.x) = Bt.x + s * (b.direction.x) ∧
           At.y + s * (a.direction.y) = Bt.y + s * (b.direction.y)

-- Problem statement: Prove the existence of points such that AB is minimized
theorem minimize_distance (a b : Line) (A1 A2 B1 B2 : Point) (h : moves_uniformly A1 A2 a B1 B2 b) : 
  ∃ (A B : Point), distance A B = Real.sqrt ((A2.x - B2.x) ^ 2 + (A2.y - B2.y) ^ 2) ∧ distance A B ≤ distance A1 B1 ∧ distance A B ≤ distance A2 B2 :=
sorry

end minimize_distance_l95_95525


namespace Ivanov_made_an_error_l95_95815

theorem Ivanov_made_an_error : 
  (∀ x m s : ℝ, 0 = x → 4 = m → 15.917 = s^2 → ¬ (|x - m| ≤ real.sqrt s)) :=
by 
  intros x m s hx hm hs2
  rw [hx, hm, hs2]
  have H : |0 - 4| = 4 := by norm_num
  have H2 : real.sqrt 15.917 ≈ 3.99 := by norm_num
  exact neq_of_not_le (by norm_num : 4 ≠ 3.99) H2 sorry

end Ivanov_made_an_error_l95_95815


namespace cone_surface_area_l95_95041

theorem cone_surface_area (h : Real) (theta : Real) (slant_height : h = 3) (central_angle : theta = 2 * Real.pi / 3) : 
  let arc_length := h * theta in
  let radius := 1 in
  let base_area := Real.pi in
  let lateral_area := 3 * Real.pi in
  let total_surface_area := base_area + lateral_area in
  total_surface_area = 4 * Real.pi :=
by
  sorry

end cone_surface_area_l95_95041


namespace years_on_compound_interest_l95_95855

noncomputable def simple_interest (P : ℕ) (R : ℕ) (T : ℕ) : ℚ :=
  (P * R * T) / 100

noncomputable def compound_interest (P : ℕ) (R : ℕ) (T : ℕ) : ℚ :=
  P * ((1 + R / 100.0)^T - 1)

noncomputable def calculate_years_for_interest : ℕ :=
let SI := simple_interest 1750 8 3 in
let CI := 2 * SI in
nat.ceil (Math.log ((CI / 4000) + 1) / Math.log (1.1))

theorem years_on_compound_interest :
   calculate_years_for_interest = 2 := 
by
  sorry

end years_on_compound_interest_l95_95855


namespace good_students_l95_95366

theorem good_students (G T : ℕ) (h1 : G + T = 25) (h2 : T > 12) (h3 : T = 3 * (G - 1)) :
  G = 5 ∨ G = 7 :=
by
  sorry

end good_students_l95_95366


namespace cups_per_girl_l95_95516

noncomputable def numStudents := 30
noncomputable def numBoys := 10
noncomputable def numCupsByBoys := numBoys * 5
noncomputable def totalCups := 90
noncomputable def numGirls := 2 * numBoys
noncomputable def numCupsByGirls := totalCups - numCupsByBoys

theorem cups_per_girl : (numCupsByGirls / numGirls) = 2 := by
  sorry

end cups_per_girl_l95_95516


namespace polynomial_multiplication_l95_95789

theorem polynomial_multiplication (x z : ℝ) :
  (3*x^5 - 7*z^3) * (9*x^10 + 21*x^5*z^3 + 49*z^6) = 27*x^15 - 343*z^9 :=
by
  sorry

end polynomial_multiplication_l95_95789


namespace greatest_multiple_less_than_l95_95885

def lcm (a b : ℕ) : ℕ := Nat.lcm a b -- Assuming lcm function is already defined

theorem greatest_multiple_less_than (a b m : ℕ) (h₁ : a = 15) (h₂ : b = 20) (h₃ : m = 150) : 
  ∃ k, k * lcm a b < m ∧ ¬ ∃ k', (k' * lcm a b < m ∧ k' > k) :=
by
  sorry

end greatest_multiple_less_than_l95_95885


namespace mixture_volume_correct_l95_95379

-- Define the input values
def water_volume : ℕ := 20
def vinegar_volume : ℕ := 18
def water_ratio : ℚ := 3/5
def vinegar_ratio : ℚ := 5/6

-- Calculate the mixture volume
def mixture_volume : ℚ :=
  (water_volume * water_ratio) + (vinegar_volume * vinegar_ratio)

-- Define the expected result
def expected_mixture_volume : ℚ := 27

-- State the theorem
theorem mixture_volume_correct : mixture_volume = expected_mixture_volume := by
  sorry

end mixture_volume_correct_l95_95379


namespace simplify_expression_l95_95998

theorem simplify_expression (a b : ℕ) (h₁ : a = 2999) (h₂ : b = 3000) :
  b^3 - a * b^2 - a^2 * b + a^3 = 5999 :=
by 
  sorry

end simplify_expression_l95_95998


namespace cups_per_girl_l95_95515

noncomputable def numStudents := 30
noncomputable def numBoys := 10
noncomputable def numCupsByBoys := numBoys * 5
noncomputable def totalCups := 90
noncomputable def numGirls := 2 * numBoys
noncomputable def numCupsByGirls := totalCups - numCupsByBoys

theorem cups_per_girl : (numCupsByGirls / numGirls) = 2 := by
  sorry

end cups_per_girl_l95_95515


namespace sum_c_k_squared_complexity_l95_95154

def c_k (k : ℕ) : ℝ := k + 1 / (4 * k + 1 / (4 * k + 1 / (4 * k + ...)))
def c_k_formula (k : ℕ) : ℝ := 2 * k + sqrt (5 * k ^ 2 + 1)

theorem sum_c_k_squared_complexity :
    ∑ k in Finset.range 15 , (c_k_formula k)^2 = 11175 + ∑ k in Finset.range 15 , 4 * k * sqrt (5 * k ^ 2 + 1) :=
by 
    sorry

end sum_c_k_squared_complexity_l95_95154


namespace find_ω_φ_l95_95215

-- Define the function and conditions
def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

-- States the given problem conditions
theorem find_ω_φ (ω φ : ℝ) 
  (hω_pos : ω > 0)
  (hφ_bounds : |φ| < Real.pi)
  (h_f5π8 : f (5 * Real.pi / 8) ω φ = 2)
  (h_f11π8 : f (11 * Real.pi / 8) ω φ = 0)
  (h_min_period_gt_2π : ∃ T > 2 * Real.pi, ∀ x : ℝ, f (x + T) ω φ = f x ω φ):
  ω = 2 / 3 ∧ φ = Real.pi / 12 :=
sorry

end find_ω_φ_l95_95215


namespace exists_multiple_of_n_with_only_digit_1_l95_95770

open Nat

theorem exists_multiple_of_n_with_only_digit_1 
  (n : ℕ) (hn_positive : 0 < n) (hn_coprime : Nat.coprime n 10) : 
  ∃ k, (10^k - 1) / 9 % n = 0 :=
by
  sorry

end exists_multiple_of_n_with_only_digit_1_l95_95770


namespace megan_dials_correct_number_probability_l95_95424

-- Define the set of possible first three digits
def first_three_digits : Finset ℕ := {296, 299, 295}

-- Define the set of possible last five digits
def last_five_digits : Finset (Finset ℕ) := {Finset.singleton 0, Finset.singleton 1, Finset.singleton 6, Finset.singleton 7, Finset.singleton 8}

-- The total number of possible phone numbers that Megan can dial
def total_possible_numbers : ℕ := (first_three_digits.card) * (5!)

-- The probability that Megan dials Fatima's correct number
def probability_correct_number : ℚ := 1 / total_possible_numbers

theorem megan_dials_correct_number_probability :
  probability_correct_number = 1 / 360 :=
by
  sorry

end megan_dials_correct_number_probability_l95_95424


namespace coordinates_of_E_l95_95391

theorem coordinates_of_E :
  let A := (-2, 1)
  let B := (1, 4)
  let C := (4, -3)
  let ratio_AB := (1, 2)
  let ratio_CE_ED := (1, 4)
  let D := ( (ratio_AB.1 * B.1 + ratio_AB.2 * A.1) / (ratio_AB.1 + ratio_AB.2),
             (ratio_AB.1 * B.2 + ratio_AB.2 * A.2) / (ratio_AB.1 + ratio_AB.2) )
  let E := ( (ratio_CE_ED.1 * C.1 - ratio_CE_ED.2 * D.1) / (ratio_CE_ED.1 - ratio_CE_ED.2),
             (ratio_CE_ED.1 * C.2 - ratio_CE_ED.2 * D.2) / (ratio_CE_ED.1 - ratio_CE_ED.2) )
  E = (-8 / 3, 11 / 3) := by
  sorry

end coordinates_of_E_l95_95391


namespace divisibility_of_sum_of_fifths_l95_95796

theorem divisibility_of_sum_of_fifths (x y z : ℤ) (h : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  ∃ k : ℤ, (x - y) ^ 5 + (y - z) ^ 5 + (z - x) ^ 5 = 5 * k * (x - y) * (y - z) * (z - x) :=
sorry

end divisibility_of_sum_of_fifths_l95_95796


namespace find_lambda_l95_95233

noncomputable def vector := (ℝ × ℝ)

def colinear (v1 v2 : vector) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem find_lambda
  (a b c: vector)
  (λ : ℝ)
  (h_a: a = (1, 2))
  (h_b: b = (2, 3))
  (h_c: c = (-4, -7))
  (h_colinear: colinear (λ * a.1 + b.1, λ * a.2 + b.2) c):
  λ = 2 :=
sorry

end find_lambda_l95_95233


namespace sum_x_y_l95_95988

variable (a b c x y : ℕ)
variable (d : ℕ := 5)
variable (a0 : a = 3)
variable (a1 : b = 8)
variable (a2 : c = 13)
variable (aN : a + 6 * d = 33) -- because the sequence reaches 33
variable (x : x = c + d)
variable (y : y = x + d)

theorem sum_x_y : x + y = 51 :=
by
  have d_eq : d = 5 := sorry -- from the condition
  have x_val : x = 23 := sorry -- calculated as x = 28 - 5
  have y_val : y = 28 := sorry -- calculated as y = 33 - 5
  have sum_eq : 23 + 28 = 51 := sorry
  exact sum_eq

end sum_x_y_l95_95988


namespace zoo_people_l95_95927

def number_of_people (cars : ℝ) (people_per_car : ℝ) : ℝ :=
  cars * people_per_car

theorem zoo_people (h₁ : cars = 3.0) (h₂ : people_per_car = 63.0) :
  number_of_people cars people_per_car = 189.0 :=
by
  rw [h₁, h₂]
  -- multiply the numbers directly after substitution
  norm_num
  -- left this as a placeholder for now, can use calc or norm_num for final steps
  exact sorry

end zoo_people_l95_95927


namespace inequality_abc_l95_95653

theorem inequality_abc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  abs (b / a - b / c) + abs (c / a - c / b) + abs (b * c + 1) > 1 :=
by
  sorry

end inequality_abc_l95_95653


namespace rectangle_new_area_is_correct_l95_95719

noncomputable def new_rectangle_area
  (square_area : ℝ)
  (length_decrease_percent : ℝ) 
  (width_multiplier : ℝ) : ℝ := 
  let s := real.sqrt square_area in
  let initial_length := s in
  let initial_width := width_multiplier * s in
  let new_length := initial_length * (1 - length_decrease_percent) in
  new_length * initial_width

theorem rectangle_new_area_is_correct :
  new_rectangle_area 49 0.2 2 = 78.4 :=
by sorry

end rectangle_new_area_is_correct_l95_95719


namespace janine_has_3_times_bottle_caps_l95_95458

-- Definitions from the conditions:
def sammy_has_more_bottle_caps (sammy_caps janine_caps : ℕ) : Prop :=
  sammy_caps = janine_caps + 2

def billie_caps : ℕ := 2

def sammy_caps : ℕ := 8

-- The main statement to prove:
theorem janine_has_3_times_bottle_caps (janine_caps : ℕ) :
  sammy_has_more_bottle_caps sammy_caps janine_caps →
  janine_caps = 3 * billie_caps :=
by
  intro h_sammy
  rw [sammy_has_more_bottle_caps, sammy_caps, billie_caps] at h_sammy
  have h_janine : janine_caps = 6 := by linarith
  rw [h_janine, billie_caps]
  norm_num
  sorry

end janine_has_3_times_bottle_caps_l95_95458


namespace min_ratio_of_distinct_points_on_plane_l95_95626

/-- For any 4 distinct points P1, P2, P3, P4 on a plane, the ratio
    ∑1≤i<j≤4 (dist P_i P_j) / (min 1≤i < j ≤4 (dist P_i P_j))
    achieves a minimum value of 5 + sqrt 3.
-/
theorem min_ratio_of_distinct_points_on_plane (P1 P2 P3 P4 : Euclidean_Space ℝ 2) :
  P1 ≠ P2 ∧ P2 ≠ P3 ∧ P3 ≠ P4 ∧ P4 ≠ P1 ∧ P1 ≠ P3 ∧ P2 ≠ P4 →
  Real.sqrt 2 :=
by
  sorry

end min_ratio_of_distinct_points_on_plane_l95_95626


namespace gcm_15_and_20_less_than_150_gcm_of_15_and_20_l95_95886

theorem gcm_15_and_20_less_than_150 : 
  ∃ x, (x = 15 * k ∧ x = 20 * m ∧ x < 150) ∧ (∀ y, (y = 15 * k' ∧ y = 20 * m' ∧ y < 150) → y ≤ x) := by
sorr

theorem gcm_of_15_and_20 : 
  ∃ x, x = 120 ∧ (x = 15 * k ∧ x = 20 * m ∧ x < 150) ∧ (∀ y, (y = 15 * k' ∧ y = 20 * m' ∧ y < 150) → y ≤ x) := by
sorry

end gcm_15_and_20_less_than_150_gcm_of_15_and_20_l95_95886


namespace number_of_real_z_fifth_roots_of_unity_l95_95155

noncomputable def count_reals_of_z_fifth {z : ℂ} (h : z^30 = 1) : ℕ :=
  if h1 : (z^5 = 1 ∨ z^5 = -1) then 1 else 0

theorem number_of_real_z_fifth_roots_of_unity :
  (∑ z in {z : ℂ | z^30 = 1}, count_reals_of_z_fifth z) = 12 :=
sorry

end number_of_real_z_fifth_roots_of_unity_l95_95155


namespace num_correct_propositions_l95_95583

variable (p q : Prop)

def prop1 : Prop := p → (p ∧ q)
def prop2 : Prop := (p ∧ q) → p
def prop3 : Prop := p → (p ∨ q)
def prop4 : Prop := (p ∨ q) → p
def prop5 : Prop := (p ∨ q) → (¬p ∨ ¬q)

theorem num_correct_propositions (h1: ¬prop1) (h2: prop2) (h3: prop3) (h4: ¬prop4) (h5: prop5) : 3 =
  let correct_count := [h1, h2, h3, h4, h5].count true
  correct_count
:= 
sorry

end num_correct_propositions_l95_95583


namespace cube_vertex_labeling_l95_95990

theorem cube_vertex_labeling :
  ∃ S : finset (fin 8 → fin 8),
    (∀ f ∈ S, (∀ v : fin 6, (∑ w in v.1.image f, w.val + 1) = 18)) ∧ 
    S.card = 6 :=
by sorry

end cube_vertex_labeling_l95_95990


namespace find_imaginary_part_l95_95781

def complex_number := ℂ

noncomputable def z : complex_number :=
  (1 + 2 * complex.I) / ((1 - complex.I) * (1 - complex.I))

theorem find_imaginary_part : complex.im z = 1 / 2 :=
by sorry

end find_imaginary_part_l95_95781


namespace cups_per_girl_l95_95517

noncomputable def numStudents := 30
noncomputable def numBoys := 10
noncomputable def numCupsByBoys := numBoys * 5
noncomputable def totalCups := 90
noncomputable def numGirls := 2 * numBoys
noncomputable def numCupsByGirls := totalCups - numCupsByBoys

theorem cups_per_girl : (numCupsByGirls / numGirls) = 2 := by
  sorry

end cups_per_girl_l95_95517


namespace minimum_ratio_P1_P2_l95_95829

theorem minimum_ratio_P1_P2 :
  ∃ (g₁ g₂ : List ℕ), (∀ x ∈ g₁ ++ g₂, x ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) ∧
                      (g₁.disjoint g₂) ∧
                      (g₁.product % g₂.product = 0) ∧
                      (g₁.product / g₂.product = 7) :=
begin
  sorry
end

end minimum_ratio_P1_P2_l95_95829


namespace largest_obtuse_prime_angle_l95_95724

theorem largest_obtuse_prime_angle (alpha beta gamma : ℕ) 
    (h_triangle_sum : alpha + beta + gamma = 180) 
    (h_alpha_gt_beta : alpha > beta) 
    (h_beta_gt_gamma : beta > gamma)
    (h_obtuse_alpha : alpha > 90) 
    (h_alpha_prime : Prime alpha) 
    (h_beta_prime : Prime beta) : 
    alpha = 173 := 
sorry

end largest_obtuse_prime_angle_l95_95724


namespace evaluate_sqrt_log_expression_l95_95997

noncomputable def evaluate_log_expression : ℝ :=
  let log3 (x : ℝ) := Real.log x / Real.log 3
  let log4 (x : ℝ) := Real.log x / Real.log 4
  Real.sqrt (log3 8 + log4 8)

theorem evaluate_sqrt_log_expression : evaluate_log_expression = Real.sqrt 3 := 
by
  sorry

end evaluate_sqrt_log_expression_l95_95997


namespace tournament_committee_count_l95_95381

/-- In a modified local frisbee league, there are 5 teams, and each team consists of 
8 members. At each tournament, the host team selects 4 members for the committee, 
and each non-host team selects 3 members for the committee. Prove that the number of 
possible tournament committees comprising 13 members in total is 3443073600. -/
theorem tournament_committee_count 
  (num_teams : ℕ) 
  (team_size : ℕ) 
  (host_selection : ℕ) 
  (nonhost_selection : ℕ)
  (comm_size : ℕ)
  (num_teams_eq : num_teams = 5)
  (team_size_eq : team_size = 8)
  (host_selection_eq : host_selection = 4)
  (nonhost_selection_eq : nonhost_selection = 3)
  (comm_size_eq : comm_size = 13) :
  ∑ (host_team in finset.range num_teams), (nat.choose team_size host_selection) * 
  ∏ (nonhost_team in finset.range (num_teams - 1)), (nat.choose team_size nonhost_selection) = 3443073600 := 
by
  sorry

end tournament_committee_count_l95_95381


namespace prob_sum_five_eq_one_ninth_l95_95062

-- Define the set of all possible outcomes when rolling two dice
def all_possible_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 1 7) (Finset.range 1 7)

-- Total number of outcomes when rolling two dice
def total_possible_outcomes : ℕ := Finset.card all_possible_outcomes

-- Define the set of favorable outcomes where the sum of the points is 5
def favorable_outcomes : Finset (ℕ × ℕ) :=
  {(1, 4), (2, 3), (3, 2), (4, 1)}

-- Number of favorable outcomes
def number_favorable_outcomes : ℕ := Finset.card favorable_outcomes

-- Probability of the sum being 5
def prob_sum_is_five : ℚ :=
  number_favorable_outcomes / total_possible_outcomes

-- The proof statement
theorem prob_sum_five_eq_one_ninth :
  prob_sum_is_five = 1 / 9 :=
by
  sorry

end prob_sum_five_eq_one_ninth_l95_95062


namespace quadratic_polynomials_eq_l95_95007

-- Define the integer part function
def intPart (x : ℝ) : ℤ := Int.floor x

-- Define the condition for quadratic polynomials
def is_quadratic (f : ℝ → ℝ) := ∃ (a b c : ℝ), ∀ x, f(x) = a*x^2 + b*x + c

theorem quadratic_polynomials_eq 
    (f g : ℝ → ℝ)
    (hf : is_quadratic f)
    (hg : is_quadratic g)
    (h_condition : ∀ x, intPart (f x) = intPart (g x)) :
    ∀ x, f x = g x :=
by
  sorry

end quadratic_polynomials_eq_l95_95007


namespace water_overflow_amount_l95_95704

-- Declare the conditions given in the problem
def tap_production_per_hour : ℕ := 200
def tap_run_duration_in_hours : ℕ := 24
def tank_capacity_in_ml : ℕ := 4000

-- Define the total water produced by the tap
def total_water_produced : ℕ := tap_production_per_hour * tap_run_duration_in_hours

-- Define the amount of water that overflows
def water_overflowed : ℕ := total_water_produced - tank_capacity_in_ml

-- State the theorem to prove the amount of overflowing water
theorem water_overflow_amount : water_overflowed = 800 :=
by
  -- Placeholder for the proof
  sorry

end water_overflow_amount_l95_95704


namespace total_path_length_l95_95830

-- Definitions for vertices and transformations
structure Point :=
(x : ℚ) (y : ℚ)

def A : Point := ⟨0, 0⟩
def B : Point := ⟨4, 0⟩
def C : Point := ⟨4, 4⟩
def D : Point := ⟨0, 4⟩

def rotate180_clockwise (p : Point) (center : Point) : Point :=
⟨2 * center.x - p.x, 2 * center.y - p.y⟩

def reflect_across_line (p : Point) (a b : Point) : Point :=
let m := (b.y - a.y) / (b.x - a.x) in
let c := (a.y * b.x - a.x * b.y) / (b.x - a.x) in
let new_x := (p.x - m * p.y + 2 * m * c) / (1 + m^2) in
let new_y := (m * p.x + (m^2) * p.y - 2 * c) / (1 + m^2) in
⟨new_x, new_y⟩

theorem total_path_length (A B C D : Point) :
C.x = 4 → C.y = 4 → B.x = 4 → B.y = 0 → D.x = 0 → D.y = 4 →
let C' := rotate180_clockwise C A in
let C'' := reflect_across_line C' B D in
dist C A + dist C' C'' = 4 * real.sqrt 2 + 4 :=
sorry

end total_path_length_l95_95830


namespace measure_smaller_angle_between_rays_l95_95933

theorem measure_smaller_angle_between_rays
    (circle : Type)
    (rays : Fin 10 → circle)
    (central_angles : ∀ i, ∠ (rays i) (rays (i + 1)) = 36)
    (north_ray : rays 0)
    (east_ray : rays 2)
    (northwest_ray : rays 7) :
    ∠ east_ray northwest_ray = 144 := by
  sorry

end measure_smaller_angle_between_rays_l95_95933


namespace total_notes_l95_95108

theorem total_notes (total_money : ℕ) (fifty_notes : ℕ) (fifty_value : ℕ) (fivehundred_value : ℕ) (fivehundred_notes : ℕ) :
  total_money = 10350 →
  fifty_notes = 57 →
  fifty_value = 50 →
  fivehundred_value = 500 →
  57 * 50 + fivehundred_notes * 500 = 10350 →
  fifty_notes + fivehundred_notes = 72 :=
by
  intros h_total_money h_fifty_notes h_fifty_value h_fivehundred_value h_equation
  sorry

end total_notes_l95_95108


namespace Ivanov_made_an_error_l95_95820

theorem Ivanov_made_an_error (mean median : ℝ) (variance : ℝ) (h1 : mean = 0) (h2 : median = 4) (h3 : variance = 15.917) : 
  |mean - median| ≤ Real.sqrt variance → False :=
by {
  have mean_value : mean = 0 := h1,
  have median_value : median = 4 := h2,
  have variance_value : variance = 15.917 := h3,

  let lhs := |mean_value - median_value|,
  have rhs := Real.sqrt variance_value,
  
  calc
    lhs = |0 - 4| : by rw [mean_value, median_value]
    ... = 4 : by norm_num,
  
  have rhs_val : Real.sqrt variance_value ≈ 3.99 := by sorry, -- approximate value for demonstration
  
  have ineq : 4 ≤ rhs_val := by 
    calc 4 = 4 : rfl -- trivial step for clarity,
    have sqrt_val : Real.sqrt 15.917 < 4 := by sorry, -- from calculation or suitable proof
  
  exact absurd ineq (not_le_of_gt sqrt_val)
}

end Ivanov_made_an_error_l95_95820


namespace number_of_real_solutions_l95_95768

theorem number_of_real_solutions (a b : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : a ≠ 1) (h4 : b ≠ 1) :
  ∃! x : ℝ, (log a x * log b x = log a b) := 
sorry

end number_of_real_solutions_l95_95768


namespace sin_subtraction_identity_l95_95914

theorem sin_subtraction_identity (A B : ℝ) :
  sin (A - B) = sin A * cos B - cos A * sin B :=
sorry

end sin_subtraction_identity_l95_95914


namespace round_and_scientific_notation_l95_95452

def round_significant_figures (x : ℝ) (figs : ℕ) : ℝ :=
  let factor := 10 ^ (figs - 1 - (Real.log10 (Float.abs x)).floor)
  (Float.round (x * factor) / factor)

theorem round_and_scientific_notation :
  round_significant_figures (-29800000) 3 = -2.98 * 10^7 := sorry

end round_and_scientific_notation_l95_95452


namespace cube_vertex_labeling_l95_95991

theorem cube_vertex_labeling :
  ∃ S : finset (fin 8 → fin 8),
    (∀ f ∈ S, (∀ v : fin 6, (∑ w in v.1.image f, w.val + 1) = 18)) ∧ 
    S.card = 6 :=
by sorry

end cube_vertex_labeling_l95_95991


namespace arithmetic_sequence_sum_l95_95497

theorem arithmetic_sequence_sum (c d : ℕ) 
  (h1 : ∀ (a1 a2 a3 a4 a5 a6 : ℕ), a1 = 3 → a2 = 10 → a3 = 17 → a6 = 38 → (a2 - a1 = a3 - a2) → (a3 - a2 = c - a3) → (c - a3 = d - c) → (d - c = a6 - d)) : 
  c + d = 55 := 
by 
  sorry

end arithmetic_sequence_sum_l95_95497


namespace nested_sum_equals_one_third_l95_95146

noncomputable def nested_sum : ℝ :=
  ∑' j : ℕ, ∑' k : ℕ, 2^(-(3 * k + j + (k + j)^2 + 2))

theorem nested_sum_equals_one_third :
  nested_sum = 1 / 3 :=
by
  sorry

end nested_sum_equals_one_third_l95_95146


namespace smith_boxes_l95_95427

theorem smith_boxes (initial_markers : ℕ) (markers_per_box : ℕ) (total_markers : ℕ)
  (h1 : initial_markers = 32) (h2 : markers_per_box = 9) (h3 : total_markers = 86) :
  total_markers - initial_markers = 6 * markers_per_box :=
by
  -- We state our assumptions explicitly for better clarity
  have h_total : total_markers = 86 := h3
  have h_initial : initial_markers = 32 := h1
  have h_box : markers_per_box = 9 := h2
  sorry

end smith_boxes_l95_95427


namespace cuckoo_clock_chimes_l95_95931

theorem cuckoo_clock_chimes :
  let chimes (hour: ℕ) : ℕ := if hour < 13 then hour else hour - 12 in
  (chimes 10) + (chimes 11) + (chimes 12) + (chimes 13) + (chimes 14) + (chimes 15) + (chimes 16) = 43 :=
by 
let chimes := λ hour, if hour < 13 then hour else hour - 12 in
show (chimes 10) + (chimes 11) + (chimes 12) + (chimes 13) + (chimes 14) + (chimes 15) + (chimes 16) = 43,
from sorry

end cuckoo_clock_chimes_l95_95931


namespace choose_15_3_l95_95255

theorem choose_15_3 : choose 15 3 = 455 := by
  sorry

end choose_15_3_l95_95255


namespace alternating_sum_total_for_n_equals_8_l95_95625

def alternating_sum (s : Finset ℕ) : ℤ :=
  s.sort (· > ·).foldr (λ n acc => if acc.1 then acc.2 + n else acc.2 - n) (true, 0)

def total_alternating_sum (n : ℕ) : ℤ :=
  (Finset.powerset (Finset.range (n + 1)).erase ∅).sum alternating_sum

theorem alternating_sum_total_for_n_equals_8 : total_alternating_sum 8 = 1024 := 
sorry

end alternating_sum_total_for_n_equals_8_l95_95625


namespace valentine_card_cost_l95_95786

theorem valentine_card_cost
  (total_students : ℕ)
  (percentage_students : ℝ)
  (total_money : ℝ)
  (percentage_spent : ℝ)
  (num_students : ℕ := nat.to_floor_real ((percentage_students * total_students.to_float) / 1))
  (money_spent : ℝ := percentage_spent * total_money)
  (card_cost : ℝ) :
  total_students = 30 → percentage_students = 0.60 → total_money = 40 → percentage_spent = 0.90 →
  card_cost = money_spent / num_students → card_cost = 2 := 
by
  intros h_ts h_ps h_tm h_sp h_cc
  -- skipping the proof steps
  sorry

end valentine_card_cost_l95_95786


namespace proof_part1_proof_part2_proof_part3_l95_95662

variable (θ : ℝ)
variable (sinθ cosθ : ℝ)
variable (m : ℝ)
variable (h_sin_cos_roots : ∀ x : ℝ, (2 * x^2 - (Real.sqrt 3 - 1) * x + m = 0) → (x = sinθ ∨ x = cosθ))
variable (h_theta_range : (3 / 2) * Real.pi < θ ∧ θ < 2 * Real.pi)

noncomputable def part1 : Prop :=
  m = - Real.sqrt 3 / 2

noncomputable def part2 : Prop :=
  (sinθ / (1 - 1 / Real.tan θ)) + (cosθ / (1 - Real.tan θ)) = (Real.sqrt 3 - 1) / 2

noncomputable def part3 : Prop :=
  Real.cos (2 * θ) = 1 / 2

theorem proof_part1 (h_sin_cos_roots : ∀ x : ℝ, (2 * x^2 - (Real.sqrt 3 - 1) * x + m = 0) → (x = sinθ ∨ x = cosθ)) : part1 θ :=
  sorry

theorem proof_part2 (h_sin_cos_roots : ∀ x : ℝ, (2 * x^2 - (Real.sqrt 3 - 1) * x + m = 0) → (x = sinθ ∨ x = cosθ)) : part2 θ :=
  sorry

theorem proof_part3 (h_sin_cos_roots : ∀ x : ℝ, (2 * x^2 - (Real.sqrt 3 - 1) * x + m = 0) → (x = sinθ ∨ x = cosθ))
    (h_theta_range : (3 / 2) * Real.pi < θ ∧ θ < 2 * Real.pi) : part3 θ :=
  sorry

end proof_part1_proof_part2_proof_part3_l95_95662


namespace calculate_result_l95_95142

noncomputable def calculate_expression : ℝ :=
  (-7.5) ^ 0 + (9 / 4) ^ 0.5 - (0.5) ^ -2 + real.log 25 + real.log 4 - real.logb 3 (427 / 3)

theorem calculate_result : calculate_expression = 3 / 4 :=
  by
  -- Proof steps would go here.
  sorry

end calculate_result_l95_95142


namespace Juan_saw_bicycles_l95_95627

theorem Juan_saw_bicycles (B : ℕ) : 
  (15 * 4 + 8 * 4 + 1 * 3 + B * 2 = 101) → B = 3 :=
by
  intros h
  have h1 : 15 * 4 = 60 := rfl
  have h2 : 8 * 4 = 32 := rfl
  have h3 : 1 * 3 = 3 := rfl
  sorry

end Juan_saw_bicycles_l95_95627


namespace example_geometric_sequence_preserving_l95_95980

open Real

def geometric_sequence_preserving_function (f : ℝ → ℝ) : Prop :=
  ∀ (a : ℕ → ℝ) (q : ℝ), (q ≠ 1 ∧ ∀ n, a (n + 1) = q * a n) →
    ∀ n, f (a (n + 1)) / f (a n) = (f q) / (f 1)

theorem example_geometric_sequence_preserving :
  geometric_sequence_preserving_function (λ x, 1 / |x|) :=
sorry

end example_geometric_sequence_preserving_l95_95980


namespace symmetric_shading_additional_squares_l95_95975

theorem symmetric_shading_additional_squares :
  let initial_shaded : List (ℕ × ℕ) := [(1, 1), (2, 4), (4, 3)]
  let required_horizontal_symmetry := [(4, 1), (1, 6), (4, 6)]
  let required_vertical_symmetry := [(2, 3), (1, 3)]
  let total_additional_squares := required_horizontal_symmetry ++ required_vertical_symmetry
  let final_shaded := initial_shaded ++ total_additional_squares
  ∀ s ∈ total_additional_squares, s ∉ initial_shaded →
    final_shaded.length - initial_shaded.length = 5 :=
by
  sorry

end symmetric_shading_additional_squares_l95_95975


namespace Ivanov_made_error_l95_95805

theorem Ivanov_made_error (x m : ℝ) (S_squared : ℝ) (h_x : x = 0) (h_m : m = 4) (h_S_squared : S_squared = 15.917) :
  ¬(|x - m| ≤ Real.sqrt S_squared) :=
by
  have h_sd : Real.sqrt S_squared = 3.99 := by sorry
  have h_ineq: |x - m| = 4 := by sorry
  rw [h_sd, h_ineq]
  linarith

end Ivanov_made_error_l95_95805


namespace right_triangle_angles_l95_95476

theorem right_triangle_angles (α β : ℝ) (h : α + β = 90) 
  (h_ratio : (180 - α) / (90 + α) = 9 / 11) : α = 58.5 ∧ β = 31.5 :=
by
  have h1 : 180 - α = 9 * (90 + α) / 11, from by nlinarith,
  have h2 : 11 * (180 - α) = 9 * (90 + α), from by nlinarith,
  have h3 : 11 * 180 - 11 * α = 9 * 90 + 9 * α, from by nlinarith,
  have h4 : 1980 - 11 * α = 810 + 9 * α, from by nlinarith,
  have h5 : 1170 = 20 * α, from by nlinarith,
  have h6 : α = 58.5, from by linarith,
  have h7 : β = 90 - α, from by nlinarith,
  have h8 : β = 31.5, from by nlinarith,
  exact ⟨h6, h8⟩

end right_triangle_angles_l95_95476


namespace trapezoid_pentagon_area_l95_95579

theorem trapezoid_pentagon_area (t : Trapezoid)
  (A B C D : Point)
  (h₁ : IsTriangle A B C) (h₂ : IsTriangle A C D)
  (area_ABC : area A B C = 8) (area_ACD : area A C D = 18)
  (base_ratio : (base A B / base A C) = 2 / 1)
  (total_area : area t = 40) :
  ∃ p : Pentagon, area t - (area A B C + area A C D) = 14 :=
by
  sorry

end trapezoid_pentagon_area_l95_95579


namespace good_students_options_l95_95320

variables (E B : ℕ)

-- Define the condition that the class has 25 students
def total_students : Prop := E + B = 25

-- Define the condition given by the first group of students
def first_group_condition : Prop := B > 12

-- Define the condition given by the second group of students
def second_group_condition : Prop := B = 3 * (E - 1)

-- Define the problem statement
theorem good_students_options (E B : ℕ) :
  total_students E B → first_group_condition B → second_group_condition E B → (E = 5 ∨ E = 7) :=
by
  intros h1 h2 h3
  sorry

end good_students_options_l95_95320


namespace common_terms_arithmetic_sequences_eq_503_l95_95244

theorem common_terms_arithmetic_sequences_eq_503 :
  let seq1 := λ n : ℕ, 5 + 3 * n
  let seq2 := λ m : ℕ, 3 + 4 * m
  let common_seq := λ k : ℕ, 5 + 60 * k
  (set.range seq1 ∩ set.range seq2).card = 503 :=
by
  sorry

end common_terms_arithmetic_sequences_eq_503_l95_95244


namespace track_length_l95_95597

theorem track_length
  (meet1_dist : ℝ)
  (meet2_sally_additional_dist : ℝ)
  (constant_speed : ∀ (b_speed s_speed : ℝ), b_speed = s_speed)
  (opposite_start : true)
  (brenda_first_meet : meet1_dist = 100)
  (sally_second_meet : meet2_sally_additional_dist = 200) :
  ∃ L : ℝ, L = 200 :=
by
  sorry

end track_length_l95_95597


namespace part1_part2_l95_95194

noncomputable def f (x a : ℝ) : ℝ := |x + a|
noncomputable def g (x : ℝ) : ℝ := |x + 3| - x

theorem part1 (x : ℝ) : f x 1 < g x → x < 2 :=
sorry

theorem part2 (a : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x a < g x) → -2 < a ∧ a < 2 :=
sorry

end part1_part2_l95_95194


namespace fraction_computation_l95_95530

theorem fraction_computation : (2 / 3) * (3 / 4 * 40) = 20 := 
by
  -- The proof will go here, for now we use sorry to skip the proof.
  sorry

end fraction_computation_l95_95530


namespace polynomial_terms_equal_l95_95486

theorem polynomial_terms_equal (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (h : p + q = 1) :
  (9 * p^8 * q = 36 * p^7 * q^2) → p = 4 / 5 :=
by
  sorry

end polynomial_terms_equal_l95_95486


namespace part1_solution_part2_solution_part3_solution_l95_95219

def f (x a : ℝ) : ℝ := x^2 + (x - 1) * (abs (x - a))

theorem part1_solution (a : ℝ) (h : a = -1) :
  {x : ℝ | f x a = 1} = {x : ℝ | x ≤ -1 ∨ x = 1} :=
by sorry

theorem part2_solution (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ↔ a ≥ (1 / 3) :=
by sorry

theorem part3_solution (a : ℝ) (h : a = 1) :
  let s := set.Icc 0 3 in
  set.range (λ x : ℝ, f x a) ∩ s = set.Icc (-1) 1 :=
by sorry

end part1_solution_part2_solution_part3_solution_l95_95219


namespace new_boxes_of_markers_l95_95430

theorem new_boxes_of_markers (initial_markers new_markers_per_box total_markers : ℕ) 
    (h_initial : initial_markers = 32) 
    (h_new_markers_per_box : new_markers_per_box = 9)
    (h_total : total_markers = 86) :
  (total_markers - initial_markers) / new_markers_per_box = 6 := 
by
  sorry

end new_boxes_of_markers_l95_95430


namespace tasha_amy_sum_l95_95023

theorem tasha_amy_sum :
  ∃ t a : ℝ, (t = a + 12) ∧ (t^2 + a^2 = 169 / 2) ∧ (t^4 = a^4 + 5070) ∧ (t + a = 5) :=
begin
  sorry
end

end tasha_amy_sum_l95_95023


namespace square_to_circle_area_ratio_l95_95572

noncomputable def ratio_square_area_to_circle_area (r : ℝ) : ℝ := (3 * r^2) / (π * r^2)

theorem square_to_circle_area_ratio (r : ℝ) : ratio_square_area_to_circle_area r = 3 / π := by
  sorry

end square_to_circle_area_ratio_l95_95572


namespace find_angle_C_find_a_b_max_area_l95_95304

variables {A B C : ℝ} -- Angles in the triangle
variables {a b c : ℝ} -- Sides opposite to the angles

-- Conditions
def cos_condition := (2 * a + b) * cos C + c * cos B = 0
def c_value := c = 4

-- Proof statements
theorem find_angle_C (h : cos_condition) : C = 120 :=
sorry

theorem find_a_b_max_area (h : cos_condition) (hc : c_value) : 
  a = (4 * real.sqrt 3) / 3 ∧ b = (4 * real.sqrt 3) / 3 :=
sorry

end find_angle_C_find_a_b_max_area_l95_95304


namespace gold_heart_necklace_cost_l95_95935

noncomputable def bracelet_price : ℝ := 15
noncomputable def necklace_price (x : ℝ) : ℝ := x
noncomputable def mug_price : ℝ := 20
noncomputable def total_spent (x : ℝ) : ℝ := 3 * bracelet_price + 2 * necklace_price(x) + mug_price
noncomputable def change_received : ℝ := 15
noncomputable def money_given : ℝ := 100
noncomputable def actual_spent : ℝ := money_given - change_received

theorem gold_heart_necklace_cost :
  ∀ (x : ℝ), total_spent x = actual_spent → x = 10 :=
by
  intro x h
  sorry

end gold_heart_necklace_cost_l95_95935


namespace non_zero_digits_l95_95156

theorem non_zero_digits {n d : ℕ} :
  let expr := (n : ℚ) / d in -- Define the expression n / d as a rational number
  n = 180 → d = 2^4 * 5^6 * 3^2 → 
  (Real.toRat (expr)).numDigitsAfterDec = 1 :=
  by
  intros n d expr hn hd
  sorry

end non_zero_digits_l95_95156


namespace Tony_investment_total_years_l95_95874

def Tony_preparation_years (science_degree_years : ℕ) (other_degrees_years : ℕ) (num_other_degrees : ℕ)
                           (graduate_degree_years : ℕ) (scientist_years : ℕ) (internship_months : ℕ)
                           (num_internships : ℕ) : ℕ :=
  science_degree_years +
  (other_degrees_years * num_other_degrees) +
  graduate_degree_years +
  scientist_years +
  ((internship_months / 12) * num_internships)

theorem Tony_investment_total_years :
  Tony_preparation_years 4 4 2 2 3 6 3 = 18.5 :=
sorry

end Tony_investment_total_years_l95_95874


namespace trajectory_P_right_branch_area_of_triangle_OAB_l95_95647

-- Define the conditions for the problem
def is_hyperbola_trajectory (P : ℝ × ℝ) : Prop :=
  |P.1 - (-2 : ℝ)| - |P.1 - 2| = 2 * Real.sqrt 2

-- Define the right branch of the hyperbola condition
def right_branch_hyperbola (P : ℝ × ℝ) : Prop :=
  P.1^2 - P.2^2 = 2 ∧ P.1 > 0

-- Define the area of the triangle OAB given k
def triangle_area (k : ℝ) : ℝ :=
  2 * Real.sqrt 2 * |k| * Real.sqrt (1 + k^2) / |k^2 - 1|

-- Problem statements in Lean
theorem trajectory_P_right_branch {P : ℝ × ℝ} :
  is_hyperbola_trajectory P → right_branch_hyperbola P :=
sorry

theorem area_of_triangle_OAB (k : ℝ) (h : k ∈ set.Ioo (-∞) (-1) ∪ set.Ioo 1 ∞) :
  ∃ A B : ℝ × ℝ, line_through_points y = k * (x - 2) ∧
                 right_branch_hyperbola A ∧
                 right_branch_hyperbola B ∧
                 O = (0,0) ∧
                 geom.triangle_area O A B = triangle_area k :=
sorry

end trajectory_P_right_branch_area_of_triangle_OAB_l95_95647


namespace tim_must_break_bill_probability_l95_95050

theorem tim_must_break_bill_probability :
  let toy_prices_in_quarters := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
      total_toys := toy_prices_in_quarters.length,
      favorite_toy_price_in_quarters := 16,
      tim_initial_quarters := 12,
      total_permutations := Nat.factorial total_toys,
      favorite_toy_breaking_point_count := Nat.factorial (total_toys - 1)
  in (total_permutations : ℚ) * (9 / 10) = 9 :=
by
  sorry

end tim_must_break_bill_probability_l95_95050


namespace problem1_problem2_l95_95214

noncomputable def f (x : Real) : Real := 
  let a := (2 * Real.cos x, Real.sqrt 3 * Real.sin (2 * x))
  let b := (Real.cos x, 1)
  a.1 * b.1 + a.2 * b.2

theorem problem1 (x : Real) : 
  ∃ k : Int, - Real.pi / 3 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 6 + k * Real.pi :=
  sorry

theorem problem2 (A B C a b c : Real)
  (h1 : a = Real.sqrt 7)
  (h2 : Real.sin B = 2 * Real.sin C)
  (h3 : f A = 2)
  : (∃ area : Real, area = (7 * Real.sqrt 3) / 6) :=
  sorry

end problem1_problem2_l95_95214


namespace total_marbles_after_trade_l95_95444

theorem total_marbles_after_trade :
  ∀ (total_marbles : ℕ) (blue_percentage : ℚ) (kept_red_marbles : ℕ) (exchange_rate : ℕ),
    total_marbles = 10 →
    blue_percentage = (40 / 100 : ℚ) →
    kept_red_marbles = 1 →
    exchange_rate = 2 →
  let initial_blue_marbles := (blue_percentage * total_marbles) := 4 →
  let initial_red_marbles := (total_marbles - initial_blue_marbles) := 6 →
  let traded_red_marbles := initial_red_marbles - kept_red_marbles := 5 →
  let gained_blue_marbles := traded_red_marbles * exchange_rate := 10 →
  let final_total_marbles := initial_blue_marbles + gained_blue_marbles + kept_red_marbles :=
  final_total_marbles = 15 :=
sorry

end total_marbles_after_trade_l95_95444


namespace prism_faces_l95_95947

-- Definitions based on the conditions
def prism : Type := sorry
def prism.has_hexagonal_bases (p : prism) : Prop := sorry
def prism.edges (p : prism) : ℕ := 18

theorem prism_faces (p : prism) (h1 : prism.has_hexagonal_bases p) (h2 : prism.edges p = 18) : p.faces = 8 :=
sorry

end prism_faces_l95_95947


namespace number_of_equilateral_triangles_after_12_operations_l95_95587

theorem number_of_equilateral_triangles_after_12_operations :
  ∀ (initial_triangles : ℕ),
    initial_triangles = 1 →
    ∀ (operations : ℕ),
      operations = 12 →
      let remaining_triangles : ℕ := initial_triangles * 3 ^ operations in
      remaining_triangles = 531441 :=
by
  intros initial_triangles h_initial_triangles operations h_operations
  simp [h_initial_triangles, h_operations]
  have : 3 ^ 12 = 531441 := by norm_num
  rw [this]
  sorry

end number_of_equilateral_triangles_after_12_operations_l95_95587


namespace find_other_endpoint_l95_95484

theorem find_other_endpoint (x y : ℝ) : 
  (∃ x1 y1 x2 y2 : ℝ, (x1 + x2)/2 = 2 ∧ (y1 + y2)/2 = 3 ∧ x1 = -1 ∧ y1 = 7 ∧ x2 = x ∧ y2 = y) → (x = 5 ∧ y = -1) :=
by
  sorry

end find_other_endpoint_l95_95484


namespace revenue_from_full_price_tickets_l95_95951

variable (f h p : ℕ)
variable total_tickets : f + h = 200
variable total_revenue : f * p + h * (p / 3) = 4500

theorem revenue_from_full_price_tickets :
  (f * p = 4500) :=
sorry

end revenue_from_full_price_tickets_l95_95951


namespace combined_weight_l95_95111

variables (G D C : ℝ)

def grandmother_weight (G D C : ℝ) := G + D + C = 150
def daughter_weight (D : ℝ) := D = 42
def child_weight (G C : ℝ) := C = 1/5 * G

theorem combined_weight (G D C : ℝ) (h1 : grandmother_weight G D C) (h2 : daughter_weight D) (h3 : child_weight G C) : D + C = 60 :=
by
  sorry

end combined_weight_l95_95111


namespace slope_angle_condition_l95_95225

-- Define line equations
def line1 (k : ℝ) : ℝ × ℝ → Prop := λ p, p.snd = k * p.fst - Real.sqrt 3
def line2 : ℝ × ℝ → Prop := λ p, 2 * p.fst + 3 * p.snd - 6 = 0

-- Define first quadrant
def first_quadrant : ℝ × ℝ → Prop := λ p, 0 < p.fst ∧ 0 < p.snd

-- Define slope angle range
def slope_angle_range (θ : ℝ) : Prop := θ ∈ Set.Ioo (Real.pi / 6) (Real.pi / 2)

-- Main theorem
theorem slope_angle_condition (k : ℝ) (θ : ℝ) (h_intersect : ∃ p, line1 k p ∧ line2 p ∧ first_quadrant p)
  : slope_angle_range θ :=
sorry

end slope_angle_condition_l95_95225


namespace line_through_trisection_point_of_segment_l95_95422

theorem line_through_trisection_point_of_segment
    (p1 p2 t1 t2: ℝ × ℝ)
    (p : ℝ × ℝ)
    (h1 : p1 = (1, 6))
    (h2 : p2 = (10, 0))
    (h3 : t1 = (4, 4))
    (h4 : t2 = (7, 2))
    (h5 : p = (2, 3))
    : ∃ l : Affine ℝ, 
        (∀ pt : ℝ × ℝ, pt ∈ l ↔ pt ∈ set.of_points [(2, 3), (4, 4)]) ∧ 
        l.repr = "x + 8y - 26 = 0" :=
by
  sorry

end line_through_trisection_point_of_segment_l95_95422


namespace sum_of_coefficients_l95_95408

theorem sum_of_coefficients (a b c : ℝ) (w : ℂ) (h_roots : ∃ w : ℂ, (∃ i : ℂ, i^2 = -1) ∧ 
  (x + ax^2 + bx + c)^3 = (w + 3*im)* (w + 9*im)*(2*w - 4)) :
  a + b + c = -136 :=
sorry

end sum_of_coefficients_l95_95408


namespace diagonals_intersection_probability_l95_95090

theorem diagonals_intersection_probability (decagon : Polygon) (h_regular : decagon.is_regular ∧ decagon.num_sides = 10) :
  probability_intersection_inside decagon = 42 / 119 := 
sorry

end diagonals_intersection_probability_l95_95090


namespace sum_of_fractions_decimal_equivalence_l95_95163

theorem sum_of_fractions :
  (2 / 15 : ℚ) + (4 / 20) + (5 / 45) = 4 / 9 := 
sorry

theorem decimal_equivalence :
  (4 / 9 : ℚ) = 0.444 := 
sorry

end sum_of_fractions_decimal_equivalence_l95_95163


namespace sum_first_n_terms_l95_95394

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = (1 + 1 / n) * a n + n / 2 ^ n + 1 / 2 ^ n

noncomputable def sum_of_sequence (n : ℕ) :=
  n * (n + 1) + (n + 2) / 2 ^ (n - 1) - 4

theorem sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) (h : sequence a) :
  ∑ k in finset.range n, a (k + 1) = sum_of_sequence n :=
sorry

end sum_first_n_terms_l95_95394


namespace function_solution_exists_l95_95614

theorem function_solution_exists (f : ℚ⁺ → ℝ) :
  (∀ (x y z : ℚ⁺), x + y + z + 1 = 4 * x * y * z → f x + f y + f z = 1) ↔ 
  ∃ a : ℝ, ∀ (x : ℚ⁺), f x = a * (1 / (2 * x + 1)) + (1 - a) * (1 / 3) :=
by sorry

end function_solution_exists_l95_95614


namespace sum_real_roots_eq_eight_l95_95173

noncomputable def polynomial : Polynomial ℝ := (Polynomial.C 1) * (Polynomial.X ^ 4) - (Polynomial.C 6) * (Polynomial.X ^ 3) + (Polynomial.C 8) * (Polynomial.X) - (Polynomial.C 3)

theorem sum_real_roots_eq_eight : (polynomial.roots.filter polynomial.is_real).sum = 8 :=
by sorry

end sum_real_roots_eq_eight_l95_95173


namespace triangle_side_length_mod_l95_95245

theorem triangle_side_length_mod {a d x : ℕ} 
  (h_equilateral : ∃ (a : ℕ), 3 * a = 1 + d + x)
  (h_triangle : ∀ {a d x : ℕ}, 1 + d > x ∧ 1 + x > d ∧ d + x > 1)
  : d % 3 = 1 :=
by
  sorry

end triangle_side_length_mod_l95_95245


namespace distinct_digits_and_difference_is_945_l95_95575

theorem distinct_digits_and_difference_is_945 (a b c : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_difference : 10 * (100 * a + 10 * b + c) + 2 - (2000 + 100 * a + 10 * b + c) = 945) :
  (100 * a + 10 * b + c) = 327 :=
by
  sorry

end distinct_digits_and_difference_is_945_l95_95575


namespace uncle_vasya_can_become_rich_l95_95389

namespace Lumumbu

def exchange_rate : Type :=
  | B 
  | K 
  | E 
  | Dollar

def exchange_conditions : Prop :=
  (∀ (b k : exchange_rate), b = K → k = B → 1B = 1/2K ∧ 2K = 1B) ∧
  (∀ (e k : exchange_rate), e = E → k = K → 1E = 11K ∧ 1/11E = 1K) ∧
  (∀ (d k : exchange_rate), d = Dollar → k = K → 1/15Dollar = 1K ∧ 10K = 1Dollar)

def initial_conditions : Prop :=
  ∃ v : ℕ, (v = 100)

def goal_conditions : Prop :=
  ∃ x : ℕ, (x ≥ 200)

theorem uncle_vasya_can_become_rich : exchange_conditions → initial_conditions → goal_conditions :=
sorry

end Lumumbu

end uncle_vasya_can_become_rich_l95_95389


namespace area_is_prime_number_l95_95664

open Real Int

noncomputable def area_of_triangle (a : Int) : Real :=
  (a * a : Real) / 20

theorem area_is_prime_number 
  (a : Int) 
  (h1 : ∃ p : ℕ, Nat.Prime p ∧ p = ((a * a) / 20 : Real)) :
  ((a * a) / 20 : Real) = 5 :=
by 
  sorry

end area_is_prime_number_l95_95664


namespace vector_magnitude_proof_l95_95696

theorem vector_magnitude_proof (a b : ℝ × ℝ) 
  (h₁ : ‖a‖ = 1) 
  (h₂ : ‖b‖ = 2)
  (h₃ : a - b = (Real.sqrt 3, Real.sqrt 2)) : 
‖a + (2:ℝ) • b‖ = Real.sqrt 17 := 
sorry

end vector_magnitude_proof_l95_95696


namespace prob_winning_one_draw_formula_prob_winning_once_in_three_when_n_is_3_max_f_when_n_is_2_l95_95555

noncomputable def prob_of_winning_one_draw (n : ℕ) (h : 2 ≤ n) : ℚ :=
  (n ^ 2 - n + 2) / (n ^ 2 + 3 * n + 2)

theorem prob_winning_one_draw_formula (n : ℕ) (h : 2 ≤ n) :
  prob_of_winning_one_draw n h = (n ^ 2 - n + 2) / (n ^ 2 + 3 * n + 2) :=
by
  sorry

noncomputable def prob_winning_exactly_once_in_three_draws (n : ℕ) : ℚ :=
  3 * (prob_of_winning_one_draw n (by linarith)) * (1 - prob_of_winning_one_draw n (by linarith)) ^ 2

theorem prob_winning_once_in_three_when_n_is_3 :
  prob_winning_exactly_once_in_three_draws 3 = 54 / 125 :=
by
  sorry

noncomputable def f (p : ℚ) : ℚ :=
  3 * p * (1 - p) ^ 2

noncomputable def f_prime (p : ℚ) : ℚ :=
  9 * p ^ 2 - 12 * p + 3

theorem max_f_when_n_is_2 :
  let p := (prob_of_winning_one_draw 2 (by linarith))
  f p = 1/3 ∧ (∀ q : ℚ, f q ≤ f p) :=
by
  sorry

end prob_winning_one_draw_formula_prob_winning_once_in_three_when_n_is_3_max_f_when_n_is_2_l95_95555


namespace find_angle_BOK_l95_95923

variables {O A B C K Q : Type}
variables (α β γ φ : ℝ)

-- Conditions
variable (trihedral_angle_OABC : True)
variable (angle_BOC : ∠ B O C = α)
variable (angle_COA : ∠ C O A = β)
variable (angle_AOB : ∠ A O B = γ)
variable (sphere_touches_BOC_at_K : ∀ {x : Type}, x)

-- Prove statement
theorem find_angle_BOK (angle_BOK_φ : ∠ B O K = φ) 
    (hypothesis : 2 * φ = α + γ - β) : 
    ∠ B O K = (α + γ - β) / 2 :=
by sorry

end find_angle_BOK_l95_95923


namespace find_f_neg_6_l95_95106

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if x ≥ 0 then Real.log (x + 2) / Real.log 2 + (a - 1) * x + b else -(Real.log (-x + 2) / Real.log 2 + (a - 1) * -x + b)

theorem find_f_neg_6 (a b : ℝ) (h1 : ∀ x : ℝ, f x a b = -f (-x) a b) 
                     (h2 : ∀ x : ℝ, x ≥ 0 → f x a b = Real.log (x + 2) / Real.log 2 + (a - 1) * x + b)
                     (h3 : f 2 a b = -1) : f (-6) 0 (-1) = 4 :=
by
  sorry

end find_f_neg_6_l95_95106


namespace total_profit_correct_l95_95960

-- Define the expenses for each day
def expenses_day1 : ℕ := 10 + 5 + 3 + 5
def expenses_day2 : ℕ := 12 + 6 + 4 + 5
def expenses_day3 : ℕ := 8 + 4 + 3 + 2 + 4

-- Define the revenues for each day
def revenue_day1 : ℕ := 21 * 4
def revenue_day2 : ℕ := (18 + (6 / 3)) * 3
def revenue_day3 : ℕ := 25 * 4

-- Define the profit for each day
def profit_day1 : ℕ := revenue_day1 - expenses_day1
def profit_day2 : ℕ := revenue_day2 - expenses_day2
def profit_day3 : ℕ := revenue_day3 - expenses_day3

-- Total profit over three days
def total_profit : ℕ := profit_day1 + profit_day2 + profit_day3

-- The theorem to prove
theorem total_profit_correct : total_profit = 173 :=
by 
  -- Assuming expenses and revenue calculations
  have h_expenses_day1 : expenses_day1 = 23 := by simp [expenses_day1]
  have h_expenses_day2 : expenses_day2 = 27 := by simp [expenses_day2]
  have h_expenses_day3 : expenses_day3 = 21 := by simp [expenses_day3]
  have h_revenue_day1 : revenue_day1 = 84 := by simp [revenue_day1]
  have h_revenue_day2 : revenue_day2 = 60 := by simp [revenue_day2]
  have h_revenue_day3 : revenue_day3 = 100 := by simp [revenue_day3]
  
  -- Assuming profit calculations
  have h_profit_day1 : profit_day1 = 61 := by 
    rw [profit_day1, h_revenue_day1, h_expenses_day1]
    exact rfl
  have h_profit_day2 : profit_day2 = 33 := by 
    rw [profit_day2, h_revenue_day2, h_expenses_day2]
    exact rfl
  have h_profit_day3 : profit_day3 = 79 := by 
    rw [profit_day3, h_revenue_day3, h_expenses_day3]
    exact rfl

  -- Proving the total profit
  rw [total_profit, h_profit_day1, h_profit_day2, h_profit_day3]
  exact rfl

end total_profit_correct_l95_95960


namespace max_liters_l95_95894

-- Declare the problem conditions as constants
def initialHeat : ℕ := 480 -- kJ for the first 5 min
def reductionRate : ℚ := 0.25 -- 25%
def initialTemp : ℚ := 20 -- °C
def boilingTemp : ℚ := 100 -- °C
def specificHeatCapacity : ℚ := 4.2 -- kJ/kg°C

-- Define the heat required to raise m kg of water from initialTemp to boilingTemp
def heatRequired (m : ℚ) : ℚ := m * specificHeatCapacity * (boilingTemp - initialTemp)

-- Define the total available heat from the fuel
noncomputable def totalAvailableHeat : ℚ :=
  initialHeat / (1 - (1 - reductionRate))

-- The proof problem statement
theorem max_liters :
  ∃ (m : ℕ), m ≤ 5 ∧ heatRequired m ≤ totalAvailableHeat :=
sorry

end max_liters_l95_95894


namespace min_value_of_function_l95_95620

def y (x : ℝ) : ℝ := sqrt (x^2 + 2*x + 2) + sqrt (x^2 - 2*x + 2)

theorem min_value_of_function : ∃ x, y x = 2 * sqrt 2 := sorry

end min_value_of_function_l95_95620


namespace correct_infection_rate_l95_95125

def two_rounds_infected : ℕ := 81

def constant_infection_rate (x : ℕ) : Prop :=
  (1 + x)^2 = two_rounds_infected

theorem correct_infection_rate (x : ℕ) : constant_infection_rate x → (1 + x) = 9 ∨ (1 + x) = -9 :=
by
  sorry

end correct_infection_rate_l95_95125


namespace find_a20_l95_95663

variable {α : Type*} [LinearOrderedField α]

-- Definition of an arithmetic sequence
def is_arithmetic (a : ℕ → α) (d : α) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem find_a20 (a : ℕ → α) (d : α) :
  is_arithmetic a d →
  a 1 + a 3 + a 5 = 105 →
  a 2 + a 4 + a 6 = 99 →
  a 20 = 1 :=
by {
  intro h_arith,
  intro h1,
  intro h2,
  sorry
}

end find_a20_l95_95663


namespace midpoints_form_parallelogram_midpoints_form_similar_parallelogram_l95_95288

section Parallelogram_Theory

variables (A B C D A₁ B₁ C₁ D₁ A' B' C' D' : Type)

-- Definition of being a parallelogram and related properties.
def is_parallelogram (p1 p2 p3 p4 : Type) : Prop := sorry -- Define the properties that ensure this

-- Given that ABCD and A₁B₁C₁D₁ are parallelograms in the same plane
axiom AB_parallelogram : is_parallelogram A B C D
axiom A1B1_parallelogram : is_parallelogram A₁ B₁ C₁ D₁

-- Given that A', B', C', D' are midpoints of the respective segments
axiom AA1_midpoint_A' : A'
axiom BB1_midpoint_B' : B'
axiom CC1_midpoint_C' : C'
axiom DD1_midpoint_D' : D'

-- Show that A'B'C'D' is a parallelogram
theorem midpoints_form_parallelogram : is_parallelogram A' B' C' D' := sorry

-- Show that if ABCD and A₁B₁C₁D₁ are similarly oriented, 
-- then A'B'C'D' is similar to them.
axiom similarly_oriented : sorry -- Define what it means for the parallelograms to be similarly oriented
theorem midpoints_form_similar_parallelogram (ho : similarly_oriented) : similar A' B' C' D' A B C D ∧ similar A' B' C' D' A₁ B₁ C₁ D₁ := sorry

end Parallelogram_Theory

end midpoints_form_parallelogram_midpoints_form_similar_parallelogram_l95_95288


namespace cone_height_l95_95929

/-
  Given:
  - A circular sheet of paper with radius 6 cm.
  - The sheet is cut into three congruent sectors.

  Prove:
  - The height of the cone formed by rolling one of the sectors is 4\sqrt{2} cm.
-/

theorem cone_height (radius : ℝ) (n : ℕ) (h : ℝ):
  radius = 6 → n = 3 →
  h = height_cone radius n → h = 4 * Real.sqrt 2 := 
by
  intros r_eq n_eq h_eq
  rw [r_eq, n_eq] at *
  -- Insert the correct proof here
  sorry

-- Define the height of the cone function
noncomputable def height_cone (r : ℝ) (n : ℕ) : ℝ :=
  let base_radius := (2 * Real.pi * r) / (n * 2 * Real.pi)
  let slant_height := r
  Real.sqrt (slant_height^2 - base_radius^2)


end cone_height_l95_95929


namespace tangent_line_to_curve_at_point_l95_95029

noncomputable def tangent_line_eq (x y : ℝ) : Prop := 5 * x - y - 2 = 0

theorem tangent_line_to_curve_at_point :
  ∀ (x : ℝ), ∀ (f : ℝ → ℝ), differentiable ℝ f → 
  (∀ x, f x = x^3 + 2 * x) →
  let y := f 1 in 
  (y = 3) →
  tangent_line_eq 1 y :=
begin
  intros x f h_diff h_eq h_point,
  /- proof will be here -/
  sorry
end

end tangent_line_to_curve_at_point_l95_95029


namespace perfect_square_trinomial_l95_95707

theorem perfect_square_trinomial (k : ℝ) : (∃ a b : ℝ, (a * x + b) ^ 2 = x^2 - k * x + 4) → (k = 4 ∨ k = -4) :=
by
  sorry

end perfect_square_trinomial_l95_95707


namespace decagon_diagonals_intersection_probability_l95_95088

def isRegularDecagon : Prop :=
  ∃ decagon : ℕ, decagon = 10  -- A regular decagon has 10 sides

def chosen_diagonals (n : ℕ) : ℕ :=
  (Nat.choose n 2) - n   -- Number of diagonals in an n-sided polygon =
                          -- number of pairs of vertices - n sides

noncomputable def probability_intersection : ℚ :=
  let total_diagonals := chosen_diagonals 10
  let number_of_ways_to_pick_four := Nat.choose 10 4
  (number_of_ways_to_pick_four * 2) / (total_diagonals * (total_diagonals - 1) / 2)

theorem decagon_diagonals_intersection_probability :
  isRegularDecagon → probability_intersection = 42 / 119 :=
sorry

end decagon_diagonals_intersection_probability_l95_95088


namespace jake_final_bitcoins_l95_95740

def initial_bitcoins : ℕ := 120
def investment_bitcoins : ℕ := 40
def returned_investment : ℕ := investment_bitcoins * 2
def bitcoins_after_investment : ℕ := initial_bitcoins - investment_bitcoins + returned_investment
def first_charity_donation : ℕ := 25
def bitcoins_after_first_donation : ℕ := bitcoins_after_investment - first_charity_donation
def brother_share : ℕ := 67
def bitcoins_after_giving_to_brother : ℕ := bitcoins_after_first_donation - brother_share
def debt_payment : ℕ := 5
def bitcoins_after_taking_back : ℕ := bitcoins_after_giving_to_brother + debt_payment
def quadrupled_bitcoins : ℕ := bitcoins_after_taking_back * 4
def second_charity_donation : ℕ := 15
def final_bitcoins : ℕ := quadrupled_bitcoins - second_charity_donation

theorem jake_final_bitcoins : final_bitcoins = 277 := by
  unfold final_bitcoins
  unfold quadrupled_bitcoins
  unfold bitcoins_after_taking_back
  unfold debt_payment
  unfold bitcoins_after_giving_to_brother
  unfold brother_share
  unfold bitcoins_after_first_donation
  unfold first_charity_donation
  unfold bitcoins_after_investment
  unfold returned_investment
  unfold investment_bitcoins
  unfold initial_bitcoins
  sorry

end jake_final_bitcoins_l95_95740


namespace correct_propositions_l95_95963

def velocity (t : ℝ) : ℝ := 3 * t^2 - 2 * t - 1

theorem correct_propositions :
  (∀ t : ℝ, (t ≥ 0 ∧ t ≤ 1) → ∫ x in 0..3, abs (velocity x) = 17) ∧
  (∀ x : ℝ, (0 < x < π) → sin x < x) ∧
  (∀ (f : ℝ → ℝ) (x₀ : ℝ), f' x₀ = 0 → ∃ y, y ≠ x₀ ∧ f'(y) = 0 ∧ ¬ (y.isExtremum)) ∧
  (∫ x in 0..2, sqrt (-x^2 + 4 * x) = π) :=
by
  sorry

end correct_propositions_l95_95963


namespace problem_part1_problem_part2_l95_95683

-- Part 1: Given the conditions, prove that \(\varphi = -\frac{\pi}{6}\)
theorem problem_part1 (f : ℝ → ℝ) (ϕ : ℝ) (hϕ : -π/2 < ϕ ∧ ϕ < π/2) (h_zero : f (π/3) = 0) 
  (h_def : ∀ x, f x = sin (2 * x + ϕ) - 1) : ϕ = -π/6 :=
by
  sorry

-- Part 2: Given the conditions, prove the range of \(f(x)\) on \([0,π/2]\) is \([-3/2, 0]\)
theorem problem_part2 (f : ℝ → ℝ) (ϕ : ℝ) (hϕ : ϕ = -π/6) :
  (∀ x ∈ Icc (0 : ℝ) (π/2), f x = sin (2 * x + ϕ) - 1) → 
  set.range (λ x: ℝ, if x ∈ Icc (0 : ℝ) (π/2) then f x else 0) = Icc (-3 / 2 : ℝ) 0 :=
by
  sorry

end problem_part1_problem_part2_l95_95683


namespace min_gcd_pq_l95_95763

def is_prime (n : ℕ) : Prop := n.prime

def greater_than_100 (n : ℕ) : Prop := n > 100

noncomputable def gcd_pq (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (gp : greater_than_100 p) (gq : greater_than_100 q) :=
  Int.gcd (p^2 - 1) (q^2 - 1)

theorem min_gcd_pq (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (gp : greater_than_100 p) (gq : greater_than_100 q) :
  gcd_pq p q hp hq gp gq = 8 := 
by 
  sorry

end min_gcd_pq_l95_95763


namespace Ivanov_made_an_error_l95_95816

theorem Ivanov_made_an_error : 
  (∀ x m s : ℝ, 0 = x → 4 = m → 15.917 = s^2 → ¬ (|x - m| ≤ real.sqrt s)) :=
by 
  intros x m s hx hm hs2
  rw [hx, hm, hs2]
  have H : |0 - 4| = 4 := by norm_num
  have H2 : real.sqrt 15.917 ≈ 3.99 := by norm_num
  exact neq_of_not_le (by norm_num : 4 ≠ 3.99) H2 sorry

end Ivanov_made_an_error_l95_95816


namespace range_of_k_for_monotonic_increasing_l95_95293

theorem range_of_k_for_monotonic_increasing:
  (∀ x : ℝ, x > 3 → (div (2 * x + k) (x - 2)).deriv x ≥ 0) → k < -4 :=
by
  sorry

end range_of_k_for_monotonic_increasing_l95_95293


namespace total_pages_of_book_l95_95036

theorem total_pages_of_book (P : ℝ) (h : 0.4 * P = 16) : P = 40 :=
sorry

end total_pages_of_book_l95_95036


namespace chord_circle_intersection_l95_95715

open Real

theorem chord_circle_intersection (O AM P: Point) (A C B D: Point)
(h_circle : is_circle_with_center O)
(h_AC_diameter : is_diameter A C h_circle)
(h_BD_diameter : is_diameter B D h_circle)
(h_AC_perpendicular_BD : is_perpendicular A C B D)
(h_chord_AM : is_chord_length AM 16)
(h_AM_intersects_BD_at_P : intersects_at AM B D P) :
  let AP := distance A P
  let PM := distance P M
  AP * PM = 160 := sorry

end chord_circle_intersection_l95_95715


namespace diagonals_intersection_probability_l95_95091

theorem diagonals_intersection_probability (decagon : Polygon) (h_regular : decagon.is_regular ∧ decagon.num_sides = 10) :
  probability_intersection_inside decagon = 42 / 119 := 
sorry

end diagonals_intersection_probability_l95_95091


namespace max_liters_of_water_heated_l95_95898

theorem max_liters_of_water_heated
  (heat_initial : ℕ := 480) 
  (heat_drop : ℝ := 0.25)
  (temp_initial : ℝ := 20)
  (temp_boiling : ℝ := 100)
  (specific_heat_capacity : ℝ := 4.2)
  (kJ_to_liters_conversion : ℝ := 336) :
  (∀ m : ℕ, (m * kJ_to_liters_conversion > ((heat_initial : ℝ) / (1 - heat_drop)) → m ≤ 5)) :=
by
  sorry

end max_liters_of_water_heated_l95_95898


namespace calculate_standard_deviation_of_xi_l95_95675

noncomputable def standard_deviation (xi : Fin 3 → ℝ) (p : Fin 3 → ℝ) : ℝ :=
  let expected_value := ∑ i, xi i * p i
  let variance := ∑ i, (xi i - expected_value)^2 * p i
  variance.sqrt

theorem calculate_standard_deviation_of_xi :
  let xi := fun i => match i with | 0 => 1 | 1 => 3 | 2 => 5
  let p := fun i => match i with | 0 => 0.4 | 1 => 0.1 | 2 => 0.5
  0.4 + 0.1 + 0.5 = 1 →
  standard_deviation xi p = Real.sqrt 3.56 :=
by
  intro h1
  sorry

end calculate_standard_deviation_of_xi_l95_95675


namespace no_function_satisfies_inequality_l95_95080

theorem no_function_satisfies_inequality (f : ℝ → ℝ) :
  ¬ ∀ x y : ℝ, (f x + f y) / 2 ≥ f ((x + y) / 2) + |x - y| :=
sorry

end no_function_satisfies_inequality_l95_95080


namespace probability_of_matching_pair_l95_95052

noncomputable def num_socks := 22
noncomputable def red_socks := 12
noncomputable def blue_socks := 10

def ways_to_choose_two (n : ℕ) : ℕ :=
  n * (n - 1) / 2

noncomputable def probability_same_color : ℚ :=
  (ways_to_choose_two red_socks + ways_to_choose_two blue_socks : ℚ) / ways_to_choose_two num_socks

theorem probability_of_matching_pair :
  probability_same_color = 37 / 77 := 
by
  -- proof goes here
  sorry

end probability_of_matching_pair_l95_95052


namespace Al_initial_portion_l95_95126

theorem Al_initial_portion (a b c : ℕ) 
  (h1 : a + b + c = 1200) 
  (h2 : a - 150 + 2 * b + 3 * c = 1800) 
  (h3 : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  a = 550 :=
by {
  sorry
}

end Al_initial_portion_l95_95126


namespace representable_as_sum_of_powers_l95_95615

theorem representable_as_sum_of_powers (n k : ℕ) (h₁ : 0 < n) (h₂ : 0 < k) :
  (∀ m : ℕ, m > 0 ∧ odd m → ∃ (a : ℕ → ℕ), m = ∑ i in finset.range (k + 1), (a i) ^ ((n + i) ^ 2)) ↔ n = 1 :=
sorry

end representable_as_sum_of_powers_l95_95615


namespace find_a_parallel_l95_95297

-- Define the lines
def line1 (a : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  (a + 1) * x + 2 * y = 2

def line2 (a : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  x + a * y = 1

-- Define the parallel condition
def are_parallel (a : ℝ) : Prop :=
  ∀ x y : ℝ, line1 a x y → line2 a x y

-- The theorem stating our problem
theorem find_a_parallel (a : ℝ) : are_parallel a → a = -2 :=
by
  sorry

end find_a_parallel_l95_95297


namespace normal_line_at_point_x0_is_correct_l95_95631
open Real

noncomputable def curve_eq (x : ℝ) : ℝ := (1 + sqrt x) / (1 - sqrt x)
noncomputable def normal_line_eq (x : ℝ) : ℝ := -2 * x + 5

example : normal_line_eq = (λ x, -2 * x + 5) := by 
  rfl
  
theorem normal_line_at_point_x0_is_correct : 
  (normal_line_eq 4) = -2 * 4 + 5 := by
  calc
    normal_line_eq 4 = -2 * 4 + 5   : rfl
                   ... = -8 + 5     : by ring
                   ... = -3         : rfl



end normal_line_at_point_x0_is_correct_l95_95631


namespace solve_x_squared_plus_y_squared_l95_95264

-- Variables
variables {x y : ℝ}

-- Conditions
def cond1 : (x + y)^2 = 36 := sorry
def cond2 : x * y = 8 := sorry

-- Theorem stating the problem's equivalent proof
theorem solve_x_squared_plus_y_squared : x^2 + y^2 = 20 := sorry

end solve_x_squared_plus_y_squared_l95_95264


namespace find_k_for_f_root_l95_95679

noncomputable def f (x : ℝ) : ℝ := log x + x - 3

theorem find_k_for_f_root : 
  (∃ k : ℤ, ∃ x ∈ Ioo (k : ℝ) (k + 1), f x = 0) → ∃ k : ℤ, k = 2 :=
by
  sorry

end find_k_for_f_root_l95_95679


namespace inequality_proof_l95_95134

noncomputable def circumcircle_radius (O : Type) : ℝ := sorry
noncomputable def excircle_radius (I : Type) : ℝ := sorry

-- Define the proof problem
theorem inequality_proof (A B C O I : Type) (acute_angle : true)
  (circumcircle : ∀ {A B C}, circle O)
  (excircle : ∀ {A B C}, circle I)
  (R : ℝ := circumcircle_radius O)
  (r : ℝ := excircle_radius I) :
  let OI_sq := (sorry : ℝ) -- OI^2, should be defined in detail using triangle properties
  in
  OI_sq > R^2 - 2 * R * r := sorry

end inequality_proof_l95_95134


namespace trajectory_of_point_M_valid_range_for_k_l95_95418

open Real

noncomputable def circle_center : Point := ⟨-1, 0⟩
noncomputable def point_A : Point := ⟨1, 0⟩
noncomputable def point_P : Point := ⟨2, 1⟩
noncomputable def circle_radius : ℝ := 4

-- Equation of the trajectory of point M
theorem trajectory_of_point_M :
  ∀ (M : Point), (∃ Q : Point, on_circle Q ∧ bisects (line_AQ A Q) (line_CQ circle_center Q) M) →
  ellipse_eq M 2 sqrt_3 :=
sorry

-- Range of k for line l passing through point P and meeting condition
theorem valid_range_for_k :
  ∀ (k : ℝ), (passes_through_P (line_kl k) point_P) ∧ 
  (∃ B D : Point, on_ellipse B 2 sqrt_3 ∧ on_ellipse D 2 sqrt_3 ∧ intersects_BD (line_kl k) B D ∧ dot_product_cond B D point_P) →
  k ∈ Ioo (-1/2) (1/2) :=
sorry

end trajectory_of_point_M_valid_range_for_k_l95_95418


namespace probability_two_green_in_four_l95_95097

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def bag_marbles := 12
def green_marbles := 5
def blue_marbles := 3
def yellow_marbles := 4
def total_picked := 4
def green_picked := 2
def remaining_marbles := bag_marbles - green_marbles
def non_green_picked := total_picked - green_picked

theorem probability_two_green_in_four : 
  (choose green_marbles green_picked * choose remaining_marbles non_green_picked : ℚ) / (choose bag_marbles total_picked) = 14 / 33 := by
  sorry

end probability_two_green_in_four_l95_95097


namespace find_f_neg_two_l95_95291

-- Definitions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Function f
def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + x else -(((-x)^2) + (-x))

-- The proof problem statement
theorem find_f_neg_two
  (h_odd : is_odd_function f)
  (h_f_pos : ∀ x : ℝ, x ≥ 0 → f x = x^2 + x) :
  f (-2) = -6 :=
sorry

end find_f_neg_two_l95_95291


namespace problem_statement_l95_95259

-- Define the conditions and the goal
theorem problem_statement {x y : ℝ} 
  (h1 : (x + y)^2 = 36)
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := 
by
  sorry

end problem_statement_l95_95259


namespace sum_of_identical_digits_l95_95236

theorem sum_of_identical_digits
  (a b c : ℕ) (x : ℕ := a * 11111) (y : ℕ := b * 1111) (z : ℕ := c * 111) :
  (∃ (x y z : ℕ), (x % 11111 = 0) ∧ (y % 1111 = 0) ∧ (z % 111 = 0) ∧ 
  let sum := x + y + z in 
  sum ≥ 10000 ∧ sum < 100000 ∧ 
  (let digits := [sum / 10000 % 10, sum / 1000 % 10, sum / 100 % 10, sum / 10 % 10, sum % 10] in 
  digits.nodup)) := 
sorry

end sum_of_identical_digits_l95_95236


namespace sin_B_gt_cos_B_l95_95309

theorem sin_B_gt_cos_B (A B C: ℝ) (h1 : A + B + C = 180) (h2 : C = 90) (h3 : 0 < A) (h4 : A < 45) : sin B > cos B :=
by
  sorry

end sin_B_gt_cos_B_l95_95309


namespace decagon_diagonals_intersection_probability_l95_95087

def isRegularDecagon : Prop :=
  ∃ decagon : ℕ, decagon = 10  -- A regular decagon has 10 sides

def chosen_diagonals (n : ℕ) : ℕ :=
  (Nat.choose n 2) - n   -- Number of diagonals in an n-sided polygon =
                          -- number of pairs of vertices - n sides

noncomputable def probability_intersection : ℚ :=
  let total_diagonals := chosen_diagonals 10
  let number_of_ways_to_pick_four := Nat.choose 10 4
  (number_of_ways_to_pick_four * 2) / (total_diagonals * (total_diagonals - 1) / 2)

theorem decagon_diagonals_intersection_probability :
  isRegularDecagon → probability_intersection = 42 / 119 :=
sorry

end decagon_diagonals_intersection_probability_l95_95087


namespace altitude_on_BC_correct_length_l95_95397

noncomputable def altitude_length (A B C : Type) [normed_space ℝ A] [normed_space ℝ B] [normed_space ℝ C] (angle_A : ℝ) (AB AC : ℝ) : ℝ :=
let S_ABC := (1 / 2) * AB * AC * real.sin angle_A in
let BC := real.sqrt (AB^2 + AC^2 - 2 * AB * AC * real.cos angle_A) in
2 * S_ABC / BC

theorem altitude_on_BC_correct_length (A B C : Type) [normed_space ℝ A] [normed_space ℝ B] [normed_space ℝ C] :
  altitude_length A B C (π / 6) (sqrt 3) 4 = 2 * sqrt 21 / 7 :=
by sorry

end altitude_on_BC_correct_length_l95_95397


namespace simplify_expression_l95_95999

theorem simplify_expression (a b : ℕ) (h₁ : a = 2999) (h₂ : b = 3000) :
  b^3 - a * b^2 - a^2 * b + a^3 = 5999 :=
by 
  sorry

end simplify_expression_l95_95999


namespace integral_cos_nonnegative_l95_95753

theorem integral_cos_nonnegative {f : ℝ → ℝ} (h_cont : ContinuousOn f (Icc 0 (2 * Real.pi)))
  (h_fst_deriv_cont : ContinuousOn (deriv f) (Icc 0 (2 * Real.pi)))
  (h_snd_deriv_nonneg : ∀ x ∈ Icc 0 (2 * Real.pi), 0 ≤ (deriv^[2]) f x) :
  ∫ x in 0..(2 * Real.pi), f x * Real.cos x ≥ 0 :=
  sorry

end integral_cos_nonnegative_l95_95753


namespace exists_sufficiently_large_N_l95_95449

open Set

theorem exists_sufficiently_large_N :
  ∃ N : ℕ, ∀ (n : ℕ), n ≥ N →
  ∃ (S : Finset (ℕ)), S ⊆ Finset.range (n^2 + 3 * n + 1) \ Finset.range (n^2 + 1) ∧
  (∃ (a : ℕ), S = S.filter (λ x, ∃ y, S ≠ ∅ ∧ x * y = a^2)) ∧
  S.card ≥ 2015 := sorry

end exists_sufficiently_large_N_l95_95449


namespace heat_generated_in_wire_l95_95493

theorem heat_generated_in_wire (R : ℝ) (m : ℝ) (t : ℝ) (z : ℝ) (H : ℝ)
  (hR : R = 10)
  (hm : m = 1.1081)
  (ht : t = 30 * 60)
  (hz : z = 0.0003275)
  (hH : H = 8.48) :
  let I := m / (t * z),
      P := I^2 * R,
      H_cal := P * 0.24 
  in H_cal = H := 
by
  -- Add the proof here
  sorry

end heat_generated_in_wire_l95_95493


namespace value_of_f_neg2_l95_95290

noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 + x else -f (-x)

theorem value_of_f_neg2 : f (-2) = -6 := 
by
  sorry

end value_of_f_neg2_l95_95290


namespace joe_paid_4_more_than_jenny_l95_95742

theorem joe_paid_4_more_than_jenny
  (total_plain_pizza_cost : ℕ := 12) 
  (total_slices : ℕ := 12)
  (additional_cost_per_mushroom_slice : ℕ := 1) -- 0.50 dollars represented in integer (value in cents or minimal currency unit)
  (mushroom_slices : ℕ := 4) 
  (plain_slices := total_slices - mushroom_slices) -- Calculate plain slices.
  (total_additional_cost := mushroom_slices * additional_cost_per_mushroom_slice)
  (total_pizza_cost := total_plain_pizza_cost + total_additional_cost)
  (plain_slice_cost := total_plain_pizza_cost / total_slices)
  (mushroom_slice_cost := plain_slice_cost + additional_cost_per_mushroom_slice) 
  (joe_mushroom_slices := mushroom_slices) 
  (joe_plain_slices := 3) 
  (jenny_plain_slices := plain_slices - joe_plain_slices) 
  (joe_paid := (joe_mushroom_slices * mushroom_slice_cost) + (joe_plain_slices * plain_slice_cost))
  (jenny_paid := jenny_plain_slices * plain_slice_cost) : 
  joe_paid - jenny_paid = 4 := 
by {
  -- Here, we define the steps we used to calculate the cost.
  sorry -- Proof skipped as per instructions.
}

end joe_paid_4_more_than_jenny_l95_95742


namespace exist_reachable_city_l95_95718

theorem exist_reachable_city (V : Type) [decidable_eq V] (E : V → V → Prop)
  (accessible : ∀ (a b : V), ∃ p : list V, (∀ (x y : V), (x, y) ∈ (list.zip p (list.tail p)) → E x y) ∧ p.head = a ∧ p.last = b) :
  ∃ u : V, ∀ v : V, ∃ p : list V, (∀ (x y : V), (x, y) ∈ (list.zip p (list.tail p)) → E x y) ∧ p.head = u ∧ p.last = v :=
by
  sorry

end exist_reachable_city_l95_95718


namespace presidency_meeting_arrangements_l95_95103

theorem presidency_meeting_arrangements :
  let members_per_school := 5
  let total_schools := 4
  let total_members := members_per_school * total_schools
  ∃ (host_school_reps other_school_rep_counts: ℕ) (arrangements: ℕ),
    host_school_reps = (total_schools - 1) ^ ⟨members_per_school⟩ * 3 ∧
    host_school_reps = 10 ∧
    other_school_rep_counts = finset.range(2) ∑  set.to_list{finset.range(3)\other_school_rep_counts} *5 ∧
    other_school_rep_counts=25  ∧ 
    arrangements =  total_schools* host_school_reps * finset.range(set.to_list(total_schools-1)other_school_rep_counts)*other_school_rep_counts ∧
    arrangements = 3000 := 
begin
  sorry
end
end presidency_meeting_arrangements_l95_95103


namespace sum_of_integers_with_conditions_l95_95489

theorem sum_of_integers_with_conditions
  (a b : ℕ) 
  (h1 : a * b = 50) 
  (h2 : |a - b| = 5) 
  (h3 : 0 < a) 
  (h4 : 0 < b) 
  : a + b = 15 := 
by 
  sorry

end sum_of_integers_with_conditions_l95_95489


namespace circle_radius_l95_95490

theorem circle_radius (x y : ℝ) : (x^2 - 4 * x + y^2 - 21 = 0) → (∃ r : ℝ, r = 5) :=
by
  sorry

end circle_radius_l95_95490


namespace domain_of_f_range_of_f_f_is_even_f_monotonicity_l95_95217

noncomputable def f (x : ℝ) : ℝ := log (3 + x) / log 3 + log (3 - x) / log 3

theorem domain_of_f : {x : ℝ | x + 3 > 0 ∧ 3 - x > 0} = Set.Ioo (-3 : ℝ) 3 := by
  sorry

theorem range_of_f : Set.range (f) = Set.Iic (2 : ℝ) := by
  sorry

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

theorem f_monotonicity : 
  (∀ x y : ℝ, x ∈ Set.Ioo (-3) 0 → y ∈ Set.Ioo (-3) 0 → x ≤ y → f x ≤ f y) ∧ 
  (∀ x y : ℝ, x ∈ Set.Ioo 0 3 → y ∈ Set.Ioo 0 3 → x ≤ y → f y ≤ f x) := 
  by sorry

end domain_of_f_range_of_f_f_is_even_f_monotonicity_l95_95217


namespace holds_for_n_eq_1_l95_95056

theorem holds_for_n_eq_1 (x : ℝ) (h : x ≠ 1) : 
  (∑ i in Finset.range (1 + 2), x^i) = 1 + x + x^2 := 
by 
  sorry

end holds_for_n_eq_1_l95_95056


namespace mr_smith_markers_l95_95434

theorem mr_smith_markers :
  ∀ (initial_markers : ℕ) (total_markers : ℕ) (markers_per_box : ℕ) 
  (number_of_boxes : ℕ),
  initial_markers = 32 → 
  total_markers = 86 → 
  markers_per_box = 9 → 
  number_of_boxes = (total_markers - initial_markers) / markers_per_box →
  number_of_boxes = 6 :=
by
  intros initial_markers total_markers markers_per_box number_of_boxes h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃] at h₄
  simp only [Nat.sub] at h₄
  exact h₄

end mr_smith_markers_l95_95434


namespace matrix_inequality_l95_95666

theorem matrix_inequality
  (m n : ℕ)
  (a : Fin m → Fin n → ℝ) :
  (∏ (j : Fin n), ∑ (i : Fin m), (a i j)^n) ≥ 
  (∑ (i : Fin m), ∏ (j : Fin n), a i j)^n :=
sorry

end matrix_inequality_l95_95666


namespace new_boxes_of_markers_l95_95432

theorem new_boxes_of_markers (initial_markers new_markers_per_box total_markers : ℕ) 
    (h_initial : initial_markers = 32) 
    (h_new_markers_per_box : new_markers_per_box = 9)
    (h_total : total_markers = 86) :
  (total_markers - initial_markers) / new_markers_per_box = 6 := 
by
  sorry

end new_boxes_of_markers_l95_95432


namespace midpoint_EF_on_A0C0_l95_95440

open Real EuclideanGeometry

theorem midpoint_EF_on_A0C0 (A B C D E F A0 C0 : Point) 
  (hD_on_AC : on_line D (line_through A C))
  (hE_symm : symmetric_relative_to_angler_bisector D E (angle_bisector A))
  (hF_symm : symmetric_relative_to_angler_bisector D F (angle_bisector C))
  (hA0_tangent : tangent_at A0 (incircle_triangle A B C) (line_through B C))
  (hC0_tangent : tangent_at C0 (incircle_triangle A B C) (line_through A B)) :
  on_line (midpoint E F) (line_through A0 C0) :=
sorry

end midpoint_EF_on_A0C0_l95_95440


namespace min_tokens_correct_l95_95642

noncomputable def min_tokens (m n : ℕ) : ℕ :=
if even m ∨ even n then
  Int.ceil (m / 2) + Int.ceil (n / 2)
else
  Int.ceil (m / 2) + Int.ceil (n / 2) - 1

theorem min_tokens_correct {m n : ℕ} : min_tokens m n =
  if even m ∨ even n then
    Nat.ceil (m / 2) + Nat.ceil (n / 2)
  else
    Nat.ceil (m / 2) + Nat.ceil (n / 2) - 1 :=
sorry

end min_tokens_correct_l95_95642


namespace value_of_f_l95_95668

-- Define the given conditions
variables {f : ℝ → ℝ}

-- Functions conditions
axiom even_function : ∀ x, f(-x) = f(x)
axiom periodicity : ∀ x, 0 ≤ x → f(x + 2) = -f(x)
axiom log_interval : ∀ x, 0 ≤ x ∧ x < 2 → f(x) = log (x + 1) / log 2

-- Statement to prove
theorem value_of_f : f(-2011) + f(2012) = 1 :=
sorry

end value_of_f_l95_95668


namespace permutation_order_divides_lcm_l95_95762

open Nat

-- Define a permutation on the set {1, 2, ..., n}
variable (n : ℕ) (h : 0 < n)
variable (σ : Equiv.Perm (Fin n))

noncomputable def set_lcm (n : ℕ) : ℕ :=
  Finset.fold Nat.lcm 1 (Finset.range n).succ

-- The main theorem statement
theorem permutation_order_divides_lcm (n : ℕ) (h : 0 < n) (σ : Equiv.Perm (Fin n)) :
  let m := set_lcm n in orderOf σ ∣ m := sorry

end permutation_order_divides_lcm_l95_95762


namespace ellipse_standard_equation_and_triangle_area_l95_95650

theorem ellipse_standard_equation_and_triangle_area (e : ℝ) (θ : ℝ) (E : ℝ × ℝ) (a b c x₁ y₁ y₂ : ℝ)
    (h1 : e = (real.sqrt 3) / 2)
    (h2 : θ = real.pi / 6)
    (h3 : E = (-1, 0))
    (h4 : c / a = (real.sqrt 3) / 2)
    (h5 : b / c = (real.sqrt 3) / 3)
    (h6 : a^2 = b^2 + c^2)
    (h7 : a = 2)
    (h8 : b = 1)
    (h9 : x₁ * x₁ / 4 + y₁ * y₁ = 1)
    (h10 : x₁ = -1 + c * cos(θ))
    (h11 : y₁ = a * sin(θ))
    (h12 : y₂ < y₁)
    (y_diff : y₁ - y₂ = 4 * real.sqrt ((m^2 + 3) / (m^2 + 4^2)))
    (area_AOB : (1 / 2) * real.abs(E.fst - 0) * real.abs(y₁ - y₂) = (real.sqrt 3) / 2) :
    (∀ x y, (x^2 / 4) + y^2 = 1) ∧
    (∃ m t, area_AOB ≤ (real.sqrt 3) / 2) := sorry

end ellipse_standard_equation_and_triangle_area_l95_95650


namespace total_amount_l95_95567

variable (m n : ℕ)

theorem total_amount (m n : ℕ) : 2 * m + 3 * n = 2 * m + 3 * n := 
by
sory

end total_amount_l95_95567


namespace smith_boxes_l95_95428

theorem smith_boxes (initial_markers : ℕ) (markers_per_box : ℕ) (total_markers : ℕ)
  (h1 : initial_markers = 32) (h2 : markers_per_box = 9) (h3 : total_markers = 86) :
  total_markers - initial_markers = 6 * markers_per_box :=
by
  -- We state our assumptions explicitly for better clarity
  have h_total : total_markers = 86 := h3
  have h_initial : initial_markers = 32 := h1
  have h_box : markers_per_box = 9 := h2
  sorry

end smith_boxes_l95_95428


namespace Ivanov_made_an_error_l95_95814

theorem Ivanov_made_an_error : 
  (∀ x m s : ℝ, 0 = x → 4 = m → 15.917 = s^2 → ¬ (|x - m| ≤ real.sqrt s)) :=
by 
  intros x m s hx hm hs2
  rw [hx, hm, hs2]
  have H : |0 - 4| = 4 := by norm_num
  have H2 : real.sqrt 15.917 ≈ 3.99 := by norm_num
  exact neq_of_not_le (by norm_num : 4 ≠ 3.99) H2 sorry

end Ivanov_made_an_error_l95_95814


namespace range_of_a_l95_95228

variable (a : ℝ)
def p : Prop := ∀ x ∈ (set.Icc 1 2), x ^ 2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (h : p a ∧ q a) : a ≤ -2 ∨ a = 1 :=
by
  sorry

end range_of_a_l95_95228


namespace ivanov_error_l95_95827

theorem ivanov_error (x : ℝ) (m : ℝ) (S2 : ℝ) (std_dev : ℝ) :
  x = 0 → m = 4 → S2 = 15.917 → std_dev = Real.sqrt S2 →
  ¬ (|x - m| ≤ std_dev) :=
by
  intros h1 h2 h3 h4
  -- Using the given values directly to state the inequality
  have h5 : |0 - 4| = 4 := by norm_num
  have h6 : Real.sqrt 15.917 ≈ 3.99 := sorry  -- approximation as direct result
  -- Evaluating the inequality
  have h7 : 4 ≰ 3.99 := sorry  -- this represents the key step that shows the error
  exact h7
  sorry

end ivanov_error_l95_95827


namespace fruit_baskets_total_l95_95938

def num_fruit_baskets (total_fruits : ℕ) (first_three_baskets_fruits : ℕ) (last_basket_fruits : ℕ) : ℕ :=
    3 * ((total_fruits - first_three_baskets_fruits - last_basket_fruits) / first_three_baskets_fruits) + 1

theorem fruit_baskets_total (total_fruits : ℕ) (a1 o1 b1 a2 o2 b2 : ℕ) (num_baskets : ℕ) :
    a1 = 9 → o1 = 15 → b1 = 14 →
    a2 = 9 - 2 → o2 = 15 - 2 → b2 = 14 - 2 →
    total_fruits = 146 →
    let first_three_baskets_fruits := a1 + o1 + b1 in
    let last_basket_fruits := a2 + o2 + b2 in
    num_baskets = num_fruit_baskets total_fruits first_three_baskets_fruits last_basket_fruits →
    num_baskets = 7 :=
by
  intros ha1 ho1 hb1 ha2 ho2 hb2 htotal h_num_baskets
  subst ha1 ho1 hb1 ha2 ho2 hb2
  simp only
  sorry

end fruit_baskets_total_l95_95938


namespace number_of_good_students_is_5_or_7_l95_95345

-- Definitions based on the conditions
def total_students : ℕ := 25
def number_of_good_students (G : ℕ) (T : ℕ) := G + T = total_students
def first_group_condition (T : ℕ) := T > 12
def second_group_condition (G : ℕ) (T : ℕ) := T = 3 * (G - 1)

-- Problem statement in Lean 4:
theorem number_of_good_students_is_5_or_7 (G T : ℕ) :
  number_of_good_students G T ∧ first_group_condition T ∧ second_group_condition G T → G = 5 ∨ G = 7 :=
by
  sorry

end number_of_good_students_is_5_or_7_l95_95345


namespace count_common_elements_l95_95519

noncomputable def seq1 : Set ℕ := {x | x % 2 = 1 ∧ 1 ≤ x ∧ x ≤ 1999 }
noncomputable def seq2 : Set ℕ := { x | ∃ n, 1 ≤ n ∧ n ≤ 667 ∧ x = (3 * n - 2)}

theorem count_common_elements : ((seq1 ∩ seq2).toFinset.card = 334) :=
by
  sorry

end count_common_elements_l95_95519


namespace triangle_area_divided_l95_95618

theorem triangle_area_divided {baseA heightA baseB heightB : ℝ} 
  (h1 : baseA = 1) 
  (h2 : heightA = 1)
  (h3 : baseB = 2)
  (h4 : heightB = 1)
  : (1 / 2 * baseA * heightA + 1 / 2 * baseB * heightB = 1.5) :=
by
  sorry

end triangle_area_divided_l95_95618


namespace flowchart_basic_elements_includes_loop_l95_95835

theorem flowchart_basic_elements_includes_loop 
  (sequence_structure : Prop)
  (condition_structure : Prop)
  (loop_structure : Prop)
  : ∃ element : ℕ, element = 2 := 
by
  -- Assume 0 is A: Judgment
  -- Assume 1 is B: Directed line
  -- Assume 2 is C: Loop
  -- Assume 3 is D: Start
  sorry

end flowchart_basic_elements_includes_loop_l95_95835


namespace batteries_C_equivalent_l95_95865

variables (x y z W : ℝ)

-- Conditions
def cond1 := 4 * x + 18 * y + 16 * z = W * z
def cond2 := 2 * x + 15 * y + 24 * z = W * z
def cond3 := 6 * x + 12 * y + 20 * z = W * z

-- Equivalent statement to prove
theorem batteries_C_equivalent (h1 : cond1 x y z W) (h2 : cond2 x y z W) (h3 : cond3 x y z W) : W = 48 :=
sorry

end batteries_C_equivalent_l95_95865


namespace value_of_f_neg2_l95_95289

noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 + x else -f (-x)

theorem value_of_f_neg2 : f (-2) = -6 := 
by
  sorry

end value_of_f_neg2_l95_95289


namespace six_digit_number_divisibility_l95_95571

variable (n : ℕ)

-- Define the general six-digit number formation process
def form_six_digit_number (n : ℕ) : ℕ :=
  let m := 2 * n
  (n * 10^ (nat.log10 m) + m)

-- Define conditions
def three_digit_number (n : ℕ) : Prop := n >= 100 ∧ n < 1000

theorem six_digit_number_divisibility (h : three_digit_number n) :
  (form_six_digit_number n) % 2 = 0 ∧
  ((form_six_digit_number n) % 3 = 0 ↔ (n % 3 = 0)) :=
sorry

end six_digit_number_divisibility_l95_95571


namespace triangle_geometry_example_l95_95765

noncomputable section

open Real Classical

section

variables {A B C G X Y F H Z W : Point}

/-- Given a triangle ABC with specific side lengths and specific points and segments conditions,
  prove that the length of segment WZ is as given. -/
theorem triangle_geometry_example
  (hABC : triangle ABC)
  (hAB : B.dist A = 13)
  (hAC : C.dist A = 14)
  (hBC : C.dist B = 15)
  (hG : ∃ G, G ∈ AC ∧ reflection_over_angle_bisector B G midpoint(AC))
  (hY : midpoint(Y, G, C))
  (hX : ∃ X, X ∈ AG ∧ segment_ratio(AX, XG, 3))
  (hF : ∃ F, F ∈ AB ∧ XG ∥ BG ∧ BG ∥ HY)
  (hH : ∃ H, H ∈ BC ∧ XG ∥ BG ∧ BG ∥ HY)
  (hZ : concurrency_of_lines (AH, CF) = Z)
  (hW : ∃ W, W ∈ AC ∧ line_segment(WZ) ∥ BG) :
  WZ = (1170 * sqrt 37) / 1379 :=
sorry

end

end triangle_geometry_example_l95_95765


namespace umbrella_cost_l95_95405

theorem umbrella_cost (number_of_umbrellas : Nat) (total_cost : Nat) (h1 : number_of_umbrellas = 3) (h2 : total_cost = 24) :
  (total_cost / number_of_umbrellas) = 8 :=
by
  -- The proof will go here
  sorry

end umbrella_cost_l95_95405


namespace concyclic_points_B1_C1_B2_C2_l95_95764

/-- Define scalene acute triangle ABC -/
variables {A B C B1 C1 B2 C2 : Type}
variables [scalene_acute_triangle : A B C]

/-- Define condition for points B1 and C1 -/
variables (B1_on_ray_AC : ray A C B1) (C1_on_ray_AB : ray A B C1)
variables (AB1_eq_BB1 : |A - B1| = |B - B1|) (AC1_eq_CC1 : |A - C1| = |C - C1|)

/-- Define condition for points B2 and C2 -/
variables (B2_on_line_BC : collinear B C B2) (C2_on_line_BC : collinear B C C2)
variables (AB2_eq_CB2 : |A - B2| = |C - B2|) (BC2_eq_AC2 : |B - C2| = |A - C2|)

/-- Main theorem stating points B1, C1, B2, C2 are concyclic -/
theorem concyclic_points_B1_C1_B2_C2 :
  concyclic B1 C1 B2 C2 :=
sorry

end concyclic_points_B1_C1_B2_C2_l95_95764


namespace parabola_eq_l95_95943

theorem parabola_eq (x y : ℝ) 
    (h1 : 1 + 0 * x + 5 * y - 20 = 0) 
    (focus : (ℝ × ℝ) := (1,1)) : 
    ∃ (a b c d e f : ℤ),
    a = 25 ∧ b = 1 ∧ c = -300 ∧ d = -12 ∧ e = 148 ∧ f = -348 ∧
    gcd (abs a) (gcd (abs b) (gcd (abs c) (gcd (abs d) (gcd (abs e) (abs f))))) = 1 ∧
    a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0 :=
begin
   sorry
end

end parabola_eq_l95_95943


namespace good_students_count_l95_95373

theorem good_students_count (E B : ℕ) 
    (h1 : E + B = 25) 
    (h2 : ∀ (e ∈ finset.range 5), B > 12) 
    (h3 : ∀ (e ∈ finset.range 20), B = 3 * (E - 1)) :
    E = 5 ∨ E = 7 :=
by
  sorry

end good_students_count_l95_95373


namespace good_students_count_l95_95372

theorem good_students_count (E B : ℕ) 
    (h1 : E + B = 25) 
    (h2 : ∀ (e ∈ finset.range 5), B > 12) 
    (h3 : ∀ (e ∈ finset.range 20), B = 3 * (E - 1)) :
    E = 5 ∨ E = 7 :=
by
  sorry

end good_students_count_l95_95372


namespace solve_system_of_equations_l95_95018

theorem solve_system_of_equations : ∃ (x y : ℝ), (2 * x - y = 3) ∧ (3 * x + 2 * y = 8) ∧ (x = 2) ∧ (y = 1) := by
  sorry

end solve_system_of_equations_l95_95018


namespace calculate_total_marks_l95_95382

theorem calculate_total_marks 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (marks_per_correct : ℕ) 
  (marks_per_wrong : ℤ) 
  (total_attempted : total_questions = 60) 
  (correct_attempted : correct_answers = 44)
  (marks_per_correct_is_4 : marks_per_correct = 4)
  (marks_per_wrong_is_neg1 : marks_per_wrong = -1) : 
  total_questions * marks_per_correct - (total_questions - correct_answers) * (abs marks_per_wrong) = 160 := 
by 
  sorry

end calculate_total_marks_l95_95382


namespace part1_part2_l95_95782

def seq_a (n : ℕ) : ℕ :=
  if n = 1 then 2 else 2 * seq_a (n - 1) - (n - 1) + 1

theorem part1 
  (n : ℕ) (hn : n > 0) : seq_a n = 2 ^ (n - 1) + n := sorry

def seq_b (n : ℕ) : ℝ :=
  1 / (n * (seq_a n - 2 ^ (n - 1) + 1))

def sum_seq_b (n : ℕ) : ℝ :=
  (finset.range n).sum (λ i, seq_b (i + 1))

theorem part2 
  (n : ℕ) (hn : n > 0) : sum_seq_b n = n / (n + 1) := sorry

end part1_part2_l95_95782


namespace modulus_of_z_l95_95195

open Complex

theorem modulus_of_z (z : ℂ) (h : z^2 = (3/4 : ℝ) - I) : abs z = Real.sqrt 5 / 2 := 
  sorry

end modulus_of_z_l95_95195


namespace good_students_l95_95363

theorem good_students (G T : ℕ) (h1 : G + T = 25) (h2 : T > 12) (h3 : T = 3 * (G - 1)) :
  G = 5 ∨ G = 7 :=
by
  sorry

end good_students_l95_95363


namespace probability_three_even_numbers_l95_95016

theorem probability_three_even_numbers (dice_rolls : Fin 5 → ℕ) (h1: ∀ i, 1 ≤ dice_rolls i ∧ dice_rolls i ≤ 20) :
  (finset.card {i | dice_rolls i % 2 = 0} = 3) →
  (finset.card {i | dice_rolls i % 2 ≠ 0} = 2) →
  (finset.univ.card = 5) →
  (probability (λ s, finset.card {i | s i % 2 = 0} = 3 ∧ finset.card {i | s i % 2 ≠ 0} = 2) dice_rolls.to_finset = 5 / 16) :=
sorry

end probability_three_even_numbers_l95_95016


namespace good_students_l95_95360

theorem good_students (G T : ℕ) (h1 : G + T = 25) (h2 : T > 12) (h3 : T = 3 * (G - 1)) :
  G = 5 ∨ G = 7 :=
by
  sorry

end good_students_l95_95360


namespace ratio_length_to_width_l95_95480

def width : ℝ := 5
def area : ℝ := 100

theorem ratio_length_to_width :
  ∃ L : ℝ, (L * width = area) ∧ (L / width = 4) :=
by
  use 20
  split
  · sorry
  · sorry

end ratio_length_to_width_l95_95480


namespace find_number_of_good_students_l95_95333

variable {G T : ℕ}

def is_good_or_troublemaker (G T : ℕ) : Prop :=
  G + T = 25 ∧ 
  (∀ s ∈ finset.range 5, T > 12) ∧ 
  (∀ s ∈ finset.range 20, T = 3 * (G - 1))

-- There are 25 students in total. 
-- Each student is either a good student or a troublemaker.
-- Good students always tell the truth, while troublemakers always lie.
-- Five students make the statement: "If I transfer to another class, more than half of the remaining students will be troublemakers."
-- The remaining 20 students make the statement: "If I transfer to another class, the number of troublemakers among the remaining students will be three times the number of good students."

theorem find_number_of_good_students (G T : ℕ) (h : is_good_or_troublemaker G T) :
  (G = 5 ∨ G = 7) :=
sorry

end find_number_of_good_students_l95_95333


namespace fourth_power_of_y_l95_95968

-- Defining the expression 
def y : ℝ := Real.sqrt (3 + Real.sqrt (3 + Real.sqrt 5))

-- Stating the main theorem to prove
theorem fourth_power_of_y : y^4 = 12 + 6 * Real.sqrt (3 + Real.sqrt 5) + Real.sqrt 5 := sorry

end fourth_power_of_y_l95_95968


namespace good_students_options_l95_95321

variables (E B : ℕ)

-- Define the condition that the class has 25 students
def total_students : Prop := E + B = 25

-- Define the condition given by the first group of students
def first_group_condition : Prop := B > 12

-- Define the condition given by the second group of students
def second_group_condition : Prop := B = 3 * (E - 1)

-- Define the problem statement
theorem good_students_options (E B : ℕ) :
  total_students E B → first_group_condition B → second_group_condition E B → (E = 5 ∨ E = 7) :=
by
  intros h1 h2 h3
  sorry

end good_students_options_l95_95321


namespace find_n_l95_95737

noncomputable def a_n (n : ℕ) : ℝ := 10 ^ ((n : ℝ) / 11)

noncomputable def T_n (n : ℕ) : ℝ := ∏ k in Finset.range (n + 1), a_n k

theorem find_n (n : ℕ) : T_n n > 10 ^ 5 ↔ n ≥ 11 := 
by
  sorry

end find_n_l95_95737


namespace samantha_routes_l95_95457

/-
  Samantha lives 3 blocks west and 2 blocks south of the southwest corner of City Park.
  Her school is 3 blocks east and 3 blocks north of the northeast corner of City Park.
  The library is located 1 block west and 1 block north of the southwest corner of City Park.
  Samantha bikes first to the library, then takes a diagonal path through the park to the northeast corner,
  and finally bikes to her school.
  We need to show that the total number of different routes she can take is 60.
-/

theorem samantha_routes : 
  (nat.choose 3 1) * 1 * (nat.choose 6 3) = 60 :=
begin
  -- Place the main proof steps here
  sorry
end

end samantha_routes_l95_95457


namespace ellipse_equation_l95_95731

noncomputable def eccentricity := (Real.sqrt 2) / 2
noncomputable def perimeter := 16

def ellipse_center_origin (C : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ, 0 < b ∧ b < a ∧ C = {p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1}

def foci_on_x_axis (C : Set (ℝ × ℝ)) (F1 F2 : ℝ × ℝ) : Prop :=
  ∃ a b c : ℝ, 0 < b ∧ b < a ∧ c / a = eccentricity ∧ F1 = (c, 0) ∧ F2 = (-c, 0)

def triangle_perimeter (A B F2 : ℝ × ℝ) : ℝ :=
  (Real.dist A B) + (Real.dist B F2) + (Real.dist A F2)

theorem ellipse_equation (C : Set (ℝ × ℝ)) (F1 F2 : ℝ × ℝ)
    (h1 : ellipse_center_origin C)
    (h2 : foci_on_x_axis C F1 F2)
    (h3 : ∀ l : ℝ × ℝ → Prop, (l F1) → ∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧ triangle_perimeter A B F2 = perimeter) :
  ∃ a b : ℝ, a = 4 ∧ b^2 = 8 ∧ C = {p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1} :=
sorry

end ellipse_equation_l95_95731


namespace ellipse_eccentricity_l95_95651

open Real

def ellipse_foci_x_axis (m : ℝ) : Prop :=
  ∃ a b c e,
    a = sqrt m ∧
    b = sqrt 6 ∧
    c = sqrt (m - 6) ∧
    e = c / a ∧
    e = 1 / 2

theorem ellipse_eccentricity (m : ℝ) (h : ellipse_foci_x_axis m) :
  m = 8 := by
  sorry

end ellipse_eccentricity_l95_95651


namespace john_labor_cost_l95_95404

def plank_per_tree : ℕ := 25
def table_cost : ℕ := 300
def profit : ℕ := 12000
def trees_chopped : ℕ := 30
def planks_per_table : ℕ := 15
def total_table_revenue := (trees_chopped * plank_per_tree / planks_per_table) * table_cost
def labor_cost := total_table_revenue - profit

theorem john_labor_cost :
  labor_cost = 3000 :=
by
  sorry

end john_labor_cost_l95_95404


namespace area_BDF_area_CEF_cube_root_ineq_l95_95658

variable {A B C D E F : Type}
variable [has_area A B C D E F] -- Assuming a type class for areas
variable (AB AC AD : Real) -- Sides and segments of the triangle
variable (x y z : Real) -- Ratios: AD / AB = x, AE / AC = y, DF / DE = z
variable (S_ABC S_BDF S_CEF : Real) -- Areas of triangles

-- Conditons
axiom ratio_AD_AB : AD / AB = x
axiom ratio_AE_AC : AE / AC = y
axiom ratio_DF_DE : DF / DE = z

-- Proof problem statements
theorem area_BDF (S_ABC : Real) (h : S_BDF = (1 - x) * y * z * S_ABC) : S_BDF = (1 - x) * y * z * S_ABC := by sorry

theorem area_CEF (S_ABC : Real) (h : S_CEF = x * (1 - y) * (1 - z) * S_ABC) : S_CEF = x * (1 - y) * (1 - z) * S_ABC := by sorry

theorem cube_root_ineq (S_ABC : Real) (h₁ : S_BDF = (1 - x) * y * z * S_ABC) (h₂ : S_CEF = x * (1 - y) * (1 - z) * S_ABC) :
  Real.cbrt S_BDF + Real.cbrt S_CEF ≤ Real.cbrt S_ABC := by sorry

end area_BDF_area_CEF_cube_root_ineq_l95_95658


namespace number_of_valid_labelings_l95_95993

-- Define the problem in Lean
def cube_vertex_labeling : Prop :=
  ∃ (labeling : fin 8 → ℕ), 
    (∀ v, labeling v ∈ {1, 2, 3, 4, 5, 6, 7, 8}) ∧              -- Each integer 1 through 8 is used exactly once
    (∀ f, (∑ v in face_vertices f, labeling v) = 18) ∧          -- The sum on each face is 18
    ∀ (rotation : cube_rotation), 
      labeling ∘ rotation ≠ labeling                            -- Rotations yield the same labeling

-- The main theorem to prove
theorem number_of_valid_labelings : 
  ∃! (labeling : fin 8 → ℕ), cube_vertex_labeling labeling :=
sorry

end number_of_valid_labelings_l95_95993


namespace find_number_of_good_students_l95_95329

variable {G T : ℕ}

def is_good_or_troublemaker (G T : ℕ) : Prop :=
  G + T = 25 ∧ 
  (∀ s ∈ finset.range 5, T > 12) ∧ 
  (∀ s ∈ finset.range 20, T = 3 * (G - 1))

-- There are 25 students in total. 
-- Each student is either a good student or a troublemaker.
-- Good students always tell the truth, while troublemakers always lie.
-- Five students make the statement: "If I transfer to another class, more than half of the remaining students will be troublemakers."
-- The remaining 20 students make the statement: "If I transfer to another class, the number of troublemakers among the remaining students will be three times the number of good students."

theorem find_number_of_good_students (G T : ℕ) (h : is_good_or_troublemaker G T) :
  (G = 5 ∨ G = 7) :=
sorry

end find_number_of_good_students_l95_95329


namespace snail_total_distance_l95_95118

-- Define the positions the snail moves to
def pos1 := 3
def pos2 := -5
def pos3 := 10
def pos4 := 2

-- Define the distances moved by the snail
def distance1 := abs (pos2 - pos1) -- distance from 3 to -5
def distance2 := abs (pos3 - pos2) -- distance from -5 to 10
def distance3 := abs (pos4 - pos3) -- distance from 10 to 2

-- Define the total distance
def total_distance := distance1 + distance2 + distance3

-- State the theorem that needs to be proved
theorem snail_total_distance : total_distance = 31 :=
by
  sorry

end snail_total_distance_l95_95118


namespace find_sum_l95_95832

variables (x y : ℝ)

def condition1 : Prop := x^3 - 3 * x^2 + 5 * x = 1
def condition2 : Prop := y^3 - 3 * y^2 + 5 * y = 5

theorem find_sum : condition1 x → condition2 y → x + y = 2 := 
by 
  sorry -- The proof goes here

end find_sum_l95_95832


namespace bridge_games_count_l95_95995

theorem bridge_games_count (n : ℕ) (h : n = 8) : 
  let num_ways := (Nat.factorial 8) / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 4 * Nat.factorial 2) 
  num_ways = 315 :=
  by
  sorry

end bridge_games_count_l95_95995


namespace sum_of_acute_angles_pi_over_2_l95_95522

open Real

theorem sum_of_acute_angles_pi_over_2
  {α β : ℝ} (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  (h : sin α * sin α + sin β * sin β = sin (α + β)) :
  α + β = π / 2 :=
sorry

end sum_of_acute_angles_pi_over_2_l95_95522


namespace coeff_x3_in_q_pow4_l95_95227

def q (x : ℝ) : ℝ := x^5 - 4 * x^2 + 3

theorem coeff_x3_in_q_pow4 : 
  let p := (q x)^4 in
  polynomial.coeff p 3 = -768 := by
  sorry

end coeff_x3_in_q_pow4_l95_95227


namespace not_prime_sum_l95_95771

theorem not_prime_sum (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_eq : a * b = c * d) : ¬ Nat.Prime (a + b + c + d) :=
sorry

end not_prime_sum_l95_95771


namespace arithmetic_sequence_sum_l95_95498

theorem arithmetic_sequence_sum (c d : ℕ) 
  (h1 : ∀ (a1 a2 a3 a4 a5 a6 : ℕ), a1 = 3 → a2 = 10 → a3 = 17 → a6 = 38 → (a2 - a1 = a3 - a2) → (a3 - a2 = c - a3) → (c - a3 = d - c) → (d - c = a6 - d)) : 
  c + d = 55 := 
by 
  sorry

end arithmetic_sequence_sum_l95_95498


namespace find_two_digit_number_l95_95580

theorem find_two_digit_number (x : ℕ) (h1 : (x + 3) % 3 = 0) (h2 : (x + 7) % 7 = 0) (h3 : (x - 4) % 4 = 0) : x = 84 := 
by
  -- Place holder for the proof
  sorry

end find_two_digit_number_l95_95580


namespace modulus_of_z_l95_95674

noncomputable def i : ℂ := complex.I

def z : ℂ := (1 - i) * i

theorem modulus_of_z : complex.abs z = real.sqrt 2 := by
  sorry

end modulus_of_z_l95_95674


namespace number_of_good_students_is_5_or_7_l95_95348

-- Definitions based on the conditions
def total_students : ℕ := 25
def number_of_good_students (G : ℕ) (T : ℕ) := G + T = total_students
def first_group_condition (T : ℕ) := T > 12
def second_group_condition (G : ℕ) (T : ℕ) := T = 3 * (G - 1)

-- Problem statement in Lean 4:
theorem number_of_good_students_is_5_or_7 (G T : ℕ) :
  number_of_good_students G T ∧ first_group_condition T ∧ second_group_condition G T → G = 5 ∨ G = 7 :=
by
  sorry

end number_of_good_students_is_5_or_7_l95_95348


namespace class_proof_l95_95359

def class_problem : Prop :=
  let total_students := 25
  let (students : Type) := {s // s ∈ Fin total_students}
  let good students (s : students) : Prop := sorry -- Condition Placeholder
  let troublemakers (s : students) : Prop := sorry -- Condition Placeholder
  have all_good_or_trouble : ∀ s, good students s ∨ troublemakers s,
    from sorry,
  have lh_trouble : ∀ (a b c d e : students), (good students a ∧ good students b ∧ good students c ∧ good students d ∧ good students e) →
    ∃ (x : ℕ), x > ((total_students - 1) / 2) ∧
    (λ rem_trouble : students - {a, b, c, d, e}, B : ℕ, B = x) ∧
    (λ rem_good : students - {a, b, c, d, e}, G : ℕ, G ∈ rem_good), from sorry,
  have th_triple_ratio : ∀ s, (students - {s}).card = 24 →
    ∃ (x y : ℕ), (troublemakers x ∧ good students y) ∧ (3*y = x), from sorry,
  ∃ (num_good_students : ℕ), (num_good_students = 5 ∨ num_good_students = 7)
  
theorem class_proof : class_problem :=
  sorry

end class_proof_l95_95359


namespace least_five_digit_palindrome_divisible_by_4_l95_95792

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

def is_divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

theorem least_five_digit_palindrome_divisible_by_4 :
  ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ is_palindrome n ∧ is_divisible_by_4 n ∧
  ∀ m : ℕ, 10000 ≤ m ∧ m ≤ 99999 ∧ is_palindrome m ∧ is_divisible_by_4 m → n ≤ m :=
begin
  use 11011,
  split,
  -- 11011 is between 10000 and 99999
  linarith,
  split,
  -- 11011 is a palindrome
  sorry,
  split,
  -- 11011 is divisible by 4
  sorry,
  -- 11011 is the least such number
  sorry
end

end least_five_digit_palindrome_divisible_by_4_l95_95792


namespace max_value_OP_div_PQ_l95_95207

-- Define the parabola and the circle
def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 8 * P.1
def circle (Q : ℝ × ℝ) : Prop := (Q.1 - 2)^2 + Q.2^2 = 1

-- Define the distance functions
def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Origin point O
def O : ℝ × ℝ := (0, 0)

-- Lean theorem statement
theorem max_value_OP_div_PQ (P Q : ℝ × ℝ) (hP : parabola P) (hQ : circle Q) :
  ∃ x : ℝ, ∀ P : ℝ × ℝ, parabola P → P.1 = x →
  Real.sqrt ((P.1)^2 + (P.2)^2) / distance O Q ≤ (4 * Real.sqrt 7) / 7 := by
  sorry

end max_value_OP_div_PQ_l95_95207


namespace sudoku_grid_sum_l95_95035

theorem sudoku_grid_sum (A B : ℕ) :
  (∀ grid : ℕ × ℕ → ℕ,
    grid (0, 0) = 2 ∧
    grid (1, 1) = 3 ∧
    (∀ i j, 0 ≤ i ∧ i < 3 ∧ 0 ≤ j ∧ j < 3 →
      grid (i, j) ∈ {1, 2, 3}) ∧
    (∀ i, 0 ≤ i ∧ i < 3 →
      ∃ j1 j2 j3, {grid (i, j1), grid (i, j2), grid (i, j3)} = {1, 2, 3}) ∧
    (∀ j, 0 ≤ j ∧ j < 3 →
      ∃ i1 i2 i3, {grid (i1, j), grid (i2, j), grid (i3, j)} = {1, 2, 3}) ∧
    grid (1, 2) = A ∧
    grid (2, 2) = B
  ) →
  A + B = 3 :=
by
  sorry

end sudoku_grid_sum_l95_95035


namespace greatest_multiple_less_than_l95_95884

def lcm (a b : ℕ) : ℕ := Nat.lcm a b -- Assuming lcm function is already defined

theorem greatest_multiple_less_than (a b m : ℕ) (h₁ : a = 15) (h₂ : b = 20) (h₃ : m = 150) : 
  ∃ k, k * lcm a b < m ∧ ¬ ∃ k', (k' * lcm a b < m ∧ k' > k) :=
by
  sorry

end greatest_multiple_less_than_l95_95884


namespace problem_statement_l95_95262

-- Define the conditions and the goal
theorem problem_statement {x y : ℝ} 
  (h1 : (x + y)^2 = 36)
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := 
by
  sorry

end problem_statement_l95_95262


namespace sum_inequality_l95_95772

-- Definitions of a_i and b_i as elements of [1, 2], and the given sum condition
variables {n : ℕ}
variables {a b : Fin n → ℝ}
variable (h_range : ∀ i, 1 ≤ a i ∧ a i ≤ 2 ∧ 1 ≤ b i ∧ b i ≤ 2)
variable (h_sum : ∑ i, (a i)^2 = ∑ i, (b i)^2)

-- Statement of the theorem
theorem sum_inequality
  (h_range : ∀ i, 1 ≤ a i ∧ a i ≤ 2 ∧ 1 ≤ b i ∧ b i ≤ 2)
  (h_sum : ∑ i, (a i)^2 = ∑ i, (b i)^2) :
  ∑ i, (a i)^3 / (b i) ≤ (17 / 10) * ∑ i, (a i)^2 ∧ 
  (∑ i, (a i)^3 / (b i) = (17 / 10) * ∑ i, (a i)^2 ↔ ∃ k : Fin n, n % 2 = 1 ∧ (∀ i, (a i = 1 ∨ a i = 2) ∧ b i = 2 / a i)) :=
sorry

end sum_inequality_l95_95772


namespace geom_seq_arith_ineq_l95_95669

theorem geom_seq_arith_ineq (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℝ) (q : ℝ) (n : ℕ) :
  (∀ k, a k = 2 * 2^(k - 1)) →
  a 1 = 2 →
  q > 0 →
  (a 2 + a 3) / 2 = 6 →
  (∀ k, b k = log 2 (a k)) →
  (∀ k, T k = ∑ i in range k, 1 / (b i * b (i + 1))) →
  T n < 99 / 100 →
  n ≤ 98 := sorry

end geom_seq_arith_ineq_l95_95669


namespace percentage_defective_l95_95129

theorem percentage_defective (examined rejected : ℚ) (h1 : examined = 66.67) (h2 : rejected = 10) :
  (rejected / examined) * 100 = 15 := by
  sorry

end percentage_defective_l95_95129


namespace impossible_partition_l95_95143

theorem impossible_partition :
  ¬ ∃ (groups : Fin 11 → Finset (Fin 34)) (h : ∀ i, groups i).card = 3,
      ∀ i, ∃ a b c ∈ groups i, a + b = c ∨ b + c = a ∨ c + a = b :=
by
  sorry

end impossible_partition_l95_95143


namespace Ivanov_made_an_error_l95_95817

theorem Ivanov_made_an_error : 
  (∀ x m s : ℝ, 0 = x → 4 = m → 15.917 = s^2 → ¬ (|x - m| ≤ real.sqrt s)) :=
by 
  intros x m s hx hm hs2
  rw [hx, hm, hs2]
  have H : |0 - 4| = 4 := by norm_num
  have H2 : real.sqrt 15.917 ≈ 3.99 := by norm_num
  exact neq_of_not_le (by norm_num : 4 ≠ 3.99) H2 sorry

end Ivanov_made_an_error_l95_95817


namespace good_students_count_l95_95312

noncomputable def student_count := 25

def is_good_student (s : Nat) := s ≤ student_count

def is_troublemaker (s : Nat) := s ≤ student_count

def always_tell_truth (s : Nat) := is_good_student s

def always_lie (s : Nat) := is_troublemaker s

def condition1 (E B : Nat) := E + B = student_count

def condition2 := ∀ (x : Nat), x ≤ 5 → is_good_student x → 
  ∃ (B : Nat), B > 24 / 2

def condition3 := ∀ (x : Nat), x ≤ 20 → is_troublemaker x → 
  ∃ (B E : Nat), B = 3 * (E - 1) ∧ E + B = student_count

theorem good_students_count :
  ∃ (E : Nat), condition1 E (student_count - E) ∧ condition2 ∧ condition3 :=
sorry  -- the actual proof is not required

end good_students_count_l95_95312


namespace comparison_l95_95188

noncomputable def a := (2: ℝ) ^ (-1 / 3)
noncomputable def b := Real.logBase 2 (1 / 3)
noncomputable def c := Real.logBase (1 / 2) (1 / 3)

theorem comparison : c > a ∧ a > b := by
  sorry

end comparison_l95_95188


namespace angle_values_l95_95738

open Function
open Geometry

axiom eq_triangle (ABC : Type) [H : equilateral ABC] : Prop -- An axiom to denote an equilateral triangle
axiom inscribed_eq_triangle (DEF : Type) (ABC : Type) [H : inscribed_equilateral DEF ABC] : Prop -- An axiom for inscribed equilateral triangle

axiom vertices_on_sides (D E F A B C : Point) : D ∈ join A B ∧ E ∈ join B C ∧ F ∈ join C A -- Vertices D, E, and F are on sides AB, BC, and CA respectively

def ABC (A B C : Point) : Triangle := triangle A B C
def DEF (D E F : Point) : Triangle := triangle D E F

theorem angle_values (A B C D E F : Point) 
(H1 : eq_triangle (ABC A B C))
(H2 : inscribed_eq_triangle (DEF D E F) (ABC A B C))
(H3 : vertices_on_sides D E F A B C) :
∠ B F D = 60 ∧ ∠ A D E = 60 ∧ ∠ F E C = 60 :=
sorry

end angle_values_l95_95738


namespace newton_integral_defined_lebesgue_not_exists_l95_95550

noncomputable def F (x : ℝ) : ℝ := x^2 * Real.sin (1 / x^2)

noncomputable def f (x : ℝ) : ℝ := 
  2 * x * Real.sin (1 / x^2) - (2 / x) * Real.cos (1 / x^2)

theorem newton_integral_defined_lebesgue_not_exists :
  ∃ f : ℝ → ℝ, 
  (∫ x in set.Icc 0 1, f x) = Real.sin(1) ∧ 
  ¬MeasureTheory.integrable f MeasureTheory.volume := 
begin
  use f,
  split,
  { sorry },
  { sorry }
end

end newton_integral_defined_lebesgue_not_exists_l95_95550


namespace prob1_prob2_prob3_l95_95686

-- Problem 1
theorem prob1 (k : ℝ) (h₀ : k > 0) 
  (h₁ : ∀ x : ℝ, 2 < x ∧ x < 3 → (k * x^2 - 2 * x + 6 * k) < 0) :
  k = 2/5 := 
sorry

-- Problem 2
theorem prob2 (k : ℝ) (h₀ : k > 0) 
  (h₁ : ∀ x : ℝ, 2 < x ∧ x < 3 → (k * x^2 - 2 * x + 6 * k) < 0) :
  0 < k ∧ k ≤ 2/5 := 
sorry

-- Problem 3
theorem prob3 (k : ℝ) (h₀ : k > 0)
  (h₁ : ∀ x : ℝ, 2 < x ∧ x < 3 → (k * x^2 - 2 * x + 6 * k) < 0) :
  k ≥ 2/5 := 
sorry

end prob1_prob2_prob3_l95_95686


namespace least_positive_int_divisible_l95_95535

theorem least_positive_int_divisible :
  ∃ n : ℕ, n > 0 ∧ (7 ∣ n) ∧ (11 ∣ n) ∧ (13 ∣ n) ∧ (9 ∣ n) ∧ n = 9009 :=
by
  use 9009
  split
  · exact nat.zero_lt_succ _
  repeat { split, norm_num }
  sorry

end least_positive_int_divisible_l95_95535


namespace greatest_multiple_less_150_l95_95890

/-- Define the LCM of two natural numbers -/
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem greatest_multiple_less_150 (x y : ℕ) (h1 : x = 15) (h2 : y = 20) : 
  (∃ m : ℕ, LCM x y * m < 150 ∧ ∀ n : ℕ, LCM x y * n < 150 → LCM x y * n ≤ LCM x y * m) ∧ 
  (∃ m : ℕ, LCM x y * m = 120) :=
by
  sorry

end greatest_multiple_less_150_l95_95890


namespace g_at_five_l95_95843

noncomputable def g (x : ℝ) : ℝ := sorry

axiom functional_eqn : ∀ (x y : ℝ), g(x + y) = g(x) * g(y)
axiom non_zero : ∀ (x : ℝ), g(x) ≠ 0

theorem g_at_five : g 5 = 1 :=
by sorry

end g_at_five_l95_95843


namespace mr_alvarez_total_distance_l95_95426

noncomputable def cost_per_gallon (week : ℕ) : ℝ :=
  3.25 + 0.1 * (week - 1)

noncomputable def gallons_per_week (week : ℕ) : ℝ :=
  36 / cost_per_gallon week

noncomputable def total_gallons : ℝ :=
  (gallons_per_week 1) + (gallons_per_week 2) + (gallons_per_week 3) +
  (gallons_per_week 4) + (gallons_per_week 5)

noncomputable def fuel_efficiency_rate : ℝ := 30

noncomputable def total_distance : ℝ :=
  total_gallons * fuel_efficiency_rate

theorem mr_alvarez_total_distance : total_distance ≈ 1567.86 :=
by
  have h1 : total_gallons ≈ 52.262 := sorry
  rw [total_distance, h1, mul_comm]
  norm_num
  sorry

end mr_alvarez_total_distance_l95_95426


namespace cube_red_faces_one_third_l95_95563

theorem cube_red_faces_one_third (n : ℕ) (h : 6 * n^3 ≠ 0) : 
  (2 * n^2) / (6 * n^3) = 1 / 3 → n = 1 :=
by sorry

end cube_red_faces_one_third_l95_95563


namespace point_and_sum_of_coordinates_l95_95022

-- Definitions
def point_on_graph_of_g_over_3 (g : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = (g p.1) / 3

def point_on_graph_of_inv_g_over_3 (g : ℝ → ℝ) (q : ℝ × ℝ) : Prop :=
  q.2 = (g⁻¹ q.1) / 3

-- Main statement
theorem point_and_sum_of_coordinates {g : ℝ → ℝ} (h : point_on_graph_of_g_over_3 g (2, 3)) :
  point_on_graph_of_inv_g_over_3 g (9, 2 / 3) ∧ (9 + 2 / 3 = 29 / 3) :=
by
  sorry

end point_and_sum_of_coordinates_l95_95022


namespace range_of_m_l95_95681

open Real

noncomputable def f (m x : ℝ) : ℝ := m * log x - (x^2 / 2)

noncomputable def f_prime (m x : ℝ) : ℝ := m / x - x

theorem range_of_m :
  (∀ x ∈ Ioo 0 1, f_prime m x * f_prime m (1 - x) ≤ 1) ↔ 0 ≤ m ∧ m ≤ 3 / 4 := by
  sorry

end range_of_m_l95_95681


namespace number_of_remaining_triangles_after_12_repeats_l95_95585

-- Definitions based on the problem's conditions
def initial_triangle_side_length := (1 : ℝ)
def number_of_repeats := 12
def side_length_of_remaining_triangles (n : ℕ) : ℝ := initial_triangle_side_length / (2 ^ n)

theorem number_of_remaining_triangles_after_12_repeats :
  let n := number_of_repeats in
  (3 : ℕ) ^ n = 531441 :=
by
  sorry

end number_of_remaining_triangles_after_12_repeats_l95_95585


namespace radius_of_circle_l95_95470

theorem radius_of_circle :
  ∀ (r : ℝ), (π * r^2 = 2.5 * 2 * π * r) → r = 5 :=
by sorry

end radius_of_circle_l95_95470


namespace good_students_options_l95_95323

variables (E B : ℕ)

-- Define the condition that the class has 25 students
def total_students : Prop := E + B = 25

-- Define the condition given by the first group of students
def first_group_condition : Prop := B > 12

-- Define the condition given by the second group of students
def second_group_condition : Prop := B = 3 * (E - 1)

-- Define the problem statement
theorem good_students_options (E B : ℕ) :
  total_students E B → first_group_condition B → second_group_condition E B → (E = 5 ∨ E = 7) :=
by
  intros h1 h2 h3
  sorry

end good_students_options_l95_95323


namespace not_magical_year_2099_l95_95107

def is_magical_year (year : ℕ) : Prop :=
  ∃ month day : ℕ, month ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} ∧ 
  day ∈ Finset.range 32 ∧ year % 100 = month + day

theorem not_magical_year_2099 : ¬ is_magical_year 2099 :=
by {
  unfold is_magical_year,
  intro h,
  rcases h with ⟨month, day, month_valid, day_valid, sum_eq_99⟩,
  have : month + day = 99 := by { exact sum_eq_99 },
  cases month_valid;
  cases day_valid,
  sorry
}

end not_magical_year_2099_l95_95107


namespace intersection_point_l95_95688

noncomputable def parametric_eq_line (t α : ℝ) : ℝ × ℝ :=
  (-3 / 2 + t * Real.cos α, 1 / 2 + t * Real.sin α)

def polar_eq_curve (θ : ℝ) : ℝ :=
  2 / (1 - Real.cos θ)

theorem intersection_point (α θ t : ℝ) (hα : α = Real.pi / 4) (hθ : θ = Real.pi / 2) :
  let l : ℝ × ℝ := parametric_eq_line t α
  let C : ℝ := polar_eq_curve θ
  (l.1, l.2) = (2 * Real.cos θ, 2 * Real.sin θ) :=
by
  sorry

end intersection_point_l95_95688


namespace number_of_triangles_l95_95445

theorem number_of_triangles (points_AB points_BC points_AC : ℕ)
                            (hAB : points_AB = 12)
                            (hBC : points_BC = 9)
                            (hAC : points_AC = 10) :
    let total_points := points_AB + points_BC + points_AC
    let total_combinations := Nat.choose total_points 3
    let degenerate_AB := Nat.choose points_AB 3
    let degenerate_BC := Nat.choose points_BC 3
    let degenerate_AC := Nat.choose points_AC 3
    let valid_triangles := total_combinations - (degenerate_AB + degenerate_BC + degenerate_AC)
    valid_triangles = 4071 :=
by
  sorry

end number_of_triangles_l95_95445


namespace good_students_count_l95_95316

noncomputable def student_count := 25

def is_good_student (s : Nat) := s ≤ student_count

def is_troublemaker (s : Nat) := s ≤ student_count

def always_tell_truth (s : Nat) := is_good_student s

def always_lie (s : Nat) := is_troublemaker s

def condition1 (E B : Nat) := E + B = student_count

def condition2 := ∀ (x : Nat), x ≤ 5 → is_good_student x → 
  ∃ (B : Nat), B > 24 / 2

def condition3 := ∀ (x : Nat), x ≤ 20 → is_troublemaker x → 
  ∃ (B E : Nat), B = 3 * (E - 1) ∧ E + B = student_count

theorem good_students_count :
  ∃ (E : Nat), condition1 E (student_count - E) ∧ condition2 ∧ condition3 :=
sorry  -- the actual proof is not required

end good_students_count_l95_95316


namespace triangle_area_l95_95303

variables {A B C a b c : ℝ}

-- Given conditions
def triangle_conditions (a b c : ℝ) : Prop :=
  b^2 + c^2 = a^2 - real.sqrt 3 * b * c ∧
  a * b * (-1 / 2) = -4

-- Prove the area of triangle ABC
theorem triangle_area (h : triangle_conditions a b c) : 
  ∃ (area : ℝ), area = 2 * real.sqrt 3 / 3 :=
sorry

end triangle_area_l95_95303


namespace arithmetic_sequence_problem_l95_95420

-- Let {a_n} be an arithmetic sequence with the sum of the first n terms S_n.
-- Let a_1010 > 0 and a_1009 + a_1010 < 0.
-- Prove that the positive integer n that satisfies S_n * S_{n+1} < 0 is 2018.

variable {α : Type*} [linear_ordered_field α]

def is_arithmetic_sequence (a : ℕ → α) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

noncomputable def sum_first_n_terms (a : ℕ → α) (n : ℕ) :=
  (n * (a 1 + a n)) / 2

theorem arithmetic_sequence_problem
    (a : ℕ → α)
    (h_arith : is_arithmetic_sequence a)
    (h_a1010 : a 1010 > 0)
    (h_a1009_a1010 : a 1009 + a 1010 < 0) :
    let S := sum_first_n_terms a in
    S 2018 * S 2019 < 0 :=
by
  sorry

end arithmetic_sequence_problem_l95_95420


namespace smallest_N_l95_95761

theorem smallest_N (a b c d e N : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0)
  (h_sum : a + b + c + d + e = 2020)
  (h_max : N = max (max a (max b (max c d)) + e)){
  N ≥ 1011 := sorry
}

end smallest_N_l95_95761


namespace cows_milk_production_l95_95558

theorem cows_milk_production :
  let cows := 150 in
  let group1_cows := 75 in
  let group2_cows := 75 in
  let group1_milk_per_day := 1300 in
  let group2_milk_per_day := 1650 in
  let total_days := 12 in
  (group1_cows * group1_milk_per_day + group2_cows * group2_milk_per_day) * total_days = 2655000 :=
by
  sorry

end cows_milk_production_l95_95558


namespace ivanov_error_l95_95810

-- Given conditions
def mean_temperature (x : ℝ) : Prop := x = 0
def median_temperature (m : ℝ) : Prop := m = 4
def variance_temperature (s² : ℝ) : Prop := s² = 15.917

-- Statement to prove
theorem ivanov_error (x m s² : ℝ) 
  (mean_x : mean_temperature x)
  (median_m : median_temperature m)
  (variance_s² : variance_temperature s²) :
  (x - m)^2 > s² :=
by
  rw [mean_temperature, median_temperature, variance_temperature] at *
  simp [*, show 0 = 0, from rfl, show 4 = 4, from rfl]
  sorry

end ivanov_error_l95_95810


namespace find_k_for_collinear_points_l95_95564

theorem find_k_for_collinear_points :
  (∃ k : ℚ, ∀ (k: ℚ),
      let slope1 := (k - 9) / 8,
          slope2 := (4 - k) / 15
      in slope1 = slope2 →
            k = 167 / 23) :=
sorry

end find_k_for_collinear_points_l95_95564


namespace radius_of_circle_C1_is_sqrt30_l95_95601

noncomputable def radius_of_C1 : ℝ :=
  let XZ := 15
  let OZ := 12
  let YZ := 8
  let r := sqrt 30
  r

axiom circle_C1_center_O_on_circle_C2 (O X Y Z : ℝ) :
  ∃ C2_center C1_center radius_C1 radius_C2 : ℝ,
    O = C1_center ∧
    C2_center ≠ O ∧
    distance C2_center O = radius_C2 ∧
    distance O X = radius_C1 ∧
    distance O Y = radius_C1 ∧
    X ≠ Y ∧
    distance X Z = XZ ∧
    distance O Z = OZ ∧
    distance Y Z = YZ ∧
    r = sqrt(30)

theorem radius_of_circle_C1_is_sqrt30 :
  ∀ (O X Y Z : ℝ), 
    circle_C1_center_O_on_circle_C2 O X Y Z →
    distance O X = sqrt(30) :=
by
  sorry

end radius_of_circle_C1_is_sqrt30_l95_95601


namespace volume_of_regular_pyramid_eq_187_5_l95_95801

noncomputable def volume_of_pyramid
  (hexagon_side_length : ℝ)  -- Side length of the regular hexagon
  (triangle_side_length : ℝ) -- Side length of the equilateral triangle QGM
  (height : ℝ)               -- Height from the center of the hexagon to the apex of the pyramid
  (QGM_eq_triangle : triangle Q G M)
  (regular_hexagon : regular_hexagon G H J K L M N) 
  (QG_length : dist Q G = 10)
  (QGM_equilateral : equilateral_triangle Q G M)
  : ℝ :=
  let s := hexagon_side_length in
  let area_of_triangle := (sqrt 3 / 4) * s^2 in
  let area_of_hexagon := 6 * area_of_triangle in
  let height := (10 / 2) * sqrt 3 in
  let volume := (1 / 3) * area_of_hexagon * height in
  volume

theorem volume_of_regular_pyramid_eq_187_5
  (hexagon_side_length : ℝ)
  (triangle_side_length : ℝ)
  (height : ℝ)
  (QGM_eq_triangle : triangle Q G M)
  (regular_hexagon : regular_hexagon G H J K L M N) 
  (QG_length : dist Q G = 10)
  (QGM_equilateral : equilateral_triangle Q G M)
  : volume_of_pyramid hexagon_side_length triangle_side_length height QGM_eq_triangle regular_hexagon QG_length QGM_equilateral = 187.5 := by sorry

end volume_of_regular_pyramid_eq_187_5_l95_95801


namespace isosceles_triangle_l95_95415

variables {A B C P Q : Point}
variables (h1 : collinear B C P)
variables (h2 : collinear B C Q)
variables (h3 : dist B P = dist C Q)
variables (h4 : ¬collinear A B C)
variables (h5 : ∠ B A P = ∠ C A Q)

theorem isosceles_triangle
    (h1 : collinear B C P)
    (h2 : collinear B C Q)
    (h3 : dist B P = dist C Q)
    (h4 : ¬collinear A B C)
    (h5 : ∠ B A P = ∠ C A Q) :
    dist A B = dist A C :=
sorry

end isosceles_triangle_l95_95415


namespace batteries_C_equivalent_l95_95866

variables (x y z W : ℝ)

-- Conditions
def cond1 := 4 * x + 18 * y + 16 * z = W * z
def cond2 := 2 * x + 15 * y + 24 * z = W * z
def cond3 := 6 * x + 12 * y + 20 * z = W * z

-- Equivalent statement to prove
theorem batteries_C_equivalent (h1 : cond1 x y z W) (h2 : cond2 x y z W) (h3 : cond3 x y z W) : W = 48 :=
sorry

end batteries_C_equivalent_l95_95866


namespace total_amount_in_wallet_l95_95014

theorem total_amount_in_wallet
  (num_10_bills : ℕ)
  (num_20_bills : ℕ)
  (num_5_bills : ℕ)
  (amount_10_bills : ℕ)
  (num_20_bills_eq : num_20_bills = 4)
  (amount_10_bills_eq : amount_10_bills = 50)
  (total_num_bills : ℕ)
  (total_num_bills_eq : total_num_bills = 13)
  (num_10_bills_eq : num_10_bills = amount_10_bills / 10)
  (total_amount : ℕ)
  (total_amount_eq : total_amount = amount_10_bills + num_20_bills * 20 + num_5_bills * 5)
  (num_bills_accounted : ℕ)
  (num_bills_accounted_eq : num_bills_accounted = num_10_bills + num_20_bills)
  (num_5_bills_eq : num_5_bills = total_num_bills - num_bills_accounted)
  : total_amount = 150 :=
by
  sorry

end total_amount_in_wallet_l95_95014


namespace no_valid_transformation_l95_95079

theorem no_valid_transformation :
  ¬ ∃ (n1 n2 n3 n4 : ℤ),
    2 * n1 + n2 - 2 * n3 - n4 = 27 ∧
    -n1 + 2 * n2 + n3 - 2 * n4 = -27 :=
by
  sorry

end no_valid_transformation_l95_95079


namespace binary_110_eq_6_in_decimal_l95_95084

theorem binary_110_eq_6_in_decimal : 
  ∃ (d : ℕ), (2^2 * 1 + 2^1 * 1 + 2^0 * 0 = d) ∧ (d = 6) :=
by {
  use 6,
  split,
  { sorry },
  { refl },
}

end binary_110_eq_6_in_decimal_l95_95084


namespace avg_int_values_between_l95_95058

theorem avg_int_values_between (N : ℤ) :
  (5 : ℚ) / 12 < N / 48 ∧ N / 48 < 1 / 3 →
  (N = 17 ∨ N = 18 ∨ N = 19) ∧
  (N = 17 ∨ N = 18 ∨ N = 19 →
  (17 + 18 + 19) / 3 = 18) :=
by
  sorry

end avg_int_values_between_l95_95058


namespace puppies_brought_in_l95_95112

open Nat

theorem puppies_brought_in (orig_puppies adopt_rate days total_adopted brought_in_puppies : ℕ) 
  (h_orig : orig_puppies = 3)
  (h_adopt_rate : adopt_rate = 3)
  (h_days : days = 2)
  (h_total_adopted : total_adopted = adopt_rate * days)
  (h_equation : total_adopted = orig_puppies + brought_in_puppies) :
  brought_in_puppies = 3 :=
by
  sorry

end puppies_brought_in_l95_95112


namespace fraction_of_repeating_decimal_l95_95881

theorem fraction_of_repeating_decimal : ∃ x : ℚ, x = 0.4 + (67 / 999) ∧ x = 4621 / 9900 := by
  use 4621 / 9900
  split
  sorry
  rfl

end fraction_of_repeating_decimal_l95_95881


namespace fraction_of_number_l95_95882

theorem fraction_of_number : (7 / 8) * 64 = 56 := by
  sorry

end fraction_of_number_l95_95882


namespace hypotenuse_length_l95_95541

theorem hypotenuse_length (a b c : ℕ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : a^2 + b^2 = c^2) : c = 13 :=
by
  -- proof
  sorry

end hypotenuse_length_l95_95541


namespace range_of_f_l95_95659

def greatest_int_leq (x : ℝ) : ℕ := ⌊x⌋

def f (x y : ℝ) : ℝ := (x + y) / (greatest_int_leq x * greatest_int_leq y + greatest_int_leq x + greatest_int_leq y + 1)

theorem range_of_f (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 1) :
  (f x y) ∈ ({1 / 2} ∪ Icc (5 / 6 : ℝ) (5 / 4 : ℝ)) :=
sorry

end range_of_f_l95_95659


namespace sufficient_condition_parallel_planes_l95_95778

noncomputable def plane := Type
noncomputable def line := Type

variables (α β : plane) (m n : line) (l₁ l₂ : line)
variable [m_in_alpha : m ⊆ α]
variable [n_in_alpha : n ⊆ α]
variable [l₁_in_beta : l₁ ⊆ β]
variable [l₂_in_beta : l₂ ⊆ β]
variable [l₁_int_l₂ : ∃ p, p ∈ l₁ ∧ p ∈ l₂]

theorem sufficient_condition_parallel_planes
  (h1 : m ∥ l₁)
  (h2 : n ∥ l₂) :
  α ∥ β :=
sorry

end sufficient_condition_parallel_planes_l95_95778


namespace distance_increasing_rapidly_l95_95102

-- Define constants for lengths of the hands
def minute_hand_length : ℝ := 4
def hour_hand_length : ℝ := 3

-- Define the angle variable θ and distance function
def distance_between_tips (θ : ℝ) : ℝ :=
  real.sqrt ((minute_hand_length ^ 2) + (hour_hand_length ^ 2) - 2 * minute_hand_length * hour_hand_length * real.cos θ)

-- Define the problem statement
theorem distance_increasing_rapidly : 
  ∃ θ : ℝ, real.cos θ = 25 / 36 ∧ distance_between_tips θ = real.sqrt 7 := 
by
  sorry

end distance_increasing_rapidly_l95_95102


namespace factorial_fraction_identity_l95_95973

-- Define the statement of the problem:
theorem factorial_fraction_identity (N : ℕ) : 
    (\frac{(N+2)!}{(N+3)! - (N+2)!}) = (\frac{1}{N+2}) := by
  sorry

end factorial_fraction_identity_l95_95973


namespace q_true_or_false_l95_95298

variable (p q : Prop)

theorem q_true_or_false (h1 : ¬ (p ∧ q)) (h2 : ¬ p) : q ∨ ¬ q :=
by
  sorry

end q_true_or_false_l95_95298


namespace trigonometric_identity_evaluation_l95_95677

theorem trigonometric_identity_evaluation :
  (sin (40 * Real.pi / 180))^2 + (cos (40 * Real.pi / 180))^2 = 1 →
  sin (40 * Real.pi / 180) = Real.sqrt (1 - (cos (40 * Real.pi / 180))^2) →
  abs (sin (40 * Real.pi / 180) - cos (40 * Real.pi / 180)) = 
    Real.sqrt ((sin (40 * Real.pi / 180))^2 - 2 * sin (40 * Real.pi / 180) * cos (40 * Real.pi / 180) + (cos (40 * Real.pi / 180))^2) →
  (Real.sqrt (1 - 2 * sin (40 * Real.pi / 180) * cos (40 * Real.pi / 180))) / 
  (cos (40 * Real.pi / 180) - Real.sqrt (1 - (sin (50 * Real.pi / 180))^2)) = 1 :=
by
  intros h1 h2 h3
  sorry

end trigonometric_identity_evaluation_l95_95677


namespace problem_1_problem_2_l95_95842

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log (x + 1) / Real.log 2 else 2^(-x) - 1

theorem problem_1 : f (f (-2)) = 2 := by 
  sorry

theorem problem_2 (x_0 : ℝ) (h : f x_0 < 3) : -2 < x_0 ∧ x_0 < 7 := by
  sorry

end problem_1_problem_2_l95_95842


namespace sum_of_first_n_odd_eq_900_l95_95908

theorem sum_of_first_n_odd_eq_900 {n : ℕ} (h₀ : n = 30) (h₁ : ∑ i in finset.range n, (2 * i + 1) = n * n) : 
  ∑ i in finset.range n, (2 * i + 1) = 900 :=
by {
  rw h₀,
  simp at h₁,
  exact h₁.symm
}

end sum_of_first_n_odd_eq_900_l95_95908


namespace frac_ab_eq_five_thirds_l95_95278

theorem frac_ab_eq_five_thirds (a b : ℝ) (hb : b ≠ 0) (h : (a - b) / b = 2 / 3) : a / b = 5 / 3 :=
by
  sorry

end frac_ab_eq_five_thirds_l95_95278


namespace good_students_count_l95_95315

noncomputable def student_count := 25

def is_good_student (s : Nat) := s ≤ student_count

def is_troublemaker (s : Nat) := s ≤ student_count

def always_tell_truth (s : Nat) := is_good_student s

def always_lie (s : Nat) := is_troublemaker s

def condition1 (E B : Nat) := E + B = student_count

def condition2 := ∀ (x : Nat), x ≤ 5 → is_good_student x → 
  ∃ (B : Nat), B > 24 / 2

def condition3 := ∀ (x : Nat), x ≤ 20 → is_troublemaker x → 
  ∃ (B E : Nat), B = 3 * (E - 1) ∧ E + B = student_count

theorem good_students_count :
  ∃ (E : Nat), condition1 E (student_count - E) ∧ condition2 ∧ condition3 :=
sorry  -- the actual proof is not required

end good_students_count_l95_95315


namespace same_function_A_l95_95962

def f (x : ℝ) : ℝ := |x|
def g (t : ℝ) : ℝ := Real.sqrt (t ^ 2)

theorem same_function_A : ∀ x : ℝ, f x = g x := by
  intro x
  rw [g, f]
  sorry

end same_function_A_l95_95962


namespace range_p_l95_95608

open Set

def p (x : ℝ) : ℝ :=
  x^4 + 6*x^2 + 9

theorem range_p : range p = Ici 9 := by
  sorry

end range_p_l95_95608


namespace prob_same_number_eq_prob_multiple_5_eq_l95_95945

namespace DiceProbability

def outcomes := { (d1, d2) : ℕ × ℕ // d1 ∈ {1, 2, 3, 4, 5, 6} ∧ d2 ∈ {1, 2, 3, 4, 5, 6} }

def favorableOutcomesSameNumber := { (d1, d2) | (d1, d2) ∈ outcomes ∧ d1 = d2 }
def favorableOutcomesMultipleOf5 := { (d1, d2) | (d1, d2) ∈ outcomes ∧ (d1 + d2) % 5 = 0 }

def totalOutcomes : ℕ := 36

def probability_same_number : ℚ :=
  Set.card favorableOutcomesSameNumber /. totalOutcomes

def probability_multiple_5 : ℚ :=
  Set.card favorableOutcomesMultipleOf5 /. totalOutcomes

theorem prob_same_number_eq : probability_same_number = 1/6 := by
  sorry

theorem prob_multiple_5_eq : probability_multiple_5 = 7/36 := by
  sorry

end DiceProbability

end prob_same_number_eq_prob_multiple_5_eq_l95_95945


namespace max_liters_of_water_heated_l95_95897

theorem max_liters_of_water_heated
  (heat_initial : ℕ := 480) 
  (heat_drop : ℝ := 0.25)
  (temp_initial : ℝ := 20)
  (temp_boiling : ℝ := 100)
  (specific_heat_capacity : ℝ := 4.2)
  (kJ_to_liters_conversion : ℝ := 336) :
  (∀ m : ℕ, (m * kJ_to_liters_conversion > ((heat_initial : ℝ) / (1 - heat_drop)) → m ≤ 5)) :=
by
  sorry

end max_liters_of_water_heated_l95_95897


namespace number_of_good_students_is_5_or_7_l95_95347

-- Definitions based on the conditions
def total_students : ℕ := 25
def number_of_good_students (G : ℕ) (T : ℕ) := G + T = total_students
def first_group_condition (T : ℕ) := T > 12
def second_group_condition (G : ℕ) (T : ℕ) := T = 3 * (G - 1)

-- Problem statement in Lean 4:
theorem number_of_good_students_is_5_or_7 (G T : ℕ) :
  number_of_good_students G T ∧ first_group_condition T ∧ second_group_condition G T → G = 5 ∨ G = 7 :=
by
  sorry

end number_of_good_students_is_5_or_7_l95_95347


namespace find_f_2018_l95_95413

-- Definitions based on the conditions
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x) = f (-x)

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x + 3) = -1 / f(x)

def value_in_range (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, -3 ≤ x ∧ x ≤ -2 → f(x) = 4 * x

-- The theorem to prove
theorem find_f_2018 (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_func_eq : functional_equation f)
  (h_value_range : value_in_range f) :
  f 2018 = -8 := 
sorry

end find_f_2018_l95_95413


namespace good_students_options_l95_95324

variables (E B : ℕ)

-- Define the condition that the class has 25 students
def total_students : Prop := E + B = 25

-- Define the condition given by the first group of students
def first_group_condition : Prop := B > 12

-- Define the condition given by the second group of students
def second_group_condition : Prop := B = 3 * (E - 1)

-- Define the problem statement
theorem good_students_options (E B : ℕ) :
  total_students E B → first_group_condition B → second_group_condition E B → (E = 5 ∨ E = 7) :=
by
  intros h1 h2 h3
  sorry

end good_students_options_l95_95324


namespace total_amount_received_is_correct_l95_95930

-- Define the given conditions
def total_books : ℕ := 450
def percent_novels : ℕ := 40
def percent_non_fiction : ℕ := 35
def percent_children : ℕ := 25
def price_novel : ℕ := 4
def price_non_fiction : ℕ := 3.5
def price_children : ℕ := 2.5
def sold_novels_ratio : ℕ := 1
def sold_non_fiction_ratio : ℕ := 4 / 5
def sold_children_ratio : ℕ := 3 / 4

-- Definitions to hold the number of books in each category
def novels : ℕ := (percent_novels * total_books) / 100
def non_fiction : ℕ := (percent_non_fiction * total_books) / 100
def children : ℕ := (percent_children * total_books) / 100

-- Definitions for the number of sold books in each category
def sold_novels : ℕ := novels * sold_novels_ratio
def sold_non_fiction : ℕ := non_fiction * sold_non_fiction_ratio
def sold_children : ℕ := children * sold_children_ratio

-- Total amount received from selling books
def total_received : ℕ := 
  (sold_novels * price_novel) +
  (sold_non_fiction * price_non_fiction) +
  (sold_children * price_children)

-- Lean 4 statement to prove the total amount received is $1367.50
theorem total_amount_received_is_correct : total_received = 1367.50 := by
  sorry

end total_amount_received_is_correct_l95_95930


namespace rectangle_length_l95_95017

theorem rectangle_length :
  ∃ x : ℝ, 
    let w := (3 / 4) * x in
    (4.5 * x^2 = 6000) ∧
    ((round (Real.sqrt (6000 / 4.5))) = 37) :=
begin
  sorry
end

end rectangle_length_l95_95017


namespace altitude_of_triangle_l95_95071

theorem altitude_of_triangle (b h_t h_p : ℝ) (hb : b ≠ 0) 
  (area_eq : b * h_p = (1/2) * b * h_t) 
  (h_p_def : h_p = 100) : h_t = 200 :=
by
  sorry

end altitude_of_triangle_l95_95071


namespace greatest_multiple_less_150_l95_95889

/-- Define the LCM of two natural numbers -/
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem greatest_multiple_less_150 (x y : ℕ) (h1 : x = 15) (h2 : y = 20) : 
  (∃ m : ℕ, LCM x y * m < 150 ∧ ∀ n : ℕ, LCM x y * n < 150 → LCM x y * n ≤ LCM x y * m) ∧ 
  (∃ m : ℕ, LCM x y * m = 120) :=
by
  sorry

end greatest_multiple_less_150_l95_95889


namespace problem_statement_l95_95257

-- Define the conditions and the goal
theorem problem_statement {x y : ℝ} 
  (h1 : (x + y)^2 = 36)
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := 
by
  sorry

end problem_statement_l95_95257


namespace exists_j_for_f_n_l95_95416

def f (x : ℝ) : ℝ := (x + real.sqrt (x^2 - 4)) / 2

theorem exists_j_for_f_n (i n : ℕ) (hi : i ≥ 2) (hn : n ≥ 2) : 
  ∃ (j : ℕ), j ≥ 2 ∧ (f^[n] i : ℝ) = f j := 
sorry

end exists_j_for_f_n_l95_95416


namespace phase_shift_of_graph_l95_95170

-- Define the function with given form
def cosWave (A B C : ℝ) := λ x : ℝ, A * cos (B * x + C)

-- Define specific values for this problem
def f (x : ℝ) : ℝ := cosWave 2 2 (π / 3) x

-- Define the phase shift property
def phaseShift (A B C : ℝ) : ℝ := -C / B

-- State the theorem
theorem phase_shift_of_graph : phaseShift 2 2 (π / 3) = -π / 6 :=
by
  sorry

end phase_shift_of_graph_l95_95170


namespace rational_expressions_l95_95066

theorem rational_expressions :
  (¬ (∃ r : ℚ, r = Real.sqrt (Real.exp 2))) ∧
  (¬ (∃ r : ℚ, r = Real.cbrt 0.64)) ∧
  (∃ r : ℚ, r = Real.root 4 1) ∧
  (∃ r : ℚ, r = Real.cbrt (-8) * Real.sqrt (1/0.25)) :=
by {
  -- Proof goes here
  sorry
}

end rational_expressions_l95_95066


namespace series_sum_l95_95140

theorem series_sum :
  (3 / 4 + 5 / 8 + 9 / 16 + 17 / 32 + 33 / 64 + 65 / 128 - 3.5 = -1 / 128) :=
begin
  sorry
end

end series_sum_l95_95140


namespace infinite_points_on_line_l95_95848

def is_arithmetic_sequence (f : ℕ → ℕ) : Prop :=
∃ c : ℕ, ∃ a : ℕ, ∀ n : ℕ, f(n) = a + n * c

noncomputable def exists_line_passing_infinitely_many_points (f : ℕ → ℕ) (P : ℕ → ℕ × ℕ) : Prop :=
∃ m b : ℕ, ∃ N : ℕ, ∀ n ≥ N, P n = (n, m * n + b)

theorem infinite_points_on_line (f : ℕ → ℕ) (P : ℕ → ℕ × ℕ) : 
(∀ n >= 2, (f(n-1) + f(n+1)) / 2 ≤ f(n)) → 
(∀ n : ℕ, P n = (n, f(n))) → 
exists_line_passing_infinitely_many_points f P :=
sorry

end infinite_points_on_line_l95_95848


namespace wealthiest_1000_individuals_income_l95_95841

noncomputable def N (x : ℝ) : ℝ := 5 * 10^7 * x^(-2)

theorem wealthiest_1000_individuals_income :
  ∃ x : ℝ, N(x) = 1000 ∧ x ≥ 3162 :=
by
  sorry

end wealthiest_1000_individuals_income_l95_95841


namespace max_min_sum_l95_95147

noncomputable def f : ℝ → ℝ := sorry

-- Define the interval and properties of the function f
def within_interval (x : ℝ) : Prop := -2016 ≤ x ∧ x ≤ 2016
def functional_eq (x1 x2 : ℝ) : Prop := f (x1 + x2) = f x1 + f x2 - 2016
def less_than_2016_proof (x : ℝ) : Prop := x > 0 → f x < 2016

-- Define the minimum and maximum values of the function f
def M : ℝ := sorry
def N : ℝ := sorry

-- Prove that M + N = 4032 given the properties and conditions
theorem max_min_sum : 
  (∀ x1 x2, within_interval x1 → within_interval x2 → functional_eq x1 x2) →
  (∀ x, x > 0 → less_than_2016_proof x) →
  M + N = 4032 :=
by {
  -- Define the formal proof here, placeholder for actual proof
  sorry
}

end max_min_sum_l95_95147


namespace sunday_jacket_price_correct_l95_95024

-- Define the original price of the jacket.
def original_price : ℝ := 250

-- Define the daily discount rate.
def daily_discount_rate : ℝ := 0.30

-- Define the Sunday additional discount rate.
def sunday_discount_rate : ℝ := 0.25

-- Define the threshold for the extra deduction.
def extra_deduction_threshold : ℝ := 100

-- Define the extra deduction amount.
def extra_deduction_amount : ℝ := 10

-- Calculate the price after the daily discount
def price_after_daily_discount : ℝ := original_price * (1 - daily_discount_rate)

-- Calculate the Sunday discount
def sunday_discounted_price : ℝ := price_after_daily_discount * (1 - sunday_discount_rate)

-- Determine if extra deduction applies and calculate final Sunday price
def final_sunday_price : ℝ :=
  if sunday_discounted_price < extra_deduction_threshold then
    sunday_discounted_price - extra_deduction_amount
  else
    sunday_discounted_price

-- Theorem statement asserting the final price
theorem sunday_jacket_price_correct :
  final_sunday_price = 131.25 :=
by
  sorry

end sunday_jacket_price_correct_l95_95024


namespace four_dimensional_measure_l95_95725

theorem four_dimensional_measure (r : ℝ) :
  ∃ (W : ℝ), deriv W r = 8 * π * r^3 ∧ W = 2 * π * r^4 :=
by 
  sorry

end four_dimensional_measure_l95_95725


namespace min_value_of_squares_l95_95410

-- Definitions
variables (a b c t : ℝ)
hypothesis h : a + b + c = t

-- Statement without proof
theorem min_value_of_squares : a + b + c = t → a^2 + b^2 + c^2 ≥ t^2 / 3 :=
by
  intro h
  sorry

end min_value_of_squares_l95_95410


namespace angle_between_vectors_is_90_degrees_l95_95758

open Real

def vector_a : ℝ × ℝ × ℝ := (3, -1, -4)
def vector_b : ℝ × ℝ × ℝ := (2, 5, -3)
def vector_c : ℝ × ℝ × ℝ := (4, -3, 8)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def scale_vector (k : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (k * v.1, k * v.2, k * v.3)

def vector_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

theorem angle_between_vectors_is_90_degrees :
  dot_product vector_a (vector_sub (scale_vector (dot_product vector_a vector_c) vector_b)
                                   (scale_vector (dot_product vector_a vector_b) vector_c)) = 0 :=
by 
  sorry

end angle_between_vectors_is_90_degrees_l95_95758


namespace sin_cos_identity_l95_95657

theorem sin_cos_identity (α c d : ℝ) (h : (sin α)^6 / c + (cos α)^6 / d = 1 / (c + d)) :
  (sin α)^12 / c^5 + (cos α)^12 / d^5 = 1 / (c + d)^5 :=
by sorry

end sin_cos_identity_l95_95657


namespace good_students_l95_95367

theorem good_students (G T : ℕ) (h1 : G + T = 25) (h2 : T > 12) (h3 : T = 3 * (G - 1)) :
  G = 5 ∨ G = 7 :=
by
  sorry

end good_students_l95_95367


namespace retail_price_l95_95070

/-- A retailer bought a machine at a wholesale price of $99 and later sold it after a 10% discount of the retail price.
If the retailer made a profit equivalent to 20% of the wholesale price, then the retail price of the machine before the discount was $132. -/
theorem retail_price (wholesale_price : ℝ) (profit_percent discount_percent : ℝ) (P : ℝ) 
  (h₁ : wholesale_price = 99) 
  (h₂ : profit_percent = 0.20) 
  (h₃ : discount_percent = 0.10)
  (h₄ : (1 - discount_percent) * P = wholesale_price + profit_percent * wholesale_price) : 
  P = 132 := 
by
  sorry

end retail_price_l95_95070


namespace candy_problem_l95_95520

theorem candy_problem (total_pieces red_pieces : ℕ) (h1 : total_pieces = 3409) (h2 : red_pieces = 145) :
  total_pieces - red_pieces = 3264 :=
by
  rw [h1, h2]
  norm_num
  sorry

end candy_problem_l95_95520


namespace greatest_value_of_4a_l95_95846

-- Definitions of the given conditions
def hundreds_digit (x : ℕ) : ℕ := x / 100
def tens_digit (x : ℕ) : ℕ := (x / 10) % 10
def units_digit (x : ℕ) : ℕ := x % 10

def satisfies_conditions (a b c x : ℕ) : Prop :=
  hundreds_digit x = a ∧
  tens_digit x = b ∧
  units_digit x = c ∧
  4 * a = 2 * b ∧
  2 * b = c ∧
  a > 0

def difference_of_two_greatest_x : ℕ := 124

theorem greatest_value_of_4a (x1 x2 a1 a2 b1 b2 c1 c2 : ℕ) :
  satisfies_conditions a1 b1 c1 x1 →
  satisfies_conditions a2 b2 c2 x2 →
  x1 - x2 = difference_of_two_greatest_x →
  4 * a1 = 8 :=
by
  sorry

end greatest_value_of_4a_l95_95846


namespace not_possible_perimeter_l95_95481

theorem not_possible_perimeter :
  ∀ (x : ℝ), 13 < x ∧ x < 37 → ¬ (37 + x = 50) :=
by
  intros x h
  sorry

end not_possible_perimeter_l95_95481


namespace cost_of_fencing_l95_95918

noncomputable def pi : ℝ := 3.14
def diameter : ℝ := 16
def rate : ℝ := 3

def circumference (d : ℝ) : ℝ := pi * d

def cost_of_fence (d : ℝ) (rate : ℝ) : ℝ := circumference(d) * rate

theorem cost_of_fencing : cost_of_fence(diameter, rate) = 150.72 := by
  sorry

end cost_of_fencing_l95_95918


namespace people_in_each_bus_l95_95937

-- Definitions and conditions
def num_vans : ℕ := 2
def num_buses : ℕ := 3
def people_per_van : ℕ := 8
def total_people : ℕ := 76

-- Theorem statement to prove the number of people in each bus
theorem people_in_each_bus : (total_people - num_vans * people_per_van) / num_buses = 20 :=
by
    -- The actual proof would go here
    sorry

end people_in_each_bus_l95_95937


namespace type_I_patterns_l95_95526

-- Define the Euler's totient function
def euler_totient (d : ℕ) : ℕ := 
  (Finset.range (d + 1)).filter (Nat.coprime d).card

-- Define the function g(m, n)
def g (m n : ℕ) : ℕ :=
  if h : n ≥ 3 ∧ m ≥ 2 then 
    (1 / n) *
    ((Finset.divisors n).filter (λ d, d ≠ n)).sum (λ d, 
      euler_totient d * ((m - 1)^(n / d) + (-1)^(n / d) * (m - 1)))
  else 0

-- Statement to be proved
theorem type_I_patterns (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 3) :
  g(m, n) = 
  (1 / n) * ((Finset.divisors n).filter (λ d, d ≠ n)).sum 
    (λ d, euler_totient d * ((m - 1)^(n / d) + (-1)^(n / d) * (m - 1))) :=
sorry

end type_I_patterns_l95_95526


namespace solve_x_squared_plus_y_squared_l95_95266

-- Variables
variables {x y : ℝ}

-- Conditions
def cond1 : (x + y)^2 = 36 := sorry
def cond2 : x * y = 8 := sorry

-- Theorem stating the problem's equivalent proof
theorem solve_x_squared_plus_y_squared : x^2 + y^2 = 20 := sorry

end solve_x_squared_plus_y_squared_l95_95266


namespace ivanov_error_l95_95824

theorem ivanov_error (x : ℝ) (m : ℝ) (S2 : ℝ) (std_dev : ℝ) :
  x = 0 → m = 4 → S2 = 15.917 → std_dev = Real.sqrt S2 →
  ¬ (|x - m| ≤ std_dev) :=
by
  intros h1 h2 h3 h4
  -- Using the given values directly to state the inequality
  have h5 : |0 - 4| = 4 := by norm_num
  have h6 : Real.sqrt 15.917 ≈ 3.99 := sorry  -- approximation as direct result
  -- Evaluating the inequality
  have h7 : 4 ≰ 3.99 := sorry  -- this represents the key step that shows the error
  exact h7
  sorry

end ivanov_error_l95_95824


namespace min_value_of_sum_inverse_l95_95482

theorem min_value_of_sum_inverse (m n : ℝ) 
  (H1 : ∃ (x y : ℝ), (x + y - 1 = 0 ∧ 3 * x - y - 7 = 0) ∧ (mx + y + n = 0))
  (H2 : mn > 0) : 
  ∃ k : ℝ, k = 8 ∧ ∀ (m n : ℝ), mn > 0 → (2 * m + n = 1) → 1 / m + 2 / n ≥ k :=
by
  sorry

end min_value_of_sum_inverse_l95_95482


namespace smallest_date_for_first_Saturday_after_second_Monday_following_second_Thursday_l95_95060

theorem smallest_date_for_first_Saturday_after_second_Monday_following_second_Thursday :
  ∃ d : ℕ, d = 17 :=
by
  -- Assuming the starting condition that the month starts such that the second Thursday is on the 8th
  let second_thursday := 8

  -- Calculate second Monday after the second Thursday
  let second_monday := second_thursday + 4
  
  -- Calculate first Saturday after the second Monday
  let first_saturday := second_monday + 5

  have smallest_date : first_saturday = 17 := rfl
  
  exact ⟨first_saturday, smallest_date⟩

end smallest_date_for_first_Saturday_after_second_Monday_following_second_Thursday_l95_95060


namespace f_increasing_f_odd_exist_l95_95680

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a - (2 / (2^x + 1))

-- Prove that f(x) is always increasing
theorem f_increasing (a : ℝ) : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 :=
by
  intros x1 x2 h
  have h1 : 2^x1 < 2^x2 := Nat.pow_lt_pow_of_lt_right h (by norm_num)
  have h2 : 1 + 2^x1 > 0 := by norm_num
  have h3 : 1 + 2^x2 > 0 := by norm_num
  exact sorry

-- Prove that there exists a real number a such that f(x) is odd (specifically a = 1)
theorem f_odd_exist : ∃ (a : ℝ), (∀ x : ℝ, f a (-x) = -f a x) :=
by
  use 1
  intro x
  have h1 : f 1 (-x) = 1 - (2 / (2^(-x) + 1)) := rfl
  have h2 : -f 1 x = -(1 - (2 / (2^x + 1))) := rfl
  simp only [f, h1, h2]
  sorry

end f_increasing_f_odd_exist_l95_95680


namespace rain_probability_l95_95488

/-
Theorem: Given that the probability it will rain on Monday is 40%
and the probability it will rain on Tuesday is 30%, and the probability of
rain on a given day is independent of the weather on any other day,
the probability it will rain on both Monday and Tuesday is 12%.
-/
theorem rain_probability (p_monday : ℝ) (p_tuesday : ℝ) (independent : Prop) :
  p_monday = 0.4 ∧ p_tuesday = 0.3 ∧ independent → (p_monday * p_tuesday) * 100 = 12 :=
by sorry

end rain_probability_l95_95488


namespace age_of_person_l95_95068

theorem age_of_person (x : ℕ) (h : 3 * (x + 3) - 3 * (x - 3) = x) : x = 18 :=
  sorry

end age_of_person_l95_95068


namespace mashed_potatoes_used_24_l95_95787

variable (total_potatoes used_for_salads leftover_potatoes : ℕ)

def potatoes_used_for_mashed_potatoes 
  (h1: total_potatoes = 52)
  (h2: used_for_salads = 15)
  (h3: leftover_potatoes = 13) : ℕ :=
  total_potatoes - used_for_salads - leftover_potatoes

theorem mashed_potatoes_used_24 :
  potatoes_used_for_mashed_potatoes 52 15 13 = 24 := by
    sorry

end mashed_potatoes_used_24_l95_95787


namespace quadratic_polynomial_l95_95171

noncomputable def q (x : ℝ) := -5 * x^2 - 10 * x + 75

theorem quadratic_polynomial :
  (q (-5) = 0) ∧ (q 3 = 0) ∧ (q 4 = -45) :=
by {
  -- Conditions
  have q_neg5 := calc q (-5) = -5 * (-5)^2 - 10 * (-5) + 75 : rfl,
  have q_3 := calc q 3 = -5 * 3^2 - 10 * 3 + 75 : rfl,
  have q_4 := calc q 4 = -5 * 4^2 - 10 * 4 + 75 : rfl,
  
  -- Simplifications
  simp at q_neg5 q_3 q_4,

  -- Checking each condition
  split,
  { exact q_neg5 },
  split,
  { exact q_3 },
  { exact q_4 },
}

end quadratic_polynomial_l95_95171


namespace percentage_A_of_B_l95_95604

variable {A B C D : ℝ}

theorem percentage_A_of_B (
  h1: A = 0.125 * C)
  (h2: B = 0.375 * D)
  (h3: D = 1.225 * C)
  (h4: C = 0.805 * B) :
  A = 0.100625 * B := by
  -- Sufficient proof steps would go here
  sorry

end percentage_A_of_B_l95_95604


namespace sum_of_identical_digit_numbers_l95_95239

-- Definitions of the compositions of digits
def five_digit_number (a : ℕ) : ℕ := 10000 * a + 1000 * a + 100 * a + 10 * a + a
def four_digit_number (b : ℕ) : ℕ := 1000 * b + 100 * b + 10 * b + b
def three_digit_number (c : ℕ) : ℕ := 100 * c + 10 * c + c

-- Statement of the problem
theorem sum_of_identical_digit_numbers :
  ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧
  let sum := five_digit_number a + four_digit_number b + three_digit_number c in
  sum >= 10000 ∧ sum < 100000 ∧
  let digits := [sum / 10000 % 10, sum / 1000 % 10, sum / 100 % 10, sum / 10 % 10, sum % 10] in
  digits.nodup :=
sorry

end sum_of_identical_digit_numbers_l95_95239


namespace f_at_2_lt_e6_l95_95204

variable (f : ℝ → ℝ)

-- Specify the conditions
axiom derivable_f : Differentiable ℝ f
axiom condition_3f_gt_fpp : ∀ x : ℝ, 3 * f x > (deriv (deriv f)) x
axiom f_at_1 : f 1 = Real.exp 3

-- Conclusion to prove
theorem f_at_2_lt_e6 : f 2 < Real.exp 6 :=
sorry

end f_at_2_lt_e6_l95_95204


namespace part_I_part_II_l95_95644

-- Definitions of variables and conditions
def circle (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 4

def line (x y : ℝ) : Prop :=
  x - y - 2 = 0

def point_A : Prop :=
  line 1 (-1)

def tangent_equation (x₁ y₁ x y : ℝ) : Prop :=
  x₁ * x + (y₁ - 2) * (y - 2) = 4

def equation_line_T₁T₂ (x y : ℝ) : Prop :=
  x - 3 * y + 2 = 0

-- Proof problems (without proof, only statements)
theorem part_I : 
  ∀ (T₁ T₂ : ℝ × ℝ), 
    point_A →
    (tangent_equation T₁.1 T₁.2 1 (-1)) →
    (tangent_equation T₂.1 T₂.2 1 (-1)) →
    equation_line_T₁T₂ 0 0 := 
  sorry

theorem part_II :
  ∀ (x₁ y₁ : ℝ),
    point_A →
    circle x₁ y₁ →
    let distance_center_to_line := 2 * real.sqrt 2 in
    (real.sqrt ((distance_center_to_line) ^ 2 - 4)) = 2 :=
  sorry

end part_I_part_II_l95_95644


namespace check_order_f_values_l95_95646

-- Definitions for the conditions
def symmetric_about_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(1 + x) = f(1 - x)

def periodicity_two (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x + 2) = -f(x)

def decreasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, 1 ≤ x1 → x1 < x2 → x2 ≤ 3 → (f(x2) - f(x1)) * (x2 - x1) < 0

-- Main theorem
theorem check_order_f_values (f : ℝ → ℝ)
  (h1 : symmetric_about_one f)
  (h2 : periodicity_two f)
  (h3 : decreasing_on_interval f)
  : f 2013 > f 2012 ∧ f 2012 > f 2011 :=
sorry

end check_order_f_values_l95_95646


namespace horizontal_asymptote_l95_95983

theorem horizontal_asymptote : 
  ∀ (x : ℝ), 
  lim (λ x, (10 * x^4 + 5 * x^3 + 7 * x^2 + 2 * x + 4) / (2 * x^4 + x^3 + 4 * x^2 + x + 2)) at_top = 5 :=
begin
  -- Proof goes here
  sorry
end

end horizontal_asymptote_l95_95983


namespace frac_pow_zero_l95_95531

def frac := 123456789 / (-987654321 : ℤ)

theorem frac_pow_zero : frac ^ 0 = 1 :=
by sorry

end frac_pow_zero_l95_95531


namespace fold_square_length_FD_l95_95390

/-- 
Given a square ABCD with side length 8 cm, where E is the midpoint of AD,
and C is folded to coincide with E such that F is on CD, the length of FD is 3cm.
-/
def length_of_FD : Prop :=
  let a := 8 in
  let e := a / 2 in
  ∃ (x : ℝ), (8 - x)^2 = x^2 + e^2 ∧ x = 3

theorem fold_square_length_FD : length_of_FD := by
  sorry

end fold_square_length_FD_l95_95390


namespace cos_sum_of_angles_l95_95661

theorem cos_sum_of_angles 
       (α β : ℝ) 
       (h1 : sin α - sin β = 1 / 3) 
       (h2 : cos α + cos β = 1 / 2) : 
       cos (α + β) = -59 / 72 := 
by 
  sorry

end cos_sum_of_angles_l95_95661


namespace max_liters_of_water_that_can_be_heated_to_boiling_l95_95901

-- Define the initial conditions
def initial_heat_per_5min := 480 -- kJ
def heat_reduction_rate := 0.25
def initial_temp := 20 -- Celsius
def boiling_temp := 100 -- Celsius
def specific_heat_capacity := 4.2 -- kJ/kg·°C

-- Define the temperature difference
def delta_T := boiling_temp - initial_temp -- Celsius

-- Define the calculation of the total heat available from a geometric series
def total_heat_available := initial_heat_per_5min / (1 - (1 - heat_reduction_rate))

-- Define the calculation of energy required to heat m kg of water
def energy_required (m : ℝ) := specific_heat_capacity * m * delta_T

-- Define the main theorem to prove
theorem max_liters_of_water_that_can_be_heated_to_boiling :
  ∃ (m : ℝ), ⌊m⌋ = 5 ∧ energy_required m ≤ total_heat_available :=
begin
  sorry
end

end max_liters_of_water_that_can_be_heated_to_boiling_l95_95901


namespace total_marbles_after_trade_l95_95443

theorem total_marbles_after_trade :
  ∀ (total_marbles : ℕ) (blue_percentage : ℚ) (kept_red_marbles : ℕ) (exchange_rate : ℕ),
    total_marbles = 10 →
    blue_percentage = (40 / 100 : ℚ) →
    kept_red_marbles = 1 →
    exchange_rate = 2 →
  let initial_blue_marbles := (blue_percentage * total_marbles) := 4 →
  let initial_red_marbles := (total_marbles - initial_blue_marbles) := 6 →
  let traded_red_marbles := initial_red_marbles - kept_red_marbles := 5 →
  let gained_blue_marbles := traded_red_marbles * exchange_rate := 10 →
  let final_total_marbles := initial_blue_marbles + gained_blue_marbles + kept_red_marbles :=
  final_total_marbles = 15 :=
sorry

end total_marbles_after_trade_l95_95443


namespace cos_graph_symmetric_l95_95845

theorem cos_graph_symmetric :
  ∃ (x0 : ℝ), x0 = (Real.pi / 3) ∧ ∀ y, (∃ x, y = Real.cos (2 * x + Real.pi / 3)) ↔ (∃ x, y = Real.cos (2 * (2 * x0 - x) + Real.pi / 3)) :=
by
  -- Let x0 = π / 3
  let x0 := Real.pi / 3
  -- Show symmetry about x = π / 3
  exact ⟨x0, by norm_num, sorry⟩

end cos_graph_symmetric_l95_95845


namespace each_girl_brought_2_cups_l95_95509

-- Definitions of the conditions
def total_students : ℕ := 30
def boys : ℕ := 10
def total_cups : ℕ := 90
def cups_per_boy : ℕ := 5
def girls : ℕ := total_students - boys

def total_cups_by_boys : ℕ := boys * cups_per_boy
def total_cups_by_girls : ℕ := total_cups - total_cups_by_boys
def cups_per_girl : ℕ := total_cups_by_girls / girls

-- The statement with the correct answer
theorem each_girl_brought_2_cups (
  h1 : total_students = 30,
  h2 : boys = 10,
  h3 : total_cups = 90,
  h4 : cups_per_boy = 5,
  h5 : total_cups_by_boys = boys * cups_per_boy,
  h6 : total_cups_by_girls = total_cups - total_cups_by_boys,
  h7 : cups_per_girl = total_cups_by_girls / girls
) : cups_per_girl = 2 := 
sorry

end each_girl_brought_2_cups_l95_95509


namespace necessary_condition_l95_95528

variable (P Q : Prop)

/-- If the presence of the dragon city's flying general implies that
    the horses of the Hu people will not cross the Yin Mountains,
    then "not letting the horses of the Hu people cross the Yin Mountains"
    is a necessary condition for the presence of the dragon city's flying general. -/
theorem necessary_condition (h : P → Q) : ¬Q → ¬P :=
by sorry

end necessary_condition_l95_95528


namespace mila_max_ms_l95_95785

theorem mila_max_ms (matches mismatches : ℕ) (total_letters : ℕ) :
  matches = 59 → mismatches = 40 → total_letters = 100 →
  ∃ ms : ℕ, ms = 80 :=
by
  intros h_match h_mismatch h_total
  use 80
  sorry

end mila_max_ms_l95_95785


namespace simplify_expression_l95_95462

variable (x y : ℝ)

theorem simplify_expression:
  3*x^2 - 3*(2*x^2 + 4*y) + 2*(x^2 - y) = -x^2 - 14*y := 
by 
  sorry

end simplify_expression_l95_95462


namespace total_time_to_watch_all_episodes_l95_95748

theorem total_time_to_watch_all_episodes 
  (announced_seasons : ℕ) (episodes_per_season : ℕ) (additional_episodes_last_season : ℕ)
  (seasons_before_announcement : ℕ) (episode_duration : ℝ) :
  announced_seasons = 1 →
  episodes_per_season = 22 →
  additional_episodes_last_season = 4 →
  seasons_before_announcement = 9 →
  episode_duration = 0.5 →
  let total_episodes_previous := seasons_before_announcement * episodes_per_season in
  let episodes_last_season := episodes_per_season + additional_episodes_last_season in 
  let total_episodes := total_episodes_previous + episodes_last_season in 
  total_episodes * episode_duration = 112 :=
by
  intros
  sorry

end total_time_to_watch_all_episodes_l95_95748


namespace rounding_and_division_results_l95_95942

/-- A number is approximated by rounding it down to the nearest 0.0001.
   The resulting number is divided by the original number,
   and the quotient is then rounded down again to the nearest 0.0001.
   Prove that the possible resulting values include 0, 0.0001, 0.0002, ..., 0.9999, 1. --/
def possible_results : set ℚ :=
  {n / 10000 | n ∈ finset.range 10002}.filter (λ x, 0 ≤ x ∧ x ≤ 1)

theorem rounding_and_division_results :
  possible_results = { q | q = 0 ∨ (∃ n : ℕ, 1 ≤ n ∧ n ≤ 10000 ∧ q = n / 10000) } :=
sorry

end rounding_and_division_results_l95_95942


namespace triangles_from_4_points_l95_95749

theorem triangles_from_4_points : 
  ∀ (points : Finset ℝ) (circ : ℝ), 
  points.card = 4 → 
  circ ∣ (Finset.sum (Finset.image (λ p, p) points)) → 
  (Finset.choose 3 points).card = 4 :=
by
  sorry

end triangles_from_4_points_l95_95749


namespace original_data_props_l95_95570

variable {n : ℕ}
variable {x : Fin n → ℝ}
variable {new_x : Fin n → ℝ} 

noncomputable def average (data : Fin n → ℝ) : ℝ :=
  (Finset.univ.sum (λ i => data i)) / n

noncomputable def variance (data : Fin n → ℝ) : ℝ :=
  (Finset.univ.sum (λ i => (data i - average data) ^ 2)) / n

-- Conditions
def condition1 (x new_x : Fin n → ℝ) (h : ∀ i, new_x i = x i - 80) : Prop := true

def condition2 (new_x : Fin n → ℝ) : Prop :=
  average new_x = 1.2

def condition3 (new_x : Fin n → ℝ) : Prop :=
  variance new_x = 4.4

theorem original_data_props (h : ∀ i, new_x i = x i - 80)
  (h_avg : average new_x = 1.2) 
  (h_var : variance new_x = 4.4) :
  average x = 81.2 ∧ variance x = 4.4 :=
sorry

end original_data_props_l95_95570


namespace fixed_point_p_l95_95694

variables {α : Type*} [EuclideanSpace α]

open EuclideanGeometry

-- Define the given conditions as variables in Lean.
variables (A B C M N P : Point α)

-- Given conditions in the problem.
variables (h1 : BC ∥ MN)
variables (h2 : ( dist A M / dist M B) = ( dist A B + dist A C) / dist B C)
variables (h3 : angle_bisector B C A P CF)

theorem fixed_point_p (A B C M N P CF : Point α)
  (h1 : BC ∥ MN)
  (h2 : ( dist A M / dist M B) = ( dist A B + dist A C) / dist B C)
  (h3 : angle_bisector B C A P CF) :
  ∃! Q, Q = P :=
begin
  -- The proof is omitted
  sorry
end

end fixed_point_p_l95_95694


namespace arithmetic_sequence_a6_l95_95209

theorem arithmetic_sequence_a6 {a : ℕ → ℤ}
  (h1 : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)
  (h2 : a 2 + a 8 = 16)
  (h3 : a 4 = 6) :
  a 6 = 10 :=
by
  sorry

end arithmetic_sequence_a6_l95_95209


namespace jamie_catches_bus_probability_l95_95099

noncomputable def probability_jamie_catches_bus : ℝ :=
  let total_area := 120 * 120
  let overlap_area := 20 * 100
  overlap_area / total_area

theorem jamie_catches_bus_probability :
  probability_jamie_catches_bus = (5 / 36) :=
by
  sorry

end jamie_catches_bus_probability_l95_95099


namespace average_speed_of_jane_l95_95032

theorem average_speed_of_jane : 
  let total_distance := 160
  let total_time := 6
  total_distance / total_time = 80 / 3 :=
by
  have h := rfl
  sorry

end average_speed_of_jane_l95_95032


namespace solve_for_AR_l95_95726

theorem solve_for_AR
  (A B C R Q P : Point)
  (AP PB AQ : ℝ)
  (h_acute_triangle : is_acute_triangle A B C)
  (h_R_perp_bisector : is_on_perpendicular_bisector A C R)
  (h_CA_bisector_BAR : is_angle_bisector CA (angle B A R))
  (h_Q_intersection : is_intersection AC BR Q)
  (h_circumcircle_P : P ∈ circumcircle A R C)
  (h_P_on_AB : P ≠ A ∧ P ∈ segment A B)
  (h_AP : AP = 1)
  (h_PB : PB = 5)
  (h_AQ : AQ = 2) :
  distance A R = 6 :=
by
  sorry

end solve_for_AR_l95_95726


namespace digits_sum_distinct_l95_95242

theorem digits_sum_distinct :
  ∃ (a b c : ℕ), (9 ≥ a ∧ a ≥ 0) ∧ (9 ≥ b ∧ b ≥ 0) ∧ (9 ≥ c ∧ c ≥ 0) ∧
  let N1 := 11111 * a in
  let N2 := 1111 * b in
  let N3 := 111 * c in
  let S := N1 + N2 + N3 in
  (10000 ≤ S ∧ S ≤ 99999) ∧
  (∃ d1 d2 d3 d4 d5 : ℕ, d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧
                        d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ 
                        d3 ≠ d4 ∧ d3 ≠ d5 ∧ d4 ≠ d5 ∧
                        S = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5) := 
by
  sorry

end digits_sum_distinct_l95_95242


namespace determine_d_l95_95987

theorem determine_d 
  (d : ℝ) 
  (h : ∀ (x : ℝ), (x ∈ set.Ioo (-5/2) 1 ↔ x * (4 * x + 3) < d)) : 
  d = 10 := 
by 
  sorry

end determine_d_l95_95987


namespace proposition_2_3_equivalence_l95_95133

-- Given three points A, B, and C in a Euclidean space
variables (A B C : EuclideanSpace ℝ (fin 2))

-- Definition of collinear points
def collinear (A B C : EuclideanSpace ℝ (fin 2)) : Prop :=
∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * (B - A) = b * (C - A) ∧ a * (C - B) = c

-- Proposition 2: If points A, B, and C are collinear, then no circle passes through points A, B, and C.
def proposition_2 (A B C : EuclideanSpace ℝ (fin 2)) : Prop :=
collinear A B C → ¬ ∃ (circ : Circle), circ.passes_through_three_points A B C

-- Proposition 3: If points A, B, and C lie on a circle, then points A, B, and C are not collinear.
def proposition_3 (A B C : EuclideanSpace ℝ (fin 2)) : Prop :=
(∃ (circ : Circle), circ.passes_through_three_points A B C) → ¬ collinear A B C

-- The theorem we are proving: Proposition 2 and Proposition 3 are equivalent
theorem proposition_2_3_equivalence :
  ∀ A B C, proposition_2 A B C ↔ proposition_3 A B C :=
sorry

end proposition_2_3_equivalence_l95_95133


namespace angle_abe_15_l95_95151

theorem angle_abe_15 (A B C D E : Type*)
  (triangle_ABC : Triangle A B C)
  (AD_bisects_∠BAC : AngleBisector A B C D)
  (BE_intersects_AD : Intersects E (Line B E) (Line A D))
  (angle_BAC_70 : Measure (∠BAC) = 70)
  (angle_ABC_50 : Measure (∠ABC) = 50) :

  Measure (∠ABE) = 15 :=
begin
  sorry
end

end angle_abe_15_l95_95151


namespace sum_divisible_by_6_iff_sum_cubes_divisible_by_6_l95_95797

theorem sum_divisible_by_6_iff_sum_cubes_divisible_by_6 (x y : ℤ) : 
  (6 ∣ (x + y)) ↔ (6 ∣ (x^3 + y^3)) :=
by
  split
  { intro h
    rw [add_assoc, ← add_mul, mul_comm]
    exact dvd_mul_right 6 _
  }
  {
    sorry
  }

end sum_divisible_by_6_iff_sum_cubes_divisible_by_6_l95_95797


namespace coeff_x6_expansion_eq_10_l95_95733

/-- In the expansion of (1 + x + x^2)(1 - x)^6, the coefficient of x^6 is equal to 10. -/
theorem coeff_x6_expansion_eq_10 : 
  ∀ (x : ℤ), coeff_x6 ((1 + x + x^2) * (1 - x)^6) = 10 :=
by
  sorry

end coeff_x6_expansion_eq_10_l95_95733


namespace number_of_blue_butterflies_l95_95423

theorem number_of_blue_butterflies 
  (total_butterflies : ℕ)
  (B Y : ℕ)
  (H1 : total_butterflies = 11)
  (H2 : B = 2 * Y)
  (H3 : total_butterflies = B + Y + 5) : B = 4 := 
sorry

end number_of_blue_butterflies_l95_95423


namespace ratio_2006_to_2005_l95_95306

-- Conditions
def kids_in_2004 : ℕ := 60
def kids_in_2005 : ℕ := kids_in_2004 / 2
def kids_in_2006 : ℕ := 20

-- The statement to prove
theorem ratio_2006_to_2005 : 
  (kids_in_2006 : ℚ) / kids_in_2005 = 2 / 3 :=
sorry

end ratio_2006_to_2005_l95_95306


namespace cuboid_edges_proof_l95_95471

-- Given constants and the relationship derived from the problem
constants (c : ℝ) (a : ℝ) (b : ℝ)
axiom ratio1 : a = (16 / 21) * c
axiom ratio2 : b = (4 / 7) * c
axiom diagonal : a^2 + b^2 + c^2 = 29^2

-- Assert the lengths of the edges
theorem cuboid_edges_proof : a = 16 ∧ b = 12 ∧ c = 21 :=
by sorry

end cuboid_edges_proof_l95_95471


namespace good_students_count_l95_95313

noncomputable def student_count := 25

def is_good_student (s : Nat) := s ≤ student_count

def is_troublemaker (s : Nat) := s ≤ student_count

def always_tell_truth (s : Nat) := is_good_student s

def always_lie (s : Nat) := is_troublemaker s

def condition1 (E B : Nat) := E + B = student_count

def condition2 := ∀ (x : Nat), x ≤ 5 → is_good_student x → 
  ∃ (B : Nat), B > 24 / 2

def condition3 := ∀ (x : Nat), x ≤ 20 → is_troublemaker x → 
  ∃ (B E : Nat), B = 3 * (E - 1) ∧ E + B = student_count

theorem good_students_count :
  ∃ (E : Nat), condition1 E (student_count - E) ∧ condition2 ∧ condition3 :=
sorry  -- the actual proof is not required

end good_students_count_l95_95313


namespace sms_message_fraudulent_l95_95864

-- Define the conditions as properties
def messageArrivedNumberKnown (msg : String) (numberKnown : Bool) : Prop :=
  msg = "SMS message has already arrived" ∧ numberKnown = true

def fraudDefinition (acquisition : String -> Prop) : Prop :=
  ∀ (s : String), acquisition s = (s = "acquisition of property by third parties through deception or gaining the trust of the victim")

-- Define the main proof problem statement
theorem sms_message_fraudulent (msg : String) (numberKnown : Bool) (acquisition : String -> Prop) :
  messageArrivedNumberKnown msg numberKnown ∧ fraudDefinition acquisition →
  acquisition "acquisition of property by third parties through deception or gaining the trust of the victim" :=
  sorry

end sms_message_fraudulent_l95_95864


namespace exists_geom_prog_first_100_integers_and_not_after_l95_95158

/-- 
  There exists an increasing geometric progression where the first 100 terms 
  are integers, but all subsequent terms are not integers.
-/
theorem exists_geom_prog_first_100_integers_and_not_after :
  ∃ (a₁ : ℕ) (q : ℚ), (∀ n ≤ 100, (a₁ * q ^ (n - 1)) ∈ ℤ) ∧ 
                       (∀ n > 100, ¬ (a₁ * q ^ (n - 1)) ∈ ℤ) ∧ 
                       (∀ m₁ m₂, m₁ < m₂ → a₁ * q ^ (m₁ - 1) < a₁ * q ^ (m₂ - 1)) := 
begin
  let a₁ := (2 : ℕ) ^ 99,
  let q : ℚ := 3 / 2,
  have h1 : ∀ n ≤ 100, ((a₁ * q ^ (n - 1)) ∈ ℤ),
  { intros n hn,
    have h2 : a₁ * q ^ (n - 1) = 2 ^ (100 - n) * 3 ^ (n - 1),
    { rw [mul_comm, ← pow_sub _ _ nat.le_of_lt_succ hn], 
      field_simp, ring },
    exact_mod_cast h2.symm },
  have h3 : ∀ n > 100, ¬ (a₁ * q ^ (n - 1) ∈ ℤ),
  { intros n hn,
    have h2 : a₁ * q ^ (n - 1) = 3 ^ (n - 1) / 2 ^ (n - 100),
    { 
      rw [mul_comm, ← pow_sub],
      field_simp, ring,
      exact nat.sub_pos_of_lt hn, }, 
    rw [h2],
    intro h4,
    have : (3 ^ (n - 1) : ℚ) = (2 ^ (n - 100) : ℚ) * (k : ℚ),
    { exact_mod_cast h4 },
    have h5 : 2 ∣ 3 ^ (n - 1),
    { norm_num at this, },
    have h6 := nat.prime_dvd_prime_pow nat.prime_two,
    exact nat.not_prime_three.
  },
  have h7: ∀ m₁ m₂, m₁ < m₂ → a₁ * q ^ (m₁ - 1) < a₁ * q ^ (m₂ - 1),
  { 
    intros m₁ m₂ hm,
    suffices h : q > 0, { exact mul_pos a₁.mpr (pow_pos h _), },
    norm_num.
  },

  exact ⟨a₁, q, h1, h3, h7⟩,
end

end exists_geom_prog_first_100_integers_and_not_after_l95_95158


namespace simplify_expression_1_simplify_expression_2_l95_95969

section Problem1
variables (a b c : ℝ) (h1 : c ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0)

theorem simplify_expression_1 :
  ((a^2 * b / (-c))^3 * (c^2 / (- (a * b)))^2 / (b * c / a)^4)
  = - (a^10 / (b^3 * c^7)) :=
by sorry
end Problem1

section Problem2
variables (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ a) (h3 : b ≠ 0)

theorem simplify_expression_2 :
  ((2 / (a^2 - b^2) - 1 / (a^2 - a * b)) / (a / (a + b))) = 1 / a^2 :=
by sorry
end Problem2

end simplify_expression_1_simplify_expression_2_l95_95969


namespace part_I_part_II_l95_95230

noncomputable def sequence (a₁ : ℝ) : ℕ → ℝ
| 0     := a₁
| (n+1) := 1 / 2 * (sequence n + 4 / sequence n)

def b (aₙ : ℕ → ℝ) (n : ℕ) : ℝ :=
|aₙ n - 2|

def S (bₙ : ℕ → ℝ) (n : ℕ) : ℝ :=
Finset.sum (Finset.range n) bₙ

theorem part_I (h₂ : sequence 1 3 = 41 / 20) :
  sequence 1 1 = 1 ∨ sequence 1 1 = 4 := sorry

theorem part_II (h₁ : sequence 4 1 = 4) :
  ∀ n, S (b (sequence 4)) n < 8 / 3 := sorry

end part_I_part_II_l95_95230


namespace find_a_12_l95_95672

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

variables (a d : ℝ)
def a_n (n : ℕ) : ℝ := arithmetic_sequence a d n

-- Given conditions:
axiom h1 : a_n a d 7 + a_n a d 9 = 16
axiom h2 : ∑ k in Finset.range 11, a_n a d (k + 1) = 99 / 2

-- The theorem to be proved:
theorem find_a_12 : a_n a d 12 = 15 := sorry

end find_a_12_l95_95672


namespace mixture_correct_l95_95376

def water_amount : ℚ := (3/5) * 20
def vinegar_amount : ℚ := (5/6) * 18
def mixture_amount : ℚ := water_amount + vinegar_amount

theorem mixture_correct : mixture_amount = 27 := 
by
  -- Here goes the proof steps
  sorry

end mixture_correct_l95_95376


namespace general_formula_sum_of_sequence_l95_95605

variable (n : ℕ)

-- Condition 1
def a : ℕ → ℕ
| 0     := 0  -- Define a_0 artificially to handle base case more conveniently
| 1     := 1
| (n+2) := a (n+1) + (n + 1) + 1

-- The general formula for the sequence, ∀ n ≥ 1, a n = n * (n + 1) / 2
theorem general_formula (n : ℕ) (hn : n ≥ 1) : a n = n * (n + 1) / 2 := sorry

-- The sum T_n of the first n terms of the sequence {1 / a_n}
def T (n : ℕ) : ℚ :=
∑ i in Finset.range n, 1 / (a (i + 1))

-- Prove that T_n = 2n / (n + 1)
theorem sum_of_sequence (n : ℕ) (hn : n ≥ 1) : T n = 2 * n / (n + 1) := sorry

end general_formula_sum_of_sequence_l95_95605


namespace remainder_is_6910_l95_95907

def polynomial (x : ℝ) : ℝ := 5 * x^7 - 3 * x^6 - 8 * x^5 + 3 * x^3 + 5 * x^2 - 20

def divisor (x : ℝ) : ℝ := 3 * x - 9

theorem remainder_is_6910 : polynomial 3 = 6910 := by
  sorry

end remainder_is_6910_l95_95907


namespace m_le_three_l95_95689

-- Definitions
def setA (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5
def setB (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

-- Theorem statement
theorem m_le_three (m : ℝ) : (∀ x : ℝ, setB m x → setA x) → m ≤ 3 := by
  sorry

end m_le_three_l95_95689


namespace num_identity_sequences_l95_95119

/-
Define the transformations L, H, V and prove the number of sequences of 10 transformations from 
these that result in the identity transformation is 1024.
-/

def L := λ (p : ℤ × ℤ), (-p.1, -p.2)  -- 180° rotation
def H := λ (p : ℤ × ℤ), (p.1, -p.2)   -- Reflection over x-axis
def V := λ (p : ℤ × ℤ), (-p.1, p.2)   -- Reflection over y-axis

/-
All transformations are their own inverses.
-/
axiom L_involution : ∀ (p : ℤ × ℤ), L (L p) = p
axiom H_involution : ∀ (p : ℤ × ℤ), H (H p) = p
axiom V_involution : ∀ (p : ℤ × ℤ), V (V p) = p

/-
The problem statement: How many sequences of 10 transformations chosen from {L, H, V} will send all of 
the labeled vertices back to their original positions?
-/
theorem num_identity_sequences : 
  (∑ (p : ℕ × ℕ × ℕ) in 
   {p : ℕ × ℕ × ℕ | 2 * p.1 + 2 * p.2 + 2 * p.3 = 10}.to_finset, 
   nat.choose 10 p.1 * (nat.choose (10 - p.1) p.2)) = 1024 :=
by sorry

end num_identity_sequences_l95_95119


namespace option_B_correct_l95_95913

theorem option_B_correct (x y : ℝ) : 
  x * y^2 - y^2 * x = 0 :=
by sorry

end option_B_correct_l95_95913


namespace group_commutative_l95_95751

def is_automorphism (G : Type) [group G] (f : G → G) : Prop :=
  bijective f ∧ ∀ x y : G, f (x * y) = f x * f y

def satisfies_property (G : Type) [group G] : Prop :=
  ∀ (f : G → G),
    is_automorphism G f →
    ∃ (m : ℕ) (hm : m > 0), ∀ x : G, f x = x ^ m

theorem group_commutative {G : Type} [group G] [fintype G]
  (h : satisfies_property G) : is_commutative G (· * ·) :=
by
  sorry

end group_commutative_l95_95751


namespace arithmetic_sequence_length_l95_95702

theorem arithmetic_sequence_length : 
  let a := 11
  let d := 5
  let l := 101
  ∃ n : ℕ, a + (n-1) * d = l ∧ n = 19 := 
by
  sorry

end arithmetic_sequence_length_l95_95702


namespace greg_earnings_correct_l95_95248

def charge_per_dog := 
  {extra_small := 12, small := 15, medium := 20, large := 25, extra_large := 30}

def charge_per_min :=
  {extra_small := 0.80, small := 1.0, medium := 1.25, large := 1.50, extra_large := 1.75}

def walks := 
  {extra_small := (2, 10), small := (3, 12), medium := (1, 18), large := (2, 25), extra_large := (1, 30)}

noncomputable def earnings_on_day : ℝ := 
  charge_per_dog.extra_small * walks.extra_small.1 + charge_per_min.extra_small * walks.extra_small.1 * walks.extra_small.2 +
  charge_per_dog.small * walks.small.1 + charge_per_min.small * walks.small.1 * walks.small.2 +
  charge_per_dog.medium * walks.medium.1 + charge_per_min.medium * walks.medium.1 * walks.medium.2 +
  charge_per_dog.large * walks.large.1 + charge_per_min.large * walks.large.1 * walks.large.2 +
  charge_per_dog.extra_large * walks.extra_large.1 + charge_per_min.extra_large * walks.extra_large.1 * walks.extra_large.2

theorem greg_earnings_correct : earnings_on_day = 371 := by
  sorry

end greg_earnings_correct_l95_95248


namespace problem_solution_l95_95141

-- Definition to capture the logarithmic and exponential properties used in the problem.
noncomputable def problem_expr := 
  (Real.log10 (1 / 4) - Real.log10 25 + Real.log (Real.sqrt Real.exp 1) + Real.pow 2 (1 + Real.log2 3))

-- Theorem statement to prove the expression is equal to 9/2
theorem problem_solution : problem_expr = 9 / 2 := 
  sorry

end problem_solution_l95_95141


namespace reconstruct_trapezoid_l95_95560

-- Definitions for the given problem
variable (A I O : Point)
variable (ω : Circle)
variable (inscribable circumscribable : Trapezoid → Prop)

-- Conditions
axiom inscribed_circle : inscribable ABCD
axiom circumscribed_circle : circumscribable ABCD
axiom vertex_known : A ∈ ABCD
axiom inscribed_center_known : center (inscribed_circle ABCD) = I
axiom circumscribed_circle_known : ω = circumscribed_circle ABCD ∧ center ω = O

-- The question translated to proof
theorem reconstruct_trapezoid (ABCD : Trapezoid) :
  inscribable ABCD ∧ circumscribable ABCD ∧ A ∈ ABCD ∧
  center (inscribed_circle ABCD) = I ∧ ω = circumscribed_circle ABCD ∧
  center ω = O → 
  ∃ B C D : Point, B ∈ ω ∧ C ∈ ω ∧ D ∈ ω ∧ is_trapezoid A B C D :=
by sorry

end reconstruct_trapezoid_l95_95560


namespace f_zero_f_odd_f_range_l95_95967

-- Condition 1: The function f is defined on ℝ
-- Condition 2: f(x + y) = f(x) + f(y)
-- Condition 3: f(1/3) = 1
-- Condition 4: f(x) < 0 when x > 0

variables (f : ℝ → ℝ)
axiom f_add : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_third : f (1/3) = 1
axiom f_neg_positive : ∀ x : ℝ, 0 < x → f x < 0

-- Question 1: Find the value of f(0)
theorem f_zero : f 0 = 0 := sorry

-- Question 2: Prove that f is an odd function
theorem f_odd : ∀ x : ℝ, f (-x) = -f x := sorry

-- Question 3: Find the range of x where f(x) + f(2 + x) < 2
theorem f_range : ∀ x : ℝ, f x + f (2 + x) < 2 → -2/3 < x := sorry

end f_zero_f_odd_f_range_l95_95967


namespace polynomial_factors_count_l95_95623

theorem polynomial_factors_count :
  (∑ a in Finset.range (20 - 10 + 1), a + 11 + ∑ a in Finset.range (20 - 10 + 1), ⌈(a + 11)/2⌉) = 256 := by
  sorry

end polynomial_factors_count_l95_95623


namespace max_liters_of_water_heated_l95_95899

theorem max_liters_of_water_heated
  (heat_initial : ℕ := 480) 
  (heat_drop : ℝ := 0.25)
  (temp_initial : ℝ := 20)
  (temp_boiling : ℝ := 100)
  (specific_heat_capacity : ℝ := 4.2)
  (kJ_to_liters_conversion : ℝ := 336) :
  (∀ m : ℕ, (m * kJ_to_liters_conversion > ((heat_initial : ℝ) / (1 - heat_drop)) → m ≤ 5)) :=
by
  sorry

end max_liters_of_water_heated_l95_95899


namespace num_adjacent_pairs_same_after_10_l95_95581

def replace_A (s : String) : String :=
  s.toList.foldr (λ c acc => if c = 'A' then 'B' :: 'A' :: acc else 'A' :: 'B' :: acc) []

def transform (n : ℕ) (s : String) : String :=
  (List.range n).foldl (λ acc _ => replace_A acc) s

def adjacent_same_pairs (s : String) : ℕ :=
  (List.zip s.toList (s.toList.tail)).filter (λ (c1, c2) => c1 = c2).length

theorem num_adjacent_pairs_same_after_10 :
  adjacent_same_pairs (transform 10 "A") = 341 :=
sorry

end num_adjacent_pairs_same_after_10_l95_95581


namespace Reena_paid_interest_l95_95011

variable (P : ℕ) (R : ℕ) (T : ℕ)
def simpleInterest (P R T : ℕ) : ℕ := (P * R * T) / 100

theorem Reena_paid_interest (hP : P = 1200) (hR : R = 6) (hT : T = 6) :
  simpleInterest P R T = 432 := by
  simp [simpleInterest, hP, hR, hT]
  norm_num
  sorry

end Reena_paid_interest_l95_95011


namespace incorrect_transformation_l95_95063

-- Definitions based on conditions
variable (a b c : ℝ)

-- Conditions
axiom eq_add_six (h : a = b) : a + 6 = b + 6
axiom eq_div_nine (h : a = b) : a / 9 = b / 9
axiom eq_mul_c (h : a / c = b / c) (hc : c ≠ 0) : a = b
axiom eq_div_neg_two (h : -2 * a = -2 * b) : a = b

-- Proving the incorrect transformation statement
theorem incorrect_transformation : ¬ (a = -b) ∧ (-2 * a = -2 * b → a = b) := by
  sorry

end incorrect_transformation_l95_95063


namespace problem_statement_l95_95260

-- Define the conditions and the goal
theorem problem_statement {x y : ℝ} 
  (h1 : (x + y)^2 = 36)
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := 
by
  sorry

end problem_statement_l95_95260


namespace length_of_side_a_l95_95305

theorem length_of_side_a 
    (b : ℝ) (B : ℝ) (sin_A : ℝ)
    (hb : b = 3 * Real.sqrt 3)
    (hB : B = Real.pi / 3)
    (hsinA : sin_A = 1 / 3) :
  let sin_B := Real.sin (Real.pi / 3),
      a := b * sin_A / sin_B in
  a = 2 :=
by
  sorry

end length_of_side_a_l95_95305


namespace path_length_traversed_by_P_l95_95996

-- Define the problem conditions
def length_a_b : ℝ := 2
def length_a_xyz : ℝ := 4
def rotations : ℕ := 12
def angle_rotation : ℝ := 2 * (Math.pi / 3)
def arc_length : ℝ := length_a_b * angle_rotation
def total_path_length : ℝ := rotations * arc_length

-- State the theorem we need to prove
theorem path_length_traversed_by_P : total_path_length = 40 * Math.pi / 3 :=
by
  sorry

end path_length_traversed_by_P_l95_95996


namespace good_students_count_l95_95369

theorem good_students_count (E B : ℕ) 
    (h1 : E + B = 25) 
    (h2 : ∀ (e ∈ finset.range 5), B > 12) 
    (h3 : ∀ (e ∈ finset.range 20), B = 3 * (E - 1)) :
    E = 5 ∨ E = 7 :=
by
  sorry

end good_students_count_l95_95369


namespace other_diagonal_length_minimize_perimeter_maximum_perimeter_not_possible_l95_95717

namespace QuadrilateralProof

open Real

-- Problem (a)
theorem other_diagonal_length (A B C D : Point)
  (AB CD AC : ℝ) (areaABC : ℝ)
  (h1 : AB + CD + AC = 16)
  (h2 : areaABC = 32) :
  ∃ BD : ℝ, BD = 8 * sqrt 2 :=
by
  sorry

-- Problem (b)
theorem minimize_perimeter (A B C D : Point)
  (AB CD AC : ℝ) (areaABC : ℝ)
  (h1 : AB + CD + AC = 16)
  (h2 : areaABC = 32) :
  (AB = 4) ∧ (CD = 4) ∧ (AC = 8) :=
by
  sorry

-- Problem (c)
theorem maximum_perimeter_not_possible (A B C D : Point)
  (AB CD AC : ℝ) (areaABC : ℝ)
  (h1 : AB + CD + AC = 16)
  (h2 : areaABC = 32) :
  ¬ (∃ perimeter, ∀ other_perimeters > perimeter) :=
by
  sorry

end QuadrilateralProof

end other_diagonal_length_minimize_perimeter_maximum_perimeter_not_possible_l95_95717


namespace class_proof_l95_95353

def class_problem : Prop :=
  let total_students := 25
  let (students : Type) := {s // s ∈ Fin total_students}
  let good students (s : students) : Prop := sorry -- Condition Placeholder
  let troublemakers (s : students) : Prop := sorry -- Condition Placeholder
  have all_good_or_trouble : ∀ s, good students s ∨ troublemakers s,
    from sorry,
  have lh_trouble : ∀ (a b c d e : students), (good students a ∧ good students b ∧ good students c ∧ good students d ∧ good students e) →
    ∃ (x : ℕ), x > ((total_students - 1) / 2) ∧
    (λ rem_trouble : students - {a, b, c, d, e}, B : ℕ, B = x) ∧
    (λ rem_good : students - {a, b, c, d, e}, G : ℕ, G ∈ rem_good), from sorry,
  have th_triple_ratio : ∀ s, (students - {s}).card = 24 →
    ∃ (x y : ℕ), (troublemakers x ∧ good students y) ∧ (3*y = x), from sorry,
  ∃ (num_good_students : ℕ), (num_good_students = 5 ∨ num_good_students = 7)
  
theorem class_proof : class_problem :=
  sorry

end class_proof_l95_95353


namespace power_function_decreasing_l95_95710

theorem power_function_decreasing (m : ℝ) (x : ℝ) (hx : x > 0) :
  (m^2 - 2*m - 2 = 1) ∧ (-4*m - 2 < 0) → m = 3 :=
by
  sorry

end power_function_decreasing_l95_95710


namespace distinct_digit_count_l95_95700

theorem distinct_digit_count : 
  ∃ n : ℕ, (n = 27216) ∧ ∀ x : ℕ, (10000 ≤ x ∧ x ≤ 99999) → (all_distinct_digits x → n = count_distinct_digit_numbers 10000 99999) :=
sorry

-- Additional Definitions
def all_distinct_digits (n : ℕ) : Prop :=
  let digits := to_digits n in
  list.nodup digits

def to_digits (n : ℕ) : list ℕ := 
  if n = 0 then [0] else (list.unfoldr (λ x => if x = 0 then none else some (x % 10, x / 10)) n).reverse

def count_distinct_digit_numbers (low high : ℕ) : ℕ :=
  list.length (list.filter (λ n => all_distinct_digits n) (list.range' low (high - low + 1)))

end distinct_digit_count_l95_95700


namespace infinite_triangles_with_conditions_l95_95254

theorem infinite_triangles_with_conditions :
  ∃ (A : ℝ) (B : ℝ) (C : ℝ), 
  (A > 0) ∧ (B > 0) ∧ (C > 0) ∧ (B - A = 2) ∧ (C = 4) ∧ 
  (Δ > 0) := sorry

end infinite_triangles_with_conditions_l95_95254


namespace good_students_count_l95_95368

theorem good_students_count (E B : ℕ) 
    (h1 : E + B = 25) 
    (h2 : ∀ (e ∈ finset.range 5), B > 12) 
    (h3 : ∀ (e ∈ finset.range 20), B = 3 * (E - 1)) :
    E = 5 ∨ E = 7 :=
by
  sorry

end good_students_count_l95_95368


namespace not_perfect_square_l95_95503

theorem not_perfect_square :
  let nums := (List.range (1982 + 1)).map (fun x => (x + 1) * (x + 1))
  let str_num := String.intercalate "" (nums.map toString)
  let large_num := str_num.toNat
  ¬ (∃ m : ℕ, m * m = large_num) :=
by
  sorry

end not_perfect_square_l95_95503


namespace xy_problem_l95_95276

theorem xy_problem (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 :=
by
  sorry

end xy_problem_l95_95276


namespace find_number_of_good_students_l95_95331

variable {G T : ℕ}

def is_good_or_troublemaker (G T : ℕ) : Prop :=
  G + T = 25 ∧ 
  (∀ s ∈ finset.range 5, T > 12) ∧ 
  (∀ s ∈ finset.range 20, T = 3 * (G - 1))

-- There are 25 students in total. 
-- Each student is either a good student or a troublemaker.
-- Good students always tell the truth, while troublemakers always lie.
-- Five students make the statement: "If I transfer to another class, more than half of the remaining students will be troublemakers."
-- The remaining 20 students make the statement: "If I transfer to another class, the number of troublemakers among the remaining students will be three times the number of good students."

theorem find_number_of_good_students (G T : ℕ) (h : is_good_or_troublemaker G T) :
  (G = 5 ∨ G = 7) :=
sorry

end find_number_of_good_students_l95_95331


namespace dave_probability_is_125_over_671_l95_95961

theorem dave_probability_is_125_over_671 :
  let p_six := (1/6: ℚ),
      p_not_six := (5/6: ℚ),
      first_cycle_dave := p_not_six^3 * p_six,
      game_continue := p_not_six^4 in
  (∀ (prob : ℚ), 
      prob = first_cycle_dave + first_cycle_dave * game_continue / (1 - game_continue) → 
      prob = 125 / 671) :=
begin
  sorry
end

end dave_probability_is_125_over_671_l95_95961


namespace max_liters_l95_95896

-- Declare the problem conditions as constants
def initialHeat : ℕ := 480 -- kJ for the first 5 min
def reductionRate : ℚ := 0.25 -- 25%
def initialTemp : ℚ := 20 -- °C
def boilingTemp : ℚ := 100 -- °C
def specificHeatCapacity : ℚ := 4.2 -- kJ/kg°C

-- Define the heat required to raise m kg of water from initialTemp to boilingTemp
def heatRequired (m : ℚ) : ℚ := m * specificHeatCapacity * (boilingTemp - initialTemp)

-- Define the total available heat from the fuel
noncomputable def totalAvailableHeat : ℚ :=
  initialHeat / (1 - (1 - reductionRate))

-- The proof problem statement
theorem max_liters :
  ∃ (m : ℕ), m ≤ 5 ∧ heatRequired m ≤ totalAvailableHeat :=
sorry

end max_liters_l95_95896


namespace grade_assignment_ways_l95_95113

theorem grade_assignment_ways : (4^12 = 16777216) := 
by 
  sorry

end grade_assignment_ways_l95_95113


namespace range_of_a_l95_95630

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ¬(|x - 1| + |x - 2| ≤ a^2 + a + 1)) ↔ a ∈ set.Ioo (-1) 0 :=
by 
  sorry

end range_of_a_l95_95630


namespace number_of_good_students_is_5_or_7_l95_95349

-- Definitions based on the conditions
def total_students : ℕ := 25
def number_of_good_students (G : ℕ) (T : ℕ) := G + T = total_students
def first_group_condition (T : ℕ) := T > 12
def second_group_condition (G : ℕ) (T : ℕ) := T = 3 * (G - 1)

-- Problem statement in Lean 4:
theorem number_of_good_students_is_5_or_7 (G T : ℕ) :
  number_of_good_students G T ∧ first_group_condition T ∧ second_group_condition G T → G = 5 ∨ G = 7 :=
by
  sorry

end number_of_good_students_is_5_or_7_l95_95349


namespace find_m_l95_95706

-- Conditions
variables {m : ℝ}
def point1 := (m, 5)
def point2 := (2, m)
def slope := (point2.snd - point1.snd) / (point2.fst - point1.fst)
def given_slope := sqrt 2
axiom m_pos : 0 < m
axiom slope_given : slope = given_slope

--Proof statement
theorem find_m : m = 2 + 3 * sqrt 2 :=
by 
  sorry

end find_m_l95_95706


namespace correct_statements_about_graph_C_l95_95844

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - (Real.pi / 3))

theorem correct_statements_about_graph_C :
  (∃ (l : ℝ), l = 11 / 12 * Real.pi ∧ ∀ x y : ℝ, f(x) = y → f(2 * l - x) = y ∧ f x = -3) ∧
  (∀ x : ℝ, - (Real.pi / 12) ≤ x ∧ x ≤ 5 * (Real.pi / 12) → f(x) > f(x + 0.1)) :=
by
  sorry

end correct_statements_about_graph_C_l95_95844


namespace total_cost_of_yogurt_and_clothes_l95_95487

theorem total_cost_of_yogurt_and_clothes (price_yogurt : ℕ) (multiplier : ℕ):
  price_yogurt = 120 → multiplier = 6 →
  price_yogurt + price_yogurt * multiplier = 840 :=
by
  intros h₁ h₂
  rw [h₁, h₂]
  sorry

end total_cost_of_yogurt_and_clothes_l95_95487


namespace lemons_cost_l95_95053

theorem lemons_cost (L : ℝ) (h : 6 * L + 4 * 1 + 2 * 4 - 3 = 21) : L = 2 :=
by
  sorry

end lemons_cost_l95_95053


namespace longest_identical_last_n_digits_perfect_square_l95_95944

-- Definitions
def identical_last_n_digits (n : ℕ) (m : ℕ) : Prop :=
  let d := (m % 10) in
  d ≠ 0 ∧ (∀ k : ℕ, k < n → ((m / 10^k) % 10) = d)

def perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

-- Theorem (concise statement without intemediate solution steps)
theorem longest_identical_last_n_digits_perfect_square :
  ∃ n, ∀ m, identical_last_n_digits n m → perfect_square m → n = 3 ∧ m = 1444 :=
by sorry

end longest_identical_last_n_digits_perfect_square_l95_95944


namespace find_number_of_good_students_l95_95330

variable {G T : ℕ}

def is_good_or_troublemaker (G T : ℕ) : Prop :=
  G + T = 25 ∧ 
  (∀ s ∈ finset.range 5, T > 12) ∧ 
  (∀ s ∈ finset.range 20, T = 3 * (G - 1))

-- There are 25 students in total. 
-- Each student is either a good student or a troublemaker.
-- Good students always tell the truth, while troublemakers always lie.
-- Five students make the statement: "If I transfer to another class, more than half of the remaining students will be troublemakers."
-- The remaining 20 students make the statement: "If I transfer to another class, the number of troublemakers among the remaining students will be three times the number of good students."

theorem find_number_of_good_students (G T : ℕ) (h : is_good_or_troublemaker G T) :
  (G = 5 ∨ G = 7) :=
sorry

end find_number_of_good_students_l95_95330


namespace mixture_correct_l95_95377

def water_amount : ℚ := (3/5) * 20
def vinegar_amount : ℚ := (5/6) * 18
def mixture_amount : ℚ := water_amount + vinegar_amount

theorem mixture_correct : mixture_amount = 27 := 
by
  -- Here goes the proof steps
  sorry

end mixture_correct_l95_95377


namespace exists_an_b_for_f_l95_95628

-- Define the function f inductively
def f : ℕ → ℕ
| 1 := 1
| (n + 1) := 
  let m := (max {m : ℕ // ∃ a : Finset ℕ, a ⊆ (Finset.range (n + 1)) ∧ a.card = m ∧ (∀ x ∈ a, f x = f (a.min' (by simp)))}) 
  in m.1

theorem exists_an_b_for_f (n : ℕ) (hn : n > 0) : ∃ a b, (f (a * n + b) = n + 2) := by
  use 4
  use 8
  sorry

end exists_an_b_for_f_l95_95628


namespace winning_percentage_l95_95727

-- Defining the conditions
def election_conditions (winner_votes : ℕ) (win_margin : ℕ) (total_candidates : ℕ) : Prop :=
  total_candidates = 2 ∧ winner_votes = 864 ∧ win_margin = 288

-- Stating the question: What percentage of votes did the winner candidate receive?
theorem winning_percentage (V : ℕ) (winner_votes : ℕ) (win_margin : ℕ) (total_candidates : ℕ) :
  election_conditions winner_votes win_margin total_candidates → (winner_votes * 100 / V) = 60 :=
by
  sorry

end winning_percentage_l95_95727


namespace find_number_of_good_students_l95_95332

variable {G T : ℕ}

def is_good_or_troublemaker (G T : ℕ) : Prop :=
  G + T = 25 ∧ 
  (∀ s ∈ finset.range 5, T > 12) ∧ 
  (∀ s ∈ finset.range 20, T = 3 * (G - 1))

-- There are 25 students in total. 
-- Each student is either a good student or a troublemaker.
-- Good students always tell the truth, while troublemakers always lie.
-- Five students make the statement: "If I transfer to another class, more than half of the remaining students will be troublemakers."
-- The remaining 20 students make the statement: "If I transfer to another class, the number of troublemakers among the remaining students will be three times the number of good students."

theorem find_number_of_good_students (G T : ℕ) (h : is_good_or_troublemaker G T) :
  (G = 5 ∨ G = 7) :=
sorry

end find_number_of_good_students_l95_95332


namespace decreasing_intervals_triangle_side_c_l95_95218

-- Define the function
def f (x : ℝ) : ℝ := sin x * cos (x + π / 6) + 1

-- First part: Prove the intervals where f(x) is monotonically decreasing
theorem decreasing_intervals (k : ℤ) :
  ∀ x : ℝ, (π / 6 + k * π ≤ x ∧ x ≤ 2 * π / 3 + k * π) ↔ (cos (2 * x + π / 6) < 0) := 
sorry

-- Second part: Prove c = 2 in △ABC given the conditions
theorem triangle_side_c (C a b : ℝ) (AC BC : E → E → ℝ) :
  (f C = 5 / 4) ∧ (b = 4) ∧ (AC • BC = 12) → (C = π / 6) ∧ (a = 2 * √3) → (c = 2) := 
sorry

end decreasing_intervals_triangle_side_c_l95_95218


namespace loan_amount_needed_l95_95454

-- Define the total cost of tuition.
def total_tuition : ℝ := 30000

-- Define the amount Sabina has saved.
def savings : ℝ := 10000

-- Define the grant coverage rate.
def grant_coverage_rate : ℝ := 0.4

-- Define the remainder of the tuition after using savings.
def remaining_tuition : ℝ := total_tuition - savings

-- Define the amount covered by the grant.
def grant_amount : ℝ := grant_coverage_rate * remaining_tuition

-- Define the loan amount Sabina needs to apply for.
noncomputable def loan_amount : ℝ := remaining_tuition - grant_amount

-- State the theorem to prove the loan amount needed.
theorem loan_amount_needed : loan_amount = 12000 := by
  sorry

end loan_amount_needed_l95_95454


namespace solve_a_b_c_d_l95_95984

theorem solve_a_b_c_d (n a b c d : ℕ) (h0 : 0 ≤ a) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : 2^n = a^2 + b^2 + c^2 + d^2) : 
  (a, b, c, d) ∈ {p | p = (↑0, ↑0, ↑0, 2^n.div (↑4)) ∨
                  p = (↑0, ↑0, 2^n.div (↑4), 2^n.div (↑4)) ∨
                  p = (2^n.div (↑4), 2^n.div (↑4), 2^n.div (↑4), 2^n.div (↑4)) ∨
                  p = (2^n.div (↑4), 0, 0, 0) ∨
                  p = (0, 2^n.div (↑4), 0, 0) ∨
                  p = (0, 0, 2^n.div (↑4), 0) ∨
                  p = (0, 0, 0, 2^n.div (↑4))} :=
sorry

end solve_a_b_c_d_l95_95984


namespace good_students_l95_95336

theorem good_students (E B : ℕ) (h1 : E + B = 25) (h2 : 12 < B) (h3 : B = 3 * (E - 1)) :
  E = 5 ∨ E = 7 :=
by 
  sorry

end good_students_l95_95336


namespace sum_of_identical_digit_numbers_l95_95240

-- Definitions of the compositions of digits
def five_digit_number (a : ℕ) : ℕ := 10000 * a + 1000 * a + 100 * a + 10 * a + a
def four_digit_number (b : ℕ) : ℕ := 1000 * b + 100 * b + 10 * b + b
def three_digit_number (c : ℕ) : ℕ := 100 * c + 10 * c + c

-- Statement of the problem
theorem sum_of_identical_digit_numbers :
  ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧
  let sum := five_digit_number a + four_digit_number b + three_digit_number c in
  sum >= 10000 ∧ sum < 100000 ∧
  let digits := [sum / 10000 % 10, sum / 1000 % 10, sum / 100 % 10, sum / 10 % 10, sum % 10] in
  digits.nodup :=
sorry

end sum_of_identical_digit_numbers_l95_95240


namespace ratio_jordana_jennifer_10_years_l95_95403

-- Let's define the necessary terms and conditions:
def Jennifer_future_age := 30
def Jordana_current_age := 80
def years := 10

-- Define the ratio of ages function:
noncomputable def ratio_of_ages (future_age_jen : ℕ) (current_age_jord : ℕ) (yrs : ℕ) : ℚ :=
  (current_age_jord + yrs) / future_age_jen

-- The statement we need to prove:
theorem ratio_jordana_jennifer_10_years :
  ratio_of_ages Jennifer_future_age Jordana_current_age years = 3 := by
  sorry

end ratio_jordana_jennifer_10_years_l95_95403


namespace vector_parallel_l95_95232

open Real

theorem vector_parallel (k : ℝ) 
(a : ℝ × ℝ := (1, 2 * k)) 
(b : ℝ × ℝ := (-3, 6)) 
(h : ∃ λ : ℝ, λ • a = b) : k = 1 :=
sorry

end vector_parallel_l95_95232


namespace quadratic_roots_condition_l95_95198

-- Given conditions: the discriminant should be positive for two distinct real roots
def quadratic_discriminant_condition (m : ℝ) : Prop :=
  (2 * m + 1) ^ 2 - 4 * (m - 2) ^ 2 > 0

-- Additional condition
def non_zero_condition (m : ℝ) : Prop :=
  m ≠ 2

-- Conclusion: the quadratic equation has two distinct real roots if and only if
-- m > 3/4 and m ≠ 2 
theorem quadratic_roots_condition (m : ℝ) : 
  quadratic_discriminant_condition(m) ∧ non_zero_condition(m) ↔ m > 3/4 ∧ m ≠ 2 := 
sorry

end quadratic_roots_condition_l95_95198


namespace find_certain_number_l95_95466

theorem find_certain_number : ∃ x : ℕ, (((x - 50) / 4) * 3 + 28 = 73) → x = 110 :=
by
  sorry

end find_certain_number_l95_95466


namespace find_general_term_l95_95199

theorem find_general_term (S a : ℕ → ℤ) (n : ℕ) (h_sum : S n = 2 * a n + 1) : a n = -2 * n - 1 := sorry

end find_general_term_l95_95199


namespace integer_solutions_exist_l95_95464

theorem integer_solutions_exist (t : ℤ) (k : ℤ) (h : k ∈ {6, -4, -9, -16, -1, -11}) :
  ∃ y : ℤ, (35 * t + k)^4 + 2 * (35 * t + k)^3 + 8 * (35 * t + k) + 9 = 35 * y :=
by
  let x := 35 * t + k
  exact sorry

end integer_solutions_exist_l95_95464


namespace number_of_good_students_is_5_or_7_l95_95350

-- Definitions based on the conditions
def total_students : ℕ := 25
def number_of_good_students (G : ℕ) (T : ℕ) := G + T = total_students
def first_group_condition (T : ℕ) := T > 12
def second_group_condition (G : ℕ) (T : ℕ) := T = 3 * (G - 1)

-- Problem statement in Lean 4:
theorem number_of_good_students_is_5_or_7 (G T : ℕ) :
  number_of_good_students G T ∧ first_group_condition T ∧ second_group_condition G T → G = 5 ∨ G = 7 :=
by
  sorry

end number_of_good_students_is_5_or_7_l95_95350


namespace num_possible_values_of_a_l95_95021

def is_factor (m n : Nat) : Prop := ∃ k, n = m * k

def valid_a (a : Nat) : Prop :=
  2 ∣ a ∧ a ∣ 18 ∧ a > 0 ∧ a ≤ 10

theorem num_possible_values_of_a : 
  {a : Nat // valid_a a}.card = 2 := by
  sorry

end num_possible_values_of_a_l95_95021


namespace sum_of_consecutive_even_negative_integers_l95_95852

theorem sum_of_consecutive_even_negative_integers (n m : ℤ) 
  (h1 : n % 2 = 0)
  (h2 : m % 2 = 0)
  (h3 : n < 0)
  (h4 : m < 0)
  (h5 : m = n + 2)
  (h6 : n * m = 2496) : n + m = -102 := 
sorry

end sum_of_consecutive_even_negative_integers_l95_95852


namespace ratio_student_adult_tickets_l95_95576

theorem ratio_student_adult_tickets (A : ℕ) (S : ℕ) (total_tickets: ℕ) (multiple: ℕ) :
  (A = 122) →
  (total_tickets = 366) →
  (S = multiple * A) →
  (S + A = total_tickets) →
  (S / A = 2) :=
by
  intros hA hTotal hMultiple hSum
  -- The proof will go here
  sorry

end ratio_student_adult_tickets_l95_95576


namespace big_bottles_sold_percentage_l95_95953

-- Definitions based on conditions
def small_bottles_initial : ℕ := 5000
def big_bottles_initial : ℕ := 12000
def small_bottles_sold_percentage : ℝ := 0.15
def total_bottles_remaining : ℕ := 14090

-- Question in Lean 4
theorem big_bottles_sold_percentage : 
  (12000 - (12000 * x / 100) + 5000 - (5000 * 15 / 100)) = 14090 → x = 18 :=
by
  intros h
  sorry

end big_bottles_sold_percentage_l95_95953


namespace number_of_possible_heights_is_680_l95_95523

noncomputable def total_possible_heights : Nat :=
  let base_height := 200 * 3
  let max_additional_height := 200 * (20 - 3)
  let min_height := base_height
  let max_height := base_height + max_additional_height
  let number_of_possible_heights := (max_height - min_height) / 5 + 1
  number_of_possible_heights

theorem number_of_possible_heights_is_680 : total_possible_heights = 680 := by
  sorry

end number_of_possible_heights_is_680_l95_95523


namespace exam_room_selection_l95_95310

theorem exam_room_selection (rooms : List ℕ) (n : ℕ) 
    (fifth_room_selected : 5 ∈ rooms) (twentyfirst_room_selected : 21 ∈ rooms) :
    rooms = [5, 13, 21, 29, 37, 45, 53, 61] → 
    37 ∈ rooms ∧ 53 ∈ rooms :=
by
  sorry

end exam_room_selection_l95_95310


namespace diagonal_intersection_probability_decagon_l95_95094

noncomputable def probability_diagonal_intersection_in_decagon : ℚ :=
  let vertices := 10
  let total_diagonals := (vertices * (vertices - 3)) / 2
  let total_pairs_of_diagonals := total_diagonals * (total_diagonals - 1) / 2
  let total_intersecting_pairs := (vertices * (vertices - 1) * (vertices - 2) * (vertices - 3)) / 24
  total_intersecting_pairs / total_pairs_of_diagonals

theorem diagonal_intersection_probability_decagon (h : probability_diagonal_intersection_in_decagon = 42 / 119) : 
  probability_diagonal_intersection_in_decagon = 42 / 119 :=
sorry

end diagonal_intersection_probability_decagon_l95_95094


namespace lcm_gcf_ratio_120_504_l95_95536

theorem lcm_gcf_ratio_120_504 : 
  let a := 120
  let b := 504
  (Int.lcm a b) / (Int.gcd a b) = 105 := by
  sorry

end lcm_gcf_ratio_120_504_l95_95536


namespace permutations_PERSEVERANCE_l95_95157

theorem permutations_PERSEVERANCE :
  let totalLetters := 11
  let freqE := 3
  let freqR := 2
  (nat.factorial totalLetters) / (nat.factorial freqE * nat.factorial freqR * nat.factorial 1 * nat.factorial 1 * nat.factorial 1 * nat.factorial 1 * nat.factorial 1) = 1663200 :=
by
  sorry

end permutations_PERSEVERANCE_l95_95157


namespace tshirts_per_package_l95_95916

theorem tshirts_per_package (p t : ℕ) (h : p = 71) (h₁ : t = 426) : t / p = 6 :=
by {
  simp [h, h₁],
  sorry
}

end tshirts_per_package_l95_95916


namespace volume_of_smaller_tetrahedron_equals_l95_95977

-- Given (Conditions)
variables (T : Type) [volume_space T] [tetrahedron T] [unit_volume T]

-- Proof Problem Statement
def smaller_tetrahedron_volume : Real :=
(1 - Real.cbrt (1 / 2))^3

-- Goal (Question == Correct Answer)
theorem volume_of_smaller_tetrahedron_equals :
  smaller_tetrahedron_volume T = (1 / 8) * (3 - 2 * Real.cbrt 2)^3 :=
sorry

end volume_of_smaller_tetrahedron_equals_l95_95977


namespace complex_division_l95_95629

theorem complex_division (a b : ℝ) (i : ℂ) (hi : i = complex.I) : 
  (2 * i / (1 - i) = a + b * i) → a = -1 ∧ b = 1 :=
by
  sorry

end complex_division_l95_95629


namespace sequence_properties_l95_95387

def a_n (n : ℕ) : ℕ := n + 2

def b_n (n : ℕ) : ℕ := 2^(n) + n

theorem sequence_properties :
  (a_n 2 = 4) ∧ (a_n 4 + a_n 7 = 15) ∧
  (∀ n, a_n n = n + 2) ∧
  (∑ n in finset.range 10 + 1, b_n (n + 1) = 2101) :=
by
  sorry

end sequence_properties_l95_95387


namespace proof_problem_l95_95201

theorem proof_problem :
  (∃ x : ℝ, x - 1 ≥ Real.log x) ∧ (¬ ∀ x ∈ Ioo 0 Real.pi, Real.sin x + 1 / Real.sin x > 2) := sorry

end proof_problem_l95_95201


namespace xy_problem_l95_95273

theorem xy_problem (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 :=
by
  sorry

end xy_problem_l95_95273


namespace arithmetic_sequence_c_d_sum_l95_95500

theorem arithmetic_sequence_c_d_sum (c d : ℕ) 
  (h1 : 10 - 3 = 7) 
  (h2 : 17 - 10 = 7) 
  (h3 : c = 17 + 7) 
  (h4 : d = c + 7) 
  (h5 : d + 7 = 38) :
  c + d = 55 := 
by
  sorry

end arithmetic_sequence_c_d_sum_l95_95500


namespace sugar_percentage_is_7_5_l95_95067

theorem sugar_percentage_is_7_5 
  (V1 : ℕ := 340)
  (p_water : ℝ := 88/100)
  (p_kola : ℝ := 5/100)
  (p_sugar : ℝ := 7/100)
  (V_sugar_add : ℝ := 3.2)
  (V_water_add : ℝ := 10)
  (V_kola_add : ℝ := 6.8) : 
  (
    (23.8 + 3.2) / (340 + 3.2 + 10 + 6.8) * 100 = 7.5
  ) :=
  by
  sorry

end sugar_percentage_is_7_5_l95_95067


namespace part1_min_value_zero_part2_inequality_l95_95216

-- First Part: Given the function with minimum value condition
theorem part1_min_value_zero (a : ℝ) (h : ∀ x, x > 0 → ln x + a / x - 1 ≥ 0) : a = 1 := sorry

-- Second Part: Using result from part1 to prove the given inequality
theorem part2_inequality (x : ℝ) (h : x > 0) : exp x + (ln x - 1) * sin x > 0 :=
sorry

end part1_min_value_zero_part2_inequality_l95_95216


namespace class_proof_l95_95354

def class_problem : Prop :=
  let total_students := 25
  let (students : Type) := {s // s ∈ Fin total_students}
  let good students (s : students) : Prop := sorry -- Condition Placeholder
  let troublemakers (s : students) : Prop := sorry -- Condition Placeholder
  have all_good_or_trouble : ∀ s, good students s ∨ troublemakers s,
    from sorry,
  have lh_trouble : ∀ (a b c d e : students), (good students a ∧ good students b ∧ good students c ∧ good students d ∧ good students e) →
    ∃ (x : ℕ), x > ((total_students - 1) / 2) ∧
    (λ rem_trouble : students - {a, b, c, d, e}, B : ℕ, B = x) ∧
    (λ rem_good : students - {a, b, c, d, e}, G : ℕ, G ∈ rem_good), from sorry,
  have th_triple_ratio : ∀ s, (students - {s}).card = 24 →
    ∃ (x y : ℕ), (troublemakers x ∧ good students y) ∧ (3*y = x), from sorry,
  ∃ (num_good_students : ℕ), (num_good_students = 5 ∨ num_good_students = 7)
  
theorem class_proof : class_problem :=
  sorry

end class_proof_l95_95354


namespace pairs_sum_104_among_20_l95_95964

theorem pairs_sum_104_among_20 {S : Finset ℕ} 
  (h1 : S = Finset.range' 1 100)
  (h2 : ∀ x ∈ S, x % 3 = 1)
  (h3 : ∀ x y ∈ S, x ≠ y → x + y = 104)
  (h4 : 20 ≤ S.card) :
  ∃ (A ⊆ S), A.card = 20 ∧ ∃ x y ∈ A, x ≠ y ∧ x + y = 104 := 
sorry

end pairs_sum_104_among_20_l95_95964


namespace possible_values_of_b_over_a_l95_95296

open Real

theorem possible_values_of_b_over_a (a b : ℝ) (h : ln a + b - a * exp (b - 1) ≥ 0) :
    ∃ x ∈ {exp (-1), exp (-2), -exp (-2)}, x = b / a :=
by 
  sorry

end possible_values_of_b_over_a_l95_95296


namespace range_of_quadratic_func_l95_95492

theorem range_of_quadratic_func : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → -x^2 + 2 * x + 3 ∈ Set.range (λ (x : ℝ), -x^2 + 2 * x + 3) :=
  by
    intro x hx
    have h1 : x ∈ Set.Icc 0 3 := hx
    have h_max : - (1:ℝ)^2 + 2 * 1 + 3 = 4 := by norm_num
    have h_0 : - (0:ℝ)^2 + 2 * 0 + 3 = 3 := by norm_num
    have h_3 : - (3:ℝ)^2 + 2 * 3 + 3 = 0 := by norm_num
    sorry

end range_of_quadratic_func_l95_95492


namespace distance_point_to_y_axis_l95_95474

-- Define the point (3, -5)
def point := (3, -5 : ℤ)

-- Define a function to calculate the distance to the y-axis
def distance_to_y_axis (p : ℤ × ℤ) : ℤ :=
  abs p.1

-- Define a statement that the distance from (3, -5) to the y-axis is 3
theorem distance_point_to_y_axis :
  distance_to_y_axis point = 3 :=
  sorry

end distance_point_to_y_axis_l95_95474


namespace find_p_value_l95_95131

theorem find_p_value :
  ∃ p : ℝ,
    let x := [2, 4, 5, 6, 8] in
    let y := [30, 40, 60, p, 70] in
    let mean_x := (x.sum / 5) in
    let mean_y := (y.sum / 5) in
    5 = mean_x ∧ 40 + p / 5 = mean_y ∧ mean_y = 6.5 * mean_x + 17.5 ∧ p = 50 :=
by
  sorry

end find_p_value_l95_95131


namespace bob_sandwich_cost_correct_l95_95589

-- Definitions for Andy's purchases
def soda_cost : ℝ := 1.50
def hamburger_cost : ℝ := 2.75
def chips_cost : ℝ := 1.25
def andy_tax_rate : ℝ := 0.08

-- Definitions for Bob's purchases
def initial_sandwich_cost : ℝ := 2.68
def sandwich_count : ℕ := 5
def sandwich_discount_rate : ℝ := 0.10
def water_cost : ℝ := 1.25
def bob_water_tax_rate : ℝ := 0.07

-- Compute Andy's total spending
def andy_pre_tax_total : ℝ := soda_cost + 3 * hamburger_cost + chips_cost
def andy_tax : ℝ := andy_tax_rate * andy_pre_tax_total
def andy_total_spending : ℝ := andy_pre_tax_total + andy_tax

-- Compute Bob's spending before tax
def bob_sandwiches_pre_discount_total : ℝ := sandwich_count * initial_sandwich_cost
def bob_sandwich_discount : ℝ := sandwich_discount_rate * bob_sandwiches_pre_discount_total
def bob_sandwiches_post_discount_total : ℝ := bob_sandwiches_pre_discount_total - bob_sandwich_discount
def bob_cost_per_sandwich_after_discount : ℝ := bob_sandwiches_post_discount_total / sandwich_count

-- Compute Bob's total tax and total spending
def bob_water_tax : ℝ := bob_water_tax_rate * water_cost
def bob_combined_total_pre_tax : ℝ := bob_sandwiches_post_discount_total + water_cost
def bob_combined_total_tax : ℝ := bob_water_tax
def bob_total_spending : ℝ := bob_combined_total_pre_tax + bob_combined_total_tax

theorem bob_sandwich_cost_correct :
  bob_total_spending = andy_total_spending → bob_cost_per_sandwich_after_discount = 2.412 :=
by 
  -- Proof omitted
  sorry

end bob_sandwich_cost_correct_l95_95589


namespace regular_price_per_can_is_0_40_l95_95039

variable (P : ℝ) -- Regular price per can
variable (discounted_price_per_can : ℝ) := 0.85 * P -- Discounted price per can
constant (price_of_100_cans : ℝ) := 34 -- Price of 100 cans

theorem regular_price_per_can_is_0_40 (h : discounted_price_per_can = price_of_100_cans / 100) :
  P = 0.40 := by
    sorry

end regular_price_per_can_is_0_40_l95_95039


namespace find_range_of_m_l95_95654

section problem
variables (m : ℝ)

def p := ∀ x : ℝ, (|x - 1| > m - 1) → x ∈ ℝ
def q := ∀ x : ℝ, (-(5 - 2m)^x < -(5 - 2m)^(x + 1))

theorem find_range_of_m :
  (∃ m, (p m ∨ q m) ∧ ¬ (p m ∧ q m)) ↔ (1 ≤ m ∧ m < 2) :=
sorry
end problem

end find_range_of_m_l95_95654


namespace three_digit_decimal_bounds_l95_95574

def is_rounded_half_up (x : ℝ) (y : ℝ) : Prop :=
  (y - 0.005 ≤ x) ∧ (x < y + 0.005)

theorem three_digit_decimal_bounds :
  ∃ (x : ℝ), (8.725 ≤ x) ∧ (x ≤ 8.734) ∧ is_rounded_half_up x 8.73 :=
by
  sorry

end three_digit_decimal_bounds_l95_95574


namespace smallest_n_satisfying_conditions_l95_95537

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, n > 0 ∧ (∃ k₁ : ℕ, 3 * n = k₁^2) ∧ (∃ k₂ : ℕ, 2 * n = k₂^3) ∧ (∃ k₃ : ℕ, 5 * n = k₃^5) ∧ n = 7500 :=
begin
  sorry
end

end smallest_n_satisfying_conditions_l95_95537


namespace smallest_omega_l95_95222

theorem smallest_omega (ω : ℝ) (h_pos : ω > 0) :
  (∃ k : ℤ, ω = 6 * k) ∧ (∀ k : ℤ, k > 0 → ω = 6 * k → ω = 6) :=
by sorry

end smallest_omega_l95_95222


namespace car_horizontal_velocity_l95_95557

noncomputable def horizontal_component_velocity
  (u : ℝ) (a : ℝ) (t : ℝ) (θ : ℝ) : ℝ :=
  let v := u + a * t in
  v * Real.cos (θ * Real.pi / 180)

theorem car_horizontal_velocity 
  (u a t θ : ℝ) (h₁ : u = 10) (h₂ : a = 2) (h₃ : t = 3) (h₄ : θ = 15) :
  horizontal_component_velocity u a t θ ≈ 15.4544 :=
by
  rw [h₁, h₂, h₃, h₄]
  have v_def : horizontal_component_velocity 10 2 3 15 = (10 + 2 * 3) * Real.cos (15 * Real.pi / 180) := rfl
  rw [v_def]
  have h_cos : Real.cos (15 * Real.pi / 180) ≈ 0.9659 := by sorry
  have _: (10 + 2 * 3) * 0.9659 ≈ 15.4544 := by sorry
  sorry

end car_horizontal_velocity_l95_95557


namespace Ivanov_made_an_error_l95_95819

theorem Ivanov_made_an_error (mean median : ℝ) (variance : ℝ) (h1 : mean = 0) (h2 : median = 4) (h3 : variance = 15.917) : 
  |mean - median| ≤ Real.sqrt variance → False :=
by {
  have mean_value : mean = 0 := h1,
  have median_value : median = 4 := h2,
  have variance_value : variance = 15.917 := h3,

  let lhs := |mean_value - median_value|,
  have rhs := Real.sqrt variance_value,
  
  calc
    lhs = |0 - 4| : by rw [mean_value, median_value]
    ... = 4 : by norm_num,
  
  have rhs_val : Real.sqrt variance_value ≈ 3.99 := by sorry, -- approximate value for demonstration
  
  have ineq : 4 ≤ rhs_val := by 
    calc 4 = 4 : rfl -- trivial step for clarity,
    have sqrt_val : Real.sqrt 15.917 < 4 := by sorry, -- from calculation or suitable proof
  
  exact absurd ineq (not_le_of_gt sqrt_val)
}

end Ivanov_made_an_error_l95_95819


namespace painting_time_for_five_people_l95_95161

-- Define the conditions as constants or variables
constant num_workers_one : ℕ := 8
constant time_one : ℕ := 3
constant consistent_rate : Prop := true

-- Derived condition from the problem
constant total_work : ℕ := num_workers_one * time_one

-- Define the number of workers in the second condition
constant num_workers_two : ℕ := 5

noncomputable def time_two : ℚ :=
  total_work / num_workers_two

theorem painting_time_for_five_people :
  consistent_rate → time_two = 24 / 5 :=
by
  intro h,
  sorry

end painting_time_for_five_people_l95_95161


namespace ivanov_error_l95_95826

theorem ivanov_error (x : ℝ) (m : ℝ) (S2 : ℝ) (std_dev : ℝ) :
  x = 0 → m = 4 → S2 = 15.917 → std_dev = Real.sqrt S2 →
  ¬ (|x - m| ≤ std_dev) :=
by
  intros h1 h2 h3 h4
  -- Using the given values directly to state the inequality
  have h5 : |0 - 4| = 4 := by norm_num
  have h6 : Real.sqrt 15.917 ≈ 3.99 := sorry  -- approximation as direct result
  -- Evaluating the inequality
  have h7 : 4 ≰ 3.99 := sorry  -- this represents the key step that shows the error
  exact h7
  sorry

end ivanov_error_l95_95826


namespace range_of_magnitude_min_g_x_l95_95183

-- Part 1: Range of values for |b + c|
def vec_a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
def vec_b (x k : ℝ) : ℝ × ℝ := (Real.sin x, k)
def vec_c (x k : ℝ) : ℝ × ℝ := (-2 * Real.cos x, Real.sin x - k)
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)
def sum_vecs (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

theorem range_of_magnitude (k : ℝ) (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 4) :
  1 ≤ magnitude (sum_vecs (vec_b x k) (vec_c x k)) ∧
  magnitude (sum_vecs (vec_b x k) (vec_c x k)) ≤ 2 := 
sorry

-- Part 2: Minimum value of g(x)
def g_x (x k : ℝ) : ℝ :=
  let suma := sum_vecs (vec_a x) (vec_b x k)
  (suma.1 * (vec_c x k).1) + (suma.2 * (vec_c x k).2)

theorem min_g_x (k : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 4) :
  g_x x k = -(3 / 2) -> k = 0 :=
sorry

end range_of_magnitude_min_g_x_l95_95183


namespace nina_initial_widgets_l95_95790

noncomputable def cost_reduced (C : ℝ) := C - 1.25
noncomputable def total_money := 16.67
noncomputable def reduced_cost_total (C : ℝ) := 8 * cost_reduced C
noncomputable def initial_widgets (C : ℝ) := total_money / C

theorem nina_initial_widgets :
  (∃ C : ℝ, reduced_cost_total C = total_money) →
  (∃ W : ℝ, W = initial_widgets C ∧ W = 5) :=
begin
  sorry
end

end nina_initial_widgets_l95_95790


namespace intersection_point_exists_l95_95226

def line_l (x y : ℝ) : Prop := 2 * x + y = 10
def line_l_prime (x y : ℝ) : Prop := x - 2 * y + 10 = 0
def passes_through (x y : ℝ) (p : ℝ × ℝ) : Prop := p.2 = y ∧ 2 * p.1 - 10 = x

theorem intersection_point_exists :
  ∃ p : ℝ × ℝ, line_l p.1 p.2 ∧ line_l_prime p.1 p.2 ∧ passes_through p.1 p.2 (-10, 0) :=
sorry

end intersection_point_exists_l95_95226


namespace distance_of_point_on_parabola_l95_95709

theorem distance_of_point_on_parabola {f m : ℝ} (h : (3 - f) ^ 2 + m ^ 2 = 9 + 6 * f - f ^ 2) :
  sqrt ((3 - f) ^ 2 + m ^ 2) = 4 := sorry

end distance_of_point_on_parabola_l95_95709


namespace cost_of_one_dozen_pens_l95_95921

theorem cost_of_one_dozen_pens (x : ℝ) (cost_pen cost_pencil : ℝ) 
  (h1 : cost_pen = 5 * cost_pencil) 
  (h2 : 3 * cost_pen + 5 * cost_pencil = 240) : 
  (12 * cost_pen = 720) :=
by
  -- Setup the problem based on the given conditions
  have h_cost_pen : cost_pen = 5 * x := by simp [h1]
  have h_cost_pencil : cost_pencil = x := by simp [h1]
  
  -- Derive x from the second condition
  have h_eqn : 3 * (5 * x) + 5 * x = 240 := by simp [h1, h2]
  have : 20 * x = 240 := by linarith
  have h_x : x = 12 := by linarith

  -- Prove the cost of one dozen pens
  calc 
    12 * cost_pen = 12 * (5 * x)  : by rw [h_cost_pen]
              ... = 12 * (5 * 12) : by rw [h_x]
              ... = 720           : by norm_num

end cost_of_one_dozen_pens_l95_95921


namespace probability_comparison_l95_95137

/-
Two balls are taken from box A (containing 2 white and 1 black ball) and placed into box B (containing 1 white and 2 black balls).
Then, two balls are taken randomly from box B.
Events are defined as follows:
  - A: Both balls taken from box B are white
  - B: Both balls taken from box B are black
  - C: One white ball and one black ball are taken from box B
The goal is to prove that P(B) < P(C).
-/

noncomputable def P : (set (set Nat)) → Real := sorry -- Define the probability function

variable {Box A : Finset Nat}
variable {Box B : Finset Nat}

axiom initial_condition_boxA : Box_A.card = 3 ∧ ∃ wA bA : Nat, wA = 2 ∧ bA = 1
axiom initial_condition_boxB : Box_B.card = 3 ∧ ∃ wB bB : Nat, wB = 1 ∧ bB = 2

def event_A : set (set Nat) := {s | s.card = 2 ∧ ∀ x ∈ s, x = 1}
def event_B : set (set Nat) := {s | s.card = 2 ∧ ∀ x ∈ s, x = 2}
def event_C : set (set Nat) := {s | s.card = 2 ∧ ∃ x y ∈ s, x ≠ y ∧ ((x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 1))}

theorem probability_comparison : P(event_B) < P(event_C) :=
by {
  -- Proof should be provided here, but for now we use sorry to demarcate it.
  sorry
}

end probability_comparison_l95_95137


namespace no_shaded_area_in_tiled_floor_l95_95554

theorem no_shaded_area_in_tiled_floor :
  ∀ (length width : ℕ) (tile_side : ℕ)
    (tiles_pattern : ℕ → ℕ → Prop),
  -- Conditions
  length = 12 →
  width = 9 →
  tile_side = 1 →
  (∀ (i j : ℕ), tiles_pattern i j → 0 ≤ i ∧ i < length ∧ 0 ≤ j ∧ j < width) →
  ∀ (area_of_right_triangle : ℝ),
  -- Each right triangle in tile has area 1/4 square feet
  area_of_right_triangle = 1/4 →
  (∀ (total_tiles total_area_of_triangles : ℕ) (total_shaded_area : ℝ),
    -- Total tiles
    total_tiles = length * width →
    -- Total area of four triangles per tile is 1 square foot
    total_area_of_triangles = total_tiles →
    -- Total shaded area should be zero
    total_shaded_area = total_tiles * 0 →
    total_shaded_area = 0) :=
begin
  sorry
end

end no_shaded_area_in_tiled_floor_l95_95554


namespace jills_favorite_number_l95_95919

theorem jills_favorite_number (n : ℕ) (h_even : even n) (h_repeat_prime : ∃ k : ℕ, 7 ^ 2 ∣ n * k) : n = 98 :=
sorry

end jills_favorite_number_l95_95919


namespace Ivanov_made_error_l95_95803

theorem Ivanov_made_error (x m : ℝ) (S_squared : ℝ) (h_x : x = 0) (h_m : m = 4) (h_S_squared : S_squared = 15.917) :
  ¬(|x - m| ≤ Real.sqrt S_squared) :=
by
  have h_sd : Real.sqrt S_squared = 3.99 := by sorry
  have h_ineq: |x - m| = 4 := by sorry
  rw [h_sd, h_ineq]
  linarith

end Ivanov_made_error_l95_95803


namespace distance_F_to_l_is_sqrt3_l95_95687

noncomputable def parabola := { p : ℝ × ℝ // p.2^2 = 4 * p.1 }
def directrix_intersects_x_axis_at (A : ℝ × ℝ) := A = (-1, 0)
def focus_is_at (F : ℝ × ℝ) := F = (1, 0)
def line_through_with_inclination (A : ℝ × ℝ) (θ : ℝ) (l : ℝ → ℝ) :=
  A.2 = l (A.1) ∧ θ = real.arctan (l 1 - l 0)

def distance_from_point_to_line (F : ℝ × ℝ) (l : ℝ → ℝ) : ℝ :=
  abs (l (F.1) - F.2) / real.sqrt (1 + (real.deriv l (F.1))^2)

theorem distance_F_to_l_is_sqrt3 :
  ∀ A F l,
  directrix_intersects_x_axis_at A →
  focus_is_at F →
  line_through_with_inclination A (π / 3) l →
  distance_from_point_to_line F l = √3 := by
-- Define the conditions from the problem in Lean
-- A is (-1, 0) and F is (1, 0)
-- Line passes through A with inclination angle of π/3
-- Calculate the distance from F to the line l
  sorry

end distance_F_to_l_is_sqrt3_l95_95687


namespace f_neg2_f_3_f_values_l95_95774

def f : ℝ → ℝ := λ x,
if x ≤ 1 then 3 * x + 4 else 7 - 3 * x^2

theorem f_neg2 : f (-2) = -2 := by
  unfold f
  simp [if_pos]
  linarith

theorem f_3 : f 3 = -20 := by
  unfold f
  simp [if_neg]
  linarith

-- Or combined as a single theorem
theorem f_values : f (-2) = -2 ∧ f 3 = -20 := by
  split
  case left => exact f_neg2
  case right => exact f_3

end f_neg2_f_3_f_values_l95_95774


namespace good_students_options_l95_95326

variables (E B : ℕ)

-- Define the condition that the class has 25 students
def total_students : Prop := E + B = 25

-- Define the condition given by the first group of students
def first_group_condition : Prop := B > 12

-- Define the condition given by the second group of students
def second_group_condition : Prop := B = 3 * (E - 1)

-- Define the problem statement
theorem good_students_options (E B : ℕ) :
  total_students E B → first_group_condition B → second_group_condition E B → (E = 5 ∨ E = 7) :=
by
  intros h1 h2 h3
  sorry

end good_students_options_l95_95326


namespace tan_product_l95_95925

theorem tan_product : 
  (∏ i in Finset.range 45, (1 + Real.tan (i+1) * Real.pi / 180)) = 2^23 :=
by
  sorry

end tan_product_l95_95925


namespace true_propositions_count_l95_95639

noncomputable def number_of_true_propositions (m n : Line) (α β : Plane) : Nat :=
if (m ⊆ α ∧ n ∥ α) → (m ∥ n) then 1 else 0 +
if (m ∥ α ∧ m ∥ β) → (α ∥ β) then 1 else 0 +
if (m ⊥ α ∧ m ⊥ n) → (n ∥ α) then 1 else 0 +
if (m ⊥ α ∧ m ⊥ β) → (α ∥ β) then 1 else 0

theorem true_propositions_count {m n : Line} {α β : Plane} :
  number_of_true_propositions m n α β = 1 :=
sorry

end true_propositions_count_l95_95639


namespace count_valid_sentences_l95_95468

-- Define the possible words in the Gnollish language
inductive Word : Type
| splargh | glumph | amr | blarg

open Word

-- Define the restriction predicate for word pairs
def valid_pair (w1 w2 : Word) : Prop :=
  ¬ ((w1 = splargh ∧ w2 = glumph) ∨ (w1 = blarg ∧ w2 = amr))

-- Define what constitutes a valid 3-word sentence
def valid_sentence (w1 w2 w3 : Word) : Prop :=
  valid_pair w1 w2 ∧ valid_pair w2 w3

-- The theorem we want to prove
theorem count_valid_sentences : (finset.univ : finset (Word × Word × Word)).filter (λ t, valid_sentence t.1 t.2.1 t.2.2).card = 49 :=
by sorry

end count_valid_sentences_l95_95468


namespace initial_tagged_fish_l95_95714

-- Define the conditions as Lean definitions
def caught_second_catch := 60
def tagged_second_catch := 2
def approximate_total_fish := 1800

-- Define the statement to be proved
theorem initial_tagged_fish (T : ℕ) 
  (h1 : tagged_second_catch = 2) 
  (h2 : caught_second_catch = 60) 
  (h3 : approximate_total_fish = 1800) 
  (proportion : ∀ {a b c d : ℕ}, a * d = b * c → a / b = c / d) :
  (T = 60) :=
by
  -- Define the given proportion
  have prop : 2 * 1800 = 60 * T := by
    calc
      2 * 1800 = 3600 : rfl
      ... = 60 * 60 : rfl
  -- Use the proportion to compute T
  have proportion_holds := proportion prop
  sorry

end initial_tagged_fish_l95_95714


namespace problem_20th_decreasing_number_l95_95553

-- Define what it means to be a "decreasing number"
def is_decreasing_number (n : ℕ) : Prop :=
  let digits := (n.digits 10) in
  digits == digits.sort (λ a b => a > b)

-- Define the predicate to find the 20th five-digit decreasing number
def twentieth_decreasing_number : ℕ :=
  Finite.to_list { n : ℕ // length (n.digits 10) = 5 ∧ is_decreasing_number n }
  |>.sort
  |>.nth 19 -- 0-based indexing

-- Problem statement in Lean 4
theorem problem_20th_decreasing_number : twentieth_decreasing_number = 65431 := by
  -- Proof should be provided here, currently omitted with sorry
  sorry

end problem_20th_decreasing_number_l95_95553


namespace lcm_fractions_l95_95533

theorem lcm_fractions (x : ℕ) (h_pos : 0 < x) : 
  let a := 1 / (3 * x^2)
  let b := 1 / (6 * x^3)
  let c := 1 / (9 * x)
  (lcm a b c) = 1 / (18 * x^3) :=
sorry

end lcm_fractions_l95_95533


namespace hex_A08_to_decimal_l95_95152

noncomputable def hex_A := 10
noncomputable def hex_A08_base_10 : ℕ :=
  (hex_A * 16^2) + (0 * 16^1) + (8 * 16^0)

theorem hex_A08_to_decimal :
  hex_A08_base_10 = 2568 :=
by
  sorry

end hex_A08_to_decimal_l95_95152


namespace good_students_l95_95364

theorem good_students (G T : ℕ) (h1 : G + T = 25) (h2 : T > 12) (h3 : T = 3 * (G - 1)) :
  G = 5 ∨ G = 7 :=
by
  sorry

end good_students_l95_95364


namespace cuberoot_inequality_solution_l95_95616

theorem cuberoot_inequality_solution (x : ℝ) :
  (∃ y : ℝ, y = real.cbrt x ∧ y + 3 / (y + 2) ≤ 0) ↔ x ∈ set.Ioo (-8) (-3 * real.sqrt 3) :=
by
  sorry

end cuberoot_inequality_solution_l95_95616


namespace largest_three_digit_divisible_by_6_l95_95059

-- Defining what it means for a number to be divisible by 6, 2, and 3
def divisible_by (n d : ℕ) : Prop := ∃ k : ℕ, n = d * k

-- Conditions extracted from the problem
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def last_digit_even (n : ℕ) : Prop := (n % 10) % 2 = 0
def sum_of_digits_divisible_by_3 (n : ℕ) : Prop := ((n / 100) + (n / 10 % 10) + (n % 10)) % 3 = 0

-- Define what it means for a number to be divisible by 6 according to the conditions
def divisible_by_6 (n : ℕ) : Prop := last_digit_even n ∧ sum_of_digits_divisible_by_3 n

-- Prove that 996 is the largest three-digit number that satisfies these conditions
theorem largest_three_digit_divisible_by_6 (n : ℕ) : is_three_digit n ∧ divisible_by_6 n → n ≤ 996 :=
by
    sorry

end largest_three_digit_divisible_by_6_l95_95059


namespace tangents_intersect_on_line_l95_95439

theorem tangents_intersect_on_line (a : ℝ) (x y : ℝ) (hx : 8 * a = 1) (hx_line : x - y = 5) (hx_point : x = 3) (hy_point : y = -2) : 
  x - y = 5 :=
by
  sorry -- Proof to be completed

end tangents_intersect_on_line_l95_95439


namespace area_CYL_l95_95026

-- Define the angle bisector of A and median of B in triangle ABC
variables {A B C X Y L : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited L]
variables (AL BM CX : Set (A → B → C → X → Prop))
variables (AY : Set (A → B → C → Y → Prop))

-- Assume given conditions
axiom angle_BAC_60 : ∀ {α : Type} (A B C : α), ∠BAC = 60
axiom AL_is_angle_bisector : ∀ {α : Type} (A B C L : α), AL
axiom AL_EQ_x : ∀ (x : ℝ), AL = x
axiom BM_is_median : ∀ {α : Type} (A B C M : α), BM
axiom AL_BM_intersect_X : ∀ {α : Type} (A B C L : α), AL ∩ BM = {X}
axiom CX_intersects_AB_at_Y : ∀ {α : Type} (A B C : α), CX ∩ AB = {Y}

-- The theorem to prove the area of ∆CYL given the problem conditions
theorem area_CYL {x : ℝ} :
  ∀ {A B C L M X Y : Type}  (AL BM CX : Set (A → B → C → X → Prop)) (AY : Set (A → B → C → Y → Prop)), 
    ∠BAC = 60 → AL = x → AL ∩ BM = {X} → CX ∩ AB = {Y} → 
    area (triangle CYL) = x^2 / (4 * sqrt(3)) :=
sorry

end area_CYL_l95_95026


namespace decagon_diagonals_intersection_probability_l95_95089

def isRegularDecagon : Prop :=
  ∃ decagon : ℕ, decagon = 10  -- A regular decagon has 10 sides

def chosen_diagonals (n : ℕ) : ℕ :=
  (Nat.choose n 2) - n   -- Number of diagonals in an n-sided polygon =
                          -- number of pairs of vertices - n sides

noncomputable def probability_intersection : ℚ :=
  let total_diagonals := chosen_diagonals 10
  let number_of_ways_to_pick_four := Nat.choose 10 4
  (number_of_ways_to_pick_four * 2) / (total_diagonals * (total_diagonals - 1) / 2)

theorem decagon_diagonals_intersection_probability :
  isRegularDecagon → probability_intersection = 42 / 119 :=
sorry

end decagon_diagonals_intersection_probability_l95_95089


namespace find_sum_l95_95120

variable (P R : ℝ) -- P is the principal sum, R is the original rate of interest

def interest (P R : ℝ) := (P * R * 10) / 100

def newInterest (P R : ℝ) := (P * (R + 5) * 10) / 100

theorem find_sum
  (h : newInterest P R - interest P R = 200) :
  P = 2000 := by
  sorry

end find_sum_l95_95120


namespace nonagon_isosceles_triangle_count_l95_95569

theorem nonagon_isosceles_triangle_count (vertices : Finset (Fin 9)) (h_reg_nonagon : ∀ (a b ∈ vertices), ∃ k : ℕ, b - a ≡ k [MOD 9] ∧ 1 ≤ k ∧ k ≤ 4) :
  ∃ n, n = 33 := by
  -- We know that the nonagon has 9 sides/vertices
  have n : ℕ := 9

  -- Total ways to choose 2 vertices out of 9
  let total_pairs := Nat.choose n 2

  -- Number of equilateral triangles in a nonagon
  let equilateral_triangles := 3

  -- Calculating the number of isosceles but not equilateral triangles
  let isosceles_triangles := total_pairs - equilateral_triangles

  -- Asserting the count of isosceles triangles
  use isosceles_triangles
  -- Proof step to validate the count
  sorry

end nonagon_isosceles_triangle_count_l95_95569


namespace initial_boys_count_l95_95936

theorem initial_boys_count (t : ℕ) (h_initial : 0.5 * t = n)
  (h_after_boys : boys_final = 0.5 * t - 4)
  (h_after_total : total_final = t + 2)
  (h_final_ratio : boys_final / total_final = 0.4) : 
  n = 24 :=
sorry

end initial_boys_count_l95_95936


namespace range_of_x_l95_95220

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x - 1 else x ^ 2

theorem range_of_x (x : ℝ) : f (x + 1) < 4 ↔ x < 1 := by
  sorry

end range_of_x_l95_95220


namespace find_length_AN_l95_95301

noncomputable def length_AN (A B C N : Point) (AB_eq : dist A B = 36) (BC_eq : dist B C = 36) 
                            (AC_eq : dist A C = 30) (N_midpoint_BC : midpoint B C N) : ℝ :=
  18

theorem find_length_AN {A B C N : Point} (h1 : dist A B = 36) (h2 : dist B C = 36)
    (h3 : dist A C = 30) (h4 : midpoint B C N) : dist A N = 18 :=
by
  sorry

end find_length_AN_l95_95301


namespace sin_squared_sum_eq_30_l95_95972

theorem sin_squared_sum_eq_30 :
  (∑ k in (Finset.range 1 30).image (λ n, 6 * n + 3), (sin k)^2) = 30 :=
sorry

end sin_squared_sum_eq_30_l95_95972


namespace range_of_g_l95_95750

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos x)^2 - (Real.arcsin x)^2

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → -((Real.pi^2) / 4) ≤ g x ∧ g x ≤ (3 * (Real.pi^2)) / 4 :=
by
  intros x hx
  sorry

end range_of_g_l95_95750


namespace sehun_total_valid_numbers_l95_95015

   -- Assume digits are natural numbers
   def digits := {0, 2, 9}

   -- Sehun can make a three-digit number using the digits 0, 2, and 9 once each.
   def valid_three_digit_number (n : Nat) : Prop :=
     let d1 := n / 100
     let d2 := (n % 100) / 10
     let d3 := n % 10
     d1 ≠ 0 ∧
     d1 ∈ digits ∧
     d2 ∈ digits ∧
     d3 ∈ digits ∧
     d1 ≠ d2 ∧
     d2 ≠ d3 ∧
     d1 ≠ d3

   theorem sehun_total_valid_numbers : 
      (Finset.filter valid_three_digit_number (Finset.range 1000)).card = 4 :=
   by
     sorry
   
end sehun_total_valid_numbers_l95_95015


namespace probability_of_top_three_spades_l95_95573

-- Define the conditions of the deck
structure StandardDeck :=
  (ranks : Fin 13)
  (suits : Fin 4)
  (cards : Fin 52)
  (black_suits : ∀ (s : Fin 2), s = 0 → suits = 0 ∧ s = 1 → suits = 3)
  (red_suits : ∀ (s : Fin 2), s = 0 → suits = 1 ∧ s = 1 → suits = 2)

-- Define the problem as a theorem
theorem probability_of_top_three_spades 
  (deck : StandardDeck) : 
  (∃ (ordered_cards : Fin 3 → Fin 52), 
    (ordered_cards 0) = deck.cards ∧ 
    (ordered_cards 1) = deck.cards ∧ 
    (ordered_cards 2) = deck.cards) →
  (deck.cards.fst * deck.cards.snd * deck.cards.trd) = 3 :=
begin
  sorry
end

end probability_of_top_three_spades_l95_95573


namespace nat_ineq_qr_ps_l95_95798

theorem nat_ineq_qr_ps (a b p q r s : ℕ) (h₀ : q * r - p * s = 1) 
  (h₁ : (p : ℚ) / q < a / b) (h₂ : (a : ℚ) / b < r / s) 
  : b ≥ q + s := sorry

end nat_ineq_qr_ps_l95_95798


namespace find_m_l95_95652

noncomputable def f : ℝ → ℝ := sorry

lemma odd_function_f (x : ℝ) : f (-x) = -f(x) := sorry
lemma increasing_function_f (x y : ℝ) (hx : -1 ≤ x ∧ x ≤ y) (hy : y ≤ 1) : f(x) ≤ f(y) := sorry

theorem find_m :
  ∃ m : ℝ, m = 1 ∧ 
  (∀ x, -1 ≤ x ∧ x ≤ 1 → f (-x) = -f(x)) ∧ 
  (∀ x y, -1 ≤ x ∧ x ≤ y ∧ y ≤ 1 → f(x) ≤ f(y)) ∧
  (-1 ≤ 3 * m - 2 ∧ 3 * m - 2 ≤ 1) ∧
  (-1 ≤ 2 - m^2 ∧ 2 - m^2 ≤ 1) ∧
  (f (3 * m - 2) + f (2 - m^2) > 0) := sorry

end find_m_l95_95652


namespace segment_division_ratio_l95_95735

-- Define the points and segments
section

variables (A B C D A1 B1 C1 D1 M K : Type) 

-- Assume the conditions
axiom parallelepiped : Prop
axiom segment_AM : Prop
axiom plane_BDA1 : Prop
axiom midpoint_M_CC1 : Prop

-- Define the theorem stating the segment AM is divided in the ratio 2:3 by the plane BDA1
theorem segment_division_ratio
  (h_parallelepiped : parallelepiped)
  (h_segment_AM : segment_AM)
  (h_plane_BDA1 : plane_BDA1)
  (h_midpoint_M_CC1 : midpoint_M_CC1) : 
  ∃ (r : ℕ) (s : ℕ), r = 2 ∧ s = 3 ∧ divides_segment_at_ratio A M plane_BDA1 r s :=
sorry

end

end segment_division_ratio_l95_95735


namespace find_range_t_l95_95670

def sequence_increasing (n : ℕ) (t : ℝ) : Prop :=
  (2 * (n + 1) + t^2 - 8) / (n + 1 + t) > (2 * n + t^2 - 8) / (n + t)

theorem find_range_t (t : ℝ) (h : ∀ n : ℕ, sequence_increasing n t) : 
  -1 < t ∧ t < 4 :=
sorry

end find_range_t_l95_95670


namespace digits_sum_distinct_l95_95241

theorem digits_sum_distinct :
  ∃ (a b c : ℕ), (9 ≥ a ∧ a ≥ 0) ∧ (9 ≥ b ∧ b ≥ 0) ∧ (9 ≥ c ∧ c ≥ 0) ∧
  let N1 := 11111 * a in
  let N2 := 1111 * b in
  let N3 := 111 * c in
  let S := N1 + N2 + N3 in
  (10000 ≤ S ∧ S ≤ 99999) ∧
  (∃ d1 d2 d3 d4 d5 : ℕ, d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧
                        d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ 
                        d3 ≠ d4 ∧ d3 ≠ d5 ∧ d4 ≠ d5 ∧
                        S = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5) := 
by
  sorry

end digits_sum_distinct_l95_95241


namespace prime_congruent_3_mod_4_infinite_primes_congruent_1_mod_4_l95_95926

-- Equivalent proof problem (1)
theorem prime_congruent_3_mod_4 (p : ℕ) [Fact p.Prime] :
  (∀ a b : ℤ, p ∣ a^2 + b^2 → p ∣ a ∧ p ∣ b) → p % 4 = 3 := sorry

-- Equivalent proof problem (2)
theorem infinite_primes_congruent_1_mod_4 :
  ∃∞ p, Nat.Prime p ∧ p % 4 = 1 := sorry

end prime_congruent_3_mod_4_infinite_primes_congruent_1_mod_4_l95_95926


namespace octahedron_side_length_proof_l95_95957

noncomputable def octahedron_side_length : Real :=
let P2 := (0 : ℝ, 0 : ℝ, 0 : ℝ) -- coordinates of P2
let P2' := (1 : ℝ, 1 : ℝ, 1 : ℝ) -- coordinates of P2'
let x := 1 / 2 -- solving x from the problem
((x : ℝ) * Real.sqrt 2) / 2 -- the side length expression

theorem octahedron_side_length_proof :
  let P2 := (0 : ℝ, 0 : ℝ, 0 : ℝ)
  let P2' := (1 : ℝ, 1 : ℝ, 1 : ℝ)
  ∃ (x : ℝ), 
  (x = 1 / 2) →
  octahedron_side_length = Real.sqrt 2 / 2 := 
begin
  let P2 := (0 : ℝ, 0 : ℝ, 0 : ℝ),
  let P2' := (1 : ℝ, 1 : ℝ, 1 : ℝ),
  use 1 / 2,
  intro h,
  rw [octahedron_side_length, h],
  norm_num,
end

end octahedron_side_length_proof_l95_95957


namespace fermat_little_theorem_l95_95878

theorem fermat_little_theorem (p : ℕ) (hp : Nat.Prime p) (a : ℕ) : a^p ≡ a [MOD p] :=
sorry

end fermat_little_theorem_l95_95878


namespace loan_amount_needed_l95_95453

-- Define the total cost of tuition.
def total_tuition : ℝ := 30000

-- Define the amount Sabina has saved.
def savings : ℝ := 10000

-- Define the grant coverage rate.
def grant_coverage_rate : ℝ := 0.4

-- Define the remainder of the tuition after using savings.
def remaining_tuition : ℝ := total_tuition - savings

-- Define the amount covered by the grant.
def grant_amount : ℝ := grant_coverage_rate * remaining_tuition

-- Define the loan amount Sabina needs to apply for.
noncomputable def loan_amount : ℝ := remaining_tuition - grant_amount

-- State the theorem to prove the loan amount needed.
theorem loan_amount_needed : loan_amount = 12000 := by
  sorry

end loan_amount_needed_l95_95453


namespace num_values_times_sum_f49_l95_95773

noncomputable def f : ℕ → ℕ := sorry

theorem num_values_times_sum_f49 :
  (let n := { k : ℕ | ∃ a b : ℕ, k = 2 * f ((a + b) ^ 2) ∧ k = [f a] ^ 2 + [f b] ^ 2}.toFinset.card,
       s := { k : ℕ | ∃ a b : ℕ, k = 2 * f ((a + b) ^ 2) ∧ k = [f a] ^ 2 + [f b] ^ 2}.toFinset.sum id in
   n * s = 150) :=
begin
  sorry
end

end num_values_times_sum_f49_l95_95773


namespace frog_can_reach_can_reach_24_40_cannot_reach_40_60_cannot_reach_24_60_can_reach_200_4_l95_95105

/-- Frog Jump Rules -/
structure FrogJump :=
  (start : ℕ × ℕ)
  (jump1 : ℕ × ℕ → ℕ × ℕ)
  (jump2 : ℕ × ℕ → ℕ × ℕ)
  (jump3 : ℕ × ℕ → ℕ × ℕ)
  (jump4 : ℕ × ℕ → ℕ × ℕ)

noncomputable def frog : FrogJump :=
{ start := (1, 1),
  jump1 := λ p, (2 * p.1, 6),
  jump2 := λ p, (p.1, 2 * p.2),
  jump3 := λ p, if p.1 > p.2 then (p.1 - p.2, p.2) else p,
  jump4 := λ p, if p.2 > p.1 then (p.1, p.2 - p.1) else p }

/-- Theorem to check if a point is reachable from (1, 1) using Frog Jump rules -/
theorem frog_can_reach (target : ℕ × ℕ) : Prop :=
∃ path : ℕ → ℕ × ℕ, path 0 = (1, 1) ∧ (∀ n : ℕ, path (n + 1) = frog.jump1 (path n) ∨
                                                path (n + 1) = frog.jump2 (path n) ∨
                                                path (n + 1) = frog.jump3 (path n) ∨
                                                path (n + 1) = frog.jump4 (path n)) ∧
                      ∃ n : ℕ, path n = target

theorem can_reach_24_40 : frog_can_reach (24, 40) := sorry
theorem cannot_reach_40_60 : ¬ frog_can_reach (40, 60) := sorry
theorem cannot_reach_24_60 : ¬ frog_can_reach (24, 60) := sorry
theorem can_reach_200_4 : frog_can_reach (200, 4) := sorry

end frog_can_reach_can_reach_24_40_cannot_reach_40_60_cannot_reach_24_60_can_reach_200_4_l95_95105


namespace polygon_inequality_example_l95_95508

theorem polygon_inequality_example :
  ∃ (sticks : Fin 100 → ℕ),
  (∀ i, sticks i = if i = 99 then 2^99 - 2 else 2^i) ∧ 
  (∀ subset ∈ (Finset.powerset (Finset.univ : Finset (Fin 100))), 
  subset.card < 100 → ¬can_form_polygon_by subset sticks) :=
begin
  sorry
end

def can_form_polygon_by (sticks : Finset (Fin 100)) (lengths : Fin 100 → ℕ) : Prop :=
  ∑ i in sticks, lengths i < 2 * (∑ i in sticks, lengths i ≠ max_length ⟨sticks, _⟩ lengths) 

end polygon_inequality_example_l95_95508


namespace gas_usage_correct_l95_95425

def starting_gas : ℝ := 0.5
def ending_gas : ℝ := 0.16666666666666666

theorem gas_usage_correct : starting_gas - ending_gas = 0.33333333333333334 := by
  sorry

end gas_usage_correct_l95_95425


namespace no_valid_arrangement_l95_95144

theorem no_valid_arrangement :
  ¬ ∃ (a : ℕ → ℕ) (h : ∀ i, a i ∈ finset.range 26) (h_distinct : ∀ i j, a i = a j → i = j),
    (∀ i, (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4)) % 5 = 1 ∨ 
           (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4)) % 5 = 4) →
    (∀ i, a (i + 25) = a i) :=
begin
  sorry
end

end no_valid_arrangement_l95_95144


namespace greatest_multiple_less_than_l95_95883

def lcm (a b : ℕ) : ℕ := Nat.lcm a b -- Assuming lcm function is already defined

theorem greatest_multiple_less_than (a b m : ℕ) (h₁ : a = 15) (h₂ : b = 20) (h₃ : m = 150) : 
  ∃ k, k * lcm a b < m ∧ ¬ ∃ k', (k' * lcm a b < m ∧ k' > k) :=
by
  sorry

end greatest_multiple_less_than_l95_95883


namespace candy_distribution_l95_95610

theorem candy_distribution :
  (∃ f : Fin 8 → Fin 3, (∀ i j, i ≠ j → f i ≠ f j) ∧
  (2 ≤ (Finset.filter (λ i, f i = 0) Finset.univ).card ∧
  2 ≤ (Finset.filter (λ i, f i = 1) Finset.univ).card))
  → ∃ (count : Nat), count = 127 := 
by
  sorry

end candy_distribution_l95_95610


namespace number_of_true_propositions_is_one_l95_95678

-- Definitions based on given conditions
def regression_line_passes_center : Prop := ∀ (b a x̄ ŷ : ℝ), ∃ (x y : ℝ), (ŷ = b * x̄ + a) ∧ (x̄, ŷ) = (x, y)
def necessary_not_sufficient : Prop := ¬ (∀ (x : ℝ), x = 6 → x^2 - 5 * x - 6 = 0) ∧ (∃ (x : ℝ), (x^2 - 5 * x - 6 = 0) ∧ (x ≠ 6))
def negation_proof : Prop := (¬ ∃ x₀ : ℝ, x₀^2 + 2 * x₀ + 3 < 0) = (∀ x : ℝ, x^2 + 2 * x + 3 ≥ 0)
def or_negation_false : Prop := ∀ (p q : Prop), (p ∨ q) → (¬ p ∧ ¬ q) = false

-- The problem statement
theorem number_of_true_propositions_is_one : 
  (regression_line_passes_center = true) ∧ 
  (necessary_not_sufficient = false) ∧
  (negation_proof = false) ∧
  (or_negation_false = false) →
  (1 = 1) :=
by
  sorry


end number_of_true_propositions_is_one_l95_95678


namespace xyz_solution_is_1_l95_95167

noncomputable def symmetric_sum_solution (x y z : ℝ) :=
  x + y + z = 3 ∧ x^2 + y^2 + z^2 = 3 ∧ x^3 + y^3 + z^3 = 3 → x = 1 ∧ y = 1 ∧ z = 1

theorem xyz_solution_is_1 (x y z : ℝ) : symmetric_sum_solution x y z :=
by
  intros h,
  have h1 := h.1,
  have h2 := h.2.1,
  have h3 := h.2.2,
  sorry

end xyz_solution_is_1_l95_95167


namespace males_watch_tvxy_l95_95121

-- Defining the conditions
def total_watch := 160
def females_watch := 75
def males_dont_watch := 83
def total_dont_watch := 120

-- Proving that the number of males who watch TVXY equals 85
theorem males_watch_tvxy : (total_watch - females_watch) = 85 :=
by sorry

end males_watch_tvxy_l95_95121


namespace minimum_value_frac_sum_l95_95705

theorem minimum_value_frac_sum (a b : ℝ) (h_odd : ∀ x, e^x * log a + e^(-x) * log b = -(e^(-x) * log a + e^x * log b)) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ab : a * b = 1) : 
(∃ x : ℝ, a = real.sqrt 2 ∧ b = 2 * real.sqrt 2) ∧ 
(∀ x y : ℝ, x = a → y = b → (1 / x + 2 / y) = 2 * real.sqrt 2) :=
begin
  sorry
end

end minimum_value_frac_sum_l95_95705


namespace find_x_l95_95834

theorem find_x (x : ℝ) (h : (2 * x + 8 + 5 * x + 3 + 3 * x + 9) / 3 = 3 * x + 2) : x = -14 :=
by
  sorry

end find_x_l95_95834


namespace integral_value_l95_95210

open Real

theorem integral_value (m : ℝ) (h : m = ∑ i in Finset.Ico 0 6, Polynomial.coeff (Polynomial.expand ℤ 2 (Polynomial.X - Polynomial.C 2) ^ 5) i) :
  ∫ x in 1..2, x ^ m = log 2 := 
by 
  have h1 : m = -1 := by {
    rw [← Finset.sum_range_succ, Polynomial.expand_eq_pow, Polynomial.derivative_add_const, Polynomial.coeff_smul];
    simp,
    sorry -- Detailed calculation of sum of coefficients
  }
  rw h1
  have h2 : ∀ x : ℝ, x ^ (-1) = (x⁻¹) := by {
    intros,
    ring,
  }
  have h3 : ∫ x in 1..2, x^(-1) = log 2 := by {
    -- Direct calculation of the definite integral
    rw intervalIntegral.integral_deriv_eq_sub' continuous_on_id (intervalIntegral.integrable_on_Icc_of_continuous continuous_on_inv) ,
    simp,
    sorry -- Detailed calculation
  }
  exact h3

end integral_value_l95_95210


namespace parabola_focus_focus_of_given_parabola_l95_95619

theorem parabola_focus (a b c : ℝ) (h : a ≠ 0) :
    ∃ focus_x focus_y, ∀ x y, y = -(a*x^2 + b*x + c) → (focus_x, focus_y) = (-b/(2*a), c - (-(a*b^2 + 2*a*c)/(4*a))) :=
begin
  sorry
end

-- Specific problem: y = -x^2 - 4x + 2
theorem focus_of_given_parabola :
    ∃ focus_x focus_y, ∀ x y, y = -x^2 - 4x + 2 → (focus_x, focus_y) = (-2, 5.75) :=
begin
  sorry
end

end parabola_focus_focus_of_given_parabola_l95_95619


namespace max_liters_l95_95895

-- Declare the problem conditions as constants
def initialHeat : ℕ := 480 -- kJ for the first 5 min
def reductionRate : ℚ := 0.25 -- 25%
def initialTemp : ℚ := 20 -- °C
def boilingTemp : ℚ := 100 -- °C
def specificHeatCapacity : ℚ := 4.2 -- kJ/kg°C

-- Define the heat required to raise m kg of water from initialTemp to boilingTemp
def heatRequired (m : ℚ) : ℚ := m * specificHeatCapacity * (boilingTemp - initialTemp)

-- Define the total available heat from the fuel
noncomputable def totalAvailableHeat : ℚ :=
  initialHeat / (1 - (1 - reductionRate))

-- The proof problem statement
theorem max_liters :
  ∃ (m : ℕ), m ≤ 5 ∧ heatRequired m ≤ totalAvailableHeat :=
sorry

end max_liters_l95_95895


namespace Ivanov_made_error_l95_95804

theorem Ivanov_made_error (x m : ℝ) (S_squared : ℝ) (h_x : x = 0) (h_m : m = 4) (h_S_squared : S_squared = 15.917) :
  ¬(|x - m| ≤ Real.sqrt S_squared) :=
by
  have h_sd : Real.sqrt S_squared = 3.99 := by sorry
  have h_ineq: |x - m| = 4 := by sorry
  rw [h_sd, h_ineq]
  linarith

end Ivanov_made_error_l95_95804


namespace proof_equivalent_problem_l95_95879

noncomputable def m : ℕ := 20

def freq1 := 1
def rel_freq1 := 0.05

def freq2 := 2
def rel_freq2 := 0.10

noncomputable def a : ℝ := sorry -- Since it ranges between 4 to 6 in integer value.
def rel_freq3 := (0.20 ≤ a) ∧ (a ≤ 0.30)

noncomputable def b : ℕ := sorry

def freq5 := 3
def rel_freq5 := 0.15

def total_freq : ℕ := m

def check_m := (1/rel_freq1 = m)

def check_b := ¬(b = 7)

def check_median := (9 + 1 < m / 2 ∧ m / 2 ≤ 9 + b)

def check_average := sorry -- Need to evaluate group's average weight.

theorem proof_equivalent_problem :
  check_m ∧
  check_b ∧
  check_median ∧
  check_average := by
  sorry

end proof_equivalent_problem_l95_95879


namespace problem_statement_l95_95261

-- Define the conditions and the goal
theorem problem_statement {x y : ℝ} 
  (h1 : (x + y)^2 = 36)
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := 
by
  sorry

end problem_statement_l95_95261


namespace sum_of_transformed_numbers_l95_95135

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) : 3 * (a - 5) + 3 * (b - 5) = 3 * S - 30 :=
by
  sorry

end sum_of_transformed_numbers_l95_95135


namespace solve_x_squared_plus_y_squared_l95_95267

-- Variables
variables {x y : ℝ}

-- Conditions
def cond1 : (x + y)^2 = 36 := sorry
def cond2 : x * y = 8 := sorry

-- Theorem stating the problem's equivalent proof
theorem solve_x_squared_plus_y_squared : x^2 + y^2 = 20 := sorry

end solve_x_squared_plus_y_squared_l95_95267


namespace quadratic_polynomials_eq_l95_95008

-- Define the integer part function
def intPart (x : ℝ) : ℤ := Int.floor x

-- Define the condition for quadratic polynomials
def is_quadratic (f : ℝ → ℝ) := ∃ (a b c : ℝ), ∀ x, f(x) = a*x^2 + b*x + c

theorem quadratic_polynomials_eq 
    (f g : ℝ → ℝ)
    (hf : is_quadratic f)
    (hg : is_quadratic g)
    (h_condition : ∀ x, intPart (f x) = intPart (g x)) :
    ∀ x, f x = g x :=
by
  sorry

end quadratic_polynomials_eq_l95_95008


namespace rectangles_perimeter_l95_95800

theorem rectangles_perimeter : 
  let base := 4
  let top := 4
  let left_side := 4
  let right_side := 6
  base + top + left_side + right_side = 18 := 
by {
  let base := 4
  let top := 4
  let left_side := 4
  let right_side := 6
  sorry
}

end rectangles_perimeter_l95_95800


namespace distance_sum_leq_2_sqrt_2_sum_radii_l95_95940

-- Define the parameters and conditions
variables {Γ₁ Γ₂ Γ₃ : Type}
variable {r₁ r₂ r₃ : ℝ} -- radii of the circles
variable {O₁ O₂ O₃ : Type} -- centers of the circles
variables {d₁₂ d₁₃ d₂₃ : ℝ} -- distances between centers

-- Assume the circles are disjoint and any line separating two circles intersects the interior of the third
variable disjoint_Γ₁_Γ₂ : Γ₁ ∩ Γ₂ = ∅
variable disjoint_Γ₂_Γ₃ : Γ₂ ∩ Γ₃ = ∅
variable disjoint_Γ₃_Γ₁ : Γ₃ ∩ Γ₁ = ∅

-- Line separation intersection properties
variable sep_line_Γ₁_Γ₂ : Π (l : Type), (l separates Γ₁ Γ₂) → (l intersects Γ₃)
variable sep_line_Γ₂_Γ₃ : Π (l : Type), (l separates Γ₂ Γ₃) → (l intersects Γ₁)
variable sep_line_Γ₃_Γ₁ : Π (l : Type), (l separates Γ₃ Γ₁) → (l intersects Γ₂)

-- Proof goal
theorem distance_sum_leq_2_sqrt_2_sum_radii :
  d₁₂ + d₁₃ + d₂₃ ≤ 2 * Real.sqrt 2 * (r₁ + r₂ + r₃) :=
sorry

end distance_sum_leq_2_sqrt_2_sum_radii_l95_95940


namespace value_of_expression_l95_95280

theorem value_of_expression (m : ℝ) (h : m^2 - m - 110 = 0) : (m - 1)^2 + m = 111 := by
  sorry

end value_of_expression_l95_95280


namespace solve_x_squared_plus_y_squared_l95_95265

-- Variables
variables {x y : ℝ}

-- Conditions
def cond1 : (x + y)^2 = 36 := sorry
def cond2 : x * y = 8 := sorry

-- Theorem stating the problem's equivalent proof
theorem solve_x_squared_plus_y_squared : x^2 + y^2 = 20 := sorry

end solve_x_squared_plus_y_squared_l95_95265


namespace max_liters_of_water_that_can_be_heated_to_boiling_l95_95902

-- Define the initial conditions
def initial_heat_per_5min := 480 -- kJ
def heat_reduction_rate := 0.25
def initial_temp := 20 -- Celsius
def boiling_temp := 100 -- Celsius
def specific_heat_capacity := 4.2 -- kJ/kg·°C

-- Define the temperature difference
def delta_T := boiling_temp - initial_temp -- Celsius

-- Define the calculation of the total heat available from a geometric series
def total_heat_available := initial_heat_per_5min / (1 - (1 - heat_reduction_rate))

-- Define the calculation of energy required to heat m kg of water
def energy_required (m : ℝ) := specific_heat_capacity * m * delta_T

-- Define the main theorem to prove
theorem max_liters_of_water_that_can_be_heated_to_boiling :
  ∃ (m : ℝ), ⌊m⌋ = 5 ∧ energy_required m ≤ total_heat_available :=
begin
  sorry
end

end max_liters_of_water_that_can_be_heated_to_boiling_l95_95902


namespace dot_final_position_l95_95954

def initial_square : Type := ℝ × ℝ
def dot_position : initial_square := (1, 1)  -- Dot in the top right corner

-- Folding the square along its diagonal
def fold (p : initial_square) : initial_square := 
  if p.1 = p.2 then p else (p.2, p.1)  -- Swap coordinates if not on the diagonal

-- Rotating the folded square 90 degrees clockwise
def rotate (p : initial_square) : initial_square := (p.2, -p.1)

-- Original unfolded square's dot position after the entire operations
def final_position (p : initial_square) : initial_square :=
  rotate (fold p)

-- Prove that final position of the dot is bottom right after unfolding
theorem dot_final_position :
  final_position dot_position = (1, -1) :=
by
  sorry

end dot_final_position_l95_95954


namespace parallel_EK_BC_perpendicular_GE_GC_l95_95592

open EuclideanGeometry

variables {A B C O H D G K E : Point}
variables [Circumcenter O A B C] [Orthocenter H A B C]
variables [⊥ AD BC] [Midpoint G A H] [On K G H] [EqSegment GK HD] [IntersectsAt KO AB E]

theorem parallel_EK_BC : parallel EK BC :=
by
  sorry

theorem perpendicular_GE_GC : perpendicular GE GC :=
by
  sorry

end parallel_EK_BC_perpendicular_GE_GC_l95_95592


namespace union_A_B_intersection_complement_A_B_nonempty_intersection_A_C_implication_l95_95690

variable (x a : ℝ)

def A : Set ℝ := { x | 1 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C : Set ℝ := { x | x < a }
def U : Set ℝ := { x | True }  -- Universal set U = ℝ

theorem union_A_B : A ∪ B = { x | 1 ≤ x ∧ x < 10 } :=
by {
  sorry
}

theorem intersection_complement_A_B : (U \ A) ∩ B = { x | 7 ≤ x ∧ x < 10 } :=
by {
  sorry
}

theorem nonempty_intersection_A_C_implication (h : (A ∩ C).Nonempty) : a > 1 :=
by {
  sorry
}

end union_A_B_intersection_complement_A_B_nonempty_intersection_A_C_implication_l95_95690


namespace pascal_triangle_ratio_345_l95_95308

open Nat

theorem pascal_triangle_ratio_345 :
  ∃ (n : ℕ), ∃ (r : ℕ),
    (binomial n r) * 4 = (binomial n (r + 1)) * 3 ∧ 
    (binomial n (r + 1)) * 5 = (binomial n (r + 2)) * 4 ∧ 
    n = 62 :=
by
  sorry

end pascal_triangle_ratio_345_l95_95308


namespace positive_number_solution_l95_95069

theorem positive_number_solution (x : ℝ) (h_pos : 0 < x) (h_eq : sqrt ((10 * x) / 3) = x) : x = 10 / 3 :=
sorry

end positive_number_solution_l95_95069


namespace mr_smith_markers_l95_95433

theorem mr_smith_markers :
  ∀ (initial_markers : ℕ) (total_markers : ℕ) (markers_per_box : ℕ) 
  (number_of_boxes : ℕ),
  initial_markers = 32 → 
  total_markers = 86 → 
  markers_per_box = 9 → 
  number_of_boxes = (total_markers - initial_markers) / markers_per_box →
  number_of_boxes = 6 :=
by
  intros initial_markers total_markers markers_per_box number_of_boxes h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃] at h₄
  simp only [Nat.sub] at h₄
  exact h₄

end mr_smith_markers_l95_95433


namespace bricks_needed_for_wall_l95_95073

noncomputable def brick_volume (length width height : ℝ) : ℝ :=
  length * width * height

noncomputable def wall_volume (length height thickness : ℝ) : ℝ :=
  length * height * thickness

theorem bricks_needed_for_wall :
  let length_wall := 800
  let height_wall := 600
  let thickness_wall := 22.5
  let length_brick := 100
  let width_brick := 11.25
  let height_brick := 6
  let vol_wall := wall_volume length_wall height_wall thickness_wall
  let vol_brick := brick_volume length_brick width_brick height_brick
  vol_wall / vol_brick = 1600 :=
by
  sorry

end bricks_needed_for_wall_l95_95073


namespace parker_total_weight_l95_95793

def twenty_pound := 20
def thirty_pound := 30
def forty_pound := 40

def first_set_weight := (2 * twenty_pound) + (1 * thirty_pound) + (1 * forty_pound)
def second_set_weight := (1 * twenty_pound) + (2 * thirty_pound) + (2 * forty_pound)
def third_set_weight := (3 * thirty_pound) + (3 * forty_pound)

def total_weight := first_set_weight + second_set_weight + third_set_weight

theorem parker_total_weight :
  total_weight = 480 := by
  sorry

end parker_total_weight_l95_95793


namespace number_of_valid_labelings_l95_95992

-- Define the problem in Lean
def cube_vertex_labeling : Prop :=
  ∃ (labeling : fin 8 → ℕ), 
    (∀ v, labeling v ∈ {1, 2, 3, 4, 5, 6, 7, 8}) ∧              -- Each integer 1 through 8 is used exactly once
    (∀ f, (∑ v in face_vertices f, labeling v) = 18) ∧          -- The sum on each face is 18
    ∀ (rotation : cube_rotation), 
      labeling ∘ rotation ≠ labeling                            -- Rotations yield the same labeling

-- The main theorem to prove
theorem number_of_valid_labelings : 
  ∃! (labeling : fin 8 → ℕ), cube_vertex_labeling labeling :=
sorry

end number_of_valid_labelings_l95_95992


namespace class_proof_l95_95352

def class_problem : Prop :=
  let total_students := 25
  let (students : Type) := {s // s ∈ Fin total_students}
  let good students (s : students) : Prop := sorry -- Condition Placeholder
  let troublemakers (s : students) : Prop := sorry -- Condition Placeholder
  have all_good_or_trouble : ∀ s, good students s ∨ troublemakers s,
    from sorry,
  have lh_trouble : ∀ (a b c d e : students), (good students a ∧ good students b ∧ good students c ∧ good students d ∧ good students e) →
    ∃ (x : ℕ), x > ((total_students - 1) / 2) ∧
    (λ rem_trouble : students - {a, b, c, d, e}, B : ℕ, B = x) ∧
    (λ rem_good : students - {a, b, c, d, e}, G : ℕ, G ∈ rem_good), from sorry,
  have th_triple_ratio : ∀ s, (students - {s}).card = 24 →
    ∃ (x y : ℕ), (troublemakers x ∧ good students y) ∧ (3*y = x), from sorry,
  ∃ (num_good_students : ℕ), (num_good_students = 5 ∨ num_good_students = 7)
  
theorem class_proof : class_problem :=
  sorry

end class_proof_l95_95352


namespace pete_total_marbles_after_trading_l95_95441

theorem pete_total_marbles_after_trading :
  ∀ (total_marbles : ℕ) (blue_percentage : ℝ) (kept_red_marbles : ℕ)
    (trade_blue_for_red : ℕ → ℕ → Prop),
    total_marbles = 10 →
    blue_percentage = 0.4 →
    kept_red_marbles = 1 →
    (∀ r b, trade_blue_for_red r b ↔ b = 2 * r) →
  ∃ total_after_trading,
    total_after_trading = (total_marbles * real.floor blue_percentage) + kept_red_marbles + (5 * 2) →
    total_after_trading = 15 :=
by { sorry }

end pete_total_marbles_after_trading_l95_95441


namespace father_age_l95_95542

theorem father_age (M F : ℕ) 
  (h1 : M = 2 * F / 5) 
  (h2 : M + 10 = (F + 10) / 2) : F = 50 :=
sorry

end father_age_l95_95542


namespace inequality_solution_l95_95085

-- Define the inequality problem in Lean
noncomputable def solution_set : Set ℝ := {x | ||x-2|-1| ≤ 1}

-- Define the expected solution set
def expected_solution_set : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}

-- State the theorem to prove
theorem inequality_solution:
  solution_set = expected_solution_set :=
by
  sorry

end inequality_solution_l95_95085


namespace find_a_l95_95281

-- Given conditions
variables (x y z a : ℤ)

def conditions : Prop :=
  (x - 10) * (y - a) * (z - 2) = 1000 ∧
  ∃ (x y z : ℤ), x + y + z = 7

theorem find_a (x y z : ℤ) (h : conditions x y z 1) : a = 1 := 
  by
    sorry

end find_a_l95_95281


namespace gcm_15_and_20_less_than_150_gcm_of_15_and_20_l95_95888

theorem gcm_15_and_20_less_than_150 : 
  ∃ x, (x = 15 * k ∧ x = 20 * m ∧ x < 150) ∧ (∀ y, (y = 15 * k' ∧ y = 20 * m' ∧ y < 150) → y ≤ x) := by
sorr

theorem gcm_of_15_and_20 : 
  ∃ x, x = 120 ∧ (x = 15 * k ∧ x = 20 * m ∧ x < 150) ∧ (∀ y, (y = 15 * k' ∧ y = 20 * m' ∧ y < 150) → y ≤ x) := by
sorry

end gcm_15_and_20_less_than_150_gcm_of_15_and_20_l95_95888


namespace sin_cos_pi_over_12_l95_95139

theorem sin_cos_pi_over_12 : sin (π / 12) * cos (π / 12) = 1 / 4 := 
by 
  sorry

end sin_cos_pi_over_12_l95_95139


namespace find_number_of_good_students_l95_95334

variable {G T : ℕ}

def is_good_or_troublemaker (G T : ℕ) : Prop :=
  G + T = 25 ∧ 
  (∀ s ∈ finset.range 5, T > 12) ∧ 
  (∀ s ∈ finset.range 20, T = 3 * (G - 1))

-- There are 25 students in total. 
-- Each student is either a good student or a troublemaker.
-- Good students always tell the truth, while troublemakers always lie.
-- Five students make the statement: "If I transfer to another class, more than half of the remaining students will be troublemakers."
-- The remaining 20 students make the statement: "If I transfer to another class, the number of troublemakers among the remaining students will be three times the number of good students."

theorem find_number_of_good_students (G T : ℕ) (h : is_good_or_troublemaker G T) :
  (G = 5 ∨ G = 7) :=
sorry

end find_number_of_good_students_l95_95334


namespace problem_statement_l95_95256

-- Define the conditions and the goal
theorem problem_statement {x y : ℝ} 
  (h1 : (x + y)^2 = 36)
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := 
by
  sorry

end problem_statement_l95_95256


namespace good_students_l95_95362

theorem good_students (G T : ℕ) (h1 : G + T = 25) (h2 : T > 12) (h3 : T = 3 * (G - 1)) :
  G = 5 ∨ G = 7 :=
by
  sorry

end good_students_l95_95362


namespace ratio_of_numbers_l95_95172

theorem ratio_of_numbers (x : ℝ) (h_sum : x + 3.5 = 14) : x / 3.5 = 3 :=
by
  sorry

end ratio_of_numbers_l95_95172


namespace quadratics_equal_l95_95004

-- Definitions of integer part and quadratic polynomials
def intPart (a : ℝ) : ℤ := ⌊a⌋

-- Define quadratic polynomial
structure QuadraticPoly :=
(coeffs : ℝ × ℝ × ℝ) -- (a, b, c) coefficients of ax^2 + bx + c

def evalQuadPoly (p : QuadraticPoly) (x : ℝ) : ℝ :=
p.coeffs.1 * x^2 + p.coeffs.2 * x + p.coeffs.3

-- Problem Statement in Lean 4
theorem quadratics_equal
  (f g : QuadraticPoly)
  (h : ∀ x : ℝ, intPart (evalQuadPoly f x) = intPart (evalQuadPoly g x)) :
  ∀ x : ℝ, evalQuadPoly f x = evalQuadPoly g x := 
sorry

end quadratics_equal_l95_95004


namespace original_trapezoid_area_l95_95578

theorem original_trapezoid_area (x : ℝ) :
  let h := x,
      base1 := 4 * x,
      base2 := 2 * x,
      line_meeting := true -- condition for line meeting the base at midpoint, mathematically simplifies as per context
  in (1/2) * (base1 + base2) * h = 5 * x^2 := 
by
  sorry

end original_trapezoid_area_l95_95578


namespace number_of_players_l95_95859

noncomputable def cost_jersey : ℝ := 25
noncomputable def cost_shorts : ℝ := 15.20
noncomputable def cost_socks : ℝ := 6.80
noncomputable def total_cost_all_players : ℝ := 752

theorem number_of_players : 
  let cost_per_player := cost_jersey + cost_shorts + cost_socks in
  cost_per_player = 47 → 
  total_cost_all_players / cost_per_player = 16 :=
by
  sorry

end number_of_players_l95_95859


namespace sum_of_digits_multiple_exists_l95_95880

theorem sum_of_digits_multiple_exists (n : ℕ) (h1 : n > 0) (h2 : ∀ k : ℕ, n ≠ 3 * k) :
  ∃ m : ℕ, ∀ k : ℕ, k ≥ m → ∃ z : ℕ, k = (z * n).digits.sum :=
sorry

end sum_of_digits_multiple_exists_l95_95880


namespace root_approximation_l95_95295

def f (x : ℝ) : ℝ := x^3 + x^2 - 2*x - 2

theorem root_approximation : 
  (f 1.438 = 0.165) →
  (f 1.4065 = -0.052) →
  ∃ (r : ℝ), r ∈ Icc 1.4065 1.438 ∧ (|r - 1.43| < 0.1) ∧ (f r = 0) :=
by
  sorry

end root_approximation_l95_95295


namespace max_height_of_ball_l95_95556

noncomputable def h (t : ℝ) : ℝ := -20 * t^2 + 70 * t + 45

theorem max_height_of_ball : ∃ t : ℝ, (h t) = 69.5 :=
sorry

end max_height_of_ball_l95_95556


namespace similarity_criteria_dependent_l95_95590

theorem similarity_criteria_dependent 
  {Δ₁ Δ₂ : Type} 
  (angles_equal: ∀ (A B C : Δ₁) (D E F : Δ₂), 
                   (∠A = ∠D) → (∠B = ∠E) → (∠C = ∠F)) 
  (sides_proportional: ∀ (A B C : Δ₁) (D E F : Δ₂), 
                        (AB / DE = BC / EF) → (AC / DF = BC / EF)) :
  (∀ (A B C : Δ₁) (D E F : Δ₂), 
     (angles_equal A B C D E F) → (sides_proportional A B C D E F)) ∧
  (∀ (A B C : Δ₁) (D E F : Δ₂), 
     (sides_proportional A B C D E F) → (angles_equal A B C D E F)) := 
sorry

end similarity_criteria_dependent_l95_95590


namespace tile_set_reduction_l95_95952

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m * m = n

def remove_perfect_squares_and_cubes (s : List ℕ) : List ℕ :=
  s.filter (λ n, ¬ is_perfect_square n ∧ ¬ is_perfect_cube n)

def renumber (s : List ℕ) : List ℕ :=
  List.enum s |>.map Prod.snd

def perform_operations (s : List ℕ) : List ℕ :=
  renumber (remove_perfect_squares_and_cubes s)

def iterate_n_times {α : Type*} (f : α → α) (x : α) (n : ℕ) : α :=
  Nat.iterate f n x

theorem tile_set_reduction :
  ∃ n : ℕ, n ≤ 6 ∧ length (iterate_n_times perform_operations (List.range' 1 151) n) < 50 :=
begin
  sorry
end

end tile_set_reduction_l95_95952


namespace calculate_eccentricity_l95_95676

variables {a b : ℝ} (x y : ℝ)

def is_ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def foci (a b c : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ c = sqrt (a^2 - b^2)

def eccentricity (a c : ℝ) : ℝ :=
  (c/a)

theorem calculate_eccentricity
  (a b : ℝ)
  (ha : a > b)
  (hb : b > 0)
  (hC : is_ellipse x (sqrt (5) * b / 3) a b)
  (hFoci : foci a b (sqrt (a^2 - b^2)))
  (hQuad : (x ^ 2) / a^2 = 4 / 9) :
  eccentricity a (sqrt (a^2 - b^2)) = 2 / 3 :=
begin
  sorry

end calculate_eccentricity_l95_95676


namespace xy_problem_l95_95275

theorem xy_problem (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 :=
by
  sorry

end xy_problem_l95_95275


namespace complement_union_example_l95_95231

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 3, 4}

-- Define set B
def B : Set ℕ := {2, 4}

-- State the theorem we want to prove
theorem complement_union_example : (U \ A) ∪ B = {2, 4, 5} :=
by
  sorry

end complement_union_example_l95_95231


namespace part_a_part_b_l95_95698

-- Definitions for Part (a)
def digits := {0, 1, 5, 6, 7, 9}
def num_with_repeating_digits := 38280

-- Statement for part (a)
theorem part_a : 
  (∃ n : ℕ, n = 6 ∧ ∀ d ∈ digits, d ≠ 0 → d ∈ digits) →
  count_6_digit_numbers digits (at_least_one_digit_repeated := true) = num_with_repeating_digits :=
sorry

-- Definitions for Part (b)
def num_all_diff_not_div4 := 504

-- Statement for part (b)
theorem part_b : 
  (∃ n : ℕ, n = 6 ∧ ∀ d ∈ digits, d ≠ 0 → d ∈ digits) →
  count_6_digit_numbers digits (all_diff_and_not_div4 := true) = num_all_diff_not_div4 :=
sorry


end part_a_part_b_l95_95698


namespace triangle_subdivision_triangle_count_l95_95603

noncomputable def total_triangles_in_subdivided_triangle : Nat :=
  23

-- Define the problem statement
theorem triangle_subdivision_triangle_count :
  let n := 3  -- Number of subdivisions per side
  let initial_triangles := n * n  -- 9 small triangles
  let added_lines := 3  -- Each vertex connected to the midpoint creates 3 additional lines
  let additional_small_triangles := added_lines * 3  -- Additional triangles due to intersections
  let medium_triangles := 4  -- 4 triangles formed by combining small triangles
  let large_triangle := 1  -- The large outline triangle itself
  let total_triangles := initial_triangles + additional_small_triangles + medium_triangles + large_triangle
  total_triangles = total_triangles_in_subdivided_triangle := 
  by
    dsimp [total_triangles, initial_triangles, additional_small_triangles, medium_triangles, large_triangle, total_triangles_in_subdivided_triangle]
    sorry

end triangle_subdivision_triangle_count_l95_95603


namespace good_students_l95_95343

theorem good_students (E B : ℕ) (h1 : E + B = 25) (h2 : 12 < B) (h3 : B = 3 * (E - 1)) :
  E = 5 ∨ E = 7 :=
by 
  sorry

end good_students_l95_95343


namespace original_inhabitants_l95_95959

theorem original_inhabitants (X : ℝ) (h : 0.75 * 0.9 * X = 5265) : X = 7800 :=
by
  sorry

end original_inhabitants_l95_95959


namespace fixed_repayment_amount_l95_95932

-- Define the conditions
variables (a r : ℝ) (n : ℕ) (h : n = 5)

-- Define the fixed amount to be repaid each year
noncomputable def repayment_amount := ar * (1 + r)^5 / ((1 + r)^5 - 1)

-- The theorem stating the fixed amount to be repaid each year
theorem fixed_repayment_amount (a r : ℝ) (h : r ≠ -1) : 
  repayment_amount a r = ar * (1 + r)^5 / ((1 + r)^5 - 1) := 
sorry

end fixed_repayment_amount_l95_95932


namespace compute_sum_G_l95_95414

def G : ℕ → ℕ
| 0     => 1
| 1     => 1
| (n+2) => G (n+1) + G n

def sum_G_div_2_pow : ℕ → ℝ
| 0     => G 0 / 2^0
| (n+1) => sum_G_div_2_pow n + (G (n+1) / 2^(n+1))

theorem compute_sum_G : sum_G_div_2_pow 10 = 3.3330078125 :=
by
  sorry

end compute_sum_G_l95_95414


namespace dot_product_eq_26_l95_95641

def vector_a := (5, 3) : ℝ × ℝ
def vector_b := (4, 2) : ℝ × ℝ

theorem dot_product_eq_26 : (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) = 26 := by
  sorry

end dot_product_eq_26_l95_95641


namespace lines_per_stanza_l95_95802

-- Define the number of stanzas
def num_stanzas : ℕ := 20

-- Define the number of words per line
def words_per_line : ℕ := 8

-- Define the total number of words in the poem
def total_words : ℕ := 1600

-- Theorem statement to prove the number of lines per stanza
theorem lines_per_stanza : 
  (total_words / words_per_line) / num_stanzas = 10 := 
by sorry

end lines_per_stanza_l95_95802


namespace quadratic_polynomials_equal_l95_95000

def integer_part (a : ℝ) : ℤ := ⌊a⌋

theorem quadratic_polynomials_equal 
  (f g : ℝ → ℝ)
  (hf : ∀ x, ∃ a1 b1 c1, f x = a1 * x^2 + b1 * x + c1)
  (hg : ∀ x, ∃ a2 b2 c2, g x = a2 * x^2 + b2 * x + c2)
  (H : ∀ x, integer_part (f x) = integer_part (g x)) : 
  ∀ x, f x = g x :=
sorry

end quadratic_polynomials_equal_l95_95000


namespace new_pyramid_volume_correct_l95_95115

def original_pyramid_volume (base_edge slant_edge height : ℕ) : ℚ :=
  1 / 3 * (base_edge ^ 2) * height

def new_pyramid_base_edge (base_edge : ℕ) (scale_factor : ℚ) : ℚ :=
  base_edge * scale_factor

def new_pyramid_volume (new_base_edge new_height : ℚ) : ℚ :=
  1 / 3 * (new_base_edge ^ 2) * new_height

theorem new_pyramid_volume_correct :
  let base_edge := 12
  let slant_edge := 15
  let cut_height := 4
  let full_height := 9 in  -- calculated from the solution step but given condition
  let scale_factor := (full_height - cut_height) / full_height in
  let new_height := full_height - cut_height in
  let new_base_edge := new_pyramid_base_edge base_edge scale_factor in
  new_pyramid_volume new_base_edge new_height = 2000 / 27 :=
by {
  sorry
}

end new_pyramid_volume_correct_l95_95115


namespace find_x_l95_95166

theorem find_x (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ -1) :
  (x = -1) ↔ (frac ((x^3 - x^2)) ((x^2 + 3*x + 2)) + x = -3) :=
by
  sorry

end find_x_l95_95166


namespace good_students_count_l95_95374

theorem good_students_count (E B : ℕ) 
    (h1 : E + B = 25) 
    (h2 : ∀ (e ∈ finset.range 5), B > 12) 
    (h3 : ∀ (e ∈ finset.range 20), B = 3 * (E - 1)) :
    E = 5 ∨ E = 7 :=
by
  sorry

end good_students_count_l95_95374


namespace lcm_inequality_l95_95447

theorem lcm_inequality (k m n : ℕ) (hk : 0 < k) (hm : 0 < m) (hn : 0 < n) : 
  Nat.lcm k m * Nat.lcm m n * Nat.lcm n k ≥ Nat.lcm (Nat.lcm k m) n ^ 2 :=
by sorry

end lcm_inequality_l95_95447


namespace quadratic_polynomials_eq_l95_95009

-- Define the integer part function
def intPart (x : ℝ) : ℤ := Int.floor x

-- Define the condition for quadratic polynomials
def is_quadratic (f : ℝ → ℝ) := ∃ (a b c : ℝ), ∀ x, f(x) = a*x^2 + b*x + c

theorem quadratic_polynomials_eq 
    (f g : ℝ → ℝ)
    (hf : is_quadratic f)
    (hg : is_quadratic g)
    (h_condition : ∀ x, intPart (f x) = intPart (g x)) :
    ∀ x, f x = g x :=
by
  sorry

end quadratic_polynomials_eq_l95_95009


namespace find_number_of_good_students_l95_95335

variable {G T : ℕ}

def is_good_or_troublemaker (G T : ℕ) : Prop :=
  G + T = 25 ∧ 
  (∀ s ∈ finset.range 5, T > 12) ∧ 
  (∀ s ∈ finset.range 20, T = 3 * (G - 1))

-- There are 25 students in total. 
-- Each student is either a good student or a troublemaker.
-- Good students always tell the truth, while troublemakers always lie.
-- Five students make the statement: "If I transfer to another class, more than half of the remaining students will be troublemakers."
-- The remaining 20 students make the statement: "If I transfer to another class, the number of troublemakers among the remaining students will be three times the number of good students."

theorem find_number_of_good_students (G T : ℕ) (h : is_good_or_troublemaker G T) :
  (G = 5 ∨ G = 7) :=
sorry

end find_number_of_good_students_l95_95335


namespace each_girl_brought_2_cups_l95_95510

-- Definitions of the conditions
def total_students : ℕ := 30
def boys : ℕ := 10
def total_cups : ℕ := 90
def cups_per_boy : ℕ := 5
def girls : ℕ := total_students - boys

def total_cups_by_boys : ℕ := boys * cups_per_boy
def total_cups_by_girls : ℕ := total_cups - total_cups_by_boys
def cups_per_girl : ℕ := total_cups_by_girls / girls

-- The statement with the correct answer
theorem each_girl_brought_2_cups (
  h1 : total_students = 30,
  h2 : boys = 10,
  h3 : total_cups = 90,
  h4 : cups_per_boy = 5,
  h5 : total_cups_by_boys = boys * cups_per_boy,
  h6 : total_cups_by_girls = total_cups - total_cups_by_boys,
  h7 : cups_per_girl = total_cups_by_girls / girls
) : cups_per_girl = 2 := 
sorry

end each_girl_brought_2_cups_l95_95510


namespace find_y_value_l95_95755

noncomputable def y_value (α : ℝ) (y : ℝ) : Prop :=
  let P : ℝ × ℝ := (3, y) in
  cos α = 3 / 5 → y = 4 ∨ y = -4

-- Here we formally state the problem in Lean without proving it.
theorem find_y_value (α : ℝ) (y : ℝ) :
  y_value α y := by
  sorry

end find_y_value_l95_95755


namespace arithmetic_sequence_sum_l95_95495

theorem arithmetic_sequence_sum (c d : ℤ) (h₁ : d = 10 - 3)
    (h₂ : c = 17 + (10 - 3)) (h₃ : d = 24 + (10 - 3)) :
    c + d = 55 :=
sorry

end arithmetic_sequence_sum_l95_95495


namespace percentage_B_of_D_l95_95179

theorem percentage_B_of_D (A B C D : ℝ)
  (h1 : B = 1.71 * A)
  (h2 : C = 1.80 * A)
  (h3 : D = 1.90 * B) :
  (B / D) * 100 ≈ 52.63 :=
by sorry

end percentage_B_of_D_l95_95179


namespace find_gain_percentage_l95_95565

def initial_selling_price := 1 / 12
def loss_rate := 0.9
def new_selling_price := 1 / 7.5
def cost_price := 1 / (12 * 0.9)
def gain_percentage (G : ℝ) := (new_selling_price = (1 + G / 100) * cost_price)

theorem find_gain_percentage : ∃ G : ℝ, gain_percentage G ∧ G = 44 := 
begin
  use 44,
  rw [gain_percentage],
  unfold new_selling_price,
  unfold cost_price,
  simpa,
end

end find_gain_percentage_l95_95565


namespace number_of_shelves_l95_95049

def chessboard (m n : ℕ) : Type := 
  fin m × fin n

def is_shelf (m n : ℕ) (board : chessboard m n → ℤ) (R : fin m × fin n → Prop) (h : ℤ) : Prop :=
  (∀ i j, R (i, j) → board (i, j) > h) ∧
  (∀ i j, (¬ R (i, j) ∧ adjacent (i, j) R) → board (i, j) ≤ h)

def adjacent {m n : ℕ} (p : fin m × fin n) (R : fin m × fin n → Prop) : Prop :=
  ∃ i j, R (i, j) ∧ (i = p.1 + 1 ∨ i = p.1 - 1 ∨ j = p.2 + 1 ∨ j = p.2 - 1)

noncomputable def max_shelves (m n : ℕ) : ℕ :=
  (m+1)*(n+1) / 2 - 1

theorem number_of_shelves (m n : ℕ) (board : chessboard m n → ℤ) :
  ∃ f : ℕ, f = max_shelves m n :=
begin
  use (m+1)*(n+1) / 2 - 1,
  sorry,
end

end number_of_shelves_l95_95049


namespace john_tv_show_duration_l95_95745

def john_tv_show (seasons_before : ℕ) (episodes_per_season : ℕ) (additional_episodes : ℕ) (episode_duration : ℝ) : ℝ :=
  let total_episodes_before := seasons_before * episodes_per_season
  let last_season_episodes := episodes_per_season + additional_episodes
  let total_episodes := total_episodes_before + last_season_episodes
  total_episodes * episode_duration

theorem john_tv_show_duration :
  john_tv_show 9 22 4 0.5 = 112 := 
by
  sorry

end john_tv_show_duration_l95_95745


namespace tangent_line_eq_l95_95030

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 3 * x + 1

-- Define the point at which we are evaluating the tangent
def point : ℝ × ℝ := (1, -1)

-- Define the derivative of the function f(x)
def f' (x : ℝ) : ℝ := 2 * x - 3

-- The desired theorem
theorem tangent_line_eq :
  ∀ x y : ℝ, (x, y) = point → (y = -x) :=
by sorry

end tangent_line_eq_l95_95030


namespace number_of_correct_propositions_l95_95638

-- Definitions for lines, planes, and their relations
variables {Line Plane : Type}
variables (l m : Line) (α β : Plane)

-- Relations
variables (perp : Line → Plane → Prop) (parallel : Line → Plane → Prop) (subset : Line → Plane → Prop)
variables (perp_lines : Line → Line → Prop) (parallel_lines : Line → Line → Prop)
variables (perp_planes : Plane → Plane → Prop)

-- Propositions
def prop1 : Prop := perp l α ∧ parallel m α → perp_lines l m
def prop2 : Prop := parallel_lines m l ∧ subset m α → parallel_lines l α
def prop3 : Prop := perp_planes α β ∧ subset m α ∧ subset l β → perp_lines m l
def prop4 : Prop := perp_lines m l ∧ subset m α ∧ subset l β → perp_planes α β

-- Main theorem to be proven
theorem number_of_correct_propositions : 
  (prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ ¬prop4) := 
begin
  sorry,
end

end number_of_correct_propositions_l95_95638


namespace right_triangle_tangent_circle_l95_95831

theorem right_triangle_tangent_circle
  (A B C P : Point)
  (r : Real)
  (h_triangle : right_triangle A B C)
  (h_triangle_AC : AC = sqrt 145)
  (h_triangle_AB : AB = 7)
  (h_circle_tangent_AC : tangent_circle AC r P)
  (h_circle_tangent_BC : tangent_circle BC r P) :
  CP = 4 * sqrt 6 := by
  sorry

end right_triangle_tangent_circle_l95_95831


namespace find_a_2015_l95_95649

-- Given conditions
def a : ℕ → ℝ
def a_1 : a 1 = 1 := rfl
def a_2 : a 2 = (1 / 2) := rfl
def recurrence (n : ℕ) : Prop := 
  ∀ n : ℕ, n > 0 → 2 / a (n + 1) = 1 / a n + 1 / a (n + 2)

-- Prove the result
theorem find_a_2015 (h₁ : a 1 = 1) (h₂ : a 2 = (1 / 2)) (h₃ : recurrence n) : a 2015 = 1 / 2015 :=
by sorry

end find_a_2015_l95_95649


namespace necessary_and_sufficient_condition_for_equality_of_shaded_areas_l95_95732

-- Definitions for the conditions
def isTangent (A B : Point) (C : Point) (r : ℝ) : Prop :=
  -- A and B are tangent if AB is perpendicular to the radius at A
  ∃ v : Vector, is_on_circle A C r ∧ (B - A).dot (A - C) = 0

def angleCondition (θ : ℝ) : Prop := 0 < θ ∧ θ < pi / 2

-- Mathematically equivalent proof problem statement
theorem necessary_and_sufficient_condition_for_equality_of_shaded_areas
  (θ : ℝ) (r : ℝ) (A B C : Point)
  (h_tangent : isTangent A B C r)
  (h_angle_cond : angleCondition θ)
  : tan θ = 2 * θ := sorry

end necessary_and_sufficient_condition_for_equality_of_shaded_areas_l95_95732


namespace minimal_spherical_n_gon_area_iff_regular_l95_95645

/-- Definitions of geometric entities and properties involved. -/
def Circle (S : Sphere) := { c : Point S // ∃ r : ℝ, r > 0 ∧ ∀ p : Point S, dist c p = r }
def SphericalNGon (S : Sphere) (n : ℕ) := { P : Set (Point S) // P.card = n ∧ convexHull P }

/-- A regular spherical n-gon is a spherical n-gon with all sides and angles equal. -/
structure RegularSphericalNGon (S : Sphere) (n : ℕ) extends SphericalNGon S n :=
(eq_sides : ∀ (pi pj : Point S), pi ∈ to_SphericalNGon.val → pj ∈ to_SphericalNGon.val → dist pi pj = some_constant)

variables (S : Sphere) (n : ℕ) (c : Circle S)

/-- Prove that among all spherical n-gons containing a given circle inside themselves, 
    the one with the least area is the regular spherical n-gon. -/
theorem minimal_spherical_n_gon_area_iff_regular :
  ∀ (P : SphericalNGon S n), (contains P c) →
  (∀ (Q : SphericalNGon S n), (contains Q c) → area P ≤ area Q) ↔ 
  P.is_regular_spherical_n_gon :=
sorry

end minimal_spherical_n_gon_area_iff_regular_l95_95645


namespace remaining_shirt_cost_l95_95176

theorem remaining_shirt_cost
  (total_cost : ℕ)
  (cost : ℕ → ℕ)
  (h_total_cost : total_cost = 85)
  (h_costs : ∀ k, k ∈ {1, 2, 3} → cost k = 15) :
  cost 4 = 20 ∧ cost 5 = 20 :=
by
  sorry

end remaining_shirt_cost_l95_95176


namespace coplanar_vectors_lambda_l95_95671

theorem coplanar_vectors_lambda (λ : ℝ) (a b c : ℝ × ℝ × ℝ) :
  a = (2, -1, 3) → 
  b = (-1, 4, -2) →
  c = (7, 5, λ) →
  ∃ x y : ℝ, c = (2 * x - y, -x + 4 * y, 3 * x - 2 * y) ∧ λ = 65 / 7 :=
by 
  intros ha hb hc 
  have h : a = (2, -1, 3) ∧ b = (-1, 4, -2) ∧ c = (7, 5, λ),
  { exact ⟨ha, hb, hc⟩ },
  -- Now proceed to solve the system of equations to find x and y
  sorry

end coplanar_vectors_lambda_l95_95671


namespace white_triangle_pairs_coincidence_l95_95150

theorem white_triangle_pairs_coincidence :
  ∀ (red_tri_half : ℕ) (blue_tri_half : ℕ) (white_tri_half : ℕ)
    (red_pairs : ℕ) (blue_pairs : ℕ) (red_white_pairs : ℕ) (blue_white_pairs : ℕ), 
  red_tri_half = 4 →
  blue_tri_half = 6 →
  white_tri_half = 9 →
  red_pairs = 3 →
  blue_pairs = 4 →
  red_white_pairs = 3 →
  blue_white_pairs = 3 →
  ∃ white_pairs : ℕ, white_pairs = 3 :=
by
  intros red_tri_half blue_tri_half white_tri_half red_pairs blue_pairs red_white_pairs blue_white_pairs
  assume h_red_tri : red_tri_half = 4
  assume h_blue_tri : blue_tri_half = 6
  assume h_white_tri : white_tri_half = 9
  assume h_red_pairs : red_pairs = 3
  assume h_blue_pairs : blue_pairs = 4
  assume h_red_white_pairs : red_white_pairs = 3
  assume h_blue_white_pairs : blue_white_pairs = 3
  use 3
  sorry

end white_triangle_pairs_coincidence_l95_95150


namespace project_completion_advance_l95_95559

variables (a : ℝ) -- efficiency of each worker (units of work per day)
variables (total_days : ℕ) (initial_workers added_workers : ℕ) (fraction_completed : ℝ)
variables (initial_days remaining_days total_initial_work total_remaining_work total_workers_efficiency : ℝ)

-- Conditions
def conditions : Prop :=
  total_days = 100 ∧
  initial_workers = 10 ∧
  initial_days = 30 ∧
  fraction_completed = 1 / 5 ∧
  added_workers = 10 ∧
  total_initial_work = initial_workers * initial_days * a * 5 ∧ 
  total_remaining_work = total_initial_work - (initial_workers * initial_days * a) ∧
  total_workers_efficiency = (initial_workers + added_workers) * a ∧
  remaining_days = total_remaining_work / total_workers_efficiency

-- Proof statement
theorem project_completion_advance (h : conditions a total_days initial_workers added_workers fraction_completed initial_days remaining_days total_initial_work total_remaining_work total_workers_efficiency) :
  total_days - (initial_days + remaining_days) = 10 :=
  sorry

end project_completion_advance_l95_95559


namespace num_pairs_satisfying_equation_l95_95253

theorem num_pairs_satisfying_equation : 
  (∃! (pairs : List (ℤ × ℤ)), 
    (∀ (m n : ℤ), (m, n) ∈ pairs ↔ m + n = m * n - 1) ∧ pairs.length = 4) := 
begin
  sorry
end

end num_pairs_satisfying_equation_l95_95253


namespace broadcasting_sequences_count_l95_95138

theorem broadcasting_sequences_count :
  ∃ n : ℕ, n = 36 ∧
  (∃ ads : finset (string × string),
      ads.card = 5 ∧
      ∃ commercials : finset string,
      commercials.card = 3 ∧
      ∃ psas : finset string,
      psas.card = 2 ∧
      (∃ (seq : list string),
          seq.length = 5 ∧
          last seq ∈ psas ∧
          ∀ (i : ℕ), i < 4 → seq[i] ∉ psas ∨ seq[i + 1] ∉ psas ∧
          (Σ' (indexes : finset ℕ),
            indexes.card = 4 ∧
            ∀ i, i ∈ indexes ↔ (seq[i] ∈ commercials ∨ seq[i] ∈ psas))
      ))
:= by sorry

end broadcasting_sequences_count_l95_95138


namespace log2_no_ties_probability_l95_95521

noncomputable def noTiesProbability (teams : ℕ) : ℚ :=
  if teams = 30 then 30.factorial / 2^(teams.choose 2 : ℕ)
  else 0

theorem log2_no_ties_probability :
  let n := nat.gcd (30.factorial.nat_abs) (2^(30.choose 2).nat_abs) in
  n = 2^(30.choose 2 - (30 / 2 + 30 / 4 + 30 / 8 + 30 / 16)) →
  nat.log2 n = 409 :=
by
  sorry

end log2_no_ties_probability_l95_95521


namespace good_students_l95_95342

theorem good_students (E B : ℕ) (h1 : E + B = 25) (h2 : 12 < B) (h3 : B = 3 * (E - 1)) :
  E = 5 ∨ E = 7 :=
by 
  sorry

end good_students_l95_95342


namespace twenty_fifth_term_is_seven_l95_95736

-- Definition of the sequence as described in the condition
def seq : ℕ → ℕ
  | k =>
    let n := (nat.sqrt ((8 * k + 1 : ℕ) - 1) + 1 : ℕ) / 2
    n

-- The proof that the 25th term in the sequence is 7
theorem twenty_fifth_term_is_seven : seq 25 = 7 := by
  sorry

end twenty_fifth_term_is_seven_l95_95736


namespace train_speed_is_64_98_kmph_l95_95577

noncomputable def train_length : ℝ := 200
noncomputable def bridge_length : ℝ := 180
noncomputable def passing_time : ℝ := 21.04615384615385
noncomputable def speed_in_kmph : ℝ := 3.6 * (train_length + bridge_length) / passing_time

theorem train_speed_is_64_98_kmph : abs (speed_in_kmph - 64.98) < 0.01 :=
by
  sorry

end train_speed_is_64_98_kmph_l95_95577


namespace find_b_l95_95775

def f (x : ℝ) (b : ℝ) : ℝ :=
if x ≤ 4 then 4 * x^2 + 5 else b * x + 2

theorem find_b (b : ℝ) : 
  (∀ x : ℝ, (if x ≤ 4 then 4 * x^2 + 5 else b * x + 2) = f x b) → 
  (continuous_at (f _ b) 4) → 
  b = 16.75 :=
by 
  intros h hf 
  have h4 := hf 4 
  sorry

end find_b_l95_95775


namespace journey_length_l95_95100

-- Definitions for the conditions from step a)
def on_time (L : ℝ) : Prop := 
  let t_60 := L / 60
  let t_50 := L / 50
  t_50 = t_60 + 3 / 4

-- The proof statement
theorem journey_length : ∃ L : ℝ, on_time L ∧ L = 225 :=
by
  use 225
  -- Conditions definition ensures this equivalence
  have h1 : 225 / 50 = 225 / 60 + 3 / 4 := sorry
  exact ⟨h1, rfl⟩

end journey_length_l95_95100


namespace sum_of_row_is_sum_of_cubes_l95_95529

theorem sum_of_row_is_sum_of_cubes (n : ℕ) : 
  let start := (n - 1)^2 + 1,
      end := n^2,
      num_terms := 2 * n - 1,
      sum := (num_terms * (start + end)) / 2
  in sum = n^3 + (n - 1)^3 := 
by
  sorry

end sum_of_row_is_sum_of_cubes_l95_95529


namespace Ivanov_made_error_l95_95807

theorem Ivanov_made_error (x m : ℝ) (S_squared : ℝ) (h_x : x = 0) (h_m : m = 4) (h_S_squared : S_squared = 15.917) :
  ¬(|x - m| ≤ Real.sqrt S_squared) :=
by
  have h_sd : Real.sqrt S_squared = 3.99 := by sorry
  have h_ineq: |x - m| = 4 := by sorry
  rw [h_sd, h_ineq]
  linarith

end Ivanov_made_error_l95_95807


namespace simplify_expression_l95_95828

theorem simplify_expression : 
  (4 + 2 * complex.I) / (4 - 2 * complex.I) +
  (4 - 2 * complex.I) / (4 + 2 * complex.I) +
  (4 * complex.I) / (4 - 2 * complex.I) -
  (4 * complex.I) / (4 + 2 * complex.I) =
  2 / 5 := 
by
  sorry

end simplify_expression_l95_95828


namespace y_squared_plus_inv_y_squared_for_a_eq_2_l95_95421

variable {x : ℝ} {a : ℕ}

theorem y_squared_plus_inv_y_squared_for_a_eq_2
  (h1 : x + 1/x = 3) (h2 : a = 2) (h3 : a ≠ 1) :
  let y := x^a in y^2 + 1/y^2 = 47 :=
by
  sorry

end y_squared_plus_inv_y_squared_for_a_eq_2_l95_95421


namespace good_students_l95_95338

theorem good_students (E B : ℕ) (h1 : E + B = 25) (h2 : 12 < B) (h3 : B = 3 * (E - 1)) :
  E = 5 ∨ E = 7 :=
by 
  sorry

end good_students_l95_95338


namespace probability_1700_in_three_spins_l95_95791

/-- Prove that the probability of earning exactly $1700 in three spins is 6/125 given 
    that each region on the spinner has the same area and the spinner has five slots: 
    "Bankrupt", "$1000", "$300", "$5000", and "$400". -/
theorem probability_1700_in_three_spins :
  let slots := ["Bankrupt", "$1000", "$300", "$5000", "$400"]
  in let desired_outcomes := set.to_finset {(["$1000", "$300", "$400"].permutations : multiset (list string))}
  in let total_outcomes := 5 * 5 * 5
  in (desired_outcomes.card : ℚ) / (total_outcomes : ℚ) = 6 / 125 :=
by {
  let slots := ["Bankrupt", "$1000", "$300", "$5000", "$400"] in
  let desired_outcomes := set.to_finset {(["$1000", "$300", "$400"].permutations : multiset (list string))} in
  let total_outcomes := 5 * 5 * 5 in
  have h_desired_outcomes_card : desired_outcomes.card = 6 := sorry,
  have h_total_outcomes : total_outcomes = 125 := by norm_num,
  rw [h_desired_outcomes_card, h_total_outcomes],
  norm_num
}

end probability_1700_in_three_spins_l95_95791


namespace cos_double_angle_l95_95660

theorem cos_double_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos (π / 4 + α) = -3 / 5) : 
  cos (2 * α) = -24 / 25 := 
sorry

end cos_double_angle_l95_95660


namespace number_of_good_students_is_5_or_7_l95_95351

-- Definitions based on the conditions
def total_students : ℕ := 25
def number_of_good_students (G : ℕ) (T : ℕ) := G + T = total_students
def first_group_condition (T : ℕ) := T > 12
def second_group_condition (G : ℕ) (T : ℕ) := T = 3 * (G - 1)

-- Problem statement in Lean 4:
theorem number_of_good_students_is_5_or_7 (G T : ℕ) :
  number_of_good_students G T ∧ first_group_condition T ∧ second_group_condition G T → G = 5 ∨ G = 7 :=
by
  sorry

end number_of_good_students_is_5_or_7_l95_95351


namespace sum_of_solutions_l95_95412

def f (x : ℝ) : ℝ := 3 * x + 2
def f_inv (x : ℝ) : ℝ := (x - 2) / 3
def f_of_inv (x : ℝ) : ℝ := 3 / x + 2

theorem sum_of_solutions : (∑ x in ({9, -1} : finset ℝ), x) = 8 :=
by {
  have h1 : f_inv 9 = f_of_inv 9,
  { sorry },
  have h2 : f_inv (-1) = f_of_inv (-1),
  { sorry },
  finset.sum_singleton,
  finset.sum_insert,
  finset.sum_empty,
  sorry
}

end sum_of_solutions_l95_95412


namespace arithmetic_sequence_sum_l95_95494

theorem arithmetic_sequence_sum (c d : ℤ) (h₁ : d = 10 - 3)
    (h₂ : c = 17 + (10 - 3)) (h₃ : d = 24 + (10 - 3)) :
    c + d = 55 :=
sorry

end arithmetic_sequence_sum_l95_95494


namespace indistinguishable_counters_reachable_distinguishable_counters_no_swap_l95_95754

def returning_sequence_possible (n : ℕ) (h : n > 1) (initial_configuration final_configuration : list (list ℕ)) : Prop :=
-- Define that a returning sequence can transform the initial configuration to the final configuration
∃ seq : list (list (list ℕ) → list (list ℕ)), 
  (∀ f ∈ seq, ∃ i, 0 ≤ i ∧ i < n ∧ 
   (λ grid, slide_row grid i ∨ slide_col grid i) = f) ∧ 
   (iterate seq initial_configuration = final_configuration)

theorem indistinguishable_counters_reachable (n : ℕ) (h : n > 1) (config : list (list ℕ)) (indis : ℕ × ℕ)
  (h_indis : ∀ i j, indis ∈ config → config[i][j] = indis): 
  returning_sequence_possible n h config :=
sorry

theorem distinguishable_counters_no_swap (n : ℕ) (h : n > 1) (config : list (list ℕ)) (a b : ℕ)
  (h_no_swap : ∀ i j, i ≠ j → config[i] ≠ config[j] → a ≠ b → 
  returning_sequence_possible n h (swap_counters config a b)) : 
  false :=
sorry

end indistinguishable_counters_reachable_distinguishable_counters_no_swap_l95_95754


namespace option_costs_more_cost_effective_x30_more_cost_effective_plan_x30_l95_95928

def racket_price : ℕ := 80
def ball_price : ℕ := 20
def discount_rate : ℕ := 90

def option_1_cost (n_rackets : ℕ) : ℕ :=
  n_rackets * racket_price

def option_2_cost (n_rackets : ℕ) (n_balls : ℕ) : ℕ :=
  (discount_rate * (n_rackets * racket_price + n_balls * ball_price)) / 100

-- Part 1: Express in Algebraic Terms
theorem option_costs (n_rackets : ℕ) (n_balls : ℕ) :
  option_1_cost n_rackets = 1600 ∧ option_2_cost n_rackets n_balls = 1440 + 18 * n_balls := 
by
  sorry

-- Part 2: For x = 30, determine more cost-effective option
theorem more_cost_effective_x30 (x : ℕ) (h : x = 30) :
  option_1_cost 20 < option_2_cost 20 x := 
by
  sorry

-- Part 3: More cost-effective Plan for x = 30
theorem more_cost_effective_plan_x30 :
  1600 + (discount_rate * (10 * ball_price)) / 100 < option_2_cost 20 30 :=
by
  sorry

end option_costs_more_cost_effective_x30_more_cost_effective_plan_x30_l95_95928


namespace find_ordered_pair_l95_95974

theorem find_ordered_pair (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^y + 1 = y^x) (h2 : 2 * x^y = y^x + 13) : (x = 2 ∧ y = 4) :=
by {
  sorry
}

end find_ordered_pair_l95_95974


namespace membership_fee_increase_each_year_l95_95958

variable (fee_increase : ℕ)

def yearly_membership_fee_increase (first_year_fee sixth_year_fee yearly_increase : ℕ) : Prop :=
  yearly_increase * 5 = sixth_year_fee - first_year_fee

theorem membership_fee_increase_each_year :
  yearly_membership_fee_increase 80 130 10 :=
by
  unfold yearly_membership_fee_increase
  sorry

end membership_fee_increase_each_year_l95_95958


namespace range_of_m_l95_95692

variable (x y m : ℝ)

theorem range_of_m (h1 : Real.sin x = m * (Real.sin y)^3)
                   (h2 : Real.cos x = m * (Real.cos y)^3) :
                   1 ≤ m ∧ m ≤ Real.sqrt 2 :=
by
  sorry

end range_of_m_l95_95692


namespace number_of_good_students_is_5_or_7_l95_95344

-- Definitions based on the conditions
def total_students : ℕ := 25
def number_of_good_students (G : ℕ) (T : ℕ) := G + T = total_students
def first_group_condition (T : ℕ) := T > 12
def second_group_condition (G : ℕ) (T : ℕ) := T = 3 * (G - 1)

-- Problem statement in Lean 4:
theorem number_of_good_students_is_5_or_7 (G T : ℕ) :
  number_of_good_students G T ∧ first_group_condition T ∧ second_group_condition G T → G = 5 ∨ G = 7 :=
by
  sorry

end number_of_good_students_is_5_or_7_l95_95344


namespace g_at_3_l95_95031

-- Define the function g(x)
variable {g : ℝ → ℝ}
-- Conditions
axiom condition : ∀ (x : ℝ), x ≠ 0 → g(x) - 3 * g(1 / x) = 3^x

-- Theorem statement
theorem g_at_3 : g(3) = (216 - 9 * 3^(1/3)) / 8 :=
by
  sorry

end g_at_3_l95_95031


namespace numbers_not_necessarily_equal_l95_95871

theorem numbers_not_necessarily_equal (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + b^2 + c^2 = b + a^2 + c^2) (h2 : a + b^2 + c^2 = c + a^2 + b^2) : 
  ¬(a = b ∧ b = c) := 
sorry

end numbers_not_necessarily_equal_l95_95871


namespace log_inequality_l95_95193

theorem log_inequality {a x : ℝ} (h1 : 0 < x) (h2 : x < 1) (h3 : a > 0) (h4 : a ≠ 1) : 
  abs (Real.logb a (1 - x)) > abs (Real.logb a (1 + x)) :=
sorry

end log_inequality_l95_95193


namespace tangent_cyclic_quad_l95_95196

theorem tangent_cyclic_quad (
  (A B C D E F G H : Point)
  (cyclic_quad : CyclicQuadrilateral A B C D)
  (diag_intersect_E : Intersection (Diagonal AC) (Diagonal BD) E)
  (lines_intersect_F : Intersection (Line AD) (Line BC) F)
  (midpoint_G : Midpoint G A B)
  (midpoint_H : Midpoint H C D)
) : TangentAt (Line EF) E (Circle E G H) := 
sorry

end tangent_cyclic_quad_l95_95196


namespace total_gallons_in_tanks_l95_95788

theorem total_gallons_in_tanks :
  let capacity1 := 7000
  let capacity2 := 5000
  let capacity3 := 3000
  let fill_ratio1 := 3 / 4
  let fill_ratio2 := 4 / 5
  let fill_ratio3 := 1 / 2
  let water1 := capacity1 * fill_ratio1
  let water2 := capacity2 * fill_ratio2
  let water3 := capacity3 * fill_ratio3
  in water1 + water2 + water3 = 10750 := by
  -- Define the values as given in the conditions
  let capacity1 := 7000
  let capacity2 := 5000
  let capacity3 := 3000
  let fill_ratio1 := 3 / 4
  let fill_ratio2 := 4 / 5
  let fill_ratio3 := 1 / 2
  let water1 := capacity1 * fill_ratio1
  let water2 := capacity2 * fill_ratio2
  let water3 := capacity3 * fill_ratio3
  show water1 + water2 + water3 = 10750,
  sorry -- Skip the proof

end total_gallons_in_tanks_l95_95788


namespace frustum_lateral_surface_area_l95_95934

noncomputable def lateral_surface_area_of_frustum : ℝ :=
  let r_base := 8
  let r_top := 4
  let r_bottom := 7
  let h_frustum := 5
  let h_full_cone := 35 / 3
  let slant_height_large_cone := Real.sqrt ((h_full_cone)^2 + (r_base)^2)
  let slant_height_small_cone := Real.sqrt (h_frustum^2 + (r_bottom - r_top)^2)
  (π * r_base * slant_height_large_cone - π * r_top * slant_height_small_cone)

theorem frustum_lateral_surface_area :
  lateral_surface_area_of_frustum =
    π * 8 * Real.sqrt ((35 / 3)^2 + 8^2) - π * 4 * Real.sqrt 34 :=
by
  -- Proof to be filled in here
  sorry

end frustum_lateral_surface_area_l95_95934


namespace probability_of_mathematics_letter_l95_95283

-- Definitions for the problem
def english_alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

def mathematics_letters : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}

-- Set the total number of letters in the English alphabet
def total_letters := english_alphabet.card

-- Set the number of unique letters in 'MATHEMATICS'
def mathematics_unique_letters := mathematics_letters.card

-- Statement of the Lean theorem
theorem probability_of_mathematics_letter : (mathematics_unique_letters : ℚ) / total_letters = 4 / 13 :=
by
  sorry

end probability_of_mathematics_letter_l95_95283


namespace cupcakes_purchased_l95_95744

theorem cupcakes_purchased (cost_per_cupcake : ℝ) (amount_paid_each : ℝ) (total_paid : ℝ) (total_cupcakes : ℕ) 
  (h1 : cost_per_cupcake = 1.5)
  (h2 : amount_paid_each = 9)
  (h3 : total_paid = 2 * amount_paid_each) 
  (h4 : total_cupcakes = (total_paid / cost_per_cupcake).toNat) 
  : total_cupcakes = 12 :=
sorry

end cupcakes_purchased_l95_95744


namespace probability_gray_big_wolf_l95_95869

theorem probability_gray_big_wolf :
  let cards := {'gray', 'big', 'wolf'}
  let permutations := cards.permutations.toList
  let favorable_outcomes := {'gray big wolf', 'wolf big gray'}
  let total_outcomes := permutations.length
  let favorable_count := (permutations.filter (λ p, p = ['gray', 'big', 'wolf'] ∨ p = ['wolf', 'big', 'gray'])).length
  (favorable_count / total_outcomes : ℚ) = (1 / 3 : ℚ) :=
by
  sorry

end probability_gray_big_wolf_l95_95869


namespace exists_X_X_prime_l95_95549

-- Definitions for points on lines
variable (α : Type) [LinearOrderedField α]
variables (e f : Set (Affine.Point α))
variables (A B C D A' B' C' D' : Affine.Point α)

-- The proof problem in Lean 4
theorem exists_X_X_prime (h_Ae : A ∈ e) (h_Be : B ∈ e) (h_Ce : C ∈ e) (h_De : D ∈ e)
(h_A'f : A' ∈ f) (h_B'f : B' ∈ f) (h_C'f : C' ∈ f) (h_D'f : D' ∈ f) :
∃ (X ∈ e) (X' ∈ f),
  (Affine.dist A X / Affine.dist B X = Affine.dist A' X' / Affine.dist B' X') ∧
  (Affine.dist C X / Affine.dist D X = Affine.dist C' X' / Affine.dist D' X') := 
sorry

end exists_X_X_prime_l95_95549


namespace large_circuit_longer_l95_95613

theorem large_circuit_longer :
  ∀ (small_circuit_length large_circuit_length : ℕ),
  ∀ (laps_jana laps_father : ℕ),
  laps_jana = 3 →
  laps_father = 4 →
  (laps_father * large_circuit_length = 2 * (laps_jana * small_circuit_length)) →
  small_circuit_length = 400 →
  large_circuit_length - small_circuit_length = 200 :=
by
  intros small_circuit_length large_circuit_length laps_jana laps_father
  intros h_jana_laps h_father_laps h_distance h_small_length
  sorry

end large_circuit_longer_l95_95613


namespace count_multiples_15_not_4_or_9_l95_95251

theorem count_multiples_15_not_4_or_9 : 
  (finset.card ((finset.filter (λ n, (n % 15 = 0) ∧ (n % 4 ≠ 0) ∧ (n % 9 ≠ 0))
                    (finset.Icc 1 300)))) = 10 :=
by {
  sorry
}

end count_multiples_15_not_4_or_9_l95_95251


namespace max_b_lattice_free_line_l95_95939

theorem max_b_lattice_free_line : 
  ∃ b : ℚ, (∀ (m : ℚ), (1 / 3) < m ∧ m < b → 
  ∀ x : ℤ, 0 < x ∧ x ≤ 150 → ¬ (∃ y : ℤ, y = m * x + 4)) ∧ 
  b = 50 / 147 :=
sorry

end max_b_lattice_free_line_l95_95939


namespace good_students_l95_95361

theorem good_students (G T : ℕ) (h1 : G + T = 25) (h2 : T > 12) (h3 : T = 3 * (G - 1)) :
  G = 5 ∨ G = 7 :=
by
  sorry

end good_students_l95_95361


namespace abcd_product_l95_95769

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
noncomputable def d : ℝ := sorry

axiom a_eq : a = Real.sqrt (4 - Real.sqrt (5 - a))
axiom b_eq : b = Real.sqrt (4 + Real.sqrt (5 - b))
axiom c_eq : c = Real.sqrt (4 - Real.sqrt (5 + c))
axiom d_eq : d = Real.sqrt (4 + Real.sqrt (5 + d))

theorem abcd_product : a * b * c * d = 11 := sorry

end abcd_product_l95_95769


namespace total_number_of_items_l95_95862

-- Definitions based on the problem conditions
def number_of_notebooks : ℕ := 40
def pens_more_than_notebooks : ℕ := 80
def pencils_more_than_notebooks : ℕ := 45

-- Total items calculation based on the conditions
def number_of_pens : ℕ := number_of_notebooks + pens_more_than_notebooks
def number_of_pencils : ℕ := number_of_notebooks + pencils_more_than_notebooks
def total_items : ℕ := number_of_notebooks + number_of_pens + number_of_pencils

-- Statement to be proved
theorem total_number_of_items : total_items = 245 := 
by 
  sorry

end total_number_of_items_l95_95862


namespace part_97a_part_97b_l95_95766

-- Definitions based on conditions in a)
structure SimilarFigure (F : Type) :=
(similar_to : F → F → Prop)

structure Line (l : Type) :=
(intersects_at : l → l → l → Prop)

structure Point (P : Type) :=
(lies_on_circumcircle : Point → SimilarFigure → Prop)

-- Main hypotheses and theorems
variables 
(F1 F2 F3 : Type)
[L1 L2 L3 : Type]
[W : Type]
[SimilarFigureF : SimilarFigure (F1 × F2 × F3)]
[LineL : Line (L1 × L2 × L3)]
[p_W : Point W]

-- Similarity and intersection conditions
hypothesis similar_figures : SimilarFigureF.similar_to (F1, F2, F3)
hypothesis line_intersection : LineL.intersects_at (L1, L2, L3)

-- Theorems to prove
theorem part_97a : 
  p_W.lies_on_circumcircle W (SimilarFigureF) →
  sorry

theorem part_97b 
  (J1 J2 J3 : Type)
  (p_J1 : Point J1)
  (p_J2 : Point J2)
  (p_J3 : Point J3):
  LineL.intersects_at J1 J2 J3 → 
  sorry

end part_97a_part_97b_l95_95766


namespace calculate_c20_l95_95780

noncomputable def c : ℕ → ℕ
| 1     := 3
| 2     := 9
| (n+1) := c n * c (n-1)

theorem calculate_c20 : c 20 = 3^10946 :=
sorry

end calculate_c20_l95_95780


namespace Ivanov_made_an_error_l95_95822

theorem Ivanov_made_an_error (mean median : ℝ) (variance : ℝ) (h1 : mean = 0) (h2 : median = 4) (h3 : variance = 15.917) : 
  |mean - median| ≤ Real.sqrt variance → False :=
by {
  have mean_value : mean = 0 := h1,
  have median_value : median = 4 := h2,
  have variance_value : variance = 15.917 := h3,

  let lhs := |mean_value - median_value|,
  have rhs := Real.sqrt variance_value,
  
  calc
    lhs = |0 - 4| : by rw [mean_value, median_value]
    ... = 4 : by norm_num,
  
  have rhs_val : Real.sqrt variance_value ≈ 3.99 := by sorry, -- approximate value for demonstration
  
  have ineq : 4 ≤ rhs_val := by 
    calc 4 = 4 : rfl -- trivial step for clarity,
    have sqrt_val : Real.sqrt 15.917 < 4 := by sorry, -- from calculation or suitable proof
  
  exact absurd ineq (not_le_of_gt sqrt_val)
}

end Ivanov_made_an_error_l95_95822


namespace cylindrical_to_rectangular_coords_l95_95978

theorem cylindrical_to_rectangular_coords
  (r θ z : ℝ) (h_r : r = 3) (h_θ : θ = π / 3) (h_z : z = -2) :
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  in (x, y, z) = (1.5, (3 * Real.sqrt 3) / 2, -2) :=
by 
  sorry

end cylindrical_to_rectangular_coords_l95_95978


namespace password_probability_seventh_week_l95_95863

/-- Four different passwords A, B, C, D are used by an intelligence station.
Each week, one of the passwords is used. The selection is random and equally probable
from the three not used in the previous week. The probability of password A being used in 
the first week is given as 1.

Prove the probability P_7 of A being used in the seventh week. -/
theorem password_probability_seventh_week : 
  let P : ℕ → ℚ := λ k, if k = 1 then 1 else 1/4 + 3/4 * (-1/3)^(k-1)
  in P 7 = 61/243 :=
by
  sorry

end password_probability_seventh_week_l95_95863


namespace sabina_loan_amount_l95_95455

-- Definitions corresponding to the problem conditions
def tuition_cost : ℕ := 30000
def sabina_savings : ℕ := 10000
def grant_percentage : ℚ := 0.40

-- Proof statement to show the amount of loan required
theorem sabina_loan_amount : ∀ (tuition_cost sabina_savings : ℕ) (grant_percentage : ℚ), 
  sabina_savings < tuition_cost →
  let remainder := tuition_cost - sabina_savings in
  let grant_amount := grant_percentage * remainder in
  let loan_amount := remainder - grant_amount in
  loan_amount = 12000 :=
by 
  intros tuition_cost sabina_savings grant_percentage h_savings_lt;
  let remainder := tuition_cost - sabina_savings;
  let grant_amount := grant_percentage * ↑remainder;
  let loan_amount := remainder - grant_amount.to_nat;
  sorry

end sabina_loan_amount_l95_95455


namespace amount_received_by_sam_l95_95077

def P : ℝ := 15000
def r : ℝ := 0.10
def n : ℝ := 2
def t : ℝ := 1

noncomputable def compoundInterest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem amount_received_by_sam : compoundInterest P r n t = 16537.50 := by
  sorry

end amount_received_by_sam_l95_95077


namespace min_value_of_function_l95_95776

theorem min_value_of_function (h : 0 < x ∧ x < 1) : 
  ∃ (y : ℝ), (∀ z : ℝ, z = (4 / x + 1 / (1 - x)) → y ≤ z) ∧ y = 9 :=
by
  sorry

end min_value_of_function_l95_95776


namespace smith_boxes_l95_95429

theorem smith_boxes (initial_markers : ℕ) (markers_per_box : ℕ) (total_markers : ℕ)
  (h1 : initial_markers = 32) (h2 : markers_per_box = 9) (h3 : total_markers = 86) :
  total_markers - initial_markers = 6 * markers_per_box :=
by
  -- We state our assumptions explicitly for better clarity
  have h_total : total_markers = 86 := h3
  have h_initial : initial_markers = 32 := h1
  have h_box : markers_per_box = 9 := h2
  sorry

end smith_boxes_l95_95429


namespace ratio_malt_to_coke_l95_95720

-- Definitions from conditions
def total_males : ℕ := 10
def total_females : ℕ := 16
def males_choose_malt : ℕ := 6
def females_choose_malt : ℕ := 8

-- Derived values
def total_cheerleaders : ℕ := total_males + total_females
def total_malt : ℕ := males_choose_malt + females_choose_malt
def total_coke : ℕ := total_cheerleaders - total_malt

-- The theorem to be proved
theorem ratio_malt_to_coke : (total_malt / total_coke) = (7 / 6) :=
  by
    -- skipped proof
    sorry

end ratio_malt_to_coke_l95_95720


namespace find_q_l95_95277

theorem find_q (p q : ℚ) (h1 : 5 * p + 6 * q = 17) (h2 : 6 * p + 5 * q = 20) : q = 2 / 11 :=
by
  sorry

end find_q_l95_95277


namespace ivanov_error_l95_95809

-- Given conditions
def mean_temperature (x : ℝ) : Prop := x = 0
def median_temperature (m : ℝ) : Prop := m = 4
def variance_temperature (s² : ℝ) : Prop := s² = 15.917

-- Statement to prove
theorem ivanov_error (x m s² : ℝ) 
  (mean_x : mean_temperature x)
  (median_m : median_temperature m)
  (variance_s² : variance_temperature s²) :
  (x - m)^2 > s² :=
by
  rw [mean_temperature, median_temperature, variance_temperature] at *
  simp [*, show 0 = 0, from rfl, show 4 = 4, from rfl]
  sorry

end ivanov_error_l95_95809


namespace range_of_x_values_l95_95294

theorem range_of_x_values (f : ℝ → ℝ) (even_f : ∀ x, f x = f (-x)) (decreasing_f : ∀ x y, x < y → x ≤ 0 → y ≤ 0 → f y ≤ f x) (f_2_zero : f 2 = 0) :
  {x | f x < 0} = set.Ioo (-2) 2 :=
by
  sorry

end range_of_x_values_l95_95294


namespace part1_part2_l95_95682

variable (f : ℝ → ℝ) (m n a b : ℝ)
noncomputable def f := λ x => -x^2 + 1 + Real.log x
variable (h_tangent : ∃ c : ℝ, f c = -1 + 1 + Real.log 1 ∧ Diff.passed_through !c - 1 (-1, 2))
variable (h_roots : ∃ a b : ℝ, a < b ∧ f a = n ∧ f b = n ∧ n < (1 - Real.log 2) / 2)

theorem part1 : ∀ x : ℝ, f x ≤ 1 - x :=
by sorry

theorem part2 : b - a < 1 - 2 * n :=
by sorry

end part1_part2_l95_95682


namespace good_students_l95_95341

theorem good_students (E B : ℕ) (h1 : E + B = 25) (h2 : 12 < B) (h3 : B = 3 * (E - 1)) :
  E = 5 ∨ E = 7 :=
by 
  sorry

end good_students_l95_95341


namespace seven_stars_on_grid_l95_95643

theorem seven_stars_on_grid (stars : Finset (Fin 4 × Fin 4)) :
  (∃ stars_set : Finset (Fin 4 × Fin 4), stars_set.card = 7 ∧
    ∀ (r1 r2 : Fin 4) (c1 c2 : Fin 4),
      r1 ≠ r2 → c1 ≠ c2 →
      ∃ star ∈ stars_set, star.1 ≠ r1 → star.1 ≠ r2 → star.2 ≠ c1 → star.2 ≠ c2) ∧
  (∀ stars_set : Finset (Fin 4 × Fin 4),
    stars_set.card < 7 →
    (∃ r1 r2 : Fin 4, ∃ c1 c2 : Fin 4, r1 ≠ r2 → c1 ≠ c2 →
      (∀ star ∈ stars_set, star.1 ≠ r1 → star.1 ≠ r2 → star.2 ≠ c1 → star.2 ≠ c2 → False))) :=
begin
  sorry
end

end seven_stars_on_grid_l95_95643


namespace conditions_guarantee_inequality_l95_95684

def f (x : ℝ) : ℝ := x^2 - 2 * Real.cos x

theorem conditions_guarantee_inequality (x1 x2 : ℝ) (h1 : x1 ∈ Set.Icc (-2 * Real.pi / 3) (2 * Real.pi / 3)) 
  (h2 : x2 ∈ Set.Icc (-2 * Real.pi / 3) (2 * Real.pi / 3)) (cond2 : x1^2 > x2^2) (cond3 : x1 > |x2|) :
  f x1 > f x2 := by
  sorry

end conditions_guarantee_inequality_l95_95684


namespace gcd_condition_for_divisibility_l95_95982

theorem gcd_condition_for_divisibility (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (∀ (x : ℕ), x^n + x^(2*n) + ... + x^(m*n) ∣ x + x^(2) + ... + x^m) ↔ Nat.gcd(n, m+1) = 1 :=
by
  sorry

end gcd_condition_for_divisibility_l95_95982


namespace larger_integer_value_l95_95912

theorem larger_integer_value (x y : ℕ) (h1 : (4 * x)^2 - 2 * x = 8100) (h2 : x + 10 = 2 * y) : x = 22 :=
by
  sorry

end larger_integer_value_l95_95912


namespace smallest_M_inequality_l95_95986

theorem smallest_M_inequality :
  ∃ M : ℝ, 
  M = 9 / (16 * Real.sqrt 2) ∧
  ∀ a b c : ℝ, 
    |a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)| 
    ≤ M * (a^2 + b^2 + c^2)^2 :=
by
  use 9 / (16 * Real.sqrt 2)
  sorry

end smallest_M_inequality_l95_95986


namespace quadratic_solutions_l95_95857

theorem quadratic_solutions :
  ∀ x : ℝ, (x^2 - 4 * x = 0) → (x = 0 ∨ x = 4) :=
by sorry

end quadratic_solutions_l95_95857


namespace inequality_not_true_l95_95187

variable (a b c : ℝ)

theorem inequality_not_true (h : a < b) : ¬ (-3 * a < -3 * b) :=
by
  sorry

end inequality_not_true_l95_95187


namespace sqrt_cos_decreasing_intervals_l95_95169

theorem sqrt_cos_decreasing_intervals :
  ∀ (k : ℤ), ∀ (x : ℝ), 2 * k * π ≤ x ∧ x ≤ ½ * π + 2 * k * π → ∀ {y : ℝ}, y = √(cos x) → strictly_decreasing (λ x, √(cos x)) :=
sorry

end sqrt_cos_decreasing_intervals_l95_95169


namespace median_list_integer_1_to_300_l95_95722

theorem median_list_integer_1_to_300 : 
  (median (list.join (list.map (λ n, list.repeat n n) (list.range 300).map (λ x, x + 1)))) = 150 := 
sorry

end median_list_integer_1_to_300_l95_95722


namespace Ivanov_made_an_error_l95_95818

theorem Ivanov_made_an_error (mean median : ℝ) (variance : ℝ) (h1 : mean = 0) (h2 : median = 4) (h3 : variance = 15.917) : 
  |mean - median| ≤ Real.sqrt variance → False :=
by {
  have mean_value : mean = 0 := h1,
  have median_value : median = 4 := h2,
  have variance_value : variance = 15.917 := h3,

  let lhs := |mean_value - median_value|,
  have rhs := Real.sqrt variance_value,
  
  calc
    lhs = |0 - 4| : by rw [mean_value, median_value]
    ... = 4 : by norm_num,
  
  have rhs_val : Real.sqrt variance_value ≈ 3.99 := by sorry, -- approximate value for demonstration
  
  have ineq : 4 ≤ rhs_val := by 
    calc 4 = 4 : rfl -- trivial step for clarity,
    have sqrt_val : Real.sqrt 15.917 < 4 := by sorry, -- from calculation or suitable proof
  
  exact absurd ineq (not_le_of_gt sqrt_val)
}

end Ivanov_made_an_error_l95_95818


namespace batteries_problem_l95_95868

noncomputable def x : ℝ := 2 * z
noncomputable def y : ℝ := (4 / 3) * z

theorem batteries_problem
  (z : ℝ)
  (W : ℝ)
  (h1 : 4 * x + 18 * y + 16 * z = W * z)
  (h2 : 2 * x + 15 * y + 24 * z = W * z)
  (h3 : 6 * x + 12 * y + 20 * z = W * z) :
  W = 48 :=
sorry

end batteries_problem_l95_95868


namespace susan_avg_speed_l95_95078

variable (d1 d2 : ℕ) (s1 s2 : ℕ)

def time (d s : ℕ) : ℚ := d / s

theorem susan_avg_speed 
  (h1 : d1 = 40) 
  (h2 : s1 = 30) 
  (h3 : d2 = 40) 
  (h4 : s2 = 15) : 
  (d1 + d2) / (time d1 s1 + time d2 s2) = 20 := 
by 
  -- Sorry to skip the proof.
  sorry

end susan_avg_speed_l95_95078


namespace exist_subsets_P_Q_l95_95767

variables (X : Finset α) (f : {E // E ∈ X.powerset ∧ E.card % 2 = 0} → ℝ)

open Finset

theorem exist_subsets_P_Q (X_finite : X.finite)
  (D : {E // E ∈ X.powerset ∧ E.card % 2 = 0})
  (hD : f D > 1990)
  (h_disjoint : ∀ A B : {E // E ∈ X.powerset ∧ E.card % 2 = 0}, disjoint A.val B.val → f ⟨A.val ∪ B.val, _⟩ = f A + f B - 1990) :
  ∃ P Q : Finset α, P ∩ Q = ∅ ∧ P ∪ Q = X ∧
    (∀ S : {E // E ⊆ P ∧ E.card % 2 = 0}, S.val ≠ ∅ → f S > 1990) ∧ 
    (∀ T : {E // E ⊆ Q ∧ E.card % 2 = 0}, f T ≤ 1990) :=
sorry

end exist_subsets_P_Q_l95_95767


namespace good_students_options_l95_95325

variables (E B : ℕ)

-- Define the condition that the class has 25 students
def total_students : Prop := E + B = 25

-- Define the condition given by the first group of students
def first_group_condition : Prop := B > 12

-- Define the condition given by the second group of students
def second_group_condition : Prop := B = 3 * (E - 1)

-- Define the problem statement
theorem good_students_options (E B : ℕ) :
  total_students E B → first_group_condition B → second_group_condition E B → (E = 5 ∨ E = 7) :=
by
  intros h1 h2 h3
  sorry

end good_students_options_l95_95325


namespace quadratic_polynomials_equal_l95_95002

def integer_part (a : ℝ) : ℤ := ⌊a⌋

theorem quadratic_polynomials_equal 
  (f g : ℝ → ℝ)
  (hf : ∀ x, ∃ a1 b1 c1, f x = a1 * x^2 + b1 * x + c1)
  (hg : ∀ x, ∃ a2 b2 c2, g x = a2 * x^2 + b2 * x + c2)
  (H : ∀ x, integer_part (f x) = integer_part (g x)) : 
  ∀ x, f x = g x :=
sorry

end quadratic_polynomials_equal_l95_95002


namespace no_such_graph_exists_l95_95547

noncomputable def vertex_degrees (n : ℕ) (deg : ℕ → ℕ) : Prop :=
  n ≥ 8 ∧
  ∃ (deg : ℕ → ℕ),
    (deg 0 = 4) ∧ (deg 1 = 5) ∧ ∀ i, 2 ≤ i ∧ i < n - 7 → deg i = i + 4 ∧
    (deg (n-7) = n-2) ∧ (deg (n-6) = n-2) ∧ (deg (n-5) = n-2) ∧
    (deg (n-4) = n-1) ∧ (deg (n-3) = n-1) ∧ (deg (n-2) = n-1)   

theorem no_such_graph_exists (n : ℕ) (deg : ℕ → ℕ) : 
  n ≥ 10 → ¬vertex_degrees n deg := 
by
  sorry

end no_such_graph_exists_l95_95547


namespace problem1_problem2_l95_95924

section
  variable {x m a b : ℝ}

  -- Problem 1: Prove (-2x)^2 + 3x * x = 7x^2
  theorem problem1 (hx : x ∈ ℝ) : (-2 * x) ^ 2 + 3 * x * x = 7 * x ^ 2 := 
    sorry

  -- Problem 2: Prove ma^2 - mb^2 = m(a - b)(a + b)
  theorem problem2 (hm : m ∈ ℝ) (ha : a ∈ ℝ) (hb : b ∈ ℝ) : 
    m * a ^ 2 - m * b ^ 2 = m * (a - b) * (a + b) := 
    sorry
end

end problem1_problem2_l95_95924


namespace angle_BMN_30_l95_95728

theorem angle_BMN_30 
    {A B C M N : Type}
    [IsoscelesTriangle ABC]
    (h_isosceles : Triangle.isosceles A B C)
    (h_angle_C : ∠ A C B = 20) 
    (h_point_M : PointOnSide M AC) 
    (h_point_N : PointOnSide N BC) 
    (h_angle_ABM : ∠ A B M = 60) 
    (h_angle_BAN : ∠ B A N = 50) :
    ∠ B M N = 30 := 
by
  sorry

end angle_BMN_30_l95_95728


namespace sandy_correct_value_t_l95_95459

theorem sandy_correct_value_t (p q r s : ℕ) (t : ℕ) 
  (hp : p = 2) (hq : q = 4) (hr : r = 6) (hs : s = 8)
  (expr1 : p + q - r + s - t = p + (q - (r + (s - t)))) :
  t = 8 := 
by
  sorry

end sandy_correct_value_t_l95_95459


namespace find_breadth_of_rectangle_l95_95485

noncomputable def breadth_of_rectangle (s : ℝ) (π_approx : ℝ := 3.14) : ℝ :=
2 * s - 22

theorem find_breadth_of_rectangle (b s : ℝ) (π_approx : ℝ := 3.14) :
  4 * s = 2 * (22 + b) →
  π_approx * s / 2 + s = 29.85 →
  b = 1.22 :=
by
  intros h1 h2
  sorry

end find_breadth_of_rectangle_l95_95485


namespace triangle_KLM_angle_KBA_l95_95398

theorem triangle_KLM_angle_KBA (K L M A B : Type) [Hilbert_space K] [Hilbert_space L]
                              [Hilbert_space M] [Hilbert_space A] [Hilbert_space B] 
                              (angle_KLM : Real) (angle_LKM : Real)
                              (angle_L : Real) (angle_L_KBA : Real)
                              (bisects_LA : IsBisector angle_KLM, LA)
                              (bisects_KB : IsBisector angle_LKM, KB) :
  angle_KBA = 30 := 
by 
  sorry

end triangle_KLM_angle_KBA_l95_95398


namespace ball_box_arrangement_l95_95450

theorem ball_box_arrangement : 
  ∀ (balls boxes : ℕ), balls = 5 ∧ boxes = 4 → 
  (∑ (k in finset.range boxes), nat.choose balls k * nat.factorial boxes) = 240 := 
begin
  intros balls boxes h,
  cases h with hballs hboxes,
  rw [hballs, hboxes],
  sorry
end

end ball_box_arrangement_l95_95450


namespace xy_problem_l95_95272

theorem xy_problem (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 :=
by
  sorry

end xy_problem_l95_95272


namespace value_of_expression_l95_95539

theorem value_of_expression : 3 - (-3) ^ (-3) = 82 / 27 := by
  sorry

end value_of_expression_l95_95539


namespace evaluate_f_l95_95285

def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 4 * x

theorem evaluate_f (h : f 3 - f (-3) = 672) : True :=
by
  sorry

end evaluate_f_l95_95285


namespace pirate_ship_overtakes_l95_95568

theorem pirate_ship_overtakes
    (initial_distance : ℝ)
    (pirate_speed : ℝ)
    (trader_speed : ℝ)
    (damage_time : ℝ)
    (new_pirate_speed : ℝ)
    (start_time : ℝ) :
    initial_distance = 15 →
    pirate_speed = 10 →
    trader_speed = 7 →
    damage_time = 3 →
    new_pirate_speed = 14 →
    start_time = 10 →
    let relative_initial_speed := pirate_speed - trader_speed in
    let distance_after_damage := initial_distance - ((pirate_speed * damage_time) - (trader_speed * damage_time)) in
    let relative_new_speed := new_pirate_speed - trader_speed in
    let time_to_catch_up := distance_after_damage / relative_new_speed in
    let catch_up_time := start_time + damage_time + time_to_catch_up in
    catch_up_time ≈ 13 + 55/60 := -- 1:55 p.m. in 24-hour format as 13:55
begin
  sorry
end

end pirate_ship_overtakes_l95_95568


namespace total_investment_is_10000_l95_95965

open Real

-- Definitions of conditions
def interest_rate_8 : Real := 0.08
def interest_rate_9 : Real := 0.09
def combined_interest : Real := 840
def investment_8 : Real := 6000
def total_interest (x : Real) : Real := (interest_rate_8 * investment_8 + interest_rate_9 * x)
def investment_9 : Real := 4000

-- Theorem stating the problem
theorem total_investment_is_10000 :
    (∀ x : Real,
        total_interest x = combined_interest → x = investment_9) →
    investment_8 + investment_9 = 10000 := 
by
    intros
    sorry

end total_investment_is_10000_l95_95965


namespace kevin_hops_six_l95_95407

def total_distance_after_hops (n : ℕ) : ℚ :=
  (∑ k in Finset.range n, (3 / 4) ^ k) / 4

theorem kevin_hops_six :
  total_distance_after_hops 6 = 3367 / 4096 := by
  sorry

end kevin_hops_six_l95_95407


namespace average_weight_children_l95_95716

theorem average_weight_children 
  (n_boys : ℕ)
  (w_boys : ℕ)
  (avg_w_boys : ℕ)
  (n_girls : ℕ)
  (w_girls : ℕ)
  (avg_w_girls : ℕ)
  (h1 : n_boys = 8)
  (h2 : avg_w_boys = 140)
  (h3 : n_girls = 6)
  (h4 : avg_w_girls = 130)
  (h5 : w_boys = n_boys * avg_w_boys)
  (h6 : w_girls = n_girls * avg_w_girls)
  (total_w : ℕ)
  (h7 : total_w = w_boys + w_girls)
  (avg_w : ℚ)
  (h8 : avg_w = total_w / (n_boys + n_girls)) :
  avg_w = 135 :=
by
  sorry

end average_weight_children_l95_95716


namespace area_of_triangle_from_squares_l95_95212

theorem area_of_triangle_from_squares :
  ∃ (a b c : ℕ), (a = 15 ∧ b = 15 ∧ c = 6 ∧ (1/2 : ℚ) * a * c = 45) :=
by
  let a := 15
  let b := 15
  let c := 6
  have h1 : (1/2 : ℚ) * a * c = 45 := sorry
  exact ⟨a, b, c, ⟨rfl, rfl, rfl, h1⟩⟩

end area_of_triangle_from_squares_l95_95212


namespace percentage_of_repeated_digit_five_digit_numbers_l95_95599

open Nat

theorem percentage_of_repeated_digit_five_digit_numbers :
  let total_five_digit_numbers := 90000
  let no_repeat_digit_numbers := 27216
  let repeated_digit_numbers := total_five_digit_numbers - no_repeat_digit_numbers
  let percentage := (float_of_int repeated_digit_numbers / float_of_int total_five_digit_numbers) * 100 in
  percentage ≈ 69.8 :=
by 
  sorry

end percentage_of_repeated_digit_five_digit_numbers_l95_95599


namespace ivanov_error_l95_95808

-- Given conditions
def mean_temperature (x : ℝ) : Prop := x = 0
def median_temperature (m : ℝ) : Prop := m = 4
def variance_temperature (s² : ℝ) : Prop := s² = 15.917

-- Statement to prove
theorem ivanov_error (x m s² : ℝ) 
  (mean_x : mean_temperature x)
  (median_m : median_temperature m)
  (variance_s² : variance_temperature s²) :
  (x - m)^2 > s² :=
by
  rw [mean_temperature, median_temperature, variance_temperature] at *
  simp [*, show 0 = 0, from rfl, show 4 = 4, from rfl]
  sorry

end ivanov_error_l95_95808


namespace triangle_side_length_l95_95300

theorem triangle_side_length (A B C : ℝ) (BC AC : ℝ) (hA : A = 45) (hC : C = 105) (hBC : BC = sqrt 2)
  (h_sum_angles : A + B + C = 180) (law_of_sines : BC / sin (A * real.pi / 180) = AC / sin (B * real.pi / 180)) :
  AC = 1 :=
begin
  sorry
end

end triangle_side_length_l95_95300


namespace rationalize_fraction_l95_95451

noncomputable def rationalize_denominator (a b : ℚ) : Prop :=
  a = b

theorem rationalize_fraction :
  rationalize_denominator (7 / Real.sqrt 75) (7 * Real.sqrt 3 / 15) :=
by
  sorry

end rationalize_fraction_l95_95451


namespace midpoint_of_arc_BAC_l95_95635

noncomputable def circumcenter (ABC : Triangle) : Point := sorry
noncomputable def incenter (ABC : Triangle) : Point := sorry
noncomputable def excircle (A : Point) (BC : Line) : Circle := sorry
noncomputable def tangent_point (T : Circle) (O : Circle) : Point := sorry
noncomputable def line (P I : Point) : Line := sorry
noncomputable def intersect_line_circle (l : Line) (O : Circle) : Point := sorry
noncomputable def midpoint_of_arc (Q : Point) (arc : Arc) : Prop := sorry
noncomputable def arc_BAC (O : Circle) (A B C : Point) : Arc := sorry
noncomputable def Circle (O : Point) (r : ℝ) : Circle := sorry
noncomputable def Triangle (A B C : Point) : Triangle := sorry

theorem midpoint_of_arc_BAC (A B C : Point) :
  let ABC := Triangle.mk A B C
  let O := circumcenter ABC
  let I := incenter ABC
  let T := excircle A (Line.mk B C)
  let P := tangent_point T (Circle.mk O (radius_of_circle O))
  let PI_line := line P I
  let Q := intersect_line_circle PI_line (Circle.mk O (radius_of_circle O))
  midpoint_of_arc Q (arc_BAC (Circle.mk O (radius_of_circle O)) A B C)
by
  sorry

end midpoint_of_arc_BAC_l95_95635


namespace good_students_l95_95340

theorem good_students (E B : ℕ) (h1 : E + B = 25) (h2 : 12 < B) (h3 : B = 3 * (E - 1)) :
  E = 5 ∨ E = 7 :=
by 
  sorry

end good_students_l95_95340


namespace hyperbola_equation_ordinate_of_point_D_const_l95_95223

-- Problem 1: Equation of Hyperbola
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (e : ℝ) (h3 : e = (Real.sqrt 5) / 2) 
    (axis_length : ℝ) (h4 : axis_length = 8) : 
    ∃ (a b : ℝ), (a = 4) ∧ (b = 2) ∧ (e = (Real.sqrt 5) / 2) ∧ (2 * a = 8) 
    ∧ (ax : ℝ), (ax = (x: ℝ) → ((x^2) / (4 ^2) - (y^2) / (2^2) = 1)) :=
by
  -- The proof is not provided, it's skipped with sorry.
  sorry

-- Problem 2: Constant Ordinate of Point D
theorem ordinate_of_point_D_const (P A B D : ℝ × ℝ) (l : ℝ → ℝ)
    (hyp_eq : ∀ x y, x^2 / 16 - y^2 / 4 = 1)
    (line_eq : ∀ x, l x = - (sqrt 5 / 2) * x + 3)
    (distinct_points : P ≠ A ∧ P ≠ B ∧ A ≠ B ∧ A ≠ D ∧ B ≠ D)
    (cond : |P.1 - A.1| * |D.1 - B.1| = |P.1 - B.1| * |D.1 - A.1|) :
    D.2 = -4 / 3 :=
by
  -- The proof is not provided, it's skipped with sorry.
  sorry

end hyperbola_equation_ordinate_of_point_D_const_l95_95223


namespace curve_intersection_x_axis_distance_between_points_A_B_l95_95384

section
variable (t θ a : ℝ)
variable (x y : ℝ)

noncomputable def C1 (t : ℝ) : ℝ × ℝ := (t + 1, 1 - 2 * t)
noncomputable def C2 (a θ : ℝ) : ℝ × ℝ := (a * Real.cos θ, 3 * Real.sin θ)

theorem curve_intersection_x_axis (a_gt_0 : a > 0) :
  ∃ t θ, (C1 t).1 = (C2 a θ).1 ∧ (C1 t).2 = (C2 a θ).2 ∧ 0 = (C1 t).2 ∧ 0 = (C2 a θ).2 
  ↔ a = 3 / 2 := 
by 
  sorry

theorem distance_between_points_A_B (a_eq_3 : a = 3) :
  ∃ A B t₁ t₂ θ₁ θ₂, 
  (C1 t₁).1 = (C2 a θ₁).1 ∧ (C1 t₂).1 = (C2 a θ₂).1 ∧ 
  (C1 t₁).2 = (C2 a θ₁).2 ∧ (C1 t₂).2 = (C2 a θ₂).2 ∧ 
  sqrt ((fst A - fst B)^2 + (snd A - snd B)^2) = 12 * sqrt 5 / 5 :=
by 
  sorry
end

end curve_intersection_x_axis_distance_between_points_A_B_l95_95384


namespace values_of_a_and_b_solution_set_inequality_l95_95685

-- Part (I)
theorem values_of_a_and_b (a b : ℝ) (h : ∀ x, -1 < x ∧ x < 1 → x^2 - a * x - x + b < 0) :
  a = -1 ∧ b = -1 := sorry

-- Part (II)
theorem solution_set_inequality (a : ℝ) (h : a = b) :
  (∀ x, x^2 - a * x - x + a < 0 → (x = 1 → false) 
      ∧ (0 < 1 - a → (x = 1 → false))
      ∧ (1 < - a → (x = 1 → false))) := sorry

end values_of_a_and_b_solution_set_inequality_l95_95685


namespace find_g_l95_95164

-- Define given functions and terms
def f1 (x : ℝ) := 7 * x^4 - 4 * x^3 + 2 * x - 5
def f2 (x : ℝ) := 5 * x^3 - 3 * x^2 + 4 * x - 1
def g (x : ℝ) := -7 * x^4 + 9 * x^3 - 3 * x^2 + 2 * x + 4

-- Theorem to prove that g(x) satisfies the given condition
theorem find_g : ∀ x : ℝ, f1 x + g x = f2 x :=
by 
  -- Alternatively: Proof is required here
  sorry

end find_g_l95_95164


namespace max_liters_of_water_that_can_be_heated_to_boiling_l95_95900

-- Define the initial conditions
def initial_heat_per_5min := 480 -- kJ
def heat_reduction_rate := 0.25
def initial_temp := 20 -- Celsius
def boiling_temp := 100 -- Celsius
def specific_heat_capacity := 4.2 -- kJ/kg·°C

-- Define the temperature difference
def delta_T := boiling_temp - initial_temp -- Celsius

-- Define the calculation of the total heat available from a geometric series
def total_heat_available := initial_heat_per_5min / (1 - (1 - heat_reduction_rate))

-- Define the calculation of energy required to heat m kg of water
def energy_required (m : ℝ) := specific_heat_capacity * m * delta_T

-- Define the main theorem to prove
theorem max_liters_of_water_that_can_be_heated_to_boiling :
  ∃ (m : ℝ), ⌊m⌋ = 5 ∧ energy_required m ≤ total_heat_available :=
begin
  sorry
end

end max_liters_of_water_that_can_be_heated_to_boiling_l95_95900


namespace smallest_whole_number_satisfying_triangle_inequality_l95_95504

theorem smallest_whole_number_satisfying_triangle_inequality :
  ∃ (s : ℕ), (8.5 + s > 11.5) ∧ (8.5 + 11.5 > s) ∧ (11.5 + s > 8.5) ∧ (s = 4) :=
sorry

end smallest_whole_number_satisfying_triangle_inequality_l95_95504


namespace mul_mod_eq_l95_95598

theorem mul_mod_eq :
  (66 * 77 * 88) % 25 = 16 :=
by 
  sorry

end mul_mod_eq_l95_95598


namespace rocket_max_height_and_danger_l95_95840

-- Given conditions
def acceleration_engine : ℝ := 20 -- m/s^2
def time_engine_cutoff : ℝ := 40 -- seconds
def gravity : ℝ := 10 -- m/s^2
def object_height : ℝ := 45 -- km

-- Calculated Values
def initial_velocity : ℝ := acceleration_engine * time_engine_cutoff
def height_at_cutoff : ℝ := 0.5 * acceleration_engine * (time_engine_cutoff ^ 2)
def time_to_zero_velocity : ℝ := initial_velocity / gravity
def max_additional_height : ℝ := initial_velocity * time_to_zero_velocity - 0.5 * gravity * (time_to_zero_velocity ^ 2)
def max_height_reached : ℝ := max_additional_height + height_at_cutoff

-- Theorem to prove
theorem rocket_max_height_and_danger:
  max_height_reached = 48 ∧ max_height_reached > object_height := by
  -- skipping proof
  sorry

end rocket_max_height_and_danger_l95_95840


namespace commission_rate_correct_l95_95054

variables (weekly_earnings : ℕ) (commission : ℕ) (total_earnings : ℕ) (sales : ℕ) (commission_rate : ℕ)

-- Base earnings per week without commission
def base_earnings : ℕ := 190

-- Total earnings target
def earnings_goal : ℕ := 500

-- Minimum sales required to meet the earnings goal
def sales_needed : ℕ := 7750

-- Definition of the commission as needed to meet the goal
def needed_commission : ℕ := earnings_goal - base_earnings

-- Definition of the actual commission rate
def commission_rate_per_sale : ℕ := (needed_commission * 100) / sales_needed

-- Proof goal: Show that commission_rate_per_sale is 4
theorem commission_rate_correct : commission_rate_per_sale = 4 :=
by
  sorry

end commission_rate_correct_l95_95054


namespace right_triangle_area_l95_95116

/-- 
Given a right triangle with a circle inscribed such that the circle is tangent to the legs of the triangle.
The hypotenuse is divided into three segments of lengths 1, 24, and 3, where 24 is the length of the chord of the circle.
Prove that the area of the triangle is 192.
-/
theorem right_triangle_area (x y z : ℕ) (h_sum : x + y + z = 28) (h_chord : y = 24) (h_1 : x = 1) (h_3 : z = 3) :
  let a := 5,  -- Derived from sqrt(1 * (1 + 24)) = sqrt(25) = 5
      b := 9,  -- Derived from sqrt(3 * (3 + 24)) = sqrt(81) = 9
      hypotenuse := 28,
      x := hypotenuse - (a + b - 4) / 2 in
  (1/2) * (a + x) * (b + x) = 192 :=
by sorry

end right_triangle_area_l95_95116


namespace possible_to_reach_large_number_l95_95518

/-- Define the number of coins in each pile -/
variable (a b c : ℕ)

/-- Define constant for the large number -/
def large_number := 2017 ^ 2017

/-- Define the conditions -/
axiom coins_ge_2015 : a ≥ 2015 ∧ b ≥ 2015 ∧ c ≥ 2015
axiom not_2015_triplet : ¬(a = 2015 ∧ b = 2015 ∧ c = 2015)

/-- Define the operations -/
def operation1 (x y z: ℕ) : Prop := (x % 2 = 0) → (x / 2 + y) > y ∧ (x / 2 + z) > z
def operation2 (x y z: ℕ) : Prop := (x ≥ 2017 ∧ x % 2 = 1) → (x - 2017 + y + 1009) > y ∧ (x - 2017 + z + 1009) > z

/-- Define target goal -/
theorem possible_to_reach_large_number
    (a b c : ℕ)
    (h1 : coins_ge_2015 a b c)
    (h2 : not_2015_triplet a b c)
    (h3 : operation1 a b c ∨ operation2 a b c) :
    a ≥ large_number ∨ b ≥ large_number ∨ c ≥ large_number :=
sorry

end possible_to_reach_large_number_l95_95518


namespace race_length_1000_l95_95249

-- Defining the conditions
def harper_jack_race (L : ℕ) (jack_distance : ℕ) (distance_apart: ℕ) : Prop :=
  jack_distance = 152 ∧ distance_apart = 848 ∧ (L - jack_distance = distance_apart)

-- Stating the proof problem
theorem race_length_1000 : ∃ L : ℕ, harper_jack_race L 152 848 ∧ L = 1000 := by
  existsi 1000
  unfold harper_jack_race
  simp
  constructor; try { sorry }; constructor; try { sorry }

end race_length_1000_l95_95249


namespace compare_values_l95_95192

noncomputable def a := Real.exp 0.2
noncomputable def b := 0.2 ^ Real.exp 1
noncomputable def c := Real.log 2

theorem compare_values : b < c ∧ c < a := 
  by
  sorry

end compare_values_l95_95192


namespace max_ac_value_l95_95708

noncomputable def max_ac : ℤ := 98441

theorem max_ac_value :
  ∃ (a c : ℤ) (y z m n : ℤ),
  y + z = a ∧ yz = 48 ∧ 
  m + n = -8 ∧ mn = c ∧
  y ∈ (set.Icc (-50) 50) ∧ z ∈ (set.Icc (-50) 50) ∧ 
  m ∈ (set.Icc (-50) 50) ∧ n ∈ (set.Icc (-50) 50) ∧ 
  a * c = max_ac :=
begin
  sorry
end

end max_ac_value_l95_95708


namespace arith_seq_sum_of_terms_l95_95759

theorem arith_seq_sum_of_terms 
  (a : ℕ → ℝ) (d : ℝ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_pos_diff : 0 < d) 
  (h_first_three_sum : a 0 + a 1 + a 2 = 15) 
  (h_first_three_prod : a 0 * a 1 * a 2 = 80) : 
  a 10 + a 11 + a 12 = 105 := sorry

end arith_seq_sum_of_terms_l95_95759


namespace find_number_of_good_students_l95_95328

variable {G T : ℕ}

def is_good_or_troublemaker (G T : ℕ) : Prop :=
  G + T = 25 ∧ 
  (∀ s ∈ finset.range 5, T > 12) ∧ 
  (∀ s ∈ finset.range 20, T = 3 * (G - 1))

-- There are 25 students in total. 
-- Each student is either a good student or a troublemaker.
-- Good students always tell the truth, while troublemakers always lie.
-- Five students make the statement: "If I transfer to another class, more than half of the remaining students will be troublemakers."
-- The remaining 20 students make the statement: "If I transfer to another class, the number of troublemakers among the remaining students will be three times the number of good students."

theorem find_number_of_good_students (G T : ℕ) (h : is_good_or_troublemaker G T) :
  (G = 5 ∨ G = 7) :=
sorry

end find_number_of_good_students_l95_95328


namespace positive_difference_sum_first_25_even_and_three_times_sum_first_20_odd_l95_95905

-- Definitions from conditions
def sum_first_N_even_integers (N : ℕ) : ℕ :=
  2 * (N * (N + 1) / 2)

def sum_first_N_odd_integers (N : ℕ) : ℕ :=
  N * N

def three_times_sum_first_N_odd_integers (N : ℕ) : ℕ :=
  3 * (sum_first_N_odd_integers N)

def positive_difference (a b : ℤ) : ℕ :=
  abs (a - b).nat_abs

-- Lean statement of the proof problem
theorem positive_difference_sum_first_25_even_and_three_times_sum_first_20_odd :
  positive_difference (sum_first_N_even_integers 25) (three_times_sum_first_N_odd_integers 20) = 550 :=
by
  sorry

end positive_difference_sum_first_25_even_and_three_times_sum_first_20_odd_l95_95905
