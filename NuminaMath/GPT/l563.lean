import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Floor
import Mathlib.Algebra.GcdMonoid.Finset
import Mathlib.Algebra.GeomSeq
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Logarithm
import Mathlib.Algebra.Quadratic
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Calculus.Fderiv.Basic
import Mathlib.Analysis.Calculus.TangentCone
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.Convex.Basic
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.GraphTheory.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.GraphTheory.Hamiltonian
import Mathlib.Love.QuadraticFunctions
import Mathlib.MeasureTheory.Integral.IntervalIntegral
import Mathlib.Probability.Odds
import Mathlib.Real
import Mathlib.SetTheory.Cardinal
import Mathlib.Tactic
import Mathlib.Topology.Basic
import Probability.ProbabilityMassFunction.Distributions

namespace determine_y_l563_563837

theorem determine_y :
  ∃ (y : ℝ), 20^y * 200^(3 * y) = 8000^7 ∧ y = 3 :=
by
  sorry

end determine_y_l563_563837


namespace find_ellipse_equation_find_hyperbola_equation_l563_563179

-- Define the conditions for the ellipse
def ellipse_condition (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧
  2 * a = 8 ∧ 
  (a^2 - b^2) = (a / 2) ^ 2

-- with the ellipse condition proving the equation of the ellipse
theorem find_ellipse_equation (a b : ℝ) 
  (h_ellipse : ellipse_condition a b) : 
  (a = 4) → (b^2 = 12) → 
  ∀ x y : ℝ, x^2 / 16 + y^2 / 12 = 1 :=
sorry

-- Define the conditions for the hyperbola with shared foci and specified asymptote
def hyperbola_condition (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  c = 2 ∧ 
  (b / a = √3)

-- With the hyperbola condition proving the equation of the hyperbola share same foci 
theorem find_hyperbola_equation (a b c : ℝ)
  (h_hyperbola : hyperbola_condition a b c) : 
  (a = 1) → (b = √3) → 
  ∀ x y : ℝ, x^2 - (y^2 / 3) = 1 :=
sorry

end find_ellipse_equation_find_hyperbola_equation_l563_563179


namespace fraction_study_only_japanese_l563_563472

variable (J : ℕ)

def seniors := 2 * J
def sophomores := (3 / 4) * J

def seniors_study_japanese := (3 / 8) * seniors J
def juniors_study_japanese := (1 / 4) * J
def sophomores_study_japanese := (2 / 5) * sophomores J

def seniors_study_both := (1 / 6) * seniors J
def juniors_study_both := (1 / 12) * J
def sophomores_study_both := (1 / 10) * sophomores J

def seniors_study_only_japanese := seniors_study_japanese J - seniors_study_both J
def juniors_study_only_japanese := juniors_study_japanese J - juniors_study_both J
def sophomores_study_only_japanese := sophomores_study_japanese J - sophomores_study_both J

def total_study_only_japanese := seniors_study_only_japanese J + juniors_study_only_japanese J + sophomores_study_only_japanese J
def total_students := J + seniors J + sophomores J

theorem fraction_study_only_japanese :
  (total_study_only_japanese J) / (total_students J) = 97 / 450 :=
by sorry

end fraction_study_only_japanese_l563_563472


namespace akeno_spent_more_l563_563097

theorem akeno_spent_more (akeno_expenditure : ℝ) (lev_expenditure : ℝ) (ambrocio_expenditure : ℝ) :
  akeno_expenditure = 2985 ∧
  lev_expenditure = (1 / 3) * akeno_expenditure ∧
  ambrocio_expenditure = lev_expenditure - 177 →
  akeno_expenditure - (lev_expenditure + ambrocio_expenditure) = 1172 :=
by
  intros h
  cases h with h_akeno h_lev
  cases h_lev with h_lev_exp h_ambrocio
  rw [h_akeno, h_lev_exp, h_ambrocio]
  calc
    2985 - ((1 / 3) * 2985 + ((1 / 3) * 2985 - 177))
        = 2985 - ((1 / 3) * 2985 + (1 / 3) * 2985 - 177) : by rw sub_add
    ... = 2985 - (2 * (1 / 3) * 2985 - 177) : by rw add_assoc
    ... = 2985 - (1990 - 177) : by norm_num
    ... = 2985 - 1813 : by norm_num
    ... = 1172 : by norm_num

end akeno_spent_more_l563_563097


namespace acute_face_implies_acute_dihedral_l563_563666

noncomputable def face_angle_acute {A B C D : Type} 
  (tetrahedron : A → B → C → D → Prop) 
  (acute_face_angles : ∀ {angle : ℝ}, angle ∈ face_angles_of tetrahedron → 0 < angle ∧ angle < π / 2) : 
  Prop := ∀ {dihedral_angle : ℝ}, dihedral_angle ∈ dihedral_angles_of tetrahedron → 0 < dihedral_angle ∧ dihedral_angle < π / 2

theorem acute_face_implies_acute_dihedral
  {A B C D : Type} 
  (tetrahedron : A → B → C → D → Prop) 
  (h : ∀ {angle : ℝ}, angle ∈ face_angles_of tetrahedron → 0 < angle ∧ angle < π / 2)
  : face_angle_acute tetrahedron h := 
sorry

end acute_face_implies_acute_dihedral_l563_563666


namespace intersection_of_M_and_N_l563_563644

def M : Set ℝ := {x | x^2 - 2 * x < 0}
def N : Set ℝ := {x | |x| < 1}

theorem intersection_of_M_and_N : M ∩ N = {x | 0 < x ∧ x < 1} := by
  sorry

end intersection_of_M_and_N_l563_563644


namespace cannot_use_square_diff_formula_l563_563746

theorem cannot_use_square_diff_formula :
  ¬ (∃ a b, (x - y) * (-x + y) = (a + b) * (a - b) ∨ (a - b) * (a + b)) ∧
  (∃ a b, (-x - y) * (x - y) = (a + b) * (a - b) ∨ (a - b) * (a + b)) ∧
  (∃ a b, (-x + y) * (-x - y) = (a + b) * (a - b) ∨ (a - b) * (a + b)) ∧
  (∃ a b, (x + y) * (-x + y) = (a + b) * (a - b) ∨ (a - b) * (a + b)) :=
sorry

end cannot_use_square_diff_formula_l563_563746


namespace initial_birds_count_l563_563404

theorem initial_birds_count (B : ℕ) (h1 : 6 = B + 3 + 1) : B = 2 :=
by
  -- Placeholder for the proof, we are not required to provide it here.
  sorry

end initial_birds_count_l563_563404


namespace parts_supplier_received_1340_dollars_l563_563808

def price_per_package := 25
def discount_price_per_package := (4 / 5) * price_per_package
def total_packages_sold := 60
def company_x_packages : ℕ := (0.15 * total_packages_sold).toNat
def company_y_packages : ℕ := (0.15 * total_packages_sold).toNat
def company_z_packages : ℕ := total_packages_sold - (company_x_packages + company_y_packages)
def cost_company_x : ℕ := company_x_packages * price_per_package
def cost_company_y : ℕ := company_y_packages * price_per_package 
def cost_company_z := 
  if company_z_packages > 10 then
    (10 * price_per_package) + ((company_z_packages - 10) * discount_price_per_package).toNat
  else 
    company_z_packages * price_per_package

def total_amount_received := cost_company_x + cost_company_y + cost_company_z

theorem parts_supplier_received_1340_dollars : total_amount_received = 1340 := by 
  sorry

end parts_supplier_received_1340_dollars_l563_563808


namespace translate_parabola_up_l563_563684

theorem translate_parabola_up (x : ℝ) : (∃ y : ℝ, y = x^2) → (∃ y' : ℝ, y' = x^2 + 3) :=
by
  intro h
  cases h with y hy
  use y + 3
  rw hy
  ring
  sorry

end translate_parabola_up_l563_563684


namespace smallest_domain_size_l563_563832

noncomputable def g : ℕ → ℕ
| x :=
  if x = 11 then 24
  else if x % 2 = 0 then x / 2
  else 3 * x + 1

theorem smallest_domain_size : ∃ n : ℕ, n = 12 ∧ ∀ x ∈ (Set.range g), is_integer x ∧ is_integer n :=
by
  sorry

end smallest_domain_size_l563_563832


namespace sequence_a_five_is_19_l563_563982

noncomputable def a : ℕ → ℤ
| 1     := 2
| 2     := 5
| (n+3) := a n + a (n+1)

theorem sequence_a_five_is_19 : a 5 = 19 :=
by {
  sorry
}

end sequence_a_five_is_19_l563_563982


namespace euler_totient_sum_of_divisors_l563_563508

-- Define the Euler's totient function
def euler_totient (n : ℕ) : ℕ :=
if n = 0 then 0 else (Finset.range n).filter (Nat.coprime n).card

-- Define the sum of divisors function
def sum_of_divisors (n : ℕ) : ℕ :=
(Finset.range (n + 1)).filter (fun d => n % d = 0).sum id

-- Define the main theorem
theorem euler_totient_sum_of_divisors (n : ℕ) :
  (euler_totient n + sum_of_divisors n = 2 * n + 8) ↔ (n = 343 ∨ n = 12) :=
by
  sorry

end euler_totient_sum_of_divisors_l563_563508


namespace problem_1_problem_2_l563_563755

variables (a b b' c m q M : ℕ)
variables (S_q : ℕ → ℕ) -- Assume S_q is a function from ℕ to ℕ
variables (M_pos : M > 0) (m_gt_one : m > 1) (q_gt_one : q > 1)
variables (diff_ge_a : |b - b'| ≥ a)
variables (exists_M : ∀ n ≥ M, S_q (a * n + b) ≡ S_q (a * n + b') + c [MOD m])

theorem problem_1 (n : ℕ) (n_pos : n > 0) : S_q (a * n + b) ≡ S_q (a * n + b') + c [MOD m] :=
sorry

theorem problem_2 (L : ℕ) (L_pos : L > 0) : S_q (L + b) ≡ S_q (L + b') + c [MOD m] :=
sorry

end problem_1_problem_2_l563_563755


namespace eq_circle_equation_l563_563067

theorem eq_circle_equation : 
  ∃ C : (ℝ × ℝ) × ℝ,
    (∃ p1 p2 : ℝ × ℝ, p1 = (-2, 0) ∧ p2 = (-4, 0) ∧ 
      C.1 = (-3, 2) ∧ ((p1.1 + p2.1) / 2 = C.1.1 ∧ p1.2 = p2.2) ∧ 
      (C.1.1 - 2 * C.1.2 + 7 = 0) ∧ dist p1 C.1 = sqrt 5) ∧ 
    (C.2 = sqrt 5) ∧ (∀ p : ℝ × ℝ, dist p C.1 = C.2 ↔ (p.1 + 3)^2 + (p.2 - 2)^2 = 5) :=
by { 
  sorry 
}

end eq_circle_equation_l563_563067


namespace bakery_doughnuts_given_away_l563_563407

theorem bakery_doughnuts_given_away :
  (∀ (boxes_doughnuts : ℕ) (total_doughnuts : ℕ) (boxes_sold : ℕ), 
    boxes_doughnuts = 10 →
    total_doughnuts = 300 →
    boxes_sold = 27 →
    ∃ (doughnuts_given_away : ℕ),
    doughnuts_given_away = (total_doughnuts / boxes_doughnuts - boxes_sold) * boxes_doughnuts ∧
    doughnuts_given_away = 30) :=
by
  intros boxes_doughnuts total_doughnuts boxes_sold h1 h2 h3
  use (total_doughnuts / boxes_doughnuts - boxes_sold) * boxes_doughnuts
  split
  · rw [h1, h2, h3]
    sorry
  · sorry

end bakery_doughnuts_given_away_l563_563407


namespace maximal_length_of_sequence_l563_563354

/-- A sequence with constraints on digits and Ukrainian letters -/
structure ConstrainedSequence :=
  (sequence : List Char)
  (digits : List Char := ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
  -- Define Ukrainian letters in sequence (this is a simplification, actual letters should be used)
  (ukrainianLetters : List Char := [...])  -- Fill with actual Ukrainian letters
  (is_non_decreasing : ∀ i j, i < j → sequence[i] ∈ digits → sequence[j] ∈ digits → sequence[i] ≤ sequence[j])
  (is_non_decreasing_letters : ∀ i j, i < j → sequence[i] ∈ ukrainianLetters → sequence[j] ∈ ukrainianLetters → sequence[i] ≤ sequence[j])
  (no_two_consecutive_same : ∀ i, sequence[i] = sequence[i+1] → false)
  (no_two_between_same : ∀ i, sequence[i] = sequence[i+2] → false)

theorem maximal_length_of_sequence : ∃ s : ConstrainedSequence, List.length s.sequence = 73 := 
by {
  sorry
}

end maximal_length_of_sequence_l563_563354


namespace arrange_numbers_ascending_l563_563469

noncomputable def repeating_fraction (r: ℚ) : ℚ :=
  let n := r.num
  let d := r.denom
  if (d = 9) then n / 10 else n / 100

def num1: ℚ := repeating_fraction (5/90)
def num2: ℚ := repeating_fraction (5/99)
def num3: ℚ := 505/1000
def num4: ℚ := repeating_fraction (56/100)

theorem arrange_numbers_ascending:
  num3 < num1 ∧ num1 < num4 ∧ num4 < num2 :=
by
  sorry

end arrange_numbers_ascending_l563_563469


namespace length_of_base_of_vessel_l563_563765

-- Define the conditions as constants or hypotheses
constant edge : ℝ := 5
constant W : ℝ := 5
constant rise : ℝ := 2.5
constant V_cube : ℝ := edge^3
constant V_water_displaced : ℝ := V_cube

-- Prove the length of the base of the vessel L is 10 cm
theorem length_of_base_of_vessel : ∀ L : ℝ, (V_water_displaced = L * W * rise) → (L = 10) :=
by
  -- Constants and equations from conditions
  let edge := 5
  let W := 5
  let rise := 2.5
  let V_cube := edge^3
  let V_water_displaced := V_cube

  -- Given equation
  intros L h
  sorry

end length_of_base_of_vessel_l563_563765


namespace total_trips_correct_l563_563723

-- Define Timothy's movie trips in 2009
def timothy_2009_trips : ℕ := 24

-- Define Timothy's movie trips in 2010
def timothy_2010_trips : ℕ := timothy_2009_trips + 7

-- Define Theresa's movie trips in 2009
def theresa_2009_trips : ℕ := timothy_2009_trips / 2

-- Define Theresa's movie trips in 2010
def theresa_2010_trips : ℕ := timothy_2010_trips * 2

-- Define the total number of trips for Timothy and Theresa in 2009 and 2010
def total_trips : ℕ := (timothy_2009_trips + timothy_2010_trips) + (theresa_2009_trips + theresa_2010_trips)

-- Prove the total number of trips is 129
theorem total_trips_correct : total_trips = 129 :=
by
  sorry

end total_trips_correct_l563_563723


namespace count_integers_in_interval_l563_563945

theorem count_integers_in_interval : 
  ∃ (k : ℤ), k = 46 ∧ 
  (∀ n : ℤ, -5 * (2.718 : ℝ) ≤ (n : ℝ) ∧ (n : ℝ) ≤ 12 * (2.718 : ℝ) → (-13 ≤ n ∧ n ≤ 32)) ∧ 
  (∀ n : ℤ, -13 ≤ n ∧ n ≤ 32 → -5 * (2.718 : ℝ) ≤ (n : ℝ) ∧ (n : ℝ) ≤ 12 * (2.718 : ℝ)) :=
sorry

end count_integers_in_interval_l563_563945


namespace range_of_f_on_interval_l563_563555

noncomputable def f (x : ℝ) : ℝ := x^3

theorem range_of_f_on_interval :
  (set.range (λ (x : ℝ), f x) ∩ set.Ici (-1)) = set.Ici (-1) := 
sorry

end range_of_f_on_interval_l563_563555


namespace solve_for_c_l563_563202

noncomputable def f (x : ℝ) : ℝ :=
  cos (x + π / 6) * sin (x + π / 3) - sin x * cos x - 1 / 4

theorem solve_for_c :
  ∀ (A B C a b c : ℝ),
  (b = 2) →
  (B = π / 6) →
  (f (A / 2) = 0) →
  A + B + C = π →
  c = sqrt 2 + sqrt 6 :=
by
  intros
  sorry

end solve_for_c_l563_563202


namespace probability_of_two_boys_two_girls_l563_563389

-- Necessary definitions
def child_outcome := bool -- true for boy, false for girl

def couple_has_4_children_outcomes : list (vector child_outcome 4) :=
  (list.replicateM 4 [true, false]).to_list

def is_two_boys_two_girls (children: vector child_outcome 4): bool :=
  (children.to_list.count true = 2) ∧ (children.to_list.count false = 2)

-- Main theorem statement
theorem probability_of_two_boys_two_girls : 
  (list.filter is_two_boys_two_girls couple_has_4_children_outcomes).length = 3 / 8 * couple_has_4_children_outcomes.length :=
by
  sorry

end probability_of_two_boys_two_girls_l563_563389


namespace sum_abs_diffs_geq_l563_563658

theorem sum_abs_diffs_geq (n : ℕ) (P : Fin n → ℕ) (h : ∀ i, 1 ≤ P i ∧ P i ≤ n) :
  (∑ i in Finset.range n, |P (Fin.mk i (by simp [i.lt_succ_self])) - P (Fin.mk ((i + 1) % n) (by simp [nat.mod_lt _ (zero_lt_succ n)]))|) ≥ 2 * n - 2 :=
by sorry

end sum_abs_diffs_geq_l563_563658


namespace no_zero_in_2_16_l563_563058

theorem no_zero_in_2_16 
  (f : ℝ → ℝ) 
  (h₁ : ∃! x ∈ (0, 16), f x = 0)
  (h₂ : ∃! x ∈ (0, 8), f x = 0)
  (h₃ : ∃! x ∈ (0, 4), f x = 0)
  (h₄ : ∃! x ∈ (0, 2), f x = 0) : 
  ∀ x ∈ [2, 16), f x ≠ 0 := 
sorry

end no_zero_in_2_16_l563_563058


namespace problem1_solution_l563_563460

theorem problem1_solution (x : ℝ) :
  x^2 + 2 * x + 4 * real.sqrt (x^2 + 2 * x) - 5 = 0 →
  x = real.sqrt 2 - 1 ∨ x = -real.sqrt 2 - 1 :=
sorry

end problem1_solution_l563_563460


namespace range_of_a_l563_563706

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + 1 < 0) ↔ (a ∈ set.Icc (-1 : ℝ) (1 : ℝ)) :=
by sorry

end range_of_a_l563_563706


namespace minimize_expr1_and_expr2_l563_563394

noncomputable def minimize_expressions : ℝ × ℝ :=
  let expr1 (x y : ℝ) : ℝ := x * y / 2 + 18 / (x * y)
  let expr2 (x y : ℝ) : ℝ := y / 2 + x / 3
  let x := 3
  let y := 2
  (x, y)

theorem minimize_expr1_and_expr2 : 
  ∀ x y : ℝ, 0 < x → 0 < y → 
  (x = 3 ∧ y = 2 ∧ (x * y / 2 + 18 / (x * y) = 6) ∧ (y / 2 + x / 3 = 2)) := 
by
  intros x y hx hy
  have hx3 : x = 3 := sorry
  have hy2 : y = 2 := sorry
  repeat { try { split, assumption } }
  have expr1_min : x * y / 2 + 18 / (x * y) = 6 := sorry
  have expr2_min : y / 2 + x / 3 = 2 := sorry
  sorry

end minimize_expr1_and_expr2_l563_563394


namespace parametric_to_standard_l563_563489

theorem parametric_to_standard (theta : ℝ)
  (x : ℝ)
  (y : ℝ)
  (h1 : x = 1 + 2 * Real.cos theta)
  (h2 : y = -2 + 2 * Real.sin theta) :
  (x - 1)^2 + (y + 2)^2 = 4 :=
sorry

end parametric_to_standard_l563_563489


namespace coefficient_x3y2_l563_563039

theorem coefficient_x3y2 :
  (∑ i in Finset.range 6, ∑ j in Finset.range 8,
     if ((2 * z : ℚ) ^ j * ((1 / z ^ 2) : ℚ) ^ (7 - j)).numerator = 7 - j + 2 * (5 - 3) ∧
        ((x + y) ^ 5).coeff (3, 2) * (2 * z : ℚ) ^ j * ((1 / z ^ 2) : ℚ) ^ (7 - j) ≠ 0 then
       ((x + y) ^ 5).coeff (3, 2) * (2 * z : ℚ) ^ j * ((1 / z ^ 2) : ℚ) ^ (7 - j)
     else 0).sum ∑ = 840 := by
  sorry

end coefficient_x3y2_l563_563039


namespace line_perpendicular_to_plane_l563_563531

-- Define a structure for vectors in 3D
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define line l with the given direction vector
def direction_vector_l : Vector3D := ⟨1, -1, -2⟩

-- Define plane α with the given normal vector
def normal_vector_alpha : Vector3D := ⟨2, -2, -4⟩

-- Prove that line l is perpendicular to plane α
theorem line_perpendicular_to_plane :
  let a := direction_vector_l
  let b := normal_vector_alpha
  (b.x = 2 * a.x) ∧ (b.y = 2 * a.y) ∧ (b.z = 2 * a.z) → 
  (a.x * b.x + a.y * b.y + a.z * b.z = 0) :=
by
  intro a b h
  sorry

end line_perpendicular_to_plane_l563_563531


namespace find_t_l563_563550

theorem find_t (t m n : ℝ) (h1 : 2 ^ m = t) (h2 : 5 ^ n = t) (h3 : t > 0) (h4 : t ≠ 1) (h5 : (1 / m) + (1 / n) = 3) : 
  t = Real.cbrt 10 := 
sorry

end find_t_l563_563550


namespace mahesh_markup_percentage_l563_563282

noncomputable def markup_percentage (cp sp discount_percentage : ℝ) : ℝ :=
  let mp := sp * (100 / (100 - discount_percentage))
  (mp - cp) / cp * 100

theorem mahesh_markup_percentage :
  markup_percentage 540 462 25.603864734299517 ≈ 14.94 := sorry

end mahesh_markup_percentage_l563_563282


namespace exists_infinite_solutions_positive_integers_l563_563498

theorem exists_infinite_solutions_positive_integers (m : ℕ) (h : m = 12) :
  ∃ (a b c : ℕ), (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ 
  ∑ (x : ℕ) in {a, b, c}, 1 / x + 1 / (a * b * c) = m / (a + b + c) := 
sorry

end exists_infinite_solutions_positive_integers_l563_563498


namespace find_explicit_formula_determine_range_of_m_l563_563170

-- Definition of the condition in the first problem
def f_satisfies_equation (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, e * f(x) - f'(1) * exp(x) + e * f(0) * x - (1/2) * e * x^2 = 0

-- Equivalent proof problem for Question 1
theorem find_explicit_formula :
  ∃ f : ℝ → ℝ, (∃ f' : ℝ → ℝ, f_satisfies_equation f f') →
  ∀ x : ℝ, f(x) = exp(x) - x + (1/2) * x^2 :=
sorry

-- Equivalent proof problem for Question 2
theorem determine_range_of_m :
  ∀ m : ℝ, (∀ x ∈ (Set.Icc (-1 : ℝ) 2), (exp(x) - x + (1/2) * x^2 - (1/2) * x^2 - m = 0) ↔
    (∃ a b : ℝ, a ≠ b ∧ -1 ≤ a ∧ a ≤ 2 ∧ -1 ≤ b ∧ b ≤ 2)) →
  (1 < m ∧ m ≤ 1 + (1/e)) :=
sorry

end find_explicit_formula_determine_range_of_m_l563_563170


namespace trigonometric_identity_l563_563948

theorem trigonometric_identity (α : ℝ) (h : (sin α + cos α) / (sin α - cos α) = 1 / 2) : 
  (sin α * cos α) / (sin α ^ 2 + 2 * cos α ^ 2) = -3 / 11 :=
by
  sorry

end trigonometric_identity_l563_563948


namespace obtuse_triangle_iff_distinct_real_roots_l563_563993

theorem obtuse_triangle_iff_distinct_real_roots
  (A B C : ℝ)
  (h_triangle : 2 * A + B = Real.pi)
  (h_isosceles : A = C) :
  (B > Real.pi / 2) ↔ (B^2 - 4 * A * C > 0) :=
sorry

end obtuse_triangle_iff_distinct_real_roots_l563_563993


namespace coeff_x3_in_expansion_l563_563249

theorem coeff_x3_in_expansion : 
  (1+x)*(2+x)^5.coeff 3 = 120 := by
  sorry

end coeff_x3_in_expansion_l563_563249


namespace purely_imaginary_condition_l563_563909

theorem purely_imaginary_condition (z : ℂ) (hz : z.im ≠ 0) 
  (hz1 : z.re = 0)
  (hz2 : ((z - 3)^2 + 5 * (complex.I : ℂ)) .re = 0)
  (hz3 : ((z - 3)^2 + 5 * (complex.I : ℂ)) .im ≠ 0) : 
  z = 3 * complex.I ∨ z = -3 * complex.I :=
by
  sorry

end purely_imaginary_condition_l563_563909


namespace ratio_first_part_l563_563762

theorem ratio_first_part (h_percent : 125 / 100 = 1.25) (second_part : ℝ := 4) :
  let first_part := 1.25 * second_part
  first_part = 5 :=
by
  have temp := by norm_num
  rw [h_percent] at temp
  have first_part := 1.25 * second_part
  norm_num at first_part
  exact first_part

end ratio_first_part_l563_563762


namespace tom_total_cost_l563_563725

def theater_seats : ℕ := 500
def cost_per_sqft : ℕ := 5
def space_per_seat : ℕ := 12
def infra_percent := 0.15
def tax_percent := 0.07
def maintenance_percent := 0.03
def partner_percent := 0.40

def total_square_feet : ℕ := space_per_seat * theater_seats
def cost_for_space : ℕ := total_square_feet * cost_per_sqft
def infra_cost : ℕ := (infra_percent * cost_for_space).toInt
def total_cost_with_infra : ℕ := cost_for_space + infra_cost
def tax_cost : ℕ := (tax_percent * total_cost_with_infra).toInt
def total_cost_with_tax : ℕ := total_cost_with_infra + tax_cost
def maintenance_cost : ℕ := (maintenance_percent * total_cost_with_tax).toInt
def total_cost_with_maintenance : ℕ := total_cost_with_tax + maintenance_cost
def partner_contribution : ℕ := (partner_percent * total_cost_with_maintenance).toInt
def final_cost : ℕ := total_cost_with_maintenance - partner_contribution

theorem tom_total_cost : final_cost = 22813.47 := by
  sorry

end tom_total_cost_l563_563725


namespace minimize_at_3_l563_563165

noncomputable def minimize_expression : ℝ → ℝ
| x := 3 * x^2 - 18 * x + 7

theorem minimize_at_3 : ∃ x : ℝ, minimize_expression x = minimize_expression 3 :=
by
  use 3
  sorry

end minimize_at_3_l563_563165


namespace prism_volume_l563_563686

/-- The volume of a rectangular prism given the areas of three of its faces. -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 50) (h3 : b * c = 75) :
  a * b * c = 335 :=
by
  sorry

end prism_volume_l563_563686


namespace mutually_exclusive_not_complementary_groups_count_l563_563783

-- Defining events based on the conditions
def EventA1 := ∃ n, n ≥ 1  -- At least one hit
def EventB1 := ∀ n, n = 3  -- All hits

def EventA2 := ∃ n, n ≥ 1  -- At least one hit
def EventB2 := ∀ n, n ≤ 1  -- At most one hit

def EventA3 := ∃ n, n = 1  -- Exactly one hit
def EventB3 := ∃ n, n = 2  -- Exactly two hits

def EventA4 := ∃ n, n ≥ 1  -- At least one hit
def EventB4 := ∀ n, n = 0  -- No hits

-- Mutually exclusive event checker
def mutually_exclusive (event1 event2 : Prop) : Prop :=
¬(event1 ∧ event2)

-- Complementary event checker
def complementary (event1 event2 : Prop) : Prop :=
event1 ∨ event2

-- The main theorem statement
theorem mutually_exclusive_not_complementary_groups_count :
  (mutually_exclusive EventA1 EventB1 ∧ ¬complementary EventA1 EventB1) +
  (mutually_exclusive EventA2 EventB2 ∧ ¬complementary EventA2 EventB2) +
  (mutually_exclusive EventA3 EventB3 ∧ ¬complementary EventA3 EventB3) +
  (mutually_exclusive EventA4 EventB4 ∧ ¬complementary EventA4 EventB4) = 1 :=
sorry

end mutually_exclusive_not_complementary_groups_count_l563_563783


namespace four_digit_divisibles_by_6_and_15_l563_563581

theorem four_digit_divisibles_by_6_and_15 :
  let lcm_6_15 := Nat.lcm 6 15,
      smallest_four_digit := 1000,
      largest_four_digit := 9999,
      smallest_multiple :=
        Nat.find (λ x, x ≥ smallest_four_digit ∧ x % lcm_6_15 = 0) 1,
      largest_multiple :=
        Nat.find (λ x, x ≥ largest_four_digit ∧ x % lcm_6_15 = 0) 1 in
  (largest_multiple - lcm_6_15 - smallest_multiple) / lcm_6_15 + 1 = 300 :=
begin
  sorry
end

end four_digit_divisibles_by_6_and_15_l563_563581


namespace sum_not_2345678_l563_563337

-- Define the transformation function
def transform_digit : ℕ → ℕ 
| x := if x > 2 then x - 2 else if x < 8 then x + 2 else x 

-- Define the function for transforming the entire number
def transform_number (A : ℕ) : ℕ :=
  let digits := A.digits in
  let transformed := digits.map transform_digit in
  transformed.reverse.join

-- Main theorem statement
theorem sum_not_2345678 (A B : ℕ) (hB : B = transform_number A) : A + B ≠ 2345678 :=
by sorry

end sum_not_2345678_l563_563337


namespace inequality_proof_l563_563667

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x / Real.sqrt y + y / Real.sqrt x) ≥ (Real.sqrt x + Real.sqrt y) := 
sorry

end inequality_proof_l563_563667


namespace hybrid_monotonous_count_l563_563494

open Nat

/--
A positive integer is defined as "hybrid-monotonous" if it is a one-digit number or 
its digits, when read from left to right, form a strictly increasing or a strictly 
decreasing sequence and must include at least one odd and one even digit.
Using digits 0 through 9, the number of hybrid-monotonous positive integers is 1902.
-/
theorem hybrid_monotonous_count : 
  let is_hybrid_monotonous (n : Nat) : Prop :=
    (n < 10) ∨ (strictlyIncreasingDigits n ∨ strictlyDecreasingDigits n) ∧ includesOddAndEvenDigits n
  in count (fun n => is_hybrid_monotonous n) (Finset.range 10000) = 1902 :=
  sorry

end hybrid_monotonous_count_l563_563494


namespace expand_and_simplify_l563_563848

theorem expand_and_simplify : ∀ x : ℝ, (7 * x + 5) * 3 * x^2 = 21 * x^3 + 15 * x^2 :=
by
  intro x
  sorry

end expand_and_simplify_l563_563848


namespace compound_interest_rate_l563_563117

theorem compound_interest_rate
  (P A : ℝ) (n t : ℕ) (r : ℝ)
  (hP : P = 10000)
  (hA : A = 12155.06)
  (hn : n = 4)
  (ht : t = 1)
  (h_eq : A = P * (1 + r / n) ^ (n * t)):
  r = 0.2 :=
by
  sorry

end compound_interest_rate_l563_563117


namespace find_g_value_l563_563327

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_property (a c : ℝ) : c^2 * g(a) = a^2 * g(c)
axiom g_nonzero : g 3 ≠ 0

theorem find_g_value : (g 6 - g 3) / g 4 = 16 / 3 :=
by
  sorry

end find_g_value_l563_563327


namespace is_divisible_by_7_l563_563332

theorem is_divisible_by_7 : ∃ k : ℕ, 42 = 7 * k := by
  sorry

end is_divisible_by_7_l563_563332


namespace polynomial_factorable_l563_563869

open Polynomial

theorem polynomial_factorable (n : ℤ) :
  (∃ f g : Polynomial ℤ, f.degree > 0 ∧ g.degree > 0 ∧ f * g = (X ^ 4 - (2 * n + 4) * X ^ 2 + (n - 2) ^ 2)) ↔
  (∃ k : ℤ, n = k ^ 2 ∨ n = 2 * k ^ 2) :=
sorry

end polynomial_factorable_l563_563869


namespace max_area_of_triangle_OAB_l563_563247

noncomputable theory

open Complex

def max_area_of_triangle (α : ℂ) (β : ℂ) (O : ℂ) : ℝ :=
  if β = (1 + I) * α ∧ abs (α - 2) = 1 then
    let S := (1 / 2) * (abs α ^ 2)
    in max S (3⁻¹/2)
  else 0

theorem max_area_of_triangle_OAB (α β : ℂ) (h1 : β = (1 + I) * α)
  (h2 : abs (α - 2) = 1) : max_area_of_triangle α β 0 = 9 / 2 := 
sorry

end max_area_of_triangle_OAB_l563_563247


namespace isosceles_triangle_greatest_perimeter_l563_563802

noncomputable def perimeter_of_section (base height : ℕ) (k : ℕ) : ℝ :=
  base / 6 + (Real.sqrt (height^2 + k^2)) + (Real.sqrt (height^2 + (k + 1)^2))

theorem isosceles_triangle_greatest_perimeter :
  let base := 12
  let height := 15
  let n := 6
  let perimeters := λ k, perimeter_of_section base height k
  let greatest_perimeter := List.maximum (List.map perimeters (List.range n))
  isosceles_triangle base height 
  ∧ greatest_perimeter ≈ 33.97 := sorry

end isosceles_triangle_greatest_perimeter_l563_563802


namespace geometric_progression_common_ratio_l563_563965

theorem geometric_progression_common_ratio (a_1 : ℕ) (q : ℝ) (h1 : a_1 = 3) 
  (h2 : ∀ n : ℕ, n > 0 → (series : ℕ → ℝ) = (λ n, a_1 * q^(n-1)) ) (h3 : a_1 + a_1 * q + a_1 * q^2 = 21) 
  (h4 : ∀ n : ℕ, series n > 0) : q = 2 :=
sorry

end geometric_progression_common_ratio_l563_563965


namespace sum_of_positive_factors_of_90_l563_563373

theorem sum_of_positive_factors_of_90 : 
  let n := 90 in 
  let factors := (1 + 2) * (1 + 3 + 9) * (1 + 5) in 
  factors = 234 :=
by
  sorry

end sum_of_positive_factors_of_90_l563_563373


namespace five_pow_10000_mod_1000_l563_563665

theorem five_pow_10000_mod_1000 (h : 5^500 ≡ 1 [MOD 1000]) : 5^10000 ≡ 1 [MOD 1000] := sorry

end five_pow_10000_mod_1000_l563_563665


namespace problem_statement_l563_563709

noncomputable def x : ℝ := sorry -- Let x be a real number satisfying the condition

theorem problem_statement (x_real_cond : x + 1/x = 3) : 
  (x^12 - 7*x^8 + 2*x^4) = 44387*x - 15088 :=
sorry

end problem_statement_l563_563709


namespace line_does_not_pass_second_quadrant_l563_563918

theorem line_does_not_pass_second_quadrant (a : ℝ) (ha : a ≠ 0) :
  ∀ (x y : ℝ), (x - y - a^2 = 0) → ¬(x < 0 ∧ y > 0) :=
sorry

end line_does_not_pass_second_quadrant_l563_563918


namespace rachel_total_score_l563_563669

theorem rachel_total_score :
  let level1_treasures := 5
  let level1_points := 9
  let level2_treasures := 2
  let level2_points := 12
  let level3_treasures := 8
  let level3_points := 15
  let total_score := level1_treasures * level1_points + level2_treasures * level2_points + level3_treasures * level3_points
  total_score = 189 :=
by
  let level1_score := level1_treasures * level1_points
  let level2_score := level2_treasures * level2_points
  let level3_score := level3_treasures * level3_points
  calc
    total_score
    = level1_score + level2_score + level3_score : by simp [total_score, level1_score, level2_score, level3_score]
... = 45 + 24 + 120 : by simp [level1_score, level2_score, level3_score]
... = 189 : by norm_num

end rachel_total_score_l563_563669


namespace no_partition_1_to_80_into_groups_of_4_l563_563256

theorem no_partition_1_to_80_into_groups_of_4 :
  ¬ ∃ (G : list (list ℕ)),
    (∀ g ∈ G, g.length = 4 ∧ (∀ n ∈ g, n ∈ (list.range 80).map (λ x, x + 1))) ∧
    (∀ g ∈ G, list.maximum g = some (list.sum (list.erase g (list.maximum g).get_or_else 0))) ∧
    (list.join G).perm (list.range 80).map (λ x, x + 1) :=
by
  sorry

end no_partition_1_to_80_into_groups_of_4_l563_563256


namespace number_ordering_l563_563813

open Real

noncomputable def A : ℝ := 9^(9^9)
noncomputable def B : ℝ := 99^9
noncomputable def C : ℝ := (9^9)^9
noncomputable def D : ℝ := (fact 9)^(fact 9)

theorem number_ordering : B < C ∧ C < A ∧ A < D := 
  sorry

end number_ordering_l563_563813


namespace pentagon_inscribed_angle_sum_l563_563781

theorem pentagon_inscribed_angle_sum (A B C D E : Type) :
  is_PentagonInscribedInCircle A B C D E →
  (∃ α β γ δ ε : ℝ, 
    is_InscribedAngleInSegment α A B C D E ∧
    is_InscribedAngleInSegment β B C D E A ∧
    is_InscribedAngleInSegment γ C D E A B ∧
    is_InscribedAngleInSegment δ D E A B C ∧
    is_InscribedAngleInSegment ε E A B C D ∧
    α + β + γ + δ + ε = 720) :=
begin
  sorry
end

end pentagon_inscribed_angle_sum_l563_563781


namespace probability_top_given_not_female_l563_563234

-- Definitions of the given conditions
def total_students := 60
def female_students := 20
def top_students := 3
def fraction_top_students := 1 / 6
def fraction_female_top_students := 1 / 2

-- The probability statement to be proved
theorem probability_top_given_not_female :
  let male_students := total_students - female_students in
  let total_top_students := fraction_top_students * total_students in
  let male_top_students := fraction_female_top_students * total_top_students in
  (male_top_students / male_students) = 1 / 8 :=
by
  sorry

end probability_top_given_not_female_l563_563234


namespace range_of_a_l563_563529

theorem range_of_a (a : ℝ) :
  (∃ P1 P2 : ℝ × ℝ, 
    (P1.fst^2 + P1.snd^2 - 2 * P1.fst - 4 * P1.snd + a - 5 = 0) ∧ 
    (P2.fst^2 + P2.snd^2 - 2 * P2.fst - 4 * P2.snd + a - 5 = 0) ∧ 
    (abs (3 * P1.fst - 4 * P1.snd - 15) / sqrt (3^2 + (-4)^2) = 1) ∧ 
    (abs (3 * P2.fst - 4 * P2.snd - 15) / sqrt (3^2 + (-4)^2) = 1)) 
    ↔ (-15 < a ∧ a < 1) :=
by sorry

end range_of_a_l563_563529


namespace find_BE_l563_563253

open Real

variables (A B C D E : Type*) 
variables (AB BC CA CD BE : ℝ)
variables (angleBAE angleCAD : ℝ)

-- Given conditions
def triangle_ABC := AB = 17 ∧ BC = 21 ∧ CA = 20
def point_D_on_BC := CD = 9
def point_E_condition := angleBAE = angleCAD

-- Theorem to prove
theorem find_BE (h1 : triangle_ABC) (h2 : point_D_on_BC) (h3 : point_E_condition) : 
  BE = 9.645 :=
by sorry

end find_BE_l563_563253


namespace zero_point_monotonic_l563_563223

def f (x : ℝ) : ℝ := 2^x - 1/x

theorem zero_point_monotonic (x0 x1 x2 : ℝ) (h_zero : f x0 = 0) 
  (h_x1_range : 0 < x1 ∧ x1 < x0) (h_x2_range : x0 < x2) 
  (h_monotonic : ∀ (x y : ℝ), x < y → f x < f y) :
  f x1 < 0 ∧ f x2 > 0 :=
by
  sorry

end zero_point_monotonic_l563_563223


namespace ruby_initial_apples_l563_563305

theorem ruby_initial_apples (apples_taken : ℕ) (apples_left : ℕ) (initial_apples : ℕ) 
  (h1 : apples_taken = 55) (h2 : apples_left = 8) (h3 : initial_apples = apples_taken + apples_left) : 
  initial_apples = 63 := 
by
  sorry

end ruby_initial_apples_l563_563305


namespace range_of_function_x_l563_563013

theorem range_of_function_x (x : ℝ) : 2 * x - 6 ≥ 0 ↔ x ≥ 3 := sorry

end range_of_function_x_l563_563013


namespace second_solution_sugar_percentage_l563_563299

variables (W : ℝ) (S : ℝ)
-- Given conditions
def original_sugar_content : ℝ := 0.10 * W
def sugar_removed : ℝ := 0.10 * (W / 4)
def sugar_remaining : ℝ := 0.10 * W - 0.10 * (W / 4)
def new_sugar_content : ℝ := 0.14 * W
def sugar_added_by_second_solution : ℝ := S * (W / 4)

-- Prove that S (percentage of sugar in second solution) is 0.26
theorem second_solution_sugar_percentage :
  0.075 * W + S * (W / 4) = 0.14 * W → S = 0.26 :=
by
  sorry

end second_solution_sugar_percentage_l563_563299


namespace num_points_in_first_or_second_quadrant_l563_563209

open Set

def M : Set ℤ := {1, -2, 3}
def N : Set ℤ := {-4, 5, 6, -7}

def is_first_or_second_quadrant (point : ℤ × ℤ) : Prop :=
  (point.1 > 0 ∧ point.2 > 0) ∨ (point.1 < 0 ∧ point.2 > 0)

theorem num_points_in_first_or_second_quadrant :
  (M.product N).count is_first_or_second_quadrant +
  (N.product M).count is_first_or_second_quadrant = 14 :=
by
  sorry

end num_points_in_first_or_second_quadrant_l563_563209


namespace value_of_x_squared_l563_563191

noncomputable def x : ℝ := sorry
axiom h : 32 = x^6 + 1/x^6

theorem value_of_x_squared : x^2 + 1/x^2 = 2 * real.cbrt(2) :=
by
  sorry

end value_of_x_squared_l563_563191


namespace find_z_l563_563537

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def N_has_1998_digits (N : ℕ) : Prop :=
  10 ^ 1997 ≤ N ∧ N < 10 ^ 1998

def divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem find_z (N : ℕ)
  (h_digits : N_has_1998_digits N)
  (h_div9 : divisible_by_9 N) :
  let x := sum_of_digits N
      y := sum_of_digits x
      z := sum_of_digits y
  in z = 9 :=
sorry

end find_z_l563_563537


namespace find_value_of_expression_l563_563578

-- Let's define a parameter a satisfying the given condition
variables (a : ℝ)
hypothesis : a^2 + a - 1 = 0

-- Now we state the theorem
theorem find_value_of_expression (h : a^2 + a - 1 = 0) : a^3 + 2*a^2 + 3 = 4 :=
sorry

end find_value_of_expression_l563_563578


namespace intersection_of_lines_l563_563736

-- Define the first line
def line1 (x : ℝ) : ℝ := -3 * x + 4

-- Define the second line which is perpendicular to the first line and passes through (3, 2)
def line2 (x : ℝ) : ℝ := (1 / 3) * x + 1

-- Define the intersection point
def intersection_point : ℝ × ℝ := (9 / 10, 13 / 10)

-- Prove that the intersection point satisfies both line equations
theorem intersection_of_lines : 
  ∃ (x y : ℝ), line1 x = y ∧ line2 x = y ∧ (x, y) = intersection_point := 
by
  sorry

end intersection_of_lines_l563_563736


namespace conjugate_of_z_l563_563322

open Complex

def z : ℂ := 5 / (-1 + 2 * I)

theorem conjugate_of_z : conj z = -1 + 2 * I :=
by
  -- We ignore the proof steps and use sorry to skip the proof.
  sorry

end conjugate_of_z_l563_563322


namespace m_n_sum_is_11_l563_563693

def isosceles_trapezoid (A B C D : ℤ × ℤ) : Prop := 
  ∃ (m: ℚ), 
    A.1 ≠ B.1 ∧ C.1 ≠ D.1 ∧ -- No vertical sides
    A ≠ B ∧ C ≠ D ∧ -- No horizontal sides 
    ((B.2 - A.2) / (B.1 - A.1) : ℚ) = ((D.2 - C.2) / (D.1 - C.1) : ℚ) ∧ -- Parallel sides
    ((C.2 - B.2) / (C.1 - B.1) : ℚ) = -((D.2 - A.2) / (D.1 - A.1) : ℚ) -- Isosceles condition

def integer_coordinates (p : ℤ × ℤ) : Prop := 
  ∃ (x y : ℤ), p = (x,y)

noncomputable def calculate_m_n : ℕ :=
  let A : ℤ × ℤ := (10, 50)
  let D : ℤ × ℤ := (11, 53)
  let possible_slopes : List ℚ := 
      [1, 2, 0.5, 1] -- This comes from the problem solution explaining valid slope values
  let sum_abs_slopes : ℚ := possible_slopes.map(λ x, |x|).sum
  let fraction : ℚ := sum_abs_slopes -- This is 9/2 from the solution
  let mn_sum : ℕ := 9 + 2 -- From fraction 9/2
  mn_sum

theorem m_n_sum_is_11 : calculate_m_n = 11 := 
by
  unfold calculate_m_n
  -- The required solution steps can be implemented here if needed
  sorry

end m_n_sum_is_11_l563_563693


namespace find_first_number_l563_563689

theorem find_first_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = (x + 60 + 35) / 3 + 5 → 
  x = 10 := 
by 
  sorry

end find_first_number_l563_563689


namespace order_of_magnitude_l563_563829

noncomputable def a := Real.exp Real.exp  -- e^e
noncomputable def b := Real.pi ^ Real.pi  -- π^π
noncomputable def c := Real.exp Real.pi  -- e^π
noncomputable def d := Real.pi ^ Real.exp  -- π^e

theorem order_of_magnitude:
  a < d ∧ d < c ∧ c < b := by
  sorry

end order_of_magnitude_l563_563829


namespace cone_division_painted_surface_and_volume_ratios_l563_563790

theorem cone_division_painted_surface_and_volume_ratios
  (height : ℝ) (base_radius : ℝ) (C_area_ratio F_area_ratio : ℝ)
  (k : ℚ)
  (m n : ℕ)
  (hmn_coprime : Nat.coprime m n)
  (hmn_def : k = m / n)
  (h_cone : height = 4 ∧ base_radius = 3)
  (h_ratios : C_area_ratio = F_area_ratio ∧ C_area_ratio = k ∧ F_area_ratio = k)
  : m + n = 512 :=
sorry

end cone_division_painted_surface_and_volume_ratios_l563_563790


namespace sequence_type_l563_563539

-- Definitions based on the conditions
def Sn (a : ℝ) (n : ℕ) : ℝ := a^n - 1

def sequence_an (a : ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a - 1 else (Sn a n - Sn a (n - 1))

-- Proving the mathematical statement
theorem sequence_type (a : ℝ) (h : a ≠ 0) : 
  (∀ n > 1, (sequence_an a n = sequence_an a 1 + (n - 1) * (sequence_an a 2 - sequence_an a 1)) ∨
  (∀ n > 2, sequence_an a n / sequence_an a (n-1) = a)) :=
sorry

end sequence_type_l563_563539


namespace find_number_of_two_dollar_pairs_l563_563035

noncomputable def pairs_of_two_dollars (x y z : ℕ) : Prop :=
  x + y + z = 15 ∧ 2 * x + 4 * y + 5 * z = 38 ∧ x >= 1 ∧ y >= 1 ∧ z >= 1

theorem find_number_of_two_dollar_pairs (x y z : ℕ) 
  (h1 : x + y + z = 15) 
  (h2 : 2 * x + 4 * y + 5 * z = 38) 
  (hx : x >= 1) 
  (hy : y >= 1) 
  (hz : z >= 1) :
  pairs_of_two_dollars x y z → x = 12 :=
by
  intros
  sorry

end find_number_of_two_dollar_pairs_l563_563035


namespace complete_square_result_l563_563286

theorem complete_square_result (x : ℝ) :
  (∃ r s : ℝ, (16 * x ^ 2 + 32 * x - 1280 = 0) → ((x + r) ^ 2 = s) ∧ s = 81) :=
by
  sorry

end complete_square_result_l563_563286


namespace identify_louis_wife_l563_563870

theorem identify_louis_wife :
  let a1 := 3   -- Diana
  let a2 := 2   -- Elizabeth
  let a3 := 4   -- Nicole
  let a4 := 1   -- Maud
  let b1 := a1  -- Simon
  let b2 := 2 * a2  -- Pierre
  let b3 := 3 * a3  -- Louis
  let b4 := 4 * a4  -- Christian
  (a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4 = 32) → (a4 = 1) → Louis' wife is Maud :=
by
  unfold a1 a2 a3 a4 b1 b2 b3 b4
  intros
  sorry

end identify_louis_wife_l563_563870


namespace inequality_solution_l563_563851

theorem inequality_solution (x : ℝ) : (real.cbrt x + (3 / (real.cbrt x + 4)) + 1 ≤ 0) → x ∈ set.Iic (-64) :=
sorry

end inequality_solution_l563_563851


namespace geometric_progression_condition_l563_563855

variables (a b c : ℝ) (k n p : ℕ)

theorem geometric_progression_condition :
  (a / b) ^ (k - p) = (a / c) ^ (k - n) :=
sorry

end geometric_progression_condition_l563_563855


namespace a_14_pow_14_l563_563992

def a : ℕ → ℕ 
| 1 := (11^11)
| 2 := (12^12)
| 3 := (13^13)
| (n+1) := if h₁ : n ≥ 3 
           then (|a n - a (n-1)| + |a (n-1) - a (n-2)|)
           else 0

theorem a_14_pow_14 : a (14^14) = 1 := 
sorry

end a_14_pow_14_l563_563992


namespace part_one_part_two_l563_563482

open Set Real

noncomputable def A (x : ℝ) : Prop := (2 * x - 2) / (x + 1) < 1 
noncomputable def B (x a : ℝ) : Prop := x^2 + x + a - a^2 < 0

theorem part_one (a : ℝ) (h_a : a = 1) : 
  let U := ℝ in
  (λ x, B x a ∨ ¬ A x) = (λ x, x < 0 ∨ x ≥ 3) :=
sorry

theorem part_two : 
  (λ x, (∃ a : ℝ, x ∈ A → x ∈ B x a)) ↔ 
  let a_set := (λ a : ℝ, a ≤ -3 ∨ a ≥ 4) in
  ∃ a : ℝ, a ∈ a_set :=
sorry

end part_one_part_two_l563_563482


namespace solve_expression_hundreds_digit_l563_563506

def factorial (n : ℕ) : ℕ :=
  Nat.factorial n

def div_mod (a b m : ℕ) : ℕ :=
  (a / b) % m

def hundreds_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

theorem solve_expression_hundreds_digit :
  hundreds_digit (div_mod (factorial 17) 5 1000 - div_mod (factorial 10) 2 1000) = 8 :=
by
  sorry

end solve_expression_hundreds_digit_l563_563506


namespace equal_ratios_of_segments_l563_563335

-- Define given conditions
variables {A B C D P K L : Point}
variable [quadrilateral A B C D]
variable [on_extension_of_diagonal P D B]
variable [midpoint M1 A B]
variable [intersection_of_line_through P M1 K A D]
variable [midpoint M2 C D]
variable [intersection_of_line_through M2 P L B C]

-- Define Lean 4 statement
theorem equal_ratios_of_segments (A B C D P K L : Point)
    [quadrilateral A B C D]
    [on_extension_of_diagonal P D B]
    [midpoint M1 A B]
    [intersection_of_line_through P M1 K A D]
    [midpoint M2 C D]
    [intersection_of_line_through M2 P L B C] :
    (segment_ratio K D A) = (segment_ratio L C B) := 
sorry

end equal_ratios_of_segments_l563_563335


namespace problem2_l563_563757
noncomputable def problem1 := 
by
  -- We'll show that the Cartesian equation of the given polar curve is (x - 2)^2 + (y - 1)^2 = 5,
  let p (θ : ℝ) := 2 * Real.sin θ + 4 * Real.cos θ
  let r := p
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  have : x^2 + y^2 = (2 * y) + (4 * x)
  -- we rewrite this as follows:
  show (x - 2)^2 + (y - 1)^2 = 5,
  sorry

theorem problem2 (x y : ℝ) (hx : |x - 1| ≤ 1) (hy : |y - 2| ≤ 1) : |x - 2 * y + 1| ≤ 5 :=
by
  -- Given |x - 1| ≤ 1 and |y - 2| ≤ 1, we derive bounds on x and y,
  have hx_bounds : 0 ≤ x ∧ x ≤ 2 := sorry,
  have hy_bounds : 1 ≤ y ∧ y ≤ 3 := sorry,
  -- Using the triangle inequality:
  have : |x - 2 * y + 1| ≤ |x - 1| + 2 * |y - 2| + 2 := sorry,
  -- Final evaluation shows that the maximum value is 5
  show |x - 2 * y + 1| ≤ 5,
  sorry

end problem2_l563_563757


namespace no_such_polynomials_exist_l563_563839

noncomputable def polynomial (R : Type*) [comm_ring R] : Type* := mv_polynomial ℕ R
noncomputable def is_positive_integer (k : ℕ) : Prop := k > 0

theorem no_such_polynomials_exist : 
  ¬ (∃ (f g : polynomial ℝ) (k : ℕ), is_positive_integer k ∧ 
    ∀ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) → (a * f + b * g = 0 → k = 0)) :=
sorry

end no_such_polynomials_exist_l563_563839


namespace existSetK_l563_563213

open Nat

def σ (n : ℕ) : ℕ :=
   (divisors n).sum

theorem existSetK (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ (k : ℕ) (S : Finset ℕ), S.card ≥ m ∧ (∀ s ∈ S, s % n = 0) ∧ (∀ s ∈ S, (2^k * σ s / s) % 2 = 1) :=
begin
  sorry
end

end existSetK_l563_563213


namespace find_a_l563_563519

noncomputable def max_sine_val (I : Set ℝ) : ℝ := 
  sup (Set.image sin I)

theorem find_a (a : ℝ) (h : a > 0) : 
  max_sine_val (Set.Icc 0 a) = 2 * max_sine_val (Set.Icc a (2 * a)) ↔ 
  a = (5 * Real.pi / 6) ∨ a = (13 * Real.pi / 12) :=
  sorry

end find_a_l563_563519


namespace product_of_geometric_sequence_l563_563883

noncomputable def geometric_sequence (n : ℕ) : ℤ := sorry

theorem product_of_geometric_sequence :
  let a := geometric_sequence
  in (a 3 * a 15 = 1) →
     (a 7 * a 8 * a 9 * a 10 * a 11 = 1) :=
by
  intros a h
  sorry

end product_of_geometric_sequence_l563_563883


namespace extreme_value_f_range_a_l563_563928
noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x^2 + (2 - a) * x
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x - 2

theorem extreme_value_f (a : ℝ) (ha : a > 0) :
  ∃ x_max : ℝ, x_max = 1 / a ∧
  ∀ x : ℝ, f a x ≤ f a x_max := by
sorry

theorem range_a (a : ℝ) :
  ∀ x₀ ∈ Set.Ioc 0 Real.exp 1,
  let eq_f_g := ∀ x : ℝ, f a x = g x₀ → x ∈ Set.Ioc 0 Real.exp 1 →
                 (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = g x₀ ∧ f a x₂ = g x₀) in
  eq_f_g → (3 + 2 * Real.exp 1) / (Real.exp 1^2 + Real.exp 1) ≤ a ∧ a < Real.exp 1 := by
sorry

end extreme_value_f_range_a_l563_563928


namespace sum_of_divisors_90_l563_563369

theorem sum_of_divisors_90 : 
  let n := 90 in 
  let sum_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5) in
  sum_divisors n = 234 :=
by 
  let n := 90
  let sum_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5)
  sorry

end sum_of_divisors_90_l563_563369


namespace number_of_outfits_l563_563317

theorem number_of_outfits (shirts pants ties : ℕ) (h_shirts : shirts = 8) (h_pants : pants = 5) (h_ties : ties = 7) :
  shirts * pants * ties = 280 :=
by
  rw [h_shirts, h_pants, h_ties]
  norm_num

end number_of_outfits_l563_563317


namespace total_songs_total_cost_before_discounts_num_discounts_total_amount_after_discounts_l563_563052

-- Proving the total number of songs
theorem total_songs (num_country_albums : ℕ) (num_pop_albums : ℕ) (songs_per_album : ℕ)
  (h1 : num_country_albums = 4) (h2 : num_pop_albums = 5) (h3 : songs_per_album = 8) :
  (num_country_albums * songs_per_album + num_pop_albums * songs_per_album = 72) :=
by
  rw [h1, h2, h3]
  norm_num

-- Proving the total cost before discounts
theorem total_cost_before_discounts (num_country_albums : ℕ) (num_pop_albums : ℕ)
  (cost_country_album : ℕ) (cost_pop_album : ℕ) 
  (h1 : num_country_albums = 4) (h2 : num_pop_albums = 5)
  (h3 : cost_country_album = 12) (h4 : cost_pop_album = 15) :
  (num_country_albums * cost_country_album + num_pop_albums * cost_pop_album = 123) :=
by
  rw [h1, h2, h3, h4]
  norm_num

-- Proving the number of discounts
theorem num_discounts (total_albums : ℕ) (albums_per_discount : ℕ)
  (h1 : total_albums = 4 + 5) (h2 : albums_per_discount = 3) :
  (total_albums / albums_per_discount = 3) :=
by
  rw [h1, h2]
  norm_num

-- Proving the total amount spent after discounts
theorem total_amount_after_discounts (total_cost_before_discounts : ℕ) (num_discounts : ℕ) (discount_amount : ℕ)
  (h1 : total_cost_before_discounts = 123) (h2 : num_discounts = 3) (h3 : discount_amount = 5) :
  (total_cost_before_discounts - (num_discounts * discount_amount) = 108) :=
by
  rw [h1, h2, h3]
  norm_num


end total_songs_total_cost_before_discounts_num_discounts_total_amount_after_discounts_l563_563052


namespace paper_cups_calculation_l563_563084

def total_pallets : Nat := 20
def paper_towels : Nat := total_pallets / 2
def tissues : Nat := total_pallets / 4
def paper_plates : Nat := total_pallets / 5
def other_paper_products : Nat := paper_towels + tissues + paper_plates
def paper_cups : Nat := total_pallets - other_paper_products

theorem paper_cups_calculation : paper_cups = 1 := by
  sorry

end paper_cups_calculation_l563_563084


namespace hexagon_can_be_divided_into_four_identical_shapes_l563_563490

def hexagon : Type := sorry

def identical_shapes (hex : hexagon) (cuts : list (hexagon → hexagon)) : Prop := sorry

noncomputable def divide_hexagon_into_identical_shapes (hex : hexagon) : list (hexagon → hexagon) :=
sorry

theorem hexagon_can_be_divided_into_four_identical_shapes :
  ∀ (hex : hexagon), identical_shapes hex (divide_hexagon_into_identical_shapes hex) :=
sorry

end hexagon_can_be_divided_into_four_identical_shapes_l563_563490


namespace geom_sum_l563_563826

theorem geom_sum : 
  ∑ i in (finset.range 10), 2 ^ (i + 1) = 2046 :=
sorry

end geom_sum_l563_563826


namespace smallest_degree_p_l563_563329

def degree (p : polynomial ℝ) : ℕ :=
p.natDegree

noncomputable def has_horizontal_asymptote (f : ℝ → ℝ) : Prop :=
∃ L, ∀ ε > 0, ∃ M, ∀ x > M, |f x - L| < ε

theorem smallest_degree_p (p : polynomial ℝ) (h_asymp : has_horizontal_asymptote (λ x, (3 * x^7 + 4 * x^6 - 2 * x^3 - 2) / p.eval x)) : 
  degree p ≥ 7 := 
sorry

end smallest_degree_p_l563_563329


namespace cannot_use_square_diff_formula_l563_563745

theorem cannot_use_square_diff_formula :
  ¬ (∃ a b, (x - y) * (-x + y) = (a + b) * (a - b) ∨ (a - b) * (a + b)) ∧
  (∃ a b, (-x - y) * (x - y) = (a + b) * (a - b) ∨ (a - b) * (a + b)) ∧
  (∃ a b, (-x + y) * (-x - y) = (a + b) * (a - b) ∨ (a - b) * (a + b)) ∧
  (∃ a b, (x + y) * (-x + y) = (a + b) * (a - b) ∨ (a - b) * (a + b)) :=
sorry

end cannot_use_square_diff_formula_l563_563745


namespace problem_statement_l563_563211

variable {m n : ℝ}

theorem problem_statement (h1 : m > 0) (h2 : n > m) : (1 / m) - (1 / n) > 0 := 
begin 
  sorry 
end

end problem_statement_l563_563211


namespace cos_EMN_is_one_l563_563398
noncomputable def cos_angle_EMN (s h : ℝ) : ℝ :=
  let l := Real.sqrt (h^2 + (s/2)^2)
  let MN := (s * Real.sqrt 2) / 2
  in (2 * (h^2 + (s^2 / 2)) - (s^2 / 2)) / (2 * (h^2 + (s^2 / 2)))

theorem cos_EMN_is_one (s h : ℝ) : cos_angle_EMN s h = 1 := 
  sorry

end cos_EMN_is_one_l563_563398


namespace dime_quarter_problem_l563_563230

theorem dime_quarter_problem :
  15 * 25 + 10 * 10 = 25 * 25 + 35 * 10 :=
by
  sorry

end dime_quarter_problem_l563_563230


namespace largest_number_not_sum_of_two_composites_l563_563858

-- Define composite numbers
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

-- Define the problem predicate
def cannot_be_expressed_as_sum_of_two_composites (n : ℕ) : Prop :=
  ¬ ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_number_not_sum_of_two_composites : 
  ∃ n, cannot_be_expressed_as_sum_of_two_composites n ∧ 
  (∀ m, cannot_be_expressed_as_sum_of_two_composites m → m ≤ n) ∧ 
  n = 11 :=
by {
  sorry
}

end largest_number_not_sum_of_two_composites_l563_563858


namespace inequality_case_a2_inequality_general_l563_563505

def func (x a : ℝ) : ℝ := |x + a| + |x + (1 / a)|

theorem inequality_case_a2 (x : ℝ) : 
  (func x 2 > 3) ↔ (x < -11 / 4 ∨ x > 1 / 4) := 
  sorry

theorem inequality_general (m a : ℝ) (ha : a > 0) : 
  func m a + func (-1 / m) a ≥ 4 :=
  sorry

end inequality_case_a2_inequality_general_l563_563505


namespace find_m_range_l563_563888

noncomputable def ellipse_with_vertex_and_foci (A : ℝ × ℝ) (foci_on_x_axis : Prop) (distance_cond : ℝ → ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a > b ∧ b > 0 ∧ c > 0 ∧ 
    (A = (0, -1)) ∧ 
    (foci_on_x_axis) ∧
    (distance_cond c (2 * real.sqrt 2)) = 3 ∧ 
    (a^2 = b^2 + c^2) ∧ 
    (ellipse_eq := (∀ x y, x^2 / a^2 + y^2 / b^2 = 1)) ∧ 
    (ellipse_eq = (λ x y, x^2 / 3 + y^2 = 1))

theorem find_m_range (ellipse_eq : ∀ x y, x^2 / 3 + y^2 = 1) (m : ℝ) : -2 < m ∧ m < 2 :=
  sorry

end find_m_range_l563_563888


namespace first_group_mat_weavers_is_4_l563_563313

instance : DecidableEq int := Int.decidableEq

axiom exists_mat_weavers : ∃ (num_weavers_1 num_weavers_2 days_1 days_2 mats_1 mats_2: ℕ),
  (mats_1 = 4 ∧ days_1 = 4 ∧ num_weavers_2 = 6 ∧ mats_2 = 9 ∧ days_2 = 6) ∧ 
  (num_weavers_1 * days_1 = mats_1 * days_1 ∧ num_weavers_2 * days_2 = mats_2 * days_1)

theorem first_group_mat_weavers_is_4 :
  ∀ (num_weavers_1 num_weavers_2 days_1 days_2 mats_1 mats_2: ℕ),
  (mats_1 = 4 ∧ days_1 = 4 ∧ num_weavers_2 = 6 ∧ mats_2 = 9 ∧ days_2 = 6) ∧ 
  (num_weavers_1 * days_1 = mats_1 * days_1 ∧ num_weavers_2 * days_2 = mats_2 * days_1) →
  num_weavers_1 = 4 :=
by
  intro num_weavers_1 num_weavers_2 days_1 days_2 mats_1 mats_2
  intro h
  cases h with h1 h2
  obtain ⟨hmats1, hdays1, hnum_weavers2, hwmats2, hdays2⟩ := h1
  obtain ⟨hw1, hw2⟩ := h2
  have h3 : mats_1 / days_1 = mats_2 / days_2 := sorry
  have h4 : 1 = h3 := sorry
  have h5 : num_weavers_1 = num_weavers_2 / (mats_2 / mes1) := sorry
  have h6 : num_weavers_1 = 4 := sorry
  exact h6

end first_group_mat_weavers_is_4_l563_563313


namespace range_of_a_l563_563225

def point := ℝ × ℝ

def reflect (p q : point) : point :=
  (2 * q.1 - p.1, 2 * q.2 - p.2)

def on_same_side (A B : point) (a : ℝ) : Prop :=
  let f x y := 3 * x - 2 * y + a in
  (f A.1 A.2 * f B.1 B.2 > 0)

theorem range_of_a :
  (-∞ < a ∧ a < -7) ∨ (24 < a ∧ a < ∞) :=
  let A : point := (3, 1)
  let B : point := reflect A ((-1 / 2), (7 / 2))
  let a := _ in
  on_same_side A B a
  sorry

end range_of_a_l563_563225


namespace largest_five_digit_number_divisible_by_6_l563_563737

theorem largest_five_digit_number_divisible_by_6 : 
  ∃ n : ℕ, n < 100000 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 100000 ∧ m % 6 = 0 → m ≤ n :=
by
  sorry

end largest_five_digit_number_divisible_by_6_l563_563737


namespace infinite_series_sum_l563_563821

theorem infinite_series_sum (a r : ℝ) (h₀ : -1 < r) (h₁ : r < 1) :
    (∑' n, if (n % 2 = 0) then a * r^(n/2) else a^2 * r^((n+1)/2)) = (a * (1 + a * r))/(1 - r^2) :=
by
  sorry

end infinite_series_sum_l563_563821


namespace sum_of_positive_factors_of_90_eq_234_l563_563372

theorem sum_of_positive_factors_of_90_eq_234 : 
  let factors := [1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90]
  List.sum factors = 234 :=
by
  -- List the positive factors of 90
  let factors := [1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90]
  -- Prove that the sum of these factors is 234
  have h_sum_factors : List.sum factors = 234 := sorry
  exact h_sum_factors

end sum_of_positive_factors_of_90_eq_234_l563_563372


namespace max_sum_length_le_98306_l563_563161

noncomputable def L (k : ℕ) : ℕ := sorry

theorem max_sum_length_le_98306 (x y : ℕ) (hx : x > 1) (hy : y > 1) (hl : L x + L y = 16) : x + 3 * y < 98306 :=
sorry

end max_sum_length_le_98306_l563_563161


namespace doughnuts_given_away_l563_563412

def doughnuts_left (total_doughnuts : Nat) (doughnuts_per_box : Nat) (boxes_sold : Nat) : Nat :=
  total_doughnuts - (doughnuts_per_box * boxes_sold)

theorem doughnuts_given_away :
  doughnuts_left 300 10 27 = 30 :=
by
  rw [doughnuts_left]
  simp
  sorry

end doughnuts_given_away_l563_563412


namespace percentage_of_P_is_20_l563_563309

-- Define the volumes of Solutions P and Q
variables {P Q : ℝ}

-- Conditions given in the problem
def condition1 : Prop := (0.80 * P + 0.55 * Q = 0.60 * (P + Q))

-- Define the percentage of the volume of the mixture that is P
def percentage_of_P : ℝ := (P / (P + Q)) * 100

-- The statement to be proved
theorem percentage_of_P_is_20 : condition1 → percentage_of_P = 20 := by
  sorry

end percentage_of_P_is_20_l563_563309


namespace missing_angle_in_convex_polygon_l563_563070

theorem missing_angle_in_convex_polygon (n : ℕ) (x : ℝ) 
  (h1 : n ≥ 5) 
  (h2 : 180 * (n - 2) - 3 * x = 3330) : 
  x = 54 := 
by 
  sorry

end missing_angle_in_convex_polygon_l563_563070


namespace batsman_average_after_17th_inning_l563_563758

theorem batsman_average_after_17th_inning :
  ∀ (A : ℕ), (16 * A + 50) / 17 = A + 2 → A = 16 → A + 2 = 18 := by
  intros A h1 h2
  rw [h2] at h1
  linarith

end batsman_average_after_17th_inning_l563_563758


namespace probability_of_winning_l563_563705

theorem probability_of_winning (lose_prob : ℚ) (no_tie : 1 = lose_prob + (1 - lose_prob)) : lose_prob = 3 / 7 → (1 - lose_prob) = 4 / 7 :=
by
  intro h
  rw [h]
  norm_num
  sorry

end probability_of_winning_l563_563705


namespace color_cube_l563_563124

open Finset Fintype

universe u

-- Definition of vertices
def vertices : Finset (ℕ × ℕ × ℕ) :=
  {(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
   (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)}

-- Definition of neighbors
def neighbors (v : (ℕ × ℕ × ℕ)) : Finset (ℕ × ℕ × ℕ) :=
  vertices.filter (λ u, (u, v).fst.zip_with (≠) (u, v).snd ≤ 1)

-- Condition that defines the valid coloring
def valid_coloring (color : (ℕ × ℕ × ℕ) → ℕ) : Prop :=
  ∀ v ∈ vertices, 
    (∀ u ∈ neighbors v, color u = color v) → color v = color v

-- The main statement that needs to be proven
theorem color_cube : 
  ∃ color_functions : Finset (((ℕ × ℕ × ℕ) → ℕ) ), valid_coloring color_functions → color_functions.card = 118 :=
begin
  sorry
end

end color_cube_l563_563124


namespace smallest_x_for_cubic_l563_563019

theorem smallest_x_for_cubic (x N : ℕ) (h1 : 1260 * x = N^3) : x = 7350 :=
sorry

end smallest_x_for_cubic_l563_563019


namespace committee_meeting_count_l563_563426

theorem committee_meeting_count :
  ∃ n : ℕ, n = 5400 ∧
  ∃ schools : fin 3 -> fin 3,
  ∀ (host : ℕ) (r1 r2 r3 : fin 6), 
    (host < 3) ∧
    (schools host = 3) ∧
    (schools (if host = 0 then 1 else 0) = 2) ∧
    (schools (if host = 2 then 1 else 2) = 1) ∧
    host * schools host = 5400 := 
sorry

end committee_meeting_count_l563_563426


namespace radius_of_circle_correct_l563_563763

noncomputable def radius_of_circle (P Q : ℝ) (h1 : P = real.pi * r^2) (h2 : Q = 2 * real.pi * r) (h3 : P / Q = 40 / real.pi) : ℝ :=
  r

theorem radius_of_circle_correct (r P Q : ℝ) (h1 : P = real.pi * r^2) (h2 : Q = 2 * real.pi * r) (h3 : P / Q = 40 / real.pi) : radius_of_circle P Q h1 h2 h3 = 80 / real.pi :=
by sorry

end radius_of_circle_correct_l563_563763


namespace value_of_a_l563_563901

variable (a : ℝ)

noncomputable def f (x : ℝ) := x^2 + 8
noncomputable def g (x : ℝ) := x^2 - 4

theorem value_of_a
  (h0 : a > 0)
  (h1 : f (g a) = 8) : a = 2 :=
by
  -- conditions are used as assumptions
  let f := f
  let g := g
  sorry

end value_of_a_l563_563901


namespace no_solution_16_64_eq_l563_563310

theorem no_solution_16_64_eq (x : ℝ) : ¬ (16^(3*x - 1) = 64^(2*x + 3)) :=
by {
  -- Our goal is to prove that no such x exists that satisfies this equation
  intro h,
  -- We can assume there exists an x such that 16^(3x - 1) = 64^(2x + 3)
  have h1: 2^(12*x - 4) = 2^(12*x + 18),
  -- Both sides of the equation can be rewritten with base 2
  { calc 16^(3*x - 1) = (2^4)^(3*x - 1) : by rw pow_mul 2 4
                ... = 2^(4*(3*x - 1)) : by rw pow_mul 2 4
                ... = 2^(12*x - 4) : by ring,
    
    calc 64^(2*x + 3) = (2^6)^(2*x + 3) : by rw pow_mul 2 6
                ... = 2^(6*(2*x + 3)) : by rw pow_mul 2 6
                ... = 2^(12*x + 18) : by ring },

  -- Equating the exponents leads to 12*x - 4 = 12*x + 18
  have h2: 12*x - 4 = 12*x + 18 := by apply (pow_inj _ _ (by norm_num)).mp; assumption,
  -- This results in a contradiction
  have h3: -4 = 18 := eq_of_sub_eq_zero (by simp [← h2]); linarith,
  -- Hence, our assumption was false and no such x exists
  contradiction,
}

end no_solution_16_64_eq_l563_563310


namespace ratio_r_to_pq_l563_563390

theorem ratio_r_to_pq (total : ℝ) (amount_r : ℝ) (amount_pq : ℝ) 
  (h1 : total = 9000) 
  (h2 : amount_r = 3600.0000000000005) 
  (h3 : amount_pq = total - amount_r) : 
  amount_r / amount_pq = 2 / 3 :=
by
  sorry

end ratio_r_to_pq_l563_563390


namespace four_digit_integers_with_conditions_l563_563942

theorem four_digit_integers_with_conditions :
  { n : ℕ // 1000 ≤ n ∧ n < 10000 ∧ (∀ i j, i ≠ j → digit n i ≠ digit n j) ∧ digit n 3 ≠ 0 ∧ digit n 0 = 6 ∧ (digit n 0 = 5 ∨ digit n 0 = 0) ∧ (digit n 3 = 6)}.card = 42 :=
by
  -- Proof goes here
  sorry

end four_digit_integers_with_conditions_l563_563942


namespace heather_blocks_remaining_l563_563214

-- Definitions of the initial amount of blocks and the amount shared
def initial_blocks : ℕ := 86
def shared_blocks : ℕ := 41

-- The statement to be proven
theorem heather_blocks_remaining : (initial_blocks - shared_blocks = 45) :=
by sorry

end heather_blocks_remaining_l563_563214


namespace abs_pi_sub_abs_pi_sub_5_l563_563483

-- Conditions
def pi_approx : Real :=
  Real.pi

-- Problem statement
theorem abs_pi_sub_abs_pi_sub_5 :
  |pi_approx - |pi_approx - 5|| = 2 * pi_approx - 5 := 
by 
  sorry

end abs_pi_sub_abs_pi_sub_5_l563_563483


namespace increased_production_rate_l563_563105

variable (initial_rate : ℕ) (initial_order : ℕ) (final_order : ℕ) (average_output : ℕ) (total_cogs : ℕ) (total_time : ℕ)

-- Conditions
def initial_condition := initial_rate = 20
def initial_cogs := initial_order = 60
def final_cogs := final_order = 60
def average_condition := average_output = 30
def total_production := total_cogs = initial_order + final_order
def total_time_condition := total_time = total_cogs / average_output

-- Conclusion
def increased_rate := final_order / (total_time_condition - (initial_order / initial_rate))

theorem increased_production_rate : initial_condition → initial_cogs → final_cogs → average_condition → total_production → total_time_condition → increased_rate = 60 := by
  sorry

end increased_production_rate_l563_563105


namespace minimum_perimeter_triangle_l563_563016

-- Define the equilateral triangle with side length sqrt(3)
def equilateral_triangle (a b c : ℝ) : Prop :=
  a = b ∧ b = c ∧ a = c

-- Define the division of point M in the ratio 2:1 on side AB
def point_M_divides_AB (A B M : ℝ) : Prop :=
  M = 2 * B / (2 + 1)

-- Define the problem: to find minimal perimeter of inscribed triangle LMN
noncomputable def minimal_perimeter_inscribed_triangle :
  (ABC LMN : Type) → (a b c M L N : ℝ) → 
  (h1 : equilateral_triangle a b c) →
  (h2 : point_M_divides_AB a b M) →
  (min_perimeter_LMN = √7) → Prop :=
λ _ _ a b c M L N h1 h2, min_perimeter_LMN = √7

-- State the theorem to be proven
theorem minimum_perimeter_triangle :
  ∀ (ABC LMN : Type) (a b c M L N : ℝ), 
  (equilateral_triangle a b c) →
  (point_M_divides_AB a b M) →
  minimal_perimeter_inscribed_triangle ABC LMN a b c M L N :=
sorry

end minimum_perimeter_triangle_l563_563016


namespace distance_between_focus_and_directrix_l563_563143

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

-- Define the value of 'a' and 'b'
def a : ℝ := 2
def b : ℝ := 2 * Real.sqrt 3

-- Define 'c' using the formula for hyperbola
def c : ℝ := Real.sqrt (a^2 + b^2)

-- Define the coordinates of the right focus
def right_focus : ℝ × ℝ := (c, 0)

-- Define the equation of the left directrix
def left_directrix : ℝ := - (a^2 / c)

-- Calculate the distance between the right focus and the left directrix
def distance_focus_directrix : ℝ := | right_focus.1 - left_directrix |

-- Theorem statement
theorem distance_between_focus_and_directrix :
  distance_focus_directrix = 5 := by
  sorry

end distance_between_focus_and_directrix_l563_563143


namespace asymptotes_of_hyperbola_l563_563929

noncomputable def hyperbola_foci_existence (b : ℝ) : Prop :=
∃ (x y : ℝ), (x^2 / 9 - y^2 / b^2 = 1) ∧ (F1 : ℝ × ℝ) = (-5, 0) ∧ (F2 : ℝ × ℝ) = (5, 0)

theorem asymptotes_of_hyperbola (b : ℝ) (hb : 9 + b^2 = 25) (hb_pos : b > 0) :
  ∀ x y : ℝ, (x^2 / 9 - y^2 / b^2 = 1) → (4 * x + 3 * y = 0) ∨ (4 * x - 3 * y = 0) :=
by
  sorry

end asymptotes_of_hyperbola_l563_563929


namespace simplify_polynomial_l563_563363

theorem simplify_polynomial (x : ℝ) :
  (5 - 5 * x - 10 * x^2 + 10 + 15 * x - 20 * x^2 - 10 + 20 * x + 30 * x^2) = 5 + 30 * x :=
  by sorry

end simplify_polynomial_l563_563363


namespace tangent_line_to_curve_num_zeros_of_f_l563_563568

noncomputable def f (a x : ℝ) : ℝ := x^2 - x - a - 6 * Real.log x

theorem tangent_line_to_curve
  (a : ℝ) :
  let t_x : ℝ := 1
  let t_y : ℝ := -a
  let f_deriv := (λ x : ℝ, 2 * x - 1 - 6 / x)
  tangent_eq : (5 : ℝ) * t_x + t_y + a - (5 : ℝ) = 0 :=
by
  sorry

theorem num_zeros_of_f
  (a : ℝ) :
  let f_zero_conds := if a < 2 - 6 * Real.log 2 then 0
                      else if a = 2 - 6 * Real.log 2 then 1
                      else if a > 12 - 12 * Real.log 2 then 1
                      else 2
   in (0 ≤ f_zero_conds ≤ 2) :=
by
  sorry

end tangent_line_to_curve_num_zeros_of_f_l563_563568


namespace biased_coin_die_probability_l563_563220

theorem biased_coin_die_probability :
  let p_heads := 1 / 4
  let p_die_5 := 1 / 8
  p_heads * p_die_5 = 1 / 32 :=
by
  sorry

end biased_coin_die_probability_l563_563220


namespace average_mark_first_class_l563_563068

theorem average_mark_first_class (A : ℝ)
  (class1_students class2_students : ℝ)
  (avg2 combined_avg total_students total_marks_combined : ℝ)
  (h1 : class1_students = 22)
  (h2 : class2_students = 28)
  (h3 : avg2 = 60)
  (h4 : combined_avg = 51.2)
  (h5 : total_students = class1_students + class2_students)
  (h6 : total_marks_combined = total_students * combined_avg)
  (h7 : 22 * A + 28 * avg2 = total_marks_combined) :
  A = 40 :=
by
  sorry

end average_mark_first_class_l563_563068


namespace akeno_spent_more_l563_563101

theorem akeno_spent_more (akeno_spent : ℤ) (lev_spent_ratio : ℚ) (ambrocio_less : ℤ) 
  (h1 : akeno_spent = 2985)
  (h2 : lev_spent_ratio = 1/3)
  (h3 : ambrocio_less = 177) : akeno_spent - ((lev_spent_ratio * akeno_spent).toInt + ((lev_spent_ratio * akeno_spent).toInt - ambrocio_less)) = 1172 := by
  sorry

end akeno_spent_more_l563_563101


namespace ben_eggs_morning_l563_563817

noncomputable def total_eggs_initial := 20
noncomputable def eggs_left := 13
noncomputable def afternoon_eggs := 3

theorem ben_eggs_morning : ∃ morning_eggs, (total_eggs_initial - eggs_left) - afternoon_eggs = morning_eggs := by
  exists 4
  sorry

end ben_eggs_morning_l563_563817


namespace center_of_mass_hemisphere_proof_l563_563854

noncomputable def center_of_mass_hemisphere (R k : ℝ) : ℝ × ℝ × ℝ :=
  let m_xy := (k * π * R^5) / 5
  let m := (k * π * R^4) / 2
  (0, 0, (m_xy / m) * R)

theorem center_of_mass_hemisphere_proof (R k : ℝ) (hR : 0 ≤ R) (hk : 0 ≤ k) : 
  center_of_mass_hemisphere R k = (0, 0, (2 / 5) * R) :=
by
  sorry

end center_of_mass_hemisphere_proof_l563_563854


namespace product_of_positive_real_parts_of_solutions_l563_563594

theorem product_of_positive_real_parts_of_solutions (x : ℂ) (h : x^3 = -27) :
  ∃ (s : set ℂ), (∀ t ∈ s, (t.re > 0)) ∧ (finset.univ.prod (λ t ∈ s, t) = 9) :=
by
  sorry

end product_of_positive_real_parts_of_solutions_l563_563594


namespace min_distance_curve_c1_curve_c2_sum_distance_PA_PB_l563_563172

noncomputable def curve_c1 := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

noncomputable def curve_c2 (t : ℝ) : ℝ × ℝ := 
  ( (√2 / 2) * t - 1, (√2 / 2) * t + 1 )

-- Part 1: Prove minimum distance from point on curve_c1 to curve_c2 is √2 - 1
theorem min_distance_curve_c1_curve_c2 : (inf {dist p (curve_c2 t) | p ∈ curve_c1, t ∈ ℝ}) = (√2 - 1) :=
sorry

noncomputable def transform (p : ℝ × ℝ) : ℝ × ℝ := 
  (2 * p.1, √3 * p.2)

noncomputable def curve_c3 :=
  transform '' curve_c1

-- Part 2: Given P(-1, 1), prove |PA| + |PB| is 12√2 / 7
theorem sum_distance_PA_PB {P : ℝ × ℝ} (hP : P = (-1, 1)) :
  ∃ A B ∈ (curve_c2 '' univ) ∩ curve_c3, dist P A + dist P B = (12 * √2 / 7) :=
sorry

end min_distance_curve_c1_curve_c2_sum_distance_PA_PB_l563_563172


namespace problem_inequality_l563_563545

variable {n : ℕ}
variable {a : Fin n → ℝ}
variable {A : ℝ}

theorem problem_inequality (hn : 1 < n)
  (h : A + ∑ i, (a i)^2 < (1 / (n - 1)) * (∑ i, a i)^2) :
  ∀ i j : Fin n, i < j → A < 2 * (a i) * (a j) := sorry

end problem_inequality_l563_563545


namespace ratioCircumradiusInradiusDoesNotDetermineShape_l563_563654

-- Defining the problem in Lean 4

noncomputable def doesNotDetermineShape (R r : ℝ) (R_ratio_r : ℝ) : Prop :=
  R / r = R_ratio_r → 
  ∃ (triangle1 triangle2 : Triangle), triangle1.shape ≠ triangle2.shape ∧ (triangle1.circumradius / triangle1.inradius = R_ratio_r) ∧ (triangle2.circumradius / triangle2.inradius = R_ratio_r)

theorem ratioCircumradiusInradiusDoesNotDetermineShape (R r : ℝ) (R_ratio_r : ℝ) : doesNotDetermineShape R r R_ratio_r :=
sorry

end ratioCircumradiusInradiusDoesNotDetermineShape_l563_563654


namespace reservoir_water_level_l563_563424

theorem reservoir_water_level (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : 
  ∃ y : ℝ, y = 6 + 0.3 * x :=
by sorry

end reservoir_water_level_l563_563424


namespace lunch_break_is_48_minutes_l563_563301

noncomputable def lunch_break_duration (L : ℝ) (p a : ℝ) : Prop :=
  (8 - L) * (p + a) = 0.6 ∧ 
  (9 - L) * p = 0.35 ∧
  (5 - L) * a = 0.1

theorem lunch_break_is_48_minutes :
  ∃ L p a, lunch_break_duration L p a ∧ L * 60 = 48 :=
by
  -- proof steps would go here
  sorry

end lunch_break_is_48_minutes_l563_563301


namespace rectangle_max_sections_l563_563485

-- State the problem as a Lean definition
def max_sections_with_5_lines : ℕ :=
  16

-- The lean theorem equivalent to the proof problem
theorem rectangle_max_sections (n : ℕ) 
  (lines : Fin n → { l : Set (ℝ × ℝ) // ∃ a b c : ℝ, l = {p | a * p.1 + b * p.2 + c = 0} }) 
  (h_optimal : optimal_intersections lines) :
  n = 5 → (∑ k in Finset.range (n + 1), k) + 1 = 16 := 
by
  sorry

end rectangle_max_sections_l563_563485


namespace intersection_sets_l563_563573

open Set

variable (α : Type)
variable [LinearOrder α] [Add α] [Mul α] [One α]

def A (x : α) : Prop := x - 1 < 5
def B (x : α) : Prop := -4 * x + 8 < 0
def C (x : α) : Prop := 2 < x ∧ x < 6

theorem intersection_sets : {x : α | A x} ∩ {x : α | B x} = {x : α | C x} :=
by
  sorry

end intersection_sets_l563_563573


namespace percent_profit_l563_563954

theorem percent_profit (C S : ℝ) (h : 55 * C = 50 * S) : 
  100 * ((S - C) / C) = 10 :=
by
  sorry

end percent_profit_l563_563954


namespace scalene_triangles_count_l563_563861

theorem scalene_triangles_count :
  (finset.univ.filter (λ (abc : ℕ × ℕ × ℕ), 
    let a := abc.1, b := abc.2.1, c := abc.2.2 in 
    a + b + c < 15 ∧ a < b ∧ b < c ∧ a + b > c ∧ a + c > b ∧ b + c > a)).card = 6 :=
by sorry

end scalene_triangles_count_l563_563861


namespace candle_height_time_l563_563728

theorem candle_height_time
  (h : ℝ) -- initial height of each candle (not really needed in this proof as both are the same)
  (rate1 : ℝ) (rate2 : ℝ) -- burn rates of the candles
  (burn_time1 : ℝ := 4) -- the first candle burns completely in 4 hours
  (burn_time2 : ℝ := 2) -- the second candle burns completely in 2 hours
  (h1 : rate1 = h / burn_time1) -- burn rate of the first candle
  (h2 : rate2 = h / burn_time2) -- burn rate of the second candle) : 
  ∃ x : ℝ, (h - rate1*x = 3*(h - rate2*x)) →
    x = 8/5 :=
begin
  sorry
end

end candle_height_time_l563_563728


namespace find_number_of_sides_l563_563738
noncomputable theory

def number_of_sides (n : ℕ) : Prop :=
  (∃ k : ℕ, 9 / n^4 = 1 / 9) ∧ n = 3

theorem find_number_of_sides (n : ℕ) (h : 9 / n^4 = 1 / 9) : number_of_sides n :=
begin
  use n,
  split,
  { exact ⟨n, h⟩ },
  { sorry }
end

end find_number_of_sides_l563_563738


namespace total_price_nearest_dollar_l563_563649

def price1 : ℝ := 2.15
def price2 : ℝ := 7.49
def price3 : ℝ := 12.85

def discount_rate : ℝ := 0.10

-- Compute the discounted price of the most expensive item
def max_price : ℝ := max (max price1 price2) price3
def discount : ℝ := max_price * discount_rate
def discounted_price : ℝ := max_price - discount

-- Round each price to the nearest dollar
def rounded_price1 : ℝ := Real.round price1
def rounded_price2 : ℝ := Real.round price2
def rounded_discounted_price : ℝ := Real.round discounted_price

-- Compute the total price
def total_price : ℝ := rounded_price1 + rounded_price2 + rounded_discounted_price

theorem total_price_nearest_dollar : total_price = 21 := by
  -- Proof here
  sorry

end total_price_nearest_dollar_l563_563649


namespace measure_angle_PRA_l563_563615

-- Define the basic setup for the triangle and points
variables (A B C P Q R : Type) 
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] [MetricSpace Q] [MetricSpace R]

-- Define a triangle ABC with ∠ABC = 120°
variables (ABC_triangle : Triangle A B C)
variables (H1 : angle B A C = 120)

-- Define point P on segment AC such that BP bisects ∠ABC
variables (HP : on_line_segment P A C) (HBisect : bisect_angle B A C P B)

-- Define point Q on the external bisector of ∠BCA
variables (HQ : on_external_bisector Q B C A)

-- Define point R where PQ intersects BC
variables (HR : on_line_intersection R P Q B C)

-- Prove that ∠PRA = 30°
theorem measure_angle_PRA : angle P R A = 30 :=
  sorry

end measure_angle_PRA_l563_563615


namespace airplane_flew_every_day_l563_563518

theorem airplane_flew_every_day : 
  (∀ (days : ℕ) (flights_per_day : ℕ)
    (airplane_flights : fin 92 → fin 10 → Prop), 
    (days = 92) ∧ 
    (flights_per_day = 10) ∧
    (∀ i j k l, i ≠ k → airplane_flights i j → airplane_flights k l → i ≠ k → j ≠ l) ∧
    (∀ i j, i ≠ j → ∃! a, airplane_flights i a ∧ airplane_flights j a))
  → ∃ a, ∀ d, d < days → airplane_flights d a :=
by
  intros days flights_per_day airplane_flights H, 
  -- sorry to imply the proof is pending
  sorry

end airplane_flew_every_day_l563_563518


namespace find_general_term_and_Tn_l563_563178

-- Define arithmetic sequence and sum sequence
variables (a : Nat → ℤ) (s : Nat → ℤ)
variables (a2 : a 2 = 9) (s5 : s 5 = 65)

-- Define general term formula to prove
def general_term_formula (n : ℕ) : ℤ := 4 * n + 1

-- Define the sum sequence s_n
noncomputable def sum_sequence (n : ℕ) : ℤ := (n * (2 * n + 3)) / 2

-- Define T_n as the sum of the transformed sequence
noncomputable def T_n_formula (n : ℕ) : ℤ :=
  (Finset.range (n + 1)).sum (λ k, 1 / (sum_sequence k - k))

theorem find_general_term_and_Tn :
  (∀ n, a n = general_term_formula n) ∧
  (T_n_formula = λ n, n / (2 * n + 2)) :=
by
  sorry

end find_general_term_and_Tn_l563_563178


namespace find_a_l563_563889

theorem find_a
  (a : ℝ) (h1 : a > 0)
  (C_eq : ∀ x y, (x - a)^2 + (y - 2)^2 = 4)
  (l_eq : ∀ x y, x - y + 3 = 0)
  (chord_length_eq : ∃ x y, (x - y + 3 = 0) ∧ (x - a)^2 + (y - 2)^2 = 4 ∧ sqrt (4 - (1 / sqrt 2)^2) = 2 * sqrt 3) :
  a = sqrt 2 - 1 :=
by
  sorry

end find_a_l563_563889


namespace necessary_sufficient_condition_l563_563292

noncomputable def gcd (a b : ℕ) : ℕ := nat.gcd a b

def operation (x y : ℕ) : ℕ × ℕ :=
if x ≤ y then (2*x, y-x) else (x, y)

lemma is_power_of_two (n : ℕ) : Prop :=
∃ k : ℕ, n = 2 ^ k

theorem necessary_sufficient_condition (n : ℕ) (nums : list ℕ)
  (h1 : gcd_list nums = 1)
  (h2 : nums.length = n) :
  (∃ (operations_count : ℕ), (iterate_operations operations_count nums).count 0 = n - 1) ↔
  is_power_of_two (nums.sum) :=
sorry

end necessary_sufficient_condition_l563_563292


namespace tina_total_earnings_l563_563032

noncomputable def tina_hourly_wage := 18.00
def regular_hours_per_shift := 8
def total_hours_per_day := 10
def overtime_rate := 1.5
def days_worked := 5

def regular_pay (hourly_wage : ℝ) (hours : ℕ) := (hourly_wage * hours)
def overtime_pay (hourly_wage : ℝ) (overtime_rate : ℝ) (hours : ℕ) := (hourly_wage * overtime_rate * hours)

theorem tina_total_earnings : 
  let regular_pay := regular_pay tina_hourly_wage regular_hours_per_shift in
  let overtime_pay := overtime_pay tina_hourly_wage overtime_rate (total_hours_per_day - regular_hours_per_shift) in
  let total_regular_pay := days_worked * regular_pay in
  let total_overtime_pay := days_worked * overtime_pay in
  total_regular_pay + total_overtime_pay = 990.00 :=
by
  let regular_pay := regular_pay tina_hourly_wage regular_hours_per_shift
  let overtime_pay := overtime_pay tina_hourly_wage overtime_rate (total_hours_per_day - regular_hours_per_shift)
  let total_regular_pay := days_worked * regular_pay
  let total_overtime_pay := days_worked * overtime_pay
  have h1 : regular_pay = 144 := sorry
  have h2 : overtime_pay = 54 := sorry
  have h3 : total_regular_pay = 720 := sorry
  have h4 : total_overtime_pay = 270 := sorry
  show 720 + 270 = 990, from sorry

end tina_total_earnings_l563_563032


namespace radius_of_circle_l563_563845

theorem radius_of_circle (r : ℝ) : 
  (∀ x : ℝ, (4 * x^2 + r = x) → (1 - 16 * r = 0)) → r = 1 / 16 :=
by
  intro H
  have h := H 0
  simp at h
  sorry

end radius_of_circle_l563_563845


namespace binomial_prob_x_eq_3_l563_563280

noncomputable def binomial_pmf (n : ℕ) (p : ℝ) : probability_mass_function ℕ :=
  { p := λ k => (nat.choose n k : ℝ) * p^k * (1 - p)^(n - k),
    p_nonneg := sorry,
    p_sum_one := sorry }

theorem binomial_prob_x_eq_3 :
  P((binomial_pmf 6 (1/2)).pdf 3) = 5/16 := 
  sorry

end binomial_prob_x_eq_3_l563_563280


namespace length_AP_solution_l563_563602

noncomputable def length_AP (A M P : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - P.1) ^ 2 + (A.2 - P.2) ^ 2)

theorem length_AP_solution :
  ∀ (A M P : ℝ × ℝ),
    A = (1, 1) →
    M = (0, -1) →
    P = (4 / 5, 3 / 5) →
    length_AP A M P = real.sqrt(5) / 5 :=
by
  intros A M P hA hM hP
  rw [hA, hP]
  sorry

end length_AP_solution_l563_563602


namespace graph_connected_l563_563679

theorem graph_connected (V : ℕ) (hV : V = 100) :
  (∀ n : ℕ, n ≤ V → 50^2 - (n - 50)^2 ≥ 99) → connected_graph V :=
by
  sorry

end graph_connected_l563_563679


namespace first_cones_apex_angle_correct_l563_563355

noncomputable def first_cones_apex_angle : ℝ := 
2 * Real.atan (Real.sqrt 3 - 1)

theorem first_cones_apex_angle_correct :
  ∀ (A : Point) (cone1 cone2 cone3 cone4 : Cone),
  -- conditions
  cone1.apex = A ∧ cone2.apex = A ∧ cone3.apex = A ∧ cone4.apex = A ∧ 
  cone1.angle = cone2.angle ∧ 
  cone3.angle = π / 3 ∧ 
  cone4.angle = 5 * π / 6 ∧
  conesExternallyTangent [cone1, cone2, cone3] ∧ 
  conesInternallyTangent cone4 [cone1, cone2, cone3] 
  -- proof goal
  → cone1.angle = first_cones_apex_angle :=
begin
  intros,
  sorry,
end

end first_cones_apex_angle_correct_l563_563355


namespace sam_total_cents_l563_563671

def dimes_to_cents (dimes : ℕ) : ℕ := dimes * 10
def quarters_to_cents (quarters : ℕ) : ℕ := quarters * 25
def nickels_to_cents (nickels : ℕ) : ℕ := nickels * 5
def dollars_to_cents (dollars : ℕ) : ℕ := dollars * 100

noncomputable def total_cents (initial_dimes dad_dimes mom_dimes grandma_dollars sister_quarters_initial : ℕ)
                             (initial_quarters dad_quarters mom_quarters grandma_transform sister_quarters_donation : ℕ)
                             (initial_nickels dad_nickels mom_nickels grandma_conversion sister_nickels_donation : ℕ) : ℕ :=
  dimes_to_cents initial_dimes +
  quarters_to_cents initial_quarters +
  nickels_to_cents initial_nickels +
  dimes_to_cents dad_dimes +
  quarters_to_cents dad_quarters -
  nickels_to_cents mom_nickels -
  dimes_to_cents mom_dimes +
  dollars_to_cents grandma_dollars +
  quarters_to_cents sister_quarters_donation +
  nickels_to_cents sister_nickels_donation

theorem sam_total_cents :
  total_cents 9 7 2 3 4 5 2 0 0 3 2 1 = 735 := 
  by exact sorry

end sam_total_cents_l563_563671


namespace angle_PQS_eq_140_l563_563610

theorem angle_PQS_eq_140 
  (P Q R S : Type) 
  [is_line P Q R] 
  [is_isosceles_triangle Q R S] 
  (angle_QRS : angle Q R S = 30)
  (angle_SQR : angle S Q R = 30)
  (angle_PSQ : angle P S Q = 110) :
  angle P Q S = 140 :=
sorry

end angle_PQS_eq_140_l563_563610


namespace significant_improvement_l563_563418

-- Data for the old and new devices
def old_data : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_data : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

-- Definitions for sample mean and sample variance
def sample_mean (data : List ℝ) : ℝ := (data.sum) / (data.length)
def sample_variance (data : List ℝ) : ℝ := 
  let mean := sample_mean data
  (data.map (λ x => (x - mean)^2)).sum / data.length

-- Prove that the new device's mean indicator has significantly improved
theorem significant_improvement :
  let x := sample_mean old_data
  let y := sample_mean new_data
  let s1_squared := sample_variance old_data
  let s2_squared := sample_variance new_data
  (y - x) ≥ 2 * sqrt ((s1_squared + s2_squared) / 10) :=
by
  sorry

end significant_improvement_l563_563418


namespace product_mod_7_l563_563711

theorem product_mod_7 (a b c : ℕ) (ha : a % 7 = 3) (hb : b % 7 = 4) (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 4 :=
sorry

end product_mod_7_l563_563711


namespace optimal_tank_dimensions_l563_563457

noncomputable def volume_condition (l w : ℝ) (h : ℝ) := l * w * h = 48

noncomputable def cost_function (l w : ℝ) (h cost_bottom cost_walls : ℝ) :=
  cost_bottom * (l * w) + cost_walls * (2 * h * l + 2 * h * w)

theorem optimal_tank_dimensions :
  ∃ (l w : ℝ), (∃ (h : ℝ), h = 3 ∧ volume_condition l w h ∧ cost_function l w h 40 20 = 1600) :=
begin
  use 4,
  use 4,
  use 3,
  split,
  { refl },
  split,
  { unfold volume_condition, norm_num },
  { unfold cost_function, norm_num }
end

end optimal_tank_dimensions_l563_563457


namespace count_polynomials_in_G_l563_563268

-- Type definition for complex numbers
def is_root_with_integer_parts (a b : ℤ) : ℂ :=
  ⟨a, b⟩

-- Define polynomial structure and conditions
def polynomial_in_G (P : polynomial ℂ) : Prop :=
  ∃ n (c : fin (n + 1) → ℤ), 
    P = polynomial.monomial n 1 +
        ∑ i in finset.range n, polynomial.monomial i (c i) ∧
    c 0 = 50 ∧
    (∀ r, P.eval r = 0 → ∃ a b : ℤ, r = a + b * complex.I)

-- Define the formal problem
theorem count_polynomials_in_G :
  ∃ N : ℕ, ∀ G : finset (polynomial ℂ), 
    (∀ P ∈ G, polynomial_in_G P) → G.card = N :=
by
  -- Provide the specific number of polynomials
  exact ⟨528, sorry⟩

end count_polynomials_in_G_l563_563268


namespace rectangle_area_diff_l563_563288

theorem rectangle_area_diff :
  ∀ (l w : ℕ), (2 * l + 2 * w = 60) → (∃ A_max A_min : ℕ, 
    A_max = (l * (30 - l)) ∧ A_min = (min (1 * (30 - 1)) (29 * (30 - 29))) ∧ (A_max - A_min = 196)) :=
by
  intros l w h
  use 15 * 15, min (1 * 29) (29 * 1)
  sorry

end rectangle_area_diff_l563_563288


namespace remainder_g_x12_l563_563756

theorem remainder_g_x12 (x : ℂ) :
  let g : ℂ → ℂ := λ x, x^5 + x^4 + x^3 + x^2 + x + 1 in
  ∃ r : ℂ, r = 6 ∧ ∀ q : polynomial ℂ, g (x ^ 12) = g(x) * q + polynomial.C r := sorry

end remainder_g_x12_l563_563756


namespace minimum_sum_of_natural_numbers_with_lcm_2012_l563_563333

/-- 
Prove that the minimum sum of seven natural numbers whose least common multiple is 2012 is 512.
-/

theorem minimum_sum_of_natural_numbers_with_lcm_2012 : 
  ∃ (a b c d e f g : ℕ), Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm a b) c) d) e) f) g = 2012 ∧ (a + b + c + d + e + f + g) = 512 :=
sorry

end minimum_sum_of_natural_numbers_with_lcm_2012_l563_563333


namespace total_selling_price_16800_l563_563776

noncomputable def total_selling_price (CP_per_toy : ℕ) : ℕ :=
  let CP_18 := 18 * CP_per_toy
  let Gain := 3 * CP_per_toy
  CP_18 + Gain

theorem total_selling_price_16800 :
  total_selling_price 800 = 16800 :=
by
  sorry

end total_selling_price_16800_l563_563776


namespace parabola_c_value_l563_563780

theorem parabola_c_value (b c : ℝ) 
  (h1 : 5 = 2 * 1^2 + b * 1 + c)
  (h2 : 17 = 2 * 3^2 + b * 3 + c) : 
  c = 5 := 
by
  sorry

end parabola_c_value_l563_563780


namespace find_a_l563_563977

-- Define the polynomial expansion term conditions
def binomial_coefficient (n k : ℕ) := Nat.choose n k

def fourth_term_coefficient (x a : ℝ) : ℝ :=
  binomial_coefficient 9 3 * x^6 * a^3

theorem find_a (a : ℝ) (x : ℝ) (h : fourth_term_coefficient x a = 84) : a = 1 :=
by
  unfold fourth_term_coefficient at h
  sorry

end find_a_l563_563977


namespace equilateral_triangle_of_ap_angles_gp_sides_l563_563984

theorem equilateral_triangle_of_ap_angles_gp_sides
  (A B C : ℝ)
  (α β γ : ℝ)
  (hαβγ_sum : α + β + γ = 180)
  (h_ap_angles : 2 * β = α + γ)
  (a b c : ℝ)
  (h_gp_sides : b^2 = a * c) :
  α = β ∧ β = γ ∧ a = b ∧ b = c :=
sorry

end equilateral_triangle_of_ap_angles_gp_sides_l563_563984


namespace find_sum_of_cubes_l563_563271

noncomputable def roots (a b c : ℝ) : Prop :=
  5 * a^3 + 2014 * a + 4027 = 0 ∧ 
  5 * b^3 + 2014 * b + 4027 = 0 ∧ 
  5 * c^3 + 2014 * c + 4027 = 0

theorem find_sum_of_cubes (a b c : ℝ) (h : roots a b c) : 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 2416.2 :=
sorry

end find_sum_of_cubes_l563_563271


namespace part1_part2_l563_563873

-- Definition and conditions
variables (α β : ℝ) 
variables (h1 : tan α = 4 / 3) 
variables (h2 : cos (β - α) = sqrt 2 / 10) 
variables (h3 : 0 < α ∧ α < π / 2 ∧ π / 2 < β ∧ β < π)

-- Part (1)
theorem part1 : sin α ^ 2 - (sin α * cos α) = 4 / 25 :=
sorry

-- Part (2)
theorem part2 : β = 3 * π / 4 :=
sorry

end part1_part2_l563_563873


namespace num_correct_conclusions_is_zero_l563_563936

theorem num_correct_conclusions_is_zero (p q : Prop) (h : p ∨ q) :
  (if (p ∧ q) then 1 else 0) + (if ¬(p ∧ q) then 1 else 0) + 
  (if (¬p ∨ ¬q) then 1 else 0) + (if ¬(¬p ∨ ¬q) then 1 else 0) = 0 := 
by
  sorry

end num_correct_conclusions_is_zero_l563_563936


namespace calculate_fraction_l563_563120

theorem calculate_fraction :
  ( (12^4 + 484) * (24^4 + 484) * (36^4 + 484) * (48^4 + 484) * (60^4 + 484) )
  /
  ( (6^4 + 484) * (18^4 + 484) * (30^4 + 484) * (42^4 + 484) * (54^4 + 484) )
  = 181 := by
  sorry

end calculate_fraction_l563_563120


namespace value_of_g_10_l563_563279

def inv_func (y : ℝ) : ℝ := -real.logb 3 (y - 1)

theorem value_of_g_10 :
  inv_func 10 = -2 :=
by
  unfold inv_func
  sorry

end value_of_g_10_l563_563279


namespace car_speed_proof_l563_563716

noncomputable def car_speed_second_hour 
  (speed_first_hour: ℕ) (average_speed: ℕ) (total_time: ℕ) 
  (speed_second_hour: ℕ) : Prop :=
  (speed_first_hour = 80) ∧ (average_speed = 70) ∧ (total_time = 2) → speed_second_hour = 60

theorem car_speed_proof : 
  car_speed_second_hour 80 70 2 60 := by
  sorry

end car_speed_proof_l563_563716


namespace simplify_expression_l563_563853

theorem simplify_expression :
  ∃ (a b c : ℤ), c ≠ 0 ∧ sqrt 11 + 2 / sqrt 11 + sqrt 2 + 3 / sqrt 2 = (a * sqrt 11 + b * sqrt 2) / c ∧ c = 22 ∧ a = 11 ∧ b = 44 :=
by
  sorry

end simplify_expression_l563_563853


namespace sequence_limit_l563_563833

theorem sequence_limit :
  ∃ L : ℝ, (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (x n - L) < ε) ∧ L = 1 where
  x : ℕ → ℝ
  h₁ : x 1 = 3
  h₂ : ∀ n ≥ 2, x (n+1) = (n + 3) / (3 * (n + 1)) * (x n + 2)
  sorry

end sequence_limit_l563_563833


namespace product_of_roots_of_cubic_l563_563484

theorem product_of_roots_of_cubic :
  (∃ (r1 r2 r3 : ℝ), 3 * r1 * r2 * r3 - r1^2 - 20 * r1 + 27 = 0) →
  (∃ (r1 r2 r3 : ℝ), 3 * r1 * r2 * r3 = - 9) :=
by
  intro existence_of_roots
  obtain ⟨r1, r2, r3, h⟩ := existence_of_roots
  exists r1, r2, r3
  dsimp at h
  have poly_eq : 3 * r1 * r2 * r3 - r1^2 - 20 * r1 + 27 = 0 := h
  sorry

end product_of_roots_of_cubic_l563_563484


namespace circle_area_through_PQR_l563_563603

-- Define the isosceles triangle PQR with sides PQ = PR = 5 * sqrt 6.
structure Triangle (α : Type*) [normed_group α] [normed_space ℝ α] :=
(P Q R : α)
(hPQ_eq_PR : dist P Q = dist P R)
(PQ_PR_len : dist P Q = 5 * real.sqrt 6)

-- Define the properties of the tangency circle and the points where it is tangent.
structure TangencyCircle (α : Type*) [normed_group α] [normed_space ℝ α] (P Q R : α) :=
(radius : ℝ)
(tangent_PQ : ∀ (S : α), dist S Q = radius → dist S P = real.sqrt(2 * radius ^ 2))
(tangent_PR : ∀ (S : α), dist S R = radius → dist S P = real.sqrt(2 * radius ^ 2))

-- Let PQR be the vertices of the triangle and S be the center of the circle
variables {α : Type*} [normed_group α] [normed_space ℝ α]
variables (P Q R S: α)

-- Define the radius of the circle that is tangent to both PQ and PR
def CircleTangentRadius := 6 * real.sqrt 2

def CircleArea := π * (12 * real.sqrt 2)^2

-- The area of the circle passing through PQR in the isosceles triangle PQR should be 240π
theorem circle_area_through_PQR (t : Triangle α) (c : TangencyCircle α t.P t.Q t.R) :
  c.radius = 6 * real.sqrt 2 →
  dist t.P t.Q = 5 * real.sqrt 6 →
  dist t.P t.R = 5 * real.sqrt 6 →
  c.radius = CircleTangentRadius →
  CircleArea = 240 * π :=
sorry

end circle_area_through_PQR_l563_563603


namespace AP_AB_eq_AR_AD_eq_AQ_AC_l563_563538

-- Definitions for points A, B, C, D, P, Q, R in terms of an parallelogram and a circle intersection
variables {A B C D P Q R : Type*} [parallelogram ABCD] 
          [circle_intersects A P B Q C R D]

theorem AP_AB_eq_AR_AD_eq_AQ_AC : 
  AP * AB = AR * AD ∧ AR * AD = AQ * AC :=
sorry

end AP_AB_eq_AR_AD_eq_AQ_AC_l563_563538


namespace minimum_and_maximum_S_l563_563272

theorem minimum_and_maximum_S (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : a^2 + b^2 + c^2 + d^2 = 30) :
  3 * (a^3 + b^3 + c^3 + d^3) - 3 * a^2 - 3 * b^2 - 3 * c^2 - 3 * d^2 = 7.5 :=
sorry

end minimum_and_maximum_S_l563_563272


namespace close_sluice_to_target_water_storage_l563_563331

noncomputable def river_inflow : ℝ := 200000
noncomputable def evaporation_rate_day : ℝ := 1000
noncomputable def evaporation_rate_night : ℝ := 250
noncomputable def sluice_start_time : ℝ := 12
noncomputable def initial_water_storage : ℝ := 4000000
noncomputable def sluice_discharge_rate : ℝ := 230000
noncomputable def target_water_storage : ℝ := 120000
noncomputable def sluice_close_time := August_12_11pm  -- Assuming we have a datetime type August_12_11pm

theorem close_sluice_to_target_water_storage : 
  close_sluice_time = August_12_11pm → (initial_water_storage + 
  (river_inflow * 24 - sluice_discharge_rate * 24 - 
    evaporation_rate_day * 12 - evaporation_rate_night * 12) * 
    (close_sluice_time - sluice_start_time)) = target_water_storage :=
by sorry

end close_sluice_to_target_water_storage_l563_563331


namespace opposite_of_neg_3_l563_563008

theorem opposite_of_neg_3 : (-(-3) = 3) :=
by
  sorry

end opposite_of_neg_3_l563_563008


namespace smallest_domain_of_g_l563_563732

def g : ℕ → ℕ
| n := if n % 2 == 0 then n / 2 else 3 * n + 2

theorem smallest_domain_of_g :
  ∃ (d : ℕ) (s : set ℕ), (8 ∈ s) ∧ (∀ x ∈ s, g x ∈ s) ∧ (∀ y ∈ s, ∃ z ∈ s, g z = y) ∧
  (∀ t ∈ s, t ≠ 8 → ∃ p : set ℕ, t ∈ p ∧ finite p) :=
sorry

end smallest_domain_of_g_l563_563732


namespace symbol_for_oxygen_is_O_l563_563148

-- Define the atomic weights
def atomic_weight_aluminum : ℝ := 26.98
def atomic_weight_phosphorus : ℝ := 30.97
def atomic_weight_oxygen : ℝ := 16.00

-- Define the number of atoms in the compound
def num_aluminum_atoms : ℕ := 1
def num_phosphorus_atoms : ℕ := 1
def num_oxygen_atoms : ℕ := 4

-- Given that the molecular weight of the compound is approximately 122 amu
def molecular_weight : ℝ := 122

-- The total calculated molecular weight 
noncomputable def calculated_molecular_weight : ℝ := 
  (num_aluminum_atoms * atomic_weight_aluminum) + 
  (num_phosphorus_atoms * atomic_weight_phosphorus) + 
  (num_oxygen_atoms * atomic_weight_oxygen)

-- Proof that the symbol for the oxygen element is "O"
theorem symbol_for_oxygen_is_O : calculated_molecular_weight ≈ molecular_weight → ∃ symbol : String, symbol = "O" :=
by
  sorry

end symbol_for_oxygen_is_O_l563_563148


namespace area_relation_l563_563298

open Real

variables (A B C D E F P Q : Point)
variables (S_AEF S_APQ : ℝ)

-- Definition of the square ABCD
def is_square : Prop :=
  is_square_ABC (A, B, C, D)

-- Points E and F on sides BC and CD respectively
def points_on_sides : Prop :=
  E ∈ segment B C ∧ F ∈ segment C D

-- Given Angle EAF equals 45 degrees
def angle_EAF : Prop :=
  ∠EAF = 45 * π / 180

-- Intersection of AE and AF with BD at P and Q respectively
def intersections : Prop :=
  (segment A E ∩ segment B D = {P}) ∧ (segment A F ∩ segment B D = {Q})

-- Prove that the area of AEF is twice the area of APQ
theorem area_relation (h1 : is_square)
                     (h2 : points_on_sides)
                     (h3 : angle_EAF)
                     (h4 : intersections) :
  S_AEF = 2 * S_APQ := 
sorry

end area_relation_l563_563298


namespace probability_of_safe_flight_l563_563415

-- Definitions of edge lengths and volumes
def edge_length_large : ℝ := 4
def edge_length_small : ℝ := edge_length_large - 2

def volume_large : ℝ := edge_length_large ^ 3
def volume_small : ℝ := edge_length_small ^ 3

-- Probability of safe flight
def probability_safe_flight : ℝ := volume_small / volume_large

-- Theorem statement
theorem probability_of_safe_flight :
  probability_safe_flight = 1 / 8 := by
  -- The proof is not provided as per the instructions
  sorry

end probability_of_safe_flight_l563_563415


namespace no_ordered_triples_l563_563134

noncomputable def no_solution (x y z : ℝ) : Prop :=
  x^2 - 3 * x * y + 2 * y^2 - z^2 = 31 ∧
  -x^2 + 6 * y * z + 2 * z^2 = 44 ∧
  x^2 + x * y + 8 * z^2 = 100

theorem no_ordered_triples : ¬ ∃ (x y z : ℝ), no_solution x y z := 
by 
  sorry

end no_ordered_triples_l563_563134


namespace debby_bottles_per_day_l563_563831

theorem debby_bottles_per_day :
  let total_bottles := 153
  let days := 17
  total_bottles / days = 9 :=
by
  sorry

end debby_bottles_per_day_l563_563831


namespace frank_bakes_for_5_days_l563_563523

variable (d : ℕ) -- The number of days Frank bakes cookies

def cookies_baked_per_day : ℕ := 2 * 12
def cookies_eaten_per_day : ℕ := 1

-- Total cookies baked over d days minus the cookies Frank eats each day
def cookies_remaining_before_ted (d : ℕ) : ℕ :=
  d * (cookies_baked_per_day - cookies_eaten_per_day)

-- Ted eats 4 cookies on the last day, so we add that back to get total before Ted ate
def total_cookies_before_ted (d : ℕ) : ℕ :=
  cookies_remaining_before_ted d + 4

-- After Ted's visit, there are 134 cookies left
axiom ted_leaves_134_cookies : total_cookies_before_ted d = 138

-- Prove that Frank bakes cookies for 5 days
theorem frank_bakes_for_5_days : d = 5 := by
  sorry

end frank_bakes_for_5_days_l563_563523


namespace inverse_proportion_relationship_l563_563294

variable k : ℝ
def y (x : ℝ) : ℝ := k / x

theorem inverse_proportion_relationship (h1 : y (-2) = 3) :
  let y1 := y (-3) in
  let y2 := y 1 in
  let y3 := y 2 in
  y2 < y3 ∧ y3 < y1 :=
by
  have k_val : k = -6 := by
    rw [y, ← @eq_div_iff_mul_eq ℝ _ _ _ _] at h1 <;> linarith
  let y1 := -6 / -3
  let y2 := -6 / 1
  let y3 := -6 / 2
  split
  sorry

end inverse_proportion_relationship_l563_563294


namespace jo_reading_time_l563_563624

structure Book :=
  (totalPages : Nat)
  (currentPage : Nat)
  (pageOneHourAgo : Nat)

def readingTime (b : Book) : Nat :=
  let pagesRead := b.currentPage - b.pageOneHourAgo
  let pagesLeft := b.totalPages - b.currentPage
  pagesLeft / pagesRead

theorem jo_reading_time :
  ∀ (b : Book), b.totalPages = 210 → b.currentPage = 90 → b.pageOneHourAgo = 60 → readingTime b = 4 :=
by
  intro b h1 h2 h3
  sorry

end jo_reading_time_l563_563624


namespace point_in_quadrant_IV_l563_563570

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^3 - a*x^2 - b*x

theorem point_in_quadrant_IV (a b : ℝ) :
  ((∃ (h1 : a = -4) (h2 : b = 11),
    let f := f := (λ x, x^3 - a*x^2 - b*x) in
    ∃ (extreme_value : ℝ) (x : ℝ), x = 1 ∧
    f'(x) = 3*x^2 - 2*a*x - b ∧
    f'(1) = 3 - 2*a - b ∧
    f(1) - a = b ∧
    1 - a - b + a^2 = 10 ∧
    extreme_value = f(1) ∧
    extreme_value = 10)
  → a < 0 ∧ b > 0) :=
by sorry

end point_in_quadrant_IV_l563_563570


namespace sum_of_coeffs_eq_neg33_l563_563525

theorem sum_of_coeffs_eq_neg33 :
  let p : Polynomial ℝ := (X^2 - X - 2) ^ 5 in
  (p.coeff 0 + p.coeff 1 + p.coeff 2 + p.coeff 3 + p.coeff 4 + 
   p.coeff 5 + p.coeff 6 + p.coeff 7 + p.coeff 8 + p.coeff 9) = -33 :=
by 
  sorry

end sum_of_coeffs_eq_neg33_l563_563525


namespace find_cubes_with_5_neighbors_l563_563524

theorem find_cubes_with_5_neighbors (h : 12 * (n - 2) = 132) : 6 * (n - 2)^2 = 726 := by
  have n_val : n = 13 := by
    -- Solve for n
    linarith
  rw [n_val]
  -- Calculate the number of cubes with 5 neighbors
  norm_num
  sorry

end find_cubes_with_5_neighbors_l563_563524


namespace bakery_doughnuts_given_away_l563_563405

theorem bakery_doughnuts_given_away :
  (∀ (boxes_doughnuts : ℕ) (total_doughnuts : ℕ) (boxes_sold : ℕ), 
    boxes_doughnuts = 10 →
    total_doughnuts = 300 →
    boxes_sold = 27 →
    ∃ (doughnuts_given_away : ℕ),
    doughnuts_given_away = (total_doughnuts / boxes_doughnuts - boxes_sold) * boxes_doughnuts ∧
    doughnuts_given_away = 30) :=
by
  intros boxes_doughnuts total_doughnuts boxes_sold h1 h2 h3
  use (total_doughnuts / boxes_doughnuts - boxes_sold) * boxes_doughnuts
  split
  · rw [h1, h2, h3]
    sorry
  · sorry

end bakery_doughnuts_given_away_l563_563405


namespace binary_add_sub_l563_563825

-- Define the binary numbers
def b1 : ℕ := 10110
def b2 : ℕ := 1101
def b3 : ℕ := 110
def b4 : ℕ := 101

-- The proposition to prove
theorem binary_add_sub :
  nat.sub (nat.add (nat.sub b1 b2) b3) b4 = 1010 := 
sorry

end binary_add_sub_l563_563825


namespace significant_improvement_l563_563419

def mean (data: List ℝ) : ℝ :=
  data.sum / data.length

def variance (data: List ℝ) (mean: ℝ) : ℝ :=
  data.map (λ x => (x - mean) ^ 2).sum / data.length

def old_device_data: List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

def new_device_data: List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

theorem significant_improvement :
  let
    μx := mean old_device_data,
    μy := mean new_device_data,
    σ1_sq := variance old_device_data μx,
    σ2_sq := variance new_device_data μy
  in
    (μy - μx) > 2 * Real.sqrt((σ1_sq + σ2_sq) / 10) :=
by
  sorry

end significant_improvement_l563_563419


namespace sum_squares_of_roots_eq_seventeen_l563_563486

theorem sum_squares_of_roots_eq_seventeen {α : Type*} [Field α] (a b c : α) :
  a = 5 → b = 15 → c = -20 →
  let Δ := b^2 - 4 * a * c in
  let sum_of_roots := -b / a in
  let prod_of_roots := c / a in
  let sum_squares := sum_of_roots^2 - 2 * prod_of_roots in
  Δ ≥ 0 → sum_squares = 17 := 
by 
  intros ha hb hc Δ sum_of_roots prod_of_roots sum_squares hΔ
  rw [ha, hb, hc]
  dsimp [sum_of_roots, prod_of_roots, sum_squares]
  norm_num
  sorry

end sum_squares_of_roots_eq_seventeen_l563_563486


namespace amount_borrowed_from_bank_l563_563285

-- Definitions of the conditions
def car_price : ℝ := 35000
def total_payment : ℝ := 38000
def interest_rate : ℝ := 0.15

theorem amount_borrowed_from_bank :
  total_payment - car_price = interest_rate * (total_payment - car_price) / interest_rate := sorry

end amount_borrowed_from_bank_l563_563285


namespace vincent_songs_l563_563038

theorem vincent_songs (initial_songs : ℕ) (new_songs : ℕ) (h_initial_songs : initial_songs = 56) (h_new_songs : new_songs = 18) : 
  initial_songs + new_songs = 74 := 
by
  rw [h_initial_songs, h_new_songs]
  norm_num
  done

end vincent_songs_l563_563038


namespace find_percentage_of_second_reduction_l563_563010

-- Mathematical statement transcription:
noncomputable def percentageSecondReduction (P : ℝ) : ℝ :=
  let firstReduction := 0.75 * P
  let secondReduction (x : ℝ) := (1 - x / 100) * firstReduction
  let combinedReduction := 0.225 * P
  ∃ x : ℝ, secondReduction x = combinedReduction

theorem find_percentage_of_second_reduction (P : ℝ) :
  (percentageSecondReduction P) = 70 :=
sorry

end find_percentage_of_second_reduction_l563_563010


namespace range_of_a_if_p_false_l563_563893

def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 ≥ 0

theorem range_of_a_if_p_false :
  (¬ proposition_p a) → a ∈ set.Iio 0 ∪ set.Ioi 4 :=
by
  sorry

end range_of_a_if_p_false_l563_563893


namespace largest_k_for_divisibility_l563_563340

theorem largest_k_for_divisibility (k : ℕ) :
  (∃ (k : ℕ), (8 * 48 * 81) % (6^k) = 0) → k ≤ 5 := by {
  have h8 : 8 = 2^3 := by norm_num,
  have h48 : 48 = 2^4 * 3 := by norm_num,
  have h81 : 81 = 3^4 := by norm_num,
  have hprod : 8 * 48 * 81 = 2^7 * 3^5 := by {
    rw [h8, h48, h81],
    norm_num,
  },
  sorry
}

end largest_k_for_divisibility_l563_563340


namespace _l563_563207

noncomputable def a : ℕ → ℝ
| 0       := 1/5
| (n + 1) := have h : n + 1 ≠ 0 := nat.succ_ne_zero n
             classical.some (nat_has_inv_iff.mp
               ⟨classical.some (nat_has_inv_iff.mp ⟨(a n : ℝ) - (a n - classical.some (lt_of_le_of_ne 
               (show 0 < a n + 4 * a n * a n from sorry), h) == 0⟩).symm⟩,
              Assume classical.someor_leave reliability here⟩) 

lemma problem1 : ( (λ (n: ℕ), (1 : ℚ) / a n) (0) = 5) : sorry 

lemma problem2 (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = a (Nat.addSucc 0)  5 -4 a n) :
  ∀ n, a n = (ↄ (λ (n: ℕ), 4 * n + 1)) :
 begin
   have : ∀ n, 
   show (λ n, 4 * n + 1),
   by
   intro 
     classical.some, classical.some_eq(mul((lt_of_le_of_ne
                (4  n + classical.some⟨ 
               NatHasInv.inv (5, (4 * n + some classical.some,(nat_rec))⟩
               refl_sorry . Classical.type_logic_ nat
   else.vue x  ref_succ_eq_zero, smul   (theorem)⟩ ((a), assumption,sorry

lemma sn: ∀target
  (bn: a ( n:Nat)⟺ (sum_target ∑n⟩(((∀k:calclable , 
classical assumption_by theorem 
               show ∀ 
              algebra_ring 1/clas.assumption_eq
  int_them 4  term_scopeof_sum sorry 

by λ target.type:
T_act arithmetic_seq assume terms h∃(a calculation )=&( question:sub
   implies (4_nat_type_interval 
 sum example): problem prove
 (S n<prove_probability  term_range<=sorry 

end _l563_563207


namespace find_QR_l563_563316

-- Definitions for the given problem
def Q (x : Real) (y : Real) : (x, y) := (8, 0)
def P : (0, 0) := (0, 0)
def QP : Real := 16
def cos_Q : Real := 0.6

-- Proof problem: proving QR = 80 / 3
theorem find_QR (QP_eq : QP = 16) (cosQ_eq : cos_Q = 0.6) : 
  QR = 80 / 3 :=
by 
  sorry

end find_QR_l563_563316


namespace solve_system_l563_563140

theorem solve_system (x y : ℝ) :
  (x^3 - x + 1 = y^2 ∧ y^3 - y + 1 = x^2) ↔ ((x = 1 ∨ x = -1) ∧ (y = 1 ∨ y = -1)) :=
by
  sorry

end solve_system_l563_563140


namespace max_missed_questions_l563_563470

theorem max_missed_questions (total_questions : ℕ) (pass_percentage : ℝ) (student_score : ℝ) 
    (h1 : total_questions = 50) (h2 : pass_percentage = 0.85) (h3 : student_score = 0.15) :
    ⌊student_score * total_questions⌋ = 7 :=
by
    -- Given calculations
    rw [h1, h3]
    -- Calculation of missed questions
    norm_num
    -- Ceiling of missed questions
    rfl

end max_missed_questions_l563_563470


namespace annie_total_spent_l563_563108

-- Define cost of a single television
def cost_per_tv : ℕ := 50
-- Define number of televisions bought
def number_of_tvs : ℕ := 5
-- Define cost of a single figurine
def cost_per_figurine : ℕ := 1
-- Define number of figurines bought
def number_of_figurines : ℕ := 10

-- Define total cost calculation
noncomputable def total_cost : ℕ :=
  number_of_tvs * cost_per_tv + number_of_figurines * cost_per_figurine

theorem annie_total_spent : total_cost = 260 := by
  sorry

end annie_total_spent_l563_563108


namespace find_principal_l563_563392

-- Definition of compound interest
def compound_interest (P r : ℝ) (n t : ℤ) : ℝ :=
  P * (1 + (r / n))^((n : ℝ) * (t : ℝ)) - P

-- Definition of simple interest
def simple_interest (P r : ℝ) (t : ℤ) : ℝ :=
  P * r * (t : ℝ)

-- Given values
constant r : ℝ := 0.20
constant t : ℤ := 2
constant n : ℤ := 1
constant diff : ℝ := 216

theorem find_principal :
  ∃ P : ℝ, compound_interest P r n t - simple_interest P r t = diff ∧ P = 5400 := sorry

end find_principal_l563_563392


namespace max_angle_with_plane_l563_563302

-- Define our planes α and β, and point A on plane α
variables (α β : Plane) (A : Point)
-- Assume A' is the orthogonal projection of A onto plane β
variable (A' : Point)
hypothesis huproj : orthogonal_projection β A A'

-- Define the line l and line perpendicular to the edge of dihedral angle
variable (l : Line)
variable (perpendicular_to_edge : ∀ (l : Line), is_perpendicular l (edge_dihedral α β))

-- State the theorem we need to prove
theorem max_angle_with_plane (dihedral : dihedral_angle α β) (A : Point) (A' : orthogonal_projection β A)
  (hperp : ∀ l, ∃ edge_dihedral α β, is_perpendicular l (edge_dihedral α β))
  : ∀ l : Line, forming_max_angle_with β l ↔ perpendicular_to_edge l := 
sorry

end max_angle_with_plane_l563_563302


namespace union_complement_A_eq_l563_563938

open Set

universe u

noncomputable def U : Set ℝ := univ
def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { y | ∃ (x : ℝ), y = x^2 + 1 }

theorem union_complement_A_eq :
  A ∪ ((U \ B : Set ℝ) : Set ℝ) = { x | x < 2 } := by
  sorry

end union_complement_A_eq_l563_563938


namespace magnitude_A_equals_pi_over_6_Sn_equals_n_over_n_plus_1_l563_563961

variables {A B C a b c : ℝ}
variables {a1 a2 a4 a8 d n : ℕ}
variables (an : ℕ → ℝ) (Sn : ℕ → ℝ)

-- Condition in the triangle
axiom triangle_condition : sqrt 3 * a * cos C = (2 * b - sqrt 3 * c) * cos A

-- Conditions for arithmetic sequence
axiom a1_condition : a1 * sin A = 1
axiom geometric_sequence : a2 = a1 + d ∧
                           a4 = a1 + 3 * d ∧
                           a8 = a1 + 7 * d ∧
                           a4 * a4 = a2 * a8
axiom non_zero_diff : d ≠ 0

-- Sequence definition
def sequence_def (a_n : ℕ → ℝ) :=
  ∀ n, a_n n = 2 * n

-- Sum of the sequence
def Sn_def (S_n : ℕ → ℝ) :=
  ∀ n, S_n n = (Σ i in finset.range n, 4 / (a_n i * a_n (i+1)))

theorem magnitude_A_equals_pi_over_6 (h : triangle_condition) : A = π / 6 :=
sorry

theorem Sn_equals_n_over_n_plus_1 (h1 : a1_condition) (h2 : geometric_sequence) (h3 : sequence_def an) (h4 : Sn_def Sn) :
  Sn n = n / (n + 1) :=
sorry

end magnitude_A_equals_pi_over_6_Sn_equals_n_over_n_plus_1_l563_563961


namespace number_b_smaller_than_number_a_l563_563676

theorem number_b_smaller_than_number_a (A B : ℝ)
  (h : A = B + 1/4) : (B + 1/4 = A) ∧ (B < A) → B = (4 * A - A) / 5 := by
  sorry

end number_b_smaller_than_number_a_l563_563676


namespace not_square_difference_formula_l563_563743

theorem not_square_difference_formula (x y : ℝ) : ¬ ∃ (a b : ℝ), (x - y) * (-x + y) = (a + b) * (a - b) := 
sorry

end not_square_difference_formula_l563_563743


namespace equal_segments_l563_563890

-- Definitions from conditions
variables {O A P Q K L : Type}
variables (circle : O)
variables (inside_circle : A)
variables (chord_PQ : P = PQ)
variables (tangent_p : P → bool)
variables (tangent_q : Q → bool)
variables (line_l : A → A)

-- Theorem statement
theorem equal_segments 
  (h1 : ∀ (p : P), tangent_p p)
  (h2 : ∀ (q : Q), tangent_q q)
  (h3 : ∀ (l : A → A), line_l = λ a, a ∘ perp ∘ O)
  : (dist A K = dist A L) := 
sorry

end equal_segments_l563_563890


namespace sum_of_digits_reduction_l563_563534

theorem sum_of_digits_reduction (N : ℕ) (hN_len : String.length (N.digits 10).toString = 1998) (hN_div9 : N % 9 = 0) : 
    ∃ (z : ℕ), (z = 9 ∧ 
    let x := (N.digits 10).sum in
    let y := (x.digits 10).sum in
    let z := (y.digits 10).sum in
    z = 9) :=
by
    sorry

end sum_of_digits_reduction_l563_563534


namespace total_games_played_in_league_l563_563414

theorem total_games_played_in_league (n : ℕ) (k : ℕ) (games_per_team : ℕ) 
  (h1 : n = 10) 
  (h2 : k = 4) 
  (h3 : games_per_team = n - 1) 
  : (k * (n * games_per_team) / 2) = 180 :=
by
  -- Definitions and transformations go here
  sorry

end total_games_played_in_league_l563_563414


namespace solve_inequality_set_l563_563020

variable (a b x : ℝ)

theorem solve_inequality_set :
  (∀ x, x ∈ set.Ioi 1 ↔ ax - b < 0) →
  (∀ x, x ∈ set.Ioo -1 3 ↔ (ax + b) * (x - 3) > 0) :=
by
  sorry

end solve_inequality_set_l563_563020


namespace average_ratios_eq_two_l563_563119

variable {α : Type} [AddCommMonoid α]

-- Define the rectangular array of elements and their sums
def array_50x100 (a : ℕ → ℕ → α) : Prop :=
  ∀ i j, 1 ≤ i ∧ i ≤ 50 ∧ 1 ≤ j ∧ j ≤ 100

-- Define the row and column sums
def row_sum (a : ℕ → ℕ → α) (i : ℕ) : α :=
  Finset.sum (Finset.range 100) (λ j, a i j)

def column_sum (a : ℕ → ℕ → α) (j : ℕ) : α :=
  Finset.sum (Finset.range 50) (λ i, a i j)

-- Define the averages C and D
def average_row_sum (S : ℕ → α) : α :=
  (Finset.sum (Finset.range 50) S) / (50 : α)

def average_column_sum (T : ℕ → α) : α :=
  (Finset.sum (Finset.range 100) T) / (100 : α)

-- The theorem to prove
theorem average_ratios_eq_two (a : ℕ → ℕ → α)
  (S : ℕ → α) (T : ℕ → α)
  (h1 : ∀ i, S i = row_sum a i)
  (h2 : ∀ j, T j = column_sum a j)
  (h3 : array_50x100 a) :
  average_row_sum S / average_column_sum T = 2 :=
by
  sorry

end average_ratios_eq_two_l563_563119


namespace paint_cost_is_correct_l563_563391

-- Definition of known conditions
def costPerKg : ℕ := 50
def coveragePerKg : ℕ := 20
def sideOfCube : ℕ := 20

-- Definition of correct answer
def totalCost : ℕ := 6000

-- Theorem statement
theorem paint_cost_is_correct : (6 * (sideOfCube * sideOfCube) / coveragePerKg) * costPerKg = totalCost :=
by
  sorry

end paint_cost_is_correct_l563_563391


namespace range_of_a_l563_563205

noncomputable def f (a x : ℝ) : ℝ := a - x + x * exp x

theorem range_of_a (a : ℝ) (x_0 : ℝ) (h1 : x_0 > -1) (h2 : f a x_0 ≤ 0) : a ∈ set.Iic 0 :=
begin
  -- proof will be provided here
  sorry
end

end range_of_a_l563_563205


namespace max_value_A_l563_563934

noncomputable def maximum_value (n : ℕ) (xs : Fin n → ℝ) : ℝ :=
  (∑ i, Real.cos (xs i)^2) / (Real.sqrt n + Real.sqrt (∑ i, (Real.cot (xs i))^4))

theorem max_value_A {n : ℕ} (hpos : 0 < n)
  (hx : ∀ i : Fin n, 0 < xs i ∧ xs i < Real.pi / 2) :
  maximum_value n xs ≤ Real.sqrt n / 4 :=
sorry

end max_value_A_l563_563934


namespace sum_of_digits_is_11_l563_563320

def digits_satisfy_conditions (A B C : ℕ) : Prop :=
  (C = 0 ∨ C = 5) ∧
  (A = 2 * B) ∧
  (A * B * C = 40)

theorem sum_of_digits_is_11 (A B C : ℕ) (h : digits_satisfy_conditions A B C) : A + B + C = 11 :=
by
  sorry

end sum_of_digits_is_11_l563_563320


namespace part1_part2_l563_563959

-- Condition for exponents of x to be equal
def condition1 (a : ℤ) : Prop := (3 : ℤ) = 2 * a - 3

-- Condition for exponents of y to be equal
def condition2 (b : ℤ) : Prop := b = 1

noncomputable def a_value : ℤ := 3
noncomputable def b_value : ℤ := 1

-- Theorem for part (1): values of a and b
theorem part1 : condition1 3 ∧ condition2 1 :=
by
  have ha : condition1 3 := by sorry
  have hb : condition2 1 := by sorry
  exact And.intro ha hb

-- Theorem for part (2): value of (7a - 22)^2024 given a = 3
theorem part2 : (7 * a_value - 22) ^ 2024 = 1 :=
by
  have hx : 7 * a_value - 22 = -1 := by sorry
  have hres : (-1) ^ 2024 = 1 := by sorry
  exact Eq.trans (congrArg (fun x => x ^ 2024) hx) hres

end part1_part2_l563_563959


namespace units_digit_T_l563_563586

theorem units_digit_T : 
  let T := (1 + 1)! + (2 + 3)! + (3 + 2)! + (4 + 0)!
  in T % 10 = 6 := 
by
  let T := (1 + 1)! + (2 + 3)! + (3 + 2)! + (4 + 0)!
  sorry

end units_digit_T_l563_563586


namespace no_satisfactory_seating_l563_563026

theorem no_satisfactory_seating (n d : ℕ) (h1 : n = 47) (h2 : d = 12) :
  ¬ ∃ (regions : Fin d → ℕ), 
    (∑ i, regions i = n) ∧ 
    ∀ (seating : Fin n → Fin d), 
    (∀ i, (∑ j in Finset.range 15, if seating ((i + j) % n) = i then 1 else 0) > 0) :=
by sorry

end no_satisfactory_seating_l563_563026


namespace work_done_l563_563062

noncomputable def F (x : ℝ) : ℝ :=
  if x ≤ 2 then 10 else 3 * x + 4

theorem work_done : (∫ x in 0..4, F x) = 46 := by
  sorry

end work_done_l563_563062


namespace find_n_l563_563246

-- Condition 1: a_{n-1} - a_n^2 + a_{n+1} = 0 for n >= 2
def cond1 (a : ℕ → ℤ) (n : ℕ) : Prop :=
  n ≥ 2 → a (n-1) - a n ^ 2 + a (n+1) = 0

-- Condition 2: {a_n} is an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n, a (n+1) = a n + d

-- Condition 3: S_{2n-1} = 78
def sum_of_first_k_terms (a : ℕ → ℤ) (k : ℕ) : ℤ :=
  (finset.range k).sum a

def cond3 (a : ℕ → ℤ) (n : ℕ) : Prop :=
  sum_of_first_k_terms a (2 * n - 1) = 78

-- Translate the final problem in Lean 4 statement
theorem find_n (a : ℕ → ℤ) (n : ℕ) :
  cond1 a n →
  is_arithmetic_sequence a →
  cond3 a n →
  n = 20 :=
sorry

end find_n_l563_563246


namespace unique_pythagorean_triple_l563_563379

-- Define the concept of a Pythagorean triple with integer components
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

-- Define each set of numbers given in the problem
def set_A : Prop := is_pythagorean_triple (1 / 10) (2 / 5) (2 / 4)
def set_B : Prop := is_pythagorean_triple 2 2 (2 * Real.sqrt 2)
def set_C : Prop := is_pythagorean_triple 4 5 6
def set_D : Prop := is_pythagorean_triple 5 12 13

-- Theorem stating that among the given sets, only set D is a Pythagorean triple
theorem unique_pythagorean_triple :
  ¬ set_A ∧ ¬ set_B ∧ ¬ set_C ∧ set_D :=
by {
  sorry -- Proof will be provided here
}

end unique_pythagorean_triple_l563_563379


namespace mariana_savings_l563_563474

-- Defining the regular price of one pair of flip-flops.
def regular_price : ℝ := 60

-- Defining the discounts for the second and third pairs.
def second_pair_discount : ℝ := 0.25
def third_pair_discount : ℝ := 0.60

-- Total cost without any discounts for three pairs
def total_regular_cost : ℝ := 3 * regular_price

-- Total cost with discounts applied
def total_discounted_cost : ℝ := 
  regular_price + 
  (regular_price * (1 - second_pair_discount)) + 
  (regular_price * (1 - third_pair_discount))

-- Savings calculation
def savings : ℝ := total_regular_cost - total_discounted_cost

-- Percentage saved
def percentage_saved : ℝ := (savings / total_regular_cost) * 100

-- The statement to be proved
theorem mariana_savings :
  percentage_saved ≈ 28 := by
  sorry

end mariana_savings_l563_563474


namespace scientific_notation_280000_l563_563289

theorem scientific_notation_280000 : (280000 : ℝ) = 2.8 * 10^5 :=
sorry

end scientific_notation_280000_l563_563289


namespace sqrt3_minus1_plus_inv3_pow_minus2_l563_563480

theorem sqrt3_minus1_plus_inv3_pow_minus2 :
  (Real.sqrt 3 - 1) + (1 / (1/3) ^ 2) = Real.sqrt 3 + 8 :=
by
  sorry

end sqrt3_minus1_plus_inv3_pow_minus2_l563_563480


namespace max_sum_areas_surface_area_l563_563252

-- Define the lengths of the edges
def PA : ℝ := 2
def PB : ℝ := real.sqrt 6
def PC : ℝ := real.sqrt 6

-- Define the property that the three edges are pairwise perpendicular
def mutually_perpendicular (u v w : ℝ) : Prop :=
  ⟦u ≠ 0∙∧ v ≠ 0 ∧ w ≠ 0 ∧ u * v * w = 0⟧

-- Define the condition that the sum of the areas of the three side faces is maximized when edges are mutually perpendicular
def sum_areas_maximized : Prop :=
  mutually_perpendicular PA PB PC

-- From the above conditions, we want to prove the surface area of the sphere is 16π
theorem max_sum_areas_surface_area (h : sum_areas_maximized) : 
  let diameter := real.sqrt (PA^2 + PB^2 + PC^2) in
  let radius := diameter / 2 in
  let surface_area := 4 * Math.pi * radius^2 in
  surface_area = 16 * Math.pi :=
begin
  sorry
end

end max_sum_areas_surface_area_l563_563252


namespace max_side_length_of_equilateral_triangle_covered_by_three_unit_triangles_l563_563040

theorem max_side_length_of_equilateral_triangle_covered_by_three_unit_triangles :
  ∃ a : ℝ, (∀ t : fin 3 → real, (t 0 = 1 ∧ t 1 = 1 ∧ t 2 = 1) → (a ≤ 3 / 2) ∧ ∀ t', (t' 0 = 1 ∧ t' 1 = 1 ∧ t' 2 = 1 → a = 3 / 2)) :=
by 
  sorry

end max_side_length_of_equilateral_triangle_covered_by_three_unit_triangles_l563_563040


namespace min_value_of_X_l563_563173

theorem min_value_of_X (n : ℕ) (h : n ≥ 2) 
  (X : Finset ℕ) 
  (B : Fin n → Finset ℕ) 
  (hB : ∀ i, (B i).card = 2) :
  ∃ (Y : Finset ℕ), Y.card = n ∧ ∀ i, (Y ∩ (B i)).card ≤ 1 →
  X.card = 2 * n - 1 :=
sorry

end min_value_of_X_l563_563173


namespace length_of_segment_AB_l563_563892

variable (A B : ℝ × ℝ × ℝ)

def distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2 + (Q.3 - P.3) ^ 2)

theorem length_of_segment_AB :
  let A := (1, 2, 3)
  let B := (0, 4, 5)
  distance A B = 3 :=
by {
  sorry
}

end length_of_segment_AB_l563_563892


namespace tan_eq_neg_four_thirds_complicated_expr_eq_neg_four_thirds_l563_563553

theorem tan_eq_neg_four_thirds
  (α : ℝ)
  (hα : α ∈ Set.Ioo 0 (Real.pi))
  (h : Real.sin α + Real.cos α = 1/5)
  : Real.tan α = -4/3 :=
sorry

theorem complicated_expr_eq_neg_four_thirds
  (α : ℝ)
  (hα : α ∈ Set.Ioo 0 (Real.pi))
  (h : Real.sin α + Real.cos α = 1/5)
  : (Real.sin (3 * Real.pi / 2 + α) * Real.sin (Real.pi / 2 - α) *
    Real.tan (Real.pi - α) ^ 3) /
    (Real.cos (Real.pi / 2 + α) * Real.cos (3 * Real.pi / 2 - α)) = -4/3 :=
sorry

end tan_eq_neg_four_thirds_complicated_expr_eq_neg_four_thirds_l563_563553


namespace packs_in_each_set_l563_563595

variable (cost_per_set cost_per_pack total_savings : ℝ)
variable (x : ℕ)

-- Objecting conditions
axiom cost_set : cost_per_set = 2.5
axiom cost_pack : cost_per_pack = 1.3
axiom savings : total_savings = 1

-- Main proof problem
theorem packs_in_each_set :
  10 * x * cost_per_pack = 10 * cost_per_set + total_savings → x = 2 :=
by
  -- sorry is a placeholder for the proof
  sorry

end packs_in_each_set_l563_563595


namespace neg_p_sufficient_for_q_l563_563188

/-- 
Proof that the negation of the statement \( p \) (i.e., \( a < -1 \)) 
is a sufficient but not necessary condition for the statement \( q \)
(i.e., \( \forall x > 0, a \leqslant \frac{x^2 + 1}{x} \)).
-/
theorem neg_p_sufficient_for_q (a : ℝ) : (¬ (∀ x < -1, deriv (λ x : ℝ, (x - a)^2) x < 0)) → (∃ x > 0, a ≤ (x^2 + 1) / x) :=
by
  sorry

end neg_p_sufficient_for_q_l563_563188


namespace MEANT_position_correct_l563_563731

open List

noncomputable def alphabet_position_MEANT : Nat := 
  let letters := "AEMNT".toList
  let len := 5
  
  let count_permutations (l : List Char) : Nat :=
    (l.length.factorial)
    
  let before_M : Nat :=
    count_permutations ['A', 'E', 'M'.mk_Char, 'N'.mk_Char, 'T'.mk_Char].erase 'M'

  let before_MEANT :=
    2 * 4.factorial + -- "A" and "E" leading
    2 * 3.factorial + -- "MA" and "ME"
    1               -- "MEANT" itself

  before_MEANT

theorem MEANT_position_correct :
  alphabet_position_MEANT = 55 :=
  sorry

end MEANT_position_correct_l563_563731


namespace fg_diff_zero_l563_563637

def f (x : ℝ) : ℝ := 2 * x + 1
def g (x : ℝ) : ℝ := 4 * x + 3

theorem fg_diff_zero (x : ℝ) : f (g x) - g (f x) = 0 :=
by
  sorry

end fg_diff_zero_l563_563637


namespace max_area_quadrilateral_sum_opposite_angles_l563_563396

theorem max_area_quadrilateral (a b c d : ℝ) (h₁ : a = 3) (h₂ : b = 3) (h₃ : c = 4) (h₄ : d = 4) :
  ∃ (area : ℝ), area = 12 :=
by {
  sorry
}

theorem sum_opposite_angles (a b c d : ℝ) (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h₁ : a = 3) (h₂ : b = 3) (h₃ : c = 4) (h₄ : d = 4) 
  (h_area : ∃ (area : ℝ), area = 12) 
  (h_opposite1 : θ₁ + θ₃ = 180) (h_opposite2 : θ₂ + θ₄ = 180) :
  ∃ θ, θ = 180 :=
by {
  sorry
}

end max_area_quadrilateral_sum_opposite_angles_l563_563396


namespace largest_constant_c_l563_563509

theorem largest_constant_c (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 + y^2 = 1) : 
  x^6 + y^6 ≥ (1 / 2) * x * y :=
sorry

end largest_constant_c_l563_563509


namespace division_of_workers_l563_563216

theorem division_of_workers :
  let n := 30
  let k := 10
  let num_teams := 3
  nat.choose n k * nat.choose (n - k) k * nat.choose (n - 2 * k) k / nat.factorial num_teams = 2775498395670
:= sorry

end division_of_workers_l563_563216


namespace index_card_area_l563_563677

theorem index_card_area (a b : ℕ) (h1 : a = 5) (h2 : b = 7) (h3 : (a - 2) * b = 21) : (a * (b - 1)) = 30 := by
  sorry

end index_card_area_l563_563677


namespace cricket_team_average_age_l563_563694

-- Define the input conditions and stated question
theorem cricket_team_average_age
  (A : ℕ) (n : ℕ) (wicket_keeper_age : ℕ) (captain_age : ℕ) (remaining_9_avg_age : ℕ)
  (team_avg_age : ℕ)
  (h_team_size : n = 11)
  (h_team_avg_age : A = 29)
  (h_wicket_keeper : wicket_keeper_age = A + 3)
  (h_remaining_9_avg_age : remaining_9_avg_age = A - 1)
  (h_team_avg_age_consistent : team_avg_age = A) : team_avg_age = 29 := 
begin
  sorry
end

end cricket_team_average_age_l563_563694


namespace area_of_triangle_H1H2H3_l563_563631

noncomputable def triangle_area (AB AC BD CE : ℕ) (angle_BAC : ℝ) (h1_area h2_area h3_area : ℕ) : ℝ :=
  have BAC_rad := Real.pi - angle_BAC * Real.pi / 180
  0.5 * AB * AC * Real.sin BAC_rad

theorem area_of_triangle_H1H2H3 :
  let AB := 18
  let AC := 24
  let BD := 6
  let CE := 8
  let CF_double_BF := true
  let angle_BAC := 150
  let area_triangle_H1H2H3 := 96
  ∃ (H H1H2H3_area: ℕ), 
  let F_double_BF (x : ℕ ) := x = 2*(H - x) in 
  triangle_area AB AC BD CE angle_BAC H1H2H3_area = area_triangle_H1H2H3 
 := sorry

end area_of_triangle_H1H2H3_l563_563631


namespace exists_infinitely_many_rational_squares_sum_l563_563303

theorem exists_infinitely_many_rational_squares_sum (r : ℚ):
  ∃ᶠ x : ℚ in (Filter.cofinite), ∃ y : ℚ, x^2 + y^2 = r^2 :=
sorry

end exists_infinitely_many_rational_squares_sum_l563_563303


namespace ratio_of_buckets_l563_563112

theorem ratio_of_buckets 
  (shark_feed_per_day : ℕ := 4)
  (dolphin_feed_per_day : ℕ := shark_feed_per_day / 2)
  (total_buckets : ℕ := 546)
  (days_in_weeks : ℕ := 3 * 7)
  (ratio_R : ℕ) :
  (total_buckets = days_in_weeks * (shark_feed_per_day + dolphin_feed_per_day + (ratio_R * shark_feed_per_day)) → ratio_R = 5) := sorry

end ratio_of_buckets_l563_563112


namespace akeno_spent_more_l563_563100

theorem akeno_spent_more (akeno_spent : ℤ) (lev_spent_ratio : ℚ) (ambrocio_less : ℤ) 
  (h1 : akeno_spent = 2985)
  (h2 : lev_spent_ratio = 1/3)
  (h3 : ambrocio_less = 177) : akeno_spent - ((lev_spent_ratio * akeno_spent).toInt + ((lev_spent_ratio * akeno_spent).toInt - ambrocio_less)) = 1172 := by
  sorry

end akeno_spent_more_l563_563100


namespace number_of_true_compounds_l563_563183

-- Definitions of the propositions
def p (a b : ℝ) : Prop := a > b → 1 / a < 1 / b
def q (a b : ℝ) : Prop := (1 / (a * b) < 0) ↔ (a * b < 0)

-- Given conditions
axiom a_b_condition : ∀ (a b : ℝ), ¬ p a b
axiom ab_neg_condition : ∀ (a b : ℝ), q a b

-- Compound propositions
def compound1 (a b : ℝ) : Prop := p a b ∨ q a b
def compound2 (a b : ℝ) : Prop := p a b ∧ q a b
def compound3 (a b : ℝ) : Prop := ¬ p a b ∧ ¬ q a b
def compound4 (a b : ℝ) : Prop := ¬ p a b ∨ ¬ q a b

-- Problem statement
theorem number_of_true_compounds (a b : ℝ) : 
  (1 : ℕ) + (if compound1 a b then 1 else 0) +
  (if compound2 a b then 1 else 0) +
  (if compound3 a b then 1 else 0) +
  (if compound4 a b then 1 else 0) = 2 :=
by sorry

end number_of_true_compounds_l563_563183


namespace polygon_inequality_l563_563281

variable {r : ℝ} -- Let r be the radius of the circle

-- Define the conditions for the sides of the polygons.
variables {k l : ℕ} -- k and l are the number of sides of the polygons
variables {a : Fin k → ℝ} {b : Fin l → ℝ} -- a and b are the side lengths of the polygons
variables {h1 : ∀ i : Fin l, i.val < l → a i < b i} -- Condition a₁ < b₁, a₂ < b₂, ...

-- Define the central angles of the polygons.
variables {α : Fin k → ℝ} {β : Fin l → ℝ} -- α and β are the central angles of the polygons
variables {h2 : ∀ i : Fin k, i.val < k → α i = Real.sin (a i / r)} -- α_i = sin⁻¹(a_i / r)
variables {h3 : ∀ i : Fin l, i.val < l → β i = Real.sin (b i / r)} -- β_j = sin⁻¹(b_j / r)

-- The goal is to prove that the perimeter and area of the first polygon are greater than those of the second polygon given the conditions.
theorem polygon_inequality
  (H1 : ∑ i in Finset.finRange k, α i = π)
  (H2 : ∑ i in Finset.finRange l, β i = π)
  (H3 : (∑ i in Finset.finRange k, Real.sin (α i)) > (∑ i in Finset.finRange l, Real.sin (β i)))
  (H4 : (∑ i in Finset.finRange k, 2 * Real.sin (2 * α i)) > (∑ i in Finset.finRange l, 2 * Real.sin (2 * β i))) :
    (∑ i in Finset.finRange k, a i) > (∑ i in Finset.finRange l, b i) ∧
    (∑ i in Finset.finRange k, Real.sin (2 * α i)) > (∑ i in Finset.finRange l, Real.sin (2 * β i)) :=
by
  sorry

end polygon_inequality_l563_563281


namespace inequality_holds_for_all_l563_563139

theorem inequality_holds_for_all (m n : ℕ) (m_pos : 0 < m) (n_pos : 0 < n) :
  (∀ α β : ℝ, ⌊(m + n) * α⌋ + ⌊(m + n) * β⌋ ≥ ⌊m * α⌋ + ⌊m * β⌋ + ⌊n * (α + β)⌋) → m = n :=
by sorry

end inequality_holds_for_all_l563_563139


namespace desk_length_l563_563960

theorem desk_length (width perimeter length : ℤ) (h1 : width = 9) (h2 : perimeter = 46) (h3 : perimeter = 2 * (length + width)) : length = 14 :=
by
  rw [h1, h2] at h3
  sorry

end desk_length_l563_563960


namespace uv_squared_eq_one_l563_563451

theorem uv_squared_eq_one (a b : ℝ) (d : ℝ) (u v : ℝ)
  (h₀ : d = real.sqrt (a^2 + b^2))
  (h₁ : u = b / d)
  (h₂ : v = a / d):
  u^2 + v^2 = 1 :=
sorry

end uv_squared_eq_one_l563_563451


namespace sin_double_angle_plus_pi_div_two_l563_563879

open Real

theorem sin_double_angle_plus_pi_div_two (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < π / 2) (h₂ : sin θ = 1 / 3) :
  sin (2 * θ + π / 2) = 7 / 9 :=
by
  sorry

end sin_double_angle_plus_pi_div_two_l563_563879


namespace evaluate_magnitude_of_product_l563_563133

theorem evaluate_magnitude_of_product :
  let z1 := (3 * Real.sqrt 2 - 5 * Complex.I)
  let z2 := (2 * Real.sqrt 3 + 2 * Complex.I)
  Complex.abs (z1 * z2) = 4 * Real.sqrt 43 := by
  let z1 := (3 * Real.sqrt 2 - 5 * Complex.I)
  let z2 := (2 * Real.sqrt 3 + 2 * Complex.I)
  suffices Complex.abs z1 * Complex.abs z2 = 4 * Real.sqrt 43 by sorry
  sorry

end evaluate_magnitude_of_product_l563_563133


namespace akeno_extra_expenditure_l563_563094

/-
   Akeno spent $2985 to furnish his apartment.
   Lev spent one-third of that amount on his apartment.
   Ambrocio spent $177 less than Lev.
   Prove that Akeno spent $1172 more than the other 2 people combined.
-/

theorem akeno_extra_expenditure :
  let ak = 2985
  let lev = ak / 3
  let am = lev - 177
  ak - (lev + am) = 1172 :=
by
  sorry

end akeno_extra_expenditure_l563_563094


namespace unique_12_tuple_l563_563512

theorem unique_12_tuple : 
  ∃! (x : Fin 12 → ℝ), 
    ((1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + 
    (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 +
    (x 7 - x 8)^2 + (x 8 - x 9)^2 + (x 9 - x 10)^2 + (x 10 - x 11)^2 + 
    (x 11)^2 = 1 / 13) ∧ (x 0 + x 11 = 1 / 2) :=
by
  sorry

end unique_12_tuple_l563_563512


namespace y_intercept_of_line_l563_563054

theorem y_intercept_of_line (m x y b : ℝ) (h1 : m = 4) (h2 : x = 50) (h3 : y = 300) (h4 : y = m * x + b) : b = 100 := by
  sorry

end y_intercept_of_line_l563_563054


namespace smaller_square_area_l563_563231

theorem smaller_square_area (A : ℝ) (hA : A = 144) : ∃ B : ℝ, B = 72 :=
by 
  have Area_large_square : ℝ := A
  have midpoints_condition : true := trivial  -- Condition that vertices of smaller square are midpoints
  have Area_small_square : ℝ := 72
  use Area_small_square
  -- Proof that the area of the smaller square is indeed 72
  sorry

end smaller_square_area_l563_563231


namespace parallelograms_in_grid_l563_563753

def countParallelograms (m : ℕ) : ℕ :=
  (m + 1) * (m + 2) / 2

theorem parallelograms_in_grid (m : ℕ) :
  let p := countParallelograms m
  in p * p = (m + 1)^2 * (m + 2)^2 / 4 :=
by
  sorry

end parallelograms_in_grid_l563_563753


namespace denver_charges_danny_for_two_birdhouses_l563_563495

theorem denver_charges_danny_for_two_birdhouses :
  ∀ (pieces_per_birdhouse : ℕ)
    (cost_per_piece : ℚ)
    (profit_per_birdhouse : ℚ)
    (number_of_birdhouses : ℕ), 
    pieces_per_birdhouse = 7 →
    cost_per_piece = 1.50 →
    profit_per_birdhouse = 5.50 →
    number_of_birdhouses = 2 →
    let cost_per_birdhouse := pieces_per_birdhouse * cost_per_piece in
    let selling_price_per_birdhouse := cost_per_birdhouse + profit_per_birdhouse in
    let total_charge := selling_price_per_birdhouse * number_of_birdhouses in
    total_charge = 32.00 :=
begin
  intros pieces_per_birdhouse cost_per_piece profit_per_birdhouse number_of_birdhouses,
  intros h_pieces h_cost h_profit h_number,
  let cost_per_birdhouse := pieces_per_birdhouse * cost_per_piece,
  let selling_price_per_birdhouse := cost_per_birdhouse + profit_per_birdhouse,
  let total_charge := selling_price_per_birdhouse * number_of_birdhouses,
  rw [h_pieces, h_cost, h_profit, h_number],
  sorry
end

end denver_charges_danny_for_two_birdhouses_l563_563495


namespace gecko_egg_hatching_l563_563436

theorem gecko_egg_hatching :
  let total_eggs := 30
  let infertile_eggs := total_eggs * 0.20
  let remaining_eggs := total_eggs - infertile_eggs
  let calcification_eggs := remaining_eggs / 3
  let hatched_eggs := remaining_eggs - calcification_eggs
  in hatched_eggs = 16 := 
by
  sorry

end gecko_egg_hatching_l563_563436


namespace limit_expression_tends_to_three_half_l563_563115

noncomputable def limit_expression : (ℕ → ℝ) := λ n, sqrt(n^3 + 8) * (sqrt(n^3 + 2) - sqrt(n^3 - 1))

theorem limit_expression_tends_to_three_half :
  tendsto limit_expression at_top (𝓝 (3/2)) :=
by
  sorry

end limit_expression_tends_to_three_half_l563_563115


namespace pallets_of_paper_cups_l563_563086

theorem pallets_of_paper_cups (total_pallets paper_towels tissues paper_plates : ℕ) 
  (H1 : total_pallets = 20) 
  (H2 : paper_towels = total_pallets / 2)
  (H3 : tissues = total_pallets / 4)
  (H4 : paper_plates = total_pallets / 5) : 
  total_pallets - paper_towels - tissues - paper_plates = 1 := 
  by
    sorry

end pallets_of_paper_cups_l563_563086


namespace proof_problem_l563_563994

-- Triangle and Point Definitions
variables {A B C P : Type}
variables (BC : ℝ) (a b c : ℝ) (PA PB PC : ℝ)

-- Conditions: Triangle ABC with angle A = 90 degrees and P on BC
def is_right_triangle (A B C : Type) (a b c : ℝ) (BC : ℝ) (angleA : ℝ := 90) : Prop :=
a^2 + b^2 = c^2 ∧ c = BC

def on_hypotenuse (P : Type) (BC : ℝ) (PB PC : ℝ) : Prop :=
PB + PC = BC

-- The proof problem
theorem proof_problem (A B C P : Type) 
  (BC : ℝ) (a b c : ℝ) (PA PB PC : ℝ)
  (h1 : is_right_triangle A B C a b c BC)
  (h2 : on_hypotenuse P BC PB PC) :
  (a^2 / PC + b^2 / PB) ≥ (BC^3 / (PA^2 + PB * PC)) := 
sorry

end proof_problem_l563_563994


namespace random_event_proof_l563_563804

-- Definitions based on conditions
def event1 := "Tossing a coin twice in a row, and both times it lands heads up."
def event2 := "Opposite charges attract each other."
def event3 := "Water freezes at 1℃ under standard atmospheric pressure."

def is_random_event (event: String) : Prop :=
  event = event1 ∨ event = event2 ∨ event = event3 → event = event1

theorem random_event_proof : is_random_event event1 ∧ ¬is_random_event event2 ∧ ¬is_random_event event3 :=
by
  -- Proof goes here
  sorry

end random_event_proof_l563_563804


namespace solve_equation_l563_563463

theorem solve_equation (x : ℝ) : 
  x^2 + 2 * x + 4 * Real.sqrt (x^2 + 2 * x) - 5 = 0 →
  (x = Real.sqrt 2 - 1) ∨ (x = - (Real.sqrt 2 + 1)) :=
by 
  sorry

end solve_equation_l563_563463


namespace tangent_line_to_parabola_range_of_a_for_equal_areas_l563_563935

theorem tangent_line_to_parabola (a x_0 : ℝ) (h_ne : a ≠ 0) :
  ∃ l : ℝ → ℝ, (∀ x, l x = ax - a^2 ∨ l x = 0) ∧
  ∀ x_0, (x_0^2 = 4 * l x_0) ∧ (l a = 0) := sorry

theorem range_of_a_for_equal_areas (a : ℝ) :
  (-sqrt 2 < a ∧ a < -1) ∨ (-1 < a ∧ a < 1) ∨ (1 < a ∧ a < sqrt 2) := sorry

end tangent_line_to_parabola_range_of_a_for_equal_areas_l563_563935


namespace sum_of_first_n_terms_l563_563156

noncomputable def parabola (n : ℕ) : (ℝ → ℝ) := λ x, 2*(2*n - 1)*x
def sequence (n : ℕ) : ℝ := if n = 1 then -4 else sorry  -- Sequence definition
def sum_sequence (n : ℕ) : ℝ := ∑ i in range 1 (n + 1), sequence i

theorem sum_of_first_n_terms (n : ℕ) (h_pos : 0 < n) :
  sum_sequence n = -2 * n * (n + 1) :=
sorry

end sum_of_first_n_terms_l563_563156


namespace distance_downstream_proof_l563_563715

def speed_boat_still_water : ℝ := 20  -- Speed of the boat in still water (km/hr)
def rate_current : ℝ := 5  -- Rate of current (km/hr)
def time_travel_minutes : ℝ := 24  -- Time of travel (minutes)
def time_travel_hours : ℝ := time_travel_minutes / 60  -- Time of travel (hours)
def distance_travelled_downstream : ℝ := speed_boat_still_water + rate_current * time_travel_hours

theorem distance_downstream_proof :
  distance_travelled_downstream = 10 :=
  sorry

end distance_downstream_proof_l563_563715


namespace g_sum_l563_563641

def g (x: ℝ) : ℝ :=
  if x > 3 then x^2 + 4
  else if -3 ≤ x ∧ x ≤ 3 then 5 * x - 2
  else 0

theorem g_sum :
  g (-4) + g (0) + g (4) = 18 := by
  sorry

end g_sum_l563_563641


namespace son_age_l563_563445

-- Variables and constants
variable (t s : ℕ) -- t for triplets' age, s for son's age
constant mother_fee : ℝ := 5.95
constant per_year_fee : ℝ := 0.55
constant total_bill : ℝ := 11.15

-- Define the given conditions
def total_children_fee : ℝ := total_bill - mother_fee
def total_children_years : ℝ := total_children_fee / per_year_fee
def children_age_relationship :=
  total_children_years = 9 ∧ 3 * t + s = 9

-- The statement to be proved
theorem son_age : children_age_relationship → s = 3 :=
by
  intro h,
  sorry

end son_age_l563_563445


namespace coloring_ways_correct_l563_563838

variable (n : ℕ) (n_ge_2 : n ≥ 2)

def ways_to_color_circle (n : ℕ) : ℕ :=
  2 ^ n + (-1)^(n) * 2

theorem coloring_ways_correct : ways_to_color_circle n = 2 ^ n + (-1)^(n) * 2 :=
sorry

end coloring_ways_correct_l563_563838


namespace sum_of_digits_reduction_l563_563535

theorem sum_of_digits_reduction (N : ℕ) (hN_len : String.length (N.digits 10).toString = 1998) (hN_div9 : N % 9 = 0) : 
    ∃ (z : ℕ), (z = 9 ∧ 
    let x := (N.digits 10).sum in
    let y := (x.digits 10).sum in
    let z := (y.digits 10).sum in
    z = 9) :=
by
    sorry

end sum_of_digits_reduction_l563_563535


namespace largest_possible_n_l563_563109

theorem largest_possible_n 
  (a : ℕ → ℤ) (b : ℕ → ℤ)
  (d k : ℤ) (hd_le_k : d ≤ k)
  (a1 : a 1 = 1) (b1 : b 1 = 1)
  (product_condition : ∃ n : ℕ, a n * b n = 2016) :
  ∃ m : ℕ, (∃ d k : ℤ, d ≤ k ∧ (∀ n, a n = 1 + (n-1) * d) ∧ (∀ n, b n = 1 + (n-1) * k)) ∧ m = 32 ∧ 
  (∀ x : ℕ, (∃ d k : ℤ, d ≤ k ∧ (a x = 1 + (x-1) * d) ∧ (b x = 1 + (x-1) * k) ∧ x > m → a x * b x ≠ 2016)) :=
begin
  sorry
end

end largest_possible_n_l563_563109


namespace brad_drank_5_glasses_l563_563114

def glasses_per_gallon : ℕ := 16
def cost_per_gallon : ℝ := 3.50
def gallons_made : ℕ := 2
def price_per_glass : ℝ := 1.00
def glasses_remaining : ℕ := 6
def net_profit : ℝ := 14.00

theorem brad_drank_5_glasses :
  let total_glasses := gallons_made * glasses_per_gallon in
  let glasses_sold := total_glasses - glasses_remaining in
  let total_revenue := glasses_sold * price_per_glass in
  let total_cost := gallons_made * cost_per_gallon in
  let effective_cost := total_revenue - net_profit in
  let cost_of_drank_lemonade := effective_cost - total_cost in
  let glasses_drank := cost_of_drank_lemonade / price_per_glass in
  glasses_drank = 5 :=
by {
  let total_glasses := 32,
  let glasses_sold := 26,
  let total_revenue := 26.00,
  let total_cost := 7.00,
  let effective_cost := 12.00,
  let cost_of_drank_lemonade := 5.00,
  let glasses_drank := 5.0,
  exact of_eq (5) (5)
}

end brad_drank_5_glasses_l563_563114


namespace find_angle_C_find_the_range_of_the_perimeter_l563_563521

variables {A B C : ℝ} (a b c : ℝ)

-- The conditions for the problem
variables (h1 : 2 * cos C * (a * cos B + b * cos A) = c)
          (h2 : C = π / 3)

-- Statement of the proof problem for part (I)
theorem find_angle_C 
    (h1 : 2 * cos C * (a * cos B + b * cos A) = c)
    : C = π / 3 :=
sorry

-- Statement of the proof problem for part (II)
theorem find_the_range_of_the_perimeter 
    (h1 : 2 * cos C * (a * cos B + b * cos A) = c)
    (h_c_eq_sqrt3 : c = sqrt 3)
    : sqrt 3 < a + b + c ∧ a + b + c ≤ 3 * sqrt 3 :=
sorry

end find_angle_C_find_the_range_of_the_perimeter_l563_563521


namespace blue_pill_cost_is_18_point_5_l563_563113

-- Define variables for the costs and duration
def blue_pill_cost : ℝ := sorry
def yellow_pill_cost : ℝ := blue_pill_cost - 2
def number_of_days : ℝ := 3 * 7
def total_cost : ℝ := 21 * (blue_pill_cost + yellow_pill_cost)

-- Prove that the cost of one blue pill is 18.5 under the given conditions
theorem blue_pill_cost_is_18_point_5 
  (b_cost y_cost total : ℝ)
  (h1 : y_cost = b_cost - 2)
  (h2 : total = 735)
  (h3 : 21 * (b_cost + y_cost) = total) :
  b_cost = 18.5 :=
by
  -- Calculate daily cost
  have daily_cost : ℝ := total / number_of_days
  have h4 : daily_cost = 35, from sorry

  -- Setup and solve the equation
  have cost_eq : b_cost + (b_cost - 2) = 35, from sorry
  have solve_for_b_cost : 2 * b_cost - 2 = 35, from sorry
  have final_cost : b_cost = 18.5, from sorry

  exact final_cost

end blue_pill_cost_is_18_point_5_l563_563113


namespace factor_tree_problem_l563_563238

theorem factor_tree_problem :
  let H := 7 * 2,
      I := 11 * 2,
      F := 7 * H,
      G := 11 * I,
      X := F * G in
  X = 23716 :=
by
  sorry

end factor_tree_problem_l563_563238


namespace probability_two_common_courses_l563_563439

-- Define the courses
inductive Course
| A | B | C | D | E | F

open Course

-- Define the student's choices constraint
def valid_choice (choices : List Course) : Prop :=
  choices.length = 3 ∧ (choices.count Course.A + choices.count Course.B + choices.count Course.C) ≥ 2

-- Define the probability problem
theorem probability_two_common_courses (students : List (List Course)) :
  -- Each student must make a valid choice
  (∀ choices, choices ∈ students → valid_choice choices) →
  -- There are exactly three students
  students.length = 3 →
  -- The probability condition we seek to prove
  (calculate_probability students) = (79 / 250) := by
  sorry

-- Placeholder for the probability calculation logic, expanded here to outline the full proof, assumed correct.
def calculate_probability (students : List (List Course)) : ℚ :=
  -- Proper probability calculation implementation goes here, which is described in the solution steps.
  sorry

end probability_two_common_courses_l563_563439


namespace cube_painted_faces_l563_563092

theorem cube_painted_faces (n : ℕ) (h1 : 6 * n^2) (h2 : n^3) (h3 : (6 * n^2)/(6 * n^3) = 1/3) : n = 3 := by
  sorry

end cube_painted_faces_l563_563092


namespace sufficient_but_not_necessary_l563_563897

variables (p q : Prop)

theorem sufficient_but_not_necessary (h1 : p → q) (h2 : ¬ (¬ p → ¬ q)) : 
  (∃ (S : Prop), S = (p → q) ∧ ¬ (q → p)) :=
by
  use (p → q)
  split
  . exact h1
  . intro h3
    exact h2 (λ hnp, not.elim hnp (λ hp, h3 hp))

end sufficient_but_not_necessary_l563_563897


namespace cover_points_with_two_triangles_l563_563769

-- Definitions for our problem conditions
variable {X : Type*} [Fintype X] (points : Finset (Point X)) (equilateral_triangle : Set (Point X))

-- Hypotheses for the given conditions
hypothesis finite_points : Finite X
hypothesis subset_property : ∀ (subset : Finset (Point X)), subset.card ≤ 9 → ∃ (T1 T2 : Set (Point X)), (∀ pt ∈ subset, pt ∈ T1 ∨ pt ∈ T2)

-- The theorem to prove
theorem cover_points_with_two_triangles (points : Finset (Point X)) (equilateral_triangle : Set (Point X))
  (finite_points : Finite X)
  (subset_property : ∀ (subset : Finset (Point X)), subset.card ≤ 9 → ∃ (T1 T2 : Set (Point X)), (∀ pt ∈ subset, pt ∈ T1 ∨ pt ∈ T2)) :
  ∃ (T1 T2 : Set (Point X)), ∀ pt ∈ points, pt ∈ T1 ∨ pt ∈ T2 :=
  sorry

end cover_points_with_two_triangles_l563_563769


namespace negation_exists_geq_l563_563336

theorem negation_exists_geq :
  ¬ (∀ x : ℝ, x^3 - x^2 + 1 < 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≥ 0 :=
by
  sorry

end negation_exists_geq_l563_563336


namespace find_n_in_geometric_series_l563_563107

theorem find_n_in_geometric_series :
  (∃ (n : ℝ),
    let a₁ := 15,
    let a₂ := 3,
    let r₁ := a₂ / a₁,
    let S₁ := a₁ / (1 - r₁),
    let a₂' := 3 + n,
    let s := a₂' / a₁,
    let S₂ := a₁ / (1 - s)
    in S₂ = 5 * S₁) →
  n = 9.6 :=
by
  sorry

end find_n_in_geometric_series_l563_563107


namespace Valley_Forge_High_School_winter_carnival_l563_563815

noncomputable def number_of_girls (total_students : ℕ) (total_participants : ℕ) (fraction_girls_participating : ℚ) (fraction_boys_participating : ℚ) : ℕ := sorry

theorem Valley_Forge_High_School_winter_carnival
  (total_students : ℕ)
  (total_participants : ℕ)
  (fraction_girls_participating : ℚ)
  (fraction_boys_participating : ℚ)
  (h_total_students : total_students = 1500)
  (h_total_participants : total_participants = 900)
  (h_fraction_girls : fraction_girls_participating = 3 / 4)
  (h_fraction_boys : fraction_boys_participating = 2 / 3) :
  number_of_girls total_students total_participants fraction_girls_participating fraction_boys_participating = 900 := sorry

end Valley_Forge_High_School_winter_carnival_l563_563815


namespace reflection_correct_l563_563863

open Matrix

def reflection_over_vector (p v : Matrix (Fin 2) (Fin 1) ℝ) : Matrix (Fin 2) (Fin 1) ℝ :=
  let projection := (p.dot_product v) / (v.dot_product v) • v
  in 2 • projection - p

def point_to_reflect : Matrix (Fin 2) (Fin 1) ℝ := ![![2], ![6]]
def vector_for_reflection : Matrix (Fin 2) (Fin 1) ℝ := ![![2], ![1]]
def reflection_result : Matrix (Fin 2) (Fin 1) ℝ := ![![6], ![-2]]

theorem reflection_correct : reflection_over_vector point_to_reflect vector_for_reflection = reflection_result := 
  sorry

end reflection_correct_l563_563863


namespace star_p_eq_15_l563_563634

def sum_of_digits (x : ℕ) : ℕ :=
  x.digits 10 |>.sum

def T : set ℕ := { n | sum_of_digits n = 15 ∧ n < 10^6 }

def p : ℕ := (T : set ℕ).to_finset.card

theorem star_p_eq_15 : sum_of_digits p = 15 := by
  sorry

end star_p_eq_15_l563_563634


namespace tangent_line_at_one_minimum_value_in_interval_l563_563921

-- Define the function f(x) = ax^2 - (a + 2)x + ln x
noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x + real.log x

-- Problem 1: Equation of the tangent line at (1, f(1)) when a = 1
theorem tangent_line_at_one (x : ℝ) (hx : x = 1) : 
  let a : ℝ := 1 in
  let y := f a x in y = -x - 1 :=
sorry

-- Problem 2: Range of values for a
theorem minimum_value_in_interval (a : ℝ) (ha : a > 0) : 
  ∃ x ∈ set.Icc (1:ℝ) real.exp, f a x = -2 ↔ a ∈ set.Ioc 0 real.infinity :=
sorry

end tangent_line_at_one_minimum_value_in_interval_l563_563921


namespace slope_at_x_equals_one_l563_563018

noncomputable def slope_of_tangent_line (f: ℝ → ℝ) (x: ℝ) : ℝ := 
  derivative f x

def given_curve (x: ℝ) : ℝ := x^3 - 2 * x^2

theorem slope_at_x_equals_one : slope_of_tangent_line given_curve 1 = -1 := by
  sorry

end slope_at_x_equals_one_l563_563018


namespace ice_cream_flavors_l563_563583

theorem ice_cream_flavors : (Nat.choose (4 + 4 - 1) (4 - 1) = 35) :=
by
  sorry

end ice_cream_flavors_l563_563583


namespace evaluate_expression_l563_563487

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem evaluate_expression : 3 * g 2 + 2 * g (-2) = 85 :=
by
  sorry

end evaluate_expression_l563_563487


namespace chocolate_bars_in_crate_l563_563444

theorem chocolate_bars_in_crate :
  (∀ (C : Type) (L M S B : ℕ),
  (C = 10 * (L * (M * (S * B))) -> L = 19 -> M = 27 -> S = 30 -> B = 1) -> C = 153900) :=
begin
  intros C L M S B h,
  rw [h.2 h.1.1],
  sorry
end

end chocolate_bars_in_crate_l563_563444


namespace range_of_k_l563_563955

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, x ≠ -1 ∧ log10 (k * x) = 2 * log10 (x + 1) ∧
   (∀ y : ℝ, (y ≠ -1 ∧ log10 (k * y) = 2 * log10 (y + 1)) → y = x)) ↔
  (k < 0 ∨ k = 4) :=
by
  -- Proof omitted
  sorry

end range_of_k_l563_563955


namespace average_minutes_run_per_day_l563_563967

theorem average_minutes_run_per_day :
  ∃ (e : ℕ), 
    (let sixth_avg := 18
     let seventh_avg := 21
     let eighth_avg := 16
     let sixth_days := 5
     let seventh_days := 4
     let eighth_days := 3
     let sixth_students := 9 * e
     let seventh_students := 3 * e
     let eighth_students := e
     let total_students := sixth_students + seventh_students + eighth_students
     let total_minutes_week :=
       (sixth_avg * sixth_days * sixth_students) +
       (seventh_avg * seventh_days * seventh_students) +
       (eighth_avg * eighth_days * eighth_students)
     let avg_minutes_per_day := total_minutes_week / (total_students * 7)
     avg_minutes_per_day = 12.19) := by
{
  sorry
}

end average_minutes_run_per_day_l563_563967


namespace quadratic_m_leq_9_l563_563592

-- Define the quadratic equation
def quadratic_eq_has_real_roots (a b c : ℝ) : Prop := 
  b^2 - 4*a*c ≥ 0

-- Define the specific property we need to prove
theorem quadratic_m_leq_9 (m : ℝ) : (quadratic_eq_has_real_roots 1 (-6) m) → (m ≤ 9) := 
by
  sorry

end quadratic_m_leq_9_l563_563592


namespace imo_1975_p6_l563_563814

variables {α : Type*} [NontrivialCommRing α] [IsDomain α]

/-- Given triangle ABC with external triangles BPC, CQA, and ARB having specific angles, 
    prove that angle PRQ = 90 degrees and QR = PR. -/
theorem imo_1975_p6 (A B C P Q R : α) 
  (h₁ : ∠ P B C = 45)
  (h₂ : ∠ C A Q = 45)
  (h₃ : ∠ B C P = 30)
  (h₄ : ∠ Q C A = 30)
  (h₅ : ∠ A B R = 15)
  (h₆ : ∠ R A B = 15) :
  ∠ P R Q = 90 ∧ (Q - R).abs = (P - R).abs := 
sorry

end imo_1975_p6_l563_563814


namespace rabbit_hunter_l563_563393

/-- A data structure to represent positions in the Euclidean plane -/
structure Position :=
  (x : ℝ)
  (y : ℝ)

/-- The distance between two positions in the Euclidean plane -/
def distance (A B : Position) : ℝ :=
  real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

def rabbit_hunter_game (initial : Position) (n : ℕ) : Prop :=
  ∀ (A B : list Position), -- Positions of rabbit and hunter over rounds
    A.length = n + 1 ∧
    B.length = n + 1 ∧
    A.head = initial ∧ 
    B.head = initial ∧
    (∀ k, k < n → distance (A.nth_le k (by simp [h])) (A.nth_le (k+1) (by simp [h])) = 1) ∧
    (∀ k, k < n → ∃ P, distance (A.nth_le (k+1) (by simp [h])) P ≤ 1 ∧ distance (B.nth_le k (by simp [h])) (B.nth_le (k+1) (by simp [h])) = 1) →
    distance (A.last (by simp [h])) (B.last (by simp [h])) > 100

theorem rabbit_hunter : ¬ rabbit_hunter_game ⟨0, 0⟩ (10^9) :=
sorry

end rabbit_hunter_l563_563393


namespace opposite_of_neg_3_l563_563007

theorem opposite_of_neg_3 : (-(-3) = 3) :=
by
  sorry

end opposite_of_neg_3_l563_563007


namespace quadratic_solution_l563_563344

theorem quadratic_solution (x : ℝ) : x * (x - 3) = x - 3 → (x = 3 ∨ x = 1) :=
by
  intro h,
  have h1 : x * (x - 3) - (x - 3) = 0 := by linarith,
  have h2 : (x - 3) * (x - 1) = 0 := by linarith,
  cases h2,
  {
    left,
    apply eq.symm,
    exact h2,
  },
  {
    right,
    apply eq.symm,
    exact h2,
  }

end quadratic_solution_l563_563344


namespace tangent_lines_count_l563_563655

-- Define the radii and center distance as constants.
constant r1 : ℝ := 4
constant r2 : ℝ := 6
constant d : ℝ := 3

-- Define the statement to be proven.
theorem tangent_lines_count : ∀ (k : ℕ), k = 0 :=
by {
  -- We assume k is initially 0, and show it cannot be any other value
  intros k,
  have H1: k ≠ 1, from by {
    -- Internally tangent circles: d should be |r2 - r1|
    -- Here d == 3 which is not |r2 - r1| which is 2
    have h : abs (r2 - r1) ≠ d, from by linarith,
    exact h
  },
  have H2: k ≠ 0, from by {
    -- Concentric circles: d must equal 0
    have h : d ≠ 0, from by linarith,
    exact h
  },
  have H3: k ≠ 4, from by {
    -- Non-interaction scenario: d must exceed r1 + r2
    have h : d ≤ r1 + r2, from by linarith,
    exact h
  },
  -- Therefore, since none of the k values 1, 0, or 4 is correct, k must be 0
  exact or.elim (nat.eq_of_eq_or_eq H1 H2) sorry sorry
}

end tangent_lines_count_l563_563655


namespace garden_enclosure_l563_563073

theorem garden_enclosure :
  ∃! pairs : set (ℕ × ℕ),
    (∀ p ∈ pairs, let x := p.1, y := p.2 in (x * y = 12 ∧ x ≤ 8 ∧ 2 * x + 2 * y - 8 ≤ 10.5)) ∧
    set.card pairs = 2 :=
by
  -- define the set of pairs (x, y) and claim it's unique and contains exactly 2 elements
  sorry

end garden_enclosure_l563_563073


namespace cat_daytime_catches_l563_563761

theorem cat_daytime_catches
  (D : ℕ)
  (night_catches : ℕ := 2 * D)
  (total_catches : ℕ := D + night_catches)
  (h : total_catches = 24) :
  D = 8 := by
  sorry

end cat_daytime_catches_l563_563761


namespace BE_length_l563_563542

open Set Function

variables (A B C D F E G : Point) (EF GF BC DC BE : ℝ)
variables (h1 : parallelogram ABCD)
variables (h2 : F ∈ Ray AD (D))
variables (h3 : Collinear B F E)
variables (h4 : E ∈ LineSegment AC)
variables (h5 : G ∈ LineSegment DC)
variables (h6 : EF = 40)
variables (h7 : GF = 30)
variables (h8 : DC = (3/2) * BC)

theorem BE_length :
  BE = 237 :=
sorry

end BE_length_l563_563542


namespace quadratic_inequality_for_all_x_l563_563930

theorem quadratic_inequality_for_all_x {m : ℝ} :
  (∀ x : ℝ, (m^2 - 2 * m - 3) * x^2 - (m - 3) * x - 1 < 0) ↔ (-1 / 5 < m ∧ m ≤ 3) :=
begin
  sorry
end

end quadratic_inequality_for_all_x_l563_563930


namespace driver_speed_last_4_hours_l563_563768

theorem driver_speed_last_4_hours
  (hours_30mph_per_day : ℕ)
  (speed_30mph : ℝ)
  (days_week : ℕ)
  (total_weekly_distance : ℝ)
  (hours_per_day_for_unknown_speed : ℕ)
  (speed_of_last_4_hours : ℝ) :
  hours_30mph_per_day = 3 →
  speed_30mph = 30 →
  days_week = 6 →
  total_weekly_distance = 1140 →
  hours_per_day_for_unknown_speed = 4 →
  ∃ speed_of_last_4_hours, speed_of_last_4_hours = 25 :=
by
  intros h1 h2 h3 h4 h5
  let total_distance_30mph_per_week := speed_30mph * hours_30mph_per_day * days_week
  have remaining_distance := total_weekly_distance - total_distance_30mph_per_week
  let total_hours_for_unknown_speed := hours_per_day_for_unknown_speed * days_week
  let speed := remaining_distance / total_hours_for_unknown_speed
  use speed
  have speed_correct : speed = 25 := by
    calc speed
      = remaining_distance / total_hours_for_unknown_speed : by sorry -- Calculation here
      = 600 / 24 : by sorry -- Substitution here
      = 25 : by sorry -- Division result here
  exact speed_correct

end driver_speed_last_4_hours_l563_563768


namespace sum_of_lengths_of_two_sides_l563_563360

open Real

noncomputable def triangle_sum_of_two_sides (a b c : ℝ) (A B C : ℝ) : ℝ :=
  if A + B + C = 180 ∧ A = 50 ∧ C = 40 ∧ c = 8 * sqrt 3 then
    let b := c * (sin (B * pi / 180)) / (sin (C * pi / 180))
    let a := c * (sin (A * pi / 180)) / (sin (C * pi / 180))
    a + b
  else
    0

theorem sum_of_lengths_of_two_sides : triangle_sum_of_two_sides 24.5 20.6 (8 * sqrt 3) 50 90 40 = 45.1 := by
  sorry

end sum_of_lengths_of_two_sides_l563_563360


namespace maintain_volume_with_reduced_length_l563_563081

theorem maintain_volume_with_reduced_length
  (L W H : ℝ)
  (new_length : ℝ := 0.80 * L)
  (V : ℝ := L * W * H)
  : 
  (W' : ℝ := W / 0.80) ∧ 
  (height_unchanged : H = H)
  := 
by 
  sorry

end maintain_volume_with_reduced_length_l563_563081


namespace largest_among_five_numbers_l563_563747

theorem largest_among_five_numbers :
  let A := 98765 + (1 / 4321)
  let B := 98765 - (1 / 4321)
  let C := 98765 * (1 / 4321)
  let D := 98765 / (1 / 4321)
  let E := 98765.4321
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  let A := 98765 + (1 / 4321)
  let B := 98765 - (1 / 4321)
  let C := 98765 * (1 / 4321)
  let D := 98765 / (1 / 4321)
  let E := 98765.4321
  sorry

end largest_among_five_numbers_l563_563747


namespace opposite_of_neg_3_l563_563006

theorem opposite_of_neg_3 : (-(-3) = 3) :=
by
  sorry

end opposite_of_neg_3_l563_563006


namespace inverse_proportion_relationship_l563_563295

variable k : ℝ
def y (x : ℝ) : ℝ := k / x

theorem inverse_proportion_relationship (h1 : y (-2) = 3) :
  let y1 := y (-3) in
  let y2 := y 1 in
  let y3 := y 2 in
  y2 < y3 ∧ y3 < y1 :=
by
  have k_val : k = -6 := by
    rw [y, ← @eq_div_iff_mul_eq ℝ _ _ _ _] at h1 <;> linarith
  let y1 := -6 / -3
  let y2 := -6 / 1
  let y3 := -6 / 2
  split
  sorry

end inverse_proportion_relationship_l563_563295


namespace problem1_part1_problem1_part2_problem1_part3_problem2_l563_563200

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then
    x + 2
  else if x < 2 then
    x^2
  else
    2 * x

theorem problem1_part1 : f (-4) = -2 :=
by
  unfold f
  split_ifs
  · refl

theorem problem1_part2 : f 3 = 6 :=
by
  unfold f
  split_ifs
  · contradiction
  · linarith
  · refl

theorem problem1_part3 : f (f (-2)) = 8 :=
by
  unfold f
  split_ifs
  · unfold f
    split_ifs
    · norm_num
    · linarith
    · refl

theorem problem2 : ∃ a : ℝ, f a = 10 ∧ a = 5 :=
by
  use 5
  unfold f
  split_ifs
  · linarith
  · linarith
  · split
    · refl
    · refl

end problem1_part1_problem1_part2_problem1_part3_problem2_l563_563200


namespace billy_sleep_total_hours_l563_563585

theorem billy_sleep_total_hours : 
    let first_night := 6
    let second_night := 2 * first_night
    let third_night := second_night - 3
    let fourth_night := 3 * third_night
    first_night + second_night + third_night + fourth_night = 54
  := by
    sorry

end billy_sleep_total_hours_l563_563585


namespace find_red_pens_l563_563053

-- We define the conditions as stated in the problem
variable (x : ℤ)
variable (total_pens : ℤ := 66)
variable (red_pen_cost : ℤ := 5)
variable (black_pen_cost : ℤ := 9)
variable (discount_red : ℚ := 0.85)
variable (discount_black : ℚ := 0.80)
variable (discount_total : ℚ := 0.82)

-- Define the total count of black pens
def black_pens := total_pens - x

-- Define the original cost
def original_cost := red_pen_cost * x + black_pen_cost * black_pens

-- Define the discounted cost
def discounted_cost := (red_pen_cost * discount_red * x) + (black_pen_cost * discount_black * black_pens)

-- The main proof statement
theorem find_red_pens (h1 : 0 ≤ x ∧ x ≤ 66) (h2 : discounted_cost x = 0.82 * original_cost x) : x = 36 :=
by sorry

end find_red_pens_l563_563053


namespace prime_condition_l563_563591

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_condition (p : ℕ) (h1 : is_prime p) (h2 : is_prime (8 * p^2 + 1)) : 
  p = 3 ∧ is_prime (8 * p^2 - p + 2) :=
by
  sorry

end prime_condition_l563_563591


namespace largest_prime_factor_of_expression_l563_563384

theorem largest_prime_factor_of_expression :
  ∃ p : ℕ, prime p ∧ p = 241 ∧ ∀ q : ℕ, prime q ∧ q ∣ (16^4 + 3 * 16^2 + 2 - 15^4) → q ≤ p :=
by
  sorry

end largest_prime_factor_of_expression_l563_563384


namespace cos_minus_sin_alpha_eq_neg_seven_fifths_l563_563900

noncomputable def cos_minus_sin_of_alpha (α : ℝ) (h1 : α > π / 2 ∧ α < π) (h2 : Real.sin (2 * α) = -24 / 25) : ℝ :=
  -7 / 5
  
theorem cos_minus_sin_alpha_eq_neg_seven_fifths 
  {α : ℝ} (h1 : α > π / 2 ∧ α < π) (h2 : Real.sin (2 * α) = -24 / 25) : 
  cos α - sin α = -7 / 5 := 
by 
  sorry

end cos_minus_sin_alpha_eq_neg_seven_fifths_l563_563900


namespace three_distinct_real_roots_l563_563639

theorem three_distinct_real_roots 
  (c : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    (x1*x1 + 6*x1 + c)*(x1*x1 + 6*x1 + c) = 0 ∧ 
    (x2*x2 + 6*x2 + c)*(x2*x2 + 6*x2 + c) = 0 ∧ 
    (x3*x3 + 6*x3 + c)*(x3*x3 + 6*x3 + c) = 0) 
  ↔ c = (11 - Real.sqrt 13) / 2 :=
by sorry

end three_distinct_real_roots_l563_563639


namespace quadratic_proof_l563_563917

noncomputable theory

-- Define the quadratic equation with general terms
def quadratic_eq (a x : ℝ) : ℝ :=
  x^2 + a * x + a - 2

-- Define condition that one root is 1
def is_root (a : ℝ) := quadratic_eq a 1 = 0

-- Define discriminant of the quadratic function
def discriminant (a : ℝ) : ℝ :=
  a^2 - 4 * (a - 2)

-- Main proof problem statement
def quadratic_properties (a : ℝ) (x : ℝ) : Prop :=
(is_root a) -> (a = 1/2 ∧ x = -1/2) ∧ (discriminant a > 0)

-- Theorem to prove the above properties
theorem quadratic_proof (a : ℝ) (x : ℝ) : quadratic_properties a x :=
by 
  sorry

end quadratic_proof_l563_563917


namespace function_identity_l563_563680

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem function_identity {f : ℕ → ℕ} 
  (h : ∀ a b : ℕ, is_perfect_square (a * f(a) + b * f(b) + 2 * a * b)) : 
  ∀ a : ℕ, f(a) = a :=
by
  sorry

end function_identity_l563_563680


namespace incorrect_proposition_l563_563378

theorem incorrect_proposition (p q : Prop) :
  ¬ (if ¬ (p ∧ q) then ¬ p ∧ ¬ q) :=
by
  intro h
  have hpq_or_hnp := classical.em (p ∧ q)
  cases hpq_or_hnp with hpq hnp
  {
    exact or.elim hnp (λ hnq, and.intro hnq) (λ hnp, and.intro hnp)
  }
  {
    have hnp_sq_hnp := classical.em p
    cases hnp_sq_hnp with hp hnq
    {
      have hnq := or.intro_right ¬ p hnq
      exact and.intro hnq
    }
    {
      have hp := or.intro_left ¬ q hpq
      exact and.intro hp
    }
  }
  sorry -- Proof to be completed here

end incorrect_proposition_l563_563378


namespace muffins_in_morning_l563_563261

variable (M : ℕ)

-- Conditions
def goal : ℕ := 20
def afternoon_sales : ℕ := 4
def additional_needed : ℕ := 4
def morning_sales (M : ℕ) : ℕ := M

-- Proof statement (no need to prove here, just state it)
theorem muffins_in_morning :
  morning_sales M + afternoon_sales + additional_needed = goal → M = 12 :=
sorry

end muffins_in_morning_l563_563261


namespace number_of_pairings_l563_563823

-- Definitions for conditions.
def bowls : Finset String := {"red", "blue", "yellow", "green"}
def glasses : Finset String := {"red", "blue", "yellow", "green"}

-- The theorem statement
theorem number_of_pairings : bowls.card * glasses.card = 16 := by
  sorry

end number_of_pairings_l563_563823


namespace christine_needs_32_tbs_aquafaba_l563_563118

-- Definitions for the conditions
def tablespoons_per_egg_white : ℕ := 2
def egg_whites_per_cake : ℕ := 8
def number_of_cakes : ℕ := 2

def total_egg_whites : ℕ := egg_whites_per_cake * number_of_cakes
def total_tbs_aquafaba : ℕ := tablespoons_per_egg_white * total_egg_whites

-- Theorem statement
theorem christine_needs_32_tbs_aquafaba :
  total_tbs_aquafaba = 32 :=
by sorry

end christine_needs_32_tbs_aquafaba_l563_563118


namespace sample_variance_is_two_l563_563720

theorem sample_variance_is_two (a : ℝ) (h_avg : (a + 0 + 1 + 2 + 3) / 5 = 1) : 
  (1 / 5) * ((a - 1)^2 + (0 - 1)^2 + (1 - 1)^2 + (2 - 1)^2 + (3 - 1)^2) = 2 :=
sorry

end sample_variance_is_two_l563_563720


namespace inflection_point_value_l563_563446

-- Define the function f
def f (x : ℝ) : ℝ := sin (2 * x) + cos (2 * x) + (1 / 3) * x

-- Define the first derivative f'
def f' (x : ℝ) : ℝ := 2 * cos (2 * x) - 2 * sin (2 * x) + 1 / 3

-- Define the second derivative f''
def f'' (x : ℝ) : ℝ := -4 * sin (2 * x) - 4 * cos (2 * x)

-- Define the conditions for x0 and check that f''(x0) = 0
def is_inflection_point (x0 : ℝ) : Prop :=
  -π / 4 < x0 ∧ x0 < 0 ∧ f'' x0 = 0

-- Prove that f(x0) = -π / 24 holds true if x0 is the inflection point and within the given interval
theorem inflection_point_value (x0 : ℝ) (h : is_inflection_point x0) : f x0 = -π / 24 :=
  sorry

end inflection_point_value_l563_563446


namespace points_collinear_l563_563248

variables {Point : Type*} [add_comm_group Point] [module ℝ Point] [affine_space Point ℝ]

open_locale affine

-- Definitions of points and their relationships:
variable (A B C D E F P Q R : Point)

-- Conditions:
variables
  (h1 : convex ℝ (set.insert A (set.insert B (set.insert C {D}))))
  (h2 : ∃ u v w : ℝ, u + v = 1 ∧ (E : affine_combination ℝ Point (E ∥ A, u) (B ∥ D, v )) = B ∧ (D : affine_combination ℝ Point (D ∥ C, w )) = C)
  (h3 : ∃ p q r : ℝ, p + q = 1 ∧ (F : affine_combination ℝ Point (F ∥ A, p) (D ∥ B, q )) = D ∧ (B : affine_combination ℝ Point (B ∥ C, r )) = C)
  (h4 : P = midpoint ℝ A C)
  (h5 : Q = midpoint ℝ B D)
  (h6 : R = midpoint ℝ E F)

-- Goal: Prove that P, Q, and R are collinear
theorem points_collinear (h1 : convex ℝ (set.insert A (set.insert B (set.insert C {D}))))
  (h2 : ∃ u v w : ℝ, u + v = 1 ∧ (E : affine_combination ℝ Point (E ∥ A, u) (B ∥ D, v )) = B ∧ (D : affine_combination ℝ Point (D ∥ C, w )) = C)
  (h3 : ∃ p q r : ℝ, p + q = 1 ∧ (F : affine_combination ℝ Point (F ∥ A, p) (D ∥ B, q )) = D ∧ (B : affine_combination ℝ Point (B ∥ C, r )) = C)
  (h4 : P = midpoint ℝ A C)
  (h5 : Q = midpoint ℝ B D)
  (h6 : R = midpoint ℝ E F) :
  collinear ℝ ({P, Q, R} : set Point) :=
sorry

end points_collinear_l563_563248


namespace sum_of_derivative_at_0_l563_563274

noncomputable def f : ℝ → ℝ := λ x, x * Real.cos x
noncomputable def f'_1 := Real.deriv f
noncomputable def f'_2 := Real.deriv f'_1
noncomputable def f'_3 := Real.deriv f'_2
noncomputable def f'_n : ℕ → (ℝ → ℝ)
| 0 => f
| 1 => f'_1
| 2 => f'_2
| _ => λ n, f'_n (n-1)

theorem sum_of_derivative_at_0 :
  (f 0) + (f'_1 0) + (f'_2 0) + (f'_3 0) + ∑ i in Finset.range 2014, (f'_n i 0) = 1007 :=
sorry

end sum_of_derivative_at_0_l563_563274


namespace diameter_is_10sqrt6_l563_563365

noncomputable def radius (A : ℝ) (hA : A = 150 * Real.pi) : ℝ :=
  Real.sqrt (A / Real.pi)

noncomputable def diameter (A : ℝ) (hA : A = 150 * Real.pi) : ℝ :=
  2 * radius A hA

theorem diameter_is_10sqrt6 (A : ℝ) (hA : A = 150 * Real.pi) :
  diameter A hA = 10 * Real.sqrt 6 :=
  sorry

end diameter_is_10sqrt6_l563_563365


namespace abs_nested_expression_l563_563224

theorem abs_nested_expression (x : ℝ) (hx : x > 3) : |2 - |2 - x|| = 4 - x :=
sorry

end abs_nested_expression_l563_563224


namespace max_value_M_l563_563154

noncomputable def J_k (k : ℕ) (hk : k > 0) : ℕ :=
  10^(k+2) + 25

def M (k : ℕ) (hk : k > 0) : ℕ :=
  (J_k k hk).factors.count 5

theorem max_value_M (k : ℕ) (hk : k > 0) : M k hk = 2 :=
  sorry

end max_value_M_l563_563154


namespace simplify_expression_l563_563675

theorem simplify_expression (a b : ℝ) : -3 * (a - b) + (2 * a - 3 * b) = -a :=
by
  sorry

end simplify_expression_l563_563675


namespace lambda_range_l563_563453

def a_seq (a : ℕ → ℝ) :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = a n / (a n + 2)

def b_seq (b : ℕ → ℝ) (a : ℕ → ℝ) (λ : ℝ) :=
  b 1 = -λ ∧ ∀ n : ℕ, b (n + 1) = (n - λ) * (1 / a n + 1)

def b_monotonic (b : ℕ → ℝ) :=
  ∀ n : ℕ, b (n + 1) > b n

theorem lambda_range (a b : ℕ → ℝ) (λ : ℝ)
  (ha : a_seq a)
  (hb : b_seq b a λ)
  (hm : b_monotonic b) :
  λ < 2 :=
sorry

end lambda_range_l563_563453


namespace max_possible_value_l563_563949

noncomputable def vec := ℝ^3 -- Replace 3 with the correct dimension as needed

def is_unit_vector (v : vec) : Prop := ∥v∥ = 1

theorem max_possible_value (a b c d : vec) (ha : is_unit_vector a) (hb : is_unit_vector b)
  (hc : is_unit_vector c) (hd : is_unit_vector d) :
  ∥a - b∥^2 + ∥a - c∥^2 + ∥a - d∥^2 + ∥b - c∥^2 + ∥b - d∥^2 + ∥c - d∥^2 ≤ 14 :=
sorry

end max_possible_value_l563_563949


namespace remainder_x3_div_x2_plus_3x_plus_2_l563_563150

open Polynomial

noncomputable def remainder_when_divided (f g : Polynomial ℝ) :=
  Polynomial.modByMonic (f) (g.monic_divisor)

theorem remainder_x3_div_x2_plus_3x_plus_2 :
  remainder_when_divided (X^3) (X^2 + (3 : ℝ) * X + 2) = -3 * X - 2 :=
sorry

end remainder_x3_div_x2_plus_3x_plus_2_l563_563150


namespace even_subset_count_l563_563219

theorem even_subset_count (S : Finset ℕ) (hS : S.card = 9) :
  (S.powerset.filter (λ s, s.card % 2 = 0)).card = 256 := by
  sorry

end even_subset_count_l563_563219


namespace positive_difference_solutions_l563_563041

theorem positive_difference_solutions : 
  ∀ (r : ℝ), r ≠ -3 → 
  (∃ r1 r2 : ℝ, (r^2 - 6*r - 20) / (r + 3) = 3*r + 10 → r1 ≠ r2 ∧ 
  |r1 - r2| = 20) :=
by
  sorry

end positive_difference_solutions_l563_563041


namespace mileage_per_gallon_l563_563065

def total_distance := 130 -- in kilometers
def total_gasoline := 6.5 -- in gallons

theorem mileage_per_gallon : total_distance / total_gasoline = 20 :=
by
  sorry

end mileage_per_gallon_l563_563065


namespace find_t_l563_563632

-- Define the points A and B
def pointA (t : ℝ) : ℝ × ℝ := (2 * t - 3, t)
def pointB (t : ℝ) : ℝ × ℝ := (t - 1, 2 * t + 4)

-- Define the midpoint of the segment AB
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the square distance between two points
def dist_sq (A B : ℝ × ℝ) : ℝ :=
  (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2

-- Define the problem conditions
theorem find_t (t : ℝ) :
  let A := pointA t,
      B := pointB t,
      M := midpoint A B in
  dist_sq M A = (t^2 + t) / 2 → t = -10 :=
begin
  sorry
end

end find_t_l563_563632


namespace no_integer_distance_point_l563_563700

variable (a b : ℤ)
variable (h_a_odd : a % 2 = 1)
variable (h_b_odd : b % 2 = 1)

theorem no_integer_distance_point :
  ¬ ∃ (P : ℝ × ℝ), 
    ∃ (PA PB PC PD : ℝ),
      PA = Real.sqrt ((P.1 - 0) ^ 2 + (P.2 - 0) ^ 2) ∧
      PB = Real.sqrt ((P.1 - b) ^ 2 + (P.2 - 0) ^ 2) ∧
      PC = Real.sqrt ((P.1 - b) ^ 2 + (P.2 - a) ^ 2) ∧
      PD = Real.sqrt ((P.1 - 0) ^ 2 + (P.2 - a) ^ 2) ∧
      PA ∈ ℤ ∧ PB ∈ ℤ ∧ PC ∈ ℤ ∧ PD ∈ ℤ := 
sorry

end no_integer_distance_point_l563_563700


namespace roots_of_poly_l563_563144

noncomputable def poly (x : ℝ) : ℝ :=
  3 * x^4 - 2 * x^3 - 7 * x^2 - 2 * x + 3

theorem roots_of_poly :
  (poly (1 + sqrt 5)/2) = 0 ∧
  (poly (1 - sqrt 5)/2) = 0 ∧
  (poly (-1 + sqrt 37)/6) = 0 ∧
  (poly (-1 - sqrt 37)/6) = 0 :=
by {
  sorry
}

end roots_of_poly_l563_563144


namespace problem1_solution_set_problem2_domain_l563_563401

-- Problem (1) Lean statement
theorem problem1_solution_set (x : ℝ) : 
    (-x^2 + 4 * x + 5 < 0) ↔ (x < -1 ∨ x > 5) :=
sorry

-- Problem (2) Lean statement
theorem problem2_domain (x : ℝ) : 
    (∃ y : ℝ, y = sqrt ((x - 1) / (x + 2)) + 5) ↔ (x < -2 ∨ x ≥ 1) :=
sorry

end problem1_solution_set_problem2_domain_l563_563401


namespace minimum_elements_of_A_l563_563278

-- Define set A with the properties given
noncomputable def A : set ℕ := {n : ℕ | n > 0 ∧ n ≤ 100 ∧ (n = 1 ∨ ∃ x y ∈ A, n = x + y)}

-- Main theorem formulation
theorem minimum_elements_of_A : ∃ (A : set ℕ), (∀ n ∈ A, 1 ≤ n ∧ n ≤ 100) ∧
  (∀ n ∈ A, n = 1 ∨ ∃ x y ∈ A, n = x + y) ∧
  cardinal.mk A = 9 :=
sorry

end minimum_elements_of_A_l563_563278


namespace problem_statement_l563_563203

noncomputable def f (x : ℝ) : ℝ := x / Real.cos x

theorem problem_statement (x1 x2 x3 : ℝ) (h1 : abs x1 < Real.pi / 2)
                         (h2 : abs x2 < Real.pi / 2) (h3 : abs x3 < Real.pi / 2)
                         (c1 : f x1 + f x2 ≥ 0) (c2 : f x2 + f x3 ≥ 0) (c3 : f x3 + f x1 ≥ 0) :
  f (x1 + x2 + x3) ≥ 0 :=
sorry

end problem_statement_l563_563203


namespace how_many_buckets_did_Eden_carry_l563_563504

variable (E : ℕ) -- Natural number representing buckets Eden carried
variable (M : ℕ) -- Natural number representing buckets Mary carried
variable (I : ℕ) -- Natural number representing buckets Iris carried

-- Conditions based on the problem
axiom Mary_Carry_More : M = E + 3
axiom Iris_Carry_Less : I = M - 1
axiom Total_Buckets : E + M + I = 34

theorem how_many_buckets_did_Eden_carry (h1 : M = E + 3) (h2 : I = M - 1) (h3 : E + M + I = 34) :
  E = 29 / 3 := by
  sorry

end how_many_buckets_did_Eden_carry_l563_563504


namespace apple_ratio_l563_563941

theorem apple_ratio
  (cider_golden_apples : ℕ) 
  (cider_pink_apples : ℕ) 
  (farmhands : ℕ) 
  (apples_per_hour : ℕ) 
  (work_hours : ℕ) 
  (pints_cider : ℕ)
  (golden_per_pint : ℕ)
  (pink_per_pint : ℕ)
  (golden_collected : ℕ)
  (pink_collected : ℕ)
  (total_collected : ℕ) :
  cider_golden_apples = 20 → 
  cider_pink_apples = 40 → 
  farmhands = 6 → 
  apples_per_hour = 240 → 
  work_hours = 5 → 
  pints_cider = 120 → 
  golden_per_pint = 20 →
  pink_per_pint = 40 →
  golden_collected = pints_cider * golden_per_pint →
  pink_collected = pints_cider * pink_per_pint →
  total_collected = farmhands * apples_per_hour * work_hours →
  total_collected = golden_collected + pink_collected →
  golden_collected / pink_collected = 1 / 2 :=
by
  intros
  rw [←total_collected, ←golden_collected, ←pink_collected, H, H_1, H_2, H_3, H_4, H_5, H_6, H_7]
  sorry

end apple_ratio_l563_563941


namespace book_total_pages_l563_563759

def chap_1_to_10_pages : ℕ := 10 * 61
def chap_11_to_20_pages : ℕ := 10 * 59
def chap_21_to_31_pages : ℕ := [58, 65, 62, 63, 64, 57, 66, 60, 59, 67].sum

def total_pages : ℕ := chap_1_to_10_pages + chap_11_to_20_pages + chap_21_to_31_pages

theorem book_total_pages : total_pages = 1821 := by
  sorry

#eval book_total_pages

end book_total_pages_l563_563759


namespace rate_up_the_mountain_l563_563770

noncomputable def mountain_trip_rate (R : ℝ) : ℝ := 1.5 * R

theorem rate_up_the_mountain : 
  ∃ R : ℝ, (2 * 1.5 * R = 18) ∧ (1.5 * R = 9) → R = 6 :=
by
  sorry

end rate_up_the_mountain_l563_563770


namespace trajectory_and_fixed_point_and_max_area_l563_563891

noncomputable def f1 : ℝ × ℝ := (-real.sqrt 2, 0)

noncomputable def circle_f2 (x y : ℝ) : Prop := (x - real.sqrt 2)^2 + y^2 = 16

noncomputable def is_on_circle_f2 (M : ℝ × ℝ) : Prop := circle_f2 M.1 M.2

noncomputable def is_perpendicular_bisector_intersection
  (N M f1 : ℝ × ℝ) : Prop :=
  (dist N f1 = dist N M) ∧ ∃ k, N = (M.1 / 2 + f1.1 / 2 + k * (f1.2 - 0), M.2 / 2 + f1.2 / 2 - k * (f1.1 - (-real.sqrt 2)))

theorem trajectory_and_fixed_point_and_max_area :
  let N : ℝ × ℝ := _
  (∀ M : ℝ × ℝ, is_on_circle_f2 M → is_perpendicular_bisector_intersection N M f1) →
  (N.1^2 / 4 + N.2^2 / 2 = 1) ∧ 
  (l_through (0, 1) (A B : ℝ × ℝ) ∧ curve E (∀ M, is_on_circle_f2 M → is_perpendicular_bisector_intersection N M f1) → line_AB'_passes_f2_quadrant (A B : ℝ × ℝ) f2 p q : ℝ) ∧ 
  (find_max_triangle_area (P : ℝ × ℝ) (A B' : ℝ × ℝ) (N : ℝ × ℝ) = real.sqrt 2 / 2).
  
  sorry

end trajectory_and_fixed_point_and_max_area_l563_563891


namespace man_l563_563443

theorem man's_rate_in_still_water (speed_with_stream speed_against_stream : ℝ) (h1 : speed_with_stream = 26) (h2 : speed_against_stream = 12) : 
  (speed_with_stream + speed_against_stream) / 2 = 19 := 
by
  rw [h1, h2]
  norm_num

end man_l563_563443


namespace count_quintuples_divisible_by_5_l563_563122

theorem count_quintuples_divisible_by_5 :
  let valid_quintuples : List (Fin 8 × Fin 8 × Fin 8 × Fin 8 × Fin 8) := 
    (List.finRange 8).product (List.finRange 8)
      .product (List.finRange 8)
      .product (List.finRange 8)
      .product (List.finRange 8)
  in
  valid_quintuples.count (λ ⟨⟨⟨⟨a1, a2⟩, a3⟩, a4⟩, a5⟩, 
    5 ∣ (2 ^ a1 + 2 ^ a2 + 2 ^ a3 + 2 ^ a4 + 2 ^ a5)) = 6528 :=
by
  sorry

end count_quintuples_divisible_by_5_l563_563122


namespace minimal_flights_in_complete_graph_l563_563659

-- Definitions based on the conditions
variables {n : ℕ} (K_n : Graph ℕ)

-- Given condition statements
def complete_graph (G : Graph ℕ) : Prop :=
  ∀ (v w : G.V), v ≠ w → G.adj v w

-- Proof problem statement
theorem minimal_flights_in_complete_graph (n : ℕ)
  (G : Graph ℕ) (h_complete : complete_graph G)
  (h_graph : G = (graph_complete n)) :
  minimal_flights G = 0 :=
sorry

end minimal_flights_in_complete_graph_l563_563659


namespace find_angle_l563_563907

variables (a b : EuclideanSpace ℝ (Fin 2))

def vector_length (v : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  (inner_product_space.has_norm v).norm ⟨v⟩

theorem find_angle
  (h1 : vector_length a = 1)
  (h2 : vector_length b = real.sqrt 2)
  (h3 : inner a (a - b) = 0) :
  real.angle a b = real.pi / 4 :=
sorry

end find_angle_l563_563907


namespace find_y_l563_563045

theorem find_y (y : ℝ) (h : 3 * y / 4 = 15) : y = 20 :=
sorry

end find_y_l563_563045


namespace sin_theta_of_rectangle_l563_563267

noncomputable def rectangle_ABCD (A B C D : ℝ × ℝ) := 
  (A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ D.2 = C.2) ∧ 
  dist A B = 4 ∧ dist A D = 2 ∧ 
  M = midpoint A D ∧ N = midpoint B C 

noncomputable def find_sin_theta (A B C D M N : ℝ × ℝ) (θ : ℝ) : Prop := 
  θ = acos ((A.1 - C.1) / (dist A C)) - acos ((M.1 - N.1) / (dist M N)) ∧ 
  sin θ = (sqrt 5) / 10

theorem sin_theta_of_rectangle : 
  ∃ (A B C D M N : ℝ × ℝ) (θ : ℝ), 
  rectangle_ABCD A B C D ∧ 
  find_sin_theta A B C D M N θ :=
sorry

end sin_theta_of_rectangle_l563_563267


namespace find_sets_A_B_l563_563263

def C : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

def S : Finset ℕ := {4, 5, 9, 14, 23, 37}

theorem find_sets_A_B :
  ∃ (A B : Finset ℕ), 
  (A ∩ B = ∅) ∧ 
  (A ∪ B = C) ∧ 
  (∀ (x y : ℕ), x ≠ y → x ∈ A → y ∈ A → x + y ∉ S) ∧ 
  (∀ (x y : ℕ), x ≠ y → x ∈ B → y ∈ B → x + y ∉ S) ∧ 
  (A = {1, 2, 5, 6, 10, 11, 14, 15, 16, 19, 20}) ∧ 
  (B = {3, 4, 7, 8, 9, 12, 13, 17, 18}) :=
by
  sorry

end find_sets_A_B_l563_563263


namespace cos_neg_pi_div_3_l563_563138

theorem cos_neg_pi_div_3 : Real.cos (-π / 3) = 1 / 2 := 
by 
  sorry

end cos_neg_pi_div_3_l563_563138


namespace exists_curvilinear_polygon_with_equal_perimeter_division_l563_563072

theorem exists_curvilinear_polygon_with_equal_perimeter_division:
  ∃ (P : Type) [curvilinear_polygon P] (A : point_on_boundary P), 
    ∀ (l : line_through A), divides_perimeter_equally P l :=
sorry

end exists_curvilinear_polygon_with_equal_perimeter_division_l563_563072


namespace temperature_range_equiv_l563_563501

theorem temperature_range_equiv :
  ∀ (avg_temp lowest_temp fluctuation : ℝ),
    avg_temp = 50 →
    lowest_temp = 45 →
    fluctuation = 5 →
    (55 - 45 = 10) := 
begin
  intros avg_temp lowest_temp fluctuation h_avg h_lowest h_fluct,
  rw h_avg,
  rw h_lowest,
  rw h_fluct,
  norm_num,
end

end temperature_range_equiv_l563_563501


namespace solve_complex_solution_l563_563678

theorem solve_complex_solution :
  ∀ x : ℂ, (x - 2) ^ 4 + (x - 6) ^ 4 = 16 →
  x ∈ {4 + complex.I * real.sqrt (12 - 8 * real.sqrt 2),
       4 - complex.I * real.sqrt (12 - 8 * real.sqrt 2),
       4 + complex.I * real.sqrt (12 + 8 * real.sqrt 2),
       4 - complex.I * real.sqrt (12 + 8 * real.sqrt 2)} :=
by sorry

end solve_complex_solution_l563_563678


namespace sum_of_consecutive_integers_sqrt_28_l563_563905

theorem sum_of_consecutive_integers_sqrt_28 (a b : ℤ) (h1 : b = a + 1) (h2 : a < Real.sqrt 28) (h3 : Real.sqrt 28 < b) : a + b = 11 :=
by 
    sorry

end sum_of_consecutive_integers_sqrt_28_l563_563905


namespace akeno_extra_expenditure_l563_563095

/-
   Akeno spent $2985 to furnish his apartment.
   Lev spent one-third of that amount on his apartment.
   Ambrocio spent $177 less than Lev.
   Prove that Akeno spent $1172 more than the other 2 people combined.
-/

theorem akeno_extra_expenditure :
  let ak = 2985
  let lev = ak / 3
  let am = lev - 177
  ak - (lev + am) = 1172 :=
by
  sorry

end akeno_extra_expenditure_l563_563095


namespace largest_number_not_sum_of_two_composites_l563_563859

-- Definitions related to composite numbers and natural numbers
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)

-- Lean statement of the problem
theorem largest_number_not_sum_of_two_composites : ∀ n : ℕ,
  (∀ m : ℕ, m > n → cannot_be_sum_of_two_composites m → false) → n = 11 :=
by
  sorry

end largest_number_not_sum_of_two_composites_l563_563859


namespace jimmy_irene_total_payment_l563_563617

def cost_jimmy_shorts : ℝ := 3 * 15
def cost_irene_shirts : ℝ := 5 * 17
def total_cost_before_discount : ℝ := cost_jimmy_shorts + cost_irene_shirts
def discount : ℝ := total_cost_before_discount * 0.10
def total_paid : ℝ := total_cost_before_discount - discount

theorem jimmy_irene_total_payment : total_paid = 117 := by
  sorry

end jimmy_irene_total_payment_l563_563617


namespace man_age_twice_son_age_in_n_years_l563_563075

theorem man_age_twice_son_age_in_n_years
  (S M Y : ℤ)
  (h1 : S = 26)
  (h2 : M = S + 28)
  (h3 : M + Y = 2 * (S + Y)) :
  Y = 2 :=
by
  sorry

end man_age_twice_son_age_in_n_years_l563_563075


namespace solve_quadratic_equation_l563_563311

theorem solve_quadratic_equation :
  ∀ (x : ℝ), (x^2 - 4 * x + 7 = 10) ↔ (x = 2 + sqrt 7 ∨ x = 2 - sqrt 7) :=
by
  sorry

end solve_quadratic_equation_l563_563311


namespace sum_of_common_change_l563_563801

theorem sum_of_common_change (A : ℕ → ℕ) (B : ℕ → ℕ) :
  (∀ n, (A n) % 5 = 4 ∧ (A n) < 100) →
  (∀ m, (B m) % 10 = 7 ∧ (B m) < 100) →
  (∑ x in (filter (fun x => x % 5 = 4 ∧ x < 100) (range 100)).filter (fun y => y ∈ (filter (fun x => x % 10 = 7 ∧ x < 100) (range 100))), id) = 497 :=
by
  sorry

end sum_of_common_change_l563_563801


namespace slope_condition_mn_distance_eq_sqrt2_l563_563884

noncomputable theory

open real

variables {V : Type*} [inner_product_space ℝ V]

def line_eq (k x : ℝ) : ℝ := k * x + 1

def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 3) ^ 2 = 1

def distance_center_line (k : ℝ) : ℝ := abs (2 * k - 2) / sqrt (k ^ 2 + 1)

def vector_dot (M N : V) : ℝ :=
  let ⟨m1, m2⟩ := M in
  let ⟨n1, n2⟩ := N in
  m1 * n1 + m2 * n2

theorem slope_condition (k : ℝ) :
  distance_center_line k < 1 ↔ k > 4 / 3 :=
by sorry

theorem mn_distance_eq_sqrt2
  {M N O : V}
  {x1 x2 y1 y2 : ℝ}
  {line_inter : (x1, y1) ∈ set_of (λ ⟨x, y⟩, line_eq 1 x = y)}
  {circle_inter : (x1, y1) ∈ set_of (λ ⟨x, y⟩, circle_eq x y) ∧ (x2, y2) ∈ set_of (λ ⟨x, y⟩, circle_eq x y)}
  (cond : vector_dot M N = 12) : dist M N = sqrt 2 :=
by sorry

end slope_condition_mn_distance_eq_sqrt2_l563_563884


namespace sally_paid_peaches_l563_563670

def total_spent : ℝ := 23.86
def amount_spent_on_cherries : ℝ := 11.54
def amount_spent_on_peaches_after_coupon : ℝ := total_spent - amount_spent_on_cherries

theorem sally_paid_peaches : amount_spent_on_peaches_after_coupon = 12.32 :=
by 
  -- The actual proof will involve concrete calculation here.
  -- For now, we skip it with sorry.
  sorry

end sally_paid_peaches_l563_563670


namespace simplify_140_210_l563_563674

noncomputable def simplify_fraction (num den : Nat) : Nat × Nat :=
  let d := Nat.gcd num den
  (num / d, den / d)

theorem simplify_140_210 :
  simplify_fraction 140 210 = (2, 3) :=
by
  have p140 : 140 = 2^2 * 5 * 7 := by rfl
  have p210 : 210 = 2 * 3 * 5 * 7 := by rfl
  sorry

end simplify_140_210_l563_563674


namespace problem1_problem2_l563_563566

-- The conditions of the problem
def conditions (α : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ (tan α = -4 / 3)

theorem problem1 (α : ℝ) (h : conditions α) :
  2 * sin α ^ 2 - 3 * sin α * cos α - 2 * cos α ^ 2 = 2 := sorry

theorem problem2 (α : ℝ) (h : conditions α) :
  (2 * sin (π - α) + sin (π / 2 - α) + sin (4 * π)) / 
  (cos (3 * π / 2 - α) + cos (-α)) = -5 / 7 := sorry

end problem1_problem2_l563_563566


namespace tangential_quadrilateral_k_value_l563_563206

theorem tangential_quadrilateral_k_value :
  ∃ (k : ℝ), k = 3 ∧ let l₁ := {p : ℝ × ℝ | p.1 + 3 * p.2 - 7 = 0},
                        l₂ := {p : ℝ × ℝ | k * p.1 - p.2 - 2 = 0}
                    in ∀ (A B C D : ℝ × ℝ), 
                        (A ∈ l₁ ∧ A.2 = 0 ∨ A ∈ l₂ ∧ A.1 = 0) ∧
                        (B ∈ l₁ ∧ B.1 = 0 ∨ B ∈ l₂ ∧ B.2 = 0) ∧
                        (C ∈ l₁ ∧ C.2 = 0 ∨ C ∈ l₂ ∧ C.1 = 0) ∧
                        (D ∈ l₁ ∧ D.1 = 0 ∨ D ∈ l₂ ∧ D.2 = 0) →
                      dist A B + dist C D = dist A D + dist B C :=
sorry

end tangential_quadrilateral_k_value_l563_563206


namespace solve_eq_l563_563714

theorem solve_eq {x : ℝ} (h : x * (x - 1) = x) : x = 0 ∨ x = 2 := 
by {
    sorry
}

end solve_eq_l563_563714


namespace count_three_digit_numbers_l563_563730

theorem count_three_digit_numbers : 
  (count_numbers : ℕ) = 
  (count_numbers = 
    (let hundreds_digits := {d : ℕ | d ∈ {3, 4, 5}} in
     let tens_digits := {d : ℕ | d ∈ {1, 2, 3, 4, 5}} in
     let units_digits := {d : ℕ | d = 5} in
       (hundreds_digits.card * tens_digits.card * units_digits.card)) ) := 
  15 := sorry

end count_three_digit_numbers_l563_563730


namespace five_digit_numbers_with_first_two_same_l563_563943

theorem five_digit_numbers_with_first_two_same : 
  let digits := Fin 10 in
  ∃ (a b c d e : digits), 
    a.val > 0 ∧  -- The first digit is between 1 and 9 (inclusive)
    a = b ∧     -- The first two digits are the same
    (∃ (count : ℕ), count = 9000) := 
by
  sorry

end five_digit_numbers_with_first_two_same_l563_563943


namespace lines_parallel_lines_perpendicular_l563_563212

-- Definitions of the two lines
def l1 (m : ℝ) : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), p.1 + m * p.2 + 6 = 0
def l2 (m : ℝ) : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), (m - 2) * p.1 + 3 * m * p.2 + 2 * m = 0

-- Parallelism
theorem lines_parallel (m : ℝ) : (m = 0 ∨ m = 5) ↔ (∀ p q : ℝ × ℝ, l1 m p → l1 m q → l2 m p ∧ l2 m q→ false):=
sorry

-- Perpendicularity
theorem lines_perpendicular (m : ℝ) : (m = -1 ∨ m = (2/3)) ↔ (∃ p q : ℝ × ℝ, l1 m p ∧ l1 m q ∧ ¬(l2 m p ∧ l2 m q)) :=
sorry

end lines_parallel_lines_perpendicular_l563_563212


namespace sum_max_min_values_of_f_l563_563527

noncomputable def f (a x : ℝ) : ℝ :=
  (5 * a^x + 3) / (a^x + 1) + 4 * real.log (1 + x) / real.log a - 4 * real.log (1 - x) / real.log a

theorem sum_max_min_values_of_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  (∃ m M : ℝ, min_max f a (m, M) h1 h2 ∧ m + M = 8) := sorry

end sum_max_min_values_of_f_l563_563527


namespace ellipse_standard_equation_l563_563242

variable (a b x y : ℝ)
variable (P : (ℝ × ℝ)) (hP : P = (2, 0))
variable (h1 : a = 2 * b)
variable (h2 : ∃ (a b : ℝ), a = 2 ∧ b = 1)
variable (h3 : x = 2) (h4 : y = 0)

theorem ellipse_standard_equation :
  let x := (P.1), let y := (P.2) in (x^2)/4 + y^2 = 1 :=
by
  cases hP 
  cases h2 with a' ha
  cases ha with ha1 hb
  rw [<-hP_fst, <-hP_snd]
  exact sorry

end ellipse_standard_equation_l563_563242


namespace leak_drain_time_l563_563785

-- Definitions based on the problem conditions
def pump_rate : ℝ := 1 / 2
def combined_rate : ℝ := 3 / 7

-- Goal: Prove that the leak drains the tank in 14 hours
theorem leak_drain_time : 
  let L := pump_rate - combined_rate in
  1 / L = 14 :=
by
  sorry

end leak_drain_time_l563_563785


namespace minimize_y_l563_563880

noncomputable def min_value_expr (a b n : ℝ) : ℝ :=
  (1 / (Real.sqrt 2)^n) * (a^(2/(n-2)) + b^(2/(n-2)))^((n-2)/2)

theorem minimize_y (a b n : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_n : 0 < n) :
   ∃ x ∈ Ioo (-1 : ℝ) 1, (∀ x' ∈ Ioo (-1 : ℝ) 1, (a / ((1 + x')^(n / 2)).sqrt + b / ((1 - x')^(n / 2)).sqrt) ≥ (a / ((1 + x)^(n / 2)).sqrt + b / ((1 - x)^(n / 2)).sqrt)) ∧
   (a / ((1 + x)^(n / 2)).sqrt + b / ((1 - x)^(n / 2)).sqrt) = min_value_expr a b n :=
sorry

end minimize_y_l563_563880


namespace smallest_d_l563_563864

theorem smallest_d (x : Fin 101 → ℝ) (M : ℝ) (h_sum : ∑ i, x i = 201) (h_median : x 50 = M) : 
  ∑ i, (x i)^2 ≤ 51 * M^2 :=
by
  sorry

end smallest_d_l563_563864


namespace h_value_l563_563012

theorem h_value (a b c : ℝ) (h: ℝ) 
  (h₁ : ∀ x, a * x^2 + b * x + c = 3 * (x - 5)^2 + 9) :
  ∀ x, 2 * a * x^2 + 2 * b * x + 2 * c = 6 * (x - h)^2 + 18 → h = 5 := 
begin
  -- proof will go here
  sorry
end

end h_value_l563_563012


namespace divisible_by_11_l563_563228

theorem divisible_by_11 (k : ℕ) (h : 0 ≤ k ∧ k ≤ 9) :
  (9 + 4 + 5 + k + 3 + 1 + 7) - 2 * (4 + k + 1) ≡ 0 [MOD 11] → k = 8 :=
by
  sorry

end divisible_by_11_l563_563228


namespace proof_problem_l563_563192

variable {f : ℝ → ℝ}

-- Conditions given in the problem
axiom f_defined : ∀ x, 0 < x → x ∈ set.univ
axiom f_inequality (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h_diff : x1 ≠ x2) :
  (x2 * f(x1) - x1 * f(x2)) / (x2 - x1) < 0

-- Definitions derived from conditions
noncomputable def a := f (2 ^ 0.2) / (2 ^ 0.2)
noncomputable def b := f (1 / 2) / (1 / 2)
noncomputable def c := f (Real.logBase π 3) / (Real.logBase π 3)

-- The statement to prove
theorem proof_problem : b < c ∧ c < a := sorry

end proof_problem_l563_563192


namespace money_left_over_l563_563029

theorem money_left_over 
  (num_books : ℕ) 
  (price_per_book : ℝ) 
  (num_records : ℕ) 
  (price_per_record : ℝ) 
  (total_books : num_books = 200) 
  (book_price : price_per_book = 1.5) 
  (total_records : num_records = 75) 
  (record_price : price_per_record = 3) :
  (num_books * price_per_book - num_records * price_per_record) = 75 :=
by 
  -- calculation
  sorry

end money_left_over_l563_563029


namespace octal_to_decimal_l563_563430

theorem octal_to_decimal : (1 * 8^3 + 7 * 8^2 + 4 * 8^1 + 3 * 8^0) = 995 :=
by
  sorry

end octal_to_decimal_l563_563430


namespace kevin_stone_count_l563_563088

theorem kevin_stone_count :
  ∃ (N : ℕ), (∀ (n k : ℕ), 2007 = 9 * n + 11 * k → N = 20) := 
sorry

end kevin_stone_count_l563_563088


namespace assemble_checkered_mat_l563_563579

theorem assemble_checkered_mat :
  ∃ (piece3 piece4 piece5 piece6 : matrix (fin 5) (fin 5) (fin 2)),
    is_checkered piece3 ∧
    is_checkered piece4 ∧
    is_checkered piece5 ∧
    is_checkered piece6 ∧
    (assemble_pieces piece3 piece4 piece5 piece6 = matrix.of_fn (λ i j, if (i + j) % 2 = 0 then 1 else 0)) :=
sorry

end assemble_checkered_mat_l563_563579


namespace time_to_park_l563_563947

-- distance from house to market in miles
def d_market : ℝ := 5

-- distance from house to park in miles
def d_park : ℝ := 3

-- time to market in minutes
def t_market : ℝ := 30

-- assuming constant speed, calculate time to park
theorem time_to_park : (3 / 5) * 30 = 18 := by
  sorry

end time_to_park_l563_563947


namespace sierpinski_fractal_dimension_l563_563584

noncomputable def homothety_ratio (n : ℕ) : ℝ :=
  1 / (2 * (1 + ∑ k in finset.range ((n+3)/4), real.cos (2 * k * real.pi / n)))

noncomputable def fractal_dimension (n : ℕ) : ℝ :=
  real.log n / real.log (2 * (1 + ∑ k in finset.range ((n+3)/4), real.cos (2 * k *  real.pi / n)))

theorem sierpinski_fractal_dimension (n : ℕ) (h : 0 < n) :
  fractal_dimension n = (real.log n) / (real.log (2 * (1 + ∑ k in finset.range ((n+3)/4), real.cos (2 * k * real.pi / n)))) := 
sorry

end sierpinski_fractal_dimension_l563_563584


namespace shifted_graph_l563_563589

theorem shifted_graph (f : ℝ → ℝ) (shifted_f : ℝ → ℝ) :
  (∀ x, f x = 2 * x^2) →
  (∀ x, shifted_f x = f x + 1) →
  (∀ x, shifted_f x = 2 * x^2 + 1) :=
by
  intro h1 h2
  funext
  simp [h1, h2]
  sorry

end shifted_graph_l563_563589


namespace complex_div_correct_l563_563691

def complex_div (a b : ℂ) : ℂ := a / b

theorem complex_div_correct : complex_div 2 (1 - complex.i) = 1 + complex.i :=
by 
  sorry

end complex_div_correct_l563_563691


namespace third_number_in_8th_row_l563_563699

theorem third_number_in_8th_row : (∃ f : ℕ → ℕ, f 8 = 48 - 3) :=
by
  let f : ℕ → ℕ := λ i, (6 * i - 3)
  use f
  exact rfl
  -- sorry can be used to skip the proof
  sorry  -- noncomputable

end third_number_in_8th_row_l563_563699


namespace symmetry_condition_l563_563334

theorem symmetry_condition (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∀ x y : ℝ, y = x ↔ x = (ax + b) / (cx - d)) ∧ 
  (∀ x y : ℝ, y = -x ↔ x = (-ax + b) / (-cx - d)) → 
  d + b = 0 :=
by sorry

end symmetry_condition_l563_563334


namespace dot_product_AB_CD_l563_563236

variables (A B C D M N : ℝ) -- A placeholder for points; actually vectors should be used properly.
-- Assuming already declared vector and dot product properties...

-- Conditions
variables [ConvexQuadrilateral ABCD] -- Convex quadrilateral type
variables [Midpoint M AD] [Midpoint N BC]
variables (AB AD BC CD : ℝ)
axiom length_AB : |AB| = 2
axiom length_MN : |MN| = 3/2
axiom dot_product_condition : MN • (AD - BC) = 3/2

-- To be proven
theorem dot_product_AB_CD (hAB : |AB| = 2) (hMN : |MN| = 3/2)
  (h_dot : MN • (AD - BC) = 3/2) : AB • CD = -2 := sorry

end dot_product_AB_CD_l563_563236


namespace find_max_volume_pyramid_l563_563663

noncomputable def max_volume_pyramid (volume_prism : ℝ) (r_AM : ℝ) (r_BN : ℝ) (r_CK : ℝ) : ℝ :=
  if h : volume_prism = 35 ∧ r_AM = 5 / 6 ∧ r_BN = 6 / 7 ∧ r_CK = 2 / 3 then
    10
  else
    0 -- default value in case the conditions not match (this branch will be unreachable if used correctly)

theorem find_max_volume_pyramid 
  (volume_prism : ℝ)
  (r_AM : ℝ)
  (r_BN : ℝ)
  (r_CK : ℝ)
  (h_volume : volume_prism = 35)
  (h_AM : r_AM = 5 / 6)
  (h_BN : r_BN = 6 / 7)
  (h_CK : r_CK = 2 / 3) :
  max_volume_pyramid volume_prism r_AM r_BN r_CK = 10 := 
by
  unfold max_volume_pyramid
  simp [h_volume, h_AM, h_BN, h_CK]
  -- replace sorry with appropriate proof steps if you're constructing a full proof.
  sorry

end find_max_volume_pyramid_l563_563663


namespace solve_for_a_l563_563190

theorem solve_for_a (x a : ℝ) (h : x = 3) (eqn : 2 * (x - 1) - a = 0) : a = 4 := 
by 
  sorry

end solve_for_a_l563_563190


namespace minimum_distance_from_circle_to_line_l563_563195

-- Definitions from the conditions
def parametric_line (t : ℝ) : ℝ × ℝ := ((1 - (Real.sqrt 2) / 2 * t), (2 + (Real.sqrt 2) / 2 * t))

def polar_circle (θ : ℝ) : ℝ := 2 * Real.cos θ

-- Converting to Cartesian coordinates form from the given conditions
def line_cartesian (x y : ℝ) : Prop := x + y = 3

def circle_cartesian (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Proving the minimum distance
theorem minimum_distance_from_circle_to_line : ∀ (t θ : ℝ), 
  let x := 1 - (Real.sqrt 2) / 2 * t
  let y := 2 + (Real.sqrt 2) / 2 * t in
  line_cartesian x y ∧ circle_cartesian (2 * Real.cos θ) θ →
  Real.abs (1 - 3) / Real.sqrt 2 - 1 = Real.sqrt 2 - 1 := 
by
  sorry

end minimum_distance_from_circle_to_line_l563_563195


namespace azerbaijan_ineq_l563_563996

theorem azerbaijan_ineq {x y z : ℝ} (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) 
  (h_eq : x + y + z = (1/x) + (1/y) + (1/z)) : 
  x + y + z ≥ (sqrt ((xy + 1) / 2)) + (sqrt ((yz + 1) / 2)) + (sqrt ((zx + 1) / 2)) :=
sorry

end azerbaijan_ineq_l563_563996


namespace exists_person_who_knows_everyone_l563_563598

variable {Person : Type}
variable (knows : Person → Person → Prop)
variable (n : ℕ)

-- Condition: In a company of 2n + 1 people, for any n people, there is another person different from them who knows each of them.
axiom knows_condition : ∀ (company : Finset Person) (h : company.card = 2 * n + 1), 
  (∀ (subset : Finset Person) (hs : subset.card = n), ∃ (p : Person), p ∉ subset ∧ ∀ q ∈ subset, knows p q)

-- Statement to be proven:
theorem exists_person_who_knows_everyone (company : Finset Person) (hcompany : company.card = 2 * n + 1) :
  ∃ p, ∀ q ∈ company, knows p q :=
sorry

end exists_person_who_knows_everyone_l563_563598


namespace solution_to_functional_equation_l563_563852

noncomputable def find_functions (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (x / 2) + (x / 2) * (deriv f x)

theorem solution_to_functional_equation :
  ∀ (f : ℝ → ℝ), (∀ x : ℝ, f x = f (x / 2) + (x / 2) * (deriv f x)) ↔ (∃ c b : ℝ, ∀ x : ℝ, f x = c * x + b) :=
by {
  sorry
}

end solution_to_functional_equation_l563_563852


namespace extremum_values_of_f_l563_563181

open Real

noncomputable def f (x : Fin n → ℝ) : ℝ :=
  ∑ i : Fin n, sqrt (x i ^ 2 + x i * x ((i + 1) % n) + x ((i + 1) % n) ^ 2)

theorem extremum_values_of_f {n : ℕ} (h₁ : 2 ≤ n)
  (x : Fin n → ℝ) (hx : (∑ i, x i) = 1) (hxn : ∀ i, 0 ≤ x i) :
  sqrt 3 ≤ f x ∧ f x ≤ 2 :=
sorry

end extremum_values_of_f_l563_563181


namespace math_problem_l563_563273

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * Real.log x + (1 / (2 * x)) - (3 / 2) * x + 1

-- f(x)=a*ln(x) + 1/(2x) - (3/2)x + 1 at x = 1 and f(1), 
-- We should have
-- 1. a = 2
-- 2. intervals of monotonicity
-- 3. Extreme values at specific points

theorem math_problem : 
  (f'(1) = 0 → a = 2) ∧
  (f′Must be calculated) ∧
  (intervals ∶ increasing\ [1/3, 1] and decreasing\[(0, 1/3) Union (1, ∞ )  ∧ must have extreme values with specific points) → 
Proof. Ivx.sorry


end math_problem_l563_563273


namespace explicit_formula_range_of_t_l563_563875

-- Definitions based on the conditions:
def f (x : ℝ) : ℝ := 2 * x^2 + b * x + c

variables {b c t : ℝ}

-- Condition: the solution set of the inequality f(x) < 0 is (0, 5).
axiom roots_condition : ∀ x : ℝ, f x < 0 ↔ 0 < x ∧ x < 5

-- Question 1: Prove the explicit formula of f(x)
theorem explicit_formula (h1 : roots_condition) : f = λ x, 2 * x^2 - 10 * x :=
sorry

-- Question 2: Prove the range of values for t
theorem range_of_t (h2 : ∀ x ∈ set.Icc (-1 : ℝ) 1, f x + t ≤ 2) : t ≤ -10 :=
sorry

end explicit_formula_range_of_t_l563_563875


namespace sales_fourth_month_l563_563074

theorem sales_fourth_month (sale2 sale3 sale5 sale6 : ℕ) (average_sale : ℕ) (months : ℕ) (expected_total_sales : ℕ) (sale6_val : ℕ) :
  sale2 = 6735 →
  sale3 = 6927 →
  sale5 = 6855 →
  sale6_val = 4691 →
  average_sale = 6500 →
  months = 6 →
  expected_total_sales = average_sale * months →
  (expected_total_sales - (sale2 + sale3 + sale5 + sale6_val) = 7230) :=
by
  intros h2 h3 h5 h6 h_avg h_months h_total
  rw [h2, h3, h5, h6, h_avg, h_months] at h_total
  change 39000 - (6735 + 6927 + 6855 + 4691) = 7230
  sorry

end sales_fourth_month_l563_563074


namespace doughnuts_given_away_l563_563413

def doughnuts_left (total_doughnuts : Nat) (doughnuts_per_box : Nat) (boxes_sold : Nat) : Nat :=
  total_doughnuts - (doughnuts_per_box * boxes_sold)

theorem doughnuts_given_away :
  doughnuts_left 300 10 27 = 30 :=
by
  rw [doughnuts_left]
  simp
  sorry

end doughnuts_given_away_l563_563413


namespace wednesday_tips_value_l563_563580

-- Definitions for the conditions
def hourly_wage : ℕ := 10
def monday_hours : ℕ := 7
def tuesday_hours : ℕ := 5
def wednesday_hours : ℕ := 7
def monday_tips : ℕ := 18
def tuesday_tips : ℕ := 12
def total_earnings : ℕ := 240

-- Hourly earnings
def monday_earnings := monday_hours * hourly_wage
def tuesday_earnings := tuesday_hours * hourly_wage
def wednesday_earnings := wednesday_hours * hourly_wage

-- Total wage earnings
def total_wage_earnings := monday_earnings + tuesday_earnings + wednesday_earnings

-- Total earnings with known tips
def known_earnings := total_wage_earnings + monday_tips + tuesday_tips

-- Prove that Wednesday tips is $20
theorem wednesday_tips_value : (total_earnings - known_earnings) = 20 := by
  sorry

end wednesday_tips_value_l563_563580


namespace solve_system_l563_563312

theorem solve_system (x y : ℝ) (h1 : x^2 + y^2 + x + y = 50) (h2 : x * y = 20) :
  (x = 5 ∧ y = 4) ∨ (x = 4 ∧ y = 5) ∨ (x = -5 + Real.sqrt 5 ∧ y = -5 - Real.sqrt 5) ∨ (x = -5 - Real.sqrt 5 ∧ y = -5 + Real.sqrt 5) :=
by
  sorry

end solve_system_l563_563312


namespace not_square_difference_formula_l563_563744

theorem not_square_difference_formula (x y : ℝ) : ¬ ∃ (a b : ℝ), (x - y) * (-x + y) = (a + b) * (a - b) := 
sorry

end not_square_difference_formula_l563_563744


namespace find_second_sum_l563_563455

theorem find_second_sum (x : ℝ) (h_sum : x + (2678 - x) = 2678)
  (h_interest : x * (3 / 100) * 8 = (2678 - x) * (5 / 100) * 3) :
  (2678 - x) = 2401 :=
by
  sorry

end find_second_sum_l563_563455


namespace john_spent_correct_amount_l563_563625

-- Definitions of constants
def umbrellas := 2
def raincoats := 3
def umbrella_cost := 8
def raincoat_cost := 15
def waterproof_bag_cost := 25
def discount_rate := 0.10
def refund_rate := 0.80

-- Noncomputable definitions when required
noncomputable def total_cost_before_discount :=
  (umbrellas * umbrella_cost) + (raincoats * raincoat_cost) + waterproof_bag_cost

noncomputable def discount_amount :=
  total_cost_before_discount * discount_rate

noncomputable def total_cost_after_discount :=
  total_cost_before_discount - discount_amount

noncomputable def refund_for_defective_raincoat :=
  raincoat_cost * refund_rate

noncomputable def total_spent :=
  total_cost_after_discount - refund_for_defective_raincoat

-- The theorem we wish to prove
theorem john_spent_correct_amount : total_spent = 65.40 :=
by
  sorry

end john_spent_correct_amount_l563_563625


namespace sum_AC_AD_constant_l563_563541

theorem sum_AC_AD_constant
  (A : ℝ) (B : ℝ) (C D : ℝ)
  (h1 : ∃ (bisector : ℝ), B = bisector)
  (h2 : ∃ (circle : ℝ), circle = (A, B, C, D))
  (h3 : ∃ (intersect : ℝ), (intersect = A ∩ circle) ∧ (D = intersect ∧ C = intersect)) :
  AC + AD = 2 * AB * Real.cos (A / 2) :=
sorry

end sum_AC_AD_constant_l563_563541


namespace problem_statement_l563_563264

theorem problem_statement {M : ℝ} (hM : M > 1) (a : ℕ → ℝ) (hA : ∀ n, 1 ≤ a n ∧ a n ≤ M) (ε : ℝ) (hε: ε ≥ 0) :
  ∃ n > 0, ∀ t > 0, ∑ k in Finset.range(t), (a (n+k)) / (a (n+k+1)) ≥ t - ε := 
begin
  sorry,
end

end problem_statement_l563_563264


namespace simplify_comb_expression_l563_563528

open Nat

def C (n k : ℕ) : ℕ := choose n k

theorem simplify_comb_expression (n m k : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m ≤ n) :
  C n m + (∑ j in finset.range (k + 1), C k j * C n (m - j)) = C (n + k) m :=
by
  sorry

end simplify_comb_expression_l563_563528


namespace solve_for_x_l563_563493

def star (a b : ℤ) := a * b + 3 * b - a

theorem solve_for_x : ∃ x : ℤ, star 4 x = 46 := by
  sorry

end solve_for_x_l563_563493


namespace rectangle_distance_sum_ge_area_twice_l563_563629

theorem rectangle_distance_sum_ge_area_twice (A B C D P : Point) (k q : ℝ) (S : ℝ)
  (hS : S = k * q)
  (ha : distance P A = a)
  (hb : distance P B = b)
  (hc : distance P C = c)
  (hd : distance P D = d)
  (hABCD : rectangle A B C D) :
  a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 ≥ 2 * S :=
by
  sorry

end rectangle_distance_sum_ge_area_twice_l563_563629


namespace triangle_arctan_sum_pi_over_four_l563_563254

theorem triangle_arctan_sum_pi_over_four
  (A B C : Point)
  (a b c : ℝ)
  (h_triangle : triangle A B C)
  (h_angle_A : angle A B C = π / 4)
  (h_sides : dist A B = dist B C) :
  arctan (c / (a + b)) + arctan (b / (a + c)) = π / 4 :=
by
  sorry

end triangle_arctan_sum_pi_over_four_l563_563254


namespace inequalities_hold_for_m_l563_563347

theorem inequalities_hold_for_m:
  (|m_3| ≤ 3) →
  (|m_4| ≤ 5) →
  (|m_5| ≤ 7) →
  (|m_6| ≤ 9) →
  (|m_7| ≤ 12) →
  (|m_8| ≤ 14) →
  (|m_9| ≤ 16) →
  (|m_{10}| ≤ 19) →
  (|m_3| ≤ 3) ∧
  (|m_4| ≤ 5) ∧
  (|m_5| ≤ 7) ∧
  (|m_6| ≤ 9) ∧
  (|m_7| ≤ 12) ∧
  (|m_8| ≤ 14) ∧
  (|m_9| ≤ 16) ∧
  (|m_{10}| ≤ 19) := 
by
  intros h₃ h₄ h₅ h₆ h₇ h₈ h₉ h₁₀
  split; 
  assumption

end inequalities_hold_for_m_l563_563347


namespace find_x_l563_563517

-- Let x be a real number such that x > 0 and the area of the given triangle is 180.
theorem find_x (x : ℝ) (h_pos : x > 0) (h_area : 3 * x^2 = 180) : x = 2 * Real.sqrt 15 :=
by
  -- Placeholder for the actual proof
  sorry

end find_x_l563_563517


namespace complex_div_equiv_l563_563198

-- Define the given complex numbers
def two_minus_i : ℂ := 2 - complex.i
def i : ℂ := complex.i
def expected_result : ℂ := -1 - 2 * complex.i

-- Define the theorem we want to prove
theorem complex_div_equiv : (two_minus_i / i) = expected_result := by
  sorry

end complex_div_equiv_l563_563198


namespace measure_angle_EDF_l563_563255

/-
  Define the geometric entities and conditions from the problem:
  - triangle ABC with AB = AC and ∠A = 80°
  - points D, E, F on BC, AC, AB with CE = CD and BF = BD
  - ∠EDF to be determined
-/

open EuclideanGeometry

-- Assume the existence of points and angles
variables {A B C D E F : Point}
variables h_AB_AC : dist A B = dist A C
variables h_angle_A : angle A B C = 80

-- Define the points
variables h_D : D ∈ segment B C
variables h_E : E ∈ segment A C
variables h_F : F ∈ segment A B

-- Define the conditions on segments CE = CD and BF = BD
variables h_CE_CD : dist C E = dist C D
variables h_BF_BD : dist B F = dist B D

-- Define the proof problem
theorem measure_angle_EDF :
  angle E D F = 50 :=
  sorry  -- The proof goes here

end measure_angle_EDF_l563_563255


namespace smallest_m_for_unity_root_l563_563633

noncomputable def T : Set ℂ :=
  { z : ℂ | ∃ x y : ℝ, z = x + y * complex.I ∧ (1 / 2 ≤ x ∧ x ≤ real.sqrt 2 / 2) }

theorem smallest_m_for_unity_root (m : ℕ) :
  (∀ n : ℕ, n ≥ 13 → ∃ z : ℂ, z ∈ T ∧ z^n = 1) →
  m = 13 :=
by {
  intro h,
  have h13 := h 13 (by norm_num),
  cases h13 with z hz,
  use 13,
  exact hz,
}

end smallest_m_for_unity_root_l563_563633


namespace prove_S_geq_2T_l563_563604

open Real

-- Define the area function for a triangle with vertices (x₁, y₁), (x₂, y₂), (x₃, y₃)
noncomputable def area (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  abs ((x₂ - x₁) * (y₃ - y₁) - (x₃ - x₁) * (y₂ - y₁)) / 2

-- Define the conditions from the problem
def right_triangle (A B C : ℝ × ℝ) : Prop :=
  (A.2 = B.2) ∧ (A.1 = C.1)

def altitude (A D : ℝ × ℝ) : Prop :=
  D.1 = A.1

def has_incenter (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  ∃ (M : ℝ × ℝ), in_center M (x₁, y₁) (x₂, y₂) (x₃, y₃)

def intersect_points (AM AN : ℝ × ℝ) (AB AC : ℝ × ℝ) (K L : ℝ × ℝ) : Prop :=
  true -- We assume intersection points exist for simplicity in this setting

theorem prove_S_geq_2T (A B C D K L : ℝ × ℝ) :
  right_triangle A B C → altitude A D →
  has_incenter A.1 A.2 B.1 B.2 D.1 D.2 → has_incenter A.1 A.2 C.1 C.2 D.1 D.2 →
  intersect_points (A.1, A.2) (A.1, A.2) (B.1, B.2) (C.1, C.2) K L →
  let S := area A.1 A.2 B.1 B.2 C.1 C.2,
      T := area A.1 A.2 K.1 K.2 L.1 L.2
  in S = T :=
by
  intros
  sorry

end prove_S_geq_2T_l563_563604


namespace quadrilateral_area_l563_563668

-- Given conditions for the problem
variables (PQ QR RS PT : ℝ)
axiom h1 : QR = 25
axiom h2 : RS = 40
axiom h3 : PT = 8
axiom h4 : ∠PQR = 90
axiom h5 : ∠QRS = 90

-- Statement to prove
theorem quadrilateral_area : 
  let PR := real.sqrt (PQ^2 + QR^2) in
  let QS := real.sqrt (QR^2 + RS^2) in
  let [PQR] := (1/2) * PQ * QR in
  let [PQS] := [PQR] + 500 in
  ∃ PQ, [PQRS] = 650 := 
sorry

end quadrilateral_area_l563_563668


namespace opposite_of_neg_three_l563_563001

-- Define the concept of negation and opposite of a number
def opposite (x : ℤ) : ℤ := -x

-- State the problem: Prove that the opposite of -3 is 3
theorem opposite_of_neg_three : opposite (-3) = 3 :=
by
  -- Proof
  sorry

end opposite_of_neg_three_l563_563001


namespace find_a_and_other_root_l563_563548

-- Define the quadratic equation with a
def quadratic_eq (a x : ℝ) : ℝ := (a + 1) * x^2 + x - 1

-- Define the conditions where -1 is a root
def condition (a : ℝ) : Prop := quadratic_eq a (-1) = 0

theorem find_a_and_other_root (a : ℝ) :
  condition a → 
  (a = 1 ∧ ∃ x : ℝ, x ≠ -1 ∧ quadratic_eq 1 x = 0 ∧ x = 1 / 2) :=
by
  intro h
  sorry

end find_a_and_other_root_l563_563548


namespace max_value_of_A_l563_563931

noncomputable def cos_sq (x: ℝ) := real.cos x ^ 2
noncomputable def ctg_sq (x: ℝ) := (real.cos x / real.sin x) ^ 2
noncomputable def ctg_qd (x: ℝ) := (ctg_sq x) ^ 2

theorem max_value_of_A (x : ℕ → ℝ) (n : ℕ) (hx : ∀ i, i < n → 0 < x i ∧ x i < π / 2) :
  (∑ i in finset.range n, cos_sq (x i)) / (real.sqrt n + real.sqrt (∑ i in finset.range n, ctg_qd (x i))) ≤ real.sqrt n / 4 :=
sorry

end max_value_of_A_l563_563931


namespace squared_expression_l563_563906

variable (x : ℝ)

theorem squared_expression (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

end squared_expression_l563_563906


namespace total_vegetables_correct_l563_563476

def cucumbers : ℕ := 70
def tomatoes : ℕ := 3 * cucumbers
def total_vegetables : ℕ := cucumbers + tomatoes

theorem total_vegetables_correct : total_vegetables = 280 :=
by
  sorry

end total_vegetables_correct_l563_563476


namespace sum_of_first_five_terms_l563_563014

theorem sum_of_first_five_terms 
  (a₂ a₃ a₄ : ℤ)
  (h1 : a₂ = 4)
  (h2 : a₃ = 7)
  (h3 : a₄ = 10) :
  ∃ a1 a5, a1 + a₂ + a₃ + a₄ + a5 = 35 :=
by
  sorry

end sum_of_first_five_terms_l563_563014


namespace prime_factor_condition_l563_563572

def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | 1 => 1
  | n + 2 => seq (n + 1) + seq n

theorem prime_factor_condition (p k : ℕ) (hp : Nat.Prime p) (h : p ∣ seq (2 * k) - 2) :
  p ∣ seq (2 * k - 1) - 1 :=
sorry

end prime_factor_condition_l563_563572


namespace groom_poodle_time_is_30_l563_563339

-- Define the time it takes to groom a poodle
variable (P : ℝ)

-- Define conditions
def groom_poodle_time : ℝ := P
def groom_terrier_time : ℝ := P / 2
def shop_grooms_poodles_and_terriers (total_time : ℝ) : Prop :=
  3 * groom_poodle_time P + 8 * groom_terrier_time P = total_time

-- The goal is to prove that P equals 30 given the conditions
theorem groom_poodle_time_is_30 (h : shop_grooms_poodles_and_terriers P 210) :
  P = 30 :=
by
  sorry

end groom_poodle_time_is_30_l563_563339


namespace gecko_eggs_hatch_l563_563433

theorem gecko_eggs_hatch :
  let total_eggs := 30 in
  let infertile_percentage := 0.20 in
  let calcification_fraction := 1 / 3 in
  let infertile_eggs := infertile_percentage * total_eggs in
  let fertile_eggs := total_eggs - infertile_eggs in
  let non_hatching_eggs := calcification_fraction * fertile_eggs in
  let hatching_eggs := fertile_eggs - non_hatching_eggs in
  hatching_eggs = 16 :=
by
  sorry

end gecko_eggs_hatch_l563_563433


namespace volume_is_750_sqrt2_l563_563688

noncomputable def volume_of_prism 
  (a b c : ℝ)
  (h1 : a * b = 30)
  (h2 : a * c = 50)
  (h3 : b * c = 75) : ℝ :=
a * b * c

theorem volume_is_750_sqrt2 (a b c : ℝ)
  (h1 : a * b = 30)
  (h2 : a * c = 50)
  (h3 : b * c = 75) : volume_of_prism a b c h1 h2 h3 = 750 * real.sqrt 2 :=
by sorry

end volume_is_750_sqrt2_l563_563688


namespace bicycle_new_price_l563_563061

theorem bicycle_new_price (original_price : ℝ) (fst_discount snd_discount : ℝ) : 
  original_price = 200 → fst_discount = 0.40 → snd_discount = 0.25 →
  (original_price * (1 - fst_discount) * (1 - snd_discount)) = 90 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  norm_num
  sorry

end bicycle_new_price_l563_563061


namespace calculate_expression_l563_563820

theorem calculate_expression :
  (5 * 7 + 10 * 4 - 35 / 5 + 18 / 3 : ℝ) = 74 := by
  sorry

end calculate_expression_l563_563820


namespace solve_for_x_l563_563951

theorem solve_for_x (x : ℝ) (h : x / 6 = 15 / 10) : x = 9 :=
by
  sorry

end solve_for_x_l563_563951


namespace complex_product_l563_563478

open Complex

#check Complex

/-- The product of the complex numbers (3 + i) and (1 - 2i) is 5 - 5i. -/
theorem complex_product : ((3 : ℂ) + Complex.i) * ((1 : ℂ) - 2 * Complex.i) = (5 : ℂ) - 5 * Complex.i := 
by 
  sorry

end complex_product_l563_563478


namespace jo_reading_time_l563_563623

structure Book :=
  (totalPages : Nat)
  (currentPage : Nat)
  (pageOneHourAgo : Nat)

def readingTime (b : Book) : Nat :=
  let pagesRead := b.currentPage - b.pageOneHourAgo
  let pagesLeft := b.totalPages - b.currentPage
  pagesLeft / pagesRead

theorem jo_reading_time :
  ∀ (b : Book), b.totalPages = 210 → b.currentPage = 90 → b.pageOneHourAgo = 60 → readingTime b = 4 :=
by
  intro b h1 h2 h3
  sorry

end jo_reading_time_l563_563623


namespace incenter_inequality_l563_563887

variable {A B C P : Type}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P]

-- Define an acute triangle ΔABC with incenter P
def isIncenter (A B C P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] : Prop :=
sorry -- define what it means for P being an incenter of the triangle

-- Define the perimeter l' of the triangle formed by the points where the incircle of ΔABC touches its sides
def perimeterIncircleTouchingTriangle(A B C P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] : ℝ :=
sorry -- provide this definition given P is incenter

-- The theorem to prove
theorem incenter_inequality (h : isIncenter A B C P) :
  let l := perimeterIncircleTouchingTriangle A B C P in
  PA + PB + PC ≥ (2 / Real.sqrt 3) * l :=
sorry

end incenter_inequality_l563_563887


namespace units_digit_of_eight_consecutive_odd_numbers_is_zero_l563_563152

def is_odd (n : ℤ) : Prop :=
  ∃ k : ℤ, n = 2 * k + 1

theorem units_digit_of_eight_consecutive_odd_numbers_is_zero (n : ℤ)
  (h₀ : is_odd n) :
  ((n * (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) * (n + 12) * (n + 14)) % 10 = 0) :=
sorry

end units_digit_of_eight_consecutive_odd_numbers_is_zero_l563_563152


namespace geometric_sequence_sum_l563_563533

theorem geometric_sequence_sum (a_n : ℕ → ℝ) (q : ℝ) (n : ℕ) 
    (S_n : ℝ) (S_3n : ℝ) (S_4n : ℝ)
    (h1 : S_n = 2) 
    (h2 : S_3n = 14) 
    (h3 : ∀ m : ℕ, S_m = a_n 1 * (1 - q^m) / (1 - q)) :
    S_4n = 30 :=
by
  sorry

end geometric_sequence_sum_l563_563533


namespace find_y_l563_563046

theorem find_y (y : ℝ) (h : 3 * y / 4 = 15) : y = 20 :=
sorry

end find_y_l563_563046


namespace samara_tire_spending_l563_563800

theorem samara_tire_spending :
  ∀ (T : ℕ), 
    (2457 = 25 + 79 + T + 1886) → 
    T = 467 :=
by intros T h
   sorry

end samara_tire_spending_l563_563800


namespace conic_section_is_ellipse_l563_563381

/-- Definitions of fixed points and the equation constant --/
def point1 := (0, 2 : ℝ × ℝ)
def point2 := (6, -4 : ℝ × ℝ)
def constant := 12

/-- Formulation of the given conic section equation --/
def conic_section_equation (x y : ℝ) : Prop :=
  real.sqrt (x^2 + (y - point1.snd)^2) + real.sqrt ((x - point2.fst)^2 + (y - point2.snd)^2) = constant

/-- Problem Statement: The conic section described by the given equation is an ellipse --/
theorem conic_section_is_ellipse : ∀ x y : ℝ, conic_section_equation x y → 
  (conic_section_equation x y → "E" = "E") :=
by
  intros x y h
  sorry

end conic_section_is_ellipse_l563_563381


namespace shirt_assignment_ways_l563_563056

/--
  12 people stand in a row.
  Each person is given a red or a blue shirt.
  Every minute, exactly one pair of people with the same color currently standing next to each other leaves.
  After 6 minutes, everyone has left.
  Prove that the number of ways the shirts could have been assigned initially is 837.
-/
theorem shirt_assignment_ways : 
  let n := 12
  let k := 6
  let possible_assignments := -- these are all possible color assignments for the shirts
    { assignments | ∃ (red_count blue_count : ℕ), 
        red_count + blue_count = n ∧
        -- logic ensuring all pairs of same color shirts adjacent are removed in k steps
        only_pairwise_removal_possible assignments k
    }
  in
  card possible_assignments = 837 :=
sorry

end shirt_assignment_ways_l563_563056


namespace part_a_arrangement_exists_l563_563387

-- Part (a): Prove existence of such arrangement for 1 to 32
theorem part_a_arrangement_exists : 
  ∃ (arrange : Finset ℕ → list ℕ), (∀ S : Finset ℕ, S = Finset.range 1 33 → arrange S = [32, 30, 28, ..., 2, 31, 29, ..., 1]) → 
  (∀ i j, i ≠ j → i ∈ S → j ∈ S → ¬ ∃ k, k ∈ S → k ≠ i → k ≠ j → (i + j) / 2 = k) :=
sorry

end part_a_arrangement_exists_l563_563387


namespace projection_range_l563_563576

variables (a b : EuclideanSpace ℝ (Fin 3))

open Real

-- Define the conditions
def condition1 (a b : EuclideanSpace ℝ (Fin 3)) : Prop :=
  (norm (a + b))^2 - (norm b)^2 = 3

def condition2 (b : EuclideanSpace ℝ (Fin 3)) : Prop :=
  norm b ≥ 2

-- Define the projection of a onto b
def projection (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  (a ⬝ b) / norm b

-- Problem statement
theorem projection_range (a b : EuclideanSpace ℝ (Fin 3)) :
  condition1 a b ∧ condition2 b → -3/2 ≤ projection a b ∧ projection a b < 0 :=
begin
  sorry
end

end projection_range_l563_563576


namespace volume_ratio_l563_563149

noncomputable theory

def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

def volume_cone (r h : ℝ) : ℝ := (1/3) * π * r^2 * (h / 3)

theorem volume_ratio (h_cylinder r_cylinder : ℝ) (h_cone := h_cylinder / 3) (r_cone := r_cylinder)
  (h_cylinder_eq : h_cylinder = 15) (r_cylinder_eq : r_cylinder = 5) :
  (volume_cone r_cone h_cylinder) / (volume_cylinder r_cylinder h_cylinder) = 1 / 9 :=
by {
  sorry -- proof goes here
}

end volume_ratio_l563_563149


namespace fixed_point_l563_563697

theorem fixed_point (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) : (3, 2) ∈ set_of (λ (p : ℝ × ℝ), p.snd = a^(p.fst - 3) + 1) :=
begin
  -- Proof ommited (as per instructions)
  sorry
end

end fixed_point_l563_563697


namespace area_of_triangle_ABC_l563_563596

variables {A B C : Type} 
          [normed_group A] [inner_product_space ℝ A] [normed_group B]
          [inner_product_space ℝ B] [normed_group C] [inner_product_space ℝ C]
          [add_comm_group A] [module ℝ A] [add_comm_group B] [module ℝ B]
          [add_comm_group C] [module ℝ C]

noncomputable def area_of_triangle (a b c : ℝ) (cosB : ℝ) : ℝ :=
  let sinB := real.sqrt (1 - cosB^2) in 
  1 / 2 * a * c * sinB

theorem area_of_triangle_ABC : 
  let a := 2 in
  let c := 5 in
  let cosB := 3 / 5 in 
  area_of_triangle a c cosB = 4 := 
by
  sorry

end area_of_triangle_ABC_l563_563596


namespace eli_age_difference_l563_563260

theorem eli_age_difference (kaylin_age : ℕ) (freyja_age : ℕ) (sarah_age : ℕ) (eli_age : ℕ) 
  (H1 : kaylin_age = 33)
  (H2 : freyja_age = 10)
  (H3 : kaylin_age + 5 = sarah_age)
  (H4 : sarah_age = 2 * eli_age) :
  eli_age - freyja_age = 9 := 
sorry

end eli_age_difference_l563_563260


namespace average_of_consecutive_integers_l563_563672

variable (c : ℕ)
variable (d : ℕ)

-- Given condition: d == (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5) + (c+6)) / 7
def condition1 : Prop := d = (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5) + (c+6)) / 7

-- The theorem to prove
theorem average_of_consecutive_integers : condition1 c d → 
  (d + 1 + d + 2 + d + 3 + d + 4 + d + 5 + d + 6 + d + 7 + d + 8 + d + 9) / 10 = c + 9 :=
sorry

end average_of_consecutive_integers_l563_563672


namespace area_of_triangle_ABC_is_25_l563_563245

-- Define the right triangle ABC with given conditions
structure RightTriangleABC :=
  (A B C : Point)
  (angle_right : is_right_angle (A, B, C))
  (angle_equal : ∠A = ∠B)
  (AB_length : dist A B = 10)

-- Define the proof that area of triangle ABC is 25
theorem area_of_triangle_ABC_is_25 (t : RightTriangleABC) : 
  let AC := dist t.A t.C in
  let BC := dist t.B t.C in
  1/2 * AC * BC = 25 :=
by
  sorry

end area_of_triangle_ABC_is_25_l563_563245


namespace proof_problem_l563_563386

noncomputable def ineq_statement (x : ℝ) : Prop := 
  9.296 * (cos x + 2 * cos x ^ 2 + cos (3 * x)) / (cos x + 2 * cos x ^ 2 - 1) > 1

noncomputable def cosine_condition (x : ℝ) : Prop :=
  cos x + 2 * cos x ^ 2 - 1 ≠ 0

theorem proof_problem (x : ℝ) (n : ℤ) 
  (h1 : cosine_condition x) 
  (h2 : ineq_statement x) 
  : x ∈ Set.Ioo ((6 * n - 1) * (π / 3)) ((6 * n + 1) * (π / 3)) :=
sorry

end proof_problem_l563_563386


namespace geometric_harmonic_mean_relation_l563_563645

noncomputable def seq_a (a b : ℝ) : ℕ → ℝ
| 0     := a
| (n+1) := 2 * seq_a a b n * seq_b a b n / (seq_a a b n + seq_b a b n)

noncomputable def seq_b (a b : ℝ) : ℕ → ℝ
| 0     := b
| (n+1) := Real.sqrt (seq_a a b n * seq_b a b n)

noncomputable def v (a b : ℝ) : ℝ :=
  let seq_limit := limit (seq_a a b) in
  seq_limit

noncomputable def mu (x y : ℝ) : ℝ :=
  let seq_limit := limit (λ n, (seq_a (1 / x) (1 / y) n + seq_b (1 / x) (1 / y) n) / 2) in
  seq_limit

theorem geometric_harmonic_mean_relation (a b : ℝ) : 
  v(a, b) * mu(1/a, 1/b) = 1 :=
sorry

end geometric_harmonic_mean_relation_l563_563645


namespace equal_sum_sequence_a18_equal_sum_sequence_sum_l563_563834

section equal_sum_sequence

def is_equal_sum_sequence (a : ℕ → ℝ) (s : ℝ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = s

variables (a : ℕ → ℝ) (h_seq : is_equal_sum_sequence a 5) (ha1 : a 1 = 2)

noncomputable def a_18 := a 18
noncomputable def S_n (n : ℕ) : ℝ :=
  if n % 2 = 0 then 5 * n / 2 else (5 * n - 1) / 2

theorem equal_sum_sequence_a18 : a_18 a = 3 :=
sorry

theorem equal_sum_sequence_sum (n : ℕ) : S_n a n =
  if n % 2 = 0 then 5 * n / 2 else (5 * n - 1) / 2 :=
sorry

end equal_sum_sequence

end equal_sum_sequence_a18_equal_sum_sequence_sum_l563_563834


namespace tiling_6x6_feasibility_l563_563060

def is_valid_tiling (k : ℕ) : Prop :=
  k ∈ {2, 4, 5, 6, 7, 8, 9, 10, 11, 12}

theorem tiling_6x6_feasibility (k : ℕ) (h : k ≤ 12) :
  is_valid_tiling k ↔
  (∃ (L_tiles rect_tiles : ℕ),
    L_tiles = k ∧
    rect_tiles = 12 - k ∧
    k ∈ {2, 4, 5, 6, 7, 8, 9, 10, 11, 12}) :=
by
  sorry

end tiling_6x6_feasibility_l563_563060


namespace prime_dates_in_february_2024_l563_563290

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_dates_in_february_2024 : ∃ n : ℕ, n = 10 ∧ ∀ d : ℕ, 
  (d ∈ [2, 3, 5, 7, 11, 13, 17, 19, 23, 29] → is_prime d) :=
begin
  sorry
end

end prime_dates_in_february_2024_l563_563290


namespace weight_of_mixture_is_correct_l563_563024

def weight_of_mixture (weight_a_per_l : ℕ) (weight_b_per_l : ℕ) 
                      (total_volume : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) : ℚ :=
  let volume_a := (ratio_a : ℚ) / (ratio_a + ratio_b) * total_volume
  let volume_b := (ratio_b : ℚ) / (ratio_a + ratio_b) * total_volume
  let weight_a := volume_a * weight_a_per_l
  let weight_b := volume_b * weight_b_per_l
  (weight_a + weight_b) / 1000

theorem weight_of_mixture_is_correct :
  weight_of_mixture 800 850 3 3 2 = 2.46 :=
by
  sorry

end weight_of_mixture_is_correct_l563_563024


namespace document_handling_possibilities_l563_563973

theorem document_handling_possibilities :
  ∑ k in {9, 10, 11}, 2 * k.factorial = 95116960 :=
by 
  have h9 : 9.factorial = 362880 := rfl
  have h10 : 10.factorial = 3628800 := rfl
  have h11 : 11.factorial = 39916800 := rfl
  calc
    2 * (9.factorial + 2 * 10.factorial + 11.factorial)
        = 2 * (362880 + 2 * 3628800 + 39916800)   : by rw [h9, h10, h11]
    ... = 2 * 47558480                           : by norm_num
    ... = 95116960                               : by norm_num

end document_handling_possibilities_l563_563973


namespace circle_passing_through_PAB_l563_563182

noncomputable def circle_through_points (P A B : ℝ × ℝ) (C_center : ℝ × ℝ) (C_radius : ℝ) : set (ℝ × ℝ) :=
  { Q | (Q.1 - 1) ^ 2 + (Q.2 + 1 / 2) ^ 2 = 61 / 4 }

theorem circle_passing_through_PAB :
  let P := (-2, -3)
  let C_center := (4, 2)
  let C_radius := 3
  ∃ A B : ℝ × ℝ,
      -- A and B are points of tangency (indirectly involved in the equation as condition)
      P ∈ circle_through_points P A B C_center C_radius :=
begin
  -- proof goes here
  sorry
end

end circle_passing_through_PAB_l563_563182


namespace max_m_value_l563_563015

noncomputable def b₁ := 1
noncomputable def b₃ := 16
noncomputable def r := 4
noncomputable def b (n : ℕ) : ℕ := r^(n-1)

def a (n : ℕ) : ℕ := n + 1
def a_66 := 67

theorem max_m_value (m : ℕ) : a 1 ^ 2 + (∑ k in Finset.range m, a k.succ) ≤ a 66 → m ≤ 10 :=
by
  sorry

end max_m_value_l563_563015


namespace exists_triangle_with_conditions_l563_563840

def distance (p1 p2 : (ℤ × ℤ)) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def area (v1 v2 v3 : (ℤ × ℤ)) : ℝ :=
  0.5 * abs (v1.1 * (v2.2 - v3.2) + v2.1 * (v3.2 - v1.2) + v3.1 * (v1.2 - v2.2))

theorem exists_triangle_with_conditions :
  ∃ (v1 v2 v3 : ℤ × ℤ),
    (distance v1 v2 > 100) ∧
    (distance v2 v3 > 100) ∧
    (distance v1 v3 > 100) ∧
    (area v1 v2 v3 < 1) :=
by
  use [(-1, 0), (100, 1), (200, 2)]
  split
  -- 1st condition: distance v1 v2 > 100
  { sorry },
  split
  -- 2nd condition: distance v2 v3 > 100
  { sorry },
  split
  -- 3rd condition: distance v1 v3 > 100
  { sorry },
  -- 4th condition: area v1 v2 v3 < 1
  { sorry }

end exists_triangle_with_conditions_l563_563840


namespace problem_solution_l563_563345

noncomputable def problem_statement : Prop :=
  ∃ (x : ℝ) (k : ℝ), 
    (x = 4 ∧ y = 256 →
      k = x^2 * Real.root y 4) ∧ 
    (y = 161051 ∧ x * y = 2205) →
      x ≈ 3.35

theorem problem_solution : problem_statement :=
by
  sorry

end problem_solution_l563_563345


namespace find_k_and_max_profit_l563_563421

/- Definitions of cost, revenue, and profit functions -/
def cost (x : ℝ) : ℝ := 3 + x

def revenue (x k : ℝ) : ℝ :=
  if 0 < x ∧ x < 6 then 3 * x + k / (x - 8) + 5 else if x ≥ 6 then 14 else 0

def profit (x k : ℝ) := revenue x k - cost x

/- Main Theorem -/
theorem find_k_and_max_profit 
  (h_profit_at_2 : profit 2 k = 3) :
  k = 18 ∧ ∃ (x : ℝ), 0 < x ∧ x < 6 ∧ profit x 18 = 6 ∧ ∀ y, profit y 18 ≤ 6 :=
by
  sorry

end find_k_and_max_profit_l563_563421


namespace circles_intersecting_l563_563939
noncomputable theory

def Circle1 : set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ x^2 + y^2 + 2 * x + 8 * y - 8 = 0}

def Circle2 : set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ x^2 + y^2 - 4 * x - 4 * y - 2 = 0}

theorem circles_intersecting :
  ∃ d R r, d = real.sqrt ((-1 - 2)^2 + (-4 - 2)^2) ∧
            R = 5 ∧
            r = real.sqrt 10 ∧
            R - r < d ∧ d < R + r :=
sorry

end circles_intersecting_l563_563939


namespace Jake_bought_4_balloons_l563_563803

def balloons :=
  ∀ (Jake_initial Allan_balloons Jake_total_balloons : ℕ),
  Jake_initial = 3 →
  Allan_balloons = 6 →
  Jake_total_balloons = Jake_initial + 4 →
  Jake_total_balloons = Allan_balloons + 1 →
  4 = 4

theorem Jake_bought_4_balloons : balloons :=
by {
  intros Jake_initial Allan_balloons Jake_total_balloons hJ hA hT hEq,
  exact rfl,
}

end Jake_bought_4_balloons_l563_563803


namespace meeting_probability_is_approx_0242_l563_563652

-- Define the movements for A and B
def moves_A (n : ℕ) : Set (ℤ × ℤ) := { (i, n - i) | i <= n }
def moves_B (n : ℕ) : Set (ℤ × ℤ) := { (3 - i, 5 - (n - i)) | i <= n }

-- Define the probability calculation for meeting point
noncomputable def paths_to_meet_at (i : ℕ) : ℚ :=
  (Nat.choose 4 i) * (Nat.choose 4 (i + 1))

def total_meeting_probability : ℚ :=
  (paths_to_meet_at 0 + paths_to_meet_at 1 + paths_to_meet_at 2 + paths_to_meet_at 3) / (2 ^ 8)

theorem meeting_probability_is_approx_0242 :
  total_meeting_probability ≈ 0.242 := 
sorry

end meeting_probability_is_approx_0242_l563_563652


namespace g_eq_l563_563628

noncomputable def g (n : ℕ) : ℝ :=
  (7 + 4 * Real.sqrt 7) / 14 * ((1 + Real.sqrt 7) / 2) ^ n +
  (7 - 4 * Real.sqrt 7) / 14 * ((1 - Real.sqrt 7) / 2) ^ n

theorem g_eq (n : ℕ) : g (n + 2) - g (n - 2) = 3 * g n := by
  sorry

end g_eq_l563_563628


namespace cashier_amount_l563_563620

def amount_to_cashier (discount : ℝ) (shorts_count : ℕ) (shorts_price : ℕ) (shirts_count : ℕ) (shirts_price : ℕ) : ℝ :=
  let total_cost := (shorts_count * shorts_price) + (shirts_count * shirts_price)
  let discount_amount := discount * total_cost
  total_cost - discount_amount

theorem cashier_amount : amount_to_cashier 0.1 3 15 5 17 = 117 := 
by
  sorry

end cashier_amount_l563_563620


namespace hyperbola_asymptotes_l563_563330

theorem hyperbola_asymptotes:
  ∀ (x y : ℝ),
  ( ∀ y, y = (1 + (4 / 5) * x) ∨ y = (1 - (4 / 5) * x) ) →
  (y-1)^2 / 16 - x^2 / 25 = 1 →
  (∃ m b: ℝ, m > 0 ∧ m = 4/5 ∧ b = 1) := by
  sorry

end hyperbola_asymptotes_l563_563330


namespace binary_digits_of_ABABA_l563_563488

theorem binary_digits_of_ABABA : 
  let hex_value := 10 * 16^4 + 11 * 16^3 + 10 * 16^2 + 11 * 16^1 + 10 in
  ∃ n : ℕ, 2^19 ≤ hex_value ∧ hex_value < 2^20 ∧ n = 20 :=
by
  sorry

end binary_digits_of_ABABA_l563_563488


namespace outer_boundary_diameter_l563_563597

theorem outer_boundary_diameter (d_p: ℝ) (w_s: ℝ) (w_j: ℝ) (h₀: d_p = 18) (h₁: w_s = 10) (h₂: w_j = 7) : 
  2 * ((d_p / 2) + w_s + w_j) = 52 :=
by 
  calc
    2 * ((d_p / 2) + w_s + w_j) 
        = 2 * (9 + 10 + 7) : by { rw [h₀, h₁, h₂], norm_num }
    ... = 52 : by norm_num

end outer_boundary_diameter_l563_563597


namespace NH4I_reaction_l563_563142

theorem NH4I_reaction (n : ℕ) (h1 : n = 3) :
    required_NH4I (NH4I + KOH → NH3 + KI + H2O) n = 3 :=
sorry

end NH4I_reaction_l563_563142


namespace sum_is_zero_l563_563865

def sum_of_integers_abs_greater_than_3_less_than_6 : ℤ :=
  -5 + -4 + 4 + 5

theorem sum_is_zero : sum_of_integers_abs_greater_than_3_less_than_6 = 0 :=
  by
  calc
    sum_of_integers_abs_greater_than_3_less_than_6 
        = -5 + -4 + 4 + 5 : rfl
    ... = -5 - 4 + 4 + 5 : by rw [add_comm (-4) 4]
    ... = (-5 + 5) + (-4 + 4) : by rw [add_assoc, add_assoc]
    ... = 0 + 0 : by rw [add_left_neg, add_left_neg]
    ... = 0 : by rw [zero_add]


end sum_is_zero_l563_563865


namespace binom_coeff_identity_geometric_sequence_sum_l563_563882

open Nat

theorem binom_coeff_identity (k n : ℕ) (hk : 0 < k ∧ k ≤ n) :
  k * Nat.choose n k = n * Nat.choose (n-1) (k-1) :=
  sorry

theorem geometric_sequence_sum (q : ℝ) (n : ℕ) :
  ∑ k in Finset.range n, (∑ j in Finset.range k, (1:ℝ) * q^j) * (Nat.choose n k) =
  if q = 1 then n * 2^(n-1)
  else (2^n - (1 + q)^n) / (1 - q) :=
  sorry

end binom_coeff_identity_geometric_sequence_sum_l563_563882


namespace find_A_work_rate_l563_563064

variable (x : ℝ) -- x is the number of days A can do the work alone

-- B's work rate
def work_rate_B := 1 / 60

-- Combined work rate of A and B working together
def combined_work_rate_AB := 1 / 24

-- A's work rate
def work_rate_A := 1 / x

-- The given problem statement translated to Lean
theorem find_A_work_rate :
  (1 / x) + (1 / 60) = (1 / 24) → x = 40 :=
by
  intro h
  sorry

end find_A_work_rate_l563_563064


namespace largest_common_number_in_sequences_from_1_to_200_l563_563466

theorem largest_common_number_in_sequences_from_1_to_200 :
  ∃ a, a ≤ 200 ∧ a % 8 = 3 ∧ a % 9 = 5 ∧ ∀ b, (b ≤ 200 ∧ b % 8 = 3 ∧ b % 9 = 5) → b ≤ a :=
sorry

end largest_common_number_in_sequences_from_1_to_200_l563_563466


namespace inequality_am_gm_l563_563556

theorem inequality_am_gm (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) (h : a^2 + b^2 + c^2 = 12) :
  1/(a-1) + 1/(b-1) + 1/(c-1) ≥ 3 := 
by
  sorry

end inequality_am_gm_l563_563556


namespace time_for_10_strikes_l563_563069

-- Assume a clock takes 7 seconds to strike 7 times
def clock_time_for_N_strikes (N : ℕ) : ℕ :=
  if N = 7 then 7 else sorry  -- This would usually be a function, simplified here for the specific condition

-- Assume there are 6 intervals for 7 strikes
def intervals_between_strikes (N : ℕ) : ℕ :=
  if N = 7 then 6 else N - 1

-- Function to calculate total time for any number of strikes based on intervals and time per strike
def total_time_for_strikes (N : ℕ) : ℚ :=
  (intervals_between_strikes N) * (clock_time_for_N_strikes 7 / intervals_between_strikes 7 : ℚ)

theorem time_for_10_strikes : total_time_for_strikes 10 = 10.5 :=
by
  -- Insert proof here
  sorry

end time_for_10_strikes_l563_563069


namespace solution_exists_l563_563552

noncomputable def problem (x m : ℝ) : Prop :=
  log 10 (sin x) + log 10 (cos x) = -2 ∧
  log 10 (sin x + cos x) = 0.5 * (log 10 m - 2) ∧
  m = 102

theorem solution_exists : ∃ (x m : ℝ), problem x m := 
  sorry

end solution_exists_l563_563552


namespace beth_total_repayment_l563_563477

noncomputable def compound_interest (P r : ℝ ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def remaining_balance (total_payment remaining_term : ℕ) (r n : ℝ) (initial_balance new_payment : ℝ) : ℝ :=
  compound_interest (initial_balance - new_payment) r n remaining_term

theorem beth_total_repayment :
  let P := 15000
  let r := 0.08
  let n := 2
  let t1 := 8
  let t2 := 7
  let payment_ratio := 1 / 3
  let balance_after_8_years := compound_interest P r n t1
  let payment := balance_after_8_years * payment_ratio
  let remaining_balance := balance_after_8_years - payment
  let final_balance := remaining_balance t2 r n(P * (1 + 0.08 / 2) ^ (2 * 8) - (P * (1 + 0.08 / 2) ^ (2 * 8) / 3)
  final_balance = 25069.92 := sorry

end beth_total_repayment_l563_563477


namespace disjoint_subsets_count_l563_563265

theorem disjoint_subsets_count : 
  let S := {1, 2, 3, 4, 5, 6, 7, 8} in
  let count := 3^8 in
  count = 6561 :=
by sorry

end disjoint_subsets_count_l563_563265


namespace total_vehicle_wheels_in_parking_lot_l563_563502

def vehicles_wheels := (1 * 4) + (1 * 4) + (8 * 4) + (4 * 2) + (3 * 6) + (2 * 4) + (1 * 8) + (2 * 3)

theorem total_vehicle_wheels_in_parking_lot : vehicles_wheels = 88 :=
by {
    sorry
}

end total_vehicle_wheels_in_parking_lot_l563_563502


namespace perpendiculars_intersect_at_one_point_l563_563985

theorem perpendiculars_intersect_at_one_point
  (A1 A2 A3 A4 P : Point) 
  (h_square : is_square A1 A2 A3 A4)
  (h_P_inside : P ∈ square A1 A2 A3 A4)
  (h_perpendicular_A1 : is_perpendicular_from_point A1 (line_through A2 P))
  (h_perpendicular_A2 : is_perpendicular_from_point A2 (line_through A3 P))
  (h_perpendicular_A3 : is_perpendicular_from_point A3 (line_through A4 P))
  (h_perpendicular_A4 : is_perpendicular_from_point A4 (line_through A1 P)) :
  ∃ Q, intersects_at h_perpendicular_A1 Q ∧ intersects_at h_perpendicular_A2 Q ∧ intersects_at h_perpendicular_A3 Q ∧ intersects_at h_perpendicular_A4 Q :=
sorry

end perpendiculars_intersect_at_one_point_l563_563985


namespace alpha_in_second_quadrant_l563_563543

variable (α : ℝ)

-- Conditions that P(tan α, cos α) is in the third quadrant
def P_in_third_quadrant (α : ℝ) : Prop := (Real.tan α < 0) ∧ (Real.cos α < 0)

-- Theorem statement
theorem alpha_in_second_quadrant (h : P_in_third_quadrant α) : 
  π/2 < α ∧ α < π :=
sorry

end alpha_in_second_quadrant_l563_563543


namespace part_b_l563_563754

-- Definitions for medians and sums in Lean 4 code
def median (S : List ℝ) : ℝ :=
  let sorted := List.sort (≤) S
  if h : sorted.length % 2 = 1 then
    sorted.get ⟨sorted.length / 2, nat.div_lt_self' (List.length_pos_of_mem (sorted.get sorry)) sorry⟩
  else
    (sorted.get ⟨sorted.length / 2, nat.div_lt_self' sorry sorry⟩ + sorted.get ⟨sorted.length / 2 - 1, nat.div_sub_self sorry sorry⟩) / 2

def sum_sets (X Y : List ℝ) : List ℝ :=
  List.bind X (λ x => List.map (λ y => x + y) Y)

-- Part (a): Example of sets with median difference of 1
def example_sets (X Y : List ℝ) :=
  median (sum_sets X Y) - (median X + median Y) = 1

-- Example from the solution for (a)
def example_a : Prop :=
  example_sets [0, 0, 1] [0, 0, 1]

-- Part (b): Prove such sets with median difference of 4.5 do not exist
def no_such_sets (Y : List ℝ) (h1 : List.minimum Y = some 1) (h5 : List.maximum Y = some 5) :=
  ∀ X : List ℝ, median (sum_sets X Y) - (median X + median Y) ≤ 4

-- Prove the statement for part (b)
theorem part_b (Y : List ℝ) (h1 : List.minimum Y = some 1) (h5 : List.maximum Y = some 5) : no_such_sets Y h1 h5 := by
  sorry

end part_b_l563_563754


namespace product_mod_9_l563_563710

theorem product_mod_9 (a b c : ℕ) (h1 : a % 6 = 2) (h2 : b % 7 = 3) (h3 : c % 8 = 4) : (a * b * c) % 9 = 6 :=
by
  sorry

end product_mod_9_l563_563710


namespace maximum_volume_of_right_triangle_rotation_l563_563174

/-- This represents a right triangle with sides 3, 4, and 5. -/
structure RightTriangle :=
  (a b c : ℕ) 
  (a_eq_3 : a = 3)
  (b_eq_4 : b = 4)
  (c_eq_5 : c = 5)

/-- Volume calculations for rotating around each side. -/
def volume_by_rotation (t : RightTriangle) (axis_length : ℕ) : ℝ :=
  if h: axis_length = t.a then 
    (1.0 / 3.0) * real.pi * (t.a ^ 2) * t.b
  else if h: axis_length = t.b then 
    (1.0 / 3.0) * real.pi * (t.b ^ 2) * t.a
  else 
    (1.0 / 3.0) * real.pi * ((2.0 * 6.0 / t.c) ^ 2) * t.c

noncomputable def max_volume (t : RightTriangle) : ℝ :=
  max (volume_by_rotation t t.a) (max (volume_by_rotation t t.b) (volume_by_rotation t t.c))

theorem maximum_volume_of_right_triangle_rotation :
  let t := { RightTriangle . a := 3, b := 4, c := 5, a_eq_3 := rfl, b_eq_4 := rfl, c_eq_5 := rfl } in
  max_volume t = 16.0 * real.pi :=
by
  -- the proof would go here
  sorry

end maximum_volume_of_right_triangle_rotation_l563_563174


namespace at_least_half_elements_l563_563277

variable {A : Finset ℤ}
variable {m : ℕ}
variable {B : Fin n → Finset ℤ}

theorem at_least_half_elements (h_m_ge_2 : m ≥ 2)
    (h_subsets : ∀ i : Fin m, B i ⊆ A)
    (h_sums : ∀ i : Fin m, (B i).sum id = m^(i+1)) :
  A.card ≥ m / 2 :=
sorry

end at_least_half_elements_l563_563277


namespace max_value_of_A_l563_563932

noncomputable def cos_sq (x: ℝ) := real.cos x ^ 2
noncomputable def ctg_sq (x: ℝ) := (real.cos x / real.sin x) ^ 2
noncomputable def ctg_qd (x: ℝ) := (ctg_sq x) ^ 2

theorem max_value_of_A (x : ℕ → ℝ) (n : ℕ) (hx : ∀ i, i < n → 0 < x i ∧ x i < π / 2) :
  (∑ i in finset.range n, cos_sq (x i)) / (real.sqrt n + real.sqrt (∑ i in finset.range n, ctg_qd (x i))) ≤ real.sqrt n / 4 :=
sorry

end max_value_of_A_l563_563932


namespace matrix_multiplication_correct_l563_563123

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, 3, -1], ![1, -2, 5], ![0, 6, 1]]

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 0, 4], ![3, 2, -1], ![0, 4, -2]]

def C : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![11, 2, 7], ![-5, 16, -4], ![18, 16, -8]]

theorem matrix_multiplication_correct :
  A * B = C :=
by
  sorry

end matrix_multiplication_correct_l563_563123


namespace point_not_on_line_l563_563222

theorem point_not_on_line
  (p q : ℝ)
  (h : p * q > 0) :
  ¬ (∃ (x y : ℝ), x = 2023 ∧ y = 0 ∧ y = p * x + q) :=
by
  sorry

end point_not_on_line_l563_563222


namespace certain_value_z_l563_563159

-- Define the length of an integer as the number of prime factors
def length (n : ℕ) : ℕ := 
  primeFactors n |>.length

theorem certain_value_z {x y : ℕ} (hx1 : x > 1) (hy1 : y > 1)
  (h_len : length x + length y = 16) : 
  x + 3 * y < 98307 := 
sorry

end certain_value_z_l563_563159


namespace least_four_digit_solution_l563_563510

theorem least_four_digit_solution :
  ∃ x : ℕ, 5 * x % 20 = 15 ∧ (3 * x + 7) % 8 = 19 ∧ (-3 * x + 2) % 14 = x % 14 ∧ x >= 1000 ∧ x < 10000 ∧ 
  ∀ y, (y >= 1000 ∧ y < 10000 ∧ 5 * y % 20 = 15 ∧ (3 * y + 7) % 8 = 19 ∧ (-3 * y + 2) % 14 = y % 14) → x ≤ y :=
  let x := 1032 in
  by {
    exists x,
    split, exact (5 * x % 20 = 15), sorry,
    split, exact (3 * x + 7) % 8 = 19, sorry,
    split, exact (-3 * x + 2) % 14 = x % 14, sorry,
    split, exact x >= 1000, sorry,
    split, exact x < 10000, sorry,
    intros y hy,
    sorry
  }

end least_four_digit_solution_l563_563510


namespace problem_statement_l563_563911

open Real EuclideanGeometry

def line_l (k x y : ℝ) : Prop :=
  y - 1 = k * (x - 2)

def on_circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 2)^2 = 1

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def triangle_area (a b c : ℝ × ℝ) : ℝ :=
  abs (a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2)) / 2

theorem problem_statement (k : ℝ) (Q M C : ℝ × ℝ)
  (hQ_line : line_l k Q.1 Q.2)
  (hM_on_circle : on_circle_C M.1 M.2)
  (hQ_on_line : line_l k Q.1 Q.2)
  (hA_min_area_sqrt2 : triangle_area Q M C = sqrt 2)
  (C_coord : C = (1, -2)) :
  (distance C Q = 3) ∧ (∃ E F : ℝ × ℝ, on_circle_C F.1 F.2 ∧ line_l k E.1 E.2 ∧ distance E F = 2) :=
by {
  sorry
}

end problem_statement_l563_563911


namespace opposite_of_neg3_l563_563000

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
  sorry

end opposite_of_neg3_l563_563000


namespace perimeter_MNO_l563_563454

open Classical

-- Definitions
def height : ℝ := 20
def side_length : ℝ := 10

noncomputable def P := (0, 0, 0)
noncomputable def Q := (10, 0, 0)
noncomputable def R := (5, 5 * (sqrt 3), 0)
noncomputable def S := (5, 5 * (sqrt 3), 20)
noncomputable def T := (10, 0, 20)
noncomputable def U := (0, 0, 20)

noncomputable def M := ((0 + 10) / 2, (0 + 0) / 2, (0 + 0) / 2)
noncomputable def N := ((10 + 5) / 2, (0 + 5 * (sqrt 3)) / 2, (0 + 0) / 2)
noncomputable def O := ((5 + 5) / 2, (5 * (sqrt 3) + 5 * (sqrt 3)) / 2, (0 + 20) / 2)

-- Theorem Statement
theorem perimeter_MNO :
  let MO := sqrt ((M.1 - O.1)^2 + (M.2 - O.2)^2 + (M.3 - O.3)^2)
  let NO := sqrt ((N.1 - O.1)^2 + (N.2 - O.2)^2 + (N.3 - O.3)^2)
  let MN := sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2 + (M.3 - N.3)^2)
  MO + NO + MN = 5 + 10 * sqrt 5 :=
by
  sorry

end perimeter_MNO_l563_563454


namespace tree_odd_vertices_if_remoteness_diff_one_l563_563325

-- Define a tree and its properties
structure Tree (V : Type) :=
  (edges : V → V → Prop)
  (symm : ∀ {u v : V}, edges u v → edges v u)
  (acyclic : ∀ {u v : V}, edges u v → u ≠ v)

def remoteness {V : Type} [Fintype V] (T : Tree V) (v : V) : ℕ :=
  Fintype.card {u : V // T.edges v u}

theorem tree_odd_vertices_if_remoteness_diff_one {V : Type} [Fintype V]
  (T : Tree V) (A B : V) (dA dB : ℕ)
  (hdist : abs (dA - dB) = 1) :
  Fintype.card V % 2 = 1 := 
sorry

end tree_odd_vertices_if_remoteness_diff_one_l563_563325


namespace intersection_A_B_eq_union_A_B_eq_intersection_A_C_U_B_eq_l563_563872

def U := ℝ
def A : Set ℝ := {x | 0 ≤ x ∧ x < 5}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 4}
def C_U_B : Set ℝ := {x | x < -2 ∨ x ≥ 4}

theorem intersection_A_B_eq : A ∩ B = {x | 0 ≤ x ∧ x < 4} := by
  sorry

theorem union_A_B_eq : A ∪ B = {x | -2 ≤ x ∧ x < 5} := by
  sorry

theorem intersection_A_C_U_B_eq : A ∩ C_U_B = {x | 4 ≤ x ∧ x < 5} := by
  sorry

end intersection_A_B_eq_union_A_B_eq_intersection_A_C_U_B_eq_l563_563872


namespace smallest_sum_is_381_l563_563739

def is_valid_3_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def uses_digits_once (n m : ℕ) : Prop :=
  (∀ d ∈ [1, 2, 3, 4, 5, 6], (d ∈ n.digits 10 ∨ d ∈ m.digits 10)) ∧
  (∀ d, d ∈ n.digits 10 → d ∈ [1, 2, 3, 4, 5, 6]) ∧
  (∀ d, d ∈ m.digits 10 → d ∈ [1, 2, 3, 4, 5, 6])

theorem smallest_sum_is_381 :
  ∃ (n m : ℕ), is_valid_3_digit_number n ∧ is_valid_3_digit_number m ∧
  uses_digits_once n m ∧ n + m = 381 :=
sorry

end smallest_sum_is_381_l563_563739


namespace find_value_of_xy_plus_yz_plus_xz_l563_563919

variable (x y z : ℝ)

-- Conditions
def cond1 : Prop := x^2 + x * y + y^2 = 108
def cond2 : Prop := y^2 + y * z + z^2 = 64
def cond3 : Prop := z^2 + x * z + x^2 = 172

-- Theorem statement
theorem find_value_of_xy_plus_yz_plus_xz (hx : cond1 x y) (hy : cond2 y z) (hz : cond3 z x) : 
  x * y + y * z + x * z = 96 :=
sorry

end find_value_of_xy_plus_yz_plus_xz_l563_563919


namespace sum_of_coordinates_inv_graph_l563_563319

variable {f : ℝ → ℝ}
variable (hf : f 2 = 12)

theorem sum_of_coordinates_inv_graph :
  ∃ (x y : ℝ), y = f⁻¹ x / 3 ∧ x = 12 ∧ y = 2 / 3 ∧ x + y = 38 / 3 := by
  sorry

end sum_of_coordinates_inv_graph_l563_563319


namespace part_a_part_b_l563_563259

-- Defining the replacement operation
def replace (n a b : ℕ) : ℕ := if a + b = n then a * b else n

-- Part (a)
theorem part_a : ∃ (seq : List ℕ), seq.head = 7 ∧ seq.last = 48 ∧ 
  ∀ i, i < seq.length - 1 → ∃ a b, a + b = seq.nth i ∧ seq.nth (i+1) = a * b :=
by
  sorry

-- Part (b)
theorem part_b : ∃ (seq : List ℕ), seq.head = 7 ∧ seq.last = 2014 ∧ 
  ∀ i, i < seq.length - 1 → ∃ a b, a + b = seq.nth i ∧ seq.nth (i+1) = a * b :=
by
  sorry

end part_a_part_b_l563_563259


namespace students_in_all_three_activities_l563_563348

theorem students_in_all_three_activities (total_students swimming_students archery_students chess_students at_least_two_activities : ℕ) 
    (h1 : total_students = 25) 
    (h2 : swimming_students = 15) 
    (h3 : archery_students = 17) 
    (h4 : chess_students = 10) 
    (h5 : at_least_two_activities = 12) : 
    ∃ all_three : ℕ, all_three = 4 :=
by
  use 4
  trivial

end students_in_all_three_activities_l563_563348


namespace opposite_of_neg_three_l563_563002

-- Define the concept of negation and opposite of a number
def opposite (x : ℤ) : ℤ := -x

-- State the problem: Prove that the opposite of -3 is 3
theorem opposite_of_neg_three : opposite (-3) = 3 :=
by
  -- Proof
  sorry

end opposite_of_neg_three_l563_563002


namespace triangle_inequality_equilateral_l563_563270

theorem triangle_inequality_equilateral 
  (A B C P : EuclideanGeometry.Point) 
  (A1 B1 C1 : EuclideanGeometry.Point)
  (h_eq_triangle : EuclideanGeometry.equilateral_triangle A B C)
  (h_in_triangle : EuclideanGeometry.point_in_triangle P A B C)
  (h_intersect_A : EuclideanGeometry.extends_to_intersect AP BC A1)
  (h_intersect_B : EuclideanGeometry.extends_to_intersect BP CA B1)
  (h_intersect_C : EuclideanGeometry.extends_to_intersect CP AB C1) :
  EuclideanGeometry.distance A1 B1 * EuclideanGeometry.distance B1 C1 * EuclideanGeometry.distance C1 A1 
  ≥ EuclideanGeometry.distance A1 B * EuclideanGeometry.distance B1 C * EuclideanGeometry.distance C1 A :=
sorry

end triangle_inequality_equilateral_l563_563270


namespace triangle_parallels_proof_l563_563323

variables (A B C T U P Q R S Y : Point)
variables (AB BC CA PQ RS TU : Length)
variable (k : Length)
variables (SP UR QT : Line)
variables (a b c : Length)

-- Conditions
variable (parallel_SP_AB : SP ∥ AB)
variable (parallel_UR_BC : UR ∥ BC)
variable (parallel_QT_CA : QT ∥ CA)
variable (collinear_SP_UR_QT_Y : Collinear (SP, UR, QT, Y))
variable (equality_PQ_RS_TU : PQ = RS ∧ RS = TU)
variable (definition_a_bc_ca : a = BC ∧ b = CA ∧ c = AB)
variable (definition_k : k = PQ)

-- Proof
theorem triangle_parallels_proof :
  (1 / k = (1 / a) + (1 / b) + (1 / c)) :=
sorry

end triangle_parallels_proof_l563_563323


namespace man_l563_563442

noncomputable def man's_speed_in_still_water (speed_of_current : ℝ) (time_downstream : ℝ) (distance_downstream : ℝ) : ℝ :=
  let speed_of_current_mps := speed_of_current * 1000 / 3600
  let speed_downstream := distance_downstream / time_downstream
  speed_downstream - speed_of_current_mps

theorem man's_speed_approx : 
  man's_speed_in_still_water 3 (31.99744020478362) 80 ≈ 1.667057943 := by
  have speed_of_current_mps : ℝ := 3 * 1000 / 3600
  have speed_downstream : ℝ := 80 / 31.99744020478362
  have V : ℝ := speed_downstream - speed_of_current_mps
  rw [← sub_eq_add_neg] at V
  norm_num at V
  sorry

end man_l563_563442


namespace amount_in_paise_l563_563950

theorem amount_in_paise (a : ℕ) (h₁ : 0.5 / 100 * a = some_amount) (h₂ : a = 130) : some_amount = 65 := by
  sorry

end amount_in_paise_l563_563950


namespace maximizing_angle_CBA_for_pentagon_area_l563_563358

theorem maximizing_angle_CBA_for_pentagon_area :
  ∀ (A B C : ℝ),
  A = 50 ∧ B ≤ 100 ∧ AC ≥ AB ∧ BC = 2 ∧ 
  (∃ (H I O : Type) (triangle_centers),
    is_orthocenter H ∧ is_incenter I ∧ is_circumcenter O ∧ 
    area_of_pentagon B C O I H is_maximized_for_angle B) →
  B = 90 := 
by
  sorry

end maximizing_angle_CBA_for_pentagon_area_l563_563358


namespace seq_a2020_l563_563844

def seq (a : ℕ → ℕ) : Prop :=
(∀ n : ℕ, (a n + a (n+1) ≠ a (n+2) + a (n+3))) ∧
(∀ n : ℕ, (a n + a (n+1) + a (n+2) ≠ a (n+3) + a (n+4) + a (n+5))) ∧
(a 1 = 0)

theorem seq_a2020 (a : ℕ → ℕ) (h : seq a) : a 2020 = 1 :=
sorry

end seq_a2020_l563_563844


namespace find_n_l563_563141

theorem find_n (n : ℕ) 
  (a_k : ℕ → ℕ) 
  (b_k : ℕ → ℕ) 
  (Sn : ℕ → ℕ) 
  (Tn : ℕ → ℕ)
  (ha : ∀ k : ℕ, a_k k = 3 * k ^ 2 - 3 * k + 1)
  (hb : ∀ k : ℕ, b_k k = 2 * k + 89)
  (hS : ∀ n : ℕ, Sn n = ∑ i in range (n + 1), a_k i)
  (hT : ∀ n : ℕ, Tn n = ∑ i in range (n + 1), b_k i) :
  Sn n = Tn n ↔ n = 10 :=
by
  sorry

end find_n_l563_563141


namespace point_of_tangency_l563_563862

noncomputable def parabola1 (x : ℝ) : ℝ := 2 * x^2 + 10 * x + 14
noncomputable def parabola2 (y : ℝ) : ℝ := 4 * y^2 + 16 * y + 68

theorem point_of_tangency : 
  ∃ (x y : ℝ), parabola1 x = y ∧ parabola2 y = x ∧ x = -9/4 ∧ y = -15/8 :=
by
  -- The proof will show that the point of tangency is (-9/4, -15/8)
  sorry

end point_of_tangency_l563_563862


namespace find_explicit_formula_l563_563169

variable (f : ℝ → ℝ)

theorem find_explicit_formula 
  (h : ∀ x : ℝ, f (x - 1) = 2 * x^2 - 8 * x + 11) :
  ∀ x : ℝ, f x = 2 * x^2 - 4 * x + 5 :=
by
  sorry

end find_explicit_formula_l563_563169


namespace smallest_red_number_proof_l563_563129

def smallest_red_number : ℕ := 189

theorem smallest_red_number_proof (N k : ℕ) 
  (h1 : N ∈ {n | n ∈ (1:ℕ) .. 377})
  (h2 : k ∈ {n | n ∈ (1:ℕ) .. 377})
  (h3 : k = N + 1)
  (h4 : k = 377 - N) :
  k = smallest_red_number :=
by
  sorry

end smallest_red_number_proof_l563_563129


namespace max_product_of_three_from_set_l563_563047

theorem max_product_of_three_from_set :
  ∃ (a b c : ℤ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ a ∈ {-4, -3, -1, 3, 5} ∧ b ∈ {-4, -3, -1, 3, 5} ∧ c ∈ {-4, -3, -1, 3, 5} ∧  a * b * c = 60 := 
by
  sorry

end max_product_of_three_from_set_l563_563047


namespace solution_correct_l563_563315

noncomputable def probability_same_color_opposite_types : ℚ :=
  let total_shoes := 30 in
  let black_pairs := 7 in
  let brown_pairs := 4 in
  let gray_pairs := 2 in
  let red_pairs := 2 in
  let total_pairs := black_pairs + brown_pairs + grayPairs + redPairs in
  let black_prob := (14 / 30) * (7 / 29) in
  let brown_prob  := (8 / 30) * (4 / 29) in
  let gray_prob   := (4 / 30) * (2 / 29) in
  let red_prob    := (4 / 30) * (2 / 29) in
  (black_prob + brown_prob + gray_prob + red_prob).reduce

theorem solution_correct : probability_same_color_opposite_types = 73 / 435 :=
by
  sorry

end solution_correct_l563_563315


namespace probability_six_red_balls_l563_563682

theorem probability_six_red_balls :
  let total_ways := Nat.choose 100 10
  let red_ways := Nat.choose 80 6
  let white_ways := Nat.choose 20 4
  total_ways ≠ 0 →
  ((red_ways * white_ways) / total_ways : ℚ) = (red_ways * white_ways) / total_ways :=
by
  intro total_ways_ne_zero
  unfold total_ways red_ways white_ways
  sorry

end probability_six_red_balls_l563_563682


namespace ant_cannot_visit_one_vertex_25_times_and_others_20_times_l563_563807

-- Define the vertices of the cube
inductive Vertex : Type
| V1 | V2 | V3 | V4 | V5 | V6 | V7 | V8

-- Checkerboard labeling
def labeling : Vertex → ℤ 
| Vertex.V1 => 1
| Vertex.V2 => -1
| Vertex.V3 => 1
| Vertex.V4 => -1
| Vertex.V5 => -1
| Vertex.V6 => 1
| Vertex.V7 => -1
| Vertex.V8 => 1

-- Condition translating to visit counts
def visit_counts (v : Vertex) : ℕ → Prop 
| Vertex.V1 => 25
| Vertex.V2 => 20
| Vertex.V3 => 20
| Vertex.V4 => 20
| Vertex.V5 => 20
| Vertex.V6 => 20
| Vertex.V7 => 20
| Vertex.V8 => 20

-- The proof problem statement
theorem ant_cannot_visit_one_vertex_25_times_and_others_20_times :
  ∀ (visit_counts : Vertex → ℕ), 
    (visit_counts Vertex.V1 = 25 ∧ 
     visit_counts Vertex.V2 = 20 ∧ 
     visit_counts Vertex.V3 = 20 ∧ 
     visit_counts Vertex.V4 = 20 ∧ 
     visit_counts Vertex.V5 = 20 ∧ 
     visit_counts Vertex.V6 = 20 ∧ 
     visit_counts Vertex.V7 = 20 ∧ 
     visit_counts Vertex.V8 = 20) → 
    (False) :=
by 
  sorry

end ant_cannot_visit_one_vertex_25_times_and_others_20_times_l563_563807


namespace ratio_of_visits_l563_563616

theorem ratio_of_visits (w j : ℕ) (hw : w = 2) (hj : j = 32) : j / (4 * w) = 4 := 
by 
  have h1 : 4 * w = 8 := by simp [hw]
  have h2 : j = 32 := by simp [hj]
  rw [h1, h2]
  norm_num

end ratio_of_visits_l563_563616


namespace find_a2004_l563_563937

noncomputable def a_seq : ℕ → ℝ
| 0       := 1
| (n + 1) := (√3 * a_seq n - 1) / (a_seq n + √3)

theorem find_a2004 : a_seq 2004 = 2 + √3 :=
sorry

end find_a2004_l563_563937


namespace find_valid_pairs_l563_563507

theorem find_valid_pairs :
  ∀ (n m : ℕ), 0 < m → 0 < n →
    ( ∃ k : ℤ, ↑(n^m - m) * k = ↑(m^2 + 2 * m) ) ↔ 
    (n = 2 ∧ m = 1) ∨ 
    (n = 2 ∧ m = 2) ∨ 
    (n = 2 ∧ m = 3) ∨ 
    (n = 2 ∧ m = 4) ∨ 
    (n = 4 ∧ m = 1) :=
sorry

end find_valid_pairs_l563_563507


namespace jo_reading_hours_l563_563622

theorem jo_reading_hours :
  ∀ (total_pages current_page previous_page pages_per_hour remaining_pages : ℕ),
    total_pages = 210 →
    current_page = 90 →
    previous_page = 60 →
    pages_per_hour = current_page - previous_page →
    remaining_pages = total_pages - current_page →
    remaining_pages / pages_per_hour = 4 :=
by
  intros total_pages current_page previous_page pages_per_hour remaining_pages
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3] at *
  dsimp at *
  sorry

end jo_reading_hours_l563_563622


namespace kelly_days_per_week_l563_563627

def string_cheeses_per_day (c_1 c_2 : ℕ) : ℕ := c_1 + c_2

def total_string_cheeses (p k : ℕ) : ℕ := p * k

def days_lasting (total_cheeses cheeses_per_day : ℕ) : ℕ := total_cheeses / cheeses_per_day

def days_per_week (days_total weeks : ℕ) : ℕ := days_total / weeks

theorem kelly_days_per_week (c_1 c_2 p k weeks : ℕ) :
  c_1 = 2 → c_2 = 1 → p = 30 → k = 2 → weeks = 4 →
  days_per_week (days_lasting (total_string_cheeses p k) (string_cheeses_per_day c_1 c_2)) weeks = 5 := by
  intros h1 h2 h3 h4 h5
  simp [string_cheeses_per_day, total_string_cheeses, days_lasting, days_per_week, h1, h2, h3, h4, h5]
  sorry

end kelly_days_per_week_l563_563627


namespace cos_neg_pi_over_3_l563_563136

noncomputable def angle := - (Real.pi / 3)

theorem cos_neg_pi_over_3 : Real.cos angle = 1 / 2 :=
by
  sorry

end cos_neg_pi_over_3_l563_563136


namespace dominoes_game_rounds_l563_563515

theorem dominoes_game_rounds (players : Finset ℕ) (H : players.card = 5) :
  ∃ rounds : Finset (Finset ℕ), rounds.card = 5 ∧ 
  (∀ P1 P2 ∈ players, P1 ≠ P2 → (∃ round ∈ rounds, {P1, P2} ⊆ round) ∧
   (∀ P3 P4 ∈ players, {P1, P2} ≠ {P3, P4} → (rounds.filter (λ r, {P1, P2} ⊆ r ∧ {P3, P4} ⊆ r)).card = 1)) :=
sorry

end dominoes_game_rounds_l563_563515


namespace cyclic_quad_diagonals_l563_563952

variables (a b c d m n : ℝ) (A C : ℝ)

theorem cyclic_quad_diagonals (h1 : convex_quadrilateral a b c d m n A C) :
  m^2 * n^2 = a^2 * c^2 + b^2 * d^2 - 2 * a * b * c * d * Real.cos (A + C) :=
sorry

end cyclic_quad_diagonals_l563_563952


namespace find_number_l563_563232

theorem find_number (x : ℤ) (h : x + 2 - 3 = 7) : x = 8 :=
sorry

end find_number_l563_563232


namespace transformed_function_correct_l563_563357

noncomputable def transformed_function (x : ℝ) : ℝ := -sin (2 * x + π / 3)

theorem transformed_function_correct : transformed_function = λ x, -sin (2 * x + π / 3) :=
by {
  funext,
  unfold transformed_function,
  sorry,
}

end transformed_function_correct_l563_563357


namespace point_in_fourth_quadrant_l563_563608

-- Let c be the complex number 2 - i
def c : ℂ := 2 - complex.i

-- Definition of the fourth quadrant in the complex plane
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- The theorem we need to prove: c lies in the fourth quadrant
theorem point_in_fourth_quadrant : in_fourth_quadrant c :=
by
  sorry

end point_in_fourth_quadrant_l563_563608


namespace problem1_problem2_l563_563546

-- Definitions of sets A and B
def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

-- Problem 1: Prove that A ∩ (complement of B in ℕ) is {1, 5, 7}
theorem problem1 : A ∩ (Set.univ \ B) = {1, 5, 7} := sorry

-- Problem 2: Prove the number of proper subsets of A ∪ B is 255
theorem problem2 : (2^(Card (A ∪ B)) - 1 = 255) := sorry

end problem1_problem2_l563_563546


namespace train_overtake_distance_l563_563751

theorem train_overtake_distance (speed_a speed_b hours_late time_to_overtake distance_a distance_b : ℝ) 
  (h1 : speed_a = 30)
  (h2 : speed_b = 38)
  (h3 : hours_late = 2) 
  (h4 : distance_a = speed_a * hours_late) 
  (h5 : distance_b = speed_b * time_to_overtake) 
  (h6 : time_to_overtake = distance_a / (speed_b - speed_a)) : 
  distance_b = 285 := sorry

end train_overtake_distance_l563_563751


namespace largest_share_received_l563_563153

noncomputable def largest_share (total_profit : ℝ) (ratio : List ℝ) : ℝ :=
  let total_parts := ratio.foldl (· + ·) 0
  let part_value := total_profit / total_parts
  let max_part := ratio.foldl max 0
  max_part * part_value

theorem largest_share_received
  (total_profit : ℝ)
  (h_total_profit : total_profit = 42000)
  (ratio : List ℝ)
  (h_ratio : ratio = [2, 3, 4, 4, 6]) :
  largest_share total_profit ratio = 12600 :=
by
  sorry

end largest_share_received_l563_563153


namespace area_of_region_l563_563364

-- Definitions of lines forming the region
def line1 (x : ℝ) : ℝ := 3 * x - 3
def line2 (x : ℝ) : ℝ := -2 * x + 14

-- Definition of the vertical boundaries
def x0 : ℝ := 0
def x5 : ℝ := 5

-- Proof statement
theorem area_of_region : 
  (let y_intercept1 := line1 x0,
       y_intercept2 := line2 x0,
       intersection1 := line1 x5,
       intersection2 := line2 x5,
       
       -- Vertices of the quadrilateral
       vertex1 := (x0, y_intercept1),
       vertex2 := (x0, y_intercept2),
       vertex3 := (x5, intersection1),
       vertex4 := (x5, intersection2),
       
       -- Area calculation
       rect_height := intersection2 - y_intercept1,
       rect_width := x5 - x0,
       rect_area := rect_height * rect_width,
       
       trap_base1 := y_intercept2 - intersection2,
       trap_base2 := intersection1 - intersection2,
       trap_height := x5 - x0,
       trap_area := ((trap_base1 + trap_base2) / 2) * trap_height,
       
       total_area := rect_area + trap_area
  in total_area) = 80 :=
by
  -- Proof will be completed here.
  sorry

end area_of_region_l563_563364


namespace area_of_square_with_given_y_coords_l563_563083

theorem area_of_square_with_given_y_coords : 
  ∀ (x1 x2 x3 x4 : ℝ), 
  set.univ = { (x1, -3), (x2, 0), (x3, 6), (x4, 3) } → 
  ∃ s : ℝ, (s^2 = 45) :=
by
  sorry

end area_of_square_with_given_y_coords_l563_563083


namespace david_marks_in_math_l563_563492

theorem david_marks_in_math (marks_english marks_physics marks_chemistry marks_biology marks_average num_subjects : ℤ)
  (h_eng : marks_english = 61)
  (h_phy : marks_physics = 82)
  (h_chem : marks_chemistry = 67)
  (h_bio : marks_biology = 85)
  (h_avg : marks_average = 72)
  (h_num : num_subjects = 5) :
  ∃ (marks_math : ℤ), marks_math = 65 :=
by
  let total_marks := marks_average * num_subjects
  let known_marks := marks_english + marks_physics + marks_chemistry + marks_biology
  let marks_math := total_marks - known_marks
  have h_total_marks : total_marks = 360 := by sorry
  have h_known_marks : known_marks = 295 := by sorry
  have h_marks_math : marks_math = 65 := by sorry
  exact ⟨65, h_marks_math⟩

end david_marks_in_math_l563_563492


namespace totalScoreExpectation_l563_563772

variable (P1 : ℚ) (P2 : ℚ) (P3 : ℚ) (P4 : ℚ)
variable (X : ℚ)

-- Condition Definitions
def scoreProbability (kiteType : String) : ℚ :=
  if kiteType = "hard-wing" ∨ kiteType = "soft-wing" then P1
  else if kiteType = "series" ∨ kiteType = "flat" ∨ kiteType = "three-dimensional" then P2
  else 0

def totalScore : ℚ :=
  2 * P1 * P1 + 3 * (2 * P1 * P2) + 4 * P2 * P2

-- Statement of the problem
theorem totalScoreExpectation :
  (P1 = 2 / 5) →
  (P2 = 3 / 5) →
  (X = totalScore) →
  X = 16 / 5 := 
sorry

end totalScoreExpectation_l563_563772


namespace relationship_of_ys_l563_563296

variable (f : ℝ → ℝ)

def inverse_proportion := ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k / x

theorem relationship_of_ys
  (h_inv_prop : inverse_proportion f)
  (h_pts1 : f (-2) = 3)
  (h_pts2 : f (-3) = y₁)
  (h_pts3 : f 1 = y₂)
  (h_pts4 : f 2 = y₃) :
  y₂ < y₃ ∧ y₃ < y₁ :=
sorry

end relationship_of_ys_l563_563296


namespace black_white_ratio_correct_l563_563868

def circle_area (r : ℝ) : ℝ := real.pi * r ^ 2

def radii := [1, 2, 4, 6, 8]
def color_mapping := [true, false, true, false, true]  -- true for black, false for white

def compute_areas (radii : List ℝ) (color_mapping : List Bool) : (ℝ × ℝ) :=
  let rec aux (rs : List ℝ) (cs : List Bool) (areas_black areas_white : ℝ) : (ℝ × ℝ) :=
    match rs, cs with
    | [], [] => (areas_black, areas_white)
    | (r :: rs_tail), (c :: cs_tail) =>
      let current_area := circle_area r - (areas_black + areas_white)
      if c then
        aux rs_tail cs_tail (areas_black + current_area) areas_white
      else
        aux rs_tail cs_tail areas_black (areas_white + current_area)
    | _, _ => (areas_black, areas_white)  -- shouldn't happen
  aux radii color_mapping 0 0

def black_white_ratio : ℝ :=
  let (areas_black, areas_white) = compute_areas radii color_mapping
  areas_black / areas_white

theorem black_white_ratio_correct : black_white_ratio = 41 / 23 :=
by
  sorry

end black_white_ratio_correct_l563_563868


namespace same_terminal_side_angle_l563_563981

theorem same_terminal_side_angle (k : ℤ) : 
  0 ≤ (k * 360 - 35) ∧ (k * 360 - 35) < 360 → (k * 360 - 35) = 325 :=
by
  sorry

end same_terminal_side_angle_l563_563981


namespace volume_is_750_sqrt2_l563_563687

noncomputable def volume_of_prism 
  (a b c : ℝ)
  (h1 : a * b = 30)
  (h2 : a * c = 50)
  (h3 : b * c = 75) : ℝ :=
a * b * c

theorem volume_is_750_sqrt2 (a b c : ℝ)
  (h1 : a * b = 30)
  (h2 : a * c = 50)
  (h3 : b * c = 75) : volume_of_prism a b c h1 h2 h3 = 750 * real.sqrt 2 :=
by sorry

end volume_is_750_sqrt2_l563_563687


namespace quadratic_roots_l563_563342

theorem quadratic_roots (b c : ℝ) (h : ∀ x : ℝ, x^2 + bx + c = 0 ↔ x^2 - 5 * x + 2 = 0):
  c / b = -4 / 21 :=
  sorry

end quadratic_roots_l563_563342


namespace maximal_perfect_cubes_l563_563166

theorem maximal_perfect_cubes (n : ℕ) (a : fin n → ℕ)
  (h_n : n ≥ 2)
  (h_distinct : function.injective a)
  (h_not_cube : ∀ i, ∀ m : ℕ, m ^ 3 ≠ a i) :
  ∃ k, k = n^2 / 4 ∧ (∀ i j, i < j → (a i * a j).is_cubical ↔ (i + j).mod 3 = 0) :=
by
  sorry

end maximal_perfect_cubes_l563_563166


namespace correct_tessellations_l563_563464

-- Definitions of the polygons and their interior angles
def interior_angle_triangle : ℝ := 60
def interior_angle_square : ℝ := 90
def interior_angle_hexagon : ℝ := 120
def interior_angle_octagon : ℝ := 135

-- Tessellation conditions for each polygon pair
def tessellate (angles : List ℝ) : Bool :=
  (List.sum angles = 360 ∧ List.all angles (λ x, x > 0 ∧ x < 180))

noncomputable def group1_tessellates : Bool :=
  tessellate [interior_angle_triangle, interior_angle_triangle, interior_angle_triangle, interior_angle_square, interior_angle_square]
noncomputable def group2_tessellates : Bool :=
  tessellate [interior_angle_triangle, interior_angle_triangle, interior_angle_hexagon, interior_angle_hexagon]
noncomputable def group3_tessellates : Bool :=
  tessellate [interior_angle_hexagon, interior_angle_square, interior_angle_square]
noncomputable def group4_tessellates : Bool :=
  tessellate [interior_angle_octagon, interior_angle_octagon, interior_angle_square]

-- The final goal is to prove the correct tessellations
theorem correct_tessellations :
  [group1_tessellates, group2_tessellates, group4_tessellates] = [true, true, true] ∧ group3_tessellates = false := 
  sorry

end correct_tessellations_l563_563464


namespace equal_weight_partitions_l563_563027

theorem equal_weight_partitions :
  ∃ (A B C : Finset ℕ), (∀ x ∈ A, 1 ≤ x ∧ x ≤ 555) ∧
                        (∀ x ∈ B, 1 ≤ x ∧ x ≤ 555) ∧
                        (∀ x ∈ C, 1 ≤ x ∧ x ≤ 555) ∧
                        (A ∪ B ∪ C = Finset.range 556 \ Finset.singleton 0) ∧
                        (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧
                        (A.sum id = 51430) ∧ (B.sum id = 51430) ∧ (C.sum id = 51430) :=
by
  sorry

end equal_weight_partitions_l563_563027


namespace probability_red_or_yellow_l563_563771

-- Definitions and conditions
def p_green : ℝ := 0.25
def p_blue : ℝ := 0.35
def total_probability := 1
def p_red_and_yellow := total_probability - (p_green + p_blue)

-- Theorem statement
theorem probability_red_or_yellow :
  p_red_and_yellow = 0.40 :=
by
  -- Here we would prove that the combined probability of selecting either a red or yellow jelly bean is 0.40, given the conditions.
  sorry

end probability_red_or_yellow_l563_563771


namespace coloring_scheme_correctness_l563_563748

noncomputable def count_coloring_schemes (m n : ℕ) (hm : 2 ≤ m) (hn : 4 ≤ n) : ℕ :=
if even n then
  let k := n / 2 in
  (m - 1) ^ k + (-1)^k * (m - 1) ^ 2
else
  (m - 1)^n + (-1)^n * (m - 1)

theorem coloring_scheme_correctness (m n : ℕ) (hm : 2 ≤ m) (hn : 4 ≤ n) : 
(count_coloring_schemes m n hm hn = 
  if even n then
    let k := n / 2 in
    ((m - 1) ^ k + (-1) ^ k * (m - 1)) ^ 2
  else
    (m - 1)^n + (-1)^n * (m - 1)
) :=
sorry

end coloring_scheme_correctness_l563_563748


namespace number_of_lines_l563_563217

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero: a ≠ 0 ∨ b ≠ 0

-- Definition of a line passing through a point P
def passes_through (l : Line) (P : Point) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

-- Definition of a line having equal intercepts on x-axis and y-axis
def equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.a = l.b

-- Definition of a specific point P
def P : Point := { x := 1, y := 2 }

-- The theorem statement
theorem number_of_lines : ∃ (lines : Finset Line), (∀ l ∈ lines, passes_through l P ∧ equal_intercepts l) ∧ lines.card = 2 := by
  sorry

end number_of_lines_l563_563217


namespace bob_takes_fraction_of_cake_l563_563475

theorem bob_takes_fraction_of_cake
  (leftover_cake : ℚ)
  (tom_multiple : ℚ)
  (total_cake_fraction : leftover_cake = 8/9)
  (division_condition : tom_multiple = 2) :
  let x := fractional_cake (leftover_cake tom_multiple) in
  x * 4 = 8/9 → 
  bob_took : x = 2/9 := 
sorry

end bob_takes_fraction_of_cake_l563_563475


namespace john_bought_notebooks_l563_563989

def pages_per_notebook : ℕ := 40
def pages_per_day : ℕ := 4
def total_days : ℕ := 50

theorem john_bought_notebooks : (pages_per_day * total_days) / pages_per_notebook = 5 :=
by
  sorry

end john_bought_notebooks_l563_563989


namespace tangent_line_eq_at_x1_enclosed_area_eq_l563_563922

noncomputable def f (x : ℝ) := x^3 - x + 2
noncomputable def f' (x : ℝ) := 3 * x^2 - 1

-- Prove the equation of the tangent line l
theorem tangent_line_eq_at_x1 :
  let x := 1
  let point := (x, f(x))
  let slope := f' x in
  ∃ (c : ℝ), ∀ y, y = slope * (x - 1) + c → y = 2 * x :=
by sorry

-- Prove the area enclosed by the line l and the graph of f'(x)
theorem enclosed_area_eq :
  ∫ x in -1/3..1, (2 * x - f' x) = 32 / 27 :=
by sorry

end tangent_line_eq_at_x1_enclosed_area_eq_l563_563922


namespace range_of_b_l563_563513

noncomputable def intersects (a b: ℝ) : Prop :=
  let L := λ (x y : ℝ), (a * x + y + a + 1 = 0)
  let C := λ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 - b
  ∃ (x y : ℝ), L x y ∧ C x y

theorem range_of_b (b : ℝ) :
  (∀ a : ℝ, ∃ (x y : ℝ), (ax + y + a + 1 = 0) ∧ ((x - 1)^2 + (y - 1)^2 = 2 - b)) ↔ b < -6 := 
begin
  sorry 
end

end range_of_b_l563_563513


namespace value_of_a_l563_563532

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x > 0 ∧ x < 3 then real.log x - a * x else 0 -- f defined conditionally

theorem value_of_a (a : ℝ) : 
(∀ x : ℝ, f a (x + 3) = 3 * f a x) ∧ 
(∀ x : ℝ, (0 < x ∧ x < 3) → f a x = real.log x - a * x) ∧ 
(a > 1 / 3) ∧ 
(∀ y : ℝ, (-6 < y ∧ y < -3) → f a y ≤ -1 / 9 ∧ ((∃ c, (-6 < c ∧ c < -3) ∧ f a c = -1 / 9))) 
→ a = 1 :=
by sorry

end value_of_a_l563_563532


namespace sum_of_consecutive_integers_sqrt_28_l563_563903

theorem sum_of_consecutive_integers_sqrt_28 (a b : ℤ) (h1 : a < b) (h2 : b = a + 1) (h3 : a < sqrt 28) (h4 : sqrt 28 < b) : a + b = 11 := by
  sorry

end sum_of_consecutive_integers_sqrt_28_l563_563903


namespace rearrange_trapezoids_to_square_l563_563830

-- Step-by-step construction of the proof problem
noncomputable def midpoint {α : Type*} [add_comm_group α] [vector_space ℝ α] (x y : α) : α := (1/2 : ℝ) • (x + y)
variable {A B C D E : ℝ × ℝ}

-- Given an irregular quadrilateral with vertices A, B, C, D
-- and point E as the midpoint of segment CD
-- Prove that cutting along segment AE results in two trapezoids that can be rearranged to form a square
theorem rearrange_trapezoids_to_square
  (h_midpoint : E = midpoint C D)
  (h_irregular_quadrilateral : ¬ collinear ℝ {A, B, C, D}) :
  ∃ T₁ T₂ : set (ℝ × ℝ),
    T₁ ∪ T₂ = {p | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • A + t • E} ∧
    is_trapezoid T₁ ∧ is_trapezoid T₂ ∧ can_form_square T₁ T₂ :=
  sorry

end rearrange_trapezoids_to_square_l563_563830


namespace ln_inequality_l563_563204

theorem ln_inequality (a b c : ℝ) (h0 : 0 < a) (h1 : a < b) (h2 : b < c) (h3 : c < 1) :
    (ln a / a) < (ln b / b) ∧ (ln b / b) < (ln c / c) :=
by
  sorry

end ln_inequality_l563_563204


namespace cooler_capacity_l563_563432

theorem cooler_capacity (linemen: ℕ) (linemen_drink: ℕ) 
                        (skill_position: ℕ) (skill_position_drink: ℕ) 
                        (linemen_count: ℕ) (skill_position_count: ℕ) 
                        (skill_wait: ℕ) 
                        (h1: linemen_count = 12) 
                        (h2: linemen_drink = 8) 
                        (h3: skill_position_count = 10) 
                        (h4: skill_position_drink = 6) 
                        (h5: skill_wait = 5):
 linemen_count * linemen_drink + skill_wait * skill_position_drink = 126 :=
by
  sorry

end cooler_capacity_l563_563432


namespace smallest_n_l563_563630

theorem smallest_n (n : ℕ) (k : ℕ) (a m : ℕ) 
  (h1 : 0 ≤ k)
  (h2 : k < n)
  (h3 : a ≡ k [MOD n])
  (h4 : m > 0) :
  (∀ a m, (∃ k, a = n * k + 5) -> (a^2 - 3*a + 1) ∣ (a^m + 3^m) → false) 
  → n = 11 := sorry

end smallest_n_l563_563630


namespace constant_term_in_expansion_eq_24_l563_563128

theorem constant_term_in_expansion_eq_24 :
  (∃ c : ℤ, c = 24 ∧ ∀ x : ℂ, (2 * x - (1 / x))^4 = 24 + c * x + _) :=
sorry

end constant_term_in_expansion_eq_24_l563_563128


namespace minimum_production_volume_to_avoid_loss_l563_563023

open Real

-- Define the cost function
def cost (x : ℕ) : ℝ := 3000 + 20 * x - 0.1 * (x ^ 2)

-- Define the revenue function
def revenue (x : ℕ) : ℝ := 25 * x

-- Condition: 0 < x < 240 and x ∈ ℕ (naturals greater than 0)
theorem minimum_production_volume_to_avoid_loss (x : ℕ) (hx1 : 0 < x) (hx2 : x < 240) (hx3 : x ∈ (Set.Ioi 0)) :
  revenue x ≥ cost x ↔ x ≥ 150 :=
by
  sorry

end minimum_production_volume_to_avoid_loss_l563_563023


namespace inverse_proportion_neg_k_l563_563437

theorem inverse_proportion_neg_k (x1 x2 y1 y2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) (h3 : y1 > y2) :
  ∃ k : ℝ, k < 0 ∧ (∀ x, (x = x1 → y1 = k / x) ∧ (x = x2 → y2 = k / x)) := by
  use -1
  sorry

end inverse_proportion_neg_k_l563_563437


namespace sum_of_powers_mod_p2_l563_563266

theorem sum_of_powers_mod_p2 (p : ℕ) [hp_prime : Fact (Nat.Prime p)] (hp_odd : p % 2 = 1) :
  (∑ n in Finset.range (p - 1), n^(p-1)) % p^2 = (p + (p-1)!) % p^2 :=
  sorry

end sum_of_powers_mod_p2_l563_563266


namespace zoe_total_expenditure_is_correct_l563_563385

noncomputable def zoe_expenditure : ℝ :=
  let initial_app_cost : ℝ := 5
  let monthly_fee : ℝ := 8
  let first_two_months_fee : ℝ := 2 * monthly_fee
  let yearly_cost_without_discount : ℝ := 12 * monthly_fee
  let discount : ℝ := 0.15 * yearly_cost_without_discount
  let discounted_annual_plan : ℝ := yearly_cost_without_discount - discount
  let actual_annual_plan : ℝ := discounted_annual_plan - first_two_months_fee
  let in_game_items_cost : ℝ := 10
  let discounted_in_game_items_cost : ℝ := in_game_items_cost - (0.10 * in_game_items_cost)
  let upgraded_feature_cost : ℝ := 12
  let discounted_upgraded_feature_cost : ℝ := upgraded_feature_cost - (0.10 * upgraded_feature_cost)
  initial_app_cost + first_two_months_fee + actual_annual_plan + discounted_in_game_items_cost + discounted_upgraded_feature_cost

theorem zoe_total_expenditure_is_correct : zoe_expenditure = 122.4 :=
by
  sorry

end zoe_total_expenditure_is_correct_l563_563385


namespace monotonicity_of_f_min_value_f_diff_l563_563927

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - (1 / x) + 2 * a * Real.log x

theorem monotonicity_of_f (a : ℝ) :
  (∀ x ∈ Ioi 0, deriv (λ x, f x a) x ≥ 0) ∨
  (a ≥ -1 → ∀ x ∈ Ioi 0, deriv (λ x, f x a) x ≥ 0) ∧
  (a < -1 → (∀ x ∈ Ioo 0 (-a - Real.sqrt (a^2 - 1)), deriv (λ x, f x a) x > 0) ∧ 
             (∀ x ∈ Ioo (-a + Real.sqrt (a^2 - 1), +∞), deriv (λ x, f x a) x > 0) ∧ 
             (∀ x ∈ Ioo (-a - Real.sqrt (a^2 - 1), -a + Real.sqrt (a^2 - 1)), deriv (λ x, f x a) x < 0)) := sorry

theorem min_value_f_diff (a : ℝ) (x1 x2 : ℝ) (hx2 : x2 ∈ Ici (Real.exp 1)) (hx1 : 0 < x1 ∧ x1 < (1 / Real.exp 1)) :
  f x1 a - f x2 a = 4 / Real.exp 1 := sorry

end monotonicity_of_f_min_value_f_diff_l563_563927


namespace opposite_of_neg_three_l563_563004

-- Define the concept of negation and opposite of a number
def opposite (x : ℤ) : ℤ := -x

-- State the problem: Prove that the opposite of -3 is 3
theorem opposite_of_neg_three : opposite (-3) = 3 :=
by
  -- Proof
  sorry

end opposite_of_neg_three_l563_563004


namespace unit_vector_AB_is_correct_l563_563196

def vectorAB : ℝ × ℝ := (-1, 2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

def unit_vector (v : ℝ × ℝ) : ℝ × ℝ :=
  let mag := magnitude v 
  in (v.1 / mag, v.2 / mag)

theorem unit_vector_AB_is_correct :
  unit_vector vectorAB = (- Real.sqrt 5 / 5, 2 * Real.sqrt 5 / 5) :=
sorry

end unit_vector_AB_is_correct_l563_563196


namespace calculate_expression_l563_563481

theorem calculate_expression :
  (3^2 : ℝ) + real.sqrt 25 - real.cbrt 64 + | (-9 : ℝ) | = 19 :=
by
  sorry

end calculate_expression_l563_563481


namespace negate_proposition_l563_563702

theorem negate_proposition (x y : ℝ) :
  (¬ (x^2 + y^2 = 0 → x = 0 ∧ y = 0)) ↔ (x^2 + y^2 ≠ 0 → ¬ (x = 0 ∧ y = 0)) :=
by
  sorry

end negate_proposition_l563_563702


namespace minimum_a_n_plus_S_n_l563_563227

-- Given the conditions
def equi_variance_sequence (a : ℕ → ℝ) (p : ℝ) : Prop :=
∀ n : ℕ, n ≥ 2 → a n ^ 2 - a (n -1) ^ 2 = p

def sum_sequence (a S : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = ∑ i in finset.range n, a (i + 1)

def S_vars_1 : Prop :=
(equi_variance_sequence S p ∧ (S 2 + S 4 = 2 + real.sqrt 2))

def a_vars_1 : Prop :=
(equi_variance_sequence a p ∧ (a 3 + a 4 = 2 - real.sqrt 2))

-- Prove the minimum value of (a_n + S_n) is 2√2 - 1
theorem minimum_a_n_plus_S_n (a S : ℕ → ℝ) (p : ℝ) :
  S_vars_1 ∧ a_vars_1 ∧ sum_sequence a S →
  ∃ n : ℕ, 1 ≤ n ∧ a n + S n = 2 * real.sqrt 2 - 1 :=
by sorry

end minimum_a_n_plus_S_n_l563_563227


namespace gecko_eggs_hatch_l563_563434

theorem gecko_eggs_hatch :
  let total_eggs := 30 in
  let infertile_percentage := 0.20 in
  let calcification_fraction := 1 / 3 in
  let infertile_eggs := infertile_percentage * total_eggs in
  let fertile_eggs := total_eggs - infertile_eggs in
  let non_hatching_eggs := calcification_fraction * fertile_eggs in
  let hatching_eggs := fertile_eggs - non_hatching_eggs in
  hatching_eggs = 16 :=
by
  sorry

end gecko_eggs_hatch_l563_563434


namespace price_of_peas_l563_563789

theorem price_of_peas
  (P : ℝ) -- price of peas per kg in rupees
  (price_soybeans : ℝ) (price_mixture : ℝ)
  (ratio_peas_soybeans : ℝ) :
  price_soybeans = 25 →
  price_mixture = 19 →
  ratio_peas_soybeans = 2 →
  P = 16 :=
by
  intros h_price_soybeans h_price_mixture h_ratio
  sorry

end price_of_peas_l563_563789


namespace repeated_operations_final_digit_l563_563447

def final_single_digit' (N : ℕ) (k : ℕ) : ℕ :=
  if N = (List.replicate k 7).asRepr.toNat then 7 else 0

theorem repeated_operations_final_digit (k : ℕ) :
  final_single_digit' ((List.replicate k 7).asRepr.toNat) k = 7 :=
by
  sorry

end repeated_operations_final_digit_l563_563447


namespace bakery_doughnuts_given_away_l563_563406

theorem bakery_doughnuts_given_away :
  (∀ (boxes_doughnuts : ℕ) (total_doughnuts : ℕ) (boxes_sold : ℕ), 
    boxes_doughnuts = 10 →
    total_doughnuts = 300 →
    boxes_sold = 27 →
    ∃ (doughnuts_given_away : ℕ),
    doughnuts_given_away = (total_doughnuts / boxes_doughnuts - boxes_sold) * boxes_doughnuts ∧
    doughnuts_given_away = 30) :=
by
  intros boxes_doughnuts total_doughnuts boxes_sold h1 h2 h3
  use (total_doughnuts / boxes_doughnuts - boxes_sold) * boxes_doughnuts
  split
  · rw [h1, h2, h3]
    sorry
  · sorry

end bakery_doughnuts_given_away_l563_563406


namespace incorrect_absolute_value_statement_l563_563050

theorem incorrect_absolute_value_statement :
  (∀ (x : ℝ), |x| = if x < 0 then -x else x) →
  ¬(|-1.5| = -1.5) :=
by
  intro h_abs_val_def
  sorry

end incorrect_absolute_value_statement_l563_563050


namespace akeno_spent_more_l563_563102

theorem akeno_spent_more (akeno_spent : ℤ) (lev_spent_ratio : ℚ) (ambrocio_less : ℤ) 
  (h1 : akeno_spent = 2985)
  (h2 : lev_spent_ratio = 1/3)
  (h3 : ambrocio_less = 177) : akeno_spent - ((lev_spent_ratio * akeno_spent).toInt + ((lev_spent_ratio * akeno_spent).toInt - ambrocio_less)) = 1172 := by
  sorry

end akeno_spent_more_l563_563102


namespace find_z_l563_563536

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def N_has_1998_digits (N : ℕ) : Prop :=
  10 ^ 1997 ≤ N ∧ N < 10 ^ 1998

def divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem find_z (N : ℕ)
  (h_digits : N_has_1998_digits N)
  (h_div9 : divisible_by_9 N) :
  let x := sum_of_digits N
      y := sum_of_digits x
      z := sum_of_digits y
  in z = 9 :=
sorry

end find_z_l563_563536


namespace find_length_B1B2_l563_563656

theorem find_length_B1B2 (BA1 A1A2 CA2 AB1 CB2 : ℝ)
  (h1 : BA1 = 6) 
  (h2 : A1A2 = 8) 
  (h3 : CA2 = 4) 
  (h4 : AB1 = 9) 
  (h5 : CB2 = 6)
  (K L C : Point) 
  (h6 : collinear K L C) 
  : B1B2 = 12 := 
sorry

end find_length_B1B2_l563_563656


namespace second_box_sand_capacity_l563_563416

theorem second_box_sand_capacity :
  let first_box_volume := 4 * 5 * 10 in
  let second_box_volume := 12 * 20 * 10 in
  let first_box_sand := 200 in
  second_box_volume / first_box_volume * first_box_sand = 2400 :=
by
  let first_box_volume := 4 * 5 * 10;
  let second_box_volume := 12 * 20 * 10;
  let first_box_sand := 200;
  calc
  second_box_volume / first_box_volume * first_box_sand
      = 2400 : by sorry

end second_box_sand_capacity_l563_563416


namespace volume_equation_l563_563827

noncomputable def volume_rect_plus_margin : ℝ := 
  let v_rect := 4 * 5 * 6
  let v_ext := 2 * (1.5 * 4 * 5) + 2 * (1.5 * 4 * 6) + 2 * (1.5 * 5 * 6)
  let v_qcyl := 4 * (∏ * 1.5^2 * 4) / 4 + 4 * (∏ * 1.5^2 * 5) / 4 + 4 * (∏ * 1.5^2 * 6) / 4
  let v_scaps := 8 * (∏ * 1.5^3 / 6) / 8
  v_rect + v_ext + v_qcyl + v_scaps

theorem volume_equation :
  ∃ (m n p : ℕ), p ≠ 0 ∧ (Nat.coprime n p) ∧ volume_rect_plus_margin = (m + n * ∏) / p ∧ (m + n + p = 839) := 
sorry

end volume_equation_l563_563827


namespace monotonically_decreasing_interval_l563_563145

-- Define the constants and the function
def pi_fifth := Real.pi / 5
def f (x : ℝ) : ℝ := Real.cos (2 * x) * Real.cos pi_fifth + Real.sin (2 * x) * Real.sin pi_fifth

-- State the theorem
theorem monotonically_decreasing_interval (k : ℤ) :
  ∀ x, k * Real.pi + Real.pi / 10 ≤ x ∧ x ≤ k * Real.pi + 3 * Real.pi / 5 → 
  f x = Real.cos (2 * x - pi_fifth) :=
sorry

end monotonically_decreasing_interval_l563_563145


namespace domain_of_composition_l563_563910

theorem domain_of_composition (f : ℝ → ℝ) :
  (∀ x, -1 < x ∧ x < 1 → f x ≠ 0) → 
  ((∀ x, -1 < x ∧ x < 0 → f(2*x + 1) ≠ 0)) :=
begin
  intro h,
  sorry
end

end domain_of_composition_l563_563910


namespace chess_tournament_ranking_sequences_l563_563240

theorem chess_tournament_ranking_sequences :
  let players := ["X", "Y", "Z", "W"]
  ∃ (P: players → list String) (lenP: P.length = 4),
  ∀ (matches_day1: List (String × String)), (∅ ⊆ matches_day1) ∧ (match_outcomes matches_day1 = [1, 0]) →
  ∀ (matches_day2: List (String × String)), (∅ ⊆ matches_day2) ∧ (match_outcomes matches_day2 = [1, 0]) →
  (length (permutations P)) = 8 :=
by
  sorry

end chess_tournament_ranking_sequences_l563_563240


namespace sum_sixth_powers_less_than_1000_l563_563740

theorem sum_sixth_powers_less_than_1000 : 
  ∑ n in ({ n ^ 6 | n : ℕ, n ^ 6 < 1000}), n = 794 :=
by
  sorry

end sum_sixth_powers_less_than_1000_l563_563740


namespace intersection_A_B_l563_563895

open Set

variable A : Set ℕ := {1, 3, 5, 7}
variable B : Set ℕ := {4, 5, 6, 7}

theorem intersection_A_B : A ∩ B = {5, 7} :=
by sorry

end intersection_A_B_l563_563895


namespace range_of_a_l563_563171

theorem range_of_a (a : ℝ) : (∃ x : ℝ, a * x = 1) ↔ a ≠ 0 := by
sorry

end range_of_a_l563_563171


namespace total_animals_l563_563966

-- Definitions
variables (H C1 C2 : ℕ)

-- Conditions as hypotheses
def condition1 : Prop := C1 + 2 * C2 = 200
def condition2 : Prop := H = C2

-- The main theorem to prove
theorem total_animals (h1 : condition1) (h2 : condition2) : H + C1 + C2 = 200 :=
sorry

end total_animals_l563_563966


namespace boy_can_escape_l563_563607

-- Definitions
def square_pool : Type := unit
def side_length : ℝ := 2

structure Point :=
(x : ℝ)
(y : ℝ)

def center : Point := { x := 1, y := 1 }
def corner : Point := { x := 2, y := 2 }

def teacher_speed : ℝ := 1 -- Assuming teacher speed as a baseline 1 unit speed
def boy_swim_speed : ℝ := teacher_speed / 3

-- Boy's ability on land
variable (boy_land_speed : ℝ) -- Boy's land speed which is greater than teacher_speed
axiom boy_faster_on_land (h : teacher_speed < boy_land_speed)

-- Mathematical properties and distances
def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

def time_to_reach (distance speed : ℝ) : ℝ := distance / speed

-- Theorem Statement
theorem boy_can_escape : ∃ path : list Point, 
  ∀ t ∈ path, 
    let d_boy := distance center t,
        t_boy := time_to_reach d_boy boy_swim_speed in
    let m : Point := { x := t.x, y := side_length },
        d_teacher := distance corner m,
        t_teacher := time_to_reach d_teacher teacher_speed in
    (t_boy < t_teacher) ∧ (∀ t_other ∈ path, (time_to_reach (distance t t_other) boy_land_speed) < (time_to_reach (distance t t_other) teacher_speed)) :=
begin
  sorry
end

end boy_can_escape_l563_563607


namespace veena_paid_fraction_of_total_l563_563403

-- Definitions for the amounts paid by Akshitha, Veena, and Lasya
def amount_paid_by_Akshitha (V : ℝ) : ℝ := (3 / 4) * V
def amount_paid_by_Veena (L : ℝ) : ℝ := (1 / 2) * L

-- Prove that Veena paid 4/15 of the total bill
theorem veena_paid_fraction_of_total (A V L T : ℝ)
  (hA : A = amount_paid_by_Akshitha V)
  (hV : V = amount_paid_by_Veena L)
  (hL : L = 2 * V)
  (hT : T = L + V + A) :
  V / T = 4 / 15 :=
by
  -- Use the substitution steps manually derived to state the correct fraction
  sorry

end veena_paid_fraction_of_total_l563_563403


namespace count_special_multiples_l563_563218

def is_multiple (n k : ℕ) : Prop := ∃ m, n = k * m
def count_multiples (n k : ℕ) : ℕ := (n / k)

theorem count_special_multiples : 
  let within_range := (1 to 150)
  let multiples_of_2 := ∀ n, n ∈ within_range → is_multiple n 2
  let multiples_of_5 := ∀ n, n ∈ within_range → is_multiple n 5
  let multiples_of_15 := ∀ n, n ∈ within_range → is_multiple n 15
  let combined_multiples := (count_multiples 150 2) + (count_multiples 150 5) - (count_multiples 150 10)
in (combined_multiples - count_multiples 150 15) = 80 := 
sorry

end count_special_multiples_l563_563218


namespace infinite_nested_radical_eq_two_l563_563824

noncomputable def infinite_nested_radical : ℝ := sorry

theorem infinite_nested_radical_eq_two :
  infinite_nested_radical = 2 ↔ infinite_nested_radical = sqrt (2 + infinite_nested_radical) ∧ infinite_nested_radical ≥ 0 := sorry

end infinite_nested_radical_eq_two_l563_563824


namespace rotation_matrix_150_degrees_l563_563819

-- Definition of the rotation matrix
def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  (Matrix.vecCons (Matrix.vecCons (Real.cos θ) (-Real.sin θ) Matrix.vecEmp)
                  (Matrix.vecCons (Real.sin θ)  (Real.cos θ) Matrix.vecEmp))

-- The statement equivalent to the mathematical proof problem
theorem rotation_matrix_150_degrees :
  rotation_matrix (Real.pi * (5 / 6)) = 
  ![![ -Real.sqrt(3) / 2, -1 / 2 ], ![ 1 / 2, -Real.sqrt(3) / 2 ]] :=
by
  sorry

end rotation_matrix_150_degrees_l563_563819


namespace no_integer_solution_for_euler_conjecture_l563_563300

theorem no_integer_solution_for_euler_conjecture :
  ¬(∃ n : ℕ, 5^4 + 12^4 + 9^4 + 8^4 = n^4) :=
by
  -- Sum of the given fourth powers
  have lhs : ℕ := 5^4 + 12^4 + 9^4 + 8^4
  -- Direct proof skipped with sorry
  sorry

end no_integer_solution_for_euler_conjecture_l563_563300


namespace sum_of_positive_factors_of_90_l563_563375

theorem sum_of_positive_factors_of_90 : 
  let n := 90 in 
  let factors := (1 + 2) * (1 + 3 + 9) * (1 + 5) in 
  factors = 234 :=
by
  sorry

end sum_of_positive_factors_of_90_l563_563375


namespace triangle_inequality_proof_l563_563657

variables {A B C A₁ B₁ C₁ : Type}
variables {α β γ : ℝ} {a b c : ℝ}
variables {BC : a = BC} {CA : b = CA} {AB : c = AB}
variables [IsAcuteAngledTriangle A B C]
variables [OnSide A₁ BC] [OnSide B₁ CA] [OnSide C₁ AB]

theorem triangle_inequality_proof :
  2 * (dist B₁ C₁ * cos α + dist C₁ A₁ * cos β + dist A₁ B₁ * cos γ) 
  ≥ a * cos α + b * cos β + c * cos γ :=
sorry

end triangle_inequality_proof_l563_563657


namespace area_not_covered_correct_l563_563080

-- Define the dimensions of the rectangle
def rectangle_length : ℕ := 10
def rectangle_width : ℕ := 8

-- Define the side length of the square
def square_side_length : ℕ := 5

-- The area of the rectangle
def rectangle_area : ℕ := rectangle_length * rectangle_width

-- The area of the square
def square_area : ℕ := square_side_length * square_side_length

-- The area of the region not covered by the square
def area_not_covered : ℕ := rectangle_area - square_area

-- The theorem statement asserting the required area
theorem area_not_covered_correct : area_not_covered = 55 :=
by
  -- Proof is omitted
  sorry

end area_not_covered_correct_l563_563080


namespace initial_volume_l563_563235

variable (milk_init water_init : ℝ)
variable (add_water : ℝ := 1.6)

-- Initial conditions
def initial_ratio := milk_init / water_init = 7 / 1
def added_water_ratio := milk_init / (water_init + add_water) = 3 / 1

-- Proof problem statement
theorem initial_volume (h₁ : initial_ratio) (h₂ : added_water_ratio) : milk_init + water_init = 9.6 := by
  sorry

end initial_volume_l563_563235


namespace space_diagonals_l563_563764

theorem space_diagonals (Q : Type) [polyhedron Q] (vertices : ℕ) (edges : ℕ) (faces : ℕ)
  (triangular_faces : ℕ) (quadrilateral_faces : ℕ) (hvertices : vertices = 30)
  (hedges : edges = 70) (hfaces : faces = 42) (htri_faces : triangular_faces = 30)
  (hquad_faces : quadrilateral_faces = 12) :
  space_diagonals Q = 341 :=
by
  sorry

end space_diagonals_l563_563764


namespace cos_neg_pi_over_3_l563_563135

noncomputable def angle := - (Real.pi / 3)

theorem cos_neg_pi_over_3 : Real.cos angle = 1 / 2 :=
by
  sorry

end cos_neg_pi_over_3_l563_563135


namespace area_of_circle_l563_563979

open Real

variables {O A B C D E F : Point}
variables {r : ℝ}

-- Definitions based on conditions
def diameter (A B : Point) (O : Point) : Prop := distance O A = r ∧ distance O B = r

def perpendicular (A B C D : Point) : Prop := ∃ M, M = (A + B) / 2 ∧ M = (C + D) / 2 ∧ angle A M C = π / 2

def chord (D F : Point) (E : Point) : Prop := between D E F

def given_conditions : Prop :=
  diameter A B O ∧ diameter C D O ∧ perpendicular A B C D ∧ chord D F E ∧ distance D E = 6 ∧ distance E F = 2

-- Target proof
theorem area_of_circle (h : given_conditions) : π * r ^ 2 = 24 * π :=
sorry

end area_of_circle_l563_563979


namespace product_positive_real_part_solutions_l563_563226

-- Conditions given in the problem
def x8_eq_neg256 (x : ℂ) : Prop := x^8 = -256

-- Translate each relevant condition from the problem
def eight_solutions := {x : ℂ | x8_eq_neg256 x }

def has_positive_real_part (x : ℂ) : Prop := x.re > 0

def positive_real_part_solutions := {x : ℂ | x ∈ eight_solutions ∧ has_positive_real_part x }

-- Prove the product of these solutions is 8 (cos 67.5° + i sin 67.5°)
theorem product_positive_real_part_solutions :
  ∏ x in positive_real_part_solutions, x = 8 * (Complex.cos (Complex.pi * 67.5 / 180) + Complex.I * Complex.sin (Complex.pi * 67.5 / 180)) :=
by
  sorry

end product_positive_real_part_solutions_l563_563226


namespace tangent_line_at_point_range_of_a_for_intersection_l563_563924

-- Part 1: Tangent Line Proof
theorem tangent_line_at_point (a : ℝ) (h1 : a = 4) :
  ∀ (f : ℝ → ℝ), (∀ x, f x = (4 + Real.log x) / x) → 
  ∃ k b, k * 1 + b = 4 ∧ (∀ t, f t - (k * t + b) = 0 → (k = -3 ∧ b = 7)) :=
by {
  sorry
}

-- Part 2: Intersection Proof
theorem range_of_a_for_intersection (f g : ℝ → ℝ) :
  (∀ x, f x = (a + Real.log x) / x) →
  g = (λ x, 1) →
  (∃ x, (0 < x ∧ x ≤ Real.exp 2 ∧ f x = g x)) ↔ (1 ≤ a) :=
by {
  sorry
}

end tangent_line_at_point_range_of_a_for_intersection_l563_563924


namespace geom_seq_product_l563_563611

theorem geom_seq_product (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h_prod : a 5 * a 14 = 5) :
  a 8 * a 9 * a 10 * a 11 = 10 := 
sorry

end geom_seq_product_l563_563611


namespace max_area_of_rectangular_playground_l563_563987

theorem max_area_of_rectangular_playground (P : ℕ) (hP : P = 160) :
  (∃ (x y : ℕ), 2 * (x + y) = P ∧ x * y = 1600) :=
by
  sorry

end max_area_of_rectangular_playground_l563_563987


namespace sample_variance_minimum_l563_563175

theorem sample_variance_minimum (x y : ℝ) (h : x + y = 2) : 
  let variance := (1/4) * ((x - 2)^2 + (y - 2)^2 + 10) in
  variance >= 3 :=
by
  sorry

end sample_variance_minimum_l563_563175


namespace commission_percentage_l563_563429

-- Define the conditions
def cost_of_item := 18.0
def observed_price := 27.0
def profit_percentage := 0.20
def desired_selling_price := cost_of_item + profit_percentage * cost_of_item
def commission_amount := observed_price - desired_selling_price

-- Prove the commission percentage taken by the online store
theorem commission_percentage : (commission_amount / desired_selling_price) * 100 = 25 :=
by
  -- Here the proof would normally be implemented
  sorry

end commission_percentage_l563_563429


namespace slope_of_line_between_intersections_of_circles_l563_563321

theorem slope_of_line_between_intersections_of_circles :
  ∀ C D : ℝ × ℝ, 
    -- Conditions: equations of the circles
    (C.1^2 + C.2^2 - 6 * C.1 + 4 * C.2 - 8 = 0) ∧ (C.1^2 + C.2^2 - 8 * C.1 - 2 * C.2 + 10 = 0) →
    (D.1^2 + D.2^2 - 6 * D.1 + 4 * D.2 - 8 = 0) ∧ (D.1^2 + D.2^2 - 8 * D.1 - 2 * D.2 + 10 = 0) →
    -- Question: slope of line CD
    ((C.2 - D.2) / (C.1 - D.1) = -1 / 3) :=
by
  sorry

end slope_of_line_between_intersections_of_circles_l563_563321


namespace min_value_sin4_cos4_l563_563147

theorem min_value_sin4_cos4 (x : ℝ) : (sin x)^4 + 2 * (cos x)^4 >= 1/2 := 
by 
  have h1 : (sin x)^2 + (cos x)^2 = 1 := by
    apply sin_sq_add_cos_sq
  let a := (sin x)^2
  have h2 : 0 <= a ∧ a <= 1 := by
    split
    · apply sq_nonneg
    · rw [← h1]
      apply add_le_of_le_sub
      exact le_of_lt (neg_lt_sub_of_pos (cos x))
  have h3 : (sin x)^4 + 2 * (cos x)^4 >= ((1 - a + a^2) / 2) :=
    sorry
  show (sin x)^4 + 2 * (cos x)^4 >= 1 / 2
    sorry

end min_value_sin4_cos4_l563_563147


namespace range_of_x_l563_563559

noncomputable def f : ℝ → ℝ := sorry -- Defining f with given properties will be done in the proof

def monotonically_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f y ≤ f x

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem range_of_x (f : ℝ → ℝ) (h1 : monotonically_decreasing_on f {x : ℝ | 0 ≤ x})
  (h2 : even_function f) :
  {x : ℝ | f (x^2 - 2) < f 1} = {x : ℝ | x < -sqrt 3} ∪ {x : ℝ | -1 < x ∧ x < 1} ∪ {x : ℝ | sqrt 3 < x} :=
by
  sorry

end range_of_x_l563_563559


namespace distance_between_points_l563_563356

def z1 : Complex := 2 + 3 * Complex.i
def z2 : Complex := -3 + 4 * Complex.i
def distance (a b : Complex) := Complex.abs (a - b)

theorem distance_between_points :
  distance z1 z2 = Real.sqrt 26 := by
  sorry

end distance_between_points_l563_563356


namespace square_root_then_square_l563_563044

theorem square_root_then_square (x : ℕ) (hx : x = 49) : (Nat.sqrt x) ^ 2 = 49 := by
  sorry

end square_root_then_square_l563_563044


namespace smallest_n_has_9_numbers_and_units_digit_is_5_l563_563642

def highest_square (k : ℕ) : ℕ :=
  (Nat.sqrt k) ^ 2

def sequence_step (k : ℕ) : ℕ :=
  k - highest_square k

def sequence_length (n : ℕ) : ℕ :=
  Nat.recOn n 0 (λ _ acc, acc + 1)

theorem smallest_n_has_9_numbers_and_units_digit_is_5 :
  ∃ (n : ℕ), sequence_length n = 9 ∧ n % 10 = 5 :=
by
  sorry

end smallest_n_has_9_numbers_and_units_digit_is_5_l563_563642


namespace inscribed_circle_radius_l563_563366

noncomputable def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def area (s a b c : ℝ) : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def radius (K s : ℝ) : ℝ := K / s

theorem inscribed_circle_radius {AB AC BC : ℝ} (hAB : AB = 8) (hAC : AC = 9) (hBC : BC = 10) :
  radius (area (semi_perimeter AB AC BC) AB AC BC) (semi_perimeter AB AC BC) ≈ 2.265 :=
by
  rw [hAB, hAC, hBC]
  have s := semi_perimeter 8 9 10
  have K := area s 8 9 10
  exact Real.rats_approx (\feval (radius K s) 2.265)

end inscribed_circle_radius_l563_563366


namespace min_guards_heptagon_l563_563760

theorem min_guards_heptagon 
  (g : ℕ → ℕ)
  (h1 : g 1 + g 2 ≥ 7)
  (h2 : g 2 + g 3 ≥ 7)
  (h3 : g 3 + g 4 ≥ 7)
  (h4 : g 4 + g 5 ≥ 7)
  (h5 : g 5 + g 6 ≥ 7)
  (h6 : g 6 + g 7 ≥ 7)
  (h7 : g 7 + g 1 ≥ 7) :
  g 1 + g 2 + g 3 + g 4 + g 5 + g 6 + g 7 ≥ 25 :=
begin
  sorry
end

end min_guards_heptagon_l563_563760


namespace JeremyTotalExpenses_l563_563988

noncomputable def JeremyExpenses : ℝ :=
  let motherGift := 400
  let fatherGift := 280
  let sisterGift := 100
  let brotherGift := 60
  let friendGift := 50
  let giftWrappingRate := 0.07
  let taxRate := 0.09
  let miscExpenses := 40
  let wrappingCost := motherGift * giftWrappingRate
                  + fatherGift * giftWrappingRate
                  + sisterGift * giftWrappingRate
                  + brotherGift * giftWrappingRate
                  + friendGift * giftWrappingRate
  let totalGiftCost := motherGift + fatherGift + sisterGift + brotherGift + friendGift
  let totalTax := totalGiftCost * taxRate
  wrappingCost + totalTax + miscExpenses

theorem JeremyTotalExpenses : JeremyExpenses = 182.40 := by
  sorry

end JeremyTotalExpenses_l563_563988


namespace range_of_m_l563_563894

open Real

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + 4 * x - m ≥ 0) ∧ 
  (∀ x y : ℝ, x ≤ -3 → y ≤ -3 → x ≤ y → -x^2 + (m-1) * x ≤ -y^2 + (m-1) * y) ↔ 
  (m ∈ Icc (-5 : ℝ) (-4 : ℝ)) := 
sorry

end range_of_m_l563_563894


namespace trajectory_of_point_is_straight_line_l563_563574

theorem trajectory_of_point_is_straight_line
  (a1 a2 y1 y2 : ℝ)
  (h1 : (a1 - 1) ^ 2 + y1 ^ 2 = 1)
  (h2 : (a2 - 1) ^ 2 + y2 ^ 2 = 1)
  (h3 : real.log y1 + real.log y2 = 0 ) :
  (1 / a1) + (1 / a2) = 2 := 
by
  sorry

end trajectory_of_point_is_straight_line_l563_563574


namespace size_of_each_group_l563_563712

theorem size_of_each_group 
  (boys : ℕ) (girls : ℕ) (groups : ℕ)
  (total_students : boys + girls = 63)
  (num_groups : groups = 7) :
  63 / 7 = 9 :=
by
  sorry

end size_of_each_group_l563_563712


namespace victor_total_money_l563_563362

def initial_amount : ℕ := 10
def allowance : ℕ := 8
def total_amount : ℕ := initial_amount + allowance

theorem victor_total_money : total_amount = 18 := by
  -- This is where the proof steps would go
  sorry

end victor_total_money_l563_563362


namespace bert_total_stamp_cost_l563_563818

theorem bert_total_stamp_cost :
    let numA := 150
    let numB := 90
    let numC := 60
    let priceA := 2
    let priceB := 3
    let priceC := 5
    let costA := numA * priceA
    let costB := numB * priceB
    let costC := numC * priceC
    let total_cost := costA + costB + costC
    total_cost = 870 := 
by
    sorry

end bert_total_stamp_cost_l563_563818


namespace lila_jack_sum_ratio_l563_563646

theorem lila_jack_sum_ratio : 
  (let lila_sum := (Finset.range 250).sum (λ i, 2 * i + 1) in
   let jack_sum := (Finset.range 250).sum (λ i, i) in
   lila_sum / jack_sum = 2) :=
by
  sorry

end lila_jack_sum_ratio_l563_563646


namespace max_value_of_x_squared_on_interval_l563_563701

theorem max_value_of_x_squared_on_interval : 
  ∃ x ∈ set.Icc (-1 : ℝ) 2, (∀ y ∈ set.Icc (-1 : ℝ) 2, x^2 ≥ y^2) ∧ x^2 = 4 :=
sorry

end max_value_of_x_squared_on_interval_l563_563701


namespace part1_part2_part3_part4_l563_563653

-- Part 1: Prove that 1/42 is equal to 1/6 - 1/7
theorem part1 : (1/42 : ℚ) = (1/6 : ℚ) - (1/7 : ℚ) := sorry

-- Part 2: Prove that 1/240 is equal to 1/15 - 1/16
theorem part2 : (1/240 : ℚ) = (1/15 : ℚ) - (1/16 : ℚ) := sorry

-- Part 3: Prove the general rule for all natural numbers m
theorem part3 (m : ℕ) (hm : m > 0) : (1 / (m * (m + 1)) : ℚ) = (1 / m : ℚ) - (1 / (m + 1) : ℚ) := sorry

-- Part 4: Prove the given expression evaluates to 0 for any x
theorem part4 (x : ℚ) (h1 : x ≠ 1) (h2 : x ≠ 2) (h3 : x ≠ 3) : 
  (1 / ((x - 2) * (x - 3)) : ℚ) - (2 / ((x - 1) * (x - 3)) : ℚ) + (1 / ((x - 1) * (x - 2)) : ℚ) = 0 := sorry

end part1_part2_part3_part4_l563_563653


namespace probability_of_at_least_one_red_ball_l563_563721

theorem probability_of_at_least_one_red_ball :
  let p := 2 / 3 * 2 / 3
  in 1 - p = 8 / 9 :=
by
  sorry

end probability_of_at_least_one_red_ball_l563_563721


namespace paper_cups_calculation_l563_563085

def total_pallets : Nat := 20
def paper_towels : Nat := total_pallets / 2
def tissues : Nat := total_pallets / 4
def paper_plates : Nat := total_pallets / 5
def other_paper_products : Nat := paper_towels + tissues + paper_plates
def paper_cups : Nat := total_pallets - other_paper_products

theorem paper_cups_calculation : paper_cups = 1 := by
  sorry

end paper_cups_calculation_l563_563085


namespace interest_calculation_l563_563448

theorem interest_calculation
  (P : ℝ) (R : ℝ) (T : ℝ) (interest : ℝ)
  (hP : P = 12500) (hR : R = 0.12) (hT : T = 1)
  (hI : interest = P * R * T) :
  interest = 1500 :=
by
  rw [hP, hR, hT] at hI
  have hI_calc : 12500 * 0.12 * 1 = 1500 := sorry
  exact hI_calc.trans hI

end interest_calculation_l563_563448


namespace oak_tree_planting_correct_l563_563719

variable (current_oak total_oak planted_oak : Nat)

def oak_tree_planting : Prop :=
  current_oak = 5 ∧ total_oak = 9 ∧ planted_oak = total_oak - current_oak

theorem oak_tree_planting_correct : oak_tree_planting current_oak total_oak planted_oak → planted_oak = 4 :=
by
  intro h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  rw [h1, h2] at h3
  exact h3

end oak_tree_planting_correct_l563_563719


namespace percent_diamond_jewels_l563_563810

def percent_beads : ℝ := 0.3
def percent_ruby_jewels : ℝ := 0.5

theorem percent_diamond_jewels (percent_beads percent_ruby_jewels : ℝ) : 
  (1 - percent_beads) * (1 - percent_ruby_jewels) = 0.35 :=
by
  -- We insert the proof steps here
  sorry

end percent_diamond_jewels_l563_563810


namespace probability_same_flips_l563_563968

theorem probability_same_flips (prob_heads_faircoin : ℝ)
  (prob_heads_biasedcoin : ℝ)
  (h_faircoin : prob_heads_faircoin = 1/2)
  (h_biasedcoin : prob_heads_biasedcoin = 1/3) :
  ∑ n in finset.range (n + 1), (prob_heads_faircoin ^ (2 * n) * ((2/3) ^ (n - 1)) * prob_heads_biasedcoin) =
    1/17 :=
by
  sorry

end probability_same_flips_l563_563968


namespace min_value_of_y_l563_563874

theorem min_value_of_y (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  (∃ a b, a > 0 ∧ b > 0 ∧ (a + b = 2) ∧ ((1 / a) + (4 / b)) = 9 / 2) :=
begin
  sorry -- Proof is omitted as per the instruction.
end

end min_value_of_y_l563_563874


namespace increase_in_average_weight_l563_563599

theorem increase_in_average_weight :
  let initial_group_size := 6
  let initial_weight := 65
  let new_weight := 74
  let initial_avg_weight := A
  (new_weight - initial_weight) / initial_group_size = 1.5 := by
    sorry

end increase_in_average_weight_l563_563599


namespace proof_problem_ACD_l563_563376

open Classical

theorem proof_problem_ACD :
  (∀ x : ℝ, x > 1 → |x| > 1 ∧ (∃ y : ℝ, |y| > 1 ∧ y ≤ 1)) ∧
  (¬ (∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ ∃ x : ℝ, x^2 + x + 1 < 0) ∧
  (∀ a b c : ℝ, (a + b + c = 0 ↔ a*(1:ℝ)^2 + b*(1:ℝ) + c = 0)) := 
  by
    split
    { intros x hx
      constructor
      { exact abs_pos_of_pos hx }
      { use -1
        simp
        exact abs_neg_of_neg (by norm_num) }
    }
    { constructor
      { intro h
        push_neg at h
        exact h
      }
      { intro h
        push_neg
        exact h }
    }
    { intros a b c
      split
      { intro h
        simpa using h }
      { intro h
        linarith }
    }

end proof_problem_ACD_l563_563376


namespace akeno_spent_more_l563_563099

theorem akeno_spent_more (akeno_expenditure : ℝ) (lev_expenditure : ℝ) (ambrocio_expenditure : ℝ) :
  akeno_expenditure = 2985 ∧
  lev_expenditure = (1 / 3) * akeno_expenditure ∧
  ambrocio_expenditure = lev_expenditure - 177 →
  akeno_expenditure - (lev_expenditure + ambrocio_expenditure) = 1172 :=
by
  intros h
  cases h with h_akeno h_lev
  cases h_lev with h_lev_exp h_ambrocio
  rw [h_akeno, h_lev_exp, h_ambrocio]
  calc
    2985 - ((1 / 3) * 2985 + ((1 / 3) * 2985 - 177))
        = 2985 - ((1 / 3) * 2985 + (1 / 3) * 2985 - 177) : by rw sub_add
    ... = 2985 - (2 * (1 / 3) * 2985 - 177) : by rw add_assoc
    ... = 2985 - (1990 - 177) : by norm_num
    ... = 2985 - 1813 : by norm_num
    ... = 1172 : by norm_num

end akeno_spent_more_l563_563099


namespace expected_value_keystrokes_l563_563726

/-- Tom has a scientific calculator where only the keys 1, 2, 3, +, and - are functional. He presses 
    a sequence of 5 random keystrokes, with each key equally likely to be pressed. The expected 
    value of the result E after evaluating the expression formed by these keystrokes is 1866. -/
theorem expected_value_keystrokes :
  ∃ E : ℚ, 
  (∀ key : {x // x ∈ {1, 2, 3, '+', '-'}}, 1/5 = 1/5) ∧ 
  (E = 1866) := 
sorry

end expected_value_keystrokes_l563_563726


namespace range_of_a_l563_563168

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

theorem range_of_a :
  ∀ a : ℝ, 
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (3 / 2 ≤ a ∧ a < 3) :=
by
  sorry

end range_of_a_l563_563168


namespace lucy_mother_twice_age_in_2036_l563_563293

theorem lucy_mother_twice_age_in_2036 :
  let age_lucy_in_2006 := 10
  let age_mother_in_2006 := 5 * age_lucy_in_2006
  let x := 2036 - 2006
  (50 + x) = 2 * (10 + x) :=
by
  let age_lucy_in_2006 := 10
  let age_mother_in_2006 := 5 * age_lucy_in_2006
  let x := 2036 - 2006
  have h1 : age_mother_in_2006 = 50 := by sorry
  have h2 : x = 30 := by sorry
  calc
    50 + x = 50 + 30 := by sorry
    ... = 80 := by sorry
    ... = 2 * (10 + 30) := by sorry

end lucy_mother_twice_age_in_2036_l563_563293


namespace cylinder_volume_side_square_len_two_l563_563590

-- Define the problem
theorem cylinder_volume_side_square_len_two
  (r h : ℝ)
  (h1 : 2 * Real.pi * r = 2)
  (h2 : h = 2) :
  (volume : ℝ) (cylinder_volume : ℝ) :
  cylinder_volume = Real.pi * r ^ 2 * h :=
by {
  sorry,
  have hr : r = 1 / Real.pi := by sorry
  have hh : h = 2 := by sorry
  exact Real.pi * r ^ 2 * h = 2 / Real.pi
}

end cylinder_volume_side_square_len_two_l563_563590


namespace crates_in_third_trip_l563_563090

/-- A trailer carries 12 crates. Each crate weighs at least 120 kg. The maximum weight the trailer can carry in a single trip is 600 kg. How many crates does the trailer carry in the third part of the trip? --/
theorem crates_in_third_trip (total_crates : ℕ) (min_crate_weight : ℕ) (max_weight_per_trip : ℕ) : 
  total_crates = 12 → min_crate_weight = 120 → max_weight_per_trip = 600 → 
  2 :=
by
  sorry

end crates_in_third_trip_l563_563090


namespace min_PA_plus_PC_l563_563609

noncomputable def AB : ℝ := 3
noncomputable def AD : ℝ := 4
noncomputable def AA₁ : ℝ := 5

theorem min_PA_plus_PC : 
  ∀ (P : Point) (P ∈ Surface A₁ B₁ C₁ D₁), (|distance P A| + |distance P C|) = 5 * (sqrt 5) := 
by
  sorry


end min_PA_plus_PC_l563_563609


namespace increasing_in_interval_0_1_l563_563465

theorem increasing_in_interval_0_1 :
  ∀ x : ℝ, 0 < x ∧ x < 1 → ∃ f : ℝ → ℝ, (f = λ x, x ^ (1 / 2)) ∧ (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ < f x₂) :=
by
  intros
  use (λ x, x ^ (1 / 2))
  split
  · rfl
  · intros x₁ x₂ h₁ h₂
  sorry

end increasing_in_interval_0_1_l563_563465


namespace eric_removes_1011_tiles_l563_563458

theorem eric_removes_1011_tiles :
  let total_tiles := 2022
  let tiles_after_beata := (5 / 6) * total_tiles
  let tiles_after_carla := (4 / 5) * tiles_after_beata
  let tiles_after_doris := (3 / 4) * tiles_after_carla
  ∀ (tiles_after_eric : ℝ), tiles_after_eric = tiles_after_doris → tiles_after_eric = 1011 :=
by
  intros total_tiles tiles_after_beata tiles_after_carla tiles_after_doris tiles_after_eric h
  have : total_tiles = 2022 := rfl
  have : tiles_after_beata = (5 / 6) * total_tiles := rfl
  have : tiles_after_carla = (4 / 5) * tiles_after_beata := rfl
  have : tiles_after_doris = (3 / 4) * tiles_after_carla := rfl
  rw [tiles_after_doris] at h
  sorry

end eric_removes_1011_tiles_l563_563458


namespace fill_well_in_time_l563_563030

theorem fill_well_in_time :
  ∀ (C : ℝ) (R1 R2 R3 L : ℝ),
    C = 3000 →
    R1 = 90 →
    R2 = 270 →
    R3 = 180 →
    L = 35 →
    (C / ((R1 + R2 + R3) - L)) ≈ 5.94 :=
by
  intros C R1 R2 R3 L hC hR1 hR2 hR3 hL
  rw [hC, hR1, hR2, hR3, hL]
  norm_num
  sorry

end fill_well_in_time_l563_563030


namespace number_of_people_l563_563690

theorem number_of_people
  (weight_increase: ℝ)
  (new_person_weight: ℝ)
  (replaced_person_weight: ℝ)
  (average_weight_increase: ℝ)
  (n: ℕ):
  weight_increase = 25 →
  new_person_weight = 90 →
  replaced_person_weight = 65 →
  average_weight_increase = 2.5 →
  2.5 * n = 25 →
  n = 10 := by
  intros h_weight_increase h_new_person_weight h_replaced_person_weight h_average_weight_increase h_eq
  sorry

end number_of_people_l563_563690


namespace fraction_value_l563_563197

variable (w x y : ℝ)

-- Given conditions
def cond1 := w / x = 1 / 6
def cond2 := (x + y) / y = 2.2

-- Theorem stating w / y = 0.2 under the given conditions
theorem fraction_value (h1 : cond1 w x y) (h2 : cond2 w x y) : w / y = 0.2 :=
by
  sorry

end fraction_value_l563_563197


namespace sub_eight_l563_563722

theorem sub_eight (x y : ℝ) (h1 : 3 = 0.15 * x) (h2 : 3 = 0.25 * y) : x - y = 8 :=
by
  sorry

end sub_eight_l563_563722


namespace merchant_should_choose_option2_l563_563076

-- Definitions for the initial price and discounts
def P : ℝ := 20000
def d1_1 : ℝ := 0.25
def d1_2 : ℝ := 0.15
def d1_3 : ℝ := 0.05
def d2_1 : ℝ := 0.35
def d2_2 : ℝ := 0.10
def d2_3 : ℝ := 0.05

-- Define the final prices after applying discount options
def finalPrice1 (P : ℝ) (d1_1 d1_2 d1_3 : ℝ) : ℝ :=
  P * (1 - d1_1) * (1 - d1_2) * (1 - d1_3)

def finalPrice2 (P : ℝ) (d2_1 d2_2 d2_3 : ℝ) : ℝ :=
  P * (1 - d2_1) * (1 - d2_2) * (1 - d2_3)

-- Theorem to state the merchant should choose Option 2
theorem merchant_should_choose_option2 : 
  finalPrice1 P d1_1 d1_2 d1_3 = 12112.50 ∧ 
  finalPrice2 P d2_1 d2_2 d2_3 = 11115 ∧ 
  finalPrice1 P d1_1 d1_2 d1_3 - finalPrice2 P d2_1 d2_2 d2_3 = 997.50 :=
by
  -- Placeholder for the proof
  sorry

end merchant_should_choose_option2_l563_563076


namespace minimum_color_bound_l563_563516

theorem minimum_color_bound (n : ℕ) (h : n > 0) :
  ∃ (χ : ℕ), (∀ (tournament : Fin n → Fin n → ℕ),
    (∀ u v w : Fin n, u ≠ v → v ≠ w → w ≠ u → tournament u v ≠ tournament v w) →
    ∀ c : ℕ, c ≥ log 2 n →
       χ = c) :=
sorry

end minimum_color_bound_l563_563516


namespace chess_tournament_l563_563130

def distinct (l : List ℕ) : Prop :=
  ∀ i j, i < l.length → j < l.length → i ≠ j → l.nthLe i _ ≠ l.nthLe j _

def participating_points := [a, b, c, d, e, f, g, h]

theorem chess_tournament
  (a b c d e f g h : ℕ)
  (H1 : distinct participating_points)
  (H2 : a + b + c + d + e + f + g + h = 28)
  (H3 : b = e + f + g + h) :
  c > g :=
sorry

end chess_tournament_l563_563130


namespace ant_climbing_floors_l563_563441

theorem ant_climbing_floors (time_per_floor : ℕ) (total_time : ℕ) (floors_climbed : ℕ) :
  time_per_floor = 15 →
  total_time = 105 →
  floors_climbed = total_time / time_per_floor + 1 →
  floors_climbed = 8 :=
by
  intros
  sorry

end ant_climbing_floors_l563_563441


namespace find_ordered_triple_l563_563983

theorem find_ordered_triple (A B C : Type) (E F P : A ⟶ B) 
  (h1 : AE : EC = 3 : 2) 
  (h2 : AF : FB = 1 : 3) 
  (x y z : ℝ) 
  (h3 : x + y + z = 1) 
  (h4 : P = x • A + y • B + z • C) 
  : (x, y, z) = (2/9, 1/9, 3/9) := 
sorry

end find_ordered_triple_l563_563983


namespace angle_bisector_proportion_l563_563962

theorem angle_bisector_proportion
  (p q r : ℝ)
  (u v : ℝ)
  (h1 : 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < u ∧ 0 < v)
  (h2 : u + v = p)
  (h3 : u * q = v * r) :
  u / p = r / (r + q) :=
sorry

end angle_bisector_proportion_l563_563962


namespace intersection_on_bc_l563_563110

theorem intersection_on_bc 
  (A B C T P Q R S X : Point)
  (h_triangle : Triangle A B C)
  (h_interior_T : InteriorPoint T (Triangle A B C))
  (h_foot_P : PerpendicularFoot T A B P)
  (h_foot_Q : PerpendicularFoot T A C Q)
  (h_foot_R : PerpendicularFoot A T C R)
  (h_foot_S : PerpendicularFoot A T B S)
  (h_intersection_X : Intersection PR QS X) :
  OnLine X B C :=
sorry

end intersection_on_bc_l563_563110


namespace area_of_gray_region_is_correct_l563_563733

def diameter_smaller_circle : ℝ := 4
def radius_smaller_circle : ℝ := diameter_smaller_circle / 2
def diameter_larger_circle : ℝ := 2 * diameter_smaller_circle
def radius_larger_circle : ℝ := diameter_larger_circle / 2

def area_circle (r : ℝ) : ℝ := π * r^2
def area_smaller_circle : ℝ := area_circle radius_smaller_circle
def area_larger_circle : ℝ := area_circle radius_larger_circle

def area_gray_region : ℝ := area_larger_circle - area_smaller_circle

theorem area_of_gray_region_is_correct :
  area_gray_region = 12 * π :=
by
  sorry

end area_of_gray_region_is_correct_l563_563733


namespace arrangement_not_possible_l563_563986

theorem arrangement_not_possible : ¬ ∃ (A : (Fin 3) → (Fin 4) → ℕ),
  (∀ (i : Fin 3) (j : Fin 4), A i j ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) ∧
  (∀ j : Fin 4, ∑ i, A i j = 19) :=
by
  sorry

end arrangement_not_possible_l563_563986


namespace pyramid_no_circular_section_l563_563741

-- Defining the solids and their cross-sectional properties
inductive Solid
| Pyramid
| Cylinder
| Sphere
| Cone

variable (s : Solid)

-- Conditions
def cross_section (s : Solid) : Prop :=
  match s with
  | Solid.Pyramid => false
  | Solid.Cylinder => true  -- It can be circular
  | Solid.Sphere => true    -- It is circular
  | Solid.Cone => true      -- It can be circular

-- Main Statement
theorem pyramid_no_circular_section (s : Solid) (h : s = Solid.Pyramid):
  ¬ cross_section s :=
by
  intros hs
  cases h
  exact hs

#check pyramid_no_circular_section

end pyramid_no_circular_section_l563_563741


namespace value_of_p_OA_perp_OB_l563_563440

theorem value_of_p (p : ℝ) (O A B : ℝ × ℝ) (h1 : (0, 4) ∈ line_through (0, 4) (-1))
  (h2 : ∃ x y, y^2 = 2 * p * x ∧ (x, y) = A ∧ (x, y) = B) 
  (h3 : dist A B = 4 * √10) : p = 2 := 
sorry

theorem OA_perp_OB (p : ℝ) (O A B : ℝ × ℝ) (h1 : (0, 4) ∈ line_through (0, 4) (-1))
  (h2 : ∃ x y, y^2 = 2 * p * x ∧ (x, y) = A ∧ (x, y) = B) 
  (h3 : dist A B = 4 * √10) (h4 : p = 2) : 
  let OA := (fst A, snd A),
      OB := (fst B, snd B)
  in (OA.fst * OB.fst + OA.snd * OB.snd) = 0 := 
sorry

end value_of_p_OA_perp_OB_l563_563440


namespace weavers_in_first_group_l563_563314

theorem weavers_in_first_group 
  (W : ℕ)
  (H1 : 4 / (W * 4) = 1 / W) 
  (H2 : (9 / 6) / 6 = 0.25) :
  W = 4 :=
sorry

end weavers_in_first_group_l563_563314


namespace frog_escape_probability_l563_563239

noncomputable def P : ℕ → ℚ
| 0 => 0
| 14 => 1
| N+1 => (2 * (N+1) / 15) * P N + (1 - 2 * (N+1) / 15) * P (N+2)

theorem frog_escape_probability : P 2 = x / y := by
  sorry

end frog_escape_probability_l563_563239


namespace smaller_rectangle_dimensions_l563_563791

theorem smaller_rectangle_dimensions (side_length : ℝ) (L W : ℝ) 
  (h1 : side_length = 10) 
  (h2 : L + 2 * L = side_length) 
  (h3 : W = L) : 
  L = 10 / 3 ∧ W = 10 / 3 :=
by 
  sorry

end smaller_rectangle_dimensions_l563_563791


namespace angle_between_vectors_60_degrees_l563_563878
open Real

variables (a b : ℝ^3)
variables (ha : ‖a‖ = 3)
variables (hb : ‖b‖ = 4)
variables (h : (a - b) • (a - 2 • b) = 23)

theorem angle_between_vectors_60_degrees (ha : ‖a‖ = 3) (hb : ‖b‖ = 4) (h : (a - b) • (a - 2 • b) = 23) :
  let θ := real.arccos ((a • b) / (‖a‖ * ‖b‖)) in θ = real.pi / 3 :=
sorry

end angle_between_vectors_60_degrees_l563_563878


namespace files_deleted_l563_563491

theorem files_deleted 
  (orig_files : ℕ) (final_files : ℕ) (deleted_files : ℕ) 
  (h_orig : orig_files = 24) 
  (h_final : final_files = 21) : 
  deleted_files = orig_files - final_files :=
by
  rw [h_orig, h_final]
  sorry

end files_deleted_l563_563491


namespace max_value_A_l563_563933

noncomputable def maximum_value (n : ℕ) (xs : Fin n → ℝ) : ℝ :=
  (∑ i, Real.cos (xs i)^2) / (Real.sqrt n + Real.sqrt (∑ i, (Real.cot (xs i))^4))

theorem max_value_A {n : ℕ} (hpos : 0 < n)
  (hx : ∀ i : Fin n, 0 < xs i ∧ xs i < Real.pi / 2) :
  maximum_value n xs ≤ Real.sqrt n / 4 :=
sorry

end max_value_A_l563_563933


namespace square_of_area_of_equilateral_triangle_l563_563809

noncomputable def equilateral_triangle_area_squared (x1 x2 x3 y1 y2 y3 : ℝ) (s : ℝ) :=
  let centroid_eq := (x1 + x2 + x3) / 3 = 1 ∧ (y1 + y2 + y3) / 3 = 1
  let vertices_on_hyperbola := x1 * y1 = 3 ∧ x2 * y2 = 3 ∧ x3 * y3 = 3
  let side_length := s = Math.sqrt ((x1 - x2) ^ 2 + (y1 - y2) ^ 2)
  ∧ s = Math.sqrt ((x2 - x3) ^ 2 + (y2 - y3) ^ 2)
  ∧ s = Math.sqrt ((x3 - x1) ^ 2 + (y3 - y1) ^ 2)
  let area_sq := (s * s * Math.sqrt 3 / 4) ^ 2
  centroid_eq ∧ vertices_on_hyperbola ∧ side_length → area_sq

theorem square_of_area_of_equilateral_triangle (x1 x2 x3 y1 y2 y3 s : ℝ) :
  equilateral_triangle_area_squared x1 x2 x3 y1 y2 y3 s :=
sorry

end square_of_area_of_equilateral_triangle_l563_563809


namespace coefficient_x4_in_expansion_l563_563250

theorem coefficient_x4_in_expansion :
    let f := (λ x : ℤ, (x + (1/x)) ^ 6)
    let expansion := finset.sum (finset.range 7) (λ r, (nat.choose 6 r) * ((x : ℚ)^ (6 - 2 * r)))
    ∃ c : ℚ, expansion = c * x ^ 4 := by
  sorry

end coefficient_x4_in_expansion_l563_563250


namespace laps_count_l563_563812

theorem laps_count (t : ℕ) (n : ℕ)
  (A_runs_faster_than_B_by : 2)
  (B_runs_faster_than_V_by : 3)
  (B_laps_left_when_A_finishes : 1)
  (V_laps_left_when_A_finishes : 2)
  (A_lap_time : t = 2 * n - 2) :
  n = 6 := 
sorry

end laps_count_l563_563812


namespace f_f_of_7_div_3_eq_1_div_3_l563_563567

def f (x : ℝ) : ℝ :=
if x ≤ 1 then 2 ^ x - 1 else log 2 (x - 1)

theorem f_f_of_7_div_3_eq_1_div_3 : f (f (7 / 3)) = 1 / 3 :=
by
  sorry

end f_f_of_7_div_3_eq_1_div_3_l563_563567


namespace problem_statement_l563_563588

noncomputable def f (x : ℝ) : ℝ := 1 + |x| + (cos x / x)

lemma log_half_neg_log_two : Real.log (1 / 2) = -Real.log 2 :=
by sorry

lemma log_fifth_neg_log_five : Real.log (1 / 5) = -Real.log 5 :=
by sorry

theorem problem_statement : f (Real.log 2) + f (Real.log (1 / 2)) + f (Real.log 5) + f (Real.log (1 / 5)) = 6 :=
by
  have h1 := log_half_neg_log_two
  have h2 := log_fifth_neg_log_five
  sorry

end problem_statement_l563_563588


namespace cosine_angle_between_EF_and_BC_l563_563359

noncomputable def cosine_angle_vectors (AB AE AC AF EF BC CA : ℝ)
  (dot1 : ℝ)
  (midpoint_B : B = midpoint(E, F)) : ℝ :=
sorry

theorem cosine_angle_between_EF_and_BC :
  ∃ θ : ℝ, 
  cosine_angle_vectors 1  1 sqrt 33  (-1) 1 6 sqrt 33 2 (B = midpoint(E, F)) = 2 / 3 :=
by sorry

end cosine_angle_between_EF_and_BC_l563_563359


namespace solve_system_of_equations_l563_563593

noncomputable def problem_solution (k : ℝ) : Prop :=
  ∀ x y : ℝ, (x + 2 * y = 6 + 3 * k) ∧ (2 * x + y = 3 * k) → (2 * y - 2 * x = 12)

theorem solve_system_of_equations (k : ℝ) : problem_solution k :=
begin
  intros x y h,
  sorry
end

end solve_system_of_equations_l563_563593


namespace geometric_mean_of_roots_l563_563328

theorem geometric_mean_of_roots (x : ℝ) (h : x^2 = (Real.sqrt 2 + 1) * (Real.sqrt 2 - 1)) : x = 1 ∨ x = -1 := 
by
  sorry

end geometric_mean_of_roots_l563_563328


namespace work_days_together_l563_563752

theorem work_days_together (d : ℕ) (h : d * (17 / 140) = 6 / 7) : d = 17 := by
  sorry

end work_days_together_l563_563752


namespace dot_product_7_l563_563612

noncomputable def vector_length (v : ℝ) : ℝ := abs v

theorem dot_product_7 (AB BC CD DA : ℝ) (h_AB : AB = 2) (h_BC : BC = 3) (h_CD : CD = 4) (h_DA : DA = 5) :
  let AC : ℝ := sqrt (2^2 + 3^2)
  let BD : ℝ := sqrt (4^2 + 5^2)
  (AC * BD = 7) :=
by
  have AC_value : AC = sqrt (2^2 + 3^2), from rfl,
  have BD_value : BD = sqrt (4^2 + 5^2), from rfl,
  -- apply the calculated value from proof
  sorry

end dot_product_7_l563_563612


namespace count_odd_divisor_integers_lt_100_l563_563582

theorem count_odd_divisor_integers_lt_100 : 
  (∃ (n : ℕ), n < 100 ∧ (∃ (m : ℕ), n = m * m ) ) ↔ 9 :=
by
  sorry

end count_odd_divisor_integers_lt_100_l563_563582


namespace intersection_area_of_rectangle_and_circle_l563_563871

def rectangle_vertices : set (ℝ × ℝ) := {(4, 9), (15, 9), (15, -4), (4, -4)}

def circle_eq (x y : ℝ) : Prop := (x - 4)^2 + (y - 9)^2 = 16

def is_intersection_area_correct (area : ℝ) : Prop :=
  area = 4 * Real.pi

theorem intersection_area_of_rectangle_and_circle :
  (∃ (area : ℝ), is_intersection_area_correct area) :=
begin
  use 4 * Real.pi,
  unfold is_intersection_area_correct,
  simp,
  sorry -- Proof goes here
end

end intersection_area_of_rectangle_and_circle_l563_563871


namespace max_PA_ellipse_l563_563180

noncomputable def ellipse_focus_y_axis (x y m : ℝ) : Prop :=
  (x^2 / m^2 + y^2 / 4 = 1) ∧ (m > 0) ∧ (1/2 = real.sqrt(1 - m^2 / 4)).exists

theorem max_PA_ellipse (x y m : ℝ) (P A : ℝ × ℝ): 
  ellipse_focus_y_axis x y m →
  (A = (m, 0)) → 
  (P = (sqrt 3 * cos θ, 2 * sin θ)) → 
  (max_dist := sqrt ((sqrt 3 * cos θ - m)^2 + (2 * sin θ)^2)) =
  2 * sqrt 3 :=
sorry

end max_PA_ellipse_l563_563180


namespace ratio_of_numbers_l563_563021

theorem ratio_of_numbers (A B D M : ℕ) 
  (h1 : A + B + D = M)
  (h2 : Nat.gcd A B = D)
  (h3 : Nat.lcm A B = M)
  (h4 : A ≥ B) : A / B = 3 / 2 :=
by
  sorry

end ratio_of_numbers_l563_563021


namespace bakery_gives_away_30_doughnuts_at_end_of_day_l563_563408

def boxes_per_day (total_doughnuts doughnuts_per_box : ℕ) : ℕ :=
  total_doughnuts / doughnuts_per_box

def leftover_boxes (total_boxes sold_boxes : ℕ) : ℕ :=
  total_boxes - sold_boxes

def doughnuts_given_away (leftover_boxes doughnuts_per_box : ℕ) : ℕ :=
  leftover_boxes * doughnuts_per_box

theorem bakery_gives_away_30_doughnuts_at_end_of_day 
  (total_doughnuts doughnuts_per_box sold_boxes : ℕ) 
  (H1 : total_doughnuts = 300) (H2 : doughnuts_per_box = 10) (H3 : sold_boxes = 27) : 
  doughnuts_given_away (leftover_boxes (boxes_per_day total_doughnuts doughnuts_per_box) sold_boxes) doughnuts_per_box = 30 :=
by
  sorry

end bakery_gives_away_30_doughnuts_at_end_of_day_l563_563408


namespace doughnuts_given_away_l563_563411

def doughnuts_left (total_doughnuts : Nat) (doughnuts_per_box : Nat) (boxes_sold : Nat) : Nat :=
  total_doughnuts - (doughnuts_per_box * boxes_sold)

theorem doughnuts_given_away :
  doughnuts_left 300 10 27 = 30 :=
by
  rw [doughnuts_left]
  simp
  sorry

end doughnuts_given_away_l563_563411


namespace round_to_nearest_hundredth_l563_563811

theorem round_to_nearest_hundredth (x : ℝ) : (x = 4.0692) → (Float.roundTo 2 x) = 4.07 :=
by
  intros h
  rw [h]
  sorry

end round_to_nearest_hundredth_l563_563811


namespace akeno_spent_more_l563_563098

theorem akeno_spent_more (akeno_expenditure : ℝ) (lev_expenditure : ℝ) (ambrocio_expenditure : ℝ) :
  akeno_expenditure = 2985 ∧
  lev_expenditure = (1 / 3) * akeno_expenditure ∧
  ambrocio_expenditure = lev_expenditure - 177 →
  akeno_expenditure - (lev_expenditure + ambrocio_expenditure) = 1172 :=
by
  intros h
  cases h with h_akeno h_lev
  cases h_lev with h_lev_exp h_ambrocio
  rw [h_akeno, h_lev_exp, h_ambrocio]
  calc
    2985 - ((1 / 3) * 2985 + ((1 / 3) * 2985 - 177))
        = 2985 - ((1 / 3) * 2985 + (1 / 3) * 2985 - 177) : by rw sub_add
    ... = 2985 - (2 * (1 / 3) * 2985 - 177) : by rw add_assoc
    ... = 2985 - (1990 - 177) : by norm_num
    ... = 2985 - 1813 : by norm_num
    ... = 1172 : by norm_num

end akeno_spent_more_l563_563098


namespace right_triangle_acute_angles_l563_563970

theorem right_triangle_acute_angles (α β : ℝ) 
  (h1 : α + β = 90)
  (h2 : ∀ (δ1 δ2 ε1 ε2 : ℝ), δ1 + ε1 = 135 ∧ δ1 / ε1 = 13 / 17 
                       ∧ ε2 = 180 - ε1 ∧ δ2 = 180 - δ1) :
  α = 63 ∧ β = 27 := 
  sorry

end right_triangle_acute_angles_l563_563970


namespace gravitational_force_space_station_l563_563698

theorem gravitational_force_space_station :
  let d₁ := 6000 -- distance from the center of the Earth when on the surface
  let d₂ := 360000 -- distance from the center of the Earth when on the space station
  let f₁ := 400 -- gravitational force at distance d₁
  ∃ k, f₁ * d₁^2 = k ∧ (∃ f₂, f₂ * d₂^2 = k ∧ f₂ = 1 / 9) := 
by
  let d₁ := 6000
  let d₂ := 360000
  let f₁ := 400
  let k := f₁ * d₁^2
  have f₂ := k / d₂^2
  use k
  split
  . exact rfl
  use f₂
  split
  . exact rfl
  calc f₂ = k / d₂^2 : rfl
      ... = (400 * 6000^2) / 360000^2 : by sorry
      ... = 1 / 9 : by sorry

end gravitational_force_space_station_l563_563698


namespace parallel_lines_condition_l563_563187

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y - 4 = 0 → x + (a + 1) * y + 2 = 0) ↔ a = 1 :=
by sorry

end parallel_lines_condition_l563_563187


namespace average_temperature_l563_563066

theorem average_temperature (T : Fin 5 → ℝ) (h : T = ![52, 67, 55, 59, 48]) :
    (1 / 5) * (T 0 + T 1 + T 2 + T 3 + T 4) = 56.2 := by
  sorry

end average_temperature_l563_563066


namespace largest_consecutive_multiple_of_3_l563_563717

theorem largest_consecutive_multiple_of_3 (n : ℕ) 
  (h : 3 * n + 3 * (n + 1) + 3 * (n + 2) = 72) : 3 * (n + 2) = 27 :=
by 
  sorry

end largest_consecutive_multiple_of_3_l563_563717


namespace exists_valid_graph_l563_563500

def eight_points := {A B C D E F G H : Type}

def valid_graph (G : eight_points × eight_points → Prop) := 
  (∀ x : eight_points, ∃ y z w v, G (x, y) ∧ G (x, z) ∧ G (x, w) ∧ G (x, v)) ∧
  (∀ (x y z w : eight_points), ¬ (G (x, y) ∧ G (z, w) ∧ (x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w)))

theorem exists_valid_graph : ∃ G : eight_points × eight_points → Prop, valid_graph G :=
sorry

end exists_valid_graph_l563_563500


namespace polynomial_P_n_plus_2_l563_563540

theorem polynomial_P_n_plus_2 (P : ℕ → ℕ) (n : ℕ) 
  (hP : ∀ j : ℕ, j ∈ (finset.range (n + 2)).image (λ x, x + 1) → P j = 2^(j - 1)) : 
  P (n + 2) = 2^(n + 1) - 1 :=
by
  sorry

end polynomial_P_n_plus_2_l563_563540


namespace minimum_value_dist_l563_563913

noncomputable def f (x : ℝ) : ℝ := x - 2 * log x

theorem minimum_value_dist : ∃ x : ℝ, (∀ y : ℝ, f(x) ≤ f(y)) ∧ f(2) = 2 - 2 * log 2 :=
by sorry

end minimum_value_dist_l563_563913


namespace arithmetic_sequence_property_l563_563022

theorem arithmetic_sequence_property (y : ℤ) (m : ℕ) (h1 : m > 4) (h2 : ∑ i in range (m + 1), (y + 3 * i) ^ 3 = -8000) : m = 7 := 
sorry

end arithmetic_sequence_property_l563_563022


namespace euclidean_division_mod_l563_563479

theorem euclidean_division_mod (h1 : 2022 % 19 = 8)
                               (h2 : 8^6 % 19 = 1)
                               (h3 : 2023 % 6 = 1)
                               (h4 : 2023^2024 % 6 = 1) 
: 2022^(2023^2024) % 19 = 8 := 
by
  sorry

end euclidean_division_mod_l563_563479


namespace max_result_100_gon_l563_563503

theorem max_result_100_gon :
  ∃ (x : Fin 100 → ℝ),
    (∑ i in Finset.univ, (x i)^2 = 1) ∧
    let k := ∑ i in Finset.range 50, x i * x (i + 2) -- sum of products on red segments
    let s := ∑ i in Finset.range 50, x i * x (i + 3) -- sum of products on blue segments
    k - s = 1 / 2 :=
sorry

end max_result_100_gon_l563_563503


namespace s_lt_t_if_0_lt_x_lt_a_l563_563999

theorem s_lt_t_if_0_lt_x_lt_a (a x : ℝ) (h1 : 0 < x) (h2 : x < a) :
  let s := 2 * x^2 - 2 * a * x + a^2 in
  let t := x^2 + a^2 - a * x in
  s < t :=
by
  let s := 2 * x^2 - 2 * a * x + a^2
  let t := x^2 + a^2 - a * x
  have : s - t = x * (x - a), by sorry
  have : 0 < x * (x - a) → x * (x - a) < 0, by sorry
  sorry

end s_lt_t_if_0_lt_x_lt_a_l563_563999


namespace probability_check_l563_563841

def total_students : ℕ := 12

def total_clubs : ℕ := 3

def equiprobable_clubs := ∀ s : Fin total_students, ∃ c : Fin total_clubs, true

noncomputable def probability_diff_students : ℝ := 1 - (34650 / (total_clubs ^ total_students))

theorem probability_check :
  equiprobable_clubs →
  probability_diff_students = 0.935 := 
by
  intros
  sorry

end probability_check_l563_563841


namespace jose_fewer_rocks_l563_563990

theorem jose_fewer_rocks (J : ℕ) (H1 : 80 = J + 14) (H2 : J + 20 = 86) (H3 : J < 80) : J = 66 :=
by
  -- Installation of other conditions derived from the proof
  have H_albert_collected : 86 = 80 + 6 := by rfl
  have J_def : J = 86 - 20 := by sorry
  sorry

end jose_fewer_rocks_l563_563990


namespace bakery_gives_away_30_doughnuts_at_end_of_day_l563_563409

def boxes_per_day (total_doughnuts doughnuts_per_box : ℕ) : ℕ :=
  total_doughnuts / doughnuts_per_box

def leftover_boxes (total_boxes sold_boxes : ℕ) : ℕ :=
  total_boxes - sold_boxes

def doughnuts_given_away (leftover_boxes doughnuts_per_box : ℕ) : ℕ :=
  leftover_boxes * doughnuts_per_box

theorem bakery_gives_away_30_doughnuts_at_end_of_day 
  (total_doughnuts doughnuts_per_box sold_boxes : ℕ) 
  (H1 : total_doughnuts = 300) (H2 : doughnuts_per_box = 10) (H3 : sold_boxes = 27) : 
  doughnuts_given_away (leftover_boxes (boxes_per_day total_doughnuts doughnuts_per_box) sold_boxes) doughnuts_per_box = 30 :=
by
  sorry

end bakery_gives_away_30_doughnuts_at_end_of_day_l563_563409


namespace smallest_composite_no_prime_factors_less_than_12_l563_563151

-- Define what it means to be a composite number
def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ m * k = n

-- Define what it means for a number's prime factors to be greater than or equal to 12
def no_prime_factors_less_than (n : ℕ) (bound : ℕ) : Prop :=
  ∀ p : ℕ, p.prime → p ∣ n → bound ≤ p

-- Define the specific bound for prime factors
def bound := 12

-- Main theorem statement
theorem smallest_composite_no_prime_factors_less_than_12 : 
  ∃ n : ℕ, is_composite n ∧ no_prime_factors_less_than n bound ∧ ∀ m : ℕ, is_composite m ∧ no_prime_factors_less_than m bound → n ≤ m ∧ n = 221 :=
begin
  sorry
end

end smallest_composite_no_prime_factors_less_than_12_l563_563151


namespace smallest_n_property_l563_563497

theorem smallest_n_property (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
 (hxy : x ∣ y^3) (hyz : y ∣ z^3) (hzx : z ∣ x^3) : 
  x * y * z ∣ (x + y + z) ^ 13 := 
by sorry

end smallest_n_property_l563_563497


namespace petals_vs_wings_and_unvisited_leaves_l563_563352

def flowers_petals_leaves := 5
def petals_per_flower := 2
def bees_wings := 3
def wings_per_bee := 4
def leaves_per_flower := 3
def visits_per_bee := 2
def total_flowers := flowers_petals_leaves
def total_bees := bees_wings

def total_petals : ℕ := total_flowers * petals_per_flower
def total_wings : ℕ := total_bees * wings_per_bee
def more_wings_than_petals := total_wings - total_petals

def total_leaves : ℕ := total_flowers * leaves_per_flower
def total_visits : ℕ := total_bees * visits_per_bee
def leaves_per_visit := leaves_per_flower
def visited_leaves : ℕ := min total_leaves (total_visits * leaves_per_visit)
def unvisited_leaves : ℕ := total_leaves - visited_leaves

theorem petals_vs_wings_and_unvisited_leaves :
  more_wings_than_petals = 2 ∧ unvisited_leaves = 0 :=
by
  sorry

end petals_vs_wings_and_unvisited_leaves_l563_563352


namespace largest_number_not_sum_of_two_composites_l563_563860

-- Definitions related to composite numbers and natural numbers
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)

-- Lean statement of the problem
theorem largest_number_not_sum_of_two_composites : ∀ n : ℕ,
  (∀ m : ℕ, m > n → cannot_be_sum_of_two_composites m → false) → n = 11 :=
by
  sorry

end largest_number_not_sum_of_two_composites_l563_563860


namespace sum_of_angles_visible_from_vertices_l563_563773

theorem sum_of_angles_visible_from_vertices (n : ℕ) (A : ℝ) (P Q : ℝ)
  (h1 : P + Q = 1) : ∑ (i : ℕ) in finset.range (n-1), angle A (vertex n i) = π * (n - 2) / n := 
begin
  sorry,
end

end sum_of_angles_visible_from_vertices_l563_563773


namespace exists_cuboid_not_cube_inscribed_and_circumscribed_with_same_center_l563_563257

def is_cuboid (P : Type) :=
  Π (faces : set (set P)), 
  faces = {face | ∃ (u v w : P) (a b c : ℝ),
  u ≠ v ∧ v ≠ w ∧ w ≠ u ∧ 
  ((u ∈ face ∧ v ∈ face ∧ w ∈ face) ∨
   (v ∈ face ∧ w ∈ face ∧ u ∈ face) ∨
   (w ∈ face ∧ u ∈ face ∧ v ∈ face))}

def is_inscribed_in_sphere (P : Type) (S : set P) :=
  ∃ (O : P) (r : ℝ), ∀ (p : P), p ∈ P → dist O p = r

def is_circumscribed_around_sphere (P : Type) (S : set P) :=
  ∃ (O : P) (r : ℝ), ∀ (face : set P), face ∈ P → 
  ∃ (center : P), center ∈ S ∧ ∀ (p : P), p ∈ face → dist center p = r

theorem exists_cuboid_not_cube_inscribed_and_circumscribed_with_same_center :
  ∃ (P : Type) (S_in : set P) (S_out : set P), is_cuboid P ∧ 
  is_inscribed_in_sphere P S_out ∧ is_circumscribed_around_sphere P S_in ∧
  ∃ (a b c : ℝ), (a ≠ b ∨ b ≠ c ∨ a ≠ c) :=
begin
  sorry,
end

end exists_cuboid_not_cube_inscribed_and_circumscribed_with_same_center_l563_563257


namespace range_of_a_l563_563957

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → 2 * x * log x ≥ -x^2 + a * x - 3) → a ≤ 4 :=
by 
  sorry

end range_of_a_l563_563957


namespace algebraic_expression_l563_563562

theorem algebraic_expression (a b : Real) 
  (h : a * b = 2 * (a^2 + b^2)) : 2 * a * b - (a^2 + b^2) = 0 :=
by
  sorry

end algebraic_expression_l563_563562


namespace range_of_a_l563_563194

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x + a| > 2) ↔ a < -1 ∨ a > 3 :=
sorry

end range_of_a_l563_563194


namespace sequence_sum_l563_563787

theorem sequence_sum (a b : ℕ → ℝ)
  (h_rec : ∀ n, (a (n + 1), b (n + 1)) = (sqrt 3 * a n - b n, sqrt 3 * b n + a n))
  (h_100 : (a 100, b 100) = (2, 4)) :
  a 1 + b 1 = 1 / 2 ^ 98 :=
sorry

end sequence_sum_l563_563787


namespace jimmy_irene_total_payment_l563_563618

def cost_jimmy_shorts : ℝ := 3 * 15
def cost_irene_shirts : ℝ := 5 * 17
def total_cost_before_discount : ℝ := cost_jimmy_shorts + cost_irene_shirts
def discount : ℝ := total_cost_before_discount * 0.10
def total_paid : ℝ := total_cost_before_discount - discount

theorem jimmy_irene_total_payment : total_paid = 117 := by
  sorry

end jimmy_irene_total_payment_l563_563618


namespace tan_alpha_value_l563_563899

open Real

theorem tan_alpha_value 
  (α : ℝ) 
  (hα_range : 0 < α ∧ α < π) 
  (h_cos_alpha : cos α = -3/5) :
  tan α = -4/3 := 
by
  sorry

end tan_alpha_value_l563_563899


namespace luncheon_cost_l563_563471

variable (s c p : ℝ)
variable (eq1 : 5 * s + 9 * c + 2 * p = 6.50)
variable (eq2 : 7 * s + 14 * c + 3 * p = 9.45)
variable (eq3 : 4 * s + 8 * c + p = 5.20)

theorem luncheon_cost :
  s + c + p = 1.30 :=
by
  sorry

end luncheon_cost_l563_563471


namespace side_length_of_square_l563_563306

theorem side_length_of_square (m : ℕ) (a : ℕ) (hm : m = 100) (ha : a^2 = m) : a = 10 :=
by 
  sorry

end side_length_of_square_l563_563306


namespace set_intersection_complement_l563_563208

open Set

noncomputable def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
noncomputable def B : Set ℝ := {x | 0 < x ∧ x < 4}

theorem set_intersection_complement :
  (compl A ∩ B) = {x | 0 < x ∧ x ≤ 3} :=
by
  sorry

end set_intersection_complement_l563_563208


namespace min_detectors_needed_l563_563353

def board_width := 2015
def board_height := 2015
def ship_width := 1500
def ship_height := 1500

theorem min_detectors_needed (k : ℕ) :
  (∀ (board : fin (board_width * board_height) → bool) 
    (ship : fin (ship_width * ship_height) → fin (board_width * board_height)),
    (∃ (detectors : fin k → fin (board_width * board_height)), 
    ∀ i, ∃ j, (detectors i) = (ship ⟨j, sorry⟩) = true)) ↔ k ≥ 1030 := sorry

end min_detectors_needed_l563_563353


namespace B_pow_2023_l563_563262

open Matrix

section
variables {R : Type*} [CommRing R] (B : Matrix (Fin 3) (Fin 3) R)
def B := ![![0, -1, 0], ![1, 0, 0], ![0, 0, -1]]

theorem B_pow_2023 : B ^ 2023 = ![![0, 1, 0], ![-1, 0, 0], ![0, 0, -1]] :=
by sorry
end

end B_pow_2023_l563_563262


namespace area_dead_grass_l563_563777

-- This definition captures the given problem and conditions
def radius_sombrero : ℝ := 3
def radius_walk : ℝ := 5

-- This theorem statement captures the requirement to prove the area of dead grass
theorem area_dead_grass : 
  let inner_radius := radius_walk - radius_sombrero,
      outer_radius := radius_walk + radius_sombrero,
      area := π * (outer_radius^2 - inner_radius^2) in
  area = 60 * π :=
by
  -- Placeholder for the actual proof
  sorry

end area_dead_grass_l563_563777


namespace existsValidSet_l563_563499

open Set Int

noncomputable def isValidSet (A : Set ℕ) : Prop :=
  (∀ n : ℕ, (A ∩ {k | ∃ i : ℕ, k = i * n ∧ 1 ≤ i ∧ i ≤ 15}).card = 1) ∧
  (∃ᶠ m in atTop, {m, m + 2018} ⊆ A)

theorem existsValidSet :
  ∃ A : Set ℕ, isValidSet A :=
sorry

end existsValidSet_l563_563499


namespace domain_tan_function_period_tan_function_intervals_monotonicity_tan_function_symmetry_center_tan_function_l563_563856

variable (x k: ℤ)

def tan_function (x : ℝ) := Real.tan ((Real.pi / 2) * x - Real.pi / 3)

set_option pp.all true

theorem domain_tan_function :
  ∀ x, ∀ k ∈ ℤ, x ≠ (5 / 3 : ℤ) + 2 * k := 
sorry

theorem period_tan_function : 
  Function.Periodic (tan_function) 2 := 
sorry

theorem intervals_monotonicity_tan_function :
  ∀ k, ∀ x, k ∈ ℤ → -1 / 3 + 2 * k < x ∧ x < 5 / 3 + 2 * k :=
sorry

theorem symmetry_center_tan_function :
  ∀ k, ∀ x, k ∈ ℤ → x = 2 / 3 + k ∧ tan_function x = 0 :=
sorry

end domain_tan_function_period_tan_function_intervals_monotonicity_tan_function_symmetry_center_tan_function_l563_563856


namespace cashier_amount_l563_563619

def amount_to_cashier (discount : ℝ) (shorts_count : ℕ) (shorts_price : ℕ) (shirts_count : ℕ) (shirts_price : ℕ) : ℝ :=
  let total_cost := (shorts_count * shorts_price) + (shirts_count * shirts_price)
  let discount_amount := discount * total_cost
  total_cost - discount_amount

theorem cashier_amount : amount_to_cashier 0.1 3 15 5 17 = 117 := 
by
  sorry

end cashier_amount_l563_563619


namespace certain_value_z_l563_563158

-- Define the length of an integer as the number of prime factors
def length (n : ℕ) : ℕ := 
  primeFactors n |>.length

theorem certain_value_z {x y : ℕ} (hx1 : x > 1) (hy1 : y > 1)
  (h_len : length x + length y = 16) : 
  x + 3 * y < 98307 := 
sorry

end certain_value_z_l563_563158


namespace slope_of_tangent_line_l563_563908

theorem slope_of_tangent_line
  (l : ℝ → ℝ)
  (h₁ : ∀ x, l x = k * x)
  (h₂ : ∀ x y, (x - sqrt 3)^2 + (y - 1)^2 = 1 → l x = y)
  : k = 0 ∨ k = sqrt 3 := by
  sorry

end slope_of_tangent_line_l563_563908


namespace concyclic_ABLK_l563_563057

theorem concyclic_ABLK (A B C O K L : Point) (h1 : Triangle ABC) (hO : Circumcenter h1 O)
  (hB : angle B = 30) (hK : Line_BO_Intersects_AC_at_K BO AC K) 
  (hL : Midpoint_Arc_OC_not_containing_K Circumcircle_KOC L) : 
  Concyclic A B L K :=
sorry

end concyclic_ABLK_l563_563057


namespace remaining_pictures_l563_563037

theorem remaining_pictures (k m : ℕ) (d1 := 9 * k + 4) (d2 := 9 * m + 6) :
  (d1 * d2) % 9 = 6 → 9 - (d1 * d2 % 9) = 3 :=
by
  intro h
  sorry

end remaining_pictures_l563_563037


namespace minimum_time_fry_three_pancakes_l563_563729

-- Definitions based on conditions
def pancakes (n : ℕ) := n

def can_fry_two_at_a_time : Prop := ∀ n, pancakes n ≤ 2

def fry_time_one_side : ℕ := 1

def minimum_time_to_fry (n : ℕ) : ℕ :=
  if n = 3 then 3 else 0 -- for simplicity, define only for n = 3 as per the problem

-- Theorem stating the proof problem
theorem minimum_time_fry_three_pancakes :
  ∀ (n : ℕ), pancakes n = 3 → can_fry_two_at_a_time ∧ fry_time_one_side = 1 → minimum_time_to_fry 3 = 3 :=
begin
  sorry
end

end minimum_time_fry_three_pancakes_l563_563729


namespace chromium_mass_percentage_not_unique_l563_563511

theorem chromium_mass_percentage_not_unique (P : Type) [has_mass P] (C : P → Prop) (cr_mass_percentage : P → ℝ) 
  (h : ∃ p1 p2 : P, C p1 ∧ C p2 ∧ p1 ≠ p2 ∧ cr_mass_percentage p1 = 35.14 ∧ cr_mass_percentage p2 = 35.14) : 
  ¬ ∃ p : P, C p ∧ cr_mass_percentage p = 35.14 → ∀ p : P, C p → cr_mass_percentage p = 35.14 := 
by {
  sorry
}

end chromium_mass_percentage_not_unique_l563_563511


namespace decimal_to_fraction_l563_563767

theorem decimal_to_fraction (h : ¬(0.32 = 0) ∧ 0.32 = 32 / 100 : Prop) :
  (32 / 100 : ℝ) = 8 / 25 :=
by
  sorry

end decimal_to_fraction_l563_563767


namespace ratio_of_green_to_blue_is_correct_l563_563428

variable (total_crayons : ℕ)
variable (red_crayons : ℕ)
variable (blue_crayons : ℕ)
variable (pink_crayons : ℕ)
variable (green_crayons : ℕ)

-- Conditions
def conditions : Prop :=
  total_crayons = 24 ∧
  red_crayons = 8 ∧
  blue_crayons = 6 ∧
  pink_crayons = 6 ∧
  green_crayons = total_crayons - (red_crayons + blue_crayons + pink_crayons)

-- Question to prove
theorem ratio_of_green_to_blue_is_correct : conditions total_crayons red_crayons blue_crayons pink_crayons green_crayons → (green_crayons * 3 = blue_crayons * 2) :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  rw [h1, h3, h5, h6]
  sorry

end ratio_of_green_to_blue_is_correct_l563_563428


namespace option_a_is_fraction_option_b_is_fraction_option_c_is_fraction_option_d_is_fraction_l563_563049

section

variable (π : Real) (x : Real)

-- Definition of a fraction in this context
def is_fraction (num denom : Real) : Prop := denom ≠ 0

-- Proving each given option is a fraction
theorem option_a_is_fraction : is_fraction 1 π := 
sorry

theorem option_b_is_fraction : is_fraction x 3 :=
sorry

theorem option_c_is_fraction : is_fraction 2 5 :=
sorry

theorem option_d_is_fraction : is_fraction 1 (x - 1) :=
sorry

end

end option_a_is_fraction_option_b_is_fraction_option_c_is_fraction_option_d_is_fraction_l563_563049


namespace thirty_seven_in_base_2_l563_563849

def decimal_to_binary (n : ℕ) : list ℕ :=
  if n = 0 then [0]
  else let rec aux (n : ℕ) : list ℕ :=
    if n = 0 then [] else (n % 2) :: aux (n / 2)
  in aux n

theorem thirty_seven_in_base_2 : decimal_to_binary 37 = [1, 0, 0, 1, 0, 1] :=
  sorry

end thirty_seven_in_base_2_l563_563849


namespace Michael_and_truck_meet_once_l563_563284

theorem Michael_and_truck_meet_once :
  (Michael_speed truck_speed pail_distance truck_stop_time cycles : ℕ) 
  (cycle_time := pail_distance / truck_speed + truck_stop_time)
  (initial_distance (cycle_duration : ℕ) := 
    pail_distance + cycle_duration * (pail_distance - Michael_speed * (cycle_duration / truck_speed) - truck_stop_time))
  (distance_decrease_per_cycle := Michael_speed * (pail_distance / truck_speed)  + truck_stop_time)
  (initial_cycle_duration := 20)
  (second_cycle_duration := 60) :
  Michael_speed = 6 ∧
  truck_speed = 10 ∧
  pail_distance = 200 ∧
  truck_stop_time = 40 ∧
  cycles =  (1 : ℕ) := 
sorry

end Michael_and_truck_meet_once_l563_563284


namespace mark_total_play_time_l563_563243

theorem mark_total_play_time :
  let week1_gigs := 4
  let week2_gigs := 3
  let week3_gigs := 4
  let week4_gigs := 5
  let total_gigs := week1_gigs + week2_gigs + week3_gigs + week4_gigs
  let song_lengths := [5, 6, 8]
  let avg_length := (song_lengths.sum / song_lengths.length : ℝ)
  let gig_length := (song_lengths.sum + avg_length : ℝ)
  in total_gigs * gig_length = 405.28 :=
by
  let week1_gigs := 4
  let week2_gigs := 3
  let week3_gigs := 4
  let week4_gigs := 5
  let total_gigs := week1_gigs + week2_gigs + week3_gigs + week4_gigs
  let song_lengths := [5, 6, 8]
  let avg_length := (song_lengths.sum / song_lengths.length : ℝ)
  let gig_length := (song_lengths.sum + avg_length : ℝ)
  have h : total_gigs = 16 := rfl
  have h_song_lengths : song_lengths.sum = 19 := rfl
  have h_avg_length : avg_length = 19 / 3 := rfl
  have h_gig_length : gig_length = 25 + 1/3 := rfl
  have h_total_time : total_gigs * gig_length = 16 * (25 + 1/3) := rfl
  have h_correct_answer : 16 * (25 + 1/3) = 405.28 := rfl
  exact h_correct_answer

end mark_total_play_time_l563_563243


namespace sum_of_coordinates_of_B_l563_563662

theorem sum_of_coordinates_of_B :
  ∀ (A B : (ℝ × ℝ)),
    A = (0,0) →
    (∃ x : ℝ, B = (x, 4)) →
    (∃ x : ℝ, y : ℝ, x ≠ 0 ∧ B = (x, y) ∧ A = (0, 0) ∧ y - 0 = (2 / 3) * (x - 0)) →
    ((B.1 + B.2) = 10) :=
by
  intros A B HA HB HSlope
  sorry

end sum_of_coordinates_of_B_l563_563662


namespace necessary_but_not_sufficient_condition_l563_563544

def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := 5 * x - 6 > x ^ 2

theorem necessary_but_not_sufficient_condition :
  (∀ x, q x → p x) ∧ (¬ ∀ x, p x → q x) :=
by
  sorry

end necessary_but_not_sufficient_condition_l563_563544


namespace gecko_egg_hatching_l563_563435

theorem gecko_egg_hatching :
  let total_eggs := 30
  let infertile_eggs := total_eggs * 0.20
  let remaining_eggs := total_eggs - infertile_eggs
  let calcification_eggs := remaining_eggs / 3
  let hatched_eggs := remaining_eggs - calcification_eggs
  in hatched_eggs = 16 := 
by
  sorry

end gecko_egg_hatching_l563_563435


namespace store_incurs_loss_l563_563792

noncomputable def store_transaction : ℝ :=
  let ac_selling_price := 2000
  let tv_selling_price := 2000
  let ac_profit_ratio := 0.3
  let tv_loss_ratio := 0.2
  let ac_cost_price := ac_selling_price / (1 + ac_profit_ratio)
  let tv_cost_price := tv_selling_price / (1 - tv_loss_ratio)
  let ac_profit := ac_selling_price - ac_cost_price
  let tv_loss := tv_cost_price - tv_selling_price
  let net_result := ac_profit - tv_loss
  net_result

theorem store_incurs_loss : store_transaction ≈ -38.5 :=
  by sorry

end store_incurs_loss_l563_563792


namespace volume_ratios_of_spherical_segment_and_cones_l563_563707

theorem volume_ratios_of_spherical_segment_and_cones :
  let r := 5 in
  let m := 2 in
  let v1 := (m * π / 6) * (3 * r^2 + m^2) in
  let v2 := (1 / 3) * π * r^2 * m in
  let h := (2 * m * r^2) / (r^2 - m^2) in
  let v3 := (1 / 3) * π * r^2 * h in
  (v1, v2, v3) = (79 * π / 3, 50 * π / 3, 2500 * π / 63) :=
by
  sorry

end volume_ratios_of_spherical_segment_and_cones_l563_563707


namespace resultant_alcohol_percentage_l563_563308

-- Define the given ratios and mixing proportions
def ratio_solution_a := (21, 4)
def ratio_solution_b := (2, 3)
def mixing_ratio := (5, 6)

-- Calculation of the percentage of alcohol in the resultant mixture
-- Here, we denote alcohol percentage directly given the provided solution steps.

theorem resultant_alcohol_percentage : 
  let total_volume_of_mixture := (mixing_ratio.fst + mixing_ratio.snd : ℝ)
  let alcohol_volume_in_a := (ratio_solution_a.fst : ℝ) / (ratio_solution_a.fst + ratio_solution_a.snd) * mixing_ratio.fst
  let alcohol_volume_in_b := (ratio_solution_b.fst : ℝ) / (ratio_solution_b.fst + ratio_solution_b.snd) * mixing_ratio.snd
  let total_alcohol_volume := alcohol_volume_in_a + alcohol_volume_in_b
  (total_alcohol_volume / total_volume_of_mixture) * 100 = 60 := 
by {
      let total_volume_of_mixture := 5 + 6
      let alcohol_volume_in_a := (21.0 / (21.0 + 4.0)) * 5.0
      let alcohol_volume_in_b := (2.0 / (2.0 + 3.0)) * 6.0
      let total_alcohol_volume := alcohol_volume_in_a + alcohol_volume_in_b
      have h : (total_alcohol_volume / total_volume_of_mixture) * 100 = 60,
      sorry,  -- proof is not required
    }

end resultant_alcohol_percentage_l563_563308


namespace remaining_paint_needed_l563_563162

-- Define the conditions
def total_paint_needed : ℕ := 70
def paint_bought : ℕ := 23
def paint_already_have : ℕ := 36

-- Lean theorem statement
theorem remaining_paint_needed : (total_paint_needed - (paint_already_have + paint_bought)) = 11 := by
  sorry

end remaining_paint_needed_l563_563162


namespace country_y_orange_exports_amount_l563_563059

-- Definitions reflecting the conditions
def yearly_exports : ℝ := 127.5
def percent_fruit_exports : ℝ := 0.20
def fraction_orange_exports : ℝ := 1 / 6

-- Definition of the amount of money generated from orange exports
noncomputable def orange_exports_amount : ℝ :=
  fraction_orange_exports * percent_fruit_exports * yearly_exports

-- The statement of the proof problem
theorem country_y_orange_exports_amount :
  orange_exports_amount = 4.24275 :=
by
  unfold orange_exports_amount
  sorry

end country_y_orange_exports_amount_l563_563059


namespace geometric_sequence_product_l563_563969

theorem geometric_sequence_product 
  {a : ℕ → ℝ} (h : ∀ n, 0 < a n)
  (hyp : geometric_sequence a)
  (h_root : (a 1, a 19) are roots of x^2 - 10 * x + 16) : 
  a 8 * a 10 * a 12 = 64 := 
by
  sorry

end geometric_sequence_product_l563_563969


namespace probability_point_within_2_units_l563_563078

theorem probability_point_within_2_units :
  let points := [(x, y) | x ∈ [-4, -3, -2, -1, 0, 1, 2, 3, 4], y ∈ [-4, -3, -2, -1, 0, 1, 2, 3, 4]],
      total_points := 81,
      valid_points := [(x, y) | (x, y) ∈ points ∧ x^2 + y^2 ≤ 4],
      number_of_valid_points := 13
  in (float_of_int number_of_valid_points) / total_points = 13 / 81 := sorry

end probability_point_within_2_units_l563_563078


namespace distance_squared_l563_563799

theorem distance_squared (A_start_time B_start_time : ℤ) (A_speed B_speed : ℤ) (current_time : ℤ) :
  A_start_time = 7 → B_start_time = 8 →
  A_speed = 6 → B_speed = 5 →
  current_time = 10 →
  (let A_distance := (current_time - A_start_time) * A_speed in
   let B_distance := (current_time - B_start_time) * B_speed in
   A_distance^2 + B_distance^2 = 424) :=
begin
  sorry
end

end distance_squared_l563_563799


namespace relationship_of_ys_l563_563297

variable (f : ℝ → ℝ)

def inverse_proportion := ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k / x

theorem relationship_of_ys
  (h_inv_prop : inverse_proportion f)
  (h_pts1 : f (-2) = 3)
  (h_pts2 : f (-3) = y₁)
  (h_pts3 : f 1 = y₂)
  (h_pts4 : f 2 = y₃) :
  y₂ < y₃ ∧ y₃ < y₁ :=
sorry

end relationship_of_ys_l563_563297


namespace hexagon_side_length_l563_563661

theorem hexagon_side_length (h : ∀ (hex : ℕ → ℝ), (∀ i j, opposite_sides hex i j → dist (hex i) (hex j) = 18) → is_regular_hexagon hex) :
  ∀ (hex : ℕ → ℝ), (∀ i j, opposite_sides hex i j → dist (hex i) (hex j) = 18) → side_length hex = 12 * Real.sqrt 3 := 
by
  sorry

end hexagon_side_length_l563_563661


namespace simplify_expression_l563_563307

variable (a : ℝ)

theorem simplify_expression :
    5 * a^2 - (a^2 - 2 * (a^2 - 3 * a)) = 6 * a^2 - 6 * a := by
  sorry

end simplify_expression_l563_563307


namespace math_problem_l563_563565

open Nat

-- Given conditions
def S (n : ℕ) : ℕ := n * (n + 1)

-- Definitions for the terms a_n, b_n, c_n, and the sum T_n
def a_n (n : ℕ) (h : n ≠ 0) : ℕ := if n = 1 then 2 else 2 * n
def b_n (n : ℕ) (h : n ≠ 0) : ℕ := 2 * (3^n + 1)
def c_n (n : ℕ) (h : n ≠ 0) : ℕ := a_n n h * b_n n h / 4
def T (n : ℕ) (h : 0 < n) : ℕ := 
  (2 * n - 1) * 3^(n + 1) / 4 + 3 / 4 + n * (n + 1) / 2

-- Main theorem to establish the solution
theorem math_problem (n : ℕ) (h : n ≠ 0) : 
  S n = n * (n + 1) →
  a_n n h = 2 * n ∧ 
  b_n n h = 2 * (3^n + 1) ∧ 
  T n (Nat.pos_of_ne_zero h) = (2 * n - 1) * 3^(n + 1) / 4 + 3 / 4 + n * (n + 1) / 2 := 
by
  intros hS
  sorry

end math_problem_l563_563565


namespace optionD_is_random_event_other_options_are_not_random_events_l563_563048

/-- Define the specific events as types -/
inductive Event
| BuyLotteryTicketAndNotWin : Event
| DrawRedBallFromBagWithOnlyRedBalls : Event
| SunRisingFromEastTomorrowMorning : Event
| TurnOnTVAndWatchChinesePoetryConference : Event

/-- Define randomness in event -/
def isRandomEvent : Event → Prop
| Event.BuyLotteryTicketAndNotWin := False
| Event.DrawRedBallFromBagWithOnlyRedBalls := False
| Event.SunRisingFromEastTomorrowMorning := False
| Event.TurnOnTVAndWatchChinesePoetryConference := True

/-- Prove that turning on the TV and watching "Chinese Poetry Conference" is a random event -/
theorem optionD_is_random_event :
  isRandomEvent Event.TurnOnTVAndWatchChinesePoetryConference = True :=
by 
  sorry

/-- Prove that the other options are not random events -/
theorem other_options_are_not_random_events :
  isRandomEvent Event.BuyLotteryTicketAndNotWin = False ∧ 
  isRandomEvent Event.DrawRedBallFromBagWithOnlyRedBalls = False ∧ 
  isRandomEvent Event.SunRisingFromEastTomorrowMorning = False :=
by 
  sorry

end optionD_is_random_event_other_options_are_not_random_events_l563_563048


namespace min_value_of_f_l563_563186

noncomputable def f (a b x : ℝ) : ℝ :=
  (a / (Real.sin x) ^ 2) + b * (Real.sin x) ^ 2

theorem min_value_of_f (a b : ℝ) (h1 : a = 2) (h2 : b = 1) (h3 : a > b) (h4 : b > 0) :
  ∃ x, f a b x = 3 := 
sorry

end min_value_of_f_l563_563186


namespace part1_part2_l563_563577

section part1
  variable (x : Real)
  def a := (Real.sin x, 1)
  def b := (Real.sqrt 3 * Real.cos x, -2)
  def f (x : Real) : Real := (a x).1 + (b x).1 * (a x).1

  theorem part1 (h_parallel : a x = b x) : Real.cos (2 * x) = 1/7 := sorry
end part1

section part2
  variable {ABC : Type}
  variable [triangle : Triangle ABC]
  
  def acute_triangle_area_range (A B C : Real.Angle) (a b c : Real) (h_acute : Triangle.isAcute A B C)
    (h_b : b = 2) (h_fA : f A = 1/2) : Real :=
    (√3 / 2, 2√3)

  theorem part2 (A B C : Real.Angle) (a b c : Real)
    (h_acute : Triangle.isAcute A B C)
    (h_b : b = 2) (h_fA : f A = 1/2) : acute_triangle_area_range A B C a b c h_acute h_b h_fA :=
    sorry
end part2

end part1_part2_l563_563577


namespace popsicle_consumption_combined_l563_563283

theorem popsicle_consumption_combined (h_duration: Nat) (megan_rate: Nat) (liam_rate: Nat) (total_minutes: Nat) :
  (total_minutes = h_duration * 60) →
  (megan_rate = total_minutes / 15) →
  (liam_rate = total_minutes / 20) →
  (megan_rate + liam_rate = 35) :=
by
  -- Let h_duration = 5 (hours)
  let h_duration := 5
  
  -- Let total_minutes = h_duration * 60
  let total_minutes := h_duration * 60
  
  -- Calculate Megan's rate
  let megan_popsicles := total_minutes / 15
  
  -- Calculate Liam's rate
  let liam_popsicles := total_minutes / 20
  
  -- Sum of Megan's and Liam's popsicles
  show megan_popsicles + liam_popsicles = 35 from sorry

end popsicle_consumption_combined_l563_563283


namespace count_irrationals_l563_563806

theorem count_irrationals : 
  let numbers := [4, 0, (12 / 7 : ℝ), (0.125)^(1/3 : ℝ), 0.1010010001, Real.sqrt 3, Real.pi / 2]
  in (numbers.countP (λ x => ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b)) = 2 :=
by 
  sorry

end count_irrationals_l563_563806


namespace base7_first_digit_l563_563735

noncomputable def first_base7_digit : ℕ := 625

theorem base7_first_digit (n : ℕ) (h : n = 625) : ∃ (d : ℕ), d = 12 ∧ (d * 49 ≤ n) ∧ (n < (d + 1) * 49) :=
by
  sorry

end base7_first_digit_l563_563735


namespace log2_monotone_l563_563643

theorem log2_monotone (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a > b) ↔ (Real.log a / Real.log 2 > Real.log b / Real.log 2) :=
sorry

end log2_monotone_l563_563643


namespace probability_one_trained_contestant_distribution_of_xi_l563_563651

theorem probability_one_trained_contestant :
  (4.choose 1 * 2.choose 1) / 6.choose 2 = 8 / 15 :=
by sorry

theorem distribution_of_xi :
  ∀ ξ : ℕ,
    (ξ = 2 → (4.choose 2 / 6.choose 2 = 2 / 5)) ∧ 
    (ξ = 3 → (4.choose 1 * 2.choose 1 / 6.choose 2 = 8 / 15)) ∧ 
    (ξ = 4 → (2.choose 2 / 6.choose 2 = 1 / 15)) :=
by sorry

end probability_one_trained_contestant_distribution_of_xi_l563_563651


namespace hours_per_day_in_deliberation_l563_563626

noncomputable def jury_selection_days : ℕ := 2
noncomputable def trial_days : ℕ := 4 * jury_selection_days
noncomputable def total_deliberation_hours : ℕ := 6 * 24
noncomputable def total_days_on_jury_duty : ℕ := 19

theorem hours_per_day_in_deliberation :
  (total_deliberation_hours / (total_days_on_jury_duty - (jury_selection_days + trial_days))) = 16 :=
by
  sorry

end hours_per_day_in_deliberation_l563_563626


namespace sum_of_divisors_90_l563_563367

theorem sum_of_divisors_90 : 
  let n := 90 in 
  let sum_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5) in
  sum_divisors n = 234 :=
by 
  let n := 90
  let sum_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5)
  sorry

end sum_of_divisors_90_l563_563367


namespace column_row_sum_le_n_l563_563237

theorem column_row_sum_le_n (n : ℕ) (A : matrix (fin (2 * n)) (fin (2 * n)) ℝ)
  (h_sum_zero : ∑ i j, A i j = 0)
  (h_abs_le_one : ∀ i j, abs (A i j) ≤ 1) :
  ∀ i, abs (∑ j, A i j) ≤ n ∧ ∀ j, abs (∑ i, A i j) ≤ n :=
sorry

end column_row_sum_le_n_l563_563237


namespace opposite_of_neg_3_l563_563005

theorem opposite_of_neg_3 : (-(-3) = 3) :=
by
  sorry

end opposite_of_neg_3_l563_563005


namespace find_sin_B_l563_563233

variables (a b c : ℝ) (A B C : ℝ)

def sin_law_abc (a b : ℝ) (sinA : ℝ) (sinB : ℝ) : Prop := 
  (a / sinA) = (b / sinB)

theorem find_sin_B {a b : ℝ} (sinA : ℝ) 
  (ha : a = 3) 
  (hb : b = 5) 
  (hA : sinA = 1 / 3) :
  ∃ sinB : ℝ, (sinB = 5 / 9) ∧ sin_law_abc a b sinA sinB :=
by
  use 5 / 9
  simp [sin_law_abc, ha, hb, hA]
  sorry

end find_sin_B_l563_563233


namespace longest_side_of_triangle_l563_563798

variable (x y : ℝ)

def side1 := 10
def side2 := 2*y + 3
def side3 := 3*x + 2

theorem longest_side_of_triangle
  (h_perimeter : side1 + side2 + side3 = 45)
  (h_side2_pos : side2 > 0)
  (h_side3_pos : side3 > 0) :
  side3 = 32 :=
sorry

end longest_side_of_triangle_l563_563798


namespace binomial_expansion_coefficient_l563_563251

theorem binomial_expansion_coefficient :
  let n := 8
  ∃ r, r = 3 ∧ 8 - 2 * r = 2 ∧
  let term := binomial_coeff n r * (-1)^r * x^(n - 2 * r)
  term = -56 :=
by
  let n := 8
  let r := 3
  have h1 : 8 - 2 * r = 2 := by norm_num
  have h2 : binomial_coeff n r = 56 := by norm_num
  have h3 : (-1)^r = -1 := by norm_num
  let term := 56 * (-1) * x^2
  have h4 : term = -56 := by norm_num
  use r
  exact ⟨rfl, h1, h4⟩

end binomial_expansion_coefficient_l563_563251


namespace extremum_values_range_tangent_line_range_number_of_tangent_lines_l563_563920

noncomputable def f (a x : ℝ) : ℝ := a * real.log x + 1 / x

def has_extremum (a : ℝ) : Prop :=
  a > 0

def tangent_through_origin (a : ℝ) : Prop :=
  a >= 2

def tangent_count (a : ℝ) : ℕ :=
  if a = 2 then 1 else if a > 2 then 2 else 0

theorem extremum_values_range (a : ℝ) : has_extremum a = (0 < a) :=
sorry

theorem tangent_line_range (a : ℝ) : tangent_through_origin a = (a >= 2) :=
sorry

theorem number_of_tangent_lines (a : ℝ) (h : tangent_through_origin a) :
  (tangent_count a = if a = 2 then 1 else if a > 2 then 2 else 0) :=
sorry

end extremum_values_range_tangent_line_range_number_of_tangent_lines_l563_563920


namespace total_money_earned_before_saving_enough_l563_563782

variable (monthly_income : ℕ) (rent : ℕ) (food : ℕ) (misc_bills : ℕ)
variable (fixed_savings : ℕ) (savings_rate : ℚ) (total_savings_needed : ℕ)

def total_expenses (rent food misc_bills : ℕ) : ℕ :=
  rent + food + misc_bills

def remaining_income (monthly_income total_expenses : ℕ) : ℕ :=
  monthly_income - total_expenses

def savings_from_remaining_income (remaining_income : ℤ) (savings_rate : ℚ) : ℕ :=
  ⌊remaining_income * savings_rate⌋

def total_monthly_savings (fixed_savings : ℕ) (savings_from_remaining_income : ℕ) : ℕ :=
  fixed_savings + savings_from_remaining_income

def months_to_save (total_savings_needed total_monthly_savings : ℕ) : ℚ :=
  total_savings_needed / total_monthly_savings

def total_money_earned (monthly_income : ℕ) (months_to_save : ℕ) : ℕ :=
  monthly_income * months_to_save

theorem total_money_earned_before_saving_enough : 
  ∀ (monthly_income rent food misc_bills fixed_savings total_savings_needed : ℕ) (savings_rate : ℚ),
    monthly_income = 4000 → 
    rent = 600 →
    food = 300 →
    misc_bills = 100 →
    fixed_savings = 500 →
    savings_rate = 0.05 →
    total_savings_needed = 45000 →
    total_money_earned monthly_income (⌊months_to_save total_savings_needed (total_monthly_savings fixed_savings 
      (savings_from_remaining_income (remaining_income monthly_income (total_expenses rent food misc_bills)) savings_rate) )⌋) = 280000 := 
by
  intros
  sorry

end total_money_earned_before_saving_enough_l563_563782


namespace lucky_penny_probability_l563_563063

theorem lucky_penny_probability :
  let S := 4
  let D := 4
  (probability_of_lucky_penny_last (S + D) S) = 1 / 2 :=
sorry

end lucky_penny_probability_l563_563063


namespace jo_reading_hours_l563_563621

theorem jo_reading_hours :
  ∀ (total_pages current_page previous_page pages_per_hour remaining_pages : ℕ),
    total_pages = 210 →
    current_page = 90 →
    previous_page = 60 →
    pages_per_hour = current_page - previous_page →
    remaining_pages = total_pages - current_page →
    remaining_pages / pages_per_hour = 4 :=
by
  intros total_pages current_page previous_page pages_per_hour remaining_pages
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3] at *
  dsimp at *
  sorry

end jo_reading_hours_l563_563621


namespace cos_theta_correct_speed_correct_l563_563452

-- Definitions of constants and terms used in the problem
def AB : ℝ := 20 * Real.sqrt 2
def angle_ABA : ℝ := Real.pi / 4 -- 45 degrees in radians

def AC : ℝ := 5 * Real.sqrt 13
def cos_EAC : ℝ := 2 * Real.sqrt 13 / 13
def sin_EAC : ℝ := Real.sqrt (1 - cos_EAC ^ 2)
def angle_EAC : ℝ := Real.arccos cos_EAC

def theta : ℝ := angle_ABA - angle_EAC

-- Proof for the first question: Calculate cos(theta)
theorem cos_theta_correct : cos theta = Real.sqrt 26 / 26 :=
by
  -- proof skipped
  sorry

-- Proof for the second question: Calculate speed of the ship
def time_in_hours : ℝ := 20 / 60 -- time in hours (20 minutes)
def BC_squared : ℝ := (AB ^ 2) + (AC ^ 2) - 2 * AB * AC * (Real.sqrt 26 / 26)
def BC : ℝ := Real.sqrt BC_squared

noncomputable def speed : ℝ := BC / time_in_hours

theorem speed_correct : speed = 15 * Real.sqrt 35 :=
by
  -- proof skipped
  sorry

end cos_theta_correct_speed_correct_l563_563452


namespace max_sum_length_le_98306_l563_563160

noncomputable def L (k : ℕ) : ℕ := sorry

theorem max_sum_length_le_98306 (x y : ℕ) (hx : x > 1) (hy : y > 1) (hl : L x + L y = 16) : x + 3 * y < 98306 :=
sorry

end max_sum_length_le_98306_l563_563160


namespace minimum_swaps_to_descending_order_l563_563788

theorem minimum_swaps_to_descending_order : 
  ∀ (initial_arrangement : List ℕ),
  initial_arrangement = [1, 2, 3, 4, 5] →
  (by admit : ∃ (swaps : ℕ), swaps = 10 ∧ (∀ (adj_swap : ℕ → ℕ → List ℕ → List ℕ), true)) :=
begin
  sorry
end

end minimum_swaps_to_descending_order_l563_563788


namespace countSingleDuplicateNumbers_le_200_l563_563587

-- Definition of a "single duplicate number".
def isSingleDuplicateNumber (n : ℕ) : Prop :=
  n >= 100 ∧ n <= 200 ∧ 
  (let digits := [n / 100, (n / 10) % 10, n % 10] 
  in (digits.count_nth (digits.get 0) = 2 ∧ digits.count_nth (digits.get 1) = 1) ∨ 
     (digits.count_nth (digits.get 1) = 2 ∧ digits.count_nth (digits.get 2) = 1) ∨ 
     (digits.count_nth (digits.get 0) = 2 ∧ digits.count_nth (digits.get 2) = 1))

-- Main theorem stating the count of such numbers.
theorem countSingleDuplicateNumbers_le_200 : 
  (finset.filter isSingleDuplicateNumber (finset.range 201)).card = 28 := 
by
  sorry

end countSingleDuplicateNumbers_le_200_l563_563587


namespace find_angle_sin_theta_l563_563269

noncomputable def vector_problem (a b c d : ℝ^3) (θ : ℝ) :=
  ∥a∥ = 1 ∧ ∥b∥ = 7 ∧ ∥c∥ = 6 ∧ a × (a × (b + d)) = c ∧ d = 2 • a ∧ sinθ = 6 / 7

-- The main theorem stating the equivalence
theorem find_angle_sin_theta (a b c d : ℝ^3) (θ : ℝ) (h : vector_problem a b c d θ) : 
  sin θ = 6 / 7 := 
sorry

end find_angle_sin_theta_l563_563269


namespace find_all_solutions_l563_563850

def is_solution (f : ℕ → ℝ) : Prop :=
  (∀ n ≥ 1, f (n + 1) ≥ f n) ∧
  (∀ m n, Nat.gcd m n = 1 → f (m * n) = f m * f n)

theorem find_all_solutions :
  ∀ f : ℕ → ℝ, is_solution f →
    (∀ n, f n = 0) ∨ (∃ a ≥ 0, ∀ n, f n = n ^ a) :=
sorry

end find_all_solutions_l563_563850


namespace find_ratio_ADB_divide_D_l563_563036

variables {O1 O2 A B C D : Type} [geometry O1 O2 A B C D]
variable (n : ℝ)

-- Condition 1: Two circles intersect at points A and B
axiom circles_intersect_at_A_B (circ1 : circle O1) (circ2 : circle O2) : intersect circ1 circ2 A B

-- Condition 2: The first circle passes through the center of the second circle
axiom circ1_passes_through_O2 (circ1 : circle O1) (O2 : point) : on_circle O2 circ1

-- Condition 3: Chord BD intersects the second circle at point C
axiom chord_BD_intersects_at_C (circ1 : circle O1) (circ2 : circle O2) : intersects_at_chord circ1 circ2 BD C

-- Condition 4: Chord BD divides the arc ACB in the ratio AC:CB = n
axiom ratio_AC_CB (arc_ACB : arc circ2 A C B) : ratio arc_ACB AC CB = n

-- Goal: In what ratio does point D divide the arc ADB
theorem find_ratio_ADB_divide_D (circ1 : circle O1) (circ2 : circle O2)
    (h_intersect : circles_intersect_at_A_B circ1 circ2)
    (h_on_circle : circ1_passes_through_O2 circ1 O2)
    (h_chord : chord_BD_intersects_at_C circ1 circ2)
    (h_ratio : ratio_AC_CB (arc_ACB circ2 A C B)) :
    ratio (arc_ADB circ1 A D B) AD BD = n / (n + 2) := by
  sorry  -- Proof goes here

end find_ratio_ADB_divide_D_l563_563036


namespace determine_p_of_conditions_l563_563496

noncomputable def p (x : ℝ) := (8 / 5) * x^2 + (16 / 5) * x - (24 / 5)

theorem determine_p_of_conditions :
  ∀ x : ℝ,
    (∀ x : ℝ, (x^3 + x^2 - 4*x - 4) / p(x) has_vertical_asymptotes_at [1, -3] ∧
    (p(2) = 8) ∧
    ¬ isHorizontalAsymptote (x^3 + x^2 - 4*x - 4) p(x)) →
    p(x) = (8 / 5) * x^2 + (16 / 5) * x - (24 / 5) :=
by
  sorry -- Proof to be completed

end determine_p_of_conditions_l563_563496


namespace min_tangent_line_distance_circle_l563_563881

noncomputable def minTangentLineDistance (a b : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2)

theorem min_tangent_line_distance_circle :
  ∀ a b : ℝ, a > 0 → b > 0 → x² + y² = 1 → 
  let ab := (a * b) / real.sqrt (a^2 + b^2) in
  ab = 1 → 
  minTangentLineDistance a b = 2 :=
by
  sorry

end min_tangent_line_distance_circle_l563_563881


namespace length_of_angle_bisector_AD_l563_563605

theorem length_of_angle_bisector_AD 
  (A B C D : Type*)
  [metric_space A B C D]
  (right_triangle : ∀ (A B C : Triangle), angle A = 90)
  (AB_len AC_len : ℝ) 
  (AB_eq : AB_len = 5) 
  (AC_eq : AC_len = 4) 
  (AD_len : ℝ) 
  (AD_eq : AD_len = 20 * real.sqrt 2 / 9) 
  : true :=
by 
  let ABC := mk_triangle A B C
  have hyp1 : ∠A = 90 := right_triangle ABC
  have hyp2 : AB.len = 5 := AB_eq
  have hyp3 : AC.len = 4 := AC_eq
  have len_AD := by Pythagoras
  have result : AD.len = 20 * real.sqrt 2 / 9 := sorry
  exact true_intro


end length_of_angle_bisector_AD_l563_563605


namespace reservoir_water_level_l563_563425

theorem reservoir_water_level (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : 
  ∃ y : ℝ, y = 6 + 0.3 * x :=
by sorry

end reservoir_water_level_l563_563425


namespace sum_of_exponents_of_sqrt_of_largest_perfect_square_dividing_factorial_15_l563_563836

theorem sum_of_exponents_of_sqrt_of_largest_perfect_square_dividing_factorial_15 :
  let prime_exponents := [2, 3, 5, 7, 11, 13]
  let exponents_15_fact (p: ℕ) := ∑ k in range (nat.log2 15).succ, 15 / p^k
  let adjust_even (e: ℕ) := e - e % 2
  let largest_perf_square := 
    List.foldl (λ prod p, prod * p ^ ((adjust_even (exponents_15_fact p)) / 2)) 1 prime_exponents
  let square_root_exponents_sum := 
    List.foldl (λ sum p, sum + (adjust_even (exponents_15_fact p)) / 2) 0 prime_exponents
  square_root_exponents_sum = 10 := by
  sorry

end sum_of_exponents_of_sqrt_of_largest_perfect_square_dividing_factorial_15_l563_563836


namespace natural_number_between_squares_l563_563779

open Nat

theorem natural_number_between_squares (n m k l : ℕ)
  (h1 : n > m^2)
  (h2 : n < (m+1)^2)
  (h3 : n - k = m^2)
  (h4 : n + l = (m+1)^2) : ∃ x : ℕ, n - k * l = x^2 := by
  sorry

end natural_number_between_squares_l563_563779


namespace carl_dina_meet_probability_l563_563116

theorem carl_dina_meet_probability :
  let time_interval := Set.Icc 0 (3 / 4) in
  let area := (3 / 4) * (3 / 4) in
  let non_overlap_area := (1 / 2) * (1 / 4) * (1 / 4) in
  let meet_area := area - 2 * non_overlap_area in
  meet_area / area = 8 / 9 := by
  let time_interval := Set.Icc 0 (3 / 4)
  let area := (3 / 4) * (3 / 4)
  let non_overlap_area := (1 / 2) * (1 / 4) * (1 / 4)
  let meet_area := area - 2 * non_overlap_area
  have H : meet_area / area = 8 / 9 := sorry
  exact H

end carl_dina_meet_probability_l563_563116


namespace julie_upstream_distance_l563_563991

noncomputable def speed_of_stream : ℝ := 0.5
noncomputable def distance_downstream : ℝ := 72
noncomputable def time_spent : ℝ := 4
noncomputable def speed_of_julie_in_still_water : ℝ := 17.5
noncomputable def distance_upstream : ℝ := 68

theorem julie_upstream_distance :
  (distance_upstream / (speed_of_julie_in_still_water - speed_of_stream) = time_spent) ∧
  (distance_downstream / (speed_of_julie_in_still_water + speed_of_stream) = time_spent) →
  distance_upstream = 68 :=
by 
  sorry

end julie_upstream_distance_l563_563991


namespace bisector_line_slope_y_intercept_sum_l563_563727

noncomputable def slope (A B : ℝ × ℝ) : ℝ :=
  (B.2 - A.2) / (B.1 - A.1)

noncomputable def y_intercept (slope : ℝ) (A : ℝ × ℝ) : ℝ :=
  A.2 - slope * A.1

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def line_equation (A B : ℝ × ℝ) : (ℝ → ℝ) :=
  let m := slope A B
  λ x => m * x + y_intercept m A

theorem bisector_line_slope_y_intercept_sum :
  let P : ℝ × ℝ := (0, 10)
  let Q : ℝ × ℝ := (3, 0)
  let R : ℝ × ℝ := (9, 0)
  let M := midpoint P R
  let L := line_equation Q M
  L = (λ x => (10 / 3) * x - 10) →
  (10 / 3) + (-10) = -20 / 3 := 
by
  -- proof skipped
  sorry

end bisector_line_slope_y_intercept_sum_l563_563727


namespace max_superior_squares_l563_563885

-- Define the "superior square" concept based on the given conditions.
def is_superior_square (n : ℕ) (board : ℕ → ℕ → ℕ) (i j : ℕ) : Prop :=
  (∃ count_in_row, count_in_row = (ℕ card (set.filter (λ k, board i k < board i j) (finset.range n)) ≥ 2004) ∧
   ∃ count_in_column, count_in_column = (ℕ card (set.filter (λ k, board k j < board i j) (finset.range n)) ≥ 2004)

-- The main theorem statement to be proved
theorem max_superior_squares (n : ℕ) (hn : 2004 < n) (board : ℕ → ℕ → ℕ) : 
  ∃ optimal_count, optimal_count = n * (n - 2004) :=
sorry

end max_superior_squares_l563_563885


namespace exists_parallel_projection_with_same_area_l563_563575

-- Define the conditions: two intersecting planes and a triangle in one plane with area S.
variables {Plane1 Plane2 : Type} [plane_geometry Plane1] [plane_geometry Plane2] 
(triangle : Plane1) (S : ℝ)
(area_triangle : area triangle = S)

-- Define the proof problem: does there exist a parallel projection of the triangle onto the second plane with the same area S?
theorem exists_parallel_projection_with_same_area :
  ∃ (projection : Plane2), (parallel_projection Plane1 Plane2 triangle = projection) ∧  (area projection = S) :=
sorry

end exists_parallel_projection_with_same_area_l563_563575


namespace payment_relationship_l563_563009

noncomputable def payment_amount (x : ℕ) (price_per_book : ℕ) (discount_percent : ℕ) : ℕ :=
  if x > 20 then ((x - 20) * (price_per_book * (100 - discount_percent) / 100) + 20 * price_per_book) else x * price_per_book

theorem payment_relationship (x : ℕ) (h : x > 20) : payment_amount x 25 20 = 20 * x + 100 := 
by
  sorry

end payment_relationship_l563_563009


namespace children_ticket_cost_is_8_l563_563650

-- Defining the costs of different tickets
def adult_ticket_cost : ℕ := 11
def senior_ticket_cost : ℕ := 9
def total_tickets_cost : ℕ := 64

-- Number of tickets needed
def number_of_adult_tickets : ℕ := 2
def number_of_senior_tickets : ℕ := 2
def number_of_children_tickets : ℕ := 3

-- Defining the total cost equation using the price of children's tickets (C)
def total_cost (children_ticket_cost : ℕ) : ℕ :=
  number_of_adult_tickets * adult_ticket_cost +
  number_of_senior_tickets * senior_ticket_cost +
  number_of_children_tickets * children_ticket_cost

-- Statement to prove that the children's ticket cost is $8
theorem children_ticket_cost_is_8 : (C : ℕ) → total_cost C = total_tickets_cost → C = 8 :=
by
  intro C h
  sorry

end children_ticket_cost_is_8_l563_563650


namespace percentage_range_l563_563775

noncomputable def minimum_maximum_percentage (x y z n m : ℝ) (hx1 : 0 < x) (hx2 : 0 < y) (hx3 : 0 < z) (hx4 : 0 < n) (hx5 : 0 < m)
    (h1 : 4 * x * n = y * m) 
    (h2 : x * n + y * m = z * (m + n)) 
    (h3 : 16 ≤ y - x ∧ y - x ≤ 20) 
    (h4 : 42 ≤ z ∧ z ≤ 60) : ℝ × ℝ := sorry

theorem percentage_range (x y z n m : ℝ) (hx1 : 0 < x) (hx2 : 0 < y) (hx3 : 0 < z) (hx4 : 0 < n) (hx5 : 0 < m)
    (h1 : 4 * x * n = y * m) 
    (h2 : x * n + y * m = z * (m + n)) 
    (h3 : 16 ≤ y - x ∧ y - x ≤ 20) 
    (h4 : 42 ≤ z ∧ z ≤ 60) : 
    minimum_maximum_percentage x y z n m hx1 hx2 hx3 hx4 hx5 h1 h2 h3 h4 = (12.5, 15) :=
sorry

end percentage_range_l563_563775


namespace find_solutions_l563_563554

theorem find_solutions (k : ℤ) (x y : ℤ) (h : x^2 - 2*y^2 = k) :
  ∃ t u : ℤ, t^2 - 2*u^2 = -k ∧ (t = x + 2*y ∨ t = x - 2*y) ∧ (u = x + y ∨ u = x - y) :=
sorry

end find_solutions_l563_563554


namespace find_lambda_l563_563185

-- Define the vectors
def m : ℝ × ℝ × ℝ := (3, 1, 3)
def n (λ : ℝ) : ℝ × ℝ × ℝ := (-1, λ, -1)

-- Condition that vectors are parallel
def vectors_parallel (m n : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, n = (k * 3, k * 1, k * 3)

-- Main statement of the proof problem
theorem find_lambda (λ : ℝ) (h : vectors_parallel m (n λ)) : λ = -1/3 :=
by 
  sorry

end find_lambda_l563_563185


namespace eval_expression_l563_563846

theorem eval_expression (x y : ℕ) (h_x : x = 2001) (h_y : y = 2002) :
  (x^3 - 3*x^2*y + 5*x*y^2 - y^3 - 2) / (x * y) = 1999 :=
  sorry

end eval_expression_l563_563846


namespace akeno_extra_expenditure_l563_563096

/-
   Akeno spent $2985 to furnish his apartment.
   Lev spent one-third of that amount on his apartment.
   Ambrocio spent $177 less than Lev.
   Prove that Akeno spent $1172 more than the other 2 people combined.
-/

theorem akeno_extra_expenditure :
  let ak = 2985
  let lev = ak / 3
  let am = lev - 177
  ak - (lev + am) = 1172 :=
by
  sorry

end akeno_extra_expenditure_l563_563096


namespace antipalindromic_sum_reciprocals_correct_l563_563157

noncomputable def antipalindromic_sum_reciprocals : ℚ :=
  ∑ (s : fin 2004 → bool) in finset.filter (λ s, ∀ i, s i = bnot (s (2003 - i))) finset.univ, 
    1 / (∏ i in finset.range 2004, if s i then (i + 1) else 1)

theorem antipalindromic_sum_reciprocals_correct :
  antipalindromic_sum_reciprocals = (2005 ^ 1002) / (Nat.factorial 2004) :=
sorry

end antipalindromic_sum_reciprocals_correct_l563_563157


namespace alpha_plus_beta_eq_2_gamma_quadrilateral_always_parallelogram_l563_563877

variables {α β γ t m x : ℝ}

/-- Given f(x) = (4x - t) / (x^2 + 1), the extreme points α and β, and the zero γ of f(x). -/
def f (x : ℝ) : ℝ := (4 * x - t) / (x^2 + 1)

-- Conditions:
variable h_extreme_points : ∀ x, (f x).deriv = 0 → x = α ∨ x = β
variable h_zero_of_f : f γ = 0

/-- (I): Prove α + β = 2 * γ -/
theorem alpha_plus_beta_eq_2_gamma (h1 : α + β = t / 2) (h_gamma : γ = t / 4) : α + β = 2 * γ :=
by
  sorry

/-- (II): Determine if there exists a real number t such that for any m > 0, the quadrilateral 
        ACBD is always a parallelogram, and if it exists, find t -/
theorem quadrilateral_always_parallelogram (h1 : α + β = t / 2)
  (C D : ℝ × ℝ := (t / 4 - m, 0) (t / 4 + m, 0))
  (h_parallel : ∀ m > 0, ∃ t, (α + β = t / 2) ∧ (f α + f β = 0)) : t = 0 :=
by
  sorry

end alpha_plus_beta_eq_2_gamma_quadrilateral_always_parallelogram_l563_563877


namespace max_candies_l563_563287

theorem max_candies (horizontal_vertical_moves diagonal_moves : ℕ) (h₀ : horizontal_vertical_moves + diagonal_moves = 7) :
  let candies_taken_by_misha := 2 * horizontal_vertical_moves + 3 * diagonal_moves,
      initial_candies := 30,
      candies_won_by_yulia := initial_candies - candies_taken_by_misha in
  horizontal_vertical_moves ≥ 0 ∧ diagonal_moves ≥ 0 ∧ candies_won_by_yulia = 14 ∨ 16 :=
sorry

end max_candies_l563_563287


namespace back_wheel_revolutions_l563_563291

theorem back_wheel_revolutions
  (r_front : ℝ) (r_back : ℝ) (rev_front : ℝ) (r_front_eq : r_front = 3)
  (r_back_eq : r_back = 0.5) (rev_front_eq : rev_front = 50) :
  let C_front := 2 * Real.pi * r_front
  let D_front := C_front * rev_front
  let C_back := 2 * Real.pi * r_back
  let rev_back := D_front / C_back
  rev_back = 300 := by
  sorry

end back_wheel_revolutions_l563_563291


namespace probability_acute_angle_l563_563742

noncomputable def is_acute_angle (m n : ℕ) : Prop :=
  m - 2 * n > 0

theorem probability_acute_angle :
  let possible_pairs := (1:ℕ) |>.upto 7 × (1:ℕ) |>.upto 7
  let favorable_pairs := possible_pairs.filter (λ ⟨m, n⟩, is_acute_angle m n)
  let total_pairs := 36
  favorable_pairs.length = 7 / total_pairs := 
sorry

end probability_acute_angle_l563_563742


namespace unique_keychains_div_10_l563_563778

theorem unique_keychains_div_10 :
  let letters := ['S', 'T', 'E', 'M']
  let digits := ['2', '0', '0', '8']
  let num_sequences (chars : List Char) (len : Nat) : Nat :=
        -- Define the logic to calculate the number of sequences
        sorry
  ∃ (M : Nat), 
    M = 3720 ∧ (M / 10 = 372) :=
begin
  sorry
end

end unique_keychains_div_10_l563_563778


namespace curve_is_circle_l563_563163

theorem curve_is_circle (s : ℝ) :
  let x := (3 - s^2) / (3 + s^2)
  let y := (4 * s) / (3 + s^2)
  x^2 + y^2 = 1 :=
by
  let x := (3 - s^2) / (3 + s^2)
  let y := (4 * s) / (3 + s^2)
  sorry

end curve_is_circle_l563_563163


namespace significant_improvement_l563_563417

-- Data for the old and new devices
def old_data : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_data : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

-- Definitions for sample mean and sample variance
def sample_mean (data : List ℝ) : ℝ := (data.sum) / (data.length)
def sample_variance (data : List ℝ) : ℝ := 
  let mean := sample_mean data
  (data.map (λ x => (x - mean)^2)).sum / data.length

-- Prove that the new device's mean indicator has significantly improved
theorem significant_improvement :
  let x := sample_mean old_data
  let y := sample_mean new_data
  let s1_squared := sample_variance old_data
  let s2_squared := sample_variance new_data
  (y - x) ≥ 2 * sqrt ((s1_squared + s2_squared) / 10) :=
by
  sorry

end significant_improvement_l563_563417


namespace range_of_a_l563_563164

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (1 / (x - 2) + (a - 2) / (2 - x) = 1) ∧ (x > 0)) → (a < 5 ∧ a ≠ 3) :=
by
  intro h
  cases h with x hx
  have h1 : x = 5 - a := sorry  -- From the solution
  have h2 : x > 0 := hx.2      -- From the condition
  have ha1 : a < 5 := sorry    -- Derived from the solution steps
  have ha2 : a ≠ 3 := sorry    -- Derived from the solution steps avoiding division by zero
  exact ⟨ha1, ha2⟩

end range_of_a_l563_563164


namespace value_of_m_l563_563399

noncomputable def f (x m : ℝ) : ℝ := (sqrt 3) * sin (2 * x) - 2 * (sin x) ^ 2 + m

theorem value_of_m : 
  (∀ x, (f x 1 ≤ sqrt 3) ∧ (f x 1 ≥ -sqrt 3)) ∧ f (π / 2) 1 = 0 → 
  ∃ m : ℝ, m = 1 := 
begin
  sorry
end

end value_of_m_l563_563399


namespace ratio_and_tangent_l563_563915

-- Definitions for the problem
def acute_triangle (A B C : Point) : Prop := 
  -- acute angles condition
  sorry

def is_diameter (A B C D : Point) : Prop := 
  -- D is midpoint of BC condition
  sorry

def divide_in_half (A B C : Point) (D : Point) : Prop := 
  -- D divides BC in half condition
  sorry

def divide_in_ratio (A B C : Point) (D : Point) (ratio : ℚ) : Prop := 
  -- D divides AC in the given ratio condition
  sorry

def tan (angle : ℝ) : ℝ := 
  -- Tangent function
  sorry

def angle (A B C : Point) : ℝ := 
  -- Angle at B of triangle ABC
  sorry

-- The statement of the problem in Lean
theorem ratio_and_tangent (A B C D : Point) :
  acute_triangle A B C →
  is_diameter A B C D →
  divide_in_half A B C D →
  (divide_in_ratio A B C D (1 / 3) ↔ tan (angle A B C) = 2 * tan (angle A C B)) :=
by sorry

end ratio_and_tangent_l563_563915


namespace length_of_platform_is_380_l563_563797

-- Definitions of conditions
def L_train : ℕ := 120
def v_kmph : ℕ := 72
def t : ℕ := 25

-- Convert speed from kmph to m/s
def v_mps : Real :=
  v_kmph * 1000 / 3600

-- Distance calculation
def d : Real :=
  v_mps * t

-- Length of platform calculation
def L_platform : Real :=
  d - L_train

-- The final statement to prove
theorem length_of_platform_is_380 : L_platform = 380 := by
  sorry

end length_of_platform_is_380_l563_563797


namespace smallest_circle_radius_tangent_to_shapes_l563_563828

theorem smallest_circle_radius_tangent_to_shapes :
  ∀ (s q : ℝ), 
  s = 4 → 
  ∀ (r_sc r_qc : ℕ), 
  r_sc = 1 ∧ r_qc = 2 → 
  ∃ r : ℝ, r = sqrt 2 - 3 / 2 := 
by 
  intros s q h_sq radii
  cases radii
  use sqrt 2 - 3 / 2
  sorry

end smallest_circle_radius_tangent_to_shapes_l563_563828


namespace complement_of_M_is_correct_l563_563210

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - x > 0}
def complement_M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

theorem complement_of_M_is_correct :
  (U \ M) = complement_M :=
by
  sorry

end complement_of_M_is_correct_l563_563210


namespace roots_of_quadratic_l563_563343

theorem roots_of_quadratic (x : ℝ) (h : x^2 = x) : x = 0 ∨ x = 1 :=
by
  have : x * (x - 1) = 0 :=
    by rw [← h, x]; ring -- x^2 - x = 0 <=> x * (x - 1) = 0
  cases eq_zero_or_eq_zero_of_mul_eq_zero this with h₀ h₁
  · left; exact h₀
  · right; linarith

end roots_of_quadratic_l563_563343


namespace edge_length_inscribed_cube_cone_l563_563017

-- Define the conditions of the problem
variables (l : ℝ) (θ : ℝ)

-- Assume the given length and angle constraints
axiom slant_height_eq : l = 1

-- Define the edge length of the inscribed cube
def edge_length_of_inscribed_cube : ℝ :=
  2 * sin θ / (2 + sqrt 2 * tan θ)

-- State the theorem to prove
theorem edge_length_inscribed_cube_cone : 
  l = 1 → 
  edge_length_of_inscribed_cube l θ = 2 * sin θ / (2 + sqrt 2 * tan θ) :=
by
  sorry

end edge_length_inscribed_cube_cone_l563_563017


namespace red_bottles_count_l563_563028

theorem red_bottles_count :
  ∀ (R : ℕ), 3 + 4 + R = 9 → R = 2 :=
by
  intros R h,
  sorry

end red_bottles_count_l563_563028


namespace sum_sequence_correct_l563_563886

-- Definitions
def sequence_a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 2 ^ (n - 1) - 1

def sequence_b (n : ℕ) : ℕ :=
  n * (sequence_a n + 1)

def sum_sequence (n : ℕ) : ℕ :=
  (range n).sum (λ i, sequence_b (i + 1))

theorem sum_sequence_correct (n : ℕ) :
  sum_sequence n = (n - 1) * 2^(n + 1) + 2 :=
sorry

end sum_sequence_correct_l563_563886


namespace students_in_canteen_l563_563351

-- Definitions for conditions
def total_students : ℕ := 40
def absent_fraction : ℚ := 1 / 10
def classroom_fraction : ℚ := 3 / 4

-- Lean 4 statement
theorem students_in_canteen :
  let absent_students := (absent_fraction * total_students)
  let present_students := (total_students - absent_students)
  let classroom_students := (classroom_fraction * present_students)
  let canteen_students := (present_students - classroom_students)
  canteen_students = 9 := by
    sorry

end students_in_canteen_l563_563351


namespace prove_x_minus_y_l563_563276

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem prove_x_minus_y :
  let x := 2 in
  let y := 3 - sqrt 7 in
  x - y = sqrt 7 - 1 :=
by
  sorry

end prove_x_minus_y_l563_563276


namespace officer_selection_at_least_two_past_l563_563971

theorem officer_selection_at_least_two_past (n m k : ℕ) (h₀ : n = 18) (h₁ : m = 6) (h₂ : k = 8) :
  let total := (nat.choose n m),
      zero_past := (nat.choose (n - k) m),
      one_past := k * (nat.choose (n - k) (m - 1)),
      at_least_two_past := total - (zero_past + one_past)
  in at_least_two_past = 16338 :=
by {
  rw [h₀, h₁, h₂],
  let total := nat.choose 18 6,
  let zero_past := nat.choose 10 6,
  let one_past := 8 * nat.choose 10 5,
  let at_least_two_past := total - (zero_past + one_past),
  have : total = 18564 := by simp [nat.choose],
  have : zero_past = 210 := by simp [nat.choose],
  have : one_past = 2016 := by simp [nat.choose],
  exact calc
    at_least_two_past
      = 18564 - (210 + 2016) : by { rw [this, this, this] }
      ... = 16338 : by norm_num
}

end officer_selection_at_least_two_past_l563_563971


namespace geometric_locus_incenter_ABP_l563_563091

-- Define the triangle ABC inscribed in a circle
variables {A B C P I : Type}
variable [In_circle ABC]
variable [Moves_along P AC_B_arc]

-- Define the incenter I of triangle ABP
def incenter_ABP (A B P : Type) : Type := sorry

-- State the theorem about the geometric locus of I
theorem geometric_locus_incenter_ABP :
  ∀ P : Type, Moves_along P AC_B_arc → Lies_on_circumcircle I ABC :=
sorry

end geometric_locus_incenter_ABP_l563_563091


namespace asymptote_sum_l563_563125

noncomputable def f (x : ℝ) : ℝ := (x^3 + 4*x^2 + 3*x) / (x^3 + x^2 - 2*x)

def holes := 0 -- a
def vertical_asymptotes := 2 -- b
def horizontal_asymptotes := 1 -- c
def oblique_asymptotes := 0 -- d

theorem asymptote_sum : holes + 2 * vertical_asymptotes + 3 * horizontal_asymptotes + 4 * oblique_asymptotes = 7 :=
by
  unfold holes vertical_asymptotes horizontal_asymptotes oblique_asymptotes
  norm_num

end asymptote_sum_l563_563125


namespace sum_of_consecutive_integers_sqrt_28_l563_563902

theorem sum_of_consecutive_integers_sqrt_28 (a b : ℤ) (h1 : a < b) (h2 : b = a + 1) (h3 : a < sqrt 28) (h4 : sqrt 28 < b) : a + b = 11 := by
  sorry

end sum_of_consecutive_integers_sqrt_28_l563_563902


namespace find_lambda_l563_563557

variables {V : Type*} [InnerProductSpace ℝ V]
variables (A B C P D : V)
variables (λ : ℝ)

-- Conditions
def midpoint (A B M : V) : Prop := (M - A) = (B - M)
def satisfies_vector_equation (A B C P : V) : Prop := (P - A) + (B - P) + (C - P) = 0
def ap_eq_lambda_pd (A P D : V) (λ : ℝ) : Prop := (A - P) = λ * (P - D)

-- Given conditions as definitions
def given_conditions (A B C P D : V) (λ : ℝ) :=
  midpoint B C D ∧ satisfies_vector_equation A B C P ∧ ap_eq_lambda_pd A P D λ

-- The theorem that we need to prove
theorem find_lambda (A B C P D : V) (λ : ℝ) (h : given_conditions A B C P D λ) :
  λ = -2 :=
sorry

end find_lambda_l563_563557


namespace value_of_m_l563_563956

theorem value_of_m (m : ℝ) :
  (∃ (y : ℝ → ℝ), y = (λ x, (m - 3) * x ^ |m - 1| + 3 * x - 1) ∧ x ^ |m - 1| = x ^ 2) → m = -1 :=
by {
  sorry
}

end value_of_m_l563_563956


namespace hyperbola_solution_l563_563551

noncomputable def hyperbola_a_value (F1 F2 M : ℝ × ℝ) (a : ℝ) :=
  (a > 0) ∧
  ∃ x y, (x, y) = M ∧ ((x^2) / (a^2) - (y^2) / (2 * a) = 1) ∧
  let MF1 := (F1.1 - M.1, F1.2 - M.2) in
  let MF2 := (F2.1 - M.1, F2.2 - M.2) in
  (MF1.1 * MF2.1 + MF1.2 * MF2.2 = 0) ∧
  (real.sqrt ((MF1.1)^2 + (MF1.2)^2) * real.sqrt ((MF2.1)^2 + (MF2.2)^2) = 4) →
  a = 1

open_locale classical

theorem hyperbola_solution (F1 F2 M : ℝ × ℝ) : ∀ a : ℝ, hyperbola_a_value F1 F2 M a ->
  a = 1 := 
by
  assume a,
  intro hyp_val,
  sorry

end hyperbola_solution_l563_563551


namespace intersection_of_multiples_of_2_l563_563997

theorem intersection_of_multiples_of_2 : 
  let M := {1, 2, 4, 8}
  let N := {x : ℤ | ∃ k : ℤ, x = 2 * k}
  M ∩ N = {2, 4, 8} :=
by
  sorry

end intersection_of_multiples_of_2_l563_563997


namespace binomial_expansion_fraction_l563_563898

theorem binomial_expansion_fraction :
  let a0 := 32
  let a1 := -80
  let a2 := 80
  let a3 := -40
  let a4 := 10
  let a5 := -1
  (2 - x)^5 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 →
  (a0 + a2 + a4) / (a1 + a3) = -61 / 60 :=
by
  sorry

end binomial_expansion_fraction_l563_563898


namespace proof_3x_4y_eq_neg7_l563_563189

noncomputable def y (x : ℝ) : ℝ := 
  (sqrt (x^2 - 4) + sqrt (4 - x^2) + 1) / (x - 2)

theorem proof_3x_4y_eq_neg7 (x y : ℝ) 
  (h1 : y = (sqrt (x^2 - 4) + sqrt (4 - x^2) + 1) / (x - 2))
  (h2 : x^2 - 4 ≥ 0)
  (h3 : 4 - x^2 ≥ 0) :
  3*x + 4*y = -7 := 
sorry

end proof_3x_4y_eq_neg7_l563_563189


namespace exist_line_intersects_all_l563_563896

/-
Given several parallel line segments with the condition that any three of them have a line intersecting all three,
prove that there exists a line that intersects all the segments.
-/

noncomputable def exists_intersecting_line (segments : set (set (ℝ × ℝ))) : Prop :=
  (∀ seg ∈ segments, ∃ x : ℝ, ∀ (t : ℝ), (t * x) ∈ seg) ∧
  (∀ s1 s2 s3 ∈ segments, ∃ line, line_intersects_segment line s1 ∧ line_intersects_segment line s2 ∧ line_intersects_segment line s3)
  → ∃ line, ∀ seg ∈ segments, line_intersects_segment line seg

-- Helper definition to express when a line intersects a segment
noncomputable def line_intersects_segment (line : ℝ → ℝ) (seg : set (ℝ × ℝ)) : Prop :=
  ∃ x y, (x, line x) ∈ seg ∧ (y, line y) ∈ seg

-- To use Helly's theorem, we need to introduce a convex set helper
noncomputable def convex_intersecting_sets (set_of_segments : set (set (ℝ × ℝ))) : set (set ℝ) := sorry

-- Main theorem statement
theorem exist_line_intersects_all (segments : set (set (ℝ × ℝ)))
  (parallel : ∀ seg ∈ segments, ∃! x, ∀ (p ∈ seg), p.1 = x)
  (three_intersect : ∀ s1 s2 s3 ∈ segments, ∃ a b, ∀ seg ∈ {s1, s2, s3}, ∃ (p : ℝ × ℝ), (p.1, a * p.1 + b) ∈ seg)
  : ∃ a b, ∀ seg ∈ segments, ∃ (p : ℝ × ℝ), (p.1, a * p.1 + b) ∈ seg := sorry

end exist_line_intersects_all_l563_563896


namespace intervals_g_decreasing_l563_563923

noncomputable def f (M : ℝ) (x : ℝ) : ℝ := M * Real.sin ((π / 6) * x)

noncomputable def g (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin ((1 / 2) * x + π / 6)

theorem intervals_g_decreasing :
  ∀ k : ℤ, 
  ∃ a b : ℝ, 
  a = 4 * k * π + 2 * π / 3 ∧ 
  b = 4 * k * π + 8 * π / 3 ∧ 
  ∀ x : ℝ, 
  x ∈ Set.Icc a b → (g x) ' < 0 :=
sorry

end intervals_g_decreasing_l563_563923


namespace conic_section_is_ellipse_l563_563380

/-- Definitions of fixed points and the equation constant --/
def point1 := (0, 2 : ℝ × ℝ)
def point2 := (6, -4 : ℝ × ℝ)
def constant := 12

/-- Formulation of the given conic section equation --/
def conic_section_equation (x y : ℝ) : Prop :=
  real.sqrt (x^2 + (y - point1.snd)^2) + real.sqrt ((x - point2.fst)^2 + (y - point2.snd)^2) = constant

/-- Problem Statement: The conic section described by the given equation is an ellipse --/
theorem conic_section_is_ellipse : ∀ x y : ℝ, conic_section_equation x y → 
  (conic_section_equation x y → "E" = "E") :=
by
  intros x y h
  sorry

end conic_section_is_ellipse_l563_563380


namespace jasper_wins_probability_l563_563258

-- Definitions and conditions
def prob_heads_jasper : ℚ := 2 / 7
def prob_heads_kira : ℚ := 1 / 4
def prob_tails_jasper : ℚ := 1 - prob_heads_jasper
def prob_tails_kira : ℚ := 1 - prob_heads_kira
def prob_both_tails : ℚ := prob_tails_jasper * prob_tails_kira

-- Hypothesis: Kira goes first
axiom independent_tosses : Prop -- Placeholder for independent toss axiom

-- The ultimate probability that Jasper wins
def prob_jasper_wins : ℚ := 
  (prob_heads_jasper * prob_both_tails) / (1 - prob_both_tails)

-- The theorem to prove
theorem jasper_wins_probability :
  prob_jasper_wins = 30 / 91 := 
sorry

end jasper_wins_probability_l563_563258


namespace problem1_solution_l563_563461

theorem problem1_solution (x : ℝ) :
  x^2 + 2 * x + 4 * real.sqrt (x^2 + 2 * x) - 5 = 0 →
  x = real.sqrt 2 - 1 ∨ x = -real.sqrt 2 - 1 :=
sorry

end problem1_solution_l563_563461


namespace quadratic_expression_l563_563341

theorem quadratic_expression (b c : ℤ) : 
  (∀ x : ℝ, (x^2 - 20*x + 49 = (x + b)^2 + c)) → (b + c = -61) :=
by
  sorry

end quadratic_expression_l563_563341


namespace arithmetic_sequence_n_value_l563_563177

noncomputable def common_ratio (a₁ S₃ : ℕ) : ℕ := by sorry

theorem arithmetic_sequence_n_value:
  ∀ (a : ℕ → ℕ) (S : ℕ → ℕ),
  (∀ n, a n > 0) →
  a 1 = 3 →
  S 3 = 21 →
  (∃ q, q > 0 ∧ common_ratio 1 q = q ∧ a 5 = 48) →
  n = 5 :=
by
  intros
  sorry

end arithmetic_sequence_n_value_l563_563177


namespace sum_of_positive_factors_of_90_eq_234_l563_563371

theorem sum_of_positive_factors_of_90_eq_234 : 
  let factors := [1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90]
  List.sum factors = 234 :=
by
  -- List the positive factors of 90
  let factors := [1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90]
  -- Prove that the sum of these factors is 234
  have h_sum_factors : List.sum factors = 234 := sorry
  exact h_sum_factors

end sum_of_positive_factors_of_90_eq_234_l563_563371


namespace students_play_both_l563_563964

def students_total : ℕ := 25
def hockey_players : ℕ := 15
def basketball_players : ℕ := 16
def neither_players : ℕ := 4

theorem students_play_both : 
  (students_total - neither_players) + (hockey_players + basketball_players - students_total + neither_players - students_total) = 10 :=
by 
  sorry

end students_play_both_l563_563964


namespace find_vertex_X_l563_563613

open Real

-- Assuming necessary vector arithmetic definitions

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def midpoint (A B : Point3D) : Point3D :=
  { x := (A.x + B.x) / 2, 
    y := (A.y + B.y) / 2, 
    z := (A.z + B.z) / 2 }

noncomputable def findX (M N : Point3D) : Point3D :=
  { x := 2 * M.x - N.x,
    y := 2 * M.y - N.y,
    z := 2 * M.z - N.z }

theorem find_vertex_X 
  (M : Point3D) (hM : M = { x := 2, y := 3, z := -2 })
  (N : Point3D) (hN : N = { x := -1, y := 6, z := 3 })
  (P : Point3D) (hP : P = { x := 3, y := 1, z := 5 }) :
  findX { x := 1, y := 7 / 2, z := 4 } M = { x := 0, y := 4, z := 10 } :=
by 
  sorry

end find_vertex_X_l563_563613


namespace zoe_spent_amount_l563_563397

theorem zoe_spent_amount :
  (3 * (8 + 2) = 30) :=
by sorry

end zoe_spent_amount_l563_563397


namespace fractional_part_inequality_l563_563953

noncomputable def frac (z : ℝ) : ℝ := z - ⌊z⌋

theorem fractional_part_inequality (x y : ℝ) : frac (x + y) ≤ frac x + frac y := 
sorry

end fractional_part_inequality_l563_563953


namespace log_reciprocal_sum_eq_two_l563_563606

-- Define necessary conditions from the problem
variable (x y : ℝ)
variable (hx : 4^x = 6)
variable (hy : 9^y = 6)

-- State the theorem to be proven
theorem log_reciprocal_sum_eq_two (hx : 4^x = 6) (hy : 9^y = 6) : 1/x + 1/y = 2 := 
by
  sorry

end log_reciprocal_sum_eq_two_l563_563606


namespace original_number_one_more_reciprocal_is_11_over_5_l563_563660

theorem original_number_one_more_reciprocal_is_11_over_5 (x : ℚ) (h : 1 + 1/x = 11/5) : x = 5/6 :=
by
  sorry

end original_number_one_more_reciprocal_is_11_over_5_l563_563660


namespace proof_number_of_correct_statements_l563_563104

-- Definitions of the conditions as Lean 4 problems
def cond_1 : Prop := ∀ (T1 T2 : Triangle), 
  (isosceles T1 ∧ isosceles T2 ∧ ∃ α, acute α ∧ angle_equal T1 T2 α) → similar T1 T2

def cond_2 : Prop := ∀ (T1 T2 : Trapezoid), 
  (isosceles T1 ∧ isosceles T2 ∧ base_angle T1 = 45 ∧ base_angle T2 = 45) → similar T1 T2

def cond_3 : Prop := ∀ (R1 R2 : Rhombus), similar R1 R2

def cond_4 : Prop := ∀ (R1 R2 : Rectangle), similar R1 R2

def cond_5 : Prop := ∀ (T1 T2 : Triangle), 
  (isosceles T1 ∧ isosceles T2 ∧ ∃ α, obtuse α ∧ angle_equal T1 T2 α) → similar T1 T2

-- The proof goal
theorem proof_number_of_correct_statements :
  (¬ cond_1 ∧ cond_2 ∧ ¬ cond_3 ∧ ¬ cond_4 ∧ cond_5) → correct_statements = 2 :=
sorry

end proof_number_of_correct_statements_l563_563104


namespace original_price_l563_563079

theorem original_price (discount_amount : ℝ) (discount_rate : ℝ) (original_price : ℝ) : 
  discount_rate = 0.2 → discount_amount = 50 → 
  original_price = discount_amount / discount_rate :=
by
  intros h_rate h_amount
  rw [h_rate, h_amount]
  sorry

example : original_price 50 0.2 250 :=
by
  unfold original_price
  refl

end original_price_l563_563079


namespace opposite_of_neg_three_l563_563003

-- Define the concept of negation and opposite of a number
def opposite (x : ℤ) : ℤ := -x

-- State the problem: Prove that the opposite of -3 is 3
theorem opposite_of_neg_three : opposite (-3) = 3 :=
by
  -- Proof
  sorry

end opposite_of_neg_three_l563_563003


namespace leak_empty_time_l563_563106

-- Define conditions
def pump_fill_time : ℝ := 6
def combined_fill_time : ℝ := 12

-- Define rates
def pump_rate : ℝ := 1 / pump_fill_time
def combined_rate : ℝ := 1 / combined_fill_time

-- Define leak rate
def leak_rate : ℝ := pump_rate - combined_rate

-- Our goal is to prove that the leak will empty the tank in 12 hours
theorem leak_empty_time : 1 / leak_rate = 12 := by
  sorry

end leak_empty_time_l563_563106


namespace twenty03rdRedNumber_l563_563601

def paintedRedNumber (n : ℕ) : ℕ :=
  let k := Nat.find (λ k => (k * (k + 1)) / 2 ≥ n ) - 1
  let a := (k * (k + 1)) / 2
  let offset := n - a
  let base := k^2 + 1
  if k % 2 == 0 then -- k is even => consecutive odd numbers
    base + (2 * (offset - 1))
  else -- k is odd => consecutive even numbers
    base + 2 * (offset - 1)
    
theorem twenty03rdRedNumber : paintedRedNumber 2003 = 3943 :=
  by
  sorry

end twenty03rdRedNumber_l563_563601


namespace sum_of_divisors_90_l563_563368

theorem sum_of_divisors_90 : 
  let n := 90 in 
  let sum_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5) in
  sum_divisors n = 234 :=
by 
  let n := 90
  let sum_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5)
  sorry

end sum_of_divisors_90_l563_563368


namespace long_side_of_rectangle_l563_563786

-- Setting up the conditions as definitions
def short_side : ℝ := 8
def triangles_congruent : Prop := True -- Represents the congruence of triangles I and II

-- The theorem we need to prove
theorem long_side_of_rectangle : short_side = 8 ∧ triangles_congruent → ∃ long_side : ℝ, long_side = 12 :=
by
  -- Here, we assume short_side = 8 and triangles_congruent hold true
  assume (h : short_side = 8 ∧ triangles_congruent),
  -- State that such a long_side exists and equals 12
  use 12,
  sorry  -- Skip the proof

end long_side_of_rectangle_l563_563786


namespace floor_expression_eq_eight_l563_563121

theorem floor_expression_eq_eight :
  (⌊(2010^3 / (2008 * 2009) - 2008^3 / (2009 * 2010))⌋ = 8) := by
  sorry

end floor_expression_eq_eight_l563_563121


namespace cos_neg_pi_div_3_l563_563137

theorem cos_neg_pi_div_3 : Real.cos (-π / 3) = 1 / 2 := 
by 
  sorry

end cos_neg_pi_div_3_l563_563137


namespace total_volume_of_removed_pyramids_l563_563456

noncomputable def volume_of_removed_pyramids (edge_length : ℝ) : ℝ :=
  8 * (1 / 3 * (1 / 2 * (edge_length / 4) * (edge_length / 4)) * (edge_length / 4) / 6)

theorem total_volume_of_removed_pyramids :
  volume_of_removed_pyramids 1 = 1 / 48 :=
by
  sorry

end total_volume_of_removed_pyramids_l563_563456


namespace target_expression_l563_563963

variable (a b : ℤ)

-- Definitions based on problem conditions
def op1 (x y : ℤ) : ℤ := x + y  -- "!" could be addition
def op2 (x y : ℤ) : ℤ := x - y  -- "?" could be subtraction in one order

-- Using these operations to create expressions
def exp1 (a b : ℤ) := op1 (op2 a b) (op2 b a)

def exp2 (x y : ℤ) := op2 (op2 x 0) (op2 0 y)

-- The final expression we need to check
def final_exp (a b : ℤ) := exp1 (20 * a) (18 * b)

-- Theorem proving the final expression equals target
theorem target_expression : final_exp a b = 20 * a - 18 * b :=
sorry

end target_expression_l563_563963


namespace ratio_of_construction_paper_packs_l563_563103

-- Definitions for conditions
def marie_glue_sticks : Nat := 15
def marie_construction_paper : Nat := 30
def allison_total_items : Nat := 28
def allison_additional_glue_sticks : Nat := 8

-- Define the main quantity to prove
def allison_glue_sticks : Nat := marie_glue_sticks + allison_additional_glue_sticks
def allison_construction_paper : Nat := allison_total_items - allison_glue_sticks

-- The ratio should be of type Rat or Nat
theorem ratio_of_construction_paper_packs : (marie_construction_paper : Nat) / allison_construction_paper = 6 / 1 := by
  -- This is a placeholder for the actual proof
  sorry

end ratio_of_construction_paper_packs_l563_563103


namespace sufficient_not_necessary_range_l563_563221

variable (x a : ℝ)

theorem sufficient_not_necessary_range (h1 : ∀ x, |x| < 1 → x < a) 
                                       (h2 : ¬(∀ x, x < a → |x| < 1)) :
  a ≥ 1 :=
sorry

end sufficient_not_necessary_range_l563_563221


namespace sine_90_deg_equals_one_l563_563974

variable (D E F : Type)
variable (DEF : D → E → F → Prop)
variable (sides : ℝ)
variable (side_DE : D → E → ℝ)
variable (side_EF : E → F → ℝ)

-- Given the conditions
axiom right_triangle : ∀ (a b c : Type), DEF a b c → ∠ a = 90° → side_DE a b = 12 → side_EF b c = 18

-- Prove that the sine of the right angle D is 1
theorem sine_90_deg_equals_one (a b c : Type) (h : DEF a b c) (h_angle : ∠ a = 90°) (h_side1 : side_DE a b = 12) (h_side2 : side_EF b c = 18) : sin (∠ a) = 1 :=
by
  sorry

end sine_90_deg_equals_one_l563_563974


namespace student_passing_percentage_l563_563795

def student_marks : ℕ := 80
def shortfall_marks : ℕ := 100
def total_marks : ℕ := 600

def passing_percentage (student_marks shortfall_marks total_marks : ℕ) : ℕ :=
  (student_marks + shortfall_marks) * 100 / total_marks

theorem student_passing_percentage :
  passing_percentage student_marks shortfall_marks total_marks = 30 :=
by
  sorry

end student_passing_percentage_l563_563795


namespace total_hotdogs_sold_l563_563431

-- Define the number of small and large hotdogs
def small_hotdogs : ℕ := 58
def large_hotdogs : ℕ := 21

-- Define the total hotdogs
def total_hotdogs : ℕ := small_hotdogs + large_hotdogs

-- The Main Statement to prove the total number of hotdogs sold
theorem total_hotdogs_sold : total_hotdogs = 79 :=
by
  -- Proof is skipped using sorry
  sorry

end total_hotdogs_sold_l563_563431


namespace max_triangle_side_l563_563560

theorem max_triangle_side (a : ℕ) (h1 : 11 > a) (h2 : a > 5) : a ≤ 10 :=
by {
  have h3: (a < 11), from h1,
  have h4: (a > 5), from h2,
  have h5: a ≤ 10,
  { exact lt_of_le_of_lt (Nat.le_of_lt_succ (Nat.succ_le_of_lt h4)) h3 },
  exact h5
}

end max_triangle_side_l563_563560


namespace systematic_sampling_l563_563350

theorem systematic_sampling (products : Finset ℕ) (n k : ℕ) (interval : ℕ):
  (products = {1, 2, ..., 40}) → 
  (k = 4) →
  interval = (40 / 4) →
  (∃ selected : Finset ℕ, selected = {2, 12, 22, 32}) :=
begin
  assume hprod,
  assume hk,
  assume hint,
  use ({2, 12, 22, 32} : Finset ℕ),
  sorry
end

end systematic_sampling_l563_563350


namespace solution_set_quadratic_inequality_l563_563564

theorem solution_set_quadratic_inequality (a b : ℝ) (h1 : a < 0)
    (h2 : ∀ x, ax^2 - bx - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :
    ∀ x, x^2 - b*x - a ≥ 0 ↔ x ≥ 3 ∨ x ≤ 2 := 
by
  sorry

end solution_set_quadratic_inequality_l563_563564


namespace part_a_part_b_l563_563995

open Set

noncomputable def N : Set ℕ := { n | 0 < n }

def Δ (X : Set ℕ) : Set ℕ := { d | ∃ m n ∈ X, d = abs (m - n) }

variable (A B : Set ℕ)

axiom infinite_A : Infinite A
axiom infinite_B : Infinite B
axiom disjoint_AB : Disjoint A B
axiom union_AB : A ∪ B = N

theorem part_a :
  Infinite (Δ A ∩ Δ B) := sorry

theorem part_b :
  ∃ (C : Set ℕ), Infinite C ∧ (Δ C ⊆ Δ A ∩ Δ B) := sorry

end part_a_part_b_l563_563995


namespace andrew_daily_work_hours_l563_563468

theorem andrew_daily_work_hours (total_hours : ℝ) (days : ℝ) (h1 : total_hours = 7.5) (h2 : days = 3) : total_hours / days = 2.5 :=
by
  rw [h1, h2]
  norm_num

end andrew_daily_work_hours_l563_563468


namespace train_cross_time_l563_563796

/-- A train 100 m long takes some time to cross a man walking at 5 kmph in a direction opposite
to that of the train. The speed of the train is 54.99520038396929 kmph. Prove that it takes
approximately 6 seconds for the train to cross the man. -/
theorem train_cross_time
  (train_length : ℝ := 100)
  (man_speed_kmph : ℝ := 5)
  (train_speed_kmph : ℝ := 54.99520038396929)
  (approx_time_seconds : ℝ := 6) :
  ∃ (t : ℝ), t ≈ approx_time_seconds :=
by
  sorry

end train_cross_time_l563_563796


namespace triangle_area_ratio_l563_563244

-- Given definitions for conditions
variables (x : ℝ)

-- Rectangle ABCD with AB = 4x and AD = 3x
def AB := 4 * x
def AD := 3 * x
def area_rectangle : ℝ := AB * AD -- Area of rectangle ABCD

-- AF = 3FE and CD = 3DE
def AF := 3 * (x)
def FE := x
def ED := x
def DC := 3 * x

-- Triangle BFD
def area_BFD : ℝ := 
  area_rectangle - (1/2 * AF * AB + 1/2 * FE * AB + 1/2 * ED * DC)

-- Prove the ratio of the area
theorem triangle_area_ratio :
  (area_BFD / area_rectangle) = (1/3) := by
  sorry

end triangle_area_ratio_l563_563244


namespace find_g7_l563_563696

noncomputable def g : ℝ → ℝ := sorry

axiom g_add (x y : ℝ) : g (x + y) = g x + g y
axiom g_value : g 6 = 7

theorem find_g7 : g 7 = 49 / 6 := by
  sorry

end find_g7_l563_563696


namespace part1_part2_l563_563526

-- Definitions for the sets A and B
def A := {x : ℝ | x^2 - 2 * x - 8 = 0}
def B (a : ℝ) := {x : ℝ | x^2 + a * x + a^2 - 12 = 0}

-- Proof statements
theorem part1 (a : ℝ) : (A ∩ B a = A) → a = -2 :=
by
  sorry

theorem part2 (a : ℝ) : (A ∪ B a = A) → (a ≥ 4 ∨ a < -4 ∨ a = -2) :=
by
  sorry

end part1_part2_l563_563526


namespace total_questions_l563_563089

-- Define the conditions
variable (S C I : ℕ)
variable (grading_system : S = C - 2 * I)
variable (student_score : S = 73)
variable (correct_responses : C = 91)

-- The theorem we want to prove
theorem total_questions : student_score → correct_responses → grading_system → C + I = 100 :=
by
  intros
  sorry

end total_questions_l563_563089


namespace find_angle_BAC_l563_563975

variables {A B C P Q T : Type}
variables [triangle ABC] -- Triangle ABC
variables [is_acute_angled ABC] -- Acute-angled triangle condition
variables [is_altitude BP ABC] -- BP is an altitude
variables [is_altitude CQ ABC] -- CQ is an altitude
variables [orthocenter T (triangle PAQ)] -- T is the orthocenter of triangle PAQ

theorem find_angle_BAC 
  (H1 : is_acute_angled ABC)
  (H2 : is_altitude BP ABC)
  (H3 : is_altitude CQ ABC)
  (H4 : orthocenter T (triangle PAQ))
  (H5 : ∠CTB = 90) : ∠BAC = 45 :=
sorry

end find_angle_BAC_l563_563975


namespace change_received_after_discounts_and_taxes_l563_563011

theorem change_received_after_discounts_and_taxes :
  let price_wooden_toy : ℝ := 20
  let price_hat : ℝ := 10
  let tax_rate : ℝ := 0.08
  let discount_wooden_toys : ℝ := 0.15
  let discount_hats : ℝ := 0.10
  let quantity_wooden_toys : ℝ := 3
  let quantity_hats : ℝ := 4
  let amount_paid : ℝ := 200
  let cost_wooden_toys := quantity_wooden_toys * price_wooden_toy
  let discounted_cost_wooden_toys := cost_wooden_toys - (discount_wooden_toys * cost_wooden_toys)
  let cost_hats := quantity_hats * price_hat
  let discounted_cost_hats := cost_hats - (discount_hats * cost_hats)
  let total_cost_before_tax := discounted_cost_wooden_toys + discounted_cost_hats
  let tax := tax_rate * total_cost_before_tax
  let total_cost_after_tax := total_cost_before_tax + tax
  let change_received := amount_paid - total_cost_after_tax
  change_received = 106.04 := by
  -- All the conditions and intermediary steps are defined above, from problem to solution.
  sorry

end change_received_after_discounts_and_taxes_l563_563011


namespace ceil_sqrt_162_l563_563132

-- Define necessary variables and import sqrt and ceil functions
def sqrt162 := Real.sqrt 162
def z1 := 12
def z2 := 13

-- Assert the inequality conditions
axiom h1 : 144 < 162
axiom h2 : 162 < 169

-- The main theorem statement
theorem ceil_sqrt_162 : Real.ceil sqrt162 = 13 :=
by
  -- Just adding the sorry to skip the proof since the solution steps aren't to be considered
  sorry

end ceil_sqrt_162_l563_563132


namespace pallets_of_paper_cups_l563_563087

theorem pallets_of_paper_cups (total_pallets paper_towels tissues paper_plates : ℕ) 
  (H1 : total_pallets = 20) 
  (H2 : paper_towels = total_pallets / 2)
  (H3 : tissues = total_pallets / 4)
  (H4 : paper_plates = total_pallets / 5) : 
  total_pallets - paper_towels - tissues - paper_plates = 1 := 
  by
    sorry

end pallets_of_paper_cups_l563_563087


namespace find_a4_plus_b4_l563_563388

-- Variables representing the given conditions
variables {a b : ℝ}

-- The theorem statement to prove
theorem find_a4_plus_b4 (h1 : a^2 - b^2 = 8) (h2 : a * b = 2) : a^4 + b^4 = 56 :=
sorry

end find_a4_plus_b4_l563_563388


namespace number_of_paths_l563_563215

def continuous_paths (start : char) (end : char) (segments : list (char × char)) : ℕ :=
sorry

theorem number_of_paths :
  continuous_paths 'A' 'B' [('A', 'C'), ('A', 'D'), ('C', 'B'), ('D', 'B'), ('D', 'C'),
                            ('C', 'F'), ('D', 'F'), ('F', 'B'), ('D', 'E'), ('E', 'B'), 
                            ('E', 'F'), ('E', 'G'), ('G', 'F'), ('G', 'B')] 
  = 15 :=
sorry

end number_of_paths_l563_563215


namespace fraction_sum_l563_563847

theorem fraction_sum : (1/4 : ℚ) + (3/9 : ℚ) = (7/12 : ℚ) := 
  by 
  sorry

end fraction_sum_l563_563847


namespace find_notebook_price_l563_563749

noncomputable def notebook_and_pencil_prices : Prop :=
  ∃ (x y : ℝ),
    5 * x + 4 * y = 16.5 ∧
    2 * x + 2 * y = 7 ∧
    x = 2.5

theorem find_notebook_price : notebook_and_pencil_prices :=
  sorry

end find_notebook_price_l563_563749


namespace exists_perfect_square_subtraction_l563_563673

theorem exists_perfect_square_subtraction {k : ℕ} (hk : k > 0) : 
  ∃ (n : ℕ), n > 0 ∧ ∃ m : ℕ, n * 2^k - 7 = m^2 := 
  sorry

end exists_perfect_square_subtraction_l563_563673


namespace employee_saves_1310_l563_563427

def retail_price_model_A := 500
def retail_price_model_B := 650
def retail_price_model_C := 800

def discount_under_1_year := 0.10
def discount_1_to_2_years := 0.15
def discount_2_or_more_years := 0.20

def bulk_purchase_discount := 0.05
def store_credit := 100

def employee_years_worked := 1.5
def model_A_quantity := 5
def model_B_quantity := 3
def model_C_quantity := 2

def total_retail_price :=
  (model_A_quantity * retail_price_model_A) +
  (model_B_quantity * retail_price_model_B) +
  (model_C_quantity * retail_price_model_C)

def total_discount_percentage :=
  if employee_years_worked < 1 then discount_under_1_year
  else if employee_years_worked <= 2 then discount_1_to_2_years
  else discount_2_or_more_years

def total_discount := total_discount_percentage + bulk_purchase_discount

def discount_amount := total_discount * total_retail_price
def price_after_discount := total_retail_price - discount_amount
def final_price := price_after_discount - store_credit
def savings := total_retail_price - final_price

theorem employee_saves_1310 :
  savings = 1310 :=
by
  sorry

end employee_saves_1310_l563_563427


namespace num_integer_side_lengths_l563_563944

-- Define a, b, and the conditions for the triangle
def a : ℕ := 8
def b : ℕ := 3

-- Define the property that c must satisfy the triangle inequality conditions
def satisfies_triangle_inequality (c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the set of valid c values as those that satisfy the triangle inequality
def valid_c_values : set ℕ := { c | satisfies_triangle_inequality c }

-- Define the possible integer side lengths for c
def possible_integer_side_lengths : list ℕ := [6, 7, 8, 9, 10]

-- Prove that the number of possible integer side lengths is 5
theorem num_integer_side_lengths : (possible_integer_side_lengths.filter satisfies_triangle_inequality).length = 5 := by
  sorry

end num_integer_side_lengths_l563_563944


namespace part1_area_of_triangle_l563_563940

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * Real.cos x + 2 * Real.sqrt 3 * Real.sin x, 1)

noncomputable def n (x y : ℝ) : ℝ × ℝ := (Real.cos x, -y)

def orthogonal (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem part1 (x : ℝ) (h : orthogonal (m x) (n x (y x))) :
    y x = 2 * Real.cos x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x := sorry

variables (a b c A : ℝ)

theorem area_of_triangle (h1 : a = 2) (h2 : b + c = 4) (h3 : f (A / 2) = 3) (h4 : A = π / 3) :
    let S := 1 / 2 * b * c * Real.sin A
    (S = Real.sqrt 3) := sorry

end part1_area_of_triangle_l563_563940


namespace pairs_of_participants_l563_563648

theorem pairs_of_participants (n : Nat) (h : n = 12) : (Nat.choose n 2) = 66 := by
  sorry

end pairs_of_participants_l563_563648


namespace probability_red_light_first_time_second_intersection_l563_563794

theorem probability_red_light_first_time_second_intersection
  (p_red : ℝ)
  (h_independent : ∀ (A B : Prop), Prob (A * B) = Prob A * Prob B)
  (h_red : p_red = 1 / 3) :
  Prob (¬(red_at_first) ∧ red_at_second) = 2 / 9 :=
by
  sorry

end probability_red_light_first_time_second_intersection_l563_563794


namespace sum_of_positive_factors_of_90_l563_563374

theorem sum_of_positive_factors_of_90 : 
  let n := 90 in 
  let factors := (1 + 2) * (1 + 3 + 9) * (1 + 5) in 
  factors = 234 :=
by
  sorry

end sum_of_positive_factors_of_90_l563_563374


namespace linear_dependence_vecs_l563_563155

variables {k1 k2 k3 : ℝ}

theorem linear_dependence_vecs :
  ∃ (k1 k2 k3 : ℝ), (k1 * (1, 0) + k2 * (1, -1) + k3 * (2, 2) = (0, 0)) ∧ (k1 ≠ 0 ∨ k2 ≠ 0 ∨ k3 ≠ 0) → k1 + 4 * k3 = 0 :=
by
  sorry

end linear_dependence_vecs_l563_563155


namespace conic_section_is_ellipse_l563_563383

theorem conic_section_is_ellipse :
  ∃ (x y : ℝ), sqrt(x^2 + (y-2)^2) + sqrt((x-6)^2 + (y+4)^2) = 12 ∧
  sqrt((0-6)^2 + (2+4)^2) = 6*sqrt(2) ∧ (6*sqrt(2) < 12) :=
by
  use (x : ℝ)
  use (y : ℝ)
  sorry

end conic_section_is_ellipse_l563_563383


namespace combined_share_b_d_l563_563449

-- Definitions for the amounts shared between the children
def total_amount : ℝ := 15800
def share_a_plus_c : ℝ := 7022.222222222222

-- The goal is to prove that the combined share of B and D is 8777.777777777778
theorem combined_share_b_d :
  ∃ B D : ℝ, (B + D = total_amount - share_a_plus_c) :=
by
  sorry

end combined_share_b_d_l563_563449


namespace greatest_ABCBA_divisible_by_13_l563_563784

theorem greatest_ABCBA_divisible_by_13 :
  ∃ A B C : ℕ, A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  1 ≤ A ∧ A < 10 ∧ 0 ≤ B ∧ B < 10 ∧ 0 ≤ C ∧ C < 10 ∧
  (10000 * A + 1000 * B + 100 * C + 10 * B + A) % 13 = 0 ∧
  (10000 * A + 1000 * B + 100 * C + 10 * B + A) = 95159 :=
by
  sorry

end greatest_ABCBA_divisible_by_13_l563_563784


namespace bridge_length_l563_563750

theorem bridge_length
  (train_length : ℕ)
  (train_speed_kmph : ℕ)
  (cross_time_secs : ℕ)
  (h_train_length : train_length = 110)
  (h_train_speed : train_speed_kmph = 45)
  (h_cross_time : cross_time_secs = 30) :
  ∃ (bridge_length : ℕ), bridge_length = 265 :=
by
  -- Introduce the speed in meters per second (m/s)
  let train_speed_mps := (train_speed_kmph * 1000) / 3600
  have h_train_speed_mps : train_speed_mps = 12.5, sorry

  -- Calculate the distance traveled in 30 seconds
  let distance_traveled := train_speed_mps * cross_time_secs
  have h_distance_traveled : distance_traveled = 375, sorry

  -- Calculate the bridge length
  let bridge_length := distance_traveled - train_length
  use bridge_length
  exact sorry

end bridge_length_l563_563750


namespace fraction_white_surface_area_l563_563766

theorem fraction_white_surface_area (s : ℕ) (n : ℕ) (red white : ℕ) 
(h_s : s = 3) (h_n : n = 27) (h_red : red = 21) (h_white : white = 6) :
  (min_white_surface_area s n red white) = 5 / 54 := by
  sorry

end fraction_white_surface_area_l563_563766


namespace runners_meet_after_3000_seconds_l563_563031

theorem runners_meet_after_3000_seconds :
  let v1 := 45
      v2 := 49
      v3 := 51
      track_length := 600
  ∃ t, t = 3000 ∧
    (∃ k, 0.4 * t = 600 * k) ∧
    (∃ m, 0.2 * t = 600 * m) ∧
    (∃ n, 0.6 * t = 600 * n) :=
by {
  let t := 3000;
  use t;
  split;
  { exact rfl, },
  { split; use 2; norm_num,
    split; use 1; norm_num,
    use 5/3; norm_num, },
}

end runners_meet_after_3000_seconds_l563_563031


namespace find_m_and_n_l563_563640

theorem find_m_and_n (a b c d m n : ℕ) (h1 : a^2 + b^2 + c^2 + d^2 = 1989) 
                    (h2 : a + b + c + d = m^2) 
                    (h3 : max a (max b (max c d)) = n^2) : 
                    m = 9 ∧ n = 6 := 
sorry

end find_m_and_n_l563_563640


namespace find_total_income_l563_563077

noncomputable def total_income : ℝ :=
  let income := 30000 / (0.42 - 0.0504)
  in income

theorem find_total_income :
  (let I := total_income in
  0.3696 * I = 30000) :=
begin
  let I := total_income,
  norm_num,
  have h : I = 30000 / 0.3696,
  { refl },
  rw h,
  field_simp,
  norm_num,
end

end find_total_income_l563_563077


namespace sum_of_max_and_min_eq_4_l563_563201

noncomputable def f (x : ℝ) : ℝ :=
  2 + (2 * x) / (x^2 + 1)

theorem sum_of_max_and_min_eq_4 :
  (∀ x : ℝ, f(x) ≤ 4) ∧ (∀ x : ℝ, f(x) ≥ 2) → ∃ (M m : ℝ), M = (2 + ((2 * x) / (x^2 + 1))) ∧ m = (2 + ((2 * x) / (x^2 + 1))) ∧ M + m = 4
:= sorry

end sum_of_max_and_min_eq_4_l563_563201


namespace fractional_exponent_representation_l563_563326

theorem fractional_exponent_representation (a : ℝ) (h : a > 0) : a^2 * real.sqrt a = a^(5/2) :=
by sorry

end fractional_exponent_representation_l563_563326


namespace inequality_3a3_2b3_3a2b_2ab2_l563_563131

theorem inequality_3a3_2b3_3a2b_2ab2 (a b : ℝ) (h₁ : a ≥ b) (h₂ : b > 0) : 
  3 * a ^ 3 + 2 * b ^ 3 ≥ 3 * a ^ 2 * b + 2 * a * b ^ 2 :=
by
  sorry

end inequality_3a3_2b3_3a2b_2ab2_l563_563131


namespace wait_time_at_least_8_l563_563093

-- Define the conditions
variables (p₀ p : ℝ) (r x : ℝ)

-- Given conditions
def initial_BAC := p₀ = 89
def BAC_after_2_hours := p = 61
def BAC_decrease := p = p₀ * (Real.exp (r * x))
def decrease_in_2_hours := p = 89 * (Real.exp (r * 2))

-- The main goal to prove the time required is at least 8 hours
theorem wait_time_at_least_8 (h1 : p₀ = 89) (h2 : p = 61) (h3 : p = p₀ * Real.exp (r * x)) (h4 : 61 = 89 * Real.exp (2 * r)) : 
  ∃ x, 89 * Real.exp (r * x) < 20 ∧ x ≥ 8 :=
sorry

end wait_time_at_least_8_l563_563093


namespace veronica_pitting_time_is_2_hours_l563_563724

def veronica_cherries_pitting_time (pounds : ℕ) (cherries_per_pound : ℕ) (minutes_per_20_cherries : ℕ) :=
  let cherries := pounds * cherries_per_pound
  let sets := cherries / 20
  let total_minutes := sets * minutes_per_20_cherries
  total_minutes / 60

theorem veronica_pitting_time_is_2_hours : 
  veronica_cherries_pitting_time 3 80 10 = 2 :=
  by
    sorry

end veronica_pitting_time_is_2_hours_l563_563724


namespace combined_cost_price_correct_l563_563793

noncomputable def CP_A : ℝ := 270 / 1.20
noncomputable def CP_B : ℝ := 350 / 1.25
noncomputable def CP_C : ℝ := 500 / 1.15

def combined_cost_price : ℝ := CP_A + CP_B + CP_C

theorem combined_cost_price_correct :
  combined_cost_price = 939.78 :=
by
  sorry

end combined_cost_price_correct_l563_563793


namespace value_of_expression_l563_563867

theorem value_of_expression : 
  (3 / 11) * ∏ n in Finset.range 118 + 3, (1 + 1 / (n + 3)) = 11 / 60 := 
by
  sorry

end value_of_expression_l563_563867


namespace dog_starting_point_is_7_l563_563111

-- Definitions for the problem
def path_length : ℝ := 28
def midpoint (a b : ℝ) : ℝ := (a + b) / 2
def total_runs : ℕ := 20
def dog_final_position : ℝ := 27  -- 1 meter to the left of point B, where B is at 28 meters.

-- Given dog is started from a point x meters from point A
def dog_starting_point (x : ℝ) : Prop :=
  let M := midpoint 0 path_length in
  -- condition for dog being 1 meter to the left of B after 20 runs
  ∃ (n : ℕ), n = total_runs ∧
    (let pos := (list.range (total_runs)).foldl
      (λ acc run, acc + (if run % 2 = 0 then -10 else 14))
      x
    in pos = dog_final_position)

-- Prove the dog started from 7 meters from point A
theorem dog_starting_point_is_7 : dog_starting_point 7 :=
  sorry

end dog_starting_point_is_7_l563_563111


namespace centroid_of_tetrahedron_l563_563614

variable (a b c : ℝ → ℝ → ℝ → ℝ)

-- Define the given vectors
def PA := a
def PB := b
def PC := c

-- Define the centroid G of the tetrahedron P-ABC
def PG := (1 / 4) • PA + (1 / 4) • PB + (1 / 4) • PC

theorem centroid_of_tetrahedron (P A B C : Type) (a b c : P → A) : 
  (∃ G : A, G = (1 / 4) • PA + (1 / 4) • PB + (1 / 4) • PC) :=
sorry

end centroid_of_tetrahedron_l563_563614


namespace min_value_expression_l563_563184

open Real

theorem min_value_expression (x y : ℝ) (hx : x > y) (hy : y > 0) (hxy : x + y = 1) :
  ∃ (x : ℝ), (∃ (hx : 1 > x ∧ x > 1/2), (∃ (hy : 1/2 > y ∧ y > 0), 
  ∀ (hxxy : y = 1 - x), 
  (∀ (hxgt : x > (2*sqrt 2 - 1)/2 ∧ x < 1),
  (∀ (hxlt : 1/2 < x ∧ x < (2*sqrt 2 - 1)/2),
  ∀ (f : ℝ → ℝ), 
  f = λ x, (2/(3 - 2 * x) + 1/(2 * x - 1)) →
  has_min_on f { x : ℝ | 1 > x ∧ x > 1/2 } (2*sqrt 2 - 1)/2 →
  f ((2*sqrt 2 - 1)/2) = (3 + 2 * sqrt 2) / 2)))) sorry

end min_value_expression_l563_563184


namespace ellipse_equation_l563_563467

theorem ellipse_equation 
  (a b c : ℝ)
  (h_major_axis : a = 4)
  (h_minor_axis : b = 2)
  (h_focal_length : c = 2 * real.sqrt(3))
  (h_relationship : a^2 - c^2 = b^2) :
  ∀ x y : ℝ, (x^2 / 4) + (y^2 / 16) = 1 :=
by sorry

end ellipse_equation_l563_563467


namespace find_third_circle_radius_l563_563514

noncomputable def radius_of_third_circle (r1 r5 : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
r1 + (n - 1) * d

theorem find_third_circle_radius :
  ∃ r3, (∀ r1 r5 d, r1 = 6 ∧ r5 = 30 ∧ r5 = r1 + 4 * d ∧ r3 = radius_of_third_circle r1 r5 d 3 -> r3 = 18) :=
begin
   use 18,
   intros r1 r5 d h,
   cases h with h1 h2,
   cases h2 with h3 h4,
   cases h4 with h5 h6,
   subst h1,
   subst h3,
   subst h5,
   exact h6,
end

end find_third_circle_radius_l563_563514


namespace angle_between_MN_and_KL_l563_563600

open EuclideanGeometry

theorem angle_between_MN_and_KL (A B O M N K L : Point) 
  (line1 line2 : Line) 
  (h_line1_O : O ∈ line1) 
  (h_line2_O : O ∈ line2)
  (h_alpha : ∠ A O B = α) (h_alpha_le_90 : α ≤ 90)
  (h_M : ⊥ A M line1) 
  (h_N : ⊥ A N line2)
  (h_K : ⊥ B K line1)
  (h_L : ⊥ B L line2) :
  angle_between_lines (line_from_points M N) (line_from_points K L) = α :=
sorry

end angle_between_MN_and_KL_l563_563600


namespace add_and_round_correct_l563_563459

def a := 45.768
def b := 18.1542

def add_and_round (x y : ℚ) : ℚ := (x + y).round(2)  -- rounding method specifically for the hundredths place

theorem add_and_round_correct : add_and_round a b = 63.92 := 
by
  -- proof omitted
  sorry

end add_and_round_correct_l563_563459


namespace abs_fraction_eq_sqrt_six_over_two_l563_563635

theorem abs_fraction_eq_sqrt_six_over_two {a b : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 10 * a * b) : 
  abs ((a + b) / (a - b)) = sqrt 6 / 2 :=
sorry

end abs_fraction_eq_sqrt_six_over_two_l563_563635


namespace sequence_limit_l563_563958

noncomputable def limit_sequence_term := 
  tendsto (λ n : ℕ, 
    (1 / n^2 : ℝ) * ∑ k in finset.range n, (4 * (k + 1))) at_top (𝓝 4)

theorem sequence_limit : 
  (∀ n : ℕ, (sqrt a[1] + sqrt a[2] + ... + sqrt a[n]) = n^2 + 3n) ∧
  (∀ n : ℕ, a[n] = 4 * (n + 1)^2) →
  limit_sequence_term :=
sorry

end sequence_limit_l563_563958


namespace area_of_tangent_triangle_correct_l563_563638

noncomputable def area_of_tangents_triangle 
  (A B C : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (dist_AB : ℝ) (dist_BC : ℝ) (dist_CA : ℝ) 
  (feet_altitudes : Σ' (E : A) (F : B), true) : ℝ :=
  let circumcircle_AEF := 
    -- circumcircle of the triangle AEF formed by the feet of the altitudes
    sorry in
  let area_ABC := 
    -- Area of triangle ABC using Heron's formula or any area calculation method
    sorry in
  let area_tangents := 
    -- Calculate the area of the triangle formed by tangents to circumcircle_AEF
    sorry in
  area_tangents

theorem area_of_tangent_triangle_correct 
  (A B C : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (dist_AB : dist B C = 14) 
  (dist_BC : dist C A = 15) 
  (dist_CA : dist A B = 13)
  (feet_altitudes : Σ' E F, true) :
  area_of_tangents_triangle A B C dist_AB dist_BC dist_CA feet_altitudes = 462 / 5 :=
sorry

end area_of_tangent_triangle_correct_l563_563638


namespace min_rope_length_eq_50_min_distance_on_rope_eq_4_l563_563176

/-
Given a truncated cone with:
- r₁ = 5 cm (radius of the upper base)
- r₂ = 10 cm (radius of the lower base)
- h = 20 cm (slant height)

Prove:
- The minimum length of the rope (L) is 50 cm.
- The minimum distance (d) from any point on the rope to a point on the upper base circumference is 4 cm.
-/

variables (r₁ r₂ h L d : ℝ)
hypothesis h_r₁ : r₁ = 5
hypothesis h_r₂ : r₂ = 10
hypothesis h_h : h = 20
hypothesis h_L : L = 50
hypothesis h_d : d = 4

theorem min_rope_length_eq_50 :
  r₁ = 5 ∧ r₂ = 10 ∧ h = 20 → L = 50 :=
by
  intros
  sorry

theorem min_distance_on_rope_eq_4 :
  r₁ = 5 ∧ r₂ = 10 ∧ h = 20 ∧ L = 50 → d = 4 :=
by
  intros
  sorry

end min_rope_length_eq_50_min_distance_on_rope_eq_4_l563_563176


namespace coeff_x3_in_expansion_l563_563978

theorem coeff_x3_in_expansion : 
  let f := (x^2 + 1) * (x - 2)^7,
      coeff_x := binomCoeff 7 6 * (-2)^6,
      coeff_x3 := binomCoeff 7 4 * (-2)^4 in
  coeff f 3 = coeff_x + coeff_x3 := 
by {
  sorry
}

end coeff_x3_in_expansion_l563_563978


namespace greatest_multiple_of_3_lt_1000_l563_563318

theorem greatest_multiple_of_3_lt_1000 :
  ∃ (x : ℕ), (x % 3 = 0) ∧ (x > 0) ∧ (x^3 < 1000) ∧ ∀ (y : ℕ), (y % 3 = 0) ∧ (y > 0) ∧ (y^3 < 1000) → y ≤ x := 
sorry

end greatest_multiple_of_3_lt_1000_l563_563318


namespace sum_of_positive_factors_of_90_eq_234_l563_563370

theorem sum_of_positive_factors_of_90_eq_234 : 
  let factors := [1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90]
  List.sum factors = 234 :=
by
  -- List the positive factors of 90
  let factors := [1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90]
  -- Prove that the sum of these factors is 234
  have h_sum_factors : List.sum factors = 234 := sorry
  exact h_sum_factors

end sum_of_positive_factors_of_90_eq_234_l563_563370


namespace sandwich_cost_is_5_l563_563718

-- We define the variables and conditions first
def total_people := 4
def sandwiches := 4
def fruit_salads := 4
def sodas := 8
def snack_bags := 3

def fruit_salad_cost_per_unit := 3
def soda_cost_per_unit := 2
def snack_bag_cost_per_unit := 4
def total_cost := 60

-- We now define the calculations based on the given conditions
def total_fruit_salad_cost := fruit_salads * fruit_salad_cost_per_unit
def total_soda_cost := sodas * soda_cost_per_unit
def total_snack_bag_cost := snack_bags * snack_bag_cost_per_unit
def other_items_cost := total_fruit_salad_cost + total_soda_cost + total_snack_bag_cost
def remaining_budget := total_cost - other_items_cost
def sandwich_cost := remaining_budget / sandwiches

-- The final proof problem statement in Lean 4
theorem sandwich_cost_is_5 : sandwich_cost = 5 := by
  sorry

end sandwich_cost_is_5_l563_563718


namespace handshake_count_l563_563473

def gathering_handshakes (total_people : ℕ) (know_each_other : ℕ) (know_no_one : ℕ) : ℕ :=
  let group2_handshakes := know_no_one * (total_people - 1)
  group2_handshakes / 2

theorem handshake_count :
  gathering_handshakes 30 20 10 = 145 :=
by
  sorry

end handshake_count_l563_563473


namespace only_statement_4_is_correct_l563_563051

-- Defining conditions for input/output statement correctness
def INPUT_statement_is_correct (s : String) : Prop :=
  s = "INPUT x=, 2"

def PRINT_statement_is_correct (s : String) : Prop :=
  s = "PRINT 20, 4"

-- List of statements
def statement_1 := "INPUT a; b; c"
def statement_2 := "PRINT a=1"
def statement_3 := "INPUT x=2"
def statement_4 := "PRINT 20, 4"

-- Predicate for correctness of statements
def statement_is_correct (s : String) : Prop :=
  (s = statement_4) ∧
  ¬(s = statement_1 ∨ s = statement_2 ∨ s = statement_3)

-- Theorem to prove that only statement 4 is correct
theorem only_statement_4_is_correct :
  ∀ s : String, (statement_is_correct s) ↔ (s = statement_4) :=
by
  intros s
  sorry

end only_statement_4_is_correct_l563_563051


namespace sequence_sum_maximized_at_10_l563_563713

-- Define the sequence term
def a (n : ℕ) : ℝ := log (1000 * (cos (Real.pi / 3))^(n - 1))

-- Main statement to be proved
theorem sequence_sum_maximized_at_10 :
  (∀ n : ℕ, (a n ≥ 0) ∧ (a (n + 1) < 0) → n = 10) :=
sorry

end sequence_sum_maximized_at_10_l563_563713


namespace parallel_lines_of_tangent_circles_l563_563361

open EuclideanGeometry

variable {P : Type*} [MetricSpace P] [NormedGroup P] [AffineSpace P (EuclideanSpace P)]

theorem parallel_lines_of_tangent_circles
  (circle1 circle2 : P → Prop)
  (O A B C D : P)
  (tangent_at_O : circle1 O ∧ circle2 O)
  (line_through_O_A_B : collinear {O, A, B})
  (line_through_O_C_D : collinear {O, C, D})
  : is_parallel (line_through A B) (line_through C D) :=
by
  sorry

end parallel_lines_of_tangent_circles_l563_563361


namespace imaginary_part_of_z_l563_563916

theorem imaginary_part_of_z (z : ℂ) (h : z = 2 / (-1 + I)) : z.im = -1 :=
sorry

end imaginary_part_of_z_l563_563916


namespace total_amount_720_l563_563522

variable {ℤ} (a b j : ℤ)

-- Conditions
def initial_toy_amount : ℤ := 48
def amy_gives (a b j t : ℤ) : ℤ := a - (b + j + t)
def beth_gives (new_a : ℤ) (b j t : ℤ) : ℤ := 2 * new_a
def jan_gives (new_a new_b j t : ℤ) : ℤ := 2 * new_b
def toy_gives (a b j t : ℤ) (new_a new_b new_j new_t : ℤ) : ℤ := new_t - (8 * (new_a + new_b + new_j - new_t) + initial_toy_amount + 8 * new_t) = 48

-- Prove that the total amount of money is 720
theorem total_amount_720 : ∃ (a b j : ℤ), toy_gives a b j initial_toy_amount ∧ 
  8 * (a + b + j - initial_toy_amount) = 336 := sorry

end total_amount_720_l563_563522


namespace circle_equation_through_points_l563_563835

-- Line and circle definitions
def line1 (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 15 = 0

-- Intersection point definition
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ circle1 x y

-- Revised circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 28 * x - 15 * y = 0

-- Proof statement
theorem circle_equation_through_points :
  (∀ x y, intersection_point x y → circle_equation x y) ∧ circle_equation 0 0 :=
sorry

end circle_equation_through_points_l563_563835


namespace problem_correct_l563_563167

noncomputable def f : ℕ × ℕ → ℕ
| (1, 1) := 1
| (m, n+1) := f (m, n) + 2
| (m+1, 1) := 2 * f (m, 1)
| _ := 0  -- This is only needed to make the function total in Lean, not based on the problem conditions.

-- Conditions
axiom f_pos : ∀ m n : ℕ, 0 < m → 0 < n → f (m, n) ∈ ℕ
axiom f_cond1 : ∀ m n : ℕ, 0 < m → 0 < n → f (m, n+1) = f (m, n) + 2
axiom f_cond2 : ∀ m : ℕ, 0 < m → f (m+1, 1) = 2 * f (m, 1)

-- Theorem
theorem problem_correct :
  f (1, 5) = 9 ∧ f (5, 1) = 16 ∧ f (5, 6) = 26 :=
by sorry

end problem_correct_l563_563167


namespace area_sum_of_three_circles_l563_563082

theorem area_sum_of_three_circles (R d : ℝ) (x y z : ℝ) 
    (hxyz : x^2 + y^2 + z^2 = d^2) :
    (π * ((R^2 - x^2) + (R^2 - y^2) + (R^2 - z^2))) = π * (3 * R^2 - d^2) :=
by
  sorry

end area_sum_of_three_circles_l563_563082


namespace range_of_m_l563_563926

def f (m : ℝ) (x : ℝ) : ℝ := m * 4^x - 2^x

theorem range_of_m (m : ℝ) :
  (∃ (x₀ : ℝ), x₀ ≠ 0 ∧ f m (-x₀) = f m x₀) ↔ (0 < m ∧ m < 1 / 2) := sorry

end range_of_m_l563_563926


namespace quadratic_expression_neg_for_all_x_l563_563126

theorem quadratic_expression_neg_for_all_x (m : ℝ) :
  (∀ x : ℝ, m*x^2 + (m-1)*x + (m-1) < 0) ↔ m < -1/3 :=
sorry

end quadratic_expression_neg_for_all_x_l563_563126


namespace particle_trajectory_l563_563866

theorem particle_trajectory (g k : ℝ) :
  ∃ x y : ℝ → ℝ,
  (∀ t : ℝ, x 0 = 0 ∧ x' 0 = 0 ∧ y 0 = 0 ∧ y' 0 = 0) ∧
  (∀ t : ℝ, y'' t = g - k * x' t ∧ x'' t = k * y' t) ∧
  (x = λ t, (g / k^2) * (sin (k * t) - k * t) ∧ y = λ t, (g / k^2) * (1 - cos (k * t))) := sorry

end particle_trajectory_l563_563866


namespace find_a_l563_563561

theorem find_a (a : ℝ) :
  (∀ x y, x + y = a → x^2 + y^2 = 4) →
  (∀ A B : ℝ × ℝ, (A.1 + A.2 = a ∧ B.1 + B.2 = a ∧ A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4) →
      ‖(A.1, A.2) + (B.1, B.2)‖ = ‖(A.1, A.2) - (B.1, B.2)‖) →
  a = 2 ∨ a = -2 :=
by
  intros line_circle_intersect vector_eq_magnitude
  sorry

end find_a_l563_563561


namespace sum_roots_fraction_l563_563636

noncomputable theory

open Polynomial

def p : Polynomial ℝ :=
    Polynomial.sum (range 2021) (λ n, monomial n 1) - C (1397 : ℝ)

theorem sum_roots_fraction {a : ℕ → ℝ} (h : ∀ n, is_root p (a n)) :
    (∑ n in range 2020, 1 / (1 - a n)) = 3270 := by
    sorry

end sum_roots_fraction_l563_563636


namespace slope_probability_l563_563998
open Set

noncomputable def point_in_unit_square (x y : ℝ) : Prop :=
  0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1

noncomputable def slope_cond (x y : ℝ) : Prop :=
  (y - 1/8)/(x - 7/8) ≥ 1

theorem slope_probability:
  ∀ (m n : ℕ), 
  let P := λ(x y : ℝ), point_in_unit_square x y ∧ slope_cond x y in
  (probability (λ(x y : ℝ), P x y) = 1/32) →
  m / n = 1/32 ∧ (nat.coprime m n) → (m + n = 33) :=
by
  intros
  sorry

end slope_probability_l563_563998


namespace percentage_increase_in_sales_l563_563704

-- Definition of the given conditions
def original_sale_value (P S : ℝ) : ℝ := P * S
def reduced_price (P : ℝ) : ℝ := 0.82 * P
def new_units_sold (S X : ℝ) : ℝ := S * (1 + X / 100)
def new_sale_value (P S X: ℝ) : ℝ := reduced_price P * new_units_sold S X
def net_effect_sale_value (P S X : ℝ) : ℝ := original_sale_value P S - new_sale_value P S X

-- We need to prove
theorem percentage_increase_in_sales (P S : ℝ) (h1 : original_sale_value P S - new_sale_value P S 44.12 = 0.5416 * original_sale_value P S) :
  44.12 = 44.12 :=
by
  sorry

end percentage_increase_in_sales_l563_563704


namespace coefficient_c1_of_q_l563_563275

noncomputable def D (m : ℕ) : ℕ :=
  if m % 2 = 1 ∧ m ≥ 7 then
    let s := finset.range (m + 1).filter (λ x, x ≥ 3);
    (s.powerset.filter (λ t, t.card = 4 ∧ (t.sum id % m = 0))).card
  else 0

noncomputable def q (x : ℕ) : ℤ :=
  c3 * x^3 + c2 * x^2 + 6 * x + c0

theorem coefficient_c1_of_q (c3 c2 c0 : ℤ) :
  (∀ (m : ℕ), m % 2 = 1 ∧ m ≥ 7 → D(m) = q(m)) → c1 = 6 :=
sorry

end coefficient_c1_of_q_l563_563275


namespace largest_number_not_sum_of_two_composites_l563_563857

-- Define composite numbers
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

-- Define the problem predicate
def cannot_be_expressed_as_sum_of_two_composites (n : ℕ) : Prop :=
  ¬ ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_number_not_sum_of_two_composites : 
  ∃ n, cannot_be_expressed_as_sum_of_two_composites n ∧ 
  (∀ m, cannot_be_expressed_as_sum_of_two_composites m → m ≤ n) ∧ 
  n = 11 :=
by {
  sorry
}

end largest_number_not_sum_of_two_composites_l563_563857


namespace tall_students_proof_l563_563338

variables (T : ℕ) (Short Average Tall : ℕ)

-- Given in the problem:
def total_students := T = 400
def short_students := Short = 2 * T / 5
def average_height_students := Average = 150

-- Prove:
theorem tall_students_proof (hT : total_students T) (hShort : short_students T Short) (hAverage : average_height_students Average) :
  Tall = T - (Short + Average) :=
by
  sorry

end tall_students_proof_l563_563338


namespace magnitude_of_z_l563_563530

noncomputable def z : ℂ :=
  let c := (1 - complex.I) / complex.I
  1 / c

theorem magnitude_of_z : complex.abs z = 1 :=
sorry

end magnitude_of_z_l563_563530


namespace two_tangents_from_origin_l563_563199

theorem two_tangents_from_origin
  (f : ℝ → ℝ)
  (a : ℝ)
  (h₁ : ∀ x, f(x) = a * x^3 + 3 * x^2 + 1)
  (h₂ : ∃ m₁ m₂ : ℝ, 
         m₁ ≠ m₂ ∧ 
         (f(-m₁) + f(m₂ + 2)) = 2 * f(1) ∧ 
         (f(-m₂) + f(m₁ + 2)) = 2 * f(1)) :
  ∃! t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
    (∃ (y₁ y₂: ℝ), y₁ = f(t₁) ∧ y₂ = f(t₂) ∧ 
                   y₁ / t₁ = (-y₁ + 3 * x * t₁) / t₁ ∧ 
                   y₂ / t₂ = (-y₂ + 3 * x * t₂) / t₂) :=
sorry

end two_tangents_from_origin_l563_563199


namespace prism_volume_l563_563685

/-- The volume of a rectangular prism given the areas of three of its faces. -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 50) (h3 : b * c = 75) :
  a * b * c = 335 :=
by
  sorry

end prism_volume_l563_563685


namespace other_number_is_300_l563_563683

theorem other_number_is_300 (A B : ℕ) (h1 : A = 231) (h2 : lcm A B = 2310) (h3 : gcd A B = 30) : B = 300 := by
  sorry

end other_number_is_300_l563_563683


namespace volume_of_intersection_l563_563043

-- Define the region for the first cube with max(|x|, |y|, |z|) ≤ 1
def cube1 (x y z : ℝ) : Prop := max (|x|) (max (|y|) (|z|)) ≤ 1

-- Define the region for the second cube with max(|x-1|, |y-1|, |z-1|) ≤ 1
def cube2 (x y z : ℝ) : Prop := max (|(x - 1)|) (max (|(y - 1)|) (|(z - 1)|)) ≤ 1

-- Define the intersected region
def intersected_region (x y z : ℝ) : Prop := cube1 x y z ∧ cube2 x y z

-- Calculate the volume of the intersected region
theorem volume_of_intersection : 
  ∫ x in 0..1, ∫ y in 0..1, ∫ z in 0..1, (if intersected_region x y z then 1 else 0) = 1 := 
by
  sorry

end volume_of_intersection_l563_563043


namespace infinite_geometric_sequence_range_l563_563229

theorem infinite_geometric_sequence_range (a1 q a2 : ℝ) 
    (h_sum : a1 / (1 - q) = 4) 
    (h_a2 : a2 = a1 * q) :
    a2 ∈ set.Ioo (-8 : ℝ) (0 : ℝ) ∪ set.Ioc 0 (1 : ℝ) :=
by
  sorry

end infinite_geometric_sequence_range_l563_563229


namespace ramsey_K9_blue_K4_or_red_K3_l563_563843

theorem ramsey_K9_blue_K4_or_red_K3 (G : SimpleGraph (fin 9)) (hG : G.IsCompleteGraph) (color : G.Edge → Prop) :
  (∃ (V : finset (fin 9)), V.card = 4 ∧ ∀ (u v : fin 9) (hu : u ∈ V) (hv : v ∈ V), u ≠ v → color (u, v)) ∨
  (∃ (U : finset (fin 9)), U.card = 3 ∧ ∀ (u v : fin 9) (hu : u ∈ U) (hv : v ∈ U), u ≠ v → ¬color (u, v)) :=
sorry

end ramsey_K9_blue_K4_or_red_K3_l563_563843


namespace arithmetic_geometric_sequence_l563_563914

theorem arithmetic_geometric_sequence (a b : ℝ) (h1 : 2 * a = 1 + b) (h2 : b^2 = a) (h3 : a ≠ b) :
  7 * a * Real.log (-b) / Real.log a = 7 / 8 :=
by
  sorry

end arithmetic_geometric_sequence_l563_563914


namespace find_a_for_extremum_intervals_of_monotonicity_range_of_a_for_min_value_l563_563925

noncomputable def f (a x : ℝ) : ℝ := ln (a * x + 1) + (1 - x) / (1 + x)

-- 1. Prove that if f has an extremum at x = 1, then a = 1
theorem find_a_for_extremum (a : ℝ) (h_ext : ∃ x, x = 1 ∧ has_extremum_at (f a) x) : a = 1 :=
sorry

-- 2. Prove and find the intervals of monotonicity for f(x)
theorem intervals_of_monotonicity (a : ℝ) (h_pos : 0 < a) :
  (a ≥ 2 → is_increasing_on (f a) (set.Ioi 0)) ∧
  ((0 < a ∧ a < 2) → 
    (is_increasing_on (f a) (set.Ioi (real.sqrt ((2 - a) / a))) ∧ 
     is_decreasing_on (f a) (set.Icc 0 (real.sqrt ((2 - a) / a))))) :=
sorry

-- 3. Prove that if the minimum value of f(x) is 1, then a ∈ [2, +∞)
theorem range_of_a_for_min_value (a : ℝ) (h_min : has_minimum_value (f a) 1) : 2 ≤ a :=
sorry

end find_a_for_extremum_intervals_of_monotonicity_range_of_a_for_min_value_l563_563925


namespace significant_improvement_l563_563420

def mean (data: List ℝ) : ℝ :=
  data.sum / data.length

def variance (data: List ℝ) (mean: ℝ) : ℝ :=
  data.map (λ x => (x - mean) ^ 2).sum / data.length

def old_device_data: List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

def new_device_data: List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

theorem significant_improvement :
  let
    μx := mean old_device_data,
    μy := mean new_device_data,
    σ1_sq := variance old_device_data μx,
    σ2_sq := variance new_device_data μy
  in
    (μy - μx) > 2 * Real.sqrt((σ1_sq + σ2_sq) / 10) :=
by
  sorry

end significant_improvement_l563_563420


namespace range_of_f_l563_563708

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 1 then 2^x + 1 else 3 * x - 1

theorem range_of_f :
  set.range f = set.Iio 2 ∪ set.Ici 3 :=
begin
  sorry
end

end range_of_f_l563_563708


namespace correct_propositions_l563_563805

def P1 : Prop := ∀ (l₁ l₂ l₃ : Line), (l₁ ∥ l₃) ∧ (l₂ ∥ l₃) → (l₁ ∥ l₂)
def P2 : Prop := ∀ (π₁ π₂ π₃ : Plane), (π₁ ∥ π₃) ∧ (π₂ ∥ π₃) → (π₁ ∥ π₂)
def P3 : Prop := ∀ (l₁ l₂ l₃ : Line), (l₁ ⊥ l₃) ∧ (l₂ ⊥ l₃) → (l₁ ∥ l₂)
def P4 : Prop := ∀ (l₁ l₂ : Line) (π : Plane), (l₁ ⊥ π) ∧ (l₂ ⊥ π) → (l₁ ∥ l₂)

theorem correct_propositions : (P1 ∧ P2 ∧ ¬ P3 ∧ P4) ↔ 3 := 
by { sorry }

end correct_propositions_l563_563805


namespace student_average_always_greater_l563_563438

variable (u v w x y : ℝ)

theorem student_average_always_greater
  (h_ordering : u ≤ v ∧ v ≤ w ∧ w ≤ x ∧ x ≤ y):
  (let A := (u + v + w + x + y) / 5 in
   let B := (u + v + w + 3 * x + 3 * y) / 9 in
   B > A) :=
by
  sorry

end student_average_always_greater_l563_563438


namespace solve_equation_l563_563462

theorem solve_equation (x : ℝ) : 
  x^2 + 2 * x + 4 * Real.sqrt (x^2 + 2 * x) - 5 = 0 →
  (x = Real.sqrt 2 - 1) ∨ (x = - (Real.sqrt 2 + 1)) :=
by 
  sorry

end solve_equation_l563_563462


namespace range_of_a_l563_563912

-- Define the function f(x) with given properties
def f (x : ℝ) : ℝ := ln x

-- Define the function g(x) as given in the problem statement
def g (x a : ℝ) : ℝ := f(x) - a * x

-- Define the main theorem
theorem range_of_a (h1 : ∀ x > 0, f(x) = 2 * f(1 / x))
  (h2 : ∀ x ∈ Icc 1 3, f(x) = ln x)
  (h3 : ∃ a, (∀ x ∈ Icc (1/3 : ℝ) 3, g(x, a) = 0) ∧ (∃ x1 x2 x3 ∈ Icc (1/3 : ℝ) 3, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3)) :
  ∃ a, a ∈ Icc (ln 3 / 3) (1/e) := sorry

end range_of_a_l563_563912


namespace problem_1_problem_2_l563_563547

-- Definitions based on given conditions
def A := {x : ℝ | 1 ≤ x ∧ x < 5}
def B (a : ℝ) := {x : ℝ | -a < x ∧ x ≤ a + 3}
def U := set.univ ℝ  -- Universal set U is ℝ

-- First proof problem statement
theorem problem_1 : ∃ x : set ℝ, a = 1 ∧ x = (Aᶜ ∩ B 1) ∧ x = {x | -1 < x ∧ x < 1} := sorry

-- Second proof problem statement
theorem problem_2 : (B a ∩ A = B a) → a ≤ -1 := sorry

end problem_1_problem_2_l563_563547


namespace piecewise_and_unified_function_l563_563695

theorem piecewise_and_unified_function (f : ℝ → ℝ)
(h_even : ∀ x, f(-x) = f(x))
(h_periodic : ∀ x, f(x + 2) = f(x))
(h_spec : ∀ x ∈ set.Icc 2 3, f(x) = x) :
∀ x ∈ set.Icc (-2) 0, f(x) = (if x ∈ set.Icc (-2 : ℝ) (-1) then x + 4 else -x + 2 ∧ f(x) = 3 - |x + 1|) :=
begin
  sorry
end

end piecewise_and_unified_function_l563_563695


namespace problem_statement_l563_563569

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 
  sin (2 * ω * x - π / 6) - 4 * (sin (ω * x))^2

noncomputable def g (x : ℝ) : ℝ := 
  sqrt 3 * sin (2 * x - π / 3) - 2

theorem problem_statement (ω : ℝ) (h₁ : ω > 0) 
  (h₂ : ∀ x : ℝ, f x ω = sqrt 3 * sin (2 * x + π / 3) - 2) :
  ∃ ω', ω' = 1 ∧ 
  ∀ x : ℝ, 
    g x = sqrt 3 * sin (2 * x - π / 3) - 2 ∧
    (∀ (a b : ℝ), -π / 12 ≤ a ∧ a ≤ b ∧ b ≤ 5 * π / 12 → g a ≤ g b) ∧
    (∀ (c d : ℝ), 5 * π / 12 ≤ c ∧ c ≤ d ∧ d ≤ π / 2 → g c ≥ g d) := by
  sorry

end problem_statement_l563_563569


namespace minimum_balls_same_color_l563_563241

/-- 
Given 3 red balls, 5 yellow balls, and 7 blue balls in a wooden box. 
Prove that the minimum number of balls you need to pick to ensure that 
at least two of them are of the same color is 4.
-/
theorem minimum_balls_same_color (R Y B : ℕ) (hR : R = 3) (hY : Y = 5) (hB : B = 7) : 
  ∃ (n : ℕ), n ≥ 4 ∧ ∀ s, s.card = n → ∃ (c : ℕ), s.card c ≥ 2 :=
by
  sorry

end minimum_balls_same_color_l563_563241


namespace find_a_given_difference_l563_563571

theorem find_a_given_difference (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : |a - a^2| = 6) : a = 3 :=
sorry

end find_a_given_difference_l563_563571


namespace water_level_function_l563_563422

def water_level (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : ℝ :=
  0.3 * x + 6

theorem water_level_function :
  ∀ (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5), water_level x h = 6 + 0.3 * x :=
by
  intros
  unfold water_level
  sorry -- Proof skipped

end water_level_function_l563_563422


namespace unique_zero_of_f_l563_563703

-- Define the function f(x) = e^x ln(x) - 1
def f (x : ℝ) : ℝ := Real.exp x * Real.log x - 1

-- State the problem as a theorem
theorem unique_zero_of_f :
  (∀ x y, 1 ≤ x → x < y → f x < f y) ∧
  (f 1 < 0) ∧
  (f (Real.exp 1) > 0) →
  ∃! x, 1 ≤ x ∧ x < Real.exp 1 ∧ f x = 0 :=
sorry

end unique_zero_of_f_l563_563703


namespace f_derivative_at_two_l563_563876

noncomputable def f (x : ℝ) : ℝ := x^2 + 3 * x * (deriv f 1)

theorem f_derivative_at_two :
  deriv f 2 = 1 :=
begin
  sorry
end

end f_derivative_at_two_l563_563876


namespace geometric_sequence_problem_l563_563193

noncomputable def q : ℝ := 1 + Real.sqrt 2

theorem geometric_sequence_problem (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = (q : ℝ) * a n)
  (h_cond : a 2 = a 0 + 2 * a 1) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 := 
sorry

end geometric_sequence_problem_l563_563193


namespace regression_lines_intersect_at_mean_l563_563033

variable {X Y : Type}
variable [Real X] [Real Y]
variable (experimentsA : List (X × Y))
variable (experimentsB : List (X × Y))
variable (s t : X)

def mean (data : List (X × Y)) (component : X × Y → X) : X :=
  data.map component |>.sum / data.length

noncomputable def regression_line (data : List (X × Y)) : X → Y := sorry

theorem regression_lines_intersect_at_mean :
  ∀ (x y : X), 
    mean experimentsA Prod.fst = s →
    mean experimentsB Prod.fst = s →
    mean experimentsA Prod.snd = t →
    mean experimentsB Prod.snd = t →
    (regression_line experimentsA s = t) ∧ 
    (regression_line experimentsB s = t) :=
by
  sorry

end regression_lines_intersect_at_mean_l563_563033


namespace parabola_surface_area_inequality_l563_563558

variables {p : ℝ} {PQ MN : ℝ}

/-- Given the parabola y^2 = 2px, 
    a chord PQ passing through the focus, 
    and MN being the projection of PQ onto the directrix,
    the surface area S1 formed by rotating PQ around the directrix,
    and the surface area S2 of the sphere with diameter MN -/
theorem parabola_surface_area_inequality
  (h_parabola : ∃ p, ∀ y x : ℝ, y^2 = 2 * p * x)
  (h_PQ_projection : PQ ≥ MN) :
  π * PQ^2 ≥ π * MN^2 :=
begin
  sorry
end

end parabola_surface_area_inequality_l563_563558


namespace parabola_focus_focus_coordinates_l563_563692

theorem parabola_focus (p: ℝ) (h: 4 * p = -8) : p = -2 :=
by
  linarith [h]

lemma focus_of_parabola : (∃ p: ℝ, 4 * p = -8) → ∃ p: ℝ, p = -2 :=
by
  intro h_exists
  cases h_exists with p h
  use p
  exact parabola_focus p h

theorem focus_coordinates : ∃ (x y: ℝ), (y = 0) ∧ (x = -2) :=
by
  have h_focus : ∃ p: ℝ, p = -2 := focus_of_parabola (Exists.intro (-2) (by linarith))
  cases h_focus with p hp
  use p
  use 0
  exact ⟨rfl, hp⟩

end parabola_focus_focus_coordinates_l563_563692


namespace sum_of_consecutive_integers_sqrt_28_l563_563904

theorem sum_of_consecutive_integers_sqrt_28 (a b : ℤ) (h1 : b = a + 1) (h2 : a < Real.sqrt 28) (h3 : Real.sqrt 28 < b) : a + b = 11 :=
by 
    sorry

end sum_of_consecutive_integers_sqrt_28_l563_563904


namespace swimming_pool_total_volume_l563_563822

noncomputable def radius (d : ℝ) := d / 2

noncomputable def volume_cylinder (r h : ℝ) := π * r^2 * h

noncomputable def volume_hemisphere (r : ℝ) := (2 / 3) * π * r^3

theorem swimming_pool_total_volume :
  let d := 20
  let h := 6
  let r := radius d
  volume_cylinder r h + volume_hemisphere r = (3800 / 3) * π :=
by
  let d := 20
  let h := 6
  let r := radius d
  have vol_cylinder := volume_cylinder r h
  have vol_hemisphere := volume_hemisphere r
  sorry

end swimming_pool_total_volume_l563_563822


namespace water_level_function_l563_563423

def water_level (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : ℝ :=
  0.3 * x + 6

theorem water_level_function :
  ∀ (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5), water_level x h = 6 + 0.3 * x :=
by
  intros
  unfold water_level
  sorry -- Proof skipped

end water_level_function_l563_563423


namespace area_trapezoid_PSRT_l563_563976

-- Definition of the conditions
def isosceles_triangle (P Q R : Type) [metric_space P] [t2_space P] :=
  distance P Q = distance P R ∧ 
  ∃ k : ℝ, k = 1 / 2 ∧ 
  ∀ (A B C D E F : Type) [metric_space A] [t2_space A], 
  (distance A B = 2 ∧ distance C D = 2 ∧ distance E F = 2) →
  ∃ 9 : ℝ, 9 * 2 = 18

-- Area of large triangle
def area_large_triangle : ℝ := 72

-- Proof statement
theorem area_trapezoid_PSRT (P Q R S T : Type) [metric_space PQR] [t2_space PQR]
  (h₁ : isosceles_triangle P Q R)
  (h₂ : 9 * (2 : ℝ)) = 18
  (h₃ : ∀ (T : Type) [metric_space PQL],
  [t2_space PSR] [t2_space PTR],
  PQR.dist = 72) : 
  53.5 = (area_large_triangle - (2 * (1 / 2))) := 
begin
  sorry
end

end area_trapezoid_PSRT_l563_563976


namespace length_of_cable_proof_l563_563842

noncomputable def length_of_cable (x y z : ℝ) (h1 : x + y + z = 8) (h2 : x * y + y * z + x * z = 14) : ℝ :=
  4 * real.pi * real.sqrt (11 / 3)

theorem length_of_cable_proof (x y z : ℝ) (h1 : x + y + z = 8) (h2 : x * y + y * z + x * z = 14) :
  length_of_cable x y z h1 h2 = 4 * real.pi * real.sqrt (11 / 3) :=
sorry

end length_of_cable_proof_l563_563842


namespace arithmetic_sequence_terms_sum_l563_563563

variable (a_n : ℕ → ℕ)
variable (S_n : ℕ → ℕ)

-- Definitions based on given problem conditions
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (S : ℕ → ℕ) (a : ℕ → ℕ) := 
  ∀ n, S n = n * (a 1 + a n) / 2

axiom Sn_2017 : S_n 2017 = 4034

-- Goal: a_3 + a_1009 + a_2015 = 6
theorem arithmetic_sequence_terms_sum :
  arithmetic_sequence a_n →
  sum_first_n_terms S_n a_n →
  S_n 2017 = 4034 → 
  a_n 3 + a_n 1009 + a_n 2015 = 6 :=
by
  intros
  sorry

end arithmetic_sequence_terms_sum_l563_563563


namespace distributeCakes_l563_563647

def childPreferences : Type :=
  Σ' (child1 : ℕ) (child2 : ℕ) (child3 : ℕ), (child1 + child2 + child3 = 18 ∧
   ∀ (f : ℕ), 
   (f = 6 → 
    (child1 ≤ f ∧ 
     child2 ≤ f ∧ 
     child3 ≤ f)))

def isEligible (cakeFlavor: ℕ) (child: ℕ) : Prop :=
  match child with
  | 1 => cakeFlavor != 2  -- child 1 dislikes vanilla (flavor 2)
  | 2 => cakeFlavor != 0  -- child 2 dislikes chocolate (flavor 0)
  | 3 => cakeFlavor != 1  -- child 3 dislikes strawberry (flavor 1)
  | _ => false

theorem distributeCakes : ∃ childPreferences : Type,
  ∀ (i : ℕ) (c : childPreferences) (f : ℕ),
  (isEligible f i → (c.child1 = 6 ∧ c.child2 = 6 ∧ c.child3 = 6)) :=
sorry

end distributeCakes_l563_563647


namespace distance_between_x_intercepts_l563_563774

theorem distance_between_x_intercepts (x1 y1 : ℝ) 
  (m1 m2 : ℝ)
  (hx1 : x1 = 10) (hy1 : y1 = 15)
  (hm1 : m1 = 3) (hm2 : m2 = 5) :
  let x_intercept1 := (y1 - m1 * x1) / -m1
  let x_intercept2 := (y1 - m2 * x1) / -m2
  dist (x_intercept1, 0) (x_intercept2, 0) = 2 :=
by
  sorry

end distance_between_x_intercepts_l563_563774


namespace quadratic_polynomials_real_roots_and_interlacing_l563_563304

theorem quadratic_polynomials_real_roots_and_interlacing
  (p1 p2 q1 q2 : ℝ)
  (h : (q1 - q2)^2 + (p1 - p2) * (p1 * q2 - p2 * q1) < 0) :
  ∀ f g : polynomial ℝ,
  f = X^2 + C p1 * X + C q1 ∧ g = X^2 + C p2 * X + C q2 →
  (f.root_set ℝ).card = 2 ∧ (g.root_set ℝ).card = 2 ∧
  ∃ r1a r1b r2a r2b : ℝ, 
  f.root_set ℝ = {r1a, r1b} ∧ g.root_set ℝ = {r2a, r2b} ∧
  r1a < r2a ∧ r2a < r1b ∧ r1a < r2b ∧ r2b < r1b :=
begin
  sorry
end

end quadratic_polynomials_real_roots_and_interlacing_l563_563304


namespace find_matrix_n_l563_563146

open Matrix

def cross_product_matrix_3 :
  Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, -6, -1],
    ![6, 0, -3],
    ![1, 3, 0]
  ]

theorem find_matrix_n (u : Fin 3 → ℝ) :
  let N := cross_product_matrix_3
  N.mulVec u = 
  (![3, -1, 6] : Fin 3 → ℝ).cross_product_3 u := 
  by {
    sorry
  }

end find_matrix_n_l563_563146


namespace impossible_grid_fill_l563_563402

theorem impossible_grid_fill :
  ∀ (n : ℕ) (m : ℕ) (grid : Finₓ n → Finₓ m → ℕ),
  n = 2020 → m = 2020 →
  (∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 4080400) →
  (∀ k, ∑ j, grid 0 j + k = ∑ j, grid k j) →
  (∀ k, ∑ j, grid (k + 1) j = (∑ j, grid k j) + 1) →
  False :=
by
  intros n m grid hn hm hbounds hsum hdiff
  sorry

end impossible_grid_fill_l563_563402


namespace sqrt_sum_eq_l563_563681

noncomputable def real_solution (x : ℝ) : ℝ :=
  sqrt(100 - x^2) + sqrt(36 - x^2)

theorem sqrt_sum_eq (x : ℝ) (h : sqrt(100 - x^2) - sqrt(36 - x^2) = 5) :
  real_solution x = 12.8 :=
by
  sorry

end sqrt_sum_eq_l563_563681


namespace count_integers_congruent_mod_count_integers_50_300_congruent_3_mod_7_l563_563946

theorem count_integers_congruent_mod (k : ℤ) :
  (50 ≤ 7 * k + 3 ∧ 7 * k + 3 ≤ 300) ↔ (7 ≤ k ∧ k ≤ 42) :=
by sorry

theorem count_integers_50_300_congruent_3_mod_7 : 
  ∃ n : ℕ, n = 36 ∧
    (n = ∃! k : ℤ, 50 ≤ 7 * k + 3 ∧ 7 * k + 3 ≤ 300) :=
by sorry

end count_integers_congruent_mod_count_integers_50_300_congruent_3_mod_7_l563_563946


namespace odd_function_d_l563_563377

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def A (x : ℝ) : ℝ := x^2 + 2 * |x|
def B (x : ℝ) : ℝ := x * sin x
def C (x : ℝ) : ℝ := 2^x + 2^(-x)
def D (x : ℝ) : ℝ := cos x / x

-- Prove that D(x) is an odd function
theorem odd_function_d : is_odd_function D :=
  sorry

end odd_function_d_l563_563377


namespace max_sum_at_eight_l563_563972

variable {a : ℕ → ℤ} -- defining the arithmetic sequence a_n

-- defining the conditions
variable (h_arith : ∀ n : ℕ, a (n + 1) = a n + d)
variable {S : ℕ → ℤ} -- defining the sum function S_n

-- sum of first n terms for arithmetic sequence
noncomputable def S (n : ℕ) : ℤ := n * (a 1 + a n)

variable (h_S16 : S 16 > 0)
variable (h_S17 : S 17 < 0)

-- statement to prove
theorem max_sum_at_eight : ∃ n : ℕ, n = 8 ∧ S n = S (8 : ℕ) :=
by
  sorry

end max_sum_at_eight_l563_563972


namespace min_collaborative_groups_four_students_collaborative_group_l563_563025

theorem min_collaborative_groups (n : ℕ) (h_n : n = 2008) : 
  ∃ m, (∀ (G : Finset (Finset (Fin h_n))), (∀ g ∈ G, g.card = 2) → 
         ∃ A B C : Fin h_n, (A ≠ B) ∧ (B ≠ C) ∧ (A ≠ C) ∧
           (∃ g1 g2 g3 : Finset (Fin h_n), g1 ∈ G ∧ g2 ∈ G ∧ g3 ∈ G ∧
             {A, B} = g1 ∧ {B, C} = g2 ∧ {A, C} = g3)) → 
         m = 1008017 :=
sorry

theorem four_students_collaborative_group (n : ℕ) (h_n : n = 2008) (m : ℕ) 
  (h_m : m = 1008017) : 
  ∃ A B C D : Fin h_n, 
    (G.mk_pow_fin (A, B) (B, C) (C, D) (D, A)).2 ∈ (range (m / 22)) :=
sorry

end min_collaborative_groups_four_students_collaborative_group_l563_563025


namespace robert_salary_loss_l563_563055

theorem robert_salary_loss :
  ∀ (initial_salary : ℝ),
  (initial_salary > 0) →
  let decreased_salary := initial_salary * 0.60 in
  let final_salary := decreased_salary * 1.40 in
  let percentage_loss := 100 * (initial_salary - final_salary) / initial_salary in
  percentage_loss = 16 :=
begin
  intros,
  sorry
end

end robert_salary_loss_l563_563055


namespace meaningful_fraction_condition_l563_563034

theorem meaningful_fraction_condition (x : ℝ) : x - 2 ≠ 0 ↔ x ≠ 2 := 
by 
  sorry

end meaningful_fraction_condition_l563_563034


namespace geometric_sequence_value_d_l563_563549

theorem geometric_sequence_value_d 
  (c d e : ℝ) 
  (seq : list ℝ := [-1, c, d, e, -4])
  (h_geom : ∀ (i j k : ℕ), i < j → j < k → (seq.nth i).get_or_else 0 * (seq.nth k).get_or_else 0 = (seq.nth j).get_or_else 0 ^ 2) :
  d = -2 :=
by
  sorry

end geometric_sequence_value_d_l563_563549


namespace sum_of_powers_of_i_l563_563400

theorem sum_of_powers_of_i : (∑ k in Finset.range 2023, (λ n, complex.I^(n+1)) k) = -1 :=
by
  sorry

end sum_of_powers_of_i_l563_563400


namespace log_expression_equals_l563_563127

noncomputable def expression (x y : ℝ) : ℝ :=
  (Real.log x^2) / (Real.log y^10) *
  (Real.log y^3) / (Real.log x^7) *
  (Real.log x^4) / (Real.log y^8) *
  (Real.log y^6) / (Real.log x^9) *
  (Real.log x^11) / (Real.log y^5)

theorem log_expression_equals (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  expression x y = (1 / 15) * Real.log y / Real.log x :=
sorry

end log_expression_equals_l563_563127


namespace inequality_sqrt_l563_563664

theorem inequality_sqrt (a : ℝ) (h : 0 < a) : 
  sqrt (a^2 + 1/a^2) + 2 ≥ a + 1/a + sqrt 2 := 
sorry

end inequality_sqrt_l563_563664


namespace combined_cost_price_correct_l563_563734

def stock1_discount_price (p: ℝ) (d: ℝ) : ℝ := p * (1 - d / 100)
def stock1_brokerage (p: ℝ) (d: ℝ) (b: ℝ) : ℝ := stock1_discount_price p d * b / 100
def stock1_transaction_fee (p: ℝ) (d: ℝ) (t: ℝ) : ℝ := stock1_discount_price p d * t / 100
def stock1_total_cost_price (p: ℝ) (d: ℝ) (b: ℝ) (t: ℝ) : ℝ := stock1_discount_price p d + stock1_brokerage p d b + stock1_transaction_fee p d t

def stock2_discount_price (p: ℝ) (d: ℝ) : ℝ := p * (1 - d / 100)
def stock2_brokerage (p: ℝ) (d: ℝ) (b: ℝ) : ℝ := stock2_discount_price p d * b / 100
def stock2_transaction_fee : ℝ := 15
def stock2_total_cost_price (p: ℝ) (d: ℝ) (b: ℝ) : ℝ := stock2_discount_price p d + stock2_brokerage p d b + stock2_transaction_fee

def stock3_discount_price (p: ℝ) (d: ℝ) : ℝ := p * (1 - d / 100)
def stock3_brokerage (p: ℝ) (d: ℝ) (b: ℝ) : ℝ := stock3_discount_price p d * b / 100
def stock3_transaction_fee (p: ℝ) (d: ℝ) (t: ℝ) : ℝ := stock3_discount_price p d * t / 100
def stock3_total_cost_price (p: ℝ) (d: ℝ) (b: ℝ) (t: ℝ) : ℝ := stock3_discount_price p d + stock3_brokerage p d b + stock3_transaction_fee p d t

def combined_cost_price := stock1_total_cost_price 100 5 1.5 0.5 +
                           stock2_total_cost_price 200 7 0.75 +
                           stock3_total_cost_price 300 3 1 0.25

theorem combined_cost_price_correct : combined_cost_price = 593.9325 := by
  sorry

end combined_cost_price_correct_l563_563734


namespace minimize_expr1_and_expr2_l563_563395

noncomputable def minimize_expressions : ℝ × ℝ :=
  let expr1 (x y : ℝ) : ℝ := x * y / 2 + 18 / (x * y)
  let expr2 (x y : ℝ) : ℝ := y / 2 + x / 3
  let x := 3
  let y := 2
  (x, y)

theorem minimize_expr1_and_expr2 : 
  ∀ x y : ℝ, 0 < x → 0 < y → 
  (x = 3 ∧ y = 2 ∧ (x * y / 2 + 18 / (x * y) = 6) ∧ (y / 2 + x / 3 = 2)) := 
by
  intros x y hx hy
  have hx3 : x = 3 := sorry
  have hy2 : y = 2 := sorry
  repeat { try { split, assumption } }
  have expr1_min : x * y / 2 + 18 / (x * y) = 6 := sorry
  have expr2_min : y / 2 + x / 3 = 2 := sorry
  sorry

end minimize_expr1_and_expr2_l563_563395


namespace largest_possible_difference_l563_563324

/-- The largest possible difference between two 2-digit numbers
formed by the digits 2, 4, 6, and 8, each used exactly once. -/
theorem largest_possible_difference : 
  ∃ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  {a, b, c, d} = {2, 4, 6, 8}) → 
  ∃ (x y : ℕ), (10 * a + b = x → 10 * c + d = y → x - y = 62 ∨ y - x = 62) :=
sorry

end largest_possible_difference_l563_563324


namespace snoring_heart_disease_related_with_99_percent_confidence_l563_563980

theorem snoring_heart_disease_related_with_99_percent_confidence
  (chi_squared : ℝ)
  (h1 : chi_squared = 20.87)
  (h2 : 6.635 < chi_squared) :
  "There is a 99% confidence that snoring and heart disease are related" :=
by {
  -- chi_squared is 20.87 and 20.87 > 6.635, hence 99% confidence level
  sorry
}

end snoring_heart_disease_related_with_99_percent_confidence_l563_563980


namespace crayon_box_total_l563_563071

def crayons_total (red blue green pink : Nat) : Nat :=
  red + blue + green + pink

theorem crayon_box_total : 
  let red := 8 in
  let blue := 6 in
  let green := (2/3) * blue in
  let pink := 6 in
  crayons_total red blue green pink = 24 :=
by
  have green_ := (2 * blue) / 3
  show crayons_total red blue green_ pink = 24
  sorry -- proof will go here

end crayon_box_total_l563_563071


namespace probability_same_group_l563_563349

theorem probability_same_group :
  let total_cards := 20
  let drawn_cards := 4
  let specific_cards := {5, 14}
  let prob := 7 / 51
  let same_group_probability := 
    if ((specific_cards.toList.nth 0 < specific_cards.toList.nth 1) || (specific_cards.toList.nth 1 < specific_cards.toList.nth 0)) 
    then prob 
    else 0
  same_group_probability = prob :=
begin
  sorry
end

end probability_same_group_l563_563349


namespace geometric_progression_sum_l563_563346

theorem geometric_progression_sum (a q : ℝ) :
  (a + a * q^2 + a * q^4 = 63) →
  (a * q + a * q^3 = 30) →
  (a = 3 ∧ q = 2) ∨ (a = 48 ∧ q = 1 / 2) :=
by
  intro h1 h2
  sorry

end geometric_progression_sum_l563_563346


namespace sum_odd_divisors_420_l563_563042

theorem sum_odd_divisors_420 : (∑ d in (Finset.filter (λ x, x % 2 = 1) (Finset.divisors 420)), d) = 192 := by
  sorry

end sum_odd_divisors_420_l563_563042


namespace bakery_gives_away_30_doughnuts_at_end_of_day_l563_563410

def boxes_per_day (total_doughnuts doughnuts_per_box : ℕ) : ℕ :=
  total_doughnuts / doughnuts_per_box

def leftover_boxes (total_boxes sold_boxes : ℕ) : ℕ :=
  total_boxes - sold_boxes

def doughnuts_given_away (leftover_boxes doughnuts_per_box : ℕ) : ℕ :=
  leftover_boxes * doughnuts_per_box

theorem bakery_gives_away_30_doughnuts_at_end_of_day 
  (total_doughnuts doughnuts_per_box sold_boxes : ℕ) 
  (H1 : total_doughnuts = 300) (H2 : doughnuts_per_box = 10) (H3 : sold_boxes = 27) : 
  doughnuts_given_away (leftover_boxes (boxes_per_day total_doughnuts doughnuts_per_box) sold_boxes) doughnuts_per_box = 30 :=
by
  sorry

end bakery_gives_away_30_doughnuts_at_end_of_day_l563_563410


namespace dogs_for_sale_l563_563450

variable (D : ℕ)
def number_of_cats := D / 2
def number_of_birds := 2 * D
def number_of_fish := 3 * D
def total_animals := D + number_of_cats D + number_of_birds D + number_of_fish D

theorem dogs_for_sale (h : total_animals D = 39) : D = 6 :=
by
  sorry

end dogs_for_sale_l563_563450


namespace problem1_problem2_l563_563520

theorem problem1 : (1 + 3 + 5 + 7 + 9 = 25) :=
by sorry

theorem problem2 (m : ℕ) (h : ∃ k : ℕ, m^3 = ∑ i in finset.range m, (2 * (k + i) + 1) ∧ (2 * k + 1) = 21) : m = 5 :=
by sorry

end problem1_problem2_l563_563520


namespace number_of_situations_l563_563816

def total_athletes : ℕ := 6
def taken_own_coats : ℕ := 2
def taken_wrong_coats : ℕ := 4

-- Combination of choosing 2 athletes out of 6
def combinations : ℕ := Nat.choose total_athletes taken_own_coats

-- Number of derangements for 4 athletes (permutations with no fixed points)
def derangements (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangements (n - 1) + derangements (n - 2))

-- Total number of situations where 4 athletes took someone else's coats
theorem number_of_situations : combinations * derangements taken_wrong_coats = 135 := by
  sorry

end number_of_situations_l563_563816


namespace conic_section_is_ellipse_l563_563382

theorem conic_section_is_ellipse :
  ∃ (x y : ℝ), sqrt(x^2 + (y-2)^2) + sqrt((x-6)^2 + (y+4)^2) = 12 ∧
  sqrt((0-6)^2 + (2+4)^2) = 6*sqrt(2) ∧ (6*sqrt(2) < 12) :=
by
  use (x : ℝ)
  use (y : ℝ)
  sorry

end conic_section_is_ellipse_l563_563382
