import Mathlib
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.GeomSeq
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Lcm
import Mathlib.Algebra.QuadraticResidue
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Complex.Roots
import Mathlib.Analysis.Integral
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Trigonometry.Tan
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Permutations
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Seq.Regular
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Cycle
import Mathlib.Probability.Basic
import Mathlib.ProbabilityTheory
import Mathlib.Set.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactics
import ProbabilityTheory
import data.complex.basic
import data.real.basic
import data.set.basic

namespace point_P_on_parallel_triangle_l193_193017

variable {A B C D P : Type*}
variables [MetricSpace A B C D P]
variables [EmetricSpace A B C D P]

def is_shortest_height (D P A B C : Type*) : Prop :=
  ∀(x : Type*), dist D P ≤ dist D x

def is_on_parallel_triangle (P A B C : Type*) : Prop :=
  -- We should define what it means for a point to be on the triangle with sides parallel to ABC.
  sorry -- Definition should be provided based on geometric constraints.

theorem point_P_on_parallel_triangle
  (A B C D P : Type*)
  [MetricSpace A B C D P]
  [EmetricSpace A B C D P]
  (h1 : is_shortest_height D P A B C):
  is_on_parallel_triangle P A B C :=
sorry -- The proof should be provided here.

end point_P_on_parallel_triangle_l193_193017


namespace course_selection_schemes_l193_193213

theorem course_selection_schemes :
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  choose_2_courses + choose_3_courses = 64 :=
by
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  show choose_2_courses + choose_3_courses = 64 from sorry

end course_selection_schemes_l193_193213


namespace systematic_sampling_example_l193_193383

theorem systematic_sampling_example :
  ∃ (selected : Finset ℕ), 
    selected = {10, 30, 50, 70, 90} ∧
    ∀ n ∈ selected, 1 ≤ n ∧ n ≤ 100 ∧ 
    (∃ k, k > 0 ∧ k * 20 - 10∈ selected ∧ k * 20 - 10 ∈ Finset.range 101) := 
by
  sorry

end systematic_sampling_example_l193_193383


namespace total_course_selection_schemes_l193_193194

theorem total_course_selection_schemes :
  let n_physical_education := 4
  let n_art := 4
  let total_courses := n_physical_education + n_art in
  let choose2_courses := (Nat.choose n_physical_education 1) * (Nat.choose n_art 1)
  let choose3_courses := (Nat.choose n_physical_education 2 * Nat.choose n_art 1) + (Nat.choose n_physical_education 1 * Nat.choose n_art 2) in
  total_courses = n_physical_education + n_art →
  choose2_courses + choose3_courses = 64 :=
by
  intros n_physical_education n_art total_courses choose2_courses choose3_courses h
  have h_choose2_courses: choose2_courses = 16 := by
    simp [n_physical_education, n_art, Nat.choose]
  have h_choose3_courses: choose3_courses = 48 := by
    simp [n_physical_education, n_art, Nat.choose]
  rw [h_choose2_courses, h_choose3_courses]
  exact Nat.add_eq_right.2 rfl

end total_course_selection_schemes_l193_193194


namespace total_course_selection_schemes_l193_193186

theorem total_course_selection_schemes :
  let n_physical_education := 4
  let n_art := 4
  let total_courses := n_physical_education + n_art in
  let choose2_courses := (Nat.choose n_physical_education 1) * (Nat.choose n_art 1)
  let choose3_courses := (Nat.choose n_physical_education 2 * Nat.choose n_art 1) + (Nat.choose n_physical_education 1 * Nat.choose n_art 2) in
  total_courses = n_physical_education + n_art →
  choose2_courses + choose3_courses = 64 :=
by
  intros n_physical_education n_art total_courses choose2_courses choose3_courses h
  have h_choose2_courses: choose2_courses = 16 := by
    simp [n_physical_education, n_art, Nat.choose]
  have h_choose3_courses: choose3_courses = 48 := by
    simp [n_physical_education, n_art, Nat.choose]
  rw [h_choose2_courses, h_choose3_courses]
  exact Nat.add_eq_right.2 rfl

end total_course_selection_schemes_l193_193186


namespace seohyun_initial_marbles_l193_193468

variable (M : ℤ)

theorem seohyun_initial_marbles (h1 : (2 / 3) * M = 12) (h2 : (1 / 2) * M + 12 = M) : M = 36 :=
sorry

end seohyun_initial_marbles_l193_193468


namespace sequence_eventually_integers_l193_193848

theorem sequence_eventually_integers (A : ℚ) (n k : ℕ) 
  (hA : A = (∏ i in finset.range n, (2 * i + 1) : ℚ) / (∏ i in finset.range n, (2 * (i + 1)) : ℚ)) 
  (hk : k ≥ 2 * n) : 2^k * A ∈ ℤ := by
  sorry

end sequence_eventually_integers_l193_193848


namespace course_selection_count_l193_193216

-- Definitions for the conditions
def num_PE_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_PE_courses + num_art_courses

-- The number of different course selection schemes
theorem course_selection_count : 
  (choose num_PE_courses 1) * (choose num_art_courses 1) + 
  (choose num_PE_courses 2) * (choose num_art_courses 1) + 
  (choose num_PE_courses 1) * (choose num_art_courses 2) = 64 := 
by sorry

end course_selection_count_l193_193216


namespace limit_sequence_l193_193324

theorem limit_sequence {a b : ℕ} :
  (∀ n : ℕ, 0 < n → a = 2 ∧ b = 3) → (real.log (3/2) ≠ 0) →
  tendsto (λ n : ℕ, (a^(n+1) + b^(n+1)) / (a^n + b^n)) at_top (𝓝 3) :=
begin
  intros h1 h2,
  sorry,
end

end limit_sequence_l193_193324


namespace area_ratio_rect_sq_l193_193929

variable (s : ℝ)

def side_len_sq (S : ℝ) : Prop := s = S
def longer_side_rect (R : ℝ) : Prop := R = 1.2 * s
def shorter_side_rect (R : ℝ) : Prop := R = 0.8 * s
def area_sq (S : ℝ) : ℝ := S * S
def area_rect (R_long R_short : ℝ) : ℝ := R_long * R_short
def ratio_area (areaR areaS : ℝ) : ℝ := areaR / areaS

theorem area_ratio_rect_sq (s S R_long R_short : ℝ) (h1 : side_len_sq s S) (h2 : longer_side_rect s R_long) (h3 : shorter_side_rect s R_short) :
  ratio_area (area_rect R_long R_short) (area_sq S) = 24/25 :=
by
  sorry

end area_ratio_rect_sq_l193_193929


namespace find_values_A_B_l193_193537

theorem find_values_A_B :
  ∃ A B: ℕ, (A = 3 ∧ B = 6 ∧ 
  let N := A * 10^10 + 1234567890 + B in
  (N % 1 = 0) ∧ (N % 2 = 0) ∧ (N % 3 = 0) ∧ (N % 4 = 0) ∧ (N % 5 ≠ 0) ∧ (N % 6 = 0) ∧ (N % 7 = 0) ∧ (N % 8 = 0) ∧ (N % 9 = 0) ∧
  (N % 15 ≠ 0) ∧ (N % 21 ≠ 0) ∧ (N % 35 ≠ 0) ∧ (N % 45 ≠ 0))) :=
  sorry

end find_values_A_B_l193_193537


namespace sequence_conditions_l193_193698

theorem sequence_conditions (n : ℕ) (x : Fin n → ℕ) :
  (∀ j, 0 ≤ j ∧ j ≤ n → x j = Nat.card (set_of (λ k, x k = j))) →
  (x = ![1, 2, 1, 0] ∨ x = ![2, 0, 2, 0] ∨ (∃ k: ℕ, n ≥ 6 ∧ x = ![n - 3, 2, 1] ++ mk_array (n - 3) 0 ++ ![1])) :=
by
  sorry

end sequence_conditions_l193_193698


namespace problem_statement_l193_193044

noncomputable def f : ℚ → ℝ 
| a if a > 0 := if a % 1 = 0 then 2 * a else sorry
| _ := 0

theorem problem_statement :
  ( ∀ (a b : ℚ), 0 < a ∧ 0 < b → f(a * b) = f(a) + f(b) )
  ∧ ( ∀ p : ℕ, prime p → f(p) = 2 * p )
  → f (2 / 19) < 5 :=
begin
  sorry
end

end problem_statement_l193_193044


namespace tan_double_angle_identity_l193_193348

theorem tan_double_angle_identity (theta : ℝ) : 
  (tan (2 * θ) = (2 * tan θ) / (1 - tan θ ^ 2)) → 
  (tan (22.5 * (Real.pi / 180)) / (1 - tan (22.5 * (Real.pi / 180)) ^ 2) = 1 / 2) := 
by
  sorry

end tan_double_angle_identity_l193_193348


namespace garden_perimeter_l193_193453

theorem garden_perimeter (A : ℝ) (P : ℝ) : 
  (A = 97) → (P = 40) :=
by
  sorry

end garden_perimeter_l193_193453


namespace negative_values_count_l193_193376

theorem negative_values_count :
  let S := {n : ℕ | n^2 < 196 }
  in S.card = 13 :=
by
  sorry

end negative_values_count_l193_193376


namespace union_A_B_l193_193773

def A : Set ℕ := {1, 2}
def B : Set ℕ := {x | ∃ (a ∈ A) (b ∈ A), x = a + b}

theorem union_A_B :
  A ∪ B = {1, 2, 3, 4} :=
by
  sorry

end union_A_B_l193_193773


namespace not_perfect_square_l193_193517

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, k^2 = 3^n + 2 * 17^n := sorry

end not_perfect_square_l193_193517


namespace rectangle_area_l193_193337

theorem rectangle_area (l w r: ℝ) (h1 : l = 2 * r) (h2 : w = r) : l * w = 2 * r^2 :=
by sorry

end rectangle_area_l193_193337


namespace crackers_per_friend_l193_193497

theorem crackers_per_friend (total_crackers : ℕ) (num_friends : ℕ) (n : ℕ) 
  (h1 : total_crackers = 8) 
  (h2 : num_friends = 4)
  (h3 : total_crackers / num_friends = n) : n = 2 :=
by
  sorry

end crackers_per_friend_l193_193497


namespace car_total_distance_l193_193655

theorem car_total_distance (D : ℝ) :
  (D / 4 + D / 5 + D / 6 = 37) -> (3 * D = 180) :=
by
  intro h
  have hD : D = 60 :=
    calc
      D = (37 * 60) / 37 : by ring
      ... = 60 : by ring
  rw hD
  calc
    3 * 60 = 180 : by ring

end car_total_distance_l193_193655


namespace tom_current_yellow_tickets_l193_193962

-- Definitions based on conditions provided
def yellow_to_red (y : ℕ) : ℕ := y * 10
def red_to_blue (r : ℕ) : ℕ := r * 10
def yellow_to_blue (y : ℕ) : ℕ := (yellow_to_red y) * 10

def tom_red_tickets : ℕ := 3
def tom_blue_tickets : ℕ := 7

def tom_total_blue_tickets : ℕ := (red_to_blue tom_red_tickets) + tom_blue_tickets
def tom_needed_blue_tickets : ℕ := 163

-- Proving that Tom currently has 2 yellow tickets
theorem tom_current_yellow_tickets : (tom_total_blue_tickets + tom_needed_blue_tickets) / yellow_to_blue 1 = 2 :=
by
  sorry

end tom_current_yellow_tickets_l193_193962


namespace calc_result_neg2xy2_pow3_l193_193329

theorem calc_result_neg2xy2_pow3 (x y : ℝ) : 
  (-2 * x * y^2)^3 = -8 * x^3 * y^6 := 
by 
  sorry

end calc_result_neg2xy2_pow3_l193_193329


namespace a_2017_is_negative_1008_l193_193739

-- Sequence definition
def a : ℕ → ℤ
| 0 := 0  -- typically we use 1-based, but Lean is 0-based so adjust as necessary
| (n+1) := -|(a n) + (n + 1) |

theorem a_2017_is_negative_1008 : a 2016 = -1008 :=
sorry

end a_2017_is_negative_1008_l193_193739


namespace area_quadrilateral_proof_l193_193883

noncomputable def quadrilateral_condition (ABCD: Type) 
  [quad: quadrilateral ABCD] {A B C D E : Point ABCD} 
  (h1: angle ABC = 90) (h2: angle ACD = 90)
  (AC: 𝕜) (h3: AC = 25) (CD: 𝕜) (h4: CD = 20) (AE: 𝕜) (h5: AE = 8) 
  (intersect: AC ∩ BD = E) : Prop :=
area ABCD = 550

theorem area_quadrilateral_proof (ABCD: Type) 
  [quad : quadrilateral ABCD] {A B C D E : Point ABCD}
  (h1: angle ABC = 90) (h2: angle ACD = 90)
  (AC: 𝕜) (h3: AC = 25) (CD: 𝕜) (h4: CD = 20) (AE: 𝕜) (h5: AE = 8) 
  (intersect: AC ∩ BD = E) :
  quadrilateral_condition ABCD := 
sorry

end area_quadrilateral_proof_l193_193883


namespace circ_of_circle_l193_193136

theorem circ_of_circle (C : ℝ) : 
  (∀ (v1 v2 t : ℝ), v1 = 7 ∧ v2 = 8 ∧ t = 40 → (v1 + v2) * t = C) → C = 600 :=
by 
  intro h
  have h1 := h 7 8 40
  simp at h1
  exact h1 ⟨rfl, rfl, rfl⟩

end circ_of_circle_l193_193136


namespace sphere_ratios_l193_193113

theorem sphere_ratios (r1 r2 : ℝ) (h : r1 / r2 = 1 / 3) :
  (4 * π * r1^2) / (4 * π * r2^2) = 1 / 9 ∧ (4 / 3 * π * r1^3) / (4 / 3 * π * r2^3) = 1 / 27 :=
by
  sorry

end sphere_ratios_l193_193113


namespace cab_driver_income_second_day_l193_193183

theorem cab_driver_income_second_day
  (d1 d3 d4 d5 : ℕ)
  (avg_income : ℕ)
  (total_days : ℕ)
  (income_day2 : ℕ) 
  (h1 : d1 = 300)
  (h2 : d3 = 750)
  (h3 : d4 = 400)
  (h4 : d5 = 500)
  (h5 : avg_income = 420)
  (h6 : total_days = 5) :
  income_day2 = 150 := by 
  have h7 : total_income = total_days * avg_income := by sorry
  have h8 : known_income_sum = d1 + d3 + d4 + d5 := by sorry
  have h9 : income_day2 = total_income - known_income_sum := by sorry
  sorry

end cab_driver_income_second_day_l193_193183


namespace value_of_a3_minus_a2_l193_193393

theorem value_of_a3_minus_a2 : 
  (∃ S : ℕ → ℕ, (∀ n : ℕ, S n = n^2) ∧ (S 3 - S 2 - (S 2 - S 1)) = 2) :=
sorry

end value_of_a3_minus_a2_l193_193393


namespace value_of_b_l193_193948

theorem value_of_b (a b : ℕ) (h1 : a * b = 2 * (a + b) + 10) (h2 : b - a = 5) : b = 9 := 
by {
  -- Proof is not required, so we use sorry to complete the statement
  sorry
}

end value_of_b_l193_193948


namespace total_number_of_course_selection_schemes_l193_193247

-- Define the total number of courses
def total_courses := 8

-- Define number of physical education and art courses
def pe_courses := 4
def art_courses := 4

-- Define selections: students choose 2 or 3 courses
def course_selections : Finset (Finset ℕ) :=
  (Finset.powerset (Finset.range total_courses)).filter (λ s, s.card = 2 ∨ s.card = 3)

-- Define condition: at least 1 course from each category
def valid_selections : Finset (Finset ℕ) :=
  course_selections.filter (λ s, ∃ pe art, s = pe ∪ art ∧ pe.card ≠ 0 ∧ art.card ≠ 0 ∧ 
                             pe ⊆ Finset.range pe_courses ∧ art ⊆ (Finset.range total_courses).filter (λ x, x ≥ pe_courses))

theorem total_number_of_course_selection_schemes : valid_selections.card = 64 := 
by sorry

end total_number_of_course_selection_schemes_l193_193247


namespace total_course_selection_schemes_l193_193272

theorem total_course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose_two :=
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 1)
  let choose_three :=
    (Nat.choose PE_courses 2) * (Nat.choose Art_courses 1) +
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 2)
  total_courses = 8
  ∧ (choose_two + choose_three = 64) :=
by
  sorry

end total_course_selection_schemes_l193_193272


namespace smallest_positive_period_triangle_area_l193_193433

noncomputable def f (x : ℝ) : ℝ := (Real.sin (x) + Real.sqrt 3 * Real.cos (x)) * Real.sin (x) - 2

theorem smallest_positive_period (x : ℝ) :
  x > 0 → Function.periodic f π :=
sorry

theorem triangle_area (A : ℝ) (B C a b c S : ℝ) :
  0 < A ∧ A < π/2 → -- A is an acute angle
  a = 2 * Real.sqrt 3 →
  c = 4 →
  Real.sin (2 * A - π / 6) = 1 →
  b = Real.sqrt (12 - a^2 + 2 * a * Real.cos A) →
  S = 1/2 * c * b * Real.sin A →
  S = 2 * Real.sqrt 3 :=
sorry

end smallest_positive_period_triangle_area_l193_193433


namespace improper_fraction_not_in_lowest_terms_count_l193_193713

theorem improper_fraction_not_in_lowest_terms_count :
  (∃ count : ℕ, count = 69 ∧
  ∀ N : ℤ, 1 ≤ N ∧ N ≤ 2023 →
    ¬ is_coprime (N^2 + 4 : ℤ) (N + 5) → count = 69) := sorry

end improper_fraction_not_in_lowest_terms_count_l193_193713


namespace total_course_selection_schemes_l193_193191

theorem total_course_selection_schemes :
  let n_physical_education := 4
  let n_art := 4
  let total_courses := n_physical_education + n_art in
  let choose2_courses := (Nat.choose n_physical_education 1) * (Nat.choose n_art 1)
  let choose3_courses := (Nat.choose n_physical_education 2 * Nat.choose n_art 1) + (Nat.choose n_physical_education 1 * Nat.choose n_art 2) in
  total_courses = n_physical_education + n_art →
  choose2_courses + choose3_courses = 64 :=
by
  intros n_physical_education n_art total_courses choose2_courses choose3_courses h
  have h_choose2_courses: choose2_courses = 16 := by
    simp [n_physical_education, n_art, Nat.choose]
  have h_choose3_courses: choose3_courses = 48 := by
    simp [n_physical_education, n_art, Nat.choose]
  rw [h_choose2_courses, h_choose3_courses]
  exact Nat.add_eq_right.2 rfl

end total_course_selection_schemes_l193_193191


namespace correct_transformation_l193_193914

-- Assume g(x) is defined as described in the problem
noncomputable def g : ℝ → ℝ := sorry

-- Define the transformation function
noncomputable def transformed_g (x : ℝ) : ℝ := g ((x + 3) / -2) + 1

-- Main theorem statement to prove the transformation is correctly applied
theorem correct_transformation :
  ∀ x, (y = g(x) ∧ -5 ≤ x ∧ x ≤ 5) →
       (y' = transformed_g(x') ∧ -11 ≤ x' ∧ x' ≤ 17) :=
sorry

end correct_transformation_l193_193914


namespace course_selection_count_l193_193217

-- Definitions for the conditions
def num_PE_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_PE_courses + num_art_courses

-- The number of different course selection schemes
theorem course_selection_count : 
  (choose num_PE_courses 1) * (choose num_art_courses 1) + 
  (choose num_PE_courses 2) * (choose num_art_courses 1) + 
  (choose num_PE_courses 1) * (choose num_art_courses 2) = 64 := 
by sorry

end course_selection_count_l193_193217


namespace triangle_angle_problem_l193_193012

theorem triangle_angle_problem
  (CEF_eq : ∀ {A B C : Type} [has_angle A B C], A ∈ (triangle CEF) → B ∈ (triangle CEF) → C ∈ (triangle CEF) → ∠ A B C = 60)
  (DIH_eq : ∀ {A B C : Type} [has_angle A B C], A ∈ (triangle DIH) → B ∈ (triangle DIH) → C ∈ (triangle DIH) → ∠ A B C = 60)
  (angle_FCD : ∠ F C D = 75)
  (angle_MDC : ∠ MDC = 55) :
  x = 40 :=
sorry

end triangle_angle_problem_l193_193012


namespace shortest_distance_point_on_circle_to_line_l193_193951

theorem shortest_distance_point_on_circle_to_line
  (P : ℝ × ℝ)
  (hP : (P.1 + 1)^2 + (P.2 - 2)^2 = 1) :
  ∃ (d : ℝ), d = 3 :=
sorry

end shortest_distance_point_on_circle_to_line_l193_193951


namespace candies_count_l193_193528

theorem candies_count :
  ∃ n, (n = 35 ∧ ∃ x, x ≥ 11 ∧ n = 3 * (x - 1) + 2) ∧ ∃ y, y ≤ 9 ∧ n = 4 * (y - 1) + 3 :=
  by {
    sorry
  }

end candies_count_l193_193528


namespace limit_sequence_l193_193323

theorem limit_sequence {a b : ℕ} :
  (∀ n : ℕ, 0 < n → a = 2 ∧ b = 3) → (real.log (3/2) ≠ 0) →
  tendsto (λ n : ℕ, (a^(n+1) + b^(n+1)) / (a^n + b^n)) at_top (𝓝 3) :=
begin
  intros h1 h2,
  sorry,
end

end limit_sequence_l193_193323


namespace exponent_calculation_l193_193333

theorem exponent_calculation : 
  (-27 : ℝ)^(2/3) * (9 : ℝ)^(-3/2)  = (1/3 : ℝ) :=
by 
  -- Conditions 
  have h1 : (-27 : ℝ) = (-3 : ℝ)^3 := by norm_num,
  have h2 : (9 : ℝ) = (3 : ℝ)^2 := by norm_num,
  
  -- Simplify using the conditions
  calc 
    (-27 : ℝ)^(2/3) * (9 : ℝ)^(-3/2)
      = ((-3 : ℝ)^3)^(2/3) * ((3 : ℝ)^2)^(-3/2) : by rw [h1, h2]
  ... = ((-3 : ℝ)^(3*(2/3))) * ((3 : ℝ)^(2*(-3/2))) : by rw [real.rpow_mul, real.rpow_mul]
  ... = ((-3 : ℝ)^(2)) * ((3 : ℝ)^(-3)) : by norm_num
  ... = ((9 : ℝ)) * ((1 / (3 : ℝ)^3)) : by rw [real.rpow_neg, real.rpow_nat_cast]; norm_num
  ... = (9 : ℝ) * (1 / 27) : by norm_num
  ... = (1 / 3 : ℝ) : by norm_num

end exponent_calculation_l193_193333


namespace course_selection_schemes_count_l193_193239

-- Definitions based on the conditions
def num_physical_education_courses : ℕ := 4
def num_art_courses : ℕ := 4

-- Required to choose 2 or 3 courses, with at least one from each category
def valid_selection_cases : list (ℕ × ℕ) := [(1, 1), (1, 2), (2, 1)]

-- Calculate the number of ways to choose k courses from n courses using combination formula
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Calculate the total number of different course selection schemes
def total_course_selection_schemes : ℕ :=
  let case1 := C num_physical_education_courses 1 * C num_art_courses 1
  let case2a := C num_physical_education_courses 2 * C num_art_courses 1
  let case2b := C num_physical_education_courses 1 * C num_art_courses 2
  case1 + case2a + case2b

-- The assertion that the total number of different course selection schemes is 64
theorem course_selection_schemes_count : total_course_selection_schemes = 64 := by
  -- sorry placeholder for proof
  sorry

end course_selection_schemes_count_l193_193239


namespace acute_triangle_inequality_l193_193484

variable (f : ℝ → ℝ)
variable {A B : ℝ}
variable (h₁ : ∀ x : ℝ, x * (f'' x) - 2 * (f x) > 0)
variable (h₂ : A + B < Real.pi / 2 ∧ 0 < A ∧ 0 < B)

theorem acute_triangle_inequality :
  f (Real.cos A) * (Real.sin B) ^ 2 < f (Real.sin B) * (Real.cos A) ^ 2 := 
  sorry

end acute_triangle_inequality_l193_193484


namespace polynomial_irreducible_l193_193490

theorem polynomial_irreducible (n : ℤ) (hn : n > 1) : irreducible (Polynomial.Coe (ℤ) (Polynomial.X ^ n + 5 * Polynomial.X ^ (n - 1) + 3)) :=
sorry

end polynomial_irreducible_l193_193490


namespace janet_waiting_time_l193_193835

-- Define the speeds and distance
def janet_speed : ℝ := 30 -- miles per hour
def sister_speed : ℝ := 12 -- miles per hour
def lake_width : ℝ := 60 -- miles

-- Define the travel times
def janet_travel_time : ℝ := lake_width / janet_speed
def sister_travel_time : ℝ := lake_width / sister_speed

-- The theorem to be proved
theorem janet_waiting_time : 
  sister_travel_time - janet_travel_time = 3 := 
by 
  sorry

end janet_waiting_time_l193_193835


namespace functions_have_same_value_at_x_neg_six_l193_193972

theorem functions_have_same_value_at_x_neg_six :
  ∀ (x : ℝ), (2 * x + 1 = x - 5) ↔ x = -6 :=
by 
  intro x
  split
  · intro h
    have : 2 * x + 6 = x := by linarith
    exact eq_of_sub_eq_zero (by linarith)
  · intro h
    rw h
    linarith

end functions_have_same_value_at_x_neg_six_l193_193972


namespace no_points_in_unit_cube_l193_193813

theorem no_points_in_unit_cube :
  ∀ (points : set (ℝ × ℝ × ℝ)), points.card = 1956 →
    ∃ x y z : ℝ, x >= 0 ∧ x <= 12 ∧ y >= 0 ∧ y <= 12 ∧ z >= 0 ∧ z <= 12 ∧
      (∀ (p : ℝ × ℝ × ℝ), p ∈ points → (p.1 < x ∨ p.1 > x + 1 ∨ p.2 < y ∨ p.2 > y + 1 ∨ p.3 < z ∨ p.3 > z + 1)) :=
by {
  sorry
}

end no_points_in_unit_cube_l193_193813


namespace bug_on_point_5_after_2015_jumps_l193_193370

def next_point (p : ℕ) : ℕ :=
  if p % 2 == 1 then (p + 2) % 5 else (p + 1) % 5

def bug_position (start : ℕ) (jumps : ℕ) : ℕ :=
  Nat.iterate jumps next_point start

theorem bug_on_point_5_after_2015_jumps :
  bug_position 5 2015 = 5 :=
by 
  -- Proof can be constructed here
  sorry 

end bug_on_point_5_after_2015_jumps_l193_193370


namespace total_number_of_course_selection_schemes_l193_193253

-- Define the total number of courses
def total_courses := 8

-- Define number of physical education and art courses
def pe_courses := 4
def art_courses := 4

-- Define selections: students choose 2 or 3 courses
def course_selections : Finset (Finset ℕ) :=
  (Finset.powerset (Finset.range total_courses)).filter (λ s, s.card = 2 ∨ s.card = 3)

-- Define condition: at least 1 course from each category
def valid_selections : Finset (Finset ℕ) :=
  course_selections.filter (λ s, ∃ pe art, s = pe ∪ art ∧ pe.card ≠ 0 ∧ art.card ≠ 0 ∧ 
                             pe ⊆ Finset.range pe_courses ∧ art ⊆ (Finset.range total_courses).filter (λ x, x ≥ pe_courses))

theorem total_number_of_course_selection_schemes : valid_selections.card = 64 := 
by sorry

end total_number_of_course_selection_schemes_l193_193253


namespace integral_f_l193_193764

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x^2
  else if 1 < x ∧ x ≤ Real.exp 1 then 1 / x
  else 0

theorem integral_f :
  ∫ x in 0..Real.exp 1, f x = 4 / 3 := by
  sorry

end integral_f_l193_193764


namespace louise_average_speed_is_correct_l193_193670

-- Define the distances
def total_distance : ℝ := 3 * d
def chantal_first_segment_distance : ℝ := 2 * d
def chantal_second_segment_distance : ℝ := d
def chantal_return_first_segment_distance : ℝ := (2/3) * d 

-- Speed definitions
def chantal_first_segment_speed : ℝ := 5
def chantal_second_segment_speed : ℝ := 3
def chantal_return_speed : ℝ := 4

-- Calculate time for each segment
def chantal_first_segment_time : ℝ := chantal_first_segment_distance / chantal_first_segment_speed
def chantal_second_segment_time : ℝ := chantal_second_segment_distance / chantal_second_segment_speed
def chantal_return_time : ℝ := chantal_return_first_segment_distance / chantal_return_speed

-- Total time until Louise and Chantal meet
def total_time : ℝ := chantal_first_segment_time + chantal_second_segment_time + chantal_return_time

-- Louise travels (2/3)d in the same total time
def louise_meeting_distance : ℝ := (2/3) * d

-- Average speed of Louise
def louise_speed : ℝ := louise_meeting_distance / total_time

-- Theorem statement
theorem louise_average_speed_is_correct : louise_speed = 20 / 27 := by
  sorry

end louise_average_speed_is_correct_l193_193670


namespace star_point_angle_l193_193118

theorem star_point_angle (n : ℕ) (h : n > 4) (h₁ : n ≥ 3) :
  ∃ θ : ℝ, θ = (n-2) * 180 / n :=
by
  sorry

end star_point_angle_l193_193118


namespace B_profit_l193_193180

-- Definitions based on conditions
def investment_ratio (B_invest A_invest : ℕ) : Prop := A_invest = 3 * B_invest
def period_ratio (B_period A_period : ℕ) : Prop := A_period = 2 * B_period
def total_profit (total : ℕ) : Prop := total = 28000
def B_share (total : ℕ) := total / 7

-- Theorem statement based on the proof problem
theorem B_profit (B_invest A_invest B_period A_period total : ℕ)
  (h1 : investment_ratio B_invest A_invest)
  (h2 : period_ratio B_period A_period)
  (h3 : total_profit total) :
  B_share total = 4000 :=
by
  sorry

end B_profit_l193_193180


namespace irrational_count_is_4_l193_193311

def number_list : List Real :=
  [ -Real.sqrt 5, -3 * Real.pi, 1/2, 3.1415, Real.cbrt 64, 
    let rec seq : ℕ → ℝ 
      | 0 => 0.1
      | n => seq n.pred * 10 + if n % 2 = 0 then 1 else 6
    in seq 10 / 10, Real.sqrt 9, Real.sqrt 8 ]

def is_irrational (x : Real) : Prop :=
  ¬ ∃ (q : ℚ), x = q

theorem irrational_count_is_4 : 
  List.countp is_irrational number_list = 4 :=
sorry

end irrational_count_is_4_l193_193311


namespace problem1_problem2_l193_193666

/-
  Problem 1: Prove that the following expression equals 4 + sqrt 6:
  sqrt 48 ÷ sqrt 3 - sqrt (1 / 2) * sqrt 12 + sqrt 24
-/
theorem problem1 : sqrt 48 / sqrt 3 - sqrt (1 / 2) * sqrt 12 + sqrt 24 = 4 + sqrt 6 :=
by
  sorry

/-
  Problem 2: Prove that the following expression equals 1:
  (sqrt 27 - sqrt 12) / sqrt 3
-/
theorem problem2 : (sqrt 27 - sqrt 12) / sqrt 3 = 1 :=
by
  sorry

end problem1_problem2_l193_193666


namespace calc_op_l193_193551

def op (a b : ℕ) := (a + b) * (a - b)

theorem calc_op : (op 5 2)^2 = 441 := 
by 
  sorry

end calc_op_l193_193551


namespace jed_correct_speed_l193_193809

def fine_per_mph := 16
def jed_fine := 256
def speed_limit := 50

def jed_speed : Nat := speed_limit + jed_fine / fine_per_mph

theorem jed_correct_speed : jed_speed = 66 := by
  sorry

end jed_correct_speed_l193_193809


namespace fraction_C_D_l193_193344

noncomputable def C : ℝ := ∑' n, if n % 6 = 0 then 0 else if n % 2 = 0 then ((-1)^(n/2 + 1) / (↑n^2)) else 0
noncomputable def D : ℝ := ∑' n, if n % 6 = 0 then ((-1)^(n/6 + 1) / (↑n^2)) else 0

theorem fraction_C_D : C / D = 37 := sorry

end fraction_C_D_l193_193344


namespace election_winning_percentage_l193_193178

def total_votes (a b c : ℕ) : ℕ := a + b + c

def winning_percentage (votes_winning : ℕ) (total : ℕ) : ℚ :=
(votes_winning * 100 : ℚ) / total

theorem election_winning_percentage (a b c : ℕ) (h_votes : a = 6136 ∧ b = 7636 ∧ c = 11628) :
  winning_percentage c (total_votes a b c) = 45.78 := by
  sorry

end election_winning_percentage_l193_193178


namespace solve_f_sqrt_2009_l193_193048

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom f_never_zero : ∀ x : ℝ, f x ≠ 0
axiom functional_eq : ∀ x y : ℝ, f (x - y) = 2009 * f x * f y

theorem solve_f_sqrt_2009 :
  f (sqrt 2009) = 1 / 2009 := sorry

end solve_f_sqrt_2009_l193_193048


namespace find_a_value_l193_193793

theorem find_a_value (a : ℕ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 := 
by 
  sorry -- Placeholder for the proof

end find_a_value_l193_193793


namespace average_sales_correct_l193_193908

def sales : List ℕ := [90, 75, 55, 130, 110, 85]

def refund : ℕ := 25

def adjusted_sales := sales.take 4 ++ [(sales.nth! 4 - refund)] ++ [sales.nth! 5]

def total_sales : ℕ := adjusted_sales.sum

def months : ℕ := 6

def average_sales : ℚ := total_sales / months

theorem average_sales_correct : average_sales = 520 / 6 := by
  unfold average_sales total_sales adjusted_sales total_sales sales months
  sorry

end average_sales_correct_l193_193908


namespace identical_lines_unique_pair_l193_193360

theorem identical_lines_unique_pair :
  ∃! (a b : ℚ), 2 * (0 : ℚ) + a * (0 : ℚ) + 10 = 0 ∧ b * (0 : ℚ) - 3 * (0 : ℚ) - 15 = 0 ∧ 
  (-2 / a = b / 3) ∧ (-10 / a = 5) :=
by {
  -- Given equations in slope-intercept form:
  -- y = -2 / a * x - 10 / a
  -- y = b / 3 * x + 5
  -- Slope and intercept comparison leads to equations:
  -- -2 / a = b / 3
  -- -10 / a = 5
  sorry
}

end identical_lines_unique_pair_l193_193360


namespace john_must_sell_134_pens_l193_193029

-- Define the conditions of the problem
def cost_per_pen := 8 / 5
def selling_per_pen := 10 / 4
def profit_per_pen := selling_per_pen - cost_per_pen
def required_profit := 120
def number_of_pens := required_profit / profit_per_pen

-- State the proof problem
theorem john_must_sell_134_pens : number_of_pens.ceil = 134 := by
  sorry

end john_must_sell_134_pens_l193_193029


namespace area_of_triangle_l193_193967

noncomputable def triangle_area := 
  let A := (-3 : ℝ, 1 : ℝ)
  let B := (7 : ℝ, 1 : ℝ)
  let C := (5 : ℝ, -3 : ℝ)
  let AB := (B.1 - A.1)
  let height := (1 - (-3))
  (1 / 2) * AB * height

theorem area_of_triangle : triangle_area = 20 := 
by
  let A := (-3 : ℝ, 1 : ℝ)
  let B := (7 : ℝ, 1 : ℝ)
  let C := (5 : ℝ, -3 : ℝ)
  let AB := B.1 - A.1
  let height := 1 - (-3)
  have AB_val : AB = 10 := sorry
  have height_val : height = 4 := sorry
  calc
    triangle_area = (1 / 2) * AB * height : by refl
    ... = (1 / 2) * 10 * 4 : by rw [AB_val, height_val]
    ... = 20 : by norm_num

end area_of_triangle_l193_193967


namespace part1_part2_part3_l193_193723
noncomputable def f (x : ℝ) : ℝ := (4 * x) / (3 * x^2 + 3)
noncomputable def g (x a : ℝ) : ℝ := (1 / 2) * x^2 - Math.log x - a
def mu (x : ℝ) : ℝ := (1 / 2) * x^2 - Math.log x

-- Part (1)
theorem part1 (x : ℝ) (h : 0 < x ∧ x < 2) : f x ∈ (Set.Ioc 0 (2 / 3)) :=
sorry

-- Part (2)
theorem part2 (a x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) (ha : g x a = 0) : a ∈ Set.Icc (1 / 2) (2 - Math.log 2) :=
sorry

-- Part (3)
theorem part3 (a : ℝ) 
  (h : ∀ x1 : ℝ, 0 < x1 ∧ x1 < 2 → (∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 2 ∧ f x1 = g x2 a)) :
  a ∈ Set.Icc (1 / 2) (4 / 3 - Math.log 2) :=
sorry

end part1_part2_part3_l193_193723


namespace part1_part2_l193_193414

-- Definition of f(x)
def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

-- Theorem statement for part (1)
theorem part1 : f 2 + f (1/2) = 1 ∧ f 3 + f (1/3) = 1 := by
  sorry

-- Theorem statement for part (2)
theorem part2 (x : ℝ) (hx : x ≠ 0) : f x + f (1 / x) = 1 := by
  sorry

end part1_part2_l193_193414


namespace sqrt_fraction_rational_l193_193423

-- Given the conditions
variables (r q n : ℚ)
hypothesis (h : (1 / (r + q * n)) + (1 / (q + r * n)) = 1 / (r + q))

-- The goal will be to prove that \(\sqrt{\frac{n-3}{n+1}}\) is a rational number.
theorem sqrt_fraction_rational (hrational : ∀ r q n : ℚ, (1 / (r + q * n)) + (1 / (q + r * n)) = 1 / (r + q)) :
  ∃ t : ℚ, t^2 = (n - 3) / (n + 1) :=
begin
  -- The proof will go here
  sorry,
end

end sqrt_fraction_rational_l193_193423


namespace balls_into_boxes_l193_193071

theorem balls_into_boxes :
  let balls := {1, 2, 3, 4}
  let boxes := {1, 2, 3}
  (∃ f : balls → boxes, (∀ b ∈ boxes, ∃ x ∈ balls, f x = b)) ∧ 
  fintype.card (set_of (λ f : balls → boxes, ∀ b ∈ boxes, ∃ x ∈ balls, f x = b)) = 36 := sorry

end balls_into_boxes_l193_193071


namespace part_i_extremum_part_i_max_value_part_ii_increasing_l193_193870

open Real

noncomputable def f (a x : ℝ) : ℝ := a * x - a / x - 2 * log x 

theorem part_i_extremum (a : ℝ) :
  (∃ x, x = 2 ∧ deriv (λ x, f a x) x = 0) ↔ a = 4 / 5 :=
sorry

theorem part_i_max_value :
  f (4 / 5) (1 / 2) = 2 * log 2 - 6 / 5 :=
sorry

theorem part_ii_increasing (a : ℝ) :
  (∀ x > 0, deriv (λ x, f a x) x ≥ 0) ↔ a ≥ 1 :=
sorry

end part_i_extremum_part_i_max_value_part_ii_increasing_l193_193870


namespace megan_finished_problems_l193_193875

theorem megan_finished_problems
  (initial_problems : ℕ)
  (pages_left : ℕ)
  (problems_per_page : ℕ)
  (initial_problems_eq : initial_problems = 40)
  (pages_left_eq : pages_left = 2)
  (problems_per_page_eq : problems_per_page = 7) :
  initial_problems - (pages_left * problems_per_page) = 26 :=
by {
  rw [initial_problems_eq, pages_left_eq, problems_per_page_eq],
  norm_num,
  sorry
}

end megan_finished_problems_l193_193875


namespace MH_tangent_to_omega_l193_193849

/-- Let  $\omega$ be the  $A$-excircle of triangle  $ABC$ and  $M$ the midpoint of side  $BC$. 
$G$ is the pole of  $AM$ w.r.t  $\omega$ and  $H$ is the midpoint of segment  $AG$. 
Prove that  $MH$ is tangent to  $\omega$. --/
theorem MH_tangent_to_omega 
  {A B C M G H : Type*} 
  (triangle_ABC : is_triangle A B C)
  (excircle_A : is_A_excircle ω A)
  (M_midpoint : is_midpoint M B C)
  (G_pole : is_pole G (line_through A M) ω)
  (H_midpoint : is_midpoint H A G) : 
  is_tangent (line_through M H) ω :=
begin
  sorry
end

end MH_tangent_to_omega_l193_193849


namespace total_course_selection_schemes_l193_193257

theorem total_course_selection_schemes (PE_courses : ℕ) (Art_courses : ℕ) : 
  PE_courses = 4 → Art_courses = 4 → 
  (finset.card (finset.powerset_len 2 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b ∈ s), a < PE_courses ∧ b ≥ PE_courses) + 
   finset.card (finset.powerset_len 3 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b c ∈ s), (a < PE_courses ∧ b < PE_courses ∧ c ≥ PE_courses) ∨ 
                        (a < PE_courses ∧ b ≥ PE_courses ∧ c ≥ PE_courses ∧ a ≠ b ≠ c)) = 64 :=
by
  sorry

end total_course_selection_schemes_l193_193257


namespace problem_statement_l193_193998

theorem problem_statement : ((26.3 * 12 * 20) / 3 + 125 - Real.sqrt 576 = 21141) :=
by
  sorry

end problem_statement_l193_193998


namespace volume_sum_correct_l193_193338

-- Define the dimensions of the rectangular parallelepiped
def length := 2
def width := 3
def height := 4

-- Define a function to calculate the volume of the parallelepiped
def volume_parallelepiped : ℕ := length * width * height

-- Define the total volume including points within one unit of the parallelepiped
noncomputable def total_volume : ℝ := 76 + (31 * Real.pi / 3)

-- Define the expected values for m, n, p
def m := 228
def n := 31
def p := 3

-- Prove that m + n + p equals 262
theorem volume_sum_correct : m + n + p = 262 :=
by
  -- The proof would go here; we simply state the expected result
  sorry

end volume_sum_correct_l193_193338


namespace simplify_expression_l193_193078

theorem simplify_expression :
  ((4 * 7) / (12 * 14)) * ((9 * 12 * 14) / (4 * 7 * 9)) ^ 2 = 1 := 
by
  sorry

end simplify_expression_l193_193078


namespace drone_altitude_l193_193710

theorem drone_altitude (h c d : ℝ) (HC HD CD : ℝ)
  (HCO_eq : h^2 + c^2 = HC^2)
  (HDO_eq : h^2 + d^2 = HD^2)
  (CD_eq : c^2 + d^2 = CD^2) 
  (HC_val : HC = 170)
  (HD_val : HD = 160)
  (CD_val : CD = 200) :
  h = 50 * Real.sqrt 29 :=
by
  sorry

end drone_altitude_l193_193710


namespace percentage_of_students_who_like_blue_l193_193065

theorem percentage_of_students_who_like_blue (B : ℝ) (h1 : ∃ n, n = 200)
  (h2 : ∃ (y : ℝ), y = 144)
  (h3 : ∀ (x : ℝ), x = 0.4 * (200 - (B / 100 * 200))) :
  B = 30 :=
by
  -- Intro variables and given conditions
  intro n hn
  intro y hy
  intro x hx
  -- Prove B = 30
  sorry

end percentage_of_students_who_like_blue_l193_193065


namespace min_distance_sums_2sqrt2_l193_193725

noncomputable def min_distance_sum (x y : ℝ) : ℝ :=
  sqrt (x^2 + y^2) + sqrt (x^2 + (y-1)^2) + sqrt ((x-1)^2 + y^2) + sqrt ((x-1)^2 + (y-1)^2)

theorem min_distance_sums_2sqrt2 (x y : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) :
  min_distance_sum x y = 2 * sqrt 2 :=
sorry

end min_distance_sums_2sqrt2_l193_193725


namespace janet_wait_time_l193_193840

theorem janet_wait_time
  (janet_speed : ℝ)
  (sister_speed : ℝ)
  (lake_width : ℝ)
  (janet_time : ℝ)
  (sister_time : ℝ) :
  janet_speed = 30 →
  sister_speed = 12 →
  lake_width = 60 →
  janet_time = lake_width / janet_speed →
  sister_time = lake_width / sister_speed →
  (sister_time - janet_time = 3) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end janet_wait_time_l193_193840


namespace function_is_even_with_period_l193_193541

-- Define the function
def f (x : ℝ) : ℝ := -cos (2 * x)

-- Prove that f is even and its smallest positive period is π
theorem function_is_even_with_period:
  (∀ x : ℝ, f x = f (-x)) ∧ (∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = π) :=
by {
  sorry
}

end function_is_even_with_period_l193_193541


namespace tablet_price_after_discounts_l193_193509

-- Define the conditions
def original_price : ℝ := 250
def monday_discount : ℝ := 0.30
def wednesday_discount : ℝ := 0.25

-- Define the expected final price after the series of discounts
def final_price : ℝ := 131.25

-- State the theorem to prove
theorem tablet_price_after_discounts : 
  let price_after_monday := original_price * (1 - monday_discount),
      price_after_wednesday := price_after_monday * (1 - wednesday_discount)
  in price_after_wednesday = final_price := 
by 
  -- The proof is omitted for simplicity
  sorry

end tablet_price_after_discounts_l193_193509


namespace product_of_distances_is_one_l193_193133

theorem product_of_distances_is_one (k : ℝ) (x1 x2 : ℝ)
  (h1 : x1^2 - k*x1 - 1 = 0)
  (h2 : x2^2 - k*x2 - 1 = 0)
  (h3 : x1 ≠ x2) :
  (|x1| * |x2| = 1) :=
by
  -- Proof goes here
  sorry

end product_of_distances_is_one_l193_193133


namespace smallest_area_of_P_l193_193512

noncomputable def smallest_possible_area_of_polygon : ℝ :=
  let points := [(0, 6), (1, 3), (2, 0), (3, 6), (4, 3), (5, 0), (6, 6), (7, 3), (8, 0), (9, 6), (10, 3), (11, 0)] in
  let hull := convex_hull ℝ points in
  area hull

theorem smallest_area_of_P (P : set (ℝ × ℝ)) (hP : convex P) (h_labels: ∀ p ∈ P, ∃ k, p = ((k % 10), (k // 10)) ∧ k % 7 = 0) :
  area P = 63 :=
by
  sorry

end smallest_area_of_P_l193_193512


namespace lower_side_length_is_correct_l193_193436

noncomputable def length_of_lower_side
  (a b h : ℝ) (A : ℝ) 
  (cond1 : a = b + 3.4)
  (cond2 : h = 5.2)
  (cond3 : A = 100.62) : ℝ :=
b

theorem lower_side_length_is_correct
  (a b h : ℝ) (A : ℝ)
  (cond1 : a = b + 3.4)
  (cond2 : h = 5.2)
  (cond3 : A = 100.62)
  (ha : A = (1/2) * (a + b) * h) : b = 17.65 :=
by
  sorry

end lower_side_length_is_correct_l193_193436


namespace proof_u_g_3_l193_193863

noncomputable def u (x : ℝ) : ℝ := Real.sqrt (5 * x + 2)

noncomputable def g (x : ℝ) : ℝ := 7 - u x

theorem proof_u_g_3 :
  u (g 3) = Real.sqrt (37 - 5 * Real.sqrt 17) :=
sorry

end proof_u_g_3_l193_193863


namespace percentage_apples_is_50_percent_l193_193583

-- Definitions for the given conditions
def initial_apples : ℕ := 10
def initial_oranges : ℕ := 5
def added_oranges : ℕ := 5

-- Defining the proof problem
theorem percentage_apples_is_50_percent :
  let total_fruits := initial_apples + initial_oranges + added_oranges in
  let apples_percentage := (initial_apples * 100) / total_fruits in
  apples_percentage = 50 :=
by
  sorry

end percentage_apples_is_50_percent_l193_193583


namespace min_value_frac_ineq_l193_193399

theorem min_value_frac_ineq (a b : ℝ) (h1 : a > 1) (h2 : b > 2) (h3 : a + b = 5) : 
  (1 / (a - 1) + 9 / (b - 2)) = 8 :=
sorry

end min_value_frac_ineq_l193_193399


namespace sugar_percentage_after_addition_l193_193984

-- Defining the initial conditions
def initial_volume : ℝ := 340
def initial_water_percentage : ℝ := 0.75
def initial_kola_percentage : ℝ := 0.05
def initial_sugar_percentage : ℝ := 1 - initial_water_percentage - initial_kola_percentage

-- Calculating initial amounts
def initial_water_volume : ℝ := initial_water_percentage * initial_volume
def initial_kola_volume : ℝ := initial_kola_percentage * initial_volume
def initial_sugar_volume : ℝ := initial_sugar_percentage * initial_volume

-- Additional components
def additional_sugar : ℝ := 3.2
def additional_water : ℝ := 12
def additional_kola : ℝ := 6.8

-- New amounts after addition
def new_sugar_volume : ℝ := initial_sugar_volume + additional_sugar
def new_water_volume : ℝ := initial_water_volume + additional_water
def new_kola_volume : ℝ := initial_kola_volume + additional_kola

-- Total new volume
def new_total_volume : ℝ := new_sugar_volume + new_water_volume + new_kola_volume

-- Percentage of sugar in the new solution
def new_sugar_percentage : ℝ := (new_sugar_volume / new_total_volume) * 100

-- Lean proof with the statement only
theorem sugar_percentage_after_addition :
  new_sugar_percentage ≈ 19.67 :=
begin
  sorry
end

end sugar_percentage_after_addition_l193_193984


namespace number_of_triangles_with_perimeter_27_l193_193780

theorem number_of_triangles_with_perimeter_27 : 
  ∃ (n : ℕ), (∀ (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ a + b + c = 27 → a + b > c ∧ a + c > b ∧ b + c > a → 
  n = 19 ) :=
  sorry

end number_of_triangles_with_perimeter_27_l193_193780


namespace principal_value_of_arg_conjugate_l193_193409

noncomputable def principal_value_arg_conjugate (θ : ℝ) (h1 : π / 2 < θ) (h2 : θ < π) : ℝ :=
  3 * π / 4 - θ / 2

theorem principal_value_of_arg_conjugate (θ : ℝ) (h1 : π / 2 < θ) (h2 : θ < π) :
  let z := 1 - real.sin θ + complex.I * real.cos θ
  let conjugate_z := complex.conj z
  complex.arg conjugate_z = principal_value_arg_conjugate θ h1 h2 :=
begin
  sorry
end

end principal_value_of_arg_conjugate_l193_193409


namespace find_k_l193_193400

variable (a b : ℝ → ℝ → ℝ)
variable {k : ℝ}

-- Defining conditions
axiom a_perpendicular_b : ∀ x y, a x y = 0
axiom a_unit_vector : a 1 0 = 1
axiom b_unit_vector : b 0 1 = 1
axiom sum_perpendicular_to_k_diff : ∀ x y, (a x y + b x y) * (k * a x y - b x y) = 0

theorem find_k : k = 1 :=
sorry

end find_k_l193_193400


namespace graph_is_hyperbola_l193_193157

theorem graph_is_hyperbola : ∀ x y : ℝ, (x + y) ^ 2 = x ^ 2 + y ^ 2 + 2 * x + 2 * y ↔ (x - 1) * (y - 1) = 1 := 
by {
  sorry
}

end graph_is_hyperbola_l193_193157


namespace taxi_fare_l193_193122

theorem taxi_fare (total_fare tip initial_distance_cost extra_distance_rate total_distance target_distance : ℝ) (h1 : total_fare = 15) (h2 : tip = 3) (h3 : initial_distance_cost = 3) (h4 : extra_distance_rate = 3) (h5 : target_distance = 3.75) :
  let remaining_fare := total_fare - tip,
      base_distance := 0.75,
      fare_for_base_distance := initial_distance_cost,
      extra_distance := target_distance - base_distance,
      fare_for_extra_distance := extra_distance_rate * extra_distance in
      fare_for_base_distance + fare_for_extra_distance = remaining_fare :=
by
  sorry

end taxi_fare_l193_193122


namespace trig_identity_l193_193738

theorem trig_identity (x y r : ℝ) (hx : x = -3) (hy : y = 4) (hr : r = 5) (ha : r = Real.sqrt (x^2 + y^2)) :
  Real.sin (Real.arcsin (y / r)) + 2 * Real.cos (Real.arccos (x / r)) = -2 / 5 := 
by
  rw [hx, hy, hr]
  have hxr : x / r = -3 / 5 := by norm_num
  have hyr : y / r = 4 / 5 := by norm_num
  have hsin : Real.sin (Real.arcsin (y / r)) = 4 / 5 := by rw [hyr, Real.sin_arcsin]; norm_num
  have hcos : Real.cos (Real.arccos (x / r)) = -3 / 5 := by rw [hxr, Real.cos_arccos]; norm_num
  rw [hsin, hcos]
  norm_num
  sorry

end trig_identity_l193_193738


namespace january_first_day_l193_193816

def jan_days : ℕ := 31
def tuesdays_in_jan : ℕ := 5
def saturdays_in_jan : ℕ := 4

theorem january_first_day (jan_days = 31) (tuesdays_in_jan = 5) (saturdays_in_jan = 4) : 
  day_of_week 1 = "Saturday" :=
by 
  sorry

end january_first_day_l193_193816


namespace coeff_x20_in_expansion_l193_193454

theorem coeff_x20_in_expansion :
  let term1 := (finset.range 24).sum (λ i, x^i)
  let term2 := (finset.range 14).sum (λ i, x^i) in
  coeff (term1 * (term2 ^ 2)) 20 = 36 :=
by
  sorry

end coeff_x20_in_expansion_l193_193454


namespace total_course_selection_schemes_l193_193187

theorem total_course_selection_schemes :
  let n_physical_education := 4
  let n_art := 4
  let total_courses := n_physical_education + n_art in
  let choose2_courses := (Nat.choose n_physical_education 1) * (Nat.choose n_art 1)
  let choose3_courses := (Nat.choose n_physical_education 2 * Nat.choose n_art 1) + (Nat.choose n_physical_education 1 * Nat.choose n_art 2) in
  total_courses = n_physical_education + n_art →
  choose2_courses + choose3_courses = 64 :=
by
  intros n_physical_education n_art total_courses choose2_courses choose3_courses h
  have h_choose2_courses: choose2_courses = 16 := by
    simp [n_physical_education, n_art, Nat.choose]
  have h_choose3_courses: choose3_courses = 48 := by
    simp [n_physical_education, n_art, Nat.choose]
  rw [h_choose2_courses, h_choose3_courses]
  exact Nat.add_eq_right.2 rfl

end total_course_selection_schemes_l193_193187


namespace course_selection_schemes_l193_193209

theorem course_selection_schemes :
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  choose_2_courses + choose_3_courses = 64 :=
by
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  show choose_2_courses + choose_3_courses = 64 from sorry

end course_selection_schemes_l193_193209


namespace maximal_participating_countries_l193_193712

-- Define the conditions and the statement to prove
theorem maximal_participating_countries :
  ∃ n : ℕ, (∀ (p : ℕ) (v : set (set ℕ)) (n_v : ℕ),
    n_v = 9 ∧ (∀ c : set ℕ, c ∈ v → c.card = 3) ∧
    (∀ c₁ c₂ : set ℕ, c₁ ∈ v ∧ c₂ ∈ v → c₁ ≠ c₂ → c₁ ∩ c₂ = ∅) ∧
    (∀ c₁ c₂ c₃ : set ℕ, c₁ ∈ v ∧ c₂ ∈ v ∧ c₃ ∈ v → (c₁ ∩ c₂ ∩ c₃).card = 0) →
    n = 56) :=
sorry

end maximal_participating_countries_l193_193712


namespace train_cross_time_l193_193831

-- Define the given constants
def len_train : ℝ := 55
def speed_train_kmph : ℝ := 36

-- Convert the speed from km/hr to m/s
def speed_train : ℝ := speed_train_kmph * (1000 / 3600)

-- Statement to prove
theorem train_cross_time : (len_train / speed_train) = 5.5 := by
  sorry

end train_cross_time_l193_193831


namespace cos_theta_passes_point_l193_193758

noncomputable def distance_from_origin (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

def cos_angle (x y : ℝ) : ℝ :=
  x / (distance_from_origin x y)

theorem cos_theta_passes_point (α : ℝ) (h: (-1, 2) ∈ { p : ℝ × ℝ | cos_angle p.1 p.2 = - (Real.sqrt 5) / 5 }) : 
  cos_angle (-1) 2 = - (Real.sqrt 5) / 5 :=
by {
  -- here we are using assumptions directly in the statement to facilitate the conditions as premises.
  sorry
}

end cos_theta_passes_point_l193_193758


namespace fit_105_balls_fit_106_balls_l193_193138

-- Defining the given conditions
def diameter := 1
def radius := diameter / 2
def box_length := 10
def box_width := 10
def box_height := 1
def unit_ball_diameter := diameter
def unit_ball_radius := radius

-- Proving the first part: Fitting 105 unit balls
theorem fit_105_balls : ∃ (arrangement : ℕ), arrangement = 105 ∧ fits_into_box arrangement unit_ball_radius box_length box_width box_height := sorry

-- Proving the second part: Fitting 106 unit balls
theorem fit_106_balls : ∃ (arrangement : ℕ), arrangement = 106 ∧ fits_into_box arrangement unit_ball_radius box_length box_width box_height := sorry

end fit_105_balls_fit_106_balls_l193_193138


namespace even_in_deg_for_even_edges_polyhedron_l193_193728

theorem even_in_deg_for_even_edges_polyhedron 
  (P : Polyhedron) 
  (even_edges : even P.num_edges) : 
  ∃ (f : P.edge → Direction), ∀ v : P.vertex, even (P.in_degree v f) := 
sorry

end even_in_deg_for_even_edges_polyhedron_l193_193728


namespace sum_of_mean_median_and_mode_is_5_4_l193_193148

-- Given the list
def nums := [1,2,1,4,3,1,2,4,1,5]

-- Define the mean
def mean (l : List ℕ) : ℝ :=
  (l.sum : ℝ) / l.length

-- Define the median
def median (l : List ℕ) : ℝ :=
  let sorted := l.qsort (λ x y => x < y)
  if h : l.length % 2 = 1 then
    sorted.get ⟨l.length / 2, sorry⟩
  else
    let i := l.length / 2
    (sorted.get ⟨i - 1, sorry⟩ + sorted.get ⟨i, sorry⟩ : ℕ) / 2

-- Define the mode
def mode (l : List ℕ) : ℕ :=
  l.foldr (λ x m, if l.count x > l.count m then x else m) 0

-- Define the sum of mean, median, and mode
def sum_mean_median_mode (l : List ℕ) : ℝ :=
  mean l + median l + (mode l : ℝ)

#eval sum_mean_median_mode nums  -- Expected to output 5.4

-- The theorem that states the sum of mean, median, and mode is 5.4
theorem sum_of_mean_median_and_mode_is_5_4 : 
  sum_mean_median_mode nums = 5.4 := sorry

end sum_of_mean_median_and_mode_is_5_4_l193_193148


namespace fewest_candies_l193_193650

-- Defining the conditions
def condition1 (x : ℕ) := x % 21 = 5
def condition2 (x : ℕ) := x % 22 = 3
def condition3 (x : ℕ) := x > 500

-- Stating the main theorem
theorem fewest_candies : ∃ x : ℕ, condition1 x ∧ condition2 x ∧ condition3 x ∧ x = 509 :=
  sorry

end fewest_candies_l193_193650


namespace total_course_selection_schemes_l193_193255

theorem total_course_selection_schemes (PE_courses : ℕ) (Art_courses : ℕ) : 
  PE_courses = 4 → Art_courses = 4 → 
  (finset.card (finset.powerset_len 2 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b ∈ s), a < PE_courses ∧ b ≥ PE_courses) + 
   finset.card (finset.powerset_len 3 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b c ∈ s), (a < PE_courses ∧ b < PE_courses ∧ c ≥ PE_courses) ∨ 
                        (a < PE_courses ∧ b ≥ PE_courses ∧ c ≥ PE_courses ∧ a ≠ b ≠ c)) = 64 :=
by
  sorry

end total_course_selection_schemes_l193_193255


namespace sets_are_different_l193_193486

theorem sets_are_different (a b c : ℤ) (h₀ : 0 < a) (h₁ : a < c - 1) (h₂ : 1 < b) (h₃ : b < c) : 
  ¬ (Finset.image (λ k : ℤ, (k * b % c)) (Finset.range (a + 1)) = Finset.range (a + 1)) :=
begin
  sorry
end

end sets_are_different_l193_193486


namespace reflection_equation_l193_193294

theorem reflection_equation
  (incident_line : ∀ x y : ℝ, 2 * x - y + 2 = 0)
  (reflection_axis : ∀ x y : ℝ, x + y - 5 = 0) :
  ∃ x y : ℝ, x - 2 * y + 7 = 0 :=
by
  sorry

end reflection_equation_l193_193294


namespace max_value_of_f_l193_193544

def f (x : ℝ) : ℝ := Math.cos x - Real.sqrt 3 * Math.sin x

theorem max_value_of_f :
  ∃ x ∈ set.Icc (0 : ℝ) (Real.pi / 2), 
  ∀ y ∈ set.Icc (0 : ℝ) (Real.pi / 2), f(x) ≥ f(y) ∧ f(x) = 1 :=
sorry

end max_value_of_f_l193_193544


namespace disagreeing_parents_in_each_level_l193_193116

theorem disagreeing_parents_in_each_level :
  let primary_total := 300 in
  let primary_agreed_percentage := 0.30 in
  let intermediate_total := 250 in
  let intermediate_agreed_percentage := 0.20 in
  let secondary_total := 250 in
  let secondary_agreed_percentage := 0.10 in
  primary_total - (primary_agreed_percentage * primary_total).toInt = 210 ∧
  intermediate_total - (intermediate_agreed_percentage * intermediate_total).toInt = 200 ∧
  secondary_total - (secondary_agreed_percentage * secondary_total).toInt = 225 :=
by
  sorry

end disagreeing_parents_in_each_level_l193_193116


namespace min_value_fraction_sum_l193_193906

theorem min_value_fraction_sum (p q r a b : ℝ) (hpq : 0 < p) (hq : p < q) (hr : q < r)
  (h_sum : p + q + r = a) (h_prod_sum : p * q + q * r + r * p = b) (h_prod : p * q * r = 48) :
  ∃ (min_val : ℝ), min_val = (1 / p) + (2 / q) + (3 / r) ∧ min_val = 3 / 2 :=
sorry

end min_value_fraction_sum_l193_193906


namespace simplify_expression_l193_193077

theorem simplify_expression (x y : ℝ) : 3 * y - 5 * x + 2 * y + 4 * x = 5 * y - x :=
by
  sorry

end simplify_expression_l193_193077


namespace correct_calculation_option_l193_193976

theorem correct_calculation_option :
  (∀ a : ℝ, 3 * a^5 - a^5 ≠ 3) ∧
  (∀ a : ℝ, a^2 + a^5 ≠ a^7) ∧
  (∀ a : ℝ, a^5 + a^5 = 2 * a^5) ∧
  (∀ x y : ℝ, x^2 * y + x * y^2 ≠ 2 * x^3 * y^3) :=
by
  sorry

end correct_calculation_option_l193_193976


namespace trapezoid_area_eq_14_point_4_l193_193911

noncomputable def area_of_trapezoid (OC OD : ℝ) (OC_eq : OC = 2) (OD_eq : OD = 4) : ℝ :=
  let CD := real.sqrt (OC ^ 2 + OD ^ 2) in
  let r := (4 * real.sqrt 5) / 5 in
  let H := 2 * r in
  let sum_bases := (18 * real.sqrt 5) / 5 in
  1 / 2 * H * sum_bases

theorem trapezoid_area_eq_14_point_4 :
  area_of_trapezoid 2 4 = 14.4 :=
by
  unfold area_of_trapezoid;
  rw [OC_eq, OD_eq];
  have sqrt_20 : real.sqrt (2 ^ 2 + 4 ^ 2) = 2 * real.sqrt 5 := by sorry;
  rw sqrt_20;
  have r_eq : (4 * real.sqrt 5) / 5 = (8 * real.sqrt 5) / 10 := by sorry;
  rw r_eq;
  have H_eq : 2 * ((4 * real.sqrt 5) / 5) = 8 * real.sqrt 5 / 5 := by sorry;
  rw H_eq;
  have sum_bases_eq : (18 * real.sqrt 5) / 5 = 72 * real.sqrt 5 / 20 := by sorry;
  rw sum_bases_eq;
  have area_eq : (1 / 2) * (8 * real.sqrt 5 / 5) * (18 * real.sqrt 5 / 5) = 144 / 10 := by sorry;
  rw area_eq;
  norm_num

end trapezoid_area_eq_14_point_4_l193_193911


namespace probability_of_three_digit_number_div_by_three_l193_193579

noncomputable def probability_three_digit_div_by_three : ℚ :=
  let digit_mod3_groups := 
    {rem0 := {3, 6, 9}, rem1 := {1, 4, 7}, rem2 := {2, 5, 8}} in
  let valid_combinations :=
    (finset.card (finset.powersetLen 3 digit_mod3_groups.rem0) +
     (finset.card digit_mod3_groups.rem0 * finset.card digit_mod3_groups.rem1 * finset.card digit_mod3_groups.rem2) +
     finset.card (finset.powersetLen 3 digit_mod3_groups.rem1) +
     finset.card (finset.powersetLen 3 digit_mod3_groups.rem2))
  in
  let total_combinations := finset.card (finset.powersetLen 3 (finset.univ : finset (fin 9))) in
  (valid_combinations : ℚ) / total_combinations

theorem probability_of_three_digit_number_div_by_three :
  probability_three_digit_div_by_three = 5 / 14 := by
  -- provide proof here
  sorry

end probability_of_three_digit_number_div_by_three_l193_193579


namespace divisors_not_multiples_of_14_l193_193492

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 2
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 3
def is_perfect_fifth (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 5
def is_perfect_seventh (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 7

def n : ℕ := 2^2 * 3^3 * 5^5 * 7^7

theorem divisors_not_multiples_of_14 :
  is_perfect_square (n / 2) →
  is_perfect_cube (n / 3) →
  is_perfect_fifth (n / 5) →
  is_perfect_seventh (n / 7) →
  (∃ d : ℕ, d = 240) :=
by
  sorry

end divisors_not_multiples_of_14_l193_193492


namespace people_who_joined_are_6_l193_193080

noncomputable def number_of_people_joined (truck_capacity : ℕ) 
(rating_per_person : ℕ) 
(initial_workers : ℕ) 
(initial_hours : ℕ) 
(additional_hours : ℕ) 
(blocks_filled_initially : ℕ) 
(remaining_blocks : ℕ) 
: ℕ :=
let total_initial_block_fill := initial_workers * rating_per_person * initial_hours in
let remaining_blocks_calculated := truck_capacity - total_initial_block_fill in
let additional_people_joined := ((remaining_blocks_calculated / (rating_per_person * additional_hours)) - initial_workers) in
additional_people_joined

theorem people_who_joined_are_6 : 
  number_of_people_joined 6000 250 2 4 2 (2 * 250 * 4) (6000 - (2 * 250 * 4)) = 6 := by sorry

end people_who_joined_are_6_l193_193080


namespace number_of_triangles_with_perimeter_27_l193_193781

theorem number_of_triangles_with_perimeter_27 : 
  ∃ (n : ℕ), (∀ (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ a + b + c = 27 → a + b > c ∧ a + c > b ∧ b + c > a → 
  n = 19 ) :=
  sorry

end number_of_triangles_with_perimeter_27_l193_193781


namespace part_a_even_n_l193_193620

theorem part_a_even_n (n : ℕ) (H: ∃ (tiles : fin (5 * n) → fin 2 → fin 5 × fin n), 
  (∀ (i: fin (5 * n)), ∃ (j: fin 2), tiles i j = tiles i (1 - j))): Even n :=
by
  sorry

end part_a_even_n_l193_193620


namespace gravel_amount_l193_193285

theorem gravel_amount (total_material sand gravel : ℝ) 
  (h1 : total_material = 14.02) 
  (h2 : sand = 8.11) 
  (h3 : gravel = total_material - sand) : 
  gravel = 5.91 :=
  sorry

end gravel_amount_l193_193285


namespace a_sequence_c_sequence_l193_193757

-- Define the sequences a_n, b_n, and c_n
def b (n : ℕ) : ℕ := (2 * n - 1) * 3 ^ n + 4

def a : ℕ → ℕ
| 1        := 7
| (n + 2) := 4 * (n + 2) * 3 ^ (n + 1)

def c (n : ℕ) : ℕ := 3 + (n - 1) * 3^(n + 1) + 4 * n

-- Prove the formula for the sequence a_n
theorem a_sequence (n : ℕ) : 
  ∑ i in finset.range (n + 1), a i = b n :=
sorry

-- Prove the formula for the sequence c_n
theorem c_sequence (n : ℕ) : 
  ∑ i in finset.range (n + 1), b i = c n :=
sorry

end a_sequence_c_sequence_l193_193757


namespace probability_both_quitters_from_first_tribe_l193_193556

theorem probability_both_quitters_from_first_tribe
  (total_people : ℕ)
  (tribe1 : ℕ)
  (tribe2 : ℕ)
  (quitting : ℕ)
  (A_in_tribe1 : Prop)
  (A_quits : Prop)
  (prob : ℚ)
  (h_total : total_people = 18)
  (h_tribes : tribe1 = 9 ∧ tribe2 = 9)
  (h_quitting : quitting = 2)
  (h_A_in_tribe1 : A_in_tribe1)
  (h_A_quits : A_quits ∧ A_in_tribe1)
  (h_prob : prob = 8 / 153) :
  prob = (probability (both_quitters_from_first_tribe)) :=
  sorry

end probability_both_quitters_from_first_tribe_l193_193556


namespace course_selection_schemes_l193_193206

theorem course_selection_schemes :
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  choose_2_courses + choose_3_courses = 64 :=
by
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  show choose_2_courses + choose_3_courses = 64 from sorry

end course_selection_schemes_l193_193206


namespace find_s_l193_193481

theorem find_s (n r s c d : ℝ) (h1 : c^2 - n * c + 3 = 0) (h2 : d^2 - n * d + 3 = 0) 
  (h3 : (c + 1/d)^2 - r * (c + 1/d) + s = 0) (h4 : (d + 1/c)^2 - r * (d + 1/c) + s = 0) 
  (h5 : c * d = 3) : s = 16 / 3 := 
by
  sorry

end find_s_l193_193481


namespace zero_in_interval_l193_193124

noncomputable def f (x : ℝ) : ℝ := log x / log 2 - 7 / x

theorem zero_in_interval :
  (∃ x, x ∈ (set.Ioo 3 4) ∧ f x = 0) :=
begin
  -- f is continuous on (0, ∞)
  sorry
end

end zero_in_interval_l193_193124


namespace prob_from_boxes_l193_193131

-- Define the probability theory space and events
open_locale big_operators

def is_prime (n : ℕ) : Prop := nat.prime n

noncomputable def boxA_prob : ℚ := 19 / 30
noncomputable def boxB_prob : ℚ := (11 : ℚ) / 25

theorem prob_from_boxes :
  let boxA : finset ℕ := finset.range 31,
      boxB := finset.range' 10 35,
      prime_or_gt28 := (boxB.filter (λ n, is_prime n)).union (boxB.filter (λ n, n > 28)),
      boxB_prime_or_gt28 := prime_or_gt28.card
  in boxA_prob * boxB_prob = 209 / 750 :=
by {
  let boxA := finset.range 31,
  let boxB := finset.range' 10 35,
  let prime_or_gt28 := (boxB.filter (λ n, is_prime n)).union (boxB.filter (λ n, n > 28)),
  let boxB_prime_or_gt28 := prime_or_gt28.card,
  have A_prob := boxA_prob,
  have B_prob := boxB_prob,
  linarith,
  sorry
}

end prob_from_boxes_l193_193131


namespace speed_in_still_water_l193_193296

theorem speed_in_still_water (v_m v_s : ℝ)
  (downstream : 48 = (v_m + v_s) * 3)
  (upstream : 34 = (v_m - v_s) * 4) :
  v_m = 12.25 :=
by
  sorry

end speed_in_still_water_l193_193296


namespace find_all_positive_real_solutions_l193_193700

noncomputable def is_solution (x y z : ℝ) : Prop :=
  x = 1 / (y^2 + y - 1) ∧
  y = 1 / (z^2 + z - 1) ∧
  z = 1 / (x^2 + x - 1)

theorem find_all_positive_real_solutions :
  ∀ (x y z : ℝ),
  (0 < x ∧ 0 < y ∧ 0 < z ∧ is_solution x y z) ↔ 
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨
   (x = 2 * Real.cos (2 * Real.pi / 9) ∧ y = 2 * Real.cos (2 * Real.pi / 9) ∧ z = 2 * Real.cos (2 * Real.pi / 9)) ∨
   (x = 2 * Real.cos (4 * Real.pi / 9) ∧ y = 2 * Real.cos (4 * Real.pi / 9) ∧ z = 2 * Real.cos (4 * Real.pi / 9)) ∨
   (x = 2 * Real.cos (8 * Real.pi / 9) ∧ y = 2 * Real.cos (8 * Real.pi / 9) ∧ z = 2 * Real.cos (8 * Real.pi / 9))) :=
begin
  sorry
end

end find_all_positive_real_solutions_l193_193700


namespace tractor_time_l193_193910

-- Define variables and conditions
variable (d : ℝ) -- The distance parameter
variable (v1 : ℝ) (v : ℝ) -- Speeds of the car before and after encountering the tractor
variable (t1 t2 t3 : ℝ) -- Time parameters

-- Condition definitions
def conditions : Prop :=
  -- Midpoint speed and time condition
  v1 = 2 * d ∧
  t1 = d / v ∧
  t2 = 2 / v

-- Proof statement that the total time after encountering the tractor is 5 hours
theorem tractor_time
  (h : conditions d v1 v):
  t1 + t2 = 5 :=
by
  sorry

end tractor_time_l193_193910


namespace distinct_values_1980_sequence_l193_193016

theorem distinct_values_1980_sequence :
  let seq := λ k, (k^2) / 1980
  let floored_seq := λ n, (finset.range n).image (λ k => ⌊(k ^ 2 : ℤ) / 1980⌋)
  (finset.card (floored_seq 1980)) = 1486 :=
by {
  let seq := λ k, (k^2) / 1980
  let floored_seq := λ n, (finset.range n).image (λ k => ⌊(k ^ 2 : ℤ) / 1980⌋)
  exact sorry
}

end distinct_values_1980_sequence_l193_193016


namespace selection_problem_l193_193594

open Nat

noncomputable def combination (n k : ℕ) : ℕ := n.choose k

theorem selection_problem :
  let ways1 := combination 54 2 * combination 58 1,
      ways2 := combination 54 1 * combination 58 2 in
  ways1 + ways2 = combination 54 2 * combination 58 1 + combination 54 1 * combination 58 2 :=
by
  let ways1 := combination 54 2 * combination 58 1
  let ways2 := combination 54 1 * combination 58 2
  sorry

end selection_problem_l193_193594


namespace b_value_if_real_div_l193_193388

def imaginary_unit : ℂ := complex.I

def z1 (b : ℝ) : ℂ := 3 - b * imaginary_unit

def z2 : ℂ := 1 - 2 * imaginary_unit

theorem b_value_if_real_div (b : ℝ) (h : (z1 b / z2).im = 0) : b = 6 := by
  sorry

end b_value_if_real_div_l193_193388


namespace total_course_selection_schemes_l193_193231

theorem total_course_selection_schemes (
  pe_courses art_courses : Finset ℕ
) : 
  pe_courses.card = 4 →
  art_courses.card = 4 →
  let total_schemes := 
    (pe_courses.card.choose 1 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 2 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 1 * art_courses.card.choose 2) in
  total_schemes = 64 := 
by
  intros h1 h2
  let case1 := pe_courses.card.choose 1 * art_courses.card.choose 1 -- 4 * 4
  let case2_1 := pe_courses.card.choose 2 * art_courses.card.choose 1 -- 6 * 4
  let case2_2 := pe_courses.card.choose 1 * art_courses.card.choose 2 -- 4 * 6
  let case2 := case2_1 + case2_2 -- 24 + 24
  let total_schemes := case1 + case2 -- 16 + 48
  have hcard := by
    simp [Nat.choose]
    exact h1
  have hcase1 : case1 = 16 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_1 : case2_1 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_2 : case2_2 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2 : case2 = 48 := by {
    rw [hcase2_1, hcase2_2]
    norm_num
  }
  have htotal_schemes : total_schemes = 64 := by {
    rw [hcase1, hcase2]
    norm_num
  }
  exact htotal_schemes

end total_course_selection_schemes_l193_193231


namespace geometric_common_ratio_sum_of_H_9_l193_193731

noncomputable def geometric_sequence (a_1 q : ℝ) (n : ℕ) : ℕ → ℝ
| 0 := a_1
| (n+1) := geometric_sequence a_1 q n * q

noncomputable def sum_of_geometric_sequence (a_1 q : ℝ) (n : ℕ) : ℝ :=
if q = 1 then a_1 * n
else a_1 * (1 - q ^ n) / (1 - q)

noncomputable def b_n (a_1 q : ℝ) (n : ℕ) : ℝ := float.log2 (geometric_sequence a_1 q (n + 1))

noncomputable def T_n (a_1 q : ℝ) (n : ℕ) : ℝ := (n : ℝ) * float.log2 (a_1 * q) + (n * (n + 1) / 2) * float.log2 q

theorem geometric_common_ratio (S_4 a_1 a_5 : ℝ) : 
  sum_of_geometric_sequence a_1 (a_5 / a_1) 4 = a_5 - a_1 → 
  a_5 / a_1 = -1 ∨ a_5 / a_1 = 2 :=
sorry

theorem sum_of_H_9 (a_1 : ℝ) : 
  let q := 2 in 
  (T_n a_1 q 4 = 2 * b_n a_1 q 4) → 
  (∑ i in Finset.range 9, (1 / (b_n a_1 q i * b_n a_1 q (i + 1)))) = 9 / 10 :=
sorry

end geometric_common_ratio_sum_of_H_9_l193_193731


namespace semicircle_area_ratio_l193_193965

theorem semicircle_area_ratio (r : ℝ) (r_pos : 0 < r) :
  let radius_of_circle := (3 * r) / 2 in
  let area_semicircle_A := (1 / 2) * π * r^2 in
  let area_semicircle_B := (1 / 2) * π * (2 * r)^2 in
  let combined_area_semicircles := area_semicircle_A + area_semicircle_B in
  let area_circle := π * radius_of_circle^2 in
  (combined_area_semicircles / area_circle = 10 / 9) :=
by
  sorry

end semicircle_area_ratio_l193_193965


namespace point_in_fourth_quadrant_l193_193092

-- Define the complex number under consideration
def z : ℂ := 1 - 2 * complex.I

-- Define the quadrant identifiers (for clarity, although typically these would be better encapsulated)
def isInFourthQuadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- The problem statement
theorem point_in_fourth_quadrant : isInFourthQuadrant z :=
by
  sorry

end point_in_fourth_quadrant_l193_193092


namespace similar_triangles_area_ratio_l193_193119

theorem similar_triangles_area_ratio (r : ℚ) (h : r = 1/3) : (r^2) = 1/9 :=
by
  sorry

end similar_triangles_area_ratio_l193_193119


namespace f_increasing_in_interval_f_min_value_f_max_value_l193_193762

noncomputable section

def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

-- Monotonicity proof
theorem f_increasing_in_interval : ∀ (x1 x2 : ℝ), 3 ≤ x1 → x1 < x2 → x2 ≤ 5 → f x1 < f x2 :=
by
  sorry

-- Minimum value proof
theorem f_min_value : ∃ x ∈ Icc (3 : ℝ) 5, f x = 5 / 4 :=
by
  use 3
  constructor
  { split
    · norm_num
    · norm_num }
  norm_num

-- Maximum value proof
theorem f_max_value : ∃ x ∈ Icc (3 : ℝ) 5, f x = 3 / 2 :=
by
  use 5
  constructor
  { split
    · norm_num
    · norm_num }
  norm_num

end f_increasing_in_interval_f_min_value_f_max_value_l193_193762


namespace range_of_a_inequality_solution_set_l193_193769

noncomputable def quadratic_condition_holds (a : ℝ) : Prop :=
∀ (x : ℝ), x^2 - 2 * a * x + a > 0

theorem range_of_a (a : ℝ) (h : quadratic_condition_holds a) : 0 < a ∧ a < 1 := sorry

theorem inequality_solution_set (a x : ℝ) (h1 : 0 < a) (h2 : a < 1) : (a^(x^2 - 3) < a^(2 * x) ∧ a^(2 * x) < 1) ↔ x > 3 := sorry

end range_of_a_inequality_solution_set_l193_193769


namespace vector_ratio_l193_193397

-- Define four non-collinear points O, A, B, and C in a plane.
variables {V : Type*} [inner_product_space ℝ V]
variables (O A B C : V)

-- Given condition
def condition (O A B C : V) : Prop :=
  (O -ᵥ A) - 3 • (O -ᵥ B) + 2 • (O -ᵥ C) = 0

-- The proof goal
theorem vector_ratio (O A B C : V) (h : condition O A B C) : 
  (∥A - B∥ / ∥B - C∥) = 2 :=
sorry

end vector_ratio_l193_193397


namespace increasing_function_range_of_a_l193_193107

variable {f : ℝ → ℝ}

theorem increasing_function_range_of_a (a : ℝ) (h : ∀ x : ℝ, 3 * a * x^2 ≥ 0) : a > 0 :=
sorry

end increasing_function_range_of_a_l193_193107


namespace area_of_enclosed_region_l193_193045

def g (x : ℝ) : ℝ := 1 - real.sqrt (1 - (x - 1)^2)

theorem area_of_enclosed_region :
  (let area := real.pi / 4 in
   real.floor (100 * area + 0.5) / 100 = 0.79) :=
by
  sorry

end area_of_enclosed_region_l193_193045


namespace seq_a_general_formula_sum_seq_b_l193_193015

namespace MySequenceProofs

def seq_a (n : ℕ) : ℝ :=
  if h : n > 0 then
    match n with
    | 1 => 1
    | k + 1 => seq_a k / (2 * seq_a k + 1)
  else 0

def seq_b (n : ℕ) : ℝ :=
  if n > 0 then seq_a n * seq_a (n + 1) else 0

def T_n (n : ℕ) : ℝ :=
  if n > 0 then ((List.range n).map seq_b).sum else 0

theorem seq_a_general_formula : ∀ n : ℕ, n > 0 → seq_a n = 1 / (2 * n - 1) := by
  sorry

theorem sum_seq_b : ∀ n : ℕ, n > 0 → T_n n = n / (2 * (n + 1) - 1) := by
  sorry

end MySequenceProofs

end seq_a_general_formula_sum_seq_b_l193_193015


namespace knowledge_contest_rankings_l193_193814

theorem knowledge_contest_rankings:
  let students := {1, 2, 3, 4, 5}
  let permutations := students.to_finset.powerset.to_list.permutations
  let valid_permutations := permutations.filter (λ perm, 
    perm.head ≠ 1 ∧ perm.index 2 ≠ perm.length - 1)
  valid_permutations.card = 54 :=
by
  sorry

end knowledge_contest_rankings_l193_193814


namespace train_length_is_310m_l193_193646

-- Define the conditions
def trainSpeedKmH : ℝ := 45
def timeToPassBridgeSec : ℝ := 36
def lengthOfBridgeM : ℝ := 140

-- Define the equivalent assertions for speed in m/s and total distance
def trainSpeedMS : ℝ := (trainSpeedKmH * 1000) / 3600
def totalDistance : ℝ := trainSpeedMS * timeToPassBridgeSec

-- The goal
def lengthOfTrain : ℝ := totalDistance - lengthOfBridgeM

theorem train_length_is_310m : lengthOfTrain = 310 :=
by
  -- Proof construction goes here
  sorry

end train_length_is_310m_l193_193646


namespace find_smallest_n_l193_193708

theorem find_smallest_n :
  ∃ n : ℤ, 4 * n^2 - 28 * n + 48 < 0 ∧ ∀ m : ℤ, 4 * m^2 - 28 * m + 48 < 0 → m ≥ 4 :=
begin
  sorry
end

end find_smallest_n_l193_193708


namespace student_rank_left_l193_193645

theorem student_rank_left {n m : ℕ} (h1 : n = 10) (h2 : m = 6) : (n - m + 1) = 5 := by
  sorry

end student_rank_left_l193_193645


namespace base_r_correct_l193_193450

theorem base_r_correct (r : ℕ) :
  (5 * r ^ 2 + 6 * r) + (4 * r ^ 2 + 2 * r) = r ^ 3 + r ^ 2 → r = 8 := 
by 
  sorry

end base_r_correct_l193_193450


namespace find_age_l193_193907

-- Definitions for the columns and values.
def columns := [1, 2, 4, 8]

def age_in_columns (age : ℕ) : List ℕ :=
  List.filter (fun column => (age &&& column) ≠ 0) columns

-- Theorems to be proved.
theorem find_age (age : ℕ) (h : age < 16) :
  age = List.sum (age_in_columns age) := by
  sorry

end find_age_l193_193907


namespace partial_fraction_product_l193_193346

theorem partial_fraction_product : 
  (∃ A B C : ℚ, 
    (∀ x : ℚ, x ≠ 3 ∧ x ≠ -3 ∧ x ≠ 5 → 
      (x^2 - 21) / ((x - 3) * (x + 3) * (x - 5)) = A / (x - 3) + B / (x + 3) + C / (x - 5))
      ∧ (A * B * C = -1/16)) := 
    sorry

end partial_fraction_product_l193_193346


namespace equation_of_line_projection_l193_193983

theorem equation_of_line_projection (x y : ℝ) (m : ℝ) (x1 x2 : ℝ) (d : ℝ)
  (h1 : (5, 3) ∈ {(x, y) | y = 3 + m * (x - 5)})
  (h2 : x1 = (16 + 20 * m - 12) / (4 * m + 3))
  (h3 : x2 = (1 + 20 * m - 12) / (4 * m + 3))
  (h4 : abs (x1 - x2) = 1) :
  (y = 3 * x - 12 ∨ y = -4.5 * x + 25.5) :=
sorry

end equation_of_line_projection_l193_193983


namespace range_x_l193_193438

theorem range_x (x : ℝ) : (2 * x - 1)⁻² > (x + 1)⁻² → 0 < x ∧ x < 2 ∧ x ≠ 1 / 2 :=
by
  intro h
  sorry

end range_x_l193_193438


namespace students_in_class_l193_193958

theorem students_in_class {S : ℕ} 
  (h1 : 20 < S)
  (h2 : S < 30)
  (chess_club_condition : ∃ (n : ℕ), S = 3 * n) 
  (draughts_club_condition : ∃ (m : ℕ), S = 4 * m) : 
  S = 24 := 
sorry

end students_in_class_l193_193958


namespace triangle_area_ratio_l193_193466

theorem triangle_area_ratio 
  (X Y Z P : Type) -- declare vertices as types
  (hXY : XY = 20) -- declare the length XY
  (hXZ : XZ = 30) -- declare the length XZ
  (hYZ : YZ = 26) -- declare the length YZ
  (hXP_bisector : XP_is_angle_bisector) -- declare XP as an angle bisector
  : (area X Y P) / (area X Z P) = 2 / 3 := 
sorry -- proof is omitted, as instructed

end triangle_area_ratio_l193_193466


namespace second_studio_students_l193_193587

theorem second_studio_students (total_students students_first_studio students_third_studio : ℕ) 
  (h_total : total_students = 376) (h_first : students_first_studio = 110) (h_third : students_third_studio = 131) : 
  total_students - (students_first_studio + students_third_studio) = 135 := 
by {
  rw [h_total, h_first, h_third],
  norm_num,
}

end second_studio_students_l193_193587


namespace ma_and_Ma_l193_193474

variable {a : ℝ} (h_a : 0 < a ∧ a < 1)
variable {f : ℝ → ℝ} (h_f_bij : bijective f) (h_f_inc : ∀ x y, x ≤ y → f x ≤ f y) (h_f_id : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x ∧ f x ≤ 1)

noncomputable def f_inv : ℝ → ℝ := (classical.some (h_f_bij)).inv_fun

theorem ma_and_Ma (h_f_inv : function.left_inverse f_inv f ∧ function.right_inverse f_inv f) :
  a ≤ f a + f_inv a ∧ f a + f_inv a ≤ a + 1 :=
by
  sorry

end ma_and_Ma_l193_193474


namespace total_course_selection_schemes_l193_193263

theorem total_course_selection_schemes (PE_courses : ℕ) (Art_courses : ℕ) : 
  PE_courses = 4 → Art_courses = 4 → 
  (finset.card (finset.powerset_len 2 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b ∈ s), a < PE_courses ∧ b ≥ PE_courses) + 
   finset.card (finset.powerset_len 3 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b c ∈ s), (a < PE_courses ∧ b < PE_courses ∧ c ≥ PE_courses) ∨ 
                        (a < PE_courses ∧ b ≥ PE_courses ∧ c ≥ PE_courses ∧ a ≠ b ≠ c)) = 64 :=
by
  sorry

end total_course_selection_schemes_l193_193263


namespace correct_propositions_l193_193652

-- Definitions of perpendicularity and parallelism for lines and planes
variable {Line Plane : Type}
variable (perpendicular parallel : Line → Plane → Prop)

-- Define relationships in the problem
axiom m : Line
axiom n : Line
axiom alpha : Plane
axiom beta : Plane
axiom gamma : Plane

-- Proposition ①
def prop1 : Prop := (perpendicular m alpha) ∧ (parallel n alpha) → perpendicular m n

-- Proposition ④
def prop4 : Prop := (parallel alpha beta) ∧ (parallel beta gamma) ∧ (perpendicular m alpha) → perpendicular m gamma

-- The final theorem combining both
theorem correct_propositions : prop1 ∧ prop4 := by
  sorry

end correct_propositions_l193_193652


namespace bob_distance_correct_l193_193637

-- Define the problem conditions as constants and variables.
def side_length : ℝ := 3  -- Each side of the octagon measures 3 km.
def walk_distance : ℝ := 7  -- Bob walks a distance of 7 km along the perimeter.

-- Calculate the coordinates after walking.
def coord1 : ℝ × ℝ := (side_length, 0)
def coord2 : ℝ × ℝ := (side_length - (side_length * real.sqrt 2 / 2), (side_length * real.sqrt 2 / 2))
def coord3 : ℝ × ℝ := (side_length - (side_length * real.sqrt 2 / 2), (side_length * real.sqrt 2 / 2) - 1)

-- Define the final distance Bob travels from the starting point.
def bob_distance_from_start : ℝ := real.sqrt ((coord3.1 - 0) ^ 2 + (coord3.2 - 0) ^ 2)

-- The proof statement.
theorem bob_distance_correct : bob_distance_from_start = real.sqrt 26 := by sorry

end bob_distance_correct_l193_193637


namespace correct_calculation_option_l193_193977

theorem correct_calculation_option :
  (∀ a : ℝ, 3 * a^5 - a^5 ≠ 3) ∧
  (∀ a : ℝ, a^2 + a^5 ≠ a^7) ∧
  (∀ a : ℝ, a^5 + a^5 = 2 * a^5) ∧
  (∀ x y : ℝ, x^2 * y + x * y^2 ≠ 2 * x^3 * y^3) :=
by
  sorry

end correct_calculation_option_l193_193977


namespace course_selection_schemes_l193_193195

theorem course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose (n k : ℕ) := nat.choose n k
  
  (choose 4 1 * choose 4 1 + 
   choose 4 2 * choose 4 1 + 
   choose 4 1 * choose 4 2) = 64 := 
by {
  -- calculating the combinations
  have comb_2_courses : choose 4 1 * choose 4 1 = 16 := by sorry,
  have comb_2_pe_1_art : choose 4 2 * choose 4 1 = 24 := by sorry,
  have comb_1_pe_2_art : choose 4 1 * choose 4 2 = 24 := by sorry,
  have total_comb : 16 + 24 + 24 = 64 := by {
    simp [comb_2_courses, comb_2_pe_1_art, comb_1_pe_2_art]
  },
  exact total_comb
}

end course_selection_schemes_l193_193195


namespace range_of_k1k2k3_l193_193752

-- Define the given circles and their intersections
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8 * x - 8 * y + 28 = 0
def curve_N (x y : ℝ) : Prop := x^2 + 4 * y^2 = 4
def point_A (x y : ℝ) : Prop := (x, y) = (2, 0)
def point_B (x y : ℝ) : Prop := (x, y) = (-2, 0)

-- Define points and lines with their slopes
def line_OP (x y k : ℝ) : Prop := y = k * x
def slope_MA (b a : ℝ) : ℝ := b / (a + 2)
def slope_MB (b a : ℝ) : ℝ := b / (a - 2)

-- Condition on the point M belonging to curve_N
def point_M_on_curve_N (a b : ℝ) : Prop := a^2 + 4 * b^2 = 4

-- Define ranges for the slopes
noncomputable def k1_k2 (b a : ℝ) : ℝ := (b / (a + 2)) * (b / (a - 2)) = -1/4

-- Define the quadratic equation for the slope range
def k3_range (k : ℝ) : Prop := (4 - Real.sqrt 7) / 3 ≤ k ∧ k ≤ (4 + Real.sqrt 7) / 3

theorem range_of_k1k2k3 (k1 k2 k3 : ℝ) (hz : k1 * k2 = -1/4) :
  ( (4 - Real.sqrt 7) / 12 ≤ hz * k3 ∧ hz * k3 ≤ (4 + Real.sqrt 7) / 12 ) :=
sorry

end range_of_k1k2k3_l193_193752


namespace problem1_problem2_l193_193797

-- Restated Problem 1: Prove the range of values of x satisfying |x^2 - 1| > 1 is (-∞, -√2) ∪ (√2, ∞).
theorem problem1 (x : ℝ) : |x^2 - 1| > 1 ↔ x ∈ set.Iio (-real.sqrt 2) ∪ set.Ioi (real.sqrt 2) :=
by
  sorry

-- Restated Problem 2: For any two distinct positive numbers a and b, prove that |a^3 + b^3 - 2ab√ab| > |a^2b + ab^2 - 2ab√ab|.
theorem problem2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) : 
  |a^3 + b^3 - 2 * a * b * real.sqrt (a * b)| > |a^2 * b + a * b^2 - 2 * a * b * real.sqrt (a * b)| :=
by
  sorry

end problem1_problem2_l193_193797


namespace people_relationship_l193_193999

def rows := ℕ
def number_of_people (x : rows) : ℕ := x + 39

theorem people_relationship (x : rows) (h1 : 1 ≤ x ∧ x ≤ 60) : 
  number_of_people x = x + 39 := by
sorry

end people_relationship_l193_193999


namespace min_shadow_area_l193_193287

-- Define the given conditions
variables (a b : ℝ) (h : b > a)

-- Define the problem statement
theorem min_shadow_area (h : b > a) : 
  let x := (a * b) / (b - a) in
  let A := x^2 in
  A = (a * b / (b - a))^2 :=
sorry

end min_shadow_area_l193_193287


namespace p_necessary_not_sufficient_for_p_and_q_l193_193565

-- Define statements p and q as propositions
variables (p q : Prop)

-- Prove that "p is true" is a necessary but not sufficient condition for "p ∧ q is true"
theorem p_necessary_not_sufficient_for_p_and_q : (p ∧ q → p) ∧ (p → ¬ (p ∧ q)) :=
by sorry

end p_necessary_not_sufficient_for_p_and_q_l193_193565


namespace total_selection_schemes_l193_193284

-- Define the given conditions
def num_phys_ed_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_phys_ed_courses + num_art_courses
def valid_course_combos : finset ℕ := {2, 3}

-- Define the number of selection schemes with the constraints
def selection_schemes : nat :=
  (num_phys_ed_courses.choose 1 * num_art_courses.choose 1) + 
  (num_phys_ed_courses.choose 2 * num_art_courses.choose 1 + 
   num_phys_ed_courses.choose 1 * num_art_courses.choose 2)

-- State the theorem to be proved
theorem total_selection_schemes : selection_schemes = 64 := by
  sorry

end total_selection_schemes_l193_193284


namespace bridge_length_l193_193647

noncomputable def train_length : ℝ := 360
noncomputable def train_speed_kph : ℝ := 30
noncomputable def passing_time_seconds : ℝ := 60

theorem bridge_length :
  let train_speed_m_s := train_speed_kph * 1000 / 3600,
      total_distance := train_speed_m_s * passing_time_seconds,
      bridge_length := total_distance - train_length
  in bridge_length = 139.8 :=
by
  sorry

end bridge_length_l193_193647


namespace OK_eq_KB_l193_193734

-- Definitions of given entities and conditions
structure GeometrySetup where
  O A B C E K : Type
  circle : Set O  -- The circle centered at O
  tangentA : A ∈ circle -- Point A is tangent to the circle
  tangentB : B ∈ circle -- Point B is tangent to the circle
  rayA_parOB : ∀ (lineOA : Line A O) (lineOB : Line O B), Parallel lineOA lineOB /- Ray starting at A is parallel to OB -/
  C_on_circle : C ∈ circle  -- Point C lies on the circle where ray from A intersects
  E_on_OC : E ∈ Seg O C  -- Point E lies on segment OC
  K_intersect : ∀ (lineAE : Line A E) (lineOB : Line O B), 
                K ∈ (lineAE ∩ lineOB)  -- Lines AE and OB intersect at point K

theorem OK_eq_KB {setup : GeometrySetup} : 
  distance setup.O setup.K = distance setup.K setup.B := 
sorry -- directly given required conclusion

end OK_eq_KB_l193_193734


namespace max_deviation_temperature_l193_193444

-- Define the temperature conversion function from Fahrenheit to Celsius
def fahrenheitToCelsius (F : ℝ) : ℝ := (F - 32) * (5 / 9)

-- State the theorem about the maximum deviation
theorem max_deviation_temperature :
  ∀ F : ℝ,
  let rounded_F := Real.floor (F + 0.5)
  let C := fahrenheitToCelsius F
  let rounded_C := Real.floor (C + 0.5)
  abs (fahrenheitToCelsius rounded_F - rounded_C) ≤ 13 / 18 := sorry

end max_deviation_temperature_l193_193444


namespace product_of_integers_is_correct_l193_193128

theorem product_of_integers_is_correct (a b c d e : ℤ)
  (h_pairwise_sums : Multiset {x : ℤ // ∃ i j, [i, j] ∈ ([a, b, c, d, e].pair_combinations) ∧ x = i + j } = {-1, 4, 6, 9, 10, 11, 15, 16, 20, 22}.noprob) :
  a * b * c * d * e = -4914 :=        
sorry

end product_of_integers_is_correct_l193_193128


namespace total_selection_schemes_l193_193277

-- Define the given conditions
def num_phys_ed_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_phys_ed_courses + num_art_courses
def valid_course_combos : finset ℕ := {2, 3}

-- Define the number of selection schemes with the constraints
def selection_schemes : nat :=
  (num_phys_ed_courses.choose 1 * num_art_courses.choose 1) + 
  (num_phys_ed_courses.choose 2 * num_art_courses.choose 1 + 
   num_phys_ed_courses.choose 1 * num_art_courses.choose 2)

-- State the theorem to be proved
theorem total_selection_schemes : selection_schemes = 64 := by
  sorry

end total_selection_schemes_l193_193277


namespace total_selection_schemes_l193_193275

-- Define the given conditions
def num_phys_ed_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_phys_ed_courses + num_art_courses
def valid_course_combos : finset ℕ := {2, 3}

-- Define the number of selection schemes with the constraints
def selection_schemes : nat :=
  (num_phys_ed_courses.choose 1 * num_art_courses.choose 1) + 
  (num_phys_ed_courses.choose 2 * num_art_courses.choose 1 + 
   num_phys_ed_courses.choose 1 * num_art_courses.choose 2)

-- State the theorem to be proved
theorem total_selection_schemes : selection_schemes = 64 := by
  sorry

end total_selection_schemes_l193_193275


namespace derivative_of_y_l193_193175

noncomputable def y (x : ℝ) : ℝ := (1 - Real.cos (2 * x)) / (1 + Real.cos (2 * x))
noncomputable def dy_dx (x : ℝ) : ℝ := deriv y x

-- Theorem statement and proof placeholder
theorem derivative_of_y (x : ℝ) : dy_dx x = (4 * Real.sin (2 * x)) / ((1 + Real.cos (2 * x))^2) :=
sorry

end derivative_of_y_l193_193175


namespace regular_ngon_integer_coords_l193_193717

theorem regular_ngon_integer_coords (n : ℕ) (h : n ≥ 3) :
  (∃ (vertices : Fin n → ℤ × ℤ), is_regular_ngon vertices n) ↔ n = 4 :=
sorry

end regular_ngon_integer_coords_l193_193717


namespace B_values_for_divisibility_l193_193105

theorem B_values_for_divisibility (B : ℕ) (h : 4 + B + B + B + 2 ≡ 0 [MOD 9]) : B = 1 ∨ B = 4 ∨ B = 7 :=
by sorry

end B_values_for_divisibility_l193_193105


namespace luke_bike_vs_bus_slowness_l193_193496

theorem luke_bike_vs_bus_slowness
  (luke_bus_time : ℕ)
  (paula_ratio : ℚ)
  (total_travel_time : ℕ)
  (paula_total_bus_time : ℕ)
  (luke_total_travel_time_lhs : ℕ)
  (luke_total_travel_time_rhs : ℕ)
  (bike_time : ℕ)
  (ratio : ℚ) :
  luke_bus_time = 70 ∧
  paula_ratio = 3 / 5 ∧
  total_travel_time = 504 ∧
  paula_total_bus_time = 2 * (paula_ratio * luke_bus_time) ∧
  luke_total_travel_time_lhs = luke_bus_time + bike_time ∧
  luke_total_travel_time_rhs + paula_total_bus_time = total_travel_time ∧
  bike_time = ratio * luke_bus_time ∧
  ratio = bike_time / luke_bus_time →
  ratio = 5 :=
sorry

end luke_bike_vs_bus_slowness_l193_193496


namespace boat_speed_in_still_water_l193_193625

theorem boat_speed_in_still_water :
  ∀ (V_b V_s : ℝ) (distance time : ℝ),
  V_s = 5 →
  time = 4 →
  distance = 84 →
  (distance / time) = V_b + V_s →
  V_b = 16 :=
by
  -- Given definitions and values
  intros V_b V_s distance time
  intro hV_s
  intro htime
  intro hdistance
  intro heq
  sorry -- Placeholder for the actual proof

end boat_speed_in_still_water_l193_193625


namespace percentile_45_is_78_normal_distribution_probability_l193_193980

def dataSet : List ℕ := [91, 72, 75, 85, 64, 92, 76, 78, 86, 79]

theorem percentile_45_is_78 : 
  let sortedData := dataSet.qsort (≤)
  let position := (dataSet.length * 45) / 100
  dataSet.nth! position = 78 := 
by
  let sortedData := [64, 72, 75, 76, 78, 79, 85, 86, 91, 92]
  let position := 4.5.ceil
  have h_pos : position = 5 := rfl —round up from 4.5
  rw [h_pos]
  exact rfl

def standard_normal (ξ : ℝ → Prop) := \forall x, ξ x = exp (-x*x/2)

theorem normal_distribution_probability (p : ℝ) (ξ : ℝ → Prop) : 
  standard_normal ξ →
  P(ξ > 1) = p →
  P(-1 ≤ ξ ∧ ξ ≤ 0) = 1 / 2 - p :=
by
  intro h_std h_p
  have : \forall x:Real, ξ x = exp (-x*x/2) := h_std
  calc
    P(-1 ≤ ξ ∧ ξ ≤ 0) = P(1 - 1/2 - p) := by sorry


end percentile_45_is_78_normal_distribution_probability_l193_193980


namespace course_selection_schemes_l193_193204

theorem course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose (n k : ℕ) := nat.choose n k
  
  (choose 4 1 * choose 4 1 + 
   choose 4 2 * choose 4 1 + 
   choose 4 1 * choose 4 2) = 64 := 
by {
  -- calculating the combinations
  have comb_2_courses : choose 4 1 * choose 4 1 = 16 := by sorry,
  have comb_2_pe_1_art : choose 4 2 * choose 4 1 = 24 := by sorry,
  have comb_1_pe_2_art : choose 4 1 * choose 4 2 = 24 := by sorry,
  have total_comb : 16 + 24 + 24 = 64 := by {
    simp [comb_2_courses, comb_2_pe_1_art, comb_1_pe_2_art]
  },
  exact total_comb
}

end course_selection_schemes_l193_193204


namespace leak_drain_time_l193_193165

noncomputable def pump_rate : ℚ := 1/2
noncomputable def leak_empty_rate : ℚ := 1 / (1 / pump_rate - 5/11)

theorem leak_drain_time :
  let pump_rate := 1/2
  let combined_rate := 5/11
  let leak_rate := pump_rate - combined_rate
  1 / leak_rate = 22 :=
  by
    -- Definition of pump rate
    let pump_rate := 1/2
    -- Definition of combined rate
    let combined_rate := 5/11
    -- Definition of leak rate
    let leak_rate := pump_rate - combined_rate
    -- Calculate leak drain time
    show 1 / leak_rate = 22
    sorry

end leak_drain_time_l193_193165


namespace negative_values_of_x_l193_193378

theorem negative_values_of_x : 
  let f (x : ℤ) := Int.sqrt (x + 196)
  ∃ (n : ℕ), (f (n ^ 2 - 196) > 0 ∧ f (n ^ 2 - 196) = n) ∧ ∃ k : ℕ, k = 13 :=
by
  sorry

end negative_values_of_x_l193_193378


namespace course_selection_schemes_l193_193201

theorem course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose (n k : ℕ) := nat.choose n k
  
  (choose 4 1 * choose 4 1 + 
   choose 4 2 * choose 4 1 + 
   choose 4 1 * choose 4 2) = 64 := 
by {
  -- calculating the combinations
  have comb_2_courses : choose 4 1 * choose 4 1 = 16 := by sorry,
  have comb_2_pe_1_art : choose 4 2 * choose 4 1 = 24 := by sorry,
  have comb_1_pe_2_art : choose 4 1 * choose 4 2 = 24 := by sorry,
  have total_comb : 16 + 24 + 24 = 64 := by {
    simp [comb_2_courses, comb_2_pe_1_art, comb_1_pe_2_art]
  },
  exact total_comb
}

end course_selection_schemes_l193_193201


namespace subset_S_T_l193_193046

def S : Set (ℝ × ℝ) := {p | (p.1 ^ 2 - p.2 ^ 2) % 2 = 1}
def T : Set (ℝ × ℝ) := {p | sin (2 * π * p.1 ^ 2) - sin (2 * π * p.2 ^ 2) = cos (2 * π * p.1 ^ 2) - cos (2 * π * p.2 ^ 2)}

theorem subset_S_T : S ⊆ T := by
  sorry

end subset_S_T_l193_193046


namespace trains_clear_time_l193_193992

noncomputable def time_for_trains_to_pass (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let relative_speed := (speed1 + speed2) * (1000 / 3600) -- Convert kmph to m/s
  let total_length := length1 + length2
  total_length / relative_speed

theorem trains_clear_time (length1 length2 : ℝ) (speed1 speed2 : ℝ) :
  length1 = 120 →
  length2 = 320 →
  speed1 = 42 →
  speed2 = 30 →
  time_for_trains_to_pass length1 length2 speed1 speed2 = 22 :=
begin
  intros h_length1 h_length2 h_speed1 h_speed2,
  subst h_length1,
  subst h_length2,
  subst h_speed1,
  subst h_speed2,
  simp [time_for_trains_to_pass],
  norm_num,
end

end trains_clear_time_l193_193992


namespace total_course_selection_schemes_l193_193225

theorem total_course_selection_schemes (
  pe_courses art_courses : Finset ℕ
) : 
  pe_courses.card = 4 →
  art_courses.card = 4 →
  let total_schemes := 
    (pe_courses.card.choose 1 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 2 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 1 * art_courses.card.choose 2) in
  total_schemes = 64 := 
by
  intros h1 h2
  let case1 := pe_courses.card.choose 1 * art_courses.card.choose 1 -- 4 * 4
  let case2_1 := pe_courses.card.choose 2 * art_courses.card.choose 1 -- 6 * 4
  let case2_2 := pe_courses.card.choose 1 * art_courses.card.choose 2 -- 4 * 6
  let case2 := case2_1 + case2_2 -- 24 + 24
  let total_schemes := case1 + case2 -- 16 + 48
  have hcard := by
    simp [Nat.choose]
    exact h1
  have hcase1 : case1 = 16 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_1 : case2_1 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_2 : case2_2 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2 : case2 = 48 := by {
    rw [hcase2_1, hcase2_2]
    norm_num
  }
  have htotal_schemes : total_schemes = 64 := by {
    rw [hcase1, hcase2]
    norm_num
  }
  exact htotal_schemes

end total_course_selection_schemes_l193_193225


namespace find_m_value_l193_193798

noncomputable theory
open Real

-- Define the conditions
def point_distance_condition (m : ℝ) : Prop :=
  abs (-3 * m - 1) = abs (-2 * m)

-- Define the target result
def correct_values (m : ℝ) : Prop :=
  m = -1 ∨ m = -1/5

-- State the theorem
theorem find_m_value (m : ℝ) (h : point_distance_condition m) : correct_values m := 
sorry

end find_m_value_l193_193798


namespace each_charity_gets_75_l193_193846

theorem each_charity_gets_75 
  (cookies_each := 6 * 12) 
  (brownies_each := 4 * 12)
  (muffins_each := 3 * 12)
  (price_cookie := 1.5)
  (price_brownie := 2)
  (price_muffin := 2.5)
  (cost_cookie := 0.25)
  (cost_brownie := 0.5)
  (cost_muffin := 0.75)
  (charities := 3) : 
  let revenue := (cookies_each * price_cookie) + (brownies_each * price_brownie) + (muffins_each * price_muffin),
      cost := (cookies_each * cost_cookie) + (brownies_each * cost_brownie) + (muffins_each * cost_muffin),
      profit := revenue - cost,
      per_charity := profit / charities in
  per_charity = 75 := by
  sorry

end each_charity_gets_75_l193_193846


namespace possible_values_of_expression_l193_193753

theorem possible_values_of_expression (x y : ℝ) (hxy : x + 2 * y = 2) (hx_pos : x > 0) (hy_pos : y > 0) :
  ∃ v, v = 21 / 4 ∧ (1 / x + 2 / y) = v :=
sorry

end possible_values_of_expression_l193_193753


namespace value_of_b_l193_193336

theorem value_of_b (b x y : ℝ) (h1 : sqrt (x * y) = b ^ b) (h2 : log b (x ^ (log b y)) + log b (y ^ (log b x)) = 4 * b ^ 4) : 
  0 < b ∧ b ≤ 1 / sqrt 2 := 
sorry

end value_of_b_l193_193336


namespace find_positive_integers_n_l193_193358

def number_of_digits (n : ℕ) : ℕ := (n.toString.length)

def prime_factors (n : ℕ) : List ℕ := 
  List.filter Prime (n.factors)

def sum_exponents (n : ℕ) : ℕ :=
  (n.factors.map (λ p, (n.factorization.find p).getD 0)).sum

def satisfies_conditions (n : ℕ) : Prop :=
  number_of_digits n = prime_factors n.length ∧ 
  prime_factors n.sum = sum_exponents n

theorem find_positive_integers_n :
  {n : ℕ // satisfies_conditions n} = {4, 48, 72} :=
by
  sorry

end find_positive_integers_n_l193_193358


namespace count_valid_x_correct_l193_193435

noncomputable def count_valid_x : ℕ := 
  (finset.Icc 250 333).card

theorem count_valid_x_correct :
  count_valid_x = 84 :=
by
  sorry

end count_valid_x_correct_l193_193435


namespace find_D_l193_193905

theorem find_D (D E F : ℝ) (h : ∀ x : ℝ, x ≠ 1 → x ≠ -2 → (1 / (x^3 - 3*x^2 - 4*x + 12)) = (D / (x - 1)) + (E / (x + 2)) + (F / (x + 2)^2)) :
    D = -1 / 15 :=
by
  -- the proof is omitted as per the instructions
  sorry

end find_D_l193_193905


namespace problem_A_problem_B_problem_C_problem_D_l193_193981

-- Proof Problem 1
theorem problem_A (n : ℕ) (p : ℚ) (X : ℕ →. ℚ) (h1 : X = (λ n, n * p)) (h2 : (λ n, n * p * (1 - p)) = 20) : p = 1 / 3 :=
sorry

-- Proof Problem 2
theorem problem_B (data : list ℕ) (sorted_data := list.sorted data) (k := 5) (percentile := list.nth sorted_data (k - 1)) (h1 : data = [91, 72, 75, 85, 64, 92, 76, 78, 86, 79]) (h2 : percentile = some 78) : true :=
sorry

-- Proof Problem 3
theorem problem_C (ξ : ℝ →. ℝ) (p : ℝ) (h1 : ξ = (λ 0 1), 1 - p) : (P (-1 ≤ ξ ≤ 0) = 1 / 2 - p) :=
sorry

-- Proof Problem 4
theorem problem_D (students11 students12 total : ℕ) (selected11 selected12 : ℕ) (h1 : students11 = 400) (h2 : students12 = 360) (h3 : total = 57) (h4 : selected11 = 20) (sampled_ratio := selected11 / students11) (h5 : sampled_ratio = selected12 / students12) : selected12 = 18 :=
sorry

end problem_A_problem_B_problem_C_problem_D_l193_193981


namespace diana_prob_higher_than_apollo_sum_l193_193691

def die_rolls : Finset ℕ := {1, 2, 3, 4, 5, 6}
def apollo_sums : Finset ℕ := (Finset.product die_rolls die_rolls).image (λ p, p.1 + p.2)

theorem diana_prob_higher_than_apollo_sum :
  (∑ d in die_rolls, ∑ s in apollo_sums, if d > s then 1 else 0).toReal / (6 * 36) = 5 / 108 := 
  sorry

end diana_prob_higher_than_apollo_sum_l193_193691


namespace perpendicular_AK_BC_l193_193008

open EuclideanGeometry

noncomputable theory

variable {A B C D E F K : Point}

-- Let’s define the configurations given in the conditions
structure angle_bisector (A B C D : Point) : Prop :=
(bisects : angle_eq (angle A B D) (angle A D C))

structure perpendicular (p1 p2: Point) (p3 p4 : Line) : Prop :=
(is_perpendicular : angle_eq (angle p1 p2 p3) (pi.div 2))

structure intersects (p1 p2 : Line) (p : Point) : Prop :=
(intersect : ∃ p, p1.contains p ∧ p2.contains p)

def acute_triangle (A B C : Point) : Prop :=
  ∀ (a b c : R), a + b + c = pi 

-- Given conditions formalized into Lean 4 structures.
variables (h₁ : acute_triangle A B C)
          (h₂ : D ∈ (line_through B C))
          (h₃ : angle_bisector A B C D)
          (h₄ : perpendicular D F (line_through D F) (line_through A C))
          (h₅ : perpendicular D E (line_through D E) (line_through A B))
          (h₆ : intersects (line_through B F) (line_through C E) K)

-- Theorem that we need to prove.
theorem perpendicular_AK_BC : perpendicular A K (line_through A K) (line_through B C) :=
  sorry

end perpendicular_AK_BC_l193_193008


namespace chord_length_ne_l193_193760

-- Define the ellipse
def ellipse (x y : ℝ) := (x^2 / 8) + (y^2 / 4) = 1

-- Define the first line
def line_l (k x : ℝ) := (k * x + 1)

-- Define the second line
def line_l_option_D (k x y : ℝ) := (k * x + y - 2)

-- Prove the chord length inequality for line_l_option_D
theorem chord_length_ne (k : ℝ) :
  ∀ x y : ℝ, ellipse x y →
  ∃ x1 x2 y1 y2 : ℝ, ellipse x1 y1 ∧ line_l k x1 = y1 ∧ ellipse x2 y2 ∧ line_l k x2 = y2 ∧
  ∀ x3 x4 y3 y4 : ℝ, ellipse x3 y3 ∧ line_l_option_D k x3 y3 = 0 ∧ ellipse x4 y4 ∧ line_l_option_D k x4 y4 = 0 →
  dist (x1, y1) (x2, y2) ≠ dist (x3, y3) (x4, y4) :=
sorry

end chord_length_ne_l193_193760


namespace measure_angle_ACB_l193_193010

-- Definitions of angles and the conditions
def angle_ABD := 140
def angle_BAC := 105
def supplementary_angle (α β : ℕ) := α + β = 180
def angle_sum_property (α β γ : ℕ) := α + β + γ = 180

-- Theorem to prove the measure of angle ACB
theorem measure_angle_ACB (angle_ABD : ℕ) 
                         (angle_BAC : ℕ) 
                         (h1 : supplementary_angle angle_ABD 40)
                         (h2 : angle_sum_property 40 angle_BAC 35) :
  angle_sum_property 40 105 35 :=
sorry

end measure_angle_ACB_l193_193010


namespace probability_of_green_is_correct_l193_193679

structure ContainerBalls :=
  (red : ℕ)
  (green : ℕ)

def ContainerA : ContainerBalls := ⟨3, 5⟩
def ContainerB : ContainerBalls := ⟨5, 5⟩
def ContainerC : ContainerBalls := ⟨7, 3⟩
def ContainerD : ContainerBalls := ⟨4, 6⟩

noncomputable def probability_green_ball (c : ContainerBalls) : ℚ :=
  c.green / (c.red + c.green)

noncomputable def total_probability_green : ℚ :=
  (1 / 4) * probability_green_ball ContainerA +
  (1 / 4) * probability_green_ball ContainerB +
  (1 / 4) * probability_green_ball ContainerC +
  (1 / 4) * probability_green_ball ContainerD

theorem probability_of_green_is_correct : total_probability_green = 81 / 160 := 
  sorry

end probability_of_green_is_correct_l193_193679


namespace num_valid_arrangements_l193_193085

def valid_arrangements : Nat := 5! - 2 * 4! + 3!

theorem num_valid_arrangements : 
  valid_arrangements = 78 :=
by
  unfold valid_arrangements
  rw [Nat.factorial_succ, Nat.factorial_succ, Nat.factorial_three, Nat.factorial_four]
  calc
    5 * 24 - 2 * 24 + 6 = 120 - 48 + 6 : by norm_num
    ... = 78 : by norm_num
  sorry

end num_valid_arrangements_l193_193085


namespace largest_interval_includes_2_invertible_l193_193340

-- Defining the quadratic function g
def g (x : ℝ) := 3 * x^2 - 9 * x + 4

-- The interval [3/2, ∞)
def interval := set.Ici (3 / 2 : ℝ)

-- The proof goal
theorem largest_interval_includes_2_invertible :
  (2 : ℝ) ∈ interval ∧ ∀ x1 x2 ∈ interval, g x1 = g x2 → x1 = x2 := 
by 
  sorry

end largest_interval_includes_2_invertible_l193_193340


namespace log_function_fixed_point_l193_193155

theorem log_function_fixed_point (a : ℝ) (x y : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  (y = log a (x + 2) + 1) → (x = -1 ∧ y = 1) :=
sorry

end log_function_fixed_point_l193_193155


namespace no_real_roots_l193_193640

def P (n : ℕ) : Polynomial ℝ :=
  match n with
  | 0 => 1
  | k + 1 => X^(17 * (k + 1)) - P k

theorem no_real_roots (n : ℕ) : ¬ ∃ x : ℝ, (P n).eval x = 0 := 
sorry

end no_real_roots_l193_193640


namespace perimeter_of_ADEF_l193_193804

variable (A B C D E F : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
variable (AB AC BC AD AE AF BD BE BF DE EF : ℝ)
variable (BAC : Angles)

-- Conditions
axiom AB_eq_20 : AB = 20
axiom AC_eq_24 : AC = 24
axiom BC_eq_18 : BC = 18
axiom angle_BAC_eq_60 : BAC = 60
axiom DE_parallel_AC : ParallelLine D E A C
axiom EF_parallel_AB : ParallelLine E F A B

def perimeter_ADEF := AD + DE + EF + AF

theorem perimeter_of_ADEF : perimeter_ADEF AD DE EF AF = 44 := by
  -- proof skipped
  sorry

end perimeter_of_ADEF_l193_193804


namespace course_selection_count_l193_193224

-- Definitions for the conditions
def num_PE_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_PE_courses + num_art_courses

-- The number of different course selection schemes
theorem course_selection_count : 
  (choose num_PE_courses 1) * (choose num_art_courses 1) + 
  (choose num_PE_courses 2) * (choose num_art_courses 1) + 
  (choose num_PE_courses 1) * (choose num_art_courses 2) = 64 := 
by sorry

end course_selection_count_l193_193224


namespace abs_discriminant_inequality_l193_193727

theorem abs_discriminant_inequality 
  (a b c A B C : ℝ) 
  (ha : a ≠ 0) 
  (hA : A ≠ 0) 
  (h : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) : 
  |b^2 - 4 * a * c| ≤ |B^2 - 4 * A * C| :=
sorry

end abs_discriminant_inequality_l193_193727


namespace course_selection_schemes_count_l193_193237

-- Definitions based on the conditions
def num_physical_education_courses : ℕ := 4
def num_art_courses : ℕ := 4

-- Required to choose 2 or 3 courses, with at least one from each category
def valid_selection_cases : list (ℕ × ℕ) := [(1, 1), (1, 2), (2, 1)]

-- Calculate the number of ways to choose k courses from n courses using combination formula
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Calculate the total number of different course selection schemes
def total_course_selection_schemes : ℕ :=
  let case1 := C num_physical_education_courses 1 * C num_art_courses 1
  let case2a := C num_physical_education_courses 2 * C num_art_courses 1
  let case2b := C num_physical_education_courses 1 * C num_art_courses 2
  case1 + case2a + case2b

-- The assertion that the total number of different course selection schemes is 64
theorem course_selection_schemes_count : total_course_selection_schemes = 64 := by
  -- sorry placeholder for proof
  sorry

end course_selection_schemes_count_l193_193237


namespace boys_meet_once_excluding_start_and_finish_l193_193135

-- Define the initial conditions as constants
def speeds := (6 : ℝ, 10 : ℝ)
def starting_point := 'A'

-- Define the problem statement: They meet 1 time, excluding start and finish
theorem boys_meet_once_excluding_start_and_finish (d : ℝ) :
  ∃ t : ℝ, ∀ t₀ : ℝ, (t₀ > 0 ∧ t₀ < d / (lcm speeds.1 speeds.2)) →
  (t₀ / (speeds.1 * speeds.2)).floor % (lcm speeds.1 speeds.2 / (speeds.1 + speeds.2)) = 1 :=
sorry

end boys_meet_once_excluding_start_and_finish_l193_193135


namespace simplify_expression_as_single_fraction_l193_193608

variable (d : ℚ)

theorem simplify_expression_as_single_fraction :
  (5 + 4*d)/9 + 3 = (32 + 4*d)/9 := 
by
  sorry

end simplify_expression_as_single_fraction_l193_193608


namespace min_fraction_l193_193750

noncomputable def x := 1099

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def sum_of_digits (n : ℕ) : ℕ :=
  n.divmod 1000 % 10 + 
  n.divmod 100 % 10 + 
  n.divmod 10 % 10 + 
  n % 10

theorem min_fraction (x : ℕ) (y : ℕ) (h₁ : is_four_digit x) (h₂ : sum_of_digits x = y) :
  x = 1099 :=
sorry

end min_fraction_l193_193750


namespace area_of_WXYZ_l193_193005

structure Quadrilateral (α : Type _) :=
  (W : α) (X : α) (Y : α) (Z : α)
  (WZ ZW' WX XX' XY YY' YZ Z'W : ℝ)
  (area_WXYZ : ℝ)

theorem area_of_WXYZ' (WXYZ : Quadrilateral ℝ) 
  (h1 : WXYZ.WZ = 10) 
  (h2 : WXYZ.ZW' = 10)
  (h3 : WXYZ.WX = 6)
  (h4 : WXYZ.XX' = 6)
  (h5 : WXYZ.XY = 7)
  (h6 : WXYZ.YY' = 7)
  (h7 : WXYZ.YZ = 12)
  (h8 : WXYZ.Z'W = 12)
  (h9 : WXYZ.area_WXYZ = 15) : 
  ∃ area_WXZY' : ℝ, area_WXZY' = 45 :=
sorry

end area_of_WXYZ_l193_193005


namespace selling_price_l193_193301

theorem selling_price (profit_percent : ℝ) (cost_price : ℝ) (h_profit : profit_percent = 5) (h_cp : cost_price = 2400) :
  let profit := (profit_percent / 100) * cost_price 
  let selling_price := cost_price + profit
  selling_price = 2520 :=
by
  sorry

end selling_price_l193_193301


namespace root_in_interval_l193_193504

-- Given conditions
variables {a b c : ℝ}
variable {P : ℝ → ℝ}
def polynomial := P = λ x, x^3 + a * x^2 + b * x + c

-- Main statement to be proved
theorem root_in_interval (h_poly : polynomial) (h_roots : ∀ r : ℝ, P r = 0 → r = r) (h_sum : -2 ≤ a + b + c ∧ a + b + c ≤ 0) :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ P x = 0 :=
sorry

end root_in_interval_l193_193504


namespace planting_equation_l193_193892

def condition1 (x : ℕ) : ℕ := 5 * x + 3
def condition2 (x : ℕ) : ℕ := 6 * x - 4

theorem planting_equation (x : ℕ) : condition1 x = condition2 x := by
  sorry

end planting_equation_l193_193892


namespace total_course_selection_schemes_l193_193233

theorem total_course_selection_schemes (
  pe_courses art_courses : Finset ℕ
) : 
  pe_courses.card = 4 →
  art_courses.card = 4 →
  let total_schemes := 
    (pe_courses.card.choose 1 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 2 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 1 * art_courses.card.choose 2) in
  total_schemes = 64 := 
by
  intros h1 h2
  let case1 := pe_courses.card.choose 1 * art_courses.card.choose 1 -- 4 * 4
  let case2_1 := pe_courses.card.choose 2 * art_courses.card.choose 1 -- 6 * 4
  let case2_2 := pe_courses.card.choose 1 * art_courses.card.choose 2 -- 4 * 6
  let case2 := case2_1 + case2_2 -- 24 + 24
  let total_schemes := case1 + case2 -- 16 + 48
  have hcard := by
    simp [Nat.choose]
    exact h1
  have hcase1 : case1 = 16 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_1 : case2_1 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_2 : case2_2 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2 : case2 = 48 := by {
    rw [hcase2_1, hcase2_2]
    norm_num
  }
  have htotal_schemes : total_schemes = 64 := by {
    rw [hcase1, hcase2]
    norm_num
  }
  exact htotal_schemes

end total_course_selection_schemes_l193_193233


namespace simplify_and_evaluate_l193_193896

theorem simplify_and_evaluate (x y : ℝ) (h_x : x = sqrt 3 + 1) (h_y : y = sqrt 3) :
  ( ( (3 * x + y) / (x^2 - y^2) + (2 * x) / (y^2 - x^2) ) / (2 / (x^2 * y - x * y^2)) ) = (3 + sqrt 3) / 2 :=
by
  sorry

end simplify_and_evaluate_l193_193896


namespace simple_interest_rate_l193_193168

theorem simple_interest_rate (P A : ℝ) (T : ℕ) (R : ℝ) 
  (P_pos : P = 800) (A_pos : A = 950) (T_pos : T = 5) :
  R = 3.75 :=
by
  sorry

end simple_interest_rate_l193_193168


namespace inverse_function_of_exponential_l193_193748

noncomputable def f (x : ℝ) : ℝ :=
  3 ^ x

theorem inverse_function_of_exponential (x: ℝ) : 
  Function.LeftInverse (λ y, Real.log y / Real.log 3) f :=
begin
  intro y,
  have : f(2) = 9,
  { 
    simp [f],
    ring,
  },
  sorry -- Placeholder for the proof
end

end inverse_function_of_exponential_l193_193748


namespace find_a_range_l193_193427

-- Definitions of sets A, B, C, and their complements
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}
def B : Set ℝ := {x : ℝ | 3x - 1 < x + 5}
def C (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}
def complement_A : Set ℝ := {x : ℝ | x ≤ 1 ∨ x ≥ 4}

-- The proof problem statement
theorem find_a_range (a : ℝ) (h : complement_A ∩ C a = C a) : a ≤ 1 :=
sorry

end find_a_range_l193_193427


namespace scientific_notation_correct_l193_193522

-- Define the original number
def original_number : ℝ := 0.00000036

-- Define the scientific notation form
def scientific_notation_form (a : ℝ) (n : ℤ) : ℝ := a * 10 ^ n

-- Define the significant figures and the exponent for scientific notation
def a : ℝ := 3.6
def n : ℤ := -7

-- State the theorem
theorem scientific_notation_correct : 
  scientific_notation_form a n = original_number :=
by sorry

end scientific_notation_correct_l193_193522


namespace second_orange_marker_probability_l193_193662

-- Define the conditions
def markers := ℕ
def is_blue (m : markers) : Prop := m = 0
def is_orange (m : markers) : Prop := m = 1

def configurations :=
  {s : list markers // s.length = 3 ∧ 
    (∀ m ∈ s, is_blue m ∨ is_orange m)}

def prob_config (c : configurations) : ℚ :=
  match c with
  | ⟨[0, 0, 0], _⟩ := 1/8
  | ⟨[0, 0, 1], _⟩ := 3/8
  | ⟨[0, 1, 1], _⟩ := 3/8
  | ⟨[1, 1, 1], _⟩ := 1/8
  | _ := 0

-- Define the probability of drawing an orange marker from a configuration after adding an additional orange marker
def prob_draw_orange (c : configurations) : ℚ :=
  match c with
  | ⟨[0, 0, 0], _⟩ := 1/4
  | ⟨[0, 0, 1], _⟩ := 1/2
  | ⟨[0, 1, 1], _⟩ := 3/4
  | ⟨[1, 1, 1], _⟩ := 1
  | _ := 0

-- Define the conditional probability
def prob_second_orange_given_first_orange : ℚ :=
  (3/16 + 9/32 + 1/8) / (1/32 + 3/16 + 9/32 + 1/8)

-- Finally, prove that the probability equals 19/20
theorem second_orange_marker_probability :
  prob_second_orange_given_first_orange = 19/20 :=
by sorry

end second_orange_marker_probability_l193_193662


namespace sequence_length_l193_193676

theorem sequence_length (a : ℕ) (h : a = 10800) (h1 : ∀ n, (n ≠ 0 → ∃ m, n = 2 * m ∧ m ≠ 0) ∧ 2 ∣ n)
  : ∃ k : ℕ, k = 5 := 
sorry

end sequence_length_l193_193676


namespace total_kids_played_with_l193_193033

-- Define the conditions as separate constants
def kidsMonday : Nat := 12
def kidsTuesday : Nat := 7

-- Prove the total number of kids Julia played with
theorem total_kids_played_with : kidsMonday + kidsTuesday = 19 := 
by
  sorry

end total_kids_played_with_l193_193033


namespace digits_sum_gt_2023_l193_193076

theorem digits_sum_gt_2023 : ∀ (n : ℕ), n = 2023 → sum_of_digits (2^(2^(2 * n))) > n :=
by
  intros n hn
  rw hn
  sorry

end digits_sum_gt_2023_l193_193076


namespace minimum_value_of_y_l193_193939

theorem minimum_value_of_y (x : ℝ) (h : x > 0) : (∃ y, y = (x^2 + 1) / x ∧ y ≥ 2) ∧ (∃ y, y = (x^2 + 1) / x ∧ y = 2) :=
by
  sorry

end minimum_value_of_y_l193_193939


namespace isosceles_triangle_BE_length_l193_193819

/-- In an isosceles triangle ABC with ∠B = 30° and AB = BC = 6, the altitude CD of triangle ABC and the altitude DE of triangle BDC are drawn. Prove that the length of BE is 4.5. -/
theorem isosceles_triangle_BE_length (A B C D E : Point) :
  is_isosceles_triangle A B C ∧ ∠ B = 30 ∧ AB = 6 ∧ BC = 6 ∧ 
  is_altitude C D A B ∧ is_altitude D E B C → 
  BE = 4.5 :=
sorry

end isosceles_triangle_BE_length_l193_193819


namespace total_course_selection_schemes_l193_193188

theorem total_course_selection_schemes :
  let n_physical_education := 4
  let n_art := 4
  let total_courses := n_physical_education + n_art in
  let choose2_courses := (Nat.choose n_physical_education 1) * (Nat.choose n_art 1)
  let choose3_courses := (Nat.choose n_physical_education 2 * Nat.choose n_art 1) + (Nat.choose n_physical_education 1 * Nat.choose n_art 2) in
  total_courses = n_physical_education + n_art →
  choose2_courses + choose3_courses = 64 :=
by
  intros n_physical_education n_art total_courses choose2_courses choose3_courses h
  have h_choose2_courses: choose2_courses = 16 := by
    simp [n_physical_education, n_art, Nat.choose]
  have h_choose3_courses: choose3_courses = 48 := by
    simp [n_physical_education, n_art, Nat.choose]
  rw [h_choose2_courses, h_choose3_courses]
  exact Nat.add_eq_right.2 rfl

end total_course_selection_schemes_l193_193188


namespace smallest_number_among_0_neg3_2_neg2_l193_193653

theorem smallest_number_among_0_neg3_2_neg2 : 
  ∀ (a b c d : Int), a = 0 ∧ b = -3 ∧ c = 2 ∧ d = -2 → (b < d ∧ b < a ∧ b < c) :=
by
  intro a b c d
  intro h
  cases h
  split
  case h.left =>
    calc b < d : by sorry 
  case h.right.left =>
    calc b < a : by sorry 
  case h.right.right =>
    calc b < c : by sorry

end smallest_number_among_0_neg3_2_neg2_l193_193653


namespace problem_statement_l193_193936

-- Definitions for the problem conditions
def line_eq (x : ℝ) : ℝ := -1/2 * x + 8

-- Definitions for points P and Q
def P : ℝ × ℝ := (16, 0)
def Q : ℝ × ℝ := (0, 8)

-- T (r, s) is on line PQ
def on_line_segment (r s : ℝ) : Prop := s = line_eq r ∧ 0 ≤ r ∧ r ≤ 16

-- Areas of the triangles
def area_POQ : ℝ := 1/2 * 16 * 8
def area_TOP (r s : ℝ) : ℝ := 1/2 * 16 * s

-- Proof problem statement
theorem problem_statement (r s : ℝ) (h₁ : on_line_segment r s)
  (h₂ : area_POQ = 4 * area_TOP r s) : r + s = 14 := by
  sorry

end problem_statement_l193_193936


namespace reduce_to_original_l193_193885

theorem reduce_to_original (x : ℝ) (factor : ℝ) (original : ℝ) :
  original = x → factor = 1/1000 → x * factor = 0.0169 :=
by
  intros h1 h2
  sorry

end reduce_to_original_l193_193885


namespace slope_of_parametric_line_l193_193420

theorem slope_of_parametric_line :
  ∀ (t : ℝ), let x := 1 + 2 * t,
                 y := 2 - 3 * t in
  ∃ (m b : ℝ), y = m * x + b ∧ m = -3 / 2 :=
by
  sorry

end slope_of_parametric_line_l193_193420


namespace smallest_b_for_perfect_square_l193_193145

theorem smallest_b_for_perfect_square (b : ℤ) (h1 : b > 4) (h2 : ∃ n : ℤ, 3 * b + 4 = n * n) : b = 7 :=
by
  sorry

end smallest_b_for_perfect_square_l193_193145


namespace problem_statement_l193_193418

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

def M : set ℝ := { x | -2 < x ∧ x < 2 }

theorem problem_statement (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : |a * b + 4| > |a + b| :=
sorry

end problem_statement_l193_193418


namespace period_2_not_max_1_increasing_2_3_symmetry_axis_correct_statements_l193_193051

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 3^x 
else if -1 ≤ x ∧ x < 0 then 3^(-x)
else if fractional_part x = 0 then 3^(2 - fractional_part x)
else 3^(1 - fractional_part x)

theorem period_2 (x : ℝ) : f(x + 2) = f(x) :=
by sorry

theorem not_max_1 : ¬ (∀ x, f(x) ≤ 1 ∧ f(x) ≥ 0) :=
by sorry

theorem increasing_2_3 : ∀ x y, 2 < x ∧ x < y ∧ y < 3 → f(x) < f(y) :=
by sorry

theorem symmetry_axis : ∀ x, f(4 - x) = f(x + 2) :=
by sorry

theorem correct_statements : 
(period_2) ∧ 
not not_max_1 ∧ 
increasing_2_3 ∧ 
symmetry_axis → 
({1, 3, 4}) :=
by sorry

end period_2_not_max_1_increasing_2_3_symmetry_axis_correct_statements_l193_193051


namespace sum_of_divisors_divisible_by_24_l193_193782

theorem sum_of_divisors_divisible_by_24 (n : ℕ) (h : ∃ k : ℕ, n + 1 = 24 * k) :
  24 ∣ ∑ d in divisors (n), d :=
  sorry

end sum_of_divisors_divisible_by_24_l193_193782


namespace student_marks_equals_125_l193_193303

-- Define the maximum marks
def max_marks : ℕ := 500

-- Define the percentage required to pass
def pass_percentage : ℚ := 33 / 100

-- Define the marks required to pass
def pass_marks : ℚ := pass_percentage * max_marks

-- Define the marks by which the student failed
def fail_by_marks : ℕ := 40

-- Define the obtained marks by the student
def obtained_marks : ℚ := pass_marks - fail_by_marks

-- Prove that the obtained marks are 125
theorem student_marks_equals_125 : obtained_marks = 125 := by
  sorry

end student_marks_equals_125_l193_193303


namespace no_four_points_with_all_odd_distances_l193_193022

theorem no_four_points_with_all_odd_distances :
  ∀ (A B C D : ℝ × ℝ),
    (∃ (x y z p q r : ℕ),
      (x = dist A B ∧ x % 2 = 1) ∧
      (y = dist B C ∧ y % 2 = 1) ∧
      (z = dist C D ∧ z % 2 = 1) ∧
      (p = dist D A ∧ p % 2 = 1) ∧
      (q = dist A C ∧ q % 2 = 1) ∧
      (r = dist B D ∧ r % 2 = 1))
    → false :=
by
  sorry

end no_four_points_with_all_odd_distances_l193_193022


namespace all_lines_l_pass_through_unique_point_l193_193729

theorem all_lines_l_pass_through_unique_point
  (circle : Type) [metric_space circle]
  (O : circle) -- The center of the circle
  (M N : circle) (AB : set circle) -- Fixed chord MN and diameter AB
  (h1 : M ≠ N)
  (h2 : ¬(M ∈ AB ∨ N ∈ AB))
  (h3 : AB ⊂ {p : circle | dist O p = dist O M}) -- AB is a diameter
  (C : circle)
  (hC : ∃ A B ∈ AB, line_through A M ∩ line_through B N = {C} ∧ A ≠ B) :
  ∃ P : circle, ∀ A B ∈ AB, line_through C P ∩ line_through (perp_bisector AB) C = {P} :=
sorry

end all_lines_l_pass_through_unique_point_l193_193729


namespace sum_of_interior_angles_l193_193478

noncomputable def sum_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)
def ratio := 8.5

theorem sum_of_interior_angles (n : ℕ) (hn : n > 2)
  (h : ∀ (i : ℕ) (hi : i < n), (let b := 360 / n in let a := ratio * b in true)) :
  sum_interior_angles n = 3060 :=
by
  sorry

end sum_of_interior_angles_l193_193478


namespace john_can_drive_150_miles_l193_193031

theorem john_can_drive_150_miles (mpg : ℕ) (price_per_gallon : ℕ) (total_dollars : ℕ)
  (h1 : mpg = 25) (h2 : price_per_gallon = 5) (h3 : total_dollars = 30) :
  (total_dollars / price_per_gallon) * mpg = 150 :=
by
  rw [h1, h2, h3]
  sorry

end john_can_drive_150_miles_l193_193031


namespace smallest_value_d_squared_l193_193510

noncomputable def parallelogram_area : ℂ → ℝ :=
  λ z, |z * (1/z) + z * ((z + 1/z) - z)| / 2

theorem smallest_value_d_squared
  (z : ℂ)
  (h1 : ∃ (z : ℂ), parallelogram_area z = (35 / 37))
  (h2 : z.re > 0) :
  (complex.abs (z + (1/z))) ^ 2 = (50 / 37) :=
sorry

end smallest_value_d_squared_l193_193510


namespace arithmetic_sum_mod_20_l193_193331

theorem arithmetic_sum_mod_20 (a d l : ℕ) (n : ℕ) (h_a : a = 2) (h_d : d = 5) (h_l : l = 142) :
  let sum := (n * (a + l)) / 2 in
  (sum % 20) = 8 :=
by
  sorry

end arithmetic_sum_mod_20_l193_193331


namespace calculate_area_of_triangle_l193_193451

noncomputable def area_of_equilateral_triangle (z : ℂ) (ω : ℂ) : ℂ :=
  if ω = complex.exp (2 * real.pi * complex.I / 3) ∨ ω = complex.exp (-2 * real.pi * complex.I / 3) then
    let s := complex.abs (z^2 - z) in
    if s = complex.sqrt 3 then
      (complex.sqrt 3 / 4) * s^2
    else
      0  -- Not possible as per conditions (for type completeness).
  else
    0  -- Not possible as per conditions (for type completeness).

theorem calculate_area_of_triangle (z : ℂ) (hz : z ≠ 0) (hz_eq : z^2 + z + 1 = 0) (ω : ℂ) (hω : ω = complex.exp (2 * real.pi * complex.I / 3) ∨ ω = complex.exp (-2 * real.pi * complex.I / 3)) :
  area_of_equilateral_triangle z ω = 3 * complex.sqrt 3 / 4 :=
by {
  sorry
}

end calculate_area_of_triangle_l193_193451


namespace total_course_selection_schemes_l193_193266

theorem total_course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose_two :=
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 1)
  let choose_three :=
    (Nat.choose PE_courses 2) * (Nat.choose Art_courses 1) +
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 2)
  total_courses = 8
  ∧ (choose_two + choose_three = 64) :=
by
  sorry

end total_course_selection_schemes_l193_193266


namespace boat_distance_travel_l193_193626

-- Define the conditions
def speed_boat : ℝ := 10 -- speed of the boat in still water (mph)
def speed_stream : ℝ := 2 -- speed of the stream (mph)
def time_difference : ℝ := 1.5 -- difference in travel time between downstream and upstream (hours)

-- Define the problem statement
theorem boat_distance_travel : 
  ∃ D : ℝ, 
    (D / (speed_boat - speed_stream) - D / (speed_boat + speed_stream)) = time_difference 
  → D = 36 :=
begin
  sorry
end

end boat_distance_travel_l193_193626


namespace tangent_circle_l193_193390

noncomputable theory

-- Definitions for the geometric setup
variables {A B C D E F G H : Point}

-- Conditions from the problem
def cyclic_quadrilateral (A B C D : Point) : Prop := ∃ c : Circle, A ∈ c ∧ B ∈ c ∧ C ∈ c ∧ D ∈ c
def intersection_of_diagonals (AC BD E : Point) : Prop := line_through A C ∩ line_through B D = {E}
def intersection_of_lines (AD BC F : Point) : Prop := line_through A D ∩ line_through B C = {F}
def is_midpoint (G A B : Point) : Prop := dist A G = dist G B ∧ collinear A G B
def midpoint_AB (G : Point) : Prop := is_midpoint G A B
def midpoint_CD (H : Point) : Prop := is_midpoint H C D

-- Final goal to prove in Lean
theorem tangent_circle (cyclic_quad : cyclic_quadrilateral A B C D)
  (inter_diag : intersection_of_diagonals A C E)
  (inter_lines : intersection_of_lines A D F)
  (mid_ab : midpoint_AB G)
  (mid_cd : midpoint_CD H) :
  tangent_line_at EF E (circumcircle E G H) :=
begin
  sorry -- proof goes here
end

end tangent_circle_l193_193390


namespace tangent_line_equation_l193_193101

noncomputable def f : ℝ → ℝ := λ x, Real.exp (2 - x)

theorem tangent_line_equation :
  ∃ (m : ℝ), (∀ (x y : ℝ), (y = f x) → (y = -e^3 * x)) := by
  sorry

end tangent_line_equation_l193_193101


namespace points_segments_triangle_l193_193826

theorem points_segments_triangle (N : ℕ) (P : Fin 2N → ℝ × ℝ) (segments: Finset (Fin 2N × Fin 2N)) (hsegments : segments.card = N^2 + 1) :
  ∃ (i j k : Fin 2N), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (i, j) ∈ segments ∧ (j, k) ∈ segments ∧ (i, k) ∈ segments :=
sorry

end points_segments_triangle_l193_193826


namespace least_n_for_obtuse_triangle_l193_193856

namespace obtuse_triangle

-- Define angles and n
def alpha (n : ℕ) : ℝ := 59 + n * 0.02
def beta : ℝ := 60
def gamma (n : ℕ) : ℝ := 61 - n * 0.02

-- Define condition for the triangle being obtuse
def is_obtuse_triangle (n : ℕ) : Prop :=
  alpha n > 90 ∨ gamma n > 90

-- Statement about the smallest n such that the triangle is obtuse
theorem least_n_for_obtuse_triangle : ∃ n : ℕ, n = 1551 ∧ is_obtuse_triangle n :=
by
  -- existence proof ends here, details for proof to be provided separately
  sorry

end obtuse_triangle

end least_n_for_obtuse_triangle_l193_193856


namespace sum_s_r_l193_193861

-- Define conditions as lean assumptions
def domain_r := {-2, -1, 0, 1}
def range_r := {-1, 1, 3, 5}
def domain_s := {0, 1, 2, 3}
def s (x : Int) : Int := 2 * x + 2

-- Define the specific values r(x) can take and the sum we need to prove.
theorem sum_s_r (r : Int → Int) 
  (h_domain_r : ∀ x, x ∈ domain_r → r x ∈ range_r)
  (h_domain_s : ∀ x, x ∈ domain_s → s x = 2 * x + 2) :
  (∀ x, x ∈ domain_r → r x ∈ domain_s ∪ range_r) →
  (s (1) + s (3) = 12) :=
by
  sorry

end sum_s_r_l193_193861


namespace hexagon_side_length_l193_193021

-- Let s be the length of a side of the equilateral triangle ABC
variable (s : ℝ)

-- The area of each smaller equilateral triangle PAB, QBC, and RCA
def area_of_triangle (s : ℝ) := (sqrt 3 / 4) * (s / 2) ^ 2

-- The total area cut off is the sum of the areas of the three smaller triangles
def total_area_cut_off (s : ℝ) := 3 * area_of_triangle s

-- The total area cut off is given as 150 cm²
axiom total_area_eq : total_area_cut_off s = 150

-- Prove that the side length of the regular hexagon is 20 * sqrt 3 / 3
theorem hexagon_side_length : (s / 2) = 20 * sqrt 3 / 3 :=
  by
  sorry

end hexagon_side_length_l193_193021


namespace course_selection_schemes_l193_193197

theorem course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose (n k : ℕ) := nat.choose n k
  
  (choose 4 1 * choose 4 1 + 
   choose 4 2 * choose 4 1 + 
   choose 4 1 * choose 4 2) = 64 := 
by {
  -- calculating the combinations
  have comb_2_courses : choose 4 1 * choose 4 1 = 16 := by sorry,
  have comb_2_pe_1_art : choose 4 2 * choose 4 1 = 24 := by sorry,
  have comb_1_pe_2_art : choose 4 1 * choose 4 2 = 24 := by sorry,
  have total_comb : 16 + 24 + 24 = 64 := by {
    simp [comb_2_courses, comb_2_pe_1_art, comb_1_pe_2_art]
  },
  exact total_comb
}

end course_selection_schemes_l193_193197


namespace line_intersects_parabola_at_9_units_apart_l193_193689

theorem line_intersects_parabola_at_9_units_apart :
  ∃ m b, (∃ (k1 k2 : ℝ), 
              (y1 = k1^2 + 6*k1 - 4) ∧ 
              (y2 = k2^2 + 6*k2 - 4) ∧ 
              (y1 = m*k1 + b) ∧ 
              (y2 = m*k2 + b) ∧ 
              |y1 - y2| = 9) ∧ 
          (0 ≠ b) ∧ 
          ((1 : ℝ) = 2*m + b) ∧ 
          (m = 4 ∧ b = -7)
:= sorry

end line_intersects_parabola_at_9_units_apart_l193_193689


namespace ricardo_coin_difference_l193_193887

-- Define the conditions
def ricardo_num_coins : ℕ := 1980
def penny_value : ℕ := 1
def dime_value : ℕ := 10

-- Using the given constraints
theorem ricardo_coin_difference :
  ∀ (p : ℕ), 1 ≤ p ∧ p ≤ ricardo_num_coins - 1 →
  (penny_value * p + dime_value * (ricardo_num_coins - p) - (9 * p)) ≤ 19791 ∧
  (penny_value * p + dime_value * (ricardo_num_coins - p) - (9 * 1979)) ≠ (0 - 4 × 9) →
  17802 = 19791 - 1989 := 
by 
  sorry

end ricardo_coin_difference_l193_193887


namespace percentage_increase_per_hour_l193_193470

noncomputable def john_telethon (first_hours second_hours total_income first_hourly_rate : ℕ) : ℚ :=
  let first_period_income := first_hours * first_hourly_rate
  let remaining_income := total_income - first_period_income
  (remaining_income / second_hours : ℚ)

theorem percentage_increase_per_hour
  (first_hours second_hours total_income first_hourly_rate : ℕ)
  (h1 : first_hours = 12)
  (h2 : second_hours = 14)
  (h3 : first_hourly_rate = 5000)
  (h4 : total_income = 144000) :
  let second_hourly_rate := john_telethon first_hours second_hours total_income first_hourly_rate in
  (second_hourly_rate - first_hourly_rate) / first_hourly_rate * 100 = 20 :=
by
  sorry

end percentage_increase_per_hour_l193_193470


namespace isosceles_trapezoid_area_l193_193658

theorem isosceles_trapezoid_area (m n : ℝ) :
  m > 0 → n > 0 →
  ∃ S, S = 2 * Real.sqrt (m * n) * (m + n) :=
by
  intros hm hn
  use 2 * Real.sqrt (m * n) * (m + n)
  sorry

end isosceles_trapezoid_area_l193_193658


namespace correct_fraction_l193_193001

theorem correct_fraction (x y : ℕ) (h1 : 480 * 5 / 6 = 480 * x / y + 250) : x / y = 5 / 16 :=
by
  sorry

end correct_fraction_l193_193001


namespace find_S_10_l193_193744
noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a(n + 1) - a n = a 2 - a 1

noncomputable def sum_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = ∑ i in finset.range n, a i

theorem find_S_10
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum_sequence : sum_sequence S a)
  (h_a3 : a 3 = 16)
  (h_S20 : S 20 = 20)
  (h_natural : ∀ n, n ∈ ℕ ∧ n > 0) :
  S 10 = 110 :=
by
  sorry

end find_S_10_l193_193744


namespace range_of_a_solution_set_of_inequality_l193_193771

-- Lean statement for Part 1
theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  ∀ x : ℝ, x^2 - 2 * a * x + a > 0 :=
by
  sorry

-- Lean statement for Part 2
theorem solution_set_of_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  { x : ℝ | a^(x^2 - 3) < a^(2 * x) ∧ a^(2 * x) < 1 } = { x : ℝ | x > 3 } :=
by
  sorry

end range_of_a_solution_set_of_inequality_l193_193771


namespace sin_double_angle_subst_l193_193385

open Real

theorem sin_double_angle_subst 
  (α : ℝ)
  (h : sin (α + π / 6) = -1 / 3) :
  sin (2 * α - π / 6) = -7 / 9 := 
by
  sorry

end sin_double_angle_subst_l193_193385


namespace omega_range_l193_193411

noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x) - cos (ω * x)

theorem omega_range {ω : ℝ} (h : ω > 2 / 3) 
  (H : ∀ k : ℤ, (k * π + 3 * π / 4) / ω ≤ 2 * π ∨ (k * π + 3 * π / 4) / ω ≥ 3 * π)
  : ω ∈ set.Icc (7 / 8 : ℝ) (11 / 12 : ℝ) :=
sorry

end omega_range_l193_193411


namespace sequence_multiple_of_n_l193_193950

noncomputable theory

def a : ℕ → ℕ
| 1 := 1
| 2 := 2
| 3 := 24
| n := if h : n ≥ 4 then (6 * a (n-1) ^ 2 * a (n-3) - 8 * a (n-1) * a (n-2) ^ 2) / (a (n-2) * a (n-3))
       else 0 -- Handles the cases where n < 4 which shouldn't occur since we have a catch-all fallback

theorem sequence_multiple_of_n (n : ℕ) (hn : n ≥ 1) : n ∣ a n := 
by
  sorry

end sequence_multiple_of_n_l193_193950


namespace amicable_pairs_lower_bound_l193_193993
open Set

-- Definitions and proof problem statement
variables {V : Type*}

structure Graph :=
  (vertices : Set V) -- set of vertices
  (edges : Set (V × V)) -- set of edges
  (simple_graph : ∀ ⦃x y⦄, (x, y) ∈ edges → (y, x) ∈ edges ∧ x ≠ y)

def even (n : ℕ) : Prop := n % 2 = 0

def amicable (G : Graph) (x y : V) : Prop :=
  ∃ z, (z ∈ G.vertices ∧ (x, z) ∈ G.edges ∧ (y, z) ∈ G.edges)

noncomputable def choose (n k : ℕ) : ℕ :=
  @nat.choose nat.noncomputable_defaults n k

theorem amicable_pairs_lower_bound
  (n : ℕ) (h_even : even n) (h_pos : 0 < n)
  (G : Graph) (h_vertices : G.vertices.finite)
  (h_card_vertices : G.vertices.to_finset.card = n)
  (h_edges : G.edges.to_finset.card = n * n / 4) :
  (∃ x y, amicable G x y) :=
sorry

end amicable_pairs_lower_bound_l193_193993


namespace natural_numbers_count_l193_193636

def number_of_valid_natural_numbers (m a n k : ℕ) :=
  ∃ (k : ℕ) (a : ℕ) (m : ℕ) (n : ℕ),
  1 ≤ a ∧ a ≤ 9 ∧ m < 10 ^ k ∧
  10^k * a = 8 * m ∧ n = 0

theorem natural_numbers_count: (∃ (s : finset ℕ), s.card = 7 ∧ ∀ x ∈ s, number_of_valid_natural_numbers x) := sorry

end natural_numbers_count_l193_193636


namespace course_selection_schemes_l193_193214

theorem course_selection_schemes :
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  choose_2_courses + choose_3_courses = 64 :=
by
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  show choose_2_courses + choose_3_courses = 64 from sorry

end course_selection_schemes_l193_193214


namespace property_value_initial_value_l193_193164

theorem property_value_initial_value 
  (V : ℝ) (r : ℝ) (n : ℕ) (value_end_3_years : V = 21093) 
  (annual_rate_depreciation : r = 0.0625) 
  (number_of_years : n = 3) : 
  let P := V / (1 - r)^n 
  in P ≈ 25592.31 := 
by
  sorry

end property_value_initial_value_l193_193164


namespace geometric_sequence_sixth_term_l193_193913

theorem geometric_sequence_sixth_term 
  (a : ℝ) (r : ℝ) (a9 : ℝ)
  (h1 : a = 1024)
  (h2 : a9 = 16)
  (h3 : ∀ n, a9 = a * r ^ (n - 1)) :
  a * r ^ 5 = 4 * sqrt 2 :=
by
  sorry

end geometric_sequence_sixth_term_l193_193913


namespace probability_hugo_prime_given_win_l193_193446

/-- In a modified game, each of 5 players, including Hugo, rolls an 8-sided die. 
A prime number roll is considered a winning roll, and the highest prime number wins the game. 
If there's a tie with the highest prime number, the tied players roll again until one wins. 
What is the probability that Hugo's first roll was a prime number, given that he won the game? 
-/
theorem probability_hugo_prime_given_win :
  (P(hugoFirstRollIsPrime | hugoWins) = 5 / 32) :=
sorry

end probability_hugo_prime_given_win_l193_193446


namespace bucket_full_weight_l193_193973

theorem bucket_full_weight (p q : ℝ) (x y : ℝ) 
    (h1 : x + (3/4) * y = p) 
    (h2 : x + (1/3) * y = q) : 
    x + y = (8 * p - 3 * q) / 5 :=
sorry

end bucket_full_weight_l193_193973


namespace union_A_B_for_m_eq_neg3_range_of_m_for_A_inter_B_empty_l193_193050

def A (m : ℝ) : Set ℝ := {x | 2 * m - 1 < x ∧ x < m}
def B : Set ℝ := {x | -4 ≤ x ∧ x ≤ 5}

theorem union_A_B_for_m_eq_neg3 :
  A (-3) ∪ B = {x | -7 < x ∧ x ≤ 5} := by
    sorry

theorem range_of_m_for_A_inter_B_empty :
  (A m ∩ B = ∅) ↔ (m ∈ Iic (-4) ∪ Ici 1) := by
    sorry

end union_A_B_for_m_eq_neg3_range_of_m_for_A_inter_B_empty_l193_193050


namespace find_ffnegthree_eq_neg2_l193_193745

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
if x > 0 then log x / log (1/3) else a^x + b

theorem find_ffnegthree_eq_neg2 (a b : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1)
  (h₃ : f 0 a b = 2) (h₄ : f (-1) a b = 3) : f (f (-3) a b) a b = -2 :=
begin
  sorry
end

end find_ffnegthree_eq_neg2_l193_193745


namespace only_prime_p_l193_193354

open Nat

theorem only_prime_p (p : ℕ) (hp : Prime p) (hq : Prime (3 * p^2 + 1)) : p = 2 :=
by
  sorry

end only_prime_p_l193_193354


namespace segment_parallel_to_base_l193_193832

noncomputable def acute_triangle (A B C M A1 B1 C1 L N : Point) : Prop :=
∃ (circA : Circumcircle A B C) (circB B C C1 : Circle), 
  acute_angle A B C ∧ 
  is_angle_bisector A M ∧ 
  second_intersection_of_angle_bisector circA A M A1 ∧ 
  second_intersection_of BM circB B M B1 ∧ 
  second_intersection_of CM circB C M C1 ∧ 
  lines_intersect A B C1 A1 L ∧ 
  lines_intersect A C B1 A1 N 

theorem segment_parallel_to_base (A B C M A1 B1 C1 L N : Point) (h : acute_triangle A B C M A1 B1 C1 L N) :
  parallel LN BC :=
sorry

end segment_parallel_to_base_l193_193832


namespace total_number_of_course_selection_schemes_l193_193246

-- Define the total number of courses
def total_courses := 8

-- Define number of physical education and art courses
def pe_courses := 4
def art_courses := 4

-- Define selections: students choose 2 or 3 courses
def course_selections : Finset (Finset ℕ) :=
  (Finset.powerset (Finset.range total_courses)).filter (λ s, s.card = 2 ∨ s.card = 3)

-- Define condition: at least 1 course from each category
def valid_selections : Finset (Finset ℕ) :=
  course_selections.filter (λ s, ∃ pe art, s = pe ∪ art ∧ pe.card ≠ 0 ∧ art.card ≠ 0 ∧ 
                             pe ⊆ Finset.range pe_courses ∧ art ⊆ (Finset.range total_courses).filter (λ x, x ≥ pe_courses))

theorem total_number_of_course_selection_schemes : valid_selections.card = 64 := 
by sorry

end total_number_of_course_selection_schemes_l193_193246


namespace concurrent_bisectors_l193_193316

theorem concurrent_bisectors 
  (A B C P Q R X Y Z : Type)
  (trisect_angles : Triangle → Type)
  (adjacent_trisectors_intersect : Triangle → Triangle)
  (bisectors_intersect : ∀ (tri : Triangle) (ang : Angle), Point)
  (PX QY RZ : Line)
  (concurrent : Line → Line → Line → Prop)
  (cond1 : trisect_angles (Triangle.mk A B C))
  (cond2 : adjacent_trisectors_intersect (Triangle.mk A B C) = Triangle.mk P Q R)
  (cond3 : bisectors_intersect (Triangle.mk A B C) (Angle.mk A B C) = X ∧
           bisectors_intersect (Triangle.mk A B C) (Angle.mk B C A) = Y ∧
           bisectors_intersect (Triangle.mk A B C) (Angle.mk C A B) = Z) :
  concurrent (Line.mk P X) (Line.mk Q Y) (Line.mk R Z) := sorry

end concurrent_bisectors_l193_193316


namespace ribbons_problem_l193_193815

/-
    In a large box of ribbons, 1/3 are yellow, 1/4 are purple, 1/6 are orange, and the remaining 40 ribbons are black.
    Prove that the total number of orange ribbons is 27.
-/

theorem ribbons_problem :
  ∀ (total : ℕ), 
    (1 / 3 : ℚ) * total + (1 / 4 : ℚ) * total + (1 / 6 : ℚ) * total + 40 = total →
    (1 / 6 : ℚ) * total = 27 := sorry

end ribbons_problem_l193_193815


namespace total_course_selection_schemes_l193_193269

theorem total_course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose_two :=
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 1)
  let choose_three :=
    (Nat.choose PE_courses 2) * (Nat.choose Art_courses 1) +
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 2)
  total_courses = 8
  ∧ (choose_two + choose_three = 64) :=
by
  sorry

end total_course_selection_schemes_l193_193269


namespace collinear_M_N_T_l193_193661

variables {A B P Q R S M N T : Type}
variables (AP BQ AR BS AM MB PN NQ RT TS : ℝ)
variables (is_collinear : A → B → P → Q → R → S → M → N → T → Prop)

noncomputable def given_conditions (AP BQ AR BS AM MB PN NQ RT TS : ℝ) : Prop :=
  (AP / BQ = AR / BS) ∧ (AM / MB = PN / NQ ∧ PN / NQ = RT / TS)

theorem collinear_M_N_T
  (h1 : AP / BQ = AR / BS)
  (h2 : AM / MB = PN / NQ ∧ PN / NQ = RT / TS) :
  is_collinear A B P Q R S M N T :=
sorry

end collinear_M_N_T_l193_193661


namespace divisor_condition_l193_193382

noncomputable def polynomial := Polynomial ℤ

theorem divisor_condition (a : ℤ) :
  (x^2 - 2x + (a : polynomial)) ∣ (x^13 + 2x + 180 : polynomial) ↔ a = 3 :=
-- Proof omitted.
sorry

end divisor_condition_l193_193382


namespace percentage_apples_basket_l193_193586

theorem percentage_apples_basket :
  let initial_apples := 10
  let initial_oranges := 5
  let added_oranges := 5
  let total_apples := initial_apples
  let total_oranges := initial_oranges + added_oranges
  let total_fruits := total_apples + total_oranges
  (total_apples / total_fruits) * 100 = 50 :=
by
  sorry

end percentage_apples_basket_l193_193586


namespace unique_solution_x_eq_zero_l193_193357

theorem unique_solution_x_eq_zero (x : ℝ) (h : (9^x + 16^x) / (15^x + 24^x) = 8 / 5) : x = 0 :=
sorry

end unique_solution_x_eq_zero_l193_193357


namespace largest_value_l193_193606

-- Define the five expressions as given in the conditions
def exprA : ℕ := 3 + 1 + 2 + 8
def exprB : ℕ := 3 * 1 + 2 + 8
def exprC : ℕ := 3 + 1 * 2 + 8
def exprD : ℕ := 3 + 1 + 2 * 8
def exprE : ℕ := 3 * 1 * 2 * 8

-- Define the theorem stating that exprE is the largest value
theorem largest_value : exprE = 48 ∧ exprE > exprA ∧ exprE > exprB ∧ exprE > exprC ∧ exprE > exprD := by
  sorry

end largest_value_l193_193606


namespace natural_numbers_count_l193_193635

def number_of_valid_natural_numbers (m a n k : ℕ) :=
  ∃ (k : ℕ) (a : ℕ) (m : ℕ) (n : ℕ),
  1 ≤ a ∧ a ≤ 9 ∧ m < 10 ^ k ∧
  10^k * a = 8 * m ∧ n = 0

theorem natural_numbers_count: (∃ (s : finset ℕ), s.card = 7 ∧ ∀ x ∈ s, number_of_valid_natural_numbers x) := sorry

end natural_numbers_count_l193_193635


namespace find_s_l193_193482

theorem find_s (c d n r s : ℝ) 
(h1 : c * d = 3)
(h2 : ∃ p q : ℝ, (p + q = r) ∧ (p * q = s) ∧ (p = c + 1/d ∧ q = d + 1/c)) :
s = 16 / 3 :=
by
  sorry

end find_s_l193_193482


namespace total_course_selection_schemes_l193_193261

theorem total_course_selection_schemes (PE_courses : ℕ) (Art_courses : ℕ) : 
  PE_courses = 4 → Art_courses = 4 → 
  (finset.card (finset.powerset_len 2 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b ∈ s), a < PE_courses ∧ b ≥ PE_courses) + 
   finset.card (finset.powerset_len 3 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b c ∈ s), (a < PE_courses ∧ b < PE_courses ∧ c ≥ PE_courses) ∨ 
                        (a < PE_courses ∧ b ≥ PE_courses ∧ c ≥ PE_courses ∧ a ≠ b ≠ c)) = 64 :=
by
  sorry

end total_course_selection_schemes_l193_193261


namespace sine_double_angle_plus_cos_gt_one_l193_193068

theorem sine_double_angle_plus_cos_gt_one (x : ℝ) (h1 : 0 < x) (h2 : x < π / 3) :
  sin (2 * x) + cos x > 1 :=
sorry

end sine_double_angle_plus_cos_gt_one_l193_193068


namespace surface_area_circumscribed_sphere_l193_193006

theorem surface_area_circumscribed_sphere
  (S A B C : Type) [MetricSpace S] [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (SA_perpendicular_ABC : SA ⊥ ABC)
  (angle_BAC : angle B A C = 120)
  (SA_equals_AC : dist S A = 2 ∧ dist A C = 2)
  (AB_equals_1 : dist A B = 1) :
  surface_area (circumscribed_sphere (tetrahedron S A B C)) = (40 * π) / 3 :=
sorry

end surface_area_circumscribed_sphere_l193_193006


namespace anusha_share_l193_193072

theorem anusha_share (A B E D G X : ℝ) 
  (h1: 20 * A = X)
  (h2: 15 * B = X)
  (h3: 8 * E = X)
  (h4: 12 * D = X)
  (h5: 10 * G = X)
  (h6: A + B + E + D + G = 950) : 
  A = 112 := 
by 
  sorry

end anusha_share_l193_193072


namespace vector_parallel_cond_l193_193431

-- We define the vectors and the condition of parallelism
def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (-2, Real.logBase 2 m)

-- Lean statement to prove the correct value of m
theorem vector_parallel_cond (m : ℝ) (h : b m = (λ m, -2, Real.logBase 2 m) m)
  (parallel : (λ (a b : ℝ × ℝ), ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2) a (b m)) :
  m = 1 / 16 :=
sorry

end vector_parallel_cond_l193_193431


namespace hearty_total_beads_l193_193777

-- Definition of the problem conditions
def blue_beads_per_package (r : ℕ) : ℕ := 2 * r
def red_beads_per_package : ℕ := 40
def red_packages : ℕ := 5
def blue_packages : ℕ := 3

-- Define the total number of beads Hearty has
def total_beads (r : ℕ) (rp : ℕ) (bp : ℕ) : ℕ :=
  (rp * red_beads_per_package) + (bp * blue_beads_per_package red_beads_per_package)

-- The theorem to be proven
theorem hearty_total_beads : total_beads red_beads_per_package red_packages blue_packages = 440 := by
  sorry

end hearty_total_beads_l193_193777


namespace arc_MTN_constant_l193_193464

-- Define the equilateral triangle ABC
variables {A B C O T M N : Point}
variable {r : ℝ}

-- The radius of the circle is equal to the altitude of the equilateral triangle
axiom altitude_eq_radius (triangle_equi : IsEquilateral A B C) :
  r = (distance A B) * (√3 / 2)

-- Define the tangent point T and intersecting points M and N as variables on the side AB and lines AC, BC respectively.
axiom rolling_circle (tangent_point : IsTangent T (Circle r O) (Line A B))
axiom intersect_points (intersect_M : Intersects M (Line A C) (Circle r O))
axiom intersect_points (intersect_N : Intersects N (Line B C) (Circle r O))

-- Define our proof objective
theorem arc_MTN_constant (triangle_equi : IsEquilateral A B C)
  (tangent_point : IsTangent T (Circle r O) (Line A B))
  (intersect_M : Intersects M (Line A C) (Circle r O))
  (intersect_N : Intersects N (Line B C) (Circle r O)) :
  measure_of_arc_MTN r M T N = 60 :=
sorry

end arc_MTN_constant_l193_193464


namespace notebook_and_pen_prices_l193_193014

theorem notebook_and_pen_prices (x y : ℕ) (h1 : 2 * x + y = 30) (h2 : x = 2 * y) :
  x = 12 ∧ y = 6 :=
by
  sorry

end notebook_and_pen_prices_l193_193014


namespace smallest_positive_period_of_f_range_of_f_on_interval_l193_193721

def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem smallest_positive_period_of_f : ∀ x : ℝ, f x = f (x + π) := by sorry

theorem range_of_f_on_interval : ∀ y:ℝ, y ∈ [1, Real.sqrt 2] ↔ ∃ x : ℝ, 0 ≤ x ∧ x ≤ π/4 ∧ f x = y := by sorry

end smallest_positive_period_of_f_range_of_f_on_interval_l193_193721


namespace min_pos_period_tan_l193_193938

theorem min_pos_period_tan (A ω : ℝ) (ϕ : ℝ) : 
  (0 < ω) → 
  (∀ x, tan (ω * x + ϕ) = A * tan (3 * x)) → 
  ∃ T, T = π / 3 := by
  intros hω hfun
  use (π / 3)
  sorry

end min_pos_period_tan_l193_193938


namespace sales_growth_rate_l193_193949

theorem sales_growth_rate :
  ∃ (x : ℝ), 0.1 ≤ x ∧
  let february_sales := 400 in
  let march_sales := february_sales * 1.1 in
  let may_sales := march_sales * (1 + x)^2 in
  may_sales = 633.6 :=
sorry

end sales_growth_rate_l193_193949


namespace total_selection_schemes_l193_193283

-- Define the given conditions
def num_phys_ed_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_phys_ed_courses + num_art_courses
def valid_course_combos : finset ℕ := {2, 3}

-- Define the number of selection schemes with the constraints
def selection_schemes : nat :=
  (num_phys_ed_courses.choose 1 * num_art_courses.choose 1) + 
  (num_phys_ed_courses.choose 2 * num_art_courses.choose 1 + 
   num_phys_ed_courses.choose 1 * num_art_courses.choose 2)

-- State the theorem to be proved
theorem total_selection_schemes : selection_schemes = 64 := by
  sorry

end total_selection_schemes_l193_193283


namespace valid_numbers_are_135_and_144_l193_193353

noncomputable def find_valid_numbers : List ℕ :=
  let numbers := [135, 144]
  numbers.filter (λ n =>
    let a := n / 100
    let b := (n / 10) % 10
    let c := n % 10
    n = (100 * a + 10 * b + c) ∧ n = a * b * c * (a + b + c)
  )

theorem valid_numbers_are_135_and_144 :
  find_valid_numbers = [135, 144] :=
by
  sorry

end valid_numbers_are_135_and_144_l193_193353


namespace textile_manufacturing_expenses_l193_193305

theorem textile_manufacturing_expenses :
  ∃ (M : ℝ), 
    let sales_per_loom := 500000 / 100,
        total_sales := 500000,
        establishment_charges := 75000,
        profit_100_looms := total_sales - M - establishment_charges,
        profit_99_looms := (99 * sales_per_loom) - (0.99 * M) - establishment_charges in
    (profit_100_looms - profit_99_looms = 3500) 
    → M = 150000 := sorry

end textile_manufacturing_expenses_l193_193305


namespace length_of_QS_l193_193461

variable (Q R S : ℝ)
variable (RS : ℝ := 10)
variable (cosR : ℝ := 3/5)

def QR : ℝ := (cosR * RS)

def QS : ℝ := sqrt (RS^2 - QR^2)

theorem length_of_QS : QS = 8 := by
  sorry

end length_of_QS_l193_193461


namespace range_of_p_l193_193864

theorem range_of_p (p: ℝ): 
  {x | x^2 + (p + 2) * x + 1 = 0} ⊆ set_of (λ x, x < 0) ↔ p ∈ Ioo (-4:ℝ) (⨅ x, x) ∨ p ≥ 0 :=
sorry

end range_of_p_l193_193864


namespace max_tags_in_original_positions_l193_193879

theorem max_tags_in_original_positions :
  ∃ (s : list (ℕ × ℕ)) (perm : list (ℕ × ℕ)), 
  (∀ i < 100, s.nth i = some (i / 10, i % 10)) ∧
  (∀ i < 99, 
    let (a, b) := s.nth i ∧
    let (c, d) := s.nth (i + 1) 
    in (c = a + 1 ∨ c = a - 1 ∨ d = b + 1 ∨ d = b - 1)) ∧
  (∀ i < 100, perm.nth i = s.nth i) ∧ (∃ (k : ℕ), k = 50) :=
sorry

end max_tags_in_original_positions_l193_193879


namespace quadrilateral_sides_equality_l193_193109

theorem quadrilateral_sides_equality 
  (a b c d : ℕ) 
  (h1 : (b + c + d) % a = 0) 
  (h2 : (a + c + d) % b = 0) 
  (h3 : (a + b + d) % c = 0) 
  (h4 : (a + b + c) % d = 0) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d :=
sorry

end quadrilateral_sides_equality_l193_193109


namespace num_ways_exactly_two_referees_match_seat_number_l193_193086

theorem num_ways_exactly_two_referees_match_seat_number :
  let referees := {1, 2, 3, 4, 5}
  let seats := {1, 2, 3, 4, 5}
  ∃ (arrangement : Array ℕ), arrangement.length = 5 ∧
    ∀ i, i ∈ referees → i ∈ seats ∧
    (∃ a b, a ≠ b ∧ arrangement[a] = a ∧ arrangement[b] = b) ∧
    (∀ c ≠ a, c ≠ b → arrangement[c] ≠ c) → 
    (finset.card (finset.filter (λ r, r ∈ referee ∧ arrangement[r] = r) refereees) = 2)
= 20 := sorry

end num_ways_exactly_two_referees_match_seat_number_l193_193086


namespace find_v_approximate_value_l193_193957

noncomputable def concentration (lambda : ℝ) (m : ℝ) (v0 : ℝ) (t : ℝ) (v : ℝ) : ℝ :=
  (1 - lambda) * (m / v0) + lambda * (m / v0) * Real.exp (-v * t)

theorem find_v_approximate_value :
  ∀ (m v0 : ℝ), let lambda := 4 / 5
                 let rho := concentration lambda m v0
                 (ρ1 := rho 1)
                 (ρ2 := rho 2)
                 (h : ρ1 = 3 / 2 * ρ2)
  in v ≈ 1.7917 :=
by
  sorry

end find_v_approximate_value_l193_193957


namespace problem_solution_l193_193310

def line (a b c : ℝ) : Prop := a*x + b*y + c = 0

def parallel_lines (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1*b2 - a2*b1 = 0

theorem problem_solution :
  let l1 := line 1 (-2) 1,
      l2 := line 2 (-1) 1,
      l3 := line 2 (-4) 2,
      l4 := line 2 4 1,
      l5 := line 2 (-4) 1 in
      parallel_lines 1 (-2) 1 2 (-4) 1 :=
by sorry

end problem_solution_l193_193310


namespace median_of_students_l193_193351

def heights : List ℝ := [1.72, 1.78, 1.80, 1.69, 1.76]

def median_is (l : List ℝ) (m : ℝ) : Prop :=
  let sorted := l.sort (≤)
  let n := sorted.length
  n > 0 ∧ n % 2 = 1 ∧ sorted.nth (n / 2) = some m

theorem median_of_students : median_is heights 1.76 :=
  sorry

end median_of_students_l193_193351


namespace total_fruits_915_l193_193580

variables (A B Bo C : ℕ)

def total_fruits (A B Bo C : ℕ) := A + B + Bo + C

def conditions (A B Bo C : ℕ) : Prop :=
  (A = 3 * B) ∧
  (B = 3 * Bo / 4) ∧
  (C = 5 * A) ∧
  (Bo = 60)

theorem total_fruits_915 : ∃ (A B Bo C : ℕ), conditions A B Bo C ∧ total_fruits A B Bo C = 915 :=
begin
  sorry
end

end total_fruits_915_l193_193580


namespace find_integers_l193_193699

theorem find_integers (n : ℕ) (h1 : n < 10^100)
  (h2 : n ∣ 2^n) (h3 : n - 1 ∣ 2^n - 1) (h4 : n - 2 ∣ 2^n - 2) :
  n = 2^2 ∨ n = 2^4 ∨ n = 2^16 ∨ n = 2^256 := by
  sorry

end find_integers_l193_193699


namespace student_answered_two_questions_incorrectly_l193_193130

/-
  Defining the variables and conditions for the problem.
  x: number of questions answered correctly,
  y: number of questions not answered,
  z: number of questions answered incorrectly.
-/

theorem student_answered_two_questions_incorrectly (x y z : ℕ) 
  (h1 : x + y + z = 6) 
  (h2 : 8 * x + 2 * y = 20) : z = 2 :=
by
  /- We know the total number of questions is 6.
     And the total score is 20 with the given scoring rules.
     Thus, we need to prove that z = 2 under these conditions. -/
  sorry

end student_answered_two_questions_incorrectly_l193_193130


namespace total_course_selection_schemes_l193_193189

theorem total_course_selection_schemes :
  let n_physical_education := 4
  let n_art := 4
  let total_courses := n_physical_education + n_art in
  let choose2_courses := (Nat.choose n_physical_education 1) * (Nat.choose n_art 1)
  let choose3_courses := (Nat.choose n_physical_education 2 * Nat.choose n_art 1) + (Nat.choose n_physical_education 1 * Nat.choose n_art 2) in
  total_courses = n_physical_education + n_art →
  choose2_courses + choose3_courses = 64 :=
by
  intros n_physical_education n_art total_courses choose2_courses choose3_courses h
  have h_choose2_courses: choose2_courses = 16 := by
    simp [n_physical_education, n_art, Nat.choose]
  have h_choose3_courses: choose3_courses = 48 := by
    simp [n_physical_education, n_art, Nat.choose]
  rw [h_choose2_courses, h_choose3_courses]
  exact Nat.add_eq_right.2 rfl

end total_course_selection_schemes_l193_193189


namespace option_b_correct_l193_193604

theorem option_b_correct (a b c : ℝ) (h1 : a > b) (h2 : c > 0) : ac > bc :=
by sorry

end option_b_correct_l193_193604


namespace total_course_selection_schemes_l193_193260

theorem total_course_selection_schemes (PE_courses : ℕ) (Art_courses : ℕ) : 
  PE_courses = 4 → Art_courses = 4 → 
  (finset.card (finset.powerset_len 2 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b ∈ s), a < PE_courses ∧ b ≥ PE_courses) + 
   finset.card (finset.powerset_len 3 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b c ∈ s), (a < PE_courses ∧ b < PE_courses ∧ c ≥ PE_courses) ∨ 
                        (a < PE_courses ∧ b ≥ PE_courses ∧ c ≥ PE_courses ∧ a ≠ b ≠ c)) = 64 :=
by
  sorry

end total_course_selection_schemes_l193_193260


namespace discount_is_5_percent_l193_193302

-- Defining the conditions
def cost_per_iphone : ℕ := 600
def total_cost_3_iphones : ℕ := 3 * cost_per_iphone
def savings : ℕ := 90

-- Calculating the discount percentage
def discount_percentage : ℕ := (savings * 100) / total_cost_3_iphones

-- Stating the theorem
theorem discount_is_5_percent : discount_percentage = 5 :=
  sorry

end discount_is_5_percent_l193_193302


namespace remove_three_weights_l193_193060

-- Problem Statement:
-- Given weights 1 to n (where n > 6) initially in equilibrium on a balance scale,
-- prove that it's always possible to remove 3 weights and still have the scale in equilibrium.

theorem remove_three_weights (n : ℕ) 
  (h : n > 6) 
  (weights : Finset ℕ) 
  (h_weights : weights = Finset.range (n + 1) \ {0})
  (scale : list (Finset ℕ) × list (Finset ℕ))
  (h_scale_equilibrium : ∑ x in scale.1.bind id, x = ∑ x in scale.2.bind id, x) :
  ∃ (removed_weights : Finset ℕ), 
    removed_weights.card = 3 ∧ 
    (∑ x in (scale.1.bind id).erase₀, x + ∑ x in removed_weights, x = 
    ∑ x in (scale.2.bind id).erase₀, x + ∑ x in removed_weights, x) := 
begin
  sorry
end

end remove_three_weights_l193_193060


namespace find_angle_B_l193_193803

noncomputable def angle_B (a b c : ℝ) (B C : ℝ) : Prop :=
b = 2 * Real.sqrt 3 ∧ c = 2 ∧ C = Real.pi / 6 ∧
(Real.sin B = (b * Real.sin C) / c ∧ b > c → (B = Real.pi / 3 ∨ B = 2 * Real.pi / 3))

theorem find_angle_B :
  ∃ (B : ℝ), angle_B 1 (2 * Real.sqrt 3) 2 B (Real.pi / 6) :=
by
  sorry

end find_angle_B_l193_193803


namespace arc_length_of_curve_8cos3t_8sin3t_l193_193169

def parametric_arc_length (x y: ℝ → ℝ) (t1 t2: ℝ): ℝ :=
  ∫ t in t1..t2, sqrt ((x.deriv t)^2 + (y.deriv t)^2)

theorem arc_length_of_curve_8cos3t_8sin3t: parametric_arc_length (λ t => 8 * (cos t)^3) (λ t => 8 * (sin t)^3) 0 (π / 6) = 3 :=
by
  -- The detailed proof steps would be provided here.
  sorry

end arc_length_of_curve_8cos3t_8sin3t_l193_193169


namespace students_in_each_group_is_9_l193_193996

-- Define the number of students trying out for the trivia teams
def total_students : ℕ := 36

-- Define the number of students who didn't get picked for the team
def students_not_picked : ℕ := 9

-- Define the number of groups the remaining students are divided into
def number_of_groups : ℕ := 3

-- Define the function that calculates the number of students in each group
def students_per_group (total students_not_picked number_of_groups : ℕ) : ℕ :=
  (total - students_not_picked) / number_of_groups

-- Theorem: Given the conditions, the number of students in each group is 9
theorem students_in_each_group_is_9 : students_per_group total_students students_not_picked number_of_groups = 9 := 
by 
  -- proof skipped
  sorry

end students_in_each_group_is_9_l193_193996


namespace tangent_line_equation_l193_193102

noncomputable def f : ℝ → ℝ := λ x, Real.exp (2 - x)

theorem tangent_line_equation :
  ∃ (m : ℝ), (∀ (x y : ℝ), (y = f x) → (y = -e^3 * x)) := by
  sorry

end tangent_line_equation_l193_193102


namespace locus_of_vertex_M_l193_193570

open Set

def Point := ℝ × ℝ

-- Define the vertices and the directed segments
variables (A K N M : Point) (AB AC : Set Point) (d : Point)

-- The conditions:
-- K and N move along AB and AC respectively
-- Sides of triangle KMN are parallel to three given lines
def moving_vertices (K N : Point) (AB AC : Set Point) : Prop := 
  K ∈ AB ∧ N ∈ AC

-- Define the direction vector (which should be computed based on the problem context)
def direction_vector (A M : Point) : Point := (M.1 - A.1, M.2 - A.2)

-- The sought locus is the ray starting from A and passing through some point M
def locus_of_M (A d : Point) : Set Point := 
  { P | ∃ (t : ℝ), t ≥ 0 ∧ P = (A.1 + t * d.1, A.2 + t * d.2) }

-- Definition of our problem statement
theorem locus_of_vertex_M :
  ∀ (A K N M : Point) (AB AC : Set Point) (d : Point),
  moving_vertices K N AB AC → 
  parallel_sides_of_triangle K M N →
  M ∈ locus_of_M A d :=
sorry

end locus_of_vertex_M_l193_193570


namespace total_surface_area_hemisphere_l193_193909

theorem total_surface_area_hemisphere (A : ℝ) (r : ℝ) : (A = 100 * π) → (r = 10) → (2 * π * r^2 + A = 300 * π) :=
by
  intro hA hr
  sorry

end total_surface_area_hemisphere_l193_193909


namespace largest_of_seven_consecutive_numbers_l193_193990

theorem largest_of_seven_consecutive_numbers (a b c d e f g : ℤ) (h1 : a + 1 = b)
                                             (h2 : b + 1 = c) (h3 : c + 1 = d)
                                             (h4 : d + 1 = e) (h5 : e + 1 = f)
                                             (h6 : f + 1 = g)
                                             (h_avg : (a + b + c + d + e + f + g) / 7 = 20) :
    g = 23 :=
by
  sorry

end largest_of_seven_consecutive_numbers_l193_193990


namespace fg_of_5_eq_140_l193_193440

def g (x : ℝ) : ℝ := 4 * x + 5
def f (x : ℝ) : ℝ := 6 * x - 10

theorem fg_of_5_eq_140 : f (g 5) = 140 := by
  sorry

end fg_of_5_eq_140_l193_193440


namespace course_selection_count_l193_193215

-- Definitions for the conditions
def num_PE_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_PE_courses + num_art_courses

-- The number of different course selection schemes
theorem course_selection_count : 
  (choose num_PE_courses 1) * (choose num_art_courses 1) + 
  (choose num_PE_courses 2) * (choose num_art_courses 1) + 
  (choose num_PE_courses 1) * (choose num_art_courses 2) = 64 := 
by sorry

end course_selection_count_l193_193215


namespace course_selection_count_l193_193220

-- Definitions for the conditions
def num_PE_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_PE_courses + num_art_courses

-- The number of different course selection schemes
theorem course_selection_count : 
  (choose num_PE_courses 1) * (choose num_art_courses 1) + 
  (choose num_PE_courses 2) * (choose num_art_courses 1) + 
  (choose num_PE_courses 1) * (choose num_art_courses 2) = 64 := 
by sorry

end course_selection_count_l193_193220


namespace curve_is_line_l193_193688

-- Define the problem in terms of polar to Cartesian conversion
noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

-- The given polar equation
def polar_eq (θ : ℝ) : ℝ :=
  1 / (Real.sin θ + Real.cos θ)

-- The conclusion that this represents a line in Cartesian coordinates
theorem curve_is_line (θ : ℝ) : 
  let r := polar_eq θ
  let (x, y) := polar_to_cartesian r θ
  x + y = 1 := sorry

end curve_is_line_l193_193688


namespace solve_system_l193_193903

theorem solve_system :
  ∃ x y z : ℝ, (x = 20 ∧ y = 22 ∧ z = 23 ∧ 
  (x^2 - 23*y - 25*z = -681) ∧ 
  (y^2 - 21*x - 21*z = -419) ∧ 
  (z^2 - 19*x - 21*y = -313)) :=
by
  use 20, 22, 23
  split
  . refl
  split
  . refl
  split
  . refl
  split
  . sorry
  split
  . sorry
  . sorry

end solve_system_l193_193903


namespace total_photos_in_newspaper_l193_193447

-- Definitions for the conditions
def sectionA_pages : ℕ := 25
def sectionA_photos_per_page : ℕ := 4
def sectionB_pages : ℕ := 18
def sectionB_photos_per_page : ℕ := 6
def sectionC_Monday_pages : ℕ := 12
def sectionC_Monday_photos_per_page : ℕ := 5
def sectionC_Tuesday_pages : ℕ := 15
def sectionC_Tuesday_photos_per_page : ℕ := 3

-- Proving the total number of photos
theorem total_photos_in_newspaper : 
  sectionA_pages * sectionA_photos_per_page + 
  sectionB_pages * sectionB_photos_per_page + 
  (sectionC_Monday_pages * sectionC_Monday_photos_per_page + 
   sectionC_Tuesday_pages * sectionC_Tuesday_photos_per_page + 
  (sectionC_Monday_pages * sectionC_Monday_photos_per_page + 
   sectionC_Tuesday_pages * sectionC_Tuesday_photos_per_page) = 521 :=
by 
  sorry

end total_photos_in_newspaper_l193_193447


namespace strongest_teams_different_groups_l193_193825

theorem strongest_teams_different_groups (total_teams : ℕ) (group_size : ℕ) 
  (strongest_teams : list ℕ) (h1 : total_teams = 20) (h2 : group_size = 10) 
  (h3 : strongest_teams.length = 2) : ℚ :=
by 
  have h4 : total_teams - 1 = 19 := by linarith,
  exact (group_size : ℚ) / (h4 : ℚ)
-- The expected probability is 10/19

end strongest_teams_different_groups_l193_193825


namespace find_sphere_center_l193_193037

variable (O : Point)
variable (a b c : ℝ)
variable (A B C : Point)
variable (p q r : ℝ)
variable (center : Point → Point → Point → Point → Point)

-- Definitions based on conditions
def is_origin (P : Point) : Prop := P = (0, 0, 0)
def fixed_point (x y z : ℝ) (P : Point) : Prop := P = (x, y, z)
def intersects_axes (A B C : Point) : Prop := 
  ∃ α β γ : ℝ, A = (α, 0, 0) ∧ B = (0, β, 0) ∧ C = (0, 0, γ)
def center_of_sphere (P Q R S T : Point) : Prop := T = (p, q, r)

-- The main theorem
theorem find_sphere_center (hO : is_origin O)
    (hFixed : fixed_point a b c (a, b, c))
    (hInter : intersects_axes A B C)
    (hCenter : center_of_sphere O A B C (p, q, r))
    : 
      2 * a / p + 2 * b / q + 2 * c / r = 2 :=
sorry

end find_sphere_center_l193_193037


namespace max_plus_min_value_eq_six_l193_193384

variable (a : ℝ)
variable (f : ℝ → ℝ)

namespace Proof
  theorem max_plus_min_value_eq_six (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1)
    (h_f_def : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x = (4 * a^x + 2) / (a^x + 1) + x * cos x)
    (M : ℝ) (hm : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x ≤ M)
    (N : ℝ) (hn : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → N ≤ f x)
    (h_M_max : ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ M = f x)
    (h_N_min : ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ N = f x) :
    M + N = 6 := 
  by
    sorry
end Proof

end max_plus_min_value_eq_six_l193_193384


namespace measure_angle_QPR_l193_193671

-- Define the conditions of the problem
variables (D E F P Q R : Type)
variables [euclidean_geometry D] [euclidean_geometry E] [euclidean_geometry F]
variables [euclidean_geometry P] [euclidean_geometry Q] [euclidean_geometry R]
variables (Ω : circle)

-- Given conditions
def condition_1 : Prop := excircle Ω (triangle D E F) D
def condition_2 : Prop := circumcircle Ω (triangle P Q R)
def condition_3 : Prop := on_line P E F
def condition_4 : Prop := on_line Q D E
def condition_5 : Prop := on_line R D F
def condition_6 : angle D = 50
def condition_7 : angle E = 70
def condition_8 : angle F = 60

-- Prove the measured angle
theorem measure_angle_QPR :
  condition_1 → condition_2 → condition_3 → condition_4 → condition_5 → condition_6 → condition_7 → condition_8 → 
  angle QPR = 60 :=
by {
  intros h1 h2 h3 h4 h5 h6 h7 h8,
  sorry
}

end measure_angle_QPR_l193_193671


namespace village_pop_growth_is_101_l193_193805
noncomputable def population_growth : ℕ :=
  let c := 7                 -- Pop in 1991 = c^3
  let d := c + 1             -- Pop in 2001 = d^3 + 10
  let e := 9                 -- Pop in 2011 = e^3
  in (e^3 - c^3) * 100 / c^3 -- Percent increase calculation

theorem village_pop_growth_is_101 :
  population_growth = 101 :=
by
  sorry

end village_pop_growth_is_101_l193_193805


namespace isosceles_right_triangle_ratio_l193_193657

theorem isosceles_right_triangle_ratio (a : ℝ) (h : a > 0) : (2 * a) / (Real.sqrt (a^2 + a^2)) = Real.sqrt 2 :=
by
  sorry

end isosceles_right_triangle_ratio_l193_193657


namespace int_fourth_power_l193_193439

noncomputable def x : ℕ := 3^12 * 3^8

theorem int_fourth_power : ∃ y : ℕ, y^4 = x ∧ y = 243 :=
by
  use 243
  split
  . rw [x, pow_succ, pow_succ, mul_assoc, mul_comm (3 * 3^11) (3^8), ← pow_add, show 12 + 8 = 20 from rfl, pow_succ]
  . sorry

end int_fourth_power_l193_193439


namespace pyramid_volume_l193_193880

structure Point :=
  (x y z : ℝ)

def distance (p q : Point) : ℝ :=
  real.sqrt ((q.x - p.x)^2 + (q.y - p.y)^2 + (q.z - p.z)^2)

noncomputable def volume_pyramid (P A B C : Point) : ℝ :=
  let area_ABC := 0.5 * distance P A * distance P B in
  (1 / 3) * area_ABC * distance P C

theorem pyramid_volume (P A B C : Point)
  (h1 : distance P A = 12)
  (h2 : distance P B = 12)
  (h3 : distance P C = 7)
  (h4 : P.x ≠ A.x ∧ P.y ≠ A.y ∧ P.z ≠ A.z) -- Ensure P, A, B, C are not collinear
  (h5 : (P.x - A.x) * (P.x - B.x) + (P.y - A.y) * (P.y - B.y) + (P.z - A.z) * (P.z - B.z) = 0)
  (h6 : (P.x - A.x) * (P.x - C.x) + (P.y - A.y) * (P.y - C.y) + (P.z - A.z) * (P.z - C.z) = 0)
  (h7 : (P.x - B.x) * (P.x - C.x) + (P.y - B.y) * (P.y - C.y) + (P.z - B.z) * (P.z - C.z) = 0) :
  volume_pyramid P A B C = 168 :=
sorry

end pyramid_volume_l193_193880


namespace proof_problem_l193_193860

noncomputable def roots : Type := { r : ℂ // (∃ s t, r ≠ s ∧ s ≠ t ∧ t ≠ r ∧ 
  polynomial.eval r (polynomial.C 1 * polynomial.X ^ 3 + polynomial.C (-6) * polynomial.X ^ 2 + 
  polynomial.C 11 * polynomial.X - polynomial.C 16) = 0 ∧ 
  polynomial.eval s (polynomial.C 1 * polynomial.X ^ 3 + polynomial.C (-6) * polynomial.X ^ 2 + 
  polynomial.C 11 * polynomial.X - polynomial.C 16) = 0 ∧ 
  polynomial.eval t (polynomial.C 1 * polynomial.X ^ 3 + polynomial.C (-6) * polynomial.X ^ 2 + 
  polynomial.C 11 * polynomial.X - polynomial.C 16) = 0) }

theorem proof_problem (r s t : roots) : 
  (r.1 + s.1) / t.1 + (s.1 + t.1) / r.1 + (t.1 + r.1) / s.1 = 11 / 8 :=
by
  -- proof goes here
  sorry

end proof_problem_l193_193860


namespace possible_values_of_k_l193_193394

section
variables {V : Type} [Fintype V] [DecidableEq V] (G : SimpleGraph V)
variable [Fintype (G.Vertex)]
variable [DecidableRel G.adj]

def vertices_count := Fintype.card V

noncomputable def num_vertices_group : ℕ := 2019

noncomputable def k_is_possible_value (k : ℕ) : Prop := 
  k = 2019 ∨ k = 2019^2 - 2019 ∨ k = 2019^2

theorem possible_values_of_k
  (G : SimpleGraph (Fin 4038)) 
  (h : G.Vertices.card = 4038)
  (split : ∃ (A B : Finset (Fin 4038)), A.card = 2019 ∧ B.card = 2019 ∧ A ∪ B = Finset.univ ∧ A ∩ B = ∅)
  (k : ℕ)
  (H : ∃ (A B : Finset (Fin 4038)), A.card = 2019 ∧ B.card = 2019 ∧ A ∩ B = ∅ ∧ G.edgeFinset.filter (λ e, (e.val.1 ∈ A ∧ e.val.2 ∈ B) ∨ (e.val.1 ∈ B ∧ e.val.2 ∈ A)).card = k) :
  k_is_possible_value k :=
begin
  sorry
end
end

end possible_values_of_k_l193_193394


namespace company_l193_193630

theorem company's_output_value_2003 :
  ∀ (P0 : ℝ) (r : ℝ) (n : ℕ), P0 = 1000 → r = 0.10 → n = 3 → (P0 * (1 + r)^n) = 1331 :=
by
  intros P0 r n hP0 hr hn
  rw [hP0, hr, hn]
  norm_num
  -- Provided correct execution and further proof elaboration are left.
  sorry

end company_l193_193630


namespace train_overtake_l193_193591

theorem train_overtake :
  let speedA := 30 -- speed of Train A in miles per hour
  let speedB := 38 -- speed of Train B in miles per hour
  let lead_timeA := 2 -- lead time of Train A in hours
  let distanceA := speedA * lead_timeA -- distance traveled by Train A in the lead time
  let t := 7.5 -- time in hours Train B travels to catch up Train A
  let total_distanceB := speedB * t -- total distance traveled by Train B in time t
  total_distanceB = 285 := 
by
  sorry

end train_overtake_l193_193591


namespace team_points_difference_l193_193458

   -- Definitions for points of each member
   def Max_points : ℝ := 7
   def Dulce_points : ℝ := 5
   def Val_points : ℝ := 4 * (Max_points + Dulce_points)
   def Sarah_points : ℝ := 2 * Dulce_points
   def Steve_points : ℝ := 2.5 * (Max_points + Val_points)

   -- Definition for total points of their team
   def their_team_points : ℝ := Max_points + Dulce_points + Val_points + Sarah_points + Steve_points

   -- Definition for total points of the opponents' team
   def opponents_team_points : ℝ := 200

   -- The main theorem to prove
   theorem team_points_difference : their_team_points - opponents_team_points = 7.5 := by
     sorry
   
end team_points_difference_l193_193458


namespace conditional_probability_l193_193184

variable (Ω : Type) 

-- Define events A and B as subsets of Ω
variable (A B : Set Ω)

-- Define probability measure P
variable (P : MeasureTheory.Measure Ω)

-- Define the given conditions as hypotheses
hypothesis h_A : P A = 1 / 2
hypothesis h_AB : P (A ∩ B) = 1 / 6

-- State the theorem: the conditional probability
theorem conditional_probability : P[B | A] = 1 / 3 :=
by
    sorry

end conditional_probability_l193_193184


namespace general_term_a_sum_first_n_terms_b_l193_193733

noncomputable theory
open_locale big_operators

-- Given sequence {a_n} with the sum of its first n terms denoted as S_n, and a_1 = 0
-- For any n in the set of positive natural numbers, we have n * a_(n+1) = S_n + n * (n + 1)

-- We aim to find the general term formula for the sequence {a_n}
def sequence_a_n (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ ∀ n : ℕ, n > 0 → n * a (n + 1) = S n + n * (n + 1)

-- We aim to prove that the general term formula for a_n is a_n = 2n - 2
theorem general_term_a (a S : ℕ → ℤ) (h : sequence_a_n a S) :
  ∀ n : ℕ, n > 0 → a n = 2 * n - 2 :=
sorry

-- If the sequence {b_n} satisfies a_n + log_2 n = log_2 b_n
-- We aim to find the sum of the first n terms of the sequence {b_n}, denoted as T_n

-- We define the sequence {b_n}
def sequence_b_n (a b : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a n + (nat.log n / nat.log 2) = nat.log (b n) / nat.log 2

-- We define the sum of the first n terms of {b_n}, denoted as T_n
def sum_b_n (b : ℕ → ℤ) (T : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, T n = ∑ i in finset.range n, b (i + 1)

-- We aim to prove the correct answer for T_n
theorem sum_first_n_terms_b (a b T : ℕ → ℤ) (ha : sequence_a_n a (λ n, ∑ i in finset.range n, a (i + 1)))
  (hb : sequence_b_n a b) (hs : sum_b_n b T) :
  ∀ n : ℕ, n > 0 → T n = (1 : ℚ) / 9 * (3 * n - 1) * 4 ^ n + 1 / 9 :=
sorry

end general_term_a_sum_first_n_terms_b_l193_193733


namespace extremum_a_value_l193_193766

theorem extremum_a_value (a : ℝ) (h : ∃ x : ℝ, x = 1 ∧ deriv (fun x : ℝ => a * x - 2 * log x + 2) x = 0) : a = 2 :=
sorry

end extremum_a_value_l193_193766


namespace intersection_M_N_l193_193742

def M : Set ℝ := { x : ℝ | Real.log x / Real.log 2 < 2 }
def N : Set ℝ := { x : ℝ | x^2 - x - 2 < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l193_193742


namespace repeating_block_length_seven_thirteen_l193_193697

theorem repeating_block_length_seven_thirteen : (∃ n : ℕ, repeating_block_length (7 / 13) = 6) := sorry

end repeating_block_length_seven_thirteen_l193_193697


namespace quinn_books_per_week_l193_193884

theorem quinn_books_per_week (books_per_coupon : ℕ) (weeks : ℕ) (coupons : ℕ) (books_total : ℕ) :
  books_per_coupon = 5 →
  weeks = 10 →
  coupons = 4 →
  books_total = books_per_coupon * coupons →
  books_total / weeks = 2 :=
begin
  intros h1 h2 h3 h4,
  rw [h1, h2, h3] at h4,
  have h5 : books_total = 20, by { exact h4 },
  rw h2 at h5,
  exact (nat.div_eq_of_eq_mul_left (by norm_num : 10 ≠ 0) h5).symm,
end

end quinn_books_per_week_l193_193884


namespace Delta_max_success_ratio_l193_193334

theorem Delta_max_success_ratio :
  ∀ (x y z w u v : ℝ),
  (0 < x / y ∧ x / y < 2/3) ∧
  (0 < z / w ∧ z / w < 4/5) ∧
  (0 < u / v ∧ u / v < 9/10) ∧
  (y + w + v = 600) →
  (6*x + 5*z + (40/3)*u < 2400) →
  x + z + u ≤ 399 :=
begin
  sorry
end

end Delta_max_success_ratio_l193_193334


namespace problem_l193_193891

theorem problem (
  n : ℤ
) : 
  n = 6 ∨ n = -6 ∧ (36 ∈ {32, 33, 34, 35, 36}) :=
by 
  sorry

end problem_l193_193891


namespace sqrt_of_quarter_l193_193952

-- Definitions as per conditions
def is_square_root (x y : ℝ) : Prop := x^2 = y

-- Theorem statement proving question == answer given conditions
theorem sqrt_of_quarter : is_square_root 0.5 0.25 ∧ is_square_root (-0.5) 0.25 ∧ (∀ x, is_square_root x 0.25 → (x = 0.5 ∨ x = -0.5)) :=
by
  -- Skipping proof with sorry
  sorry

end sqrt_of_quarter_l193_193952


namespace b_cong_zero_l193_193475

theorem b_cong_zero (a b c m : ℤ) (h₀ : 1 < m) (h : ∀ (n : ℕ), (a ^ n + b * n + c) % m = 0) : b % m = 0 :=
  sorry

end b_cong_zero_l193_193475


namespace countDivisorsOf72Pow8_l193_193696

-- Definitions of conditions in Lean 4
def isPerfectSquare (a b : ℕ) : Prop := a % 2 = 0 ∧ b % 2 = 0
def isPerfectCube (a b : ℕ) : Prop := a % 3 = 0 ∧ b % 3 = 0
def isPerfectSixthPower (a b : ℕ) : Prop := a % 6 = 0 ∧ b % 6 = 0

def countPerfectSquares : ℕ := 13 * 9
def countPerfectCubes : ℕ := 9 * 6
def countPerfectSixthPowers : ℕ := 5 * 3

-- The proof problem to prove the number of such divisors is 156
theorem countDivisorsOf72Pow8:
  (countPerfectSquares + countPerfectCubes - countPerfectSixthPowers) = 156 :=
by
  sorry

end countDivisorsOf72Pow8_l193_193696


namespace coefficient_x21_expansion_l193_193322

theorem coefficient_x21_expansion : let f := (1 + x + x^2 + ... + x^20) * (1 + x + x^2 + ... + x^15)^2 in
  ∃ c : ℤ, c = 51 ∧ coeff f 21 = c := sorry

end coefficient_x21_expansion_l193_193322


namespace minimum_value_l193_193437

theorem minimum_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : log 4 (a + 4 * b) = log 2 (2 * sqrt (a * b))) : 
a + b ≥ 9 / 4 :=
sorry

end minimum_value_l193_193437


namespace area_ratio_l193_193917

-- Define the problem conditions
def Square (s : ℝ) := s > 0
def Rectangle (longer shorter : ℝ) := longer = 1.2 * shorter ∧ shorter = 0.8 * shorter

-- Define a function to calculate the area of square
def area_square (s : ℝ) : ℝ := s * s

-- Define a function to calculate the area of rectangle
def area_rectangle (longer shorter : ℝ) : ℝ := longer * shorter

-- State the proof problem
theorem area_ratio (s : ℝ) (h_square : Square s) :
  let longer := 1.2 * s
  let shorter := 0.8 * s
  area_rectangle longer shorter / area_square s = 24 / 25 :=
by
  sorry

end area_ratio_l193_193917


namespace packs_sold_in_other_villages_l193_193607

theorem packs_sold_in_other_villages
  (packs_v1 : ℕ) (packs_v2 : ℕ) (h1 : packs_v1 = 23) (h2 : packs_v2 = 28) :
  packs_v1 + packs_v2 = 51 := 
by {
  sorry
}

end packs_sold_in_other_villages_l193_193607


namespace find_f_l193_193869

noncomputable def f : ℕ → ℕ :=
  λ n, if 0 < n then ((n * (n + 1)) / 2 : ℕ) + ∑ i in Finset.range n, g i 1 else 0

noncomputable def g : ℕ → ℕ → ℕ :=
  λ n m, if 0 < n ∧ 0 < m then some_valued_function n m else 0

axiom f_base : f 1 = 1

axiom f_cond : ∀ m n : ℕ, 0 < m → 0 < n → f (m + n) ≥ f m + f n + m * n

axiom g_cond : ∀ n m : ℕ, 0 < n → 0 < m → g n m ≥ 0

theorem find_f (n : ℕ) (h : 0 < n) :
  f n = (n * (n + 1)) / 2 + ∑ i in Finset.range n, g i 1 :=
sorry

end find_f_l193_193869


namespace find_distance_ab_l193_193994

noncomputable theory
open_locale classical

-- Define the problem constants
variables (a k : ℝ) (x y : ℝ) (hxy : x > y)

-- Define the distance from A to B as a variable
def distance_ab : ℝ := 2 * a * k

-- First encounter equation: (z + a) / (z - a) = x / y
def first_encounter (z : ℝ) : Prop := (z + a) / (z - a) = x / y

-- Second encounter equation: ((2k + 1) * z) / ((2k - 1) * z) = x / y
def second_encounter (z : ℝ) : Prop := ((2 * k + 1) * z) / ((2 * k - 1) * z) = x / y

-- The theorem statement
theorem find_distance_ab (h1 : first_encounter a k x y (distance_ab a k))
                         (h2 : second_encounter a k x y (distance_ab a k)) :
  distance_ab a k = 2 * a * k := 
sorry

end find_distance_ab_l193_193994


namespace gcd_4004_10010_l193_193703

theorem gcd_4004_10010 : Nat.gcd 4004 10010 = 2002 :=
by
  have h1 : 4004 = 4 * 1001 := by norm_num
  have h2 : 10010 = 10 * 1001 := by norm_num
  sorry

end gcd_4004_10010_l193_193703


namespace triangle_area_RSZ_l193_193004

theorem triangle_area_RSZ {EFGH : Parallelogram} (m : ℝ)
  (ER bisects FG at X : Point)
  (ER meets EH at R : Point) 
  (GS bisects EH at Y : Point)
  (GS meets EF at S : Point)
  (ER meets GS at Z : Point)
  (area_EFGH : ParallelogramArea EFGH = m):
  TriangleArea (Triangle.mk R S Z) = 9 * m / 8 := 
sorry

end triangle_area_RSZ_l193_193004


namespace find_speed_l193_193876

variable (d : ℝ) (t : ℝ)
variable (h1 : d = 50 * (t + 1/12))
variable (h2 : d = 70 * (t - 1/12))

theorem find_speed (d t : ℝ)
  (h1 : d = 50 * (t + 1/12))
  (h2 : d = 70 * (t - 1/12)) :
  58 = d / t := by
  sorry

end find_speed_l193_193876


namespace course_selection_count_l193_193221

-- Definitions for the conditions
def num_PE_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_PE_courses + num_art_courses

-- The number of different course selection schemes
theorem course_selection_count : 
  (choose num_PE_courses 1) * (choose num_art_courses 1) + 
  (choose num_PE_courses 2) * (choose num_art_courses 1) + 
  (choose num_PE_courses 1) * (choose num_art_courses 2) = 64 := 
by sorry

end course_selection_count_l193_193221


namespace course_selection_schemes_l193_193212

theorem course_selection_schemes :
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  choose_2_courses + choose_3_courses = 64 :=
by
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  show choose_2_courses + choose_3_courses = 64 from sorry

end course_selection_schemes_l193_193212


namespace function_max_value_l193_193799

theorem function_max_value (a b : ℝ) (f : ℝ → ℝ) 
    (h0 : ∀ x, f x = |a * sin x + b * cos x - 1| + |b * sin x - a * cos x|)
    (h1 : ∃ x, f x = 11) : a^2 + b^2 = 50 :=
sorry

end function_max_value_l193_193799


namespace second_multiple_of_three_l193_193954

theorem second_multiple_of_three (n : ℕ) (h : 3 * (n - 1) + 3 * (n + 1) = 150) : 3 * n = 75 :=
sorry

end second_multiple_of_three_l193_193954


namespace probability_of_three_digit_number_div_by_three_l193_193578

noncomputable def probability_three_digit_div_by_three : ℚ :=
  let digit_mod3_groups := 
    {rem0 := {3, 6, 9}, rem1 := {1, 4, 7}, rem2 := {2, 5, 8}} in
  let valid_combinations :=
    (finset.card (finset.powersetLen 3 digit_mod3_groups.rem0) +
     (finset.card digit_mod3_groups.rem0 * finset.card digit_mod3_groups.rem1 * finset.card digit_mod3_groups.rem2) +
     finset.card (finset.powersetLen 3 digit_mod3_groups.rem1) +
     finset.card (finset.powersetLen 3 digit_mod3_groups.rem2))
  in
  let total_combinations := finset.card (finset.powersetLen 3 (finset.univ : finset (fin 9))) in
  (valid_combinations : ℚ) / total_combinations

theorem probability_of_three_digit_number_div_by_three :
  probability_three_digit_div_by_three = 5 / 14 := by
  -- provide proof here
  sorry

end probability_of_three_digit_number_div_by_three_l193_193578


namespace domain_and_range_of_k_l193_193290

noncomputable def h : ℝ → ℝ := sorry -- assume h is defined somehow

theorem domain_and_range_of_k :
  (2 ≤ x ∧ x ≤ 5) ∧ (-1 ≤ k x ∧ k x ≤ 0) :=
begin
  -- assume the function h has known properties
  have h_domain : ∀ x, (1 ≤ x ∧ x ≤ 4) → (2 ≤ h x ∧ h x ≤ 3) := sorry,
  let k := λ x, 2 - h (x - 1),
  -- define k and prove the domain and range based on h's properties
  split,
  { -- Prove domain of k
    intros x,
    intro h1,
    have h2 : 1 ≤ (x - 1) ∧ (x - 1) ≤ 4, from sorry,
    exact ⟨sorry, sorry⟩ -- fill in the details based on the transformation
  },
  { -- Prove range of k
    intros y,
    intro h3,
    have h4 : -1 ≤ (2 - y) ∧ (2 - y) ≤ 0, from sorry,
    exact ⟨sorry, sorry⟩ -- fill in the details based on the transformation
  }
end

end domain_and_range_of_k_l193_193290


namespace largest_real_root_f_iter_l193_193487

def f (x : ℝ) : ℝ := x^2 + 12 * x + 30

theorem largest_real_root_f_iter (x : ℝ) :
  (∃ r : ℝ, r = -6 + real_root 8 6) →
  f (f (f x)) = 0 → x = -6 + real_root 8 6 := sorry

end largest_real_root_f_iter_l193_193487


namespace distance_focus_directrix_l193_193823

def parabola_focus_distance (p : ℝ) (h : p > 0) : ℝ :=
  real.sqrt ( (p / 2 - 1)^2 + 0^2 )

noncomputable def solve_parabola_distance (p : ℝ) (h : p > 0) (dist_point_focus : parabola_focus_distance p h = 4) : ℝ :=
  p

theorem distance_focus_directrix : solve_parabola_distance 6 _ _ = 6 :=
by
  sorry

end distance_focus_directrix_l193_193823


namespace quintuple_count_l193_193704

theorem quintuple_count :
  ∃ (S : Set (ℝ × ℝ × ℝ × ℝ × ℝ)), 
    (∀ (x ∈ S), ∃ a b c d e, 
      (x = (a, b, c, d, e)) ∧ 
      (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0) ∧ 
      (a^2 + b^2 + c^2 + d^2 + e^2 = 5) ∧ 
      ((a + b + c + d + e) * (a^3 + b^3 + c^3 + d^3 + e^3) = 25)) ∧ 
    (S.toFinset.card = 31) :=
by
  sorry

end quintuple_count_l193_193704


namespace symmetry_of_x_l193_193049

theorem symmetry_of_x (n : ℕ) (x : Fin n → ℝ)
  (h₁ : ∀ i j : Fin n, i < j → x i < x j)
  (h₂ : ∀ i : Fin n, 0 < x i ∧ x i < 1)
  (h₃ : ∀ i : Fin n, (∑ j in ({0, (n+1) \ {i}}).attach, (1 / (x i - x j))) = 0) :
  ∀ i : Fin n, x (n + 1 - i) = 1 - x i :=
sorry

end symmetry_of_x_l193_193049


namespace counterexample_to_proposition_l193_193156

theorem counterexample_to_proposition (x y : ℤ) (h1 : x = -1) (h2 : y = -2) : x > y ∧ ¬ (x^2 > y^2) := by
  sorry

end counterexample_to_proposition_l193_193156


namespace select_four_people_for_tasks_l193_193129

theorem select_four_people_for_tasks : 
    ∃ (N : ℕ), N = Nat.choose 10 4 * Nat.choose 4 2 * Nat.choose 2 1 := by
  exists 2520
  sorry

end select_four_people_for_tasks_l193_193129


namespace tangent_line_through_origin_l193_193100

def function_f (x : ℝ) : ℝ := Real.exp (2 - x)

theorem tangent_line_through_origin (x y : ℝ) : 
  (∃ x₀ : ℝ, y = function_f x₀ - function_f x₀ * (x - x₀) ∧ function_f (x₀) = Real.exp (2 - x₀) ∧ x₀ = -1) → 
  y = -Real.exp 3 * x := 
sorry

end tangent_line_through_origin_l193_193100


namespace square_area_possible_l193_193824

open Complex

noncomputable def square_area (z : ℂ) : ℝ := 
  let z1 := z
  let z2 := z^2
  let z3 := z^3
  let z4 := z^4
  if (z2 - z1) ≠ 0 ∧ (z3 - z2) ≠ 0 ∧ (z4 - z3) ≠ 0 ∧ (z1 - z4) ≠ 0 then
    abs (z1 - z2) ^ 2 * abs (z1 - z3) ^ 2
  else 
    0

theorem square_area_possible (z : ℂ) (hz : z ≠ 0 ∧ z^2 ≠ 0 ∧ z^3 ≠ 0 ∧ z^4 ≠ 0 ∧ 
  abs (z^4 - z^3) = abs (Complex.i * (z^3 - z^2)) ∧ 
  abs (z^3 - z^2) = abs (Complex.i * (z^2 - z))) :
  square_area z = 10 :=
sorry

end square_area_possible_l193_193824


namespace total_numbers_l193_193531

-- Setting up constants and conditions
variables (n : ℕ)
variables (s1 s2 s3 : ℕ → ℝ)

-- Conditions
axiom avg_all : (s1 n + s2 n + s3 n) / n = 2.5
axiom avg_2_1 : s1 2 / 2 = 1.1
axiom avg_2_2 : s2 2 / 2 = 1.4
axiom avg_2_3 : s3 2 / 2 = 5.0

-- Proposed theorem to prove
theorem total_numbers : n = 6 :=
by
  sorry

end total_numbers_l193_193531


namespace trig_identity_l193_193386

theorem trig_identity (α β θ : ℝ) 
  (h1 : sin θ + cos θ = 2 * sin α)
  (h2 : sin (2 * θ) = 2 * sin β ^ 2) : 
  cos (2 * β) = 2 * cos (2 * α) :=
by sorry

end trig_identity_l193_193386


namespace union_of_A_and_B_l193_193850

variable {α : Type*}

def A (x : ℝ) : Prop := x - 1 > 0
def B (x : ℝ) : Prop := 0 < x ∧ x ≤ 3

theorem union_of_A_and_B : ∀ x : ℝ, (A x ∨ B x) ↔ (0 < x) :=
by
  sorry

end union_of_A_and_B_l193_193850


namespace total_course_selection_schemes_l193_193259

theorem total_course_selection_schemes (PE_courses : ℕ) (Art_courses : ℕ) : 
  PE_courses = 4 → Art_courses = 4 → 
  (finset.card (finset.powerset_len 2 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b ∈ s), a < PE_courses ∧ b ≥ PE_courses) + 
   finset.card (finset.powerset_len 3 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b c ∈ s), (a < PE_courses ∧ b < PE_courses ∧ c ≥ PE_courses) ∨ 
                        (a < PE_courses ∧ b ≥ PE_courses ∧ c ≥ PE_courses ∧ a ≠ b ≠ c)) = 64 :=
by
  sorry

end total_course_selection_schemes_l193_193259


namespace prob_divisible_by_3_of_three_digits_l193_193575

-- Define the set of digits available
def digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Three digits are to be chosen from this set
def choose_three_digits (s : Finset ℕ) : Finset (Finset ℕ) := s.powerset.filter (λ x, x.card = 3)

-- Define the property of the sum of digits being divisible by 3
def divisible_by_3 (s : Finset ℕ) : Prop := s.sum id % 3 = 0

-- Total combinations of choosing 3 out of 9 digits
def total_combinations : ℕ := (digits.card.choose 3)

-- Valid combinations where sum of digits is divisible by 3
def valid_combinations : Finset (Finset ℕ) := (choose_three_digits digits).filter divisible_by_3

-- Finally, the probability of a three-digit number being divisible by 3
def probability : ℕ × ℕ := (valid_combinations.card, total_combinations)

theorem prob_divisible_by_3_of_three_digits :
  probability = (5, 14) :=
by
  -- Proof to be filled
  sorry

end prob_divisible_by_3_of_three_digits_l193_193575


namespace problem_inequality_l193_193554

theorem problem_inequality (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 := 
by 
  sorry

end problem_inequality_l193_193554


namespace range_b_of_min_value_l193_193417

def f (x a b : ℝ) : ℝ := x + 2 * b / x + a

theorem range_b_of_min_value 
  (a b : ℝ)
  (h_a : a > 0)
  (h_f_min : ∃ x ∈ Set.Ici a, f x a b = 2) : 
  b < 1 / 2 :=
sorry

end range_b_of_min_value_l193_193417


namespace isosceles_triangle_sine_base_angle_l193_193955

theorem isosceles_triangle_sine_base_angle (m : ℝ) (θ : ℝ) 
  (h1 : m > 0)
  (h2 : θ > 0 ∧ θ < π / 2)
  (h_base_height : m * (Real.sin θ) = (m * 2 * (Real.sin θ) * (Real.cos θ))) :
  Real.sin θ = (Real.sqrt 15) / 4 := 
sorry

end isosceles_triangle_sine_base_angle_l193_193955


namespace complete_the_square_solution_l193_193152

theorem complete_the_square_solution :
  ∃ c d, (∀ x: ℝ, x^2 + 6 * x - 5 = (x + c)^2 - d) ∧ (d = 14) :=
by
  use 3, 14
  intro x
  sorry

end complete_the_square_solution_l193_193152


namespace closest_integer_to_cube_root_l193_193140

theorem closest_integer_to_cube_root (a b c : ℤ) (h1 : a = 7^3) (h2 : b = 9^3) (h3 : c = 3) : 
  Int.round (Real.cbrt (a + b + c)) = 10 := by
  sorry

end closest_integer_to_cube_root_l193_193140


namespace ratio_area_of_rectangle_to_square_l193_193933

theorem ratio_area_of_rectangle_to_square (s : ℝ) :
  (1.2 * s * 0.8 * s) / (s * s) = 24 / 25 :=
by
  sorry

end ratio_area_of_rectangle_to_square_l193_193933


namespace total_selection_schemes_l193_193280

-- Define the given conditions
def num_phys_ed_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_phys_ed_courses + num_art_courses
def valid_course_combos : finset ℕ := {2, 3}

-- Define the number of selection schemes with the constraints
def selection_schemes : nat :=
  (num_phys_ed_courses.choose 1 * num_art_courses.choose 1) + 
  (num_phys_ed_courses.choose 2 * num_art_courses.choose 1 + 
   num_phys_ed_courses.choose 1 * num_art_courses.choose 2)

-- State the theorem to be proved
theorem total_selection_schemes : selection_schemes = 64 := by
  sorry

end total_selection_schemes_l193_193280


namespace increasing_f2_f2_in_A_f2_inequality_l193_193563

-- Define the function f2
def f2 (x : ℝ) : ℝ := 4 - 6 * (1 / 2) ^ x

-- The domain of f2 is [0,+∞)
def domain_f2 (x : ℝ) : Prop := x ≥ 0

-- The range of f2 is within [-2, 4)
def range_f2 (y : ℝ) : Prop := y ∈ Ico (-2 : ℝ) 4

-- f2 is increasing on [0,+∞)
theorem increasing_f2 : ∀ x y, domain_f2 x → domain_f2 y → x ≤ y → f2 x ≤ f2 y := sorry

-- f2 belongs to set A
theorem f2_in_A : ∀ x, domain_f2 x → ∃ y, range_f2 (f2 x) ∧ increasing_f2 := sorry

-- The inequality f(x) + f(x+2) < 2f(x+1) holds for f2
theorem f2_inequality : ∀ x, domain_f2 x → f2 x + f2 (x + 2) < 2 * f2 (x + 1) := sorry

end increasing_f2_f2_in_A_f2_inequality_l193_193563


namespace valid_numbers_count_l193_193633

def count_valid_numbers : ℕ :=
  sorry

theorem valid_numbers_count :
  count_valid_numbers = 7 :=
sorry

end valid_numbers_count_l193_193633


namespace course_selection_count_l193_193222

-- Definitions for the conditions
def num_PE_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_PE_courses + num_art_courses

-- The number of different course selection schemes
theorem course_selection_count : 
  (choose num_PE_courses 1) * (choose num_art_courses 1) + 
  (choose num_PE_courses 2) * (choose num_art_courses 1) + 
  (choose num_PE_courses 1) * (choose num_art_courses 2) = 64 := 
by sorry

end course_selection_count_l193_193222


namespace area_ratio_l193_193916

-- Define the problem conditions
def Square (s : ℝ) := s > 0
def Rectangle (longer shorter : ℝ) := longer = 1.2 * shorter ∧ shorter = 0.8 * shorter

-- Define a function to calculate the area of square
def area_square (s : ℝ) : ℝ := s * s

-- Define a function to calculate the area of rectangle
def area_rectangle (longer shorter : ℝ) : ℝ := longer * shorter

-- State the proof problem
theorem area_ratio (s : ℝ) (h_square : Square s) :
  let longer := 1.2 * s
  let shorter := 0.8 * s
  area_rectangle longer shorter / area_square s = 24 / 25 :=
by
  sorry

end area_ratio_l193_193916


namespace limit_of_sequence_l193_193325

noncomputable def sequence (n : ℕ) : ℝ := (2^(n+1) + 3^(n+1)) / (2^n + 3^n)

theorem limit_of_sequence : 
  tendsto sequence at_top (𝓝 3) :=
sorry

end limit_of_sequence_l193_193325


namespace remainder_sum_l193_193056

theorem remainder_sum:
  (∀ k, a = 60 * k + 53) →
  (∀ j, b = 45 * j + 22) →
  ∃ m, (a + b) % 30 = 15 :=
by
  intro ha hb
  use a, b
  sorry

end remainder_sum_l193_193056


namespace number_of_sandwiches_l193_193095

theorem number_of_sandwiches (breads meats cheeses : ℕ)
  (hb : breads = 5)
  (hm : meats = 7)
  (hc : cheeses = 5)
  (restricted1_breads : ℕ)
  (restricted2_cheeses : ℕ)
  (hr1 : restricted1_breads = breads)
  (hr2 : restricted2_cheeses = cheeses) :
  (breads * meats * cheeses) - restricted1_breads - restricted2_cheeses = 165 :=
by
  rw [hb, hm, hc, hr1, hr2]
  exact rfl

end number_of_sandwiches_l193_193095


namespace product_of_integers_is_correct_l193_193127

theorem product_of_integers_is_correct (a b c d e : ℤ)
  (h_pairwise_sums : Multiset {x : ℤ // ∃ i j, [i, j] ∈ ([a, b, c, d, e].pair_combinations) ∧ x = i + j } = {-1, 4, 6, 9, 10, 11, 15, 16, 20, 22}.noprob) :
  a * b * c * d * e = -4914 :=        
sorry

end product_of_integers_is_correct_l193_193127


namespace course_selection_schemes_l193_193207

theorem course_selection_schemes :
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  choose_2_courses + choose_3_courses = 64 :=
by
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  show choose_2_courses + choose_3_courses = 64 from sorry

end course_selection_schemes_l193_193207


namespace find_f_of_minus_one_l193_193618

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f : ℝ → ℝ :=
  λ x, if x > 0 then 3^x - 4 else - (3^(-x) - 4)

theorem find_f_of_minus_one :
  odd_function f ∧ (∀ x, x > 0 → f x = 3^x - 4) → f (-1) = 1 :=
by
  intros hff hfx
  -- Proof is intentionally left out
  sorry

end find_f_of_minus_one_l193_193618


namespace shortest_path_length_l193_193463

open Real

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

def inside_circle (p : ℝ × ℝ) (center : ℝ × ℝ) (r : ℝ) : Prop :=
  (distance p center) < r

theorem shortest_path_length :
  let P₁ := (0, 0) in
  let P₂ := (9, 12) in
  let C := (6, 8) in
  let r := 6 in
  (¬ inside_circle P₁ C r) ∧
  (¬ inside_circle P₂ C r) →
  distance P₁ C = 10 ∧
  distance P₂ C ≠ r ∧
  ∃ B C', 
    distance P₁ B = 8 ∧
    distance C' P₂ = 8 ∧
    ∃ θ ≈ 36.87, 
      let arc_length := θ / 360 * 2 * π * r in
      arc_length ≈ 14.72 →
  distance P₁ B + arc_length + distance C' P₂ = 30.72 := 
sorry

end shortest_path_length_l193_193463


namespace negation_example_l193_193941

theorem negation_example : (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x : ℝ, x^2 < 0) :=
by
  sorry

end negation_example_l193_193941


namespace polynomial_remainder_l193_193686

theorem polynomial_remainder (x : ℝ) :
  ∃ (Q : ℝ → ℝ) (a b : ℝ),
    (x^150 = (x^2 - 5*x + 6) * Q x + (a*x + b)) ∧
    (2 * a + b = 2^150) ∧
    (3 * a + b = 3^150) ∧ 
    (a = 3^150 - 2^150) ∧ 
    (b = 2^150 - 2 * 3^150 + 2 * 2^150) := sorry

end polynomial_remainder_l193_193686


namespace final_balloons_remaining_intact_l193_193293

def balloons_initial := 200
def balloons_blow_up_half_hour := (1 / 5 : Real) * balloons_initial
def balloons_remaining_after_half_hour := balloons_initial - balloons_blow_up_half_hour
def balloons_blow_up_next_hour := (30 / 100 : Real) * balloons_remaining_after_half_hour
def balloons_remaining_after_next_hour := balloons_remaining_after_half_hour - balloons_blow_up_next_hour
def balloons_double_durability := Real.floor((10 / 100 : Real) * balloons_remaining_after_next_hour)
def balloons_regular_remaining := balloons_remaining_after_next_hour - balloons_double_durability
def blow_up_remaining := 2 * balloons_blow_up_next_hour

theorem final_balloons_remaining_intact : 
  (balloons_regular_remaining <= blow_up_remaining → 
   balloons_double_durability = 11) :=
by 
  sorry

end final_balloons_remaining_intact_l193_193293


namespace sum_sequence_formula_l193_193391

def sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) : Prop :=
  ↑n > 0 →
  S n - S (n - 1) = a n ∧
  S n * S n - 2 * S n - a n * S n + 1 = 0

theorem sum_sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, sequence a S n) →
  ∀ n, S n = n / (n + 1) :=
by
  intro h
  sorry

end sum_sequence_formula_l193_193391


namespace extreme_value_f_g_gt_one_l193_193868

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.exp x * (a * x + x * Real.cos x + 1)

theorem extreme_value_f : f 0 = 0 :=
by
  sorry

theorem g_gt_one (a : ℝ) (h : a > -1) (x : ℝ) (hx : 0 < x) (hx1 : x < 1) : g x a > 1 :=
by
  sorry

end extreme_value_f_g_gt_one_l193_193868


namespace find_s_l193_193480

theorem find_s (n r s c d : ℝ) (h1 : c^2 - n * c + 3 = 0) (h2 : d^2 - n * d + 3 = 0) 
  (h3 : (c + 1/d)^2 - r * (c + 1/d) + s = 0) (h4 : (d + 1/c)^2 - r * (d + 1/c) + s = 0) 
  (h5 : c * d = 3) : s = 16 / 3 := 
by
  sorry

end find_s_l193_193480


namespace coefficient_of_term_l193_193091

noncomputable def coefficient (term : ℤ → ℚ) (n : ℤ) : ℚ :=
  term n

theorem coefficient_of_term : coefficient (λ n, if n = 2 then -1/2 else 0) 2 = -1/2 := 
by
  sorry

end coefficient_of_term_l193_193091


namespace Kristyna_number_l193_193472

theorem Kristyna_number (k n : ℕ) (h1 : k = 6 * n + 3) (h2 : 3 * n + 1 + 2 * n = 1681) : k = 2019 := 
by
  -- Proof goes here
  sorry

end Kristyna_number_l193_193472


namespace no_real_roots_of_P_l193_193643

noncomputable def P : ℕ → ℝ → ℝ
| 0     := λ x, 1
| (n+1) := λ x, x^(17*(n+1)) - P n x

theorem no_real_roots_of_P (n : ℕ) : ∀ x : ℝ, P n x ≠ 0 :=
  sorry

end no_real_roots_of_P_l193_193643


namespace value_diff_l193_193112

theorem value_diff (a b : ℕ) (h1 : a * b = 2 * (a + b) + 14) (h2 : b = 8) : b - a = 3 :=
by
  sorry

end value_diff_l193_193112


namespace correct_option_is_C_l193_193975

-- Definitions of the expressions given in the conditions
def optionA (a : ℝ) : ℝ := 3 * a^5 - a^5
def optionB (a : ℝ) : ℝ := a^2 + a^5
def optionC (a : ℝ) : ℝ := a^5 + a^5
def optionD (x y : ℝ) : ℝ := x^2 * y + x * y^2

-- The problem is to prove that optionC is correct and the others are not
theorem correct_option_is_C (a x y : ℝ) :
  (optionC a = 2 * a^5) ∧ 
  (optionA a ≠ 3) ∧ 
  (optionB a ≠ a^7) ∧ 
  (optionD x y ≠ 2 * (x ^ 3) * (y ^ 3)) :=
by
  sorry

end correct_option_is_C_l193_193975


namespace find_a_l193_193493

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem find_a (a : ℝ) (h : is_pure_imaginary ((a + complex.I) / (1 + complex.I))) : a = -1 :=
by
  sorry

end find_a_l193_193493


namespace heights_inequality_l193_193485

theorem heights_inequality (a b c h_a h_b h_c : ℝ)
  (ha : h_a = b * sin (C : ℝ))
  (hb : h_b = c * sin (A : ℝ))
  (hc : h_c = a * sin (B : ℝ))
  (S : (1/2) * a * b * sin (C : ℝ) = (1/2) * a * h_a) :
  (h_b^2 + h_c^2) / a^2 + (h_c^2 + h_a^2) / b^2 + (h_a^2 + h_b^2) / c^2 ≤ 9/2 :=
by
  sorry

end heights_inequality_l193_193485


namespace rationalize_denominator_l193_193520

theorem rationalize_denominator : (35 : ℝ) / Real.sqrt 15 = (7 / 3 : ℝ) * Real.sqrt 15 :=
by
  sorry

end rationalize_denominator_l193_193520


namespace total_course_selection_schemes_l193_193274

theorem total_course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose_two :=
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 1)
  let choose_three :=
    (Nat.choose PE_courses 2) * (Nat.choose Art_courses 1) +
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 2)
  total_courses = 8
  ∧ (choose_two + choose_three = 64) :=
by
  sorry

end total_course_selection_schemes_l193_193274


namespace eqn_of_C_l193_193822

-- Definitions 
def P := (x : ℝ, y : ℝ)  -- Define point P with coordinates x and y
def C := { p : P // p.fst^2 + (p.snd^2 / 4) = 1 }  -- Curve C as an ellipse

-- Conditions for Problem 1
def sum_dist_from_foci (P : P) : Prop := (dist P (-sqrt(3), 0) + dist P (sqrt(3), 0)) = 4

-- Proof Problem 1 Statement (Equation of C)
theorem eqn_of_C (P : P) (h : sum_dist_from_foci P) : P ∈ C := sorry

end eqn_of_C_l193_193822


namespace smaller_number_is_25_l193_193569

theorem smaller_number_is_25 (x y : ℕ) (h1 : x + y = 62) (h2 : y = x + 12) : x = 25 :=
by sorry

end smaller_number_is_25_l193_193569


namespace find_R_plus_S_l193_193791

theorem find_R_plus_S (d e R S : ℝ) 
  (h1 : d + 3 = 0)
  (h2 : 7 * d + 3 * e = 0)
  (h3 : R = 3 * d + e + 7)
  (h4 : S = 7 * e) :
  R + S = 54 :=
by
  sorry

end find_R_plus_S_l193_193791


namespace nine_pow_1000_mod_13_l193_193562

theorem nine_pow_1000_mod_13 :
  (9^1000) % 13 = 9 :=
by
  have h1 : 9^1 % 13 = 9 := by sorry
  have h2 : 9^2 % 13 = 3 := by sorry
  have h3 : 9^3 % 13 = 1 := by sorry
  have cycle : ∀ n, 9^(3 * n + 1) % 13 = 9 := by sorry
  exact (cycle 333)

end nine_pow_1000_mod_13_l193_193562


namespace monster_consumption_l193_193289

theorem monster_consumption (a b c : ℕ) (h1 : a = 121) (h2 : b = 2 * a) (h3 : c = 2 * b) : a + b + c = 847 :=
by
  -- hypothesis introduction
  have h4 : a = 121 := h1,
  have h5 : b = 242 := by rw [h2, h4]; norm_num,
  have h6 : c = 484 := by rw [h3, h5]; norm_num,
  -- proof
  rw [h4, h5, h6]; norm_num

end monster_consumption_l193_193289


namespace problem_inequality_l193_193726

open Real

theorem problem_inequality (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  1 / cos α ^ 2 + 1 / (sin α ^ 2 * sin β ^ 2 * cos β ^ 2) ≥ 9 := 
begin
  sorry
end

end problem_inequality_l193_193726


namespace no_real_roots_of_P_l193_193642

noncomputable def P : ℕ → ℝ → ℝ
| 0     := λ x, 1
| (n+1) := λ x, x^(17*(n+1)) - P n x

theorem no_real_roots_of_P (n : ℕ) : ∀ x : ℝ, P n x ≠ 0 :=
  sorry

end no_real_roots_of_P_l193_193642


namespace two_a_squared_plus_two_b_squared_l193_193785

theorem two_a_squared_plus_two_b_squared :
  ∀ (a b : ℝ), a + b = 5 → ab = 3 → 2 * a ^ 2 + 2 * b ^ 2 = 38 :=
begin
  intros a b h1 h2,
  sorry
end

end two_a_squared_plus_two_b_squared_l193_193785


namespace complex_midpoint_l193_193009

def midpoint_complex (z1 z2 : ℂ) : ℂ :=
  (z1 + z2) / 2

theorem complex_midpoint :
  midpoint_complex (6 + 5 * Complex.I) (-2 + 3 * Complex.I) = 2 + 4 * Complex.I :=
by
  sorry

end complex_midpoint_l193_193009


namespace inscribed_circle_radius_in_triangle_DEF_l193_193678

theorem inscribed_circle_radius_in_triangle_DEF 
  (DE DF EF : ℝ) 
  (hDE : DE = 8) 
  (hDF : DF = 10) 
  (hEF : EF = 12) : 
  let s := (DE + DF + EF) / 2 in
  s = 15 → 
  let K := real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
  K = 15 * real.sqrt 7 →
  let r := K / s in
  r = real.sqrt 7 :=
by
  -- Insert proof here
  sorry

end inscribed_circle_radius_in_triangle_DEF_l193_193678


namespace train_length_l193_193299

theorem train_length :
  let speed_kmph := 54
  let conversion_factor := 1000 / 3600
  let speed_mps := speed_kmph * conversion_factor
  let time_seconds := 3.9996800255979523
  let distance := speed_mps * time_seconds
  distance = 59.9952 :=
by
  let speed_kmph := 54
  let conversion_factor := 1000 / 3600
  let speed_mps := speed_kmph * conversion_factor
  let time_seconds := 3.9996800255979523
  let distance := speed_mps * time_seconds
  have h1 : speed_mps = 15 := by sorry -- (Conversion step, converted value should be pre-known)
  have h2 : distance = 15 * 3.9996800255979523 := by sorry -- (Calculation step)
  have h3: distance = 59.9952 := by sorry -- (Final result)
  exact h3

end train_length_l193_193299


namespace range_of_a_l193_193795

theorem range_of_a:
  (∀ x ∈ set.Icc (1 : ℝ) 2, |x^2 - a| + |x + a| = |x^2 + x|) →
  a ∈ set.Icc (-1 : ℝ) 1 := 
sorry

end range_of_a_l193_193795


namespace proof_f_2_log2_3_l193_193412

def f (x : ℝ) : ℝ :=
  if x < 4 then (1/2)^x * f (x + 1) else sorry -- Placeholder for the portion of the function undefined for x >= 4

theorem proof_f_2_log2_3 : f (2 + (real.log 3 / real.log 2)) = 1 / 24 :=
  sorry

end proof_f_2_log2_3_l193_193412


namespace area_triangle_from_roots_l193_193098

theorem area_triangle_from_roots (a b c : ℝ)
  (h : (Polynomial.X^3 - 6 * Polynomial.X^2 + 11 * Polynomial.X - (6 / 5)).roots = {a, b, c}) :
  Real.sqrt(72 / 5) = 6 * Real.sqrt(5) / 5 :=
by sorry

end area_triangle_from_roots_l193_193098


namespace find_fiftieth_term_l193_193540

variable {a1 a15 : ℤ}
variable {a50: ℤ}
variable {d: ℚ}

noncomputable def arithmetic_sequence (a1 a15 a50 : ℤ) (d: ℚ) :=
  ∀ (n : ℕ), a1 + (n - 1) * d

theorem find_fiftieth_term 
  (h1 : a1 = 7) 
  (h2 : a15 = 41)
  (h3 : d = (a15 - a1) / 14) 
  (h4 : a50 = a1 + 49 * d) :
  a50 = 126 :=
by
  sorry

end find_fiftieth_term_l193_193540


namespace BBB_div_by_9_l193_193104

open Nat

theorem BBB_div_by_9 (B : ℕ) (h1 : 4 * 10^4 + B * 10^3 + B * 10^2 + B * 10 + 2 ≡ 0 [MOD 9]) (h2 : B ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  B = 4 :=
by
  have mod9_eq : (4 * 10^4 + (B + B + B) * 10^2 + 2) ≡ (4 + B + B + B + 2) [MOD 9] := Nat.mod_eq_of_lt
  sorry

end BBB_div_by_9_l193_193104


namespace find_f_prime_one_l193_193387

noncomputable def f (f' : ℝ → ℝ) (x : ℝ) := x^2 + 2 * x * f' 1

theorem find_f_prime_one (f' : ℝ → ℝ) (hf : ∀ x, deriv (λ x, x^2 + 2 * x * f' 1) x = 2 * x + 2 * f' 1) : 
  f' 1 = -2 :=
by
  sorry

end find_f_prime_one_l193_193387


namespace Jasmine_payment_l193_193026

theorem Jasmine_payment :
  let coffee_price_per_pound := 2.50
  let milk_price_per_gallon := 3.50
  let coffee_pounds := 4
  let milk_gallons := 2
  let discount_rate := 0.10
  let tax_rate := 0.08
  let cost_of_coffee := coffee_pounds * coffee_price_per_pound
  let cost_of_milk := milk_gallons * milk_price_per_gallon
  let total_cost_before_discount := cost_of_coffee + cost_of_milk
  let discount := discount_rate * total_cost_before_discount
  let discounted_price := total_cost_before_discount - discount
  let taxes := tax_rate * discounted_price
  let final_amount := discounted_price + taxes
  (final_amount ≈ 16.52) :=
by
  sorry

end Jasmine_payment_l193_193026


namespace solve_system_l193_193904

theorem solve_system :
  ∃ x y z : ℝ, (x = 20 ∧ y = 22 ∧ z = 23 ∧ 
  (x^2 - 23*y - 25*z = -681) ∧ 
  (y^2 - 21*x - 21*z = -419) ∧ 
  (z^2 - 19*x - 21*y = -313)) :=
by
  use 20, 22, 23
  split
  . refl
  split
  . refl
  split
  . refl
  split
  . sorry
  split
  . sorry
  . sorry

end solve_system_l193_193904


namespace painted_area_correct_l193_193288

def cylinder_height := 15
def cylinder_radius := 5
def fraction_painted := 0.75

noncomputable def total_painted_area : ℝ :=
  let r := cylinder_radius
  let h := cylinder_height
  let lateral_area := 2 * Real.pi * r * h
  let end_area := 2 * Real.pi * (r^2)
  (fraction_painted * lateral_area) + end_area

theorem painted_area_correct : total_painted_area = 162.5 * Real.pi := sorry

end painted_area_correct_l193_193288


namespace no_infinite_prime_sequence_l193_193882

theorem no_infinite_prime_sequence (p : ℕ → ℕ)
  (h : ∀ k : ℕ, Nat.Prime (p k) ∧ p (k + 1) = 5 * p k + 4) :
  ¬ ∀ n : ℕ, Nat.Prime (p n) :=
by
  sorry

end no_infinite_prime_sequence_l193_193882


namespace number_of_boxes_to_fill_l193_193843

-- Define conditions
def daily_donuts := 100
def days_in_year := 365
def jeff_daily_consumption_percent := 0.04
def friend_chris_monthly := 15
def friend_sam_monthly := 20
def friend_emily_monthly := 25
def charity_donation_percent := 0.10
def neighbor_yearly_percent := 0.06
def donuts_per_box := 50
def months_in_year := 12

-- Define helper calculations based on conditions
def total_donuts_per_year := daily_donuts * days_in_year
def jeff_yearly_consumption := (jeff_daily_consumption_percent * daily_donuts) * days_in_year
def friends_yearly_consumption := (friend_chris_monthly + friend_sam_monthly + friend_emily_monthly) * months_in_year
def charity_yearly_donation := (charity_donation_percent * (daily_donuts * 30)) * months_in_year
def neighbor_yearly_consumption := neighbor_yearly_percent * total_donuts_per_year
def remaining_donuts := total_donuts_per_year - (jeff_yearly_consumption + friends_yearly_consumption + charity_yearly_donation + neighbor_yearly_consumption)

-- Statement to be proved
theorem number_of_boxes_to_fill : remaining_donuts / donuts_per_box = 570 :=
by sorry

end number_of_boxes_to_fill_l193_193843


namespace inequality_sum_lt_sqrt_n_l193_193036

theorem inequality_sum_lt_sqrt_n (n : ℕ) (x : Fin n → ℝ) :
  (∑ j in Finset.range n, x j / (1 + ∑ i in Finset.range (j+1), (x i)^2)) < Real.sqrt n :=
sorry

end inequality_sum_lt_sqrt_n_l193_193036


namespace sqrt_div_l193_193151

theorem sqrt_div (x: ℕ) (h1: Nat.sqrt 144 * Nat.sqrt 144 = 144) (h2: 144 = 12 * 12) (h3: 2 * x = 12) : x = 6 :=
sorry

end sqrt_div_l193_193151


namespace opposite_numbers_reciprocal_values_l193_193746

theorem opposite_numbers_reciprocal_values (a b m n : ℝ) (h₁ : a + b = 0) (h₂ : m * n = 1) : 5 * a + 5 * b - m * n = -1 :=
by sorry

end opposite_numbers_reciprocal_values_l193_193746


namespace father_three_times_marika_in_year_l193_193058

-- Define the given conditions as constants.
def marika_age_2004 : ℕ := 8
def father_age_2004 : ℕ := 32

-- Define the proof goal.
theorem father_three_times_marika_in_year :
  ∃ (x : ℕ), father_age_2004 + x = 3 * (marika_age_2004 + x) → 2004 + x = 2008 := 
by {
  sorry
}

end father_three_times_marika_in_year_l193_193058


namespace geckos_sold_in_two_years_l193_193456

theorem geckos_sold_in_two_years :
  let geckos_half_last_year := 46 in
  let geckos_second_half_last_year := geckos_half_last_year + (0.20 * geckos_half_last_year) in
  let geckos_first_half_two_years_ago := 3 * geckos_half_last_year in
  let geckos_second_half_two_years_ago := geckos_first_half_two_years_ago - (0.15 * geckos_first_half_two_years_ago) in
  let total_geckos_sold := geckos_half_last_year + geckos_second_half_last_year + geckos_first_half_two_years_ago + geckos_second_half_two_years_ago in
  total_geckos_sold = 356 :=
by {
  let geckos_half_last_year := 46;
  let geckos_second_half_last_year := geckos_half_last_year + (0.20 * geckos_half_last_year);
  let geckos_first_half_two_years_ago := 3 * geckos_half_last_year;
  let geckos_second_half_two_years_ago := geckos_first_half_two_years_ago - (0.15 * geckos_first_half_two_years_ago);
  let total_geckos_sold := geckos_half_last_year + geckos_second_half_last_year + geckos_first_half_two_years_ago + geckos_second_half_two_years_ago;
  have h := 46 + (46 + 0.20 * 46) + (3 * 46) - (0.15 * (3 * 46));
  have h2 := 46 + 55 + 138 + 117;
  suffices : total_geckos_sold = 356, by { 
  	sorry;
  };
  left; exact this;
  sorry
}

end geckos_sold_in_two_years_l193_193456


namespace median_score_interval_l193_193677

theorem median_score_interval 
  (students_55_59 : ℕ = 10) 
  (students_60_64 : ℕ = 15) 
  (students_65_69 : ℕ = 12) 
  (students_70_74 : ℕ = 17) 
  (students_75_79 : ℕ = 13) 
  (students_80_84 : ℕ = 15) 
  (students_85_89 : ℕ = 18) 
  (total_students : ℕ = 100) : 
  -- The theorem we want to prove
  70 ≤ calculate_median [55, 60, 65, 70, 75, 80, 85] [10, 15, 12, 17, 13, 15, 18] ∧ 
  calculate_median [55, 60, 65, 70, 75, 80, 85] [10, 15, 12, 17, 13, 15, 18] ≤ 74 := 
sorry

end median_score_interval_l193_193677


namespace find_extrema_sum_floor_diff_l193_193035

variables {n : ℕ} {x : Fin n → ℝ}

-- Conditions
def condition (n : ℕ) (x : Fin n → ℝ) : Prop :=
  n ≥ 2 ∧ (Perm (Finset.map ⟨floor, sorry⟩ (Finset.univ : Finset (Fin n))) (Finset.range n))

-- To be proved
theorem find_extrema_sum_floor_diff
  (hn : n ≥ 2)
  (hperm : Perm (Finset.map ⟨floor, sorry⟩ (Finset.univ : Finset (Fin n))) (Finset.range n)) : 
  ∃ k ∈ (Set.range ⟨λ i, ⟨i, sorry⟩⟩.to_finset), 
    (2 - n) ≤ k ∧ k ≤ (n - 1) ∧
    (∀ sum_value, sum_value = (∑ i in Finset.range (n - 1), ⌊x i.succ - x i⌋) → sum_value = k) :=
sorry

end find_extrema_sum_floor_diff_l193_193035


namespace compare_values_l193_193747

theorem compare_values (a b c : ℝ) (h₁ : a = Real.log 0.5 / Real.log 0.6) (h₂ : b = Real.log 0.5) (h₃ : c = 0.6 ^ 0.5) : a < b ∧ b < c :=
by
  sorry

end compare_values_l193_193747


namespace calculate_angle_sum_l193_193292

-- Defining the problem, conditions, and necessary assumptions
variables {P Q R S T U M : Point}
variables (PQ PR ST SU : LineSegment)
variables {angleQPR angleTUS : ℝ}
variables {angleRPM angleMPT : ℝ}

-- Conditions:
axiom h1 : IsIsoscelesTriangle P Q R
axiom h2 : IsIsoscelesTriangle S T U
axiom h3 : PQ = PR
axiom h4 : ST = SU
axiom h5 : angleQPR = 40
axiom h6 : angleTUS = 50
axiom h7 : IsMidpoint M T U
axiom h8 : Perpendicular (Line.through P M) (Line.through S T)

-- Goal:
theorem calculate_angle_sum
  (h9 : InternalAngle P Q R angleQPR)
  (h10 : InternalAngle T U S angleTUS)
  (h11 : InternalAngle R P M angleRPM)
  (h12 : InternalAngle M P T angleMPT) :
  angleRPM + angleMPT = 95 :=
sorry

end calculate_angle_sum_l193_193292


namespace relationship_a_b_monotonicity_g_l193_193415

-- Definitions from the problem
def f (x : ℝ) := Real.log x
def g (x : ℝ) (a b : ℝ) := (f x) + a * x^2 + b * x

-- Question 1: Relationship between a and b
theorem relationship_a_b (a b : ℝ) 
  (h : ∀ x, g x a b = Real.log x + a * x^2 + b * x) 
  (tangent_parallel : g' 1 a b = 0) :
  b = -2 * a - 1 := sorry

-- Question 2: Monotonicity of g(x) for a ≥ 0
theorem monotonicity_g (a : ℝ) (h_ge_0 : a ≥ 0) : 
  (a = 0 → ∀ x : ℝ, 0 < x ∧ x < 1 → g' x a (-2*a-1) > 0 ∧ x > 1 → g' x a (-2*a-1) < 0) ∧
  (0 < a ∧ a < 1/2 → ∀ x : ℝ, if 0 < x ∧ x < 1 ∨ x > 1/2/a then g' x a (-2*a-1) > 0 else g' x a (-2*a-1) < 0) ∧
  (a = 1/2 → ∀ x : ℝ, 0 < x → g' x a (-2*a-1) ≥ 0) ∧
  (a > 1/2 → ∀ x : ℝ, if 0 < x ∧ x < 1/2/a ∨ x > 1 then g' x a (-2*a-1) > 0 else g' x a (-2*a-1) < 0) :=
sorry

end relationship_a_b_monotonicity_g_l193_193415


namespace max_value_min_expression_l193_193139

def f (x y : ℝ) : ℝ :=
  x^3 + (y-4)*x^2 + (y^2-4*y+4)*x + (y^3-4*y^2+4*y)

theorem max_value_min_expression (a b c : ℝ) (h₁: a ≠ b) (h₂: b ≠ c) (h₃: c ≠ a)
  (hab : f a b = f b c) (hbc : f b c = f c a) :
  (max (min (a^4 - 4*a^3 + 4*a^2) (min (b^4 - 4*b^3 + 4*b^2) (c^4 - 4*c^3 + 4*c^2))) 1) = 1 :=
sorry

end max_value_min_expression_l193_193139


namespace valid_numbers_count_l193_193634

def count_valid_numbers : ℕ :=
  sorry

theorem valid_numbers_count :
  count_valid_numbers = 7 :=
sorry

end valid_numbers_count_l193_193634


namespace length_of_side_a_correct_area_of_triangle_correct_sin_2B_correct_l193_193802

noncomputable def length_of_side_a (b c : ℝ) (A : ℝ) : ℝ := 
  real.sqrt (b^2 + c^2 - 2 * b * c * real.cos A)

noncomputable def area_of_triangle (b c : ℝ) (A : ℝ) : ℝ :=
  (1 / 2) * b * c * real.sin A

noncomputable def sin_2B (b c A : ℝ) : ℝ :=
  let a := real.sqrt (b^2 + c^2 - 2 * b * c * real.cos A) in
  let sin_B := b * real.sin A / a in
  let cos_B := real.sqrt (1 - sin_B^2) in
  2 * sin_B * cos_B

variable (b c : ℝ) (A : ℝ)
#check b = 4
#check c = 5
#check A = real.pi / 3

theorem length_of_side_a_correct (b : ℝ) (c : ℝ) (A : ℝ) (h1 : b = 4) (h2 : c = 5) (h3 : A = real.pi / 3): 
  length_of_side_a b c A = real.sqrt 21 := 
by 
  simp [length_of_side_a, h1, h2, h3]
  sorry

theorem area_of_triangle_correct (b : ℝ) (c : ℝ) (A : ℝ) (h1 : b = 4) (h2 : c = 5) (h3 : A = real.pi / 3): 
  area_of_triangle b c A = 5 * real.sqrt 3 := 
by 
  simp [area_of_triangle, h1, h2, h3]
  sorry

theorem sin_2B_correct (b : ℝ) (c : ℝ) (A : ℝ) (h1 : b = 4) (h2 : c = 5) (h3 : A = real.pi / 3): 
  sin_2B b c A = 4 * real.sqrt 3 / 7 := 
by 
  simp [sin_2B, h1, h2, h3]
  sorry

end length_of_side_a_correct_area_of_triangle_correct_sin_2B_correct_l193_193802


namespace trapezoidal_field_base_count_l193_193089

theorem trapezoidal_field_base_count
  (A : ℕ) (h : ℕ) (b1 b2 : ℕ)
  (hdiv8 : ∃ m n : ℕ, b1 = 8 * m ∧ b2 = 8 * n)
  (area_eq : A = (h * (b1 + b2)) / 2)
  (A_val : A = 1400)
  (h_val : h = 50) :
  (∃ pair1 pair2 pair3, (pair1 + pair2 + pair3 = (b1 + b2))) :=
by
  sorry

end trapezoidal_field_base_count_l193_193089


namespace cos_B_plus_C_value_of_c_l193_193801

variable (A B C a b c S : ℝ)

-- Conditions
def angle_conditions : Prop :=
  a = 2 * b ∧
  ∃ k, k * k = 15 ∧ sin A + sin B = 2 * sin C ∧
  (1/2) * b * c * sin A = S

-- Prove cos(B + C) = 1/4 under the given conditions
theorem cos_B_plus_C : angle_conditions A B C a b c (sqrt 15 / 4) → 
  cos (B + C) = 1/4 :=
sorry

-- Prove c = 4 * sqrt 2 given S = sqrt(15)^4^(1/8) / 3
theorem value_of_c : angle_conditions A B C a b c (sqrt[8] (15) / 3) → 
  c = 4 * sqrt 2 :=
sorry

end cos_B_plus_C_value_of_c_l193_193801


namespace monotonic_increasing_interval_l193_193940

noncomputable def f : ℝ → ℝ := λ x, 3 * Real.sin ( (2 * Real.pi / 3) - 2 * x)

theorem monotonic_increasing_interval : 
  ∃ a b, a = 7 * Real.pi / 12 ∧ b = 13 * Real.pi / 12 ∧ ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y :=
sorry

end monotonic_increasing_interval_l193_193940


namespace correct_option_is_C_l193_193974

-- Definitions of the expressions given in the conditions
def optionA (a : ℝ) : ℝ := 3 * a^5 - a^5
def optionB (a : ℝ) : ℝ := a^2 + a^5
def optionC (a : ℝ) : ℝ := a^5 + a^5
def optionD (x y : ℝ) : ℝ := x^2 * y + x * y^2

-- The problem is to prove that optionC is correct and the others are not
theorem correct_option_is_C (a x y : ℝ) :
  (optionC a = 2 * a^5) ∧ 
  (optionA a ≠ 3) ∧ 
  (optionB a ≠ a^7) ∧ 
  (optionD x y ≠ 2 * (x ^ 3) * (y ^ 3)) :=
by
  sorry

end correct_option_is_C_l193_193974


namespace slope_of_line_l193_193367

theorem slope_of_line : 
  let A := Real.sin (Real.pi / 6)
  let B := Real.cos (5 * Real.pi / 6)
  (- A / B) = Real.sqrt 3 / 3 :=
by
  sorry

end slope_of_line_l193_193367


namespace course_selection_schemes_count_l193_193238

-- Definitions based on the conditions
def num_physical_education_courses : ℕ := 4
def num_art_courses : ℕ := 4

-- Required to choose 2 or 3 courses, with at least one from each category
def valid_selection_cases : list (ℕ × ℕ) := [(1, 1), (1, 2), (2, 1)]

-- Calculate the number of ways to choose k courses from n courses using combination formula
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Calculate the total number of different course selection schemes
def total_course_selection_schemes : ℕ :=
  let case1 := C num_physical_education_courses 1 * C num_art_courses 1
  let case2a := C num_physical_education_courses 2 * C num_art_courses 1
  let case2b := C num_physical_education_courses 1 * C num_art_courses 2
  case1 + case2a + case2b

-- The assertion that the total number of different course selection schemes is 64
theorem course_selection_schemes_count : total_course_selection_schemes = 64 := by
  -- sorry placeholder for proof
  sorry

end course_selection_schemes_count_l193_193238


namespace min_ab_l193_193784

theorem min_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : Real.log2 (a + 4 * b) = Real.log2 a + Real.log2 b) : a * b ≥ 16 :=
sorry

end min_ab_l193_193784


namespace find_ellipse_eq_find_max_area_of_triangle_l193_193737

-- Definitions from conditions
def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop := (x^2) / (a^2) + (y^2) / (b^2) = 1
def eccentricity (a c : ℝ) : ℝ := c / a
def point_on_ellipse (a b : ℝ) (x y : ℝ) := ellipse_eq a b x y
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
def origin : ℝ × ℝ := (0, 0)
def distance (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
def line_eq (m n : ℝ) (y : ℝ) : ℝ := m * y + n

-- Proof problems
theorem find_ellipse_eq (h1 : ellipse_eq a b (sqrt 3) (1/2)) (h2 : eccentricity a (sqrt a^2 - b^2) = sqrt 3 / 2) (ha : a > b > 0) : ellipse_eq 2 1 x y :=
sorry

theorem find_max_area_of_triangle (h1 : ellipse_eq 2 1 x y) (h2 : distance origin (midpoint P Q) = 1) : ∃ a_max : ℝ, a_max = 1 :=
sorry

end find_ellipse_eq_find_max_area_of_triangle_l193_193737


namespace course_selection_count_l193_193219

-- Definitions for the conditions
def num_PE_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_PE_courses + num_art_courses

-- The number of different course selection schemes
theorem course_selection_count : 
  (choose num_PE_courses 1) * (choose num_art_courses 1) + 
  (choose num_PE_courses 2) * (choose num_art_courses 1) + 
  (choose num_PE_courses 1) * (choose num_art_courses 2) = 64 := 
by sorry

end course_selection_count_l193_193219


namespace countable_independent_bernoulli_l193_193341

/-- 
On the probability space \(( (0,1], \mathscr{B}((0,1]), \mathrm{P} )\), 
it is impossible to define more than a countable number of independent 
Bernoulli random variables \(\left\{\xi_{t}\right\}_{t \in T}\) taking two values 0 and 1 
with non-zero probability.
-/
theorem countable_independent_bernoulli 
  (P : MeasureTheory.ProbabilityMeasure (set.Ioc 0 1))
  (ξ : ℕ → MeasureTheory.MeasurableSpace (set.Ioc 0 1) → bool)
  (ξ_indep : ∀ m n, m ≠ n → MeasureTheory.Indep (ξ m) (ξ n))
  (ξ_bernoulli : ∀ n, MeasureTheory.ProbabilityMeasure (ξ n = 0) = 1 / 2 
                     ∧ MeasureTheory.ProbabilityMeasure (ξ n = 1) = 1 / 2) : 
  ∃ (T : set ℕ), T.countable ∧ ∀ t ∈ T, MeasureTheory.ProbabilityMeasure (ξ t = 0) > 0 ∧ MeasureTheory.ProbabilityMeasure (ξ t = 1) > 0 :=
sorry

end countable_independent_bernoulli_l193_193341


namespace logarithm_comparison_l193_193675

noncomputable def f (x : ℝ) : ℝ := (Real.log (x + 1)) / (Real.log x)

theorem logarithm_comparison 
  (h1 : ∀ x > 1, f x = (Real.log (x + 1)) / (Real.log x))
  (h2 : ∀ x > 1, deriv f x < 0)
  (h3 : ∀ m n > 1, m > n → (Real.log (m + 1) / Real.log (m)) < (Real.log (n + 1) / Real.log (n)) → (Real.log (m + 1) / Real.log (n + 1)) < (Real.log (m) / Real.log (n))) :
  Real.log 626 / Real.log 17 < Real.log 5 / Real.log 2 ∧ Real.log 5 / Real.log 2 < 5 / 2 := sorry

end logarithm_comparison_l193_193675


namespace wholesome_bakery_loaves_on_wednesday_l193_193088

theorem wholesome_bakery_loaves_on_wednesday :
  ∀ (L_wed L_thu L_fri L_sat L_sun L_mon : ℕ),
    L_thu = 7 →
    L_fri = 10 →
    L_sat = 14 →
    L_sun = 19 →
    L_mon = 25 →
    L_thu - L_wed = 2 →
    L_wed = 5 :=
by intros L_wed L_thu L_fri L_sat L_sun L_mon;
   intros H_thu H_fri H_sat H_sun H_mon H_diff;
   sorry

end wholesome_bakery_loaves_on_wednesday_l193_193088


namespace total_course_selection_schemes_l193_193229

theorem total_course_selection_schemes (
  pe_courses art_courses : Finset ℕ
) : 
  pe_courses.card = 4 →
  art_courses.card = 4 →
  let total_schemes := 
    (pe_courses.card.choose 1 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 2 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 1 * art_courses.card.choose 2) in
  total_schemes = 64 := 
by
  intros h1 h2
  let case1 := pe_courses.card.choose 1 * art_courses.card.choose 1 -- 4 * 4
  let case2_1 := pe_courses.card.choose 2 * art_courses.card.choose 1 -- 6 * 4
  let case2_2 := pe_courses.card.choose 1 * art_courses.card.choose 2 -- 4 * 6
  let case2 := case2_1 + case2_2 -- 24 + 24
  let total_schemes := case1 + case2 -- 16 + 48
  have hcard := by
    simp [Nat.choose]
    exact h1
  have hcase1 : case1 = 16 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_1 : case2_1 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_2 : case2_2 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2 : case2 = 48 := by {
    rw [hcase2_1, hcase2_2]
    norm_num
  }
  have htotal_schemes : total_schemes = 64 := by {
    rw [hcase1, hcase2]
    norm_num
  }
  exact htotal_schemes

end total_course_selection_schemes_l193_193229


namespace round_to_nearest_tenth_l193_193523

theorem round_to_nearest_tenth : round_to_nearest 0.1 45.24567 = 45.2 :=
by
  sorry

end round_to_nearest_tenth_l193_193523


namespace john_plays_periods_l193_193030

theorem john_plays_periods
  (PointsPer4Minutes : ℕ := 7)
  (PeriodDurationMinutes : ℕ := 12)
  (TotalPoints : ℕ := 42) :
  (TotalPoints / PointsPer4Minutes) / (PeriodDurationMinutes / 4) = 2 := by
  sorry

end john_plays_periods_l193_193030


namespace contest_paths_correct_l193_193997

noncomputable def count_contest_paths : Nat := sorry

theorem contest_paths_correct : count_contest_paths = 127 := sorry

end contest_paths_correct_l193_193997


namespace find_p_correct_l193_193942

noncomputable def find_p (a b p : ℝ) : Prop :=
  (a ≠ 0 ∧ b ≠ 0) ∧
  (a^2 - 5 * p * a + 2 * p^3 = 0) ∧
  (b = a^2 / 4) ∧
  (a + b = 5 * p) ∧
  (a * b = 2 * p^3) ∧
  (exists (a : ℝ), a = 2 * p)

theorem find_p_correct (p : ℝ) : find_p 3 := 
  sorry

end find_p_correct_l193_193942


namespace union_A_B_complement_U_A_num_subsets_intersection_A_B_l193_193424

open Set

def U : Set ℕ := {4, 5, 6, 7, 8, 9, 10, 11, 12}
def A : Set ℕ := {6, 8, 10, 12}
def B : Set ℕ := {1, 6, 8}

theorem union_A_B : A ∪ B = {1, 6, 8, 10, 12} := by
  sorry

theorem complement_U_A : U \ A = {4, 5, 7, 9, 11} := by
  sorry

theorem num_subsets_intersection_A_B : (A ∩ B).powerset.card = 4 := by
  sorry

end union_A_B_complement_U_A_num_subsets_intersection_A_B_l193_193424


namespace calculate_expression_l193_193668

theorem calculate_expression : (|-2| : ℝ) + sqrt 2 * real.tan (real.pi / 4) - sqrt 8 - (2023 - real.pi)^0 = 1 - sqrt 2 :=
by sorry

end calculate_expression_l193_193668


namespace limit_identity_l193_193413

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log (3 * x) + 8 * x

theorem limit_identity : 
  tendsto (fun h : ℝ => (f (1 - 2 * h) - f 1) / h) (𝓝 0) (𝓝 (-20)) :=
by 
  sorry

end limit_identity_l193_193413


namespace unknown_ages_sum_inconsistency_l193_193811

theorem unknown_ages_sum_inconsistency 
  (class_size : ℕ)
  (class_average_age : ℕ)
  (group1_size : ℕ) 
  (group1_average_age : ℕ)
  (group2_size : ℕ) 
  (group2_average_age : ℕ)
  (group3_size : ℕ) 
  (group3_average_age : ℕ) 
  (unknown_students_count : ℕ)
  (sum_of_unknown_ages : ℕ)
  (total_class_age := class_size * class_average_age)
  (total_group1_age := group1_size * group1_average_age)
  (total_group2_age := group2_size * group2_average_age)
  (total_group3_age := group3_size * group3_average_age)
  (known_total_age := total_group1_age + total_group2_age + total_group3_age)
  (calculated_unknown_total_age := total_class_age - known_total_age)
  (discrepancy := calculated_unknown_total_age ≠ sum_of_unknown_ages) 
  : discrepancy :=
by
  -- Definitions and conditions
  have class_size_def : class_size = 25 := rfl
  have class_average_age_def : class_average_age = 18 := rfl
  have group1_size_def : group1_size = 8 := rfl
  have group1_average_age_def : group1_average_age = 16 := rfl
  have group2_size_def : group2_size = 10 := rfl
  have group2_average_age_def : group2_average_age = 20 := rfl
  have group3_size_def : group3_size = 5 := rfl
  have group3_average_age_def : group3_average_age = 17 := rfl
  have unknown_students_count_def : unknown_students_count = 2 := rfl

  have sum_of_unknown_ages_def : sum_of_unknown_ages = 35 := rfl
  
  -- We construct the total known age
  have total_known_age :
    known_total_age = 128 + 200 + 85 :=
      by
        simp [known_total_age, total_group1_age, total_group2_age, total_group3_age, group1_size_def, group1_average_age_def, group2_size_def, group2_average_age_def, group3_size_def, group3_average_age_def]

  -- We simplify the total class age
  have total_class_age :
    total_class_age = 450 :=
      by
        simp [total_class_age, class_size_def, class_average_age_def]

  -- We calculate total ages discrepancy
  have calculated_total_unknown_age :
    calculated_unknown_total_age = 450 - 413 :=
      by
        simp [calculated_unknown_total_age, total_class_age, known_total_age]

  have sum_discrepancy :
    37 ≠ 35 :=
      by norm_num

  -- Thus we show the discrepancy
  exact sum_discrepancy

end unknown_ages_sum_inconsistency_l193_193811


namespace car_acceleration_at_2_seconds_l193_193561

theorem car_acceleration_at_2_seconds :
  (∀ t : ℝ, s t = 2 * t^3 - 5 * t^2 + 2) →
  ∃ a : ℝ, (∀ t : ℝ, s' t = 6 * t^2 - 10 * t) ∧ s' 2 = a ∧ a = 4 :=
  by
    sorry

end car_acceleration_at_2_seconds_l193_193561


namespace area_ratio_rect_sq_l193_193927

variable (s : ℝ)

def side_len_sq (S : ℝ) : Prop := s = S
def longer_side_rect (R : ℝ) : Prop := R = 1.2 * s
def shorter_side_rect (R : ℝ) : Prop := R = 0.8 * s
def area_sq (S : ℝ) : ℝ := S * S
def area_rect (R_long R_short : ℝ) : ℝ := R_long * R_short
def ratio_area (areaR areaS : ℝ) : ℝ := areaR / areaS

theorem area_ratio_rect_sq (s S R_long R_short : ℝ) (h1 : side_len_sq s S) (h2 : longer_side_rect s R_long) (h3 : shorter_side_rect s R_short) :
  ratio_area (area_rect R_long R_short) (area_sq S) = 24/25 :=
by
  sorry

end area_ratio_rect_sq_l193_193927


namespace samantha_routes_l193_193889

-- Define the positions relative to the grid
structure Position where
  x : Int
  y : Int

-- Define the initial conditions and path constraints
def house : Position := ⟨-3, -2⟩
def sw_corner_of_park : Position := ⟨0, 0⟩
def ne_corner_of_park : Position := ⟨8, 5⟩
def school : Position := ⟨11, 8⟩

-- Define the combinatorial function for calculating number of ways
def binom (n k : Nat) : Nat := Nat.choose n k

-- Route segments based on the constraints
def ways_house_to_sw_corner : Nat := binom 5 2
def ways_through_park : Nat := 1
def ways_ne_corner_to_school : Nat := binom 6 3

-- Total number of routes
def total_routes : Nat := ways_house_to_sw_corner * ways_through_park * ways_ne_corner_to_school

-- The statement to be proven
theorem samantha_routes : total_routes = 200 := by
  sorry

end samantha_routes_l193_193889


namespace total_course_selection_schemes_l193_193227

theorem total_course_selection_schemes (
  pe_courses art_courses : Finset ℕ
) : 
  pe_courses.card = 4 →
  art_courses.card = 4 →
  let total_schemes := 
    (pe_courses.card.choose 1 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 2 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 1 * art_courses.card.choose 2) in
  total_schemes = 64 := 
by
  intros h1 h2
  let case1 := pe_courses.card.choose 1 * art_courses.card.choose 1 -- 4 * 4
  let case2_1 := pe_courses.card.choose 2 * art_courses.card.choose 1 -- 6 * 4
  let case2_2 := pe_courses.card.choose 1 * art_courses.card.choose 2 -- 4 * 6
  let case2 := case2_1 + case2_2 -- 24 + 24
  let total_schemes := case1 + case2 -- 16 + 48
  have hcard := by
    simp [Nat.choose]
    exact h1
  have hcase1 : case1 = 16 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_1 : case2_1 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_2 : case2_2 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2 : case2 = 48 := by {
    rw [hcase2_1, hcase2_2]
    norm_num
  }
  have htotal_schemes : total_schemes = 64 := by {
    rw [hcase1, hcase2]
    norm_num
  }
  exact htotal_schemes

end total_course_selection_schemes_l193_193227


namespace unique_positive_solution_cos_arcsin_tan_arccos_l193_193365

theorem unique_positive_solution_cos_arcsin_tan_arccos (x : ℝ) (h : 0 < x ∧ x ≤ 1) :
  cos (arcsin (tan (arccos x))) = x ↔ x = 1 :=
by
  sorry

end unique_positive_solution_cos_arcsin_tan_arccos_l193_193365


namespace unique_positive_solution_cos_arcsin_tan_arccos_l193_193364

theorem unique_positive_solution_cos_arcsin_tan_arccos (x : ℝ) (h : 0 < x ∧ x ≤ 1) :
  cos (arcsin (tan (arccos x))) = x ↔ x = 1 :=
by
  sorry

end unique_positive_solution_cos_arcsin_tan_arccos_l193_193364


namespace sum_even_deg_coeff_l193_193567

theorem sum_even_deg_coeff (x : ℕ) : 
  (3 - 2*x)^3 * (2*x + 1)^4 = (3 - 2*x)^3 * (2*x + 1)^4 →
  (∀ (x : ℕ), (3 - 2*x)^3 * (2*1 + 1)^4 =  81 ∧ 
  (3 - 2*(-1))^3 * (2*(-1) + 1)^4 = 125 → 
  (81 + 125) / 2 = 103) :=
by
  sorry

end sum_even_deg_coeff_l193_193567


namespace probability_three_two_digit_numbers_l193_193621

noncomputable def probability_exactly_three_two_digit (n k : ℕ) (p q : ℚ) : ℚ :=
  (nat.choose n k : ℚ) * (p ^ k) * (q ^ (n - k))

theorem probability_three_two_digit_numbers :
  probability_exactly_three_two_digit 6 3 (1 / 4) (3 / 4) = 135 / 1024 :=
by
  trivial -- sorry to skip the proof step

end probability_three_two_digit_numbers_l193_193621


namespace total_selection_schemes_l193_193278

-- Define the given conditions
def num_phys_ed_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_phys_ed_courses + num_art_courses
def valid_course_combos : finset ℕ := {2, 3}

-- Define the number of selection schemes with the constraints
def selection_schemes : nat :=
  (num_phys_ed_courses.choose 1 * num_art_courses.choose 1) + 
  (num_phys_ed_courses.choose 2 * num_art_courses.choose 1 + 
   num_phys_ed_courses.choose 1 * num_art_courses.choose 2)

-- State the theorem to be proved
theorem total_selection_schemes : selection_schemes = 64 := by
  sorry

end total_selection_schemes_l193_193278


namespace combined_sum_is_115_over_3_l193_193862

def geometric_series_sum (a : ℚ) (r : ℚ) : ℚ :=
  if h : abs r < 1 then a / (1 - r) else 0

def arithmetic_series_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def combined_series_sum : ℚ :=
  let geo_sum := geometric_series_sum 5 (-1/2)
  let arith_sum := arithmetic_series_sum 3 2 5
  geo_sum + arith_sum

theorem combined_sum_is_115_over_3 : combined_series_sum = 115 / 3 := 
  sorry

end combined_sum_is_115_over_3_l193_193862


namespace jed_speeding_l193_193807

-- Define the constants used in the conditions
def F := 16
def T := 256
def S := 50

theorem jed_speeding : (T / F) + S = 66 := 
by sorry

end jed_speeding_l193_193807


namespace total_snowfall_l193_193806

variable (morning_snowfall : ℝ) (afternoon_snowfall : ℝ)

theorem total_snowfall {morning_snowfall afternoon_snowfall : ℝ} (h_morning : morning_snowfall = 0.12) (h_afternoon : afternoon_snowfall = 0.5) :
  morning_snowfall + afternoon_snowfall = 0.62 :=
sorry

end total_snowfall_l193_193806


namespace total_course_selection_schemes_l193_193185

theorem total_course_selection_schemes :
  let n_physical_education := 4
  let n_art := 4
  let total_courses := n_physical_education + n_art in
  let choose2_courses := (Nat.choose n_physical_education 1) * (Nat.choose n_art 1)
  let choose3_courses := (Nat.choose n_physical_education 2 * Nat.choose n_art 1) + (Nat.choose n_physical_education 1 * Nat.choose n_art 2) in
  total_courses = n_physical_education + n_art →
  choose2_courses + choose3_courses = 64 :=
by
  intros n_physical_education n_art total_courses choose2_courses choose3_courses h
  have h_choose2_courses: choose2_courses = 16 := by
    simp [n_physical_education, n_art, Nat.choose]
  have h_choose3_courses: choose3_courses = 48 := by
    simp [n_physical_education, n_art, Nat.choose]
  rw [h_choose2_courses, h_choose3_courses]
  exact Nat.add_eq_right.2 rfl

end total_course_selection_schemes_l193_193185


namespace cone_surface_area_l193_193751

open Real

/--
Given a cone for which both the front view and side view are equilateral triangles with side length 4, 
prove that the surface area of the cone is 12π.
-/
theorem cone_surface_area :
    ∃ (cone : Cone) (h₁ : cone.front_view = EquilateralTriangle 4) (h₂ : cone.side_view = EquilateralTriangle 4),
      surface_area cone = 12 * π :=
sorry

end cone_surface_area_l193_193751


namespace eval_neg_i_3_minus_5i_l193_193174

-- The imaginary unit i
def i : ℂ := complex.I

-- Statement of the problem as a theorem
theorem eval_neg_i_3_minus_5i : -i * (3 - 5 * i) = -5 - 3 * i :=
by
  -- We will replace this part with actual proof later
  sorry

end eval_neg_i_3_minus_5i_l193_193174


namespace contractor_initial_people_l193_193286

theorem contractor_initial_people (P : ℕ) (days_total days_done : ℕ) 
  (percent_done : ℚ) (additional_people : ℕ) (T : ℕ) :
  days_total = 50 →
  days_done = 25 →
  percent_done = 0.4 →
  additional_people = 90 →
  T = P + additional_people →
  (P : ℚ) * 62.5 = (T : ℚ) * 50 →
  P = 360 :=
by
  intros h_days_total h_days_done h_percent_done h_additional_people h_T h_eq
  sorry

end contractor_initial_people_l193_193286


namespace prism_height_l193_193074

theorem prism_height (AB AC vol : ℝ) (h1 : AB = sqrt 2) (h2 : AC = sqrt 2) (h3 : vol = 3.0000000000000004) : 
  let base_area := (1/2) * AB * AC in
  vol = base_area * (vol / base_area) := by
  sorry

end prism_height_l193_193074


namespace reciprocal_of_minus_one_is_minus_one_l193_193560

theorem reciprocal_of_minus_one_is_minus_one : ∃ x : ℝ, (-1) * x = 1 ∧ x = -1 :=
by
  use -1
  split
  case left =>
    show -1 * -1 = 1
    sorry
  case right =>
    show -1 = -1
    sorry

end reciprocal_of_minus_one_is_minus_one_l193_193560


namespace greatest_power_of_two_factor_l193_193142

theorem greatest_power_of_two_factor (a b c d : ℕ) (h1 : a = 10) (h2 : b = 1006) (h3 : c = 6) (h4 : d = 503) :
  ∃ k : ℕ, 2^k ∣ (a^b - c^d) ∧ ∀ j : ℕ, 2^j ∣ (a^b - c^d) → j ≤ 503 :=
sorry

end greatest_power_of_two_factor_l193_193142


namespace course_selection_schemes_l193_193205

theorem course_selection_schemes :
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  choose_2_courses + choose_3_courses = 64 :=
by
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  show choose_2_courses + choose_3_courses = 64 from sorry

end course_selection_schemes_l193_193205


namespace area_ratio_l193_193915

-- Define the problem conditions
def Square (s : ℝ) := s > 0
def Rectangle (longer shorter : ℝ) := longer = 1.2 * shorter ∧ shorter = 0.8 * shorter

-- Define a function to calculate the area of square
def area_square (s : ℝ) : ℝ := s * s

-- Define a function to calculate the area of rectangle
def area_rectangle (longer shorter : ℝ) : ℝ := longer * shorter

-- State the proof problem
theorem area_ratio (s : ℝ) (h_square : Square s) :
  let longer := 1.2 * s
  let shorter := 0.8 * s
  area_rectangle longer shorter / area_square s = 24 / 25 :=
by
  sorry

end area_ratio_l193_193915


namespace A_intersection_B_eq_intersection_set_l193_193426

def A : Set ℝ := {x : ℝ | x * (x - 2) < 0}
def B : Set ℝ := {x : ℝ | x > 1}
def intersection_set := {x : ℝ | 1 < x ∧ x < 2}

theorem A_intersection_B_eq_intersection_set : A ∩ B = intersection_set := by
  sorry

end A_intersection_B_eq_intersection_set_l193_193426


namespace max_distance_of_a_and_c_l193_193476

noncomputable def M_max 
    (a b c : EuclideanSpace ℝ (Fin 2)) 
    (h1 : ∥a∥ = 2) 
    (h2 : ∥b∥ = 2) 
    (h3 : InnerProductSpace.inner a b = 2) 
    (h4 : InnerProductSpace.inner c (a + 2 • b - 2 • c) = 2) 
    : Real :=
  ∥a - c∥ 

theorem max_distance_of_a_and_c 
    (a b c : EuclideanSpace ℝ (Fin 2)) 
    (h1 : ∥a∥ = 2) 
    (h2 : ∥b∥ = 2) 
    (h3 : InnerProductSpace.inner a b = 2) 
    (h4 : InnerProductSpace.inner c (a + 2 • b - 2 • c) = 2) 
    : M_max a b c h1 h2 h3 h4 = (sqrt 3 + sqrt 7) / 2 :=
  sorry

end max_distance_of_a_and_c_l193_193476


namespace line_does_not_pass_through_fourth_quadrant_l193_193783

theorem line_does_not_pass_through_fourth_quadrant
  (A B C : ℝ) (hAB : A * B < 0) (hBC : B * C < 0) :
  ¬ ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ A * x + B * y + C = 0 :=
by
  sorry

end line_does_not_pass_through_fourth_quadrant_l193_193783


namespace find_a3_l193_193730

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

theorem find_a3 (a : ℕ → ℝ) (r : ℝ)
  (h1 : geometric_sequence a r)
  (h2 : a 0 * a 1 * a 2 * a 3 * a 4 = 32):
  a 2 = 2 :=
sorry

end find_a3_l193_193730


namespace pizza_slices_left_l193_193160

-- Defining the initial number of pizza slices
def initial_slices : ℕ := 16

-- Defining one-fourth of the pizza during dinner time
def dinner_fraction : ℚ := 1 / 4

-- Defining one-fourth of the remaining pizza eaten by Yves
def yves_fraction : ℚ := 1 / 4

-- Defining the slices eaten by each sibling
def slices_per_sibling : ℕ := 2

-- Theorem to prove the number of slices of pizza left is 5
theorem pizza_slices_left :
    let eaten_at_dinner := initial_slices * dinner_fraction
    let remaining_after_dinner := initial_slices - eaten_at_dinner
    let eaten_by_yves := remaining_after_dinner * yves_fraction
    let remaining_after_yves := remaining_after_dinner - eaten_by_yves
    let eaten_by_siblings := 2 * slices_per_sibling
    let final_remaining := remaining_after_yves - eaten_by_siblings
    final_remaining = 5 :=
by
  have step1 : let eaten_at_dinner := initial_slices * dinner_fraction
               let remaining_after_dinner := initial_slices - eaten_at_dinner
               eaten_at_dinner = 4 := by norm_num
  have step2 : let remaining_after_dinner := initial_slices - eaten_at_dinner
               remaining_after_dinner = 12 := by norm_num
  have step3 : let eaten_by_yves := remaining_after_dinner * yves_fraction
               eaten_by_yves = 3 := by norm_num
  have step4 : let remaining_after_yves := remaining_after_dinner - eaten_by_yves
               remaining_after_yves = 9 := by norm_num
  have step5 : let eaten_by_siblings := 2 * slices_per_sibling
               final_remaining = remaining_after_yves - eaten_by_siblings
               eaten_by_siblings = 4 := by norm_num
  show final_remaining = 5 from calc
    final_remaining 
      = remaining_after_yves - eaten_by_siblings := by norm_num
      ... = 5 := by norm_num

end pizza_slices_left_l193_193160


namespace find_principal_l193_193617

-- Define the given conditions and prove the principal amount is approximately 8057.43
theorem find_principal (P r R1 R2 : ℝ) (h1 : 8840 = P * (1 + r)^2) (h2 : 9261 = P * (1 + r)^3) :
  P ≈ 8057.43 :=
by
  -- the proof itself would go here
  sorry

end find_principal_l193_193617


namespace minute_hand_distance_l193_193545

noncomputable def distance_traveled (length_of_minute_hand : ℝ) (time_duration : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * length_of_minute_hand
  let revolutions := time_duration / 60
  circumference * revolutions

theorem minute_hand_distance :
  distance_traveled 8 45 = 12 * Real.pi :=
by
  sorry

end minute_hand_distance_l193_193545


namespace eight_people_lineup_with_adam_and_eve_together_l193_193110

/-- The number of ways to arrange eight people in a line such that Adam and Eve are always next to each other is 10080. -/
theorem eight_people_lineup_with_adam_and_eve_together : 
  ∃ (total_arrangements : ℕ), total_arrangements = 10080 ∧ 
  (∃ (places : finset (perm (fin 8))), 
    ∀ (p : perm (fin 8)), 
      (p ∈ places → 
        ((∃ (i : fin 7), p i.inj = some 0) ∧ 
         ((p i.inj = some 1) → (p (i+1).inj = some 0))) ∨ 
        ((p i.inj = some 1) ∧ 
         ((p (i+1).inj = some 0) → (p i.inj = some 1)))
      )
  ) :=
begin
  have factorial7 := nat.factorial 7,
  have fact_value := factorial7 * 2,
  use fact_value,
  split,
  { exact 10080 },
  { sorry }
end

end eight_people_lineup_with_adam_and_eve_together_l193_193110


namespace yuan_older_than_david_l193_193612

theorem yuan_older_than_david (David_age : ℕ) (Yuan_age : ℕ) 
  (h1 : Yuan_age = 2 * David_age) 
  (h2 : David_age = 7) : 
  Yuan_age - David_age = 7 := by
  sorry

end yuan_older_than_david_l193_193612


namespace smallest_t_l193_193969

theorem smallest_t (t : ℕ) (h : ∃ x : Fin t → ℤ, (∑ i, (x i)^3) = 2002^2002) : t = 4 :=
sorry

end smallest_t_l193_193969


namespace trigonometric_identity_l193_193163

theorem trigonometric_identity (t : ℝ) (ht : sin t ≠ 0) : 
  (4 * cos t ^ 2 - 1) / sin t = (cos t / sin t) * (1 + 2 * (2 * cos t ^ 2 - 1)) ↔ 
  ∃ k : ℤ, t = π / 3 * (3 * k + 1) ∨ t = π / 3 * (3 * k - 1) :=
sorry

end trigonometric_identity_l193_193163


namespace course_selection_schemes_l193_193196

theorem course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose (n k : ℕ) := nat.choose n k
  
  (choose 4 1 * choose 4 1 + 
   choose 4 2 * choose 4 1 + 
   choose 4 1 * choose 4 2) = 64 := 
by {
  -- calculating the combinations
  have comb_2_courses : choose 4 1 * choose 4 1 = 16 := by sorry,
  have comb_2_pe_1_art : choose 4 2 * choose 4 1 = 24 := by sorry,
  have comb_1_pe_2_art : choose 4 1 * choose 4 2 = 24 := by sorry,
  have total_comb : 16 + 24 + 24 = 64 := by {
    simp [comb_2_courses, comb_2_pe_1_art, comb_1_pe_2_art]
  },
  exact total_comb
}

end course_selection_schemes_l193_193196


namespace minute_hand_distance_traveled_in_45_minutes_l193_193547

-- Definitions for the problem
def minute_hand_length : ℝ := 8 -- The length of the minute hand
def time_in_minutes : ℝ := 45 -- Time in minutes
def revolutions_per_hour : ℝ := 1 -- The number of revolutions per hour for the minute hand (60 minutes)

-- Circumference of the circle
def circumference (radius : ℝ) : ℝ := 2 * Real.pi * radius

-- Number of revolutions the minute hand makes in the given time
def number_of_revolutions (time: ℝ) (total_minutes: ℝ) : ℝ :=
  time / total_minutes * revolutions_per_hour

-- Total distance traveled by the tip of the minute hand
def total_distance_traveled (radius : ℝ) (time: ℝ) : ℝ :=
  circumference(radius) * number_of_revolutions(time, 60)

-- Theorem to prove
theorem minute_hand_distance_traveled_in_45_minutes :
  total_distance_traveled(minute_hand_length, time_in_minutes) = 12 * Real.pi :=
by
  -- Placeholder for the proof
  sorry

end minute_hand_distance_traveled_in_45_minutes_l193_193547


namespace area_of_region_l193_193595

theorem area_of_region :
  (∀ x y : ℝ, x^2 + y^2 + 8 * x - 6 * y = -9) → 
  area := 16 * π :=
by sorry

end area_of_region_l193_193595


namespace kishore_expenses_l193_193308

noncomputable def total_salary (savings : ℕ) (percent : ℝ) : ℝ :=
savings / percent

noncomputable def total_expenses (rent milk groceries education petrol : ℕ) : ℕ :=
  rent + milk + groceries + education + petrol

noncomputable def miscellaneous_expenses (total_salary : ℝ) (total_expenses : ℕ) (savings : ℕ) : ℝ :=
  total_salary - (total_expenses + savings)

theorem kishore_expenses :
  total_salary 2160 0.1 - (total_expenses 5000 1500 4500 2500 2000 + 2160) = 3940 := by
  sorry

end kishore_expenses_l193_193308


namespace length_bc_l193_193964

theorem length_bc {A B E C : Point} 
  (r1 r2 : ℝ) 
  (h_r1 : r1 = 4) 
  (h_r2 : r2 = 6) 
  (h1 : dist A B = r1 + r2) 
  (h2 : tangent_line_to_circle A B E C)
  : dist B C = 0.77 := by
  sorry

end length_bc_l193_193964


namespace positive_solution_count_l193_193362

theorem positive_solution_count :
  ∀ x : ℝ, x > 0 → (cos (arcsin (tan (arccos x))) = x) → x = 1 :=
by
  sorry

end positive_solution_count_l193_193362


namespace avery_work_time_l193_193025

theorem avery_work_time :
  ∀ (t : ℝ),
    (1/2 * t + 1/4 * 1 = 1) → t = 1 :=
by
  intros t h
  sorry

end avery_work_time_l193_193025


namespace real_part_largest_modulus_root_l193_193538

theorem real_part_largest_modulus_root 
  (z : ℂ) 
  (h : 5 * z^4 + 10 * z^3 + 10 * z^2 + 5 * z + 1 = 0) 
  (h_modulus : ∀ w : ℂ, (5 * w^4 + 10 * w^3 + 10 * w^2 + 5 * w + 1 = 0) → |z| ≥ |w|) : 
  z.re = -1/2 :=
sorry

end real_part_largest_modulus_root_l193_193538


namespace Jane_buys_three_bagels_l193_193694

theorem Jane_buys_three_bagels (b m c : ℕ) (h1 : b + m + c = 5) (h2 : 80 * b + 60 * m + 100 * c = 400) : b = 3 := 
sorry

end Jane_buys_three_bagels_l193_193694


namespace f_monotonically_increasing_f_eq_zero_solutions_l193_193765

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := (Real.cos x)^2 + 2 * Real.sqrt 3 * (Real.sin x) * (Real.cos x) - (Real.sin x)^2

-- Intervals where the function is monotonically increasing
theorem f_monotonically_increasing (k : ℤ) : 
  ∀ x : ℝ, k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 → 
  f x = 2 * Real.sin (2 * x + Real.pi / 6) := 
sorry

-- Solutions of f(x) = 0 within the interval (0, π]
theorem f_eq_zero_solutions (x : ℝ) : 
  (0 < x ∧ x ≤ Real.pi) ∧ (x = 5 * Real.pi / 12 ∨ x = 11 * Real.pi / 12) →
  f x = 0 := 
sorry

end f_monotonically_increasing_f_eq_zero_solutions_l193_193765


namespace find_xyz_sum_l193_193171

theorem find_xyz_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + x * y + y^2 = 12)
  (h2 : y^2 + y * z + z^2 = 25)
  (h3 : z^2 + z * x + x^2 = 37) :
  x * y + y * z + z * x = 20 :=
sorry

end find_xyz_sum_l193_193171


namespace paigeRatio_l193_193624

/-- The total number of pieces in the chocolate bar -/
def totalPieces : ℕ := 60

/-- Michael takes half of the chocolate bar -/
def michaelPieces : ℕ := totalPieces / 2

/-- Mandy gets a fixed number of pieces -/
def mandyPieces : ℕ := 15

/-- The number of pieces left after Michael takes his share -/
def remainingPiecesAfterMichael : ℕ := totalPieces - michaelPieces

/-- The number of pieces Paige takes -/
def paigePieces : ℕ := remainingPiecesAfterMichael - mandyPieces

/-- The ratio of the number of pieces Paige takes to the number of pieces left after Michael takes his share is 1:2 -/
theorem paigeRatio :
  paigePieces / (remainingPiecesAfterMichael / 15) = 1 := sorry

end paigeRatio_l193_193624


namespace exists_convex_polyhedron_l193_193023

noncomputable def convex_polyhedron_exists : Prop :=
  ∃ (P : Polyhedron), 
    convex P ∧
    (∃ S : CrossSection, is_triangle S ∧ not (S passes_through_vertices)) ∧
    (∀ v : Vertex, (∑ (e : Edge) in incident_edges v, 1) = 5)

theorem exists_convex_polyhedron (h1 : convex_polyhedron_exists) : 
  ∃ P : Polyhedron, 
    convex P ∧
    (∃ S : CrossSection, is_triangle S ∧ not (S passes_through_vertices)) ∧
    (∀ v : Vertex, (∑ (e : Edge) in incident_edges v, 1) = 5) :=
sorry

end exists_convex_polyhedron_l193_193023


namespace solve_for_x_l193_193079

theorem solve_for_x (x : ℤ) (h_eq : (7 * x - 5) / (x - 2) = 2 / (x - 2)) (h_cond : x ≠ 2) : x = 1 := by
  sorry

end solve_for_x_l193_193079


namespace number_of_scenarios_l193_193295

theorem number_of_scenarios (companies : Finset ℕ) (representatives_A : ℤ) (representatives_others : ℕ) 
  (h1 : companies.card = 5) (h2 : representatives_A = 2) (h3 : ∀ n ∈ companies \ {0}, representatives_others = 1) : 
  ( -- condition to ensure that there are exactly 5 companies and all companies from 1 to 4 have 1 representative
    representatives_others.card = 4 ∧ representatives_A.card = 2
  ) → 
  -- provide the correct answer with the conditions:
  ∑ x in (companies.product companies).filter (λ p, p.1 ≠ p.2), 1 = 16 :=
sorry

end number_of_scenarios_l193_193295


namespace ratio_area_of_rectangle_to_square_l193_193931

theorem ratio_area_of_rectangle_to_square (s : ℝ) :
  (1.2 * s * 0.8 * s) / (s * s) = 24 / 25 :=
by
  sorry

end ratio_area_of_rectangle_to_square_l193_193931


namespace correct_formula_l193_193772

def relationship (x y : ℕ) : Prop :=
  (x = 0 ∧ y = 200) ∨ (x = 1 ∧ y = 170) ∨ (x = 2 ∧ y = 120) ∨ (x = 3 ∧ y = 50) ∨ (x = 4 ∧ y = 0)

theorem correct_formula :
  ∀ (x y : ℕ), relationship x y → y = 200 - 15 * x - 15 * x^2 :=
by
  intros x y h
  cases h with h0 h1
  case inl { cases h0 with h0x h0y; simp [h0x, h0y] }
  case inr {
    cases h1 with h1
    case inl { cases h1 with h1x h1y; simp [h1x, h1y] }
    case inr {
      cases h1 with h2
      case inl { cases h2 with h2x h2y; simp [h2x, h2y] }
      case inr {
        cases h2 with h3
        case inl { cases h3 with h3x h3y; simp [h3x, h3y] }
        case inr { cases h3 with h4x h4y; simp [h4x, h4y] }
      }
    }
  }

end correct_formula_l193_193772


namespace find_length_l193_193971

-- Definitions of the conditions
def width : ℕ := 5
def height : ℕ := 2
def surface_area : ℕ := 104

-- The query: finding the length of the rectangular solid
theorem find_length (L : ℕ) (h_w : width = 5) (h_h : height = 2) (h_sa : surface_area = 104) : L = 6 :=
sorry

end find_length_l193_193971


namespace dot_product_b_c_l193_193040

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)
variables (h_norm_a : ∥a∥ = 1)
variables (h_norm_b : ∥b∥ = 1)
variables (h_norm_ab : ∥a + b∥ = sqrt 3)
variables (h_c : c = a + 3 • b + 2 • (a × b))

theorem dot_product_b_c :
  inner_product_space.dot_product b c = 7 / 2 :=
sorry

end dot_product_b_c_l193_193040


namespace area_ratio_rect_sq_l193_193925

variable (s : ℝ)

def side_len_sq (S : ℝ) : Prop := s = S
def longer_side_rect (R : ℝ) : Prop := R = 1.2 * s
def shorter_side_rect (R : ℝ) : Prop := R = 0.8 * s
def area_sq (S : ℝ) : ℝ := S * S
def area_rect (R_long R_short : ℝ) : ℝ := R_long * R_short
def ratio_area (areaR areaS : ℝ) : ℝ := areaR / areaS

theorem area_ratio_rect_sq (s S R_long R_short : ℝ) (h1 : side_len_sq s S) (h2 : longer_side_rect s R_long) (h3 : shorter_side_rect s R_short) :
  ratio_area (area_rect R_long R_short) (area_sq S) = 24/25 :=
by
  sorry

end area_ratio_rect_sq_l193_193925


namespace positive_slope_asymptote_l193_193705

def hyperbola (x y : ℝ) :=
  Real.sqrt ((x - 1) ^ 2 + (y + 2) ^ 2) - Real.sqrt ((x - 6) ^ 2 + (y + 2) ^ 2) = 4

theorem positive_slope_asymptote :
  ∃ (m : ℝ), m = 0.75 ∧ (∃ x y, hyperbola x y) :=
sorry

end positive_slope_asymptote_l193_193705


namespace total_course_selection_schemes_l193_193234

theorem total_course_selection_schemes (
  pe_courses art_courses : Finset ℕ
) : 
  pe_courses.card = 4 →
  art_courses.card = 4 →
  let total_schemes := 
    (pe_courses.card.choose 1 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 2 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 1 * art_courses.card.choose 2) in
  total_schemes = 64 := 
by
  intros h1 h2
  let case1 := pe_courses.card.choose 1 * art_courses.card.choose 1 -- 4 * 4
  let case2_1 := pe_courses.card.choose 2 * art_courses.card.choose 1 -- 6 * 4
  let case2_2 := pe_courses.card.choose 1 * art_courses.card.choose 2 -- 4 * 6
  let case2 := case2_1 + case2_2 -- 24 + 24
  let total_schemes := case1 + case2 -- 16 + 48
  have hcard := by
    simp [Nat.choose]
    exact h1
  have hcase1 : case1 = 16 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_1 : case2_1 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_2 : case2_2 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2 : case2 = 48 := by {
    rw [hcase2_1, hcase2_2]
    norm_num
  }
  have htotal_schemes : total_schemes = 64 := by {
    rw [hcase1, hcase2]
    norm_num
  }
  exact htotal_schemes

end total_course_selection_schemes_l193_193234


namespace books_bought_l193_193094

def cost_price_of_books (n : ℕ) (C : ℝ) (S : ℝ) : Prop :=
  n * C = 16 * S

def gain_or_loss_percentage (gain_loss_percent : ℝ) : Prop :=
  gain_loss_percent = 0.5

def loss_selling_price (C : ℝ) (S : ℝ) (gain_loss_percent : ℝ) : Prop :=
  S = (1 - gain_loss_percent) * C
  
theorem books_bought (n : ℕ) (C : ℝ) (S : ℝ) (gain_loss_percent : ℝ) 
  (h1 : cost_price_of_books n C S) 
  (h2 : gain_or_loss_percentage gain_loss_percent) 
  (h3 : loss_selling_price C S gain_loss_percent) : 
  n = 8 := 
sorry 

end books_bought_l193_193094


namespace games_planned_to_attend_this_month_l193_193842

theorem games_planned_to_attend_this_month (T A_l P_l M_l P_m : ℕ) 
  (h1 : T = 12) 
  (h2 : P_l = 17) 
  (h3 : M_l = 16) 
  (h4 : A_l = P_l - M_l) 
  (h5 : T = A_l + P_m) : P_m = 11 :=
by 
  sorry

end games_planned_to_attend_this_month_l193_193842


namespace max_divisors_and_sum_l193_193506

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

def sum_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).sum id

theorem max_divisors_and_sum (h : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → (num_divisors 20) ≥ (num_divisors n)) :
  sum_divisors 20 = 42 :=
by
  sorry

end max_divisors_and_sum_l193_193506


namespace triangle_proportion_l193_193443

theorem triangle_proportion (p q r x y : ℝ)
  (h1 : x / q = y / r)
  (h2 : x + y = p) :
  y / r = p / (q + r) := sorry

end triangle_proportion_l193_193443


namespace polynomial_constant_for_large_n_l193_193355

theorem polynomial_constant_for_large_n (n : ℕ) (p : ℤ[X]) (h : ∀ k : ℕ, k ≤ n + 1 → 0 ≤ p.eval k ∧ p.eval k ≤ ↑n) : 4 ≤ n ↔ ∀ k : ℕ, k ≤ n + 1 → p.eval 0 = p.eval k := by
  sorry

end polynomial_constant_for_large_n_l193_193355


namespace find_m_find_area_l193_193800

-- Definitions based on conditions:
def triangle_angle_sum (A B C : ℝ) : Prop := A + B + C = 180
def arithmetic_sequence (A B C : ℝ) : Prop := 2 * B = A + C

-- The first proof problem:
theorem find_m 
  (A B C a b c m : ℝ)
  (h1 : triangle_angle_sum A B C)
  (h2 : arithmetic_sequence B A C)
  (h3 : A = 60)
  (h4 : a^2 - c^2 = b^2 - m * b * c) :
  m = 1 := by
  sorry

-- Definitions and assumptions for the second proof problem:
def law_of_cosines (a b c A : ℝ) : Prop := b^2 + c^2 - 2 * b * c * Real.cos A = a^2

-- The second proof problem:
theorem find_area
  (A : ℝ)
  (a b c : ℝ)
  (h1 : A = 60)
  (h2 : a = Real.sqrt 3)
  (h3 : b + c = 3)
  (h4 : law_of_cosines a b c A)
  (h5 : b * c = 2) :
  let area := 0.5 * b * c * Real.sin A in
  area = Real.sqrt 3 / 2 := by
  sorry

end find_m_find_area_l193_193800


namespace geometric_series_sum_l193_193542

theorem geometric_series_sum (a r : ℝ)
  (h₁ : a / (1 - r) = 15)
  (h₂ : a / (1 - r^4) = 9) :
  r = 1 / 3 :=
sorry

end geometric_series_sum_l193_193542


namespace find_product_of_integers_l193_193126

theorem find_product_of_integers
  (a b c d e : ℤ)
  (h_sets : Multiset (a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e) =
             {-1, 4, 6, 9, 10, 11, 15, 16, 20, 22}) :
  a * b * c * d * e = -4914 :=
sorry

end find_product_of_integers_l193_193126


namespace course_selection_schemes_l193_193199

theorem course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose (n k : ℕ) := nat.choose n k
  
  (choose 4 1 * choose 4 1 + 
   choose 4 2 * choose 4 1 + 
   choose 4 1 * choose 4 2) = 64 := 
by {
  -- calculating the combinations
  have comb_2_courses : choose 4 1 * choose 4 1 = 16 := by sorry,
  have comb_2_pe_1_art : choose 4 2 * choose 4 1 = 24 := by sorry,
  have comb_1_pe_2_art : choose 4 1 * choose 4 2 = 24 := by sorry,
  have total_comb : 16 + 24 + 24 = 64 := by {
    simp [comb_2_courses, comb_2_pe_1_art, comb_1_pe_2_art]
  },
  exact total_comb
}

end course_selection_schemes_l193_193199


namespace total_cost_correct_l193_193318

def typeA_price := 25
def typeA_tax := 0.06
def typeB_price := 35
def typeB_tax := 0.08

def discount (p : ℕ → ℕ → ℕ → ℕ) (n : ℕ) : ℕ :=
  if n <= 3 then p n 20 0
  else if n <= 6 then p n 20 15
  else p n 20 15 25

def apply_discount (base_price discount1 discount2 discount3 : ℕ) : ℕ :=
  let d1 := base_price * discount1 / 100
  let d2 := (base_price - d1) * discount2 / 100
  let d3 := (base_price - d1 - d2) * discount3 / 100
  base_price - d1 - d2 - d3

def calculate_cost (num_chairs type_price type_tax) : ℕ :=
  let total_price := (discount apply_discount num_chairs type_price)
  total_price + total_price * type_tax / 100

theorem total_cost_correct : calculate_cost 8 typeA_price typeA_tax + calculate_cost 5 typeB_price typeB_tax = 286.82 :=
sorry

end total_cost_correct_l193_193318


namespace max_value_circumscribed_quadrilateral_l193_193555

theorem max_value_circumscribed_quadrilateral (ABCD : Type) 
  (h_circumscribed : isCircumscribed ABCD) (r : ℝ) (hr : r = 1) :
  ∃ AC BD : ℝ, isDiagonals ABCD AC BD ∧ (∃ s : ℝ, s = \frac{1}{4} ∧ 
  (abs (\frac{1}{AC^2} + \frac{1}{BD^2}) = s)) :=
begin
  sorry
end

end max_value_circumscribed_quadrilateral_l193_193555


namespace min_value_y_l193_193968

theorem min_value_y : ∀ (x : ℝ), ∃ y_min : ℝ, y_min = (x^2 + 16 * x + 10) ∧ ∀ (x' : ℝ), (x'^2 + 16 * x' + 10) ≥ y_min := 
by 
  sorry

end min_value_y_l193_193968


namespace quadratic_sequence_geom_l193_193872

theorem quadratic_sequence_geom
  (a_n a_{n+1} : ℝ) (α β : ℝ)
  (h_quad : 0 = a_n * α^2 - a_{n+1} * α + 1 ∧ 0 = a_n * β^2 - a_{n+1} * β + 1)
  (h_cond : 6 * α - 2 * α * β + 6 * β = 3) :
  a_{n+1} = 1 / 2 * a_n + 1 / 3 ∧ ∀ n, let a_n_minus_2_over_3 := a_n - 2 / 3 in ∃ r, ∀ m, a_n_minus_2_over_3 ^ m = r * a_n_minus_2_over_3 :=
  by
  sorry

end quadratic_sequence_geom_l193_193872


namespace total_course_selection_schemes_l193_193230

theorem total_course_selection_schemes (
  pe_courses art_courses : Finset ℕ
) : 
  pe_courses.card = 4 →
  art_courses.card = 4 →
  let total_schemes := 
    (pe_courses.card.choose 1 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 2 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 1 * art_courses.card.choose 2) in
  total_schemes = 64 := 
by
  intros h1 h2
  let case1 := pe_courses.card.choose 1 * art_courses.card.choose 1 -- 4 * 4
  let case2_1 := pe_courses.card.choose 2 * art_courses.card.choose 1 -- 6 * 4
  let case2_2 := pe_courses.card.choose 1 * art_courses.card.choose 2 -- 4 * 6
  let case2 := case2_1 + case2_2 -- 24 + 24
  let total_schemes := case1 + case2 -- 16 + 48
  have hcard := by
    simp [Nat.choose]
    exact h1
  have hcase1 : case1 = 16 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_1 : case2_1 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_2 : case2_2 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2 : case2 = 48 := by {
    rw [hcase2_1, hcase2_2]
    norm_num
  }
  have htotal_schemes : total_schemes = 64 := by {
    rw [hcase1, hcase2]
    norm_num
  }
  exact htotal_schemes

end total_course_selection_schemes_l193_193230


namespace product_y_coordinates_l193_193513

theorem product_y_coordinates : 
  ∀ y : ℝ, (∀ P : ℝ × ℝ, P.1 = -1 ∧ (P.1 - 4)^2 + (P.2 - 3)^2 = 64 → P = (-1, y)) →
  ((3 + Real.sqrt 39) * (3 - Real.sqrt 39) = -30) :=
by
  intros y h
  sorry

end product_y_coordinates_l193_193513


namespace parabola_translation_l193_193134

theorem parabola_translation :
  ∀(x y : ℝ), y = - (1 / 3) * (x - 5) ^ 2 + 3 →
  ∃(x' y' : ℝ), y' = -(1/3) * x'^2 + 6 := by
  sorry

end parabola_translation_l193_193134


namespace parallel_lines_slope_equal_l193_193406

theorem parallel_lines_slope_equal (k : ℝ) : (∀ x : ℝ, 2 * x = k * x + 3) → k = 2 :=
by
  intros
  sorry

end parallel_lines_slope_equal_l193_193406


namespace quadratic_residue_solution_l193_193893

theorem quadratic_residue_solution 
  (p : ℕ) [Fact (Nat.Prime p)]
  (a b : ℕ)
  (h_a : ¬ p ∣ a)
  (h_b : ¬ p ∣ b)
  (has_solution_a : ∃ x y : ℤ, x^2 - a = p * y)
  (has_solution_b : ∃ x y : ℤ, x^2 - b = p * y)
  : ∃ x y : ℤ, x^2 - a * b = p * y :=
sorry

end quadratic_residue_solution_l193_193893


namespace course_selection_count_l193_193223

-- Definitions for the conditions
def num_PE_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_PE_courses + num_art_courses

-- The number of different course selection schemes
theorem course_selection_count : 
  (choose num_PE_courses 1) * (choose num_art_courses 1) + 
  (choose num_PE_courses 2) * (choose num_art_courses 1) + 
  (choose num_PE_courses 1) * (choose num_art_courses 2) = 64 := 
by sorry

end course_selection_count_l193_193223


namespace tangents_to_hyperbola_l193_193304

noncomputable def eccentricity_of_hyperbola 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h4 : let x^2 + y^2 = a^2/4, tangent_line x^2 + y^2 = a^2/4 intersects hyperbola x^2/a^2 - y^2/b^2 = 1 at point E)
  (h5 : ∃ P, F(-c, 0), ∀ O, OF = (1/2)(OE + OP)) 
  : ℝ := 
  have |OF| = c := sorry,
  have |OE| = a/2 := sorry,
  have |EF| = sqrt(c^2 - (a^2 / 4)) := sorry,
  have |PF| = 2 * sqrt(c^2 - (a^2 / 4)) := sorry,
  have |PF'| = a := sorry,
  have h6 : |PF| - |PF'| = 2 * a := sorry,
  e_formula c a b := sqrt(10) / 2

theorem tangents_to_hyperbola 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h4 : let x^2 + y^2 = a^2 / 4, tangent_line x^2 + y^2 = a^2 / 4 intersects hyperbola x^2 / a^2 - y^2 / b^2 = 1 at point E)
  (h5 : ∃ P, F(-c, 0), ∀ O, OF = (1 / 2) (OE + OP)) 
  : eccentricity_of_hyperbola a b c = sqrt(10) / 2 := 
by {
  sorry  
}

end tangents_to_hyperbola_l193_193304


namespace solution_of_linear_system_l193_193404

theorem solution_of_linear_system (a b : ℚ) :
  ∃ x y : ℚ, (a - b) * x - (a + b) * y = a + b ∧ x = 0 ∧ y = -1 :=
by
  use 0
  use -1
  sorry

end solution_of_linear_system_l193_193404


namespace relationship_abc_l193_193720

noncomputable def a : ℝ := 5 ^ (Real.log 3.4 / Real.log 3)
noncomputable def b : ℝ := 5 ^ (Real.log 3.6 / Real.log 3)
noncomputable def c : ℝ := (1 / 5) ^ (Real.log 0.5 / Real.log 3)

theorem relationship_abc : b > a ∧ a > c :=
by
  -- Assumptions derived from logarithmic properties.
  have h1 : Real.log 2 < Real.log 3.4 := sorry
  have h2 : Real.log 3.4 < Real.log 3.6 := sorry
  have h3 : Real.log 0.5 < 0 := sorry
  have h4 : Real.log 2 / Real.log 3 = Real.log 2 := sorry
  have h5 : Real.log 0.5 / Real.log 3 = -Real.log 2 := sorry

  -- Monotonicity of exponential function.
  apply And.intro
  { exact sorry }
  { exact sorry }

end relationship_abc_l193_193720


namespace area_projection_range_l193_193827

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x
noncomputable def VABCD_regular_quad_pyramid (AB VA : ℝ) := AB = 2 ∧ VA = sqrt 6

theorem area_projection_range :
  (∀ (regular_quad_pyramid : ℝ → ℝ → Prop) (AB VA : ℝ)
  (rotate_around_AB : Prop) (CD_parallel_to_plane_α : Prop),
  regular_quad_pyramid AB VA → 
  rotate_around_AB →
  CD_parallel_to_plane_α →
  (2 : ℝ) ≤ range (projection_area (pyramid_projection_on_plane_α VABCD_regular_quad_pyramid) CD_parallel_to_plane_α) ∧
  range (projection_area (pyramid_projection_on_plane_α VABCD_regular_quad_pyramid) CD_parallel_to_plane_α) ≤ 4 :=
begin
  intros,
  have key_prop : regular_quad_pyramid 2 (sqrt 6) := by sorry,
  exact key_prop,
end

end area_projection_range_l193_193827


namespace find_s_l193_193483

theorem find_s (c d n r s : ℝ) 
(h1 : c * d = 3)
(h2 : ∃ p q : ℝ, (p + q = r) ∧ (p * q = s) ∧ (p = c + 1/d ∧ q = d + 1/c)) :
s = 16 / 3 :=
by
  sorry

end find_s_l193_193483


namespace john_blue_pens_l193_193028

variables (R B Bl : ℕ)

axiom total_pens : R + B + Bl = 31
axiom black_more_red : B = R + 5
axiom blue_twice_black : Bl = 2 * B

theorem john_blue_pens : Bl = 18 :=
by
  apply sorry

end john_blue_pens_l193_193028


namespace find_R_plus_S_l193_193790

theorem find_R_plus_S (d e R S : ℝ) 
  (h1 : d + 3 = 0)
  (h2 : 7 * d + 3 * e = 0)
  (h3 : R = 3 * d + e + 7)
  (h4 : S = 7 * e) :
  R + S = 54 :=
by
  sorry

end find_R_plus_S_l193_193790


namespace janet_wait_time_l193_193838

theorem janet_wait_time 
  (janet_speed : ℝ)
  (sister_speed : ℝ)
  (lake_width : ℝ)
  (h1 : janet_speed = 30)
  (h2 : sister_speed = 12)
  (h3 : lake_width = 60) 
  :
  60 / 12 - 60 / 30 = 3 :=
by
  sorry

end janet_wait_time_l193_193838


namespace minimum_width_l193_193890

theorem minimum_width (w : ℝ) (h_area : w * (w + 15) ≥ 200) : w ≥ 10 :=
by
  sorry

end minimum_width_l193_193890


namespace most_accurate_value_l193_193818

theorem most_accurate_value (C : ℝ) (err : ℝ) (announced_value : ℝ) 
  (hC : C = 2.43865) (herr : err = 0.00312) (hannounced : announced_value = 2.44) : 
  ∀ (x : ℝ), C - err ≤ x ∧ x ≤ C + err → (Float.ofReal (Real.round_near x 2)) = announced_value := 
by 
  sorry

end most_accurate_value_l193_193818


namespace surface_area_sphere_l193_193462

-- Definitions based on conditions
def SA : ℝ := 3
def SB : ℝ := 4
def SC : ℝ := 5
def vertices_perpendicular : Prop := ∀ (a b c : ℝ), (a = SA ∧ b = SB ∧ c = SC) → (a * b * c = SA * SB * SC)

-- Definition of the theorem based on problem and correct answer
theorem surface_area_sphere (h1 : vertices_perpendicular) : 
  4 * Real.pi * ((Real.sqrt (SA^2 + SB^2 + SC^2)) / 2)^2 = 50 * Real.pi :=
by
  -- skip the proof
  sorry

end surface_area_sphere_l193_193462


namespace B_values_for_divisibility_l193_193106

theorem B_values_for_divisibility (B : ℕ) (h : 4 + B + B + B + 2 ≡ 0 [MOD 9]) : B = 1 ∨ B = 4 ∨ B = 7 :=
by sorry

end B_values_for_divisibility_l193_193106


namespace max_value_of_f_l193_193495

/-- Define the function f as the minimum of three expressions -/
def f (x : ℝ) : ℝ := min (min (4*x + 1) (x + 4)) (-x + 8)

/-- Prove that the maximum value of f(x) within its domain is 6 -/
theorem max_value_of_f : ∃ x : ℝ, f x = 6 :=
by
  sorry

end max_value_of_f_l193_193495


namespace compute_fraction_l193_193672

theorem compute_fraction : (2015 : ℝ) / ((2015 : ℝ)^2 - (2016 : ℝ) * (2014 : ℝ)) = 2015 :=
by {
  sorry
}

end compute_fraction_l193_193672


namespace true_propositions_count_l193_193944

theorem true_propositions_count (a b : ℝ) (c : ℝ) :
  (¬(a > b ∧ c^2 = 0 ∧ ac^2 = bc^2)) →
  ((a > b   → ac^2 > bc^2) ∧
  ((ac^2 > bc^2 → a > b) ∧
  ((¬(a > b) ∧ ¬(ac^2 > bc^2))))) ↔ 2 = 2 := 
by {
  sorry
}

end true_propositions_count_l193_193944


namespace problem1_problem2_l193_193332

-- Defining the expression in problem (1)
def expr1 : ℚ := ((9/4)^(1/2)) - ((-9.6)^0) - ((27/8)^(-2/3)) + ((1/10)^(-2))

-- Proving it is equal to (1801/18)
theorem problem1 : expr1 = 1801 / 18 :=
by 
  -- The proof would go here, but we include sorry to skip it
  sorry

-- Defining the condition and the expression in problem (2)
variables {x : ℝ} (hx : x + x⁻¹ = 3)

-- Defining the required fraction in problem (2)
def expr2 : ℝ := (x^(1/2) + x^(-1/2)) / (x^2 + x^(-2) + 3)

-- Proving it is equal to (sqrt 5 / 10)
theorem problem2 : hx → expr2 = (√5) / 10 :=
by 
  -- The proof would go here, but we include sorry to skip it
  sorry

end problem1_problem2_l193_193332


namespace inequality_solution_l193_193898

theorem inequality_solution (x : ℝ) : 
  (3 - (1 / (3 * x + 4)) < 5) ↔ (x < -4 / 3) := 
by
  sorry

end inequality_solution_l193_193898


namespace total_course_selection_schemes_l193_193190

theorem total_course_selection_schemes :
  let n_physical_education := 4
  let n_art := 4
  let total_courses := n_physical_education + n_art in
  let choose2_courses := (Nat.choose n_physical_education 1) * (Nat.choose n_art 1)
  let choose3_courses := (Nat.choose n_physical_education 2 * Nat.choose n_art 1) + (Nat.choose n_physical_education 1 * Nat.choose n_art 2) in
  total_courses = n_physical_education + n_art →
  choose2_courses + choose3_courses = 64 :=
by
  intros n_physical_education n_art total_courses choose2_courses choose3_courses h
  have h_choose2_courses: choose2_courses = 16 := by
    simp [n_physical_education, n_art, Nat.choose]
  have h_choose3_courses: choose3_courses = 48 := by
    simp [n_physical_education, n_art, Nat.choose]
  rw [h_choose2_courses, h_choose3_courses]
  exact Nat.add_eq_right.2 rfl

end total_course_selection_schemes_l193_193190


namespace ellipse_and_area_proof_l193_193754

/-
  Problem conditions:
  - The foci of the ellipse are F1(-2, 0) and F2(2, 0).
  - One of the vertices on the minor axis is B(0, -sqrt(5)).
-/

noncomputable def ellipse_equation : Prop := 
  ∃ (a b : ℝ) (c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0), 
    c = 2 ∧ b = Real.sqrt 5 ∧ a^2 = b^2 + c^2 ∧ 
    (a > b) ∧ 
    (∀ (x y : ℝ), 
        (x, y) ∈ set_of (λ p : ℝ × ℝ, p.1^2 / 9 + p.2^2 / 5 = 1) → 
        (∃ (x y : ℝ), 
          y = x + 2 ∧ 
          5 * x^2 + 9 * y^2 = 45))

noncomputable def area_triangle : Prop := 
  ∃ (y1 y2 : ℝ), 
    y1 + y2 = 10 / 7 ∧ 
    y1 * y2 = -(25 / 14) ∧ 
    (1 / 2) * 4 * Real.abs (y1 - y2) = 6 * Real.sqrt 50 / 7

-- Main theorem statement combining both proofs
theorem ellipse_and_area_proof : ellipse_equation ∧ area_triangle := 
  by 
    sorry

end ellipse_and_area_proof_l193_193754


namespace shaded_area_is_4_over_3_l193_193644

-- Define the side length of the square
def side_length : ℝ := 2

-- Define the angle β with its cosine value
def β (x : ℝ) : Prop := 
  0 < x ∧ x < π/2 ∧ cos x = 3/5

-- Objective: Prove the shaded area is 4/3
theorem shaded_area_is_4_over_3 (x : ℝ) (hβ: β x) : ∃ a : ℝ, a = 4/3 := by
  sorry

end shaded_area_is_4_over_3_l193_193644


namespace intersection_M_N_eq_l193_193774

def setM : Set ℝ := {y | ∃ x : ℝ, y = Real.log10 (x^2 + 1)}
def setN : Set ℝ := {x | x > 1}

theorem intersection_M_N_eq :
  {y | ∃ x ∈ setN, y = Real.log10 (x^2 + 1)} = {y | y > 0} :=
by
  sorry

end intersection_M_N_eq_l193_193774


namespace difference_in_money_l193_193507

structure MoneyData :=
  (initial_amount : ℕ)
  (spent_amount : ℕ)
  (exchange_rate : Float)

structure TravelerData :=
  (us_money : ℕ)
  (foreign_money : MoneyData)

def calc_usd (money : MoneyData) : Float :=
  (money.initial_amount - money.spent_amount) * money.exchange_rate

def calc_total_usd (us_money : ℕ) (foreign_data : MoneyData) : Float :=
  us_money.toFloat - (foreign_data.spent_amount.toFloat * foreign_data.exchange_rate) + 
    calc_usd foreign_data

def oliver_remaining_money : Float :=
  calc_total_usd 140 (MoneyData.mk 7000 3000 0.0091) + 200.1

def william_remaining_money : Float :=
  calc_total_usd 150 (MoneyData.mk 350 150 0.78) + 171.1

theorem difference_in_money :
  (oliver_remaining_money - william_remaining_money) = -100.6 :=
  by
  sorry

end difference_in_money_l193_193507


namespace range_of_m_l193_193724

theorem range_of_m (m : ℝ) (h1 : ∀ x : ℝ, (x^2 + 1) * (x^2 - 8*x - 20) ≤ 0 → (-2 ≤ x → x ≤ 10))
    (h2 : ∀ x : ℝ, x^2 - 2*x + 1 - m^2 ≤ 0 → (1 - m ≤ x → x ≤ 1 + m))
    (h3 : m > 0)
    (h4 : ∀ x : ℝ, ¬ ((x^2 + 1) * (x^2 - 8*x - 20) ≤ 0) → ¬ (x^2 - 2*x + 1 - m^2 ≤ 0) → (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m)) :
  m ≥ 9 := 
sorry

end range_of_m_l193_193724


namespace continuity_condition_l193_193381

def f (x : ℝ) (b c : ℝ) : ℝ :=
if x > 2 then 3 * x + b else 5 * x + c

theorem continuity_condition (b c : ℝ) : (∀ x, continuous_at (f x b c) 2) ↔ b - c = 4 := 
by 
  sorry

end continuity_condition_l193_193381


namespace janet_wait_time_l193_193841

theorem janet_wait_time
  (janet_speed : ℝ)
  (sister_speed : ℝ)
  (lake_width : ℝ)
  (janet_time : ℝ)
  (sister_time : ℝ) :
  janet_speed = 30 →
  sister_speed = 12 →
  lake_width = 60 →
  janet_time = lake_width / janet_speed →
  sister_time = lake_width / sister_speed →
  (sister_time - janet_time = 3) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end janet_wait_time_l193_193841


namespace increase_function_a_seq_increasing_b_seq_decreasing_seq_relation_l193_193069

open Real

-- Defining the sequences
noncomputable def a_seq (n : ℕ) : ℝ := (1 + (1 : ℝ) / n) ^ n
noncomputable def b_seq (n : ℕ) : ℝ := (1 + (1 : ℝ) / n) ^ (n + 1)

theorem increase_function (x : ℝ) (hx : 0 < x) : 
  ((1:ℝ) + 1 / x) ^ x < (1 + 1 / (x + 1)) ^ (x + 1) := sorry

theorem a_seq_increasing (n : ℕ) (hn : 0 < n) : 
  a_seq n < a_seq (n + 1) := sorry

theorem b_seq_decreasing (n : ℕ) (hn : 0 < n) : 
  b_seq (n + 1) < b_seq n := sorry

theorem seq_relation (n : ℕ) (hn : 0 < n) : 
  a_seq n < b_seq n := sorry

end increase_function_a_seq_increasing_b_seq_decreasing_seq_relation_l193_193069


namespace cosine_smallest_angle_of_triangle_with_consecutive_sides_l193_193564

theorem cosine_smallest_angle_of_triangle_with_consecutive_sides (n : ℕ) (h : n ≥ 1)
  (h_triangle : (n: ℝ), (n+1: ℝ), (n+2: ℝ) is a triangle)
  (largest_angle_twice_smallest_angle : ∃ θ : ℝ, θ > 0 ∧ largest_angle = 2 * θ):
  cos (smallest_angle) = 3 / 4 :=
sorry

end cosine_smallest_angle_of_triangle_with_consecutive_sides_l193_193564


namespace max_books_borrowed_l193_193615

theorem max_books_borrowed (students : ℕ) (no_books : ℕ) (one_book : ℕ) (two_books : ℕ) (more_books : ℕ)
  (h_students : students = 30)
  (h_no_books : no_books = 5)
  (h_one_book : one_book = 12)
  (h_two_books : two_books = 8)
  (h_more_books : more_books = students - no_books - one_book - two_books)
  (avg_books : ℕ)
  (h_avg_books : avg_books = 2) :
  ∃ max_books : ℕ, max_books = 20 := 
by 
  sorry

end max_books_borrowed_l193_193615


namespace extremum_a_value_l193_193767

theorem extremum_a_value (a : ℝ) (h : ∃ x : ℝ, x = 1 ∧ deriv (fun x : ℝ => a * x - 2 * log x + 2) x = 0) : a = 2 :=
sorry

end extremum_a_value_l193_193767


namespace total_course_selection_schemes_l193_193228

theorem total_course_selection_schemes (
  pe_courses art_courses : Finset ℕ
) : 
  pe_courses.card = 4 →
  art_courses.card = 4 →
  let total_schemes := 
    (pe_courses.card.choose 1 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 2 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 1 * art_courses.card.choose 2) in
  total_schemes = 64 := 
by
  intros h1 h2
  let case1 := pe_courses.card.choose 1 * art_courses.card.choose 1 -- 4 * 4
  let case2_1 := pe_courses.card.choose 2 * art_courses.card.choose 1 -- 6 * 4
  let case2_2 := pe_courses.card.choose 1 * art_courses.card.choose 2 -- 4 * 6
  let case2 := case2_1 + case2_2 -- 24 + 24
  let total_schemes := case1 + case2 -- 16 + 48
  have hcard := by
    simp [Nat.choose]
    exact h1
  have hcase1 : case1 = 16 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_1 : case2_1 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_2 : case2_2 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2 : case2 = 48 := by {
    rw [hcase2_1, hcase2_2]
    norm_num
  }
  have htotal_schemes : total_schemes = 64 := by {
    rw [hcase1, hcase2]
    norm_num
  }
  exact htotal_schemes

end total_course_selection_schemes_l193_193228


namespace nth_equation_sum_l193_193505

theorem nth_equation_sum (n : ℕ) (hn : n ≥ 1) :
  (∑ k in finset.range (2 * n - 1), (n + k)) = (2 * n - 1)^2 :=
by
  sorry

end nth_equation_sum_l193_193505


namespace contrapositive_correct_l193_193093

-- Define the main condition: If a ≥ 1/2, then ∀ x ≥ 0, f(x) ≥ 0
def main_condition (a : ℝ) (f : ℝ → ℝ) : Prop :=
  a ≥ 1/2 → ∀ x : ℝ, x ≥ 0 → f x ≥ 0

-- Define the contrapositive statement: If ∃ x ≥ 0 such that f(x) < 0, then a < 1/2
def contrapositive (a : ℝ) (f : ℝ → ℝ) : Prop :=
  (∃ x : ℝ, x ≥ 0 ∧ f x < 0) → a < 1/2

-- Theorem to prove that the contrapositive statement is correct
theorem contrapositive_correct (a : ℝ) (f : ℝ → ℝ) :
  main_condition a f ↔ contrapositive a f :=
by
  sorry

end contrapositive_correct_l193_193093


namespace pens_at_end_l193_193054

-- Define the main variable
variable (x : ℝ)

-- Define the conditions as functions
def initial_pens (x : ℝ) := x
def mike_gives (x : ℝ) := 0.5 * x
def after_mike (x : ℝ) := x + (mike_gives x)
def after_cindy (x : ℝ) := 2 * (after_mike x)
def give_sharon (x : ℝ) := 0.25 * (after_cindy x)

-- Define the final number of pens
def final_pens (x : ℝ) := (after_cindy x) - (give_sharon x)

-- The theorem statement
theorem pens_at_end (x : ℝ) : final_pens x = 2.25 * x :=
by sorry

end pens_at_end_l193_193054


namespace jessie_current_weight_l193_193649

theorem jessie_current_weight :
  ∀ (initial_weight lost_weight : ℕ), initial_weight = 192 → lost_weight = 126 → initial_weight - lost_weight = 66 :=
by
  intros initial_weight lost_weight h_initial h_lost
  have h1 : initial_weight = 192 := h_initial
  have h2 : lost_weight = 126 := h_lost
  rw [h1, h2]
  sorry

end jessie_current_weight_l193_193649


namespace true_inequalities_count_l193_193057

-- Define the nonzero real numbers and their squares
variables {x y a b : ℝ}

-- Define the conditions provided in the problem
def condition1 := x ≠ 0 ∧ y ≠ 0 ∧ a ≠ 0 ∧ b ≠ 0
def condition2 := x^2 < a^2
def condition3 := y^2 < b^2

-- Define the hypotheses and the theorem to prove the correct number of true inequalities
theorem true_inequalities_count :
  condition1 → condition2 → condition3 → 
  (x^2 + y^2 < a^2 + b^2) ∧ (x^2 * y^2 < a^2 * b^2) → 2 :=
by
  intros _ _ _ _
  exact 2

end true_inequalities_count_l193_193057


namespace inequality_relation_l193_193043

def a := (0.2: ℝ)^0.3
def b := Real.log 2 / Real.log 0.3
def c := Real.log 0.2 / Real.log 0.3

theorem inequality_relation : b < a ∧ a < c := by
  sorry

end inequality_relation_l193_193043


namespace smallest_positive_period_of_f_symmetry_center_of_f_range_of_f_in_interval_l193_193430

open Real

noncomputable def a (x : ℝ) : ℝ × ℝ := (5 * sqrt 3 * cos x, cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (sin x, 2 * cos x)

noncomputable def f (x : ℝ) : ℝ := 
  let a_dot_b := (a x).1 * (b x).1 + (a x).2 * (b x).2
  let b_norm_sq := (b x).1 ^ 2 + (b x).2 ^ 2
  a_dot_b + b_norm_sq + 3 / 2

theorem smallest_positive_period_of_f :
  ∀ x, f (x + π) = f x := sorry

theorem symmetry_center_of_f :
  ∃ k : ℤ, ∀ x, f x = 5 ↔ x = (-π / 12 + k * (π / 2) : ℝ) := sorry

theorem range_of_f_in_interval :
  ∀ x, (π / 6 ≤ x ∧ x ≤ π / 2) → (5 / 2 ≤ f x ∧ f x ≤ 10) := sorry

end smallest_positive_period_of_f_symmetry_center_of_f_range_of_f_in_interval_l193_193430


namespace length_of_OP_l193_193588

theorem length_of_OP (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
    (√(a^2 + b^2 + c^2)) = 5 * (√2) :=
by 
  sorry

end length_of_OP_l193_193588


namespace course_selection_schemes_count_l193_193244

-- Definitions based on the conditions
def num_physical_education_courses : ℕ := 4
def num_art_courses : ℕ := 4

-- Required to choose 2 or 3 courses, with at least one from each category
def valid_selection_cases : list (ℕ × ℕ) := [(1, 1), (1, 2), (2, 1)]

-- Calculate the number of ways to choose k courses from n courses using combination formula
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Calculate the total number of different course selection schemes
def total_course_selection_schemes : ℕ :=
  let case1 := C num_physical_education_courses 1 * C num_art_courses 1
  let case2a := C num_physical_education_courses 2 * C num_art_courses 1
  let case2b := C num_physical_education_courses 1 * C num_art_courses 2
  case1 + case2a + case2b

-- The assertion that the total number of different course selection schemes is 64
theorem course_selection_schemes_count : total_course_selection_schemes = 64 := by
  -- sorry placeholder for proof
  sorry

end course_selection_schemes_count_l193_193244


namespace nonagon_diagonal_angle_less_than_seven_l193_193070

theorem nonagon_diagonal_angle_less_than_seven (n : ℕ) (h₀ : n = 9) :
  ∃ θ < 7, some_diagonals_form_angle θ n :=
by
  sorry

end nonagon_diagonal_angle_less_than_seven_l193_193070


namespace prime_dates_in_2012_l193_193320

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_months := {2, 3, 5, 7, 11}
def prime_days := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}

def days_in_month (month : ℕ) (is_leap_year : Bool) : ℕ :=
  if month = 2 then if is_leap_year then 29 else 28 else
  if month = 4 ∨ month = 6 ∨ month = 9 ∨ month = 11 then 30 else 31

theorem prime_dates_in_2012 : 
  let feb := 9, mar_may_jul := 11, nov := 10
  feb + mar_may_jul + mar_may_jul + mar_may_jul + nov = 52 := 
by sorry

end prime_dates_in_2012_l193_193320


namespace problem_statement_l193_193859

-- Define the odd function and the conditions given
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Main theorem statement
theorem problem_statement (f : ℝ → ℝ) 
  (h_odd : odd_function f)
  (h_periodic : ∀ x : ℝ, f (x + 1) = f (3 - x))
  (h_f1 : f 1 = -2) :
  2012 * f 2012 - 2013 * f 2013 = -4026 := 
sorry

end problem_statement_l193_193859


namespace reciprocal_neg1_l193_193557

theorem reciprocal_neg1 : ∃ x : ℝ, -1 * x = 1 ∧ x = -1 :=
by {
  use (-1),
  split,
  { exact by norm_num, },
  { refl, }
}

end reciprocal_neg1_l193_193557


namespace unit_circle_chords_l193_193829

theorem unit_circle_chords (
    s t u v : ℝ
) (hs : s = 1) (ht : t = 1) (hu : u = 2) (hv : v = 3) :
    (v - u = 1) ∧ (v * u = 6) ∧ (v^2 - u^2 = 5) :=
by
  have h1 : v - u = 1 := by rw [hv, hu]; norm_num
  have h2 : v * u = 6 := by rw [hv, hu]; norm_num
  have h3 : v^2 - u^2 = 5 := by rw [hv, hu]; norm_num
  exact ⟨h1, h2, h3⟩

end unit_circle_chords_l193_193829


namespace find_coefficients_l193_193428

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Definitions based on conditions
def A' (A B : V) : V := (3 : ℝ) • (B - A) + A
def B' (B C : V) : V := (3 : ℝ) • (C - B) + C

-- The problem statement
theorem find_coefficients (A A' B B' : V) (p q r : ℝ) 
  (hB : B = (1/4 : ℝ) • A + (3/4 : ℝ) • A') 
  (hC : C = (1/4 : ℝ) • B + (3/4 : ℝ) • B') : 
  ∃ (p q r : ℝ), A = p • A' + q • B + r • B' ∧ p = 4/13 ∧ q = 12/13 ∧ r = 48/13 :=
sorry

end find_coefficients_l193_193428


namespace krios_population_limit_l193_193820

theorem krios_population_limit (initial_population : ℕ) (acre_per_person : ℕ) (total_acres : ℕ) (doubling_years : ℕ) :
  initial_population = 150 →
  acre_per_person = 2 →
  total_acres = 35000 →
  doubling_years = 30 →
  ∃ (years_from_2005 : ℕ), years_from_2005 = 210 ∧ (initial_population * 2^(years_from_2005 / doubling_years)) ≥ total_acres / acre_per_person :=
by
  intros
  sorry

end krios_population_limit_l193_193820


namespace circle_area_of_circumscribed_triangle_l193_193628

theorem circle_area_of_circumscribed_triangle :
  let a := 12
  let b := 12
  let c := 10
  let height := Real.sqrt (a^2 - (c / 2)^2)
  let A := (1 / 2) * c * height
  let R := (a * b * c) / (4 * A)
  π * R^2 = (5184 / 119) * π := 
by
  let a := 12
  let b := 12
  let c := 10
  let height := Real.sqrt (a^2 - (c / 2)^2)
  let A := (1 / 2) * c * height
  let R := (a * b * c) / (4 * A)
  have h1 : height = Real.sqrt (a^2 - (c / 2)^2) := by sorry
  have h2 : A = (1 / 2) * c * height := by sorry
  have h3 : R = (a * b * c) / (4 * A) := by sorry
  have h4 : π * R^2 = (5184 / 119) * π := by sorry
  exact h4

end circle_area_of_circumscribed_triangle_l193_193628


namespace sin_alpha_value_l193_193441

-- Definitions
def vertex_of_angle_coincides_with_origin (α : ℝ) : Prop := true
def initial_side_on_positive_x_axis (α : ℝ) : Prop := true
def terminal_side_on_ray (α : ℝ) : Prop := 3 * Math.cos α + 4 * Math.sin α = 0 

-- Theorem statement
theorem sin_alpha_value (α : ℝ) 
    (h1 : vertex_of_angle_coincides_with_origin α)
    (h2 : initial_side_on_positive_x_axis α)
    (h3 : terminal_side_on_ray α) : 
    Math.sin α = -3 / 5 := 
sorry

end sin_alpha_value_l193_193441


namespace pizza_slices_left_l193_193162

def initial_slices : ℕ := 16
def eaten_during_dinner : ℕ := initial_slices / 4
def remaining_after_dinner : ℕ := initial_slices - eaten_during_dinner
def yves_eaten : ℕ := remaining_after_dinner / 4
def remaining_after_yves : ℕ := remaining_after_dinner - yves_eaten
def siblings_eaten : ℕ := 2 * 2
def remaining_after_siblings : ℕ := remaining_after_yves - siblings_eaten

theorem pizza_slices_left : remaining_after_siblings = 5 := by
  sorry

end pizza_slices_left_l193_193162


namespace dodgeball_cost_l193_193115

theorem dodgeball_cost (B : ℝ) 
  (hb1 : 1.20 * B = 90) 
  (hb2 : B / 15 = 5) :
  ∃ (cost_per_dodgeball : ℝ), cost_per_dodgeball = 5 := by
sorry

end dodgeball_cost_l193_193115


namespace sum_of_cubes_eq_96_over_7_l193_193568

-- Define the conditions from the problem
variables (a r : ℝ)
axiom condition_sum : a / (1 - r) = 2
axiom condition_sum_squares : a^2 / (1 - r^2) = 6

-- Define the correct answer that we expect to prove
theorem sum_of_cubes_eq_96_over_7 :
  a^3 / (1 - r^3) = 96 / 7 :=
sorry

end sum_of_cubes_eq_96_over_7_l193_193568


namespace carl_weight_l193_193665

variable (C R B : ℕ)

theorem carl_weight (h1 : B = R + 9) (h2 : R = C + 5) (h3 : B = 159) : C = 145 :=
by
  sorry

end carl_weight_l193_193665


namespace find_m_l193_193619

-- Given conditions
variable (U : Set ℕ) (A : Set ℕ) (m : ℕ)
variable (hU : U = {1, 2, 3, 4})
variable (hA : A = { x ∈ U | x^2 - 5 * x + m = 0 })
variable (hCUA : U \ A = {1, 4})

-- Prove that m = 6
theorem find_m (U A : Set ℕ) (m : ℕ) 
               (hU : U = {1, 2, 3, 4}) 
               (hA : A = { x ∈ U | x^2 - 5 * x + m = 0 }) 
               (hCUA : U \ A = {1, 4}) : 
  m = 6 := 
sorry

end find_m_l193_193619


namespace car_cost_difference_l193_193947

-- Definitions based on the problem's conditions
def car_cost_ratio (C A : ℝ) := C / A = 3 / 2
def ac_cost := 1500

-- Theorem statement that needs proving
theorem car_cost_difference (C A : ℝ) (h1 : car_cost_ratio C A) (h2 : A = ac_cost) : C - A = 750 := 
by sorry

end car_cost_difference_l193_193947


namespace geom_mean_points_exist_l193_193350

-- Definitions for points, segments, and triangle
structure Triangle := 
  (A B C : Point)

noncomputable def exists_points_on_segment_bc_geom_mean 
  (T : Triangle) 
  (D D' D'' : Point)
  (is_on_segment : D ∈ segment T.B T.C)
  (angle_A : ℝ) 
  : Prop := 
  ∃ (D' D'') ∈ segment T.B T.C, 
  if angle_A > π / 2 then 
    (T.AD^2 = T.BD * T.DC ∧ T.AD' = T.BD' * T.DC') ∧ 
    (T.AD'' = T.BD'' * T.DC'')
  else if angle_A = π / 2 then 
    (T.AD = T.BD * T.DC)
  else
    false

-- Lean statement declaration
theorem geom_mean_points_exist (T : Triangle) (D D' D'' : Point)
  (is_on_segment : D ∈ segment T.B T.C)
  (angle_A : ℝ) 
  : exists_points_on_segment_bc_geom_mean T D D' D'' is_on_segment angle_A :=
sorry

end geom_mean_points_exist_l193_193350


namespace percentage_apples_is_50_percent_l193_193582

-- Definitions for the given conditions
def initial_apples : ℕ := 10
def initial_oranges : ℕ := 5
def added_oranges : ℕ := 5

-- Defining the proof problem
theorem percentage_apples_is_50_percent :
  let total_fruits := initial_apples + initial_oranges + added_oranges in
  let apples_percentage := (initial_apples * 100) / total_fruits in
  apples_percentage = 50 :=
by
  sorry

end percentage_apples_is_50_percent_l193_193582


namespace average_score_is_7_standard_deviation_is_2_l193_193002

def scores : List ℕ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]

def average_score : ℕ :=
  scores.sum / scores.length

def variance : ℕ :=
  let mean := average_score
  (scores.map (λ x => (x - mean) * (x - mean))).sum / scores.length

def standard_deviation : ℕ :=
  Int.sqrt variance

theorem average_score_is_7 :
  average_score = 7 :=
by
  sorry

theorem standard_deviation_is_2 :
  standard_deviation = 2 :=
by
  sorry

end average_score_is_7_standard_deviation_is_2_l193_193002


namespace probability_sum_dice_eq_5_l193_193603

theorem probability_sum_dice_eq_5 : 
  let outcomes := [(1, 4), (2, 3), (3, 2), (4, 1)] in
  let total_outcomes := 36 in
  (length outcomes) / total_outcomes = 1 / 9 :=
by 
  sorry

end probability_sum_dice_eq_5_l193_193603


namespace sufficient_but_not_necessary_condition_l193_193398

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x|

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a < 0 → ∀ x ≥ 0, f a x = 2 * x - a ∧ monotone (f a)) ∧
  (∀ a ≥ 0, ∀ x ≥ 0, f a x > 0) :=
by sorry

end sufficient_but_not_necessary_condition_l193_193398


namespace josie_animal_counts_l193_193471

/-- Josie counted 80 antelopes, 34 more rabbits than antelopes, 42 fewer hyenas than 
the total number of antelopes and rabbits combined, some more wild dogs than hyenas, 
and the number of leopards was half the number of rabbits. The total number of animals 
Josie counted was 605. Prove that the difference between the number of wild dogs 
and hyenas Josie counted is 50. -/
theorem josie_animal_counts :
  ∃ (antelopes rabbits hyenas wild_dogs leopards : ℕ),
    antelopes = 80 ∧
    rabbits = antelopes + 34 ∧
    hyenas = (antelopes + rabbits) - 42 ∧
    leopards = rabbits / 2 ∧
    (antelopes + rabbits + hyenas + wild_dogs + leopards = 605) ∧
    wild_dogs - hyenas = 50 := 
by
  sorry

end josie_animal_counts_l193_193471


namespace quadratic_roots_l193_193543

theorem quadratic_roots : 
  ∀ (b c : ℝ), 
  polynomial.roots (polynomial.mk [c, b, (2 / real.sqrt 3)]) = {(1 / 2), (3 / 2)} ∧ 
  ∃ K L M : ℝ, 
  K L = K M ∧ 
  ∠LKM = 120 := 
sorry

end quadratic_roots_l193_193543


namespace opposite_of_two_reciprocal_of_negative_five_smallest_absolute_value_l193_193552

-- Define the concept of opposite for any number
def opposite (n : ℝ) : ℝ := -n

-- Define the concept of reciprocal for any number
def reciprocal (n : ℝ) : ℝ := 1 / n

-- Prove the given statements
theorem opposite_of_two : opposite 2 = -2 :=
by
  unfold opposite
  exact rfl

theorem reciprocal_of_negative_five : reciprocal (-5) = -1/5 :=
by
  unfold reciprocal
  norm_num

theorem smallest_absolute_value : ∀ (x : ℝ), x = 2 ∨ x = -2 ∨ x = -1/5 → |x| ≥ 0 := 
by
  intro x
  intro h
  apply abs_nonneg

end opposite_of_two_reciprocal_of_negative_five_smallest_absolute_value_l193_193552


namespace total_course_selection_schemes_l193_193256

theorem total_course_selection_schemes (PE_courses : ℕ) (Art_courses : ℕ) : 
  PE_courses = 4 → Art_courses = 4 → 
  (finset.card (finset.powerset_len 2 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b ∈ s), a < PE_courses ∧ b ≥ PE_courses) + 
   finset.card (finset.powerset_len 3 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b c ∈ s), (a < PE_courses ∧ b < PE_courses ∧ c ≥ PE_courses) ∨ 
                        (a < PE_courses ∧ b ≥ PE_courses ∧ c ≥ PE_courses ∧ a ≠ b ≠ c)) = 64 :=
by
  sorry

end total_course_selection_schemes_l193_193256


namespace find_K_in_sphere_volume_l193_193508

theorem find_K_in_sphere_volume :
  let s := 3
  let tetrahedron_surface_area := 4 * (s^2 * Real.sqrt 3 / 4)
  let sphere_surface_area := 4 * Real.pi * r^2
  let r := Real.sqrt (tetrahedron_surface_area / (4 * Real.pi))
  let sphere_volume := (4 / 3) * Real.pi * r^3
  let K := 27  -- This statement is inferred from the correct answer step
  in
  sphere_volume = (K * Real.sqrt 2) / Real.sqrt Real.pi :=
by
  sorry

end find_K_in_sphere_volume_l193_193508


namespace length_AE_l193_193096

-- Definitions
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (6, 0)
def C : ℝ × ℝ := (3, 3)
def D : ℝ × ℝ := (5, 0)
def E : ℝ × ℝ := (4, approximately the y-coordinate intersection) -- we acknowledge that the exact coordinates would require solving the intersection, but we'll generalize for this statement.

def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

-- Theorem
theorem length_AE :
  distance A E = (2 * (real.sqrt 13 + real.sqrt 10) * real.sqrt 13) / real.sqrt 41 :=
sorry

end length_AE_l193_193096


namespace quadratic_general_form_l193_193342

theorem quadratic_general_form (x : ℝ) :
  x * (x + 2) = 5 * (x - 2) → x^2 - 3 * x - 10 = 0 := by
  sorry

end quadratic_general_form_l193_193342


namespace perfect_square_n_fours_l193_193701

def number_with_n_fours (n : ℕ) : ℕ :=
  let base := 10^(n-1)
  in 144 * base + 4 * (base - 1) / 9

theorem perfect_square_n_fours (n : ℕ) :
  (∃ k : ℕ, k^2 = number_with_n_fours n) ↔ n = 2 ∨ n = 3 :=
by
  sorry

end perfect_square_n_fours_l193_193701


namespace inclination_angle_of_straight_line_l193_193402

theorem inclination_angle_of_straight_line (α : ℝ) (a b : ℝ) (hα : b / a = 1 / real.sqrt 3)
  (h_perpendicular : a - real.sqrt 3 * b = 0) :
  α = real.arctan (1 / real.sqrt 3) :=
sorry

end inclination_angle_of_straight_line_l193_193402


namespace f_neg_2017_l193_193761

-- Definition of the function f
def f (x : ℝ) : ℝ :=
  ((x + 1)^2 + log (sqrt (1 + 9 * x^2) - 3 * x) * cos x) / (x^2 + 1)

-- The given condition
axiom f_at_2017 : f 2017 = 2016

-- The theorem to prove
theorem f_neg_2017 : f (-2017) = -2014 :=
by
  sorry

end f_neg_2017_l193_193761


namespace focal_length_of_ellipse_l193_193448

-- Define the right-angled triangle ABC with given lengths
def TriangleABC (A B C : ℝ × ℝ) : Prop :=
  (B.fst - A.fst)^2 + (B.snd - A.snd)^2 = 1 ∧
  (C.fst - A.fst)^2 + (C.snd - A.snd)^2 = 1 ∧
  (B.fst - C.fst)^2 + (B.snd - C.snd)^2 = 1 ∧
  (B.fst - A.fst) * (C.fst - A.fst) + (B.snd - A.snd) * (C.snd - A.snd) = 0

-- Define that an ellipse exists with foci C and another point F on AB
def EllipseWithFoci (A B C F : ℝ × ℝ) : Prop :=
  ∃a b c, 
  (a^2 = b^2 + c^2) ∧
  ∀(P : ℝ × ℝ), 
  ((P = A ∨ P = B) → (dist P C + dist P F = 2 * a))

-- C is one of the foci
def CFoci (C : ℝ × ℝ) : Prop :=
  C = (1, 0)

-- F lies on line segment AB
def FOnAB (A B F : ℝ × ℝ) : Prop :=
  ∃k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ F = (k, 0)

-- Prove the focal length is √5 / 2
theorem focal_length_of_ellipse :
  ∀ (A B C F : ℝ × ℝ),
    TriangleABC A B C →
    CFoci C →
    FOnAB A B F →
    EllipseWithFoci A B C F →
    dist F C = (real.sqrt 5) / 2 :=
by
  intros A B C F hABC hCFoci hFOnAB hEllipse
  sorry

end focal_length_of_ellipse_l193_193448


namespace course_selection_count_l193_193218

-- Definitions for the conditions
def num_PE_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_PE_courses + num_art_courses

-- The number of different course selection schemes
theorem course_selection_count : 
  (choose num_PE_courses 1) * (choose num_art_courses 1) + 
  (choose num_PE_courses 2) * (choose num_art_courses 1) + 
  (choose num_PE_courses 1) * (choose num_art_courses 2) = 64 := 
by sorry

end course_selection_count_l193_193218


namespace sum_of_six_smallest_n_l193_193855

noncomputable def τ(n : ℕ) : ℕ :=
  if h : n = 0 then 0 else (Finset.range (n+1)).filter (λ d, d > 0 ∧ n % d = 0).card

theorem sum_of_six_smallest_n :
  let nums := {n : ℕ | τ(n) + τ(n + 1) = 8}
  (nums.filter (λ n, ¬ ∃ m, m < n ∧ m ∈ nums)).sum.take 6 = 68 := by
  sorry

end sum_of_six_smallest_n_l193_193855


namespace percentage_apples_basket_l193_193584

theorem percentage_apples_basket :
  let initial_apples := 10
  let initial_oranges := 5
  let added_oranges := 5
  let total_apples := initial_apples
  let total_oranges := initial_oranges + added_oranges
  let total_fruits := total_apples + total_oranges
  (total_apples / total_fruits) * 100 = 50 :=
by
  sorry

end percentage_apples_basket_l193_193584


namespace tangent_curve_line_l193_193369

theorem tangent_curve_line {b : ℝ} :
  (∃ m n : ℝ, (n = m^4) ∧ (4m + b = n) ∧ (4 = 4 * m^3)) → b = -3 :=
by
  intros h,
  sorry

end tangent_curve_line_l193_193369


namespace probability_Xiao_Yan_before_Xiao_Ming_l193_193609

theorem probability_Xiao_Yan_before_Xiao_Ming {Ω : Type} [fintype Ω] [probability_space Ω] 
  (Xiao_Jun Xiao_Yan Xiao_Ming : Ω) 
  (h : ∀ x : Ω, probability (ℙ (Xiao_Yan = x) = ℙ (Xiao_Ming = x))) :
  ℙ (Xiao_Yan < Xiao_Ming) = 1 / 2 := 
sorry

end probability_Xiao_Yan_before_Xiao_Ming_l193_193609


namespace shaded_area_l193_193452

-- Let A be the length of the side of the smaller square
def A : ℝ := 4

-- Let B be the length of the side of the larger square
def B : ℝ := 12

-- The problem is to prove that the area of the shaded region is 10 square inches
theorem shaded_area (A B : ℝ) (hA : A = 4) (hB : B = 12) :
  (A * A) - (1/2 * (B / (B + A)) * A * B) = 10 := by
  sorry

end shaded_area_l193_193452


namespace clock_angle_at_3_15_l193_193600

-- Conditions
def full_circle_degrees : ℕ := 360
def hour_degree : ℕ := full_circle_degrees / 12
def minute_degree : ℕ := full_circle_degrees / 60
def minute_position (m : ℕ) : ℕ := m * minute_degree
def hour_position (h m : ℕ) : ℕ := h * hour_degree + m * (hour_degree / 60)

-- Theorem to prove
theorem clock_angle_at_3_15 : (|minute_position 15 - hour_position 3 15| : ℚ) = 7.5 := by
  sorry

end clock_angle_at_3_15_l193_193600


namespace original_number_is_two_over_three_l193_193063

theorem original_number_is_two_over_three (x : ℚ) (h : 1 + 1/x = 5/2) : x = 2/3 :=
sorry

end original_number_is_two_over_three_l193_193063


namespace average_within_bounds_l193_193943

theorem average_within_bounds (N : ℝ) (h₁ : 15 < N) (h₂ : N < 25) :
    (∃ x ∈ {12, 15, 18, 20, 23}, (8 + 14 + N) / 3 = x) := by
  use 15
  have : (8 + 14 + N) / 3 = 15 := by sorry
  exact this

end average_within_bounds_l193_193943


namespace total_number_of_course_selection_schemes_l193_193251

-- Define the total number of courses
def total_courses := 8

-- Define number of physical education and art courses
def pe_courses := 4
def art_courses := 4

-- Define selections: students choose 2 or 3 courses
def course_selections : Finset (Finset ℕ) :=
  (Finset.powerset (Finset.range total_courses)).filter (λ s, s.card = 2 ∨ s.card = 3)

-- Define condition: at least 1 course from each category
def valid_selections : Finset (Finset ℕ) :=
  course_selections.filter (λ s, ∃ pe art, s = pe ∪ art ∧ pe.card ≠ 0 ∧ art.card ≠ 0 ∧ 
                             pe ⊆ Finset.range pe_courses ∧ art ⊆ (Finset.range total_courses).filter (λ x, x ≥ pe_courses))

theorem total_number_of_course_selection_schemes : valid_selections.card = 64 := 
by sorry

end total_number_of_course_selection_schemes_l193_193251


namespace choco_delight_remainder_l193_193352

theorem choco_delight_remainder (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 := 
by 
  sorry

end choco_delight_remainder_l193_193352


namespace fraction_equivalence_l193_193667

theorem fraction_equivalence : 
  (\dfrac{1}{4} - \dfrac{1}{5}) / (\dfrac{1}{3} - \dfrac{1}{6}) = \dfrac{3}{10} :=
by 
  sorry

end fraction_equivalence_l193_193667


namespace consecutive_pair_probability_l193_193518

theorem consecutive_pair_probability :
  let balls := ({1, 2, 3, 4, 5, 6} : Finset ℕ),
      total_selections := balls.choose 3,
      consecutive_pairs := (λ (s : Finset ℕ), s.card = 3 ∧ (∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                                       s = {a, b, c} ∧ ((a + 1 = b ∨ b + 1 = a) ∧
                                                        ¬ (c + 1 = a ∨ a + 1 = c) ∧
                                                        ¬ (c + 1 = b ∨ b + 1 = c)))) in
  ∃ num_favorable : ℕ,
    5 * 4 - 4 = 12 → -- This represents the corrected number of favorable outcomes
    let m := num_favorable,
        n := total_selections.card in
    m = 12 →
    (m / n : ℚ) = 3 / 5 :=
by
  sorry

end consecutive_pair_probability_l193_193518


namespace tangent_line_value_l193_193756

variables {x1 x2 : ℝ}

def f (x : ℝ) : ℝ := Real.log x
def g (x : ℝ) : ℝ := Real.exp x

-- Conditions stating that the line is tangent to both curves at the given points
def tangent_to_f (x1 : ℝ) : Prop := 
  ∀ x y, y = (1 / x1) * x - (1 - Real.log x1) → y = f x

def tangent_to_g (x2 : ℝ) : Prop := 
  ∀ x y, y = Real.exp x2 * x - Real.exp x2 * (x2 - 1) → y = g x

theorem tangent_line_value (h_tangent_f : tangent_to_f x1) (h_tangent_g : tangent_to_g x2) :
  (1 / x1) - (2 / (x2 - 1)) = 1 :=
sorry

end tangent_line_value_l193_193756


namespace find_a_l193_193405

noncomputable def chord_intercepted_by_circle (a : ℝ) : Prop :=
  let c := (a, 0)
  let r := 2
  let line := abs (a - 2) / real.sqrt 2
  (line ^ 2 + 2 = r ^ 2) ∧ (2 ^ 2 = 4)

theorem find_a (a : ℝ) : chord_intercepted_by_circle a → (a = 0 ∨ a = 4) :=
by
  intros
  sorry

end find_a_l193_193405


namespace new_average_production_l193_193715

-- Define the conditions
def average_production_past_n_days (n : ℕ) : ℕ := 40
def today_production : ℕ := 90
def n_value : ℕ := 9

-- Define the goal
theorem new_average_production :
  let total_past_production := average_production_past_n_days n_value * n_value,
      total_production := total_past_production + today_production,
      total_days := n_value + 1 
  in total_production / total_days = 45 :=
by
  -- Proof skipped
  sorry

end new_average_production_l193_193715


namespace ratio_area_of_rectangle_to_square_l193_193930

theorem ratio_area_of_rectangle_to_square (s : ℝ) :
  (1.2 * s * 0.8 * s) / (s * s) = 24 / 25 :=
by
  sorry

end ratio_area_of_rectangle_to_square_l193_193930


namespace angle_BCA_is_90_degrees_l193_193039

-- Define the given objects and conditions
variables {A B C T1 T2 I O: Type}
variables [Triangle A B C] [Excircle B C T1] [Excircle A C T2] [Incenter ABC I]
variables (M : Midpoint A B) (S : Symmetric I M O)

-- Define the required condition for the proof problem
theorem angle_BCA_is_90_degrees (h : lies_on_circumcircle O C T1 T2) :
  ∠ B C A = 90 :=
sorry

end angle_BCA_is_90_degrees_l193_193039


namespace find_product_of_integers_l193_193125

theorem find_product_of_integers
  (a b c d e : ℤ)
  (h_sets : Multiset (a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e) =
             {-1, 4, 6, 9, 10, 11, 15, 16, 20, 22}) :
  a * b * c * d * e = -4914 :=
sorry

end find_product_of_integers_l193_193125


namespace option_A_option_B_option_C_option_D_l193_193741

def dot_product (v w : ℝ × ℝ) := v.1 * w.1 + v.2 * w.2

def norm (w : ℝ × ℝ) := real.sqrt (w.1 * w.1 + w.2 * w.2)

def projection (v w : ℝ × ℝ) := ((dot_product v w) / (norm w)^2) * w

variables (t : ℝ)
def a := (-2, 1)
def b := (4, 2)
def c := (2, t)

-- For option A
theorem option_A : dot_product b c ≠ 0 → t ≠ 4 := by
  sorry

-- For option B
theorem option_B : a.1 / c.1 = a.2 / c.2 → t = -1 := by
  sorry

-- For option C
theorem option_C : t = 1 → projection a c = (-3 / 5) * c := by
  sorry

-- For option D
theorem option_D : t > -4 → ¬(dot_product b c > 0) → ¬(norm b * norm c * real.cos (dot_product b c) = 0) := by
  sorry

end option_A_option_B_option_C_option_D_l193_193741


namespace comparison_of_exponents_and_logs_l193_193114

noncomputable section

def exp1 := 7 ^ 0.3
def exp2 := 0.3 ^ 7
def log3 := Real.logBase 3 0.7

theorem comparison_of_exponents_and_logs : log3 < exp2 ∧ exp2 < exp1 := by
  sorry

end comparison_of_exponents_and_logs_l193_193114


namespace sequence_sum_problem_l193_193873

theorem sequence_sum_problem (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : ∀ n, S n = 2 * a n - n) :
  (2 / (a 1 * a 2) + 4 / (a 2 * a 3) + 8 / (a 3 * a 4) + 16 / (a 4 * a 5) : ℚ) = 30 / 31 := 
sorry

end sequence_sum_problem_l193_193873


namespace position_function_correct_l193_193627

-- Define the velocity function
def velocity (t : ℝ) : ℝ := 3 * t^2 - 1

-- Define the position function
def position (t : ℝ) : ℝ := t^3 - t + 0.05

-- Define the initial condition
def initial_position : ℝ := 0.05

-- State the theorem
theorem position_function_correct :
  (∀ t : ℝ, deriv (position t) = velocity t) ∧ (position 0 = initial_position) :=
by
  sorry

end position_function_correct_l193_193627


namespace parabola_point_distance_l193_193298

theorem parabola_point_distance (x y : ℝ) (hM_on_parabola : y = 4 * x^2) (h_dist_focus : sqrt ((x - 0)^2 + (y - 1/4)^2) = 1) : y = 15/16 := 
  sorry

end parabola_point_distance_l193_193298


namespace max_sum_n_l193_193852

variable {a : ℕ → ℝ} {S : ℕ → ℝ}
variable (d a1 : ℝ)

-- Definitions based on the conditions
def a_n (n : ℕ) := a1 + (n - 1) * d
def S_n (n : ℕ) := n * a1 + (n * (n - 1) / 2) * d

-- Conditions
def condition1 : Prop := a_n 5 > 0
def condition2 : Prop := a_n 1 + a_n 10 < 0

-- Statement to prove
theorem max_sum_n (h1 : condition1) (h2 : condition2) : (∃ n, S_n n = S_n 5) :=
sorry

end max_sum_n_l193_193852


namespace rectangle_to_square_area_ratio_is_24_25_l193_193921

noncomputable def rectangle_to_square_area_ratio
  (s : ℝ) -- length of side of square S
  (longer_side : ℝ := 1.2 * s) -- longer side of rectangle R
  (shorter_side : ℝ := 0.8 * s) -- shorter side of rectangle R
  (area_R : ℝ := longer_side * shorter_side) -- area of rectangle R
  (area_S : ℝ := s^2) -- area of square S
  : ℝ := 
  area_R / area_S

theorem rectangle_to_square_area_ratio_is_24_25 
  (s : ℝ)
  : rectangle_to_square_area_ratio s = 24 / 25 :=
by 
  sorry

end rectangle_to_square_area_ratio_is_24_25_l193_193921


namespace equation_of_plane_l193_193685

noncomputable def parametric_form (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2 * s - 3 * t, 4 - s + 2 * t, 1 - 3 * s - t)

theorem equation_of_plane (x y z : ℝ) : 
  (∃ s t : ℝ, parametric_form s t = (x, y, z)) → 5 * x + 11 * y + 7 * z - 61 = 0 :=
by
  sorry

end equation_of_plane_l193_193685


namespace factor_sum_l193_193788

theorem factor_sum (R S : ℝ) (h : ∃ (b c : ℝ), (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + R*x^2 + S) : R + S = 54 :=
sorry

end factor_sum_l193_193788


namespace problem_statement_l193_193858

noncomputable def even_increasing (f : ℝ → ℝ) :=
  ∀ x, f x = f (-x) ∧ ∀ x y, x < y → f x < f y

theorem problem_statement {f : ℝ → ℝ} (hf_even_incr : even_increasing f)
  (x1 x2 : ℝ) (hx1_gt_0 : x1 > 0) (hx2_lt_0 : x2 < 0) (hf_lt : f x1 < f x2) : x1 + x2 > 0 :=
sorry

end problem_statement_l193_193858


namespace vector_construction_l193_193432

variables {R : Type*} [AddCommGroup R] [Module ℝ R] 

structure Vec2 (R : Type*) :=
  (x : R)
  (y : R)

def vec_add (v1 v2 : Vec2 R) : Vec2 R :=
  ⟨v1.x + v2.x, v1.y + v2.y⟩

def vec_scale (r : ℝ) (v : Vec2 R) : Vec2 R :=
  ⟨r * v.x, r * v.y⟩

variables (a b c : Vec2 ℝ)

theorem vector_construction :
  vec_add (vec_add (vec_scale 3 a) (vec_scale (-2) b)) c = ⟨3 * a.x - 2 * b.x + c.x, 3 * a.y - 2 * b.y + c.y⟩ :=
sorry

end vector_construction_l193_193432


namespace largest_divisor_even_triplet_l193_193038

theorem largest_divisor_even_triplet :
  ∀ (n : ℕ), 24 ∣ (2 * n) * (2 * n + 2) * (2 * n + 4) :=
by intros; sorry

end largest_divisor_even_triplet_l193_193038


namespace monthly_vs_annual_interest_difference_l193_193498

-- Let P be the principal amount.
def P : ℝ := 8000

-- Let r be the annual interest rate.
def r : ℝ := 0.10

-- Let n be the number of years.
def n : ℕ := 3

-- Annual compounding formula
def annual_amount (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n

-- Monthly compounding formula, where m is the number of compounding periods per year.
def monthly_amount (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  let m : ℝ := 12
  P * (1 + r / m) ^ (n * m)

-- The difference between monthly and annual compounding amounts after 3 years.
def interest_difference : ℝ :=
  monthly_amount P r n - annual_amount P r n

-- Lean theorem statement to prove the interest difference is 111.12
theorem monthly_vs_annual_interest_difference :
  interest_difference = 111.12 := by
  sorry

end monthly_vs_annual_interest_difference_l193_193498


namespace smallest_possible_value_of_b_l193_193521

theorem smallest_possible_value_of_b (a b : ℝ) 
  (h1 : 2 < a)
  (h2 : a < b)
  (h3 : ¬(2 + a > b ∧ 2 + b > a ∧ a + b > 2))
  (h4 : ¬(\(1 / b\) + \(1 / a\) > \(1 / 2\) ∧ \(1 / a\) + \(1 / 2\) > \(1 / b\) ∧ \(1 / b\) + \(1 / 2\) > \(1 / a\))) : 
  b = 6 :=
sorry

end smallest_possible_value_of_b_l193_193521


namespace tens_digit_of_3_pow_205_l193_193149

theorem tens_digit_of_3_pow_205 : 
  let x := 3^20 in 
  let y := x^10 * 3^5 in
  (y % 100) / 10 = 4 := by
  sorry

end tens_digit_of_3_pow_205_l193_193149


namespace total_number_of_course_selection_schemes_l193_193250

-- Define the total number of courses
def total_courses := 8

-- Define number of physical education and art courses
def pe_courses := 4
def art_courses := 4

-- Define selections: students choose 2 or 3 courses
def course_selections : Finset (Finset ℕ) :=
  (Finset.powerset (Finset.range total_courses)).filter (λ s, s.card = 2 ∨ s.card = 3)

-- Define condition: at least 1 course from each category
def valid_selections : Finset (Finset ℕ) :=
  course_selections.filter (λ s, ∃ pe art, s = pe ∪ art ∧ pe.card ≠ 0 ∧ art.card ≠ 0 ∧ 
                             pe ⊆ Finset.range pe_courses ∧ art ⊆ (Finset.range total_courses).filter (λ x, x ≥ pe_courses))

theorem total_number_of_course_selection_schemes : valid_selections.card = 64 := 
by sorry

end total_number_of_course_selection_schemes_l193_193250


namespace fA_even_and_monotonically_increasing_fC_even_and_monotonically_increasing_l193_193651

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → x < y → f (x) < f (y)

def fA := λ x : ℝ, |x| - 1
def fC := λ x : ℝ, 2^x + 2^(-x)

theorem fA_even_and_monotonically_increasing :
  is_even_function fA ∧ is_monotonically_increasing fA :=
sorry

theorem fC_even_and_monotonically_increasing :
  is_even_function fC ∧ is_monotonically_increasing fC :=
sorry

end fA_even_and_monotonically_increasing_fC_even_and_monotonically_increasing_l193_193651


namespace odd_terms_in_binomial_expansion_l193_193787

theorem odd_terms_in_binomial_expansion (m n : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 1) :
  (finset.filter (λ k, (((m ^ (7 - k)) * (n ^ k)) % 2 = 1))
    (finset.range (7 + 1))).card = 8 :=
by sorry

end odd_terms_in_binomial_expansion_l193_193787


namespace range_f_is_real_l193_193886

noncomputable def f (x : ℝ) : ℝ := log (3 : ℝ) (-x)
noncomputable def g (x : ℝ) : ℝ := 3 ^ (-x)

theorem range_f_is_real : set.range f = set.univ :=
by
  sorry

end range_f_is_real_l193_193886


namespace a_2006_mod_100_l193_193866

noncomputable section

def a : ℕ → ℤ
| 0       := 21
| 1       := 35
| (n + 2) := 4 * a (n + 1) - 4 * a n + n^2

theorem a_2006_mod_100 : a 2006 % 100 = 0 := by
  sorry

end a_2006_mod_100_l193_193866


namespace convert_base_10_to_base_8_l193_193680

theorem convert_base_10_to_base_8 :
  ∀ (n : ℕ), n = 2450 → nat.to_digits 8 n = [4, 6, 2, 2] :=
by
  intros n h
  rw h
  exact nat.to_digits 8 2450 = [4, 6, 2, 2]
  sorry

end convert_base_10_to_base_8_l193_193680


namespace mrs_awesome_class_l193_193500

def num_students (b g : ℕ) : ℕ := b + g

theorem mrs_awesome_class (b g : ℕ) (h1 : b = g + 3) (h2 : 480 - (b * b + g * g) = 5) : num_students b g = 31 :=
by
  sorry

end mrs_awesome_class_l193_193500


namespace total_course_selection_schemes_l193_193264

theorem total_course_selection_schemes (PE_courses : ℕ) (Art_courses : ℕ) : 
  PE_courses = 4 → Art_courses = 4 → 
  (finset.card (finset.powerset_len 2 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b ∈ s), a < PE_courses ∧ b ≥ PE_courses) + 
   finset.card (finset.powerset_len 3 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b c ∈ s), (a < PE_courses ∧ b < PE_courses ∧ c ≥ PE_courses) ∨ 
                        (a < PE_courses ∧ b ≥ PE_courses ∧ c ≥ PE_courses ∧ a ≠ b ≠ c)) = 64 :=
by
  sorry

end total_course_selection_schemes_l193_193264


namespace infinite_series_eq_5_over_16_l193_193328

noncomputable def infinite_series_sum : ℝ :=
  ∑' (n : ℕ), (n + 1 : ℝ) / (5 ^ (n + 1))

theorem infinite_series_eq_5_over_16 :
  infinite_series_sum = 5 / 16 :=
sorry

end infinite_series_eq_5_over_16_l193_193328


namespace positive_solution_count_l193_193363

theorem positive_solution_count :
  ∀ x : ℝ, x > 0 → (cos (arcsin (tan (arccos x))) = x) → x = 1 :=
by
  sorry

end positive_solution_count_l193_193363


namespace problem1_problem2_l193_193719

-- Define the set S_n
def S_n (n : ℕ) : Set (Fin n → Bool) :=
  { A | ∀ i : Fin n, A i = true ∨ A i = false }

-- Define the distance function d(U, V)
def d {n : ℕ} (U V : Fin n → Bool) : ℕ :=
  Finset.card { i : Fin n | U i ≠ V i }

-- Problem 1: m = 15 for specific U and n = 6
theorem problem1 :
  let U : Fin 6 → Bool := λ _, true
  ∃ m, (m = Finset.card { V : Fin 6 → Bool | V ∈ S_n 6 ∧ d U V = 2 }) ∧ m = 15 :=
by
  let U : Fin 6 → Bool := λ _, true
  existsi 15
  sorry

-- Problem 2: Sum of distances
theorem problem2 (U : Fin n → Bool) :
  ∑ V in (S_n n).toFinset, d U V = n * 2^(n-1) :=
by
  sorry

end problem1_problem2_l193_193719


namespace expression_evaluation_l193_193141

theorem expression_evaluation : 4 * 10 + 5 * 11 + 12 * 4 + 4 * 9 = 179 :=
by
  sorry

end expression_evaluation_l193_193141


namespace salary_increase_needed_l193_193639

theorem salary_increase_needed (S : ℝ) (hS : S > 0) :
  let reduced_once := S * (1 - 0.2)
      reduced_twice := reduced_once * (1 - 0.1)
      after_tax := reduced_twice * (1 - 0.15)
      increased := after_tax * (1 + 0.634)
  in increased = S := 
by 
  sorry

end salary_increase_needed_l193_193639


namespace course_selection_schemes_l193_193200

theorem course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose (n k : ℕ) := nat.choose n k
  
  (choose 4 1 * choose 4 1 + 
   choose 4 2 * choose 4 1 + 
   choose 4 1 * choose 4 2) = 64 := 
by {
  -- calculating the combinations
  have comb_2_courses : choose 4 1 * choose 4 1 = 16 := by sorry,
  have comb_2_pe_1_art : choose 4 2 * choose 4 1 = 24 := by sorry,
  have comb_1_pe_2_art : choose 4 1 * choose 4 2 = 24 := by sorry,
  have total_comb : 16 + 24 + 24 = 64 := by {
    simp [comb_2_courses, comb_2_pe_1_art, comb_1_pe_2_art]
  },
  exact total_comb
}

end course_selection_schemes_l193_193200


namespace diana_can_paint_statues_l193_193614

theorem diana_can_paint_statues : (3 / 6) / (1 / 6) = 3 := 
by 
  sorry

end diana_can_paint_statues_l193_193614


namespace janet_wait_time_l193_193836

theorem janet_wait_time 
  (janet_speed : ℝ)
  (sister_speed : ℝ)
  (lake_width : ℝ)
  (h1 : janet_speed = 30)
  (h2 : sister_speed = 12)
  (h3 : lake_width = 60) 
  :
  60 / 12 - 60 / 30 = 3 :=
by
  sorry

end janet_wait_time_l193_193836


namespace right_triangle_at_angle_ABC_l193_193372

-- Definitions of the conditions
variables {A B C D E F : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
variables (AD CA AB BE : ℝ) (ABC : Triangle A B C) (F_midpoint : midPoint B C F)
variables (extension_CA : extendSegment CA AD D) (extension_AB : extendSegment AB BE E)
variables (BD EF : ℝ)
variables (BD_eq : BD = 2 * EF)

-- Statement of the proof
theorem right_triangle_at_angle_ABC :
  (BD = 2 * EF) → ∠ABC = 90 := 
  sorry

end right_triangle_at_angle_ABC_l193_193372


namespace sequence_sum_theorem_l193_193687

noncomputable def sequence_sum (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 4 - 3 * n * (n + 1) / 4 + 2 * n

theorem sequence_sum_theorem (n : ℕ) :
  (∑ k in Finset.range (n + 1), (match k with
      | 0 => 2
      | 1 => 5
      | 2 => 11
      | _ => 1.5 * k^2 - 1.5 * k + 2) : ℝ) = sequence_sum n := 
  by
  sorry

end sequence_sum_theorem_l193_193687


namespace radius_of_circle_spherical_coordinates_l193_193347

theorem radius_of_circle_spherical_coordinates :
  (∃ θ : ℝ, ∀ θ (r : ℝ), 
    ∃ (ρ θ φ : ℝ) (x y : ℝ),
    ρ = 2 ∧ φ = π / 4 ∧
    x = ρ * sin φ * cos θ ∧
    y = ρ * sin φ * sin θ ∧
    x^2 + y^2 = r^2) → r = sqrt 2 :=
by
  -- Proof skipped
  sorry

end radius_of_circle_spherical_coordinates_l193_193347


namespace work_completion_time_l193_193985

theorem work_completion_time
  (A B C : ℝ)
  (h1 : A + B = 1 / 12)
  (h2 : B + C = 1 / 15)
  (h3 : C + A = 1 / 20) :
  1 / (A + B + C) = 10 :=
by
  sorry

end work_completion_time_l193_193985


namespace bad_carrots_count_l193_193995

theorem bad_carrots_count (n : ℕ) (m : ℕ) (g : ℕ) (bad : ℕ) 
  (hn : n = 38) (hm : m = 47) (hg : g = 71) 
  (htotal : n + m = 85) 
  (hbad : bad = (n + m) - g) : bad = 14 := 
by 
  rw [hn, hm, hg, htotal] at hbad;
  linarith

end bad_carrots_count_l193_193995


namespace neg_p_true_l193_193421

theorem neg_p_true :
  (∃ x : ℝ, x^2 ≤ 0) :=
sorry

end neg_p_true_l193_193421


namespace area_of_parallelogram_l193_193988

theorem area_of_parallelogram (base height : ℝ) (h_base : base = 18) (h_height : height = 16) : 
  base * height = 288 := 
by
  sorry

end area_of_parallelogram_l193_193988


namespace square_area_eq_1296_l193_193312

theorem square_area_eq_1296 (x : ℝ) (side : ℝ) (h1 : side = 6 * x - 18) (h2 : side = 3 * x + 9) : side ^ 2 = 1296 := sorry

end square_area_eq_1296_l193_193312


namespace area_ratio_rect_sq_l193_193926

variable (s : ℝ)

def side_len_sq (S : ℝ) : Prop := s = S
def longer_side_rect (R : ℝ) : Prop := R = 1.2 * s
def shorter_side_rect (R : ℝ) : Prop := R = 0.8 * s
def area_sq (S : ℝ) : ℝ := S * S
def area_rect (R_long R_short : ℝ) : ℝ := R_long * R_short
def ratio_area (areaR areaS : ℝ) : ℝ := areaR / areaS

theorem area_ratio_rect_sq (s S R_long R_short : ℝ) (h1 : side_len_sq s S) (h2 : longer_side_rect s R_long) (h3 : shorter_side_rect s R_short) :
  ratio_area (area_rect R_long R_short) (area_sq S) = 24/25 :=
by
  sorry

end area_ratio_rect_sq_l193_193926


namespace problem_A_problem_B_problem_C_problem_D_l193_193982

-- Proof Problem 1
theorem problem_A (n : ℕ) (p : ℚ) (X : ℕ →. ℚ) (h1 : X = (λ n, n * p)) (h2 : (λ n, n * p * (1 - p)) = 20) : p = 1 / 3 :=
sorry

-- Proof Problem 2
theorem problem_B (data : list ℕ) (sorted_data := list.sorted data) (k := 5) (percentile := list.nth sorted_data (k - 1)) (h1 : data = [91, 72, 75, 85, 64, 92, 76, 78, 86, 79]) (h2 : percentile = some 78) : true :=
sorry

-- Proof Problem 3
theorem problem_C (ξ : ℝ →. ℝ) (p : ℝ) (h1 : ξ = (λ 0 1), 1 - p) : (P (-1 ≤ ξ ≤ 0) = 1 / 2 - p) :=
sorry

-- Proof Problem 4
theorem problem_D (students11 students12 total : ℕ) (selected11 selected12 : ℕ) (h1 : students11 = 400) (h2 : students12 = 360) (h3 : total = 57) (h4 : selected11 = 20) (sampled_ratio := selected11 / students11) (h5 : sampled_ratio = selected12 / students12) : selected12 = 18 :=
sorry

end problem_A_problem_B_problem_C_problem_D_l193_193982


namespace six_points_monochromatic_triangle_l193_193821

theorem six_points_monochromatic_triangle (points : Fin 6 → Pnt) (black_or_red : ∀ (i j : Fin 6), i ≠ j → color) :
  (∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ black_or_red i j = black_or_red j k ∧ black_or_red j k = black_or_red k i) :=
begin
  sorry
end

end six_points_monochromatic_triangle_l193_193821


namespace cos_x_minus_pi_over_3_l193_193749

/-- Given x in (0, π) and cos(x - π/6) = -√3/3, prove that cos(x - π/3) = (-3 + √6)/6. -/
theorem cos_x_minus_pi_over_3 {x : ℝ} 
  (h₁ : 0 < x ∧ x < π) 
  (h₂ : Real.cos (x - π / 6) = -√3 / 3) : 
  Real.cos (x - π / 3) = (-3 + √6) / 6 :=
sorry

end cos_x_minus_pi_over_3_l193_193749


namespace expression_in_scientific_notation_l193_193084

-- Conditions
def billion : ℝ := 10^9
def a : ℝ := 20.8

-- Statement
theorem expression_in_scientific_notation : a * billion = 2.08 * 10^10 := by
  sorry

end expression_in_scientific_notation_l193_193084


namespace smallest_N_value_l193_193042

noncomputable def N_minimum (a b c d e : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) (h_sum : a + b + c + d + e = 2510) : ℕ :=
  max (max (a + b) (max (b + c) (max (c + d) (d + e))))

theorem smallest_N_value (a b c d e : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) 
  (h_sum : a + b + c + d + e = 2510) : N_minimum a b c d e h_pos h_sum = 1255 :=
sorry

end smallest_N_value_l193_193042


namespace probability_of_three_digit_number_div_by_three_l193_193577

noncomputable def probability_three_digit_div_by_three : ℚ :=
  let digit_mod3_groups := 
    {rem0 := {3, 6, 9}, rem1 := {1, 4, 7}, rem2 := {2, 5, 8}} in
  let valid_combinations :=
    (finset.card (finset.powersetLen 3 digit_mod3_groups.rem0) +
     (finset.card digit_mod3_groups.rem0 * finset.card digit_mod3_groups.rem1 * finset.card digit_mod3_groups.rem2) +
     finset.card (finset.powersetLen 3 digit_mod3_groups.rem1) +
     finset.card (finset.powersetLen 3 digit_mod3_groups.rem2))
  in
  let total_combinations := finset.card (finset.powersetLen 3 (finset.univ : finset (fin 9))) in
  (valid_combinations : ℚ) / total_combinations

theorem probability_of_three_digit_number_div_by_three :
  probability_three_digit_div_by_three = 5 / 14 := by
  -- provide proof here
  sorry

end probability_of_three_digit_number_div_by_three_l193_193577


namespace prob_divisible_by_3_of_three_digits_l193_193572

-- Define the set of digits available
def digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Three digits are to be chosen from this set
def choose_three_digits (s : Finset ℕ) : Finset (Finset ℕ) := s.powerset.filter (λ x, x.card = 3)

-- Define the property of the sum of digits being divisible by 3
def divisible_by_3 (s : Finset ℕ) : Prop := s.sum id % 3 = 0

-- Total combinations of choosing 3 out of 9 digits
def total_combinations : ℕ := (digits.card.choose 3)

-- Valid combinations where sum of digits is divisible by 3
def valid_combinations : Finset (Finset ℕ) := (choose_three_digits digits).filter divisible_by_3

-- Finally, the probability of a three-digit number being divisible by 3
def probability : ℕ × ℕ := (valid_combinations.card, total_combinations)

theorem prob_divisible_by_3_of_three_digits :
  probability = (5, 14) :=
by
  -- Proof to be filled
  sorry

end prob_divisible_by_3_of_three_digits_l193_193572


namespace addition_base10_to_base5_l193_193150

theorem addition_base10_to_base5 :
  ∀ (n m : ℕ), (n = 47 ∧ m = 58) → (n + m = 105) ∧ (105_to_base_5 = 410) :=
by
  sorry

end addition_base10_to_base5_l193_193150


namespace last_two_digits_of_9_power_h_are_21_l193_193966

def a := 1
def b := 2^a
def c := 3^b
def d := 4^c
def e := 5^d
def f := 6^e
def g := 7^f
def h := 8^g

theorem last_two_digits_of_9_power_h_are_21 : (9^h) % 100 = 21 := by
  sorry

end last_two_digits_of_9_power_h_are_21_l193_193966


namespace smallest_number_of_three_l193_193589

theorem smallest_number_of_three (a b c : ℕ) (h1 : a + b + c = 78) (h2 : b = 27) (h3 : c = b + 5) :
  a = 19 :=
by
  sorry

end smallest_number_of_three_l193_193589


namespace bridget_bakery_profit_l193_193319

theorem bridget_bakery_profit :
  let loaves_baked := 60
  let cost_per_loaf := 0.80
  let morning_sold := loaves_baked / 3
  let morning_revenue := morning_sold * 3
  let loaves_remaining_after_morning := loaves_baked - morning_sold
  let afternoon_sold := loaves_remaining_after_morning * 3 / 4
  let afternoon_revenue := afternoon_sold * 2
  let loaves_remaining_after_afternoon := loaves_remaining_after_morning - afternoon_sold
  let late_afternoon_revenue := loaves_remaining_after_afternoon * 1.50
  let total_revenue := morning_revenue + afternoon_revenue + late_afternoon_revenue
  let total_cost := loaves_baked * cost_per_loaf
  let profit := total_revenue - total_cost
  in
  profit = 87 :=
by
  sorry

end bridget_bakery_profit_l193_193319


namespace area_of_square_l193_193878

-- Define the properties of the square and the points E and F
variables (A B C D E F : Type)
variable [metric_space A]
variable [metric_space B]
variable [metric_space C]
variable [metric_space D]
variable [metric_space E]
variable [metric_space F]

-- Given conditions
variables (a b c d : ℝ) (h₁: a = 30) (h₂: b = 30) (h₃: c = 30)

-- Define the square and the points
def is_square (x : ℝ) : Prop := a = b ∧ b = c ∧ c = d ∧ d = a

-- Define the theorem to prove the area of the square
theorem area_of_square (side_length : ℝ) 
  (h_square : is_square side_length)
  (h_BE : side_length = 30)
  (h_EF : side_length = 30)
  (h_FD : side_length = 30) :
  side_length ^ 2 = 810 :=
by 
  sorry

end area_of_square_l193_193878


namespace largest_expression_l193_193373

noncomputable def f (x : ℝ) : ℝ := (x + 1) * (x - 1)
noncomputable def g (x : ℝ) : ℝ := (x + 0.5) * (x - 0.5)
noncomputable def h (x : ℝ) : ℝ := (x + 1/3) * (x - 1/3)
noncomputable def k (x : ℝ) : ℝ := (x + 0.25) * (x - 0.25)

theorem largest_expression (x : ℝ) : k(x) ≥ f(x) ∧ k(x) ≥ g(x) ∧ k(x) ≥ h(x) := 
by 
    sorry

end largest_expression_l193_193373


namespace klinker_age_l193_193499

theorem klinker_age :
  ∀ (x : ℕ),
  let klinker_age_now := 48
  let julie_age_now := 12
  let tim_age_now := 8 in
  klinker_age_now + x = 2 * (julie_age_now + x) ∧
  klinker_age_now + x = 3 * (tim_age_now + x) →
  x = 12 :=
  sorry

end klinker_age_l193_193499


namespace SimonNeeds51Sticks_l193_193894

-- Given conditions as definitions
def SimonSticks (S : ℕ) := S
def GerrySticks (S : ℕ) := 2 * S / 3
def MickySticks (S : ℕ) := S + (2 * S / 3) + 9

-- Mathematical proof statement
theorem SimonNeeds51Sticks : ∃ (S : ℕ), SimonSticks S + GerrySticks S + MickySticks S = 129 ∧ S = 51 := 
by {
  let S := 51,
  have h1 : SimonSticks S = S := rfl,
  have h2 : GerrySticks S = 2 * S / 3 := rfl,
  have h3 : MickySticks S = S + (2 * S / 3) + 9 := rfl,
  have h_sum : SimonSticks S + GerrySticks S + MickySticks S = 
                S + (2 * S / 3) + (S + (2 * S / 3) + 9)
              := by simp [SimonSticks, GerrySticks, MickySticks],
  -- Showing the total sticks meet the total number required
  have h_calc : S + (2 * S / 3) + (S + (2 * S / 3) + 9) = 129
              → S = 51 := sorry,
  use 51,
  refine ⟨_, rfl⟩,
  rintro ⟨_⟩,
  sorry
}

end SimonNeeds51Sticks_l193_193894


namespace product_of_possible_b_values_l193_193935

theorem product_of_possible_b_values :
  let b := Float in
  let dist := sqrt ((3 * b - 7)^2 + (2 * b - 8)^2) in
  dist = 3 * sqrt 13 → (b := Float in
  ∃ b fl : List Nat, (∀ b t, true) := Float  b,
  (b₁ b₂ : b) × b₁ * b₂ = b := fl
  let fl := b₁ * b₂ := List  in
begin -- stub proof
  sorry
end

end product_of_possible_b_values_l193_193935


namespace dagger_result_l193_193111

-- Definition of the dagger operation based on the given condition.
def dagger (a b : ℚ) : ℚ :=
  let (m, n) := (a.num, a.denom)
  let (p, q) := (b.num, b.denom)
  (m * p : ℚ) * (2 * (q : ℚ) * (n⁻¹))

-- The given fractions.
def a : ℚ := 5 / 9
def b : ℚ := 7 / 6

-- The proof problem converted into Lean 4 statement.
theorem dagger_result : dagger a b = 140 / 3 := by
  sorry

end dagger_result_l193_193111


namespace angle_between_squares_l193_193064

theorem angle_between_squares (r : ℝ) 
  (h1 : ∀ r > 0, ∃ inscribed_square circumscribed_square, 
    inscribed_square.side = r * Real.sqrt 2 ∧ 
    circumscribed_square.side = 2 * r ∧ 
    inscribed_square.inscribed_in_circle_with_radius r ∧ 
    circumscribed_square.circumscribed_around_circle_with_radius r) :
  ∀ r > 0, ∃ x : ℝ, 
    (Real.sin x = (Real.sqrt 6 - Real.sqrt 2) / 4 ∧ 
    x = Real.arcsin((Real.sqrt 6 - Real.sqrt 2) / 4)) :=
by sorry

end angle_between_squares_l193_193064


namespace average_abcd_l193_193550

-- Define the average condition of the numbers 4, 6, 9, a, b, c, d given as 20
def average_condition (a b c d : ℝ) : Prop :=
  (4 + 6 + 9 + a + b + c + d) / 7 = 20

-- Prove that the average of a, b, c, and d is 30.25 given the above condition
theorem average_abcd (a b c d : ℝ) (h : average_condition a b c d) : 
  (a + b + c + d) / 4 = 30.25 :=
by
  sorry

end average_abcd_l193_193550


namespace _l193_193052

variable {n : ℕ}
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {b : ℕ → ℕ}
variable {T : ℕ → ℕ}

noncomputable def problem1 (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  (∀ n, S (n + 1) = S n + a n + 2) ∧ (2 * S 5 = 3 * (a 4 + a 6))

noncomputable def answer1 (a : ℕ → ℕ) : Prop :=
  a = λ n, 2 * n

noncomputable def b_formula (a : ℕ → ℕ) : ℕ → ℕ :=
  λ n, a n + (1 / 2) ^ a n

noncomputable def T_formula (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  n^2 + n + (1 / 3) - (1 / 3) * (1 / 4) ^ n
  
noncomputable theorem math_problem1 (h1 : problem1 a S)
: answer1 a ∧ (∀ n, T n = T_formula a n) :=
sorry

end _l193_193052


namespace even_not_divisible_by_4_not_sum_of_two_consecutive_odds_l193_193514

theorem even_not_divisible_by_4_not_sum_of_two_consecutive_odds (x n : ℕ) (h₁ : Even x) (h₂ : ¬ ∃ k, x = 4 * k) : x ≠ (2 * n + 1) + (2 * n + 3) := by
  sorry

end even_not_divisible_by_4_not_sum_of_two_consecutive_odds_l193_193514


namespace total_selection_schemes_l193_193282

-- Define the given conditions
def num_phys_ed_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_phys_ed_courses + num_art_courses
def valid_course_combos : finset ℕ := {2, 3}

-- Define the number of selection schemes with the constraints
def selection_schemes : nat :=
  (num_phys_ed_courses.choose 1 * num_art_courses.choose 1) + 
  (num_phys_ed_courses.choose 2 * num_art_courses.choose 1 + 
   num_phys_ed_courses.choose 1 * num_art_courses.choose 2)

-- State the theorem to be proved
theorem total_selection_schemes : selection_schemes = 64 := by
  sorry

end total_selection_schemes_l193_193282


namespace solve_problem_l193_193494

noncomputable def x_star (x : ℝ) : ℝ :=
  if h : x ≥ 2 then 2 * ((x : ℕ) / 2) else 0

noncomputable def f (x : ℝ) : ℝ :=
  x^2 * (x_star x)^2

def problem_statement : Prop :=
  let y := Real.log (f 7.2 / f 5.4) / Real.log 3 in
  abs (y - 1.261) < 0.001   

theorem solve_problem : problem_statement :=
  by sorry

end solve_problem_l193_193494


namespace total_course_selection_schemes_l193_193273

theorem total_course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose_two :=
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 1)
  let choose_three :=
    (Nat.choose PE_courses 2) * (Nat.choose Art_courses 1) +
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 2)
  total_courses = 8
  ∧ (choose_two + choose_three = 64) :=
by
  sorry

end total_course_selection_schemes_l193_193273


namespace statement1_statement2_statement3_l193_193732

variable (a b c m : ℝ)

-- Given condition
def quadratic_eq (a b c : ℝ) : Prop := a ≠ 0

-- Statement 1
theorem statement1 (h0 : quadratic_eq a b c) (h1 : ∀ x, a * x^2 + b * x + c = 0 ↔ x = 1 ∨ x = 2) : 2 * a - c = 0 :=
sorry

-- Statement 2
theorem statement2 (h0 : quadratic_eq a b c) (h2 : b = 2 * a + c) : (b^2 - 4 * a * c) > 0 :=
sorry

-- Statement 3
theorem statement3 (h0 : quadratic_eq a b c) (h3 : a * m^2 + b * m + c = 0) : b^2 - 4 * a * c = (2 * a * m + b)^2 :=
sorry

end statement1_statement2_statement3_l193_193732


namespace length_AC_AM_l193_193019

-- The conditions 
variables {A B C M : Type*}
variables [metric_space A]
variables (AB BC AC AM BM : ℝ)
variables (angle_bac : ℝ)

-- Definitions of the quantities given in the problem
def triangle_ABC := AB = 6 ∧ BC = 10 ∧ angle_bac = π / 2

-- The proof goal
theorem length_AC_AM (H : triangle_ABC AB BC (sqrt 136)) :
  AC = sqrt 136 ∧ AM = sqrt 61 :=
by
  sorry

end length_AC_AM_l193_193019


namespace course_selection_schemes_count_l193_193241

-- Definitions based on the conditions
def num_physical_education_courses : ℕ := 4
def num_art_courses : ℕ := 4

-- Required to choose 2 or 3 courses, with at least one from each category
def valid_selection_cases : list (ℕ × ℕ) := [(1, 1), (1, 2), (2, 1)]

-- Calculate the number of ways to choose k courses from n courses using combination formula
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Calculate the total number of different course selection schemes
def total_course_selection_schemes : ℕ :=
  let case1 := C num_physical_education_courses 1 * C num_art_courses 1
  let case2a := C num_physical_education_courses 2 * C num_art_courses 1
  let case2b := C num_physical_education_courses 1 * C num_art_courses 2
  case1 + case2a + case2b

-- The assertion that the total number of different course selection schemes is 64
theorem course_selection_schemes_count : total_course_selection_schemes = 64 := by
  -- sorry placeholder for proof
  sorry

end course_selection_schemes_count_l193_193241


namespace integral_triples_count_l193_193300

theorem integral_triples_count :
  let a b c := (a, b, c) in 
  (b = 2310) → (a <= b ∧ b <= c) → (a * c = 2310^2) →
  (finset.card { (a, c) | a <= b ∧ b <= c ∧ a * c = 2310^2 }) = 121 :=
by
  sorry

end integral_triples_count_l193_193300


namespace breadth_calculation_l193_193182

-- Given conditions
def length_of_boat : ℝ := 8
def sink_depth : ℝ := 0.01
def mass_of_man : ℝ := 240
def g : ℝ := 9.81
def density_of_water : ℝ := 1000

-- The breadth of the boat that we need to prove
def breadth_of_boat : ℝ := 3.06

-- Proof statement (theorem)
theorem breadth_calculation :
  let V := length_of_boat * breadth_of_boat * sink_depth in
  let W := mass_of_man * g in
  W = density_of_water * V * g →
  breadth_of_boat = 3.06 :=
by
  sorry

end breadth_calculation_l193_193182


namespace quadrilateral_BD_value_l193_193455

theorem quadrilateral_BD_value
  (A B C D : Type) 
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_D : ℝ)
  (AC : ℝ)
  (BD : ℝ)
  (BC_perpendicular_AD : Prop)
  :
  angle_A = 45 ∧
  angle_B = 45 ∧
  angle_D = 45 ∧
  AC = 10 ∧
  BC_perpendicular_AD →
  BD = 10 := 
  by
    intros,
    sorry

end quadrilateral_BD_value_l193_193455


namespace tv_show_years_l193_193179

theorem tv_show_years (s1 s2 s3 : ℕ) (e1 e2 e3 : ℕ) (avg : ℕ) :
  s1 = 8 → e1 = 15 →
  s2 = 4 → e2 = 20 →
  s3 = 2 → e3 = 12 →
  avg = 16 →
  (s1 * e1 + s2 * e2 + s3 * e3) / avg = 14 := by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end tv_show_years_l193_193179


namespace haley_comet_watching_percentage_l193_193434

def total_time_shopping : ℕ := 2
def setup_time : ℚ := 0.5
def snack_time : ℚ := 3 * 0.5
def watching_time : ℚ := 20 / 60

def total_time : ℚ := total_time_shopping + setup_time + snack_time + watching_time
def percentage_comet_watching : ℚ := (watching_time / total_time) * 100

theorem haley_comet_watching_percentage :
  percentage_comet_watching.round = 8 := by
  sorry

end haley_comet_watching_percentage_l193_193434


namespace find_a_l193_193422

noncomputable def normal_dist_X := @MeasureTheory.ProbabilityTheory.normal 1 (Real.sqrt 2)
noncomputable def normal_dist_Y := @MeasureTheory.ProbabilityTheory.normal 3 2

theorem find_a : ∃ a : ℝ, (MeasureTheory.ProbabilityTheory.Prob (fun ω => ⟨normal_dist_X, ω⟩ < 0)
                        = MeasureTheory.ProbabilityTheory.Prob (fun ω => ⟨normal_dist_Y, ω⟩ > a)) 
                        ∧ a = 3 + Real.sqrt 2 :=
begin
  use 3 + Real.sqrt 2,
  split,
  sorry,  -- proof required, but we skip it as instructed
  refl,
end

end find_a_l193_193422


namespace eccentricity_of_ellipse_l193_193736

theorem eccentricity_of_ellipse (a b: ℝ) (h0: a > b) (h1: b > 0) (rat: ∃ t: ℝ, t > 0 ∧ ∀ x y : ℝ, 0 < x → 0 < y → (x / y = 7 / 3) → 
  (x = 7 * t ∧ y = 3 * t)) : 
  (∃ e : ℝ, e = c / a ∧ ∀ x y : ℝ, c ^ 2 = a ^ 2 - b ^ 2 ∧ e = √ (c ^ 2 / a ^ 2) ∧ e = √ (1/4)) →
  e = 1/2 :=
by
  sorry

end eccentricity_of_ellipse_l193_193736


namespace elliptic_PDE_general_solution_l193_193359

theorem elliptic_PDE_general_solution
  (u : ℝ → ℝ → ℝ)
  (u_xx : ℝ → ℝ → ℝ)
  (u_xy : ℝ → ℝ → ℝ)
  (u_yy : ℝ → ℝ → ℝ) 
  (analytic_f : ℂ → ℂ)
  (h_cond : ∀ x y, u_xx x y + 4 * u_xy x y + 5 * u_yy x y = 0) :
  ∃ g : ℝ → ℝ → ℝ, (∀ x y, g x y = (analytic_f (complex.of_real (y - 2*x) + x * complex.I)).re) := 
sorry

end elliptic_PDE_general_solution_l193_193359


namespace negative_values_count_l193_193377

theorem negative_values_count :
  let S := {n : ℕ | n^2 < 196 }
  in S.card = 13 :=
by
  sorry

end negative_values_count_l193_193377


namespace ashton_sheets_l193_193590
-- Import the entire Mathlib to bring in the necessary library

-- Defining the conditions and proving the statement
theorem ashton_sheets (t j a : ℕ) (h1 : t = j + 10) (h2 : j = 32) (h3 : j + a = t + 30) : a = 40 := by
  -- Sorry placeholder for the proof
  sorry

end ashton_sheets_l193_193590


namespace pythagorean_theorem_mod_3_l193_193502

theorem pythagorean_theorem_mod_3 {x y z : ℕ} (h : x^2 + y^2 = z^2) : x % 3 = 0 ∨ y % 3 = 0 ∨ z % 3 = 0 :=
by 
  sorry

end pythagorean_theorem_mod_3_l193_193502


namespace simplify_and_evaluate_expression_evaluate_simplified_expression_l193_193526

noncomputable def simplifiedExpression (x : ℝ) : ℝ :=
  (x^2 - 4*x) / (x^2 - 16) / ((x^2 + 4*x) / (x^2 + 8*x + 16)) - (2*x / (x - 4))

theorem simplify_and_evaluate_expression :
  ∀ (x : ℝ), (x ≠ 0 ∧ x ≠ 4 ∧ x ≠ -4) → simplifiedExpression x = - (x + 4) / (x - 4) :=
begin
  assume x h,
  sorry
end

theorem evaluate_simplified_expression :
  simplifiedExpression (-2) = 1 / 3 :=
begin
  sorry
end

end simplify_and_evaluate_expression_evaluate_simplified_expression_l193_193526


namespace largest_prime_factor_among_numbers_l193_193158

-- Definitions of the numbers with their prime factors
def num1 := 39
def num2 := 51
def num3 := 77
def num4 := 91
def num5 := 121

def prime_factors (n : ℕ) : List ℕ := sorry  -- Placeholder for the prime factors function

-- Prime factors for the given numbers
def factors_num1 := prime_factors num1
def factors_num2 := prime_factors num2
def factors_num3 := prime_factors num3
def factors_num4 := prime_factors num4
def factors_num5 := prime_factors num5

-- Extract the largest prime factor from a list of factors
def largest_prime_factor (factors : List ℕ) : ℕ := sorry  -- Placeholder for the largest_prime_factor function

-- Largest prime factors for each number
def largest_prime_factor_num1 := largest_prime_factor factors_num1
def largest_prime_factor_num2 := largest_prime_factor factors_num2
def largest_prime_factor_num3 := largest_prime_factor factors_num3
def largest_prime_factor_num4 := largest_prime_factor factors_num4
def largest_prime_factor_num5 := largest_prime_factor factors_num5

theorem largest_prime_factor_among_numbers :
  largest_prime_factor_num2 = 17 ∧
  largest_prime_factor_num1 = 13 ∧
  largest_prime_factor_num3 = 11 ∧
  largest_prime_factor_num4 = 13 ∧
  largest_prime_factor_num5 = 11 ∧
  (largest_prime_factor_num2 > largest_prime_factor_num1) ∧
  (largest_prime_factor_num2 > largest_prime_factor_num3) ∧
  (largest_prime_factor_num2 > largest_prime_factor_num4) ∧
  (largest_prime_factor_num2 > largest_prime_factor_num5)
:= by
  -- skeleton proof, details to be filled in
  sorry

end largest_prime_factor_among_numbers_l193_193158


namespace find_h2_l193_193339

noncomputable def h (x : ℝ) : ℝ := 
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) - 1) / (x^15 - 1)

theorem find_h2 : h 2 = 2 :=
by 
  sorry

end find_h2_l193_193339


namespace probability_of_three_digit_number_div_by_three_l193_193576

noncomputable def probability_three_digit_div_by_three : ℚ :=
  let digit_mod3_groups := 
    {rem0 := {3, 6, 9}, rem1 := {1, 4, 7}, rem2 := {2, 5, 8}} in
  let valid_combinations :=
    (finset.card (finset.powersetLen 3 digit_mod3_groups.rem0) +
     (finset.card digit_mod3_groups.rem0 * finset.card digit_mod3_groups.rem1 * finset.card digit_mod3_groups.rem2) +
     finset.card (finset.powersetLen 3 digit_mod3_groups.rem1) +
     finset.card (finset.powersetLen 3 digit_mod3_groups.rem2))
  in
  let total_combinations := finset.card (finset.powersetLen 3 (finset.univ : finset (fin 9))) in
  (valid_combinations : ℚ) / total_combinations

theorem probability_of_three_digit_number_div_by_three :
  probability_three_digit_div_by_three = 5 / 14 := by
  -- provide proof here
  sorry

end probability_of_three_digit_number_div_by_three_l193_193576


namespace cube_root_expression_l193_193073

theorem cube_root_expression : 
  (∛(2^9 * 3^3 * 7^3) = 168) :=
sorry

end cube_root_expression_l193_193073


namespace rectangle_to_square_area_ratio_is_24_25_l193_193920

noncomputable def rectangle_to_square_area_ratio
  (s : ℝ) -- length of side of square S
  (longer_side : ℝ := 1.2 * s) -- longer side of rectangle R
  (shorter_side : ℝ := 0.8 * s) -- shorter side of rectangle R
  (area_R : ℝ := longer_side * shorter_side) -- area of rectangle R
  (area_S : ℝ := s^2) -- area of square S
  : ℝ := 
  area_R / area_S

theorem rectangle_to_square_area_ratio_is_24_25 
  (s : ℝ)
  : rectangle_to_square_area_ratio s = 24 / 25 :=
by 
  sorry

end rectangle_to_square_area_ratio_is_24_25_l193_193920


namespace no_integer_roots_of_quadratic_l193_193516

theorem no_integer_roots_of_quadratic (n : ℤ) : 
  ¬ ∃ (x : ℤ), x^2 - 16 * n * x + 7^5 = 0 := by
  sorry

end no_integer_roots_of_quadratic_l193_193516


namespace min_num_greater_than_50_l193_193953

theorem min_num_greater_than_50 (a1 a2 a3 a4 a5 : ℤ) (h_sum : a1 + a2 + a3 + a4 + a5 = 200) :
  (∃ k : ℕ, k ≥ 1 ∧ (∃ (s : Fin k → ℤ), (∀ i, 51 ≤ s i) ∧ (∃ t : Fin (5 - k) → ℤ, (∀ j, 0 ≤ t j) ∧ finset.univ.sum (λ i, s i) + finset.univ.sum (λ j, t j) = 200))) :=
by
  sorry

end min_num_greater_than_50_l193_193953


namespace smaller_angle_3_15_l193_193597

theorem smaller_angle_3_15 :
  let minute_hand_degrees := 15 * (360 / 60)
  let hour_hand_degrees := 90 + (30 / 60) * 15
  abs (hour_hand_degrees - minute_hand_degrees) = 7.5 :=
by
  let minute_hand_degrees := 15 * (360 / 60)
  let hour_hand_degrees := 90 + (30 / 60) * 15
  have h1 : minute_hand_degrees = 90 := by sorry
  have h2 : hour_hand_degrees = 97.5 := by sorry
  have h3 : abs (hour_hand_degrees - minute_hand_degrees) = 7.5 := by sorry
  exact h3

end smaller_angle_3_15_l193_193597


namespace sufficient_but_not_necessary_condition_for_line_circle_intersection_l193_193566

theorem sufficient_but_not_necessary_condition_for_line_circle_intersection :
  (∀ x y k : ℝ, (x^2 + y^2 = 1) → (y = k * x - 3) → (k ≤ -2 * sqrt 2) → ∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k * x - 3) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_line_circle_intersection_l193_193566


namespace points_for_correct_answer_l193_193610

theorem points_for_correct_answer
  (x y a b : ℕ)
  (hx : x - y = 7)
  (hsum : a + b = 43)
  (hw_score : a * x - b * (20 - x) = 328)
  (hz_score : a * y - b * (20 - y) = 27) :
  a = 25 := 
sorry

end points_for_correct_answer_l193_193610


namespace total_selection_schemes_l193_193276

-- Define the given conditions
def num_phys_ed_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_phys_ed_courses + num_art_courses
def valid_course_combos : finset ℕ := {2, 3}

-- Define the number of selection schemes with the constraints
def selection_schemes : nat :=
  (num_phys_ed_courses.choose 1 * num_art_courses.choose 1) + 
  (num_phys_ed_courses.choose 2 * num_art_courses.choose 1 + 
   num_phys_ed_courses.choose 1 * num_art_courses.choose 2)

-- State the theorem to be proved
theorem total_selection_schemes : selection_schemes = 64 := by
  sorry

end total_selection_schemes_l193_193276


namespace geometric_series_common_ratio_l193_193407

theorem geometric_series_common_ratio (a₁ q : ℝ) (S₃ : ℝ)
  (h1 : S₃ = 7 * a₁)
  (h2 : S₃ = a₁ + a₁ * q + a₁ * q^2) :
  q = 2 ∨ q = -3 :=
by
  sorry

end geometric_series_common_ratio_l193_193407


namespace solve_system_l193_193902

theorem solve_system :
  ∀ (x y z : ℝ),
  (x^2 - 23 * y - 25 * z = -681) →
  (y^2 - 21 * x - 21 * z = -419) →
  (z^2 - 19 * x - 21 * y = -313) →
  (x = 20 ∧ y = 22 ∧ z = 23) :=
by
  intros x y z h1 h2 h3
  sorry

end solve_system_l193_193902


namespace discount_percentage_l193_193946

theorem discount_percentage (original_price increased_percent sale_price discount_percent : ℝ)
  (h_original_price : original_price = 160)
  (h_increased_percent : increased_percent = 0.25)
  (h_sale_price : sale_price = 150)
  (h_discount_percent : discount_percent = 12.5) :
  sale_price = (original_price * (1 + increased_percent)) * (1 - discount_percent / 100) :=
by
  have h_increased_price : original_price * (1 + increased_percent) = 200 := by sorry
  have h_total_sale_price : (original_price * (1 + increased_percent)) * (1 - discount_percent / 100) = sale_price := by sorry
  exact Eq.trans h_total_sale_price.symm h_sale_price

end discount_percentage_l193_193946


namespace grasshoppers_swap_places_l193_193132

theorem grasshoppers_swap_places :
  ∀ (G : ℕ → ℚ → Prop), 
  (∀ t, ∃ (G1 G2 G3 : ℚ), G t G1 ∧ G t G2 ∧ G t G3 ∧ (G1 = -1 ∨ G1 = 1) ∧ (G2 = 0)) ∧
  (∀ t (G1 G2 G3 : ℚ), G t G1 ∧ G t G2 ∧ G t G3 →
    ∃ u (G1' G2' G3' : ℚ), G (t + 1) G1' ∧ G (t + 1) G2' ∧ G (t + 1) G3' ∧
      ((G1' = G1 ∧ G2' = 2*G2 - G1) ∨ (G1' = 2*G1 - G2 ∧ G2' = G2) ∨ (G1' = G1 ∧ G3' = 2*G3 - G1))) →
  ∃ t (G1 G2 G3 : ℚ), (G1 = -1 ∨ G1 = 1) ∧ (G2 = 0) ∧ (G3 = -1 ∨ G3 = 1) ∧
    (G1 = 1 ∧ G3 = -1 ∨ G1 = -1 ∧ G3 = 1) :=
begin
  sorry
end

end grasshoppers_swap_places_l193_193132


namespace find_length_BC_l193_193061

open Real

noncomputable def circle_config : Type :=
  {radius : ℝ,
   α : ℝ,
   cos_α : cos α = 1 / 3}

def length_of_BC (cfg : circle_config) : ℝ :=
  2 * cfg.radius * cos cfg.α

theorem find_length_BC (cfg : circle_config) (h_radius : cfg.radius = 9) (h_cos : cfg.cos_α) : 
  length_of_BC cfg = 6 :=
by
  -- We provide just the statement as instructed
  sorry

end find_length_BC_l193_193061


namespace circumcenter_BDX_on_omega_l193_193034

open Real EuclideanGeometry

-- Define the problem in Lean
theorem circumcenter_BDX_on_omega
  (A B C X D : Point)
  (BC : Line)
  (h_ABC : Triangle A B C)
  (h_X_on_BC : X ∈ BC ∧ seg_length B X + seg_length X C = seg_length B C)
  (h_AX_AB : seg_length A X = seg_length A B)
  (circumcircle_ABC : Circle A B C)
  (h_D_on_AX : lie_on_circle D A circumcircle_ABC)
  (h_AD_on_omega : seg_intersection (line_through A X) circumcircle_ABC = some D) :
  let circumcircle_BDX := circumcircle B D X in
  lie_on_circle (circumcenter (triangle B D X)) A circumcircle_ABC :=
sorry

end circumcenter_BDX_on_omega_l193_193034


namespace find_tan_A_prove_A_eq_2B_l193_193830

noncomputable def cosine_identity (A B : ℝ) (cos_A : ℝ) (cos_A_minus_B : ℝ) : Prop :=
  cos A = cos_A ∧ cos (A - B) = cos_A_minus_B ∧ A > B

theorem find_tan_A (A : ℝ) (cos_A := 4/5) : tan A = 3/4 :=
by
  sorry

theorem prove_A_eq_2B (A B : ℝ) {cos_A := 4/5} {cos_A_minus_B := (3 * real.sqrt 10) / 10} (h : cosine_identity A B (4/5) (3 * real.sqrt 10 / 10)) : A = 2 * B :=
by
  sorry

end find_tan_A_prove_A_eq_2B_l193_193830


namespace value_of_a100_l193_193117

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n ≥ 2, a n = (1 / (4 * a (n - 1))) + (1 / n)

theorem value_of_a100 : ∀ (a : ℕ → ℚ), sequence a -> a 100 = 101 / 200 :=
sorry

end value_of_a100_l193_193117


namespace no_real_roots_l193_193641

def P (n : ℕ) : Polynomial ℝ :=
  match n with
  | 0 => 1
  | k + 1 => X^(17 * (k + 1)) - P k

theorem no_real_roots (n : ℕ) : ¬ ∃ x : ℝ, (P n).eval x = 0 := 
sorry

end no_real_roots_l193_193641


namespace average_of_numbers_between_18_and_57_divisible_by_7_l193_193321

noncomputable def avg_divisible_by_seven (a b : ℕ) (d : ℚ) : Prop :=
  let numbers := [21, 28, 35, 42, 49, 56]
  let sum := numbers.sum
  let count := numbers.length
  d = (sum.to_rat / count)

theorem average_of_numbers_between_18_and_57_divisible_by_7:
  avg_divisible_by_seven 18 57 38.5 :=
by
  sorry

end average_of_numbers_between_18_and_57_divisible_by_7_l193_193321


namespace TotalNumberOfStudents_l193_193812

-- Definitions and conditions
def B : ℕ
def G : ℕ := 200
def ratio (B G : ℕ) : Prop := B / G = 8 / 5

-- Theorem statement
theorem TotalNumberOfStudents (h : ratio B G) : B + G = 520 :=
sorry

end TotalNumberOfStudents_l193_193812


namespace induced_charge_density_l193_193297

noncomputable def sigma_0 (x : ℝ) (h : ℝ) (tilde_f0 tilde_fh : ℝ → ℝ) : ℝ :=
  (1 / (4 * Real.pi ^ 2)) * ∫ λ in -∞..∞, ((tilde_f0 λ + tilde_fh λ * Real.exp (-|λ| * h)) / (1 - Real.exp (-2 * |λ| * h))) * Real.exp (-Complex.I * λ * x)

noncomputable def sigma_h (x : ℝ) (h : ℝ) (tilde_f0 tilde_fh : ℝ → ℝ) : ℝ :=
  (1 / (4 * Real.pi ^ 2)) * ∫ λ in -∞..∞, ((tilde_fh λ + tilde_f0 λ * Real.exp (-|λ| * h)) / (1 - Real.exp (-2 * |λ| * h))) * Real.exp (-Complex.I * λ * x)

theorem induced_charge_density (h : ℝ) (tilde_f0 tilde_fh : ℝ → ℝ) :
  ∃ σ0 σh, (σ0 = sigma_0 h tilde_f0 tilde_fh) ∧ (σh = sigma_h h tilde_f0 tilde_fh) :=
by
  sorry

end induced_charge_density_l193_193297


namespace volume_common_tetrahedra_l193_193571

variables {V : ℝ} -- the volume of the parallelepiped

-- Define the parallelepiped and the tetrahedra
def volume_of_parallelepiped (V : ℝ) : Prop :=
  -- Placeholder for the definition of the volume

def volume_of_common_tetrahedra (V : ℝ) : Prop :=
  -- Placeholder for the common part of the two tetrahedra being V/12

theorem volume_common_tetrahedra (V : ℝ) 
  (h_parallelepiped : volume_of_parallelepiped V) : 
  volume_of_common_tetrahedra (V / 12) :=
sorry -- proof goes here

end volume_common_tetrahedra_l193_193571


namespace tan_sub_pi_over_four_l193_193408

theorem tan_sub_pi_over_four (θ : ℝ) (h : Real.tan θ = 2) : Real.tan (θ - Real.pi / 4) = 1 / 3 :=
by
  sorry

end tan_sub_pi_over_four_l193_193408


namespace beads_needed_for_jewelry_l193_193083

/-
  We define the parameters based on the problem statement.
-/

def green_beads : ℕ := 3
def purple_beads : ℕ := 5
def red_beads : ℕ := 2 * green_beads
def total_beads_per_pattern : ℕ := green_beads + purple_beads + red_beads

def repeats_per_bracelet : ℕ := 3
def repeats_per_necklace : ℕ := 5

/-
  We calculate the total number of beads for 1 bracelet and 10 necklaces.
-/

def beads_per_bracelet : ℕ := total_beads_per_pattern * repeats_per_bracelet
def beads_per_necklace : ℕ := total_beads_per_pattern * repeats_per_necklace
def total_beads_needed : ℕ := beads_per_bracelet + beads_per_necklace * 10

theorem beads_needed_for_jewelry:
  total_beads_needed = 742 :=
by 
  sorry

end beads_needed_for_jewelry_l193_193083


namespace extreme_points_l193_193401

def f (a x : ℝ) : ℝ := x * (Real.log x - a * x)

theorem extreme_points (a : ℝ) (h : 0 < a ∧ a < 1/2) (x1 x2 : ℝ) (hx : x1 < x2) (hext : ∀ (f' : ℝ → ℝ), 
  (f' x1 = 0 ∧ f' x2 = 0) ∧ ∀ x, (f' x < 0 ∨ f' x > 0)) : 
  f a x1 < 0 ∧ f a x2 > -1/2 := 
sorry

end extreme_points_l193_193401


namespace positive_number_with_square_roots_l193_193403

theorem positive_number_with_square_roots (a : ℝ) (h₁ : 3 * a + 2 = -(a + 14)) :
  (3 * a + 2)^2 = 100 :=
by
  sorry

-- Ensure Lean imports correctly for build success

end positive_number_with_square_roots_l193_193403


namespace range_of_a_inequality_solution_set_l193_193768

noncomputable def quadratic_condition_holds (a : ℝ) : Prop :=
∀ (x : ℝ), x^2 - 2 * a * x + a > 0

theorem range_of_a (a : ℝ) (h : quadratic_condition_holds a) : 0 < a ∧ a < 1 := sorry

theorem inequality_solution_set (a x : ℝ) (h1 : 0 < a) (h2 : a < 1) : (a^(x^2 - 3) < a^(2 * x) ∧ a^(2 * x) < 1) ↔ x > 3 := sorry

end range_of_a_inequality_solution_set_l193_193768


namespace problem_l193_193425

open Set

def I := {1, 2, 3, 4, 5, 6}
def M := {1, 2, 6}
def N := {2, 3, 4}

theorem problem : {1, 6} = M ∩ (I \ N) :=
by
  sorry

end problem_l193_193425


namespace minute_hand_distance_traveled_in_45_minutes_l193_193548

-- Definitions for the problem
def minute_hand_length : ℝ := 8 -- The length of the minute hand
def time_in_minutes : ℝ := 45 -- Time in minutes
def revolutions_per_hour : ℝ := 1 -- The number of revolutions per hour for the minute hand (60 minutes)

-- Circumference of the circle
def circumference (radius : ℝ) : ℝ := 2 * Real.pi * radius

-- Number of revolutions the minute hand makes in the given time
def number_of_revolutions (time: ℝ) (total_minutes: ℝ) : ℝ :=
  time / total_minutes * revolutions_per_hour

-- Total distance traveled by the tip of the minute hand
def total_distance_traveled (radius : ℝ) (time: ℝ) : ℝ :=
  circumference(radius) * number_of_revolutions(time, 60)

-- Theorem to prove
theorem minute_hand_distance_traveled_in_45_minutes :
  total_distance_traveled(minute_hand_length, time_in_minutes) = 12 * Real.pi :=
by
  -- Placeholder for the proof
  sorry

end minute_hand_distance_traveled_in_45_minutes_l193_193548


namespace find_a_plus_b_l193_193173

section TriangleProblem

variables {A B C D E F K L : Type} [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq D]
          [decidable_eq E] [decidable_eq F] [decidable_eq K] [decidable_eq L]
          [has_distance A] [has_distance B] [has_distance C] [has_distance D]
          [has_distance E] [has_distance F] [has_distance K] [has_distance L]

-- Sides of the triangle ABC
variables (AB BC AC : ℝ) (hAB : AB = 15) (hBC : BC = 22) (hAC : AC = 20)

-- Points D, E, F on BC, AC, AB respectively making AD, BE, CF concurrent at K
variables (D E F : Point) (hD : D ∈ LineSegment B C) (hE : E ∈ LineSegment A C) (hF : F ∈ LineSegment A B)
          (hConcurrent : ∃ K, IsConcurrent (AD, BE, CF))

-- Given conditions
variables (hAK_KD : AK / KD = 11 / 7) (hBD : BD = 6)

-- Second intersection point of the circumcircles of △BFK and △CEK
variables (KL_squared : Real) (hIntersection : KL = second_intersection_circumcircles (circumcircle BF K) (circumcircle CEK))

-- Statement to prove
theorem find_a_plus_b : KL_squared = 486 / 11 := sorry

end TriangleProblem

end find_a_plus_b_l193_193173


namespace part_a_part_b_l193_193613

theorem part_a (A B C D P : Point) (hP : OnCircumcircle P A B C D) : 
  Let S_line := SimsonLine P ⟨B, C, D⟩
  Let T_line := SimsonLine P ⟨C, D, A⟩
  Let U_line := SimsonLine P ⟨D, A, B⟩
  Let V_line := SimsonLine P ⟨A, B, C⟩
  ∃ l : Line, ProjectionsOnSimsonLines l ⟨S_line, T_line, U_line, V_line⟩.
sorry

theorem part_b (n : ℕ) (h : n ≥ 3) (V : Fin n.succ → Point) (P : Point) 
  (hP : OnCircumcircle P (V 0) (V 1) (V 2) (V n.succ)) 
  (h_ind : ∀ m : ℕ, m < n → ∃ l : Line, ProjectionsOnSimsonLines l (SimsonLinesForNMinusOne (Fin m.succ))) : 
  ∃ l : Line, ProjectionsOnSimsonLines l (SimsonLinesForN V P). 
sorry

end part_a_part_b_l193_193613


namespace area_of_enclosed_triangle_l193_193755

noncomputable def linear_function : ℝ → ℝ := λ x, 2 * x - 1

theorem area_of_enclosed_triangle :
  let A := (-4, -9)
  let B := (3, 5)
  let C := (1 / 2, 0)
  let D := (0, -1)
  (linear_function A.1 = A.2) →
  (linear_function B.1 = B.2) →
  linear_function C.1 = C.2 →
  linear_function D.1 = D.2 →
  (1 / 2 * 1 * (1 / 2) = 1 / 4) :=
begin
  intros hA hB hC hD,
  sorry
end

end area_of_enclosed_triangle_l193_193755


namespace resistance_is_two_l193_193449

theorem resistance_is_two :
  ∃ R1 : ℝ, R1 = 2 ∧ 
  (let R2 := 5
       R3 := 6
       R_total := 0.8666666666666666 in
       1 / R_total = 1 / R1 + 1 / R2 + 1 / R3) :=
by
  sorry

end resistance_is_two_l193_193449


namespace clock_angle_at_3_15_l193_193599

-- Conditions
def full_circle_degrees : ℕ := 360
def hour_degree : ℕ := full_circle_degrees / 12
def minute_degree : ℕ := full_circle_degrees / 60
def minute_position (m : ℕ) : ℕ := m * minute_degree
def hour_position (h m : ℕ) : ℕ := h * hour_degree + m * (hour_degree / 60)

-- Theorem to prove
theorem clock_angle_at_3_15 : (|minute_position 15 - hour_position 3 15| : ℚ) = 7.5 := by
  sorry

end clock_angle_at_3_15_l193_193599


namespace fate_region_is_correct_l193_193053

noncomputable def f (x : ℝ) : ℝ := x^2 + 3 * x + 2
noncomputable def g (x : ℝ) : ℝ := 2 * x + 3

theorem fate_region_is_correct {a b : ℝ} (D : set ℝ) :
  D = {x | -2 ≤ x ∧ x ≤ -1} ∪ {x | 0 ≤ x ∧ x ≤ 1} →
  ∀ x ∈ D, |f x - g x| ≤ 1 :=
sorry

end fate_region_is_correct_l193_193053


namespace length_of_DE_in_triangle_l193_193020

noncomputable def triangle_length_DE (BC : ℝ) (C_deg: ℝ) (DE : ℝ) : Prop :=
  BC = 24 * Real.sqrt 2 ∧ C_deg = 45 ∧ DE = 12 * Real.sqrt 2

theorem length_of_DE_in_triangle :
  ∀ (BC : ℝ) (C_deg: ℝ) (DE : ℝ), (BC = 24 * Real.sqrt 2 ∧ C_deg = 45) → DE = 12 * Real.sqrt 2 :=
by
  intros BC C_deg DE h_cond
  have h_length := h_cond.2
  sorry

end length_of_DE_in_triangle_l193_193020


namespace course_selection_schemes_l193_193210

theorem course_selection_schemes :
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  choose_2_courses + choose_3_courses = 64 :=
by
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  show choose_2_courses + choose_3_courses = 64 from sorry

end course_selection_schemes_l193_193210


namespace pizza_slices_left_l193_193161

def initial_slices : ℕ := 16
def eaten_during_dinner : ℕ := initial_slices / 4
def remaining_after_dinner : ℕ := initial_slices - eaten_during_dinner
def yves_eaten : ℕ := remaining_after_dinner / 4
def remaining_after_yves : ℕ := remaining_after_dinner - yves_eaten
def siblings_eaten : ℕ := 2 * 2
def remaining_after_siblings : ℕ := remaining_after_yves - siblings_eaten

theorem pizza_slices_left : remaining_after_siblings = 5 := by
  sorry

end pizza_slices_left_l193_193161


namespace train_length_is_549_95_l193_193306

noncomputable def length_of_train 
(speed_of_train : ℝ) -- 63 km/hr
(speed_of_man : ℝ) -- 3 km/hr
(time_to_cross : ℝ) -- 32.997 seconds
: ℝ := 
(speed_of_train - speed_of_man) * (5 / 18) * time_to_cross

theorem train_length_is_549_95 (speed_of_train : ℝ) (speed_of_man : ℝ) (time_to_cross : ℝ) :
    speed_of_train = 63 → speed_of_man = 3 → time_to_cross = 32.997 →
    length_of_train speed_of_train speed_of_man time_to_cross = 549.95 :=
by
  intros h_train h_man h_time
  rw [h_train, h_man, h_time]
  norm_num
  sorry

end train_length_is_549_95_l193_193306


namespace train_length_l193_193166

-- We need to define the speed conversion and the formula for distance.
def speed_kmph_to_mps (kmph : ℝ) : ℝ := kmph * 1000 / 3600
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Given conditions
def train_speed_kmph : ℝ := 50
def crossing_time_sec : ℝ := 9

-- The main statement verifying the length of the train
theorem train_length :
  distance (speed_kmph_to_mps train_speed_kmph) crossing_time_sec ≈ 125.01 := by
  sorry

end train_length_l193_193166


namespace concurrency_or_parallelism_l193_193473

variables {A B C S_A T_A P_A Q_A : Type}
variables {theta : ℝ}
variables {l_A l_B l_C : Type}
variables (h_triangle : IsTriangle A B C)
variables (h_theta : theta < (1 / 2) * min (angle A B C) (angle B C A) (angle C A B))

-- Definitions of S_A and T_A
variables (h_S_A : LiesOnSegment S_A B C)
variables (h_T_A : LiesOnSegment T_A B C)
variables (h_angle_BAS_A : Angle A B S_A = theta)
variables (h_angle_T_AAC : Angle T_A A C = theta)

-- Definitions of projections P_A and Q_A
variables (h_P_A : PerpendicularFoot P_A B (Line A S_A))
variables (h_Q_A : PerpendicularFoot Q_A C (Line A T_A))

-- Definitions of the perpendicular bisectors l_A, l_B, and l_C
variables (h_l_A : IsPerpendicularBisector l_A P_A Q_A)
variables (h_l_B : IsPerpendicularBisector l_B _ _)
variables (h_l_C : IsPerpendicularBisector l_C _ _)

-- The theorem statement
theorem concurrency_or_parallelism :
  ConcurrentOrParallel l_A l_B l_C :=
sorry

end concurrency_or_parallelism_l193_193473


namespace janet_waiting_time_l193_193833

-- Define the speeds and distance
def janet_speed : ℝ := 30 -- miles per hour
def sister_speed : ℝ := 12 -- miles per hour
def lake_width : ℝ := 60 -- miles

-- Define the travel times
def janet_travel_time : ℝ := lake_width / janet_speed
def sister_travel_time : ℝ := lake_width / sister_speed

-- The theorem to be proved
theorem janet_waiting_time : 
  sister_travel_time - janet_travel_time = 3 := 
by 
  sorry

end janet_waiting_time_l193_193833


namespace triangle_acute_integers_count_l193_193714

def is_acute_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a ∧ 
  if c ≥ b ∧ c ≥ a then c^2 < a^2 + b^2
  else if b ≥ a ∧ b ≥ c then b^2 < a^2 + c^2
  else a^2 < b^2 + c^2

def count_acute_integers : ℕ :=
  let a := 12 in
  let b := 30 in
  Finset.card (Finset.filter (λ x, is_acute_triangle a b x) (Finset.Ico 19 42))

theorem triangle_acute_integers_count : count_acute_integers = 5 :=
by
  sorry

end triangle_acute_integers_count_l193_193714


namespace matrix_A_eigenvalue_composite_transformation_apply_point_l193_193740

noncomputable def matrix_A (a : ℝ) := ![![a, 2], ![-1, 4]]

def matrix_γ := ![![1, 0], ![0, -1]]
def point_P := ![1, 1]

theorem matrix_A_eigenvalue (a : ℝ) (h : ∀ (λ : ℝ), λ = 2 → (λ - a) * (λ - 4) + 2 = 0) : 
  matrix_A a = ![![-1, 2], ![-1, 4]] :=
sorry

theorem composite_transformation_apply_point (P : ℝ → matrix (fin 2) (fin 2) → ℝ) :
  P' = matrix.mul matrix_γ (matrix_A (-1)) ⬝ point_P 
  ∧ P' = ![3, -3] :=
sorry

end matrix_A_eigenvalue_composite_transformation_apply_point_l193_193740


namespace find_ordered_pair_l193_193429

theorem find_ordered_pair (p q : ℝ) 
    (H : (λ u v : ℝ × ℝ × ℝ, (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1))
          (3, p, -8) 
          (6, 5, q) = (0, 0, 0)) : 
    (p, q) = (5 / 2, -16) := 
sorry

end find_ordered_pair_l193_193429


namespace janet_waiting_time_l193_193834

-- Define the speeds and distance
def janet_speed : ℝ := 30 -- miles per hour
def sister_speed : ℝ := 12 -- miles per hour
def lake_width : ℝ := 60 -- miles

-- Define the travel times
def janet_travel_time : ℝ := lake_width / janet_speed
def sister_travel_time : ℝ := lake_width / sister_speed

-- The theorem to be proved
theorem janet_waiting_time : 
  sister_travel_time - janet_travel_time = 3 := 
by 
  sorry

end janet_waiting_time_l193_193834


namespace dartboard_odd_score_probability_l193_193674

theorem dartboard_odd_score_probability :
  let π := Real.pi
  let r_outer := 4
  let r_inner := 2
  let area_inner := π * r_inner * r_inner
  let area_outer := π * r_outer * r_outer
  let area_annulus := area_outer - area_inner
  let area_inner_region := area_inner / 3
  let area_outer_region := area_annulus / 3
  let odd_inner_regions := 1
  let even_inner_regions := 2
  let odd_outer_regions := 2
  let even_outer_regions := 1
  let prob_odd_inner := (odd_inner_regions * area_inner_region) / (area_inner + area_annulus)
  let prob_even_inner := (even_inner_regions * area_inner_region) / (area_inner + area_annulus)
  let prob_odd_outer := (odd_outer_regions * area_outer_region) / (area_inner + area_annulus)
  let prob_even_outer := (even_outer_regions * area_outer_region) / (area_inner + area_annulus)
  let prob_odd_region := prob_odd_inner + prob_odd_outer
  let prob_even_region := prob_even_inner + prob_even_outer
  let prob_odd_score := (prob_odd_region * prob_even_region) + (prob_even_region * prob_odd_region)
  prob_odd_score = 5 / 9 :=
by
  -- Proof omitted
  sorry

end dartboard_odd_score_probability_l193_193674


namespace problem_l193_193601

theorem problem (m : ℤ) : (10 : ℝ)^m = 10^2 * (sqrt ((10^90) / (10^(-4)) : ℝ)) → m = 49 :=
by
  sorry

end problem_l193_193601


namespace sum_fractions_l193_193673

theorem sum_fractions :
  (∑ k in Finset.range 16, k / 3 : ℚ) = 40 := by
  sorry

end sum_fractions_l193_193673


namespace perpendicular_diagonals_iff_projections_form_rectangle_l193_193881

variables {A B C D P : Type}

def is_convex_quadrilateral (A B C D : Type) := -- condition for a convex quadrilateral
sorry

def orthogonal_projection (P : Type) (l : Type) : Type := -- orthogonal projection of a point on a line
sorry

def forms_rectangle (P Q R S : Type) := -- definition for vertices forming a rectangle
sorry

theorem perpendicular_diagonals_iff_projections_form_rectangle
  (A B C D : Type)
  (h_convex : is_convex_quadrilateral A B C D) :
  (∀ P : Type, etc. sorry) ↔ (∃ P : Type, (orthogonal_projection P A B) (orthogonal_projection P B C) (orthogonal_projection P C D) (orthogonal_projection P D A) 
   (forms_rectangle (orthogonal_projection P A B) (orthogonal_projection P B C) (orthogonal_projection P C D) (orthogonal_projection P D A))) :=
sorry

end perpendicular_diagonals_iff_projections_form_rectangle_l193_193881


namespace total_course_selection_schemes_l193_193232

theorem total_course_selection_schemes (
  pe_courses art_courses : Finset ℕ
) : 
  pe_courses.card = 4 →
  art_courses.card = 4 →
  let total_schemes := 
    (pe_courses.card.choose 1 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 2 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 1 * art_courses.card.choose 2) in
  total_schemes = 64 := 
by
  intros h1 h2
  let case1 := pe_courses.card.choose 1 * art_courses.card.choose 1 -- 4 * 4
  let case2_1 := pe_courses.card.choose 2 * art_courses.card.choose 1 -- 6 * 4
  let case2_2 := pe_courses.card.choose 1 * art_courses.card.choose 2 -- 4 * 6
  let case2 := case2_1 + case2_2 -- 24 + 24
  let total_schemes := case1 + case2 -- 16 + 48
  have hcard := by
    simp [Nat.choose]
    exact h1
  have hcase1 : case1 = 16 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_1 : case2_1 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_2 : case2_2 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2 : case2 = 48 := by {
    rw [hcase2_1, hcase2_2]
    norm_num
  }
  have htotal_schemes : total_schemes = 64 := by {
    rw [hcase1, hcase2]
    norm_num
  }
  exact htotal_schemes

end total_course_selection_schemes_l193_193232


namespace erika_donut_holes_l193_193664

open Int

theorem erika_donut_holes (r₁ r₂ r₃ : ℝ) (surface_area : ℝ → ℝ) :
  r₁ = 5 ∧ r₂ = 7 ∧ r₃ = 9 ∧ surface_area = λ r, 4 * Real.pi * r^2 →
  let sa₁ := surface_area r₁ in
  let sa₂ := surface_area r₂ in
  let sa₃ := surface_area r₃ in
  sa₁ = 100 * Real.pi ∧ sa₂ = 196 * Real.pi ∧ sa₃ = 324 * Real.pi →
  let lcm_sa := Nat.lcm (100 : ℕ) (Nat.lcm (196 : ℕ) (324 : ℕ)) * Real.pi in
  (lcm_sa / sa₁) = 441 :=
by
  sorry

end erika_donut_holes_l193_193664


namespace expression_odd_l193_193786

theorem expression_odd (a b c : ℕ) (ha : odd a) (hb : odd b) (hc : 0 < c) : odd (7^a + 3 * (b - 1) * c^2) :=
sorry

end expression_odd_l193_193786


namespace john_pays_total_l193_193845

-- Definitions based on conditions
def total_cans : ℕ := 30
def price_per_can : ℝ := 0.60

-- Main statement to be proven
theorem john_pays_total : (total_cans / 2) * price_per_can = 9 := 
by
  sorry

end john_pays_total_l193_193845


namespace exists_point_D_l193_193395

-- Definitions of points and the angle
variables (O X Y A B : Point)
variables (hOX : Ray O X)
variables (hA_on_OX : On A hOX)
variables (hAngle_XOY : Angle X O Y)
variables (hB_inside_XOY : InsideAngle B (Angle X O Y))

-- Definition of line through point B intersecting OX at C and OY at D
noncomputable def C : Point := sorry
noncomputable def D : Point := sorry

-- Stating the equality to be proven
theorem exists_point_D (O X Y A B : Point)
  (hOX : Ray O X)
  (hA_on_OX : On A hOX)
  (hAngle_XOY : Angle X O Y)
  (hB_inside_XOY : InsideAngle B hAngle_XOY)
  (C D : Point)
  (hLine_through_B : On B (Line_through C D))
  (hIntersect_OX : Intersect (Line_through C D) (Line_through O X) C)
  (hIntersect_OY : Intersect (Line_through C D) (Line_through O Y) D) :
  Distance D C = Distance D A :=
sorry

end exists_point_D_l193_193395


namespace Julie_food_order_l193_193847

-- Definitions of conditions
def Letitia_order := 20
def Anton_order := 30
def individual_tip := 4
def total_tip := 3 * individual_tip
def percentage_tip := 0.20

-- Define the main theorem that needs to be proved
theorem Julie_food_order (J : ℝ) :
  (percentage_tip * (J + Letitia_order + Anton_order) = total_tip) → 
  J = 10 :=
by
  -- Skip the proof for now
  sorry

end Julie_food_order_l193_193847


namespace total_course_selection_schemes_l193_193193

theorem total_course_selection_schemes :
  let n_physical_education := 4
  let n_art := 4
  let total_courses := n_physical_education + n_art in
  let choose2_courses := (Nat.choose n_physical_education 1) * (Nat.choose n_art 1)
  let choose3_courses := (Nat.choose n_physical_education 2 * Nat.choose n_art 1) + (Nat.choose n_physical_education 1 * Nat.choose n_art 2) in
  total_courses = n_physical_education + n_art →
  choose2_courses + choose3_courses = 64 :=
by
  intros n_physical_education n_art total_courses choose2_courses choose3_courses h
  have h_choose2_courses: choose2_courses = 16 := by
    simp [n_physical_education, n_art, Nat.choose]
  have h_choose3_courses: choose3_courses = 48 := by
    simp [n_physical_education, n_art, Nat.choose]
  rw [h_choose2_courses, h_choose3_courses]
  exact Nat.add_eq_right.2 rfl

end total_course_selection_schemes_l193_193193


namespace trigonometric_identity_1_trigonometric_identity_2_l193_193524

-- Define the given angles and their sum condition
variables {α β γ : ℝ}

-- First statement to prove
theorem trigonometric_identity_1 (h : α + β + γ = real.pi):
  real.sin α * real.cos β * real.cos γ + 
  real.sin β * real.cos α * real.cos γ + 
  real.sin γ * real.cos α * real.cos β = 
  real.sin α * real.sin β * real.sin γ :=
sorry

-- Second statement to prove
theorem trigonometric_identity_2 (h : α + β + γ = real.pi):
  (real.cot (β / 2) + real.cot (γ / 2)) /
  (real.cot (α / 2) + real.cot (γ / 2)) =
  real.sin α / real.sin β :=
sorry

end trigonometric_identity_1_trigonometric_identity_2_l193_193524


namespace circle_integer_sum_l193_193695

theorem circle_integer_sum (x y : ℕ) (plist : list ℕ) :
  (∀ a ∈ plist, 1 ≤ a ∧ a ≤ 12) →   -- Each integer from 1 to 12 is placed
  (plist.length = 12) →   -- There are exactly 12 integers
  (∀ (a b : ℕ), a ∈ plist → b ∈ plist → abs (a - b) ≤ 2) →   -- Positive difference between adjacent integers is at most 2
  (plist.contains 3) → (plist.contains 4) →   -- The integers 3 and 4 are placed
  (plist.contains x) → (plist.contains y) →   -- The integers x and y are placed
  (plist.indexOf 3 + 1 = plist.indexOf 4) ∨ (plist.indexOf 4 + 1 = plist.indexOf 3) →   -- 3 and 4 are next to each other
  (x + y = 20) := sorry

end circle_integer_sum_l193_193695


namespace supremum_of_expression_l193_193690

noncomputable theory

variable (ξ : ℝ → ℝ) (E : (ℝ → ℝ) → ℝ)

axiom expectation_property1 : ∀ (ξ : ℝ → ℝ), (∀ x, 1 ≤ ξ x ∧ ξ x ≤ 2) → ∃ μ, (E ξ = μ) 

theorem supremum_of_expression :
  ∀ (ξ : ℝ → ℝ), (∀ x, 1 ≤ ξ x ∧ ξ x ≤ 2) →
  (∃ μ, μ = E ξ ∧ 
  (∃ ν, ν = E (λ x, log (ξ x)) ∧ 
  (log μ - ν = -log (log 2) - 1 + log 2))) :=
by
  sorry

end supremum_of_expression_l193_193690


namespace rhombus_area_fraction_l193_193059

theorem rhombus_area_fraction :
  let large_square_area := (6 : ℕ)^2
  let rhombus_area := (1 : ℝ) / 4
  (rhombus_area / large_square_area) = (1 : ℝ) / 144 :=
by
  sorry

end rhombus_area_fraction_l193_193059


namespace coefficient_of_x2_in_expansion_l193_193912

theorem coefficient_of_x2_in_expansion :
  let expr := (λ x : ℝ, (x + 1 / x + 2) ^ 5)
  let coeff := 120
  ∃ (x : ℝ), coefficient (expr x) 2 = coeff :=
by
  sorry

end coefficient_of_x2_in_expansion_l193_193912


namespace inequality_AM_GM_l193_193374

theorem inequality_AM_GM (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, 0 < a i) :
  (∑ i in Finset.range n, (a i / a ((i + 1) % n)) ^ n) ≥ (∑ i in Finset.range n, a ((i + 1) % n) / a i) :=
by sorry

end inequality_AM_GM_l193_193374


namespace molecular_weight_CCl4_l193_193143

theorem molecular_weight_CCl4 (MW_7moles_CCl4 : ℝ) (h : MW_7moles_CCl4 = 1064) : 
  MW_7moles_CCl4 / 7 = 152 :=
by
  sorry

end molecular_weight_CCl4_l193_193143


namespace base_for_600_is_six_l193_193602

-- Define the conditions stated in the problem
def in_base (n b : ℕ) (a0 a1 a2 a3 : ℕ) : Prop :=
  n = a3 * b^3 + a2 * b^2 + a1 * b + a0

def distinct_digits (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main statement we want to prove
theorem base_for_600_is_six :
  ∃ (b : ℕ) (A B C : ℕ) (a0 a1 a2 a3 : ℕ), 
    600 = a3 * b^3 + a2 * b^2 + a1 * b + a0 ∧
    a3 = A ∧ a2 = B ∧ a1 = C ∧ a0 = A ∧
    distinct_digits A B C ∧
    6 = b :=
begin
  sorry
end

end base_for_600_is_six_l193_193602


namespace jed_speeding_l193_193808

-- Define the constants used in the conditions
def F := 16
def T := 256
def S := 50

theorem jed_speeding : (T / F) + S = 66 := 
by sorry

end jed_speeding_l193_193808


namespace breadth_of_boat_l193_193181

theorem breadth_of_boat :
  ∀ (L : ℝ) (h : ℝ) (m : ℝ) (g : ℝ) (rho : ℝ),
    L = 8 → h = 0.01 → m = 160 → g = 9.81 → rho = 1000 →
    (L * 2 * h = (m * g) / (rho * g)) :=
by
  intros L h m g rho hL hh hm hg hrho
  sorry

end breadth_of_boat_l193_193181


namespace rectangle_to_square_area_ratio_is_24_25_l193_193923

noncomputable def rectangle_to_square_area_ratio
  (s : ℝ) -- length of side of square S
  (longer_side : ℝ := 1.2 * s) -- longer side of rectangle R
  (shorter_side : ℝ := 0.8 * s) -- shorter side of rectangle R
  (area_R : ℝ := longer_side * shorter_side) -- area of rectangle R
  (area_S : ℝ := s^2) -- area of square S
  : ℝ := 
  area_R / area_S

theorem rectangle_to_square_area_ratio_is_24_25 
  (s : ℝ)
  : rectangle_to_square_area_ratio s = 24 / 25 :=
by 
  sorry

end rectangle_to_square_area_ratio_is_24_25_l193_193923


namespace minimum_rows_needed_l193_193445

variable (n : ℕ)
variable (C : Fin n → ℕ)
variable (h1 : ∀ i : Fin n, 1 ≤ C i ∧ C i ≤ 39)
variable (h2 : ∑ i : Fin n, C i = 1990)

theorem minimum_rows_needed : 
    ∃ r : ℕ, (∀ i : Fin n, 1 ≤ C i ∧ C i ≤ 39) ∧ (∑ i : Fin n, C i = 1990) →
    r = 12 :=
sorry

end minimum_rows_needed_l193_193445


namespace area_ratio_rect_sq_l193_193928

variable (s : ℝ)

def side_len_sq (S : ℝ) : Prop := s = S
def longer_side_rect (R : ℝ) : Prop := R = 1.2 * s
def shorter_side_rect (R : ℝ) : Prop := R = 0.8 * s
def area_sq (S : ℝ) : ℝ := S * S
def area_rect (R_long R_short : ℝ) : ℝ := R_long * R_short
def ratio_area (areaR areaS : ℝ) : ℝ := areaR / areaS

theorem area_ratio_rect_sq (s S R_long R_short : ℝ) (h1 : side_len_sq s S) (h2 : longer_side_rect s R_long) (h3 : shorter_side_rect s R_short) :
  ratio_area (area_rect R_long R_short) (area_sq S) = 24/25 :=
by
  sorry

end area_ratio_rect_sq_l193_193928


namespace least_value_r_minus_p_l193_193460

theorem least_value_r_minus_p (x : ℝ) (h1 : 1 / 2 < x) (h2 : x < 5) :
  ∃ r p, r = 5 ∧ p = 1/2 ∧ r - p = 9 / 2 :=
by
  sorry

end least_value_r_minus_p_l193_193460


namespace exists_nat_square_pattern_l193_193172

theorem exists_nat_square_pattern (n : ℕ) : ∃ m : ℕ, 
  (decimal_representation (m^2) starts_with_n_ones_and_ends_with_combination_of_n_ones_and_twos) := 
by
  sorry

end exists_nat_square_pattern_l193_193172


namespace wage_increase_to_restore_l193_193794

theorem wage_increase_to_restore (W : ℝ) (hW : W > 0) :
  let new_wage := 0.7 * W in
  ((W / new_wage) - 1) * 100 = 42.86 :=
by
  sorry

end wage_increase_to_restore_l193_193794


namespace smallest_value_x_squared_plus_six_x_plus_nine_l193_193146

theorem smallest_value_x_squared_plus_six_x_plus_nine : ∀ x : ℝ, x^2 + 6 * x + 9 ≥ 0 :=
by sorry

end smallest_value_x_squared_plus_six_x_plus_nine_l193_193146


namespace total_course_selection_schemes_l193_193267

theorem total_course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose_two :=
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 1)
  let choose_three :=
    (Nat.choose PE_courses 2) * (Nat.choose Art_courses 1) +
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 2)
  total_courses = 8
  ∧ (choose_two + choose_three = 64) :=
by
  sorry

end total_course_selection_schemes_l193_193267


namespace course_selection_schemes_l193_193208

theorem course_selection_schemes :
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  choose_2_courses + choose_3_courses = 64 :=
by
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  show choose_2_courses + choose_3_courses = 64 from sorry

end course_selection_schemes_l193_193208


namespace boxes_containing_neither_l193_193032

-- Define the conditions
def total_boxes : ℕ := 15
def boxes_with_pencils : ℕ := 8
def boxes_with_pens : ℕ := 5
def boxes_with_markers : ℕ := 3
def boxes_with_pencils_and_pens : ℕ := 2
def boxes_with_pencils_and_markers : ℕ := 1
def boxes_with_pens_and_markers : ℕ := 1
def boxes_with_all_three : ℕ := 0

-- The proof problem
theorem boxes_containing_neither (h: total_boxes = 15) : 
  total_boxes - ((boxes_with_pencils - boxes_with_pencils_and_pens - boxes_with_pencils_and_markers) + 
  (boxes_with_pens - boxes_with_pencils_and_pens - boxes_with_pens_and_markers) + 
  (boxes_with_markers - boxes_with_pencils_and_markers - boxes_with_pens_and_markers) + 
  boxes_with_pencils_and_pens + boxes_with_pencils_and_markers + boxes_with_pens_and_markers) = 3 := 
by
  -- Specify that we want to use the equality of the number of boxes
  sorry

end boxes_containing_neither_l193_193032


namespace total_collisions_l193_193709

theorem total_collisions (n m : ℕ) (h_n : n = 5) (h_m : m = 5)
    (h_speed : ∀ (i : ℕ) (i' : ℕ) (j : ℕ) (j' : ℕ), 
        (i ≠ i' ∧ j ≠ j') → (i < n ∧ j < m) → (i = i' ∨ j = j') → true) :
    (n * m = 25) := 
by 
  rw [h_n, h_m]
  norm_num

end total_collisions_l193_193709


namespace A_beats_B_by_seconds_l193_193000

theorem A_beats_B_by_seconds :
  ∀ (t_A : ℝ) (distance_A distance_B : ℝ),
  t_A = 156.67 →
  distance_A = 1000 →
  distance_B = 940 →
  (distance_A * t_A = 60 * (distance_A / t_A)) →
  t_A ≠ 0 →
  ((60 * t_A / distance_A) = 9.4002) :=
by
  intros t_A distance_A distance_B h1 h2 h3 h4 h5
  sorry

end A_beats_B_by_seconds_l193_193000


namespace probability_vertex_closest_point_l193_193865

open Real Set

def S : Set (Fin 2012 → ℝ) :=
  { x | ∑ i, |x i| ≤ 1 }

def T : Set (Fin 2012 → ℝ) :=
  { x | ∀ i, |x i| ≤ 2 ∧ ∃ i, |x i| = 2 }

def is_vertex_S (v : Fin 2012 → ℝ) : Prop :=
  ∃ i, (v i = 1 ∨ v i = -1) ∧ ∀ j, j ≠ i → v j = 0

theorem probability_vertex_closest_point (p : Fin 2012 → ℝ) (hp : p ∈ T) :
  let count := (2 : ℝ) ^ (2011 : ℝ)
  let total := (2 * 2) ^ (2012 : ℝ)
  ∃! v ∈ S, is_vertex_S v ∧ (∃! w ∈ S, dist p v ≤ dist p w) → count / total = 1 / (2 ^ 2013) :=
sorry

end probability_vertex_closest_point_l193_193865


namespace negative_values_of_x_l193_193379

theorem negative_values_of_x : 
  let f (x : ℤ) := Int.sqrt (x + 196)
  ∃ (n : ℕ), (f (n ^ 2 - 196) > 0 ∧ f (n ^ 2 - 196) = n) ∧ ∃ k : ℕ, k = 13 :=
by
  sorry

end negative_values_of_x_l193_193379


namespace tangent_line_through_A_l193_193759

noncomputable def circle_radius_squared : ℝ := 4
def is_on_circle (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 = circle_radius_squared
def A : ℝ × ℝ := (1, real.sqrt 3)
def tangent_line (p : ℝ × ℝ) : ℝ → ℝ → Prop := λ x y, x + real.sqrt 3 * y - 4 = 0

theorem tangent_line_through_A :
  is_on_circle A →
  tangent_line A.1 A.2 1 (real.sqrt 3) :=
by
  intros h
  sorry

end tangent_line_through_A_l193_193759


namespace anton_wins_infinitely_l193_193315

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def anton_wins (n : ℕ) : Prop :=
  ¬ is_perfect_square n →
  ∃ m : ℕ, is_perfect_square (n + n + 2 + m)

theorem anton_wins_infinitely : ∀n ≥ 0, ¬ is_perfect_square n → ∃ k : ℕ, 
  ¬ is_perfect_square (3 * k^2 - 1) ∧ anton_wins (3 * k^2 - 1) :=
by 
  intros n h1 h2
  use k
  sorry

end anton_wins_infinitely_l193_193315


namespace maddie_milk_usage_l193_193055

-- Define the constants based on the problem conditions
def cups_per_day : ℕ := 2
def ounces_per_cup : ℝ := 1.5
def bag_cost : ℝ := 8
def ounces_per_bag : ℝ := 10.5
def weekly_coffee_expense : ℝ := 18
def gallon_milk_cost : ℝ := 4

-- Define the proof problem
theorem maddie_milk_usage : 
  (0.5 : ℝ) = (weekly_coffee_expense - 2 * ((cups_per_day * ounces_per_cup * 7) / ounces_per_bag * bag_cost)) / gallon_milk_cost :=
by 
  sorry

end maddie_milk_usage_l193_193055


namespace a_n_formula_l193_193121

def sequence_a (n : ℕ) : ℕ :=
  match n with
  | 0 => sorry  -- since the sequence is defined for positive integers, we can ignore n = 0
  | n + 1 => match n with
             | 0 => 1  -- a_1 = 1
             | _ => 2 * 3 ^ (n - 1)

def sum_S (n : ℕ) : ℕ :=
  sorry -- since this proof does not require explicitly defining this function's behavior

theorem a_n_formula (n : ℕ) (h : n > 0) :
  if n = 1 then sequence_a n = 1 else sequence_a n = 2 * 3 ^ (n - 2) :=
by
  cases n
  . -- Case n = 0, contradicting n > 0
    contradiction
  . cases n
    . simp [sequence_a], -- n = 1 case
    . sorry -- for n ≥ 2

end a_n_formula_l193_193121


namespace min_theta_value_l193_193416

noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x) + cos (ω * x)

theorem min_theta_value
  (ω > 0) 
  (h_period : ∀ x : ℝ, f ω x + |f ω x| = f ω (x + π)) 
  (h_min : ∀ x : ℝ, ∃ θ > 0, f ω x ≥ f ω θ) :
  ∃ θ > 0, θ = 5 * π / 8 :=
sorry

end min_theta_value_l193_193416


namespace problem_solution_l193_193489

noncomputable def problem_statement (n : ℕ) (k : ℕ) (a : Fin k → Fin n) : Prop :=
  ∀ i : Fin (k - 1), n ∣ a i * (a (Fin.succ i) - 1)

theorem problem_solution (n k : ℕ) (a : Fin k → Fin n)
  (h1 : 0 < n)
  (h2 : 2 ≤ k)
  (h3 : ∀ i j, i ≠ j → a i ≠ a j)
  (h4 : problem_statement n k a) :
  ¬ (n ∣ a (Fin.last k) * (a 0 - 1)) :=
by
  sorry

end problem_solution_l193_193489


namespace minimum_distance_between_projections_l193_193511

-- Define the points A, B, C on the plane
variables (A B C : Point) [IsAcuteAngle A B C]

-- Define the point M on the line AB
variables (M : Point) (M_on_AB : ∃ k : ℝ, A + k * (B - A) = M)

-- Define the projections P and Q
variables (P : Point) (P_proj_AC : ∃ l : ℝ, A + l * (C - A) = P ∧ M - P ⟂ (C - A))
variables (Q : Point) (Q_proj_BC : ∃ m : ℝ, B + m * (C - B) = Q ∧ M - Q ⟂ (C - B))

-- Define the height h from A to BC
variables (h : ℝ) (h_def : h = dist (orthocenter A B C) (foot (C - B) (A - B)))

-- Define the angle ACB
variables (angle_ACB : ℝ) (angle_ACB_def : angle A C B = angle_ACB)

-- Formulate the proof problem
theorem minimum_distance_between_projections :
  ∃ min_distance : ℝ, 
  min_distance = h * sin angle_ACB :=
sorry

end minimum_distance_between_projections_l193_193511


namespace reflection_matrix_correct_l193_193477

def matrix_S : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![1 / 3, 4 / 3, 0],
    ![-2 / 3, 1 / 3, 2 / 3],
    ![4 / 3, -2 / 3, 1 / 3]
  ]

def plane_normal : Vector3 ℚ := ⟨2, -1, 1⟩

theorem reflection_matrix_correct (v : Vector3 ℚ) :
  let reflected_v := 2 * (plane_normal.dot_product v / plane_normal.dot_product plane_normal) • plane_normal - v in
  matrix_S.mul_vec v = reflected_v := sorry

end reflection_matrix_correct_l193_193477


namespace pyramid_base_side_length_l193_193090

theorem pyramid_base_side_length
  (lateral_face_area : Real)
  (slant_height : Real)
  (s : Real)
  (h_lateral_face_area : lateral_face_area = 200)
  (h_slant_height : slant_height = 40)
  (h_area_formula : lateral_face_area = 0.5 * s * slant_height) :
  s = 10 :=
by
  sorry

end pyramid_base_side_length_l193_193090


namespace fox_initial_coins_l193_193963

theorem fox_initial_coins (x : ℤ) :
  let bonus := 10
  let toll := 50
  let after_first_crossing := 2 * (x + bonus) - toll
  let after_second_crossing := 2 * after_first_crossing - toll
  let after_third_crossing := 2 * after_second_crossing - toll
  let after_fourth_crossing := 2 * after_third_crossing - toll
  after_fourth_crossing = 0 ↔ x = 37 :=
by
  let bonus := 10
  let toll := 50
  let after_first_crossing := 2 * (x + bonus) - toll
  let after_second_crossing := 2 * after_first_crossing - toll
  let after_third_crossing := 2 * after_second_crossing - toll
  let after_fourth_crossing := 2 * after_third_crossing - toll
  calc
    after_fourth_crossing
    = 2 * (2 * (2 * (2 * (x + bonus) - toll) - toll) - toll) - toll : by sorry
    ... = 16 * x - 590 : by sorry
    ... = 0 ↔ x = 37 : by sorry
  sorry

end fox_initial_coins_l193_193963


namespace volume_tetrahedron_correct_l193_193539

noncomputable def volume_of_regular_tetrahedron (s : ℝ) : ℝ :=
  (s^3 * real.sqrt 2) / 12

theorem volume_tetrahedron_correct :
  volume_of_regular_tetrahedron (7 * real.sqrt 2) = 228.667 :=
by sorry

end volume_tetrahedron_correct_l193_193539


namespace smaller_angle_3_15_l193_193598

theorem smaller_angle_3_15 :
  let minute_hand_degrees := 15 * (360 / 60)
  let hour_hand_degrees := 90 + (30 / 60) * 15
  abs (hour_hand_degrees - minute_hand_degrees) = 7.5 :=
by
  let minute_hand_degrees := 15 * (360 / 60)
  let hour_hand_degrees := 90 + (30 / 60) * 15
  have h1 : minute_hand_degrees = 90 := by sorry
  have h2 : hour_hand_degrees = 97.5 := by sorry
  have h3 : abs (hour_hand_degrees - minute_hand_degrees) = 7.5 := by sorry
  exact h3

end smaller_angle_3_15_l193_193598


namespace alternate_binomial_sum_l193_193330

theorem alternate_binomial_sum :
  ∑ k in (range 51).map (λ n => 2 * n), ite (even k) (↑(binom 101 k) * (-1)^(k/2)) 0 = -2^50 :=
by
  sorry

end alternate_binomial_sum_l193_193330


namespace rotation_A1_l193_193776

/-- Rotation of a point (x, y) by θ degrees clockwise around the origin O is given by:
 (x', y') = (x * cos(θ) + y * sin(θ), -x * sin(θ) + y * cos(θ))
 Given point A(-1, 1) and θ = 45 degrees, we must prove the coordinates after rotation are (0, sqrt(2)). --/
theorem rotation_A1 :
  let θ := (π / 4 : ℝ) in
  let (x, y) := (-1 : ℝ, 1 : ℝ) in
  let x' := x * Real.cos θ + y * Real.sin θ in
  let y' := -x * Real.sin θ + y * Real.cos θ in
  (x', y') = (0, Real.sqrt 2) :=
by
  sorry

end rotation_A1_l193_193776


namespace num_six_digit_unique_digits_num_six_digit_with_four_odd_l193_193779

-- Proof theorems
theorem num_six_digit_unique_digits : 
  (card { n : ℕ | ∃ a b c d e f, (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ 
                      (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ 
                      (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ 
                      (d ≠ e) ∧ (d ≠ f) ∧ 
                      (e ≠ f)  ∧ {a,b,c,d,e,f} ⊆ {0,1,2,3,4,5,6,7,8,9} ∧ a ≠ 0 ∧ 
                      n = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f}) = 136080 := sorry

theorem num_six_digit_with_four_odd : 
  (card { n : ℕ | ∃ a b c d e f, (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ 
                      (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ 
                      (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ 
                      (d ≠ e) ∧ (d ≠ f) ∧ 
                      (e ≠ f)  ∧ {a,b,c,d,e,f} ⊆ {0,1,2,3,4,5,6,7,8,9} ∧ a ≠ 0 ∧ 
                      (countp (λ x, x % 2 = 1) [a,b,c,d,e,f] = 4) ∧ 
                      n = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f}) = 33600 := sorry

end num_six_digit_unique_digits_num_six_digit_with_four_odd_l193_193779


namespace find_number_l193_193307

theorem find_number (x : ℤ) (h : (((55 + x) / 7 + 40) * 5 = 555)) : x = 442 :=
sorry

end find_number_l193_193307


namespace total_course_selection_schemes_l193_193271

theorem total_course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose_two :=
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 1)
  let choose_three :=
    (Nat.choose PE_courses 2) * (Nat.choose Art_courses 1) +
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 2)
  total_courses = 8
  ∧ (choose_two + choose_three = 64) :=
by
  sorry

end total_course_selection_schemes_l193_193271


namespace sum_x_coords_of_Q4_l193_193622

open Real

noncomputable def x_coords_sum_Q1 := 120
def vertices_Q1 : ℕ := 40
def x_sum_midpoints (x_sum : ℝ) : ℝ := x_sum

theorem sum_x_coords_of_Q4 :
  let Q1_x_sum := x_coords_sum_Q1 in
  let Q2_x_sum := x_sum_midpoints Q1_x_sum in
  let Q3_x_sum := x_sum_midpoints Q2_x_sum in
  let Q4_x_sum := x_sum_midpoints Q3_x_sum in
  Q4_x_sum = 120 :=
by sorry

end sum_x_coords_of_Q4_l193_193622


namespace AX_tangent_to_ABC_circumcircle_l193_193851

theorem AX_tangent_to_ABC_circumcircle 
    (A B C D P Q X : Point)
    (h1 : ∃ (triangle : Triangle), triangle = ⟨A, B, C⟩ ∧ B.distance C > A.distance C)
    (h2 : ∃ (angle_bisector : AngleBisector), angle_bisector.bisects (angle A B C) ∧ angle_bisector.intersects BC D)
    (h3 : ∃ (circle_b : Circle), circle_b.diameter B D ∧ circle_b.intersects (circumcircle A B C) P ∧ P ≠ B)
    (h4 : ∃ (circle_c : Circle), circle_c.diameter C D ∧ circle_c.intersects (circumcircle A B C) Q ∧ Q ≠ C)
    (h5 : PQ_line_intersects_BC : Line (P, Q) ∩ Line (B, C) = {X})
    : IsTangent (Line (A, X), circumcircle A B C) at A :=
sorry

end AX_tangent_to_ABC_circumcircle_l193_193851


namespace simplify_expression_l193_193525

noncomputable def term1 : ℝ := 3 / (Real.sqrt 2 + 2)
noncomputable def term2 : ℝ := 4 / (Real.sqrt 5 - 2)
noncomputable def simplifiedExpression : ℝ := 1 / (term1 + term2)
noncomputable def finalExpression : ℝ := 1 / (11 + 4 * Real.sqrt 5 - 3 * Real.sqrt 2 / 2)

theorem simplify_expression : simplifiedExpression = finalExpression := by
  sorry

end simplify_expression_l193_193525


namespace age_of_youngest_l193_193123

theorem age_of_youngest
  (y : ℕ)
  (h1 : 4 * 25 = y + (y + 2) + (y + 7) + (y + 11)) : y = 20 :=
by
  sorry

end age_of_youngest_l193_193123


namespace round_table_people_count_l193_193075

noncomputable def total_people_seated (G B : ℕ) : Prop :=
  (G = 7 + 12) ∧ (0.75 * B = 12) ∧ (G + B = 35)

theorem round_table_people_count
  (G B : ℕ)
  (h1 : G = 7 + 12)
  (h2 : 0.75 * B = 12) :
  G + B = 35 :=
begin
  rw [h1],
  have h3 : G = 19, from h1,
  have h4 : 12 / 0.75 = 16, by sorry,
  have h5 : B = 16, from (eq.symm (div_eq_iff_mul_eq (ne_of_gt (by norm_num)).symm).mpr h2),
  rw [h3, h5],
  exact rfl,
end

end round_table_people_count_l193_193075


namespace find_g_l193_193488

theorem find_g (f g : Polynomial ℝ) (hg : g ≠ 0) (hf : f ≠ 0) (hfg : eval (g : ℝ → ℝ) = (eval f) * (eval g)) (h3 : eval g 3 = 19) :
  g = λ x, x + 16 :=
sorry

end find_g_l193_193488


namespace math_proof_l193_193722

variable (f : ℝ → ℝ) (a b c : ℝ)
variable (h1 : f = λ x, |2 * x - 1|)
variable (h2 : a < b) (h3 : b < c)
variable (h4 : f a > f c) (h5 : f c > f b)

theorem math_proof : 2 - a < 2 * c := by
  sorry

end math_proof_l193_193722


namespace novel_cost_l193_193844

-- Given conditions
variable (N : ℕ) -- cost of the novel
variable (lunch_cost : ℕ) -- cost of lunch

-- Conditions
axiom gift_amount : N + lunch_cost + 29 = 50
axiom lunch_cost_eq : lunch_cost = 2 * N

-- Question and answer tuple as a theorem
theorem novel_cost : N = 7 := 
by
  sorry -- Proof estaps are to be filled in.

end novel_cost_l193_193844


namespace hyperbola_asymptote_l193_193419

variable (a b : ℝ)
variable (x y : ℝ)

/- Definition of hyperbola -/
def hyperbola (x y a b : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1

/- Definition of parabola -/
def parabola (x y : ℝ) := y^2 = 8 * x

/- Definition of point P -/
variable (P : ℝ × ℝ)
variable (F : ℝ × ℝ)
variable (m n : ℝ)

/- Hypotheses -/
hypothesis ha_pos : a > 0
hypothesis hb_pos : b > 0
hypothesis h_focus : F = (2, 0)
hypothesis h_intersection : P = (3, n) ∨ P = (3, -n)
hypothesis h_distance_PF : |P.1 - F.1| = 5

/- Theorem stating that the equation of the hyperbola’s asymptote is √3 x ± y = 0 -/
theorem hyperbola_asymptote : hyperbola x y a b → parabola x y → 
  (a^2 = 1) → (b^2 = 3) → (√3 * x = y ∨ √3 * x = -y) :=
sorry

end hyperbola_asymptote_l193_193419


namespace sum_of_digits_of_greatest_prime_divisor_l193_193147

theorem sum_of_digits_of_greatest_prime_divisor (n : ℕ) (h : n = 16385) : 
  let p := 3277 in -- greatest prime divisor
  let sum_digits := 3 + 2 + 7 + 7 in
  ∑ d in p.digits 10, d = sum_digits := 
by 
  have divisor_prime : Nat.Prime 3277 := sorry -- proof that 3277 is prime
  have divides : 3277 ∣ 16385 := sorry -- proof that 3277 divides 16385
  sorry -- completing the proof

end sum_of_digits_of_greatest_prime_divisor_l193_193147


namespace course_selection_schemes_l193_193198

theorem course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose (n k : ℕ) := nat.choose n k
  
  (choose 4 1 * choose 4 1 + 
   choose 4 2 * choose 4 1 + 
   choose 4 1 * choose 4 2) = 64 := 
by {
  -- calculating the combinations
  have comb_2_courses : choose 4 1 * choose 4 1 = 16 := by sorry,
  have comb_2_pe_1_art : choose 4 2 * choose 4 1 = 24 := by sorry,
  have comb_1_pe_2_art : choose 4 1 * choose 4 2 = 24 := by sorry,
  have total_comb : 16 + 24 + 24 = 64 := by {
    simp [comb_2_courses, comb_2_pe_1_art, comb_1_pe_2_art]
  },
  exact total_comb
}

end course_selection_schemes_l193_193198


namespace find_C_coordinates_l193_193066

noncomputable def pointC_coordinates : Prop :=
  let A : (ℝ × ℝ) := (-2, 1)
  let B : (ℝ × ℝ) := (4, 9)
  ∃ C : (ℝ × ℝ), 
    (dist (A.1, A.2) (C.1, C.2) = 2 * dist (B.1, B.2) (C.1, C.2)) ∧ 
    C = (2, 19 / 3)

theorem find_C_coordinates : pointC_coordinates :=
  sorry

end find_C_coordinates_l193_193066


namespace prob_divisible_by_3_of_three_digits_l193_193574

-- Define the set of digits available
def digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Three digits are to be chosen from this set
def choose_three_digits (s : Finset ℕ) : Finset (Finset ℕ) := s.powerset.filter (λ x, x.card = 3)

-- Define the property of the sum of digits being divisible by 3
def divisible_by_3 (s : Finset ℕ) : Prop := s.sum id % 3 = 0

-- Total combinations of choosing 3 out of 9 digits
def total_combinations : ℕ := (digits.card.choose 3)

-- Valid combinations where sum of digits is divisible by 3
def valid_combinations : Finset (Finset ℕ) := (choose_three_digits digits).filter divisible_by_3

-- Finally, the probability of a three-digit number being divisible by 3
def probability : ℕ × ℕ := (valid_combinations.card, total_combinations)

theorem prob_divisible_by_3_of_three_digits :
  probability = (5, 14) :=
by
  -- Proof to be filled
  sorry

end prob_divisible_by_3_of_three_digits_l193_193574


namespace find_d_vector_l193_193937

theorem find_d_vector (x y t : ℝ) (v d : ℝ × ℝ)
  (hline : y = (5 * x - 7) / 2)
  (hparam : ∃ t : ℝ, (x, y) = (4, 2) + t • d)
  (hdist : ∀ {x : ℝ}, x ≥ 4 → dist (x, (5 * x - 7) / 2) (4, 2) = t) :
  d = (2 / Real.sqrt 29, 5 / Real.sqrt 29) := 
sorry

end find_d_vector_l193_193937


namespace tangent_line_at_origin_fx_positive_for_a2_max_value_on_interval_l193_193871

noncomputable theory

-- Part (1):
def f (x : ℝ) : ℝ := Real.exp x - 2 * x

def point : ℝ × ℝ := (0, f 0)

theorem tangent_line_at_origin :
  (∃ m b : ℝ, m = -1 ∧ b = 1 ∧ ∀ x : ℝ, (x, m * x + b) = (x, f x) ↔ x = 0) :=
sorry

-- Part (2):
theorem fx_positive_for_a2 :
  ∀ x : ℝ, 0 < Real.exp x - 2 * x :=
sorry

-- Part (3):
def f_a (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x

theorem max_value_on_interval (a : ℝ) (h : 1 < a) :
  ∃ M : ℝ, M = Real.exp a - a ^ 2 ∧ ∀ x : ℝ, 0 ≤ x ∧ x ≤ a → f_a x a ≤ M :=
sorry

end tangent_line_at_origin_fx_positive_for_a2_max_value_on_interval_l193_193871


namespace percentage_apples_is_50_percent_l193_193581

-- Definitions for the given conditions
def initial_apples : ℕ := 10
def initial_oranges : ℕ := 5
def added_oranges : ℕ := 5

-- Defining the proof problem
theorem percentage_apples_is_50_percent :
  let total_fruits := initial_apples + initial_oranges + added_oranges in
  let apples_percentage := (initial_apples * 100) / total_fruits in
  apples_percentage = 50 :=
by
  sorry

end percentage_apples_is_50_percent_l193_193581


namespace cube_wire_length_l193_193611

theorem cube_wire_length (edge_length : ℕ) (h : edge_length = 13) : 
  12 * edge_length = 156 :=
by
  rw [h]
  norm_num

end cube_wire_length_l193_193611


namespace price_reduction_required_l193_193631

variable (x : ℝ)
variable (profit_per_piece : ℝ := 40)
variable (initial_sales : ℝ := 20)
variable (additional_sales_per_unit_reduction : ℝ := 2)
variable (desired_profit : ℝ := 1200)

theorem price_reduction_required :
  (profit_per_piece - x) * (initial_sales + additional_sales_per_unit_reduction * x) = desired_profit → x = 20 :=
sorry

end price_reduction_required_l193_193631


namespace course_selection_schemes_count_l193_193235

-- Definitions based on the conditions
def num_physical_education_courses : ℕ := 4
def num_art_courses : ℕ := 4

-- Required to choose 2 or 3 courses, with at least one from each category
def valid_selection_cases : list (ℕ × ℕ) := [(1, 1), (1, 2), (2, 1)]

-- Calculate the number of ways to choose k courses from n courses using combination formula
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Calculate the total number of different course selection schemes
def total_course_selection_schemes : ℕ :=
  let case1 := C num_physical_education_courses 1 * C num_art_courses 1
  let case2a := C num_physical_education_courses 2 * C num_art_courses 1
  let case2b := C num_physical_education_courses 1 * C num_art_courses 2
  case1 + case2a + case2b

-- The assertion that the total number of different course selection schemes is 64
theorem course_selection_schemes_count : total_course_selection_schemes = 64 := by
  -- sorry placeholder for proof
  sorry

end course_selection_schemes_count_l193_193235


namespace jennifer_initial_money_eq_120_l193_193467

variable (X : ℚ) -- Define X as a rational number

-- Declare the conditions as variables.
variable (sandwich_expense museum_expense book_expense leftover_money : ℚ)

-- Set definitions based on the problem conditions.
def sandwich_expense := (1 / 5) * X
def museum_expense := (1 / 6) * X
def book_expense := (1 / 2) * X
def leftover_money := 16

-- Define the main theorem to prove.
theorem jennifer_initial_money_eq_120 
  (h: X - (sandwich_expense + museum_expense + book_expense) = leftover_money) : 
  X = 120 :=
by
  -- Proof omitted, add "sorry" to indicate the proof is required but not provided.
  sorry

end jennifer_initial_money_eq_120_l193_193467


namespace course_selection_schemes_count_l193_193242

-- Definitions based on the conditions
def num_physical_education_courses : ℕ := 4
def num_art_courses : ℕ := 4

-- Required to choose 2 or 3 courses, with at least one from each category
def valid_selection_cases : list (ℕ × ℕ) := [(1, 1), (1, 2), (2, 1)]

-- Calculate the number of ways to choose k courses from n courses using combination formula
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Calculate the total number of different course selection schemes
def total_course_selection_schemes : ℕ :=
  let case1 := C num_physical_education_courses 1 * C num_art_courses 1
  let case2a := C num_physical_education_courses 2 * C num_art_courses 1
  let case2b := C num_physical_education_courses 1 * C num_art_courses 2
  case1 + case2a + case2b

-- The assertion that the total number of different course selection schemes is 64
theorem course_selection_schemes_count : total_course_selection_schemes = 64 := by
  -- sorry placeholder for proof
  sorry

end course_selection_schemes_count_l193_193242


namespace triangle_sum_of_sides_correct_l193_193003

noncomputable def triangle_sum_of_sides : ℝ :=
  let angle_A := 60
  let angle_C := 30
  let side_a := 8 * Real.sqrt 3
  let sqrt3_approx := 1.732
  let side_AD := 12
  let side_BD := 4 * Real.sqrt 3
  side_AD + side_a -- which relates to sum of AC + BC

theorem triangle_sum_of_sides_correct :
  (Real.floor ((triangle_sum_of_sides * 10) + 0.5) / 10) = 25.9 :=
by
  sorry

end triangle_sum_of_sides_correct_l193_193003


namespace b_10_eq_l193_193682

/-- Define a sequence where b_1 = 1, b_2 = 1, and for all n >= 3,
    b_n = (b_(n - 1) + 2) / b_(n - 2) -/
def b : ℕ → ℚ
| 1 := 1
| 2 := 1
| (n + 3) := (b (n + 2) + 2) / b (n + 1)

/-- Prove that b_10 is equal to 149 / 51 -/
theorem b_10_eq : b 10 = 149 / 51 :=
sorry

end b_10_eq_l193_193682


namespace range_of_a_l193_193410

noncomputable def tangent_line_exists (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, (cos x₀ + sqrt 3 * sin x₀ + a) = -2

theorem range_of_a (a : ℝ) : tangent_line_exists a → -4 ≤ a ∧ a ≤ 0 := sorry

end range_of_a_l193_193410


namespace positive_solution_count_l193_193361

theorem positive_solution_count :
  ∀ x : ℝ, x > 0 → (cos (arcsin (tan (arccos x))) = x) → x = 1 :=
by
  sorry

end positive_solution_count_l193_193361


namespace percentage_apples_basket_l193_193585

theorem percentage_apples_basket :
  let initial_apples := 10
  let initial_oranges := 5
  let added_oranges := 5
  let total_apples := initial_apples
  let total_oranges := initial_oranges + added_oranges
  let total_fruits := total_apples + total_oranges
  (total_apples / total_fruits) * 100 = 50 :=
by
  sorry

end percentage_apples_basket_l193_193585


namespace candidate_B_should_be_hired_l193_193629

structure CandidateScores where
  listening : ℝ
  speaking  : ℝ
  reading   : ℝ
  writing   : ℝ

def calculate_weighted_average (scores : CandidateScores) (weights : CandidateScores) : ℝ :=
  (scores.listening * weights.listening + scores.speaking * weights.speaking +
    scores.reading * weights.reading + scores.writing * weights.writing) /
  (weights.listening + weights.speaking + weights.reading + weights.writing)

def weights : CandidateScores := {
  listening := 2,
  speaking := 1,
  reading := 3,
  writing := 4
}

def candidate_A_scores : CandidateScores := {
  listening := 85,
  speaking := 78,
  reading := 85,
  writing := 73
}

def candidate_B_scores : CandidateScores := {
  listening := 73,
  speaking := 80,
  reading := 82,
  writing := 83
}

theorem candidate_B_should_be_hired : 
  calculate_weighted_average candidate_B_scores weights > calculate_weighted_average candidate_A_scores weights :=
  sorry

end candidate_B_should_be_hired_l193_193629


namespace length_chord_AB_hyperbola_equation_correct_l193_193176

-- Problem 1 definition and statement
noncomputable def ellipse : set (ℝ × ℝ) := { p | (p.1)^2 / 4 + (p.2)^2 = 1 }

noncomputable def line_through_focus : set (ℝ × ℝ) := { p | p.2 = p.1 - sqrt 3 }

theorem length_chord_AB :
  ∃ A B : ℝ × ℝ,
    A ∈ (ellipse ∩ line_through_focus) ∧
    B ∈ (ellipse ∩ line_through_focus) ∧
    A ≠ B ∧
    (A.1 + B.1 = 8 * sqrt 3 / 5 ∧ A.1 * B.1 = 8 / 5) ∧
    sqrt (1 + 1^2) * sqrt ((A.1 + B.1)^2 - 4 * A.1 * B.1) = 8 / 5 := sorry

-- Problem 2 definition and statement
theorem hyperbola_equation_correct :
  ∃ λ : ℝ,
    (λ ≠ 0) ∧
    (let eq := (λ : ℝ) in
      ∀ x y,
        (x, y) ∈ { p : ℝ × ℝ | p.1^2 / 9 - p.2^2 / 16 = eq } →
        (x, y) = (-3, 2 * sqrt 3)) →
    λ = 1 / 4 →
    ∀ x y,
      (x, y) ∈ { p : ℝ × ℝ | 4 * x^2 / 9 - y^2 / 4 = 1 } := sorry

end length_chord_AB_hyperbola_equation_correct_l193_193176


namespace shaded_area_ratio_l193_193013

theorem shaded_area_ratio (x : ℝ) (h1 : x > 0) (AC_CB : AC = 3 * x) (CB_x : CB = x) (CD_perp_AB : CD ⊥ AB) :
  let CD := sqrt (2 * x * x)
  let shaded_area := 0.75 * π * (x * x)
  let circle_area := 2 * π * (x * x)
  shaded_area / circle_area = 3 / 8 :=
by
  sorry

end shaded_area_ratio_l193_193013


namespace circle_angle_relation_l193_193479

theorem circle_angle_relation
  (C : Circle)
  (O : Point)
  (A : Point)
  (B C D E : Point)
  (hA_outside : ¬ (A ∈ C))
  (hB_on_C : B ∈ C)
  (hC_on_C : C ∈ C)
  (hD_on_C : D ∈ C)
  (hE_on_C : E ∈ C)
  (hA_to_BC : LineThrough A B ∧ LineThrough A C)
  (hA_to_DE : LineThrough A D ∧ LineThrough A E)
  (h_order_BC : Between A B C)
  (h_order_DE : Between A D E)
  : ∠ C A E = (∠ C O E - ∠ B O D) / 2 :=
by
  sorry

end circle_angle_relation_l193_193479


namespace set_intersection_l193_193775

   -- Define set A
   def A : Set ℝ := {x : ℝ | (x - 3) / (x + 1) ≥ 0 }
   
   -- Define set B
   def B : Set ℝ := {x : ℝ | Real.log x / Real.log 2 < 2}

   -- Define the relative complement of A in the real numbers
   def complement_R (A : Set ℝ) : Set ℝ := {x : ℝ | ¬ (A x)}

   -- The main statement that needs to be proven
   theorem set_intersection :
     (complement_R A) ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by
     sorry
   
end set_intersection_l193_193775


namespace divide_square_into_n_squares_l193_193515

theorem divide_square_into_n_squares (n : ℕ) (h : n ≥ 6) : ∃ squares : finset (set ℝ × set ℝ), squares.card = n ∧ (∀ s ∈ squares, ∃ a b, s = (Icc a (a + 1) ×ˢ Icc b (b + 1))) :=
by sorry

end divide_square_into_n_squares_l193_193515


namespace solve_inequality_l193_193899

theorem solve_inequality (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  (3 - 1 / (3 * x + 4) < 5) ↔ (-4 / 3 < x) :=
by
  sorry

end solve_inequality_l193_193899


namespace rectangle_to_square_area_ratio_is_24_25_l193_193924

noncomputable def rectangle_to_square_area_ratio
  (s : ℝ) -- length of side of square S
  (longer_side : ℝ := 1.2 * s) -- longer side of rectangle R
  (shorter_side : ℝ := 0.8 * s) -- shorter side of rectangle R
  (area_R : ℝ := longer_side * shorter_side) -- area of rectangle R
  (area_S : ℝ := s^2) -- area of square S
  : ℝ := 
  area_R / area_S

theorem rectangle_to_square_area_ratio_is_24_25 
  (s : ℝ)
  : rectangle_to_square_area_ratio s = 24 / 25 :=
by 
  sorry

end rectangle_to_square_area_ratio_is_24_25_l193_193924


namespace Rachelle_pennies_l193_193062

/-- One afternoon, Rachelle, Gretchen, and Rocky threw pennies into the fountain and made wishes.
    Rachelle threw some pennies into the fountain. Gretchen threw half as many pennies as Rachelle, and
    Rocky threw one-third as many pennies as Gretchen. The total number of pennies thrown into the fountain
    by the three of them was 300. How many pennies did Rachelle throw into the fountain?
-/
theorem Rachelle_pennies 
  (R G K : ℕ) 
  (h1 : G = R / 2) 
  (h2 : K = G / 3)
  (h3 : R + G + K = 300) : R = 180 :=
begin
  sorry
end

end Rachelle_pennies_l193_193062


namespace sine_cosine_obtuse_angle_l193_193796

theorem sine_cosine_obtuse_angle :
  ∀ P : (ℝ × ℝ), P = (Real.sin 2, Real.cos 2) → (Real.sin 2 > 0) ∧ (Real.cos 2 < 0) → 
  (P.1 > 0) ∧ (P.2 < 0) :=
by
  sorry

end sine_cosine_obtuse_angle_l193_193796


namespace find_sin_theta_l193_193853

variables {a b c : Vector3}
variables (θ : real)

-- Conditions
def a_nonzero : Prop := a ≠ 0
def b_nonzero : Prop := b ≠ 0
def c_nonzero : Prop := c ≠ 0
def not_parallel (u v : Vector3) : Prop := ¬∃ (k : real), u = k • v

def conditions : Prop :=
  a_nonzero a ∧ b_nonzero b ∧ c_nonzero c ∧
  not_parallel a b ∧ not_parallel b c ∧ not_parallel a c ∧
  (a × b) × c = (1 / 4) * ∥b∥ * ∥c∥ • a

-- Proof problem
theorem find_sin_theta (h : conditions a b c): sin θ = (sqrt 15) / 4 := sorry

end find_sin_theta_l193_193853


namespace monotonic_decreasing_interval_l193_193549

noncomputable def decreasing_interval : Set ℝ :=
  { x | 0 < x ∧ x < exp (-2) }

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (x + x * log x) < (x + x * (log x + 1)) → x ∈ decreasing_interval :=
by
  sorry

end monotonic_decreasing_interval_l193_193549


namespace find_y_value_l193_193120

-- Define the points A and B
structure Point where
  x : ℝ
  y : ℝ

-- Define the slope function based on points A and B
def slope (A B : Point) : ℝ :=
  (B.y - A.y) / (B.x - A.x)

-- Define the tangent function of an angle in degrees
def tan_deg (θ : ℝ) : ℝ := Real.tan (θ * Real.pi / 180)

-- The given problem conditions
def A : Point := { x := 3, y := _ }
def B : Point := { x := 2, y := 0 }
def θ : ℝ := 120

-- Noncomputable definition for A.y as it's what we need to find
noncomputable def A_y := (-Real.sqrt 3)

-- The proof statement for the problem
theorem find_y_value : A.y = A_y :=
  by
    sorry

end find_y_value_l193_193120


namespace average_waiting_time_40_seconds_l193_193638

theorem average_waiting_time_40_seconds :
  let G := event "Light is green"
  let P_G := 1 / 3
  let P_not_G := 2 / 3
  let E_T_given_G := 0
  let E_T_given_not_G := (0 + 2) / 2
  let E_T := E_T_given_G * P_G + E_T_given_not_G * P_not_G
  E_T * 60 = 40 := by
  sorry

end average_waiting_time_40_seconds_l193_193638


namespace product_of_solutions_l193_193706

-- Definitions based on given conditions
def equation (x : ℝ) : Prop := |x| = 3 * (|x| - 2)

-- Statement of the proof problem
theorem product_of_solutions : ∃ x1 x2 : ℝ, equation x1 ∧ equation x2 ∧ x1 * x2 = -9 := by
  sorry

end product_of_solutions_l193_193706


namespace area_ratio_l193_193919

-- Define the problem conditions
def Square (s : ℝ) := s > 0
def Rectangle (longer shorter : ℝ) := longer = 1.2 * shorter ∧ shorter = 0.8 * shorter

-- Define a function to calculate the area of square
def area_square (s : ℝ) : ℝ := s * s

-- Define a function to calculate the area of rectangle
def area_rectangle (longer shorter : ℝ) : ℝ := longer * shorter

-- State the proof problem
theorem area_ratio (s : ℝ) (h_square : Square s) :
  let longer := 1.2 * s
  let shorter := 0.8 * s
  area_rectangle longer shorter / area_square s = 24 / 25 :=
by
  sorry

end area_ratio_l193_193919


namespace difference_of_interchanged_digits_l193_193097

theorem difference_of_interchanged_digits (X Y : ℕ) (h1 : X - Y = 3) :
  (10 * X + Y) - (10 * Y + X) = 27 := by
  sorry

end difference_of_interchanged_digits_l193_193097


namespace domain_of_function_proof_l193_193536

def domain_of_function := {x : ℝ | x ≥ 0 ∧ x ≠ 2 }

theorem domain_of_function_proof : 
  (domain_of_function = {x : ℝ | (x ∈ Ico 0 2) ∨ (x ∈ Ioo 2 (∞) )}) :=
sorry

end domain_of_function_proof_l193_193536


namespace inequality_solution_l193_193897

theorem inequality_solution (x : ℝ) : 
  (3 - (1 / (3 * x + 4)) < 5) ↔ (x < -4 / 3) := 
by
  sorry

end inequality_solution_l193_193897


namespace find_ab_l193_193153

theorem find_ab (a b : ℕ) (h1 : 1 <= a) (h2 : a < 10) (h3 : 0 <= b) (h4 : b < 10) (h5 : 66 * ((1 : ℝ) + ((10 * a + b : ℕ) / 100) - (↑(10 * a + b) / 99)) = 0.5) : 10 * a + b = 75 :=
by
  sorry

end find_ab_l193_193153


namespace num_perfect_square_factors_of_51200_l193_193778

/-- 51200 factorizes into prime factors 2 and 5 as 2^10 * 5^3.
A perfect square factor must have even exponents for these prime factors.
The possible even exponents for 2 (from 0 to 10) are 0, 2, 4, 6, 8, 10 (6 choices).
The possible even exponents for 5 (from 0 to 3) are 0, 2 (2 choices).
The total number of perfect square factors is 6 * 2 = 12. -/
theorem num_perfect_square_factors_of_51200 : 
  (∃ (factors : ℕ → ℕ), (factors 51200) = 2^10 * 5^3 ∧ 
    ∀ (factor : ℕ), factor ∣ 51200 → (∃ (a b : ℕ), factor = 2^a * 5^b ∧ a % 2 = 0 ∧ b % 2 = 0) →
    #{factor | factor ∣ 51200 ∧ 
              ∃ a b : ℕ, factor = 2^a * 5^b ∧ a % 2 = 0 ∧ b % 2 = 0} = 12) :=
by
  sorry

end num_perfect_square_factors_of_51200_l193_193778


namespace distinct_solutions_subtraction_eq_two_l193_193491

theorem distinct_solutions_subtraction_eq_two :
  ∃ p q : ℝ, (p ≠ q) ∧ (p > q) ∧ ((6 * p - 18) / (p^2 + 4 * p - 21) = p + 3) ∧ ((6 * q - 18) / (q^2 + 4 * q - 21) = q + 3) ∧ (p - q = 2) :=
by
  have p := -3
  have q := -5
  exists p, q
  sorry

end distinct_solutions_subtraction_eq_two_l193_193491


namespace probability_AB_not_same_intersection_l193_193693

theorem probability_AB_not_same_intersection :
  let officers := {A, B, C} in
  let intersections := {i1, i2} in
  (∀ o ∈ officers, ∃ i ∈ intersections, true) → -- At least one officer at each intersection
  (∃ p : ℚ, (p = 2/3) ∧ p = 1 - (n : ℚ) / (m : ℚ)) :=
by
  sorry

end probability_AB_not_same_intersection_l193_193693


namespace domain_f_when_a_is_minus_1_range_of_a_given_x_in_interval_range_of_b_given_conditions_l193_193763

-- Domain of f(x) when a = -1
theorem domain_f_when_a_is_minus_1 : 
  ∀ x, f (log (2 : Real) (1 + 2^x - 4^x)) → x < 0 := sorry

-- Range of a given x in (-∞, 1]
theorem range_of_a_given_x_in_interval :
  ∀ x ∈ (-∞, 1], ∃ a, 1 + 2^x + a * (4^x + 1) > 0 → a > -3/5 := sorry

-- Range of b given a = -1/2 and no intersection within [0, 1]
theorem range_of_b_given_conditions :
  ∀ b, (∀ x ∈ [0,1], log (2 : Real) (1 + 2^x - (1/2)*(4^x + 1)) ≠ x + b) → b < -2 ∨ b > 0 := sorry

end domain_f_when_a_is_minus_1_range_of_a_given_x_in_interval_range_of_b_given_conditions_l193_193763


namespace gerald_toy_cars_left_l193_193718

noncomputable def gerald_initial_toys : ℕ := 125
noncomputable def donation_percentage : ℝ := 0.35
noncomputable def new_toys_bought : ℕ := 15
noncomputable def brother_toys : ℕ := 48

theorem gerald_toy_cars_left (gerald_initial_toys : ℕ) (donation_percentage : ℝ) (new_toys_bought : ℕ) (brother_toys : ℕ) : ℕ :=
  let donated_toys := nat.floor (donation_percentage * gerald_initial_toys)
  let remaining_toys := gerald_initial_toys - donated_toys
  let after_buying := remaining_toys + new_toys_bought
  let brother_gift := brother_toys / 4
  in after_buying + brother_gift = 108
  sorry

end gerald_toy_cars_left_l193_193718


namespace matrix_multiplication_example_l193_193335

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ := ![![3, -2], ![-4, 5]]
def vector1 : Fin 2 → ℤ := ![4, -2]
def scalar : ℤ := 2
def result : Fin 2 → ℤ := ![32, -52]

theorem matrix_multiplication_example :
  scalar • (matrix1.mulVec vector1) = result := by
  sorry

end matrix_multiplication_example_l193_193335


namespace total_sampled_papers_l193_193960

-- Define the conditions
variables {A B C c : ℕ}
variable (H : A = 1260 ∧ B = 720 ∧ C = 900 ∧ c = 50)
variable (stratified_sampling : true)   -- We simply denote that stratified sampling method is used

-- Theorem to prove the total number of exam papers sampled
theorem total_sampled_papers {T : ℕ} (H : A = 1260 ∧ B = 720 ∧ C = 900 ∧ c = 50) (stratified_sampling : true) :
  T = (1260 + 720 + 900) * (50 / 900) := sorry

end total_sampled_papers_l193_193960


namespace total_course_selection_schemes_l193_193258

theorem total_course_selection_schemes (PE_courses : ℕ) (Art_courses : ℕ) : 
  PE_courses = 4 → Art_courses = 4 → 
  (finset.card (finset.powerset_len 2 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b ∈ s), a < PE_courses ∧ b ≥ PE_courses) + 
   finset.card (finset.powerset_len 3 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b c ∈ s), (a < PE_courses ∧ b < PE_courses ∧ c ≥ PE_courses) ∨ 
                        (a < PE_courses ∧ b ≥ PE_courses ∧ c ≥ PE_courses ∧ a ≠ b ≠ c)) = 64 :=
by
  sorry

end total_course_selection_schemes_l193_193258


namespace limit_problem_proof_l193_193327

noncomputable def limit_problem :=
  lim (λ n : ℕ, (3 * n^2 - 6 * n + 7) / (3 * n^2 + 20 * n - 1) ^ (-n + 1)) = Real.exp (26 / 3)

theorem limit_problem_proof :
  limit_problem := by
  sorry

end limit_problem_proof_l193_193327


namespace general_term_sequence_T_30_value_l193_193735

-- Definitions of the conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

axiom a1_a2_condition : ∀ (a : ℕ → ℝ), a 1 + a 2 = 3
axiom a4_a5_condition : ∀ (a : ℕ → ℝ), a 4 + a 5 = 5

-- General term for the sequence
theorem general_term_sequence (d : ℝ) (a : ℕ → ℝ)
  (h1 : a 1 + a 2 = 3)
  (h2 : a 4 + a 5 = 5)
  : arithmetic_sequence a d → d = 1 / 3 ∧ (∀ n : ℕ, a n = n / 3 + 1) :=
by
  sorry

-- Definition of T_n
def floor_sum (a : ℕ → ℝ) (n : ℕ) : ℕ := (Σ m in finset.range n, ⌊a m⌋)

-- Proving the value of T_30
theorem T_30_value (a : ℕ → ℝ) (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : a 1 + a 2 = 3)
  (h3 : a 4 + a 5 = 5)
  : floor_sum a 30 = 175 :=
by
  sorry

end general_term_sequence_T_30_value_l193_193735


namespace total_number_of_course_selection_schemes_l193_193249

-- Define the total number of courses
def total_courses := 8

-- Define number of physical education and art courses
def pe_courses := 4
def art_courses := 4

-- Define selections: students choose 2 or 3 courses
def course_selections : Finset (Finset ℕ) :=
  (Finset.powerset (Finset.range total_courses)).filter (λ s, s.card = 2 ∨ s.card = 3)

-- Define condition: at least 1 course from each category
def valid_selections : Finset (Finset ℕ) :=
  course_selections.filter (λ s, ∃ pe art, s = pe ∪ art ∧ pe.card ≠ 0 ∧ art.card ≠ 0 ∧ 
                             pe ⊆ Finset.range pe_courses ∧ art ⊆ (Finset.range total_courses).filter (λ x, x ≥ pe_courses))

theorem total_number_of_course_selection_schemes : valid_selections.card = 64 := 
by sorry

end total_number_of_course_selection_schemes_l193_193249


namespace remainder_when_divided_by_8_l193_193144

-- Define a binary number as a list of bits (0 or 1)
def binary_number := List Bool

-- Convert a binary number to a natural number
def binary_to_nat (b : binary_number) : ℕ :=
  b.foldl (λ (acc n) bit, acc * 2 + cond bit 1 0) 0

-- Define the specific binary number in question
def b := [1,0,1,1,1,0,1,0,0,1,0,1] 

-- Define the last three digits of the binary number
def last_three_digits : binary_number := [1, 0, 1]

-- Convert the last three digits to a natural number
def remainder := binary_to_nat last_three_digits

-- TODO: Prove that remainder is equal to 5
theorem remainder_when_divided_by_8 :
  remainder = 5 := by 
  sorry

end remainder_when_divided_by_8_l193_193144


namespace compute_expression_l193_193854

theorem compute_expression (ω : ℂ) (hω : ω^3 = 1) (hω_nonreal : ω.im ≠ 0) :
  let k := 2 in
  (k - ω + ω^2)^3 + (k + ω - ω^2)^3 = 28 :=
by
  sorry

end compute_expression_l193_193854


namespace symmetric_about_origin_l193_193534

-- Define a point in a 2D coordinate system
structure Point (T : Type) :=
(x : T)
(y : T)

-- Define point P with given coordinates
def P : Point ℝ := {x := 2, y := -4}

-- Prove the coordinates of the point symmetric with point P about the origin
theorem symmetric_about_origin (P : Point ℝ) : 
  (∃ Q : Point ℝ, Q.x = -P.x ∧ Q.y = -P.y) :=
by 
  -- Instantiate the point Q with symmetric coordinates
  let Q := { x := -P.x, y := -P.y }

  -- Prove the coordinates of Q match the expected symmetric coordinates
  use Q
  split
  sorry

end symmetric_about_origin_l193_193534


namespace simplify_and_evaluate_expression_l193_193895

theorem simplify_and_evaluate_expression (x y : ℝ) (h₁ : x = 2) (h₂ : y = -1) : 
  2 * x * y - (1 / 2) * (4 * x * y - 8 * x^2 * y^2) + 2 * (3 * x * y - 5 * x^2 * y^2) = -36 := by
  sorry

end simplify_and_evaluate_expression_l193_193895


namespace total_course_selection_schemes_l193_193192

theorem total_course_selection_schemes :
  let n_physical_education := 4
  let n_art := 4
  let total_courses := n_physical_education + n_art in
  let choose2_courses := (Nat.choose n_physical_education 1) * (Nat.choose n_art 1)
  let choose3_courses := (Nat.choose n_physical_education 2 * Nat.choose n_art 1) + (Nat.choose n_physical_education 1 * Nat.choose n_art 2) in
  total_courses = n_physical_education + n_art →
  choose2_courses + choose3_courses = 64 :=
by
  intros n_physical_education n_art total_courses choose2_courses choose3_courses h
  have h_choose2_courses: choose2_courses = 16 := by
    simp [n_physical_education, n_art, Nat.choose]
  have h_choose3_courses: choose3_courses = 48 := by
    simp [n_physical_education, n_art, Nat.choose]
  rw [h_choose2_courses, h_choose3_courses]
  exact Nat.add_eq_right.2 rfl

end total_course_selection_schemes_l193_193192


namespace female_students_count_l193_193530

theorem female_students_count 
  (total_average : ℝ)
  (male_count : ℕ)
  (male_average : ℝ)
  (female_average : ℝ)
  (female_count : ℕ) 
  (correct_female_count : female_count = 12)
  (h1 : total_average = 90)
  (h2 : male_count = 8)
  (h3 : male_average = 87)
  (h4 : female_average = 92) :
  total_average * (male_count + female_count) = male_count * male_average + female_count * female_average :=
by sorry

end female_students_count_l193_193530


namespace factor_sum_l193_193789

theorem factor_sum (R S : ℝ) (h : ∃ (b c : ℝ), (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + R*x^2 + S) : R + S = 54 :=
sorry

end factor_sum_l193_193789


namespace length_of_MN_l193_193535

open Real

theorem length_of_MN (R r : ℝ) (M K N : Point) (A B : Point) :
  (dist M A + dist M B = 6) →
  (on_circle M (center O, R)) →
  (on_circle M (center O1, r)) →
  (on_diameter K (A, B)) →
  (tangent M K) →
  (ray_MK_intersects_N M K N) →
  MN = 3 * sqrt 2 :=
begin
  sorry
end

end length_of_MN_l193_193535


namespace booksReadPerDay_l193_193501

-- Mrs. Hilt read 14 books in a week.
def totalBooksReadInWeek : ℕ := 14

-- There are 7 days in a week.
def daysInWeek : ℕ := 7

-- We need to prove that the number of books read per day is 2.
theorem booksReadPerDay :
  totalBooksReadInWeek / daysInWeek = 2 :=
by
  sorry

end booksReadPerDay_l193_193501


namespace problem_statement_l193_193007

open Real

def point := (ℝ × ℝ)

def vector (A B : point) : point := (B.1 - A.1, B.2 - A.2)

def dot_product (v w : point) : ℝ := v.1 * w.1 + v.2 * w.2

def norm (v : point) : ℝ := sqrt (v.1^2 + v.2^2)

noncomputable def t_value (A B C : point) : ℝ :=
  let AB := vector A B
  let OC := vector (0, 0) C
  (- dot_product AB OC) / (dot_product OC OC)

theorem problem_statement : 
  let A := (1, 4)
  let B := (-2, 3)
  let C := (2, -1)
  let AB := vector A B
  let AC := vector A C
  ∃ (t : ℝ), 
    dot_product AB AC = 2 ∧
    norm (AB.1 + AC.1, AB.2 + AC.2) = 2 * sqrt 10 ∧
    (let OC := vector (0, 0) C in
      t = t_value A B C ∧ 
      (dot_product (AB.1 - t * OC.1, AB.2 - t * OC.2) OC = 0)) :=
by
  let A := (1, 4)
  let B := (-2, 3)
  let C := (2, -1)
  let AB := vector A B
  let AC := vector A C
  let OC := vector (0, 0) C
  use -1
  sorry

end problem_statement_l193_193007


namespace exists_infinitely_many_positive_integer_solutions_l193_193684

theorem exists_infinitely_many_positive_integer_solutions :
  ∃ (x : Fin 1985 → ℕ) (y z : ℕ),
    (∑ i, (x i)^2 = y^3) ∧ 
    (∑ i, (x i)^3 = z^2) ∧ 
    (∀ i j, i ≠ j → x i ≠ x j) :=
sorry

end exists_infinitely_many_positive_integer_solutions_l193_193684


namespace total_number_of_course_selection_schemes_l193_193248

-- Define the total number of courses
def total_courses := 8

-- Define number of physical education and art courses
def pe_courses := 4
def art_courses := 4

-- Define selections: students choose 2 or 3 courses
def course_selections : Finset (Finset ℕ) :=
  (Finset.powerset (Finset.range total_courses)).filter (λ s, s.card = 2 ∨ s.card = 3)

-- Define condition: at least 1 course from each category
def valid_selections : Finset (Finset ℕ) :=
  course_selections.filter (λ s, ∃ pe art, s = pe ∪ art ∧ pe.card ≠ 0 ∧ art.card ≠ 0 ∧ 
                             pe ⊆ Finset.range pe_courses ∧ art ⊆ (Finset.range total_courses).filter (λ x, x ≥ pe_courses))

theorem total_number_of_course_selection_schemes : valid_selections.card = 64 := 
by sorry

end total_number_of_course_selection_schemes_l193_193248


namespace ratio_AD_DC_in_ABC_l193_193442

theorem ratio_AD_DC_in_ABC 
  (A B C D : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AB BC AC : Real) 
  (hAB : AB = 6) (hBC : BC = 8) (hAC : AC = 10) 
  (BD : Real) 
  (hBD : BD = 8) 
  (AD DC : Real)
  (hAD : AD = 2 * Real.sqrt 7)
  (hDC : DC = 10 - 2 * Real.sqrt 7) :
  AD / DC = (10 * Real.sqrt 7 + 14) / 36 :=
sorry

end ratio_AD_DC_in_ABC_l193_193442


namespace cardinality_of_set_A_lt_2_sqrt_n_l193_193867

open Nat

theorem cardinality_of_set_A_lt_2_sqrt_n 
  (n : ℕ) (h_pos : 0 < n) 
  (A : Set ℕ) (h_subset : A ⊆ {x | x ∈ range (n + 1)})
  (h_lcm : ∀ a b ∈ A, lcm a b ≤ n) : A.card < 2 * Nat.sqrt n := 
by 
  sorry

end cardinality_of_set_A_lt_2_sqrt_n_l193_193867


namespace n_value_l193_193792

theorem n_value (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^18 → n = 9 :=
by
  sorry

end n_value_l193_193792


namespace solve_system_l193_193901

theorem solve_system :
  ∀ (x y z : ℝ),
  (x^2 - 23 * y - 25 * z = -681) →
  (y^2 - 21 * x - 21 * z = -419) →
  (z^2 - 19 * x - 21 * y = -313) →
  (x = 20 ∧ y = 22 ∧ z = 23) :=
by
  intros x y z h1 h2 h3
  sorry

end solve_system_l193_193901


namespace find_real_part_of_zero_equation_l193_193707

theorem find_real_part_of_zero_equation (a b : ℝ) (z : ℂ) (h1 : z = a + complex.I*b) (h2 : (z + 2*complex.I) * (z + 4*complex.I) * z = 502*complex.I) : 
  ¬ ∃ a, (∃ b, (z = a + complex.I * b) ∧ ((z + 2 * complex.I) * (z + 4 * complex.I) * z = 502 * complex.I)) ∧ 
        (∃ c : ℝ, (c * complex.I) ≠ z ∧ ((c * complex.I + 2 * complex.I) * (c * complex.I + 4 * complex.I) * c * complex.I = 502 * complex.I)) := sorry

end find_real_part_of_zero_equation_l193_193707


namespace even_numbers_sequence_l193_193532

theorem even_numbers_sequence (n : ℕ) (h : (∑ i in finset.range n, 2 * (i + 1) / n = 14 )) : n = 13 :=
sorry

end even_numbers_sequence_l193_193532


namespace reciprocal_neg1_l193_193558

theorem reciprocal_neg1 : ∃ x : ℝ, -1 * x = 1 ∧ x = -1 :=
by {
  use (-1),
  split,
  { exact by norm_num, },
  { refl, }
}

end reciprocal_neg1_l193_193558


namespace prove_some_number_l193_193375

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem prove_some_number (some_number : ℤ) :
  (floor 6.5) * (floor (2 / 3)) + (floor 2) * 7.2 + some_number - 9.8 = 12.599999999999998 →
  some_number = 7 :=
by
  sorry

end prove_some_number_l193_193375


namespace angle_AD_plane_ABC_30_l193_193371

-- Define the given conditions
variables (a : ℝ) (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables [inhabited A] [inhabited B] [inhabited C] [inhabited D]
variables (square : A → A → A → A → Prop) (side_length : A → B → ℝ) (fold : A × A → A × A)

-- Assume the conditions
axiom square_def : ∀ (a : ℝ) (A B C D : A), (square A B C D) → (side_length A B = a) →
  (square A B C D) → (fold (A, C) = (B, a)) → (side_length B D = a)

-- The angle between a line and a plane defined mathematically
noncomputable def angle_between_line_and_plane (AD : A × A) (plane_ABC : set A) : ℝ :=
  sorry -- Placeholder for the definition of angle

-- Prove the angle between line AD and plane ABC is 30 degrees
theorem angle_AD_plane_ABC_30 : ∀ (a : ℝ) (A B C D : A), (square A B C D) → 
  (side_length A B = a) → (side_length B D = a) → 
  angle_between_line_and_plane (A, D) ({A, B, C} : set A) = 30 :=
sorry -- the proof is deferred

end angle_AD_plane_ABC_30_l193_193371


namespace jo_stairs_l193_193027

-- Define the recursive function f to count the number of ways to reach the nth step.
def f : ℕ → ℕ
| 0 := 1
| 1 := 1
| 2 := 2
| 3 := 4
| 4 := 8
| (n+5) := f (n+4) + f (n+3) + f (n+2) + f (n+1)

-- Problem statement to prove that f(10) = 401.
theorem jo_stairs : f 10 = 401 := by
  sorry

end jo_stairs_l193_193027


namespace rectangle_to_square_area_ratio_is_24_25_l193_193922

noncomputable def rectangle_to_square_area_ratio
  (s : ℝ) -- length of side of square S
  (longer_side : ℝ := 1.2 * s) -- longer side of rectangle R
  (shorter_side : ℝ := 0.8 * s) -- shorter side of rectangle R
  (area_R : ℝ := longer_side * shorter_side) -- area of rectangle R
  (area_S : ℝ := s^2) -- area of square S
  : ℝ := 
  area_R / area_S

theorem rectangle_to_square_area_ratio_is_24_25 
  (s : ℝ)
  : rectangle_to_square_area_ratio s = 24 / 25 :=
by 
  sorry

end rectangle_to_square_area_ratio_is_24_25_l193_193922


namespace Problem1_part1_Problem1_part2_Problem2_l193_193392

section Problem1_1
variable (a : ℕ → ℝ) (n : ℕ)
def T (n : ℕ) : ℝ := ∏ i in finset.range n, a (i + 1)

-- Prove that given a geometric sequence with a1 = 2016, and q = -1/2,
-- the product of the first n terms is T_n = 2016^n * (-1/2)^((n(n-1))/2)
theorem Problem1_part1 
  (a1 : ℝ) (q : ℝ)
  (ha : a 1 = a1)
  (hq : ∀ n, a (n + 1) = a1 * q^n) 
  : T a n = a1^n * q^(n * (n - 1) / 2) := by sorry

end Problem1_1

section Problem1_2

-- Prove that T_n reaches its maximum value at n = 12 given the expression for T_n
theorem Problem1_part2 
  (T : ℕ → ℝ)
  (hT : ∀ n, T n = 2016^n * (-1/2)^((n * (n - 1)) / 2))
  : ∃ n, n = 12 ∧ ∀ k, T k ≤ T n := by sorry

end Problem1_2

section Problem2

-- Prove that {an} is a geometric sequence given an > 0 and Tn * T(n+1) = (a1 * an)^(n/2) * (a1 * a(n+1))^((n+1)/2)
theorem Problem2 
  (a : ℕ → ℝ)
  (T : ℕ → ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, T n * T (n + 1) = (a 1 * a n) ^ (n / 2) * (a 1 * a (n + 1)) ^ ((n + 1) / 2)) 
  : ∃ q, ∀ n, a (n + 1) = a 1 * q^n := by sorry

end Problem2

end Problem1_part1_Problem1_part2_Problem2_l193_193392


namespace toys_sold_week2_l193_193659

-- Define the given conditions
def original_stock := 83
def toys_sold_week1 := 38
def toys_left := 19

-- Define the statement we want to prove
theorem toys_sold_week2 : (original_stock - toys_left) - toys_sold_week1 = 26 :=
by
  sorry

end toys_sold_week2_l193_193659


namespace geometry_problem_l193_193317

theorem geometry_problem 
  (A B C D G E F : Point) 
  (h1 : Inside D (Triangle A B C))
  (h2 : ∠ ADB = ∠ ACB + 90)
  (h3 : AC * BD = AD * BC) 
  (h4 : IntersectsCircumcircle AD G (Circumcircle A B C))
  (h5 : IntersectsCircumcircle BD E (Circumcircle A B C))
  (h6 : IntersectsCircumcircle CD F (Circumcircle A B C)) :
  (EF = FG) ∧ (S(△ EFG) / S(Circle T) = 1 / π) := 
by 
  sorry

end geometry_problem_l193_193317


namespace Carla_grocery_distance_l193_193669

theorem Carla_grocery_distance :
  ∀ (x : ℝ),
  (let d1 := 6 in
  let d2 := 12 in
  let d3 := 2 * d2 in
  let mpg := 25 in
  let cost_per_gallon := 2.50 in
  let total_spent := 5.00 in
  let total_gallons := total_spent / cost_per_gallon in
  let total_distance := mpg * total_gallons in
  let D := x + d1 + d2 + d3 in
  D = total_distance) →
  x = 8 :=
by
  intros x h,
  sorry

end Carla_grocery_distance_l193_193669


namespace area_of_triangle_ABC_l193_193592

-- Define the entities and their relationships
variables {A B C D : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (dist : A → A → ℝ)

noncomputable def right_triangle_angle_at (A B C : A) : Prop :=
∃ D : A, isRightTriangle A B C ∧ isPerpendicularTo (line B D) (line A C) ∧ dist B D = dist B C

variables (AD DC : ℝ)
variables (h₁ : AD = 5)
variables (h₂ : DC = 3)

theorem area_of_triangle_ABC (A B C D : A)
  (h1 : is_right_triangle A B C)
  (h2 : is_point_foot_of_altitude D B A C)
  (h3 : dist A D = AD)
  (h4 : dist D C = DC) :
  area_of_triangle A B C = 4 * (sqrt 15) :=
sorry

end area_of_triangle_ABC_l193_193592


namespace square_paper_fold_l193_193011

theorem square_paper_fold
  (ABCD : Type)
  [square ABCD]
  (AB CD : segment ABCD) (AD : segment ABCD)
  (A B C D F G H : point ABCD)
  (midpoint_AD : F = midpoint A D)
  (side_length : ∀ (x : point ABCD), ∃ y : segment ABCD, length y = 8)
  (B_fold_F : reflects B F)
  (H_on_AB : on_line H AB) :
  length (segment A H) = 3 :=
sorry

end square_paper_fold_l193_193011


namespace janet_wait_time_l193_193839

theorem janet_wait_time
  (janet_speed : ℝ)
  (sister_speed : ℝ)
  (lake_width : ℝ)
  (janet_time : ℝ)
  (sister_time : ℝ) :
  janet_speed = 30 →
  sister_speed = 12 →
  lake_width = 60 →
  janet_time = lake_width / janet_speed →
  sister_time = lake_width / sister_speed →
  (sister_time - janet_time = 3) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end janet_wait_time_l193_193839


namespace total_course_selection_schemes_l193_193270

theorem total_course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose_two :=
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 1)
  let choose_three :=
    (Nat.choose PE_courses 2) * (Nat.choose Art_courses 1) +
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 2)
  total_courses = 8
  ∧ (choose_two + choose_three = 64) :=
by
  sorry

end total_course_selection_schemes_l193_193270


namespace total_number_of_course_selection_schemes_l193_193254

-- Define the total number of courses
def total_courses := 8

-- Define number of physical education and art courses
def pe_courses := 4
def art_courses := 4

-- Define selections: students choose 2 or 3 courses
def course_selections : Finset (Finset ℕ) :=
  (Finset.powerset (Finset.range total_courses)).filter (λ s, s.card = 2 ∨ s.card = 3)

-- Define condition: at least 1 course from each category
def valid_selections : Finset (Finset ℕ) :=
  course_selections.filter (λ s, ∃ pe art, s = pe ∪ art ∧ pe.card ≠ 0 ∧ art.card ≠ 0 ∧ 
                             pe ⊆ Finset.range pe_courses ∧ art ⊆ (Finset.range total_courses).filter (λ x, x ≥ pe_courses))

theorem total_number_of_course_selection_schemes : valid_selections.card = 64 := 
by sorry

end total_number_of_course_selection_schemes_l193_193254


namespace area_ratio_l193_193918

-- Define the problem conditions
def Square (s : ℝ) := s > 0
def Rectangle (longer shorter : ℝ) := longer = 1.2 * shorter ∧ shorter = 0.8 * shorter

-- Define a function to calculate the area of square
def area_square (s : ℝ) : ℝ := s * s

-- Define a function to calculate the area of rectangle
def area_rectangle (longer shorter : ℝ) : ℝ := longer * shorter

-- State the proof problem
theorem area_ratio (s : ℝ) (h_square : Square s) :
  let longer := 1.2 * s
  let shorter := 0.8 * s
  area_rectangle longer shorter / area_square s = 24 / 25 :=
by
  sorry

end area_ratio_l193_193918


namespace remainder_n_squared_l193_193154

theorem remainder_n_squared (n : ℤ) (h : n % 5 = 3) : (n^2) % 5 = 4 := 
    sorry

end remainder_n_squared_l193_193154


namespace king_sons_ducats_l193_193108

-- Define the variables and conditions
def num_sons (n : ℕ) : Prop :=
  ∃ (a : ℕ), (∑ i in finset.range n, (a - i)) + n = 21 ∧ 
            (∑ i in finset.range n, (a - i)) + (∑ i in finset.range n, (i + 1)) = 105

-- Prove that n is 7 and total ducats distributed is 105
theorem king_sons_ducats : ∃ n, num_sons n :=
 by {
   existsi 7,
   unfold num_sons,
   existsi 14, -- Since it's the amount eldest received in the first round
   split,
   { 
     -- Prove the first part
     simp only [finset.sum_range_succ_sub, add_left_inj, finset.sum_range_id], },
   { 
     -- Prove the second part
     simp only [finset.sum_range_succ, add_comm, nat.succ_eq_add_one, add_assoc],
     norm_num, },
   sorry -- to skip the detailed steps of solution
}

end king_sons_ducats_l193_193108


namespace points_lie_on_circle_l193_193380

theorem points_lie_on_circle (u : ℝ) :
    let x := 2 * u / (1 + u^2)
    let y := (1 - u^2) / (1 + u^2)
    in x^2 + y^2 = 1 :=
by
  sorry

end points_lie_on_circle_l193_193380


namespace total_course_selection_schemes_l193_193265

theorem total_course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose_two :=
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 1)
  let choose_three :=
    (Nat.choose PE_courses 2) * (Nat.choose Art_courses 1) +
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 2)
  total_courses = 8
  ∧ (choose_two + choose_three = 64) :=
by
  sorry

end total_course_selection_schemes_l193_193265


namespace probability_cos_half_pi_l193_193519

open Real Set MeasureTheory

noncomputable def cos_half_pi_section_prob : ℝ :=
  ∫ x in (Ioc (-(1 : ℝ)) 1 : Set ℝ), indicator (Icc (-(1 : ℝ)) 1) (λ x, if 0 ≤ cos (π * x / 2) ∧ cos (π * x / 2) ≤ 1 / 2 then 1 else 0) x

theorem probability_cos_half_pi :
  cos_half_pi_section_prob = 1/3 :=
sorry

end probability_cos_half_pi_l193_193519


namespace turns_in_two_hours_l193_193648

theorem turns_in_two_hours (turns_per_30_sec : ℕ) (minutes_in_hour : ℕ) (hours : ℕ) : 
  turns_per_30_sec = 6 → 
  minutes_in_hour = 60 → 
  hours = 2 → 
  (12 * (minutes_in_hour * hours)) = 1440 := 
by
  sorry

end turns_in_two_hours_l193_193648


namespace total_number_of_course_selection_schemes_l193_193245

-- Define the total number of courses
def total_courses := 8

-- Define number of physical education and art courses
def pe_courses := 4
def art_courses := 4

-- Define selections: students choose 2 or 3 courses
def course_selections : Finset (Finset ℕ) :=
  (Finset.powerset (Finset.range total_courses)).filter (λ s, s.card = 2 ∨ s.card = 3)

-- Define condition: at least 1 course from each category
def valid_selections : Finset (Finset ℕ) :=
  course_selections.filter (λ s, ∃ pe art, s = pe ∪ art ∧ pe.card ≠ 0 ∧ art.card ≠ 0 ∧ 
                             pe ⊆ Finset.range pe_courses ∧ art ⊆ (Finset.range total_courses).filter (λ x, x ≥ pe_courses))

theorem total_number_of_course_selection_schemes : valid_selections.card = 64 := 
by sorry

end total_number_of_course_selection_schemes_l193_193245


namespace find_prime_p_l193_193356

noncomputable def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem find_prime_p (p : ℕ) (hp : p.Prime) (hsquare : isPerfectSquare (5^p + 12^p)) : p = 2 := 
sorry

end find_prime_p_l193_193356


namespace committee_rearrangements_l193_193683

open Nat

theorem committee_rearrangements : 
  let word := "COMMITTEE"
  let vowels := ['O', 'I', 'E', 'E']
  let consonants := ['C', 'M', 'M', 'T', 'T']
  let arrange_vowels := (fact 4) / (fact 2)
  let choose_positions_for_M := choose 5 2
  let choose_positions_for_T := choose 3 2
  let total_arrangements := arrange_vowels * choose_positions_for_M * choose_positions_for_T
  total_arrangements = 360 := 
by
  sorry

end committee_rearrangements_l193_193683


namespace narrowest_strip_width_equilateral_triangle_with_arcs_l193_193660

theorem narrowest_strip_width_equilateral_triangle_with_arcs :
  ∀ (side_length : ℝ) (r1 r5 : ℝ), 
  side_length = 4 ∧ r1 = 1 ∧ r5 = 5 → 
  ∃ (strip_width : ℝ), strip_width = 6 ∧
  (∀ (parallel_lines : set (set ℝ)), is_shape_contained (equilateral_triangle_with_arcs side_length r1 r5) parallel_lines → 
    (width_of_narrowest_strip parallel_lines = strip_width)) :=
begin
  intros side_length r1 r5 h,
  -- Definitions and assumptions go here
  sorry
end

end narrowest_strip_width_equilateral_triangle_with_arcs_l193_193660


namespace arithmetic_mean_34_58_l193_193596

theorem arithmetic_mean_34_58 :
  (3 / 4 : ℚ) + (5 / 8 : ℚ) / 2 = 11 / 16 := sorry

end arithmetic_mean_34_58_l193_193596


namespace rogers_crayons_l193_193888

theorem rogers_crayons : 
  let new_crayons := 2 in
  let used_crayons := 4 in
  let broken_crayons := 8 in
  new_crayons + used_crayons + broken_crayons = 14 := 
by
  sorry

end rogers_crayons_l193_193888


namespace unique_positive_solution_cos_arcsin_tan_arccos_l193_193366

theorem unique_positive_solution_cos_arcsin_tan_arccos (x : ℝ) (h : 0 < x ∧ x ≤ 1) :
  cos (arcsin (tan (arccos x))) = x ↔ x = 1 :=
by
  sorry

end unique_positive_solution_cos_arcsin_tan_arccos_l193_193366


namespace ratio_area_of_rectangle_to_square_l193_193934

theorem ratio_area_of_rectangle_to_square (s : ℝ) :
  (1.2 * s * 0.8 * s) / (s * s) = 24 / 25 :=
by
  sorry

end ratio_area_of_rectangle_to_square_l193_193934


namespace is_existential_proposition_l193_193978

-- Definitions of each condition
def even_function_symmetric_y_axis (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def all_right_square_prisms_are_parallelepipeds : Prop := ∀ (P : Type) [right_square_prism P], parallelepiped P
def two_non_intersecting_lines_parallel (L1 L2 : Type) [non_intersecting_lines L1 L2] : Prop := parallel L1 L2
def exists_real_number_ge_3 : Prop := ∃ x : ℝ, x ≥ 3

-- Assertion that statement D is an existential proposition.
theorem is_existential_proposition : exists_real_number_ge_3 := sorry

end is_existential_proposition_l193_193978


namespace find_sum_of_a_b_l193_193343

def star (a b : ℕ) : ℕ := a^b - a * b

theorem find_sum_of_a_b (a b : ℕ) (h1 : 2 ≤ a) (h2 : 2 ≤ b) (h3 : star a b = 2) : a + b = 5 := 
by
  sorry

end find_sum_of_a_b_l193_193343


namespace avg_math_chem_l193_193991

variables (M P C : ℕ)

def total_marks (M P : ℕ) := M + P = 50
def chemistry_marks (P C : ℕ) := C = P + 20

theorem avg_math_chem (M P C : ℕ) (h1 : total_marks M P) (h2 : chemistry_marks P C) :
  (M + C) / 2 = 35 :=
by
  sorry

end avg_math_chem_l193_193991


namespace prove_eccentricity_of_ellipse_l193_193743

noncomputable def eccentricity_of_ellipse (F1 F2 A B : ℝ) (a b c e : ℝ) : Prop :=
  (|F1 - F2| = 2 * c) → 
  (|A - F1| = |B - F1|) → 
  (|A - B| = 2 * b) → 
  (|A - F2| = |B - F2|) → 
  (A ≠ B) → 
  (F1 ≠ F2) → 
  (a^2 = c^2 + b^2) → 
  (|A - F1| = (sqrt 3) / 3 * |F1 - F2|) → 
  (e = c / a) → 
  (e = sqrt 3 / 3)

theorem prove_eccentricity_of_ellipse : 
  ∃ F1 F2 A B a b c e, eccentricity_of_ellipse F1 F2 A B a b c e :=
by
  sorry

end prove_eccentricity_of_ellipse_l193_193743


namespace part1_part2_l193_193041

open Real

noncomputable def f (a x : ℝ) : ℝ := abs (x - a) + (1 - a) * x

theorem part1 (a : ℝ) : f a 2 < 0 ↔ a ∈ Ioo (4 / 3) ∞ :=
sorry

theorem part2 (a : ℝ) : (∀ x, f a x ≥ 0) ↔ a ∈ Icc 0 1 :=
sorry

end part1_part2_l193_193041


namespace ellipse_xintercept_unique_l193_193656

theorem ellipse_xintercept_unique {f1 f2 : ℝ × ℝ} (hf1 : f1 = (0, 3)) (hf2 : f2 = (4, 0)) 
(hx1 : (4, 0) ∈ {p : ℝ × ℝ | ∀ q ∈ ({f1, f2} : set (ℝ × ℝ)), (p.1 - q.1)^2 + (p.2 - q.2)^2 = 5}) 
: (4, 0) ∈ {p : ℝ × ℝ | ∀ q ∈ ({f1, f2} : set (ℝ × ℝ)), (p.1 - q.1)^2 + (p.2 - q.2)^2 = 5} :=
begin
  sorry
end

end ellipse_xintercept_unique_l193_193656


namespace sqrt_diff_square_l193_193956

theorem sqrt_diff_square :
  (Real.sqrt 25 - Real.sqrt 9) ^ 2 = 4 := by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : Real.sqrt 9 = 3 := by sorry
  calc
    (Real.sqrt 25 - Real.sqrt 9) ^ 2
    = (5 - 3) ^ 2 : by rw [h1, h2]
    = 2 ^ 2         : by rfl
    = 4             : by rfl

end sqrt_diff_square_l193_193956


namespace fraction_workers_night_crew_l193_193663

variables (D N : ℕ) (B : ℕ)

-- Condition 1: Each worker on the night crew loads 3/4 as many boxes as each worker on the day crew.
def night_crew_boxes_per_worker := (3 / 4 : ℚ) * B

-- Condition 2: The day crew loads 0.64 of all the boxes loaded by the two crews.
def fraction_boxes_day_crew := (0.64 : ℚ)

theorem fraction_workers_night_crew (h1 : (D * B) / (D * B + N * (3 / 4 * B)) = 0.64) : 
  N / D = 3 / 4 :=
  sorry

end fraction_workers_night_crew_l193_193663


namespace minute_hand_distance_l193_193546

noncomputable def distance_traveled (length_of_minute_hand : ℝ) (time_duration : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * length_of_minute_hand
  let revolutions := time_duration / 60
  circumference * revolutions

theorem minute_hand_distance :
  distance_traveled 8 45 = 12 * Real.pi :=
by
  sorry

end minute_hand_distance_l193_193546


namespace find_length_of_AX_l193_193018

theorem find_length_of_AX (A B C X : Type) 
  (BC CX AC : ℝ) (AB AX BX : ℝ)
  (h1 : BC = 35)
  (h2 : CX = 30)
  (h3 : AC = 27)
  (h4 : CX bisects ∠ACB)
  (h5 : AX + BX = AB)
  (h6 : BX = AB - AX) :
  AX = 23.14 := 
sorry

end find_length_of_AX_l193_193018


namespace solve_inequality_l193_193900

theorem solve_inequality (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  (3 - 1 / (3 * x + 4) < 5) ↔ (-4 / 3 < x) :=
by
  sorry

end solve_inequality_l193_193900


namespace course_selection_schemes_l193_193203

theorem course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose (n k : ℕ) := nat.choose n k
  
  (choose 4 1 * choose 4 1 + 
   choose 4 2 * choose 4 1 + 
   choose 4 1 * choose 4 2) = 64 := 
by {
  -- calculating the combinations
  have comb_2_courses : choose 4 1 * choose 4 1 = 16 := by sorry,
  have comb_2_pe_1_art : choose 4 2 * choose 4 1 = 24 := by sorry,
  have comb_1_pe_2_art : choose 4 1 * choose 4 2 = 24 := by sorry,
  have total_comb : 16 + 24 + 24 = 64 := by {
    simp [comb_2_courses, comb_2_pe_1_art, comb_1_pe_2_art]
  },
  exact total_comb
}

end course_selection_schemes_l193_193203


namespace max_product_iff_consecutive_integers_l193_193309

noncomputable def max_product_condition {n : ℕ} : Prop :=
  ∃ (k : ℕ) (a : ℕ → ℕ), (∑ i in finset.range k, a i = n) ∧ 
  (∀ i j, i < j → a i < a j) ∧ (∀ i, a (i+1) = a i + 1)

theorem max_product_iff_consecutive_integers (n : ℕ) :
  ∃ (k : ℕ) (a : ℕ → ℕ), (∑ i in finset.range k, a i = n) ∧ 
  (∀ i j, i < j → a i < a j) ∧ 
  (∀ j ≠ i+1 → k > 1 → (a (i+1) = a i + 1)) ↔ max_product_condition :=
sorry

end max_product_iff_consecutive_integers_l193_193309


namespace course_selection_schemes_count_l193_193243

-- Definitions based on the conditions
def num_physical_education_courses : ℕ := 4
def num_art_courses : ℕ := 4

-- Required to choose 2 or 3 courses, with at least one from each category
def valid_selection_cases : list (ℕ × ℕ) := [(1, 1), (1, 2), (2, 1)]

-- Calculate the number of ways to choose k courses from n courses using combination formula
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Calculate the total number of different course selection schemes
def total_course_selection_schemes : ℕ :=
  let case1 := C num_physical_education_courses 1 * C num_art_courses 1
  let case2a := C num_physical_education_courses 2 * C num_art_courses 1
  let case2b := C num_physical_education_courses 1 * C num_art_courses 2
  case1 + case2a + case2b

-- The assertion that the total number of different course selection schemes is 64
theorem course_selection_schemes_count : total_course_selection_schemes = 64 := by
  -- sorry placeholder for proof
  sorry

end course_selection_schemes_count_l193_193243


namespace sum_of_odds_in_range_10_to_50_l193_193970

-- Condition: Defining the required sequence and sum
def sequence (n : ℕ) : ℕ := 11 + (n - 1) * 2

-- Number of terms in the sequence
def num_terms : ℕ := 20

-- First term of the sequence
def first_term : ℕ := 11

-- Last term of the sequence
def last_term : ℕ := 49

-- Sum of the sequence
def sequence_sum (n : ℕ) : ℕ := n / 2 * (first_term + last_term)

-- Final statement to prove
theorem sum_of_odds_in_range_10_to_50 : sequence_sum num_terms = 600 :=
by
  sorry

end sum_of_odds_in_range_10_to_50_l193_193970


namespace four_distinct_real_roots_l193_193857

noncomputable def f (x c : ℝ) : ℝ := x^2 + 4 * x + c

-- We need to prove that if c is in the interval (-1, 3), f(f(x)) has exactly 4 distinct real roots
theorem four_distinct_real_roots (c : ℝ) : (-1 < c) ∧ (c < 3) → 
  ∃ (x₁ x₂ x₃ x₄ : ℝ), (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) 
  ∧ (f (f x₁ c) c = 0 ∧ f (f x₂ c) c = 0 ∧ f (f x₃ c) c = 0 ∧ f (f x₄ c) c = 0) :=
by sorry

end four_distinct_real_roots_l193_193857


namespace total_selection_schemes_l193_193279

-- Define the given conditions
def num_phys_ed_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_phys_ed_courses + num_art_courses
def valid_course_combos : finset ℕ := {2, 3}

-- Define the number of selection schemes with the constraints
def selection_schemes : nat :=
  (num_phys_ed_courses.choose 1 * num_art_courses.choose 1) + 
  (num_phys_ed_courses.choose 2 * num_art_courses.choose 1 + 
   num_phys_ed_courses.choose 1 * num_art_courses.choose 2)

-- State the theorem to be proved
theorem total_selection_schemes : selection_schemes = 64 := by
  sorry

end total_selection_schemes_l193_193279


namespace sequence_a6_is_22_l193_193828

def sequence (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n % 2 = 0 then 2 * a (n - 1)
  else a (n - 1) + 2

theorem sequence_a6_is_22 : sequence (show ℕ → ℕ from λ n, sequence (λ n, sequence (λ n, sequence (λ n, 2) n) n) n) 6 = 22 := 
  by
    sorry

end sequence_a6_is_22_l193_193828


namespace math_majors_consecutive_seats_l193_193877

noncomputable def probability_math_majors_consecutive_seats : ℚ :=
  1 / 14

theorem math_majors_consecutive_seats :
  ∀ (n m p : ℕ) (n_ppl : Finset (Fin 9)) (seats : Set (Finset (Fin 9))),
    n = 4 ∧ m = 3 ∧ p = 2 ∧ n_ppl.card = 9 ∧
    (∀ s ∈ seats, s.card = 4 → (∃ k ∈ Finset.range 9, ∀ i ∈ s, i = ↑((k + i) % 9))) →
  probability (math_majors_consecutive_seats : xx : Set {s | s.card = 4} → ℚ) = probability_math_majors_consecutive_seats :=
sorry

end math_majors_consecutive_seats_l193_193877


namespace course_selection_schemes_l193_193202

theorem course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose (n k : ℕ) := nat.choose n k
  
  (choose 4 1 * choose 4 1 + 
   choose 4 2 * choose 4 1 + 
   choose 4 1 * choose 4 2) = 64 := 
by {
  -- calculating the combinations
  have comb_2_courses : choose 4 1 * choose 4 1 = 16 := by sorry,
  have comb_2_pe_1_art : choose 4 2 * choose 4 1 = 24 := by sorry,
  have comb_1_pe_2_art : choose 4 1 * choose 4 2 = 24 := by sorry,
  have total_comb : 16 + 24 + 24 = 64 := by {
    simp [comb_2_courses, comb_2_pe_1_art, comb_1_pe_2_art]
  },
  exact total_comb
}

end course_selection_schemes_l193_193202


namespace older_child_age_l193_193632

theorem older_child_age 
    (mother_bill : ℝ) 
    (child_rate : ℝ) 
    (triplets : ℕ) 
    (older_child : ℕ) 
    (total_bill : ℝ) 
    (t : ℕ) 
    (y : ℕ) 
    (triplets_count : ℕ) 
    (child_count : ℕ) : 
    triplets_count = 3 → 
    child_count = 4 → 
    mother_bill = 6.50 → 
    child_rate = 0.50 → 
    total_bill = 14.50 → 
    16 = 3 * t + y → 
    (y = 4 ∨ y = 7) :=
by
  intros _ _ _ _ _ _ _ _
  sorry

end older_child_age_l193_193632


namespace find_sixth_number_l193_193616

-- Definitions based on conditions a) and requirements
variables {A : Fin 11 → ℝ}
variable avg_all : (1 / 11) * (∑ i in Finset.finRange 11, A i) = 22
variable avg_first_6 : (1 / 6) * (∑ i in Finset.finRange 6, A i) = 19
variable avg_last_6 : (1 / 6) * (∑ i in Finset.Ico 5 11, A i) = 27

-- Theorem stating the expected conclusion
theorem find_sixth_number : A 5 = 34 := by
  sorry

end find_sixth_number_l193_193616


namespace automobile_travel_distance_5_minutes_l193_193654

variable (a r : ℝ)

theorem automobile_travel_distance_5_minutes (h0 : r ≠ 0) :
  let distance_in_feet := (2 * a) / 5
  let time_in_seconds := 300
  (distance_in_feet / r) * time_in_seconds / 3 = 40 * a / r :=
by
  sorry

end automobile_travel_distance_5_minutes_l193_193654


namespace find_rate_of_interest_l193_193533

noncomputable def rate_of_interest (P : ℝ) (r : ℝ) : Prop :=
  let CI2 := P * (1 + r)^2 - P
  let CI3 := P * (1 + r)^3 - P
  CI2 = 1200 ∧ CI3 = 1272 → r = 0.06

theorem find_rate_of_interest (P : ℝ) (r : ℝ) : rate_of_interest P r :=
by sorry

end find_rate_of_interest_l193_193533


namespace scientific_notation_60000_l193_193087

theorem scientific_notation_60000 : ∃ n : ℕ, 60000 = 6 * 10^n :=
by {
  use 4,
  sorry
}

end scientific_notation_60000_l193_193087


namespace total_course_selection_schemes_l193_193226

theorem total_course_selection_schemes (
  pe_courses art_courses : Finset ℕ
) : 
  pe_courses.card = 4 →
  art_courses.card = 4 →
  let total_schemes := 
    (pe_courses.card.choose 1 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 2 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 1 * art_courses.card.choose 2) in
  total_schemes = 64 := 
by
  intros h1 h2
  let case1 := pe_courses.card.choose 1 * art_courses.card.choose 1 -- 4 * 4
  let case2_1 := pe_courses.card.choose 2 * art_courses.card.choose 1 -- 6 * 4
  let case2_2 := pe_courses.card.choose 1 * art_courses.card.choose 2 -- 4 * 6
  let case2 := case2_1 + case2_2 -- 24 + 24
  let total_schemes := case1 + case2 -- 16 + 48
  have hcard := by
    simp [Nat.choose]
    exact h1
  have hcase1 : case1 = 16 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_1 : case2_1 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_2 : case2_2 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2 : case2 = 48 := by {
    rw [hcase2_1, hcase2_2]
    norm_num
  }
  have htotal_schemes : total_schemes = 64 := by {
    rw [hcase1, hcase2]
    norm_num
  }
  exact htotal_schemes

end total_course_selection_schemes_l193_193226


namespace ratio_A_B_l193_193681

noncomputable def A : ℝ := ∑' n : ℕ, if n % 4 = 0 then 0 else 1 / (n:ℝ) ^ 2
noncomputable def B : ℝ := ∑' k : ℕ, (-1)^(k+1) / (4 * (k:ℝ)) ^ 2

theorem ratio_A_B : A / B = 32 := by
  -- proof here
  sorry

end ratio_A_B_l193_193681


namespace polynomial_factorization_l193_193702

theorem polynomial_factorization (a b c : ℤ) (h_distinct: a ≠ b ∧ b ≠ c ∧ c ≠ a) (h_nonzero: a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  ∃ P Q : polynomial ℤ, (∀ x : ℤ, x(x - a)(x - b)(x - c) + 1 = P.eval x * Q.eval x) ↔
  (a, b, c) = (3, 2, 1) ∨ (a, b, c) = (-3, -2, -1) ∨ (a, b, c) = (-1, -2, 1) ∨ (a, b, c) = (1, 2, -1) :=
sorry

end polynomial_factorization_l193_193702


namespace collinear_points_l193_193349

-- Define collinear points function
def collinear (x1 y1 z1 x2 y2 z2 x3 y3 z3: ℝ) : Prop :=
  ∀ (a b c : ℝ), a * (y2 - y1) * (z3 - z1) + b * (z2 - z1) * (x3 - x1) + c * (x2 - x1) * (y3 - y1) = 0

-- Problem statement
theorem collinear_points (a b : ℝ)
  (h : collinear 2 a b a 3 b a b 4) :
  a + b = -2 :=
sorry

end collinear_points_l193_193349


namespace arc_length_calculation_l193_193389

theorem arc_length_calculation (C θ : ℝ) (hC : C = 72) (hθ : θ = 45) :
  (θ / 360) * C = 9 :=
by
  sorry

end arc_length_calculation_l193_193389


namespace marta_hours_worked_l193_193874

-- Definitions of the conditions in Lean 4
def total_collected : ℕ := 240
def hourly_rate : ℕ := 10
def tips_collected : ℕ := 50
def work_earned : ℕ := total_collected - tips_collected

-- Goal: To prove the number of hours worked by Marta
theorem marta_hours_worked : work_earned / hourly_rate = 19 := by
  sorry

end marta_hours_worked_l193_193874


namespace value_of_3x_plus_3y_l193_193945

theorem value_of_3x_plus_3y (x y : ℝ) (h1 : 3^x + 3^(y + 1) = 5 * real.sqrt 3) 
  (h2 : 3^(x + 1) + 3^y = 3 * real.sqrt 3) : 3^x + 3^y = 2 * real.sqrt 3 :=
by
  sorry

end value_of_3x_plus_3y_l193_193945


namespace course_selection_schemes_l193_193211

theorem course_selection_schemes :
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  choose_2_courses + choose_3_courses = 64 :=
by
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  show choose_2_courses + choose_3_courses = 64 from sorry

end course_selection_schemes_l193_193211


namespace functional_equation_solution_l193_193345

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (f(x + 1) + y - 1) = f x + y) →
  (∀ x : ℝ, f x = x) :=
begin
  sorry
end

end functional_equation_solution_l193_193345


namespace simplify_expression_l193_193527

-- Define the given expression
def given_expression (x : ℝ) : ℝ := 5 * x + 9 * x^2 + 8 - (6 - 5 * x - 3 * x^2)

-- Define the expected simplified form
def expected_expression (x : ℝ) : ℝ := 12 * x^2 + 10 * x + 2

-- The theorem we want to prove
theorem simplify_expression (x : ℝ) : given_expression x = expected_expression x := by
  sorry

end simplify_expression_l193_193527


namespace jed_correct_speed_l193_193810

def fine_per_mph := 16
def jed_fine := 256
def speed_limit := 50

def jed_speed : Nat := speed_limit + jed_fine / fine_per_mph

theorem jed_correct_speed : jed_speed = 66 := by
  sorry

end jed_correct_speed_l193_193810


namespace pizza_slices_left_l193_193159

-- Defining the initial number of pizza slices
def initial_slices : ℕ := 16

-- Defining one-fourth of the pizza during dinner time
def dinner_fraction : ℚ := 1 / 4

-- Defining one-fourth of the remaining pizza eaten by Yves
def yves_fraction : ℚ := 1 / 4

-- Defining the slices eaten by each sibling
def slices_per_sibling : ℕ := 2

-- Theorem to prove the number of slices of pizza left is 5
theorem pizza_slices_left :
    let eaten_at_dinner := initial_slices * dinner_fraction
    let remaining_after_dinner := initial_slices - eaten_at_dinner
    let eaten_by_yves := remaining_after_dinner * yves_fraction
    let remaining_after_yves := remaining_after_dinner - eaten_by_yves
    let eaten_by_siblings := 2 * slices_per_sibling
    let final_remaining := remaining_after_yves - eaten_by_siblings
    final_remaining = 5 :=
by
  have step1 : let eaten_at_dinner := initial_slices * dinner_fraction
               let remaining_after_dinner := initial_slices - eaten_at_dinner
               eaten_at_dinner = 4 := by norm_num
  have step2 : let remaining_after_dinner := initial_slices - eaten_at_dinner
               remaining_after_dinner = 12 := by norm_num
  have step3 : let eaten_by_yves := remaining_after_dinner * yves_fraction
               eaten_by_yves = 3 := by norm_num
  have step4 : let remaining_after_yves := remaining_after_dinner - eaten_by_yves
               remaining_after_yves = 9 := by norm_num
  have step5 : let eaten_by_siblings := 2 * slices_per_sibling
               final_remaining = remaining_after_yves - eaten_by_siblings
               eaten_by_siblings = 4 := by norm_num
  show final_remaining = 5 from calc
    final_remaining 
      = remaining_after_yves - eaten_by_siblings := by norm_num
      ... = 5 := by norm_num

end pizza_slices_left_l193_193159


namespace total_nuts_with_opened_or_cracked_shells_shelled_cashews_salted_l193_193623

theorem total_nuts_with_opened_or_cracked_shells :
  let pistachios := 80
  let almonds := 60
  let cashews := 40
  let pistachios_with_shells := (95 * pistachios) / 100
  let pistachios_opened := (75 * pistachios_with_shells) / 100
  let almonds_with_shells := (90 * almonds) / 100
  let almonds_cracked := (80 * almonds_with_shells) / 100
  let cashews_with_shells := (85 * cashews) / 100
  let cashews_opened := (60 * cashews_with_shells) / 100
  in pistachios_opened + almonds_cracked + cashews_opened = 120 := by 
  sorry

theorem shelled_cashews_salted :
  let cashews := 40
  let cashews_with_shells := (85 * cashews) / 100
  let cashews_salted := (70 * cashews_with_shells) / 100
  in cashews_salted = 23 := by 
  sorry

end total_nuts_with_opened_or_cracked_shells_shelled_cashews_salted_l193_193623


namespace course_selection_schemes_count_l193_193240

-- Definitions based on the conditions
def num_physical_education_courses : ℕ := 4
def num_art_courses : ℕ := 4

-- Required to choose 2 or 3 courses, with at least one from each category
def valid_selection_cases : list (ℕ × ℕ) := [(1, 1), (1, 2), (2, 1)]

-- Calculate the number of ways to choose k courses from n courses using combination formula
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Calculate the total number of different course selection schemes
def total_course_selection_schemes : ℕ :=
  let case1 := C num_physical_education_courses 1 * C num_art_courses 1
  let case2a := C num_physical_education_courses 2 * C num_art_courses 1
  let case2b := C num_physical_education_courses 1 * C num_art_courses 2
  case1 + case2a + case2b

-- The assertion that the total number of different course selection schemes is 64
theorem course_selection_schemes_count : total_course_selection_schemes = 64 := by
  -- sorry placeholder for proof
  sorry

end course_selection_schemes_count_l193_193240


namespace math_problem_l193_193396

-- Conditions
def ellipse (a b : ℝ) (a_gt_b : a > b) (b_gt_0 : b > 0) : Prop :=
  ∃ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) 

def focal_length (a b : ℝ) : Prop :=
  (a^2 - b^2 = 3)

def line_l1 (k : ℝ) (k_ne_0 : k ≠ 0) : Prop :=
  ∃ (x y : ℝ), (y = k * x)

def line_l2 (k : ℝ) : Prop :=
  ∃ (x y x1 y1 : ℝ), 
    (y + y1 = (k / 4) * (x + x1))

def perpendicular (x1 y1 x2 y2 k : ℝ) : Prop :=
  (y2 - y1) = (-1 / k) * (x2 - x1)

-- Question 1
def ellipse_C_equation : Prop :=
  ∀ (a b : ℝ), (b^2 = 1 ∧ a^2 = 4) → (b^2 / a^2 = 1 / 4)

-- Question 2
def max_area (x1 y1 : ℝ) : Prop :=
  abs (x1 * y1) = 1 → max (9 / 8 * abs (x1 * y1)) = (9 / 8)

-- Theorem to prove
theorem math_problem (a b k x1 y1 x2 y2 : ℝ) 
  (a_gt_b : a > b) (b_gt_0 : b > 0) (k_ne_0 : k ≠ 0) 
  (h1 : ellipse a b a_gt_b b_gt_0) 
  (h2 : focal_length a b)
  (h3 : line_l1 k k_ne_0)
  (h4 : line_l2 k)
  (h5 : perpendicular x1 y1 x2 y2 k) :
  ellipse_C_equation ∧ max_area x1 y1 :=

begin
  sorry,
end

end math_problem_l193_193396


namespace distance_from_blast_site_l193_193986

theorem distance_from_blast_site 
  (time_heard_second_blast : ℝ) 
  (time_second_blast_occurred : ℝ) 
  (speed_of_sound : ℝ) 
  (time_difference : ℝ)
  (distance : ℝ) :
  (time_heard_second_blast = 30 * 60 + 12) →
  (time_second_blast_occurred = 30 * 60) →
  (speed_of_sound = 330) →
  (time_difference = 12) →
  (distance = speed_of_sound * time_difference) →
  distance = 3960 := 
by
  intros h1 h2 h3 h4 h5
  rw [h3, h4]
  sorry

end distance_from_blast_site_l193_193986


namespace cone_generatrix_l193_193605

-- Definitions and conditions
structure Cone where
  vertex : Point3D
  base_center : Point3D
  base_radius : ℝ
  (radius_pos : base_radius > 0)

-- Generating a point on the circumference of the base
noncomputable def point_on_circumference (c : Cone) (θ : ℝ) : Point3D :=
  ⟨c.base_center.x + c.base_radius * Real.cos θ, 
   c.base_center.y + c.base_radius * Real.sin θ, 
   c.base_center.z⟩  -- Assuming base lies in the XY plane for simplicity

-- The statement to be proved
theorem cone_generatrix (c : Cone) (θ : ℝ) : 
  ∃ g : Line3D, 
  g = Line3D.mk c.vertex (point_on_circumference c θ) :=
by
  sorry

end cone_generatrix_l193_193605


namespace one_point_shots_count_l193_193593

-- Define the given conditions
def three_point_shots : Nat := 15
def two_point_shots : Nat := 12
def total_points : Nat := 75
def points_per_three_shot : Nat := 3
def points_per_two_shot : Nat := 2

-- Define the total points contributed by three-point and two-point shots
def three_point_total : Nat := three_point_shots * points_per_three_shot
def two_point_total : Nat := two_point_shots * points_per_two_shot
def combined_point_total : Nat := three_point_total + two_point_total

-- Formulate the theorem to prove the number of one-point shots Tyson made
theorem one_point_shots_count : combined_point_total <= total_points →
  (total_points - combined_point_total = 6) :=
by 
  -- Skip the proof
  sorry

end one_point_shots_count_l193_193593


namespace course_selection_schemes_count_l193_193236

-- Definitions based on the conditions
def num_physical_education_courses : ℕ := 4
def num_art_courses : ℕ := 4

-- Required to choose 2 or 3 courses, with at least one from each category
def valid_selection_cases : list (ℕ × ℕ) := [(1, 1), (1, 2), (2, 1)]

-- Calculate the number of ways to choose k courses from n courses using combination formula
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Calculate the total number of different course selection schemes
def total_course_selection_schemes : ℕ :=
  let case1 := C num_physical_education_courses 1 * C num_art_courses 1
  let case2a := C num_physical_education_courses 2 * C num_art_courses 1
  let case2b := C num_physical_education_courses 1 * C num_art_courses 2
  case1 + case2a + case2b

-- The assertion that the total number of different course selection schemes is 64
theorem course_selection_schemes_count : total_course_selection_schemes = 64 := by
  -- sorry placeholder for proof
  sorry

end course_selection_schemes_count_l193_193236


namespace reciprocal_of_minus_one_is_minus_one_l193_193559

theorem reciprocal_of_minus_one_is_minus_one : ∃ x : ℝ, (-1) * x = 1 ∧ x = -1 :=
by
  use -1
  split
  case left =>
    show -1 * -1 = 1
    sorry
  case right =>
    show -1 = -1
    sorry

end reciprocal_of_minus_one_is_minus_one_l193_193559


namespace total_selection_schemes_l193_193281

-- Define the given conditions
def num_phys_ed_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_phys_ed_courses + num_art_courses
def valid_course_combos : finset ℕ := {2, 3}

-- Define the number of selection schemes with the constraints
def selection_schemes : nat :=
  (num_phys_ed_courses.choose 1 * num_art_courses.choose 1) + 
  (num_phys_ed_courses.choose 2 * num_art_courses.choose 1 + 
   num_phys_ed_courses.choose 1 * num_art_courses.choose 2)

-- State the theorem to be proved
theorem total_selection_schemes : selection_schemes = 64 := by
  sorry

end total_selection_schemes_l193_193281


namespace compute_u_l193_193081

variable (a b c : ℝ)

theorem compute_u :
  (frac ac (a + b) + frac ba (b + c) + frac cb (c + a) = -12) →
  (frac bc (a + b) + frac ca (b + c) + frac ab (c + a) = 15) →
  (frac a (a + b) + frac b (b + c) + frac c (c + a) = -12) :=
by
  sorry

end compute_u_l193_193081


namespace solve_equation_l193_193716

theorem solve_equation : 
  ∀ (x : ℝ), 3^(4 * x^2 - 9 * x + 3) = 3^(-4 * x^2 + 15 * x - 11) ↔ 
  (x = (3 + Real.sqrt 2) / 2 ∨ x = (3 - Real.sqrt 2) / 2) :=
by {
  intro x,
  split,
  { intro h,
    -- Forward direction proof
    sorry },
  { intro h,
    -- Backward direction proof
    sorry },
}

end solve_equation_l193_193716


namespace rectangular_frame_wire_and_paper_area_l193_193961

theorem rectangular_frame_wire_and_paper_area :
  let l1 := 3
  let l2 := 4
  let l3 := 5
  let wire_length := (l1 + l2 + l3) * 4
  let paper_area := ((l1 * l2) + (l1 * l3) + (l2 * l3)) * 2
  wire_length = 48 ∧ paper_area = 94 :=
by
  sorry

end rectangular_frame_wire_and_paper_area_l193_193961


namespace two_integer_solutions_iff_l193_193170

theorem two_integer_solutions_iff (a : ℝ) :
  (∃ (n m : ℤ), n ≠ m ∧ |n - 1| < a * n ∧ |m - 1| < a * m ∧
    ∀ (k : ℤ), |k - 1| < a * k → k = n ∨ k = m) ↔
  (1/2 : ℝ) < a ∧ a ≤ (2/3 : ℝ) :=
by
  sorry

end two_integer_solutions_iff_l193_193170


namespace percent_sparrows_non_eagles_l193_193817

-- Definitions for the problem conditions
variables (robins_percent eagles_percent falcons_percent sparrows_percent : ℝ)

-- Define conditions
def conditions : Prop :=
  robins_percent = 0.20 ∧
  eagles_percent = 0.30 ∧
  falcons_percent = 0.15 ∧
  sparrows_percent = 1 - (robins_percent + eagles_percent + falcons_percent)

-- Translate conditions to Lean
theorem percent_sparrows_non_eagles :
  conditions robins_percent eagles_percent falcons_percent sparrows_percent → 
  let non_eagles_percent := 1 - eagles_percent in
  let percent_sparrows_non_eagles := (sparrows_percent / non_eagles_percent) * 100 in
  percent_sparrows_non_eagles = 50 :=
by
  intros h
  sorry

end percent_sparrows_non_eagles_l193_193817


namespace bus_stops_for_10_minutes_per_hour_l193_193987

noncomputable def bus_stoppage_time_per_hour (speed_without_stoppages speed_with_stoppages : ℝ) : ℝ :=
  let distance_lost := speed_without_stoppages - speed_with_stoppages
  let time_in_hours := distance_lost / speed_without_stoppages
  time_in_hours * 60

theorem bus_stops_for_10_minutes_per_hour :
  bus_stoppage_time_per_hour 54 45 = 10 :=
by
  let speed_without_stoppages := 54
  let speed_with_stoppages := 45
  let distance_lost := speed_without_stoppages - speed_with_stoppages
  let time_in_hours := distance_lost / speed_without_stoppages
  have h1 : time_in_hours = 1/6 := by
  sorry
  show bus_stoppage_time_per_hour 54 45 = 10, from
  sorry

end bus_stops_for_10_minutes_per_hour_l193_193987


namespace absolute_value_inequality_range_of_xyz_l193_193177

-- Question 1 restated
theorem absolute_value_inequality (x : ℝ) :
  (|x + 2| + |x + 3| ≤ 2) ↔ -7/2 ≤ x ∧ x ≤ -3/2 :=
sorry

-- Question 2 restated
theorem range_of_xyz (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  -1/2 ≤ x * y + y * z + z * x ∧ x * y + y * z + z * x ≤ 1 :=
sorry

end absolute_value_inequality_range_of_xyz_l193_193177


namespace percentile_45_is_78_normal_distribution_probability_l193_193979

def dataSet : List ℕ := [91, 72, 75, 85, 64, 92, 76, 78, 86, 79]

theorem percentile_45_is_78 : 
  let sortedData := dataSet.qsort (≤)
  let position := (dataSet.length * 45) / 100
  dataSet.nth! position = 78 := 
by
  let sortedData := [64, 72, 75, 76, 78, 79, 85, 86, 91, 92]
  let position := 4.5.ceil
  have h_pos : position = 5 := rfl —round up from 4.5
  rw [h_pos]
  exact rfl

def standard_normal (ξ : ℝ → Prop) := \forall x, ξ x = exp (-x*x/2)

theorem normal_distribution_probability (p : ℝ) (ξ : ℝ → Prop) : 
  standard_normal ξ →
  P(ξ > 1) = p →
  P(-1 ≤ ξ ∧ ξ ≤ 0) = 1 / 2 - p :=
by
  intro h_std h_p
  have : \forall x:Real, ξ x = exp (-x*x/2) := h_std
  calc
    P(-1 ≤ ξ ∧ ξ ≤ 0) = P(1 - 1/2 - p) := by sorry


end percentile_45_is_78_normal_distribution_probability_l193_193979


namespace ratio_area_of_rectangle_to_square_l193_193932

theorem ratio_area_of_rectangle_to_square (s : ℝ) :
  (1.2 * s * 0.8 * s) / (s * s) = 24 / 25 :=
by
  sorry

end ratio_area_of_rectangle_to_square_l193_193932


namespace prob_divisible_by_3_of_three_digits_l193_193573

-- Define the set of digits available
def digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Three digits are to be chosen from this set
def choose_three_digits (s : Finset ℕ) : Finset (Finset ℕ) := s.powerset.filter (λ x, x.card = 3)

-- Define the property of the sum of digits being divisible by 3
def divisible_by_3 (s : Finset ℕ) : Prop := s.sum id % 3 = 0

-- Total combinations of choosing 3 out of 9 digits
def total_combinations : ℕ := (digits.card.choose 3)

-- Valid combinations where sum of digits is divisible by 3
def valid_combinations : Finset (Finset ℕ) := (choose_three_digits digits).filter divisible_by_3

-- Finally, the probability of a three-digit number being divisible by 3
def probability : ℕ × ℕ := (valid_combinations.card, total_combinations)

theorem prob_divisible_by_3_of_three_digits :
  probability = (5, 14) :=
by
  -- Proof to be filled
  sorry

end prob_divisible_by_3_of_three_digits_l193_193573


namespace circumference_tank_A_l193_193529

theorem circumference_tank_A
  (h_A : ℝ := 10) -- height of tank A
  (h_B : ℝ := 6)  -- height of tank B
  (C_B : ℝ := 10) -- circumference of tank B
  (V_A_eq_0_6_V_B : ∀ V_A V_B : ℝ, V_A = 0.6 * V_B) :
  (C_A : ℝ) := 6 :=
by
  sorry

end circumference_tank_A_l193_193529


namespace limit_of_sequence_l193_193326

noncomputable def sequence (n : ℕ) : ℝ := (2^(n+1) + 3^(n+1)) / (2^n + 3^n)

theorem limit_of_sequence : 
  tendsto sequence at_top (𝓝 3) :=
sorry

end limit_of_sequence_l193_193326


namespace area_relation_l193_193459

-- Define the areas of the triangles
variables (a b c : ℝ)

-- Define the condition that triangles T_a and T_c are similar (i.e., homothetic)
-- which implies the relationship between their areas.
theorem area_relation (ha : 0 < a) (hc : 0 < c) (habc : b = Real.sqrt (a * c)) : b = Real.sqrt (a * c) := by
  sorry

end area_relation_l193_193459


namespace john_behind_l193_193469

-- Definitions derived from the problem conditions
def john_speed := 4.2 -- meters per second
def steve_speed := 3.7 -- meters per second
def john_ahead := 2 -- meters
def push_time := 28 -- seconds

-- Calculated values from conditions
def john_distance := john_speed * push_time -- total distance John covered
def steve_distance := steve_speed * push_time -- total distance Steve covered

-- Final statement incorporating the question and answer
theorem john_behind (x : ℝ) : x = john_distance - steve_distance - john_ahead := by
  have h1 : john_distance = 117.6 := by sorry
  have h2 : steve_distance = 103.6 := by sorry
  have h3 : john_distance - steve_distance - john_ahead = 12 := by sorry
  exact h3

end john_behind_l193_193469


namespace tangent_line_through_origin_l193_193099

def function_f (x : ℝ) : ℝ := Real.exp (2 - x)

theorem tangent_line_through_origin (x y : ℝ) : 
  (∃ x₀ : ℝ, y = function_f x₀ - function_f x₀ * (x - x₀) ∧ function_f (x₀) = Real.exp (2 - x₀) ∧ x₀ = -1) → 
  y = -Real.exp 3 * x := 
sorry

end tangent_line_through_origin_l193_193099


namespace find_x_l193_193368

theorem find_x (x : ℝ) (h : sqrt (2 - 5 * x) = 8) : x = -62 / 5 :=
by sorry

end find_x_l193_193368


namespace smallest_munificence_of_monic_quadratic_l193_193711

def munificence (p : ℝ → ℝ) : ℝ :=
  sup (set.image (λ x, abs (p x)) (set.Icc (-1) 1))

theorem smallest_munificence_of_monic_quadratic :
  ∃ f : ℝ → ℝ, (∀ x, f x = x^2 + b * x + c) ∧ (∀ x, (f x = x^2 + b * x + c) → munificence f = 1/2) :=
sorry

end smallest_munificence_of_monic_quadratic_l193_193711


namespace find_m_collinear_l193_193553

noncomputable def collinear_points : Prop := 
  ∃ (m : ℝ), 
    let p1 := (3 : ℝ, -4 : ℝ), 
        p2 := (6 : ℝ, 5 : ℝ), 
        p3 := (8 : ℝ, m) in
    (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

theorem find_m_collinear : collinear_points → ∃ (m : ℝ), m = 11 :=
by
  sorry

end find_m_collinear_l193_193553


namespace angle_B_pi_div_3_l193_193465

theorem angle_B_pi_div_3
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : 2 * b = a + c)
  (h2 : sin A * sin A = sin B * sin C)
  (h3 : b^2 = a * c):
  B = π / 3 :=
by
  sorry

end angle_B_pi_div_3_l193_193465


namespace andy_solves_16_problems_l193_193314

theorem andy_solves_16_problems :
  ∃ N : ℕ, 
    N = (125 - 78)/3 + 1 ∧
    (78 + (N - 1) * 3 <= 125) ∧
    N = 16 := 
by 
  sorry

end andy_solves_16_problems_l193_193314


namespace line_intersects_circle_probability_l193_193082

theorem line_intersects_circle_probability:
  let k_in_interval := ∀ k, k ≥ -1 ∧ k ≤ 1
  let circle_center := (5: ℝ, 0: ℝ)
  let circle_radius := 3
  let line_eq := ∀ x y, y = k * x
  let dist_from_center_to_line := λ k: ℝ, (|5 * k|: ℝ) / Real.sqrt (k^2 + 1)
  let intersects_condition := ∀ k, dist_from_center_to_line k < circle_radius
  ∃ p: ℝ, p = 3 / 4 ∧
  (ProbabilityTheory.Probability (exists k_in_interval, (intersects_condition k)) = p) :=
sorry

end line_intersects_circle_probability_l193_193082


namespace different_testing_methods_1_different_testing_methods_2_l193_193024

-- Definitions used in Lean 4 statement should be derived from the conditions in a).
def total_products := 10
def defective_products := 4
def non_defective_products := total_products - defective_products
def choose (n k : Nat) : Nat := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Statement (1)
theorem different_testing_methods_1 :
  let first_defective := 5
  let last_defective := 10
  let non_defective_in_first_4 := choose 6 4
  let defective_in_middle_5 := choose 5 3
  let total_methods := non_defective_in_first_4 * defective_in_middle_5 * Nat.factorial 5 * Nat.factorial 4
  total_methods = 103680 := sorry

-- Statement (2)
theorem different_testing_methods_2 :
  let first_defective := 5
  let remaining_defective := 4
  let non_defective_in_first_4 := choose 6 4
  let total_methods := non_defective_in_first_4 * Nat.factorial 5
  total_methods = 576 := sorry

end different_testing_methods_1_different_testing_methods_2_l193_193024


namespace isosceles_triangle_inscribed_ellipse_l193_193313

theorem isosceles_triangle_inscribed_ellipse
  (x y : ℝ)
  (h1 : (0, 1) ∈ {p : ℝ × ℝ | p.1 ^ 2 + 9 * p.2 ^ 2 = 9})
  (h2 : ∀ p q : ℝ × ℝ, p ∈ {p : ℝ × ℝ | p.1 ^ 2 + 9 * p.2 ^ 2 = 9} →
                        q ∈ {p : ℝ × ℝ | p.1 ^ 2 + 9 * p.2 ^ 2 = 9} →
                        p.2 = 1 →
                        (q.1 - p.1) ^ 2 + (q.2 - 1) ^ 2 = (q.2 - 1 - 1) ^ 2 → False) 
  (h3 : 0 = 0 * x) :
  let side_length_squared := (6 * real.sqrt 3 / 5) ^ 2 in
  side_length_squared = 108 / 25 := by
      sorry

end isosceles_triangle_inscribed_ellipse_l193_193313


namespace range_of_a_solution_set_of_inequality_l193_193770

-- Lean statement for Part 1
theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  ∀ x : ℝ, x^2 - 2 * a * x + a > 0 :=
by
  sorry

-- Lean statement for Part 2
theorem solution_set_of_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  { x : ℝ | a^(x^2 - 3) < a^(2 * x) ∧ a^(2 * x) < 1 } = { x : ℝ | x > 3 } :=
by
  sorry

end range_of_a_solution_set_of_inequality_l193_193770


namespace no_such_finite_set_exists_l193_193692

def satisfies_property (M : Set ℝ) : Prop :=
  ∀ a b ∈ M, 2 * a - b^2 ∈ M

theorem no_such_finite_set_exists :
  ¬∃ (M : Set ℝ), M.Finite ∧ 2 ≤ M.to_finset.card ∧ satisfies_property M :=
by
  sorry

end no_such_finite_set_exists_l193_193692


namespace combined_cost_l193_193291

variable (bench_cost : ℝ) (table_cost : ℝ)

-- Conditions
axiom bench_cost_def : bench_cost = 250.0
axiom table_cost_def : table_cost = 2 * bench_cost

-- Goal
theorem combined_cost (bench_cost : ℝ) (table_cost : ℝ) 
  (h1 : bench_cost = 250.0) (h2 : table_cost = 2 * bench_cost) : 
  table_cost + bench_cost = 750.0 :=
by
  sorry

end combined_cost_l193_193291


namespace total_number_of_course_selection_schemes_l193_193252

-- Define the total number of courses
def total_courses := 8

-- Define number of physical education and art courses
def pe_courses := 4
def art_courses := 4

-- Define selections: students choose 2 or 3 courses
def course_selections : Finset (Finset ℕ) :=
  (Finset.powerset (Finset.range total_courses)).filter (λ s, s.card = 2 ∨ s.card = 3)

-- Define condition: at least 1 course from each category
def valid_selections : Finset (Finset ℕ) :=
  course_selections.filter (λ s, ∃ pe art, s = pe ∪ art ∧ pe.card ≠ 0 ∧ art.card ≠ 0 ∧ 
                             pe ⊆ Finset.range pe_courses ∧ art ⊆ (Finset.range total_courses).filter (λ x, x ≥ pe_courses))

theorem total_number_of_course_selection_schemes : valid_selections.card = 64 := 
by sorry

end total_number_of_course_selection_schemes_l193_193252


namespace proof_problem_l193_193067

def is_odd (n : ℕ) := ∃ k : ℕ, n = 2 * k + 1
def is_even (n : ℕ) := ∃ k : ℕ, n = 2 * k

lemma odd_2017 : is_odd 2017 := sorry
lemma even_2016 : is_even 2016 := sorry

theorem proof_problem :
  (is_odd 2017 ∨ is_even 2016) = true :=
by
  have p := odd_2017
  have q := even_2016
  exact trivial

end proof_problem_l193_193067


namespace peaches_per_box_l193_193959

theorem peaches_per_box :
  ∀ 
  (peaches_per_basket : ℕ) 
  (baskets : ℕ) 
  (peaches_eaten : ℕ) 
  (boxes : ℕ),
  peaches_per_basket = 25 →
  baskets = 5 →
  peaches_eaten = 5 →
  boxes = 8 →
  (peaches_per_basket * baskets - peaches_eaten) / boxes = 15 :=
by
  intros peaches_per_basket baskets peaches_eaten boxes
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  apply sorry

end peaches_per_box_l193_193959


namespace no_appearance_1234_or_3269_no_reappearance_1975_from_2nd_time_l193_193457

def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 9
  | 2 => 7
  | 3 => 5
  | n + 4 => (seq n + seq (n + 1) + seq (n + 2) + seq (n + 3)) % 10

theorem no_appearance_1234_or_3269 : 
  ¬∃ n, seq n = 1 ∧ seq (n + 1) = 2 ∧ seq (n + 2) = 3 ∧ seq (n + 3) = 4 ∨
  seq n = 3 ∧ seq (n + 1) = 2 ∧ seq (n + 2) = 6 ∧ seq (n + 3) = 9 := 
sorry

theorem no_reappearance_1975_from_2nd_time : 
  ¬∃ n > 0, seq n = 1 ∧ seq (n + 1) = 9 ∧ seq (n + 2) = 7 ∧ seq (n + 3) = 5 :=
sorry

end no_appearance_1234_or_3269_no_reappearance_1975_from_2nd_time_l193_193457


namespace BBB_div_by_9_l193_193103

open Nat

theorem BBB_div_by_9 (B : ℕ) (h1 : 4 * 10^4 + B * 10^3 + B * 10^2 + B * 10 + 2 ≡ 0 [MOD 9]) (h2 : B ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  B = 4 :=
by
  have mod9_eq : (4 * 10^4 + (B + B + B) * 10^2 + 2) ≡ (4 + B + B + B + 2) [MOD 9] := Nat.mod_eq_of_lt
  sorry

end BBB_div_by_9_l193_193103


namespace midpoint_lattice_point_exists_l193_193047

theorem midpoint_lattice_point_exists (S : Finset (ℤ × ℤ)) (hS : S.card = 5) :
  ∃ (p1 p2 : ℤ × ℤ), p1 ∈ S ∧ p2 ∈ S ∧ p1 ≠ p2 ∧
  (∃ (x_mid y_mid : ℤ), 
    (p1.1 + p2.1) = 2 * x_mid ∧
    (p1.2 + p2.2) = 2 * y_mid) :=
by
  sorry

end midpoint_lattice_point_exists_l193_193047


namespace b_catches_a_distance_l193_193167

-- Define the initial conditions
def a_speed : ℝ := 10  -- A's speed in km/h
def b_speed : ℝ := 20  -- B's speed in km/h
def start_delay : ℝ := 3  -- B starts cycling 3 hours after A in hours

-- Define the target distance to prove
theorem b_catches_a_distance : ∃ (d : ℝ), d = 60 := 
by 
  sorry

end b_catches_a_distance_l193_193167


namespace s_3_eq_149_l193_193989

def s (n : Nat) : Nat :=
  Integer.fromNat (String.join (List.map (λ i => toString (i * i)) (List.range n)).toNat).

theorem s_3_eq_149 : s 3 = 149 :=
sorry

end s_3_eq_149_l193_193989


namespace head_start_is_81_l193_193503

def cristina_speed : ℝ := 5
def nicky_speed : ℝ := 3
def nicky_head_start_time : ℝ := 27

theorem head_start_is_81 :
  let H := nicky_speed * nicky_head_start_time in
  H = 81 :=
by
  let H := nicky_speed * nicky_head_start_time
  have : H = 3 * 27 := rfl
  have : H = 81 := by norm_num
  exact this

end head_start_is_81_l193_193503


namespace determine_cyclist_speeds_l193_193137

noncomputable def cyclist_speeds (x : ℝ) : Prop :=
  let t := x in
  let y := x - 1.5 in
  (2 * x - 1.5) * t = 270

theorem determine_cyclist_speeds : cyclist_speeds 12 :=
begin
  rw cyclist_speeds,
  let x := 12,
  let t := 12,
  let y := 12 - 1.5,
  have h1 : (2 * 12 - 1.5) * 12 = 270,
  { norm_num },
  exact h1,
end

end determine_cyclist_speeds_l193_193137


namespace total_course_selection_schemes_l193_193262

theorem total_course_selection_schemes (PE_courses : ℕ) (Art_courses : ℕ) : 
  PE_courses = 4 → Art_courses = 4 → 
  (finset.card (finset.powerset_len 2 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b ∈ s), a < PE_courses ∧ b ≥ PE_courses) + 
   finset.card (finset.powerset_len 3 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b c ∈ s), (a < PE_courses ∧ b < PE_courses ∧ c ≥ PE_courses) ∨ 
                        (a < PE_courses ∧ b ≥ PE_courses ∧ c ≥ PE_courses ∧ a ≠ b ≠ c)) = 64 :=
by
  sorry

end total_course_selection_schemes_l193_193262


namespace janet_wait_time_l193_193837

theorem janet_wait_time 
  (janet_speed : ℝ)
  (sister_speed : ℝ)
  (lake_width : ℝ)
  (h1 : janet_speed = 30)
  (h2 : sister_speed = 12)
  (h3 : lake_width = 60) 
  :
  60 / 12 - 60 / 30 = 3 :=
by
  sorry

end janet_wait_time_l193_193837


namespace total_course_selection_schemes_l193_193268

theorem total_course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose_two :=
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 1)
  let choose_three :=
    (Nat.choose PE_courses 2) * (Nat.choose Art_courses 1) +
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 2)
  total_courses = 8
  ∧ (choose_two + choose_three = 64) :=
by
  sorry

end total_course_selection_schemes_l193_193268
