import Algebra
import Mathlib
import Mathlib.Algebra.Field
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Algebra.Trigonometry.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Parity
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Setoid.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.MeasureTheory.Measure.Space
import Mathlib.Order.Floor
import Mathlib.Probability.ConditionalProbability
import Mathlib.Real
import Mathlib.Tactic
import Mathlib.Topology.Algebra.Order

namespace fraction_of_europeans_l468_468396

noncomputable def total_passengers := 240
noncomputable def north_america_fraction := 1 / 3
noncomputable def africa_fraction := 1 / 5
noncomputable def asia_fraction := 1 / 6
noncomputable def other_continents_count := 42

theorem fraction_of_europeans :
  let P := total_passengers in
  let NA := north_america_fraction * P in
  let AF := africa_fraction * P in
  let AS := asia_fraction * P in
  let other := other_continents_count in
  ∀ E : ℚ, 
  ((NA + AF + AS + E * P + other = P) → 
  (E = 1 / 8)) :=
by
  intro P NA AF AS other E h
  sorry

end fraction_of_europeans_l468_468396


namespace hua_luogeng_optimal_selection_l468_468560

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l468_468560


namespace min_number_of_girls_l468_468910

theorem min_number_of_girls (total_students boys_count girls_count : ℕ) (no_three_equal: ∀ m n : ℕ, m ≠ n → boys_count m ≠ boys_count n) (boys_at_most : boys_count ≤ 2 * (girls_count + 1)) : ∃ d : ℕ, d ≥ 6 ∧ total_students = 20 ∧ boys_count = total_students - girls_count :=
by
  let d := 6
  exists d
  sorry

end min_number_of_girls_l468_468910


namespace time_in_still_water_l468_468255

-- Define the conditions
variable (S x y : ℝ)
axiom condition1 : S / (x + y) = 6
axiom condition2 : S / (x - y) = 8

-- Define the proof statement
theorem time_in_still_water : S / x = 48 / 7 :=
by
  -- The proof is omitted
  sorry

end time_in_still_water_l468_468255


namespace min_girls_in_class_l468_468931

theorem min_girls_in_class (n : ℕ) (d : ℕ) (h1 : n = 20) 
  (h2 : ∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
       ∀ f : ℕ → set ℕ, (card (f i) ≠ card (f j) ∨ card (f j) ≠ card (f k) ∨ card (f i) ≠ card (f k))) : 
d ≥ 6 :=
by
  sorry

end min_girls_in_class_l468_468931


namespace total_elements_in_set_C_l468_468524

theorem total_elements_in_set_C 
  (c d : ℕ)
  (h1 : c = 3 * d)
  (h2 : |C ∪ D| = 4500)
  (h3 : |C ∩ D| = 1200) :
  c = 4275 := by
  sorry

end total_elements_in_set_C_l468_468524


namespace find_c_l468_468259

noncomputable def circle_eq := (x y : ℝ) → x^2 + y^2 = 9
noncomputable def line_eq (c : ℝ) := (x : ℝ) → x + c

def triangle_area_condition (area : ℝ) : Prop := 
  9 ≤ area ∧ area ≤ 36

theorem find_c (c : ℝ) : 
  (∃ (x y : ℝ), circle_eq x y ∧ y = x + c) →
  (let A := (1 / 2) * 3 * (|c| / Real.sqrt 2) in triangle_area_condition (3 * Real.sqrt 2 * |c| / 4)) →
  c ∈ Set.Icc (-4 * Real.sqrt 2) (4 * Real.sqrt 2) :=
sorry

end find_c_l468_468259


namespace optimal_selection_method_uses_golden_ratio_l468_468607

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l468_468607


namespace modulus_eq_sqrt_three_halves_l468_468015

-- Define the complex number z
def z : ℂ := complex.sin (real.pi / 3) - complex.I * complex.cos (real.pi / 6)

-- Define what we want to prove
theorem modulus_eq_sqrt_three_halves : complex.abs z = real.sqrt (3 / 2) := by
  sorry

end modulus_eq_sqrt_three_halves_l468_468015


namespace optimal_selection_method_is_golden_ratio_l468_468581

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l468_468581


namespace hua_luogeng_optimal_selection_l468_468551

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l468_468551


namespace minimum_number_of_girls_l468_468894

theorem minimum_number_of_girls (students boys girls : ℕ) 
(h1 : students = 20) (h2 : boys = students - girls)
(h3 : ∀ d, 0 ≤ d → 2 * (d + 1) ≥ boys) 
: 6 ≤ girls :=
by 
  have h4 : 20 - girls = boys, from h2.symm
  have h5 : 2 * (girls + 1) ≥ boys, from h3 girls (nat.zero_le girls)
  linarith

#check minimum_number_of_girls

end minimum_number_of_girls_l468_468894


namespace optimal_selection_method_uses_golden_ratio_l468_468865

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l468_468865


namespace A_belongs_to_transformed_plane_l468_468986

noncomputable def A := (1/2 : ℝ, 1/3 : ℝ, 1 : ℝ)
def plane (x y z k : ℝ) := 2 * x - 3 * y + 3 * z - 2 * k

theorem A_belongs_to_transformed_plane:
  let k := 1.5 in
  plane (1/2) (1/3) 1 k = 0 := 
by
  -- Given point A
  let A := (1/2 : ℝ, 1/3 : ℝ, 1 : ℝ)
  -- Given k
  let k := 1.5
  -- Proving the point lies on the new plane
  have h: plane A.1 A.2 A.3 k = 0 :=
    calc 
      plane A.1 A.2 A.3 k 
          = 2 * (1/2) - 3 * (1/3) + 3 * 1 - 2 * 1.5 : by rfl
      ... = 1 - 1 + 3 - 3 : by norm_num
      ... = 0 : by norm_num,
  exact h

end A_belongs_to_transformed_plane_l468_468986


namespace ball_selection_probability_l468_468240

theorem ball_selection_probability :
  let total_balls := 500
  let odd_balls := 250
  let even_balls := 250
  let prob_odd := (odd_balls / total_balls : ℚ)
  let prob_even := (even_balls / total_balls : ℚ)
  let prob_sequence := (prob_odd * prob_even * prob_odd * prob_even * prob_odd : ℚ)
  in prob_sequence = 1 / 32 := by
  sorry

end ball_selection_probability_l468_468240


namespace solve_digit_insertion_l468_468441

theorem solve_digit_insertion : 
  ∃ (a b c d e f g h : ℕ),
  (a, b, c, d, e, f, g, h) = (9, 9, 1, 0, 9, 1, 1, 0) ∧
  ( (10 * a + b) + (10 * c + d) ) * ( (10 * e + f) + g ) = 100 * (10 * a + b) + 10 * c + d ∧
  (a ≠ 0 ∧ c ≠ 0 ∧ e ≠ 0) ∧ 
  ∀ x : ℕ, x ∈ {a, b, c, d, e, f, g, h} → x = 9 ∨ x = 1 ∨ x = 0 := 
by
  use 9, 9, 1, 0, 9, 1, 1, 0
  -- Skipping the proof
  sorry

end solve_digit_insertion_l468_468441


namespace sum_first_40_terms_l468_468040

-- Defining the sequence a_n following the given conditions
noncomputable def a : ℕ → ℝ
| 0 => 1
| 1 => 3
| n + 2 => a (n + 1) * a (n - 1)

-- Defining the sum of the first 40 terms of the sequence
noncomputable def S40 := (Finset.range 40).sum a

-- The theorem stating the desired property
theorem sum_first_40_terms : S40 = 60 :=
sorry

end sum_first_40_terms_l468_468040


namespace optimal_selection_method_uses_golden_ratio_l468_468756

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l468_468756


namespace optimalSelectionUsesGoldenRatio_l468_468715

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l468_468715


namespace optimal_selection_method_use_golden_ratio_l468_468804

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l468_468804


namespace integer_sequence_product_l468_468475

theorem integer_sequence_product (p : ℤ) (hp1 : p > 2) (hp2 : ¬ (p % 3 = 0)) :
  ∃ (k : ℕ) (a : fin k → ℤ), 
    (∀ i, -p / 2 < a i ∧ a i < p / 2) ∧ 
    strict_mono a ∧
    ∃ m : ℕ, (∏ i : fin k, (p - a i) / |a i|) = 3 ^ m :=
sorry

end integer_sequence_product_l468_468475


namespace find_angle_between_diagonals_l468_468168

noncomputable def angle_between_diagonals (a b c : ℝ) : ℝ :=
  real.arccos (|(a^2 - b^2)| / (real.sqrt (a^2 + b^2) * real.sqrt (a^2 + b^2 + c^2)))

-- Statement of the problem as a theorem
theorem find_angle_between_diagonals (a b c : ℝ) :
  β = angle_between_diagonals a b c :=
sorry

end find_angle_between_diagonals_l468_468168


namespace does_not_uniquely_determine_l468_468967

-- Define the problem under the type of triangles and given conditions

-- Conditions for unique determination of triangles
def condition_A : Type := { right_triangle : Type } × { length_side : ℝ } × { opposite_angle : ℝ }
def condition_B : Type := { isosceles_triangle : Type } × { angle1 : ℝ } × { angle2 : ℝ }
def condition_C : Type := { equilateral_triangle : Type } × { circumscribed_circle_radius : ℝ } × { side_length : ℝ }
def condition_D : Type := { scalene_triangle : Type } × { side1 : ℝ } × { side2 : ℝ } × { included_angle : ℝ }
def condition_E : Type := { triangle : Type } × { angle1 : ℝ } × { angle2 : ℝ } × { angle3 : ℝ }

-- Proving that condition E does not uniquely determine the triangle
theorem does_not_uniquely_determine :
  ¬ (∀ (T1 T2 : condition_E), T1 = T2) :=
sorry

end does_not_uniquely_determine_l468_468967


namespace optimal_selection_method_uses_golden_ratio_l468_468743

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l468_468743


namespace optimal_selection_method_uses_golden_ratio_l468_468631

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l468_468631


namespace optimal_selection_method_uses_golden_ratio_l468_468754

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l468_468754


namespace lateral_surface_area_of_cone_l468_468882

theorem lateral_surface_area_of_cone (r l : ℝ) (h₁: r = 3) (h₂: l = 10) :
  (Real.pi * r * l) = 30 * Real.pi :=
by
  rw [h₁, h₂]
  exact (mul_assoc Real.pi 3 10).symm


end lateral_surface_area_of_cone_l468_468882


namespace three_digit_integers_count_l468_468020

theorem three_digit_integers_count : 
  ∃ (n : ℕ), n = 24 ∧
  (∃ (digits : Finset ℕ), digits = {2, 4, 7, 9} ∧
  (∀ a b c : ℕ, a ∈ digits → b ∈ digits → c ∈ digits → a ≠ b → b ≠ c → a ≠ c → 
  100 * a + 10 * b + c ∈ {y | 100 ≤ y ∧ y < 1000} → 4 * 3 * 2 = 24)) :=
by
  sorry

end three_digit_integers_count_l468_468020


namespace magnitude_vector_sum_l468_468354

variable {α : Type*} [InnerProductSpace ℝ α]

theorem magnitude_vector_sum 
  (a b : α) 
  (ha : ∥a∥ = 2) 
  (hb : ∥b∥ = 5) 
  (hab : ⟪a, b⟫ = -3) :
  ∥a + b∥ = Real.sqrt 23 := by
sorry

end magnitude_vector_sum_l468_468354


namespace greatest_integer_e_minus_3_l468_468113

noncomputable def e : ℝ := 2.718

theorem greatest_integer_e_minus_3 : 
  int.floor (e - 3) = -1 :=
by
  sorry

end greatest_integer_e_minus_3_l468_468113


namespace possible_lightest_boy_heaviest_girl_l468_468417

theorem possible_lightest_boy_heaviest_girl
    (B G : ℕ)
    (hN : B + G < 35)
    (w_boys w_girls : ℕ → ℝ)
    (avg_all : Real := 53.5)
    (avg_girls : Real := 47)
    (avg_boys : Real := 60) :
    (∑ i in Finset.range B, w_boys i) / B = avg_boys →
    (∑ i in Finset.range G, w_girls i) / G = avg_girls →
    ((∑ i in Finset.range B, w_boys i) + (∑ i in Finset.range G, w_girls i)) / (B + G) = avg_all →
    ∃ (i j : ℕ), i < B ∧ j < G ∧ w_boys i < all_weights_worst(G, w_girls) ∧ w_girls j > all_weights_best(B, w_boys) :=
sorry

/-- Helper function that returns the minimum weight of the given range of girls' weights --/
noncomputable def all_weights_worst (g: ℕ, wg: ℕ → ℝ) : ℝ :=
  Finset.min' (Finset.range g) (begin
    -- there should be at least one girl
    exact ⟨0, by simp [show 0 < g, by omega]⟩
  end)

/-- Helper function that returns the maximum weight of the given range of boys' weights --/
noncomputable def all_weights_best (b: ℕ, wb: ℕ → ℝ) : ℝ :=
  Finset.max' (Finset.range b) (begin
    -- there should be at least one boy
    exact ⟨0, by simp [show 0 < b, by omega]⟩
  end)

end possible_lightest_boy_heaviest_girl_l468_468417


namespace find_b_l468_468347

theorem find_b (b : ℤ) :
  (∃ x : ℤ, x^2 + b * x - 36 = 0 ∧ x = -9) → b = 5 :=
by
  sorry

end find_b_l468_468347


namespace suresh_worked_hours_l468_468160

theorem suresh_worked_hours :
  (suresh_time ashutosh_time remaining_hours : ℝ) (h₁ : suresh_time = 15) (h₂ : ashutosh_time = 15) (h₃ : remaining_hours = 6) :
  let suresh_hours := 9 in
  (suresh_hours / suresh_time) + (remaining_hours / ashutosh_time) = 1 :=
by
  sorry

end suresh_worked_hours_l468_468160


namespace ratio_of_texts_l468_468455

-- Definitions
def texts_about_grocery := 5
def total_texts := 33
def texts_asking := 25
def texts_police := 0.10 * (texts_about_grocery + texts_asking) -- from condition 3

-- Lean statement to be proved
theorem ratio_of_texts :
  let G := texts_about_grocery in
  let R := texts_asking in
  let P := 0.10 * (G + R) in
  G + R + P = total_texts →
  R / G = 5 := by
  sorry

end ratio_of_texts_l468_468455


namespace Q_eq_1_div_10000_l468_468471

def Q (n : ℕ) : ℝ := (∏ k in finset.range' 2 (n+1), (1 - 1 / (k : ℝ))^2)

theorem Q_eq_1_div_10000 : Q 100 = 1 / 10000 := 
by
  -- Proof goes here
  sorry

end Q_eq_1_div_10000_l468_468471


namespace optimal_selection_method_uses_golden_ratio_l468_468672

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468672


namespace optimal_selection_method_uses_golden_ratio_l468_468849

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468849


namespace expected_value_of_10_sided_die_l468_468952

def faces := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def probability (n : ℕ) : ℝ := 1 / 10
def expected_value (X : ℕ) := ∑ i in Finset.range 10 + 1, (probability i.to_nat) * (i.to_nat : ℝ)

/-- The expected value of the roll of a 10-sided die, where the sides are numbered from 1 to 10, is 5.5. -/
theorem expected_value_of_10_sided_die : expected_value 10 = 5.5 :=
sorry

end expected_value_of_10_sided_die_l468_468952


namespace original_inhabitants_proof_l468_468978

noncomputable def original_inhabitants (final_population : ℕ) : ℝ :=
  final_population / (0.75 * 0.9)

theorem original_inhabitants_proof :
  original_inhabitants 5265 = 7800 :=
by
  sorry

end original_inhabitants_proof_l468_468978


namespace optimal_selection_method_uses_golden_ratio_l468_468833

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468833


namespace books_before_purchase_l468_468875

theorem books_before_purchase (x : ℕ) (h : x + 140 = (27 / 25 : ℚ) * x) : x = 1750 :=
sorry

end books_before_purchase_l468_468875


namespace max_right_angled_triangles_in_quadrangular_pyramid_l468_468431

-- Define what a quadrangular pyramid is
structure QuadrangularPyramid :=
(base : Set (ℝ × ℝ × ℝ))
(lateral_faces : Fin 4 → Set (ℝ × ℝ × ℝ))

-- Define what a right-angled triangle is
def is_right_angled_triangle (face : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ × ℝ × ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  (dist a b) * (dist b c) = (dist a c) * (dist b a) / (dist b c) ∧ -- Pythagorean condition for right angle

-- Define the maximum number of right-angled triangular faces in a quadrangular pyramid
noncomputable def max_right_angled_triangles (p : QuadrangularPyramid) : ℕ :=
  4

-- The theorem statement
theorem max_right_angled_triangles_in_quadrangular_pyramid (p : QuadrangularPyramid) :
  (∃ (faces : Fin 4 → Set (ℝ × ℝ × ℝ)), ∀ i, is_right_angled_triangle (faces i)) → max_right_angled_triangles p = 4 :=
sorry

end max_right_angled_triangles_in_quadrangular_pyramid_l468_468431


namespace range_of_a_l468_468036

variable (a x : ℝ)

-- Proposition p
def p : Prop := (1 / 2 <= x ∧ x <= 1)

-- Proposition q
def q : Prop := (x^2 - (2 * a + 1) * x + a * (a + 1) <= 0)

-- The condition that the negation of p is a necessary but not sufficient condition for the negation of q
def condition : Prop := (¬p → ¬q) ∧ (¬q → ¬p → False)

theorem range_of_a : condition a x → 0 ≤ a ∧ a ≤ 1 / 2 :=
by
  sorry

end range_of_a_l468_468036


namespace optimalSelectionUsesGoldenRatio_l468_468711

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l468_468711


namespace probability_three_standard_dice_sum_3_l468_468965

noncomputable def probability_sum_three_dice_eq_3 : ℚ :=
  let outcomes := (finset.range 6).product (finset.range 6).product (finset.range 6)
  let favorable := outcomes.filter (λ abc : ℕ × (ℕ × ℕ), abc.1 + abc.2.1 + abc.2.2 = 2)
  (favorable.card : ℚ) / (outcomes.card : ℚ)

theorem probability_three_standard_dice_sum_3 :
  probability_sum_three_dice_eq_3 = (1/216 : ℚ) :=
sorry

end probability_three_standard_dice_sum_3_l468_468965


namespace find_y_l468_468386

theorem find_y (y : ℕ) : (1 / 8) * 2^36 = 8^y → y = 11 := by
  sorry

end find_y_l468_468386


namespace more_than_one_hundred_percent_l468_468997

-- Define the constants for total students, percentage of boys, and the number of students
def Total_students := 113.38934190276818
def Percent_boys := 0.70
def Num_students := 90

-- Define the statement that 90 students represent more than 100% of the total number of boys
theorem more_than_one_hundred_percent :
  Num_students / (Percent_boys * Total_students) > 1 :=
by
  sorry

end more_than_one_hundred_percent_l468_468997


namespace hotel_charge_comparison_l468_468541

def charge_R (R G : ℝ) (P : ℝ) : Prop :=
  P = 0.8 * R ∧ P = 0.9 * G

def discounted_charge_R (R2 : ℝ) (R : ℝ) : Prop :=
  R2 = 0.85 * R

theorem hotel_charge_comparison (R G P R2 : ℝ)
  (h1 : charge_R R G P)
  (h2 : discounted_charge_R R2 R)
  (h3 : R = 1.125 * G) :
  R2 = 0.95625 * G := by
  sorry

end hotel_charge_comparison_l468_468541


namespace matilda_smartphone_loss_percentage_l468_468503

theorem matilda_smartphone_loss_percentage :
  ∀ (initial_cost selling_price : ℝ),
  initial_cost = 300 →
  selling_price = 255 →
  let loss := initial_cost - selling_price in
  let percentage_loss := (loss / initial_cost) * 100 in
  percentage_loss = 15 :=
by
  intros initial_cost selling_price h₁ h₂
  let loss := initial_cost - selling_price
  let percentage_loss := (loss / initial_cost) * 100
  rw [h₁, h₂]
  sorry

end matilda_smartphone_loss_percentage_l468_468503


namespace optimalSelectionUsesGoldenRatio_l468_468719

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l468_468719


namespace find_y_l468_468384

theorem find_y (y : ℕ) 
  (h : (1/8) * 2^36 = 8^y) : y = 11 :=
sorry

end find_y_l468_468384


namespace conjugate_of_z_is_neg_i_l468_468014

-- Definitions based on conditions in part (a)
def z : ℂ := 1 / (complex.I ^ 3)

-- The statement that needs proof, corresponding to part (c)
theorem conjugate_of_z_is_neg_i : complex.conj z = -complex.I :=
by
  sorry

end conjugate_of_z_is_neg_i_l468_468014


namespace middle_term_binom_expansion_sum_coeff_except_constant_term_term_largest_coefficient_l468_468318

theorem middle_term_binom_expansion : 
  ∀ x : ℝ, 
  (1 - x) ^ 10 = (1 - 10x + 45x^2 - 120x^3 + 210x^4 - 252x^5 + 210x^6 - 120x^7 + 45x^8 - 10x^9 + x^10) → 
  ∃ t : ℤ, 
  t = -252 * x^5 :=
by
  sorry

theorem sum_coeff_except_constant_term : 
  ∀ x : ℝ, 
  (1 - x) ^ 10 = (1 - 10x + 45x^2 - 120x^3 + 210x^4 - 252x^5 + 210x^6 - 120x^7 + 45x^8 - 10x^9 + x^10) → 
  ∑ i in (finset.range 10).erase 0, (-1)^i * (binom 10 i) = -1 :=
by
  sorry

theorem term_largest_coefficient : 
  ∀ x : ℝ, 
  (1 - x) ^ 10 = (1 - 10x + 45x^2 - 120x^3 + 210x^4 - 252x^5 + 210x^6 - 120x^7 + 45x^8 - 10x^9 + x^10) → 
  ∃ t : ℤ, 
  t = 210 * x^4 :=
by
  sorry

end middle_term_binom_expansion_sum_coeff_except_constant_term_term_largest_coefficient_l468_468318


namespace hua_luogeng_optimal_selection_method_l468_468725

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l468_468725


namespace part1_general_formula_part2_max_n_l468_468011

variable {n : ℕ}

-- given conditions
def S : ℕ → ℕ := sorry
def a : ℕ → ℕ
| 1 := 4
| (n+1) := 2 * a n

-- problem statement part 1: general formula for a_n
theorem part1_general_formula (n : ℕ) : a n = 2^(n+1) := sorry

-- product T_n of the first n terms of the sequence a_n
def T : ℕ → ℕ 
| 0 := 1
| (n+1) := T n * a (n+1)

-- condition to find maximum n
def condition (n : ℕ) : Prop := n^2 + 3 * n - 40 ≤ 0

-- problem statement part 2: maximum n when log2 T_n ≤ 20
theorem part2_max_n (h : ∀ n, T n ≤ 2^20) : ∃ m, condition m ∧ ∀ k, condition k → k ≤ m := sorry

end part1_general_formula_part2_max_n_l468_468011


namespace roots_cubic_sum_l468_468111

noncomputable def polynomial_roots {R : Type*} [comm_ring R] [is_domain R] : polynomial R := 
  polynomial.mk 5 0 505 0 1010

variables (a b c : ℂ) -- Assuming complex roots for generality
variables (h1 : polynomial_roots.eval a = 0)
variables (h2 : polynomial_roots.eval b = 0)
variables (h3 : polynomial_roots.eval c = 0)

theorem roots_cubic_sum :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 606 :=
sorry

end roots_cubic_sum_l468_468111


namespace x_minus_y_possible_values_l468_468392

theorem x_minus_y_possible_values (x y : ℝ) (hx : x^2 = 9) (hy : |y| = 4) (hxy : x < y) : x - y = -1 ∨ x - y = -7 := 
sorry

end x_minus_y_possible_values_l468_468392


namespace optimal_selection_uses_golden_ratio_l468_468662

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l468_468662


namespace size_of_johns_donation_l468_468055

-- Define the conditions
def original_average_contribution := 75 / 1.5
def number_of_contributions_before_john := 1

-- Define the function to calculate John's donation
def johns_donation (A : ℝ) (n : ℕ) := 
  (n + 1) * 75 - n * A

-- The theorem we want to prove
theorem size_of_johns_donation :
  johns_donation original_average_contribution number_of_contributions_before_john = 100 := 
by
  sorry

end size_of_johns_donation_l468_468055


namespace optimal_selection_uses_golden_ratio_l468_468639

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l468_468639


namespace arithmetic_mean_of_weights_l468_468133

open Set

def sum_of_elements (S_i : Finset ℕ) : ℕ := S_i.sum id

def average_weight_non_empty_subsets (n : ℕ) : ℚ :=
  let S := Finset.range (n + 1).filter (λ x, x > 0)
  let non_empty_subsets := S.powerset.filter (λ T, T ≠ ∅)
  let total_weight := non_empty_subsets.sum sum_of_elements
  total_weight / ((2^n) - 1)

theorem arithmetic_mean_of_weights (n : ℕ) : 
  average_weight_non_empty_subsets n = (n * (n + 1) * 2^(n-2)) / ((2^n) - 1) :=
by
  sorry

end arithmetic_mean_of_weights_l468_468133


namespace concert_total_revenue_l468_468196

def ticket_price : ℝ := 20

def discount (count : ℕ) : ℝ :=
  if 3 ≤ count ∧ count ≤ 4 then 0.9
  else if 5 ≤ count ∧ count ≤ 6 then 0.8
  else if 7 ≤ count ∧ count ≤ 8 then 0.7
  else if 9 ≤ count ∧ count ≤ 10 then 1.1
  else if 10 < count then 1.2
  else 1

def group_sizes : List (ℕ × ℕ) :=
  [(8, 2), (5, 3), (3, 4), (2, 5), (1, 6), (1, 9)]

def total_revenue (price : ℝ) (groups : List (ℕ × ℕ)) : ℝ :=
  groups.foldr (λ (g : ℕ × ℕ) (s : ℝ), s + (g.fst * g.snd * price * discount g.snd)) 0

theorem concert_total_revenue : total_revenue ticket_price group_sizes = 1260 := by
  sorry

end concert_total_revenue_l468_468196


namespace optimal_selection_method_uses_golden_ratio_l468_468678

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468678


namespace optimalSelectionUsesGoldenRatio_l468_468722

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l468_468722


namespace optimal_selection_method_use_golden_ratio_l468_468797

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l468_468797


namespace count_valid_three_digit_integers_is_24_l468_468022

-- Define the set of digits available
def digits : Finset ℕ := {2, 4, 7, 9}

-- Define a function to count the number of valid three-digit integers
noncomputable def count_three_digit_integers : Nat := 
  Finset.card (Finset.filter (λ n : ℕ, 
    ∃ h₁ h₂ h₃, n = h₁ * 100 + h₂ * 10 + h₃ ∧ 
      h₁ ∈ digits ∧ h₂ ∈ digits ∧ h₃ ∈ digits ∧ 
      h₁ ≠ h₂ ∧ h₁ ≠ h₃ ∧ h₂ ≠ h₃
  ) (Finset.range 1000) )

-- The theorem stating the total number of different three-digit integers
theorem count_valid_three_digit_integers_is_24 : count_three_digit_integers = 24 :=
by
  sorry

end count_valid_three_digit_integers_is_24_l468_468022


namespace optimal_selection_method_uses_golden_ratio_l468_468840

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468840


namespace quadrilateral_diagonal_relation_l468_468049

variables (a b c d m n : ℝ) (A C : ℝ) -- Declare relevant variables

-- Define the proof problem as a Lean theorem statement
theorem quadrilateral_diagonal_relation
  (a b c d m n : ℝ) 
  (A C : ℝ)
  (h1 : m = (a^2 + c^2 - 2*a*c*cos A).sqrt) -- Assuming Ptolemy's relation for diagonals
  (h2 : n = (b^2 + d^2 - 2*b*d*cos C).sqrt) -- Assuming Ptolemy's relation for diagonals
  : m^2 * n^2 = a^2 * c^2 + b^2 * d^2 - 2 * a * b * c * d * cos (A + C) :=
sorry -- Proof omitted

end quadrilateral_diagonal_relation_l468_468049


namespace roots_are_positive_integers_implies_r_values_l468_468065

theorem roots_are_positive_integers_implies_r_values (r x : ℕ) (h : (r * x^2 - (2 * r + 7) * x + (r + 7) = 0) ∧ (x > 0)) :
  r = 7 ∨ r = 0 ∨ r = 1 :=
by
  sorry

end roots_are_positive_integers_implies_r_values_l468_468065


namespace isosceles_triangles_count_l468_468506

/- Define the vertices of the triangles -/
def triangle1 := [(1, 6), (3, 6), (2, 3)]
def triangle2 := [(4, 2), (4, 4), (6, 2)]
def triangle3 := [(0, 0), (3, 1), (6, 0)]
def triangle4 := [(7, 3), (6, 5), (9, 3)]
def triangle5 := [(8, 0), (9, 2), (10, 0)]

/- Function to calculate the distance between two points -/
def distance (p1 p2 : ℝ × ℝ) := Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

/- Define a function to check if a triangle is isosceles -/
def is_isosceles (triangle : List (ℝ × ℝ)) : Prop :=
  let d1 := distance triangle.nth 0 triangle.nth 1
  let d2 := distance triangle.nth 0 triangle.nth 2
  let d3 := distance triangle.nth 1 triangle.nth 2
  (d1 = d2) ∨ (d2 = d3) ∨ (d1 = d3)

/- Statement of the problem in Lean 4 -/
theorem isosceles_triangles_count : (
  is_isosceles triangle1 ∧
  is_isosceles triangle2 ∧
  is_isosceles triangle3 ∧
  ¬ is_isosceles triangle4 ∧
  is_isosceles triangle5
) → (4 : ℕ) := sorry -- Proof is omitted

end isosceles_triangles_count_l468_468506


namespace set_problem_l468_468041

open Set

def A := {0, 1, 2, 4, 5, 7}
def B := {1, 3, 6, 8, 9}
def C := {3, 7, 8}

theorem set_problem :
  (A ∩ B) ∪ C = {1, 3, 7, 8} ∧ (A ∪ C) ∩ (B ∪ C) = {1, 3, 7, 8} :=
by
  sorry

end set_problem_l468_468041


namespace quadratic_solution_identity_l468_468470

noncomputable theory
open Classical

theorem quadratic_solution_identity :
  let p q : ℝ := sorry in
  (5 * p ^ 3 - 5 * q ^ 3) / (p - q) = 185 / 9 :=
by
  let p := sorry
  let q := sorry
  have h_eq : 3 * p ^ 2 - 7 * p + 4 = 0 := sorry
  have h_eq2 : 3 * q ^ 2 - 7 * q + 4 = 0 := sorry
  sorry

end quadratic_solution_identity_l468_468470


namespace prove_a2_b2_c2_zero_l468_468107

theorem prove_a2_b2_c2_zero (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) (h5 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) : a^2 + b^2 + c^2 = 0 := 
by 
  sorry

end prove_a2_b2_c2_zero_l468_468107


namespace optimal_selection_uses_golden_ratio_l468_468668

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l468_468668


namespace fraction_in_jug_x_after_pouring_water_l468_468226

-- Define capacities and initial fractions
def initial_fraction_x := 1 / 4
def initial_fraction_y := 2 / 3
def fill_needed_y := 1 - initial_fraction_y -- 1/3

-- Define capacity of original jugs
variable (C : ℚ) -- We can assume capacities are rational for simplicity

-- Define initial water amounts in jugs x and y
def initial_water_x := initial_fraction_x * C
def initial_water_y := initial_fraction_y * C

-- Define the water needed to fill jug y
def additional_water_needed_y := fill_needed_y * C

-- Define the final fraction of water in jug x
def final_fraction_x := initial_fraction_x / 2 -- since half of the initial water is poured out

theorem fraction_in_jug_x_after_pouring_water :
  final_fraction_x = 1 / 8 := by
  sorry

end fraction_in_jug_x_after_pouring_water_l468_468226


namespace ensure_mixed_tablets_l468_468239

theorem ensure_mixed_tablets (A B : ℕ) (total : ℕ) (hA : A = 10) (hB : B = 16) (htotal : total = 18) :
  ∃ (a b : ℕ), a + b = total ∧ a ≤ A ∧ b ≤ B ∧ a > 0 ∧ b > 0 :=
by
  sorry

end ensure_mixed_tablets_l468_468239


namespace min_number_of_girls_l468_468909

theorem min_number_of_girls (total_students boys_count girls_count : ℕ) (no_three_equal: ∀ m n : ℕ, m ≠ n → boys_count m ≠ boys_count n) (boys_at_most : boys_count ≤ 2 * (girls_count + 1)) : ∃ d : ℕ, d ≥ 6 ∧ total_students = 20 ∧ boys_count = total_students - girls_count :=
by
  let d := 6
  exists d
  sorry

end min_number_of_girls_l468_468909


namespace doubled_marks_new_average_l468_468225

theorem doubled_marks_new_average (avg_marks : ℝ) (num_students : ℕ) (h_avg : avg_marks = 36) (h_num : num_students = 12) : 2 * avg_marks = 72 :=
by
  sorry

end doubled_marks_new_average_l468_468225


namespace number_of_ways_to_distribute_books_l468_468245

-- Define the total number of books.
def total_books := 8

-- Define the conditions for books in the library.
def is_valid_distribution (library_books checked_out_books : ℕ) : Prop :=
  library_books > 0 ∧ checked_out_books > 0 ∧ (library_books % 2 = 0)

-- The theorem stating there are exactly 3 valid distributions.
theorem number_of_ways_to_distribute_books :
  ∃ (ways : ℕ), ways = 3 ∧ 
    (∑ n in (finset.filter (λ n, is_valid_distribution n (total_books - n)) (finset.range (total_books + 1))), 1) = ways :=
by sorry

end number_of_ways_to_distribute_books_l468_468245


namespace range_of_omega_l468_468134

theorem range_of_omega (ω : ℝ) (hω : ω > 0)
    (∀ x ∈ set.Icc (- (real.pi / 5)) (real.pi / 4), 
        differentiable_at ℝ (λ x, 4 * real.sin (ω / 2 * x) * (1 / 2 * real.cos (ω / 2 * x)) + 1) x ∧ 
        (∀ x ∈ set.Icc (- (real.pi / 5)) (real.pi / 4), 
            deriv (λ x, 4 * real.sin (ω / 2 * x) * (1 / 2 * real.cos (ω / 2 * x)) + 1) x ≥ 0)) 
    :  0 < ω ∧ ω ≤ 2 :=
begin
  sorry 
end

end range_of_omega_l468_468134


namespace optimal_selection_method_uses_golden_ratio_l468_468851

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l468_468851


namespace percentage_loss_is_correct_l468_468497

noncomputable def initial_cost : ℝ := 300
noncomputable def selling_price : ℝ := 255
noncomputable def loss : ℝ := initial_cost - selling_price
noncomputable def percentage_loss : ℝ := (loss / initial_cost) * 100

theorem percentage_loss_is_correct :
  percentage_loss = 15 :=
sorry

end percentage_loss_is_correct_l468_468497


namespace min_girls_in_class_l468_468917

theorem min_girls_in_class : ∃ d : ℕ, 20 - d ≤ 2 * (d + 1) ∧ d ≥ 6 := by
  sorry

end min_girls_in_class_l468_468917


namespace find_value_of_a2_b2_c2_l468_468103

variable {a b c : ℝ}

theorem find_value_of_a2_b2_c2
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : a + b + c = 0)
  (h3 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 5 := 
sorry

end find_value_of_a2_b2_c2_l468_468103


namespace optimal_selection_method_is_golden_ratio_l468_468583

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l468_468583


namespace ratio_of_segments_l468_468203

-- Definitions
variables {a b : ℝ}
variables (A B C D O : Type)

-- Hypotheses
axiom common_vertex : ∀ (A B C D : Type), O ∈ A ∧ O ∈ B ∧ O ∈ C ∧ O ∈ D
axiom side_lengths : ∀ {a b : ℝ}, a > 0 ∧ b > 0
axiom diagonals : ∀ {a b : ℝ}, ∃ (AB CD : ℝ), AB = sqrt 2 * a ∧ CD = sqrt 2 * b

-- Theorem
theorem ratio_of_segments (a b : ℝ) (AB CD : ℝ) : a > 0 → b > 0 → AB = sqrt 2 * a → CD = sqrt 2 * b → AB / CD = 1 / sqrt 2 :=
by 
  intros h1 h2 h3 h4
  sorry

end ratio_of_segments_l468_468203


namespace big_al_bananas_l468_468281

-- Define conditions for the arithmetic sequence and total consumption
theorem big_al_bananas (a : ℕ) : 
  (a + (a + 6) + (a + 12) + (a + 18) + (a + 24) = 100) → 
  (a + 24 = 32) :=
by
  sorry

end big_al_bananas_l468_468281


namespace h_constant_if_polynomial_l468_468100

noncomputable def f : ℤ → ℕ+ := sorry
noncomputable def h (x y : ℤ) : ℕ+ := Int.gcd (f x) (f y)

theorem h_constant_if_polynomial (H : ∀ x y : ℤ, ∃ p : Polynomial ℤ, h x y = p.eval2 Polynomial.C Polynomial.C x y) : 
  ∃ c : ℕ+, ∀ x y : ℤ, h x y = c :=
sorry

end h_constant_if_polynomial_l468_468100


namespace eval_infinite_series_eq_4_l468_468299

open BigOperators

noncomputable def infinite_series_sum : ℝ :=
  ∑' k, (k^2) / (3^k)

theorem eval_infinite_series_eq_4 : infinite_series_sum = 4 := 
  sorry

end eval_infinite_series_eq_4_l468_468299


namespace optimal_selection_uses_golden_ratio_l468_468665

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l468_468665


namespace positive_difference_even_odd_sums_l468_468954

theorem positive_difference_even_odd_sums:
  let S25 := 2 * (List.range 25).sum
  let S10 := (List.range 10).sum + 10
  650 - S10 = 550 :=
by
  let S25 := 2 * (List.range 25).sum
  let S10 := (List.range 10).sum + 10
  have h1 : S25 = 650 := sorry
  have h2 : S10 = 100 := sorry
  have h3 : 650 - S10 = 550 := by
    rw [h1, h2]
    exact rfl
  exact h3

end positive_difference_even_odd_sums_l468_468954


namespace optimal_selection_method_is_golden_ratio_l468_468591

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l468_468591


namespace segments_form_right_triangle_l468_468429

noncomputable theory

-- Definitions of the quadrilateral and its properties
variables {A B C D : Type} [ConvexQuadrilateral A B C D]

-- Given conditions
variables (h1 : ∠ A D B + ∠ A C B = 30)
variables (h2 : ∠ C A B + ∠ D B A = 30)
variables (h3 : A D = B C)

-- Prove that DB, CA, DC form a right triangle.
theorem segments_form_right_triangle :
    ∃ (E P: Type), isRightTriangle (D B) (C A) (D C) := sorry

end segments_form_right_triangle_l468_468429


namespace lightest_boy_heaviest_girl_l468_468412

theorem lightest_boy_heaviest_girl :
  ∃ (B G : ℕ), B + G < 35 ∧
  (∃ (wb : ℕ → ℝ), (∀ i, wb i > 0) ∧ (∑ i in Finset.range B, wb i) = 60 * B ∧ (∃ i, wb i = min (Finset.range B) wb)) ∧
  (∃ (wg : ℕ → ℝ), (∀ i, wg i > 0) ∧ (∑ i in Finset.range G, wg i) = 47 * G ∧ (∃ i, wg i = max (Finset.range G) wg)) ∧
  (∑ i in Finset.range B, wb i + ∑ i in Finset.range G, wg i) = 53.5 * (B + G) :=
begin
  sorry
end

end lightest_boy_heaviest_girl_l468_468412


namespace area_scaled_by_determinant_l468_468110

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 2],
  ![5, -4]
]

theorem area_scaled_by_determinant (T : Type) (areaT : ℝ) (h1: areaT = 15) (T' : Type) :
  |det A| * areaT = 330 := by
  sorry

end area_scaled_by_determinant_l468_468110


namespace optimal_selection_method_is_golden_ratio_l468_468585

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l468_468585


namespace derivative_of_func_l468_468308

theorem derivative_of_func : ∀ x : ℝ, deriv (λ x, x^2 * exp (2 * x)) x = exp (2 * x) * (2 * x + 2 * x^2) :=
by
  intro x
  sorry

end derivative_of_func_l468_468308


namespace min_girls_in_class_l468_468933

theorem min_girls_in_class (n : ℕ) (d : ℕ) (h1 : n = 20) 
  (h2 : ∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
       ∀ f : ℕ → set ℕ, (card (f i) ≠ card (f j) ∨ card (f j) ≠ card (f k) ∨ card (f i) ≠ card (f k))) : 
d ≥ 6 :=
by
  sorry

end min_girls_in_class_l468_468933


namespace increasing_condition_minimum_condition_l468_468367

noncomputable def f (a : ℝ) (x : ℝ) := Real.log x + (2 * a) / x

theorem increasing_condition (a : ℝ) :
  (∀ x ≥ 1, Real.log x + (2 * a) / x) ≥ 0 → a ≤ 1 / 2 := 
sorry

theorem minimum_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 Real.exp 1, Real.log x + (2 * a) / x ≥ 2) → a = Real.exp 1 / 2 := 
sorry

end increasing_condition_minimum_condition_l468_468367


namespace composite_2011_2014_composite_2012_2015_l468_468091

theorem composite_2011_2014 :
  let N := 2011 * 2012 * 2013 * 2014 + 1
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ N = a * b := 
by
  let N := 2011 * 2012 * 2013 * 2014 + 1
  sorry
  
theorem composite_2012_2015 :
  let N := 2012 * 2013 * 2014 * 2015 + 1
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ N = a * b := 
by
  let N := 2012 * 2013 * 2014 * 2015 + 1
  sorry

end composite_2011_2014_composite_2012_2015_l468_468091


namespace g3_power4_l468_468528

variable (f g : ℝ → ℝ)

def condition1 (x : ℝ) (hx : x ≥ 1) : f (g x) = x^3 := sorry
def condition2 (x : ℝ) (hx : x ≥ 1) : g (f x) = x^4 := sorry
def condition3 : g 81 = 3 := sorry

theorem g3_power4 : [g 3]^4 = 3^(3 / 4) :=
by
  have hx : 3 ≥ 1 := by norm_num
  rw ← condition1 _ hx
  sorry

end g3_power4_l468_468528


namespace quadratic_trinomial_divisibility_or_factorization_l468_468458

/-- 
  Let P(n) be a quadratic trinomial with integer coefficients.
  For each positive integer n, P(n) has a proper divisor d_n, 
  i.e., 1 < d_n < P(n), such that the sequence d_1, d_2, d_3, ... 
  is increasing. Prove that either P(n) is the product of two linear 
  polynomials with integer coefficients or all the values of P(n), 
  for positive integers n, are divisible by the same integer m > 1.
-/
theorem quadratic_trinomial_divisibility_or_factorization 
  (P : ℕ → ℤ)
  (hP : ∃ a b c : ℤ, ∀ n : ℕ, P n = a * n * n + b * n + c)
  (hdiv : ∀ n : ℕ, ∃ d : ℤ, 1 < d ∧ d < P n ∧ ∃ m > n, d = P m)
  (hinc : ∀ ⦃a b⦄, a < b → (∃ d : ℤ, d ∈ range P ∧ d > a ∧ d < b) → a = b) :
  (∃ r s : ℤ, ∀ n : ℕ, P n = (n - r) * (n - s)) ∨ 
  (∃ m : ℤ, m > 1 ∧ ∀ n : ℕ, n > 0 → m ∣ P n) :=
sorry

end quadratic_trinomial_divisibility_or_factorization_l468_468458


namespace angle_in_second_quadrant_l468_468389

theorem angle_in_second_quadrant (theta : ℝ) (h : theta = 3) : 
  (real.pi / 2 < theta) ∧ (theta < real.pi) :=
by
  rw h
  -- now the problem reduces to showing pi/2 < 3 < pi
  split
  · sorry -- prove real.pi / 2 < 3
  · sorry -- prove 3 < real.pi

end angle_in_second_quadrant_l468_468389


namespace cos_B_value_perimeter_of_triangle_l468_468407

theorem cos_B_value (a b c : ℝ) (A B C : ℝ)
  (h1 : cos A = 3 / 4) (h2 : C = 2 * A) :
  cos B = 9 / 16 := 
sorry

theorem perimeter_of_triangle (a b c : ℝ)
  (h3 : a * c = 24) 
  (h4 : a = 4) (h5 : c = 6) (h6 : b = 5) :
  a + b + c = 15 :=
sorry

end cos_B_value_perimeter_of_triangle_l468_468407


namespace optimal_selection_method_uses_golden_ratio_l468_468563

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l468_468563


namespace optimal_selection_method_uses_golden_ratio_l468_468866

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l468_468866


namespace find_number_l468_468305

theorem find_number (n : ℕ) :
  (n % 12 = 11) ∧
  (n % 11 = 10) ∧
  (n % 10 = 9) ∧
  (n % 9 = 8) ∧
  (n % 8 = 7) ∧
  (n % 7 = 6) ∧
  (n % 6 = 5) ∧
  (n % 5 = 4) ∧
  (n % 4 = 3) ∧
  (n % 3 = 2) ∧
  (n % 2 = 1)
  → n = 27719 := 
sorry

end find_number_l468_468305


namespace third_vertex_y_coordinate_correct_l468_468272

noncomputable def third_vertex_y_coordinate (x1 y1 x2 y2 : ℝ) (h : y1 = y2) (h_dist : |x1 - x2| = 10) : ℝ :=
  y1 + 5 * Real.sqrt 3

theorem third_vertex_y_coordinate_correct : 
  third_vertex_y_coordinate 3 4 13 4 rfl (by norm_num) = 4 + 5 * Real.sqrt 3 :=
by
  sorry

end third_vertex_y_coordinate_correct_l468_468272


namespace optimal_selection_method_use_golden_ratio_l468_468807

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l468_468807


namespace optimal_selection_golden_ratio_l468_468695

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l468_468695


namespace optimal_selection_method_is_golden_ratio_l468_468592

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l468_468592


namespace jane_started_babysitting_at_age_18_l468_468094

-- Define the age Jane started babysitting
def jane_starting_age := 18

-- State Jane's current age
def jane_current_age : ℕ := 34

-- State the years since Jane stopped babysitting
def years_since_jane_stopped := 12

-- Calculate Jane's age when she stopped babysitting
def jane_age_when_stopped : ℕ := jane_current_age - years_since_jane_stopped

-- State the current age of the oldest person she could have babysat
def current_oldest_child_age : ℕ := 25

-- Calculate the age of the oldest child when Jane stopped babysitting
def age_oldest_child_when_stopped : ℕ := current_oldest_child_age - years_since_jane_stopped

-- State the condition that the child was no more than half her age at the time
def child_age_condition (jane_age : ℕ) (child_age : ℕ) : Prop := child_age ≤ jane_age / 2

-- The theorem to prove the age Jane started babysitting
theorem jane_started_babysitting_at_age_18
  (jane_current : jane_current_age = 34)
  (years_stopped : years_since_jane_stopped = 12)
  (current_oldest : current_oldest_child_age = 25)
  (age_when_stopped : jane_age_when_stopped = 22)
  (child_when_stopped : age_oldest_child_when_stopped = 13)
  (child_condition : ∀ {j : ℕ}, child_age_condition j age_oldest_child_when_stopped → False) :
  jane_starting_age = 18 :=
sorry

end jane_started_babysitting_at_age_18_l468_468094


namespace john_must_solve_at_least_17_correct_l468_468073

theorem john_must_solve_at_least_17_correct :
  ∀ (x : ℕ), 25 = 20 + 5 → 7 * x - (20 - x) + 2 * 5 ≥ 120 → x ≥ 17 :=
by
  intros x h1 h2
  -- Remaining steps will be included in the proof
  sorry

end john_must_solve_at_least_17_correct_l468_468073


namespace optimal_selection_method_uses_golden_ratio_l468_468746

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l468_468746


namespace optimal_selection_method_uses_golden_ratio_l468_468624

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l468_468624


namespace bounded_functions_sup_eq_max_l468_468516

noncomputable def g1 (s : ℝ) : ℝ := 2
noncomputable def g2 (s : ℝ) : ℝ := 1
noncomputable def h1 (t : ℝ) : ℝ := (Real.log 2) * 2^t
noncomputable def h2 (t : ℝ) : ℝ := (1 - t * Real.log 2) * 2^t

theorem bounded_functions_sup_eq_max :
  ∀ (x : ℝ), 
    (∀ s, 1 ≤ g1 s) → 
    (∀ s, 1 ≤ g2 s) → 
    (∀ s, g1 s < ∞) → 
    (∀ s, g2 s < ∞) → 
    sup (λ s, (g1 s)^x * (g2 s)) = max (λ t, x * (h1 t) + h2 t) :=
by {
  intros x h1_bound h2_bound _ _,
  -- Proof omitted
  sorry
}

end bounded_functions_sup_eq_max_l468_468516


namespace interest_rate_second_bank_l468_468218

theorem interest_rate_second_bank :
  ∃ r : ℝ, r = 0.065 ∧
  let total_investment := 5000 in
  let investment_first_bank := 1700 in
  let interest_rate_first_bank := 0.04 in
  let total_interest := 282.50 in
  let interest_first_bank := investment_first_bank * interest_rate_first_bank in
  let interest_second_bank := total_interest - interest_first_bank in
  let principal_second_bank := total_investment - investment_first_bank in
  interest_second_bank = principal_second_bank * r ∧
  principal_second_bank = 3300 :=
by
  use 0.065
  sorry

end interest_rate_second_bank_l468_468218


namespace rita_total_hours_l468_468151

def h_backstroke : ℕ := 50
def h_breaststroke : ℕ := 9
def h_butterfly : ℕ := 121
def h_freestyle_sidestroke_per_month : ℕ := 220
def months : ℕ := 6

def h_total : ℕ := h_backstroke + h_breaststroke + h_butterfly + (h_freestyle_sidestroke_per_month * months)

theorem rita_total_hours :
  h_total = 1500 :=
by
  sorry

end rita_total_hours_l468_468151


namespace evaluate_expression_l468_468211

-- Definitions from conditions
def a := 1296
def b := 1728

lemma a_equals_6_pow_4 : a = 6 ^ 4 := by 
  -- Use provided condition
  sorry

lemma b_equals_6_pow_3 : b = 6 ^ 3 := by
  -- Use provided condition
  sorry

lemma power_of_power (x m n : ℝ) : (x ^ m) ^ n = x ^ (m * n) := by 
  -- Use provided condition
  sorry

lemma log_of_exponent (base exp : ℝ) : log base (exp ^ 4) = 4 * log base exp := by
  -- Use provided condition
  sorry

-- The main theorem
theorem evaluate_expression : (a ^ log 6 b) ^ (1 / 4) = 216 := by
  -- Given the conditions
  rw [a_equals_6_pow_4, b_equals_6_pow_3],
  calc
    (6 ^ 4) ^ (log 6 (6^3)) ^ (1 / 4)
       = 6 ^ (4 * log 6 (6 ^ 3)) ^ (1 / 4) : by rw [power_of_power]
   ... = 6 ^ (log 6 (6 ^ 12)) ^ (1 / 4)     : by rw [log_of_exponent 6 6^3]
   ... = 6 ^ 12 ^ (1 / 4)                   : by rw [log_base.pow_self]
   ... = 6 ^ 3                              : by rw [power_of_power]
   ... = 216                                : by rfl

end evaluate_expression_l468_468211


namespace hua_luogeng_optimal_selection_method_l468_468790

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l468_468790


namespace average_speed_for_trip_l468_468287

noncomputable def average_speed (distance time : Float) : Float :=
  distance / time

def total_distance := 40.0 + 60.0 + 180.0 + 50.0 + 105.0
def total_time := 1.0 + 2.0 + 3.0 + 1.0 + 1.5

theorem average_speed_for_trip : average_speed total_distance total_time ≈ 51.18 := by
  sorry

end average_speed_for_trip_l468_468287


namespace optimal_selection_uses_golden_ratio_l468_468815

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l468_468815


namespace x4_plus_y4_l468_468043
noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

theorem x4_plus_y4 :
  (x^2 + (1 / x^2) = 7) →
  (x * y = 1) →
  (x^4 + y^4 = 47) :=
by
  intros h1 h2
  -- The proof will go here.
  sorry

end x4_plus_y4_l468_468043


namespace range_of_alpha_for_monotonic_increasing_power_function_l468_468009

theorem range_of_alpha_for_monotonic_increasing_power_function (α : ℝ) :
  (∀ x : ℝ, x > 0 → monotone_on (λ x, x ^ α) (set.Ioi 0)) ↔ α > 0 :=
begin
  sorry
end

end range_of_alpha_for_monotonic_increasing_power_function_l468_468009


namespace triangle_ratio_l468_468069

theorem triangle_ratio (ABC : Type) [triangle ABC]
  (A B C D E Q : ABC)
  (h1 : line_through C E)
  (h2 : line_through A D)
  (h3 : intersect_at Q (C, E) (A, D))
  (h4 : segment_ratio CD DB = 5/3)
  (h5 : segment_ratio AE EB = 4/3) :
  segment_ratio CQ QE = 22/7 :=
sorry

end triangle_ratio_l468_468069


namespace min_girls_in_class_l468_468919

theorem min_girls_in_class : ∃ d : ℕ, 20 - d ≤ 2 * (d + 1) ∧ d ≥ 6 := by
  sorry

end min_girls_in_class_l468_468919


namespace problem1_target_function_problem1_monotonic_increase_interval_problem2_inequality_l468_468024

noncomputable def A : ℝ := sqrt 2
noncomputable def ω : ℝ := 2
noncomputable def ϕ : ℝ := -π / 6
def πover12 : ℝ := π / 12

def f (x : ℝ) : ℝ := A * sin (ω * x + ϕ)
def g (x : ℝ) : ℝ := A * sin (2 * (x + πover12) + ϕ)

theorem problem1_target_function : 
  f x = sqrt 2 * sin (2 * x - π / 6) := sorry

theorem problem1_monotonic_increase_interval (k : ℤ) : 
  ∀ x, (k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3) 
  → ∃ y, (k * π - π / 6 ≤ y ∧ y ≤ k * π + π / 3) 
  → f y < f (y + ε) := sorry

theorem problem2_inequality (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : 
  g (x) ≥ x := sorry

end problem1_target_function_problem1_monotonic_increase_interval_problem2_inequality_l468_468024


namespace optimal_selection_method_is_golden_ratio_l468_468580

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l468_468580


namespace optimal_selection_uses_golden_ratio_l468_468653

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l468_468653


namespace optimal_selection_uses_golden_ratio_l468_468825

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l468_468825


namespace sum_pairs_mod_eq_l468_468125

open Nat

def permutation_of (l : list ℕ) (n : ℕ) := 
  ∀ x, x ∈ l ↔ x ∈ list.range (n + 1)

theorem sum_pairs_mod_eq (A B : list ℕ) (n : ℕ)
  (hA : permutation_of A n) (hB : permutation_of B n) (h_even : even n) :
  ∃ i j, i ≠ j ∧ (A.nth i).getOrElse 0 + (B.nth i).getOrElse 0 % n = (A.nth j).getOrElse 0 + (B.nth j).getOrElse 0 % n :=
by
  sorry

end sum_pairs_mod_eq_l468_468125


namespace part1_part2_part3_l468_468018

open BigOperators

noncomputable def a : ℚ := 1 / 15

def P (x : ℝ) : ℚ :=
if x = 1/5 then a * 1
else if x = 2/5 then a * 2
else if x = 3/5 then a * 3
else if x = 4/5 then a * 4
else if x = 1 then a * 5
else 0

theorem part1 : a = 1 / 15 := by
  sorry

theorem part2 : (P (3/5) + P (4/5) + P 1) = 4 / 5 := by
  sorry

theorem part3 : (P (1/5) + P (2/5) + P (3/5)) = 2 / 5 := by
  sorry

end part1_part2_part3_l468_468018


namespace optimal_selection_method_uses_golden_ratio_l468_468615

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l468_468615


namespace integral_of_2x2_cos3x_l468_468227

theorem integral_of_2x2_cos3x :
  ∫ x in (0 : ℝ)..(2 * Real.pi), (2 * x ^ 2 - 15) * Real.cos (3 * x) = (8 * Real.pi) / 9 :=
by
  sorry

end integral_of_2x2_cos3x_l468_468227


namespace find_x_val_l468_468529

theorem find_x_val (x y : ℝ) (c : ℝ) (h1 : y = 1 → x = 8) (h2 : ∀ y, x * y^3 = c) : 
  (∀ (y : ℝ), y = 2 → x = 1) :=
by
  sorry

end find_x_val_l468_468529


namespace oliver_baths_per_week_l468_468139

-- Define all the conditions given in the problem
def bucket_capacity : ℕ := 120
def num_buckets_to_fill_tub : ℕ := 14
def num_buckets_removed : ℕ := 3
def weekly_water_usage : ℕ := 9240

-- Calculate total water to fill bathtub, water removed, water used per bath, and baths per week
def total_tub_capacity : ℕ := num_buckets_to_fill_tub * bucket_capacity
def water_removed : ℕ := num_buckets_removed * bucket_capacity
def water_per_bath : ℕ := total_tub_capacity - water_removed
def baths_per_week : ℕ := weekly_water_usage / water_per_bath

theorem oliver_baths_per_week : baths_per_week = 7 := by
  sorry

end oliver_baths_per_week_l468_468139


namespace optimal_selection_method_uses_golden_ratio_l468_468841

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468841


namespace hua_luogeng_optimal_selection_method_l468_468780

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l468_468780


namespace max_sum_of_tuple_l468_468294

theorem max_sum_of_tuple (n : ℕ) (x : fin n → ℝ) 
  (h₁ : ∀ i, 0 ≤ x i) 
  (h₂ : ∑ i, x i = 1) : 
  (∑ i j in finset.off_diag_univ n, x i * x j * (x i + x j)) ≤ 1 / 4 :=
sorry

end max_sum_of_tuple_l468_468294


namespace optimal_selection_method_uses_golden_ratio_l468_468605

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l468_468605


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468772

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468772


namespace polar_equation_represents_two_circles_l468_468177

theorem polar_equation_represents_two_circles :
  ∀ (ρ θ : ℝ), ρ^2 - ρ * (2 + sin θ) + 2 * sin θ = 0 →
  (ρ = 2 ∨ ρ = sin θ) :=
by sorry

end polar_equation_represents_two_circles_l468_468177


namespace optimal_selection_uses_golden_ratio_l468_468644

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l468_468644


namespace find_monotonic_interval_find_cos_B_l468_468366

-- Definitions for given conditions
noncomputable def f (ω x : ℝ) := 2 * sqrt 3 * sin (ω * x / 2) * cos (ω * x / 2) - 2 * sin (ω * x / 2) ^ 2

axiom omega_pos : ∃ ω > 0, true
axiom period_condition : ∃ T > 0, T = 3 * π ∧ ∀ x, f ω x = f ω (x + T)

-- Part (I)
theorem find_monotonic_interval :
  ∃ k : ℤ, ∀ x, (3 * k * π - π ≤ x ∧ x ≤ 3 * k * π + π / 2) → (f (ω : ℝ) x) = sorry
  sorry

-- Part (II)
axiom sqrt3a_eq_2csinA (a c A : ℝ) : sqrt 3 * a = 2 * c * sin A
axiom f_of_A (A : ℝ) : f ω ((3 / 2) * A + π / 2) = 11 / 13

theorem find_cos_B (a b c A B C : ℝ) (h₁ : a < b) (h₂ : b < c) :
    ∃ cos_B, cos_B = (5 * sqrt 3 + 12) / 26 := sorry

end find_monotonic_interval_find_cos_B_l468_468366


namespace optimal_selection_method_uses_golden_ratio_l468_468679

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468679


namespace fourth_vertex_of_rectangle_l468_468144

theorem fourth_vertex_of_rectangle :
  ∃ (x y : ℕ), (x, y) = (5, 7) ∧ 
    ((1, 1) ∧ (5, 1) ∧ (1, 7)) ∈ 
    ({p : ℕ × ℕ | (p = (1, 1)) ∨ (p = (5, 1)) ∨ (p = (1, 7)) ∧ 
     (1 = 1) ∨ (5 = 1) ∨ (1 = 7))) :=
by
  sorry

end fourth_vertex_of_rectangle_l468_468144


namespace optimal_selection_method_uses_golden_ratio_l468_468578

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l468_468578


namespace optimal_selection_uses_golden_ratio_l468_468669

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l468_468669


namespace vanessa_weeks_to_wait_l468_468951

theorem vanessa_weeks_to_wait
  (dress_cost savings : ℕ)
  (weekly_allowance weekly_expense : ℕ)
  (h₀ : dress_cost = 80)
  (h₁ : savings = 20)
  (h₂ : weekly_allowance = 30)
  (h₃ : weekly_expense = 10) :
  let net_savings_per_week := weekly_allowance - weekly_expense,
      additional_amount_needed := dress_cost - savings in
  additional_amount_needed / net_savings_per_week = 3 :=
by
  sorry

end vanessa_weeks_to_wait_l468_468951


namespace min_girls_in_class_l468_468929

theorem min_girls_in_class (n : ℕ) (d : ℕ) (h1 : n = 20) 
  (h2 : ∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
       ∀ f : ℕ → set ℕ, (card (f i) ≠ card (f j) ∨ card (f j) ≠ card (f k) ∨ card (f i) ≠ card (f k))) : 
d ≥ 6 :=
by
  sorry

end min_girls_in_class_l468_468929


namespace ratio_PR_RQ_l468_468088

-- Define the points and distances in triangle ABC
variable (A B C N P Q R : Point)
variable (AB BC AC : Length)

-- Given conditions
axiom midpoint_N : N = midpoint A C
axiom length_AB : AB = 18
axiom length_BC : BC = 24
axiom P_on_AB : on_segment P A B
axiom Q_on_BC : on_segment Q B C
axiom intersection_R : intersects R PQ (line_through B N)
axiom BP_2BQ : distance B P = 2 * distance B Q

-- Prove the ratio PR / RQ = 1 / 3
theorem ratio_PR_RQ : distance P R / distance R Q = 1 / 3 := sorry

end ratio_PR_RQ_l468_468088


namespace day_150_of_year_N_minus_2_is_Thursday_l468_468447

-- Definitions and initial conditions
def day_of_week (d : ℕ) (start : ℕ) : string :=
  match (start + d - 1) % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | 6 => "Saturday"
  | _ => "Unknown"
  end

def N := ℕ
def start_day_of_year_N := 3 -- 3 represents Wednesday
def start_day_of_year_N_plus_1 := (start_day_of_year_N + 366) % 7
def start_day_of_year_N_minus_2 := (start_day_of_year_N - 1 - 1) % 7

theorem day_150_of_year_N_minus_2_is_Thursday :
  day_of_week 150 start_day_of_year_N_minus_2 = "Thursday" := 
  sorry

end day_150_of_year_N_minus_2_is_Thursday_l468_468447


namespace monomial_sum_l468_468066

theorem monomial_sum :
  ∀ (x y : ℝ) (a b : ℕ),
    a = 1 → b = 4 → 
    (-4 * 10^a * x^a * y^2 + 35 * x * y^(b - 2)) = -5 * x * y^2 := 
by 
  rintros x y a b ha hb
  rw [ha, hb, Nat.add_sub_cancel, pow_one]
  norm_num
  sorry 

end monomial_sum_l468_468066


namespace min_sum_of_bases_l468_468235

theorem min_sum_of_bases (a b : ℕ) (h : 3 * a + 5 = 4 * b + 2) : a + b = 13 :=
sorry

end min_sum_of_bases_l468_468235


namespace matilda_percentage_loss_l468_468500

theorem matilda_percentage_loss (initial_cost selling_price : ℕ) (h_initial : initial_cost = 300) (h_selling : selling_price = 255) :
  ((initial_cost - selling_price) * 100) / initial_cost = 15 :=
by
  rw [h_initial, h_selling]
  -- Proceed with the proof
  sorry

end matilda_percentage_loss_l468_468500


namespace optimal_selection_method_uses_golden_ratio_l468_468571

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l468_468571


namespace find_charge_federal_return_l468_468161

-- Definitions based on conditions
def charge_federal_return (F : ℝ) : ℝ := F
def charge_state_return : ℝ := 30
def charge_quarterly_return : ℝ := 80
def sold_federal_returns : ℝ := 60
def sold_state_returns : ℝ := 20
def sold_quarterly_returns : ℝ := 10
def total_revenue : ℝ := 4400

-- Lean proof statement to verify the value of F
theorem find_charge_federal_return (F : ℝ) (h : sold_federal_returns * charge_federal_return F + sold_state_returns * charge_state_return + sold_quarterly_returns * charge_quarterly_return = total_revenue) : 
  F = 50 :=
by
  sorry

end find_charge_federal_return_l468_468161


namespace hua_luogeng_optimal_selection_method_l468_468739

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l468_468739


namespace matilda_smartphone_loss_percentage_l468_468501

theorem matilda_smartphone_loss_percentage :
  ∀ (initial_cost selling_price : ℝ),
  initial_cost = 300 →
  selling_price = 255 →
  let loss := initial_cost - selling_price in
  let percentage_loss := (loss / initial_cost) * 100 in
  percentage_loss = 15 :=
by
  intros initial_cost selling_price h₁ h₂
  let loss := initial_cost - selling_price
  let percentage_loss := (loss / initial_cost) * 100
  rw [h₁, h₂]
  sorry

end matilda_smartphone_loss_percentage_l468_468501


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468776

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468776


namespace expected_value_correct_l468_468268

-- Definitions based on the problem's conditions.
def is_even (n : ℕ) : Prop := n % 2 = 0
def winnings (n : ℕ) : ℝ :=
  if n = 2 then 2
  else if n = 4 then 4
  else if n = 6 then 12
  else if n = 8 then 4
  else 0

def probability (n : ℕ) : ℝ := 1 / 8

-- Expected value calculation based on the given rules
def expected_value : ℝ :=
  (probability 1 * winnings 1) +
  (probability 2 * winnings 2) +
  (probability 3 * winnings 3) +
  (probability 4 * winnings 4) +
  (probability 5 * winnings 5) +
  (probability 6 * winnings 6) +
  (probability 7 * winnings 7) +
  (probability 8 * winnings 8)

-- Theorem stating that the expected value equals 2.75
theorem expected_value_correct : expected_value = 2.75 :=
by {
  sorry -- Proof is omitted as per instructions
}

end expected_value_correct_l468_468268


namespace factorization_correctness_l468_468265

theorem factorization_correctness :
  (∀ x : ℝ, (x + 1) * (x - 1) = x^2 - 1 → false) ∧
  (∀ x : ℝ, x^2 - 4 * x + 4 = x * (x - 4) + 4 → false) ∧
  (∀ x : ℝ, (x + 3) * (x - 4) = x^2 - x - 12 → false) ∧
  (∀ x : ℝ, x^2 - 4 = (x + 2) * (x - 2)) :=
by
  sorry

end factorization_correctness_l468_468265


namespace sin_2012_equals_neg_sin_32_l468_468990

theorem sin_2012_equals_neg_sin_32 : Real.sin (2012 * Real.pi / 180) = -Real.sin (32 * Real.pi / 180) := by
  sorry

end sin_2012_equals_neg_sin_32_l468_468990


namespace ratio_sum_eq_4_l468_468477

theorem ratio_sum_eq_4
  (x y : ℝ) (θ : ℝ) (n : ℤ)
  (hx : 0 < x) (hy : 0 < y)
  (hθ : θ ≠ (n * π / 2))
  (h1 : sin θ / x = cos θ / y)
  (h2 : cos θ ^ 4 / x ^ 4 + sin θ ^ 4 / y ^ 4 = 97 * sin (2 * θ) / (x ^ 3 * y + y ^ 3 * x)) :
  (y / x + x / y) = 4 :=
by
  sorry

end ratio_sum_eq_4_l468_468477


namespace jen_triple_flips_l468_468453

-- Definitions based on conditions
def tyler_double_flips : ℕ := 12
def flips_per_double_flip : ℕ := 2
def flips_by_tyler : ℕ := tyler_double_flips * flips_per_double_flip
def flips_ratio : ℕ := 2
def flips_per_triple_flip : ℕ := 3
def flips_by_jen : ℕ := flips_by_tyler * flips_ratio

-- Lean 4 statement
theorem jen_triple_flips : flips_by_jen / flips_per_triple_flip = 16 :=
by 
    -- Proof contents should go here. We only need the statement as per the instruction.
    sorry

end jen_triple_flips_l468_468453


namespace find_square_l468_468004

-- Define the conditions as hypotheses
theorem find_square (p : ℕ) (sq : ℕ)
  (h1 : sq + p = 75)
  (h2 : (sq + p) + p = 142) :
  sq = 8 := by
  sorry

end find_square_l468_468004


namespace optimal_selection_method_uses_golden_ratio_l468_468610

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l468_468610


namespace inverse_of_256_mod_97_l468_468349

theorem inverse_of_256_mod_97
  (h : 16⁻¹ ≡ 10 [MOD 97]) : 256⁻¹ ≡ 3 [MOD 97] :=
sorry

end inverse_of_256_mod_97_l468_468349


namespace initial_students_l468_468163

variable (n : ℝ) (W : ℝ)

theorem initial_students 
  (h1 : W = n * 15)
  (h2 : W + 11 = (n + 1) * 14.8)
  (h3 : 15 * n + 11 = 14.8 * n + 14.8)
  (h4 : 0.2 * n = 3.8) :
  n = 19 :=
sorry

end initial_students_l468_468163


namespace optimalSelectionUsesGoldenRatio_l468_468720

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l468_468720


namespace optimalSelectionUsesGoldenRatio_l468_468708

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l468_468708


namespace domain_of_f_l468_468370

noncomputable def f (x : ℝ) : ℝ := real.sqrt(x - 1) + 1 / (x - 2)

theorem domain_of_f : 
  {x : ℝ | x ≥ 1 ∧ x ≠ 2} = {x : ℝ | (1 ≤ x ∧ x < 2) ∨ (2 < x)} :=
by
  sorry

end domain_of_f_l468_468370


namespace constant_term_sara_poly_l468_468484

noncomputable def james_poly := z^4 + a * z^3 + b * z^2 + c * z + 3
noncomputable def sara_poly := z^4 + a * z^3 + d * z^2 + e * z + 3

theorem constant_term_sara_poly (a b c d e : ℝ):
  (james_poly * sara_poly = z^8 + 4 * z^7 + 5 * z^6 + 7 * z^5 + 9 * z^4 + 8 * z^3 + 6 * z^2 + 8 * z + 9) →
  (∃ k : ℝ, james_poly = z^4 + a * z^3 + b * z^2 + c * z + k ∧ sara_poly = z^4 + a * z^3 + d * z^2 + e * z + k) →
  ∃ k, k = 3 :=
by 
  sorry

end constant_term_sara_poly_l468_468484


namespace optimalSelectionUsesGoldenRatio_l468_468717

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l468_468717


namespace optimal_selection_method_use_golden_ratio_l468_468796

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l468_468796


namespace min_number_of_girls_l468_468913

theorem min_number_of_girls (total_students boys_count girls_count : ℕ) (no_three_equal: ∀ m n : ℕ, m ≠ n → boys_count m ≠ boys_count n) (boys_at_most : boys_count ≤ 2 * (girls_count + 1)) : ∃ d : ℕ, d ≥ 6 ∧ total_students = 20 ∧ boys_count = total_students - girls_count :=
by
  let d := 6
  exists d
  sorry

end min_number_of_girls_l468_468913


namespace point_locus_l468_468376

variables {A B C : Type} [MetricSpace A]

-- Definition of angle measure
noncomputable def angle (A B C : A) : ℝ := sorry

-- Definition of the condition: ∠ACB = 30 degrees
def angle_condition (A B C : A) : Prop := angle A C B = 30

-- The final proof statement in Lean
theorem point_locus (A B : A) :
  ∃ C : A, angle_condition A B C → C ∈ locus C :=
sorry

end point_locus_l468_468376


namespace count_valid_three_digit_integers_is_24_l468_468021

-- Define the set of digits available
def digits : Finset ℕ := {2, 4, 7, 9}

-- Define a function to count the number of valid three-digit integers
noncomputable def count_three_digit_integers : Nat := 
  Finset.card (Finset.filter (λ n : ℕ, 
    ∃ h₁ h₂ h₃, n = h₁ * 100 + h₂ * 10 + h₃ ∧ 
      h₁ ∈ digits ∧ h₂ ∈ digits ∧ h₃ ∈ digits ∧ 
      h₁ ≠ h₂ ∧ h₁ ≠ h₃ ∧ h₂ ≠ h₃
  ) (Finset.range 1000) )

-- The theorem stating the total number of different three-digit integers
theorem count_valid_three_digit_integers_is_24 : count_three_digit_integers = 24 :=
by
  sorry

end count_valid_three_digit_integers_is_24_l468_468021


namespace num_int_solutions_2017_l468_468046

theorem num_int_solutions_2017 :
  ∃! (sol : Finset (ℤ × ℤ)), sol.card = 4 ∧ ∀ (x, y) ∈ sol, (2 * x + y)^2 = 2017 + x^2 :=
sorry

end num_int_solutions_2017_l468_468046


namespace possible_lightest_boy_heaviest_girl_l468_468419

theorem possible_lightest_boy_heaviest_girl
    (B G : ℕ) 
    (wb : ℕ → ℕ) 
    (wg : ℕ → ℕ) 
    (N < 35) 
    (avg_weight_students : Real) 
    (avg_weight_girls : Real) 
    (avg_weight_boys : Real) : 
    (avg_weight_students = 53.5) ∧ 
    (avg_weight_girls = 47) ∧ 
    (avg_weight_boys = 60) → 
    (∃ i j : ℕ, (i < B) ∧ (j < G) ∧ (wb i < wg j)) :=
by 
  sorry

end possible_lightest_boy_heaviest_girl_l468_468419


namespace min_number_of_girls_l468_468908

theorem min_number_of_girls (total_students boys_count girls_count : ℕ) (no_three_equal: ∀ m n : ℕ, m ≠ n → boys_count m ≠ boys_count n) (boys_at_most : boys_count ≤ 2 * (girls_count + 1)) : ∃ d : ℕ, d ≥ 6 ∧ total_students = 20 ∧ boys_count = total_students - girls_count :=
by
  let d := 6
  exists d
  sorry

end min_number_of_girls_l468_468908


namespace number_above_196_is_169_l468_468212

theorem number_above_196_is_169
    (k_th_row_has_2k_minus_1_numbers : ∀ k : ℕ, k > 0 → (2 * k - 1 = List.length (row k)))
    (total_numbers_upto_row_is_k2 : ∀ k : ℕ, k > 0 → (count_upto_row k = k^2)) :
    number_directly_above 196 = 169 :=
sorry

end number_above_196_is_169_l468_468212


namespace hua_luogeng_optimal_selection_l468_468549

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l468_468549


namespace probability_commitee_at_least_one_boy_one_girl_l468_468535

theorem probability_commitee_at_least_one_boy_one_girl (total_members boys girls committee_size : ℕ)
  (h1 : total_members = 24)
  (h2 : boys = 12)
  (h3 : girls = 12)
  (h4 : committee_size = 5) :
  let total_ways := Nat.choose total_members committee_size in
  let all_boys_or_all_girls := 2 * Nat.choose boys committee_size in
  let at_least_one_boy_one_girl := total_ways - all_boys_or_all_girls in
  at_least_one_boy_one_girl * 1771 = total_ways * 1705 := 
by
  sorry

end probability_commitee_at_least_one_boy_one_girl_l468_468535


namespace max_value_ineq_l468_468128

theorem max_value_ineq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x + y)^2 / (x^2 + y^2 + xy) ≤ 4 / 3 :=
sorry

end max_value_ineq_l468_468128


namespace isosceles_triangle_l468_468445

variable (A B C D E : Type)
variable [LinearOrder A] [LinearOrder B] [LinearOrder C]

theorem isosceles_triangle
  (ABC_triangle : isTriangle ABC)
  (angle_bisector_A : isAngleBisector A B C A)
  (angle_bisector_B : isAngleBisector B A C B)
  (parallel_lines_C_to_A : areParallel (lineThrough C D) (angle_bisector_A))
  (parallel_lines_C_to_B : areParallel (lineThrough C E) (angle_bisector_B))
  (intersection_A : D == intersection (lineThrough C D) (angle_bisector_A))
  (intersection_B : E == intersection (lineThrough C E) (angle_bisector_B))
  (parallel_DE_AB : areParallel (lineThrough D E) (lineThrough A B))
: isIsoscelesTriangle ABC := sorry

end isosceles_triangle_l468_468445


namespace periodic_functions_count_l468_468316

theorem periodic_functions_count :
  ∃ (f g : ℝ → ℝ),
    (∀ x, f x = |Real.sin x|) ∧
    (∀ x (n : ℤ), n ≤ x ∧ x < n + 1 → g x = x - n) ∧
    (∃ (T_f T_g : ℝ), T_f > 0 ∧ T_g > 0 ∧ 
           (∀ x, f (x + T_f) = f x) ∧ 
           (∀ x, g (x + T_g) = g x) ∧ 
           (∀ x T, (f(x + T) + g(x + T) = f(x) + g(x) → T = 0)) ∧ 
           (∀ x T, (f(x + T) * g(x + T) = f(x) * g(x) → T = 0)))
  → 2 := 
begin
  sorry
end

end periodic_functions_count_l468_468316


namespace max_winners_in_tournament_l468_468077

open Nat

theorem max_winners_in_tournament (n : ℕ) (hn : n > 1):
  let max_winners := (n + 1) / 2
  in max_winners = n - 1 / 2 ∨ max_winners = n - 1 / 2 + 1 :=
by
  sorry

end max_winners_in_tournament_l468_468077


namespace optimal_selection_method_uses_golden_ratio_l468_468606

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l468_468606


namespace domain_of_function_l468_468369

open Set

noncomputable def domain (f : ℝ → ℝ) : Set ℝ := {x | x - 1 ≥ 0 ∧ x - 2 ≠ 0}

theorem domain_of_function :
  domain (fun x => sqrt (x - 1) + (1 / (x - 2))) = (Ico 1 2) ∪ (Ioi 2) :=
by
  sorry

end domain_of_function_l468_468369


namespace optimal_selection_golden_ratio_l468_468698

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l468_468698


namespace optimalSelectionUsesGoldenRatio_l468_468716

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l468_468716


namespace value_of_a_plus_d_l468_468221

variable (a b c d : ℝ)

theorem value_of_a_plus_d 
  (h1 : a + b = 4) 
  (h2 : b + c = 5) 
  (h3 : c + d = 3) : 
  a + d = 1 := 
by 
  sorry

end value_of_a_plus_d_l468_468221


namespace optimal_selection_method_uses_golden_ratio_l468_468674

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468674


namespace ratio_of_areas_is_one_l468_468234

theorem ratio_of_areas_is_one
    (A B C X : Type)
    (hCX_median : ∀ a b c : ℝ, c = (a + b) / 2)
    (hX_midpoint : AX = XB)
    (hAX : ∀ a b : ℝ, a = b)
    (AX_length : AX = 9)
    (BX_length : BX = 9):
  (area_triangle BCX) / (area_triangle ACX) = 1 :=
by
  sorry

end ratio_of_areas_is_one_l468_468234


namespace optimal_selection_method_is_golden_ratio_l468_468596

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l468_468596


namespace optimal_selection_method_use_golden_ratio_l468_468809

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l468_468809


namespace optimalSelectionUsesGoldenRatio_l468_468707

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l468_468707


namespace question1_question2_l468_468361

noncomputable def f (a x : ℝ) : ℝ := (a - x) * Real.exp x - 1

noncomputable def g (t m x : ℝ) : ℝ := (x - t)^2 + (Real.log x - m / t)^2

theorem question1 (a x : ℝ) : 
  let f' := λ x : ℝ, (a - Real.exp x - x * Real.exp x) in
  (∀ x, f' x = 0 ↔ x = a - 1) ∧ 
  (∀ x, x < a - 1 → 0 < f' x) ∧ 
  (∀ x, x > a - 1 → f' x < 0) ∧ 
  let f_max := (a - 1) in
  f a (a - 1) = Real.exp (a - 1) - 1 :=
  sorry

theorem question2 (t x1 x2 m : ℝ) (a : ℝ) (h_a : a = 1) :
  f 1 x1 = g t m x2 → 0 < x2 → ∃ x2, m = x2 * Real.log x2 ∧ 
  (∀ x, 0 < x ∧ x < 1 / Real.exp 1 → x * Real.log x > -1 / Real.exp 1) ∧ 
  (∀ x, x > 1 / Real.exp 1 → x * Real.log x > -1 / Real.exp 1) :=
  sorry

end question1_question2_l468_468361


namespace optimal_selection_uses_golden_ratio_l468_468651

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l468_468651


namespace proof_x_y_l468_468395

noncomputable def x_y_problem (x y : ℝ) : Prop :=
  (x^2 = 9) ∧ (|y| = 4) ∧ (x < y) → (x - y = -1 ∨ x - y = -7)

theorem proof_x_y (x y : ℝ) : x_y_problem x y :=
by
  sorry

end proof_x_y_l468_468395


namespace sum_of_solutions_l468_468115

def f (x : ℝ) : ℝ := 12 * x + 5

theorem sum_of_solutions :
  let f_inv (y : ℝ) : ℝ := (y - 5) / 12 in
  let g (x : ℝ) : ℝ := f (1 / (3 * x)) in
  ∑ x in {x : ℝ | f_inv x = g x}.to_finset, x = 65 :=
by
  sorry

end sum_of_solutions_l468_468115


namespace optimal_selection_method_use_golden_ratio_l468_468805

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l468_468805


namespace find_m_range_f_A_l468_468364

noncomputable def f (x m : ℝ) : ℝ := 
  sqrt 3 * sin x * cos x - cos x^2 + m

theorem find_m (m : ℝ) (h : f (π / 12) m = 0) : 
  m = 1 / 2 := 
sorry

theorem range_f_A (A : ℝ) (h1 : 0 < A) (h2 : A < 2 * π / 3) :
  - (1 / 2) < sin (2 * A - π / 6) ∧ sin (2 * A - π / 6) ≤ 1 :=
sorry

end find_m_range_f_A_l468_468364


namespace maximum_product_xyz_l468_468127

theorem maximum_product_xyz 
  (x y z : ℝ) 
  (h1 : 0 < x) 
  (h2 : 0 < y) 
  (h3 : 0 < z) 
  (h_sum : x + y + z = 1) 
  (h_inequality : z ≤ 3 * x) : 
  xyz_product : x * y * z ≤ 3 / 125 :=
sorry

end maximum_product_xyz_l468_468127


namespace minimize_cut_proof_l468_468277

def paper_width : ℕ := 40
def paper_height : ℕ := 30
def rect_width : ℕ := 10
def rect_height : ℕ := 5

noncomputable def minimize_cut_length : ℕ := 140 -- The given minimized total length of cuts.

theorem minimize_cut_proof : 
  ∀ (x1 y1 x2 y2 : ℕ), 
  x2 = x1 + rect_width → y2 = y1 + rect_height → 
  (minimize_cut_length = 2 * paper_height + 2 * paper_width) := 
by
  intros x1 y1 x2 y2 hx2 hy2
  -- We define the cuts;
  let cut1 := paper_height
  let cut2 := paper_height
  let cut3 := paper_width
  let cut4 := paper_width
  -- Sum them
  have total_cut_length := cut1 + cut2 + cut3 + cut4
  -- Show that this is the minimized length
  show minimize_cut_length = total_cut_length,
  sorry

end minimize_cut_proof_l468_468277


namespace spot_area_l468_468527

theorem spot_area
  (side_length : ℝ) (tether_length : ℝ)
  (side_length_eq : side_length = 1.5)
  (tether_length_eq : tether_length = 2.5) :
  let main_sector_area := π * tether_length ^ 2 * (270 / 360)
      small_sector_area := 2 * (π * (tether_length - side_length) ^ 2 * (30 / 360))
      total_area := main_sector_area + small_sector_area
  in total_area = 4.73 * π := by
{
  sorry
}

end spot_area_l468_468527


namespace parker_richie_share_ratio_l468_468141

theorem parker_richie_share_ratio (total_money : ℕ) (parker_share : ℕ) 
                                  (richie_share : ℕ) (h_total : total_money = 125)
                                  (h_parker : parker_share = 50)
                                  (h_smaller : parker_share < richie_share)
                                  (h_sum_shares : parker_share + richie_share = total_money) :
                                  parker_share / nat.gcd(parker_share, richie_share) * 2 = 2 
                                  ∧ richie_share / nat.gcd(parker_share, richie_share) * 2 = 3 := 
by
  -- Definitions and equalities derived from conditions
  have h_1 : parker_share + richie_share = total_money, from h_sum_shares,
  have h_2 : total_money = 125, from h_total,
  have h_3 : parker_share = 50, from h_parker,
  have h_4 : parker_share < richie_share, from h_smaller,

  -- Computed share for Richie based on conditions
  have h_richie_share_calculated : richie_share = total_money - parker_share,
    from calc
      richie_share = 125 - 50 : sorry, -- direct substitution since values are given

  -- Proving the ratio
  split;

  { -- For Parker's ratio
    calc
      parker_share / nat.gcd(parker_share, richie_share) * 2 = 50 / gcd(50, 75) * 2 := sorry
       ... = 1 * 2 := sorry
       ... = 2 := sorry },
  { -- For Richie's ratio
    calc
      richie_share / nat.gcd(parker_share, richie_share) * 2 = 75 / gcd(50, 75) * 2 := sorry
       ... = 1.5 * 2 := sorry
       ... = 3 := sorry }

end parker_richie_share_ratio_l468_468141


namespace optimal_selection_method_uses_golden_ratio_l468_468744

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l468_468744


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468771

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468771


namespace valid_numbers_l468_468214

noncomputable def is_valid_number (a : ℕ) : Prop :=
  ∃ b c d x y : ℕ, 
    a = b * c + d ∧
    a = 10 * x + y ∧
    x > 0 ∧ x ≤ 9 ∧ y ≤ 9 ∧
    10 * x + y = 4 * x + 4 * y

theorem valid_numbers : 
  ∃ a : ℕ, (a = 12 ∨ a = 24 ∨ a = 36 ∨ a = 48) ∧ is_valid_number a :=
by
  sorry

end valid_numbers_l468_468214


namespace time_after_850_hours_l468_468543

/--
Given the current time is 3:15 PM, prove that the time 850 hours from now 
on a 12-hour clock will be 1:15 AM.
-/
theorem time_after_850_hours (h : 850 % 12 = 10) : (15 + 850 * 60) % (12 * 60) = 75 :=
by {
  have remainder : 850 % 12 = 10, from h,
  have total_minutes_initial : 15 := 3 * 60 + 15,
  have additional_minutes : 850 * 60,
  have total_minutes_new := total_minutes_initial + additional_minutes,
  have new_time_hours := (total_minutes_new / 60) % 12,
  have new_time_minutes := total_minutes_new % 60,
  show (total_minutes_new % (12 * 60)) = 75,
  from sorry
}

end time_after_850_hours_l468_468543


namespace no_such_divisor_l468_468398

theorem no_such_divisor (n : ℕ) : 
  (n ∣ (823435 : ℕ)^15) ∧ (n^5 - n^n = 1) → false := 
by sorry

end no_such_divisor_l468_468398


namespace remaining_integers_l468_468884

def set_T : Set ℕ := {x | x ≥ 1 ∧ x ≤ 100}

def multiples_of_4 : Set ℕ := {x | x ∈ set_T ∧ x % 4 = 0}
def multiples_of_5 : Set ℕ := {x | x ∈ set_T ∧ x % 5 = 0}
def multiples_of_20 : Set ℕ := {x | x ∈ set_T ∧ x % 20 = 0}

theorem remaining_integers :
  ∃ n, n = 100 - (25 + 20 - 5) ∧ (∀ x, x ∈ set_T → ¬ (x ∈ multiples_of_4 ∨ x ∈ multiples_of_5)  → x ∈ set_T ∧ x ∉ set.subset (set.filter (λ y, (y % 4 = 0 ∨ y % 5 = 0)) set_T)) := 
  sorry

end remaining_integers_l468_468884


namespace min_number_of_girls_l468_468888

theorem min_number_of_girls (d : ℕ) (students : ℕ) (boys : ℕ → ℕ) : 
  students = 20 ∧ ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → boys i ≠ boys j ∨ boys j ≠ boys k ∨ boys i ≠ boys k → d = 6 :=
by
  sorry

end min_number_of_girls_l468_468888


namespace possible_lightest_boy_heaviest_girl_l468_468418

theorem possible_lightest_boy_heaviest_girl
    (B G : ℕ)
    (hN : B + G < 35)
    (w_boys w_girls : ℕ → ℝ)
    (avg_all : Real := 53.5)
    (avg_girls : Real := 47)
    (avg_boys : Real := 60) :
    (∑ i in Finset.range B, w_boys i) / B = avg_boys →
    (∑ i in Finset.range G, w_girls i) / G = avg_girls →
    ((∑ i in Finset.range B, w_boys i) + (∑ i in Finset.range G, w_girls i)) / (B + G) = avg_all →
    ∃ (i j : ℕ), i < B ∧ j < G ∧ w_boys i < all_weights_worst(G, w_girls) ∧ w_girls j > all_weights_best(B, w_boys) :=
sorry

/-- Helper function that returns the minimum weight of the given range of girls' weights --/
noncomputable def all_weights_worst (g: ℕ, wg: ℕ → ℝ) : ℝ :=
  Finset.min' (Finset.range g) (begin
    -- there should be at least one girl
    exact ⟨0, by simp [show 0 < g, by omega]⟩
  end)

/-- Helper function that returns the maximum weight of the given range of boys' weights --/
noncomputable def all_weights_best (b: ℕ, wb: ℕ → ℝ) : ℝ :=
  Finset.max' (Finset.range b) (begin
    -- there should be at least one boy
    exact ⟨0, by simp [show 0 < b, by omega]⟩
  end)

end possible_lightest_boy_heaviest_girl_l468_468418


namespace sum_of_consecutive_integers_l468_468881

theorem sum_of_consecutive_integers (x : ℕ) (h1 : x * (x + 1) = 930) : x + (x + 1) = 61 :=
sorry

end sum_of_consecutive_integers_l468_468881


namespace hua_luogeng_optimal_selection_method_l468_468729

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l468_468729


namespace functional_eqn_solution_l468_468306

theorem functional_eqn_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x * f x + f y) = f x ^ 2 + y) : 
  f = (λ x, x) ∨ f = (λ x, -x) := 
sorry

end functional_eqn_solution_l468_468306


namespace minimum_girls_in_class_l468_468922

theorem minimum_girls_in_class (n : ℕ) (d : ℕ) (boys : List (List ℕ)) 
  (h_students : n = 20)
  (h_boys : boys.length = n - d)
  (h_unique : ∀ i j k : ℕ, i < boys.length → j < boys.length → k < boys.length → i ≠ j → j ≠ k → i ≠ k → 
    (boys.nthLe i (by linarith)).length ≠ (boys.nthLe j (by linarith)).length ∨ 
    (boys.nthLe j (by linarith)).length ≠ (boys.nthLe k (by linarith)).length ∨ 
    (boys.nthLe k (by linarith)).length ≠ (boys.nthLe i (by linarith)).length) :
  d = 6 := 
begin
  sorry
end

end minimum_girls_in_class_l468_468922


namespace hua_luogeng_optimal_selection_method_l468_468788

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l468_468788


namespace optimal_selection_method_uses_golden_ratio_l468_468625

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l468_468625


namespace optimal_selection_method_uses_golden_ratio_l468_468618

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l468_468618


namespace optimal_selection_method_uses_golden_ratio_l468_468836

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468836


namespace correct_statements_count_l468_468869

-- Definition of the statements
def statement1 := ∀ (q : ℚ), q > 0 ∨ q < 0
def statement2 (a : ℝ) := |a| = -a → a < 0
def statement3 := ∀ (x y : ℝ), 0 = 3
def statement4 := ∀ (q : ℚ), ∃ (p : ℝ), q = p
def statement5 := 7 = 7 ∧ 10 = 10 ∧ 15 = 15

-- Define what it means for each statement to be correct
def is_correct_statement1 := statement1 = false
def is_correct_statement2 := ∀ a : ℝ, statement2 a = false
def is_correct_statement3 := statement3 = false
def is_correct_statement4 := statement4 = true
def is_correct_statement5 := statement5 = true

-- Define the problem and its correct answer
def problem := is_correct_statement1 ∧ is_correct_statement2 ∧ is_correct_statement3 ∧ is_correct_statement4 ∧ is_correct_statement5

-- Prove that the number of correct statements is 2
theorem correct_statements_count : problem → (2 = 2) :=
by
  intro h
  sorry

end correct_statements_count_l468_468869


namespace optimal_selection_method_uses_golden_ratio_l468_468616

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l468_468616


namespace three_inequalities_true_l468_468505

variables {x y a b : ℝ}
-- Declare the conditions as hypotheses
axiom h₁ : 0 < x
axiom h₂ : 0 < y
axiom h₃ : 0 < a
axiom h₄ : 0 < b
axiom hx : x^2 < a^2
axiom hy : y^2 < b^2

theorem three_inequalities_true : 
  (x^2 + y^2 < a^2 + b^2) ∧ 
  (x^2 * y^2 < a^2 * b^2) ∧ 
  (x^2 / y^2 < a^2 / b^2) :=
sorry

end three_inequalities_true_l468_468505


namespace optimal_selection_method_uses_golden_ratio_l468_468564

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l468_468564


namespace distribution_schemes_count_l468_468536

def num_distribution_schemes (volunteers pavilions : ℕ) : ℕ :=
  if volunteers = 5 ∧ pavilions = 3 then
    3 * (Nat.choose 5 2)
  else
    0

theorem distribution_schemes_count :
  num_distribution_schemes 5 3 = 30 := by
  simp [num_distribution_schemes, Nat.choose]
  sorry

end distribution_schemes_count_l468_468536


namespace min_girls_in_class_l468_468916

theorem min_girls_in_class : ∃ d : ℕ, 20 - d ≤ 2 * (d + 1) ∧ d ≥ 6 := by
  sorry

end min_girls_in_class_l468_468916


namespace notebooks_last_weeks_l468_468097

theorem notebooks_last_weeks (notebooks pages_per_notebook math_days_per_week math_pages_per_day science_days_per_week science_pages_per_day history_days_per_week history_pages_per_day : ℕ) 
  (h_notebooks : notebooks = 5) 
  (h_pages_per_notebook : pages_per_notebook = 40) 
  (h_math_days_per_week : math_days_per_week = 3) 
  (h_math_pages_per_day : math_pages_per_day = 4) 
  (h_science_days_per_week : science_days_per_week = 2) 
  (h_science_pages_per_day : science_pages_per_day = 5) 
  (h_history_days_per_week : history_days_per_week = 1) 
  (h_history_pages_per_day : history_pages_per_day = 6) : 
  ⌊ (notebooks * pages_per_notebook) / (math_pages_per_day * math_days_per_week + science_pages_per_day * science_days_per_week + history_pages_per_day * history_days_per_week) ⌋ = 7 :=
by {
  -- This proof is left as an exercise to the reader.
  sorry
}

end notebooks_last_weeks_l468_468097


namespace find_b_l468_468348

theorem find_b (b : ℤ) :
  (∃ x : ℤ, x^2 + b * x - 36 = 0 ∧ x = -9) → b = 5 :=
by
  sorry

end find_b_l468_468348


namespace optimal_selection_uses_golden_ratio_l468_468818

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l468_468818


namespace cosine_alpha_minus_2pi_l468_468322

theorem cosine_alpha_minus_2pi 
(h1 : ∃ (α : ℝ), (sin (π + α) = (3 / 5) ∧ α > 0 ∧ α < 2 * π - π / 2))
: ∃ (α : ℝ), cos (α - 2 * π) = 4 / 5 :=
by
  sorry

end cosine_alpha_minus_2pi_l468_468322


namespace cube_root_simplification_l468_468525

theorem cube_root_simplification : (∛(20^3 + 30^3 + 50^3) = 10 * ∛160) := 
by 
  sorry

end cube_root_simplification_l468_468525


namespace optimal_selection_uses_golden_ratio_l468_468649

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l468_468649


namespace right_triangle_altitudes_condition_l468_468206

noncomputable def angle_approx_38_deg_10_min : ℝ := (37.68 * Math.sin ((-1 + Math.sqrt 5) / 2))

theorem right_triangle_altitudes_condition
  (α : ℝ)
  (h_a h_b h_c c_1 m a1 : ℝ)
  (hypotenuse: ℝ)
  (cond : α = (37.68 * Math.sin ((-1 + Math.sqrt 5) / 2))) :
  (h_a = (hypotenuse * Math.sin α)) → 
  (h_b = (hypotenuse * Math.cos α)) → 
  (h_c = (hypotenuse * Math.tan α)) →
  (hypotenuse ^ 2 = h_a ^ 2 + h_b ^ 2) :=
by
  sorry

end right_triangle_altitudes_condition_l468_468206


namespace constant_in_quadratic_eq_l468_468179

theorem constant_in_quadratic_eq (C : ℝ) (x₁ x₂ : ℝ) 
  (h1 : 2 * x₁ * x₁ + 5 * x₁ - C = 0) 
  (h2 : 2 * x₂ * x₂ + 5 * x₂ - C = 0) 
  (h3 : x₁ - x₂ = 5.5) : C = 12 := 
sorry

end constant_in_quadratic_eq_l468_468179


namespace temperature_conversion_l468_468397

theorem temperature_conversion (F : ℝ) (hF : F = 86) : 
  let C := (5 / 9) * (F - 32)
  let K := C + 273.15
in C = 30 ∧ K = 303.15 :=
by
  sorry

end temperature_conversion_l468_468397


namespace angle_A_and_area_of_triangle_l468_468341

theorem angle_A_and_area_of_triangle (a b c : ℝ) (A B C : ℝ) (R : ℝ) (h1 : 2 * a * Real.cos A = c * Real.cos B + b * Real.cos C) 
(h2 : R = 2) (h3 : b^2 + c^2 = 18) :
  A = Real.pi / 3 ∧ (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 / 2 := 
by
  sorry

end angle_A_and_area_of_triangle_l468_468341


namespace optimal_selection_uses_golden_ratio_l468_468664

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l468_468664


namespace probability_of_sum_19_l468_468943

def count_card_deck := 52
def count_nine_cards := 4
def count_ten_cards := 4
def total_number_cards := count_nine_cards + count_ten_cards
def first_card_probability := total_number_cards / count_card_deck
def second_card_probability := count_nine_cards / (count_card_deck - 1)

theorem probability_of_sum_19 : first_card_probability * second_card_probability = 8 / 663 := by
  /- We know total_number_cards is 8 which is sum of 4 nines and 4 tens -/
  have total_cards : total_number_cards = 4 + 4 := by norm_num
  /- Probability calculation -/
  rw [first_card_probability, second_card_probability],
  /- Substitute probabilities in terms of rational numbers -/
  change (8 / 52) * (4 / 51) = 8 / 663,
  /- Simplify the left hand side -/
  norm_num,
  sorry

end probability_of_sum_19_l468_468943


namespace minimum_girls_in_class_l468_468925

theorem minimum_girls_in_class (n : ℕ) (d : ℕ) (boys : List (List ℕ)) 
  (h_students : n = 20)
  (h_boys : boys.length = n - d)
  (h_unique : ∀ i j k : ℕ, i < boys.length → j < boys.length → k < boys.length → i ≠ j → j ≠ k → i ≠ k → 
    (boys.nthLe i (by linarith)).length ≠ (boys.nthLe j (by linarith)).length ∨ 
    (boys.nthLe j (by linarith)).length ≠ (boys.nthLe k (by linarith)).length ∨ 
    (boys.nthLe k (by linarith)).length ≠ (boys.nthLe i (by linarith)).length) :
  d = 6 := 
begin
  sorry
end

end minimum_girls_in_class_l468_468925


namespace optimal_selection_method_uses_golden_ratio_l468_468609

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l468_468609


namespace x_minus_y_possible_values_l468_468393

theorem x_minus_y_possible_values (x y : ℝ) (hx : x^2 = 9) (hy : |y| = 4) (hxy : x < y) : x - y = -1 ∨ x - y = -7 := 
sorry

end x_minus_y_possible_values_l468_468393


namespace min_girls_in_class_l468_468930

theorem min_girls_in_class (n : ℕ) (d : ℕ) (h1 : n = 20) 
  (h2 : ∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
       ∀ f : ℕ → set ℕ, (card (f i) ≠ card (f j) ∨ card (f j) ≠ card (f k) ∨ card (f i) ≠ card (f k))) : 
d ≥ 6 :=
by
  sorry

end min_girls_in_class_l468_468930


namespace problem_equivalence_l468_468153

open Int

theorem problem_equivalence (a : ℕ → ℤ) (n : ℕ) :
  (∀ p ∈ ({7, 11, 13} : Finset ℤ), 
    (∑ i in Finset.range (n + 1), a i * (10 : ℤ) ^ i ≡ 0 [MOD p]) ↔ 
    ((∑ i in Finset.range (n - 2), a (i + 3) * (10 : ℤ) ^ i - 
      ∑ i in Finset.range 3, a i * (10 : ℤ) ^ i) ≡ 0 [MOD p])) := 
by 
  sorry

end problem_equivalence_l468_468153


namespace train_distance_after_braking_l468_468165

noncomputable def v (t : ℝ) : ℝ := 10 - t + 108 / (t + 2)

theorem train_distance_after_braking :
  abs (∫ t in 0..16, v t - (32 + 108 * Real.log 18)) < 0.1 :=
sorry

end train_distance_after_braking_l468_468165


namespace circle_radius_increase_l468_468164

variable (r n : ℝ) -- declare variables r and n as real numbers

theorem circle_radius_increase (h : 2 * π * (r + n) = 2 * (2 * π * r)) : r = n :=
by
  sorry

end circle_radius_increase_l468_468164


namespace sum_valid_x_l468_468200

theorem sum_valid_x (n : ℕ) (h_n : n = 360) :
  let x_vals := {x : ℕ | x ∣ n ∧ x ≥ 18 ∧ n / x ≥ 12} in
  ∑ x in x_vals, x = 92 :=
by
  sorry

end sum_valid_x_l468_468200


namespace optimal_selection_method_uses_golden_ratio_l468_468572

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l468_468572


namespace minimum_number_of_girls_l468_468901

theorem minimum_number_of_girls (total_students : ℕ) (d : ℕ) 
  (h_students : total_students = 20) 
  (h_unique_lists : ∀ n : ℕ, ∃! k : ℕ, k ≤ 20 - d ∧ n = 2 * k) :
  d ≥ 6 :=
sorry

end minimum_number_of_girls_l468_468901


namespace optimal_selection_golden_ratio_l468_468701

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l468_468701


namespace problem_statement_l468_468324

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem problem_statement :
  ∃ (P : ℕ), (P > 0 ∧ ¬(2 ∣ P) ∧ ¬(3 ∣ P)) →
  ∃ (M : ℕ), (∃ n : ℕ, (1 * 2 * 3 * ... * 100 = 12^n * M) ∧ is_largest_n n) →
  (2 ∣ M) ∧ (¬(3 ∣ M)) :=
begin
  sorry
end

end problem_statement_l468_468324


namespace solve_base7_addition_problem_l468_468304

def base7_addition_problem : Prop :=
  ∃ (X Y : ℕ),
    (X + 5 = 11) ∧  -- Corresponding condition for middle column gives X = 6
    (Y + 2 = 6) ∧  -- Corresponding condition for right column gives Y = 4
    (X + Y = 10)

theorem solve_base7_addition_problem : base7_addition_problem :=
by {
  use [6, 4],
  split; {norm_num}
}

end solve_base7_addition_problem_l468_468304


namespace domain_of_function_l468_468368

open Set

noncomputable def domain (f : ℝ → ℝ) : Set ℝ := {x | x - 1 ≥ 0 ∧ x - 2 ≠ 0}

theorem domain_of_function :
  domain (fun x => sqrt (x - 1) + (1 / (x - 2))) = (Ico 1 2) ∪ (Ioi 2) :=
by
  sorry

end domain_of_function_l468_468368


namespace exists_trapezoid_in_marked_vertices_l468_468138

-- Definition of a regular 1981-gon
def regular_polygon (n : Nat) : Type := sorry

-- Define the vertices marked
def marked_vertices (n : Nat) (k : Nat) : Prop := sorry

-- Trapezoid definition in the context of marked vertices
def exists_trapezoid (n k : Nat) : Prop :=
∃ T : Finset Nat, T.card = 4 ∧ (T ⊆ (Finset.range n)) ∧ (all_pairs_parallel T)

-- The main theorem statement, using provided conditions
theorem exists_trapezoid_in_marked_vertices :
  regular_polygon 1981 →
  marked_vertices 1981 64 →
  exists_trapezoid 1981 64 :=
sorry

end exists_trapezoid_in_marked_vertices_l468_468138


namespace sum_remainder_l468_468964

theorem sum_remainder (a b c : ℕ) (h1 : a % 30 = 12) (h2 : b % 30 = 9) (h3 : c % 30 = 15) :
  (a + b + c) % 30 = 6 := 
sorry

end sum_remainder_l468_468964


namespace sum_valid_x_l468_468199

theorem sum_valid_x (n : ℕ) (h_n : n = 360) :
  let x_vals := {x : ℕ | x ∣ n ∧ x ≥ 18 ∧ n / x ≥ 12} in
  ∑ x in x_vals, x = 92 :=
by
  sorry

end sum_valid_x_l468_468199


namespace cf_length_of_rectangle_l468_468437

theorem cf_length_of_rectangle (A B C D F : Point) 
  (h1 : rectangle A B C D)
  (h2 : dist A B = 30) 
  (h3 : dist B C = 15) 
  (h4 : collinear A B F) 
  (h5 : ∠ B C F = 30) :
  dist C F = 30 :=
sorry

end cf_length_of_rectangle_l468_468437


namespace triangle_CPD_equilateral_l468_468090

section geometry

variables {A B C D P Q : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space P]

/-- ABCD is a square and P is a point inside ABCD such that ∠PBA = ∠PAB = 15°. -/
variables (AB : A → B) (BC : B → C) (CD : C → D) (DA : D → A)
variables (PA : P → A) (PB : P → B)

@[anon_switch] logical_or (∀ (AB : metric_t B A), (ABCD: metric_space A)) (∀ (DA : metric B A), (DA : metric_t B A)) => 
congr ARG3 ARG1  (equilateral arg2 TRI)
in
/-- Prove that triangle CPD is an equilateral triangle -/
theorem triangle_CPD_equilateral (h1 : metric_t A (DA) (1200 → 60 + 25 = metric_space metric)):=
  sorry

end geometry

end triangle_CPD_equilateral_l468_468090


namespace sufficient_but_not_necessary_condition_l468_468374

-- Definitions for vector and magnitude
open Real

def vector_a (x : ℝ) : ℝ × ℝ := (x, 3)

def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 * v.1 + v.2 * v.2)

-- Problem statement in Lean
theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (magnitude (vector_a x) = 5) ↔ (x = 4) := by
  sorry

end sufficient_but_not_necessary_condition_l468_468374


namespace possible_lightest_boy_heaviest_girl_l468_468424

theorem possible_lightest_boy_heaviest_girl :
  ∃ (B G : ℕ) (boys_weights girls_weights : list ℕ),
    B + G < 35 ∧
    (∑ w in boys_weights, w) / B = 60 ∧
    (∑ w in girls_weights, w) / G = 47 ∧
    (∑ w in (boys_weights ++ girls_weights), w) / (B + G) = 53.5 ∧
    (∃ l, l ∈ boys_weights ∧ ∀ g, g ∈ girls_weights → l < g) :=
begin
  sorry,
end

end possible_lightest_boy_heaviest_girl_l468_468424


namespace circle_radius_l468_468183

theorem circle_radius :
  ∃ c : ℝ × ℝ, 
    c.2 = 0 ∧
    (dist c (2, 3)) = (dist c (3, 7)) ∧
    (dist c (2, 3)) = (Real.sqrt 1717) / 2 :=
by
  sorry

end circle_radius_l468_468183


namespace can_finish_books_l468_468520

-- Define reading speed, book pages, and time available
def reading_speed := 120  -- pages per hour
def book1_pages := 360    -- pages
def book2_pages := 180    -- pages
def available_time := 7   -- hours

-- Define the function to calculate time required to read a book
def time_to_read (pages : ℕ) (speed : ℕ) : ℕ := pages / speed

-- Calculate the time required to read each book
def book1_time := time_to_read book1_pages reading_speed
def book2_time := time_to_read book2_pages reading_speed

-- Calculate the total time required to read both books
def total_reading_time := book1_time + book2_time

-- Theorem: Prove that Robert can finish both books within the available time
theorem can_finish_books : total_reading_time ≤ available_time :=
by {
    unfold total_reading_time book1_time book2_time time_to_read,
    norm_num,
    sorry
}

end can_finish_books_l468_468520


namespace optimal_selection_method_uses_golden_ratio_l468_468566

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l468_468566


namespace increasing_interval_l468_468402

-- Define the function f(x) = x^2 + 2*(a - 1)*x
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2*(a - 1)*x

-- Define the condition for f(x) being increasing on [4, +∞)
def is_increasing_on_interval (a : ℝ) : Prop := 
  ∀ x y : ℝ, 4 ≤ x → x ≤ y → 
    f x a ≤ f y a

-- Define the main theorem that we need to prove
theorem increasing_interval (a : ℝ) (h : is_increasing_on_interval a) : -3 ≤ a :=
by 
  sorry -- proof is required, but omitted as per the instruction.

end increasing_interval_l468_468402


namespace optimal_selection_method_uses_golden_ratio_l468_468853

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l468_468853


namespace hua_luogeng_optimal_selection_method_l468_468794

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l468_468794


namespace optimal_selection_method_uses_golden_ratio_l468_468601

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l468_468601


namespace optimal_selection_method_uses_golden_ratio_l468_468747

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l468_468747


namespace mark_eggs_supply_l468_468489

theorem mark_eggs_supply (dozen_eggs: ℕ) (days_in_week: ℕ) : dozen_eggs = 12 → 
  days_in_week = 7 → (5 * dozen_eggs + 30) * days_in_week = 630 :=
by 
  intros h_dozen h_days;
  rw [h_dozen, h_days];
  simp;
  norm_num;
  exact rfl

end mark_eggs_supply_l468_468489


namespace number_of_meetings_l468_468981

theorem number_of_meetings (s t : ℕ) (t_eq: t = s / 60) (n甲 n乙 : ℕ)
  (h1 : n甲 = 4) (h2 : n乙 = 3) : 
  ∃ k : ℕ, k = 6 := 
begin
  sorry
end

end number_of_meetings_l468_468981


namespace value_of_y_minus_x_l468_468979

theorem value_of_y_minus_x (x y z : ℝ) 
  (h1 : x + y + z = 12) 
  (h2 : x + y = 8) 
  (h3 : y - 3 * x + z = 9) : 
  y - x = 6.5 :=
by
  -- Proof steps would go here
  sorry

end value_of_y_minus_x_l468_468979


namespace largest_angle_in_triangle_l468_468337

theorem largest_angle_in_triangle (a b : ℝ) :
  ∃ (θ : ℝ), θ = 120 ∧ ∀ (side₃ : ℝ), side₃ = sqrt (a^2 + b^2 + a * b) →
  let c := sqrt (a^2 + b^2 + a * b) in
  θ = Real.acos ((a^2 + b^2 - c^2) / (2 * a * b)) :=
by
  let c := sqrt (a^2 + b^2 + a * b)
  use 120
  rw [Real.acos_eq arccos_neg_half, Real.acos_neg_half_eq]
  exact arccos_neg_half_eq 120
  sorry

end largest_angle_in_triangle_l468_468337


namespace fraction_nonnegative_for_all_reals_l468_468157

theorem fraction_nonnegative_for_all_reals (x : ℝ) : 
  (x^2 + 2 * x + 1) / (x^2 + 4 * x + 8) ≥ 0 :=
by
  sorry

end fraction_nonnegative_for_all_reals_l468_468157


namespace x_add_y_eq_neg27_l468_468056

variables (x y : ℤ) -- Declare the variables x and y

-- Define the conditions as Lean assumptions
def condition1 : Prop := x + 1 = y - 8
def condition2 : Prop := x = 2y

-- The theorem to prove
theorem x_add_y_eq_neg27 (h1 : condition1) (h2 : condition2) : x + y = -27 :=
sorry

end x_add_y_eq_neg27_l468_468056


namespace selling_price_correct_l468_468270

variables (CP PP : ℝ)

theorem selling_price_correct (hCP : CP = 240) (hPP : PP = 0.20) : 
    CP + CP * PP = 288 := 
by 
    rw [hCP, hPP]
    norm_num
    sorry

end selling_price_correct_l468_468270


namespace find_X_l468_468190

theorem find_X (k : ℝ) (R1 R2 X1 X2 Y1 Y2 : ℝ) (h1 : R1 = k * (X1 / Y1)) (h2 : R1 = 10) (h3 : X1 = 2) (h4 : Y1 = 4) (h5 : R2 = 8) (h6 : Y2 = 5) : X2 = 2 :=
sorry

end find_X_l468_468190


namespace optimal_selection_method_uses_golden_ratio_l468_468835

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468835


namespace decrease_in_profit_when_one_loom_is_idle_l468_468977

theorem decrease_in_profit_when_one_loom_is_idle
  (total_looms : ℕ := 70)
  (sales_value : ℕ := 500000)
  (manufacturing_expenses : ℕ := 150000)
  (establishment_charges : ℕ := 75000)
  (equal_contribution : ∀ l, l ∈ (finset.range total_looms).map (λ i, i + 1) -> True) :
  (sales_value - manufacturing_expenses - establishment_charges) / total_looms 
  = 3928.57 := 
by
  -- skipping the proof
  sorry

end decrease_in_profit_when_one_loom_is_idle_l468_468977


namespace solve_sum_of_digits_l468_468086

theorem solve_sum_of_digits:
  ∀ (A B C D E : ℕ), 
  let num := 100000 + A * 10000 + B * 1000 + C * 100 + D * 10 + E in
  num * 3 = (A * 10000 + B * 1000 + C * 100 + D * 10 + E) * 10 + 1 →
  A = 4 ∧ B = 2 ∧ C = 8 ∧ D = 5 ∧ E = 7 →
  A + B + C + D + E = 26 :=
by
  intros A B C D E num hnum heq
  sorry

end solve_sum_of_digits_l468_468086


namespace simplify_expression_l468_468155

theorem simplify_expression (m : ℝ) (h₁ : m ≥ 4) (h₂ : m ≠ 8) :
  (∛(m + 4 * sqrt (m - 4)) * ∛(sqrt (m - 4) + 2) /
   (∛(m - 4 * sqrt (m - 4)) * ∛(sqrt (m - 4) - 2)) *
   (m - 4 * sqrt (m - 4)) / 2) = (m - 8) / 2 :=
by
  sorry

end simplify_expression_l468_468155


namespace optimal_selection_method_use_golden_ratio_l468_468803

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l468_468803


namespace optimal_selection_method_uses_golden_ratio_l468_468757

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l468_468757


namespace g_h_of_2_l468_468390

def g (x : ℝ) : ℝ := 3 * x^2 + 4
def h (x : ℝ) : ℝ := -2 * x^3 + 2

theorem g_h_of_2 :
  g(h(2)) = 592 := 
by
  sorry

end g_h_of_2_l468_468390


namespace median_moons_l468_468953

theorem median_moons :
  (let moons := [0, 0, 1, 3, 18, 23, 15, 2, 5] in
   let sorted_moons := List.sort moons in
   List.nth sorted_moons 4 = some 3) :=
by
  let moons := [0, 0, 1, 3, 18, 23, 15, 2, 5]
  let sorted_moons := List.sort moons
  have H : List.nth sorted_moons 4 = some 3
  sorry

end median_moons_l468_468953


namespace no_girl_can_avoid_losing_bet_l468_468939

theorem no_girl_can_avoid_losing_bet
  (G1 G2 G3 : Prop)
  (h1 : G1 ↔ ¬G2)
  (h2 : G2 ↔ ¬G3)
  (h3 : G3 ↔ ¬G1)
  : G1 ∧ G2 ∧ G3 → False := by
  sorry

end no_girl_can_avoid_losing_bet_l468_468939


namespace optimal_selection_method_uses_golden_ratio_l468_468687

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468687


namespace optimal_selection_method_uses_golden_ratio_l468_468569

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l468_468569


namespace shanghai_1994_problem_l468_468333

theorem shanghai_1994_problem
  (n : ℕ)
  (h_n : n ≥ 5)
  (a : Fin n → ℕ)
  (unique_subset_sums : ∀ (A B : Finset (Fin n)), A ≠ B → A.nonempty → B.nonempty → (A.sum (λ i, a i) ≠ B.sum (λ i, a i))) :
  (∑ i, (1 : ℚ) / a i) ≤ 2 - (1 / 2 ^ (n - 1)) :=
sorry

end shanghai_1994_problem_l468_468333


namespace minimum_number_of_girls_l468_468904

theorem minimum_number_of_girls (total_students : ℕ) (d : ℕ) 
  (h_students : total_students = 20) 
  (h_unique_lists : ∀ n : ℕ, ∃! k : ℕ, k ≤ 20 - d ∧ n = 2 * k) :
  d ≥ 6 :=
sorry

end minimum_number_of_girls_l468_468904


namespace optimal_selection_method_is_golden_ratio_l468_468582

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l468_468582


namespace optimal_selection_method_uses_golden_ratio_l468_468683

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468683


namespace minimum_girls_in_class_l468_468923

theorem minimum_girls_in_class (n : ℕ) (d : ℕ) (boys : List (List ℕ)) 
  (h_students : n = 20)
  (h_boys : boys.length = n - d)
  (h_unique : ∀ i j k : ℕ, i < boys.length → j < boys.length → k < boys.length → i ≠ j → j ≠ k → i ≠ k → 
    (boys.nthLe i (by linarith)).length ≠ (boys.nthLe j (by linarith)).length ∨ 
    (boys.nthLe j (by linarith)).length ≠ (boys.nthLe k (by linarith)).length ∨ 
    (boys.nthLe k (by linarith)).length ≠ (boys.nthLe i (by linarith)).length) :
  d = 6 := 
begin
  sorry
end

end minimum_girls_in_class_l468_468923


namespace updated_mean_l468_468182

theorem updated_mean
  (n : ℕ) (obs_mean : ℝ) (decrement : ℝ)
  (h1 : n = 50) (h2 : obs_mean = 200) (h3 : decrement = 47) :
  (obs_mean - decrement) = 153 := by
  sorry

end updated_mean_l468_468182


namespace optimal_selection_uses_golden_ratio_l468_468824

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l468_468824


namespace percentage_gain_correct_l468_468248

theorem percentage_gain_correct :
  let cost_per_bowl := 18
  let total_bowls := 250
  let selling_price_per_bowl := 22
  let sold_bowls := 215
  let broken_bowls := total_bowls - sold_bowls
  let total_cost := total_bowls * cost_per_bowl
  let total_revenue := sold_bowls * selling_price_per_bowl
  let profit := total_revenue - total_cost
  let percentage_gain := (profit / total_cost.toFloat) * 100
  percentage_gain = 5.11 := by
sorry

end percentage_gain_correct_l468_468248


namespace mark_eggs_supply_l468_468488

theorem mark_eggs_supply 
  (dozens_per_day : ℕ)
  (eggs_per_dozen : ℕ)
  (extra_eggs : ℕ)
  (days_in_week : ℕ)
  (H1 : dozens_per_day = 5)
  (H2 : eggs_per_dozen = 12)
  (H3 : extra_eggs = 30)
  (H4 : days_in_week = 7) :
  (dozens_per_day * eggs_per_dozen + extra_eggs) * days_in_week = 630 :=
by
  sorry

end mark_eggs_supply_l468_468488


namespace range_of_A_area_of_triangle_l468_468087

variables {A B C : Type*} [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
variables {a b c : ℝ} -- sides opposite to angles A, B, C respectively in triangle ABC

-- Conditions
def triangle_condition (a b c : ℝ) : Prop :=
  a^2 + a * c = b^2

def angle_range (A : ℝ) : Prop :=
  0 < A ∧ A < (π / 3)

-- Part (I): Prove the range of A
theorem range_of_A (a b c A : ℝ) (h1 : triangle_condition a b c) : angle_range A :=
sorry

-- Part (II): Compute the area given specific values
def angle_given : ℝ := π / 6
def side_a_given : ℝ := 2

def compute_area (a b c A : ℝ) : ℝ :=
  (1 / 2) * b * c * (sin A)

theorem area_of_triangle (b c : ℝ) 
  (h1 : triangle_condition side_a_given b c) 
  (h2 : compute_area side_a_given b c angle_given = 2 * sqrt 3) : 
  compute_area side_a_given b c angle_given = 2 * sqrt 3 :=
sorry

end range_of_A_area_of_triangle_l468_468087


namespace possible_lightest_boy_heaviest_girl_l468_468428

theorem possible_lightest_boy_heaviest_girl :
  ∃ (B G : ℕ) (boys_weights girls_weights : list ℕ),
    B + G < 35 ∧
    (∑ w in boys_weights, w) / B = 60 ∧
    (∑ w in girls_weights, w) / G = 47 ∧
    (∑ w in (boys_weights ++ girls_weights), w) / (B + G) = 53.5 ∧
    (∃ l, l ∈ boys_weights ∧ ∀ g, g ∈ girls_weights → l < g) :=
begin
  sorry,
end

end possible_lightest_boy_heaviest_girl_l468_468428


namespace optimal_selection_uses_golden_ratio_l468_468667

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l468_468667


namespace tax_calculation_l468_468457

theorem tax_calculation 
  (total_earnings : ℕ) 
  (deductions : ℕ) 
  (tax_paid : ℕ) 
  (tax_rate_10 : ℚ) 
  (tax_rate_20 : ℚ) 
  (taxable_income : ℕ)
  (X : ℕ)
  (h_total_earnings : total_earnings = 100000)
  (h_deductions : deductions = 30000)
  (h_tax_paid : tax_paid = 12000)
  (h_tax_rate_10 : tax_rate_10 = 10 / 100)
  (h_tax_rate_20 : tax_rate_20 = 20 / 100)
  (h_taxable_income : taxable_income = total_earnings - deductions)
  (h_tax_equation : tax_paid = (tax_rate_10 * X) + (tax_rate_20 * (taxable_income - X))) :
  X = 20000 := 
sorry

end tax_calculation_l468_468457


namespace optimal_selection_method_uses_golden_ratio_l468_468857

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l468_468857


namespace optimal_selection_uses_golden_ratio_l468_468645

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l468_468645


namespace sum_of_valid_x_values_l468_468198

theorem sum_of_valid_x_values :
  let total_students := 360
  let min_rows := 12
  let min_students_per_row := 18
  let possible_x_values := {x | x >= min_students_per_row ∧ total_students / x >= min_rows ∧ total_students % x = 0}
  sum possible_x_values = 92 :=
by
  let total_students := 360
  let min_rows := 12
  let min_students_per_row := 18
  let possible_x_values := {x | x >= min_students_per_row ∧ total_students / x >= min_rows ∧ total_students % x = 0}
  have x_vals : List ℕ := possible_x_values.toList
  exact List.sum x_vals = 92

end sum_of_valid_x_values_l468_468198


namespace smaller_greater_than_two_thirds_l468_468404

noncomputable def prob_smaller_greater : ℝ :=
  let a := 4 in -- length of interval [2/3, 2]
  let b := 4 in -- length of interval [2/3, 2]
  (a / (3:ℝ)) * (b / (3:ℝ)) / (2 * 2)

theorem smaller_greater_than_two_thirds :
  prob_smaller_greater = 4 / 9 :=
by
  sorry

end smaller_greater_than_two_thirds_l468_468404


namespace cost_per_pouch_is_20_l468_468450

theorem cost_per_pouch_is_20 :
  let boxes := 10
  let pouches_per_box := 6
  let dollars := 12
  let cents_per_dollar := 100
  let total_pouches := boxes * pouches_per_box
  let total_cents := dollars * cents_per_dollar
  let cost_per_pouch := total_cents / total_pouches
  cost_per_pouch = 20 :=
by
  sorry

end cost_per_pouch_is_20_l468_468450


namespace prove_a2_b2_c2_zero_l468_468105

theorem prove_a2_b2_c2_zero (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) (h5 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) : a^2 + b^2 + c^2 = 0 := 
by 
  sorry

end prove_a2_b2_c2_zero_l468_468105


namespace optimal_selection_method_uses_golden_ratio_l468_468611

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l468_468611


namespace trig_identity_solution_l468_468000

theorem trig_identity_solution (A ω φ b : ℝ) (h1 : A > 0) (h2 : 0 < φ ∧ φ < π) :
  (∀ x, 2 * cos x ^ 2 + sin (2 * x) = A * sin (ω * x + φ) + b) →
  (A = sqrt 2 ∧ φ = (π / 4) ∧ b = 1) :=
by
  intro h
  sorry

end trig_identity_solution_l468_468000


namespace proof_of_intersection_l468_468033

noncomputable def line_through_P_and_ellipse (m n : ℝ) : Nat :=
if 0 < m^2 + n^2 ∧ m^2 + n^2 < 3 then 2 else sorry

theorem proof_of_intersection (m n : ℝ) (h : 0 < m^2 + n^2 ∧ m^2 + n^2 < 3) :
  line_through_P_and_ellipse m n = 2 :=
by
  simp [line_through_P_and_ellipse, h]
  sorry

end proof_of_intersection_l468_468033


namespace work_complete_in_15_days_l468_468976

theorem work_complete_in_15_days :
  let A_rate := (1 : ℚ) / 20
  let B_rate := (1 : ℚ) / 30
  let C_rate := (1 : ℚ) / 10
  let all_together_rate := A_rate + B_rate + C_rate
  let work_2_days := 2 * all_together_rate
  let B_C_rate := B_rate + C_rate
  let work_next_2_days := 2 * B_C_rate
  let total_work_4_days := work_2_days + work_next_2_days
  let remaining_work := 1 - total_work_4_days
  let B_time := remaining_work / B_rate

  2 + 2 + B_time = 15 :=
by
  sorry

end work_complete_in_15_days_l468_468976


namespace optimal_selection_method_uses_golden_ratio_l468_468858

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l468_468858


namespace x_product_le_one_l468_468480

theorem x_product_le_one (a : ℕ → ℝ) (h1 : ∀ i, 1 ≤ i → i ≤ 100 → 0 < a i)
  (h2 : ∀ i, 1 ≤ i → i ≤ 50 → a i ≥ a (101 - i)) :
  let x := λ k : ℕ, (k * a (k + 1)) / ∑ i in finset.range k, a (i + 1)
  in (∏ k in finset.range 99, x (k + 1)) ^ k ≤ 1 := by
  sorry

end x_product_le_one_l468_468480


namespace hua_luogeng_optimal_selection_method_l468_468733

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l468_468733


namespace optimal_selection_method_use_golden_ratio_l468_468801

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l468_468801


namespace AC_is_diameter_and_properties_l468_468983

inductive Point : Type
| A : Point
| B : Point
| C : Point
| D : Point
| P : Point
| Q : Point
| K : Point
| T : Point

noncomputable def inscribed_circle_radius : Real := 6

constants 
  (circle : Set Point)
  (on_circle : Point → Prop)
  (is_inscribed : ∀ p : Point, on_circle p)
  (similar_triangles_ADP_QAB : True) -- Given the similarity of triangles ADP and QAB
  (CK : Real)
  (KT : Real)
  (TA : Real)

axiom segment_proportions : CK = 2 * KT ∧ KT = TA / 3

noncomputable def AC : Real := 12
noncomputable def angle_DAC : Real := 45
noncomputable def area_ABCD : Real := 68

theorem AC_is_diameter_and_properties 
  (H1 : ∀ p : Point, on_circle p)
  (H2 : similar_triangles_ADP_QAB)
  (H3 : CK = 4 * (KT / 2))
  (H4 : CK + KT + TA = AC)
  : AC = 12 ∧ angle_DAC = 45 ∧ area_ABCD = 68 :=
by
  sorry

end AC_is_diameter_and_properties_l468_468983


namespace most_likely_units_digit_is_0_l468_468297

-- Define the range of integers on the slips of paper
def slips : Finset ℕ := {n | 1 ≤ n ∧ n ≤ 10}

-- Define the random draws of integers by Jack, Jill, and Jane 
def draw : Finset (ℕ × ℕ × ℕ) :=
  { (J1, J2, J3) | J1 ∈ slips ∧ J2 ∈ slips ∧ J3 ∈ slips }

-- Define the sum of the integers picked
def sum_draw (J1 J2 J3 : ℕ) : ℕ := J1 + J2 + J3

-- Define the units digit of the sum
def units_digit (n : ℕ) : ℕ := n % 10

-- Define the proof problem
theorem most_likely_units_digit_is_0 :
  ∃ (max_digit : ℕ), (∀ n, n ∈ draw → units_digit (sum_draw n.1 n.2.1 n.2.2) = 0) :=
sorry

end most_likely_units_digit_is_0_l468_468297


namespace optimal_selection_uses_golden_ratio_l468_468823

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l468_468823


namespace parabola_focus_l468_468885

theorem parabola_focus (x y : ℝ) (a : ℝ) (h1 : x + y - 2 ≤ 0)
  (h2 : x - 2y - 2 ≤ 0) (h3 : 2 * x - y + 2 ≥ 0) (h4 : ∃ z, ∀ a > 0, (z = y - a * x) ∧ ∃! p, y = a * x + z) :
  (0, 1 / 8) = (0, 1 / (4 * (1 / 2))) :=
by sorry

end parabola_focus_l468_468885


namespace hua_luogeng_optimal_selection_l468_468548

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l468_468548


namespace prob_compl_A_inter_B_l468_468007

open ProbabilityTheory

namespace ProbabilityExample

variable {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)

-- Define events A and B
variable (A B : Set Ω)

-- Given conditions
def Independence (A B : Set Ω) : Prop := P (A ∩ B) = P A * P B

axiom prob_A : P A = 0.4
axiom prob_B : P B = 0.7
axiom independent_A_B : Independence A B

-- The proof goal
theorem prob_compl_A_inter_B : P (Aᶜ ∩ B) = 0.42 :=
by
  sorry

end ProbabilityExample

end prob_compl_A_inter_B_l468_468007


namespace total_distance_between_first_eighth_tree_l468_468298

theorem total_distance_between_first_eighth_tree : 
  ∀ (n : ℕ), (8 = n + 1) → ∑ i in (finset.range (n)) (λ k, 5 * (k + 1)) = 140 :=
by 
  intros n hn,
  sorry

end total_distance_between_first_eighth_tree_l468_468298


namespace triangle_dissection_l468_468261

/-
Given:
- ABC is an arbitrary triangle.
- M is the midpoint of BC.
We need to prove that exactly 2 pieces are needed to dissect triangle AMB and reassemble it into triangle AMC.
-/
theorem triangle_dissection (A B C M : Point) (hM : midpoint B C M) :
  ∃ n : ℕ, n = 2 ∧ can_reassemble (dissect_triangle A M B n) (triangle A M C) :=
sorry

end triangle_dissection_l468_468261


namespace minimum_number_of_girls_l468_468900

theorem minimum_number_of_girls (students boys girls : ℕ) 
(h1 : students = 20) (h2 : boys = students - girls)
(h3 : ∀ d, 0 ≤ d → 2 * (d + 1) ≥ boys) 
: 6 ≤ girls :=
by 
  have h4 : 20 - girls = boys, from h2.symm
  have h5 : 2 * (girls + 1) ≥ boys, from h3 girls (nat.zero_le girls)
  linarith

#check minimum_number_of_girls

end minimum_number_of_girls_l468_468900


namespace optimal_selection_method_uses_golden_ratio_l468_468859

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l468_468859


namespace trivia_game_points_per_question_l468_468972

theorem trivia_game_points_per_question (correct_first_half correct_second_half total_score points_per_question : ℕ) 
  (h1 : correct_first_half = 5) 
  (h2 : correct_second_half = 5) 
  (h3 : total_score = 50) 
  (h4 : correct_first_half + correct_second_half = 10) : 
  points_per_question = 5 :=
by 
  sorry

end trivia_game_points_per_question_l468_468972


namespace no_three_primes_arith_prog_lt_5_no_k_primes_arith_prog_leq_k1_l468_468992

-- First part (i)
theorem no_three_primes_arith_prog_lt_5 (p1 p2 p3 : ℕ) 
  (hp1 : Nat.Prime p1) (hp2 : Nat.Prime p2) (hp3 : Nat.Prime p3)
  (h1 : p1 > 3) (h2 : p2 > 3) (h3 : p3 > 3)
  (h_common_diff : p2 = p1 + (p2 - p1)) (h_common_diff2 : p3 = p1 + 2 * (p2 - p1))
  (h_diff_lt_5 : (p2 - p1) < 5) : False :=
by sorry

-- Second part (ii)
theorem no_k_primes_arith_prog_leq_k1 (k : ℕ) 
  (hk : k > 3) 
  (p : Fin k → ℕ) 
  (hp : ∀ i, Nat.Prime (p i)) 
  (hp_gt_k : ∀ i, p i > k) 
  (d : ℕ) 
  (h_d_le_k1 : d ≤ k+1) 
  (h_arith_prog : ∀ i, p (i : Fin k) = p 0 + i.1 * d) : False :=
by sorry

end no_three_primes_arith_prog_lt_5_no_k_primes_arith_prog_leq_k1_l468_468992


namespace slope_product_constant_l468_468093

noncomputable def parabola_equation (p : ℝ) (hp : p > 0) : Prop :=
  ∀ x y : ℝ, (x ^ 2 = 2 * y ↔ x ^ 2 = 2 * p * y)

theorem slope_product_constant :
  ∀ (x1 y1 x2 y2 k1 k2 : ℝ) (P A B : ℝ × ℝ),
  P = (2, 2) →
  A = (x1, y1) →
  B = (x2, y2) →
  (∀ k: ℝ, y1 = k * (x1 + 2) + 4 ∧ y2 = k * (x2 + 2) + 4) →
  k1 = (y1 - 2) / (x1 - 2) →
  k2 = (y2 - 2) / (x2 - 2) →
  (x1 + x2 = 2 * k) →
  (x1 * x2 = -4 * k - 8) →
  k1 * k2 = -1 := 
  sorry

end slope_product_constant_l468_468093


namespace pictures_remaining_l468_468968

-- Define the initial number of pictures taken at the zoo and museum
def zoo_pictures : Nat := 50
def museum_pictures : Nat := 8
-- Define the number of pictures deleted
def deleted_pictures : Nat := 38

-- Define the total number of pictures taken initially and remaining after deletion
def total_pictures : Nat := zoo_pictures + museum_pictures
def remaining_pictures : Nat := total_pictures - deleted_pictures

theorem pictures_remaining : remaining_pictures = 20 := 
by 
  -- This theorem states that, given the conditions, the remaining pictures count must be 20
  sorry

end pictures_remaining_l468_468968


namespace f_2015_2016_l468_468005

noncomputable def f : ℤ → ℤ := sorry

theorem f_2015_2016 (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x, f (x + 2) = -f x) (h3 : f 1 = 2) :
  f 2015 + f 2016 = -2 :=
sorry

end f_2015_2016_l468_468005


namespace optimal_selection_method_uses_golden_ratio_l468_468600

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l468_468600


namespace hua_luogeng_optimal_selection_method_l468_468728

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l468_468728


namespace checkerboard_squares_with_seven_black_squares_l468_468236

def is_checkerboard (board : ℕ → ℕ → Prop) : Prop :=
  ∀ x y, board x y = ((x + y) % 2 = 0)

def contains_at_least_k_black_squares (board : ℕ → ℕ → Prop) (n k : ℕ) := 
  (∃ (count : ℕ), count >= k ∧ ∀ (i j : ℕ), i < n ∧ j < n → (board i j) = ((i + j) % 2 = 0))

noncomputable def count_distinct_squares_with_at_least_k_black_squares (n k : ℕ) := 
  ∑ (i j: ℕ) (hi : i + n ≤ 10) (hj : j + n ≤ 10), 
    if contains_at_least_k_black_squares (λ x y => ((x + y) % 2 = 0)) n k then 1 else 0

theorem checkerboard_squares_with_seven_black_squares (B : ℕ → ℕ → Prop) : 
  is_checkerboard B → count_distinct_squares_with_at_least_k_black_squares 10 7 = 116 :=
begin
  intro h,
  sorry
end

end checkerboard_squares_with_seven_black_squares_l468_468236


namespace optimal_selection_method_use_golden_ratio_l468_468812

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l468_468812


namespace number_of_boxes_sold_on_saturday_l468_468522

theorem number_of_boxes_sold_on_saturday (S : ℝ) 
  (h : S + 1.5 * S + 1.95 * S + 2.34 * S + 2.574 * S = 720) : 
  S = 77 := 
sorry

end number_of_boxes_sold_on_saturday_l468_468522


namespace slope_of_line_l468_468081

theorem slope_of_line (x y : ℝ) (h1 : (x * y = 160)) (h2 : (y = 160 / x)) :
  (k : ℝ) (h3 : ∀ P : ℝ × ℝ, (P = (x, 8) ∨ P = (20, y) ∨ P = (0, 0)) → (P.1 ≠ 0) → (k = 8 / x ∨ k = y / 20)) :
  k = 0.8 := by
sorry

end slope_of_line_l468_468081


namespace optimal_selection_method_uses_golden_ratio_l468_468846

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468846


namespace optimal_selection_method_uses_golden_ratio_l468_468852

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l468_468852


namespace trivia_game_points_per_question_l468_468971

theorem trivia_game_points_per_question (correct_first_half correct_second_half total_score points_per_question : ℕ) 
  (h1 : correct_first_half = 5) 
  (h2 : correct_second_half = 5) 
  (h3 : total_score = 50) 
  (h4 : correct_first_half + correct_second_half = 10) : 
  points_per_question = 5 :=
by 
  sorry

end trivia_game_points_per_question_l468_468971


namespace initial_average_score_l468_468539

theorem initial_average_score (A : ℝ) :
  (∃ (A : ℝ), (16 * A = 15 * 64 + 24)) → A = 61.5 := 
by 
  sorry 

end initial_average_score_l468_468539


namespace estimated_time_to_fill_sink_l468_468224

-- Definitions for the conditions
def Tap1_time := 287
def Tap2_time := 283
def Tap3_time := 325

-- Prove that the estimated time to fill the sink by combining the rates of all three taps is approximately 100 seconds
theorem estimated_time_to_fill_sink : 
  (1 / Tap1_time + 1 / Tap2_time + 1 / Tap3_time) ≈ (1 / 100) :=
by
  sorry

end estimated_time_to_fill_sink_l468_468224


namespace integral_x_squared_eq_3_l468_468542

theorem integral_x_squared_eq_3
  (a : ℝ) (h : a = 1)
  (h_coef : ∃ (a > 0), (∃ k, (ax- \frac { \sqrt { 3}}{ 6}) ^ k= - ∑^{2}{a(x^{2})})) :
  ∫ x in -2..a, x^2 = 3 := 
by {
  rw h, 
  sorry 
}

end integral_x_squared_eq_3_l468_468542


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468767

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468767


namespace solve_sequence_solve_T_n_l468_468188

def geom_seq_pos (a : ℕ → ℝ) : Prop :=
  ∀ n, 0 < a n

theorem solve_sequence (a : ℕ → ℝ) (h1 : geom_seq_pos a) (h2 : 2 * a 1 + 3 * a 2 = 1) (h3 : (a 3)^2 = 9 * a 2 * a 6) :
  (∀ n, a n = 1 / 3 ^ n) :=
by
  sorry

def b_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), Real.logBase 3 (a i)

def T_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), 1 / b_n a (i+1)

theorem solve_T_n (a : ℕ → ℝ) (h1 : geom_seq_pos a) (h2 : 2 * a 1 + 3 * a 2 = 1) (h3 : (a 3)^2 = 9 * a 2 * a 6) :
  (∀ n : ℕ, T_n a n = -2 * (n : ℝ) / ((n + 1) : ℝ)) :=
by
  sorry

end solve_sequence_solve_T_n_l468_468188


namespace productivity_increase_is_233_33_percent_l468_468452

noncomputable def productivity_increase :
  Real :=
  let B := 1 -- represents the base number of bears made per week
  let H := 1 -- represents the base number of hours worked per week
  let P := B / H -- base productivity in bears per hour

  let B1 := 1.80 * B -- bears per week with first assistant
  let H1 := 0.90 * H -- hours per week with first assistant
  let P1 := B1 / H1 -- productivity with first assistant

  let B2 := 1.60 * B -- bears per week with second assistant
  let H2 := 0.80 * H -- hours per week with second assistant
  let P2 := B2 / H2 -- productivity with second assistant

  let B_both := B1 + B2 - B -- total bears with both assistants
  let H_both := H1 * H2 / H -- total hours with both assistants
  let P_both := B_both / H_both -- productivity with both assistants

  (P_both / P - 1) * 100

theorem productivity_increase_is_233_33_percent :
  productivity_increase = 233.33 :=
by
  sorry

end productivity_increase_is_233_33_percent_l468_468452


namespace domain_of_f_l468_468371

noncomputable def f (x : ℝ) : ℝ := real.sqrt(x - 1) + 1 / (x - 2)

theorem domain_of_f : 
  {x : ℝ | x ≥ 1 ∧ x ≠ 2} = {x : ℝ | (1 ≤ x ∧ x < 2) ∨ (2 < x)} :=
by
  sorry

end domain_of_f_l468_468371


namespace prove_a2_b2_c2_zero_l468_468108

theorem prove_a2_b2_c2_zero (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) (h5 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) : a^2 + b^2 + c^2 = 0 := 
by 
  sorry

end prove_a2_b2_c2_zero_l468_468108


namespace optimal_selection_method_uses_golden_ratio_l468_468684

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468684


namespace sum_of_roots_l468_468314

theorem sum_of_roots : 
  let p1 := 3 * x^4 + 2 * x^3 - 9 * x^2 + 6 * x - 18
  let p2 := 4 * x^3 - 20 * x^2 + 4 * x + 28
  (-Polynomial.sumRoots p1) + (Polynomial.sumRoots p2) = 13 / 3 :=
by
  sorry

end sum_of_roots_l468_468314


namespace annie_initial_money_l468_468276

def cost_of_hamburgers (n : Nat) : Nat := n * 4
def cost_of_milkshakes (m : Nat) : Nat := m * 5
def total_cost (n m : Nat) : Nat := cost_of_hamburgers n + cost_of_milkshakes m
def initial_money (n m left : Nat) : Nat := total_cost n m + left

theorem annie_initial_money : initial_money 8 6 70 = 132 := by
  sorry

end annie_initial_money_l468_468276


namespace optimal_selection_golden_ratio_l468_468696

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l468_468696


namespace angle_A_value_sin_BC_value_l468_468405

open Real

noncomputable def triangleABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ 
  A + B + C = π 

theorem angle_A_value (A B C : ℝ) (h : triangleABC a b c A B C) (h1 : cos 2 * A - 3 * cos (B + C) = 1) : 
  A = π / 3 :=
sorry

theorem sin_BC_value (A B C S b c : ℝ) (h : triangleABC a b c A B C)
  (hA : A = π / 3) (hS : S = 5 * sqrt 3) (hb : b = 5) : 
  sin B * sin C = 5 / 7 :=
sorry

end angle_A_value_sin_BC_value_l468_468405


namespace average_number_of_rabbits_is_59_l468_468430

noncomputable def weighted_average_number_of_rabbits : ℕ :=
let N1 := (10 * 12) / 2 in
let N2 := (10 * 15) / 3 in
let N3 := (10 * 18) / 4 in
(N1 * 12 + N2 * 15 + N3 * 18) / (12 + 15 + 18)

theorem average_number_of_rabbits_is_59 :
  weighted_average_number_of_rabbits = 59 :=
by
  sorry

end average_number_of_rabbits_is_59_l468_468430


namespace rhombus_area_40_30_l468_468166

-- Definitions and conditions
def is_rhombus (d1 d2 : ℝ) : Prop := 
  ∃ (a b : ℝ), d1 = 2 * a ∧ d2 = 2 * b

def diagonals (d1 d2 : ℝ) : Prop := 
  d1 = 40 ∧ d2 = 30

def area_of_rhombus (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / 2

-- Proof statement
theorem rhombus_area_40_30 : 
  ∀ d1 d2 : ℝ, diagonals d1 d2 → is_rhombus d1 d2 → area_of_rhombus d1 d2 = 600 :=
by
  intros d1 d2 h_diagonals h_rhombus
  -- Proof steps would go here
  sorry

end rhombus_area_40_30_l468_468166


namespace optimal_selection_method_use_golden_ratio_l468_468810

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l468_468810


namespace hua_luogeng_optimal_selection_method_l468_468783

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l468_468783


namespace determine_suit_cost_l468_468982

def cost_of_suit (J B V : ℕ) : Prop :=
  (J + B + V = 150)

theorem determine_suit_cost
  (J B V : ℕ)
  (h1 : J = B + V)
  (h2 : J + 2 * B = 175)
  (h3 : B + 2 * V = 100) :
  cost_of_suit J B V :=
by
  sorry

end determine_suit_cost_l468_468982


namespace optimalSelectionUsesGoldenRatio_l468_468721

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l468_468721


namespace min_number_of_girls_l468_468891

theorem min_number_of_girls (d : ℕ) (students : ℕ) (boys : ℕ → ℕ) : 
  students = 20 ∧ ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → boys i ≠ boys j ∨ boys j ≠ boys k ∨ boys i ≠ boys k → d = 6 :=
by
  sorry

end min_number_of_girls_l468_468891


namespace optimal_selection_method_use_golden_ratio_l468_468800

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l468_468800


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468774

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468774


namespace optimal_selection_uses_golden_ratio_l468_468658

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l468_468658


namespace count_irrational_numbers_l468_468267

-- Definitions of the given numbers
def num1 := -0.3333333333333   -- repeating decimal approximation
def num2 := Real.sqrt 4
def num3 := Real.sqrt 5
def num4 := 3 * Real.pi
def num5 := 3.1415
def num6 := 2.010101010101 -- repeating decimal with '01' sequence

-- Proposition to prove there are exactly two irrational numbers in the given set
theorem count_irrational_numbers :
  (∃ nums: Set ℝ, nums = {num1, num2, num3, num4, num5, num6} ∧
   ∑ n in nums, if ¬∃ (r₁ r₂ : ℤ), r₂ ≠ 0 ∧ n = r₁ / r₂ then 1 else 0 = 2) :=
by
  sorry

end count_irrational_numbers_l468_468267


namespace measure_one_kg_of_cereal_l468_468886

theorem measure_one_kg_of_cereal :
  ∃ (x y : ℕ), (let weight := 3 in
                let initial_grain := 11 in
                let side_a := weight + x in
                let side_b := initial_grain - x in
                let after_first_weighing := if side_a = side_b then (side_a, side_b) else (0, 0)
                in
                let side_c := weight in
                let remaining_grain := after_first_weighing.2 - y in
                y = 1) :=
by 
  sorry

end measure_one_kg_of_cereal_l468_468886


namespace number_of_correct_statements_l468_468870

def condition1 : Prop := ∀ q : ℚ, q > 0 ∨ q < 0  -- This omits zero, hence incorrect.
def condition2 : Prop := ∀ a : ℝ, |a| = -a → a < 0  -- This doesn't consider the case of a = 0.
def poly := 2 * x^3 - 3 * x * y + 3 * y
def condition3 : Prop := ∃ (c : ℝ), polynomial.coeff poly 2 = c  -- There is no x^2 term here.
def condition4 : Prop := ∀ q : ℚ, ∃ r : ℝ, r = q  -- All rational numbers can be represented on the number line.
def pentagonal_prism := (7, 10, 15)  -- number of faces, vertices, edges
def condition5 : Prop := pentagonal_prism = (7, 10, 15)  -- This is correct by definition.

theorem number_of_correct_statements : (if condition4 then 1 else 0) + (if condition5 then 1 else 0) = 2 := by sorry

end number_of_correct_statements_l468_468870


namespace sin_cos_value_l468_468382

theorem sin_cos_value (x : ℝ) (h : 2 * sin x = 5 * cos x) : sin x * cos x = 10 / 29 := by
  sorry

end sin_cos_value_l468_468382


namespace hypotenuse_length_l468_468254

-- Define the conditions
def right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- State the theorem using the conditions and correct answer
theorem hypotenuse_length : right_triangle 20 21 29 :=
by
  -- To be filled in by proof steps
  sorry

end hypotenuse_length_l468_468254


namespace optimal_selection_method_use_golden_ratio_l468_468798

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l468_468798


namespace ratio_of_areas_correct_l468_468519

noncomputable def ratio_of_areas (P1 P2 : ℝ) : ℝ :=
  let side_I := P1 / 4
  let side_II := P2 / 4
  let P3 := (P1 + P2) / 2
  let side_III := P3 / 3
  let area_I := side_I ^ 2
  let area_III := (real.sqrt 3) / 4 * side_III ^ 2
  area_I / area_III

theorem ratio_of_areas_correct :
  ratio_of_areas 16 36 = 144 / (169 * real.sqrt 3) :=
by
  sorry

end ratio_of_areas_correct_l468_468519


namespace expression_for_B_A_greater_than_B_l468_468044

-- Define the polynomials A and B
def A (x : ℝ) := 3 * x^2 - 2 * x + 1
def B (x : ℝ) := 2 * x^2 - x - 3

-- Prove that the given expression for B validates the equation A + B = 5x^2 - 4x - 2.
theorem expression_for_B (x : ℝ) : A x + 2 * x^2 - x - 3 = 5 * x^2 - 4 * x - 2 :=
by {
  sorry
}

-- Prove that A is always greater than B for all values of x.
theorem A_greater_than_B (x : ℝ) : A x > B x :=
by {
  sorry
}

end expression_for_B_A_greater_than_B_l468_468044


namespace inverse_variation_example_l468_468532

theorem inverse_variation_example (x y : ℝ) (k : ℝ) 
  (h1 : ∀ x y, x * y^3 = k) (h2 : 8 * (1:ℝ)^3 = k) : 
  (∃ (x : ℝ), x * (2:ℝ)^3 = k ∧ x = 1) :=
by
  have hx : 8 = k := by
    rw [←h2, one_mul]
  
  use 1
  split
  . exact hx.symm
  . rfl

end inverse_variation_example_l468_468532


namespace middle_term_expansion_sum_abs_values_coeff_l468_468233

-- Define the binomial term
def binomial_term (n k : ℕ) (a b : ℤ) : ℤ :=
  (Nat.choose n k) * (a ^ (n - k)) * (b ^ k)

-- Define conditions for the binomial (1 - x)^10
def exp_1_sub_x_10 := (1 - X)^10

-- Statement 1: Prove that the middle term is 252(-x)^5
theorem middle_term_expansion : 
  binomial_term 10 5 1 (-X) = 252 * (-X)^5 := 
  by
  sorry

-- Statement 2: Prove that the sum of absolute values of the coefficients is 1024
theorem sum_abs_values_coeff : 
  (∑ k in Finset.range (11), abs (binomial_term 10 k 1 (-1))) = 1024 := 
  by
  sorry

end middle_term_expansion_sum_abs_values_coeff_l468_468233


namespace median_is_4_l468_468064

theorem median_is_4 (a : ℤ) (h : {a, 2, 4, 0, 5} = {0, 2, 4, 5, a} ∨ {a, 2, 4, 0, 5} = {0, 2, 4, a, 5}) : a = 4 :=
sorry

end median_is_4_l468_468064


namespace initial_numbers_conditions_l468_468193

theorem initial_numbers_conditions (a b c : ℤ)
    (h : ∀ (x y z : ℤ), (x, y, z) = (17, 1967, 1983) → 
      x = y + z - 1 ∨ y = x + z - 1 ∨ z = x + y - 1) :
  (a = 2 ∧ b = 2 ∧ c = 2) → false ∧ 
  (a = 3 ∧ b = 3 ∧ c = 3) → true := 
sorry

end initial_numbers_conditions_l468_468193


namespace construct_square_from_points_l468_468945

-- Definitions of the points and the square
variables {K P R Q A B C D M N O : Point}

-- Given conditions:
axiom side_K : lies_on_side K A B
axiom side_P : lies_on_side P B C
axiom side_R : lies_on_side R C D
axiom side_Q : lies_on_side Q D A

-- Properties of diagonals and angles
axiom angle_KBR_right : ∠ K B R = 90°
axiom angle_PDR_right : ∠ P D R = 90°
axiom midpoint_M : midpoint M K P
axiom midpoint_N : midpoint N R Q
axiom diag_MN_bisects : bisects_diagonal M N A B C D
axiom square_diagonals_intersect_at_O : intersection O (diagonal A C) (diagonal B D)

-- The proof problem statement
theorem construct_square_from_points (K P R Q : Point) 
  (side_K : lies_on_side K A B) 
  (side_P : lies_on_side P B C)
  (side_R : lies_on_side R C D) 
  (side_Q : lies_on_side Q D A) : 
  ∃ (A B C D : Point), square A B C D := 
sorry

end construct_square_from_points_l468_468945


namespace neg_neg_one_gt_neg_two_l468_468288

theorem neg_neg_one_gt_neg_two : - (-1) > - (+2) := 
by
  sorry

end neg_neg_one_gt_neg_two_l468_468288


namespace probability_two_successes_l468_468940

noncomputable def prob_success : ℚ := 1 / 3

def prob_two_successes_in_three_trials : ℚ := 
  (nat.choose 3 2 : ℚ) * (prob_success ^ 2) * ((1 - prob_success) ^ 1)

theorem probability_two_successes : prob_two_successes_in_three_trials = 2 / 9 :=
by
  sorry

end probability_two_successes_l468_468940


namespace smallest_n_geometric_seq_l468_468331

noncomputable def geom_seq (a r : ℝ) (n : ℕ) : ℝ :=
  a * r ^ (n - 1)

noncomputable def S_n (a r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then n * a else a * (1 - r ^ n) / (1 - r)

theorem smallest_n_geometric_seq :
  (∃ n : ℕ, S_n (1/9) 3 n > 2018) ∧ ∀ m : ℕ, m < 10 → S_n (1/9) 3 m ≤ 2018 :=
by
  sorry

end smallest_n_geometric_seq_l468_468331


namespace a2008_is_3_over_7_l468_468478

def sequence (a : ℕ → ℚ) : Prop := 
  a 0 = 6 / 7 ∧ 
  ∀ n, a (n + 1) = if a n < (1 : ℚ) / 2 then 2 * a n else 2 * a n - 1

theorem a2008_is_3_over_7 (a : ℕ → ℚ) (h : sequence a) : 
  a 2008 = 3 / 7 := 
sorry

end a2008_is_3_over_7_l468_468478


namespace mark_eggs_supply_l468_468491

theorem mark_eggs_supply (dozen_eggs: ℕ) (days_in_week: ℕ) : dozen_eggs = 12 → 
  days_in_week = 7 → (5 * dozen_eggs + 30) * days_in_week = 630 :=
by 
  intros h_dozen h_days;
  rw [h_dozen, h_days];
  simp;
  norm_num;
  exact rfl

end mark_eggs_supply_l468_468491


namespace minimum_number_of_girls_l468_468906

theorem minimum_number_of_girls (total_students : ℕ) (d : ℕ) 
  (h_students : total_students = 20) 
  (h_unique_lists : ∀ n : ℕ, ∃! k : ℕ, k ≤ 20 - d ∧ n = 2 * k) :
  d ≥ 6 :=
sorry

end minimum_number_of_girls_l468_468906


namespace percentage_loss_is_correct_l468_468495

noncomputable def initial_cost : ℝ := 300
noncomputable def selling_price : ℝ := 255
noncomputable def loss : ℝ := initial_cost - selling_price
noncomputable def percentage_loss : ℝ := (loss / initial_cost) * 100

theorem percentage_loss_is_correct :
  percentage_loss = 15 :=
sorry

end percentage_loss_is_correct_l468_468495


namespace find_symmetric_point_l468_468309

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def M : Point := ⟨3, -3, -1⟩

def line (x y z : ℝ) : Prop := 
  (x - 6) / 5 = (y - 3.5) / 4 ∧ (x - 6) / 5 = (z + 0.5) / 0

theorem find_symmetric_point (M' : Point) :
  (line M.x M.y M.z) →
  M' = ⟨-1, 2, 0⟩ := by
  sorry

end find_symmetric_point_l468_468309


namespace sequence_inequality_l468_468883

theorem sequence_inequality 
  (a : ℕ → ℝ)
  (m n : ℕ)
  (h1 : a 1 = 21/16)
  (h2 : ∀ n ≥ 2, 2 * a n - 3 * a (n - 1) = 3 / 2^(n + 1))
  (h3 : m ≥ 2)
  (h4 : n ≤ m) :
  (a n + 3 / 2^(n + 3))^(1 / m) * (m - (2 / 3)^(n * (m - 1) / m)) < (m^2 - 1) / (m - n + 1) :=
sorry

end sequence_inequality_l468_468883


namespace infinite_perfect_squares_in_ap_l468_468340

open Nat

def is_arithmetic_progression (a d : ℕ) (an : ℕ → ℕ) : Prop :=
  ∀ n, an n = a + n * d

def is_perfect_square (x : ℕ) : Prop :=
  ∃ m, m * m = x

theorem infinite_perfect_squares_in_ap (a d : ℕ) (an : ℕ → ℕ) (m : ℕ)
  (h_arith_prog : is_arithmetic_progression a d an)
  (h_initial_square : a = m * m) :
  ∃ (f : ℕ → ℕ), ∀ n, is_perfect_square (an (f n)) :=
sorry

end infinite_perfect_squares_in_ap_l468_468340


namespace diophantine_solution_range_l468_468131

theorem diophantine_solution_range {a b c n : ℤ} (coprime_ab : Int.gcd a b = 1) :
  (∃ (x y : ℕ), a * x + b * y = c ∧ ∀ k : ℤ, k ≥ 1 → ∃ (x y : ℕ), a * (x + k * b) + b * (y - k * a) = c) → 
  ((n - 1) * a * b + a + b ≤ c ∧ c ≤ (n + 1) * a * b) :=
sorry

end diophantine_solution_range_l468_468131


namespace smallest_units_C_union_D_l468_468523

-- Definitions for the sets C and D and their sizes
def C_units : ℝ := 25.5
def D_units : ℝ := 18.0

-- Definition stating the inclusion-exclusion principle for sets C and D
def C_union_D (C_units D_units C_intersection_units : ℝ) : ℝ :=
  C_units + D_units - C_intersection_units

-- Statement to prove the minimum units in C union D
theorem smallest_units_C_union_D : ∃ h, h ≤ C_union_D C_units D_units D_units ∧ h = 25.5 := by
  sorry

end smallest_units_C_union_D_l468_468523


namespace right_triangle_of_altitude_l468_468874

theorem right_triangle_of_altitude (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]
  (h : ∀ {a b c d : ℝ}, a * b = c^2 → b = a → c = d) : 
  (∃ (AD BD CD : ℝ), AD * BD = CD^2) → 
  ∃ (ABC : Type), (is_right_triangle ABC). 
  sorry

end right_triangle_of_altitude_l468_468874


namespace average_students_is_12_l468_468408

-- Definitions based on the problem's conditions
variables (a b c : Nat)

-- Given conditions
axiom condition1 : a + b + c = 30
axiom condition2 : a + c = 19
axiom condition3 : b + c = 9

-- Prove that the number of average students (c) is 12
theorem average_students_is_12 : c = 12 := by 
  sorry

end average_students_is_12_l468_468408


namespace cost_per_pouch_is_20_l468_468451

theorem cost_per_pouch_is_20 :
  let boxes := 10
  let pouches_per_box := 6
  let dollars := 12
  let cents_per_dollar := 100
  let total_pouches := boxes * pouches_per_box
  let total_cents := dollars * cents_per_dollar
  let cost_per_pouch := total_cents / total_pouches
  cost_per_pouch = 20 :=
by
  sorry

end cost_per_pouch_is_20_l468_468451


namespace optimal_selection_method_uses_golden_ratio_l468_468675

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468675


namespace average_payment_l468_468244

theorem average_payment (total_payments : ℕ) (first_n_payments : ℕ)  (first_payment_amt : ℕ) (remaining_payment_amt : ℕ) 
  (H1 : total_payments = 104)
  (H2 : first_n_payments = 24)
  (H3 : first_payment_amt = 520)
  (H4 : remaining_payment_amt = 615)
  :
  (24 * 520 + 80 * 615) / 104 = 593.08 := 
  by 
    sorry

end average_payment_l468_468244


namespace optimal_selection_method_uses_golden_ratio_l468_468860

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l468_468860


namespace calculate_lego_set_cost_l468_468320

variable (total_revenue_after_tax : ℝ) (little_cars_base_price : ℝ)
  (discount_rate : ℝ) (tax_rate : ℝ) (num_little_cars : ℕ)
  (num_action_figures : ℕ) (num_board_games : ℕ)
  (lego_set_cost_before_tax : ℝ)

theorem calculate_lego_set_cost :
  total_revenue_after_tax = 136.50 →
  little_cars_base_price = 5 →
  discount_rate = 0.10 →
  tax_rate = 0.05 →
  num_little_cars = 3 →
  num_action_figures = 2 →
  num_board_games = 1 →
  lego_set_cost_before_tax = 85 :=
by
  sorry

end calculate_lego_set_cost_l468_468320


namespace hua_luogeng_optimal_selection_method_l468_468784

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l468_468784


namespace bisection_method_root_interval_l468_468991

noncomputable def f (x : ℝ) : ℝ := 3 * x + 3^x - 8

theorem bisection_method_root_interval : 
  f 1 < 0 ∧ f 1.5 > 0 ∧ f 1.25 > 0 → (∃ x ∈ (Ioo 1 1.25), f x = 0) :=
by
  intro h
  sorry

end bisection_method_root_interval_l468_468991


namespace minimum_number_of_girls_l468_468902

theorem minimum_number_of_girls (total_students : ℕ) (d : ℕ) 
  (h_students : total_students = 20) 
  (h_unique_lists : ∀ n : ℕ, ∃! k : ℕ, k ≤ 20 - d ∧ n = 2 * k) :
  d ≥ 6 :=
sorry

end minimum_number_of_girls_l468_468902


namespace lcm_fractions_l468_468207

noncomputable def lcm_of_fractions := (1 : ℚ) / (12 * x)

theorem lcm_fractions (x : ℚ) (hx : x ≠ 0):
  Nat.lcm(2 * x.num, 4 * x.num) = 12 * x.num → 
  (Nat.lcm(Nat.gcd (2 * x.d, 4 * x.d) * x.d, Nat.gcd (2 * x.d, 6 * x.d) * x.d) = 12 * x.d) :=
sorry

end lcm_fractions_l468_468207


namespace matilda_smartphone_loss_percentage_l468_468502

theorem matilda_smartphone_loss_percentage :
  ∀ (initial_cost selling_price : ℝ),
  initial_cost = 300 →
  selling_price = 255 →
  let loss := initial_cost - selling_price in
  let percentage_loss := (loss / initial_cost) * 100 in
  percentage_loss = 15 :=
by
  intros initial_cost selling_price h₁ h₂
  let loss := initial_cost - selling_price
  let percentage_loss := (loss / initial_cost) * 100
  rw [h₁, h₂]
  sorry

end matilda_smartphone_loss_percentage_l468_468502


namespace cos_dihedral_angle_l468_468436

-- Define the planes and lines
variable (α : Plane) (β : Plane) (γ : Plane) (AB : Line) (BC : Line)
-- Define the conditions
variable (h1 : OnLine AB α) (h2 : OnLine BC α) 
          (h3 : Perpendicular AB BC α) (h4 : OnLine AB β) (h5 : OnLine BC γ)
          (h6 : AcuteDihedralAngle α AB β (π / 3)) 
          (h7 : AcuteDihedralAngle α BC γ (π / 3))

-- Statement to prove
theorem cos_dihedral_angle (α β γ : Plane) (AB BC : Line)
  (h1 : α.OnLine AB) (h2 : α.OnLine BC)
  (h3 : α.Perpendicular AB BC)
  (h4 : β.OnLine AB) (h5 : γ.OnLine BC)
  (h6 : AcuteDihedralAngle α AB β (π / 3))
  (h7 : AcuteDihedralAngle α BC γ (π / 3)) :
  AcuteDihedralAngleCos β γ = 1 / 4 :=
sorry

end cos_dihedral_angle_l468_468436


namespace maximize_sqrt_expression_l468_468941

theorem maximize_sqrt_expression :
  let a := Real.sqrt 8
  let b := Real.sqrt 2
  (a + b) > max (max (a - b) (a * b)) (a / b) := by
  sorry

end maximize_sqrt_expression_l468_468941


namespace max_16_numbers_l468_468010

theorem max_16_numbers (a : ℕ → ℝ) (h_len : ∀ n, n < 16 → 0 < a n) (h_sum : ∑ i in Finset.range 16, a i = 100) (h_sum_sq : ∑ i in Finset.range 16, (a i)^2 = 1000) : ∀ n, n < 16 → a n ≤ 25 :=
by
  sorry

end max_16_numbers_l468_468010


namespace optimal_selection_uses_golden_ratio_l468_468817

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l468_468817


namespace optimalSelectionUsesGoldenRatio_l468_468718

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l468_468718


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468769

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468769


namespace hua_luogeng_optimal_selection_l468_468552

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l468_468552


namespace stock_net_percentage_change_l468_468184

theorem stock_net_percentage_change :
  let initial_price := 100.0
  let first_year_change := initial_price * (1 - 0.08)
  let first_year_dividend := initial_price * 0.03
  let after_first_year := first_year_change + first_year_dividend
  let second_year_change := after_first_year * (1 + 0.10)
  let second_year_dividend := second_year_change * 0.03
  let after_second_year := second_year_change + second_year_dividend
  let third_year_change := after_second_year * (1 + 0.06)
  let third_year_dividend := third_year_change * 0.03
  let final_price := third_year_change + third_year_dividend
  let net_percentage_change := ((final_price - initial_price) / initial_price) * 100
  abs (net_percentage_change - 17.52) < 0.01 :=
by
  -- proof goes here
  sorry

end stock_net_percentage_change_l468_468184


namespace sum_of_negatives_2142_l468_468880

noncomputable def sum_of_consecutive_negative_integers (n : ℤ) (h1 : n * (n + 1) = 2142) (h2 : n < 0) : ℤ := n + (n + 1)

theorem sum_of_negatives_2142 : ∃ n : ℤ, n * (n + 1) = 2142 ∧ n < 0 ∧ sum_of_consecutive_negative_integers n (by assumption) (by assumption) = -93 :=
by 
  use -47
  split
  { sorry }
  split
  { sorry }
  { sorry }

end sum_of_negatives_2142_l468_468880


namespace hua_luogeng_optimal_selection_method_l468_468736

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l468_468736


namespace hockey_no_10_goals_l468_468262

theorem hockey_no_10_goals :
  ∀ A I S : ℕ, 
  (A = 3 ∨ A = 2) ∧ (I = 1 ∨ I = 4) ∧ (S = 5 ∨ S = 6) ∧
  (A + I + S = 10) →
  False := by
  intros A I S h
  cases h with hA hRest
  cases hRest with hI hS
  cases hS with hSum total_goals
  cases hA with anton3 anton2
  case or.inl =>
    -- Anton scored 3
    cases hS with sergei5 sergei6
    case or.inr =>
      -- Sergey scored 6
      cases hI with ilya1 ilya4
      case or.inr =>
        -- Ilya scored 4
        have total_goals := 3 + 6 + 4
        sorry
    sorry
    sorry
  case or.inr =>
    -- Anton scored 2
    sorry

    cases hS with sergei5 sergei6
    case or.inr =>
      -- Sergey scored 6
      cases hI with ilya1 ilya4
      case or.inr =>
        -- Ilya scored 4
        have total_goals := 2 + 6 + 4
        sorry
    sorry
    sorry
  sorry

end hockey_no_10_goals_l468_468262


namespace hua_luogeng_optimal_selection_method_l468_468781

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l468_468781


namespace part1_part2_l468_468343

open Real

variables {a b c : ℝ}

theorem part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 1 / a + 1 / b + 1 / c = 1) :
    a + 4 * b + 9 * c ≥ 36 :=
sorry

theorem part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 1 / a + 1 / b + 1 / c = 1) :
    (b + c) / sqrt a + (a + c) / sqrt b + (a + b) / sqrt c ≥ 2 * sqrt (a * b * c) :=
sorry

end part1_part2_l468_468343


namespace find_area_of_triangle_abc_l468_468067

-- Define the scenario of the problem
def isosceles_right_triangle (A B C : Type) [triangle A B C] (right_ang_C : is_right_angle C)
(midpoint_M : is_midpoint M A B) (altitude_CH : is_altitude CH) : 
  Prop :=
right_ang_C.angle = 90 ∧ right_ang_C.is_isosceles = true ∧ AC = BC ∧ 
    CH.bisects (angle A C B) ∧ M = midpoint (A, B)

-- Given area condition
constant area_chm : Type → ℝ
constant area_abc : Type → ℝ

-- The goal is to demonstrate the following theorem
theorem find_area_of_triangle_abc (A B C M : Type) [triangle A B C] 
  (right_ang_C : is_right_angle C) 
  (midpoint_M : is_midpoint M A B) 
  (altitude_CH : is_altitude CH) 
  (h1 : isosceles_right_triangle A B C right_ang_C midpoint_M altitude_CH)
  (h_area_chm : area_chm = K) :
  area_abc = 16 * K :=
sorry

end find_area_of_triangle_abc_l468_468067


namespace xy_sum_is_2_l468_468124

theorem xy_sum_is_2 (x y : ℝ) 
  (h1 : (x - 1) ^ 3 + 1997 * (x - 1) = -1)
  (h2 : (y - 1) ^ 3 + 1997 * (y - 1) = 1) : 
  x + y = 2 := 
  sorry

end xy_sum_is_2_l468_468124


namespace max_value_of_m_l468_468323

-- Define the curves
def C1 (a x : ℝ) : ℝ := (x^2) / 2 + a * x
def C2 (a x m : ℝ) : ℝ := 2 * a^2 * Real.log x + m

-- Define the condition that tangents coincide at intersection point
def tangents_coincide_at (a x0 : ℝ) : Prop :=
  x0 + a = (2 * a^2) / x0

-- Define the main problem statement
theorem max_value_of_m (a : ℝ) (h : 0 < a) (m : ℝ) :
  (∃ x0 : ℝ, C1 a x0 = C2 a x0 m ∧ tangents_coincide_at a x0) →
  m ≤ Real.exp (1 / 2) :=
by
  sorry

end max_value_of_m_l468_468323


namespace optimal_selection_method_uses_golden_ratio_l468_468614

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l468_468614


namespace jessica_total_spent_l468_468456

noncomputable def catToyCost : ℝ := 10.22
noncomputable def cageCost : ℝ := 11.73
noncomputable def totalCost : ℝ := 21.95

theorem jessica_total_spent :
  catToyCost + cageCost = totalCost :=
sorry

end jessica_total_spent_l468_468456


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468764

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468764


namespace smallest_value_n_l468_468158

def factorial_factors_of_five (n : ℕ) : ℕ :=
  n / 5 + n / 25 + n / 125 + n / 625 + n / 3125

theorem smallest_value_n
  (a b c m n : ℕ)
  (h1 : a + b + c = 2003)
  (h2 : a = 2 * b)
  (h3 : a.factorial * b.factorial * c.factorial = m * 10 ^ n)
  (h4 : ¬ (10 ∣ m)) :
  n = 400 :=
by
  sorry

end smallest_value_n_l468_468158


namespace final_number_is_57_l468_468098

-- Define the initial range of numbers from 1 to 120
def initial_list := List.range' 1 120

-- Define the function to mark every third number from the list
def mark_every_third (lst : List ℕ) : List ℕ :=
  lst.enum.filter (λ ⟨i, _⟩, (i + 1) % 3 ≠ 1).unzip.2

-- Define the function to perform the given marking and reversing procedure
def process_list (lst : List ℕ) : List ℕ :=
  let rec aux (l : List ℕ) : List ℕ :=
    let l' := mark_every_third l
    if l'.length = 1 then l'
    else aux l'.reverse
  aux lst

-- Define the last remaining number from the processed list
def last_remaining_number : ℕ :=
  (process_list initial_list).head

-- State the theorem that the last remaining number is 57
theorem final_number_is_57 : last_remaining_number = 57 :=
by
  sorry

end final_number_is_57_l468_468098


namespace find_b_l468_468345

theorem find_b (b : ℤ) :
  ∃ (r₁ r₂ : ℤ), (r₁ = -9) ∧ (r₁ * r₂ = 36) ∧ (r₁ + r₂ = -b) → b = 13 :=
by {
  sorry
}

end find_b_l468_468345


namespace minimum_girls_in_class_l468_468924

theorem minimum_girls_in_class (n : ℕ) (d : ℕ) (boys : List (List ℕ)) 
  (h_students : n = 20)
  (h_boys : boys.length = n - d)
  (h_unique : ∀ i j k : ℕ, i < boys.length → j < boys.length → k < boys.length → i ≠ j → j ≠ k → i ≠ k → 
    (boys.nthLe i (by linarith)).length ≠ (boys.nthLe j (by linarith)).length ∨ 
    (boys.nthLe j (by linarith)).length ≠ (boys.nthLe k (by linarith)).length ∨ 
    (boys.nthLe k (by linarith)).length ≠ (boys.nthLe i (by linarith)).length) :
  d = 6 := 
begin
  sorry
end

end minimum_girls_in_class_l468_468924


namespace optimal_selection_uses_golden_ratio_l468_468828

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l468_468828


namespace possible_lightest_boy_heaviest_girl_l468_468422

theorem possible_lightest_boy_heaviest_girl
    (B G : ℕ) 
    (wb : ℕ → ℕ) 
    (wg : ℕ → ℕ) 
    (N < 35) 
    (avg_weight_students : Real) 
    (avg_weight_girls : Real) 
    (avg_weight_boys : Real) : 
    (avg_weight_students = 53.5) ∧ 
    (avg_weight_girls = 47) ∧ 
    (avg_weight_boys = 60) → 
    (∃ i j : ℕ, (i < B) ∧ (j < G) ∧ (wb i < wg j)) :=
by 
  sorry

end possible_lightest_boy_heaviest_girl_l468_468422


namespace general_formula_condition1_general_formula_condition2_general_formula_condition3_sum_Tn_l468_468358

-- Define the conditions
axiom condition1 (n : ℕ) (h : 2 ≤ n) : ∀ (a : ℕ → ℕ), a 1 = 1 → a n = n / (n-1) * a (n-1)
axiom condition2 (n : ℕ) : ∀ (S : ℕ → ℕ), 2 * S n = n^2 + n
axiom condition3 (n : ℕ) : ∀ (a : ℕ → ℕ), a 1 = 1 → a 3 = 3 → (a n + a (n+2)) = 2 * a (n+1)

-- Proof statements
theorem general_formula_condition1 : ∀ (n : ℕ) (a : ℕ → ℕ) (h : 2 ≤ n), (a 1 = 1) → (∀ n, a n = n) :=
by sorry

theorem general_formula_condition2 : ∀ (n : ℕ) (S a : ℕ → ℕ), (2 * S n = n^2 + n) → (∀ n, a n = n) :=
by sorry

theorem general_formula_condition3 : ∀ (n : ℕ) (a : ℕ → ℕ), (a 1 = 1) → (a 3 = 3) → (∀ n, a n + a (n+2) = 2 * a (n+1)) → (∀ n, a n = n) :=
by sorry

theorem sum_Tn : ∀ (b : ℕ → ℕ) (T : ℕ → ℝ), (b 1 = 2) → (b 2 + b 3 = 12) → (∀ n, T n = 2 * (1 - 1 / (n + 1))) :=
by sorry

end general_formula_condition1_general_formula_condition2_general_formula_condition3_sum_Tn_l468_468358


namespace possible_lightest_boy_heaviest_girl_l468_468427

theorem possible_lightest_boy_heaviest_girl :
  ∃ (B G : ℕ) (boys_weights girls_weights : list ℕ),
    B + G < 35 ∧
    (∑ w in boys_weights, w) / B = 60 ∧
    (∑ w in girls_weights, w) / G = 47 ∧
    (∑ w in (boys_weights ++ girls_weights), w) / (B + G) = 53.5 ∧
    (∃ l, l ∈ boys_weights ∧ ∀ g, g ∈ girls_weights → l < g) :=
begin
  sorry,
end

end possible_lightest_boy_heaviest_girl_l468_468427


namespace part1_part2_l468_468344

def A := {x : ℝ | x^2 - 2 * x - 8 ≤ 0}
def B (m : ℝ) := {x : ℝ | x^2 - 2 * x + 1 - m^2 ≤ 0}

theorem part1 (m : ℝ) (hm : m = 2) :
  A ∩ {x : ℝ | x < -1 ∨ 3 < x} = {x : ℝ | -2 ≤ x ∧ x < -1 ∨ 3 < x ∧ x ≤ 4} :=
sorry

theorem part2 :
  (∀ x, x ∈ A → x ∈ B (m : ℝ)) ↔ (0 < m ∧ m ≤ 3) :=
sorry

end part1_part2_l468_468344


namespace radian_measure_of_central_angle_l468_468537

-- Definitions of the conditions as Lean functions
def arc_length (θ r : ℝ) : ℝ := θ * r
def sector_area (θ r : ℝ) : ℝ := (1/2) * θ * r^2

-- Theorem stating that the radian measure of the central angle θ is 3/2
theorem radian_measure_of_central_angle (θ r : ℝ) 
  (h1 : arc_length θ r = 3) 
  (h2 : sector_area θ r = 3) : θ = 3 / 2 :=
by
  sorry

end radian_measure_of_central_angle_l468_468537


namespace find_values_l468_468973

theorem find_values :
  ∃ x y, x = 5 ∧ y = -2 ∧ (2 * x + y = 8 ∧ 2 * x - y = 12) :=
by
  use 5
  use -2
  simp
  split
  { ring }
  { ring }

end find_values_l468_468973


namespace hua_luogeng_optimal_selection_l468_468559

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l468_468559


namespace triangle_problem_solution_l468_468443

theorem triangle_problem_solution
  (A : ℝ)
  (b c a : ℝ)
  (sinA sinB sinC : ℝ)
  (cosA : ℝ)
  (SinRule : (a / sinA = b / sinB ∧ a / sinA = c / sinC))
  (S : sinA + sinB + sinC)
  (H_A : A = π / 3)
  (H_b : b = 1)
  (H_c : c = 4)
  (cos_A : cosA = 1 / 2)
  (sin_A : sinA = (real.sqrt 3) / 2)
  (cosine_rule : a^2 = b^2 + c^2 - 2 * b * c * cosA): 
  (a + b + c) / (sinA + sinB + sinC) = (2 * real.sqrt 39) / 3 :=
by
  sorry

end triangle_problem_solution_l468_468443


namespace no_infinite_prime_seq_l468_468515

theorem no_infinite_prime_seq (p : Nat → Nat) :
  (∀ k, Prime (p k)) →
  (∀ k, p (k + 1) = 5 * (p k) + 4) →
  False :=
by
  assume h_prime h_rec
  sorry

end no_infinite_prime_seq_l468_468515


namespace octagon_square_ratio_l468_468130

theorem octagon_square_ratio (a : ℝ) (h_a_pos : 0 < a) :
  let n := 2 * (1 + Real.sqrt 2) * a^2,
      m := 2 * a^2
  in m / n = Real.sqrt 2 / 2 :=
by
  have h_n : n = 2 * (1 + Real.sqrt 2) * a^2 :=
    by rw [n],
  have h_m : m = 2 * a^2 :=
    by rw [m],
  sorry

end octagon_square_ratio_l468_468130


namespace optimal_selection_method_uses_golden_ratio_l468_468839

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468839


namespace optimalSelectionUsesGoldenRatio_l468_468712

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l468_468712


namespace optimal_selection_uses_golden_ratio_l468_468666

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l468_468666


namespace cone_height_ratio_l468_468253

theorem cone_height_ratio (circumference : ℝ) (orig_height : ℝ) (short_volume : ℝ)
  (h_circumference : circumference = 20 * Real.pi)
  (h_orig_height : orig_height = 40)
  (h_short_volume : short_volume = 400 * Real.pi) :
  let r := circumference / (2 * Real.pi)
  let h_short := (3 * short_volume) / (Real.pi * r^2)
  (h_short / orig_height) = 3 / 10 :=
by {
  sorry
}

end cone_height_ratio_l468_468253


namespace optimal_selection_method_uses_golden_ratio_l468_468565

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l468_468565


namespace find_value_of_a2_b2_c2_l468_468101

variable {a b c : ℝ}

theorem find_value_of_a2_b2_c2
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : a + b + c = 0)
  (h3 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 5 := 
sorry

end find_value_of_a2_b2_c2_l468_468101


namespace minimum_girls_in_class_l468_468927

theorem minimum_girls_in_class (n : ℕ) (d : ℕ) (boys : List (List ℕ)) 
  (h_students : n = 20)
  (h_boys : boys.length = n - d)
  (h_unique : ∀ i j k : ℕ, i < boys.length → j < boys.length → k < boys.length → i ≠ j → j ≠ k → i ≠ k → 
    (boys.nthLe i (by linarith)).length ≠ (boys.nthLe j (by linarith)).length ∨ 
    (boys.nthLe j (by linarith)).length ≠ (boys.nthLe k (by linarith)).length ∨ 
    (boys.nthLe k (by linarith)).length ≠ (boys.nthLe i (by linarith)).length) :
  d = 6 := 
begin
  sorry
end

end minimum_girls_in_class_l468_468927


namespace infinitely_many_primes_of_form_4k_plus_3_l468_468154

theorem infinitely_many_primes_of_form_4k_plus_3 : ∀ (k : ℤ), ∃ (p : ℕ), prime p ∧ p = 4 * k + 3 :=
by
  assume k : ℤ
  -- include the conditions and relevant definitions
  sorry

end infinitely_many_primes_of_form_4k_plus_3_l468_468154


namespace linoleum_cut_rearrange_l468_468250

def linoleum : Type := sorry -- placeholder for the specific type of the linoleum piece

def A : linoleum := sorry -- define piece A
def B : linoleum := sorry -- define piece B

def cut_and_rearrange (L : linoleum) (A B : linoleum) : Prop :=
  -- Define the proposition that pieces A and B can be rearranged into an 8x8 square
  sorry

theorem linoleum_cut_rearrange (L : linoleum) (A B : linoleum) :
  cut_and_rearrange L A B :=
sorry

end linoleum_cut_rearrange_l468_468250


namespace pages_revised_once_is_30_l468_468149

noncomputable def numberOfPagesRevisedOnlyOnce
    (costFirstTime : ℝ) (costPerRevision : ℝ) (totalPages : ℕ) (pagesRevisedTwice : ℕ)
    (totalCost : ℝ) : ℕ :=
  let originalCost := costFirstTime * totalPages
  let revisedTwiceCost := costPerRevision * 2 * pagesRevisedTwice
  let remainingCost := totalCost - originalCost - revisedTwiceCost
  let pagesRevisedOnce := remainingCost / costPerRevision
  pagesRevisedOnce.toNat

theorem pages_revised_once_is_30 :
  numberOfPagesRevisedOnlyOnce 5 3 100 20 710 = 30 :=
by
  sorry

end pages_revised_once_is_30_l468_468149


namespace constant_term_in_expansion_l468_468083

theorem constant_term_in_expansion (x : ℝ) :
  let expr := (x^2 - x⁻¹ + 1)^4 in
  constant_term(expr) = 13 :=
begin
  sorry
end

end constant_term_in_expansion_l468_468083


namespace smallest_undefined_fraction_l468_468295

theorem smallest_undefined_fraction : ∃ x, (9 * x^2 - 74 * x + 8 = 0) ∧ ∀ y, (9 * y^2 - 74 * y + 8 = 0) → (x ≤ y) := 
begin
  use (2 / 9),
  split,
  { -- Proof that 9 * (2 / 9)^2 - 74 * (2 / 9) + 8 = 0
    sorry },
  { -- Proof that (2 / 9) is the smallest solution
    intros y hy,
    -- Use facts from algebra to show (2 / 9) ≤ y
    sorry }
end

end smallest_undefined_fraction_l468_468295


namespace difference_between_extreme_dimes_l468_468142

theorem difference_between_extreme_dimes :
  ∃ (n d h : ℕ), n + d + h = 150 ∧ 5 * n + 10 * d + 50 * h = 1250 ∧ (dmax : ℕ) (dmin : ℕ), 
  (dmax = 100) ∧ (dmin = 1) ∧ dmax - dmin = 99 :=
by
  sorry

end difference_between_extreme_dimes_l468_468142


namespace min_number_of_girls_l468_468892

theorem min_number_of_girls (d : ℕ) (students : ℕ) (boys : ℕ → ℕ) : 
  students = 20 ∧ ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → boys i ≠ boys j ∨ boys j ≠ boys k ∨ boys i ≠ boys k → d = 6 :=
by
  sorry

end min_number_of_girls_l468_468892


namespace prove_a2_b2_c2_zero_l468_468106

theorem prove_a2_b2_c2_zero (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) (h5 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) : a^2 + b^2 + c^2 = 0 := 
by 
  sorry

end prove_a2_b2_c2_zero_l468_468106


namespace optimal_selection_uses_golden_ratio_l468_468634

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l468_468634


namespace repeating_decimal_fraction_l468_468302

theorem repeating_decimal_fraction :
  let x := 0.888 + 0.000888 + 0.000000888 + ...
  let y := 1.222 + 0.000222 + 0.000000222 + ...
  x / y = (8 / 9) / (11 / 9) → x / y = 8 / 11 :=
sorry

end repeating_decimal_fraction_l468_468302


namespace hua_luogeng_optimal_selection_method_l468_468779

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l468_468779


namespace Reeya_fifth_subject_score_l468_468150

theorem Reeya_fifth_subject_score 
  (a1 a2 a3 a4 : ℕ) (avg : ℕ) (subjects : ℕ) (a1_eq : a1 = 55) (a2_eq : a2 = 67) (a3_eq : a3 = 76) 
  (a4_eq : a4 = 82) (avg_eq : avg = 73) (subjects_eq : subjects = 5) :
  ∃ a5 : ℕ, (a1 + a2 + a3 + a4 + a5) / subjects = avg ∧ a5 = 85 :=
by
  sorry

end Reeya_fifth_subject_score_l468_468150


namespace optimal_selection_golden_ratio_l468_468690

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l468_468690


namespace find_a_l468_468029

-- Define the function f(x) and its derivative
def f (x : ℝ) := x^3 - 2 * x^2 + a * x + 3
def f' (a : ℝ) (x : ℝ) := 3 * x^2 - 4 * x + a

-- Formal statement asserting the monotonically increasing condition implies a >= 1
theorem find_a (a : ℝ) :
  (∀ x ∈ set.Icc 1 2, 0 ≤ f' a x) → 1 ≤ a :=
by
  intro h
  have h1 : 0 ≤ f' a 1 := h 1 (set.left_mem_Icc.mpr (by linarith))
  rw [f'] at h1
  exact (show 1 ≤ a, by linarith)

end find_a_l468_468029


namespace solve_equation_l468_468526

theorem solve_equation (x : ℝ) (h : x ≠ 4) :
  (x - 3) / (4 - x) - 1 = 1 / (x - 4) → x = 3 :=
by
  sorry

end solve_equation_l468_468526


namespace optimal_selection_method_uses_golden_ratio_l468_468752

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l468_468752


namespace fair_coin_probability_l468_468201

theorem fair_coin_probability (coin_tosses : ℕ → Bool) (h_fair : ∀ n, P n = 0.5) :
  P (coin_tosses 11) = 0.5 :=
sorry

end fair_coin_probability_l468_468201


namespace mark_egg_supply_in_a_week_l468_468492

def dozen := 12
def eggs_per_day_store1 := 5 * dozen
def eggs_per_day_store2 := 30
def daily_eggs_supplied := eggs_per_day_store1 + eggs_per_day_store2
def days_per_week := 7

theorem mark_egg_supply_in_a_week : daily_eggs_supplied * days_per_week = 630 := by
  sorry

end mark_egg_supply_in_a_week_l468_468492


namespace min_value_f_prime_l468_468028

variable (m : ℝ)
noncomputable def f (x : ℝ) := x^4 * Real.cos x + m * x^2 + 2 * x
noncomputable def f' (x : ℝ) := 4 * x^3 * Real.cos x - x^4 * Real.sin x + 2 * m * x + 2

theorem min_value_f_prime : 
  (∀ x ∈ Icc (-4 : ℝ) 4, f' m x ≤ 16) → 
  ∃ y ∈ Icc (-4 : ℝ) 4, f' m y = -12 :=
by 
  sorry

end min_value_f_prime_l468_468028


namespace optimal_selection_golden_ratio_l468_468689

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l468_468689


namespace optimal_selection_method_uses_golden_ratio_l468_468686

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468686


namespace cost_per_pouch_l468_468448

theorem cost_per_pouch (boxes : ℕ) (pouches_per_box : ℕ) (total_cost_dollars : ℕ) :
  boxes = 10 →
  pouches_per_box = 6 →
  total_cost_dollars = 12 →
  (total_cost_dollars * 100) / (boxes * pouches_per_box) = 20 :=
by
  intros,
  -- proof steps here
  sorry

end cost_per_pouch_l468_468448


namespace parabola_projection_l468_468373

theorem parabola_projection
  (F P : Point)
  (y2_8x : P ∈ parabola y^2 = 8*x)
  (F_focus : focus F (parabola y^2 = 8*x))
  (E : Point)
  (E_projection : projection P y-axis = E) :
  abs(distance P F - distance P E) = 2 :=
sorry

end parabola_projection_l468_468373


namespace optimal_selection_method_is_golden_ratio_l468_468597

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l468_468597


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468775

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468775


namespace probability_two_yellows_is_one_ninth_l468_468999

def total_marbles := 4 + 5 + 6
def probability_yellow : ℚ := 5 / total_marbles
def probability_two_yellows : ℚ := probability_yellow * probability_yellow

theorem probability_two_yellows_is_one_ninth :
  probability_two_yellows = 1 / 9 :=
sorry

end probability_two_yellows_is_one_ninth_l468_468999


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468770

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468770


namespace translated_function_is_even_and_in_range_l468_468176

open Real

noncomputable def f (x : ℝ) := cos (2 * x - π / 3) ^ 2

noncomputable def g (x : ℝ) := f (x + π / 6)

theorem translated_function_is_even_and_in_range :
  ∀ x : ℝ, g x = (1 + cos (4 * x)) / 2 ∧ g x = g (-x) ∧ 0 ≤ g x ∧ g x ≤ 1 :=
by
  sorry

end translated_function_is_even_and_in_range_l468_468176


namespace possible_lightest_boy_heaviest_girl_l468_468416

theorem possible_lightest_boy_heaviest_girl
    (B G : ℕ)
    (hN : B + G < 35)
    (w_boys w_girls : ℕ → ℝ)
    (avg_all : Real := 53.5)
    (avg_girls : Real := 47)
    (avg_boys : Real := 60) :
    (∑ i in Finset.range B, w_boys i) / B = avg_boys →
    (∑ i in Finset.range G, w_girls i) / G = avg_girls →
    ((∑ i in Finset.range B, w_boys i) + (∑ i in Finset.range G, w_girls i)) / (B + G) = avg_all →
    ∃ (i j : ℕ), i < B ∧ j < G ∧ w_boys i < all_weights_worst(G, w_girls) ∧ w_girls j > all_weights_best(B, w_boys) :=
sorry

/-- Helper function that returns the minimum weight of the given range of girls' weights --/
noncomputable def all_weights_worst (g: ℕ, wg: ℕ → ℝ) : ℝ :=
  Finset.min' (Finset.range g) (begin
    -- there should be at least one girl
    exact ⟨0, by simp [show 0 < g, by omega]⟩
  end)

/-- Helper function that returns the maximum weight of the given range of boys' weights --/
noncomputable def all_weights_best (b: ℕ, wb: ℕ → ℝ) : ℝ :=
  Finset.max' (Finset.range b) (begin
    -- there should be at least one boy
    exact ⟨0, by simp [show 0 < b, by omega]⟩
  end)

end possible_lightest_boy_heaviest_girl_l468_468416


namespace total_slices_l468_468989

def pizzas : ℕ := 2
def slices_per_pizza : ℕ := 8

theorem total_slices : pizzas * slices_per_pizza = 16 :=
by
  sorry

end total_slices_l468_468989


namespace moles_of_NaHCO3_l468_468379

-- Statement of the problem in Lean 4
theorem moles_of_NaHCO3 (moles_HCl : ℕ) (balanced_equation : ∀ {x y z w : ℕ}, x = y → z = w) : moles_HCl = 3 → 3 = 3 :=
by
  intro mHCl hmHCl_eq
  have hBalanced : balanced_equation (moles_HCl) (moles_HCl) 3 3 := by sorry
  exact eq.trans hBalanced hmHCl_eq

end moles_of_NaHCO3_l468_468379


namespace distance_between_points_l468_468246

theorem distance_between_points :
  let f (x y : ℝ) := x ^ 2 * 9 + y ^ 2 - 25,
      g (x y : ℝ) := x - y ^ 2 / 8 - 2 in
  ∃ x y : ℝ, f x y = 0 ∧ g x y = 0 ∧
  distance (y : ℝ) (y' : ℝ) = (4 * real.sqrt 14 / 3) := by sorry

end distance_between_points_l468_468246


namespace area_pentagon_AEDCB_l468_468148

variable (A B C D E : Type)
variable [metric_space E]
variables [add_comm_group E] [module ℝ E] [metric_space.dist_linear_space E]
variable (a b c d e : E)
variable (AE DE : ℝ)

-- Conditions
axiom A_square_ABCD : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ dist a b = dist b c ∧ dist b c = dist c d ∧ dist c d = dist d a ∧ dist d a = 25
axiom AE_perpendicular_ED : E E dist(a, (0: square root 625)) 7 24 sorry 

-- The statement to prove
theorem area_pentagon_AEDCB : AE = 7 → DE = 24 → (area_of_square - area_of_triangle) = 541 :=
by
  intro AE_7
  intro DE_24
  sorry

end area_pentagon_AEDCB_l468_468148


namespace optimal_selection_method_uses_golden_ratio_l468_468842

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468842


namespace other_root_of_quadratic_l468_468353

theorem other_root_of_quadratic (a b : ℝ) (h : (1:ℝ) = 1) (h_root : (1:ℝ) ^ 2 + a * (1:ℝ) + 2 = 0): b = 2 :=
by
  sorry

end other_root_of_quadratic_l468_468353


namespace optimal_selection_uses_golden_ratio_l468_468822

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l468_468822


namespace distance_range_l468_468080

theorem distance_range (P Q : ℝ × ℝ)
  (P_on_C1 : ∃ α : ℝ, P.1 = 2 * Real.cos α ∧ P.2 = Real.sqrt 2 * Real.sin α)
  (Q_on_C2 : (Q.1 - 1/2)^2 + Q.2^2 = 1/4) :
  (∃ α : ℝ, |(P.1 - Q.1)^2 + (P.2 - Q.2)^2)^0.5 = if Real.cos α = 0.5 then (Real.sqrt 7 - 1) / 2 else 3) :=
begin
  sorry
end

end distance_range_l468_468080


namespace complex_number_solution_l468_468328

theorem complex_number_solution (z : ℂ) (h : z * (2 - complex.i) = 3 + complex.i) : z = 1 + complex.i := 
sorry

end complex_number_solution_l468_468328


namespace sum_of_solutions_l468_468114

def f (x : ℝ) : ℝ := 12 * x + 5

theorem sum_of_solutions :
  let f_inv (y : ℝ) : ℝ := (y - 5) / 12 in
  let g (x : ℝ) : ℝ := f (1 / (3 * x)) in
  ∑ x in {x : ℝ | f_inv x = g x}.to_finset, x = 65 :=
by
  sorry

end sum_of_solutions_l468_468114


namespace minimum_interval_l468_468030

-- Define the two functions f and g
def f (x : ℝ) : ℝ := (range 2017).sum (λ n, ((-1) ^ n) * (x ^ n) / n.succ)

def g (x : ℝ) : ℝ := (range 2017).sum (λ n, ((-1) ^ (n + 1)) * (x ^ n) / n.succ)

-- Define the function F
def F (x : ℝ) : ℝ := f (x + 4) * g (x - 5)

-- Lean statement for the proof problem
theorem minimum_interval [a b : ℤ] (h1 : ∀ x ∈ [-5, 7], F x = 0) : b - a = 12 := 
sorry

end minimum_interval_l468_468030


namespace hua_luogeng_optimal_selection_l468_468546

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l468_468546


namespace point_on_transformed_plane_l468_468123

theorem point_on_transformed_plane (A : ℝ × ℝ × ℝ) (a b c D k : ℝ) :
  (A = (-2, 3, -3)) →
  (a = 3 ∧ b = 2 ∧ c = -1 ∧ D = -2) →
  (k = 3 / 2) →
  (a * A.1 + b * A.2 + c * A.3 + k * D = 0) :=
by
  intros hA h_planes h_k
  rw [←hA, ←h_planes.1, ←h_planes.2.1, ←h_planes.2.2.1, ←h_planes.2.2.2]
  sorry

end point_on_transformed_plane_l468_468123


namespace intersection_of_S_and_T_l468_468231

def set_S : Set Real := {x | x > -2}
def set_T : Set Real := {x | -4 ≤ x ∧ x ≤ 1}

theorem intersection_of_S_and_T :
  set_S ∩ set_T = {x | -2 < x ∧ x ≤ 1} :=
by sorry

end intersection_of_S_and_T_l468_468231


namespace optimal_selection_method_uses_golden_ratio_l468_468850

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l468_468850


namespace min_girls_in_class_l468_468915

theorem min_girls_in_class : ∃ d : ℕ, 20 - d ≤ 2 * (d + 1) ∧ d ≥ 6 := by
  sorry

end min_girls_in_class_l468_468915


namespace possible_values_of_S_l468_468042

def is_five_digit_wo_0_1 (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ ∀ d ∈ n.digits 10, d ≠ 0 ∧ d ≠ 1

theorem possible_values_of_S (A B : ℕ) (S : ℕ) (C := B - 11111)
  (hA : is_five_digit_wo_0_1 A)
  (hB : is_five_digit_wo_0_1 B)
  (hS : 1000 ≤ S ∧ S < 10000)
  (h1 : S = abs (A - B))
  (h2 : abs (A - C) = 10002) :
  S = 1109 :=
sorry

end possible_values_of_S_l468_468042


namespace hua_luogeng_optimal_selection_method_l468_468785

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l468_468785


namespace hua_luogeng_optimal_selection_method_l468_468782

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l468_468782


namespace find_length_AC_l468_468433

noncomputable def length_AC_inscribed_quadrilateral 
  (ABCD : Type)
  (is_inscribed : ABCD)
  (angle_ratio : ∀ (A B C : ℝ), A / B = 2 / 3 ∧ B / C = 3 / 4)
  (CD BC : ℝ) 
  (length_CD : CD = 15)
  (length_BC : BC = 18 * Real.sqrt 3 - 7.5) : ℝ :=
  ∃ AC : ℝ, AC = 39

theorem find_length_AC
  (ABCD : Type)
  (is_inscribed : ABCD)
  (angle_ratio : ∀ (A B C : ℝ), A / B = 2 / 3 ∧ B / C = 3 / 4)
  (CD BC : ℝ) 
  (length_CD : CD = 15)
  (length_BC : BC = 18 * Real.sqrt 3 - 7.5) 
  : ∃ (AC : ℝ), AC = 39 :=
begin
  sorry
end

end find_length_AC_l468_468433


namespace pyramid_volume_is_sqrt_20_l468_468075

noncomputable def volume_of_pyramid
  (AB BC CG : ℝ) (M : ℝ × ℝ × ℝ)
  (h_AB : AB = 4) (h_BC : BC = 2) (h_CG : CG = 3)
  (h_M : M = (4, 1, 3 / 2)) : ℝ :=
  let BE := Real.sqrt ((4 ^ 2) + (2 ^ 2)) in
  let base_area := BC * BE in
  let height := M.2.2 in
  (1 / 3) * base_area * height

theorem pyramid_volume_is_sqrt_20
  (AB BC CG : ℝ) (M : ℝ × ℝ × ℝ)
  (h_AB : AB = 4) (h_BC : BC = 2) (h_CG : CG = 3)
  (h_M : M = (4, 1, 3 / 2)) :
  volume_of_pyramid AB BC CG M h_AB h_BC h_CG h_M = Real.sqrt 20 := 
  sorry

end pyramid_volume_is_sqrt_20_l468_468075


namespace hua_luogeng_optimal_selection_l468_468556

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l468_468556


namespace optimal_selection_method_uses_golden_ratio_l468_468843

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468843


namespace person_B_lower_average_price_l468_468143

noncomputable def average_price_A (x y : ℝ) : ℝ :=
  (x + y) / 2

noncomputable def average_price_B (x y : ℝ) : ℝ :=
  2 * x * y / (x + y)

theorem person_B_lower_average_price
  (x y : ℝ)
  (h : x ≠ y) :
  average_price_A x y > average_price_B x y :=
begin
  sorry
end

end person_B_lower_average_price_l468_468143


namespace equal_length_polyhedron_l468_468434

variables {V : Type*} [inner_product_space ℝ V]
variables {a b c d : V}

def orthocentric_tetrahedron (a b c d : V) : Prop :=
  let g := (a + b + c + d) / 4 in
  dist a g = dist b g ∧ dist b g = dist c g ∧ dist c g = dist d g

theorem equal_length_polyhedron 
  (h : orthocentric_tetrahedron a b c d) :
  dist (a - b - c + d) (0 : V) = dist (a + b - c - d) (0 : V)
  ∧ dist (a + b - c - d) (0 : V) = dist (a - b + c + d) (0 : V) :=
sorry

end equal_length_polyhedron_l468_468434


namespace lightest_boy_heaviest_girl_l468_468410

theorem lightest_boy_heaviest_girl :
  ∃ (B G : ℕ), B + G < 35 ∧
  (∃ (wb : ℕ → ℝ), (∀ i, wb i > 0) ∧ (∑ i in Finset.range B, wb i) = 60 * B ∧ (∃ i, wb i = min (Finset.range B) wb)) ∧
  (∃ (wg : ℕ → ℝ), (∀ i, wg i > 0) ∧ (∑ i in Finset.range G, wg i) = 47 * G ∧ (∃ i, wg i = max (Finset.range G) wg)) ∧
  (∑ i in Finset.range B, wb i + ∑ i in Finset.range G, wg i) = 53.5 * (B + G) :=
begin
  sorry
end

end lightest_boy_heaviest_girl_l468_468410


namespace optimal_selection_method_is_golden_ratio_l468_468595

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l468_468595


namespace b_range_l468_468031

def f (x : ℝ) : ℝ := Real.log x - (1 / 4) * x + (3 / (4 * x)) - 1

def g (x b : ℝ) : ℝ := x ^ 2 - 2 * b * x + 4

theorem b_range (b : ℝ) (h : ∀ x1 ∈ Set.Ioo 0 2, ∃ x2 ∈ Set.Icc 1 2, f x1 ≥ g x2 b) : b ≥ 17 / 8 :=
sorry

end b_range_l468_468031


namespace distance_to_directrix_l468_468008

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 64 + y^2 / 28 = 1

noncomputable def left_focus : ℝ × ℝ := (-6, 0)

noncomputable def right_focus : ℝ × ℝ := (6, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_to_directrix (P : ℝ × ℝ) (hP : ellipse P.1 P.2) (hPF1 : distance P left_focus = 4) :
  distance P right_focus * 4 / 3 = 16 :=
sorry

end distance_to_directrix_l468_468008


namespace length_YW_l468_468089

/-- Given conditions -/
def is_isosceles_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  A = B ∧ a = c

def XY : ℝ := 3
def XZ : ℝ := 5
def YZ : ℝ := 5
def ZW : ℝ := 2
def XW : ℝ := XZ - ZW

/-- Proof goal -/
theorem length_YW (a b c : ℝ) (theta : ℝ) (h_isosceles : is_isosceles_triangle a b c theta theta θ) (ZW : ℝ) :
  (Z - W = 2) →
  sqrt(9 + 9 - 36 * 41 / 50) = 4.30 :=
begin
  sorry
end

end length_YW_l468_468089


namespace fido_yard_area_fraction_value_of_a_plus_b_l468_468303

-- Define the conditions of the problem
def side_length_of_square (s : ℝ) : ℝ := 2 * s
def area_of_square (s : ℝ) : ℝ := (side_length_of_square s) ^ 2
def radius_of_circle (s : ℝ) : ℝ := s
def area_of_circle (s : ℝ) : ℝ := real.pi * (radius_of_circle s) ^ 2

-- State the problem as a theorem in Lean 4
theorem fido_yard_area_fraction (s : ℝ) (h : s > 0) :
  (area_of_circle s) / (area_of_square s) = real.pi / 4 :=
by
  sorry

-- Prove the value of a + b
theorem value_of_a_plus_b (s : ℝ) (h : s > 0) :
  ∃ a b : ℕ, (area_of_circle s) / (area_of_square s) = (a * real.pi) / b ∧ a + b = 5 :=
by
  use 1, 4
  constructor
  · apply fido_yard_area_fraction s h
  · simp
  sorry

end fido_yard_area_fraction_value_of_a_plus_b_l468_468303


namespace imaginary_part_proof_l468_468468

def imaginary_unit := Complex.I

def imaginary_part_of_z (z : ℂ) : ℂ :=
  Complex.im z

theorem imaginary_part_proof (z : ℂ) (h : cexp (imaginary_unit * 2019) - 1 ≠ 0) :
  z = (imaginary_unit^2018) / (imaginary_unit^2019 - 1) →
  imaginary_part_of_z z = -1 / 2 :=
by {
  intro h1,
  sorry
}

end imaginary_part_proof_l468_468468


namespace min_girls_in_class_l468_468918

theorem min_girls_in_class : ∃ d : ℕ, 20 - d ≤ 2 * (d + 1) ∧ d ≥ 6 := by
  sorry

end min_girls_in_class_l468_468918


namespace polyhedron_faces_equal_product_of_consecutive_integers_l468_468511

-- Define the concepts of a convex polyhedron and parallelogram faces
structure Polyhedron where
  faces : Finset (Finset Point)
  convex : ConvexHull faces = faces

def facesAreParallelograms (P : Polyhedron) : Prop :=
  ∀ f ∈ P.faces, isParallelogram f

-- Define the concept of distinct edge directions
def numEdgeDirections (P : Polyhedron) : ℕ := edges.countDistinctDirections

-- Define the main theorem statement
theorem polyhedron_faces_equal_product_of_consecutive_integers (P : Polyhedron) 
  (h_convex : P.convex)
  (h_parallelograms : facesAreParallelograms P)
  (k : ℕ := numEdgeDirections P) :
  ∃ n : ℕ, n = k * (k - 1) :=
by
  sorry

end polyhedron_faces_equal_product_of_consecutive_integers_l468_468511


namespace minimum_girls_in_class_l468_468928

theorem minimum_girls_in_class (n : ℕ) (d : ℕ) (boys : List (List ℕ)) 
  (h_students : n = 20)
  (h_boys : boys.length = n - d)
  (h_unique : ∀ i j k : ℕ, i < boys.length → j < boys.length → k < boys.length → i ≠ j → j ≠ k → i ≠ k → 
    (boys.nthLe i (by linarith)).length ≠ (boys.nthLe j (by linarith)).length ∨ 
    (boys.nthLe j (by linarith)).length ≠ (boys.nthLe k (by linarith)).length ∨ 
    (boys.nthLe k (by linarith)).length ≠ (boys.nthLe i (by linarith)).length) :
  d = 6 := 
begin
  sorry
end

end minimum_girls_in_class_l468_468928


namespace optimal_selection_uses_golden_ratio_l468_468656

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l468_468656


namespace max_value_of_f_on_interval_l468_468181

noncomputable def f (x : ℝ) := x^3 - 3 * x^2 + 2

theorem max_value_of_f_on_interval : 
  ∃ x ∈ set.Icc (-1 : ℝ) 1, ∀ y ∈ set.Icc (-1 : ℝ) 1, f y ≤ f x ∧ f x = 2 :=
by
  sorry

end max_value_of_f_on_interval_l468_468181


namespace optimal_selection_method_is_golden_ratio_l468_468590

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l468_468590


namespace optimal_selection_method_uses_golden_ratio_l468_468626

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l468_468626


namespace part1_part2_l468_468026

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * x + 6

-- Part (I)
theorem part1 (a : ℝ) (h : a = 5) : ∀ x : ℝ, f x 5 < 0 ↔ -3 < x ∧ x < -2 := by
  sorry

-- Part (II)
theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > 0) ↔ -2 * Real.sqrt 6 < a ∧ a < 2 * Real.sqrt 6 := by
  sorry

end part1_part2_l468_468026


namespace optimal_selection_uses_golden_ratio_l468_468636

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l468_468636


namespace math_problems_equivalence_l468_468442

noncomputable def problem_a (n : ℕ) (hn : n = 3) : Prop := 
  let points := [(x, y) | x y ∈ {1, 2, 3}]
  let p_y_eq_3_given_x_eq_2 := ((1 : ℚ) / 9) / ((3 : ℚ) / 9)
  p_y_eq_3_given_x_eq_2 = 1 / 3

noncomputable def problem_b (n : ℕ) (hn : n = 4) : Prop := 
  let points := [(x, y) | x y ∈ {1, 2, 3, 4}]
  let p_x_plus_y_eq_8 := 2 / 16
  p_x_plus_y_eq_8 = 1 / 16

noncomputable def problem_c (n : ℕ) (hn : n = 4) : Prop := 
  let points := [(x, y) | x y ∈ {1, 2, 3, 4}]
  let mean_y := (2 * 1 + 3 * 2 + 4 * 3 + 5 * 4 + 6 * 3 + 7 * 2 + 8 * 1) / 16
  mean_y = 5

noncomputable def problem_d (k : ℕ) (hk : k ≥ 2) : Prop := 
  let points := [(x, y) | x y ∈ {1, 2, ..., k}]
  let p_x_eq_k_and_y_eq_2k := 1 / (k * k)
  p_x_eq_k_and_y_eq_2k = 1 / k^2

theorem math_problems_equivalence : 
  problem_a 3 rfl ∧ ¬problem_b 4 rfl ∧ problem_c 4 rfl ∧ ∀ k (hk : k ≥ 2), problem_d k (nat.le_of_succ_le_succ hk) :=
by
  sorry

end math_problems_equivalence_l468_468442


namespace find_y_l468_468387

theorem find_y (y : ℕ) : (1 / 8) * 2^36 = 8^y → y = 11 := by
  sorry

end find_y_l468_468387


namespace hua_luogeng_optimal_selection_method_l468_468778

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l468_468778


namespace optimal_selection_method_uses_golden_ratio_l468_468630

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l468_468630


namespace optimal_selection_golden_ratio_l468_468703

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l468_468703


namespace probability_of_white_balls_from_both_boxes_l468_468435

theorem probability_of_white_balls_from_both_boxes :
  let P_white_A := 3 / (3 + 2)
  let P_white_B := 2 / (2 + 3)
  P_white_A * P_white_B = 6 / 25 :=
by
  sorry

end probability_of_white_balls_from_both_boxes_l468_468435


namespace raisins_water_percentage_l468_468045

variable (weight_grapes : ℝ) (water_content_grapes : ℝ) (weight_raisins : ℝ)

-- Definitions based on conditions
def solid_content_grapes (weight_grapes : ℝ) (water_content_grapes : ℝ) : ℝ :=
  (1 - water_content_grapes) * weight_grapes

def water_content_raisins (weight_raisins solid_content_grapes : ℝ) : ℝ :=
  weight_raisins - solid_content_grapes

def percentage_water_in_raisins (water_content_raisins weight_raisins : ℝ) : ℝ :=
  (water_content_raisins / weight_raisins) * 100

-- Lean 4 statement for the problem
theorem raisins_water_percentage : 
  weight_grapes = 50 → 
  water_content_grapes = 0.92 → 
  weight_raisins = 5 →
  percentage_water_in_raisins (water_content_raisins 5 (solid_content_grapes 50 0.92)) 5 = 20 := 
by
  intros h1 h2 h3
  sorry

end raisins_water_percentage_l468_468045


namespace profit_percentage_with_discount_is_26_l468_468256

noncomputable def cost_price : ℝ := 100
noncomputable def profit_percentage_without_discount : ℝ := 31.25
noncomputable def discount_percentage : ℝ := 4

noncomputable def selling_price_without_discount : ℝ :=
  cost_price * (1 + profit_percentage_without_discount / 100)

noncomputable def discount : ℝ := 
  discount_percentage / 100 * selling_price_without_discount

noncomputable def selling_price_with_discount : ℝ :=
  selling_price_without_discount - discount

noncomputable def profit_with_discount : ℝ := 
  selling_price_with_discount - cost_price

noncomputable def profit_percentage_with_discount : ℝ := 
  (profit_with_discount / cost_price) * 100

theorem profit_percentage_with_discount_is_26 :
  profit_percentage_with_discount = 26 := by 
  sorry

end profit_percentage_with_discount_is_26_l468_468256


namespace problem_1_problem_2_l468_468025

theorem problem_1
  (f : ℝ → ℝ)
  (f_def : ∀ x, f x = (1 / 3) * x^3 - 2 * x^2 + 3 * x)
  (tangent_line : ∀ x y, 3 * x + 3 * y - 8 = 0)
  (tangent_point : ∀ x, x = 2 → f x = (1 / 3) * 2^3 - 2 * 2^2 + 3 * 2) :
  ∃ (a b : ℝ), (a = 2) ∧ (b = 3) ∧ f = (λ x, (1 / 3) * x^3 - a * x^2 + b * x) ∧
  (f 2 = (1 / 3) * 2^3 - 2 * 2^2 + 3 * 2) ∧
  (f' 2 = -1) := sorry

theorem problem_2
  (f : ℝ → ℝ)
  (f_def : ∀ x, f x = (1 / 3) * x^3 - 2 * x^2 + 3 * x)
  (interval : set.Icc (-2 : ℝ) 2)
  (max_value : ∀ x ∈ interval, f x ≤ (4 / 3))
  (min_value : ∀ x ∈ interval, f x ≥ (-50 / 3)) :
  (∀ x, (x ∈ interval → f x ≤ (4 / 3))) ∧ (∀ x, (x ∈ interval → f x ≥ (-50 / 3))) := sorry

end problem_1_problem_2_l468_468025


namespace conjugate_z_l468_468003

theorem conjugate_z (z : ℂ) (h : (1 - complex.I) * z = 2 * complex.I) : complex.conj z = -1 - complex.I :=
sorry

end conjugate_z_l468_468003


namespace vanessa_weeks_to_wait_l468_468950

theorem vanessa_weeks_to_wait
  (dress_cost savings : ℕ)
  (weekly_allowance weekly_expense : ℕ)
  (h₀ : dress_cost = 80)
  (h₁ : savings = 20)
  (h₂ : weekly_allowance = 30)
  (h₃ : weekly_expense = 10) :
  let net_savings_per_week := weekly_allowance - weekly_expense,
      additional_amount_needed := dress_cost - savings in
  additional_amount_needed / net_savings_per_week = 3 :=
by
  sorry

end vanessa_weeks_to_wait_l468_468950


namespace optimal_selection_method_uses_golden_ratio_l468_468628

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l468_468628


namespace greatest_integer_not_exceeding_100x_l468_468479

noncomputable def x : ℝ :=
  (∑ n in Finset.range 11, (n + 1) * Real.cos (Float.pi / 180 * (n + 1))) /
  (∑ n in Finset.range 11, (n + 1) * Real.sin (Float.pi / 180 * (n + 1)))

theorem greatest_integer_not_exceeding_100x :
  ∀ x : ℝ,
  x = ( (∑ n in Finset.range 11, (n + 1) * Real.cos (Float.pi / 180 * (n + 1))) /
        (∑ n in Finset.range 11, (n + 1) * Real.sin (Float.pi / 180 * (n + 1))) ) ->
  Real.floor (100 * x) = --complete this part with the evaluated integer value
  sorry

end greatest_integer_not_exceeding_100x_l468_468479


namespace optimalSelectionUsesGoldenRatio_l468_468713

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l468_468713


namespace optimal_selection_golden_ratio_l468_468691

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l468_468691


namespace hua_luogeng_optimal_selection_l468_468544

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l468_468544


namespace optimal_selection_golden_ratio_l468_468697

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l468_468697


namespace valentines_day_is_saturday_l468_468054

theorem valentines_day_is_saturday (is_leap_year : Prop) (day_13_is_friday : Prop) : 
∀ (days_in_week : ℕ) (february_13 : ℕ), days_in_week = 7 → february_13 = 5 → 
(february_13 + 1) % days_in_week = 6 :=
by
  -- is_leap_year condition is not directly used in the proof; consider if necessary
  -- february_13 is given as 5 (Friday)
  intro days_in_week february_13 h_week h_feb_13
  rw [h_week, h_feb_13]
  calc
    (5 + 1) % 7 = 6    : by norm_num
  sorry

end valentines_day_is_saturday_l468_468054


namespace min_girls_in_class_l468_468921

theorem min_girls_in_class : ∃ d : ℕ, 20 - d ≤ 2 * (d + 1) ∧ d ≥ 6 := by
  sorry

end min_girls_in_class_l468_468921


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468761

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468761


namespace mark_eggs_supply_l468_468487

theorem mark_eggs_supply 
  (dozens_per_day : ℕ)
  (eggs_per_dozen : ℕ)
  (extra_eggs : ℕ)
  (days_in_week : ℕ)
  (H1 : dozens_per_day = 5)
  (H2 : eggs_per_dozen = 12)
  (H3 : extra_eggs = 30)
  (H4 : days_in_week = 7) :
  (dozens_per_day * eggs_per_dozen + extra_eggs) * days_in_week = 630 :=
by
  sorry

end mark_eggs_supply_l468_468487


namespace complex_number_solution_l468_468327

theorem complex_number_solution (z : ℂ) (h : z * (2 - complex.i) = 3 + complex.i) : z = 1 + complex.i := 
sorry

end complex_number_solution_l468_468327


namespace jisha_distance_first_day_l468_468095

theorem jisha_distance_first_day (h : ℕ) :
  let d := 3 * h in
  d + 4 * (h - 1) + 4 * h = 62 →
  d = 18 :=
by {
  intro hd_eq,
  sorry
}

end jisha_distance_first_day_l468_468095


namespace minimum_group_members_l468_468257

theorem minimum_group_members (people: ℕ)
    (dishes: Finset ℕ)
    (price: ℕ → ℕ)
    (total_people: people = 92)
    (distinct_dishes: dishes = {1, 2, 3, 4, 5, 6, 7, 8, 9})
    (total_price: ∀ (d : Finset ℕ), d ⊆ dishes → (∑ x in d, price x) = 10 → d.card ≤ 9)
    : (∃ S : Finset (Finset ℕ), S.card = 9 ∧ ∀ s ∈ S, (∃ n, 10 * n + 2 ≤ people ∧ ∀ p ∈ S, p = s)) := 
begin 
  sorry
end

end minimum_group_members_l468_468257


namespace annie_initial_money_l468_468273

theorem annie_initial_money (h_cost : ℕ) (m_cost : ℕ) (h_count : ℕ) (m_count : ℕ) (remaining_money : ℕ) :
  h_cost = 4 → m_cost = 5 → h_count = 8 → m_count = 6 → remaining_money = 70 →
  h_cost * h_count + m_cost * m_count + remaining_money = 132 :=
by
  intros h_cost_def m_cost_def h_count_def m_count_def remaining_money_def
  rw [h_cost_def, m_cost_def, h_count_def, m_count_def, remaining_money_def]
  sorry

end annie_initial_money_l468_468273


namespace distance_traveled_is_correct_l468_468061

noncomputable def speed_in_mph : ℝ := 23.863636363636363
noncomputable def seconds : ℝ := 2

-- constants for conversion
def miles_to_feet : ℝ := 5280
def hours_to_seconds : ℝ := 3600

-- speed in feet per second
noncomputable def speed_in_fps : ℝ := speed_in_mph * miles_to_feet / hours_to_seconds

-- distance traveled
noncomputable def distance : ℝ := speed_in_fps * seconds

theorem distance_traveled_is_correct : distance = 69.68 := by
  sorry

end distance_traveled_is_correct_l468_468061


namespace knicks_equal_knocks_l468_468051

theorem knicks_equal_knocks :
  (8 : ℝ) / (3 : ℝ) * (5 : ℝ) / (6 : ℝ) * (30 : ℝ) = 66 + 2 / 3 :=
by
  sorry

end knicks_equal_knocks_l468_468051


namespace optimal_selection_method_uses_golden_ratio_l468_468856

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l468_468856


namespace max_marbles_on_8x8_board_l468_468439

theorem max_marbles_on_8x8_board : 
  ∃ (max_marbles : ℕ), (max_marbles ≤ 64) ∧ 
  (∀ (marbles : (Fin 8 × Fin 8) → Bool), 
    (∀ x y, marbles (x, y) = true → 
    ∃ (free_neighbors : Fin 4 → (Fin 8 × Fin 8)),
      (∀ n, free_neighbors n ∈ 
            [{(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)} ∩ {(a, b) ∈ (Fin 8 × Fin 8) | marbles (a, b) = false}])) →
     marbles.card ≤ max_marbles) ∧ max_marbles = 36 := 
begin
  sorry
end

end max_marbles_on_8x8_board_l468_468439


namespace integral_values_of_a_l468_468466

theorem integral_values_of_a:
  let a : ℤ := λ a, 1 ≤ a ∧ a ≤ 100 in
  let d1 := λ a, a^2 + 3^a + a * 3^((a + 1) / 2) in
  let d2 := λ a, a^2 + 3^a - a * 3^((a + 1) / 2) in
  let prod := λ a, d1 a * d2 a in
  (∃ a : ℤ, d1 a * d2 a % 3 = 0) → ∃ n, n = 34 ∧ ∀ x : ℤ, (1 ≤ x ∧ x ≤ 100) → x % 3 = 0 → n = 34 := sorry

end integral_values_of_a_l468_468466


namespace symmetric_distance_equal_l468_468187

theorem symmetric_distance_equal
  (T₁ T₂ : Triangle)
  (homothetic : IsHomothetic T₁ T₂)
  (ω : Fin 6 → Circle)
  (tangent1 : ExternallyTangent (ω 0) (ω 2) (ω 4) (center := O₁))
  (tangent2 : ExternallyTangent (ω 1) (ω 3) (ω 5) (center := O₂))
  (tangent3 : InternallyTangent (ω 0) (ω 2) (ω 4) (center := O₃))
  (tangent4 : InternallyTangent (ω 1) (ω 3) (ω 5) (center := O₄)) :
  distance O₁ O₃ = distance O₂ O₄ :=
by
  sorry

end symmetric_distance_equal_l468_468187


namespace three_digit_integers_count_l468_468019

theorem three_digit_integers_count : 
  ∃ (n : ℕ), n = 24 ∧
  (∃ (digits : Finset ℕ), digits = {2, 4, 7, 9} ∧
  (∀ a b c : ℕ, a ∈ digits → b ∈ digits → c ∈ digits → a ≠ b → b ≠ c → a ≠ c → 
  100 * a + 10 * b + c ∈ {y | 100 ≤ y ∧ y < 1000} → 4 * 3 * 2 = 24)) :=
by
  sorry

end three_digit_integers_count_l468_468019


namespace optimal_selection_uses_golden_ratio_l468_468641

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l468_468641


namespace ratio_of_middle_angle_l468_468189

theorem ratio_of_middle_angle (A B C : ℝ) 
  (h1 : A + B + C = 180)
  (h2 : C = 5 * A)
  (h3 : A = 20) :
  B / A = 3 :=
by
  sorry

end ratio_of_middle_angle_l468_468189


namespace optimal_selection_uses_golden_ratio_l468_468830

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l468_468830


namespace probability_male_or_younger_than_70_l468_468222

def village_population : ℕ := 2700
def over_70_population : ℕ := 900
def female_population : ℕ := 1200
def female_younger_than_70_percentage : ℝ := 0.60

theorem probability_male_or_younger_than_70 :
  (let females_younger_than_70 := female_younger_than_70_percentage * (female_population : ℝ),
       females_over_70 := female_population - (females_younger_than_70 : ℕ),
       males_over_70 := over_70_population - females_over_70,
       males_population := village_population - female_population,
       total_male_or_younger_70 := males_population + (females_younger_than_70 : ℕ) in
    (total_male_or_younger_70 : ℝ) / (village_population : ℝ) = 0.8222) := sorry

end probability_male_or_younger_than_70_l468_468222


namespace problem1_problem2_l468_468342

-- Define the points
def A : (ℝ × ℝ) := (-2, 4)
def B : (ℝ × ℝ) := (3, -1)
def C : (ℝ × ℝ) := (-3, -4)

-- Define the vectors
def vecAB : (ℝ × ℝ) := (5, -5)
def vecBC : (ℝ × ℝ) := (-6, -3)
def vecCA : (ℝ × ℝ) := (1, 8)

-- Proof Problem 1: Prove 3overrightarrow(a) + overrightarrow(b) = (9, -18)
theorem problem1 : 3 • vecAB + vecBC = (9, -18) := sorry

-- Proof Problem 2: Prove the values of m and n
def m : ℝ := -1
def n : ℝ := -1

theorem problem2 : vecAB = m • vecBC + n • vecCA := by
  simp [m, n, vecAB, vecBC, vecCA]; split; norm_num; rfl

end problem1_problem2_l468_468342


namespace optimal_selection_golden_ratio_l468_468700

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l468_468700


namespace annie_initial_money_l468_468274

theorem annie_initial_money (h_cost : ℕ) (m_cost : ℕ) (h_count : ℕ) (m_count : ℕ) (remaining_money : ℕ) :
  h_cost = 4 → m_cost = 5 → h_count = 8 → m_count = 6 → remaining_money = 70 →
  h_cost * h_count + m_cost * m_count + remaining_money = 132 :=
by
  intros h_cost_def m_cost_def h_count_def m_count_def remaining_money_def
  rw [h_cost_def, m_cost_def, h_count_def, m_count_def, remaining_money_def]
  sorry

end annie_initial_money_l468_468274


namespace expression_value_l468_468960

theorem expression_value : 6^3 - 4 * 6^2 + 4 * 6 - 1 = 95 :=
by
  sorry

end expression_value_l468_468960


namespace optimal_selection_method_is_golden_ratio_l468_468586

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l468_468586


namespace line_equation_l468_468170

theorem line_equation (m : ℝ) (b : ℝ) (x y : ℝ) (π : ℝ) 
  (hθ : π / 3 = arctan m ∧ b = 2 ∧ y = m * x + b) :
  √3 * x - y + 2 = 0 :=
sorry

end line_equation_l468_468170


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468765

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468765


namespace find_x_val_l468_468530

theorem find_x_val (x y : ℝ) (c : ℝ) (h1 : y = 1 → x = 8) (h2 : ∀ y, x * y^3 = c) : 
  (∀ (y : ℝ), y = 2 → x = 1) :=
by
  sorry

end find_x_val_l468_468530


namespace hua_luogeng_optimal_selection_method_l468_468793

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l468_468793


namespace range_of_g_l468_468291

noncomputable def g (x : ℝ) : ℝ := (Real.arcsin x) + (Real.arccos x) + Real.exp (Real.atan x)

theorem range_of_g :
  set.range (fun x => g x) = set.Icc (
  (Real.pi / 2) + Real.exp (- Real.pi / 4)
  ) (
  (Real.pi / 2) + Real.exp (Real.pi / 4)
  ) :=
by
  sorry

end range_of_g_l468_468291


namespace line_OP_passes_through_M_l468_468879

-- Define the points in the triangle and their properties
variables {α : Type*} [metric_space α] [normed_add_comm_group α] [normed_space ℝ α]

-- Assume the existence of points A, B, and C forming triangle ABC
variables (A B C O H P M : α)

-- Define the conditions: O is the circumcenter, H is the orthocenter
-- P is the foot of the perpendicular from O to the altitude, M is the midpoint of AH
def is_circumcenter (O A B C : α) : Prop := sorry
def is_orthocenter (H A B C : α) : Prop := sorry
def is_foot_of_perpendicular (P O A B C : α) : Prop := sorry
def is_midpoint (M A H : α) : Prop := sorry

-- The problem statement
theorem line_OP_passes_through_M :
  is_circumcenter O A B C ∧ is_orthocenter H A B C ∧ is_foot_of_perpendicular P O A B C ∧ is_midpoint M A H →
  collinear [O, P, M] :=
sorry

end line_OP_passes_through_M_l468_468879


namespace optimal_selection_uses_golden_ratio_l468_468661

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l468_468661


namespace vol_pyramid_216_l468_468076

noncomputable def volume_of_pyramid (S A B C D K : Type) 
  [RegularTriPyramid S A B C D K] 
  (h1 : Ratio (SK : KD) = 1 : 2) 
  (h2 : DihedralAngle (Base ∡ LateralFace) = π / 6)
  (h3 : Distance K (LateralEdge S A) = 4 / sqrt(13)) 
  : Real :=
  216

variables {S A B C D K : Type}

-- Given evidence for each of the conditions
axiom h1 : Ratio (SK : KD) = 1 : 2
axiom h2 : DihedralAngle (Base ∡ LateralFace) = π / 6
axiom h3 : Distance K (LateralEdge S A) = 4 / sqrt(13)

-- Statement of the theorem
theorem vol_pyramid_216 : volume_of_pyramid S A B C D K h1 h2 h3 = 216 := 
  sorry

end vol_pyramid_216_l468_468076


namespace optimal_selection_uses_golden_ratio_l468_468647

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l468_468647


namespace log_expression_to_base_4_l468_468286

theorem log_expression_to_base_4 :
  ∃ (x : ℝ), x = log 4 (81 * (16:ℝ)^(1/3) * (16:ℝ)^(1/4)) ∧ x ≈ 4.3 := by
  sorry

end log_expression_to_base_4_l468_468286


namespace modulus_power_six_l468_468284

theorem modulus_power_six (i : ℂ) (h : |1 + i| = Real.sqrt 2) : |(1 + i)^6| = 8 := by
  sorry

end modulus_power_six_l468_468284


namespace optimal_selection_method_uses_golden_ratio_l468_468633

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l468_468633


namespace minimum_girls_in_class_l468_468926

theorem minimum_girls_in_class (n : ℕ) (d : ℕ) (boys : List (List ℕ)) 
  (h_students : n = 20)
  (h_boys : boys.length = n - d)
  (h_unique : ∀ i j k : ℕ, i < boys.length → j < boys.length → k < boys.length → i ≠ j → j ≠ k → i ≠ k → 
    (boys.nthLe i (by linarith)).length ≠ (boys.nthLe j (by linarith)).length ∨ 
    (boys.nthLe j (by linarith)).length ≠ (boys.nthLe k (by linarith)).length ∨ 
    (boys.nthLe k (by linarith)).length ≠ (boys.nthLe i (by linarith)).length) :
  d = 6 := 
begin
  sorry
end

end minimum_girls_in_class_l468_468926


namespace cone_sinθ_value_l468_468329

noncomputable def cone_sinθ (r : ℝ) (A : ℝ) (θ : ℝ) (l : ℝ) : Prop :=
  r = 5 ∧ A = 65 * Real.pi ∧ l = 13 ∧ θ = Real.arcsin (r / l) 

theorem cone_sinθ_value :
  cone_sinθ 5 (65 * Real.pi) (Real.arcsin (5 / 13)) 13 :=
by
  rw cone_sinθ
  simp
  sorry -- Proof steps would go here

end cone_sinθ_value_l468_468329


namespace house_number_and_total_houses_l468_468060

noncomputable def sum_first_n_odds(n: ℕ) : ℕ := n^2

theorem house_number_and_total_houses(x n: ℕ) (h_odd_numbers: ∀ i: ℕ, i ∈ (finset.range n).filter (λ x, x % 2 = 1) ↔ (2*i - 1) = x)
  (h_sum_left_right : sum_first_n_odds (x-1)/2 = sum_first_n_odds (n-x)/2)
  : n = 1823 ∧ x = 912 := 
by
  sorry

end house_number_and_total_houses_l468_468060


namespace hua_luogeng_optimal_selection_l468_468557

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l468_468557


namespace a_is_perfect_square_l468_468038

-- Define the sequence cₙ based on initial conditions and recurrence relations
def c : ℕ → ℤ
| 0       := 1
| 1       := 0
| 2       := 2005
| (n + 3) := -3 * c (n + 1) - 4 * c n + 2008

-- Define the sequence aₙ
def a (n : ℕ) : ℤ := 5 * (c (n + 2) - c n) * (502 - c (n - 1) - c (n - 2)) + 4^n * 2004 * 501

-- State the theorem to determine if aₙ is a perfect square for n ≥ 3.
theorem a_is_perfect_square (n : ℕ) (h : 2 < n) : ∃ k : ℤ, a n = k^2 := 
sorry

end a_is_perfect_square_l468_468038


namespace optimal_selection_method_uses_golden_ratio_l468_468855

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l468_468855


namespace minimal_t_l468_468121

noncomputable def f (n : ℕ) : ℕ := (Real.log n / Real.log 2).floor + 1

def S (n : ℕ) : Finset ℕ := Finset.range (n + 1) \ {0}

def F (S : Finset ℕ) (t : ℕ) : Type :=
  {F : Finset (Finset ℕ) // F.card = t ∧ 
                           (∀ x y ∈ S, x ≠ y → ∃ A ∈ F, (A ∩ {x, y}).card = 1) ∧ 
                           (∀ A ∈ F, A ⊆ S) ∧ 
                           (S ⊆ Finset.bUnion F id)}

theorem minimal_t (n : ℕ) (h : 2 ≤ n) : 
  ∃ t, ∀ (F, ht), @F (S n) t → t = f n := 
sorry

end minimal_t_l468_468121


namespace optimal_selection_method_uses_golden_ratio_l468_468848

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468848


namespace limit_of_f_at_0_l468_468985

noncomputable def f (x : ℝ) : ℝ := (2 * (Real.exp (π * x) - 1)) / (3 * ((1 + x)^(1/3) - 1))

theorem limit_of_f_at_0 : Filter.Tendsto f (Filter.nhds 0) (Filter.nhds (2 * π)) :=
by
  sorry

end limit_of_f_at_0_l468_468985


namespace Skittles_distribution_l468_468283

/-
Bridget's initial Skittles: 4
Henry's initial Skittles: 4
Alice's initial Skittles: 3
Charlie's initial Skittles: 7
Henry gives all Skittles to Bridget
Charlie gives 3 Skittles to Alice (half of 7 rounded down)
-/
noncomputable def Bridget_initial : ℕ := 4
noncomputable def Henry_initial : ℕ := 4
noncomputable def Alice_initial : ℕ := 3
noncomputable def Charlie_initial : ℕ := 7
noncomputable def Henry_gives_to_Bridget : ℕ := Henry_initial
noncomputable def Charlie_gives_to_Alice : ℕ := 3

theorem Skittles_distribution :
  Bridget_initial + Henry_gives_to_Bridget = 8 ∧
  Henry_initial - Henry_gives_to_Bridget = 0 ∧
  Alice_initial + Charlie_gives_to_Alice = 6 ∧
  Charlie_initial - Charlie_gives_to_Alice = 4 :=
by
  simp [Bridget_initial, Henry_initial, Alice_initial, Charlie_initial,
        Henry_gives_to_Bridget, Charlie_gives_to_Alice]
  tautology

end Skittles_distribution_l468_468283


namespace probability_of_D_in_interval_l468_468482

-- Definition of function and its domain
def f (x : ℝ) := real.sqrt (6 + x - x^2)
def D := {x : ℝ | -2 ≤ x ∧ x ≤ 3}

-- Given interval
def interval := set.Icc (-4 : ℝ) 5

-- Proof statement
theorem probability_of_D_in_interval :
  let length_D := 3 - (-2) in
  let length_interval := 5 - (-4) in
  (length_D / length_interval) = (5 / 9) :=
by
  sorry

end probability_of_D_in_interval_l468_468482


namespace E_cannot_win_all_matches_l468_468315

-- Let's define the participants and their win/loss conditions.
variables {A B C D E : Type}

-- Given conditions
def equal_wins_losses (a : A) (l w : A → ℕ) := l a = w a
def more_wins_than_losses (b : B) (l w : B → ℕ) := w b = l b + 4
def more_losses_than_wins (c d : C × D) (l w : C × D → ℕ) := 
  l c = w c + 5 ∧ l d = w d + 5

-- Hypothesis stating participant E's win condition
def E_wins_all (e : E) (w : E → ℕ) : Prop := w e > 0

-- The statement to prove
theorem E_cannot_win_all_matches (l : A → ℕ) (w : A → ℕ) 
  (l' : B → ℕ) (w' : B → ℕ) 
  (l'' : C × D → ℕ) (w'' : C × D → ℕ)
  (lE : E → ℕ) (wE : E → ℕ) :
  equal_wins_losses A l w → 
  more_wins_than_losses B l' w' → 
  more_losses_than_wins (C, D) l'' w'' → 
  E_wins_all E wE → 
  false := 
sorry

end E_cannot_win_all_matches_l468_468315


namespace preimage_of_mapping_l468_468034

def f (a b : ℝ) : ℝ × ℝ := (a + 2 * b, 2 * a - b)

theorem preimage_of_mapping : ∃ (a b : ℝ), f a b = (3, 1) ∧ (a, b) = (1, 1) :=
by
  sorry

end preimage_of_mapping_l468_468034


namespace new_circle_equation_l468_468171

-- Define the initial conditions
def initial_circle_equation (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0
def radius_of_new_circle : ℝ := 2

-- Define the target equation of the circle
def target_circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- The theorem statement
theorem new_circle_equation (x y : ℝ) :
  initial_circle_equation x y → target_circle_equation x y :=
sorry

end new_circle_equation_l468_468171


namespace nested_square_root_identity_l468_468032

theorem nested_square_root_identity (n : ℕ) (hn : n > 0) :
    √(1 + n * √(1 + (n+1) * √(1 + (n+2) * √(1 + (n+3) * √(1 + ...)))) = n + 1 :=
sorry

end nested_square_root_identity_l468_468032


namespace tangent_at_point_D_l468_468079

variables 
  (A B C D N T : Type) [hc : locale.char_zero A]
  [hb : is_finite B] 

-- Definitions based on conditions
def parallelogram (ABCD : Type) : Prop :=
  ∃ (A B C D :  AB ^ 2 =  B ^ 2 
  
variable 
  (parallelogram_ABCD : parallelogram (A := A) (B := B) (C := C) (D := D) ) 

axiom Hacute : α < pi/2 -- angle A is acute
  
def CN_eq_AB : CN = AB := sorry 

def tangent_at_D : Prop := tangent circumcircle(CBN) line(AD) point (D) 

theorem tangent_at_point_D 
  (parallelogram_ABCD : parallelogram ABCD)
  (angleA_acute : α < π / 2)
  (pointN_on_AB : point N (A B) )
  (circumcircle_tangent: tangent circumcircle(CBN) line (AD))
: tangent circumcircle(CBN) line (AD) := sorry 

end tangent_at_point_D_l468_468079


namespace Taylor_paints_in_12_hours_l468_468534

-- Define the given conditions
def Jennifer_time : ℝ := 10
def together_time : ℝ := 5.45454545455

-- Define the work rates
def Jennifer_rate := 1 / Jennifer_time
def together_rate := 1 / together_time

-- Define Taylor's time as a variable
variable (T : ℝ)
def Taylor_rate := 1 / T

-- State the problem to be proven
theorem Taylor_paints_in_12_hours (h : Taylor_rate + Jennifer_rate = together_rate) :
  T = 12 :=
by
  sorry

end Taylor_paints_in_12_hours_l468_468534


namespace hua_luogeng_optimal_selection_method_l468_468731

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l468_468731


namespace ratio_sum_not_divisor_of_total_l468_468936

theorem ratio_sum_not_divisor_of_total (total students : ℕ) (ratio1 ratio2 : ℕ) (hratio: ratio1 + ratio2 = 11) : 
  ¬ (total % (ratio1 + ratio2) = 0) :=
by
  have htotal : total = 120 := rfl
  have h1 : ratio1 + ratio2 = 11 := hratio
  have hdivisor : ¬ (120 % 11 = 0) := sorry
  exact hdivisor

end ratio_sum_not_divisor_of_total_l468_468936


namespace packs_sold_by_Robyn_l468_468521

theorem packs_sold_by_Robyn (total_packs : ℕ) (lucy_packs : ℕ) (robyn_packs : ℕ) 
  (h1 : total_packs = 98) (h2 : lucy_packs = 43) (h3 : robyn_packs = total_packs - lucy_packs) :
  robyn_packs = 55 :=
by
  rw [h1, h2] at h3
  exact h3

end packs_sold_by_Robyn_l468_468521


namespace DE_intersects_parabola_once_area_ratio_l468_468278

structure Point :=
(x : ℝ)
(y : ℝ)

def parabola (p : Point) : Prop :=
  p.y^2 = 2 * p.x

def symmetric_points (A A' : Point) : Prop :=
  A.x = -A'.x ∧ A.y = 0 ∧ A'.y = 0

def perpendicular_line (A' : Point) (B C : Point) : Prop :=
  B ∈ {p : Point | p.x = A'.x} ∧ C ∈ {p : Point | p.x = A'.x} ∧ parabola B ∧ parabola C

def point_on_segment (D A B : Point) : Prop :=
  ∃ λ : ℝ, 0 ≤ λ ∧ λ ≤ 1 ∧ D.x = A.x + λ * (B.x - A.x) ∧ D.y = A.y + λ * (B.y - A.y)

def point_on_segment_ratio (E C A : Point) (D A B : Point) : Prop :=
  ∃ λ : ℝ, 0 ≤ λ ∧ λ ≤ 1 ∧ (E.x = C.x + λ * (A.x - C.x) ∧ E.y = C.y + λ * (A.y - C.y)) ∧
           (D.x = A.x + λ * (B.x - A.x) ∧ D.y = A.y + λ * (B.y - A.y))

theorem DE_intersects_parabola_once (A A' B C D E : Point) (h1 : symmetric_points A A')
  (h2 : perpendicular_line A' B C) (h3 : point_on_segment D A B) 
  (h4 : point_on_segment_ratio E C A D A B) : 
  ∃ F : Point, F ∈ {p : Point | parabola p} ∧ (∃! g : ℝ, p ∈ {p : Point | line DE E p}) :=
sorry

theorem area_ratio (A A' B C D E F : Point) (h1 : symmetric_points A A')
  (h2 : perpendicular_line A' B C) (h3 : point_on_segment D A B) 
  (h4 : point_on_segment_ratio E C A D A B) (h5 : F ∈ {p : Point | parabola p})
  (h6 : ∃! g : ℝ, p ∈ {p : Point | line DE E p}) :
  ∃ S1 S2 : ℝ, S1 = area B C F ∧ S2 = area A D E ∧ S1 / S2 = 2 :=
sorry

end DE_intersects_parabola_once_area_ratio_l468_468278


namespace find_value_of_a2_b2_c2_l468_468102

variable {a b c : ℝ}

theorem find_value_of_a2_b2_c2
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : a + b + c = 0)
  (h3 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 5 := 
sorry

end find_value_of_a2_b2_c2_l468_468102


namespace prove_general_term_formula_and_sum_R_n_l468_468129

def a_sequence (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → S_n a_n n = 4 * S_n a_n 2 ∧ a_2n = 2 * a_n + 1

def general_term_formula (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a_n = 2 * n - 1

def b_sequence (T_n : ℕ → ℝ) (a_n : ℕ → ℝ) (lambda : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → T_n n + (a_n n + 1) / (2^n) = lambda

def c_sequence (b_n : ℕ → ℝ) (c_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → c_n n = b_n (2 * n)

def sum_R_n (R_n : ℕ → ℝ) (c_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → R_n n = (1 / 9) * (4 - (3 * n + 1) / (4^(n-1)))

theorem prove_general_term_formula_and_sum_R_n (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) (c_n : ℕ → ℝ) (R_n : ℕ → ℝ) (T_n : ℕ → ℝ) (lambda : ℝ):
  a_sequence a_n →
  general_term_formula a_n →
  b_sequence T_n a_n lambda →
  c_sequence b_n c_n →
  sum_R_n R_n c_n :=
by
  intros
  sorry

end prove_general_term_formula_and_sum_R_n_l468_468129


namespace zeros_of_f_x_minus_1_l468_468362

def f (x : ℝ) : ℝ := x^2 - 1

theorem zeros_of_f_x_minus_1 :
  (f (0 - 1) = 0) ∧ (f (2 - 1) = 0) :=
by
  sorry

end zeros_of_f_x_minus_1_l468_468362


namespace half_angle_second_quadrant_l468_468058

theorem half_angle_second_quadrant (k : ℤ) (α : ℝ) (hα : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
    ∃ j : ℤ, (j * π + π / 4 < α / 2 ∧ α / 2 < j * π + π / 2) ∨ (j * π + 5 * π / 4 < α / 2 ∧ α / 2 < (j + 1) * π / 2) :=
sorry

end half_angle_second_quadrant_l468_468058


namespace numbers_with_digit_4_between_1_and_700_l468_468380

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k, n / 10^k % 10 = d

def count_numbers_with_digit (low high d : ℕ) : ℕ :=
  (low to high).count (λ n, contains_digit n d)

theorem numbers_with_digit_4_between_1_and_700 : 
  count_numbers_with_digit 1 700 4 = 214 := sorry

end numbers_with_digit_4_between_1_and_700_l468_468380


namespace greatest_y_value_l468_468159

theorem greatest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -6) : y ≤ 3 :=
sorry

end greatest_y_value_l468_468159


namespace hua_luogeng_optimal_selection_method_l468_468786

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l468_468786


namespace right_angle_triangle_iff_arithmetic_progression_l468_468147

noncomputable def exists_right_angle_triangle_with_rational_sides_and_area (d : ℤ) : Prop :=
  ∃ (a b c : ℚ), (a^2 + b^2 = c^2) ∧ (a * b = 2 * d)

noncomputable def rational_squares_in_arithmetic_progression (x y z : ℚ) : Prop :=
  2 * y^2 = x^2 + z^2

theorem right_angle_triangle_iff_arithmetic_progression (d : ℤ) :
  (∃ (a b c : ℚ), (a^2 + b^2 = c^2) ∧ (a * b = 2 * d)) ↔ ∃ (x y z : ℚ), rational_squares_in_arithmetic_progression x y z :=
sorry

end right_angle_triangle_iff_arithmetic_progression_l468_468147


namespace negation_P_l468_468037

variable (P : Prop) (P_def : ∀ x : ℝ, Real.sin x ≤ 1)

theorem negation_P : ¬P ↔ ∃ x : ℝ, Real.sin x > 1 := by
  sorry

end negation_P_l468_468037


namespace hua_luogeng_optimal_selection_method_l468_468730

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l468_468730


namespace b_n_plus_1_le_half_f_1_squared_l468_468473

theorem b_n_plus_1_le_half_f_1_squared {n : ℕ} {a : ℕ → ℝ} (h0 : ∀ (i : ℕ), i ≤ n → 0 ≤ a i ∧ a i ≤ a 0) :
  let f : ℝ → ℝ := λ x, ∑ i in range (n+1), (a i) * x^i,
      b : ℕ → ℝ := λ j, ∑ k in range (j+1), (∑ (i, l) in finset.zip (range k.succ) (range k.succ | (i + l = j)), a i * a l)
  in b (n+1) ≤ (1/2) * (f 1)^2 :=
by
  -- skipping the proof
  sorry

end b_n_plus_1_le_half_f_1_squared_l468_468473


namespace hua_luogeng_optimal_selection_method_l468_468792

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l468_468792


namespace optimal_selection_uses_golden_ratio_l468_468652

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l468_468652


namespace optimal_selection_method_uses_golden_ratio_l468_468863

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l468_468863


namespace probability_two_red_two_blue_l468_468237

noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_two_red_two_blue :
  (choose 15 2 * choose 9 2 : ℚ) / (choose 24 4 : ℚ) = 108 / 361 :=
by
  sorry

end probability_two_red_two_blue_l468_468237


namespace minimum_number_of_girls_l468_468905

theorem minimum_number_of_girls (total_students : ℕ) (d : ℕ) 
  (h_students : total_students = 20) 
  (h_unique_lists : ∀ n : ℕ, ∃! k : ℕ, k ≤ 20 - d ∧ n = 2 * k) :
  d ≥ 6 :=
sorry

end minimum_number_of_girls_l468_468905


namespace optimal_selection_golden_ratio_l468_468699

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l468_468699


namespace hua_luogeng_optimal_selection_method_l468_468737

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l468_468737


namespace lightest_boy_heaviest_girl_l468_468409

theorem lightest_boy_heaviest_girl :
  ∃ (B G : ℕ), B + G < 35 ∧
  (∃ (wb : ℕ → ℝ), (∀ i, wb i > 0) ∧ (∑ i in Finset.range B, wb i) = 60 * B ∧ (∃ i, wb i = min (Finset.range B) wb)) ∧
  (∃ (wg : ℕ → ℝ), (∀ i, wg i > 0) ∧ (∑ i in Finset.range G, wg i) = 47 * G ∧ (∃ i, wg i = max (Finset.range G) wg)) ∧
  (∑ i in Finset.range B, wb i + ∑ i in Finset.range G, wg i) = 53.5 * (B + G) :=
begin
  sorry
end

end lightest_boy_heaviest_girl_l468_468409


namespace second_discount_percentage_l468_468269

-- Define the normal price of the article
def normal_price : ℝ := 199.99999999999997

-- Define the sale price after two discounts
def final_price : ℝ := 144.0

-- Define the first discount percentage
def first_discount_percentage : ℝ := 10.0

-- Define the price after the first discount
def price_after_first_discount : ℝ := normal_price * (1.0 - first_discount_percentage / 100.0)

-- Equation for the second discount percentage that we need to prove is 20%.
theorem second_discount_percentage : 
  ∃ D : ℝ, D = 20 ∧ 
  final_price = price_after_first_discount * (1.0 - D / 100) :=
by
  sorry

end second_discount_percentage_l468_468269


namespace sum_of_divisors_multiple_of_15_l468_468469

open Nat

theorem sum_of_divisors_multiple_of_15 :
  ∃ n : ℕ, (∀ k : ℕ, 
  (n / 2 ^ 42) = k ^ 2 ∧ (n / 3 ^ 42) = k ^ 3 ∧ (n / 5 ^ 42) = k ^ 7 ∧
  (∑ d in (divisors n).filter (λ d, 15 ∣ d), d) = (∏ p in [2, 3, 5], (p ^ 42) + 1 - 1) - 1) :=
sorry

end sum_of_divisors_multiple_of_15_l468_468469


namespace optimal_selection_method_is_golden_ratio_l468_468587

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l468_468587


namespace smallest_whole_number_larger_than_triangle_perimeter_l468_468956

theorem smallest_whole_number_larger_than_triangle_perimeter
  (s : ℝ) (h1 : 5 + 19 > s) (h2 : 5 + s > 19) (h3 : 19 + s > 5) :
  ∃ P : ℝ, P = 5 + 19 + s ∧ P < 48 ∧ ∀ n : ℤ, n > P → n = 48 :=
by
  sorry

end smallest_whole_number_larger_than_triangle_perimeter_l468_468956


namespace optimal_selection_method_uses_golden_ratio_l468_468576

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l468_468576


namespace optimal_selection_method_uses_golden_ratio_l468_468673

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468673


namespace min_value_abs_ab_l468_468876

theorem min_value_abs_ab (a b : ℝ) (hab : a ≠ 0 ∧ b ≠ 0) 
(h_perpendicular : - 1 / (a^2) * (a^2 + 1) / b = -1) :
|a * b| = 2 :=
sorry

end min_value_abs_ab_l468_468876


namespace optimal_selection_method_uses_golden_ratio_l468_468602

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l468_468602


namespace optimal_selection_method_uses_golden_ratio_l468_468599

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l468_468599


namespace optimal_selection_method_use_golden_ratio_l468_468811

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l468_468811


namespace optimal_selection_method_uses_golden_ratio_l468_468682

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468682


namespace hua_luogeng_optimal_selection_l468_468558

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l468_468558


namespace optimal_selection_method_uses_golden_ratio_l468_468861

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l468_468861


namespace min_number_of_girls_l468_468912

theorem min_number_of_girls (total_students boys_count girls_count : ℕ) (no_three_equal: ∀ m n : ℕ, m ≠ n → boys_count m ≠ boys_count n) (boys_at_most : boys_count ≤ 2 * (girls_count + 1)) : ∃ d : ℕ, d ≥ 6 ∧ total_students = 20 ∧ boys_count = total_students - girls_count :=
by
  let d := 6
  exists d
  sorry

end min_number_of_girls_l468_468912


namespace sum_of_first_35_terms_of_bn_eq_397_l468_468012

def Sn (n : ℕ) : ℚ := (n^2 / 3 : ℚ)
def an (n : ℕ) : ℚ := Sn n - Sn (n - 1)
def floor_ℚ (x : ℚ) : ℤ := int.floor x
def bn (n : ℕ) : ℤ := floor_ℚ (an n)

theorem sum_of_first_35_terms_of_bn_eq_397 :
  (∑ i in Finset.range 35, (bn (i + 1))) = 397 :=
by
  sorry

end sum_of_first_35_terms_of_bn_eq_397_l468_468012


namespace Reema_loan_problem_l468_468517

-- Define problem parameters
def Principal : ℝ := 150000
def Interest : ℝ := 42000
def ProfitRate : ℝ := 0.1
def Profit : ℝ := 25000

-- State the problem as a Lean 4 theorem
theorem Reema_loan_problem (R : ℝ) (Investment : ℝ) : 
  Principal * (R / 100) * R = Interest ∧ 
  Profit = Investment * ProfitRate * R ∧ 
  R = 5 ∧ 
  Investment = 50000 :=
by
  sorry

end Reema_loan_problem_l468_468517


namespace intersection_parallel_to_given_line_l468_468512

noncomputable def planes_and_lines (α β : set (set ℝ^3)) (c l : set ℝ^3) : Prop :=
  ∃ (a b : set ℝ^3), 
    (∀ x ∈ α, x = a ∨ x ⊥ a) ∧ (a ∥ l) ∧ 
    (∀ y ∈ β, y = b ∨ y ⊥ b) ∧ (b ∥ l) ∧ 
    (∀ z, z ∈ α ∧ z ∈ β ↔ z ∈ c)

theorem intersection_parallel_to_given_line
  (α β : set (set ℝ^3)) (c l : set ℝ^3)
  (hα : ∀ p, p ∈ α → p ∥ l) 
  (hβ : ∀ q, q ∈ β → q ∥ l) 
  (h_intersection : ∀ r, r ∈ c ↔ r ∈ α ∧ r ∈ β) :
  c ∥ l :=
sorry

end intersection_parallel_to_given_line_l468_468512


namespace no_interchangeable_tiles_l468_468238

-- Defining our basic types for the tiles
inductive TileType
| two_by_two
| one_by_four

def can_replace_tile (bathroom_tiled : list TileType) : Prop :=
  ∃ broken_tile other_tile, broken_tile ≠ other_tile ∧ 
  (broken_tile = TileType.two_by_two ∧ other_tile = TileType.one_by_four ∨
  broken_tile = TileType.one_by_four ∧ other_tile = TileType.two_by_two) ∧
  ¬ can_replace_rest bathroom_tiled broken_tile other_tile

-- Assume we have a function that, if given the list of tiled pieces with one broken
-- and given the type of the other tile, determines if replacement is possible
axiom can_replace_rest : list TileType → TileType → TileType → Prop

-- The main theorem stating the impossibility as described
theorem no_interchangeable_tiles (bathroom_tiled : list TileType) :
  can_replace_tile bathroom_tiled :=
sorry

end no_interchangeable_tiles_l468_468238


namespace harmonic_mean_pairs_count_l468_468307

theorem harmonic_mean_pairs_count :
  {n // n = ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x < y ∧ (2 * x * y = 1024 * (x + y))} = 9 :=
sorry

end harmonic_mean_pairs_count_l468_468307


namespace optimal_selection_golden_ratio_l468_468693

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l468_468693


namespace find_cos_minus_sin_l468_468002

-- Definitions from the conditions
variable (α : ℝ)
variable (h1 : 0 < α ∧ α < π)  -- Second quadrant
variable (h2 : Real.sin (2 * α) = -24 / 25)  -- Given sin 2α

-- Lean statement of the problem
theorem find_cos_minus_sin (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : Real.sin (2 * α) = -24 / 25) :
  Real.cos α - Real.sin α = -7 / 5 := 
sorry

end find_cos_minus_sin_l468_468002


namespace min_girls_in_class_l468_468932

theorem min_girls_in_class (n : ℕ) (d : ℕ) (h1 : n = 20) 
  (h2 : ∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
       ∀ f : ℕ → set ℕ, (card (f i) ≠ card (f j) ∨ card (f j) ≠ card (f k) ∨ card (f i) ≠ card (f k))) : 
d ≥ 6 :=
by
  sorry

end min_girls_in_class_l468_468932


namespace sum_of_values_l468_468059

theorem sum_of_values (N : ℝ) (R : ℝ) (hN : N ≠ 0) (h_eq : N - 3 / N = R) :
  let N1 := (-R + Real.sqrt (R^2 + 12)) / 2
  let N2 := (-R - Real.sqrt (R^2 + 12)) / 2
  N1 + N2 = R :=
by
  sorry

end sum_of_values_l468_468059


namespace minimum_value_inequality_l468_468112

theorem minimum_value_inequality (a b c : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 3) :
  let expr := (a - 2)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (3 / c - 1)^2
  in expr ≥ 4 * (Real.sqrt (Real.sqrt 9) - 5 / 4)^2 :=
by
  sorry

end minimum_value_inequality_l468_468112


namespace one_coin_heads_down_l468_468289

theorem one_coin_heads_down (n : ℕ) : 
  ∃ k : ℤ, 
    ∀ i : ℤ, 
    (0 ≤ i ∧ i < 2*n + 1) → 
    (heads_down_initial n).filter (λ x, x = i mod (2*n + 1)).length = 1 :=
by sorry

end one_coin_heads_down_l468_468289


namespace optimal_selection_method_uses_golden_ratio_l468_468619

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l468_468619


namespace optimal_selection_golden_ratio_l468_468688

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l468_468688


namespace optimal_selection_uses_golden_ratio_l468_468648

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l468_468648


namespace optimalSelectionUsesGoldenRatio_l468_468710

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l468_468710


namespace find_x_l468_468216

theorem find_x :
  ∃ x : ℝ, 3 * x = (26 - x) + 14 ∧ x = 10 :=
by sorry

end find_x_l468_468216


namespace optimal_selection_method_is_golden_ratio_l468_468593

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l468_468593


namespace hotdogs_needed_l468_468135

theorem hotdogs_needed 
  (ella_hotdogs : ℕ) (emma_hotdogs : ℕ)
  (luke_multiple : ℕ) (hunter_multiple : ℚ)
  (h_ella : ella_hotdogs = 2)
  (h_emma : emma_hotdogs = 2)
  (h_luke : luke_multiple = 2)
  (h_hunter : hunter_multiple = (3/2)) :
  ella_hotdogs + emma_hotdogs + luke_multiple * (ella_hotdogs + emma_hotdogs) + hunter_multiple * (ella_hotdogs + emma_hotdogs) = 18 := by
    sorry

end hotdogs_needed_l468_468135


namespace tangent_line_at_1_l468_468172

noncomputable def f (x : ℝ) : ℝ := x^3 + x^2

theorem tangent_line_at_1 (x y : ℝ) :
    f 1 = 2 ∧ y = f' 1 * (x - 1) + f 1 → y = 5 * x - 3 :=
by
  sorry

end tangent_line_at_1_l468_468172


namespace knocks_to_knicks_l468_468053

-- Define the conversion relationships as conditions
variable (knicks knacks knocks : ℝ)

-- 8 knicks = 3 knacks
axiom h1 : 8 * knicks = 3 * knacks

-- 5 knacks = 6 knocks
axiom h2 : 5 * knacks = 6 * knocks

-- Proving the equivalence of 30 knocks to 66.67 knicks
theorem knocks_to_knicks : 30 * knocks = 66.67 * knicks :=
by
  sorry

end knocks_to_knicks_l468_468053


namespace optimal_selection_golden_ratio_l468_468692

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l468_468692


namespace optimal_selection_method_uses_golden_ratio_l468_468749

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l468_468749


namespace vanessa_savings_weeks_l468_468948

-- Definitions of given conditions
def dress_cost : ℕ := 80
def vanessa_savings : ℕ := 20
def weekly_allowance : ℕ := 30
def weekly_spending : ℕ := 10

-- Required amount to save 
def required_savings : ℕ := dress_cost - vanessa_savings

-- Weekly savings calculation
def weekly_savings : ℕ := weekly_allowance - weekly_spending

-- Number of weeks needed to save the required amount
def weeks_needed_to_save (required_savings weekly_savings : ℕ) : ℕ :=
  required_savings / weekly_savings

-- Axiom representing the correctness of our calculation
theorem vanessa_savings_weeks : weeks_needed_to_save required_savings weekly_savings = 3 := 
  by
  sorry

end vanessa_savings_weeks_l468_468948


namespace cost_to_paint_cube_l468_468400

-- Define the given conditions
def cost_per_quart : ℝ := 3.20
def coverage_per_quart : ℝ := 60
def edge_length : ℝ := 10

-- The theorem that we need to prove
theorem cost_to_paint_cube : 
  let face_area := edge_length ^ 2 in
  let surface_area := 6 * face_area in
  let quarts_needed := surface_area / coverage_per_quart in
  let total_cost := quarts_needed * cost_per_quart in
  total_cost = 32 := 
by
  sorry

end cost_to_paint_cube_l468_468400


namespace weights_problem_l468_468938

theorem weights_problem (n : ℕ) (x : ℝ) (h_avg : ∀ (i : ℕ), i < n → ∃ (w : ℝ), w = x) 
  (h_heaviest : ∃ (w_max : ℝ), w_max = 5 * x) : n > 5 :=
by
  sorry

end weights_problem_l468_468938


namespace optimal_selection_uses_golden_ratio_l468_468655

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l468_468655


namespace joe_gets_home_in_10_minutes_l468_468096

noncomputable def joe_biking_time (x : ℝ) : ℝ := x / 20
noncomputable def joe_total_biking_time (x : ℝ) : ℝ := (x / 20) + (x / 20) + ((x + 2) / 14)
noncomputable def distance_to_home (x : ℝ) : ℝ := sqrt ((x^2) + ((x + 2)^2))
noncomputable def time_to_home_by_helicopter (d : ℝ) : ℝ := d / 78

theorem joe_gets_home_in_10_minutes :
  ∀ (x : ℝ), joe_total_biking_time x = 1 → time_to_home_by_helicopter (distance_to_home x) = 10 / 60 :=
by
  intros x hx
  sorry

end joe_gets_home_in_10_minutes_l468_468096


namespace lattice_point_distance_l468_468251

-- Define the problem-specific conditions and question
theorem lattice_point_distance (d : ℝ) : 
  (∃ (square : set (ℝ × ℝ)), 
    square = {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3030 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3030} ∧
    (∃ (lattice_points : set (ℝ × ℝ)), 
        lattice_points = {p : ℝ × ℝ | ∃ (m n : ℕ), p = ⟨m, n⟩} ∧
        (∀ p ∈ square, (∃ q ∈ lattice_points, dist p q ≤ d) ↔ dist p q ≤ 0.5))) :=
by sorry

end lattice_point_distance_l468_468251


namespace optimal_selection_method_uses_golden_ratio_l468_468612

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l468_468612


namespace optimalSelectionUsesGoldenRatio_l468_468723

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l468_468723


namespace repeating_decimal_fraction_l468_468301

theorem repeating_decimal_fraction (x : ℚ) (h : x = 7.5656) : x = 749 / 99 :=
by
  sorry

end repeating_decimal_fraction_l468_468301


namespace domain_of_g_l468_468356

noncomputable def f (m : ℝ) := (m^2 - m - 1) * (x : ℝ) ^ (-5 * m - 3)

-- Define conditions
def condition1 (m : ℝ) : Prop := m^2 - m - 1 = 1
def condition2 (m : ℝ) : Prop := -5 * m - 3 > 0

-- Define the domain of g(x)
def g_domain : Set ℝ := { x | 1 < x ∧ x < 2 }

-- Main statement
theorem domain_of_g (m : ℝ) (x : ℝ)
  (h1 : condition1 m)
  (h2 : condition2 m) :
  x ∈ g_domain :=
sorry

end domain_of_g_l468_468356


namespace number_of_correct_statements_l468_468518

-- Definitions of the conditions from the problem
def seq_is_graphical_points := true  -- Statement 1
def seq_is_finite (s : ℕ → ℝ) := ∀ n, s n = 0 -- Statement 2
def seq_decreasing_implies_finite (s : ℕ → ℝ) := (∀ n, s (n + 1) ≤ s n) → seq_is_finite s -- Statement 3

-- Prove the number of correct statements is 1
theorem number_of_correct_statements : (seq_is_graphical_points = true ∧ ¬(∃ s: ℕ → ℝ, ¬seq_is_finite s) ∧ ∃ s : ℕ → ℝ, ¬seq_decreasing_implies_finite s) → 1 = 1 :=
by
  sorry

end number_of_correct_statements_l468_468518


namespace main_theorem_l468_468474

noncomputable def twice_cont_diff (f: ℝ^d → ℝ) : Prop := sorry

def cond1 (f: ℝ^d → (ℝ positive)) : Prop :=
twice_cont_diff f

def cond2 (f: ℝ^d → (ℝ positive)) : Prop :=
∀ x ∈ ℝ^d, 
  let h := real.sqrt (f x) in 
    ∑ i: Fin d, deriv' (deriv' (h (fun j => if j = i then x i else x j))) ≤ 0

noncomputable def cond3 (f: ℝ^d → (ℝ positive)) : Prop :=
(E (norm (grad (ln (f ξ))))^2 < ∞) ∧ 
(E (∑ i: Fin d, deriv'' (f ξ)) / (f ξ) < ∞)

-- Gaussian vector in ℝ^d with unit covariance matrix
noncomputable def gaussian (d: ℕ) : Type :=
 sorry -- this needs to be defined appropriately 

variables {d: ℕ} (f: ℝ^d → (ℝ positive)) 

noncomputable def cond4 : Prop :=
 ∀ (ξ : gaussian d), covar ξ = 1

variables (ξ : gaussian d)
variables (A: ℝ^d → ℝ^d) (B: ℝ^d → ℝ^d)

theorem main_theorem (h_f : cond1 f) (h_cond2 : cond2 f) (h_cond3: cond3 f) (h_gaussian : cond4) :
  (E (norm (ξ + grad (ln f ξ) - E ξ)^2) ≤ E (norm (ξ - (E ξ))^2) = d)
  ∧ 
  (d ≥ 3 → E ((norm (ξ + A ξ/(ξ^T B ξ) - E ξ))^2) = 
  d - E ((ξ V.T A^2 ξ)/((ξ^T B ξ)^2))) := 
sorry

end main_theorem_l468_468474


namespace optimal_selection_method_uses_golden_ratio_l468_468613

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l468_468613


namespace optimal_selection_uses_golden_ratio_l468_468654

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l468_468654


namespace permutation_mod_l468_468462

theorem permutation_mod (M : ℕ) :
  (∀ perm : list (fin 15), perm.length = 15 ∧
    (∀ i ∈ finset.range 5, perm[i] ≠ 0) ∧  -- None of the first 5 letters is a D
    (∀ i ∈ finset.range 5, perm[i+5] ≠ 1) ∧  -- None of the next 5 letters is an E
    (∀ i ∈ finset.range 5, perm[i+10] ≠ 2)  -- None of the last 5 letters is an F
  ) → M = 1501 ∧ M % 1000 = 501 :=
by
sorry

end permutation_mod_l468_468462


namespace optimal_selection_method_is_golden_ratio_l468_468594

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l468_468594


namespace optimal_selection_uses_golden_ratio_l468_468642

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l468_468642


namespace optimal_selection_uses_golden_ratio_l468_468659

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l468_468659


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468760

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468760


namespace optimal_selection_method_uses_golden_ratio_l468_468573

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l468_468573


namespace a1_is_1_general_formula_a_n_l468_468132

open Nat

-- Define the sequences a, S, and T, and the initial conditions for T and S
def a (n : ℕ) : ℕ := sorry
def S (n : ℕ) : ℕ := ∑ i in range(n+1), a i
def T (n : ℕ) : ℕ := 2 * S n - n^2 

-- First proof statement: Prove that a_1 = 1
theorem a1_is_1 : a 1 = 1 := 
by
  have h1 : T 1 = 2 * S 1 - 1 := sorry
  have h2 : T 1 = S 1 := sorry
  have h3 : S 1 = a 1 := sorry
  rw [h2, h3] at h1
  exact sorry

-- Second proof statement: Prove the general formula for a_n
theorem general_formula_a_n (n : ℕ) (hn : n > 0) : a n = 3 * 2^(n-1) - 2 :=
by
  induction n with n IH
  { -- base case n = 1
    sorry
  }
  { -- induction step
    assume hn
    have h1 : T (n + 1) = 2 * S (n + 1) - (n + 1)^2 := sorry
    have h2 : a (n + 1) = 2 * a n + 2 := sorry
    rw [IH hn] at h2
    exact sorry
  }

end a1_is_1_general_formula_a_n_l468_468132


namespace optimal_selection_uses_golden_ratio_l468_468814

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l468_468814


namespace intersection_volume_l468_468944

-- Definition: Two congruent cubes sharing a common diagonal, one rotated 60 degrees around it.
structure CongruentCubesIntersection where
  a : ℝ -- edge length of the cubes
  volume_intersection : ℝ -- the volume of the intersection

-- Prove that the volume of the intersection is 3/4 * a^3
theorem intersection_volume (cubes : CongruentCubesIntersection)
  (h : ∃ a, cubes.a > 0) -- edge length a must be greater than 0
  (h1 : cubes.shared_diagonal) -- condition of sharing a diagonal (not fully formalized)
  (h2 : cubes.rotation_60deg) -- condition of 60-degree rotation (not fully formalized)
  : cubes.volume_intersection = (3/4) * cubes.a^3 := 
sorry

end intersection_volume_l468_468944


namespace concatenation_of_powers_of_3_irrational_l468_468146

noncomputable def concatenated_digits_of_powers_of_3 : ℝ :=
  Real.of_rat (nat.ceil (λ k : ℕ, 3^k).to_real)

theorem concatenation_of_powers_of_3_irrational : ¬ (∃ (p q : ℤ), q ≠ 0 ∧ concatenated_digits_of_powers_of_3 = p / q) :=
by 
  sorry

end concatenation_of_powers_of_3_irrational_l468_468146


namespace abs_sum_factors_l468_468122

theorem abs_sum_factors (T : ℤ) :
  (∀ c : ℤ, (∃ u v : ℤ, u + v = -c ∧ u * v = 2016 * c) → c ∈ T) →
  |T| = 665280 :=
by
  sorry

end abs_sum_factors_l468_468122


namespace lightest_boy_heaviest_girl_l468_468413

theorem lightest_boy_heaviest_girl :
  ∃ (B G : ℕ), B + G < 35 ∧
  (∃ (wb : ℕ → ℝ), (∀ i, wb i > 0) ∧ (∑ i in Finset.range B, wb i) = 60 * B ∧ (∃ i, wb i = min (Finset.range B) wb)) ∧
  (∃ (wg : ℕ → ℝ), (∀ i, wg i > 0) ∧ (∑ i in Finset.range G, wg i) = 47 * G ∧ (∃ i, wg i = max (Finset.range G) wg)) ∧
  (∑ i in Finset.range B, wb i + ∑ i in Finset.range G, wg i) = 53.5 * (B + G) :=
begin
  sorry
end

end lightest_boy_heaviest_girl_l468_468413


namespace binary_div_remainder_l468_468215

theorem binary_div_remainder (n : ℕ) (h : n = 0b101011100101) : n % 8 = 5 :=
by sorry

end binary_div_remainder_l468_468215


namespace Yvonne_laps_l468_468975

-- Definitions of the given conditions
def laps_swim_by_Yvonne (l_y : ℕ) : Prop := 
  ∃ l_s l_j, 
  l_s = l_y / 2 ∧ 
  l_j = 3 * l_s ∧ 
  l_j = 15

-- Theorem statement
theorem Yvonne_laps (l_y : ℕ) (h : laps_swim_by_Yvonne l_y) : l_y = 10 :=
sorry

end Yvonne_laps_l468_468975


namespace optimalSelectionUsesGoldenRatio_l468_468714

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l468_468714


namespace count_irrationals_l468_468264

open Real

def is_irrational (x : ℝ) : Prop := ¬∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def num_irrationals (l : List ℝ) : ℕ :=
  l.countp is_irrational

def number_list : List ℝ :=
  [22 / 7, Real.pi, 0, Real.sqrt (4 / 9), 0.6, 
   let rec special_num (n : ℕ) : ℝ :=
     if n = 0 then 0.1 else 1 + (10 ^ n * (2 / 10)) + special_num (n-1)
   in special_num 5
  ]

theorem count_irrationals : num_irrationals number_list = 2 := by
  sorry

end count_irrationals_l468_468264


namespace optimal_selection_method_uses_golden_ratio_l468_468844

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468844


namespace knocks_to_knicks_l468_468052

-- Define the conversion relationships as conditions
variable (knicks knacks knocks : ℝ)

-- 8 knicks = 3 knacks
axiom h1 : 8 * knicks = 3 * knacks

-- 5 knacks = 6 knocks
axiom h2 : 5 * knacks = 6 * knocks

-- Proving the equivalence of 30 knocks to 66.67 knicks
theorem knocks_to_knicks : 30 * knocks = 66.67 * knicks :=
by
  sorry

end knocks_to_knicks_l468_468052


namespace average_monthly_growth_rate_equation_l468_468279

variable (x : ℝ)

-- Definitions derived from the conditions.
def january_visitors : ℝ := 60000
def march_visitors : ℝ := 150000
def simplified_january_visitors : ℝ := january_visitors / 10000
def simplified_march_visitors : ℝ := march_visitors / 10000

-- The theorem we need to prove.
theorem average_monthly_growth_rate_equation :
  6 * (1 + x)^2 = 15 :=
sorry

end average_monthly_growth_rate_equation_l468_468279


namespace factorization_x3_minus_9xy2_l468_468230

theorem factorization_x3_minus_9xy2 (x y : ℝ) : x^3 - 9 * x * y^2 = x * (x + 3 * y) * (x - 3 * y) :=
by sorry

end factorization_x3_minus_9xy2_l468_468230


namespace angle_215_in_quadrant_III_l468_468446

theorem angle_215_in_quadrant_III (θ : ℝ) (h0 : 0 < θ) (h1 : θ < 360) : 180 < θ ∧ θ < 270 → 215 = θ → "Quadrant III" := 
by
  intros h3
  intros h4
  sorry

end angle_215_in_quadrant_III_l468_468446


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468773

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468773


namespace vanessa_savings_weeks_l468_468946

-- Definitions of given conditions
def dress_cost : ℕ := 80
def vanessa_savings : ℕ := 20
def weekly_allowance : ℕ := 30
def weekly_spending : ℕ := 10

-- Required amount to save 
def required_savings : ℕ := dress_cost - vanessa_savings

-- Weekly savings calculation
def weekly_savings : ℕ := weekly_allowance - weekly_spending

-- Number of weeks needed to save the required amount
def weeks_needed_to_save (required_savings weekly_savings : ℕ) : ℕ :=
  required_savings / weekly_savings

-- Axiom representing the correctness of our calculation
theorem vanessa_savings_weeks : weeks_needed_to_save required_savings weekly_savings = 3 := 
  by
  sorry

end vanessa_savings_weeks_l468_468946


namespace entropy_32_students_entropy_contest_winners_l468_468399

noncomputable def entropy_function (x : ℝ) (a : ℝ) : ℝ :=
  -x * real.log x / real.log a

theorem entropy_32_students : 
  entropy_function (1 / 32) 2 * 32 = 5 := by
  sorry

noncomputable def probability_contestant (n k : ℕ) : ℝ :=
  if k < n then 2^(-(k : ℝ)) else 1 - (1 - 1 / 2^(n-1))

theorem entropy_contest_winners (n : ℕ) (h : n > 1) : 
  (finset.range n).sum (λ k, entropy_function (probability_contestant n k) 2) 
    = 2 - 4 / 2^n := by
  sorry

end entropy_32_students_entropy_contest_winners_l468_468399


namespace football_players_count_l468_468071

theorem football_players_count (total_students play_long_tennis play_both neither_play : ℕ) 
  (H1 : total_students = 35) (H2 : play_long_tennis = 20) (H3 : play_both = 17) (H4 : neither_play = 6) : 
  let F := total_students - neither_play + play_both - play_long_tennis - 3 
  in F = 26 :=
by {
  sorry
}

end football_players_count_l468_468071


namespace optimal_selection_uses_golden_ratio_l468_468816

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l468_468816


namespace max_value_of_f_on_interval_l468_468877

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x

theorem max_value_of_f_on_interval :
  ∃ x ∈ Icc (-2 : ℝ) (2 : ℝ), ∀ y ∈ Icc (-2 : ℝ) (2 : ℝ), f y ≤ f x ∧ f x = 2 :=
sorry

end max_value_of_f_on_interval_l468_468877


namespace nesbitts_inequality_l468_468232

theorem nesbitts_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ (3 / 2) :=
sorry

end nesbitts_inequality_l468_468232


namespace doubled_base_and_exponent_l468_468126

theorem doubled_base_and_exponent (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0)
  (h : (2 * a) ^ (2 * b) = a ^ b * x ^ 3) : 
  x = (4 ^ b * a ^ b) ^ (1 / 3) :=
by
  sorry

end doubled_base_and_exponent_l468_468126


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468777

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468777


namespace total_gas_cost_for_trip_l468_468152

theorem total_gas_cost_for_trip
  (odometer_pick_up : ℕ := 74568)
  (odometer_grocery_store : ℕ := 74580)
  (odometer_home : ℕ := 74608)
  (miles_per_gallon : ℝ := 32)
  (price_per_gallon : ℝ := 4.30) :
  (Float.round ((odometer_home - odometer_pick_up : ℝ) / miles_per_gallon * price_per_gallon) : ℝ) = 5.38 :=
by
  sorry

end total_gas_cost_for_trip_l468_468152


namespace sum_first_10_terms_eq_55_l468_468317

noncomputable def curve_y (n : ℕ) (x : ℝ) : ℝ :=
  x^n * (1 - x)

noncomputable def tangent_y_intercept (n : ℕ) : ℝ :=
  (n + 1) * 2^n 

def sequence_b (n : ℕ) : ℝ :=
  Real.log 2 ((tangent_y_intercept n)/(n + 1))

def sum_first_10_sequence_b : ℝ :=
  (List.range 10).map (λ i, sequence_b (i + 1)) |>.sum

theorem sum_first_10_terms_eq_55 : sum_first_10_sequence_b = 55 :=
  by
  sorry

end sum_first_10_terms_eq_55_l468_468317


namespace optimal_selection_method_uses_golden_ratio_l468_468598

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l468_468598


namespace optimal_selection_method_use_golden_ratio_l468_468802

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l468_468802


namespace optimal_selection_method_uses_golden_ratio_l468_468742

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l468_468742


namespace optimal_selection_uses_golden_ratio_l468_468826

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l468_468826


namespace ratio_is_one_to_five_l468_468208

def ratio_of_minutes_to_hour (twelve_minutes : ℕ) (one_hour : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd twelve_minutes one_hour
  (twelve_minutes / gcd, one_hour / gcd)

theorem ratio_is_one_to_five : ratio_of_minutes_to_hour 12 60 = (1, 5) := 
by 
  sorry

end ratio_is_one_to_five_l468_468208


namespace sum_of_frac_num_den_l468_468213

theorem sum_of_frac_num_den (x : ℚ) (h1 : x = 0.272727...) (h2 : x = 27 / 99) :
  ((27 / 99).num + (27 / 99).denom) = 14 :=
by 
  have h3 : x = 3 / 11 := by sorry
  sorry

end sum_of_frac_num_den_l468_468213


namespace morgan_hula_hooping_time_l468_468504

-- Definitions based on conditions
def nancy_can_hula_hoop : ℕ := 10
def casey_can_hula_hoop : ℕ := nancy_can_hula_hoop - 3
def morgan_can_hula_hoop : ℕ := 3 * casey_can_hula_hoop

-- Theorem statement to show the solution is correct
theorem morgan_hula_hooping_time : morgan_can_hula_hoop = 21 :=
by
  sorry

end morgan_hula_hooping_time_l468_468504


namespace alexis_dresses_l468_468092

-- Definitions based on the conditions
def isabella_total : ℕ := 13
def alexis_total : ℕ := 3 * isabella_total
def alexis_pants : ℕ := 21

-- Theorem statement
theorem alexis_dresses : alexis_total - alexis_pants = 18 := by
  sorry

end alexis_dresses_l468_468092


namespace minimum_number_of_girls_l468_468907

theorem minimum_number_of_girls (total_students : ℕ) (d : ℕ) 
  (h_students : total_students = 20) 
  (h_unique_lists : ∀ n : ℕ, ∃! k : ℕ, k ≤ 20 - d ∧ n = 2 * k) :
  d ≥ 6 :=
sorry

end minimum_number_of_girls_l468_468907


namespace sequence_property_l468_468039

noncomputable def sequence (n : ℕ+) : ℕ+ :=
  if n % 2 = 1 then (n + 1)^2 / 4 else (n^2 - 2 * n + 8) / 4

theorem sequence_property :
  (∀ n : ℕ+, sequence n = if n % 2 = 1 then (n + 1)^2 / 4 else (n^2 - 2 * n + 8) / 4) ∧
  (∀ (n : ℕ+), (∑ i in finset.range (2 * n), 1 / sequence (i + 1)) < 7 / 2) 
:=
begin
  sorry
end

end sequence_property_l468_468039


namespace hua_luogeng_optimal_selection_method_l468_468734

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l468_468734


namespace box_length_l468_468241

theorem box_length 
  (L : ℕ) 
  (h1 : ∃ S : ℕ, 90 * S^3 = L * 15 * 6 ∧ (S = 3)) :
  L = 27 := by 
  cases h1 with S h1,
  cases h1 with h_volume h_side_length,
  have : S = 3 := h_side_length,
  rw this at h_volume,
  norm_num at h_volume,
  exact h_volume

end box_length_l468_468241


namespace optimal_selection_method_uses_golden_ratio_l468_468621

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l468_468621


namespace fibonacci_even_iff_mod_three_l468_468509

def fibonacci : ℕ → ℕ 
| 0 := 0
| 1 := 1
| (n + 2) := fibonacci n + fibonacci (n + 1)

theorem fibonacci_even_iff_mod_three (n : ℕ) : (fibonacci n % 2 = 0) ↔ (n % 3 = 0) :=
by sorry

end fibonacci_even_iff_mod_three_l468_468509


namespace find_polynomial_l468_468355

-- Declare the variable x as a real number
variable (x : ℝ)

-- Define the polynomials
def P (x : ℝ) := -x^2 + 5*x - 3

lemma polynomial_identity : (P x) + (x^2 - 2*x + 1) = x^2 - 2*x + 1 :=
begin
  -- Lean will require the actual proof here, so we place sorry for now
  sorry
end

-- Main statement we need to prove
theorem find_polynomial : P x = 2*x^2 - 7*x + 4 :=
begin
  -- As per the instruction, we skip the proof and provide a placeholder
  sorry
end

end find_polynomial_l468_468355


namespace triangle_altitude_inequality_l468_468513

theorem triangle_altitude_inequality (a b c m_a m_b m_c : ℝ)
  (h_ma : m_a = (2 / 3) * sqrt ((b + c - a) * (b + c + a) / (b + c))) 
  (h_mb : m_b = (2 / 3) * sqrt ((a + c - b) * (a + c + b) / (a + c)))
  (h_mc : m_c = (2 / 3) * sqrt ((a + b - c) * (a + b + c) / (a + b))) :
  (a / m_a)^2 + (b / m_b)^2 + (c / m_c)^2 ≥ 4 := 
by
  sorry

end triangle_altitude_inequality_l468_468513


namespace min_number_of_girls_l468_468890

theorem min_number_of_girls (d : ℕ) (students : ℕ) (boys : ℕ → ℕ) : 
  students = 20 ∧ ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → boys i ≠ boys j ∨ boys j ≠ boys k ∨ boys i ≠ boys k → d = 6 :=
by
  sorry

end min_number_of_girls_l468_468890


namespace find_b_l468_468346

theorem find_b (b : ℤ) :
  ∃ (r₁ r₂ : ℤ), (r₁ = -9) ∧ (r₁ * r₂ = 36) ∧ (r₁ + r₂ = -b) → b = 13 :=
by {
  sorry
}

end find_b_l468_468346


namespace hua_luogeng_optimal_selection_method_l468_468726

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l468_468726


namespace find_y_l468_468385

theorem find_y (y : ℕ) 
  (h : (1/8) * 2^36 = 8^y) : y = 11 :=
sorry

end find_y_l468_468385


namespace part1_part2_l468_468438

open Real

noncomputable def a_value := 2 * sqrt 2

noncomputable def line_cartesian_eqn (x y : ℝ) : Prop :=
  x + y - 4 = 0

noncomputable def point_on_line (ρ θ : ℝ) :=
  ρ * cos (θ - π / 4) = a_value

noncomputable def curve_param_eqns (θ : ℝ) : (ℝ × ℝ) :=
  (sqrt 3 * cos θ, sin θ)

noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x + y - 4) / sqrt 2

theorem part1 (P : ℝ × ℝ) (ρ θ : ℝ) : 
  P = (4, π / 2) ∧ point_on_line ρ θ → 
  a_value = 2 * sqrt 2 ∧ line_cartesian_eqn 4 (4 * tan (π / 4)) :=
sorry

theorem part2 :
  (∀ θ : ℝ, distance_to_line (sqrt 3 * cos θ) (sin θ) ≤ 3 * sqrt 2) ∧
  (∃ θ : ℝ, distance_to_line (sqrt 3 * cos θ) (sin θ) = 3 * sqrt 2) :=
sorry

end part1_part2_l468_468438


namespace expiry_time_correct_l468_468974

def factorial (n : Nat) : Nat := match n with
| 0 => 1
| n + 1 => (n + 1) * factorial n

def seconds_in_a_day : Nat := 86400
def seconds_in_an_hour : Nat := 3600
def donation_time_seconds : Nat := 8 * seconds_in_an_hour
def expiry_seconds : Nat := factorial 8

def time_of_expiry (donation_time : Nat) (expiry_time : Nat) : Nat :=
  (donation_time + expiry_time) % seconds_in_a_day

def time_to_HM (time_seconds : Nat) : Nat × Nat :=
  let hours := time_seconds / seconds_in_an_hour
  let minutes := (time_seconds % seconds_in_an_hour) / 60
  (hours, minutes)

def is_correct_expiry_time : Prop :=
  let (hours, minutes) := time_to_HM (time_of_expiry donation_time_seconds expiry_seconds)
  hours = 19 ∧ minutes = 12

theorem expiry_time_correct : is_correct_expiry_time := by
  sorry

end expiry_time_correct_l468_468974


namespace intersection_of_A_and_B_values_of_a_and_b_l468_468481

noncomputable def A : Set ℝ := { x | x + 2 < 0 }
noncomputable def B : Set ℝ := { x | (x + 3) * (x - 1) > 0 }

theorem intersection_of_A_and_B : A ∩ B = { x | x < -3 } :=
sorry

theorem values_of_a_and_b (a b : ℝ) (h : { x | ax^2 + 2x + b > 0 } = A ∪ B) : 
    a = 2 ∧ b = -4 :=
sorry

end intersection_of_A_and_B_values_of_a_and_b_l468_468481


namespace symmetric_line_eq_l468_468372

-- Define the lines l1 and l
def l1 : ℝ → ℝ := λ x, 2 * x
def l : ℝ → ℝ := λ x, 3 * x + 3

-- Define the symmetric line l2
noncomputable def l2 : ℝ → ℝ := λ x, (2 * x - 21) / 11

-- Prove that the line l2 is symmetric to l1 with respect to l
theorem symmetric_line_eq :
  ∀ x y, (y = l1 x) → (l x = y) → (11 * x - 2 * y + 21 = 0) :=
begin
  sorry
end

end symmetric_line_eq_l468_468372


namespace prism_dimensions_l468_468538

theorem prism_dimensions (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 45) (h3 : b * c = 60) : 
  a = 7.2 ∧ b = 9.6 ∧ c = 14.4 :=
by {
  -- Proof skipped for now
  sorry
}

end prism_dimensions_l468_468538


namespace optimal_selection_uses_golden_ratio_l468_468646

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l468_468646


namespace optimal_selection_method_uses_golden_ratio_l468_468751

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l468_468751


namespace locus_of_B_l468_468377

variable (A Q B : Point)

def isMidpoint (M : Point) (A B : Point) : Prop :=
  dist A M = dist M B

def centroid (Q : Point) (A B C : Point) : Prop :=
  3 * Q = A + B + C

def isAcuteAngled (A B C : Point) : Prop :=
  ∀ (X Y Z : Point), (X ≠ A ∧ X ≠ B ∧ X ≠ C ∧ Y ≠ A ∧ Y ≠ B ∧ Y ≠ C ∧ Z ≠ A ∧ Z ≠ B ∧ Z ≠ C) →
    angle X Y Z < π / 2

theorem locus_of_B (A Q : Point) :
  ∃ M : Point, (dist Q M = (1/2) * dist A Q) ∧
  ∃ A1 : Point, (dist M A1 = dist A M ∧
  isMidpoint M B C ∧
  isAcuteAngled A B C ∧
  centroid Q A B C) →
  ∀ B : Point, (¬ onCircle (circleWithDiameter (A, M)) B ∧ 
                ¬ onCircle (circleWithDiameter (M, A1)) B ∧ 
                 onCircle (circleWithDiameter (A, A1)) B) :=
begin
  sorry
end

end locus_of_B_l468_468377


namespace optimal_selection_method_uses_golden_ratio_l468_468562

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l468_468562


namespace product_not_integer_l468_468514

theorem product_not_integer : ¬ ∃ (n : ℕ), 
  n = ∏ k in Finset.filter (λ x, x % 2 = 1) (Finset.range 2016), (1 + (1 / (k : ℝ))) := sorry

end product_not_integer_l468_468514


namespace optimal_selection_method_uses_golden_ratio_l468_468750

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l468_468750


namespace sum_of_myfavorite_digits_l468_468084

theorem sum_of_myfavorite_digits :
  ∃ (m y f a v o r i t e : ℕ),
    m ≠ y ∧ m ≠ f ∧ m ≠ a ∧ m ≠ v ∧ m ≠ o ∧ m ≠ r ∧ m ≠ i ∧ m ≠ t ∧ m ≠ e ∧
    y ≠ f ∧ y ≠ a ∧ y ≠ v ∧ y ≠ o ∧ y ≠ r ∧ y ≠ i ∧ y ≠ t ∧ y ≠ e ∧
    f ≠ a ∧ f ≠ v ∧ f ≠ o ∧ f ≠ r ∧ f ≠ i ∧ f ≠ t ∧ f ≠ e ∧
    a ≠ v ∧ a ≠ o ∧ a ≠ r ∧ a ≠ i ∧ a ≠ t ∧ a ≠ e ∧
    v ≠ o ∧ v ≠ r ∧ v ≠ i ∧ v ≠ t ∧ v ≠ e ∧
    o ≠ r ∧ o ≠ i ∧ o ≠ t ∧ o ≠ e ∧
    r ≠ i ∧ r ≠ t ∧ r ≠ e ∧
    i ≠ t ∧ i ≠ e ∧
    t ≠ e ∧
    {m, y, f, a, v, o, r, i, t, e} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    m + y + f + a + v + o + r + i + t + e = 45 :=
by
  existsi (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
  simp
  exact ⟨rfl, sorry⟩

end sum_of_myfavorite_digits_l468_468084


namespace all_roots_ge_one_l468_468325
-- Import the necessary library

-- Define the conditions of the problem
variables {R : Type*} [linear_ordered_comm_ring R] [algebra ℂ R]
variables (a : ℕ → R) {n : ℕ}

-- Define the polynomial and the conditions on the coefficients
def polynomial (a : ℕ → R) (n : ℕ) : polynomial R :=
∑ i in range (n + 1), monomial i (a i)

-- The proof statement
theorem all_roots_ge_one (h : ∀ i j, i ≤ j ∧ j ≤ n → a i ≥ a j) :
  ∀ z : ℂ, (polynomial a n).is_root z → complex.abs z ≥ 1 :=
begin
  sorry
end

end all_roots_ge_one_l468_468325


namespace possible_lightest_boy_heaviest_girl_l468_468421

theorem possible_lightest_boy_heaviest_girl
    (B G : ℕ) 
    (wb : ℕ → ℕ) 
    (wg : ℕ → ℕ) 
    (N < 35) 
    (avg_weight_students : Real) 
    (avg_weight_girls : Real) 
    (avg_weight_boys : Real) : 
    (avg_weight_students = 53.5) ∧ 
    (avg_weight_girls = 47) ∧ 
    (avg_weight_boys = 60) → 
    (∃ i j : ℕ, (i < B) ∧ (j < G) ∧ (wb i < wg j)) :=
by 
  sorry

end possible_lightest_boy_heaviest_girl_l468_468421


namespace B_worked_10_days_l468_468242

-- Define the work rates
def work_rate_A : ℝ := 1 / 40
def work_rate_B : ℝ := 1 / 40
def work_rate_C : ℝ := 1 / 20

-- Define the work done by A in 10 days
def work_A_in_10_days : ℝ := 10 * work_rate_A

-- Define the work done by C in 10 days
def work_C_in_10_days : ℝ := 10 * work_rate_C

-- Define the total work of the project
def total_work : ℝ := 1

-- Define the remaining work to be done by B
def remaining_work_B : ℝ := total_work - work_A_in_10_days - work_C_in_10_days

-- Define the number of days B worked
def days_B_worked : ℝ := remaining_work_B / work_rate_B

-- The theorem stating the number of days B worked
theorem B_worked_10_days : days_B_worked = 10 := 
sorry

end B_worked_10_days_l468_468242


namespace problem_l468_468296

-- Definitions based on the provided conditions
def frequency_varies (freq : Real) : Prop := true -- Placeholder definition
def probability_is_stable (prob : Real) : Prop := true -- Placeholder definition
def is_random_event (event : Type) : Prop := true -- Placeholder definition
def is_random_experiment (experiment : Type) : Prop := true -- Placeholder definition
def is_sum_of_events (event1 event2 : Prop) : Prop := event1 ∨ event2 -- Definition of sum of events
def mutually_exclusive (A B : Prop) : Prop := ¬(A ∧ B) -- Definition of mutually exclusive events
def complementary_events (A B : Prop) : Prop := A ↔ ¬B -- Definition of complementary events
def equally_likely_events (events : List Prop) : Prop := true -- Placeholder definition

-- Translation of the questions and correct answers
theorem problem (freq prob : Real) (event experiment : Type) (A B : Prop) (events : List Prop) :
  (¬(frequency_varies freq = probability_is_stable prob)) ∧ -- 1
  ((is_random_event event) ≠ (is_random_experiment experiment)) ∧ -- 2
  (probability_is_stable prob) ∧ -- 3
  (is_sum_of_events A B) ∧ -- 4
  (mutually_exclusive A B → ¬(probability_is_stable (1 - prob))) ∧ -- 5
  (¬(equally_likely_events events)) :=  -- 6
by
  sorry

end problem_l468_468296


namespace optimal_selection_method_uses_golden_ratio_l468_468575

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l468_468575


namespace obtuse_angle_vens_l468_468205

theorem obtuse_angle_vens (vens_in_circle : ℕ) (degrees_in_circle : ℕ) (angle_deg : ℕ) : 
  vens_in_circle = 600 ∧ degrees_in_circle = 360 ∧ angle_deg = 120 →
  (angle_deg * (vens_in_circle / degrees_in_circle) = 200) :=
by
  intro h
  cases h with 
    h1 h2
  cases h2 with
    h2 h3
  sorry

end obtuse_angle_vens_l468_468205


namespace sin_80_eq_neg_k_div_sqrt_1_add_k_squared_l468_468352

noncomputable def sin_identity (k : ℝ) : ℝ :=
  if h: (tan 100 = k) then -k / sqrt (1 + k^2) else 0

theorem sin_80_eq_neg_k_div_sqrt_1_add_k_squared (k : ℝ) (hk : tan 100 = k) : 
  sin 80 = - k / sqrt (1 + k^2) :=
by
  sorry

end sin_80_eq_neg_k_div_sqrt_1_add_k_squared_l468_468352


namespace sum_coordinates_is_60_l468_468937

theorem sum_coordinates_is_60 :
  let points := [(5 + Real.sqrt 91, 13), (5 - Real.sqrt 91, 13), (5 + Real.sqrt 91, 7), (5 - Real.sqrt 91, 7)]
  let x_coords_sum := (5 + Real.sqrt 91) + (5 - Real.sqrt 91) + (5 + Real.sqrt 91) + (5 - Real.sqrt 91)
  let y_coords_sum := 13 + 13 + 7 + 7
  x_coords_sum + y_coords_sum = 60 :=
by
  sorry

end sum_coordinates_is_60_l468_468937


namespace optimal_selection_uses_golden_ratio_l468_468819

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l468_468819


namespace optimal_selection_method_uses_golden_ratio_l468_468570

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l468_468570


namespace optimal_selection_method_uses_golden_ratio_l468_468680

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468680


namespace fraction_of_menu_l468_468136

def total_dishes (total : ℕ) : Prop := 
  6 = (1/4:ℚ) * total

def vegan_dishes (vegan : ℕ) (soy_free : ℕ) : Prop :=
  vegan = 6 ∧ soy_free = vegan - 5

theorem fraction_of_menu (total vegan soy_free : ℕ) (h1 : total_dishes total)
  (h2 : vegan_dishes vegan soy_free) : (soy_free:ℚ) / total = 1 / 24 := 
by sorry

end fraction_of_menu_l468_468136


namespace slope_probability_sum_l468_468463

noncomputable def probability_geq_slope (P : ℝ × ℝ) : (ℚ × ℚ) :=
  let square := set.Icc (0, 0) (1, 1)
  let region := {p : ℝ × ℝ | 4 * p.snd - p.fst ≥ 1 ∧ p ∈ square}
  let area_region := ∫⁻ p in region, (1 : ℝ)
  let area_square := 1
  (area_region / area_square).num_denom

theorem slope_probability_sum (m n : ℚ) (hmn : probability_geq_slope (0, 0) = (m, n)) :
  m + n = 7 :=
sorry

end slope_probability_sum_l468_468463


namespace optimal_selection_method_uses_golden_ratio_l468_468759

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l468_468759


namespace triangle_inequality_sqrt_sum_three_l468_468381

theorem triangle_inequality_sqrt_sum_three
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b) :
  3 ≤ (Real.sqrt (a / (-a + b + c)) + 
       Real.sqrt (b / (a - b + c)) + 
       Real.sqrt (c / (a + b - c))) := 
sorry

end triangle_inequality_sqrt_sum_three_l468_468381


namespace hem_dress_time_time_to_hem_dress_is_10_minutes_l468_468454

theorem hem_dress_time (h_length_ft : ℕ) (stitch1_len_in : ℚ) (stitch2_len_in : ℚ) 
  (stitches_per_min : ℕ) (switch_time_s : ℕ) : ℕ :=
  let h_length_in := h_length_ft * 12 in
  let cycle_len_in := stitch1_len_in + stitch2_len_in in
  let cycles_needed := (h_length_in : ℚ) / cycle_len_in in
  let cycles_per_min := stitches_per_min / 2 in
  let time_stitching := cycles_needed / cycles_per_min in
  let switch_time_min := switch_time_s * (1 / 60 : ℚ) in
  let num_switches := cycles_needed - 1 in
  let time_switching := num_switches * switch_time_min in
  let total_time := time_stitching + time_switching in
  total_time.round -- rounding to the nearest minute

theorem time_to_hem_dress_is_10_minutes : 
  hem_dress_time 3 (1/4 : ℚ) (3/8 : ℚ) 24 5 = 10 :=
sorry

end hem_dress_time_time_to_hem_dress_is_10_minutes_l468_468454


namespace optimal_selection_method_uses_golden_ratio_l468_468574

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l468_468574


namespace sin_cos_identity_l468_468321

theorem sin_cos_identity (α : ℝ) (h : cos α - sin α = 1 / 2) : sin α * cos α = 3 / 8 :=
sorry

end sin_cos_identity_l468_468321


namespace minimum_number_of_girls_l468_468897

theorem minimum_number_of_girls (students boys girls : ℕ) 
(h1 : students = 20) (h2 : boys = students - girls)
(h3 : ∀ d, 0 ≤ d → 2 * (d + 1) ≥ boys) 
: 6 ≤ girls :=
by 
  have h4 : 20 - girls = boys, from h2.symm
  have h5 : 2 * (girls + 1) ≥ boys, from h3 girls (nat.zero_le girls)
  linarith

#check minimum_number_of_girls

end minimum_number_of_girls_l468_468897


namespace optimal_selection_method_uses_golden_ratio_l468_468676

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468676


namespace scientists_born_in_july_percentage_l468_468873

theorem scientists_born_in_july_percentage (total_scientists july_scientists : ℕ) (h_total : total_scientists = 120) (h_july : july_scientists = 15) :
  (july_scientists : ℚ) / total_scientists * 100 = 12.5 :=
by
  have frac_eq : (july_scientists : ℚ) / total_scientists = 15 / 120 := by rw [h_july, h_total]
  have percent_eq : 15 / 120 * 100 = 12.5 := by norm_num
  rw [frac_eq, percent_eq]
  sorry

end scientists_born_in_july_percentage_l468_468873


namespace hua_luogeng_optimal_selection_method_l468_468738

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l468_468738


namespace optimal_selection_method_is_golden_ratio_l468_468584

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l468_468584


namespace minimum_number_of_girls_l468_468899

theorem minimum_number_of_girls (students boys girls : ℕ) 
(h1 : students = 20) (h2 : boys = students - girls)
(h3 : ∀ d, 0 ≤ d → 2 * (d + 1) ≥ boys) 
: 6 ≤ girls :=
by 
  have h4 : 20 - girls = boys, from h2.symm
  have h5 : 2 * (girls + 1) ≥ boys, from h3 girls (nat.zero_le girls)
  linarith

#check minimum_number_of_girls

end minimum_number_of_girls_l468_468899


namespace knicks_equal_knocks_l468_468050

theorem knicks_equal_knocks :
  (8 : ℝ) / (3 : ℝ) * (5 : ℝ) / (6 : ℝ) * (30 : ℝ) = 66 + 2 / 3 :=
by
  sorry

end knicks_equal_knocks_l468_468050


namespace mean_home_runs_l468_468175

theorem mean_home_runs :
  let players_with_5 := 3
  let players_with_6 := 4
  let players_with_8 := 2
  let players_with_9 := 1
  let players_with_11 := 1
  let total_home_runs := (5 * players_with_5) + (6 * players_with_6) + (8 * players_with_8) + (9 * players_with_9) + (11 * players_with_11)
  let total_players := players_with_5 + players_with_6 + players_with_8 + players_with_9 + players_with_11
  (total_home_runs / total_players : ℚ) = 75 / 11 :=
by
  sorry

end mean_home_runs_l468_468175


namespace optimal_selection_method_uses_golden_ratio_l468_468671

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468671


namespace optimal_selection_method_uses_golden_ratio_l468_468677

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468677


namespace optimal_selection_method_uses_golden_ratio_l468_468568

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l468_468568


namespace sum_of_x_satisfying_equation_l468_468117

-- Define the function f
def f (x : ℝ) : ℝ := 12 * x + 5

-- The theorem stating the sum of all x that satisfy the equation is 65
theorem sum_of_x_satisfying_equation :
  let g (x : ℝ) : ℝ := f⁻¹ (x)
  ∑ x in { x : ℝ | g x = f ((3 * x)⁻¹)}, x = 65 :=
sorry

end sum_of_x_satisfying_equation_l468_468117


namespace optimal_selection_method_uses_golden_ratio_l468_468620

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l468_468620


namespace optimal_selection_method_uses_golden_ratio_l468_468753

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l468_468753


namespace max_circle_size_l468_468996

/-- Define the property that a list of natural numbers satisfies the given conditions. -/
def valid_circle (nums : List ℕ) : Prop :=
  ∀ i, 0 ≤ i → i < nums.length →
  nums.get ⟨i, by simp [nat.add_sub_cancel]⟩ + nums.get ⟨(i + 1) % nums.length, nat.mod_lt _ (nat.lt_of_le_of_lt (nat.zero_le i) (nat.lt_of_lt_of_le i (List.length_pred nums nums.length)))⟩ % 3 = 0

/-- The maximum possible value of N such that N different natural numbers (none exceeding 1000)
    written in a circle have the property that the sum of any two adjacent numbers is divisible by 3, is 664. -/
theorem max_circle_size : ∃ (N : ℕ), N = 664 ∧ 
  ∃ (nums : List ℕ) (h1 : nums.length = N), 
  (∀ n ∈ nums, n ≤ 1000) ∧ (List.nodup nums) ∧ valid_circle nums := 
begin
  sorry
end

end max_circle_size_l468_468996


namespace find_ellipse_and_area_l468_468357

noncomputable def ellipse_standard_eq (a b : ℝ) (h1 : a > b > 0) (h2 : a = 2) (h3 : b^2 = 3) : Prop :=
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 / 3 = 1)

noncomputable def triangle_OMN_area (y1 y2 : ℝ) (h1 : y1 + y2 = -2 * real.sqrt 3 / 5) (h2 : y1 * y2 = -9 / 5) : ℝ :=
  1 / 2 * 1 * real.abs (y1 - y2)

theorem find_ellipse_and_area :
  ∃ (a b : ℝ) (h1 : a > b > 0) (h2 : a = 2) (h3 : b^2 = 3) (y1 y2 : ℝ)
    (hy1 : y1 + y2 = -2 * real.sqrt 3 / 5) (hy2 : y1 * y2 = -9 / 5),
    ellipse_standard_eq a b h1 h2 h3 ∧ triangle_OMN_area y1 y2 hy1 hy2 = 4 * real.sqrt 3 / 5 :=
  sorry

end find_ellipse_and_area_l468_468357


namespace final_money_after_bets_l468_468249

theorem final_money_after_bets (initial_money : ℕ) (num_bets : ℕ) (num_wins : ℕ) (num_losses : ℕ)
  (multiplier_win : ℚ) (multiplier_loss : ℚ) (final_money : ℚ) :
  initial_money = 120 →
  num_bets = 8 →
  num_wins = 4 →
  num_losses = 4 →
  multiplier_win = 3 / 2 →
  multiplier_loss = 2 / 3 →
  final_money = initial_money * (multiplier_win ^ num_wins) * (multiplier_loss ^ num_losses) →
  final_money = 120 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6] at h7
  have calc (initial_money : ℚ) * (multiplier_win ^ num_wins) * (multiplier_loss ^ num_losses) = 
            120 * ((3 / 2) ^ 4) * ((2 / 3) ^ 4) : rfl
  have eq1 : ((3 / 2 : ℚ) ^ 4) = 81 / 16 := by norm_num
  have eq2 : ((2 / 3 : ℚ) ^ 4) = 16 / 81 := by norm_num
  rw [eq1, eq2] at calc
  linarith
  sorry

end final_money_after_bets_l468_468249


namespace bike_owners_without_car_l468_468070

variable (T B C : ℕ) (H1 : T = 500) (H2 : B = 450) (H3 : C = 200)

theorem bike_owners_without_car (total bike_owners car_owners : ℕ) 
  (h_total : total = 500) (h_bike_owners : bike_owners = 450) (h_car_owners : car_owners = 200) : 
  (bike_owners - (bike_owners + car_owners - total)) = 300 := by
  sorry

end bike_owners_without_car_l468_468070


namespace sum_of_positive_integers_reverse_base5_base9_l468_468313

theorem sum_of_positive_integers_reverse_base5_base9 :
  ∑ n in (Finset.filter (λ n, ∃ a₀ a₁, 0 ≤ a₀ ∧ a₀ ≤ 4 ∧ 0 ≤ a₁ ∧ a₁ ≤ 8 ∧ n = 5 * a₁ + a₀ ∧ n = 9 * a₀ + a₁) (Finset.range 100)), n = 110 := by
  sorry

end sum_of_positive_integers_reverse_base5_base9_l468_468313


namespace simplify_fraction_l468_468156

-- Define the numbers involved and state their GCD
def num1 := 90
def num2 := 8100

-- State the GCD condition using a Lean 4 statement
def gcd_condition (a b : ℕ) := Nat.gcd a b = 90

-- Define the original fraction and the simplified fraction
def original_fraction := num1 / num2
def simplified_fraction := 1 / 90

-- State the proof problem that the original fraction simplifies to the simplified fraction
theorem simplify_fraction : gcd_condition num1 num2 → original_fraction = simplified_fraction := 
by
  sorry

end simplify_fraction_l468_468156


namespace mark_eggs_supply_l468_468486

theorem mark_eggs_supply 
  (dozens_per_day : ℕ)
  (eggs_per_dozen : ℕ)
  (extra_eggs : ℕ)
  (days_in_week : ℕ)
  (H1 : dozens_per_day = 5)
  (H2 : eggs_per_dozen = 12)
  (H3 : extra_eggs = 30)
  (H4 : days_in_week = 7) :
  (dozens_per_day * eggs_per_dozen + extra_eggs) * days_in_week = 630 :=
by
  sorry

end mark_eggs_supply_l468_468486


namespace angle_bisectors_triangle_l468_468444

theorem angle_bisectors_triangle
  (A B C I D K E : Type)
  (triangle : ∀ (A B C : Type), Prop)
  (is_incenter : ∀ (I A B C : Type), Prop)
  (is_on_arc_centered_at : ∀ (X Y : Type), Prop)
  (is_altitude_intersection : ∀ (X Y : Type), Prop)
  (angle_BIC : ∀ (B C : Type), ℝ)
  (angle_DKE : ∀ (D K E : Type), ℝ)
  (α β γ : ℝ)
  (h_sum_ang : α + β + γ = 180) :
  is_incenter I A B C →
  is_on_arc_centered_at D A → is_on_arc_centered_at K A → is_on_arc_centered_at E A →
  is_altitude_intersection E A →
  angle_BIC B C = 180 - (β + γ) / 2 →
  angle_DKE D K E = (360 - α) / 2 →
  angle_BIC B C + angle_DKE D K E = 270 :=
by sorry

end angle_bisectors_triangle_l468_468444


namespace shortest_side_of_triangle_l468_468180

theorem shortest_side_of_triangle (a b c : ℕ) (h₀ : a = 21) (h₁ : a + b + c = 48)
  (h₂ : sqrt ((24 : ℚ) * (24 - 21) * (24 - b) * (24 - c)) ∈ ℤ) : b = 10 :=
  sorry

end shortest_side_of_triangle_l468_468180


namespace optimal_selection_method_uses_golden_ratio_l468_468745

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l468_468745


namespace volume_center_sum_eq_l468_468460

noncomputable def P := \sum_{i=0}^{n} \lambda_i P_i

variables (n : ℕ) (P : ℝ^n)
variables (P : ℝ^n → ℝ^n ) (v C : set (ℝ^n) → ℝ) 

def simplex (S : list ℝ) : Prop := 
  length S = n + 1 ∧ ¬affine_independent ℝ S

def inside_simplex (S : list ℝ) (P : ℝ^n) : Prop :=
  ∑ λ_i |= λ_i_λ_i * vertex_S[i i] == P = Σ_gen_V = 1 ∧ λ_i>0 

def replace_vertex (V : list ℝ) (i : ℕ) (P : ℝ^n) : list ℝ :=
  V.update i P

theorem volume_center_sum_eq {S : list ℝ} (hS : simplex S) (P : ℝ^ n)
  (inside_simplex S P) :
  ∑ i in range (n+1), v (replace_vertex S i P) * C (replace_vertex S i P)
  = v S * C S :=
sorry

end volume_center_sum_eq_l468_468460


namespace sum_of_valid_x_values_l468_468197

theorem sum_of_valid_x_values :
  let total_students := 360
  let min_rows := 12
  let min_students_per_row := 18
  let possible_x_values := {x | x >= min_students_per_row ∧ total_students / x >= min_rows ∧ total_students % x = 0}
  sum possible_x_values = 92 :=
by
  let total_students := 360
  let min_rows := 12
  let min_students_per_row := 18
  let possible_x_values := {x | x >= min_students_per_row ∧ total_students / x >= min_rows ∧ total_students % x = 0}
  have x_vals : List ℕ := possible_x_values.toList
  exact List.sum x_vals = 92

end sum_of_valid_x_values_l468_468197


namespace optimal_selection_method_uses_golden_ratio_l468_468604

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l468_468604


namespace train_length_l468_468219

theorem train_length
  (speed_kmph : ℕ)
  (platform_length : ℕ)
  (time_seconds : ℕ)
  (speed_m_per_s : speed_kmph * 1000 / 3600 = 20)
  (distance_covered : 20 * time_seconds = 520)
  (platform_eq : platform_length = 280)
  (time_eq : time_seconds = 26) :
  ∃ L : ℕ, L = 240 := by
  sorry

end train_length_l468_468219


namespace optimal_selection_method_uses_golden_ratio_l468_468755

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l468_468755


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468762

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468762


namespace arithmetic_sequence_sum_l468_468173

theorem arithmetic_sequence_sum :
  let a : ℕ → ℤ := λ n, 1 + (n - 1) * 2 in
  a 1 + a 2 + a 3 + a 4 + a 5 = 25 :=
by
  let a : ℕ → ℤ := λ n, 1 + (n - 1) * 2
  show a 1 + a 2 + a 3 + a 4 + a 5 = 25
  sorry

end arithmetic_sequence_sum_l468_468173


namespace min_number_of_girls_l468_468887

theorem min_number_of_girls (d : ℕ) (students : ℕ) (boys : ℕ → ℕ) : 
  students = 20 ∧ ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → boys i ≠ boys j ∨ boys j ≠ boys k ∨ boys i ≠ boys k → d = 6 :=
by
  sorry

end min_number_of_girls_l468_468887


namespace sum_of_numbers_base5_reverse_base9_l468_468311

theorem sum_of_numbers_base5_reverse_base9 : 
  ∑ n in { n : ℕ | ∃ (d : ℕ) (a : ℕ → ℕ),
              (∀ i, i ≤ d → a i < 5) ∧
              (∀ i, i ≤ d → n = n + 5^i * a (d - i)) ∧
              (∀ i, i ≤ d → n = n + 9^i * a i)
          }, n = 31 :=
by
  sorry

end sum_of_numbers_base5_reverse_base9_l468_468311


namespace possible_lightest_boy_heaviest_girl_l468_468426

theorem possible_lightest_boy_heaviest_girl :
  ∃ (B G : ℕ) (boys_weights girls_weights : list ℕ),
    B + G < 35 ∧
    (∑ w in boys_weights, w) / B = 60 ∧
    (∑ w in girls_weights, w) / G = 47 ∧
    (∑ w in (boys_weights ++ girls_weights), w) / (B + G) = 53.5 ∧
    (∃ l, l ∈ boys_weights ∧ ∀ g, g ∈ girls_weights → l < g) :=
begin
  sorry,
end

end possible_lightest_boy_heaviest_girl_l468_468426


namespace eccentricity_of_ellipse_l468_468167

noncomputable def ellipse_eccentricity (a b : ℝ) (h : a = 3 * b) : ℝ :=
  let c := real.sqrt (a^2 - b^2)
  (c / a)

theorem eccentricity_of_ellipse (b : ℝ) (hb : 0 < b) :
  ellipse_eccentricity (3 * b) b (by linarith) = 2 * real.sqrt 2 / 3 :=
by 
  let a := 3 * b
  have ha : a = 3 * b := by rfl
  dsimp [ellipse_eccentricity]
  rw [ha]
  have h : real.sqrt (a^2 - b^2) = 2 * real.sqrt 2 * b := sorry
  rw [h]
  field_simp
  ring

#check eccentricity_of_ellipse

end eccentricity_of_ellipse_l468_468167


namespace optimal_selection_uses_golden_ratio_l468_468660

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l468_468660


namespace triangles_overlap_area_l468_468290

-- Define the points for the triangles
structure Point where
  x : ℝ
  y : ℝ

-- Define the triangles using their vertices
def Triangle1 : List Point := [⟨0, 0⟩, ⟨3, 0⟩, ⟨1, 2⟩]
def Triangle2 : List Point := [⟨3, 2⟩, ⟨0, 1⟩, ⟨1, 0⟩]

-- The theorem to prove the area of the region where the triangles overlap
theorem triangles_overlap_area : 
  let overlap_area := 1.5 in
  overlap_area = 1.5 := sorry

end triangles_overlap_area_l468_468290


namespace percentage_loss_is_correct_l468_468496

noncomputable def initial_cost : ℝ := 300
noncomputable def selling_price : ℝ := 255
noncomputable def loss : ℝ := initial_cost - selling_price
noncomputable def percentage_loss : ℝ := (loss / initial_cost) * 100

theorem percentage_loss_is_correct :
  percentage_loss = 15 :=
sorry

end percentage_loss_is_correct_l468_468496


namespace angle_A_eq_pi_div_3_area_of_triangle_l468_468068

variable {A B C a b c : ℝ}

-- Condition 1: Given trigonometric equation holds.
def trig_eq_1 (A : ℝ) : Prop := 
  2 * sin (2 * A) * cos A - sin (3 * A) + sqrt 3 * cos A = sqrt 3

-- Condition 2: Sides and angle relationship.
def side_condition (a A B C : ℝ) : Prop := 
  (a = 1) ∧ (sin A + sin (B - C) = 2 * sin (2 * C))

-- Solution part 1: Prove measure of angle A.
theorem angle_A_eq_pi_div_3 (h : trig_eq_1 A) : A = π / 3 :=
  sorry

-- Solution part 2: Prove the area of triangle ABC.
theorem area_of_triangle (h₁ : side_condition a (π / 3) B C) (h₂ : a = 1) : 
  sin (π / 3) * (sqrt 3 / 3) = sqrt 3 / 6 :=
  sorry

end angle_A_eq_pi_div_3_area_of_triangle_l468_468068


namespace possible_lightest_boy_heaviest_girl_l468_468425

theorem possible_lightest_boy_heaviest_girl :
  ∃ (B G : ℕ) (boys_weights girls_weights : list ℕ),
    B + G < 35 ∧
    (∑ w in boys_weights, w) / B = 60 ∧
    (∑ w in girls_weights, w) / G = 47 ∧
    (∑ w in (boys_weights ++ girls_weights), w) / (B + G) = 53.5 ∧
    (∃ l, l ∈ boys_weights ∧ ∀ g, g ∈ girls_weights → l < g) :=
begin
  sorry,
end

end possible_lightest_boy_heaviest_girl_l468_468425


namespace number_of_terms_added_l468_468204

-- Define the sequence p(n)
def p (n : ℕ) : ℝ := (Finset.range n).sum (λ i => 1 / (2^(i+1)))

-- Theorem to prove the number of terms added from n = k to n = k+1 is 2^k
theorem number_of_terms_added (k : ℕ) : (Finset.Ico (2^k + 1) (2^(k+1) + 1)).card = 2^k := 
sorry

end number_of_terms_added_l468_468204


namespace matilda_percentage_loss_l468_468499

theorem matilda_percentage_loss (initial_cost selling_price : ℕ) (h_initial : initial_cost = 300) (h_selling : selling_price = 255) :
  ((initial_cost - selling_price) * 100) / initial_cost = 15 :=
by
  rw [h_initial, h_selling]
  -- Proceed with the proof
  sorry

end matilda_percentage_loss_l468_468499


namespace final_result_l468_468533

theorem final_result 
  (a b c : ℝ)
  (h1 : a + b = 37) 
  (h2 : b + c = 58) 
  (h3 : c + a = 72) : 
  (a + b + c) - 10 = 73.5 := 
by 
  calc
    a + b + c  = (a + b + c)              : by rfl
            ... = 83.5                     : by 
            {
              have h : (a + b) + (b + c) + (c + a) = 37 + 58 + 72, 
              { rw [h1, h2, h3] },
              rw [add_assoc, add_assoc, add_left_comm a, ← add_assoc, add_comm b, add_assoc c a],
              rw [add_assoc (37 : ℝ), add_comm 58, add_assoc, add_right_comm 72, ← add_assoc],
              norm_num at h,
              rw ← two_mul at h,
              norm_num at h,
              exact h
            }
            ... - 10 = 83.5 - 10           : by refl
            ... = 73.5                     : by norm_num

end final_result_l468_468533


namespace proof_problem_l468_468001

-- Definitions representing planes, lines, points, and perpendicularity/parallelism relationships
variable (A : Point) (a b c : Line) (M N β : Plane)

-- Conditions from the problem (①, ②, ③, ④)
axiom cond1 {a : Line} {M N : Plane} (h1 : a ⊥ M) (h2 : M ⊥ N) : ¬ (a ∥ N)
axiom cond2 {a b c : Line} {M : Plane} (h1 : a ⊥ M) (h2 : b ∥ M) (h3 : c ∥ a) : a ⊥ b ∧ c ⊥ b
axiom cond3 {a b : Line} {M : Plane} (h1 : a ⊥ M) (h2 : ¬ (b ⊆ M)) (h3 : b ∥ M) : b ⊥ a
axiom cond4 {a b c : Line} {β : Plane} (h1 : a ⊆ β) (h2 : b ∩ β = A) (h3 : c = projection b β) (h4 : a ⊥ c) : a ⊥ b

open _root_.classical

-- Prove that propositions ②, ③, ④ are correct
theorem proof_problem :
  cond2 ∧ cond3 ∧ cond4 :=
by 
  sorry

end proof_problem_l468_468001


namespace brianne_yard_is_6_times_larger_l468_468293

variables (Derrick_yard_length : ℕ) (Alex_yard_length : ℕ) (Brianne_yard_length : ℕ)

def Derrick_yard_length_condition : Derrick_yard_length = 10 := 
  rfl

def Alex_yard_length_condition : Alex_yard_length = Derrick_yard_length / 2 := 
  rfl

def Brianne_yard_length_condition : Brianne_yard_length = 30 := 
  rfl

def brianne_yard_larger_than_alex_yard (Derrick_yard_length Alex_yard_length Brianne_yard_length : ℕ) 
  (h1 : Derrick_yard_length = 10)
  (h2 : Alex_yard_length = Derrick_yard_length / 2)
  (h3 : Brianne_yard_length = 30) : Prop := 
  Brianne_yard_length = 6 * Alex_yard_length

theorem brianne_yard_is_6_times_larger 
  (Derrick_yard_length Alex_yard_length Brianne_yard_length : ℕ)
  (h1 : Derrick_yard_length = 10)
  (h2 : Alex_yard_length = Derrick_yard_length / 2)
  (h3 : Brianne_yard_length = 30) : 
  brianne_yard_larger_than_alex_yard Derrick_yard_length Alex_yard_length Brianne_yard_length h1 h2 h3 :=
  sorry

end brianne_yard_is_6_times_larger_l468_468293


namespace money_distribution_l468_468260

theorem money_distribution (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 310) (h3 : C = 10) : A + B + C = 500 :=
by
  sorry

end money_distribution_l468_468260


namespace hua_luogeng_optimal_selection_method_l468_468787

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l468_468787


namespace optimal_selection_method_uses_golden_ratio_l468_468623

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l468_468623


namespace possible_lightest_boy_heaviest_girl_l468_468420

theorem possible_lightest_boy_heaviest_girl
    (B G : ℕ) 
    (wb : ℕ → ℕ) 
    (wg : ℕ → ℕ) 
    (N < 35) 
    (avg_weight_students : Real) 
    (avg_weight_girls : Real) 
    (avg_weight_boys : Real) : 
    (avg_weight_students = 53.5) ∧ 
    (avg_weight_girls = 47) ∧ 
    (avg_weight_boys = 60) → 
    (∃ i j : ℕ, (i < B) ∧ (j < G) ∧ (wb i < wg j)) :=
by 
  sorry

end possible_lightest_boy_heaviest_girl_l468_468420


namespace line_perp_plane_and_parallel_plane_implies_planes_perp_l468_468391

open Set

variable (l : Line) (α β : Plane)

def perpend (a b : Plane) : Prop := ∀ (x ∈ a) (y ∈ b), x ≠ y → x - y ∉ span ℝ ({a.normal} : Set (ℝ^3))
def parallel (a b : Plane) : Prop := ∀ (x ∈ a) (y ∈ b), x ≠ y → x - y ∈ span ℝ ({a.normal})

theorem line_perp_plane_and_parallel_plane_implies_planes_perp  
  (h1 : ∃ (α β : Plane), α ≠ β ∧ l ∈ α ∧ l ∈ β)
  (h2 : l ∈ α ∧ l ∈ β)
  (h3 : ∀ (l ∈ α), perpend l α)
  (h4 : ∀ (l ∈ β), parallel l β)
  : perpend α β :=
sorry

end line_perp_plane_and_parallel_plane_implies_planes_perp_l468_468391


namespace optimal_selection_method_uses_golden_ratio_l468_468748

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l468_468748


namespace optimal_selection_method_uses_golden_ratio_l468_468567

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l468_468567


namespace count_polynomials_in_H_l468_468461

-- Definitions and conditions initialization
def polynomials_in_H (z : ℂ) (n: ℕ) (c : fin n → ℤ) : polynomial ℂ :=
  polynomial.C 36 + (∑ i in finset.range n, polynomial.C (c i) * polynomial.X ^ (n - 1 - i))

def is_distinct_roots (p : polynomial ℂ) (roots : fin (degree p) → ℂ) : Prop :=
  ∀ i j, i ≠ j → roots i ≠ roots j

def is_root_of_form (p : polynomial ℂ) (roots : fin (degree p) → ℂ) : Prop :=
  ∀ i, ∃ (a b : ℤ), roots i = a + b * complex.I

-- Theorem statement
theorem count_polynomials_in_H :
  ∃ (H : set (polynomial ℂ)),
    (∀ (p : polynomial ℂ), p ∈ H ↔ 
      ∃ (n : ℕ) (c : fin n → ℤ) (roots : fin (degree p) → ℂ),
        p = polynomials_in_H 1 n c ∧ 
        is_distinct_roots p roots ∧ 
        is_root_of_form p roots) ∧
    H.finite ∧
    H.card = 15 := 
sorry

end count_polynomials_in_H_l468_468461


namespace minimum_number_of_girls_l468_468896

theorem minimum_number_of_girls (students boys girls : ℕ) 
(h1 : students = 20) (h2 : boys = students - girls)
(h3 : ∀ d, 0 ≤ d → 2 * (d + 1) ≥ boys) 
: 6 ≤ girls :=
by 
  have h4 : 20 - girls = boys, from h2.symm
  have h5 : 2 * (girls + 1) ≥ boys, from h3 girls (nat.zero_le girls)
  linarith

#check minimum_number_of_girls

end minimum_number_of_girls_l468_468896


namespace exactly_two_overlap_l468_468194

-- Define the concept of rectangles
structure Rectangle :=
  (width : ℕ)
  (height : ℕ)

-- Define the given rectangles
def rect1 : Rectangle := ⟨4, 6⟩
def rect2 : Rectangle := ⟨4, 6⟩
def rect3 : Rectangle := ⟨4, 6⟩

-- Hypothesis defining the overlapping areas
def overlap1_2 : ℕ := 4 * 2 -- first and second rectangles overlap in 8 cells
def overlap2_3 : ℕ := 2 * 6 -- second and third rectangles overlap in 12 cells
def overlap1_3 : ℕ := 0    -- first and third rectangles do not directly overlap

-- Total overlap calculation
def total_exactly_two_overlap : ℕ := (overlap1_2 + overlap2_3)

-- The theorem we need to prove
theorem exactly_two_overlap (rect1 rect2 rect3 : Rectangle) : total_exactly_two_overlap = 14 := sorry

end exactly_two_overlap_l468_468194


namespace optimal_selection_method_uses_golden_ratio_l468_468847

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468847


namespace optimal_selection_method_uses_golden_ratio_l468_468617

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l468_468617


namespace hua_luogeng_optimal_selection_method_l468_468735

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l468_468735


namespace cosine_angle_between_a_b_l468_468351

variables (e1 e2 : ℝ^3) [normed_group ℝ e1] [normed_group ℝ e2]
variables (a b : ℝ^3)

def is_unit_vector (v : ℝ^3) := ∥v∥ = 1

variable (angle_60_deg_cos : ℝ := 1/2)

variables (a_def : a = 2 • e1 + e2)
variables (b_def : b = -3 • e1 + 2 • e2)
variables (e1_unit : is_unit_vector e1)
variables (e2_unit : is_unit_vector e2)
variables (angle_e1_e2 : inner_product_space.angle e1 e2 = 60 * (π / 180))

theorem cosine_angle_between_a_b : real_inner a b = -1 / 2 := by sorry

end cosine_angle_between_a_b_l468_468351


namespace no_four_primes_exist_l468_468145

theorem no_four_primes_exist (a b c d : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b)
  (hc : Nat.Prime c) (hd : Nat.Prime d) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : (1 / a : ℚ) + (1 / d) = (1 / b) + (1 / c)) : False := sorry

end no_four_primes_exist_l468_468145


namespace f_increasing_on_1_to_infty_f_max_min_on_1_to_4_l468_468360

def f (x : ℝ) : ℝ := x + 1 / x

theorem f_increasing_on_1_to_infty : ∀ {x1 x2 : ℝ}, 1 ≤ x1 → x1 < x2 → f x1 < f x2 :=
by sorry

theorem f_max_min_on_1_to_4 : 
  (∀ x ∈ set.Icc (1:ℝ) (4:ℝ), f x ≤ (17 / 4)) ∧ 
  (∃ x ∈ set.Icc (1:ℝ) (4:ℝ), f x = 17 / 4) ∧ 
  (∀ x ∈ set.Icc (1:ℝ) (4:ℝ), f x ≥ 2) ∧ 
  (∃ x ∈ set.Icc (1:ℝ) (4:ℝ), f x = 2) :=
by sorry

end f_increasing_on_1_to_infty_f_max_min_on_1_to_4_l468_468360


namespace cost_per_pouch_l468_468449

theorem cost_per_pouch (boxes : ℕ) (pouches_per_box : ℕ) (total_cost_dollars : ℕ) :
  boxes = 10 →
  pouches_per_box = 6 →
  total_cost_dollars = 12 →
  (total_cost_dollars * 100) / (boxes * pouches_per_box) = 20 :=
by
  intros,
  -- proof steps here
  sorry

end cost_per_pouch_l468_468449


namespace sum_of_series_l468_468300

variable {n : ℕ}

def a_n (k : ℕ) : ℕ := k^2 + k + 1

def S_n (n : ℕ) : ℕ :=
  2 * (∑ k in Finset.range (n + 1), (a_n k))

theorem sum_of_series (n : ℕ) :
  S_n n = n * (n + 1) * (2 * n + 10) / 3 := by
  sorry

end sum_of_series_l468_468300


namespace tangent_lines_slope_one_tangent_line_at_P_l468_468016

variable (x y : ℝ)

def curve_equation : ℝ → ℝ := fun x => (1 / 3) * x^3 + (4 / 3)

theorem tangent_lines_slope_one :
  let slope1_tangent1 (x y : ℝ) := (3 * x - 3 * y + 2 = 0)
  let slope1_tangent2 (x y : ℝ) := (x - y + 2 = 0)
  ∃ x₀ y₀, curve_equation x₀ = y₀ ∧ (y₀ = 5/3 ∧ x₀ = 1 ∨ y₀ = 1 ∧ x₀ = -1) ∧ 
           (slope1_tangent1 x₀ y₀ ∨ slope1_tangent2 x₀ y₀) :=
sorry

theorem tangent_line_at_P :
  let P : ℝ × ℝ := (2, 4)
  let slope2_tangent (x y : ℝ) := (4 * x - y - 4 = 0)
  curve_equation P.1 = P.2 → slope2_tangent P.1 P.2 :=
sorry

end tangent_lines_slope_one_tangent_line_at_P_l468_468016


namespace comparison_abc_l468_468063

open Real

-- Definitions as per conditions:
def f : ℝ → ℝ := sorry
axiom even_f : ∀ x, f x = f (-x)
axiom mono_decreasing_neg_inf_to_0 : ∀ x y, x ≤ 0 → y ≤ 0 → x < y → f x > f y

def a := f (log 2 3)
def b := f (log 4 5)
def c := f (2^(3/2))

-- The theorem to prove:
theorem comparison_abc : b < a ∧ a < c :=
by
  sorry

end comparison_abc_l468_468063


namespace noelle_homework_assignments_l468_468137

theorem noelle_homework_assignments : 
  (∀ n, 1 ≤ n ∧ n ≤ 10 → n * 1 = n) ∧
  (∀ n, 11 ≤ n ∧ n ≤ 15 → (n - 10) * 2 = (n - 10) * 2) ∧
  (∀ n, 16 ≤ n ∧ n ≤ 20 → (n - 15) * 3 = (n - 15) * 3) ∧
  (∀ n, 21 ≤ n ∧ n ≤ 25 → (n - 20) * 4 = (n - 20) * 4) ∧
  (∀ n, 26 ≤ n ∧ n ≤ 30 → (n - 25) * 5 = (n - 25) * 5) →
  let assignments := (10 * 1) + (5 * 2) + (5 * 3) + (5 * 4) + (5 * 5) in
  assignments = 80 :=
by sorry

end noelle_homework_assignments_l468_468137


namespace volume_of_pyramid_is_one_third_l468_468540

def is_right_triangle (P A B: ℝ) := ∠ P A B = π/2
def isSquareBase (ABCD: ℝ) := ∃ a, ∀ (A B C D: ℝ),
  dist A B = a ∧ dist B C = a ∧ dist C D = a ∧ dist D A = a ∧ dist A C = dist B D

theorem volume_of_pyramid_is_one_third 
  (h1 : isSquareBase ℝ)
  (h2 : ∀ P A B, is_right_triangle P A B)
  (height_apex : ∀ P (ABCD: ℝ), dist P ABCD = 1)
  (dihedral_angle_apex : ∀ P A, ∠ P A = 2 * π / 3) :
  (∃ V, V = 1/3) :=
sorry

end volume_of_pyramid_is_one_third_l468_468540


namespace perpendicular_lines_l468_468023

theorem perpendicular_lines :
  ∃ y x : ℝ, (4 * y - 3 * x = 15) ∧ (3 * y + 4 * x = 12) :=
by
  sorry

end perpendicular_lines_l468_468023


namespace optimal_selection_method_uses_golden_ratio_l468_468867

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l468_468867


namespace ratio_of_areas_l468_468942

open Classical
noncomputable theory

variable (s : ℝ)
variable (A B C D E F G H : ℝ × ℝ)

-- Definitions of points and ratios
def equilateral (A B C: ℝ × ℝ) := 
  dist A B = dist B C ∧ dist B C = dist C A

def trisect (A B C D E F : ℝ × ℝ) :=
  dist A D = dist D B ∧ dist B E = dist E C ∧ dist C F = dist F A

def midpoint (P Q M: ℝ × ℝ) :=
  dist P M = dist M Q

def area (A B C: ℝ × ℝ) : ℝ := 
  abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) 
        - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1)) / 2

-- Problem statement
theorem ratio_of_areas :
  equilateral A B C →
  trisect A B C D E F →
  midpoint D F G →
  midpoint E F H →
  let shaded_area := area D G H in
  let total_area := area A B C in
  let non_shaded_area := total_area - shaded_area in
  shaded_area / non_shaded_area = 1 / 11 :=
sorry

end ratio_of_areas_l468_468942


namespace sum_of_first_30_odd_natural_numbers_l468_468209

theorem sum_of_first_30_odd_natural_numbers : 
  (∑ k in finset.range 30, (2 * k + 1)) = 900 := 
by
  sorry

end sum_of_first_30_odd_natural_numbers_l468_468209


namespace ratio_su_ut_l468_468406

variables {P Q R T S U M : Type*}
variable [linear_ordered_field R]

def midpoint (A B : R × R) : R × R := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def condition_midpoint (P Q R : R × R) : Prop :=
P = midpoint Q R

def distance (A B : R × R) : R :=
real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def divides_externally (T P R : R × R) (m n : R) : Prop :=
T = ((m * R.1 + n * P.1) / (m + n), (m * R.2 + n * P.2) / (m + n))

def collinear (A B C : R × R) : Prop :=
∃ k : R, B.1 = A.1 + k * (C.1 - A.1) ∧ B.2 = A.2 + k * (C.2 - A.2)

noncomputable def SU_UT (S U T : R × R) : R :=
distance S U / distance U T

theorem ratio_su_ut (P Q R T S U M : R × R)
  (hp_mid : condition_midpoint P Q R)
  (hq : distance P Q = 15)
  (hr : distance P R = 20)
  (ht_ext : divides_externally T P R 1 3)
  (hs_pos : ∃ x : R, distance P S = x ∧ distance S Q = x * 2)
  (hu_intersect : collinear S U T ∧ collinear P U M) :
  SU_UT S U T = 1 / 4 := sorry

end ratio_su_ut_l468_468406


namespace vertical_asymptote_unique_l468_468963

-- Given definitions from conditions
def g (x : ℝ) (c : ℝ) : ℝ := (x^2 - 3 * x + c) / (x^2 - x - 12)

-- The statement to prove the correct answer
theorem vertical_asymptote_unique (c : ℝ) :
  (∃! (x : ℝ), (x^2 - x - 12) = 0 → x = 4 ∨ x = -3 ∧ (x^2 - 3 * x + c) ≠ 0)
  ↔ (c = -4 ∨ c = -18) := by
  sorry

end vertical_asymptote_unique_l468_468963


namespace optimal_selection_method_uses_golden_ratio_l468_468837

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468837


namespace number_of_constructed_spheres_l468_468375

variable (A1 A2 : Plane)
variable (varrho1 r1 varrho2 r2 : ℝ)
variable (T : Plane)

theorem number_of_constructed_spheres : 
  ∃ O1 O2 : Point,
    (distance_from_center O1 A1 = sqrt (r1^2 - varrho1^2)) ∧
    (distance_from_center O2 A2 = sqrt (r2^2 - varrho2^2)) ∧
    (distance_between_centers O1 O2 = abs (r1 ± r2)) ∧
    (has_common_tangent_plane_parallel_to O1 O2 T) ∧
    (touches_projection_plane O1 projection_plane) ∧
    (number_of_constructed_solutions = 64) :=
  sorry

end number_of_constructed_spheres_l468_468375


namespace candidate_votes_percentage_l468_468243

theorem candidate_votes_percentage
  (P : ℕ → Prop) (cast_votes : ℕ) (lost_by : ℕ) (candidate_got_percentage : ℕ) :
  (cast_votes = 8000) →
  (lost_by = 4000) →
  let candidate_votes := (candidate_got_percentage / 100) * cast_votes in
  let rival_votes := candidate_votes + lost_by in
  rival_votes = cast_votes →
  candidate_got_percentage = 25 :=
by
  intros H_cast_votes H_lost_by
  simp [H_cast_votes, H_lost_by]
  sorry

end candidate_votes_percentage_l468_468243


namespace count_false_propositions_l468_468508

-- Define the propositions as Lean definitions
def prop₁ : Prop := ∀ (T : Triangle) (P : Plane), (PlaneParallelToSides T P) → (PlaneParallelToThirdSide T P)
def prop₂ : Prop := ∀ (T : Triangle) (L : Line), (LinePerpendicularToSides T L) → (LinePerpendicularToThirdSide T L)
def prop₃ : Prop := ∀ (T : Triangle) (P : Plane), (PlaneEquidistantFromVertices T P) → (PlaneParallelToPlaneContainingTriangle T P)

-- Define the main theorem to be proved
theorem count_false_propositions : (if prop₁ ∧ prop₂ ∧ ¬prop₃ then 1 else if ¬prop₁ ∧ prop₂ ∧ ¬prop₃ then 2 else if ¬prop₁ ∧ ¬prop₂ ∧ ¬prop₃ then 3 else 0) = 1 :=
sorry

end count_false_propositions_l468_468508


namespace find_ellipse_equation_l468_468006

variable (a b c : ℝ)
variable (F1A : ℝ)

-- Conditions
axiom h1 : a > b ∧ b > 0
axiom h2 : F1A = (Real.sqrt 10) + (Real.sqrt 5)
axiom h3 : a = Real.sqrt 2 * c
axiom h4 : a + c = Real.sqrt 10 + Real.sqrt 5

-- Conclusion to prove
theorem find_ellipse_equation (h1 h2 h3 h4): (a = Real.sqrt 10) ∧ (b = Real.sqrt 5) ∧ (∀ x y : ℝ, (x^2) / 10 + (y^2) / 5 = 1) :=
  sorry

end find_ellipse_equation_l468_468006


namespace optimal_selection_method_is_golden_ratio_l468_468588

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l468_468588


namespace quotient_is_8_l468_468955

def dividend : ℕ := 125
def divisor : ℕ := 15
def remainder : ℕ := 5

theorem quotient_is_8 (dividend = 125) (divisor = 15) (remainder = 5) : (dividend - remainder) / divisor = 8 := 
by sorry

end quotient_is_8_l468_468955


namespace optimal_selection_golden_ratio_l468_468694

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l468_468694


namespace perpendicular_lines_l468_468403

theorem perpendicular_lines (a : ℝ) : 
  (2 * (a + 1) * a + a * 2 = 0) ↔ (a = -2 ∨ a = 0) :=
by 
  sorry

end perpendicular_lines_l468_468403


namespace optimal_selection_method_uses_golden_ratio_l468_468622

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l468_468622


namespace calc_area_FDBG_l468_468202

structure Triangle (α : Type _) :=
(A B C : α)

structure Point (α : Type _) :=
(x y : α)

def midpoint {α : Type _} [has_add α] [has_div α] (p1 p2 : Point α) : Point α :=
{ x := (p1.x + p2.x) / 2,
  y := (p1.y + p2.y) / 2 }

noncomputable def sin (x : ℝ) : ℝ := sorry

noncomputable def area_of_triangle (A B C : Point ℝ) : ℝ :=
0.5 * (B.x - A.x) * (C.y - A.y) - 0.5 * (B.y - A.y) * (C.x - A.x)

noncomputable def area_of_quadrilateral (A B C D : Point ℝ) : ℝ :=
area_of_triangle A B C + area_of_triangle A C D

theorem calc_area_FDBG :
  let A := Point.mk 0 0,
      B := Point.mk 40 0,
      C := Point.mk 20 (2 * (100 / 20)), -- derived from area calculation and geometry
      D := midpoint A B,
      E := midpoint A C,
      F := Point.mk sorry sorry, -- coordinates derived from intersect info
      G := Point.mk sorry sorry, -- coordinates derived from intersect info
      FDBG := area_of_quadrilateral F D B G
  in FDBG = 16.67 := sorry

end calc_area_FDBG_l468_468202


namespace expression_value_l468_468959

theorem expression_value : 6^3 - 4 * 6^2 + 4 * 6 - 1 = 95 :=
by
  sorry

end expression_value_l468_468959


namespace optimal_selection_uses_golden_ratio_l468_468657

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l468_468657


namespace daily_wage_c_l468_468220

theorem daily_wage_c (a_days b_days c_days total_earnings : ℕ)
  (ratio_a_b ratio_b_c : ℚ)
  (a_wage b_wage c_wage : ℚ) :
  a_days = 6 →
  b_days = 9 →
  c_days = 4 →
  total_earnings = 1480 →
  ratio_a_b = 3 / 4 →
  ratio_b_c = 4 / 5 →
  b_wage = ratio_a_b * a_wage → 
  c_wage = ratio_b_c * b_wage → 
  a_days * a_wage + b_days * b_wage + c_days * c_wage = total_earnings →
  c_wage = 100 / 3 :=
by
  intros
  sorry

end daily_wage_c_l468_468220


namespace polynomial_zero_check_l468_468252

theorem polynomial_zero_check {P : Polynomial ℤ} (α β : ℤ) :
  P = Polynomial.monomial 4 1 + Polynomial.monomial 3 (α - 4) + Polynomial.monomial 2 (β - 4 * α + 3) +
       Polynomial.monomial 1 (some_integer_coefficient) + Polynomial.monomial 0 (some_integer_constant) →
  P.has_root (Complex.of_real (-3 / 2) + Complex.I * Complex.of_real (sqrt 15 / 2)) :=
begin
  sorry
end

end polynomial_zero_check_l468_468252


namespace mark_egg_supply_in_a_week_l468_468493

def dozen := 12
def eggs_per_day_store1 := 5 * dozen
def eggs_per_day_store2 := 30
def daily_eggs_supplied := eggs_per_day_store1 + eggs_per_day_store2
def days_per_week := 7

theorem mark_egg_supply_in_a_week : daily_eggs_supplied * days_per_week = 630 := by
  sorry

end mark_egg_supply_in_a_week_l468_468493


namespace min_fraction_value_l468_468872

theorem min_fraction_value (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_tangent : 2 * a + b = 2) :
  (8 * a + b) / (a * b) ≥ 9 :=
by
  sorry

end min_fraction_value_l468_468872


namespace system_of_equations_solution_l468_468987

theorem system_of_equations_solution :
  ∀ (x : Fin 100 → ℝ), 
  (x 0 + x 1 + x 2 = 0) ∧ 
  (x 1 + x 2 + x 3 = 0) ∧ 
  -- Continue for all other equations up to
  (x 98 + x 99 + x 0 = 0) ∧ 
  (x 99 + x 0 + x 1 = 0)
  → ∀ (i : Fin 100), x i = 0 :=
by
  intros x h
  -- We can insert the detailed solving steps here
  sorry

end system_of_equations_solution_l468_468987


namespace triangle_centroid_l468_468507

theorem triangle_centroid (A B C M : Point)
(Hin_triangle: M ∈ triangle A B C)
(equal_areas: area (triangle A M B) = area (triangle A M C) ∧ area (triangle A M B) = area (triangle B M C) ):
M = centroid A B C :=
by
  sorry

end triangle_centroid_l468_468507


namespace points_per_question_l468_468969

theorem points_per_question (first_half_correct : ℕ) (second_half_correct : ℕ) (final_score : ℕ) :
  first_half_correct = 5 → 
  second_half_correct = 5 → 
  final_score = 50 → 
  (final_score / (first_half_correct + second_half_correct) = 5) :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  have h_total : first_half_correct + second_half_correct = 10 := by rw [h1, h2]
  rw h_total
  exact Nat.div_eq_of_eq_mul_right (by norm_num) h3

end points_per_question_l468_468969


namespace intersection_product_eq_three_l468_468035

noncomputable def curve_cartesian (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + (y - 1) ^ 2 = 5

noncomputable def point_on_line (t : ℝ) : ℝ × ℝ :=
  (1 + (Real.sqrt 3) / 2 * t, 2 + 1 / 2 * t)

noncomputable def line_intersects_curve (t : ℝ) : Prop :=
  curve_cartesian (fst (point_on_line t)) (snd (point_on_line t))

theorem intersection_product_eq_three :
  (∃ t1 t2 : ℝ, line_intersects_curve t1 ∧ line_intersects_curve t2 ∧ abs t1 * abs t2 = 3) :=
sorry

end intersection_product_eq_three_l468_468035


namespace find_fraction_l468_468057

variable (x y z : ℝ)

theorem find_fraction (h : (x - y) / (z - y) = -10) : (x - z) / (y - z) = 11 := 
by
  sorry

end find_fraction_l468_468057


namespace equation_of_trajectory_product_of_distances_l468_468332

theorem equation_of_trajectory {C : Type*} (F : ℝ × ℝ) (x y : ℝ) :
  F = (-Real.sqrt 3, 0) →
  (x + Real.sqrt 3)^2 + y^2 = 16 →  -- Condition for \(O_2\)
  (x - 2 * Real.sqrt 3)^2 + y^2 = 13 →  -- Contain \(O_2\)'s equation transformed
  (C = SetOf fun p : ℝ × ℝ => ((p.1)^2 / 4) + (p.2)^2 = 1) := sorry

theorem product_of_distances (P A B : Type*) (M N : ℝ × ℝ) (x₀ y₀ : ℝ) :
  A = (2, 0) →
  B = (0, 1) →
  ((x₀^2) / 4 + y₀^2 = 1 ∧ x₀ ≠ 0 ∧ y₀ ≠ 0) → -- P ∈ C and not on axes
  P = (x₀, y₀) →
  (M = (0, (-2 * y₀) / (x₀ - 2))) →
  (N = (-x₀ / (y₀ - 1), 0)) →
  abs ((2 * x₀) / (y₀ - 1)) * abs ((2 * y₀) / (x₀ - 2)) = 4 := sorry

end equation_of_trajectory_product_of_distances_l468_468332


namespace optimal_selection_method_uses_golden_ratio_l468_468832

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468832


namespace optimal_selection_method_uses_golden_ratio_l468_468629

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l468_468629


namespace David_average_marks_l468_468292

theorem David_average_marks :
  let english := 72
  let mathematics := 45
  let physics := 72
  let chemistry := 77
  let biology := 75
  (english + mathematics + physics + chemistry + biology) / 5 = 68.2 := 
by
  let english := 72
  let mathematics := 45
  let physics := 72
  let chemistry := 77
  let biology := 75
  have h1 : english + mathematics + physics + chemistry + biology = 341 := by sorry
  have h2 : 341 / 5 = 68.2 := by sorry
  show (english + mathematics + physics + chemistry + biology) / 5 = 68.2 from by sorry

end David_average_marks_l468_468292


namespace hua_luogeng_optimal_selection_method_l468_468732

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l468_468732


namespace race_time_a_beats_b_l468_468980

/-- In a 1000 meter race, if runner A beats runner B by 51 meters (or equivalently, 11 seconds),
    prove that the time Ta taken by runner A to complete the race is approximately 215.686 seconds. -/
theorem race_time_a_beats_b (Ta Tb Va Vb : ℝ) 
  (h1 : Va * Ta = 1000) 
  (h2 : Vb * Tb = 949) 
  (h3 : Ta = Tb - 11) : 
  Ta ≈ 215.686 := 
sorry

end race_time_a_beats_b_l468_468980


namespace trigonometric_identity_l468_468994

theorem trigonometric_identity
    (α φ : ℝ) :
    4 * Real.cos α * Real.cos φ * Real.cos (α - φ) - 2 * (Real.cos (α - φ))^2 - Real.cos (2 * φ) = Real.cos (2 * α) :=
by
  sorry

end trigonometric_identity_l468_468994


namespace shifted_function_expression_l468_468162

def original_function (x : ℝ) : ℝ := 5 * x^2

def shift_right (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x, f (x - a)

def shift_down (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x, f x - b

theorem shifted_function_expression :
  (shift_down (shift_right original_function 3) 2) = λ x, 5 * (x - 3)^2 - 2 :=
by sorry

end shifted_function_expression_l468_468162


namespace coefficient_of_x3_in_expansion_l468_468082

theorem coefficient_of_x3_in_expansion :
  let n := 5
  let sum_of_coeffs := (1 + 2 : ℤ)^n
  sum_of_coeffs = 243 →
  let coeff_of_x3 := 2^3 * Nat.choose n 3
  coeff_of_x3 = 80 :=
by
  intros,
  let n := 5
  let sum_of_coeffs := 3^n
  let coeff_of_x3 := 2^3 * Nat.choose n 3
  have h_sum : sum_of_coeffs = 243 := by sorry
  have h_coeff : coeff_of_x3 = 80 := by sorry
  exact h_coeff

end coefficient_of_x3_in_expansion_l468_468082


namespace optimal_selection_golden_ratio_l468_468705

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l468_468705


namespace optimal_selection_method_uses_golden_ratio_l468_468838

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468838


namespace minimum_number_of_girls_l468_468895

theorem minimum_number_of_girls (students boys girls : ℕ) 
(h1 : students = 20) (h2 : boys = students - girls)
(h3 : ∀ d, 0 ≤ d → 2 * (d + 1) ≥ boys) 
: 6 ≤ girls :=
by 
  have h4 : 20 - girls = boys, from h2.symm
  have h5 : 2 * (girls + 1) ≥ boys, from h3 girls (nat.zero_le girls)
  linarith

#check minimum_number_of_girls

end minimum_number_of_girls_l468_468895


namespace diplomats_conference_l468_468223

theorem diplomats_conference (D : ℝ) 
  (French_speakers : ℝ) (Non_Hindi_speakers : ℝ) 
  (Neither_speakers : ℝ) (Both_speakers : ℝ) 
  (h1 : French_speakers = 20) 
  (h2 : Non_Hindi_speakers = 32) 
  (h3 : Neither_speakers = 0.2 * D) 
  (h4 : Both_speakers = 0.1 * D):
  (20 + 32 - h4) / 0.1 = D :=
by
  sorry

end diplomats_conference_l468_468223


namespace vanessa_weeks_to_wait_l468_468949

theorem vanessa_weeks_to_wait
  (dress_cost savings : ℕ)
  (weekly_allowance weekly_expense : ℕ)
  (h₀ : dress_cost = 80)
  (h₁ : savings = 20)
  (h₂ : weekly_allowance = 30)
  (h₃ : weekly_expense = 10) :
  let net_savings_per_week := weekly_allowance - weekly_expense,
      additional_amount_needed := dress_cost - savings in
  additional_amount_needed / net_savings_per_week = 3 :=
by
  sorry

end vanessa_weeks_to_wait_l468_468949


namespace correct_statements_count_l468_468868

-- Definition of the statements
def statement1 := ∀ (q : ℚ), q > 0 ∨ q < 0
def statement2 (a : ℝ) := |a| = -a → a < 0
def statement3 := ∀ (x y : ℝ), 0 = 3
def statement4 := ∀ (q : ℚ), ∃ (p : ℝ), q = p
def statement5 := 7 = 7 ∧ 10 = 10 ∧ 15 = 15

-- Define what it means for each statement to be correct
def is_correct_statement1 := statement1 = false
def is_correct_statement2 := ∀ a : ℝ, statement2 a = false
def is_correct_statement3 := statement3 = false
def is_correct_statement4 := statement4 = true
def is_correct_statement5 := statement5 = true

-- Define the problem and its correct answer
def problem := is_correct_statement1 ∧ is_correct_statement2 ∧ is_correct_statement3 ∧ is_correct_statement4 ∧ is_correct_statement5

-- Prove that the number of correct statements is 2
theorem correct_statements_count : problem → (2 = 2) :=
by
  intro h
  sorry

end correct_statements_count_l468_468868


namespace unique_expression_nonneg_int_l468_468510

theorem unique_expression_nonneg_int (A : ℕ) : ∃ (x y : ℕ), A = (x + y)^2 + 3*x + y / 2 := 
begin
  sorry
end

end unique_expression_nonneg_int_l468_468510


namespace cost_of_each_soda_l468_468957

theorem cost_of_each_soda :
  let sandwich_cost := 3.49
  let num_sandwiches := 2
  let total_cost := 10.46
  let num_sodas := 4
  let cost_of_sandwiches := num_sandwiches * sandwich_cost
  let total_cost_of_sodas := total_cost - cost_of_sandwiches
  let cost_of_soda := total_cost_of_sodas / num_sodas
  by
    have h_cost_of_sandwiches: cost_of_sandwiches = 6.98 := by sorry
    have h_total_cost_of_sodas: total_cost_of_sodas = 3.48 := by sorry
    have h_cost_of_soda: cost_of_soda = 0.87 := by sorry
    exact h_cost_of_soda

end cost_of_each_soda_l468_468957


namespace geometry_of_OPQRS_l468_468359

noncomputable def points_in_3D_distinct
  (P Q R S : ℝ × ℝ × ℝ)
  (O : ℝ × ℝ × ℝ := (0, 0, 0))
  (hP : P ≠ Q)
  (hQ : Q ≠ R)
  (hR : R ≠ S)
  (hS : S ≠ P)
  (hR_def : R = (P.1 + Q.1, P.2 + Q.2, P.3 + Q.3)) :
  Prop :=
  let OP := (P.1, P.2, P.3)
  let OQ := (Q.1, Q.2, Q.3)
  let OS := (S.1, S.2, S.3)
  (linear_independent ℝ [OP, OQ, OS] ∨ (OP.1 * (OQ.2 * OS.3 - OQ.3 * OS.2) - 
    OP.2 * (OQ.1 * OS.3 - OQ.3 * OS.1) + OP.3 * (OQ.1 * OS.2 - OQ.2 * OS.1) = 0))

theorem geometry_of_OPQRS :
  ∀ (P Q R S : ℝ × ℝ × ℝ) (O : ℝ × ℝ × ℝ)
  (hP : P ≠ Q) (hQ : Q ≠ R) (hR : R ≠ S) (hS : S ≠ P) 
  (hR_def : R = (P.1 + Q.1, P.2 + Q.2, P.3 + Q.3)),
  points_in_3D_distinct P Q R S O hP hQ hR hS hR_def :=
by
  intros
  sorry

end geometry_of_OPQRS_l468_468359


namespace optimal_selection_method_uses_golden_ratio_l468_468603

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l468_468603


namespace evaluate_expression_l468_468962

theorem evaluate_expression : 6^3 - 4 * 6^2 + 4 * 6 - 1 = 95 :=
by
  sorry

end evaluate_expression_l468_468962


namespace optimal_selection_method_use_golden_ratio_l468_468799

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l468_468799


namespace optimal_selection_method_uses_golden_ratio_l468_468579

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l468_468579


namespace optimal_selection_method_uses_golden_ratio_l468_468681

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468681


namespace optimal_selection_uses_golden_ratio_l468_468643

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l468_468643


namespace complex_number_solution_l468_468326

theorem complex_number_solution (z : ℂ) (h : z * (2 - complex.i) = 3 + complex.i) : z = 1 + complex.i := 
sorry

end complex_number_solution_l468_468326


namespace optimal_selection_uses_golden_ratio_l468_468638

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l468_468638


namespace optimal_selection_uses_golden_ratio_l468_468821

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l468_468821


namespace hua_luogeng_optimal_selection_l468_468561

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l468_468561


namespace polynomial_nonzero_term_count_l468_468285

def polynomial_expr_1 := (x - 5) * (3 * x^2 - 2 * x + 8)
def polynomial_expr_2 := 2 * (x^3 + 3 * x^2 - 4 * x)
def polynomial_expr := polynomial_expr_1 - polynomial_expr_2

theorem polynomial_nonzero_term_count : 
  -- Given the polynomial expression
  polynomial_expr = x^3 - 23 * x^2 + 26 * x - 40 →
  -- Prove that there are 4 nonzero terms in it
  4 = 4 :=
by
  intros
  sorry

end polynomial_nonzero_term_count_l468_468285


namespace optimal_selection_method_uses_golden_ratio_l468_468632

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l468_468632


namespace DE_passes_through_midpoint_l468_468464

variable (Γ : Type) [circle Γ]
variable (O A B C D E M N : Point Γ)
variable (diameter_AB : diameter A B)
variable (perpendicular_AB_CD : ⟂(Line A B) (Line C D))
variable (midpoint_M_OC : (Line O C).midpoint = M)
variable (chord_AE : ∃ E, chord A E ∧ Line A E ∋ M)
variable (point_N : ∃ N, point_on N (Line D E) ∧ point_on N (Line B C))
variable (midpoint_N_BC : (Line B C).midpoint = N)

theorem DE_passes_through_midpoint :
  (Line D E).point_on midpoint_N_BC := sorry

end DE_passes_through_midpoint_l468_468464


namespace locus_of_M_l468_468335

theorem locus_of_M (ABC : Triangle) (P : Point) (γ : ℝ) (hP : P ∈ ℓ_AB) :
  let A' := intersection (parallel_line_through P BC) AC in
  let B' := intersection (parallel_line_through P AC) BC in
  let circAAP := circumcircle A A' P in
  let circBBP := circumcircle B B' P in
  let M := second_intersection circAAP circBBP in
  locus_cond M AB γ :=
  if hγ : γ < 90 then 
    on_arC_same_side_as_C(M AB (2 * γ))
  else 
    on_arc_opposite_side_from_C(M AB (360 - 2 * γ)) :=
sorry -- proof to be completed

end locus_of_M_l468_468335


namespace hua_luogeng_optimal_selection_l468_468547

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l468_468547


namespace inverse_variation_example_l468_468531

theorem inverse_variation_example (x y : ℝ) (k : ℝ) 
  (h1 : ∀ x y, x * y^3 = k) (h2 : 8 * (1:ℝ)^3 = k) : 
  (∃ (x : ℝ), x * (2:ℝ)^3 = k ∧ x = 1) :=
by
  have hx : 8 = k := by
    rw [←h2, one_mul]
  
  use 1
  split
  . exact hx.symm
  . rfl

end inverse_variation_example_l468_468531


namespace optimal_selection_uses_golden_ratio_l468_468831

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l468_468831


namespace fraction_infinite_decimal_l468_468266

theorem fraction_infinite_decimal :
  ∃ (f : ℚ), f ∉ { x | ∃ (m n : ℕ), (n ≠ 0) ∧ (x = m / n) ∧ (∀ p ∈ nat.prime_factors n, p = 2 ∨ p = 5) } ∧ f ∈ ({3 / 7}) :=
by
  sorry

end fraction_infinite_decimal_l468_468266


namespace inv_log_base3_l468_468178

theorem inv_log_base3 (x : ℝ) (h : x > 0) : (∃ y : ℝ, x = 3^y) ↔ (y = log 3 x) :=
by
  sorry

end inv_log_base3_l468_468178


namespace optimal_selection_golden_ratio_l468_468702

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l468_468702


namespace mark_egg_supply_in_a_week_l468_468494

def dozen := 12
def eggs_per_day_store1 := 5 * dozen
def eggs_per_day_store2 := 30
def daily_eggs_supplied := eggs_per_day_store1 + eggs_per_day_store2
def days_per_week := 7

theorem mark_egg_supply_in_a_week : daily_eggs_supplied * days_per_week = 630 := by
  sorry

end mark_egg_supply_in_a_week_l468_468494


namespace vector_collinear_l468_468118

variables {𝕜 : Type*} [NormedField 𝕜] [NormedSpace 𝕜 ℝ]
variables (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0)

theorem vector_collinear (h : ∥a + b∥ = ∥a∥ - ∥b∥) : ∃ λ : 𝕜, b = λ • a :=
sorry

end vector_collinear_l468_468118


namespace sum_of_positive_integers_reverse_base5_base9_l468_468312

theorem sum_of_positive_integers_reverse_base5_base9 :
  ∑ n in (Finset.filter (λ n, ∃ a₀ a₁, 0 ≤ a₀ ∧ a₀ ≤ 4 ∧ 0 ≤ a₁ ∧ a₁ ≤ 8 ∧ n = 5 * a₁ + a₀ ∧ n = 9 * a₀ + a₁) (Finset.range 100)), n = 110 := by
  sorry

end sum_of_positive_integers_reverse_base5_base9_l468_468312


namespace probability_blue_given_glass_l468_468191

-- Defining the various conditions given in the problem
def total_red_balls : ℕ := 5
def total_blue_balls : ℕ := 11
def red_glass_balls : ℕ := 2
def red_wooden_balls : ℕ := 3
def blue_glass_balls : ℕ := 4
def blue_wooden_balls : ℕ := 7
def total_balls : ℕ := total_red_balls + total_blue_balls
def total_glass_balls : ℕ := red_glass_balls + blue_glass_balls

-- The mathematically equivalent proof problem statement.
theorem probability_blue_given_glass :
  (blue_glass_balls : ℚ) / (total_glass_balls : ℚ) = 2 / 3 := by
sorry

end probability_blue_given_glass_l468_468191


namespace optimal_selection_method_uses_golden_ratio_l468_468608

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l468_468608


namespace integer_values_satisfying_inequality_l468_468047

theorem integer_values_satisfying_inequality :
  {x : ℤ | (x-1)^2 < 8}.to_finset.card = 5 := 
sorry

end integer_values_satisfying_inequality_l468_468047


namespace distinct_pairs_reciprocal_sum_l468_468378

theorem distinct_pairs_reciprocal_sum :
  {p : ℕ × ℕ | p.1 > 0 ∧ p.2 > 0 ∧ (↑(p.1)⁻¹ + ↑(p.2)⁻¹ = (5 : ℚ)⁻¹)}.to_finset.card = 3 :=
by sorry

end distinct_pairs_reciprocal_sum_l468_468378


namespace total_pages_l468_468099

noncomputable def P : ℕ := 468

theorem total_pages (h1 : ∃ P : ℕ, Karthik_read_first_week : ℚ := 7 / 13)
                    (h2 : ∃ P : ℕ, Karthik_read_second_week : ℚ := 5 / 9)
                    (h3 : ∃ R : ℕ, Unread_pages := 96) : P = 468 :=
by
  exact sorry

end total_pages_l468_468099


namespace intersection_point_moving_point_l468_468085

-- Define curves C1 and C2
def C1 (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < π / 2) : ℝ := 4 * Real.cos θ
def C2 (θ : ℝ) : ℝ := 3 / Real.cos θ

-- Define the statement for the intersection point
theorem intersection_point (θ : ℝ) (ρ : ℝ) (hθ : 0 ≤ θ ∧ θ < π / 2) :
  (θ = π / 6 ∧ ρ = 2 * Real.sqrt 3) ↔ (ρ = C1 θ hθ ∧ ρ = C2 θ) :=
sorry

-- Define point Q and relation between OQ and QP
def point_Q (θ₀ : ℝ) (hθ₀ : 0 ≤ θ₀ ∧ θ₀ < π / 2) : ℝ := 4 * Real.cos θ₀

theorem moving_point (θ : ℝ) (ρ : ℝ) (hθ : 0 ≤ θ ∧ θ < π / 2) :
  (ρ = 10 * Real.cos θ ∧ θ ∈ [0, π / 2)) ↔
  (∃ (θ₀ ρ₀ : ℝ), ρ₀ = point_Q θ₀ hθ ∧ ρ₀ = 2 / 5 * ρ ∧ θ₀ = θ) :=
sorry

end intersection_point_moving_point_l468_468085


namespace math_club_attendance_l468_468280

theorem math_club_attendance:
  ∀ (total_students : ℕ) (total_sessions : ℕ) (students_three_sessions : ℕ)
    (students_two_sessions : ℕ) (students_one_session : ℕ) (total_attendance_marks : ℕ),
  total_students = 20 →
  total_sessions = 4 →
  students_three_sessions = 9 →
  students_two_sessions = 5 →
  students_one_session = 3 →
  total_attendance_marks = total_students * total_sessions →
  let attendance_three := students_three_sessions * 3 in
  let attendance_two := students_two_sessions * 2 in
  let attendance_one := students_one_session * 1 in
  let total_partial_attendance := attendance_three + attendance_two + attendance_one in
  let remaining_attendance := total_attendance_marks - total_partial_attendance in
  let students_all_sessions := remaining_attendance / 4 in
  students_all_sessions = 10 :=
by
  intros _ _ _ _ _ _
  intros htotal_students htotal_sessions hstudents_three_sessions hstudents_two_sessions hstudents_one_session htotal_attendance_marks
  have h_attendance_three := hstudents_three_sessions ▸ rfl
  have h_attendance_two := hstudents_two_sessions ▸ rfl
  have h_attendance_one := hstudents_one_session ▸ rfl
  unfold total_attendance_marks
  have h_total_attendance := htotal_attendance_marks ▸ rfl
  unfold attendance_three at *
  unfold attendance_two at *
  unfold attendance_one at *
  unfold total_partial_attendance at * 
  unfold remaining_attendance at *
  have h_total_partial_attendance : total_partial_attendance = 27 + 10 + 3 := by rfl
  unfold total_partial_attendance 
  have h_remaining_attendance : remaining_attendance = 80 - 40 := by rfl
  unfold remaining_attendance 
  have h_students_all_sessions : students_all_sessions = 40 / 4 := by rfl
  unfold students_all_sessions 
  have h_students_all_sessions_correct : students_all_sessions = 10 := by rfl
  exact h_students_all_sessions_correct

end math_club_attendance_l468_468280


namespace optimal_selection_method_uses_golden_ratio_l468_468854

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l468_468854


namespace points_per_question_l468_468970

theorem points_per_question (first_half_correct : ℕ) (second_half_correct : ℕ) (final_score : ℕ) :
  first_half_correct = 5 → 
  second_half_correct = 5 → 
  final_score = 50 → 
  (final_score / (first_half_correct + second_half_correct) = 5) :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  have h_total : first_half_correct + second_half_correct = 10 := by rw [h1, h2]
  rw h_total
  exact Nat.div_eq_of_eq_mul_right (by norm_num) h3

end points_per_question_l468_468970


namespace hua_luogeng_optimal_selection_method_l468_468789

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l468_468789


namespace last_three_digits_of_2_pow_15000_l468_468217

-- We need to define the given condition as a hypothesis and then state the goal.
theorem last_three_digits_of_2_pow_15000 :
  (2 ^ 500 ≡ 1 [MOD 1250]) → (2 ^ 15000 ≡ 1 [MOD 1000]) := by
  sorry

end last_three_digits_of_2_pow_15000_l468_468217


namespace optimal_selection_method_use_golden_ratio_l468_468813

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l468_468813


namespace optimalSelectionUsesGoldenRatio_l468_468709

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l468_468709


namespace vanessa_savings_weeks_l468_468947

-- Definitions of given conditions
def dress_cost : ℕ := 80
def vanessa_savings : ℕ := 20
def weekly_allowance : ℕ := 30
def weekly_spending : ℕ := 10

-- Required amount to save 
def required_savings : ℕ := dress_cost - vanessa_savings

-- Weekly savings calculation
def weekly_savings : ℕ := weekly_allowance - weekly_spending

-- Number of weeks needed to save the required amount
def weeks_needed_to_save (required_savings weekly_savings : ℕ) : ℕ :=
  required_savings / weekly_savings

-- Axiom representing the correctness of our calculation
theorem vanessa_savings_weeks : weeks_needed_to_save required_savings weekly_savings = 3 := 
  by
  sorry

end vanessa_savings_weeks_l468_468947


namespace possible_lightest_boy_heaviest_girl_l468_468414

theorem possible_lightest_boy_heaviest_girl
    (B G : ℕ)
    (hN : B + G < 35)
    (w_boys w_girls : ℕ → ℝ)
    (avg_all : Real := 53.5)
    (avg_girls : Real := 47)
    (avg_boys : Real := 60) :
    (∑ i in Finset.range B, w_boys i) / B = avg_boys →
    (∑ i in Finset.range G, w_girls i) / G = avg_girls →
    ((∑ i in Finset.range B, w_boys i) + (∑ i in Finset.range G, w_girls i)) / (B + G) = avg_all →
    ∃ (i j : ℕ), i < B ∧ j < G ∧ w_boys i < all_weights_worst(G, w_girls) ∧ w_girls j > all_weights_best(B, w_boys) :=
sorry

/-- Helper function that returns the minimum weight of the given range of girls' weights --/
noncomputable def all_weights_worst (g: ℕ, wg: ℕ → ℝ) : ℝ :=
  Finset.min' (Finset.range g) (begin
    -- there should be at least one girl
    exact ⟨0, by simp [show 0 < g, by omega]⟩
  end)

/-- Helper function that returns the maximum weight of the given range of boys' weights --/
noncomputable def all_weights_best (b: ℕ, wb: ℕ → ℝ) : ℝ :=
  Finset.max' (Finset.range b) (begin
    -- there should be at least one boy
    exact ⟨0, by simp [show 0 < b, by omega]⟩
  end)

end possible_lightest_boy_heaviest_girl_l468_468414


namespace arrangement_of_people_l468_468995

theorem arrangement_of_people (P : Fin 5 → Type) (h : ∃ i, P i = true ∧ i = 2) : 
  ∃ n : Nat, n = 4! ∧ n = 24 := 
by
  sorry

end arrangement_of_people_l468_468995


namespace hua_luogeng_optimal_selection_l468_468553

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l468_468553


namespace original_flow_rate_l468_468195

theorem original_flow_rate :
  ∃ F : ℚ, 
  (F * 0.75 * 0.4 * 0.6 - 1 = 2) ∧
  (F = 50/3) :=
by
  sorry

end original_flow_rate_l468_468195


namespace optimal_selection_uses_golden_ratio_l468_468827

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l468_468827


namespace optimal_selection_method_use_golden_ratio_l468_468806

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l468_468806


namespace sculpture_paint_area_correct_l468_468271

def sculpture_exposed_area (edge_length : ℝ) (num_cubes_layer1 : ℕ) (num_cubes_layer2 : ℕ) (num_cubes_layer3 : ℕ) : ℝ :=
  let area_top_layer1 := num_cubes_layer1 * edge_length ^ 2
  let area_side_layer1 := 8 * 3 * edge_length ^ 2
  let area_top_layer2 := num_cubes_layer2 * edge_length ^ 2
  let area_side_layer2 := 10 * edge_length ^ 2
  let area_top_layer3 := num_cubes_layer3 * edge_length ^ 2
  let area_side_layer3 := num_cubes_layer3 * 4 * edge_length ^ 2
  area_top_layer1 + area_side_layer1 + area_top_layer2 + area_side_layer2 + area_top_layer3 + area_side_layer3

theorem sculpture_paint_area_correct :
  sculpture_exposed_area 1 12 6 2 = 62 := by
  sorry

end sculpture_paint_area_correct_l468_468271


namespace solution_set_inequality_l468_468186

theorem solution_set_inequality (x : ℝ) : x * (x - 1) > 0 ↔ x < 0 ∨ x > 1 :=
by sorry

end solution_set_inequality_l468_468186


namespace calories_in_250_grams_of_lemonade_l468_468485

theorem calories_in_250_grams_of_lemonade:
  ∀ (lemon_juice_grams sugar_grams water_grams total_grams: ℕ)
    (lemon_juice_cal_per_100 sugar_cal_per_100 total_cal: ℕ),
  lemon_juice_grams = 150 →
  sugar_grams = 150 →
  water_grams = 300 →
  total_grams = lemon_juice_grams + sugar_grams + water_grams →
  lemon_juice_cal_per_100 = 30 →
  sugar_cal_per_100 = 386 →
  total_cal = (lemon_juice_grams * lemon_juice_cal_per_100 / 100) + (sugar_grams * sugar_cal_per_100 / 100) →
  (250:ℕ) * total_cal / total_grams = 260 :=
by
  intros lemon_juice_grams sugar_grams water_grams total_grams lemon_juice_cal_per_100 sugar_cal_per_100 total_cal
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end calories_in_250_grams_of_lemonade_l468_468485


namespace train_speed_l468_468258

theorem train_speed (distance : ℝ) (time : ℝ) (speed : ℝ) :
  distance = 210 ∧ time = 13 → speed = 16.15 := 
by
  intro h
  cases h with hd ht
  rw [hd, ht]
  -- As we're proving an approximation, we'll use sorry to omit the computational proof.
  sorry

end train_speed_l468_468258


namespace angle_AMC_obtuse_triangle_AOD_equilateral_l468_468228

-- Definitions based on the conditions
def Triangle (A B C : Type) := sorry -- Placeholder for the actual triangle definition
def midpoint {A B : Type} (C : Type) : Prop := sorry
def angle_eq (a b : ℝ) : Prop := sorry
def circumcenter (A B C : Type) : Type := sorry
def is_obtuse (angle : ℝ) : Prop := sorry
def is_equilateral (A B C : Type) : Prop := sorry

variables {A B C M D O : Type}
variables {γ : ℝ}

-- Conditions
axiom triangle_ABC : Triangle A B C
axiom M_midpoint_BC : midpoint M (B, C)
axiom angle_MAB_eq_C : angle_eq (angle M A B) γ
axiom angle_MAC_15 : angle_eq (angle M A C) (15 * (π / 180))
axiom O_circumcenter_ADC : circumcenter O A D C

-- Required Proofs
theorem angle_AMC_obtuse : is_obtuse (angle A M C) :=
sorry

theorem triangle_AOD_equilateral : is_equilateral A O D :=
sorry

end angle_AMC_obtuse_triangle_AOD_equilateral_l468_468228


namespace optimal_selection_method_uses_golden_ratio_l468_468685

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468685


namespace optimal_selection_method_uses_golden_ratio_l468_468834

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468834


namespace polynomial_root_sum_l468_468192

theorem polynomial_root_sum
  (m n : ℤ)
  (h : ∃ (α : ℝ), α = 9 + Real.sqrt 11 ∧ polynomial.aeval α (Polynomial.C n + Polynomial.X * Polynomial.C m + Polynomial.X ^ 2) = 0) :
  m + n = 52 :=
sorry

end polynomial_root_sum_l468_468192


namespace unused_sector_angle_l468_468319

 noncomputable def cone_volume (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

 noncomputable def slant_height (r h : ℝ) : ℝ := real.sqrt (r^2 + h^2)

 noncomputable def central_angle_used (r R : ℝ) : ℝ := (r / R) * 360

 noncomputable def unused_angle (central_angle_used : ℝ) : ℝ := 360 - central_angle_used

 theorem unused_sector_angle {DE : ℝ} (h : ℝ) (volume : ℝ) :
   let r := 15 in
   let cone_vol := cone_volume r h in
   cone_vol = 675 * π →
   h = 9 →
   slant_height r h = real.sqrt (225 + 81) →
   DE = real.sqrt (225 + 81) →
   central_angle_used r DE = (r / DE) * 360 →
   unused_angle ((r / DE) * 360) = 51 :=
 by
  intros r cone_vol cone_eq h_eq slant_eq de_eq central_eq
  sorry

end unused_sector_angle_l468_468319


namespace hua_luogeng_optimal_selection_method_l468_468727

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l468_468727


namespace optimal_selection_method_uses_golden_ratio_l468_468845

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468845


namespace sum_of_x_satisfying_equation_l468_468116

-- Define the function f
def f (x : ℝ) : ℝ := 12 * x + 5

-- The theorem stating the sum of all x that satisfy the equation is 65
theorem sum_of_x_satisfying_equation :
  let g (x : ℝ) : ℝ := f⁻¹ (x)
  ∑ x in { x : ℝ | g x = f ((3 * x)⁻¹)}, x = 65 :=
sorry

end sum_of_x_satisfying_equation_l468_468116


namespace hua_luogeng_optimal_selection_method_l468_468795

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l468_468795


namespace slope_of_line_FM_equation_of_ellipse_range_of_slope_OP_l468_468339

-- Definitions for conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def focus (c : ℝ) : ℝ × ℝ :=
  (-c, 0)

def eccentricity (a c : ℝ) : Prop :=
  (a^2 - (focus c).1^2 = a^2 / 3)

def intercepts_circle (c b : ℝ) (x y : ℝ) : Prop :=
  ∃kx ky, x^2 + y^2 = b^2 / 4 ∧ kx^2 + ky^2 = c^2

def line_segment_length (x1 y1 x2 y2 : ℝ) (len : ℝ) : Prop :=
  sqrt ((x2 - x1)^2 + (y2 - y1)^2) = len

def quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

-- Mathematically equivalent proof problem
theorem slope_of_line_FM (a b c x y : ℝ) :
  ellipse a b x y →
  eccentricity a c → 
  quadrant x y →
  intercepts_circle c b x y →
  line_segment_length (-c) 0 x y (4 * sqrt(3) / 3) →
  (y / (x + c)) = sqrt(3) / 3 := sorry

theorem equation_of_ellipse (a b c x y : ℝ) :
  ellipse a b x y →
  eccentricity a c →
  line_segment_length (-c) 0 x y (4 * sqrt(3) / 3) →
  ∃ x y, (x^2 / 3 + y^2 / 2 = 1) := sorry

theorem range_of_slope_OP (a b c x y : ℝ) (t : ℝ) :
  ellipse a b x y →
  eccentricity a c →
  (y / (x + c)) > sqrt(2) →
  ∀ x y, x ∈ (-3 / 2, -1) ∪ (-1, 0) →
  let m := y / x in 
  m ∈ (-(∞ : ℝ), -2 * sqrt(3) / 3) ∪ (sqrt(2) / 3, 2 * sqrt(3) / 3) := sorry

end slope_of_line_FM_equation_of_ellipse_range_of_slope_OP_l468_468339


namespace optimal_selection_method_uses_golden_ratio_l468_468670

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l468_468670


namespace minimum_number_of_girls_l468_468903

theorem minimum_number_of_girls (total_students : ℕ) (d : ℕ) 
  (h_students : total_students = 20) 
  (h_unique_lists : ∀ n : ℕ, ∃! k : ℕ, k ≤ 20 - d ∧ n = 2 * k) :
  d ≥ 6 :=
sorry

end minimum_number_of_girls_l468_468903


namespace number_of_true_contrapositives_l468_468174

-- Definitions of the propositions
def prop1 := ∀ (T1 T2 : Triangle), congruent T1 T2 → corresponding_angles T1 T2
def prop2 := ∀ (a : ℝ), a < 0 → abs a = -a
def prop3 := ∀ (S : Square), all_sides_equal S
def prop4 := ∀ (P : Parallelogram), opposite_angles_equal P

-- Definitions of the contrapositives
def contrapositive1 := ∀ (T1 T2 : Triangle), corresponding_angles T1 T2 → congruent T1 T2
def contrapositive2 := ∀ (a : ℝ), (abs a = -a ∧ a < 0)
def contrapositive3 := ∀ (Q : Quadrilateral), all_sides_equal Q → is_square Q
def contrapositive4 := ∀ (Q : Quadrilateral), opposite_angles_equal Q → is_parallelogram Q

-- Proof of the main statement
theorem number_of_true_contrapositives : 
  (∃ n : ℕ, n = 1 ∧ (
  (contrapositive1 = true → ∃ n, n = 1) ∧ 
  (contrapositive2 = true → ∃ n, n = 1) ∧ 
  (contrapositive3 = true → ∃ n, n = 1) ∧ 
  (contrapositive4 = true → ∃ n, n = 1)
  )) :=
  sorry

end number_of_true_contrapositives_l468_468174


namespace min_fraction_expression_l468_468483

theorem min_fraction_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : 1 / a + 1 / b = 1) : 
  ∃ a b, ∃ (h : 1 / a + 1 / b = 1), a > 1 ∧ b > 1 ∧ (1 / (a - 1) + 4 / (b - 1)) = 4 := 
by 
  sorry

end min_fraction_expression_l468_468483


namespace min_girls_in_class_l468_468935

theorem min_girls_in_class (n : ℕ) (d : ℕ) (h1 : n = 20) 
  (h2 : ∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
       ∀ f : ℕ → set ℕ, (card (f i) ≠ card (f j) ∨ card (f j) ≠ card (f k) ∨ card (f i) ≠ card (f k))) : 
d ≥ 6 :=
by
  sorry

end min_girls_in_class_l468_468935


namespace optimal_selection_method_uses_golden_ratio_l468_468627

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l468_468627


namespace delta_bounds_l468_468330

theorem delta_bounds (a b m : ℝ) (f : ℝ → ℝ) 
  (h₁ : f = λ x, x^2 + a * x + b) 
  (h₂ : abs (f m) ≤ 1/4) 
  (h₃ : abs (f (m + 1)) ≤ 1/4) : 
  0 ≤ a^2 - 4 * b ∧ a^2 - 4 * b ≤ 2 :=
sorry

end delta_bounds_l468_468330


namespace optimal_selection_uses_golden_ratio_l468_468635

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l468_468635


namespace sum_first_20_terms_l468_468350

theorem sum_first_20_terms :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ),
  (∀ n : ℕ, a n * (4 + cos (n * π)) = n * (2 - cos (n * π)))
  → (S n = ∑ i in finset.range n, a i)
  → S 20 = 122 := 
by intros a S h_a_n h_S
   sorry

end sum_first_20_terms_l468_468350


namespace evaluate_expression_l468_468961

theorem evaluate_expression : 6^3 - 4 * 6^2 + 4 * 6 - 1 = 95 :=
by
  sorry

end evaluate_expression_l468_468961


namespace optimal_selection_method_uses_golden_ratio_l468_468864

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l468_468864


namespace number_of_correct_statements_l468_468871

def condition1 : Prop := ∀ q : ℚ, q > 0 ∨ q < 0  -- This omits zero, hence incorrect.
def condition2 : Prop := ∀ a : ℝ, |a| = -a → a < 0  -- This doesn't consider the case of a = 0.
def poly := 2 * x^3 - 3 * x * y + 3 * y
def condition3 : Prop := ∃ (c : ℝ), polynomial.coeff poly 2 = c  -- There is no x^2 term here.
def condition4 : Prop := ∀ q : ℚ, ∃ r : ℝ, r = q  -- All rational numbers can be represented on the number line.
def pentagonal_prism := (7, 10, 15)  -- number of faces, vertices, edges
def condition5 : Prop := pentagonal_prism = (7, 10, 15)  -- This is correct by definition.

theorem number_of_correct_statements : (if condition4 then 1 else 0) + (if condition5 then 1 else 0) = 2 := by sorry

end number_of_correct_statements_l468_468871


namespace triangle_ratio_l468_468336

-- Given: Triangle ABC with \angle B = 90^\circ
-- Points E and D are on sides AC and BC respectively
-- such that AE = EC and \angle ADB = \angle EDC
-- To prove: CD / BD = 2 / 1

theorem triangle_ratio (A B C E D : Type) [LinearOrder A] 
    [Ring B] [DivisionRing C] [Field D]
    (hAngle : ∠ B = 90)
    (hE_mid : AE = EC)
    (hAngles : ∠ ADB = ∠ EDC)
    (hMedian_intersect : D ∈ segment AC) : 
    CD / BD = 2 / 1 :=
begin
    sorry
end

end triangle_ratio_l468_468336


namespace cube_volume_is_8_l468_468140

theorem cube_volume_is_8 (a : ℕ) 
  (h_cond : (a+2) * (a-2) * a = a^3 - 8) : 
  a^3 = 8 := 
by
  sorry

end cube_volume_is_8_l468_468140


namespace average_unchanged_and_variance_decreases_l468_468072

-- Given conditions
variables (n : ℕ) (avg s_sq : ℝ)
-- Xiao Ming's score
variable (xm_score : ℝ)

-- Assuming conditions
def cond_initial_students := n = 39
def cond_avg_score := avg = 105
def cond_variance := s_sq = 20
def cond_xiao_ming_score := xm_score = 105

-- Prove statements
theorem average_unchanged_and_variance_decreases
  (h_initial_students : cond_initial_students)
  (h_avg_score : cond_avg_score)
  (h_variance : cond_variance)
  (h_xiao_ming_score : cond_xiao_ming_score) :
  avg = 105 ∧ sq (30 / 38) = s_sq - 0.5 :=
sorry -- placeholder for the actual proof, not required here

end average_unchanged_and_variance_decreases_l468_468072


namespace optimal_selection_method_uses_golden_ratio_l468_468862

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l468_468862


namespace proof_x_y_l468_468394

noncomputable def x_y_problem (x y : ℝ) : Prop :=
  (x^2 = 9) ∧ (|y| = 4) ∧ (x < y) → (x - y = -1 ∨ x - y = -7)

theorem proof_x_y (x y : ℝ) : x_y_problem x y :=
by
  sorry

end proof_x_y_l468_468394


namespace median_sit_ups_l468_468878

-- Let's define the set of number of sit-ups
def sit_ups : List ℝ := [50, 45, 48, 47]

-- We can define a function to compute the median of a list of real numbers
noncomputable def median (l : List ℝ) : ℝ :=
  let sorted_l := l.qsort (≤)
  if (sorted_l.length % 2 = 1) then
    sorted_l.get (sorted_l.length / 2)
  else
    (sorted_l.get (sorted_l.length / 2 - 1) + sorted_l.get (sorted_l.length / 2)) / 2

-- Theorem stating that the median of sit_ups is 47.5
theorem median_sit_ups : median sit_ups = 47.5 := 
by
  sorry

end median_sit_ups_l468_468878


namespace amount_of_solution_added_l468_468993

variable (x : ℝ)

-- Condition: The solution contains 90% alcohol
def solution_alcohol_amount (x : ℝ) : ℝ := 0.9 * x

-- Condition: Total volume of the new mixture after adding 16 liters of water
def total_volume (x : ℝ) : ℝ := x + 16

-- Condition: The percentage of alcohol in the new mixture is 54%
def new_mixture_alcohol_amount (x : ℝ) : ℝ := 0.54 * (total_volume x)

-- The proof goal: the amount of solution added is 24 liters
theorem amount_of_solution_added : new_mixture_alcohol_amount x = solution_alcohol_amount x → x = 24 :=
by
  sorry

end amount_of_solution_added_l468_468993


namespace optimal_selection_uses_golden_ratio_l468_468640

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l468_468640


namespace hua_luogeng_optimal_selection_method_l468_468791

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l468_468791


namespace optimal_selection_method_uses_golden_ratio_l468_468577

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l468_468577


namespace minimum_number_of_girls_l468_468898

theorem minimum_number_of_girls (students boys girls : ℕ) 
(h1 : students = 20) (h2 : boys = students - girls)
(h3 : ∀ d, 0 ≤ d → 2 * (d + 1) ≥ boys) 
: 6 ≤ girls :=
by 
  have h4 : 20 - girls = boys, from h2.symm
  have h5 : 2 * (girls + 1) ≥ boys, from h3 girls (nat.zero_le girls)
  linarith

#check minimum_number_of_girls

end minimum_number_of_girls_l468_468898


namespace find_value_of_a2_b2_c2_l468_468104

variable {a b c : ℝ}

theorem find_value_of_a2_b2_c2
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : a + b + c = 0)
  (h3 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 5 := 
sorry

end find_value_of_a2_b2_c2_l468_468104


namespace polynomial_coeff_bound_l468_468476

theorem polynomial_coeff_bound {α : Type*} [linear_ordered_field α] 
  (a b c d : α) 
  (h : ∀ x : α, abs x < 1 → abs (a * x^3 + b * x^2 + c * x + d) ≤ 1) : 
  abs a + abs b + abs c + abs d ≤ 7 := 
by sorry

end polynomial_coeff_bound_l468_468476


namespace total_digits_written_total_digit_1_appearances_digit_at_position_2016_l468_468984

-- Problem 1
theorem total_digits_written : 
  let digits_1_to_9 := 9
  let digits_10_to_99 := 90 * 2
  let digits_100_to_999 := 900 * 3
  digits_1_to_9 + digits_10_to_99 + digits_100_to_999 = 2889 := 
by
  sorry

-- Problem 2
theorem total_digit_1_appearances : 
  let digit_1_as_1_digit := 1
  let digit_1_as_2_digits := 10 + 9
  let digit_1_as_3_digits := 100 + 9 * 10 + 9 * 10
  digit_1_as_1_digit + digit_1_as_2_digits + digit_1_as_3_digits = 300 := 
by
  sorry

-- Problem 3
theorem digit_at_position_2016 : 
  let position_1_to_99 := 9 + 90 * 2
  let remaining_positions := 2016 - position_1_to_99
  let three_digit_positions := remaining_positions / 3
  let specific_number := 100 + three_digit_positions - 1
  specific_number % 10 = 8 := 
by
  sorry

end total_digits_written_total_digit_1_appearances_digit_at_position_2016_l468_468984


namespace parabola_standard_equation_and_m_range_l468_468334

theorem parabola_standard_equation_and_m_range 
  (M : ℝ × ℝ) 
  (hM : M = (Real.sqrt 3, -2 * Real.sqrt 3))
  (symm_y : ∀ x y, (x, y) ∈ set_of (λ p : ℝ × ℝ, p.1 ^ 2 = -2 * p.2) ↔ (x, -y) ∈ set_of (λ p : ℝ × ℝ, p.1 ^ 2 = -2 * p.2)) : 
  ∃ p : ℝ, p = Real.sqrt 3 / 4 ∧ 
  ∀ x y, x ^ 2 = - (Real.sqrt 3 / 2) * y ↔ 
  ∀ m : ℝ, m < Real.sqrt 3 / 8 do 
  sorry

end parabola_standard_equation_and_m_range_l468_468334


namespace intersect_diagonals_l468_468432

/-- 
In a regular eighteen-sided polygon, prove that the diagonals \(A_{0}A_{p+3}\), \(A_{p+1}A_{18-r}\), 
and \(A_{1}A_{p+q+3}\) intersect at a single point in the following cases:
- Case a: \( \{p, q, r\} = \{1, 3, 4\} \)
- Case b: \( \{p, q, r\} = \{2, 2, 3\} \)
--/
theorem intersect_diagonals 
  (p q r : ℕ) (h_cases : {p, q, r} = {1, 3, 4} ∨ {p, q, r} = {2, 2, 3}) :
  ∃ (P : Point), 
  collinear {A₀, A_{p+3}, P} ∧
  collinear {A_{p+1}, A_{18-r}, P} ∧
  collinear {A₁, A_{p+q+3}, P} :=
sorry

end intersect_diagonals_l468_468432


namespace optimal_selection_uses_golden_ratio_l468_468650

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l468_468650


namespace optimal_selection_method_use_golden_ratio_l468_468808

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l468_468808


namespace derivative_at_0_l468_468017

noncomputable def f : ℝ → ℝ :=
λ x, if x = 0 then 0 else sqrt(1 + log (1 + 3 * x^2 * cos (2 / x))) - 1

theorem derivative_at_0 : deriv f 0 = 0 :=
by sorry

end derivative_at_0_l468_468017


namespace optimal_selection_method_is_golden_ratio_l468_468589

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l468_468589


namespace min_number_of_girls_l468_468914

theorem min_number_of_girls (total_students boys_count girls_count : ℕ) (no_three_equal: ∀ m n : ℕ, m ≠ n → boys_count m ≠ boys_count n) (boys_at_most : boys_count ≤ 2 * (girls_count + 1)) : ∃ d : ℕ, d ≥ 6 ∧ total_students = 20 ∧ boys_count = total_students - girls_count :=
by
  let d := 6
  exists d
  sorry

end min_number_of_girls_l468_468914


namespace hua_luogeng_optimal_selection_l468_468554

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l468_468554


namespace books_sold_on_monday_l468_468263

def InitialStock : ℕ := 800
def BooksNotSold : ℕ := 600
def BooksSoldTuesday : ℕ := 10
def BooksSoldWednesday : ℕ := 20
def BooksSoldThursday : ℕ := 44
def BooksSoldFriday : ℕ := 66

def TotalBooksSold : ℕ := InitialStock - BooksNotSold
def BooksSoldAfterMonday : ℕ := BooksSoldTuesday + BooksSoldWednesday + BooksSoldThursday + BooksSoldFriday

theorem books_sold_on_monday : 
  TotalBooksSold - BooksSoldAfterMonday = 60 := by
  sorry

end books_sold_on_monday_l468_468263


namespace find_f_values_l468_468467

noncomputable def monotonic_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f(x) < f(y)

theorem find_f_values
  (f : ℕ → ℕ)
  (h_monotonic : monotonic_increasing (f : ℝ → ℝ))
  (h_nat_star : ∀ n : ℕ, 0 < n → 0 < f n)
  (h_eq : ∀ n : ℕ, 0 < n → f (f n) = 2 * n + 1) :
  f 1 = 2 ∧ f 2 = 3 :=
by
  sorry

end find_f_values_l468_468467


namespace optimal_selection_uses_golden_ratio_l468_468663

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l468_468663


namespace optimal_selection_golden_ratio_l468_468704

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l468_468704


namespace area_of_hexagon_l468_468109
noncomputable theory

-- Define the trapezoid and its properties
structure Trapezoid :=
  (A B C D : Point)
  (AB CD : Line)
  (AB_parallel_CD : AB ∥ CD)
  (AB_length : Real := 9)
  (BC_length : Real := 3)
  (CD_length : Real := 13)
  (DA_length : Real := 5)
  (D_right_angle : ∠D = 90)

-- Define the points P and Q
structure PointsPAndQ :=
  (P Q : Point)
  (A_Bisector : Bisector ∠A)
  (B_Bisector : Bisector ∠B)
  (C_Bisector : Bisector ∠C)
  (D_Bisector : Bisector ∠D)
  (P_meet_A_D_bisectors : P ∈ A_Bisector ∧ P ∈ D_Bisector)
  (Q_meet_B_C_bisectors : Q ∈ B_Bisector ∧ Q ∈ C_Bisector)

-- Definition of the problem in Lean
theorem area_of_hexagon (trapezoid : Trapezoid) (pointsP_Q : PointsPAndQ) : 
  area_of_hexagon(ABQCDP) = 18 * √5 :=
sorry

end area_of_hexagon_l468_468109


namespace trajectory_equation_incircle_radius_l468_468013

-- Definition of the problem context
def point (α : Type) := α × α

variables (P Q : point ℝ) (O : point ℝ) (M N : point ℝ)

-- Given conditions
def condition_1 (P Q : point ℝ) : Prop := Q.fst = -2
def condition_2 (P Q : point ℝ) : Prop := P.snd = Q.snd
def condition_3 (P Q : point ℝ) : Prop := P.fst * Q.fst + P.snd * Q.snd = 0

-- Points M and N fixed
def point_M := point ℝ := (-1/2, 0)
def point_N := point ℝ := (1/2, 0)

-- Prove the first part
theorem trajectory_equation (P Q : point ℝ) (h1 : condition_1 P Q) (h2 : condition_2 P Q) (h3 : condition_3 P Q) : P.snd * P.snd = 2 * P.fst :=
sorry

-- Definitions for the incircle radius problem (Rn is a placeholder)
variables (x1 x2 x3 y1 y2 y3 : ℝ)
def radius_formula (x2 : ℝ) (y2 : ℝ) : ℝ := 
  1 / (real.sqrt((1 / (2 * x2)) + (1 / (x2 + 1/2)^2)) + (1 / (x2 + 1/2)))

theorem incircle_radius (x2 y2 : ℝ) : 
  (∃ x1 x3 y1 y3 : ℝ, 0 < x1 ∧ x1 < x2 ∧ radius_formula x2 y2 = 1 / (real.sqrt((1 / (2 * x2)) + (1 / (x2 + 1/2)^2)) + (1 / (x2 + 1/2)))) :=
sorry

end trajectory_equation_incircle_radius_l468_468013


namespace optimal_selection_uses_golden_ratio_l468_468829

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l468_468829


namespace min_girls_in_class_l468_468920

theorem min_girls_in_class : ∃ d : ℕ, 20 - d ≤ 2 * (d + 1) ∧ d ≥ 6 := by
  sorry

end min_girls_in_class_l468_468920


namespace graph_shift_correct_l468_468027

def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)
def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem graph_shift_correct :
  ∀ x : ℝ, f (x - Real.pi / 8) = g x :=
by
  intro x
  calc
    f (x - Real.pi / 8) = Real.sin (2 * (x - Real.pi / 8) + Real.pi / 4) : rfl
    ...               = g x : sorry

end graph_shift_correct_l468_468027


namespace optimal_selection_method_uses_golden_ratio_l468_468758

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l468_468758


namespace hua_luogeng_optimal_selection_l468_468550

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l468_468550


namespace sum_fifth_power_divisible_by_30_l468_468472

theorem sum_fifth_power_divisible_by_30
  (n : ℕ)
  (a : ℕ → ℕ)
  (h : ∑ i in Finset.range n, a i ≡ 0 [MOD 30]) :
  ∑ i in Finset.range n, (a i)^5 ≡ 0 [MOD 30] := 
sorry

end sum_fifth_power_divisible_by_30_l468_468472


namespace interval_monotonically_increasing_range_g_l468_468365

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sqrt 3) * Real.sin (x + (Real.pi / 4)) * Real.cos (x + (Real.pi / 4)) + Real.sin (2 * x) - 1

noncomputable def g (x : ℝ) : ℝ :=
  2 * Real.sin (2 * x + (2 * Real.pi / 3)) - 1

theorem interval_monotonically_increasing :
  ∃ (k : ℤ), ∀ (x : ℝ), (k * Real.pi - (5 * Real.pi / 12) ≤ x ∧ x ≤ k * Real.pi + (Real.pi / 12)) → 0 ≤ deriv f x :=
sorry

theorem range_g (m : ℝ) : 
  ∃ (x : ℝ), (0 ≤ x ∧ x ≤ Real.pi / 2) → g x = m ↔ -3 ≤ m ∧ m ≤ Real.sqrt 3 - 1 :=
sorry

end interval_monotonically_increasing_range_g_l468_468365


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468763

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468763


namespace hua_luogeng_optimal_selection_l468_468555

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l468_468555


namespace value_of_a_minus_c_l468_468401

def avg (x y : ℝ) : ℝ := (x + y) / 2

theorem value_of_a_minus_c (a b c : ℝ) (h1 : avg a b = 115) (h2 : avg b c = 160) :
  a - c = -90 := 
by
  sorry

end value_of_a_minus_c_l468_468401


namespace possible_lightest_boy_heaviest_girl_l468_468415

theorem possible_lightest_boy_heaviest_girl
    (B G : ℕ)
    (hN : B + G < 35)
    (w_boys w_girls : ℕ → ℝ)
    (avg_all : Real := 53.5)
    (avg_girls : Real := 47)
    (avg_boys : Real := 60) :
    (∑ i in Finset.range B, w_boys i) / B = avg_boys →
    (∑ i in Finset.range G, w_girls i) / G = avg_girls →
    ((∑ i in Finset.range B, w_boys i) + (∑ i in Finset.range G, w_girls i)) / (B + G) = avg_all →
    ∃ (i j : ℕ), i < B ∧ j < G ∧ w_boys i < all_weights_worst(G, w_girls) ∧ w_girls j > all_weights_best(B, w_boys) :=
sorry

/-- Helper function that returns the minimum weight of the given range of girls' weights --/
noncomputable def all_weights_worst (g: ℕ, wg: ℕ → ℝ) : ℝ :=
  Finset.min' (Finset.range g) (begin
    -- there should be at least one girl
    exact ⟨0, by simp [show 0 < g, by omega]⟩
  end)

/-- Helper function that returns the maximum weight of the given range of boys' weights --/
noncomputable def all_weights_best (b: ℕ, wb: ℕ → ℝ) : ℝ :=
  Finset.max' (Finset.range b) (begin
    -- there should be at least one boy
    exact ⟨0, by simp [show 0 < b, by omega]⟩
  end)

end possible_lightest_boy_heaviest_girl_l468_468415


namespace min_girls_in_class_l468_468934

theorem min_girls_in_class (n : ℕ) (d : ℕ) (h1 : n = 20) 
  (h2 : ∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
       ∀ f : ℕ → set ℕ, (card (f i) ≠ card (f j) ∨ card (f j) ≠ card (f k) ∨ card (f i) ≠ card (f k))) : 
d ≥ 6 :=
by
  sorry

end min_girls_in_class_l468_468934


namespace min_number_of_girls_l468_468889

theorem min_number_of_girls (d : ℕ) (students : ℕ) (boys : ℕ → ℕ) : 
  students = 20 ∧ ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → boys i ≠ boys j ∨ boys j ≠ boys k ∨ boys i ≠ boys k → d = 6 :=
by
  sorry

end min_number_of_girls_l468_468889


namespace ratio_of_students_to_boys_is_simplified_l468_468998

-- Definitions and conditions
def total_population : ℕ := 140
def percentage_boys : ℚ := 0.50
def num_students : ℕ := 98

-- Calculate the number of boys
def num_boys := (percentage_boys * total_population : ℚ).natAbs

-- Ratio of students to boys simplified
def ratio_students_to_boys := (num_students : ℚ) / (num_boys : ℚ)
def simplified_ratio_students_to_boys := ((7 : ℚ) / (5 : ℚ))

-- The statement to prove
theorem ratio_of_students_to_boys_is_simplified :
  ratio_students_to_boys = simplified_ratio_students_to_boys :=
sorry

end ratio_of_students_to_boys_is_simplified_l468_468998


namespace equilateral_triangle_sum_of_cubes_l468_468465

variables {α : Type*} [InnerProductSpace ℝ α]

/-- Given an equilateral triangle ABC and a point P inside it, the sum of cubed distances times the sine of the respective angles for points within the triangle satisfy the given equality. -/
theorem equilateral_triangle_sum_of_cubes (A B C P : α) 
  (hABC : ∀ x y z: α, ∥x - y∥ = ∥y - z∥ ∧ ∥y - z∥ = ∥z - x∥) 
  (hP : ∃ x y z: ℝ, ∥P - A∥ = x ∧ ∥P - B∥ = y ∧ ∥P - C∥ = z) :
  ∥P - A∥^3 * real.sin (∠BAP) + ∥P - B∥^3 * real.sin (∠CBP) + ∥P - C∥^3 * real.sin (∠ACP) =
  ∥P - A∥^3 * real.sin (∠CAP) + ∥P - B∥^3 * real.sin (∠ABP) + ∥P - C∥^3 * real.sin (∠BCP) :=
by
  sorry

end equilateral_triangle_sum_of_cubes_l468_468465


namespace min_number_of_girls_l468_468911

theorem min_number_of_girls (total_students boys_count girls_count : ℕ) (no_three_equal: ∀ m n : ℕ, m ≠ n → boys_count m ≠ boys_count n) (boys_at_most : boys_count ≤ 2 * (girls_count + 1)) : ∃ d : ℕ, d ≥ 6 ∧ total_students = 20 ∧ boys_count = total_students - girls_count :=
by
  let d := 6
  exists d
  sorry

end min_number_of_girls_l468_468911


namespace optimal_selection_uses_golden_ratio_l468_468820

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l468_468820


namespace hua_luogeng_optimal_selection_method_l468_468741

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l468_468741


namespace combined_remaining_cards_l468_468282

theorem combined_remaining_cards :
  let Brandon_cards := 20
  let Malcom_cards := Brandon_cards + 12
  let Ella_cards := Malcom_cards - 5
  let Malcom_gives := (2 * Malcom_cards) / 3 
  let Malcom_remaining := Malcom_cards - (Malcom_gives).toInt
  let Ella_gives := Ella_cards / 4
  let Ella_remaining := Ella_cards - (Ella_gives).toInt
  (Malcom_remaining + Ella_remaining) = 32 := by
sorry

end combined_remaining_cards_l468_468282


namespace min_number_of_girls_l468_468893

theorem min_number_of_girls (d : ℕ) (students : ℕ) (boys : ℕ → ℕ) : 
  students = 20 ∧ ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → boys i ≠ boys j ∨ boys j ≠ boys k ∨ boys i ≠ boys k → d = 6 :=
by
  sorry

end min_number_of_girls_l468_468893


namespace polynomial_division_l468_468120

noncomputable def polynomial_of_degree_1992 (P : ℂ[X]) := degree P = 1992

def distinct_roots (P : ℂ[X]) := ∀ z : ℂ, is_root P z → multiplicity z P = 1

def exists_a_seq_dividing (P : ℂ[X]) :=
  ∃ a : ℕ → ℂ, P ∣ (λ z, (λ z, (λ z, iterate 1992 (λ (x z), (z - a x)^2) z - a 1991) z - a 1990) z - ... - a 1) 0

theorem polynomial_division (P : ℂ[X]) (h_degree : polynomial_of_degree_1992 P) (h_distinct : distinct_roots P) :
  exists_a_seq_dividing P :=
sorry

end polynomial_division_l468_468120


namespace solve_for_x_l468_468988

variables {A B C m n x : ℝ}

-- Existing conditions
def A_rate_condition : A = (B + C) / m := sorry
def B_rate_condition : B = (C + A) / n := sorry
def C_rate_condition : C = (A + B) / x := sorry

-- The theorem to be proven
theorem solve_for_x (A_rate_condition : A = (B + C) / m)
                    (B_rate_condition : B = (C + A) / n)
                    (C_rate_condition : C = (A + B) / x)
                    : x = (2 + m + n) / (m * n - 1) := by
  sorry

end solve_for_x_l468_468988


namespace ellipse_and_triang_properties_l468_468338

theorem ellipse_and_triang_properties
  (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
  (h_eccentricity : ∃ (c : ℝ), c = a / 2 ∧ a^2 - b^2 = c^2)
  (h_focus_dist : distance (0, 0) (c, 0) = 1)
  (h_PF1_center : ∃ (P Q : ℝ × ℝ), |P - F1| + |Q - F2| = 4 ∧ ∃ λ : ℝ, vector_length (P - F2) = λ * vector_length (F2 - Q)) :
  (ellipse_eq : ∃ (x y: ℝ), x^2 / 4 + y^2 / 3 = 1) ∧
  (triangle_perimeter : ∃ (P Q: ℝ × ℝ), 4 * a = 8) ∧
  (max_incircle_area : ∃ (r : ℝ), r = 3 / 4 ∧ m = 0 → λ = 1) :=
sorry

end ellipse_and_triang_properties_l468_468338


namespace hua_luogeng_optimal_selection_method_l468_468740

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l468_468740


namespace five_people_line_up_count_l468_468078

def num_ways_to_line_up (n : ℕ) (cond1 cond2 : ℕ → Prop) : ℕ :=
  if n = 5 then 4 * 3 * 3 * 2 * 1 else 0

theorem five_people_line_up_count 
  (people : Fin 5)
  (youngest oldest : Fin 5)
  (not_first : ∀ (p : Fin 5), cond1 p → p ≠ youngest)
  (not_last : ∀ (p : Fin 5), cond2 p → p ≠ oldest) :
  num_ways_to_line_up 5 (λ p, p ≠ youngest) (λ p, p ≠ oldest) = 72 :=
sorry

end five_people_line_up_count_l468_468078


namespace max_value_E_zero_l468_468459

noncomputable def E (a b c : ℝ) : ℝ :=
  a * b * c * (a - b * c^2) * (b - c * a^2) * (c - a * b^2)

theorem max_value_E_zero (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≥ b * c^2) (h2 : b ≥ c * a^2) (h3 : c ≥ a * b^2) :
  E a b c ≤ 0 :=
by
  sorry

end max_value_E_zero_l468_468459


namespace sum_of_coefficients_l468_468440

-- Definition of the given binomial expression
def binomial_expansion (x : ℝ) : ℝ := (1 - x) ^ 5

-- Proof statement: sum of the coefficients for x = 1 in the expansion of (1 - x)^5
theorem sum_of_coefficients (x : ℝ) (h : x = 1) : binomial_expansion x = 0 := by
  rw [h]
  simp [binomial_expansion]
  norm_num
  -- sum of coefficients is equal to the value of the binomial expansion when x = 1, which is 0.
  sorry

end sum_of_coefficients_l468_468440


namespace sum_of_roots_f_eq_a_l468_468363

theorem sum_of_roots_f_eq_a
  (f : ℝ → ℝ)
  (a : ℝ)
  (hx1 : ∀ x, 0 ≤ x ∧ x ≤ 2 * π → f x = sqrt 3 * sin x + cos x)
  (hx2 : 0 < a ∧ a < 1):
  ∃ x1 x2, f x1 = a ∧ f x2 = a ∧ x1 + x2 = 8 * π / 3 := sorry

end sum_of_roots_f_eq_a_l468_468363


namespace angle_CBD_of_isosceles_triangle_l468_468229

theorem angle_CBD_of_isosceles_triangle (A B C D : Type) [angle : A → A → A → ℝ]
  (isosceles : angle A B C = angle B C A)
  (m_angle_C : angle A C B = 50) :
  angle B C D = 115 :=
sorry

end angle_CBD_of_isosceles_triangle_l468_468229


namespace count_triangles_and_right_triangles_l468_468048

/-- 
  A rectangle which is not a square has vertices A, B, C, D.
  P is the intersection point of the diagonals AC and BD.
-/
structure Rectangle (α : Type*) :=
  (A B C D P : α)
  (is_rectangle : ∀ (AB BC CD DA AC BD : α), True)
  (is_not_square : A ≠ C ∨ B ≠ D)
  (diagonal_intersection : True -- Assume this represents that P is the intersection of AC and BD
  )

/--The number of triangles formed by the vertices A, B, C, D and the intersection point P 
with all triangles having a common vertex A is 5.
/--The number of these triangles that are right-angled is 3.
-/
theorem count_triangles_and_right_triangles (R : Rectangle α) : 
∃ (triangles : set (α × α × α)), 
  (|triangles| = 5) ∧ (|{t : α × α × α | is_right_triangle t}| = 3) := 
sorry

end count_triangles_and_right_triangles_l468_468048


namespace total_cost_l468_468210

def sandwich_cost : ℝ := 1.49
def soda_cost : ℝ := 0.87
def num_sandwiches : ℝ := 2
def num_sodas : ℝ := 4

theorem total_cost (h1 : sandwich_cost = 1.49) 
                   (h2 : soda_cost = 0.87) 
                   (h3 : num_sandwiches = 2) 
                   (h4 : num_sodas = 4) : 
                   (num_sandwiches * sandwich_cost + num_sodas * soda_cost) = 6.46 := 
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_cost_l468_468210


namespace imaginary_part_of_z_l468_468388

theorem imaginary_part_of_z (z : ℂ) (h : (z / (1 - I)) = (3 + I)) : z.im = -2 :=
sorry

end imaginary_part_of_z_l468_468388


namespace sum_of_numbers_base5_reverse_base9_l468_468310

theorem sum_of_numbers_base5_reverse_base9 : 
  ∑ n in { n : ℕ | ∃ (d : ℕ) (a : ℕ → ℕ),
              (∀ i, i ≤ d → a i < 5) ∧
              (∀ i, i ≤ d → n = n + 5^i * a (d - i)) ∧
              (∀ i, i ≤ d → n = n + 9^i * a i)
          }, n = 31 :=
by
  sorry

end sum_of_numbers_base5_reverse_base9_l468_468310


namespace matilda_percentage_loss_l468_468498

theorem matilda_percentage_loss (initial_cost selling_price : ℕ) (h_initial : initial_cost = 300) (h_selling : selling_price = 255) :
  ((initial_cost - selling_price) * 100) / initial_cost = 15 :=
by
  rw [h_initial, h_selling]
  -- Proceed with the proof
  sorry

end matilda_percentage_loss_l468_468498


namespace solution_for_4_minus_c_l468_468383

-- Define the conditions as Lean hypotheses
theorem solution_for_4_minus_c (c d : ℚ) (h1 : 4 + c = 5 - d) (h2 : 5 + d = 9 + c) : 4 - c = 11 / 2 :=
by
  sorry

end solution_for_4_minus_c_l468_468383


namespace minimum_value_distance_minimum_distance_achievable_l468_468062

noncomputable def minimum_distance (z : ℂ) : ℝ := Complex.abs(z - 3)

theorem minimum_value_distance (z : ℂ) (h : Complex.abs(z - (1 + 2 * Complex.I)) = 2) : 
  minimum_distance z ≥ 2 * Real.sqrt 2 - 2 :=
sorry

theorem minimum_distance_achievable :
  ∃ z : ℂ, Complex.abs(z - (1 + 2 * Complex.I)) = 2 ∧ minimum_distance z = 2 * Real.sqrt 2 - 2 :=
sorry

end minimum_value_distance_minimum_distance_achievable_l468_468062


namespace hua_luogeng_optimal_selection_l468_468545

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l468_468545


namespace hua_luogeng_optimal_selection_method_l468_468724

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l468_468724


namespace alex_correct_percentage_l468_468074

theorem alex_correct_percentage (y : ℝ) (hy_pos : y > 0) : 
  (5 / 7) * 100 = 71.43 := 
by
  sorry

end alex_correct_percentage_l468_468074


namespace optimal_selection_uses_golden_ratio_l468_468637

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l468_468637


namespace optimalSelectionUsesGoldenRatio_l468_468706

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l468_468706


namespace percentage_gain_correct_l468_468247

theorem percentage_gain_correct :
  let cost_per_bowl := 18
  let total_bowls := 250
  let selling_price_per_bowl := 22
  let sold_bowls := 215
  let broken_bowls := total_bowls - sold_bowls
  let total_cost := total_bowls * cost_per_bowl
  let total_revenue := sold_bowls * selling_price_per_bowl
  let profit := total_revenue - total_cost
  let percentage_gain := (profit / total_cost.toFloat) * 100
  percentage_gain = 5.11 := by
sorry

end percentage_gain_correct_l468_468247


namespace possible_lightest_boy_heaviest_girl_l468_468423

theorem possible_lightest_boy_heaviest_girl
    (B G : ℕ) 
    (wb : ℕ → ℕ) 
    (wg : ℕ → ℕ) 
    (N < 35) 
    (avg_weight_students : Real) 
    (avg_weight_girls : Real) 
    (avg_weight_boys : Real) : 
    (avg_weight_students = 53.5) ∧ 
    (avg_weight_girls = 47) ∧ 
    (avg_weight_boys = 60) → 
    (∃ i j : ℕ, (i < B) ∧ (j < G) ∧ (wb i < wg j)) :=
by 
  sorry

end possible_lightest_boy_heaviest_girl_l468_468423


namespace perfect_square_sequence_l468_468185

theorem perfect_square_sequence :
  ∀ n, n ∈ ℕ → ∃ m, a n = m^2 :=
begin
  -- Define the sequence a
  let a : ℕ → ℤ := λ n,
    if n = 0 then 0
    else if n = 1 then 1
    else if n = 2 then 1
    else if n ≥ 3 then a (n-1) + a (n-2) - 2 * a (n-3)
    else 0,
  
  -- Define a helper function to prove that the sequence terms are perfect squares
  sorry
end

end perfect_square_sequence_l468_468185


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468766

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468766


namespace equality_am_bn_l468_468169

theorem equality_am_bn (m n : ℝ) (x : ℝ) (a b : ℝ) (hmn : m ≠ n) (hm : m ≠ 0) (hn : n ≠ 0) :
  ((x + m) ^ 2 - (x + n) ^ 2 = (m - n) ^ 2) → (x = am + bn) → (a = 0 ∧ b = -1) :=
by
  intro h1 h2
  sorry

end equality_am_bn_l468_468169


namespace mark_eggs_supply_l468_468490

theorem mark_eggs_supply (dozen_eggs: ℕ) (days_in_week: ℕ) : dozen_eggs = 12 → 
  days_in_week = 7 → (5 * dozen_eggs + 30) * days_in_week = 630 :=
by 
  intros h_dozen h_days;
  rw [h_dozen, h_days];
  simp;
  norm_num;
  exact rfl

end mark_eggs_supply_l468_468490


namespace lightest_boy_heaviest_girl_l468_468411

theorem lightest_boy_heaviest_girl :
  ∃ (B G : ℕ), B + G < 35 ∧
  (∃ (wb : ℕ → ℝ), (∀ i, wb i > 0) ∧ (∑ i in Finset.range B, wb i) = 60 * B ∧ (∃ i, wb i = min (Finset.range B) wb)) ∧
  (∃ (wg : ℕ → ℝ), (∀ i, wg i > 0) ∧ (∑ i in Finset.range G, wg i) = 47 * G ∧ (∃ i, wg i = max (Finset.range G) wg)) ∧
  (∑ i in Finset.range B, wb i + ∑ i in Finset.range G, wg i) = 53.5 * (B + G) :=
begin
  sorry
end

end lightest_boy_heaviest_girl_l468_468411


namespace annie_initial_money_l468_468275

def cost_of_hamburgers (n : Nat) : Nat := n * 4
def cost_of_milkshakes (m : Nat) : Nat := m * 5
def total_cost (n m : Nat) : Nat := cost_of_hamburgers n + cost_of_milkshakes m
def initial_money (n m left : Nat) : Nat := total_cost n m + left

theorem annie_initial_money : initial_money 8 6 70 = 132 := by
  sorry

end annie_initial_money_l468_468275


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468768

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l468_468768


namespace bc_over_ad_l468_468119

-- Define the rectangular prism
structure RectangularPrism :=
(length width height : ℝ)

-- Define the problem parameters
def B : RectangularPrism := ⟨2, 4, 5⟩

-- Define the volume form of S(r)
def volume (a b c d : ℝ) (r : ℝ) : ℝ := a * r^3 + b * r^2 + c * r + d

-- Prove that the relationship holds
theorem bc_over_ad (a b c d : ℝ) (r : ℝ) (h_a : a = (4 * π) / 3) (h_b : b = 11 * π) (h_c : c = 76) (h_d : d = 40) :
  (b * c) / (a * d) = 15.67 := by
  sorry

end bc_over_ad_l468_468119


namespace solve_equation_l468_468958

theorem solve_equation (x : ℝ) :
  (3 / x - (1 / x * 6 / x) = -2.5) ↔ (x = (-3 + Real.sqrt 69) / 5 ∨ x = (-3 - Real.sqrt 69) / 5) :=
by {
  sorry
}

end solve_equation_l468_468958


namespace incorrect_constant_term_l468_468966

theorem incorrect_constant_term :
  let p := -2 * X^3 + 4 * X - 2 in
  polynomial.coeff p 0 ≠ 2 :=
sorry

end incorrect_constant_term_l468_468966
